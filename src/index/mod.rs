pub mod lsh;
pub mod tiling;

use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use memmap2::{Mmap, MmapMut};
use rayon::prelude::*;

use crate::sketch::{MinimizerConfig, minimizers};
use crate::types::{RefBases, RefSeq, Reference, Strand};

const INDEX_MAGIC: &[u8; 8] = b"KIRAIDX3";
const INDEX_VERSION: u32 = 3;

/// Older magic accepted on read but never written.
const INDEX_MAGIC_LEGACY_V2: &[u8; 8] = b"KIRAIDX2";

/// On-disk Occ record size: ref_id (u32) + pos (u32) + strand (u8) + 3 bytes padding.
const OCC_DISK_SIZE: usize = 12;

/// Occurrence of a minimizer on the reference.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Occ {
    pub ref_id: u32,
    pub pos: u32,
    pub strand: Strand,
}

/// Hash → bucket-id lookup.
enum HashLookup {
    Mph(kira_kv_engine::Index),
    Sorted(Vec<u64>),
}

impl HashLookup {
    fn lookup(&self, hash: u64) -> Option<usize> {
        match self {
            HashLookup::Mph(idx) => match idx.lookup_u64_fast(hash) {
                Some(Ok(id)) => Some(id),
                Some(Err(_)) => None,
                None => idx.lookup_u64(hash).ok(),
            },
            HashLookup::Sorted(arr) => arr.binary_search(&hash).ok(),
        }
    }
    /// No-alloc batch lookup.
    fn lookup_batch_into(
        &self,
        keys: &[u64],
        canon_scratch: &mut [u64],
        out: &mut [Option<usize>],
    ) {
        match self {
            HashLookup::Mph(idx) => {
                idx.lookup_batch_u64_simd_into(keys, canon_scratch, out);
            }
            HashLookup::Sorted(arr) => {
                for (i, k) in keys.iter().enumerate() {
                    out[i] = arr.binary_search(k).ok();
                }
            }
        }
    }
    fn key_count(&self) -> usize {
        match self {
            HashLookup::Mph(idx) => idx.len(),
            HashLookup::Sorted(arr) => arr.len(),
        }
    }
}

/// Hot-cache entry: pre-materialised occurrences for a frequently-queried minimizer.
#[derive(Clone)]
pub struct HotBucketEntry {
    pub occs: Vec<(u32, u32, Strand)>,
}

/// A minimizer index table. See the module-level doc for layout details.
pub struct MinimizerIndex {
    pub k: usize,
    pub w: usize,
    pub max_occ: usize,
    /// `hash → bucket id`. `None` for empty indices (e.g. `build_short=false`).
    hash_lookup: Option<HashLookup>,
    /// Optional hot-bucket cache.
    pub hot_cache: Option<rustc_hash::FxHashMap<u32, HotBucketEntry>>,
    /// `bucket_offsets[id]` = byte offset into `flat_occs` (in records).
    bucket_offsets: OffsetStorage,
    /// Concatenated `Occ` records, grouped by bucket id.
    flat_occs: OccStorage,
}

impl std::fmt::Debug for MinimizerIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MinimizerIndex")
            .field("k", &self.k)
            .field("w", &self.w)
            .field("max_occ", &self.max_occ)
            .field(
                "hash_lookup_keys",
                &self.hash_lookup.as_ref().map(|x| x.key_count()),
            )
            .field(
                "hash_lookup_kind",
                &self.hash_lookup.as_ref().map(|x| match x {
                    HashLookup::Mph(_) => "mph",
                    HashLookup::Sorted(_) => "sorted",
                }),
            )
            .field("n_buckets", &self.bucket_offsets.len().saturating_sub(1))
            .finish()
    }
}

impl Clone for MinimizerIndex {
    fn clone(&self) -> Self {
        let hash_lookup = self.hash_lookup.as_ref().map(|lk| match lk {
            HashLookup::Mph(idx) => {
                let bytes = idx.to_bytes().expect("index serialise");
                HashLookup::Mph(
                    kira_kv_engine::Index::from_bytes(&bytes).expect("index deserialise"),
                )
            }
            HashLookup::Sorted(arr) => HashLookup::Sorted(arr.clone()),
        });
        Self {
            k: self.k,
            w: self.w,
            max_occ: self.max_occ,
            hash_lookup,
            hot_cache: self.hot_cache.clone(),
            bucket_offsets: self.bucket_offsets.clone(),
            flat_occs: self.flat_occs.clone(),
        }
    }
}

/// Storage for `bucket_offsets`: either owned in memory or pointing into the on-disk mmap.
#[derive(Clone, Debug)]
enum OffsetStorage {
    Owned(Vec<u32>),
    Mmap { offset: usize, len: usize },
}

impl OffsetStorage {
    fn get(&self, idx: usize, mmap: Option<&[u8]>) -> u32 {
        match self {
            OffsetStorage::Owned(v) => v[idx],
            OffsetStorage::Mmap { offset, len: _ } => {
                let data = mmap.expect("mmap required for mmap offsets");
                let base = *offset + idx * 4;
                u32::from_le_bytes([data[base], data[base + 1], data[base + 2], data[base + 3]])
            }
        }
    }
    fn len(&self) -> usize {
        match self {
            OffsetStorage::Owned(v) => v.len(),
            OffsetStorage::Mmap { len, .. } => *len,
        }
    }
}

/// Storage for `flat_occs`: either owned `Vec<Occ>` or mmap'd raw bytes.
#[derive(Clone, Debug)]
enum OccStorage {
    Owned(Vec<Occ>),
    Mmap { offset: usize },
}

impl OccStorage {
    fn get(&self, idx: usize, mmap: Option<&[u8]>) -> Occ {
        match self {
            OccStorage::Owned(v) => v[idx],
            OccStorage::Mmap { offset } => {
                let data = mmap.expect("mmap required for mmap occs");
                let base = *offset + idx * OCC_DISK_SIZE;
                let ref_id = u32::from_le_bytes([
                    data[base],
                    data[base + 1],
                    data[base + 2],
                    data[base + 3],
                ]);
                let pos = u32::from_le_bytes([
                    data[base + 4],
                    data[base + 5],
                    data[base + 6],
                    data[base + 7],
                ]);
                let strand = u8_to_strand(data[base + 8]);
                Occ {
                    ref_id,
                    pos,
                    strand,
                }
            }
        }
    }
}

impl MinimizerIndex {
    pub fn build(reference: &Reference, k: usize, w: usize, max_occ: usize) -> Self {
        Self::build_with_budget(reference, k, w, max_occ, ram_half_budget_bytes())
    }

    /// RAM-bounded build.
    pub fn build_with_budget(
        reference: &Reference,
        k: usize,
        w: usize,
        max_occ: usize,
        ram_budget_bytes: usize,
    ) -> Self {
        let cfg = MinimizerConfig { k, w };
        let n_seqs = reference.sequences.len();
        let total_bp: usize = reference.sequences.iter().map(|s| s.len(None)).sum();
        let bytes_per_min = std::mem::size_of::<crate::types::Minimizer>();
        // Tied minima in low-complexity sequence can emit almost one
        // minimizer per base. Average-density estimates (~2/w) are unsafe for
        // deciding whether the all-in-memory sort can fit.
        let estimated_mins = total_bp;
        let estimated_flat_bytes = estimated_mins.saturating_mul(std::mem::size_of::<FlatMin>());
        if estimated_flat_bytes > ram_budget_bytes / 3 {
            return Self::build_external(reference, cfg, k, w, max_occ, ram_budget_bytes);
        }

        // Per-chunk peak buffer fits in 1/8 of the RAM budget.
        let chunk_byte_budget = (ram_budget_bytes / 8).max(64 * 1024 * 1024);
        let bp_per_chunk = (chunk_byte_budget.saturating_mul(w.max(1)) / bytes_per_min).max(1);

        // Pack sequences into chunks where the sum of bp ≤ bp_per_chunk.
        let mut chunks: Vec<(usize, Vec<(usize, &RefSeq)>)> = Vec::new();
        let mut current: Vec<(usize, &RefSeq)> = Vec::new();
        let mut current_bp = 0usize;
        for (rid, seq) in reference.sequences.iter().enumerate() {
            let seq_bp = seq.len(None);
            if !current.is_empty() && current_bp + seq_bp > bp_per_chunk {
                chunks.push((current_bp, std::mem::take(&mut current)));
                current_bp = 0;
            }
            current.push((rid, seq));
            current_bp += seq_bp;
        }
        if !current.is_empty() {
            chunks.push((current_bp, current));
        }
        let n_chunks = chunks.len();

        let t_total = Instant::now();
        eprintln!(
            "[KIRA_INDEX] (k={k} w={w}) building from {n_seqs} seq(s), {:.1} Mbp; \
             RAM budget {:.1} GB → chunk≈{:.1} Mbp ({} chunk(s))",
            total_bp as f64 / 1e6,
            ram_budget_bytes as f64 / (1u64 << 30) as f64,
            bp_per_chunk as f64 / 1e6,
            n_chunks,
        );

        let mut flat: Vec<FlatMin> = Vec::new();
        let mut total_mins: usize = 0;
        let mut t_extract_acc = std::time::Duration::ZERO;

        for (chunk_idx, (chunk_bp, chunk)) in chunks.into_iter().enumerate() {
            let t = Instant::now();
            let chunk_mins: Vec<(u32, Vec<crate::types::Minimizer>)> = chunk
                .par_iter()
                .map(|(rid, seq)| (*rid as u32, minimizers(seq.bases(None), &cfg)))
                .collect();
            t_extract_acc += t.elapsed();
            let chunk_total: usize = chunk_mins.iter().map(|(_, v)| v.len()).sum();
            total_mins += chunk_total;

            flat.reserve(chunk_total);
            for (rid_u32, mins) in chunk_mins {
                for m in mins {
                    flat.push(FlatMin::pack(m.hash, rid_u32, m.pos, m.strand));
                }
            }

            eprintln!(
                "[KIRA_INDEX] (k={k} w={w}) chunk {}/{}: {:.1} Mbp, +{:.1}M mins → \
                 {:.1}M flat total ({:.2} GB, elapsed={:.1}s)",
                chunk_idx + 1,
                n_chunks,
                chunk_bp as f64 / 1e6,
                chunk_total as f64 / 1e6,
                flat.len() as f64 / 1e6,
                (flat.len() * std::mem::size_of::<FlatMin>()) as f64 / (1u64 << 30) as f64,
                t_total.elapsed().as_secs_f64(),
            );
        }

        let t_sort = Instant::now();
        eprintln!(
            "[KIRA_INDEX] (k={k} w={w}) sorting {:.1}M minimizers by hash...",
            flat.len() as f64 / 1e6
        );
        flat.par_sort_unstable_by_key(|t| t.hash);
        let t_sort_dur = t_sort.elapsed();
        eprintln!(
            "[KIRA_INDEX] (k={k} w={w}) sort done in {:.1}s ({:.0}M items/s)",
            t_sort_dur.as_secs_f64(),
            flat.len() as f64 / t_sort_dur.as_secs_f64().max(0.001) / 1e6
        );

        let t_group = Instant::now();
        let mut n_unique = 0usize;
        if !flat.is_empty() {
            n_unique = 1;
            for i in 1..flat.len() {
                if flat[i].hash != flat[i - 1].hash {
                    n_unique += 1;
                }
            }
        }
        eprintln!(
            "[KIRA_INDEX] (k={k} w={w}) phase C: scanning for unique hashes ({:.0}M expected)...",
            n_unique as f64 / 1e6,
        );
        let mut unique_hashes: Vec<u64> = Vec::with_capacity(n_unique);
        let mut bucket_lens: Vec<u32> = Vec::with_capacity(n_unique);
        let mut i = 0;
        while i < flat.len() {
            let h = flat[i].hash;
            unique_hashes.push(h);
            let mut j = i + 1;
            while j < flat.len() && flat[j].hash == h {
                j += 1;
            }
            let retained = (j - i).min(max_occ.saturating_add(1));
            bucket_lens.push(
                u32::try_from(retained).expect("retained minimizer bucket length exceeds u32"),
            );
            i = j;
        }
        let t_group_dur = t_group.elapsed();
        eprintln!(
            "[KIRA_INDEX] (k={k} w={w}) phase C done in {:.1}s: {} unique bucket(s)",
            t_group_dur.as_secs_f64(),
            unique_hashes.len(),
        );
        log_memory_state(&format!("k={k} w={w} after phase C"));

        let use_mph = std::env::var("KIRA_INDEX_USE_MPH")
            .map(|v| v.trim() != "0")
            .unwrap_or(true);
        let t_lookup = Instant::now();
        let (lookup, n_slots, assigned_ids): (HashLookup, usize, Option<Vec<u32>>) = if use_mph {
            eprintln!(
                "[KIRA_INDEX] (k={k} w={w}) phase D: building PtrHash25 over {:.1}M keys \
                 (no progress output; expect ~30s/100M keys on modern CPUs). \
                 Set KIRA_INDEX_USE_MPH=0 to use a sorted-array fallback (instant build, \
                 ~5× slower per lookup).",
                unique_hashes.len() as f64 / 1e6,
            );
            let keys_bytes: Vec<[u8; 8]> = unique_hashes.iter().map(|h| h.to_le_bytes()).collect();
            let mph = kira_kv_engine::IndexBuilder::new()
                .with_parallel_build(true)
                .with_build_fast_profile(true)
                .build_index(keys_bytes)
                .expect("PtrHash25 build");
            eprintln!(
                "[KIRA_INDEX] (k={k} w={w}) phase D done in {:.1}s (mph), looking up ids...",
                t_lookup.elapsed().as_secs_f64(),
            );

            let mut assigned: Vec<u32> = Vec::with_capacity(unique_hashes.len());
            let mut max_id_u: u32 = 0;
            for h in &unique_hashes {
                let id = mph
                    .lookup_u64(*h)
                    .expect("PtrHash25 hash present in build set");
                if id as u32 > max_id_u {
                    max_id_u = id as u32;
                }
                assigned.push(id as u32);
            }
            (HashLookup::Mph(mph), max_id_u as usize + 1, Some(assigned))
        } else {
            eprintln!(
                "[KIRA_INDEX] (k={k} w={w}) phase D: sorted-array lookup over {:.1}M keys \
                 (KIRA_INDEX_USE_MPH=0; binary-search lookups ~100 ns each at this scale)",
                unique_hashes.len() as f64 / 1e6,
            );
            let n = unique_hashes.len();
            (
                HashLookup::Sorted(std::mem::take(&mut unique_hashes)),
                n,
                None,
            )
        };
        let t_lookup_dur = t_lookup.elapsed();

        let t_perm = Instant::now();
        eprintln!(
            "[KIRA_INDEX] (k={k} w={w}) phase E: allocating final layout \
             (offsets={:.2} MB, occs={:.2} GB)...",
            ((n_slots + 1) * 4) as f64 / (1u64 << 20) as f64,
            (bucket_lens.iter().map(|&n| n as usize).sum::<usize>() * OCC_DISK_SIZE) as f64
                / (1u64 << 30) as f64,
        );
        let mut final_offsets: Vec<u32> = vec![0u32; n_slots + 1];

        if let Some(assigned) = assigned_ids.as_ref() {
            for (i, &id) in assigned.iter().enumerate() {
                final_offsets[id as usize + 1] = bucket_lens[i];
            }
        } else {
            // Sorted: slot i lengths come straight from bucket_lens[i].
            for (i, &len) in bucket_lens.iter().enumerate() {
                final_offsets[i + 1] = len;
            }
        }
        // Prefix sum → start offsets.
        for k_idx in 1..=n_slots {
            final_offsets[k_idx] = final_offsets[k_idx]
                .checked_add(final_offsets[k_idx - 1])
                .expect("index occurrence offsets exceed u32; split/tile the reference");
        }

        let stored_occurrences = final_offsets[n_slots] as usize;
        let mut final_flat_occs = vec![
            Occ {
                ref_id: 0,
                pos: 0,
                strand: Strand::Forward,
            };
            stored_occurrences
        ];
        let mut unique_idx = 0usize;
        let mut fi = 0usize;
        while fi < flat.len() {
            let h = flat[fi].hash;
            let mut fj = fi + 1;
            while fj < flat.len() && flat[fj].hash == h {
                fj += 1;
            }
            let id: usize = match assigned_ids.as_ref() {
                Some(a) => a[unique_idx] as usize,
                None => unique_idx,
            };
            let dst_start = final_offsets[id] as usize;
            let retain_end = fi + (fj - fi).min(max_occ.saturating_add(1));
            for (off, src_idx) in (fi..retain_end).enumerate() {
                final_flat_occs[dst_start + off] = flat[src_idx].to_occ();
            }
            unique_idx += 1;
            fi = fj;
        }
        let t_perm_dur = t_perm.elapsed();
        drop(flat);
        drop(bucket_lens);
        drop(unique_hashes); // already taken above if sorted path
        drop(assigned_ids);

        eprintln!(
            "[KIRA_INDEX] (k={k} w={w}) done: {:.0}M minimizers → {:.0}M unique buckets \
             [extract={:.1}s, sort={:.1}s, group={:.1}s, lookup({})={:.1}s, perm={:.1}s, \
              total={:.1}s]",
            total_mins as f64 / 1e6,
            n_unique as f64 / 1e6,
            t_extract_acc.as_secs_f64(),
            t_sort_dur.as_secs_f64(),
            t_group_dur.as_secs_f64(),
            if use_mph { "mph" } else { "sorted" },
            t_lookup_dur.as_secs_f64(),
            t_perm_dur.as_secs_f64(),
            t_total.elapsed().as_secs_f64(),
        );

        Self {
            k,
            w,
            max_occ,
            hash_lookup: Some(lookup),
            hot_cache: None,
            bucket_offsets: OffsetStorage::Owned(final_offsets),
            flat_occs: OccStorage::Owned(final_flat_occs),
        }
    }

    fn build_external(
        reference: &Reference,
        cfg: MinimizerConfig,
        k: usize,
        w: usize,
        max_occ: usize,
        ram_budget_bytes: usize,
    ) -> Self {
        let run_bytes = (ram_budget_bytes / 8).max(16 * 1024 * 1024);
        let segment_bp = (run_bytes.saturating_mul(w.max(1))
            / std::mem::size_of::<crate::types::Minimizer>())
        .max(1_000_000);
        let temp_parent = std::env::var_os("KIRA_INDEX_TMPDIR")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let temp = tempfile::Builder::new()
            .prefix("kira-index-")
            .tempdir_in(&temp_parent)
            .unwrap_or_else(|e| {
                panic!(
                    "create external-index temp directory in {}: {e}",
                    temp_parent.display()
                )
            });
        eprintln!(
            "[KIRA_INDEX] (k={k} w={w}) estimated occurrence stream exceeds RAM budget; \
             spilling sorted {:.1} Mbp segments to {}",
            segment_bp as f64 / 1e6,
            temp.path().display()
        );

        let flank = k.saturating_add(w).max(1);
        let mut run_paths = Vec::new();
        let mut total_mins = 0usize;
        for (ref_id, seq) in reference.sequences.iter().enumerate() {
            let bases = seq.bases(None);
            let mut owned_start = 0usize;
            while owned_start < bases.len() {
                let owned_end = owned_start.saturating_add(segment_bp).min(bases.len());
                let slice_start = owned_start.saturating_sub(flank);
                let slice_end = owned_end.saturating_add(flank).min(bases.len());
                let mins = minimizers(&bases[slice_start..slice_end], &cfg);
                let mut run = Vec::with_capacity(mins.len());
                for m in mins {
                    let global_pos = slice_start.saturating_add(m.pos as usize);
                    if global_pos >= owned_start && global_pos < owned_end {
                        let pos = u32::try_from(global_pos)
                            .expect("reference coordinate exceeds u32 index format");
                        run.push(FlatMin::pack(
                            m.hash,
                            u32::try_from(ref_id).expect("reference id exceeds u32"),
                            pos,
                            m.strand,
                        ));
                    }
                }
                total_mins += run.len();
                run.par_sort_unstable_by_key(|m| (m.hash, m.ref_strand, m.pos));
                let path = temp.path().join(format!("run-{:06}.bin", run_paths.len()));
                let file = File::create(&path).expect("create external-index run");
                let mut writer = BufWriter::new(file);
                for item in &run {
                    write_flat_min(&mut writer, *item).expect("write external-index run");
                }
                writer.flush().expect("flush external-index run");
                run_paths.push(path);
                owned_start = owned_end;
            }
        }

        let mut readers: Vec<BufReader<File>> = run_paths
            .iter()
            .map(|p| BufReader::new(File::open(p).expect("open external-index run")))
            .collect();
        let mut heap =
            std::collections::BinaryHeap::<std::cmp::Reverse<(u64, u32, u32, usize)>>::new();
        for (run_idx, reader) in readers.iter_mut().enumerate() {
            if let Some(item) = read_flat_min(reader).expect("read external-index run") {
                heap.push(std::cmp::Reverse((
                    item.hash,
                    item.ref_strand,
                    item.pos,
                    run_idx,
                )));
            }
        }

        let grouped_path = temp.path().join("grouped.bin");
        let mut grouped_writer =
            BufWriter::new(File::create(&grouped_path).expect("create grouped index stream"));
        let mut unique_hashes = Vec::<u64>::new();
        let mut bucket_lens = Vec::<u32>::new();
        let retain_limit = max_occ.saturating_add(1);
        while let Some(std::cmp::Reverse((hash, ref_strand, pos, run_idx))) = heap.pop() {
            let mut retained = Vec::with_capacity(retain_limit.min(1024));
            retained.push(FlatMin {
                hash,
                ref_strand,
                pos,
            });
            refill_run_heap(&mut readers, &mut heap, run_idx);
            while heap.peek().is_some_and(|next| next.0.0 == hash) {
                let std::cmp::Reverse((_, ref_strand, pos, run_idx)) = heap.pop().unwrap();
                if retained.len() < retain_limit {
                    retained.push(FlatMin {
                        hash,
                        ref_strand,
                        pos,
                    });
                }
                refill_run_heap(&mut readers, &mut heap, run_idx);
            }
            unique_hashes.push(hash);
            let count = u32::try_from(retained.len()).expect("retained bucket exceeds u32");
            bucket_lens.push(count);
            grouped_writer
                .write_all(&hash.to_le_bytes())
                .expect("write grouped hash");
            grouped_writer
                .write_all(&count.to_le_bytes())
                .expect("write grouped count");
            for item in retained {
                grouped_writer
                    .write_all(&item.ref_strand.to_le_bytes())
                    .and_then(|_| grouped_writer.write_all(&item.pos.to_le_bytes()))
                    .expect("write grouped occurrence");
            }
        }
        grouped_writer.flush().expect("flush grouped index stream");
        if unique_hashes.is_empty() {
            return empty_minimizer_index(k, w, max_occ);
        }

        let use_mph = std::env::var("KIRA_INDEX_USE_MPH")
            .map(|v| v.trim() != "0")
            .unwrap_or(true);
        let (lookup, n_slots, assigned_ids): (HashLookup, usize, Option<Vec<u32>>) = if use_mph {
            let keys_bytes: Vec<[u8; 8]> = unique_hashes.iter().map(|h| h.to_le_bytes()).collect();
            let mph = kira_kv_engine::IndexBuilder::new()
                .with_parallel_build(true)
                .with_build_fast_profile(true)
                .build_index(keys_bytes)
                .expect("PtrHash25 build");
            let assigned: Vec<u32> = unique_hashes
                .iter()
                .map(|h| {
                    u32::try_from(
                        mph.lookup_u64(*h)
                            .expect("PtrHash25 hash present in build set"),
                    )
                    .expect("MPH slot exceeds u32")
                })
                .collect();
            let n_slots = assigned.iter().copied().max().unwrap_or(0) as usize + 1;
            (HashLookup::Mph(mph), n_slots, Some(assigned))
        } else {
            let n = unique_hashes.len();
            (HashLookup::Sorted(unique_hashes), n, None)
        };

        let mut offsets = vec![0u32; n_slots + 1];
        if let Some(assigned) = assigned_ids.as_ref() {
            for (i, &id) in assigned.iter().enumerate() {
                offsets[id as usize + 1] = bucket_lens[i];
            }
        } else {
            offsets[1..].copy_from_slice(&bucket_lens);
        }
        for i in 1..offsets.len() {
            offsets[i] = offsets[i]
                .checked_add(offsets[i - 1])
                .expect("index occurrence offsets exceed u32; split/tile the reference");
        }

        let mut occurrences = vec![
            Occ {
                ref_id: 0,
                pos: 0,
                strand: Strand::Forward,
            };
            offsets[n_slots] as usize
        ];
        let mut reader =
            BufReader::new(File::open(&grouped_path).expect("open grouped index stream"));
        for unique_idx in 0..bucket_lens.len() {
            let hash = read_u64_io(&mut reader).expect("read grouped hash");
            let count = read_u32_io(&mut reader).expect("read grouped count") as usize;
            let slot = assigned_ids
                .as_ref()
                .map_or(unique_idx, |ids| ids[unique_idx] as usize);
            let dst = offsets[slot] as usize;
            for off in 0..count {
                let ref_strand = read_u32_io(&mut reader).expect("read grouped ref/strand");
                let pos = read_u32_io(&mut reader).expect("read grouped position");
                occurrences[dst + off] = FlatMin {
                    hash,
                    ref_strand,
                    pos,
                }
                .to_occ();
            }
        }
        eprintln!(
            "[KIRA_INDEX] (k={k} w={w}) external build complete: {:.1}M minimizers, \
             {:.1}M buckets, {:.1}M retained occurrences",
            total_mins as f64 / 1e6,
            bucket_lens.len() as f64 / 1e6,
            occurrences.len() as f64 / 1e6,
        );
        Self {
            k,
            w,
            max_occ,
            hash_lookup: Some(lookup),
            hot_cache: None,
            bucket_offsets: OffsetStorage::Owned(offsets),
            flat_occs: OccStorage::Owned(occurrences),
        }
    }

    /// Build the hot-bucket cache.
    pub fn build_hot_cache(&mut self, mmap: Option<&[u8]>, top_n: usize, max_total_occs: usize) {
        use rustc_hash::FxHashMap;
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        if self.hash_lookup.is_none() || top_n == 0 {
            self.hot_cache = None;
            return;
        }

        let n_offsets = self.bucket_offsets.len();
        if n_offsets < 2 {
            self.hot_cache = None;
            return;
        }
        let n_slots = n_offsets - 1;

        let t = std::time::Instant::now();
        eprintln!(
            "[KIRA_HOT_CACHE] scanning {} slots for top {} largest buckets (cap {} M occs)...",
            n_slots,
            top_n,
            max_total_occs / 1_000_000,
        );

        let mut top: BinaryHeap<Reverse<(usize, u32)>> = BinaryHeap::with_capacity(top_n + 1);
        for slot in 0..n_slots {
            let s = self.bucket_offsets.get(slot, mmap) as usize;
            let e = self.bucket_offsets.get(slot + 1, mmap) as usize;
            let len = e - s;
            if len == 0 || len > self.max_occ {
                continue;
            }
            if top.len() < top_n {
                top.push(Reverse((len, slot as u32)));
            } else if let Some(&Reverse((min_len, _))) = top.peek() {
                if len > min_len {
                    top.pop();
                    top.push(Reverse((len, slot as u32)));
                }
            }
        }

        let mut sel: Vec<(usize, u32)> = top.into_iter().map(|Reverse(t)| t).collect();
        sel.sort_unstable_by_key(|&(len, _)| Reverse(len));

        let mut cache: FxHashMap<u32, HotBucketEntry> = FxHashMap::default();
        let mut total_occs = 0usize;
        for (len, slot) in sel {
            if total_occs + len > max_total_occs {
                break;
            }
            let s = self.bucket_offsets.get(slot as usize, mmap) as usize;
            let e = self.bucket_offsets.get(slot as usize + 1, mmap) as usize;
            let mut occs = Vec::with_capacity(e - s);
            for i in s..e {
                let o = self.flat_occs.get(i, mmap);
                occs.push((o.ref_id, o.pos, o.strand));
            }
            total_occs += occs.len();
            cache.insert(slot, HotBucketEntry { occs });
        }

        eprintln!(
            "[KIRA_HOT_CACHE] cached {} buckets, {} occs ({:.1} MB) in {:.2}s",
            cache.len(),
            total_occs,
            total_occs as f64 * 12.0 / 1e6,
            t.elapsed().as_secs_f64(),
        );
        self.hot_cache = Some(cache);
    }

    /// Look up a slot ID in the hot cache.
    #[inline]
    pub fn hot_lookup(&self, slot: u32) -> Option<&[(u32, u32, Strand)]> {
        let cache = self.hot_cache.as_ref()?;
        cache.get(&slot).map(|e| e.occs.as_slice())
    }

    /// One-shot bucket descriptor.
    pub fn bucket(&self, mmap: Option<&[u8]>, hash: u64) -> Option<(usize, usize)> {
        let lookup = self.hash_lookup.as_ref()?;
        let id = lookup.lookup(hash)?;
        if id + 1 >= self.bucket_offsets.len() {
            return None;
        }
        let start = self.bucket_offsets.get(id, mmap) as usize;
        let end = self.bucket_offsets.get(id + 1, mmap) as usize;
        Some((start, end))
    }

    /// SIMD-batch bucket lookup with no per-call allocations.
    pub fn bucket_batch_into(
        &self,
        mmap: Option<&[u8]>,
        hashes: &[u64],
        canon_scratch: &mut [u64],
        ids_scratch: &mut [Option<usize>],
        out: &mut [Option<(usize, usize)>],
    ) {
        let n = hashes.len();
        let Some(lookup) = self.hash_lookup.as_ref() else {
            for slot in &mut out[..n] {
                *slot = None;
            }
            return;
        };
        lookup.lookup_batch_into(hashes, canon_scratch, ids_scratch);
        let n_offsets = self.bucket_offsets.len();
        for (i, id_opt) in ids_scratch[..n].iter().enumerate() {
            out[i] = match *id_opt {
                Some(id) if id + 1 < n_offsets => {
                    let start = self.bucket_offsets.get(id, mmap) as usize;
                    let end = self.bucket_offsets.get(id + 1, mmap) as usize;
                    Some((start, end))
                }
                _ => None,
            };
        }
    }

    /// Look up the number of `Occ` records associated with `hash`.
    pub fn bucket_len(&self, mmap: Option<&[u8]>, hash: u64) -> Option<usize> {
        self.bucket(mmap, hash).map(|(s, e)| e - s)
    }

    pub fn for_each_occ<F: FnMut(Occ)>(&self, mmap: Option<&[u8]>, hash: u64, f: &mut F) {
        let Some((start, end)) = self.bucket(mmap, hash) else {
            return;
        };
        for idx in start..end {
            f(self.flat_occs.get(idx, mmap));
        }
    }
}

/// Packed minimizer tuple used only in the index-build pipeline.
#[derive(Clone, Copy)]
#[repr(C)]
struct FlatMin {
    hash: u64,
    ref_strand: u32,
    pos: u32,
}

const STRAND_MASK: u32 = 1 << 31;
const REF_ID_MASK: u32 = !STRAND_MASK;

impl FlatMin {
    #[inline]
    fn pack(hash: u64, ref_id: u32, pos: u32, strand: Strand) -> Self {
        assert!(
            ref_id & STRAND_MASK == 0,
            "reference id must fit in 31 bits"
        );
        let ref_strand = ref_id
            | match strand {
                Strand::Forward => 0,
                Strand::Reverse => STRAND_MASK,
            };
        Self {
            hash,
            ref_strand,
            pos,
        }
    }

    #[inline]
    fn to_occ(self) -> Occ {
        Occ {
            ref_id: self.ref_strand & REF_ID_MASK,
            pos: self.pos,
            strand: if self.ref_strand & STRAND_MASK != 0 {
                Strand::Reverse
            } else {
                Strand::Forward
            },
        }
    }
}

fn write_flat_min(writer: &mut impl Write, item: FlatMin) -> std::io::Result<()> {
    writer.write_all(&item.hash.to_le_bytes())?;
    writer.write_all(&item.ref_strand.to_le_bytes())?;
    writer.write_all(&item.pos.to_le_bytes())
}

fn read_flat_min(reader: &mut impl Read) -> std::io::Result<Option<FlatMin>> {
    let mut hash = [0u8; 8];
    match reader.read_exact(&mut hash) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let mut ref_strand = [0u8; 4];
    let mut pos = [0u8; 4];
    reader.read_exact(&mut ref_strand)?;
    reader.read_exact(&mut pos)?;
    Ok(Some(FlatMin {
        hash: u64::from_le_bytes(hash),
        ref_strand: u32::from_le_bytes(ref_strand),
        pos: u32::from_le_bytes(pos),
    }))
}

fn refill_run_heap(
    readers: &mut [BufReader<File>],
    heap: &mut std::collections::BinaryHeap<std::cmp::Reverse<(u64, u32, u32, usize)>>,
    run_idx: usize,
) {
    if let Some(item) = read_flat_min(&mut readers[run_idx]).expect("read external-index run") {
        heap.push(std::cmp::Reverse((
            item.hash,
            item.ref_strand,
            item.pos,
            run_idx,
        )));
    }
}

fn read_u32_io(reader: &mut impl Read) -> std::io::Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64_io(reader: &mut impl Read) -> std::io::Result<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

/// Multi-resolution index for short and long reads.
#[derive(Clone, Debug)]
pub struct Index {
    pub reference: Reference,
    pub short: MinimizerIndex,
    pub long: MinimizerIndex,
    pub mmap: Option<Arc<Mmap>>,
}

/// Configuration for building a multi-resolution minimizer index.
#[derive(Clone, Copy, Debug)]
pub struct IndexConfig {
    pub short_k: usize,
    pub short_w: usize,
    pub long_k: usize,
    pub long_w: usize,
    pub max_occ: usize,
    /// Which sub-indices to actually build.
    pub build_short: bool,
    pub build_long: bool,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            short_k: 19,
            short_w: 10,
            long_k: 15,
            long_w: 10,
            max_occ: 500,
            build_short: true,
            build_long: true,
        }
    }
}

impl Index {
    fn mmap_bytes(&self) -> Option<&[u8]> {
        self.mmap.as_deref().map(|m| &m[..])
    }

    pub fn ref_bases(&self, ref_id: usize) -> &[u8] {
        self.reference.sequences[ref_id].bases(self.mmap_bytes())
    }

    pub fn bucket_len(&self, table: &MinimizerIndex, hash: u64) -> Option<usize> {
        table.bucket_len(self.mmap_bytes(), hash)
    }

    /// One-shot bucket descriptor — resolves the MPH lookup once and returns.
    pub fn bucket(&self, table: &MinimizerIndex, hash: u64) -> Option<(usize, usize)> {
        table.bucket(self.mmap_bytes(), hash)
    }

    /// SIMD-batch bucket lookup — forwards to [`MinimizerIndex::bucket_batch_into`].
    pub fn bucket_batch_into(
        &self,
        table: &MinimizerIndex,
        hashes: &[u64],
        canon_scratch: &mut [u64],
        ids_scratch: &mut [Option<usize>],
        out: &mut [Option<(usize, usize)>],
    ) {
        table.bucket_batch_into(self.mmap_bytes(), hashes, canon_scratch, ids_scratch, out)
    }

    /// Fetch a single `Occ` record from a bucket slice.
    pub fn occ_at(&self, table: &MinimizerIndex, idx: usize) -> Occ {
        table.flat_occs.get(idx, self.mmap_bytes())
    }

    pub fn for_each_occ<F: FnMut(Occ)>(&self, table: &MinimizerIndex, hash: u64, mut f: F) {
        table.for_each_occ(self.mmap_bytes(), hash, &mut f)
    }
    pub fn build(reference: Reference, cfg: IndexConfig) -> Self {
        let t0 = Instant::now();
        let short = if cfg.build_short {
            eprintln!(
                "[KIRA_INDEX] building short index (k={} w={})",
                cfg.short_k, cfg.short_w
            );
            let s = MinimizerIndex::build(&reference, cfg.short_k, cfg.short_w, cfg.max_occ);
            log_memory_state("after short build");
            s
        } else {
            eprintln!("[KIRA_INDEX] short index skipped (use --only=long)");
            empty_minimizer_index(cfg.short_k, cfg.short_w, cfg.max_occ)
        };

        let long = if cfg.build_long {
            eprintln!(
                "[KIRA_INDEX] building long index (k={} w={})",
                cfg.long_k, cfg.long_w
            );
            let l = MinimizerIndex::build(&reference, cfg.long_k, cfg.long_w, cfg.max_occ);
            log_memory_state("after long build");
            l
        } else {
            eprintln!("[KIRA_INDEX] long index skipped (use --only=short)");
            empty_minimizer_index(cfg.long_k, cfg.long_w, cfg.max_occ)
        };

        eprintln!(
            "[KIRA_INDEX] index assembly done in {:.2}s",
            t0.elapsed().as_secs_f64()
        );
        Self {
            reference,
            short,
            long,
            mmap: None,
        }
    }

    /// Build a CGK side-index from this `Index`'s reference contigs.
    pub fn build_cgk_index(
        &self,
        seed: u64,
        window_len: usize,
        stride: usize,
        n_banks: usize,
    ) -> crate::alignment::cgk::CgkIndex {
        let scheme = crate::alignment::cgk::FingerprintScheme::new(seed, window_len, n_banks);
        let seqs: Vec<(u32, &[u8])> = self
            .reference
            .sequences
            .iter()
            .enumerate()
            .map(|(i, s)| (i as u32, s.bases(self.mmap_bytes())))
            .collect();
        crate::alignment::cgk::CgkIndex::build_from_sequences(scheme, stride, seqs)
    }

    /// Build a SimHash-LSH side-index from this `Index`'s reference contigs.
    pub fn build_lsh_index(
        &self,
        window_len: usize,
        stride: usize,
        top_bits: u32,
    ) -> crate::index::lsh::LshIndex {
        let seqs: Vec<(u32, &[u8])> = self
            .reference
            .sequences
            .iter()
            .enumerate()
            .map(|(i, s)| (i as u32, s.bases(self.mmap_bytes())))
            .collect();
        crate::index::lsh::LshIndex::build_parallel(seqs, window_len, stride, top_bits)
    }

    /// Convenience: build the LSH index AND owned base copies, and install
    /// it as the process-global rescue. Returns the total entry count for
    /// telemetry.
    pub fn install_lsh_rescue(
        &self,
        cfg: crate::alignment::AlignmentConfig,
        window_len: usize,
        stride: usize,
        top_bits: u32,
        max_candidates: usize,
        max_lsh_hamming: u32,
        max_window_mismatches: u32,
    ) -> Result<usize> {
        let index = self.build_lsh_index(window_len, stride, top_bits);
        let entries = index.entry_count();
        let ref_bases: Vec<Vec<u8>> = self
            .reference
            .sequences
            .iter()
            .map(|s| s.bases(self.mmap_bytes()).to_vec())
            .collect();
        let rescue = crate::alignment::lsh_rescue::LshRescue {
            index,
            ref_bases,
            cfg,
            max_candidates,
            max_lsh_hamming,
            max_window_mismatches,
        };
        crate::alignment::lsh_rescue::set_global_rescue(rescue)
            .map_err(|e| anyhow::anyhow!("install LSH rescue: {}", e))?;
        Ok(entries)
    }

    /// Convenience: build the CGK index AND owned base copies.
    pub fn install_cgk_rescue(
        &self,
        cfg: crate::alignment::AlignmentConfig,
        seed: u64,
        window_len: usize,
        stride: usize,
        n_banks: usize,
        max_candidates: usize,
        min_bank_hits: u32,
    ) -> Result<usize> {
        let index = self.build_cgk_index(seed, window_len, stride, n_banks);
        let entries = index.entry_count();
        let ref_bases: Vec<Vec<u8>> = self
            .reference
            .sequences
            .iter()
            .map(|s| s.bases(self.mmap_bytes()).to_vec())
            .collect();
        let rescue = crate::alignment::cgk::CgkRescue {
            index,
            ref_bases,
            cfg,
            max_candidates,
            min_bank_hits,
        };
        crate::alignment::cgk::set_global_rescue(rescue)
            .map_err(|e| anyhow::anyhow!("install CGK rescue: {}", e))?;
        Ok(entries)
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let total_size = compute_index_size(self);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path.as_ref())
            .context("create index file")?;
        file.set_len(total_size as u64)
            .context("resize index file")?;

        let mut mmap = unsafe { MmapMut::map_mut(&file).context("mmap index for write")? };
        let mut cursor = Cursor::new(&mut mmap[..]);

        cursor.write_all(INDEX_MAGIC)?;
        write_u32(&mut cursor, INDEX_VERSION)?;

        write_u32(&mut cursor, self.reference.sequences.len() as u32)?;
        for seq in &self.reference.sequences {
            write_bytes(&mut cursor, seq.name.as_bytes())?;
            let bases = seq.bases(self.mmap_bytes());
            write_bytes(&mut cursor, bases)?;
        }

        write_minimizer_index(&mut cursor, &self.short, self.mmap_bytes())?;
        write_minimizer_index(&mut cursor, &self.long, self.mmap_bytes())?;
        mmap.flush().context("flush index mmap")?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref()).context("open index file")?;
        let mmap = Arc::new(unsafe { Mmap::map(&file).context("mmap index for read")? });
        let mut cursor = Cursor::new(&mmap[..]);
        let file_len = mmap.len() as u64;

        let mut magic = [0u8; 8];
        cursor.read_exact(&mut magic).context("read index magic")?;
        let is_v2 = &magic == INDEX_MAGIC_LEGACY_V2;
        let is_v3 = &magic == INDEX_MAGIC;
        if !is_v2 && !is_v3 {
            anyhow::bail!(
                "invalid index magic {:?} (expected KIRAIDX2 or KIRAIDX3); \
                 rebuild the index with the current binary",
                std::str::from_utf8(&magic).unwrap_or("<non-utf8>"),
            );
        }
        let version = read_u32(&mut cursor)?;
        let expected_version = if is_v2 { 2 } else { INDEX_VERSION };
        if version != expected_version {
            anyhow::bail!(
                "unsupported index version {} for magic {:?}",
                version,
                std::str::from_utf8(&magic).unwrap_or("<non-utf8>"),
            );
        }
        if is_v2 {
            eprintln!(
                "[KIRA_INDEX] loading legacy KIRAIDX2 index (sorted-array variant unavailable; \
                 layout is otherwise compatible)"
            );
        }

        let seq_count = read_u32(&mut cursor)? as usize;
        if seq_count > mmap.len() / 8 {
            anyhow::bail!("invalid index: sequence count {seq_count} exceeds file bounds");
        }
        let mut sequences = Vec::with_capacity(seq_count);
        for _ in 0..seq_count {
            let name =
                String::from_utf8(read_bytes_bounded(&mut cursor, file_len, "sequence name")?)
                    .context("decode seq name")?;
            let bases_len = read_u32(&mut cursor)? as usize;
            let bases_offset = cursor.position() as usize;
            checked_advance(&mut cursor, bases_len, file_len, "reference bases")?;
            sequences.push(RefSeq {
                name,
                bases: RefBases::Mmap {
                    offset: bases_offset,
                    len: bases_len,
                },
            });
        }

        let reference = Reference { sequences };
        let short = read_minimizer_index_mmap(&mut cursor, file_len)?;
        let long = read_minimizer_index_mmap(&mut cursor, file_len)?;
        if cursor.position() != file_len {
            anyhow::bail!(
                "invalid index: {} trailing byte(s)",
                file_len.saturating_sub(cursor.position())
            );
        }
        Ok(Self {
            reference,
            short,
            long,
            mmap: Some(mmap),
        })
    }
}

fn compute_index_size(index: &Index) -> usize {
    let mut size = 0usize;
    size += INDEX_MAGIC.len();
    size += 4; // version
    size += 4; // seq count
    for seq in &index.reference.sequences {
        size += 4 + seq.name.len();
        size += 4 + seq.len(index.mmap.as_deref().map(|m| &m[..]));
    }
    size += minimizer_index_size(&index.short);
    size += minimizer_index_size(&index.long);
    size
}

/// Build an empty MinimizerIndex used when the caller asks us to skip building a sub-index.
fn empty_minimizer_index(k: usize, w: usize, max_occ: usize) -> MinimizerIndex {
    MinimizerIndex {
        k,
        w,
        max_occ,
        hash_lookup: None,
        hot_cache: None,
        bucket_offsets: OffsetStorage::Owned(vec![0u32]),
        flat_occs: OccStorage::Owned(Vec::new()),
    }
}

/// On-disk layout per sub-index (KIRAIDX3 format):
const LOOKUP_KIND_NONE: u32 = 0;
const LOOKUP_KIND_MPH: u32 = 1;
const LOOKUP_KIND_SORTED: u32 = 2;

fn minimizer_index_size(idx: &MinimizerIndex) -> usize {
    let mut size = 0usize;
    size += 4 * 3; // k, w, max_occ
    size += 4; // lookup_kind
    if let Some(lookup) = idx.hash_lookup.as_ref() {
        match lookup {
            HashLookup::Mph(mph) => {
                size += 8; // blob len
                let blob = mph.to_bytes().expect("size: serialize MPH");
                size += blob.len();
            }
            HashLookup::Sorted(arr) => {
                size += 8; // n_keys
                size += arr.len() * 8;
            }
        }
    }
    size += 8; // n_offsets
    size += idx.bucket_offsets.len() * 4;
    let n_occs = idx
        .bucket_offsets
        .get(idx.bucket_offsets.len().saturating_sub(1), None) as usize;
    size += 8; // n_occs
    size += n_occs * OCC_DISK_SIZE;
    size
}

fn write_minimizer_index<W: Write>(
    writer: &mut W,
    idx: &MinimizerIndex,
    mmap: Option<&[u8]>,
) -> Result<()> {
    write_u32(writer, idx.k as u32)?;
    write_u32(writer, idx.w as u32)?;
    write_u32(writer, idx.max_occ as u32)?;

    match idx.hash_lookup.as_ref() {
        Some(HashLookup::Mph(mph)) => {
            write_u32(writer, LOOKUP_KIND_MPH)?;
            let blob = mph
                .to_bytes()
                .map_err(|e| anyhow::anyhow!("MPH serialize: {e:?}"))?;
            write_u64(writer, blob.len() as u64)?;
            writer.write_all(&blob).context("write MPH blob")?;
        }
        Some(HashLookup::Sorted(arr)) => {
            write_u32(writer, LOOKUP_KIND_SORTED)?;
            write_u64(writer, arr.len() as u64)?;
            for h in arr {
                writer
                    .write_all(&h.to_le_bytes())
                    .context("write sorted hash")?;
            }
        }
        None => {
            write_u32(writer, LOOKUP_KIND_NONE)?;
        }
    }

    let n_offsets = idx.bucket_offsets.len();
    write_u64(writer, n_offsets as u64)?;
    for i in 0..n_offsets {
        let v = idx.bucket_offsets.get(i, mmap);
        write_u32(writer, v)?;
    }
    let n_occs = if n_offsets == 0 {
        0
    } else {
        idx.bucket_offsets.get(n_offsets - 1, mmap) as usize
    };
    write_u64(writer, n_occs as u64)?;
    for i in 0..n_occs {
        let occ = idx.flat_occs.get(i, mmap);
        write_u32(writer, occ.ref_id)?;
        write_u32(writer, occ.pos)?;
        write_u8(writer, strand_to_u8(occ.strand))?;
        writer.write_all(&[0u8, 0u8, 0u8])?;
    }
    Ok(())
}

fn read_minimizer_index_mmap<R: Read + Seek>(
    reader: &mut R,
    file_len: u64,
) -> Result<MinimizerIndex> {
    let k = read_u32(reader)? as usize;
    let w = read_u32(reader)? as usize;
    let max_occ = read_u32(reader)? as usize;
    if k == 0 || k >= 32 || w == 0 {
        anyhow::bail!("invalid minimizer parameters in index: k={k}, w={w}");
    }

    let lookup_kind = read_u32(reader)?;
    let hash_lookup = match lookup_kind {
        LOOKUP_KIND_NONE => None,
        LOOKUP_KIND_MPH => {
            let blob_len = read_u64(reader)? as usize;
            ensure_remaining(reader, blob_len, file_len, "MPH blob")?;
            let mut blob = vec![0u8; blob_len];
            reader.read_exact(&mut blob).context("read MPH blob")?;
            Some(HashLookup::Mph(
                kira_kv_engine::Index::from_bytes(&blob)
                    .map_err(|e| anyhow::anyhow!("MPH deserialize: {e:?}"))?,
            ))
        }
        LOOKUP_KIND_SORTED => {
            let n_keys = read_u64(reader)? as usize;
            let key_bytes = n_keys
                .checked_mul(8)
                .ok_or_else(|| anyhow::anyhow!("sorted hash table size overflow"))?;
            ensure_remaining(reader, key_bytes, file_len, "sorted hash table")?;
            let mut arr = Vec::with_capacity(n_keys);
            let mut buf = [0u8; 8];
            for _ in 0..n_keys {
                reader.read_exact(&mut buf).context("read sorted hash")?;
                arr.push(u64::from_le_bytes(buf));
            }
            Some(HashLookup::Sorted(arr))
        }
        other => anyhow::bail!("unknown hash-lookup kind {}", other),
    };

    let n_offsets = read_u64(reader)? as usize;
    if n_offsets == 0 {
        anyhow::bail!("invalid index: minimizer offset table is empty");
    }
    let offsets_bytes = n_offsets
        .checked_mul(4)
        .ok_or_else(|| anyhow::anyhow!("minimizer offset table size overflow"))?;
    ensure_remaining(reader, offsets_bytes, file_len, "minimizer offsets")?;
    let off_offset = reader.stream_position()? as usize;
    let mut previous = 0u32;
    for i in 0..n_offsets {
        let offset = read_u32(reader)?;
        if i == 0 && offset != 0 {
            anyhow::bail!("invalid index: first minimizer offset is {offset}, expected 0");
        }
        if offset < previous {
            anyhow::bail!("invalid index: minimizer offsets are not monotonic");
        }
        previous = offset;
    }
    let bucket_offsets = OffsetStorage::Mmap {
        offset: off_offset,
        len: n_offsets,
    };

    let n_occs = read_u64(reader)? as usize;
    if previous as usize != n_occs {
        anyhow::bail!(
            "invalid index: final bucket offset {} does not equal occurrence count {n_occs}",
            previous
        );
    }
    let occ_bytes = n_occs
        .checked_mul(OCC_DISK_SIZE)
        .ok_or_else(|| anyhow::anyhow!("occurrence table size overflow"))?;
    ensure_remaining(reader, occ_bytes, file_len, "occurrence table")?;
    let occ_offset = reader.stream_position()? as usize;
    checked_advance(reader, occ_bytes, file_len, "occurrence table")?;
    let flat_occs = OccStorage::Mmap { offset: occ_offset };

    Ok(MinimizerIndex {
        k,
        w,
        max_occ,
        hash_lookup,
        hot_cache: None,
        bucket_offsets,
        flat_occs,
    })
}

fn ensure_remaining<R: Seek>(
    reader: &mut R,
    bytes: usize,
    file_len: u64,
    label: &str,
) -> Result<()> {
    let start = reader.stream_position()?;
    let end = start
        .checked_add(bytes as u64)
        .ok_or_else(|| anyhow::anyhow!("invalid index: {label} range overflow"))?;
    if end > file_len {
        anyhow::bail!("invalid index: {label} ends at byte {end}, file has {file_len} bytes");
    }
    Ok(())
}

fn checked_advance<R: Seek>(
    reader: &mut R,
    bytes: usize,
    file_len: u64,
    label: &str,
) -> Result<()> {
    ensure_remaining(reader, bytes, file_len, label)?;
    let end = reader
        .stream_position()?
        .checked_add(bytes as u64)
        .ok_or_else(|| anyhow::anyhow!("invalid index: {label} range overflow"))?;
    reader.seek(SeekFrom::Start(end))?;
    Ok(())
}

fn read_bytes_bounded<R: Read + Seek>(
    reader: &mut R,
    file_len: u64,
    label: &str,
) -> Result<Vec<u8>> {
    let len = read_u32(reader)? as usize;
    ensure_remaining(reader, len, file_len, label)?;
    let mut buf = vec![0u8; len];
    reader
        .read_exact(&mut buf)
        .with_context(|| format!("read {label}"))?;
    Ok(buf)
}

/// Half of the physical RAM, in bytes — the cap we let the chunked index builder spend on its.
fn ram_half_budget_bytes() -> usize {
    if let Ok(s) = std::env::var("KIRA_INDEX_RAM_MB")
        && let Ok(mb) = s.parse::<usize>()
    {
        return mb.saturating_mul(1024 * 1024);
    }
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let raw_total = sys.total_memory() as usize;
    let total_bytes = if raw_total < (1 << 30) {
        raw_total.saturating_mul(1024) // looked like kB
    } else {
        raw_total
    };
    (total_bytes / 2).max(1 << 30) // never go below 1 GB regardless of probe
}

/// Print current process / system memory utilisation.
fn log_memory_state(label: &str) {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let raw_used = sys.used_memory() as usize;
    let raw_total = sys.total_memory() as usize;
    let (used_mb, total_mb) = if raw_total < (1 << 30) {
        (raw_used / 1024, raw_total / 1024)
    } else {
        (raw_used / (1024 * 1024), raw_total / (1024 * 1024))
    };
    eprintln!(
        "[KIRA_INDEX] mem [{label}]: used={}MB / total={}MB ({}%)",
        used_mb,
        total_mb,
        used_mb
            .saturating_mul(100)
            .checked_div(total_mb)
            .unwrap_or(0)
    );
}

fn write_u32<W: Write>(writer: &mut W, value: u32) -> Result<()> {
    writer
        .write_all(&value.to_le_bytes())
        .context("write u32")?;
    Ok(())
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf).context("read u32")?;
    Ok(u32::from_le_bytes(buf))
}

fn write_u64<W: Write>(writer: &mut W, value: u64) -> Result<()> {
    writer
        .write_all(&value.to_le_bytes())
        .context("write u64")?;
    Ok(())
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf).context("read u64")?;
    Ok(u64::from_le_bytes(buf))
}

fn write_u8<W: Write>(writer: &mut W, value: u8) -> Result<()> {
    writer.write_all(&[value]).context("write u8")?;
    Ok(())
}

fn write_bytes<W: Write>(writer: &mut W, bytes: &[u8]) -> Result<()> {
    write_u32(writer, bytes.len() as u32)?;
    writer.write_all(bytes).context("write bytes")?;
    Ok(())
}

fn strand_to_u8(strand: Strand) -> u8 {
    match strand {
        Strand::Forward => 0,
        Strand::Reverse => 1,
    }
}

fn u8_to_strand(v: u8) -> Strand {
    if v == 0 {
        Strand::Forward
    } else {
        Strand::Reverse
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(seq: &[u8]) -> Reference {
        Reference {
            sequences: vec![RefSeq {
                name: "chr1".to_string(),
                bases: RefBases::Owned(seq.to_vec()),
            }],
        }
    }

    fn occurrences(index: &MinimizerIndex, hash: u64) -> Vec<(u32, u32, u8)> {
        let mut out = Vec::new();
        index.for_each_occ(None, hash, &mut |occ| {
            out.push((
                occ.ref_id,
                occ.pos,
                u8::from(matches!(occ.strand, Strand::Reverse)),
            ));
        });
        out.sort_unstable();
        out
    }

    #[test]
    fn repetitive_bucket_retains_only_cap_plus_sentinel() {
        let reference = reference(b"AAAAAAAAAAAAAAAA");
        let index = MinimizerIndex::build_with_budget(&reference, 3, 2, 3, usize::MAX);
        let hash = minimizers(
            reference.sequences[0].bases(None),
            &MinimizerConfig { k: 3, w: 2 },
        )[0]
        .hash;
        assert_eq!(index.bucket_len(None, hash), Some(4));
    }

    #[test]
    fn flat_min_preserves_full_u32_coordinate() {
        let packed = FlatMin::pack(7, 123, u32::MAX, Strand::Reverse);
        let occ = packed.to_occ();
        assert_eq!(occ.ref_id, 123);
        assert_eq!(occ.pos, u32::MAX);
        assert!(matches!(occ.strand, Strand::Reverse));
    }

    #[test]
    fn external_builder_matches_in_memory_buckets() {
        let reference = reference(b"ACGTTGCATGTCGCATGATGCATGAGAGCTACGTTGCATGTCGCATGATG");
        let cfg = MinimizerConfig { k: 5, w: 4 };
        let in_memory = MinimizerIndex::build_with_budget(&reference, cfg.k, cfg.w, 4, usize::MAX);
        let external = MinimizerIndex::build_external(&reference, cfg, cfg.k, cfg.w, 4, 1);
        let mut hashes: Vec<u64> = minimizers(reference.sequences[0].bases(None), &cfg)
            .into_iter()
            .map(|m| m.hash)
            .collect();
        hashes.sort_unstable();
        hashes.dedup();
        for hash in hashes {
            assert_eq!(
                occurrences(&external, hash),
                occurrences(&in_memory, hash),
                "hash {hash}"
            );
        }
    }
}
