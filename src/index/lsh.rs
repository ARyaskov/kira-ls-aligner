//! Locality-sensitive hashing via SimHash for fuzzy reference seeding.
//!
//! Reference windows are SimHashed once at build time and bucketed by the
//! top `top_bits` of their fingerprint. At query time, a read window's
//! SimHash lands in one bucket and the candidates inside that bucket are
//! filtered to those within `max_hamming` bits. The downstream verifier
//! (see [`crate::alignment::lsh_rescue`]) then does a base-level Hamming
//! check on the full read against each candidate slice before accepting.
//!
//! Single-bucket lookup means a read whose SimHash's top bits got flipped
//! by a mismatch will miss its true bucket. The rescue wrapper mitigates
//! this by probing multiple non-overlapping windows in the read — different
//! windows hit different buckets, so any one of them landing correctly is
//! enough.

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::sketch::simhash::{hamming_distance, simhash_window};

/// Default top-bits used for bucket keys. With a uniform fingerprint
/// distribution and `top_bits = 20`, the bucket space is 1 M wide — large
/// enough that a 3 Gbp reference (≈ 200 M windows at stride 16) still
/// averages ~200 entries per bucket, which the verifier can chew through.
pub const DEFAULT_TOP_BITS: u32 = 20;

/// Default Hamming bound for in-bucket filtering. Single-base flips
/// empirically move the SimHash by ≤ ~6 bits in our tests; 6 keeps recall
/// at 1-2 mismatches high without flooding the verifier with junk.
pub const DEFAULT_MAX_HAMMING: u32 = 6;

/// Default stride between reference windows. Half the default window length
/// gives 2× overlapping coverage so a read window aligns to *some* indexed
/// window with offset ≤ stride/2 from a real reference position.
pub const DEFAULT_STRIDE: usize = 16;

/// Pack `(ref_id, ref_pos)` into a u64 — matches the layout used by CGK.
#[inline]
fn pack_window(ref_id: u32, pos: u32) -> u64 {
    ((ref_id as u64) << 32) | (pos as u64)
}

#[inline]
fn unpack_window(packed: u64) -> (u32, u32) {
    ((packed >> 32) as u32, packed as u32)
}

/// LSH index: top-bits bucket → list of `(full_hash, packed(ref_id, pos))`.
#[derive(Clone, Debug)]
pub struct LshIndex {
    buckets: FxHashMap<u32, Vec<(u64, u64)>>,
    /// Window length used at build time. Query windows MUST match this.
    pub window_len: usize,
    /// Stride between successive reference windows.
    pub stride: usize,
    /// Number of high-order bits used as the bucket key.
    pub top_bits: u32,
}

impl LshIndex {
    /// Build serially from an iterator of `(ref_id, bases)`. Use
    /// [`Self::build_parallel`] when the iterator is materialised.
    pub fn build<'a, I>(seqs: I, window_len: usize, stride: usize, top_bits: u32) -> Self
    where
        I: IntoIterator<Item = (u32, &'a [u8])>,
    {
        assert!(
            top_bits > 0 && top_bits <= 32,
            "top_bits must be in (0, 32]"
        );
        assert!(window_len > 0, "window_len must be positive");
        assert!(stride > 0, "stride must be positive");
        let shift = 64 - top_bits;
        let mut buckets: FxHashMap<u32, Vec<(u64, u64)>> = FxHashMap::default();
        for (ref_id, bases) in seqs {
            ingest_contig(&mut buckets, ref_id, bases, window_len, stride, shift);
        }
        Self {
            buckets,
            window_len,
            stride,
            top_bits,
        }
    }

    /// Parallel build across contigs via rayon. Each contig produces a local
    /// bucket map; the maps are merged on the main thread. Single-contig
    /// references (E. coli) still run on one thread.
    pub fn build_parallel(
        seqs: Vec<(u32, &[u8])>,
        window_len: usize,
        stride: usize,
        top_bits: u32,
    ) -> Self {
        assert!(
            top_bits > 0 && top_bits <= 32,
            "top_bits must be in (0, 32]"
        );
        assert!(window_len > 0, "window_len must be positive");
        assert!(stride > 0, "stride must be positive");
        let shift = 64 - top_bits;

        let local_maps: Vec<FxHashMap<u32, Vec<(u64, u64)>>> = seqs
            .par_iter()
            .map(|(ref_id, bases)| {
                let mut local: FxHashMap<u32, Vec<(u64, u64)>> = FxHashMap::default();
                ingest_contig(&mut local, *ref_id, bases, window_len, stride, shift);
                local
            })
            .collect();

        let est: usize = local_maps.iter().map(|m| m.len()).sum();
        let mut buckets: FxHashMap<u32, Vec<(u64, u64)>> =
            FxHashMap::with_capacity_and_hasher(est, Default::default());
        for local in local_maps {
            for (k, mut v) in local {
                buckets.entry(k).or_default().append(&mut v);
            }
        }
        Self {
            buckets,
            window_len,
            stride,
            top_bits,
        }
    }

    /// Total `(hash, position)` entries across all buckets.
    pub fn entry_count(&self) -> usize {
        self.buckets.values().map(|v| v.len()).sum()
    }

    /// Number of populated buckets.
    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

    /// Approximate memory footprint in bytes — sum of all `(u64, u64)`
    /// payloads plus the bucket map overhead estimate. Used in startup logs.
    pub fn approx_bytes(&self) -> usize {
        let payload = self.entry_count() * std::mem::size_of::<(u64, u64)>();
        let overhead = self.buckets.len() * 32; // FxHashMap bucket entry estimate
        payload + overhead
    }

    /// Fuzzy lookup. Returns deduplicated `(ref_id, ref_pos)` candidates
    /// whose stored fingerprint is within `max_dist` Hamming of `query_hash`.
    ///
    /// Only the single bucket keyed by `query_hash`'s top bits is probed.
    /// If a mismatch flipped a high-order bit, the query lands in a sibling
    /// bucket and the matching window is missed — that's the recall
    /// tradeoff documented at the module level.
    pub fn lookup_fuzzy(&self, query_hash: u64, max_dist: u32) -> Vec<(u32, u32)> {
        let shift = 64 - self.top_bits;
        let bucket_key = (query_hash >> shift) as u32;
        let Some(candidates) = self.buckets.get(&bucket_key) else {
            return Vec::new();
        };
        let mut out: Vec<(u32, u32)> = Vec::new();
        for (stored_hash, packed) in candidates {
            if hamming_distance(*stored_hash, query_hash) <= max_dist {
                out.push(unpack_window(*packed));
            }
        }
        out
    }

    /// Same as [`Self::lookup_fuzzy`] but writes into a caller-owned buffer
    /// to avoid per-query allocation in tight loops.
    pub fn lookup_fuzzy_into(&self, query_hash: u64, max_dist: u32, out: &mut Vec<(u32, u32)>) {
        out.clear();
        let shift = 64 - self.top_bits;
        let bucket_key = (query_hash >> shift) as u32;
        let Some(candidates) = self.buckets.get(&bucket_key) else {
            return;
        };
        for (stored_hash, packed) in candidates {
            if hamming_distance(*stored_hash, query_hash) <= max_dist {
                out.push(unpack_window(*packed));
            }
        }
    }
}

/// Slide `window_len` over `bases` at `stride`, push each SimHash into
/// `buckets` keyed by the top `top_bits` of the fingerprint.
fn ingest_contig(
    buckets: &mut FxHashMap<u32, Vec<(u64, u64)>>,
    ref_id: u32,
    bases: &[u8],
    window_len: usize,
    stride: usize,
    shift: u32,
) {
    if bases.len() < window_len {
        return;
    }
    let last_start = bases.len() - window_len;
    let mut pos: usize = 0;
    while pos <= last_start {
        let window = &bases[pos..pos + window_len];
        if let Some(hash) = simhash_window(window) {
            let bucket_key = (hash >> shift) as u32;
            buckets
                .entry(bucket_key)
                .or_default()
                .push((hash, pack_window(ref_id, pos as u32)));
        }
        pos += stride;
    }
}

#[cfg(test)]
#[path = "../../tests/unit/index_lsh.rs"]
mod tests;
