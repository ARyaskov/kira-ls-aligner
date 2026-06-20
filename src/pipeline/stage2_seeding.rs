use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::index::Index;
use crate::seq::reverse_complement_into;
use crate::types::{Anchor, ReadRecord, Strand};

use super::stage1_sketch::{ReadSketch, SketchBatch};

/// Stage 2 output: anchors per read.
#[derive(Clone, Debug)]
pub struct SeedBatch {
    pub reads: Vec<ReadRecord>,
    pub anchors: Vec<Vec<Anchor>>,
    pub stats: SeedBatchStats,
}

/// Per-batch seeding stats.
#[derive(Clone, Debug, Default)]
pub struct SeedBatchStats {
    pub anchors_before_prune: usize,
    pub anchors_after_prune: usize,
}

#[derive(Clone, Debug)]
struct ProtoAnchor {
    ref_id: u32,
    strand: Strand,
    diag: i32,
    read_start: u32,
    read_end: u32,
    ref_start: u32,
    ref_end: u32,
    hits: u32,
}

#[derive(Clone, Debug)]
struct AnchorCandidate {
    proto: ProtoAnchor,
    score: i32,
}

pub fn run(input: SketchBatch, index: &Index, cfg: crate::seeding::SeedingConfig) -> SeedBatch {
    let mut reads = input.reads;
    let sketches = input.sketches;
    let mut stats = SeedBatchStats::default();

    let results: Vec<(Vec<Anchor>, usize, u32)> = reads
        .par_iter()
        .zip(sketches.par_iter())
        .map_init(ThreadCtx::default, |ctx, (read, sketch)| {
            seed_one(read, sketch, index, cfg, ctx)
        })
        .collect();

    let mut anchors: Vec<Vec<Anchor>> = Vec::with_capacity(results.len());
    for (i, (a, before, min_occ)) in results.into_iter().enumerate() {
        stats.anchors_before_prune += before;
        stats.anchors_after_prune += a.len();
        reads[i].repeat_min_occ = min_occ;
        anchors.push(a);
    }

    SeedBatch {
        reads,
        anchors,
        stats,
    }
}

fn seed_one(
    read: &ReadRecord,
    sketch: &ReadSketch,
    index: &Index,
    cfg: crate::seeding::SeedingConfig,
    ctx: &mut ThreadCtx,
) -> (Vec<Anchor>, usize, u32) {
    let table = if read.seq.len() >= cfg.long_read_threshold {
        &index.long
    } else {
        &index.short
    };

    ctx.proto.clear();
    ctx.merged.clear();
    ctx.candidates.clear();
    ctx.diag_counts.clear();
    ctx.rc_ready = false;

    // ranked/pruned seeding: limit each minimizer bucket to top-K occurrences
    let mins = &sketch.minimizers;
    let read_len = read.seq.len() as u32;
    let k = sketch.k as u32;

    let n_mins = mins.len();
    let proto_soft_limit = anchor_limit(read.seq.len(), sketch.k)
        .saturating_mul(16)
        .max(4096);
    grow_scratch(ctx, n_mins);
    ctx.hashes_scratch.clear();
    for m in mins {
        ctx.hashes_scratch.push(m.hash);
    }
    index.bucket_batch_into(
        table,
        &ctx.hashes_scratch,
        &mut ctx.canon_scratch,
        &mut ctx.ids_scratch,
        &mut ctx.buckets_scratch,
    );

    let mut min_occ: u32 = u32::MAX;
    for (m_idx, m) in mins.iter().enumerate() {
        let (start, end) = match ctx.buckets_scratch[m_idx] {
            Some(range) => range,
            None => continue,
        };
        let bucket_len = end - start;
        if bucket_len >= 1 {
            min_occ = min_occ.min(bucket_len as u32);
        }
        if bucket_len == 0 || bucket_len > cfg.max_occ {
            continue;
        }

        ctx.occs.clear();
        let slot_opt = ctx.ids_scratch[m_idx];
        let mut hot_hit = false;
        if let Some(slot) = slot_opt {
            if let Some(cached) = table.hot_lookup(slot as u32) {
                ctx.occs.extend_from_slice(cached);
                hot_hit = true;
            }
        }
        if !hot_hit {
            for occ_idx in start..end {
                let o = index.occ_at(table, occ_idx);
                ctx.occs.push((o.ref_id, o.pos, o.strand));
            }
        }

        // Retain enough copies to avoid the old lowest-coordinate bias without
        // letting moderate repeats dominate seeding and chaining work.
        let k_hits = cfg.max_hits_per_minimizer.max(1);
        // Sample repeat occurrences by a deterministic hash instead of reference
        // coordinate. Coordinate-order sampling systematically maps dispersed
        // repeats to the first copy in the genome.
        let take_n = k_hits.min(ctx.occs.len());
        if take_n < ctx.occs.len() {
            ctx.occs
                .select_nth_unstable_by_key(take_n, |(rid, pos, strand)| {
                    occurrence_sample_key(m.hash, *rid, *pos, *strand)
                });
        }
        for &(rid, pos, ref_strand) in ctx.occs[..take_n].iter() {
            let (read_pos, strand, diag) = relative_seed_coordinates(
                read.seq.len(),
                sketch.k,
                m.pos,
                m.strand,
                pos,
                ref_strand,
            );
            ctx.proto.push(ProtoAnchor {
                ref_id: rid,
                strand,
                diag,
                read_start: read_pos,
                read_end: read_pos + k,
                ref_start: pos,
                ref_end: pos + k,
                hits: 1,
            });
        }
        if ctx.proto.len() > proto_soft_limit {
            compact_proto_anchors(
                &mut ctx.proto,
                &mut ctx.merged,
                proto_soft_limit,
                read.seq.len() as u64,
            );
        }
    }

    merge_exact_seed_runs(&mut ctx.proto, &mut ctx.merged);
    let before = ctx.merged.len();
    ctx.anchors_before_prune = before;

    for proto in ctx.merged.drain(..) {
        let score = proto
            .read_end
            .saturating_sub(proto.read_start)
            .min(proto.ref_end.saturating_sub(proto.ref_start)) as i32;
        ctx.candidates.push(AnchorCandidate { proto, score });
    }

    let max_anchors_per_read = anchor_limit(read.seq.len(), sketch.k);
    let max_anchors_per_diag = if read.seq.len() >= cfg.long_read_threshold {
        64
    } else {
        16
    };

    ctx.candidates.sort_by_key(|c| {
        (
            std::cmp::Reverse(c.score),
            occurrence_sample_key(
                read.seq.len() as u64,
                c.proto.ref_id,
                c.proto.ref_start,
                c.proto.strand,
            ),
        )
    });
    ctx.anchors.clear();

    for cand_idx in 0..ctx.candidates.len() {
        if ctx.anchors.len() >= max_anchors_per_read {
            break;
        }
        let cand = &ctx.candidates[cand_idx];
        let key = (cand.proto.ref_id, cand.proto.strand, cand.proto.diag);
        let count = ctx.diag_counts.entry(key).or_insert(0);
        if *count >= max_anchors_per_diag {
            continue;
        }
        *count += 1;

        let anchor = extend_proto(read, index, cand, &mut ctx.rc_buf, &mut ctx.rc_ready);
        if anchor.read_end.saturating_sub(anchor.read_start) >= cfg.min_anchor_len
            && anchor.read_end <= read_len
        {
            ctx.anchors.push(anchor);
        }
    }
    ctx.anchors.sort_by_key(|a| {
        (
            a.ref_id,
            u8::from(a.strand),
            a.ref_start,
            a.read_start,
            a.ref_end,
            a.read_end,
        )
    });
    ctx.anchors.dedup_by(|a, b| {
        a.ref_id == b.ref_id
            && a.strand == b.strand
            && a.ref_start == b.ref_start
            && a.ref_end == b.ref_end
            && a.read_start == b.read_start
            && a.read_end == b.read_end
    });

    let min_occ = if min_occ == u32::MAX { 0 } else { min_occ };
    (std::mem::take(&mut ctx.anchors), before, min_occ)
}

#[inline]
fn relative_seed_coordinates(
    read_len: usize,
    k: usize,
    minimizer_pos: u32,
    minimizer_strand: Strand,
    ref_pos: u32,
    ref_strand: Strand,
) -> (u32, Strand, i32) {
    let is_rev = ref_strand != minimizer_strand;
    let read_pos = if is_rev {
        read_len
            .saturating_sub(minimizer_pos as usize)
            .saturating_sub(k) as u32
    } else {
        minimizer_pos
    };
    let strand = if is_rev {
        Strand::Reverse
    } else {
        Strand::Forward
    };
    (read_pos, strand, ref_pos as i32 - read_pos as i32)
}

#[inline]
fn occurrence_sample_key(hash: u64, ref_id: u32, pos: u32, strand: Strand) -> u64 {
    let mut x = hash ^ ((ref_id as u64) << 32) ^ pos as u64 ^ ((u8::from(strand) as u64) << 63);
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn merge_exact_seed_runs(input: &mut Vec<ProtoAnchor>, output: &mut Vec<ProtoAnchor>) {
    input.sort_by_key(|p| {
        (
            p.ref_id,
            u8::from(p.strand),
            p.diag,
            p.read_start,
            p.ref_start,
        )
    });
    output.clear();
    for proto in input.drain(..) {
        if let Some(last) = output.last_mut() {
            let same_diagonal = last.ref_id == proto.ref_id
                && last.strand == proto.strand
                && last.diag == proto.diag;
            // Overlap/touch guarantees every base in the merged interval was
            // covered by an exact k-mer. Never bridge an unchecked gap.
            if same_diagonal && proto.read_start <= last.read_end && proto.ref_start <= last.ref_end
            {
                last.read_end = last.read_end.max(proto.read_end);
                last.ref_end = last.ref_end.max(proto.ref_end);
                last.hits = last.hits.saturating_add(proto.hits);
                continue;
            }
        }
        output.push(proto);
    }
}

fn compact_proto_anchors(
    input: &mut Vec<ProtoAnchor>,
    scratch: &mut Vec<ProtoAnchor>,
    limit: usize,
    salt: u64,
) {
    merge_exact_seed_runs(input, scratch);
    std::mem::swap(input, scratch);
    scratch.clear();
    if input.len() > limit {
        input.select_nth_unstable_by_key(limit, |p| {
            occurrence_sample_key(
                salt ^ p.diag as u64,
                p.ref_id,
                p.ref_start ^ p.read_start,
                p.strand,
            )
        });
        input.truncate(limit);
    }
}

fn anchor_limit(read_len: usize, k: usize) -> usize {
    if read_len < 500 {
        return 256;
    }
    let seeds = read_len.div_ceil(k.max(1));
    seeds.saturating_mul(8).clamp(512, 8192)
}

/// Run exact-match extension around an anchor candidate.
fn extend_proto(
    read: &ReadRecord,
    index: &Index,
    cand: &AnchorCandidate,
    rc_buf: &mut Vec<u8>,
    rc_ready: &mut bool,
) -> Anchor {
    let ref_seq = index.ref_bases(cand.proto.ref_id as usize);
    let (read_seq, strand): (&[u8], Strand) = if cand.proto.strand == Strand::Reverse {
        if !*rc_ready {
            reverse_complement_into(&read.seq, rc_buf);
            *rc_ready = true;
        }
        (rc_buf.as_slice(), Strand::Reverse)
    } else {
        (read.seq.as_slice(), Strand::Forward)
    };

    let mut q_start = cand.proto.read_start as i32;
    let mut r_start = cand.proto.ref_start as i32;
    let mut q_end = cand.proto.read_end as i32;
    let mut r_end = cand.proto.ref_end as i32;

    while q_start > 0 && r_start > 0 {
        let qb = read_seq[(q_start - 1) as usize];
        let rb = ref_seq[(r_start - 1) as usize];
        if qb != rb {
            break;
        }
        q_start -= 1;
        r_start -= 1;
    }
    while (q_end as usize) < read_seq.len() && (r_end as usize) < ref_seq.len() {
        let qb = read_seq[q_end as usize];
        let rb = ref_seq[r_end as usize];
        if qb != rb {
            break;
        }
        q_end += 1;
        r_end += 1;
    }

    let len = (q_end - q_start).max(0) as u32;
    let score = len as i32;

    Anchor {
        read_start: q_start as u32,
        read_end: q_end as u32,
        ref_id: cand.proto.ref_id,
        ref_start: r_start as u32,
        ref_end: r_end as u32,
        strand,
        score,
    }
}

#[derive(Default)]
struct ThreadCtx {
    proto: Vec<ProtoAnchor>,
    merged: Vec<ProtoAnchor>,
    diag_counts: FxHashMap<(u32, Strand, i32), usize>,
    occs: Vec<(u32, u32, Strand)>,
    candidates: Vec<AnchorCandidate>,
    anchors: Vec<Anchor>,
    rc_buf: Vec<u8>,
    rc_ready: bool,
    anchors_before_prune: usize,
    /// Per-read minimizer hash batch — populated then passed to `Index::bucket_batch_into`.
    hashes_scratch: Vec<u64>,
    /// Canonical-hash scratch for `lookup_batch_u64_simd_into`.
    canon_scratch: Vec<u64>,
    /// MPH-id results for `lookup_batch_u64_simd_into`.
    ids_scratch: Vec<Option<usize>>,
    /// `(start, end)` bucket descriptors — same length as `hashes_scratch`.
    buckets_scratch: Vec<Option<(usize, usize)>>,
}

/// Grow per-thread scratch buffers to fit at least `n` minimizers.
#[inline]
fn grow_scratch(ctx: &mut ThreadCtx, n: usize) {
    if ctx.canon_scratch.len() < n {
        ctx.canon_scratch.resize(n, 0);
    }
    if ctx.ids_scratch.len() < n {
        ctx.ids_scratch.resize(n, None);
    }
    if ctx.buckets_scratch.len() < n {
        ctx.buckets_scratch.resize(n, None);
    }
    if ctx.hashes_scratch.capacity() < n {
        ctx.hashes_scratch
            .reserve(n - ctx.hashes_scratch.capacity());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_strand_is_relative_not_reference_canonical_strand() {
        let cases = [
            (Strand::Forward, Strand::Forward, Strand::Forward),
            (Strand::Reverse, Strand::Reverse, Strand::Forward),
            (Strand::Forward, Strand::Reverse, Strand::Reverse),
            (Strand::Reverse, Strand::Forward, Strand::Reverse),
        ];
        for (read_strand, ref_strand, expected) in cases {
            let (_, actual, _) =
                relative_seed_coordinates(100, 15, 20, read_strand, 200, ref_strand);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn reverse_seed_uses_oriented_query_coordinate() {
        let (read_pos, strand, diag) =
            relative_seed_coordinates(100, 15, 20, Strand::Forward, 200, Strand::Reverse);
        assert_eq!(read_pos, 65);
        assert_eq!(strand, Strand::Reverse);
        assert_eq!(diag, 135);
    }

    #[test]
    fn exact_seed_runs_do_not_bridge_unchecked_gaps() {
        let mk = |q: u32, r: u32| ProtoAnchor {
            ref_id: 0,
            strand: Strand::Forward,
            diag: r as i32 - q as i32,
            read_start: q,
            read_end: q + 15,
            ref_start: r,
            ref_end: r + 15,
            hits: 1,
        };
        let mut input = vec![mk(0, 100), mk(10, 110), mk(30, 130)];
        let mut output = Vec::new();
        merge_exact_seed_runs(&mut input, &mut output);
        assert_eq!(output.len(), 2);
        assert_eq!((output[0].read_start, output[0].read_end), (0, 25));
        assert_eq!((output[1].read_start, output[1].read_end), (30, 45));
    }

    #[test]
    fn long_read_anchor_limit_scales_with_length() {
        assert_eq!(anchor_limit(150, 19), 256);
        assert!(anchor_limit(10_000, 19) > 256);
        assert!(anchor_limit(1_000_000, 19) <= 8192);
    }
}
