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
    let reads = input.reads;
    let sketches = input.sketches;
    let mut stats = SeedBatchStats::default();

    let results: Vec<(Vec<Anchor>, usize)> = reads
        .par_iter()
        .zip(sketches.par_iter())
        .map_init(ThreadCtx::default, |ctx, (read, sketch)| {
            seed_one(read, sketch, index, cfg, ctx)
        })
        .collect();

    let mut anchors: Vec<Vec<Anchor>> = Vec::with_capacity(results.len());
    for (a, before) in results {
        stats.anchors_before_prune += before;
        stats.anchors_after_prune += a.len();
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
) -> (Vec<Anchor>, usize) {
    let table = if read.seq.len() >= cfg.long_read_threshold {
        &index.long
    } else {
        &index.short
    };

    ctx.proto.clear();
    ctx.candidates.clear();
    ctx.diag_counts.clear();
    ctx.rc_ready = false;

    // ranked/pruned seeding: limit each minimizer bucket to top-K occurrences
    let mins = &sketch.minimizers;
    let read_len = read.seq.len() as u32;
    let k = sketch.k as u32;

    let n_mins = mins.len();
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

    for (m_idx, m) in mins.iter().enumerate() {
        let (start, end) = match ctx.buckets_scratch[m_idx] {
            Some(range) => range,
            None => continue,
        };
        let bucket_len = end - start;
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

        let k_hits = if bucket_len <= 8 {
            8
        } else if bucket_len <= 32 {
            4
        } else {
            2
        };
        ctx.occs
            .sort_by_key(|(rid, pos, strand)| (*rid, *pos, *strand as u8));
        let take_n = k_hits.min(ctx.occs.len());
        for &(rid, pos, strand) in ctx.occs[..take_n].iter() {
            let is_rev = strand != m.strand;
            let read_pos = if is_rev {
                (read.seq.len() - m.pos as usize - sketch.k) as u32
            } else {
                m.pos
            };
            let diag = pos as i32 - read_pos as i32;
            let key = (rid, strand, diag);
            let entry = ctx.proto.entry(key).or_insert_with(|| ProtoAnchor {
                ref_id: rid,
                strand,
                diag,
                read_start: read_pos,
                read_end: read_pos + k,
                ref_start: pos,
                ref_end: pos + k,
                hits: 0,
            });
            entry.hits += 1;
            entry.read_start = entry.read_start.min(read_pos);
            entry.read_end = entry.read_end.max(read_pos + k);
            entry.ref_start = entry.ref_start.min(pos);
            entry.ref_end = entry.ref_end.max(pos + k);
        }
    }

    let before = ctx.proto.len();
    ctx.anchors_before_prune = before;

    for (_, proto) in ctx.proto.drain() {
        let score = (proto.hits as i32) * (sketch.k as i32);
        ctx.candidates.push(AnchorCandidate { proto, score });
    }

    // Hard caps and diagonal pruning
    const MAX_ANCHORS_PER_READ: usize = 64;
    const MAX_ANCHORS_PER_DIAG: usize = 8;

    ctx.candidates.sort_by_key(|c| std::cmp::Reverse(c.score));
    ctx.anchors.clear();

    for cand_idx in 0..ctx.candidates.len() {
        if ctx.anchors.len() >= MAX_ANCHORS_PER_READ {
            break;
        }
        let cand = &ctx.candidates[cand_idx];
        let key = (cand.proto.ref_id, cand.proto.strand, cand.proto.diag);
        let count = ctx.diag_counts.entry(key).or_insert(0);
        if *count >= MAX_ANCHORS_PER_DIAG {
            continue;
        }
        *count += 1;

        // Optional exact extension: only for top few candidates.
        let anchor = if ctx.anchors.len() < 8 {
            let cand = &ctx.candidates[cand_idx];
            extend_proto(read, index, cand, &mut ctx.rc_buf, &mut ctx.rc_ready)
        } else {
            let cand = &ctx.candidates[cand_idx];
            Anchor {
                read_start: cand.proto.read_start,
                read_end: cand.proto.read_end.min(read_len),
                ref_id: cand.proto.ref_id,
                ref_start: cand.proto.ref_start,
                ref_end: cand.proto.ref_end,
                strand: cand.proto.strand,
                score: cand.score,
            }
        };
        ctx.anchors.push(anchor);
    }

    (std::mem::take(&mut ctx.anchors), before)
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
    proto: FxHashMap<(u32, Strand, i32), ProtoAnchor>,
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
        ctx.hashes_scratch.reserve(n - ctx.hashes_scratch.capacity());
    }
}
