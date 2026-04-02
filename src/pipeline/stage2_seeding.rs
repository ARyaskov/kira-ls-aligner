use std::collections::HashMap;

use rayon::prelude::*;

use crate::index::Index;
use crate::seq::reverse_complement;
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

    // ranked/pruned seeding: limit each minimizer bucket to top-K occurrences
    let mins = &sketch.minimizers;
    let read_len = read.seq.len() as u32;
    let k = sketch.k as u32;

    for m in mins {
        let bucket_len = match index.bucket_len(table, m.hash) {
            Some(len) => len,
            None => continue,
        };
        if bucket_len == 0 || bucket_len > cfg.max_occ {
            continue;
        }

        let mut occs: Vec<(u32, u32, Strand)> = Vec::new();
        index.for_each_occ(table, m.hash, |o| {
            occs.push((o.ref_id, o.pos, o.strand));
        });

        // Deterministic ranking: prefer smaller buckets and coherent diagonals.
        // Keep only top-K occurrences per minimizer.
        let k_hits = if bucket_len <= 8 {
            8
        } else if bucket_len <= 32 {
            4
        } else {
            2
        };
        occs.sort_by_key(|(rid, pos, strand)| (*rid, *pos, *strand as u8));
        for (rid, pos, strand) in occs.into_iter().take(k_hits) {
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
    let mut diag_counts: HashMap<(u32, Strand, i32), usize> = HashMap::new();
    ctx.anchors.clear();

    for cand in ctx.candidates.iter() {
        if ctx.anchors.len() >= MAX_ANCHORS_PER_READ {
            break;
        }
        let key = (cand.proto.ref_id, cand.proto.strand, cand.proto.diag);
        let count = diag_counts.entry(key).or_insert(0);
        if *count >= MAX_ANCHORS_PER_DIAG {
            continue;
        }
        *count += 1;

        // Optional exact extension: only for top few candidates
        let anchor = if ctx.anchors.len() < 8 {
            extend_proto(read, index, cand)
        } else {
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

fn extend_proto(read: &ReadRecord, index: &Index, cand: &AnchorCandidate) -> Anchor {
    let ref_seq = index.ref_bases(cand.proto.ref_id as usize);
    let rc_read = reverse_complement(&read.seq);
    let (read_seq, strand) = if cand.proto.strand == Strand::Reverse {
        (&rc_read, Strand::Reverse)
    } else {
        (&read.seq, Strand::Forward)
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
    proto: HashMap<(u32, Strand, i32), ProtoAnchor>,
    candidates: Vec<AnchorCandidate>,
    anchors: Vec<Anchor>,
    anchors_before_prune: usize,
}
