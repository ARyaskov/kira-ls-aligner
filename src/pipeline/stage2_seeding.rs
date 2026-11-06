use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::index::Index;
use crate::seq::{common_prefix_len, common_suffix_len, reverse_complement_into};
use crate::types::{Anchor, ReadRecord, Strand};

use super::stage0_input::InputBatch;
use super::stage1_sketch::{ReadSketch, SketchBatch};
use super::stage1_sketch::{SketchBatchStats, SketchConfig, batch_stats};

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
    pub adaptive_retries: usize,
    pub repeat_rescues: usize,
}

type SeedResult = (Vec<Anchor>, usize, u32);

thread_local! {
    static SEED_SCRATCH: std::cell::RefCell<ThreadCtx> =
        std::cell::RefCell::new(ThreadCtx::default());
}

/// Fuse sketching and seeding in pair-aligned blocks. The two minimizer
/// buffers per worker are reused across jobs and batches, instead of retaining
/// a separate heap allocation for every read's sketch until seeding finishes.
pub fn run_fused(
    input: InputBatch,
    index: &Index,
    sketch_cfg: SketchConfig,
    cfg: crate::seeding::SeedingConfig,
    skip_mask: &[bool],
) -> (SeedBatch, SketchBatchStats) {
    let mut reads = input.reads;
    let blocks: Vec<_> = reads
        .par_chunks(128)
        .enumerate()
        .map(|(block_idx, block)| {
            SEED_SCRATCH.with(|cell| {
                let mut ctx = cell.borrow_mut();
                let mut sketches = std::mem::take(&mut ctx.sketches);
                let before_retries = ctx.adaptive_retries;
                let before_rescues = ctx.repeat_rescues;
                let mut total_mins = 0;
                let mut out = Vec::with_capacity(block.len());
                for (template_idx, pair) in block.chunks(2).enumerate() {
                    for (mate, read) in pair.iter().enumerate() {
                        let s = &mut sketches[mate];
                        (s.k, s.w) = if read.seq.len() >= sketch_cfg.long_read_threshold {
                            (sketch_cfg.long_k, sketch_cfg.long_w)
                        } else {
                            (sketch_cfg.short_k, sketch_cfg.short_w)
                        };
                        s.minimizers.clear();
                        let idx = block_idx * 128 + template_idx * 2 + mate;
                        if !skip_mask.get(idx).copied().unwrap_or(false) {
                            crate::sketch::minimizers_into(
                                &read.seq,
                                &crate::sketch::MinimizerConfig { k: s.k, w: s.w },
                                &mut s.minimizers,
                            );
                        }
                        total_mins += s.minimizers.len();
                    }
                    out.extend(
                        seed_template(pair, &sketches[..pair.len()], index, cfg, &mut ctx)
                            .into_iter()
                            .take(pair.len()),
                    );
                }
                ctx.sketches = sketches;
                (
                    out,
                    total_mins,
                    ctx.adaptive_retries - before_retries,
                    ctx.repeat_rescues - before_rescues,
                )
            })
        })
        .collect();
    let mut anchors = Vec::with_capacity(reads.len());
    let mut stats = SeedBatchStats::default();
    let mut total_mins = 0;
    for (block, mins, retries, rescues) in blocks {
        total_mins += mins;
        stats.adaptive_retries += retries;
        stats.repeat_rescues += rescues;
        for (a, before, min_occ) in block {
            reads[anchors.len()].repeat_min_occ = min_occ;
            stats.anchors_before_prune += before;
            stats.anchors_after_prune += a.len();
            anchors.push(a);
        }
    }
    let sketch_stats = batch_stats(&reads, total_mins);
    (
        SeedBatch {
            reads,
            anchors,
            stats,
        },
        sketch_stats,
    )
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
    /// Deterministic tiebreak among equally-scoring candidates. Precomputed so
    /// the sort comparator does not rehash O(n log n) times per read.
    tiebreak: u64,
}

/// Width of a mate-hint bin, in bases. Coarse on purpose: binning turns the
/// window test into a couple of hash probes instead of a range scan.
const HINT_BIN: u32 = 512;

/// Most distinct hint bins a mate may occupy for its hints to be trusted. A mate
/// scattered over many loci carries no positional information, and following it
/// anyway promotes wrong placements.
const HINT_MAX_BINS: usize = 8;

/// Positions where a read's mate plausibly sits, as a coarse bin set.
#[derive(Default)]
struct MateHints {
    bins: rustc_hash::FxHashSet<(u32, u32)>,
}

impl MateHints {
    fn build_from(&mut self, anchors: &[Anchor]) {
        self.bins.clear();
        for a in anchors {
            self.bins.insert((a.ref_id, a.ref_start / HINT_BIN));
        }
    }

    /// Usable only when the mate is concentrated (see [`HINT_MAX_BINS`]).
    fn is_usable(&self) -> bool {
        !self.bins.is_empty() && self.bins.len() <= HINT_MAX_BINS
    }

    /// Does the mate sit within `window` bases of `(ref_id, pos)`?
    fn supports(&self, ref_id: u32, pos: u32, window: u32) -> bool {
        let lo = pos.saturating_sub(window) / HINT_BIN;
        let hi = pos.saturating_add(window) / HINT_BIN;
        (lo..=hi).any(|bin| self.bins.contains(&(ref_id, bin)))
    }
}

pub fn run(input: SketchBatch, index: &Index, cfg: crate::seeding::SeedingConfig) -> SeedBatch {
    let mut reads = input.reads;
    let sketches = input.sketches;
    let mut stats = SeedBatchStats::default();

    // Paired input is laid out R1, R2, R1, R2 …, so a chunk of two is one
    // template and both mates can inform each other.
    let results: Vec<(SeedResult, usize, usize)> = reads
        .par_chunks(2)
        .zip(sketches.par_chunks(2))
        .map_init(ThreadCtx::default, |ctx, (read_pair, sketch_pair)| {
            let before = ctx.adaptive_retries;
            let before_rescues = ctx.repeat_rescues;
            let [first, second] = seed_template(read_pair, sketch_pair, index, cfg, ctx);
            [
                (
                    first,
                    ctx.adaptive_retries - before,
                    ctx.repeat_rescues - before_rescues,
                ),
                (second, 0, 0),
            ]
        })
        .flatten()
        .collect();

    let mut anchors: Vec<Vec<Anchor>> = Vec::with_capacity(results.len());
    for (i, ((a, before, min_occ), retries, rescues)) in
        results.into_iter().take(reads.len()).enumerate()
    {
        stats.adaptive_retries += retries;
        stats.repeat_rescues += rescues;
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

/// Seed one template (a mate pair, or a single read at the tail of the batch).
///
/// The first mate is seeded blind and its anchors hint the second. It is only
/// re-seeded with the second's hints if its own sampling had to discard
/// occurrences — otherwise a hint could not change its outcome.
fn seed_template(
    read_pair: &[ReadRecord],
    sketch_pair: &[ReadSketch],
    index: &Index,
    cfg: crate::seeding::SeedingConfig,
    ctx: &mut ThreadCtx,
) -> [SeedResult; 2] {
    let paired = cfg.mate_window.is_some()
        && read_pair.len() == 2
        && read_pair[0].pair_role != crate::types::PairRole::Unpaired
        && read_pair[1].pair_role != crate::types::PairRole::Unpaired;

    if !paired {
        return std::array::from_fn(|i| {
            if i >= read_pair.len() {
                return (Vec::new(), 0, 0);
            }
            let (a, before, occ, _) =
                seed_adaptive(&read_pair[i], &sketch_pair[i], index, cfg, ctx, None);
            (a, before, occ)
        });
    }

    let (a0, before0, occ0, truncated0) =
        seed_adaptive(&read_pair[0], &sketch_pair[0], index, cfg, ctx, None);

    let mut hints0 = std::mem::take(&mut ctx.hints_a);
    hints0.build_from(&a0);
    let hint_for_second = hints0.is_usable().then_some(&hints0);
    let (a1, before1, occ1, _) = seed_adaptive(
        &read_pair[1],
        &sketch_pair[1],
        index,
        cfg,
        ctx,
        hint_for_second,
    );

    // Only the mate that lost occurrences can gain from a second pass.
    let (a0, before0, occ0) = if truncated0 {
        let mut hints1 = std::mem::take(&mut ctx.hints_b);
        hints1.build_from(&a1);
        let out = if hints1.is_usable() {
            let (a, before, occ, _) = seed_adaptive(
                &read_pair[0],
                &sketch_pair[0],
                index,
                cfg,
                ctx,
                Some(&hints1),
            );
            (a, before, occ)
        } else {
            (a0, before0, occ0)
        };
        ctx.hints_b = hints1;
        out
    } else {
        (a0, before0, occ0)
    };
    ctx.hints_a = hints0;

    [(a0, before0, occ0), (a1, before1, occ1)]
}

/// Give discarded-repeat reads one bounded retry when retained anchors cover
/// less than half the query. Build-truncated buckets are then re-seeded with
/// longer exact seeds. Every pass is merged so widening never drops a locus.
fn seed_adaptive(
    read: &ReadRecord,
    sketch: &ReadSketch,
    index: &Index,
    cfg: crate::seeding::SeedingConfig,
    ctx: &mut ThreadCtx,
    hints: Option<&MateHints>,
) -> (Vec<Anchor>, usize, u32, bool) {
    let (mut anchors, mut before, occ, discarded) = seed_one(read, sketch, index, cfg, ctx, hints);
    if !discarded || anchor_coverage(read.seq.len(), &anchors, ctx) * 2 >= read.seq.len() {
        return (anchors, before, occ, discarded);
    }

    let mut occ = occ;
    if read.seq.len() < cfg.long_read_threshold {
        let table = &index.short;
        let mut wider = cfg;
        wider.max_occ = cfg.max_occ.saturating_mul(4).min(table.max_occ);
        wider.max_hits_per_minimizer = cfg
            .max_hits_per_minimizer
            .saturating_mul(4)
            .min(64)
            .max(cfg.max_hits_per_minimizer);
        if wider.max_occ > cfg.max_occ || wider.max_hits_per_minimizer > cfg.max_hits_per_minimizer
        {
            ctx.adaptive_retries += 1;
            let (extra, extra_before, extra_occ, _) =
                seed_one(read, sketch, index, wider, ctx, hints);
            anchors.extend(extra);
            sort_dedup_anchors(&mut anchors);
            before += extra_before;
            occ = occ.min(extra_occ);
        }
    }

    if anchor_coverage(read.seq.len(), &anchors, ctx) * 2 < read.seq.len() {
        let (extra, extra_before) = seed_repeat_rescue(read, sketch, index, cfg, ctx, hints);
        if !extra.is_empty() {
            ctx.repeat_rescues += 1;
            anchors.extend(extra);
            sort_dedup_anchors(&mut anchors);
            before += extra_before;
        }
    }
    (anchors, before, occ, discarded)
}

fn anchor_coverage(read_len: usize, anchors: &[Anchor], ctx: &mut ThreadCtx) -> usize {
    ctx.coverage.clear();
    ctx.coverage
        .extend(anchors.iter().map(|a| (a.read_start, a.read_end)));
    ctx.coverage.sort_unstable();
    let (mut end, mut covered) = (0, 0);
    for &(start, stop) in &ctx.coverage {
        covered += stop.saturating_sub(end.max(start)) as usize;
        end = end.max(stop);
    }
    covered.min(read_len)
}

/// Occurrence cap the seeder applies to a table: the smaller of the run-time
/// cap and the cap the index was built with. Buckets over the build cap are
/// stored truncated to `index_max_occ + 1` records as a "too repetitive"
/// sentinel, so a larger run-time cap must never treat them as complete.
#[inline]
pub fn effective_max_occ(seeding_max_occ: usize, index_max_occ: usize) -> usize {
    seeding_max_occ.min(index_max_occ)
}

/// Re-seed only build-truncated primary buckets. Both the number of queried
/// minimizers and occurrences per longer seed are bounded by index construction.
fn seed_repeat_rescue(
    read: &ReadRecord,
    sketch: &ReadSketch,
    index: &Index,
    cfg: crate::seeding::SeedingConfig,
    ctx: &mut ThreadCtx,
    hints: Option<&MateHints>,
) -> (Vec<Anchor>, usize) {
    const MAX_MINIMIZERS: usize = 64;

    let table = if read.seq.len() >= cfg.long_read_threshold {
        &index.long
    } else {
        &index.short
    };
    let Some(extension) = index.repeat_rescue_extension(table) else {
        return (Vec::new(), 0);
    };
    ctx.proto.clear();
    ctx.merged.clear();
    ctx.candidates.clear();
    ctx.diag_counts.clear();
    ctx.rc_ready = false;

    let k = sketch.k as u32;
    let proto_soft_limit = anchor_limit(read.seq.len(), sketch.k)
        .saturating_mul(16)
        .max(4096);
    let take_per_seed = cfg.max_hits_per_minimizer.saturating_mul(4).clamp(1, 64);
    let mut attempted = 0;
    for m in &sketch.minimizers {
        let Some((start, end)) = index.bucket(table, m.hash) else {
            continue;
        };
        if end - start <= table.max_occ {
            continue;
        }
        attempted += 1;
        if attempted > MAX_MINIMIZERS {
            break;
        }
        let Some(hash) =
            crate::sketch::extended_minimizer_hash(&read.seq, m.pos, sketch.k, m.strand, extension)
        else {
            continue;
        };
        let Some((start, end)) = index.repeat_rescue_bucket(table, hash) else {
            continue;
        };
        ctx.occs.clear();
        for idx in start..end {
            let occ = index
                .repeat_rescue_occ_at(table, idx)
                .expect("rescue bucket requires rescue index");
            ctx.occs.push((occ.ref_id, occ.pos, occ.strand));
        }
        let take_n = take_per_seed.min(ctx.occs.len());
        if take_n < ctx.occs.len() {
            match (hints, cfg.mate_window) {
                (Some(h), Some(window)) => {
                    ctx.occs
                        .select_nth_unstable_by_key(take_n, |(rid, pos, strand)| {
                            (
                                !h.supports(*rid, *pos, window),
                                occurrence_sample_key(hash, *rid, *pos, *strand),
                            )
                        });
                }
                _ => {
                    ctx.occs
                        .select_nth_unstable_by_key(take_n, |(rid, pos, strand)| {
                            occurrence_sample_key(hash, *rid, *pos, *strand)
                        });
                }
            }
        }
        for &(rid, pos, ref_strand) in &ctx.occs[..take_n] {
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
    finish_anchors(read, index, sketch.k, cfg, ctx)
}

/// Seed one read. Returns its anchors, the pre-prune anchor count, the minimum
/// seed occurrence count, and whether any bucket had to be truncated.
fn seed_one(
    read: &ReadRecord,
    sketch: &ReadSketch,
    index: &Index,
    cfg: crate::seeding::SeedingConfig,
    ctx: &mut ThreadCtx,
    hints: Option<&MateHints>,
) -> (Vec<Anchor>, usize, u32, bool) {
    let mut truncated_any = false;
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

    let max_occ = effective_max_occ(cfg.max_occ, table.max_occ);
    let mut min_occ: u32 = u32::MAX;
    for (m_idx, m) in mins.iter().enumerate() {
        let (start, end) = match ctx.buckets_scratch[m_idx] {
            Some(range) => range,
            None => continue,
        };
        let bucket_len = end - start;
        if bucket_len >= 1 {
            let true_count = ctx.ids_scratch[m_idx]
                .and_then(|slot| index.bucket_count_at(table, slot))
                .unwrap_or(bucket_len as u32);
            min_occ = min_occ.min(true_count);
        }
        if bucket_len == 0 || bucket_len > max_occ {
            // A build sentinel remains ineligible for primary seeding; the flag
            // makes it eligible for the separate longer-seed rescue pass.
            truncated_any |= bucket_len > max_occ;
            continue;
        }

        let k_hits = cfg.max_hits_per_minimizer.max(1);
        ctx.occs.clear();
        let slot_opt = ctx.ids_scratch[m_idx];
        let mut hot_hit = false;
        if bucket_len > k_hits
            && let Some(slot) = slot_opt
            && let Some(cached) = table.hot_lookup(slot as u32)
        {
            ctx.occs.extend_from_slice(cached);
            hot_hit = true;
        }
        if bucket_len > k_hits && !hot_hit {
            for occ_idx in start..end {
                let o = index.occ_at(table, occ_idx);
                ctx.occs.push((o.ref_id, o.pos, o.strand));
            }
        }

        // Retain enough copies to avoid the old lowest-coordinate bias without
        // letting moderate repeats dominate seeding and chaining work.
        // Sample repeat occurrences by a deterministic hash instead of reference
        // coordinate. Coordinate-order sampling systematically maps dispersed
        // repeats to the first copy in the genome.
        let take_n = k_hits.min(bucket_len);
        if take_n < bucket_len {
            // Over the cap: copies will be dropped either way. Rank
            // mate-supported ones first and let the hash decide among equals,
            // so the hint only reorders a choice that was arbitrary.
            truncated_any = true;
            match (hints, cfg.mate_window) {
                (Some(h), Some(window)) => {
                    ctx.occs
                        .select_nth_unstable_by_key(take_n, |(rid, pos, strand)| {
                            (
                                !h.supports(*rid, *pos, window),
                                occurrence_sample_key(m.hash, *rid, *pos, *strand),
                            )
                        });
                }
                _ => {
                    ctx.occs
                        .select_nth_unstable_by_key(take_n, |(rid, pos, strand)| {
                            occurrence_sample_key(m.hash, *rid, *pos, *strand)
                        });
                }
            }
        }
        for occurrence in 0..take_n {
            // Small buckets need neither sampling nor a materialized copy.
            // Read the contiguous postings directly, avoiding a hot-cache hash
            // probe and an intermediate Vec write for every unique minimizer.
            let (rid, pos, ref_strand) = if bucket_len <= k_hits {
                let o = index.occ_at(table, start + occurrence);
                (o.ref_id, o.pos, o.strand)
            } else {
                ctx.occs[occurrence]
            };
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

    let (anchors, before) = finish_anchors(read, index, sketch.k, cfg, ctx);
    let min_occ = if min_occ == u32::MAX { 0 } else { min_occ };
    (anchors, before, min_occ, truncated_any)
}

fn finish_anchors(
    read: &ReadRecord,
    index: &Index,
    k: usize,
    cfg: crate::seeding::SeedingConfig,
    ctx: &mut ThreadCtx,
) -> (Vec<Anchor>, usize) {
    merge_exact_seed_runs(&mut ctx.proto, &mut ctx.merged);
    let before = ctx.merged.len();
    ctx.anchors_before_prune = before;

    let read_len = read.seq.len() as u32;
    let read_len_salt = read.seq.len() as u64;
    for proto in ctx.merged.drain(..) {
        let score = proto
            .read_end
            .saturating_sub(proto.read_start)
            .min(proto.ref_end.saturating_sub(proto.ref_start)) as i32;
        let tiebreak =
            occurrence_sample_key(read_len_salt, proto.ref_id, proto.ref_start, proto.strand);
        ctx.candidates.push(AnchorCandidate {
            proto,
            score,
            tiebreak,
        });
    }

    // An anchor is never shorter than the seed it grew from, so a requirement
    // above `k` discards every seed whose exact match stops at exactly `k` —
    // a mismatch on both flanks, the defining shape of a seed in a divergent
    // region. The short preset ships min_anchor_len=20 against k=19, so cap the
    // requirement at the k in use. `KIRA_ANCHOR_CAP_K=0` restores the old rule.
    let min_anchor_len = if anchor_cap_k() {
        cfg.min_anchor_len.min(k as u32)
    } else {
        cfg.min_anchor_len
    };
    let max_anchors_per_read = anchor_limit(read.seq.len(), k);
    let max_anchors_per_diag = if read.seq.len() >= cfg.long_read_threshold {
        64
    } else {
        16
    };

    // Unstable sort is safe (and allocation-free) because the key is a total
    // order on candidate content: after the hash tiebreak the coordinate fields
    // separate everything but identical candidates, so output stays reproducible.
    ctx.candidates.sort_unstable_by_key(|c| {
        (
            std::cmp::Reverse(c.score),
            c.tiebreak,
            c.proto.ref_id,
            c.proto.ref_start,
            c.proto.read_start,
            u8::from(c.proto.strand),
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
        if anchor.read_end.saturating_sub(anchor.read_start) >= min_anchor_len
            && anchor.read_end <= read_len
        {
            ctx.anchors.push(anchor);
        }
    }
    sort_dedup_anchors(&mut ctx.anchors);
    (std::mem::take(&mut ctx.anchors), before)
}

fn sort_dedup_anchors(anchors: &mut Vec<Anchor>) {
    anchors.sort_unstable_by_key(|a| {
        (
            a.ref_id,
            u8::from(a.strand),
            a.ref_start,
            a.read_start,
            a.ref_end,
            a.read_end,
        )
    });
    anchors.dedup_by(|a, b| {
        a.ref_id == b.ref_id
            && a.strand == b.strand
            && a.ref_start == b.ref_start
            && a.ref_end == b.ref_end
            && a.read_start == b.read_start
            && a.read_end == b.read_end
    });
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
    input.sort_unstable_by_key(|p| {
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

/// Whether `min_anchor_len` is capped at the seed length `k` (see `seed_one`).
/// Default on; `KIRA_ANCHOR_CAP_K=0` restores the pre-fix behaviour.
fn anchor_cap_k() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("KIRA_ANCHOR_CAP_K")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
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

    // Extend left through the matching suffix of everything before the seed,
    // right through the matching prefix of everything after it.
    let left =
        common_suffix_len(&read_seq[..q_start as usize], &ref_seq[..r_start as usize]) as i32;
    q_start -= left;
    r_start -= left;

    let right = common_prefix_len(
        &read_seq[(q_end as usize).min(read_seq.len())..],
        &ref_seq[(r_end as usize).min(ref_seq.len())..],
    ) as i32;
    q_end += right;
    r_end += right;

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
    sketches: [ReadSketch; 2],
    coverage: Vec<(u32, u32)>,
    adaptive_retries: usize,
    repeat_rescues: usize,
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
    /// Mate-hint bin sets, reused across templates.
    hints_a: MateHints,
    hints_b: MateHints,
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
mod mate_hint_tests {
    use super::*;

    fn anchor_at(ref_id: u32, ref_start: u32) -> Anchor {
        Anchor {
            read_start: 0,
            read_end: 19,
            ref_id,
            ref_start,
            ref_end: ref_start + 19,
            strand: Strand::Forward,
            score: 19,
        }
    }

    #[test]
    fn supports_covers_the_window_and_nothing_beyond() {
        let mut h = MateHints::default();
        h.build_from(&[anchor_at(0, 10_000)]);
        // Inside the window on both sides.
        assert!(h.supports(0, 10_000, 500));
        assert!(h.supports(0, 10_400, 500));
        assert!(h.supports(0, 9_600, 500));
        // A different contig is never support, however close the coordinate.
        assert!(!h.supports(1, 10_000, 500));
        // Far away on the same contig: the classic dispersed-repeat copy.
        assert!(!h.supports(0, 5_000_000, 500));
    }

    #[test]
    fn window_is_honoured_at_bin_granularity() {
        let mut h = MateHints::default();
        h.build_from(&[anchor_at(0, 0)]);
        // Bins are HINT_BIN wide, so support extends to the end of the bin the
        // window reaches into — coarser than the window, never narrower.
        assert!(h.supports(0, HINT_BIN - 1, 0));
        assert!(h.supports(0, HINT_BIN * 3, HINT_BIN * 3));
        assert!(!h.supports(0, HINT_BIN * 10, HINT_BIN));
    }

    #[test]
    fn scattered_mate_is_not_usable() {
        let mut h = MateHints::default();
        // A mate concentrated in a few places is trustworthy.
        let focused: Vec<Anchor> = (0..HINT_MAX_BINS as u32)
            .map(|i| anchor_at(0, i * HINT_BIN))
            .collect();
        h.build_from(&focused);
        assert!(
            h.is_usable(),
            "a mate in {HINT_MAX_BINS} bins should be usable"
        );

        // One bin more and it carries no positional information.
        let scattered: Vec<Anchor> = (0..HINT_MAX_BINS as u32 + 1)
            .map(|i| anchor_at(0, i * HINT_BIN * 100))
            .collect();
        h.build_from(&scattered);
        assert!(!h.is_usable(), "a scattered mate must be ignored");
    }

    #[test]
    fn unmapped_mate_yields_no_hints() {
        let mut h = MateHints::default();
        h.build_from(&[]);
        assert!(!h.is_usable());
        assert!(!h.supports(0, 0, 1000));
    }

    #[test]
    fn hints_are_rebuilt_not_accumulated() {
        let mut h = MateHints::default();
        h.build_from(&[anchor_at(0, 1_000)]);
        h.build_from(&[anchor_at(0, 9_000_000)]);
        // Reusing the set across templates must not leave the previous mate's
        // positions behind, or a read is hinted toward its neighbour's locus.
        assert!(!h.supports(0, 1_000, 500));
        assert!(h.supports(0, 9_000_000, 500));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(
        copies: usize,
        cap: usize,
    ) -> (
        Index,
        ReadRecord,
        SketchConfig,
        crate::seeding::SeedingConfig,
    ) {
        let mut state = 0x1234_5678_9abc_def0u64;
        let seq: Vec<u8> = (0..300)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                b"ACGT"[(state >> 32) as usize % 4]
            })
            .collect();
        let reference = crate::types::Reference {
            sequences: (0..copies)
                .map(|i| crate::types::RefSeq {
                    name: format!("copy{i}"),
                    bases: crate::types::RefBases::Owned(seq.clone()),
                })
                .collect(),
        };
        let index = Index::build(
            reference,
            crate::index::IndexConfig {
                short_k: 15,
                short_w: 5,
                long_k: 15,
                long_w: 5,
                max_occ: cap,
                build_short: true,
                build_long: true,
            },
        );
        let read = ReadRecord::new_unpaired("read".into(), seq[50..200].to_vec(), None);
        let sketch = SketchConfig {
            short_k: 15,
            short_w: 5,
            long_k: 15,
            long_w: 5,
            long_read_threshold: 1000,
        };
        let seeding = crate::seeding::SeedingConfig {
            min_anchor_len: 15,
            max_occ: 2,
            max_hits_per_minimizer: 2,
            long_read_threshold: 1000,
            mate_window: None,
        };
        (index, read, sketch, seeding)
    }

    #[test]
    fn fused_sketch_seeding_matches_staged_across_blocks_masks_and_odd_tail() {
        let (index, read, cfg, seed_cfg) = fixture(1, 100);
        let input = InputBatch {
            reads: vec![read; 259],
        };
        let mask: Vec<bool> = (0..259).map(|i| i % 7 == 0).collect();
        let staged_sketch = super::super::stage1_sketch::run_with_mask(input.clone(), cfg, &mask);
        let expected_stats = staged_sketch.stats;
        let staged = run(staged_sketch, &index, seed_cfg);
        for _ in 0..2 {
            // buffers reused on the second batch
            let (fused, stats) = run_fused(input.clone(), &index, cfg, seed_cfg, &mask);
            assert_eq!(stats.avg_minimizers, expected_stats.avg_minimizers);
            assert_eq!(stats.read_len_p50, expected_stats.read_len_p50);
            assert_eq!(fused.reads.len(), 259);
            for (i, (a, b)) in fused.anchors.iter().zip(&staged.anchors).enumerate() {
                let coords = |xs: &[Anchor]| {
                    xs.iter()
                        .map(|a| {
                            (
                                a.ref_id,
                                u8::from(a.strand),
                                a.ref_start,
                                a.ref_end,
                                a.read_start,
                                a.read_end,
                                a.score,
                            )
                        })
                        .collect::<Vec<_>>()
                };
                assert_eq!(coords(a), coords(b), "read {i}");
                assert_eq!(a.is_empty(), mask[i]);
                assert_eq!(
                    fused.reads[i].repeat_min_occ,
                    staged.reads[i].repeat_min_occ
                );
            }
        }
    }

    #[test]
    fn adaptive_retry_and_longer_seeds_recover_complete_and_truncated_repeats() {
        for build_cap in [100, 2] {
            let (index, read, cfg, seed_cfg) = fixture(6, build_cap);
            let sketch = ReadSketch {
                minimizers: crate::sketch::minimizers(
                    &read.seq,
                    &crate::sketch::MinimizerConfig {
                        k: cfg.short_k,
                        w: cfg.short_w,
                    },
                ),
                k: cfg.short_k,
                w: cfg.short_w,
            };
            let mut ctx = ThreadCtx::default();
            let (initial, _, _, _) = seed_one(&read, &sketch, &index, seed_cfg, &mut ctx, None);
            assert!(initial.is_empty());
            let (adaptive, _, min_occ, _) =
                seed_adaptive(&read, &sketch, &index, seed_cfg, &mut ctx, None);
            assert_eq!(adaptive.len(), 6, "build cap {build_cap}");
            assert_eq!(min_occ, 6, "true repeat count, build cap {build_cap}");
            assert!(ctx.adaptive_retries > 0);
            assert!(
                adaptive.iter().any(|a| a.ref_id == 5),
                "rescue must recover occurrences omitted from primary postings"
            );
            if build_cap == 2 {
                assert_eq!(ctx.repeat_rescues, 1);
            }
        }
    }

    /// The run-time cap never exceeds the build cap: an index built with
    /// --max-occ 100 stores over-cap buckets truncated to 101 records, which
    /// a seeder capped at 500 must not read as complete buckets.
    #[test]
    fn effective_cap_is_bounded_by_the_index_cap() {
        assert_eq!(effective_max_occ(500, 100), 100);
        assert_eq!(effective_max_occ(32, 500), 32);
        assert_eq!(effective_max_occ(500, 500), 500);
        // The sentinel length (index cap + 1) is always over the cap.
        assert!(101 > effective_max_occ(500, 100));
    }

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
