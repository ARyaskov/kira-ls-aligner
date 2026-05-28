use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use crate::alignment::prefilter::{
    PrefilterOutcome, PrefilterReason, PrefilterResult, UngappedStats, build_ungapped_alignment,
    min_len_required, prefilter_chain,
};
use crate::alignment::{
    AlignmentConfig, AnchorSpan, BatchInput, FastPathKind, align_batch_simd, align_chain_with_meta,
    exact_match_alignment, try_fast_dp_alignment,
};
#[cfg(feature = "cuda")]
use crate::alignment::push_u32_decimal;
#[cfg(feature = "cuda")]
use crate::types::AlignmentKind;
use crate::index::Index;
use crate::seq::reverse_complement;
use crate::simd::{self, SimdMode};
use crate::types::{Alignment, Chain, CigarKind, CigarOp, MateInfo, ReadRecord, Strand};

use super::stage3_chaining::ChainBatch;

/// Stage 4 output: alignments per read.
#[derive(Clone, Debug)]
pub struct AlignBatch {
    pub reads: Vec<ReadRecord>,
    pub alignments: Vec<Vec<Alignment>>,
    pub unmapped_mate_info: Vec<Option<crate::types::MateInfo>>,
    pub stats: AlignmentBatchStats,
}

/// Alignment stage configuration.
#[derive(Clone, Copy, Debug)]
pub struct AlignmentStageConfig {
    pub cfg: AlignmentConfig,
    pub min_chain_ratio: f32,
    pub accept_enable: bool,
    pub accept_only_top1: bool,
    pub accept_span_slack: usize,
    pub accept_min_identity: f32,
    pub accept_max_mismatches: usize,
    pub accept_require_score_margin: i32,
    pub dp_topk: usize,
    pub dp_abort_margin: i32,
    pub debug_prefilter: bool,
    pub debug_prefilter_n: usize,
    pub debug_force_accept: bool,
    pub debug_force_accept_n: usize,
    pub long_read_threshold: usize,
    pub max_alignments: usize,
    pub short_preset: bool,
}

/// Per-batch alignment stats.
#[derive(Clone, Debug, Default)]
pub struct AlignmentBatchStats {
    pub reads: usize,
    pub chains_total: usize,
    pub chains_used: usize,
    pub dp_attempts: usize,
    pub dp_simd: usize,
    pub dp_scalar: usize,
    pub dp_reads: usize,
    pub dp_topk: usize,
    pub dp_abort_margin: i32,
    pub dp_early_abort: usize,
    pub exact_matches: usize,
    pub prefilter_accept: usize,
    pub prefilter_reject: usize,
    pub prefilter_fallback: usize,
    pub prefilter_reason_counts: [usize; PrefilterReason::COUNT],
    pub ungapped_score_p95: i32,
    pub ungapped_span_p95: usize,
    pub ungapped_mismatches_p95: usize,
    pub ungapped_identity_p90: f32,
    pub accept_len_p50: usize,
    pub accept_len_p95: usize,
    pub fallback_len_p50: usize,
    pub fallback_len_p95: usize,
    pub bucket_counts: [usize; 3],
    pub sum_read_len: usize,
    /// Reads resolved by the Bit-Packed Spectral fast path (CPU).
    pub packed_spectral_resolved: usize,
    /// Reads resolved by the byte Spectral Sieve fast path.
    pub spectral_sieve_resolved: usize,
    /// Reads resolved by the WFA fast path.
    pub wfa_resolved: usize,
    /// Reads resolved by CGK rescue.
    pub cgk_rescue_resolved: usize,
    /// Reads resolved by the CUDA spectral dispatcher.
    pub gpu_spectral_resolved: usize,
    /// Reads that finished stage 4 without an alignment.
    pub unmapped: usize,
}

impl AlignmentBatchStats {
    pub fn add(&mut self, other: &AlignmentBatchStats) {
        self.reads += other.reads;
        self.chains_total += other.chains_total;
        self.chains_used += other.chains_used;
        self.dp_attempts += other.dp_attempts;
        self.dp_simd += other.dp_simd;
        self.dp_scalar += other.dp_scalar;
        self.dp_reads += other.dp_reads;
        self.dp_topk = other.dp_topk;
        self.dp_abort_margin = other.dp_abort_margin;
        self.dp_early_abort += other.dp_early_abort;
        self.exact_matches += other.exact_matches;
        self.prefilter_accept += other.prefilter_accept;
        self.prefilter_reject += other.prefilter_reject;
        self.prefilter_fallback += other.prefilter_fallback;
        for (dst, src) in self
            .prefilter_reason_counts
            .iter_mut()
            .zip(other.prefilter_reason_counts.iter())
        {
            *dst += *src;
        }
        self.bucket_counts[0] += other.bucket_counts[0];
        self.bucket_counts[1] += other.bucket_counts[1];
        self.bucket_counts[2] += other.bucket_counts[2];
        self.sum_read_len += other.sum_read_len;
        self.packed_spectral_resolved += other.packed_spectral_resolved;
        self.spectral_sieve_resolved += other.spectral_sieve_resolved;
        self.wfa_resolved += other.wfa_resolved;
        self.cgk_rescue_resolved += other.cgk_rescue_resolved;
        self.gpu_spectral_resolved += other.gpu_spectral_resolved;
        self.unmapped += other.unmapped;
        self.ungapped_score_p95 = other.ungapped_score_p95;
        self.ungapped_span_p95 = other.ungapped_span_p95;
        self.ungapped_mismatches_p95 = other.ungapped_mismatches_p95;
        self.ungapped_identity_p90 = other.ungapped_identity_p90;
        self.accept_len_p50 = other.accept_len_p50;
        self.accept_len_p95 = other.accept_len_p95;
        self.fallback_len_p50 = other.fallback_len_p50;
        self.fallback_len_p95 = other.fallback_len_p95;
    }

    pub fn avg_read_len(&self) -> f32 {
        if self.reads == 0 {
            0.0
        } else {
            self.sum_read_len as f32 / self.reads as f32
        }
    }
}

struct SimdJob<'a> {
    read_idx: usize,
    read_seq: Vec<u8>,
    ref_window: &'a [u8],
    win_start: u32,
    chain: AnchorSpan,
    is_rev: bool,
    abort_score: i32,
}

struct ScalarJob {
    read_idx: usize,
    chain: AnchorSpan,
    abort_score: i32,
}

// Confidence blends coverage, diagonal consistency, ungapped score and chain margin.
fn chain_confidence(
    chain: &Chain,
    read_len: usize,
    score_margin: i32,
    ungapped: Option<&UngappedStats>,
    cfg: AlignmentConfig,
) -> f32 {
    let mut cov = 0usize;
    for a in chain.anchors.iter() {
        cov += (a.read_end - a.read_start) as usize;
    }
    let coverage = (cov as f32 / read_len.max(1) as f32).min(1.0);

    let mut min_diag = i32::MAX;
    let mut max_diag = i32::MIN;
    for a in chain.anchors.iter() {
        let d = a.ref_start as i32 - a.read_start as i32;
        min_diag = min_diag.min(d);
        max_diag = max_diag.max(d);
    }
    let diag_span = (max_diag - min_diag).max(0) as f32;
    let diag_score = (1.0 - (diag_span / (read_len as f32 * 0.2 + 1.0))).clamp(0.0, 1.0);

    let margin_score = (score_margin as f32 / 40.0).clamp(0.0, 1.0);
    let ungapped_norm = ungapped
        .map(|m| {
            (m.score as f32 / (read_len as f32 * cfg.match_score.max(1) as f32)).clamp(0.0, 1.0)
        })
        .unwrap_or(0.0);

    0.4 * coverage + 0.2 * diag_score + 0.2 * ungapped_norm + 0.2 * margin_score
}

struct PerReadResult<'a> {
    accepted: Vec<Alignment>,
    simd_jobs: Vec<SimdJob<'a>>,
    scalar_jobs: Vec<ScalarJob>,
    dp_used: bool,
    short_read: bool,
    bucket: usize,
    sum_read_len: usize,
    chains_total: usize,
    chains_used: usize,
    exact_matches: usize,
    prefilter_accept: usize,
    prefilter_reject: usize,
    prefilter_fallback: usize,
    prefilter_reason_counts: [usize; PrefilterReason::COUNT],
    ungapped_stats: Vec<UngappedStats>,
    accept_lens: Vec<u16>,
    fallback_lens: Vec<u16>,
    potential_accepts: usize,
    /// Top chain score, used as denominator in the chain-ratio MAPQ proxy.
    chain_best: i32,
    /// Second-best chain score.
    chain_second: i32,
    /// Per-read attribution counters — set when a fast-DP cascade step produces the primary.
    packed_spectral_resolved: usize,
    spectral_sieve_resolved: usize,
    wfa_resolved: usize,
    cgk_rescue_resolved: usize,
}

impl<'a> Default for PerReadResult<'a> {
    fn default() -> Self {
        Self {
            accepted: Vec::new(),
            simd_jobs: Vec::new(),
            scalar_jobs: Vec::new(),
            dp_used: false,
            short_read: false,
            bucket: 0,
            sum_read_len: 0,
            chains_total: 0,
            chains_used: 0,
            exact_matches: 0,
            prefilter_accept: 0,
            prefilter_reject: 0,
            prefilter_fallback: 0,
            prefilter_reason_counts: [0; PrefilterReason::COUNT],
            ungapped_stats: Vec::new(),
            accept_lens: Vec::new(),
            fallback_lens: Vec::new(),
            potential_accepts: 0,
            chain_best: 0,
            chain_second: 0,
            packed_spectral_resolved: 0,
            spectral_sieve_resolved: 0,
            wfa_resolved: 0,
            cgk_rescue_resolved: 0,
        }
    }
}

fn process_read_prefilter<'a>(
    idx: usize,
    read: &ReadRecord,
    chain_list: &[Chain],
    index: &'a Index,
    cfg: AlignmentStageConfig,
    lanes: usize,
    force_counter: &AtomicUsize,
    debug_counter: &AtomicUsize,
    accept_allowed: bool,
    multi_alignments_enabled: bool,
) -> PerReadResult<'a> {
    let mut res = PerReadResult::default();
    let len = read.seq.len();
    let short_read = len <= 300;
    let read_fwd = read.seq.as_slice();
    let mut read_rev_cache: Option<Vec<u8>> = None;
    res.short_read = short_read;
    res.sum_read_len = len;
    res.bucket = if len <= 120 {
        0
    } else if len <= 150 {
        1
    } else {
        2
    };
    res.chains_total = chain_list.len();
    if chain_list.is_empty() {
        return res;
    }

    let best = chain_list[0].score;
    let min_score = if best > 0 {
        (best as f32 * cfg.min_chain_ratio).ceil() as i32
    } else {
        best
    };
    let chain_second = chain_list.get(1).map(|c| c.score).unwrap_or(0);
    let score_margin = if chain_list.len() > 1 {
        best - chain_second
    } else {
        i32::MAX
    };
    res.chain_best = best;
    res.chain_second = chain_second;

    let effective_dp_topk = if short_read { 1 } else { cfg.dp_topk.max(1) };
    let mut accepted_read = false;
    let mut selected = 0usize;
    for chain in chain_list.iter() {
        if chain.score < min_score {
            continue;
        }
        if selected >= effective_dp_topk {
            break;
        }
        let is_top1 = selected == 0;
        selected += 1;
        let chain_rank = selected - 1;
        res.chains_used += 1;

        let ref_seq = index.ref_bases(chain.ref_id as usize);
        let span = AnchorSpan {
            ref_id: chain.ref_id,
            ref_start: chain.ref_start,
            ref_end: chain.ref_end,
            read_start: chain.read_start,
            read_end: chain.read_end,
            strand: chain.strand,
        };
        let is_rev = matches!(span.strand, Strand::Reverse);
        let read_seq = if is_rev {
            read_rev_cache
                .get_or_insert_with(|| reverse_complement(read_fwd))
                .as_slice()
        } else {
            read_fwd
        };
        let read_len = read_seq.len();

        if let Some(aln) =
            exact_match_alignment(read_len, read_seq, ref_seq, &span, cfg.cfg, is_rev)
        {
            res.exact_matches += 1;
            res.accepted.push(aln);
            continue;
        }

        let forced = cfg.debug_force_accept
            && is_top1
            && force_counter.load(Ordering::Relaxed) < cfg.debug_force_accept_n;
        if forced {
            force_counter.fetch_add(1, Ordering::Relaxed);
        }

        let PrefilterOutcome {
            result,
            metrics,
            reason,
        } = prefilter_chain(
            read_seq,
            ref_seq,
            &span,
            cfg.cfg,
            is_top1,
            accept_allowed,
            cfg.accept_only_top1,
            cfg.accept_span_slack,
            cfg.accept_min_identity,
            cfg.accept_max_mismatches,
            cfg.accept_require_score_margin,
            score_margin,
            cfg.long_read_threshold,
            short_read,
            multi_alignments_enabled,
        );

        if let Some(m) = metrics {
            res.ungapped_stats.push(m);
            if cfg.short_preset && cfg.max_alignments == 1 && is_top1 {
                let min_len = min_len_required(
                    read_len,
                    m.identity_x10000,
                    m.mism as usize,
                    cfg.accept_span_slack,
                );
                if (m.len as usize) >= min_len
                    && (m.mism as usize) <= cfg.accept_max_mismatches
                    && (m.identity_x10000 as u32)
                        >= (cfg.accept_min_identity * 100.0).round() as u32
                {
                    res.potential_accepts += 1;
                }
            }
        }
        let abort_score = metrics
            .map(|m| m.score.saturating_sub(cfg.dp_abort_margin))
            .unwrap_or(i32::MIN / 4);

        res.prefilter_reason_counts[reason.idx()] += 1;

        if cfg.debug_prefilter
            && is_top1
            && debug_counter.load(Ordering::Relaxed) < cfg.debug_prefilter_n
        {
            debug_counter.fetch_add(1, Ordering::Relaxed);
            if let Some(m) = metrics {
                eprintln!(
                    "[KIRA_DEBUG_PREFILTER] read_id={} read_len={} ungapped_len={} mism={} matches={} identity_x100={} score={} chain_rank={} q={}..{} r={}..{} decision={:?} reason={}",
                    read.id,
                    read_len,
                    m.len,
                    m.mism,
                    m.matches,
                    ((m.identity_x10000 as f64) / 100.0),
                    m.score,
                    chain_rank,
                    span.read_start,
                    span.read_end,
                    span.ref_start,
                    span.ref_end,
                    result,
                    reason.as_str()
                );
            } else {
                eprintln!(
                    "[KIRA_DEBUG_PREFILTER] read_id={} read_len={} ungapped_len=NA mism=NA matches=NA identity=NA score=NA chain_rank={} q={}..{} r={}..{} decision={:?} reason={}",
                    read.id,
                    read_len,
                    chain_rank,
                    span.read_start,
                    span.read_end,
                    span.ref_start,
                    span.ref_end,
                    result,
                    reason.as_str()
                );
            }
        }

        let confidence = if is_top1 && cfg.max_alignments <= 1 {
            chain_confidence(chain, read_len, score_margin, metrics.as_ref(), cfg.cfg)
        } else {
            0.0
        };

        let final_result = if matches!(result, PrefilterResult::Fallback)
            && confidence >= if short_read { 0.65 } else { 0.85 }
            && metrics.is_some()
            && is_top1
            && short_read
            && cfg.max_alignments <= 1
        {
            let m = metrics.as_ref().unwrap();
            let aln = build_ungapped_alignment(read_seq, ref_seq, m, &span, cfg.cfg);
            res.prefilter_reason_counts[PrefilterReason::Accepted.idx()] += 1;
            PrefilterResult::Accept(aln)
        } else if forced {
            PrefilterResult::Accept(build_forced_accept(read_seq, ref_seq, &span, cfg.cfg))
        } else {
            result
        };

        match final_result {
            PrefilterResult::Accept(aln) => {
                res.prefilter_accept += 1;
                if let Some(m) = metrics {
                    res.accept_lens.push(m.len);
                }
                res.accepted.push(aln);
                accepted_read = true;
            }
            PrefilterResult::Reject => {
                res.prefilter_reject += 1;
                continue;
            }
            PrefilterResult::Fallback => {
                res.prefilter_fallback += 1;
                if let Some(m) = metrics {
                    res.fallback_lens.push(m.len);
                }
            }
        }

        if accepted_read {
            break;
        }

        #[cfg(feature = "cuda")]
        let cpu_fast_path_enabled = !crate::cuda::dispatcher::is_active();
        #[cfg(not(feature = "cuda"))]
        let cpu_fast_path_enabled = true;
        if cpu_fast_path_enabled {
            if let Some((aln, fast_kind)) =
                try_fast_dp_alignment(read_seq, ref_seq, &span, chain.score, cfg.cfg, is_rev)
            {
                res.accepted.push(aln);
                res.dp_used = true;
                match fast_kind {
                    FastPathKind::PackedSpectral => res.packed_spectral_resolved += 1,
                    FastPathKind::SpectralSieve => res.spectral_sieve_resolved += 1,
                    FastPathKind::Wfa => res.wfa_resolved += 1,
                    FastPathKind::CgkRescue => res.cgk_rescue_resolved += 1,
                }
                break;
            }
        }

        let use_simd = res.bucket < 2 && lanes > 1;
        if use_simd {
            if let Some((win_start, ref_window)) =
                build_simd_window(ref_seq, chain, read_len, cfg.cfg)
            {
                res.simd_jobs.push(SimdJob {
                    read_idx: idx,
                    read_seq: read_seq.to_vec(),
                    ref_window,
                    win_start,
                    chain: span,
                    is_rev,
                    abort_score,
                });
                res.dp_used = true;
                continue;
            }
        }
        res.scalar_jobs.push(ScalarJob {
            read_idx: idx,
            chain: span,
            abort_score,
        });
        res.dp_used = true;
    }

    res
}

pub fn run(input: ChainBatch, index: &Index, cfg: AlignmentStageConfig) -> AlignBatch {
    let reads = input.reads;
    let chains = input.chains;

    let simd_mode = simd::detect_cached();
    let lanes = match simd_mode {
        SimdMode::Avx2 => 16,
        SimdMode::Neon => 4,
        SimdMode::Scalar => 1,
    };

    let mut stats = AlignmentBatchStats::default();
    stats.reads = reads.len();
    stats.dp_topk = cfg.dp_topk.max(1);
    stats.dp_abort_margin = cfg.dp_abort_margin;

    let force_counter = AtomicUsize::new(0);
    let debug_counter = AtomicUsize::new(0);

    let accept_allowed = cfg.accept_enable && cfg.max_alignments <= 1;
    let multi_alignments_enabled = cfg.max_alignments > 1;

    let per_read: Vec<PerReadResult<'_>> = reads
        .par_iter()
        .enumerate()
        .map(|(idx, read)| {
            process_read_prefilter(
                idx,
                read,
                &chains[idx],
                index,
                cfg,
                lanes,
                &force_counter,
                &debug_counter,
                accept_allowed,
                multi_alignments_enabled,
            )
        })
        .collect();

    // Sequential merge of per-read results (cheap; ~1.5M reads but tiny per-read work).
    let mut short_read_batch = false;
    let mut alignments: Vec<Vec<Alignment>> = Vec::with_capacity(reads.len());
    let mut simd_jobs: Vec<SimdJob<'_>> = Vec::new();
    let mut scalar_jobs: Vec<ScalarJob> = Vec::new();
    let mut bucket_map: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut dp_used = vec![false; reads.len()];
    let mut potential_accepts: usize = 0;
    let mut ungapped_stats: Vec<UngappedStats> = Vec::new();
    let mut accept_lens: Vec<u16> = Vec::new();
    let mut fallback_lens: Vec<u16> = Vec::new();
    let mut chain_best_per_read: Vec<i32> = Vec::with_capacity(reads.len());
    let mut chain_second_per_read: Vec<i32> = Vec::with_capacity(reads.len());

    for (idx, mut res) in per_read.into_iter().enumerate() {
        if res.short_read {
            short_read_batch = true;
        }
        stats.sum_read_len += res.sum_read_len;
        stats.bucket_counts[res.bucket] += 1;
        stats.chains_total += res.chains_total;
        stats.chains_used += res.chains_used;
        stats.exact_matches += res.exact_matches;
        stats.prefilter_accept += res.prefilter_accept;
        stats.prefilter_reject += res.prefilter_reject;
        stats.prefilter_fallback += res.prefilter_fallback;
        for (dst, src) in stats
            .prefilter_reason_counts
            .iter_mut()
            .zip(res.prefilter_reason_counts.iter())
        {
            *dst += *src;
        }
        potential_accepts += res.potential_accepts;
        ungapped_stats.append(&mut res.ungapped_stats);
        accept_lens.append(&mut res.accept_lens);
        fallback_lens.append(&mut res.fallback_lens);
        stats.packed_spectral_resolved += res.packed_spectral_resolved;
        stats.spectral_sieve_resolved += res.spectral_sieve_resolved;
        stats.wfa_resolved += res.wfa_resolved;
        stats.cgk_rescue_resolved += res.cgk_rescue_resolved;
        alignments.push(std::mem::take(&mut res.accepted));
        chain_best_per_read.push(res.chain_best);
        chain_second_per_read.push(res.chain_second);
        dp_used[idx] = res.dp_used;
        for job in res.simd_jobs.into_iter() {
            let read_len = job.read_seq.len();
            let job_idx = simd_jobs.len();
            bucket_map.entry(read_len).or_default().push(job_idx);
            simd_jobs.push(job);
        }
        scalar_jobs.extend(res.scalar_jobs);
    }

    #[cfg(feature = "cuda")]
    {
        if crate::cuda::dispatcher::is_active() && !simd_jobs.is_empty() {
            let t0 = std::time::Instant::now();
            let accepted = gpu_dispatch_spectral(&simd_jobs, &mut alignments, cfg.cfg);
            let dispatch_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let n_dispatched = accepted.len();
            let n_accepted = accepted.iter().filter(|a| **a).count();
            stats.gpu_spectral_resolved += n_accepted;
            if n_accepted > 0 {
                // Drop GPU-accepted jobs and rebuild the bucket map.
                let mut idx = 0usize;
                simd_jobs.retain(|_| {
                    let keep = !accepted[idx];
                    idx += 1;
                    keep
                });
                bucket_map.clear();
                for (new_idx, job) in simd_jobs.iter().enumerate() {
                    bucket_map.entry(job.read_seq.len()).or_default().push(new_idx);
                }
            }
            eprintln!(
                "[KIRA_GPU] dispatched {} jobs, accepted {} ({:.2} ms = {:.1} jobs/ms)",
                n_dispatched,
                n_accepted,
                dispatch_ms,
                n_dispatched as f64 / dispatch_ms.max(0.001),
            );
        }
    }

    let mut scores: Vec<i32> = ungapped_stats.iter().map(|m| m.score).collect();
    let mut lens: Vec<u16> = ungapped_stats.iter().map(|m| m.len).collect();
    let mut mism: Vec<u16> = ungapped_stats.iter().map(|m| m.mism).collect();
    let mut ids: Vec<u16> = ungapped_stats.iter().map(|m| m.identity_x10000).collect();

    stats.ungapped_score_p95 = percentile_i32(&mut scores, 95);
    stats.ungapped_span_p95 = percentile_u16(&mut lens, 95) as usize;
    stats.ungapped_mismatches_p95 = percentile_u16(&mut mism, 95) as usize;
    let id_x100 = percentile_u16(&mut ids, 90);
    stats.ungapped_identity_p90 = (id_x100 as f32) / 100.0;
    accept_lens.sort_unstable();
    fallback_lens.sort_unstable();
    stats.accept_len_p50 = percentile_u16_sorted(&accept_lens, 50) as usize;
    stats.accept_len_p95 = percentile_u16_sorted(&accept_lens, 95) as usize;
    stats.fallback_len_p50 = percentile_u16_sorted(&fallback_lens, 50) as usize;
    stats.fallback_len_p95 = percentile_u16_sorted(&fallback_lens, 95) as usize;

    stats.dp_reads = dp_used.iter().filter(|v| **v).count();
    if short_read_batch {
        stats.dp_topk = 1;
        debug_assert!(stats.dp_attempts <= stats.reads);
    }

    if cfg.debug_prefilter
        && cfg.short_preset
        && cfg.max_alignments == 1
        && potential_accepts > 0
        && stats.prefilter_accept == 0
    {
        eprintln!(
            "[KIRA_DEBUG_PREFILTER] warning: potential_accepts={} but accept_count=0",
            potential_accepts
        );
    }

    let mut simd_batches: Vec<Vec<usize>> = Vec::new();
    let mut simd_fallback: Vec<usize> = Vec::new();
    for indices in bucket_map.values() {
        for chunk in indices.chunks(lanes) {
            if chunk.len() == lanes {
                simd_batches.push(chunk.to_vec());
            } else {
                simd_fallback.extend_from_slice(chunk);
            }
        }
    }

    let simd_results: Vec<(usize, Alignment, bool)> = simd_batches
        .par_iter()
        .flat_map(|batch| {
            let inputs: Vec<BatchInput<'_>> = batch
                .iter()
                .map(|&idx| {
                    let job = &simd_jobs[idx];
                    BatchInput {
                        read_seq: job.read_seq.as_slice(),
                        ref_window: job.ref_window,
                        win_start: job.win_start,
                        chain: job.chain,
                        is_rev: job.is_rev,
                        abort_score: job.abort_score,
                    }
                })
                .collect();
            let alns = align_batch_simd(&inputs, cfg.cfg, simd_mode);
            batch
                .iter()
                .zip(alns.into_iter())
                .map(|(&idx, (aln, early))| (simd_jobs[idx].read_idx, aln, early))
                .collect::<Vec<_>>()
        })
        .collect();

    stats.dp_simd += simd_results.len();
    stats.dp_attempts += simd_results.len();
    stats.dp_early_abort += simd_results.iter().filter(|(_, _, early)| *early).count();

    for (idx, aln, _) in simd_results {
        alignments[idx].push(aln);
    }

    for idx in simd_fallback {
        let job = &simd_jobs[idx];
        scalar_jobs.push(ScalarJob {
            read_idx: job.read_idx,
            chain: job.chain,
            abort_score: job.abort_score,
        });
    }

    let scalar_results: Vec<(usize, Alignment, bool)> = scalar_jobs
        .par_iter()
        .map(|job| {
            let read = &reads[job.read_idx];
            let ref_seq = index.ref_bases(job.chain.ref_id as usize);
            let (aln, early) =
                align_chain_with_meta(read, ref_seq, &job.chain, cfg.cfg, job.abort_score);
            (job.read_idx, aln, early)
        })
        .collect();

    stats.dp_scalar += scalar_results.len();
    stats.dp_attempts += scalar_results.len();
    stats.dp_early_abort += scalar_results.iter().filter(|(_, _, early)| *early).count();

    for (idx, aln, _) in scalar_results {
        alignments[idx].push(aln);
    }

    for (i, alns) in alignments.iter_mut().enumerate() {
        if alns.len() != 1 {
            continue;
        }
        let cb = chain_best_per_read[i];
        let cs = chain_second_per_read[i];
        if cb <= 0 || cs <= 0 {
            continue;
        }
        let first = &mut alns[0];
        if first.xs_score.is_some() {
            continue;
        }
        let sub = ((first.score as i64) * (cs as i64) / (cb as i64)) as i32;
        if sub > 0 {
            first.xs_score = Some(sub);
        }
    }

    super::split_read::emit_supplementary_alignments(
        &reads,
        &mut alignments,
        &chains,
        index,
        cfg.cfg,
    );

    stats.unmapped = alignments.iter().filter(|a| a.is_empty()).count();

    let unmapped_mate_info = vec![None; reads.len()];
    AlignBatch {
        reads,
        alignments,
        unmapped_mate_info,
        stats,
    }
}

/// Submit every `SimdJob` to the GPU dispatcher as a single batch of spectral-sieve scans.
#[cfg(feature = "cuda")]
fn gpu_dispatch_spectral(
    simd_jobs: &[SimdJob<'_>],
    alignments: &mut [Vec<Alignment>],
    cfg: AlignmentConfig,
) -> Vec<bool> {
    use crate::alignment::bitpacked::PackedDna;
    use crate::cuda::{CudaJob, dispatcher};

    let cuda_jobs: Vec<CudaJob> = simd_jobs
        .par_iter()
        .map(|job| {
            let pr = PackedDna::pack(&job.read_seq);
            let pt = PackedDna::pack(job.ref_window);
            let shifted = pt.pre_shifted_window();
            let read_len = job.read_seq.len();
            let max_mism = crate::alignment::router::spectral_max_mismatches(read_len);
            CudaJob {
                read_packed: pr.bits,
                read_nucs: read_len as u32,
                ref_shifted: shifted,
                ref_nucs: job.ref_window.len() as u32,
                max_mismatches: max_mism as u32,
            }
        })
        .collect();

    let Some(results) = dispatcher::dispatch(cuda_jobs) else {
        return vec![false; simd_jobs.len()];
    };
    debug_assert_eq!(results.len(), simd_jobs.len());

    if std::env::var_os("KIRA_GPU_DIAG").is_some() {
        let diag_n = std::env::var("KIRA_GPU_DIAG_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5usize);
        diagnose_gpu_vs_cpu(simd_jobs, &results, diag_n);
    }

    let mut accepted = vec![false; simd_jobs.len()];
    for (i, result) in results.iter().enumerate() {
        if result.shift < 0 {
            continue; // GPU rejected — fall through to CPU SW
        }
        let job = &simd_jobs[i];
        let aln = build_alignment_from_gpu(job, result, cfg);
        alignments[job.read_idx].push(aln);
        accepted[i] = true;
    }
    accepted
}

/// Compare GPU dispatch results against a fresh CPU `bitpacked::scan` call for the first `n` jobs.
#[cfg(feature = "cuda")]
fn diagnose_gpu_vs_cpu(
    simd_jobs: &[SimdJob<'_>],
    gpu_results: &[crate::cuda::CudaResult],
    n: usize,
) {
    use crate::alignment::bitpacked::{self, PackedDna};
    let limit = simd_jobs.len().min(n);
    let mut disagreements = 0usize;
    for (i, (job, gpu)) in simd_jobs.iter().zip(gpu_results.iter()).take(limit).enumerate() {
        let read_len = job.read_seq.len();
        let max_mism = crate::alignment::router::spectral_max_mismatches(read_len);
        let pr = PackedDna::pack(&job.read_seq);
        let pt = PackedDna::pack(job.ref_window);
        let shifted = pt.pre_shifted_window();
        let cpu_hit = bitpacked::scan(&pr, &shifted, job.ref_window.len(), max_mism);

        let (cpu_str, cpu_accepts) = match cpu_hit {
            Some(h) => (
                format!("shift={}, mism={}", h.shift, h.mismatches),
                true,
            ),
            None => ("NONE".to_string(), false),
        };
        let gpu_accepts = gpu.shift >= 0;
        let gpu_str = if gpu_accepts {
            format!("shift={}, mism={}", gpu.shift, gpu.mismatches)
        } else {
            "NONE".to_string()
        };

        let mut min_mism = usize::MAX;
        let mut min_at = 0usize;
        let n_shifts = job.ref_window.len() - read_len + 1;
        for t in 0..n_shifts {
            let mism = job
                .read_seq
                .iter()
                .zip(job.ref_window[t..t + read_len].iter())
                .filter(|(a, b)| a != b)
                .count();
            if mism < min_mism {
                min_mism = mism;
                min_at = t;
            }
        }
        let oracle_accepts = min_mism <= max_mism;

        if cpu_accepts != gpu_accepts {
            disagreements += 1;
            eprintln!(
                "[KIRA_GPU_DIAG] job#{i}: GPU={} | CPU={} | naive: min_mism={} at shift={} (threshold={}, oracle should {})  read_len={} ref_len={}",
                gpu_str,
                cpu_str,
                min_mism,
                min_at,
                max_mism,
                if oracle_accepts { "ACCEPT" } else { "REJECT" },
                read_len,
                job.ref_window.len(),
            );
        }
    }
    eprintln!(
        "[KIRA_GPU_DIAG] sampled {} jobs, {} GPU≠CPU disagreements",
        limit, disagreements
    );
}

/// Build a SAM-compatible `Alignment` from a single GPU spectral result.
#[cfg(feature = "cuda")]
fn build_alignment_from_gpu(
    job: &SimdJob<'_>,
    result: &crate::cuda::CudaResult,
    cfg: AlignmentConfig,
) -> Alignment {
    let read_len = job.read_seq.len();
    let shift = result.shift as usize;
    let ref_aligned = &job.ref_window[shift..shift + read_len];

    let cigar = vec![CigarOp {
        len: read_len as u32,
        op: CigarKind::Match,
    }];

    let mut nm: u32 = 0;
    let mut md_bytes: Vec<u8> = Vec::with_capacity(16);
    let mut match_run: u32 = 0;
    for (qb, rb) in job.read_seq.iter().zip(ref_aligned.iter()) {
        if qb == rb {
            match_run += 1;
        } else {
            nm += 1;
            push_u32_decimal(&mut md_bytes, match_run);
            md_bytes.push(*rb);
            match_run = 0;
        }
    }
    push_u32_decimal(&mut md_bytes, match_run);
    // SAFETY: only ASCII digits and ACGTN bytes were pushed.
    let md = unsafe { String::from_utf8_unchecked(md_bytes) };

    let matches = (read_len - result.mismatches as usize) as i32;
    let mismatches = result.mismatches;
    let sw_score = matches * cfg.match_score - mismatches * cfg.mismatch;

    let ref_start_global = job.win_start + shift as u32;
    let ref_end_global = ref_start_global + read_len as u32;

    Alignment {
        kind: AlignmentKind::DpAligned,
        ref_id: job.chain.ref_id,
        ref_start: ref_start_global,
        ref_end: ref_end_global,
        read_start: 0,
        read_end: read_len as u32,
        cigar,
        score: sw_score,
        mapq: 0,
        is_rev: job.is_rev,
        is_secondary: false,
        is_supplementary: false,
        nm,
        md,
        as_score: sw_score,
        xs_score: None,
        xs_strand: None,
        mate: MateInfo::default(),
    }
}

fn build_forced_accept(
    read_seq: &[u8],
    ref_seq: &[u8],
    chain: &AnchorSpan,
    cfg: AlignmentConfig,
) -> Alignment {
    let read_len = read_seq.len();
    let expected_ref_start = chain.ref_start as i32 - chain.read_start as i32;
    let ref_start = expected_ref_start.max(0) as usize;
    let span = read_len.min(ref_seq.len().saturating_sub(ref_start));
    let ref_end = ref_start + span;
    let mut cigar = Vec::new();
    if chain.read_start > 0 {
        cigar.push(CigarOp {
            len: chain.read_start,
            op: CigarKind::SoftClip,
        });
    }
    cigar.push(CigarOp {
        len: span as u32,
        op: CigarKind::Match,
    });
    if chain.read_end < read_len as u32 {
        cigar.push(CigarOp {
            len: (read_len as u32 - chain.read_end),
            op: CigarKind::SoftClip,
        });
    }

    let mut nm = 0u32;
    let mut md_bytes: Vec<u8> = Vec::with_capacity(16);
    let mut run = 0u32;
    for i in 0..span {
        let qb = read_seq[i];
        let rb = ref_seq[ref_start + i];
        if qb == rb {
            run += 1;
        } else {
            nm += 1;
            crate::alignment::push_u32_decimal(&mut md_bytes, run);
            md_bytes.push(rb);
            run = 0;
        }
    }
    crate::alignment::push_u32_decimal(&mut md_bytes, run);
    // SAFETY: only ASCII digits and ACGTN bases were pushed.
    let md = unsafe { String::from_utf8_unchecked(md_bytes) };

    let mism = nm as i32;
    let matches = span as i32 - mism;
    let score = matches * cfg.match_score - mism * cfg.mismatch;

    Alignment {
        kind: crate::types::AlignmentKind::AcceptedUngapped,
        ref_id: chain.ref_id,
        ref_start: ref_start as u32,
        ref_end: ref_end as u32,
        read_start: 0,
        read_end: span as u32,
        cigar,
        score,
        mapq: 60,
        is_rev: chain.strand == Strand::Reverse,
        is_secondary: false,
        is_supplementary: false,
        nm,
        md,
        as_score: score,
        xs_score: None,
        xs_strand: None,
        mate: MateInfo::default(),
    }
}

fn percentile_i32(values: &mut Vec<i32>, pct: usize) -> i32 {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    let idx = (values.len() - 1) * pct / 100;
    values[idx]
}

fn build_simd_window<'a>(
    ref_seq: &'a [u8],
    chain: &Chain,
    read_len: usize,
    cfg: AlignmentConfig,
) -> Option<(u32, &'a [u8])> {
    let band = cfg.bandwidth.max(1) as usize;
    let flank = band * 2;
    let win_len = read_len + flank * 2;
    let desired_start = chain.ref_start as i32 - chain.read_start as i32 - flank as i32;
    if desired_start < 0 {
        return None;
    }
    let start = desired_start as usize;
    if start + win_len > ref_seq.len() {
        return None;
    }
    let window = &ref_seq[start..start + win_len];
    Some((start as u32, window))
}

fn percentile_u16(values: &mut Vec<u16>, pct: usize) -> u16 {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    percentile_u16_sorted(values, pct)
}

fn percentile_u16_sorted(values: &[u16], pct: usize) -> u16 {
    if values.is_empty() {
        return 0;
    }
    let idx = (values.len() - 1) * pct / 100;
    values[idx]
}
