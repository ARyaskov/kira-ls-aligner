pub mod chunk_io;
pub mod insert_estimate;
pub mod mode;
pub mod pairing;
pub mod split_read;
pub mod stage0_input;
pub mod stage1_sketch;
pub mod stage2_seeding;
pub mod stage3_chaining;
pub mod stage4_alignment;
pub mod stage5_scoring;
pub mod stage6_output;
pub mod tiled;

use std::time::{Duration, Instant};

use anyhow::Result;

use std::sync::{Arc, RwLock};

use crate::alignment::AlignmentConfig;
use crate::alignment::ac_batch::{self, AcBatchOutput};
use crate::alignment::junc_bed::JunctionIndex;
use crate::alignment::splice::SpliceConfig;
use crate::chaining::ChainingConfig;
use crate::exec::DualPool;
use crate::index::Index;
use crate::io::{OutputConfig, SamFormatter};
use crate::mapq::MapqConfig;
use crate::pipeline::insert_estimate::InsertEstimator;
use crate::pipeline::pairing::{
    PairedConfig, RescueConfig, apply_pairing, pair_rerank, rescue_discordant_pairs,
    rescue_unmapped_mates,
};
use crate::pipeline::stage1_sketch::{
    SketchBatchStats, SketchConfig, run_with_mask as sketch_run_with_mask,
};
use crate::pipeline::stage2_seeding::{SeedBatchStats, run as seed_run};
use crate::pipeline::stage3_chaining::{ChainingBatchStats, run as chain_run};
use crate::pipeline::stage4_alignment::{
    AlignmentBatchStats, AlignmentStageConfig, run as align_run,
};
use crate::pipeline::stage5_scoring::run as score_run;
use crate::pipeline::stage6_output::serialize as output_serialize;
use crate::seeding::SeedingConfig;

/// Pipeline configuration aggregated across stages.
#[derive(Clone, Copy, Debug)]
pub struct PipelineConfig {
    pub sketch: SketchConfig,
    pub seeding: SeedingConfig,
    pub chaining: ChainingConfig,
    pub alignment: AlignmentConfig,
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
    pub min_chain_ratio: f32,
    pub short_preset: bool,
    pub mapq: MapqConfig,
    pub output: OutputConfig,
    /// Paired-end ingestion + proper-pair policy.
    pub paired: PairedConfig,
    /// Splice-aware alignment policy.
    pub splice: SpliceConfig,
}

/// Per-batch stage timing results (stages 1-6).
#[derive(Clone, Debug)]
pub struct PipelineStageTimes {
    pub stages: [Duration; 6],
}

impl PipelineStageTimes {
    pub fn total(&self) -> Duration {
        self.stages.iter().copied().sum()
    }
}

/// Per-batch pipeline stats.
#[derive(Clone, Debug)]
pub struct PipelineBatchStats {
    pub times: PipelineStageTimes,
    pub align: AlignmentBatchStats,
    pub sketch: SketchBatchStats,
    pub seed: SeedBatchStats,
    pub chaining: ChainingBatchStats,
}

/// Result of `Pipeline::process_batch_serialized` — stats plus a ready-to-write SAM byte buffer.
pub struct PipelineBatchOutput {
    pub stats: PipelineBatchStats,
    pub sam_buf: Vec<u8>,
}

pub struct Pipeline {
    pub config: PipelineConfig,
    /// Optional in-memory junction annotation (`--junc-bed`).
    pub junctions: Option<Arc<JunctionIndex>>,
    /// Position tolerance for junction lookup (matches the CLI flag).
    pub junc_bed_tolerance: u32,
    /// Insert-size estimator.
    pub insert_estimator: Arc<RwLock<InsertEstimator>>,
    /// Hybrid-aware executor. The compute/light methods remain useful routing
    /// labels, while both use the full selected worker set because stages are
    /// executed serially.
    pub pool: Arc<DualPool>,
}

impl Pipeline {
    pub fn new(config: PipelineConfig) -> Self {
        Self::with_pool(
            config,
            Arc::new(
                DualPool::homogeneous(
                    std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(1),
                )
                .expect("homogeneous fallback pool always builds"),
            ),
        )
    }

    /// Build a pipeline that routes stages through a caller-supplied
    /// hybrid-aware executor. The standalone callers (`Aligner`,
    /// `run_tiled`) build a single `DualPool` once at startup and share
    /// it across every batch.
    pub fn with_pool(config: PipelineConfig, pool: Arc<DualPool>) -> Self {
        let estimator = Arc::new(RwLock::new(InsertEstimator::new(config.paired)));
        Self {
            config,
            junctions: None,
            junc_bed_tolerance: 2,
            insert_estimator: estimator,
            pool,
        }
    }

    /// Attach a parsed junction annotation.
    pub fn with_junctions(mut self, junctions: Option<Arc<JunctionIndex>>, tolerance: u32) -> Self {
        self.junctions = junctions;
        self.junc_bed_tolerance = tolerance;
        self
    }

    /// Run only stages 1-4 (sketch → seed → chain → align).
    ///
    /// Stage routing labels distinguish compute-heavy and bookkeeping work;
    /// both consume the complete configured pool.
    pub fn process_to_align_batch(
        &self,
        input: stage0_input::InputBatch,
        index: &Index,
    ) -> crate::pipeline::stage4_alignment::AlignBatch {
        let ac = self.pool.install_compute(|| {
            ac_batch::run(
                &input.reads,
                index,
                self.config.alignment,
                self.config.max_alignments,
            )
        });
        let skip_mask = ac.resolved_mask();
        let sketch = self
            .pool
            .install_compute(|| sketch_run_with_mask(input, self.config.sketch, &skip_mask));
        let seeds = self
            .pool
            .install_light(|| seed_run(sketch, index, self.config.seeding));
        let chains = self
            .pool
            .install_light(|| chain_run(seeds, self.config.chaining));
        let mut align = self.pool.install_compute(|| {
            if self.config.splice.enabled {
                crate::alignment::splice::splice_align_batch(
                    chains,
                    index,
                    self.config.alignment,
                    self.config.splice,
                    self.junctions.as_deref(),
                    self.junc_bed_tolerance,
                )
            } else {
                align_run(
                    chains,
                    index,
                    AlignmentStageConfig {
                        cfg: self.config.alignment,
                        min_chain_ratio: self.config.min_chain_ratio,
                        accept_enable: self.config.accept_enable,
                        accept_only_top1: self.config.accept_only_top1,
                        accept_span_slack: self.config.accept_span_slack,
                        accept_min_identity: self.config.accept_min_identity,
                        accept_max_mismatches: self.config.accept_max_mismatches,
                        accept_require_score_margin: self.config.accept_require_score_margin,
                        dp_topk: self.config.dp_topk,
                        dp_abort_margin: self.config.dp_abort_margin,
                        debug_prefilter: self.config.debug_prefilter,
                        debug_prefilter_n: self.config.debug_prefilter_n,
                        debug_force_accept: self.config.debug_force_accept,
                        debug_force_accept_n: self.config.debug_force_accept_n,
                        long_read_threshold: self.config.long_read_threshold,
                        max_alignments: self.config.max_alignments,
                        short_preset: self.config.short_preset,
                    },
                )
            }
        });
        merge_ac_alignments(&mut align.alignments, ac);
        align
    }

    /// Run stages 1-6 producing a ready-to-write SAM byte buffer.
    pub fn process_batch_serialized(
        &self,
        input: stage0_input::InputBatch,
        index: &Index,
        formatter: &SamFormatter,
        read_group: Option<&str>,
    ) -> Result<PipelineBatchOutput> {
        let mut stages = [Duration::ZERO; 6];

        let t0 = Instant::now();
        // Stage 1 (AC + sketch) is SIMD-heavy → P-pool. Capture stats
        // before crossing the pool boundary so the install closure only
        // returns owned data.
        let (ac, sketch) = self.pool.install_compute(|| {
            let ac = ac_batch::run(
                &input.reads,
                index,
                self.config.alignment,
                self.config.max_alignments,
            );
            let skip_mask = ac.resolved_mask();
            let sketch = sketch_run_with_mask(input, self.config.sketch, &skip_mask);
            (ac, sketch)
        });
        let ac_stats = ac.stats;
        let sketch_stats = sketch.stats;
        stages[0] = t0.elapsed();

        // Stage 2 seeding — memory-bound hash lookups.
        let t1 = Instant::now();
        let seeds = self
            .pool
            .install_light(|| seed_run(sketch, index, self.config.seeding));
        let seed_stats = seeds.stats.clone();
        stages[1] = t1.elapsed();

        // Stage 3 chaining — scalar predecessor DP.
        let t2 = Instant::now();
        let chains = self
            .pool
            .install_light(|| chain_run(seeds, self.config.chaining));
        let chaining_stats = chains.stats.clone();
        stages[2] = t2.elapsed();

        let t3 = Instant::now();
        let t_stage4;
        let t_rescue_unmapped;
        let t_pair_rerank;
        let t_rescue_discordant;
        let t_apply_pairing;
        let ts4 = Instant::now();
        // Stage 4 alignment — banded SW / spectral / WFA, the hottest
        // SIMD path in the whole pipeline → P-pool.
        let mut align = self.pool.install_compute(|| {
            if self.config.splice.enabled {
                crate::alignment::splice::splice_align_batch(
                    chains,
                    index,
                    self.config.alignment,
                    self.config.splice,
                    self.junctions.as_deref(),
                    self.junc_bed_tolerance,
                )
            } else {
                align_run(
                    chains,
                    index,
                    AlignmentStageConfig {
                        cfg: self.config.alignment,
                        min_chain_ratio: self.config.min_chain_ratio,
                        accept_enable: self.config.accept_enable,
                        accept_only_top1: self.config.accept_only_top1,
                        accept_span_slack: self.config.accept_span_slack,
                        accept_min_identity: self.config.accept_min_identity,
                        accept_max_mismatches: self.config.accept_max_mismatches,
                        accept_require_score_margin: self.config.accept_require_score_margin,
                        dp_topk: self.config.dp_topk,
                        dp_abort_margin: self.config.dp_abort_margin,
                        debug_prefilter: self.config.debug_prefilter,
                        debug_prefilter_n: self.config.debug_prefilter_n,
                        debug_force_accept: self.config.debug_force_accept,
                        debug_force_accept_n: self.config.debug_force_accept_n,
                        long_read_threshold: self.config.long_read_threshold,
                        max_alignments: self.config.max_alignments,
                        short_preset: self.config.short_preset,
                    },
                )
            }
        });
        merge_ac_alignments(&mut align.alignments, ac);
        t_stage4 = ts4.elapsed();

        if std::env::var_os("KIRA_STATS").is_some() {
            eprintln!(
                "[KIRA_AC] reads={} eligible={} resolved={} ambiguous={} fwd_hits={} rev_hits={} build={:.2}ms scan={:.2}ms",
                ac_stats.n_reads,
                ac_stats.reads_eligible,
                ac_stats.reads_resolved,
                ac_stats.reads_ambiguous,
                ac_stats.fwd_hits,
                ac_stats.rev_hits,
                ac_stats.build_ms,
                ac_stats.scan_ms,
            );
        }

        let paired_cfg = self
            .insert_estimator
            .read()
            .map(|e| e.current())
            .unwrap_or(self.config.paired);

        let tru = Instant::now();
        {
            let reads = &align.reads;
            let alns = &mut align.alignments;
            let pcfg = paired_cfg;
            let acfg = self.config.alignment;
            self.pool.install_compute(move || {
                rescue_unmapped_mates(reads, alns, index, &pcfg, acfg, RescueConfig::default())
            });
        }
        t_rescue_unmapped = tru.elapsed();

        let tpr = Instant::now();
        pair_rerank(
            &align.reads,
            &mut align.alignments,
            &paired_cfg,
            self.config.dp_topk.max(2),
        );
        t_pair_rerank = tpr.elapsed();

        let trd = Instant::now();
        {
            let reads = &align.reads;
            let alns = &mut align.alignments;
            let pcfg = paired_cfg;
            let acfg = self.config.alignment;
            self.pool.install_compute(move || {
                rescue_discordant_pairs(reads, alns, index, &pcfg, acfg, RescueConfig::default())
            });
        }
        t_rescue_discordant = trd.elapsed();

        let tap = Instant::now();
        apply_pairing(
            &align.reads,
            &mut align.alignments,
            &mut align.unmapped_mate_info,
            &paired_cfg,
        );
        t_apply_pairing = tap.elapsed();

        let refined_after = if paired_cfg.is_paired() {
            self.insert_estimator
                .write()
                .ok()
                .and_then(|mut e| e.observe_batch(&align.alignments))
        } else {
            None
        };
        let paired_cfg_final = refined_after.unwrap_or(paired_cfg);

        stages[3] = t3.elapsed();
        if std::env::var_os("KIRA_STATS").is_some() {
            eprintln!(
                "[KIRA_STAGE3_BREAKDOWN] stage4_align={:.3}ms rescue_unmapped={:.3}ms pair_rerank={:.3}ms rescue_discordant={:.3}ms apply_pairing={:.3}ms total_s3={:.3}ms",
                t_stage4.as_secs_f64() * 1000.0,
                t_rescue_unmapped.as_secs_f64() * 1000.0,
                t_pair_rerank.as_secs_f64() * 1000.0,
                t_rescue_discordant.as_secs_f64() * 1000.0,
                t_apply_pairing.as_secs_f64() * 1000.0,
                stages[3].as_secs_f64() * 1000.0,
            );
        }

        let t4 = Instant::now();
        let pair_ctx = if paired_cfg_final.is_paired() {
            Some(crate::mapq::PairMapqContext {
                insert_mean: paired_cfg_final.insert_mean,
                insert_sd: paired_cfg_final.insert_sd,
                discordant_cap: 10,
            })
        } else {
            None
        };
        // Stage 5 MAPQ — light scalar math.
        let scored = self
            .pool
            .install_light(|| score_run(align, self.config.mapq, pair_ctx));
        let align_stats = scored.stats.clone();
        stages[4] = t4.elapsed();

        // Stage 6 SAM emit — pure string formatting.
        let t5 = Instant::now();
        // Keep the final aggregate buffer owned by the caller thread.  On
        // Windows, repeatedly allocating it on a Rayon worker and freeing it
        // on a writer thread makes the process retain roughly one SAM file's
        // worth of heap across a long run.
        let sam_buf = output_serialize(
            scored,
            formatter,
            read_group,
            self.config.output,
            self.config.max_alignments,
        );
        stages[5] = t5.elapsed();

        Ok(PipelineBatchOutput {
            stats: PipelineBatchStats {
                times: PipelineStageTimes { stages },
                align: align_stats,
                sketch: sketch_stats,
                seed: seed_stats,
                chaining: chaining_stats,
            },
            sam_buf,
        })
    }
}

/// Move AC-stage alignments into the cascade output. Reads resolved by AC had
/// their minimizers suppressed in stage 1, so their cascade slot is empty —
/// we overwrite it with the perfect-match alignment(s) produced by AC.
fn merge_ac_alignments(cascade: &mut [Vec<crate::types::Alignment>], mut ac: AcBatchOutput) {
    debug_assert_eq!(cascade.len(), ac.alignments.len());
    for (dst, src) in cascade.iter_mut().zip(ac.alignments.iter_mut()) {
        if !src.is_empty() {
            *dst = std::mem::take(src);
        }
    }
}
