pub mod mode;
pub mod stage0_input;
pub mod stage1_sketch;
pub mod stage2_seeding;
pub mod stage3_chaining;
pub mod stage4_alignment;
pub mod stage5_scoring;
pub mod stage6_output;

use std::time::{Duration, Instant};

use anyhow::Result;

use crate::alignment::AlignmentConfig;
use crate::chaining::ChainingConfig;
use crate::index::Index;
use crate::io::{OutputConfig, SamWriter};
use crate::mapq::MapqConfig;
use crate::pipeline::stage1_sketch::{SketchBatchStats, SketchConfig, run as sketch_run};
use crate::pipeline::stage2_seeding::{SeedBatchStats, run as seed_run};
use crate::pipeline::stage3_chaining::{ChainingBatchStats, run as chain_run};
use crate::pipeline::stage4_alignment::{
    AlignmentBatchStats, AlignmentStageConfig, run as align_run,
};
use crate::pipeline::stage5_scoring::run as score_run;
use crate::pipeline::stage6_output::run as output_run;
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

pub struct Pipeline {
    pub config: PipelineConfig,
}

impl Pipeline {
    pub fn new(config: PipelineConfig) -> Self {
        Self { config }
    }

    pub fn process_batch(
        &self,
        input: stage0_input::InputBatch,
        index: &Index,
        writer: &mut SamWriter,
        read_group: Option<&str>,
    ) -> Result<PipelineBatchStats> {
        let mut stages = [Duration::ZERO; 6];

        let t0 = Instant::now();
        let sketch = sketch_run(input, self.config.sketch);
        let sketch_stats = sketch.stats;
        stages[0] = t0.elapsed();

        let t1 = Instant::now();
        let seeds = seed_run(sketch, index, self.config.seeding);
        let seed_stats = seeds.stats.clone();
        stages[1] = t1.elapsed();

        let t2 = Instant::now();
        let chains = chain_run(seeds, self.config.chaining);
        let chaining_stats = chains.stats.clone();
        stages[2] = t2.elapsed();

        let t3 = Instant::now();
        let align = align_run(
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
        );
        stages[3] = t3.elapsed();

        let t4 = Instant::now();
        let scored = score_run(align, self.config.mapq);
        let align_stats = scored.stats.clone();
        stages[4] = t4.elapsed();

        let t5 = Instant::now();
        output_run(
            scored,
            writer,
            read_group,
            self.config.output,
            self.config.max_alignments,
        )?;
        stages[5] = t5.elapsed();

        Ok(PipelineBatchStats {
            times: PipelineStageTimes { stages },
            align: align_stats,
            sketch: sketch_stats,
            seed: seed_stats,
            chaining: chaining_stats,
        })
    }
}
