use rayon::prelude::*;

use crate::sketch::{MinimizerConfig, minimizers};
use crate::types::{Minimizer, ReadRecord};

use super::stage0_input::InputBatch;

/// Sketch stage configuration.
#[derive(Clone, Copy, Debug)]
pub struct SketchConfig {
    pub short_k: usize,
    pub short_w: usize,
    pub long_k: usize,
    pub long_w: usize,
    pub long_read_threshold: usize,
}

/// Minimizer sketch for a read.
#[derive(Clone, Debug)]
pub struct ReadSketch {
    pub minimizers: Vec<Minimizer>,
    pub k: usize,
    pub w: usize,
}

/// Stage 1 output.
#[derive(Clone, Debug)]
pub struct SketchBatch {
    pub reads: Vec<ReadRecord>,
    pub sketches: Vec<ReadSketch>,
    pub stats: SketchBatchStats,
}

/// Sketch batch statistics (sampled).
#[derive(Clone, Copy, Debug, Default)]
pub struct SketchBatchStats {
    pub read_len_p50: usize,
    pub read_len_p90: usize,
    pub avg_read_len: f32,
    pub avg_minimizers: f32,
}

pub fn run(input: InputBatch, cfg: SketchConfig) -> SketchBatch {
    let reads = input.reads;
    let sketches: Vec<ReadSketch> = reads
        .par_iter()
        .map(|read| {
            let (k, w) = if read.seq.len() >= cfg.long_read_threshold {
                (cfg.long_k, cfg.long_w)
            } else {
                (cfg.short_k, cfg.short_w)
            };
            let mins = minimizers(&read.seq, &MinimizerConfig { k, w });
            ReadSketch {
                minimizers: mins,
                k,
                w,
            }
        })
        .collect();

    let sample_n = reads.len().min(256);
    let mut lens: Vec<usize> = reads.iter().take(sample_n).map(|r| r.seq.len()).collect();
    lens.sort_unstable();
    let read_len_p50 = if lens.is_empty() {
        0
    } else {
        lens[(lens.len() - 1) * 50 / 100]
    };
    let read_len_p90 = if lens.is_empty() {
        0
    } else {
        lens[(lens.len() - 1) * 90 / 100]
    };
    let total_len: usize = reads.iter().map(|r| r.seq.len()).sum();
    let avg_read_len = if reads.is_empty() {
        0.0
    } else {
        total_len as f32 / reads.len() as f32
    };
    let total_mins: usize = sketches.iter().map(|s| s.minimizers.len()).sum();
    let avg_minimizers = if sketches.is_empty() {
        0.0
    } else {
        total_mins as f32 / sketches.len() as f32
    };
    let stats = SketchBatchStats {
        read_len_p50,
        read_len_p90,
        avg_read_len,
        avg_minimizers,
    };
    SketchBatch {
        reads,
        sketches,
        stats,
    }
}
