use rayon::prelude::*;

use crate::chaining::{ChainScratch, ChainingConfig, ChainingStats, chain_anchors_with_scratch};
use crate::types::{Chain, ReadRecord};

use super::stage2_seeding::{SeedBatch, SeedBatchStats};

/// Stage 3 output: chains per read.
#[derive(Clone, Debug)]
pub struct ChainBatch {
    pub reads: Vec<ReadRecord>,
    pub chains: Vec<Vec<Chain>>,
    pub stats: ChainingBatchStats,
    pub seed_stats: SeedBatchStats,
}

/// Per-batch chaining stats.
#[derive(Clone, Debug, Default)]
pub struct ChainingBatchStats {
    pub anchors_used_for_chaining: usize,
    pub chains_pruned_early: usize,
}

pub fn run(input: SeedBatch, cfg: ChainingConfig) -> ChainBatch {
    let reads = input.reads;
    let seed_stats = input.stats.clone();
    let mut stats = ChainingBatchStats::default();

    // One working set per worker, reused across that worker's reads.
    let results: Vec<(Vec<Chain>, ChainingStats)> = input
        .anchors
        .par_iter()
        .map_init(ChainScratch::default, |scratch, anchors| {
            let mut s = ChainingStats::default();
            let chains = chain_anchors_with_scratch(anchors, cfg, &mut s, scratch);
            (chains, s)
        })
        .collect();

    let mut chains: Vec<Vec<Chain>> = Vec::with_capacity(results.len());
    for (c, s) in results {
        stats.anchors_used_for_chaining += s.anchors_used;
        stats.chains_pruned_early += s.chains_pruned;
        chains.push(c);
    }

    ChainBatch {
        reads,
        chains,
        stats,
        seed_stats,
    }
}
