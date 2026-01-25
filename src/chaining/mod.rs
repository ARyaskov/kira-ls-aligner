pub mod rmq;

use crate::types::{Anchor, Chain};

pub use rmq::ChainingStats;

/// Chaining parameters.
#[derive(Clone, Copy, Debug)]
pub struct ChainingConfig {
    pub max_dist: u32,
    pub max_anchors: usize,
    pub max_chains: usize,
    pub gap_open: i32,
    pub gap_extend: i32,
    pub log_gap: f32,
    pub rmq_window: usize,
}

/// Chain anchors into candidate alignments using RMQ/Fenwick-based chaining.
pub fn chain_anchors(
    anchors: &[Anchor],
    cfg: ChainingConfig,
    stats: &mut ChainingStats,
) -> Vec<Chain> {
    rmq::chain_anchors_rmq(anchors, cfg, stats)
}
