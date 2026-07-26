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
    /// Materialise each surviving chain's anchor path into [`Chain::anchors`].
    /// Only the splice-aware aligner reads that field.
    pub keep_anchors: bool,
}

impl ChainingConfig {
    /// Enable [`Self::keep_anchors`] — required by the splice path.
    pub fn with_anchors(mut self) -> Self {
        self.keep_anchors = true;
        self
    }
}

pub use rmq::ChainScratch;

/// Chain anchors into candidate alignments using bounded predecessor DP.
pub fn chain_anchors(
    anchors: &[Anchor],
    cfg: ChainingConfig,
    stats: &mut ChainingStats,
) -> Vec<Chain> {
    let mut scratch = ChainScratch::default();
    rmq::chain_anchors_rmq(anchors, cfg, stats, &mut scratch)
}

/// [`chain_anchors`] using a caller-owned working set, so the batch driver can
/// keep one [`ChainScratch`] per worker instead of allocating per read.
pub fn chain_anchors_with_scratch(
    anchors: &[Anchor],
    cfg: ChainingConfig,
    stats: &mut ChainingStats,
    scratch: &mut ChainScratch,
) -> Vec<Chain> {
    rmq::chain_anchors_rmq(anchors, cfg, stats, scratch)
}
