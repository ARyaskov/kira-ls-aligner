use rayon::prelude::*;

use crate::mapq::{MapqConfig, PairMapqContext, assign_mapq};
use crate::types::{Alignment, MateInfo};

use super::stage4_alignment::{AlignBatch, AlignmentBatchStats};

/// Stage 5 output: alignments with MAPQ and primary selection.
#[derive(Clone, Debug)]
pub struct ScoredBatch {
    pub reads: Vec<crate::types::ReadRecord>,
    pub alignments: Vec<Vec<Alignment>>,
    pub unmapped_mate_info: Vec<Option<MateInfo>>,
    pub stats: AlignmentBatchStats,
}

/// Run stage 5 (MAPQ assignment + primary selection).
pub fn run(input: AlignBatch, cfg: MapqConfig, pair_ctx: Option<PairMapqContext>) -> ScoredBatch {
    let reads = input.reads;
    let mut alignments = input.alignments;
    let unmapped_mate_info = input.unmapped_mate_info;
    let stats = input.stats;

    alignments
        .par_iter_mut()
        .zip(reads.par_iter())
        .for_each(|(alns, read)| {
            assign_mapq(alns, read.seq.len(), cfg, pair_ctx);
        });

    ScoredBatch {
        reads,
        alignments,
        unmapped_mate_info,
        stats,
    }
}
