use rayon::prelude::*;

use crate::mapq::{MapqConfig, assign_mapq};
use crate::types::Alignment;

use super::stage4_alignment::{AlignBatch, AlignmentBatchStats};

/// Stage 5 output: alignments with MAPQ and primary selection.
#[derive(Clone, Debug)]
pub struct ScoredBatch {
    pub reads: Vec<crate::types::ReadRecord>,
    pub alignments: Vec<Vec<Alignment>>,
    pub stats: AlignmentBatchStats,
}

pub fn run(input: AlignBatch, cfg: MapqConfig) -> ScoredBatch {
    let reads = input.reads;
    let mut alignments = input.alignments;
    let stats = input.stats;

    alignments
        .par_iter_mut()
        .zip(reads.par_iter())
        .for_each(|(alns, read)| {
            assign_mapq(alns, read.seq.len(), cfg);
        });

    ScoredBatch {
        reads,
        alignments,
        stats,
    }
}
