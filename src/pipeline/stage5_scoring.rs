use rayon::prelude::*;

use crate::mapq::{
    MapqConfig, PairMapqContext, assign_mapq_preserving_primary, assign_mapq_with_qual,
};
use crate::types::{Alignment, MateInfo, PairRole};

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
    run_with_primary_policy(input, cfg, pair_ctx, false)
}

/// [`run`] with `preserve_primary` forcing slot 0 to stay primary for
/// unpaired reads too — needed when an earlier stage chose the primary on
/// grounds other than score (bwa-mem `-5`: smallest read coordinate).
pub fn run_with_primary_policy(
    input: AlignBatch,
    cfg: MapqConfig,
    pair_ctx: Option<PairMapqContext>,
    preserve_primary: bool,
) -> ScoredBatch {
    let reads = input.reads;
    let mut alignments = input.alignments;
    let unmapped_mate_info = input.unmapped_mate_info;
    let stats = input.stats;

    alignments
        .par_iter_mut()
        .zip(reads.par_iter())
        .for_each(|(alns, read)| {
            if read.pair_role == PairRole::Unpaired && !preserve_primary {
                assign_mapq_with_qual(
                    alns,
                    read.seq.len(),
                    read.qual.as_deref(),
                    cfg,
                    pair_ctx,
                    read.repeat_min_occ,
                );
            } else {
                assign_mapq_preserving_primary(
                    alns,
                    read.seq.len(),
                    read.qual.as_deref(),
                    cfg,
                    pair_ctx,
                    read.repeat_min_occ,
                );
            }
        });

    ScoredBatch {
        reads,
        alignments,
        unmapped_mate_info,
        stats,
    }
}
