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
    /// Whether any non-primary, non-supplementary candidate of each read
    /// sits on an ALT contig — parallel to `alignments`; drives the bwa-mem
    /// `-h` second (ALT) XA cap in stage 6.
    pub xa_has_alt: Vec<bool>,
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

    // `XA` ALT metadata: any qualifying ALT candidate among the counted
    // (non-primary, non-supplementary) candidates selects the `-h` second
    // cap in stage 6. A primary-only ALT placement must stay `false`.
    let xa_has_alt: Vec<bool> = alignments
        .par_iter()
        .map(|alns| {
            alns.iter()
                .skip(1)
                .filter(|a| !a.is_supplementary)
                .any(|a| cfg.is_alt(a.ref_id))
        })
        .collect();

    ScoredBatch {
        reads,
        alignments,
        xa_has_alt,
        unmapped_mate_info,
        stats,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AlignmentKind, CigarKind, CigarOp};

    fn aln(ref_id: u32, secondary: bool) -> Alignment {
        Alignment {
            kind: AlignmentKind::DpAligned,
            ref_id,
            ref_start: 100,
            ref_end: 150,
            read_start: 0,
            read_end: 50,
            cigar: vec![CigarOp {
                len: 50,
                op: CigarKind::Match,
            }],
            score: 50,
            mapq: 0,
            is_rev: false,
            is_secondary: secondary,
            is_supplementary: false,
            nm: 0,
            md: "50".to_string(),
            as_score: 50,
            xs_score: None,
            xs_strand: None,
            mate: MateInfo::default(),
        }
    }

    fn run_stage5(mask: Option<&'static [bool]>, alns: Vec<Vec<Alignment>>) -> ScoredBatch {
        let reads: Vec<crate::types::ReadRecord> = (0..alns.len())
            .map(|i| crate::types::ReadRecord {
                id: format!("r{i}"),
                seq: vec![b'A'; 50],
                qual: None,
                pair_role: PairRole::Unpaired,
                repeat_min_occ: 1,
                comment: None,
            })
            .collect();
        run(
            AlignBatch {
                reads,
                alignments: alns,
                unmapped_mate_info: Vec::new(),
                stats: Default::default(),
            },
            MapqConfig {
                short_read_len: 500,
                mapq_cap_short: 60,
                mapq_cap_long: 60,
                alt_mask: mask,
            },
            None,
        )
    }

    /// `xa_has_alt` counts ALT candidates among the non-primary,
    /// non-supplementary candidates only: a primary-only ALT placement must
    /// stay `false`.
    #[test]
    fn xa_has_alt_flags_secondary_alt_candidates_only() {
        let mask: &'static [bool] = Box::leak(vec![false, true].into_boxed_slice());
        let scored = run_stage5(
            Some(mask),
            vec![
                // Secondary on the ALT contig -> true.
                vec![aln(0, false), aln(1, true)],
                // Primary-only ALT with a non-ALT secondary -> false.
                vec![aln(1, false), aln(0, true)],
                // No ALT anywhere -> false.
                vec![aln(0, false), aln(0, true)],
            ],
        );
        assert_eq!(scored.xa_has_alt, vec![true, false, false]);
    }

    #[test]
    fn xa_has_alt_is_false_without_alt_mask() {
        let scored = run_stage5(None, vec![vec![aln(0, false), aln(1, true)]]);
        assert_eq!(scored.xa_has_alt, vec![false]);
    }
}
