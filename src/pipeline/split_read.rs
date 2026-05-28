//! Split-read / supplementary alignment detection.

use rayon::prelude::*;

use crate::alignment::{AlignmentConfig, AnchorSpan, align_chain_with_meta};
use crate::index::Index;
use crate::types::{Alignment, Chain, ReadRecord};

/// Minimum chain-score ratio (vs primary) to consider a chain for supplementary alignment.
const SUPP_MIN_SCORE_RATIO: f32 = 0.5;

/// Minimum read-span for a supplementary candidate.
const SUPP_MIN_READ_SPAN: u32 = 50;

/// Maximum read-region overlap with the primary, as a percentage of the candidate chain's own.
const SUPP_MAX_OVERLAP_PCT: u32 = 50;

/// Maximum supplementary alignments to add per read.
const MAX_SUPP_PER_READ: usize = 4;

/// Scan stage-4 alignments for split-read evidence and append supplementary alignments in place.
pub fn emit_supplementary_alignments(
    reads: &[ReadRecord],
    alignments: &mut [Vec<Alignment>],
    chains: &[Vec<Chain>],
    index: &Index,
    cfg: AlignmentConfig,
) {
    debug_assert_eq!(reads.len(), alignments.len());
    debug_assert_eq!(reads.len(), chains.len());

    alignments
        .par_iter_mut()
        .zip(reads.par_iter())
        .zip(chains.par_iter())
        .for_each(|((alns, read), chain_list)| {
            if alns.is_empty() || chain_list.len() < 2 {
                return;
            }
            let picks = pick_supplementary_chains(&alns[0], chain_list);
            for chain in picks {
                let span = AnchorSpan {
                    ref_id: chain.ref_id,
                    ref_start: chain.ref_start,
                    ref_end: chain.ref_end,
                    read_start: chain.read_start,
                    read_end: chain.read_end,
                    strand: chain.strand,
                };
                let ref_seq = index.ref_bases(chain.ref_id as usize);
                let (mut aln, _early) =
                    align_chain_with_meta(read, ref_seq, &span, cfg, i32::MIN / 8);
                aln.is_supplementary = true;
                alns.push(aln);
            }
        });
}

/// Pick chains worth running a supplementary SW for, given the primary alignment that stage 4.
fn pick_supplementary_chains<'a>(
    primary: &Alignment,
    chains: &'a [Chain],
) -> Vec<&'a Chain> {
    let primary_score = chains[0].score.max(1);
    let min_score = ((primary_score as f32) * SUPP_MIN_SCORE_RATIO).ceil() as i32;

    let primary_rs = primary.read_start;
    let primary_re = primary.read_end;

    let mut picks: Vec<&Chain> = Vec::new();
    for chain in chains.iter().skip(1) {
        if chain.score < min_score {
            // Chains are score-sorted; nothing below this rank survives.
            break;
        }
        let span = chain.read_end.saturating_sub(chain.read_start);
        if span < SUPP_MIN_READ_SPAN {
            continue;
        }
        // Compete-vs-primary check: high read overlap → secondary, not supp.
        let overlap_pp = read_overlap_pct(
            primary_rs,
            primary_re,
            chain.read_start,
            chain.read_end,
        );
        if overlap_pp > SUPP_MAX_OVERLAP_PCT {
            continue;
        }
        let dup = picks.iter().any(|c| {
            read_overlap_pct(c.read_start, c.read_end, chain.read_start, chain.read_end)
                > SUPP_MAX_OVERLAP_PCT
        });
        if dup {
            continue;
        }
        picks.push(chain);
        if picks.len() >= MAX_SUPP_PER_READ {
            break;
        }
    }
    picks
}

/// Percentage overlap between `[a_start..a_end)` and `[b_start..b_end)` as a share of the.
fn read_overlap_pct(a_start: u32, a_end: u32, b_start: u32, b_end: u32) -> u32 {
    let a_len = a_end.saturating_sub(a_start);
    let b_len = b_end.saturating_sub(b_start);
    if a_len == 0 || b_len == 0 {
        return 0;
    }
    let overlap_start = a_start.max(b_start);
    let overlap_end = a_end.min(b_end);
    if overlap_end <= overlap_start {
        return 0;
    }
    let overlap = overlap_end - overlap_start;
    (overlap.saturating_mul(100)) / a_len
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AlignmentKind, CigarKind, CigarOp, MateInfo, Strand};

    fn make_chain(score: i32, read_start: u32, read_end: u32, ref_id: u32) -> Chain {
        Chain {
            anchors: Vec::new(),
            score,
            ref_id,
            read_start,
            read_end,
            ref_start: 1000,
            ref_end: 1000 + (read_end - read_start),
            strand: Strand::Forward,
        }
    }

    fn make_aln(read_start: u32, read_end: u32) -> Alignment {
        Alignment {
            kind: AlignmentKind::DpAligned,
            ref_id: 0,
            ref_start: 1000,
            ref_end: 1000 + (read_end - read_start),
            read_start,
            read_end,
            cigar: vec![CigarOp {
                len: read_end - read_start,
                op: CigarKind::Match,
            }],
            score: 100,
            mapq: 60,
            is_rev: false,
            is_secondary: false,
            is_supplementary: false,
            nm: 0,
            md: "0".to_string(),
            as_score: 100,
            xs_score: None,
            xs_strand: None,
            mate: MateInfo::default(),
        }
    }

    #[test]
    fn picks_disjoint_high_score_chain_as_supp_candidate() {
        let primary = make_aln(0, 80);
        let chains = vec![
            make_chain(100, 0, 80, 0),
            make_chain(70, 80, 150, 1),
        ];
        let picks = pick_supplementary_chains(&primary, &chains);
        assert_eq!(picks.len(), 1);
        assert_eq!(picks[0].read_start, 80);
    }

    #[test]
    fn rejects_overlapping_chain() {
        let primary = make_aln(0, 150);
        let chains = vec![
            make_chain(100, 0, 150, 0),
            make_chain(90, 10, 150, 1),
        ];
        let picks = pick_supplementary_chains(&primary, &chains);
        assert!(picks.is_empty());
    }

    #[test]
    fn rejects_low_score_chain() {
        // Disjoint but score < 50 % of primary → noise, skip.
        let primary = make_aln(0, 80);
        let chains = vec![
            make_chain(100, 0, 80, 0),
            make_chain(30, 80, 150, 1), // 0.3 × primary
        ];
        let picks = pick_supplementary_chains(&primary, &chains);
        assert!(picks.is_empty());
    }

    #[test]
    fn rejects_short_chain() {
        // Disjoint, high-score, but span 30 bp < 50 bp threshold.
        let primary = make_aln(0, 80);
        let chains = vec![
            make_chain(100, 0, 80, 0),
            make_chain(80, 100, 130, 1),
        ];
        let picks = pick_supplementary_chains(&primary, &chains);
        assert!(picks.is_empty());
    }

    #[test]
    fn dedups_clustered_chains_in_same_region() {
        // Two near-identical chains at read[80..150] — pick only one.
        let primary = make_aln(0, 80);
        let chains = vec![
            make_chain(100, 0, 80, 0),
            make_chain(80, 80, 150, 1),
            make_chain(75, 82, 150, 2),
        ];
        let picks = pick_supplementary_chains(&primary, &chains);
        assert_eq!(picks.len(), 1);
        assert_eq!(picks[0].ref_id, 1, "first qualifying candidate wins");
    }

    #[test]
    fn caps_at_max_supp_per_read() {
        let primary = make_aln(0, 200);
        let mut chains = vec![make_chain(200, 0, 200, 0)];
        for k in 0..6u32 {
            let start = 200 + k * 60;
            chains.push(make_chain(150, start, start + 55, k + 1));
        }
        let picks = pick_supplementary_chains(&primary, &chains);
        assert_eq!(picks.len(), MAX_SUPP_PER_READ);
    }

    #[test]
    fn skips_when_no_second_chain() {
        // Sole chain → nothing to do.
        let primary = make_aln(0, 80);
        let chains = vec![make_chain(100, 0, 80, 0)];
        let picks = pick_supplementary_chains(&primary, &chains);
        assert!(picks.is_empty());
    }

    #[test]
    fn overlap_pct_basic_cases() {
        // Disjoint → 0.
        assert_eq!(read_overlap_pct(0, 50, 50, 100), 0);
        // Identical → 100.
        assert_eq!(read_overlap_pct(0, 100, 0, 100), 100);
        // Smaller `b` fully inside larger `a` → share of `a` (50/100).
        assert_eq!(read_overlap_pct(0, 100, 50, 100), 50);
        assert_eq!(read_overlap_pct(0, 100, 50, 150), 50);
        // Larger `b` fully covers smaller `a` → 100 % of `a`.
        assert_eq!(read_overlap_pct(50, 100, 0, 100), 100);
        // Empty interval → 0.
        assert_eq!(read_overlap_pct(50, 50, 0, 100), 0);
    }
}
