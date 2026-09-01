//! Indel left-normalization for emitted CIGARs.
//!
//! An indel inside a homopolymer or tandem repeat has many equivalent
//! placements, and the DP traceback picks one essentially arbitrarily (per
//! read, per strand, per SIMD/scalar kernel). The same biological variant then
//! appears at different coordinates in different reads, which splits pileup
//! support (caller FN) and can spawn phantom alleles (caller FP) — the classic
//! driver of INDEL-precision loss with mpileup-class callers.
//!
//! [`left_normalize_indels`] shifts every indel as far left as the sequences
//! allow (the GATK LeftAlignIndels / `bcftools norm` convention), so all
//! equivalent placements converge on one canonical CIGAR. The transformation
//! is exact: it only moves an I/D block across flanking **matching** M bases,
//! so the implied alignment (which read bases align to which ref bases, and
//! the match/mismatch status of every aligned column) is preserved, NM is
//! unchanged, and the alignment's leftmost coordinate never moves.
//!
//! Mechanically, for a block `xM gI` the one-base left shift
//! `xM gI → (x-1)M gI 1M` is valid iff the ref base preceding the block equals
//! the last inserted read base (the inserted string rotates implicitly through
//! the read sequence). For `xM gD` the shift is valid iff the ref base
//! preceding the block equals the last deleted ref base. Iterating to a fixed
//! point yields the leftmost placement; no periodicity split is needed because
//! single-base steps already reach it.

use rayon::prelude::*;

use crate::types::{CigarKind, CigarOp, ReadRecord};

/// `KIRA_LEFT_NORM` — kill-switch for indel left-normalization. Default ON;
/// set to `0`/`off` to emit traceback-native CIGARs (pre-normalization
/// behaviour) for A/B benchmarking.
fn normalize_enabled() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("KIRA_LEFT_NORM")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("off"))
            .unwrap_or(true)
    })
}

#[inline]
fn is_acgt(b: u8) -> bool {
    matches!(b, b'A' | b'C' | b'G' | b'T')
}

/// Left-normalize every indel in `cigar` against `ref_slice` (the whole
/// contig; `ref_start` is the alignment's 0-based leftmost ref coordinate) and
/// the strand-oriented read sequence.
///
/// Returns `Some(normalized_cigar)` when anything moved, `None` when the CIGAR
/// is already canonical (or has no indels). The caller is responsible for
/// recomputing NM/MD from the new CIGAR.
pub fn left_normalize_indels(
    cigar: &[CigarOp],
    ref_slice: &[u8],
    ref_start: u32,
    read_oriented: &[u8],
) -> Option<Vec<CigarOp>> {
    if !cigar
        .iter()
        .any(|o| matches!(o.op, CigarKind::Ins | CigarKind::Del))
    {
        return None;
    }
    let rs = ref_start as usize;
    let mut ops = cigar.to_vec();
    let mut changed = false;

    // Query/ref coordinates of each op's start, relative to the alignment
    // start. Recomputed after every structural mutation.
    let positions = |ops: &[CigarOp]| -> (Vec<usize>, Vec<usize>) {
        let mut qpos = Vec::with_capacity(ops.len());
        let mut rpos = Vec::with_capacity(ops.len());
        let (mut q, mut r) = (0usize, 0usize);
        for op in ops {
            qpos.push(q);
            rpos.push(r);
            match op.op {
                CigarKind::Match | CigarKind::Ins | CigarKind::SoftClip => q += op.len as usize,
                CigarKind::Del | CigarKind::Skipped => {}
            }
            match op.op {
                CigarKind::Match | CigarKind::Del | CigarKind::Skipped => r += op.len as usize,
                CigarKind::Ins | CigarKind::SoftClip => {}
            }
        }
        (qpos, rpos)
    };

    let mut i = 0usize;
    while i < ops.len() {
        let kind = ops[i].op;
        if !matches!(kind, CigarKind::Ins | CigarKind::Del) {
            i += 1;
            continue;
        }
        let g = ops[i].len as usize;
        let (qpos, rpos) = positions(&ops);
        let (bq, br) = (qpos[i], rpos[i]);

        // Left-shift budget: run of *matching* aligned bases immediately
        // before the block (a mismatch or a non-M op stops the scan — an indel
        // may never shift across a mismatching column or a clip/skip).
        let m_run = if i > 0 && ops[i - 1].op == CigarKind::Match {
            ops[i - 1].len as usize
        } else {
            0
        };
        let mut shift_max = 0usize;
        while shift_max < m_run
            && br - shift_max >= 1
            && bq - shift_max >= 1
            && is_acgt(
                ref_slice
                    .get(rs + br - shift_max - 1)
                    .copied()
                    .unwrap_or(b'N'),
            )
            && ref_slice.get(rs + br - shift_max - 1) == read_oriented.get(bq - shift_max - 1)
        {
            shift_max += 1;
        }

        // Maximal equivalent left shift, one base per step.
        let mut shift = 0usize;
        while shift < shift_max {
            let before = ref_slice.get(rs + br - shift - 1).copied();
            let ok = match kind {
                CigarKind::Del => before == ref_slice.get(rs + br + g - shift - 1).copied(),
                _ => before == read_oriented.get(bq + g - 1).copied(),
            };
            if !ok {
                break;
            }
            shift += 1;
        }
        if shift == 0 {
            i += 1;
            continue;
        }
        changed = true;

        if shift == m_run {
            // The gap crosses the entire preceding M block: swap them, then
            // merge with a same-kind gap now adjacent on the left. The merged
            // gap may itself shift further, so re-examine this index.
            ops.swap(i - 1, i);
            i -= 1;
            if i > 0 && ops[i - 1].op == kind {
                ops[i - 1].len += ops[i].len;
                ops.remove(i);
                i -= 1;
            }
            continue;
        }
        // Partial shift: shorten the M block and re-insert the crossed matches
        // on the gap's right.
        ops[i - 1].len -= shift as u32;
        ops.insert(
            i + 1,
            CigarOp {
                len: shift as u32,
                op: CigarKind::Match,
            },
        );
        i += 1;
    }

    if !changed {
        return None;
    }
    // Compaction: merges created by gap crossings can leave adjacent same-kind
    // ops (e.g. `3M 2M` after a partial shift).
    let mut compact: Vec<CigarOp> = Vec::with_capacity(ops.len());
    for op in ops {
        if let Some(last) = compact.last_mut() {
            if last.op == op.op {
                last.len += op.len;
                continue;
            }
        }
        compact.push(op);
    }
    Some(compact)
}

/// Left-normalize every alignment in the batch, recomputing NM/MD for the
/// records that moved. Runs after pairing/rescue and before MAPQ; positions
/// never change, so TLEN, insert-size estimates and locus concordance are
/// unaffected. `ref_bases` resolves a `ref_id` to its contig sequence (Index
/// or bare Reference, depending on the caller).
pub fn normalize_alignments<'a, F>(
    reads: &[ReadRecord],
    alignments: &mut [Vec<crate::types::Alignment>],
    ref_bases: F,
) where
    F: Fn(u32) -> &'a [u8] + Sync,
{
    if !normalize_enabled() {
        return;
    }
    alignments
        .par_iter_mut()
        .zip(reads.par_iter())
        .for_each(|(alns, read)| {
            let mut rc_scratch: Vec<u8> = Vec::new();
            for aln in alns.iter_mut() {
                let ref_seq = ref_bases(aln.ref_id);
                let oriented: &[u8] = if aln.is_rev {
                    crate::seq::reverse_complement_into(&read.seq, &mut rc_scratch)
                } else {
                    read.seq.as_slice()
                };
                if let Some(cigar) =
                    left_normalize_indels(&aln.cigar, ref_seq, aln.ref_start, oriented)
                {
                    aln.cigar = cigar;
                    let (nm, md) = crate::alignment::compute_nm_md(
                        oriented,
                        ref_seq,
                        0,
                        aln.ref_start as usize,
                        &aln.cigar,
                    );
                    aln.nm = nm;
                    aln.md = md;
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ops(spec: &[(u32, CigarKind)]) -> Vec<CigarOp> {
        spec.iter().map(|&(len, op)| CigarOp { len, op }).collect()
    }

    fn cigar_string(ops: &[CigarOp]) -> String {
        ops.iter().map(|o| o.to_string()).collect()
    }

    const M: CigarKind = CigarKind::Match;
    const I: CigarKind = CigarKind::Ins;
    const D: CigarKind = CigarKind::Del;
    const S: CigarKind = CigarKind::SoftClip;

    /// Insertion inside an A-run slides to the leftmost equivalent position.
    #[test]
    fn homopolymer_insertion_shifts_left() {
        let reference = b"ACGTAAAATGCA";
        let read = b"ACGTAAAAATGCA"; // one extra A in the run
        let cigar = ops(&[(7, M), (1, I), (5, M)]);
        let out = left_normalize_indels(&cigar, reference, 0, read).expect("should move");
        assert_eq!(cigar_string(&out), "4M1I8M");
    }

    /// Deletion crossing the whole preceding M block lands at the front;
    /// ref_start must not move.
    #[test]
    fn deletion_crossing_match_block_becomes_leading() {
        let reference = b"TTTTGG";
        let read = b"TTTGG";
        let cigar = ops(&[(3, M), (1, D), (2, M)]);
        let out = left_normalize_indels(&cigar, reference, 0, read).expect("should move");
        assert_eq!(cigar_string(&out), "1D5M");
    }

    /// A unique-context indel is already leftmost and must not move.
    #[test]
    fn unique_context_indel_is_stable() {
        let reference = b"ACGTAA";
        let read = b"ACGAA";
        let cigar = ops(&[(3, M), (1, D), (2, M)]);
        assert!(left_normalize_indels(&cigar, reference, 0, read).is_none());
    }

    /// An indel may never shift across a mismatching column.
    #[test]
    fn mismatch_blocks_the_shift() {
        let reference = b"AAAA";
        let read = b"ATA"; // SNP at read pos 1 + 1 bp deletion
        let cigar = ops(&[(2, M), (1, D), (1, M)]);
        assert!(left_normalize_indels(&cigar, reference, 0, read).is_none());
    }

    /// Insertion crossing the whole M block becomes a leading I.
    #[test]
    fn insertion_crossing_match_block_becomes_leading() {
        let reference = b"AAATTT";
        let read = b"AAAATT";
        let cigar = ops(&[(3, M), (1, I), (2, M)]);
        let out = left_normalize_indels(&cigar, reference, 0, read).expect("should move");
        assert_eq!(cigar_string(&out), "1I5M");
    }

    /// Two nearby deletions merge when the second slides into the first, and
    /// the merged event is re-examined (leftmost placement of the 3D).
    #[test]
    fn adjacent_deletions_merge_and_reshift() {
        // ref: TTT CC AAAAA GGGGG — read deletes both the CC and one A.
        let reference = b"TTTCCAAAAAGGGGG";
        let read = b"TTTAAAAGGGGG";
        let cigar = ops(&[(3, M), (2, D), (4, M), (1, D), (5, M)]);
        let out = left_normalize_indels(&cigar, reference, 0, read).expect("should move");
        assert_eq!(cigar_string(&out), "3M3D9M");
    }

    /// Tandem-repeat deletion slides by whole units across the repeat.
    #[test]
    fn tandem_repeat_deletion_shifts_to_leftmost_unit() {
        let reference = b"AAACACACACAA"; // CA x3 tandem at 3..9
        let read = b"AAACACACAA"; // one CA unit deleted
        let cigar = ops(&[(7, M), (2, D), (3, M)]);
        let out = left_normalize_indels(&cigar, reference, 0, read).expect("should move");
        assert_eq!(cigar_string(&out), "2M2D8M");
    }

    /// A leading indel (nothing to shift across) is stable.
    #[test]
    fn leading_indel_is_stable() {
        let reference = b"AAATTT";
        let read = b"AAAATT";
        let cigar = ops(&[(1, I), (5, M)]);
        assert!(left_normalize_indels(&cigar, reference, 0, read).is_none());
    }

    /// Gapless CIGARs short-circuit.
    #[test]
    fn gapless_cigar_is_untouched() {
        let reference = b"ACGTACGT";
        let read = b"ACGTACGT";
        let cigar = ops(&[(8, M)]);
        assert!(left_normalize_indels(&cigar, reference, 0, read).is_none());
    }

    /// Soft clips bound the shift: an indel next to a clip cannot move past
    /// the aligned block, but still shifts within it.
    #[test]
    fn soft_clip_bounds_shift() {
        let reference = b"GGAAAATT";
        // read = "CC" clip + "AAAA" aligned at ref 2..6 + extra A in the run + "TT".
        let read = b"CCAAAAATT";
        let cigar = ops(&[(2, S), (4, M), (1, I), (2, M)]);
        let out = left_normalize_indels(&cigar, reference, 2, read).expect("should move");
        // Slides to the front of the A-run but not past the aligned block.
        assert_eq!(cigar_string(&out), "2S1I6M");
    }

    /// Non-zero ref_start offsets are handled (shift uses absolute indexing).
    #[test]
    fn nonzero_ref_start() {
        let reference = b"CCCCAAAATTTT";
        let read = b"AAAAATTTT"; // extra A in the run at ref 4..8
        let cigar = ops(&[(3, M), (1, I), (5, M)]);
        let out = left_normalize_indels(&cigar, reference, 4, read).expect("should move");
        assert_eq!(cigar_string(&out), "1I8M");
    }

    /// Query consumption is invariant under normalization.
    #[test]
    fn query_consumption_is_preserved() {
        let reference = b"TTTCCAAAAAGGGGG";
        let read = b"TTTAAAAGGGGG";
        let cigar = ops(&[(3, M), (2, D), (4, M), (1, D), (5, M)]);
        let consumed = |ops: &[CigarOp]| -> u32 {
            ops.iter()
                .filter(|o| matches!(o.op, M | I | S))
                .map(|o| o.len)
                .sum()
        };
        let out = left_normalize_indels(&cigar, reference, 0, read).unwrap();
        assert_eq!(consumed(&cigar), consumed(&out));
        assert_eq!(consumed(&out) as usize, read.len());
    }
}
