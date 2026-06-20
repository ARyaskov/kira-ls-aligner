//! Integration tests for `alignment::wfa::wfa_align_semi_global`.
//!
//! Each WFA result is cross-checked by replaying the CIGAR against the
//! same penalties — the replay score MUST equal the reported score, and
//! the consumed text length MUST equal `text_end`.

use kira_ls_aligner::alignment::wfa::{WfaPenalties, wfa_align_semi_global};
use kira_ls_aligner::types::{CigarKind, CigarOp};

fn pen() -> WfaPenalties {
    WfaPenalties {
        mismatch: 4,
        gap_open: 6,
        gap_extend: 1,
    }
}

fn cigar_str(ops: &[CigarOp]) -> String {
    ops.iter().map(|o| o.to_string()).collect::<String>()
}

fn apply_cigar(pattern: &[u8], text: &[u8], cigar: &[CigarOp]) -> (i32, usize) {
    let p = pen();
    let mut score = 0;
    let mut qi = 0usize;
    let mut ti = 0usize;
    for op in cigar {
        match op.op {
            CigarKind::Match => {
                for _ in 0..op.len {
                    if pattern[qi] != text[ti] {
                        score += p.mismatch;
                    }
                    qi += 1;
                    ti += 1;
                }
            }
            CigarKind::Ins => {
                score += p.gap_open + p.gap_extend * op.len as i32;
                qi += op.len as usize;
            }
            CigarKind::Del => {
                score += p.gap_open + p.gap_extend * op.len as i32;
                ti += op.len as usize;
            }
            CigarKind::SoftClip => {
                qi += op.len as usize;
            }
            CigarKind::Skipped => {
                // WFA never emits N; defensive for exhaustiveness.
                ti += op.len as usize;
            }
        }
    }
    assert_eq!(qi, pattern.len(), "CIGAR did not consume full pattern");
    (score, ti)
}

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[test]
fn exact_match_zero_score() {
    let aln = wfa_align_semi_global(b"ACGTACGT", b"ACGTACGTAA", pen(), 20).unwrap();
    assert_eq!(aln.score, 0);
    assert_eq!(cigar_str(&aln.cigar), "8M");
    assert_eq!(aln.text_end, 8);
}

#[test]
fn single_mismatch() {
    let pattern = b"ACGTACGT";
    let text = b"ACATACGT";
    let aln = wfa_align_semi_global(pattern, text, pen(), 20).unwrap();
    let (replay_score, replay_text) = apply_cigar(pattern, text, &aln.cigar);
    assert_eq!(replay_score, aln.score);
    assert_eq!(replay_text, aln.text_end);
    assert_eq!(aln.score, pen().mismatch);
}

#[test]
fn single_insertion_in_query() {
    let pattern = b"ACGTACGT";
    let text = b"ACGACGT";
    let aln = wfa_align_semi_global(pattern, text, pen(), 20).unwrap();
    let (replay_score, _) = apply_cigar(pattern, text, &aln.cigar);
    assert_eq!(replay_score, aln.score);
    let p = pen();
    assert_eq!(aln.score, p.gap_open + p.gap_extend);
}

#[test]
fn single_deletion_in_query() {
    let pattern = b"ACGACGT";
    let text = b"ACGTACGT";
    let aln = wfa_align_semi_global(pattern, text, pen(), 20).unwrap();
    let (replay_score, _) = apply_cigar(pattern, text, &aln.cigar);
    assert_eq!(replay_score, aln.score);
    let p = pen();
    assert_eq!(aln.score, p.gap_open + p.gap_extend);
}

#[test]
fn cutoff_returns_none() {
    // With (x=4, o=6, e=1), the optimal alignment of pattern AAAAAAAA against
    // text TTTTTTTT is an 8-long insertion gap (cost 14), NOT 8 mismatches
    // (cost 32). Free trailing text + a gap is sometimes cheaper than
    // mismatching — this is correct semi-global behaviour.
    let pattern = b"AAAAAAAA";
    let text = b"TTTTTTTT";
    let p = pen();
    let expected = p.gap_open + p.gap_extend * 8;
    assert!(wfa_align_semi_global(pattern, text, p, 5).is_none());
    let aln = wfa_align_semi_global(pattern, text, p, expected).unwrap();
    assert_eq!(aln.score, expected);
    let (replay, _) = apply_cigar(pattern, text, &aln.cigar);
    assert_eq!(replay, aln.score);
}

#[test]
fn long_run_with_two_mismatches() {
    let pattern: Vec<u8> = b"ACGT".repeat(20).to_vec();
    let mut text = pattern.clone();
    text[15] = if text[15] == b'A' { b'C' } else { b'A' };
    text[60] = if text[60] == b'C' { b'A' } else { b'C' };
    let aln = wfa_align_semi_global(&pattern, &text, pen(), 20).unwrap();
    let (replay_score, _) = apply_cigar(&pattern, &text, &aln.cigar);
    assert_eq!(replay_score, aln.score);
    assert_eq!(aln.score, 2 * pen().mismatch);
}

#[test]
fn replay_consistency_random() {
    let bases = b"ACGT";
    let mut rng = 0xC0FFEEu64;
    for _ in 0..50 {
        let m = 20 + (xorshift(&mut rng) as usize % 80);
        let pattern: Vec<u8> = (0..m)
            .map(|_| bases[(xorshift(&mut rng) as usize) % 4])
            .collect();
        let mut text: Vec<u8> = pattern.clone();
        let num_errors = (xorshift(&mut rng) as usize) % 4;
        for _ in 0..num_errors {
            let kind = xorshift(&mut rng) % 3;
            let pos = (xorshift(&mut rng) as usize) % text.len();
            match kind {
                0 => {
                    text[pos] = bases[(xorshift(&mut rng) as usize) % 4];
                }
                1 => {
                    text.insert(pos, bases[(xorshift(&mut rng) as usize) % 4]);
                }
                _ => {
                    if text.len() > 1 {
                        text.remove(pos);
                    }
                }
            }
        }
        for _ in 0..5 {
            text.push(bases[(xorshift(&mut rng) as usize) % 4]);
        }
        let aln = wfa_align_semi_global(&pattern, &text, pen(), 80).expect("score within bound");
        let (replay_score, replay_text) = apply_cigar(&pattern, &text, &aln.cigar);
        assert_eq!(
            replay_score,
            aln.score,
            "replay-score != reported-score for pattern={:?} text={:?} cigar={}",
            String::from_utf8_lossy(&pattern),
            String::from_utf8_lossy(&text),
            cigar_str(&aln.cigar),
        );
        assert_eq!(replay_text, aln.text_end);
    }
}
