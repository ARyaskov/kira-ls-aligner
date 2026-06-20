//! Tests for WFA2 modes added on top of the base wavefront aligner:
//! adaptive pruning (`adaptive_drop`) and ends-free leading text
//! (`text_begin_free`). Adaptive results are cross-checked by replaying the
//! CIGAR; parity with the exact algorithm is asserted when the drop is generous.

use kira_ls_aligner::alignment::wfa::{WfaOptions, WfaPenalties, wfa_align, wfa_score_only};
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

/// Replay a CIGAR starting at `text[text_start..]`; returns (score, abs_text_end).
fn apply_cigar(pattern: &[u8], text: &[u8], text_start: usize, cigar: &[CigarOp]) -> (i32, usize) {
    let p = pen();
    let mut score = 0;
    let mut qi = 0usize;
    let mut ti = text_start;
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
            CigarKind::SoftClip => qi += op.len as usize,
            CigarKind::Skipped => ti += op.len as usize,
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

/// With a generous drop, adaptive pruning must match the exact optimum on
/// low-divergence reads, and every adaptive CIGAR must replay to its score.
#[test]
fn adaptive_matches_exact_low_divergence() {
    let bases = b"ACGT";
    let mut rng = 0xBADC0FFEu64;
    for _ in 0..60 {
        let m = 40 + (xorshift(&mut rng) as usize % 100);
        let pattern: Vec<u8> = (0..m)
            .map(|_| bases[(xorshift(&mut rng) as usize) % 4])
            .collect();
        let mut text = pattern.clone();
        // 0..3 edits → stays well within a drop of 50.
        let num_errors = (xorshift(&mut rng) as usize) % 4;
        for _ in 0..num_errors {
            let pos = (xorshift(&mut rng) as usize) % text.len();
            match xorshift(&mut rng) % 3 {
                0 => text[pos] = bases[(xorshift(&mut rng) as usize) % 4],
                1 => text.insert(pos, bases[(xorshift(&mut rng) as usize) % 4]),
                _ => {
                    if text.len() > 1 {
                        text.remove(pos);
                    }
                }
            }
        }
        for _ in 0..6 {
            text.push(bases[(xorshift(&mut rng) as usize) % 4]);
        }

        let exact =
            wfa_align(&pattern, &text, pen(), WfaOptions::exact(200)).expect("exact within budget");
        let adaptive = wfa_align(
            &pattern,
            &text,
            pen(),
            WfaOptions {
                max_score: 200,
                adaptive_drop: Some(50),
                text_begin_free: 0,
            },
        )
        .expect("adaptive within budget");

        // Self-consistency of the adaptive CIGAR.
        let (replay, end) = apply_cigar(&pattern, &text, adaptive.text_start, &adaptive.cigar);
        assert_eq!(
            replay,
            adaptive.score,
            "adaptive cigar {} replay mismatch",
            cigar_str(&adaptive.cigar)
        );
        assert_eq!(end, adaptive.text_end);

        // Heuristic can never beat the optimum; with a wide drop it ties it.
        assert!(
            adaptive.score >= exact.score,
            "adaptive must not beat exact"
        );
        assert_eq!(
            adaptive.score, exact.score,
            "wide-drop adaptive should equal exact"
        );
    }
}

/// A tight drop may return a higher-cost path (or None), but whatever it returns
/// must be a self-consistent alignment and never cheaper than the true optimum.
#[test]
fn adaptive_tight_drop_self_consistent() {
    let bases = b"ACGT";
    let mut rng = 0x1357_9bdfu64;
    for _ in 0..40 {
        let m = 50 + (xorshift(&mut rng) as usize % 80);
        let pattern: Vec<u8> = (0..m)
            .map(|_| bases[(xorshift(&mut rng) as usize) % 4])
            .collect();
        let mut text = pattern.clone();
        for _ in 0..(xorshift(&mut rng) as usize % 5) {
            let pos = (xorshift(&mut rng) as usize) % text.len();
            text[pos] = bases[(xorshift(&mut rng) as usize) % 4];
        }
        for _ in 0..6 {
            text.push(bases[(xorshift(&mut rng) as usize) % 4]);
        }

        let exact = wfa_align(&pattern, &text, pen(), WfaOptions::exact(400)).unwrap();
        if let Some(adaptive) = wfa_align(
            &pattern,
            &text,
            pen(),
            WfaOptions {
                max_score: 400,
                adaptive_drop: Some(3),
                text_begin_free: 0,
            },
        ) {
            let (replay, end) = apply_cigar(&pattern, &text, adaptive.text_start, &adaptive.cigar);
            assert_eq!(replay, adaptive.score);
            assert_eq!(end, adaptive.text_end);
            assert!(adaptive.score >= exact.score);
        }
    }
}

/// The linear-memory score-only forward engine is exact: its cost must equal
/// the full aligner's reported score on every case (and agree on infeasibility).
#[test]
fn score_only_matches_full_aligner() {
    let bases = b"ACGT";
    let mut rng = 0x9e37_79b9u64;
    for _ in 0..200 {
        let m = 20 + (xorshift(&mut rng) as usize % 120);
        let pattern: Vec<u8> = (0..m)
            .map(|_| bases[(xorshift(&mut rng) as usize) % 4])
            .collect();
        let mut text = pattern.clone();
        let num_errors = (xorshift(&mut rng) as usize) % 6;
        for _ in 0..num_errors {
            let pos = (xorshift(&mut rng) as usize) % text.len();
            match xorshift(&mut rng) % 3 {
                0 => text[pos] = bases[(xorshift(&mut rng) as usize) % 4],
                1 => text.insert(pos, bases[(xorshift(&mut rng) as usize) % 4]),
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

        // Deliberately tight budgets too, to exercise the None path.
        for budget in [8i32, 30, 200] {
            let full =
                wfa_align(&pattern, &text, pen(), WfaOptions::exact(budget)).map(|a| a.score);
            let score = wfa_score_only(&pattern, &text, pen(), WfaOptions::exact(budget));
            assert_eq!(
                full,
                score,
                "score-only {:?} != full {:?} at budget {} for pattern={} text={}",
                score,
                full,
                budget,
                String::from_utf8_lossy(&pattern),
                String::from_utf8_lossy(&text),
            );
        }
    }
}

/// Ends-free leading text: when the pattern sits at text offset 5, pinning the
/// read to text[0] forces a 5 bp leading deletion (cost open+5·ext), whereas
/// `text_begin_free=5` aligns it for free with `text_start = 5`.
#[test]
fn ends_free_skips_leading_text() {
    let pattern = b"ACGTACGTACGTACGT"; // 16 bp
    let mut text = Vec::new();
    text.extend_from_slice(b"TTTTT"); // 5 bp leading prefix, no match
    text.extend_from_slice(pattern);
    text.extend_from_slice(b"GG");
    let p = pen();

    // Pinned at text[0]: optimal is a 5 bp leading deletion then 16M.
    let pinned = wfa_align(pattern, &text, p, WfaOptions::exact(200)).unwrap();
    assert_eq!(pinned.text_start, 0);
    assert_eq!(pinned.score, p.gap_open + p.gap_extend * 5);
    let (replay, end) = apply_cigar(pattern, &text, pinned.text_start, &pinned.cigar);
    assert_eq!(replay, pinned.score);
    assert_eq!(end, pinned.text_end);

    // Ends-free: skip the 5 leading bases for free → exact match at offset 5.
    let free = wfa_align(
        pattern,
        &text,
        p,
        WfaOptions {
            max_score: 200,
            adaptive_drop: None,
            text_begin_free: 5,
        },
    )
    .unwrap();
    assert_eq!(free.score, 0, "exact match should be free");
    assert_eq!(free.text_start, 5, "read starts at text offset 5");
    assert_eq!(cigar_str(&free.cigar), "16M");
    assert_eq!(free.text_end, 21);
    assert!(free.score < pinned.score, "ends-free must beat pinned here");
}
