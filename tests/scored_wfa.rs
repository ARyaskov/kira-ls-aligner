//! Independent affine DP oracle: checks optimality, not just CIGAR validity.
use kira_ls_aligner::alignment::{AlignmentConfig, wfa};
use kira_ls_aligner::types::CigarKind;

fn cfg(a: i32, b: i32, o: i32, e: i32) -> AlignmentConfig {
    AlignmentConfig {
        match_score: a,
        mismatch: b,
        gap_open: o,
        gap_extend: e,
        bandwidth: 50,
        xdrop: 0,
        clip_penalty: 5,
    }
}

fn oracle(q: &[u8], r: &[u8], c: AlignmentConfig, lead: usize) -> i32 {
    let width = r.len() + 1;
    let neg = i32::MIN / 8;
    let mut h = vec![neg; (q.len() + 1) * width];
    let mut ins = h.clone();
    let mut del = h.clone();
    let lead = lead.min(r.len().saturating_sub(1));
    for (j, cell) in h.iter_mut().enumerate().take(width) {
        *cell = if j <= lead {
            0
        } else {
            -c.gap_open - c.gap_extend * (j - lead) as i32
        };
    }
    for i in 1..=q.len() {
        h[i * width] = -c.gap_open - c.gap_extend * i as i32;
        ins[i * width] = h[i * width];
        for j in 1..width {
            let p = i * width + j;
            ins[p] = (h[p - width] - c.gap_open - c.gap_extend).max(ins[p - width] - c.gap_extend);
            del[p] = (h[p - 1] - c.gap_open - c.gap_extend).max(del[p - 1] - c.gap_extend);
            let substitution = if q[i - 1] == r[j - 1] {
                c.match_score
            } else {
                -c.mismatch
            };
            h[p] = (h[p - width - 1] + substitution).max(ins[p]).max(del[p]);
        }
    }
    *h[q.len() * width..].iter().max().unwrap()
}

fn replay(q: &[u8], r: &[u8], aln: &wfa::WfaAlignment, c: AlignmentConfig) -> i32 {
    let (mut qi, mut ri, mut score) = (0, aln.text_start, 0);
    for op in &aln.cigar {
        match op.op {
            CigarKind::Match => {
                for _ in 0..op.len {
                    score += if q[qi] == r[ri] {
                        c.match_score
                    } else {
                        -c.mismatch
                    };
                    qi += 1;
                    ri += 1;
                }
            }
            CigarKind::Ins => {
                qi += op.len as usize;
                score -= c.gap_open + c.gap_extend * op.len as i32;
            }
            CigarKind::Del => {
                ri += op.len as usize;
                score -= c.gap_open + c.gap_extend * op.len as i32;
            }
            _ => panic!("unexpected CIGAR op"),
        }
    }
    assert_eq!(qi, q.len());
    assert_eq!(ri, aln.text_end);
    assert_eq!(c.match_score * q.len() as i32 - score, aln.score);
    score
}

fn next(state: &mut u64) -> usize {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 32) as usize
}

#[test]
fn scored_wfa_matches_independent_affine_oracle_and_score_only() {
    let mut rng = 0x6a09e667f3bcc909;
    for trial in 0..1800 {
        let c = [cfg(1, 4, 6, 1), cfg(2, 3, 2, 1), cfg(3, 2, 0, 2)][trial % 3];
        let qlen = 1 + next(&mut rng) % 28;
        let rlen = 1 + next(&mut rng) % 35;
        let q: Vec<u8> = (0..qlen).map(|_| b"ACGT"[next(&mut rng) % 4]).collect();
        let r: Vec<u8> = (0..rlen).map(|_| b"ACGT"[next(&mut rng) % 4]).collect();
        let lead = next(&mut rng) % 5;
        let opts = wfa::WfaOptions {
            max_score: 300,
            adaptive_drop: None,
            text_begin_free: lead as i32,
        };
        let expected = oracle(&q, &r, c, lead);
        let aln = wfa::wfa_align_scored(&q, &r, c, opts).expect("ample budget");
        assert_eq!(
            replay(&q, &r, &aln, c),
            expected,
            "trial={trial} q={q:?} r={r:?}"
        );
        assert_eq!(wfa::wfa_score_only_scored(&q, &r, c, opts), Some(expected));
        let loss = q.len() as i32 * c.match_score - expected;
        if loss > 0 {
            let bounded = wfa::WfaOptions {
                max_score: loss - 1,
                ..opts
            };
            assert!(wfa::wfa_align_scored(&q, &r, c, bounded).is_none());
            assert!(wfa::wfa_score_only_scored(&q, &r, c, bounded).is_none());
        }
    }
}

#[test]
fn match_rewards_change_the_optimal_path() {
    let c = cfg(2, 3, 2, 1);
    let mut rng = 0xbb67ae8584caa73b;
    let mut improved = 0;
    for _ in 0..300 {
        let q: Vec<u8> = (0..18).map(|_| b"ACGT"[next(&mut rng) % 4]).collect();
        let r: Vec<u8> = (0..24).map(|_| b"ACGT"[next(&mut rng) % 4]).collect();
        let opts = wfa::WfaOptions::exact(200);
        let old = wfa::wfa_align(
            &q,
            &r,
            wfa::WfaPenalties {
                mismatch: c.mismatch,
                gap_open: c.gap_open,
                gap_extend: c.gap_extend,
            },
            opts,
        )
        .unwrap();
        let new = wfa::wfa_align_scored(&q, &r, c, opts).unwrap();
        // The legacy result's cost uses another model, so replay with a copied
        // transformed loss solely to reuse the independent CIGAR checker.
        let mut old_replayed = old.clone();
        let mut qi = 0;
        let mut ri = old.text_start;
        let mut old_score = 0;
        for op in &old.cigar {
            match op.op {
                CigarKind::Match => {
                    for _ in 0..op.len {
                        old_score += if q[qi] == r[ri] {
                            c.match_score
                        } else {
                            -c.mismatch
                        };
                        qi += 1;
                        ri += 1;
                    }
                }
                CigarKind::Ins => {
                    qi += op.len as usize;
                    old_score -= c.gap_open + c.gap_extend * op.len as i32;
                }
                CigarKind::Del => {
                    ri += op.len as usize;
                    old_score -= c.gap_open + c.gap_extend * op.len as i32;
                }
                _ => unreachable!(),
            }
        }
        old_replayed.score = c.match_score * q.len() as i32 - old_score;
        assert_eq!(replay(&q, &r, &old_replayed, c), old_score);
        let new_score = replay(&q, &r, &new, c);
        assert!(new_score >= old_score);
        improved += usize::from(new_score > old_score);
    }
    assert!(improved > 0, "must exercise the old objective mismatch");
}

#[test]
fn shifted_seed_origin_recovers_leading_indel_without_clipping() {
    use kira_ls_aligner::alignment::{AnchorSpan, try_fast_dp_alignment};
    use kira_ls_aligner::types::Strand;
    let mut state = 0x123456789abcdef0;
    let reference: Vec<u8> = (0..400).map(|_| b"ACGT"[next(&mut state) % 4]).collect();
    let mut read = reference[100..250].to_vec();
    read.splice(10..10, b"TGCAC".iter().copied());
    let span = AnchorSpan {
        ref_id: 0,
        read_start: 40,
        read_end: 155,
        ref_start: 135,
        ref_end: 250,
        strand: Strand::Forward,
    };
    let (aln, _) = try_fast_dp_alignment(&read, &reference, &span, 115, cfg(1, 4, 6, 1), false)
        .expect("bounded ends-free recovery");
    assert_eq!(aln.ref_start, 100);
    assert_eq!(aln.read_start, 0);
    assert_eq!(aln.read_end, 155);
    assert!(
        aln.cigar
            .iter()
            .any(|op| op.op == CigarKind::Ins && op.len == 5)
    );
    assert!(!aln.cigar.iter().any(|op| op.op == CigarKind::SoftClip));
}
