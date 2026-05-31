use super::*;
use crate::alignment::BatchInput;
use crate::types::Strand;

fn dummy_chain(read_len: u32) -> crate::alignment::AnchorSpan {
    crate::alignment::AnchorSpan {
        ref_id: 0,
        ref_start: 0,
        ref_end: read_len,
        read_start: 0,
        read_end: read_len,
        strand: Strand::Forward,
    }
}

fn cfg() -> AlignmentConfig {
    AlignmentConfig {
        match_score: 1,
        mismatch: 4,
        gap_open: 6,
        gap_extend: 1,
        bandwidth: 50,
        xdrop: 50,
        clip_penalty: 5,
    }
}

fn rand_dna(len: usize, mut state: u64) -> Vec<u8> {
    let alphabet = b"ACGT";
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        out.push(alphabet[(state >> 60) as usize & 0x3]);
    }
    out
}

/// Ground-truth full O(q·r) banded SW, no boundary quirks. Returns just
/// the best-cell score so we can verify the i8 kernel directly without
/// going through the production `banded_sw` (which carries a long-standing
/// off-by-one at the matrix top-left — see `prev_diag` fallback).
fn truth_sw(read: &[u8], reference: &[u8], cfg: AlignmentConfig) -> i32 {
    let q = read.len();
    let r = reference.len();
    let neg = i32::MIN / 4;
    let mut prev_h = vec![0i32; r + 1];
    let mut prev_e = vec![neg; r + 1];
    let mut cur_h = vec![0i32; r + 1];
    let mut cur_e = vec![neg; r + 1];
    let mut best = 0i32;
    for i in 1..=q {
        cur_h[0] = 0;
        cur_e[0] = neg;
        let mut cur_f = neg;
        for j in 1..=r {
            let score = if read[i - 1] == reference[j - 1] {
                cfg.match_score
            } else {
                -cfg.mismatch
            };
            let h_match = prev_h[j - 1] + score;
            let e = (prev_h[j] - cfg.gap_open).max(prev_e[j] - cfg.gap_extend);
            let f = (cur_h[j - 1] - cfg.gap_open).max(cur_f - cfg.gap_extend);
            let h = h_match.max(e).max(f).max(0);
            cur_h[j] = h;
            cur_e[j] = e;
            cur_f = f;
            best = best.max(h);
        }
        std::mem::swap(&mut prev_h, &mut cur_h);
        std::mem::swap(&mut prev_e, &mut cur_e);
    }
    best
}

#[test]
fn int8_matches_truth_on_clean_reads() {
    if !is_x86_feature_detected!("avxvnni") {
        eprintln!("skip: CPU lacks AVX-VNNI");
        return;
    }
    // Sub-saturation: 100 bp reads, match=1 ⇒ max H = 100 < 120.
    let cfg = cfg();
    let read_len = 100usize;
    let ref_pad = 30usize;
    let lanes = LANES;

    let mut reads: Vec<Vec<u8>> = Vec::with_capacity(lanes);
    let mut refs: Vec<Vec<u8>> = Vec::with_capacity(lanes);
    for k in 0..lanes {
        let r = rand_dna(read_len + ref_pad, 0xCAFE_BABE_DEAD ^ (k as u64));
        let q = r[ref_pad / 2..ref_pad / 2 + read_len].to_vec();
        reads.push(q);
        refs.push(r);
    }
    let inputs: Vec<BatchInput<'_>> = reads
        .iter()
        .zip(refs.iter())
        .map(|(q, r)| BatchInput {
            read_seq: q.as_slice(),
            ref_window: r.as_slice(),
            win_start: 0,
            chain: dummy_chain(read_len as u32),
            is_rev: false,
            abort_score: i32::MIN / 8,
        })
        .collect();

    let i8_results = unsafe { sw_batch_int8(&inputs, cfg) };
    assert_eq!(i8_results.len(), lanes);

    for k in 0..lanes {
        let want = truth_sw(&reads[k], &refs[k], cfg);
        assert_eq!(
            i8_results[k].score, want,
            "lane {k}: i8={} truth={}",
            i8_results[k].score, want
        );
        // Every lane has a perfect substring → score must hit read_len.
        assert_eq!(i8_results[k].score, read_len as i32, "lane {k}");
    }
}

#[test]
fn int8_matches_truth_with_mismatches() {
    if !is_x86_feature_detected!("avxvnni") {
        eprintln!("skip: CPU lacks AVX-VNNI");
        return;
    }
    // clip_penalty=0 so the kernel surfaces the global-max cell rather
    // than biasing toward an end-of-read alignment.
    let cfg = AlignmentConfig { clip_penalty: 0, ..cfg() };
    let read_len = 80usize;
    let ref_pad = 20usize;
    let lanes = LANES;

    let mut reads: Vec<Vec<u8>> = Vec::with_capacity(lanes);
    let mut refs: Vec<Vec<u8>> = Vec::with_capacity(lanes);
    for k in 0..lanes {
        let r = rand_dna(read_len + ref_pad, 0x1234_5678 ^ (k as u64));
        let mut q = r[ref_pad / 2..ref_pad / 2 + read_len].to_vec();
        // 2 mismatches per lane, deterministic positions per lane.
        let p1 = (k * 7) % read_len;
        let p2 = (k * 11 + 23) % read_len;
        for &pos in &[p1, p2] {
            q[pos] = match q[pos] {
                b'A' => b'C',
                b'C' => b'G',
                b'G' => b'T',
                _ => b'A',
            };
        }
        reads.push(q);
        refs.push(r);
    }
    let inputs: Vec<BatchInput<'_>> = reads
        .iter()
        .zip(refs.iter())
        .map(|(q, r)| BatchInput {
            read_seq: q.as_slice(),
            ref_window: r.as_slice(),
            win_start: 0,
            chain: dummy_chain(read_len as u32),
            is_rev: false,
            abort_score: i32::MIN / 8,
        })
        .collect();

    let i8_results = unsafe { sw_batch_int8(&inputs, cfg) };
    for k in 0..lanes {
        let want = truth_sw(&reads[k], &refs[k], cfg);
        assert_eq!(
            i8_results[k].score, want,
            "lane {k}: i8={} truth={}",
            i8_results[k].score, want
        );
    }
}

#[test]
fn int8_path_viable_thresholds() {
    let c = cfg();
    assert!(int8_path_viable(100, c)); // 100 * 1 = 100 < 120
    assert!(int8_path_viable(119, c)); // 119 < 120
    assert!(!int8_path_viable(120, c)); // 120 ≥ 120 — not viable
    assert!(!int8_path_viable(150, c)); // 150 — past threshold

    let big_match = AlignmentConfig { match_score: 2, ..c };
    assert!(!int8_path_viable(100, big_match)); // 100 * 2 = 200
    assert!(int8_path_viable(50, big_match)); // 50 * 2 = 100 < 120
}
