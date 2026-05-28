//! Integration tests for the CGK embedding + rescue pipeline.
//!
//! Exercises three layers:
//!
//! 1. **Embedding properties** — repeated edits keep the embedded Hamming
//!    distance under a sublinear ceiling (the core CGK promise).
//! 2. **Index recall** — when a read carries an indel vs. the reference
//!    window it should align to, the CGK side-index returns the true
//!    window as a candidate.
//! 3. **End-to-end rescue** — given a synthetic reference and a read
//!    with a known indel, the rescue path returns a sane alignment at
//!    the truth position with non-trivial WFA-scored quality.
//!
//! These tests share advice/seed conventions with the unit tests inside
//! `src/alignment/cgk.rs`; if you change the embedding constants there,
//! re-tune the bounds in this file too.

use kira_ls_aligner::alignment::AlignmentConfig;
use kira_ls_aligner::alignment::cgk::{
    CgkIndex, CgkRescue, DEFAULT_BANK_COUNT, FingerprintScheme, embed, EmbeddingAdvice,
};
use kira_ls_aligner::types::Strand;

fn rng_byte(state: &mut u64) -> u8 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^= z >> 31;
    b"ACGT"[(z & 3) as usize]
}

fn random_dna(len: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    (0..len).map(|_| rng_byte(&mut s)).collect()
}

fn apply_one_insertion(s: &[u8], pos: usize, base: u8) -> Vec<u8> {
    let mut v = Vec::with_capacity(s.len());
    v.extend_from_slice(&s[..pos]);
    v.push(base);
    v.extend_from_slice(&s[pos..s.len() - 1]);
    v
}

fn apply_one_deletion(s: &[u8], pos: usize, pad: u8) -> Vec<u8> {
    let mut v = Vec::with_capacity(s.len());
    v.extend_from_slice(&s[..pos]);
    v.extend_from_slice(&s[pos + 1..]);
    v.push(pad);
    v
}

fn hamming_bits(a: &[u64], b: &[u64], n_bits: usize) -> usize {
    let n_full = n_bits / 64;
    let tail_bits = n_bits & 63;
    let mut h = 0usize;
    for i in 0..n_full {
        h += (a[i] ^ b[i]).count_ones() as usize;
    }
    if tail_bits > 0 {
        let mask = (1u64 << tail_bits) - 1;
        h += ((a[n_full] ^ b[n_full]) & mask).count_ones() as usize;
    }
    h
}

#[test]
fn embedding_hamming_grows_sublinearly_with_edit_distance() {
    // Demonstrate the CGK property: as we accumulate more indels, the
    // embedded Hamming distance grows but stays well below the
    // would-have-been-linear-on-tail-length naive baseline (~75 % of
    // the bits past the first edit).
    let advice = EmbeddingAdvice::new(2024, 900);
    let n_words = (6 * 150usize).div_ceil(64);

    for n_edits in [0usize, 1, 2, 3, 5] {
        // Run multiple seeds and take the max so we report a worst-case
        // ceiling rather than a lucky single observation.
        let mut max_h = 0usize;
        for seed in [11u64, 23, 47, 67, 89] {
            let mut x = random_dna(150, seed.wrapping_mul(7));
            let mut y = x.clone();
            // Apply n_edits insertions at evenly spaced positions.
            for k in 0..n_edits {
                let pos = 10 + k * 25;
                let base = b"ACGT"[k % 4];
                y = apply_one_insertion(&y, pos, base);
            }
            let mut ex = vec![0u64; n_words];
            let mut ey = vec![0u64; n_words];
            embed(&x, &advice, &mut ex);
            embed(&y, &advice, &mut ey);
            let h = hamming_bits(&ex, &ey, 900);
            if h > max_h {
                max_h = h;
            }
            // Silence "unused mut" — `x` is mutated in spirit by the
            // edit-application but the binding is on `y` here.
            let _ = &mut x;
        }
        // Loose ceiling: even at d=5 indels, Hamming stays well below
        // 50 % of the active region. The hardcoded threshold below is
        // empirically calibrated across seeds with slack so this test
        // isn't a flaky tight bound.
        let max_active = 600; // first ~2n bits hold real data; tail is padding
        assert!(
            max_h < max_active * 50 / 100,
            "n_edits={}, max embedded Hamming={} bits (≥ 50% of active region {})",
            n_edits,
            max_h,
            max_active
        );
    }
}

#[test]
fn cgk_index_recalls_indel_edited_reads() {
    // Build a CGK side-index on a synthetic 50 kb reference, then query
    // with indel-bearing copies of several known windows. Track recall.
    let scheme = FingerprintScheme::new(2024, 150, DEFAULT_BANK_COUNT);
    let ref_bases = random_dna(50_000, 12345);
    let idx = CgkIndex::build_from_sequences(scheme, 30, std::iter::once((0u32, ref_bases.as_slice())));

    let mut ins_hits = 0;
    let mut del_hits = 0;
    let trials = [600usize, 1500, 3000, 7500, 15000, 30000, 45000];
    for &true_pos in &trials {
        let win = &ref_bases[true_pos..true_pos + 150];
        let ins = apply_one_insertion(win, 75, b'C');
        // Pure 1-base deletion: drop position 60, pull in the byte AFTER
        // the window so the read still aligns naturally to the same
        // 150-byte slot. Padding with a constant ('A') would synthesise
        // an extra mismatch and turn this into a 2-edit case.
        let next_base = ref_bases
            .get(true_pos + 150)
            .copied()
            .unwrap_or(b'A');
        let del = apply_one_deletion(win, 60, next_base);
        let ins_cands = idx.query(&ins, 1);
        if ins_cands
            .iter()
            .any(|&(rid, pos)| rid == 0 && pos == true_pos as u32)
        {
            ins_hits += 1;
        }
        let del_cands = idx.query(&del, 1);
        if del_cands
            .iter()
            .any(|&(rid, pos)| rid == 0 && pos == true_pos as u32)
        {
            del_hits += 1;
        }
    }
    // Recall floor: ≥ 5/7 for insertions, ≥ 4/7 for deletions. The
    // asymmetry is intrinsic to the CGK advance rule: an insertion's
    // "extra advance" couples back to the random walk faster than a
    // deletion's "missing advance" because the input-bit-XOR-advice
    // term is not symmetric on data direction. With 16 banks recall is
    // already much higher than the strict O(d²) Hamming bound would
    // predict — the bound below is conservative to avoid flakiness.
    assert!(
        ins_hits >= 5,
        "insertion recall = {} / {}",
        ins_hits,
        trials.len()
    );
    assert!(
        del_hits >= 4,
        "deletion recall = {} / {}",
        del_hits,
        trials.len()
    );
}

#[test]
fn rescue_finds_alignment_at_truth_position() {
    // Full pipeline: build CGK index on a synthetic reference, install
    // a CgkRescue with the alignment config, generate a read with a
    // known indel, run rescue, verify the resulting alignment lands at
    // the truth position with a sensible WFA score.
    let scheme = FingerprintScheme::new(0xDEADBEEF, 150, DEFAULT_BANK_COUNT);
    let ref_bases = random_dna(30_000, 42);
    let index = CgkIndex::build_from_sequences(scheme, 30, std::iter::once((0u32, ref_bases.as_slice())));

    let cfg = AlignmentConfig {
        match_score: 1,
        mismatch: 4,
        gap_open: 6,
        gap_extend: 1,
        bandwidth: 32,
        xdrop: 100,
        clip_penalty: 5,
    };
    let rescue = CgkRescue {
        index,
        ref_bases: vec![ref_bases.clone()],
        cfg,
        max_candidates: 16,
        min_bank_hits: 1,
    };

    let true_pos = 12_000usize;
    let win = &ref_bases[true_pos..true_pos + 150];
    let read = apply_one_insertion(win, 75, b'C');

    let aln = rescue
        .rescue(&read, Strand::Forward)
        .expect("rescue should return an alignment for an indel-bearing read");

    // Stride-30 windowing means the recovered position is the stride-
    // aligned window enclosing the truth, which is true_pos itself
    // (12_000 is a multiple of 30). Allow ±60 bp slack for the WFA
    // pad — `text_end - text_start` may bleed into a neighbouring
    // window's territory.
    let drift = (aln.ref_start as i32 - true_pos as i32).abs();
    assert!(
        drift <= 60,
        "rescue landed at ref_start={} but truth={}; drift={} > 60",
        aln.ref_start,
        true_pos,
        drift
    );

    // Alignment must consume the entire read.
    assert_eq!(aln.read_start, 0);
    assert_eq!(aln.read_end as usize, read.len());

    // Score should be positive and dominated by matches: 149 matches
    // (one base eaten by the insertion) + 1 insertion penalty
    // = 149 - (gap_open + gap_extend) = 149 - 7 = 142 expected.
    // Real score may be a bit lower if WFA picks a slightly different
    // split; require ≥ 110 as a sanity lower bound.
    assert!(
        aln.score >= 110,
        "rescue alignment score={} (expected ≥ 110 for d=1)",
        aln.score
    );
}

#[test]
fn rescue_returns_none_for_unrelated_read() {
    // Sanity: a completely random read that doesn't match any window
    // should not produce a rescue alignment (or if it does, the score
    // should be terrible — but in budget-bounded WFA it usually just
    // returns None).
    let scheme = FingerprintScheme::new(0xCAFE, 150, DEFAULT_BANK_COUNT);
    let ref_bases = random_dna(30_000, 100);
    let index =
        CgkIndex::build_from_sequences(scheme, 30, std::iter::once((0u32, ref_bases.as_slice())));

    let cfg = AlignmentConfig {
        match_score: 1,
        mismatch: 4,
        gap_open: 6,
        gap_extend: 1,
        bandwidth: 32,
        xdrop: 100,
        clip_penalty: 5,
    };
    let rescue = CgkRescue {
        index,
        ref_bases: vec![ref_bases.clone()],
        cfg,
        max_candidates: 16,
        min_bank_hits: 1,
    };

    // Use a different random seed so the read has no correlation with
    // the reference.
    let unrelated = random_dna(150, 999_999);
    let aln = rescue.rescue(&unrelated, Strand::Forward);
    // Unrelated reads should land none → falls through to banded SW in
    // the real cascade. If WFA happens to produce something with the
    // budget, it must have a junk score. Assert "if Some, score is
    // low" rather than strict None — the budget tuning may admit a
    // weak alignment occasionally.
    if let Some(aln) = aln {
        assert!(
            aln.score < 80,
            "unrelated-read rescue produced score={}, expected low (< 80)",
            aln.score
        );
    }
}
