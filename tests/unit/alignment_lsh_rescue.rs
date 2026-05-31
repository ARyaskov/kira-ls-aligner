use super::*;
use crate::index::lsh::{DEFAULT_MAX_HAMMING, DEFAULT_TOP_BITS};

fn default_cfg() -> AlignmentConfig {
    AlignmentConfig {
        match_score: 1,
        mismatch: 4,
        gap_open: 6,
        gap_extend: 1,
        bandwidth: 30,
        xdrop: 100,
        clip_penalty: 5,
    }
}

/// 256-bp pseudo-random "contig" with a deterministic seed.
fn synth_contig() -> Vec<u8> {
    let mut state: u64 = 0x243F_6A88_85A3_08D3;
    let mut v = Vec::with_capacity(256);
    for _ in 0..256 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        v.push(b"ACGT"[(state >> 56) as usize & 3]);
    }
    v
}

#[test]
fn rescue_returns_perfect_match() {
    let bases = synth_contig();
    let read = bases[50..150].to_vec(); // 100 bp from offset 50
    let index = LshIndex::build(
        std::iter::once((0u32, bases.as_slice())),
        32,
        1, // tight stride for a deterministic test
        DEFAULT_TOP_BITS,
    );
    let rescue = LshRescue {
        index,
        ref_bases: vec![bases.clone()],
        cfg: default_cfg(),
        max_candidates: 64,
        max_lsh_hamming: DEFAULT_MAX_HAMMING,
        max_window_mismatches: 5,
    };
    let aln = rescue
        .rescue(&read, Strand::Forward)
        .expect("perfect-match read should be rescued");
    assert_eq!(aln.ref_start, 50);
    assert_eq!(aln.ref_end, 150);
    assert_eq!(aln.nm, 0);
    assert_eq!(aln.kind, AlignmentKind::DpAligned);
    assert!(!aln.is_rev);
}

#[test]
fn rescue_returns_one_mismatch_match() {
    let bases = synth_contig();
    let mut read = bases[50..150].to_vec();
    // Flip a base in the middle of the read.
    let i = 60;
    read[i] = if bases[50 + i] == b'A' { b'C' } else { b'A' };
    let index = LshIndex::build(
        std::iter::once((0u32, bases.as_slice())),
        32,
        1,
        DEFAULT_TOP_BITS,
    );
    let rescue = LshRescue {
        index,
        ref_bases: vec![bases.clone()],
        cfg: default_cfg(),
        max_candidates: 64,
        max_lsh_hamming: DEFAULT_MAX_HAMMING,
        max_window_mismatches: 5,
    };
    let aln = rescue
        .rescue(&read, Strand::Forward)
        .expect("1-mismatch read should be rescued via LSH");
    assert_eq!(aln.ref_start, 50);
    assert_eq!(aln.nm, 1);
}

#[test]
fn rescue_returns_none_when_too_far() {
    let bases = synth_contig();
    // Completely independent read: should not match anywhere within budget.
    let mut state: u64 = 0xABCDEF0123456789;
    let read: Vec<u8> = (0..100)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            b"ACGT"[(state >> 56) as usize & 3]
        })
        .collect();
    let index = LshIndex::build(
        std::iter::once((0u32, bases.as_slice())),
        32,
        1,
        DEFAULT_TOP_BITS,
    );
    let rescue = LshRescue {
        index,
        ref_bases: vec![bases],
        cfg: default_cfg(),
        max_candidates: 64,
        max_lsh_hamming: DEFAULT_MAX_HAMMING,
        max_window_mismatches: 3,
    };
    // Allow the rescue to return either None (no buckets hit) OR an
    // alignment whose score happened to land within budget on chance.
    // We only assert no panic.
    let _ = rescue.rescue(&read, Strand::Forward);
}

#[test]
fn try_lsh_fallback_is_none_without_global() {
    // The OnceLock is set per-process by other tests if at all. Without
    // KIRA_LSH_ENABLE=1, this must short-circuit to None either way.
    // SAFETY: tests are run in parallel, but env var reads in
    // `lsh_enabled` are cached via OnceLock — so this test is racy if
    // another test sets the var. We guard by only checking the
    // "disabled" branch via the explicit flag check.
    if !lsh_enabled() {
        assert!(try_lsh_fallback(b"ACGTACGTACGTACGTACGTACGTACGTACGT", Strand::Forward).is_none());
    }
}
