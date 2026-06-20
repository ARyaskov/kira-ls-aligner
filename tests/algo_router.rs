//! Integration tests for `alignment::router` — read-length-driven aligner
//! selection and budget calculations.
//!
//! These tests assume defaults (no env-var overrides). The OnceLock cache
//! inside the router will capture stale KIRA_* values if the test process
//! has them set — keep CI clean.

use kira_ls_aligner::alignment::router::{
    AlignerKind, choose_aligner, myers_bound_floor, myers_reject_bound, wfa_score_budget,
};
use kira_ls_aligner::alignment::{
    AlignmentConfig, AnchorSpan, FastPathKind, try_fast_dp_alignment,
};
use kira_ls_aligner::types::Strand;

#[test]
fn short_reads_get_fast_path() {
    for len in [50usize, 150, 300] {
        let kind = choose_aligner(len);
        assert!(
            matches!(
                kind,
                AlignerKind::PackedSpectral | AlignerKind::SpectralSieve | AlignerKind::Wfa
            ),
            "unexpected short-read kind: {kind:?} for len={len}",
        );
    }
}

#[test]
fn long_reads_get_banded_sw() {
    assert!(matches!(choose_aligner(301), AlignerKind::BandedSw));
    assert!(matches!(choose_aligner(1_000), AlignerKind::BandedSw));
    assert!(matches!(choose_aligner(10_000), AlignerKind::BandedSw));
}

#[test]
fn wfa_budget_grows_with_length() {
    let b1 = wfa_score_budget(50, 4, 6, 1);
    let b2 = wfa_score_budget(500, 4, 6, 1);
    assert!(b2 >= b1, "budget should not shrink for longer reads");
    assert!(b1 > 0);
}

#[test]
fn myers_bound_reasonable() {
    assert!(myers_reject_bound(150) >= 4);
    assert!(myers_reject_bound(10) >= myers_bound_floor() as usize);
}

/// Build a 150 bp non-repetitive synthetic reference + chain anchor span
/// at offset 2 (so the read sits at ref[2..152]). The 2 bp padding gives
/// the fast-path window the slack it expects.
fn synthetic_bases(len: usize) -> Vec<u8> {
    let mut state = 0x9e37_79b9_u32;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            b"ACGT"[((state >> 30) & 3) as usize]
        })
        .collect()
}

fn synthetic_fixture(read: Vec<u8>) -> (Vec<u8>, Vec<u8>, AnchorSpan) {
    let base = synthetic_bases(200);
    let mut reference: Vec<u8> = b"TT".to_vec();
    reference.extend_from_slice(&base);
    reference.extend_from_slice(b"TT");
    let span = AnchorSpan {
        ref_id: 0,
        ref_start: 2,
        ref_end: 2 + read.len() as u32,
        read_start: 0,
        read_end: read.len() as u32,
        strand: Strand::Forward,
    };
    (reference, read, span)
}

fn default_cfg() -> AlignmentConfig {
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

/// All-matches 150 bp read must resolve via the packed-spectral path
/// (KIRA_ALGO default = `packed`). This is the dominant case for clean
/// short reads and the only one that gets the SIMD popcount fast path.
#[test]
fn exact_match_resolves_via_packed_spectral() {
    let base = synthetic_bases(200);
    let read = base[..150].to_vec();
    let (reference, read, span) = synthetic_fixture(read);
    let cfg = default_cfg();

    let result = try_fast_dp_alignment(&read, &reference, &span, 150, cfg, false);
    let (_aln, kind) = result.expect("fast path should resolve a clean 150bp match");
    assert_eq!(
        kind,
        FastPathKind::PackedSpectral,
        "all-match 150bp read should hit the packed spectral path"
    );
}

/// A 150 bp read with a single-base insertion at position 75 cannot be
/// resolved by either spectral path (they're ungapped only) but WFA
/// finds it within the score budget. The returned `FastPathKind` must
/// be `Wfa` so the counter attributes correctly.
#[test]
fn single_insertion_resolves_via_wfa() {
    let base = synthetic_bases(200);
    // Read = ref[0..75] + 'G' (insertion) + ref[75..149]. Length stays 150.
    let mut read: Vec<u8> = base[..75].to_vec();
    read.push(b'G');
    read.extend_from_slice(&base[75..149]);
    assert_eq!(read.len(), 150);

    let (reference, read, span) = synthetic_fixture(read);
    let cfg = default_cfg();

    let result = try_fast_dp_alignment(&read, &reference, &span, 150, cfg, false);
    let (_aln, kind) = result.expect("WFA should resolve a single-bp insertion");
    assert_eq!(
        kind,
        FastPathKind::Wfa,
        "1bp insertion should fall through spectral and resolve via WFA"
    );
}

/// A read with too many mismatches AND no realistic ungapped position
/// should be rejected by every fast path (Myers reject bound stops WFA),
/// returning `None`. The caller then falls through to banded SW.
#[test]
fn hopeless_read_returns_none() {
    // Random-looking read with no relation to the reference.
    let read: Vec<u8> = (0..150).map(|i| b"NNNNNNNNN"[i % 9]).collect();
    let (reference, read, span) = synthetic_fixture(read);
    let cfg = default_cfg();

    let result = try_fast_dp_alignment(&read, &reference, &span, 150, cfg, false);
    assert!(
        result.is_none(),
        "all-N read against ACGT reference should not resolve in any fast path"
    );
}
