//! Regression tests for the MD tag generator used in the prefilter
//! `build_ungapped_alignment` path.
//!
//! We replicate the canonical MD construction from the SAMtools spec and
//! compare it to whatever `build_ungapped_alignment` produces. This guards
//! against silent breakage of the fast `push_u32_decimal` path.

use kira_ls_aligner::alignment::AlignmentConfig;
use kira_ls_aligner::alignment::AnchorSpan;
use kira_ls_aligner::alignment::prefilter::{
    PrefilterResult, prefilter_chain,
};
use kira_ls_aligner::types::Strand;

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

/// Canonical MD/NM construction for an ungapped (no-indel) alignment.
fn expected_md_and_nm(read: &[u8], reference: &[u8]) -> (String, u32) {
    assert_eq!(read.len(), reference.len());
    let mut md = String::new();
    let mut run: u32 = 0;
    let mut nm: u32 = 0;
    for (&q, &r) in read.iter().zip(reference.iter()) {
        if q == r {
            run += 1;
        } else {
            nm += 1;
            md.push_str(&run.to_string());
            md.push(r as char);
            run = 0;
        }
    }
    md.push_str(&run.to_string());
    (md, nm)
}

fn accept_with(read: &[u8], reference: &[u8]) -> (String, u32) {
    let chain = AnchorSpan {
        ref_id: 0,
        ref_start: 0,
        ref_end: read.len() as u32,
        read_start: 0,
        read_end: read.len() as u32,
        strand: Strand::Forward,
    };
    let outcome = prefilter_chain(
        read,
        reference,
        &chain,
        cfg(),
        /* is_top1 */ true,
        /* accept_enable */ true,
        /* accept_only_top1 */ true,
        /* accept_span_slack */ 15,
        /* accept_min_identity */ 98.0,
        /* accept_max_mismatches */ 5,
        /* accept_require_score_margin */ 0,
        /* score_margin */ i32::MAX,
        /* long_read_threshold */ 300,
        /* short_read */ true,
        /* multi_alignments_enabled */ false,
    );
    match outcome.result {
        PrefilterResult::Accept(aln) => (aln.md, aln.nm),
        other => panic!("expected Accept, got {:?}", other),
    }
}

#[test]
fn md_matches_canonical_full_match() {
    let read = b"ACGT".repeat(37); // 148 bp
    let mut read = read;
    read.extend_from_slice(b"AC");
    let reference = read.clone();
    let (expected_md, expected_nm) = expected_md_and_nm(&read, &reference);
    let (md, nm) = accept_with(&read, &reference);
    assert_eq!(md, expected_md);
    assert_eq!(nm, expected_nm);
    assert_eq!(md, "150");
    assert_eq!(nm, 0);
}

#[test]
fn md_matches_canonical_single_mismatch() {
    let read = b"ACGT".repeat(37);
    let mut read = read;
    read.extend_from_slice(b"AC");
    let mut reference = read.clone();
    reference[10] = b'T'; // 1 mismatch at position 10
    let (expected_md, expected_nm) = expected_md_and_nm(&read, &reference);
    let (md, nm) = accept_with(&read, &reference);
    assert_eq!(md, expected_md);
    assert_eq!(nm, expected_nm);
    assert_eq!(nm, 1);
}

#[test]
fn md_matches_canonical_multiple_mismatches() {
    // 2 mismatches keeps identity ≥ 98.7%, above the prefilter floor (98.5%).
    let read = b"ACGT".repeat(37);
    let mut read = read;
    read.extend_from_slice(b"AC");
    let mut reference = read.clone();
    reference[5] = b'T';
    reference[99] = b'A';
    let (expected_md, expected_nm) = expected_md_and_nm(&read, &reference);
    let (md, nm) = accept_with(&read, &reference);
    assert_eq!(md, expected_md);
    assert_eq!(nm, expected_nm);
    assert_eq!(nm, 2);
}

#[test]
fn md_matches_canonical_trailing_mismatch() {
    let read = b"ACGT".repeat(37);
    let mut read = read;
    read.extend_from_slice(b"AC");
    let mut reference = read.clone();
    let last = reference.len() - 1;
    reference[last] = if reference[last] == b'A' { b'T' } else { b'A' };
    let (expected_md, expected_nm) = expected_md_and_nm(&read, &reference);
    let (md, nm) = accept_with(&read, &reference);
    assert_eq!(md, expected_md);
    assert_eq!(nm, expected_nm);
    assert_eq!(nm, 1);
}

#[test]
fn md_matches_canonical_leading_mismatch() {
    let read = b"ACGT".repeat(37);
    let mut read = read;
    read.extend_from_slice(b"AC");
    let mut reference = read.clone();
    reference[0] = if reference[0] == b'A' { b'T' } else { b'A' };
    let (expected_md, expected_nm) = expected_md_and_nm(&read, &reference);
    let (md, nm) = accept_with(&read, &reference);
    assert_eq!(md, expected_md);
    assert_eq!(nm, expected_nm);
    assert_eq!(nm, 1);
}
