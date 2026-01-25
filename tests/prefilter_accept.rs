use kira_ls_aligner::alignment::AlignmentConfig;
use kira_ls_aligner::alignment::AnchorSpan;
use kira_ls_aligner::alignment::prefilter::{PrefilterReason, PrefilterResult, prefilter_chain};
use kira_ls_aligner::types::Strand;

fn cfg() -> AlignmentConfig {
    AlignmentConfig {
        match_score: 1,
        mismatch: 4,
        gap_open: 6,
        gap_extend: 1,
        bandwidth: 50,
        xdrop: 50,
    }
}

#[test]
fn prefilter_accepts_short_top1_low_mism() {
    let mut read = b"ACGT".repeat(37); // 148 bp
    read.extend_from_slice(b"AC"); // 150 bp
    let mut ref_seq = read.clone();
    ref_seq[10] = b'T'; // 1 mismatch
    let chain = AnchorSpan {
        ref_id: 0,
        ref_start: 0,
        ref_end: read.len() as u32,
        read_start: 0,
        read_end: read.len() as u32,
        strand: Strand::Forward,
    };

    let outcome = prefilter_chain(
        &read,
        &ref_seq,
        &chain,
        cfg(),
        true,
        true,
        true,
        5,
        98.0,
        2,
        0,
        i32::MAX,
        300,
        true,
        false,
    );

    match outcome.result {
        PrefilterResult::Accept(_) => {}
        _ => panic!("expected accept for short top1 low mism"),
    }
    assert_eq!(outcome.reason, PrefilterReason::Accepted);
}

#[test]
fn prefilter_rejects_short_span() {
    let read = b"A".repeat(400);
    let ref_seq = b"T".repeat(400);
    let chain = AnchorSpan {
        ref_id: 0,
        ref_start: 0,
        ref_end: 44,
        read_start: 0,
        read_end: 44,
        strand: Strand::Forward,
    };

    let outcome = prefilter_chain(
        &read,
        &ref_seq,
        &chain,
        cfg(),
        true,
        true,
        true,
        5,
        98.0,
        2,
        0,
        i32::MAX,
        300,
        true,
        false,
    );

    match outcome.result {
        PrefilterResult::Reject => {}
        _ => panic!("expected reject for short span"),
    }
    assert_eq!(outcome.reason, PrefilterReason::SpanTooSmall);
}

#[test]
fn prefilter_not_top1_fallback() {
    let read = b"A".repeat(400);
    let ref_seq = b"T".repeat(400);
    let chain = AnchorSpan {
        ref_id: 0,
        ref_start: 0,
        ref_end: 400,
        read_start: 0,
        read_end: 400,
        strand: Strand::Forward,
    };

    let outcome = prefilter_chain(
        &read,
        &ref_seq,
        &chain,
        cfg(),
        false,
        true,
        true,
        5,
        98.0,
        2,
        0,
        i32::MAX,
        300,
        true,
        false,
    );

    assert!(matches!(outcome.result, PrefilterResult::Fallback));
    assert_eq!(outcome.reason, PrefilterReason::NotTop1);
}

#[test]
fn prefilter_mism_too_high() {
    let read = b"A".repeat(150);
    let ref_seq = b"T".repeat(150);
    let chain = AnchorSpan {
        ref_id: 0,
        ref_start: 0,
        ref_end: 150,
        read_start: 0,
        read_end: 150,
        strand: Strand::Forward,
    };

    let outcome = prefilter_chain(
        &read,
        &ref_seq,
        &chain,
        cfg(),
        true,
        true,
        true,
        5,
        98.0,
        2,
        0,
        i32::MAX,
        300,
        true,
        false,
    );

    assert!(matches!(outcome.result, PrefilterResult::Fallback));
    assert_eq!(outcome.reason, PrefilterReason::MismTooHigh);
}

#[test]
fn prefilter_not_short_preset() {
    let read = b"A".repeat(400);
    let ref_seq = b"T".repeat(400);
    let chain = AnchorSpan {
        ref_id: 0,
        ref_start: 0,
        ref_end: 400,
        read_start: 0,
        read_end: 400,
        strand: Strand::Forward,
    };

    let outcome = prefilter_chain(
        &read,
        &ref_seq,
        &chain,
        cfg(),
        true,
        true,
        true,
        5,
        98.0,
        2,
        0,
        i32::MAX,
        300,
        false,
        false,
    );

    assert!(matches!(outcome.result, PrefilterResult::Fallback));
    assert_eq!(outcome.reason, PrefilterReason::NotShortPreset);
}
