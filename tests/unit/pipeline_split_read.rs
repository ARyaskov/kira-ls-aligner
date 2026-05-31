use super::*;
use crate::types::{AlignmentKind, CigarKind, CigarOp, MateInfo, Strand};

fn make_chain(score: i32, read_start: u32, read_end: u32, ref_id: u32) -> Chain {
    Chain {
        anchors: Vec::new(),
        score,
        ref_id,
        read_start,
        read_end,
        ref_start: 1000,
        ref_end: 1000 + (read_end - read_start),
        strand: Strand::Forward,
    }
}

fn make_aln(read_start: u32, read_end: u32) -> Alignment {
    Alignment {
        kind: AlignmentKind::DpAligned,
        ref_id: 0,
        ref_start: 1000,
        ref_end: 1000 + (read_end - read_start),
        read_start,
        read_end,
        cigar: vec![CigarOp {
            len: read_end - read_start,
            op: CigarKind::Match,
        }],
        score: 100,
        mapq: 60,
        is_rev: false,
        is_secondary: false,
        is_supplementary: false,
        nm: 0,
        md: "0".to_string(),
        as_score: 100,
        xs_score: None,
        xs_strand: None,
        mate: MateInfo::default(),
    }
}

#[test]
fn picks_disjoint_high_score_chain_as_supp_candidate() {
    let primary = make_aln(0, 80);
    let chains = vec![
        make_chain(100, 0, 80, 0),
        make_chain(70, 80, 150, 1),
    ];
    let picks = pick_supplementary_chains(&primary, &chains);
    assert_eq!(picks.len(), 1);
    assert_eq!(picks[0].read_start, 80);
}

#[test]
fn rejects_overlapping_chain() {
    let primary = make_aln(0, 150);
    let chains = vec![
        make_chain(100, 0, 150, 0),
        make_chain(90, 10, 150, 1),
    ];
    let picks = pick_supplementary_chains(&primary, &chains);
    assert!(picks.is_empty());
}

#[test]
fn rejects_low_score_chain() {
    // Disjoint but score < 50 % of primary → noise, skip.
    let primary = make_aln(0, 80);
    let chains = vec![
        make_chain(100, 0, 80, 0),
        make_chain(30, 80, 150, 1), // 0.3 × primary
    ];
    let picks = pick_supplementary_chains(&primary, &chains);
    assert!(picks.is_empty());
}

#[test]
fn rejects_short_chain() {
    // Disjoint, high-score, but span 30 bp < 50 bp threshold.
    let primary = make_aln(0, 80);
    let chains = vec![
        make_chain(100, 0, 80, 0),
        make_chain(80, 100, 130, 1),
    ];
    let picks = pick_supplementary_chains(&primary, &chains);
    assert!(picks.is_empty());
}

#[test]
fn dedups_clustered_chains_in_same_region() {
    // Two near-identical chains at read[80..150] — pick only one.
    let primary = make_aln(0, 80);
    let chains = vec![
        make_chain(100, 0, 80, 0),
        make_chain(80, 80, 150, 1),
        make_chain(75, 82, 150, 2),
    ];
    let picks = pick_supplementary_chains(&primary, &chains);
    assert_eq!(picks.len(), 1);
    assert_eq!(picks[0].ref_id, 1, "first qualifying candidate wins");
}

#[test]
fn caps_at_max_supp_per_read() {
    let primary = make_aln(0, 200);
    let mut chains = vec![make_chain(200, 0, 200, 0)];
    for k in 0..6u32 {
        let start = 200 + k * 60;
        chains.push(make_chain(150, start, start + 55, k + 1));
    }
    let picks = pick_supplementary_chains(&primary, &chains);
    assert_eq!(picks.len(), MAX_SUPP_PER_READ);
}

#[test]
fn skips_when_no_second_chain() {
    // Sole chain → nothing to do.
    let primary = make_aln(0, 80);
    let chains = vec![make_chain(100, 0, 80, 0)];
    let picks = pick_supplementary_chains(&primary, &chains);
    assert!(picks.is_empty());
}

#[test]
fn overlap_pct_basic_cases() {
    // Disjoint → 0.
    assert_eq!(read_overlap_pct(0, 50, 50, 100), 0);
    // Identical → 100.
    assert_eq!(read_overlap_pct(0, 100, 0, 100), 100);
    // Smaller `b` fully inside larger `a` → share of `a` (50/100).
    assert_eq!(read_overlap_pct(0, 100, 50, 100), 50);
    assert_eq!(read_overlap_pct(0, 100, 50, 150), 50);
    // Larger `b` fully covers smaller `a` → 100 % of `a`.
    assert_eq!(read_overlap_pct(50, 100, 0, 100), 100);
    // Empty interval → 0.
    assert_eq!(read_overlap_pct(50, 50, 0, 100), 0);
}
