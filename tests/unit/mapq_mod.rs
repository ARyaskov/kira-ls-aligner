use super::*;
use crate::types::{AlignmentKind, CigarKind, CigarOp, MateInfo};

fn make_aln(score: i32, read_start: u32, read_end: u32, ref_id: u32) -> Alignment {
    Alignment {
        kind: AlignmentKind::DpAligned,
        ref_id,
        ref_start: 0,
        ref_end: read_end - read_start,
        read_start,
        read_end,
        cigar: vec![CigarOp {
            len: read_end - read_start,
            op: CigarKind::Match,
        }],
        score,
        mapq: 0,
        is_rev: false,
        is_secondary: false,
        is_supplementary: false,
        nm: 0,
        md: "0".to_string(),
        as_score: score,
        xs_score: None,
        xs_strand: None,
        mate: MateInfo::default(),
    }
}

fn cfg() -> MapqConfig {
    MapqConfig {
        short_read_len: 300,
        mapq_cap_short: 60,
        mapq_cap_long: 60,
    }
}

#[test]
fn single_alignment_is_primary() {
    let mut alns = vec![make_aln(100, 0, 150, 0)];
    assign_mapq(&mut alns, 150, cfg(), None, 1);
    assert!(!alns[0].is_secondary);
    assert!(!alns[0].is_supplementary);
    assert_eq!(alns[0].mapq, 60);
}

#[test]
fn overlapping_alt_becomes_secondary() {
    // Two alignments both covering 0..150 → 100% overlap
    let mut alns = vec![make_aln(100, 0, 150, 0), make_aln(80, 0, 150, 1)];
    assign_mapq(&mut alns, 150, cfg(), None, 1);
    assert!(!alns[0].is_secondary);
    assert!(!alns[0].is_supplementary);
    assert!(alns[1].is_secondary);
    assert!(!alns[1].is_supplementary);
    assert_eq!(alns[1].mapq, 0);
}

#[test]
fn nonoverlapping_alt_becomes_supplementary() {
    // Long read 0..1000 split: primary covers 0..500, second covers 500..1000 → 0% overlap
    let mut alns = vec![make_aln(300, 0, 500, 0), make_aln(280, 500, 1000, 0)];
    assign_mapq(&mut alns, 1000, cfg(), None, 1);
    assert!(!alns[0].is_secondary && !alns[0].is_supplementary);
    assert!(alns[1].is_supplementary);
    assert!(!alns[1].is_secondary);
    assert!(alns[1].mapq > 0); // supplementary retains MAPQ
}

#[test]
fn partially_overlapping_below_50pct_is_supplementary() {
    // Primary 0..500, second 400..900 → overlap 100bp on min-len 500 → 20%
    let mut alns = vec![make_aln(300, 0, 500, 0), make_aln(280, 400, 900, 0)];
    assign_mapq(&mut alns, 900, cfg(), None, 1);
    assert!(alns[1].is_supplementary);
}

#[test]
fn partially_overlapping_above_50pct_is_secondary() {
    // Primary 0..500, second 100..500 → overlap 400bp on min-len 400 → 100%
    let mut alns = vec![make_aln(300, 0, 500, 0), make_aln(280, 100, 500, 0)];
    assign_mapq(&mut alns, 500, cfg(), None, 1);
    assert!(alns[1].is_secondary);
    assert!(!alns[1].is_supplementary);
}

#[test]
fn xs_score_proxy_grades_single_alignment() {
    let mut aln = make_aln(150, 0, 150, 0);
    aln.xs_score = Some(120);
    let mut alns = vec![aln];
    assign_mapq(&mut alns, 150, cfg(), None, 1);
    assert_eq!(alns[0].mapq, 24);
    assert_eq!(alns[0].xs_score, Some(120));
}

#[test]
fn sub_at_or_below_floor_yields_cap() {
    for sub in [0, 1, 50, 74, 75] {
        let mut aln = make_aln(150, 0, 150, 0);
        aln.xs_score = Some(sub);
        let mut alns = vec![aln];
        assign_mapq(&mut alns, 150, cfg(), None, 1);
        assert_eq!(alns[0].mapq, 60, "sub={} should keep MAPQ at cap", sub);
    }
}

#[test]
fn sub_near_parity_yields_low_mapq() {
    // Sub very close to best → genuinely ambiguous → low MAPQ.
    let mut aln = make_aln(150, 0, 150, 0);
    aln.xs_score = Some(148); // 98.7% of best
    let mut alns = vec![aln];
    assign_mapq(&mut alns, 150, cfg(), None, 1);
    // floor=75, span=75, diff=2 → mapq = 2·60/75 = 1.
    assert!(
        alns[0].mapq <= 2,
        "near-parity sub should yield very low MAPQ, got {}",
        alns[0].mapq
    );
}

#[test]
fn xs_score_proxy_at_parity_yields_zero() {
    let mut aln = make_aln(150, 0, 150, 0);
    aln.xs_score = Some(150);
    let mut alns = vec![aln];
    assign_mapq(&mut alns, 150, cfg(), None, 1);
    assert_eq!(alns[0].mapq, 0);
}

#[test]
fn weak_xs_score_proxy_keeps_near_cap() {
    let mut aln = make_aln(150, 0, 150, 0);
    aln.xs_score = Some(7); // ~5 % of best
    let mut alns = vec![aln];
    assign_mapq(&mut alns, 150, cfg(), None, 1);
    assert!(
        alns[0].mapq >= 55,
        "weak XS proxy should keep MAPQ near cap, got {}",
        alns[0].mapq
    );
}

#[test]
fn no_xs_score_keeps_cap() {
    let mut alns = vec![make_aln(150, 0, 150, 0)];
    assign_mapq(&mut alns, 150, cfg(), None, 1);
    assert_eq!(alns[0].mapq, 60);
    assert_eq!(alns[0].xs_score, None);
}

#[test]
fn opposite_strand_overlap_uses_original_query_coordinates() {
    let primary = make_aln(300, 100, 300, 0);
    let mut reverse = make_aln(280, 700, 900, 1);
    reverse.is_rev = true;
    let mut alns = vec![primary, reverse];
    assign_mapq(&mut alns, 1000, cfg(), None, 1);
    assert!(alns[1].is_secondary);
    assert!(!alns[1].is_supplementary);
}

#[test]
fn low_fastq_quality_caps_mapq() {
    let mut alns = vec![make_aln(150, 0, 150, 0)];
    let qual = vec![b'+'; 150]; // Phred Q10
    assign_mapq_with_qual(&mut alns, 150, Some(&qual), cfg(), None, 1);
    assert_eq!(alns[0].mapq, 20);
}

#[test]
fn paired_primary_lock_survives_higher_single_end_score() {
    let locked = make_aln(140, 0, 150, 0);
    let higher = make_aln(150, 0, 150, 1);
    let mut alns = vec![locked, higher];
    assign_mapq_preserving_primary(&mut alns, 150, None, cfg(), None, 1);
    assert_eq!(alns[0].ref_id, 0);
    assert!(!alns[0].is_secondary);
    assert!(alns[1].is_secondary);
}

#[test]
fn low_seed_copy_count_does_not_destroy_unique_mapq() {
    let mut alns = vec![make_aln(150, 0, 150, 0)];
    assign_mapq(&mut alns, 150, cfg(), None, 2);
    assert_eq!(alns[0].mapq, 60);
}

#[test]
fn highly_repetitive_seed_set_caps_mapq() {
    let mut alns = vec![make_aln(150, 0, 150, 0)];
    assign_mapq(&mut alns, 150, cfg(), None, 200);
    assert_eq!(alns[0].mapq, 3);
}

fn pair_ctx() -> PairMapqContext {
    PairMapqContext {
        insert_mean: 570,
        insert_sd: 155,
        discordant_cap: 10,
    }
}

fn paired_mate(
    ref_id: u32,
    mate_pos: u32,
    tlen: i32,
    is_proper: bool,
    mate_is_unmapped: bool,
) -> crate::types::MateInfo {
    crate::types::MateInfo {
        is_paired: true,
        is_proper_pair: is_proper,
        mate_is_unmapped,
        mate_is_rev: false,
        is_first_in_pair: true,
        is_second_in_pair: false,
        mate_ref_id: Some(ref_id),
        mate_pos,
        tlen,
    }
}

#[test]
fn discordant_pair_capped_at_discordant_cap() {
    // Proper-pair flag off + |TLEN - mean| > 3σ → cap MAPQ at 10.
    let mut aln = make_aln(150, 0, 150, 0);
    aln.mate = paired_mate(0, 50_000, 50_000, false, false);
    let mut alns = vec![aln];
    assign_mapq(&mut alns, 150, cfg(), Some(pair_ctx()), 1);
    // Would have been MQ 60 without pair context.
    assert_eq!(alns[0].mapq, 10);
}

#[test]
fn proper_pair_not_capped() {
    // proper_pair = true → discordant cap is inactive even with no sd.
    let mut aln = make_aln(150, 0, 150, 0);
    aln.mate = paired_mate(0, 600, 600, true, false);
    let mut alns = vec![aln];
    assign_mapq(&mut alns, 150, cfg(), Some(pair_ctx()), 1);
    assert_eq!(alns[0].mapq, 60);
}

#[test]
fn in_window_pair_not_capped_even_if_proper_flag_missing() {
    let mut aln = make_aln(150, 0, 150, 0);
    // TLEN 600, mean 570, sd 155 → deviation 30, well inside 3σ = 465.
    aln.mate = paired_mate(0, 600, 600, false, false);
    let mut alns = vec![aln];
    assign_mapq(&mut alns, 150, cfg(), Some(pair_ctx()), 1);
    assert_eq!(alns[0].mapq, 60);
}

#[test]
fn unmapped_mate_not_capped() {
    let mut aln = make_aln(150, 0, 150, 0);
    aln.mate = paired_mate(0, 0, 0, false, true);
    let mut alns = vec![aln];
    assign_mapq(&mut alns, 150, cfg(), Some(pair_ctx()), 1);
    assert_eq!(alns[0].mapq, 60);
}

#[test]
fn cross_chr_mate_not_capped_here() {
    let mut aln = make_aln(150, 0, 150, 0);
    // ref_id=0 on the alignment, mate on ref_id=2 — cross-chr.
    aln.mate = paired_mate(2, 0, 0, false, false);
    let mut alns = vec![aln];
    assign_mapq(&mut alns, 150, cfg(), Some(pair_ctx()), 1);
    assert_eq!(alns[0].mapq, 60);
}

#[test]
fn supplementary_inherits_discordant_cap() {
    let mut primary = make_aln(300, 0, 500, 0);
    primary.mate = paired_mate(0, 50_000, 50_000, false, false);
    let mut supp = make_aln(50, 500, 1000, 0);
    supp.mate = paired_mate(0, 50_000, 50_000, false, false);
    let mut alns = vec![primary, supp];
    assign_mapq(&mut alns, 1000, cfg(), Some(pair_ctx()), 1);
    assert_eq!(alns[0].mapq, 10);
    assert!(alns[1].is_supplementary);
    // Supp gets its own MAPQ (compute_mapq(50, 0, 60) = 60), capped to 10.
    assert_eq!(alns[1].mapq, 10);
}

#[test]
fn real_second_alignment_beats_xs_proxy() {
    let mut primary = make_aln(150, 0, 150, 0);
    primary.xs_score = Some(10); // stale proxy
    let secondary = make_aln(120, 0, 150, 1);
    let mut alns = vec![primary, secondary];
    assign_mapq(&mut alns, 150, cfg(), None, 1);
    assert_eq!(alns[0].mapq, 24);
    assert_eq!(alns[0].xs_score, Some(120));
}

#[test]
fn supplementary_does_not_deflate_primary_mapq() {
    let primary = make_aln(80, 0, 80, 0);
    let supp = make_aln(70, 80, 150, 1);
    let mut alns = vec![primary, supp];
    assign_mapq(&mut alns, 150, cfg(), None, 1);
    assert_eq!(
        alns[0].mapq, 60,
        "primary keeps cap when only competition is a disjoint supp"
    );
    assert!(alns[1].is_supplementary);
}

#[test]
fn coregion_secondary_still_deflates_primary_mapq() {
    let primary = make_aln(150, 0, 150, 0);
    let competitor = make_aln(120, 0, 150, 1); // 100% overlap
    let mut alns = vec![primary, competitor];
    assign_mapq(&mut alns, 150, cfg(), None, 1);
    // Same as `real_second_alignment_beats_xs_proxy`: mapq = 24.
    assert_eq!(alns[0].mapq, 24);
    assert!(alns[1].is_secondary);
}
