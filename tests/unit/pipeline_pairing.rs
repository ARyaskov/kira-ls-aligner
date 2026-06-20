use super::*;

fn dummy_aln(ref_id: u32, ref_start: u32, ref_end: u32, is_rev: bool) -> Alignment {
    Alignment {
        kind: crate::types::AlignmentKind::AcceptedUngapped,
        ref_id,
        ref_start,
        ref_end,
        read_start: 0,
        read_end: ref_end - ref_start,
        cigar: vec![crate::types::CigarOp {
            len: ref_end - ref_start,
            op: crate::types::CigarKind::Match,
        }],
        score: 100,
        mapq: 60,
        is_rev,
        is_secondary: false,
        is_supplementary: false,
        nm: 0,
        md: format!("{}", ref_end - ref_start),
        as_score: 100,
        xs_score: None,
        xs_strand: None,
        mate: MateInfo::default(),
    }
}

fn dummy_read(id: &str, role: PairRole) -> ReadRecord {
    ReadRecord {
        id: id.to_string(),
        seq: vec![b'A'; 150],
        qual: None,
        pair_role: role,
        repeat_min_occ: 1,
    }
}

#[test]
fn insert_spec_parses_2_and_4_field_forms() {
    let mut cfg = PairedConfig::default();
    cfg.apply_insert_spec("50,500").unwrap();
    assert_eq!((cfg.insert_min, cfg.insert_max), (50, 500));

    cfg.apply_insert_spec("0,1500,300,40").unwrap();
    assert_eq!(
        (
            cfg.insert_min,
            cfg.insert_max,
            cfg.insert_mean,
            cfg.insert_sd
        ),
        (0, 1500, 300, 40)
    );

    assert!(cfg.apply_insert_spec("100").is_err());
    assert!(cfg.apply_insert_spec("500,100").is_err()); // min > max
}

#[test]
fn proper_pair_fr_orientation_in_window() {
    let mut cfg = PairedConfig::default();
    cfg.mode = IngestMode::TwoFile;
    // R1 forward at 100, R2 reverse at 350 → fragment 250..500, len 400
    let reads = vec![
        dummy_read("frag1", PairRole::R1),
        dummy_read("frag1", PairRole::R2),
    ];
    let mut alns = vec![
        vec![dummy_aln(0, 100, 250, false)],
        vec![dummy_aln(0, 350, 500, true)],
    ];
    let mut umi = vec![None; reads.len()];
    apply_pairing(&reads, &mut alns, &mut umi, &cfg);
    // R1 leftmost → tlen +400
    assert!(alns[0][0].mate.is_paired);
    assert!(alns[0][0].mate.is_proper_pair);
    assert_eq!(alns[0][0].mate.tlen, 400);
    assert!(alns[0][0].mate.is_first_in_pair);
    // R2 rightmost → tlen -400
    assert_eq!(alns[1][0].mate.tlen, -400);
    assert!(alns[1][0].mate.is_second_in_pair);
    assert!(alns[1][0].mate.is_proper_pair);
}

#[test]
fn not_proper_when_different_refs() {
    let mut cfg = PairedConfig::default();
    cfg.mode = IngestMode::TwoFile;
    let reads = vec![dummy_read("x", PairRole::R1), dummy_read("x", PairRole::R2)];
    let mut alns = vec![
        vec![dummy_aln(0, 100, 250, false)],
        vec![dummy_aln(1, 100, 250, true)],
    ];
    let mut umi = vec![None; reads.len()];
    apply_pairing(&reads, &mut alns, &mut umi, &cfg);
    assert!(alns[0][0].mate.is_paired);
    assert!(!alns[0][0].mate.is_proper_pair);
    assert_eq!(alns[0][0].mate.tlen, 0);
}

#[test]
fn mate_unmapped_marks_correct_bit() {
    let mut cfg = PairedConfig::default();
    cfg.mode = IngestMode::TwoFile;
    let reads = vec![dummy_read("x", PairRole::R1), dummy_read("x", PairRole::R2)];
    let mut alns = vec![vec![dummy_aln(0, 100, 250, false)], vec![]];
    let mut umi = vec![None; reads.len()];
    apply_pairing(&reads, &mut alns, &mut umi, &cfg);
    // R1's mate is unmapped → 0x8 should be set on R1
    assert!(alns[0][0].mate.is_paired);
    assert!(alns[0][0].mate.mate_is_unmapped);
    assert!(!alns[0][0].mate.is_proper_pair);
    assert_eq!(alns[0][0].mate.mate_ref_id, None);
}

#[test]
fn unpaired_mode_is_noop() {
    let cfg = PairedConfig::default(); // mode = Unpaired
    let reads = vec![dummy_read("r1", PairRole::Unpaired)];
    let mut alns = vec![vec![dummy_aln(0, 0, 100, false)]];
    let mut umi = vec![None; reads.len()];
    apply_pairing(&reads, &mut alns, &mut umi, &cfg);
    assert!(!alns[0][0].mate.is_paired);
}

fn dummy_aln_score(
    ref_id: u32,
    ref_start: u32,
    ref_end: u32,
    is_rev: bool,
    score: i32,
) -> Alignment {
    let mut a = dummy_aln(ref_id, ref_start, ref_end, is_rev);
    a.score = score;
    a
}

#[test]
fn rerank_keeps_concordant_pair_at_slot_0() {
    let mut cfg = PairedConfig::default();
    cfg.mode = IngestMode::TwoFile;
    cfg.insert_mean = 400;
    cfg.insert_sd = 100;
    let reads = vec![dummy_read("r", PairRole::R1), dummy_read("r", PairRole::R2)];
    let mut alns = vec![
        vec![
            dummy_aln_score(0, 100, 250, false, 150),
            dummy_aln_score(1, 1000, 1150, false, 100),
        ],
        vec![dummy_aln_score(0, 350, 500, true, 145)],
    ];
    pair_rerank(&reads, &mut alns, &cfg, 5);
    // Slot 0 of R1 stays on chr0. Score was boosted by 30/2=15.
    assert_eq!(alns[0][0].ref_id, 0);
    assert!(alns[0][0].score >= 150);
}

#[test]
fn rerank_promotes_pair_consistent_alternative() {
    let mut cfg = PairedConfig::default();
    cfg.mode = IngestMode::TwoFile;
    cfg.insert_mean = 400;
    cfg.insert_sd = 100;
    let reads = vec![dummy_read("r", PairRole::R1), dummy_read("r", PairRole::R2)];
    let mut alns = vec![
        vec![
            dummy_aln_score(1, 50_000, 50_150, false, 200),
            dummy_aln_score(0, 100, 250, false, 180),
        ],
        vec![dummy_aln_score(0, 350, 500, true, 140)],
    ];
    pair_rerank(&reads, &mut alns, &cfg, 5);
    let chr0_score = alns[0]
        .iter()
        .find(|a| a.ref_id == 0)
        .map(|a| a.score)
        .unwrap();
    let chr1_score = alns[0]
        .iter()
        .find(|a| a.ref_id == 1)
        .map(|a| a.score)
        .unwrap();
    assert!(
        chr0_score > chr1_score,
        "concordant alt (chr0={chr0_score}) didn't overtake lone primary (chr1={chr1_score})"
    );
}

#[test]
fn rerank_no_bonus_when_all_combos_discordant() {
    let mut cfg = PairedConfig::default();
    cfg.mode = IngestMode::TwoFile;
    cfg.insert_mean = 400;
    cfg.insert_sd = 100;
    let reads = vec![dummy_read("r", PairRole::R1), dummy_read("r", PairRole::R2)];
    let r1_orig = 150;
    let r2_orig = 140;
    let mut alns = vec![
        vec![dummy_aln_score(0, 100, 250, false, r1_orig)],
        vec![dummy_aln_score(0, 100_000, 100_150, true, r2_orig)],
    ];
    pair_rerank(&reads, &mut alns, &cfg, 5);
    assert_eq!(alns[0][0].score, r1_orig);
    assert_eq!(alns[1][0].score, r2_orig);
}

#[test]
fn pair_is_concordant_respects_3sigma() {
    let mut cfg = PairedConfig::default();
    cfg.insert_mean = 500;
    cfg.insert_sd = 100;
    cfg.estimator_locked = true;
    // In-window: TLEN 600
    let a = MatePrimaryScored {
        ref_id: 0,
        ref_start: 0,
        ref_end: 150,
        is_rev: false,
        score: 100,
    };
    let b = MatePrimaryScored {
        ref_id: 0,
        ref_start: 450,
        ref_end: 600,
        is_rev: true,
        score: 100,
    };
    assert!(pair_is_concordant(&a, &b, &cfg));
    // Out of window: TLEN 1000 (5σ)
    let b_far = MatePrimaryScored {
        ref_start: 850,
        ref_end: 1000,
        ..b
    };
    assert!(!pair_is_concordant(&a, &b_far, &cfg));
    // Cross-chr
    let b_chr = MatePrimaryScored { ref_id: 1, ..b };
    assert!(!pair_is_concordant(&a, &b_chr, &cfg));
    // Wrong orientation (RR)
    let b_rr = MatePrimaryScored { is_rev: false, ..b };
    assert!(!pair_is_concordant(&a, &b_rr, &cfg));
}

#[test]
fn concordance_window_falls_back_to_min_max_during_bootstrap() {
    let mut cfg = PairedConfig::default();
    cfg.insert_min = 0;
    cfg.insert_max = 1000;
    cfg.insert_mean = 200;
    cfg.insert_sd = 50;
    cfg.estimator_locked = false;
    let (lo, hi) = concordance_window(&cfg);
    assert_eq!((lo, hi), (0, 1000));

    cfg.insert_mean = 570;
    cfg.insert_sd = 155;
    cfg.estimator_locked = true;
    let (lo, hi) = concordance_window(&cfg);
    assert_eq!((lo, hi), (570 - 3 * 155, 570 + 3 * 155));
}

#[test]
fn proper_pair_window_uses_5sigma_after_lock() {
    let mut cfg = PairedConfig::default();
    cfg.insert_mean = 570;
    cfg.insert_sd = 155;
    cfg.estimator_locked = true;
    let (lo, hi) = proper_pair_window(&cfg);
    assert_eq!((lo, hi), (0, 570 + 5 * 155));
    // Bootstrap fallback is still [insert_min, insert_max].
    cfg.estimator_locked = false;
    cfg.insert_min = 0;
    cfg.insert_max = 1000;
    assert_eq!(proper_pair_window(&cfg), (0, 1000));
}

#[test]
fn classify_pair_marks_out_of_5sigma_pair_improper_after_lock() {
    let mut cfg = PairedConfig::default();
    cfg.mode = IngestMode::TwoFile;
    cfg.insert_mean = 500;
    cfg.insert_sd = 100;
    cfg.insert_min = 0;
    cfg.insert_max = 5_000; // legacy field set wide on purpose
    cfg.estimator_locked = true;

    // TLEN 1100 = 6 σ from mean = 500
    let p1 = MatePrimary {
        ref_id: 0,
        ref_start: 0,
        ref_end: 150,
        is_rev: false,
    };
    let p2 = MatePrimary {
        ref_id: 0,
        ref_start: 950,
        ref_end: 1100,
        is_rev: true,
    };
    let (proper, _, _) = classify_pair(&p1, &p2, &cfg);
    assert!(
        !proper,
        "TLEN 1100 = 6 σ from mean 500 must be classified discordant post-lock"
    );

    // TLEN 900 = 4 σ → inside 5 σ → proper.
    let p2_in = MatePrimary {
        ref_id: 0,
        ref_start: 750,
        ref_end: 900,
        is_rev: true,
    };
    let (proper_in, _, _) = classify_pair(&p1, &p2_in, &cfg);
    assert!(proper_in, "TLEN 900 = 4 σ from mean must stay proper");
}
