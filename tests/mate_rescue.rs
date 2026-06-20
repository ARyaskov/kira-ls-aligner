//! Mate-rescue integration: feed a synthetic pair where one mate has a
//! mapped primary and the other has zero alignments. After
//! `rescue_unmapped_mates`, the previously-empty side should carry a
//! banded-SW alignment landing inside the insert window.

use kira_ls_aligner::alignment::AlignmentConfig;
use kira_ls_aligner::index::{Index, IndexConfig};
use kira_ls_aligner::io::IngestMode;
use kira_ls_aligner::pipeline::pairing::{PairedConfig, RescueConfig, rescue_unmapped_mates};
use kira_ls_aligner::types::{
    Alignment, AlignmentKind, CigarKind, CigarOp, MateInfo, PairRole, ReadRecord, RefBases, RefSeq,
    Reference,
};

/// Build a deterministic 4 kb reference whose first 2 kb is repeat-free
/// random-looking ACGT so unique anchoring + rescue both have something
/// to work with.
fn synth_reference() -> Reference {
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut buf = Vec::with_capacity(4000);
    let bases = b"ACGT";
    for _ in 0..4000 {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        buf.push(bases[(seed >> 33) as usize & 3]);
    }
    Reference {
        sequences: vec![RefSeq {
            name: "synth".to_string(),
            bases: RefBases::Owned(buf),
        }],
    }
}

fn build_index(reference: &Reference) -> Index {
    Index::build(
        reference.clone(),
        IndexConfig {
            short_k: 19,
            short_w: 10,
            long_k: 19,
            long_w: 10,
            max_occ: 500,
            build_short: true,
            build_long: false,
        },
    )
}

fn mk_paired_read(id: &str, seq: Vec<u8>, role: PairRole) -> ReadRecord {
    ReadRecord {
        id: id.to_string(),
        seq,
        qual: None,
        pair_role: role,
        repeat_min_occ: 1,
    }
}

fn mk_aln(ref_id: u32, ref_start: u32, ref_end: u32, is_rev: bool, score: i32) -> Alignment {
    Alignment {
        kind: AlignmentKind::DpAligned,
        ref_id,
        ref_start,
        ref_end,
        read_start: 0,
        read_end: ref_end - ref_start,
        cigar: vec![CigarOp {
            len: ref_end - ref_start,
            op: CigarKind::Match,
        }],
        score,
        mapq: 60,
        is_rev,
        is_secondary: false,
        is_supplementary: false,
        nm: 0,
        md: format!("{}", ref_end - ref_start),
        as_score: score,
        xs_score: None,
        xs_strand: None,
        mate: MateInfo::default(),
    }
}

fn align_cfg() -> AlignmentConfig {
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

#[test]
fn rescues_unmapped_r2_inside_insert_window() {
    let reference = synth_reference();
    let ref_bytes = match &reference.sequences[0].bases {
        RefBases::Owned(v) => v.clone(),
        _ => panic!("expected owned"),
    };
    let index = build_index(&reference);

    // R1 maps forward at position 500..650. The true R2 sits at 700..850
    // and on the reverse strand (FR convention). For the test, the read
    // sequence we feed is the FORWARD-strand read — rescue handles the
    // reverse-complementing internally based on the anchor strand.
    let r2_true_start = 700;
    let r2_true_end = 850;
    let r2_forward_bases = ref_bytes[r2_true_start..r2_true_end].to_vec();
    // The actual R2 read on disk is reverse-complemented (mate strand is
    // opposite to R1). rescue_unmapped_mates expects the read seq as
    // stored in the ReadRecord — which is the original sequencer output.
    // For a -R2 strand orientation that's already reverse-complemented
    // before we receive it; in our synthetic test, simulate that by
    // reverse-complementing manually.
    let mut r2_seq_in_read = r2_forward_bases.clone();
    r2_seq_in_read.reverse();
    for b in r2_seq_in_read.iter_mut() {
        *b = match *b {
            b'A' => b'T',
            b'C' => b'G',
            b'G' => b'C',
            b'T' => b'A',
            x => x,
        };
    }

    let reads = vec![
        mk_paired_read("pair1", ref_bytes[500..650].to_vec(), PairRole::R1),
        mk_paired_read("pair1", r2_seq_in_read, PairRole::R2),
    ];

    // R1 has a high-score primary; R2 has nothing (chaining failed).
    let mut alignments: Vec<Vec<Alignment>> = vec![vec![mk_aln(0, 500, 650, false, 150)], vec![]];

    let mut paired_cfg = PairedConfig::default();
    paired_cfg.mode = IngestMode::TwoFile;
    paired_cfg.insert_max = 1000;
    let rescue_cfg = RescueConfig::default();

    rescue_unmapped_mates(
        &reads,
        &mut alignments,
        &index,
        &paired_cfg,
        align_cfg(),
        rescue_cfg,
    );

    // R2 must now have at least one alignment, on the reverse strand,
    // landing inside the rescue window [650, 650+1000).
    assert_eq!(alignments[0].len(), 1, "R1 unchanged");
    assert!(!alignments[1].is_empty(), "R2 should be rescued, got empty");
    let r2_aln = &alignments[1][0];
    assert!(
        r2_aln.is_rev,
        "rescued R2 must be on reverse strand (FR pair)"
    );
    assert!(
        (r2_aln.ref_start as usize) >= 650
            && (r2_aln.ref_end as usize) <= 650 + paired_cfg.insert_max as usize,
        "rescued R2 ref_start={} ref_end={} outside window [650, 1650)",
        r2_aln.ref_start,
        r2_aln.ref_end
    );
    // The rescued window should overlap the true insert position 700..850.
    assert!(
        r2_aln.ref_start as usize <= r2_true_end && r2_aln.ref_end as usize >= r2_true_start,
        "rescued region missed the true insert site"
    );
}

#[test]
fn skips_rescue_when_both_mapped() {
    let reference = synth_reference();
    let index = build_index(&reference);

    let reads = vec![
        mk_paired_read("p", vec![b'A'; 150], PairRole::R1),
        mk_paired_read("p", vec![b'T'; 150], PairRole::R2),
    ];
    let mut alignments = vec![
        vec![mk_aln(0, 100, 250, false, 150)],
        vec![mk_aln(0, 500, 650, true, 150)],
    ];
    let prev_r1_len = alignments[0].len();
    let prev_r2_len = alignments[1].len();

    let mut paired_cfg = PairedConfig::default();
    paired_cfg.mode = IngestMode::TwoFile;

    rescue_unmapped_mates(
        &reads,
        &mut alignments,
        &index,
        &paired_cfg,
        align_cfg(),
        RescueConfig::default(),
    );

    assert_eq!(alignments[0].len(), prev_r1_len);
    assert_eq!(alignments[1].len(), prev_r2_len);
}

#[test]
fn skips_rescue_when_both_unmapped() {
    let reference = synth_reference();
    let index = build_index(&reference);
    let reads = vec![
        mk_paired_read("p", vec![b'A'; 150], PairRole::R1),
        mk_paired_read("p", vec![b'T'; 150], PairRole::R2),
    ];
    let mut alignments: Vec<Vec<Alignment>> = vec![vec![], vec![]];

    let mut paired_cfg = PairedConfig::default();
    paired_cfg.mode = IngestMode::TwoFile;

    rescue_unmapped_mates(
        &reads,
        &mut alignments,
        &index,
        &paired_cfg,
        align_cfg(),
        RescueConfig::default(),
    );

    assert!(alignments[0].is_empty());
    assert!(alignments[1].is_empty());
}

#[test]
fn skips_rescue_when_unpaired_mode() {
    let reference = synth_reference();
    let index = build_index(&reference);
    let reads = vec![mk_paired_read("p", vec![b'A'; 150], PairRole::Unpaired)];
    let mut alignments: Vec<Vec<Alignment>> = vec![vec![]];

    rescue_unmapped_mates(
        &reads,
        &mut alignments,
        &index,
        &PairedConfig::default(), // Unpaired
        align_cfg(),
        RescueConfig::default(),
    );

    assert!(alignments[0].is_empty());
}

#[test]
fn locked_estimator_narrows_rescue_window_to_3sigma() {
    // After lock-in, rescue_unmapped_mates should restrict the SW search
    // to `anchor_edge + insert_mean ± 3σ` instead of the ±insert_max
    // sweep used during bootstrap. We engineer a case where the *true*
    // R2 sits OUTSIDE the narrow window (so a locked-config rescue must
    // refuse it) but well INSIDE the wide bootstrap window (so an
    // unlocked-config rescue with the same insert_max picks it up).
    // Without this distinction, the wide bootstrap window was the
    // primary driver of TLEN-std blow-up on real PE data, since SW
    // happily latches onto whatever positive-scoring match it finds in
    // a 1500 bp sweep.
    let reference = synth_reference();
    let ref_bytes = match &reference.sequences[0].bases {
        RefBases::Owned(v) => v.clone(),
        _ => panic!("expected owned"),
    };
    let index = build_index(&reference);

    // R1 forward at 500..650 — anchor_edge = 650.
    let r1_start = 500usize;
    let r1_end = 650usize;
    // True R2 at 2500..2650 — TLEN ~ 2150. Inside [+anchor_edge,
    // +anchor_edge+insert_max=1500] = [650, 2150]? actually exact edge.
    // Push it just over so the locked window definitely excludes it:
    let r2_true_start = 2500usize;
    let r2_true_end = 2650usize;

    let r1_seq = ref_bytes[r1_start..r1_end].to_vec();
    let mut r2_seq_in_read = ref_bytes[r2_true_start..r2_true_end].to_vec();
    r2_seq_in_read.reverse();
    for b in r2_seq_in_read.iter_mut() {
        *b = match *b {
            b'A' => b'T',
            b'C' => b'G',
            b'G' => b'C',
            b'T' => b'A',
            x => x,
        };
    }
    let reads = vec![
        mk_paired_read("p", r1_seq, PairRole::R1),
        mk_paired_read("p", r2_seq_in_read.clone(), PairRole::R2),
    ];

    // Configure a *locked* PairedConfig with a tight mean / σ
    // characteristic of a real Illumina library. The 3 σ rescue
    // window is then [650 + 200 - 150, 650 + 200 + 150] = [700, 1000]
    // — well below the true R2 at 2500..2650.
    let mut locked_cfg = PairedConfig::default();
    locked_cfg.mode = IngestMode::TwoFile;
    locked_cfg.insert_mean = 200;
    locked_cfg.insert_sd = 50;
    locked_cfg.insert_min = 0;
    locked_cfg.insert_max = 3_000;
    locked_cfg.estimator_locked = true;

    let mut alignments_locked: Vec<Vec<Alignment>> = vec![
        vec![mk_aln(0, r1_start as u32, r1_end as u32, false, 150)],
        vec![],
    ];
    rescue_unmapped_mates(
        &reads,
        &mut alignments_locked,
        &index,
        &locked_cfg,
        align_cfg(),
        RescueConfig::default(),
    );
    // The narrow [700, 1000] window can't reach the true mate, so even
    // if SW finds *something* in that window, it must not be the true
    // 2500..2650 placement.
    for a in alignments_locked[1].iter() {
        assert!(
            a.ref_start < r2_true_start as u32 || a.ref_end > r2_true_end as u32,
            "locked-rescue should not have reached the out-of-window true mate at {r2_true_start}..{r2_true_end}, got [{}, {})",
            a.ref_start,
            a.ref_end
        );
        // And whatever it found must lie inside the narrow window
        // [700, 1000] (allowing the off-by-one of partial alignment).
        assert!(
            a.ref_start >= 700 && a.ref_end <= 1000,
            "locked-rescue strayed outside the 3σ window: [{}, {})",
            a.ref_start,
            a.ref_end
        );
    }

    // Now repeat with the same numeric prior but estimator_locked=false:
    // the rescue window expands to [anchor_edge, anchor_edge+insert_max]
    // = [650, 3650], which now contains the true mate at 2500..2650.
    let mut unlocked_cfg = locked_cfg;
    unlocked_cfg.estimator_locked = false;
    let mut alignments_unlocked: Vec<Vec<Alignment>> = vec![
        vec![mk_aln(0, r1_start as u32, r1_end as u32, false, 150)],
        vec![],
    ];
    rescue_unmapped_mates(
        &reads,
        &mut alignments_unlocked,
        &index,
        &unlocked_cfg,
        align_cfg(),
        RescueConfig::default(),
    );
    let r2_aln = alignments_unlocked[1]
        .first()
        .expect("unlocked rescue should reach the true mate");
    assert!(r2_aln.is_rev, "rescued R2 must be on reverse strand");
    assert!(
        r2_aln.ref_start as usize <= r2_true_end && r2_aln.ref_end as usize >= r2_true_start,
        "unlocked rescue missed the true mate site"
    );
}
