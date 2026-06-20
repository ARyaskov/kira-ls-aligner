//! Mate-rescue for the both-mapped-but-discordant case: when stage 4
//! placed R1 and R2 in valid but far-apart positions, `rescue_discordant_pairs`
//! should run a banded SW for the lower-MAPQ side inside the anchor's
//! expected insert window and replace the discordant primary if a
//! better-fitting placement exists there.

use kira_ls_aligner::alignment::AlignmentConfig;
use kira_ls_aligner::index::{Index, IndexConfig};
use kira_ls_aligner::io::IngestMode;
use kira_ls_aligner::pipeline::pairing::{PairedConfig, RescueConfig, rescue_discordant_pairs};
use kira_ls_aligner::types::{
    Alignment, AlignmentKind, CigarKind, CigarOp, MateInfo, PairRole, ReadRecord, RefBases, RefSeq,
    Reference,
};

fn synth_reference() -> Reference {
    let mut seed = 0xfeed_face_cafe_d00du64;
    let mut buf = Vec::with_capacity(100_000);
    let bases = b"ACGT";
    for _ in 0..100_000 {
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

fn mk_aln(
    ref_id: u32,
    ref_start: u32,
    ref_end: u32,
    is_rev: bool,
    score: i32,
    mapq: u8,
) -> Alignment {
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
        mapq,
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

fn rc(seq: &[u8]) -> Vec<u8> {
    let mut out: Vec<u8> = seq
        .iter()
        .map(|&b| match b {
            b'A' => b'T',
            b'C' => b'G',
            b'G' => b'C',
            b'T' => b'A',
            x => x,
        })
        .collect();
    out.reverse();
    out
}

#[test]
fn rescues_discordant_r2_into_insert_window() {
    // Setup: R1 maps confidently at chr0:1000-1150. R2's TRUE position is
    // chr0:1400-1550 (FR pair, insert 550). Stage 4 placed R2 at
    // chr0:50000-50150 (a wrong-position multi-mapper) — geometry-wise
    // discordant from R1.
    let reference = synth_reference();
    let ref_bytes = match &reference.sequences[0].bases {
        RefBases::Owned(v) => v.clone(),
        _ => panic!("expected owned"),
    };
    let index = build_index(&reference);

    let r1_start = 1000;
    let r1_end = 1150;
    // Centre the true R2 position deep inside the rescue window so the SW
    // can find the full 150-base match. anchor_ref_end (1150) +
    // insert_mean (550) = expected centre at 1700; we put R2 at 1625..1775
    // so the perfect-match score is the maximum 150 and easily clears the
    // min_score floor.
    let r2_true_start = 1625;
    let r2_true_end = 1775;

    let r1_seq = ref_bytes[r1_start..r1_end].to_vec();
    // R2 is reverse-strand in the pair → on disk, the sequencer outputs
    // the reverse-complement of the reference at the true position.
    let r2_seq_in_read = rc(&ref_bytes[r2_true_start..r2_true_end]);

    let reads = vec![
        mk_paired_read("disc1", r1_seq, PairRole::R1),
        mk_paired_read("disc1", r2_seq_in_read, PairRole::R2),
    ];

    // Stage-4 placements: R1 correct; R2 at a wrong-position multi-mapper
    // (same strand as R1, far from the pair). R1 has higher MAPQ so it's
    // the anchor. R2's score is lower than what a perfect rescue produces,
    // so the rescue will replace it.
    let mut alignments: Vec<Vec<Alignment>> = vec![
        vec![mk_aln(0, r1_start as u32, r1_end as u32, false, 150, 60)],
        vec![mk_aln(0, 50_000, 50_150, false, 100, 50)],
    ];

    let mut paired_cfg = PairedConfig::default();
    paired_cfg.mode = IngestMode::TwoFile;
    paired_cfg.insert_mean = 550;
    paired_cfg.insert_sd = 80;
    paired_cfg.insert_min = 0;
    paired_cfg.insert_max = 1500;

    rescue_discordant_pairs(
        &reads,
        &mut alignments,
        &index,
        &paired_cfg,
        align_cfg(),
        RescueConfig::default(),
    );

    // R2 should now have a NEW primary inside the rescue window
    // [r1_end + mean - 3sd, r1_end + mean + 3sd] = [1460, 1940],
    // overlapping the true position 1625..1775.
    let r2_primary = &alignments[1][0];
    assert!(
        r2_primary.is_rev,
        "rescued R2 must be on reverse strand (FR pair), got is_rev={}",
        r2_primary.is_rev
    );
    let lo = 1150u32 + 550 - 3 * 80;
    let hi = 1150u32 + 550 + 3 * 80;
    assert!(
        r2_primary.ref_start >= lo && r2_primary.ref_end <= hi,
        "rescued R2 ref [{}, {}) outside expected window [{}, {})",
        r2_primary.ref_start,
        r2_primary.ref_end,
        lo,
        hi
    );
    // And it should land at the true insert site.
    assert!(
        r2_primary.ref_start <= r2_true_end as u32 && r2_primary.ref_end >= r2_true_start as u32,
        "rescued region [{}, {}) didn't overlap true site [{}, {})",
        r2_primary.ref_start,
        r2_primary.ref_end,
        r2_true_start,
        r2_true_end
    );
    // The old discordant placement (at 50000) should still be in the vec
    // but at slot 1 (rescue inserts new at slot 0).
    assert!(
        alignments[1].iter().any(|a| a.ref_start == 50_000),
        "old discordant alignment should still be present as secondary"
    );
}

#[test]
fn no_rescue_when_pair_is_concordant() {
    // Both mates already in the expected insert window → no rescue should
    // fire and alignment vec should be unchanged.
    let reference = synth_reference();
    let index = build_index(&reference);

    let reads = vec![
        mk_paired_read("ok", vec![b'A'; 150], PairRole::R1),
        mk_paired_read("ok", vec![b'T'; 150], PairRole::R2),
    ];
    // R1 forward at 1000, R2 reverse at 1400 → TLEN 550, within 550±240
    let mut alignments = vec![
        vec![mk_aln(0, 1000, 1150, false, 150, 60)],
        vec![mk_aln(0, 1400, 1550, true, 150, 60)],
    ];
    let original_starts: Vec<u32> = vec![alignments[0][0].ref_start, alignments[1][0].ref_start];

    let mut paired_cfg = PairedConfig::default();
    paired_cfg.mode = IngestMode::TwoFile;
    paired_cfg.insert_mean = 550;
    paired_cfg.insert_sd = 80;
    paired_cfg.insert_max = 1500;

    rescue_discordant_pairs(
        &reads,
        &mut alignments,
        &index,
        &paired_cfg,
        align_cfg(),
        RescueConfig::default(),
    );

    assert_eq!(alignments[0][0].ref_start, original_starts[0]);
    assert_eq!(alignments[1][0].ref_start, original_starts[1]);
    assert_eq!(alignments[0].len(), 1);
    assert_eq!(alignments[1].len(), 1);
}

#[test]
fn no_rescue_when_unpaired() {
    let reference = synth_reference();
    let index = build_index(&reference);
    let reads = vec![mk_paired_read("u", vec![b'A'; 150], PairRole::Unpaired)];
    let mut alignments: Vec<Vec<Alignment>> = vec![vec![mk_aln(0, 0, 150, false, 100, 60)]];
    let paired_cfg = PairedConfig::default(); // unpaired mode
    rescue_discordant_pairs(
        &reads,
        &mut alignments,
        &index,
        &paired_cfg,
        align_cfg(),
        RescueConfig::default(),
    );
    assert_eq!(alignments[0].len(), 1);
}
