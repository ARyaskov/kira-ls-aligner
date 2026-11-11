//! `XA:Z:` is built from the full candidate list, not from the records that
//! survive the per-read output cap, so the default configuration (one record
//! per read) still reports alternative hits as bwa-mem does.

use std::sync::Arc;

use kira_ls_aligner::io::{OutputConfig, SamFormatter};
use kira_ls_aligner::pipeline::stage5_scoring::ScoredBatch;
use kira_ls_aligner::pipeline::stage6_output::serialize;
use kira_ls_aligner::types::{
    Alignment, AlignmentKind, CigarKind, CigarOp, MateInfo, PairRole, ReadRecord, RefBases, RefSeq,
    Reference,
};

fn aln(ref_id: u32, ref_start: u32, score: i32, secondary: bool) -> Alignment {
    Alignment {
        kind: AlignmentKind::DpAligned,
        ref_id,
        ref_start,
        ref_end: ref_start + 50,
        read_start: 0,
        read_end: 50,
        cigar: vec![CigarOp {
            len: 50,
            op: CigarKind::Match,
        }],
        score,
        mapq: if secondary { 0 } else { 30 },
        is_rev: false,
        is_secondary: secondary,
        is_supplementary: false,
        nm: 1,
        md: "20A29".to_string(),
        as_score: score,
        xs_score: None,
        xs_strand: None,
        mate: MateInfo::default(),
    }
}

#[test]
fn xa_lists_secondaries_pruned_by_the_output_cap() {
    let reference = Reference {
        sequences: vec![
            RefSeq {
                name: "chr1".into(),
                bases: RefBases::Owned(vec![b'A'; 2000]),
            },
            RefSeq {
                name: "chr2".into(),
                bases: RefBases::Owned(vec![b'C'; 2000]),
            },
        ],
    };
    let formatter = SamFormatter::new(Arc::new(reference));
    let batch = ScoredBatch {
        reads: vec![ReadRecord {
            id: "r".into(),
            seq: vec![b'A'; 50],
            qual: Some(vec![b'I'; 50]),
            pair_role: PairRole::Unpaired,
            repeat_min_occ: 1,
            comment: None,
        }],
        alignments: vec![vec![
            aln(0, 100, 50, false),
            aln(1, 400, 48, true),
            aln(0, 900, 45, true),
        ]],
        xa_has_alt: vec![false],
        unmapped_mate_info: vec![None],
        stats: Default::default(),
    };
    // max_alignments = 1 is the CLI default: only the primary record is written.
    let sam =
        String::from_utf8(serialize(batch, &formatter, None, OutputConfig::full(), 1)).unwrap();
    let lines: Vec<&str> = sam.lines().collect();
    assert_eq!(lines.len(), 1, "only the primary record is emitted:\n{sam}");
    assert!(
        lines[0].contains("\tXA:Z:chr2,+401,50M,1;chr1,+901,50M,1;"),
        "XA must list the pruned secondaries:\n{sam}"
    );
}

/// bwa-mem `-h INT[,INT]` suppression rule: a qualifying ALT candidate among
/// the counted candidates selects the second (ALT) cap, but the second cap
/// applies universally; either cap exceeded suppresses the entire tag —
/// candidates are never truncated.
fn xa_formatter() -> SamFormatter {
    let reference = Reference {
        sequences: (0..4)
            .map(|i| RefSeq {
                name: format!("chr{i}"),
                bases: RefBases::Owned(vec![b'A'; 2000]),
            })
            .collect(),
    };
    SamFormatter::new(Arc::new(reference))
}

/// `primary + n` secondary candidates on distinct contigs.
fn xa_candidates(n: usize) -> Vec<Alignment> {
    let mut alns = vec![aln(0, 100, 50, false)];
    for i in 0..n {
        alns.push(aln((i % 3 + 1) as u32, 400 + 100 * i as u32, 48, true));
    }
    alns
}

fn capped(
    fmt: &SamFormatter,
    alns: &[Alignment],
    first: u32,
    alt: u32,
    has_alt: bool,
) -> Option<String> {
    let mut buf = Vec::new();
    fmt.append_xa_capped(&mut buf, alns, first, alt, has_alt)
        .then(|| String::from_utf8(buf).expect("XA value is ASCII"))
}

#[test]
fn non_alt_over_first_cap_suppresses_everything() {
    let fmt = xa_formatter();
    let alns = xa_candidates(3);
    assert_eq!(
        capped(&fmt, &alns, 2, 200, false),
        None,
        "count 3 > first 2"
    );
}

#[test]
fn alt_over_first_cap_within_second_cap_emits_everything() {
    let fmt = xa_formatter();
    let alns = xa_candidates(3);
    let xa = capped(&fmt, &alns, 2, 200, true).expect("ALT candidates select the second cap");
    assert_eq!(
        xa.matches(';').count(),
        3,
        "all candidates listed, none truncated:\n{xa}"
    );
}

#[test]
fn alt_over_second_cap_suppresses_everything() {
    let fmt = xa_formatter();
    let alns = xa_candidates(3);
    assert_eq!(capped(&fmt, &alns, 2, 2, true), None, "count 3 > alt cap 2");
}

#[test]
fn first_cap_above_second_still_obeys_universal_second_cap() {
    let fmt = xa_formatter();
    let alns = xa_candidates(3);
    assert_eq!(
        capped(&fmt, &alns, 5, 2, false),
        None,
        "second cap applies even when the first is larger"
    );
}

#[test]
fn zero_candidates_emits_nothing_regardless_of_caps() {
    let fmt = xa_formatter();
    let alns = vec![aln(0, 100, 50, false)];
    let mut buf = Vec::new();
    assert!(!fmt.append_xa_capped(&mut buf, &alns, 0, 0, true));
    assert!(buf.is_empty());
}

/// Text-path counterpart of the cap rule: `has_alt` from stage 5 decides
/// which cap applies when the candidate count exceeds the first cap.
#[test]
fn stage5_has_alt_selects_the_second_cap_in_sam_output() {
    let fmt = xa_formatter();
    let alns = xa_candidates(3); // primary + 3 secondaries, none supplementary
    let reads = [("with-alt", true), ("no-alt", false)]
        .map(|(id, _)| ReadRecord {
            id: id.into(),
            seq: vec![b'A'; 50],
            qual: Some(vec![b'I'; 50]),
            pair_role: PairRole::Unpaired,
            repeat_min_occ: 1,
            comment: None,
        })
        .to_vec();
    let mut cfg = OutputConfig::full();
    cfg.xa_max = 2; // first cap below the candidate count; ALT cap 200
    let sam = String::from_utf8(serialize(
        ScoredBatch {
            reads,
            alignments: vec![alns.clone(), alns],
            xa_has_alt: vec![true, false],
            unmapped_mate_info: vec![None, None],
            stats: Default::default(),
        },
        &fmt,
        None,
        cfg,
        1,
    ))
    .unwrap();
    let lines: Vec<&str> = sam.lines().filter(|l| !l.starts_with('@')).collect();
    assert!(lines.len() >= 2, "two primary records expected:\n{sam}");
    assert!(
        lines.iter().any(|l| l.contains("\tXA:Z:")),
        "has_alt=true read must keep its XA list:\n{sam}"
    );
    assert!(
        lines.iter().any(|l| !l.contains("\tXA:Z:")),
        "has_alt=false read over the first cap must drop XA entirely:\n{sam}"
    );
}
