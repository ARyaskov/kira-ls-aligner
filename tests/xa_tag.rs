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
