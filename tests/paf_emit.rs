//! PAF emit verification.
//!
//! Cover the minimap2-compatible 12 mandatory columns + `tp/NM/AS/dv`
//! tags, plus the strand-flip convention for query coordinates on
//! reverse-strand alignments. We drive `SamFormatter::append_paf`
//! directly so the test doesn't depend on the full pipeline plumbing.

use std::sync::Arc;

use kira_ls_aligner::io::{OutputConfig, SamFormatter};
use kira_ls_aligner::types::{
    Alignment, AlignmentKind, CigarKind, CigarOp, MateInfo, PairRole, ReadRecord, RefBases, RefSeq,
    Reference,
};

fn make_reference() -> Reference {
    Reference {
        sequences: vec![RefSeq {
            name: "chr1".to_string(),
            bases: RefBases::Owned(vec![b'A'; 1_000_000]),
        }],
    }
}

fn mk_read(id: &str, seq_len: usize) -> ReadRecord {
    ReadRecord {
        id: id.to_string(),
        seq: vec![b'A'; seq_len],
        qual: None,
        pair_role: PairRole::Unpaired,
        repeat_min_occ: 1,
        comment: None,
    }
}

fn mk_aln(
    ref_start: u32,
    ref_end: u32,
    read_start: u32,
    read_end: u32,
    is_rev: bool,
    nm: u32,
) -> Alignment {
    Alignment {
        kind: AlignmentKind::DpAligned,
        ref_id: 0,
        ref_start,
        ref_end,
        read_start,
        read_end,
        cigar: vec![CigarOp {
            len: ref_end - ref_start,
            op: CigarKind::Match,
        }],
        score: (ref_end - ref_start) as i32 - nm as i32 * 4,
        mapq: 60,
        is_rev,
        is_secondary: false,
        is_supplementary: false,
        nm,
        md: format!("{}", ref_end - ref_start),
        as_score: (ref_end - ref_start) as i32 - nm as i32 * 4,
        xs_score: None,
        xs_strand: None,
        mate: MateInfo::default(),
    }
}

fn parse_paf(line: &[u8]) -> Vec<String> {
    let s = std::str::from_utf8(line).expect("utf-8");
    s.trim_end_matches('\n')
        .split('\t')
        .map(String::from)
        .collect()
}

#[test]
fn forward_primary_emits_12_cols_and_tags() {
    let fmt = SamFormatter::new(Arc::new(make_reference()));
    let read = mk_read("r1", 150);
    let aln = mk_aln(1000, 1150, 0, 150, false, 2);
    let mut buf = Vec::new();
    fmt.append_paf(&mut buf, &read, &aln, OutputConfig::paf());
    let cols = parse_paf(&buf);
    assert!(
        cols.len() >= 12,
        "expected at least 12 mandatory cols, got {}",
        cols.len()
    );
    assert_eq!(cols[0], "r1", "col 1: qname");
    assert_eq!(cols[1], "150", "col 2: qlen");
    assert_eq!(cols[2], "0", "col 3: qstart");
    assert_eq!(cols[3], "150", "col 4: qend");
    assert_eq!(cols[4], "+", "col 5: strand");
    assert_eq!(cols[5], "chr1", "col 6: tname");
    assert_eq!(cols[6], "1000000", "col 7: tlen");
    assert_eq!(cols[7], "1000", "col 8: tstart");
    assert_eq!(cols[8], "1150", "col 9: tend");
    assert_eq!(cols[9], "148", "col 10: matches (150 - NM 2)");
    assert_eq!(cols[10], "150", "col 11: block_len");
    assert_eq!(cols[11], "60", "col 12: mapq");
    // Tags appear after col 12. Look for tp:A:P, NM:i:2, AS:i:142, dv:f:0.0133
    let tags: Vec<String> = cols.iter().skip(12).cloned().collect();
    assert!(
        tags.contains(&"tp:A:P".to_string()),
        "tp:A:P missing in {tags:?}"
    );
    assert!(tags.contains(&"NM:i:2".to_string()), "NM tag missing");
    assert!(
        tags.iter().any(|t| t.starts_with("AS:i:")),
        "AS tag missing"
    );
    assert!(
        tags.iter().any(|t| t.starts_with("dv:f:")),
        "dv tag missing"
    );
}

#[test]
fn reverse_alignment_flips_query_coords() {
    let fmt = SamFormatter::new(Arc::new(make_reference()));
    let read = mk_read("r1", 150);
    // Read was reverse-complemented and read_start/read_end refer to
    // positions in that RC'd read. For a flush-aligned 150bp read
    // is_rev=true, read_start=0, read_end=150 ⇒ forward-strand qstart=0,
    // qend=150 (the entire read covered, just on the other strand).
    let aln = mk_aln(1000, 1150, 0, 150, true, 0);
    let mut buf = Vec::new();
    fmt.append_paf(&mut buf, &read, &aln, OutputConfig::paf());
    let cols = parse_paf(&buf);
    assert_eq!(cols[2], "0");
    assert_eq!(cols[3], "150");
    assert_eq!(cols[4], "-");
}

#[test]
fn reverse_with_soft_clip_flips_correctly() {
    let fmt = SamFormatter::new(Arc::new(make_reference()));
    let read = mk_read("r1", 200);
    // Read 0..200, reverse alignment, but read_start=20 read_end=180.
    // Forward-strand: qstart = 200 - 180 = 20, qend = 200 - 20 = 180.
    let aln = mk_aln(1000, 1160, 20, 180, true, 0);
    let mut buf = Vec::new();
    fmt.append_paf(&mut buf, &read, &aln, OutputConfig::paf());
    let cols = parse_paf(&buf);
    assert_eq!(cols[2], "20", "qstart after flip");
    assert_eq!(cols[3], "180", "qend after flip");
    assert_eq!(cols[4], "-");
}

#[test]
fn secondary_alignment_tags_tp_a_s() {
    let fmt = SamFormatter::new(Arc::new(make_reference()));
    let read = mk_read("r1", 150);
    let mut aln = mk_aln(2000, 2150, 0, 150, false, 1);
    aln.is_secondary = true;
    aln.mapq = 0;
    let mut buf = Vec::new();
    fmt.append_paf(&mut buf, &read, &aln, OutputConfig::paf());
    let cols = parse_paf(&buf);
    let tags: Vec<&String> = cols.iter().skip(12).collect();
    assert!(tags.iter().any(|t| t.as_str() == "tp:A:S"));
}

#[test]
fn supplementary_alignment_tags_tp_a_i() {
    let fmt = SamFormatter::new(Arc::new(make_reference()));
    let read = mk_read("r1", 1500);
    let mut aln = mk_aln(1000, 2500, 0, 1500, false, 0);
    aln.is_supplementary = true;
    let mut buf = Vec::new();
    fmt.append_paf(&mut buf, &read, &aln, OutputConfig::paf());
    let cols = parse_paf(&buf);
    let tags: Vec<&String> = cols.iter().skip(12).collect();
    assert!(tags.iter().any(|t| t.as_str() == "tp:A:I"));
}

#[test]
fn dv_tag_is_nm_over_block_len() {
    let fmt = SamFormatter::new(Arc::new(make_reference()));
    let read = mk_read("r1", 100);
    let aln = mk_aln(1000, 1100, 0, 100, false, 10);
    let mut buf = Vec::new();
    fmt.append_paf(&mut buf, &read, &aln, OutputConfig::paf());
    let cols = parse_paf(&buf);
    // NM=10, block_len=100 → dv = 0.10
    let dv = cols
        .iter()
        .find_map(|c| c.strip_prefix("dv:f:"))
        .expect("dv tag");
    assert_eq!(dv, "0.1000", "dv exact value");
}

#[test]
fn block_len_excludes_soft_clips() {
    let fmt = SamFormatter::new(Arc::new(make_reference()));
    let read = mk_read("r1", 150);
    let mut aln = mk_aln(1000, 1100, 25, 125, false, 0);
    // Add soft-clip ops on both ends.
    aln.cigar = vec![
        CigarOp {
            len: 25,
            op: CigarKind::SoftClip,
        },
        CigarOp {
            len: 100,
            op: CigarKind::Match,
        },
        CigarOp {
            len: 25,
            op: CigarKind::SoftClip,
        },
    ];
    let mut buf = Vec::new();
    fmt.append_paf(&mut buf, &read, &aln, OutputConfig::paf());
    let cols = parse_paf(&buf);
    // block_len should be 100 (the M op), not 150.
    assert_eq!(cols[10], "100", "block_len excludes S ops");
    // matches = 100 - NM(0) = 100
    assert_eq!(cols[9], "100");
}
