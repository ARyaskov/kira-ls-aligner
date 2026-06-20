//! Cover SAM flag emission and paired-mate field formatting through the
//! public `SamFormatter` API.
//!
//! We rely on `SamFormatter` rather than the private `sam_flag()` helper so
//! the test exercises the same path used by the production pipeline — any
//! breakage in tag/field ordering is caught here too.

use std::sync::Arc;

use kira_ls_aligner::io::{OutputConfig, SamFormatter};
use kira_ls_aligner::types::{
    Alignment, AlignmentKind, CigarKind, CigarOp, MateInfo, PairRole, ReadRecord, RefBases, RefSeq,
    Reference,
};

fn make_reference() -> Reference {
    Reference {
        sequences: vec![
            RefSeq {
                name: "chr1".to_string(),
                bases: RefBases::Owned(vec![b'A'; 1000]),
            },
            RefSeq {
                name: "chr2".to_string(),
                bases: RefBases::Owned(vec![b'C'; 1000]),
            },
        ],
    }
}

fn base_alignment(ref_id: u32, ref_start: u32, ref_end: u32, is_rev: bool) -> Alignment {
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

fn r1_record() -> ReadRecord {
    ReadRecord {
        id: "test_read".to_string(),
        seq: vec![b'A'; 150],
        qual: Some(vec![b'I'; 150]),
        pair_role: PairRole::R1,
        repeat_min_occ: 1,
    }
}

fn r2_record() -> ReadRecord {
    ReadRecord {
        id: "test_read".to_string(),
        seq: vec![b'T'; 150],
        qual: Some(vec![b'I'; 150]),
        pair_role: PairRole::R2,
        repeat_min_occ: 1,
    }
}

fn unpaired_record() -> ReadRecord {
    ReadRecord {
        id: "test_read".to_string(),
        seq: vec![b'A'; 150],
        qual: Some(vec![b'I'; 150]),
        pair_role: PairRole::Unpaired,
        repeat_min_occ: 1,
    }
}

/// Parse a single SAM line into (flag, rname, pos, mapq, cigar, rnext, pnext, tlen).
/// Stops at first 11 fields — enough for flag testing.
fn parse_sam(line: &[u8]) -> (u32, String, u32, u32, String, String, u32, i32) {
    let s = std::str::from_utf8(line).expect("valid utf-8");
    let s = s.trim_end_matches('\n');
    let cols: Vec<&str> = s.split('\t').collect();
    assert!(
        cols.len() >= 11,
        "expected ≥11 SAM columns, got {}",
        cols.len()
    );
    (
        cols[1].parse().unwrap(), // flag
        cols[2].to_string(),      // rname
        cols[3].parse().unwrap(), // pos
        cols[4].parse().unwrap(), // mapq
        cols[5].to_string(),      // cigar
        cols[6].to_string(),      // rnext
        cols[7].parse().unwrap(), // pnext
        cols[8].parse().unwrap(), // tlen
    )
}

#[test]
fn unpaired_mapped_emits_no_paired_bits() {
    let fmt = SamFormatter::new(Arc::new(make_reference()));
    let aln = base_alignment(0, 100, 250, false);
    let mut buf = Vec::new();
    fmt.append_alignment(
        &mut buf,
        &unpaired_record(),
        &aln,
        None,
        None,
        OutputConfig::full(),
    );
    let (flag, rname, pos, _mapq, _cigar, rnext, pnext, tlen) = parse_sam(&buf);
    // Unpaired, forward, mapped → only 0x0 bits or 0x10 if rev. We set forward.
    assert_eq!(flag & 0x1, 0, "should not be paired");
    assert_eq!(flag & 0x10, 0, "should not be reverse");
    assert_eq!(rname, "chr1");
    assert_eq!(pos, 101);
    assert_eq!(rnext, "*");
    assert_eq!(pnext, 0);
    assert_eq!(tlen, 0);
}

#[test]
fn paired_proper_r1_sets_correct_bits_and_fields() {
    let fmt = SamFormatter::new(Arc::new(make_reference()));
    let mut aln = base_alignment(0, 100, 250, false);
    aln.mate = MateInfo {
        is_paired: true,
        is_proper_pair: true,
        mate_is_unmapped: false,
        mate_is_rev: true, // mate on reverse strand → 0x20
        is_first_in_pair: true,
        is_second_in_pair: false,
        mate_ref_id: Some(0), // same chr → RNEXT='='
        mate_pos: 350,
        tlen: 400,
    };
    let mut buf = Vec::new();
    fmt.append_alignment(
        &mut buf,
        &r1_record(),
        &aln,
        None,
        None,
        OutputConfig::full(),
    );
    let (flag, rname, pos, _mapq, _cigar, rnext, pnext, tlen) = parse_sam(&buf);
    assert!(flag & 0x1 != 0, "0x1 paired");
    assert!(flag & 0x2 != 0, "0x2 proper pair");
    assert_eq!(flag & 0x4, 0, "0x4 unmapped must be clear");
    assert_eq!(flag & 0x8, 0, "0x8 mate unmapped must be clear");
    assert_eq!(flag & 0x10, 0, "0x10 reverse must be clear (forward)");
    assert!(flag & 0x20 != 0, "0x20 mate reverse");
    assert!(flag & 0x40 != 0, "0x40 first in pair (R1)");
    assert_eq!(flag & 0x80, 0, "0x80 must be clear on R1");
    assert_eq!(rname, "chr1");
    assert_eq!(pos, 101);
    assert_eq!(rnext, "=", "RNEXT='=' when mate on same chr");
    assert_eq!(pnext, 351, "PNEXT is mate_pos+1");
    assert_eq!(tlen, 400);
}

#[test]
fn paired_r2_reverse_sets_correct_bits_and_negative_tlen() {
    let fmt = SamFormatter::new(Arc::new(make_reference()));
    let mut aln = base_alignment(0, 350, 500, true);
    aln.mate = MateInfo {
        is_paired: true,
        is_proper_pair: true,
        mate_is_unmapped: false,
        mate_is_rev: false,
        is_first_in_pair: false,
        is_second_in_pair: true,
        mate_ref_id: Some(0),
        mate_pos: 100,
        tlen: -400,
    };
    let mut buf = Vec::new();
    fmt.append_alignment(
        &mut buf,
        &r2_record(),
        &aln,
        None,
        None,
        OutputConfig::full(),
    );
    let (flag, _rname, _pos, _mapq, _cigar, rnext, pnext, tlen) = parse_sam(&buf);
    assert!(flag & 0x1 != 0);
    assert!(flag & 0x2 != 0);
    assert!(flag & 0x10 != 0, "this read is rev");
    assert_eq!(flag & 0x20, 0, "mate forward");
    assert_eq!(flag & 0x40, 0, "R2 should not have 0x40");
    assert!(flag & 0x80 != 0, "R2 sets 0x80");
    assert_eq!(rnext, "=");
    assert_eq!(pnext, 101);
    assert_eq!(tlen, -400);
}

#[test]
fn reverse_alignment_emits_reverse_complemented_seq_and_reversed_qual() {
    let fmt = SamFormatter::new(Arc::new(make_reference()));
    let read = ReadRecord {
        id: "reverse_payload".to_string(),
        seq: b"AaCGTN".to_vec(),
        qual: Some(b"ABCDEF".to_vec()),
        pair_role: PairRole::Unpaired,
        repeat_min_occ: 1,
    };
    let mut aln = base_alignment(0, 100, 106, true);
    aln.read_end = 6;
    aln.cigar[0].len = 6;
    let mut buf = Vec::new();
    fmt.append_alignment(&mut buf, &read, &aln, None, None, OutputConfig::full());
    let line = std::str::from_utf8(&buf).unwrap().trim_end();
    let cols: Vec<&str> = line.split('\t').collect();
    assert_eq!(cols[9], "NACGtT");
    assert_eq!(cols[10], "FEDCBA");
}

#[test]
fn mate_on_different_ref_emits_full_rname() {
    let fmt = SamFormatter::new(Arc::new(make_reference()));
    let mut aln = base_alignment(0, 100, 250, false);
    aln.mate = MateInfo {
        is_paired: true,
        is_proper_pair: false, // different refs ⇒ not proper
        mate_is_unmapped: false,
        mate_is_rev: false,
        is_first_in_pair: true,
        is_second_in_pair: false,
        mate_ref_id: Some(1), // chr2
        mate_pos: 200,
        tlen: 0,
    };
    let mut buf = Vec::new();
    fmt.append_alignment(
        &mut buf,
        &r1_record(),
        &aln,
        None,
        None,
        OutputConfig::full(),
    );
    let (flag, _rname, _pos, _mapq, _cigar, rnext, _pnext, _tlen) = parse_sam(&buf);
    assert!(flag & 0x1 != 0);
    assert_eq!(flag & 0x2, 0, "should NOT be proper pair (different refs)");
    assert_eq!(rnext, "chr2", "RNEXT spelled out when on different chr");
}

#[test]
fn mate_unmapped_sets_0x8_and_uses_self_rname() {
    let fmt = SamFormatter::new(Arc::new(make_reference()));
    let mut aln = base_alignment(0, 100, 250, false);
    aln.mate = MateInfo {
        is_paired: true,
        is_proper_pair: false,
        mate_is_unmapped: true,
        mate_is_rev: false,
        is_first_in_pair: true,
        is_second_in_pair: false,
        mate_ref_id: None,
        mate_pos: 0,
        tlen: 0,
    };
    let mut buf = Vec::new();
    fmt.append_alignment(
        &mut buf,
        &r1_record(),
        &aln,
        None,
        None,
        OutputConfig::full(),
    );
    let (flag, _rname, pos, _mapq, _cigar, rnext, pnext, tlen) = parse_sam(&buf);
    assert!(flag & 0x1 != 0);
    assert!(flag & 0x8 != 0, "0x8 mate unmapped");
    // SAM convention: RNEXT='=' so the mate can be re-paired at the same coord.
    assert_eq!(rnext, "=");
    assert_eq!(
        pnext, pos,
        "PNEXT == POS when mate is at same coord placeholder"
    );
    assert_eq!(tlen, 0);
}

#[test]
fn unmapped_paired_record_has_full_flag_set() {
    let fmt = SamFormatter::new(Arc::new(make_reference()));
    let mate = MateInfo {
        is_paired: true,
        is_proper_pair: false,
        mate_is_unmapped: false,
        mate_is_rev: true,
        is_first_in_pair: false,
        is_second_in_pair: true,
        mate_ref_id: Some(0),
        mate_pos: 250,
        tlen: 0,
    };
    let mut buf = Vec::new();
    fmt.append_unmapped_with_mate(&mut buf, &r2_record(), Some(&mate));
    let (flag, rname, pos, mapq, cigar, rnext, pnext, _tlen) = parse_sam(&buf);
    assert!(flag & 0x1 != 0, "paired");
    assert!(flag & 0x4 != 0, "this segment unmapped");
    assert_eq!(flag & 0x8, 0, "mate is mapped");
    assert!(flag & 0x20 != 0, "mate reverse");
    assert!(flag & 0x80 != 0, "R2");
    assert_eq!(rname, "*");
    assert_eq!(pos, 0);
    assert_eq!(mapq, 0);
    assert_eq!(cigar, "*");
    // RNEXT is the mate's ref name when mate is mapped.
    assert_eq!(rnext, "chr1");
    assert_eq!(pnext, 251);
}

#[test]
fn supplementary_keeps_mapq_and_sets_0x800() {
    let fmt = SamFormatter::new(Arc::new(make_reference()));
    let mut aln = base_alignment(0, 500, 650, false);
    aln.is_supplementary = true;
    aln.mapq = 45;
    let mut buf = Vec::new();
    fmt.append_alignment(
        &mut buf,
        &unpaired_record(),
        &aln,
        None,
        None,
        OutputConfig::full(),
    );
    let (flag, _rname, _pos, mapq, _cigar, _rnext, _pnext, _tlen) = parse_sam(&buf);
    assert!(flag & 0x800 != 0, "0x800 supplementary");
    assert_eq!(flag & 0x100, 0, "must not be secondary");
    assert_eq!(mapq, 45, "supplementary retains MAPQ");
}
