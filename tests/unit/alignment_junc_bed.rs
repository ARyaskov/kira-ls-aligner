use super::*;
use crate::types::{RefBases, RefSeq};

fn test_reference() -> Reference {
    Reference {
        sequences: vec![
            RefSeq {
                name: "chr1".to_string(),
                bases: RefBases::Owned(vec![b'A'; 100_000]),
            },
            RefSeq {
                name: "chr2".to_string(),
                bases: RefBases::Owned(vec![b'A'; 100_000]),
            },
        ],
    }
}

#[test]
fn parses_bed6_single_intron() {
    let bed = b"chr1\t1000\t2000\tjunc1\t0\t+\nchr1\t3000\t4500\tjunc2\t0\t-\n";
    let idx = JunctionIndex::from_bed_reader(&bed[..], &test_reference()).unwrap();
    assert_eq!(idx.len(), 2);
    assert_eq!(
        idx.lookup(0, 1000, 2000, 0),
        Some(Strand::Forward),
        "first junction"
    );
    assert_eq!(
        idx.lookup(0, 3000, 4500, 0),
        Some(Strand::Reverse),
        "second junction"
    );
}

#[test]
fn parses_bed12_multi_exon() {
    let bed = b"chr1\t1000\t5000\ttx1\t0\t+\t1000\t5000\t0\t3\t100,200,500,\t0,1000,3500,\n";
    let idx = JunctionIndex::from_bed_reader(&bed[..], &test_reference()).unwrap();
    assert_eq!(idx.len(), 2);
    assert_eq!(idx.lookup(0, 1100, 2000, 0), Some(Strand::Forward));
    assert_eq!(idx.lookup(0, 2200, 4500, 0), Some(Strand::Forward));
}

#[test]
fn comment_and_header_lines_skipped() {
    let bed = b"# comment\ntrack name=test\nbrowser position chr1:1\nchr1\t1000\t2000\n";
    let idx = JunctionIndex::from_bed_reader(&bed[..], &test_reference()).unwrap();
    assert_eq!(idx.len(), 1);
}

#[test]
fn unknown_chrom_silently_skipped() {
    let bed = b"chr99\t1000\t2000\tj\t0\t+\nchr1\t500\t600\tj2\t0\t+\n";
    let idx = JunctionIndex::from_bed_reader(&bed[..], &test_reference()).unwrap();
    assert_eq!(idx.len(), 1, "only chr1 record should be loaded");
}

#[test]
fn tolerance_lookup_finds_nearby() {
    let bed = b"chr1\t1000\t2000\tj\t0\t+\n";
    let idx = JunctionIndex::from_bed_reader(&bed[..], &test_reference()).unwrap();
    // Exact match
    assert!(idx.lookup(0, 1000, 2000, 0).is_some());
    // ±1 wobble within tolerance
    assert!(idx.lookup(0, 1001, 2000, 2).is_some());
    assert!(idx.lookup(0, 999, 2001, 2).is_some());
    // Out of tolerance
    assert!(idx.lookup(0, 1010, 2000, 2).is_none());
}

#[test]
fn empty_bed_yields_empty_index() {
    let bed = b"";
    let idx = JunctionIndex::from_bed_reader(&bed[..], &test_reference()).unwrap();
    assert!(idx.is_empty());
}

#[test]
fn malformed_lines_are_skipped_not_fatal() {
    let bed = b"chr1\tnotanumber\t2000\tx\t0\t+\nchr1\t1000\t2000\tok\t0\t+\n";
    let idx = JunctionIndex::from_bed_reader(&bed[..], &test_reference()).unwrap();
    assert_eq!(idx.len(), 1, "malformed line skipped, good one kept");
}
