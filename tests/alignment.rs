use kira_ls_aligner::alignment::{AlignmentConfig, AnchorSpan, align_chain};
use kira_ls_aligner::types::{ReadRecord, Strand};

#[test]
fn alignment_finds_full_match() {
    let read = ReadRecord {
        id: "r1".to_string(),
        seq: b"ACGTACGT".to_vec(),
        qual: None,
    };
    let reference = b"TTACGTACGTGG".to_vec();
    let span = AnchorSpan {
        ref_id: 0,
        ref_start: 2,
        ref_end: 10,
        read_start: 0,
        read_end: 8,
        strand: Strand::Forward,
    };
    let cfg = AlignmentConfig {
        match_score: 2,
        mismatch: 4,
        gap_open: 6,
        gap_extend: 1,
        bandwidth: 50,
        xdrop: 40,
    };
    let aln = align_chain(&read, &reference, &span, cfg, i32::MIN / 4);
    let cigar = aln
        .cigar
        .iter()
        .map(|op| op.to_string())
        .collect::<String>();
    assert!(cigar.contains("8M"));
    assert_eq!(aln.nm, 0);
}
