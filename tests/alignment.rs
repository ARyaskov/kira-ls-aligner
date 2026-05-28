use kira_ls_aligner::alignment::{AlignmentConfig, AnchorSpan, align_chain};
use kira_ls_aligner::types::{CigarKind, ReadRecord, Strand};

#[test]
fn alignment_finds_full_match() {
    let read = ReadRecord::new_unpaired("r1".to_string(), b"ACGTACGT".to_vec(), None);
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
        clip_penalty: 5,
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

fn end_indel_fixture() -> (Vec<u8>, Vec<u8>) {
    let base: Vec<u8> = (0..150).map(|i| b"CGTA"[i % 4]).collect();
    let mut reference: Vec<u8> = b"TT".to_vec();
    reference.extend_from_slice(&base);
    reference.extend_from_slice(b"TT");
    let mut read_seq: Vec<u8> = base[..145].to_vec();
    read_seq.push(b'G');
    read_seq.extend_from_slice(&base[145..149]);
    assert_eq!(read_seq.len(), 150);
    (reference, read_seq)
}

#[test]
fn end_of_read_insertion_emits_i_op_not_softclip() {
    let (reference, read_seq) = end_indel_fixture();
    let read = ReadRecord::new_unpaired("r1".to_string(), read_seq, None);
    let span = AnchorSpan {
        ref_id: 0,
        ref_start: 2,
        ref_end: 147,
        read_start: 0,
        read_end: 145,
        strand: Strand::Forward,
    };
    let cfg = AlignmentConfig {
        match_score: 1,
        mismatch: 4,
        gap_open: 6,
        gap_extend: 1,
        bandwidth: 50,
        xdrop: 50,
        clip_penalty: 5,
    };

    let aln = align_chain(&read, &reference, &span, cfg, i32::MIN / 4);
    let has_i = aln.cigar.iter().any(|op| op.op == CigarKind::Ins);
    let trailing_soft_len = aln
        .cigar
        .last()
        .filter(|op| op.op == CigarKind::SoftClip)
        .map(|op| op.len)
        .unwrap_or(0);
    let cigar_str = aln
        .cigar
        .iter()
        .map(|op| op.to_string())
        .collect::<String>();
    assert!(
        has_i,
        "expected I op in CIGAR with clip_penalty in effect, got: {}",
        cigar_str
    );
    assert!(
        trailing_soft_len <= 1,
        "expected no large trailing soft-clip, got: {}",
        cigar_str
    );
}

#[test]
fn end_of_read_insertion_softclips_without_penalty() {
    let (reference, read_seq) = end_indel_fixture();
    let read = ReadRecord::new_unpaired("r1".to_string(), read_seq, None);
    let span = AnchorSpan {
        ref_id: 0,
        ref_start: 2,
        ref_end: 147,
        read_start: 0,
        read_end: 145,
        strand: Strand::Forward,
    };
    let cfg = AlignmentConfig {
        match_score: 1,
        mismatch: 4,
        gap_open: 6,
        gap_extend: 1,
        bandwidth: 50,
        xdrop: 50,
        clip_penalty: 0,
    };

    let aln = align_chain(&read, &reference, &span, cfg, i32::MIN / 4);
    let has_i = aln.cigar.iter().any(|op| op.op == CigarKind::Ins);
    assert!(
        !has_i,
        "with clip_penalty=0 the local-SW soft-clip should still win"
    );
}
