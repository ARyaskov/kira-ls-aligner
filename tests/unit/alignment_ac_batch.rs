use super::*;
use crate::types::{PairRole, RefBases, RefSeq, Reference};

fn aln_cfg() -> AlignmentConfig {
    AlignmentConfig {
        match_score: 1,
        mismatch: 4,
        gap_open: 6,
        gap_extend: 1,
        bandwidth: 20,
        xdrop: 100,
        clip_penalty: 5,
    }
}

fn make_ref(seqs: Vec<(&str, &[u8])>) -> Reference {
    Reference {
        sequences: seqs
            .into_iter()
            .map(|(name, bases)| RefSeq {
                name: name.to_string(),
                bases: RefBases::Owned(bases.to_vec()),
            })
            .collect(),
    }
}

fn read(id: &str, seq: &[u8]) -> ReadRecord {
    ReadRecord {
        id: id.to_string(),
        seq: seq.to_vec(),
        qual: None,
        pair_role: PairRole::Unpaired,
        repeat_min_occ: 1,
    }
}

fn make_index(reference: Reference) -> Index {
    // The AC stage only touches `index.ref_bases` / `index.reference`, so
    // the cheapest valid `Index` is one built with both sub-indices skipped.
    let cfg = crate::index::IndexConfig {
        short_k: 5,
        short_w: 1,
        long_k: 5,
        long_w: 1,
        max_occ: 16,
        build_short: false,
        build_long: false,
    };
    Index::build(reference, cfg)
}

/// Pad a small `reads` batch so the AC stage is not short-circuited by
/// the `MIN_ELIGIBLE_READS` heuristic. Decoy reads are pure-G; their RC
/// is pure-C, so they never collide with the A/G/C/T patterns used in
/// the tests below.
fn pad_with_decoys(reads: &mut Vec<ReadRecord>) {
    let decoy = b"GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG".to_vec(); // 44 bp
    for i in 0..MIN_ELIGIBLE_READS {
        reads.push(read(&format!("decoy{i}"), &decoy));
    }
}

/// 48 bp non-palindromic pattern: RC is `"CCCCCCTTTTTT" * 4`, which shares
/// no substring of length >= 30 with the forward pattern.
fn asymmetric_pattern() -> Vec<u8> {
    b"AAAAAAGGGGGGAAAAAAGGGGGGAAAAAAGGGGGGAAAAAAGGGGGG".to_vec()
}

#[test]
fn forward_exact_match_yields_one_alignment() {
    let pattern = asymmetric_pattern();
    let mut reference_bytes = vec![b'A'; 30];
    reference_bytes.extend_from_slice(&pattern);
    reference_bytes.extend_from_slice(&vec![b'A'; 30]);
    let reference = make_ref(vec![("ref1", &reference_bytes)]);
    let index = make_index(reference);

    let mut reads = vec![read("r0", &pattern)];
    pad_with_decoys(&mut reads);
    let out = run(&reads, &index, aln_cfg(), 1);

    assert_eq!(out.alignments.len(), reads.len(), "one slot per read");
    assert_eq!(out.alignments[0].len(), 1, "expected exactly one hit");
    let aln = &out.alignments[0][0];
    assert_eq!(aln.ref_id, 0);
    assert_eq!(aln.ref_start, 30);
    assert_eq!(aln.ref_end, 30 + pattern.len() as u32);
    assert!(!aln.is_rev);
    assert_eq!(aln.nm, 0);
    assert_eq!(aln.cigar.len(), 1);
    assert_eq!(aln.cigar[0].op, CigarKind::Match);
    assert_eq!(aln.cigar[0].len as usize, pattern.len());
}

#[test]
fn reverse_complement_match_marks_is_rev() {
    // The reference contains RC(pattern) but never the forward pattern.
    let pattern = asymmetric_pattern();
    let rc = crate::seq::reverse_complement(&pattern);
    let mut reference_bytes = vec![b'A'; 20];
    reference_bytes.extend_from_slice(&rc);
    reference_bytes.extend_from_slice(&vec![b'A'; 20]);
    let reference = make_ref(vec![("ref1", &reference_bytes)]);
    let index = make_index(reference);

    let mut reads = vec![read("r0", &pattern)];
    pad_with_decoys(&mut reads);
    let out = run(&reads, &index, aln_cfg(), 1);

    assert_eq!(out.alignments[0].len(), 1, "expected exactly one RC hit");
    let aln = &out.alignments[0][0];
    assert!(aln.is_rev, "RC match must set is_rev");
    assert_eq!(aln.ref_start, 20);
    assert_eq!(aln.nm, 0);
}

#[test]
fn read_with_n_falls_through() {
    let mut read_seq = asymmetric_pattern();
    read_seq[10] = b'N';
    let pattern_clean = asymmetric_pattern();
    let mut reference_bytes = vec![b'A'; 20];
    reference_bytes.extend_from_slice(&pattern_clean);
    reference_bytes.extend_from_slice(&vec![b'A'; 20]);
    let reference = make_ref(vec![("ref1", &reference_bytes)]);
    let index = make_index(reference);

    let mut reads = vec![read("r0", &read_seq)];
    pad_with_decoys(&mut reads);
    let out = run(&reads, &index, aln_cfg(), 1);
    assert!(
        out.alignments[0].is_empty(),
        "N-bearing read must fall through to cascade"
    );
}

#[test]
fn short_read_below_min_pattern_len_is_skipped() {
    // 20 bp read is below MIN_PATTERN_LEN=30 and should not be matched.
    let pattern = b"AAAAAAGGGGGGAAAAAAGGGG".to_vec(); // 22 bp < 30
    let mut reference_bytes = vec![b'A'; 20];
    reference_bytes.extend_from_slice(&pattern);
    reference_bytes.extend_from_slice(&vec![b'A'; 20]);
    let reference = make_ref(vec![("ref1", &reference_bytes)]);
    let index = make_index(reference);

    let mut reads = vec![read("r0", &pattern)];
    pad_with_decoys(&mut reads);
    let out = run(&reads, &index, aln_cfg(), 1);
    assert!(
        out.alignments[0].is_empty(),
        "sub-MIN_PATTERN_LEN read must not be matched"
    );
}

#[test]
fn empty_batch_returns_empty_output() {
    let reference = make_ref(vec![("ref1", b"ACGTACGT")]);
    let index = make_index(reference);
    let out = run(&[], &index, aln_cfg(), 1);
    assert!(out.alignments.is_empty());
    assert_eq!(out.stats.n_reads, 0);
}

#[test]
fn ambiguous_exact_match_falls_through_when_only_one_alignment_requested() {
    let pattern = asymmetric_pattern();
    let mut reference_bytes = Vec::new();
    reference_bytes.extend_from_slice(&pattern);
    reference_bytes.extend_from_slice(&vec![b'T'; 20]);
    reference_bytes.extend_from_slice(&pattern);
    let index = make_index(make_ref(vec![("ref1", &reference_bytes)]));
    let mut reads = vec![read("repeat", &pattern)];
    pad_with_decoys(&mut reads);

    let out = run(&reads, &index, aln_cfg(), 1);
    assert!(out.alignments[0].is_empty());
    assert_eq!(out.stats.reads_ambiguous, 1);
}

#[test]
fn ambiguous_exact_match_retains_competitors_when_requested() {
    let pattern = asymmetric_pattern();
    let mut reference_bytes = Vec::new();
    reference_bytes.extend_from_slice(&pattern);
    reference_bytes.extend_from_slice(&vec![b'T'; 20]);
    reference_bytes.extend_from_slice(&pattern);
    let index = make_index(make_ref(vec![("ref1", &reference_bytes)]));
    let mut reads = vec![read("repeat", &pattern)];
    pad_with_decoys(&mut reads);

    let out = run(&reads, &index, aln_cfg(), 2);
    assert_eq!(out.alignments[0].len(), 2);
    assert_eq!(out.alignments[0][0].score, out.alignments[0][1].score);
}
