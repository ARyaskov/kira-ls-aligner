//! End-to-end splice-aware alignment.
//!
//! Strategy:
//!   1. Synthesise a 4 kb genomic reference with a GT-AG intron at
//!      `exon1[2000:bp..2200) | intron[2200..2700) | exon2[2700..2900)`.
//!   2. Build a "read" by concatenating `exon1 + exon2` (no intron
//!      content) — the way RNA-seq sees a spliced transcript.
//!   3. Run the splice-aware aligner through the public `align_spliced_chain`
//!      API with a hand-crafted `Chain` covering both exons.
//!   4. Assert the resulting CIGAR contains an `N` op and the
//!      `xs_strand` is `Forward` (the donor `GT` / acceptor `AG` votes).
//!
//! This exercises the splice path without going through the full
//! chaining pipeline, which keeps the test deterministic and fast.

use kira_ls_aligner::alignment::AlignmentConfig;
use kira_ls_aligner::alignment::splice::{
    SpliceConfig, SpliceStrandPolicy, align_spliced_chain, detect_splice_strand,
};
use kira_ls_aligner::types::{
    Anchor, Chain, CigarKind, PairRole, ReadRecord, Strand,
};

fn synth_dna(seed: u64, len: usize) -> Vec<u8> {
    let mut s = seed;
    let bases = b"ACGT";
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        out.push(bases[(s >> 33) as usize & 3]);
    }
    out
}

#[test]
fn signal_table_matches_design() {
    // Forward canonical
    assert_eq!(detect_splice_strand(*b"GT", *b"AG"), Some(Strand::Forward));
    assert_eq!(detect_splice_strand(*b"GC", *b"AG"), Some(Strand::Forward));
    assert_eq!(detect_splice_strand(*b"AT", *b"AC"), Some(Strand::Forward));
    // Reverse
    assert_eq!(detect_splice_strand(*b"CT", *b"AC"), Some(Strand::Reverse));
    assert_eq!(detect_splice_strand(*b"GT", *b"AT"), Some(Strand::Reverse));
    // Non-canonical
    assert!(detect_splice_strand(*b"AA", *b"GG").is_none());
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
fn two_exon_alignment_emits_n_op_and_forward_xs_strand() {
    // Build the reference: 4 kb of random ACGT, with a forced GT-AG
    // intron between positions 2200 and 2700.
    let mut ref_seq = synth_dna(0xDEAD_BEEF_CAFE_BABE, 4000);
    // Force donor GT at position 2200 (start of intron).
    ref_seq[2200] = b'G';
    ref_seq[2201] = b'T';
    // Force acceptor AG at position 2698-2699 (last two bytes of intron).
    ref_seq[2698] = b'A';
    ref_seq[2699] = b'G';

    // Read sequence = exon1[2000..2200) || exon2[2700..2900)
    let mut read_seq = ref_seq[2000..2200].to_vec();
    read_seq.extend_from_slice(&ref_seq[2700..2900]);
    assert_eq!(read_seq.len(), 400, "synthesised read must be 400 bp");

    let read = ReadRecord {
        id: "rna_read_1".to_string(),
        seq: read_seq,
        qual: None,
        pair_role: PairRole::Unpaired,
        repeat_min_occ: 1,
    };

    // Hand-crafted two-anchor chain — one anchor per exon. In a real
    // pipeline the chainer would produce something equivalent if the
    // exons each contributed at least one minimizer.
    let chain = Chain {
        ref_id: 0,
        strand: Strand::Forward,
        score: 350,
        anchors: vec![
            Anchor {
                read_start: 0,
                read_end: 200,
                ref_id: 0,
                ref_start: 2000,
                ref_end: 2200,
                strand: Strand::Forward,
                score: 200,
            },
            Anchor {
                read_start: 200,
                read_end: 400,
                ref_id: 0,
                ref_start: 2700,
                ref_end: 2900,
                strand: Strand::Forward,
                score: 200,
            },
        ],
        read_start: 0,
        read_end: 400,
        ref_start: 2000,
        ref_end: 2900,
    };

    let splice_cfg = SpliceConfig {
        enabled: true,
        min_intron: 30,
        max_intron: 200_000,
        strand_policy: SpliceStrandPolicy::Auto,
        require_signal: false,
        splice_flank: 20,
        min_exon_len: 15,
        polya_min_len: 10,
    };
    let aln = align_spliced_chain(&chain, &read, &ref_seq, align_cfg(), splice_cfg, None, 2)
        .expect("splice alignment should succeed");

    // Expect an N op in the CIGAR.
    let has_n = aln.cigar.iter().any(|op| op.op == CigarKind::Skipped);
    assert!(
        has_n,
        "expected at least one N op in CIGAR, got {:?}",
        aln.cigar
    );
    // Intron length should be 500 (2700-2200).
    let total_n: u32 = aln
        .cigar
        .iter()
        .filter(|op| op.op == CigarKind::Skipped)
        .map(|op| op.len)
        .sum();
    assert_eq!(total_n, 500, "intron length");

    // XS:A: must be `+` because we crafted GT/AG signals (forward canonical).
    assert_eq!(aln.xs_strand, Some(Strand::Forward));
    // Alignment span on reference must cover the whole gene (2000..2900).
    assert_eq!(aln.ref_start, 2000);
    assert_eq!(aln.ref_end, 2900);
    // Read fully consumed.
    assert_eq!(aln.read_start, 0);
    assert_eq!(aln.read_end, 400);
    // NM should be 0 (perfect synthetic).
    assert_eq!(aln.nm, 0, "synthetic read should be perfect match");
}

#[test]
fn small_gap_below_min_intron_stays_as_d_op() {
    // Same setup but the gap between exons is too small to be an intron.
    let mut ref_seq = synth_dna(0x1234_5678_9abc_def0, 1000);
    // No GT/AG forcing — just a short 10 bp gap.
    ref_seq[300] = b'G';
    ref_seq[301] = b'T';
    ref_seq[308] = b'A';
    ref_seq[309] = b'G';

    let mut read_seq = ref_seq[100..300].to_vec();
    read_seq.extend_from_slice(&ref_seq[310..510]);
    let read = ReadRecord {
        id: "x".to_string(),
        seq: read_seq,
        qual: None,
        pair_role: PairRole::Unpaired,
        repeat_min_occ: 1,
    };

    let chain = Chain {
        ref_id: 0,
        strand: Strand::Forward,
        score: 350,
        anchors: vec![
            Anchor {
                read_start: 0,
                read_end: 200,
                ref_id: 0,
                ref_start: 100,
                ref_end: 300,
                strand: Strand::Forward,
                score: 200,
            },
            Anchor {
                read_start: 200,
                read_end: 400,
                ref_id: 0,
                ref_start: 310,
                ref_end: 510,
                strand: Strand::Forward,
                score: 200,
            },
        ],
        read_start: 0,
        read_end: 400,
        ref_start: 100,
        ref_end: 510,
    };

    let splice_cfg = SpliceConfig {
        enabled: true,
        min_intron: 30, // 10 bp gap < 30 → won't be classified as intron
        max_intron: 200_000,
        strand_policy: SpliceStrandPolicy::Auto,
        require_signal: false,
        splice_flank: 20,
        min_exon_len: 15,
        polya_min_len: 10,
    };
    let aln = align_spliced_chain(&chain, &read, &ref_seq, align_cfg(), splice_cfg, None, 2)
        .unwrap();

    let has_n = aln.cigar.iter().any(|op| op.op == CigarKind::Skipped);
    assert!(
        !has_n,
        "10 bp gap is below min_intron; expected D op, got N: {:?}",
        aln.cigar
    );
    // Should have a D op for the 10 bp ref gap.
    let has_d = aln.cigar.iter().any(|op| op.op == CigarKind::Del);
    assert!(has_d, "expected D op for small ref-gap");
    // No XS strand because no junctions were classified.
    assert!(aln.xs_strand.is_none());
}

#[test]
fn non_canonical_signal_with_require_emits_d_not_n() {
    // GA-GA gap — non-canonical. require_signal=true should refuse to
    // mark it as intron and emit D instead.
    let mut ref_seq = synth_dna(0xFEED_FACE_8BAD_F00D, 5000);
    ref_seq[1000] = b'G';
    ref_seq[1001] = b'A'; // donor GA, not GT
    ref_seq[1098] = b'G';
    ref_seq[1099] = b'A'; // acceptor GA, not AG

    let mut read_seq = ref_seq[900..1000].to_vec();
    read_seq.extend_from_slice(&ref_seq[1100..1200]);
    let read = ReadRecord {
        id: "x".to_string(),
        seq: read_seq,
        qual: None,
        pair_role: PairRole::Unpaired,
        repeat_min_occ: 1,
    };

    let chain = Chain {
        ref_id: 0,
        strand: Strand::Forward,
        score: 200,
        anchors: vec![
            Anchor {
                read_start: 0,
                read_end: 100,
                ref_id: 0,
                ref_start: 900,
                ref_end: 1000,
                strand: Strand::Forward,
                score: 100,
            },
            Anchor {
                read_start: 100,
                read_end: 200,
                ref_id: 0,
                ref_start: 1100,
                ref_end: 1200,
                strand: Strand::Forward,
                score: 100,
            },
        ],
        read_start: 0,
        read_end: 200,
        ref_start: 900,
        ref_end: 1200,
    };

    let splice_cfg = SpliceConfig {
        enabled: true,
        min_intron: 30,
        max_intron: 200_000,
        strand_policy: SpliceStrandPolicy::Auto,
        require_signal: true, // reject non-canonical → must be D not N
        splice_flank: 20,
        min_exon_len: 15,
        polya_min_len: 10,
    };
    let aln = align_spliced_chain(&chain, &read, &ref_seq, align_cfg(), splice_cfg, None, 2)
        .unwrap();
    let has_n = aln.cigar.iter().any(|op| op.op == CigarKind::Skipped);
    assert!(
        !has_n,
        "require_signal=true with GA/GA must not emit N: {:?}",
        aln.cigar
    );
}
