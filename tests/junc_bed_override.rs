//! Annotation-guided splice tests — verify that `--junc-bed` lookups
//! cause non-canonical donor/acceptor gaps to still be classified as
//! introns (N op) when the BED file records the junction, and that
//! signal-based classification continues to work alongside it.
//!
//! Drives `align_spliced_chain` directly through the public API so the
//! test stays deterministic (no FASTQ I/O, no chainer dependency).

use std::io::BufReader;

use kira_ls_aligner::alignment::AlignmentConfig;
use kira_ls_aligner::alignment::junc_bed::JunctionIndex;
use kira_ls_aligner::alignment::splice::{SpliceConfig, SpliceStrandPolicy, align_spliced_chain};
use kira_ls_aligner::types::{
    Anchor, Chain, CigarKind, PairRole, ReadRecord, RefBases, RefSeq, Reference, Strand,
};

fn synth_dna(seed: u64, len: usize) -> Vec<u8> {
    let mut s = seed;
    let bases = b"ACGT";
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        out.push(bases[(s >> 33) as usize & 3]);
    }
    out
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

fn two_exon_chain() -> (Vec<u8>, ReadRecord, Chain) {
    // Same skeleton as splice_e2e but WITHOUT forcing GT/AG signal —
    // we want a "non-canonical" gap that we can override via BED.
    let mut ref_seq = synth_dna(0xABCD_1234_5678_FEDC, 4000);
    // Force a non-canonical AA-CC pair at the splice site:
    ref_seq[2200] = b'A';
    ref_seq[2201] = b'A'; // donor "AA" — non-canonical
    ref_seq[2698] = b'C';
    ref_seq[2699] = b'C'; // acceptor "CC" — non-canonical

    let mut read_seq = ref_seq[2000..2200].to_vec();
    read_seq.extend_from_slice(&ref_seq[2700..2900]);
    let read = ReadRecord {
        id: "rna_with_bed".to_string(),
        seq: read_seq,
        qual: None,
        pair_role: PairRole::Unpaired,
        repeat_min_occ: 1,
        comment: None,
    };
    let chain = Chain {
        ref_id: 0,
        strand: Strand::Forward,
        score: 400,
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
    (ref_seq, read, chain)
}

fn ref_with(ref_seq: &[u8]) -> Reference {
    Reference {
        sequences: vec![RefSeq {
            name: "chr1".to_string(),
            bases: RefBases::Owned(ref_seq.to_vec()),
        }],
    }
}

#[test]
fn non_canonical_signal_with_bed_match_emits_n() {
    let (ref_seq, read, chain) = two_exon_chain();
    let reference = ref_with(&ref_seq);

    // BED records the junction at (2200, 2700) on the + strand.
    let bed = b"chr1\t2200\t2700\tjunc1\t0\t+\n";
    let junc_idx = JunctionIndex::from_bed_reader(BufReader::new(&bed[..]), &reference).unwrap();
    assert_eq!(junc_idx.len(), 1);

    let splice_cfg = SpliceConfig {
        enabled: true,
        min_intron: 30,
        max_intron: 200_000,
        strand_policy: SpliceStrandPolicy::Auto,
        require_signal: true, // strict! Without BED override, would emit D
        splice_flank: 20,
        min_exon_len: 15,
        polya_min_len: 10,
    };
    let aln = align_spliced_chain(
        &chain,
        &read,
        &ref_seq,
        align_cfg(),
        splice_cfg,
        Some(&junc_idx),
        2,
    )
    .expect("alignment should succeed");

    // BED-overridden junction → N op despite non-canonical signal.
    let has_n = aln.cigar.iter().any(|op| op.op == CigarKind::Skipped);
    assert!(
        has_n,
        "expected N op from BED override (non-canonical signal + require_signal=true): \
         {:?}",
        aln.cigar
    );
    assert_eq!(
        aln.xs_strand,
        Some(Strand::Forward),
        "BED strand should win"
    );
}

#[test]
fn no_bed_with_non_canonical_signal_and_require_emits_d() {
    let (ref_seq, read, chain) = two_exon_chain();

    let splice_cfg = SpliceConfig {
        enabled: true,
        min_intron: 30,
        max_intron: 200_000,
        strand_policy: SpliceStrandPolicy::Auto,
        require_signal: true,
        splice_flank: 20,
        min_exon_len: 15,
        polya_min_len: 10,
    };
    // No junc-bed → require_signal=true rejects non-canonical → D op.
    let aln =
        align_spliced_chain(&chain, &read, &ref_seq, align_cfg(), splice_cfg, None, 2).unwrap();

    let has_n = aln.cigar.iter().any(|op| op.op == CigarKind::Skipped);
    assert!(
        !has_n,
        "without BED override, require_signal=true must reject non-canonical: \
         {:?}",
        aln.cigar
    );
}

#[test]
fn bed_with_tolerance_matches_offset_junction() {
    let (ref_seq, read, chain) = two_exon_chain();
    let reference = ref_with(&ref_seq);

    // BED entry is ±2 bp off from the actual chain boundaries.
    let bed = b"chr1\t2202\t2699\tjunc1\t0\t-\n";
    let junc_idx = JunctionIndex::from_bed_reader(BufReader::new(&bed[..]), &reference).unwrap();

    let splice_cfg = SpliceConfig {
        enabled: true,
        min_intron: 30,
        max_intron: 200_000,
        strand_policy: SpliceStrandPolicy::Auto,
        require_signal: true,
        splice_flank: 20,
        min_exon_len: 15,
        polya_min_len: 10,
    };
    let aln = align_spliced_chain(
        &chain,
        &read,
        &ref_seq,
        align_cfg(),
        splice_cfg,
        Some(&junc_idx),
        3, // tolerance 3 should catch the ±2 offset
    )
    .unwrap();
    let has_n = aln.cigar.iter().any(|op| op.op == CigarKind::Skipped);
    assert!(has_n, "tolerance lookup should accept ±2 bp offset");
    // BED records '-' strand — the override wins over signal-derived
    // strand (which would be None since signal is AA/CC non-canonical).
    assert_eq!(aln.xs_strand, Some(Strand::Reverse));
}

#[test]
fn bed_with_zero_tolerance_rejects_offset_junction() {
    let (ref_seq, read, chain) = two_exon_chain();
    let reference = ref_with(&ref_seq);

    // BED at ±2 bp offset; tolerance 0 should NOT match.
    let bed = b"chr1\t2202\t2699\tjunc1\t0\t-\n";
    let junc_idx = JunctionIndex::from_bed_reader(BufReader::new(&bed[..]), &reference).unwrap();

    let splice_cfg = SpliceConfig {
        enabled: true,
        min_intron: 30,
        max_intron: 200_000,
        strand_policy: SpliceStrandPolicy::Auto,
        require_signal: true,
        splice_flank: 20,
        min_exon_len: 15,
        polya_min_len: 10,
    };
    let aln = align_spliced_chain(
        &chain,
        &read,
        &ref_seq,
        align_cfg(),
        splice_cfg,
        Some(&junc_idx),
        0, // exact match only
    )
    .unwrap();
    let has_n = aln.cigar.iter().any(|op| op.op == CigarKind::Skipped);
    assert!(
        !has_n,
        "tolerance=0 + offset BED entry should not match; expected D op fallback"
    );
}
