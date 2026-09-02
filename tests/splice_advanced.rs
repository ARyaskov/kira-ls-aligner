//! Tests for the splice MVP+ features: site refinement, splice-aware
//! MAPQ score boost, polyA trimming.

use kira_ls_aligner::alignment::AlignmentConfig;
use kira_ls_aligner::alignment::splice::{SpliceConfig, SpliceStrandPolicy, align_spliced_chain};
use kira_ls_aligner::types::{Anchor, Chain, CigarKind, PairRole, ReadRecord, Strand};

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

fn splice_cfg_default() -> SpliceConfig {
    SpliceConfig {
        enabled: true,
        min_intron: 30,
        max_intron: 200_000,
        strand_policy: SpliceStrandPolicy::Auto,
        require_signal: false,
        splice_flank: 20,
        min_exon_len: 15,
        polya_min_len: 10,
    }
}

/// Splice site refinement: when the chainer's boundary is 3 bp shy of
/// the true GT donor, refinement should slide the boundary right by 3 bp
/// (because read_gap=3) and pick the canonical signal.
#[test]
fn refinement_slides_boundary_to_canonical_signal() {
    // Reference layout (positions):
    //   0..1000   random ACGT
    //   1000..1003 GTT — chainer-suggested boundary is at 1000, but...
    //   1003..1005 GT (the *real* donor) — true splice site is here
    //   1005..1499 intron body (random)
    //   1499..1501 AG (acceptor)
    //   1501..2000 random
    let mut ref_seq = synth_dna(0xCAFE_BABE_0001_0002, 2000);
    // No-canonical bases at 1000..1002:
    ref_seq[1000] = b'C';
    ref_seq[1001] = b'C';
    ref_seq[1002] = b'T';
    // Canonical donor 3 bp to the right:
    ref_seq[1003] = b'G';
    ref_seq[1004] = b'T';
    // Acceptor at end:
    ref_seq[1499] = b'A';
    ref_seq[1500] = b'G';

    // Read = exon1[0..1003) || exon2[1501..1800)
    // Anchor 0 ends at read=1000 / ref=1000 (3 bp short of true).
    // Anchor 1 starts at read=1003 / ref=1501 (matches true acceptor).
    // read_gap = 3, ref_gap = 501 — refinement should slide donor to 1003.
    let mut read_seq = ref_seq[0..1003].to_vec();
    read_seq.extend_from_slice(&ref_seq[1501..1800]);
    let read = ReadRecord {
        id: "jitter".to_string(),
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
                read_end: 1000,
                ref_id: 0,
                ref_start: 0,
                ref_end: 1000,
                strand: Strand::Forward,
                score: 200,
            },
            Anchor {
                read_start: 1003,
                read_end: 1302,
                ref_id: 0,
                ref_start: 1501,
                ref_end: 1800,
                strand: Strand::Forward,
                score: 200,
            },
        ],
        read_start: 0,
        read_end: 1302,
        ref_start: 0,
        ref_end: 1800,
    };

    let aln = align_spliced_chain(
        &chain,
        &read,
        &ref_seq,
        align_cfg(),
        splice_cfg_default(),
        None,
        2,
    )
    .expect("alignment should succeed");

    // The refined intron should be 1501 - 1003 = 498 bp (not 1501 - 1000 = 501).
    let n_lengths: Vec<u32> = aln
        .cigar
        .iter()
        .filter(|op| op.op == CigarKind::Skipped)
        .map(|op| op.len)
        .collect();
    assert_eq!(n_lengths.len(), 1, "exactly one N op expected");
    assert_eq!(
        n_lengths[0], 498,
        "intron length should reflect the refined boundary"
    );
    // XS strand inferred from the GT/AG signal at the refined boundary.
    assert_eq!(aln.xs_strand, Some(Strand::Forward));
}

/// Splice-aware MAPQ: an alignment with a canonical-signal junction
/// should score higher than the same alignment with a non-canonical
/// (no-signal) junction. We compare scores directly because MAPQ is
/// derived from score in stage 5.
#[test]
fn canonical_signal_boosts_score_over_non_canonical() {
    let mut ref_canon = synth_dna(0x1111_2222_3333_4444, 4000);
    let mut ref_nc = ref_canon.clone();
    // Canonical splice in `ref_canon`: GT/AG at 2200/2698-2699
    ref_canon[2200] = b'G';
    ref_canon[2201] = b'T';
    ref_canon[2698] = b'A';
    ref_canon[2699] = b'G';
    // Non-canonical in `ref_nc`: AA/CC at the same positions
    ref_nc[2200] = b'A';
    ref_nc[2201] = b'A';
    ref_nc[2698] = b'C';
    ref_nc[2699] = b'C';

    // Build a read mostly from ref_canon (same exon bodies on both
    // references because positions 0..2200 and 2700..2900 of the two
    // refs are different but both consistent with their own ref). For
    // the test we use ref_canon-derived read against each ref in turn.
    let mut read_seq = ref_canon[2000..2200].to_vec();
    read_seq.extend_from_slice(&ref_canon[2700..2900]);
    let read = ReadRecord {
        id: "x".to_string(),
        seq: read_seq.clone(),
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

    let aln_canon = align_spliced_chain(
        &chain,
        &read,
        &ref_canon,
        align_cfg(),
        splice_cfg_default(),
        None,
        2,
    )
    .unwrap();
    // For nc, the per-exon SW score may differ because the underlying ref
    // bases differ — adjust read to match nc's exon bodies for a fair
    // comparison.
    let mut read_nc = ref_nc[2000..2200].to_vec();
    read_nc.extend_from_slice(&ref_nc[2700..2900]);
    let read_nc_record = ReadRecord {
        id: "x".to_string(),
        seq: read_nc,
        qual: None,
        pair_role: PairRole::Unpaired,
        repeat_min_occ: 1,
        comment: None,
    };
    let aln_nc = align_spliced_chain(
        &chain,
        &read_nc_record,
        &ref_nc,
        align_cfg(),
        splice_cfg_default(),
        None,
        2,
    )
    .unwrap();

    // Canonical version should have a higher score because of the +10
    // signal bonus per junction. (The per-exon SW scores are identical
    // because both reads perfectly match their respective references.)
    assert!(
        aln_canon.score > aln_nc.score,
        "canonical-signal junction should boost score: canon={} nc={}",
        aln_canon.score,
        aln_nc.score
    );
}

/// PolyA trimming: a forward-strand read with a 50 bp polyA tail
/// should have those 50 bp soft-clipped, not aligned to reference A's.
#[test]
fn polya_tail_is_soft_clipped() {
    let mut ref_seq = synth_dna(0xDEAD_BEEF_FACE_8BAD, 4000);
    // Force a canonical GT/AG so the splice path emits an N op.
    ref_seq[2200] = b'G';
    ref_seq[2201] = b'T';
    ref_seq[2698] = b'A';
    ref_seq[2699] = b'G';

    // Two-exon read with 50 bp polyA tail at the 3' end.
    let mut read_seq = ref_seq[2000..2200].to_vec();
    read_seq.extend_from_slice(&ref_seq[2700..2900]);
    read_seq.extend(std::iter::repeat(b'A').take(50));
    let original_read_len = read_seq.len();
    let read = ReadRecord {
        id: "polya_read".to_string(),
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

    let aln = align_spliced_chain(
        &chain,
        &read,
        &ref_seq,
        align_cfg(),
        splice_cfg_default(),
        None,
        2,
    )
    .unwrap();

    // Trailing soft-clip should include the 50 bp polyA.
    let last_op = aln.cigar.last().expect("non-empty cigar");
    assert_eq!(
        last_op.op,
        CigarKind::SoftClip,
        "last op should be soft-clip from polyA trim"
    );
    assert!(
        last_op.len >= 50,
        "trailing soft-clip should include the 50 bp polyA, got {} bp",
        last_op.len
    );
    // read_end should not include the polyA tail.
    assert!(
        aln.read_end <= (original_read_len - 50) as u32,
        "read_end should exclude polyA: {} vs original_len-50={}",
        aln.read_end,
        original_read_len - 50
    );
}

/// PolyA disabled (min_len=0): a polyA-tail read stays as-is (no
/// extra soft-clip from trimming).
#[test]
fn polya_disabled_when_min_len_zero() {
    let mut ref_seq = synth_dna(0xAAAA_BBBB_CCCC_DDDD, 4000);
    ref_seq[2200] = b'G';
    ref_seq[2201] = b'T';
    ref_seq[2698] = b'A';
    ref_seq[2699] = b'G';

    let mut read_seq = ref_seq[2000..2200].to_vec();
    read_seq.extend_from_slice(&ref_seq[2700..2900]);
    read_seq.extend(std::iter::repeat(b'A').take(50));
    let read = ReadRecord {
        id: "x".to_string(),
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

    let mut cfg = splice_cfg_default();
    cfg.polya_min_len = 0; // disabled
    let aln = align_spliced_chain(&chain, &read, &ref_seq, align_cfg(), cfg, None, 2).unwrap();

    // Without polyA trim, the trailing soft-clip should be just the
    // unmapped tail (read_len - last anchor.read_end = 450 - 400 = 50)
    // but those 50 bases happen to be the polyA. So a single 50 bp S op
    // is expected and it stays at the same length as before — but it's
    // a *normal* end-of-read soft-clip, not from polyA logic.
    let last_op = aln.cigar.last().unwrap();
    assert_eq!(last_op.op, CigarKind::SoftClip);
    assert_eq!(last_op.len, 50, "natural trailing soft-clip preserved");
}
