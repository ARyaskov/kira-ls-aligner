//! Split-read / supplementary alignment integration test.
//!
//! Constructs a chimeric scenario: a 200 bp read whose first 100 bp match
//! one reference region and last 100 bp match a *different* region (the
//! signature of a structural variant — translocation, large insertion, or
//! a read crossing a repeat boundary). After stage 4 +
//! `emit_supplementary_alignments`, the read must have:
//!   * A primary alignment at one of the two regions (slot 0).
//!   * A supplementary alignment at the other (`is_supplementary = true`).
//!   * Together they cover non-overlapping read regions.
//!
//! This is the test that would have caught the "0 supplementary records"
//! regression that motivated the split-read pass.

use kira_ls_aligner::alignment::AlignmentConfig;
use kira_ls_aligner::index::{Index, IndexConfig};
use kira_ls_aligner::pipeline::split_read::emit_supplementary_alignments;
use kira_ls_aligner::types::{
    Alignment, AlignmentKind, Chain, CigarKind, CigarOp, MateInfo, PairRole, ReadRecord, RefBases,
    RefSeq, Reference, Strand,
};

/// Deterministic ACGT stream — good enough that 100-bp anchors aren't
/// shared by chance between the two halves of the reference.
fn rand_dna(len: usize, seed: u64) -> Vec<u8> {
    let mut seed = seed;
    let bases = b"ACGT";
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        out.push(bases[(seed >> 33) as usize & 3]);
    }
    out
}

/// Two 2 kb contigs of distinct random DNA. We'll splice a chimeric read
/// from one half of each — neither half overlaps the other on the read,
/// which is exactly the split-read pattern bwa-mem flags with 0x800.
fn chimeric_reference() -> Reference {
    Reference {
        sequences: vec![
            RefSeq {
                name: "chrA".to_string(),
                bases: RefBases::Owned(rand_dna(2000, 0x1111_2222_3333_4444)),
            },
            RefSeq {
                name: "chrB".to_string(),
                bases: RefBases::Owned(rand_dna(2000, 0xAAAA_BBBB_CCCC_DDDD)),
            },
        ],
    }
}

fn build_index(reference: &Reference) -> Index {
    Index::build(
        reference.clone(),
        IndexConfig {
            short_k: 19,
            short_w: 10,
            long_k: 19,
            long_w: 10,
            max_occ: 500,
            build_short: true,
            build_long: false,
        },
    )
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

fn ref_bytes<'a>(reference: &'a Reference, ref_id: usize) -> &'a [u8] {
    match &reference.sequences[ref_id].bases {
        RefBases::Owned(v) => v.as_slice(),
        _ => panic!("expected owned"),
    }
}

/// A bare alignment standing in for stage 4's primary output — just
/// enough fields for the supplementary pass to read `read_start`,
/// `read_end`, `score`. The pass uses the primary alignment + the chain
/// list; the chains here come from a hand-built list (the chaining stage
/// is not exercised because we want to isolate the supplementary logic).
fn primary_aln(read_start: u32, read_end: u32, ref_id: u32, ref_start: u32) -> Alignment {
    let n = read_end - read_start;
    Alignment {
        kind: AlignmentKind::DpAligned,
        ref_id,
        ref_start,
        ref_end: ref_start + n,
        read_start,
        read_end,
        cigar: vec![CigarOp {
            len: n,
            op: CigarKind::Match,
        }],
        score: n as i32,
        mapq: 60,
        is_rev: false,
        is_secondary: false,
        is_supplementary: false,
        nm: 0,
        md: format!("{n}"),
        as_score: n as i32,
        xs_score: None,
        xs_strand: None,
        mate: MateInfo::default(),
    }
}

fn chain(read_start: u32, read_end: u32, ref_id: u32, ref_start: u32, score: i32) -> Chain {
    Chain {
        anchors: Vec::new(),
        score,
        ref_id,
        read_start,
        read_end,
        ref_start,
        ref_end: ref_start + (read_end - read_start),
        strand: Strand::Forward,
    }
}

#[test]
fn chimeric_read_emits_supplementary() {
    let reference = chimeric_reference();
    let index = build_index(&reference);
    let a = ref_bytes(&reference, 0);
    let b = ref_bytes(&reference, 1);

    // Splice a 200-bp chimeric read: first 100 bp from chrA[300..400],
    // last 100 bp from chrB[1000..1100]. Stage 4's primary alignment
    // would land on chrA covering read[0..100] (the highest-scoring
    // chain); we want the supplementary pass to also align the chrB
    // chain covering read[100..200] and mark it 0x800.
    let mut read_seq = Vec::with_capacity(200);
    read_seq.extend_from_slice(&a[300..400]);
    read_seq.extend_from_slice(&b[1000..1100]);

    let read = ReadRecord {
        id: "chimeric_read".to_string(),
        seq: read_seq,
        qual: None,
        pair_role: PairRole::Unpaired,
        repeat_min_occ: 1,
        comment: None,
    };

    // Hand-built chain list: stage 4's primary (chrA) at slot 0,
    // disjoint chrB chain at slot 1. Realistic stage 3 output for this
    // synthetic scenario.
    let primary_chain = chain(0, 100, 0, 300, 100);
    let supp_chain = chain(100, 200, 1, 1000, 100);
    let chains = vec![vec![primary_chain, supp_chain]];

    // Stage 4 already produced the primary; the supplementary pass adds
    // the chimeric chunk.
    let primary = primary_aln(0, 100, 0, 300);
    let mut alignments: Vec<Vec<Alignment>> = vec![vec![primary]];

    emit_supplementary_alignments(
        std::slice::from_ref(&read),
        &mut alignments,
        &chains,
        &index,
        align_cfg(),
    );

    let alns = &alignments[0];
    assert_eq!(
        alns.len(),
        2,
        "expected primary + 1 supplementary, got {}",
        alns.len()
    );

    // Slot 0 stays the primary (split-read pass is additive).
    assert!(!alns[0].is_supplementary);

    // The added record is on chrB and marked supplementary.
    let supp = &alns[1];
    assert!(
        supp.is_supplementary,
        "second alignment should be is_supplementary = true"
    );
    assert_eq!(supp.ref_id, 1, "supplementary should map to chrB");
    // Read region of the supp should land in the second half (≈ [100..200])
    // — the SW kernel may shift by a few bases, so we just check the
    // bulk is past the midpoint.
    assert!(
        supp.read_start >= 90,
        "supplementary read_start = {} should be past primary's end",
        supp.read_start
    );
}

#[test]
fn no_supplementary_when_only_one_chain() {
    // Degenerate case: single chain → nothing to do. Just confirms the
    // pass doesn't accidentally invent a supplementary from thin air.
    let reference = chimeric_reference();
    let index = build_index(&reference);
    let read = ReadRecord {
        id: "uniq".to_string(),
        seq: ref_bytes(&reference, 0)[300..400].to_vec(),
        qual: None,
        pair_role: PairRole::Unpaired,
        repeat_min_occ: 1,
        comment: None,
    };
    let chains = vec![vec![chain(0, 100, 0, 300, 100)]];
    let mut alignments: Vec<Vec<Alignment>> = vec![vec![primary_aln(0, 100, 0, 300)]];

    emit_supplementary_alignments(
        std::slice::from_ref(&read),
        &mut alignments,
        &chains,
        &index,
        align_cfg(),
    );

    assert_eq!(alignments[0].len(), 1, "no supp should be added");
}

#[test]
fn no_supplementary_for_overlapping_chains() {
    // Two chains that both cover the same read region (a typical
    // ambiguous repeat mapping) → second is a *secondary* candidate, not
    // a supplementary. The split-read pass should reject it.
    let reference = chimeric_reference();
    let index = build_index(&reference);
    let a = ref_bytes(&reference, 0);
    let b = ref_bytes(&reference, 1);

    // Read matches chrA[300..400] perfectly. Same 100 bp also happens to
    // exist (synthetically) somewhere on chrB — chain list reflects that.
    let read = ReadRecord {
        id: "ambiguous".to_string(),
        seq: a[300..400].to_vec(),
        qual: None,
        pair_role: PairRole::Unpaired,
        repeat_min_occ: 1,
        comment: None,
    };
    // Both chains cover read[0..100] — 100% overlap.
    let chains = vec![vec![chain(0, 100, 0, 300, 100), chain(0, 100, 1, 500, 90)]];
    let _ = b; // not used for content; the index already knows chrB exists
    let mut alignments: Vec<Vec<Alignment>> = vec![vec![primary_aln(0, 100, 0, 300)]];

    emit_supplementary_alignments(
        std::slice::from_ref(&read),
        &mut alignments,
        &chains,
        &index,
        align_cfg(),
    );

    assert_eq!(
        alignments[0].len(),
        1,
        "overlapping co-region chain must not become supplementary"
    );
}
