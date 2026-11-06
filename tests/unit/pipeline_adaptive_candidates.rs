use super::*;
use crate::types::{RefBases, RefSeq, Reference};

fn config() -> AlignmentConfig {
    AlignmentConfig {
        match_score: 1,
        mismatch: 4,
        gap_open: 6,
        gap_extend: 1,
        bandwidth: 50,
        xdrop: 100,
        clip_penalty: 5,
    }
}

fn sequence() -> Vec<u8> {
    let mut s = 0x123456789abcdef0u64;
    (0..400)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            b"ACGT"[(s >> 33) as usize % 4]
        })
        .collect()
}

fn index(sequences: Vec<Vec<u8>>) -> Index {
    Index::build(
        Reference {
            sequences: sequences
                .into_iter()
                .enumerate()
                .map(|(i, bases)| RefSeq {
                    name: format!("c{i}"),
                    bases: RefBases::Owned(bases),
                })
                .collect(),
        },
        crate::index::IndexConfig {
            build_short: false,
            build_long: false,
            ..Default::default()
        },
    )
}

fn chains(n: usize) -> Vec<Chain> {
    (0..n)
        .map(|ref_id| Chain {
            ref_id: ref_id as u32,
            strand: Strand::Forward,
            read_start: 0,
            read_end: 150,
            ref_start: 100,
            ref_end: 250,
            score: 100,
            anchors: Vec::new(),
        })
        .collect()
}

#[test]
fn exact_ties_keep_two_cached_candidates_without_redundant_tracebacks() {
    let seq = sequence();
    let idx = index(vec![seq.clone(); 4]);
    let ranking =
        two_tier_reorder(&seq[100..250], &mut None, &chains(4), &idx, config(), 0, 5).unwrap();
    assert_eq!(ranking.dp_limit, 2);
    assert_eq!(ranking.order, vec![0, 1, 2, 3]);
    assert!(ranking.scores.iter().all(|&s| s == Some(150)));
    assert!(
        ranking.cached.iter().all(Option::is_none),
        "exact evidence needs no CIGAR/MD allocation"
    );
}

#[test]
fn clear_winner_saves_tracebacks_and_keeps_measured_competitor_score() {
    let seq = sequence();
    let mut other = seq.clone();
    for at in [130, 160, 190] {
        other[at] = if other[at] == b'A' { b'C' } else { b'A' };
    }
    let idx = index(vec![other, seq.clone()]);
    let ranking =
        two_tier_reorder(&seq[100..250], &mut None, &chains(2), &idx, config(), 0, 5).unwrap();
    assert_eq!(ranking.order, vec![1, 0]);
    assert_eq!(ranking.dp_limit, 1);
    assert_eq!(ranking.scores, vec![Some(135), Some(150)]);
}

#[test]
fn failed_bounded_probe_keeps_two_candidate_budget() {
    let seq = sequence();
    let idx = index(vec![vec![b'A'; 400], seq.clone()]);
    let ranking =
        two_tier_reorder(&seq[100..250], &mut None, &chains(2), &idx, config(), 0, 5).unwrap();
    assert_eq!(ranking.dp_limit, 2);
    assert!(ranking.scores[0].is_none());
}

#[test]
fn distinguishable_near_ties_expand_to_three_candidates() {
    let seq = sequence();
    let mut one = seq.clone();
    one[140] = if one[140] == b'A' { b'C' } else { b'A' };
    let mut two = one.clone();
    two[200] = if two[200] == b'A' { b'C' } else { b'A' };
    let idx = index(vec![seq.clone(), one, two]);
    let ranking =
        two_tier_reorder(&seq[100..250], &mut None, &chains(3), &idx, config(), 0, 5).unwrap();
    assert_eq!(ranking.dp_limit, 3);
    assert_eq!(ranking.scores, vec![Some(150), Some(145), Some(140)]);
}
