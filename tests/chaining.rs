use kira_ls_aligner::chaining::{ChainingConfig, ChainingStats, chain_anchors};
use kira_ls_aligner::types::{Anchor, Strand};

#[test]
fn chaining_prefers_longer_chain() {
    let anchors = vec![
        Anchor {
            read_start: 0,
            read_end: 10,
            ref_id: 0,
            ref_start: 100,
            ref_end: 110,
            strand: Strand::Forward,
            score: 10,
        },
        Anchor {
            read_start: 12,
            read_end: 22,
            ref_id: 0,
            ref_start: 112,
            ref_end: 122,
            strand: Strand::Forward,
            score: 10,
        },
        Anchor {
            read_start: 30,
            read_end: 35,
            ref_id: 0,
            ref_start: 130,
            ref_end: 135,
            strand: Strand::Forward,
            score: 5,
        },
    ];
    let cfg = ChainingConfig {
        max_dist: 1000,
        max_anchors: 100,
        max_chains: 2,
        gap_open: 2,
        gap_extend: 1,
        log_gap: 0.1,
        rmq_window: 64,
    };
    let mut stats = ChainingStats::default();
    let chains = chain_anchors(&anchors, cfg, &mut stats);
    assert!(!chains.is_empty());
    assert!(chains[0].score >= chains[chains.len() - 1].score);
}

fn cfg() -> ChainingConfig {
    ChainingConfig {
        max_dist: 10_000,
        max_anchors: 100,
        max_chains: 4,
        gap_open: 6,
        gap_extend: 1,
        log_gap: 1.0,
        rmq_window: 64,
    }
}

#[test]
fn collinear_sparse_anchors_are_not_charged_linear_distance() {
    let anchors = vec![
        Anchor {
            read_start: 0,
            read_end: 20,
            ref_id: 0,
            ref_start: 100,
            ref_end: 120,
            strand: Strand::Forward,
            score: 20,
        },
        Anchor {
            read_start: 1000,
            read_end: 1020,
            ref_id: 0,
            ref_start: 1100,
            ref_end: 1120,
            strand: Strand::Forward,
            score: 20,
        },
    ];
    let mut stats = ChainingStats::default();
    let chains = chain_anchors(&anchors, cfg(), &mut stats);
    assert_eq!(chains[0].anchors.len(), 2);
    assert_eq!(chains[0].score, 40);
}

#[test]
fn valid_predecessor_is_used_when_higher_score_candidate_has_bad_geometry() {
    let anchors = vec![
        Anchor {
            read_start: 0,
            read_end: 20,
            ref_id: 0,
            ref_start: 100,
            ref_end: 120,
            strand: Strand::Forward,
            score: 20,
        },
        Anchor {
            read_start: 130,
            read_end: 150,
            ref_id: 0,
            ref_start: 105,
            ref_end: 125,
            strand: Strand::Forward,
            score: 100,
        },
        Anchor {
            read_start: 120,
            read_end: 140,
            ref_id: 0,
            ref_start: 220,
            ref_end: 240,
            strand: Strand::Forward,
            score: 20,
        },
    ];
    let mut stats = ChainingStats::default();
    let chains = chain_anchors(&anchors, cfg(), &mut stats);
    assert!(chains.iter().any(|c| {
        c.anchors.len() == 2 && c.anchors[0].read_start == 0 && c.anchors[1].read_start == 120
    }));
}
