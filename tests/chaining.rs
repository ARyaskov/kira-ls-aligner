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
