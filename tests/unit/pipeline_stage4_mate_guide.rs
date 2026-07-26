//! Mate-guided candidate search (stage 4).
//!
//! The load-bearing invariant is the *uniqueness* requirement: inside a repeat
//! family the mate's own chains land on every copy, so "has a concordant
//! partner" is true for all of them and discriminates nothing. Promoting anyway
//! picks an arbitrary copy and suppresses the real competitor from the top-k DP,
//! producing confidently-scored misplacements.

use super::*;
use crate::pipeline::pairing::PairedConfig;

fn cfg() -> PairedConfig {
    PairedConfig {
        mode: crate::io::IngestMode::TwoFile,
        insert_min: 100,
        insert_max: 600,
        insert_mean: 350,
        insert_sd: 50,
        estimator_locked: false,
    }
}

fn chain(ref_id: u32, ref_start: u32, strand: Strand) -> Chain {
    Chain {
        anchors: Vec::new(),
        score: 100,
        ref_id,
        read_start: 0,
        read_end: 150,
        ref_start,
        ref_end: ref_start + 150,
        strand,
    }
}

#[test]
fn convergent_pair_inside_window_looks_paired() {
    let window = crate::pipeline::pairing::concordance_window(&cfg());
    let fwd = chain(0, 1_000, Strand::Forward);
    let rev = chain(0, 1_200, Strand::Reverse);
    assert!(chains_look_paired(&fwd, &rev, window));
    // Symmetric: the leftmost mate is the one that must be forward.
    assert!(chains_look_paired(&rev, &fwd, window));
}

#[test]
fn divergent_orientation_is_not_a_pair() {
    let window = crate::pipeline::pairing::concordance_window(&cfg());
    let left = chain(0, 1_000, Strand::Reverse);
    let right = chain(0, 1_200, Strand::Forward);
    assert!(!chains_look_paired(&left, &right, window));
}

#[test]
fn same_strand_or_other_contig_is_not_a_pair() {
    let window = crate::pipeline::pairing::concordance_window(&cfg());
    let a = chain(0, 1_000, Strand::Forward);
    assert!(!chains_look_paired(&a, &chain(0, 1_200, Strand::Forward), window));
    assert!(!chains_look_paired(&a, &chain(1, 1_200, Strand::Reverse), window));
}

#[test]
fn fragment_outside_the_insert_window_is_not_a_pair() {
    let window = crate::pipeline::pairing::concordance_window(&cfg());
    let fwd = chain(0, 1_000, Strand::Forward);
    // insert_max is 600; 1_000..9_000+150 is far outside it.
    assert!(!chains_look_paired(&fwd, &chain(0, 9_000, Strand::Reverse), window));
}

#[test]
fn picks_the_single_mate_plausible_locus() {
    let chains = vec![
        chain(0, 500_000, Strand::Forward), // best-scoring, mate-implausible
        chain(0, 900_000, Strand::Forward), // ditto
        chain(0, 1_000, Strand::Forward),   // the one the mate supports
    ];
    let mate = vec![chain(0, 1_200, Strand::Reverse)];
    assert_eq!(
        unique_mate_supported_chain(&chains, &mate, &cfg(), 5),
        Some(2)
    );
}

#[test]
fn repeat_family_with_several_plausible_copies_yields_no_pick() {
    // Both copies of the repeat pair equally well with a mate chain, so the
    // mate carries no information and must not reorder anything.
    let chains = vec![
        chain(0, 1_000, Strand::Forward),
        chain(0, 50_000, Strand::Forward),
    ];
    let mate = vec![
        chain(0, 1_200, Strand::Reverse),
        chain(0, 50_200, Strand::Reverse),
    ];
    assert_eq!(unique_mate_supported_chain(&chains, &mate, &cfg(), 5), None);
}

#[test]
fn no_pick_without_a_mate_or_without_a_choice() {
    let chains = vec![
        chain(0, 1_000, Strand::Forward),
        chain(0, 900_000, Strand::Forward),
    ];
    // No mate chains at all (single-end, or the mate was unmapped).
    assert_eq!(unique_mate_supported_chain(&chains, &[], &cfg(), 5), None);
    // A single candidate is not a choice.
    let mate = vec![chain(0, 1_200, Strand::Reverse)];
    assert_eq!(
        unique_mate_supported_chain(&chains[..1], &mate, &cfg(), 5),
        None
    );
}

#[test]
fn support_search_is_bounded_by_max_k() {
    // Only the first `max_k` candidates are probed, so a plausible locus ranked
    // below the cutoff is not picked (and cannot silently cost O(n*m) work).
    let mut chains = vec![chain(0, 500_000, Strand::Forward)];
    for i in 0..8 {
        chains.push(chain(0, 600_000 + i * 10_000, Strand::Forward));
    }
    chains.push(chain(0, 1_000, Strand::Forward));
    let mate = vec![chain(0, 1_200, Strand::Reverse)];
    assert_eq!(unique_mate_supported_chain(&chains, &mate, &cfg(), 3), None);
    let last = chains.len();
    assert_eq!(
        unique_mate_supported_chain(&chains, &mate, &cfg(), last),
        Some(last - 1)
    );
}
