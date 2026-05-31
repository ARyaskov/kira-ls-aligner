use super::*;
use crate::types::{RefBases, RefSeq};

fn ref_with(sizes: &[usize]) -> Reference {
    Reference {
        sequences: sizes
            .iter()
            .enumerate()
            .map(|(i, &n)| RefSeq {
                name: format!("c{i}"),
                bases: RefBases::Owned(vec![b'A'; n]),
            })
            .collect(),
    }
}

fn assert_complete(plan: &TilePlan, n_contigs: usize) {
    // Tiles must cover [0, n_contigs) exactly with no overlap.
    let mut cursor = 0usize;
    for t in &plan.tiles {
        assert_eq!(t.contig_start, cursor, "non-contiguous tile start");
        assert_eq!(t.global_ref_id_offset, t.contig_start as u32);
        cursor = t.contig_end;
    }
    assert_eq!(cursor, n_contigs);
}

#[test]
fn empty_reference_zero_tiles() {
    let plan = plan_tiles(&ref_with(&[]), 1_000_000);
    assert_eq!(plan.n_tiles(), 0);
}

#[test]
fn small_reference_one_tile() {
    let plan = plan_tiles(&ref_with(&[100, 200, 300]), 10_000);
    assert_eq!(plan.n_tiles(), 1);
    assert_eq!(plan.tiles[0].total_bytes, 600);
    assert_complete(&plan, 3);
}

#[test]
fn greedy_packing_respects_budget() {
    let plan = plan_tiles(&ref_with(&[2, 5, 1, 3]), 4);
    assert_eq!(plan.n_tiles(), 3);
    assert_eq!(plan.tiles[0].contig_start, 0);
    assert_eq!(plan.tiles[0].contig_end, 1);
    assert_eq!(plan.tiles[1].contig_start, 1);
    assert_eq!(plan.tiles[1].contig_end, 2);
    assert_eq!(plan.tiles[2].contig_start, 2);
    assert_eq!(plan.tiles[2].contig_end, 4);
    assert_complete(&plan, 4);
}

#[test]
fn oversized_singleton_contig_isolated() {
    let plan = plan_tiles(&ref_with(&[10, 100, 5]), 50);
    // c0 (10) fits. c1 (100) > 50 — flush c0, then c1 alone, then c2 alone.
    assert_eq!(plan.n_tiles(), 3);
    assert_eq!(plan.tiles[0].contig_end, 1);
    assert_eq!(plan.tiles[1].contig_start, 1);
    assert_eq!(plan.tiles[1].contig_end, 2);
    assert_complete(&plan, 3);
}

#[test]
fn global_ref_id_remap() {
    let plan = plan_tiles(&ref_with(&[100, 100, 100, 100]), 200);
    // Two tiles: c0+c1, c2+c3.
    assert_eq!(plan.n_tiles(), 2);
    assert_eq!(plan.tiles[0].global_ref_id(0), 0);
    assert_eq!(plan.tiles[0].global_ref_id(1), 1);
    assert_eq!(plan.tiles[1].global_ref_id(0), 2);
    assert_eq!(plan.tiles[1].global_ref_id(1), 3);
}

#[test]
fn sub_reference_clones_just_the_tile() {
    let parent = ref_with(&[10, 20, 30, 40]);
    let plan = plan_tiles(&parent, 50);
    assert_eq!(plan.n_tiles(), 3);
    let sub = plan.tiles[0].build_sub_reference(&parent);
    assert_eq!(sub.sequences.len(), 2);
    assert_eq!(sub.sequences[0].name, "c0");
    assert_eq!(sub.sequences[1].name, "c1");
}
