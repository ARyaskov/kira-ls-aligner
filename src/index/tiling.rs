//! Reference tiling for `--split-prefix` mode.

use crate::types::Reference;

/// A single tile: a contiguous slice of contigs from the parent reference.
#[derive(Clone, Copy, Debug)]
pub struct Tile {
    /// First contig index (inclusive) into `parent.sequences`.
    pub contig_start: usize,
    /// Last contig index (exclusive) into `parent.sequences`.
    pub contig_end: usize,
    /// Equal to `contig_start` — the offset we add to a local `ref_id`.
    pub global_ref_id_offset: u32,
    /// Sum of contig lengths in this tile, in bases.
    pub total_bytes: u64,
}

impl Tile {
    /// Build a sub-`Reference` containing just this tile's contigs.
    pub fn build_sub_reference(&self, parent: &Reference) -> Reference {
        Reference {
            sequences: parent.sequences[self.contig_start..self.contig_end].to_vec(),
        }
    }

    /// Translate a local `ref_id` (0..tile_contigs.len()) to a global `ref_id`.
    #[inline]
    pub fn global_ref_id(&self, local_ref_id: u32) -> u32 {
        self.global_ref_id_offset + local_ref_id
    }
}

/// Full tile plan: ordered list of tiles covering every contig in the parent reference exactly once.
#[derive(Clone, Debug)]
pub struct TilePlan {
    pub tiles: Vec<Tile>,
}

impl TilePlan {
    /// Total number of tiles. Equal to 1 for refs that fit in `target_bytes`.
    pub fn n_tiles(&self) -> usize {
        self.tiles.len()
    }

    /// Returns true when only one tile is needed — caller can fast-path the legacy single-pass.
    pub fn is_trivial(&self) -> bool {
        self.tiles.len() <= 1
    }
}

/// Greedy bin-pack reference contigs into tiles of size ≤ `target_bytes`.
pub fn plan_tiles(reference: &Reference, target_bytes: u64) -> TilePlan {
    let mut tiles: Vec<Tile> = Vec::new();
    if reference.sequences.is_empty() {
        return TilePlan { tiles };
    }
    let target = target_bytes.max(1);

    let mut start = 0usize;
    let mut acc_bytes: u64 = 0;

    for i in 0..reference.sequences.len() {
        let contig_bytes = reference.sequences[i].len(None) as u64;

        if acc_bytes > 0 && acc_bytes + contig_bytes > target {
            tiles.push(Tile {
                contig_start: start,
                contig_end: i,
                global_ref_id_offset: start as u32,
                total_bytes: acc_bytes,
            });
            start = i;
            acc_bytes = 0;
        }

        if contig_bytes > target && acc_bytes == 0 {
            eprintln!(
                "[KIRA_TILE] warning: contig {:?} ({} bp) exceeds --tile-bytes ({}) — \
                 making it a singleton tile",
                reference.sequences[i].name, contig_bytes, target
            );
            tiles.push(Tile {
                contig_start: i,
                contig_end: i + 1,
                global_ref_id_offset: i as u32,
                total_bytes: contig_bytes,
            });
            start = i + 1;
            acc_bytes = 0;
            continue;
        }

        acc_bytes += contig_bytes;
    }

    // Flush the open tile.
    if start < reference.sequences.len() {
        tiles.push(Tile {
            contig_start: start,
            contig_end: reference.sequences.len(),
            global_ref_id_offset: start as u32,
            total_bytes: acc_bytes,
        });
    }

    TilePlan { tiles }
}

#[cfg(test)]
mod tests {
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
}
