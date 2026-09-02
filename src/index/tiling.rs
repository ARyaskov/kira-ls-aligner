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
            crate::kira_warn!("[KIRA_TILE] warning: contig {:?} ({} bp) exceeds --tile-bytes ({}) — \
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
#[path = "../../tests/unit/index_tiling.rs"]
mod tests;
