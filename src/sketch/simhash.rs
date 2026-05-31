//! SimHash (Charikar 2002) for fixed-length DNA windows.
//!
//! Maps a DNA window to a 64-bit fingerprint such that windows with small
//! Hamming distance in base-space have small Hamming distance in
//! fingerprint-space. Feeds [`crate::index::lsh`], which buckets fingerprints
//! by their top bits to seed reads with 1-2 mismatches that the exact-match
//! minimizer index would otherwise miss.
//!
//! The mapping is `(base, position) → random 64-bit projection`. Each of the
//! 64 output bits accumulates a signed sum over the projection bits of every
//! `(base, position)` pair in the window; the final bit is the sign of that
//! sum. Sign-projection is the property that makes the output distance
//! correlate with input cosine distance (Charikar 2002, Thm 1).

use std::sync::OnceLock;

/// Default window length in bases.
pub const DEFAULT_WINDOW_LEN: usize = 32;

/// Maximum supported window length — bounds the precomputed projection table.
pub const MAX_WINDOW_LEN: usize = 64;

/// Pseudo-random `(position, base) → u64` projection table.
///
/// Indexed as `position * 4 + base_code`. Built once on first use via
/// SplitMix64 from a fixed seed so the table is deterministic across runs
/// and across the build / query sides of the index.
fn feature_table() -> &'static [u64] {
    static TABLE: OnceLock<Vec<u64>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = Vec::with_capacity(MAX_WINDOW_LEN * 4);
        let mut state: u64 = 0xC0FFEE_DEADBEEF;
        for _ in 0..(MAX_WINDOW_LEN * 4) {
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            table.push(z ^ (z >> 31));
        }
        table
    })
}

#[inline]
fn base_code(b: u8) -> Option<u8> {
    match b {
        b'A' | b'a' => Some(0),
        b'C' | b'c' => Some(1),
        b'G' | b'g' => Some(2),
        b'T' | b't' => Some(3),
        _ => None,
    }
}

/// 64-bit SimHash of a DNA window.
///
/// Returns `None` if the window length exceeds [`MAX_WINDOW_LEN`] or
/// contains any non-ACGT base (N). Non-ACGT bases are rejected rather than
/// mapped to a fixed code because the latter would silently degrade the
/// distance-preservation property for windows that span low-complexity
/// regions.
pub fn simhash_window(window: &[u8]) -> Option<u64> {
    if window.is_empty() || window.len() > MAX_WINDOW_LEN {
        return None;
    }
    let table = feature_table();
    let mut sums = [0i16; 64];
    for (i, &b) in window.iter().enumerate() {
        let code = base_code(b)?;
        let feature = table[i * 4 + code as usize];
        // Each bit position: +1 if the projection bit is 1, -1 otherwise.
        for bit in 0..64 {
            sums[bit] += if (feature >> bit) & 1 == 1 { 1 } else { -1 };
        }
    }
    let mut hash = 0u64;
    for (bit, &s) in sums.iter().enumerate() {
        if s > 0 {
            hash |= 1u64 << bit;
        }
    }
    Some(hash)
}

/// Hamming distance between two 64-bit fingerprints. Uses native popcount,
/// which compiles to `POPCNT` on x86_64 and `CNT` on aarch64.
#[inline]
pub fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

#[cfg(test)]
#[path = "../../tests/unit/sketch_simhash.rs"]
mod tests;
