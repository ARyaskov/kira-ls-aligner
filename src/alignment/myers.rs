//! Myers (1999) bit-parallel edit distance, multi-block.

/// Minimum width of the block buffer when computing Peq.
const BLOCK_BITS: usize = 64;

/// Map a DNA base to a 0..4 index.
#[inline]
fn base_to_idx(b: u8) -> Option<usize> {
    match b {
        b'A' | b'a' => Some(0),
        b'C' | b'c' => Some(1),
        b'G' | b'g' => Some(2),
        b'T' | b't' => Some(3),
        _ => None,
    }
}

/// Reusable per-thread state for [`bounded_edit_distance`]; the buffers are
/// resized in place and fully overwritten before use.
#[derive(Default)]
struct MyersScratch {
    peq: [Vec<u64>; 4],
    vp: Vec<u64>,
    vn: Vec<u64>,
}

thread_local! {
    static MYERS_SCRATCH: std::cell::RefCell<MyersScratch> =
        std::cell::RefCell::new(MyersScratch::default());
}

/// Per-character pattern bitmasks, written into `peq` (resized as needed).
fn build_peq_into(pattern: &[u8], peq: &mut [Vec<u64>; 4]) {
    let blocks = pattern.len().div_ceil(BLOCK_BITS);
    for lane in peq.iter_mut() {
        lane.clear();
        lane.resize(blocks, 0u64);
    }
    for (i, &b) in pattern.iter().enumerate() {
        if let Some(c) = base_to_idx(b) {
            peq[c][i / BLOCK_BITS] |= 1u64 << (i % BLOCK_BITS);
        }
    }
}

/// Compute the (semi-global) bounded edit distance between `pattern` and the best matching.
pub fn bounded_edit_distance(pattern: &[u8], text: &[u8], max_k: usize) -> Option<(usize, usize)> {
    let m = pattern.len();
    let n = text.len();
    if m == 0 {
        // Empty pattern always matches at position 0 with zero edits.
        let _ = max_k;
        return Some((0, 0));
    }
    if n == 0 {
        // Empty text: distance is m (delete every pattern char).
        return if m <= max_k { Some((m, 0)) } else { None };
    }

    // Short reads run entirely on stack arrays, which keeps per-block bounds
    // checks out of a loop that runs `text_len * blocks` times.
    if pattern.len() <= FIXED_BLOCKS * BLOCK_BITS {
        return bounded_edit_distance_fixed(pattern, text, max_k);
    }
    MYERS_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        let MyersScratch { peq, vp, vn } = &mut *scratch;
        bounded_edit_distance_inner(pattern, text, max_k, peq, vp, vn)
    })
}

/// Pattern blocks handled by the fixed-array path (4 x 64 = 256 bases).
const FIXED_BLOCKS: usize = 4;

/// Test-only: force the heap path regardless of pattern length, so the two
/// storage implementations can be differentially compared on identical input.
#[cfg(test)]
pub fn bounded_edit_distance_heap_for_test(
    pattern: &[u8],
    text: &[u8],
    max_k: usize,
) -> Option<(usize, usize)> {
    if pattern.is_empty() {
        return Some((0, 0));
    }
    if text.is_empty() {
        return if pattern.len() <= max_k {
            Some((pattern.len(), 0))
        } else {
            None
        };
    }
    MYERS_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        let MyersScratch { peq, vp, vn } = &mut *scratch;
        bounded_edit_distance_inner(pattern, text, max_k, peq, vp, vn)
    })
}
/// Fixed-capacity variant of [`bounded_edit_distance_inner`]: identical
/// recurrence and result, stack arrays instead of heap buffers.
fn bounded_edit_distance_fixed(
    pattern: &[u8],
    text: &[u8],
    max_k: usize,
) -> Option<(usize, usize)> {
    let m = pattern.len();
    let n = text.len();
    let blocks = m.div_ceil(BLOCK_BITS);
    debug_assert!(blocks <= FIXED_BLOCKS);

    let mut peq = [[0u64; FIXED_BLOCKS]; 4];
    for (i, &b) in pattern.iter().enumerate() {
        if let Some(c) = base_to_idx(b) {
            peq[c][i / BLOCK_BITS] |= 1u64 << (i % BLOCK_BITS);
        }
    }

    let top_used = m - (blocks - 1) * BLOCK_BITS;
    let top_block_mask: u64 = if top_used == BLOCK_BITS {
        !0u64
    } else {
        (1u64 << top_used) - 1
    };
    let high_bit_in_top: u64 = 1u64 << (top_used - 1);

    let mut vp = [0u64; FIXED_BLOCKS];
    let mut vn = [0u64; FIXED_BLOCKS];
    vp[..blocks].fill(!0u64);
    vp[blocks - 1] &= top_block_mask;

    let mut err = m;
    let mut best_score = m;
    let mut best_pos = 0usize;
    // Empty lane for text characters outside ACGT, so they match nothing.
    let zero_lane = [0u64; FIXED_BLOCKS];

    for (j, &c) in text.iter().enumerate() {
        let peq_c: &[u64; FIXED_BLOCKS] = match base_to_idx(c) {
            Some(ci) => &peq[ci],
            None => &zero_lane,
        };

        let mut hin_p: u64 = 0;
        let mut hin_n: u64 = 0;
        let mut add_carry: bool = false;

        for b in 0..blocks {
            let eq = peq_c[b];
            let vp_b = vp[b];
            let vn_b = vn[b];

            let x = eq | vn_b;
            // (eq & vp) + vp + carry, carried across blocks.
            let av = eq & vp_b;
            let (s1, c1) = av.overflowing_add(vp_b);
            let (sum_lo, c2) = s1.overflowing_add(u64::from(add_carry));
            add_carry = c1 | c2;

            let d0 = (sum_lo ^ vp_b) | x;
            let hp = vn_b | !(d0 | vp_b);
            let hn = d0 & vp_b;

            let hp_shift = (hp << 1) | hin_p;
            let hn_shift = (hn << 1) | hin_n;
            hin_p = hp >> 63;
            hin_n = hn >> 63;

            let new_vp = hn_shift | !(d0 | hp_shift);
            let new_vn = d0 & hp_shift;

            let mask = if b == blocks - 1 { top_block_mask } else { !0u64 };
            vp[b] = new_vp & mask;
            vn[b] = new_vn & mask;

            if b == blocks - 1 {
                if hp & high_bit_in_top != 0 {
                    err += 1;
                }
                if hn & high_bit_in_top != 0 {
                    err -= 1;
                }
            }
        }

        if err <= best_score {
            best_score = err;
            best_pos = j + 1;
        }
        // Exact early reject — see the comment in `bounded_edit_distance_inner`.
        if best_score > max_k && err.saturating_sub(n - 1 - j) > max_k {
            return None;
        }
    }

    if best_score <= max_k {
        Some((best_score, best_pos))
    } else {
        None
    }
}

fn bounded_edit_distance_inner(
    pattern: &[u8],
    text: &[u8],
    max_k: usize,
    peq: &mut [Vec<u64>; 4],
    vp: &mut Vec<u64>,
    vn: &mut Vec<u64>,
) -> Option<(usize, usize)> {
    let m = pattern.len();
    let n = text.len();

    build_peq_into(pattern, peq);
    let blocks = peq[0].len();

    let top_used = m - (blocks - 1) * BLOCK_BITS;
    let top_block_mask: u64 = if top_used == BLOCK_BITS {
        !0u64
    } else {
        (1u64 << top_used) - 1
    };

    // High bit of the top block — used to decide ±1 error after each text column.
    let high_bit_in_top: u64 = 1u64 << (top_used - 1);

    vp.clear();
    vp.resize(blocks, !0u64);
    vp[blocks - 1] &= top_block_mask;
    vn.clear();
    vn.resize(blocks, 0u64);

    // Current edit distance against the full pattern at the latest text column.
    let mut err = m;
    let mut best_score = m;
    let mut best_pos = 0usize;

    // Iterate over text characters.
    for (j, &c) in text.iter().enumerate() {
        let peq_c: &[u64] = if let Some(ci) = base_to_idx(c) {
            &peq[ci]
        } else {
            &peq[0]
        };
        let n_in_text = base_to_idx(c).is_none();

        let mut hin_p: u64 = 0;
        let mut hin_n: u64 = 0;

        // Carry for the (eq & vp) + vp wide addition across blocks.
        let mut add_carry: u128 = 0;

        for b in 0..blocks {
            let eq = if n_in_text { 0 } else { peq_c[b] };
            let vp_b = vp[b];
            let vn_b = vn[b];

            // X = eq | vn
            let x = eq | vn_b;
            // sum = (eq & vp) + vp + carry  (use u128 to capture carry-out)
            let av = eq & vp_b;
            let sum128 = (av as u128) + (vp_b as u128) + add_carry;
            let sum_lo = sum128 as u64;
            add_carry = sum128 >> 64; // 0 or 1

            // D0 = (sum ^ vp) | x
            let d0 = (sum_lo ^ vp_b) | x;
            // HP = vn | !(d0 | vp)
            let hp = vn_b | !(d0 | vp_b);
            // HN = d0 & vp
            let hn = d0 & vp_b;

            let hp_shift = (hp << 1) | hin_p;
            let hn_shift = (hn << 1) | hin_n;
            let new_hin_p = hp >> 63;
            let new_hin_n = hn >> 63;

            let new_vp = hn_shift | !(d0 | hp_shift);
            let new_vn = d0 & hp_shift;

            // Mask top block to ignore out-of-range bits.
            let mask = if b == blocks - 1 {
                top_block_mask
            } else {
                !0u64
            };
            vp[b] = new_vp & mask;
            vn[b] = new_vn & mask;

            if b == blocks - 1 {
                if hp & high_bit_in_top != 0 {
                    err += 1;
                }
                if hn & high_bit_in_top != 0 {
                    err -= 1;
                }
            }

            hin_p = new_hin_p;
            hin_n = new_hin_n;
        }

        if err <= best_score {
            best_score = err;
            best_pos = j + 1;
        }

        // Exact early reject: `err` moves by at most 1 per remaining column, so
        // once neither it nor the running best can reach `max_k`, the answer is
        // already `None`.
        if best_score > max_k && err.saturating_sub(n - 1 - j) > max_k {
            return None;
        }
    }

    if best_score <= max_k {
        Some((best_score, best_pos))
    } else {
        None
    }
}

#[cfg(test)]
mod storage_equivalence_tests {
    use super::*;

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// The fixed-array and heap implementations must agree exactly — same
    /// distance *and* same end position — for every input the pipeline can hand
    /// them. Two-tier locus ranking consumes the distance, so any divergence
    /// silently reorders candidate loci.
    #[test]
    fn fixed_and_heap_paths_agree_exactly() {
        let bases = b"ACGTN";
        let mut rng = 0xFACE_1357_2468_0BADu64;
        for _ in 0..4000 {
            let m = 1 + (xorshift(&mut rng) as usize % 256);
            let n = 1 + (xorshift(&mut rng) as usize % 300);
            let pattern: Vec<u8> = (0..m)
                .map(|_| bases[(xorshift(&mut rng) as usize) % 5])
                .collect();
            let text: Vec<u8> = (0..n)
                .map(|_| bases[(xorshift(&mut rng) as usize) % 5])
                .collect();
            for max_k in [0usize, 1, 3, m / 8, m / 4, m / 2, m] {
                let fixed = bounded_edit_distance_fixed(&pattern, &text, max_k);
                let heap = bounded_edit_distance_heap_for_test(&pattern, &text, max_k);
                assert_eq!(
                    fixed, heap,
                    "m={m} n={n} max_k={max_k}\npattern={:?}\ntext={:?}",
                    String::from_utf8_lossy(&pattern),
                    String::from_utf8_lossy(&text)
                );
            }
        }
    }
}
