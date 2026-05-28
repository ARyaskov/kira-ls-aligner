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

/// Per-character pattern bitmasks.
fn build_peq(pattern: &[u8]) -> [Vec<u64>; 4] {
    let m = pattern.len();
    let blocks = m.div_ceil(BLOCK_BITS);
    let mut peq: [Vec<u64>; 4] = [
        vec![0u64; blocks],
        vec![0u64; blocks],
        vec![0u64; blocks],
        vec![0u64; blocks],
    ];
    for (i, &b) in pattern.iter().enumerate() {
        if let Some(c) = base_to_idx(b) {
            peq[c][i / BLOCK_BITS] |= 1u64 << (i % BLOCK_BITS);
        }
    }
    peq
}

/// Compute the (semi-global) bounded edit distance between `pattern` and the best matching.
pub fn bounded_edit_distance(
    pattern: &[u8],
    text: &[u8],
    max_k: usize,
) -> Option<(usize, usize)> {
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

    let peq = build_peq(pattern);
    let blocks = peq[0].len();

    let top_used = m - (blocks - 1) * BLOCK_BITS;
    let top_block_mask: u64 = if top_used == BLOCK_BITS {
        !0u64
    } else {
        (1u64 << top_used) - 1
    };

    // High bit of the top block — used to decide ±1 error after each text column.
    let high_bit_in_top: u64 = 1u64 << (top_used - 1);

    let mut vp: Vec<u64> = vec![!0u64; blocks];
    vp[blocks - 1] &= top_block_mask;
    let mut vn: Vec<u64> = vec![0u64; blocks];

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
    }

    if best_score <= max_k {
        Some((best_score, best_pos))
    } else {
        None
    }
}
