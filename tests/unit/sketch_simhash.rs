use super::*;

#[test]
fn identical_windows_hash_equal() {
    let w = b"ACGTACGTACGTACGTACGTACGTACGTACGT";
    let a = simhash_window(w).unwrap();
    let b = simhash_window(w).unwrap();
    assert_eq!(a, b);
}

#[test]
fn distinct_windows_hash_differ() {
    let a = simhash_window(b"ACGTACGTACGTACGTACGTACGTACGTACGT").unwrap();
    let b = simhash_window(b"TGCATGCATGCATGCATGCATGCATGCATGCA").unwrap();
    // Sanity: completely-different windows should disagree in many bits.
    assert!(
        hamming_distance(a, b) >= 20,
        "expected wildly different hashes, got Hamming distance {}",
        hamming_distance(a, b)
    );
}

#[test]
fn single_base_flip_stays_close() {
    let base = b"ACGTACGTACGTACGTACGTACGTACGTACGT";
    let mut mutated = base.to_vec();
    mutated[5] = b'G'; // was 'C'
    let h0 = simhash_window(base).unwrap();
    let h1 = simhash_window(&mutated).unwrap();
    let d = hamming_distance(h0, h1);
    // One base flip = one feature swap. Bits flip wherever the two
    // features (table[5*4+1] vs table[5*4+2]) disagree AND the swap
    // tips the sign of `sums[bit]`. Empirically d ≤ 32; we use a
    // loose upper bound of 24 here as a regression guard.
    assert!(
        d < 24,
        "single-base flip moved hash by {d} bits — should be ≤ 24"
    );
}

#[test]
fn two_base_flips_stay_close() {
    let base = b"ACGTACGTACGTACGTACGTACGTACGTACGT";
    let mut mutated = base.to_vec();
    mutated[5] = b'G';
    mutated[20] = b'A';
    let h0 = simhash_window(base).unwrap();
    let h1 = simhash_window(&mutated).unwrap();
    let d = hamming_distance(h0, h1);
    assert!(d < 32, "two-base flip should leave most bits intact, got {d}");
}

#[test]
fn rejects_n_base() {
    let w = b"ACGTACGTACGTACGTACGTACGTACGTANGT"; // N at index 29
    assert!(simhash_window(w).is_none());
}

#[test]
fn rejects_empty_and_oversize() {
    assert!(simhash_window(b"").is_none());
    let too_long = vec![b'A'; MAX_WINDOW_LEN + 1];
    assert!(simhash_window(&too_long).is_none());
}

#[test]
fn random_pair_distance_is_around_half() {
    // For two independent random inputs, the SimHash Hamming distance
    // should hover around 32 (half of 64). Average over a few seeds.
    let mut state: u64 = 0x243F_6A88_85A3_08D3;
    let mut total: u64 = 0;
    const N: usize = 64;
    for _ in 0..N {
        let mut a = Vec::with_capacity(32);
        let mut b = Vec::with_capacity(32);
        for _ in 0..32 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            a.push(b"ACGT"[(state >> 56) as usize & 3]);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            b.push(b"ACGT"[(state >> 56) as usize & 3]);
        }
        let ha = simhash_window(&a).unwrap();
        let hb = simhash_window(&b).unwrap();
        total += hamming_distance(ha, hb) as u64;
    }
    let avg = total as f64 / N as f64;
    // Allow a generous window — we only want to catch a kernel that
    // collapsed to a constant or is wildly biased.
    assert!(
        (20.0..=44.0).contains(&avg),
        "average random-pair Hamming distance {avg} outside [20, 44]"
    );
}
