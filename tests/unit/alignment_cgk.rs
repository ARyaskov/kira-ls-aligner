use super::*;

fn rng_byte(state: &mut u64) -> u8 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^= z >> 31;
    let codes = b"ACGT";
    codes[(z & 3) as usize]
}

fn random_dna(len: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    (0..len).map(|_| rng_byte(&mut s)).collect()
}

fn hamming_bits(a: &[u64], b: &[u64], n_bits: usize) -> usize {
    let n_full = n_bits / 64;
    let tail_bits = n_bits & 63;
    let mut h = 0usize;
    for i in 0..n_full {
        h += (a[i] ^ b[i]).count_ones() as usize;
    }
    if tail_bits > 0 {
        let mask = (1u64 << tail_bits) - 1;
        h += ((a[n_full] ^ b[n_full]) & mask).count_ones() as usize;
    }
    h
}

#[test]
fn advice_is_deterministic() {
    let a = EmbeddingAdvice::new(42, 900);
    let b = EmbeddingAdvice::new(42, 900);
    for i in 0..900 {
        assert_eq!(a.get(i), b.get(i), "mismatch at bit {}", i);
    }
}

#[test]
fn advice_bit_distribution_is_balanced() {
    let a = EmbeddingAdvice::new(0xCAFE, 900);
    let ones = (0..900).filter(|&i| a.get(i) != 0).count();
    assert!(
        ones >= 900 * 35 / 100 && ones <= 900 * 65 / 100,
        "advice 1-bit count {} outside [35%, 65%]",
        ones
    );
}

#[test]
fn embed_is_deterministic() {
    let advice = EmbeddingAdvice::new(7, 900);
    let input = random_dna(150, 1);
    let n_words = (6 * 150usize).div_ceil(64);
    let mut out_a = vec![0u64; n_words];
    let mut out_b = vec![0u64; n_words];
    embed(&input, &advice, &mut out_a);
    embed(&input, &advice, &mut out_b);
    assert_eq!(out_a, out_b);
}

#[test]
fn embed_preserves_first_two_bits() {
    let advice = EmbeddingAdvice::new(99, 900);
    let input = random_dna(150, 2);
    let n_words = (6 * 150usize).div_ceil(64);
    let mut out = vec![0u64; n_words];
    embed(&input, &advice, &mut out);
    let first_input_bit = read_input_bit(&input, 0);
    let first_output_bit = read_packed_bit(&out, 0, 900);
    assert_eq!(
        first_output_bit, first_input_bit,
        "first output bit ({}) != first input bit ({})",
        first_output_bit, first_input_bit
    );
}

#[test]
fn identical_inputs_produce_identical_embeddings() {
    let advice = EmbeddingAdvice::new(11, 900);
    let input = random_dna(150, 3);
    let n_words = (6 * 150usize).div_ceil(64);
    let mut a = vec![0u64; n_words];
    let mut b = vec![0u64; n_words];
    embed(&input, &advice, &mut a);
    embed(&input, &advice, &mut b);
    assert_eq!(hamming_bits(&a, &b, 900), 0);
}

fn apply_one_insertion(s: &[u8], pos: usize, base: u8) -> Vec<u8> {
    let mut v = Vec::with_capacity(s.len());
    v.extend_from_slice(&s[..pos]);
    v.push(base);
    v.extend_from_slice(&s[pos..s.len() - 1]);
    v
}

fn apply_one_deletion(s: &[u8], pos: usize, pad: u8) -> Vec<u8> {
    let mut v = Vec::with_capacity(s.len());
    v.extend_from_slice(&s[..pos]);
    v.extend_from_slice(&s[pos + 1..]);
    v.push(pad);
    v
}

fn apply_one_substitution(s: &[u8], pos: usize, base: u8) -> Vec<u8> {
    let mut v = s.to_vec();
    v[pos] = base;
    v
}

#[test]
fn edit_distance_one_substitution_keeps_hamming_bounded() {
    let mut max_h = 0usize;
    for seed in [2024u64, 31, 99, 17, 123456] {
        let advice = EmbeddingAdvice::new(seed, 900);
        let x = random_dna(150, seed.wrapping_mul(13));
        let new_base = if x[70] == b'G' { b'C' } else { b'G' };
        let y = apply_one_substitution(&x, 70, new_base);
        let n_words = (6 * 150usize).div_ceil(64);
        let mut ex = vec![0u64; n_words];
        let mut ey = vec![0u64; n_words];
        embed(&x, &advice, &mut ex);
        embed(&y, &advice, &mut ey);
        let h = hamming_bits(&ex, &ey, 900);
        assert!(h > 0, "1-substitution produced 0 embedded Hamming");
        if h > max_h {
            max_h = h;
        }
    }
    assert!(
        max_h <= 120,
        "max embedded Hamming for 1-substitution across seeds = {}, expected ≤ 120",
        max_h
    );
}

#[test]
fn edit_distance_one_insertion_gives_small_hamming() {
    let mut max_h = 0usize;
    for seed in [2024u64, 31, 99, 17, 123456] {
        let advice = EmbeddingAdvice::new(seed, 900);
        let x = random_dna(150, seed.wrapping_mul(7));
        let y = apply_one_insertion(&x, 73, b'G');
        let n_words = (6 * 150usize).div_ceil(64);
        let mut ex = vec![0u64; n_words];
        let mut ey = vec![0u64; n_words];
        embed(&x, &advice, &mut ex);
        embed(&y, &advice, &mut ey);
        let h = hamming_bits(&ex, &ey, 900);
        if h > max_h {
            max_h = h;
        }
    }
    assert!(
        max_h <= 80,
        "max embedded Hamming for 1-insertion across seeds = {}, expected ≤ 80",
        max_h
    );
}

#[test]
fn edit_distance_one_deletion_gives_small_hamming() {
    let mut max_h = 0usize;
    for seed in [2024u64, 31, 99, 17, 123456] {
        let advice = EmbeddingAdvice::new(seed, 900);
        let x = random_dna(150, seed.wrapping_mul(7));
        let y = apply_one_deletion(&x, 60, b'A');
        let n_words = (6 * 150usize).div_ceil(64);
        let mut ex = vec![0u64; n_words];
        let mut ey = vec![0u64; n_words];
        embed(&x, &advice, &mut ex);
        embed(&y, &advice, &mut ey);
        let h = hamming_bits(&ex, &ey, 900);
        if h > max_h {
            max_h = h;
        }
    }
    assert!(
        max_h <= 80,
        "max embedded Hamming for 1-deletion across seeds = {}, expected ≤ 80",
        max_h
    );
}

#[test]
fn fingerprint_is_deterministic() {
    let scheme = FingerprintScheme::new(31415, 150, 16);
    let input = random_dna(150, 5);
    let a = scheme.fingerprints(&input);
    let b = scheme.fingerprints(&input);
    assert_eq!(a, b);
    assert_eq!(a.len(), 16);
}

#[test]
fn fingerprint_changes_on_substitution_at_probe_pos() {
    let scheme = FingerprintScheme::new(271828, 150, 16);
    let x = random_dna(150, 17);
    let base_fps = scheme.fingerprints(&x);
    let mut any_change = false;
    for pos in 0..150 {
        let new_base = if x[pos] == b'A' { b'T' } else { b'A' };
        let y = apply_one_substitution(&x, pos, new_base);
        let new_fps = scheme.fingerprints(&y);
        if new_fps != base_fps {
            any_change = true;
            break;
        }
    }
    assert!(
        any_change,
        "no single substitution changed any fingerprint — \
        probe positions cover none of the input?"
    );
}

#[test]
fn index_recalls_inserted_window() {
    let scheme = FingerprintScheme::new(2024, 150, 16);
    let ref_bases = random_dna(10_000, 42);
    let idx = CgkIndex::build_from_sequences(
        scheme,
        30,
        std::iter::once((0u32, ref_bases.as_slice())),
    );
    let win = &ref_bases[300..450];
    let candidates = idx.query(win, 1);
    assert!(
        candidates.iter().any(|&(rid, pos)| rid == 0 && pos == 300),
        "self-query did not recall its own window; candidates = {:?}",
        candidates
    );
}

#[test]
fn index_recalls_indel_variant_window() {
    let scheme = FingerprintScheme::new(2024, 150, 16);
    let ref_bases = random_dna(20_000, 99);
    let idx = CgkIndex::build_from_sequences(
        scheme,
        30,
        std::iter::once((0u32, ref_bases.as_slice())),
    );
    let mut hits = 0;
    let trials = [(600usize, 75usize), (900, 30), (1200, 100), (1500, 60), (1800, 90)];
    for &(true_pos, edit_pos) in &trials {
        let win = &ref_bases[true_pos..true_pos + 150];
        let edited = apply_one_insertion(win, edit_pos, b'C');
        let candidates = idx.query(&edited, 1);
        if candidates
            .iter()
            .any(|&(rid, pos)| rid == 0 && pos == true_pos as u32)
        {
            hits += 1;
        }
    }
    assert!(
        hits >= 4,
        "indel-edited recall: {} / {} trials hit; expected ≥ 4",
        hits,
        trials.len()
    );
}

#[test]
fn pack_unpack_roundtrips() {
    for &(rid, pos) in &[(0u32, 0u32), (1, 100), (12345, 67890), (u32::MAX, u32::MAX)] {
        let (r, p) = unpack_window(pack_window(rid, pos));
        assert_eq!((r, p), (rid, pos));
    }
}
