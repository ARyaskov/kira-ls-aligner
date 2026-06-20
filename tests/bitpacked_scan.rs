//! Integration tests for `alignment::bitpacked::scan` — 2-bit-packed
//! first-acceptable Hamming scan. Tests verify (a) returned distance
//! matches actual Hamming distance at the reported shift, and (b) the
//! threshold semantics: scan succeeds at max=naive_optimum, fails at
//! max=naive_optimum-1.
//!
//! The `shift_bits_right` private helper has its own inline test in
//! src/alignment/bitpacked.rs — it's an implementation detail, not part
//! of the public API.

use kira_ls_aligner::alignment::bitpacked::{PackedDna, scan};

fn naive_mismatches(read: &[u8], ref_window: &[u8], shift: usize) -> usize {
    read.iter()
        .zip(ref_window[shift..].iter())
        .take(read.len())
        .filter(|(a, b)| a != b)
        .count()
}

fn naive_best_shift(read: &[u8], ref_window: &[u8]) -> (usize, usize) {
    let r = read.len();
    let n_shifts = ref_window.len() - r + 1;
    let mut best = r + 1;
    let mut best_t = 0;
    for t in 0..n_shifts {
        let m = naive_mismatches(read, ref_window, t);
        if m < best {
            best = m;
            best_t = t;
        }
    }
    (best_t, best)
}

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[test]
fn pack_roundtrip_short() {
    // ACGT in low-first 2-bit: A=00, C=01, G=10, T=11 → byte = 0b11100100 = 0xE4
    let p = PackedDna::pack(b"ACGTACGT");
    assert_eq!(p.bits.len(), 2);
    assert_eq!(p.bits[0], 0xE4);
    assert_eq!(p.bits[1], 0xE4);
}

#[test]
fn pack_handles_padding() {
    // 5 nucleotides → 2 bytes, last byte has 3 pad nulls (zero = A code).
    let p = PackedDna::pack(b"ACGTA");
    assert_eq!(p.bits.len(), 2);
    assert_eq!(p.bits[1] & 0b11, 0);
    assert_eq!(p.bits[1] >> 2, 0);
}

#[test]
fn scan_exact_match_at_zero() {
    let read = b"ACGTACGTACGT";
    let p_read = PackedDna::pack(read);
    let shifted = PackedDna::pack(read).pre_shifted_window();
    let hit = scan(&p_read, &shifted, read.len(), 0).unwrap();
    assert_eq!(hit.shift, 0);
    assert_eq!(hit.mismatches, 0);
}

#[test]
fn scan_exact_match_in_middle() {
    let read = b"ACGTACGT";
    let text = b"AAAAACGTACGTAAAA";
    let p_read = PackedDna::pack(read);
    let shifted = PackedDna::pack(text).pre_shifted_window();
    let hit = scan(&p_read, &shifted, text.len(), 0).unwrap();
    assert_eq!(hit.shift, 4);
    assert_eq!(hit.mismatches, 0);
}

#[test]
fn scan_rejects_when_too_many_mismatches() {
    let read = b"AAAAAAAA";
    let text = b"TTTTTTTTTTTT";
    let p_read = PackedDna::pack(read);
    let shifted = PackedDna::pack(text).pre_shifted_window();
    assert!(scan(&p_read, &shifted, text.len(), 3).is_none());
    let hit = scan(&p_read, &shifted, text.len(), 8).unwrap();
    assert_eq!(hit.mismatches, 8);
}

#[test]
fn scan_long_read_avx_path() {
    let read: Vec<u8> = (0..150).map(|i| b"ACGT"[i % 4]).collect();
    let mut text: Vec<u8> = (0..200).map(|i| b"ACGT"[(i + 1) % 4]).collect();
    text[25..25 + 150].copy_from_slice(&read);
    text[25 + 7] = if read[7] == b'A' { b'T' } else { b'A' };
    text[25 + 100] = if read[100] == b'C' { b'G' } else { b'C' };

    let p_read = PackedDna::pack(&read);
    let shifted = PackedDna::pack(&text).pre_shifted_window();

    let max_mism = 5;
    let hit = scan(&p_read, &shifted, text.len(), max_mism).unwrap();
    assert_eq!(naive_mismatches(&read, &text, hit.shift), hit.mismatches);
    assert!(hit.mismatches <= max_mism);
}

#[test]
fn random_corpus_self_consistent() {
    // Property: every returned hit must (a) have mismatches ≤ threshold,
    // (b) report a mismatch count matching the actual Hamming distance at
    // the reported shift. We don't assert strict optimum because scan
    // exits on first-acceptable for performance.
    let bases = b"ACGT";
    let mut rng = 0xC0FFEEBADu64;
    for _ in 0..200 {
        let r = 16 + (xorshift(&mut rng) as usize % 180);
        let w = r + (xorshift(&mut rng) as usize % 100);
        let read: Vec<u8> = (0..r)
            .map(|_| bases[(xorshift(&mut rng) as usize) % 4])
            .collect();
        let mut text: Vec<u8> = (0..w)
            .map(|_| bases[(xorshift(&mut rng) as usize) % 4])
            .collect();
        if xorshift(&mut rng) % 2 == 0 && w >= r {
            let start = (xorshift(&mut rng) as usize) % (w - r + 1);
            text[start..start + r].copy_from_slice(&read);
        }

        let (_naive_t, naive_m) = naive_best_shift(&read, &text);
        let p_read = PackedDna::pack(&read);
        let shifted = PackedDna::pack(&text).pre_shifted_window();

        let hit = scan(&p_read, &shifted, text.len(), r).unwrap();
        assert_eq!(
            naive_mismatches(&read, &text, hit.shift),
            hit.mismatches,
            "self-inconsistent hit at r={r} w={w}",
        );

        let tight = scan(&p_read, &shifted, text.len(), naive_m).unwrap();
        assert!(tight.mismatches <= naive_m);
        if naive_m > 0 {
            assert!(scan(&p_read, &shifted, text.len(), naive_m - 1).is_none());
        }
    }
}
