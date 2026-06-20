//! Integration tests for `alignment::spectral::scan` — multi-shift Hamming
//! scan with early-pruning. Cross-checked against a brute-force naive
//! reference for both optimum shift and reported distance.

use kira_ls_aligner::alignment::spectral::{SpectralHit, scan};

fn naive(read: &[u8], ref_window: &[u8]) -> SpectralHit {
    let r = read.len();
    let n_shifts = ref_window.len() - r + 1;
    let mut best_matches = 0usize;
    let mut best_shift = 0usize;
    for t in 0..n_shifts {
        let matches = read
            .iter()
            .zip(ref_window[t..t + r].iter())
            .filter(|(a, b)| a == b)
            .count();
        if matches > best_matches {
            best_matches = matches;
            best_shift = t;
        }
    }
    SpectralHit {
        shift: best_shift,
        mismatches: r - best_matches,
        matches: best_matches,
    }
}

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[test]
fn rejects_window_smaller_than_read() {
    assert!(scan(b"ACGT", b"ACG", 5).is_none());
}

#[test]
fn exact_match_at_zero() {
    let hit = scan(b"ACGTACGT", b"ACGTACGT", 0).unwrap();
    assert_eq!(hit.shift, 0);
    assert_eq!(hit.mismatches, 0);
    assert_eq!(hit.matches, 8);
}

#[test]
fn exact_match_in_middle() {
    let hit = scan(b"ACGTACGT", b"NNNNACGTACGTNNNN", 0).unwrap();
    assert_eq!(hit.shift, 4);
    assert_eq!(hit.mismatches, 0);
}

#[test]
fn single_mismatch_anywhere() {
    let read = b"ACGTACGT";
    let text = b"NNACGTAXGTNN";
    let hit = scan(read, text, 5).unwrap();
    let nh = naive(read, text);
    assert_eq!(hit.shift, nh.shift);
    assert_eq!(hit.mismatches, nh.mismatches);
    assert_eq!(hit.mismatches, 1);
}

#[test]
fn rejects_when_above_threshold() {
    let read = b"AAAAAAAAAA";
    let text = b"TTTTTTTTTTTT";
    assert!(scan(read, text, 3).is_none());
    let hit = scan(read, text, 10).unwrap();
    assert_eq!(hit.mismatches, 10);
}

#[test]
fn longer_read_avx_path() {
    // 150-bp read exercises both SIMD chunks (4 × 32 = 128) and 22-byte tail.
    let read: Vec<u8> = (0..150).map(|i| b"ACGT"[i % 4]).collect();
    let mut text: Vec<u8> = (0..200).map(|i| b"ACGT"[(i + 7) % 4]).collect();
    text[30..30 + 150].copy_from_slice(&read);
    text[30 + 5] = if read[5] == b'A' { b'T' } else { b'A' };
    text[30 + 100] = if read[100] == b'C' { b'G' } else { b'C' };

    let hit = scan(&read, &text, 5).unwrap();
    let nh = naive(&read, &text);
    assert_eq!(hit.shift, nh.shift);
    assert_eq!(hit.mismatches, nh.mismatches);
    assert_eq!(hit.shift, 30);
    assert_eq!(hit.mismatches, 2);
}

#[test]
fn random_corpus_matches_naive() {
    let bases = b"ACGT";
    let mut rng = 0xFEEDFACEu64;
    for _ in 0..200 {
        let r = 16 + (xorshift(&mut rng) as usize % 200);
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
        let nh = naive(&read, &text);
        let hit = scan(&read, &text, r).unwrap();
        // Multiple shifts can tie; verify the reported shift achieves the
        // claimed Hamming distance and matches the naive optimum.
        let actual = (0..r).filter(|&i| read[i] != text[hit.shift + i]).count();
        assert_eq!(
            hit.mismatches, actual,
            "Reported mismatches mismatched actual at r={r} w={w}"
        );
        assert_eq!(
            hit.mismatches, nh.mismatches,
            "Optimum mismatches differ from naive at r={r} w={w}"
        );
    }
}
