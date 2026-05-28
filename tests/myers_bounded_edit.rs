//! Integration tests for `alignment::myers::bounded_edit_distance`.
//!
//! Each Myers result is cross-checked against a naive O(n·m) semi-global
//! DP — the gold standard for edit distance with free leading/trailing
//! text gaps and full pattern consumption.

use kira_ls_aligner::alignment::myers::bounded_edit_distance;

fn naive_semi_global(pattern: &[u8], text: &[u8]) -> (usize, usize) {
    let m = pattern.len();
    let n = text.len();
    let mut prev = vec![0usize; n + 1];
    let mut cur = vec![0usize; n + 1];
    for i in 1..=m {
        cur[0] = i;
        for j in 1..=n {
            let cost = if pattern[i - 1] == text[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j - 1] + cost)
                .min(prev[j] + 1)
                .min(cur[j - 1] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    let mut best = prev[0];
    let mut pos = 0;
    for (j, &v) in prev.iter().enumerate() {
        if v < best {
            best = v;
            pos = j;
        }
    }
    (best, pos)
}

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[test]
fn empty_pattern_matches_at_zero() {
    assert_eq!(bounded_edit_distance(b"", b"ACGT", 0), Some((0, 0)));
}

#[test]
fn empty_text_distance_is_pattern_length() {
    assert_eq!(bounded_edit_distance(b"ACGT", b"", 4), Some((4, 0)));
}

#[test]
fn exact_match_one_block() {
    let pattern = b"ACGTACGTACGT";
    let text = b"NNACGTACGTACGTNN";
    let (d, end) = bounded_edit_distance(pattern, text, 5).unwrap();
    let (nd, nend) = naive_semi_global(pattern, text);
    assert_eq!(d, 0);
    assert_eq!(d, nd);
    assert_eq!(end, nend);
}

#[test]
fn one_mismatch_matches_naive() {
    let pattern = b"ACGTACGTACGT";
    let text = b"NNACGTAXGTACGTNN";
    let (d, end) = bounded_edit_distance(pattern, text, 3).unwrap();
    let (nd, nend) = naive_semi_global(pattern, text);
    assert_eq!(d, nd);
    assert_eq!(end, nend);
}

#[test]
fn one_insertion_matches_naive() {
    let pattern = b"ACGTACGT";
    let text = b"NNACGTAAACGTNN";
    let (d, end) = bounded_edit_distance(pattern, text, 3).unwrap();
    let (nd, nend) = naive_semi_global(pattern, text);
    assert_eq!(d, nd);
    assert_eq!(end, nend);
}

#[test]
fn one_deletion_matches_naive() {
    let pattern = b"ACGTACGT";
    let text = b"NNACGACGTNN";
    let (d, end) = bounded_edit_distance(pattern, text, 3).unwrap();
    let (nd, nend) = naive_semi_global(pattern, text);
    assert_eq!(d, nd);
    assert_eq!(end, nend);
}

#[test]
fn cutoff_returns_none() {
    let pattern = b"AAAAAAAA";
    let text = b"TTTTTTTT";
    assert_eq!(bounded_edit_distance(pattern, text, 3), None);
    assert_eq!(
        bounded_edit_distance(pattern, text, 8).map(|(d, _)| d),
        Some(8)
    );
}

#[test]
fn multi_block_64_boundary() {
    let pattern: Vec<u8> = (0..64).map(|i| b"ACGT"[i % 4]).collect();
    let mut text = pattern.clone();
    text[20] = b'A';
    let (d, _) = bounded_edit_distance(&pattern, &text, 5).unwrap();
    let (nd, _) = naive_semi_global(&pattern, &text);
    assert_eq!(d, nd);
}

#[test]
fn multi_block_65_boundary() {
    let pattern: Vec<u8> = (0..65).map(|i| b"ACGT"[i % 4]).collect();
    let mut text = pattern.clone();
    text[64] = if pattern[64] == b'A' { b'C' } else { b'A' };
    let (d, _) = bounded_edit_distance(&pattern, &text, 5).unwrap();
    let (nd, _) = naive_semi_global(&pattern, &text);
    assert_eq!(d, nd);
}

#[test]
fn multi_block_150_random() {
    let pattern: Vec<u8> = (0..150).map(|i| b"ACGT"[(i * 7 + 3) % 4]).collect();
    let mut text = pattern.clone();
    text[10] = if text[10] == b'A' { b'T' } else { b'A' };
    text[77] = if text[77] == b'C' { b'G' } else { b'C' };
    text[140] = if text[140] == b'G' { b'A' } else { b'G' };
    let (d, _) = bounded_edit_distance(&pattern, &text, 10).unwrap();
    let (nd, _) = naive_semi_global(&pattern, &text);
    assert_eq!(d, nd);
}

#[test]
fn random_corpus_matches_naive() {
    let bases = b"ACGT";
    let mut rng = 0xDEADBEEFu64;
    for _ in 0..100 {
        let m = 1 + (xorshift(&mut rng) as usize % 192);
        let n = m + (xorshift(&mut rng) as usize % 64);
        let pattern: Vec<u8> = (0..m)
            .map(|_| bases[(xorshift(&mut rng) as usize) % 4])
            .collect();
        let mut text: Vec<u8> = (0..n)
            .map(|_| bases[(xorshift(&mut rng) as usize) % 4])
            .collect();
        if xorshift(&mut rng) % 2 == 0 && n >= m {
            let start = (xorshift(&mut rng) as usize) % (n - m + 1);
            text[start..start + m].copy_from_slice(&pattern);
        }
        let (nd, _) = naive_semi_global(&pattern, &text);
        let (d, end) = bounded_edit_distance(&pattern, &text, m).expect("max_k = m always succeeds");
        assert_eq!(
            d, nd,
            "distance mismatch for m={m} n={n} pattern={:?} text={:?}",
            String::from_utf8_lossy(&pattern),
            String::from_utf8_lossy(&text)
        );
        // Multiple text prefixes can tie on min distance; verify the
        // reported end-position genuinely reaches the minimum.
        let (d_at_end, _) = naive_semi_global(&pattern, &text[..end]);
        assert_eq!(d_at_end, nd, "Myers end={end} does not reach min distance for m={m} n={n}");
    }
}
