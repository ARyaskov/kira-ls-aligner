//! Spectral Sieve — exhaustive multi-shift Hamming search.

/// Result of a successful multi-shift Hamming search.
#[derive(Clone, Copy, Debug)]
pub struct SpectralHit {
    /// Best shift in the reference window: read aligns to `ref_window[shift..shift + read.len()]`.
    pub shift: usize,
    /// Hamming distance at that shift (matches that should appear in MD/NM).
    pub mismatches: usize,
    /// Convenience: `read.len() - mismatches`.
    pub matches: usize,
}

/// Scan all valid shifts and return the one with the fewest mismatches.
pub fn scan(read: &[u8], ref_window: &[u8], max_mismatches: usize) -> Option<SpectralHit> {
    let r = read.len();
    let w = ref_window.len();
    if r == 0 || w < r {
        return None;
    }
    let n_shifts = w - r + 1;
    let need_matches = r.saturating_sub(max_mismatches);

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime AVX2 detection above.
            return unsafe { scan_avx2(read, ref_window, n_shifts, need_matches) };
        }
    }
    scan_scalar(read, ref_window, n_shifts, need_matches)
}

/// Scalar reference implementation with the same early-termination logic as the AVX2 path.
fn scan_scalar(
    read: &[u8],
    ref_window: &[u8],
    n_shifts: usize,
    need_matches: usize,
) -> Option<SpectralHit> {
    let r = read.len();
    let mut best_matches = 0usize;
    let mut best_shift = 0usize;
    for t in 0..n_shifts {
        let target = need_matches.max(best_matches + 1);
        let mut matches = 0usize;
        for i in 0..r {
            if read[i] == ref_window[t + i] {
                matches += 1;
            } else {
                // After a mismatch, check whether we can still reach target.
                let remaining = r - i - 1;
                if matches + remaining < target {
                    matches = 0; // sentinel: shift abandoned
                    break;
                }
            }
        }
        if matches > best_matches {
            best_matches = matches;
            best_shift = t;
            if best_matches == r {
                break; // perfect — nothing can be better
            }
        }
    }
    if best_matches >= need_matches {
        Some(SpectralHit {
            shift: best_shift,
            mismatches: r - best_matches,
            matches: best_matches,
        })
    } else {
        None
    }
}

/// AVX2 implementation: for every candidate shift, count matches in 32-byte SIMD chunks using.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_avx2(
    read: &[u8],
    ref_window: &[u8],
    n_shifts: usize,
    need_matches: usize,
) -> Option<SpectralHit> {
    use std::arch::x86_64::{
        __m256i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8,
    };

    let r = read.len();
    let read_ptr = read.as_ptr();
    let ref_ptr = ref_window.as_ptr();

    let mut best_matches: u32 = 0;
    let mut best_shift: usize = 0;

    for t in 0..n_shifts {
        let target = need_matches.max(best_matches as usize + 1) as u32;
        let mut matches: u32 = 0;
        let mut i: usize = 0;
        let mut abandoned = false;
        while i + 32 <= r {
            // SAFETY: bounded by loop condition; both pointers are valid for
            let read_v =
                unsafe { _mm256_loadu_si256(read_ptr.add(i) as *const __m256i) };
            let ref_v =
                unsafe { _mm256_loadu_si256(ref_ptr.add(t + i) as *const __m256i) };
            let eq = _mm256_cmpeq_epi8(read_v, ref_v);
            let mask = _mm256_movemask_epi8(eq) as u32;
            matches += mask.count_ones();
            i += 32;
            // After each chunk, check whether this shift can still reach `target`.
            let remaining = (r - i) as u32;
            if matches + remaining < target {
                abandoned = true;
                break;
            }
        }
        if abandoned {
            continue;
        }
        // Tail (< 32 bytes): scalar fallback for the trailing bytes.
        for j in i..r {
            // SAFETY: indices in-range by loop condition.
            let a = unsafe { *read_ptr.add(j) };
            let b = unsafe { *ref_ptr.add(t + j) };
            if a == b {
                matches += 1;
            }
        }
        if matches > best_matches {
            best_matches = matches;
            best_shift = t;
            if best_matches as usize == r {
                break; // perfect match — nothing can be better
            }
        }
    }

    if best_matches as usize >= need_matches {
        Some(SpectralHit {
            shift: best_shift,
            mismatches: r - best_matches as usize,
            matches: best_matches as usize,
        })
    } else {
        None
    }
}
