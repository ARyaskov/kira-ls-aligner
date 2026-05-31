//! Bit-Packed Spectral Sieve — multi-shift Hamming search over 2-bit DNA.

/// Final result reported on success — same shape as `spectral::SpectralHit` so the integration.
#[derive(Clone, Copy, Debug)]
pub struct PackedHit {
    pub shift: usize,
    pub mismatches: usize,
    pub matches: usize,
}

#[inline]
fn encode_base(b: u8) -> u8 {
    match b {
        b'A' | b'a' => 0b00,
        b'C' | b'c' => 0b01,
        b'G' | b'g' => 0b10,
        b'T' | b't' => 0b11,
        _ => 0b00,
    }
}

/// 2-bit packed DNA with bookkeeping for the actual nucleotide count.
#[derive(Clone, Debug)]
pub struct PackedDna {
    /// `bits[i]` holds nucleotides `4i .. 4i+4`, low-bit-first within the byte.
    pub bits: Vec<u8>,
    /// Number of nucleotides actually represented.
    pub len: usize,
}

impl PackedDna {
    /// Pack `seq` into 2-bit form.
    pub fn pack(seq: &[u8]) -> Self {
        let bytes = seq.len().div_ceil(4);
        let mut bits = vec![0u8; bytes];
        for (i, &b) in seq.iter().enumerate() {
            bits[i / 4] |= encode_base(b) << ((i % 4) * 2);
        }
        Self {
            bits,
            len: seq.len(),
        }
    }

    /// Produce 4 bit-shifted copies of this packed buffer, at bit-offsets 0, 2, 4, 6.
    pub fn pre_shifted_window(&self) -> [Vec<u8>; 4] {
        [
            self.bits.clone(),
            shift_bits_right(&self.bits, 2),
            shift_bits_right(&self.bits, 4),
            shift_bits_right(&self.bits, 6),
        ]
    }
}

/// Shift a packed buffer right by `bit_offset` bits (in {0, 2, 4, 6}).
fn shift_bits_right(bits: &[u8], bit_offset: usize) -> Vec<u8> {
    debug_assert!(bit_offset < 8);
    if bit_offset == 0 {
        return bits.to_vec();
    }
    let mut out = vec![0u8; bits.len()];
    for i in 0..bits.len() {
        let lo = bits[i] >> bit_offset;
        let hi = if i + 1 < bits.len() {
            bits[i + 1] << (8 - bit_offset)
        } else {
            0
        };
        out[i] = lo | hi;
    }
    out
}

/// Scan all valid shifts and return one with mismatches ≤ `max_mismatches`.
pub fn scan(
    read_packed: &PackedDna,
    ref_pre_shifted: &[Vec<u8>; 4],
    ref_len: usize,
    max_mismatches: usize,
) -> Option<PackedHit> {
    let r = read_packed.len;
    if r == 0 || ref_len < r {
        return None;
    }
    let n_shifts = ref_len - r + 1;
    let need_matches = r.saturating_sub(max_mismatches);
    let read_bytes = read_packed.bits.len();

    for t in 0..n_shifts {
        let phase = t % 4;
        let byte_off = t / 4;
        let ref_buf = &ref_pre_shifted[phase];
        if byte_off + read_bytes > ref_buf.len() {
            continue;
        }
        let mismatches = mismatch_count(
            &read_packed.bits,
            &ref_buf[byte_off..byte_off + read_bytes],
            r,
        );
        let matches = r - mismatches;
        if matches >= need_matches {
            return Some(PackedHit {
                shift: t,
                mismatches,
                matches,
            });
        }
    }
    None
}

/// Exhaustive variant of [`scan`] — returns the shift with the **minimum** mismatch count across.
pub fn scan_best(
    read_packed: &PackedDna,
    ref_pre_shifted: &[Vec<u8>; 4],
    ref_len: usize,
) -> Option<PackedHit> {
    let r = read_packed.len;
    if r == 0 || ref_len < r {
        return None;
    }
    let n_shifts = ref_len - r + 1;
    let read_bytes = read_packed.bits.len();

    let mut best_shift = 0usize;
    let mut best_mism = usize::MAX;
    for t in 0..n_shifts {
        let phase = t % 4;
        let byte_off = t / 4;
        let ref_buf = &ref_pre_shifted[phase];
        if byte_off + read_bytes > ref_buf.len() {
            continue;
        }
        let mismatches = mismatch_count(
            &read_packed.bits,
            &ref_buf[byte_off..byte_off + read_bytes],
            r,
        );
        if mismatches < best_mism {
            best_mism = mismatches;
            best_shift = t;
            // Perfect hit — no possible improvement, stop early.
            if best_mism == 0 {
                break;
            }
        }
    }
    if best_mism == usize::MAX {
        None
    } else {
        Some(PackedHit {
            shift: best_shift,
            mismatches: best_mism,
            matches: r - best_mism,
        })
    }
}

/// Count mismatched nucleotide pairs between two 2-bit packed buffers of equal byte length.
fn mismatch_count(read: &[u8], ref_packed: &[u8], n_nucleotides: usize) -> usize {
    debug_assert_eq!(read.len(), ref_packed.len());
    let n_bytes = read.len();

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime AVX2 detection above.
            return unsafe { mismatch_count_avx2(read, ref_packed, n_nucleotides) };
        }
    }
    mismatch_count_scalar(read, ref_packed, n_bytes, n_nucleotides)
}

fn mismatch_count_scalar(
    read: &[u8],
    ref_packed: &[u8],
    n_bytes: usize,
    n_nucleotides: usize,
) -> usize {
    let mut total: u32 = 0;
    let full_bytes = n_nucleotides / 4;
    let tail_pairs = n_nucleotides % 4;
    for i in 0..full_bytes {
        let xor = read[i] ^ ref_packed[i];
        let pair_or = xor | (xor >> 1);
        let pair_mask = pair_or & 0x55;
        total += pair_mask.count_ones();
    }
    if tail_pairs != 0 && full_bytes < n_bytes {
        let xor = read[full_bytes] ^ ref_packed[full_bytes];
        let pair_or = xor | (xor >> 1);
        let pair_mask = pair_or & 0x55;
        // Keep only the lowest `tail_pairs` pair-bits (positions 0, 2, 4, …).
        let keep = ((1u8 << (tail_pairs * 2)) - 1) & 0x55;
        total += (pair_mask & keep).count_ones();
    }
    total as usize
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mismatch_count_avx2(
    read: &[u8],
    ref_packed: &[u8],
    n_nucleotides: usize,
) -> usize {
    use std::arch::x86_64::{
        __m256i, _mm256_and_si256, _mm256_loadu_si256, _mm256_or_si256, _mm256_set1_epi8,
        _mm256_srli_epi64, _mm256_storeu_si256, _mm256_xor_si256,
    };

    let n_bytes = read.len();
    let full_bytes = n_nucleotides / 4;
    let tail_pairs = n_nucleotides % 4;

    let mask = _mm256_set1_epi8(0x55u8 as i8);

    let mut total: u64 = 0;
    let mut i: usize = 0;

    while i + 32 <= full_bytes {
        // SAFETY: bounded by loop condition; slices length-validated above.
        let a = unsafe { _mm256_loadu_si256(read.as_ptr().add(i) as *const __m256i) };
        let b = unsafe { _mm256_loadu_si256(ref_packed.as_ptr().add(i) as *const __m256i) };
        let xor = _mm256_xor_si256(a, b);
        let shifted = _mm256_srli_epi64(xor, 1);
        let pair_or = _mm256_or_si256(xor, shifted);
        let pair_mask = _mm256_and_si256(pair_or, mask);
        let mut tmp = [0u64; 4];
        // SAFETY: tmp is a stack-allocated 32-byte buffer.
        unsafe {
            _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, pair_mask);
        }
        total += tmp[0].count_ones() as u64
            + tmp[1].count_ones() as u64
            + tmp[2].count_ones() as u64
            + tmp[3].count_ones() as u64;
        i += 32;
    }

    // Scalar tail over remaining full bytes.
    while i < full_bytes {
        let xor = read[i] ^ ref_packed[i];
        let pair_or = xor | (xor >> 1);
        let pair_mask = pair_or & 0x55;
        total += pair_mask.count_ones() as u64;
        i += 1;
    }

    // Partial last byte (e.g. read length not a multiple of 4).
    if tail_pairs != 0 && full_bytes < n_bytes {
        let xor = read[full_bytes] ^ ref_packed[full_bytes];
        let pair_or = xor | (xor >> 1);
        let pair_mask = pair_or & 0x55;
        let keep = ((1u8 << (tail_pairs * 2)) - 1) & 0x55;
        total += (pair_mask & keep).count_ones() as u64;
    }

    total as usize
}

#[cfg(test)]
#[path = "../../tests/unit/alignment_bitpacked.rs"]
mod tests;
