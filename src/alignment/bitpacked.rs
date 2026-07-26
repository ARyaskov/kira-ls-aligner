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
    /// False when the input contained anything outside A/C/G/T. The 2-bit
    /// representation cannot distinguish those symbols from A, so callers
    /// must not use it as an exact mismatch certificate in that case.
    pub valid: bool,
}

impl PackedDna {
    /// Pack `seq` into 2-bit form.
    pub fn pack(seq: &[u8]) -> Self {
        let mut bits = Vec::new();
        let valid = pack_into(seq, &mut bits);
        Self {
            bits,
            len: seq.len(),
            valid,
        }
    }

    /// Produce 4 bit-shifted copies of this packed buffer, at bit-offsets 0, 2, 4, 6.
    pub fn pre_shifted_window(&self) -> [Vec<u8>; 4] {
        let mut out = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        pre_shift_into(&self.bits, &mut out);
        out
    }
}

/// Pack `seq` into an existing buffer, returning whether every symbol was ACGT.
/// Allocation-free variant of [`PackedDna::pack`].
pub fn pack_into(seq: &[u8], bits: &mut Vec<u8>) -> bool {
    let n = seq.len();
    let bytes = n.div_ceil(4);
    bits.clear();
    bits.resize(bytes, 0u8);
    let mut valid = true;

    // Build each output byte from four bases at once: the per-base
    // `bits[i / 4] |= ...` form serialises on a load-store dependency.
    let chunks = n / 4;
    for c in 0..chunks {
        let q = &seq[c * 4..c * 4 + 4];
        valid &= is_acgt(q[0]) & is_acgt(q[1]) & is_acgt(q[2]) & is_acgt(q[3]);
        bits[c] = encode_base(q[0])
            | (encode_base(q[1]) << 2)
            | (encode_base(q[2]) << 4)
            | (encode_base(q[3]) << 6);
    }
    // Tail of 1-3 bases; the unused high bits stay zero.
    let rest = chunks * 4;
    if rest < n {
        let mut byte = 0u8;
        for (j, &b) in seq[rest..].iter().enumerate() {
            valid &= is_acgt(b);
            byte |= encode_base(b) << (j * 2);
        }
        bits[chunks] = byte;
    }
    valid
}

#[inline(always)]
fn is_acgt(b: u8) -> bool {
    matches!(b, b'A' | b'a' | b'C' | b'c' | b'G' | b'g' | b'T' | b't')
}

/// Fill `out` with the 4 bit-shifted copies of `bits` (offsets 0, 2, 4, 6).
/// Allocation-free variant of [`PackedDna::pre_shifted_window`].
pub fn pre_shift_into(bits: &[u8], out: &mut [Vec<u8>; 4]) {
    for (phase, dst) in out.iter_mut().enumerate() {
        shift_bits_right_into(bits, phase * 2, dst);
    }
}

/// Shift a packed buffer right by `bit_offset` bits (in {0, 2, 4, 6}) into `out`.
fn shift_bits_right_into(bits: &[u8], bit_offset: usize, out: &mut Vec<u8>) {
    debug_assert!(bit_offset < 8);
    out.clear();
    out.resize(bits.len(), 0u8);
    if bit_offset == 0 {
        out.copy_from_slice(bits);
        return;
    }
    for i in 0..bits.len() {
        let lo = bits[i] >> bit_offset;
        let hi = if i + 1 < bits.len() {
            bits[i + 1] << (8 - bit_offset)
        } else {
            0
        };
        out[i] = lo | hi;
    }
}

/// Scan all valid shifts and return one with mismatches ≤ `max_mismatches`.
pub fn scan(
    read_packed: &PackedDna,
    ref_pre_shifted: &[Vec<u8>; 4],
    ref_len: usize,
    max_mismatches: usize,
) -> Option<PackedHit> {
    if !read_packed.valid {
        return None;
    }
    let hit = scan_best(read_packed, ref_pre_shifted, ref_len)?;
    (hit.mismatches <= max_mismatches).then_some(hit)
}

/// Exhaustive best/second-best mismatch counts. The second value is the best
/// count at a distinct shift and is used to reject locally ambiguous ungapped
/// placements.
pub fn scan_best_with_second(
    read_packed: &PackedDna,
    ref_pre_shifted: &[Vec<u8>; 4],
    ref_len: usize,
) -> Option<(PackedHit, Option<usize>)> {
    scan_best_with_second_raw(
        &read_packed.bits,
        read_packed.len,
        read_packed.valid,
        ref_pre_shifted,
        ref_len,
    )
}

/// Slice-based [`scan_best_with_second`] — lets callers hold the packed read and
/// the pre-shifted reference in reusable scratch instead of a fresh `PackedDna`.
pub fn scan_best_with_second_raw(
    read_bits: &[u8],
    read_len: usize,
    read_valid: bool,
    ref_pre_shifted: &[Vec<u8>; 4],
    ref_len: usize,
) -> Option<(PackedHit, Option<usize>)> {
    let r = read_len;
    if r == 0 || ref_len < r || !read_valid {
        return None;
    }
    let n_shifts = ref_len - r + 1;
    let read_bytes = read_bits.len();

    let mut best_shift = 0usize;
    let mut best_mism = usize::MAX;
    let mut second_mism = usize::MAX;
    for t in 0..n_shifts {
        let phase = t % 4;
        let byte_off = t / 4;
        let ref_buf = &ref_pre_shifted[phase];
        if byte_off + read_bytes > ref_buf.len() {
            continue;
        }
        let mismatches =
            mismatch_count(read_bits, &ref_buf[byte_off..byte_off + read_bytes], r);
        if mismatches < best_mism {
            second_mism = best_mism;
            best_mism = mismatches;
            best_shift = t;
        } else if mismatches < second_mism {
            second_mism = mismatches;
        }
    }
    if best_mism == usize::MAX {
        None
    } else {
        Some((
            PackedHit {
                shift: best_shift,
                mismatches: best_mism,
                matches: r - best_mism,
            },
            (second_mism != usize::MAX).then_some(second_mism),
        ))
    }
}

/// Exhaustive variant of [`scan`] — returns the shift with the **minimum** mismatch count across.
pub fn scan_best(
    read_packed: &PackedDna,
    ref_pre_shifted: &[Vec<u8>; 4],
    ref_len: usize,
) -> Option<PackedHit> {
    scan_best_with_second(read_packed, ref_pre_shifted, ref_len).map(|(best, _)| best)
}

/// Count mismatched nucleotide pairs between two 2-bit packed buffers of equal byte length.
/// Below this many packed bytes (256 nucleotides) the 64-bit word kernel wins:
/// the AVX2 one spills its accumulator to the stack to popcount it, and that
/// round trip costs more than the extra 64-bit operations it saves. It also
/// avoids the runtime feature check, which runs once per candidate shift.
const U64_KERNEL_MAX_BYTES: usize = 64;

fn mismatch_count(read: &[u8], ref_packed: &[u8], n_nucleotides: usize) -> usize {
    debug_assert_eq!(read.len(), ref_packed.len());
    let n_bytes = read.len();

    if n_bytes <= U64_KERNEL_MAX_BYTES {
        return mismatch_count_u64(read, ref_packed, n_bytes, n_nucleotides);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime AVX2 detection above.
            return unsafe { mismatch_count_avx2(read, ref_packed, n_nucleotides) };
        }
    }
    mismatch_count_scalar(read, ref_packed, n_bytes, n_nucleotides)
}

/// Count mismatching 2-bit codes eight packed bytes at a time. A pair differs iff
/// either bit differs, so `x | (x >> 1)` masked to each pair's low bit marks
/// exactly the differing pairs.
fn mismatch_count_u64(
    read: &[u8],
    ref_packed: &[u8],
    n_bytes: usize,
    n_nucleotides: usize,
) -> usize {
    const PAIR_LOW_BITS: u64 = 0x5555_5555_5555_5555;
    let full_bytes = n_nucleotides / 4;
    let tail_pairs = n_nucleotides % 4;

    let mut total: u32 = 0;
    let mut i = 0usize;
    while i + 8 <= full_bytes {
        let a = u64::from_le_bytes(read[i..i + 8].try_into().unwrap());
        let b = u64::from_le_bytes(ref_packed[i..i + 8].try_into().unwrap());
        let xor = a ^ b;
        total += ((xor | (xor >> 1)) & PAIR_LOW_BITS).count_ones();
        i += 8;
    }
    while i < full_bytes {
        let xor = read[i] ^ ref_packed[i];
        total += ((xor | (xor >> 1)) & 0x55).count_ones();
        i += 1;
    }
    if tail_pairs != 0 && full_bytes < n_bytes {
        let xor = read[full_bytes] ^ ref_packed[full_bytes];
        let pair_mask = (xor | (xor >> 1)) & 0x55;
        // Keep only the lowest `tail_pairs` pair-bits (positions 0, 2, 4, …).
        let keep = ((1u8 << (tail_pairs * 2)) - 1) & 0x55;
        total += (pair_mask & keep).count_ones();
    }
    total as usize
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
unsafe fn mismatch_count_avx2(read: &[u8], ref_packed: &[u8], n_nucleotides: usize) -> usize {
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
