/// SIMD dispatch helpers. Unsafe code is confined to this module.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SimdMode {
    Scalar,
    Avx2,
    Neon,
}

/// Detect the best available SIMD mode at runtime.
pub fn detect() -> SimdMode {
    #[cfg(all(target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            return SimdMode::Avx2;
        }
    }
    #[cfg(all(target_arch = "aarch64"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return SimdMode::Neon;
        }
    }
    SimdMode::Scalar
}

/// Count mismatches between two equal-length slices.
pub fn count_mismatches(a: &[u8], b: &[u8]) -> usize {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0;
    }
    match detect() {
        #[cfg(target_arch = "x86_64")]
        SimdMode::Avx2 => unsafe { avx2::count_mismatches(a, b) },
        #[cfg(target_arch = "aarch64")]
        SimdMode::Neon => unsafe { neon::count_mismatches(a, b) },
        _ => scalar_count_mismatches(a, b),
    }
}

fn scalar_count_mismatches(a: &[u8], b: &[u8]) -> usize {
    let len = a.len().min(b.len());
    let mut mismatches = 0usize;
    for i in 0..len {
        if a[i] != b[i] {
            mismatches += 1;
        }
    }
    mismatches
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::{__m256i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8};

    #[target_feature(enable = "avx2")]
    pub unsafe fn count_mismatches(a: &[u8], b: &[u8]) -> usize {
        let mut i = 0usize;
        let len = a.len().min(b.len());
        let mut mismatches = 0usize;
        while i + 32 <= len {
            let pa = unsafe { a.as_ptr().add(i) } as *const __m256i;
            let pb = unsafe { b.as_ptr().add(i) } as *const __m256i;
            let va = unsafe { _mm256_loadu_si256(pa) };
            let vb = unsafe { _mm256_loadu_si256(pb) };
            let eq = _mm256_cmpeq_epi8(va, vb);
            let mask = _mm256_movemask_epi8(eq) as u32;
            let matches = mask.count_ones() as usize;
            mismatches += 32 - matches;
            i += 32;
        }
        while i < len {
            if a[i] != b[i] {
                mismatches += 1;
            }
            i += 1;
        }
        mismatches
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::{uint8x16_t, vaddvq_u8, vceqq_u8, vld1q_u8, vshrq_n_u8};

    #[target_feature(enable = "neon")]
    pub unsafe fn count_mismatches(a: &[u8], b: &[u8]) -> usize {
        let mut i = 0usize;
        let len = a.len().min(b.len());
        let mut mismatches = 0usize;
        while i + 16 <= len {
            let pa = unsafe { a.as_ptr().add(i) };
            let pb = unsafe { b.as_ptr().add(i) };
            let va: uint8x16_t = unsafe { vld1q_u8(pa) };
            let vb: uint8x16_t = unsafe { vld1q_u8(pb) };
            let eq = vceqq_u8(va, vb);
            let ones = vshrq_n_u8(eq, 7);
            let matches = vaddvq_u8(ones) as usize;
            mismatches += 16 - matches;
            i += 16;
        }
        while i < len {
            if a[i] != b[i] {
                mismatches += 1;
            }
            i += 1;
        }
        mismatches
    }
}
