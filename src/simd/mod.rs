use std::sync::OnceLock;

/// SIMD dispatch helpers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SimdMode {
    Scalar,
    Avx2,
    Neon,
}

/// Detect the best available SIMD mode at runtime via kira-simd.
pub fn detect() -> SimdMode {
    match kira_simd::backend::active_backend() {
        kira_simd::backend::Backend::Avx2 => SimdMode::Avx2,
        kira_simd::backend::Backend::Neon => SimdMode::Neon,
        _ => SimdMode::Scalar,
    }
}

#[inline]
pub fn detect_cached() -> SimdMode {
    static MODE: OnceLock<SimdMode> = OnceLock::new();
    *MODE.get_or_init(detect)
}

/// Count mismatches between two equal-length slices.
#[inline]
pub fn count_mismatches(a: &[u8], b: &[u8]) -> usize {
    kira_simd::compare::count_mismatches(a, b)
}

/// Count mismatches with early stop when limit is exceeded.
#[inline]
pub fn count_mismatches_bounded(a: &[u8], b: &[u8], max_mismatches: u32) -> u32 {
    kira_simd::compare::count_mismatches_bounded(a, b, max_mismatches)
}

/// out[i] = ref_pos[i] - query_pos[i].
#[inline]
pub fn compute_diagonals(ref_pos: &[u32], query_pos: &[u32], out_diags: &mut [i32]) {
    kira_simd::compare::compute_diagonals_i32(ref_pos, query_pos, out_diags)
}

/// out[i] = 1 if read_pos[i] < read_len else 0.
#[inline]
pub fn mask_read_pos_in_range(read_pos: &[u32], read_len: u32, mask: &mut [u8]) {
    kira_simd::compare::mask_lt_u32(read_pos, read_len, mask)
}
