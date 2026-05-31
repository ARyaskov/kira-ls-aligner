//! 32-lane INT8 banded Smith-Waterman, gated on AVX-VNNI.
//!
//! AVX2 already exposes saturating i8 add (`vpaddsb`) and i8 max (`vpmaxsb`), so
//! the SW max-plus recurrence can run at 32 lanes per 256-bit register — twice
//! the lane count of the existing i16 path. The CPU-feature gate is AVX-VNNI,
//! not because the recurrence is a sum-of-products (it isn't — there is no
//! natural place to spend `vpdpbusd` in `H = max(0, H_diag + s, E, F)`), but
//! because AVX-VNNI implies an Alder Lake / Sapphire Rapids / Zen 4-class
//! pipeline where saturated i8 throughput is reliably ~2× the i16 path. On
//! older AVX2 CPUs the i8 path does not win consistently and we keep the i16
//! kernel.
//!
//! ## Saturation
//!
//! Per-cell H is bounded by `read_len × match_score` minus penalty mass. At the
//! default `match=1`, this exceeds the i8 range `[-128, 127]` for reads above
//! ~120 bp. The dispatcher refuses the i8 path past `INT8_SAFE_MAX_H`. The
//! kernel additionally tracks a per-lane saturation flag and, for any lane
//! that did saturate, re-runs that lane through the scalar banded SW so the
//! returned result is always correct.
//!
//! ## Layout
//!
//! Same as the i16 path: one __m256i per ref column for the H/E rolling rows,
//! and a `(q_len+1) × (r_len+1) × 32` flat trace buffer. Trace codes are
//! 0=stop, 1=match (diagonal), 2=insert (up), 3=delete (left).

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

use crate::alignment::{AlignmentConfig, BatchInput, SwResult, banded_sw_internal, push_cigar};
use crate::types::CigarKind;

/// Maximum H value the i8 kernel is willing to commit to without falling back.
///
/// Leaves headroom below 127 for one more `+match_score` step before saturation.
pub const INT8_SAFE_MAX_H: i32 = 120;

/// Returns true if the i8 kernel is safe given config and read length.
///
/// `read_len * match_score` is an upper bound on H for any monotone-increasing
/// SW trajectory; if even that bound fits below the saturation threshold we
/// know no lane will saturate regardless of input.
#[inline]
pub fn int8_path_viable(read_len: usize, cfg: AlignmentConfig) -> bool {
    let ms = cfg.match_score.max(1) as i64;
    let upper = (read_len as i64).saturating_mul(ms);
    upper < INT8_SAFE_MAX_H as i64
        && cfg.match_score.unsigned_abs() <= 32
        && cfg.mismatch.unsigned_abs() <= 32
        && cfg.gap_open.unsigned_abs() <= 32
        && cfg.gap_extend.unsigned_abs() <= 32
}

pub const LANES: usize = 32;

struct Scratch {
    prev_h: Vec<__m256i>,
    prev_e: Vec<__m256i>,
    cur_h: Vec<__m256i>,
    cur_e: Vec<__m256i>,
    trace: Vec<u8>,
    read_cols: Vec<u8>,
    ref_cols: Vec<u8>,
}

impl Scratch {
    const fn new() -> Self {
        Self {
            prev_h: Vec::new(),
            prev_e: Vec::new(),
            cur_h: Vec::new(),
            cur_e: Vec::new(),
            trace: Vec::new(),
            read_cols: Vec::new(),
            ref_cols: Vec::new(),
        }
    }
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Scratch> =
        const { std::cell::RefCell::new(Scratch::new()) };
}

/// 32-lane i8 SW. Caller guarantees AVX-VNNI is available and all inputs have
/// the same `read_seq.len()` and `ref_window.len()`.
#[target_feature(enable = "avx2,avxvnni")]
pub(crate) unsafe fn sw_batch_int8(
    inputs: &[BatchInput<'_>],
    cfg: AlignmentConfig,
) -> Vec<SwResult> {
    let lanes = inputs.len().min(LANES);
    let q_len = inputs[0].read_seq.len();
    let r_len = inputs[0].ref_window.len();

    let v_zero = _mm256_setzero_si256();
    let v_neg = _mm256_set1_epi8(-100);

    SCRATCH.with(|scratch| {
        let mut s = scratch.borrow_mut();
        let Scratch {
            prev_h,
            prev_e,
            cur_h,
            cur_e,
            trace,
            read_cols,
            ref_cols,
        } = &mut *s;

        prev_h.clear();
        prev_h.resize(r_len + 1, v_zero);
        prev_e.clear();
        prev_e.resize(r_len + 1, v_neg);
        cur_h.clear();
        cur_h.resize(r_len + 1, v_zero);
        cur_e.clear();
        cur_e.resize(r_len + 1, v_neg);
        trace.clear();
        trace.resize((q_len + 1) * (r_len + 1) * LANES, 0);
        read_cols.clear();
        read_cols.resize((q_len + 1) * LANES, 0);
        ref_cols.clear();
        ref_cols.resize((r_len + 1) * LANES, 0);

        // SAFETY: caller enables avx2+avxvnni.
        unsafe {
            sw_batch_int8_inner(
                inputs, cfg, lanes, q_len, r_len, prev_h, prev_e, cur_h, cur_e, trace,
                read_cols, ref_cols,
            )
        }
    })
}

#[target_feature(enable = "avx2,avxvnni")]
#[allow(clippy::too_many_arguments)]
unsafe fn sw_batch_int8_inner(
    inputs: &[BatchInput<'_>],
    cfg: AlignmentConfig,
    lanes: usize,
    q_len: usize,
    r_len: usize,
    prev_h: &mut Vec<__m256i>,
    prev_e: &mut Vec<__m256i>,
    cur_h: &mut Vec<__m256i>,
    cur_e: &mut Vec<__m256i>,
    trace: &mut [u8],
    read_cols: &mut [u8],
    ref_cols: &mut [u8],
) -> Vec<SwResult> {
    let v_zero = _mm256_setzero_si256();
    let v_neg = _mm256_set1_epi8(-100);
    let v_go = _mm256_set1_epi8(-(cfg.gap_open as i8));
    let v_ge = _mm256_set1_epi8(-(cfg.gap_extend as i8));
    let v_match = _mm256_set1_epi8(cfg.match_score as i8);
    let v_mism = _mm256_set1_epi8(-(cfg.mismatch as i8));
    let v_one = _mm256_set1_epi8(1);
    let v_two = _mm256_set1_epi8(2);
    let v_three = _mm256_set1_epi8(3);
    let v_sat = _mm256_set1_epi8(INT8_SAFE_MAX_H as i8);

    let trace_w = r_len + 1;

    // Transpose read/ref into lane-interleaved buffers. read_cols[i * 32 + k]
    // is the i-th base of lane k.
    for i in 1..=q_len {
        let base = i * LANES;
        for k in 0..lanes {
            read_cols[base + k] = inputs[k].read_seq[i - 1];
        }
    }
    for j in 1..=r_len {
        let base = j * LANES;
        for k in 0..lanes {
            ref_cols[base + k] = inputs[k].ref_window[j - 1];
        }
    }

    let mut best_v = v_zero;
    let mut best_qlen_v = v_zero;
    let mut sat_v = v_zero;
    let mut best_i = [0i32; LANES];
    let mut best_j = [0i32; LANES];
    let mut best_qlen_j = [0i32; LANES];

    let mut abort_scores = [0i32; LANES];
    for k in 0..lanes {
        abort_scores[k] = inputs[k].abort_score;
    }
    let mut lane_done = [false; LANES];

    let mut best_arr = [0i8; LANES];

    for i in 1..=q_len {
        cur_h[0] = v_zero;
        cur_e[0] = v_neg;
        let mut cur_f = v_neg;

        let read_v =
            unsafe { _mm256_loadu_si256(read_cols.as_ptr().add(i * LANES) as *const __m256i) };

        for j in 1..=r_len {
            let ref_v =
                unsafe { _mm256_loadu_si256(ref_cols.as_ptr().add(j * LANES) as *const __m256i) };

            let eq = _mm256_cmpeq_epi8(read_v, ref_v);
            // eq is 0xFF for match, 0x00 for mismatch — blendv selects v_match
            // when the MSB of the selector byte is 1.
            let score_vec = _mm256_blendv_epi8(v_mism, v_match, eq);

            let h_diag = unsafe { *prev_h.get_unchecked(j - 1) };
            let h_match = _mm256_adds_epi8(h_diag, score_vec);

            let e_from_h = _mm256_adds_epi8(unsafe { *prev_h.get_unchecked(j) }, v_go);
            let e_from_e = _mm256_adds_epi8(unsafe { *prev_e.get_unchecked(j) }, v_ge);
            let e = _mm256_max_epi8(e_from_h, e_from_e);

            let f_from_h = _mm256_adds_epi8(unsafe { *cur_h.get_unchecked(j - 1) }, v_go);
            let f_from_f = _mm256_adds_epi8(cur_f, v_ge);
            let f = _mm256_max_epi8(f_from_h, f_from_f);

            let h_tmp = _mm256_max_epi8(h_match, e);
            let h_tmp = _mm256_max_epi8(h_tmp, f);
            let h = _mm256_max_epi8(h_tmp, v_zero);

            unsafe {
                *cur_h.get_unchecked_mut(j) = h;
                *cur_e.get_unchecked_mut(j) = e;
            }
            cur_f = f;

            // Traceback codes.
            let is_zero = _mm256_cmpeq_epi8(h, v_zero);
            let h_eq_match = _mm256_cmpeq_epi8(h, h_match);
            let h_eq_e = _mm256_cmpeq_epi8(h, e);
            let is_match = _mm256_andnot_si256(is_zero, h_eq_match);
            let is_e = _mm256_andnot_si256(is_zero, _mm256_andnot_si256(h_eq_match, h_eq_e));

            let tr = _mm256_blendv_epi8(v_three, v_two, is_e);
            let tr = _mm256_blendv_epi8(tr, v_one, is_match);
            let tr = _mm256_blendv_epi8(tr, v_zero, is_zero);

            unsafe {
                let dst = trace.as_mut_ptr().add((i * trace_w + j) * LANES) as *mut __m256i;
                _mm256_storeu_si256(dst, tr);
            }

            // Best-score update. cmpgt sets 0xFF for lanes where h > best_v.
            let new_best = _mm256_cmpgt_epi8(h, best_v);
            best_v = _mm256_max_epi8(best_v, h);

            let mask = _mm256_movemask_epi8(new_best) as u32;
            if mask != 0 {
                let mut bits = mask;
                while bits != 0 {
                    let k = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    if k < lanes {
                        best_i[k] = i as i32;
                        best_j[k] = j as i32;
                    }
                }
            }

            if i == q_len {
                let new_qlen = _mm256_cmpgt_epi8(h, best_qlen_v);
                best_qlen_v = _mm256_max_epi8(best_qlen_v, h);
                let m = _mm256_movemask_epi8(new_qlen) as u32;
                if m != 0 {
                    let mut bits = m;
                    while bits != 0 {
                        let k = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        if k < lanes {
                            best_qlen_j[k] = j as i32;
                        }
                    }
                }
            }

            // Saturation watch: any cell at or above INT8_SAFE_MAX_H is suspect.
            let sat_now = _mm256_cmpgt_epi8(h, v_sat);
            sat_v = _mm256_or_si256(sat_v, sat_now);
        }

        let remaining = (q_len - i) as i32;
        unsafe {
            _mm256_storeu_si256(best_arr.as_mut_ptr() as *mut __m256i, best_v);
        }
        let mut all_done = true;
        for k in 0..lanes {
            if lane_done[k] {
                continue;
            }
            let abort = abort_scores[k];
            if abort > i32::MIN / 8 {
                let max_possible = best_arr[k] as i32 + remaining * cfg.match_score;
                if max_possible < abort {
                    lane_done[k] = true;
                }
            }
            if !lane_done[k] {
                all_done = false;
            }
        }
        if all_done {
            break;
        }

        std::mem::swap(prev_h, cur_h);
        std::mem::swap(prev_e, cur_e);
    }

    // Reduce vector state.
    let mut best_score_arr = [0i8; LANES];
    let mut best_qlen_score_arr = [0i8; LANES];
    let mut sat_arr = [0u8; LANES];
    unsafe {
        _mm256_storeu_si256(best_score_arr.as_mut_ptr() as *mut __m256i, best_v);
        _mm256_storeu_si256(best_qlen_score_arr.as_mut_ptr() as *mut __m256i, best_qlen_v);
        _mm256_storeu_si256(sat_arr.as_mut_ptr() as *mut __m256i, sat_v);
    }

    let clip_pen_i8 = cfg.clip_penalty.clamp(0, i8::MAX as i32) as i8;

    let mut results = Vec::with_capacity(lanes);
    for k in 0..lanes {
        // Saturated → fall back to scalar SW for this lane so we never return
        // a wrong score under heavy positive scoring drift.
        if sat_arr[k] != 0 {
            results.push(banded_sw_internal(
                inputs[k].read_seq,
                inputs[k].ref_window,
                0,
                cfg,
                inputs[k].abort_score,
            ));
            continue;
        }

        let local_score = best_score_arr[k];
        let qlen_score = best_qlen_score_arr[k];
        let (start_i, start_j, bs) = if qlen_score > 0
            && qlen_score.saturating_add(clip_pen_i8) > local_score
        {
            (q_len as i32, best_qlen_j[k], qlen_score as i32)
        } else {
            (best_i[k], best_j[k], local_score as i32)
        };

        let mut cigar = Vec::new();
        let mut i = start_i;
        let mut j = start_j;
        let read_end = i;
        let ref_end = j as u32;

        while i > 0 && j > 0 {
            let idx = (i as usize * trace_w + j as usize) * LANES + k;
            let tr = unsafe { *trace.get_unchecked(idx) };
            if tr == 0 {
                break;
            }
            match tr {
                1 => {
                    push_cigar(&mut cigar, CigarKind::Match, 1);
                    i -= 1;
                    j -= 1;
                }
                2 => {
                    push_cigar(&mut cigar, CigarKind::Ins, 1);
                    i -= 1;
                }
                3 => {
                    push_cigar(&mut cigar, CigarKind::Del, 1);
                    j -= 1;
                }
                _ => break,
            }
        }

        cigar.reverse();
        results.push(SwResult {
            ref_start: j as u32,
            ref_end,
            read_start: i,
            read_end,
            score: bs,
            cigar,
            early_abort: lane_done[k],
        });
    }

    results
}

#[cfg(test)]
#[path = "../../tests/unit/alignment_sw_int8_vnni.rs"]
mod tests;
