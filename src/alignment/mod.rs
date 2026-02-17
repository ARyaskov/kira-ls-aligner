pub mod prefilter;

use crate::seq::reverse_complement_into;
use crate::simd::{self, SimdMode};
use crate::types::{Alignment, AlignmentKind, CigarKind, CigarOp, ReadRecord, Strand};

/// Alignment scoring configuration.
#[derive(Clone, Copy, Debug)]
pub struct AlignmentConfig {
    pub match_score: i32,
    pub mismatch: i32,
    pub gap_open: i32,
    pub gap_extend: i32,
    pub bandwidth: i32,
    pub xdrop: i32,
}

/// Summary span for alignment stage.
#[derive(Clone, Copy, Debug)]
pub struct AnchorSpan {
    pub ref_id: u32,
    pub ref_start: u32,
    pub ref_end: u32,
    pub read_start: u32,
    pub read_end: u32,
    pub strand: Strand,
}

/// Batched alignment input (short-read SIMD path).
#[derive(Clone, Debug)]
pub struct BatchInput<'a> {
    pub read_seq: &'a [u8],
    pub ref_window: &'a [u8],
    pub win_start: u32,
    pub chain: AnchorSpan,
    pub is_rev: bool,
    pub abort_score: i32,
}

/// Zero-copy read orientation adapter.
#[derive(Clone, Copy, Debug)]
pub struct OrientedRead<'a> {
    seq: &'a [u8],
    strand: Strand,
}

impl<'a> OrientedRead<'a> {
    pub fn new(seq: &'a [u8], strand: Strand) -> Self {
        Self { seq, strand }
    }

    pub fn is_rev(self) -> bool {
        self.strand == Strand::Reverse
    }

    pub fn contiguous<'b>(&'b self, scratch: &'b mut Vec<u8>) -> &'b [u8] {
        if self.strand == Strand::Reverse {
            reverse_complement_into(self.seq, scratch)
        } else {
            self.seq
        }
    }
}

/// Orient a read sequence for a given strand.
pub fn oriented_read(read: &ReadRecord, strand: Strand) -> OrientedRead<'_> {
    OrientedRead::new(read.seq.as_slice(), strand)
}

/// Attempt fast exact-match alignment (no DP).
pub fn exact_match_alignment(
    read_len: usize,
    read_seq: &[u8],
    ref_seq: &[u8],
    chain: &AnchorSpan,
    cfg: AlignmentConfig,
    is_rev: bool,
) -> Option<Alignment> {
    if chain.read_start != 0 || chain.read_end as usize != read_len {
        return None;
    }
    let ref_start = chain.ref_start as usize;
    if ref_start + read_len > ref_seq.len() {
        return None;
    }
    if chain.ref_end - chain.ref_start != read_len as u32 {
        return None;
    }
    let ref_slice = &ref_seq[ref_start..ref_start + read_len];
    if simd::count_mismatches(read_seq, ref_slice) != 0 {
        return None;
    }
    let cigar = vec![CigarOp {
        len: read_len as u32,
        op: CigarKind::Match,
    }];
    let score = cfg.match_score * read_len as i32;
    Some(Alignment {
        kind: AlignmentKind::AcceptedUngapped,
        ref_id: chain.ref_id,
        ref_start: chain.ref_start,
        ref_end: chain.ref_start + read_len as u32,
        read_start: 0,
        read_end: read_len as u32,
        cigar,
        score,
        mapq: 0,
        is_rev,
        is_secondary: false,
        is_supplementary: false,
        nm: 0,
        md: read_len.to_string(),
        as_score: score,
        xs_score: None,
    })
}

/// Align a chain using banded Smith-Waterman around the anchor span.
pub fn align_chain_with_meta(
    read: &ReadRecord,
    ref_seq: &[u8],
    chain: &AnchorSpan,
    cfg: AlignmentConfig,
    abort_score: i32,
) -> (Alignment, bool) {
    let oriented = oriented_read(read, chain.strand);
    let is_rev = oriented.is_rev();
    let mut scratch = Vec::new();
    let read_seq = oriented.contiguous(&mut scratch);
    let (aln, early, _) =
        align_oriented_chain_with_meta(read_seq, is_rev, ref_seq, chain, cfg, abort_score);
    (aln, early)
}

pub fn align_oriented_chain_with_meta(
    read_seq: &[u8],
    is_rev: bool,
    ref_seq: &[u8],
    chain: &AnchorSpan,
    cfg: AlignmentConfig,
    abort_score: i32,
) -> (Alignment, bool, usize) {
    let read_len = read_seq.len();

    if let Some(aln) = exact_match_alignment(read_len, read_seq, ref_seq, chain, cfg, is_rev) {
        return (aln, false, 0);
    }

    let (win_start, win_end) =
        clamp_window(ref_seq.len(), chain.ref_start, chain.ref_end, cfg.bandwidth);
    let ref_window = &ref_seq[win_start as usize..win_end as usize];
    let offset = chain.ref_start as i32 - win_start as i32 - chain.read_start as i32;

    let sw = banded_sw(read_seq, ref_window, offset, cfg, abort_score);
    let early = sw.early_abort;
    (
        build_alignment(read_seq, ref_window, win_start, chain, is_rev, sw),
        early,
        0,
    )
}

/// Align a chain using banded Smith-Waterman around the anchor span.
pub fn align_chain(
    read: &ReadRecord,
    ref_seq: &[u8],
    chain: &AnchorSpan,
    cfg: AlignmentConfig,
    abort_score: i32,
) -> Alignment {
    align_chain_with_meta(read, ref_seq, chain, cfg, abort_score).0
}

/// Align a batch of short-read inputs with SIMD where possible.
pub fn align_batch_simd(
    inputs: &[BatchInput<'_>],
    cfg: AlignmentConfig,
    mode: SimdMode,
) -> Vec<(Alignment, bool)> {
    if inputs.is_empty() {
        return Vec::new();
    }
    let read_len = inputs[0].read_seq.len();
    let ref_len = inputs[0].ref_window.len();
    if inputs
        .iter()
        .any(|i| i.read_seq.len() != read_len || i.ref_window.len() != ref_len)
    {
        return inputs
            .iter()
            .map(|i| align_chain_from_window_with_meta(i, cfg))
            .collect();
    }

    let sw_results: Vec<SwResult> = match mode {
        #[cfg(target_arch = "x86_64")]
        SimdMode::Avx2 => unsafe { sw_batch_avx2(inputs, cfg) },
        #[cfg(target_arch = "aarch64")]
        SimdMode::Neon => unsafe { sw_batch_neon(inputs, cfg) },
        _ => inputs
            .iter()
            .map(|i| banded_sw(i.read_seq, i.ref_window, 0, cfg, i.abort_score))
            .collect(),
    };

    sw_results
        .into_iter()
        .zip(inputs.iter())
        .map(|(sw, input)| {
            let early = sw.early_abort;
            (
                build_alignment(
                    input.read_seq,
                    input.ref_window,
                    input.win_start,
                    &input.chain,
                    input.is_rev,
                    sw,
                ),
                early,
            )
        })
        .collect()
}

fn align_chain_from_window_with_meta(
    input: &BatchInput<'_>,
    cfg: AlignmentConfig,
) -> (Alignment, bool) {
    let sw = banded_sw(input.read_seq, input.ref_window, 0, cfg, input.abort_score);
    let early = sw.early_abort;
    (
        build_alignment(
            input.read_seq,
            input.ref_window,
            input.win_start,
            &input.chain,
            input.is_rev,
            sw,
        ),
        early,
    )
}

struct SwResult {
    ref_start: u32,
    ref_end: u32,
    read_start: i32,
    read_end: i32,
    score: i32,
    cigar: Vec<CigarOp>,
    early_abort: bool,
}

fn build_alignment(
    read_seq: &[u8],
    ref_window: &[u8],
    win_start: u32,
    chain: &AnchorSpan,
    is_rev: bool,
    sw: SwResult,
) -> Alignment {
    let read_len = read_seq.len();
    let mut cigar = sw.cigar;

    if sw.read_start > 0 {
        cigar.insert(
            0,
            CigarOp {
                len: sw.read_start as u32,
                op: CigarKind::SoftClip,
            },
        );
    }
    if sw.read_end < read_len as i32 {
        cigar.push(CigarOp {
            len: (read_len as i32 - sw.read_end) as u32,
            op: CigarKind::SoftClip,
        });
    }

    let ref_start = win_start + sw.ref_start as u32;
    let ref_end = win_start + sw.ref_end as u32;

    let (nm, md) = compute_nm_md(
        read_seq,
        ref_window,
        sw.read_start as usize,
        sw.ref_start as usize,
        &cigar,
    );

    Alignment {
        kind: AlignmentKind::DpAligned,
        ref_id: chain.ref_id,
        ref_start,
        ref_end,
        read_start: sw.read_start as u32,
        read_end: sw.read_end as u32,
        cigar,
        score: sw.score,
        mapq: 0,
        is_rev,
        is_secondary: false,
        is_supplementary: false,
        nm,
        md,
        as_score: sw.score,
        xs_score: None,
    }
}

fn banded_sw(
    read: &[u8],
    reference: &[u8],
    offset: i32,
    cfg: AlignmentConfig,
    abort_score: i32,
) -> SwResult {
    let q_len = read.len();
    let r_len = reference.len();
    let band = cfg.bandwidth.max(1);

    let mut prev_h: Vec<i32> = Vec::new();
    let mut prev_e: Vec<i32> = Vec::new();
    let mut prev_start = 1i32;

    let mut trace_rows: Vec<Vec<u8>> = vec![Vec::new(); q_len + 1];
    let mut row_starts: Vec<i32> = vec![1i32; q_len + 1];

    let mut best_score = 0;
    let mut best_i = 0usize;
    let mut best_j = 0usize;

    let mut early_abort = false;

    for i in 1..=q_len {
        let center = i as i32 + offset;
        let j_start = (center - band).max(1);
        let j_end = (center + band).min(r_len as i32);
        if j_start > j_end {
            row_starts[i] = 1;
            trace_rows[i] = Vec::new();
            prev_h = Vec::new();
            prev_e = Vec::new();
            prev_start = 1;
            continue;
        }
        let row_len = (j_end - j_start + 1) as usize;
        row_starts[i] = j_start;

        let mut cur_h = vec![0i32; row_len];
        let mut cur_e = vec![i32::MIN / 4; row_len];
        let mut cur_f = vec![i32::MIN / 4; row_len];
        let mut trace = vec![0u8; row_len];
        let mut row_best = 0i32;

        for j in j_start..=j_end {
            let idx = (j - j_start) as usize;
            let (h_diag, score_diag) =
                if let Some((h, s)) = prev_diag(i, j, prev_start, &prev_h, read, reference, cfg) {
                    (h, s)
                } else {
                    (0, 0)
                };
            let h_match = h_diag + score_diag;

            let e = prev_cell(j, prev_start, &prev_h)
                .map(|v| {
                    (v - cfg.gap_open).max(
                        prev_cell(j, prev_start, &prev_e).unwrap_or(i32::MIN / 4) - cfg.gap_extend,
                    )
                })
                .unwrap_or(i32::MIN / 4);

            let f = if idx > 0 {
                (cur_h[idx - 1] - cfg.gap_open).max(cur_f[idx - 1] - cfg.gap_extend)
            } else {
                i32::MIN / 4
            };
            cur_e[idx] = e;
            cur_f[idx] = f;

            let mut h = 0;
            let mut tr = 0u8;
            if h_match >= e && h_match >= f && h_match > 0 {
                h = h_match;
                tr = 1;
            } else if e >= f && e > 0 {
                h = e;
                tr = 2;
            } else if f > 0 {
                h = f;
                tr = 3;
            }

            cur_h[idx] = h;
            trace[idx] = tr;
            row_best = row_best.max(h);

            if h > best_score {
                best_score = h;
                best_i = i;
                best_j = j as usize;
            }
        }

        trace_rows[i] = trace;
        prev_h = cur_h;
        prev_e = cur_e;
        prev_start = j_start;

        if cfg.xdrop > 0 && best_score - row_best > cfg.xdrop {
            early_abort = true;
            break;
        }
        if abort_score > i32::MIN / 8 {
            let remaining = (q_len - i) as i32;
            let max_possible = best_score + remaining * cfg.match_score;
            if max_possible < abort_score {
                early_abort = true;
                break;
            }
        }
    }

    let mut cigar = Vec::new();
    let mut i = best_i as i32;
    let mut j = best_j as i32;
    let read_end = i;
    let ref_end = j as u32;

    while i > 0 && j > 0 {
        let row_start = row_starts[i as usize];
        let idx = (j - row_start) as usize;
        if idx >= trace_rows[i as usize].len() {
            break;
        }
        let tr = trace_rows[i as usize][idx];
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
    SwResult {
        ref_start: j as u32,
        ref_end,
        read_start: i,
        read_end,
        score: best_score,
        cigar,
        early_abort,
    }
}

fn prev_cell(j: i32, prev_start: i32, row: &[i32]) -> Option<i32> {
    let idx = j - prev_start;
    if idx < 0 || idx as usize >= row.len() {
        None
    } else {
        Some(row[idx as usize])
    }
}

fn prev_diag(
    i: usize,
    j: i32,
    prev_start: i32,
    prev_h: &[i32],
    read: &[u8],
    reference: &[u8],
    cfg: AlignmentConfig,
) -> Option<(i32, i32)> {
    let idx = j - 1 - prev_start;
    if idx < 0 || idx as usize >= prev_h.len() {
        return None;
    }
    let h = prev_h[idx as usize];
    let qb = read[i - 1];
    let rb = reference[(j - 1) as usize];
    let score = if qb == rb {
        cfg.match_score
    } else {
        -cfg.mismatch
    };
    Some((h, score))
}

fn push_cigar(cigar: &mut Vec<CigarOp>, op: CigarKind, len: u32) {
    if let Some(last) = cigar.last_mut() {
        if last.op == op {
            last.len += len;
            return;
        }
    }
    cigar.push(CigarOp { len, op });
}

fn clamp_window(ref_len: usize, ref_start: u32, ref_end: u32, bandwidth: i32) -> (u32, u32) {
    let flank = (bandwidth.max(1) as u32).saturating_mul(2).max(50);
    let start = ref_start.saturating_sub(flank);
    let end = (ref_end + flank).min(ref_len as u32);
    (start, end.max(start + 1))
}

fn compute_nm_md(
    read: &[u8],
    reference: &[u8],
    read_start: usize,
    ref_start: usize,
    cigar: &[CigarOp],
) -> (u32, String) {
    let mut nm = 0u32;
    let mut md = String::new();
    let mut match_count = 0u32;
    let mut qpos = read_start;
    let mut rpos = ref_start;

    for op in cigar {
        match op.op {
            CigarKind::Match => {
                for _ in 0..op.len {
                    let qb = read.get(qpos).copied().unwrap_or(b'N');
                    let rb = reference.get(rpos).copied().unwrap_or(b'N');
                    if qb == rb {
                        match_count += 1;
                    } else {
                        nm += 1;
                        md.push_str(&match_count.to_string());
                        md.push(rb as char);
                        match_count = 0;
                    }
                    qpos += 1;
                    rpos += 1;
                }
            }
            CigarKind::Ins => {
                nm += op.len;
                qpos += op.len as usize;
            }
            CigarKind::Del => {
                nm += op.len;
                md.push_str(&match_count.to_string());
                md.push('^');
                for _ in 0..op.len {
                    let rb = reference.get(rpos).copied().unwrap_or(b'N');
                    md.push(rb as char);
                    rpos += 1;
                }
                match_count = 0;
            }
            CigarKind::SoftClip => {
                qpos += op.len as usize;
            }
        }
    }
    md.push_str(&match_count.to_string());
    (nm, md)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sw_batch_avx2(inputs: &[BatchInput<'_>], cfg: AlignmentConfig) -> Vec<SwResult> {
    use std::arch::x86_64::{
        __m256i, _mm256_add_epi32, _mm256_loadu_si256, _mm256_max_epi32, _mm256_set1_epi32,
        _mm256_storeu_si256,
    };

    let lanes = inputs.len();
    let q_len = inputs[0].read_seq.len();
    let r_len = inputs[0].ref_window.len();

    let neg_inf = i32::MIN / 4;
    let v_zero = _mm256_set1_epi32(0);
    let v_go = _mm256_set1_epi32(-cfg.gap_open);
    let v_ge = _mm256_set1_epi32(-cfg.gap_extend);

    let mut prev_h: Vec<__m256i> = vec![v_zero; r_len + 1];
    let mut prev_e: Vec<__m256i> = vec![_mm256_set1_epi32(neg_inf); r_len + 1];
    let mut cur_h: Vec<__m256i> = vec![v_zero; r_len + 1];
    let mut cur_e: Vec<__m256i> = vec![_mm256_set1_epi32(neg_inf); r_len + 1];

    let mut trace: Vec<Vec<u8>> = vec![vec![0u8; (q_len + 1) * (r_len + 1)]; lanes];
    let mut best_score = vec![0i32; lanes];
    let mut best_i = vec![0usize; lanes];
    let mut best_j = vec![0usize; lanes];
    let abort_scores: Vec<i32> = inputs.iter().map(|i| i.abort_score).collect();
    let mut lane_done = vec![false; lanes];

    let mut h_buf = [0i32; 8];
    let mut hm_buf = [0i32; 8];
    let mut e_buf = [0i32; 8];
    let mut f_buf = [0i32; 8];

    for i in 1..=q_len {
        cur_h[0] = v_zero;
        cur_e[0] = _mm256_set1_epi32(neg_inf);
        let mut cur_f = _mm256_set1_epi32(neg_inf);

        let mut read_row = [0u8; 8];
        for lane in 0..lanes {
            read_row[lane] = inputs[lane].read_seq[i - 1];
        }

        for j in 1..=r_len {
            let mut ref_col = [0u8; 8];
            for lane in 0..lanes {
                ref_col[lane] = inputs[lane].ref_window[j - 1];
            }
            let mut score_arr = [0i32; 8];
            for lane in 0..lanes {
                score_arr[lane] = if read_row[lane] == ref_col[lane] {
                    cfg.match_score
                } else {
                    -cfg.mismatch
                };
            }
            let score_vec = unsafe { _mm256_loadu_si256(score_arr.as_ptr() as *const __m256i) };

            let h_diag = prev_h[j - 1];
            let h_match = _mm256_add_epi32(h_diag, score_vec);

            let e_from_h = _mm256_add_epi32(prev_h[j], v_go);
            let e_from_e = _mm256_add_epi32(prev_e[j], v_ge);
            let e = _mm256_max_epi32(e_from_h, e_from_e);

            let f_from_h = _mm256_add_epi32(cur_h[j - 1], v_go);
            let f_from_f = _mm256_add_epi32(cur_f, v_ge);
            let f = _mm256_max_epi32(f_from_h, f_from_f);

            let mut h = _mm256_max_epi32(h_match, e);
            h = _mm256_max_epi32(h, f);
            h = _mm256_max_epi32(h, v_zero);

            cur_h[j] = h;
            cur_e[j] = e;
            cur_f = f;

            unsafe { _mm256_storeu_si256(h_buf.as_mut_ptr() as *mut __m256i, h) };
            unsafe { _mm256_storeu_si256(hm_buf.as_mut_ptr() as *mut __m256i, h_match) };
            unsafe { _mm256_storeu_si256(e_buf.as_mut_ptr() as *mut __m256i, e) };
            unsafe { _mm256_storeu_si256(f_buf.as_mut_ptr() as *mut __m256i, f) };

            for lane in 0..lanes {
                let idx = i * (r_len + 1) + j;
                let hval = h_buf[lane];
                if hval == 0 {
                    trace[lane][idx] = 0;
                } else if hval == hm_buf[lane] {
                    trace[lane][idx] = 1;
                } else if hval == e_buf[lane] {
                    trace[lane][idx] = 2;
                } else {
                    trace[lane][idx] = 3;
                }
                if hval > best_score[lane] {
                    best_score[lane] = hval;
                    best_i[lane] = i;
                    best_j[lane] = j;
                }
            }
        }

        let remaining = (q_len - i) as i32;
        let mut all_done = true;
        for lane in 0..lanes {
            if lane_done[lane] {
                continue;
            }
            let abort = abort_scores[lane];
            if abort > i32::MIN / 8 {
                let max_possible = best_score[lane] + remaining * cfg.match_score;
                if max_possible < abort {
                    lane_done[lane] = true;
                }
            }
            if !lane_done[lane] {
                all_done = false;
            }
        }
        if all_done {
            break;
        }

        std::mem::swap(&mut prev_h, &mut cur_h);
        std::mem::swap(&mut prev_e, &mut cur_e);
    }

    let mut results = Vec::with_capacity(lanes);
    for lane in 0..lanes {
        let mut cigar = Vec::new();
        let mut i = best_i[lane] as i32;
        let mut j = best_j[lane] as i32;
        let read_end = i;
        let ref_end = j as u32;

        while i > 0 && j > 0 {
            let idx = i as usize * (r_len + 1) + j as usize;
            let tr = trace[lane][idx];
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
            score: best_score[lane],
            cigar,
            early_abort: lane_done[lane],
        });
    }

    results
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sw_batch_neon(inputs: &[BatchInput<'_>], cfg: AlignmentConfig) -> Vec<SwResult> {
    use std::arch::aarch64::{int32x4_t, vaddq_s32, vld1q_s32, vmaxq_s32, vst1q_s32};

    fn splat(val: i32) -> int32x4_t {
        let arr = [val; 4];
        unsafe { vld1q_s32(arr.as_ptr()) }
    }

    let lanes = inputs.len();
    let q_len = inputs[0].read_seq.len();
    let r_len = inputs[0].ref_window.len();

    let neg_inf = i32::MIN / 4;
    let v_zero = splat(0);
    let v_neg = splat(neg_inf);
    let v_go = splat(-cfg.gap_open);
    let v_ge = splat(-cfg.gap_extend);

    let mut prev_h: Vec<int32x4_t> = vec![v_zero; r_len + 1];
    let mut prev_e: Vec<int32x4_t> = vec![v_neg; r_len + 1];
    let mut cur_h: Vec<int32x4_t> = vec![v_zero; r_len + 1];
    let mut cur_e: Vec<int32x4_t> = vec![v_neg; r_len + 1];

    let mut trace: Vec<Vec<u8>> = vec![vec![0u8; (q_len + 1) * (r_len + 1)]; lanes];
    let mut best_score = vec![0i32; lanes];
    let mut best_i = vec![0usize; lanes];
    let mut best_j = vec![0usize; lanes];
    let abort_scores: Vec<i32> = inputs.iter().map(|i| i.abort_score).collect();
    let mut lane_done = vec![false; lanes];

    let mut h_buf = [0i32; 4];
    let mut hm_buf = [0i32; 4];
    let mut e_buf = [0i32; 4];
    let mut f_buf = [0i32; 4];

    for i in 1..=q_len {
        cur_h[0] = v_zero;
        cur_e[0] = v_neg;
        let mut cur_f = v_neg;

        let mut read_row = [0u8; 4];
        for lane in 0..lanes {
            read_row[lane] = inputs[lane].read_seq[i - 1];
        }

        for j in 1..=r_len {
            let mut ref_col = [0u8; 4];
            for lane in 0..lanes {
                ref_col[lane] = inputs[lane].ref_window[j - 1];
            }
            let mut score_arr = [0i32; 4];
            for lane in 0..lanes {
                score_arr[lane] = if read_row[lane] == ref_col[lane] {
                    cfg.match_score
                } else {
                    -cfg.mismatch
                };
            }
            let score_vec = vld1q_s32(score_arr.as_ptr());

            let h_diag = prev_h[j - 1];
            let h_match = vaddq_s32(h_diag, score_vec);

            let e_from_h = vaddq_s32(prev_h[j], v_go);
            let e_from_e = vaddq_s32(prev_e[j], v_ge);
            let e = vmaxq_s32(e_from_h, e_from_e);

            let f_from_h = vaddq_s32(cur_h[j - 1], v_go);
            let f_from_f = vaddq_s32(cur_f, v_ge);
            let f = vmaxq_s32(f_from_h, f_from_f);

            let mut h = vmaxq_s32(h_match, e);
            h = vmaxq_s32(h, f);
            h = vmaxq_s32(h, v_zero);

            cur_h[j] = h;
            cur_e[j] = e;
            cur_f = f;

            vst1q_s32(h_buf.as_mut_ptr(), h);
            vst1q_s32(hm_buf.as_mut_ptr(), h_match);
            vst1q_s32(e_buf.as_mut_ptr(), e);
            vst1q_s32(f_buf.as_mut_ptr(), f);

            for lane in 0..lanes {
                let idx = i * (r_len + 1) + j;
                let hval = h_buf[lane];
                if hval == 0 {
                    trace[lane][idx] = 0;
                } else if hval == hm_buf[lane] {
                    trace[lane][idx] = 1;
                } else if hval == e_buf[lane] {
                    trace[lane][idx] = 2;
                } else {
                    trace[lane][idx] = 3;
                }
                if hval > best_score[lane] {
                    best_score[lane] = hval;
                    best_i[lane] = i;
                    best_j[lane] = j;
                }
            }
        }

        let remaining = (q_len - i) as i32;
        let mut all_done = true;
        for lane in 0..lanes {
            if lane_done[lane] {
                continue;
            }
            let abort = abort_scores[lane];
            if abort > i32::MIN / 8 {
                let max_possible = best_score[lane] + remaining * cfg.match_score;
                if max_possible < abort {
                    lane_done[lane] = true;
                }
            }
            if !lane_done[lane] {
                all_done = false;
            }
        }
        if all_done {
            break;
        }

        std::mem::swap(&mut prev_h, &mut cur_h);
        std::mem::swap(&mut prev_e, &mut cur_e);
    }

    let mut results = Vec::with_capacity(lanes);
    for lane in 0..lanes {
        let mut cigar = Vec::new();
        let mut i = best_i[lane] as i32;
        let mut j = best_j[lane] as i32;
        let read_end = i;
        let ref_end = j as u32;

        while i > 0 && j > 0 {
            let idx = i as usize * (r_len + 1) + j as usize;
            let tr = trace[lane][idx];
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
            score: best_score[lane],
            cigar,
            early_abort: lane_done[lane],
        });
    }

    results
}
