//! Wavefront Alignment (WFA) — affine-gap, semi-global.

use crate::types::{CigarKind, CigarOp};

/// Sentinel for "no offset reached on this diagonal".
const NONE_OFFSET: i32 = i32::MIN / 4;

/// Affine-gap penalties as positive WFA costs.
#[derive(Clone, Copy, Debug)]
pub struct WfaPenalties {
    /// Cost of a single mismatch.
    pub mismatch: i32,
    /// Cost of opening a gap. The first extend step adds `extend` on top.
    pub gap_open: i32,
    /// Cost of every extend (including the one paired with `open`).
    pub gap_extend: i32,
}

/// WFA2-style alignment options.
#[derive(Clone, Copy, Debug)]
pub struct WfaOptions {
    /// Maximum total cost to explore before giving up (`None` ⇒ caller budget).
    pub max_score: i32,
    /// Adaptive wavefront pruning (WFA-adaptive heuristic). `Some(drop)` discards
    /// diagonals lagging more than `drop` antidiagonals behind the furthest-
    /// reaching point each step, bounding both the wavefront width (→ O(s·drop)
    /// memory instead of O(s²)) and runtime on divergent/noisy reads. `None` is
    /// the exact, unpruned algorithm. Heuristic: a `Some` result may be
    /// cost ≥ the true optimum if the optimal path dipped below the drop band.
    pub adaptive_drop: Option<i32>,
    /// Ends-free: number of leading TEXT bases that may be skipped for free
    /// (a free 5' gap on the reference). `0` reproduces the original
    /// query-global / text-prefix semantics where the read must start at
    /// `text[0]`. The trailing text gap is always free (semi-global).
    pub text_begin_free: i32,
}

impl WfaOptions {
    /// Exact (unpruned) semi-global options with the given cost ceiling — the
    /// historical `wfa_align_semi_global` behaviour.
    #[inline]
    pub fn exact(max_score: i32) -> Self {
        Self {
            max_score,
            adaptive_drop: None,
            text_begin_free: 0,
        }
    }
}

/// Final alignment output.
#[derive(Clone, Debug)]
pub struct WfaAlignment {
    /// Total alignment cost (sum of mismatch + gap penalties).
    pub score: i32,
    /// CIGAR consuming the *entire* pattern, in 5'→3' order.
    pub cigar: Vec<CigarOp>,
    /// First text position consumed.
    pub text_start: usize,
    /// One past the last consumed text position.
    pub text_end: usize,
}

/// A wavefront at one (score, layer).
#[derive(Clone, Debug)]
struct WaveFront {
    lo: i32,
    hi: i32,
    offsets: Vec<i32>,
}

impl WaveFront {
    fn empty() -> Self {
        Self {
            lo: 0,
            hi: -1, // empty range
            offsets: Vec::new(),
        }
    }

    #[inline]
    fn get(&self, k: i32) -> i32 {
        if k < self.lo || k > self.hi {
            NONE_OFFSET
        } else {
            // Safe because k - lo is in 0..=hi-lo and offsets.len() == hi-lo+1.
            self.offsets[(k - self.lo) as usize]
        }
    }

    fn is_empty(&self) -> bool {
        self.hi < self.lo
    }
}

/// Per-score wavefront cache.
type History = Vec<WaveFront>;

fn ensure_score_capacity(hist: &mut History, score: usize) {
    if hist.len() <= score {
        hist.resize_with(score + 1, WaveFront::empty);
    }
}

/// Thread-local pool of `Vec<i32>` buffers for wavefront offsets.
#[derive(Default)]
struct WfaScratch {
    m_hist: History,
    i_hist: History,
    d_hist: History,
    pool: Vec<Vec<i32>>,
}

impl WfaScratch {
    fn reset(&mut self) {
        for wf in self.m_hist.drain(..) {
            self.pool.push(wf.offsets);
        }
        for wf in self.i_hist.drain(..) {
            self.pool.push(wf.offsets);
        }
        for wf in self.d_hist.drain(..) {
            self.pool.push(wf.offsets);
        }
    }
}

thread_local! {
    static WFA_SCRATCH: std::cell::RefCell<WfaScratch> =
        std::cell::RefCell::new(WfaScratch::default());
}

/// Greedy match extension along a single diagonal — scalar reference.
#[inline]
fn extend_diagonal_scalar(pattern: &[u8], text: &[u8], k: i32, mut offset: i32) -> i32 {
    let m = pattern.len() as i32;
    let n = text.len() as i32;
    loop {
        let i = offset;
        let j = offset + k;
        if i >= m || j < 0 || j >= n {
            break;
        }
        if pattern[i as usize] != text[j as usize] {
            break;
        }
        offset += 1;
    }
    offset
}

/// AVX2 fast-path: compare 32 bytes of pattern vs text at a time and jump directly to the first.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn extend_diagonal_avx2(pattern: &[u8], text: &[u8], k: i32, offset: i32) -> i32 {
    use std::arch::x86_64::{__m256i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8};

    let m = pattern.len() as i32;
    let n = text.len() as i32;
    let j0 = offset + k;
    if offset < 0 || j0 < 0 || offset >= m || j0 >= n {
        return extend_diagonal_scalar(pattern, text, k, offset);
    }

    let mut off = offset;
    // Bulk: 32-byte chunks while both pattern and text have a full chunk left.
    loop {
        let i = off as usize;
        let j = (off + k) as usize;
        // Need 32 bytes of headroom in both buffers.
        if i + 32 > pattern.len() || j + 32 > text.len() {
            break;
        }
        // SAFETY: bounded by the checks above.
        let pat_v = unsafe { _mm256_loadu_si256(pattern.as_ptr().add(i) as *const __m256i) };
        let txt_v = unsafe { _mm256_loadu_si256(text.as_ptr().add(j) as *const __m256i) };
        let eq = _mm256_cmpeq_epi8(pat_v, txt_v);
        let mask = _mm256_movemask_epi8(eq) as u32;
        let inv = !mask;
        if inv == 0 {
            // All 32 matched — move on.
            off += 32;
            continue;
        }
        off += inv.trailing_zeros() as i32;
        return off;
    }
    // Tail: scalar walk for the last <32 bytes.
    extend_diagonal_scalar(pattern, text, k, off)
}

#[inline]
fn extend_diagonal(pattern: &[u8], text: &[u8], k: i32, offset: i32) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime check guards the AVX2 implementation.
            return unsafe { extend_diagonal_avx2(pattern, text, k, offset) };
        }
    }
    extend_diagonal_scalar(pattern, text, k, offset)
}

/// Extend the M wavefront at score `s` greedily across matches, in place.
fn extend_wavefront(pattern: &[u8], text: &[u8], wf: &mut WaveFront) {
    for k in wf.lo..=wf.hi {
        let idx = (k - wf.lo) as usize;
        let off = wf.offsets[idx];
        if off == NONE_OFFSET {
            continue;
        }
        wf.offsets[idx] = extend_diagonal(pattern, text, k, off);
    }
}

/// Adaptive wavefront pruning (WFA-adaptive). Drops diagonals whose furthest
/// cell lags more than `drop` antidiagonals (`i + j`) behind the global maximum,
/// then compacts the offset range. Applied to the M wavefront after extension;
/// the furthest-reaching diagonal — which is what `check_done` keys on — has the
/// maximum antidiagonal and is therefore never pruned.
fn prune_wavefront(wf: &mut WaveFront, drop: i32) {
    if wf.is_empty() {
        return;
    }
    // antidiagonal of diagonal k at offset o: i + j = o + (o + k) = 2o + k.
    let mut max_ad = i32::MIN;
    for k in wf.lo..=wf.hi {
        let o = wf.offsets[(k - wf.lo) as usize];
        if o != NONE_OFFSET {
            max_ad = max_ad.max(2 * o + k);
        }
    }
    if max_ad == i32::MIN {
        return; // all-empty
    }
    let thresh = max_ad - drop;
    // Tighten the live range and null interior laggards.
    let mut new_lo = wf.lo;
    while new_lo <= wf.hi {
        let o = wf.offsets[(new_lo - wf.lo) as usize];
        if o != NONE_OFFSET && 2 * o + new_lo >= thresh {
            break;
        }
        new_lo += 1;
    }
    let mut new_hi = wf.hi;
    while new_hi >= new_lo {
        let o = wf.offsets[(new_hi - wf.lo) as usize];
        if o != NONE_OFFSET && 2 * o + new_hi >= thresh {
            break;
        }
        new_hi -= 1;
    }
    if new_lo > new_hi {
        *wf = WaveFront::empty();
        return;
    }
    for k in new_lo..=new_hi {
        let idx = (k - wf.lo) as usize;
        let o = wf.offsets[idx];
        if o != NONE_OFFSET && 2 * o + k < thresh {
            wf.offsets[idx] = NONE_OFFSET;
        }
    }
    if new_lo != wf.lo || new_hi != wf.hi {
        let start = (new_lo - wf.lo) as usize;
        let len = (new_hi - new_lo + 1) as usize;
        wf.offsets.copy_within(start..start + len, 0);
        wf.offsets.truncate(len);
        wf.lo = new_lo;
        wf.hi = new_hi;
    }
}

/// Check whether the pattern has been fully consumed on any diagonal in `wf`.
fn check_done(wf: &WaveFront, pattern_len: i32) -> Option<i32> {
    for k in wf.lo..=wf.hi {
        let off = wf.offsets[(k - wf.lo) as usize];
        if off >= pattern_len {
            return Some(k);
        }
    }
    None
}

/// Look up a historical wavefront; returns `None` if score is out of range or the slot is empty.
#[inline]
fn get_src(hist: &History, score: i32) -> Option<&WaveFront> {
    if score < 0 {
        return None;
    }
    let s = score as usize;
    if s >= hist.len() {
        return None;
    }
    let w = &hist[s];
    if w.is_empty() { None } else { Some(w) }
}

/// Compute the next M/I/D wavefronts at score `s` from history.
fn step(
    s: i32,
    pen: WfaPenalties,
    m_hist: &History,
    i_hist: &History,
    d_hist: &History,
    pool: &mut Vec<Vec<i32>>,
) -> (WaveFront, WaveFront, WaveFront) {
    let s_x = s - pen.mismatch;
    let s_oe = s - pen.gap_open - pen.gap_extend;
    let s_e = s - pen.gap_extend;

    let src_m_x = get_src(m_hist, s_x);
    let src_m_oe = get_src(m_hist, s_oe);
    let src_i_e = get_src(i_hist, s_e);
    let src_d_e = get_src(d_hist, s_e);

    // Determine new I range: from source diagonals shifted by -1 (k = k_src - 1).
    let mut i_lo = i32::MAX;
    let mut i_hi = i32::MIN;
    if let Some(w) = src_m_oe {
        i_lo = i_lo.min(w.lo - 1);
        i_hi = i_hi.max(w.hi - 1);
    }
    if let Some(w) = src_i_e {
        i_lo = i_lo.min(w.lo - 1);
        i_hi = i_hi.max(w.hi - 1);
    }
    // Determine new D range: from source diagonals shifted by +1 (k = k_src + 1).
    let mut d_lo = i32::MAX;
    let mut d_hi = i32::MIN;
    if let Some(w) = src_m_oe {
        d_lo = d_lo.min(w.lo + 1);
        d_hi = d_hi.max(w.hi + 1);
    }
    if let Some(w) = src_d_e {
        d_lo = d_lo.min(w.lo + 1);
        d_hi = d_hi.max(w.hi + 1);
    }
    // Determine new M range: from M[s-x] (same diagonals), I[s], D[s].
    let mut m_lo = i32::MAX;
    let mut m_hi = i32::MIN;
    if let Some(w) = src_m_x {
        m_lo = m_lo.min(w.lo);
        m_hi = m_hi.max(w.hi);
    }
    if i_lo <= i_hi {
        m_lo = m_lo.min(i_lo);
        m_hi = m_hi.max(i_hi);
    }
    if d_lo <= d_hi {
        m_lo = m_lo.min(d_lo);
        m_hi = m_hi.max(d_hi);
    }

    // Helper: pull a Vec<i32> from the pool, size it, fill with NONE_OFFSET.
    let mut acquire = |len: usize| -> Vec<i32> {
        let mut v = pool.pop().unwrap_or_default();
        v.clear();
        v.resize(len, NONE_OFFSET);
        v
    };

    // Fill I, D, then M.
    let mut new_i = if i_lo <= i_hi {
        WaveFront {
            lo: i_lo,
            hi: i_hi,
            offsets: acquire((i_hi - i_lo + 1) as usize),
        }
    } else {
        WaveFront::empty()
    };
    for k in new_i.lo..=new_i.hi {
        let mut best = NONE_OFFSET;
        if let Some(w) = src_m_oe {
            let v = w.get(k + 1);
            if v != NONE_OFFSET {
                best = best.max(v + 1);
            }
        }
        if let Some(w) = src_i_e {
            let v = w.get(k + 1);
            if v != NONE_OFFSET {
                best = best.max(v + 1);
            }
        }
        new_i.offsets[(k - new_i.lo) as usize] = best;
    }

    let mut new_d = if d_lo <= d_hi {
        WaveFront {
            lo: d_lo,
            hi: d_hi,
            offsets: acquire((d_hi - d_lo + 1) as usize),
        }
    } else {
        WaveFront::empty()
    };
    for k in new_d.lo..=new_d.hi {
        let mut best = NONE_OFFSET;
        if let Some(w) = src_m_oe {
            let v = w.get(k - 1);
            if v != NONE_OFFSET {
                best = best.max(v);
            }
        }
        if let Some(w) = src_d_e {
            let v = w.get(k - 1);
            if v != NONE_OFFSET {
                best = best.max(v);
            }
        }
        new_d.offsets[(k - new_d.lo) as usize] = best;
    }

    let mut new_m = if m_lo <= m_hi {
        WaveFront {
            lo: m_lo,
            hi: m_hi,
            offsets: acquire((m_hi - m_lo + 1) as usize),
        }
    } else {
        WaveFront::empty()
    };
    for k in new_m.lo..=new_m.hi {
        let mut best = NONE_OFFSET;
        if let Some(w) = src_m_x {
            let v = w.get(k);
            if v != NONE_OFFSET {
                best = best.max(v + 1);
            }
        }
        let vi = new_i.get(k);
        if vi != NONE_OFFSET {
            best = best.max(vi);
        }
        let vd = new_d.get(k);
        if vd != NONE_OFFSET {
            best = best.max(vd);
        }
        new_m.offsets[(k - new_m.lo) as usize] = best;
    }

    (new_m, new_i, new_d)
}

/// Align `pattern` to a prefix of `text` (semi-global) under affine-gap WFA.
///
/// Back-compat wrapper: exact (unpruned), read starts at `text[0]`.
#[inline]
pub fn wfa_align_semi_global(
    pattern: &[u8],
    text: &[u8],
    pen: WfaPenalties,
    max_score: i32,
) -> Option<WfaAlignment> {
    wfa_align(pattern, text, pen, WfaOptions::exact(max_score))
}

/// Align `pattern` against `text` under affine-gap WFA with WFA2 options
/// (adaptive pruning, ends-free leading text). The trailing text gap is always
/// free. Returns the lowest-cost alignment whose cost ≤ `opts.max_score`.
pub fn wfa_align(
    pattern: &[u8],
    text: &[u8],
    pen: WfaPenalties,
    opts: WfaOptions,
) -> Option<WfaAlignment> {
    let m = pattern.len() as i32;
    let n = text.len() as i32;
    if m == 0 {
        return Some(WfaAlignment {
            score: 0,
            cigar: Vec::new(),
            text_start: 0,
            text_end: 0,
        });
    }
    let max_score = opts.max_score;
    // Free leading-text diagonals 0..=tbf (clamped to the text length).
    let tbf = opts.text_begin_free.max(0).min((n - 1).max(0));

    WFA_SCRATCH.with(|scratch_cell| {
        let mut scratch = scratch_cell.borrow_mut();
        scratch.reset();

        let WfaScratch {
            m_hist,
            i_hist,
            d_hist,
            pool,
        } = &mut *scratch;

        let mut m0_offsets = pool.pop().unwrap_or_default();
        m0_offsets.clear();
        m0_offsets.resize((tbf + 1) as usize, 0);
        let mut m0 = WaveFront {
            lo: 0,
            hi: tbf,
            offsets: m0_offsets,
        };
        extend_wavefront(pattern, text, &mut m0);
        if let Some(drop) = opts.adaptive_drop {
            prune_wavefront(&mut m0, drop);
        }

        if let Some(k_end) = check_done(&m0, m) {
            // Store m0 so build_cigar can read it.
            ensure_score_capacity(m_hist, 0);
            m_hist[0] = m0;
            let (cigar, k_start) =
                build_cigar(pattern, text, m_hist, i_hist, d_hist, 0, k_end, pen);
            let end_text = m + k_end;
            return Some(WfaAlignment {
                score: 0,
                cigar,
                text_start: k_start.max(0) as usize,
                text_end: end_text as usize,
            });
        }
        ensure_score_capacity(m_hist, 0);
        m_hist[0] = m0;

        for s in 1..=max_score {
            let (mut new_m, new_i, new_d) = step(s, pen, m_hist, i_hist, d_hist, pool);
            extend_wavefront(pattern, text, &mut new_m);
            if let Some(drop) = opts.adaptive_drop {
                prune_wavefront(&mut new_m, drop);
            }

            if let Some(k_end) = check_done(&new_m, m) {
                ensure_score_capacity(m_hist, s as usize);
                ensure_score_capacity(i_hist, s as usize);
                ensure_score_capacity(d_hist, s as usize);
                m_hist[s as usize] = new_m;
                i_hist[s as usize] = new_i;
                d_hist[s as usize] = new_d;
                let end_text = m + k_end;
                let (cigar, k_start) =
                    build_cigar(pattern, text, m_hist, i_hist, d_hist, s, k_end, pen);
                return Some(WfaAlignment {
                    score: s,
                    cigar,
                    text_start: k_start.max(0) as usize,
                    text_end: end_text as usize,
                });
            }

            ensure_score_capacity(m_hist, s as usize);
            ensure_score_capacity(i_hist, s as usize);
            ensure_score_capacity(d_hist, s as usize);
            m_hist[s as usize] = new_m;
            i_hist[s as usize] = new_i;
            d_hist[s as usize] = new_d;
        }

        None
    })
}

/// Retire a score layer once it falls outside the recurrence look-back window,
/// recycling its offset buffer to the pool so peak memory stays bounded.
#[inline]
fn retire_layer(hist: &mut History, pool: &mut Vec<Vec<i32>>, s: usize) {
    if s < hist.len() {
        let wf = std::mem::replace(&mut hist[s], WaveFront::empty());
        if !wf.offsets.is_empty() {
            pool.push(wf.offsets);
        }
    }
}

/// Exact optimal semi-global WFA **cost** in O(width · penalty-depth) memory.
///
/// This is a forward, linear-history WFA engine: it runs the same M/I/D
/// recurrence as [`wfa_align`] but, because no traceback is needed,
/// it retires score layers as soon as they fall outside the recurrence
/// look-back window (`max(mismatch, gap_open+gap_extend)`), recycling their
/// offset buffers. Peak heavy memory is therefore bounded by a constant number
/// of wavefronts rather than the full O(s²) history, making it a cheap,
/// memory-flat acceptance/cost pre-check. Returns the optimal cost, or `None`
/// if no alignment exists within `opts.max_score`.
pub fn wfa_score_only(
    pattern: &[u8],
    text: &[u8],
    pen: WfaPenalties,
    opts: WfaOptions,
) -> Option<i32> {
    let m = pattern.len() as i32;
    let n = text.len() as i32;
    if m == 0 {
        return Some(0);
    }
    let max_score = opts.max_score;
    let tbf = opts.text_begin_free.max(0).min((n - 1).max(0));
    let lookback = pen
        .mismatch
        .max(pen.gap_open + pen.gap_extend)
        .max(pen.gap_extend);

    WFA_SCRATCH.with(|scratch_cell| {
        let mut scratch = scratch_cell.borrow_mut();
        scratch.reset();
        let WfaScratch {
            m_hist,
            i_hist,
            d_hist,
            pool,
        } = &mut *scratch;

        let mut m0_offsets = pool.pop().unwrap_or_default();
        m0_offsets.clear();
        m0_offsets.resize((tbf + 1) as usize, 0);
        let mut m0 = WaveFront {
            lo: 0,
            hi: tbf,
            offsets: m0_offsets,
        };
        extend_wavefront(pattern, text, &mut m0);
        if let Some(drop) = opts.adaptive_drop {
            prune_wavefront(&mut m0, drop);
        }
        if check_done(&m0, m).is_some() {
            return Some(0);
        }
        ensure_score_capacity(m_hist, 0);
        m_hist[0] = m0;

        for s in 1..=max_score {
            let (mut new_m, new_i, new_d) = step(s, pen, m_hist, i_hist, d_hist, pool);
            extend_wavefront(pattern, text, &mut new_m);
            if let Some(drop) = opts.adaptive_drop {
                prune_wavefront(&mut new_m, drop);
            }
            let done = check_done(&new_m, m).is_some();
            ensure_score_capacity(m_hist, s as usize);
            ensure_score_capacity(i_hist, s as usize);
            ensure_score_capacity(d_hist, s as usize);
            m_hist[s as usize] = new_m;
            i_hist[s as usize] = new_i;
            d_hist[s as usize] = new_d;
            if done {
                return Some(s);
            }
            // Retire the layer that just fell outside the look-back window.
            let retire = s - lookback - 1;
            if retire >= 0 {
                retire_layer(m_hist, pool, retire as usize);
                retire_layer(i_hist, pool, retire as usize);
                retire_layer(d_hist, pool, retire as usize);
            }
        }
        None
    })
}

/// Layer identifier for traceback state machine.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Layer {
    M,
    I,
    D,
}

/// Reconstruct CIGAR by walking the wavefront history backwards from.
#[allow(clippy::too_many_arguments)]
fn build_cigar(
    pattern: &[u8],
    text: &[u8],
    m_hist: &History,
    i_hist: &History,
    d_hist: &History,
    s_final: i32,
    k_final: i32,
    pen: WfaPenalties,
) -> (Vec<CigarOp>, i32) {
    let m_len = pattern.len() as i32;

    let mut ops: Vec<CigarOp> = Vec::new();
    let mut s = s_final;
    let mut k = k_final;
    let mut offset = m_len; // pattern position at termination
    let mut layer = Layer::M;

    let get_layer = |layer: Layer, score: i32, k: i32| -> i32 {
        let hist = match layer {
            Layer::M => m_hist,
            Layer::I => i_hist,
            Layer::D => d_hist,
        };
        if score < 0 || (score as usize) >= hist.len() {
            return NONE_OFFSET;
        }
        let w = &hist[score as usize];
        if w.is_empty() { NONE_OFFSET } else { w.get(k) }
    };

    // Helper to push a matching/diag run starting at offset, ending at target_offset.
    let push_diag = |ops: &mut Vec<CigarOp>, run_len: i32| {
        if run_len <= 0 {
            return;
        }
        push_cigar(ops, CigarKind::Match, run_len as u32);
    };

    while !(s == 0 && layer == Layer::M && offset == 0) {
        match layer {
            Layer::M => {
                let mut walk = offset;
                loop {
                    if walk == 0 {
                        break;
                    }
                    // Stop as soon as this cell is the landing point of a
                    // backward edge (gap closure or mismatch) at the current
                    // score, BEFORE consuming a diagonal match. The recurrence
                    // sets M[s][k] = max(M[s-x][k]+1, I[s][k], D[s][k]) and then
                    // greedily extends; the pre-extension offset (where an edge
                    // lands) is therefore ≤ the post-extension offset we start
                    // from. When extension added nothing, the edge lands exactly
                    // at `offset`, so it must be tested here first — the original
                    // code decremented through a coincidental match before
                    // checking, walking one cell too far and stranding the
                    // traceback with no resolvable edge (CIGAR under-consumed the
                    // read). When extension added ≥1 base, none of the sources
                    // equal the current cell, so these checks can't stop early.
                    // At s==0 the I/D layers are empty and M[s-x] is out of range,
                    // so the walk runs straight to the pattern start.
                    let i_off = get_layer(Layer::I, s, k);
                    if i_off == walk {
                        break;
                    }
                    let d_off = get_layer(Layer::D, s, k);
                    if d_off == walk {
                        break;
                    }
                    if s >= pen.mismatch {
                        let prev = get_layer(Layer::M, s - pen.mismatch, k);
                        if prev != NONE_OFFSET && prev + 1 == walk {
                            break;
                        }
                    }
                    // Otherwise walk down through one diagonal match.
                    let i = walk - 1;
                    let j = walk - 1 + k;
                    if i < 0 || j < 0 {
                        break;
                    }
                    if i >= pattern.len() as i32 || j >= text.len() as i32 {
                        // Can't have arrived here via match — offset out of range.
                        break;
                    }
                    if pattern[i as usize] != text[j as usize] {
                        break;
                    }
                    walk -= 1;
                }
                let diag_run = offset - walk;
                push_diag(&mut ops, diag_run);
                offset = walk;

                if s == 0 && offset == 0 {
                    break;
                }

                let mut took_edge = false;
                if s >= pen.mismatch {
                    let prev = get_layer(Layer::M, s - pen.mismatch, k);
                    if prev != NONE_OFFSET && prev + 1 == offset {
                        push_cigar(&mut ops, CigarKind::Match, 1); // mismatch shows up as M in CIGAR
                        offset -= 1;
                        s -= pen.mismatch;
                        took_edge = true;
                    }
                }
                if !took_edge {
                    let i_off = get_layer(Layer::I, s, k);
                    if i_off == offset {
                        layer = Layer::I;
                        took_edge = true;
                    }
                }
                if !took_edge {
                    let d_off = get_layer(Layer::D, s, k);
                    if d_off == offset {
                        layer = Layer::D;
                        took_edge = true;
                    }
                }
                if !took_edge {
                    // Shouldn't happen if WFA is correct.
                    break;
                }
            }
            Layer::I => {
                let mut took = false;
                if s >= pen.gap_open + pen.gap_extend {
                    let prev = get_layer(Layer::M, s - pen.gap_open - pen.gap_extend, k + 1);
                    if prev != NONE_OFFSET && prev + 1 == offset {
                        push_cigar(&mut ops, CigarKind::Ins, 1);
                        offset -= 1;
                        k += 1;
                        s -= pen.gap_open + pen.gap_extend;
                        layer = Layer::M;
                        took = true;
                    }
                }
                if !took && s >= pen.gap_extend {
                    let prev = get_layer(Layer::I, s - pen.gap_extend, k + 1);
                    if prev != NONE_OFFSET && prev + 1 == offset {
                        push_cigar(&mut ops, CigarKind::Ins, 1);
                        offset -= 1;
                        k += 1;
                        s -= pen.gap_extend;
                        // stay in I
                        took = true;
                    }
                }
                if !took {
                    break;
                }
            }
            Layer::D => {
                let mut took = false;
                if s >= pen.gap_open + pen.gap_extend {
                    let prev = get_layer(Layer::M, s - pen.gap_open - pen.gap_extend, k - 1);
                    if prev != NONE_OFFSET && prev == offset {
                        push_cigar(&mut ops, CigarKind::Del, 1);
                        // pattern position unchanged; text decreases by 1; k decreases
                        k -= 1;
                        s -= pen.gap_open + pen.gap_extend;
                        layer = Layer::M;
                        took = true;
                    }
                }
                if !took && s >= pen.gap_extend {
                    let prev = get_layer(Layer::D, s - pen.gap_extend, k - 1);
                    if prev != NONE_OFFSET && prev == offset {
                        push_cigar(&mut ops, CigarKind::Del, 1);
                        k -= 1;
                        s -= pen.gap_extend;
                        // stay in D
                        took = true;
                    }
                }
                if !took {
                    break;
                }
            }
        }
    }

    ops.reverse();
    // At termination `k` is the diagonal of the alignment start; with
    // `text_begin_free` seeding, the start text position is exactly this `k`.
    (ops, k)
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

#[cfg(test)]
mod underconsume_tests {
    use super::*;

    fn query_consumed(cigar: &[CigarOp]) -> u32 {
        cigar
            .iter()
            .map(|op| match op.op {
                CigarKind::Match | CigarKind::Ins | CigarKind::SoftClip => op.len,
                CigarKind::Del | CigarKind::Skipped => 0,
            })
            .sum()
    }

    // Regression for an early-terminating WFA traceback. Captured from
    // `cargo run -- mem ecoli.fa sub_50k.fastq` on tag v0.4.0: a 150bp read whose
    // optimal alignment closes an insertion gap exactly at the pattern end (no
    // greedy extension on the final M cell). The M-layer traceback used to consume
    // one coincidental match before testing the gap-close edge, walking past it and
    // emitting `[1M]` (query_consumed=1), which tripped the io/mod.rs
    // `consumed == seq_len` assertion. The CIGAR must span the whole read.
    #[test]
    fn wfa_cigar_spans_full_read() {
        let read = b"ATAGTCGAGCAGGTAATAACGCCCTTCGTGGCGGAACACCAGGTCGATAAAGCCTTTTAACATGCCACGTACCTGCATGAACTCCAGCGGCGGGCAGCCTGCGGATAGCGGGGCAAAATGGCGGATTAAAGTATAAAACCGCCTGGGGGT";
        let text = b"ATAGTCGAGCAGGTAATAACGCCCTTCGTGGCGGAACACCAGGTCGATAAAGCCTTTTAACATGCCACGTACCTGCATGAACTCCAGCGGCGGGCAGCCTGCGGATAGCGGGTCAAACTGGCGGATTAACGTATCAAGCTGACTGGCGATAAGCGGTTCACTAATCGGCAGATAAAACTCCATCTCCACCTGTTTATTGC";
        let pen = WfaPenalties {
            mismatch: 4,
            gap_open: 6,
            gap_extend: 1,
        };
        let opts = WfaOptions::exact(150);
        let aln = wfa_align(read, text, pen, opts).expect("wfa should align");

        assert_eq!(
            query_consumed(&aln.cigar),
            read.len() as u32,
            "WFA CIGAR must consume the entire pattern; got {:?}",
            aln.cigar
        );
        // The optimal path: 134 matches (3 mismatches inside) then the trailing
        // 16 read bases as an insertion. Score = 3*4 + (6 + 16*1) = 34.
        assert_eq!(
            aln.cigar,
            vec![
                CigarOp {
                    len: 134,
                    op: CigarKind::Match,
                },
                CigarOp {
                    len: 16,
                    op: CigarKind::Ins,
                },
            ],
        );
        assert_eq!(aln.score, 34);
        assert_eq!(aln.text_start, 0);
        assert_eq!(aln.text_end, 134);
    }

    /// Small deterministic PRNG (LCG) — no external rand dependency, fully
    /// reproducible across runs.
    struct Lcg(u64);
    impl Lcg {
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn base(&mut self) -> u8 {
            b"ACGT"[(self.next_u64() % 4) as usize]
        }
    }

    /// Replay a WFA CIGAR against (read, text), returning (cost, query_consumed,
    /// text_consumed_end). A correct WFA result must satisfy:
    ///   cost == score, query_consumed == read.len(), text_end == aln.text_end.
    fn replay(read: &[u8], text: &[u8], aln: &WfaAlignment, pen: WfaPenalties) -> (i32, usize, usize) {
        let mut qi = 0usize;
        let mut ti = aln.text_start;
        let mut cost = 0i32;
        for op in &aln.cigar {
            match op.op {
                CigarKind::Match => {
                    for _ in 0..op.len {
                        if qi >= read.len() || ti >= text.len() {
                            return (cost, qi, ti); // defensive; outer asserts will flag it
                        }
                        if read[qi] != text[ti] {
                            cost += pen.mismatch;
                        }
                        qi += 1;
                        ti += 1;
                    }
                }
                CigarKind::Ins | CigarKind::SoftClip => {
                    cost += pen.gap_open + pen.gap_extend * op.len as i32;
                    qi += op.len as usize;
                }
                CigarKind::Del | CigarKind::Skipped => {
                    cost += pen.gap_open + pen.gap_extend * op.len as i32;
                    ti += op.len as usize;
                }
            }
        }
        (cost, qi, ti)
    }

    /// Stress the traceback specifically for the bug class: an exact reference
    /// prefix followed by a diverging 3' tail, which drives the optimal alignment
    /// to resolve an indel or mismatch cluster right at the pattern end (where the
    /// final M cell closes a gap with no greedy extension). Every result must span
    /// the whole read AND be score-consistent when the CIGAR is replayed.
    #[test]
    fn wfa_traceback_score_consistent_across_end_divergence() {
        let pen = WfaPenalties {
            mismatch: 4,
            gap_open: 6,
            gap_extend: 1,
        };
        let mut rng = Lcg(0x1234_5678_9abc_def0);
        let text: Vec<u8> = (0..260).map(|_| rng.base()).collect();

        let mut cases = 0u32;
        for prefix in [110usize, 120, 130, 134, 140, 145, 148, 149, 150] {
            for tail in [1usize, 2, 3, 4, 6, 8, 12, 16, 20] {
                let mut read = text[..prefix].to_vec();
                for _ in 0..tail {
                    read.push(rng.base());
                }
                let Some(aln) = wfa_align(&read, &text, pen, WfaOptions::exact(400)) else {
                    continue;
                };
                cases += 1;
                assert_eq!(
                    query_consumed(&aln.cigar),
                    read.len() as u32,
                    "prefix={prefix} tail={tail}: CIGAR {:?} must span the whole read",
                    aln.cigar
                );
                let (cost, qi, ti) = replay(&read, &text, &aln, pen);
                assert_eq!(qi, read.len(), "prefix={prefix} tail={tail}: query replay length");
                assert_eq!(ti, aln.text_end, "prefix={prefix} tail={tail}: text_end mismatch");
                assert_eq!(
                    cost, aln.score,
                    "prefix={prefix} tail={tail}: replayed cost != reported score; cigar={:?}",
                    aln.cigar
                );
            }
        }
        assert!(cases >= 60, "expected many exercised cases, got {cases}");
    }
}
