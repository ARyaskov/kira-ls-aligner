pub mod ac_batch;
pub mod bitpacked;
pub mod cgk;
pub mod junc_bed;
pub mod lsh_rescue;
pub mod myers;
pub mod normalize;
pub mod prefilter;
pub mod router;
pub mod spectral;
pub mod splice;
#[cfg(target_arch = "x86_64")]
pub mod sw_int8_vnni;
pub mod wfa;

use crate::seq::reverse_complement_into;
use crate::simd::{self, SimdMode};
use crate::types::{Alignment, AlignmentKind, CigarKind, CigarOp, MateInfo, ReadRecord, Strand};

/// Alignment scoring configuration.
#[derive(Clone, Copy, Debug)]
pub struct AlignmentConfig {
    pub match_score: i32,
    pub mismatch: i32,
    pub gap_open: i32,
    pub gap_extend: i32,
    pub bandwidth: i32,
    pub xdrop: i32,
    /// bwa-mem `-L`. Cost of a tail soft-clip vs. extending through an indel.
    pub clip_penalty: i32,
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

/// Which step of the fast-DP cascade produced an alignment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FastPathKind {
    /// Bit-Packed Spectral — 2-bit DNA + SWAR popcount, ungapped only.
    PackedSpectral,
    /// Byte-resolution Spectral Sieve — ungapped multi-shift Hamming.
    SpectralSieve,
    /// Wavefront Alignment — affine-gap semi-global, handles indels.
    Wfa,
    /// SimHash-LSH fuzzy-seed rescue: catches reads with 1-2 mismatches
    /// that the exact-minimizer cascade missed. Ungapped only.
    LshRescue,
    /// CGK-rescued alignment: cascade exhausted its primary path.
    CgkRescue,
}

/// Try to build a DP-quality alignment via the routed fast aligner.
pub fn try_fast_dp_alignment(
    read_seq: &[u8],
    ref_seq: &[u8],
    chain: &AnchorSpan,
    chain_score: i32,
    cfg: AlignmentConfig,
    is_rev: bool,
) -> Option<(Alignment, FastPathKind)> {
    let result = try_fast_dp_alignment_inner(read_seq, ref_seq, chain, chain_score, cfg, is_rev);
    if result.is_some() {
        return result;
    }
    let strand = if is_rev {
        Strand::Reverse
    } else {
        Strand::Forward
    };
    // SimHash-LSH is mismatch-only and cheaper to verify than CGK's WFA
    // candidate scan, so probe it first; CGK still picks up indel-bearing
    // reads after.
    if let Some(aln) = lsh_rescue::try_lsh_fallback(read_seq, strand) {
        return Some((aln, FastPathKind::LshRescue));
    }
    if let Some(aln) = cgk::try_cgk_fallback(read_seq, strand) {
        return Some((aln, FastPathKind::CgkRescue));
    }
    None
}

/// Cheap WFA score-only probe that gates the ungapped fast-path accept in the accuracy
/// profile. Returns `true` when a gapped alignment of the read against the seed window is
/// STRICTLY cheaper (in WFA cost) than the accepted ungapped alignment with `ungapped_mism`
/// mismatches — i.e. the M-CIGAR is hiding a real (near-read-end) indel and the read should be
/// routed to the WFA traceback instead of frozen as mismatches.
///
/// The WFA budget is set one below the ungapped cost (`ungapped_mism * mismatch - 1`), so
/// `wfa_score_only` explores only that far and returns `Some` only when a genuinely cheaper,
/// indel-bearing alignment exists. SNP-only reads have no cheaper gapped alignment, so it
/// returns `None` after little work and the fast-path accept stands. Uses the same seed window
/// as `try_fast_dp_alignment_inner` and exact (unpruned) WFA so the gate cannot false-accept by
/// pruning away the winning gap diagonal.
pub fn ungapped_beaten_by_gap(
    read_seq: &[u8],
    ref_seq: &[u8],
    chain: &AnchorSpan,
    ungapped_mism: u32,
    cfg: AlignmentConfig,
) -> bool {
    if ungapped_mism == 0 {
        return false;
    }
    let read_len = read_seq.len();
    if read_len == 0 {
        return false;
    }
    let expected_ref_start = chain.ref_start as i32 - chain.read_start as i32;
    if expected_ref_start < 0 {
        return false;
    }
    let win_start = expected_ref_start as usize;
    if win_start >= ref_seq.len() {
        return false;
    }
    let band = cfg.bandwidth.max(1) as usize;
    let pad = band;
    let max_text_len = ref_seq.len() - win_start;
    let text_len = (read_len + pad).min(max_text_len);
    if text_len < read_len {
        return false;
    }
    let text = &ref_seq[win_start..win_start + text_len];

    let ungapped_cost = ungapped_mism as i32 * cfg.mismatch;
    if ungapped_cost <= 0 {
        return false;
    }
    let pen = wfa::WfaPenalties {
        mismatch: cfg.mismatch,
        gap_open: cfg.gap_open,
        gap_extend: cfg.gap_extend,
    };
    let opts = wfa::WfaOptions {
        max_score: ungapped_cost - 1,
        adaptive_drop: None,
        text_begin_free: 0,
    };
    wfa::wfa_score_only(read_seq, text, pen, opts).is_some()
}

/// Bit-parallel Myers edit distance of `read_seq` against the seed window of `chain` (semi-global
/// on the reference — best ending position in the window). Returns `None` when the edit distance
/// exceeds the Myers reject bound: a locus too divergent to be a real competitor. Used by the
/// two-tier candidate search to rank competing loci by true edit cost — far more discriminating
/// than the coarse chain (anchor-coverage) score — with bounded, per-call memory (fixed bit-vector
/// state, unlike an unbounded WFA score-only whose wavefront pool balloons on divergent loci) and
/// bit-parallel speed. Lower cost = better locus; the same seed window as `try_fast_dp_alignment`.
pub fn locus_edit_cost(
    read_seq: &[u8],
    ref_seq: &[u8],
    chain: &AnchorSpan,
    cfg: AlignmentConfig,
) -> Option<usize> {
    let read_len = read_seq.len();
    if read_len == 0 {
        return None;
    }
    let expected_ref_start = chain.ref_start as i32 - chain.read_start as i32;
    if expected_ref_start < 0 {
        return None;
    }
    let win_start = expected_ref_start as usize;
    if win_start >= ref_seq.len() {
        return None;
    }
    let band = cfg.bandwidth.max(1) as usize;
    let pad = band;
    let max_text_len = ref_seq.len() - win_start;
    let text_len = (read_len + pad).min(max_text_len);
    if text_len < read_len {
        return None;
    }
    let text = &ref_seq[win_start..win_start + text_len];
    let max_k = router::myers_reject_bound(read_len);
    myers::bounded_edit_distance(read_seq, text, max_k).map(|(dist, _pos)| dist)
}

/// Original cascade body.
fn try_fast_dp_alignment_inner(
    read_seq: &[u8],
    ref_seq: &[u8],
    chain: &AnchorSpan,
    chain_score: i32,
    cfg: AlignmentConfig,
    is_rev: bool,
) -> Option<(Alignment, FastPathKind)> {
    let read_len = read_seq.len();
    if read_len == 0 {
        return None;
    }
    let kind = router::choose_aligner(read_len);
    // Long reads or `KIRA_ALGO=sw` → banded SW immediately.
    if matches!(kind, router::AlignerKind::BandedSw) {
        return None;
    }
    if !router::fast_path_worth_attempting(chain_score, read_len) {
        return None;
    }

    let expected_ref_start = chain.ref_start as i32 - chain.read_start as i32;
    if expected_ref_start < 0 {
        return None;
    }
    let win_start = expected_ref_start as usize;
    if win_start >= ref_seq.len() {
        return None;
    }
    let band = cfg.bandwidth.max(1) as usize;
    let pad = band;
    let max_text_len = ref_seq.len() - win_start;
    let text_len = (read_len + pad).min(max_text_len);
    if text_len < read_len {
        // Not enough reference to host the pattern — fall back.
        return None;
    }
    let text = &ref_seq[win_start..win_start + text_len];

    // `SpectralSieve` and `PackedSpectral` resolve to the same kernel, so probe
    // once and attribute the result to whichever kind was routed.
    if matches!(
        kind,
        router::AlignerKind::PackedSpectral | router::AlignerKind::SpectralSieve
    ) && let Some(aln) =
        try_packed_spectral_alignment(read_seq, text, win_start, chain, cfg, is_rev)
    {
        let attribution = if matches!(kind, router::AlignerKind::PackedSpectral) {
            FastPathKind::PackedSpectral
        } else {
            FastPathKind::SpectralSieve
        };
        return Some((aln, attribution));
    }
    // Spectral didn't apply (indel suspected) — fall through to WFA below.

    let myers_k = router::myers_reject_bound(read_len);
    myers::bounded_edit_distance(read_seq, text, myers_k)?;

    let pen = wfa::WfaPenalties {
        mismatch: cfg.mismatch,
        gap_open: cfg.gap_open,
        gap_extend: cfg.gap_extend,
    };
    let budget = router::wfa_score_budget(read_len, pen.mismatch, pen.gap_open, pen.gap_extend);
    // Leading-edge slack for WFA only (the spectral/packed paths above require the
    // read to start at text[0]). `text` begins exactly at the seed-implied win_start
    // with only trailing slack, so a 5' deletion before the seed is unrepresentable.
    // When KIRA_WFA_LEAD > 0, extend the WFA text upstream and free those bases.
    // lead == 0 (default) reproduces prior behavior byte-for-byte.
    let lead = (router::wfa_lead().max(0) as usize)
        .min(win_start)
        .min(band);
    let (wfa_text, wfa_text_start, text_begin_free) = if lead > 0 {
        (
            &ref_seq[win_start - lead..win_start + text_len],
            win_start - lead,
            lead as i32,
        )
    } else {
        (text, win_start, router::wfa_ends_free())
    };
    let opts = wfa::WfaOptions {
        max_score: budget,
        adaptive_drop: router::wfa_adaptive_drop(),
        text_begin_free,
    };
    let wfa_aln = wfa::wfa_align(read_seq, wfa_text, pen, opts)?;

    let aln = wfa_result_to_alignment(
        read_seq,
        wfa_text,
        wfa_text_start,
        chain.ref_id,
        wfa_aln,
        cfg,
        is_rev,
    )?;
    Some((aln, FastPathKind::Wfa))
}

/// Convert a `wfa::WfaAlignment` produced by `wfa_align_semi_global` into our in-pipeline.
pub fn wfa_result_to_alignment(
    read_seq: &[u8],
    text: &[u8],
    win_start_global: usize,
    ref_id: u32,
    wfa_aln: wfa::WfaAlignment,
    cfg: AlignmentConfig,
    is_rev: bool,
) -> Option<Alignment> {
    let read_len = read_seq.len();
    let cigar = wfa_aln.cigar;
    let (cigar_ref, cigar_query) = cigar.iter().fold((0usize, 0usize), |(r, q), op| {
        let l = op.len as usize;
        match op.op {
            CigarKind::Match => (r + l, q + l),
            CigarKind::Ins | CigarKind::SoftClip => (r, q + l),
            CigarKind::Del | CigarKind::Skipped => (r + l, q),
        }
    });
    // The WFA CIGAR must span the entire pattern: the engine is query-global,
    // so every read base is an M or I op (there are no soft clips). A CIGAR that
    // consumes a different number of query bases than the read length means the
    // traceback terminated early (see `wfa::build_cigar`). Reject it rather than
    // emit a malformed alignment with the wrong reference coordinates — the
    // caller's cascade then falls through to banded SW / rescue (or the read is
    // left unmapped), and the io SAM emitter's `consumed == seq_len` invariant
    // is never violated.
    if cigar_query != read_len
        || wfa_aln.text_start.saturating_add(cigar_ref) > text.len()
        || wfa_aln.text_end > text.len()
    {
        return None;
    }
    let mut sw_score: i32 = 0;
    let mut nm: u32 = 0;
    let mut md_bytes: Vec<u8> = Vec::with_capacity(16);
    let mut match_run: u32 = 0;
    let mut qi = 0usize;
    let mut ti = wfa_aln.text_start;
    for op in &cigar {
        match op.op {
            CigarKind::Match => {
                for _ in 0..op.len {
                    let qb = read_seq[qi];
                    let rb = text[ti];
                    if qb == rb {
                        sw_score += cfg.match_score;
                        match_run += 1;
                    } else {
                        sw_score -= cfg.mismatch;
                        nm += 1;
                        push_u32_decimal(&mut md_bytes, match_run);
                        md_bytes.push(rb);
                        match_run = 0;
                    }
                    qi += 1;
                    ti += 1;
                }
            }
            CigarKind::Ins => {
                sw_score -= cfg.gap_open + cfg.gap_extend * op.len as i32;
                nm += op.len;
                qi += op.len as usize;
            }
            CigarKind::Del => {
                sw_score -= cfg.gap_open + cfg.gap_extend * op.len as i32;
                nm += op.len;
                push_u32_decimal(&mut md_bytes, match_run);
                md_bytes.push(b'^');
                for _ in 0..op.len {
                    md_bytes.push(text[ti]);
                    ti += 1;
                }
                match_run = 0;
            }
            CigarKind::SoftClip => {
                qi += op.len as usize;
            }
            CigarKind::Skipped => {
                push_u32_decimal(&mut md_bytes, match_run);
                match_run = 0;
                ti += op.len as usize;
            }
        }
    }
    push_u32_decimal(&mut md_bytes, match_run);
    // SAFETY: only ASCII digits, ACGTN and '^' were pushed.
    let md = unsafe { String::from_utf8_unchecked(md_bytes) };

    // `text_start` is 0 unless ends-free leading text was enabled; honoring it
    // keeps ref_start correct when the read aligns past the window start.
    let ref_start_global = win_start_global + wfa_aln.text_start;
    let ref_end_global = win_start_global + wfa_aln.text_end;

    Some(Alignment {
        kind: AlignmentKind::DpAligned,
        ref_id,
        ref_start: ref_start_global as u32,
        ref_end: ref_end_global as u32,
        read_start: 0,
        read_end: read_len as u32,
        cigar,
        score: sw_score,
        mapq: 0,
        is_rev,
        is_secondary: false,
        is_supplementary: false,
        nm,
        md,
        as_score: sw_score,
        xs_score: None,
        xs_strand: None,
        mate: MateInfo::default(),
    })
}

/// Reusable per-thread buffers for the 2-bit packed spectral scan, which would
/// otherwise allocate six `Vec`s per fast-path candidate.
#[derive(Default)]
struct PackedScratch {
    read_bits: Vec<u8>,
    ref_bits: Vec<u8>,
    ref_shifted: [Vec<u8>; 4],
}

thread_local! {
    static PACKED_SCRATCH: std::cell::RefCell<PackedScratch> =
        std::cell::RefCell::new(PackedScratch::default());
}

/// Run the packed spectral scan against `text` using thread-local scratch.
/// Returns the best hit and the best mismatch count at a distinct shift.
fn packed_scan_scratch(
    read_seq: &[u8],
    text: &[u8],
) -> Option<(bitpacked::PackedHit, Option<usize>)> {
    PACKED_SCRATCH.with(|cell| {
        let mut s = cell.borrow_mut();
        let PackedScratch {
            read_bits,
            ref_bits,
            ref_shifted,
        } = &mut *s;
        let read_valid = bitpacked::pack_into(read_seq, read_bits);
        if !read_valid {
            return None;
        }
        if !bitpacked::pack_into(text, ref_bits) {
            return None;
        }
        bitpacked::pre_shift_into(ref_bits, ref_shifted);
        bitpacked::scan_best_with_second_raw(
            read_bits,
            read_seq.len(),
            read_valid,
            ref_shifted,
            text.len(),
        )
    })
}

/// Bit-Packed Spectral fast path.
fn try_packed_spectral_alignment(
    read_seq: &[u8],
    text: &[u8],
    win_start: usize,
    chain: &AnchorSpan,
    cfg: AlignmentConfig,
    is_rev: bool,
) -> Option<Alignment> {
    let read_len = read_seq.len();
    let max_mism = router::spectral_max_mismatches(read_len);

    let (hit, second_mismatches) = packed_scan_scratch(read_seq, text)?;
    let ref_aligned = &text[hit.shift..hit.shift + read_len];
    if !spectral_hit_is_certified(read_seq, ref_aligned, &hit, second_mismatches, max_mism) {
        return None;
    }

    let cigar = vec![CigarOp {
        len: read_len as u32,
        op: CigarKind::Match,
    }];

    let mut nm: u32 = 0;
    let mut md_bytes: Vec<u8> = Vec::with_capacity(16);
    let mut match_run: u32 = 0;
    for (qb, rb) in read_seq.iter().zip(ref_aligned.iter()) {
        if qb == rb {
            match_run += 1;
        } else {
            nm += 1;
            push_u32_decimal(&mut md_bytes, match_run);
            md_bytes.push(*rb);
            match_run = 0;
        }
    }
    push_u32_decimal(&mut md_bytes, match_run);
    // SAFETY: only ASCII digits and ACGTN bases were pushed.
    let md = unsafe { String::from_utf8_unchecked(md_bytes) };

    let matches = hit.matches as i32;
    let mismatches = hit.mismatches as i32;
    let sw_score = matches * cfg.match_score - mismatches * cfg.mismatch;

    let ref_start_global = win_start + hit.shift;
    let ref_end_global = ref_start_global + read_len;

    Some(Alignment {
        kind: AlignmentKind::DpAligned,
        ref_id: chain.ref_id,
        ref_start: ref_start_global as u32,
        ref_end: ref_end_global as u32,
        read_start: 0,
        read_end: read_len as u32,
        cigar,
        score: sw_score,
        mapq: 0,
        is_rev,
        is_secondary: false,
        is_supplementary: false,
        nm,
        md,
        as_score: sw_score,
        xs_score: None,
        xs_strand: None,
        mate: MateInfo::default(),
    })
}

fn spectral_hit_is_certified(
    read: &[u8],
    reference: &[u8],
    hit: &bitpacked::PackedHit,
    second_mismatches: Option<usize>,
    max_mismatches: usize,
) -> bool {
    if read.len() != reference.len()
        || hit.mismatches > max_mismatches
        || second_mismatches.is_some_and(|second| second <= hit.mismatches.saturating_add(1))
    {
        return false;
    }
    if hit.mismatches == 0 {
        return true;
    }
    // The ungapped path is a certificate, not a heuristic aligner. More than
    // two residual differences, terminal differences, or a compact mismatch
    // cluster are cheap signals that a small indel is being represented as M.
    if hit.mismatches > 2 {
        return false;
    }
    let mut mismatch_pos = [usize::MAX; 2];
    let mut n = 0usize;
    for (i, (&q, &r)) in read.iter().zip(reference).enumerate() {
        if q != r {
            if n < mismatch_pos.len() {
                mismatch_pos[n] = i;
            }
            n += 1;
        }
    }
    if n != hit.mismatches {
        return false;
    }
    let edge = read.len().min(8);
    if mismatch_pos[..n]
        .iter()
        .any(|&p| p < edge || p.saturating_add(edge) >= read.len())
    {
        return false;
    }
    if n == 2 && mismatch_pos[1].saturating_sub(mismatch_pos[0]) <= 4 {
        return false;
    }

    // If deleting one base from either sequence after the first mismatch makes
    // the remaining suffix nearly exact, the Hamming hit is an indel proxy.
    let p = mismatch_pos[0];
    let suffix_len = read.len().saturating_sub(p + 1);
    let strict = cert_strict();
    // Strict mode shortens the minimum suffix so near-end indels are still probed.
    let min_suffix = if strict { 4 } else { 8 };
    if suffix_len >= min_suffix {
        let read_insertion_like =
            bounded_hamming(&read[p + 1..], &reference[p..reference.len() - 1], 1) <= 1;
        let ref_insertion_like =
            bounded_hamming(&read[p..read.len() - 1], &reference[p + 1..], 1) <= 1;
        if read_insertion_like || ref_insertion_like {
            return false;
        }
        // A 2bp indel shifts the suffix by two; the 1-base shift above cannot reveal
        // it, so a clean 2bp-indel read would otherwise be certified as 2 mismatches.
        if strict && suffix_len >= 6 {
            let read_ins2_like =
                bounded_hamming(&read[p + 2..], &reference[p..reference.len() - 2], 1) <= 1;
            let ref_ins2_like =
                bounded_hamming(&read[p..read.len() - 2], &reference[p + 2..], 1) <= 1;
            if read_ins2_like || ref_ins2_like {
                return false;
            }
        }
    }
    true
}

fn bounded_hamming(a: &[u8], b: &[u8], limit: usize) -> usize {
    if a.len() != b.len() {
        return limit + 1;
    }
    let mut mismatches = 0usize;
    for (&x, &y) in a.iter().zip(b) {
        mismatches += usize::from(x != y);
        if mismatches > limit {
            break;
        }
    }
    mismatches
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
        xs_strand: None,
        mate: MateInfo::default(),
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

/// Public banded-SW wrapper.
pub fn banded_sw_public(
    read: &[u8],
    reference: &[u8],
    offset: i32,
    cfg: AlignmentConfig,
    abort_score: i32,
) -> SwPublicResult {
    let r = banded_sw(read, reference, offset, cfg, abort_score);
    SwPublicResult {
        ref_start: r.ref_start,
        ref_end: r.ref_end,
        read_start: r.read_start,
        read_end: r.read_end,
        score: r.score,
        cigar: r.cigar,
        early_abort: r.early_abort,
    }
}

/// Public mirror of the internal `SwResult` — separate type so we can keep the private one private.
#[derive(Clone, Debug)]
pub struct SwPublicResult {
    pub ref_start: u32,
    pub ref_end: u32,
    pub read_start: i32,
    pub read_end: i32,
    pub score: i32,
    pub cigar: Vec<CigarOp>,
    pub early_abort: bool,
}

/// `KIRA_RESCUE_NO_FAST` — force mate rescue through the wide banded-SW path.
/// Read once: `align_in_window` runs for every rescued mate.
#[inline]
fn rescue_no_fast() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| std::env::var_os("KIRA_RESCUE_NO_FAST").is_some())
}

/// Align a read against a *wide* reference window — no chain anchor required.
pub fn align_in_window(
    read_seq: &[u8],
    ref_window: &[u8],
    win_start: u32,
    ref_id: u32,
    is_rev: bool,
    cfg: AlignmentConfig,
    min_score: i32,
) -> Option<Alignment> {
    let read_len = read_seq.len();
    if read_len == 0 || ref_window.is_empty() || read_len > ref_window.len() {
        return None;
    }
    if rescue_no_fast() {
        return align_in_window_wide_sw(
            read_seq, ref_window, win_start, ref_id, is_rev, cfg, min_score,
        );
    }

    {
        if let Some((best, second)) = packed_scan_scratch(read_seq, ref_window) {
            let max_mism = router::spectral_max_mismatches(read_len);
            let target = &ref_window[best.shift..best.shift + read_len];
            if spectral_hit_is_certified(read_seq, target, &best, second, max_mism) {
                if let Some(aln) = build_spectral_alignment(
                    read_seq, ref_window, win_start, ref_id, is_rev, cfg, min_score, &best,
                ) {
                    return Some(aln);
                }
            }

            let band = cfg.bandwidth.max(20);
            let mut narrow_cfg = cfg;
            narrow_cfg.bandwidth = band;
            let sw = banded_sw(
                read_seq,
                ref_window,
                best.shift as i32,
                narrow_cfg,
                i32::MIN / 8,
            );
            if sw.score >= min_score {
                let chain = AnchorSpan {
                    ref_id,
                    ref_start: win_start + sw.ref_start,
                    ref_end: win_start + sw.ref_end,
                    read_start: sw.read_start.max(0) as u32,
                    read_end: sw.read_end.max(0) as u32,
                    strand: if is_rev {
                        crate::types::Strand::Reverse
                    } else {
                        crate::types::Strand::Forward
                    },
                };
                return Some(build_alignment(
                    read_seq, ref_window, win_start, &chain, is_rev, sw,
                ));
            }
            if !rescue_wide_fallback() {
                return None;
            }
        }
    }
    align_in_window_wide_sw(
        read_seq, ref_window, win_start, ref_id, is_rev, cfg, min_score,
    )
}

/// `KIRA_RESCUE_WIDE=1` — after the banded rescue attempt at the best ungapped
/// diagonal misses `min_score`, also run the full-window SW. Default **off**.
///
/// That fallback sets the band to half the window, i.e. an unbanded
/// O(read x window) scalar DP, and it was reached for most discordant pairs
/// because `min_score` there is the mate's existing score. It cost 22% of the
/// alignment stage while changing nothing: over two 800k/300k-read regression
/// sets — the second carrying indels up to 30 bp in 20% of reads, i.e. exactly
/// the off-diagonal case it exists for — enabling it moved zero reads in
/// placement, MAPQ or CIGAR.
///
/// This only skips the *second* DP. When the packed scan cannot run at all
/// (ambiguous bases in the window) the wide search still happens, since there is
/// no best diagonal to band around.
#[inline]
fn rescue_wide_fallback() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("KIRA_RESCUE_WIDE")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Wide-band SW path — the original `align_in_window` body.
fn align_in_window_wide_sw(
    read_seq: &[u8],
    ref_window: &[u8],
    win_start: u32,
    ref_id: u32,
    is_rev: bool,
    cfg: AlignmentConfig,
    min_score: i32,
) -> Option<Alignment> {
    let mut wide_cfg = cfg;
    wide_cfg.bandwidth = (ref_window.len() as i32 / 2).max(cfg.bandwidth);
    let sw = banded_sw(read_seq, ref_window, 0, wide_cfg, i32::MIN / 8);
    if sw.score < min_score {
        return None;
    }
    let chain = AnchorSpan {
        ref_id,
        ref_start: win_start + sw.ref_start,
        ref_end: win_start + sw.ref_end,
        read_start: sw.read_start.max(0) as u32,
        read_end: sw.read_end.max(0) as u32,
        strand: if is_rev {
            crate::types::Strand::Reverse
        } else {
            crate::types::Strand::Forward
        },
    };
    Some(build_alignment(
        read_seq, ref_window, win_start, &chain, is_rev, sw,
    ))
}

/// Helper that builds an ungapped `Alignment` from a `PackedHit`.
fn build_spectral_alignment(
    read_seq: &[u8],
    ref_window: &[u8],
    win_start: u32,
    ref_id: u32,
    is_rev: bool,
    cfg: AlignmentConfig,
    min_score: i32,
    hit: &bitpacked::PackedHit,
) -> Option<Alignment> {
    let read_len = read_seq.len();
    let matches = hit.matches as i32;
    let mismatches = hit.mismatches as i32;
    let sw_score = matches * cfg.match_score - mismatches * cfg.mismatch;
    if sw_score < min_score {
        return None;
    }

    let ref_aligned = &ref_window[hit.shift..hit.shift + read_len];
    let mut nm: u32 = 0;
    let mut md_bytes: Vec<u8> = Vec::with_capacity(16);
    let mut match_run: u32 = 0;
    for (qb, rb) in read_seq.iter().zip(ref_aligned.iter()) {
        if qb == rb {
            match_run += 1;
        } else {
            nm += 1;
            push_u32_decimal(&mut md_bytes, match_run);
            md_bytes.push(*rb);
            match_run = 0;
        }
    }
    push_u32_decimal(&mut md_bytes, match_run);
    // SAFETY: only ASCII digits and ACGTN bases were pushed.
    let md = unsafe { String::from_utf8_unchecked(md_bytes) };

    let cigar = vec![CigarOp {
        len: read_len as u32,
        op: CigarKind::Match,
    }];
    let ref_start_global = win_start + hit.shift as u32;
    let ref_end_global = ref_start_global + read_len as u32;

    Some(Alignment {
        kind: AlignmentKind::DpAligned,
        ref_id,
        ref_start: ref_start_global,
        ref_end: ref_end_global,
        read_start: 0,
        read_end: read_len as u32,
        cigar,
        score: sw_score,
        mapq: 0,
        is_rev,
        is_secondary: false,
        is_supplementary: false,
        nm,
        md,
        as_score: sw_score,
        xs_score: None,
        xs_strand: None,
        mate: MateInfo::default(),
    })
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
        SimdMode::AvxVnni => unsafe { sw_dispatch_avx_vnni(inputs, cfg, read_len) },
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

pub(crate) struct SwResult {
    pub ref_start: u32,
    pub ref_end: u32,
    pub read_start: i32,
    pub read_end: i32,
    pub score: i32,
    pub cigar: Vec<CigarOp>,
    pub early_abort: bool,
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

    let ref_start = win_start + sw.ref_start;
    let ref_end = win_start + sw.ref_end;

    // `cigar` already contains leading soft clips, so replay starts at query
    // position zero. Starting at `sw.read_start` would count the clip twice.
    let (nm, md) = compute_nm_md(read_seq, ref_window, 0, sw.ref_start as usize, &cigar);

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
        xs_strand: None,
        mate: MateInfo::default(),
    }
}

/// Scalar banded Smith-Waterman — also the fallback for INT8 lanes that saturate.
pub(crate) fn banded_sw_internal(
    read: &[u8],
    reference: &[u8],
    offset: i32,
    cfg: AlignmentConfig,
    abort_score: i32,
) -> SwResult {
    banded_sw(read, reference, offset, cfg, abort_score)
}

/// Whether the banded DP re-centers its band on the running-max column of the
/// previous row (Suzuki–Kasahara / libgaba-style *adaptive* banding) instead of
/// pinning it to the static seed diagonal `i + offset`.
///
/// A fixed band loses indels that push the optimal path off the seed diagonal —
/// the classic "indel near the read end is modelled as mismatches and clipped"
/// failure. The adaptive band follows the score frontier, so an indel mid-read
/// shifts the band and the post-indel bases stay inside it. For a gapless
/// alignment the adaptive centre equals `i + offset`, so this is a strict
/// superset of the previous behaviour. Kill-switch: `KIRA_ADAPTIVE_BAND=0`.
#[inline]
fn adaptive_band_enabled() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("KIRA_ADAPTIVE_BAND")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .map(|v| v != 0)
            .unwrap_or(true)
    })
}

/// Optional override for the gapped-DP x-drop, from `KIRA_DP_XDROP`. `None` (default)
/// uses `cfg.xdrop`, keeping the prefilter and DP x-drop coupled as before. Decoupling
/// lets the DP run a looser x-drop (so the wide band scores through an indel+SNP
/// cluster) while the prefilter's ungapped extension stays tight.
#[inline]
fn dp_xdrop_override() -> Option<i32> {
    use std::sync::OnceLock;
    static CELL: OnceLock<Option<i32>> = OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("KIRA_DP_XDROP")
            .ok()
            .and_then(|v| v.trim().parse::<i32>().ok())
            .filter(|&v| v >= 0)
    })
}

/// Stricter ungapped certification, from `KIRA_CERT_STRICT`. When enabled, the
/// indel-proxy check in `spectral_hit_is_certified` also tests 2-base shifts and uses
/// a shorter minimum suffix, so disguised 1-2bp indels fall through to WFA instead of
/// being emitted as M-only. Default off (prior behavior).
#[inline]
fn cert_strict() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("KIRA_CERT_STRICT")
            .ok()
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("off"))
            .unwrap_or(false)
    })
}

/// Maximum the adaptive band centre may deviate from the static seed diagonal
/// (`KIRA_ADAPTIVE_MAX_DRIFT`, default 16). Unbounded re-centering chases
/// spurious matches across long non-matching stretches (e.g. the wrong half of
/// a chimeric read), so the drift is capped. The band's own half-width is still
/// added around the (clamped) centre, so the reachable diagonal range is
/// `±(band + max_drift)` — ample for real read-end/mid-read indels while
/// keeping the band anchored to the seed.
#[inline]
fn adaptive_max_drift() -> i32 {
    use std::sync::OnceLock;
    static CELL: OnceLock<i32> = OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("KIRA_ADAPTIVE_MAX_DRIFT")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .filter(|&v| v >= 0)
            .unwrap_or(16)
    })
}

fn banded_sw(
    read: &[u8],
    reference: &[u8],
    offset: i32,
    cfg: AlignmentConfig,
    abort_score: i32,
) -> SwResult {
    banded_sw_with(
        read,
        reference,
        offset,
        cfg,
        abort_score,
        adaptive_band_enabled(),
    )
}

/// Per-thread reusable buffers for the scalar banded DP. Mirrors the thread-local
/// scratch the SIMD kernels already use, so a 150bp read no longer allocates ~4
/// Vecs per row (cur_h/cur_e/cur_f/trace) plus the outer buffers on every call.
#[derive(Default)]
struct ScalarSwScratch {
    prev_h: Vec<i32>,
    prev_e: Vec<i32>,
    cur_h: Vec<i32>,
    cur_e: Vec<i32>,
    cur_f: Vec<i32>,
    trace_rows: Vec<Vec<u8>>,
    row_starts: Vec<i32>,
}

thread_local! {
    static SCALAR_SW_SCRATCH: std::cell::RefCell<ScalarSwScratch> =
        std::cell::RefCell::new(ScalarSwScratch::default());
}

/// Core banded DP. `adaptive` selects Suzuki–Kasahara band re-centering vs the
/// static seed-diagonal band; split out so tests can exercise both modes in one
/// process (the public `banded_sw` reads the choice once from the environment).
fn banded_sw_with(
    read: &[u8],
    reference: &[u8],
    offset: i32,
    cfg: AlignmentConfig,
    abort_score: i32,
    adaptive: bool,
) -> SwResult {
    let q_len = read.len();
    let r_len = reference.len();
    let band = cfg.bandwidth.max(1);
    // Gapped-DP x-drop, decoupled from the prefilter's ungapped x-drop (both default
    // to cfg.xdrop). KIRA_DP_XDROP lets the wide adaptive band actually score through
    // an indel+SNP cluster without the prefilter-tight x-drop aborting the sweep.
    let dp_xdrop = dp_xdrop_override().unwrap_or(cfg.xdrop);

    // Take the per-thread scratch buffers into locals; they are returned to the
    // thread-local at the single exit point below. Used exactly like the old
    // freshly-allocated Vecs — only the storage source changed.
    let (mut prev_h, mut prev_e, mut cur_h, mut cur_e, mut cur_f, mut trace_rows, mut row_starts) =
        SCALAR_SW_SCRATCH.with(|c| {
            let mut s = c.borrow_mut();
            (
                std::mem::take(&mut s.prev_h),
                std::mem::take(&mut s.prev_e),
                std::mem::take(&mut s.cur_h),
                std::mem::take(&mut s.cur_e),
                std::mem::take(&mut s.cur_f),
                std::mem::take(&mut s.trace_rows),
                std::mem::take(&mut s.row_starts),
            )
        });
    prev_h.clear();
    prev_e.clear();
    let mut prev_start = 1i32;

    // Recycle the inner trace Vecs across calls; ensure a slot for every row.
    if trace_rows.len() < q_len + 1 {
        trace_rows.resize_with(q_len + 1, Vec::new);
    }
    row_starts.clear();
    row_starts.resize(q_len + 1, 1i32);

    let mut best_score = 0;
    let mut best_i = 0usize;
    let mut best_j = 0usize;
    let mut best_qlen_score: i32 = 0;
    let mut best_qlen_j: usize = 0;

    let mut early_abort = false;

    // Column the next row's band is centred on. For row 1 this is the static
    // seed diagonal; thereafter it tracks the previous row's best-scoring column
    // (+1 to advance one step down the diagonal) so the band follows indel drift,
    // but is clamped to within `max_drift` of the static diagonal.
    let max_drift = if adaptive { adaptive_max_drift() } else { 0 };
    let mut adaptive_center = 1i32 + offset;

    for i in 1..=q_len {
        let static_center = i as i32 + offset;
        let center = if adaptive {
            adaptive_center.clamp(static_center - max_drift, static_center + max_drift)
        } else {
            static_center
        };
        let j_start = (center - band).max(1);
        let j_end = (center + band).min(r_len as i32);
        if j_start > j_end {
            row_starts[i] = 1;
            trace_rows[i].clear();
            prev_h.clear();
            prev_e.clear();
            prev_start = 1;
            continue;
        }
        let row_len = (j_end - j_start + 1) as usize;
        row_starts[i] = j_start;

        cur_h.clear();
        cur_h.resize(row_len, 0i32);
        cur_e.clear();
        cur_e.resize(row_len, i32::MIN / 4);
        cur_f.clear();
        cur_f.resize(row_len, i32::MIN / 4);
        let mut trace = std::mem::take(&mut trace_rows[i]);
        trace.clear();
        trace.resize(row_len, 0u8);
        let mut row_best = 0i32;
        // Column of this row's best-scoring cell (-1 if no positive cell), used
        // to re-centre the adaptive band for the next row.
        let mut row_best_j = -1i32;

        for j in j_start..=j_end {
            let idx = (j - j_start) as usize;
            let (h_diag, score_diag) =
                if let Some((h, s)) = prev_diag(i, j, prev_start, &prev_h, read, reference, cfg) {
                    (h, s)
                } else {
                    // No diagonal cell in the previous band (local-alignment start or band
                    // edge). Local SW: the implicit H above-left is the 0 boundary, but the
                    // base pair STILL scores. Returning score 0 here dropped the first matched
                    // base at the matrix top-left (soft-clip) — a long-standing off-by-one.
                    let qb = read[i - 1];
                    let rb = reference[(j - 1) as usize];
                    let s = if qb == rb { cfg.match_score } else { -cfg.mismatch };
                    (0, s)
                };
            let h_match = h_diag + score_diag;

            let gap_open_first = cfg.gap_open.saturating_add(cfg.gap_extend);
            let e_from_h = prev_cell(j, prev_start, &prev_h)
                .map(|v| v - gap_open_first)
                .unwrap_or(i32::MIN / 4);
            let e_from_e = prev_cell(j, prev_start, &prev_e)
                .map(|v| v - cfg.gap_extend)
                .unwrap_or(i32::MIN / 4);
            let e = e_from_h.max(e_from_e);

            let (f_from_h, f_from_f) = if idx > 0 {
                (
                    cur_h[idx - 1] - gap_open_first,
                    cur_f[idx - 1] - cfg.gap_extend,
                )
            } else {
                (i32::MIN / 4, i32::MIN / 4)
            };
            let f = f_from_h.max(f_from_f);
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
            if e_from_e > e_from_h {
                tr |= TRACE_E_EXTEND;
            }
            if f_from_f > f_from_h {
                tr |= TRACE_F_EXTEND;
            }
            trace[idx] = tr;
            if h > row_best {
                row_best = h;
                row_best_j = j;
            }

            if h > best_score {
                best_score = h;
                best_i = i;
                best_j = j as usize;
            }
            if i == q_len && h > best_qlen_score {
                best_qlen_score = h;
                best_qlen_j = j as usize;
            }
        }

        trace_rows[i] = trace;
        std::mem::swap(&mut prev_h, &mut cur_h);
        std::mem::swap(&mut prev_e, &mut cur_e);
        prev_start = j_start;

        if adaptive {
            // Track the score frontier: centre next row on this row's best
            // column (advanced one step). If the row had no positive cell, keep
            // marching the band diagonally so it never stalls. Clamp to within
            // `max_drift` of the next row's static diagonal so spurious matches
            // in a non-matching stretch can't drag the band away from the seed.
            let raw = if row_best_j >= 0 {
                row_best_j + 1
            } else {
                adaptive_center + 1
            };
            let static_next = (i as i32 + 1) + offset;
            adaptive_center = raw.clamp(static_next - max_drift, static_next + max_drift);
        }

        if dp_xdrop > 0 && best_score - row_best > dp_xdrop {
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

    if best_qlen_score > 0 && best_qlen_score.saturating_add(cfg.clip_penalty) > best_score {
        best_score = best_qlen_score;
        best_i = q_len;
        best_j = best_qlen_j;
    }

    let mut cigar = Vec::new();
    let mut i = best_i as i32;
    let mut j = best_j as i32;
    let read_end = i;
    let ref_end = j as u32;

    let mut state = TraceState::H;
    while i > 0 && j > 0 {
        let row_start = row_starts[i as usize];
        let idx = (j - row_start) as usize;
        if idx >= trace_rows[i as usize].len() {
            break;
        }
        let tr = trace_rows[i as usize][idx];
        match state {
            TraceState::H => match tr & TRACE_DIR_MASK {
                TRACE_STOP => break,
                TRACE_DIAG => {
                    push_cigar(&mut cigar, CigarKind::Match, 1);
                    i -= 1;
                    j -= 1;
                }
                TRACE_E => state = TraceState::E,
                TRACE_F => state = TraceState::F,
                _ => break,
            },
            TraceState::E => {
                push_cigar(&mut cigar, CigarKind::Ins, 1);
                i -= 1;
                state = if tr & TRACE_E_EXTEND != 0 {
                    TraceState::E
                } else {
                    TraceState::H
                };
            }
            TraceState::F => {
                push_cigar(&mut cigar, CigarKind::Del, 1);
                j -= 1;
                state = if tr & TRACE_F_EXTEND != 0 {
                    TraceState::F
                } else {
                    TraceState::H
                };
            }
        }
    }

    cigar.reverse();
    let result = SwResult {
        ref_start: j as u32,
        ref_end,
        read_start: i,
        read_end,
        score: best_score,
        cigar,
        early_abort,
    };
    // Return the scratch buffers (with their grown capacity) for the next call.
    SCALAR_SW_SCRATCH.with(|c| {
        let mut s = c.borrow_mut();
        s.prev_h = prev_h;
        s.prev_e = prev_e;
        s.cur_h = cur_h;
        s.cur_e = cur_e;
        s.cur_f = cur_f;
        s.trace_rows = trace_rows;
        s.row_starts = row_starts;
    });
    result
}

pub(crate) const TRACE_STOP: u8 = 0;
pub(crate) const TRACE_DIAG: u8 = 1;
pub(crate) const TRACE_E: u8 = 2;
pub(crate) const TRACE_F: u8 = 3;
pub(crate) const TRACE_DIR_MASK: u8 = 0x03;
pub(crate) const TRACE_E_EXTEND: u8 = 0x04;
pub(crate) const TRACE_F_EXTEND: u8 = 0x08;

#[derive(Clone, Copy)]
enum TraceState {
    H,
    E,
    F,
}

pub(crate) fn traceback_interleaved(
    trace: &[u8],
    trace_width: usize,
    lane_stride: usize,
    lane: usize,
    mut i: i32,
    mut j: i32,
) -> (Vec<CigarOp>, i32, i32) {
    let mut cigar = Vec::new();
    let mut state = TraceState::H;
    while i > 0 && j > 0 {
        let idx = (i as usize * trace_width + j as usize) * lane_stride + lane;
        let Some(&tr) = trace.get(idx) else {
            break;
        };
        match state {
            TraceState::H => match tr & TRACE_DIR_MASK {
                TRACE_STOP => break,
                TRACE_DIAG => {
                    push_cigar(&mut cigar, CigarKind::Match, 1);
                    i -= 1;
                    j -= 1;
                }
                TRACE_E => state = TraceState::E,
                TRACE_F => state = TraceState::F,
                _ => break,
            },
            TraceState::E => {
                push_cigar(&mut cigar, CigarKind::Ins, 1);
                i -= 1;
                state = if tr & TRACE_E_EXTEND != 0 {
                    TraceState::E
                } else {
                    TraceState::H
                };
            }
            TraceState::F => {
                push_cigar(&mut cigar, CigarKind::Del, 1);
                j -= 1;
                state = if tr & TRACE_F_EXTEND != 0 {
                    TraceState::F
                } else {
                    TraceState::H
                };
            }
        }
    }
    cigar.reverse();
    (cigar, i, j)
}

#[cfg(target_arch = "aarch64")]
fn traceback_dense(trace: &[u8], trace_width: usize, i: i32, j: i32) -> (Vec<CigarOp>, i32, i32) {
    traceback_interleaved(trace, trace_width, 1, 0, i, j)
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

pub(crate) fn push_cigar(cigar: &mut Vec<CigarOp>, op: CigarKind, len: u32) {
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

/// Append decimal digits of `v` to a byte buffer without allocating.
#[inline]
pub(crate) fn push_u32_decimal(out: &mut Vec<u8>, mut v: u32) {
    if v == 0 {
        out.push(b'0');
        return;
    }
    let mut tmp = [0u8; 10];
    let mut i = 0usize;
    while v > 0 {
        tmp[i] = b'0' + (v % 10) as u8;
        v /= 10;
        i += 1;
    }
    for idx in (0..i).rev() {
        out.push(tmp[idx]);
    }
}

pub(crate) fn compute_nm_md(
    read: &[u8],
    reference: &[u8],
    read_start: usize,
    ref_start: usize,
    cigar: &[CigarOp],
) -> (u32, String) {
    let mut nm = 0u32;
    let mut md_bytes: Vec<u8> = Vec::with_capacity(16);
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
                        push_u32_decimal(&mut md_bytes, match_count);
                        md_bytes.push(rb);
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
                push_u32_decimal(&mut md_bytes, match_count);
                md_bytes.push(b'^');
                for _ in 0..op.len {
                    let rb = reference.get(rpos).copied().unwrap_or(b'N');
                    md_bytes.push(rb);
                    rpos += 1;
                }
                match_count = 0;
            }
            CigarKind::SoftClip => {
                qpos += op.len as usize;
            }
            CigarKind::Skipped => {
                push_u32_decimal(&mut md_bytes, match_count);
                match_count = 0;
                rpos += op.len as usize;
            }
        }
    }
    push_u32_decimal(&mut md_bytes, match_count);
    // SAFETY: only ASCII digits, ACGTN and '^' were pushed — all valid UTF-8.
    let md = unsafe { String::from_utf8_unchecked(md_bytes) };
    (nm, md)
}

#[cfg(target_arch = "x86_64")]
struct SwAvx2Scratch {
    prev_h: Vec<std::arch::x86_64::__m256i>,
    prev_e: Vec<std::arch::x86_64::__m256i>,
    cur_h: Vec<std::arch::x86_64::__m256i>,
    cur_e: Vec<std::arch::x86_64::__m256i>,
    // Cell-major trace, transposed read/ref byte gathers.
    trace: Vec<u8>,
    read_cols: Vec<u8>,
    ref_cols: Vec<u8>,
}

#[cfg(target_arch = "x86_64")]
impl SwAvx2Scratch {
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

#[cfg(target_arch = "x86_64")]
thread_local! {
    static SW_SCRATCH: std::cell::RefCell<SwAvx2Scratch> =
        const { std::cell::RefCell::new(SwAvx2Scratch::new()) };
}

/// Dispatcher between the i8 (32-lane) and i16 (16-lane) AVX2 paths.
///
/// When the i8 path is viable for this read length and scoring, take it; the
/// kernel internally per-lane-falls-back to scalar SW if any cell saturates.
/// Otherwise split the batch into 16-wide AVX2 i16 calls.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn sw_dispatch_avx_vnni(
    inputs: &[BatchInput<'_>],
    cfg: AlignmentConfig,
    read_len: usize,
) -> Vec<SwResult> {
    if sw_int8_vnni::int8_path_viable(read_len, cfg) && inputs.len() <= sw_int8_vnni::LANES {
        // SAFETY: target_feature on this function already enabled avx2+avxvnni.
        return unsafe { sw_int8_vnni::sw_batch_int8(inputs, cfg) };
    }
    // Fall through to the i16 path; it caps at 16 lanes per call.
    let mut results = Vec::with_capacity(inputs.len());
    for chunk in inputs.chunks(16) {
        // SAFETY: avx2 is enabled.
        let part = unsafe { sw_batch_avx2(chunk, cfg) };
        results.extend(part);
    }
    results
}

/// 16-lane i16 SIMD Smith-Waterman.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sw_batch_avx2(inputs: &[BatchInput<'_>], cfg: AlignmentConfig) -> Vec<SwResult> {
    use std::arch::x86_64::{_mm256_set1_epi16, _mm256_setzero_si256};

    const LANES: usize = 16;
    let lanes = inputs.len().min(LANES);
    let q_len = inputs[0].read_seq.len();
    let r_len = inputs[0].ref_window.len();

    let v_zero = _mm256_setzero_si256();
    let v_neg = _mm256_set1_epi16(-16384);

    SW_SCRATCH.with(|scratch| {
        let mut s = scratch.borrow_mut();
        // Destructure to get independent &mut borrows of each buffer field.
        let SwAvx2Scratch {
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

        // SAFETY: this caller already has AVX2 enabled via target_feature.
        unsafe {
            sw_batch_avx2_inner(
                inputs, cfg, lanes, q_len, r_len, prev_h, prev_e, cur_h, cur_e, trace, read_cols,
                ref_cols,
            )
        }
    })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn sw_batch_avx2_inner(
    inputs: &[BatchInput<'_>],
    cfg: AlignmentConfig,
    lanes: usize,
    q_len: usize,
    r_len: usize,
    prev_h: &mut Vec<std::arch::x86_64::__m256i>,
    prev_e: &mut Vec<std::arch::x86_64::__m256i>,
    cur_h: &mut Vec<std::arch::x86_64::__m256i>,
    cur_e: &mut Vec<std::arch::x86_64::__m256i>,
    trace: &mut [u8],
    read_cols: &mut [u8],
    ref_cols: &mut [u8],
) -> Vec<SwResult> {
    use std::arch::x86_64::{
        __m128i, __m256i, _mm_cmpeq_epi8, _mm_loadu_si128, _mm_storeu_si128, _mm256_adds_epi16,
        _mm256_and_si256, _mm256_andnot_si256, _mm256_blendv_epi8, _mm256_castsi256_si128,
        _mm256_cmpeq_epi16, _mm256_cmpgt_epi16, _mm256_cvtepi8_epi16, _mm256_max_epi16,
        _mm256_or_si256, _mm256_packus_epi16, _mm256_permute4x64_epi64, _mm256_set1_epi16,
        _mm256_setzero_si256, _mm256_storeu_si256,
    };

    const LANES: usize = 16;

    let v_zero = _mm256_setzero_si256();
    let v_go = _mm256_set1_epi16(-(cfg.gap_open.saturating_add(cfg.gap_extend)) as i16);
    let v_ge = _mm256_set1_epi16(-cfg.gap_extend as i16);
    let v_match = _mm256_set1_epi16(cfg.match_score as i16);
    let v_mism = _mm256_set1_epi16(-cfg.mismatch as i16);
    let v_one = _mm256_set1_epi16(1);
    let v_two = _mm256_set1_epi16(2);
    let v_three = _mm256_set1_epi16(3);
    let v_four = _mm256_set1_epi16(TRACE_E_EXTEND as i16);
    let v_eight = _mm256_set1_epi16(TRACE_F_EXTEND as i16);
    let v_neg = _mm256_set1_epi16(-16384);

    let trace_w = r_len + 1;

    // SIMD best-score / best-position tracking.
    let mut best_v = v_zero;
    let mut best_i_v = v_zero;
    let mut best_j_v = v_zero;
    let mut best_qlen_v = v_zero;
    let mut best_qlen_j_v = v_zero;

    let mut abort_scores = [0i32; LANES];
    for k in 0..lanes {
        abort_scores[k] = inputs[k].abort_score;
    }
    let mut lane_done = [false; LANES];

    for i in 1..=q_len {
        let base = i * LANES;
        for k in 0..lanes {
            read_cols[base + k] = inputs[k].read_seq[i - 1];
        }
    }
    // Same idea for the reference window — gather once into a tightly packed buffer.
    for j in 1..=r_len {
        let base = j * LANES;
        for k in 0..lanes {
            ref_cols[base + k] = inputs[k].ref_window[j - 1];
        }
    }

    let mut best_arr = [0i16; LANES];

    for i in 1..=q_len {
        cur_h[0] = v_zero;
        cur_e[0] = v_neg;
        let mut cur_f = v_neg;

        // Load 16 read bytes (one per lane) into the low 128 bits of a vector.
        let read_v =
            unsafe { _mm_loadu_si128(read_cols.as_ptr().add(i * LANES) as *const __m128i) };

        for j in 1..=r_len {
            // Load 16 reference bytes (one per lane).
            let ref_v =
                unsafe { _mm_loadu_si128(ref_cols.as_ptr().add(j * LANES) as *const __m128i) };

            let eq8 = _mm_cmpeq_epi8(read_v, ref_v);
            let eq16 = _mm256_cvtepi8_epi16(eq8);
            let score_vec = _mm256_blendv_epi8(v_mism, v_match, eq16);

            let h_diag = unsafe { *prev_h.get_unchecked(j - 1) };
            let h_match = _mm256_adds_epi16(h_diag, score_vec);

            let e_from_h = _mm256_adds_epi16(unsafe { *prev_h.get_unchecked(j) }, v_go);
            let e_from_e = _mm256_adds_epi16(unsafe { *prev_e.get_unchecked(j) }, v_ge);
            let e = _mm256_max_epi16(e_from_h, e_from_e);
            let e_ext = _mm256_cmpgt_epi16(e_from_e, e_from_h);

            let f_from_h = _mm256_adds_epi16(unsafe { *cur_h.get_unchecked(j - 1) }, v_go);
            let f_from_f = _mm256_adds_epi16(cur_f, v_ge);
            let f = _mm256_max_epi16(f_from_h, f_from_f);
            let f_ext = _mm256_cmpgt_epi16(f_from_f, f_from_h);

            let h_tmp = _mm256_max_epi16(h_match, e);
            let h_tmp = _mm256_max_epi16(h_tmp, f);
            let h = _mm256_max_epi16(h_tmp, v_zero);

            unsafe {
                *cur_h.get_unchecked_mut(j) = h;
                *cur_e.get_unchecked_mut(j) = e;
            }
            cur_f = f;

            let is_zero = _mm256_cmpeq_epi16(h, v_zero);
            let h_eq_match = _mm256_cmpeq_epi16(h, h_match);
            let h_eq_e = _mm256_cmpeq_epi16(h, e);
            let is_match = _mm256_andnot_si256(is_zero, h_eq_match);
            let is_e = _mm256_andnot_si256(is_zero, _mm256_andnot_si256(h_eq_match, h_eq_e));

            let tr = _mm256_blendv_epi8(v_three, v_two, is_e);
            let tr = _mm256_blendv_epi8(tr, v_one, is_match);
            let tr = _mm256_blendv_epi8(tr, v_zero, is_zero);
            let tr = _mm256_or_si256(tr, _mm256_and_si256(e_ext, v_four));
            let tr = _mm256_or_si256(tr, _mm256_and_si256(f_ext, v_eight));

            // Pack 16 × i16 → 16 × u8 into the low 128 bits of one vector and store.
            let packed = _mm256_packus_epi16(tr, tr);
            let packed = _mm256_permute4x64_epi64::<0xD8>(packed);
            let low = _mm256_castsi256_si128(packed);
            unsafe {
                let dst = trace.as_mut_ptr().add((i * trace_w + j) * LANES) as *mut __m128i;
                _mm_storeu_si128(dst, low);
            }

            // SIMD best-score / best-position update.
            let new_best = _mm256_cmpgt_epi16(h, best_v);
            best_v = _mm256_max_epi16(best_v, h);
            let i_vec = _mm256_set1_epi16(i as i16);
            let j_vec = _mm256_set1_epi16(j as i16);
            best_i_v = _mm256_blendv_epi8(best_i_v, i_vec, new_best);
            best_j_v = _mm256_blendv_epi8(best_j_v, j_vec, new_best);

            if i == q_len {
                let new_qlen_best = _mm256_cmpgt_epi16(h, best_qlen_v);
                best_qlen_v = _mm256_max_epi16(best_qlen_v, h);
                best_qlen_j_v = _mm256_blendv_epi8(best_qlen_j_v, j_vec, new_qlen_best);
            }
        }

        // Scalar abort check (once per row, not per cell).
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

    // Reduce SIMD best vectors to per-lane arrays for traceback.
    let mut best_score_arr = [0i16; LANES];
    let mut best_i_arr = [0i16; LANES];
    let mut best_j_arr = [0i16; LANES];
    let mut best_qlen_score_arr = [0i16; LANES];
    let mut best_qlen_j_arr = [0i16; LANES];
    unsafe {
        _mm256_storeu_si256(best_score_arr.as_mut_ptr() as *mut __m256i, best_v);
        _mm256_storeu_si256(best_i_arr.as_mut_ptr() as *mut __m256i, best_i_v);
        _mm256_storeu_si256(best_j_arr.as_mut_ptr() as *mut __m256i, best_j_v);
        _mm256_storeu_si256(
            best_qlen_score_arr.as_mut_ptr() as *mut __m256i,
            best_qlen_v,
        );
        _mm256_storeu_si256(best_qlen_j_arr.as_mut_ptr() as *mut __m256i, best_qlen_j_v);
    }

    let clip_pen_i16 = cfg.clip_penalty.clamp(0, i16::MAX as i32) as i16;

    let mut results = Vec::with_capacity(lanes);
    for k in 0..lanes {
        let local_score = best_score_arr[k];
        let qlen_score = best_qlen_score_arr[k];
        let (start_i, start_j, bs) =
            if qlen_score > 0 && qlen_score.saturating_add(clip_pen_i16) > local_score {
                (q_len as i32, best_qlen_j_arr[k] as i32, qlen_score as i32)
            } else {
                (
                    best_i_arr[k] as i32,
                    best_j_arr[k] as i32,
                    local_score as i32,
                )
            };

        let read_end = start_i;
        let ref_end = start_j as u32;
        let (cigar, i, j) = traceback_interleaved(trace, trace_w, LANES, k, start_i, start_j);
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
    let v_go = splat(-(cfg.gap_open.saturating_add(cfg.gap_extend)));
    let v_ge = splat(-cfg.gap_extend);

    let mut prev_h: Vec<int32x4_t> = vec![v_zero; r_len + 1];
    let mut prev_e: Vec<int32x4_t> = vec![v_neg; r_len + 1];
    let mut cur_h: Vec<int32x4_t> = vec![v_zero; r_len + 1];
    let mut cur_e: Vec<int32x4_t> = vec![v_neg; r_len + 1];

    let mut trace: Vec<Vec<u8>> = vec![vec![0u8; (q_len + 1) * (r_len + 1)]; lanes];
    let mut best_score = vec![0i32; lanes];
    let mut best_i = vec![0usize; lanes];
    let mut best_j = vec![0usize; lanes];
    let mut best_qlen_score = vec![0i32; lanes];
    let mut best_qlen_j = vec![0usize; lanes];
    let abort_scores: Vec<i32> = inputs.iter().map(|i| i.abort_score).collect();
    let mut lane_done = vec![false; lanes];

    let mut h_buf = [0i32; 4];
    let mut hm_buf = [0i32; 4];
    let mut e_buf = [0i32; 4];
    let mut f_buf = [0i32; 4];
    let mut e_from_h_buf = [0i32; 4];
    let mut e_from_e_buf = [0i32; 4];
    let mut f_from_h_buf = [0i32; 4];
    let mut f_from_f_buf = [0i32; 4];

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
            vst1q_s32(e_from_h_buf.as_mut_ptr(), e_from_h);
            vst1q_s32(e_from_e_buf.as_mut_ptr(), e_from_e);
            vst1q_s32(f_from_h_buf.as_mut_ptr(), f_from_h);
            vst1q_s32(f_from_f_buf.as_mut_ptr(), f_from_f);

            for lane in 0..lanes {
                let idx = i * (r_len + 1) + j;
                let hval = h_buf[lane];
                let mut tr = if hval == 0 {
                    TRACE_STOP
                } else if hval == hm_buf[lane] {
                    TRACE_DIAG
                } else if hval == e_buf[lane] {
                    TRACE_E
                } else {
                    TRACE_F
                };
                if e_from_e_buf[lane] > e_from_h_buf[lane] {
                    tr |= TRACE_E_EXTEND;
                }
                if f_from_f_buf[lane] > f_from_h_buf[lane] {
                    tr |= TRACE_F_EXTEND;
                }
                trace[lane][idx] = tr;
                if hval > best_score[lane] {
                    best_score[lane] = hval;
                    best_i[lane] = i;
                    best_j[lane] = j;
                }
                if i == q_len && hval > best_qlen_score[lane] {
                    best_qlen_score[lane] = hval;
                    best_qlen_j[lane] = j;
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
        let (start_i, start_j, bs) = if best_qlen_score[lane] > 0
            && best_qlen_score[lane].saturating_add(cfg.clip_penalty) > best_score[lane]
        {
            (
                q_len as i32,
                best_qlen_j[lane] as i32,
                best_qlen_score[lane],
            )
        } else {
            (best_i[lane] as i32, best_j[lane] as i32, best_score[lane])
        };

        let read_end = start_i;
        let ref_end = start_j as u32;
        let (cigar, i, j) = traceback_dense(&trace[lane], r_len + 1, start_i, start_j);
        results.push(SwResult {
            ref_start: j as u32,
            ref_end,
            read_start: i,
            read_end,
            score: bs,
            cigar,
            early_abort: lane_done[lane],
        });
    }

    results
}

#[cfg(test)]
mod adaptive_band_tests {
    use super::*;

    fn cfg(bandwidth: i32) -> AlignmentConfig {
        AlignmentConfig {
            match_score: 1,
            mismatch: 4,
            gap_open: 6,
            gap_extend: 1,
            bandwidth,
            xdrop: 1000,
            clip_penalty: 5,
        }
    }

    /// Deterministic pseudo-random ACGT reference (no trivial repeats that would
    /// create alternative equal-scoring placements).
    fn gen_ref(len: usize) -> Vec<u8> {
        let bases = b"ACGT";
        let mut s = 0x1234_5678_9abc_def0u64;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                bases[(s as usize) & 3]
            })
            .collect()
    }

    fn del_len(cigar: &[CigarOp]) -> u32 {
        cigar
            .iter()
            .filter(|o| o.op == CigarKind::Del)
            .map(|o| o.len)
            .sum()
    }

    /// Two 3 bp deletions push the optimal diagonal to +6, beyond a half-width-4
    /// fixed band. Adaptive banding re-centers and recovers the full alignment;
    /// the fixed band clips the tail after the drift exceeds the band.
    #[test]
    fn adaptive_recovers_drifted_indels_fixed_band_clips() {
        let reference = gen_ref(120);
        // read = ref[0..30] + ref[33..60] + ref[63..120]  → two 3 bp deletions.
        let mut read = Vec::new();
        read.extend_from_slice(&reference[0..30]);
        read.extend_from_slice(&reference[33..60]);
        read.extend_from_slice(&reference[63..120]);
        assert_eq!(read.len(), 114);

        let abort = i32::MIN / 8;
        let adaptive = banded_sw_with(&read, &reference, 0, cfg(4), abort, true);
        let fixed = banded_sw_with(&read, &reference, 0, cfg(4), abort, false);

        let aligned = |r: &SwResult| r.read_end - r.read_start;

        // Adaptive: keeps BOTH 3 bp deletions and aligns the whole read. (The former
        // first-row off-by-one that soft-clipped base 0 is fixed, so the leading base is
        // now recovered — allow only a 1 bp tail slack.)
        assert_eq!(
            del_len(&adaptive.cigar),
            6,
            "adaptive should keep both 3bp dels"
        );
        assert!(
            aligned(&adaptive) >= 113,
            "adaptive aligned {} of 114",
            aligned(&adaptive)
        );

        // Fixed band: once cumulative drift exceeds the half-width it loses the
        // tail — fewer dels recovered, less read aligned, lower score.
        assert!(
            del_len(&fixed.cigar) < 6,
            "fixed kept {} del bp",
            del_len(&fixed.cigar)
        );
        assert!(
            adaptive.score > fixed.score + 15,
            "adaptive {} should clearly beat fixed {}",
            adaptive.score,
            fixed.score
        );
        assert!(
            aligned(&adaptive) > aligned(&fixed) + 15,
            "adaptive aligned {} vs fixed {}",
            aligned(&adaptive),
            aligned(&fixed)
        );
    }

    /// On a gapless read the adaptive centre equals `i + offset`, so both modes
    /// must produce byte-identical results — adaptive is a strict superset.
    #[test]
    fn gapless_adaptive_equals_fixed() {
        let reference = gen_ref(120);
        let read = reference[10..90].to_vec(); // 80 bp exact substring
        let abort = i32::MIN / 8;
        let adaptive = banded_sw_with(&read, &reference, 10, cfg(6), abort, true);
        let fixed = banded_sw_with(&read, &reference, 10, cfg(6), abort, false);
        // For a gapless read the adaptive centre coincides with the static one,
        // so results must be byte-identical.
        assert_eq!(adaptive.score, fixed.score);
        assert_eq!(adaptive.cigar, fixed.cigar);
        assert_eq!(adaptive.read_start, fixed.read_start);
        assert_eq!(adaptive.read_end, fixed.read_end);
        assert_eq!(del_len(&adaptive.cigar), 0);
        assert!(
            adaptive.read_end - adaptive.read_start >= 79,
            "should align ~all 80 bp"
        );
    }

    /// Regression for the banded-SW first-row off-by-one. When the read's very first base
    /// matches the reference it must be ALIGNED (M), not dropped as a leading soft-clip.
    /// Before the fix, `prev_diag`'s None branch returned score 0, so the top-left matched
    /// base was never scored: the alignment started at read position 1 and lost one match.
    /// A 60 bp exact substring therefore scored 59 (not 60) and began at read_start 1.
    #[test]
    fn first_base_match_is_not_softclipped() {
        let reference = gen_ref(80);
        let read = reference[0..60].to_vec(); // exact, starts at ref[0], no indel
        let abort = i32::MIN / 8;
        for adaptive in [false, true] {
            let r = banded_sw_with(&read, &reference, 0, cfg(8), abort, adaptive);
            assert_eq!(r.read_start, 0, "first base aligned (adaptive={adaptive})");
            assert_eq!(r.read_end, 60, "full read aligned (adaptive={adaptive})");
            assert_eq!(del_len(&r.cigar), 0);
            assert_eq!(r.score, 60, "all 60 matches scored (adaptive={adaptive})");
            assert_ne!(
                r.cigar.first().map(|o| o.op),
                Some(CigarKind::SoftClip),
                "no leading soft-clip (adaptive={adaptive})"
            );
        }
    }

    /// The ungapped-accept gate `ungapped_beaten_by_gap`: a read whose ungapped mismatches
    /// are strictly cheaper resolved as a deletion must be flagged (routed to WFA traceback),
    /// while a pure-substitution read must not (no gap beats its mismatch cost).
    #[test]
    fn gap_gate_flags_indel_not_snps() {
        let reference = gen_ref(160);
        let c = cfg(50);
        let flip = |b: u8| if b == b'A' { b'C' } else { b'A' };
        let mk_span = |read_len: usize| AnchorSpan {
            ref_id: 0,
            ref_start: 10,
            ref_end: 10 + read_len as u32,
            read_start: 0,
            read_end: read_len as u32,
            strand: Strand::Forward,
        };

        // Pure substitutions: 60 bp read = ref[10..70] with 2 flips. No gap is cheaper
        // than the 2-mismatch cost, so the ungapped accept must stand.
        let mut snp = reference[10..70].to_vec();
        snp[20] = flip(snp[20]);
        snp[40] = flip(snp[40]);
        assert!(
            !ungapped_beaten_by_gap(&snp, &reference, &mk_span(snp.len()), 2, c),
            "SNP-only read must not be flagged"
        );

        // 2 bp deletion at read pos 48 (read = ref[10..58] ++ ref[60..72]); the ungapped
        // placement smears the tail into many mismatches, but a 2 bp D costs only 8 — the
        // gate must detect that a gapped alignment is strictly cheaper.
        let mut del = Vec::new();
        del.extend_from_slice(&reference[10..58]);
        del.extend_from_slice(&reference[60..72]);
        assert!(
            ungapped_beaten_by_gap(&del, &reference, &mk_span(del.len()), 6, c),
            "deletion-bearing read must be flagged for WFA traceback"
        );

        // Exact read: ungapped_mism = 0 short-circuits to not-beaten.
        let exact = reference[10..70].to_vec();
        assert!(!ungapped_beaten_by_gap(&exact, &reference, &mk_span(exact.len()), 0, c));
    }

    #[test]
    fn leading_softclip_is_not_counted_twice_in_md() {
        let chain = AnchorSpan {
            ref_id: 0,
            ref_start: 0,
            ref_end: 4,
            read_start: 2,
            read_end: 6,
            strand: Strand::Forward,
        };
        let sw = SwResult {
            ref_start: 0,
            ref_end: 4,
            read_start: 2,
            read_end: 6,
            score: 4,
            cigar: vec![CigarOp {
                len: 4,
                op: CigarKind::Match,
            }],
            early_abort: false,
        };
        let aln = build_alignment(b"TTACGT", b"ACGT", 0, &chain, false, sw);
        assert_eq!(aln.nm, 0);
        assert_eq!(aln.md, "4");
        assert_eq!(
            aln.cigar,
            vec![
                CigarOp {
                    len: 2,
                    op: CigarKind::SoftClip,
                },
                CigarOp {
                    len: 4,
                    op: CigarKind::Match,
                },
            ]
        );
    }

    #[test]
    fn affine_traceback_score_matches_replayed_cigar() {
        let reference = gen_ref(90);
        let mut read = reference[..40].to_vec();
        read.extend_from_slice(b"TTGCA");
        read.extend_from_slice(&reference[40..]);
        let cfg = cfg(20);
        let result = banded_sw_with(&read, &reference, 0, cfg, i32::MIN / 8, true);

        let mut qi = result.read_start as usize;
        let mut ri = result.ref_start as usize;
        let mut replay = 0i32;
        for op in &result.cigar {
            match op.op {
                CigarKind::Match => {
                    for _ in 0..op.len {
                        replay += if read[qi] == reference[ri] {
                            cfg.match_score
                        } else {
                            -cfg.mismatch
                        };
                        qi += 1;
                        ri += 1;
                    }
                }
                CigarKind::Ins => {
                    replay -= cfg.gap_open + cfg.gap_extend * op.len as i32;
                    qi += op.len as usize;
                }
                CigarKind::Del => {
                    replay -= cfg.gap_open + cfg.gap_extend * op.len as i32;
                    ri += op.len as usize;
                }
                CigarKind::SoftClip => qi += op.len as usize,
                CigarKind::Skipped => ri += op.len as usize,
            }
        }
        assert_eq!(replay, result.score);
        assert!(
            result
                .cigar
                .iter()
                .any(|op| op.op == CigarKind::Ins && op.len >= 5)
        );
    }

    /// `wfa_result_to_alignment` must reject (return `None`) any WFA CIGAR that
    /// does not consume the entire read, rather than emit an alignment whose
    /// CIGAR under-spans the query — that malformed record would later trip the
    /// io SAM emitter's `consumed == seq_len` assertion. Defense-in-depth behind
    /// the `wfa::build_cigar` traceback fix.
    #[test]
    fn wfa_result_rejects_underconsuming_cigar() {
        let read = b"ACGTACGTACGTACGT"; // 16 bp
        let text = b"ACGTACGTACGTACGTACGT"; // 20 bp window
        let cfg = cfg(16);

        // A truncated CIGAR consuming only 4 of 16 query bases (the bug shape).
        let short = wfa::WfaAlignment {
            score: 0,
            cigar: vec![CigarOp {
                len: 4,
                op: CigarKind::Match,
            }],
            text_start: 0,
            text_end: 4,
        };
        assert!(
            wfa_result_to_alignment(read, text, 0, 0, short, cfg, false).is_none(),
            "under-consuming WFA CIGAR must be rejected"
        );

        // A well-formed full-length CIGAR is still accepted.
        let full = wfa::WfaAlignment {
            score: 0,
            cigar: vec![CigarOp {
                len: 16,
                op: CigarKind::Match,
            }],
            text_start: 0,
            text_end: 16,
        };
        let aln = wfa_result_to_alignment(read, text, 0, 0, full, cfg, false)
            .expect("full-length WFA CIGAR must be accepted");
        assert_eq!(aln.read_end, 16);
    }
}
