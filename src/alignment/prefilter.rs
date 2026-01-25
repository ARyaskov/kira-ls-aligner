use crate::alignment::AlignmentConfig;
use crate::simd;
use crate::types::{Alignment, AlignmentKind, CigarKind, CigarOp, Strand};

/// Prefilter decision for a chain.
#[derive(Clone, Debug)]
pub enum PrefilterResult {
    Accept(Alignment),
    Reject,
    Fallback,
}

/// Prefilter reason codes for accept/fallback decisions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefilterReason {
    Accepted,
    Disabled,
    NotShortPreset,
    NotTop1,
    MaxAlignmentsNotOne,
    SpanTooSmall,
    MismTooHigh,
    IdentityTooLow,
    IndelSuspect,
    Other(&'static str),
}

impl PrefilterReason {
    pub const COUNT: usize = 10;

    pub fn idx(self) -> usize {
        match self {
            PrefilterReason::Accepted => 0,
            PrefilterReason::Disabled => 1,
            PrefilterReason::NotShortPreset => 2,
            PrefilterReason::NotTop1 => 3,
            PrefilterReason::MaxAlignmentsNotOne => 4,
            PrefilterReason::SpanTooSmall => 5,
            PrefilterReason::MismTooHigh => 6,
            PrefilterReason::IdentityTooLow => 7,
            PrefilterReason::IndelSuspect => 8,
            PrefilterReason::Other(_) => 9,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            PrefilterReason::Accepted => "accept",
            PrefilterReason::Disabled => "disabled",
            PrefilterReason::NotShortPreset => "not_short_preset",
            PrefilterReason::NotTop1 => "not_top1",
            PrefilterReason::MaxAlignmentsNotOne => "max_alignments_not_one",
            PrefilterReason::SpanTooSmall => "span_too_small",
            PrefilterReason::MismTooHigh => "mism_too_high",
            PrefilterReason::IdentityTooLow => "id_too_low",
            PrefilterReason::IndelSuspect => "indel_suspect",
            PrefilterReason::Other(reason) => reason,
        }
    }
}

/// Aggregated metrics from ungapped prefilter.
#[derive(Clone, Copy, Debug)]
pub struct UngappedStats {
    pub len: u16,
    pub mism: u16,
    pub matches: u16,
    pub identity_x10000: u16,
    pub score: i32,
    pub read_start: usize,
    pub ref_start: usize,
}

/// Prefilter outcome with reason for debugging.
#[derive(Clone, Debug)]
pub struct PrefilterOutcome {
    pub result: PrefilterResult,
    pub metrics: Option<UngappedStats>,
    pub reason: PrefilterReason,
}

pub fn identity_x10000(len: usize, mism: usize) -> u16 {
    if len == 0 {
        return 0;
    }
    let matches = len.saturating_sub(mism) as u32;
    let id = matches.saturating_mul(10_000) / (len as u32);
    id.min(u16::MAX as u32) as u16
}

/// Run ungapped SIMD extension + Myers bounded edit distance prefilter.
pub fn prefilter_chain(
    read_seq: &[u8],
    ref_seq: &[u8],
    chain: &crate::alignment::AnchorSpan,
    cfg: AlignmentConfig,
    is_top1: bool,
    accept_enable: bool,
    accept_only_top1: bool,
    accept_span_slack: usize,
    accept_min_identity: f32,
    accept_max_mismatches: usize,
    accept_require_score_margin: i32,
    score_margin: i32,
    long_read_threshold: usize,
    _short_preset: bool,
    multi_alignments_enabled: bool,
) -> PrefilterOutcome {
    let read_len = read_seq.len();
    if read_len == 0 {
        return PrefilterOutcome {
            result: PrefilterResult::Fallback,
            metrics: None,
            reason: PrefilterReason::Other("read_len0"),
        };
    }

    if chain.read_start > chain.read_end || chain.ref_start > chain.ref_end {
        return PrefilterOutcome {
            result: PrefilterResult::Fallback,
            metrics: None,
            reason: PrefilterReason::Other("non_monotonic"),
        };
    }

    let anchor_len = (chain.read_end - chain.read_start) as usize;
    if anchor_len == 0 {
        return PrefilterOutcome {
            result: PrefilterResult::Fallback,
            metrics: None,
            reason: PrefilterReason::Other("anchor0"),
        };
    }

    let (left_ext, right_ext) = ungapped_extend(read_seq, ref_seq, chain, cfg);
    let total_len = anchor_len + left_ext + right_ext;
    if total_len == 0 {
        return PrefilterOutcome {
            result: PrefilterResult::Fallback,
            metrics: None,
            reason: PrefilterReason::Other("span0"),
        };
    }

    let ungapped_read_start = chain.read_start as i32 - left_ext as i32;
    let ungapped_ref_start = chain.ref_start as i32 - left_ext as i32;
    if ungapped_read_start < 0 || ungapped_ref_start < 0 {
        return PrefilterOutcome {
            result: PrefilterResult::Fallback,
            metrics: None,
            reason: PrefilterReason::Other("neg_start"),
        };
    }
    let ungapped_read_start = ungapped_read_start as usize;
    let ungapped_ref_start = ungapped_ref_start as usize;
    if ungapped_read_start + total_len > read_len || ungapped_ref_start + total_len > ref_seq.len()
    {
        return PrefilterOutcome {
            result: PrefilterResult::Fallback,
            metrics: None,
            reason: PrefilterReason::Other("span_oob"),
        };
    }
    let read_ungapped = &read_seq[ungapped_read_start..ungapped_read_start + total_len];
    let ref_ungapped = &ref_seq[ungapped_ref_start..ungapped_ref_start + total_len];

    let hamming = simd::count_mismatches(read_ungapped, ref_ungapped);
    let score = (total_len.saturating_sub(hamming) as i32) * cfg.match_score
        - (hamming as i32) * cfg.mismatch;
    let len_u16 = u16::try_from(total_len).unwrap_or(u16::MAX);
    let mism_u16 = u16::try_from(hamming.min(total_len)).unwrap_or(u16::MAX);
    let matches_u16 = len_u16.saturating_sub(mism_u16);
    let id_x10000 = identity_x10000(total_len, hamming);

    debug_assert!(mism_u16 == 0 || id_x10000 < 10_000);

    let metrics = UngappedStats {
        len: len_u16,
        mism: mism_u16,
        matches: matches_u16,
        identity_x10000: id_x10000,
        score,
        read_start: ungapped_read_start,
        ref_start: ungapped_ref_start,
    };
    if total_len < read_len / 2 {
        return PrefilterOutcome {
            result: PrefilterResult::Reject,
            metrics: Some(metrics),
            reason: PrefilterReason::SpanTooSmall,
        };
    }

    if !accept_enable {
        return PrefilterOutcome {
            result: PrefilterResult::Fallback,
            metrics: Some(metrics),
            reason: PrefilterReason::Disabled,
        };
    }

    if multi_alignments_enabled {
        return PrefilterOutcome {
            result: PrefilterResult::Fallback,
            metrics: Some(metrics),
            reason: PrefilterReason::MaxAlignmentsNotOne,
        };
    }

    if accept_only_top1 && !is_top1 {
        return PrefilterOutcome {
            result: PrefilterResult::Fallback,
            metrics: Some(metrics),
            reason: PrefilterReason::NotTop1,
        };
    }

    if read_len > long_read_threshold {
        return PrefilterOutcome {
            result: PrefilterResult::Fallback,
            metrics: Some(metrics),
            reason: PrefilterReason::NotShortPreset,
        };
    }

    if accept_require_score_margin > 0 && score_margin < accept_require_score_margin {
        return PrefilterOutcome {
            result: PrefilterResult::Fallback,
            metrics: Some(metrics),
            reason: PrefilterReason::Other("margin_fail"),
        };
    }

    if chain.ref_end.saturating_sub(chain.ref_start)
        != chain.read_end.saturating_sub(chain.read_start)
    {
        return PrefilterOutcome {
            result: PrefilterResult::Fallback,
            metrics: Some(metrics),
            reason: PrefilterReason::IndelSuspect,
        };
    }

    let hard_max_mism = ((read_len * 5) + 149) / 150;
    let max_mism = accept_max_mismatches.min(hard_max_mism);
    if hamming > max_mism {
        return PrefilterOutcome {
            result: PrefilterResult::Fallback,
            metrics: Some(metrics),
            reason: PrefilterReason::MismTooHigh,
        };
    }

    let min_id_x10000 = (accept_min_identity * 100.0).round() as u16;
    if id_x10000 < min_id_x10000 || id_x10000 < 9_850 {
        return PrefilterOutcome {
            result: PrefilterResult::Fallback,
            metrics: Some(metrics),
            reason: PrefilterReason::IdentityTooLow,
        };
    }

    let min_len = min_len_required(read_len, id_x10000, hamming, accept_span_slack);
    if total_len < min_len {
        return PrefilterOutcome {
            result: PrefilterResult::Fallback,
            metrics: Some(metrics),
            reason: PrefilterReason::SpanTooSmall,
        };
    }

    let aln = build_ungapped_alignment(read_seq, ref_seq, &metrics, chain, cfg);
    PrefilterOutcome {
        result: PrefilterResult::Accept(aln),
        metrics: Some(metrics),
        reason: PrefilterReason::Accepted,
    }
}

pub(crate) fn min_len_required(
    read_len: usize,
    id_x10000: u16,
    mism: usize,
    span_slack: usize,
) -> usize {
    let mut slack = 0usize;
    if id_x10000 >= 9_933 {
        slack = 15;
    } else if id_x10000 >= 9_900 && mism <= 2 {
        slack = 10;
    } else if id_x10000 >= 9_870 && mism <= 3 {
        slack = 5;
    }
    if span_slack > 0 {
        slack = slack.min(span_slack);
    }
    let min_len = read_len.saturating_sub(slack);
    let floor = (read_len * 85) / 100;
    floor.max(min_len)
}
fn ungapped_extend(
    read_seq: &[u8],
    ref_seq: &[u8],
    chain: &crate::alignment::AnchorSpan,
    cfg: AlignmentConfig,
) -> (usize, usize) {
    let read_len = read_seq.len();
    let ref_len = ref_seq.len();

    let mut left = 0usize;
    let mut right = 0usize;

    let max_left = chain.read_start.min(chain.ref_start) as usize;
    let max_right = (read_len as u32 - chain.read_end).min(ref_len as u32 - chain.ref_end) as usize;

    let block = 32usize;
    let mut score = 0i32;
    let mut best = 0i32;

    let mut i = 0usize;
    while i + block <= max_left {
        let read_off = chain.read_start as usize - i - block;
        let ref_off = chain.ref_start as usize - i - block;
        let mism = simd::count_mismatches(
            &read_seq[read_off..read_off + block],
            &ref_seq[ref_off..ref_off + block],
        ) as i32;
        let matches = block as i32 - mism;
        score += matches * cfg.match_score - mism * cfg.mismatch;
        if score > best {
            best = score;
            left = i + block;
        } else if best - score > cfg.xdrop {
            break;
        }
        i += block;
    }

    score = 0;
    best = 0;
    i = 0;
    while i + block <= max_right {
        let read_off = chain.read_end as usize + i;
        let ref_off = chain.ref_end as usize + i;
        let mism = simd::count_mismatches(
            &read_seq[read_off..read_off + block],
            &ref_seq[ref_off..ref_off + block],
        ) as i32;
        let matches = block as i32 - mism;
        score += matches * cfg.match_score - mism * cfg.mismatch;
        if score > best {
            best = score;
            right = i + block;
        } else if best - score > cfg.xdrop {
            break;
        }
        i += block;
    }

    (left, right)
}

pub(crate) fn build_ungapped_alignment(
    read_seq: &[u8],
    ref_seq: &[u8],
    metrics: &UngappedStats,
    chain: &crate::alignment::AnchorSpan,
    cfg: AlignmentConfig,
) -> Alignment {
    let read_len = read_seq.len();
    let span = metrics.len as usize;
    let read_start = metrics.read_start;
    let read_end = read_start + span;
    let ref_start = metrics.ref_start;
    let ref_end = ref_start + span;

    let mut cigar = Vec::new();
    if read_start > 0 {
        cigar.push(CigarOp {
            len: read_start as u32,
            op: CigarKind::SoftClip,
        });
    }
    cigar.push(CigarOp {
        len: span as u32,
        op: CigarKind::Match,
    });
    if read_end < read_len {
        cigar.push(CigarOp {
            len: (read_len - read_end) as u32,
            op: CigarKind::SoftClip,
        });
    }

    let mut nm = 0u32;
    let mut md = String::new();
    let mut run = 0u32;
    for i in 0..span {
        let qb = read_seq[read_start + i];
        let rb = ref_seq[ref_start + i];
        if qb == rb {
            run += 1;
        } else {
            nm += 1;
            md.push_str(&run.to_string());
            md.push(rb as char);
            run = 0;
        }
    }
    md.push_str(&run.to_string());

    let mism = nm as i32;
    let matches = span as i32 - mism;
    let score = matches * cfg.match_score - mism * cfg.mismatch;

    Alignment {
        kind: AlignmentKind::AcceptedUngapped,
        ref_id: chain.ref_id,
        ref_start: ref_start as u32,
        ref_end: ref_end as u32,
        read_start: read_start as u32,
        read_end: read_end as u32,
        cigar,
        score,
        mapq: 60,
        is_rev: chain.strand == Strand::Reverse,
        is_secondary: false,
        is_supplementary: false,
        nm,
        md,
        as_score: score,
        xs_score: None,
    }
}
