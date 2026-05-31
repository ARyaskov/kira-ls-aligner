//! Splice-aware alignment for RNA-seq reads.

use rayon::prelude::*;

use crate::alignment::junc_bed::JunctionIndex;
use crate::alignment::{AlignmentConfig, banded_sw_public};
use crate::index::Index;
use crate::seq::reverse_complement;
use crate::types::{
    Alignment, AlignmentKind, CigarKind, CigarOp, Chain, MateInfo, ReadRecord, Strand,
};

/// Configuration for splice-aware alignment.
#[derive(Clone, Copy, Debug)]
pub struct SpliceConfig {
    pub enabled: bool,
    pub min_intron: u32,
    pub max_intron: u32,
    pub strand_policy: SpliceStrandPolicy,
    pub require_signal: bool,
    /// Half-width of the donor/acceptor search window at each anchor gap.
    pub splice_flank: u32,
    /// Minimum size of a recoverable short exon.
    pub min_exon_len: u32,
    /// Minimum trailing/leading polyA run length to soft-clip.
    pub polya_min_len: u32,
}

impl Default for SpliceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_intron: 30,
            max_intron: 200_000,
            strand_policy: SpliceStrandPolicy::Auto,
            require_signal: false,
            splice_flank: 20,
            min_exon_len: 15,
            polya_min_len: 10,
        }
    }
}

/// How to decide the transcript strand for SAM `XS:A:` emission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpliceStrandPolicy {
    Auto,
    Forward,
    Reverse,
    None,
}

/// Detect transcript strand from a donor (intron 5') / acceptor (intron 3') dinucleotide pair.
#[inline]
pub fn detect_splice_strand(donor: [u8; 2], acceptor: [u8; 2]) -> Option<Strand> {
    let d = [donor[0].to_ascii_uppercase(), donor[1].to_ascii_uppercase()];
    let a = [
        acceptor[0].to_ascii_uppercase(),
        acceptor[1].to_ascii_uppercase(),
    ];
    // Forward-strand canonical / near-canonical
    let fwd_canonical = (&d, &a) == (b"GT", b"AG")
        || (&d, &a) == (b"GC", b"AG")
        || (&d, &a) == (b"AT", b"AC");
    if fwd_canonical {
        return Some(Strand::Forward);
    }
    let rev_canonical = (&d, &a) == (b"CT", b"AC")
        || (&d, &a) == (b"CT", b"GC")
        || (&d, &a) == (b"GT", b"AT");
    if rev_canonical {
        return Some(Strand::Reverse);
    }
    None
}

/// Splice-aware alignment of a chain.
#[derive(Clone, Copy, Debug)]
struct RefinedSplice {
    /// Refined donor position (intron 5' edge) on the reference.
    donor_pos: u32,
    /// Refined acceptor position (intron 3' edge, exclusive) on the reference.
    acceptor_pos: u32,
    /// Strand voted by either the BED file or the signal pair.
    strand: Option<Strand>,
    /// How the refinement was decided — useful for splice-aware MAPQ.
    confidence: JunctionConfidence,
    /// Number of read bases that move from the inter-anchor gap to the LEFT exon's match.
    left_shift: u32,
}

/// Per-junction confidence tier — drives splice-aware MAPQ adjustment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JunctionConfidence {
    /// Junction matches a `--junc-bed` annotation (highest confidence).
    AnnotatedBed,
    /// Donor + acceptor are a canonical pair (GT/AG, GC/AG, AT/AC, or their reverse-strand RCs).
    CanonicalSignal,
    /// Neither annotation nor a canonical signal — accepted only because `require_signal` was off.
    NoSignal,
}

impl JunctionConfidence {
    /// Score contribution toward the splice-aware MAPQ bump.
    fn mapq_weight(self) -> i32 {
        match self {
            JunctionConfidence::AnnotatedBed => 10,
            JunctionConfidence::CanonicalSignal => 5,
            JunctionConfidence::NoSignal => 0,
        }
    }
}

/// Window-scan refinement of a splice boundary.
#[allow(clippy::too_many_arguments)]
fn refine_splice_boundary(
    ref_seq: &[u8],
    read: &[u8],
    f1: u32,
    f2: u32,
    r1: u32,
    r2: u32,
    cfg: AlignmentConfig,
    junctions: Option<&JunctionIndex>,
    junc_tolerance: u32,
    ref_id: u32,
) -> Option<RefinedSplice> {
    if f2 <= f1 || r2 < r1 {
        return None;
    }
    let read_gap = r2 - r1;
    let mut best: Option<(i32, RefinedSplice)> = None;

    let ref_len = ref_seq.len() as u32;
    for left_shift in 0..=read_gap {
        let donor_pos = f1 + left_shift;
        let acceptor_pos = f2 - (read_gap - left_shift);
        // Donor needs 2 ref bytes; acceptor needs 2 ref bytes BEFORE it.
        if donor_pos + 2 > ref_len || acceptor_pos < 2 {
            continue;
        }
        let donor = [
            ref_seq[donor_pos as usize],
            ref_seq[donor_pos as usize + 1],
        ];
        let acceptor = [
            ref_seq[acceptor_pos as usize - 2],
            ref_seq[acceptor_pos as usize - 1],
        ];
        let signal_strand = detect_splice_strand(donor, acceptor);
        let bed_strand = junctions
            .and_then(|j| j.lookup(ref_id, donor_pos, acceptor_pos, junc_tolerance));

        let confidence = if bed_strand.is_some() {
            JunctionConfidence::AnnotatedBed
        } else if signal_strand.is_some() {
            JunctionConfidence::CanonicalSignal
        } else {
            JunctionConfidence::NoSignal
        };
        let signal_bonus: i32 = match confidence {
            JunctionConfidence::AnnotatedBed => 20,
            JunctionConfidence::CanonicalSignal => 10,
            JunctionConfidence::NoSignal => 0,
        };

        // Match score across the two extension regions.
        let mut match_score: i32 = 0;
        // Left extension: read[r1..r1+left_shift] vs ref[f1..donor_pos]
        for k in 0..left_shift {
            let rb = read[(r1 + k) as usize];
            let tb = ref_seq[(f1 + k) as usize];
            if rb == tb {
                match_score += cfg.match_score;
            } else {
                match_score -= cfg.mismatch;
            }
        }
        // Right extension: read[r1+left_shift..r2] vs ref[acceptor_pos..f2]
        let right_len = read_gap - left_shift;
        for k in 0..right_len {
            let rb = read[(r1 + left_shift + k) as usize];
            let tb = ref_seq[(acceptor_pos + k) as usize];
            if rb == tb {
                match_score += cfg.match_score;
            } else {
                match_score -= cfg.mismatch;
            }
        }

        let total = signal_bonus + match_score;
        let strand = bed_strand.or(signal_strand);
        let cand = RefinedSplice {
            donor_pos,
            acceptor_pos,
            strand,
            confidence,
            left_shift,
        };
        if best.as_ref().map_or(true, |(s, _)| total > *s) {
            best = Some((total, cand));
        }
    }
    best.map(|(_, r)| r)
}

pub fn align_spliced_chain(
    chain: &Chain,
    read: &ReadRecord,
    ref_seq: &[u8],
    cfg: AlignmentConfig,
    splice_cfg: SpliceConfig,
    junctions: Option<&JunctionIndex>,
    junc_tolerance: u32,
) -> Option<Alignment> {
    if chain.anchors.is_empty() {
        return None;
    }
    let read_len = read.seq.len();
    let is_rev = matches!(chain.strand, Strand::Reverse);
    let read_seq_owned;
    let read_seq: &[u8] = if is_rev {
        read_seq_owned = reverse_complement(&read.seq);
        &read_seq_owned
    } else {
        &read.seq
    };

    let mut cigar: Vec<CigarOp> = Vec::new();
    let mut total_score: i32 = 0;
    let mut total_nm: u32 = 0;
    let mut md_bytes: Vec<u8> = Vec::with_capacity(32);
    let mut md_run: u32 = 0;
    // Per-junction strand votes for the final xs_strand aggregation.
    let mut fwd_votes: u32 = 0;
    let mut rev_votes: u32 = 0;
    let mut splice_confidence_sum: i32 = 0;
    let mut splice_junction_count: u32 = 0;

    let first = &chain.anchors[0];
    let last = chain.anchors.last().unwrap();
    let first_read_start = first.read_start;
    let last_read_end = last.read_end;
    if first_read_start as usize > read_len || last_read_end as usize > read_len {
        return None;
    }

    // Leading soft-clip — bases before the first anchor are unmapped.
    if first_read_start > 0 {
        cigar.push(CigarOp {
            len: first_read_start,
            op: CigarKind::SoftClip,
        });
    }

    for idx in 0..chain.anchors.len() {
        let a = &chain.anchors[idx];
        if a.read_end > read_len as u32 {
            return None;
        }
        if (a.ref_end as usize) > ref_seq.len() {
            return None;
        }
        // Sanity: read/ref ranges must be monotonic.
        if a.read_end < a.read_start || a.ref_end < a.ref_start {
            return None;
        }

        let r_lo = a.read_start as usize;
        let r_hi = a.read_end as usize;
        let t_lo = a.ref_start as usize;
        let t_hi = a.ref_end as usize;
        let mut exon_cfg = cfg;
        exon_cfg.bandwidth = exon_cfg.bandwidth.max(50);
        let sw = banded_sw_public(
            &read_seq[r_lo..r_hi],
            &ref_seq[t_lo..t_hi],
            0,
            exon_cfg,
            i32::MIN / 8,
        );
        total_score += sw.score;

        let mut qpos = 0usize; // index inside read_seq[r_lo..r_hi]
        let mut rpos = 0usize; // index inside ref_seq[t_lo..t_hi]
        for op in &sw.cigar {
            push_cigar(&mut cigar, op.op, op.len);
            match op.op {
                CigarKind::Match => {
                    for _ in 0..op.len {
                        let qb = read_seq[r_lo + qpos];
                        let rb = ref_seq[t_lo + rpos];
                        if qb == rb {
                            md_run += 1;
                        } else {
                            push_decimal(&mut md_bytes, md_run);
                            md_bytes.push(rb);
                            md_run = 0;
                            total_nm += 1;
                        }
                        qpos += 1;
                        rpos += 1;
                    }
                }
                CigarKind::Ins => {
                    total_nm += op.len;
                    qpos += op.len as usize;
                }
                CigarKind::Del => {
                    total_nm += op.len;
                    push_decimal(&mut md_bytes, md_run);
                    md_bytes.push(b'^');
                    for _ in 0..op.len {
                        md_bytes.push(ref_seq[t_lo + rpos]);
                        rpos += 1;
                    }
                    md_run = 0;
                }
                CigarKind::SoftClip => {
                    qpos += op.len as usize;
                }
                CigarKind::Skipped => {
                    push_decimal(&mut md_bytes, md_run);
                    md_run = 0;
                    rpos += op.len as usize;
                }
            }
        }

        // Inter-anchor link.
        if idx + 1 < chain.anchors.len() {
            let next = &chain.anchors[idx + 1];
            if next.read_start < a.read_end || next.ref_start < a.ref_end {
                return None;
            }
            let read_gap = next.read_start - a.read_end;
            let ref_gap = next.ref_start - a.ref_end;

            // Decide: indel, intron, or fallback-D.
            let is_intron_candidate =
                ref_gap >= splice_cfg.min_intron && ref_gap <= splice_cfg.max_intron;
            let small_read_gap = read_gap <= 3; // up to 3 bp wobble allowed

            if is_intron_candidate && small_read_gap {
                let refined = refine_splice_boundary(
                    ref_seq,
                    read_seq,
                    a.ref_end,
                    next.ref_start,
                    a.read_end,
                    next.read_start,
                    cfg,
                    junctions,
                    junc_tolerance,
                    chain.ref_id,
                );
                let (donor_pos, acceptor_pos, resolved_strand, confidence, left_shift) =
                    match refined {
                        Some(r) => (
                            r.donor_pos,
                            r.acceptor_pos,
                            r.strand,
                            r.confidence,
                            r.left_shift,
                        ),
                        None => {
                            (a.ref_end, next.ref_start, None, JunctionConfidence::NoSignal, 0)
                        }
                    };
                let want_n = match confidence {
                    JunctionConfidence::AnnotatedBed => true,
                    JunctionConfidence::CanonicalSignal => true,
                    JunctionConfidence::NoSignal => !splice_cfg.require_signal,
                };
                if let Some(s) = resolved_strand {
                    match s {
                        Strand::Forward => fwd_votes += 1,
                        Strand::Reverse => rev_votes += 1,
                    }
                }
                if want_n {
                    let right_shift = read_gap - left_shift;
                    let intron_len = acceptor_pos - donor_pos;

                    if left_shift > 0 {
                        for k in 0..left_shift {
                            let qb = read_seq[(a.read_end + k) as usize];
                            let tb = ref_seq[(a.ref_end + k) as usize];
                            if qb == tb {
                                md_run += 1;
                                total_score += cfg.match_score;
                            } else {
                                push_decimal(&mut md_bytes, md_run);
                                md_bytes.push(tb);
                                md_run = 0;
                                total_nm += 1;
                                total_score -= cfg.mismatch;
                            }
                        }
                        push_cigar(&mut cigar, CigarKind::Match, left_shift);
                    }

                    // The intron: N op, closes MD run.
                    push_cigar(&mut cigar, CigarKind::Skipped, intron_len);
                    push_decimal(&mut md_bytes, md_run);
                    md_run = 0;

                    // Right extension as Match.
                    if right_shift > 0 {
                        for k in 0..right_shift {
                            let qb = read_seq[(a.read_end + left_shift + k) as usize];
                            let tb = ref_seq[(acceptor_pos + k) as usize];
                            if qb == tb {
                                md_run += 1;
                                total_score += cfg.match_score;
                            } else {
                                push_decimal(&mut md_bytes, md_run);
                                md_bytes.push(tb);
                                md_run = 0;
                                total_nm += 1;
                                total_score -= cfg.mismatch;
                            }
                        }
                        push_cigar(&mut cigar, CigarKind::Match, right_shift);
                    }

                    // Track confidence for splice-aware MAPQ.
                    splice_confidence_sum += confidence.mapq_weight();
                    splice_junction_count += 1;
                    continue;
                }
                // Fall through to D if signal required and not found.
            }
            // Regular indel handling (small intron / no signal).
            if read_gap > 0 {
                push_cigar(&mut cigar, CigarKind::Ins, read_gap);
                total_nm += read_gap;
            }
            if ref_gap > 0 {
                push_cigar(&mut cigar, CigarKind::Del, ref_gap);
                total_nm += ref_gap;
                push_decimal(&mut md_bytes, md_run);
                md_bytes.push(b'^');
                // Emit reference bases — bounded loop for safety.
                let rstart = a.ref_end as usize;
                let rend = (rstart + ref_gap as usize).min(ref_seq.len());
                for k in rstart..rend {
                    md_bytes.push(ref_seq[k]);
                }
                md_run = 0;
            }
        }
    }

    // Trailing soft-clip.
    if (last_read_end as usize) < read_len {
        cigar.push(CigarOp {
            len: read_len as u32 - last_read_end,
            op: CigarKind::SoftClip,
        });
    }
    push_decimal(&mut md_bytes, md_run);

    let xs_strand = match splice_cfg.strand_policy {
        SpliceStrandPolicy::None => None,
        SpliceStrandPolicy::Forward => {
            if fwd_votes + rev_votes > 0 {
                Some(Strand::Forward)
            } else {
                None
            }
        }
        SpliceStrandPolicy::Reverse => {
            if fwd_votes + rev_votes > 0 {
                Some(Strand::Reverse)
            } else {
                None
            }
        }
        SpliceStrandPolicy::Auto => {
            if fwd_votes > rev_votes {
                Some(Strand::Forward)
            } else if rev_votes > fwd_votes {
                Some(Strand::Reverse)
            } else {
                None
            }
        }
    };

    // SAFETY: only ASCII digits, ACGTN and '^' were pushed.
    let md = unsafe { String::from_utf8_unchecked(md_bytes) };

    let splice_score_boost = splice_confidence_sum;
    let total_score = total_score + splice_score_boost;

    // Build the alignment, then run polyA trimming as a final pass.
    let aln = Alignment {
        kind: AlignmentKind::DpAligned,
        ref_id: chain.ref_id,
        ref_start: first.ref_start,
        ref_end: last.ref_end,
        read_start: first_read_start,
        read_end: last_read_end,
        cigar,
        score: total_score,
        mapq: 0,
        is_rev,
        is_secondary: false,
        is_supplementary: false,
        nm: total_nm,
        md,
        as_score: total_score,
        xs_score: None,
        xs_strand,
        mate: MateInfo::default(),
    };
    let _ = splice_junction_count; // currently unused — reserved for future per-junction MAPQ
    let aln = trim_polya(aln, read_seq, splice_cfg.polya_min_len);
    Some(aln)
}

/// Detect and soft-clip a leading or trailing polyA / polyT run.
fn trim_polya(aln: Alignment, read_seq: &[u8], min_len: u32) -> Alignment {
    if min_len == 0 {
        return aln;
    }
    let read_len = read_seq.len() as u32;
    if aln.read_end <= aln.read_start {
        return aln;
    }
    let (look_at_end, target_base) = if aln.is_rev {
        (false, b'T')
    } else {
        (true, b'A')
    };

    // Count the run.
    let mut run: u32 = 0;
    if look_at_end {
        let mut i = aln.read_end;
        while i > aln.read_start {
            i -= 1;
            if read_seq[i as usize] == target_base {
                run += 1;
            } else {
                break;
            }
        }
    } else {
        let mut i = aln.read_start;
        while i < aln.read_end {
            if read_seq[i as usize] == target_base {
                run += 1;
                i += 1;
            } else {
                break;
            }
        }
    }
    if run < min_len {
        return aln;
    }

    let Alignment {
        kind,
        ref_id,
        mut ref_start,
        mut ref_end,
        mut read_start,
        mut read_end,
        cigar,
        score,
        mapq,
        is_rev,
        is_secondary,
        is_supplementary,
        nm,
        md,
        as_score,
        xs_score,
        xs_strand,
        mate,
    } = aln;

    let mut new_cigar: Vec<CigarOp> = Vec::with_capacity(cigar.len() + 1);
    if look_at_end {
        // Trim from the end.
        let mut peel_remaining: u32 = run;
        let mut iter = cigar.into_iter().rev();
        let mut tail_softclip: u32 = 0;
        let mut tail_ops: Vec<CigarOp> = Vec::new();
        for op in iter.by_ref() {
            if peel_remaining == 0 {
                tail_ops.push(op);
                continue;
            }
            match op.op {
                CigarKind::Match | CigarKind::Ins => {
                    let consume = peel_remaining.min(op.len);
                    tail_softclip += consume;
                    peel_remaining -= consume;
                    let remaining = op.len - consume;
                    if op.op == CigarKind::Match {
                        // ref also shrinks
                        ref_end = ref_end.saturating_sub(consume);
                    }
                    if remaining > 0 {
                        tail_ops.push(CigarOp { len: remaining, op: op.op });
                    }
                }
                CigarKind::Del | CigarKind::Skipped => {
                    if op.op == CigarKind::Del {
                        ref_end = ref_end.saturating_sub(op.len);
                    } else {
                        ref_end = ref_end.saturating_sub(op.len);
                    }
                }
                CigarKind::SoftClip => {
                    // Existing soft-clip — fold in.
                    tail_softclip += op.len;
                }
            }
        }
        // Reverse tail_ops to original order.
        tail_ops.reverse();
        for op in tail_ops {
            new_cigar.push(op);
        }
        // Coalesce trailing soft-clip with the polyA peel.
        let total_sc = tail_softclip;
        if total_sc > 0 {
            new_cigar.push(CigarOp { len: total_sc, op: CigarKind::SoftClip });
        }
        // Adjust read_end.
        let _ = read_len;
        read_end = read_end.saturating_sub(run);
        let trailing_unclipped = read_len.saturating_sub(read_end);
        if trailing_unclipped > total_sc {
            // adjust the last SoftClip
            if let Some(last) = new_cigar.last_mut() {
                if last.op == CigarKind::SoftClip {
                    last.len = trailing_unclipped;
                }
            }
        }
    } else {
        // Trim from the start (reverse alignment, polyT prefix).
        let mut peel_remaining: u32 = run;
        let mut head_softclip: u32 = 0;
        let mut head_ops: Vec<CigarOp> = Vec::new();
        let mut consumed_head = false;
        for op in cigar.into_iter() {
            if peel_remaining == 0 {
                head_ops.push(op);
                continue;
            }
            match op.op {
                CigarKind::SoftClip if !consumed_head => {
                    head_softclip += op.len;
                }
                CigarKind::Match | CigarKind::Ins => {
                    consumed_head = true;
                    let consume = peel_remaining.min(op.len);
                    head_softclip += consume;
                    peel_remaining -= consume;
                    let remaining = op.len - consume;
                    if op.op == CigarKind::Match {
                        ref_start = ref_start.saturating_add(consume);
                    }
                    if remaining > 0 {
                        head_ops.push(CigarOp { len: remaining, op: op.op });
                    }
                }
                CigarKind::Del | CigarKind::Skipped => {
                    consumed_head = true;
                    ref_start = ref_start.saturating_add(op.len);
                }
                _ => {
                    head_ops.push(op);
                }
            }
        }
        if head_softclip > 0 {
            new_cigar.push(CigarOp { len: head_softclip, op: CigarKind::SoftClip });
        }
        new_cigar.extend(head_ops);
        read_start = read_start.saturating_add(run);
    }

    Alignment {
        kind,
        ref_id,
        ref_start,
        ref_end,
        read_start,
        read_end,
        cigar: new_cigar,
        score,
        mapq,
        is_rev,
        is_secondary,
        is_supplementary,
        nm,
        md,
        as_score,
        xs_score,
        xs_strand,
        mate,
    }
}

/// Splice-aware batch alignment: take a `ChainBatch` and produce a fully-populated `AlignBatch` by.
pub fn splice_align_batch(
    input: crate::pipeline::stage3_chaining::ChainBatch,
    index: &Index,
    align_cfg: AlignmentConfig,
    splice_cfg: SpliceConfig,
    junctions: Option<&JunctionIndex>,
    junc_tolerance: u32,
) -> crate::pipeline::stage4_alignment::AlignBatch {
    let reads = input.reads;
    let chains = input.chains;

    const CHIMERA_TOPK: usize = 3;
    let alignments: Vec<Vec<Alignment>> = reads
        .par_iter()
        .zip(chains.par_iter())
        .map(|(read, chain_list)| {
            let mut out = Vec::new();
            for chain in chain_list.iter().take(CHIMERA_TOPK) {
                let ref_seq = index.ref_bases(chain.ref_id as usize);
                if let Some(aln) = align_spliced_chain(
                    chain,
                    read,
                    ref_seq,
                    align_cfg,
                    splice_cfg,
                    junctions,
                    junc_tolerance,
                ) {
                    out.push(aln);
                }
            }
            out
        })
        .collect();

    let unmapped_mate_info = vec![None; reads.len()];
    let mut stats = crate::pipeline::stage4_alignment::AlignmentBatchStats::default();
    stats.reads = reads.len();
    stats.chains_used = alignments.iter().map(|a| a.len()).sum();
    crate::pipeline::stage4_alignment::AlignBatch {
        reads,
        alignments,
        unmapped_mate_info,
        stats,
    }
}

/// Append a CIGAR op, coalescing consecutive same-kind ops.
fn push_cigar(out: &mut Vec<CigarOp>, op: CigarKind, len: u32) {
    if len == 0 {
        return;
    }
    if let Some(last) = out.last_mut() {
        if last.op == op {
            last.len += len;
            return;
        }
    }
    out.push(CigarOp { len, op });
}

/// Append a decimal number to a byte buffer (used for MD construction).
fn push_decimal(out: &mut Vec<u8>, mut v: u32) {
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
    for k in (0..i).rev() {
        out.push(tmp[k]);
    }
}

#[cfg(test)]
#[path = "../../tests/unit/alignment_splice.rs"]
mod tests;
