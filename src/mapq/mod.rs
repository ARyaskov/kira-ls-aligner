use crate::types::{Alignment, AlignmentKind, CigarKind};

/// MAPQ configuration.
#[derive(Clone, Copy, Debug)]
pub struct MapqConfig {
    pub short_read_len: usize,
    pub mapq_cap_short: u8,
    pub mapq_cap_long: u8,
}

/// Insert-size context used to cap MAPQ on discordant pairs.
#[derive(Clone, Copy, Debug)]
pub struct PairMapqContext {
    pub insert_mean: u32,
    pub insert_sd: u32,
    /// MAPQ ceiling for discordant pairs.
    pub discordant_cap: u8,
}

impl Default for PairMapqContext {
    fn default() -> Self {
        Self {
            insert_mean: 200,
            insert_sd: 50,
            discordant_cap: 10,
        }
    }
}

/// Assign MAPQ and primary/secondary/supplementary flags.
pub fn assign_mapq(
    alignments: &mut [Alignment],
    read_len: usize,
    cfg: MapqConfig,
    pair_ctx: Option<PairMapqContext>,
    repeat_min_occ: u32,
) {
    assign_mapq_with_qual(alignments, read_len, None, cfg, pair_ctx, repeat_min_occ);
}

/// Assign MAPQ using FASTQ qualities when they are available.
pub fn assign_mapq_with_qual(
    alignments: &mut [Alignment],
    read_len: usize,
    qual: Option<&[u8]>,
    cfg: MapqConfig,
    pair_ctx: Option<PairMapqContext>,
    repeat_min_occ: u32,
) {
    assign_mapq_impl(
        alignments,
        read_len,
        qual,
        cfg,
        pair_ctx,
        repeat_min_occ,
        false,
    );
}

/// Assign MAPQ while preserving a primary selected by joint paired-end
/// scoring and mate rescue.
pub fn assign_mapq_preserving_primary(
    alignments: &mut [Alignment],
    read_len: usize,
    qual: Option<&[u8]>,
    cfg: MapqConfig,
    pair_ctx: Option<PairMapqContext>,
    repeat_min_occ: u32,
) {
    assign_mapq_impl(
        alignments,
        read_len,
        qual,
        cfg,
        pair_ctx,
        repeat_min_occ,
        true,
    );
}

fn assign_mapq_impl(
    alignments: &mut [Alignment],
    read_len: usize,
    qual: Option<&[u8]>,
    cfg: MapqConfig,
    pair_ctx: Option<PairMapqContext>,
    repeat_min_occ: u32,
    preserve_primary: bool,
) {
    if alignments.is_empty() {
        return;
    }
    if !preserve_primary {
        alignments.sort_by_key(|a| std::cmp::Reverse(a.score));
    }
    let best = alignments[0].score.max(1);
    let (primary_read_start, primary_read_end) = original_read_interval(&alignments[0], read_len);
    let sub_real = alignments
        .iter()
        .skip(1)
        .filter_map(|a| {
            let (read_start, read_end) = original_read_interval(a, read_len);
            (read_overlap_pct(primary_read_start, primary_read_end, read_start, read_end) >= 50)
                .then_some(a.score)
        })
        .max();
    let sub = sub_real.unwrap_or_else(|| alignments[0].xs_score.unwrap_or(0));

    let cap = if read_len <= cfg.short_read_len {
        cfg.mapq_cap_short
    } else {
        cfg.mapq_cap_long
    } as i32;

    let occ_cap = repeat_occ_cap(repeat_min_occ) as i32;
    let id_cap = identity_mapq_cap(&alignments[0], read_len) as i32;
    let qual_cap = quality_mapq_cap(qual) as i32;
    // Mate-rescue placements (single forced window, no genome-wide competitor
    // search) must not claim full confidence. The score model would emit `cap`
    // here (sub=0 for the sole window hit), so apply the rescue ceiling.
    let rescue_cap = if alignments[0].kind == AlignmentKind::Rescued {
        rescue_mapq_cap() as i32
    } else {
        i32::MAX
    };
    let primary_mapq = compute_mapq(best, sub, cap, read_len)
        .min(occ_cap)
        .min(id_cap)
        .min(qual_cap)
        .min(rescue_cap);

    alignments[0].mapq = primary_mapq as u8;
    alignments[0].xs_score = if sub > 0 { Some(sub) } else { None };
    alignments[0].is_secondary = false;
    alignments[0].is_supplementary = false;
    apply_discordant_cap(&mut alignments[0], pair_ctx);

    for aln in alignments.iter_mut().skip(1) {
        let (read_start, read_end) = original_read_interval(aln, read_len);
        let overlap_pct =
            read_overlap_pct(primary_read_start, primary_read_end, read_start, read_end);
        if overlap_pct < 50 {
            aln.is_supplementary = true;
            aln.is_secondary = false;
            let s_id_cap = identity_mapq_cap(aln, read_len) as i32;
            // A supplementary segment is part of the same placement decision;
            // it must not become more confident than the primary.
            aln.mapq = primary_mapq.min(occ_cap).min(s_id_cap).min(qual_cap) as u8;
            aln.xs_score = None;
            apply_discordant_cap(aln, pair_ctx);
        } else {
            // Same read region competing with the primary → secondary.
            aln.is_secondary = true;
            aln.is_supplementary = false;
            aln.mapq = 0;
            aln.xs_score = None;
        }
    }
}

/// If `aln`'s mate context is discordant under `ctx`, cap MAPQ at `ctx.discordant_cap`.
fn apply_discordant_cap(aln: &mut Alignment, ctx: Option<PairMapqContext>) {
    let Some(ctx) = ctx else { return };
    let m = &aln.mate;
    if !m.is_paired || m.mate_is_unmapped {
        return;
    }
    let Some(mate_ref) = m.mate_ref_id else {
        return;
    };
    if mate_ref != aln.ref_id {
        return;
    }
    // Same-chr, paired, mapped — apply the geometry test.
    let is_discordant = if m.is_proper_pair {
        false
    } else if ctx.insert_sd > 0 {
        let abs_tlen = m.tlen.unsigned_abs();
        let mean = ctx.insert_mean as i64;
        let sd = ctx.insert_sd as i64;
        let deviation = (abs_tlen as i64 - mean).abs();
        deviation > 3 * sd
    } else {
        true
    };
    if is_discordant && aln.mapq > ctx.discordant_cap {
        aln.mapq = ctx.discordant_cap;
    }
}

/// MAPQ ceiling implied by a read's seed copy-number (`repeat_min_occ`).
///
/// A small per-seed copy count does not prove that the full read is ambiguous:
/// the intersection of several 2-copy seeds can still identify one locus.
/// Real close competitors are handled by the best/sub score model. This cap is
/// therefore conservative until every usable seed is highly repetitive.
/// Disabled with `KIRA_REPEAT_MAPQ=0`.
fn repeat_occ_cap(occ: u32) -> u8 {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    let enabled = *ENABLED.get_or_init(|| {
        // Enabled by default. `repeat_min_occ=0` means unknown/no usable seed
        // and imposes no cap.
        std::env::var("KIRA_REPEAT_MAPQ")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("off"))
            .unwrap_or(true)
    });
    if !enabled || occ <= 1 {
        return 60;
    }
    match occ {
        0..=4 => 60,
        5..=16 => 40,
        17..=64 => 20,
        65..=128 => 10,
        _ => 3,
    }
}

/// Fraction of aligned (M) columns that are mismatches, excluding indel bases.
/// A correctly placed read is ~0-2% (sequencing errors + a het SNP); >5% signals
/// the read is at the wrong locus (paralog) or is garbage (the NM~100 alignments
/// kira otherwise emits — and even hands MAPQ 60).
fn identity_mismatch_rate(aln: &Alignment) -> f64 {
    let mut gap = 0u32;
    let mut m = 0u32;
    for op in &aln.cigar {
        match op.op {
            CigarKind::Ins | CigarKind::Del => gap += op.len,
            CigarKind::Match => m += op.len,
            _ => {}
        }
    }
    if m == 0 {
        return 0.0;
    }
    let mismatches = aln.nm.saturating_sub(gap) as f64;
    mismatches / m as f64
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum IdMapqMode {
    Off,
    /// Binary cliff: full MAPQ up to 13% mismatch, 0 beyond. Backward-compatible default.
    Binary,
    /// Graded ceiling: full up to 4%, linear down to 0 at 13%, hard 0 beyond.
    Ramp,
}

/// `KIRA_ID_MAPQ` mode. `0`/`off` → Off; `ramp`/`graded` → Ramp; anything else → Binary.
fn id_mapq_mode() -> IdMapqMode {
    use std::sync::OnceLock;
    static MODE: OnceLock<u8> = OnceLock::new();
    let m = *MODE.get_or_init(|| match std::env::var("KIRA_ID_MAPQ") {
        Ok(v) if v == "0" || v.eq_ignore_ascii_case("off") => 0,
        Ok(v) if v.eq_ignore_ascii_case("ramp") || v.eq_ignore_ascii_case("graded") => 2,
        _ => 1,
    });
    match m {
        0 => IdMapqMode::Off,
        2 => IdMapqMode::Ramp,
        _ => IdMapqMode::Binary,
    }
}

/// MAPQ ceiling from alignment identity — the bwa `-T` reject expressed as a MAPQ
/// cap. Clean reads keep full MAPQ; low-identity placements (likely wrong locus)
/// are capped so the caller's min-MQ filter drops them.
///
/// Default (Binary) keeps only the clear garbage (>13% mismatch — the near-random,
/// deeply-negative-score alignments kira otherwise emits at MAPQ 60; bwa never
/// reports these). The paralog band (NM ~5-19) is left to the competing-locus MAPQ
/// that dp_topk=2 enables. `KIRA_ID_MAPQ=ramp` instead grades the paralog band
/// directly (needs GIAB validation vs the binary cliff); `KIRA_ID_MAPQ=0` disables.
fn identity_mapq_cap(aln: &Alignment, _read_len: usize) -> u8 {
    let mode = id_mapq_mode();
    if mode == IdMapqMode::Off {
        return 60;
    }
    let rate = identity_mismatch_rate(aln);
    match mode {
        IdMapqMode::Off => 60,
        IdMapqMode::Binary => {
            if rate <= 0.13 {
                60
            } else {
                0
            }
        }
        IdMapqMode::Ramp => {
            const LO: f64 = 0.04;
            const HI: f64 = 0.13;
            if rate <= LO {
                60
            } else if rate >= HI {
                0
            } else {
                (60.0 * (HI - rate) / (HI - LO)).round().clamp(0.0, 60.0) as u8
            }
        }
    }
}

/// MAPQ ceiling for mate-rescue placements (`AlignmentKind::Rescued`). Default 30 —
/// the historical intent of the dead `aln.mapq = 30` store in pairing.rs. Tunable via
/// `KIRA_RESCUE_MAPQ_CAP`; set to 60 to effectively disable the cap.
fn rescue_mapq_cap() -> u8 {
    use std::sync::OnceLock;
    static CAP: OnceLock<u8> = OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("KIRA_RESCUE_MAPQ_CAP")
            .ok()
            .and_then(|s| s.parse::<u8>().ok())
            .unwrap_or(30)
            .min(60)
    })
}

/// Sub-alignment floor as a fraction of the primary score.
const SUB_FLOOR_DIVISOR: i32 = 2;

/// Compute MAPQ from a best/sub score pair under a cap.
fn compute_mapq(best: i32, sub: i32, cap: i32, read_len: usize) -> i32 {
    let best = best.max(1);
    let floor = best / SUB_FLOOR_DIVISOR;
    if sub < floor {
        return cap;
    }
    if sub >= best {
        return 0;
    }
    let relative_gap = (best - sub) as f64 / best as f64;
    let length_scale = ((read_len.max(1) as f64) / 100.0).sqrt().clamp(0.75, 3.0);
    // Approximate posterior error from score separation. The slope is uncalibrated
    // (default 22.5, intentionally conservative); sweep `KIRA_MAPQ_BETA` against a
    // GIAB truth set to fit it to the empirical mismap rate.
    let p_error = (-mapq_beta() * relative_gap * length_scale)
        .exp()
        .clamp(1e-6, 1.0);
    (-10.0 * p_error.log10()).round().clamp(0.0, cap as f64) as i32
}

/// Slope of the MAPQ posterior-error model (`compute_mapq`). Higher ⇒ steeper ⇒
/// more confident at a given best/sub separation. Default 22.5; override with
/// `KIRA_MAPQ_BETA` to recalibrate. Invalid/non-positive values fall back to 22.5.
fn mapq_beta() -> f64 {
    use std::sync::OnceLock;
    static BETA: OnceLock<f64> = OnceLock::new();
    *BETA.get_or_init(|| {
        std::env::var("KIRA_MAPQ_BETA")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(22.5)
    })
}

/// Percentage overlap between `[a_start..a_end)` and `[b_start..b_end)` as a share of the.
fn read_overlap_pct(a_start: u32, a_end: u32, b_start: u32, b_end: u32) -> u32 {
    let a_len = a_end.saturating_sub(a_start);
    let b_len = b_end.saturating_sub(b_start);
    if a_len == 0 || b_len == 0 {
        return 0;
    }
    let overlap_start = a_start.max(b_start);
    let overlap_end = a_end.min(b_end);
    if overlap_end <= overlap_start {
        return 0;
    }
    let overlap = overlap_end - overlap_start;
    (overlap.saturating_mul(100)) / a_len.min(b_len)
}

#[inline]
fn original_read_interval(aln: &Alignment, read_len: usize) -> (u32, u32) {
    let len = u32::try_from(read_len).unwrap_or(u32::MAX);
    if aln.is_rev {
        (
            len.saturating_sub(aln.read_end),
            len.saturating_sub(aln.read_start),
        )
    } else {
        (aln.read_start, aln.read_end)
    }
}

/// Low-quality reads cannot support a highly confident placement even when the
/// candidate search found no close competitor. FASTQ qualities are Phred+33;
/// Q30 and above leave the standard MAPQ-60 ceiling unchanged.
fn quality_mapq_cap(qual: Option<&[u8]>) -> u8 {
    let Some(qual) = qual.filter(|q| !q.is_empty()) else {
        return 60;
    };
    let sum: u64 = qual
        .iter()
        .map(|&q| q.saturating_sub(33).min(60) as u64)
        .sum();
    let mean_q = (sum / qual.len() as u64) as u8;
    mean_q.saturating_mul(2).min(60)
}

#[cfg(test)]
#[path = "../../tests/unit/mapq_mod.rs"]
mod tests;
