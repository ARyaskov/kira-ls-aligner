use crate::types::{Alignment, CigarKind};

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
    if alignments.is_empty() {
        return;
    }
    alignments.sort_by_key(|a| std::cmp::Reverse(a.score));
    let best = alignments[0].score.max(1);
    let primary_read_start = alignments[0].read_start;
    let primary_read_end = alignments[0].read_end;
    let sub_real = alignments.iter().skip(1).find_map(|a| {
        if read_overlap_pct(
            primary_read_start,
            primary_read_end,
            a.read_start,
            a.read_end,
        ) >= 50
        {
            Some(a.score)
        } else {
            None
        }
    });
    let sub = sub_real.unwrap_or_else(|| alignments[0].xs_score.unwrap_or(0));

    let cap = if read_len <= cfg.short_read_len {
        cfg.mapq_cap_short
    } else {
        cfg.mapq_cap_long
    } as i32;

    let occ_cap = repeat_occ_cap(repeat_min_occ) as i32;
    let id_cap = identity_mapq_cap(&alignments[0], read_len) as i32;
    let primary_mapq = compute_mapq(best, sub, cap).min(occ_cap).min(id_cap);

    alignments[0].mapq = primary_mapq as u8;
    alignments[0].xs_score = if sub > 0 { Some(sub) } else { None };
    alignments[0].is_secondary = false;
    alignments[0].is_supplementary = false;
    apply_discordant_cap(&mut alignments[0], pair_ctx);

    for aln in alignments.iter_mut().skip(1) {
        let overlap_pct =
            read_overlap_pct(primary_read_start, primary_read_end, aln.read_start, aln.read_end);
        if overlap_pct < 50 {
            aln.is_supplementary = true;
            aln.is_secondary = false;
            let s_id_cap = identity_mapq_cap(aln, read_len) as i32;
            aln.mapq = compute_mapq(aln.score.max(1), 0, cap).min(occ_cap).min(s_id_cap) as u8;
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
    let Some(mate_ref) = m.mate_ref_id else { return };
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
/// `occ == 1` (a uniquely-placeable seed exists) imposes no cap. `occ >= 2`
/// means every seed also occurs elsewhere, so the read maps to ~`occ` loci and
/// its placement is a guess — bwa-style, such reads get near-zero MAPQ. This is
/// the fix for confident paralog/repeat mismapping (the dominant SNP false-positive
/// source). Disabled with `KIRA_REPEAT_MAPQ=0`.
fn repeat_occ_cap(occ: u32) -> u8 {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    let enabled = *ENABLED.get_or_init(|| {
        // Default OFF: the seed-occurrence MAPQ cap did not improve F1 in validation
        // (the FP-causing reads are min_occ=1). Opt in with KIRA_REPEAT_MAPQ=1.
        std::env::var("KIRA_REPEAT_MAPQ")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("off"))
            .unwrap_or(false)
    });
    if !enabled || occ <= 1 {
        return 60;
    }
    match occ {
        2 => 3,
        3 => 2,
        _ => 0,
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

/// MAPQ ceiling from alignment identity — the bwa `-T` reject expressed as a MAPQ
/// cap. Clean reads keep full MAPQ; low-identity placements (likely wrong locus)
/// are capped so the caller's min-MQ filter drops them. Disabled with KIRA_ID_MAPQ=0.
fn identity_mapq_cap(aln: &Alignment, _read_len: usize) -> u8 {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    let enabled = *ENABLED.get_or_init(|| {
        std::env::var("KIRA_ID_MAPQ")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("off"))
            .unwrap_or(true)
    });
    if !enabled {
        return 60;
    }
    let rate = identity_mismatch_rate(aln);
    // Only the clear garbage (>13% mismatch — the near-random, deeply-negative-score
    // alignments kira otherwise emits at MAPQ 60; bwa never reports these). The
    // paralog band (NM ~5-19) is left to the now-honest competing-locus MAPQ that
    // dp_topk=2 enables (the true copy gets a real second-best score).
    if rate <= 0.13 { 60 } else { 0 }
}

/// Sub-alignment floor as a fraction of the primary score.
const SUB_FLOOR_DIVISOR: i32 = 2;

/// Compute MAPQ from a best/sub score pair under a cap.
fn compute_mapq(best: i32, sub: i32, cap: i32) -> i32 {
    let best = best.max(1);
    let floor = best / SUB_FLOOR_DIVISOR;
    if sub < floor {
        return cap;
    }
    if sub >= best {
        return 0;
    }
    let span = (best - floor).max(1);
    let diff = best - sub;
    (diff * cap / span).clamp(0, cap)
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
    (overlap.saturating_mul(100)) / a_len
}

#[cfg(test)]
#[path = "../../tests/unit/mapq_mod.rs"]
mod tests;
