use crate::types::Alignment;

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

    let primary_mapq = compute_mapq(best, sub, cap);

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
            aln.mapq = compute_mapq(aln.score.max(1), 0, cap) as u8;
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
mod tests {
    use super::*;
    use crate::types::{AlignmentKind, CigarKind, CigarOp, MateInfo};

    fn make_aln(score: i32, read_start: u32, read_end: u32, ref_id: u32) -> Alignment {
        Alignment {
            kind: AlignmentKind::DpAligned,
            ref_id,
            ref_start: 0,
            ref_end: read_end - read_start,
            read_start,
            read_end,
            cigar: vec![CigarOp {
                len: read_end - read_start,
                op: CigarKind::Match,
            }],
            score,
            mapq: 0,
            is_rev: false,
            is_secondary: false,
            is_supplementary: false,
            nm: 0,
            md: "0".to_string(),
            as_score: score,
            xs_score: None,
            xs_strand: None,
            mate: MateInfo::default(),
        }
    }

    fn cfg() -> MapqConfig {
        MapqConfig {
            short_read_len: 300,
            mapq_cap_short: 60,
            mapq_cap_long: 60,
        }
    }

    #[test]
    fn single_alignment_is_primary() {
        let mut alns = vec![make_aln(100, 0, 150, 0)];
        assign_mapq(&mut alns, 150, cfg(), None);
        assert!(!alns[0].is_secondary);
        assert!(!alns[0].is_supplementary);
        assert_eq!(alns[0].mapq, 60);
    }

    #[test]
    fn overlapping_alt_becomes_secondary() {
        // Two alignments both covering 0..150 → 100% overlap
        let mut alns = vec![make_aln(100, 0, 150, 0), make_aln(80, 0, 150, 1)];
        assign_mapq(&mut alns, 150, cfg(), None);
        assert!(!alns[0].is_secondary);
        assert!(!alns[0].is_supplementary);
        assert!(alns[1].is_secondary);
        assert!(!alns[1].is_supplementary);
        assert_eq!(alns[1].mapq, 0);
    }

    #[test]
    fn nonoverlapping_alt_becomes_supplementary() {
        // Long read 0..1000 split: primary covers 0..500, second covers 500..1000 → 0% overlap
        let mut alns = vec![make_aln(300, 0, 500, 0), make_aln(280, 500, 1000, 0)];
        assign_mapq(&mut alns, 1000, cfg(), None);
        assert!(!alns[0].is_secondary && !alns[0].is_supplementary);
        assert!(alns[1].is_supplementary);
        assert!(!alns[1].is_secondary);
        assert!(alns[1].mapq > 0); // supplementary retains MAPQ
    }

    #[test]
    fn partially_overlapping_below_50pct_is_supplementary() {
        // Primary 0..500, second 400..900 → overlap 100bp on min-len 500 → 20%
        let mut alns = vec![make_aln(300, 0, 500, 0), make_aln(280, 400, 900, 0)];
        assign_mapq(&mut alns, 900, cfg(), None);
        assert!(alns[1].is_supplementary);
    }

    #[test]
    fn partially_overlapping_above_50pct_is_secondary() {
        // Primary 0..500, second 100..500 → overlap 400bp on min-len 400 → 100%
        let mut alns = vec![make_aln(300, 0, 500, 0), make_aln(280, 100, 500, 0)];
        assign_mapq(&mut alns, 500, cfg(), None);
        assert!(alns[1].is_secondary);
        assert!(!alns[1].is_supplementary);
    }

    #[test]
    fn xs_score_proxy_grades_single_alignment() {
        let mut aln = make_aln(150, 0, 150, 0);
        aln.xs_score = Some(120);
        let mut alns = vec![aln];
        assign_mapq(&mut alns, 150, cfg(), None);
        assert_eq!(alns[0].mapq, 24);
        assert_eq!(alns[0].xs_score, Some(120));
    }

    #[test]
    fn sub_at_or_below_floor_yields_cap() {
        for sub in [0, 1, 50, 74, 75] {
            let mut aln = make_aln(150, 0, 150, 0);
            aln.xs_score = Some(sub);
            let mut alns = vec![aln];
            assign_mapq(&mut alns, 150, cfg(), None);
            assert_eq!(alns[0].mapq, 60, "sub={} should keep MAPQ at cap", sub);
        }
    }

    #[test]
    fn sub_near_parity_yields_low_mapq() {
        // Sub very close to best → genuinely ambiguous → low MAPQ.
        let mut aln = make_aln(150, 0, 150, 0);
        aln.xs_score = Some(148); // 98.7% of best
        let mut alns = vec![aln];
        assign_mapq(&mut alns, 150, cfg(), None);
        // floor=75, span=75, diff=2 → mapq = 2·60/75 = 1.
        assert!(alns[0].mapq <= 2, "near-parity sub should yield very low MAPQ, got {}", alns[0].mapq);
    }

    #[test]
    fn xs_score_proxy_at_parity_yields_zero() {
        let mut aln = make_aln(150, 0, 150, 0);
        aln.xs_score = Some(150);
        let mut alns = vec![aln];
        assign_mapq(&mut alns, 150, cfg(), None);
        assert_eq!(alns[0].mapq, 0);
    }

    #[test]
    fn weak_xs_score_proxy_keeps_near_cap() {
        let mut aln = make_aln(150, 0, 150, 0);
        aln.xs_score = Some(7); // ~5 % of best
        let mut alns = vec![aln];
        assign_mapq(&mut alns, 150, cfg(), None);
        assert!(
            alns[0].mapq >= 55,
            "weak XS proxy should keep MAPQ near cap, got {}",
            alns[0].mapq
        );
    }

    #[test]
    fn no_xs_score_keeps_cap() {
        let mut alns = vec![make_aln(150, 0, 150, 0)];
        assign_mapq(&mut alns, 150, cfg(), None);
        assert_eq!(alns[0].mapq, 60);
        assert_eq!(alns[0].xs_score, None);
    }

    fn pair_ctx() -> PairMapqContext {
        PairMapqContext {
            insert_mean: 570,
            insert_sd: 155,
            discordant_cap: 10,
        }
    }

    fn paired_mate(
        ref_id: u32,
        mate_pos: u32,
        tlen: i32,
        is_proper: bool,
        mate_is_unmapped: bool,
    ) -> crate::types::MateInfo {
        crate::types::MateInfo {
            is_paired: true,
            is_proper_pair: is_proper,
            mate_is_unmapped,
            mate_is_rev: false,
            is_first_in_pair: true,
            is_second_in_pair: false,
            mate_ref_id: Some(ref_id),
            mate_pos,
            tlen,
        }
    }

    #[test]
    fn discordant_pair_capped_at_discordant_cap() {
        // Proper-pair flag off + |TLEN - mean| > 3σ → cap MAPQ at 10.
        let mut aln = make_aln(150, 0, 150, 0);
        aln.mate = paired_mate(0, 50_000, 50_000, false, false);
        let mut alns = vec![aln];
        assign_mapq(&mut alns, 150, cfg(), Some(pair_ctx()));
        // Would have been MQ 60 without pair context.
        assert_eq!(alns[0].mapq, 10);
    }

    #[test]
    fn proper_pair_not_capped() {
        // proper_pair = true → discordant cap is inactive even with no sd.
        let mut aln = make_aln(150, 0, 150, 0);
        aln.mate = paired_mate(0, 600, 600, true, false);
        let mut alns = vec![aln];
        assign_mapq(&mut alns, 150, cfg(), Some(pair_ctx()));
        assert_eq!(alns[0].mapq, 60);
    }

    #[test]
    fn in_window_pair_not_capped_even_if_proper_flag_missing() {
        let mut aln = make_aln(150, 0, 150, 0);
        // TLEN 600, mean 570, sd 155 → deviation 30, well inside 3σ = 465.
        aln.mate = paired_mate(0, 600, 600, false, false);
        let mut alns = vec![aln];
        assign_mapq(&mut alns, 150, cfg(), Some(pair_ctx()));
        assert_eq!(alns[0].mapq, 60);
    }

    #[test]
    fn unmapped_mate_not_capped() {
        let mut aln = make_aln(150, 0, 150, 0);
        aln.mate = paired_mate(0, 0, 0, false, true);
        let mut alns = vec![aln];
        assign_mapq(&mut alns, 150, cfg(), Some(pair_ctx()));
        assert_eq!(alns[0].mapq, 60);
    }

    #[test]
    fn cross_chr_mate_not_capped_here() {
        let mut aln = make_aln(150, 0, 150, 0);
        // ref_id=0 on the alignment, mate on ref_id=2 — cross-chr.
        aln.mate = paired_mate(2, 0, 0, false, false);
        let mut alns = vec![aln];
        assign_mapq(&mut alns, 150, cfg(), Some(pair_ctx()));
        assert_eq!(alns[0].mapq, 60);
    }

    #[test]
    fn supplementary_inherits_discordant_cap() {
        let mut primary = make_aln(300, 0, 500, 0);
        primary.mate = paired_mate(0, 50_000, 50_000, false, false);
        let mut supp = make_aln(50, 500, 1000, 0);
        supp.mate = paired_mate(0, 50_000, 50_000, false, false);
        let mut alns = vec![primary, supp];
        assign_mapq(&mut alns, 1000, cfg(), Some(pair_ctx()));
        assert_eq!(alns[0].mapq, 10);
        assert!(alns[1].is_supplementary);
        // Supp gets its own MAPQ (compute_mapq(50, 0, 60) = 60), capped to 10.
        assert_eq!(alns[1].mapq, 10);
    }

    #[test]
    fn real_second_alignment_beats_xs_proxy() {
        let mut primary = make_aln(150, 0, 150, 0);
        primary.xs_score = Some(10); // stale proxy
        let secondary = make_aln(120, 0, 150, 1);
        let mut alns = vec![primary, secondary];
        assign_mapq(&mut alns, 150, cfg(), None);
        assert_eq!(alns[0].mapq, 24);
        assert_eq!(alns[0].xs_score, Some(120));
    }

    #[test]
    fn supplementary_does_not_deflate_primary_mapq() {
        let primary = make_aln(80, 0, 80, 0);
        let supp = make_aln(70, 80, 150, 1);
        let mut alns = vec![primary, supp];
        assign_mapq(&mut alns, 150, cfg(), None);
        assert_eq!(
            alns[0].mapq, 60,
            "primary keeps cap when only competition is a disjoint supp"
        );
        assert!(alns[1].is_supplementary);
    }

    #[test]
    fn coregion_secondary_still_deflates_primary_mapq() {
        let primary = make_aln(150, 0, 150, 0);
        let competitor = make_aln(120, 0, 150, 1); // 100% overlap
        let mut alns = vec![primary, competitor];
        assign_mapq(&mut alns, 150, cfg(), None);
        // Same as `real_second_alignment_beats_xs_proxy`: mapq = 24.
        assert_eq!(alns[0].mapq, 24);
        assert!(alns[1].is_secondary);
    }
}
