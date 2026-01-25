use crate::types::Alignment;

/// MAPQ configuration.
#[derive(Clone, Copy, Debug)]
pub struct MapqConfig {
    pub short_read_len: usize,
    pub mapq_cap_short: u8,
    pub mapq_cap_long: u8,
}

/// Assign MAPQ and primary/secondary flags.
pub fn assign_mapq(alignments: &mut [Alignment], read_len: usize, cfg: MapqConfig) {
    if alignments.is_empty() {
        return;
    }
    alignments.sort_by_key(|a| std::cmp::Reverse(a.score));
    let best = alignments[0].score.max(1);
    let sub = alignments.get(1).map(|a| a.score).unwrap_or(0);

    let cap = if read_len <= cfg.short_read_len {
        cfg.mapq_cap_short
    } else {
        cfg.mapq_cap_long
    } as i32;

    let diff = (best - sub).max(0) as i32;
    let mut mapq = if sub == 0 {
        cap
    } else {
        let raw = diff * 40 / best;
        raw.min(cap)
    };

    if mapq < 0 {
        mapq = 0;
    }
    alignments[0].mapq = mapq as u8;
    alignments[0].xs_score = if sub > 0 { Some(sub) } else { None };
    alignments[0].is_secondary = false;

    for aln in alignments.iter_mut().skip(1) {
        aln.mapq = 0;
        aln.is_secondary = true;
        aln.xs_score = None;
    }
}
