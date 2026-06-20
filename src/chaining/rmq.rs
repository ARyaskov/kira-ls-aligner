use crate::chaining::ChainingConfig;
use crate::types::{Anchor, Chain, Strand};

#[derive(Default, Clone, Debug)]
pub struct ChainingStats {
    pub anchors_used: usize,
    pub chains_pruned: usize,
}

/// Bounded predecessor chaining in the style of minimap2's short-range DP.
///
/// A scalar Fenwick maximum is not sufficient here because transition cost
/// depends on both query/reference distance and diagonal drift. We keep a
/// coordinate-sorted window of predecessor candidates and score each valid
/// transition explicitly. `rmq_window` bounds work per anchor.
pub fn chain_anchors_rmq(
    anchors: &[Anchor],
    cfg: ChainingConfig,
    stats: &mut ChainingStats,
) -> Vec<Chain> {
    if anchors.is_empty() {
        return Vec::new();
    }

    let mut filtered = anchors.to_vec();
    if filtered.len() > cfg.max_anchors {
        // Keep informative anchors without preferring low reference
        // coordinates, then restore coordinate order for chaining.
        filtered.select_nth_unstable_by_key(cfg.max_anchors, anchor_rank_key);
        filtered.truncate(cfg.max_anchors);
        stats.chains_pruned += anchors.len() - filtered.len();
    }
    filtered.sort_by_key(anchor_coord_key);

    let mut chains = Vec::new();
    let mut start = 0usize;
    while start < filtered.len() {
        let (ref_id, strand) = (filtered[start].ref_id, filtered[start].strand);
        let mut end = start + 1;
        while end < filtered.len()
            && filtered[end].ref_id == ref_id
            && filtered[end].strand == strand
        {
            end += 1;
        }
        chains.extend(chain_group(&filtered[start..end], cfg, stats));
        start = end;
    }

    chains.sort_by_key(|c| {
        (
            std::cmp::Reverse(c.score),
            c.ref_id,
            u8::from(c.strand),
            c.ref_start,
            c.read_start,
        )
    });
    dedup_near_identical_chains(&mut chains);
    chains.truncate(cfg.max_chains);
    chains
}

fn chain_group(anchors: &[Anchor], cfg: ChainingConfig, stats: &mut ChainingStats) -> Vec<Chain> {
    let n = anchors.len();
    if n == 0 {
        return Vec::new();
    }
    stats.anchors_used += n;

    let mut dp = vec![0i32; n];
    let mut prev = vec![None; n];
    let mut has_successor = vec![false; n];
    let predecessor_limit = cfg.rmq_window.max(1);

    for i in 0..n {
        let cur = &anchors[i];
        dp[i] = cur.score.max(anchor_len(cur) as i32);
        let mut examined = 0usize;
        let scan_limit = predecessor_limit.saturating_mul(4);

        for (scanned, j) in (0..i).rev().enumerate() {
            let candidate = &anchors[j];
            let ref_delta = cur.ref_start.saturating_sub(candidate.ref_start);
            if ref_delta > cfg.max_dist {
                break;
            }
            if examined >= predecessor_limit || scanned >= scan_limit {
                break;
            }

            let Some(transition) = transition_score(candidate, cur, cfg) else {
                continue;
            };
            examined += 1;
            let score = dp[j].saturating_add(transition);
            if score > dp[i] {
                dp[i] = score;
                prev[i] = Some(j);
            }
        }
        if let Some(j) = prev[i] {
            has_successor[j] = true;
        }
    }

    let mut endpoints: Vec<usize> = (0..n).filter(|&i| !has_successor[i]).collect();
    endpoints.sort_by_key(|&i| std::cmp::Reverse(dp[i]));
    if endpoints.is_empty() {
        endpoints.extend(0..n);
        endpoints.sort_by_key(|&i| std::cmp::Reverse(dp[i]));
    }

    endpoints
        .into_iter()
        .take(cfg.max_chains.saturating_mul(2).max(1))
        .filter_map(|end| build_chain(anchors, &dp, &prev, end))
        .collect()
}

fn transition_score(prev: &Anchor, cur: &Anchor, cfg: ChainingConfig) -> Option<i32> {
    if cur.read_start <= prev.read_start || cur.ref_start <= prev.ref_start {
        return None;
    }
    let dq = cur.read_start - prev.read_start;
    let dr = cur.ref_start - prev.ref_start;
    if dq.max(dr) > cfg.max_dist {
        return None;
    }

    let query_overlap = prev.read_end.saturating_sub(cur.read_start);
    let ref_overlap = prev.ref_end.saturating_sub(cur.ref_start);
    let overlap = query_overlap.max(ref_overlap);
    let contribution = anchor_len(cur).saturating_sub(overlap) as i32;
    if contribution <= 0 {
        return None;
    }

    let q_gap = cur.read_start.saturating_sub(prev.read_end) as i32;
    let r_gap = cur.ref_start.saturating_sub(prev.ref_end) as i32;
    let diagonal_error = (q_gap - r_gap).unsigned_abs() as i32;
    let penalty = if diagonal_error == 0 {
        0
    } else {
        let log_penalty = ((diagonal_error + 1) as f32).log2() * cfg.log_gap;
        cfg.gap_open
            .saturating_add(cfg.gap_extend.saturating_mul(diagonal_error))
            .saturating_add(log_penalty.round() as i32)
    };
    Some(contribution.saturating_sub(penalty))
}

fn build_chain(
    anchors: &[Anchor],
    dp: &[i32],
    prev: &[Option<usize>],
    endpoint: usize,
) -> Option<Chain> {
    let mut path = Vec::new();
    let mut cur = Some(endpoint);
    while let Some(i) = cur {
        path.push(anchors[i].clone());
        cur = prev[i];
    }
    path.reverse();
    let first = path.first()?;
    let last = path.last()?;
    Some(Chain {
        score: dp[endpoint],
        ref_id: first.ref_id,
        read_start: first.read_start,
        read_end: last.read_end,
        ref_start: first.ref_start,
        ref_end: last.ref_end,
        strand: first.strand,
        anchors: path,
    })
}

fn dedup_near_identical_chains(chains: &mut Vec<Chain>) {
    let mut kept: Vec<Chain> = Vec::with_capacity(chains.len());
    for chain in chains.drain(..) {
        let duplicate = kept.iter().any(|other| {
            if chain.ref_id != other.ref_id || chain.strand != other.strand {
                return false;
            }
            let q_overlap = interval_overlap(
                chain.read_start,
                chain.read_end,
                other.read_start,
                other.read_end,
            );
            let r_overlap = interval_overlap(
                chain.ref_start,
                chain.ref_end,
                other.ref_start,
                other.ref_end,
            );
            let q_short = (chain.read_end - chain.read_start)
                .min(other.read_end - other.read_start)
                .max(1);
            let r_short = (chain.ref_end - chain.ref_start)
                .min(other.ref_end - other.ref_start)
                .max(1);
            q_overlap.saturating_mul(100) / q_short >= 90
                && r_overlap.saturating_mul(100) / r_short >= 90
        });
        if !duplicate {
            kept.push(chain);
        }
    }
    *chains = kept;
}

#[inline]
fn interval_overlap(a0: u32, a1: u32, b0: u32, b1: u32) -> u32 {
    a1.min(b1).saturating_sub(a0.max(b0))
}

#[inline]
fn anchor_len(anchor: &Anchor) -> u32 {
    anchor
        .read_end
        .saturating_sub(anchor.read_start)
        .min(anchor.ref_end.saturating_sub(anchor.ref_start))
}

fn anchor_coord_key(anchor: &Anchor) -> (u32, u8, u32, u32, u32, u32) {
    (
        anchor.ref_id,
        u8::from(anchor.strand),
        anchor.ref_start,
        anchor.read_start,
        anchor.ref_end,
        anchor.read_end,
    )
}

fn anchor_rank_key(anchor: &Anchor) -> (std::cmp::Reverse<i32>, u64) {
    let mut x = ((anchor.ref_id as u64) << 32)
        ^ anchor.ref_start as u64
        ^ ((anchor.read_start as u64) << 17)
        ^ ((u8::from(anchor.strand) as u64) << 63);
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    (
        std::cmp::Reverse(anchor.score.max(anchor_len(anchor) as i32)),
        x.wrapping_mul(0x94d0_49bb_1331_11eb) ^ (x >> 31),
    )
}

impl From<Strand> for u8 {
    fn from(s: Strand) -> Self {
        match s {
            Strand::Forward => 0,
            Strand::Reverse => 1,
        }
    }
}
