use crate::chaining::ChainingConfig;
use crate::types::{Anchor, Chain, Strand};

#[derive(Default, Clone, Debug)]
pub struct ChainingStats {
    pub anchors_used: usize,
    pub chains_pruned: usize,
}

/// RMQ/Fenwick O(n log n) chaining; replaces quadratic DP within bands.
pub fn chain_anchors_rmq(
    anchors: &[Anchor],
    cfg: ChainingConfig,
    stats: &mut ChainingStats,
) -> Vec<Chain> {
    if anchors.is_empty() {
        return Vec::new();
    }
    let mut filtered: Vec<Anchor> = anchors.to_vec();
    filtered.sort_by_key(|a| (a.ref_id, u8::from(a.strand), a.ref_start, a.read_start));
    if filtered.len() > cfg.max_anchors {
        filtered.truncate(cfg.max_anchors);
    }

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
        let group = &filtered[start..end];
        // RMQ/Fenwick tuning strategy (default: mild). Switch to aggressive
        // by commenting the line below and uncommenting the aggressive call.
        let tuned = tune_cfg_mild(cfg, group);
        // let tuned = tune_cfg_aggressive(cfg, group);
        let mut group_chains = chain_group_rmq(group, tuned, stats);
        chains.append(&mut group_chains);
        start = end;
    }

    chains.sort_by_key(|c| std::cmp::Reverse(c.score));
    if chains.len() > cfg.max_chains {
        chains.truncate(cfg.max_chains);
    }
    chains
}

fn chain_group_rmq(
    anchors: &[Anchor],
    cfg: ChainingConfig,
    stats: &mut ChainingStats,
) -> Vec<Chain> {
    let n = anchors.len();
    if n == 0 {
        return Vec::new();
    }
    stats.anchors_used += n;

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by_key(|&i| (anchors[i].ref_end, anchors[i].read_end));

    let mut read_ends: Vec<u32> = indices.iter().map(|&i| anchors[i].read_end).collect();
    read_ends.sort_unstable();
    read_ends.dedup();

    let mut fenwick = Fenwick::new(read_ends.len());

    let mut dp = vec![0i32; n];
    let mut prev = vec![None::<usize>; n];
    let mut best_score = 0i32;

    for &idx in indices.iter() {
        let cur = &anchors[idx];
        let pos = lower_bound_u32(&read_ends, cur.read_start);
        let (best_val, best_idx) = fenwick.query(pos);
        let mut score = cur.score;
        if let Some(j) = best_idx {
            let prev_a = &anchors[j];
            let dq = cur.read_start as i32 - prev_a.read_end as i32;
            let dr = cur.ref_start as i32 - prev_a.ref_end as i32;
            if dq >= 0 && dr >= 0 {
                let gap = dq.max(dr);
                if gap as u32 <= cfg.max_dist {
                    let penalty = gap_penalty(gap, cfg);
                    score = (best_val + cur.score - penalty).max(score);
                    if score > cur.score {
                        prev[idx] = Some(j);
                    }
                }
            }
        }
        dp[idx] = score;
        best_score = best_score.max(score);

        // prune dominated anchors early
        if best_score - score > cfg.gap_open * 2 {
            stats.chains_pruned += 1;
            continue;
        }

        let key = lower_bound_u32(&read_ends, cur.read_end);
        fenwick.update(key, score, idx);
    }

    build_chains(anchors, &dp, &prev, cfg.max_chains)
}

fn build_chains(
    anchors: &[Anchor],
    dp: &[i32],
    prev: &[Option<usize>],
    max_chains: usize,
) -> Vec<Chain> {
    let n = anchors.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by_key(|&i| std::cmp::Reverse(dp[i]));
    let mut used = vec![false; n];
    let mut chains = Vec::new();

    for &idx in indices.iter() {
        if used[idx] {
            continue;
        }
        let mut chain_anchors = Vec::new();
        let mut cur = Some(idx);
        while let Some(i) = cur {
            used[i] = true;
            chain_anchors.push(anchors[i].clone());
            cur = prev[i];
        }
        chain_anchors.reverse();
        if chain_anchors.is_empty() {
            continue;
        }
        let ref_id = chain_anchors[0].ref_id;
        let strand = chain_anchors[0].strand;
        let read_start = chain_anchors.first().unwrap().read_start;
        let read_end = chain_anchors.last().unwrap().read_end;
        let ref_start = chain_anchors.first().unwrap().ref_start;
        let ref_end = chain_anchors.last().unwrap().ref_end;
        let score = dp[idx];
        chains.push(Chain {
            anchors: chain_anchors,
            score,
            ref_id,
            read_start,
            read_end,
            ref_start,
            ref_end,
            strand,
        });
        if chains.len() >= max_chains {
            break;
        }
    }
    chains
}

fn gap_penalty(gap: i32, cfg: ChainingConfig) -> i32 {
    if gap <= 0 {
        return 0;
    }
    let log_pen = (gap as f32 + 1.0).ln() * cfg.log_gap;
    cfg.gap_open + cfg.gap_extend * gap + log_pen.round() as i32
}

fn tune_cfg_mild(cfg: ChainingConfig, anchors: &[Anchor]) -> ChainingConfig {
    let mut tuned = cfg;
    let n = anchors.len();
    if n >= 4096 {
        tuned.max_dist = (cfg.max_dist as f32 * 0.85).max(64.0) as u32;
        tuned.gap_open = (cfg.gap_open as f32 * 1.10).round() as i32;
        tuned.log_gap = cfg.log_gap * 1.05;
    } else if n >= 1024 {
        tuned.max_dist = (cfg.max_dist as f32 * 0.90).max(64.0) as u32;
        tuned.gap_open = (cfg.gap_open as f32 * 1.05).round() as i32;
        tuned.log_gap = cfg.log_gap * 1.02;
    }
    tuned
}

#[allow(dead_code)]
fn tune_cfg_aggressive(cfg: ChainingConfig, anchors: &[Anchor]) -> ChainingConfig {
    let mut tuned = cfg;
    let n = anchors.len();
    if n >= 4096 {
        tuned.max_dist = (cfg.max_dist as f32 * 0.70).max(48.0) as u32;
        tuned.gap_open = (cfg.gap_open as f32 * 1.25).round() as i32;
        tuned.log_gap = cfg.log_gap * 1.12;
    } else if n >= 1024 {
        tuned.max_dist = (cfg.max_dist as f32 * 0.80).max(48.0) as u32;
        tuned.gap_open = (cfg.gap_open as f32 * 1.15).round() as i32;
        tuned.log_gap = cfg.log_gap * 1.08;
    }
    tuned
}

fn lower_bound_u32(arr: &[u32], value: u32) -> usize {
    let mut left = 0usize;
    let mut right = arr.len();
    while left < right {
        let mid = (left + right) / 2;
        if arr[mid] < value {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left.min(arr.len().saturating_sub(1))
}

struct Fenwick {
    n: usize,
    tree: Vec<(i32, Option<usize>)>,
}

impl Fenwick {
    fn new(n: usize) -> Self {
        Self {
            n,
            tree: vec![(i32::MIN / 4, None); n + 1],
        }
    }

    fn update(&mut self, mut idx: usize, val: i32, anchor_idx: usize) {
        idx += 1;
        while idx <= self.n {
            if val > self.tree[idx].0 {
                self.tree[idx] = (val, Some(anchor_idx));
            }
            idx += idx & (!idx + 1);
        }
    }

    fn query(&self, mut idx: usize) -> (i32, Option<usize>) {
        idx += 1;
        let mut best = (i32::MIN / 4, None);
        while idx > 0 {
            if self.tree[idx].0 > best.0 {
                best = self.tree[idx];
            }
            idx &= idx - 1;
        }
        best
    }
}

impl From<Strand> for u8 {
    fn from(s: Strand) -> Self {
        match s {
            Strand::Forward => 0,
            Strand::Reverse => 1,
        }
    }
}
