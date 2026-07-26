use crate::chaining::ChainingConfig;
use crate::types::{Anchor, Chain, Strand};

/// Multiplier on `rmq_window` for the raw-candidate scan backstop in `chain_group`.
/// Default 4 (current behavior). Raising it via `KIRA_CHAIN_SCAN_MULT` lets the
/// predecessor scan reach the full `max_dist` window in dense anchor clusters, at the
/// cost of more work per anchor. The `examined` (valid-transition) cap is unaffected.
fn chain_scan_mult() -> usize {
    use std::sync::OnceLock;
    static MULT: OnceLock<usize> = OnceLock::new();
    *MULT.get_or_init(|| {
        std::env::var("KIRA_CHAIN_SCAN_MULT")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v >= 1)
            .unwrap_or(4)
    })
}

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
/// Reusable per-worker working set for [`chain_anchors_rmq`], so chaining does
/// not allocate its DP arrays and anchor paths once per read.
#[derive(Default)]
pub struct ChainScratch {
    filtered: Vec<Anchor>,
    dp: Vec<i32>,
    prev: Vec<u32>,
    has_successor: Vec<bool>,
    endpoints: Vec<u32>,
    /// Anchor indices (into `filtered`) of every candidate chain's path, packed
    /// end to end; a `ChainSpan` refers to its slice by offset and length.
    path_buf: Vec<u32>,
    spans: Vec<ChainSpan>,
}

/// Sentinel for "no predecessor" in [`ChainScratch::prev`]; `u32::MAX` cannot be
/// a valid anchor index because `max_anchors` is far below it.
const NO_PREV: u32 = u32::MAX;

/// A candidate chain before its anchor path is materialised.
#[derive(Clone, Copy)]
struct ChainSpan {
    score: i32,
    ref_id: u32,
    strand: Strand,
    read_start: u32,
    read_end: u32,
    ref_start: u32,
    ref_end: u32,
    path_start: u32,
    path_len: u32,
}

pub fn chain_anchors_rmq(
    anchors: &[Anchor],
    cfg: ChainingConfig,
    stats: &mut ChainingStats,
    scratch: &mut ChainScratch,
) -> Vec<Chain> {
    if anchors.is_empty() {
        return Vec::new();
    }

    scratch.filtered.clear();
    scratch.filtered.extend_from_slice(anchors);
    if scratch.filtered.len() > cfg.max_anchors {
        // Keep informative anchors without preferring low reference
        // coordinates, then restore coordinate order for chaining.
        scratch
            .filtered
            .select_nth_unstable_by_key(cfg.max_anchors, anchor_rank_key);
        scratch.filtered.truncate(cfg.max_anchors);
        stats.chains_pruned += anchors.len() - scratch.filtered.len();
    }
    scratch.filtered.sort_by_key(anchor_coord_key);

    scratch.spans.clear();
    scratch.path_buf.clear();
    let mut start = 0usize;
    while start < scratch.filtered.len() {
        let (ref_id, strand) = (
            scratch.filtered[start].ref_id,
            scratch.filtered[start].strand,
        );
        let mut end = start + 1;
        while end < scratch.filtered.len()
            && scratch.filtered[end].ref_id == ref_id
            && scratch.filtered[end].strand == strand
        {
            end += 1;
        }
        chain_group(start, end, cfg, stats, scratch);
        start = end;
    }

    scratch.spans.sort_by_key(|c| {
        (
            std::cmp::Reverse(c.score),
            c.ref_id,
            u8::from(c.strand),
            c.ref_start,
            c.read_start,
        )
    });
    dedup_near_identical_spans(&mut scratch.spans);
    scratch.spans.truncate(cfg.max_chains);

    // Materialise the surviving chains, and their anchor paths only if asked.
    scratch
        .spans
        .iter()
        .map(|s| Chain {
            score: s.score,
            ref_id: s.ref_id,
            read_start: s.read_start,
            read_end: s.read_end,
            ref_start: s.ref_start,
            ref_end: s.ref_end,
            strand: s.strand,
            anchors: if cfg.keep_anchors {
                scratch.path_buf[s.path_start as usize..(s.path_start + s.path_len) as usize]
                    .iter()
                    .map(|&idx| scratch.filtered[idx as usize].clone())
                    .collect()
            } else {
                Vec::new()
            },
        })
        .collect()
}

/// Chain the anchors in `scratch.filtered[group_start..group_end]`, appending
/// one [`ChainSpan`] per candidate chain (and its path) to `scratch`.
fn chain_group(
    group_start: usize,
    group_end: usize,
    cfg: ChainingConfig,
    stats: &mut ChainingStats,
    scratch: &mut ChainScratch,
) {
    let n = group_end - group_start;
    if n == 0 {
        return;
    }
    stats.anchors_used += n;

    let anchors = &scratch.filtered[group_start..group_end];
    let dp = &mut scratch.dp;
    let prev = &mut scratch.prev;
    let has_successor = &mut scratch.has_successor;
    dp.clear();
    dp.resize(n, 0i32);
    prev.clear();
    prev.resize(n, NO_PREV);
    has_successor.clear();
    has_successor.resize(n, false);
    let predecessor_limit = cfg.rmq_window.max(1);

    for i in 0..n {
        let cur = &anchors[i];
        dp[i] = cur.score.max(anchor_len(cur) as i32);
        let mut examined = 0usize;
        // `scan_limit` is the raw-candidate backstop that prevents O(n²) on dense
        // anchor clusters. At the default ×4 it can be exhausted before `ref_delta`
        // reaches `max_dist`, hiding a true cross-diagonal (indel-spanning) predecessor
        // behind nearer noise anchors. KIRA_CHAIN_SCAN_MULT widens it for A/B on GIAB.
        let scan_limit = predecessor_limit.saturating_mul(chain_scan_mult());

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
                prev[i] = j as u32;
            }
        }
        if prev[i] != NO_PREV {
            has_successor[prev[i] as usize] = true;
        }
    }

    let endpoints = &mut scratch.endpoints;
    endpoints.clear();
    endpoints.extend((0..n as u32).filter(|&i| !has_successor[i as usize]));
    if endpoints.is_empty() {
        endpoints.extend(0..n as u32);
    }
    endpoints.sort_by_key(|&i| std::cmp::Reverse(dp[i as usize]));

    let path_buf = &mut scratch.path_buf;
    let spans = &mut scratch.spans;
    for &end in endpoints
        .iter()
        .take(cfg.max_chains.saturating_mul(2).max(1))
    {
        // Walk the predecessor links into the shared path buffer end-to-start,
        // then reverse that slice in place.
        let path_start = path_buf.len();
        let mut cur = end;
        loop {
            path_buf.push(group_start as u32 + cur);
            if prev[cur as usize] == NO_PREV {
                break;
            }
            cur = prev[cur as usize];
        }
        path_buf[path_start..].reverse();

        let first = &anchors[(path_buf[path_start] as usize) - group_start];
        let last = &anchors[(path_buf[path_buf.len() - 1] as usize) - group_start];
        spans.push(ChainSpan {
            score: dp[end as usize],
            ref_id: first.ref_id,
            strand: first.strand,
            read_start: first.read_start,
            read_end: last.read_end,
            ref_start: first.ref_start,
            ref_end: last.ref_end,
            path_start: path_start as u32,
            path_len: (path_buf.len() - path_start) as u32,
        });
    }
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

/// Drop chains that cover essentially the same read and reference interval as an
/// already-kept, higher-scoring one. Retains order; operates in place.
fn dedup_near_identical_spans(spans: &mut Vec<ChainSpan>) {
    let mut kept = 0usize;
    for i in 0..spans.len() {
        let chain = spans[i];
        let duplicate = spans[..kept].iter().any(|other| {
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
            spans[kept] = chain;
            kept += 1;
        }
    }
    spans.truncate(kept);
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
