//! Algorithm router for the DP fallback path.

use std::sync::OnceLock;

/// Which aligner the router prefers for a given read.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AlignerKind {
    /// Bit-Packed Spectral — 2-bit DNA encoding + SWAR-popcount multi-shift Hamming.
    PackedSpectral,
    /// Spectral Sieve — byte-resolution multi-shift Hamming, *ungapped only*.
    SpectralSieve,
    /// Wavefront alignment, semi-global affine. Handles indels.
    Wfa,
    /// Banded Smith-Waterman (existing AVX2/scalar path).
    BandedSw,
}

/// Per-thread cache of the algorithm preference parsed from `KIRA_ALGO`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AlgoPreference {
    Packed,
    Spectral,
    Wfa,
    Sw,
}

fn parse_algo(s: &str) -> AlgoPreference {
    match s.trim().to_ascii_lowercase().as_str() {
        "packed" | "bitpacked" | "bit-packed" | "swar" => AlgoPreference::Packed,
        "spectral" | "sieve" => AlgoPreference::Spectral,
        "wfa" => AlgoPreference::Wfa,
        "sw" | "banded" | "banded_sw" => AlgoPreference::Sw,
        // Unrecognised → packed
        _ => AlgoPreference::Packed,
    }
}

#[inline]
pub fn algo_preference() -> AlgoPreference {
    static CELL: OnceLock<AlgoPreference> = OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("KIRA_ALGO")
            .map(|s| parse_algo(&s))
            .unwrap_or(AlgoPreference::Packed)
    })
}

/// Default maximum read length for which WFA is the first attempt.
pub const DEFAULT_WFA_MAX_READ_LEN: usize = 300;

/// Default error-rate budget for WFA.
///
/// Raised 15 → 25 to widen the WFA window on HG002/HG38: at 150 bp the
/// budget is `150 * 25/100 * mismatch ≈ 37 * mismatch` score units, which
/// covers the typical real-read indel + 1–2 SNP case that the previous
/// 15% budget was cascading to banded SW. Override with `KIRA_WFA_BUDGET_PCT`.
pub const DEFAULT_WFA_BUDGET_PCT: u32 = 25;

/// Default Myers reject bound: `read_len * pct / 100 + floor`.
///
/// Raised 10 → 15 so the cheap edit-distance pre-screen ahead of WFA stops
/// dropping reads with 1 het SNP + a small indel (≈3–4 edits in 150 bp,
/// hits the prior bound 19; new bound 26). Override with `KIRA_MYERS_BOUND_PCT`.
pub const DEFAULT_MYERS_BOUND_PCT: u32 = 15;
pub const DEFAULT_MYERS_BOUND_FLOOR: u32 = 4;

/// Default Spectral Sieve mismatch budget defaults.
///
/// Floor raised 5 → 8 so very short reads (<100 bp where the percent rounds
/// below floor) still accept reads sitting on a het site plus 1–2 sequencing
/// errors. Percent unchanged (8 % of read_len). Override with
/// `KIRA_SPECTRAL_MISM_PCT` / `KIRA_SPECTRAL_MISM_FLOOR`.
pub const DEFAULT_SPECTRAL_MISM_PCT: u32 = 3;
pub const DEFAULT_SPECTRAL_MISM_FLOOR: u32 = 2;

fn env_u32(name: &'static str, default: u32) -> u32 {
    static CELLS: OnceLock<std::sync::Mutex<std::collections::HashMap<&'static str, u32>>> =
        OnceLock::new();
    let cells = CELLS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut map = cells.lock().unwrap();
    if let Some(v) = map.get(&name) {
        return *v;
    }
    let v = std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(default);
    map.insert(name, v);
    v
}

fn env_usize(name: &'static str, default: usize) -> usize {
    env_u32(name, default as u32) as usize
}

#[inline]
pub fn wfa_max_read_len() -> usize {
    env_usize("KIRA_WFA_MAX_LEN", DEFAULT_WFA_MAX_READ_LEN)
}

#[inline]
pub fn wfa_budget_pct() -> u32 {
    env_u32("KIRA_WFA_BUDGET_PCT", DEFAULT_WFA_BUDGET_PCT)
}

#[inline]
pub fn myers_bound_pct() -> u32 {
    env_u32("KIRA_MYERS_BOUND_PCT", DEFAULT_MYERS_BOUND_PCT)
}

#[inline]
pub fn myers_bound_floor() -> u32 {
    env_u32("KIRA_MYERS_BOUND_FLOOR", DEFAULT_MYERS_BOUND_FLOOR)
}

/// WFA-adaptive pruning drop, from `KIRA_WFA_ADAPTIVE` (antidiagonals).
///
/// `Some(d)` for `d > 0` enables the WFA2 adaptive heuristic — bounding the
/// wavefront width to ~`2d`, giving O(s·d) memory and near-linear time on
/// similar sequences at the cost of exactness. Unset / `0` ⇒ `None` (exact).
#[inline]
pub fn wfa_adaptive_drop() -> Option<i32> {
    static CELL: OnceLock<Option<i32>> = OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("KIRA_WFA_ADAPTIVE")
            .ok()
            .and_then(|s| s.trim().parse::<i32>().ok())
            .filter(|&v| v > 0)
    })
}

/// Free leading-reference bases for ends-free WFA, from `KIRA_WFA_ENDS_FREE`.
/// `0` (default) keeps the read pinned to the window start.
#[inline]
pub fn wfa_ends_free() -> i32 {
    env_u32("KIRA_WFA_ENDS_FREE", 0) as i32
}

/// Leading-reference slack for the fast-path WFA window, from `KIRA_WFA_LEAD`.
/// Extends the WFA text this many bases upstream of the seed-implied window start
/// and marks them free-to-skip, so a 5' deletion *before* the seed is representable
/// (the trailing edge already has `bandwidth` slack; the leading edge had none).
/// `0` (default) reproduces prior behavior exactly. Capped at the bandwidth by the caller.
#[inline]
pub fn wfa_lead() -> i32 {
    env_u32("KIRA_WFA_LEAD", 0) as i32
}

/// Choose the preferred aligner for a read of the given length.
#[inline]
pub fn choose_aligner(read_len: usize) -> AlignerKind {
    if read_len > wfa_max_read_len() {
        return AlignerKind::BandedSw;
    }
    match algo_preference() {
        AlgoPreference::Packed => AlignerKind::PackedSpectral,
        AlgoPreference::Spectral => AlignerKind::SpectralSieve,
        AlgoPreference::Wfa => AlignerKind::Wfa,
        AlgoPreference::Sw => AlignerKind::BandedSw,
    }
}

/// Score budget for WFA, derived from read length and identity expectation.
#[inline]
pub fn wfa_score_budget(read_len: usize, mismatch: i32, gap_open: i32, gap_extend: i32) -> i32 {
    let pct = wfa_budget_pct() as i32;
    let mism_budget = (read_len as i32 * pct) / 100 * mismatch;
    let indel_budget = gap_open + gap_extend * 4;
    mism_budget.max(indel_budget * 3)
}

/// Predicate: should this chain even attempt a fast path?
#[inline]
pub fn fast_path_worth_attempting(chain_score: i32, read_len: usize) -> bool {
    static CELL: OnceLock<bool> = OnceLock::new();
    let gate_enabled = *CELL.get_or_init(|| {
        std::env::var("KIRA_CHAIN_QUALITY_GATE")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .map(|v| v != 0)
            .unwrap_or(false)
    });
    if !gate_enabled {
        return true;
    }
    if chain_score <= 0 || read_len == 0 {
        return false;
    }
    let threshold_pct = env_u32("KIRA_CHAIN_QUALITY_PCT", 70);
    let ratio_x100 = (chain_score as u32).saturating_mul(100) / (read_len as u32);
    ratio_x100 >= threshold_pct
}

/// Maximum mismatches Spectral Sieve will accept for an ungapped alignment.
#[inline]
pub fn spectral_max_mismatches(read_len: usize) -> usize {
    let pct = env_u32("KIRA_SPECTRAL_MISM_PCT", DEFAULT_SPECTRAL_MISM_PCT) as usize;
    let floor = env_u32("KIRA_SPECTRAL_MISM_FLOOR", DEFAULT_SPECTRAL_MISM_FLOOR) as usize;
    ((read_len * pct) / 100).max(floor)
}

/// Bound for the Myers cheap-reject filter.
#[inline]
pub fn myers_reject_bound(read_len: usize) -> usize {
    let pct = myers_bound_pct() as usize;
    let floor = myers_bound_floor() as usize;
    (read_len * pct) / 100 + floor
}
