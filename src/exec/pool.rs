//! Hybrid-aware rayon executor pinned to selected P-cores and E-cores.
//!
//! Pipeline stages currently execute serially. A separate P-pool and E-pool
//! therefore left one class of cores idle for every stage. `DualPool` keeps the
//! routing API but uses one affinity-pinned pool containing the selected P and
//! E logical CPUs, so each parallel stage can consume the complete thread
//! budget.
//!
//! On homogeneous hosts (AMD, older Intel, unknown topology) the pool
//! collapses to a single un-pinned rayon pool covering every logical CPU,
//! so we never penalise non-hybrid runs.
//!
//! Pinning uses [`core_affinity`] inside each worker's `start_handler` so
//! the underlying OS calls (`SetThreadAffinityMask`, `sched_setaffinity`)
//! happen on the worker itself rather than on the spawning thread.

use anyhow::{Context, Result};
use rayon::{ThreadPool, ThreadPoolBuilder};

use super::affinity::{HybridTopology, detect_topology};

/// Override knobs surfaced through the CLI. `None` means "auto".
#[derive(Clone, Copy, Debug, Default)]
pub struct DualPoolConfig {
    /// Force a specific number of P-pool workers. Ignored when the host
    /// is homogeneous.
    pub p_threads: Option<usize>,
    /// Force a specific number of E-pool workers. Ignored when the host
    /// is homogeneous.
    pub e_threads: Option<usize>,
    /// Force the total worker count for the homogeneous fallback (and the
    /// upper bound of `p + e` when topology is auto-detected). `None`
    /// means use every logical CPU.
    pub total_threads: Option<usize>,
}

enum Inner {
    Hybrid {
        pool: ThreadPool,
        p_threads: usize,
        e_threads: usize,
    },
    Homogeneous {
        pool: ThreadPool,
        threads: usize,
    },
}

pub struct DualPool {
    inner: Inner,
}

impl DualPool {
    /// Auto-detect the host topology and build the appropriate pool layout.
    ///
    /// * Hybrid host → one pinned pool sized from the detected P/E layout
    ///   (clamped by `cfg.p_threads` / `cfg.e_threads` when provided).
    /// * Homogeneous host → single unpinned pool sized by `cfg.total_threads`
    ///   (or every logical CPU when unset).
    ///
    /// Set `KIRA_HYBRID_DISABLE=1` to force the homogeneous fallback even
    /// on detected hybrid CPUs — used to A/B benchmark the routing
    /// against the baseline rayon scheduler without a recompile.
    pub fn new(cfg: DualPoolConfig) -> Result<Self> {
        if std::env::var_os("KIRA_HYBRID_DISABLE").is_none() {
            if let Some(topo) = detect_topology() {
                return Self::build_hybrid(topo, cfg);
            }
        }
        Self::build_homogeneous(cfg)
    }

    /// Force a homogeneous (single, unpinned) pool. Used by the tiled
    /// pipeline path until we wire DualPool through tiled too.
    pub fn homogeneous(threads: usize) -> Result<Self> {
        Self::build_homogeneous(DualPoolConfig {
            total_threads: Some(threads),
            ..Default::default()
        })
    }

    fn build_hybrid(topo: HybridTopology, cfg: DualPoolConfig) -> Result<Self> {
        let explicit_p = cfg.p_threads;
        let explicit_e = cfg.e_threads;

        // Decide the target P / E worker counts before we map them to
        // logical IDs. This is where the physical-core bias lives:
        // when the budget fits in one-thread-per-physical-core, we
        // never spawn SMT-sibling workers (avoids pinning two pool
        // threads onto the same execution units, which would kneecap
        // SIMD throughput).
        let n_p_phys = topo.n_p_physical();
        let n_p_logical = topo.p_logical().len();
        let n_e_logical = topo.e_logical().len();

        let (p_target, e_target) = decide_worker_counts(
            cfg.total_threads,
            explicit_p,
            explicit_e,
            n_p_phys,
            n_p_logical,
            n_e_logical,
        );

        let p_ids = pick_p_logical_ids(&topo, p_target);
        let e_ids = pick_e_logical_ids(&topo, e_target);

        // If the user collapsed everything onto one side, fall back to a
        // single pool of that side — keeps semantics simple downstream.
        if p_ids.is_empty() && e_ids.is_empty() {
            return Self::build_homogeneous(cfg);
        }
        if e_ids.is_empty() {
            return Self::build_homogeneous(DualPoolConfig {
                total_threads: Some(p_ids.len()),
                ..Default::default()
            });
        }
        if p_ids.is_empty() {
            return Self::build_homogeneous(DualPoolConfig {
                total_threads: Some(e_ids.len()),
                ..Default::default()
            });
        }

        let p_threads = p_ids.len();
        let e_threads = e_ids.len();
        let mut logical_ids = p_ids;
        logical_ids.extend(e_ids);
        let pool = build_pinned_pool(&logical_ids, "kira-h")?;

        Ok(Self {
            inner: Inner::Hybrid {
                pool,
                p_threads,
                e_threads,
            },
        })
    }

    fn build_homogeneous(cfg: DualPoolConfig) -> Result<Self> {
        let threads = cfg
            .total_threads
            .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
            .unwrap_or(1)
            .max(1);
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("kira-w-{i}"))
            .build()
            .context("build homogeneous thread pool")?;
        Ok(Self {
            inner: Inner::Homogeneous { pool, threads },
        })
    }

    /// Number of selected P-core workers. Equal to total threads on
    /// homogeneous hosts.
    pub fn p_threads(&self) -> usize {
        match &self.inner {
            Inner::Hybrid { p_threads, .. } => *p_threads,
            Inner::Homogeneous { threads, .. } => *threads,
        }
    }

    /// Number of selected E-core workers. Zero on homogeneous hosts.
    pub fn e_threads(&self) -> usize {
        match &self.inner {
            Inner::Hybrid { e_threads, .. } => *e_threads,
            Inner::Homogeneous { .. } => 0,
        }
    }

    pub fn total_threads(&self) -> usize {
        match &self.inner {
            Inner::Hybrid {
                p_threads,
                e_threads,
                ..
            } => p_threads + e_threads,
            Inner::Homogeneous { threads, .. } => *threads,
        }
    }

    pub fn is_hybrid(&self) -> bool {
        matches!(self.inner, Inner::Hybrid { .. })
    }

    /// Run a SIMD-heavy stage on the configured worker pool.
    pub fn install_compute<R, F>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        match &self.inner {
            Inner::Hybrid { pool, .. } => pool.install(f),
            Inner::Homogeneous { pool, .. } => pool.install(f),
        }
    }

    /// Run a light bookkeeping stage on the configured worker pool.
    pub fn install_light<R, F>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        match &self.inner {
            Inner::Hybrid { pool, .. } => pool.install(f),
            Inner::Homogeneous { pool, .. } => pool.install(f),
        }
    }

    /// Used by drivers that want a single pool to install() the outer batch loop.
    pub fn install_driver<R, F>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        self.install_compute(f)
    }
}

/// Decide how many workers to spawn on each pool. Bias toward one
/// worker per physical P-core; only add SMT siblings when the user's
/// total-threads budget exceeds the physical-core count.
///
/// On an i7-12700 (8 P-physical + 4 E-physical = 12 phys, 16 + 4 = 20
/// logical):
///   * `-t 8`  → 4 P + 4 E   (E-cores fit, P-cores fill the rest)
///   * `-t 12` → 8 P + 4 E   (one per physical core — the BWA shape)
///   * `-t 16` → 12 P + 4 E  (first 4 P-cores grow an SMT sibling)
///   * `-t 20` → 16 P + 4 E  (full saturation, every logical CPU pinned)
fn decide_worker_counts(
    total: Option<usize>,
    explicit_p: Option<usize>,
    explicit_e: Option<usize>,
    n_p_phys: usize,
    n_p_logical: usize,
    n_e_logical: usize,
) -> (usize, usize) {
    debug_assert!(n_p_phys >= 1 && n_p_logical >= n_p_phys && n_e_logical >= 1);

    // Honour both explicit overrides — the user opted in.
    if let (Some(p), Some(e)) = (explicit_p, explicit_e) {
        return (p.max(1).min(n_p_logical), e.min(n_e_logical));
    }

    let total = total.unwrap_or(n_p_logical + n_e_logical).max(1);

    // Single-sided overrides.
    if let Some(p) = explicit_p {
        let p_keep = p.max(1).min(n_p_logical);
        let e_keep = total.saturating_sub(p_keep).min(n_e_logical);
        return (p_keep, e_keep);
    }
    if let Some(e) = explicit_e {
        let e_keep = e.min(n_e_logical);
        let p_keep = total.saturating_sub(e_keep).max(1).min(n_p_logical);
        return (p_keep, e_keep);
    }

    // Pure auto path.
    if total >= n_p_logical + n_e_logical {
        // Plenty of budget — use every logical CPU.
        return (n_p_logical, n_e_logical);
    }

    // Keep E-cores intact when they fit; that puts at most 4 threads on
    // the E-pool (E-cores have no SMT) and the rest of the budget on the
    // P-pool, biased toward distinct physical P-cores.
    let e_keep = n_e_logical.min(total.saturating_sub(1));
    let p_budget = total.saturating_sub(e_keep).max(1);
    // Prefer one-per-physical-core until we exhaust them.
    let p_keep = p_budget.min(n_p_logical);
    (p_keep, e_keep)
}

/// Map a P-thread budget to concrete logical CPU IDs, distributing the
/// first `n_p_phys` workers across distinct physical P-cores before
/// adding SMT siblings.
fn pick_p_logical_ids(topo: &HybridTopology, target: usize) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::with_capacity(target);
    let n_phys = topo.n_p_physical();
    if target == 0 || n_phys == 0 {
        return out;
    }
    // Round 1: first SMT sibling of each physical core (preferred).
    for sibs in &topo.p_physical {
        if let Some(&first) = sibs.first() {
            out.push(first);
            if out.len() == target {
                return out;
            }
        }
    }
    // Round 2..: add 2nd, 3rd, … sibling per physical core until we hit
    // the target. On Alder Lake P-cores there's exactly one SMT sibling,
    // so the loop terminates after round 2.
    let max_round = topo.p_physical.iter().map(|s| s.len()).max().unwrap_or(1);
    for round in 1..max_round {
        for sibs in &topo.p_physical {
            if let Some(&id) = sibs.get(round) {
                out.push(id);
                if out.len() == target {
                    return out;
                }
            }
        }
    }
    out
}

fn pick_e_logical_ids(topo: &HybridTopology, target: usize) -> Vec<usize> {
    topo.e_logical().into_iter().take(target).collect()
}

fn build_pinned_pool(logical_ids: &[usize], name_prefix: &str) -> Result<ThreadPool> {
    let ids: Vec<usize> = logical_ids.to_vec();
    let prefix = name_prefix.to_string();
    let n = ids.len();
    let pool = ThreadPoolBuilder::new()
        .num_threads(n)
        .thread_name(move |i| format!("{prefix}-{i}"))
        .start_handler(move |worker_idx| {
            // Map worker index → logical CPU. `core_affinity` is opt-in:
            // failures are non-fatal (we still run, just without pinning).
            if let Some(&cpu) = ids.get(worker_idx) {
                let target = core_affinity::CoreId { id: cpu };
                let _ = core_affinity::set_for_current(target);
            }
        })
        .build()
        .with_context(|| format!("build pinned thread pool '{name_prefix}' with {n} workers"))?;
    Ok(pool)
}

#[cfg(test)]
#[path = "../../tests/unit/exec_pool.rs"]
mod tests;
