//! Data-driven insert-size estimation for paired-end runs.

use std::sync::RwLock;

use crate::pipeline::pairing::PairedConfig;
use crate::types::Alignment;

/// Number of unique-proper-pair TLEN samples needed before locking in a refined insert estimate.
pub const MIN_SAMPLES: usize = 1024;

/// Scale factor mapping MAD → σ under a Gaussian distribution: `σ ≈ 1.4826 × MAD`.
const MAD_TO_SIGMA: f64 = 1.4826;

/// Multiplier from σ → max insert window for the *configured* insert range.
const WINDOW_SIGMA: f64 = 6.0;

/// One running estimator instance — owned by `Pipeline` and shared across batch threads via.
#[derive(Debug)]
pub struct InsertEstimator {
    /// User-supplied prior.
    prior: PairedConfig,
    /// Absolute TLENs from proper pairs observed so far.
    samples: Vec<u32>,
    /// `Some(cfg)` once `MIN_SAMPLES` collected and re-fitted.
    locked: Option<PairedConfig>,
}

impl InsertEstimator {
    /// Build an estimator seeded with the user-supplied PE config.
    pub fn new(prior: PairedConfig) -> Self {
        Self {
            prior,
            samples: Vec::with_capacity(MIN_SAMPLES * 2),
            locked: None,
        }
    }

    /// Build a `RwLock`-wrapped estimator ready to drop into `Pipeline`.
    pub fn shared(prior: PairedConfig) -> RwLock<Self> {
        RwLock::new(Self::new(prior))
    }

    /// Current best PE config.
    pub fn current(&self) -> PairedConfig {
        self.locked.unwrap_or(self.prior)
    }

    /// `true` once the estimator has finalized a refined estimate from observed data.
    pub fn is_locked(&self) -> bool {
        self.locked.is_some()
    }

    /// Push the absolute TLEN of one proper pair into the sample buffer.
    pub fn push_sample(&mut self, abs_tlen: u32) -> Option<PairedConfig> {
        if self.locked.is_some() {
            return None;
        }
        self.samples.push(abs_tlen);
        if self.samples.len() >= MIN_SAMPLES {
            let refined = self.fit();
            self.locked = Some(refined);
            // Drop samples to free memory — we won't need them again.
            self.samples = Vec::new();
            Some(refined)
        } else {
            None
        }
    }

    /// Convenience: walk a batch of (read, alignments) and push every alignment whose.
    pub fn observe_batch(&mut self, alignments: &[Vec<Alignment>]) -> Option<PairedConfig> {
        if self.locked.is_some() {
            return None;
        }
        let mut latest: Option<PairedConfig> = None;
        for alns in alignments.iter() {
            if alns.len() != 1 {
                continue;
            }
            let primary = &alns[0];
            let m = &primary.mate;
            if !m.is_proper_pair || m.mate_is_unmapped {
                continue;
            }
            if m.tlen <= 0 {
                continue;
            }
            let abs_tlen = m.tlen as u32;
            if let Some(cfg) = self.push_sample(abs_tlen) {
                latest = Some(cfg);
            }
        }
        latest
    }

    /// Fit a refined `PairedConfig` from the accumulated samples using **median + MAD**.
    fn fit(&self) -> PairedConfig {
        debug_assert!(self.samples.len() >= MIN_SAMPLES);
        let mut sorted: Vec<u32> = self.samples.clone();
        sorted.sort_unstable();
        let median = percentile_sorted(&sorted, 0.5);

        // MAD = median of absolute deviations from the median.
        let mut devs: Vec<u32> = sorted
            .iter()
            .map(|&x| (x as i64 - median as i64).unsigned_abs() as u32)
            .collect();
        devs.sort_unstable();
        let mad = percentile_sorted(&devs, 0.5) as f64;
        let sd = (mad * MAD_TO_SIGMA).round().max(1.0);

        let mean_u = median as i64;
        let sd_u = sd as i64;
        let half = (sd_u * WINDOW_SIGMA as i64).max(0);
        let min_new = (mean_u - half).max(0) as u32;
        let max_new = (mean_u + half).max(min_new as i64 + 1) as u32;

        PairedConfig {
            mode: self.prior.mode,
            insert_min: min_new,
            insert_max: max_new,
            insert_mean: mean_u.max(0) as u32,
            insert_sd: sd_u.max(1) as u32,
            estimator_locked: true,
        }
    }
}

/// Sample-percentile helper for an already-sorted slice.
fn percentile_sorted(sorted: &[u32], q: f64) -> u32 {
    if sorted.is_empty() {
        return 0;
    }
    let q = q.clamp(0.0, 1.0);
    let idx = ((sorted.len() as f64 - 1.0) * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(test)]
#[path = "../../tests/unit/pipeline_insert_estimate.rs"]
mod tests;
