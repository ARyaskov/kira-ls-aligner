//! Data-driven insert-size estimation for paired-end runs.

use std::sync::RwLock;

use crate::pipeline::pairing::PairedConfig;
use crate::types::Alignment;

/// Number of unique same-contig FR-pair TLEN samples needed before locking
/// in a refined insert estimate. Large enough to span several batches so a
/// length- or position-sorted FASTQ does not decide the estimate on its own.
pub const MIN_SAMPLES: usize = 4096;

/// Multiplier from σ → proper-pair window half-width (bwa-mem uses 4σ).
const WINDOW_SIGMA: f64 = 4.0;

/// σ floor as a fraction of the mean and in bases: a PCR-free or simulated
/// library with zero spread must not collapse the proper-pair and rescue
/// windows to a few bases.
const SD_FLOOR_FRACTION: f64 = 0.05;
const SD_FLOOR_BASES: f64 = 5.0;

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
    /// Build an estimator seeded with the user-supplied PE config. A prior
    /// that is already locked (bwa-mem `-I`) is final: nothing is sampled
    /// and the run never replaces it.
    pub fn new(prior: PairedConfig) -> Self {
        let locked = prior.estimator_locked.then_some(prior);
        Self {
            prior,
            samples: if locked.is_some() {
                Vec::new()
            } else {
                Vec::with_capacity(MIN_SAMPLES * 2)
            },
            locked,
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

    /// Walk a batch and push the fragment length of every read that is the
    /// forward mate of a uniquely placed, same-contig FR pair. The
    /// proper-pair flag is deliberately *not* required: it is decided by the
    /// prior window, so sampling only proper pairs would censor the fit at
    /// the prior (a 1.5 kb library never escapes a 0..1000 default). Chimeric
    /// and long-fragment outliers are removed by the percentile trim in
    /// [`Self::fit`] instead, as bwa-mem does.
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
            if !m.is_paired
                || m.mate_is_unmapped
                || m.mate_ref_id != Some(primary.ref_id)
                || primary.is_rev
                || !m.mate_is_rev
                || m.tlen <= 0
            {
                continue;
            }
            let abs_tlen = m.tlen as u32;
            if let Some(cfg) = self.push_sample(abs_tlen) {
                latest = Some(cfg);
            }
        }
        latest
    }

    /// Fit a refined `PairedConfig` from the accumulated samples the way
    /// bwa-mem's `mem_pestat` does: keep the samples inside
    /// `[p25 - 2·IQR, p75 + 2·IQR]`, take their mean and standard deviation,
    /// floor σ, and set the proper-pair window to `mean ± 4σ`.
    fn fit(&self) -> PairedConfig {
        debug_assert!(self.samples.len() >= MIN_SAMPLES);
        let mut sorted: Vec<u32> = self.samples.clone();
        sorted.sort_unstable();
        let p25 = percentile_sorted(&sorted, 0.25) as f64;
        let p75 = percentile_sorted(&sorted, 0.75) as f64;
        let iqr = (p75 - p25).max(0.0);
        let low = (p25 - 2.0 * iqr).max(0.0);
        let high = p75 + 2.0 * iqr;

        let kept: Vec<f64> = sorted
            .iter()
            .map(|&x| x as f64)
            .filter(|&x| x >= low && x <= high)
            .collect();
        let n = kept.len().max(1) as f64;
        let mean = kept.iter().sum::<f64>() / n;
        let var = kept.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
        let sd = var
            .sqrt()
            .max(mean * SD_FLOOR_FRACTION)
            .max(SD_FLOOR_BASES)
            .round();

        let half = sd * WINDOW_SIGMA;
        let min_new = (mean - half).max(0.0).round() as u32;
        let max_new = ((mean + half).round() as u32).max(min_new + 1);

        PairedConfig {
            mode: self.prior.mode,
            insert_min: min_new,
            insert_max: max_new,
            insert_mean: mean.round().max(0.0) as u32,
            insert_sd: sd as u32,
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
