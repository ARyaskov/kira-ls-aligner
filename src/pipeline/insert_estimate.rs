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
mod tests {
    use super::*;
    use crate::io::IngestMode;

    fn prior() -> PairedConfig {
        PairedConfig {
            mode: IngestMode::TwoFile,
            insert_min: 0,
            insert_max: 1000,
            insert_mean: 200,
            insert_sd: 50,
            estimator_locked: false,
        }
    }

    #[test]
    fn current_returns_prior_until_locked() {
        let est = InsertEstimator::new(prior());
        let cfg = est.current();
        assert_eq!(cfg.insert_mean, 200);
        assert_eq!(cfg.insert_sd, 50);
        assert!(!est.is_locked());
    }

    #[test]
    fn locks_in_after_min_samples() {
        let mut est = InsertEstimator::new(prior());
        for i in 0..MIN_SAMPLES {
            let phase = (i as f64) * 0.1;
            let noise = phase.sin() * 60.0 + (phase * 1.7).cos() * 60.0;
            let tlen = (570.0 + noise).max(50.0) as u32;
            let triggered = est.push_sample(tlen);
            if i + 1 < MIN_SAMPLES {
                assert!(triggered.is_none(), "locked too early at {i}");
            } else {
                assert!(triggered.is_some(), "expected lock at {i}");
            }
        }
        assert!(est.is_locked());
        let refined = est.current();
        assert!(
            (refined.insert_mean as i64 - 570).abs() < 30,
            "mean drifted: {}",
            refined.insert_mean
        );
        // SD should be in the 30..120 range for this distribution.
        assert!(
            (10..=200).contains(&refined.insert_sd),
            "sd implausible: {}",
            refined.insert_sd
        );
    }

    #[test]
    fn push_after_lock_is_noop() {
        let mut est = InsertEstimator::new(prior());
        for _ in 0..MIN_SAMPLES {
            est.push_sample(500);
        }
        assert!(est.is_locked());
        let before = est.current();
        // Push wildly different values; should be ignored.
        for _ in 0..1000 {
            assert!(est.push_sample(10_000).is_none());
        }
        let after = est.current();
        assert_eq!(before.insert_mean, after.insert_mean);
        assert_eq!(before.insert_sd, after.insert_sd);
    }

    #[test]
    fn trimmed_estimate_resists_outliers() {
        let mut est = InsertEstimator::new(prior());
        let outliers = MIN_SAMPLES / 20;
        for _ in 0..(MIN_SAMPLES - outliers) {
            est.push_sample(500);
        }
        for _ in 0..outliers {
            est.push_sample(10_000);
        }
        let refined = est.current();
        assert!(
            (refined.insert_mean as i64 - 500).abs() < 100,
            "outliers leaked into mean: {}",
            refined.insert_mean
        );
    }

    #[test]
    fn median_mad_handles_asymmetric_chimeric_tail() {
        let mut est = InsertEstimator::new(prior());
        let bulk = (MIN_SAMPLES * 85) / 100;
        let tail = MIN_SAMPLES - bulk;
        for i in 0..bulk {
            let phase = (i as f64) * 0.07;
            let noise = phase.sin() * 80.0 + (phase * 1.7).cos() * 60.0;
            let tlen = (570.0 + noise).max(50.0) as u32;
            est.push_sample(tlen);
        }
        for i in 0..tail {
            // Chimeric / long-fragment tail: spread between 1500 and 3000.
            let phase = (i as f64) * 0.13;
            let lr = (phase.sin() * 0.5 + 0.5) * 1500.0; // 0..1500
            est.push_sample(1500u32 + lr as u32);
        }
        let refined = est.current();
        assert!(
            (refined.insert_mean as i64 - 570).abs() < 60,
            "median drifted under asymmetric tail: got {}",
            refined.insert_mean
        );
        assert!(
            (10..=200).contains(&refined.insert_sd),
            "MAD-derived σ implausible: {}",
            refined.insert_sd
        );
        assert!(refined.estimator_locked);
    }

    #[test]
    fn observe_batch_requires_unique_alignment_slot() {
        use crate::types::{Alignment, AlignmentKind, CigarKind, CigarOp, MateInfo};

        fn mk(tlen: i32, mapq: u8, mate_set: bool) -> Alignment {
            Alignment {
                kind: AlignmentKind::DpAligned,
                ref_id: 0,
                ref_start: 0,
                ref_end: 150,
                read_start: 0,
                read_end: 150,
                cigar: vec![CigarOp {
                    len: 150,
                    op: CigarKind::Match,
                }],
                score: 150,
                mapq,
                is_rev: false,
                is_secondary: false,
                is_supplementary: false,
                nm: 0,
                md: "150".to_string(),
                as_score: 150,
                xs_score: None,
                xs_strand: None,
                mate: MateInfo {
                    is_paired: mate_set,
                    is_proper_pair: mate_set,
                    mate_is_unmapped: false,
                    mate_is_rev: true,
                    is_first_in_pair: true,
                    is_second_in_pair: false,
                    mate_ref_id: Some(0),
                    mate_pos: 400,
                    tlen,
                },
            }
        }

        let mut est = InsertEstimator::new(prior());
        let mut batch: Vec<Vec<Alignment>> = Vec::new();
        for i in 0..2048 {
            if i % 2 == 0 {
                batch.push(vec![mk(570, 0, true)]);
            } else {
                batch.push(vec![mk(570, 0, true), mk(570, 0, true)]);
            }
        }
        let _ = est.observe_batch(&batch);
        assert!(est.is_locked(), "should have collected enough unique-slot samples to lock in");
        let refined = est.current();
        assert_eq!(refined.insert_mean, 570);
    }
}
