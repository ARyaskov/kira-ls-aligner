//! Auto read-mode selection (`-x auto`).
//!
//! The mode is decided once, on the first batch, from the read-length
//! distribution only: that is the one signal available before any seeding
//! or alignment has run, and it separates the three tunings cleanly
//! (Illumina-length reads, long reads, and a mixed library). The `hybrid`
//! profile is the fallback for anything in between.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReadMode {
    Short,
    Long,
    Hybrid,
}

/// Read-length summary of the batch the decision is made on.
#[derive(Clone, Copy, Debug, Default)]
pub struct ModeFeatures {
    pub read_len_p50: usize,
    pub read_len_p90: usize,
    /// Reads the percentiles were computed over.
    pub n_reads: usize,
}

impl ModeFeatures {
    /// Length percentiles over `lengths` (any order).
    pub fn from_read_lengths(lengths: &mut [usize]) -> Self {
        if lengths.is_empty() {
            return Self::default();
        }
        lengths.sort_unstable();
        let percentile = |pct: usize| lengths[((lengths.len() - 1) * pct) / 100];
        Self {
            read_len_p50: percentile(50),
            read_len_p90: percentile(90),
            n_reads: lengths.len(),
        }
    }
}

/// Short when the bulk of the reads are Illumina-length, long when the
/// median or the 90th percentile is kilobase-scale, hybrid for a mixed
/// library (short median, long tail) and for everything in between.
pub fn classify(features: ModeFeatures) -> ReadMode {
    let p50 = features.read_len_p50;
    let p90 = features.read_len_p90;
    if p50 <= 300 && p90 <= 400 {
        return ReadMode::Short;
    }
    if p50 <= 400 && p90 >= 1000 {
        return ReadMode::Hybrid;
    }
    if p50 >= 1000 || p90 >= 2000 {
        return ReadMode::Long;
    }
    ReadMode::Hybrid
}

use crate::pipeline::PipelineConfig;

#[derive(Clone, Debug)]
pub struct ReadModeProfiles {
    pub short: PipelineConfig,
    pub long: PipelineConfig,
    pub hybrid: PipelineConfig,
    pub decided: Option<ReadMode>,
}

impl ReadModeProfiles {
    pub fn select(&self, mode: ReadMode) -> PipelineConfig {
        match mode {
            ReadMode::Short => self.short,
            ReadMode::Long => self.long,
            ReadMode::Hybrid => self.hybrid,
        }
    }
}
