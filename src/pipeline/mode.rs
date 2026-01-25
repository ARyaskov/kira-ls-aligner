#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReadMode {
    Short,
    Long,
    Hybrid,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ModeFeatures {
    pub read_len_p50: usize,
    pub read_len_p90: usize,
    pub avg_minimizers: f32,
    pub ungapped_len_p95: usize,
    pub ungapped_mism_p95: usize,
    pub ungapped_id_p90: f32,
    pub chains_per_read: f32,
}

pub fn classify(features: ModeFeatures) -> ReadMode {
    let p50 = features.read_len_p50;
    let p90 = features.read_len_p90;
    let err_rate = if features.ungapped_len_p95 > 0 {
        features.ungapped_mism_p95 as f32 / features.ungapped_len_p95 as f32
    } else {
        0.0
    };

    if p50 <= 300 {
        return ReadMode::Short;
    }
    if p90 >= 1000 || p50 >= 1000 {
        return ReadMode::Long;
    }
    if p90 <= 300 && p50 <= 200 && err_rate <= 0.05 {
        return ReadMode::Short;
    }
    if err_rate >= 0.1 || features.avg_minimizers < 5.0 {
        return ReadMode::Long;
    }

    if features.chains_per_read >= 5.0 && p90 <= 500 {
        return ReadMode::Short;
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
