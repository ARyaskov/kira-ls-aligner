use kira_ls_aligner::pipeline::mode::{ModeFeatures, ReadMode, classify};

#[test]
fn mode_short_for_illumina_like() {
    let f = ModeFeatures {
        read_len_p50: 150,
        read_len_p90: 150,
        avg_minimizers: 30.0,
        ungapped_len_p95: 150,
        ungapped_mism_p95: 2,
        ungapped_id_p90: 99.0,
        chains_per_read: 2.0,
    };
    assert_eq!(classify(f), ReadMode::Short);
}

#[test]
fn mode_long_for_ont_like() {
    let f = ModeFeatures {
        read_len_p50: 5000,
        read_len_p90: 8000,
        avg_minimizers: 200.0,
        ungapped_len_p95: 5000,
        ungapped_mism_p95: 800,
        ungapped_id_p90: 85.0,
        chains_per_read: 1.5,
    };
    assert_eq!(classify(f), ReadMode::Long);
}

#[test]
fn mode_hybrid_for_mixed_short_and_long_batch() {
    let f = ModeFeatures {
        read_len_p50: 150,
        read_len_p90: 5000,
        avg_minimizers: 50.0,
        ..ModeFeatures::default()
    };
    assert_eq!(classify(f), ReadMode::Hybrid);
}
