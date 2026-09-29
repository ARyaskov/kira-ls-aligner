use kira_ls_aligner::pipeline::mode::{ModeFeatures, ReadMode, classify};

fn features(p50: usize, p90: usize) -> ModeFeatures {
    ModeFeatures {
        read_len_p50: p50,
        read_len_p90: p90,
        n_reads: 1000,
    }
}

#[test]
fn mode_short_for_illumina_like() {
    assert_eq!(classify(features(150, 150)), ReadMode::Short);
    assert_eq!(classify(features(250, 300)), ReadMode::Short);
}

#[test]
fn mode_long_for_ont_like() {
    assert_eq!(classify(features(5000, 8000)), ReadMode::Long);
    assert_eq!(classify(features(800, 2500)), ReadMode::Long);
}

#[test]
fn mode_hybrid_for_mixed_short_and_long_batch() {
    assert_eq!(classify(features(150, 5000)), ReadMode::Hybrid);
    // Neither clearly short nor clearly long.
    assert_eq!(classify(features(500, 800)), ReadMode::Hybrid);
}

#[test]
fn features_come_from_length_percentiles() {
    let mut lengths: Vec<usize> = (1..=100).collect();
    let f = ModeFeatures::from_read_lengths(&mut lengths);
    assert_eq!((f.read_len_p50, f.read_len_p90, f.n_reads), (50, 90, 100));
    let f = ModeFeatures::from_read_lengths(&mut Vec::new());
    assert_eq!(f.n_reads, 0);
}
