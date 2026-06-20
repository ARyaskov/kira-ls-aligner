use super::*;

#[test]
fn auto_pairs_two_fastqs_without_explicit_flag() {
    let (mode, auto) = resolve_paired_mode(false, false, 2).unwrap();
    assert_eq!(mode, IngestMode::TwoFile);
    assert!(auto, "auto-detect notice must be flagged for caller");
}

#[test]
fn one_or_many_fastqs_stay_unpaired_without_explicit_flag() {
    let (mode, auto) = resolve_paired_mode(false, false, 1).unwrap();
    assert_eq!(mode, IngestMode::Unpaired);
    assert!(!auto);
    let (mode, auto) = resolve_paired_mode(false, false, 3).unwrap();
    assert_eq!(mode, IngestMode::Unpaired);
    assert!(!auto);
}

#[test]
fn explicit_paired_does_not_trigger_auto_notice() {
    let (mode, auto) = resolve_paired_mode(true, false, 2).unwrap();
    assert_eq!(mode, IngestMode::TwoFile);
    assert!(!auto, "explicit --paired must not raise the notice");
}

#[test]
fn interleaved_takes_one_file() {
    let (mode, _) = resolve_paired_mode(false, true, 1).unwrap();
    assert_eq!(mode, IngestMode::Interleaved);
    assert!(resolve_paired_mode(false, true, 2).is_err());
}

#[test]
fn paired_with_wrong_count_errors() {
    assert!(resolve_paired_mode(true, false, 1).is_err());
    assert!(resolve_paired_mode(true, false, 3).is_err());
}

#[test]
fn full_output_defaults_to_accuracy_path() {
    assert!(!resolve_accept_enable(None, false));
}

#[test]
fn fast_output_defaults_to_ungapped_accept() {
    assert!(resolve_accept_enable(None, true));
}

#[test]
fn explicit_accept_setting_overrides_output_mode() {
    assert!(resolve_accept_enable(Some(true), false));
    assert!(!resolve_accept_enable(Some(false), true));
}
