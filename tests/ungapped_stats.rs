use kira_ls_aligner::alignment::prefilter::identity_x10000;

#[test]
fn identity_x10000_matches_len_mism() {
    assert_eq!(identity_x10000(150, 0), 10000);
    assert_eq!(identity_x10000(150, 1), 9933);
    assert_eq!(identity_x10000(100, 2), 9800);
}
