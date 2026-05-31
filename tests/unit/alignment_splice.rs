use super::*;

#[test]
fn canonical_gt_ag_is_forward() {
    assert_eq!(detect_splice_strand(*b"GT", *b"AG"), Some(Strand::Forward));
}

#[test]
fn near_canonical_gc_ag_is_forward() {
    assert_eq!(detect_splice_strand(*b"GC", *b"AG"), Some(Strand::Forward));
}

#[test]
fn u12_at_ac_is_forward() {
    assert_eq!(detect_splice_strand(*b"AT", *b"AC"), Some(Strand::Forward));
}

#[test]
fn reverse_complement_signals_are_reverse() {
    assert_eq!(detect_splice_strand(*b"CT", *b"AC"), Some(Strand::Reverse));
    assert_eq!(detect_splice_strand(*b"GT", *b"AT"), Some(Strand::Reverse));
}

#[test]
fn non_canonical_returns_none() {
    assert_eq!(detect_splice_strand(*b"AA", *b"TT"), None);
    assert_eq!(detect_splice_strand(*b"GG", *b"CC"), None);
    assert_eq!(detect_splice_strand(*b"NN", *b"NN"), None);
}

#[test]
fn lowercase_input_is_normalised() {
    assert_eq!(detect_splice_strand(*b"gt", *b"ag"), Some(Strand::Forward));
}
