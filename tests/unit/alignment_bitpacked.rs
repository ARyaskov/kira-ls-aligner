use super::*;

#[test]
fn shift_aligns_one_nucleotide() {
    let p = PackedDna::pack(b"ACGT");
    let s = shift_bits_right(&p.bits, 2);
    assert_eq!(s[0], 0x39);
}

#[test]
fn ambiguous_bases_invalidate_exact_certificate() {
    assert!(!PackedDna::pack(b"ACNT").valid);
    assert!(!PackedDna::pack(b"acgu").valid);
}

#[test]
fn scan_returns_global_best_not_first_acceptable_shift() {
    let read = PackedDna::pack(b"ACGT");
    let reference = PackedDna::pack(b"TTTTACGT");
    let shifted = reference.pre_shifted_window();
    let hit = scan(&read, &shifted, reference.len, 4).expect("hit");
    assert_eq!(hit.shift, 4);
    assert_eq!(hit.mismatches, 0);
}

#[test]
fn scan_reports_tied_second_best() {
    let read = PackedDna::pack(b"ACGT");
    let reference = PackedDna::pack(b"ACGTACGT");
    let shifted = reference.pre_shifted_window();
    let (best, second) = scan_best_with_second(&read, &shifted, reference.len).expect("hit");
    assert_eq!(best.mismatches, 0);
    assert_eq!(second, Some(0));
}
