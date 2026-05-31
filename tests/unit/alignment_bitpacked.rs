use super::*;

#[test]
fn shift_aligns_one_nucleotide() {
    let p = PackedDna::pack(b"ACGT");
    let s = shift_bits_right(&p.bits, 2);
    assert_eq!(s[0], 0x39);
}
