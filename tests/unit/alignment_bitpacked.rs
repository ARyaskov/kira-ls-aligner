use super::*;

#[test]
fn shift_aligns_one_nucleotide() {
    let p = PackedDna::pack(b"ACGT");
    let mut s = Vec::new();
    shift_bits_right_into(&p.bits, 2, &mut s);
    assert_eq!(s[0], 0x39);
}

#[test]
fn pre_shift_into_matches_owned_pre_shifted_window() {
    let p = PackedDna::pack(b"ACGTTGCAAGGCTTAC");
    let owned = p.pre_shifted_window();
    let mut reused = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    // Pre-dirty the buffers: `pre_shift_into` must fully overwrite them, since
    // callers reuse one scratch across candidates of differing lengths.
    for (i, buf) in reused.iter_mut().enumerate() {
        buf.resize(64 + i, 0xAA);
    }
    pre_shift_into(&p.bits, &mut reused);
    for phase in 0..4 {
        assert_eq!(owned[phase], reused[phase], "phase {phase}");
    }
}

#[test]
fn pack_into_matches_owned_pack_and_reports_validity() {
    let mut bits = vec![0xFFu8; 99];
    let valid = pack_into(b"ACGTACGTA", &mut bits);
    let owned = PackedDna::pack(b"ACGTACGTA");
    assert!(valid);
    assert_eq!(bits, owned.bits);
    assert!(!pack_into(b"ACNT", &mut bits));
}

#[test]
fn scan_raw_matches_packed_dna_entry_point() {
    let read = b"ACGTTGCAAGGCTTACGGAT";
    let reference = b"TTACGTTGCAAGGCTTACGGATCC";
    let rp = PackedDna::pack(read);
    let tp = PackedDna::pack(reference);
    let shifted = tp.pre_shifted_window();
    let via_struct = scan_best_with_second(&rp, &shifted, reference.len());
    let via_raw = scan_best_with_second_raw(
        &rp.bits,
        rp.len,
        rp.valid,
        &shifted,
        reference.len(),
    );
    assert_eq!(
        via_struct.map(|(h, s)| (h.shift, h.mismatches, s)),
        via_raw.map(|(h, s)| (h.shift, h.mismatches, s)),
    );
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
