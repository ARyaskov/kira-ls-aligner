use kira_kv_engine::{Index, IndexBuilder};

const ASSIGNMENT_BATCH: usize = 4096;

pub(super) fn build_mph(hashes: &[u64]) -> (Index, Vec<u32>) {
    let key_bytes: Vec<[u8; 8]> = hashes.iter().map(|hash| hash.to_le_bytes()).collect();
    let index = IndexBuilder::new()
        .with_parallel_build(true)
        .build_index_ref(&key_bytes)
        .expect("PtrHash25 build");
    drop(key_bytes);
    let assigned = assign_slots(&index, hashes);
    (index, assigned)
}

fn assign_slots(index: &Index, hashes: &[u64]) -> Vec<u32> {
    let slots = index.slot_capacity();
    assert_eq!(slots, hashes.len(), "new minimizer MPH must be minimal");
    let mut assigned = Vec::with_capacity(hashes.len());
    let mut seen = vec![0u64; slots.div_ceil(64)];
    let mut canonical = [0u64; ASSIGNMENT_BATCH];
    let mut results = [None; ASSIGNMENT_BATCH];
    for chunk in hashes.chunks(ASSIGNMENT_BATCH) {
        index.lookup_batch_u64_simd_into(chunk, &mut canonical, &mut results);
        for result in &results[..chunk.len()] {
            let slot = result.expect("PtrHash25 hash present in build set");
            assert!(slot < slots, "MPH slot exceeds minimizer bucket count");
            let mask = 1u64 << (slot % 64);
            let word = &mut seen[slot / 64];
            assert_eq!(*word & mask, 0, "two minimizer hashes share an MPH slot");
            *word |= mask;
            assigned.push(u32::try_from(slot).expect("MPH slot exceeds u32"));
        }
    }
    assigned
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(mut key: u64) -> u64 {
        key = (key ^ (key >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        key = (key ^ (key >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        key ^ (key >> 31)
    }

    #[test]
    fn assigned_slots_preserve_known_keys_and_form_a_bijection_at_batch_tails() {
        for count in [0, 1, 15, 16, 63, 64, 65, 4095, 4096, 4097, 8193] {
            let hashes: Vec<_> = (0..count as u64).map(hash).collect();
            let (index, assigned) = build_mph(&hashes);
            assert_eq!(index.len(), count);
            assert_eq!(assigned.len(), count);
            let mut seen = vec![false; count];
            for (&key, &slot) in hashes.iter().zip(&assigned) {
                let slot = slot as usize;
                assert!(slot < count);
                assert!(!seen[slot]);
                seen[slot] = true;
                assert_eq!(index.lookup_u64(key).unwrap(), slot);
            }
            assert!(seen.into_iter().all(|present| present));
        }
    }

    #[test]
    #[should_panic(expected = "PtrHash25 build")]
    fn duplicate_minimizer_hashes_are_rejected() {
        let _ = build_mph(&[17, 17]);
    }
}
