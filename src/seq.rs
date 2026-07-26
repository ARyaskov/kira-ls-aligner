#[inline]
fn comp_base(b: u8) -> u8 {
    match b {
        b'A' | b'a' => b'T',
        b'C' | b'c' => b'G',
        b'G' | b'g' => b'C',
        b'T' | b't' => b'A',
        _ => b'N',
    }
}

/// Reverse-complement DNA sequence into a new owned buffer.
pub fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(seq.len());
    out.extend(seq.iter().rev().map(|&b| comp_base(b)));
    out
}

/// Reverse-complement into a caller-provided scratch buffer.
pub fn reverse_complement_into<'a>(seq: &[u8], out: &'a mut Vec<u8>) -> &'a [u8] {
    out.clear();
    out.reserve(seq.len());
    out.extend(seq.iter().rev().map(|&b| comp_base(b)));
    out.as_slice()
}

/// Length of the longest common prefix of `a` and `b`, eight bytes per step.
#[inline]
pub fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    let n = a.len().min(b.len());
    let mut i = 0usize;
    while i + 8 <= n {
        let x = u64::from_le_bytes(a[i..i + 8].try_into().unwrap())
            ^ u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        if x != 0 {
            // Little-endian: the lowest set bit is in the earliest differing byte.
            return i + (x.trailing_zeros() / 8) as usize;
        }
        i += 8;
    }
    while i < n && a[i] == b[i] {
        i += 1;
    }
    i
}

/// Length of the longest common suffix of `a` and `b`.
#[inline]
pub fn common_suffix_len(a: &[u8], b: &[u8]) -> usize {
    let n = a.len().min(b.len());
    let mut i = 0usize;
    while i + 8 <= n {
        let ao = a.len() - i - 8;
        let bo = b.len() - i - 8;
        let x = u64::from_le_bytes(a[ao..ao + 8].try_into().unwrap())
            ^ u64::from_le_bytes(b[bo..bo + 8].try_into().unwrap());
        if x != 0 {
            // Walking backwards, the *highest* set bit is the latest differing byte.
            return i + (x.leading_zeros() / 8) as usize;
        }
        i += 8;
    }
    while i < n && a[a.len() - i - 1] == b[b.len() - i - 1] {
        i += 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_prefix(a: &[u8], b: &[u8]) -> usize {
        let n = a.len().min(b.len());
        (0..n).take_while(|&i| a[i] == b[i]).count()
    }

    fn naive_suffix(a: &[u8], b: &[u8]) -> usize {
        let n = a.len().min(b.len());
        (0..n)
            .take_while(|&i| a[a.len() - i - 1] == b[b.len() - i - 1])
            .count()
    }

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    #[test]
    fn prefix_and_suffix_match_naive_scans() {
        let mut rng = 0xDEAD_BEEF_CAFE_1234u64;
        let alphabet = b"ACGT";
        for _ in 0..2000 {
            let la = (xorshift(&mut rng) % 40) as usize;
            let lb = (xorshift(&mut rng) % 40) as usize;
            let shared = (xorshift(&mut rng) as usize) % (la.min(lb) + 1);
            let common: Vec<u8> = (0..shared)
                .map(|_| alphabet[xorshift(&mut rng) as usize % 4])
                .collect();
            let mut a = common.clone();
            let mut b = common.clone();
            a.extend((shared..la).map(|_| alphabet[xorshift(&mut rng) as usize % 4]));
            b.extend((shared..lb).map(|_| alphabet[xorshift(&mut rng) as usize % 4]));
            assert_eq!(common_prefix_len(&a, &b), naive_prefix(&a, &b));
            assert_eq!(common_suffix_len(&a, &b), naive_suffix(&a, &b));
        }
    }

    #[test]
    fn handles_empty_and_single_byte_inputs() {
        assert_eq!(common_prefix_len(b"", b"ACGT"), 0);
        assert_eq!(common_suffix_len(b"", b"ACGT"), 0);
        assert_eq!(common_prefix_len(b"A", b"A"), 1);
        assert_eq!(common_suffix_len(b"A", b"A"), 1);
        assert_eq!(common_prefix_len(b"A", b"C"), 0);
        assert_eq!(common_suffix_len(b"TA", b"GA"), 1);
    }

    #[test]
    fn long_identical_runs_are_fully_counted() {
        let a = vec![b'G'; 1000];
        let mut b = vec![b'G'; 1000];
        assert_eq!(common_prefix_len(&a, &b), 1000);
        assert_eq!(common_suffix_len(&a, &b), 1000);
        b[500] = b'T';
        assert_eq!(common_prefix_len(&a, &b), 500);
        assert_eq!(common_suffix_len(&a, &b), 499);
    }
}
