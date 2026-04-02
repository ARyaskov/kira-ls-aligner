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
