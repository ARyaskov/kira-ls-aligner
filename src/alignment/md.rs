//! The one MD/NM accumulator behind every path that emits an alignment.
//!
//! Five code paths used to build `MD:Z:` by hand (the WFA converter, the
//! CIGAR replay in `compute_nm_md`, the splice aligner's exon and junction
//! loops, and the two ungapped fast paths), and they drifted: the `N`
//! handling was fixed in four places at once. This builder is the single
//! definition of the SAM rules:
//!
//! * a mismatch writes the pending match count and the *reference* base;
//! * a deletion writes the pending count, `^` and the deleted reference
//!   bases, and the next match run starts at zero;
//! * an insertion is invisible in MD (it only counts toward NM);
//! * a reference skip (`N`) is invisible in MD and NM: the match run
//!   continues across the intron (SAM spec §1.7; `samtools calmd`);
//! * the string always ends with a count, `0` when needed, so `MD:Z:0A0`
//!   style output parses everywhere.
//!
//! NM matches only equal A/C/G/T (case-insensitive) or a query `=`. Equal
//! ambiguity codes extend the MD run but count as an edit in NM.

use crate::types::{CigarKind, CigarOp};
use core::fmt::NumBuffer;

#[derive(Clone, Debug, Default)]
pub struct MdBuilder {
    text: String,
    run: u32,
    nm: u32,
}

impl MdBuilder {
    pub fn new() -> Self {
        Self::with_capacity(16)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            text: String::with_capacity(capacity),
            run: 0,
            nm: 0,
        }
    }

    fn flush_run(&mut self) {
        self.text
            .push_str(self.run.format_into(&mut NumBuffer::new()));
        self.run = 0;
    }

    /// Encode the MD reference alphabet. Preserve letters, including IUPAC
    /// ambiguity codes, and represent other input bytes as unknown bases.
    fn reference_letter(base: u8) -> char {
        if base.is_ascii_alphabetic() {
            char::from(base.to_ascii_uppercase())
        } else {
            'N'
        }
    }

    /// One matching column.
    #[inline]
    pub fn matched(&mut self) {
        self.run += 1;
    }

    /// One mismatching column; `ref_base` is what MD records.
    #[inline]
    pub fn mismatch(&mut self, ref_base: u8) {
        self.flush_run();
        self.text.push(Self::reference_letter(ref_base));
        self.nm += 1;
    }

    /// Record SAM tags for one aligned column. The return value is byte
    /// equality for callers accumulating the aligner's substitution score.
    #[inline]
    pub fn column(&mut self, read_base: u8, ref_base: u8) -> bool {
        if read_base == b'=' || read_base.eq_ignore_ascii_case(&ref_base) {
            self.matched();
            self.nm += u32::from(identical_base_is_edit(read_base));
        } else {
            self.mismatch(ref_base);
        }
        read_base == ref_base
    }

    /// `read.len()` aligned columns (the ungapped case). Returns the number
    /// of byte mismatches for substitution scoring; `nm()` gives SAM edit distance.
    pub fn columns(&mut self, read: &[u8], reference: &[u8]) -> u32 {
        debug_assert_eq!(read.len(), reference.len());
        let mut hamming = 0;
        for (&qb, &rb) in read.iter().zip(reference) {
            hamming += u32::from(!self.column(qb, rb));
        }
        hamming
    }

    /// `len` inserted read bases: NM only.
    #[inline]
    pub fn insertion(&mut self, len: u32) {
        self.nm += len;
    }

    /// The deleted reference bases.
    pub fn deletion(&mut self, ref_bases: &[u8]) {
        if ref_bases.is_empty() {
            return;
        }
        self.flush_run();
        self.text.push('^');
        self.text
            .extend(ref_bases.iter().copied().map(Self::reference_letter));
        self.nm += ref_bases.len() as u32;
    }

    /// A reference skip (`N` op). Nothing to record: the match run continues
    /// across the intron. Exists so call sites document the case.
    #[inline]
    pub fn skip(&mut self) {}

    /// Edit distance so far.
    #[inline]
    pub fn nm(&self) -> u32 {
        self.nm
    }

    /// Close the trailing run and return `(NM, MD)`.
    pub fn finish(mut self) -> (u32, String) {
        self.flush_run();
        (self.nm, self.text)
    }
}

#[inline]
fn identical_base_is_edit(base: u8) -> bool {
    !matches!(base.to_ascii_uppercase(), b'A' | b'C' | b'G' | b'T' | b'=')
}

/// An exact byte match needs only the number of ambiguity codes for its NM.
pub(crate) fn identical_nm(bases: &[u8]) -> u32 {
    bases
        .iter()
        .map(|&base| u32::from(identical_base_is_edit(base)))
        .sum()
}

pub(crate) fn ungapped_tags(read: &[u8], reference: &[u8]) -> (u32, String) {
    let mut md = MdBuilder::new();
    md.columns(read, reference);
    md.finish()
}

/// Append the decimal digits of `v` without allocating.
#[inline]
pub fn push_u32_decimal(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(v.format_into(&mut NumBuffer::new()).as_bytes());
}

/// Replay `cigar` over `read[read_start..]` and `reference[ref_start..]` and
/// return `(NM, MD)`. Positions past either end read as `N` so a malformed
/// CIGAR cannot panic here (it is caught by the emitter's consumption check).
pub fn nm_md_from_cigar(
    read: &[u8],
    reference: &[u8],
    read_start: usize,
    ref_start: usize,
    cigar: &[CigarOp],
) -> (u32, String) {
    let mut md = MdBuilder::new();
    let mut qpos = read_start;
    let mut rpos = ref_start;
    for op in cigar {
        let len = op.len as usize;
        match op.op {
            CigarKind::Match => {
                for _ in 0..len {
                    let qb = read.get(qpos).copied().unwrap_or(b'N');
                    let rb = reference.get(rpos).copied().unwrap_or(b'N');
                    md.column(qb, rb);
                    qpos += 1;
                    rpos += 1;
                }
            }
            CigarKind::Ins => {
                md.insertion(op.len);
                qpos += len;
            }
            CigarKind::Del => {
                let end = (rpos + len).min(reference.len());
                let start = rpos.min(end);
                md.deletion(&reference[start..end]);
                // Bases past the reference end still count toward NM.
                md.nm += (len - (end - start)) as u32;
                rpos += len;
            }
            CigarKind::SoftClip => qpos += len,
            CigarKind::Skipped => {
                md.skip();
                rpos += len;
            }
        }
    }
    md.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ops(spec: &[(u32, CigarKind)]) -> Vec<CigarOp> {
        spec.iter().map(|&(len, op)| CigarOp { len, op }).collect()
    }

    #[test]
    fn perfect_match_is_one_count() {
        let (nm, md) = nm_md_from_cigar(b"ACGT", b"ACGT", 0, 0, &ops(&[(4, CigarKind::Match)]));
        assert_eq!((nm, md.as_str()), (0, "4"));
    }

    #[test]
    fn mismatch_records_the_reference_base() {
        let (nm, md) = nm_md_from_cigar(b"ACGT", b"AGGT", 0, 0, &ops(&[(4, CigarKind::Match)]));
        assert_eq!((nm, md.as_str()), (1, "1G2"));
    }

    #[test]
    fn adjacent_mismatches_and_edges_keep_zero_counts() {
        let (nm, md) = nm_md_from_cigar(b"TTGA", b"ACGT", 0, 0, &ops(&[(4, CigarKind::Match)]));
        assert_eq!((nm, md.as_str()), (3, "0A0C1T0"));
    }

    #[test]
    fn deletion_and_insertion() {
        // read ACGTACGT vs ref ACG[TT]TACGT: 3M2D5M
        let cigar = ops(&[
            (3, CigarKind::Match),
            (2, CigarKind::Del),
            (5, CigarKind::Match),
        ]);
        let (nm, md) = nm_md_from_cigar(b"ACGTACGT", b"ACGTTTACGT", 0, 0, &cigar);
        assert_eq!((nm, md.as_str()), (2, "3^TT5"));
        let cigar = ops(&[
            (3, CigarKind::Match),
            (2, CigarKind::Ins),
            (3, CigarKind::Match),
        ]);
        let (nm, md) = nm_md_from_cigar(b"ACGAAGTA", b"ACGGTA", 0, 0, &cigar);
        assert_eq!((nm, md.as_str()), (2, "6"));
    }

    #[test]
    fn soft_clips_and_offsets_are_skipped() {
        let cigar = ops(&[
            (2, CigarKind::SoftClip),
            (3, CigarKind::Match),
            (1, CigarKind::SoftClip),
        ]);
        let (nm, md) = nm_md_from_cigar(b"NNACGN", b"XXXXACG", 0, 4, &cigar);
        assert_eq!((nm, md.as_str()), (0, "3"));
    }

    #[test]
    fn reference_skip_does_not_split_the_run() {
        let cigar = ops(&[
            (2, CigarKind::Match),
            (100, CigarKind::Skipped),
            (2, CigarKind::Match),
        ]);
        let mut reference = vec![b'T'; 104];
        reference[0] = b'A';
        reference[1] = b'C';
        reference[102] = b'G';
        reference[103] = b'A';
        let (nm, md) = nm_md_from_cigar(b"ACGA", &reference, 0, 0, &cigar);
        assert_eq!((nm, md.as_str()), (0, "4"));
    }

    #[test]
    fn builder_columns_reports_mismatches() {
        let mut md = MdBuilder::new();
        assert_eq!(md.columns(b"ACGT", b"ACCT"), 1);
        md.insertion(2);
        assert_eq!(md.nm(), 3);
        assert_eq!(md.finish(), (3, "2C1".to_string()));
    }

    #[test]
    fn decimal_push_matches_std() {
        for v in [0u32, 1, 9, 10, 99, 100, 4_294_967_295] {
            let mut out = Vec::new();
            push_u32_decimal(&mut out, v);
            assert_eq!(String::from_utf8(out).unwrap(), v.to_string());
        }
    }
}
