use std::fmt;

/// DNA strand orientation for a hit or alignment.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Strand {
    Forward,
    Reverse,
}

/// Reference sequence storage.
#[derive(Clone, Debug)]
pub enum RefBases {
    Owned(Vec<u8>),
    Mmap { offset: usize, len: usize },
}

/// A reference contig sequence.
#[derive(Clone, Debug)]
pub struct RefSeq {
    pub name: String,
    pub bases: RefBases,
}

impl RefSeq {
    pub fn bases<'a>(&'a self, mmap: Option<&'a [u8]>) -> &'a [u8] {
        match &self.bases {
            RefBases::Owned(v) => v.as_slice(),
            RefBases::Mmap { offset, len } => {
                let data = mmap.expect("mmap required for RefBases::Mmap");
                &data[*offset..*offset + *len]
            }
        }
    }

    pub fn len(&self, _mmap: Option<&[u8]>) -> usize {
        match &self.bases {
            RefBases::Owned(v) => v.len(),
            RefBases::Mmap { len, .. } => *len,
        }
    }
}

/// Reference collection.
#[derive(Clone, Debug, Default)]
pub struct Reference {
    pub sequences: Vec<RefSeq>,
}

/// A read with sequence and optional quality scores.
#[derive(Clone, Debug)]
pub struct ReadRecord {
    pub id: String,
    pub seq: Vec<u8>,
    pub qual: Option<Vec<u8>>,
}

/// A minimizer sketch entry.
#[derive(Clone, Copy, Debug)]
pub struct Minimizer {
    pub hash: u64,
    pub pos: u32,
    pub strand: Strand,
}

/// A seed hit between read and reference.
#[derive(Clone, Copy, Debug)]
pub struct SeedHit {
    pub hash: u64,
    pub read_pos: u32,
    pub ref_id: u32,
    pub ref_pos: u32,
    pub strand: Strand,
}

/// An extended exact-match anchor (MEM-like).
#[derive(Clone, Debug)]
pub struct Anchor {
    pub read_start: u32,
    pub read_end: u32,
    pub ref_id: u32,
    pub ref_start: u32,
    pub ref_end: u32,
    pub strand: Strand,
    pub score: i32,
}

/// A chain of anchors forming a candidate alignment.
#[derive(Clone, Debug)]
pub struct Chain {
    pub anchors: Vec<Anchor>,
    pub score: i32,
    pub ref_id: u32,
    pub read_start: u32,
    pub read_end: u32,
    pub ref_start: u32,
    pub ref_end: u32,
    pub strand: Strand,
}

/// A compact CIGAR operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CigarOp {
    pub len: u32,
    pub op: CigarKind,
}

/// CIGAR operation kinds (subset).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CigarKind {
    Match,
    Ins,
    Del,
    SoftClip,
}

impl fmt::Display for CigarOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let op_char = match self.op {
            CigarKind::Match => 'M',
            CigarKind::Ins => 'I',
            CigarKind::Del => 'D',
            CigarKind::SoftClip => 'S',
        };
        write!(f, "{}{}", self.len, op_char)
    }
}

/// How an alignment was produced.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AlignmentKind {
    AcceptedUngapped,
    DpAligned,
}
/// Alignment result for a single read against one reference region.
#[derive(Clone, Debug)]
pub struct Alignment {
    pub kind: AlignmentKind,
    pub ref_id: u32,
    pub ref_start: u32,
    pub ref_end: u32,
    pub read_start: u32,
    pub read_end: u32,
    pub cigar: Vec<CigarOp>,
    pub score: i32,
    pub mapq: u8,
    pub is_rev: bool,
    pub is_secondary: bool,
    pub is_supplementary: bool,
    pub nm: u32,
    pub md: String,
    pub as_score: i32,
    pub xs_score: Option<i32>,
}
