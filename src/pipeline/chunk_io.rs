//! Per-tile alignment temp file format for `--split-prefix`.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Result, Write};
use std::path::Path;

use crate::types::{Alignment, AlignmentKind, CigarKind, CigarOp, MateInfo, Strand};

const MAGIC: &[u8; 8] = b"KIRACHNK";
/// Wire format version.
const VERSION: u32 = 2;

/// Streaming writer for one tile's alignments.
pub struct ChunkWriter {
    inner: BufWriter<File>,
    last_idx: Option<u64>,
}

impl ChunkWriter {
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::create(path)?;
        let mut w = BufWriter::with_capacity(1 << 20, file);
        w.write_all(MAGIC)?;
        w.write_all(&VERSION.to_le_bytes())?;
        Ok(Self {
            inner: w,
            last_idx: None,
        })
    }

    /// Write one record.
    pub fn write_record(&mut self, read_global_idx: u64, alignments: &[Alignment]) -> Result<()> {
        if alignments.is_empty() {
            return Ok(());
        }
        if let Some(prev) = self.last_idx {
            debug_assert!(
                read_global_idx > prev,
                "ChunkWriter: read_global_idx must be strictly increasing (got {} after {})",
                read_global_idx, prev
            );
        }
        self.last_idx = Some(read_global_idx);
        self.inner.write_all(&read_global_idx.to_le_bytes())?;
        self.inner
            .write_all(&(alignments.len() as u32).to_le_bytes())?;
        for aln in alignments {
            write_alignment(&mut self.inner, aln)?;
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        self.inner.flush()
    }
}

/// Streaming reader for one tile's alignments.
pub struct ChunkReader {
    inner: BufReader<File>,
    /// Peeked next record, if any.
    peeked: Option<(u64, Vec<Alignment>)>,
    done: bool,
}

impl ChunkReader {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mut r = BufReader::with_capacity(1 << 20, file);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("bad chunk magic: {magic:?}"),
            ));
        }
        let mut ver_buf = [0u8; 4];
        r.read_exact(&mut ver_buf)?;
        let ver = u32::from_le_bytes(ver_buf);
        if ver != VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unsupported chunk version {ver} (expected {VERSION})"),
            ));
        }
        Ok(Self {
            inner: r,
            peeked: None,
            done: false,
        })
    }

    /// Return the `read_global_idx` of the next record without consuming it, or `None` at EOF.
    pub fn peek_idx(&mut self) -> Result<Option<u64>> {
        self.fill_peeked()?;
        Ok(self.peeked.as_ref().map(|(i, _)| *i))
    }

    /// Consume and return the next record, advancing the cursor.
    pub fn next_record(&mut self) -> Result<Option<(u64, Vec<Alignment>)>> {
        self.fill_peeked()?;
        Ok(self.peeked.take())
    }

    fn fill_peeked(&mut self) -> Result<()> {
        if self.peeked.is_some() || self.done {
            return Ok(());
        }
        let mut idx_buf = [0u8; 8];
        match self.inner.read_exact(&mut idx_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                self.done = true;
                return Ok(());
            }
            Err(e) => return Err(e),
        }
        let idx = u64::from_le_bytes(idx_buf);
        let mut n_buf = [0u8; 4];
        self.inner.read_exact(&mut n_buf)?;
        let n = u32::from_le_bytes(n_buf) as usize;
        let mut alns = Vec::with_capacity(n);
        for _ in 0..n {
            alns.push(read_alignment(&mut self.inner)?);
        }
        self.peeked = Some((idx, alns));
        Ok(())
    }
}

/// Multi-way merge over N tile chunk files.
pub struct MergeIter {
    readers: Vec<ChunkReader>,
}

impl MergeIter {
    pub fn new(readers: Vec<ChunkReader>) -> Self {
        Self { readers }
    }

    pub fn next_merged(&mut self) -> Result<Option<(u64, Vec<Alignment>)>> {
        // Find the minimum next index across all readers.
        let mut min_idx: Option<u64> = None;
        for r in self.readers.iter_mut() {
            if let Some(idx) = r.peek_idx()? {
                min_idx = Some(match min_idx {
                    Some(m) if m <= idx => m,
                    _ => idx,
                });
            }
        }
        let min_idx = match min_idx {
            Some(i) => i,
            None => return Ok(None),
        };
        // Pull from every reader whose peek matches `min_idx`.
        let mut combined: Vec<Alignment> = Vec::new();
        for r in self.readers.iter_mut() {
            if r.peek_idx()? == Some(min_idx) {
                if let Some((_, alns)) = r.next_record()? {
                    combined.extend(alns);
                }
            }
        }
        Ok(Some((min_idx, combined)))
    }
}

fn kind_to_byte(k: AlignmentKind) -> u8 {
    match k {
        AlignmentKind::AcceptedUngapped => 0,
        AlignmentKind::DpAligned => 1,
    }
}

fn byte_to_kind(b: u8) -> Result<AlignmentKind> {
    match b {
        0 => Ok(AlignmentKind::AcceptedUngapped),
        1 => Ok(AlignmentKind::DpAligned),
        x => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("bad AlignmentKind byte {x}"),
        )),
    }
}

fn cigar_op_to_byte(k: CigarKind) -> u8 {
    match k {
        CigarKind::Match => 0,
        CigarKind::Ins => 1,
        CigarKind::Del => 2,
        CigarKind::SoftClip => 3,
        CigarKind::Skipped => 4,
    }
}

fn byte_to_cigar_op(b: u8) -> Result<CigarKind> {
    match b {
        0 => Ok(CigarKind::Match),
        1 => Ok(CigarKind::Ins),
        2 => Ok(CigarKind::Del),
        3 => Ok(CigarKind::SoftClip),
        4 => Ok(CigarKind::Skipped),
        x => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("bad CigarKind byte {x}"),
        )),
    }
}

fn write_alignment<W: Write>(w: &mut W, a: &Alignment) -> Result<()> {
    let mut prim_flags: u8 = 0;
    if a.is_rev {
        prim_flags |= 0x1;
    }
    if a.is_secondary {
        prim_flags |= 0x2;
    }
    if a.is_supplementary {
        prim_flags |= 0x4;
    }
    let mut mate_flags: u8 = 0;
    if a.mate.is_paired {
        mate_flags |= 0x1;
    }
    if a.mate.is_proper_pair {
        mate_flags |= 0x2;
    }
    if a.mate.mate_is_unmapped {
        mate_flags |= 0x4;
    }
    if a.mate.mate_is_rev {
        mate_flags |= 0x8;
    }
    if a.mate.is_first_in_pair {
        mate_flags |= 0x10;
    }
    if a.mate.is_second_in_pair {
        mate_flags |= 0x20;
    }
    if a.mate.mate_ref_id.is_some() {
        mate_flags |= 0x40;
    }
    if a.xs_score.is_some() {
        mate_flags |= 0x80;
    }
    let mut extra_flags: u8 = 0;
    if let Some(s) = a.xs_strand {
        extra_flags |= 0x1;
        if matches!(s, Strand::Reverse) {
            extra_flags |= 0x2;
        }
    }
    w.write_all(&[
        kind_to_byte(a.kind),
        prim_flags,
        mate_flags,
        a.mapq,
        extra_flags,
    ])?;
    w.write_all(&a.ref_id.to_le_bytes())?;
    w.write_all(&a.ref_start.to_le_bytes())?;
    w.write_all(&a.ref_end.to_le_bytes())?;
    w.write_all(&a.read_start.to_le_bytes())?;
    w.write_all(&a.read_end.to_le_bytes())?;
    w.write_all(&a.nm.to_le_bytes())?;
    w.write_all(&a.score.to_le_bytes())?;
    w.write_all(&a.as_score.to_le_bytes())?;
    if let Some(xs) = a.xs_score {
        w.write_all(&xs.to_le_bytes())?;
    }
    if let Some(mr) = a.mate.mate_ref_id {
        w.write_all(&mr.to_le_bytes())?;
    }
    w.write_all(&a.mate.mate_pos.to_le_bytes())?;
    w.write_all(&a.mate.tlen.to_le_bytes())?;

    w.write_all(&(a.cigar.len() as u32).to_le_bytes())?;
    for op in &a.cigar {
        w.write_all(&op.len.to_le_bytes())?;
        w.write_all(&[cigar_op_to_byte(op.op)])?;
    }
    let md_bytes = a.md.as_bytes();
    w.write_all(&(md_bytes.len() as u32).to_le_bytes())?;
    w.write_all(md_bytes)?;
    Ok(())
}

fn read_alignment<R: Read>(r: &mut R) -> Result<Alignment> {
    let mut hdr = [0u8; 5];
    r.read_exact(&mut hdr)?;
    let kind = byte_to_kind(hdr[0])?;
    let prim_flags = hdr[1];
    let mate_flags = hdr[2];
    let mapq = hdr[3];
    let extra_flags = hdr[4];
    let xs_strand = if extra_flags & 0x1 != 0 {
        Some(if extra_flags & 0x2 != 0 {
            Strand::Reverse
        } else {
            Strand::Forward
        })
    } else {
        None
    };

    let ref_id = read_u32(r)?;
    let ref_start = read_u32(r)?;
    let ref_end = read_u32(r)?;
    let read_start = read_u32(r)?;
    let read_end = read_u32(r)?;
    let nm = read_u32(r)?;
    let score = read_i32(r)?;
    let as_score = read_i32(r)?;

    let xs_score = if mate_flags & 0x80 != 0 {
        Some(read_i32(r)?)
    } else {
        None
    };
    let mate_ref_id = if mate_flags & 0x40 != 0 {
        Some(read_u32(r)?)
    } else {
        None
    };
    let mate_pos = read_u32(r)?;
    let tlen = read_i32(r)?;

    let cigar_len = read_u32(r)? as usize;
    let mut cigar = Vec::with_capacity(cigar_len);
    for _ in 0..cigar_len {
        let l = read_u32(r)?;
        let mut ob = [0u8; 1];
        r.read_exact(&mut ob)?;
        cigar.push(CigarOp {
            len: l,
            op: byte_to_cigar_op(ob[0])?,
        });
    }
    let md_len = read_u32(r)? as usize;
    let mut md_buf = vec![0u8; md_len];
    r.read_exact(&mut md_buf)?;
    let md = String::from_utf8(md_buf).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, format!("md utf-8: {e}"))
    })?;

    Ok(Alignment {
        kind,
        ref_id,
        ref_start,
        ref_end,
        read_start,
        read_end,
        cigar,
        score,
        mapq,
        is_rev: prim_flags & 0x1 != 0,
        is_secondary: prim_flags & 0x2 != 0,
        is_supplementary: prim_flags & 0x4 != 0,
        nm,
        md,
        as_score,
        xs_score,
        xs_strand,
        mate: MateInfo {
            is_paired: mate_flags & 0x1 != 0,
            is_proper_pair: mate_flags & 0x2 != 0,
            mate_is_unmapped: mate_flags & 0x4 != 0,
            mate_is_rev: mate_flags & 0x8 != 0,
            is_first_in_pair: mate_flags & 0x10 != 0,
            is_second_in_pair: mate_flags & 0x20 != 0,
            mate_ref_id,
            mate_pos,
            tlen,
        },
    })
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_i32<R: Read>(r: &mut R) -> Result<i32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn sample_aln(ref_id: u32, score: i32, is_rev: bool) -> Alignment {
        Alignment {
            kind: AlignmentKind::DpAligned,
            ref_id,
            ref_start: 100,
            ref_end: 250,
            read_start: 0,
            read_end: 150,
            cigar: vec![CigarOp {
                len: 150,
                op: CigarKind::Match,
            }],
            score,
            mapq: 60,
            is_rev,
            is_secondary: false,
            is_supplementary: false,
            nm: 2,
            md: "75A74".to_string(),
            as_score: score,
            xs_score: Some(score - 10),
            xs_strand: None,
            mate: MateInfo {
                is_paired: true,
                is_proper_pair: true,
                mate_is_unmapped: false,
                mate_is_rev: !is_rev,
                is_first_in_pair: true,
                is_second_in_pair: false,
                mate_ref_id: Some(ref_id),
                mate_pos: 400,
                tlen: 450,
            },
        }
    }

    #[test]
    fn roundtrip_single_record_preserves_all_fields() {
        let f = NamedTempFile::new().unwrap();
        let path = f.path().to_path_buf();
        let aln = sample_aln(7, 142, false);
        {
            let mut w = ChunkWriter::create(&path).unwrap();
            w.write_record(42, std::slice::from_ref(&aln)).unwrap();
            w.finish().unwrap();
        }
        let mut r = ChunkReader::open(&path).unwrap();
        let (idx, recs) = r.next_record().unwrap().expect("one record");
        assert_eq!(idx, 42);
        assert_eq!(recs.len(), 1);
        let a = &recs[0];
        assert_eq!(a.ref_id, aln.ref_id);
        assert_eq!(a.ref_start, aln.ref_start);
        assert_eq!(a.ref_end, aln.ref_end);
        assert_eq!(a.read_start, aln.read_start);
        assert_eq!(a.read_end, aln.read_end);
        assert_eq!(a.nm, aln.nm);
        assert_eq!(a.md, aln.md);
        assert_eq!(a.score, aln.score);
        assert_eq!(a.as_score, aln.as_score);
        assert_eq!(a.xs_score, aln.xs_score);
        assert_eq!(a.is_rev, aln.is_rev);
        assert_eq!(a.cigar.len(), aln.cigar.len());
        assert_eq!(a.cigar[0].len, aln.cigar[0].len);
        assert_eq!(a.cigar[0].op, aln.cigar[0].op);
        assert_eq!(a.mate.is_paired, aln.mate.is_paired);
        assert_eq!(a.mate.is_proper_pair, aln.mate.is_proper_pair);
        assert_eq!(a.mate.mate_ref_id, aln.mate.mate_ref_id);
        assert_eq!(a.mate.mate_pos, aln.mate.mate_pos);
        assert_eq!(a.mate.tlen, aln.mate.tlen);
        // EOF
        assert!(r.next_record().unwrap().is_none());
    }

    #[test]
    fn writer_skips_empty_records() {
        let f = NamedTempFile::new().unwrap();
        let path = f.path().to_path_buf();
        {
            let mut w = ChunkWriter::create(&path).unwrap();
            w.write_record(1, &[]).unwrap();
            w.write_record(2, &[sample_aln(0, 100, false)]).unwrap();
            w.write_record(3, &[]).unwrap();
            w.write_record(4, &[sample_aln(0, 200, true)]).unwrap();
            w.finish().unwrap();
        }
        let mut r = ChunkReader::open(&path).unwrap();
        let r1 = r.next_record().unwrap().unwrap();
        assert_eq!(r1.0, 2);
        let r2 = r.next_record().unwrap().unwrap();
        assert_eq!(r2.0, 4);
        assert!(r.next_record().unwrap().is_none());
    }

    #[test]
    fn merge_iter_unions_records_at_same_idx() {
        let f1 = NamedTempFile::new().unwrap();
        let f2 = NamedTempFile::new().unwrap();
        let p1 = f1.path().to_path_buf();
        let p2 = f2.path().to_path_buf();

        {
            let mut w1 = ChunkWriter::create(&p1).unwrap();
            w1.write_record(1, &[sample_aln(0, 100, false)]).unwrap();
            w1.write_record(5, &[sample_aln(1, 50, true)]).unwrap();
            w1.finish().unwrap();
        }
        {
            let mut w2 = ChunkWriter::create(&p2).unwrap();
            w2.write_record(1, &[sample_aln(3, 120, false)]).unwrap();
            w2.write_record(7, &[sample_aln(4, 90, false)]).unwrap();
            w2.finish().unwrap();
        }

        let mut merge = MergeIter::new(vec![
            ChunkReader::open(&p1).unwrap(),
            ChunkReader::open(&p2).unwrap(),
        ]);
        // idx 1 — both tiles contribute → 2 alignments
        let (i1, a1) = merge.next_merged().unwrap().unwrap();
        assert_eq!(i1, 1);
        assert_eq!(a1.len(), 2);
        // ref_ids should be {0, 3} in some order
        let mut refs: Vec<u32> = a1.iter().map(|a| a.ref_id).collect();
        refs.sort();
        assert_eq!(refs, vec![0, 3]);
        // idx 5 — only tile 1
        let (i2, a2) = merge.next_merged().unwrap().unwrap();
        assert_eq!(i2, 5);
        assert_eq!(a2.len(), 1);
        assert_eq!(a2[0].ref_id, 1);
        // idx 7 — only tile 2
        let (i3, a3) = merge.next_merged().unwrap().unwrap();
        assert_eq!(i3, 7);
        assert_eq!(a3.len(), 1);
        assert_eq!(a3[0].ref_id, 4);
        // EOF
        assert!(merge.next_merged().unwrap().is_none());
    }

    #[test]
    fn bad_magic_errors() {
        let f = NamedTempFile::new().unwrap();
        let path = f.path().to_path_buf();
        std::fs::write(&path, b"NOTKIRACHNK").unwrap();
        // ChunkReader isn't Debug, so we can't use unwrap_err — match instead.
        match ChunkReader::open(&path) {
            Ok(_) => panic!("expected magic error"),
            Err(e) => assert!(
                format!("{e}").contains("bad chunk magic"),
                "wrong error: {e}"
            ),
        }
    }
}
