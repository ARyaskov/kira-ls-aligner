use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use memmap2::Mmap;
use needletail::parse_fastx_reader;

use crate::types::{Alignment, CigarKind, CigarOp, ReadRecord, RefBases, RefSeq, Reference};

/// Load reference sequences from FASTA using mmap.
pub fn read_reference<P: AsRef<Path>>(path: P) -> Result<Reference> {
    let mut reader = open_fastx_reader(path)?.reader;
    let mut sequences = Vec::new();
    while let Some(record) = reader.next() {
        let record = record.context("read reference record")?;
        let name = String::from_utf8_lossy(record.id()).to_string();
        let mut seq = record.seq().to_vec();
        normalize_bases(&mut seq);
        sequences.push(RefSeq {
            name,
            bases: RefBases::Owned(seq),
        });
    }
    Ok(Reference { sequences })
}

/// Stream reads in batches of approximate total bases using mmap.
pub struct ReadStream {
    readers: Vec<Box<dyn needletail::FastxReader>>,
    current: usize,
    batch_bases: usize,
    progress: ReadProgress,
}

impl ReadStream {
    pub fn new<P: AsRef<Path>>(path: P, batch_bases: usize) -> Result<Self> {
        Self::new_multi(&[path.as_ref().to_path_buf()], batch_bases)
    }

    pub fn new_multi(paths: &[std::path::PathBuf], batch_bases: usize) -> Result<Self> {
        if paths.is_empty() {
            return Err(anyhow::anyhow!("no reads input provided"));
        }
        let mut readers = Vec::with_capacity(paths.len());
        let mut total = 0u64;
        let counter = Arc::new(AtomicU64::new(0));
        for path in paths {
            let fastx = open_fastx_reader_with_counter(path, counter.clone())?;
            total = total.saturating_add(fastx.progress.total_bytes);
            readers.push(fastx.reader);
        }
        Ok(Self {
            readers,
            current: 0,
            batch_bases,
            progress: ReadProgress {
                total_bytes: total,
                read_bytes: counter,
            },
        })
    }

    pub fn next_batch(&mut self) -> Result<Option<Vec<ReadRecord>>> {
        let mut reads = Vec::new();
        let mut bases = 0usize;
        while self.current < self.readers.len() {
            if let Some(record) = self.readers[self.current].next() {
                let record = record.context("read record")?;
                let id = String::from_utf8_lossy(record.id()).to_string();
                let mut seq = record.seq().to_vec();
                normalize_bases(&mut seq);
                let qual = record.qual().map(|q| q.to_vec());
                bases += seq.len();
                reads.push(ReadRecord { id, seq, qual });
                if bases >= self.batch_bases && !reads.is_empty() {
                    break;
                }
            } else {
                self.current += 1;
            }
        }
        if reads.is_empty() {
            Ok(None)
        } else {
            Ok(Some(reads))
        }
    }

    pub fn total_bytes(&self) -> u64 {
        self.progress.total_bytes
    }

    pub fn bytes_read(&self) -> u64 {
        self.progress.bytes_read()
    }
}

struct FastxReaderWithProgress {
    reader: Box<dyn needletail::FastxReader>,
    progress: ReadProgress,
}

struct ReadProgress {
    total_bytes: u64,
    read_bytes: Arc<AtomicU64>,
}

impl ReadProgress {
    fn bytes_read(&self) -> u64 {
        self.read_bytes.load(Ordering::Relaxed)
    }
}

fn open_fastx_reader_with_counter<P: AsRef<Path>>(
    path: P,
    counter: Arc<AtomicU64>,
) -> Result<FastxReaderWithProgress> {
    let file = File::open(path.as_ref()).context("open FASTA/FASTQ")?;
    let total = file.metadata().context("stat FASTA/FASTQ")?.len();
    let mmap = unsafe { Mmap::map(&file).context("mmap FASTA/FASTQ")? };
    let reader = CountingReader::new(io::Cursor::new(mmap), counter.clone());
    let fastx_reader = parse_fastx_reader(reader).context("parse FASTA/FASTQ")?;
    Ok(FastxReaderWithProgress {
        reader: fastx_reader,
        progress: ReadProgress {
            total_bytes: total,
            read_bytes: counter,
        },
    })
}

fn open_fastx_reader<P: AsRef<Path>>(path: P) -> Result<FastxReaderWithProgress> {
    let counter = Arc::new(AtomicU64::new(0));
    open_fastx_reader_with_counter(path, counter)
}

fn normalize_bases(seq: &mut [u8]) {
    for b in seq.iter_mut() {
        *b = match *b {
            b'a' | b'A' => b'A',
            b'c' | b'C' => b'C',
            b'g' | b'G' => b'G',
            b't' | b'T' => b'T',
            _ => b'N',
        };
    }
}

struct CountingReader<R> {
    inner: R,
    counter: Arc<AtomicU64>,
}

impl<R> CountingReader<R> {
    fn new(inner: R, counter: Arc<AtomicU64>) -> Self {
        Self { inner, counter }
    }
}

impl<R: Read> Read for CountingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        if n > 0 {
            self.counter.fetch_add(n as u64, Ordering::Relaxed);
        }
        Ok(n)
    }
}

/// Output tag configuration.
#[derive(Clone, Copy, Debug)]
pub struct OutputConfig {
    pub write_nm: bool,
    pub write_md: bool,
    pub write_as: bool,
    pub write_xs: bool,
    pub write_rg: bool,
    pub write_xa: bool,
    pub write_sa: bool,
}

impl OutputConfig {
    pub fn full() -> Self {
        Self {
            write_nm: true,
            write_md: true,
            write_as: true,
            write_xs: true,
            write_rg: true,
            write_xa: true,
            write_sa: true,
        }
    }

    pub fn fast() -> Self {
        Self {
            write_nm: true,
            write_md: false,
            write_as: true,
            write_xs: false,
            write_rg: true,
            write_xa: false,
            write_sa: false,
        }
    }
}

/// SAM writer with minimal header support.
pub struct SamWriter {
    writer: BufWriter<Box<dyn Write + Send>>,
    reference: Reference,
}

impl SamWriter {
    pub fn new<P: AsRef<Path>>(path: Option<P>, reference: Reference) -> Result<Self> {
        let writer: Box<dyn Write + Send> = match path {
            Some(p) => Box::new(File::create(p).context("create SAM output")?),
            None => Box::new(io::stdout()),
        };
        Ok(Self {
            writer: BufWriter::with_capacity(1 << 20, writer),
            reference,
        })
    }

    pub fn write_header(&mut self) -> Result<()> {
        self.write_header_with_rg(None)
    }

    pub fn write_header_with_rg(&mut self, read_group: Option<&str>) -> Result<()> {
        writeln!(self.writer, "@HD\tVN:1.6\tSO:unsorted")?;
        for seq in &self.reference.sequences {
            writeln!(self.writer, "@SQ\tSN:{}\tLN:{}", seq.name, seq.len(None))?;
        }
        if let Some(rg) = read_group {
            if rg.starts_with("@RG") {
                writeln!(self.writer, "{}", rg)?;
            } else {
                writeln!(self.writer, "@RG\t{}", rg)?;
            }
        }
        Ok(())
    }

    pub fn write_batch(&mut self, buf: &[u8]) -> Result<()> {
        self.writer.write_all(buf).context("write SAM batch")?;
        Ok(())
    }

    pub fn append_unmapped(&self, buf: &mut Vec<u8>, read: &ReadRecord) {
        buf.extend_from_slice(read.id.as_bytes());
        buf.push(b'\t');
        push_u32(buf, 4);
        buf.extend_from_slice(b"\t*\t0\t0\t*\t*\t0\t0\t");
        buf.extend_from_slice(&read.seq);
        buf.push(b'\t');
        match read.qual.as_ref() {
            Some(q) => buf.extend_from_slice(q),
            None => buf.push(b'*'),
        }
        buf.push(b'\n');
    }

    pub fn append_alignment(
        &self,
        buf: &mut Vec<u8>,
        read: &ReadRecord,
        aln: &Alignment,
        read_group: Option<&str>,
        extra_tags: Option<&[u8]>,
        cfg: OutputConfig,
    ) {
        let rname = &self.reference.sequences[aln.ref_id as usize].name;
        let pos = aln.ref_start + 1; // SAM is 1-based
        let flag = sam_flag(aln);

        buf.extend_from_slice(read.id.as_bytes());
        buf.push(b'\t');
        push_u32(buf, flag as u32);
        buf.push(b'\t');
        buf.extend_from_slice(rname.as_bytes());
        buf.push(b'\t');
        push_u32(buf, pos);
        buf.push(b'\t');
        push_u32(buf, aln.mapq as u32);
        buf.push(b'\t');
        append_cigar(buf, &aln.cigar);
        buf.extend_from_slice(b"\t*\t0\t0\t");
        buf.extend_from_slice(&read.seq);
        buf.push(b'\t');
        match read.qual.as_ref() {
            Some(q) => buf.extend_from_slice(q),
            None => buf.push(b'*'),
        }

        if cfg.write_nm {
            buf.extend_from_slice(b"\tNM:i:");
            push_u32(buf, aln.nm);
        }
        if cfg.write_md {
            buf.extend_from_slice(b"\tMD:Z:");
            buf.extend_from_slice(aln.md.as_bytes());
        }
        if cfg.write_as {
            buf.extend_from_slice(b"\tAS:i:");
            push_i32(buf, aln.as_score);
        }
        if cfg.write_xs {
            if let Some(xs) = aln.xs_score {
                buf.extend_from_slice(b"\tXS:i:");
                push_i32(buf, xs);
            }
        }
        if cfg.write_rg {
            if let Some(rg) = read_group {
                buf.extend_from_slice(b"\tRG:Z:");
                buf.extend_from_slice(extract_rg_id(rg).as_bytes());
            }
        }
        if let Some(extra) = extra_tags {
            buf.extend_from_slice(extra);
        }
        buf.push(b'\n');
    }

    pub fn append_xa(&self, buf: &mut Vec<u8>, alignments: &[Alignment]) -> bool {
        if alignments.len() <= 1 {
            return false;
        }
        buf.extend_from_slice(b"\tXA:Z:");
        for aln in alignments.iter().skip(1) {
            let rname = &self.reference.sequences[aln.ref_id as usize].name;
            buf.extend_from_slice(rname.as_bytes());
            buf.push(b',');
            if aln.is_rev {
                buf.push(b'-');
            } else {
                buf.push(b'+');
            }
            push_u32(buf, aln.ref_start + 1);
            buf.push(b',');
            append_cigar(buf, &aln.cigar);
            buf.push(b',');
            push_u32(buf, aln.nm);
            buf.push(b';');
        }
        true
    }

    pub fn append_sa(&self, buf: &mut Vec<u8>, alignments: &[Alignment]) -> bool {
        let mut added = false;
        for aln in alignments.iter().filter(|a| a.is_supplementary) {
            if !added {
                buf.extend_from_slice(b"\tSA:Z:");
                added = true;
            }
            let rname = &self.reference.sequences[aln.ref_id as usize].name;
            buf.extend_from_slice(rname.as_bytes());
            buf.push(b',');
            push_u32(buf, aln.ref_start + 1);
            buf.push(b',');
            buf.push(if aln.is_rev { b'-' } else { b'+' });
            buf.push(b',');
            append_cigar(buf, &aln.cigar);
            buf.push(b',');
            push_u32(buf, aln.mapq as u32);
            buf.push(b',');
            push_u32(buf, aln.nm);
            buf.push(b';');
        }
        added
    }

    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush().context("flush SAM output")?;
        Ok(())
    }
}

fn append_cigar(buf: &mut Vec<u8>, ops: &[CigarOp]) {
    if ops.is_empty() {
        buf.push(b'*');
        return;
    }
    for op in ops {
        push_u32(buf, op.len);
        buf.push(match op.op {
            CigarKind::Match => b'M',
            CigarKind::Ins => b'I',
            CigarKind::Del => b'D',
            CigarKind::SoftClip => b'S',
        });
    }
}

fn sam_flag(aln: &Alignment) -> u16 {
    let mut flag = 0u16;
    if aln.is_rev {
        flag |= 0x10;
    }
    if aln.is_secondary {
        flag |= 0x100;
    }
    if aln.is_supplementary {
        flag |= 0x800;
    }
    flag
}

fn extract_rg_id(rg: &str) -> &str {
    let mut text = rg;
    if let Some(stripped) = text.strip_prefix("@RG\t") {
        text = stripped;
    }
    for field in text.split('\t') {
        if let Some(id) = field.strip_prefix("ID:") {
            return id;
        }
    }
    text
}

fn push_u32(buf: &mut Vec<u8>, mut v: u32) {
    if v == 0 {
        buf.push(b'0');
        return;
    }
    let mut tmp = [0u8; 10];
    let mut i = 0usize;
    while v > 0 {
        tmp[i] = b'0' + (v % 10) as u8;
        v /= 10;
        i += 1;
    }
    for idx in (0..i).rev() {
        buf.push(tmp[idx]);
    }
}

fn push_i32(buf: &mut Vec<u8>, v: i32) {
    if v < 0 {
        buf.push(b'-');
        push_u32(buf, (-v) as u32);
    } else {
        push_u32(buf, v as u32);
    }
}

#[allow(dead_code)]
fn is_softclip(op: &CigarOp) -> bool {
    matches!(op.op, CigarKind::SoftClip)
}
