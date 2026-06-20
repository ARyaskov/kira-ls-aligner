use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};
use kira_fastq::FastqReader as KiraFastqReader;
use memmap2::Mmap;
use needletail::parse_fastx_reader;

use crate::types::{
    Alignment, CigarKind, CigarOp, MateInfo, PairRole, ReadRecord, RefBases, RefSeq, Reference,
};

/// Load reference sequences from FASTA using mmap.
pub fn read_reference<P: AsRef<Path>>(path: P) -> Result<Reference> {
    let mut reader = open_fastx_reader(path)?.reader;
    let mut sequences = Vec::new();
    while let Some(record) = reader.next() {
        let record = record.context("read reference record")?;
        let raw = record.id();
        // SAM/BAM `SN:` forbids whitespace; trim defensively.
        let trimmed = raw
            .split(|b: &u8| b.is_ascii_whitespace())
            .next()
            .unwrap_or(raw);
        let name = String::from_utf8_lossy(trimmed).to_string();
        let mut seq = record.seq().to_vec();
        normalize_bases(&mut seq);
        sequences.push(RefSeq {
            name,
            bases: RefBases::Owned(seq),
        });
    }
    Ok(Reference { sequences })
}

/// Pairing strategy for `ReadStream`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IngestMode {
    Unpaired,
    TwoFile,
    Interleaved,
}

/// Stream reads in batches of approximate total bases using mmap.
pub struct ReadStream {
    readers: Vec<FastqReaderWithProgress>,
    current: usize,
    batch_bases: usize,
    progress: ReadProgress,
    completed_bytes: u64,
    mode: IngestMode,
    /// Counter for assigning stable `pair_id`s when in a paired mode.
    pair_counter: u64,
}

impl ReadStream {
    pub fn new<P: AsRef<Path>>(path: P, batch_bases: usize) -> Result<Self> {
        Self::new_multi(&[path.as_ref().to_path_buf()], batch_bases)
    }

    pub fn new_multi(paths: &[std::path::PathBuf], batch_bases: usize) -> Result<Self> {
        Self::new_multi_with_mode(paths, batch_bases, IngestMode::Unpaired)
    }

    /// Construct a `ReadStream` with an explicit pairing mode.
    pub fn new_multi_with_mode(
        paths: &[std::path::PathBuf],
        batch_bases: usize,
        mode: IngestMode,
    ) -> Result<Self> {
        if paths.is_empty() {
            return Err(anyhow::anyhow!("no reads input provided"));
        }
        match mode {
            IngestMode::TwoFile if paths.len() != 2 => {
                return Err(anyhow::anyhow!(
                    "paired (two-file) mode requires exactly 2 FASTQ inputs, got {}",
                    paths.len()
                ));
            }
            IngestMode::Interleaved if paths.len() != 1 => {
                return Err(anyhow::anyhow!(
                    "interleaved paired mode requires exactly 1 FASTQ input, got {}",
                    paths.len()
                ));
            }
            _ => {}
        }
        let mut readers = Vec::with_capacity(paths.len());
        let mut total = 0u64;
        for path in paths {
            let fastq = open_fastq_reader(path)?;
            total = total.saturating_add(fastq.total_bytes);
            readers.push(fastq);
        }
        Ok(Self {
            readers,
            current: 0,
            batch_bases,
            progress: ReadProgress {
                total_bytes: total,
                read_bytes: 0,
            },
            completed_bytes: 0,
            mode,
            pair_counter: 0,
        })
    }

    pub fn mode(&self) -> IngestMode {
        self.mode
    }

    pub fn next_batch(&mut self) -> Result<Option<Vec<ReadRecord>>> {
        match self.mode {
            IngestMode::Unpaired => self.next_batch_unpaired(),
            IngestMode::TwoFile => self.next_batch_two_file(),
            IngestMode::Interleaved => self.next_batch_interleaved(),
        }
    }

    fn next_batch_unpaired(&mut self) -> Result<Option<Vec<ReadRecord>>> {
        let mut reads = Vec::new();
        let mut bases = 0usize;
        while self.current < self.readers.len() {
            let idx = self.current;
            let record = self.readers[idx]
                .reader
                .next()
                .map_err(|e| anyhow::anyhow!("read FASTQ record: {e:?}"))?;
            if let Some(record) = record {
                let (id, mut seq, qual) = {
                    let id = extract_fastq_id(record.header());
                    let seq = record.seq().to_vec();
                    let qual = Some(record.qual().to_vec());
                    (id, seq, qual)
                };
                normalize_bases(&mut seq);
                bases += seq.len();
                reads.push(ReadRecord {
                    id,
                    seq,
                    qual,
                    pair_role: PairRole::Unpaired,
                    repeat_min_occ: 1,
                });

                let consumed = self.readers[idx]
                    .reader
                    .tell()
                    .0
                    .min(self.readers[idx].total_bytes);
                self.progress.read_bytes = self.completed_bytes.saturating_add(consumed);

                if bases >= self.batch_bases {
                    break;
                }
            } else {
                self.completed_bytes = self
                    .completed_bytes
                    .saturating_add(self.readers[idx].total_bytes);
                self.progress.read_bytes = self.completed_bytes;
                self.current += 1;
            }
        }
        if reads.is_empty() {
            Ok(None)
        } else {
            Ok(Some(reads))
        }
    }

    /// Read one record from `readers[which]` and extract.
    fn read_one(&mut self, which: usize) -> Result<Option<(String, Vec<u8>, Option<Vec<u8>>)>> {
        let rec = self.readers[which]
            .reader
            .next()
            .map_err(|e| anyhow::anyhow!("read FASTQ record (file {which}): {e:?}"))?;
        let extracted = rec.map(|r| {
            let id = extract_fastq_id(r.header());
            let mut seq = r.seq().to_vec();
            normalize_bases(&mut seq);
            let qual = Some(r.qual().to_vec());
            (id, seq, qual)
        });
        Ok(extracted)
    }

    /// Two-file paired ingestion.
    fn next_batch_two_file(&mut self) -> Result<Option<Vec<ReadRecord>>> {
        let mut reads = Vec::new();
        let mut bases = 0usize;
        loop {
            let rec1 = self.read_one(0)?;
            let rec2 = self.read_one(1)?;
            match (rec1, rec2) {
                (Some((id1, seq1, qual1)), Some((id2, seq2, qual2))) => {
                    assert_paired_ids(&id1, &id2)?;
                    bases += seq1.len() + seq2.len();
                    self.pair_counter += 1;
                    reads.push(ReadRecord {
                        id: id1,
                        seq: seq1,
                        qual: qual1,
                        pair_role: PairRole::R1,
                        repeat_min_occ: 1,
                    });
                    reads.push(ReadRecord {
                        id: id2,
                        seq: seq2,
                        qual: qual2,
                        pair_role: PairRole::R2,
                        repeat_min_occ: 1,
                    });

                    let consumed1 = self.readers[0]
                        .reader
                        .tell()
                        .0
                        .min(self.readers[0].total_bytes);
                    let consumed2 = self.readers[1]
                        .reader
                        .tell()
                        .0
                        .min(self.readers[1].total_bytes);
                    self.progress.read_bytes = consumed1.saturating_add(consumed2);

                    if bases >= self.batch_bases {
                        break;
                    }
                }
                (None, None) => break,
                (Some(_), None) | (None, Some(_)) => {
                    return Err(anyhow::anyhow!(
                        "paired FASTQ files have unequal record counts (one finished before the other)"
                    ));
                }
            }
        }
        if reads.is_empty() {
            Ok(None)
        } else {
            Ok(Some(reads))
        }
    }

    /// Single-file interleaved ingestion: consecutive records are R1/R2 of the same template.
    fn next_batch_interleaved(&mut self) -> Result<Option<Vec<ReadRecord>>> {
        let mut reads = Vec::new();
        let mut bases = 0usize;
        loop {
            let rec1 = self.read_one(0)?;
            let rec2 = self.read_one(0)?;
            match (rec1, rec2) {
                (Some((id1, seq1, qual1)), Some((id2, seq2, qual2))) => {
                    assert_paired_ids(&id1, &id2)?;
                    bases += seq1.len() + seq2.len();
                    self.pair_counter += 1;
                    reads.push(ReadRecord {
                        id: id1,
                        seq: seq1,
                        qual: qual1,
                        pair_role: PairRole::R1,
                        repeat_min_occ: 1,
                    });
                    reads.push(ReadRecord {
                        id: id2,
                        seq: seq2,
                        qual: qual2,
                        pair_role: PairRole::R2,
                        repeat_min_occ: 1,
                    });

                    let consumed = self.readers[0]
                        .reader
                        .tell()
                        .0
                        .min(self.readers[0].total_bytes);
                    self.progress.read_bytes = consumed;

                    if bases >= self.batch_bases {
                        break;
                    }
                }
                (None, None) => break,
                (Some(_), None) => {
                    return Err(anyhow::anyhow!(
                        "interleaved FASTQ has an odd number of records — R1 without matching R2"
                    ));
                }
                (None, Some(_)) => unreachable!("reader yielded record after None"),
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
}

struct ReadProgress {
    total_bytes: u64,
    read_bytes: u64,
}

impl ReadProgress {
    fn bytes_read(&self) -> u64 {
        self.read_bytes
    }
}

fn open_fastx_reader<P: AsRef<Path>>(path: P) -> Result<FastxReaderWithProgress> {
    let file = File::open(path.as_ref()).context("open FASTA/FASTQ")?;
    let mmap = unsafe { Mmap::map(&file).context("mmap FASTA/FASTQ")? };
    let reader = io::Cursor::new(mmap);
    let fastx_reader = parse_fastx_reader(reader).context("parse FASTA/FASTQ")?;
    Ok(FastxReaderWithProgress {
        reader: fastx_reader,
    })
}

struct FastqReaderWithProgress {
    reader: KiraFastqReader,
    total_bytes: u64,
}

fn open_fastq_reader<P: AsRef<Path>>(path: P) -> Result<FastqReaderWithProgress> {
    let path = path.as_ref();
    let total_bytes = File::open(path)
        .and_then(|f| f.metadata())
        .context("stat FASTQ")?
        .len();
    let reader = KiraFastqReader::from_path_auto(path)
        .map_err(|e| anyhow::anyhow!("open FASTQ/FASTQ.GZ/BGZF: {e:?}"))?;
    Ok(FastqReaderWithProgress {
        reader,
        total_bytes,
    })
}

fn extract_fastq_id(header: &[u8]) -> String {
    let header = if let Some(stripped) = header.strip_prefix(b"@") {
        stripped
    } else {
        header
    };
    let end = header
        .iter()
        .position(|b| b.is_ascii_whitespace())
        .unwrap_or(header.len());
    String::from_utf8_lossy(&header[..end]).to_string()
}

/// Compare R1/R2 read IDs allowing the `/1` / `/2` suffix used by older Illumina pipelines.
fn assert_paired_ids(r1: &str, r2: &str) -> Result<()> {
    let strip = |s: &str| -> String {
        if let Some(stripped) = s.strip_suffix("/1").or_else(|| s.strip_suffix("/2")) {
            stripped.to_string()
        } else {
            s.to_string()
        }
    };
    let c1 = strip(r1);
    let c2 = strip(r2);
    if c1 != c2 {
        return Err(anyhow::anyhow!(
            "paired read IDs disagree: R1={r1:?} R2={r2:?} (canonical: {c1:?} vs {c2:?})"
        ));
    }
    Ok(())
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

/// SAM header configuration.
#[derive(Clone, Debug, Default)]
pub struct HeaderConfig {
    pub read_group: Option<String>,
    pub pg_lines: Vec<String>,
    pub co_lines: Vec<String>,
}

/// Top-level output format selector.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum EmitFormat {
    #[default]
    Sam,
    Paf,
}

/// Output tag configuration.
#[derive(Clone, Copy, Debug)]
pub struct OutputConfig {
    pub format: EmitFormat,
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
            format: EmitFormat::Sam,
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
            format: EmitFormat::Sam,
            write_nm: true,
            write_md: false,
            write_as: true,
            write_xs: false,
            write_rg: true,
            write_xa: false,
            write_sa: false,
        }
    }

    /// Preset for minimap2-style PAF output: 12 mandatory columns plus `tp/NM/AS/dv` tags.
    pub fn paf() -> Self {
        Self {
            format: EmitFormat::Paf,
            write_nm: true,
            write_md: false,
            write_as: true,
            write_xs: false,
            write_rg: false,
            write_xa: false,
            write_sa: false,
        }
    }
}

/// SAM-formatting state independent of the I/O sink.
#[derive(Clone)]
pub struct SamFormatter {
    reference: std::sync::Arc<Reference>,
}

impl SamFormatter {
    pub fn new(reference: std::sync::Arc<Reference>) -> Self {
        // SAM `SN:` forbids whitespace; older sidecar indices may carry the full FASTA description.
        let needs_fixup = reference
            .sequences
            .iter()
            .any(|s| s.name.bytes().any(|b| b.is_ascii_whitespace()));
        let reference = if needs_fixup {
            let mut owned = (*reference).clone();
            for s in owned.sequences.iter_mut() {
                if let Some(pos) = s.name.find(|c: char| c.is_ascii_whitespace()) {
                    s.name.truncate(pos);
                }
            }
            std::sync::Arc::new(owned)
        } else {
            reference
        };
        Self { reference }
    }

    pub fn reference(&self) -> &Reference {
        &self.reference
    }

    pub fn append_unmapped(&self, buf: &mut Vec<u8>, read: &ReadRecord) {
        self.append_unmapped_with_mate(buf, read, None);
    }

    /// Append an unmapped SAM record.
    pub fn append_unmapped_with_mate(
        &self,
        buf: &mut Vec<u8>,
        read: &ReadRecord,
        mate: Option<&MateInfo>,
    ) {
        let mut flag: u16 = 0x4;
        if let Some(m) = mate {
            if m.is_paired {
                flag |= 0x1;
            }
            if m.mate_is_unmapped {
                flag |= 0x8;
            } else if m.mate_is_rev {
                flag |= 0x20;
            }
            if m.is_first_in_pair {
                flag |= 0x40;
            }
            if m.is_second_in_pair {
                flag |= 0x80;
            }
        }

        buf.extend_from_slice(read.id.as_bytes());
        buf.push(b'\t');
        push_u32(buf, flag as u32);
        // RNAME='*'  POS=0  MAPQ=0  CIGAR='*'  for unmapped
        buf.extend_from_slice(b"\t*\t0\t0\t*\t");

        // RNEXT/PNEXT/TLEN
        match mate {
            Some(m) if m.mate_ref_id.is_some() => {
                let mate_rname = &self.reference.sequences[m.mate_ref_id.unwrap() as usize].name;
                buf.extend_from_slice(mate_rname.as_bytes());
                buf.push(b'\t');
                push_u32(buf, m.mate_pos + 1);
                buf.push(b'\t');
                push_i32(buf, 0); // TLEN is 0 when this record is unmapped
                buf.push(b'\t');
            }
            _ => {
                buf.extend_from_slice(b"*\t0\t0\t");
            }
        }

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
        append_cigar_for_seq(buf, &aln.cigar, read.seq.len() as u32);

        // RNEXT / PNEXT / TLEN — non-trivial only when paired.
        if aln.mate.is_paired {
            match aln.mate.mate_ref_id {
                Some(mate_ref_id) if !aln.mate.mate_is_unmapped => {
                    buf.push(b'\t');
                    // SAM convention: emit '=' when RNEXT == RNAME.
                    if mate_ref_id == aln.ref_id {
                        buf.push(b'=');
                    } else {
                        let mate_rname = &self.reference.sequences[mate_ref_id as usize].name;
                        buf.extend_from_slice(mate_rname.as_bytes());
                    }
                    buf.push(b'\t');
                    push_u32(buf, aln.mate.mate_pos + 1);
                    buf.push(b'\t');
                    push_i32(buf, aln.mate.tlen);
                    buf.push(b'\t');
                }
                _ => {
                    buf.extend_from_slice(b"\t=\t");
                    push_u32(buf, pos);
                    buf.extend_from_slice(b"\t0\t");
                }
            }
        } else {
            buf.extend_from_slice(b"\t*\t0\t0\t");
        }

        if aln.is_rev {
            append_reverse_complement(buf, &read.seq);
        } else {
            buf.extend_from_slice(&read.seq);
        }
        buf.push(b'\t');
        match read.qual.as_ref() {
            Some(q) if aln.is_rev => buf.extend(q.iter().rev().copied()),
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
        if let Some(strand) = aln.xs_strand {
            buf.extend_from_slice(b"\tXS:A:");
            buf.push(match strand {
                crate::types::Strand::Forward => b'+',
                crate::types::Strand::Reverse => b'-',
            });
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

    /// Append a PAF record (minimap2-compatible) for one alignment.
    pub fn append_paf(
        &self,
        buf: &mut Vec<u8>,
        read: &ReadRecord,
        aln: &Alignment,
        cfg: OutputConfig,
    ) {
        let tname = &self.reference.sequences[aln.ref_id as usize].name;
        let tlen = self.reference.sequences[aln.ref_id as usize].len(None) as u32;
        let qlen = read.seq.len() as u32;
        let (qstart, qend) = if aln.is_rev {
            (
                qlen.saturating_sub(aln.read_end),
                qlen.saturating_sub(aln.read_start),
            )
        } else {
            (aln.read_start, aln.read_end)
        };

        let mut block_len: u32 = 0;
        for op in &aln.cigar {
            match op.op {
                CigarKind::Match | CigarKind::Ins | CigarKind::Del => block_len += op.len,
                CigarKind::SoftClip | CigarKind::Skipped => {}
            }
        }
        let matches = block_len.saturating_sub(aln.nm);

        buf.extend_from_slice(read.id.as_bytes());
        buf.push(b'\t');
        push_u32(buf, qlen);
        buf.push(b'\t');
        push_u32(buf, qstart);
        buf.push(b'\t');
        push_u32(buf, qend);
        buf.push(b'\t');
        buf.push(if aln.is_rev { b'-' } else { b'+' });
        buf.push(b'\t');
        buf.extend_from_slice(tname.as_bytes());
        buf.push(b'\t');
        push_u32(buf, tlen);
        buf.push(b'\t');
        push_u32(buf, aln.ref_start);
        buf.push(b'\t');
        push_u32(buf, aln.ref_end);
        buf.push(b'\t');
        push_u32(buf, matches);
        buf.push(b'\t');
        push_u32(buf, block_len);
        buf.push(b'\t');
        push_u32(buf, aln.mapq as u32);

        let tp_char = if aln.is_supplementary {
            b'I'
        } else if aln.is_secondary {
            b'S'
        } else {
            b'P'
        };
        buf.extend_from_slice(b"\ttp:A:");
        buf.push(tp_char);

        if cfg.write_nm {
            buf.extend_from_slice(b"\tNM:i:");
            push_u32(buf, aln.nm);
        }
        if cfg.write_as {
            buf.extend_from_slice(b"\tAS:i:");
            push_i32(buf, aln.as_score);
        }
        if block_len > 0 {
            let dv = aln.nm as f64 / block_len as f64;
            buf.extend_from_slice(b"\tdv:f:");
            push_f64_4(buf, dv);
        }
        if let Some(strand) = aln.xs_strand {
            buf.extend_from_slice(b"\tts:A:");
            buf.push(match strand {
                crate::types::Strand::Forward => b'+',
                crate::types::Strand::Reverse => b'-',
            });
        }
        buf.push(b'\n');
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
}

fn append_reverse_complement(buf: &mut Vec<u8>, seq: &[u8]) {
    buf.extend(seq.iter().rev().map(|&b| match b {
        b'A' => b'T',
        b'a' => b't',
        b'C' => b'G',
        b'c' => b'g',
        b'G' => b'C',
        b'g' => b'c',
        b'T' | b'U' => b'A',
        b't' | b'u' => b'a',
        _ => b'N',
    }));
}

/// SAM writer with minimal header support.
pub struct SamWriter {
    writer: BufWriter<Box<dyn Write + Send>>,
    formatter: SamFormatter,
}

impl SamWriter {
    pub fn new<P: AsRef<Path>>(path: Option<P>, reference: Reference) -> Result<Self> {
        let writer: Box<dyn Write + Send> = match path {
            Some(p) => Box::new(File::create(p).context("create SAM output")?),
            None => Box::new(io::stdout()),
        };
        Ok(Self {
            writer: BufWriter::with_capacity(1 << 20, writer),
            formatter: SamFormatter::new(std::sync::Arc::new(reference)),
        })
    }

    /// Build a writer over an arbitrary sink (e.g. an in-memory [`VecSink`]),
    /// for in-process pipelines that consume SAM without touching disk.
    pub fn from_writer(writer: Box<dyn Write + Send>, reference: Reference) -> Self {
        Self {
            writer: BufWriter::with_capacity(1 << 20, writer),
            formatter: SamFormatter::new(std::sync::Arc::new(reference)),
        }
    }

    /// Borrow the formatting half.
    pub fn formatter(&self) -> &SamFormatter {
        &self.formatter
    }

    /// Cloned formatter handle (cheap — `Arc` bump).
    pub fn formatter_handle(&self) -> SamFormatter {
        self.formatter.clone()
    }

    pub fn write_header(&mut self) -> Result<()> {
        self.write_header_with_rg(None)
    }

    pub fn write_header_with_rg(&mut self, read_group: Option<&str>) -> Result<()> {
        self.write_header_with_ctx(&HeaderConfig {
            read_group: read_group.map(|s| s.to_string()),
            ..HeaderConfig::default()
        })
    }

    /// Write the full SAM header from a `HeaderConfig`.
    pub fn write_header_with_ctx(&mut self, cfg: &HeaderConfig) -> Result<()> {
        writeln!(self.writer, "@HD\tVN:1.6\tSO:unsorted")?;
        for seq in &self.formatter.reference.sequences {
            writeln!(self.writer, "@SQ\tSN:{}\tLN:{}", seq.name, seq.len(None))?;
        }
        if let Some(rg) = cfg.read_group.as_deref() {
            if rg.starts_with("@RG") {
                writeln!(self.writer, "{rg}")?;
            } else {
                writeln!(self.writer, "@RG\t{rg}")?;
            }
        }
        for pg in &cfg.pg_lines {
            if pg.starts_with("@PG") {
                writeln!(self.writer, "{pg}")?;
            } else {
                writeln!(self.writer, "@PG\t{pg}")?;
            }
        }
        for co in &cfg.co_lines {
            if co.starts_with("@CO") {
                writeln!(self.writer, "{co}")?;
            } else {
                writeln!(self.writer, "@CO\t{co}")?;
            }
        }
        Ok(())
    }

    pub fn write_batch(&mut self, buf: &[u8]) -> Result<()> {
        self.writer.write_all(buf).context("write SAM batch")?;
        Ok(())
    }

    pub fn append_unmapped(&self, buf: &mut Vec<u8>, read: &ReadRecord) {
        self.formatter.append_unmapped(buf, read);
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
        self.formatter
            .append_alignment(buf, read, aln, read_group, extra_tags, cfg);
    }
    pub fn append_xa(&self, buf: &mut Vec<u8>, alignments: &[Alignment]) -> bool {
        self.formatter.append_xa(buf, alignments)
    }
    pub fn append_sa(&self, buf: &mut Vec<u8>, alignments: &[Alignment]) -> bool {
        self.formatter.append_sa(buf, alignments)
    }

    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush().context("flush SAM output")?;
        Ok(())
    }
}

/// In-memory `Write` sink backed by a shared `Vec<u8>`. Lets a [`SamWriter`]
/// emit SAM bytes into RAM (for in-process pipelines) instead of a file. The
/// caller keeps a clone of the `Arc` to read the bytes back after the writer
/// is flushed and dropped.
#[derive(Clone)]
pub struct VecSink(pub std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

impl Write for VecSink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.lock().expect("VecSink mutex").extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
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
            CigarKind::Skipped => b'N',
        });
    }
}

/// Bytes of the read consumed by a CIGAR: M, I, S, =, X consume query.
#[inline]
fn cigar_query_consumed(ops: &[CigarOp]) -> u32 {
    let mut n: u32 = 0;
    for op in ops {
        match op.op {
            CigarKind::Match | CigarKind::Ins | CigarKind::SoftClip => {
                n = n.saturating_add(op.len);
            }
            CigarKind::Del | CigarKind::Skipped => {}
        }
    }
    n
}

/// Emit a CIGAR that matches `seq_len` query bytes exactly.
///
/// CIGAR construction belongs to the aligner. The formatter must not hide an
/// upstream accounting bug by appending an invented soft clip.
fn append_cigar_for_seq(buf: &mut Vec<u8>, ops: &[CigarOp], seq_len: u32) {
    let consumed = cigar_query_consumed(ops);
    if consumed == seq_len {
        append_cigar(buf, ops);
        return;
    }
    debug_assert_eq!(
        consumed, seq_len,
        "alignment CIGAR consumes {consumed} query bases, expected {seq_len}"
    );
    buf.push(b'*');
}

/// Build a SAM bitwise FLAG from an Alignment.
pub(crate) fn sam_flag(aln: &Alignment) -> u16 {
    let mut flag = 0u16;
    if aln.mate.is_paired {
        flag |= 0x1;
        if aln.mate.is_proper_pair {
            flag |= 0x2;
        }
        if aln.mate.mate_is_unmapped {
            flag |= 0x8;
        }
        if aln.mate.mate_is_rev {
            flag |= 0x20;
        }
        if aln.mate.is_first_in_pair {
            flag |= 0x40;
        }
        if aln.mate.is_second_in_pair {
            flag |= 0x80;
        }
    }
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

/// Append a non-negative f64 with 4 fractional digits — matches `printf("%.4f", x)`.
fn push_f64_4(buf: &mut Vec<u8>, v: f64) {
    let v = if v < 0.0 { 0.0 } else { v };
    // Multiply by 10_000 and round to nearest integer, then split.
    let scaled = (v * 10_000.0 + 0.5) as u64;
    let int_part = (scaled / 10_000) as u32;
    let frac_part = (scaled % 10_000) as u32;
    push_u32(buf, int_part);
    buf.push(b'.');
    // Pad fractional part to 4 digits.
    let mut tmp = [0u8; 4];
    let mut f = frac_part;
    for slot in tmp.iter_mut().rev() {
        *slot = b'0' + (f % 10) as u8;
        f /= 10;
    }
    buf.extend_from_slice(&tmp);
}

#[allow(dead_code)]
fn is_softclip(op: &CigarOp) -> bool {
    matches!(op.op, CigarKind::SoftClip)
}
