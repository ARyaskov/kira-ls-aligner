use std::fs::File;
use std::io::{self, BufRead, BufWriter, Read, Write};
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
    /// Keep the FASTQ comment on each read (`-C`).
    keep_comment: bool,
}

/// `-` names standard input, as it does for bwa-mem and samtools.
#[inline]
pub fn is_stdin_path(path: &Path) -> bool {
    path.as_os_str() == "-"
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
        Self::new_multi_with_opts(paths, batch_bases, mode, false)
    }

    /// [`Self::new_multi_with_mode`] that also keeps the FASTQ comment (the
    /// header text after the first whitespace) on every read, for `-C`.
    pub fn new_multi_with_opts(
        paths: &[std::path::PathBuf],
        batch_bases: usize,
        mode: IngestMode,
        keep_comment: bool,
    ) -> Result<Self> {
        if paths.is_empty() {
            return Err(anyhow::anyhow!("no reads input provided"));
        }
        if paths.iter().filter(|p| is_stdin_path(p)).count() > 1 {
            return Err(anyhow::anyhow!(
                "only one reads input may be `-` (stdin)"
            ));
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
            keep_comment,
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
        let keep_comment = self.keep_comment;
        while self.current < self.readers.len() {
            let idx = self.current;
            let record = self.readers[idx]
                .reader
                .next()
                .map_err(|e| anyhow::anyhow!("read FASTQ record: {e:?}"))?;
            if let Some(record) = record {
                let (id, mut seq, qual, comment) = {
                    let id = extract_fastq_id(record.header());
                    let comment = if keep_comment {
                        extract_fastq_comment(record.header())
                    } else {
                        None
                    };
                    let seq = record.seq().to_vec();
                    let qual = Some(record.qual().to_vec());
                    (id, seq, qual, comment)
                };
                normalize_bases(&mut seq);
                bases += seq.len();
                reads.push(ReadRecord {
                    id,
                    seq,
                    qual,
                    pair_role: PairRole::Unpaired,
                    repeat_min_occ: 1,
                    comment,
                });

                let consumed = self.readers[idx].consumed();
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
    #[allow(clippy::type_complexity)]
    fn read_one(&mut self, which: usize) -> Result<Option<ExtractedRead>> {
        let keep_comment = self.keep_comment;
        let rec = self.readers[which]
            .reader
            .next()
            .map_err(|e| anyhow::anyhow!("read FASTQ record (file {which}): {e:?}"))?;
        let extracted = rec.map(|r| {
            let id = extract_fastq_id(r.header());
            let comment = if keep_comment {
                extract_fastq_comment(r.header())
            } else {
                None
            };
            let mut seq = r.seq().to_vec();
            normalize_bases(&mut seq);
            let qual = Some(r.qual().to_vec());
            (id, seq, qual, comment)
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
                (Some((id1, seq1, qual1, c1)), Some((id2, seq2, qual2, c2))) => {
                    assert_paired_ids(&id1, &id2)?;
                    bases += seq1.len() + seq2.len();
                    self.pair_counter += 1;
                    reads.push(ReadRecord {
                        id: id1,
                        seq: seq1,
                        qual: qual1,
                        pair_role: PairRole::R1,
                        repeat_min_occ: 1,
                        comment: c1,
                    });
                    reads.push(ReadRecord {
                        id: id2,
                        seq: seq2,
                        qual: qual2,
                        pair_role: PairRole::R2,
                        repeat_min_occ: 1,
                        comment: c2,
                    });

                    let consumed1 = self.readers[0].consumed();
                    let consumed2 = self.readers[1].consumed();
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
                (Some((id1, seq1, qual1, c1)), Some((id2, seq2, qual2, c2))) => {
                    assert_paired_ids(&id1, &id2)?;
                    bases += seq1.len() + seq2.len();
                    self.pair_counter += 1;
                    reads.push(ReadRecord {
                        id: id1,
                        seq: seq1,
                        qual: qual1,
                        pair_role: PairRole::R1,
                        repeat_min_occ: 1,
                        comment: c1,
                    });
                    reads.push(ReadRecord {
                        id: id2,
                        seq: seq2,
                        qual: qual2,
                        pair_role: PairRole::R2,
                        repeat_min_occ: 1,
                        comment: c2,
                    });

                    self.progress.read_bytes = self.readers[0].consumed();

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
    basis: ProgressBasis,
}

impl FastqReaderWithProgress {
    /// Position in the file's own bytes, for the progress bar. `tell()` means
    /// different things per backend, so the conversion lives here rather than
    /// at the three call sites.
    fn consumed(&self) -> u64 {
        let voff = self.reader.tell();
        let bytes = match self.basis {
            ProgressBasis::FileBytes => voff.get(),
            ProgressBasis::BgzfVirtual => voff.compressed(),
            ProgressBasis::Decoded => voff.get() / FASTQ_GZIP_RATIO,
        };
        bytes.min(self.total_bytes)
    }
}

/// How a reader's `tell()` maps onto progress through the file.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProgressBasis {
    /// A byte offset into the file itself (plain FASTQ).
    FileBytes,
    /// A BGZF virtual offset — its block offset is a real file position.
    BgzfVirtual,
    /// Decoded bytes consumed (gzip, and the parallel BGZF reader). Scaled
    /// back into file bytes by [`FASTQ_GZIP_RATIO`].
    Decoded,
}

/// Decompressed-to-compressed size ratio assumed when a reader reports decoded
/// bytes: 2.9x on HG002 (748 MB of BGZF holding 2.2 GB of FASTQ). Only scales
/// the progress bar, which clamps at the file size either way.
const FASTQ_GZIP_RATIO: u64 = 3;

/// Compression of an input file, from its magic bytes rather than its name —
/// `bgzip` writes BGZF into plain `.gz` names.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InputCompression {
    Plain,
    Gzip,
    Bgzf,
}

/// BGZF is gzip with an `FEXTRA` field carrying the `BC` subfield (SAM spec
/// §4.1); anything else with the gzip magic is ordinary gzip.
fn detect_compression(path: &Path) -> Result<InputCompression> {
    let mut head = [0u8; 16];
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut filled = 0usize;
    while filled < head.len() {
        match f.read(&mut head[filled..]).context("read FASTQ header bytes")? {
            0 => break,
            n => filled += n,
        }
    }
    if filled < 2 || head[0] != 0x1f || head[1] != 0x8b {
        return Ok(InputCompression::Plain);
    }
    let has_extra = filled >= 4 && (head[3] & 0x04) != 0;
    if has_extra && filled >= 14 && head[12] == b'B' && head[13] == b'C' {
        return Ok(InputCompression::Bgzf);
    }
    Ok(InputCompression::Gzip)
}

/// `KIRA_BGZF_THREADS` — workers used to inflate a BGZF input, samtools' `-@`
/// for FASTQ input. BGZF blocks are independent, so this scales: HG002 R1
/// decodes in 5.0 s on one thread, 2.1 s on two and 1.1 s on four. Decoding
/// overlaps alignment, so the default stops at 2 rather than taking cores the
/// aligner wants — and it is *per input file*, so a two-file paired run spends
/// twice this. `1` keeps the single-threaded backend (and exact, unestimated
/// BGZF progress); `0` selects the default.
fn bgzf_decode_threads() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let requested = std::env::var("KIRA_BGZF_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        if requested > 0 {
            requested.min(cores.max(1))
        } else if cores >= 4 {
            2
        } else {
            1
        }
    })
}

/// `(id, seq, qual, comment)` as pulled off one FASTQ record.
type ExtractedRead = (String, Vec<u8>, Option<Vec<u8>>, Option<String>);

fn open_fastq_reader<P: AsRef<Path>>(path: P) -> Result<FastqReaderWithProgress> {
    let path = path.as_ref();
    if is_stdin_path(path) {
        return open_stdin_reader();
    }
    let total_bytes = File::open(path)
        .and_then(|f| f.metadata())
        .context("stat FASTQ")?
        .len();
    let open_err = |e| anyhow::anyhow!("open FASTQ/FASTQ.GZ/BGZF: {e:?}");
    let threads = bgzf_decode_threads();
    let (reader, basis) = match detect_compression(path)? {
        InputCompression::Bgzf if threads > 1 => (
            KiraFastqReader::from_bgzf_path_parallel(path, threads).map_err(open_err)?,
            ProgressBasis::Decoded,
        ),
        InputCompression::Bgzf => (
            KiraFastqReader::from_path_auto(path).map_err(open_err)?,
            ProgressBasis::BgzfVirtual,
        ),
        InputCompression::Gzip => (
            KiraFastqReader::from_path_auto(path).map_err(open_err)?,
            ProgressBasis::Decoded,
        ),
        InputCompression::Plain => (
            KiraFastqReader::from_path_auto(path).map_err(open_err)?,
            ProgressBasis::FileBytes,
        ),
    };
    Ok(FastqReaderWithProgress {
        reader,
        total_bytes,
        basis,
    })
}

/// Reads from standard input, plain or gzip/BGZF (sniffed from the first two
/// bytes, as `bwa mem ref -` / `samtools import -` do). No byte total is
/// known, so the progress bar stays indeterminate.
fn open_stdin_reader() -> Result<FastqReaderWithProgress> {
    let mut stdin = io::BufReader::with_capacity(1 << 20, io::stdin());
    let magic = {
        let buf = stdin.fill_buf().context("read stdin")?;
        (buf.first().copied(), buf.get(1).copied())
    };
    // CRLF is normalised *after* decompression: a pipe delivers the file in
    // arbitrary pieces, and the stream parser only tolerates CRLF when both
    // bytes arrive in the same piece.
    let reader = if magic == (Some(0x1f), Some(0x8b)) {
        // Multi-member so BGZF (one member per block) decodes to the end.
        let dec = flate2::bufread::MultiGzDecoder::new(stdin);
        KiraFastqReader::from_reader(io::BufReader::with_capacity(1 << 20, CrlfToLf::new(dec)))
    } else {
        KiraFastqReader::from_reader(io::BufReader::with_capacity(1 << 20, CrlfToLf::new(stdin)))
    };
    Ok(FastqReaderWithProgress {
        reader,
        total_bytes: 0,
        basis: ProgressBasis::Decoded,
    })
}

/// `Read` adapter turning `\r\n` into `\n` across read boundaries, so FASTQ
/// piped from Windows tools parses the same as it does from a file.
struct CrlfToLf<R: io::Read> {
    inner: R,
    raw: Vec<u8>,
    staged: Vec<u8>,
    pos: usize,
    /// A `\r` was seen at the end of the previous piece; its fate depends on
    /// the next byte.
    pending_cr: bool,
}

impl<R: io::Read> CrlfToLf<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            raw: vec![0u8; 256 * 1024],
            staged: Vec::with_capacity(256 * 1024),
            pos: 0,
            pending_cr: false,
        }
    }

    /// Pull the next non-empty piece from `inner` into `staged`, normalised.
    /// Leaves `staged` empty only at EOF.
    fn refill(&mut self) -> io::Result<()> {
        self.staged.clear();
        self.pos = 0;
        loop {
            let n = self.inner.read(&mut self.raw)?;
            if n == 0 {
                if self.pending_cr {
                    self.staged.push(b'\r');
                    self.pending_cr = false;
                }
                return Ok(());
            }
            for &b in &self.raw[..n] {
                match (self.pending_cr, b) {
                    (true, b'\n') => {
                        self.staged.push(b'\n');
                        self.pending_cr = false;
                    }
                    (true, b'\r') => self.staged.push(b'\r'),
                    (true, other) => {
                        self.staged.push(b'\r');
                        self.staged.push(other);
                        self.pending_cr = false;
                    }
                    (false, b'\r') => self.pending_cr = true,
                    (false, other) => self.staged.push(other),
                }
            }
            if !self.staged.is_empty() {
                return Ok(());
            }
        }
    }
}

impl<R: io::Read> io::Read for CrlfToLf<R> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        if self.pos == self.staged.len() {
            self.refill()?;
            if self.staged.is_empty() {
                return Ok(0);
            }
        }
        let n = out.len().min(self.staged.len() - self.pos);
        out[..n].copy_from_slice(&self.staged[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
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

/// The FASTQ comment: header text after the first run of whitespace, with
/// tabs preserved (bwa-mem `-C` appends it verbatim, so `BC:Z:x\tRX:Z:y`
/// lands as two SAM tags). `None` when the header has no comment.
fn extract_fastq_comment(header: &[u8]) -> Option<String> {
    let header = header.strip_prefix(b"@").unwrap_or(header);
    let ws = header.iter().position(|b| b.is_ascii_whitespace())?;
    let rest = &header[ws..];
    let start = rest.iter().position(|b| !b.is_ascii_whitespace())?;
    let text = &rest[start..];
    let end = text
        .iter()
        .rposition(|b| !b.is_ascii_whitespace())
        .map(|i| i + 1)
        .unwrap_or(0);
    if end == 0 {
        return None;
    }
    Some(String::from_utf8_lossy(&text[..end]).into_owned())
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
    /// Verbatim header lines inserted after `@SQ`/`@RG` (bwa-mem `-H`).
    pub extra_lines: Vec<String>,
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
    /// `MC:Z` / `MQ:i` / `ms:i` on paired records — what `samtools fixmate -m`
    /// would add, so `samtools markdup` runs directly on the aligner output.
    pub write_mate_tags: bool,
    /// bwa-mem `-M`: flag supplementary segments as secondary (0x100) for
    /// Picard-era tools that reject 0x800.
    pub split_as_secondary: bool,
    /// bwa-mem `-Y`: keep supplementary segments soft-clipped. Off (the bwa
    /// default) hard-clips them, dropping the unaligned bases from SEQ/QUAL.
    pub soft_clip_supplementary: bool,
    /// bwa-mem `-h INT`: emit `XA` only when the read has at most this many
    /// close secondary hits; more than that is reported through MAPQ alone.
    pub xa_max: u32,
    /// bwa-mem `-C`: append the FASTQ comment verbatim to every record.
    pub append_comment: bool,
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
            write_mate_tags: true,
            split_as_secondary: false,
            soft_clip_supplementary: false,
            xa_max: 5,
            append_comment: false,
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
            write_mate_tags: true,
            split_as_secondary: false,
            soft_clip_supplementary: false,
            xa_max: 5,
            append_comment: false,
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
            write_mate_tags: false,
            split_as_secondary: false,
            soft_clip_supplementary: true,
            xa_max: 0,
            append_comment: false,
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
        self.append_unmapped_full(buf, read, mate, None, false);
    }

    /// Append an unmapped SAM record with optional trailing tags (mate tags
    /// for a read whose mate mapped) and the `-C` comment.
    pub fn append_unmapped_full(
        &self,
        buf: &mut Vec<u8>,
        read: &ReadRecord,
        mate: Option<&MateInfo>,
        extra_tags: Option<&[u8]>,
        append_comment: bool,
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
        if let Some(extra) = extra_tags {
            buf.extend_from_slice(extra);
        }
        if append_comment {
            append_fastq_comment(buf, read);
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
        let flag = sam_flag_with(aln, cfg.split_as_secondary);

        // bwa-mem hard-clips supplementary segments unless `-Y`: the clipped
        // bases become `H` and leave SEQ/QUAL, so a chimeric read's sequence is
        // stored once, on its primary record.
        let hard_clip = aln.is_supplementary && !cfg.soft_clip_supplementary;
        let (clip_lead, clip_trail) = if hard_clip {
            terminal_softclips(&aln.cigar)
        } else {
            (0, 0)
        };

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
        append_cigar_for_seq(buf, &aln.cigar, read.seq.len() as u32, hard_clip);

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

        // SEQ/QUAL in reference orientation, minus the hard-clipped ends.
        // `clip_lead` counts from the CIGAR's start, which is the start of
        // the *oriented* sequence, so the trim applies after orientation.
        let seq_len = read.seq.len();
        let keep_end = seq_len.saturating_sub(clip_trail as usize);
        let keep_start = (clip_lead as usize).min(keep_end);
        if aln.is_rev {
            append_reverse_complement_range(buf, &read.seq, keep_start, keep_end);
        } else {
            buf.extend_from_slice(&read.seq[keep_start..keep_end]);
        }
        buf.push(b'\t');
        match read.qual.as_ref() {
            Some(q) if aln.is_rev => {
                let n = q.len();
                buf.extend(q[n - keep_end..n - keep_start].iter().rev().copied())
            }
            Some(q) => buf.extend_from_slice(&q[keep_start..keep_end]),
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
        if cfg.write_xs && let Some(xs) = aln.xs_score {
            buf.extend_from_slice(b"\tXS:i:");
            push_i32(buf, xs);
        }
        if let Some(strand) = aln.xs_strand {
            buf.extend_from_slice(b"\tXS:A:");
            buf.push(match strand {
                crate::types::Strand::Forward => b'+',
                crate::types::Strand::Reverse => b'-',
            });
        }
        if cfg.write_rg && let Some(rg) = read_group {
            buf.extend_from_slice(b"\tRG:Z:");
            buf.extend_from_slice(extract_rg_id(rg).as_bytes());
        }
        if let Some(extra) = extra_tags {
            buf.extend_from_slice(extra);
        }
        if cfg.append_comment {
            append_fastq_comment(buf, read);
        }
        buf.push(b'\n');
    }

    pub fn append_xa(&self, buf: &mut Vec<u8>, alignments: &[Alignment]) -> bool {
        self.append_xa_capped(buf, alignments, u32::MAX)
    }

    /// `XA` for the secondary hits, or nothing when there are more than
    /// `xa_max` of them (bwa-mem `-h`): past that point the list stops
    /// naming a locus and only restates what MAPQ already says.
    pub fn append_xa_capped(&self, buf: &mut Vec<u8>, alignments: &[Alignment], xa_max: u32) -> bool {
        if alignments.len() <= 1 {
            return false;
        }
        let n_secondary = alignments
            .iter()
            .skip(1)
            .filter(|a| !a.is_supplementary)
            .count();
        if n_secondary == 0 || n_secondary as u32 > xa_max {
            return false;
        }
        buf.extend_from_slice(b"\tXA:Z:");
        for aln in alignments.iter().skip(1).filter(|a| !a.is_supplementary) {
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

/// Append the `-C` comment, if the read carries one, as trailing tag text.
#[inline]
fn append_fastq_comment(buf: &mut Vec<u8>, read: &ReadRecord) {
    if let Some(c) = read.comment.as_deref() {
        buf.push(b'\t');
        buf.extend_from_slice(c.as_bytes());
    }
}

/// Lengths of the leading and trailing soft clips of a CIGAR.
fn terminal_softclips(ops: &[CigarOp]) -> (u32, u32) {
    let lead = match ops.first() {
        Some(op) if op.op == CigarKind::SoftClip => op.len,
        _ => 0,
    };
    let trail = match ops.last() {
        Some(op) if op.op == CigarKind::SoftClip && ops.len() > 1 => op.len,
        _ => 0,
    };
    (lead, trail)
}

/// Reverse complement of `seq`, keeping only oriented positions
/// `[keep_start, keep_end)` of the reverse-complemented string.
fn append_reverse_complement_range(buf: &mut Vec<u8>, seq: &[u8], keep_start: usize, keep_end: usize) {
    let n = seq.len();
    // Oriented index i ↔ original index n-1-i.
    append_reverse_complement(buf, &seq[n - keep_end..n - keep_start]);
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
        for line in &cfg.extra_lines {
            writeln!(self.writer, "{line}")?;
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
/// upstream accounting bug by appending an invented soft clip. With
/// `hard_clip` the terminal soft clips are written as `H` (bwa-mem's
/// supplementary convention); the consumed-length check still runs against
/// the soft-clipped form, since that is what the aligner produced.
fn append_cigar_for_seq(buf: &mut Vec<u8>, ops: &[CigarOp], seq_len: u32, hard_clip: bool) {
    let consumed = cigar_query_consumed(ops);
    if consumed != seq_len {
        debug_assert_eq!(
            consumed, seq_len,
            "alignment CIGAR consumes {consumed} query bases, expected {seq_len}"
        );
        buf.push(b'*');
        return;
    }
    if !hard_clip || ops.is_empty() {
        append_cigar(buf, ops);
        return;
    }
    let last = ops.len() - 1;
    for (i, op) in ops.iter().enumerate() {
        push_u32(buf, op.len);
        let terminal_clip = op.op == CigarKind::SoftClip && (i == 0 || i == last);
        buf.push(match op.op {
            CigarKind::SoftClip if terminal_clip => b'H',
            CigarKind::Match => b'M',
            CigarKind::Ins => b'I',
            CigarKind::Del => b'D',
            CigarKind::SoftClip => b'S',
            CigarKind::Skipped => b'N',
        });
    }
}

/// Build a SAM bitwise FLAG from an Alignment. With bwa-mem `-M`
/// (`split_as_secondary`) a supplementary segment is flagged 0x100
/// (secondary) instead of 0x800, for tools that predate supplementary records.
pub(crate) fn sam_flag_with(aln: &Alignment, split_as_secondary: bool) -> u16 {
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
        flag |= if split_as_secondary { 0x100 } else { 0x800 };
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

#[cfg(test)]
mod format_tests {
    use super::*;
    use crate::types::{AlignmentKind, RefBases, RefSeq};

    fn reference() -> Reference {
        Reference {
            sequences: vec![RefSeq {
                name: "chr1".into(),
                bases: RefBases::Owned(vec![b'A'; 1000]),
            }],
        }
    }

    fn supp_alignment() -> Alignment {
        Alignment {
            kind: AlignmentKind::DpAligned,
            ref_id: 0,
            ref_start: 100,
            ref_end: 170,
            read_start: 80,
            read_end: 150,
            cigar: vec![
                CigarOp { len: 80, op: CigarKind::SoftClip },
                CigarOp { len: 70, op: CigarKind::Match },
            ],
            score: 70,
            mapq: 44,
            is_rev: false,
            is_secondary: false,
            is_supplementary: true,
            nm: 0,
            md: "70".into(),
            as_score: 70,
            xs_score: None,
            xs_strand: None,
            mate: MateInfo::default(),
        }
    }

    fn fields(line: &[u8]) -> Vec<String> {
        String::from_utf8_lossy(line)
            .trim_end()
            .split('\t')
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn supplementary_is_hard_clipped_by_default_and_soft_clipped_with_y() {
        let fmt = SamFormatter::new(std::sync::Arc::new(reference()));
        let read = ReadRecord::new_unpaired(
            "q".into(),
            (0..150u8).map(|i| b"ACGT"[(i % 4) as usize]).collect(),
            Some(vec![b'I'; 150]),
        );
        let aln = supp_alignment();

        let mut buf = Vec::new();
        fmt.append_alignment(&mut buf, &read, &aln, None, None, OutputConfig::full());
        let f = fields(&buf);
        assert_eq!(f[1], "2048");
        assert_eq!(f[5], "80H70M");
        assert_eq!(f[9].len(), 70, "hard-clipped bases leave SEQ");
        assert_eq!(f[10].len(), 70);
        assert_eq!(f[9].as_bytes(), &read.seq[80..150]);

        let mut cfg = OutputConfig::full();
        cfg.soft_clip_supplementary = true;
        buf.clear();
        fmt.append_alignment(&mut buf, &read, &aln, None, None, cfg);
        let f = fields(&buf);
        assert_eq!(f[5], "80S70M");
        assert_eq!(f[9].len(), 150);

        let mut cfg = OutputConfig::full();
        cfg.split_as_secondary = true;
        buf.clear();
        fmt.append_alignment(&mut buf, &read, &aln, None, None, cfg);
        assert_eq!(fields(&buf)[1], "256", "-M reports the split segment as secondary");
    }

    #[test]
    fn hard_clip_trims_the_oriented_sequence_for_reverse_strand_records() {
        let fmt = SamFormatter::new(std::sync::Arc::new(reference()));
        let seq: Vec<u8> = (0..150u8).map(|i| b"ACGT"[(i % 4) as usize]).collect();
        let read = ReadRecord::new_unpaired("q".into(), seq.clone(), Some(vec![b'I'; 150]));
        let mut aln = supp_alignment();
        aln.is_rev = true;
        let mut buf = Vec::new();
        fmt.append_alignment(&mut buf, &read, &aln, None, None, OutputConfig::full());
        let f = fields(&buf);
        // Oriented (reverse-complemented) read, minus its first 80 bases.
        let mut rc = Vec::new();
        append_reverse_complement(&mut rc, &seq);
        assert_eq!(f[9].as_bytes(), &rc[80..150]);
    }

    #[test]
    fn comment_is_appended_verbatim_when_requested() {
        let fmt = SamFormatter::new(std::sync::Arc::new(reference()));
        let mut read = ReadRecord::new_unpaired("q".into(), vec![b'A'; 20], None);
        read.comment = Some("BC:Z:ACGT\tRX:Z:TTGA".into());
        let mut cfg = OutputConfig::full();
        cfg.append_comment = true;
        let mut buf = Vec::new();
        fmt.append_unmapped_full(&mut buf, &read, None, None, true);
        let f = fields(&buf);
        assert_eq!(&f[f.len() - 2..], ["BC:Z:ACGT", "RX:Z:TTGA"]);
        buf.clear();
        fmt.append_unmapped_full(&mut buf, &read, None, None, false);
        assert!(!String::from_utf8_lossy(&buf).contains("BC:Z"));
        let _ = cfg;
    }

    #[test]
    fn fastq_comment_extraction() {
        assert_eq!(extract_fastq_comment(b"@r1 BC:Z:AC\tRX:Z:T\n"), Some("BC:Z:AC\tRX:Z:T".into()));
        assert_eq!(extract_fastq_comment(b"@r1"), None);
        assert_eq!(extract_fastq_comment(b"@r1   "), None);
        assert_eq!(extract_fastq_id(b"@r1 BC:Z:AC"), "r1");
    }

    #[test]
    fn xa_cap_suppresses_long_lists() {
        let fmt = SamFormatter::new(std::sync::Arc::new(reference()));
        let mut primary = supp_alignment();
        primary.is_supplementary = false;
        let mut alns = vec![primary];
        for _ in 0..3 {
            let mut s = supp_alignment();
            s.is_supplementary = false;
            s.is_secondary = true;
            alns.push(s);
        }
        let mut buf = Vec::new();
        assert!(fmt.append_xa_capped(&mut buf, &alns, 5));
        assert!(buf.starts_with(b"\tXA:Z:"));
        buf.clear();
        assert!(!fmt.append_xa_capped(&mut buf, &alns, 2));
        assert!(buf.is_empty());
    }

    /// The BGZF magic is what selects the parallel inflate path, and `bgzip`
    /// writes BGZF under a plain `.gz` name, so the test uses bytes not names.
    #[test]
    fn compression_is_detected_from_magic_not_extension() {
        let dir = std::env::temp_dir().join("kira_detect_compression");
        std::fs::create_dir_all(&dir).unwrap();

        let plain = dir.join("a.fastq");
        std::fs::write(&plain, b"@r\nACGT\n+\n!!!!\n").unwrap();
        assert_eq!(detect_compression(&plain).unwrap(), InputCompression::Plain);

        // gzip magic, deflate, no FEXTRA.
        let gz = dir.join("b.fastq.gz");
        std::fs::write(&gz, [0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0, 0xff, 1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(detect_compression(&gz).unwrap(), InputCompression::Gzip);

        // BGZF: FEXTRA set (flag 0x04) with the `BC` subfield at bytes 12..14.
        let bgzf = dir.join("c.fastq.gz");
        std::fs::write(
            &bgzf,
            [0x1f, 0x8b, 0x08, 0x04, 0, 0, 0, 0, 0, 0xff, 0x06, 0x00, b'B', b'C', 0x02, 0x00],
        )
        .unwrap();
        assert_eq!(detect_compression(&bgzf).unwrap(), InputCompression::Bgzf);

        // A file too short to carry magic is plain, not an error.
        let tiny = dir.join("d.fastq");
        std::fs::write(&tiny, b"@").unwrap();
        assert_eq!(detect_compression(&tiny).unwrap(), InputCompression::Plain);
    }

    #[test]
    fn progress_basis_converts_tell_into_file_bytes() {
        // A BGZF virtual offset packs the block's compressed offset above the
        // low 16 bits; reading it raw saturates the bar instantly.
        let voff = kira_fastq::offset::VirtualOffset::new(4096, 300);
        assert_eq!(voff.compressed(), 4096);
        assert!(voff.get() > 1 << 20, "raw virtual offset dwarfs the file position");

        // Decoded bytes are scaled back down by the assumed FASTQ ratio.
        assert_eq!(3_000u64 / FASTQ_GZIP_RATIO, 1_000);
    }

    #[test]
    fn crlf_to_lf_handles_pairs_split_across_reads() {
        struct Chunked<'a>(&'a [&'a [u8]], usize);
        impl io::Read for Chunked<'_> {
            fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
                if self.1 >= self.0.len() {
                    return Ok(0);
                }
                let c = self.0[self.1];
                self.1 += 1;
                out[..c.len()].copy_from_slice(c);
                Ok(c.len())
            }
        }
        let pieces: &[&[u8]] = &[b"@r\r", b"\nACGT\r\n+", b"\r\nIIII\r", b"\n", b"x\r"];
        let mut out = Vec::new();
        CrlfToLf::new(Chunked(pieces, 0)).read_to_end(&mut out).unwrap();
        assert_eq!(out, b"@r\nACGT\n+\nIIII\nx\r", "a trailing lone CR is preserved");
    }
}
