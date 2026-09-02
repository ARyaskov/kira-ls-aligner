//! Fused binary output for `--emit bam | sorted-bam | cram | sorted-cram`.
//!
//! Alignment batches leave the pipeline as SAM text — the one formatter that
//! knows every tag — and are parsed straight into kira-bam records, in
//! parallel, on a converter thread that overlaps the next batch's alignment.
//! Nothing is written to disk except the final file:
//!
//! * `bam` / `cram` go through kira-bam's writer as they arrive
//!   (multi-threaded BGZF for BAM);
//! * `sorted-*` collects records in memory up to `--sort-memory`, then runs
//!   kira-bam's in-memory coordinate sort (+ fused markdup) and writes once.
//!   A run that outgrows the budget spills everything collected so far to an
//!   unsorted BAM beside the output, keeps streaming into it, and finishes
//!   with kira-bam's external sort — the same result, one more pass.
//!
//! Before this the binary formats went through a full intermediate SAM file
//! (5+ GB on a 30× chr20) that kira-bam then re-parsed.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;

use anyhow::{Context, Result};
use kira_bam::io::{BamWriter, PgInfo, WriteOptions, append_pg, encode_records_into};
use kira_bam::types::OutputFormat;
use noodles_sam as sam;
use rayon::prelude::*;
use sam::alignment::RecordBuf;

use crate::aligner_core::Aligner;
use crate::index::Index;
use crate::io::SamFormatter;
use crate::pipeline::stage6_output;

/// Which fused output was asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmitKind {
    Bam,
    SortedBam,
    Cram,
    SortedCram,
}

impl EmitKind {
    /// The `--emit` spellings that take the fused path; `None` for SAM/PAF.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "bam" => Some(Self::Bam),
            "sorted-bam" => Some(Self::SortedBam),
            "cram" => Some(Self::Cram),
            "sorted-cram" => Some(Self::SortedCram),
            _ => None,
        }
    }

    pub fn is_sorted(self) -> bool {
        matches!(self, Self::SortedBam | Self::SortedCram)
    }

    pub fn is_cram(self) -> bool {
        matches!(self, Self::Cram | Self::SortedCram)
    }

    fn format(self) -> OutputFormat {
        if self.is_cram() {
            OutputFormat::Cram
        } else {
            OutputFormat::Bam
        }
    }
}

/// Everything the fused path needs beyond the aligner itself.
pub struct EmitOptions {
    pub kind: EmitKind,
    pub output: PathBuf,
    /// Reference FASTA — CRAM output needs it (and its `.fai`).
    pub reference: PathBuf,
    pub threads: usize,
    pub markdup: bool,
    pub bai: bool,
    /// `--sort-memory`: `auto`, `4G`, `768M`, or a byte count.
    pub sort_memory: String,
    pub no_pg: bool,
}

/// Bytes of RAM one SAM byte becomes as a decoded `RecordBuf` — sequence and
/// qualities stay one byte per base, but name, CIGAR and tags gain the
/// container overhead. Generous on purpose: overshooting the budget is a
/// slower finish, undershooting it is an OOM.
const RECORD_BYTES_PER_SAM_BYTE: usize = 2;

/// `--sort-memory auto`: a quarter of RAM, never below this.
const AUTO_SORT_MIN_BYTES: usize = 512 << 20;

pub fn run_fused(aligner: &Aligner, index: Index, reads: &[PathBuf], opts: EmitOptions) -> Result<()> {
    let cfg = aligner.config();
    if opts.kind.is_cram() {
        ensure_fai(&opts.reference)?;
    }
    if opts.bai && opts.kind.is_cram() {
        crate::kira_warn!("[KIRA] warning: --bai applies to BAM output; CRAM is written unindexed");
    }

    // Two headers: the aligner's own (unsorted) for anything written before
    // the sort, and the sorted one — `SO:coordinate` plus kira-bam's sort /
    // markdup `@PG` records — for the final file of a sorted run. The spill
    // path writes the plain header and lets `kira-bam sort` stamp its own.
    let header_text = aligner.sam_header_bytes(&index)?;
    let plain_header = parse_header(&header_text)?;
    let sorted_header = if opts.kind.is_sorted() {
        let mut h = parse_header(&set_sort_order(&header_text, "coordinate"))?;
        append_pg(&mut h, &PgInfo::new("sort", !opts.no_pg)).context("append @PG sort")?;
        if opts.markdup {
            append_pg(&mut h, &PgInfo::new("markdup", !opts.no_pg)).context("append @PG markdup")?;
        }
        h
    } else {
        plain_header.clone()
    };

    // The parse/encode work below runs on rayon's global pool; the aligner
    // keeps its own pools. Ignoring the error keeps a pool an embedding
    // program already installed.
    kira_bam::io::install_thread_pool(opts.threads);

    let formatter = SamFormatter::new(Arc::new(index.reference.clone()));
    let output_cfg = cfg.pipeline.output;
    let max_alignments = cfg.pipeline.max_alignments;
    let read_group = cfg.read_group.clone();

    let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(2);
    let conv = Converter {
        plain_header,
        sorted_header,
        opts,
    };
    let handle = thread::Builder::new()
        .name("kira-emit".into())
        .spawn(move || conv.run(rx))
        .context("spawn emit converter thread")?;

    let mut buf: Vec<u8> = Vec::new();
    let drive = aligner.align_streaming(index, reads, |scored| {
        stage6_output::serialize_into(
            scored,
            &formatter,
            read_group.as_deref(),
            output_cfg,
            max_alignments,
            &mut buf,
        );
        let out = std::mem::take(&mut buf);
        tx.send(out)
            .map_err(|_| anyhow::anyhow!("emit converter thread stopped early"))
    });
    drop(tx);
    // Converter error first: its failure is what makes the driver's send fail.
    let conv_result = handle
        .join()
        .map_err(|_| anyhow::anyhow!("emit converter thread panicked"))?;
    let final_path = conv_result?;
    drive?;

    if final_path.bai {
        kira_bam::index::run(kira_bam::cli::IndexArgs {
            input: final_path.path.clone(),
            output: None,
            bai: true,
            csi: false,
            min_shift: 14,
            depth: 5,
            threads: final_path.threads,
        })
        .context("build BAI")?;
    }
    crate::kira_info!("[KIRA] {} pipeline complete → {}", final_path.kind_name, final_path.path.display());
    Ok(())
}

/// What the converter hands back once the file is finished.
struct Finished {
    path: PathBuf,
    bai: bool,
    threads: usize,
    kind_name: &'static str,
}

/// Where the records of a run are going.
enum Sink {
    /// Unsorted output: written as they arrive.
    Stream(BamWriter),
    /// Sorted output within budget: collected for the in-memory sort.
    Collect {
        records: Vec<RecordBuf>,
        est_bytes: usize,
        budget: usize,
    },
    /// Sorted output past budget: streaming into an unsorted temp BAM that
    /// `kira-bam sort` finishes.
    Spilled {
        writer: BamWriter,
        tmp: tempfile::TempPath,
    },
}

struct Converter {
    plain_header: sam::Header,
    sorted_header: sam::Header,
    opts: EmitOptions,
}

impl Converter {
    fn write_opts(&self) -> WriteOptions {
        WriteOptions {
            // BGZF compression saturates around 8 workers; one core stays
            // with the record producer.
            compression_workers: self.opts.threads.saturating_sub(1).clamp(1, 8),
            compression_level: None,
        }
    }

    fn open_final(&self, header: sam::Header) -> Result<BamWriter> {
        let reference = self.opts.kind.is_cram().then_some(self.opts.reference.as_path());
        BamWriter::create_with_options(
            Some(&self.opts.output),
            header,
            self.opts.kind.format(),
            reference,
            self.write_opts(),
        )
        .with_context(|| format!("create {}", self.opts.output.display()))
    }

    fn run(self, rx: mpsc::Receiver<Vec<u8>>) -> Result<Finished> {
        let kind = self.opts.kind;
        let mut sink = if kind.is_sorted() {
            let budget = kira_bam::io::resolve_memory_hint(
                &self.opts.sort_memory,
                AUTO_SORT_MIN_BYTES,
                1,
                4,
            )
            .context("--sort-memory")?;
            Sink::Collect {
                records: Vec::new(),
                est_bytes: 0,
                budget,
            }
        } else {
            let mut w = self.open_final(self.plain_header.clone())?;
            w.write_header()?;
            Sink::Stream(w)
        };

        while let Ok(bytes) = rx.recv() {
            if bytes.is_empty() {
                continue;
            }
            let recs = parse_records(&self.plain_header, &bytes)?;
            match &mut sink {
                Sink::Stream(w) => write_records(w, &recs, kind.format())?,
                Sink::Spilled { writer, .. } => write_records(writer, &recs, OutputFormat::Bam)?,
                Sink::Collect {
                    records,
                    est_bytes,
                    budget,
                } => {
                    *est_bytes += bytes.len() * RECORD_BYTES_PER_SAM_BYTE;
                    records.extend(recs);
                    if *est_bytes > *budget {
                        crate::kira_warn!(
                            "[KIRA] --emit {}: records exceed --sort-memory ({} MB); spilling to an unsorted BAM and finishing with kira-bam's external sort{}",
                            emit_name(kind),
                            *budget >> 20,
                            if self.opts.markdup { " + two-pass markdup" } else { "" }
                        );
                        let dir = self
                            .opts
                            .output
                            .parent()
                            .filter(|p| !p.as_os_str().is_empty())
                            .map(Path::to_path_buf)
                            .unwrap_or_else(|| PathBuf::from("."));
                        let tmp = tempfile::Builder::new()
                            .prefix("kira-ls-aligner-")
                            .suffix(".unsorted.bam")
                            .tempfile_in(&dir)
                            .with_context(|| format!("create spill BAM in {}", dir.display()))?
                            .into_temp_path();
                        let mut writer = BamWriter::create_with_options(
                            Some(&tmp),
                            self.plain_header.clone(),
                            OutputFormat::Bam,
                            None::<&Path>,
                            self.write_opts(),
                        )
                        .context("open spill BAM")?;
                        writer.write_header()?;
                        write_records(&mut writer, records, OutputFormat::Bam)?;
                        records.clear();
                        records.shrink_to_fit();
                        sink = Sink::Spilled { writer, tmp };
                    }
                }
            }
        }

        match sink {
            Sink::Stream(w) => w.finish().context("finish output")?,
            Sink::Collect { records, .. } => {
                let sorted = kira_bam::sort::sort_and_markdup_in_memory(records, self.opts.markdup)
                    .context("in-memory sort")?;
                let mut w = self.open_final(self.sorted_header.clone())?;
                w.write_header()?;
                write_records(&mut w, &sorted, kind.format())?;
                w.finish().context("finish sorted output")?;
            }
            Sink::Spilled { writer, tmp } => {
                writer.finish().context("finish spill BAM")?;
                // kira-bam's external sort cannot fuse markdup (it needs the
                // whole run in one chunk), so with `--markdup` the sort goes
                // to a second temp file and the two-pass, disk-backed markdup
                // writes the final output. One more BAM pass, but `--markdup`
                // means marked, never "marked if it happened to fit".
                let sorted_tmp = if self.opts.markdup {
                    let dir = tmp
                        .parent()
                        .map(Path::to_path_buf)
                        .unwrap_or_else(|| PathBuf::from("."));
                    Some(
                        tempfile::Builder::new()
                            .prefix("kira-ls-aligner-")
                            .suffix(".sorted.bam")
                            .tempfile_in(&dir)
                            .with_context(|| format!("create sorted temp BAM in {}", dir.display()))?
                            .into_temp_path(),
                    )
                } else {
                    None
                };
                let sort_output = sorted_tmp
                    .as_ref()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| self.opts.output.clone());
                kira_bam::sort::run(kira_bam::cli::SortArgs {
                    input: tmp.to_path_buf(),
                    output: Some(sort_output.clone()),
                    name_sort: false,
                    threads: self.opts.threads,
                    memory: self.opts.sort_memory.clone(),
                    tmpdir: None,
                    uncompressed: false,
                    compression_level: None,
                    reference: kind.is_cram().then(|| self.opts.reference.clone()),
                    require_flags: 0,
                    filter_flags: 0,
                    no_pg: self.opts.no_pg,
                    markdup: false,
                    markdup_barcode_tag: None,
                    markdup_mode_ancient: false,
                    markdup_supplementary: false,
                })
                .context("external sort of spilled records")?;
                drop(tmp);
                if let Some(sorted) = sorted_tmp {
                    kira_bam::markdup::run(kira_bam::cli::MarkdupArgs {
                        input: sorted.to_path_buf(),
                        output: self.opts.output.clone(),
                        remove: false,
                        optical_distance: 0,
                        threads: self.opts.threads,
                        stats: None,
                        uncompressed: false,
                        reference: kind.is_cram().then(|| self.opts.reference.clone()),
                        barcode_tag: None,
                        mode_ancient: false,
                        supplementary: false,
                        no_pg: self.opts.no_pg,
                    })
                    .context("markdup after external sort")?;
                    drop(sorted);
                }
            }
        }

        Ok(Finished {
            path: self.opts.output.clone(),
            bai: self.opts.bai && !kind.is_cram(),
            threads: self.opts.threads,
            kind_name: emit_name(kind),
        })
    }
}

fn emit_name(kind: EmitKind) -> &'static str {
    match kind {
        EmitKind::Bam => "bam",
        EmitKind::SortedBam => "sorted-bam",
        EmitKind::Cram => "cram",
        EmitKind::SortedCram => "sorted-cram",
    }
}

fn parse_header(text: &[u8]) -> Result<sam::Header> {
    let mut reader = sam::io::Reader::new(text);
    reader.read_header().context("parse SAM header")
}

/// Rewrite the `@HD` `SO:` field of a SAM header text.
fn set_sort_order(text: &[u8], so: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len() + 16);
    let mut first = true;
    for line in text.split_inclusive(|&b| b == b'\n') {
        if first && line.starts_with(b"@HD") {
            let s = String::from_utf8_lossy(line);
            let rewritten: Vec<String> = s
                .trim_end()
                .split('\t')
                .map(|f| {
                    if f.starts_with("SO:") {
                        format!("SO:{so}")
                    } else {
                        f.to_string()
                    }
                })
                .collect();
            out.extend_from_slice(rewritten.join("\t").as_bytes());
            out.push(b'\n');
        } else {
            out.extend_from_slice(line);
        }
        first = false;
    }
    out
}

/// Parse a batch of SAM record lines into records, in line-aligned chunks
/// across the rayon pool. The header is only needed for reference-name
/// resolution.
fn parse_records(header: &sam::Header, bytes: &[u8]) -> Result<Vec<RecordBuf>> {
    let n_chunks = (rayon::current_num_threads() * 2).max(1);
    let target = bytes.len().div_ceil(n_chunks).max(1);
    let mut bounds = Vec::with_capacity(n_chunks + 1);
    bounds.push(0usize);
    let mut cursor = 0usize;
    while cursor < bytes.len() {
        let mut end = (cursor + target).min(bytes.len());
        if end < bytes.len() {
            match bytes[end..].iter().position(|&b| b == b'\n') {
                Some(nl) => end += nl + 1,
                None => end = bytes.len(),
            }
        }
        bounds.push(end);
        cursor = end;
    }
    let chunks: Vec<Result<Vec<RecordBuf>>> = bounds
        .windows(2)
        .collect::<Vec<_>>()
        .par_iter()
        .map(|w| {
            let slice = &bytes[w[0]..w[1]];
            let mut reader = sam::io::Reader::new(slice);
            let mut out = Vec::with_capacity(slice.len() / 300 + 1);
            let mut rec = RecordBuf::default();
            loop {
                let n = reader
                    .read_record_buf(header, &mut rec)
                    .context("parse SAM record")?;
                if n == 0 {
                    break;
                }
                out.push(std::mem::take(&mut rec));
            }
            Ok(out)
        })
        .collect();
    let mut all = Vec::new();
    for c in chunks {
        all.extend(c?);
    }
    Ok(all)
}

/// Write records: pre-encoded in parallel and streamed through BGZF for BAM,
/// one at a time for CRAM (its encoder is stateful).
fn write_records(w: &mut BamWriter, recs: &[RecordBuf], fmt: OutputFormat) -> Result<()> {
    if recs.is_empty() {
        return Ok(());
    }
    if matches!(fmt, OutputFormat::Bam | OutputFormat::UncompressedBam) {
        let n_chunks = rayon::current_num_threads().max(1);
        let chunk_len = recs.len().div_ceil(n_chunks).max(1);
        let encoded: Vec<(Vec<u8>, u64)> = {
            let hdr = w.header();
            recs.par_chunks(chunk_len)
                .map(|c| encode_records_into(hdr, c))
                .collect()
        };
        let skipped: u64 = encoded.iter().map(|(_, n)| n).sum();
        for (buf, _) in &encoded {
            w.write_preencoded(buf)?;
        }
        if skipped > 0 {
            crate::kira_warn!("[KIRA] warning: {skipped} records rejected by the BAM encoder");
        }
    } else {
        let mut skipped = 0u64;
        for r in recs {
            kira_bam::io::write_record_tolerant(w, r, "kira-ls-aligner", &mut skipped)?;
        }
        kira_bam::io::report_skipped("kira-ls-aligner", skipped);
    }
    Ok(())
}

/// CRAM needs `REF.fai`; build it in place when missing, as `samtools faidx` would.
fn ensure_fai(reference: &Path) -> Result<()> {
    let mut fai = reference.as_os_str().to_os_string();
    fai.push(".fai");
    let fai = PathBuf::from(fai);
    if fai.is_file() {
        return Ok(());
    }
    crate::kira_info!("[KIRA] building {} for CRAM output", fai.display());
    kira_bam::faidx::run(kira_bam::cli::FaidxArgs {
        fasta: reference.to_path_buf(),
        regions: Vec::new(),
        output: None,
        line_width: 60,
    })
    .with_context(|| format!("build {}", fai.display()))
}
