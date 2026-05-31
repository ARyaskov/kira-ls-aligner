use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use sysinfo::System;

use std::sync::Arc;

use crate::alignment::junc_bed::JunctionIndex;
use crate::exec::DualPool;
use crate::exec::pool::DualPoolConfig;
use crate::index::{Index, IndexConfig};
use crate::io::{HeaderConfig, ReadStream, SamWriter, read_reference};
use crate::pipeline::mode::{ModeFeatures, classify};
use crate::pipeline::stage0_input;
use crate::pipeline::stage4_alignment::AlignmentBatchStats;
use crate::pipeline::{Pipeline, PipelineConfig, PipelineStageTimes};
use crate::simd::{SimdMode, detect_cached};

/// High-level aligner configuration.
#[derive(Clone, Debug)]
pub struct AlignerConfig {
    pub threads: usize,
    /// Optional override: number of workers pinned to Performance cores.
    pub num_p_threads: Option<usize>,
    /// Optional override: number of workers pinned to Efficient cores.
    pub num_e_threads: Option<usize>,
    pub batch_bases: usize,
    pub index: IndexConfig,
    pub pipeline: PipelineConfig,
    pub auto_profiles: Option<crate::pipeline::mode::ReadModeProfiles>,
    pub read_group: Option<String>,
    /// SAM header customisation: @PG / @CO injections, @RG.
    pub header: Option<HeaderConfig>,
    /// Parsed splice junction annotation (`--junc-bed`).
    pub junctions: Option<Arc<JunctionIndex>>,
    /// Position tolerance for junction lookups (`--junc-bed-tolerance`).
    pub junc_bed_tolerance: u32,
}

/// Aligner orchestrator.
pub struct Aligner {
    cfg: AlignerConfig,
}

fn adjust_auto_params(
    cfg: &mut PipelineConfig,
    align: &AlignmentBatchStats,
    sketch: &crate::pipeline::stage1_sketch::SketchBatchStats,
) {
    let reads = align.reads.max(1) as f32;
    let accept_rate = align.prefilter_accept as f32 / reads;
    let chains_per_read = align.chains_used as f32 / reads;

    // Short-read adaptive tuning only.
    if cfg.short_preset {
        let span_slack = (sketch.read_len_p50 / 10).clamp(12, 15);
        cfg.accept_span_slack = span_slack;
        cfg.accept_min_identity = 98.5;
        cfg.accept_max_mismatches = if sketch.read_len_p50 <= 200 { 5 } else { 6 };

        if chains_per_read > 2.5 {
            cfg.dp_topk = 2;
            cfg.accept_require_score_margin = 20;
        } else {
            cfg.dp_topk = 1;
            cfg.accept_require_score_margin = 0;
        }

        if accept_rate < 0.05
            && align.ungapped_identity_p90 >= 99.0
            && align.ungapped_mismatches_p95 <= 2
        {
            cfg.accept_max_mismatches = (cfg.accept_max_mismatches + 1).min(4);
        }
        if accept_rate > 0.6 && align.ungapped_identity_p90 < 98.5 {
            cfg.accept_min_identity = (cfg.accept_min_identity + 0.5).min(99.5);
        }
    }
}

fn config_fingerprint(cfg: &PipelineConfig, threads: usize, batch_bases: usize) -> String {
    format!(
        "threads={} batch_bases={} match={} mismatch={} gap_open={} gap_extend={} bandwidth={} xdrop={} dp_topk={} dp_abort_margin={} accept_enable={} accept_only_top1={} accept_span_slack={} accept_min_id={:.2} accept_max_mism={} accept_score_margin={} max_alignments={} min_chain_ratio={:.2} short_preset={} write_nm={} write_md={} write_as={} write_xs={} write_xa={} write_sa={}",
        threads,
        batch_bases,
        cfg.alignment.match_score,
        cfg.alignment.mismatch,
        cfg.alignment.gap_open,
        cfg.alignment.gap_extend,
        cfg.alignment.bandwidth,
        cfg.alignment.xdrop,
        cfg.dp_topk,
        cfg.dp_abort_margin,
        cfg.accept_enable,
        cfg.accept_only_top1,
        cfg.accept_span_slack,
        cfg.accept_min_identity,
        cfg.accept_max_mismatches,
        cfg.accept_require_score_margin,
        cfg.max_alignments,
        cfg.min_chain_ratio,
        cfg.short_preset,
        cfg.output.write_nm,
        cfg.output.write_md,
        cfg.output.write_as,
        cfg.output.write_xs,
        cfg.output.write_xa,
        cfg.output.write_sa,
    )
}

impl Aligner {
    pub fn new(cfg: AlignerConfig) -> Self {
        Self { cfg }
    }

    pub fn run<P, R>(
        &self,
        reference_path: P,
        reads_paths: &[std::path::PathBuf],
        output_path: Option<R>,
    ) -> Result<()>
    where
        P: AsRef<Path>,
        R: AsRef<Path>,
    {
        let reference = read_reference(&reference_path).context("load reference")?;
        let index = Index::build(reference, self.cfg.index);
        self.run_with_index(index, reads_paths, output_path)
    }

    pub fn run_with_index_file<P, R>(
        &self,
        index_path: P,
        reads_paths: &[std::path::PathBuf],
        output_path: Option<R>,
    ) -> Result<()>
    where
        P: AsRef<Path>,
        R: AsRef<Path>,
    {
        let index = Index::load(index_path).context("load index")?;
        self.run_with_index(index, reads_paths, output_path)
    }

    fn run_with_index<R>(
        &self,
        mut index: Index,
        reads_paths: &[std::path::PathBuf],
        output_path: Option<R>,
    ) -> Result<()>
    where
        R: AsRef<Path>,
    {
        let hot_n: usize = std::env::var("KIRA_HOT_CACHE_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(50_000);
        if hot_n > 0 {
            let max_occs: usize = std::env::var("KIRA_HOT_CACHE_MAX_OCCS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8_000_000);
            let mmap_bytes = index.mmap.as_deref().map(|m| &m[..]);
            index.short.build_hot_cache(mmap_bytes, hot_n, max_occs);
            index.long.build_hot_cache(mmap_bytes, hot_n / 4, max_occs / 4);
        }

        let stats_enabled = std::env::var_os("KIRA_STATS").is_some();
        let mut stage0_total = Duration::ZERO;
        let mut pipeline_totals = [Duration::ZERO; 6];
        let mut total_batches: u64 = 0;
        let mut align_total = AlignmentBatchStats::default();
        let mut seed_total_before: usize = 0;
        let mut seed_total_after: usize = 0;
        let mut chain_total_used: usize = 0;
        let mut chain_total_pruned: usize = 0;
        let overall_start = Instant::now();
        let mut mode_selected: Option<(crate::pipeline::mode::ReadMode, usize)> = None;

        let output_path_buf = output_path.as_ref().map(|p| p.as_ref().to_path_buf());
        let simd_mode = if stats_enabled {
            Some(detect_cached())
        } else {
            None
        };

        let mut writer = SamWriter::new(output_path, index.reference.clone())?;
        let is_paf =
            matches!(self.cfg.pipeline.output.format, crate::io::EmitFormat::Paf);
        if !is_paf {
            match &self.cfg.header {
                Some(hdr) => writer.write_header_with_ctx(hdr)?,
                None => writer.write_header_with_rg(self.cfg.read_group.as_deref())?,
            }
        }

        let formatter = writer.formatter_handle();
        let (write_tx, write_rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(4);
        let writer_thread = std::thread::Builder::new()
            .name("kira-sam-writer".to_string())
            .spawn(move || -> Result<()> {
                let mut writer = writer;
                while let Ok(buf) = write_rx.recv() {
                    writer.write_batch(&buf)?;
                }
                writer.flush()?;
                Ok(())
            })
            .context("spawn writer thread")?;

        let pool = Arc::new(
            DualPool::new(DualPoolConfig {
                p_threads: self.cfg.num_p_threads,
                e_threads: self.cfg.num_e_threads,
                total_threads: Some(self.cfg.threads),
            })
            .context("build hybrid-aware thread pools")?,
        );
        // Always surface the pool layout — even outside KIRA_STATS — so
        // users notice when the auto-detected split disagrees with what
        // they expected (e.g. a misreported hybrid CPU falling back to a
        // single pool).
        eprintln!(
            "[KIRA_POOL] hybrid={} p_threads={} e_threads={} total={}",
            pool.is_hybrid(),
            pool.p_threads(),
            pool.e_threads(),
            pool.total_threads(),
        );

        let mut stream = ReadStream::new_multi_with_mode(
            reads_paths,
            self.cfg.batch_bases,
            self.cfg.pipeline.paired.mode,
        )?;
        let mut pipeline = Pipeline::with_pool(self.cfg.pipeline, Arc::clone(&pool))
            .with_junctions(self.cfg.junctions.clone(), self.cfg.junc_bed_tolerance);
        if stats_enabled {
            eprintln!(
                "[KIRA_CONFIG] {}",
                config_fingerprint(&pipeline.config, self.cfg.threads, self.cfg.batch_bases)
            );
        }
        let mut auto_profiles = self.cfg.auto_profiles.clone();

        let total_bytes = stream.total_bytes();
        let mut sys = if stats_enabled {
            Some(System::new())
        } else {
            None
        };
        let progress = if stats_enabled && total_bytes > 0 {
            Some(init_progress_bar(total_bytes))
        } else {
            None
        };

        // Run the batch loop on the caller's own thread. Each stage
        // installs into the appropriate pool itself, so wrapping the
        // loop in install_driver would only burn a P-pool worker as a
        // permanent driver (it would have to block on every E-pool
        // install) for no parallelism benefit.
        let drive_result: Result<()> = (|| -> Result<()> {
            let mut batch_idx: u64 = 0;
            loop {
                let fetch_start = Instant::now();
                let reads_opt = stream.next_batch()?;
                let fetch_time = fetch_start.elapsed();
                let reads = match reads_opt {
                    Some(r) => r,
                    None => break,
                };

                let stage0_start = Instant::now();
                let input = stage0_input::run(reads);
                let stage0_time = fetch_time + stage0_start.elapsed();

                let batch_out = pipeline.process_batch_serialized(
                    input,
                    &index,
                    &formatter,
                    self.cfg.read_group.as_deref(),
                )?;
                let batch_stats = batch_out.stats;
                if write_tx.send(batch_out.sam_buf).is_err() {
                    // Writer thread died — surface the error on next join.
                    break;
                }

                if let Some(profiles) = auto_profiles.as_mut() {
                    if profiles.decided.is_none() {
                        let features = ModeFeatures {
                            read_len_p50: batch_stats.sketch.read_len_p50,
                            read_len_p90: batch_stats.sketch.read_len_p90,
                            avg_minimizers: batch_stats.sketch.avg_minimizers,
                            ungapped_len_p95: batch_stats.align.ungapped_span_p95,
                            ungapped_mism_p95: batch_stats.align.ungapped_mismatches_p95,
                            ungapped_id_p90: batch_stats.align.ungapped_identity_p90,
                            chains_per_read: if batch_stats.align.reads == 0 { 0.0 } else { batch_stats.align.chains_used as f32 / batch_stats.align.reads as f32 },
                        };
                        let mode = classify(features);
                        profiles.decided = Some(mode);
                        mode_selected = Some((mode, features.read_len_p50));
                        let new_cfg = profiles.select(mode);
                        pipeline.config = new_cfg;
                        if stats_enabled {
                            eprintln!("[KIRA_MODE] selected={:?} p50={} p90={} id_p90={:.2} chains_per_read={:.2}", mode, features.read_len_p50, features.read_len_p90, features.ungapped_id_p90, features.chains_per_read);
                        }
                    }

                    if profiles.decided.is_some() {
                        adjust_auto_params(&mut pipeline.config, &batch_stats.align, &batch_stats.sketch);
                    }
                }

                if stats_enabled {
                    stage0_total += stage0_time;
                    for (dst, src) in pipeline_totals.iter_mut().zip(batch_stats.times.stages.iter()) {
                        *dst += *src;
                    }
                    total_batches += 1;
                    align_total.add(&batch_stats.align);
                    seed_total_before += batch_stats.seed.anchors_before_prune;
                    seed_total_after += batch_stats.seed.anchors_after_prune;
                    chain_total_used += batch_stats.chaining.anchors_used_for_chaining;
                    chain_total_pruned += batch_stats.chaining.chains_pruned_early;
                    print_batch_stats(batch_idx, stage0_time, &batch_stats.times, &batch_stats.align, progress.as_ref());
                    print_algo_counters(&format!("batch {}", batch_idx), &batch_stats.align);
                    eprintln!(
                        "[KIRA_SEED_STATS] batch {}: anchors_before_prune={} anchors_after_prune={} chaining_used={} chaining_pruned={}",
                        batch_idx,
                        batch_stats.seed.anchors_before_prune,
                        batch_stats.seed.anchors_after_prune,
                        batch_stats.chaining.anchors_used_for_chaining,
                        batch_stats.chaining.chains_pruned_early,
                    );
                    let dp_rate = if batch_stats.align.dp_attempts == 0 { 0.0 } else { batch_stats.align.dp_early_abort as f32 * 100.0 / batch_stats.align.dp_attempts as f32 };
                    eprintln!(
                        "[KIRA_ALIGN_STATS] batch {}: accept_rate={:.2}% fallback_rate={:.2}% dp_early_abort_rate={:.2}%",
                        batch_idx,
                        (batch_stats.align.prefilter_accept as f32 * 100.0 / batch_stats.align.reads.max(1) as f32),
                        (batch_stats.align.prefilter_fallback as f32 * 100.0 / batch_stats.align.reads.max(1) as f32),
                        dp_rate
                    );
                    update_progress(
                        progress.as_ref(),
                        &mut sys,
                        &stream,
                        self.cfg.threads,
                        output_path_buf.as_deref(),
                        simd_mode,
                    );
                }

                batch_idx += 1;
            }
            Ok(())
        })();
        drive_result?;

        drop(write_tx);
        match writer_thread.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(e.context("SAM writer thread failed")),
            Err(panic) => {
                return Err(anyhow::anyhow!(
                    "SAM writer thread panicked: {:?}",
                    panic
                ));
            }
        }

        if stats_enabled {
            if let Some(pb) = progress.as_ref() {
                pb.finish_with_message("done");
            }
            let overall = overall_start.elapsed();
            print_summary_stats(
                stage0_total,
                &pipeline_totals,
                total_batches,
                overall,
                &align_total,
                progress.as_ref(),
                mode_selected,
            );
            print_algo_counters("summary", &align_total);
            print_cascade_summary(&align_total);
            eprintln!(
                "[KIRA_SEED_STATS] summary: anchors_before_prune={} anchors_after_prune={} chaining_used={} chaining_pruned={}",
                seed_total_before, seed_total_after, chain_total_used, chain_total_pruned,
            );
            let dp_rate = if align_total.dp_attempts == 0 {
                0.0
            } else {
                align_total.dp_early_abort as f32 * 100.0 / align_total.dp_attempts as f32
            };
            eprintln!(
                "[KIRA_ALIGN_STATS] summary: accept_rate={:.2}% fallback_rate={:.2}% dp_early_abort_rate={:.2}%",
                (align_total.prefilter_accept as f32 * 100.0 / align_total.reads.max(1) as f32),
                (align_total.prefilter_fallback as f32 * 100.0 / align_total.reads.max(1) as f32),
                dp_rate
            );
        }

        Ok(())
    }

    /// Run the full alignment pipeline in-process and return the complete SAM
    /// (header + records) as a byte buffer — nothing is written to disk. For
    /// fused in-memory pipelines (e.g. `kira-bt solid`) that hand the SAM
    /// straight to sort/markdup/pileup. Reuses the same per-batch path as the
    /// file writer (`process_batch_serialized`); skips progress/stats and the
    /// adaptive per-batch mode re-selection of `run_with_index`.
    pub fn align_to_sam_bytes(
        &self,
        mut index: Index,
        reads_paths: &[std::path::PathBuf],
    ) -> Result<Vec<u8>> {
        let hot_n: usize = std::env::var("KIRA_HOT_CACHE_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(50_000);
        if hot_n > 0 {
            let max_occs: usize = std::env::var("KIRA_HOT_CACHE_MAX_OCCS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8_000_000);
            let mmap_bytes = index.mmap.as_deref().map(|m| &m[..]);
            index.short.build_hot_cache(mmap_bytes, hot_n, max_occs);
            index.long.build_hot_cache(mmap_bytes, hot_n / 4, max_occs / 4);
        }

        let pool = Arc::new(
            DualPool::new(DualPoolConfig {
                p_threads: self.cfg.num_p_threads,
                e_threads: self.cfg.num_e_threads,
                total_threads: Some(self.cfg.threads),
            })
            .context("build hybrid-aware thread pools")?,
        );

        let out = Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
        let mut writer =
            SamWriter::from_writer(Box::new(crate::io::VecSink(out.clone())), index.reference.clone());
        match &self.cfg.header {
            Some(hdr) => writer.write_header_with_ctx(hdr)?,
            None => writer.write_header_with_rg(self.cfg.read_group.as_deref())?,
        }
        let formatter = writer.formatter_handle();

        let mut stream = ReadStream::new_multi_with_mode(
            reads_paths,
            self.cfg.batch_bases,
            self.cfg.pipeline.paired.mode,
        )?;
        let pipeline = Pipeline::with_pool(self.cfg.pipeline, Arc::clone(&pool))
            .with_junctions(self.cfg.junctions.clone(), self.cfg.junc_bed_tolerance);

        while let Some(reads) = stream.next_batch()? {
            let input = stage0_input::run(reads);
            let bo = pipeline.process_batch_serialized(
                input,
                &index,
                &formatter,
                self.cfg.read_group.as_deref(),
            )?;
            writer.write_batch(&bo.sam_buf)?;
        }
        writer.flush()?;
        drop(writer);

        let bytes = Arc::try_unwrap(out)
            .map_err(|_| anyhow::anyhow!("SAM buffer still shared"))?
            .into_inner()
            .map_err(|_| anyhow::anyhow!("SAM buffer mutex poisoned"))?;
        Ok(bytes)
    }
}

fn init_progress_bar(total: u64) -> ProgressBar {
    let pb = ProgressBar::new(total);
    let style = ProgressStyle::with_template(
        "[{elapsed_precise}] {bar:40.cyan/blue} {percent:>3}% {bytes}/{total_bytes} ETA {eta_precise} {msg}",
    )
    .unwrap()
    .progress_chars("#>- ");
    pb.set_style(style);
    pb.enable_steady_tick(Duration::from_millis(200));
    pb
}

fn update_progress(
    pb: Option<&ProgressBar>,
    sys: &mut Option<System>,
    stream: &ReadStream,
    threads: usize,
    output_path: Option<&Path>,
    simd_mode: Option<SimdMode>,
) {
    let Some(pb) = pb else {
        return;
    };
    let read_bytes = stream.bytes_read();
    pb.set_position(read_bytes);

    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let mem = if let Some(sys) = sys.as_mut() {
        sys.refresh_memory();
        let total = sys.total_memory();
        let used = sys.used_memory();
        Some((used, total))
    } else {
        None
    };

    let out_size = output_path.and_then(|p| std::fs::metadata(p).ok().map(|m| m.len()));
    let mut msg = format!("threads={threads}/{cores}");
    if let Some((used, total)) = mem {
        msg.push_str(&format!(" RAM={} / {}", fmt_bytes(used), fmt_bytes(total)));
    }
    if let Some(size) = out_size {
        msg.push_str(&format!(" out={}", fmt_bytes(size)));
    }
    if let Some(mode) = simd_mode {
        let simd = match mode {
            SimdMode::Avx2 => "simd=avx2",
            SimdMode::AvxVnni => "simd=avx-vnni",
            SimdMode::Neon => "simd=neon",
            SimdMode::Scalar => "simd=scalar",
        };
        msg.push_str(&format!(" {simd}"));
    }
    msg.push_str(&format!(" cuda={}", cuda_status()));
    pb.set_message(msg);
}

fn cuda_status() -> &'static str {
    #[cfg(feature = "cuda")]
    {
        "on"
    }
    #[cfg(not(feature = "cuda"))]
    {
        "off"
    }
}

fn fmt_bytes(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    let b = bytes as f64;
    if b >= GB {
        format!("{:.2} GB", b / GB)
    } else if b >= MB {
        format!("{:.2} MB", b / MB)
    } else if b >= KB {
        format!("{:.2} KB", b / KB)
    } else {
        format!("{bytes} B")
    }
}

fn fmt_ms(d: Duration) -> String {
    format!("{:.3} ms", d.as_secs_f64() * 1000.0)
}

/// Emit the compact cascade-bucket summary used to track fast-path coverage.
///
/// Four buckets only — `spectral` (every non-DP fast path: exact, prefilter,
/// packed/byte spectral, GPU spectral, LSH and CGK rescue), `wfa`, `sw`
/// (banded SIMD + scalar) and `unmapped`. The line is machine-parseable so
/// HG002/HG38 tuning scripts can grep for it.
fn print_cascade_summary(stats: &AlignmentBatchStats) {
    let spectral = stats.exact_matches
        + stats.prefilter_accept
        + stats.packed_spectral_resolved
        + stats.spectral_sieve_resolved
        + stats.gpu_spectral_resolved
        + stats.lsh_rescue_resolved
        + stats.cgk_rescue_resolved;
    let wfa = stats.wfa_resolved;
    let sw = stats.dp_simd + stats.dp_scalar;
    let unmapped = stats.unmapped;
    let total = spectral + wfa + sw + unmapped;
    let denom = total.max(1) as f64;
    eprintln!(
        "[CASCADE] spectral={} ({:.2}%) wfa={} ({:.2}%) sw={} ({:.2}%) unmapped={} ({:.2}%) (total={})",
        spectral, spectral as f64 * 100.0 / denom,
        wfa,      wfa      as f64 * 100.0 / denom,
        sw,       sw       as f64 * 100.0 / denom,
        unmapped, unmapped as f64 * 100.0 / denom,
        total,
    );
}

/// Emit a one-line per-algorithm cascade-attribution summary.
fn print_algo_counters(label: &str, stats: &AlignmentBatchStats) {
    let total = stats.reads.max(1) as f32;
    let pct = |n: usize| n as f32 * 100.0 / total;
    let spectral_total = stats.packed_spectral_resolved
        + stats.spectral_sieve_resolved
        + stats.gpu_spectral_resolved
        + stats.prefilter_accept;
    let sw_total = stats.dp_simd + stats.dp_scalar;
    eprintln!(
        "[KIRA_ALGO] {}: reads={} | exact={} ({:.2}%) prefilter={} ({:.2}%) packed_spectral={} ({:.2}%) spectral_sieve={} ({:.2}%) gpu_spectral={} ({:.2}%) wfa={} ({:.2}%) lsh_rescue={} ({:.2}%) cgk_rescue={} ({:.2}%) sw_simd={} ({:.2}%) sw_scalar={} ({:.2}%) unmapped={} ({:.2}%) | spectral_total={} ({:.2}%) sw_total={} ({:.2}%) wfa_total={} ({:.2}%)",
        label,
        stats.reads,
        stats.exact_matches, pct(stats.exact_matches),
        stats.prefilter_accept, pct(stats.prefilter_accept),
        stats.packed_spectral_resolved, pct(stats.packed_spectral_resolved),
        stats.spectral_sieve_resolved, pct(stats.spectral_sieve_resolved),
        stats.gpu_spectral_resolved, pct(stats.gpu_spectral_resolved),
        stats.wfa_resolved, pct(stats.wfa_resolved),
        stats.lsh_rescue_resolved, pct(stats.lsh_rescue_resolved),
        stats.cgk_rescue_resolved, pct(stats.cgk_rescue_resolved),
        stats.dp_simd, pct(stats.dp_simd),
        stats.dp_scalar, pct(stats.dp_scalar),
        stats.unmapped, pct(stats.unmapped),
        spectral_total, pct(spectral_total),
        sw_total, pct(sw_total),
        stats.wfa_resolved, pct(stats.wfa_resolved),
    );
}

fn print_batch_stats(
    batch_idx: u64,
    stage0: Duration,
    stage_times: &PipelineStageTimes,
    align_stats: &AlignmentBatchStats,
    pb: Option<&ProgressBar>,
) {
    let line = format!(
        "[KIRA_STATS] batch {}: input={} sketch={} seeding={} chaining={} alignment={} scoring={} output={} total={} | reads={} chains={}/{} dp_invocations={} dp_reads={} dp_topk={} dp_early_abort={} (simd={} scalar={}) exact={} prefilter=ACCEPT/REJECT/FALLBACK={}/{}/{} prefilter_reasons=accept/disabled/not_short/not_top1/max_alignments/span/mism/id/indel/other={}/{}/{}/{}/{}/{}/{}/{}/{}/{} ungapped_score_p95={} ungapped_len_p95={} ungapped_mism_p95={} ungapped_id_p90={:.2} accept_len_p50/p95={}/{} fallback_len_p50/p95={}/{} accept_rate={:.2}% fallback_rate={:.2}% buckets={}/{}/{} avg_len={:.1}",
        batch_idx,
        fmt_ms(stage0),
        fmt_ms(stage_times.stages[0]),
        fmt_ms(stage_times.stages[1]),
        fmt_ms(stage_times.stages[2]),
        fmt_ms(stage_times.stages[3]),
        fmt_ms(stage_times.stages[4]),
        fmt_ms(stage_times.stages[5]),
        fmt_ms(stage0 + stage_times.total()),
        align_stats.reads,
        align_stats.chains_used,
        align_stats.chains_total,
        align_stats.dp_attempts,
        align_stats.dp_reads,
        align_stats.dp_topk,
        align_stats.dp_early_abort,
        align_stats.dp_simd,
        align_stats.dp_scalar,
        align_stats.exact_matches,
        align_stats.prefilter_accept,
        align_stats.prefilter_reject,
        align_stats.prefilter_fallback,
        align_stats.prefilter_reason_counts[0],
        align_stats.prefilter_reason_counts[1],
        align_stats.prefilter_reason_counts[2],
        align_stats.prefilter_reason_counts[3],
        align_stats.prefilter_reason_counts[4],
        align_stats.prefilter_reason_counts[5],
        align_stats.prefilter_reason_counts[6],
        align_stats.prefilter_reason_counts[7],
        align_stats.prefilter_reason_counts[8],
        align_stats.prefilter_reason_counts[9],
        align_stats.ungapped_score_p95,
        align_stats.ungapped_span_p95,
        align_stats.ungapped_mismatches_p95,
        align_stats.ungapped_identity_p90,
        align_stats.accept_len_p50,
        align_stats.accept_len_p95,
        align_stats.fallback_len_p50,
        align_stats.fallback_len_p95,
        (align_stats.prefilter_accept as f32 * 100.0 / align_stats.reads.max(1) as f32),
        (align_stats.prefilter_fallback as f32 * 100.0 / align_stats.reads.max(1) as f32),
        align_stats.bucket_counts[0],
        align_stats.bucket_counts[1],
        align_stats.bucket_counts[2],
        align_stats.avg_read_len(),
    );
    if let Some(pb) = pb {
        pb.println(&line);
    }
    eprintln!("{line}");
}

fn print_summary_stats(
    stage0: Duration,
    totals: &[Duration; 6],
    batches: u64,
    overall: Duration,
    align_total: &AlignmentBatchStats,
    pb: Option<&ProgressBar>,
    mode_selected: Option<(crate::pipeline::mode::ReadMode, usize)>,
) {
    let line = format!(
        "[KIRA_STATS] summary: batches={} input={} sketch={} seeding={} chaining={} alignment={} scoring={} output={} total={} | reads={} chains={}/{} dp_invocations={} dp_reads={} dp_topk={} dp_early_abort={} (simd={} scalar={}) exact={} prefilter=ACCEPT/REJECT/FALLBACK={}/{}/{} prefilter_reasons=accept/disabled/not_short/not_top1/max_alignments/span/mism/id/indel/other={}/{}/{}/{}/{}/{}/{}/{}/{}/{} ungapped_score_p95={} ungapped_len_p95={} ungapped_mism_p95={} ungapped_id_p90={:.2} accept_len_p50/p95={}/{} fallback_len_p50/p95={}/{} accept_rate={:.2}% fallback_rate={:.2}% buckets={}/{}/{} avg_len={:.1}",
        batches,
        fmt_ms(stage0),
        fmt_ms(totals[0]),
        fmt_ms(totals[1]),
        fmt_ms(totals[2]),
        fmt_ms(totals[3]),
        fmt_ms(totals[4]),
        fmt_ms(totals[5]),
        fmt_ms(overall),
        align_total.reads,
        align_total.chains_used,
        align_total.chains_total,
        align_total.dp_attempts,
        align_total.dp_reads,
        align_total.dp_topk,
        align_total.dp_early_abort,
        align_total.dp_simd,
        align_total.dp_scalar,
        align_total.exact_matches,
        align_total.prefilter_accept,
        align_total.prefilter_reject,
        align_total.prefilter_fallback,
        align_total.prefilter_reason_counts[0],
        align_total.prefilter_reason_counts[1],
        align_total.prefilter_reason_counts[2],
        align_total.prefilter_reason_counts[3],
        align_total.prefilter_reason_counts[4],
        align_total.prefilter_reason_counts[5],
        align_total.prefilter_reason_counts[6],
        align_total.prefilter_reason_counts[7],
        align_total.prefilter_reason_counts[8],
        align_total.prefilter_reason_counts[9],
        align_total.ungapped_score_p95,
        align_total.ungapped_span_p95,
        align_total.ungapped_mismatches_p95,
        align_total.ungapped_identity_p90,
        align_total.accept_len_p50,
        align_total.accept_len_p95,
        align_total.fallback_len_p50,
        align_total.fallback_len_p95,
        (align_total.prefilter_accept as f32 * 100.0 / align_total.reads.max(1) as f32),
        (align_total.prefilter_fallback as f32 * 100.0 / align_total.reads.max(1) as f32),
        align_total.bucket_counts[0],
        align_total.bucket_counts[1],
        align_total.bucket_counts[2],
        align_total.avg_read_len(),
    );
    if let Some((mode, p50)) = mode_selected {
        eprintln!(
            "[KIRA_MODE] summary: selected={:?} median_read_len={}",
            mode, p50
        );
    }
    let _ = pb;
    eprintln!("{line}");
}
