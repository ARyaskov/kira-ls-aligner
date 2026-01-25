use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::ThreadPoolBuilder;
use sysinfo::System;

use crate::index::{Index, IndexConfig};
use crate::io::{ReadStream, SamWriter, read_reference};
use crate::pipeline::mode::{ModeFeatures, classify};
use crate::pipeline::stage0_input;
use crate::pipeline::stage4_alignment::AlignmentBatchStats;
use crate::pipeline::{Pipeline, PipelineConfig, PipelineStageTimes};
use crate::simd::{SimdMode, detect};

/// High-level aligner configuration.
#[derive(Clone, Debug)]
pub struct AlignerConfig {
    pub threads: usize,
    pub batch_bases: usize,
    pub index: IndexConfig,
    pub pipeline: PipelineConfig,
    pub auto_profiles: Option<crate::pipeline::mode::ReadModeProfiles>,
    pub read_group: Option<String>,
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
        index: Index,
        reads_paths: &[std::path::PathBuf],
        output_path: Option<R>,
    ) -> Result<()>
    where
        R: AsRef<Path>,
    {
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
        let simd_mode = if stats_enabled { Some(detect()) } else { None };

        let mut writer = SamWriter::new(output_path, index.reference.clone())?;
        writer.write_header_with_rg(self.cfg.read_group.as_deref())?;

        let mut stream = ReadStream::new_multi(reads_paths, self.cfg.batch_bases)?;
        let mut pipeline = Pipeline::new(self.cfg.pipeline);
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

        let pool = ThreadPoolBuilder::new()
            .num_threads(self.cfg.threads)
            .build()
            .context("build thread pool")?;

        pool.install(|| -> Result<()> {
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

                let batch_stats = pipeline.process_batch(
                    input,
                    &index,
                    &mut writer,
                    self.cfg.read_group.as_deref(),
                )?;

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
            writer.flush()?;
            Ok(())
        })?;

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
    let mut msg = format!("threads={}/{}", threads, cores);
    if let Some((used, total)) = mem {
        msg.push_str(&format!(" RAM={} / {}", fmt_bytes(used), fmt_bytes(total)));
    }
    if let Some(size) = out_size {
        msg.push_str(&format!(" out={}", fmt_bytes(size)));
    }
    if let Some(mode) = simd_mode {
        let simd = match mode {
            SimdMode::Avx2 => "simd=avx2",
            SimdMode::Neon => "simd=neon",
            SimdMode::Scalar => "simd=scalar",
        };
        msg.push_str(&format!(" {}", simd));
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
        format!("{} B", bytes)
    }
}

fn fmt_ms(d: Duration) -> String {
    format!("{:.3} ms", d.as_secs_f64() * 1000.0)
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
        pb.println(line);
    } else {
        eprintln!("{}", line);
    }
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
        if let Some(pb) = pb {
            pb.println(format!(
                "[KIRA_MODE] summary: selected={:?} median_read_len={}",
                mode, p50
            ));
        } else {
            eprintln!(
                "[KIRA_MODE] summary: selected={:?} median_read_len={}",
                mode, p50
            );
        }
    }
    if let Some(pb) = pb {
        pb.println(line);
    } else {
        eprintln!("{}", line);
    }
}
