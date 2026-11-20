//! Split-prefix tiled aligner.

mod repeat_counts;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};

use crate::exec::DualPool;
use crate::exec::pool::DualPoolConfig;
use crate::index::tiling::TilePlan;
use crate::index::{Index, IndexConfig};
use crate::io::staged_sam::SamOutput;
use crate::io::{HeaderConfig, ReadStream};
use crate::pipeline::chunk_io::{ChunkWriter, MergeIter};
use std::sync::Arc;

use crate::alignment::junc_bed::JunctionIndex;
use crate::mapq::PairMapqContext;
use crate::pipeline::PostAlignPolicy;
use crate::pipeline::insert_estimate::InsertEstimator;
use crate::pipeline::stage0_input;
use crate::pipeline::stage4_alignment::AlignBatch;
use crate::pipeline::stage4_alignment::AlignmentBatchStats;
use crate::pipeline::stage5_scoring;
use crate::pipeline::stage6_output::serialize_chunks;
use crate::pipeline::{Pipeline, PipelineConfig};
use crate::types::{Alignment, MateInfo, Reference};
use std::sync::RwLock;

/// All knobs needed to run the tiled aligner end-to-end.
pub struct TiledRunConfig {
    pub threads: usize,
    pub num_p_threads: Option<usize>,
    pub num_e_threads: Option<usize>,
    pub batch_bases: usize,
    pub index_cfg: IndexConfig,
    pub pipeline_cfg: PipelineConfig,
    pub read_group: Option<String>,
    pub header: Option<HeaderConfig>,
    /// Temp file prefix.
    pub split_prefix: PathBuf,
    /// Optional junction annotation (`--junc-bed`).
    pub junctions: Option<Arc<JunctionIndex>>,
    pub junc_bed_tolerance: u32,
    /// Keep the FASTQ comment on every read (`-C`).
    pub keep_comment: bool,
    /// ALT-contig names (`REF.alt`); resolved against the full reference
    /// into the pipeline ALT mask before any stage uses it.
    pub alt_contigs: Option<Arc<std::collections::HashSet<String>>>,
}

/// Run the full tiled pipeline.
pub fn run_tiled(
    reference: Reference,
    reads_paths: &[PathBuf],
    output_path: Option<PathBuf>,
    cfg: TiledRunConfig,
    tile_plan: TilePlan,
) -> Result<()> {
    let pool = Arc::new(
        DualPool::new(DualPoolConfig {
            p_threads: cfg.num_p_threads,
            e_threads: cfg.num_e_threads,
            total_threads: Some(cfg.threads),
        })
        .context("build hybrid-aware thread pools for tiled run")?,
    );
    crate::kira_info!(
        "[KIRA_POOL] hybrid={} p_threads={} e_threads={} total={} (tiled)",
        pool.is_hybrid(),
        pool.p_threads(),
        pool.e_threads(),
        pool.total_threads(),
    );
    run_tiled_inner(reference, reads_paths, output_path, cfg, tile_plan, pool)
}

fn run_tiled_inner(
    reference: Reference,
    reads_paths: &[PathBuf],
    output_path: Option<PathBuf>,
    cfg: TiledRunConfig,
    tile_plan: TilePlan,
    pool: Arc<DualPool>,
) -> Result<()> {
    anyhow::ensure!(
        reads_paths
            .iter()
            .all(|path| !crate::io::is_stdin_path(path)),
        "--split-prefix requires reads that can be reopened; use a file instead of stdin"
    );
    let mut cfg = cfg;
    if tile_plan.tiles.is_empty() {
        return Err(anyhow::anyhow!("tile plan is empty (no contigs?)"));
    }

    // Resolve the ALT names against the full reference once so the mask
    // indexes global ref_ids, and install it before the per-tile pipelines
    // and the merge-phase stage 5/MAPQ read the config.
    cfg.pipeline_cfg.set_alt_mask(
        cfg.alt_contigs
            .as_ref()
            .and_then(|names| crate::aligner_core::resolve_alt_mask(names, &reference)),
    );

    crate::kira_info!(
        "[KIRA_TILE] running split-prefix pipeline with {} tile(s); temp prefix = {}",
        tile_plan.n_tiles(),
        cfg.split_prefix.display()
    );

    let scratch = scratch_directory(&cfg.split_prefix)?;
    let mut chunk_paths: Vec<PathBuf> = Vec::with_capacity(tile_plan.n_tiles());
    for (tile_idx, tile) in tile_plan.tiles.iter().enumerate() {
        let t0 = Instant::now();
        crate::kira_info!(
            "[KIRA_TILE] tile {}/{}: contigs [{}..{}], {} bytes — building index",
            tile_idx + 1,
            tile_plan.n_tiles(),
            tile.contig_start,
            tile.contig_end,
            tile.total_bytes
        );

        let sub_ref = tile.build_sub_reference(&reference);
        let index = Index::try_build(sub_ref, cfg.index_cfg)?;

        let chunk_path = scratch.path().join(format!("tile-{tile_idx}.kchunk"));
        chunk_paths.push(chunk_path.clone());
        let mut writer = ChunkWriter::create(&chunk_path)
            .with_context(|| format!("create chunk file {}", chunk_path.display()))?;
        let mut repeats =
            repeat_counts::RepeatPass::open(scratch.path(), tile_idx, tile_plan.n_tiles())?;

        let mut stream = ReadStream::new_multi_with_opts(
            reads_paths,
            cfg.batch_bases,
            cfg.pipeline_cfg.paired.mode,
            cfg.keep_comment,
        )?;
        let junctions = cfg.junctions.as_ref().map(|junctions| {
            Arc::new(junctions.for_tile(
                tile.global_ref_id_offset,
                tile.contig_end - tile.contig_start,
            ))
        });
        let pipeline = Pipeline::with_pool(cfg.pipeline_cfg.clone(), Arc::clone(&pool))
            .with_junctions(junctions, cfg.junc_bed_tolerance);

        let mut global_read_idx: u64 = 0;
        let mut total_alignments_written: u64 = 0;
        while let Some(reads) = stream.next_batch()? {
            let n = reads.len();
            let input = stage0_input::run(reads);
            let (mut align, counts) = if tile_plan.n_tiles() == 1 {
                (pipeline.process_to_align_batch(input, &index), Vec::new())
            } else {
                pipeline.process_tile_candidates(input, &index, cfg.index_cfg)
            };
            repeats.record(&align.reads, &counts)?;
            remap_alignments_to_global(&mut align.alignments, tile);

            for (i, alns) in align.alignments.iter().enumerate() {
                writer.write_record(global_read_idx + i as u64, alns)?;
                total_alignments_written += alns.len() as u64;
            }
            global_read_idx += n as u64;
        }
        writer.finish()?;
        repeats.finish()?;

        crate::kira_info!(
            "[KIRA_TILE] tile {}/{} done in {:.1}s: {} reads, {} alignments written",
            tile_idx + 1,
            tile_plan.n_tiles(),
            t0.elapsed().as_secs_f64(),
            global_read_idx,
            total_alignments_written
        );
    }

    let t_merge = Instant::now();
    crate::kira_info!(
        "[KIRA_TILE] merging {} tile(s) → final SAM",
        chunk_paths.len()
    );

    let mut merge =
        MergeIter::from_paths(&chunk_paths, scratch.path()).context("open tile merge")?;

    let (mut writer, publication) = SamOutput::open(output_path.as_deref(), &reference)?;
    if cfg.pipeline_cfg.output.format != crate::io::EmitFormat::Paf {
        match &cfg.header {
            Some(hdr) => writer.write_header_with_ctx(hdr)?,
            None => writer.write_header_with_rg(cfg.read_group.as_deref())?,
        }
    }
    let formatter = writer.formatter_handle();

    let mut stream = ReadStream::new_multi_with_opts(
        reads_paths,
        cfg.batch_bases,
        cfg.pipeline_cfg.paired.mode,
        cfg.keep_comment,
    )?;

    // The same post-stage-4 policy chain as the in-memory driver, with one
    // insert-size estimator refined across the merged batches.
    let estimator = RwLock::new(InsertEstimator::new(cfg.pipeline_cfg.paired));
    let policy = PostAlignPolicy {
        reference: &reference,
        mmap: None,
        config: &cfg.pipeline_cfg,
        pool: &pool,
    };
    let preserve_primary = cfg.pipeline_cfg.primary_5p || cfg.pipeline_cfg.alt_mask.is_some();

    let mut global_read_idx: u64 = 0;
    let mut next_merged: Option<(u64, Vec<Alignment>)> = merge.next_merged()?;

    let mut repeats = repeat_counts::RepeatSummary::open(scratch.path())?;
    while let Some(mut reads) = stream.next_batch()? {
        repeats.restore(&mut reads)?;
        let n = reads.len();
        let mut alignments: Vec<Vec<Alignment>> = vec![Vec::new(); n];
        for (i, slot) in alignments.iter_mut().enumerate() {
            let this_idx = global_read_idx + i as u64;
            while let Some((m_idx, _)) = &next_merged {
                if *m_idx < this_idx {
                    next_merged = merge.next_merged()?;
                } else {
                    break;
                }
            }
            if let Some((m_idx, _)) = &next_merged
                && *m_idx == this_idx
            {
                if let Some((_, alns)) = next_merged.take() {
                    *slot = alns;
                }
                next_merged = merge.next_merged()?;
            }
        }
        global_read_idx += n as u64;

        let paired_cfg = estimator
            .read()
            .map(|e| e.current())
            .unwrap_or(cfg.pipeline_cfg.paired);
        let mut unmapped_mate_info: Vec<Option<MateInfo>> = vec![None; n];
        let post = policy.apply(
            &reads,
            &mut alignments,
            &mut unmapped_mate_info,
            &paired_cfg,
            Some(&estimator),
        );

        let pair_ctx = if post.paired_cfg.is_paired() {
            Some(PairMapqContext {
                insert_mean: post.paired_cfg.insert_mean,
                insert_sd: post.paired_cfg.insert_sd,
                discordant_cap: 10,
            })
        } else {
            None
        };
        let scored = stage5_scoring::run_with_primary_policy(
            AlignBatch {
                reads,
                alignments,
                unmapped_mate_info,
                stats: AlignmentBatchStats::default(),
            },
            cfg.pipeline_cfg.mapq.clone(),
            pair_ctx,
            preserve_primary,
        );
        let chunks = serialize_chunks(
            scored,
            &formatter,
            cfg.read_group.as_deref(),
            cfg.pipeline_cfg.output,
            cfg.pipeline_cfg.max_alignments,
        )?;
        writer.write_chunks(&chunks)?;
    }

    repeats.finish()?;
    writer.flush()?;
    drop(writer);
    publication.publish()?;
    crate::kira_info!(
        "[KIRA_TILE] merge done in {:.1}s; cleaning up {} chunk file(s)",
        t_merge.elapsed().as_secs_f64(),
        chunk_paths.len()
    );

    Ok(())
}

fn scratch_directory(prefix: &Path) -> Result<tempfile::TempDir> {
    let parent = prefix
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = prefix
        .file_name()
        .context("split prefix requires a filename")?;
    tempfile::Builder::new()
        .prefix(name)
        .tempdir_in(parent)
        .with_context(|| format!("create tile directory for {}", prefix.display()))
}

/// Translate every `ref_id` (and `mate.mate_ref_id`) in the batch from tile-local to global.
fn remap_alignments_to_global(
    alignments: &mut [Vec<Alignment>],
    tile: &crate::index::tiling::Tile,
) {
    let offset = tile.global_ref_id_offset;
    if offset == 0 {
        // Fast path: first tile has offset 0 — local == global.
        return;
    }
    for alns in alignments.iter_mut() {
        for a in alns.iter_mut() {
            a.ref_id = a.ref_id.saturating_add(offset);
            if let Some(mr) = a.mate.mate_ref_id {
                a.mate.mate_ref_id = Some(mr.saturating_add(offset));
            }
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/pipeline_tiled.rs"]
mod tests;
