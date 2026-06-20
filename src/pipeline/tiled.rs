//! Split-prefix tiled aligner.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};

use crate::exec::DualPool;
use crate::exec::pool::DualPoolConfig;
use crate::index::tiling::TilePlan;
use crate::index::{Index, IndexConfig};
use crate::io::{HeaderConfig, ReadStream, SamWriter};
use crate::mapq::{assign_mapq_preserving_primary, assign_mapq_with_qual};
use crate::pipeline::chunk_io::{ChunkReader, ChunkWriter, MergeIter};
use std::sync::Arc;

use crate::alignment::junc_bed::JunctionIndex;
use crate::mapq::PairMapqContext;
use crate::pipeline::pairing::{
    RescueConfig, apply_pairing, pair_rerank, rescue_discordant_pairs_with_ref,
    rescue_unmapped_mates_with_ref,
};
use crate::pipeline::stage0_input;
use crate::pipeline::stage4_alignment::AlignmentBatchStats;
use crate::pipeline::stage5_scoring::ScoredBatch;
use crate::pipeline::stage6_output::serialize as output_serialize;
use crate::pipeline::{Pipeline, PipelineConfig};
use crate::types::{Alignment, MateInfo, Reference};

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
    eprintln!(
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
    if tile_plan.tiles.is_empty() {
        return Err(anyhow::anyhow!("tile plan is empty (no contigs?)"));
    }

    eprintln!(
        "[KIRA_TILE] running split-prefix pipeline with {} tile(s); temp prefix = {}",
        tile_plan.n_tiles(),
        cfg.split_prefix.display()
    );

    let mut chunk_paths: Vec<PathBuf> = Vec::with_capacity(tile_plan.n_tiles());
    for (tile_idx, tile) in tile_plan.tiles.iter().enumerate() {
        let t0 = Instant::now();
        eprintln!(
            "[KIRA_TILE] tile {}/{}: contigs [{}..{}], {} bytes — building index",
            tile_idx + 1,
            tile_plan.n_tiles(),
            tile.contig_start,
            tile.contig_end,
            tile.total_bytes
        );

        let sub_ref = tile.build_sub_reference(&reference);
        let index = Index::build(sub_ref, cfg.index_cfg);

        let chunk_path = chunk_path_for(&cfg.split_prefix, tile_idx);
        chunk_paths.push(chunk_path.clone());
        let mut writer = ChunkWriter::create(&chunk_path)
            .with_context(|| format!("create chunk file {}", chunk_path.display()))?;

        let mut stream = ReadStream::new_multi_with_mode(
            reads_paths,
            cfg.batch_bases,
            cfg.pipeline_cfg.paired.mode,
        )?;
        let pipeline = Pipeline::with_pool(cfg.pipeline_cfg, Arc::clone(&pool))
            .with_junctions(cfg.junctions.clone(), cfg.junc_bed_tolerance);

        let mut global_read_idx: u64 = 0;
        let mut total_alignments_written: u64 = 0;
        while let Some(reads) = stream.next_batch()? {
            let n = reads.len();
            let input = stage0_input::run(reads);
            let mut align = pipeline.process_to_align_batch(input, &index);
            remap_alignments_to_global(&mut align.alignments, tile);

            for (i, alns) in align.alignments.iter().enumerate() {
                writer.write_record(global_read_idx + i as u64, alns)?;
                total_alignments_written += alns.len() as u64;
            }
            global_read_idx += n as u64;
        }
        writer.finish()?;

        eprintln!(
            "[KIRA_TILE] tile {}/{} done in {:.1}s: {} reads, {} alignments written",
            tile_idx + 1,
            tile_plan.n_tiles(),
            t0.elapsed().as_secs_f64(),
            global_read_idx,
            total_alignments_written
        );
    }

    let t_merge = Instant::now();
    eprintln!(
        "[KIRA_TILE] merging {} tile(s) → final SAM",
        chunk_paths.len()
    );

    let mut readers: Vec<ChunkReader> = Vec::with_capacity(chunk_paths.len());
    for path in &chunk_paths {
        readers.push(
            ChunkReader::open(path)
                .with_context(|| format!("open chunk file {}", path.display()))?,
        );
    }
    let mut merge = MergeIter::new(readers);

    let mut writer = SamWriter::new(output_path.clone(), reference.clone())?;
    match &cfg.header {
        Some(hdr) => writer.write_header_with_ctx(hdr)?,
        None => writer.write_header_with_rg(cfg.read_group.as_deref())?,
    }
    let formatter = writer.formatter_handle();

    let mut stream = ReadStream::new_multi_with_mode(
        reads_paths,
        cfg.batch_bases,
        cfg.pipeline_cfg.paired.mode,
    )?;

    let mut global_read_idx: u64 = 0;
    let mut next_merged: Option<(u64, Vec<Alignment>)> = merge.next_merged()?;

    while let Some(reads) = stream.next_batch()? {
        let n = reads.len();
        let mut alignments: Vec<Vec<Alignment>> = vec![Vec::new(); n];
        for i in 0..n {
            let this_idx = global_read_idx + i as u64;
            while let Some((m_idx, _)) = &next_merged {
                if *m_idx < this_idx {
                    next_merged = merge.next_merged()?;
                } else {
                    break;
                }
            }
            if let Some((m_idx, _)) = &next_merged {
                if *m_idx == this_idx {
                    if let Some((_, alns)) = next_merged.take() {
                        alignments[i] = alns;
                    }
                    next_merged = merge.next_merged()?;
                }
            }
        }
        global_read_idx += n as u64;

        rescue_unmapped_mates_with_ref(
            &reads,
            &mut alignments,
            &reference,
            None,
            &cfg.pipeline_cfg.paired,
            cfg.pipeline_cfg.alignment,
            RescueConfig::default(),
        );

        pair_rerank(
            &reads,
            &mut alignments,
            &cfg.pipeline_cfg.paired,
            cfg.pipeline_cfg.dp_topk.max(2),
        );

        rescue_discordant_pairs_with_ref(
            &reads,
            &mut alignments,
            &reference,
            None,
            &cfg.pipeline_cfg.paired,
            cfg.pipeline_cfg.alignment,
            RescueConfig::default(),
        );

        let mut unmapped_mate_info: Vec<Option<MateInfo>> = vec![None; n];
        apply_pairing(
            &reads,
            &mut alignments,
            &mut unmapped_mate_info,
            &cfg.pipeline_cfg.paired,
        );

        let pair_ctx = if cfg.pipeline_cfg.paired.is_paired() {
            Some(PairMapqContext {
                insert_mean: cfg.pipeline_cfg.paired.insert_mean,
                insert_sd: cfg.pipeline_cfg.paired.insert_sd,
                discordant_cap: 10,
            })
        } else {
            None
        };
        for (i, alns) in alignments.iter_mut().enumerate() {
            if reads[i].pair_role == crate::types::PairRole::Unpaired {
                assign_mapq_with_qual(
                    alns,
                    reads[i].seq.len(),
                    reads[i].qual.as_deref(),
                    cfg.pipeline_cfg.mapq,
                    pair_ctx,
                    reads[i].repeat_min_occ,
                );
            } else {
                assign_mapq_preserving_primary(
                    alns,
                    reads[i].seq.len(),
                    reads[i].qual.as_deref(),
                    cfg.pipeline_cfg.mapq,
                    pair_ctx,
                    reads[i].repeat_min_occ,
                );
            }
        }

        let scored = ScoredBatch {
            reads,
            alignments,
            unmapped_mate_info,
            stats: AlignmentBatchStats::default(),
        };
        let sam_buf = output_serialize(
            scored,
            &formatter,
            cfg.read_group.as_deref(),
            cfg.pipeline_cfg.output,
            cfg.pipeline_cfg.max_alignments,
        );
        writer.write_batch(&sam_buf)?;
    }

    writer.flush()?;
    eprintln!(
        "[KIRA_TILE] merge done in {:.1}s; cleaning up {} chunk file(s)",
        t_merge.elapsed().as_secs_f64(),
        chunk_paths.len()
    );

    for path in &chunk_paths {
        if let Err(e) = std::fs::remove_file(path) {
            eprintln!(
                "[KIRA_TILE] warning: could not remove {}: {}",
                path.display(),
                e
            );
        }
    }
    drop(formatter);

    Ok(())
}

/// Construct the chunk file path for a given tile index.
fn chunk_path_for(prefix: &Path, tile_idx: usize) -> PathBuf {
    let mut s = prefix.as_os_str().to_owned();
    s.push(format!(".tile-{:03}.kchunk", tile_idx));
    PathBuf::from(s)
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
