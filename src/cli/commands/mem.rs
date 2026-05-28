use anyhow::{Context, Result};

use crate::aligner_core::{Aligner, AlignerConfig};
use crate::alignment::AlignmentConfig;
use crate::chaining::ChainingConfig;
use crate::index::IndexConfig;
use std::sync::Arc;

use crate::alignment::junc_bed::JunctionIndex;
use crate::alignment::splice::{SpliceConfig, SpliceStrandPolicy};
use crate::index::tiling::plan_tiles;
use crate::io::{HeaderConfig, IngestMode, OutputConfig, read_reference};
use crate::mapq::MapqConfig;
use crate::pipeline::PipelineConfig;
use crate::pipeline::pairing::PairedConfig;
use crate::pipeline::stage1_sketch::SketchConfig;
use crate::pipeline::tiled::{TiledRunConfig, run_tiled};
use crate::seeding::SeedingConfig;

use crate::cli::MemArgs;

pub fn cmd_mem(mut args: MemArgs) -> Result<()> {
    if let Some(rg) = args.read_group.as_deref() {
        if rg.contains("\\t") {
            args.read_group = Some(rg.replace("\\t", "\t"));
        }
    }

    let preset = args.preset.to_lowercase();
    let short_preset = preset == "short";
    let auto_preset = preset == "auto";
    let splice_preset = preset == "splice" || preset == "splice:hq";
    let splice_hq = preset == "splice:hq";
    let (mut short_k, mut short_w, mut long_k, mut long_w) = match preset.as_str() {
        "short" => (19, 10, 19, 10),
        "long" => (15, 10, 15, 10),
        "splice" | "splice:hq" => (15, 5, 15, 5),
        _ => (19, 10, 15, 10),
    };
    if let Some(k) = args.seed_len {
        short_k = k;
        long_k = k;
    }
    if let Some(w) = args.window_len {
        short_w = w;
        long_w = w;
    }

    let index_cfg = IndexConfig {
        short_k,
        short_w,
        long_k,
        long_w,
        max_occ: 500,
        build_short: true,
        build_long: true,
    };

    let sketch_cfg = SketchConfig {
        short_k,
        short_w,
        long_k,
        long_w,
        long_read_threshold: args.long_read_threshold,
    };

    let seeding_cfg = SeedingConfig {
        min_anchor_len: 20,
        max_occ: 500,
        long_read_threshold: args.long_read_threshold,
    };

    let (max_dist, max_anchors, bandwidth, rmq_window) = if preset == "long" {
        (10_000, 5000, 200, 1024)
    } else if preset == "short" {
        (500, 2000, 50, 256)
    } else if splice_hq {
        (200_000, 10_000, 300, 4096)
    } else if splice_preset {
        (200_000, 5000, 200, 2048)
    } else {
        (5_000, 4000, 150, 512)
    };

    let chaining_cfg = ChainingConfig {
        max_dist,
        max_anchors,
        max_chains: 5,
        gap_open: 5,
        gap_extend: 1,
        log_gap: 0.2,
        rmq_window,
    };

    let alignment_cfg = AlignmentConfig {
        match_score: args.match_score,
        mismatch: args.mismatch_penalty,
        gap_open: args.gap_open,
        gap_extend: args.gap_extend,
        bandwidth,
        xdrop: 50,
        clip_penalty: args.clip_penalty,
    };

    let mapq_cfg = MapqConfig {
        short_read_len: args.long_read_threshold,
        mapq_cap_short: 60,
        mapq_cap_long: 60,
    };

    let (short_chaining, short_align, short_dp_topk) = if preset == "long" {
        (chaining_cfg, alignment_cfg, args.dp_topk)
    } else {
        (
            ChainingConfig {
                max_dist: 500,
                max_anchors: 2000,
                max_chains: 5,
                gap_open: 5,
                gap_extend: 1,
                log_gap: 0.2,
                rmq_window: 256,
            },
            AlignmentConfig {
                match_score: args.match_score,
                mismatch: args.mismatch_penalty,
                gap_open: args.gap_open,
                gap_extend: args.gap_extend,
                bandwidth: 50,
                xdrop: 50,
                clip_penalty: args.clip_penalty,
            },
            1,
        )
    };
    let (long_chaining, long_align, long_dp_topk) = if preset == "short" {
        (chaining_cfg, alignment_cfg, args.dp_topk)
    } else {
        (
            ChainingConfig {
                max_dist: 10_000,
                max_anchors: 5000,
                max_chains: 5,
                gap_open: 5,
                gap_extend: 1,
                log_gap: 0.2,
                rmq_window: 1024,
            },
            AlignmentConfig {
                match_score: args.match_score,
                mismatch: args.mismatch_penalty,
                gap_open: args.gap_open,
                gap_extend: args.gap_extend,
                bandwidth: 200,
                xdrop: 50,
                clip_penalty: args.clip_penalty,
            },
            2,
        )
    };

    let hybrid_chaining = chaining_cfg;
    let hybrid_align = alignment_cfg;
    let hybrid_dp_topk = args.dp_topk;

    let emit_lower = args.emit.to_ascii_lowercase();
    let output_cfg = match emit_lower.as_str() {
        "paf" => OutputConfig::paf(),
        _ if args.fast_output => OutputConfig::fast(),
        _ => OutputConfig::full(),
    };

    let accept_enable = args.accept_enable.unwrap_or(true);

    let (paired_mode, auto_detected_pe) =
        resolve_paired_mode(args.paired, args.interleaved, args.reads.len())?;
    if auto_detected_pe {
        eprintln!(
            "[KIRA] auto-detected 2 FASTQ inputs as paired R1+R2 (bwa-mem convention). \
             Pass --paired to silence this notice, or provide >2 files / 1 file for \
             single-end concatenation."
        );
    }
    let mut paired_cfg = PairedConfig::default();
    paired_cfg.mode = paired_mode;
    paired_cfg
        .apply_insert_spec(&args.insert_size)
        .map_err(|e| anyhow::anyhow!("--insert-size: {e}"))?;

    let strand_policy = match args.splice_strand.to_ascii_lowercase().as_str() {
        "auto" => SpliceStrandPolicy::Auto,
        "forward" | "+" => SpliceStrandPolicy::Forward,
        "reverse" | "-" => SpliceStrandPolicy::Reverse,
        "none" => SpliceStrandPolicy::None,
        other => {
            return Err(anyhow::anyhow!(
                "--splice-strand: unknown value {:?} (accepted: auto, forward, reverse, none)",
                other
            ));
        }
    };
    let min_intron = if splice_hq && args.min_intron == 30 {
        25 // tighter default for long-read RNA-seq (some short introns)
    } else {
        args.min_intron
    };
    let splice_cfg = SpliceConfig {
        enabled: splice_preset,
        min_intron,
        max_intron: args.max_intron,
        strand_policy,
        require_signal: false,
        splice_flank: args.splice_flank,
        min_exon_len: args.min_exon,
        polya_min_len: args.polya_min_len,
    };

    let junctions_arc: Option<Arc<JunctionIndex>> = match &args.junc_bed {
        Some(path) => {
            if !splice_preset {
                return Err(anyhow::anyhow!(
                    "--junc-bed requires `-x splice` or `-x splice:hq`"
                ));
            }
            let reference = read_reference(&args.reference)
                .context("load reference for --junc-bed name resolution")?;
            let idx = JunctionIndex::from_bed_path(path, &reference)
                .with_context(|| format!("parse --junc-bed {}", path.display()))?;
            if idx.is_empty() {
                eprintln!(
                    "[KIRA_JUNCBED] warning: 0 usable junctions loaded from {}; \
                     splice path will rely on signal detection only",
                    path.display()
                );
            }
            Some(Arc::new(idx))
        }
        None => None,
    };

    let pipeline_cfg = PipelineConfig {
        sketch: sketch_cfg,
        seeding: seeding_cfg,
        chaining: chaining_cfg,
        alignment: alignment_cfg,
        accept_enable,
        accept_only_top1: args.accept_only_top1,
        accept_span_slack: args.accept_span_slack,
        accept_min_identity: args.accept_min_identity,
        accept_max_mismatches: args.accept_max_mismatches,
        accept_require_score_margin: args.accept_require_score_margin,
        dp_topk: args.dp_topk,
        dp_abort_margin: 20,
        debug_prefilter: args.debug_prefilter_n > 0,
        debug_prefilter_n: args.debug_prefilter_n,
        debug_force_accept: args.debug_force_accept,
        debug_force_accept_n: args.debug_force_accept_n,
        long_read_threshold: args.long_read_threshold,
        max_alignments: args.max_alignments,
        min_chain_ratio: args.min_chain_ratio,
        short_preset,
        mapq: mapq_cfg,
        output: output_cfg,
        paired: paired_cfg,
        splice: splice_cfg,
    };

    let auto_profiles = if auto_preset {
        Some(crate::pipeline::mode::ReadModeProfiles {
            short: PipelineConfig {
                chaining: short_chaining,
                alignment: short_align,
                dp_topk: short_dp_topk,
                short_preset: true,
                ..pipeline_cfg
            },
            long: PipelineConfig {
                chaining: long_chaining,
                alignment: long_align,
                dp_topk: long_dp_topk,
                short_preset: false,
                ..pipeline_cfg
            },
            hybrid: PipelineConfig {
                chaining: hybrid_chaining,
                alignment: hybrid_align,
                dp_topk: hybrid_dp_topk,
                short_preset: false,
                ..pipeline_cfg
            },
            decided: None,
        })
    } else {
        None
    };

    let mut header = HeaderConfig::default();
    header.read_group = args.read_group.clone();
    if !args.no_pg {
        let cli = std::env::args()
            .map(|a| {
                let a = a.replace('\t', "\\t").replace('\n', "\\n");
                if a.contains(' ') { format!("'{a}'") } else { a }
            })
            .collect::<Vec<_>>()
            .join(" ");
        let pg = format!(
            "ID:kira-ls-aligner\tPN:kira-ls-aligner\tVN:{}\tCL:{}",
            env!("CARGO_PKG_VERSION"),
            cli
        );
        header.pg_lines.push(pg);
    }
    for pg in &args.pg {
        header.pg_lines.push(pg.clone());
    }
    for co in &args.co {
        header.co_lines.push(co.clone());
    }

    let cfg = AlignerConfig {
        threads: args.threads,
        batch_bases: args.batch_bases,
        index: index_cfg,
        pipeline: pipeline_cfg,
        auto_profiles,
        read_group: args.read_group.clone(),
        header: Some(header),
        junctions: junctions_arc.clone(),
        junc_bed_tolerance: args.junc_bed_tolerance,
    };

    if let Some(split_prefix) = args.split_prefix.clone() {
        if args.index.is_some() {
            return Err(anyhow::anyhow!(
                "--split-prefix is incompatible with --index (tiled mode builds per-tile indices in memory)"
            ));
        }
        if args.tile_bytes < 100_000_000 {
            return Err(anyhow::anyhow!(
                "--tile-bytes must be at least 100 MB, got {} bytes",
                args.tile_bytes
            ));
        }
        if matches!(emit_lower.as_str(), "bam" | "sorted-bam") {
            return Err(anyhow::anyhow!(
                "--split-prefix with --emit {} is not yet supported; emit SAM and convert externally",
                emit_lower
            ));
        }
        return run_split_prefix(&args, cfg, split_prefix);
    }

    let aligner = Aligner::new(cfg);

    let resolved_index: Option<std::path::PathBuf> = if splice_preset {
        if args.index.is_some() {
            eprintln!(
                "[KIRA] warning: --index is being ignored in splice mode \
                 (k/w mismatch would produce zero alignments). Building \
                 a fresh splice-tuned index in memory."
            );
        }
        None
    } else {
        args.index.clone().or_else(|| {
            let candidate = args.reference.with_extension("kiraidx");
            if candidate.is_file() {
                eprintln!(
                    "[KIRA] auto-detected sidecar index {} for {}",
                    candidate.display(),
                    args.reference.display()
                );
                Some(candidate)
            } else {
                None
            }
        })
    };

    let emit = emit_lower.as_str();
    let want_bam = matches!(emit, "bam" | "sorted-bam");
    let want_sort = emit == "sorted-bam";
    if !matches!(emit, "" | "sam" | "paf" | "bam" | "sorted-bam") {
        return Err(anyhow::anyhow!(
            "unknown --emit value {:?} (accepted: sam, paf, bam, sorted-bam)",
            args.emit
        ));
    }

    if !want_bam {
        return if let Some(index_path) = resolved_index.as_ref() {
            aligner.run_with_index_file(index_path, &args.reads, args.output.as_ref())
        } else {
            aligner.run(&args.reference, &args.reads, args.output.as_ref())
        };
    }

    let final_path = args
        .output
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--emit bam/sorted-bam requires -o <out.bam>"))?;

    let tmp_dir = final_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")));
    let tmp = tempfile::Builder::new()
        .prefix("kira-ls-aligner-")
        .suffix(".sam")
        .tempfile_in(&tmp_dir)
        .with_context(|| format!("create tmp SAM in {}", tmp_dir.display()))?;
    let tmp_path = tmp.into_temp_path();
    let tmp_sam: std::path::PathBuf = (&tmp_path as &std::path::Path).to_path_buf();

    eprintln!(
        "[KIRA] --emit {} → writing intermediate SAM to {}",
        emit,
        tmp_sam.display()
    );

    if let Some(index_path) = resolved_index.as_ref() {
        aligner.run_with_index_file(index_path, &args.reads, Some(&tmp_sam))?;
    } else {
        aligner.run(&args.reference, &args.reads, Some(&tmp_sam))?;
    }

    eprintln!("[KIRA] aligner done, converting via kira-bam...");

    if want_sort {
        kira_bam::sort::run(kira_bam::cli::SortArgs {
            input: tmp_sam.clone(),
            output: Some(final_path.clone()),
            name_sort: false,
            threads: args.threads,
            memory: "auto".to_string(),
            tmpdir: None,
            uncompressed: false,
            require_flags: 0,
            filter_flags: 0,
            no_pg: false,
            reference: None,
            markdup: args.markdup,
            markdup_barcode_tag: None,
            markdup_mode_ancient: false,
        })?;
    } else {
        kira_bam::view::run(kira_bam::cli::ViewArgs {
            input: tmp_sam.clone(),
            regions: Vec::new(),
            exclude_regions: Vec::new(),
            output: Some(final_path.clone()),
            bam: true,
            sam: false,
            uncompressed: false,
            with_header: true,
            header_only: false,
            require_flags: 0,
            filter_flags: 0,
            min_mapq: 0,
            threads: args.threads,
            count: false,
            drop_tags: false,
            no_pg: false,
            cram: false,
            reference: None,
        })?;
        if args.markdup {
            eprintln!("[KIRA] warning: --markdup requires sorted input; skipping");
        }
    }

    if args.bai {
        kira_bam::index::run(kira_bam::cli::IndexArgs {
            input: final_path.clone(),
            output: None,
            bai: true,
            csi: false,
            min_shift: 14,
            depth: 5,
            threads: args.threads,
        })?;
    }

    drop(tmp_path);
    eprintln!("[KIRA] BAM pipeline complete → {}", final_path.display());
    Ok(())
}

/// Split-prefix entry point.
fn run_split_prefix(
    args: &crate::cli::MemArgs,
    cfg: AlignerConfig,
    split_prefix: std::path::PathBuf,
) -> anyhow::Result<()> {
    let reference = read_reference(&args.reference).context("load reference for split-prefix")?;
    let tile_plan = plan_tiles(&reference, args.tile_bytes);
    if tile_plan.tiles.is_empty() {
        return Err(anyhow::anyhow!(
            "reference {} contains no contigs",
            args.reference.display()
        ));
    }
    if tile_plan.is_trivial() {
        eprintln!(
            "[KIRA_TILE] note: reference fits in a single tile ({} bytes ≤ tile-bytes {}). \
             Tiled pipeline still runs but it's equivalent to the single-pass path.",
            tile_plan.tiles[0].total_bytes, args.tile_bytes
        );
    }
    let tiled_cfg = TiledRunConfig {
        threads: cfg.threads,
        batch_bases: cfg.batch_bases,
        index_cfg: cfg.index,
        pipeline_cfg: cfg.pipeline,
        read_group: cfg.read_group.clone(),
        header: cfg.header.clone(),
        split_prefix,
        junctions: cfg.junctions.clone(),
        junc_bed_tolerance: cfg.junc_bed_tolerance,
    };
    run_tiled(reference, &args.reads, args.output.clone(), tiled_cfg, tile_plan)
}

/// Resolve the FASTQ ingestion mode from the `--paired` / `--interleaved` flags and the input file.
pub(crate) fn resolve_paired_mode(
    paired: bool,
    interleaved: bool,
    n_reads: usize,
) -> anyhow::Result<(IngestMode, bool)> {
    let paired_active = paired || interleaved;
    if !paired_active {
        if n_reads == 2 {
            return Ok((IngestMode::TwoFile, true));
        }
        return Ok((IngestMode::Unpaired, false));
    }
    if interleaved {
        if n_reads != 1 {
            return Err(anyhow::anyhow!(
                "--interleaved requires exactly 1 FASTQ input, got {}",
                n_reads
            ));
        }
        return Ok((IngestMode::Interleaved, false));
    }
    match n_reads {
        2 => Ok((IngestMode::TwoFile, false)),
        1 => Err(anyhow::anyhow!(
            "--paired with a single FASTQ requires --interleaved (alternating R1/R2)"
        )),
        n => Err(anyhow::anyhow!(
            "--paired requires exactly 1 (with --interleaved) or 2 FASTQ inputs, got {}",
            n
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_pairs_two_fastqs_without_explicit_flag() {
        let (mode, auto) = resolve_paired_mode(false, false, 2).unwrap();
        assert_eq!(mode, IngestMode::TwoFile);
        assert!(auto, "auto-detect notice must be flagged for caller");
    }

    #[test]
    fn one_or_many_fastqs_stay_unpaired_without_explicit_flag() {
        let (mode, auto) = resolve_paired_mode(false, false, 1).unwrap();
        assert_eq!(mode, IngestMode::Unpaired);
        assert!(!auto);
        let (mode, auto) = resolve_paired_mode(false, false, 3).unwrap();
        assert_eq!(mode, IngestMode::Unpaired);
        assert!(!auto);
    }

    #[test]
    fn explicit_paired_does_not_trigger_auto_notice() {
        let (mode, auto) = resolve_paired_mode(true, false, 2).unwrap();
        assert_eq!(mode, IngestMode::TwoFile);
        assert!(!auto, "explicit --paired must not raise the notice");
    }

    #[test]
    fn interleaved_takes_one_file() {
        let (mode, _) = resolve_paired_mode(false, true, 1).unwrap();
        assert_eq!(mode, IngestMode::Interleaved);
        assert!(resolve_paired_mode(false, true, 2).is_err());
    }

    #[test]
    fn paired_with_wrong_count_errors() {
        assert!(resolve_paired_mode(true, false, 1).is_err());
        assert!(resolve_paired_mode(true, false, 3).is_err());
    }
}
