use anyhow::{Context, Result};

use crate::aligner_core::{Aligner, AlignerConfig};
use crate::alignment::AlignmentConfig;
use crate::chaining::ChainingConfig;
use crate::index::{Index, IndexConfig};
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

/// Read an i32 tuning knob from the environment, falling back to `default`.
fn env_i32(name: &str, default: i32) -> i32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Read an f32 tuning knob from the environment, falling back to `default`.
fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
}

/// Parameters for the fused in-process aligner ([`build_short_pe_aligner`]).
pub struct FusedAlignerParams {
    pub reference: std::path::PathBuf,
    pub index: Option<std::path::PathBuf>,
    pub threads: usize,
    pub num_p_threads: Option<usize>,
    pub num_e_threads: Option<usize>,
    pub batch_bases: usize,
    pub read_group: Option<String>,
    pub paired: bool,
    pub interleaved: bool,
    /// `MIN,MAX[,MEAN,SD]`.
    pub insert_size: String,
    /// Number of read files (for paired-mode auto-detection).
    pub n_read_files: usize,
}

/// Build a short-read (Illumina-style) aligner for the fused `kira-bt solid`
/// pipeline and return it plus the resolved index path. Mirrors the `-x short`
/// configuration of [`cmd_mem`] (k=19/w=10 short index, banded SW, ungapped
/// accept), with `auto_profiles = None` since the fused in-memory path uses the
/// pipeline config directly. Kept separate from `cmd_mem` to avoid the
/// file-emitting plumbing; the scoring/seeding values are intentionally the
/// same as the short preset.
pub fn build_short_pe_aligner(
    p: &FusedAlignerParams,
) -> Result<(crate::aligner_core::Aligner, Option<std::path::PathBuf>)> {
    let index_cfg = IndexConfig {
        short_k: 19,
        short_w: 10,
        long_k: 15,
        long_w: 10,
        max_occ: 500,
        build_short: true,
        build_long: true,
    };
    let sketch_cfg = SketchConfig {
        short_k: 19,
        short_w: 10,
        long_k: 15,
        long_w: 10,
        long_read_threshold: 500,
    };
    let mut seeding_cfg = SeedingConfig {
        min_anchor_len: 20,
        max_occ: 500,
        max_hits_per_minimizer: env_usize("KIRA_K_HITS", 16),
        long_read_threshold: 500,
        mate_window: None,
    };
    // KIRA_CHAIN_GAP_OPEN / KIRA_CHAIN_GAP_EXTEND / KIRA_CHAIN_MAX_DIST — chaining gap cost.
    // Lowering the gap cost lets seed anchors chain ACROSS an indel (different diagonals)
    // instead of splitting into a single-diagonal chain that hides the indel from the gapped
    // aligner. (KIRA_CHAIN_SCAN_MULT widens the predecessor scan budget — see chaining/rmq.rs.)
    let chaining_cfg = ChainingConfig {
        max_dist: env_i32("KIRA_CHAIN_MAX_DIST", 500) as u32,
        max_anchors: 2000,
        max_chains: 5,
        gap_open: env_i32("KIRA_CHAIN_GAP_OPEN", 5),
        gap_extend: env_i32("KIRA_CHAIN_GAP_EXTEND", 1),
        log_gap: 0.2,
        rmq_window: 256,
            keep_anchors: false,
    };
    let alignment_cfg = AlignmentConfig {
        match_score: 1,
        mismatch: 4,
        // KIRA_GAP_OPEN — WFA/DP gap-open cost. Default 6 makes WFA prefer a terminal mismatch over
        // a gap for near-end indels (1bp short side: mismatch 4 < gap 7), so they're never placed.
        // Lowering it places them (more indel recall) at some spurious-gap (FP) cost.
        gap_open: env_i32("KIRA_GAP_OPEN", 6),
        gap_extend: env_i32("KIRA_GAP_EXTEND", 1),
        // KIRA_BANDWIDTH / KIRA_XDROP / KIRA_CLIP_PENALTY — alignment-tuning knobs.
        bandwidth: env_i32("KIRA_BANDWIDTH", 50),
        xdrop: env_i32("KIRA_XDROP", 50),
        clip_penalty: env_i32("KIRA_CLIP_PENALTY", 5),
    };
    let mapq_cfg = MapqConfig {
        short_read_len: 500,
        mapq_cap_short: 60,
        mapq_cap_long: 60,
        alt_mask: None,
    };

    let (paired_mode, _auto) = resolve_paired_mode(p.paired, p.interleaved, p.n_read_files)?;
    let mut paired_cfg = PairedConfig::default();
    paired_cfg.mode = paired_mode;
    paired_cfg
        .apply_insert_spec(&p.insert_size)
        .map_err(|e| anyhow::anyhow!("--insert-size: {e}"))?;
    seeding_cfg.mate_window = mate_seed_window(&paired_cfg);

    let pipeline_cfg = PipelineConfig {
        sketch: sketch_cfg,
        seeding: seeding_cfg,
        chaining: chaining_cfg,
        alignment: alignment_cfg,
        accept_enable: true,
        accept_only_top1: true,
        accept_span_slack: 15,
        accept_min_identity: 98.5,
        accept_max_mismatches: 5,
        accept_require_score_margin: 0,
        dp_topk: 1,
        dp_abort_margin: 20,
        debug_prefilter: false,
        debug_prefilter_n: 0,
        debug_force_accept: false,
        debug_force_accept_n: 0,
        long_read_threshold: 500,
        max_alignments: 1,
        // KIRA_MIN_CHAIN_RATIO — min anchor-coverage fraction to keep a chain (default 0.4).
        min_chain_ratio: env_f32("KIRA_MIN_CHAIN_RATIO", 0.4),
        short_preset: true,
        mapq: mapq_cfg,
        output: OutputConfig::full(),
        paired: paired_cfg,
        splice: SpliceConfig::default(),
        min_output_score: 30,
        skip_mate_rescue: false,
        skip_pairing: false,
        primary_5p: false,
        alt_mask: None,
    };

    let mut header = HeaderConfig::default();
    header.read_group = p.read_group.clone();
    header.pg_lines.push(format!(
        "ID:kira-bt-solid\tPN:kira-bt\tVN:{}\tCL:kira-bt solid (fused)",
        env!("CARGO_PKG_VERSION")
    ));

    let cfg = AlignerConfig {
        threads: p.threads,
        num_p_threads: p.num_p_threads,
        num_e_threads: p.num_e_threads,
        batch_bases: p.batch_bases,
        index: index_cfg,
        pipeline: pipeline_cfg,
        auto_profiles: None,
        read_group: p.read_group.clone(),
        header: Some(header),
        junctions: None,
        junc_bed_tolerance: 2,
        keep_comment: false,
        alt_contigs: load_alt_contigs(&p.reference, false)?,
    };

    let resolved_index = p.index.clone().or_else(|| {
        let candidate = p.reference.with_extension("kiraidx");
        candidate.is_file().then_some(candidate)
    });

    Ok((crate::aligner_core::Aligner::new(cfg), resolved_index))
}

pub fn cmd_mem(mut args: MemArgs) -> Result<()> {
    crate::log::set_verbosity(args.verbosity);
    // Tuning-knob overrides go into the environment before anything reads a
    // knob: every `KIRA_*` is a lazily initialised `OnceLock`, and nothing has
    // touched one yet at this point.
    apply_knob_overrides(args.config.as_deref(), &args.set)?;
    let ignored = args.ignored_compat_flags();
    if !ignored.is_empty() {
        crate::kira_warn!(
            "[KIRA] note: bwa-mem flags accepted without effect: {}",
            ignored.join(" ")
        );
    }

    if let Some(rg) = args.read_group.as_deref() {
        if rg.contains("\\t") {
            args.read_group = Some(rg.replace("\\t", "\t"));
        }
    }
    let max_alignments = if args.output_all {
        // bwa-mem `-a`: report every alignment found. Stage 4 keeps at most
        // `max_chains` (5) candidates, so that is the ceiling worth asking for.
        args.max_alignments.max(5)
    } else {
        args.max_alignments
    };

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

    let mut seeding_cfg = SeedingConfig {
        min_anchor_len: 20,
        max_occ: 500,
        max_hits_per_minimizer: env_usize("KIRA_K_HITS", args.seed_occ_cap as usize),
        long_read_threshold: args.long_read_threshold,
        mate_window: None,
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
        // Only the splice aligner walks a chain's anchor path.
        keep_anchors: splice_preset,
    };

    let alignment_cfg = AlignmentConfig {
        match_score: args.match_score,
        mismatch: args.mismatch_penalty,
        gap_open: args.gap_open,
        gap_extend: args.gap_extend,
        bandwidth,
        xdrop: args.xdrop,
        clip_penalty: args.clip_penalty,
    };

    let mapq_cfg = MapqConfig {
        short_read_len: args.long_read_threshold,
        mapq_cap_short: 60,
        mapq_cap_long: 60,
        alt_mask: None,
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
            keep_anchors: false,
            },
            AlignmentConfig {
                match_score: args.match_score,
                mismatch: args.mismatch_penalty,
                gap_open: args.gap_open,
                gap_extend: args.gap_extend,
                bandwidth: 50,
                xdrop: args.xdrop,
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
            keep_anchors: false,
            },
            AlignmentConfig {
                match_score: args.match_score,
                mismatch: args.mismatch_penalty,
                gap_open: args.gap_open,
                gap_extend: args.gap_extend,
                bandwidth: 200,
                xdrop: args.xdrop,
                clip_penalty: args.clip_penalty,
            },
            2,
        )
    };

    let hybrid_chaining = chaining_cfg;
    let hybrid_align = alignment_cfg;
    let hybrid_dp_topk = args.dp_topk;

    let emit_lower = args.emit.to_ascii_lowercase();
    let mut output_cfg = match emit_lower.as_str() {
        "paf" => OutputConfig::paf(),
        _ if args.fast_output => OutputConfig::fast(),
        _ => OutputConfig::full(),
    };
    if emit_lower != "paf" {
        output_cfg.split_as_secondary = args.mark_split_secondary;
        output_cfg.soft_clip_supplementary = args.soft_clip_supplementary;
        output_cfg.xa_max = args.xa_max;
        output_cfg.append_comment = args.append_comment;
    }

    let accept_enable = resolve_accept_enable(args.accept_enable, args.fast_output);

    let (paired_mode, auto_detected_pe) =
        resolve_paired_mode(args.paired, args.interleaved, args.reads.len())?;
    if auto_detected_pe {
        crate::kira_info!("[KIRA] auto-detected 2 FASTQ inputs as paired R1+R2 (bwa-mem convention). \
             Pass --paired to silence this notice, or provide >2 files / 1 file for \
             single-end concatenation."
        );
    }
    let mut paired_cfg = PairedConfig::default();
    paired_cfg.mode = paired_mode;
    paired_cfg
        .apply_insert_spec(&args.insert_size)
        .map_err(|e| anyhow::anyhow!("--insert-size: {e}"))?;
    seeding_cfg.mate_window = mate_seed_window(&paired_cfg);

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
                crate::kira_warn!("[KIRA_JUNCBED] warning: 0 usable junctions loaded from {}; \
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
        max_alignments,
        min_chain_ratio: args.min_chain_ratio,
        short_preset,
        mapq: mapq_cfg,
        output: output_cfg,
        paired: paired_cfg,
        splice: splice_cfg,
        min_output_score: args.min_score,
        skip_mate_rescue: args.skip_mate_rescue,
        skip_pairing: args.skip_pairing,
        primary_5p: args.primary_5p,
        alt_mask: None,
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
    for item in &args.header_insert {
        header.extra_lines.extend(header_insert_lines(item)?);
    }
    if !args.no_pg {
        // Provenance the `@PG CL` cannot carry: the tuning knobs in force and
        // the effective pipeline parameters. With these a BAM header alone
        // reproduces the run.
        let knobs: Vec<String> = {
            let mut v: Vec<(String, String)> = std::env::vars()
                .filter(|(k, _)| k.starts_with("KIRA_"))
                .collect();
            v.sort();
            v.into_iter().map(|(k, val)| format!("{k}={val}")).collect()
        };
        if !knobs.is_empty() {
            header.co_lines.push(format!("kira-env:{}", knobs.join(" ")));
        }
        header.co_lines.push(format!(
            "kira-config:{}",
            crate::aligner_core::config_fingerprint(&pipeline_cfg, args.threads, args.batch_bases)
        ));
    }
    for co in &args.co {
        header.co_lines.push(co.clone());
    }

    let cfg = AlignerConfig {
        threads: args.threads,
        num_p_threads: args.num_p_threads,
        num_e_threads: args.num_e_threads,
        batch_bases: args.batch_bases,
        index: index_cfg,
        pipeline: pipeline_cfg,
        auto_profiles,
        read_group: args.read_group.clone(),
        header: Some(header),
        junctions: junctions_arc.clone(),
        junc_bed_tolerance: args.junc_bed_tolerance,
        keep_comment: args.append_comment,
        alt_contigs: load_alt_contigs(&args.reference, args.ignore_alt)?,
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
        if super::emit::EmitKind::parse(&emit_lower).is_some() {
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
            crate::kira_warn!("[KIRA] warning: --index is being ignored in splice mode \
                 (k/w mismatch would produce zero alignments). Building \
                 a fresh splice-tuned index in memory."
            );
        }
        None
    } else {
        args.index.clone().or_else(|| {
            let candidate = args.reference.with_extension("kiraidx");
            if candidate.is_file() {
                crate::kira_info!("[KIRA] auto-detected sidecar index {} for {}",
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
    let fused_kind = super::emit::EmitKind::parse(emit);
    if fused_kind.is_none() && !matches!(emit, "" | "sam" | "paf") {
        return Err(anyhow::anyhow!(
            "unknown --emit value {:?} (accepted: sam, paf, bam, sorted-bam, cram, sorted-cram)",
            args.emit
        ));
    }

    let Some(kind) = fused_kind else {
        return if let Some(index_path) = resolved_index.as_ref() {
            aligner.run_with_index_file(index_path, &args.reads, args.output.as_ref())
        } else {
            aligner.run(&args.reference, &args.reads, args.output.as_ref())
        };
    };

    let final_path = args
        .output
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--emit {emit} requires -o <output>"))?;
    if kind.is_cram()
        && !final_path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("cram"))
    {
        return Err(anyhow::anyhow!(
            "--emit {emit} requires the output path to end in .cram (got {})",
            final_path.display()
        ));
    }
    if args.markdup && !kind.is_sorted() {
        crate::kira_warn!("[KIRA] warning: --markdup requires --emit sorted-bam/sorted-cram; skipping");
    }

    // The fused path drives the pipeline itself, so it needs the index in hand.
    let index = match resolved_index.as_ref() {
        Some(p) => Index::load(p).with_context(|| format!("load index {}", p.display()))?,
        None => {
            let reference = read_reference(&args.reference).context("load reference")?;
            Index::build(reference, index_cfg)
        }
    };

    super::emit::run_fused(
        &aligner,
        index,
        &args.reads,
        super::emit::EmitOptions {
            kind,
            output: final_path,
            reference: args.reference.clone(),
            threads: args.threads,
            markdup: args.markdup && kind.is_sorted(),
            bai: args.bai,
            sort_memory: args.sort_memory.clone(),
            no_pg: args.no_pg,
        },
    )
}

/// Apply `--config FILE` then `--set KEY=VALUE` to the process environment.
///
/// Only `KIRA_*` names are accepted: the file is a knob file, not a general
/// environment, and a typo should fail loudly rather than set a dead variable.
fn apply_knob_overrides(config: Option<&std::path::Path>, sets: &[String]) -> Result<()> {
    let mut pairs: Vec<(String, String)> = Vec::new();
    if let Some(path) = config {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read --config {}", path.display()))?;
        for (lineno, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (k, v) = line.split_once('=').ok_or_else(|| {
                anyhow::anyhow!(
                    "{}:{}: expected KIRA_KNOB=value, got {:?}",
                    path.display(),
                    lineno + 1,
                    raw
                )
            })?;
            pairs.push((k.trim().to_string(), v.trim().to_string()));
        }
    }
    for s in sets {
        let (k, v) = s
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("--set expects KIRA_KNOB=value, got {s:?}"))?;
        pairs.push((k.trim().to_string(), v.trim().to_string()));
    }
    for (k, v) in pairs {
        if !k.starts_with("KIRA_") || k.len() <= 5 {
            return Err(anyhow::anyhow!(
                "tuning knob {k:?} must be a KIRA_* name (see README, \"Tuning knobs\")"
            ));
        }
        // SAFETY: called from `cmd_mem` before any worker thread exists —
        // argument parsing is the only thing that has run — so no other
        // thread can be reading the environment concurrently.
        unsafe { std::env::set_var(&k, &v) };
    }
    Ok(())
}

/// `REF.alt` (bwa-mem convention, shipped with the GRCh38 full analysis set):
/// a SAM-format file whose QNAMEs are the ALT contigs. Absent file ⇒ no ALT
/// handling; `-j` skips it.
fn load_alt_contigs(
    reference: &std::path::Path,
    ignore: bool,
) -> Result<Option<Arc<std::collections::HashSet<String>>>> {
    let mut alt = reference.as_os_str().to_os_string();
    alt.push(".alt");
    let alt = std::path::PathBuf::from(alt);
    if ignore || !alt.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&alt)
        .with_context(|| format!("read ALT contig file {}", alt.display()))?;
    let names: std::collections::HashSet<String> = text
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('@'))
        .filter_map(|l| l.split('\t').next())
        .map(str::to_string)
        .collect();
    if names.is_empty() {
        crate::kira_warn!("[KIRA] warning: {} lists no ALT contigs", alt.display());
        return Ok(None);
    }
    crate::kira_info!("[KIRA] {} ALT contigs from {}", names.len(), alt.display());
    Ok(Some(Arc::new(names)))
}

/// bwa-mem `-H STR|FILE`: header lines to insert verbatim. A value starting
/// with `@` is one line (literal `\t` becomes a tab); anything else is a file
/// of such lines.
fn header_insert_lines(item: &str) -> Result<Vec<String>> {
    if let Some(rest) = item.strip_prefix('@') {
        return Ok(vec![format!("@{}", rest.replace("\\t", "\t"))]);
    }
    let text = std::fs::read_to_string(item)
        .with_context(|| format!("read -H header file {item}"))?;
    let lines: Vec<String> = text
        .lines()
        .map(str::trim_end)
        .filter(|l| !l.is_empty())
        .map(|l| {
            if l.starts_with('@') {
                Ok(l.to_string())
            } else {
                Err(anyhow::anyhow!(
                    "-H {item}: header lines must start with '@', got {l:?}"
                ))
            }
        })
        .collect::<Result<_>>()?;
    Ok(lines)
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
        crate::kira_info!("[KIRA_TILE] note: reference fits in a single tile ({} bytes ≤ tile-bytes {}). \
             Tiled pipeline still runs but it's equivalent to the single-pass path.",
            tile_plan.tiles[0].total_bytes, args.tile_bytes
        );
    }
    let tiled_cfg = TiledRunConfig {
        threads: cfg.threads,
        num_p_threads: cfg.num_p_threads,
        num_e_threads: cfg.num_e_threads,
        batch_bases: cfg.batch_bases,
        index_cfg: cfg.index,
        pipeline_cfg: cfg.pipeline,
        read_group: cfg.read_group.clone(),
        header: cfg.header.clone(),
        split_prefix,
        junctions: cfg.junctions.clone(),
        junc_bed_tolerance: cfg.junc_bed_tolerance,
    };
    run_tiled(
        reference,
        &args.reads,
        args.output.clone(),
        tiled_cfg,
        tile_plan,
    )
}

pub(crate) fn resolve_accept_enable(explicit: Option<bool>, fast_output: bool) -> bool {
    explicit.unwrap_or(fast_output)
}

/// Window used by mate-guided seed sampling, or `None` for single-end input and
/// when `KIRA_MATE_SEED=0`. `insert_max` is deliberately generous: the window
/// only has to separate "mate is around here" from "mate is elsewhere in the
/// genome".
pub(crate) fn mate_seed_window(paired: &PairedConfig) -> Option<u32> {
    let enabled = std::env::var("KIRA_MATE_SEED")
        .map(|v| v != "0")
        .unwrap_or(true);
    if !enabled || !paired.is_paired() {
        return None;
    }
    Some(paired.insert_max.max(paired.insert_mean.saturating_add(4 * paired.insert_sd)).max(500))
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
#[path = "../../../tests/unit/cli_commands_mem.rs"]
mod tests;
