use anyhow::Result;

use crate::aligner_core::{Aligner, AlignerConfig};
use crate::alignment::AlignmentConfig;
use crate::chaining::ChainingConfig;
use crate::index::IndexConfig;
use crate::io::OutputConfig;
use crate::mapq::MapqConfig;
use crate::pipeline::PipelineConfig;
use crate::pipeline::stage1_sketch::SketchConfig;
use crate::seeding::SeedingConfig;

use crate::cli::MemArgs;

pub fn cmd_mem(args: MemArgs) -> Result<()> {
    let preset = args.preset.to_lowercase();
    let short_preset = preset == "short";
    let auto_preset = preset == "auto";
    let (mut short_k, mut short_w, mut long_k, mut long_w) = match preset.as_str() {
        "short" => (19, 10, 19, 10),
        "long" => (15, 10, 15, 10),
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
            },
            2,
        )
    };

    let hybrid_chaining = chaining_cfg;
    let hybrid_align = alignment_cfg;
    let hybrid_dp_topk = args.dp_topk;

    let output_cfg = if args.fast_output {
        OutputConfig::fast()
    } else {
        OutputConfig::full()
    };

    let accept_enable = args.accept_enable.unwrap_or(true);

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

    let cfg = AlignerConfig {
        threads: args.threads,
        batch_bases: args.batch_bases,
        index: index_cfg,
        pipeline: pipeline_cfg,
        auto_profiles,
        read_group: args.read_group,
    };

    let aligner = Aligner::new(cfg);
    if let Some(index_path) = args.index.as_ref() {
        aligner.run_with_index_file(index_path, &args.reads, args.output.as_ref())
    } else {
        aligner.run(&args.reference, &args.reads, args.output.as_ref())
    }
}
