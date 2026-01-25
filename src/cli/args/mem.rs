use std::path::PathBuf;

use clap::Parser;

/// Align reads to a reference using bwa-mem compatible options.
#[derive(Parser, Debug)]
pub struct MemArgs {
    /// Reference FASTA.
    #[arg(value_name = "REF")]
    pub reference: PathBuf,

    /// Reads FASTQ/FASTA (one or more files).
    #[arg(value_name = "READS", required = true, num_args = 1..)]
    pub reads: Vec<PathBuf>,

    /// Output SAM path (stdout if omitted).
    #[arg(short = 'o', long = "output")]
    pub output: Option<PathBuf>,

    /// Use a prebuilt index file (.kiraidx).
    #[arg(long = "index")]
    pub index: Option<PathBuf>,

    /// Faster output (omit MD/XS/XA/SA tags).
    #[arg(long = "fast-output")]
    pub fast_output: bool,

    /// Max DP alignments per read.
    #[arg(long = "max-alignments", default_value_t = 1)]
    pub max_alignments: usize,

    /// Minimum chain score ratio vs best (skip DP for lower chains).
    #[arg(long = "min-chain-ratio", default_value_t = 0.4)]
    pub min_chain_ratio: f32,

    /// Limit DP to top-K chains per read.
    #[arg(long = "dp-topk", default_value_t = 1)]
    pub dp_topk: usize,

    /// Enable ungapped ACCEPT prefilter (default: on with --fast-output).
    #[arg(long = "accept-enable", value_parser = clap::builder::BoolishValueParser::new())]
    pub accept_enable: Option<bool>,

    /// Accept span slack (read_len - slack).
    #[arg(long = "accept-span-slack", default_value_t = 15)]
    pub accept_span_slack: usize,

    /// Accept minimum identity in percent.
    #[arg(
        long = "accept-min-id",
        default_value_t = 98.5,
        alias = "accept-min-identity"
    )]
    pub accept_min_identity: f32,

    /// Accept maximum mismatches in ungapped span.
    #[arg(
        long = "accept-max-mism",
        default_value_t = 5,
        alias = "accept-max-mismatches"
    )]
    pub accept_max_mismatches: usize,

    /// Accept only top-1 chain.
    #[arg(long = "accept-only-top1", default_value_t = true, value_parser = clap::builder::BoolishValueParser::new())]
    pub accept_only_top1: bool,

    /// Require score margin over second-best chain (0 disables).
    #[arg(long = "accept-require-score-margin", default_value_t = 0)]
    pub accept_require_score_margin: i32,

    /// Debug: log prefilter decisions for first N reads per batch.
    #[arg(long = "debug-prefilter", default_value_t = 0, hide = true)]
    pub debug_prefilter_n: usize,

    /// Debug: force ACCEPT for top-1 chains on first N reads.
    #[arg(long = "debug-force-accept", hide = true)]
    pub debug_force_accept: bool,

    /// Debug: number of reads to force ACCEPT (default 100).
    #[arg(long = "debug-force-accept-n", default_value_t = 100, hide = true)]
    pub debug_force_accept_n: usize,

    /// Number of threads.
    #[arg(short = 't', long = "threads", default_value_t = 8)]
    pub threads: usize,

    /// Batch size in bases.
    #[arg(short = 'K', long = "batch", default_value_t = 1_000_000)]
    pub batch_bases: usize,

    /// Preset: short, long, or auto.
    #[arg(short = 'x', long = "preset", default_value = "auto")]
    pub preset: String,

    /// Seed length (overrides preset for both short and long indices).
    #[arg(short = 'k', long = "seed-len")]
    pub seed_len: Option<usize>,

    /// Minimizer window size (overrides preset for both short and long indices).
    #[arg(short = 'w', long = "window-len")]
    pub window_len: Option<usize>,

    /// Long-read threshold (bp).
    #[arg(long = "long-threshold", default_value_t = 500)]
    pub long_read_threshold: usize,

    /// Match score.
    #[arg(short = 'A', default_value_t = 1)]
    pub match_score: i32,

    /// Mismatch penalty.
    #[arg(short = 'B', default_value_t = 4)]
    pub mismatch_penalty: i32,

    /// Gap open penalty.
    #[arg(short = 'O', default_value_t = 6)]
    pub gap_open: i32,

    /// Gap extend penalty.
    #[arg(short = 'E', default_value_t = 1)]
    pub gap_extend: i32,

    /// Read group header line (RG:Z:...).
    #[arg(short = 'R', long = "read-group")]
    pub read_group: Option<String>,
}
