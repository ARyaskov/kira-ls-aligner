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

    /// Soft-clip penalty (bwa-mem `-L`).
    #[arg(short = 'L', long = "clip-penalty", default_value_t = 5)]
    pub clip_penalty: i32,

    /// Read group header line (RG:Z:...).
    #[arg(short = 'R', long = "read-group")]
    pub read_group: Option<String>,

    /// Use the CUDA backend for the alignment fast path.
    #[arg(long = "gpu", default_value_t = false)]
    pub gpu: bool,

    /// Treat input reads as paired-end.
    #[arg(short = 'p', long = "paired", default_value_t = false)]
    pub paired: bool,

    /// Single-file paired mode: the one input FASTQ contains alternating R1/R2 records.
    #[arg(long = "interleaved", default_value_t = false)]
    pub interleaved: bool,

    /// Insert-size constraints for proper-pair classification.
    #[arg(short = 'I', long = "insert-size", default_value = "0,1000,200,50")]
    pub insert_size: String,

    /// Maximum intron length for splice-aware alignment.
    #[arg(long = "max-intron", default_value_t = 200_000)]
    pub max_intron: u32,

    /// Minimum intron length.
    #[arg(long = "min-intron", default_value_t = 30)]
    pub min_intron: u32,

    /// Transcript strand policy for splice alignments: `auto`.
    #[arg(long = "splice-strand", default_value = "auto")]
    pub splice_strand: String,

    /// BED12/BED6 file of known splice junctions (e.g. GENCODE annotations).
    #[arg(long = "junc-bed", value_name = "PATH")]
    pub junc_bed: Option<PathBuf>,

    /// Position tolerance (bases) for matching anchor boundaries to junctions in `--junc-bed`.
    #[arg(long = "junc-bed-tolerance", default_value_t = 2)]
    pub junc_bed_tolerance: u32,

    /// Half-width of the donor/acceptor refinement window at each anchor gap (splice modes).
    #[arg(long = "splice-flank", default_value_t = 20)]
    pub splice_flank: u32,

    /// Minimum size of a recoverable short exon for the in-intron search (splice modes).
    #[arg(long = "min-exon", default_value_t = 15)]
    pub min_exon: u32,

    /// Soft-clip trailing polyA / leading polyT runs ≥ this many bases (splice modes).
    #[arg(long = "polya-min-len", default_value_t = 10)]
    pub polya_min_len: u32,

    /// Inject one or more `@PG` records into the SAM header.
    #[arg(long = "pg", value_name = "PG_LINE")]
    pub pg: Vec<String>,

    /// Inject one or more `@CO` (comment) lines into the SAM header.
    #[arg(long = "co", value_name = "COMMENT")]
    pub co: Vec<String>,

    /// Skip the auto-generated `@PG ID:kira-ls-aligner` record.
    #[arg(long = "no-PG", default_value_t = false)]
    pub no_pg: bool,

    /// Enable split-prefix tiled alignment for references that don't fit in RAM.
    #[arg(long = "split-prefix", value_name = "PREFIX")]
    pub split_prefix: Option<PathBuf>,

    /// Target size (in bytes) per tile when `--split-prefix` is set.
    #[arg(
        long = "tile-bytes",
        value_name = "BYTES",
        default_value_t = 4_000_000_000
    )]
    pub tile_bytes: u64,

    /// Output format: `sam`, `bam`, or `sorted-bam` (BAM + coordinate sort).
    #[arg(long = "emit", default_value = "sam")]
    pub emit: String,

    /// Mark PCR/optical duplicates (requires `--emit sorted-bam`).
    #[arg(long = "markdup", default_value_t = false)]
    pub markdup: bool,

    /// Build a BAI index next to the output BAM.
    #[arg(long = "bai", default_value_t = false)]
    pub bai: bool,
}

impl MemArgs {
    /// Build a synthetic `MemArgs` from the GPU server's parsed parameters.
    pub fn server_invocation(
        reference: std::path::PathBuf,
        reads: Vec<std::path::PathBuf>,
        index: Option<std::path::PathBuf>,
        output: Option<std::path::PathBuf>,
        threads: usize,
        batch_bases: usize,
        read_group: Option<String>,
    ) -> Self {
        Self {
            reference,
            reads,
            output,
            index,
            fast_output: false,
            max_alignments: 1,
            min_chain_ratio: 0.4,
            dp_topk: 1,
            accept_enable: None,
            accept_span_slack: 15,
            accept_min_identity: 98.5,
            accept_max_mismatches: 5,
            accept_only_top1: true,
            accept_require_score_margin: 0,
            debug_prefilter_n: 0,
            debug_force_accept: false,
            debug_force_accept_n: 100,
            threads,
            batch_bases,
            preset: "auto".to_string(),
            seed_len: None,
            window_len: None,
            long_read_threshold: 500,
            match_score: 1,
            mismatch_penalty: 4,
            gap_open: 6,
            gap_extend: 1,
            clip_penalty: 5,
            read_group,
            gpu: true,
            paired: false,
            interleaved: false,
            insert_size: "0,1000,200,50".to_string(),
            max_intron: 200_000,
            min_intron: 30,
            splice_strand: "auto".to_string(),
            junc_bed: None,
            junc_bed_tolerance: 2,
            splice_flank: 20,
            min_exon: 15,
            polya_min_len: 10,
            pg: Vec::new(),
            co: Vec::new(),
            no_pg: false,
            split_prefix: None,
            tile_bytes: 4_000_000_000,
            emit: "sam".to_string(),
            markdup: false,
            bai: false,
        }
    }
}
