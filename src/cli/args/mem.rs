use std::path::PathBuf;

use clap::Parser;

/// Align reads to a reference using bwa-mem compatible options.
///
/// `-h` is bwa-mem's XA cap, so clap's automatic `-h` help flag is off and
/// `--help` is declared explicitly (the same arrangement kira-bam uses for
/// `view -h`).
#[derive(Parser, Debug)]
#[command(disable_help_flag = true)]
pub struct MemArgs {
    /// Print help.
    #[arg(long = "help", action = clap::ArgAction::Help)]
    pub help: Option<bool>,

    /// Reference FASTA.
    #[arg(value_name = "REF")]
    pub reference: PathBuf,

    /// Reads FASTQ/FASTA (one or more files; `-` reads stdin, plain or gzip).
    #[arg(value_name = "READS", required = true, num_args = 1..)]
    pub reads: Vec<PathBuf>,

    /// Output SAM path (stdout if omitted).
    #[arg(short = 'o', long = "output")]
    pub output: Option<PathBuf>,

    /// Verbosity: 1 = errors, 2 = warnings, 3 = messages (default), 4 = debug.
    #[arg(short = 'v', long = "verbosity", default_value_t = 3, value_parser = clap::value_parser!(u8).range(1..=4))]
    pub verbosity: u8,

    /// Config file of `KIRA_KNOB=value` lines (`#` comments allowed). Applied
    /// before any tuning knob is read, and recorded in the `@CO kira-env`
    /// header line so the run can be reproduced from its BAM.
    #[arg(long = "config", value_name = "FILE")]
    pub config: Option<PathBuf>,

    /// Set one tuning knob, `KIRA_KNOB=value` (repeatable; wins over --config).
    #[arg(long = "set", value_name = "KEY=VALUE")]
    pub set: Vec<String>,

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

    /// Number of threads. On hybrid CPUs (Alder Lake+) this acts as an
    /// upper bound when `--num-p-threads` / `--num-e-threads` are not set
    /// explicitly; the auto-detected P/E split decides how it's spent.
    #[arg(short = 't', long = "threads", default_value_t = 8)]
    pub threads: usize,

    /// Override the number of workers pinned to Performance cores
    /// (SIMD-heavy stages). On homogeneous hosts this is ignored.
    #[arg(long = "num-p-threads", value_name = "N")]
    pub num_p_threads: Option<usize>,

    /// Override the number of workers pinned to Efficient cores
    /// (light bookkeeping + SAM emit). On homogeneous hosts this is
    /// ignored.
    #[arg(long = "num-e-threads", value_name = "N")]
    pub num_e_threads: Option<usize>,

    /// Batch size in bases.
    #[arg(short = 'K', long = "batch", default_value_t = 4_000_000)]
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

    /// Maximum reference occurrences retained per read minimizer.
    #[arg(long = "seed-occ-cap", default_value_t = 16, value_parser = clap::value_parser!(u32).range(1..))]
    pub seed_occ_cap: u32,

    /// Long-read threshold (bp).
    #[arg(long = "long-threshold", default_value_t = 500)]
    pub long_read_threshold: usize,

    /// Match score.
    #[arg(short = 'A', default_value_t = 1)]
    pub match_score: i32,

    /// Mismatch penalty.
    #[arg(short = 'B', default_value_t = 4)]
    pub mismatch_penalty: i32,

    /// Gap open penalty. bwa-mem's `INT[,INT]` (deletion,insertion) form is
    /// accepted; one affine gap model is used, so the first value applies.
    #[arg(short = 'O', default_value = "6", value_parser = parse_bwa_int_pair)]
    pub gap_open: i32,

    /// Gap extend penalty (`INT[,INT]` accepted, first value applies).
    #[arg(short = 'E', default_value = "1", value_parser = parse_bwa_int_pair)]
    pub gap_extend: i32,

    /// Soft-clip penalty (bwa-mem `-L`).
    #[arg(short = 'L', long = "clip-penalty", default_value = "5", value_parser = parse_bwa_int_pair)]
    pub clip_penalty: i32,

    /// Z-dropoff for DP extension (bwa-mem `-d`).
    #[arg(short = 'd', long = "xdrop", default_value_t = 100)]
    pub xdrop: i32,

    /// Minimum alignment score to output (bwa-mem `-T`).
    #[arg(short = 'T', long = "min-score", default_value_t = 30)]
    pub min_score: i32,

    /// Output all found alignments, not just the primary (bwa-mem `-a`).
    #[arg(short = 'a', long = "all")]
    pub output_all: bool,

    /// Emit `XA` only when the read has at most INT close secondary hits
    /// (bwa-mem `-h INT[,INT]`; the second value is accepted and ignored).
    #[arg(short = 'h', long = "xa-max", default_value = "5,200", value_parser = parse_bwa_xa_max)]
    pub xa_max: u32,

    /// Append the FASTQ comment to every SAM record (bwa-mem `-C`), so
    /// `BC:Z:`/`RX:Z:` tags written by the demultiplexer survive alignment.
    #[arg(short = 'C', long = "append-comment")]
    pub append_comment: bool,

    /// Insert header lines (bwa-mem `-H`): a string starting with `@`, or a
    /// file of such lines. Repeatable.
    #[arg(short = 'H', long = "header-insert", value_name = "STR|FILE")]
    pub header_insert: Vec<String>,

    /// Mark supplementary segments as secondary (0x100) for Picard-era tools (bwa-mem `-M`).
    #[arg(short = 'M', long = "mark-split-secondary")]
    pub mark_split_secondary: bool,

    /// Soft-clip supplementary segments instead of hard-clipping them (bwa-mem `-Y`).
    #[arg(short = 'Y', long = "soft-clip-supplementary")]
    pub soft_clip_supplementary: bool,

    /// For a split read, make the segment with the smallest read coordinate primary (bwa-mem `-5`).
    #[arg(short = '5', long = "primary-5p")]
    pub primary_5p: bool,

    /// Skip mate rescue (bwa-mem `-S`).
    #[arg(short = 'S', long = "skip-mate-rescue")]
    pub skip_mate_rescue: bool,

    /// Skip pairing; mate rescue still runs unless `-S` (bwa-mem `-P`).
    #[arg(short = 'P', long = "skip-pairing")]
    pub skip_pairing: bool,

    /// Treat ALT contigs as part of the primary assembly (bwa-mem `-j`):
    /// do not read `REF.alt`.
    #[arg(short = 'j', long = "ignore-alt")]
    pub ignore_alt: bool,

    /// Read group header line (RG:Z:...).
    #[arg(short = 'R', long = "read-group")]
    pub read_group: Option<String>,

    /// Use the CUDA backend for the alignment fast path.
    #[arg(long = "gpu", default_value_t = false)]
    pub gpu: bool,

    /// Two-file paired-end mode (R1 file, R2 file). Auto-detected from two
    /// inputs, as bwa-mem does; pass it to silence the notice.
    #[arg(long = "paired", default_value_t = false)]
    pub paired: bool,

    /// Smart pairing (bwa-mem `-p`): the single input FASTQ is interleaved,
    /// consecutive records being R1/R2 of one template.
    #[arg(short = 'p', long = "interleaved", alias = "smart-pairing", default_value_t = false)]
    pub interleaved: bool,

    // ── bwa-mem flags accepted for wrapper compatibility, without effect ──
    // Each is a bwa-mem heuristic with no counterpart in this pipeline. They
    // parse so an unmodified `bwa mem` command line runs; a `-v 2` warning
    // lists the ones that were given.
    /// bwa-mem `-r`: re-seed factor. No effect.
    #[arg(short = 'r', hide = true)]
    pub compat_reseed: Option<f32>,
    /// bwa-mem `-y`: seed occurrence for the 3rd round seeding. No effect.
    #[arg(short = 'y', hide = true)]
    pub compat_seed_occ3: Option<u32>,
    /// bwa-mem `-D`: chain drop ratio. No effect.
    #[arg(short = 'D', hide = true)]
    pub compat_chain_drop: Option<f32>,
    /// bwa-mem `-W`: minimum chain weight. No effect.
    #[arg(short = 'W', hide = true)]
    pub compat_min_chain_weight: Option<u32>,
    /// bwa-mem `-m`: maximum mate-rescue rounds. No effect.
    #[arg(short = 'm', hide = true)]
    pub compat_max_rescue_rounds: Option<u32>,
    /// bwa-mem `-U`: unpaired read pair penalty. No effect.
    #[arg(short = 'U', hide = true)]
    pub compat_unpaired_penalty: Option<i32>,
    /// bwa-mem `-e`: discard full-length exact matches. No effect.
    #[arg(short = 'e', hide = true)]
    pub compat_discard_exact: bool,
    /// bwa-mem `-c`: discard a MEM above this many occurrences. Mapped onto
    /// `--seed-occ-cap` semantics would change results; accepted, no effect.
    #[arg(short = 'c', hide = true)]
    pub compat_max_occ: Option<u32>,

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

    /// Output format: `sam`, `paf`, `bam`, `sorted-bam`, `cram`, `sorted-cram`.
    /// The binary formats stream from the aligner into kira-bam in memory —
    /// no intermediate SAM. CRAM uses `REF` (its `.fai` is built if missing).
    #[arg(long = "emit", default_value = "sam")]
    pub emit: String,

    /// Mark PCR/optical duplicates (requires `--emit sorted-bam|sorted-cram`).
    #[arg(long = "markdup", default_value_t = false)]
    pub markdup: bool,

    /// Build a BAI index next to the output BAM.
    #[arg(long = "bai", default_value_t = false)]
    pub bai: bool,

    /// Memory budget for the in-memory coordinate sort of `--emit sorted-*`
    /// (`4G`, `768M`, a byte count, or `auto` = a quarter of RAM). A run that
    /// outgrows it spills to an unsorted BAM and finishes with the external
    /// sort, which also disables the fused markdup.
    #[arg(long = "sort-memory", default_value = "auto", value_name = "SIZE")]
    pub sort_memory: String,
}

/// bwa-mem accepts `INT[,INT]` for `-O`/`-E`/`-L` (deletion,insertion or
/// 5',3'). One value drives this aligner; the first is taken.
fn parse_bwa_int_pair(s: &str) -> Result<i32, String> {
    let first = s.split(',').next().unwrap_or("").trim();
    first
        .parse::<i32>()
        .map_err(|_| format!("expected INT or INT,INT, got {s:?}"))
}

/// bwa-mem `-h INT[,INT]`: the first value caps XA hits; the second (ALT
/// hits) is accepted and ignored.
fn parse_bwa_xa_max(s: &str) -> Result<u32, String> {
    let first = s.split(',').next().unwrap_or("").trim();
    first
        .parse::<u32>()
        .map_err(|_| format!("expected INT or INT,INT, got {s:?}"))
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
            help: None,
            reference,
            reads,
            output,
            verbosity: 3,
            config: None,
            set: Vec::new(),
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
            num_p_threads: None,
            num_e_threads: None,
            batch_bases,
            preset: "auto".to_string(),
            seed_len: None,
            window_len: None,
            seed_occ_cap: 16,
            long_read_threshold: 500,
            match_score: 1,
            mismatch_penalty: 4,
            gap_open: 6,
            gap_extend: 1,
            clip_penalty: 5,
            xdrop: 100,
            min_score: 30,
            output_all: false,
            xa_max: 5,
            append_comment: false,
            header_insert: Vec::new(),
            mark_split_secondary: false,
            soft_clip_supplementary: false,
            primary_5p: false,
            skip_mate_rescue: false,
            skip_pairing: false,
            ignore_alt: false,
            read_group,
            gpu: true,
            paired: false,
            interleaved: false,
            compat_reseed: None,
            compat_seed_occ3: None,
            compat_chain_drop: None,
            compat_min_chain_weight: None,
            compat_max_rescue_rounds: None,
            compat_unpaired_penalty: None,
            compat_discard_exact: false,
            compat_max_occ: None,
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
            sort_memory: "auto".to_string(),
        }
    }

    /// The bwa-mem compatibility flags that were given but have no effect
    /// here, for a one-line warning.
    pub fn ignored_compat_flags(&self) -> Vec<&'static str> {
        let mut v = Vec::new();
        if self.compat_reseed.is_some() {
            v.push("-r");
        }
        if self.compat_seed_occ3.is_some() {
            v.push("-y");
        }
        if self.compat_chain_drop.is_some() {
            v.push("-D");
        }
        if self.compat_min_chain_weight.is_some() {
            v.push("-W");
        }
        if self.compat_max_rescue_rounds.is_some() {
            v.push("-m");
        }
        if self.compat_unpaired_penalty.is_some() {
            v.push("-U");
        }
        if self.compat_discard_exact {
            v.push("-e");
        }
        if self.compat_max_occ.is_some() {
            v.push("-c");
        }
        v
    }
}
