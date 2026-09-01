use std::path::PathBuf;

use clap::Parser;

/// Evaluate placement accuracy of a SAM file against truth-in-name read ids.
///
/// For simulated reads whose QNAME encodes the source locus
/// (`<name>:<contig>:<start>-<end>` or `<name>_<contig>_<start>_<end>`),
/// reports unmapped / correct / wrong-locus counts, stratified by MAPQ
/// threshold and by INDEL-bearing CIGARs. This is the fast regression gate for
/// accuracy work: it attributes every lost read to a pipeline stage class
/// (seeding → unmapped, placement → wrong_locus, MAPQ → below-threshold)
/// without a variant-calling round-trip.
#[derive(Parser, Debug)]
pub struct EvalArgs {
    /// SAM file to evaluate ('-' for stdin).
    #[arg(value_name = "SAM")]
    pub sam: PathBuf,

    /// Positional tolerance in bp for locus concordance.
    #[arg(long = "tolerance", default_value_t = 150)]
    pub tolerance: i64,

    /// Comma-separated MAPQ thresholds for stratified counts
    /// (simulates the caller's MAPQ filter).
    #[arg(long = "mapq-thresholds", default_value = "13,30,60")]
    pub mapq_thresholds: String,

    /// Write per-read attribution rows (TSV) to this path.
    #[arg(long = "dump-attribution", value_name = "PATH")]
    pub dump_attribution: Option<PathBuf>,
}
