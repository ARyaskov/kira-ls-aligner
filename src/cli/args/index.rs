use std::path::PathBuf;

use clap::Parser;

/// Build a minimizer index for a reference.
#[derive(Parser, Debug)]
pub struct IndexArgs {
    /// Reference FASTA.
    #[arg(value_name = "REF")]
    pub reference: PathBuf,

    /// Output index path (default: REF.kiraidx).
    #[arg(short = 'o', long = "output")]
    pub output: Option<PathBuf>,

    /// Preset: short, long, or auto.
    #[arg(short = 'x', long = "preset", default_value = "auto")]
    pub preset: String,

    /// Seed length (overrides preset for both short and long indices).
    #[arg(short = 'k', long = "seed-len")]
    pub seed_len: Option<usize>,

    /// Minimizer window size (overrides preset for both short and long indices).
    #[arg(short = 'w', long = "window-len")]
    pub window_len: Option<usize>,

    /// Maximum occurrences recorded per minimizer.
    #[arg(long = "max-occ", default_value_t = 500)]
    pub max_occ: usize,
}
