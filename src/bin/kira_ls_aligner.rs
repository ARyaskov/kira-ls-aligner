use anyhow::Result;
use clap::{Parser, Subcommand};

use kira_ls_aligner::cli::{IndexArgs, MemArgs, cmd_index, cmd_mem};

#[derive(Parser)]
#[command(name = "kira-ls-aligner")]
#[command(about = "Unified short/long read aligner", version, author)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build a minimizer index
    Index(IndexArgs),
    /// Align reads (bwa-mem compatible)
    Mem(MemArgs),
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Index(args) => cmd_index(args),
        Commands::Mem(args) => cmd_mem(args),
    }
}
