use anyhow::Result;
use clap::{Parser, Subcommand};
#[cfg(feature = "cuda")]
use anyhow::Context;

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
    /// Run a long-lived GPU server with a warm CUDA context.
    GpuServer(GpuServerArgs),
}

#[derive(Parser, Debug)]
pub struct GpuServerArgs {
    /// Thread pool size for the CPU stages (sketch / seeding / output).
    #[arg(short = 't', long = "threads", default_value_t = 8)]
    pub threads: usize,
    /// Batch size in bases for FASTQ streaming.
    #[arg(short = 'K', long = "batch", default_value_t = 10_000_000)]
    pub batch_bases: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Index(args) => cmd_index(args),
        Commands::Mem(args) => cmd_mem(args),
        Commands::GpuServer(args) => run_gpu_server(args),
    }
}

#[cfg(feature = "cuda")]
fn run_gpu_server(args: GpuServerArgs) -> Result<()> {
    kira_ls_aligner::cuda::run_gpu_server(args.threads, args.batch_bases)
        .context("GPU server failed")
}

#[cfg(not(feature = "cuda"))]
fn run_gpu_server(_args: GpuServerArgs) -> Result<()> {
    anyhow::bail!(
        "this binary was built without CUDA support — rebuild with `--features cuda` \
         and install the CUDA toolkit (>= 11.0) to use --gpu-server"
    )
}
