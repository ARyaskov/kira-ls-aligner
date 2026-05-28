//! GPU-server mode.

use std::io::{BufRead, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::cli::MemArgs;

use super::backend::CudaBackend;
use super::dispatcher;
use super::CudaError;

/// Run the interactive GPU server.
pub fn run_gpu_server(
    default_threads: usize,
    default_batch_bases: usize,
) -> Result<(), CudaError> {
    eprintln!("[KIRA_GPU] Initializing CUDA backend...");
    let init_start = std::time::Instant::now();
    {
        let mut backend = CudaBackend::new()?;
        if let Err(e) = warmup_kernel(&mut backend) {
            eprintln!("[KIRA_GPU] main-thread warmup launch failed: {e}");
        }
    } // backend dropped — context handle released for the dispatcher thread
    dispatcher::start()?;
    let warmup = init_start.elapsed();
    eprintln!(
        "[KIRA_GPU] CUDA ready ({}.{:03} s warmup).",
        warmup.as_secs(),
        warmup.subsec_millis()
    );

    eprintln!(
        "[KIRA_GPU] Enter job parameters (one per line: `ref`, `reads`, \
         `index`, `output`, `threads`, `batch`, `read-group`). Blank line \
         submits. `quit` to exit. If `index` is omitted, `<ref>.kiraidx` \
         is auto-detected when present."
    );
    eprintln!("[KIRA_GPU_READY]");

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    let mut session = Session::default();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break, // EOF / read error
        };
        let trimmed = line.trim();

        // Quit token (any case).
        if trimmed.eq_ignore_ascii_case("quit") || trimmed.eq_ignore_ascii_case("exit") {
            eprintln!("[KIRA_GPU] shutting down.");
            break;
        }

        // Blank line → submit accumulated session.
        if trimmed.is_empty() {
            if session.is_empty() {
                // No fields entered yet — keep waiting.
                continue;
            }
            let job_start = std::time::Instant::now();
            let outcome = session.run(default_threads, default_batch_bases);
            let elapsed = job_start.elapsed();
            match outcome {
                Ok(()) => {
                    println!(
                        "[KIRA_GPU_OK] elapsed={}.{:03}s",
                        elapsed.as_secs(),
                        elapsed.subsec_millis()
                    );
                }
                Err(e) => {
                    println!("[KIRA_GPU_ERR] {e:#}");
                }
            }
            let _ = stdout.flush();
            session = Session::default();
            eprintln!("[KIRA_GPU_READY]");
            continue;
        }

        // Parse `key value`.
        let mut parts = trimmed.splitn(2, char::is_whitespace);
        let key = parts.next().unwrap_or("");
        let value = parts.next().unwrap_or("").trim();
        if let Err(e) = session.set(key, value) {
            println!("[KIRA_GPU_ERR] {e:#}");
            let _ = stdout.flush();
        }
    }

    dispatcher::stop();
    Ok(())
}

/// A pending job description accumulated across stdin lines.
#[derive(Default)]
struct Session {
    reference: Option<PathBuf>,
    reads: Vec<PathBuf>,
    index: Option<PathBuf>,
    output: Option<PathBuf>,
    threads: Option<usize>,
    batch_bases: Option<usize>,
    read_group: Option<String>,
}

impl Session {
    fn is_empty(&self) -> bool {
        self.reference.is_none()
            && self.reads.is_empty()
            && self.index.is_none()
            && self.read_group.is_none()
    }

    fn set(&mut self, key: &str, value: &str) -> Result<()> {
        match key {
            "ref" | "reference" => {
                self.reference = Some(PathBuf::from(value));
            }
            "reads" => {
                self.reads.push(PathBuf::from(value));
            }
            "index" | "idx" | "i" => {
                self.index = Some(PathBuf::from(value));
            }
            "output" | "out" | "o" => {
                self.output = Some(PathBuf::from(value));
            }
            "threads" | "t" => {
                self.threads = Some(value.parse().context("threads must be a positive integer")?);
            }
            "batch" | "K" => {
                self.batch_bases =
                    Some(value.parse().context("batch must be a positive integer")?);
            }
            "read-group" | "rg" | "R" => {
                self.read_group = Some(value.to_string());
            }
            other => anyhow::bail!(
                "unknown key `{other}` (use ref / reads / index / output / threads / batch / read-group)"
            ),
        }
        Ok(())
    }

    fn run(
        &mut self,
        default_threads: usize,
        default_batch_bases: usize,
    ) -> Result<()> {
        let reference = self
            .reference
            .clone()
            .context("`ref <path>` is required before submission")?;
        anyhow::ensure!(!self.reads.is_empty(), "`reads <path>` is required");

        let resolved_index = self
            .index
            .clone()
            .or_else(|| auto_detect_sidecar_index(&reference));
        if self.index.is_none() {
            if let Some(idx) = resolved_index.as_ref() {
                eprintln!(
                    "[KIRA_GPU] auto-detected sidecar index {} for {}",
                    idx.display(),
                    reference.display()
                );
            } else {
                eprintln!(
                    "[KIRA_GPU] no sidecar index found for {}; building in-memory \
                     (this may take ~15 min for hg38-sized references). Run \
                     `kira_ls_aligner index <ref>` once to skip this on future jobs.",
                    reference.display()
                );
            }
        }

        let args = MemArgs::server_invocation(
            reference,
            self.reads.clone(),
            resolved_index,
            self.output.clone(),
            self.threads.unwrap_or(default_threads),
            self.batch_bases.unwrap_or(default_batch_bases),
            self.read_group.clone(),
        );

        crate::cli::commands::cmd_mem(args)?;
        Ok(())
    }
}

/// Look for `<ref>.kiraidx` next to the reference FASTA.
fn auto_detect_sidecar_index(reference: &std::path::Path) -> Option<PathBuf> {
    let candidate = reference.with_extension("kiraidx");
    if candidate.is_file() {
        Some(candidate)
    } else {
        None
    }
}

/// Submit a trivial kernel launch to force PTX JIT compilation on first use.
fn warmup_kernel(backend: &mut CudaBackend) -> Result<(), CudaError> {
    use super::backend::CudaJob;
    // One tiny job: 4-bp read matching itself. Verifies kernel launch path.
    let read_packed = vec![0x1Bu8]; // A C G T = 0b00 01 10 11 → 0x1B (low-first)
    let ref_phase = vec![0x1Bu8, 0u8];
    let job = CudaJob {
        read_packed,
        read_nucs: 4,
        ref_shifted: [
            ref_phase.clone(),
            ref_phase.clone(),
            ref_phase.clone(),
            ref_phase,
        ],
        ref_nucs: 4,
        max_mismatches: 0,
    };
    let _ = backend.run_batch(std::slice::from_ref(&job))?;
    Ok(())
}
