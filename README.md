# kira-ls-aligner

`kira-ls-aligner` is a unified short- and long-read sequence aligner written in Rust 2024. It combines minimap2-style minimizers and chaining with BWA-MEM2-style exact-match anchoring and output semantics. The goal is drop-in compatibility with bwa-mem pipelines while supporting long reads efficiently.

**We are in progress now. Please don't use this aligner for real tasks!**

## Features

- Multi-resolution minimizer index for short and long reads.
- MEM-like exact anchor extension and minimap2-style chaining.
- Banded Smith-Waterman alignment with affine gaps.
- SAM output compatible with bwa-mem pipelines (flags, MAPQ scale, CIGAR, tags).
- AVX2/NEON runtime detection (scalar fallback).
- Optional CUDA feature gate for future DP offload (future option).
- mmap-based reading for FASTA/FASTQ and index I/O.

## Installation

Install from crates.io (Rust 1.95+ / Windows / Linux / MacOS):

```bash
cargo install kira-ls-aligner
```

Or

Build from source (Rust 1.95+):

```bash
cargo build --release
```

The binary will be at `target/release/kira_ls_aligner` (Windows: `target\release\kira_ls_aligner.exe`).

## Quickstart

```bash
# 1) Build index
kira_ls_aligner index ref.fa

# 2) Align (auto-mode)
kira_ls_aligner mem --index ref.kiraidx ref.fa reads1.fastq -o out.sam

# 3) Enable live stats/progress
KIRA_STATS=1 kira_ls_aligner mem --index ref.kiraidx ref.fa reads.fastq -o out.sam
```

## Usage

Basic alignment:

```bash
kira_ls_aligner mem --index ref.kiraidx ref.fa reads1.fastq -t 10 -K 2000000 -o out.sam
```

Build index:

```bash
kira_ls_aligner index ref.fa -o ref.kiraidx
```

Use a prebuilt index:

```bash
kira_ls_aligner mem --index ref.kiraidx ref.fa reads1.fastq -o out.sam
```

Stats mode with progress bar:

```bash
set KIRA_STATS=1
kira_ls_aligner mem ref.fa reads.fastq -o out.sam

# PowerShell
$env:KIRA_STATS=1
kira_ls_aligner mem ref.fa reads.fastq -o out.sam

# bash
KIRA_STATS=1 kira_ls_aligner mem ref.fa reads.fastq -o out.sam
```

Auto mode selection is the default: the aligner classifies read length and quality on the first batch and chooses short/long/hybrid tuning automatically.

## CLI Options (bwa-mem compatible subset)

- `index REF` : Build a minimizer index.
- `mem REF READS...` : Align reads to reference (one or more FASTQ/FASTA files).
- `--index` : Use a prebuilt index file (REF is kept for bwa-mem compatibility).
- `--fast-output` : Omit MD/XS/XA/SA tags for speed.
- `-t, --threads` : Number of threads.
- `-k, --seed-len` : Seed length (overrides preset).
- `-w, --window-len` : Minimizer window length (overrides preset).
- `-A` : Match score.
- `-B` : Mismatch penalty.
- `-O` : Gap open penalty.
- `-E` : Gap extend penalty.
- `-K, --batch` : Batch size in bases.
- `-x, --preset` : `short`, `long`, or `auto` (default; auto-selects mode at runtime).
- `--long-threshold` : Read length cutoff for long-read settings.
- `-R, --read-group` : Read group line (e.g. `ID:rg1\tSM:sample`).
- `-o, --output` : Output SAM path (stdout if omitted).

## Presets

- `short`: `k=19`, `w=10`, tighter chaining and smaller alignment bands.
- `long`: `k=15`, `w=10`, wider chaining and alignment bands.
- `auto`: default; selects short/long/hybrid per run based on read length distribution, ungapped identity, and chain density.

- `.kiraidx` index files are memory-mapped and used zero-copy at runtime.

## SIMD / CUDA Notes

- SIMD dispatch is runtime-detected (AVX2 on x86_64, NEON on aarch64) with a scalar fallback.
- CUDA is optional (`--features cuda`) and currently exposes a stub for future DP offload.

## Kira LS Aligner vs bwa-mem2 vs minimap2 vs bwa-mem2/mm2-fast

**Goal:** a single drop-in aligner that is fast for both short and long reads while preserving bwa-mem semantics.

- **kira-ls-aligner**
  - One tool for short + long reads with auto mode selection.
  - Minimizer index + RMQ chaining + SIMD banded SW.
  - Aggressive ungapped ACCEPT for high-identity short reads.
  - SAM output aligned with bwa-mem flags/tags/MAPQ scale.

- **bwa-mem2**
  - Strong short-read performance and bwa-mem semantics.
  - FM-index based, optimized for Illumina.
  - Slower than minimap2 on long reads.

- **minimap2**
  - Excellent long-read performance and robustness.
  - Different MAPQ behavior and SAM semantics vs bwa-mem.
  - Often slower than bwa-mem2 on very short Illumina reads.

- **bwa-mem2/mm2-fast**
  - Heuristically faster but can be less stable or less portable.
  - May diverge from canonical bwa-mem/minimap2 behaviors.
  - Typically optimized for a single read regime (short or long).

**When to use kira-ls-aligner:**
- If you want one binary that auto-tunes for both read classes.
- If you need bwa-mem2-compatible SAM semantics but also want minimap2-like speed on long reads.
- If you want deterministic performance without per-dataset flag tuning.

## Test Data / Provenance

The repository contains small and large E. coli datasets for local benchmarking:

- `GCF_000005845.2_ASM584v2_genomic.fna.gz`: NCBI RefSeq E. coli K-12 MG1655 reference (GCF_000005845.2).
- `ecoli.fa`: extracted/normalized FASTA derived from the above RefSeq reference.
- `SRR2584863_1.fastq`, `SRR2584863_2.fastq`: Illumina paired-end reads from NCBI SRA (SRR2584863).
- `ref.fa`, `reads.fq`: tiny toy reference/reads for smoke testing.

Licensing note: NCBI RefSeq and SRA datasets are generally in the public domain in the U.S. (NCBI data usage policies apply). If you redistribute or publish results, please follow NCBI's data usage and citation guidance for RefSeq/SRA.

## Documentation

See `docs/pipeline.md` for detailed pipeline architecture and algorithmic notes.

## FAQ

**Q: Do I need to choose `-x short` or `-x long`?**
A: No. `-x auto` is default and uses read length + quality stats from the first batch. You can still override with `-x short/long` if needed.

**Q: Are `.kiraidx` indexes zero-copy?**
A: Yes. `.kiraidx` is mmap-backed and used zero-copy at runtime.

**Q: Can I pass multiple FASTQ files?**
A: Yes. `mem REF READS...` accepts one or more FASTQ/FASTA files.

**Q: Is output compatible with bwa-mem pipelines?**
A: Yes. SAM flags, MAPQ scale, and tags follow bwa-mem semantics as closely as possible.

**Q: How do I turn on progress + per-stage timing?**
A: Set `KIRA_STATS=1` to enable detailed stats and progress.


## License

MIT