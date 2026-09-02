# kira-ls-aligner

`kira-ls-aligner` is a unified short- and long-read sequence aligner written in Rust 2024. It combines minimap2-style minimizers and chaining with BWA-MEM2-style exact-match anchoring and output semantics. The goal is drop-in compatibility with bwa-mem pipelines while supporting long reads efficiently.

## Features

- Multi-resolution minimizer index for short and long reads.
- MEM-like exact anchor extension and minimap2-style chaining.
- Banded Smith-Waterman alignment with affine gaps.
- SAM output compatible with bwa-mem pipelines (flags, MAPQ scale, CIGAR, tags).
- AVX2/NEON runtime detection (scalar fallback).
- Optional CUDA Spectral Sieve backend for batched short-read prefiltering.
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

### Short-read accuracy profiles

Full SAM output defaults to the accuracy path and disables the ungapped ACCEPT
shortcut. `--fast-output` enables ACCEPT by default; either behavior can be
overridden explicitly with `--accept-enable true|false`.

The recommended production accuracy profile is therefore the normal full-SAM
command with the default occurrence cap:

```powershell
kira_ls_aligner mem --index ref.kiraidx ref.fa reads.fastq `
  --accept-enable false -o out.sam
```

On the tested 4M-read hg38 dataset, disabling ACCEPT improved exact primary
locus concordance with minimap2 by 0.49 percentage points and MAPQ-60
concordance by 0.51 points. It cost 12.5% wall time and 2.9% CPU versus an
ACCEPT-on control using the same binary. Use `--accept-enable true` when
throughput is more important than full-SAM locus fidelity.

An experimental maximum-accuracy profile is available:

```powershell
$env:KIRA_SHORT_DPTOPK = "3"
kira_ls_aligner mem --index ref.kiraidx ref.fa reads.fastq `
  --seed-occ-cap 32 --min-chain-ratio 0.2 --accept-enable false -o out.sam
```

It reduced unmapped reads by 6.1% and improved exact-locus concordance by 0.69
points, but used 24.7 GiB peak working set and took 2.07x the wall time of the
ACCEPT-on control. It is not the recommended default. These are concordance
results, not truth-set SNP/INDEL F1 measurements.

## Evaluation tooling (`eval`)

For simulated reads whose FASTQ id encodes the source locus
(`<name>:<contig>:<start>-<end>` or `<name>_<contig>_<start>_<end>`), the
binary scores a SAM/stdout stream and attributes every read:

```bash
kira_ls_aligner eval out.sam --tolerance 150 --mapq-thresholds 13,30,60 \
  --dump-attribution per_read.tsv
```

Reports unmapped / correct-locus / wrong-locus counts, with INDEL-bearing
reads broken out and per-MAPQ-threshold counts (what a caller's `MQ ≥ t`
filter trades off). Use it to attribute recall loss to a stage class
(seeding → unmapped, placement → wrong_locus, MAPQ → below threshold) instead
of a variant-calling round-trip. `x2 / x2` comparison runs of the regression
set (alternating arms, minimum per stage) use this plus `KIRA_STATS=1`.

```text
[EVAL] total_records=... truth_parsed=... no_truth=...
[EVAL] unmapped=... (..%)
[EVAL] mapped=... correct=... (..%) wrong_locus=... (..%)
[EVAL] indel_bearing: correct=... wrong_locus=...
[EVAL] mapq>=13: correct=... wrong=... precision=..% recall_of_correct=..%
```

## CLI Options (bwa-mem compatible)

The `mem` command line is a drop-in for `bwa mem`: the flags below carry
bwa-mem's letters and meanings, an unmodified `bwa mem` invocation parses, and
reads may come from stdin (`-`, plain or gzip). Flags bwa-mem has for
heuristics this pipeline does not run (`-r -y -D -W -m -U -e -c -j`) are
accepted without effect and listed once on stderr.

- `index REF` : Build a minimizer index.
- `mem REF READS...` : Align reads (one or more FASTQ/FASTA files, or `-`).
- `eval SAM` : Evaluate placement accuracy against truth-in-name read ids (simulation regression tool).
- `--index` : Use a prebuilt index file (REF is kept for bwa-mem compatibility).
- `-o, --output` : Output path (stdout if omitted).
- `--emit` : `sam` (default), `paf`, `bam`, `sorted-bam`, `cram`, `sorted-cram`. See [Output formats](#output-formats).
- `-t, --threads` : Number of threads.
- `-K, --batch` : Batch size in bases.
- `-p` : Smart pairing — the single input is interleaved R1/R2 (bwa-mem `-p`). Two inputs are R1/R2 files automatically; `--paired` states it explicitly.
- `-R, --read-group` : Read group line (e.g. `ID:rg1\tSM:sample`).
- `-H STR|FILE` : Insert header lines (a `@…` string or a file of them).
- `-C` : Append the FASTQ comment to each record (keeps `BC:Z:`/`RX:Z:` from the demultiplexer).
- `-v INT` : Verbosity, 1 errors … 3 messages (default) … 4 debug.
- `-T INT` : Minimum alignment score to output (30).
- `-a` : Output all found alignments (secondaries included).
- `-h INT[,INT]` : Emit `XA` only when the read has at most INT close secondary hits (5). `XA` needs secondaries to exist, i.e. `--dp-topk 2` or `-a`.
- `-M` : Flag supplementary segments as secondary (0x100) for Picard-era tools.
- `-Y` : Soft-clip supplementary segments; the default hard-clips them, as bwa-mem does.
- `-5` : For a split read, the segment with the smallest read coordinate is primary.
- `-S` / `-P` : Skip mate rescue / skip pairing.
- `-A -B -O -E -L -d` : Match, mismatch, gap open, gap extend, clip penalty, z-dropoff. `-O`/`-E`/`-L` accept bwa's `INT,INT` form (first value applies).
- `-k, --seed-len` / `-w, --window-len` : Seed length and minimizer window (note: bwa-mem's `-w` is a band width).
- `-x, --preset` : `short`, `long`, or `auto` (default; auto-selects mode at runtime).
- `--fast-output` : Omit MD/XS/XA/SA tags for speed.
- `--accept-enable` : Override the ungapped ACCEPT shortcut.
- `--seed-occ-cap` : Maximum reference occurrences retained per read minimizer.
- `--min-chain-ratio` : Keep chains within this score ratio of the best chain.
- `--long-threshold` : Read length cutoff for long-read settings.
- `--config FILE` / `--set KIRA_KNOB=value` : Tuning knobs (below) from a file or the command line, recorded in the header.

Every record of a paired run carries `MC:Z` / `MQ:i` / `ms:i` (the mate's
CIGAR, MAPQ and score — what `samtools fixmate -m` adds), so
`samtools markdup` runs directly on the output.

### Output formats

`--emit bam|sorted-bam|cram|sorted-cram` streams alignment batches straight
into the embedded kira-bam writer — no intermediate SAM file. `sorted-*` sorts
in memory within `--sort-memory` (default a quarter of RAM) and fuses
`--markdup` into that pass; a run that outgrows the budget spills to an
unsorted BAM beside the output and finishes with kira-bam's external sort.
`--bai` indexes BAM output. CRAM uses `REF` and builds `REF.fai` if missing.

```bash
kira_ls_aligner mem ref.fa R1.fq.gz R2.fq.gz -t 16 --emit sorted-bam --markdup --bai -o out.bam
```

### ALT contigs (GRCh38 full analysis set)

If `REF.alt` exists next to the reference (the bwa-mem convention; shipped
with `GRCh38_full_analysis_set_plus_decoy_hla.fa`), its contigs are ALT:

- a primary-assembly placement is not made ambiguous by its ALT copies —
  competitors on ALT contigs are left out of the MAPQ model unless the read's
  own best hit is on an ALT contig;
- a read whose best hit is on an ALT contig but which has a primary-assembly
  hit within one mismatch is reported on the primary assembly (`bwa-postalt`).

`-j` ignores the file. On a synthetic ALT copy at 0.5 % divergence this takes
primary-assembly reads from 0 to 96.9 % at MAPQ 60.

### Provenance

Unless `--no-PG`, the header records what the `@PG CL` cannot: every `KIRA_*`
knob in force (`@CO kira-env:…`) and the effective pipeline parameters
(`@CO kira-config:…`). A run is reproducible from its BAM header alone.

### Tuning knobs (environment)

Defaults are what the numbers below were measured with; each knob exists so the
default can be A/B'd without a rebuild. Set them in the environment, with
`--set`, or in a `--config` file of `KIRA_KNOB=value` lines.

| Variable | Default | Effect |
|---|---|---|
| `KIRA_LEFT_NORM` | on | Left-normalize indel placement in emitted CIGARs (GATK `LeftAlignIndels` / `bcftools norm` convention) so equivalent gap placements emit identical CIGARs. `0`/`off` emits traceback-native CIGARs. On truth-in-name simulations this took canonical indel CIGARs from 55 % to 92 % of indel-bearing reads. |
| `KIRA_MAPQ_MULT` | 0 (off) | Multiplicity penalty slope on competing loci: MAPQ drops by `γ·ln(n)` when `n` above-floor competitors exist (bwa-mem `mem_approx_mapq_se`, γ = 6.585). Off because on the simulations every read it demoted was correctly placed — wrong-locus reads already sit below MAPQ 13 through the score-gap term. |
| `KIRA_MAPQ_BETA` | 22.5 | Slope of the MAPQ score-gap posterior model. Swept 10–60 on a truth-in-name PE simulation: the MAPQ ≥ 13 trade-off is flat above 15 and MAPQ ≥ 30 carries a 0.1 % empirical mismap rate at 22.5, i.e. the default is calibrated. |
| `KIRA_PAIR_BONUS_ALL` | off | Give the pair-concordance bonus to every alignment that forms a concordant pair, not only the winning pair, so a read whose mates are concordant at two loci scores as a tie (MAPQ 0) instead of MAPQ 20 on the tie-break. Off because on GIAB chr20 that tie-break is right ~72 % of the time and at the caller's MAPQ ≥ 13 floor keeping those reads is +61 true SNPs for +30 false (F1 0.9789 vs 0.9787); `1` for precision-first pipelines. |
| `KIRA_AMBIG_DIV` | 5 | A runner-up chain within `best / DIV` of the best marks the read ambiguous and buys it a competing DP placement for MAPQ evidence. Smaller widens the band. |
| `KIRA_MATE_GUIDE` | on | Promote the candidate locus that has a plausible mate partner, when exactly one candidate has one. `0` disables. |
| `KIRA_ANCHOR_CAP_K` | on | Never require an anchor to be longer than the seed length `k`. `0` restores the old `min_anchor_len` behaviour. |
| `KIRA_TWOTIER` | on | Rank ambiguous candidate loci by bounded Myers edit cost instead of chain score. |
| `KIRA_PREFETCH` | on | Parse the next batch on a producer thread while the current one aligns. `0` restores the serial loop. |
| `KIRA_AC_DISABLE` | auto | Aho-Corasick exact-match fast path. `0` forces on, `1` forces off. See below. |
| `KIRA_RESCUE_WIDE` | off | After a banded mate-rescue attempt misses its score bar, also run a full-window SW. `1` restores it. |
| `KIRA_MATE_SEED` | on | Bias seed-occurrence sampling toward copies that have the mate nearby. `0` disables. |
| `KIRA_ALGO` | `packed` | Fast-path aligner: `packed`, `spectral`, `wfa`, `sw`. `wfa` is the gapped path used for the INDEL figures below. |

`KIRA_MATE_SEED` matters only where a minimizer occurs more often than
`--seed-occ-cap`, i.e. in repeats. Over that cap seeding *must* discard copies,
and without a mate hint it picks by a deterministic hash — in a repeat family the
true copy survives only by luck. With the hint, copies that have the mate within
the insert window sort ahead of the rest, and the hash then only breaks ties among
equals; nothing is dropped that the unhinted path would have kept. Hints are
ignored when the mate is itself scattered across more than a few loci, since a
scattered mate carries no positional information and following it promotes wrong
placements. On GIAB chr20 this is where the indel recall gain above comes from.

`KIRA_RESCUE_WIDE` is off because the second, unbanded DP it adds cost 22% of the
alignment stage while moving zero reads in placement, MAPQ or CIGAR — measured on
both regression sets below, including the one carrying 30 bp indels in 20% of
reads, which is exactly the off-diagonal case it exists for. The wide search
still runs when the packed scan cannot (ambiguous bases in the window), since
there is then no best diagonal to band around.

The Aho-Corasick path rescans the whole reference **once per batch**, so it only
pays when the reference is much smaller than the batch. It auto-enables only when
the per-batch scan is at most half the batch's own base count. Forcing it on
outside that regime is expensive: on E. coli (4.6 Mbp) with the default 4 Mbp
batch, an 800k-read run took 35.2 s with AC versus 2.9 s without, for 6 more
correctly placed reads out of 80 000.

## Presets

- `short`: `k=19`, `w=10`, tighter chaining and smaller alignment bands.
- `long`: `k=15`, `w=10`, wider chaining and alignment bands.
- `auto`: default; selects short/long/hybrid per run based on read length distribution, ungapped identity, and chain density.

- `.kiraidx` index files are memory-mapped and used zero-copy at runtime.

## SIMD / CUDA Notes

- SIMD dispatch is runtime-detected (AVX2 on x86_64, NEON on aarch64) with a scalar fallback.
- CUDA is optional (`--features cuda`) and accelerates the batched Spectral Sieve fast path.
- Building a usable CUDA binary requires an NVIDIA toolkit with `nvcc` and a supported host C++ compiler. If kernel compilation fails, the build emits a visible warning and embeds a stub PTX that is rejected at runtime.

## Benchmarks: accuracy & speed

The aligner is validated end-to-end by **variant-calling accuracy** on GIAB HG002 — the standard truth set.

**Setup**

| | |
|---|---|
| Data | GIAB HG002, **chr20**, Illumina PE 150 bp, ~30× (12.44M reads) |
| Reference / truth | GRCh38 chr20; GIAB HG002 v4.2.1 high-confidence calls + BED |
| Hardware / threads | 16 threads, prebuilt `.kiraidx` |
| Aligner config | gapped WFA path (`KIRA_ALGO=wfa`) + two-tier locus search (`KIRA_TWOTIER`), caller filters MAPQ ≥ 13 / BQ ≥ 6 |

**Measured (kira-ls-aligner + calling)**

Both arms are the same pipeline, the same prebuilt index and the same caller
settings; only the aligner library differs. Scored against the truth VCF inside
the high-confidence BED.

| Metric | | v0.4.2 | current |
|---|---|---|---|
| **SNP**   | P / R / F1 | 0.9938 / 0.9650 / **0.9792** | 0.9937 / 0.9656 / **0.9795** |
|           | TP / FP / FN | 68811 / 429 / 2497 | 68858 / 436 / 2450 |
| **INDEL** | P / R / F1 | 0.8799 / 0.8749 / **0.8774** | 0.8787 / 0.8781 / **0.8784** |
|           | TP / FP / FN | 9834 / 1342 / 1406 | 9866 / 1362 / 1370 |

SNP: 47 more true calls for 7 more false. INDEL: 32 more true calls and 36 fewer
misses for 20 more false — recall +0.0032. The indel movement comes from
mate-guided seed sampling (see the knob table); everything else in this round was
throughput work that left the output bit-identical.

| Speed (chr20, 30×, 16 threads, best of 2-3 runs) | v0.4.2 | current | |
|---|---|---|---|
| **Alignment** | 115.4 s | 91.4 s | **1.26×** |
| Sort + markdup | 16.0 s | 17.6 s | — |
| mpileup | 66.5–116.1 s | 105.4–108.5 s | see below |
| **Full pipeline (align → sort/markdup → mpileup → VCF)** | 308.6 s | 322.0 s | ~wash |

Only the alignment row is this crate's work, and it is the only row worth
quoting. `sort + markdup` and `mpileup` are the calling tool's stages; their
run-to-run spread on this host is large (mpileup ranged 66–116 s on *identical*
code), which swamps the difference in total wall time. Treat the full-pipeline
row as "unchanged", not as a measurement.

### Memory

`align_streaming` hands each batch's scored records to the caller instead of
accumulating the run as SAM text. In-process consumers should prefer it;
`align_to_sam_bytes` holds the entire run's text at once (5.2 GB on chr20) and
its consumer then parses that straight back into records.

Peak resident set of the fused `kira-bt solid` pipeline on chr20 30×, measured by
sampling the process working set:

| | peak |
|---|---|
| before | 19.7 GB |
| releasing the sorted records before the caller runs | 14.7 GB |
| + streaming records instead of SAM text | 14.4 GB |
| + converting to the caller's form by consuming, not cloning | **12.0 GB** |

Variant output is unchanged across all four. chr20 is ~2% of GRCh38, so this
makes a 32 GB machine comfortable for a chromosome, but the peak still scales
with the input.

For that, `kira-bt solid --window-mb N` processes the reference in windows:
alignments are spilled to per-window temporary BAMs as they are produced, and
each window is then sorted, deduplicated and called on its own, so peak memory
follows the window size instead of the run.

| chr20 30×, same binary | peak | wall |
|---|---|---|
| resident (default) | 12.0 GB | ~230 s |
| `--window-mb 16` | **3.4 GB** | ~295 s |

The two produce **byte-identical VCFs**. Verified also at `--window-mb 100`
(one window for the whole chromosome), which isolates the spill round-trip from
the window boundaries — that too matches. Roughly 25–30% slower, for a peak that
no longer grows with the input.

An earlier revision of this table claimed 1.75× on the full pipeline. That was
real but came from installing `mimalloc` as a `#[global_allocator]` *in the
library*, which is imposed on every consumer's whole process — and it silently
broke one: the downstream `kira-bt norm` began writing 0-byte VCFs while exiting
0. The allocator now lives only in this crate's own binary, so the fused
in-process pipeline no longer gets it, and the sort/markdup and teardown gains
went with it. See the note at the top of `src/lib.rs`.

### Throughput regression set (simulated PE, known truth)

Alongside the GIAB gate there is a fast regression set: 150 bp PE reads sampled
from a reference with the source locus encoded in the read name, so placement
correctness is checkable per read in seconds rather than hours. 800k reads,
E. coli K-12, 16 threads, in-process stage timers:

| Stage | before | after | |
|---|---|---|---|
| input | 338.1 ms | 9.3 ms | −97% |
| sketch | 241.6 ms | 213.1 ms | −12% |
| seeding | 327.3 ms | 271.0 ms | −17% |
| chaining | 76.5 ms | 36.2 ms | −53% |
| alignment | 1136.1 ms | 429.3 ms | −62% |
| output | 393.6 ms | 158.7 ms | −60% |
| **total** | **2686 ms** | **1300 ms** | **2.07×** |

Measure with `ab.sh`-style *alternating* runs (one of each arm, repeated, minimum
per stage). Sustained benchmarking drives this class of CPU from boost down to
base clock — a 2×+ swing — so two runs taken minutes apart are not comparable,
and a single-arm-then-other-arm comparison will report whichever arm ran first as
slower. `cargo bench --bench cascade_bench` gives per-function costs for the same
reason: within one criterion invocation the clock is at least stable.

Comparing shipped defaults to shipped defaults — i.e. including the corrected
Aho-Corasick gate — the same run goes from 34.6 s to about 1.3 s. Placement
accuracy over the same set: 98.888% → 98.911% correct, 8897 → 8708 misplaced,
and 401 more correctly placed reads clear a MAPQ ≥ 13 caller filter at a cost of
3 more wrong ones.

A second set with the same generator carries indels up to 30 bp in 20% of reads,
to exercise the gapped and off-diagonal paths that the high-identity set never
reaches.

These are single-machine development numbers on one bacterial reference, not a
substitute for the GIAB gate. E. coli has far less repeat structure than a human
genome, so anything that targets paralog placement is under-stated here.

**How that compares (reference)**

Variant-calling F1 is a **pipeline** metric — it depends on the *caller* as much as the aligner, so
the fair comparison is against pipelines using the same **class** of caller. The figures below are
*typical published ranges* for GIAB HG002 (whole-genome, GA4GH/precisionFDA-style) shown **for
orientation only** — they are **not** head-to-head runs on identical data/config, and kira's row is
chr20-only:

| Aligner + caller (HG002, ~30×) | SNP F1 | INDEL F1 | caller class |
|---|---|---|---|
| **kira-ls-aligner + calling** *(measured, chr20)* | **0.979** | **0.877** | mpileup |
| bwa-mem2 / novoalign + bcftools mpileup | ~0.98–0.99 | ~0.90–0.94 | mpileup (same class) |
| bwa-mem2 + GATK HaplotypeCaller | ~0.996 | ~0.993 | local reassembly |
| bwa-mem2 + DeepVariant | ~0.9995 | ~0.998 | deep learning (industrial ceiling) |

Takeaways:

- kira's **SNP** accuracy sits in the mpileup-class band and within ~0.01–0.02 F1 of the
  deep-learning ceiling — most of that residual gap is the **caller**, not the aligner.
- kira's **INDEL** F1 trails the mpileup class and is the main open item on the accuracy roadmap.
- The GATK/DeepVariant lead comes from **local reassembly / deep-learning calling**, which is
  orthogonal to alignment.

> These are development results on a single chromosome, not a certified whole-genome benchmark.
> See the versioned [benchmark gate](docs/benchmarking.md) (runtime + SNP/INDEL F1) for regression
> criteria, and record accession/checksum + exact commands next to any result you reproduce.

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

The repository contains two reference FASTA files:

Regression and release benchmark comparisons should pass the versioned
[benchmark gate](docs/benchmarking.md), which checks runtime together with SNP
and INDEL F1 instead of accepting speed-only changes.

- `ecoli.fa`: normalized E. coli K-12 MG1655 reference derived from NCBI RefSeq accession GCF_000005845.2.
- `ref.fa`: tiny toy reference for smoke testing.

Read sets, truth VCFs, and caller outputs are intentionally not versioned in
this repository. Record their accession/checksum and exact preparation command
next to every benchmark result.

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
