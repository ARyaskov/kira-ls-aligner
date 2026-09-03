# Pipeline Architecture

kira-ls-aligner is organized as a strict multi-stage pipeline. Each stage lives in its own file under `src/pipeline/` and has explicit input/output structs. This keeps data flow clear, makes stages testable in isolation, and enables targeted optimization.

## Data Flow (High-Level)

1. **Stage 0 - Input** (`stage0_input.rs`)
   - Reads a batch of input reads (FASTA/FASTQ). Multiple input files are supported.
   - Output: `InputBatch { reads }`.

2. **Stage 1 - Sketch** (`stage1_sketch.rs`)
   - Computes minimizers for each read (minimap2-style canonical k-mers).
   - Emits basic per-batch stats (read length p50/p90, avg minimizers, avg read length).
   - Output: `SketchBatch { reads, sketches }`.

3. **Stage 2 - Seeding** (`stage2_seeding.rs`)
   - Ranked/pruned seeding with diagonal aggregation.
   - Each minimizer bucket emits only top-K hits (deterministic).
   - Minimizers on the same diagonal are collapsed into ProtoAnchors.
   - Exact MEM extension is optional and only applied to top candidates.
   - Output: `SeedBatch { reads, anchors, stats }`.

4. **Stage 3 - Chaining** (`stage3_chaining.rs`)
   - RMQ/Fenwick LIS-like chaining in O(n log n).
   - Gap penalties combine linear and logarithmic components.
   - Early pruning removes dominated anchors and emits only competitive chains.
   - Output: `ChainBatch { reads, chains, stats }`.

5. **Stage 4 - Alignment** (`stage4_alignment.rs`)
   - Runs ungapped prefilter on top chains (top-K by chaining score).
   - Adds chain-confidence ACCEPT path using score margin, diagonal consistency, coverage ratio, and normalized ungapped score.
   - Prefilter may ACCEPT (skip DP), REJECT, or FALLBACK to DP.
   - Performs banded Smith-Waterman for FALLBACK chains, with SIMD (AVX2/NEON) batching where possible.
   - Uses X-drop and safe early-abort bounds to reduce wasted DP work.
   - Output: `AlignBatch { reads, alignments }`.

Between stage 4 and stage 5 (also for the tiled `--split-prefix` pipeline)
alignments pass through **indel left-normalization** (`src/alignment/normalize.rs`):
equivalent gap placements inside homopolymers/tandem repeats slide to their
leftmost CIGAR, so the same variant emits one canonical CIGAR across reads
(GATK `LeftAlignIndels` / `bcftools norm` convention). Positions and insert
sizes are untouched; NM/MD are recomputed for the moved records. Kill-switch:
`KIRA_LEFT_NORM=0`.

Three record-level policies also run there, after mate rescue and pair
re-ranking but before pairing stamps mate fields, so RNEXT/PNEXT always point
at a record that is emitted: the bwa-mem `-T` score floor (a rejected primary
unmaps the read; a runner-up is never promoted past it), the `-5` rule
(smallest read coordinate of a split read is primary) and the ALT policy (an
ALT-contig primary with a primary-assembly hit within one mismatch is swapped,
as `bwa-postalt` does). Pair re-ranking gives the concordance bonus to the
winning pair only; `KIRA_PAIR_BONUS_ALL=1` extends it to every concordant
combination, so a read whose mates are concordant at two loci becomes a MAPQ
tie instead of MAPQ 20 on the tie-break (more precise, fewer callable reads —
see the README knob table for the GIAB trade-off).

6. **Stage 5 - Scoring** (`stage5_scoring.rs`)
   - Assigns MAPQ, primary/secondary flags, and suboptimal score (`XS`).
   - Competitors on ALT contigs are left out of the score-gap model for a
     primary-assembly placement.
   - Output: `ScoredBatch { reads, alignments }`.

7. **Stage 6 - Output** (`stage6_output.rs`)
   - Writes SAM records with bwa-mem compatible flags and tags (NM, MD, AS, XS, RG,
     XA/SA) plus the mate tags `MC`/`MQ`/`ms` on paired records, built from the
     adjacent mate's primary in a pre-pass so the parallel chunks stay independent.
   - Supplementary segments are hard-clipped unless `-Y`; `-M` flags them 0x100;
     `-C` appends the FASTQ comment.

## Fused binary output (`--emit bam | sorted-bam | cram | sorted-cram`)

`src/cli/commands/emit.rs` drives the pipeline through `Aligner::align_streaming`
and hands each scored batch, serialised by stage 6, to a converter thread that
parses the SAM text into kira-bam records in parallel (line-aligned chunks on
the rayon pool). Unsorted output streams straight into kira-bam's multi-threaded
BGZF (or CRAM) writer. Sorted output collects records up to `--sort-memory`,
then runs kira-bam's in-memory coordinate sort with fused markdup and writes
once; past the budget everything collected so far is spilled to an unsorted
BAM beside the output, later batches stream into it, and kira-bam's external
sort (plus its two-pass markdup when `--markdup`) finishes the file. No
intermediate SAM is written on either path. `--bai` indexes BAM output;
CRAM takes `REF` and builds `REF.fai` if missing.

## Auto Mode Selection

When `-x auto` (default), the first batch is used to classify the dataset as short, long, or hybrid using:
- Read length distribution (p50/p90)
- Ungapped identity/mismatch stats from the prefilter
- Chain density (chains/read) and minimizer density

The selected mode is applied to subsequent batches and logged once when `KIRA_STATS=1`.

## Input decoding

FASTQ input goes through kira-fastq, which picks its backend from the file's
magic bytes rather than its name — `bgzip` writes BGZF into plain `.gz` names.
BGZF blocks are independently deflated, so BGZF input is inflated on
`KIRA_BGZF_THREADS` workers (default 2, per input file): HG002 R1, 748 MB,
decodes in 5.0 s on one thread and 2.1 s on two, producing identical records
either way — chr20 30× end to end, 98.8 s to 83.6 s for a byte-identical BAM
body. Plain gzip has no such structure and stays single-threaded.

`-` reads standard input, sniffing gzip/BGZF from the first bytes. The stream
is passed through a CRLF-normalising adapter first: kira-fastq strips `\r`
only when it sits in the same buffer fill as its `\n`, so a `\r\n` split across
two pipe reads otherwise reaches the parser as a stray `\r` and fails the
record as a sequence/quality length mismatch (verified against kira-fastq
0.4.0). File-backed inputs are memory-mapped or block-decoded and never hit
that path.

Progress accounting follows the backend, since `tell()` means different things
per source: a file offset for plain input, a BGZF virtual offset (whose block
offset is the real file position) for single-threaded BGZF, and decoded bytes
for gzip and the parallel BGZF reader — the last is scaled back into file bytes
by an assumed FASTQ compression ratio, because only the progress bar consumes it.

## Where the time goes (chr20, 30×, 16 threads, SAM output)

Measured 2026-09-01 with `KIRA_STATS=1` on HG002 chr20 (12.44M reads, 99 s wall):

| Stage | Sum over batches | Share |
|---|---|---|
| seeding | 29.8 s | 34 % |
| alignment (stage 4 incl. rescue) | 30.7 s | 35 % |
| sketch | 10.7 s | 12 % |
| chaining | 6.4 s | 7 % |
| output | 3.9 s | 4 % |
| scoring | 2.9 s | 3 % |

Inside stage 4, mate rescue is 6.9 s of the 30.7 s. 71 % of reads resolve as
exact matches, 16 % on the packed-spectral certificate, 17 % through WFA and
0.1 % through banded SW. The sketch stage spends ~14 µs per read against a
~0.7 µs minimizer kernel, so the next performance work is allocation and
memory traffic in sketch/seeding, not the DP kernels.

## Stage Details

### Stage 0 - Input
- **Input:** one or more FASTA/FASTQ files.
- **Output:** `InputBatch` containing `Vec<ReadRecord>`.
- **Notes:** uses `needletail` and mmap-backed I/O; batches are sized by CLI option `-K` (bases).

### Stage 1 - Sketch (Minimizers)
- **Input:** `InputBatch`.
- **Output:** `SketchBatch` with `ReadSketch` per read.
- **Algorithm:**
  - Canonical k-mers, rolling hash, windowed minimizers via monotonic deque.
- **Performance:**
  - O(n) per read; SIMD hooks prepared in `simd` module for future hashing acceleration.

### Stage 2 - Seeding
- **Input:** `SketchBatch` + index.
- **Output:** `SeedBatch` with pruned anchors and seeding stats.
- **Algorithm:**
  - Minimizer lookup into per-hash buckets, emit only top-K hits per bucket.
  - Aggregate hits by diagonal into ProtoAnchors before extension.
  - Exact MEM extension is optional and only applied to top candidates.
  - Hard caps for anchors per read and per diagonal are enforced before chaining.
- **Performance:**
  - Fewer anchors reduce chaining and DP pressure dramatically.

### Stage 3 - Chaining
- **Input:** `SeedBatch`.
- **Output:** `ChainBatch` with chaining stats.
- **Algorithm:**
  - Sort anchors by reference position.
  - O(n log n) RMQ/Fenwick chaining (LIS-like).
  - Gap penalties use linear + log terms.
  - Early pruning drops dominated anchors.
- **Performance:**
  - Sub-quadratic chaining even with high anchor counts.

### Stage 4 - Alignment
- **Input:** `ChainBatch` + reference.
- **Output:** `AlignBatch`.
- **Algorithm:**
  - Ungapped prefilter on top chains (top-K by chaining score).
  - Chain-confidence ACCEPT uses score margin, diagonal consistency, coverage ratio, and normalized ungapped score.
  - FALLBACK path runs banded Smith-Waterman around chain diagonals.
  - X-drop and safe early-abort bounds reduce wasted DP work.
- **Performance:**
  - Banded DP reduces memory to O(bandwidth * read_len).
  - SIMD batching is used when read lengths and windows match.

### Stage 5 - Scoring / MAPQ
- **Input:** `AlignBatch`.
- **Output:** `ScoredBatch`.
- **Algorithm:**
  - Primary alignment = best score.
  - Secondary alignments (loci covering ≥50 % of the primary's read region) are
    flagged with MAPQ 0; disjoint loci become supplementary.
  - MAPQ approximates bwa-mem/minimap2 behavior with a 60 cap, from a best/sub
    score-gap posterior (`KIRA_MAPQ_BETA`) minus a multiplicity penalty of
    `KIRA_MAPQ_MULT · ln(n)` over `n` above-floor competing loci (bwa-mem
    `mem_approx_mapq_se` term).
  - Additional ceilings: identity (`KIRA_ID_MAPQ`), repeat copy-number
    (`KIRA_REPEAT_MAPQ`), mean FASTQ quality, mate-rescue
    (`KIRA_RESCUE_MAPQ_CAP`) and discordant-pair caps.

### Stage 6 - Output
- **Input:** `ScoredBatch` + `SamWriter`.
- **Output:** SAM records.
- **Tags:** NM, MD, AS, XS, RG (with fast-output optionally suppressing heavy tags).
- **Flags:** primary/secondary/supplementary compatible with bwa-mem semantics.

## Performance Notes

- Thread parallelism is applied at the per-read level via Rayon.
- The reference index is immutable and shared across threads.
- Batch processing avoids unbounded memory usage and improves cache locality.
- CUDA support is optional behind the `cuda` feature gate and accelerates the
  batched Spectral Sieve prefilter. Builds with a stub PTX are rejected when
  the GPU backend is initialized.

## Extensibility

Each stage can be optimized independently (SIMD hashing, RMQ chaining, GPU offload) without altering the pipeline boundaries. This keeps algorithmic changes contained and testable.
