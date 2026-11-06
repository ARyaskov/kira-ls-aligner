# Adaptive search, buffer reuse, and affine scoring

Date: 2026-10-27. Baseline commit: `7732789b478de3945af653609225109bc675c8fb`.

## Changes

1. **Bounded adaptive search.** A short read with discarded repeat occurrences
   and less than 50% query coverage gets one wider seeding pass. The runtime
   occurrence limit and sample size grow by at most 4x (sample size stops at 64),
   while the index build cap is always respected. Both anchor sets are merged;
   build-truncated sentinel buckets are never treated as complete. Candidate
   verification is limited to `KIRA_TWOTIER_K` loci. A clear unpaired winner can
   use one candidate, unknown probes retain two, and distinguishable near-ties
   can use three. Paired decisions retain at least two candidates when the
   adaptive search runs. An explicit `KIRA_SHORT_DPTOPK` keeps its override.

2. **Less intermediate allocation.** Sketch and seeding execute together in
   pair-aligned blocks of 128 reads. Each worker reuses two minimizer buffers
   across blocks and batches instead of retaining a minimizer allocation per
   read until seeding ends. Small occurrence buckets are read directly without
   materializing a copy or probing the hot-bucket cache. Exact candidate
   evidence allocates neither CIGAR nor MD; verified non-exact alignments are
   reused without repeating ungapped extension or traceback. The separate stage
   APIs remain available and have an equivalence test.

3. **Consistent affine rewards in WFA.** For match reward A, mismatch penalty B,
   gap open O and gap extension E, the full-query affine score is optimized by
   minimizing the loss

   ```text
   A * query_length - alignment_score
     = (A+B) * mismatches + (A+E) * inserted_bases
       + E * deleted_bases + O * gap_runs
   ```

   WFA therefore uses distinct insertion and deletion extension costs. Its
   traceback, score-only engine, ungapped-vs-gap probe and CGK verification use
   the same reward model. The public legacy penalty-only WFA API is retained.
   Invalid transitions beyond the query/reference matrix are rejected before
   they can win a wavefront cell. Uncertain candidate ranking uses verified
   alignment scores rather than converting unit edit distances into affine
   scores. Global LSH/CGK rescue is not used as evidence for a particular locus.

## Endpoints and limits

WFA remains a full-query aligner; local SW can clip the query. These are distinct
endpoint policies, not interchangeable optimization problems. A compensating
gap near the beginning triggers a bounded search for a better seed origin
(up to 32 bases on either side, limited by the band). An exact prefix usually
identifies one retry origin; otherwise the retry uses an ends-free wavefront.
The retry replaces the initial result only if its affine score improves.
Ordinary terminal INDELs remain full-query alignments; strongly divergent
results can fall back to local SW for clipping. The endpoint search is still
heuristic and does not guarantee the globally best genomic placement.

Adaptive WFA pruning remains an explicit user option; the default exact engine
is what the independent oracle validates. No CUDA kernels were changed or
benchmarked. This work does not add a graph reference or change the seeding
scheme/index format.

Stage timers changed: stage 1 now measures AC, while stage 2 measures fused
sketch + seeding. Compare stages 1+2 together against the baseline. Verified
candidates can bypass the ungapped prefilter, so prefilter attribution counters
are not directly comparable between versions.

## Validation method

- Baseline binary built from a separate archive of the baseline commit with
  the same Cargo.lock, compiler and release profile as the changed binary.
- `cargo fmt --all -- --check`;
  `cargo clippy --locked --all-targets -- -D warnings`;
  `cargo test --locked --all-targets --no-fail-fast`.
- Independent full-matrix affine DP oracle: 1800 deterministic cases across
  three scoring configurations, leading-reference slack, short reference
  windows, traceback replay, score-only parity and insufficient score budgets.
- 300 deterministic comparisons exercise cases where the legacy penalty-only
  objective chooses a worse affine score.
- Additional regressions cover a shifted seed origin before an insertion,
  adaptive candidate budgets, exact-candidate allocation avoidance, fused/staged
  equivalence, masks, block boundaries, odd read counts, and complete repeat
  buckets versus build-truncation sentinels. In the controlled repeat test the
  strict initial pass finds zero anchors; the bounded retry finds all six loci.

## Benchmark protocol

Windows, Intel Core i7-12700, Rust 1.99.1. Two deterministic datasets contain
120,000 reads each: single-end and paired-end (60,000 templates). Reference size
is 848,000 bases. Each dataset has equal-sized exact, SNP, INDEL, paralog and
identical-repeat strata. Reads start from 150 reference bases and carry 1–12 bp
insertions/deletions or 1–3 substitutions where applicable. FASTQ quality is
constant Q40. The random seed is 20261026.

For each dataset and thread count: one warm-up per binary, then five measured
runs per binary, alternating AB/BA order. Compilation and other benchmark jobs
do not run concurrently with timed runs. Both binaries read the same index and
input bytes, emit full SAM, and use `-K 2000000`. Timing includes startup, index
loading and output. Reported wall/CPU/peak-memory values are medians. Peak memory
is Windows PeakWorkingSet, including resident mapped pages, not heap allocation
size. Repeated SAM bodies are hashed to check determinism.

Reproduce from the repository root after building/copying the original binary:

```powershell
python tools/compare_aligners.py --before target/comparison/before.exe `
  --after target/release/kira_ls_aligner.exe --reads 120000 --repeats 5 `
  --threads 1 --output target/comparison/verified-t1
python tools/compare_aligners.py --before target/comparison/before.exe `
  --after target/release/kira_ls_aligner.exe --reads 120000 --repeats 5 `
  --threads 4 --output target/comparison/verified-t4
```

## Interpretation

Source-position agreement requires the original contig, strand and start within
5 bp. INDEL recovery additionally requires the correct operation and length
within 5 bp of the injected event. CIGAR consumption and NM are independently
replayed against the reference. Some terminal deletions are unobservable from
the remaining read sequence. Identical repeat copies are intentionally
indistinguishable; choosing another identical copy is not evidence that one
mapper inferred an observable source allele incorrectly. Per-stratum counts are
retained to avoid hiding that ambiguity inside an aggregate accuracy figure.

These are synthetic stress tests, not GIAB variant-calling precision/recall or
whole-genome measurements. They validate this workload and reveal tradeoffs;
they do not establish that every library, platform or sequencing error model
improves. Whole-genome/held-out GIAB validation is still needed before making
broader accuracy claims.

## Measurements

[Raw measurements and binary hashes](improvements-1-3-results.json).

Tests: **373 passed before, 382 passed after**, zero failed. Clippy with warnings denied and formatting checks passed.

### Wall time

| Dataset | Threads | Before, s | After, s | Change | Before min–max, s | After min–max, s |
|---|---:|---:|---:|---:|---:|---:|
| single | 1 | 2.400 | 2.501 | +4.19% | 2.388–2.446 | 2.497–2.557 |
| paired | 1 | 2.740 | 2.725 | -0.56% | 2.721–2.800 | 2.709–2.758 |
| single | 4 | 1.042 | 1.100 | +5.64% | 1.010–1.438 | 1.059–1.555 |
| paired | 4 | 1.471 | 1.388 | -5.60% | 1.326–1.800 | 1.323–1.841 |

### CPU and peak working set

| Dataset | Threads | CPU before, s | CPU after, s | CPU change | Memory before, MiB | Memory after, MiB | Memory change |
|---|---:|---:|---:|---:|---:|---:|---:|
| single | 1 | 2.406 | 2.500 | +3.90% | 142.98 | 140.22 | -1.93% |
| paired | 1 | 2.750 | 2.750 | +0.00% | 157.28 | 145.50 | -7.49% |
| single | 4 | 3.609 | 3.672 | +1.73% | 180.38 | 170.17 | -5.66% |
| paired | 4 | 4.156 | 4.156 | +0.00% | 194.46 | 179.25 | -7.82% |

### Accuracy and invariants

Counts below are identical with one and four threads. Both arms produce identical SAM bodies across all five repeats within each configuration.

| Metric | Single before | Single after | Paired before | Paired after |
|---|---:|---:|---:|---:|
| Input reads | 120000 | 120000 | 120000 | 120000 |
| Mapped reads | 120000 | 120000 | 120000 | 120000 |
| Source-position agreement, all strata | 91981 | 93622 | 93538 | 95135 |
| Source-position disagreement at MAPQ >= 30 | 2280 | 639 | 2429 | 670 |
| Injected INDEL reads | 24000 | 24000 | 24000 | 24000 |
| Recovered injected INDEL | 19309 | 21617 | 19219 | 21623 |
| Invalid CIGAR or NM | 0 | 0 | 0 | 0 |

Source-position agreement by stratum (24,000 reads per stratum):

| Stratum | Single before | Single after | Paired before | Paired after |
|---|---:|---:|---:|---:|
| exact | 24000 | 24000 | 24000 | 24000 |
| snp | 24000 | 24000 | 24000 | 24000 |
| indel | 21721 | 23362 | 21575 | 23334 |
| paralog | 17126 | 17126 | 18283 | 18315 |
| repeat | 5134 | 5134 | 5680 | 5486 |

### Assessment

**This is an accuracy/memory improvement with a measured single-end speed regression, not a universal speedup.** Single-end wall time increases by approximately 4–6%; paired-end wall time is approximately unchanged with one thread and improves by approximately 6% with four. CPU time is nearly unchanged for paired-end and increases modestly for single-end. Wall-time ranges are provided above because host scheduling and CPU frequency affect individual runs.

INDEL recovery increases from 80.45% to 90.07% for single-end and from 80.08% to 90.10% for paired-end. Exact/SNP source-position agreement remains 100% on this simulation. The identical-repeat stratum has no observable source-copy identity for many reads; its source-agreement count is included above, including the paired-end decrease, rather than hidden in an aggregate claim.

No claim of unchanged or improved GIAB SNP/INDEL F1 is made: the required real data and caller round-trip were not available in this checkout.
