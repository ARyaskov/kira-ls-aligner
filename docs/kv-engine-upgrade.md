# KV engine 0.9.0 upgrade

The aligner now uses the published `kira_kv_engine` 0.9.0 release. Larger index
construction improved in this comparison; short-read alignment did not show a
consistent speedup. The measured regressions are retained below.

## Changes and costs

Both in-memory and external minimizer builds assign slots through numeric SIMD
batches of at most 4,096 keys. Each assignment is checked for a present,
in-range, unique slot. Assignment is O(N), with fixed batch scratch and an
additional approximately N/8-byte seen bitset; the 4N-byte result vector remains.
The temporary byte-key input buffer is released before assignment.

Immutable MPH tables use `Arc`, so cloning that component avoids serializing
and loading it again. This does not make an entire `Index::clone` O(1): other
owned arrays and reference data retain their existing clone behavior.

Saving writes the outer index once through a 64 KiB buffered temporary file,
streams the MPH payload, patches its length and publishes the file atomically.
It avoids the old writable whole-file mapping and repeated sizing serialization.
For M output bytes, writing remains O(M). The engine backend still has its own
scratch, approximately 0.5 bytes per key on the ordinary MPH path, so total
save scratch is not merely the 64 KiB buffer. Atomic publication requires
write access to the destination directory and space for a second file.
The implementation syncs the new file and preserves existing Unix permissions;
it does not claim directory-fsync durability across a power failure.

## Controlled comparison

Baseline: aligner `9115eec`, published engine 0.8.1. Candidate: these changes
with engine 0.9.0. Both use Rust 1.100.0, Windows x86-64/MSVC, the same release
profile and matched common registry versions/checksums. Each case runs once
per state, with no warm-up, repeated sampling or selection of favorable runs.

The small reference is deterministic synthetic DNA: 848 kb across unique,
paralogous and repeat contigs, seed 20261026. It contains 120,000 single-end
reads and 60,000 paired templates, mixing exact reads, SNPs and indels with
paralogs/repeats. Both states build their own short-read index and run `mem`
with four threads, a 2,000,000-base batch and `--no-PG`. Build workers and Rayon
workers are both set to four; inherited `KIRA_*` settings are cleared.

The separate construction case uses 25,000,000 deterministic bases, seed
20261115. It was added to inspect construction at a larger scale, rather than
to repeat the small case until its regression disappeared.

| Public CLI case | Wall seconds, before → after | Wall change | CPU seconds, before → after | Peak working set MiB, before → after |
|---|---:|---:|---:|---:|
| Small reference: build/save | 0.046486 → 0.226566 | +387.4% | 0.031250 → 0.078125 | 29.414 → 27.578 |
| Single-end: 120,000 reads | 1.077481 → 1.133131 | +5.2% | 3.265625 → 3.671875 | 157.547 → 153.711 |
| Paired-end: 60,000 pairs | 1.205806 → 1.203748 | -0.2% | 3.796875 → 3.875000 | 159.379 → 164.520 |
| 25 Mb reference: build/save | 1.561262 → 1.064550 | -31.8% | 1.609375 → 1.343750 | 403.902 → 401.918 |

The small build regresses from 46 ms to 227 ms in whole-process timing.
Both program logs nevertheless round internal construction to 0.02 s and
saving to 0.01 s. The discrepancy is not enough to identify startup,
scheduling, synchronization or filesystem activity as its cause. It remains
an observed regression. Single-end alignment takes 5.2% more wall time;
paired-end wall time is nearly unchanged, while its peak memory increases.
The larger build takes 31.8% less wall time and slightly less peak working set.

Small index size remains 3,943,072 bytes and large index size remains
133,165,499 bytes. Newly built payload hashes can differ because the engine
records its new hash identifier; equal sizes do not imply equal file bytes.
Working set is a whole-process high-water observation, not isolated index
memory. One observation gives no confidence interval or universal speed claim.

## Correctness and compatibility

Both states map all 120,000 reads in each alignment case. Full SAM-body digests
and every truth metric are identical. Single-end correct truth placements are
93,622/120,000; paired-end placements are 95,135/120,000. Exact and SNP groups
have 24,000/24,000 correct placements each in both datasets; repeat/paralog
placement is not perfect. Neither mapped fraction nor digest equality is
presented as 100% mapping accuracy. CIGAR/NM validation reports zero invalid
records, and duplicate or missing primary identities are rejected.

KIRAIDX2/3/4 readers remain supported. A frozen 0.8.1 fixture checks known
occurrences, true counts, capped buckets, cloned mmap readers and byte-identical
legacy resaving. Newly built MPH payloads require the upgraded reader; an old
aligner linked to engine 0.8.1 rejects the new inner hash identifier despite the
unchanged KIRAIDX4 outer format. Public alignment/index APIs are preserved.

The full Windows test suite passed 418 tests, with zero failures or ignored
tests. Seven new tests executed; an additional Unix permissions regression is
compiled only on Unix and did not execute here. Formatting and Clippy passed.
All-feature Rust linting used a PTX stub for checking Rust-side CUDA code.
A real CUDA build was attempted but failed because the installed CUDA 11.8
toolchain is incompatible with MSVC 18. No CUDA kernel was executed; this is
not GPU runtime verification.

## Evidence and reproduction

`tools/compare_kv_upgrade.py` imports the existing synthetic generator and
independent SAM evaluator. Use `--phase before` with the preserved baseline
binary, then `--phase after` with the candidate. Existing completed states are
preserved rather than silently repeated. Raw receipts remain under
`target/kv-upgrade/`: `before-results.json`, `after-results.json`,
`large-build-results.json`, input hashes and per-case logs. Machine-specific
logs, executables and generated references are not committed.
