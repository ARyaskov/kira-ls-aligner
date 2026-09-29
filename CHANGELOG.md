# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Mate rescue now searches a window centred on the mate's expected span instead of an offset one.
- The `/1` and `/2` mate suffixes are stripped from read names, as bwa-mem does.
- Indels are left-normalised correctly for insertions with period >= 2 (tandem repeats).
- The `MD` string is no longer split by `N` (skip) CIGAR operations.
- Chaining no longer accumulates diagonal drift across overlapping anchors.
- The splice aligner emits `M` for SNP-sized gaps between anchors rather than a spurious indel.
- The seeder occurrence cap is bounded by the index cap, with deterministic bucket truncation.
- Empty `SEQ`/`QUAL` fields are written as `*` instead of an empty column.
- Supplementary records carry the `SA` tag.
- An unmapped read is placed at its mapped mate's position (RNAME/POS), as bwa-mem does.
- `-I` accepts bwa-style `mean[,std[,max[,min]]]` insert-size specifications.
- `-w` is the DP band width, matching bwa-mem, rather than the minimizer window.
- Unknown `-x` presets are rejected; bwa-mem's `pacbio`/`ont2d`/`intractg` map to `long`.
- `--gpu` starts the CUDA dispatcher instead of being silently ignored.
- FASTA reads are accepted, as the CLI help and README already claimed.
- `XA` is built from the full candidate list before the record cap is applied.
- The binary exits quietly on `SIGPIPE`, like bwa and samtools.

## [0.4.6] - 2026-09-03

Version bump (no changelog kept).

## [0.4.5] - 2026-09-02

Version bump (no changelog kept).

## [0.4.4] - 2026-09-01

Version bump (no changelog kept).

## [0.4.3] - 2026-07-27

Version bump (no changelog kept).

## [0.4.2] - 2026-07-15

Version bump (no changelog kept).

## [0.4.1] - 2026-06-27

Version bump (no changelog kept).

## [0.4.0] - 2026-06-20

Version bump (no changelog kept).

## [0.3.0] - 2026-05-31

Version bump (no changelog kept).

## [0.2.0] - 2026-05-28

Version bump (no changelog kept).
