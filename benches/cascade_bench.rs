//! Microbench: the per-read hot path, function by function.
//!
//! Run with:
//!   cargo bench --bench cascade_bench
//!
//! The end-to-end stage timers cannot attribute a cost to a function inside a
//! stage, and on a loaded machine their spread rivals the effect being measured.
//! This isolates the primitives every short read pays for.
//!
//! Numbers are per read (150 bp), directly comparable to the stage timers: at
//! 800k reads, 1 ns/read here is 0.8 ms of single-threaded work in that stage.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use kira_ls_aligner::alignment::bitpacked::{
    PackedHit, pack_into, pre_shift_into, scan_best_with_second_raw,
};
use kira_ls_aligner::alignment::myers::bounded_edit_distance;
use kira_ls_aligner::alignment::wfa::{WfaOptions, WfaPenalties, wfa_align};
use kira_ls_aligner::seq::{common_prefix_len, common_suffix_len, reverse_complement};
use kira_ls_aligner::simd;
use kira_ls_aligner::sketch::{MinimizerConfig, minimizers_into};

const READ_LEN: usize = 150;
/// Reference window the cascade actually scans: read length plus the band.
const WINDOW_PAD: usize = 50;

fn random_dna(len: usize, mut state: u64) -> Vec<u8> {
    let alphabet = b"ACGT";
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        out.push(alphabet[(state >> 60) as usize & 0x3]);
    }
    out
}

fn perturb(seq: &[u8], mismatches: usize, mut state: u64) -> Vec<u8> {
    let mut out = seq.to_vec();
    let alphabet = b"ACGT";
    for _ in 0..mismatches {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let pos = (state as usize) % out.len();
        let mut b = alphabet[(state >> 56) as usize & 0x3];
        if b == out[pos] {
            b = alphabet[((state >> 56) as usize + 1) & 0x3];
        }
        out[pos] = b;
    }
    out
}

/// One realistic (read, reference window) pair with `mism` substitutions.
///
/// The read must start at `window[0]`, as the cascade builds it — the window
/// begins at the seed-implied reference start and the padding is trailing slack.
/// A mid-window read makes the query-global WFA spend its whole score budget on
/// leading deletions, measuring a case the pipeline never produces.
fn pair(mism: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
    let window = random_dna(READ_LEN + WINDOW_PAD, seed);
    let clean = window[..READ_LEN].to_vec();
    (perturb(&clean, mism, seed ^ 0x5A5A), window)
}

fn bench_sketch(c: &mut Criterion) {
    let read = random_dna(READ_LEN, 0x11);
    let cfg = MinimizerConfig { k: 19, w: 10 };
    let mut out = Vec::new();
    let mut g = c.benchmark_group("sketch");
    g.throughput(Throughput::Elements(1));
    g.bench_function("minimizers_150bp_k19_w10", |b| {
        b.iter(|| {
            minimizers_into(black_box(&read), black_box(&cfg), &mut out);
            black_box(out.len());
        })
    });
    g.finish();
}

fn bench_packed_scan(c: &mut Criterion) {
    let mut g = c.benchmark_group("packed_spectral");
    g.throughput(Throughput::Elements(1));

    for &mism in &[0usize, 2, 6] {
        let (read, window) = pair(mism, 0x2200 + mism as u64);
        let mut read_bits = Vec::new();
        let mut ref_bits = Vec::new();
        let mut shifted = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

        // The three pieces the cascade runs back to back, measured together:
        // this is what one fast-path candidate costs.
        g.bench_function(format!("pack+shift+scan_{mism}mism"), |b| {
            b.iter(|| {
                let rv = pack_into(black_box(&read), &mut read_bits);
                let tv = pack_into(black_box(&window), &mut ref_bits);
                pre_shift_into(&ref_bits, &mut shifted);
                let hit: Option<(PackedHit, Option<usize>)> = scan_best_with_second_raw(
                    &read_bits,
                    read.len(),
                    rv && tv,
                    &shifted,
                    window.len(),
                );
                black_box(hit);
            })
        });

        // Packing alone, to separate encoding cost from scanning cost.
        g.bench_function(format!("pack_only_{mism}mism"), |b| {
            b.iter(|| {
                let v = pack_into(black_box(&window), &mut ref_bits);
                black_box(v);
            })
        });
    }
    g.finish();
}

fn bench_myers(c: &mut Criterion) {
    let mut g = c.benchmark_group("myers");
    g.throughput(Throughput::Elements(1));
    for &mism in &[2usize, 6, 40] {
        let (read, window) = pair(mism, 0x3300 + mism as u64);
        // 15% of 150 + 4, the shipped reject bound.
        let max_k = READ_LEN * 15 / 100 + 4;
        g.bench_function(format!("bounded_edit_{mism}mism"), |b| {
            b.iter(|| {
                black_box(bounded_edit_distance(
                    black_box(&read),
                    black_box(&window),
                    max_k,
                ))
            })
        });
    }
    g.finish();
}

fn bench_wfa(c: &mut Criterion) {
    let mut g = c.benchmark_group("wfa");
    g.throughput(Throughput::Elements(1));
    let pen = WfaPenalties {
        mismatch: 4,
        gap_open: 6,
        gap_extend: 1,
    };
    for &mism in &[1usize, 3] {
        let (read, window) = pair(mism, 0x4400 + mism as u64);
        let opts = WfaOptions {
            max_score: READ_LEN as i32 * 25 / 100 * 4,
            adaptive_drop: None,
            text_begin_free: 0,
        };
        g.bench_function(format!("wfa_align_{mism}mism"), |b| {
            b.iter(|| {
                black_box(wfa_align(
                    black_box(&read),
                    black_box(&window),
                    pen,
                    opts,
                ))
            })
        });
    }
    g.finish();
}

fn bench_byte_primitives(c: &mut Criterion) {
    let mut g = c.benchmark_group("byte_primitives");
    g.throughput(Throughput::Elements(1));

    let (read, window) = pair(2, 0x5500);
    let aligned = &window[..READ_LEN];

    g.bench_function("count_mismatches_150", |b| {
        b.iter(|| black_box(simd::count_mismatches(black_box(&read), black_box(aligned))))
    });

    // Seed extension: from a 19-mer seed in the middle, outward to both ends.
    let seed_at = 60usize;
    g.bench_function("seed_extend_both_ways_150", |b| {
        b.iter(|| {
            let l = common_suffix_len(
                black_box(&read[..seed_at]),
                black_box(&aligned[..seed_at]),
            );
            let r = common_prefix_len(
                black_box(&read[seed_at + 19..]),
                black_box(&aligned[seed_at + 19..]),
            );
            black_box(l + r)
        })
    });

    g.bench_function("reverse_complement_150", |b| {
        b.iter(|| black_box(reverse_complement(black_box(&read))))
    });

    g.finish();
}

criterion_group!(
    benches,
    bench_sketch,
    bench_packed_scan,
    bench_myers,
    bench_wfa,
    bench_byte_primitives
);
criterion_main!(benches);
