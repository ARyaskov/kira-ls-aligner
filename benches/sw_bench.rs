//! Microbench: banded Smith-Waterman, i16 (16-lane) vs i8 (32-lane AVX-VNNI).
//!
//! Run with:
//!   cargo bench --bench sw_bench
//!
//! On an i7-12700 / Alder Lake with AVX-VNNI the i8 path should beat the i16
//! path by ~2× on a batch of short reads at sub-saturation scoring (per-read
//! `read_len × match_score < 120`). At default `--match=1`, 100 bp reads are
//! the sweet spot; 150 bp reads intentionally exercise the i16 fallback.
//!
//! The bench deliberately covers both lengths so a regression in either the
//! viability check or the fallback path shows up.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
#[cfg(target_arch = "x86_64")]
use kira_ls_aligner::alignment::sw_int8_vnni::int8_path_viable;
use kira_ls_aligner::alignment::{AlignmentConfig, AnchorSpan, BatchInput, align_batch_simd};
use kira_ls_aligner::simd::{self, SimdMode};
use kira_ls_aligner::types::Strand;

const SEED: u64 = 0xBEEF_FACE_DEAD;

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

/// Build N independent (read, ref_window) pairs of the requested lengths,
/// with a controlled number of mismatches per pair.
fn make_batch(n: usize, read_len: usize, ref_pad: usize, mismatches: usize) -> Batch {
    let mut reads = Vec::with_capacity(n);
    let mut refs = Vec::with_capacity(n);
    let ref_len = read_len + ref_pad;
    for lane in 0..n {
        let seed = SEED ^ (lane as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let r = random_dna(ref_len, seed);
        let q_clean = r[ref_pad / 2..ref_pad / 2 + read_len].to_vec();
        let q = perturb(&q_clean, mismatches, seed ^ 0xA5A5);
        refs.push(r);
        reads.push(q);
    }
    Batch {
        reads,
        refs,
        read_len,
        ref_len,
    }
}

struct Batch {
    reads: Vec<Vec<u8>>,
    refs: Vec<Vec<u8>>,
    read_len: usize,
    ref_len: usize,
}

impl Batch {
    fn inputs(&self) -> Vec<BatchInput<'_>> {
        self.reads
            .iter()
            .zip(self.refs.iter())
            .map(|(q, r)| BatchInput {
                read_seq: q.as_slice(),
                ref_window: r.as_slice(),
                win_start: 0,
                chain: AnchorSpan {
                    ref_id: 0,
                    ref_start: 0,
                    ref_end: self.read_len as u32,
                    read_start: 0,
                    read_end: self.read_len as u32,
                    strand: Strand::Forward,
                },
                is_rev: false,
                abort_score: i32::MIN / 8,
            })
            .collect()
    }
}

fn default_cfg() -> AlignmentConfig {
    AlignmentConfig {
        match_score: 1,
        mismatch: 4,
        gap_open: 6,
        gap_extend: 1,
        bandwidth: 50,
        xdrop: 50,
        clip_penalty: 5,
    }
}

fn bench_sw(c: &mut Criterion) {
    let cfg = default_cfg();
    let detected = simd::detect_cached();
    let has_vnni = matches!(detected, SimdMode::AvxVnni);

    // 100 bp: sub-saturation, the i8 path should win cleanly.
    bench_at(
        c,
        "sw_100bp_2mism",
        &make_batch(32, 100, 32, 2),
        cfg,
        has_vnni,
    );
    // 150 bp: borderline for the saturation watcher (max H ≈ 148 at match=1).
    bench_at(
        c,
        "sw_150bp_3mism",
        &make_batch(32, 150, 50, 3),
        cfg,
        has_vnni,
    );
}

fn bench_at(c: &mut Criterion, name: &str, batch: &Batch, cfg: AlignmentConfig, has_vnni: bool) {
    let mut g = c.benchmark_group(name);
    g.throughput(Throughput::Elements(batch.reads.len() as u64));

    let inputs = batch.inputs();
    #[cfg(target_arch = "x86_64")]
    let uses_int8 = int8_path_viable(batch.read_len, cfg);
    #[cfg(not(target_arch = "x86_64"))]
    let uses_int8 = false;

    g.bench_function("avx2_i16_2x16", |b| {
        // AVX2 handles 16 lanes per call, so process the same 32 reads as the
        // VNNI case in two calls. The group throughput is therefore valid for
        // both implementations.
        b.iter(|| {
            let mut completed = 0usize;
            for slice in inputs.chunks(16) {
                let r = align_batch_simd(black_box(slice), black_box(cfg), SimdMode::Avx2);
                completed += r.len();
                black_box(r);
            }
            black_box(completed);
        });
    });

    if has_vnni {
        let dispatch_name = if uses_int8 {
            "avx_vnni_i8_32lane"
        } else {
            "avx_vnni_i16_fallback_2x16"
        };
        g.bench_function(dispatch_name, |b| {
            // Full 32-lane batch fed straight to the i8 path.
            b.iter(|| {
                let r = align_batch_simd(
                    black_box(inputs.as_slice()),
                    black_box(cfg),
                    SimdMode::AvxVnni,
                );
                black_box(r);
            });
        });
    }

    g.finish();
    let _ = batch.ref_len;
}

criterion_group!(benches, bench_sw);
criterion_main!(benches);
