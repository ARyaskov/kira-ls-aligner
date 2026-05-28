//! Cross-check: when the `cuda` feature is enabled and a CUDA device is
//! available, the GPU kernel must produce results consistent with the CPU
//! `bitpacked::scan` reference for a corpus of randomized read/window
//! pairs.
//!
//! When the feature is off or no GPU is present, the test is a no-op.

#![cfg(feature = "cuda")]

use kira_ls_aligner::alignment::bitpacked::{self, PackedDna};
use kira_ls_aligner::cuda::{self, CudaBackend, CudaJob};

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn make_corpus(n: usize) -> Vec<(Vec<u8>, Vec<u8>, usize)> {
    let bases = b"ACGT";
    let mut rng = 0xCAFE_F00Du64;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let r = 16 + (xorshift(&mut rng) as usize % 200);
        let w = r + (xorshift(&mut rng) as usize % 80);
        let read: Vec<u8> = (0..r)
            .map(|_| bases[(xorshift(&mut rng) as usize) % 4])
            .collect();
        let mut text: Vec<u8> = (0..w)
            .map(|_| bases[(xorshift(&mut rng) as usize) % 4])
            .collect();
        if xorshift(&mut rng) % 2 == 0 && w >= r {
            let start = (xorshift(&mut rng) as usize) % (w - r + 1);
            text[start..start + r].copy_from_slice(&read);
            // Inject 0-3 mismatches for realism.
            let n_mism = (xorshift(&mut rng) as usize) % 4;
            for _ in 0..n_mism {
                let pos = start + (xorshift(&mut rng) as usize) % r;
                text[pos] = if text[pos] == b'A' { b'T' } else { b'A' };
            }
        }
        let max_mism = (r * 8 / 100).max(5);
        out.push((read, text, max_mism));
    }
    out
}

#[test]
fn gpu_results_consistent_with_cpu() {
    if !cuda::is_available() {
        eprintln!("[skipped] no CUDA device available");
        return;
    }
    let mut backend = match CudaBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[skipped] CUDA init failed: {e}");
            return;
        }
    };

    let corpus = make_corpus(64);
    let jobs: Vec<CudaJob> = corpus
        .iter()
        .map(|(read, text, max_mism)| {
            let pr = PackedDna::pack(read);
            let pt = PackedDna::pack(text);
            let shifted = pt.pre_shifted_window();
            CudaJob {
                read_packed: pr.bits,
                read_nucs: read.len() as u32,
                ref_shifted: shifted,
                ref_nucs: text.len() as u32,
                max_mismatches: *max_mism as u32,
            }
        })
        .collect();

    let results = backend
        .run_batch(&jobs)
        .expect("GPU run_batch should succeed on a healthy device");

    assert_eq!(results.len(), corpus.len());
    for (i, ((read, text, max_mism), gpu_res)) in corpus.iter().zip(results.iter()).enumerate() {
        let pr = PackedDna::pack(read);
        let pt = PackedDna::pack(text);
        let shifted = pt.pre_shifted_window();
        let cpu_hit = bitpacked::scan(&pr, &shifted, text.len(), *max_mism);

        match (cpu_hit, gpu_res.shift >= 0) {
            (Some(cpu), true) => {
                // Both accepted. Mismatches may differ if CPU and GPU pick
                // different first-acceptable shifts among ties — but both
                // should report ≤ max_mism, and the *actual* Hamming
                // distance at the GPU's reported shift must equal what the
                // GPU reported.
                assert!(
                    gpu_res.mismatches as usize <= *max_mism,
                    "case {i}: GPU mismatches {} > max {}",
                    gpu_res.mismatches,
                    max_mism
                );
                assert!(
                    cpu.mismatches <= *max_mism,
                    "case {i}: CPU mismatches {} > max {}",
                    cpu.mismatches,
                    max_mism
                );
                // Verify self-consistency: GPU's reported (shift, mism) matches
                // a hand-computed Hamming distance at that shift.
                let actual = read
                    .iter()
                    .zip(text[gpu_res.shift as usize..].iter())
                    .take(read.len())
                    .filter(|(a, b)| a != b)
                    .count();
                assert_eq!(
                    actual, gpu_res.mismatches as usize,
                    "case {i}: GPU self-inconsistent at shift {}",
                    gpu_res.shift
                );
            }
            (None, false) => {
                // Both rejected — good.
            }
            (cpu, gpu) => {
                panic!(
                    "case {i}: CPU and GPU disagree on acceptability: cpu={cpu:?}, gpu={gpu:?}, gpu_res={gpu_res:?}"
                );
            }
        }
    }
}
