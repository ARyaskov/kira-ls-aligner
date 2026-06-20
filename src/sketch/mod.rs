use std::collections::VecDeque;

use crate::types::{Minimizer, Strand};

pub mod simhash;

/// Minimizer sketch configuration.
#[derive(Clone, Copy, Debug)]
pub struct MinimizerConfig {
    pub k: usize,
    pub w: usize,
}

/// Compute minimizers for a sequence using minimap2-style canonical k-mers.
pub fn minimizers(seq: &[u8], cfg: &MinimizerConfig) -> Vec<Minimizer> {
    if cfg.k == 0 || cfg.w == 0 || cfg.k >= 32 || seq.len() < cfg.k {
        return Vec::new();
    }
    let k = cfg.k as u32;
    let w = cfg.w as u32;
    let mask: u64 = (1u64 << (2 * cfg.k)) - 1;

    let mut mins = Vec::new();
    let mut deque: VecDeque<(u64, u32, u32, Strand)> = VecDeque::new();
    let mut kmer: u64 = 0;
    let mut rkmer: u64 = 0;
    let mut valid_len: u32 = 0;
    let mut kmer_index: u32 = 0;
    let mut last_out: Option<(u64, u32)> = None;

    for (i, &b) in seq.iter().enumerate() {
        let code = base_code(b);
        if code < 4 {
            kmer = ((kmer << 2) | code as u64) & mask;
            rkmer = (rkmer >> 2) | (((3 - code) as u64) << ((k - 1) * 2));
            valid_len += 1;
            if valid_len >= k {
                let pos = (i + 1 - cfg.k) as u32;
                let (hash, strand) = if kmer <= rkmer {
                    (hash64(kmer), Strand::Forward)
                } else {
                    (hash64(rkmer), Strand::Reverse)
                };
                while let Some(back) = deque.back() {
                    // Keep the rightmost representative of equal minima.
                    // Emitting every tied k-mer makes low-complexity windows
                    // produce O(w) redundant seeds per position.
                    if back.0 < hash {
                        break;
                    }
                    deque.pop_back();
                }
                deque.push_back((hash, kmer_index, pos, strand));

                while let Some(front) = deque.front() {
                    if front.1 + w <= kmer_index {
                        deque.pop_front();
                    } else {
                        break;
                    }
                }

                if kmer_index + 1 >= w {
                    if let Some(front) = deque.front() {
                        if last_out.is_none_or(|last| last.0 != front.0 || last.1 != front.2) {
                            mins.push(Minimizer {
                                hash: front.0,
                                pos: front.2,
                                strand: front.3,
                            });
                            last_out = Some((front.0, front.2));
                        }
                    }
                }

                kmer_index += 1;
            }
        } else {
            kmer = 0;
            rkmer = 0;
            valid_len = 0;
            kmer_index = 0;
            deque.clear();
            last_out = None;
        }
    }

    mins
}

fn base_code(b: u8) -> u8 {
    match b {
        b'A' | b'a' => 0,
        b'C' | b'c' => 1,
        b'G' | b'g' => 2,
        b'T' | b't' => 3,
        _ => 4,
    }
}

fn hash64(mut x: u64) -> u64 {
    // SplitMix64
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
