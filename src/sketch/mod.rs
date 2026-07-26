use crate::types::{Minimizer, Strand};

pub mod simhash;

/// Minimizer sketch configuration.
#[derive(Clone, Copy, Debug)]
pub struct MinimizerConfig {
    pub k: usize,
    pub w: usize,
}

/// Capacity of the fixed monotonic-window ring buffer. The window holds at most
/// `w` entries, so any `w` up to this bound runs allocation-free; larger `w`
/// falls back to [`minimizers_deque`].
const RING_CAP: usize = 256;

/// Compute minimizers for a sequence using minimap2-style canonical k-mers.
pub fn minimizers(seq: &[u8], cfg: &MinimizerConfig) -> Vec<Minimizer> {
    let mut out = Vec::new();
    minimizers_into(seq, cfg, &mut out);
    out
}

/// [`minimizers`] appending into a caller-owned buffer (cleared first).
pub fn minimizers_into(seq: &[u8], cfg: &MinimizerConfig, mins: &mut Vec<Minimizer>) {
    mins.clear();
    if cfg.k == 0 || cfg.w == 0 || cfg.k >= 32 || seq.len() < cfg.k {
        return;
    }
    if cfg.w > RING_CAP {
        minimizers_deque(seq, cfg, mins);
        return;
    }
    let k = cfg.k as u32;
    let w = cfg.w as u32;
    let mask: u64 = (1u64 << (2 * cfg.k)) - 1;

    // Expected density of a (w, k)-minimizer scheme is 2/(w+1) per base.
    mins.reserve(seq.len() * 2 / (cfg.w + 1) + 8);

    // Monotonic window over (hash, kmer_index, pos, strand); `head` + `len`
    // bound the live range, entries stay in insertion order.
    let mut ring = [(0u64, 0u32, 0u32, Strand::Forward); RING_CAP];
    let mut head = 0usize;
    let mut len = 0usize;

    let mut kmer: u64 = 0;
    let mut rkmer: u64 = 0;
    let mut valid_len: u32 = 0;
    let mut kmer_index: u32 = 0;
    let mut last_out: Option<(u64, u32)> = None;

    for (i, &b) in seq.iter().enumerate() {
        let code = base_code(b);
        if code >= 4 {
            kmer = 0;
            rkmer = 0;
            valid_len = 0;
            kmer_index = 0;
            head = 0;
            len = 0;
            last_out = None;
            continue;
        }
        kmer = ((kmer << 2) | code as u64) & mask;
        rkmer = (rkmer >> 2) | (((3 - code) as u64) << ((k - 1) * 2));
        valid_len += 1;
        if valid_len < k {
            continue;
        }

        let pos = (i + 1 - cfg.k) as u32;
        let (hash, strand) = if kmer <= rkmer {
            (hash64(kmer), Strand::Forward)
        } else {
            (hash64(rkmer), Strand::Reverse)
        };
        // Keep the rightmost representative of equal minima; emitting every
        // tied k-mer costs O(w) redundant seeds per low-complexity position.
        while len > 0 && ring[(head + len - 1) % RING_CAP].0 >= hash {
            len -= 1;
        }
        // `len < w <= RING_CAP` here: the eviction above dropped everything
        // with hash >= this one, and the front eviction keeps the live range
        // at most `w` wide, so this write cannot overrun the buffer.
        ring[(head + len) % RING_CAP] = (hash, kmer_index, pos, strand);
        len += 1;

        while len > 0 && ring[head].1 + w <= kmer_index {
            head = (head + 1) % RING_CAP;
            len -= 1;
        }

        if kmer_index + 1 >= w && len > 0 {
            let front = ring[head];
            if last_out.is_none_or(|last| last.0 != front.0 || last.1 != front.2) {
                mins.push(Minimizer {
                    hash: front.0,
                    pos: front.2,
                    strand: front.3,
                });
                last_out = Some((front.0, front.2));
            }
        }

        kmer_index += 1;
    }
}

/// Heap-deque fallback for `w > RING_CAP`.
fn minimizers_deque(seq: &[u8], cfg: &MinimizerConfig, mins: &mut Vec<Minimizer>) {
    use std::collections::VecDeque;

    let k = cfg.k as u32;
    let w = cfg.w as u32;
    let mask: u64 = (1u64 << (2 * cfg.k)) - 1;

    let mut deque: VecDeque<(u64, u32, u32, Strand)> = VecDeque::new();
    let mut kmer: u64 = 0;
    let mut rkmer: u64 = 0;
    let mut valid_len: u32 = 0;
    let mut kmer_index: u32 = 0;
    let mut last_out: Option<(u64, u32)> = None;

    for (i, &b) in seq.iter().enumerate() {
        let code = base_code(b);
        if code >= 4 {
            kmer = 0;
            rkmer = 0;
            valid_len = 0;
            kmer_index = 0;
            deque.clear();
            last_out = None;
            continue;
        }
        kmer = ((kmer << 2) | code as u64) & mask;
        rkmer = (rkmer >> 2) | (((3 - code) as u64) << ((k - 1) * 2));
        valid_len += 1;
        if valid_len < k {
            continue;
        }

        let pos = (i + 1 - cfg.k) as u32;
        let (hash, strand) = if kmer <= rkmer {
            (hash64(kmer), Strand::Forward)
        } else {
            (hash64(rkmer), Strand::Reverse)
        };
        while deque.back().is_some_and(|back| back.0 >= hash) {
            deque.pop_back();
        }
        deque.push_back((hash, kmer_index, pos, strand));
        while deque.front().is_some_and(|front| front.1 + w <= kmer_index) {
            deque.pop_front();
        }
        if kmer_index + 1 >= w
            && let Some(front) = deque.front()
            && last_out.is_none_or(|last| last.0 != front.0 || last.1 != front.2)
        {
            mins.push(Minimizer {
                hash: front.0,
                pos: front.2,
                strand: front.3,
            });
            last_out = Some((front.0, front.2));
        }
        kmer_index += 1;
    }
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

#[cfg(test)]
mod ring_equivalence_tests {
    use super::*;

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// The ring-buffer and heap-deque windows must emit identical minimizers;
    /// only `w > RING_CAP` reaches the deque path in production.
    #[test]
    fn ring_and_deque_windows_agree() {
        let alphabet = b"ACGTN";
        let mut rng = 0x1234_5678_9ABC_DEF0u64;
        for trial in 0..200 {
            let len = 1 + (xorshift(&mut rng) as usize % 400);
            // Include N runs so the reset path is exercised on both sides.
            let seq: Vec<u8> = (0..len)
                .map(|_| {
                    let r = xorshift(&mut rng) as usize;
                    if r.is_multiple_of(37) {
                        b'N'
                    } else {
                        alphabet[r % 4]
                    }
                })
                .collect();
            let k = 1 + (xorshift(&mut rng) as usize % 24);
            let w = 1 + (xorshift(&mut rng) as usize % 20);
            let cfg = MinimizerConfig { k, w };

            let mut ring = Vec::new();
            minimizers_into(&seq, &cfg, &mut ring);
            let mut deque = Vec::new();
            if k != 0 && w != 0 && k < 32 && seq.len() >= k {
                minimizers_deque(&seq, &cfg, &mut deque);
            }
            assert_eq!(ring.len(), deque.len(), "trial {trial}: k={k} w={w}");
            for (a, b) in ring.iter().zip(deque.iter()) {
                assert_eq!((a.hash, a.pos, a.strand), (b.hash, b.pos, b.strand),
                    "trial {trial}: k={k} w={w}");
            }
        }
    }

    #[test]
    fn output_buffer_is_cleared_before_reuse() {
        let cfg = MinimizerConfig { k: 5, w: 3 };
        let mut buf = Vec::new();
        minimizers_into(b"ACGTACGTACGTACGT", &cfg, &mut buf);
        let first = buf.clone();
        // Reusing a populated buffer (the whole point of the `_into` form) must
        // not append to the previous result.
        minimizers_into(b"ACGTACGTACGTACGT", &cfg, &mut buf);
        assert_eq!(buf.len(), first.len());
        // A sequence too short to sketch must leave an empty buffer, not stale data.
        minimizers_into(b"AC", &cfg, &mut buf);
        assert!(buf.is_empty());
    }
}
