//! CGK (Chakraborty-Goldenberg-Koucký 2018) edit-distance embedding.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use crate::types::{Alignment, Strand};

use super::wfa::{self, WfaPenalties};
use super::{AlignmentConfig, wfa_result_to_alignment};

/// Number of probe positions per fingerprint.
pub const PROBES_PER_FINGERPRINT: usize = 64;

/// Default number of fingerprint banks.
pub const DEFAULT_BANK_COUNT: usize = 16;

/// Deterministic SplitMix64 PRNG — same constants as `sketch::hash64` so behaviour is consistent.
#[derive(Clone, Copy)]
struct SplitMix64(u64);

impl SplitMix64 {
    #[inline]
    fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }
    #[inline]
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

/// Deterministic advice bit-string used by the CGK embedding.
#[derive(Clone, Debug)]
pub struct EmbeddingAdvice {
    bits: Vec<u64>,
    len_bits: usize,
}

impl EmbeddingAdvice {
    /// Build an advice of the given bit length from the given seed.
    pub fn new(seed: u64, len_bits: usize) -> Self {
        let n_words = len_bits.div_ceil(64);
        let mut bits = vec![0u64; n_words];
        let mut rng = SplitMix64::new(seed);
        for w in bits.iter_mut() {
            *w = rng.next();
        }
        Self { bits, len_bits }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len_bits
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len_bits == 0
    }

    #[inline]
    pub fn get(&self, idx: usize) -> u8 {
        debug_assert!(idx < self.len_bits);
        let w = idx >> 6;
        let b = idx & 63;
        ((self.bits[w] >> b) & 1) as u8
    }
}

/// 2-bit DNA code.
#[inline]
fn base_code_2bit(b: u8) -> u8 {
    match b {
        b'A' | b'a' => 0,
        b'C' | b'c' => 1,
        b'G' | b'g' => 2,
        b'T' | b't' => 3,
        _ => 0,
    }
}

/// Read bit `bit_idx` of the 2-bit-encoded input stream, which has length `2 * input.len()` bits.
#[inline]
fn read_input_bit(input: &[u8], bit_idx: usize) -> u8 {
    let byte_idx = bit_idx >> 1;
    if byte_idx >= input.len() {
        return 0;
    }
    let code = base_code_2bit(input[byte_idx]);
    (code >> (bit_idx & 1)) & 1
}

/// Set bit `bit_idx` in `out` to 1. The buffer must have enough words.
#[inline]
fn set_output_bit(out: &mut [u64], bit_idx: usize) {
    let w = bit_idx >> 6;
    let b = bit_idx & 63;
    out[w] |= 1u64 << b;
}

/// Read bit `bit_idx` from a packed bit buffer (zero-padded past length).
#[inline]
fn read_packed_bit(buf: &[u64], bit_idx: usize, len_bits: usize) -> u8 {
    if bit_idx >= len_bits {
        return 0;
    }
    let w = bit_idx >> 6;
    let b = bit_idx & 63;
    ((buf[w] >> b) & 1) as u8
}

/// Embed `input` into `output` using the given advice.
pub fn embed(input: &[u8], advice: &EmbeddingAdvice, output: &mut [u64]) {
    let n_in_bits = 2 * input.len();
    let n_out_bits = 3 * n_in_bits; // = 6 * input.len()
    let n_words = n_out_bits.div_ceil(64);
    assert!(
        output.len() >= n_words,
        "output must hold {} u64 words ({} bits) — got {}",
        n_words,
        n_out_bits,
        output.len()
    );
    assert!(
        advice.len() >= n_out_bits,
        "advice too short: {} bits, need {}",
        advice.len(),
        n_out_bits
    );
    for w in &mut output[..n_words] {
        *w = 0;
    }

    let mut j_in = 0usize;
    for j_out in 0..n_out_bits {
        let x_bit = read_input_bit(input, j_in);
        if x_bit != 0 {
            set_output_bit(output, j_out);
        }
        let r_bit = advice.get(j_out);
        let advance = if j_in < n_in_bits {
            r_bit ^ x_bit
        } else {
            1
        };
        j_in += advance as usize;
    }
}

/// One Hamming-LSH bank: a fixed set of `PROBES_PER_FINGERPRINT` probe positions inside the.
#[derive(Clone, Debug)]
pub struct FingerprintBank {
    probes: [u32; PROBES_PER_FINGERPRINT],
}

impl FingerprintBank {
    /// Build a bank whose probes are uniformly sampled over `[0.
    pub fn new(seed: u64, embed_bits: usize) -> Self {
        assert!(
            embed_bits >= PROBES_PER_FINGERPRINT,
            "embed_bits must accommodate the probes ({} < {})",
            embed_bits,
            PROBES_PER_FINGERPRINT
        );
        let mut rng = SplitMix64::new(seed);
        let mut probes = [0u32; PROBES_PER_FINGERPRINT];
        for slot in probes.iter_mut() {
            *slot = (rng.next() % embed_bits as u64) as u32;
        }
        Self { probes }
    }

    /// Compute the fingerprint of `embedded` for this bank.
    pub fn fingerprint(&self, embedded: &[u64], embed_bits: usize) -> u64 {
        let mut fp = 0u64;
        for (i, &probe) in self.probes.iter().enumerate() {
            let bit = read_packed_bit(embedded, probe as usize, embed_bits) as u64;
            fp |= bit << i;
        }
        fp
    }
}

/// Set of `L` independent fingerprint banks, all sharing one embedding advice.
#[derive(Clone, Debug)]
pub struct FingerprintScheme {
    pub advice: EmbeddingAdvice,
    pub banks: Vec<FingerprintBank>,
    /// Input length in *bytes*. Embedded length is `6 × input_len` bits.
    pub input_len: usize,
}

impl FingerprintScheme {
    /// Build a scheme for inputs of fixed length `input_len` bytes.
    pub fn new(seed: u64, input_len: usize, n_banks: usize) -> Self {
        let embed_bits = 6 * input_len;
        let advice = EmbeddingAdvice::new(seed ^ 0xA11CE_5EED_u64, embed_bits);
        let mut banks = Vec::with_capacity(n_banks);
        for b in 0..n_banks {
            let bank_seed = seed.wrapping_mul(0x100000001b3).wrapping_add(b as u64);
            banks.push(FingerprintBank::new(bank_seed, embed_bits));
        }
        Self {
            advice,
            banks,
            input_len,
        }
    }

    #[inline]
    pub fn embed_bits(&self) -> usize {
        6 * self.input_len
    }

    #[inline]
    pub fn embed_words(&self) -> usize {
        self.embed_bits().div_ceil(64)
    }

    /// Embed `input` (length `self.input_len` bytes) into the caller-owned scratch buffer and.
    pub fn fingerprints_into(
        &self,
        input: &[u8],
        embed_scratch: &mut Vec<u64>,
        out: &mut Vec<u64>,
    ) {
        assert_eq!(input.len(), self.input_len);
        let n_words = self.embed_words();
        embed_scratch.clear();
        embed_scratch.resize(n_words, 0);
        embed(input, &self.advice, embed_scratch);
        let embed_bits = self.embed_bits();
        out.clear();
        out.reserve(self.banks.len());
        for bank in &self.banks {
            out.push(bank.fingerprint(embed_scratch, embed_bits));
        }
    }

    /// Convenience: allocate scratch and return the fingerprint vector.
    pub fn fingerprints(&self, input: &[u8]) -> Vec<u64> {
        let mut embed_scratch = Vec::new();
        let mut out = Vec::new();
        self.fingerprints_into(input, &mut embed_scratch, &mut out);
        out
    }
}

/// Pack (ref_id, position) into a single u64.
#[inline]
fn pack_window(ref_id: u32, pos: u32) -> u64 {
    ((ref_id as u64) << 32) | (pos as u64)
}

#[inline]
fn unpack_window(packed: u64) -> (u32, u32) {
    ((packed >> 32) as u32, packed as u32)
}

/// Lookup table from fingerprint → list of (ref_id, pos) windows that produced that fingerprint.
#[derive(Debug)]
pub struct CgkIndex {
    /// `tables[b]` maps a fingerprint produced by `scheme.banks[b]` to the list of windows.
    tables: Vec<HashMap<u64, Vec<u64>>>,
    /// Window stride used during build.
    pub stride: usize,
    /// The scheme used to build the index.
    pub scheme: FingerprintScheme,
}

impl CgkIndex {
    /// Build an empty index for the given scheme + stride.
    pub fn new(scheme: FingerprintScheme, stride: usize) -> Self {
        let tables = (0..scheme.banks.len())
            .map(|_| HashMap::<u64, Vec<u64>>::new())
            .collect();
        Self {
            tables,
            stride,
            scheme,
        }
    }

    /// Number of banks.
    #[inline]
    pub fn bank_count(&self) -> usize {
        self.tables.len()
    }

    /// Total number of (fingerprint, window) entries across all banks.
    pub fn entry_count(&self) -> usize {
        self.tables
            .iter()
            .map(|t| t.values().map(|v| v.len()).sum::<usize>())
            .sum()
    }

    /// Insert one reference window.
    pub fn insert_window(&mut self, ref_id: u32, pos: u32, window: &[u8]) {
        if window.len() != self.scheme.input_len {
            return;
        }
        let mut scratch_embed = Vec::new();
        let mut scratch_fp = Vec::new();
        self.scheme
            .fingerprints_into(window, &mut scratch_embed, &mut scratch_fp);
        let packed = pack_window(ref_id, pos);
        for (bank_idx, fp) in scratch_fp.iter().enumerate() {
            self.tables[bank_idx].entry(*fp).or_default().push(packed);
        }
    }

    /// Build by walking every contig in `(ref_id, bases)` order at the given stride.
    pub fn build_from_sequences<'a, I>(scheme: FingerprintScheme, stride: usize, seqs: I) -> Self
    where
        I: IntoIterator<Item = (u32, &'a [u8])>,
    {
        let mut idx = Self::new(scheme, stride);
        let input_len = idx.scheme.input_len;
        let mut scratch_embed = Vec::new();
        let mut scratch_fp = Vec::new();
        for (ref_id, bases) in seqs.into_iter() {
            if bases.len() < input_len {
                continue;
            }
            let n_windows = (bases.len() - input_len) / stride + 1;
            for w in 0..n_windows {
                let start = w * stride;
                let window = &bases[start..start + input_len];
                idx.scheme
                    .fingerprints_into(window, &mut scratch_embed, &mut scratch_fp);
                let packed = pack_window(ref_id, start as u32);
                for (bank_idx, fp) in scratch_fp.iter().enumerate() {
                    idx.tables[bank_idx].entry(*fp).or_default().push(packed);
                }
            }
        }
        idx
    }

    /// Query: produce a deduplicated list of candidate windows whose fingerprint matched the read.
    pub fn query(&self, read: &[u8], min_bank_hits: u32) -> Vec<(u32, u32)> {
        assert_eq!(read.len(), self.scheme.input_len);
        let mut scratch_embed = Vec::new();
        let mut scratch_fp = Vec::new();
        self.scheme
            .fingerprints_into(read, &mut scratch_embed, &mut scratch_fp);
        let mut hits: HashMap<u64, u32> = HashMap::new();
        for (bank_idx, fp) in scratch_fp.iter().enumerate() {
            if let Some(windows) = self.tables[bank_idx].get(fp) {
                for &packed in windows {
                    *hits.entry(packed).or_insert(0) += 1;
                }
            }
        }
        hits.into_iter()
            .filter(|&(_, c)| c >= min_bank_hits)
            .map(|(packed, _)| unpack_window(packed))
            .collect()
    }
}

/// Per-rescue state.
pub struct CgkRescue {
    /// CGK side-index.
    pub index: CgkIndex,
    /// Owned base sequences per contig.
    pub ref_bases: Vec<Vec<u8>>,
    /// Alignment scoring configuration the rescue runs WFA with.
    pub cfg: AlignmentConfig,
    /// Max candidate windows to verify per rescue call.
    pub max_candidates: usize,
    /// Minimum bank hits to consider a candidate.
    pub min_bank_hits: u32,
}

impl std::fmt::Debug for CgkRescue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CgkRescue")
            .field("index_entries", &self.index.entry_count())
            .field("n_contigs", &self.ref_bases.len())
            .field("max_candidates", &self.max_candidates)
            .field("min_bank_hits", &self.min_bank_hits)
            .finish()
    }
}

/// Process-global CGK rescue, set once at pipeline startup when CGK is enabled.
pub static CGK_RESCUE_GLOBAL: OnceLock<Arc<CgkRescue>> = OnceLock::new();

/// Install the rescue.
pub fn set_global_rescue(rescue: CgkRescue) -> Result<(), &'static str> {
    CGK_RESCUE_GLOBAL
        .set(Arc::new(rescue))
        .map_err(|_| "CGK rescue already set")
}

/// Cheap accessor for the installed rescue (returns `None` until `set_global_rescue` succeeds).
#[inline]
pub fn global_rescue() -> Option<Arc<CgkRescue>> {
    CGK_RESCUE_GLOBAL.get().cloned()
}

/// Runtime flag: is CGK fallback enabled? Reads `KIRA_CGK_ENABLE` once.
#[inline]
pub fn cgk_enabled() -> bool {
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("KIRA_CGK_ENABLE")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .map(|v| v != 0)
            .unwrap_or(false)
    })
}

impl CgkRescue {
    /// Run the rescue for a single read.
    pub fn rescue(&self, read_seq: &[u8], strand: Strand) -> Option<Alignment> {
        if read_seq.len() != self.index.scheme.input_len {
            return None;
        }
        let candidates = self.index.query(read_seq, self.min_bank_hits);
        if candidates.is_empty() {
            return None;
        }

        let pen = WfaPenalties {
            mismatch: self.cfg.mismatch,
            gap_open: self.cfg.gap_open,
            gap_extend: self.cfg.gap_extend,
        };
        let read_len = read_seq.len();
        let pad = self.cfg.bandwidth.max(1) as usize;
        let budget = super::router::wfa_score_budget(
            read_len,
            pen.mismatch,
            pen.gap_open,
            pen.gap_extend,
        );

        let is_rev = matches!(strand, Strand::Reverse);
        let mut best: Option<Alignment> = None;
        for (ref_id, pos) in candidates.into_iter().take(self.max_candidates) {
            let bases = match self.ref_bases.get(ref_id as usize) {
                Some(b) => b.as_slice(),
                None => continue,
            };
            let win_start = pos as usize;
            if win_start >= bases.len() {
                continue;
            }
            let max_text_len = bases.len() - win_start;
            let text_len = (read_len + pad).min(max_text_len);
            if text_len < read_len {
                continue;
            }
            let text = &bases[win_start..win_start + text_len];

            let wfa_aln = match wfa::wfa_align_semi_global(read_seq, text, pen, budget) {
                Some(a) => a,
                None => continue,
            };
            let candidate = match wfa_result_to_alignment(
                read_seq, text, win_start, ref_id, wfa_aln, self.cfg, is_rev,
            ) {
                Some(a) => a,
                None => continue,
            };
            match &best {
                None => best = Some(candidate),
                Some(b) if candidate.score > b.score => best = Some(candidate),
                _ => {}
            }
        }
        best
    }
}

/// Try CGK rescue from the cascade.
pub fn try_cgk_fallback(read_seq: &[u8], strand: Strand) -> Option<Alignment> {
    if !cgk_enabled() {
        return None;
    }
    let rescue = global_rescue()?;
    rescue.rescue(read_seq, strand)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng_byte(state: &mut u64) -> u8 {
        *state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        let codes = b"ACGT";
        codes[(z & 3) as usize]
    }

    fn random_dna(len: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        (0..len).map(|_| rng_byte(&mut s)).collect()
    }

    fn hamming_bits(a: &[u64], b: &[u64], n_bits: usize) -> usize {
        let n_full = n_bits / 64;
        let tail_bits = n_bits & 63;
        let mut h = 0usize;
        for i in 0..n_full {
            h += (a[i] ^ b[i]).count_ones() as usize;
        }
        if tail_bits > 0 {
            let mask = (1u64 << tail_bits) - 1;
            h += ((a[n_full] ^ b[n_full]) & mask).count_ones() as usize;
        }
        h
    }

    #[test]
    fn advice_is_deterministic() {
        let a = EmbeddingAdvice::new(42, 900);
        let b = EmbeddingAdvice::new(42, 900);
        for i in 0..900 {
            assert_eq!(a.get(i), b.get(i), "mismatch at bit {}", i);
        }
    }

    #[test]
    fn advice_bit_distribution_is_balanced() {
        let a = EmbeddingAdvice::new(0xCAFE, 900);
        let ones = (0..900).filter(|&i| a.get(i) != 0).count();
        assert!(
            ones >= 900 * 35 / 100 && ones <= 900 * 65 / 100,
            "advice 1-bit count {} outside [35%, 65%]",
            ones
        );
    }

    #[test]
    fn embed_is_deterministic() {
        let advice = EmbeddingAdvice::new(7, 900);
        let input = random_dna(150, 1);
        let n_words = (6 * 150usize).div_ceil(64);
        let mut out_a = vec![0u64; n_words];
        let mut out_b = vec![0u64; n_words];
        embed(&input, &advice, &mut out_a);
        embed(&input, &advice, &mut out_b);
        assert_eq!(out_a, out_b);
    }

    #[test]
    fn embed_preserves_first_two_bits() {
        let advice = EmbeddingAdvice::new(99, 900);
        let input = random_dna(150, 2);
        let n_words = (6 * 150usize).div_ceil(64);
        let mut out = vec![0u64; n_words];
        embed(&input, &advice, &mut out);
        let first_input_bit = read_input_bit(&input, 0);
        let first_output_bit = read_packed_bit(&out, 0, 900);
        assert_eq!(
            first_output_bit, first_input_bit,
            "first output bit ({}) != first input bit ({})",
            first_output_bit, first_input_bit
        );
    }

    #[test]
    fn identical_inputs_produce_identical_embeddings() {
        let advice = EmbeddingAdvice::new(11, 900);
        let input = random_dna(150, 3);
        let n_words = (6 * 150usize).div_ceil(64);
        let mut a = vec![0u64; n_words];
        let mut b = vec![0u64; n_words];
        embed(&input, &advice, &mut a);
        embed(&input, &advice, &mut b);
        assert_eq!(hamming_bits(&a, &b, 900), 0);
    }

    fn apply_one_insertion(s: &[u8], pos: usize, base: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity(s.len());
        v.extend_from_slice(&s[..pos]);
        v.push(base);
        v.extend_from_slice(&s[pos..s.len() - 1]);
        v
    }

    fn apply_one_deletion(s: &[u8], pos: usize, pad: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity(s.len());
        v.extend_from_slice(&s[..pos]);
        v.extend_from_slice(&s[pos + 1..]);
        v.push(pad);
        v
    }

    fn apply_one_substitution(s: &[u8], pos: usize, base: u8) -> Vec<u8> {
        let mut v = s.to_vec();
        v[pos] = base;
        v
    }

    #[test]
    fn edit_distance_one_substitution_keeps_hamming_bounded() {
        let mut max_h = 0usize;
        for seed in [2024u64, 31, 99, 17, 123456] {
            let advice = EmbeddingAdvice::new(seed, 900);
            let x = random_dna(150, seed.wrapping_mul(13));
            let new_base = if x[70] == b'G' { b'C' } else { b'G' };
            let y = apply_one_substitution(&x, 70, new_base);
            let n_words = (6 * 150usize).div_ceil(64);
            let mut ex = vec![0u64; n_words];
            let mut ey = vec![0u64; n_words];
            embed(&x, &advice, &mut ex);
            embed(&y, &advice, &mut ey);
            let h = hamming_bits(&ex, &ey, 900);
            assert!(h > 0, "1-substitution produced 0 embedded Hamming");
            if h > max_h {
                max_h = h;
            }
        }
        assert!(
            max_h <= 120,
            "max embedded Hamming for 1-substitution across seeds = {}, expected ≤ 120",
            max_h
        );
    }

    #[test]
    fn edit_distance_one_insertion_gives_small_hamming() {
        let mut max_h = 0usize;
        for seed in [2024u64, 31, 99, 17, 123456] {
            let advice = EmbeddingAdvice::new(seed, 900);
            let x = random_dna(150, seed.wrapping_mul(7));
            let y = apply_one_insertion(&x, 73, b'G');
            let n_words = (6 * 150usize).div_ceil(64);
            let mut ex = vec![0u64; n_words];
            let mut ey = vec![0u64; n_words];
            embed(&x, &advice, &mut ex);
            embed(&y, &advice, &mut ey);
            let h = hamming_bits(&ex, &ey, 900);
            if h > max_h {
                max_h = h;
            }
        }
        assert!(
            max_h <= 80,
            "max embedded Hamming for 1-insertion across seeds = {}, expected ≤ 80",
            max_h
        );
    }

    #[test]
    fn edit_distance_one_deletion_gives_small_hamming() {
        let mut max_h = 0usize;
        for seed in [2024u64, 31, 99, 17, 123456] {
            let advice = EmbeddingAdvice::new(seed, 900);
            let x = random_dna(150, seed.wrapping_mul(7));
            let y = apply_one_deletion(&x, 60, b'A');
            let n_words = (6 * 150usize).div_ceil(64);
            let mut ex = vec![0u64; n_words];
            let mut ey = vec![0u64; n_words];
            embed(&x, &advice, &mut ex);
            embed(&y, &advice, &mut ey);
            let h = hamming_bits(&ex, &ey, 900);
            if h > max_h {
                max_h = h;
            }
        }
        assert!(
            max_h <= 80,
            "max embedded Hamming for 1-deletion across seeds = {}, expected ≤ 80",
            max_h
        );
    }

    #[test]
    fn fingerprint_is_deterministic() {
        let scheme = FingerprintScheme::new(31415, 150, 16);
        let input = random_dna(150, 5);
        let a = scheme.fingerprints(&input);
        let b = scheme.fingerprints(&input);
        assert_eq!(a, b);
        assert_eq!(a.len(), 16);
    }

    #[test]
    fn fingerprint_changes_on_substitution_at_probe_pos() {
        let scheme = FingerprintScheme::new(271828, 150, 16);
        let x = random_dna(150, 17);
        let base_fps = scheme.fingerprints(&x);
        let mut any_change = false;
        for pos in 0..150 {
            let new_base = if x[pos] == b'A' { b'T' } else { b'A' };
            let y = apply_one_substitution(&x, pos, new_base);
            let new_fps = scheme.fingerprints(&y);
            if new_fps != base_fps {
                any_change = true;
                break;
            }
        }
        assert!(
            any_change,
            "no single substitution changed any fingerprint — \
            probe positions cover none of the input?"
        );
    }

    #[test]
    fn index_recalls_inserted_window() {
        let scheme = FingerprintScheme::new(2024, 150, 16);
        let ref_bases = random_dna(10_000, 42);
        let idx = CgkIndex::build_from_sequences(
            scheme,
            30,
            std::iter::once((0u32, ref_bases.as_slice())),
        );
        let win = &ref_bases[300..450];
        let candidates = idx.query(win, 1);
        assert!(
            candidates.iter().any(|&(rid, pos)| rid == 0 && pos == 300),
            "self-query did not recall its own window; candidates = {:?}",
            candidates
        );
    }

    #[test]
    fn index_recalls_indel_variant_window() {
        let scheme = FingerprintScheme::new(2024, 150, 16);
        let ref_bases = random_dna(20_000, 99);
        let idx = CgkIndex::build_from_sequences(
            scheme,
            30,
            std::iter::once((0u32, ref_bases.as_slice())),
        );
        let mut hits = 0;
        let trials = [(600usize, 75usize), (900, 30), (1200, 100), (1500, 60), (1800, 90)];
        for &(true_pos, edit_pos) in &trials {
            let win = &ref_bases[true_pos..true_pos + 150];
            let edited = apply_one_insertion(win, edit_pos, b'C');
            let candidates = idx.query(&edited, 1);
            if candidates
                .iter()
                .any(|&(rid, pos)| rid == 0 && pos == true_pos as u32)
            {
                hits += 1;
            }
        }
        assert!(
            hits >= 4,
            "indel-edited recall: {} / {} trials hit; expected ≥ 4",
            hits,
            trials.len()
        );
    }

    #[test]
    fn pack_unpack_roundtrips() {
        for &(rid, pos) in &[(0u32, 0u32), (1, 100), (12345, 67890), (u32::MAX, u32::MAX)] {
            let (r, p) = unpack_window(pack_window(rid, pos));
            assert_eq!((r, p), (rid, pos));
        }
    }
}
