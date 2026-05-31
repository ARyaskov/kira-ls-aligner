//! SimHash-LSH rescue path for reads the chain cascade can't anchor.
//!
//! Sister to [`super::cgk`]: where CGK handles indel-bearing reads, this
//! module handles 1-2 mismatch reads whose seed k-mers all land in the
//! wrong minimizer bucket. For each rescue, we probe a handful of read
//! windows against the prebuilt [`LshIndex`], verify the surviving
//! candidates with a full-read Hamming check, and return the best
//! ungapped alignment. WFA is never invoked from this path — if no
//! candidate clears the mismatch budget, the rescue returns `None` and
//! the cascade falls through to CGK / failure.
//!
//! Gated by `KIRA_LSH_ENABLE=1`. Off by default; when off the cascade
//! pays no extra cost.

use std::sync::{Arc, OnceLock};

use rustc_hash::FxHashSet;

use crate::index::lsh::LshIndex;
use crate::sketch::simhash::simhash_window;
use crate::types::{Alignment, AlignmentKind, CigarKind, CigarOp, MateInfo, Strand};

use super::{AlignmentConfig, push_u32_decimal};

/// Holder for the LSH side-index + the bases we need to verify candidates.
pub struct LshRescue {
    pub index: LshIndex,
    /// Owned per-contig base sequences. Same trade as [`super::cgk::CgkRescue`].
    pub ref_bases: Vec<Vec<u8>>,
    pub cfg: AlignmentConfig,
    /// Stop verifying after this many distinct (ref_id, ref_start) hits.
    pub max_candidates: usize,
    /// In-bucket Hamming filter applied to candidate fingerprints.
    pub max_lsh_hamming: u32,
    /// Per-read mismatch budget at the base level — candidates over this
    /// are discarded. Defaults are picked by the installer.
    pub max_window_mismatches: u32,
}

impl std::fmt::Debug for LshRescue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LshRescue")
            .field("index_entries", &self.index.entry_count())
            .field("index_buckets", &self.index.bucket_count())
            .field("n_contigs", &self.ref_bases.len())
            .field("max_candidates", &self.max_candidates)
            .field("max_lsh_hamming", &self.max_lsh_hamming)
            .field("max_window_mismatches", &self.max_window_mismatches)
            .finish()
    }
}

/// Process-global rescue, set once at pipeline startup.
pub static LSH_RESCUE_GLOBAL: OnceLock<Arc<LshRescue>> = OnceLock::new();

/// Install the rescue. Returns `Err` if already set.
pub fn set_global_rescue(rescue: LshRescue) -> Result<(), &'static str> {
    LSH_RESCUE_GLOBAL
        .set(Arc::new(rescue))
        .map_err(|_| "LSH rescue already set")
}

/// Cheap accessor — `None` until [`set_global_rescue`] succeeds.
#[inline]
pub fn global_rescue() -> Option<Arc<LshRescue>> {
    LSH_RESCUE_GLOBAL.get().cloned()
}

/// Runtime flag: is the LSH fallback enabled? Reads `KIRA_LSH_ENABLE` once.
#[inline]
pub fn lsh_enabled() -> bool {
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("KIRA_LSH_ENABLE")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .map(|v| v != 0)
            .unwrap_or(false)
    })
}

impl LshRescue {
    /// Run the rescue for a single read.
    pub fn rescue(&self, read_seq: &[u8], strand: Strand) -> Option<Alignment> {
        let read_len = read_seq.len();
        let win_len = self.index.window_len;
        if read_len < win_len {
            return None;
        }

        // Probe three non-overlapping windows in the read. Different read
        // positions land in different LSH buckets, so a mismatch flipping
        // the top bits of one probe doesn't sink the others. We dedupe the
        // resulting candidate set before doing the (expensive) per-base
        // verify, so the wider probe doesn't multiply work.
        let probes: [usize; 3] = if read_len >= win_len * 2 {
            [0, (read_len - win_len) / 2, read_len - win_len]
        } else {
            [0, 0, read_len - win_len]
        };

        let is_rev = matches!(strand, Strand::Reverse);
        let mut seen: FxHashSet<(u32, u32)> = FxHashSet::default();
        let mut cand_scratch: Vec<(u32, u32)> = Vec::new();
        let mut best: Option<Alignment> = None;
        let mut tried = 0usize;

        for &read_pos in &probes {
            if tried >= self.max_candidates {
                break;
            }
            let window = &read_seq[read_pos..read_pos + win_len];
            let Some(hash) = simhash_window(window) else {
                continue;
            };
            self.index
                .lookup_fuzzy_into(hash, self.max_lsh_hamming, &mut cand_scratch);

            for &(ref_id, ref_pos) in cand_scratch.iter() {
                if tried >= self.max_candidates {
                    break;
                }
                // The fingerprint matched at `ref_pos`; translate that to
                // the read's expected start on the reference.
                let global_ref_start = (ref_pos as i64) - (read_pos as i64);
                if global_ref_start < 0 {
                    continue;
                }
                let global_ref_start = global_ref_start as u32;
                if !seen.insert((ref_id, global_ref_start)) {
                    continue;
                }
                tried += 1;

                let bases = match self.ref_bases.get(ref_id as usize) {
                    Some(b) => b.as_slice(),
                    None => continue,
                };
                let s = global_ref_start as usize;
                if s + read_len > bases.len() {
                    continue;
                }
                let ref_slice = &bases[s..s + read_len];
                let mism = crate::simd::count_mismatches_bounded(
                    read_seq,
                    ref_slice,
                    self.max_window_mismatches,
                );
                if mism > self.max_window_mismatches {
                    continue;
                }
                let candidate = build_ungapped_alignment(
                    read_seq,
                    ref_slice,
                    ref_id,
                    global_ref_start,
                    self.cfg,
                    is_rev,
                    mism,
                );
                match &best {
                    None => best = Some(candidate),
                    Some(b) if candidate.score > b.score => best = Some(candidate),
                    _ => {}
                }
            }
        }
        best
    }
}

/// Build an ungapped `Alignment` for a (read, ref_slice) pair already known
/// to be within the per-read mismatch budget. Mirrors the layout used by
/// the spectral fast paths so downstream stages don't need a special case.
fn build_ungapped_alignment(
    read: &[u8],
    ref_slice: &[u8],
    ref_id: u32,
    ref_start: u32,
    cfg: AlignmentConfig,
    is_rev: bool,
    nm: u32,
) -> Alignment {
    let read_len = read.len();
    let cigar = vec![CigarOp {
        len: read_len as u32,
        op: CigarKind::Match,
    }];
    let mut md_bytes: Vec<u8> = Vec::with_capacity(16);
    let mut match_run: u32 = 0;
    for (qb, rb) in read.iter().zip(ref_slice.iter()) {
        if qb == rb {
            match_run += 1;
        } else {
            push_u32_decimal(&mut md_bytes, match_run);
            md_bytes.push(*rb);
            match_run = 0;
        }
    }
    push_u32_decimal(&mut md_bytes, match_run);
    // SAFETY: pushed bytes are ASCII digits and ACGTN bases.
    let md = unsafe { String::from_utf8_unchecked(md_bytes) };
    let matches = (read_len as u32 - nm) as i32;
    let mism = nm as i32;
    let score = matches * cfg.match_score - mism * cfg.mismatch;

    Alignment {
        kind: AlignmentKind::DpAligned,
        ref_id,
        ref_start,
        ref_end: ref_start + read_len as u32,
        read_start: 0,
        read_end: read_len as u32,
        cigar,
        score,
        mapq: 0,
        is_rev,
        is_secondary: false,
        is_supplementary: false,
        nm,
        md,
        as_score: score,
        xs_score: None,
        xs_strand: None,
        mate: MateInfo::default(),
    }
}

/// Cascade hook — `None` unless [`lsh_enabled`] returns true and a global
/// rescue has been installed.
pub fn try_lsh_fallback(read_seq: &[u8], strand: Strand) -> Option<Alignment> {
    if !lsh_enabled() {
        return None;
    }
    let rescue = global_rescue()?;
    rescue.rescue(read_seq, strand)
}

#[cfg(test)]
#[path = "../../tests/unit/alignment_lsh_rescue.rs"]
mod tests;
