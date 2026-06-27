//! Paired-end post-processing for stage 4 alignments.

use rayon::prelude::*;

use crate::alignment::{AlignmentConfig, align_in_window};
use crate::index::Index;
use crate::io::IngestMode;
use crate::seq::reverse_complement;
use crate::types::{Alignment, MateInfo, PairRole, ReadRecord, Reference};

/// Paired-end configuration (mode + insert-size policy).
#[derive(Clone, Copy, Debug)]
pub struct PairedConfig {
    pub mode: IngestMode,
    pub insert_min: u32,
    pub insert_max: u32,
    pub insert_mean: u32,
    pub insert_sd: u32,
    pub estimator_locked: bool,
}

impl Default for PairedConfig {
    fn default() -> Self {
        Self {
            mode: IngestMode::Unpaired,
            insert_min: 0,
            insert_max: 1000,
            insert_mean: 200,
            insert_sd: 50,
            estimator_locked: false,
        }
    }
}

impl PairedConfig {
    /// `true` when the configuration enables paired processing.
    #[inline]
    pub fn is_paired(&self) -> bool {
        !matches!(self.mode, IngestMode::Unpaired)
    }

    /// Parse the `-I MIN,MAX[,MEAN,SD]` CLI string into the four numeric fields.
    pub fn apply_insert_spec(&mut self, spec: &str) -> Result<(), String> {
        let parts: Vec<&str> = spec.split(',').map(|s| s.trim()).collect();
        let parse = |s: &str| -> Result<u32, String> {
            s.parse::<u32>()
                .map_err(|_| format!("invalid insert-size component {s:?}"))
        };
        match parts.len() {
            2 => {
                self.insert_min = parse(parts[0])?;
                self.insert_max = parse(parts[1])?;
            }
            4 => {
                self.insert_min = parse(parts[0])?;
                self.insert_max = parse(parts[1])?;
                self.insert_mean = parse(parts[2])?;
                self.insert_sd = parse(parts[3])?;
            }
            n => {
                return Err(format!(
                    "insert-size spec must have 2 (MIN,MAX) or 4 (MIN,MAX,MEAN,SD) values, got {n}"
                ));
            }
        }
        if self.insert_min > self.insert_max {
            return Err(format!(
                "insert-size MIN ({}) > MAX ({})",
                self.insert_min, self.insert_max
            ));
        }
        Ok(())
    }
}

/// Decorate stage-4 alignments with paired-end metadata in place.
pub fn apply_pairing(
    reads: &[ReadRecord],
    alignments: &mut [Vec<Alignment>],
    unmapped_mate_info: &mut [Option<MateInfo>],
    cfg: &PairedConfig,
) {
    if !cfg.is_paired() {
        return;
    }
    debug_assert_eq!(reads.len(), alignments.len());
    debug_assert_eq!(reads.len(), unmapped_mate_info.len());
    let mut i = 0;
    while i + 1 < reads.len() {
        let r1 = &reads[i];
        let r2 = &reads[i + 1];
        if r1.pair_role == PairRole::Unpaired || r2.pair_role == PairRole::Unpaired {
            i += 1;
            continue;
        }
        debug_assert_eq!(r1.pair_role, PairRole::R1);
        debug_assert_eq!(r2.pair_role, PairRole::R2);

        let r1_primary: Option<MatePrimary> = primary_summary(&alignments[i]);
        let r2_primary: Option<MatePrimary> = primary_summary(&alignments[i + 1]);

        // Decide proper-pair + TLEN once per pair from the primaries.
        let (proper, tlen1, tlen2) = match (r1_primary, r2_primary) {
            (Some(p1), Some(p2)) => classify_pair(&p1, &p2, cfg),
            _ => (false, 0, 0),
        };

        decorate_side(
            &mut alignments[i],
            PairRole::R1,
            r2_primary.as_ref(),
            proper,
            tlen1,
        );
        decorate_side(
            &mut alignments[i + 1],
            PairRole::R2,
            r1_primary.as_ref(),
            proper,
            tlen2,
        );

        if alignments[i].is_empty() {
            unmapped_mate_info[i] = Some(build_unmapped_mate(
                PairRole::R1,
                r2_primary.as_ref(),
                proper,
            ));
        }
        if alignments[i + 1].is_empty() {
            unmapped_mate_info[i + 1] = Some(build_unmapped_mate(
                PairRole::R2,
                r1_primary.as_ref(),
                proper,
            ));
        }

        i += 2;
    }
}

/// Compact summary of a read's primary alignment used during pairing.
#[derive(Clone, Copy, Debug)]
struct MatePrimary {
    ref_id: u32,
    ref_start: u32,
    ref_end: u32,
    is_rev: bool,
}

/// Mate-rescue threshold knobs.
#[derive(Clone, Copy, Debug)]
pub struct RescueConfig {
    pub min_anchor_score: i32,
    pub min_rescued_score: i32,
}

impl Default for RescueConfig {
    fn default() -> Self {
        Self {
            min_anchor_score: 100,
            min_rescued_score: 80,
        }
    }
}

/// Mate rescue: for paired reads where exactly one mate failed chaining but the other has a.
pub fn rescue_unmapped_mates(
    reads: &[ReadRecord],
    alignments: &mut [Vec<Alignment>],
    index: &Index,
    cfg: &PairedConfig,
    align_cfg: AlignmentConfig,
    rescue_cfg: RescueConfig,
) {
    let mmap_bytes: Option<&[u8]> = index.mmap.as_deref().map(|m| &m[..]);
    rescue_unmapped_mates_with_ref(
        reads,
        alignments,
        &index.reference,
        mmap_bytes,
        cfg,
        align_cfg,
        rescue_cfg,
    );
}

/// Same as [`rescue_unmapped_mates`] but driven by a `&Reference` (plus optional mmap backing).
pub fn rescue_unmapped_mates_with_ref(
    reads: &[ReadRecord],
    alignments: &mut [Vec<Alignment>],
    reference: &Reference,
    mmap: Option<&[u8]>,
    cfg: &PairedConfig,
    align_cfg: AlignmentConfig,
    rescue_cfg: RescueConfig,
) {
    if !cfg.is_paired() {
        return;
    }
    debug_assert_eq!(reads.len(), alignments.len());

    // Pairs are independent — parallelize over R1/R2 chunks (per-pair logic unchanged).
    alignments
        .par_chunks_mut(2)
        .zip(reads.par_chunks(2))
        .for_each(|(aln_pair, read_pair)| {
            if aln_pair.len() < 2 || read_pair.len() < 2 {
                return;
            }
            if read_pair[0].pair_role == PairRole::Unpaired
                || read_pair[1].pair_role == PairRole::Unpaired
            {
                return;
            }

            // anchor = mapped side, target = empty side
            let (anchor_local, target_local) =
                match (aln_pair[0].is_empty(), aln_pair[1].is_empty()) {
                    (true, false) => (1usize, 0usize),
                    (false, true) => (0usize, 1usize),
                    _ => return,
                };

            let anchor_score = aln_pair[anchor_local][0].score;
            if anchor_score < rescue_cfg.min_anchor_score {
                return;
            }
            let anchor_ref_id = aln_pair[anchor_local][0].ref_id;
            let anchor_ref_start = aln_pair[anchor_local][0].ref_start;
            let anchor_ref_end = aln_pair[anchor_local][0].ref_end;
            let anchor_is_rev = aln_pair[anchor_local][0].is_rev;

            let ref_seq = reference.sequences[anchor_ref_id as usize].bases(mmap);
            let ref_len = ref_seq.len() as i64;
            let (win_start_u64, win_end_u64) = if cfg.estimator_locked && cfg.insert_sd > 0 {
                let mean = cfg.insert_mean as i64;
                let half = 3 * cfg.insert_sd as i64;
                if !anchor_is_rev {
                    let centre = anchor_ref_end as i64 + mean;
                    (
                        (centre - half).max(0).min(ref_len) as u64,
                        (centre + half).max(0).min(ref_len) as u64,
                    )
                } else {
                    let centre = anchor_ref_start as i64 - mean;
                    (
                        (centre - half).max(0).min(ref_len) as u64,
                        (centre + half).max(0).min(ref_len) as u64,
                    )
                }
            } else {
                let insert_max = cfg.insert_max as u64;
                if !anchor_is_rev {
                    (
                        anchor_ref_end as u64,
                        (anchor_ref_end as u64 + insert_max).min(ref_len as u64),
                    )
                } else {
                    (
                        (anchor_ref_start as u64).saturating_sub(insert_max),
                        anchor_ref_start as u64,
                    )
                }
            };
            if win_end_u64 <= win_start_u64 {
                return;
            }
            let win_start = win_start_u64 as u32;
            let ref_window = &ref_seq[win_start as usize..win_end_u64 as usize];

            // Expected mate strand: opposite to the anchor.
            let target_is_rev = !anchor_is_rev;
            let target_seq_owned;
            let target_seq: &[u8] = if target_is_rev {
                target_seq_owned = reverse_complement(&read_pair[target_local].seq);
                &target_seq_owned
            } else {
                &read_pair[target_local].seq
            };

            if let Some(mut aln) = align_in_window(
                target_seq,
                ref_window,
                win_start,
                anchor_ref_id,
                target_is_rev,
                align_cfg,
                rescue_cfg.min_rescued_score,
            ) {
                // Mark as rescue-placed so MAPQ assignment ceiling-caps it; the
                // old `aln.mapq = 30` here was a dead store (stage-5 recomputes MAPQ).
                aln.kind = crate::types::AlignmentKind::Rescued;
                aln_pair[target_local].push(aln);
            }
        });
}

fn primary_summary(alns: &[Alignment]) -> Option<MatePrimary> {
    alns.first().map(|a| MatePrimary {
        ref_id: a.ref_id,
        ref_start: a.ref_start,
        ref_end: a.ref_end,
        is_rev: a.is_rev,
    })
}

/// Classify a pair given both primaries.
fn classify_pair(r1: &MatePrimary, r2: &MatePrimary, cfg: &PairedConfig) -> (bool, i32, i32) {
    if r1.ref_id != r2.ref_id {
        return (false, 0, 0);
    }
    // Fragment span = [min(r1.start, r2.start), max(r1.end, r2.end))
    let frag_lo = r1.ref_start.min(r2.ref_start);
    let frag_hi = r1.ref_end.max(r2.ref_end);
    let frag_len = frag_hi.saturating_sub(frag_lo);

    let r1_is_leftmost = r1.ref_start <= r2.ref_start;
    let tlen1 = if r1_is_leftmost {
        frag_len as i32
    } else {
        -(frag_len as i32)
    };
    let tlen2 = -tlen1;

    let opposite_strand = r1.is_rev != r2.is_rev;
    let convergent = if r1_is_leftmost {
        // Leftmost should be forward, rightmost reverse (FR / "innie")
        !r1.is_rev && r2.is_rev
    } else {
        !r2.is_rev && r1.is_rev
    };
    let (win_lo, win_hi) = proper_pair_window(cfg);
    let in_window = frag_len >= win_lo && frag_len <= win_hi;
    let proper = opposite_strand && convergent && in_window;

    (proper, tlen1, tlen2)
}

/// Stamp pair fields onto every alignment for one side of the pair.
fn decorate_side(
    alns: &mut [Alignment],
    role: PairRole,
    mate_primary: Option<&MatePrimary>,
    proper_pair: bool,
    tlen: i32,
) {
    let (is_r1, is_r2) = match role {
        PairRole::R1 => (true, false),
        PairRole::R2 => (false, true),
        PairRole::Unpaired => (false, false),
    };

    let (mate_ref_id, mate_pos, mate_is_rev, mate_is_unmapped) = match mate_primary {
        Some(m) => (Some(m.ref_id), m.ref_start, m.is_rev, false),
        None => (None, 0, false, true),
    };

    for aln in alns.iter_mut() {
        aln.mate = MateInfo {
            is_paired: true,
            is_proper_pair: proper_pair && !mate_is_unmapped,
            mate_is_unmapped,
            mate_is_rev,
            is_first_in_pair: is_r1,
            is_second_in_pair: is_r2,
            mate_ref_id,
            mate_pos,
            tlen,
        };
    }
}

/// Re-rank pair candidates by joint score.
pub fn pair_rerank(
    reads: &[ReadRecord],
    alignments: &mut [Vec<Alignment>],
    cfg: &PairedConfig,
    top_k: usize,
) {
    if !cfg.is_paired() || top_k == 0 {
        return;
    }
    debug_assert_eq!(reads.len(), alignments.len());

    // Concordant-promote: widen candidate window and allow primary swap + MAPQ lift.
    let promote = pair_promote_enabled();
    let top_k = if promote {
        pair_promote_topk().max(top_k)
    } else {
        top_k
    };
    let mapq_floor = if promote { pair_mapq_floor_cfg() } else { 0 };

    let pair_bonus_base: i32 = PAIR_BONUS_BASE;
    let pair_bonus_concordant: i32 = pair_bonus_concordant_cfg();

    let mut i = 0;
    while i + 1 < reads.len() {
        let r1_idx = i;
        let r2_idx = i + 1;
        let r1 = &reads[r1_idx];
        let r2 = &reads[r2_idx];
        if r1.pair_role == PairRole::Unpaired || r2.pair_role == PairRole::Unpaired {
            i += 1;
            continue;
        }

        let r1_k = alignments[r1_idx].len().min(top_k);
        let r2_k = alignments[r2_idx].len().min(top_k);
        if r1_k == 0 || r2_k == 0 {
            i += 2;
            continue;
        }

        let r1_snap: Vec<MatePrimaryScored> = alignments[r1_idx][..r1_k]
            .iter()
            .map(MatePrimaryScored::from_aln)
            .collect();
        let r2_snap: Vec<MatePrimaryScored> = alignments[r2_idx][..r2_k]
            .iter()
            .map(MatePrimaryScored::from_aln)
            .collect();

        let mut best: Option<(usize, usize, i32, i32)> = None; // (i, j, joint, bonus)
        for (ii, a1) in r1_snap.iter().enumerate() {
            for (jj, a2) in r2_snap.iter().enumerate() {
                let concordant = pair_is_concordant(a1, a2, cfg);
                let bonus = if concordant {
                    pair_bonus_base + pair_bonus_concordant
                } else {
                    0
                };
                let joint = a1.score.saturating_add(a2.score).saturating_add(bonus);
                if best.map(|(_, _, bs, _)| joint > bs).unwrap_or(true) {
                    best = Some((ii, jj, joint, bonus));
                }
            }
        }

        if let Some((ii, jj, _, bonus)) = best {
            if bonus > 0 {
                alignments[r1_idx][ii].score = alignments[r1_idx][ii].score.saturating_add(bonus);
                alignments[r2_idx][jj].score = alignments[r2_idx][jj].score.saturating_add(bonus);
            }
            if promote && bonus > 0 {
                // Lock the joint optimum into slot 0 before rescue, mate
                // decoration, and MAPQ assignment.
                if ii != 0 {
                    alignments[r1_idx].swap(0, ii);
                }
                if jj != 0 {
                    alignments[r2_idx].swap(0, jj);
                }
                if mapq_floor > 0 {
                    if let Some(a) = alignments[r1_idx].first_mut() {
                        a.mapq = a.mapq.max(mapq_floor);
                    }
                    if let Some(a) = alignments[r2_idx].first_mut() {
                        a.mapq = a.mapq.max(mapq_floor);
                    }
                }
            }
        }

        i += 2;
    }
}

/// Concordance bonus base — applied to any in-window pair to nudge the joint scorer toward.
const PAIR_BONUS_BASE: i32 = 0;

/// Extra bonus when the (R1[i], R2[j]) cross-product is fully concordant.
const PAIR_BONUS_CONCORDANT: i32 = 30;

/// Promote the best concordant pair before rescue and mate-field assignment.
/// Set `KIRA_PAIR_PROMOTE=0` to disable for an A/B comparison.
fn pair_promote_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("KIRA_PAIR_PROMOTE")
            .map(|s| s != "0" && !s.eq_ignore_ascii_case("false") && !s.eq_ignore_ascii_case("off"))
            .unwrap_or(true)
    })
}

/// `KIRA_SHORT_DPTOPK` — promotion candidate-window size (chains kept per short read).
fn pair_promote_topk() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("KIRA_SHORT_DPTOPK")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&k: &usize| k >= 1)
            .unwrap_or(2)
    })
}

/// `KIRA_PAIR_BONUS` — concordant joint-score bonus (default 30).
fn pair_bonus_concordant_cfg() -> i32 {
    static V: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("KIRA_PAIR_BONUS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(PAIR_BONUS_CONCORDANT)
    })
}

/// `KIRA_PAIR_MAPQ` — MAPQ floor applied to a promoted concordant primary (0 = no boost).
fn pair_mapq_floor_cfg() -> u8 {
    static V: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("KIRA_PAIR_MAPQ")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    })
}

/// Lightweight view of an alignment used during pair re-ranking.
#[derive(Clone, Copy, Debug)]
struct MatePrimaryScored {
    ref_id: u32,
    ref_start: u32,
    ref_end: u32,
    is_rev: bool,
    score: i32,
}

impl MatePrimaryScored {
    fn from_aln(a: &Alignment) -> Self {
        Self {
            ref_id: a.ref_id,
            ref_start: a.ref_start,
            ref_end: a.ref_end,
            is_rev: a.is_rev,
            score: a.score,
        }
    }
}

/// Returns true if `(a1, a2)` is a concordant pair under `cfg`: same reference.
fn pair_is_concordant(a1: &MatePrimaryScored, a2: &MatePrimaryScored, cfg: &PairedConfig) -> bool {
    if a1.ref_id != a2.ref_id {
        return false;
    }
    let leftmost_is_a1 = a1.ref_start <= a2.ref_start;
    let convergent = if leftmost_is_a1 {
        !a1.is_rev && a2.is_rev
    } else {
        !a2.is_rev && a1.is_rev
    };
    if !convergent {
        return false;
    }
    let frag_lo = a1.ref_start.min(a2.ref_start);
    let frag_hi = a1.ref_end.max(a2.ref_end);
    let tlen = frag_hi.saturating_sub(frag_lo);
    let (lo, hi) = concordance_window(cfg);
    tlen >= lo && tlen <= hi
}

/// 3 σ-around-mean window for **concordance** classification.
fn concordance_window(cfg: &PairedConfig) -> (u32, u32) {
    if cfg.estimator_locked && cfg.insert_sd > 0 {
        let lo = (cfg.insert_mean as i64 - 3 * cfg.insert_sd as i64).max(0) as u32;
        let hi = (cfg.insert_mean as i64 + 3 * cfg.insert_sd as i64).max(0) as u32;
        (lo, hi)
    } else {
        (cfg.insert_min, cfg.insert_max)
    }
}

/// 5 σ-around-mean window for **proper-pair** classification.
fn proper_pair_window(cfg: &PairedConfig) -> (u32, u32) {
    if cfg.estimator_locked && cfg.insert_sd > 0 {
        let lo = (cfg.insert_mean as i64 - 5 * cfg.insert_sd as i64).max(0) as u32;
        let hi = (cfg.insert_mean as i64 + 5 * cfg.insert_sd as i64).max(0) as u32;
        (lo, hi)
    } else {
        (cfg.insert_min, cfg.insert_max)
    }
}

/// Mate rescue for pairs where *both* mates are mapped but their primaries are not concordant.
pub fn rescue_discordant_pairs(
    reads: &[ReadRecord],
    alignments: &mut [Vec<Alignment>],
    index: &Index,
    cfg: &PairedConfig,
    align_cfg: AlignmentConfig,
    rescue_cfg: RescueConfig,
) {
    let mmap_bytes: Option<&[u8]> = index.mmap.as_deref().map(|m| &m[..]);
    rescue_discordant_pairs_with_ref(
        reads,
        alignments,
        &index.reference,
        mmap_bytes,
        cfg,
        align_cfg,
        rescue_cfg,
    );
}

/// Reference-only entry point for [`rescue_discordant_pairs`].
pub fn rescue_discordant_pairs_with_ref(
    reads: &[ReadRecord],
    alignments: &mut [Vec<Alignment>],
    reference: &Reference,
    mmap: Option<&[u8]>,
    cfg: &PairedConfig,
    align_cfg: AlignmentConfig,
    rescue_cfg: RescueConfig,
) {
    if !cfg.is_paired() {
        return;
    }
    debug_assert_eq!(reads.len(), alignments.len());

    let half_window: u64 = if cfg.estimator_locked && cfg.insert_sd > 0 {
        (3 * cfg.insert_sd) as u64
    } else {
        cfg.insert_max as u64
    };
    let center_offset: i64 = cfg.insert_mean.max(1) as i64;

    // Pairs are independent — parallelize over R1/R2 chunks (per-pair logic unchanged).
    alignments
        .par_chunks_mut(2)
        .zip(reads.par_chunks(2))
        .for_each(|(aln_pair, read_pair)| {
            if aln_pair.len() < 2 || read_pair.len() < 2 {
                return;
            }
            if read_pair[0].pair_role == PairRole::Unpaired
                || read_pair[1].pair_role == PairRole::Unpaired
            {
                return;
            }
            if aln_pair[0].is_empty() || aln_pair[1].is_empty() {
                // Single-side unmapped — handled by rescue_unmapped_mates.
                return;
            }

            let a1 = MatePrimaryScored::from_aln(&aln_pair[0][0]);
            let a2 = MatePrimaryScored::from_aln(&aln_pair[1][0]);
            if pair_is_concordant(&a1, &a2, cfg) {
                return;
            }

            // MAPQ is assigned after pairing. Choose from alignment evidence
            // instead of path-dependent placeholder MAPQ values.
            let (anchor_local, target_local) = {
                let left = (aln_pair[0][0].score, std::cmp::Reverse(aln_pair[0][0].nm));
                let right = (aln_pair[1][0].score, std::cmp::Reverse(aln_pair[1][0].nm));
                if left >= right {
                    (0usize, 1usize)
                } else {
                    (1usize, 0usize)
                }
            };

            let anchor_score = aln_pair[anchor_local][0].score;
            if anchor_score < rescue_cfg.min_anchor_score {
                return;
            }
            let anchor_ref_id = aln_pair[anchor_local][0].ref_id;
            let anchor_ref_start = aln_pair[anchor_local][0].ref_start as i64;
            let anchor_ref_end = aln_pair[anchor_local][0].ref_end as i64;
            let anchor_is_rev = aln_pair[anchor_local][0].is_rev;
            let target_current_score = aln_pair[target_local][0].score;

            let ref_seq = reference.sequences[anchor_ref_id as usize].bases(mmap);
            let ref_len = ref_seq.len() as i64;

            let (center, target_is_rev) = if !anchor_is_rev {
                (anchor_ref_end + center_offset, true)
            } else {
                (anchor_ref_start - center_offset, false)
            };
            let win_start = (center - half_window as i64).max(0).min(ref_len) as u64;
            let win_end = (center + half_window as i64).max(0).min(ref_len) as u64;
            if win_end <= win_start {
                return;
            }
            let win_start = win_start as u32;
            let ref_window = &ref_seq[win_start as usize..win_end as usize];

            let target_seq_owned;
            let target_seq: &[u8] = if target_is_rev {
                target_seq_owned = reverse_complement(&read_pair[target_local].seq);
                &target_seq_owned
            } else {
                &read_pair[target_local].seq
            };

            let min_score = rescue_cfg.min_rescued_score.max(target_current_score);
            if let Some(mut aln) = align_in_window(
                target_seq,
                ref_window,
                win_start,
                anchor_ref_id,
                target_is_rev,
                align_cfg,
                min_score,
            ) {
                // Mark as rescue-placed so MAPQ assignment ceiling-caps it; the
                // old `aln.mapq = 30` here was a dead store (stage-5 recomputes MAPQ).
                aln.kind = crate::types::AlignmentKind::Rescued;
                aln_pair[target_local].insert(0, aln);
            }
        });
}

/// Build the MateInfo that the unmapped emitter (stage 6) needs for a paired read with no.
fn build_unmapped_mate(
    role: PairRole,
    mate_primary: Option<&MatePrimary>,
    _proper_pair: bool,
) -> MateInfo {
    let (is_r1, is_r2) = match role {
        PairRole::R1 => (true, false),
        PairRole::R2 => (false, true),
        PairRole::Unpaired => (false, false),
    };
    let (mate_ref_id, mate_pos, mate_is_rev, mate_is_unmapped) = match mate_primary {
        Some(m) => (Some(m.ref_id), m.ref_start, m.is_rev, false),
        None => (None, 0, false, true),
    };
    MateInfo {
        is_paired: true,
        is_proper_pair: false, // unmapped reads are never "properly paired"
        mate_is_unmapped,
        mate_is_rev,
        is_first_in_pair: is_r1,
        is_second_in_pair: is_r2,
        mate_ref_id,
        mate_pos,
        tlen: 0,
    }
}

#[cfg(test)]
#[path = "../../tests/unit/pipeline_pairing.rs"]
mod tests;
