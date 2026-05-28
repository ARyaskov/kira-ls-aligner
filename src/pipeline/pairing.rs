//! Paired-end post-processing for stage 4 alignments.

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
            s.parse::<u32>().map_err(|_| format!("invalid insert-size component {s:?}"))
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

        let r1_empty = alignments[r1_idx].is_empty();
        let r2_empty = alignments[r2_idx].is_empty();
        let (anchor_idx, target_idx) = match (r1_empty, r2_empty) {
            (true, false) => (r2_idx, r1_idx),
            (false, true) => (r1_idx, r2_idx),
            _ => {
                i += 2;
                continue;
            }
        };

        let anchor = &alignments[anchor_idx][0];
        if anchor.score < rescue_cfg.min_anchor_score {
            i += 2;
            continue;
        }
        let anchor_ref_id = anchor.ref_id;
        let anchor_ref_start = anchor.ref_start;
        let anchor_ref_end = anchor.ref_end;
        let anchor_is_rev = anchor.is_rev;

        let ref_seq = reference.sequences[anchor_ref_id as usize].bases(mmap);
        let ref_len = ref_seq.len() as i64;
        let (win_start_u64, win_end_u64) = if cfg.estimator_locked && cfg.insert_sd > 0 {
            let mean = cfg.insert_mean as i64;
            let half = 3 * cfg.insert_sd as i64;
            if !anchor_is_rev {
                let centre = anchor_ref_end as i64 + mean;
                let start = (centre - half).max(0).min(ref_len);
                let end = (centre + half).max(0).min(ref_len);
                (start as u64, end as u64)
            } else {
                let centre = anchor_ref_start as i64 - mean;
                let start = (centre - half).max(0).min(ref_len);
                let end = (centre + half).max(0).min(ref_len);
                (start as u64, end as u64)
            }
        } else {
            let insert_max = cfg.insert_max as u64;
            if !anchor_is_rev {
                let win_end = (anchor_ref_end as u64 + insert_max).min(ref_len as u64);
                (anchor_ref_end as u64, win_end)
            } else {
                let win_start = (anchor_ref_start as u64).saturating_sub(insert_max);
                (win_start, anchor_ref_start as u64)
            }
        };
        if win_end_u64 <= win_start_u64 {
            i += 2;
            continue;
        }
        let win_start = win_start_u64 as u32;
        let ref_window = &ref_seq[win_start as usize..win_end_u64 as usize];

        // Expected mate strand: opposite to the anchor.
        let target_is_rev = !anchor_is_rev;
        let target_seq_owned;
        let target_seq: &[u8] = if target_is_rev {
            target_seq_owned = reverse_complement(&reads[target_idx].seq);
            &target_seq_owned
        } else {
            &reads[target_idx].seq
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
            aln.mapq = 30;
            alignments[target_idx].push(aln);
        }

        i += 2;
    }
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

    let pair_bonus_base: i32 = PAIR_BONUS_BASE;
    let pair_bonus_concordant: i32 = PAIR_BONUS_CONCORDANT;

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
                alignments[r1_idx][ii].score =
                    alignments[r1_idx][ii].score.saturating_add(bonus);
                alignments[r2_idx][jj].score =
                    alignments[r2_idx][jj].score.saturating_add(bonus);
            }
        }

        i += 2;
    }
}

/// Concordance bonus base — applied to any in-window pair to nudge the joint scorer toward.
const PAIR_BONUS_BASE: i32 = 0;

/// Extra bonus when the (R1[i], R2[j]) cross-product is fully concordant.
const PAIR_BONUS_CONCORDANT: i32 = 30;

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
        if alignments[r1_idx].is_empty() || alignments[r2_idx].is_empty() {
            // Single-side unmapped — handled by rescue_unmapped_mates.
            i += 2;
            continue;
        }

        let a1 = MatePrimaryScored::from_aln(&alignments[r1_idx][0]);
        let a2 = MatePrimaryScored::from_aln(&alignments[r2_idx][0]);
        if pair_is_concordant(&a1, &a2, cfg) {
            i += 2;
            continue;
        }

        // Pick anchor = higher-MAPQ side. Ties broken toward R1.
        let (anchor_idx, target_idx) = {
            let m1 = alignments[r1_idx][0].mapq;
            let m2 = alignments[r2_idx][0].mapq;
            if m1 >= m2 { (r1_idx, r2_idx) } else { (r2_idx, r1_idx) }
        };

        let anchor = &alignments[anchor_idx][0];
        if anchor.score < rescue_cfg.min_anchor_score {
            i += 2;
            continue;
        }
        let anchor_ref_id = anchor.ref_id;
        let anchor_ref_start = anchor.ref_start as i64;
        let anchor_ref_end = anchor.ref_end as i64;
        let anchor_is_rev = anchor.is_rev;
        let target_current_score = alignments[target_idx][0].score;

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
            i += 2;
            continue;
        }
        let win_start = win_start as u32;
        let ref_window = &ref_seq[win_start as usize..win_end as usize];

        let target_seq_owned;
        let target_seq: &[u8] = if target_is_rev {
            target_seq_owned = reverse_complement(&reads[target_idx].seq);
            &target_seq_owned
        } else {
            &reads[target_idx].seq
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
            aln.mapq = 30;
            alignments[target_idx].insert(0, aln);
        }

        i += 2;
    }
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
mod tests {
    use super::*;

    fn dummy_aln(ref_id: u32, ref_start: u32, ref_end: u32, is_rev: bool) -> Alignment {
        Alignment {
            kind: crate::types::AlignmentKind::AcceptedUngapped,
            ref_id,
            ref_start,
            ref_end,
            read_start: 0,
            read_end: ref_end - ref_start,
            cigar: vec![crate::types::CigarOp {
                len: ref_end - ref_start,
                op: crate::types::CigarKind::Match,
            }],
            score: 100,
            mapq: 60,
            is_rev,
            is_secondary: false,
            is_supplementary: false,
            nm: 0,
            md: format!("{}", ref_end - ref_start),
            as_score: 100,
            xs_score: None,
            xs_strand: None,
            mate: MateInfo::default(),
        }
    }

    fn dummy_read(id: &str, role: PairRole) -> ReadRecord {
        ReadRecord {
            id: id.to_string(),
            seq: vec![b'A'; 150],
            qual: None,
            pair_role: role,
        }
    }

    #[test]
    fn insert_spec_parses_2_and_4_field_forms() {
        let mut cfg = PairedConfig::default();
        cfg.apply_insert_spec("50,500").unwrap();
        assert_eq!((cfg.insert_min, cfg.insert_max), (50, 500));

        cfg.apply_insert_spec("0,1500,300,40").unwrap();
        assert_eq!(
            (cfg.insert_min, cfg.insert_max, cfg.insert_mean, cfg.insert_sd),
            (0, 1500, 300, 40)
        );

        assert!(cfg.apply_insert_spec("100").is_err());
        assert!(cfg.apply_insert_spec("500,100").is_err()); // min > max
    }

    #[test]
    fn proper_pair_fr_orientation_in_window() {
        let mut cfg = PairedConfig::default();
        cfg.mode = IngestMode::TwoFile;
        // R1 forward at 100, R2 reverse at 350 → fragment 250..500, len 400
        let reads = vec![
            dummy_read("frag1", PairRole::R1),
            dummy_read("frag1", PairRole::R2),
        ];
        let mut alns = vec![
            vec![dummy_aln(0, 100, 250, false)],
            vec![dummy_aln(0, 350, 500, true)],
        ];
        let mut umi = vec![None; reads.len()];
        apply_pairing(&reads, &mut alns, &mut umi, &cfg);
        // R1 leftmost → tlen +400
        assert!(alns[0][0].mate.is_paired);
        assert!(alns[0][0].mate.is_proper_pair);
        assert_eq!(alns[0][0].mate.tlen, 400);
        assert!(alns[0][0].mate.is_first_in_pair);
        // R2 rightmost → tlen -400
        assert_eq!(alns[1][0].mate.tlen, -400);
        assert!(alns[1][0].mate.is_second_in_pair);
        assert!(alns[1][0].mate.is_proper_pair);
    }

    #[test]
    fn not_proper_when_different_refs() {
        let mut cfg = PairedConfig::default();
        cfg.mode = IngestMode::TwoFile;
        let reads = vec![
            dummy_read("x", PairRole::R1),
            dummy_read("x", PairRole::R2),
        ];
        let mut alns = vec![
            vec![dummy_aln(0, 100, 250, false)],
            vec![dummy_aln(1, 100, 250, true)],
        ];
        let mut umi = vec![None; reads.len()];
        apply_pairing(&reads, &mut alns, &mut umi, &cfg);
        assert!(alns[0][0].mate.is_paired);
        assert!(!alns[0][0].mate.is_proper_pair);
        assert_eq!(alns[0][0].mate.tlen, 0);
    }

    #[test]
    fn mate_unmapped_marks_correct_bit() {
        let mut cfg = PairedConfig::default();
        cfg.mode = IngestMode::TwoFile;
        let reads = vec![
            dummy_read("x", PairRole::R1),
            dummy_read("x", PairRole::R2),
        ];
        let mut alns = vec![vec![dummy_aln(0, 100, 250, false)], vec![]];
        let mut umi = vec![None; reads.len()];
        apply_pairing(&reads, &mut alns, &mut umi, &cfg);
        // R1's mate is unmapped → 0x8 should be set on R1
        assert!(alns[0][0].mate.is_paired);
        assert!(alns[0][0].mate.mate_is_unmapped);
        assert!(!alns[0][0].mate.is_proper_pair);
        assert_eq!(alns[0][0].mate.mate_ref_id, None);
    }

    #[test]
    fn unpaired_mode_is_noop() {
        let cfg = PairedConfig::default(); // mode = Unpaired
        let reads = vec![dummy_read("r1", PairRole::Unpaired)];
        let mut alns = vec![vec![dummy_aln(0, 0, 100, false)]];
        let mut umi = vec![None; reads.len()];
        apply_pairing(&reads, &mut alns, &mut umi, &cfg);
        assert!(!alns[0][0].mate.is_paired);
    }

    fn dummy_aln_score(
        ref_id: u32,
        ref_start: u32,
        ref_end: u32,
        is_rev: bool,
        score: i32,
    ) -> Alignment {
        let mut a = dummy_aln(ref_id, ref_start, ref_end, is_rev);
        a.score = score;
        a
    }

    #[test]
    fn rerank_keeps_concordant_pair_at_slot_0() {
        let mut cfg = PairedConfig::default();
        cfg.mode = IngestMode::TwoFile;
        cfg.insert_mean = 400;
        cfg.insert_sd = 100;
        let reads = vec![
            dummy_read("r", PairRole::R1),
            dummy_read("r", PairRole::R2),
        ];
        let mut alns = vec![
            vec![
                dummy_aln_score(0, 100, 250, false, 150),
                dummy_aln_score(1, 1000, 1150, false, 100),
            ],
            vec![dummy_aln_score(0, 350, 500, true, 145)],
        ];
        pair_rerank(&reads, &mut alns, &cfg, 5);
        // Slot 0 of R1 stays on chr0. Score was boosted by 30/2=15.
        assert_eq!(alns[0][0].ref_id, 0);
        assert!(alns[0][0].score >= 150);
    }

    #[test]
    fn rerank_promotes_pair_consistent_alternative() {
        let mut cfg = PairedConfig::default();
        cfg.mode = IngestMode::TwoFile;
        cfg.insert_mean = 400;
        cfg.insert_sd = 100;
        let reads = vec![
            dummy_read("r", PairRole::R1),
            dummy_read("r", PairRole::R2),
        ];
        let mut alns = vec![
            vec![
                dummy_aln_score(1, 50_000, 50_150, false, 200),
                dummy_aln_score(0, 100, 250, false, 180),
            ],
            vec![dummy_aln_score(0, 350, 500, true, 140)],
        ];
        pair_rerank(&reads, &mut alns, &cfg, 5);
        let chr0_score = alns[0]
            .iter()
            .find(|a| a.ref_id == 0)
            .map(|a| a.score)
            .unwrap();
        let chr1_score = alns[0]
            .iter()
            .find(|a| a.ref_id == 1)
            .map(|a| a.score)
            .unwrap();
        assert!(
            chr0_score > chr1_score,
            "concordant alt (chr0={chr0_score}) didn't overtake lone primary (chr1={chr1_score})"
        );
    }

    #[test]
    fn rerank_no_bonus_when_all_combos_discordant() {
        let mut cfg = PairedConfig::default();
        cfg.mode = IngestMode::TwoFile;
        cfg.insert_mean = 400;
        cfg.insert_sd = 100;
        let reads = vec![
            dummy_read("r", PairRole::R1),
            dummy_read("r", PairRole::R2),
        ];
        let r1_orig = 150;
        let r2_orig = 140;
        let mut alns = vec![
            vec![dummy_aln_score(0, 100, 250, false, r1_orig)],
            vec![dummy_aln_score(0, 100_000, 100_150, true, r2_orig)],
        ];
        pair_rerank(&reads, &mut alns, &cfg, 5);
        assert_eq!(alns[0][0].score, r1_orig);
        assert_eq!(alns[1][0].score, r2_orig);
    }

    #[test]
    fn pair_is_concordant_respects_3sigma() {
        let mut cfg = PairedConfig::default();
        cfg.insert_mean = 500;
        cfg.insert_sd = 100;
        cfg.estimator_locked = true;
        // In-window: TLEN 600
        let a = MatePrimaryScored {
            ref_id: 0,
            ref_start: 0,
            ref_end: 150,
            is_rev: false,
            score: 100,
        };
        let b = MatePrimaryScored {
            ref_id: 0,
            ref_start: 450,
            ref_end: 600,
            is_rev: true,
            score: 100,
        };
        assert!(pair_is_concordant(&a, &b, &cfg));
        // Out of window: TLEN 1000 (5σ)
        let b_far = MatePrimaryScored {
            ref_start: 850,
            ref_end: 1000,
            ..b
        };
        assert!(!pair_is_concordant(&a, &b_far, &cfg));
        // Cross-chr
        let b_chr = MatePrimaryScored { ref_id: 1, ..b };
        assert!(!pair_is_concordant(&a, &b_chr, &cfg));
        // Wrong orientation (RR)
        let b_rr = MatePrimaryScored { is_rev: false, ..b };
        assert!(!pair_is_concordant(&a, &b_rr, &cfg));
    }

    #[test]
    fn concordance_window_falls_back_to_min_max_during_bootstrap() {
        let mut cfg = PairedConfig::default();
        cfg.insert_min = 0;
        cfg.insert_max = 1000;
        cfg.insert_mean = 200;
        cfg.insert_sd = 50;
        cfg.estimator_locked = false;
        let (lo, hi) = concordance_window(&cfg);
        assert_eq!((lo, hi), (0, 1000));

        cfg.insert_mean = 570;
        cfg.insert_sd = 155;
        cfg.estimator_locked = true;
        let (lo, hi) = concordance_window(&cfg);
        assert_eq!((lo, hi), (570 - 3 * 155, 570 + 3 * 155));
    }

    #[test]
    fn proper_pair_window_uses_5sigma_after_lock() {
        let mut cfg = PairedConfig::default();
        cfg.insert_mean = 570;
        cfg.insert_sd = 155;
        cfg.estimator_locked = true;
        let (lo, hi) = proper_pair_window(&cfg);
        assert_eq!((lo, hi), (0, 570 + 5 * 155));
        // Bootstrap fallback is still [insert_min, insert_max].
        cfg.estimator_locked = false;
        cfg.insert_min = 0;
        cfg.insert_max = 1000;
        assert_eq!(proper_pair_window(&cfg), (0, 1000));
    }

    #[test]
    fn classify_pair_marks_out_of_5sigma_pair_improper_after_lock() {
        let mut cfg = PairedConfig::default();
        cfg.mode = IngestMode::TwoFile;
        cfg.insert_mean = 500;
        cfg.insert_sd = 100;
        cfg.insert_min = 0;
        cfg.insert_max = 5_000; // legacy field set wide on purpose
        cfg.estimator_locked = true;

        // TLEN 1100 = 6 σ from mean = 500
        let p1 = MatePrimary {
            ref_id: 0,
            ref_start: 0,
            ref_end: 150,
            is_rev: false,
        };
        let p2 = MatePrimary {
            ref_id: 0,
            ref_start: 950,
            ref_end: 1100,
            is_rev: true,
        };
        let (proper, _, _) = classify_pair(&p1, &p2, &cfg);
        assert!(
            !proper,
            "TLEN 1100 = 6 σ from mean 500 must be classified discordant post-lock"
        );

        // TLEN 900 = 4 σ → inside 5 σ → proper.
        let p2_in = MatePrimary {
            ref_id: 0,
            ref_start: 750,
            ref_end: 900,
            is_rev: true,
        };
        let (proper_in, _, _) = classify_pair(&p1, &p2_in, &cfg);
        assert!(proper_in, "TLEN 900 = 4 σ from mean must stay proper");
    }
}
