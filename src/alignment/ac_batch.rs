//! Batch-level Aho-Corasick exact-match fast path.
//!
//! ~80% of typical short reads have a bit-exact match somewhere in the
//! reference. Building a single Aho-Corasick automaton from all reads in a
//! batch and scanning the reference once is far cheaper than per-read minimizer
//! lookups + chaining + DP for the exact-match subset.
//!
//! Auto-disabled on large references (the per-batch rescan outweighs the seeding
//! it saves). `KIRA_AC_DISABLE` (unset=auto, 0=on, 1=off), `KIRA_AC_MAX_REF_MB`=50.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use rayon::prelude::*;

use crate::alignment::AlignmentConfig;
use crate::index::Index;
use crate::seq::reverse_complement_into;
use crate::types::{Alignment, AlignmentKind, CigarKind, CigarOp, MateInfo, ReadRecord};

/// Minimum read length to qualify for AC matching.
/// Below this, the trie produces many spurious matches and the cascade is
/// already cheap.
const MIN_PATTERN_LEN: usize = 30;

/// Maximum read length to qualify for AC matching.
/// Long reads (>1 kb) cost more to scan than to seed.
const MAX_PATTERN_LEN: usize = 1024;

/// Skip AC entirely when the batch has fewer eligible reads than this — the
/// build cost is not amortized.
const MIN_ELIGIBLE_READS: usize = 64;

/// Per-shard reference bytes for the parallel AC scan (adjacent chunks overlap
/// by `MAX_PATTERN_LEN - 1` so boundary-spanning matches aren't missed).
const AC_SCAN_CHUNK: usize = 2 * 1024 * 1024;

/// Per-batch AC scan output, parallel to the input reads slice.
#[derive(Clone, Debug)]
pub struct AcBatchOutput {
    /// `alignments[i]` is non-empty if read `i` had at least one exact-match
    /// hit. Read indices line up 1:1 with the input batch.
    pub alignments: Vec<Vec<Alignment>>,
    /// Diagnostic counters.
    pub stats: AcBatchStats,
}

impl AcBatchOutput {
    /// Empty result — one slot per read, all empty.
    pub fn empty(n_reads: usize) -> Self {
        Self {
            alignments: vec![Vec::new(); n_reads],
            stats: AcBatchStats::default(),
        }
    }

    /// True for reads resolved by AC (alignment slot is non-empty).
    pub fn resolved_mask(&self) -> Vec<bool> {
        self.alignments.iter().map(|v| !v.is_empty()).collect()
    }
}

/// Per-batch counters for the AC stage.
#[derive(Clone, Copy, Debug, Default)]
pub struct AcBatchStats {
    pub n_reads: usize,
    pub reads_eligible: usize,
    pub reads_resolved: usize,
    pub reads_ambiguous: usize,
    pub fwd_hits: usize,
    pub rev_hits: usize,
    pub build_ms: f32,
    pub scan_ms: f32,
}

/// AC auto-disables above this total reference size (bp); override: KIRA_AC_MAX_REF_MB.
const AC_MAX_REF_BP_DEFAULT: u64 = 50_000_000;

/// AC policy from `KIRA_AC_DISABLE`: unset = Auto, `0` = ForceOn, else ForceOff.
#[derive(Clone, Copy, PartialEq)]
enum AcMode {
    Auto,
    ForceOn,
    ForceOff,
}

fn ac_mode() -> AcMode {
    static CELL: OnceLock<AcMode> = OnceLock::new();
    *CELL.get_or_init(|| {
        match std::env::var("KIRA_AC_DISABLE")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
        {
            None => AcMode::Auto,
            Some(0) => AcMode::ForceOn,
            Some(_) => AcMode::ForceOff,
        }
    })
}

fn ac_max_ref_bp() -> u64 {
    static CELL: OnceLock<u64> = OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("KIRA_AC_MAX_REF_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map(|mb| mb.saturating_mul(1_000_000))
            .unwrap_or(AC_MAX_REF_BP_DEFAULT)
    })
}

fn ac_should_run(total_ref_bp: u64) -> bool {
    let mode = ac_mode();
    let run = match mode {
        AcMode::ForceOff => false,
        AcMode::ForceOn => true,
        AcMode::Auto => total_ref_bp <= ac_max_ref_bp(),
    };
    static LOGGED: OnceLock<()> = OnceLock::new();
    LOGGED.get_or_init(|| {
        let mb = total_ref_bp as f64 / 1e6;
        let thr = ac_max_ref_bp() / 1_000_000;
        match mode {
            AcMode::Auto if run => {
                eprintln!("[KIRA_AC] auto: ENABLED (reference {mb:.0} Mbp ≤ {thr} Mbp)")
            }
            AcMode::Auto => eprintln!(
                "[KIRA_AC] auto: DISABLED (reference {mb:.0} Mbp > {thr} Mbp) — using indexed \
                 path. Force on: KIRA_AC_DISABLE=0; raise gate: KIRA_AC_MAX_REF_MB."
            ),
            AcMode::ForceOn => {
                eprintln!("[KIRA_AC] forced ON (KIRA_AC_DISABLE=0), reference {mb:.0} Mbp")
            }
            AcMode::ForceOff => eprintln!("[KIRA_AC] forced OFF (KIRA_AC_DISABLE=1)"),
        }
    });
    run
}

/// Build forward + reverse-complement AC automata, scan every contig once,
/// emit perfect-match alignments for reads with hits.
///
/// `max_alignments` caps the alignments retained per read (the rest are dropped).
pub fn run(
    reads: &[ReadRecord],
    index: &Index,
    cfg: AlignmentConfig,
    max_alignments: usize,
) -> AcBatchOutput {
    let mut out = AcBatchOutput::empty(reads.len());
    out.stats.n_reads = reads.len();

    if reads.is_empty() {
        return out;
    }

    let total_ref_bp: u64 = (0..index.reference.sequences.len())
        .map(|r| index.ref_bases(r).len() as u64)
        .sum();
    if !ac_should_run(total_ref_bp) {
        return out;
    }

    let t_build = std::time::Instant::now();

    let mut fwd_patterns: Vec<&[u8]> = Vec::with_capacity(reads.len());
    let mut fwd_to_read: Vec<usize> = Vec::with_capacity(reads.len());
    let mut rc_buffers: Vec<Vec<u8>> = Vec::with_capacity(reads.len());
    let mut rc_to_read: Vec<usize> = Vec::with_capacity(reads.len());

    for (idx, read) in reads.iter().enumerate() {
        let len = read.seq.len();
        if !(MIN_PATTERN_LEN..=MAX_PATTERN_LEN).contains(&len) {
            continue;
        }
        // Reads with N cannot match the reference byte-exactly.
        if read.seq.contains(&b'N') {
            continue;
        }
        fwd_patterns.push(read.seq.as_slice());
        fwd_to_read.push(idx);
        let mut rc = Vec::with_capacity(len);
        reverse_complement_into(&read.seq, &mut rc);
        rc_buffers.push(rc);
        rc_to_read.push(idx);
    }
    out.stats.reads_eligible = fwd_patterns.len();

    if fwd_patterns.len() < MIN_ELIGIBLE_READS {
        return out;
    }

    let rc_patterns: Vec<&[u8]> = rc_buffers.iter().map(|v| v.as_slice()).collect();

    // fwd/rev automata are independent — build concurrently.
    let (ac_fwd_opt, ac_rev_opt) =
        rayon::join(|| build_ac(&fwd_patterns), || build_ac(&rc_patterns));
    let (ac_fwd, ac_rev) = match (ac_fwd_opt, ac_rev_opt) {
        (Some(f), Some(r)) => (f, r),
        _ => return out,
    };

    out.stats.build_ms = t_build.elapsed().as_secs_f64() as f32 * 1000.0;
    let t_scan = std::time::Instant::now();

    let n_contigs = index.reference.sequences.len();

    // Flat (ref_id, chunk) work list so the scan uses all cores regardless of
    // contig count/size (per-contig parallelism starves on single-chromosome refs).
    let overlap = MAX_PATTERN_LEN - 1;
    let mut work: Vec<(u32, usize)> = Vec::new();
    for ref_id in 0..n_contigs {
        let len = index.ref_bases(ref_id).len();
        let mut c0 = 0usize;
        while c0 < len {
            work.push((ref_id as u32, c0));
            c0 += AC_SCAN_CHUNK;
        }
    }

    let fwd_counter = AtomicUsize::new(0);
    let rev_counter = AtomicUsize::new(0);

    // Each chunk owns match-starts in [c0, c0+owned_len); the +overlap tail lets
    // an in-range match complete, starts past it go to the next chunk (no dup/miss).
    // Sort key (ref_id,is_rev,start) replays serial order so the merge is identical.
    let mut hits: Vec<((u32, u8, u32), usize, Alignment)> = work
        .into_par_iter()
        .flat_map_iter(|(ref_id, c0)| {
            let ref_bases = index.ref_bases(ref_id as usize);
            let len = ref_bases.len();
            let owned_len = AC_SCAN_CHUNK.min(len - c0);
            let slice_end = (c0 + AC_SCAN_CHUNK + overlap).min(len);
            let slice = &ref_bases[c0..slice_end];
            let mut local: Vec<((u32, u8, u32), usize, Alignment)> = Vec::new();
            let mut fwd_local = 0usize;
            let mut rev_local = 0usize;

            for mat in ac_fwd.find_overlapping_iter(slice) {
                if mat.start() >= owned_len {
                    continue;
                }
                let read_idx = fwd_to_read[mat.pattern().as_usize()];
                let read_len = reads[read_idx].seq.len();
                let ref_start = (c0 + mat.start()) as u32;
                let aln = make_perfect_match(ref_id, ref_start, read_len, false, cfg);
                local.push(((ref_id, 0, ref_start), read_idx, aln));
                fwd_local += 1;
            }

            for mat in ac_rev.find_overlapping_iter(slice) {
                if mat.start() >= owned_len {
                    continue;
                }
                let read_idx = rc_to_read[mat.pattern().as_usize()];
                let read_len = reads[read_idx].seq.len();
                let ref_start = (c0 + mat.start()) as u32;
                let aln = make_perfect_match(ref_id, ref_start, read_len, true, cfg);
                local.push(((ref_id, 1, ref_start), read_idx, aln));
                rev_local += 1;
            }

            fwd_counter.fetch_add(fwd_local, Ordering::Relaxed);
            rev_counter.fetch_add(rev_local, Ordering::Relaxed);
            local
        })
        .collect();

    out.stats.fwd_hits = fwd_counter.load(Ordering::Relaxed);
    out.stats.rev_hits = rev_counter.load(Ordering::Relaxed);

    hits.sort_unstable_by_key(|(key, read_idx, _)| (*read_idx, *key));
    // A palindromic read is found by both forward and reverse automata at the
    // same locus. That is one genomic placement, not a two-locus ambiguity.
    hits.dedup_by(|a, b| a.1 == b.1 && a.0.0 == b.0.0 && a.0.2 == b.0.2);
    out.stats.scan_ms = t_scan.elapsed().as_secs_f64() as f32 * 1000.0;

    let cap = max_alignments.max(1);
    let mut start = 0usize;
    while start < hits.len() {
        let read_idx = hits[start].1;
        let mut end = start + 1;
        while end < hits.len() && hits[end].1 == read_idx {
            end += 1;
        }
        let placements = end - start;
        if placements == 1 || cap > 1 {
            out.alignments[read_idx].extend(
                hits[start..end]
                    .iter()
                    .take(cap)
                    .map(|(_, _, aln)| aln.clone()),
            );
        } else {
            // A top-1 exact shortcut has no evidence for choosing among equal
            // placements. Leave it unresolved so indexed chaining can compute
            // repeat evidence and an honest second-best/MAPQ.
            out.stats.reads_ambiguous += 1;
        }
        start = end;
    }

    out.stats.reads_resolved = out.alignments.iter().filter(|v| !v.is_empty()).count();
    out
}

fn build_ac(patterns: &[&[u8]]) -> Option<AhoCorasick> {
    if patterns.is_empty() {
        return None;
    }
    AhoCorasickBuilder::new()
        .match_kind(MatchKind::Standard)
        .ascii_case_insensitive(false)
        .build(patterns)
        .ok()
}

fn make_perfect_match(
    ref_id: u32,
    ref_start: u32,
    read_len: usize,
    is_rev: bool,
    cfg: AlignmentConfig,
) -> Alignment {
    let cigar = vec![CigarOp {
        len: read_len as u32,
        op: CigarKind::Match,
    }];
    let score = cfg.match_score * read_len as i32;
    Alignment {
        kind: AlignmentKind::AcceptedUngapped,
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
        nm: 0,
        md: read_len.to_string(),
        as_score: score,
        xs_score: None,
        xs_strand: None,
        mate: MateInfo::default(),
    }
}

#[cfg(test)]
#[path = "../../tests/unit/alignment_ac_batch.rs"]
mod tests;
