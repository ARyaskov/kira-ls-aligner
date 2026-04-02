use anyhow::Result;

use crate::index::{Index, MinimizerIndex, Occ};
use crate::seq::reverse_complement;
use crate::sketch::{MinimizerConfig, minimizers};
use crate::types::{Anchor, Minimizer, ReadRecord, SeedHit, Strand};

/// Seeding configuration.
#[derive(Clone, Copy, Debug)]
pub struct SeedingConfig {
    pub min_anchor_len: u32,
    pub max_occ: usize,
    pub long_read_threshold: usize,
}

fn add_seed_hits(
    read: &ReadRecord,
    mins: &[Minimizer],
    k: usize,
    cfg: SeedingConfig,
    mut get_len: impl FnMut(u64) -> Option<usize>,
    mut for_each: impl FnMut(u64, &mut dyn FnMut(Occ)),
    out: &mut Vec<SeedHit>,
) {
    for m in mins {
        let Some(occ_len) = get_len(m.hash) else {
            continue;
        };
        if occ_len > cfg.max_occ {
            continue;
        }
        let mut push_occ = |o: Occ| {
            let is_rev = o.strand != m.strand;
            let read_pos = if is_rev {
                (read.seq.len() - m.pos as usize - k) as u32
            } else {
                m.pos
            };
            out.push(SeedHit {
                hash: m.hash,
                read_pos,
                ref_id: o.ref_id,
                ref_pos: o.pos,
                strand: if is_rev {
                    Strand::Reverse
                } else {
                    Strand::Forward
                },
            });
        };
        for_each(m.hash, &mut push_occ);
    }
}

/// Collect seed hits for a read by computing minimizers on demand.
pub fn collect_seed_hits(
    read: &ReadRecord,
    index: &Index,
    cfg: SeedingConfig,
) -> Result<Vec<SeedHit>> {
    let (table, k, w) = select_index(read.seq.len(), index, cfg.long_read_threshold);
    let mins = minimizers(&read.seq, &MinimizerConfig { k, w });
    let mut hits = Vec::new();
    add_seed_hits(
        read,
        &mins,
        k,
        cfg,
        |hash| index.bucket_len(table, hash),
        |hash, f| index.for_each_occ(table, hash, f),
        &mut hits,
    );
    Ok(hits)
}

/// Collect seed hits using precomputed minimizers.
pub fn collect_seed_hits_from_minimizers(
    read: &ReadRecord,
    mins: &[Minimizer],
    index: &Index,
    table: &MinimizerIndex,
    k: usize,
    cfg: SeedingConfig,
) -> Vec<SeedHit> {
    let mut hits = Vec::new();
    collect_seed_hits_from_minimizers_into(read, mins, index, table, k, cfg, &mut hits);
    hits
}

/// Collect seed hits using precomputed minimizers into a scratch buffer.
pub fn collect_seed_hits_from_minimizers_into(
    read: &ReadRecord,
    mins: &[Minimizer],
    index: &Index,
    table: &MinimizerIndex,
    k: usize,
    cfg: SeedingConfig,
    out: &mut Vec<SeedHit>,
) {
    out.clear();
    add_seed_hits(
        read,
        mins,
        k,
        cfg,
        |hash| index.bucket_len(table, hash),
        |hash, f| index.for_each_occ(table, hash, f),
        out,
    );
}

/// Extend seed hits into MEM-like anchors by direct byte comparison.
pub fn extend_seeds(
    read: &ReadRecord,
    index: &Index,
    hits: &[SeedHit],
    cfg: SeedingConfig,
) -> Vec<Anchor> {
    let mut anchors = Vec::new();
    extend_seeds_into(read, index, hits, cfg, &mut anchors);
    anchors
}

/// Extend seed hits into a provided anchor buffer.
pub fn extend_seeds_into(
    read: &ReadRecord,
    index: &Index,
    hits: &[SeedHit],
    cfg: SeedingConfig,
    out: &mut Vec<Anchor>,
) {
    out.clear();
    let rc_read = reverse_complement(&read.seq);
    let k = select_index(read.seq.len(), index, cfg.long_read_threshold).1 as i32;

    for hit in hits {
        let ref_seq = index.ref_bases(hit.ref_id as usize);
        if hit.ref_pos as usize >= ref_seq.len() {
            continue;
        }
        let (read_seq, strand) = if hit.strand == Strand::Reverse {
            (&rc_read, Strand::Reverse)
        } else {
            (&read.seq, Strand::Forward)
        };
        let mut q_start = hit.read_pos as i32;
        let mut r_start = hit.ref_pos as i32;
        let mut q_end = q_start + k;
        let mut r_end = r_start + k;

        while q_start > 0 && r_start > 0 {
            let qb = read_seq[(q_start - 1) as usize];
            let rb = ref_seq[(r_start - 1) as usize];
            if qb != rb {
                break;
            }
            q_start -= 1;
            r_start -= 1;
        }
        while (q_end as usize) < read_seq.len() && (r_end as usize) < ref_seq.len() {
            let qb = read_seq[q_end as usize];
            let rb = ref_seq[r_end as usize];
            if qb != rb {
                break;
            }
            q_end += 1;
            r_end += 1;
        }

        let len = (q_end - q_start).max(0) as u32;
        if len < cfg.min_anchor_len {
            continue;
        }

        out.push(Anchor {
            read_start: q_start as u32,
            read_end: q_end as u32,
            ref_id: hit.ref_id,
            ref_start: r_start as u32,
            ref_end: r_end as u32,
            strand,
            score: len as i32,
        });
    }
}

fn select_index(
    read_len: usize,
    index: &Index,
    long_read_threshold: usize,
) -> (&MinimizerIndex, usize, usize) {
    if read_len >= long_read_threshold {
        (&index.long, index.long.k, index.long.w)
    } else {
        (&index.short, index.short.k, index.short.w)
    }
}
