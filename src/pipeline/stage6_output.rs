use anyhow::Result;
use rayon::prelude::*;

use crate::io::{EmitFormat, OutputConfig, SamFormatter, SamWriter};
use crate::types::{PairRole, ReadRecord};

use super::stage5_scoring::ScoredBatch;

/// One read ready for emission: the read, its alignments, the mate context
/// for an unmapped record, and the pre-built mate tags (`MC`/`MQ`/`ms`).
type EmitRecord = (
    ReadRecord,
    Vec<crate::types::Alignment>,
    Option<crate::types::MateInfo>,
    Option<Vec<u8>>,
);

/// Stage 6 output: write SAM records.
pub fn run(
    input: ScoredBatch,
    writer: &mut SamWriter,
    read_group: Option<&str>,
    output_cfg: OutputConfig,
    max_alignments: usize,
) -> Result<()> {
    let buf = serialize(
        input,
        writer.formatter(),
        read_group,
        output_cfg,
        max_alignments,
    );
    writer.write_batch(&buf)
}

/// Serialize a `ScoredBatch` to a `Vec<u8>` ready to feed to `SamWriter::write_batch`.
pub fn serialize(
    input: ScoredBatch,
    formatter: &SamFormatter,
    read_group: Option<&str>,
    output_cfg: OutputConfig,
    max_alignments: usize,
) -> Vec<u8> {
    let mut out = Vec::new();
    serialize_into(
        input,
        formatter,
        read_group,
        output_cfg,
        max_alignments,
        &mut out,
    );
    out
}

/// [`serialize`] into a caller-owned buffer, cleared first and keeping its
/// capacity so it can be recycled across batches.
pub fn serialize_into(
    input: ScoredBatch,
    formatter: &SamFormatter,
    read_group: Option<&str>,
    output_cfg: OutputConfig,
    max_alignments: usize,
    out: &mut Vec<u8>,
) {
    let reads: Vec<ReadRecord> = input.reads;
    let mut alignments = input.alignments;
    let unmapped_mate_info = input.unmapped_mate_info;

    if max_alignments > 0 {
        for alns in alignments.iter_mut() {
            retain_reported_alignments(alns, max_alignments);
        }
    }

    // Mate tags (MC/MQ/ms) come from the mate's primary record, which sits in
    // the adjacent slot (R1 at 2k, R2 at 2k+1). Built once per read here so
    // the parallel chunks below never look outside their own slice.
    let mate_tags: Vec<Option<Vec<u8>>> = if output_cfg.write_mate_tags
        && matches!(output_cfg.format, EmitFormat::Sam)
        && reads.iter().any(|r| r.pair_role != PairRole::Unpaired)
    {
        (0..reads.len())
            .map(|i| mate_tag_bytes(&reads, &alignments, i))
            .collect()
    } else {
        vec![None; reads.len()]
    };

    let quads: Vec<EmitRecord> = reads
        .into_iter()
        .zip(alignments)
        .zip(unmapped_mate_info)
        .zip(mate_tags)
        .map(|(((r, a), m), t)| (r, a, m, t))
        .collect();

    let chunk_size: usize = 64;

    let chunks: Vec<Vec<u8>> = quads
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut estimate = 0usize;
            for (read, alns, _mate, _tags) in chunk.iter() {
                let qual_len = read.qual.as_ref().map_or(1, |q| q.len());
                let per = read.id.len() + read.seq.len() + qual_len + 96;
                let count = if alns.is_empty() { 1 } else { alns.len() };
                estimate += per * count;
            }
            let mut buf = Vec::with_capacity(estimate);
            let mut extra_tags: Vec<u8> = Vec::new();
            for (read, alns, mate, mtags) in chunk.iter() {
                let mtags = mtags.as_deref();
                match output_cfg.format {
                    EmitFormat::Sam => {
                        if alns.is_empty() {
                            formatter.append_unmapped_full(
                                &mut buf,
                                read,
                                mate.as_ref(),
                                mtags,
                                output_cfg.append_comment,
                            );
                        } else {
                            extra_tags.clear();
                            if output_cfg.write_xa {
                                formatter.append_xa_capped(&mut extra_tags, alns, output_cfg.xa_max);
                            }
                            if output_cfg.write_sa {
                                formatter.append_sa(&mut extra_tags, alns);
                            }
                            if let Some(t) = mtags {
                                extra_tags.extend_from_slice(t);
                            }
                            let extra = if extra_tags.is_empty() {
                                None
                            } else {
                                Some(extra_tags.as_slice())
                            };
                            for (idx, aln) in alns.iter().enumerate() {
                                // Primary: XA/SA + mate tags. Supplementary:
                                // mate tags only (fixmate stamps them too).
                                // Secondary: nothing beyond the core tags.
                                let tags = if idx == 0 {
                                    extra
                                } else if aln.is_supplementary {
                                    mtags
                                } else {
                                    None
                                };
                                formatter.append_alignment(
                                    &mut buf, read, aln, read_group, tags, output_cfg,
                                );
                            }
                        }
                    }
                    EmitFormat::Paf => {
                        for aln in alns.iter() {
                            formatter.append_paf(&mut buf, read, aln, output_cfg);
                        }
                    }
                }
            }
            buf
        })
        .collect();

    let total: usize = chunks.iter().map(|c| c.len()).sum();
    out.clear();
    out.reserve(total.saturating_sub(out.capacity()));
    for c in chunks {
        out.extend_from_slice(&c);
    }
}

/// `\tMC:Z:<cigar>\tMQ:i:<mapq>\tms:i:<AS>` for read `i`'s mate, when the
/// mate is the adjacent slot and has a primary alignment. These are the tags
/// `samtools fixmate -m` derives, and `samtools markdup` keys on `MC`/`ms`.
fn mate_tag_bytes(
    reads: &[ReadRecord],
    alignments: &[Vec<crate::types::Alignment>],
    i: usize,
) -> Option<Vec<u8>> {
    let mate_idx = match reads[i].pair_role {
        PairRole::R1 => i + 1,
        PairRole::R2 => i.checked_sub(1)?,
        PairRole::Unpaired => return None,
    };
    let mate = reads.get(mate_idx)?;
    let expected = match reads[i].pair_role {
        PairRole::R1 => PairRole::R2,
        _ => PairRole::R1,
    };
    if mate.pair_role != expected {
        return None;
    }
    let primary = alignments[mate_idx].first()?;
    let mut out = Vec::with_capacity(48);
    out.extend_from_slice(b"\tMC:Z:");
    for op in &primary.cigar {
        use std::io::Write as _;
        let _ = write!(out, "{op}");
    }
    out.extend_from_slice(b"\tMQ:i:");
    out.extend_from_slice(primary.mapq.to_string().as_bytes());
    out.extend_from_slice(b"\tms:i:");
    out.extend_from_slice(primary.as_score.to_string().as_bytes());
    Some(out)
}

fn retain_reported_alignments(
    alignments: &mut Vec<crate::types::Alignment>,
    max_alignments: usize,
) {
    let mut primary_or_secondary = 0usize;
    alignments.retain(|aln| {
        if aln.is_supplementary {
            return true;
        }
        let keep = primary_or_secondary < max_alignments;
        primary_or_secondary += 1;
        keep
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Alignment, AlignmentKind, CigarKind, CigarOp, MateInfo};

    fn alignment(score: i32, supplementary: bool) -> Alignment {
        Alignment {
            kind: AlignmentKind::DpAligned,
            ref_id: 0,
            ref_start: 0,
            ref_end: 10,
            read_start: 0,
            read_end: 10,
            cigar: vec![CigarOp {
                len: 10,
                op: CigarKind::Match,
            }],
            score,
            mapq: 0,
            is_rev: false,
            is_secondary: score < 100 && !supplementary,
            is_supplementary: supplementary,
            nm: 0,
            md: "10".to_string(),
            as_score: score,
            xs_score: None,
            xs_strand: None,
            mate: MateInfo::default(),
        }
    }

    #[test]
    fn mate_tags_come_from_the_adjacent_mate_primary() {
        let mut r1 = ReadRecord::new_unpaired("t".into(), vec![b'A'; 10], None);
        r1.pair_role = PairRole::R1;
        let mut r2 = r1.clone();
        r2.pair_role = PairRole::R2;
        let reads = vec![r1, r2];
        let mut mate_primary = alignment(97, false);
        mate_primary.mapq = 37;
        let alns = vec![vec![alignment(100, false)], vec![mate_primary]];
        let tags = mate_tag_bytes(&reads, &alns, 0).expect("R1 gets its mate's tags");
        assert_eq!(tags, b"\tMC:Z:10M\tMQ:i:37\tms:i:97");
        let tags = mate_tag_bytes(&reads, &alns, 1).expect("R2 too");
        assert_eq!(tags, b"\tMC:Z:10M\tMQ:i:0\tms:i:100");
        // An unmapped mate yields nothing; unpaired reads never do.
        let alns = vec![vec![alignment(100, false)], vec![]];
        assert!(mate_tag_bytes(&reads, &alns, 0).is_none());
        let single = vec![ReadRecord::new_unpaired("s".into(), vec![b'A'; 10], None)];
        assert!(mate_tag_bytes(&single, &[vec![]], 0).is_none());
    }

    #[test]
    fn max_alignments_does_not_discard_supplementary_records() {
        let mut alignments = vec![
            alignment(100, false),
            alignment(90, false),
            alignment(80, true),
        ];
        retain_reported_alignments(&mut alignments, 1);
        assert_eq!(alignments.len(), 2);
        assert!(!alignments[0].is_supplementary);
        assert!(alignments[1].is_supplementary);
    }
}
