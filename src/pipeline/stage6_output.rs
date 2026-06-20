use anyhow::Result;
use rayon::prelude::*;

use crate::io::{EmitFormat, OutputConfig, SamFormatter, SamWriter};
use crate::types::ReadRecord;

use super::stage5_scoring::ScoredBatch;

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
    let reads: Vec<ReadRecord> = input.reads;
    let mut alignments = input.alignments;
    let unmapped_mate_info = input.unmapped_mate_info;

    if max_alignments > 0 {
        for alns in alignments.iter_mut() {
            retain_reported_alignments(alns, max_alignments);
        }
    }

    let triples: Vec<(
        ReadRecord,
        Vec<crate::types::Alignment>,
        Option<crate::types::MateInfo>,
    )> = reads
        .into_iter()
        .zip(alignments)
        .zip(unmapped_mate_info)
        .map(|((r, a), m)| (r, a, m))
        .collect();

    let chunk_size: usize = 64;

    let chunks: Vec<Vec<u8>> = triples
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut estimate = 0usize;
            for (read, alns, _mate) in chunk.iter() {
                let qual_len = read.qual.as_ref().map_or(1, |q| q.len());
                let per = read.id.len() + read.seq.len() + qual_len + 96;
                let count = if alns.is_empty() { 1 } else { alns.len() };
                estimate += per * count;
            }
            let mut buf = Vec::with_capacity(estimate);
            let mut extra_tags: Vec<u8> = Vec::new();
            for (read, alns, mate) in chunk.iter() {
                match output_cfg.format {
                    EmitFormat::Sam => {
                        if alns.is_empty() {
                            formatter.append_unmapped_with_mate(&mut buf, read, mate.as_ref());
                        } else {
                            extra_tags.clear();
                            let mut has_extra = false;
                            if output_cfg.write_xa {
                                if formatter.append_xa(&mut extra_tags, alns) {
                                    has_extra = true;
                                }
                            }
                            if output_cfg.write_sa {
                                if formatter.append_sa(&mut extra_tags, alns) {
                                    has_extra = true;
                                }
                            }
                            let extra = if has_extra {
                                Some(extra_tags.as_slice())
                            } else {
                                None
                            };
                            for (idx, aln) in alns.iter().enumerate() {
                                let tags = if idx == 0 { extra } else { None };
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
    let mut out = Vec::with_capacity(total);
    for c in chunks {
        out.extend_from_slice(&c);
    }
    out
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
