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
    let buf = serialize(input, writer.formatter(), read_group, output_cfg, max_alignments);
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
            if alns.len() > max_alignments {
                alns.truncate(max_alignments);
            }
        }
    }

    let triples: Vec<(ReadRecord, Vec<crate::types::Alignment>, Option<crate::types::MateInfo>)> =
        reads
            .into_iter()
            .zip(alignments.into_iter())
            .zip(unmapped_mate_info.into_iter())
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
                                    &mut buf,
                                    read,
                                    aln,
                                    read_group,
                                    tags,
                                    output_cfg,
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
