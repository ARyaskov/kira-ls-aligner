use anyhow::Result;

use crate::io::{OutputConfig, SamWriter};
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
    let reads: Vec<ReadRecord> = input.reads;
    let mut alignments = input.alignments;

    if max_alignments > 0 {
        for alns in alignments.iter_mut() {
            if alns.len() > max_alignments {
                alns.truncate(max_alignments);
            }
        }
    }

    let mut buf = Vec::new();
    let mut estimate = 0usize;
    for (read, alns) in reads.iter().zip(alignments.iter()) {
        let qual_len = read.qual.as_ref().map_or(1, |q| q.len());
        let per = read.id.len() + read.seq.len() + qual_len + 96;
        let count = if alns.is_empty() { 1 } else { alns.len() };
        estimate += per * count;
    }
    buf.reserve(estimate);

    let mut extra_tags: Vec<u8> = Vec::new();
    for (read, alns) in reads.iter().zip(alignments.iter()) {
        if alns.is_empty() {
            writer.append_unmapped(&mut buf, read);
        } else {
            extra_tags.clear();
            let mut has_extra = false;
            if output_cfg.write_xa {
                if writer.append_xa(&mut extra_tags, alns) {
                    has_extra = true;
                }
            }
            if output_cfg.write_sa {
                if writer.append_sa(&mut extra_tags, alns) {
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
                writer.append_alignment(&mut buf, read, aln, read_group, tags, output_cfg);
            }
        }
    }

    writer.write_batch(&buf)
}
