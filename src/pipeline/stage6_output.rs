use anyhow::{Context, Result};
use noodles_sam::alignment::RecordBuf;
use rayon::prelude::*;

use crate::io::{EmitFormat, OutputConfig, SamFormatter, SamWriter, append_cigar_ops, binary};
use crate::types::{Alignment, PairRole, ReadRecord};

use super::stage5_scoring::ScoredBatch;

/// One read ready for emission: the read, its alignments, the mate context
/// for an unmapped record, and the pre-built mate tags (`MC`/`MQ`/`ms`).
/// `(read, alignments, unmapped-mate info, mate tags, XA tag)`.
type EmitRecord = (
    ReadRecord,
    Vec<crate::types::Alignment>,
    Option<crate::types::MateInfo>,
    Option<Vec<u8>>,
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

    // `max_alignments` caps the primary/secondary *records* emitted per read
    // (supplementaries always go out). The XA tag is built from the full
    // candidate list first: bwa-mem reports the alternative hits in XA even
    // though it writes only the primary record, so pruning before the tag
    // would leave XA empty at the default settings.
    let xa_tags: Vec<Option<Vec<u8>>> =
        if output_cfg.write_xa && matches!(output_cfg.format, EmitFormat::Sam) {
            alignments
                .iter()
                .map(|alns| {
                    let mut buf = Vec::new();
                    formatter
                        .append_xa_capped(&mut buf, alns, output_cfg.xa_max)
                        .then_some(buf)
                })
                .collect()
        } else {
            vec![None; alignments.len()]
        };
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
        .zip(xa_tags)
        .map(|((((r, a), m), t), xa)| (r, a, m, t, xa))
        .collect();

    let chunk_size: usize = 64;

    let chunks: Vec<Vec<u8>> = quads
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut estimate = 0usize;
            for (read, alns, _mate, _tags, _xa) in chunk.iter() {
                let qual_len = read.qual.as_ref().map_or(1, |q| q.len());
                let per = read.id.len() + read.seq.len() + qual_len + 96;
                let count = if alns.is_empty() { 1 } else { alns.len() };
                estimate += per * count;
            }
            let mut buf = Vec::with_capacity(estimate);
            let mut extra_tags: Vec<u8> = Vec::new();
            for (read, alns, mate, mtags, xa) in chunk.iter() {
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
                            let has_supplementary = alns.iter().any(|a| a.is_supplementary);
                            for (idx, aln) in alns.iter().enumerate() {
                                // Primary: XA + SA + mate tags. Supplementary:
                                // its own SA (the other segments) + mate tags.
                                // Secondary: nothing beyond the core tags.
                                extra_tags.clear();
                                if idx == 0
                                    && let Some(xa) = xa
                                {
                                    extra_tags.extend_from_slice(xa);
                                }
                                if has_supplementary
                                    && output_cfg.write_sa
                                    && (idx == 0 || aln.is_supplementary)
                                {
                                    formatter.append_sa(&mut extra_tags, alns, idx);
                                }
                                if (idx == 0 || aln.is_supplementary)
                                    && let Some(t) = mtags
                                {
                                    extra_tags.extend_from_slice(t);
                                }
                                let tags = if extra_tags.is_empty() {
                                    None
                                } else {
                                    Some(extra_tags.as_slice())
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
    out.reserve(total);
    for c in chunks {
        out.extend_from_slice(&c);
    }
}

/// Build writer records in input order without serializing the batch to SAM.
pub(crate) fn binary_records(
    input: ScoredBatch,
    formatter: &SamFormatter,
    read_group: Option<&str>,
    cfg: OutputConfig,
    max_alignments: usize,
) -> Result<Vec<RecordBuf>> {
    const CHUNK_SIZE: usize = 128;
    let chunks: Result<Vec<Vec<RecordBuf>>> = input
        .reads
        .par_chunks(CHUNK_SIZE)
        .enumerate()
        .map(|(chunk, reads)| {
            let mut records = Vec::with_capacity(reads.len());
            for (offset, read) in reads.iter().enumerate() {
                let i = chunk * CHUNK_SIZE + offset;
                let alns = &input.alignments[i];
                let mate = cfg
                    .write_mate_tags
                    .then(|| mate_primary(&input.reads, &input.alignments, i))
                    .flatten();
                let comment = if cfg.append_comment {
                    read.comment
                        .as_deref()
                        .map(binary::parse_comment)
                        .transpose()
                        .with_context(|| format!("read {}", read.id))?
                } else {
                    None
                };
                if alns.is_empty() {
                    let mut record = binary::unmapped(read, input.unmapped_mate_info[i].as_ref())?;
                    if let Some(mate) = mate {
                        binary::mate_tags(&mut record, mate);
                    }
                    if let Some(comment) = &comment {
                        binary::append_comment(&mut record, comment)?;
                    }
                    records.push(record);
                    continue;
                }

                let mut report = alignment_report_filter(max_alignments);
                for (idx, aln) in alns.iter().enumerate().filter(|(_, aln)| report(aln)) {
                    let mut record = binary::mapped(read, aln, read_group, cfg)?;
                    if idx == 0 && cfg.write_xa && alns.len() > 1 {
                        let mut xa = Vec::new();
                        if formatter.append_xa_value(&mut xa, alns, cfg.xa_max) {
                            binary::string_tag(&mut record, *b"XA", xa);
                        }
                    }
                    if idx == 0 || aln.is_supplementary {
                        if cfg.write_sa && alns.len() > 1 {
                            let mut sa = Vec::new();
                            if formatter.append_sa_value(&mut sa, alns, idx) {
                                binary::string_tag(&mut record, *b"SA", sa);
                            }
                        }
                        if let Some(mate) = mate {
                            binary::mate_tags(&mut record, mate);
                        }
                    }
                    if let Some(comment) = &comment {
                        binary::append_comment(&mut record, comment)?;
                    }
                    records.push(record);
                }
            }
            Ok(records)
        })
        .collect();
    let chunks = chunks?;
    let mut records = Vec::with_capacity(chunks.iter().map(Vec::len).sum());
    for chunk in chunks {
        records.extend(chunk);
    }
    Ok(records)
}

/// The `MC`, `MQ` and `ms` SAM fields derived from the mate's primary alignment.
fn mate_tag_bytes(
    reads: &[ReadRecord],
    alignments: &[Vec<Alignment>],
    i: usize,
) -> Option<Vec<u8>> {
    let primary = mate_primary(reads, alignments, i)?;
    let mut out = Vec::with_capacity(48);
    out.extend_from_slice(b"\tMC:Z:");
    append_cigar_ops(&mut out, &primary.cigar);
    out.extend_from_slice(b"\tMQ:i:");
    out.extend_from_slice(primary.mapq.to_string().as_bytes());
    out.extend_from_slice(b"\tms:i:");
    out.extend_from_slice(primary.as_score.to_string().as_bytes());
    Some(out)
}

fn mate_primary<'a>(
    reads: &[ReadRecord],
    alignments: &'a [Vec<Alignment>],
    i: usize,
) -> Option<&'a Alignment> {
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
    alignments[mate_idx].first()
}

fn retain_reported_alignments(
    alignments: &mut Vec<crate::types::Alignment>,
    max_alignments: usize,
) {
    alignments.retain(alignment_report_filter(max_alignments));
}

fn alignment_report_filter(max_alignments: usize) -> impl FnMut(&Alignment) -> bool {
    let mut primary_or_secondary = 0usize;
    move |aln| {
        if max_alignments == 0 || aln.is_supplementary {
            return true;
        }
        let keep = primary_or_secondary < max_alignments;
        primary_or_secondary += 1;
        keep
    }
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

    #[test]
    fn binary_records_match_parsed_sam() -> Result<()> {
        use crate::types::Strand;
        use noodles_sam as sam;

        let header_text = b"@SQ\tSN:chr1\tLN:1000\n@SQ\tSN:chr2\tLN:1000\n";
        let header = sam::io::Reader::new(&header_text[..]).read_header()?;
        let formatter = SamFormatter::from_catalog(std::sync::Arc::new(crate::io::ContigCatalog {
            names: vec!["chr1".into(), "chr2".into()],
            lens: vec![1000, 1000],
        }));
        let mut r1 = ReadRecord::new_unpaired(
            "pair".into(),
            b"ACGTAACCTG".to_vec(),
            Some(b"ABCDEFGHIJ".to_vec()),
        );
        r1.pair_role = PairRole::R1;
        r1.comment = Some("BC:Z:ACGT\tFI:i:1\tXY:B:c,-2,3".into());
        let mut r2 = r1.clone();
        r2.pair_role = PairRole::R2;
        r2.comment = Some("BC:Z:ACGT\tFI:i:2".into());
        let fasta = ReadRecord::new_unpaired("fasta".into(), b"ACGTAACCTG".to_vec(), None);
        let mut primary = alignment(100, false);
        primary.mapq = 42;
        primary.is_rev = true;
        primary.xs_score = Some(80);
        primary.mate = MateInfo {
            is_paired: true,
            mate_is_unmapped: true,
            is_first_in_pair: true,
            ..MateInfo::default()
        };
        let mut secondary = alignment(80, false);
        secondary.ref_id = 1;
        secondary.ref_start = 12;
        secondary.mate = primary.mate;
        let mut supp = alignment(60, true);
        supp.is_rev = true;
        supp.ref_id = 1;
        supp.xs_strand = Some(Strand::Reverse);
        supp.mate = primary.mate;
        supp.cigar = vec![
            CigarOp {
                len: 2,
                op: CigarKind::SoftClip,
            },
            CigarOp {
                len: 3,
                op: CigarKind::Match,
            },
            CigarOp {
                len: 77,
                op: CigarKind::Skipped,
            },
            CigarOp {
                len: 4,
                op: CigarKind::Match,
            },
            CigarOp {
                len: 1,
                op: CigarKind::SoftClip,
            },
        ];
        let mut other = alignment(100, false);
        other.mate = MateInfo {
            is_paired: true,
            is_proper_pair: true,
            mate_ref_id: Some(1),
            mate_pos: 123,
            tlen: -250,
            ..MateInfo::default()
        };
        let input = ScoredBatch {
            reads: vec![r1, r2, fasta],
            alignments: vec![vec![primary, secondary, supp], vec![], vec![other]],
            unmapped_mate_info: vec![
                None,
                Some(MateInfo {
                    is_paired: true,
                    is_second_in_pair: true,
                    mate_is_rev: true,
                    mate_ref_id: Some(0),
                    ..MateInfo::default()
                }),
                None,
            ],
            stats: Default::default(),
        };
        let rg = Some("@RG\tID:group\tSM:sample");
        for (mut cfg, max_alignments) in [(OutputConfig::full(), 1), (OutputConfig::fast(), 0)] {
            cfg.append_comment = true;
            cfg.soft_clip_supplementary = max_alignments == 0;
            cfg.split_as_secondary = max_alignments == 0;
            let sam_text = serialize(input.clone(), &formatter, rg, cfg, max_alignments);
            let mut reader = sam::io::Reader::new(sam_text.as_slice());
            let expected = reader
                .record_bufs(&header)
                .collect::<std::io::Result<Vec<_>>>()?;
            let actual = binary_records(input.clone(), &formatter, rg, cfg, max_alignments)?;
            assert_eq!(actual, expected);
        }
        Ok(())
    }
}
