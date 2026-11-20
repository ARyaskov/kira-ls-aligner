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
    let chunks = serialize_chunks(
        input,
        writer.formatter(),
        read_group,
        output_cfg,
        max_alignments,
    )?;
    writer.write_chunks(&chunks)
}

/// Serialize a `ScoredBatch` to a `Vec<u8>` ready to feed to `SamWriter::write_batch`.
pub fn serialize(
    input: ScoredBatch,
    formatter: &SamFormatter,
    read_group: Option<&str>,
    output_cfg: OutputConfig,
    max_alignments: usize,
) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    serialize_into(
        input,
        formatter,
        read_group,
        output_cfg,
        max_alignments,
        &mut out,
    )?;
    Ok(out)
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
) -> Result<()> {
    let chunks = serialize_chunks(input, formatter, read_group, output_cfg, max_alignments)?;
    out.clear();
    out.reserve(chunks.iter().map(Vec::len).sum());
    for chunk in chunks {
        out.extend_from_slice(&chunk);
    }
    Ok(())
}

pub(crate) fn serialize_chunks(
    input: ScoredBatch,
    formatter: &SamFormatter,
    read_group: Option<&str>,
    output_cfg: OutputConfig,
    max_alignments: usize,
) -> Result<Vec<Vec<u8>>> {
    let reads: Vec<ReadRecord> = input.reads;
    let mut alignments = input.alignments;
    let xa_has_alt = input.xa_has_alt;
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
                .zip(&xa_has_alt)
                .map(|(alns, has_alt)| {
                    let mut buf = Vec::new();
                    formatter
                        .append_xa_capped(
                            &mut buf,
                            alns,
                            output_cfg.xa_max,
                            output_cfg.xa_max_alt,
                            *has_alt,
                        )
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
    let read_group_id = output_cfg.read_group_id(read_group);

    quads
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
                            extra_tags.clear();
                            if let Some(id) = read_group_id {
                                extra_tags.extend_from_slice(b"\tRG:Z:");
                                extra_tags.extend_from_slice(id.as_bytes());
                            }
                            if let Some(tags) = mtags {
                                extra_tags.extend_from_slice(tags);
                            }
                            formatter.append_unmapped_full(
                                &mut buf,
                                read,
                                mate.as_ref(),
                                Some(&extra_tags),
                                output_cfg.append_comment,
                            )?;
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
                                )?;
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
            Ok(buf)
        })
        .collect()
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
    let read_group_id = cfg.read_group_id(read_group);
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
                    if let Some(id) = read_group_id {
                        binary::string_tag(&mut record, *b"RG", id.as_bytes().to_vec());
                    }
                    if let Some((mate, score)) = mate {
                        binary::mate_tags(&mut record, mate, score);
                    }
                    if let Some(comment) = &comment {
                        binary::append_comment(&mut record, comment)
                            .with_context(|| format!("read {}", read.id))?;
                    }
                    records.push(record);
                    continue;
                }

                let mut report = alignment_report_filter(max_alignments);
                for (idx, aln) in alns.iter().enumerate().filter(|(_, aln)| report(aln)) {
                    let mut record = binary::mapped(read, aln, read_group, cfg)?;
                    if idx == 0 && cfg.write_xa && alns.len() > 1 {
                        let mut xa = Vec::new();
                        if formatter.append_xa_value(
                            &mut xa,
                            alns,
                            cfg.xa_max,
                            cfg.xa_max_alt,
                            input.xa_has_alt[i],
                        ) {
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
                        if let Some((mate, score)) = mate {
                            binary::mate_tags(&mut record, mate, score);
                        }
                    }
                    if let Some(comment) = &comment {
                        binary::append_comment(&mut record, comment)
                            .with_context(|| format!("read {}", read.id))?;
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
    let (primary, score) = mate_primary(reads, alignments, i)?;
    let mut out = Vec::with_capacity(48);
    out.extend_from_slice(b"\tMC:Z:");
    append_cigar_ops(&mut out, &primary.cigar);
    out.extend_from_slice(b"\tMQ:i:");
    out.extend_from_slice(primary.mapq.to_string().as_bytes());
    out.extend_from_slice(b"\tms:i:");
    out.extend_from_slice(score.to_string().as_bytes());
    Some(out)
}

fn mate_primary<'a>(
    reads: &[ReadRecord],
    alignments: &'a [Vec<Alignment>],
    i: usize,
) -> Option<(&'a Alignment, u32)> {
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
    let score = mate
        .qual
        .as_deref()
        .unwrap_or_default()
        .iter()
        .filter_map(|q| q.checked_sub(b'!'))
        .filter(|&q| q >= 15)
        .map(u32::from)
        .sum();
    Some((primary, score))
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
    use crate::mapq::MapqConfig;
    use crate::pipeline::stage4_alignment::AlignBatch;
    use crate::pipeline::stage5_scoring;
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
        r2.qual = Some(b"IIIIIOOO!!".to_vec());
        r2.pair_role = PairRole::R2;
        let reads = vec![r1, r2];
        let mut mate_primary = alignment(97, false);
        mate_primary.mapq = 37;
        let alns = vec![vec![alignment(100, false)], vec![mate_primary]];
        let tags = mate_tag_bytes(&reads, &alns, 0).expect("R1 gets its mate's tags");
        assert_eq!(tags, b"\tMC:Z:10M\tMQ:i:37\tms:i:338");
        let tags = mate_tag_bytes(&reads, &alns, 1).expect("R2 too");
        assert_eq!(tags, b"\tMC:Z:10M\tMQ:i:0\tms:i:0");
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
        r2.qual = Some(b"IIIIIOOO!!".to_vec());
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
        // A second candidate on the first read plus the discriminating cap
        // below make the `has_alt` wiring observable: with `xa_max = 1` (ALT
        // cap 200, `write_xa` on), the two secondaries exceed the first cap,
        // so the XA tag appears in text and binary only when `binary_records`
        // receives `has_alt = true`; a stale `false` there would suppress XA
        // and diverge from the text path.
        let mut secondary2 = alignment(80, false);
        secondary2.ref_id = 1;
        secondary2.ref_start = 40;
        secondary2.mate = primary.mate;
        let input = ScoredBatch {
            reads: vec![r1, r2, fasta],
            alignments: vec![
                vec![primary, secondary, secondary2, supp],
                vec![],
                vec![other],
            ],
            xa_has_alt: vec![true, false, false],
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
            // Discriminating `has_alt` condition: two secondaries exceed the
            // first cap, so XA survives only when the binary path receives
            // `has_alt = true` (no effect on the `write_xa = false` case).
            cfg.xa_max = 1;
            let sam_text = serialize(input.clone(), &formatter, rg, cfg, max_alignments)?;
            let mut reader = sam::io::Reader::new(sam_text.as_slice());
            let expected = reader
                .record_bufs(&header)
                .collect::<std::io::Result<Vec<_>>>()?;
            let actual = binary_records(input.clone(), &formatter, rg, cfg, max_alignments)?;
            assert!(actual.iter().any(|r| r.flags().is_unmapped()));
            let group = sam::alignment::record_buf::data::field::Value::String("group".into());
            for record in &actual {
                assert_eq!(
                    record
                        .data()
                        .get(&sam::alignment::record::data::field::Tag::READ_GROUP),
                    Some(&group)
                );
            }
            assert_eq!(actual, expected);
        }
        Ok(())
    }

    #[test]
    fn comment_validation_is_consistent_in_sam_and_binary_output() {
        let formatter = SamFormatter::from_catalog(std::sync::Arc::new(crate::io::ContigCatalog {
            names: vec!["chr1".into(), "ALT".into()],
            lens: vec![1000, 1000],
        }));
        let mut cfg = OutputConfig::full();
        cfg.append_comment = true;
        for comment in ["NM:i:0", "unexpected free text", "BC:Z:A\tBC:Z:T"] {
            let mut batch = witness_batch(70);
            batch.reads[0].comment = Some(comment.into());
            assert!(
                serialize(batch.clone(), &formatter, None, cfg, 1).is_err(),
                "{comment}"
            );
            assert!(
                binary_records(batch, &formatter, None, cfg, 1).is_err(),
                "{comment}"
            );
        }
    }

    #[test]
    fn invalid_mapped_cigar_is_an_error_for_sam_and_binary_output() {
        let formatter = SamFormatter::from_catalog(std::sync::Arc::new(crate::io::ContigCatalog {
            names: vec!["chr1".into()],
            lens: vec![1000],
        }));
        let mut batch = witness_batch(70);
        batch.alignments[0][0].cigar[0].len = 9;
        for error in [
            serialize(batch.clone(), &formatter, None, OutputConfig::full(), 1).unwrap_err(),
            binary_records(batch, &formatter, None, OutputConfig::full(), 1).unwrap_err(),
        ] {
            let message = error.to_string();
            assert!(
                message.contains("read w: invalid mapped CIGAR 9M"),
                "{message}"
            );
            assert!(
                message.contains("consumes 9 query bases, expected 10"),
                "{message}"
            );
        }
    }

    /// bwa-mem `-h 2,200` drop-filter witness batch: primary 100, three
    /// non-ALT 95 secondaries and an ALT-contig candidate at
    /// `weak_alt_score`, run through stage 5 with an ALT mask so
    /// `xa_has_alt` reflects the 0.80 filter.
    fn witness_batch(weak_alt_score: i32) -> ScoredBatch {
        let mask = std::sync::Arc::from([false, true]);
        let witness_alignment = |score: i32, ref_id: u32| {
            let mut a = alignment(score, false);
            a.ref_id = ref_id;
            a
        };
        let alns = vec![
            witness_alignment(100, 0),
            witness_alignment(95, 0),
            witness_alignment(95, 0),
            witness_alignment(95, 0),
            witness_alignment(weak_alt_score, 1),
        ];
        stage5_scoring::run(
            AlignBatch {
                reads: vec![ReadRecord::new_unpaired("w".into(), vec![b'A'; 10], None)],
                alignments: vec![alns],
                unmapped_mate_info: vec![None],
                stats: Default::default(),
            },
            MapqConfig {
                short_read_len: 500,
                mapq_cap_short: 60,
                mapq_cap_long: 60,
                alt_mask: Some(mask),
            },
            None,
        )
    }

    fn witness_formatter() -> SamFormatter {
        SamFormatter::from_catalog(std::sync::Arc::new(crate::io::ContigCatalog {
            names: vec!["chr1".into(), "chr2".into()],
            lens: vec![1000, 1000],
        }))
    }

    fn witness_cfg() -> OutputConfig {
        let mut cfg = OutputConfig::full();
        cfg.xa_max = 2;
        cfg.xa_max_alt = 200;
        cfg
    }

    /// Text-mode witness for the bwa XA drop filter: with the weak (70 < 80)
    /// ALT candidate dropped, three filtered candidates exceed the first cap
    /// and XA is suppressed entirely; before the filter stage 5 reported
    /// `has_alt = true` over four candidates and the tag was emitted. An ALT
    /// candidate surviving the filter (95 >= 80) keeps the tag emitted.
    #[test]
    fn xa_drop_filter_suppresses_text_xa_for_the_bwa_witness() {
        let formatter = witness_formatter();
        let cfg = witness_cfg();
        let weak = serialize(witness_batch(70), &formatter, None, cfg, 1).unwrap();
        assert!(
            !String::from_utf8_lossy(&weak).contains("\tXA:Z:"),
            "XA must be suppressed when the only ALT candidate falls below the 0.80 drop ratio"
        );
        let strong = serialize(witness_batch(95), &formatter, None, cfg, 1).unwrap();
        assert!(
            String::from_utf8_lossy(&strong).contains("\tXA:Z:"),
            "XA must survive when the ALT candidate passes the 0.80 drop ratio"
        );
    }

    /// Binary-mode counterpart: `binary_records` forwards the stage 5
    /// `has_alt` into `append_xa_value`, which applies the same filtered set.
    #[test]
    fn xa_drop_filter_suppresses_binary_xa_for_the_bwa_witness() -> Result<()> {
        use noodles_sam::alignment::record::data::field::Tag;

        let formatter = witness_formatter();
        let cfg = witness_cfg();
        let has_xa = |records: &[RecordBuf]| {
            records
                .iter()
                .any(|r| r.data().iter().any(|(tag, _)| tag == Tag::from(*b"XA")))
        };
        let weak = binary_records(witness_batch(70), &formatter, None, cfg, 1)?;
        assert!(!has_xa(&weak));
        let strong = binary_records(witness_batch(95), &formatter, None, cfg, 1)?;
        assert!(has_xa(&strong));
        Ok(())
    }
}
