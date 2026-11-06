//! Build binary-writer records from the same field rules used by SAM.

use std::io::Read as _;
use std::ops::Range;

use anyhow::{Context, Result, ensure};
use noodles_sam as sam;
use sam::alignment::{
    RecordBuf,
    record::{
        Flags, MappingQuality,
        cigar::{Op, op::Kind},
        data::field::Tag,
    },
    record_buf::{
        Data,
        data::field::{Value, value::Array},
    },
};

use crate::types::{Alignment, CigarKind, MateInfo, ReadRecord};

use super::{
    AlignmentTag, OutputConfig, alignment_clip, alignment_tags, append_cigar_ops,
    append_reverse_complement_range, checked_cigar, mapped_mate_fields, sam_flag_with,
    terminal_cigar_op, unmapped_fields,
};

pub(crate) fn mapped(
    read: &ReadRecord,
    aln: &Alignment,
    read_group: Option<&str>,
    cfg: OutputConfig,
) -> Result<RecordBuf> {
    let (hard_clip, keep) = alignment_clip(aln, read.seq.len(), cfg);
    let mut record = read_fields(read, keep, aln.is_rev, true)?;
    *record.flags_mut() = Flags::from(sam_flag_with(aln, cfg.split_as_secondary));
    *record.reference_sequence_id_mut() = Some(aln.ref_id as usize);
    *record.alignment_start_mut() = (aln.ref_start as usize + 1).try_into().ok();
    *record.mapping_quality_mut() = MappingQuality::new(aln.mapq);

    let ops = checked_cigar(&aln.cigar, read.seq.len() as u32);
    *record.cigar_mut() = ops
        .iter()
        .enumerate()
        .map(|(i, op)| {
            let kind = match op.op {
                CigarKind::Match => Kind::Match,
                CigarKind::Ins => Kind::Insertion,
                CigarKind::Del => Kind::Deletion,
                CigarKind::Skipped => Kind::Skip,
                CigarKind::SoftClip if hard_clip && terminal_cigar_op(i, ops.len()) => {
                    Kind::HardClip
                }
                CigarKind::SoftClip => Kind::SoftClip,
            };
            Op::new(kind, op.len as usize)
        })
        .collect();

    let (mate, tlen) = mapped_mate_fields(aln);
    set_mate_position(&mut record, mate);
    *record.template_length_mut() = tlen;
    let mut duplicate = None;
    alignment_tags(aln, read_group, cfg, |tag, value| {
        let value = match value {
            AlignmentTag::Unsigned(n) => Value::from(n),
            AlignmentTag::Signed(n) => Value::from(n),
            AlignmentTag::Text(text) => Value::String(text.into()),
            AlignmentTag::Char(c) => Value::Character(c),
        };
        let tag = Tag::from(tag);
        if record.data_mut().insert(tag, value).is_some() {
            duplicate = Some(tag);
        }
    });
    ensure!(
        duplicate.is_none(),
        "duplicate alignment tag: {duplicate:?}"
    );
    Ok(record)
}

pub(crate) fn unmapped(read: &ReadRecord, mate: Option<&MateInfo>) -> Result<RecordBuf> {
    let mut record = read_fields(read, 0..read.seq.len(), false, false)?;
    let (flags, placement) = unmapped_fields(mate);
    *record.flags_mut() = Flags::from(flags);
    *record.mapping_quality_mut() = MappingQuality::new(0);
    if let Some((id, pos)) = placement {
        *record.reference_sequence_id_mut() = Some(id as usize);
        *record.alignment_start_mut() = (pos as usize + 1).try_into().ok();
    }
    set_mate_position(&mut record, placement);
    Ok(record)
}

fn set_mate_position(record: &mut RecordBuf, position: Option<(u32, u32)>) {
    if let Some((id, pos)) = position {
        *record.mate_reference_sequence_id_mut() = Some(id as usize);
        *record.mate_alignment_start_mut() = (pos as usize + 1).try_into().ok();
    }
}

fn read_fields(
    read: &ReadRecord,
    keep: Range<usize>,
    reverse: bool,
    mapped: bool,
) -> Result<RecordBuf> {
    let mut record = RecordBuf::default();
    ensure!(!read.id.is_empty(), "empty SAM record name");
    if read.id != "*" {
        *record.name_mut() = Some(read.id.as_bytes().into());
    }
    let seq = record.sequence_mut().as_mut();
    seq.reserve(keep.len());
    if reverse {
        append_reverse_complement_range(seq, &read.seq, keep.start, keep.end);
    } else {
        seq.extend_from_slice(&read.seq[keep.clone()]);
    }
    let Some(qual) = read.qual.as_deref().filter(|q| !q.is_empty() || mapped) else {
        return Ok(record);
    };
    let qual = if mapped {
        if qual.len() != read.seq.len() || keep.is_empty() {
            return Ok(record);
        }
        if reverse {
            &qual[qual.len() - keep.end..qual.len() - keep.start]
        } else {
            &qual[keep]
        }
    } else {
        qual
    };
    if qual == b"*" {
        return Ok(record);
    }
    ensure!(
        qual.len() == seq.len(),
        "SAM quality length does not match sequence"
    );
    ensure!(
        qual.iter().all(u8::is_ascii_graphic),
        "invalid SAM quality scores"
    );
    let dst = record.quality_scores_mut().as_mut();
    if reverse {
        dst.extend(qual.iter().rev().map(|q| q - b'!'));
    } else {
        dst.extend(qual.iter().map(|q| q - b'!'));
    }
    Ok(record)
}

pub(crate) fn string_tag(record: &mut RecordBuf, tag: [u8; 2], value: Vec<u8>) {
    record
        .data_mut()
        .insert(Tag::from(tag), Value::String(value.into()));
}

pub(crate) fn mate_tags(record: &mut RecordBuf, mate: &Alignment) {
    let mut cigar = Vec::new();
    append_cigar_ops(&mut cigar, &mate.cigar);
    string_tag(record, *b"MC", cigar);
    record
        .data_mut()
        .insert(Tag::from(*b"MQ"), Value::from(mate.mapq));
    record
        .data_mut()
        .insert(Tag::from(*b"ms"), Value::from(mate.as_score));
}

pub(crate) fn parse_comment(comment: &str) -> Result<Data> {
    // The user-supplied -C payload is already SAM tag text. Parse just that
    // payload with noodles, preserving all SAM tag types and validation.
    let input = (&b"*\t4\t*\t0\t0\t*\t*\t0\t0\t*\t*\t"[..])
        .chain(comment.as_bytes())
        .chain(&b"\n"[..]);
    let mut reader = sam::io::Reader::new(input);
    let mut record = RecordBuf::default();
    reader
        .read_record_buf(&sam::Header::default(), &mut record)
        .context("parse FASTQ comment tags")?;
    Ok(std::mem::take(record.data_mut()))
}

pub(crate) fn append_comment(record: &mut RecordBuf, data: &Data) -> Result<()> {
    for (tag, value) in data.iter() {
        ensure!(
            record.data_mut().insert(tag, value.clone()).is_none(),
            "duplicate FASTQ comment tag: {tag:?}"
        );
    }
    Ok(())
}

/// Estimate the owned data in a sorted batch, including optional tag payloads.
pub(crate) fn estimated_record_bytes(record: &RecordBuf) -> usize {
    let tag_bytes: usize = record
        .data()
        .iter()
        .map(|(_, value)| match value {
            Value::String(text) | Value::Hex(text) => text.len(),
            Value::Array(Array::Int8(v)) => v.len(),
            Value::Array(Array::UInt8(v)) => v.len(),
            Value::Array(Array::Int16(v)) => v.len() * 2,
            Value::Array(Array::UInt16(v)) => v.len() * 2,
            Value::Array(Array::Int32(v)) => v.len() * 4,
            Value::Array(Array::UInt32(v)) => v.len() * 4,
            Value::Array(Array::Float(v)) => v.len() * 4,
            _ => 0,
        })
        .sum();
    // Vec growth and allocator overhead vary, so leave headroom for capacities.
    let owned = size_of::<RecordBuf>()
        + record.name().map_or(0, |name| name.len())
        + record.sequence().len()
        + record.quality_scores().len()
        + size_of_val(record.cigar().as_ref())
        + record.data().len() * size_of::<(Tag, Value)>()
        + tag_bytes;
    owned + owned / 4
}
