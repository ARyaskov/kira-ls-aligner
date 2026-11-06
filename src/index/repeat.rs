use std::io::{Read, Seek, Write};

use anyhow::Result;
use rustc_hash::FxHashSet;

use super::{
    FlatMin, OCC_DISK_SIZE, Occ, OccStorage, OffsetStorage, checked_advance, ensure_remaining,
    read_u32, read_u64, strand_to_u8, write_u8, write_u32, write_u64,
};
use crate::sketch::{MinimizerConfig, extended_minimizer_hash, minimizers};
use crate::types::Reference;

pub(super) const EXTENSION: usize = 8;
pub(super) const MAX_OCC: usize = 64;

/// Longer exact seeds for occurrences hidden by a truncated primary bucket.
/// Buckets that remain repetitive after extension are not retained.
#[derive(Clone, Debug)]
pub(super) struct RepeatRescueIndex {
    hashes: Vec<u64>,
    bucket_offsets: OffsetStorage,
    flat_occs: OccStorage,
}

impl RepeatRescueIndex {
    pub(super) fn build_from_flat(
        reference: &Reference,
        k: usize,
        max_occ: usize,
        flat: &[FlatMin],
    ) -> Option<Self> {
        let mut extended = Vec::new();
        let mut i = 0;
        while i < flat.len() {
            let hash = flat[i].hash;
            let mut j = i + 1;
            while j < flat.len() && flat[j].hash == hash {
                j += 1;
            }
            if j - i > max_occ {
                extend_occurrences(reference, k, &flat[i..j], &mut extended);
            }
            i = j;
        }
        Self::from_flat(extended)
    }

    pub(super) fn build_from_reference(
        reference: &Reference,
        k: usize,
        w: usize,
        overcap_hashes: &FxHashSet<u64>,
    ) -> Option<Self> {
        if overcap_hashes.is_empty() {
            return None;
        }
        let cfg = MinimizerConfig { k, w };
        let mut extended = Vec::new();
        for (ref_id, seq) in reference.sequences.iter().enumerate() {
            let mins = minimizers(seq.bases(None), &cfg);
            let selected: Vec<FlatMin> = mins
                .into_iter()
                .filter(|m| overcap_hashes.contains(&m.hash))
                .map(|m| {
                    FlatMin::pack(
                        m.hash,
                        u32::try_from(ref_id).expect("reference id exceeds u32"),
                        m.pos,
                        m.strand,
                    )
                })
                .collect();
            extend_occurrences(reference, k, &selected, &mut extended);
        }
        Self::from_flat(extended)
    }

    fn from_flat(mut flat: Vec<FlatMin>) -> Option<Self> {
        if flat.is_empty() {
            return None;
        }
        flat.par_sort_unstable_by_key(|m| (m.hash, m.ref_strand, m.pos));

        let mut hashes = Vec::new();
        let mut offsets = vec![0u32];
        let mut occs = Vec::new();
        let mut i = 0;
        while i < flat.len() {
            let hash = flat[i].hash;
            let mut j = i + 1;
            while j < flat.len() && flat[j].hash == hash {
                j += 1;
            }
            if j - i <= MAX_OCC {
                hashes.push(hash);
                for item in &flat[i..j] {
                    occs.push(item.to_occ());
                }
                offsets
                    .push(u32::try_from(occs.len()).expect("repeat-rescue occurrences exceed u32"));
            }
            i = j;
        }
        if hashes.is_empty() {
            return None;
        }
        Some(Self {
            hashes,
            bucket_offsets: OffsetStorage::Owned(offsets),
            flat_occs: OccStorage::Owned(occs),
        })
    }

    pub(super) fn bucket(&self, mmap: Option<&[u8]>, hash: u64) -> Option<(usize, usize)> {
        let id = self.hashes.binary_search(&hash).ok()?;
        let start = self.bucket_offsets.get(id, mmap) as usize;
        let end = self.bucket_offsets.get(id + 1, mmap) as usize;
        Some((start, end))
    }

    pub(super) fn occ_at(&self, mmap: Option<&[u8]>, idx: usize) -> Occ {
        self.flat_occs.get(idx, mmap)
    }

    pub(super) fn size(&self, mmap: Option<&[u8]>) -> usize {
        let n_occs = self.bucket_offsets.get(self.bucket_offsets.len() - 1, mmap) as usize;
        4 + 8
            + self.hashes.len() * 8
            + 8
            + self.bucket_offsets.len() * 4
            + 8
            + n_occs * OCC_DISK_SIZE
    }

    pub(super) fn write<W: Write>(&self, writer: &mut W, mmap: Option<&[u8]>) -> Result<()> {
        write_u32(writer, EXTENSION as u32)?;
        write_u64(writer, self.hashes.len() as u64)?;
        for hash in &self.hashes {
            writer.write_all(&hash.to_le_bytes())?;
        }
        write_u64(writer, self.bucket_offsets.len() as u64)?;
        for i in 0..self.bucket_offsets.len() {
            write_u32(writer, self.bucket_offsets.get(i, mmap))?;
        }
        let n_occs = self.bucket_offsets.get(self.bucket_offsets.len() - 1, mmap) as usize;
        write_u64(writer, n_occs as u64)?;
        for i in 0..n_occs {
            let occ = self.flat_occs.get(i, mmap);
            write_u32(writer, occ.ref_id)?;
            write_u32(writer, occ.pos)?;
            write_u8(writer, strand_to_u8(occ.strand))?;
            writer.write_all(&[0u8, 0u8, 0u8])?;
        }
        Ok(())
    }

    pub(super) fn read<R: Read + Seek>(reader: &mut R, file_len: u64) -> Result<Self> {
        let extension = read_u32(reader)? as usize;
        if extension != EXTENSION {
            anyhow::bail!("unsupported repeat-rescue seed extension {extension}");
        }
        let n_hashes = read_u64(reader)? as usize;
        let hash_bytes = n_hashes
            .checked_mul(8)
            .ok_or_else(|| anyhow::anyhow!("repeat-rescue hash table size overflow"))?;
        ensure_remaining(reader, hash_bytes, file_len, "repeat-rescue hashes")?;
        let mut hashes = Vec::with_capacity(n_hashes);
        for _ in 0..n_hashes {
            hashes.push(read_u64(reader)?);
        }
        if !hashes.windows(2).all(|w| w[0] < w[1]) {
            anyhow::bail!("invalid index: repeat-rescue hashes are not strictly sorted");
        }

        let n_offsets = read_u64(reader)? as usize;
        if Some(n_offsets) != n_hashes.checked_add(1) {
            anyhow::bail!("invalid index: repeat-rescue offset count does not match hashes");
        }
        let offset_bytes = n_offsets
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("repeat-rescue offset table size overflow"))?;
        ensure_remaining(reader, offset_bytes, file_len, "repeat-rescue offsets")?;
        let offset = reader.stream_position()? as usize;
        let mut previous = 0u32;
        for i in 0..n_offsets {
            let value = read_u32(reader)?;
            if (i == 0 && value != 0) || value < previous {
                anyhow::bail!("invalid index: repeat-rescue offsets are not monotonic");
            }
            previous = value;
        }
        let bucket_offsets = OffsetStorage::Mmap {
            offset,
            len: n_offsets,
        };

        let n_occs = read_u64(reader)? as usize;
        if previous as usize != n_occs {
            anyhow::bail!("invalid index: repeat-rescue occurrence count mismatch");
        }
        let occ_bytes = n_occs
            .checked_mul(OCC_DISK_SIZE)
            .ok_or_else(|| anyhow::anyhow!("repeat-rescue occurrence table size overflow"))?;
        ensure_remaining(reader, occ_bytes, file_len, "repeat-rescue occurrences")?;
        let occ_offset = reader.stream_position()? as usize;
        checked_advance(reader, occ_bytes, file_len, "repeat-rescue occurrences")?;
        Ok(Self {
            hashes,
            bucket_offsets,
            flat_occs: OccStorage::Mmap { offset: occ_offset },
        })
    }
}

fn extend_occurrences(
    reference: &Reference,
    k: usize,
    occurrences: &[FlatMin],
    out: &mut Vec<FlatMin>,
) {
    for item in occurrences {
        let occ = item.to_occ();
        let bases = reference.sequences[occ.ref_id as usize].bases(None);
        if let Some(hash) = extended_minimizer_hash(bases, occ.pos, k, occ.strand, EXTENSION) {
            out.push(FlatMin::pack(hash, occ.ref_id, occ.pos, occ.strand));
        }
    }
}

use rayon::prelude::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::MinimizerIndex;
    use crate::types::{RefBases, RefSeq};

    #[test]
    fn longer_seeds_disambiguate_repetitive_primary_buckets() {
        let unit = b"ACGTTGCAAGTC";
        let mut bases = Vec::new();
        for i in 0..4 {
            bases.extend_from_slice(unit);
            bases.push(b"ACGT"[i]);
        }
        bases.extend_from_slice(unit);
        let reference = Reference {
            sequences: vec![RefSeq {
                name: "chr1".into(),
                bases: RefBases::Owned(bases.clone()),
            }],
        };
        let k = 5;
        let primary = MinimizerIndex::build_with_budget(&reference, k, 2, 1, usize::MAX);
        let overcap: FxHashSet<u64> = minimizers(&bases, &MinimizerConfig { k, w: 2 })
            .into_iter()
            .map(|m| m.hash)
            .filter(|&h| primary.bucket_count(None, h).is_some_and(|n| n > 1))
            .collect();
        let rescue = RepeatRescueIndex::build_from_reference(&reference, k, 2, &overcap)
            .expect("rescue index");
        assert!(!rescue.hashes.is_empty());
        assert!(rescue.bucket_offsets.len() > 2);
    }
}
