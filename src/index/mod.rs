use std::fs::{File, OpenOptions};
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use memmap2::{Mmap, MmapMut};
use rustc_hash::FxHashMap;

use crate::sketch::{MinimizerConfig, minimizers};
use crate::types::{RefBases, RefSeq, Reference, Strand};

const INDEX_MAGIC: &[u8; 8] = b"KIRAIDX1";
const INDEX_VERSION: u32 = 1;

const OCC_DISK_SIZE: usize = 9;

#[derive(Clone, Debug)]
enum Bucket {
    Owned(Vec<Occ>),
    Mmap { offset: usize, len: usize },
}
/// Occurrence of a minimizer on the reference.
#[derive(Clone, Copy, Debug)]
pub struct Occ {
    pub ref_id: u32,
    pub pos: u32,
    pub strand: Strand,
}

/// A minimizer index table.
#[derive(Clone, Debug)]
pub struct MinimizerIndex {
    pub k: usize,
    pub w: usize,
    pub max_occ: usize,
    buckets: FxHashMap<u64, Bucket>,
}

impl MinimizerIndex {
    pub fn build(reference: &Reference, k: usize, w: usize, max_occ: usize) -> Self {
        let mut buckets: FxHashMap<u64, Bucket> = FxHashMap::default();
        let cfg = MinimizerConfig { k, w };
        for (rid, seq) in reference.sequences.iter().enumerate() {
            let mins = minimizers(seq.bases(None), &cfg);
            for m in mins {
                let entry = buckets
                    .entry(m.hash)
                    .or_insert_with(|| Bucket::Owned(Vec::new()));
                if let Bucket::Owned(vec) = entry {
                    vec.push(Occ {
                        ref_id: rid as u32,
                        pos: m.pos,
                        strand: m.strand,
                    });
                }
            }
        }
        Self {
            k,
            w,
            max_occ,
            buckets,
        }
    }

    pub fn bucket_len(&self, hash: u64) -> Option<usize> {
        self.buckets.get(&hash).map(|b| match b {
            Bucket::Owned(v) => v.len(),
            Bucket::Mmap { len, .. } => *len,
        })
    }

    pub fn for_each_occ<F: FnMut(Occ)>(&self, mmap: Option<&[u8]>, hash: u64, f: &mut F) {
        let Some(bucket) = self.buckets.get(&hash) else {
            return;
        };
        match bucket {
            Bucket::Owned(v) => {
                for occ in v.iter() {
                    f(*occ);
                }
            }
            Bucket::Mmap { offset, len } => {
                let data = mmap.expect("mmap required for mmap bucket");
                let start = *offset;
                let end = start + (*len * OCC_DISK_SIZE);
                let slice = &data[start..end];
                for i in 0..*len {
                    let base = i * OCC_DISK_SIZE;
                    let ref_id = u32::from_le_bytes([
                        slice[base],
                        slice[base + 1],
                        slice[base + 2],
                        slice[base + 3],
                    ]);
                    let pos = u32::from_le_bytes([
                        slice[base + 4],
                        slice[base + 5],
                        slice[base + 6],
                        slice[base + 7],
                    ]);
                    let strand = u8_to_strand(slice[base + 8]);
                    f(Occ {
                        ref_id,
                        pos,
                        strand,
                    });
                }
            }
        }
    }
}

/// Multi-resolution index for short and long reads.
#[derive(Clone, Debug)]
pub struct Index {
    pub reference: Reference,
    pub short: MinimizerIndex,
    pub long: MinimizerIndex,
    pub mmap: Option<Arc<Mmap>>,
}

/// Configuration for building a multi-resolution minimizer index.
#[derive(Clone, Copy, Debug)]
pub struct IndexConfig {
    pub short_k: usize,
    pub short_w: usize,
    pub long_k: usize,
    pub long_w: usize,
    pub max_occ: usize,
}

impl Index {
    fn mmap_bytes(&self) -> Option<&[u8]> {
        self.mmap.as_deref().map(|m| &m[..])
    }

    pub fn ref_bases(&self, ref_id: usize) -> &[u8] {
        self.reference.sequences[ref_id].bases(self.mmap_bytes())
    }

    pub fn bucket_len(&self, table: &MinimizerIndex, hash: u64) -> Option<usize> {
        table.bucket_len(hash)
    }

    pub fn for_each_occ<F: FnMut(Occ)>(&self, table: &MinimizerIndex, hash: u64, mut f: F) {
        table.for_each_occ(self.mmap_bytes(), hash, &mut f)
    }
    pub fn build(reference: Reference, cfg: IndexConfig) -> Self {
        let short = MinimizerIndex::build(&reference, cfg.short_k, cfg.short_w, cfg.max_occ);
        let long = MinimizerIndex::build(&reference, cfg.long_k, cfg.long_w, cfg.max_occ);
        Self {
            reference,
            short,
            long,
            mmap: None,
        }
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let total_size = compute_index_size(self);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path.as_ref())
            .context("create index file")?;
        file.set_len(total_size as u64)
            .context("resize index file")?;

        let mut mmap = unsafe { MmapMut::map_mut(&file).context("mmap index for write")? };
        let mut cursor = Cursor::new(&mut mmap[..]);

        cursor.write_all(INDEX_MAGIC)?;
        write_u32(&mut cursor, INDEX_VERSION)?;

        write_u32(&mut cursor, self.reference.sequences.len() as u32)?;
        for seq in &self.reference.sequences {
            write_bytes(&mut cursor, seq.name.as_bytes())?;
            let bases = seq.bases(self.mmap_bytes());
            write_bytes(&mut cursor, bases)?;
        }

        write_minimizer_index(&mut cursor, &self.short, self.mmap_bytes())?;
        write_minimizer_index(&mut cursor, &self.long, self.mmap_bytes())?;
        mmap.flush().context("flush index mmap")?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref()).context("open index file")?;
        let mmap = Arc::new(unsafe { Mmap::map(&file).context("mmap index for read")? });
        let mut cursor = Cursor::new(&mmap[..]);

        let mut magic = [0u8; 8];
        cursor.read_exact(&mut magic).context("read index magic")?;
        if &magic != INDEX_MAGIC {
            anyhow::bail!("invalid index magic");
        }
        let version = read_u32(&mut cursor)?;
        if version != INDEX_VERSION {
            anyhow::bail!("unsupported index version {}", version);
        }

        let seq_count = read_u32(&mut cursor)? as usize;
        let mut sequences = Vec::with_capacity(seq_count);
        for _ in 0..seq_count {
            let name = String::from_utf8(read_bytes(&mut cursor)?).context("decode seq name")?;
            let bases_len = read_u32(&mut cursor)? as usize;
            let bases_offset = cursor.position() as usize;
            cursor.set_position(cursor.position() + bases_len as u64);
            sequences.push(RefSeq {
                name,
                bases: RefBases::Mmap {
                    offset: bases_offset,
                    len: bases_len,
                },
            });
        }

        let reference = Reference { sequences };
        let short = read_minimizer_index_mmap(&mut cursor)?;
        let long = read_minimizer_index_mmap(&mut cursor)?;
        Ok(Self {
            reference,
            short,
            long,
            mmap: Some(mmap),
        })
    }
}

fn compute_index_size(index: &Index) -> usize {
    let mut size = 0usize;
    size += INDEX_MAGIC.len();
    size += 4; // version
    size += 4; // seq count
    for seq in &index.reference.sequences {
        size += 4 + seq.name.len();
        size += 4 + seq.len(index.mmap.as_deref().map(|m| &m[..]));
    }
    size += minimizer_index_size(&index.short);
    size += minimizer_index_size(&index.long);
    size
}

fn minimizer_index_size(idx: &MinimizerIndex) -> usize {
    let mut size = 0usize;
    size += 4 * 3; // k, w, max_occ
    size += 8; // bucket count
    for occs in idx.buckets.values() {
        size += 8; // hash
        size += 4; // occ count
        size += match occs {
            Bucket::Owned(v) => v.len() * (4 + 4 + 1),
            Bucket::Mmap { len, .. } => len * (4 + 4 + 1),
        };
    }
    size
}

fn write_minimizer_index<W: Write>(
    writer: &mut W,
    idx: &MinimizerIndex,
    mmap: Option<&[u8]>,
) -> Result<()> {
    write_u32(writer, idx.k as u32)?;
    write_u32(writer, idx.w as u32)?;
    write_u32(writer, idx.max_occ as u32)?;

    let mut buckets: Vec<(&u64, &Bucket)> = idx.buckets.iter().collect();
    buckets.sort_by_key(|(h, _)| **h);
    write_u64(writer, buckets.len() as u64)?;
    for (hash, occs) in buckets {
        write_u64(writer, *hash)?;
        match occs {
            Bucket::Owned(vec) => {
                write_u32(writer, vec.len() as u32)?;
                for occ in vec {
                    write_u32(writer, occ.ref_id)?;
                    write_u32(writer, occ.pos)?;
                    write_u8(writer, strand_to_u8(occ.strand))?;
                }
            }
            Bucket::Mmap { offset, len } => {
                write_u32(writer, *len as u32)?;
                let data = mmap.expect("mmap required for mmap bucket");
                let start = *offset;
                let end = start + (*len * OCC_DISK_SIZE);
                writer.write_all(&data[start..end])?;
            }
        }
    }
    Ok(())
}

fn read_minimizer_index_mmap<R: Read + Seek>(reader: &mut R) -> Result<MinimizerIndex> {
    let k = read_u32(reader)? as usize;
    let w = read_u32(reader)? as usize;
    let max_occ = read_u32(reader)? as usize;
    let bucket_count = read_u64(reader)? as usize;

    let mut buckets: FxHashMap<u64, Bucket> = FxHashMap::default();
    for _ in 0..bucket_count {
        let hash = read_u64(reader)?;
        let occ_count = read_u32(reader)? as usize;
        let offset = reader.stream_position()? as usize;
        let skip = (occ_count * OCC_DISK_SIZE) as u64;
        reader.seek(SeekFrom::Current(skip as i64))?;
        buckets.insert(
            hash,
            Bucket::Mmap {
                offset,
                len: occ_count,
            },
        );
    }

    Ok(MinimizerIndex {
        k,
        w,
        max_occ,
        buckets,
    })
}

fn write_u32<W: Write>(writer: &mut W, value: u32) -> Result<()> {
    writer
        .write_all(&value.to_le_bytes())
        .context("write u32")?;
    Ok(())
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf).context("read u32")?;
    Ok(u32::from_le_bytes(buf))
}

fn write_u64<W: Write>(writer: &mut W, value: u64) -> Result<()> {
    writer
        .write_all(&value.to_le_bytes())
        .context("write u64")?;
    Ok(())
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf).context("read u64")?;
    Ok(u64::from_le_bytes(buf))
}

fn write_u8<W: Write>(writer: &mut W, value: u8) -> Result<()> {
    writer.write_all(&[value]).context("write u8")?;
    Ok(())
}

fn write_bytes<W: Write>(writer: &mut W, bytes: &[u8]) -> Result<()> {
    write_u32(writer, bytes.len() as u32)?;
    writer.write_all(bytes).context("write bytes")?;
    Ok(())
}

fn read_bytes<R: Read>(reader: &mut R) -> Result<Vec<u8>> {
    let len = read_u32(reader)? as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).context("read bytes")?;
    Ok(buf)
}

fn strand_to_u8(strand: Strand) -> u8 {
    match strand {
        Strand::Forward => 0,
        Strand::Reverse => 1,
    }
}

fn u8_to_strand(v: u8) -> Strand {
    if v == 0 {
        Strand::Forward
    } else {
        Strand::Reverse
    }
}
