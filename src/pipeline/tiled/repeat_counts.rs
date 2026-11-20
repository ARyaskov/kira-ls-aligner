use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

use anyhow::{Context, Result, ensure};

use crate::pipeline::stage2_seeding::RepeatCountBlock;
use crate::types::ReadRecord;

const SUMMARY: &str = "read-repeat-minimum.bin";

/// Successive tile passes accumulate counts for the same query minimizer.
/// Each batch supplies counts in query order; scratch streams stay sequential.
pub(super) struct RepeatPass {
    previous: Option<BufReader<File>>,
    next: BufWriter<File>,
    single_tile: bool,
    final_tile: bool,
}

impl RepeatPass {
    pub(super) fn open(scratch: &Path, tile: usize, tiles: usize) -> Result<Self> {
        let partial = |slot: usize| scratch.join(format!("seed-counts-{}.bin", slot % 2));
        let final_tile = tile + 1 == tiles;
        let out = if final_tile {
            scratch.join(SUMMARY)
        } else {
            partial(tile)
        };
        let previous = if tile > 0 {
            Some(BufReader::new(File::open(partial(tile - 1))?))
        } else {
            None
        };
        Ok(Self {
            previous,
            next: BufWriter::new(File::create(out)?),
            single_tile: tiles == 1,
            final_tile,
        })
    }

    pub(super) fn record(
        &mut self,
        reads: &[ReadRecord],
        counts: &[RepeatCountBlock],
    ) -> Result<()> {
        if self.single_tile {
            // Preserve exactly what stage 2 produced, including skipped sketches.
            for read in reads {
                self.next.write_all(&read.repeat_min_occ.to_le_bytes())?;
            }
            return Ok(());
        }

        let mut read_counts = counts.iter().flat_map(RepeatCountBlock::reads);
        for _ in reads {
            let read_counts = read_counts
                .next()
                .context("missing tile occurrence counts")?;
            let n = u32::try_from(read_counts.len())?;
            if let Some(previous) = &mut self.previous {
                ensure!(
                    read_count(previous)? == n,
                    "read sketch changed between tile passes"
                );
            }
            if !self.final_tile {
                self.next.write_all(&n.to_le_bytes())?;
            }
            let mut minimum = None;
            for &local in read_counts {
                let count = match &mut self.previous {
                    Some(previous) => read_count(previous)?.saturating_add(local),
                    None => local,
                };
                if self.final_tile {
                    if count > 0 {
                        minimum = Some(minimum.map_or(count, |v: u32| v.min(count)));
                    }
                } else {
                    self.next.write_all(&count.to_le_bytes())?;
                }
            }
            if self.final_tile {
                self.next.write_all(&minimum.unwrap_or(0).to_le_bytes())?;
            }
        }
        ensure!(read_counts.next().is_none(), "extra tile occurrence counts");
        Ok(())
    }

    pub(super) fn finish(mut self) -> Result<()> {
        if let Some(previous) = &mut self.previous {
            ensure!(
                previous.fill_buf()?.is_empty(),
                "fewer reads in the current tile pass"
            );
        }
        self.next.flush().context("finish tile repeat counts")
    }
}

fn read_count(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0; 4];
    reader
        .read_exact(&mut bytes)
        .context("read tile occurrence count")?;
    Ok(u32::from_le_bytes(bytes))
}

pub(super) struct RepeatSummary(BufReader<File>);

impl RepeatSummary {
    pub(super) fn open(scratch: &Path) -> Result<Self> {
        Ok(Self(BufReader::new(File::open(scratch.join(SUMMARY))?)))
    }

    pub(super) fn restore(&mut self, reads: &mut [ReadRecord]) -> Result<()> {
        for read in reads {
            read.repeat_min_occ = read_count(&mut self.0)?;
        }
        Ok(())
    }

    pub(super) fn finish(mut self) -> Result<()> {
        ensure!(
            self.0.fill_buf()?.is_empty(),
            "fewer reads in the final tile merge"
        );
        Ok(())
    }
}
