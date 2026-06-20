//! BED12 / BED6 parser for `--junc-bed` annotation-guided splice alignment.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{Context, Result};

use crate::types::{Reference, Strand};

/// Per-junction record: donor position (intron 5'), acceptor position (intron 3', exclusive).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Junction {
    pub donor: u32,
    pub acceptor: u32,
}

/// In-memory index of known splice junctions.
#[derive(Debug, Default)]
pub struct JunctionIndex {
    /// `per_ref[ref_id]` is the set of junctions on that contig.
    per_ref: HashMap<u32, HashSet<Junction>>,
    /// Strand annotation per `(ref_id, junction)`.
    strands: HashMap<(u32, Junction), Strand>,
    /// Counts for the [`stats`] report. Updated as we parse.
    n_lines: usize,
    n_junctions: usize,
    n_skipped: usize,
}

impl JunctionIndex {
    /// Load a BED file from disk.
    pub fn from_bed_path<P: AsRef<Path>>(path: P, reference: &Reference) -> Result<Self> {
        let f = File::open(path.as_ref())
            .with_context(|| format!("open BED file {}", path.as_ref().display()))?;
        let r = BufReader::new(f);
        Self::from_bed_reader(r, reference)
    }

    /// Same as `from_bed_path` but reads from any `BufRead`.
    pub fn from_bed_reader<R: BufRead>(reader: R, reference: &Reference) -> Result<Self> {
        let name_to_id: HashMap<String, u32> = reference
            .sequences
            .iter()
            .enumerate()
            .map(|(i, s)| (s.name.clone(), i as u32))
            .collect();

        let mut idx = JunctionIndex::default();
        for (lineno, line) in reader.lines().enumerate() {
            let line = match line {
                Ok(l) => l,
                Err(e) => {
                    eprintln!(
                        "[KIRA_JUNCBED] warning: read error on line {}: {}",
                        lineno + 1,
                        e
                    );
                    idx.n_skipped += 1;
                    continue;
                }
            };
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            // BED conventional comment / header prefixes.
            if trimmed.starts_with('#')
                || trimmed.starts_with("track ")
                || trimmed.starts_with("browser ")
            {
                continue;
            }
            idx.n_lines += 1;

            let cols: Vec<&str> = trimmed.split('\t').collect();
            if cols.len() < 3 {
                eprintln!(
                    "[KIRA_JUNCBED] warning: line {} has <3 columns, skipping: {:?}",
                    lineno + 1,
                    trimmed
                );
                idx.n_skipped += 1;
                continue;
            }
            let chrom = cols[0];
            let ref_id = match name_to_id.get(chrom) {
                Some(i) => *i,
                None => {
                    idx.n_skipped += 1;
                    continue;
                }
            };
            let chrom_start: u32 = match cols[1].parse() {
                Ok(v) => v,
                Err(_) => {
                    eprintln!(
                        "[KIRA_JUNCBED] warning: line {} bad chromStart, skipping",
                        lineno + 1
                    );
                    idx.n_skipped += 1;
                    continue;
                }
            };
            let chrom_end: u32 = match cols[2].parse() {
                Ok(v) => v,
                Err(_) => {
                    eprintln!(
                        "[KIRA_JUNCBED] warning: line {} bad chromEnd, skipping",
                        lineno + 1
                    );
                    idx.n_skipped += 1;
                    continue;
                }
            };
            // Optional strand (col 6).
            let strand = if cols.len() >= 6 {
                match cols[5] {
                    "+" => Some(Strand::Forward),
                    "-" => Some(Strand::Reverse),
                    _ => None,
                }
            } else {
                None
            };

            if cols.len() >= 12 {
                // BED12 — extract per-block introns.
                let block_count: usize = match cols[9].parse() {
                    Ok(v) => v,
                    Err(_) => {
                        eprintln!(
                            "[KIRA_JUNCBED] warning: line {} bad blockCount, skipping",
                            lineno + 1
                        );
                        idx.n_skipped += 1;
                        continue;
                    }
                };
                if block_count <= 1 {
                    // Single-block transcript ⇒ no introns. Skip silently.
                    continue;
                }
                let sizes: Vec<u32> = match parse_csv_u32(cols[10]) {
                    Some(v) => v,
                    None => {
                        eprintln!(
                            "[KIRA_JUNCBED] warning: line {} bad blockSizes, skipping",
                            lineno + 1
                        );
                        idx.n_skipped += 1;
                        continue;
                    }
                };
                let starts: Vec<u32> = match parse_csv_u32(cols[11]) {
                    Some(v) => v,
                    None => {
                        eprintln!(
                            "[KIRA_JUNCBED] warning: line {} bad blockStarts, skipping",
                            lineno + 1
                        );
                        idx.n_skipped += 1;
                        continue;
                    }
                };
                if sizes.len() != block_count || starts.len() != block_count {
                    eprintln!(
                        "[KIRA_JUNCBED] warning: line {} block list length mismatch, skipping",
                        lineno + 1
                    );
                    idx.n_skipped += 1;
                    continue;
                }
                for k in 0..block_count - 1 {
                    let donor = chrom_start
                        .saturating_add(starts[k])
                        .saturating_add(sizes[k]);
                    let acceptor = chrom_start.saturating_add(starts[k + 1]);
                    if acceptor <= donor {
                        // Defensive: overlapping/zero-length intron skip.
                        continue;
                    }
                    idx.insert(ref_id, Junction { donor, acceptor }, strand);
                }
            } else {
                // BED6: treat the start/end as a single intron.
                let donor = chrom_start;
                let acceptor = chrom_end;
                if acceptor > donor {
                    idx.insert(ref_id, Junction { donor, acceptor }, strand);
                }
            }
        }
        eprintln!(
            "[KIRA_JUNCBED] loaded {} junctions from {} record(s); {} line(s) skipped",
            idx.n_junctions, idx.n_lines, idx.n_skipped
        );
        Ok(idx)
    }

    fn insert(&mut self, ref_id: u32, junction: Junction, strand: Option<Strand>) {
        let set = self.per_ref.entry(ref_id).or_default();
        if set.insert(junction) {
            self.n_junctions += 1;
        }
        if let Some(s) = strand {
            self.strands.insert((ref_id, junction), s);
        }
    }

    /// Look up a junction with positional tolerance.
    pub fn lookup(&self, ref_id: u32, donor: u32, acceptor: u32, tolerance: u32) -> Option<Strand> {
        let set = self.per_ref.get(&ref_id)?;
        // Fast path: exact match.
        let exact = Junction { donor, acceptor };
        if set.contains(&exact) {
            return self.strands.get(&(ref_id, exact)).copied();
        }
        for dd in -(tolerance as i64)..=(tolerance as i64) {
            for da in -(tolerance as i64)..=(tolerance as i64) {
                let d = (donor as i64 + dd) as u32;
                let a = (acceptor as i64 + da) as u32;
                let j = Junction {
                    donor: d,
                    acceptor: a,
                };
                if set.contains(&j) {
                    return self
                        .strands
                        .get(&(ref_id, j))
                        .copied()
                        .or(Some(Strand::Forward));
                }
            }
        }
        None
    }

    /// `true` when at least one junction was loaded for `ref_id`.
    pub fn has_ref(&self, ref_id: u32) -> bool {
        self.per_ref.contains_key(&ref_id)
    }

    /// Total number of unique junctions across all contigs.
    pub fn len(&self) -> usize {
        self.n_junctions
    }

    /// `true` if no junctions are loaded (e.g. BED file was empty or every line was skipped).
    pub fn is_empty(&self) -> bool {
        self.n_junctions == 0
    }
}

fn parse_csv_u32(s: &str) -> Option<Vec<u32>> {
    s.trim_end_matches(',')
        .split(',')
        .map(|x| x.trim().parse::<u32>().ok())
        .collect()
}

#[cfg(test)]
#[path = "../../tests/unit/alignment_junc_bed.rs"]
mod tests;
