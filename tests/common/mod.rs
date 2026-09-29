//! Shared helpers for the binary-driven integration tests
//! (`tests/e2e_binary.rs`, `tests/determinism.rs`).
//!
//! Everything here is deterministic: the reference and the simulated reads
//! come from a seeded LCG (the same recurrence `tests/mate_rescue.rs` uses),
//! so two test binaries built from the same source produce the same FASTA /
//! FASTQ bytes and the aligner's output can be compared run to run.
//!
//! Read names carry the truth locus in the format `kira_ls_aligner eval`
//! documents (`<name>:<contig>:<start>-<end>`, 0-based half-open). For a
//! pair the locus is the *fragment*; the per-mate truth is kept in
//! [`Dataset::truth`], keyed by (QNAME, mate role).

#![allow(dead_code)]

use std::collections::HashMap;
use std::ffi::OsString;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Path of the freshly built `kira_ls_aligner` binary under test.
pub const BIN: &str = env!("CARGO_BIN_EXE_kira_ls_aligner");

/// 64-bit LCG (Knuth's MMIX constants) — the recurrence `tests/mate_rescue.rs`
/// uses, so the synthetic sequence has the same statistical shape there and
/// here.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Lcg(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    /// Integer in `0..n`, taken from the high bits.
    pub fn below(&mut self, n: u64) -> u64 {
        (self.next_u64() >> 33) % n
    }

    pub fn base(&mut self) -> u8 {
        b"ACGT"[(self.next_u64() >> 33) as usize & 3]
    }
}

pub fn synth_contig(rng: &mut Lcg, len: usize) -> Vec<u8> {
    (0..len).map(|_| rng.base()).collect()
}

pub fn revcomp(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .rev()
        .map(|b| match b {
            b'A' => b'T',
            b'C' => b'G',
            b'G' => b'C',
            b'T' => b'A',
            other => *other,
        })
        .collect()
}

/// Which mate of a template a FASTQ record is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Role {
    Single,
    R1,
    R2,
}

/// Where a simulated read really comes from and how it was damaged.
#[derive(Clone, Debug)]
pub struct ReadTruth {
    pub contig: String,
    /// 0-based, half-open reference interval of the read's own bases.
    pub start: usize,
    pub end: usize,
    /// Read bases exactly as written to the FASTQ (R2 already reverse
    /// complemented, substitutions applied).
    pub seq: Vec<u8>,
    /// Number of substitutions introduced (0 for exact reads).
    pub mismatches: usize,
}

/// Temp directory that is removed on drop, even when a test panics.
pub struct TempDir {
    pub path: PathBuf,
}

impl TempDir {
    pub fn new(tag: &str) -> Self {
        let path = std::env::temp_dir().join(format!("kira_{}_{}", tag, std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create temp dir");
        TempDir { path }
    }

    pub fn join(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// Synthetic reference + simulated reads written to disk.
pub struct Dataset {
    pub dir: TempDir,
    pub reference: PathBuf,
    pub r1: PathBuf,
    pub r2: PathBuf,
    pub single: PathBuf,
    /// Contigs in FASTA order: (name, length).
    pub contigs: Vec<(String, usize)>,
    pub truth: HashMap<(String, Role), ReadTruth>,
    pub n_pairs: usize,
    pub n_single: usize,
}

/// Simulation knobs. Defaults are what `tests/e2e_binary.rs` uses.
pub struct SimParams {
    pub seed: u64,
    pub n_pairs: usize,
    pub n_single: usize,
    pub read_len: usize,
    pub frag_min: usize,
    pub frag_max: usize,
    /// One read in `mutate_every` gets 1–2 substitutions.
    pub mutate_every: usize,
}

impl Default for SimParams {
    fn default() -> Self {
        SimParams {
            seed: 0x1234_5678_9abc_def0,
            n_pairs: 300,
            n_single: 100,
            read_len: 150,
            frag_min: 300,
            frag_max: 400,
            mutate_every: 10,
        }
    }
}

fn write_fasta(path: &Path, contigs: &[(String, Vec<u8>)]) {
    let mut f = fs::File::create(path).expect("create fasta");
    for (name, seq) in contigs {
        writeln!(f, ">{name}").unwrap();
        for line in seq.chunks(80) {
            f.write_all(line).unwrap();
            f.write_all(b"\n").unwrap();
        }
    }
}

fn write_fastq(path: &Path, records: &[(String, Vec<u8>)]) {
    let mut f = fs::File::create(path).expect("create fastq");
    for (id, seq) in records {
        writeln!(f, "@{id}").unwrap();
        f.write_all(seq).unwrap();
        f.write_all(b"\n+\n").unwrap();
        f.write_all(&vec![b'I'; seq.len()]).unwrap();
        f.write_all(b"\n").unwrap();
    }
}

/// Apply `n` substitutions at distinct interior positions, always to a
/// different base. Returns the number applied.
fn mutate(rng: &mut Lcg, seq: &mut [u8], n: usize) -> usize {
    let mut done = Vec::new();
    while done.len() < n {
        // Keep away from the ends so the aligner has no reason to soft-clip.
        let pos = 20 + rng.below((seq.len() - 40) as u64) as usize;
        if done.contains(&pos) {
            continue;
        }
        let old = seq[pos];
        let mut new = old;
        while new == old {
            new = rng.base();
        }
        seq[pos] = new;
        done.push(pos);
    }
    done.len()
}

/// Build the reference (chrA 60 kb, chrB 40 kb) and the reads, and write
/// them under a fresh temp dir tagged `tag`.
pub fn build_dataset(tag: &str, p: &SimParams) -> Dataset {
    let mut rng = Lcg::new(p.seed);
    let contigs: Vec<(String, Vec<u8>)> = vec![
        ("chrA".to_string(), synth_contig(&mut rng, 60_000)),
        ("chrB".to_string(), synth_contig(&mut rng, 40_000)),
    ];

    let dir = TempDir::new(tag);
    let reference = dir.join("ref.fa");
    write_fasta(&reference, &contigs);

    let mut truth = HashMap::new();
    let mut r1_recs = Vec::with_capacity(p.n_pairs);
    let mut r2_recs = Vec::with_capacity(p.n_pairs);

    for i in 0..p.n_pairs {
        let ci = rng.below(contigs.len() as u64) as usize;
        let (cname, cseq) = &contigs[ci];
        let frag_len = p.frag_min + rng.below((p.frag_max - p.frag_min + 1) as u64) as usize;
        let fstart = rng.below((cseq.len() - frag_len) as u64) as usize;
        let fend = fstart + frag_len;
        let qname = format!("frag{i}:{cname}:{fstart}-{fend}");

        let mut r1 = cseq[fstart..fstart + p.read_len].to_vec();
        let mut r2 = revcomp(&cseq[fend - p.read_len..fend]);

        let (mm1, mm2) = if i % p.mutate_every == 0 {
            // Alternate which mate carries the damage; 1 or 2 substitutions.
            let n = 1 + rng.below(2) as usize;
            if i % (2 * p.mutate_every) == 0 {
                (mutate(&mut rng, &mut r1, n), 0)
            } else {
                (0, mutate(&mut rng, &mut r2, n))
            }
        } else {
            (0, 0)
        };

        truth.insert(
            (qname.clone(), Role::R1),
            ReadTruth {
                contig: cname.clone(),
                start: fstart,
                end: fstart + p.read_len,
                seq: r1.clone(),
                mismatches: mm1,
            },
        );
        truth.insert(
            (qname.clone(), Role::R2),
            ReadTruth {
                contig: cname.clone(),
                start: fend - p.read_len,
                end: fend,
                seq: r2.clone(),
                mismatches: mm2,
            },
        );
        r1_recs.push((format!("{qname}/1"), r1));
        r2_recs.push((format!("{qname}/2"), r2));
    }

    let mut se_recs = Vec::with_capacity(p.n_single);
    for i in 0..p.n_single {
        let ci = rng.below(contigs.len() as u64) as usize;
        let (cname, cseq) = &contigs[ci];
        let start = rng.below((cseq.len() - p.read_len) as u64) as usize;
        let end = start + p.read_len;
        let qname = format!("se{i}:{cname}:{start}-{end}");
        let mut seq = cseq[start..end].to_vec();
        // Half of the single-end reads come from the reverse strand.
        if rng.below(2) == 1 {
            seq = revcomp(&seq);
        }
        let mm = if i % p.mutate_every == 0 {
            let n = 1 + rng.below(2) as usize;
            mutate(&mut rng, &mut seq, n)
        } else {
            0
        };
        truth.insert(
            (qname.clone(), Role::Single),
            ReadTruth {
                contig: cname.clone(),
                start,
                end,
                seq: seq.clone(),
                mismatches: mm,
            },
        );
        se_recs.push((qname, seq));
    }

    let r1 = dir.join("reads_1.fq");
    let r2 = dir.join("reads_2.fq");
    let single = dir.join("single.fq");
    write_fastq(&r1, &r1_recs);
    write_fastq(&r2, &r2_recs);
    write_fastq(&single, &se_recs);

    Dataset {
        dir,
        reference,
        r1,
        r2,
        single,
        contigs: contigs.iter().map(|(n, s)| (n.clone(), s.len())).collect(),
        truth,
        n_pairs: p.n_pairs,
        n_single: p.n_single,
    }
}

/// Run the binary with `args`, returning the captured output. Panics with
/// stderr attached when the process fails.
pub fn run_bin(args: &[OsString]) -> Output {
    let out = Command::new(BIN)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("spawn {BIN}: {e}"));
    assert!(
        out.status.success(),
        "kira_ls_aligner {:?} failed with {}:\n{}",
        args,
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    out
}

/// Build the argument vector of `mem REF READS... -o OUT EXTRA...`.
pub fn mem_args(reference: &Path, reads: &[&Path], out: &Path, extra: &[&str]) -> Vec<OsString> {
    let mut args: Vec<OsString> = vec!["mem".into(), reference.into()];
    args.extend(reads.iter().map(|r| (*r).into()));
    args.push("-o".into());
    args.push(out.into());
    args.extend(extra.iter().map(|s| (*s).into()));
    args
}

/// Run `mem` with `extra` flags (after `REF READS... -o OUT`) and return the
/// SAM text written to `out`.
pub fn run_mem(reference: &Path, reads: &[&Path], out: &Path, extra: &[&str]) -> String {
    run_bin(&mem_args(reference, reads, out, extra));
    fs::read_to_string(out).expect("read SAM output")
}

// ── SAM parsing ──────────────────────────────────────────────────────────

/// One SAM alignment line, split into its mandatory columns plus raw tags.
#[derive(Clone, Debug)]
pub struct SamRecord {
    pub line: String,
    pub qname: String,
    pub flag: u16,
    pub rname: String,
    pub pos: u64,
    pub mapq: u8,
    pub cigar: String,
    pub rnext: String,
    pub pnext: u64,
    pub tlen: i64,
    pub seq: String,
    pub qual: String,
    pub tags: Vec<String>,
}

impl SamRecord {
    pub fn parse(line: &str) -> SamRecord {
        let cols: Vec<&str> = line.split('\t').collect();
        assert!(
            cols.len() >= 11,
            "SAM record has {} columns, expected >= 11: {line:?}",
            cols.len()
        );
        let num = |i: usize, what: &str| -> i64 {
            cols[i].parse::<i64>().unwrap_or_else(|_| {
                panic!("{what} column {:?} is not an integer in {line:?}", cols[i])
            })
        };
        let flag = num(1, "FLAG");
        assert!((0..=0xFFFF).contains(&flag), "FLAG out of range in {line:?}");
        let mapq = num(4, "MAPQ");
        assert!((0..=255).contains(&mapq), "MAPQ out of range in {line:?}");
        SamRecord {
            line: line.to_string(),
            qname: cols[0].to_string(),
            flag: flag as u16,
            rname: cols[2].to_string(),
            pos: num(3, "POS") as u64,
            mapq: mapq as u8,
            cigar: cols[5].to_string(),
            rnext: cols[6].to_string(),
            pnext: num(7, "PNEXT") as u64,
            tlen: num(8, "TLEN"),
            seq: cols[9].to_string(),
            qual: cols[10].to_string(),
            tags: cols[11..].iter().map(|s| s.to_string()).collect(),
        }
    }

    pub fn has(&self, bit: u16) -> bool {
        self.flag & bit != 0
    }

    pub fn is_unmapped(&self) -> bool {
        self.has(0x4)
    }

    pub fn is_primary(&self) -> bool {
        self.flag & 0x900 == 0
    }

    /// Value of tag `name` (e.g. `"NM"`), without the `NM:i:` prefix.
    pub fn tag(&self, name: &str) -> Option<&str> {
        self.tags.iter().find_map(|t| {
            let mut it = t.splitn(3, ':');
            let tag = it.next()?;
            let _ty = it.next()?;
            let val = it.next()?;
            (tag == name).then_some(val)
        })
    }

    /// The mate role a record's flags claim.
    pub fn role(&self) -> Role {
        if !self.has(0x1) {
            Role::Single
        } else if self.has(0x40) {
            Role::R1
        } else {
            Role::R2
        }
    }

    /// RNEXT with `=` resolved to RNAME.
    pub fn mate_rname(&self) -> &str {
        if self.rnext == "=" { &self.rname } else { &self.rnext }
    }
}

/// Header lines and body records of a SAM text.
pub struct Sam {
    pub header: Vec<String>,
    pub records: Vec<SamRecord>,
}

pub fn parse_sam(text: &str) -> Sam {
    let mut header = Vec::new();
    let mut records = Vec::new();
    for line in text.lines() {
        if line.is_empty() {
            continue;
        }
        if line.starts_with('@') {
            assert!(
                records.is_empty(),
                "header line after the first alignment record: {line:?}"
            );
            header.push(line.to_string());
        } else {
            records.push(SamRecord::parse(line));
        }
    }
    Sam { header, records }
}

/// Number of query bases the CIGAR consumes (M/I/S/=/X), or `None` when the
/// CIGAR is `*` or malformed.
pub fn cigar_query_len(cigar: &str) -> Option<usize> {
    if cigar == "*" {
        return None;
    }
    let mut total = 0usize;
    let mut n = 0usize;
    let mut saw_digit = false;
    for c in cigar.chars() {
        if let Some(d) = c.to_digit(10) {
            n = n * 10 + d as usize;
            saw_digit = true;
            continue;
        }
        if !saw_digit {
            return None;
        }
        match c {
            'M' | 'I' | 'S' | '=' | 'X' => total += n,
            'D' | 'N' | 'H' | 'P' => {}
            _ => return None,
        }
        n = 0;
        saw_digit = false;
    }
    (!saw_digit).then_some(total)
}

/// Body of a SAM text: every non-header line, in order.
pub fn sam_body(text: &str) -> Vec<&str> {
    text.lines()
        .filter(|l| !l.starts_with('@') && !l.is_empty())
        .collect()
}
