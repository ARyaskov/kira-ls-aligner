//! Placement-accuracy evaluation for simulated reads with a truth-in-name
//! source locus.
//!
//! The regression harness generates reads whose FASTQ id encodes the source
//! interval, so per-read placement correctness is checkable in seconds rather
//! than by a variant-calling round-trip. This module parses those ids, scores
//! a SAM file against them, and attributes every read to exactly one bucket:
//!
//! - `unmapped` — the read got no alignment (recall loss at the source),
//! - `correct` — primary placement on the truth contig within tolerance,
//! - `wrong_locus` — primary placement elsewhere (precision loss / paralog),
//! - `no_truth` — QNAME carried no parseable locus (skipped from accuracy).
//!
//! Counts are additionally stratified by MAPQ threshold, which is what a
//! caller's `MAPQ >= t` filter does: `correct` reads below the threshold are
//! recall the caller cannot use, `wrong_locus` reads above it are false
//! evidence the caller cannot remove. INDEL-bearing reads (any I/D in the
//! CIGAR) are counted separately to track the gapped paths.
//!
//! Accepted truth encodings in QNAME (after stripping a trailing `/1` or `/2`):
//! `<name>:<contig>:<start>-<end>` and `<name>_<contig>_<start>_<end>`,
//! with 0-based half-open coordinates.

use std::io::{BufRead, Write};

/// Evaluation configuration.
#[derive(Clone, Debug)]
pub struct EvalConfig {
    /// Positional tolerance (bp) for locus concordance: the primary alignment
    /// start must land within `truth.start - tolerance ..= truth.end + tolerance`.
    pub tolerance: i64,
    /// MAPQ thresholds for stratified counts (caller-filter simulation).
    pub mapq_thresholds: Vec<u8>,
}

/// A parsed truth locus from a read name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TruthLocus {
    pub contig: String,
    pub start: i64,
    pub end: i64,
}

/// Parse the truth locus out of a SAM QNAME. See the module docs for the
/// accepted encodings. Returns `None` for names without a parseable locus.
pub fn parse_truth(qname: &str) -> Option<TruthLocus> {
    let core = qname
        .strip_suffix("/1")
        .or_else(|| qname.strip_suffix("/2"))
        .unwrap_or(qname);

    // Colon form: <name>:<contig>:<start>-<end>
    if let Some((head, range)) = core.rsplit_once(':') {
        if let Some((_, contig)) = head.rsplit_once(':') {
            if let Some((s, e)) = range.rsplit_once('-') {
                if let (Ok(start), Ok(end)) = (s.parse::<i64>(), e.parse::<i64>()) {
                    return Some(TruthLocus {
                        contig: contig.to_string(),
                        start,
                        end,
                    });
                }
            }
        }
    }

    // Underscore form: <name>_<contig>_<start>_<end>
    let parts: Vec<&str> = core.rsplitn(4, '_').collect();
    if parts.len() == 4 {
        if let (Ok(end), Ok(start)) = (parts[0].parse::<i64>(), parts[1].parse::<i64>()) {
            return Some(TruthLocus {
                contig: parts[2].to_string(),
                start,
                end,
            });
        }
    }
    None
}

/// Per-MAPQ-threshold stratified counts.
#[derive(Clone, Copy, Debug, Default)]
pub struct ThresholdCounts {
    pub correct_ge: u64,
    pub wrong_ge: u64,
}

/// Evaluation counters. All ratios are derived at report time.
#[derive(Clone, Debug, Default)]
pub struct EvalCounts {
    /// Total SAM records seen (headers excluded).
    pub total: u64,
    /// Records with a parseable truth locus.
    pub truth_parsed: u64,
    /// Truth-carrying records with no alignment (flag 0x4).
    pub unmapped: u64,
    /// Primary placement concordant with the truth locus.
    pub correct: u64,
    /// Primary placement discordant with the truth locus.
    pub wrong_locus: u64,
    /// Correct placements whose CIGAR carries an insertion or deletion.
    pub indel_correct: u64,
    /// Wrong-locus placements whose CIGAR carries an insertion or deletion.
    pub indel_wrong: u64,
    /// `(threshold, counts)` in the order given by [`EvalConfig`].
    pub thresholds: Vec<(u8, ThresholdCounts)>,
}

impl EvalCounts {
    /// Truth-carrying reads that got a primary alignment.
    pub fn mapped(&self) -> u64 {
        self.truth_parsed - self.unmapped
    }

    /// Placement accuracy over mapped, truth-carrying reads.
    pub fn placement_accuracy(&self) -> f64 {
        let mapped = self.mapped();
        if mapped == 0 {
            return 0.0;
        }
        self.correct as f64 / mapped as f64
    }
}

/// Per-read attribution category, written to the dump file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Category {
    NoTruth,
    Unmapped,
    Correct,
    WrongLocus,
}

impl Category {
    fn as_str(self) -> &'static str {
        match self {
            Category::NoTruth => "no_truth",
            Category::Unmapped => "unmapped",
            Category::Correct => "correct",
            Category::WrongLocus => "wrong_locus",
        }
    }
}

/// Contig-name equality tolerant of a `chr` prefix on either side.
fn contig_eq(a: &str, b: &str) -> bool {
    a == b || a.strip_prefix("chr") == Some(b) || b.strip_prefix("chr") == Some(a)
}

/// Score one SAM file. `dump`, when given, receives one TSV row per record:
/// `qname  category  mapq  rname  pos  truth_contig  truth_start  truth_end  indel`.
pub fn evaluate<W: Write + ?Sized>(
    mut reader: impl BufRead,
    cfg: &EvalConfig,
    mut dump: Option<&mut W>,
) -> std::io::Result<EvalCounts> {
    let mut counts = EvalCounts {
        thresholds: cfg
            .mapq_thresholds
            .iter()
            .map(|&t| (t, ThresholdCounts::default()))
            .collect(),
        ..EvalCounts::default()
    };
    if let Some(w) = dump.as_mut() {
        writeln!(
            w,
            "qname\tcategory\tmapq\trname\tpos\ttruth_contig\ttruth_start\ttruth_end\tindel"
        )?;
    }

    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let text = line.trim_end();
        if text.is_empty() || text.starts_with('@') {
            continue;
        }
        counts.total += 1;

        let mut fields = text.split('\t');
        let qname = fields.next().unwrap_or("*");
        let flag: u16 = fields.next().and_then(|f| f.parse().ok()).unwrap_or(0x4);
        let rname = fields.next().unwrap_or("*");
        let pos: i64 = fields.next().and_then(|f| f.parse().ok()).unwrap_or(0);
        let mapq: u8 = fields.next().and_then(|f| f.parse().ok()).unwrap_or(0);
        let cigar = fields.next().unwrap_or("*");

        let truth = parse_truth(qname);
        let mapped = flag & 0x4 == 0;
        let has_indel = cigar.bytes().any(|b| b == b'I' || b == b'D');

        let (category, truth_ref) = match &truth {
            None => (Category::NoTruth, None),
            Some(t) => {
                counts.truth_parsed += 1;
                if !mapped {
                    counts.unmapped += 1;
                    (Category::Unmapped, Some(t))
                } else {
                    let pos0 = pos - 1;
                    let concordant = contig_eq(rname, &t.contig)
                        && pos0 >= t.start - cfg.tolerance
                        && pos0 <= t.end + cfg.tolerance;
                    if concordant {
                        counts.correct += 1;
                        if has_indel {
                            counts.indel_correct += 1;
                        }
                        for (t_thresh, c) in counts.thresholds.iter_mut() {
                            if mapq >= *t_thresh {
                                c.correct_ge += 1;
                            }
                        }
                        (Category::Correct, Some(t))
                    } else {
                        counts.wrong_locus += 1;
                        if has_indel {
                            counts.indel_wrong += 1;
                        }
                        for (t_thresh, c) in counts.thresholds.iter_mut() {
                            if mapq >= *t_thresh {
                                c.wrong_ge += 1;
                            }
                        }
                        (Category::WrongLocus, Some(t))
                    }
                }
            }
        };

        if let Some(w) = dump.as_mut() {
            let (tc, ts, te) = match truth_ref {
                Some(t) => (t.contig.as_str(), t.start, t.end),
                None => ("*", 0, 0),
            };
            writeln!(
                w,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                qname,
                category.as_str(),
                mapq,
                rname,
                pos,
                tc,
                ts,
                te,
                u8::from(has_indel),
            )?;
        }
    }
    Ok(counts)
}

/// Render the human- and grep-readable summary report.
pub fn render_report(counts: &EvalCounts) -> String {
    let mut out = String::new();
    let no_truth = counts.total - counts.truth_parsed;
    let mapped = counts.mapped();
    let pct = |n: u64, d: u64| {
        if d == 0 {
            0.0
        } else {
            n as f64 * 100.0 / d as f64
        }
    };
    out.push_str(&format!(
        "[EVAL] total_records={} truth_parsed={} no_truth={}\n",
        counts.total, counts.truth_parsed, no_truth
    ));
    out.push_str(&format!(
        "[EVAL] unmapped={} ({:.2}% of truth-parsed)\n",
        counts.unmapped,
        pct(counts.unmapped, counts.truth_parsed)
    ));
    out.push_str(&format!(
        "[EVAL] mapped={} correct={} ({:.2}%) wrong_locus={} ({:.2}%)\n",
        mapped,
        counts.correct,
        pct(counts.correct, mapped),
        counts.wrong_locus,
        pct(counts.wrong_locus, mapped),
    ));
    out.push_str(&format!(
        "[EVAL] indel_bearing: correct={} wrong_locus={}\n",
        counts.indel_correct, counts.indel_wrong
    ));
    for (t, c) in &counts.thresholds {
        let usable = c.correct_ge + c.wrong_ge;
        out.push_str(&format!(
            "[EVAL] mapq>={}: correct={} wrong={} precision={:.2}% recall_of_correct={:.2}% (n={})\n",
            t,
            c.correct_ge,
            c.wrong_ge,
            pct(c.correct_ge, usable),
            pct(c.correct_ge, counts.correct.max(1)),
            usable,
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufReader;

    #[test]
    fn parses_colon_form() {
        let t = parse_truth("sim42:chr20:1000-1150").unwrap();
        assert_eq!(t.contig, "chr20");
        assert_eq!((t.start, t.end), (1000, 1150));
    }

    #[test]
    fn parses_colon_form_with_mate_suffix() {
        let t = parse_truth("sim42:chr20:1000-1150/2").unwrap();
        assert_eq!(t.contig, "chr20");
        assert_eq!((t.start, t.end), (1000, 1150));
    }

    #[test]
    fn parses_underscore_form() {
        let t = parse_truth("read1_chr1_500_650").unwrap();
        assert_eq!(t.contig, "chr1");
        assert_eq!((t.start, t.end), (500, 650));
    }

    #[test]
    fn parses_underscore_form_with_mate_suffix() {
        let t = parse_truth("read1_chr1_500_650/1").unwrap();
        assert_eq!(t.contig, "chr1");
    }

    #[test]
    fn rejects_plain_names() {
        assert!(parse_truth("SRR1234567").is_none());
        assert!(parse_truth("read_1").is_none());
        assert!(parse_truth("sim:0001").is_none());
        assert!(parse_truth("").is_none());
    }

    fn cfg() -> EvalConfig {
        EvalConfig {
            tolerance: 150,
            mapq_thresholds: vec![13, 60],
        }
    }

    fn eval_sam(sam: &str) -> EvalCounts {
        evaluate(BufReader::new(sam.as_bytes()), &cfg(), None::<&mut Vec<u8>>).unwrap()
    }

    #[test]
    fn counts_categories_and_thresholds() {
        let sam = "\
@HD\tVN:1.6\tSO:unsorted
@SQ\tSN:chr1\tLN:1000000
sim:chr1:1000-1150\t0\tchr1\t1001\t60\t150M\t*\t0\t0\t*\t*
sim:chr1:2000-2150\t0\tchr1\t5001\t60\t150M\t*\t0\t0\t*\t*
sim:chr1:3000-3150\t4\t*\t0\t0\t*\t*\t0\t0\t*\t*
sim:chr1:4000-4150\t0\tchr1\t4051\t10\t140M1I9M\t*\t0\t0\t*\t*
SRR000001\t0\tchr1\t9999\t60\t150M\t*\t0\t0\t*\t*
";
        let c = eval_sam(sam);
        assert_eq!(c.total, 5);
        assert_eq!(c.truth_parsed, 4);
        assert_eq!(c.unmapped, 1);
        assert_eq!(c.correct, 2);
        assert_eq!(c.wrong_locus, 1);
        assert_eq!(c.indel_correct, 1);
        assert_eq!(c.indel_wrong, 0);
        // MAPQ>=13: the two correct (60, 10→no) → 1 correct; wrong is MQ60 → 1.
        let t13 = c.thresholds[0].1;
        assert_eq!((t13.correct_ge, t13.wrong_ge), (1, 1));
        // MAPQ>=60: one correct (the MQ10 indel read drops), one wrong.
        let t60 = c.thresholds[1].1;
        assert_eq!((t60.correct_ge, t60.wrong_ge), (1, 1));
    }

    #[test]
    fn tolerance_window_uses_truth_interval() {
        // Truth 1000-1150, tolerance 150 → accepted start range [850, 1300].
        let sam = "\
sim:chr1:1000-1150\t0\tchr1\t851\t60\t150M\t*\t0\t0\t*\t*
sim:chr1:1000-1150\t0\tchr1\t1301\t60\t150M\t*\t0\t0\t*\t*
sim:chr1:1000-1150\t0\tchr1\t1302\t60\t150M\t*\t0\t0\t*\t*
";
        let c = eval_sam(sam);
        assert_eq!(c.correct, 2);
        assert_eq!(c.wrong_locus, 1);
    }

    #[test]
    fn chr_prefix_is_tolerated() {
        let sam = "sim:20:1000-1150\t0\tchr20\t1001\t60\t150M\t*\t0\t0\t*\t*\n";
        let c = eval_sam(sam);
        assert_eq!(c.correct, 1);
    }

    #[test]
    fn report_renders_all_sections() {
        let sam = "\
sim:chr1:1000-1150\t0\tchr1\t1001\t60\t150M\t*\t0\t0\t*\t*
sim:chr1:2000-2150\t4\t*\t0\t0\t*\t*\t0\t0\t*\t*
";
        let c = eval_sam(sam);
        let report = render_report(&c);
        assert!(report.contains("total_records=2"));
        assert!(report.contains("unmapped=1"));
        assert!(report.contains("correct=1"));
        assert!(report.contains("mapq>=13"));
        assert!(report.contains("indel_bearing"));
    }

    #[test]
    fn dump_writes_one_row_per_record() {
        let sam = "\
sim:chr1:1000-1150\t0\tchr1\t1001\t60\t150M\t*\t0\t0\t*\t*
plain_name\t4\t*\t0\t0\t*\t*\t0\t0\t*\t*
";
        let mut buf: Vec<u8> = Vec::new();
        evaluate(BufReader::new(sam.as_bytes()), &cfg(), Some(&mut buf)).unwrap();
        let text = String::from_utf8(buf).unwrap();
        let rows: Vec<&str> = text.lines().collect();
        assert_eq!(rows.len(), 3); // header + 2 records
        assert!(rows[1].starts_with("sim:chr1:1000-1150\tcorrect\t60\tchr1\t1001"));
        assert!(rows[2].starts_with("plain_name\tno_truth\t0"));
    }
}
