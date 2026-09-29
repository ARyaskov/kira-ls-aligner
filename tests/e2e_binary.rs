//! End-to-end test through the real `kira_ls_aligner` binary.
//!
//! Builds a two-contig synthetic reference and simulated paired-end /
//! single-end reads (see `tests/common/mod.rs`), runs `index` and `mem`
//! as a user would, and checks the SAM output against both the SAM
//! specification's invariants and the simulated truth.

mod common;

use std::collections::HashMap;
use std::ffi::OsString;
use std::io::Read;
use std::process::{Command, Stdio};

use common::*;

/// Placement tolerance around the truth start, in bp.
const POS_TOLERANCE: i64 = 5;
/// Minimum fraction of reads that must land on the truth locus.
const MIN_CONCORDANCE: f64 = 0.95;

#[derive(Default)]
struct Stats {
    reads: usize,
    concordant: usize,
}

/// `@HD` first, then `@SQ` in reference order with the right lengths.
fn check_header(header: &[String], ds: &Dataset, what: &str) {
    assert!(
        header.first().is_some_and(|l| l.starts_with("@HD\t")),
        "{what}: first header line is not @HD: {:?}",
        header.first()
    );
    let sq: Vec<(String, usize)> = header
        .iter()
        .filter(|l| l.starts_with("@SQ\t"))
        .map(|l| {
            let mut sn = None;
            let mut ln = None;
            for field in l.split('\t').skip(1) {
                if let Some(v) = field.strip_prefix("SN:") {
                    sn = Some(v.to_string());
                } else if let Some(v) = field.strip_prefix("LN:") {
                    ln = Some(v.parse::<usize>().expect("LN is an integer"));
                }
            }
            (sn.expect("@SQ has SN"), ln.expect("@SQ has LN"))
        })
        .collect();
    assert_eq!(sq, ds.contigs, "{what}: @SQ lines differ from the reference");
}

/// Every SAM line must also parse with a spec-conformant reader.
fn check_parses_with_noodles(text: &str, what: &str) {
    use noodles_sam::alignment::RecordBuf;
    let mut reader = noodles_sam::io::Reader::new(text.as_bytes());
    let header = reader
        .read_header()
        .unwrap_or_else(|e| panic!("{what}: noodles rejects the header: {e}"));
    let mut record = RecordBuf::default();
    let mut n = 0usize;
    loop {
        match reader.read_record_buf(&header, &mut record) {
            Ok(0) => break,
            Ok(_) => n += 1,
            Err(e) => panic!("{what}: noodles rejects record {}: {e}", n + 1),
        }
    }
    assert_eq!(n, sam_body(text).len(), "{what}: noodles record count");
}

/// Per-record invariants that do not need the mate.
fn check_record(rec: &SamRecord, ds: &Dataset, paired: bool, stats: &mut Stats) {
    let line = &rec.line;
    assert!(rec.mapq <= 60, "MAPQ {} outside 0..=60: {line}", rec.mapq);
    assert!(
        !rec.qname.ends_with("/1") && !rec.qname.ends_with("/2"),
        "QNAME still carries a mate suffix: {line}"
    );

    if paired {
        assert!(rec.has(0x1), "paired record without 0x1: {line}");
        assert!(
            rec.has(0x40) ^ rec.has(0x80),
            "paired record must have exactly one of 0x40/0x80: {line}"
        );
    } else {
        assert_eq!(rec.flag & (0x1 | 0x40 | 0x80), 0, "single-end record with pair bits: {line}");
    }

    let truth = ds
        .truth
        .get(&(rec.qname.clone(), rec.role()))
        .unwrap_or_else(|| panic!("QNAME not in the simulated set: {line}"));

    if rec.is_unmapped() {
        // Either fully unplaced or placed at the mate (checked pair-wise).
        assert!(
            (rec.rname == "*" && rec.pos == 0) || paired,
            "unmapped single-end record carries a position: {line}"
        );
        assert_eq!(rec.cigar, "*", "unmapped record with a CIGAR: {line}");
    } else {
        assert!(
            rec.rname != "*" && rec.pos > 0,
            "mapped record (0x4 clear) without RNAME/POS: {line}"
        );
        assert!(
            ds.contigs.iter().any(|(n, _)| *n == rec.rname),
            "RNAME {} is not a reference contig: {line}",
            rec.rname
        );
        let qlen = cigar_query_len(&rec.cigar)
            .unwrap_or_else(|| panic!("malformed CIGAR {:?}: {line}", rec.cigar));
        if rec.seq != "*" {
            assert_eq!(qlen, rec.seq.len(), "CIGAR query length != SEQ length: {line}");
            assert_eq!(rec.qual.len(), rec.seq.len(), "QUAL length != SEQ length: {line}");
        }
        assert!(rec.tag("NM").is_some(), "mapped record without NM: {line}");
        assert!(
            rec.tag("MD").is_some_and(|md| !md.is_empty()),
            "mapped record without a non-empty MD: {line}"
        );
    }

    // 0x10 must agree with the orientation of SEQ relative to the read as
    // sequenced (only checkable when SEQ is the full, unclipped read).
    if rec.seq != "*" && !rec.cigar.contains('H') {
        let expected = if rec.has(0x10) { revcomp(&truth.seq) } else { truth.seq.clone() };
        assert_eq!(
            rec.seq.as_bytes(),
            expected.as_slice(),
            "SEQ does not match the read in the orientation 0x10 claims: {line}"
        );
    }

    if !rec.is_primary() {
        return;
    }
    stats.reads += 1;

    if rec.is_unmapped() {
        return;
    }
    let delta = (rec.pos as i64 - 1) - truth.start as i64;
    let concordant = rec.rname == truth.contig && delta.abs() <= POS_TOLERANCE;
    if concordant {
        stats.concordant += 1;
    } else {
        eprintln!(
            "[e2e] discordant: {} placed at {}:{} (truth {}:{})",
            rec.qname, rec.rname, rec.pos, truth.contig, truth.start
        );
        return;
    }

    // NM must reflect exactly the substitutions we introduced (the reads
    // are otherwise error-free, and the substitutions sit well inside the
    // read, so a concordant full-length alignment has no other edits).
    let nm: usize = rec.tag("NM").unwrap().parse().expect("NM is an integer");
    let full_length = rec.cigar == format!("{}M", truth.seq.len());
    if full_length {
        assert_eq!(
            nm, truth.mismatches,
            "NM differs from the {} substitution(s) introduced: {line}",
            truth.mismatches
        );
    } else {
        // A clipped or gapped alignment of a concordant read: still no more
        // edits than we introduced.
        assert!(
            nm <= truth.mismatches.max(2),
            "NM {nm} exceeds the substitutions introduced: {line}"
        );
    }
}

/// Mate-level invariants across the two primary records of a template.
fn check_pairs(sam: &Sam, ds: &Dataset) {
    let mut by_name: HashMap<&str, (Option<&SamRecord>, Option<&SamRecord>)> = HashMap::new();
    for rec in sam.records.iter().filter(|r| r.is_primary()) {
        let slot = by_name.entry(&rec.qname).or_default();
        match rec.role() {
            Role::R1 => {
                assert!(slot.0.is_none(), "two primary R1 records for {}", rec.qname);
                slot.0 = Some(rec);
            }
            Role::R2 => {
                assert!(slot.1.is_none(), "two primary R2 records for {}", rec.qname);
                slot.1 = Some(rec);
            }
            Role::Single => unreachable!("checked per record"),
        }
    }
    assert_eq!(by_name.len(), ds.n_pairs, "one template per simulated pair");

    for (qname, (r1, r2)) in by_name {
        let r1 = r1.unwrap_or_else(|| panic!("{qname}: no primary R1 record"));
        let r2 = r2.unwrap_or_else(|| panic!("{qname}: no primary R2 record"));
        assert_eq!(r1.qname, r2.qname);

        let both_mapped = !r1.is_unmapped() && !r2.is_unmapped();
        if both_mapped {
            assert_eq!(r1.tlen, -r2.tlen, "TLEN of the mates must be negatives:\n{}\n{}", r1.line, r2.line);
            assert_eq!(r1.mate_rname(), r2.rname, "R1 RNEXT != R2 RNAME:\n{}\n{}", r1.line, r2.line);
            assert_eq!(r1.pnext, r2.pos, "R1 PNEXT != R2 POS:\n{}\n{}", r1.line, r2.line);
            assert_eq!(r2.mate_rname(), r1.rname, "R2 RNEXT != R1 RNAME:\n{}\n{}", r1.line, r2.line);
            assert_eq!(r2.pnext, r1.pos, "R2 PNEXT != R1 POS:\n{}\n{}", r1.line, r2.line);
            assert!(!r1.has(0x8) && !r2.has(0x8), "0x8 set although both mates mapped:\n{}\n{}", r1.line, r2.line);
            assert_eq!(r1.has(0x20), r2.has(0x10), "R1 0x20 != R2 0x10:\n{}\n{}", r1.line, r2.line);
            assert_eq!(r2.has(0x20), r1.has(0x10), "R2 0x20 != R1 0x10:\n{}\n{}", r1.line, r2.line);
            if r1.rname == r2.rname {
                assert_ne!(r1.tlen, 0, "TLEN 0 for mates on one contig:\n{}\n{}", r1.line, r2.line);
                // Concordant, full-length pair: |TLEN| is the fragment length.
                let t1 = &ds.truth[&(r1.qname.clone(), Role::R1)];
                let t2 = &ds.truth[&(r2.qname.clone(), Role::R2)];
                let frag = (t2.end - t1.start) as i64;
                let r1_ok = r1.rname == t1.contig && (r1.pos as i64 - 1 - t1.start as i64).abs() <= POS_TOLERANCE;
                let r2_ok = r2.rname == t2.contig && (r2.pos as i64 - 1 - t2.start as i64).abs() <= POS_TOLERANCE;
                if r1_ok && r2_ok && r1.cigar.ends_with('M') && !r1.cigar.contains('S') && !r2.cigar.contains('S') {
                    assert!(
                        (r1.tlen.abs() - frag).abs() <= 2 * POS_TOLERANCE,
                        "|TLEN| {} far from fragment length {frag}:\n{}\n{}",
                        r1.tlen.abs(),
                        r1.line,
                        r2.line
                    );
                    assert!(r1.has(0x2) && r2.has(0x2), "concordant pair without 0x2:\n{}\n{}", r1.line, r2.line);
                }
            }
        } else {
            for (me, mate) in [(r1, r2), (r2, r1)] {
                if me.is_unmapped() {
                    assert!(mate.has(0x8) || mate.is_unmapped(), "mate of an unmapped read lacks 0x8:\n{}\n{}", me.line, mate.line);
                    assert_eq!(me.tlen, 0, "unmapped read with a TLEN:\n{}", me.line);
                    if me.rname != "*" {
                        // Placed at the mate: must copy the mate's coordinates.
                        assert!(!mate.is_unmapped(), "unmapped read placed at an unmapped mate:\n{}\n{}", me.line, mate.line);
                        assert_eq!(me.rname, mate.rname, "unmapped read placed away from its mate:\n{}\n{}", me.line, mate.line);
                        assert_eq!(me.pos, mate.pos, "unmapped read placed away from its mate:\n{}\n{}", me.line, mate.line);
                    }
                }
            }
        }
    }
}

#[test]
fn index_and_mem_through_the_binary() {
    let ds = build_dataset("e2e", &SimParams::default());
    let index = ds.dir.join("ref.kiraidx");

    let args: Vec<OsString> = vec![
        "index".into(),
        ds.reference.clone().into(),
        "-o".into(),
        index.clone().into(),
    ];
    run_bin(&args);
    assert!(index.is_file(), "index file was not written");

    // Paired: prebuilt index, two threads.
    let paired_text = run_mem(
        &ds.reference,
        &[&ds.r1, &ds.r2],
        &ds.dir.join("paired.sam"),
        &["--index", index.to_str().unwrap(), "-t", "2"],
    );
    // Single-end: no --index, indexing on the fly.
    let single_text = run_mem(&ds.reference, &[&ds.single], &ds.dir.join("single.sam"), &[]);

    let paired = parse_sam(&paired_text);
    let single = parse_sam(&single_text);

    check_header(&paired.header, &ds, "paired");
    check_header(&single.header, &ds, "single");
    check_parses_with_noodles(&paired_text, "paired");
    check_parses_with_noodles(&single_text, "single");

    let mut stats = Stats::default();
    for rec in &paired.records {
        check_record(rec, &ds, true, &mut stats);
    }
    for rec in &single.records {
        check_record(rec, &ds, false, &mut stats);
    }
    check_pairs(&paired, &ds);

    assert_eq!(
        stats.reads,
        2 * ds.n_pairs + ds.n_single,
        "every read must have exactly one primary record"
    );
    let concordance = stats.concordant as f64 / stats.reads as f64;
    eprintln!(
        "[e2e] concordance: {}/{} reads within {POS_TOLERANCE} bp of truth ({:.2}%)",
        stats.concordant,
        stats.reads,
        100.0 * concordance
    );
    assert!(
        concordance >= MIN_CONCORDANCE,
        "only {:.2}% of reads placed within {POS_TOLERANCE} bp of their truth locus",
        100.0 * concordance
    );

    // `mem ... | head -c 100`: a closed downstream pipe must end the process
    // quietly, as bwa and samtools do — no "Broken pipe" error on stderr.
    let mut child = Command::new(BIN)
        .arg("mem")
        .arg(&ds.reference)
        .arg(&ds.r1)
        .arg(&ds.r2)
        .args(["-t", "2"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn kira_ls_aligner");
    let mut stdout = child.stdout.take().unwrap();
    let mut head = [0u8; 100];
    stdout.read_exact(&mut head).expect("read the first 100 bytes of SAM");
    drop(stdout);
    let out = child.wait_with_output().expect("wait for kira_ls_aligner");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(head.starts_with(b"@HD\t"), "stdout does not start with @HD");
    assert!(
        !stderr.contains("Broken pipe"),
        "binary complained about the closed pipe:\n{stderr}"
    );
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        assert!(
            out.status.success() || out.status.signal() == Some(13),
            "unexpected exit status after closing the pipe: {} \n{stderr}",
            out.status
        );
    }
}
