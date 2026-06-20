//! End-to-end ingest tests for paired FASTQ modes.
//!
//! These exercise the `ReadStream` two-file and interleaved paths through
//! its public API: we write small FASTQ files to a tempdir, open them via
//! `ReadStream::new_multi_with_mode`, drain `next_batch()`, and assert the
//! resulting `ReadRecord`s carry the right `PairRole` in the right order.

use std::io::Write;

use kira_ls_aligner::io::{IngestMode, ReadStream};
use kira_ls_aligner::types::PairRole;

const R1_FASTQ: &[u8] = b"\
@read1/1
ACGTACGTACGT
+
IIIIIIIIIIII
@read2/1
TTTTTTTTTTTT
+
IIIIIIIIIIII
";

const R2_FASTQ: &[u8] = b"\
@read1/2
GGGGGGGGGGGG
+
IIIIIIIIIIII
@read2/2
AAAAAAAAAAAA
+
IIIIIIIIIIII
";

const INTERLEAVED_FASTQ: &[u8] = b"\
@pair1/1
ACGTACGTACGT
+
IIIIIIIIIIII
@pair1/2
GGGGGGGGGGGG
+
IIIIIIIIIIII
@pair2/1
TTTTTTTTTTTT
+
IIIIIIIIIIII
@pair2/2
AAAAAAAAAAAA
+
IIIIIIIIIIII
";

const MISMATCHED_R2_FASTQ: &[u8] = b"\
@foreign_read/2
GGGGGGGGGGGG
+
IIIIIIIIIIII
";

fn write_tmp(name: &str, content: &[u8]) -> std::path::PathBuf {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("kira-paired-{}-{}.fastq", std::process::id(), name));
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(content).unwrap();
    path
}

#[test]
fn two_file_pairing_yields_r1_r2_adjacent() {
    let p1 = write_tmp("twofile-r1", R1_FASTQ);
    let p2 = write_tmp("twofile-r2", R2_FASTQ);
    let paths = vec![p1.clone(), p2.clone()];
    let mut stream = ReadStream::new_multi_with_mode(&paths, 100_000, IngestMode::TwoFile).unwrap();
    let batch = stream.next_batch().unwrap().unwrap();
    assert_eq!(batch.len(), 4, "2 pairs × 2 reads each");
    assert_eq!(batch[0].pair_role, PairRole::R1);
    assert_eq!(batch[1].pair_role, PairRole::R2);
    assert_eq!(batch[2].pair_role, PairRole::R1);
    assert_eq!(batch[3].pair_role, PairRole::R2);
    // ID canonicalisation must equate `read1/1` and `read1/2`.
    assert_eq!(batch[0].id, "read1/1");
    assert_eq!(batch[1].id, "read1/2");

    // Stream is now drained.
    assert!(stream.next_batch().unwrap().is_none());

    let _ = std::fs::remove_file(p1);
    let _ = std::fs::remove_file(p2);
}

#[test]
fn interleaved_pairing_yields_r1_r2_adjacent() {
    let p = write_tmp("interleaved", INTERLEAVED_FASTQ);
    let paths = vec![p.clone()];
    let mut stream =
        ReadStream::new_multi_with_mode(&paths, 100_000, IngestMode::Interleaved).unwrap();
    let batch = stream.next_batch().unwrap().unwrap();
    assert_eq!(batch.len(), 4);
    assert_eq!(batch[0].pair_role, PairRole::R1);
    assert_eq!(batch[0].id, "pair1/1");
    assert_eq!(batch[1].pair_role, PairRole::R2);
    assert_eq!(batch[1].id, "pair1/2");
    assert_eq!(batch[2].id, "pair2/1");
    assert_eq!(batch[3].id, "pair2/2");

    let _ = std::fs::remove_file(p);
}

#[test]
fn mismatched_pair_ids_fail_loudly() {
    let p1 = write_tmp("mismatch-r1", R1_FASTQ);
    let p2 = write_tmp("mismatch-r2", MISMATCHED_R2_FASTQ);
    let paths = vec![p1.clone(), p2.clone()];
    let mut stream = ReadStream::new_multi_with_mode(&paths, 100_000, IngestMode::TwoFile).unwrap();
    let result = stream.next_batch();
    assert!(result.is_err(), "expected paired-id mismatch to surface");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("paired read IDs disagree"),
        "wrong error message: {msg}"
    );

    let _ = std::fs::remove_file(p1);
    let _ = std::fs::remove_file(p2);
}

#[test]
fn two_file_mode_requires_two_paths() {
    let p1 = write_tmp("single-mode", R1_FASTQ);
    let paths = vec![p1.clone()];
    let result = ReadStream::new_multi_with_mode(&paths, 100_000, IngestMode::TwoFile);
    assert!(result.is_err());

    let _ = std::fs::remove_file(p1);
}

#[test]
fn unpaired_mode_concatenates_files() {
    let p1 = write_tmp("unp-r1", R1_FASTQ);
    let p2 = write_tmp("unp-r2", R2_FASTQ);
    let paths = vec![p1.clone(), p2.clone()];
    let mut stream =
        ReadStream::new_multi_with_mode(&paths, 100_000, IngestMode::Unpaired).unwrap();
    let batch = stream.next_batch().unwrap().unwrap();
    // Legacy behaviour: 4 records, all Unpaired
    assert_eq!(batch.len(), 4);
    assert!(batch.iter().all(|r| r.pair_role == PairRole::Unpaired));

    let _ = std::fs::remove_file(p1);
    let _ = std::fs::remove_file(p2);
}
