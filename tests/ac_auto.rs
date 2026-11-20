use std::path::Path;
use std::process::Command;

const READ_LEN: usize = 150;

fn run_mem(
    dir: &Path,
    exact_reads: usize,
    add_unmappable: bool,
    ac_override: Option<&str>,
) -> (String, String) {
    let fasta = dir.join("ref.fa");
    let fastq = dir.join("reads.fq");
    let mut state = 9u64;
    let reference: Vec<_> = (0..4000)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            b"ACGT"[(state >> 62) as usize]
        })
        .collect();
    let reference = String::from_utf8(reference).unwrap();
    std::fs::write(&fasta, format!(">ref\n{reference}\n")).unwrap();

    let quality = "I".repeat(READ_LEN);
    let mut reads = String::new();
    use std::fmt::Write as _;
    for id in 0..exact_reads {
        let start = id % 128 * 19;
        let seq = &reference[start..start + READ_LEN];
        writeln!(reads, "@r{id}\n{seq}\n+\n{quality}").unwrap();
    }
    if add_unmappable {
        writeln!(reads, "@unmapped\n{}\n+\n{quality}", "N".repeat(READ_LEN)).unwrap();
    }
    std::fs::write(&fastq, reads).unwrap();

    let mut command = Command::new(env!("CARGO_BIN_EXE_kira_ls_aligner"));
    command
        .args(["mem", "-x", "short", "-t", "2", "-K", "10000000"])
        .args([fasta, fastq])
        .env_remove("KIRA_AC_DISABLE")
        .env("KIRA_STATS", "1");
    if let Some(value) = ac_override {
        command.env("KIRA_AC_DISABLE", value);
    }
    let output = command.output().unwrap();
    let log = String::from_utf8(output.stderr).unwrap();
    assert!(output.status.success(), "{log}");
    (String::from_utf8(output.stdout).unwrap(), log)
}

fn check_sam(sam: &str, exact_reads: usize, add_unmappable: bool) {
    let mut records = sam.lines().filter(|line| !line.starts_with('@'));
    for id in 0..exact_reads {
        let fields: Vec<_> = records.next().unwrap().split('\t').collect();
        assert_eq!(fields[0], format!("r{id}"));
        assert_eq!(fields[1].parse::<u16>().unwrap() & (0x4 | 0x10), 0);
        assert_eq!(fields[2], "ref");
        assert_eq!(fields[3].parse::<usize>().unwrap(), id % 128 * 19 + 1);
        assert_eq!(fields[5], "150M");
        assert!(fields.contains(&"NM:i:0"));
    }
    if add_unmappable {
        let fields: Vec<_> = records.next().unwrap().split('\t').collect();
        assert_eq!(fields[0], "unmapped");
        assert_ne!(fields[1].parse::<u16>().unwrap() & 4, 0);
    }
    assert!(records.next().is_none());
}

fn algo_summary(log: &str) -> &str {
    log.lines()
        .find(|line| line.contains("[KIRA_ALGO] summary:"))
        .unwrap()
}

#[test]
fn ac_matches_count_as_exact_and_not_unmapped() {
    let dir = tempfile::tempdir().unwrap();
    let (sam, log) = run_mem(dir.path(), 128, true, Some("0"));
    check_sam(&sam, 128, true);
    assert!(log.contains("eligible=128 resolved=128"), "{log}");
    let summary = algo_summary(&log);
    assert!(summary.contains("exact=128 "), "{summary}");
    assert!(summary.contains("unmapped=1 "), "{summary}");
}

#[test]
fn mem_defaults_to_indexed_seeding_on_an_ac_eligible_batch() {
    let dir = tempfile::tempdir().unwrap();
    let (sam, log) = run_mem(dir.path(), 128, false, None);
    let (off_sam, off_log) = run_mem(dir.path(), 128, false, Some("1"));
    check_sam(&sam, 128, false);
    assert!(log.contains("[KIRA_AC] disabled by default"), "{log}");
    for log in [log, off_log] {
        assert!(log.contains("eligible=0 resolved=0"), "{log}");
        assert!(algo_summary(&log).contains("unmapped=0 "));
    }
    assert!(
        sam.lines()
            .filter(|line| !line.starts_with('@'))
            .eq(off_sam.lines().filter(|line| !line.starts_with('@')))
    );
}
