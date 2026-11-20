use std::path::Path;
use std::process::Command;

const READ_LEN: usize = 150;

fn run_mem(
    dir: &Path,
    exact_reads: usize,
    add_unmappable: bool,
    ac_override: Option<&str>,
    near_paralog: bool,
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
    let mut fasta_text = format!(">ref\n{reference}\n");
    if near_paralog {
        let mut paralog = reference.as_bytes()[..READ_LEN].to_vec();
        for pos in [56, 91] {
            paralog[pos] = if paralog[pos] == b'A' { b'C' } else { b'A' };
        }
        fasta_text.push_str(&format!(
            ">paralog\n{}{}{}\n",
            &reference[2700..2800],
            String::from_utf8(paralog).unwrap(),
            &reference[3100..3200]
        ));
    }
    std::fs::write(&fasta, fasta_text).unwrap();

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
fn ac_skips_build_when_unplaced_reads_are_ineligible() {
    let dir = tempfile::tempdir().unwrap();
    let (sam, log) = run_mem(dir.path(), 128, true, Some("0"), false);
    check_sam(&sam, 128, true);
    assert!(log.contains("eligible=0 resolved=0"), "{log}");
    assert!(log.contains("build=0.00ms scan=0.00ms rescued=0"), "{log}");
    let summary = algo_summary(&log);
    assert!(summary.contains("exact=128 "), "{summary}");
    assert!(summary.contains("unmapped=1 "), "{summary}");
}

#[test]
fn mem_defaults_to_indexed_seeding_on_an_ac_eligible_batch() {
    let dir = tempfile::tempdir().unwrap();
    let (sam, log) = run_mem(dir.path(), 128, false, None, false);
    let (off_sam, off_log) = run_mem(dir.path(), 128, false, Some("1"), false);
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

#[test]
fn ac_keeps_indexed_competitor_evidence_for_an_exact_hit() {
    let dir = tempfile::tempdir().unwrap();
    let (indexed, _) = run_mem(dir.path(), 128, false, Some("1"), true);
    let (ac, log) = run_mem(dir.path(), 128, false, Some("0"), true);
    assert!(log.contains("eligible=0 resolved=0"), "{log}");
    fn first_record(sam: &str) -> Vec<&str> {
        sam.lines()
            .find(|line| line.starts_with("r0\t"))
            .unwrap()
            .split('\t')
            .collect()
    }
    let baseline = first_record(&indexed);
    assert_eq!(&baseline[2..4], &["ref", "1"]);
    assert!(baseline[4].parse::<u8>().unwrap() < 60, "{baseline:?}");
    assert!(
        baseline.iter().any(|tag| tag.starts_with("XS:i:")),
        "{baseline:?}"
    );
    assert_eq!(first_record(&ac), baseline);
}

#[test]
fn ac_only_placements_use_the_rescue_mapq_cap() {
    let dir = tempfile::tempdir().unwrap();
    let reference = dir.path().join("ref.fa");
    let reads = dir.path().join("reads.fq");
    let sequence = "AGAGTACAACGATCGTGACATGTCCGATTA";
    std::fs::write(&reference, format!(">ref\n{sequence}\n")).unwrap();
    let mut input = String::new();
    use std::fmt::Write as _;
    for i in 0..64 {
        writeln!(
            input,
            "@r{i}\n{sequence}\n+\n{}",
            "I".repeat(sequence.len())
        )
        .unwrap();
    }
    std::fs::write(&reads, input).unwrap();
    let run = |disable| {
        let output = Command::new(env!("CARGO_BIN_EXE_kira_ls_aligner"))
            .args(["mem", "-x", "short", "-k", "31", "-T", "0", "-t", "1"])
            .args([&reference, &reads])
            .env("KIRA_AC_DISABLE", disable)
            .env("KIRA_RESCUE_MAPQ_CAP", "17")
            .env("KIRA_STATS", "1")
            .output()
            .unwrap();
        let log = String::from_utf8(output.stderr).unwrap();
        assert!(output.status.success(), "{log}");
        (String::from_utf8(output.stdout).unwrap(), log)
    };
    let (indexed, _) = run("1");
    for line in indexed.lines().filter(|line| !line.starts_with('@')) {
        let fields: Vec<_> = line.split('\t').collect();
        assert_ne!(fields[1].parse::<u16>().unwrap() & 4, 0);
    }
    let (ac, log) = run("0");
    let mut n = 0;
    for line in ac.lines().filter(|line| !line.starts_with('@')) {
        let fields: Vec<_> = line.split('\t').collect();
        assert_eq!(fields[1].parse::<u16>().unwrap() & 4, 0);
        assert_eq!(&fields[2..6], &["ref", "1", "17", "30M"]);
        n += 1;
    }
    assert_eq!(n, 64);
    assert!(log.contains("rescued=64"), "{log}");
}

#[test]
fn ac_builds_patterns_for_unplaced_reads_only() {
    let dir = tempfile::tempdir().unwrap();
    let reference = dir.path().join("ref.fa");
    let reads = dir.path().join("reads.fq");

    let short_read = "AGAGTACAACGATCGTGACATGTCCGATTA";
    let mut state = 23u64;
    let full_reference: String = (0..400)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            char::from(b"ACGT"[(state >> 62) as usize])
        })
        .collect();
    let indexed = &full_reference[150..300];
    std::fs::write(
        &reference,
        format!(">ref\n{full_reference}\n>short\n{short_read}\n"),
    )
    .unwrap();

    let mut input = String::new();
    use std::fmt::Write as _;
    for i in 0..65 {
        let seq = if i == 11 { short_read } else { indexed };
        writeln!(input, "@r{i}\n{seq}\n+\n{}", "I".repeat(seq.len())).unwrap();
    }
    std::fs::write(&reads, input).unwrap();

    let run = |disable| {
        let output = Command::new(env!("CARGO_BIN_EXE_kira_ls_aligner"))
            .args(["mem", "-x", "short", "-k", "31", "-T", "0", "-t", "1"])
            .args([&reference, &reads])
            .env("KIRA_AC_DISABLE", disable)
            .env("KIRA_RESCUE_MAPQ_CAP", "17")
            .env("KIRA_STATS", "1")
            .output()
            .unwrap();
        let log = String::from_utf8(output.stderr).unwrap();
        assert!(output.status.success(), "{log}");
        let sam = String::from_utf8(output.stdout).unwrap();
        let rows: Vec<_> = sam
            .lines()
            .filter(|line| !line.starts_with('@'))
            .map(str::to_owned)
            .collect();
        (rows, log)
    };

    let (without_ac, _) = run("1");
    let (with_ac, log) = run("0");
    assert_eq!(with_ac.len(), 65);
    assert_eq!(without_ac.len(), 65);
    assert!(log.contains("reads=65 eligible=1 resolved=1"), "{log}");
    assert!(log.contains("rescued=1"), "{log}");
    for (i, (before, after)) in without_ac.iter().zip(&with_ac).enumerate() {
        if i == 11 {
            let old: Vec<_> = before.split('\t').collect();
            assert_ne!(old[1].parse::<u16>().unwrap() & 4, 0);
            let fields: Vec<_> = after.split('\t').collect();
            assert_eq!(fields[0], "r11");
            assert_eq!(&fields[2..6], &["short", "1", "17", "30M"]);
        } else {
            assert_eq!(before, after);
            let fields: Vec<_> = after.split('\t').collect();
            assert_eq!(&fields[2..4], &["ref", "151"]);
        }
    }
}
