use std::process::Command;

#[test]
fn unmapped_reads_keep_iupac_sequence_in_sam() {
    let dir = tempfile::tempdir().unwrap();
    let reference = dir.path().join("reference.fa");
    let reads = dir.path().join("reads.fq");
    std::fs::write(&reference, ">ref\n".to_owned() + &"ACGT".repeat(100) + "\n").unwrap();
    let seq = "RYSWKMBDHV";
    std::fs::write(
        &reads,
        format!("@read\n{seq}\n+\n{}\n", "I".repeat(seq.len())),
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_kira_ls_aligner"))
        .args(["mem", "-t", "1"])
        .args([reference, reads])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let sam = String::from_utf8(output.stdout).unwrap();
    let line = sam.lines().find(|line| !line.starts_with('@')).unwrap();
    let fields: Vec<_> = line.split('\t').collect();
    assert_ne!(fields[1].parse::<u16>().unwrap() & 4, 0);
    assert_eq!(fields[9], seq);
}

#[test]
fn reverse_alignment_keeps_iupac_and_replays_md_against_emitted_sequence() {
    let dir = tempfile::tempdir().unwrap();
    let reference = dir.path().join("reference.fa");
    let reads = dir.path().join("reads.fq");
    let mut state = 1337u64;
    let mut bases: Vec<u8> = (0..400)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            b"ACGT"[(state >> 62) as usize]
        })
        .collect();
    bases[190] = b'N';
    let original = &bases[120..270];
    let mut reverse = kira_ls_aligner::seq::reverse_complement(original);
    reverse[79] = b'R';
    std::fs::write(
        &reference,
        format!(">ref\n{}\n", std::str::from_utf8(&bases).unwrap()),
    )
    .unwrap();
    std::fs::write(
        &reads,
        format!(
            "@read\n{}\n+\n{}\n",
            std::str::from_utf8(&reverse).unwrap(),
            "I".repeat(reverse.len())
        ),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_kira_ls_aligner"))
        .args(["mem", "-x", "short", "-t", "1"])
        .args([reference, reads])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let sam = String::from_utf8(output.stdout).unwrap();
    let line = sam.lines().find(|line| !line.starts_with('@')).unwrap();
    let fields: Vec<_> = line.split('\t').collect();
    assert_eq!(fields[1].parse::<u16>().unwrap() & (4 | 16), 16);
    assert_eq!(fields[3], "121");
    assert_eq!(fields[5], "150M");
    let mut expected = original.to_vec();
    expected[70] = b'Y';
    assert_eq!(fields[9].as_bytes(), expected);
    assert!(fields.contains(&"NM:i:1"), "{line}");
    assert!(fields.contains(&"MD:Z:70N79"), "{line}");
}
