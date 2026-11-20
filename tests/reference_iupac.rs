use kira_ls_aligner::{
    index::{Index, IndexConfig},
    io::read_reference,
};
use std::process::Command;

#[test]
fn fasta_and_saved_index_preserve_iupac_in_md_and_reference_identity() {
    let dir = tempfile::tempdir().unwrap();
    let fasta = dir.path().join("reference.fa");
    let reads = dir.path().join("reads.fq");
    let sidecar = fasta.with_extension("kiraidx");
    let mut state = 1337u64;
    let mut bases: Vec<u8> = (0..400)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            b"ACGT"[(state >> 62) as usize]
        })
        .collect();
    let mut query = bases[120..270].to_vec();
    query[50] = b'A';
    query.remove(95);
    bases[170] = b'r';
    bases[215] = b'y';
    let write_fasta = |bases: &[u8]| {
        std::fs::write(
            &fasta,
            format!(">ref\n{}\n", std::str::from_utf8(bases).unwrap()),
        )
        .unwrap();
    };
    write_fasta(&bases);
    std::fs::write(
        &reads,
        format!(
            "@read\n{}\n+\n{}\n",
            std::str::from_utf8(&query).unwrap(),
            "I".repeat(query.len())
        ),
    )
    .unwrap();
    let run = || {
        Command::new(env!("CARGO_BIN_EXE_kira_ls_aligner"))
            .args(["mem", "-x", "short", "-t", "1"])
            .args([&fasta, &reads])
            .output()
            .unwrap()
    };
    let assert_md = |output: std::process::Output| {
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let sam = String::from_utf8(output.stdout).unwrap();
        let line = sam.lines().find(|line| !line.starts_with('@')).unwrap();
        let fields: Vec<_> = line.split('\t').collect();
        assert_eq!(fields[3], "121");
        assert!(fields.contains(&"NM:i:2"), "{line}");
        assert!(fields.contains(&"MD:Z:50R44^Y54"), "{line}");
    };
    assert_md(run());

    let cfg = IndexConfig {
        build_long: false,
        ..IndexConfig::default()
    };
    Index::try_build(read_reference(&fasta).unwrap(), cfg)
        .unwrap()
        .save(&sidecar)
        .unwrap();
    assert_md(run());

    bases[170] = b'y';
    write_fasta(&bases);
    let stale = run();
    assert!(!stale.status.success());
    assert!(String::from_utf8_lossy(&stale.stderr).contains("index does not match FASTA"));
}
