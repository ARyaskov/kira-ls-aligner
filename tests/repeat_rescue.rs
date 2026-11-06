mod common;

use std::fs;
use std::io::Write;

use common::*;

#[test]
fn prebuilt_truncated_repeat_index_still_maps_through_longer_seed_rescue() {
    let mut rng = Lcg::new(0x1234_5678_9abc_def0);
    let seq = synth_contig(&mut rng, 300);
    let dir = TempDir::new("repeat_rescue_e2e");
    let reference = dir.join("ref.fa");
    let mut fasta = fs::File::create(&reference).expect("create reference");
    for i in 0..6 {
        writeln!(fasta, ">copy{i}").unwrap();
        fasta.write_all(&seq).unwrap();
        fasta.write_all(b"\n").unwrap();
    }
    drop(fasta);

    let reads = dir.join("read.fq");
    let mut fastq = fs::File::create(&reads).expect("create reads");
    fastq.write_all(b"@repeat\n").unwrap();
    fastq.write_all(&seq[50..200]).unwrap();
    fastq.write_all(b"\n+\n").unwrap();
    fastq.write_all(&[b'I'; 150]).unwrap();
    fastq.write_all(b"\n").unwrap();
    drop(fastq);

    let index = dir.join("ref.kiraidx");
    run_bin(&[
        "index".into(),
        reference.clone().into(),
        "-o".into(),
        index.clone().into(),
        "-x".into(),
        "short".into(),
        "--only".into(),
        "short".into(),
        "-k".into(),
        "15".into(),
        "-w".into(),
        "5".into(),
        "--max-occ".into(),
        "2".into(),
    ]);
    let sam = run_mem(
        &reference,
        &[&reads],
        &dir.join("out.sam"),
        &[
            "--index",
            index.to_str().unwrap(),
            "--seed-occ-cap",
            "2",
            "-t",
            "1",
        ],
    );
    let body = sam_body(&sam);
    assert_eq!(body.len(), 1);
    let record = SamRecord::parse(body[0]);
    assert!(!record.is_unmapped(), "repeat-rescue read is unmapped");
    assert!(record.rname.starts_with("copy"));
}
