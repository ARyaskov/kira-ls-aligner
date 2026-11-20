#[test]
fn eval_reports_malformed_fields_and_excludes_unknown_mapq_from_thresholds() {
    let dir = tempfile::tempdir().unwrap();
    let sam = dir.path().join("input.sam");
    let run = || {
        std::process::Command::new(env!("CARGO_BIN_EXE_kira_ls_aligner"))
            .args(["eval", "--mapq-thresholds", "0,60"])
            .arg(&sam)
            .output()
            .unwrap()
    };

    std::fs::write(
        &sam,
        "@HD\tVN:1.6\nr:chr1:100-120\t0\tchr1\t101\tbad\t20M\t*\t0\t0\t*\t*\n",
    )
    .unwrap();
    let bad = run();
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("SAM line 2: invalid or missing MAPQ"));

    std::fs::write(
        &sam,
        concat!(
            "r:chr1:100-120\t0\tchr1\t101\t255\t20M\t*\t0\t0\t*\t*\n",
            "s:chr1:100-120\t0\tchr1\t101\t60\t20M\t*\t0\t0\t*\t*\n",
        ),
    )
    .unwrap();
    let ok = run();
    assert!(
        ok.status.success(),
        "{}",
        String::from_utf8_lossy(&ok.stderr)
    );
    let counts = String::from_utf8(ok.stdout).unwrap();
    assert!(counts.contains("mapped=2 correct=2"));
    assert!(counts.contains("unknown_mapq=1"));
    assert!(counts.contains("mapq>=0: correct=1 wrong=0"));
    assert!(counts.contains("mapq>=60: correct=1 wrong=0"));
}
