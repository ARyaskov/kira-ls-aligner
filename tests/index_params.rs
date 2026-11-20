//! `--index`: the minimizer parameters come from the index, as they do in
//! bwa-mem, unless the user pins different ones on the command line.

mod common;

use std::ffi::OsString;
use std::process::Command;

use common::*;

#[test]
fn minimizer_params_are_taken_from_the_index_unless_pinned() {
    let ds = build_dataset(
        "index_params",
        &SimParams {
            n_pairs: 40,
            n_single: 40,
            ..Default::default()
        },
    );
    let idx = ds.dir.join("ref.kiraidx");
    let idx_str = idx.to_str().unwrap();
    let index_args: Vec<OsString> = vec![
        "index".into(),
        ds.reference.clone().into(),
        "-o".into(),
        idx.clone().into(),
        "-k".into(),
        "15".into(),
        "-w".into(),
        "5".into(),
    ];
    run_bin(&index_args);

    // The default preset asks for k=19/w=10: the run must adopt the index's
    // 15/5 instead of refusing the index.
    let out = ds.dir.join("adopt.sam");
    let sam = run_mem(
        &ds.reference,
        &[&ds.single],
        &out,
        &["--index", idx_str, "-t", "2"],
    );
    let body = sam_body(&sam);
    assert_eq!(body.len(), ds.n_single, "one record per single-end read");
    let mapped = body
        .iter()
        .map(|l| SamRecord::parse(l))
        .filter(|r| !r.is_unmapped())
        .count();
    assert!(
        mapped * 10 >= ds.n_single * 9,
        "only {mapped}/{} reads mapped with the index's k/w",
        ds.n_single
    );

    // An explicit -k that disagrees with the index is an error, not a silent
    // override in either direction.
    let out2 = ds.dir.join("explicit.sam");
    let args = mem_args(
        &ds.reference,
        &[&ds.single],
        &out2,
        &["--index", idx_str, "-k", "19"],
    );
    let res = Command::new(env!("CARGO_BIN_EXE_kira_ls_aligner"))
        .args(&args)
        .output()
        .expect("spawn aligner");
    assert!(
        !res.status.success(),
        "a pinned -k must not be overridden by the index"
    );
    let err = String::from_utf8_lossy(&res.stderr);
    assert!(err.contains("mismatch"), "unexpected error text:\n{err}");
}

#[test]
fn stale_sidecar_is_rejected_before_output() {
    let ds = build_dataset("stale_sidecar", &SimParams::default());
    run_bin(&["index".into(), ds.reference.clone().into()]);
    let mut fasta = std::fs::read(&ds.reference).unwrap();
    let base = fasta.iter().position(|&b| b == b'\n').unwrap() + 1;
    fasta[base] = if fasta[base] == b'A' { b'C' } else { b'A' };
    std::fs::write(&ds.reference, fasta).unwrap();

    let out = ds.dir.join("stale.sam");
    let result = Command::new(env!("CARGO_BIN_EXE_kira_ls_aligner"))
        .args(mem_args(&ds.reference, &[&ds.single], &out, &[]))
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("index does not match FASTA"));
    assert!(!out.exists());
}

#[test]
fn invalid_minimizer_parameters_fail_before_reading_reference() {
    for args in [["-k", "32"], ["-w", "0"]] {
        let result = Command::new(env!("CARGO_BIN_EXE_kira_ls_aligner"))
            .args(["index", "missing.fa"])
            .args(args)
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr).contains("invalid short minimizer"));
    }
}

#[test]
fn a_single_table_can_seed_reads_on_both_sides_of_the_length_cutoff() {
    let ds = build_dataset("one_seed_table", &SimParams::default());
    run_bin(&[
        "index".into(),
        ds.reference.clone().into(),
        "--only".into(),
        "short".into(),
    ]);
    let out = ds.dir.join("only.sam");
    let sam = run_mem(
        &ds.reference,
        &[&ds.single],
        &out,
        &[
            "-x",
            "short",
            "--long-threshold",
            "1",
            "--set",
            "KIRA_AC_DISABLE=1",
            "-t",
            "2",
        ],
    );
    assert!(
        sam_body(&sam)
            .iter()
            .any(|line| !SamRecord::parse(line).is_unmapped())
    );
}
