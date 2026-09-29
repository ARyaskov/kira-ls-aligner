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
    assert!(!res.status.success(), "a pinned -k must not be overridden by the index");
    let err = String::from_utf8_lossy(&res.stderr);
    assert!(err.contains("mismatch"), "unexpected error text:\n{err}");
}
