//! Output determinism of the `kira_ls_aligner` binary.
//!
//! The same input must give the same SAM regardless of how the work is
//! split: across thread counts the body is byte-identical; across batch
//! sizes (`-K`) the primary placements are identical (the insert-size
//! estimator and auto-mode parameters lock on the first batch, so tags and
//! MAPQ may legitimately differ — see the note in `across_batch_sizes`).

mod common;

use std::collections::{BTreeSet, HashSet};

use common::*;

/// Enough pairs that the default read length gives well over one 100 kb
/// batch of R1 bases alone, so `-K 100000` really splits the input.
fn params() -> SimParams {
    SimParams {
        n_pairs: 800,
        n_single: 200,
        ..SimParams::default()
    }
}

/// Assert two SAM bodies are identical line for line, reporting the first
/// difference instead of dumping both files.
fn assert_bodies_identical(a: &str, b: &str, what: &str) {
    let a = sam_body(a);
    let b = sam_body(b);
    let differing = a
        .iter()
        .zip(b.iter())
        .filter(|(x, y)| x != y)
        .count()
        + a.len().abs_diff(b.len());
    if let Some((i, (x, y))) = a.iter().zip(b.iter()).enumerate().find(|(_, (x, y))| x != y) {
        panic!(
            "{what}: {differing} differing body line(s); first at record {}:\n  {x}\n  {y}",
            i + 1
        );
    }
    assert_eq!(a.len(), b.len(), "{what}: record counts differ");
}

#[test]
fn across_thread_counts() {
    let ds = build_dataset("det_threads", &params());

    let t1 = run_mem(&ds.reference, &[&ds.r1, &ds.r2], &ds.dir.join("p_t1.sam"), &["-t", "1"]);
    let t4 = run_mem(&ds.reference, &[&ds.r1, &ds.r2], &ds.dir.join("p_t4.sam"), &["-t", "4"]);
    assert!(sam_body(&t1).len() >= 2 * ds.n_pairs, "paired output is missing records");
    assert_bodies_identical(&t1, &t4, "paired, -t 1 vs -t 4");

    let s1 = run_mem(&ds.reference, &[&ds.single], &ds.dir.join("s_t1.sam"), &["-t", "1"]);
    let s4 = run_mem(&ds.reference, &[&ds.single], &ds.dir.join("s_t4.sam"), &["-t", "4"]);
    assert!(sam_body(&s1).len() >= ds.n_single, "single-end output is missing records");
    assert_bodies_identical(&s1, &s4, "single-end, -t 1 vs -t 4");
}

/// Primary placement key: (QNAME, mate role, unmapped bit, RNAME, POS).
fn placements(text: &str) -> BTreeSet<(String, Role, bool, String, u64)> {
    parse_sam(text)
        .records
        .iter()
        .filter(|r| r.is_primary())
        .map(|r| (r.qname.clone(), r.role(), r.is_unmapped(), r.rname.clone(), r.pos))
        .collect()
}

#[test]
fn across_batch_sizes() {
    let ds = build_dataset("det_batch", &params());

    let small = run_mem(
        &ds.reference,
        &[&ds.r1, &ds.r2],
        &ds.dir.join("k100k.sam"),
        &["-t", "2", "-K", "100000"],
    );
    let default = run_mem(&ds.reference, &[&ds.r1, &ds.r2], &ds.dir.join("kdef.sam"), &["-t", "2"]);

    let p_small = placements(&small);
    let p_default = placements(&default);
    assert_eq!(p_small.len(), 2 * ds.n_pairs, "one primary record per read (-K 100000)");
    assert_eq!(p_default.len(), 2 * ds.n_pairs, "one primary record per read (default -K)");

    let only_small: Vec<_> = p_small.difference(&p_default).collect();
    let only_default: Vec<_> = p_default.difference(&p_small).collect();
    assert!(
        only_small.is_empty() && only_default.is_empty(),
        "primary placements differ across -K:\n  only with -K 100000: {only_small:?}\n  only with default -K: {only_default:?}"
    );

    // Known limitation: the insert-size estimator and auto-mode parameters
    // lock on the first batch, so per-record details (MAPQ, proper-pair bit,
    // XS/XA) may differ once the input spans several batches. Report, do not
    // fail.
    let lines_small: HashSet<&str> = sam_body(&small).into_iter().collect();
    let lines_default: HashSet<&str> = sam_body(&default).into_iter().collect();
    let differing = lines_small.symmetric_difference(&lines_default).count();
    eprintln!(
        "[determinism] -K 100000 vs default -K: {differing} body line(s) differ out of {} / {}",
        lines_small.len(),
        lines_default.len()
    );
}
