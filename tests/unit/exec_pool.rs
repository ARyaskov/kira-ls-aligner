use super::*;

#[test]
fn homogeneous_executes_on_both_lanes() {
    let pool = DualPool::homogeneous(2).expect("homogeneous pool builds");
    let a = pool.install_compute(|| 1 + 2);
    let b = pool.install_light(|| 3 + 4);
    assert_eq!(a, 3);
    assert_eq!(b, 7);
    assert_eq!(pool.p_threads(), 2);
}

#[test]
fn auto_construction_succeeds() {
    // We don't care about the layout, only that the pool builds and
    // both install paths return.
    let pool = DualPool::new(DualPoolConfig::default()).expect("auto pool builds");
    assert!(pool.total_threads() >= 1);
    let s = pool.install_compute(|| pool.install_light(|| 42));
    assert_eq!(s, 42);
}

#[test]
fn worker_counts_alder_lake_i7_12700() {
    // 8 P-physical, 16 P-logical (SMT on), 4 E-physical/logical.
    let n_p_phys = 8;
    let n_p_log = 16;
    let n_e_log = 4;
    // Full saturation: every logical CPU pinned.
    assert_eq!(
        decide_worker_counts(Some(20), None, None, n_p_phys, n_p_log, n_e_log),
        (16, 4)
    );
    // `-t 12` = physical-core count → 8 P + 4 E (no SMT siblings).
    assert_eq!(
        decide_worker_counts(Some(12), None, None, n_p_phys, n_p_log, n_e_log),
        (8, 4)
    );
    // `-t 16` adds SMT siblings on top of the physical-core spread.
    assert_eq!(
        decide_worker_counts(Some(16), None, None, n_p_phys, n_p_log, n_e_log),
        (12, 4)
    );
    // `-t 8` keeps all 4 E-cores and only uses 4 P-physical.
    assert_eq!(
        decide_worker_counts(Some(8), None, None, n_p_phys, n_p_log, n_e_log),
        (4, 4)
    );
    // Tight budgets keep ≥1 P-core.
    assert_eq!(
        decide_worker_counts(Some(2), None, None, n_p_phys, n_p_log, n_e_log),
        (1, 1)
    );
    // No total set → use every detected core.
    assert_eq!(
        decide_worker_counts(None, None, None, n_p_phys, n_p_log, n_e_log),
        (16, 4)
    );
}

#[test]
fn explicit_overrides_are_honoured() {
    let n_p_phys = 8;
    let n_p_log = 16;
    let n_e_log = 4;
    // Both overrides — ignore total, take exactly what's asked for.
    assert_eq!(
        decide_worker_counts(Some(12), Some(6), Some(2), n_p_phys, n_p_log, n_e_log),
        (6, 2)
    );
    // Only P pinned — E fills the rest of the budget.
    assert_eq!(
        decide_worker_counts(Some(10), Some(8), None, n_p_phys, n_p_log, n_e_log),
        (8, 2)
    );
    // Only E pinned, P fills the rest.
    assert_eq!(
        decide_worker_counts(Some(10), None, Some(2), n_p_phys, n_p_log, n_e_log),
        (8, 2)
    );
}

#[test]
fn pick_p_logical_spreads_physical_first() {
    // 4 P-physical cores, 2 SMT siblings each → 8 logical.
    let topo = HybridTopology {
        p_physical: vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]],
        e_physical: vec![vec![8], vec![9]],
    };
    // First 4 workers land on distinct physical cores.
    assert_eq!(pick_p_logical_ids(&topo, 4), vec![0, 2, 4, 6]);
    // Next 4 fall on SMT siblings.
    assert_eq!(pick_p_logical_ids(&topo, 8), vec![0, 2, 4, 6, 1, 3, 5, 7]);
    // Partial second round still alternates.
    assert_eq!(pick_p_logical_ids(&topo, 6), vec![0, 2, 4, 6, 1, 3]);
}
