use super::*;

#[test]
fn detection_returns_consistent_shape() {
    // The result depends on host hardware; just sanity-check the
    // invariants when present.
    if let Some(t) = detect_topology() {
        assert!(t.is_split(), "topology must be split when Some is returned");
        assert!(t.total_logical() > 0);
        let p_logical = t.p_logical();
        let e_logical = t.e_logical();
        for p in &p_logical {
            assert!(!e_logical.contains(p), "logical CPU {p} appears in both pools");
        }
        // p_primary is a subset of p_logical (same for E).
        for p in t.p_primary() {
            assert!(p_logical.contains(&p));
        }
        for e in t.e_primary() {
            assert!(e_logical.contains(&e));
        }
        // Each P-physical core's first sibling should be present in
        // p_primary (and only that sibling).
        assert_eq!(t.p_primary().len(), t.n_p_physical());
        assert_eq!(t.e_primary().len(), t.n_e_physical());
    }
}
