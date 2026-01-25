use kira_ls_aligner::sketch::{MinimizerConfig, minimizers};

#[test]
fn minimizers_are_deterministic() {
    let seq = b"ACGTTGCATGTCGCATGATGCATGAGAGCT";
    let cfg = MinimizerConfig { k: 5, w: 4 };
    let a = minimizers(seq, &cfg);
    let b = minimizers(seq, &cfg);
    assert_eq!(a.len(), b.len());
    assert!(
        a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.hash == y.hash && x.pos == y.pos)
    );
}
