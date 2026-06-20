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

#[test]
fn tied_minima_are_emitted_without_leftmost_bias() {
    let mins = minimizers(b"AAAAA", &MinimizerConfig { k: 2, w: 2 });
    let positions: Vec<u32> = mins.iter().map(|m| m.pos).collect();
    assert_eq!(positions, vec![1, 2, 3]);
}
