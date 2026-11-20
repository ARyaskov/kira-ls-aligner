use super::*;

#[test]
fn overlapping_runs_own_and_clean_their_chunks_independently() {
    let root = tempfile::tempdir().unwrap();
    let prefix = root.path().join("sample");
    let first = scratch_directory(&prefix).unwrap();
    let second = scratch_directory(&prefix).unwrap();
    let a = first.path().join("tile-0.kchunk");
    let b = second.path().join("tile-0.kchunk");
    std::fs::write(&a, b"first").unwrap();
    std::fs::write(&b, b"second").unwrap();
    assert_eq!(std::fs::read(&a).unwrap(), b"first");
    drop(first);
    assert!(!a.exists());
    assert_eq!(std::fs::read(&b).unwrap(), b"second");
    drop(second);
    assert_eq!(root.path().read_dir().unwrap().count(), 0);
}

#[test]
fn counts_are_summed_by_seed_before_taking_the_minimum() {
    use crate::types::{ReadRecord, RefBases, RefSeq};
    let scratch = tempfile::tempdir().unwrap();
    let cfg = IndexConfig {
        short_k: 2,
        short_w: 1,
        long_k: 2,
        long_w: 1,
        build_short: true,
        build_long: false,
        max_occ: 1,
    };
    let mut reads = vec![
        ReadRecord::new_unpaired("query".into(), b"ACNAGNGG".to_vec(), None),
        ReadRecord::new_unpaired("absent".into(), b"GG".to_vec(), None),
    ];
    // AC occurs 1 + 6 times, AG occurs 4 + 1 times, GG is absent.
    // Both tile minima are 1, but the minimum positive global count is 5.
    for (tile, bases) in [b"ACNAGNAGNAGNAG".as_slice(), b"ACNACNACNACNACNACNAG"]
        .iter()
        .enumerate()
    {
        let reference = Reference {
            sequences: vec![RefSeq {
                name: format!("tile{tile}"),
                bases: RefBases::Owned(bases.to_vec()),
            }],
        };
        let index = Index::try_build(reference, cfg).unwrap();
        let mut pass = repeat_counts::RepeatPass::open(scratch.path(), tile, 2).unwrap();
        let ((_, _), counts) = crate::pipeline::stage2_seeding::run_fused_with_counts(
            stage0_input::run(reads.clone()),
            &index,
            crate::pipeline::stage1_sketch::SketchConfig {
                short_k: 2,
                short_w: 1,
                long_k: 2,
                long_w: 1,
                long_read_threshold: 500,
            },
            crate::seeding::SeedingConfig {
                min_anchor_len: 2,
                max_occ: 100,
                max_hits_per_minimizer: 100,
                long_read_threshold: 500,
                mate_window: None,
            },
            // Mimic one AC-resolved read; it still needs a sketch for MAPQ.
            &[tile == 0, false],
            Some((cfg, 500)),
        );
        pass.record(&reads, &counts).unwrap();
        pass.finish().unwrap();
    }
    let mut summary = repeat_counts::RepeatSummary::open(scratch.path()).unwrap();
    summary.restore(&mut reads).unwrap();
    summary.finish().unwrap();
    assert_eq!(reads[0].repeat_min_occ, 5);
    assert_eq!(reads[1].repeat_min_occ, 0);
}

#[test]
fn remap_zero_offset_is_noop() {
    use crate::index::tiling::Tile;
    let tile = Tile {
        contig_start: 0,
        contig_end: 5,
        global_ref_id_offset: 0,
        total_bytes: 100,
    };
    let mut alns = vec![vec![make_aln(3)]];
    remap_alignments_to_global(&mut alns, &tile);
    assert_eq!(alns[0][0].ref_id, 3);
}

#[test]
fn remap_nonzero_offset_adds_to_ref_id_and_mate() {
    use crate::index::tiling::Tile;
    let tile = Tile {
        contig_start: 5,
        contig_end: 8,
        global_ref_id_offset: 5,
        total_bytes: 100,
    };
    let mut a = make_aln(2);
    a.mate.mate_ref_id = Some(1);
    let mut alns = vec![vec![a]];
    remap_alignments_to_global(&mut alns, &tile);
    assert_eq!(alns[0][0].ref_id, 7, "2 + offset 5");
    assert_eq!(alns[0][0].mate.mate_ref_id, Some(6), "1 + offset 5");
}

fn make_aln(ref_id: u32) -> Alignment {
    use crate::types::{AlignmentKind, CigarKind, CigarOp};
    Alignment {
        kind: AlignmentKind::DpAligned,
        ref_id,
        ref_start: 0,
        ref_end: 100,
        read_start: 0,
        read_end: 100,
        cigar: vec![CigarOp {
            len: 100,
            op: CigarKind::Match,
        }],
        score: 100,
        mapq: 60,
        is_rev: false,
        is_secondary: false,
        is_supplementary: false,
        nm: 0,
        md: "100".to_string(),
        as_score: 100,
        xs_score: None,
        xs_strand: None,
        mate: MateInfo::default(),
    }
}
