use super::*;

#[test]
fn chunk_path_appends_tile_index() {
    let p = chunk_path_for(Path::new("/tmp/run"), 0);
    assert_eq!(p, PathBuf::from("/tmp/run.tile-000.kchunk"));
    let p = chunk_path_for(Path::new("/tmp/run.partial"), 17);
    assert_eq!(p, PathBuf::from("/tmp/run.partial.tile-017.kchunk"));
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
