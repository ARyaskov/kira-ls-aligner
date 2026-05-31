use super::*;
use tempfile::NamedTempFile;

fn sample_aln(ref_id: u32, score: i32, is_rev: bool) -> Alignment {
    Alignment {
        kind: AlignmentKind::DpAligned,
        ref_id,
        ref_start: 100,
        ref_end: 250,
        read_start: 0,
        read_end: 150,
        cigar: vec![CigarOp {
            len: 150,
            op: CigarKind::Match,
        }],
        score,
        mapq: 60,
        is_rev,
        is_secondary: false,
        is_supplementary: false,
        nm: 2,
        md: "75A74".to_string(),
        as_score: score,
        xs_score: Some(score - 10),
        xs_strand: None,
        mate: MateInfo {
            is_paired: true,
            is_proper_pair: true,
            mate_is_unmapped: false,
            mate_is_rev: !is_rev,
            is_first_in_pair: true,
            is_second_in_pair: false,
            mate_ref_id: Some(ref_id),
            mate_pos: 400,
            tlen: 450,
        },
    }
}

#[test]
fn roundtrip_single_record_preserves_all_fields() {
    let f = NamedTempFile::new().unwrap();
    let path = f.path().to_path_buf();
    let aln = sample_aln(7, 142, false);
    {
        let mut w = ChunkWriter::create(&path).unwrap();
        w.write_record(42, std::slice::from_ref(&aln)).unwrap();
        w.finish().unwrap();
    }
    let mut r = ChunkReader::open(&path).unwrap();
    let (idx, recs) = r.next_record().unwrap().expect("one record");
    assert_eq!(idx, 42);
    assert_eq!(recs.len(), 1);
    let a = &recs[0];
    assert_eq!(a.ref_id, aln.ref_id);
    assert_eq!(a.ref_start, aln.ref_start);
    assert_eq!(a.ref_end, aln.ref_end);
    assert_eq!(a.read_start, aln.read_start);
    assert_eq!(a.read_end, aln.read_end);
    assert_eq!(a.nm, aln.nm);
    assert_eq!(a.md, aln.md);
    assert_eq!(a.score, aln.score);
    assert_eq!(a.as_score, aln.as_score);
    assert_eq!(a.xs_score, aln.xs_score);
    assert_eq!(a.is_rev, aln.is_rev);
    assert_eq!(a.cigar.len(), aln.cigar.len());
    assert_eq!(a.cigar[0].len, aln.cigar[0].len);
    assert_eq!(a.cigar[0].op, aln.cigar[0].op);
    assert_eq!(a.mate.is_paired, aln.mate.is_paired);
    assert_eq!(a.mate.is_proper_pair, aln.mate.is_proper_pair);
    assert_eq!(a.mate.mate_ref_id, aln.mate.mate_ref_id);
    assert_eq!(a.mate.mate_pos, aln.mate.mate_pos);
    assert_eq!(a.mate.tlen, aln.mate.tlen);
    // EOF
    assert!(r.next_record().unwrap().is_none());
}

#[test]
fn writer_skips_empty_records() {
    let f = NamedTempFile::new().unwrap();
    let path = f.path().to_path_buf();
    {
        let mut w = ChunkWriter::create(&path).unwrap();
        w.write_record(1, &[]).unwrap();
        w.write_record(2, &[sample_aln(0, 100, false)]).unwrap();
        w.write_record(3, &[]).unwrap();
        w.write_record(4, &[sample_aln(0, 200, true)]).unwrap();
        w.finish().unwrap();
    }
    let mut r = ChunkReader::open(&path).unwrap();
    let r1 = r.next_record().unwrap().unwrap();
    assert_eq!(r1.0, 2);
    let r2 = r.next_record().unwrap().unwrap();
    assert_eq!(r2.0, 4);
    assert!(r.next_record().unwrap().is_none());
}

#[test]
fn merge_iter_unions_records_at_same_idx() {
    let f1 = NamedTempFile::new().unwrap();
    let f2 = NamedTempFile::new().unwrap();
    let p1 = f1.path().to_path_buf();
    let p2 = f2.path().to_path_buf();

    {
        let mut w1 = ChunkWriter::create(&p1).unwrap();
        w1.write_record(1, &[sample_aln(0, 100, false)]).unwrap();
        w1.write_record(5, &[sample_aln(1, 50, true)]).unwrap();
        w1.finish().unwrap();
    }
    {
        let mut w2 = ChunkWriter::create(&p2).unwrap();
        w2.write_record(1, &[sample_aln(3, 120, false)]).unwrap();
        w2.write_record(7, &[sample_aln(4, 90, false)]).unwrap();
        w2.finish().unwrap();
    }

    let mut merge = MergeIter::new(vec![
        ChunkReader::open(&p1).unwrap(),
        ChunkReader::open(&p2).unwrap(),
    ]);
    // idx 1 — both tiles contribute → 2 alignments
    let (i1, a1) = merge.next_merged().unwrap().unwrap();
    assert_eq!(i1, 1);
    assert_eq!(a1.len(), 2);
    // ref_ids should be {0, 3} in some order
    let mut refs: Vec<u32> = a1.iter().map(|a| a.ref_id).collect();
    refs.sort();
    assert_eq!(refs, vec![0, 3]);
    // idx 5 — only tile 1
    let (i2, a2) = merge.next_merged().unwrap().unwrap();
    assert_eq!(i2, 5);
    assert_eq!(a2.len(), 1);
    assert_eq!(a2[0].ref_id, 1);
    // idx 7 — only tile 2
    let (i3, a3) = merge.next_merged().unwrap().unwrap();
    assert_eq!(i3, 7);
    assert_eq!(a3.len(), 1);
    assert_eq!(a3[0].ref_id, 4);
    // EOF
    assert!(merge.next_merged().unwrap().is_none());
}

#[test]
fn bad_magic_errors() {
    let f = NamedTempFile::new().unwrap();
    let path = f.path().to_path_buf();
    std::fs::write(&path, b"NOTKIRACHNK").unwrap();
    // ChunkReader isn't Debug, so we can't use unwrap_err — match instead.
    match ChunkReader::open(&path) {
        Ok(_) => panic!("expected magic error"),
        Err(e) => assert!(
            format!("{e}").contains("bad chunk magic"),
            "wrong error: {e}"
        ),
    }
}
