use kira_ls_aligner::alignment::{AlignmentConfig, AnchorSpan, align_chain};
use kira_ls_aligner::types::{CigarKind, ReadRecord, Strand};

#[test]
fn alignment_finds_full_match() {
    let read = ReadRecord::new_unpaired("r1".to_string(), b"ACGTACGT".to_vec(), None);
    let reference = b"TTACGTACGTGG".to_vec();
    let span = AnchorSpan {
        ref_id: 0,
        ref_start: 2,
        ref_end: 10,
        read_start: 0,
        read_end: 8,
        strand: Strand::Forward,
    };
    let cfg = AlignmentConfig {
        match_score: 2,
        mismatch: 4,
        gap_open: 6,
        gap_extend: 1,
        bandwidth: 50,
        xdrop: 40,
        clip_penalty: 5,
    };
    let aln = align_chain(&read, &reference, &span, cfg, i32::MIN / 4);
    let cigar = aln
        .cigar
        .iter()
        .map(|op| op.to_string())
        .collect::<String>();
    assert!(cigar.contains("8M"));
    assert_eq!(aln.nm, 0);
}

fn end_indel_fixture() -> (Vec<u8>, Vec<u8>) {
    let base: Vec<u8> = (0..150).map(|i| b"CGTA"[i % 4]).collect();
    let mut reference: Vec<u8> = b"TT".to_vec();
    reference.extend_from_slice(&base);
    reference.extend_from_slice(b"TT");
    let mut read_seq: Vec<u8> = base[..145].to_vec();
    read_seq.push(b'G');
    read_seq.extend_from_slice(&base[145..149]);
    assert_eq!(read_seq.len(), 150);
    (reference, read_seq)
}

#[test]
fn end_of_read_insertion_emits_i_op_not_softclip() {
    let (reference, read_seq) = end_indel_fixture();
    let read = ReadRecord::new_unpaired("r1".to_string(), read_seq, None);
    let span = AnchorSpan {
        ref_id: 0,
        ref_start: 2,
        ref_end: 147,
        read_start: 0,
        read_end: 145,
        strand: Strand::Forward,
    };
    let cfg = AlignmentConfig {
        match_score: 1,
        mismatch: 4,
        gap_open: 6,
        gap_extend: 1,
        bandwidth: 50,
        xdrop: 50,
        clip_penalty: 5,
    };

    let aln = align_chain(&read, &reference, &span, cfg, i32::MIN / 4);
    let has_i = aln.cigar.iter().any(|op| op.op == CigarKind::Ins);
    let trailing_soft_len = aln
        .cigar
        .last()
        .filter(|op| op.op == CigarKind::SoftClip)
        .map(|op| op.len)
        .unwrap_or(0);
    let cigar_str = aln
        .cigar
        .iter()
        .map(|op| op.to_string())
        .collect::<String>();
    assert!(
        has_i,
        "expected I op in CIGAR with clip_penalty in effect, got: {}",
        cigar_str
    );
    assert!(
        trailing_soft_len <= 1,
        "expected no large trailing soft-clip, got: {}",
        cigar_str
    );
}

#[test]
fn end_of_read_insertion_softclips_without_penalty() {
    let (reference, read_seq) = end_indel_fixture();
    let read = ReadRecord::new_unpaired("r1".to_string(), read_seq, None);
    let span = AnchorSpan {
        ref_id: 0,
        ref_start: 2,
        ref_end: 147,
        read_start: 0,
        read_end: 145,
        strand: Strand::Forward,
    };
    let cfg = AlignmentConfig {
        match_score: 1,
        mismatch: 4,
        gap_open: 6,
        gap_extend: 1,
        bandwidth: 50,
        xdrop: 50,
        clip_penalty: 0,
    };

    let aln = align_chain(&read, &reference, &span, cfg, i32::MIN / 4);
    let has_i = aln.cigar.iter().any(|op| op.op == CigarKind::Ins);
    assert!(
        !has_i,
        "with clip_penalty=0 the local-SW soft-clip should still win"
    );
}

#[test]
fn public_simd_api_preserves_batch_cardinality_and_order() {
    use kira_ls_aligner::alignment::{BatchInput, align_batch_simd};
    use kira_ls_aligner::simd::SimdMode;

    let bases = b"ACGTTGCTAATCGTGCATGACCTG";
    let inputs: Vec<_> = (0..33)
        .map(|id| BatchInput {
            read_seq: bases,
            ref_window: bases,
            win_start: 100 * id,
            chain: AnchorSpan {
                ref_id: id,
                ref_start: 100 * id,
                ref_end: 100 * id + bases.len() as u32,
                read_start: 0,
                read_end: bases.len() as u32,
                strand: Strand::Forward,
            },
            is_rev: false,
            abort_score: 0,
        })
        .collect();

    let cfg = AlignmentConfig {
        match_score: 2,
        mismatch: 4,
        gap_open: 6,
        gap_extend: 1,
        bandwidth: 50,
        xdrop: 40,
        clip_penalty: 5,
    };
    for mode in [SimdMode::Avx2, SimdMode::Neon, SimdMode::AvxVnni] {
        for count in [0, 1, 4, 5, 16, 17, 33] {
            let results = align_batch_simd(&inputs[..count], cfg, mode);
            assert_eq!(results.len(), count);
            for (input, (aln, _)) in inputs.iter().zip(results) {
                assert_eq!(aln.ref_id, input.chain.ref_id);
                assert_eq!(
                    (aln.ref_start, aln.ref_end),
                    (input.win_start, input.chain.ref_end)
                );
                assert_eq!(aln.nm, 0);
            }
        }
    }
}

#[test]
fn scalar_and_mixed_batches_follow_the_seed_diagonal() {
    use kira_ls_aligner::alignment::{BatchInput, align_batch_simd};
    use kira_ls_aligner::simd::SimdMode;
    let read = b"GATTACAGCTTGAGTCCACG";
    let mut window = vec![b'C'; 100];
    window.extend_from_slice(read);
    let input = BatchInput {
        read_seq: read,
        ref_window: &window,
        win_start: 200,
        chain: AnchorSpan {
            ref_id: 0,
            ref_start: 303,
            ref_end: 320,
            read_start: 3,
            read_end: 20,
            strand: Strand::Forward,
        },
        is_rev: false,
        abort_score: i32::MIN / 4,
    };
    let shorter = BatchInput {
        read_seq: &read[..19],
        ..input.clone()
    };
    let cfg = AlignmentConfig {
        match_score: 2,
        mismatch: 4,
        gap_open: 6,
        gap_extend: 1,
        bandwidth: 8,
        xdrop: 40,
        clip_penalty: 5,
    };
    for batch in [vec![input.clone()], vec![input, shorter]] {
        for mode in [SimdMode::Scalar, SimdMode::Avx2, SimdMode::Neon] {
            let (aln, _) = &align_batch_simd(&batch, cfg, mode)[0];
            assert_eq!(
                (aln.ref_start, aln.ref_end, aln.score, aln.nm),
                (300, 320, 40, 0)
            );
            assert_eq!(
                aln.cigar
                    .iter()
                    .map(ToString::to_string)
                    .collect::<String>(),
                "20M"
            );
        }
    }
}

#[test]
fn avx2_scores_and_positions_survive_the_i16_boundary() {
    use kira_ls_aligner::alignment::{BatchInput, align_batch_simd};
    use kira_ls_aligner::simd::SimdMode;

    for (len, reward, offset) in [(7, 4681, 0), (8, 4096, 0), (150, 300, 0), (8, 1, 32768)] {
        let read = vec![b'G'; len];
        let mut reference = vec![b'C'; offset];
        reference.extend_from_slice(&read);
        let cfg = AlignmentConfig {
            match_score: reward,
            mismatch: 4,
            gap_open: 6,
            gap_extend: 1,
            bandwidth: 8,
            xdrop: 40,
            clip_penalty: 5,
        };
        let input = BatchInput {
            read_seq: &read,
            ref_window: &reference,
            win_start: 0,
            chain: AnchorSpan {
                ref_id: 0,
                ref_start: offset as u32,
                ref_end: (offset + len) as u32,
                read_start: 0,
                read_end: len as u32,
                strand: Strand::Forward,
            },
            is_rev: false,
            abort_score: i32::MIN / 4,
        };
        for mode in [SimdMode::Avx2, SimdMode::AvxVnni] {
            let (aln, _) = &align_batch_simd(std::slice::from_ref(&input), cfg, mode)[0];
            assert_eq!(
                (aln.score, aln.as_score),
                (reward * len as i32, reward * len as i32)
            );
            assert_eq!(
                (aln.ref_start, aln.ref_end),
                (offset as u32, (offset + len) as u32)
            );
            assert_eq!(aln.cigar.len(), 1);
            assert_eq!(aln.cigar[0].op, CigarKind::Match);
            assert_eq!(aln.cigar[0].len, len as u32);
        }
    }
}
