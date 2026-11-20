//! End-to-end integration test for `--split-prefix` (tiled alignment).
//!
//! Strategy:
//!   1. Build a synthetic FASTA with two ~2 kb contigs.
//!   2. Write a tiny paired-end FASTQ whose R1 maps to contig 0 and R2 maps
//!      to contig 1 (so both tiles must produce alignments for the merge to
//!      work).
//!   3. Run the tiled aligner with `tile_bytes` small enough to force one
//!      contig per tile.
//!   4. Parse the resulting SAM:
//!        - both reads should be mapped,
//!        - R1's RNAME = chr1, R2's RNAME = chr2,
//!        - paired flags (0x1, 0x40/0x80) are set,
//!        - global RNEXT references work across tile boundaries.
//!
//! This test exercises the *whole* tiled pipeline through its public
//! `run_tiled` entry point. It will catch regressions in:
//!   * `plan_tiles` packing logic
//!   * Per-tile ref_id remap (alignments would land on the wrong contig
//!     otherwise — the same class of bug minimap2 hit in #1058 / #1067)
//!   * Chunk-file write/read roundtrip
//!   * `MergeIter` multi-stream union
//!   * Final stage 5/6 over the global alignment set

#![allow(clippy::field_reassign_with_default)]
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use kira_ls_aligner::alignment::AlignmentConfig;
use kira_ls_aligner::alignment::splice::SpliceConfig;
use kira_ls_aligner::chaining::ChainingConfig;
use kira_ls_aligner::index::IndexConfig;
use kira_ls_aligner::index::tiling::plan_tiles;
use kira_ls_aligner::io::{HeaderConfig, IngestMode, OutputConfig};
use kira_ls_aligner::mapq::MapqConfig;
use kira_ls_aligner::pipeline::PipelineConfig;
use kira_ls_aligner::pipeline::pairing::PairedConfig;
use kira_ls_aligner::pipeline::stage1_sketch::SketchConfig;
use kira_ls_aligner::pipeline::tiled::{TiledRunConfig, run_tiled};
use kira_ls_aligner::seeding::SeedingConfig;
use kira_ls_aligner::types::{RefBases, RefSeq, Reference};

fn synth_dna(seed: u64, len: usize) -> Vec<u8> {
    let mut s = seed;
    let bases = b"ACGT";
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        out.push(bases[(s >> 33) as usize & 3]);
    }
    out
}

fn write_tmp_file(name: &str, content: &[u8]) -> PathBuf {
    let path = std::env::temp_dir().join(format!("kira-split-{}-{}", std::process::id(), name));
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(content).unwrap();
    path
}

fn write_fastq(name: &str, records: &[(&str, &[u8])]) -> PathBuf {
    let mut buf = Vec::new();
    for (id, seq) in records {
        buf.extend_from_slice(b"@");
        buf.extend_from_slice(id.as_bytes());
        buf.push(b'\n');
        buf.extend_from_slice(seq);
        buf.push(b'\n');
        buf.extend_from_slice(b"+\n");
        // dummy quality
        buf.resize(buf.len() + seq.len(), b'I');
        buf.push(b'\n');
    }
    write_tmp_file(name, &buf)
}

fn full_pipeline_cfg() -> PipelineConfig {
    let alignment_cfg = AlignmentConfig {
        match_score: 1,
        mismatch: 4,
        gap_open: 6,
        gap_extend: 1,
        bandwidth: 50,
        xdrop: 50,
        clip_penalty: 5,
    };
    let mut paired = PairedConfig::default();
    paired.mode = IngestMode::TwoFile;
    PipelineConfig {
        sketch: SketchConfig {
            short_k: 19,
            short_w: 10,
            long_k: 19,
            long_w: 10,
            long_read_threshold: 500,
        },
        seeding: SeedingConfig {
            min_anchor_len: 20,
            max_occ: 500,
            max_hits_per_minimizer: 16,
            long_read_threshold: 500,
            mate_window: None,
        },
        chaining: ChainingConfig {
            max_dist: 500,
            max_anchors: 2000,
            max_chains: 5,
            gap_open: 5,
            gap_extend: 1,
            log_gap: 0.2,
            rmq_window: 256,
            keep_anchors: false,
        },
        alignment: alignment_cfg,
        accept_enable: true,
        accept_only_top1: true,
        accept_span_slack: 15,
        accept_min_identity: 98.5,
        accept_max_mismatches: 5,
        accept_require_score_margin: 0,
        dp_topk: 1,
        dp_abort_margin: 20,
        debug_prefilter: false,
        debug_prefilter_n: 0,
        debug_force_accept: false,
        debug_force_accept_n: 100,
        long_read_threshold: 500,
        max_alignments: 1,
        min_chain_ratio: 0.4,
        short_preset: true,
        mapq: MapqConfig {
            short_read_len: 300,
            mapq_cap_short: 60,
            mapq_cap_long: 60,
            alt_mask: None,
        },
        output: OutputConfig::full(),
        paired,
        splice: SpliceConfig::default(),
        min_output_score: i32::MIN,
        skip_mate_rescue: false,
        skip_pairing: false,
        primary_5p: false,
        alt_mask: None,
    }
}

fn build_reference() -> (Reference, Vec<u8>, Vec<u8>) {
    let c0 = synth_dna(0x1111_2222_3333_4444, 2000);
    let c1 = synth_dna(0x5555_6666_7777_8888, 2000);
    let reference = Reference {
        sequences: vec![
            RefSeq {
                name: "chr1".to_string(),
                bases: RefBases::Owned(c0.clone()),
            },
            RefSeq {
                name: "chr2".to_string(),
                bases: RefBases::Owned(c1.clone()),
            },
        ],
    };
    (reference, c0, c1)
}

fn parse_sam_record(line: &str) -> Vec<&str> {
    line.trim_end_matches('\n').split('\t').collect()
}

#[test]
fn tiled_alignment_recovers_reads_across_tiles() {
    let (reference, c0, c1) = build_reference();

    // R1 = 150 bp from contig 0 starting at offset 500.
    // R2 = reverse-complement of 150 bp from contig 1 starting at 700.
    let r1_seq = &c0[500..650];
    let mut r2_seq: Vec<u8> = c1[700..850].to_vec();
    r2_seq.reverse();
    for b in r2_seq.iter_mut() {
        *b = match *b {
            b'A' => b'T',
            b'C' => b'G',
            b'G' => b'C',
            b'T' => b'A',
            x => x,
        };
    }

    let r1_path = write_fastq("r1", &[("pair1/1", r1_seq)]);
    let r2_path = write_fastq("r2", &[("pair1/2", &r2_seq)]);
    let out_path = std::env::temp_dir().join(format!("kira-split-{}-out.sam", std::process::id()));
    let prefix_path =
        std::env::temp_dir().join(format!("kira-split-{}-prefix", std::process::id()));

    // Force one contig per tile: each contig is 2000 bytes; cap at 1500.
    let tile_plan = plan_tiles(&reference, 1500);
    assert_eq!(tile_plan.n_tiles(), 2, "expected 2 tiles");

    let cfg = TiledRunConfig {
        threads: 2,
        num_p_threads: None,
        num_e_threads: None,
        batch_bases: 1_000_000,
        index_cfg: IndexConfig {
            short_k: 19,
            short_w: 10,
            long_k: 19,
            long_w: 10,
            max_occ: 500,
            build_short: true,
            build_long: false,
        },
        pipeline_cfg: full_pipeline_cfg(),
        read_group: None,
        header: Some(HeaderConfig::default()),
        split_prefix: prefix_path.clone(),
        junctions: None,
        junc_bed_tolerance: 2,
        keep_comment: false,
        alt_contigs: None,
    };

    run_tiled(
        reference,
        &[r1_path.clone(), r2_path.clone()],
        Some(out_path.clone()),
        cfg,
        tile_plan,
    )
    .expect("run_tiled");

    // Read back the SAM and check structure.
    let sam = std::fs::read_to_string(&out_path).expect("read output sam");
    let body_lines: Vec<&str> = sam.lines().filter(|l| !l.starts_with('@')).collect();
    assert_eq!(
        body_lines.len(),
        2,
        "expected exactly 2 SAM records (one per mate), got:\n{}",
        sam
    );

    let cols0 = parse_sam_record(body_lines[0]);
    let cols1 = parse_sam_record(body_lines[1]);
    assert_eq!(cols0[0], "pair1", "R1 qname (mate suffix stripped)");
    assert_eq!(cols1[0], "pair1", "R2 qname (mate suffix stripped)");

    // R1 should be mapped to chr1
    let flag0: u32 = cols0[1].parse().unwrap();
    assert_eq!(flag0 & 0x4, 0, "R1 should not be unmapped");
    assert_eq!(cols0[2], "chr1", "R1 maps to chr1");
    // R2 should be mapped to chr2
    let flag1: u32 = cols1[1].parse().unwrap();
    assert_eq!(flag1 & 0x4, 0, "R2 should not be unmapped");
    assert_eq!(cols1[2], "chr2", "R2 maps to chr2 (cross-tile pair)");

    // Paired flags must survive the tile split.
    assert!(flag0 & 0x1 != 0, "R1 paired bit");
    assert!(flag0 & 0x40 != 0, "R1 first-in-pair bit");
    assert!(flag1 & 0x1 != 0, "R2 paired bit");
    assert!(flag1 & 0x80 != 0, "R2 second-in-pair bit");

    // RNEXT should reference the OTHER tile's contig — verifies global
    // ref_id remap end-to-end.
    assert_eq!(cols0[6], "chr2", "R1 RNEXT points to R2's contig");
    assert_eq!(cols1[6], "chr1", "R2 RNEXT points to R1's contig");

    // Clean up.
    let _ = std::fs::remove_file(r1_path);
    let _ = std::fs::remove_file(r2_path);
    let _ = std::fs::remove_file(out_path);
}

#[test]
fn tiled_single_tile_is_trivial_and_still_works() {
    // When the reference is small enough to fit in one tile the pipeline
    // should still produce the right output — tile-plan trivial-case
    // shouldn't get a special-cased shortcut that bypasses the chunk I/O
    // path (it doesn't, but worth asserting in case someone adds one).
    let (reference, c0, _c1) = build_reference();
    let r1 = &c0[100..250];
    let r1_path = write_fastq("solo-r1", &[("read1", r1)]);
    let out_path = std::env::temp_dir().join(format!("kira-split-{}-solo.sam", std::process::id()));
    let prefix_path =
        std::env::temp_dir().join(format!("kira-split-{}-solo-prefix", std::process::id()));

    // Huge tile budget → 1 tile.
    let tile_plan = plan_tiles(&reference, 100_000_000);
    assert_eq!(tile_plan.n_tiles(), 1);

    let mut pipe = full_pipeline_cfg();
    pipe.paired.mode = IngestMode::Unpaired;
    let cfg = TiledRunConfig {
        threads: 1,
        num_p_threads: None,
        num_e_threads: None,
        batch_bases: 1_000_000,
        index_cfg: IndexConfig {
            short_k: 19,
            short_w: 10,
            long_k: 19,
            long_w: 10,
            max_occ: 500,
            build_short: true,
            build_long: false,
        },
        pipeline_cfg: pipe,
        read_group: None,
        header: Some(HeaderConfig::default()),
        split_prefix: prefix_path.clone(),
        junctions: None,
        junc_bed_tolerance: 2,
        keep_comment: false,
        alt_contigs: None,
    };
    run_tiled(
        reference,
        std::slice::from_ref(&r1_path),
        Some(out_path.clone()),
        cfg,
        tile_plan,
    )
    .expect("run_tiled solo");

    let sam = std::fs::read_to_string(&out_path).expect("read solo output");
    let body: Vec<&str> = sam.lines().filter(|l| !l.starts_with('@')).collect();
    assert_eq!(body.len(), 1);
    let cols = parse_sam_record(body[0]);
    assert_eq!(cols[0], "read1");
    assert_eq!(cols[2], "chr1");
    let flag: u32 = cols[1].parse().unwrap();
    assert_eq!(flag & 0x4, 0, "should be mapped");
    assert_eq!(flag & 0x1, 0, "single-end → no 0x1");

    let _ = std::fs::remove_file(r1_path);
    let _ = std::fs::remove_file(out_path);
}

#[test]
fn tiled_mapq_counts_seed_copies_on_tiles_without_a_competing_alignment() {
    use kira_ls_aligner::index::Index;
    use kira_ls_aligner::pipeline::{Pipeline, stage0_input};
    use kira_ls_aligner::types::ReadRecord;

    let root = tempfile::tempdir().unwrap();
    let query = synth_dna(6731, 120);
    // The decoy contains eight isolated copies of every query seed, in
    // reverse window order. It contains no full-length copy of the read.
    let mut fragments = Vec::new();
    for window in query.windows(19).rev() {
        for _ in 0..8 {
            fragments.extend_from_slice(window);
            fragments.extend_from_slice(&[b'N'; 32]);
        }
    }
    let reference = Reference {
        sequences: vec![
            RefSeq {
                name: "target".into(),
                bases: RefBases::Owned(query.clone()),
            },
            RefSeq {
                name: "fragments".into(),
                bases: RefBases::Owned(fragments),
            },
        ],
    };
    let mut pipe = full_pipeline_cfg();
    pipe.sketch.short_w = 1;
    pipe.sketch.long_w = 1;
    pipe.paired.mode = IngestMode::Unpaired;
    pipe.min_output_score = 100;
    let cfg = IndexConfig {
        short_k: 19,
        long_k: 19,
        short_w: 1,
        long_w: 1,
        build_short: true,
        build_long: false,
        max_occ: 500,
    };
    let index = Index::try_build(reference.clone(), cfg).unwrap();
    let read =
        ReadRecord::new_unpaired("read".into(), query.clone(), Some(vec![b'I'; query.len()]));
    let (normal, _) = Pipeline::new(pipe.clone())
        .process_batch_scored(stage0_input::run(vec![read]), &index)
        .unwrap();
    assert_eq!(normal.reads[0].repeat_min_occ, 9);
    let expected = normal.alignments[0][0].mapq;
    assert_eq!(
        expected, 40,
        "the ordinary result must exercise the repeat cap"
    );

    let reads = write_fastq("copy-count", &[("read", &query)]);
    for (tile_bytes, tiles) in [(u64::MAX, 1), (300, 2)] {
        let output = root.path().join(format!("{tiles}.sam"));
        let plan = plan_tiles(&reference, tile_bytes);
        assert_eq!(plan.n_tiles(), tiles);
        run_tiled(
            reference.clone(),
            std::slice::from_ref(&reads),
            Some(output.clone()),
            TiledRunConfig {
                threads: 2,
                num_p_threads: None,
                num_e_threads: None,
                batch_bases: 4096,
                index_cfg: cfg,
                pipeline_cfg: pipe.clone(),
                read_group: None,
                header: None,
                split_prefix: root.path().join("job"),
                junctions: None,
                junc_bed_tolerance: 0,
                keep_comment: false,
                alt_contigs: None,
            },
            plan,
        )
        .unwrap();
        let sam = std::fs::read_to_string(output).unwrap();
        let fields = parse_sam_record(sam.lines().find(|l| !l.starts_with('@')).unwrap());
        assert_eq!(fields[2], "target");
        assert_eq!(fields[4].parse::<u8>().unwrap(), expected);
    }
    std::fs::remove_file(reads).unwrap();
}

/// Regression: the tiled merge concatenates per-tile hit lists in tile
/// order, and the paired-end MAPQ path preserves slot 0, so a weaker hit
/// from tile 1 used to stay primary even when tile 2 held the true locus.
/// chr1 carries only the first 90 bp of R1's sequence; chr2 carries all
/// 150 bp. The primary must be the chr2 placement.
#[test]
fn tiled_paired_primary_is_the_best_scoring_hit_across_tiles() {
    let c0_base = synth_dna(0x1111_2222_3333_4444, 2000);
    let c1 = synth_dna(0x5555_6666_7777_8888, 2000);
    // Plant a partial copy (90 bp) of chr2[700..850) into chr1 at 300.
    let mut c0 = c0_base.clone();
    c0[300..390].copy_from_slice(&c1[700..790]);
    let reference = Reference {
        sequences: vec![
            RefSeq {
                name: "chr1".to_string(),
                bases: RefBases::Owned(c0.clone()),
            },
            RefSeq {
                name: "chr2".to_string(),
                bases: RefBases::Owned(c1.clone()),
            },
        ],
    };

    // R1 = chr2[700..850) (full 150 bp match on chr2, 90 bp partial on chr1).
    // R2 is unrelated sequence that maps nowhere, so no concordant pair can
    // promote the right hit: the choice of primary rests on score alone.
    let r1_seq = &c1[700..850];
    let r2_seq = synth_dna(0x9999_aaaa_bbbb_cccc, 150);
    let r1_path = write_fastq("xt-r1", &[("xt/1", r1_seq)]);
    let r2_path = write_fastq("xt-r2", &[("xt/2", &r2_seq)]);
    let out_path = std::env::temp_dir().join(format!("kira-split-{}-xt.sam", std::process::id()));
    let prefix_path =
        std::env::temp_dir().join(format!("kira-split-{}-xt-prefix", std::process::id()));

    let tile_plan = plan_tiles(&reference, 1500);
    assert_eq!(tile_plan.n_tiles(), 2, "expected 2 tiles");

    let mut pipe = full_pipeline_cfg();
    // Keep the chr1 partial hit as a candidate so the ordering matters.
    pipe.min_chain_ratio = 0.1;
    pipe.dp_topk = 2;
    let cfg = TiledRunConfig {
        threads: 2,
        num_p_threads: None,
        num_e_threads: None,
        batch_bases: 1_000_000,
        index_cfg: IndexConfig {
            short_k: 19,
            short_w: 10,
            long_k: 19,
            long_w: 10,
            max_occ: 500,
            build_short: true,
            build_long: false,
        },
        pipeline_cfg: pipe,
        read_group: None,
        header: Some(HeaderConfig::default()),
        split_prefix: prefix_path.clone(),
        junctions: None,
        junc_bed_tolerance: 2,
        keep_comment: false,
        alt_contigs: None,
    };
    run_tiled(
        reference,
        &[r1_path.clone(), r2_path.clone()],
        Some(out_path.clone()),
        cfg,
        tile_plan,
    )
    .expect("run_tiled");

    let sam = std::fs::read_to_string(&out_path).expect("read output sam");
    let r1_primary = sam
        .lines()
        .filter(|l| !l.starts_with('@'))
        .map(parse_sam_record)
        .find(|c| {
            let flag: u32 = c[1].parse().unwrap();
            flag & 0x40 != 0 && flag & 0x900 == 0
        })
        .expect("R1 primary record");
    assert_eq!(
        r1_primary[2], "chr2",
        "primary must be the full-length chr2 hit:\n{sam}"
    );
    assert_eq!(r1_primary[3], "701");
    assert_eq!(r1_primary[5], "150M");
    let mapq: u32 = r1_primary[4].parse().unwrap();
    assert!(
        mapq > 0,
        "a full-length unique hit must not be MAPQ 0:\n{sam}"
    );

    let _ = std::fs::remove_file(r1_path);
    let _ = std::fs::remove_file(r2_path);
    let _ = std::fs::remove_file(out_path);
}

/// The tiled driver must resolve `alt_contigs` against the full reference and
/// install the ALT mask before stage 5: a secondary candidate on an ALT
/// contig makes `xa_has_alt` true, which selects the `-h` second cap and lets
/// a candidate list over the first cap through (bwa would suppress it
/// without the ALT awareness). chr1 carries the exact copy (primary); the
/// ALT contig chr2 plus chr3/chr4 carry one-SNV copies (secondaries, score
/// 145 vs 150).
#[test]
fn tiled_alt_mask_reaches_stage5_xa() {
    let c0 = synth_dna(0x1111_2222_3333_4444, 2000);
    let seg = c0[500..650].to_vec(); // the read: 150 bp exact copy on chr1
    let variant = || -> Vec<u8> {
        let mut v = seg.clone();
        v[75] = if v[75] == b'A' { b'G' } else { b'A' };
        v
    };
    let mut c1 = synth_dna(0x5555_6666_7777_8888, 2000);
    c1[800..950].copy_from_slice(&variant());
    let mut c2 = synth_dna(0x9999_aaaa_bbbb_cccc, 2000);
    c2[1200..1350].copy_from_slice(&variant());
    let mut c3 = synth_dna(0xdddd_eeee_ffff_0000, 2000);
    c3[300..450].copy_from_slice(&variant());
    let reference = Reference {
        sequences: vec![
            RefSeq {
                name: "chr1".to_string(),
                bases: RefBases::Owned(c0.clone()),
            },
            RefSeq {
                name: "chr2".to_string(),
                bases: RefBases::Owned(c1),
            },
            RefSeq {
                name: "chr3".to_string(),
                bases: RefBases::Owned(c2),
            },
            RefSeq {
                name: "chr4".to_string(),
                bases: RefBases::Owned(c3),
            },
        ],
    };

    let r1_path = write_fastq("alt-r1", &[("alt/1", &seg)]);
    let out_path = std::env::temp_dir().join(format!("kira-split-{}-alt.sam", std::process::id()));
    let prefix_path =
        std::env::temp_dir().join(format!("kira-split-{}-alt-prefix", std::process::id()));

    let tile_plan = plan_tiles(&reference, 1500);
    assert_eq!(tile_plan.n_tiles(), 4, "expected one contig per tile");

    let mut pipe = full_pipeline_cfg();
    pipe.paired.mode = IngestMode::Unpaired;
    // Keep every SNV copy as a candidate.
    pipe.min_chain_ratio = 0.1;
    pipe.dp_topk = 4;
    // First cap below the 3 secondary candidates; ALT cap above them.
    pipe.output.xa_max = 2;
    pipe.output.xa_max_alt = 200;
    let cfg = TiledRunConfig {
        threads: 2,
        num_p_threads: None,
        num_e_threads: None,
        batch_bases: 1_000_000,
        index_cfg: IndexConfig {
            short_k: 19,
            short_w: 10,
            long_k: 19,
            long_w: 10,
            max_occ: 500,
            build_short: true,
            build_long: false,
        },
        pipeline_cfg: pipe,
        read_group: None,
        header: Some(HeaderConfig::default()),
        split_prefix: prefix_path.clone(),
        junctions: None,
        junc_bed_tolerance: 2,
        keep_comment: false,
        alt_contigs: Some(Arc::new(["chr2".to_string()].into_iter().collect())),
    };
    run_tiled(
        reference,
        std::slice::from_ref(&r1_path),
        Some(out_path.clone()),
        cfg,
        tile_plan,
    )
    .expect("run_tiled alt");

    let sam = std::fs::read_to_string(&out_path).expect("read alt output");
    let body: Vec<&str> = sam.lines().filter(|l| !l.starts_with('@')).collect();
    assert_eq!(body.len(), 1, "one unpaired read:\n{sam}");
    let cols = parse_sam_record(body[0]);
    assert_eq!(cols[2], "chr1", "exact copy must be the primary:\n{sam}");
    let xa = cols
        .iter()
        .find(|c| c.starts_with("XA:Z:"))
        .expect("a secondary ALT candidate must select the second cap and keep XA");
    assert_eq!(
        xa.matches(';').count(),
        3,
        "all 3 secondary candidates must be listed (ALT one included):\n{sam}"
    );
    assert!(
        xa.contains("chr2,") && xa.contains("chr3,") && xa.contains("chr4,"),
        "XA must name the ALT and both non-ALT variants:\n{sam}"
    );

    let _ = std::fs::remove_file(r1_path);
    let _ = std::fs::remove_file(out_path);
}
