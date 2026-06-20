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

use std::io::Write;
use std::path::PathBuf;

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
        for _ in 0..seq.len() {
            buf.push(b'I');
        }
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
        },
        chaining: ChainingConfig {
            max_dist: 500,
            max_anchors: 2000,
            max_chains: 5,
            gap_open: 5,
            gap_extend: 1,
            log_gap: 0.2,
            rmq_window: 256,
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
        },
        output: OutputConfig::full(),
        paired,
        splice: SpliceConfig::default(),
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
    assert_eq!(cols0[0], "pair1/1", "R1 qname");
    assert_eq!(cols1[0], "pair1/2", "R2 qname");

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
    };
    run_tiled(
        reference,
        &[r1_path.clone()],
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
