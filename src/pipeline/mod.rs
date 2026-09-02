pub mod chunk_io;
pub mod insert_estimate;
pub mod mode;
pub mod pairing;
pub mod split_read;
pub mod stage0_input;
pub mod stage1_sketch;
pub mod stage2_seeding;
pub mod stage3_chaining;
pub mod stage4_alignment;
pub mod stage5_scoring;
pub mod stage6_output;
pub mod tiled;

use std::time::{Duration, Instant};

use anyhow::Result;

use std::sync::{Arc, RwLock};

use crate::alignment::AlignmentConfig;
use crate::alignment::ac_batch::{self, AcBatchOutput};
use crate::alignment::junc_bed::JunctionIndex;
use crate::alignment::splice::SpliceConfig;
use crate::chaining::ChainingConfig;
use crate::exec::DualPool;
use crate::index::Index;
use crate::io::{OutputConfig, SamFormatter};
use crate::mapq::MapqConfig;
use crate::pipeline::insert_estimate::InsertEstimator;
use crate::pipeline::pairing::{
    PairedConfig, RescueConfig, apply_pairing, pair_rerank, rescue_discordant_pairs,
    rescue_unmapped_mates,
};
use crate::pipeline::stage1_sketch::{
    SketchBatchStats, SketchConfig, run_with_mask as sketch_run_with_mask,
};
use crate::pipeline::stage2_seeding::{SeedBatchStats, run as seed_run};
use crate::pipeline::stage3_chaining::{ChainingBatchStats, run as chain_run};
use crate::pipeline::stage4_alignment::{
    AlignmentBatchStats, AlignmentStageConfig, run as align_run,
};
use crate::pipeline::stage6_output::serialize_into as output_serialize_into;
use crate::seeding::SeedingConfig;

/// Pipeline configuration aggregated across stages.
#[derive(Clone, Copy, Debug)]
pub struct PipelineConfig {
    pub sketch: SketchConfig,
    pub seeding: SeedingConfig,
    pub chaining: ChainingConfig,
    pub alignment: AlignmentConfig,
    pub accept_enable: bool,
    pub accept_only_top1: bool,
    pub accept_span_slack: usize,
    pub accept_min_identity: f32,
    pub accept_max_mismatches: usize,
    pub accept_require_score_margin: i32,
    pub dp_topk: usize,
    pub dp_abort_margin: i32,
    pub debug_prefilter: bool,
    pub debug_prefilter_n: usize,
    pub debug_force_accept: bool,
    pub debug_force_accept_n: usize,
    pub long_read_threshold: usize,
    pub max_alignments: usize,
    pub min_chain_ratio: f32,
    pub short_preset: bool,
    pub mapq: MapqConfig,
    pub output: OutputConfig,
    /// Paired-end ingestion + proper-pair policy.
    pub paired: PairedConfig,
    /// Splice-aware alignment policy.
    pub splice: SpliceConfig,
    /// bwa-mem `-T`: alignments scoring below this are dropped before pairing
    /// and output; a read whose best alignment falls under it is unmapped.
    pub min_output_score: i32,
    /// bwa-mem `-S`: skip mate rescue.
    pub skip_mate_rescue: bool,
    /// bwa-mem `-P`: skip pairing — no joint re-rank, no discordant rescue.
    /// Pair flags/TLEN are still stamped from each mate's own best hit.
    pub skip_pairing: bool,
    /// bwa-mem `-5`: for a split read, the segment with the smallest read
    /// coordinate is the primary record.
    pub primary_5p: bool,
    /// Per-contig ALT flag from `REF.alt` (also carried in `mapq`). With it,
    /// a read whose best hit is on an ALT contig but which has an equivalent
    /// primary-assembly hit is reported on the primary assembly, as
    /// `bwa-postalt` does.
    pub alt_mask: Option<&'static [bool]>,
}

impl PipelineConfig {
    /// Install the ALT-contig mask (both the placement policy and the MAPQ
    /// model read it). Re-applied whenever a read-mode profile replaces the
    /// config, since profiles are built before the index is known.
    pub fn set_alt_mask(&mut self, mask: Option<&'static [bool]>) {
        self.alt_mask = mask;
        self.mapq.alt_mask = mask;
    }
}

/// Per-batch stage timing results (stages 1-6).
#[derive(Clone, Debug)]
pub struct PipelineStageTimes {
    pub stages: [Duration; 6],
}

impl PipelineStageTimes {
    pub fn total(&self) -> Duration {
        self.stages.iter().copied().sum()
    }
}

/// Per-batch pipeline stats.
#[derive(Clone, Debug)]
pub struct PipelineBatchStats {
    pub times: PipelineStageTimes,
    pub align: AlignmentBatchStats,
    pub sketch: SketchBatchStats,
    pub seed: SeedBatchStats,
    pub chaining: ChainingBatchStats,
}

/// Result of `Pipeline::process_batch_serialized` — stats plus a ready-to-write SAM byte buffer.
pub struct PipelineBatchOutput {
    pub stats: PipelineBatchStats,
    pub sam_buf: Vec<u8>,
}

pub struct Pipeline {
    pub config: PipelineConfig,
    /// Optional in-memory junction annotation (`--junc-bed`).
    pub junctions: Option<Arc<JunctionIndex>>,
    /// Position tolerance for junction lookup (matches the CLI flag).
    pub junc_bed_tolerance: u32,
    /// Insert-size estimator.
    pub insert_estimator: Arc<RwLock<InsertEstimator>>,
    /// Hybrid-aware executor. The compute/light methods remain useful routing
    /// labels, while both use the full selected worker set because stages are
    /// executed serially.
    pub pool: Arc<DualPool>,
}

impl Pipeline {
    pub fn new(config: PipelineConfig) -> Self {
        Self::with_pool(
            config,
            Arc::new(
                DualPool::homogeneous(
                    std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(1),
                )
                .expect("homogeneous fallback pool always builds"),
            ),
        )
    }

    /// Build a pipeline that routes stages through a caller-supplied
    /// hybrid-aware executor. The standalone callers (`Aligner`,
    /// `run_tiled`) build a single `DualPool` once at startup and share
    /// it across every batch.
    pub fn with_pool(config: PipelineConfig, pool: Arc<DualPool>) -> Self {
        let estimator = Arc::new(RwLock::new(InsertEstimator::new(config.paired)));
        Self {
            config,
            junctions: None,
            junc_bed_tolerance: 2,
            insert_estimator: estimator,
            pool,
        }
    }

    /// Attach a parsed junction annotation.
    pub fn with_junctions(mut self, junctions: Option<Arc<JunctionIndex>>, tolerance: u32) -> Self {
        self.junctions = junctions;
        self.junc_bed_tolerance = tolerance;
        self
    }

    /// Run only stages 1-4 (sketch → seed → chain → align).
    ///
    /// Stage routing labels distinguish compute-heavy and bookkeeping work;
    /// both consume the complete configured pool.
    pub fn process_to_align_batch(
        &self,
        input: stage0_input::InputBatch,
        index: &Index,
    ) -> crate::pipeline::stage4_alignment::AlignBatch {
        let ac = self.pool.install_compute(|| {
            ac_batch::run(
                &input.reads,
                index,
                self.config.alignment,
                self.config.max_alignments,
            )
        });
        let skip_mask = ac.resolved_mask();
        let sketch = self
            .pool
            .install_compute(|| sketch_run_with_mask(input, self.config.sketch, &skip_mask));
        let seeds = self
            .pool
            .install_light(|| seed_run(sketch, index, self.config.seeding));
        let chains = self
            .pool
            .install_light(|| chain_run(seeds, self.config.chaining));
        let mut align = self.pool.install_compute(|| {
            if self.config.splice.enabled {
                crate::alignment::splice::splice_align_batch(
                    chains,
                    index,
                    self.config.alignment,
                    self.config.splice,
                    self.junctions.as_deref(),
                    self.junc_bed_tolerance,
                )
            } else {
                align_run(
                    chains,
                    index,
                    AlignmentStageConfig {
                        cfg: self.config.alignment,
                        min_chain_ratio: self.config.min_chain_ratio,
                        accept_enable: self.config.accept_enable,
                        accept_only_top1: self.config.accept_only_top1,
                        accept_span_slack: self.config.accept_span_slack,
                        accept_min_identity: self.config.accept_min_identity,
                        accept_max_mismatches: self.config.accept_max_mismatches,
                        accept_require_score_margin: self.config.accept_require_score_margin,
                        dp_topk: self.config.dp_topk,
                        dp_abort_margin: self.config.dp_abort_margin,
                        debug_prefilter: self.config.debug_prefilter,
                        debug_prefilter_n: self.config.debug_prefilter_n,
                        debug_force_accept: self.config.debug_force_accept,
                        debug_force_accept_n: self.config.debug_force_accept_n,
                        long_read_threshold: self.config.long_read_threshold,
                        max_alignments: self.config.max_alignments,
                        short_preset: self.config.short_preset,
                        paired: self.config.paired,
                    },
                )
            }
        });
        merge_ac_alignments(&mut align.alignments, ac);
        align
    }

    /// Run stages 1-6 producing a ready-to-write SAM byte buffer.
    pub fn process_batch_serialized(
        &self,
        input: stage0_input::InputBatch,
        index: &Index,
        formatter: &SamFormatter,
        read_group: Option<&str>,
    ) -> Result<PipelineBatchOutput> {
        let mut sam_buf = Vec::new();
        let stats =
            self.process_batch_into(input, index, formatter, read_group, &mut sam_buf)?;
        Ok(PipelineBatchOutput { stats, sam_buf })
    }

    /// Run stages 1-5 and hand back the scored batch unserialised, for in-process
    /// consumers that want alignment records rather than SAM text.
    pub fn process_batch_scored(
        &self,
        input: stage0_input::InputBatch,
        index: &Index,
    ) -> Result<(stage5_scoring::ScoredBatch, PipelineBatchStats)> {
        let mut stages = [Duration::ZERO; 6];

        let t0 = Instant::now();
        // Stage 1 (AC + sketch) is SIMD-heavy → P-pool. Capture stats
        // before crossing the pool boundary so the install closure only
        // returns owned data.
        let (ac, sketch) = self.pool.install_compute(|| {
            let ac = ac_batch::run(
                &input.reads,
                index,
                self.config.alignment,
                self.config.max_alignments,
            );
            let skip_mask = ac.resolved_mask();
            let sketch = sketch_run_with_mask(input, self.config.sketch, &skip_mask);
            (ac, sketch)
        });
        let ac_stats = ac.stats;
        let sketch_stats = sketch.stats;
        stages[0] = t0.elapsed();

        // Stage 2 seeding — memory-bound hash lookups.
        let t1 = Instant::now();
        let seeds = self
            .pool
            .install_light(|| seed_run(sketch, index, self.config.seeding));
        let seed_stats = seeds.stats.clone();
        stages[1] = t1.elapsed();

        // Stage 3 chaining — scalar predecessor DP.
        let t2 = Instant::now();
        let chains = self
            .pool
            .install_light(|| chain_run(seeds, self.config.chaining));
        let chaining_stats = chains.stats.clone();
        stages[2] = t2.elapsed();

        // Read the current insert-size estimate before stage 4: the mate-guided
        // candidate search needs the concordance window to judge (R1 chain, R2
        // chain) combinations. The estimator is only *updated* after pairing, so
        // reading it here observes the same value stage 4 would have seen.
        let paired_cfg = self
            .insert_estimator
            .read()
            .map(|e| e.current())
            .unwrap_or(self.config.paired);

        let t3 = Instant::now();
        let ts4 = Instant::now();
        // Stage 4 alignment — banded SW / spectral / WFA, the hottest
        // SIMD path in the whole pipeline → P-pool.
        let mut align = self.pool.install_compute(|| {
            if self.config.splice.enabled {
                crate::alignment::splice::splice_align_batch(
                    chains,
                    index,
                    self.config.alignment,
                    self.config.splice,
                    self.junctions.as_deref(),
                    self.junc_bed_tolerance,
                )
            } else {
                align_run(
                    chains,
                    index,
                    AlignmentStageConfig {
                        cfg: self.config.alignment,
                        min_chain_ratio: self.config.min_chain_ratio,
                        accept_enable: self.config.accept_enable,
                        accept_only_top1: self.config.accept_only_top1,
                        accept_span_slack: self.config.accept_span_slack,
                        accept_min_identity: self.config.accept_min_identity,
                        accept_max_mismatches: self.config.accept_max_mismatches,
                        accept_require_score_margin: self.config.accept_require_score_margin,
                        dp_topk: self.config.dp_topk,
                        dp_abort_margin: self.config.dp_abort_margin,
                        debug_prefilter: self.config.debug_prefilter,
                        debug_prefilter_n: self.config.debug_prefilter_n,
                        debug_force_accept: self.config.debug_force_accept,
                        debug_force_accept_n: self.config.debug_force_accept_n,
                        long_read_threshold: self.config.long_read_threshold,
                        max_alignments: self.config.max_alignments,
                        short_preset: self.config.short_preset,
                        paired: paired_cfg,
                    },
                )
            }
        });
        merge_ac_alignments(&mut align.alignments, ac);
        let t_stage4 = ts4.elapsed();

        if std::env::var_os("KIRA_STATS").is_some() {
            crate::kira_info!("[KIRA_AC] reads={} eligible={} resolved={} ambiguous={} fwd_hits={} rev_hits={} build={:.2}ms scan={:.2}ms",
                ac_stats.n_reads,
                ac_stats.reads_eligible,
                ac_stats.reads_resolved,
                ac_stats.reads_ambiguous,
                ac_stats.fwd_hits,
                ac_stats.rev_hits,
                ac_stats.build_ms,
                ac_stats.scan_ms,
            );
        }

        let tru = Instant::now();
        if !self.config.skip_mate_rescue {
            let reads = &align.reads;
            let alns = &mut align.alignments;
            let pcfg = paired_cfg;
            let acfg = self.config.alignment;
            self.pool.install_compute(move || {
                rescue_unmapped_mates(reads, alns, index, &pcfg, acfg, RescueConfig::default())
            });
        }
        let t_rescue_unmapped = tru.elapsed();

        let tpr = Instant::now();
        if !self.config.skip_pairing {
            pair_rerank(
                &align.reads,
                &mut align.alignments,
                &paired_cfg,
                self.config.dp_topk.max(2),
            );
        }
        let t_pair_rerank = tpr.elapsed();

        let trd = Instant::now();
        if !self.config.skip_mate_rescue && !self.config.skip_pairing {
            let reads = &align.reads;
            let alns = &mut align.alignments;
            let pcfg = paired_cfg;
            let acfg = self.config.alignment;
            self.pool.install_compute(move || {
                rescue_discordant_pairs(reads, alns, index, &pcfg, acfg, RescueConfig::default())
            });
        }
        let t_rescue_discordant = trd.elapsed();

        // bwa-mem `-T`: the score floor applies to everything the stages above
        // produced, rescued placements included, and runs before pairing so
        // the mate's RNEXT/PNEXT never point at a record that is not emitted.
        if self.config.min_output_score > i32::MIN {
            apply_min_output_score(&mut align.alignments, self.config.min_output_score);
        }
        // bwa-mem `-5`: choose the primary segment of a split read by read
        // coordinate, also before pairing for the same reason.
        if self.config.primary_5p {
            apply_primary_5p(&align.reads, &mut align.alignments);
        }
        // ALT contigs: prefer an equivalent primary-assembly placement over an
        // ALT one (bwa-postalt), before pairing sees the coordinates.
        if let Some(mask) = self.config.alt_mask {
            apply_alt_primary_policy(&align.reads, &mut align.alignments, mask, self.config.alignment.mismatch);
        }

        // Indel left-normalization: canonicalise gap placement so equivalent
        // alignments of the same variant emit identical CIGARs (GATK
        // LeftAlignIndels / bcftools norm convention). Runs before pairing so
        // TLEN/proper-pair geometry sees final coordinates — which in fact
        // never move: normalization only slides indels within the aligned
        // span. NM/MD are recomputed for the records that moved.
        {
            let reads = &align.reads;
            let alns = &mut align.alignments;
            let index = &index;
            self.pool.install_compute(move || {
                crate::alignment::normalize::normalize_alignments(reads, alns, |ref_id| {
                    index.ref_bases(ref_id as usize)
                })
            });
        }

        let tap = Instant::now();
        apply_pairing(
            &align.reads,
            &mut align.alignments,
            &mut align.unmapped_mate_info,
            &paired_cfg,
        );
        let t_apply_pairing = tap.elapsed();

        let refined_after = if paired_cfg.is_paired() {
            self.insert_estimator
                .write()
                .ok()
                .and_then(|mut e| e.observe_batch(&align.alignments))
        } else {
            None
        };
        let paired_cfg_final = refined_after.unwrap_or(paired_cfg);

        stages[3] = t3.elapsed();
        if std::env::var_os("KIRA_STATS").is_some() {
            crate::kira_info!("[KIRA_STAGE3_BREAKDOWN] stage4_align={:.3}ms rescue_unmapped={:.3}ms pair_rerank={:.3}ms rescue_discordant={:.3}ms apply_pairing={:.3}ms total_s3={:.3}ms",
                t_stage4.as_secs_f64() * 1000.0,
                t_rescue_unmapped.as_secs_f64() * 1000.0,
                t_pair_rerank.as_secs_f64() * 1000.0,
                t_rescue_discordant.as_secs_f64() * 1000.0,
                t_apply_pairing.as_secs_f64() * 1000.0,
                stages[3].as_secs_f64() * 1000.0,
            );
        }

        let t4 = Instant::now();
        let pair_ctx = if paired_cfg_final.is_paired() {
            Some(crate::mapq::PairMapqContext {
                insert_mean: paired_cfg_final.insert_mean,
                insert_sd: paired_cfg_final.insert_sd,
                discordant_cap: 10,
            })
        } else {
            None
        };
        // Stage 5 MAPQ — light scalar math. The primary is kept as chosen
        // when an earlier policy picked it on grounds other than score.
        let preserve_primary = self.config.primary_5p || self.config.alt_mask.is_some();
        let scored = self.pool.install_light(|| {
            stage5_scoring::run_with_primary_policy(
                align,
                self.config.mapq,
                pair_ctx,
                preserve_primary,
            )
        });
        let align_stats = scored.stats.clone();
        stages[4] = t4.elapsed();

        Ok((
            scored,
            PipelineBatchStats {
                times: PipelineStageTimes { stages },
                align: align_stats,
                sketch: sketch_stats,
                seed: seed_stats,
                chaining: chaining_stats,
            },
        ))
    }

    /// [`Self::process_batch_serialized`] writing into a caller-owned SAM buffer,
    /// so the driver can recycle one buffer across batches.
    pub fn process_batch_into(
        &self,
        input: stage0_input::InputBatch,
        index: &Index,
        formatter: &SamFormatter,
        read_group: Option<&str>,
        sam_buf: &mut Vec<u8>,
    ) -> Result<PipelineBatchStats> {
        let (scored, mut stats) = self.process_batch_scored(input, index)?;

        // Stage 6 SAM emit — pure string formatting.
        let t5 = Instant::now();
        output_serialize_into(
            scored,
            formatter,
            read_group,
            self.config.output,
            self.config.max_alignments,
            sam_buf,
        );
        stats.times.stages[5] = t5.elapsed();
        Ok(stats)
    }
}

/// bwa-mem `-T`: drop alignments below `min_score`. If the primary (slot 0)
/// falls under the floor the read becomes unmapped — a secondary or
/// supplementary segment never gets promoted past a rejected primary.
fn apply_min_output_score(alignments: &mut [Vec<crate::types::Alignment>], min_score: i32) {
    for alns in alignments.iter_mut() {
        if alns.first().is_some_and(|a| a.score < min_score) {
            alns.clear();
        } else {
            alns.retain(|a| a.score >= min_score);
        }
    }
}

/// bwa-mem `-5`: among the primary and its supplementary segments, the one
/// covering the smallest read coordinate (in original read orientation)
/// becomes the primary. Secondaries are untouched.
fn apply_primary_5p(
    reads: &[crate::types::ReadRecord],
    alignments: &mut [Vec<crate::types::Alignment>],
) {
    for (alns, read) in alignments.iter_mut().zip(reads) {
        if alns.len() < 2 || !alns.iter().any(|a| a.is_supplementary) {
            continue;
        }
        let read_len = read.seq.len() as u32;
        let oriented_start = |a: &crate::types::Alignment| -> u32 {
            if a.is_rev {
                read_len.saturating_sub(a.read_end)
            } else {
                a.read_start
            }
        };
        let best = alns
            .iter()
            .enumerate()
            .filter(|(_, a)| !a.is_secondary)
            .min_by_key(|(i, a)| (oriented_start(a), *i))
            .map(|(i, _)| i);
        if let Some(i) = best
            && i != 0
        {
            alns.swap(0, i);
            alns[0].is_supplementary = false;
            alns[i].is_supplementary = true;
        }
    }
}

/// bwa-postalt lite: when the best hit is on an ALT contig and a
/// primary-assembly hit covering the same read region scores within one
/// mismatch of it, the primary-assembly hit becomes the primary record. The
/// ALT hit stays as a competitor (it still counts against an ALT primary in
/// the MAPQ model, and never against a primary-assembly one).
fn apply_alt_primary_policy(
    reads: &[crate::types::ReadRecord],
    alignments: &mut [Vec<crate::types::Alignment>],
    mask: &[bool],
    mismatch: i32,
) {
    let is_alt = |ref_id: u32| mask.get(ref_id as usize).copied().unwrap_or(false);
    for (alns, read) in alignments.iter_mut().zip(reads) {
        if alns.len() < 2 || !is_alt(alns[0].ref_id) {
            continue;
        }
        let read_len = read.seq.len() as u32;
        let interval = |a: &crate::types::Alignment| -> (u32, u32) {
            if a.is_rev {
                (read_len.saturating_sub(a.read_end), read_len.saturating_sub(a.read_start))
            } else {
                (a.read_start, a.read_end)
            }
        };
        let (ps, pe) = interval(&alns[0]);
        let floor = alns[0].score - mismatch.max(1);
        let pick = alns
            .iter()
            .enumerate()
            .skip(1)
            .filter(|(_, a)| !is_alt(a.ref_id) && !a.is_supplementary && a.score >= floor)
            .filter(|(_, a)| {
                let (s, e) = interval(a);
                let overlap = e.min(pe).saturating_sub(s.max(ps));
                overlap.saturating_mul(100) / (pe - ps).max(1) >= 50
            })
            .max_by_key(|(i, a)| (a.score, std::cmp::Reverse(*i)))
            .map(|(i, _)| i);
        if let Some(i) = pick {
            alns.swap(0, i);
            alns[0].is_secondary = false;
            alns[0].is_supplementary = false;
        }
    }
}

#[cfg(test)]
mod policy_tests {
    use super::{apply_alt_primary_policy, apply_min_output_score, apply_primary_5p};
    use crate::types::{Alignment, AlignmentKind, CigarKind, CigarOp, MateInfo, ReadRecord};

    fn aln(ref_id: u32, score: i32, read_start: u32, read_end: u32, is_rev: bool) -> Alignment {
        Alignment {
            kind: AlignmentKind::DpAligned,
            ref_id,
            ref_start: 1000,
            ref_end: 1000 + (read_end - read_start),
            read_start,
            read_end,
            cigar: vec![CigarOp {
                len: read_end - read_start,
                op: CigarKind::Match,
            }],
            score,
            mapq: 0,
            is_rev,
            is_secondary: false,
            is_supplementary: false,
            nm: 0,
            md: String::new(),
            as_score: score,
            xs_score: None,
            xs_strand: None,
            mate: MateInfo::default(),
        }
    }

    fn read(len: usize) -> ReadRecord {
        ReadRecord::new_unpaired("r".into(), vec![b'A'; len], None)
    }

    #[test]
    fn min_output_score_unmaps_read_when_primary_is_below_floor() {
        let mut alns = vec![vec![aln(0, 25, 0, 150, false), aln(1, 40, 0, 150, false)]];
        apply_min_output_score(&mut alns, 30);
        assert!(alns[0].is_empty(), "a rejected primary never promotes a runner-up");
    }

    #[test]
    fn min_output_score_prunes_only_low_secondaries_otherwise() {
        let mut alns = vec![vec![aln(0, 140, 0, 150, false), aln(1, 20, 0, 150, false)]];
        apply_min_output_score(&mut alns, 30);
        assert_eq!(alns[0].len(), 1);
        assert_eq!(alns[0][0].score, 140);
    }

    #[test]
    fn primary_5p_picks_smallest_read_coordinate_in_original_orientation() {
        // Primary covers read 80..150 on the reverse strand (original 0..70);
        // supplementary covers 0..70 forward (original 0..70 too? no: 0..70).
        // Make the supplementary the 5'-most in *original* coordinates.
        let mut primary = aln(0, 70, 80, 150, false); // original 80..150
        let mut supp = aln(1, 60, 0, 70, false); // original 0..70
        primary.is_supplementary = false;
        supp.is_supplementary = true;
        let mut alns = vec![vec![primary, supp]];
        apply_primary_5p(&[read(150)], &mut alns);
        assert_eq!(alns[0][0].ref_id, 1, "the 5'-most segment becomes primary");
        assert!(!alns[0][0].is_supplementary);
        assert!(alns[0][1].is_supplementary);
    }

    #[test]
    fn primary_5p_accounts_for_reverse_strand_segments() {
        // Reverse-strand segment at read 0..70 sits at original 80..150.
        let mut primary = aln(0, 70, 0, 70, true); // original 80..150
        let mut supp = aln(1, 60, 70, 150, false); // original 70..150 — 5'-most is 70
        primary.is_supplementary = false;
        supp.is_supplementary = true;
        let mut alns = vec![vec![primary, supp]];
        apply_primary_5p(&[read(150)], &mut alns);
        assert_eq!(alns[0][0].ref_id, 1);
    }

    #[test]
    fn alt_policy_moves_equivalent_primary_assembly_hit_to_primary() {
        let mask = &[false, true][..];
        // Best hit on the ALT contig (ref 1), primary-assembly hit one mismatch worse.
        let mut alns = vec![vec![aln(1, 150, 0, 150, false), aln(0, 146, 0, 150, false)]];
        apply_alt_primary_policy(&[read(150)], &mut alns, mask, 4);
        assert_eq!(alns[0][0].ref_id, 0);
        assert!(!alns[0][0].is_secondary);
        assert_eq!(alns[0][1].ref_id, 1);
    }

    #[test]
    fn alt_policy_keeps_alt_primary_when_primary_assembly_hit_is_clearly_worse() {
        let mask = &[false, true][..];
        let mut alns = vec![vec![aln(1, 150, 0, 150, false), aln(0, 130, 0, 150, false)]];
        apply_alt_primary_policy(&[read(150)], &mut alns, mask, 4);
        assert_eq!(alns[0][0].ref_id, 1, "a 5-mismatch gap is real divergence, not an ALT tie");
    }

    #[test]
    fn alt_policy_ignores_non_alt_primaries_and_disjoint_segments() {
        let mask = &[false, true][..];
        let mut alns = vec![vec![aln(0, 150, 0, 150, false), aln(1, 150, 0, 150, false)]];
        apply_alt_primary_policy(&[read(150)], &mut alns, mask, 4);
        assert_eq!(alns[0][0].ref_id, 0, "a primary-assembly primary is left alone");

        // ALT primary whose only primary-assembly hit covers a different read region.
        let mut alns = vec![vec![aln(1, 70, 0, 70, false), aln(0, 70, 80, 150, false)]];
        apply_alt_primary_policy(&[read(150)], &mut alns, mask, 4);
        assert_eq!(alns[0][0].ref_id, 1, "disjoint segments are not alternatives");
    }
}

/// Move AC-stage alignments into the cascade output. Reads resolved by AC had
/// their minimizers suppressed in stage 1, so their cascade slot is empty —
/// we overwrite it with the perfect-match alignment(s) produced by AC.
fn merge_ac_alignments(cascade: &mut [Vec<crate::types::Alignment>], mut ac: AcBatchOutput) {
    debug_assert_eq!(cascade.len(), ac.alignments.len());
    for (dst, src) in cascade.iter_mut().zip(ac.alignments.iter_mut()) {
        if !src.is_empty() {
            *dst = std::mem::take(src);
        }
    }
}
