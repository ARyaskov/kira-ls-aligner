use std::collections::BTreeMap;

use kira_ls_aligner::index::{Index, IndexConfig, MinimizerIndex};
use kira_ls_aligner::sketch::{MinimizerConfig, minimizers};
use kira_ls_aligner::types::{RefBases, RefSeq, Reference, Strand};

type Occurrence = (u32, u32, u8);

fn reference() -> Reference {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut sequences = Vec::new();
    for id in 0..2 {
        let mut bases: Vec<u8> = (0..4096)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                b"ACGT"[((state >> 32) & 3) as usize]
            })
            .collect();
        bases.extend_from_slice(b"NNNACGTAACGTAACGTAACGTAACGTAACGTAACGTAACGTA");
        sequences.push(RefSeq {
            name: format!("synthetic{id}"),
            bases: RefBases::Owned(bases),
        });
    }
    Reference { sequences }
}

fn expected(reference: &Reference, cfg: MinimizerConfig) -> BTreeMap<u64, Vec<Occurrence>> {
    let mut groups: BTreeMap<u64, Vec<Occurrence>> = BTreeMap::new();
    for (id, seq) in reference.sequences.iter().enumerate() {
        for minimizer in minimizers(seq.bases(None), &cfg) {
            groups.entry(minimizer.hash).or_default().push((
                id as u32,
                minimizer.pos,
                u8::from(minimizer.strand == Strand::Reverse),
            ));
        }
    }
    for occurrences in groups.values_mut() {
        occurrences.sort_unstable_by_key(|&(id, pos, reverse)| (reverse, id, pos));
    }
    groups
}

fn assert_known_buckets(
    index: &Index,
    table: &MinimizerIndex,
    groups: &BTreeMap<u64, Vec<Occurrence>>,
) {
    for (&hash, all) in groups {
        assert_eq!(index.bucket_count(table, hash), Some(all.len() as u32));
        let retained = &all[..all.len().min(table.max_occ.saturating_add(1))];
        let mut actual = Vec::new();
        index.for_each_occ(table, hash, |occ| {
            actual.push((occ.ref_id, occ.pos, u8::from(occ.strand == Strand::Reverse)));
        });
        assert_eq!(
            actual, retained,
            "wrong occurrences for known minimizer {hash}"
        );
        assert_eq!(index.bucket_len(table, hash), Some(retained.len()));
    }
}

fn assert_batch_reuse(index: &Index, table: &MinimizerIndex, known: &[u64]) {
    assert!(known.len() >= 129);
    let mut queries = known.to_vec();
    queries.extend((0..129).map(|i| u64::MAX - i));
    let n = queries.len();
    let mut canonical = vec![u64::MAX; n + 3];
    let mut ids = vec![Some(usize::MAX); n + 3];
    let mut buckets = vec![Some((usize::MAX, usize::MAX)); n + 3];
    for count in [n, 129, 65, 64, 63, 16, 15, 1, 0, n] {
        let queries = &queries[..count];
        let scalar: Vec<_> = queries
            .iter()
            .map(|&hash| index.bucket(table, hash))
            .collect();
        let canonical_suffix = canonical[count..].to_vec();
        let ids_suffix = ids[count..].to_vec();
        let buckets_suffix = buckets[count..].to_vec();
        index.bucket_batch_into(table, queries, &mut canonical, &mut ids, &mut buckets);
        assert_eq!(&buckets[..count], scalar, "batch length {count}");
        assert_eq!(&canonical[count..], canonical_suffix);
        assert_eq!(&ids[count..], ids_suffix);
        assert_eq!(&buckets[count..], buckets_suffix);
        assert_eq!(&canonical[n..], &[u64::MAX; 3]);
        assert_eq!(&ids[n..], &[Some(usize::MAX); 3]);
        assert_eq!(&buckets[n..], &[Some((usize::MAX, usize::MAX)); 3]);
    }
}

#[test]
fn upgraded_numeric_lookup_preserves_occurrence_oracle_across_build_clone_and_mmap() {
    let reference = reference();
    let cfg = MinimizerConfig { k: 7, w: 3 };
    let groups = expected(&reference, cfg);
    assert!(groups.values().any(|group| group.len() > 4));
    let known: Vec<_> = groups.keys().copied().collect();
    let directory = tempfile::tempdir().unwrap();
    for budget in [usize::MAX, 1] {
        let mut index = Index::build(
            reference.clone(),
            IndexConfig {
                short_k: cfg.k,
                short_w: cfg.w,
                long_k: cfg.k,
                long_w: cfg.w,
                max_occ: 3,
                build_short: false,
                build_long: false,
            },
        );
        index.short = MinimizerIndex::build_with_budget(&reference, cfg.k, cfg.w, 3, budget);
        assert_known_buckets(&index, &index.short, &groups);
        assert_batch_reuse(&index, &index.short, &known);
        let cloned = index.clone();
        assert_known_buckets(&cloned, &cloned.short, &groups);
        let path = directory.path().join(format!("{budget}.kiraidx"));
        let rewritten = directory.path().join(format!("{budget}.rewritten.kiraidx"));
        index.save(&path).unwrap();
        let mapped = Index::load(&path).unwrap();
        assert_known_buckets(&mapped, &mapped.short, &groups);
        assert_batch_reuse(&mapped, &mapped.short, &known);
        mapped.save(&rewritten).unwrap();
        assert_eq!(
            std::fs::read(&path).unwrap(),
            std::fs::read(&rewritten).unwrap()
        );
        let mapped_clone = mapped.clone();
        drop(mapped);
        drop(index);
        assert_known_buckets(&mapped_clone, &mapped_clone.short, &groups);
        assert_batch_reuse(&mapped_clone, &mapped_clone.short, &known);
    }
}

#[test]
fn skipped_tables_overwrite_reused_results_without_touching_output_suffix() {
    let index = Index::build(
        reference(),
        IndexConfig {
            build_short: false,
            build_long: false,
            ..IndexConfig::default()
        },
    );
    let queries = [0, 1, u64::MAX];
    let mut canonical = [17; 4];
    let mut ids = [Some(23); 4];
    let mut buckets = [Some((29, 31)); 4];
    for table in [&index.short, &index.long] {
        index.bucket_batch_into(table, &queries, &mut canonical, &mut ids, &mut buckets);
        assert_eq!(buckets, [None, None, None, Some((29, 31))]);
        assert_eq!(canonical[3], 17);
        assert_eq!(ids[3], Some(23));
        for &hash in &queries {
            assert_eq!(index.bucket(table, hash), None);
            assert_eq!(index.bucket_count(table, hash), None);
        }
    }
}

#[test]
fn frozen_v081_mph_index_keeps_reference_counts_slots_and_bytes() {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/kv_engine_v081.kiraidx");
    let loaded = Index::load(&fixture).unwrap();
    let original = reference();
    assert_eq!(loaded.reference.sequences.len(), original.sequences.len());
    for (id, sequence) in original.sequences.iter().enumerate() {
        assert_eq!(loaded.ref_bases(id), sequence.bases(None));
        assert_eq!(loaded.reference.sequences[id].name, sequence.name);
    }
    assert_eq!(
        (loaded.short.k, loaded.short.w, loaded.short.max_occ),
        (7, 3, 3)
    );
    let groups = expected(&original, MinimizerConfig { k: 7, w: 3 });
    assert_known_buckets(&loaded, &loaded.short, &groups);
    let known: Vec<_> = groups.keys().copied().collect();
    assert_batch_reuse(&loaded, &loaded.short, &known);
    let directory = tempfile::tempdir().unwrap();
    let rewritten = directory.path().join("rewritten.kiraidx");
    loaded.save(&rewritten).unwrap();
    assert_eq!(
        std::fs::read(&fixture).unwrap(),
        std::fs::read(&rewritten).unwrap()
    );
    let clone = loaded.clone();
    drop(loaded);
    assert_known_buckets(&clone, &clone.short, &groups);
}

#[test]
fn interrupted_serialization_preserves_existing_destination_and_removes_temporary_file() {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/kv_engine_v081.kiraidx");
    let directory = tempfile::tempdir().unwrap();
    let destination = directory.path().join("preserved.kiraidx");
    std::fs::copy(&fixture, &destination).unwrap();
    let original_bytes = std::fs::read(&destination).unwrap();
    let mut invalid = Index::load(&fixture).unwrap();
    invalid.reference.sequences[0].bases = RefBases::Mmap {
        offset: invalid.mmap.as_ref().unwrap().len() + 1,
        len: 1,
    };
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| invalid.save(&destination)));
    assert!(result.is_err(), "fault must interrupt serialization");
    assert_eq!(std::fs::read(&destination).unwrap(), original_bytes);
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
    let preserved = Index::load(&destination).unwrap();
    let groups = expected(&reference(), MinimizerConfig { k: 7, w: 3 });
    assert_known_buckets(&preserved, &preserved.short, &groups);
}

#[test]
fn saving_over_a_live_mapped_source_preserves_the_mapping_and_file() {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/kv_engine_v081.kiraidx");
    let directory = tempfile::tempdir().unwrap();
    let destination = directory.path().join("mapped.kiraidx");
    std::fs::copy(&fixture, &destination).unwrap();
    let original_bytes = std::fs::read(&destination).unwrap();
    let mapped = Index::load(&destination).unwrap();
    if let Err(error) = mapped.save(&destination) {
        #[cfg(windows)]
        {
            let io = error.root_cause().downcast_ref::<std::io::Error>().unwrap();
            assert_eq!(
                io.kind(),
                std::io::ErrorKind::PermissionDenied,
                "unexpected mapped replacement error: {error:#}"
            );
        }
        #[cfg(not(windows))]
        panic!("mapped atomic replacement failed: {error:#}");
    }
    assert_eq!(std::fs::read(&destination).unwrap(), original_bytes);
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
    let groups = expected(&reference(), MinimizerConfig { k: 7, w: 3 });
    assert_known_buckets(&mapped, &mapped.short, &groups);
    let reloaded = Index::load(&destination).unwrap();
    assert_known_buckets(&reloaded, &reloaded.short, &groups);
}

#[cfg(unix)]
#[test]
fn atomic_save_respects_creation_umask_and_preserves_existing_permissions() {
    use std::os::unix::fs::PermissionsExt;

    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/kv_engine_v081.kiraidx");
    let index = Index::load(&fixture).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let control = directory.path().join("ordinary-create");
    std::fs::File::create(&control).unwrap();
    let ordinary_mode = std::fs::metadata(&control).unwrap().permissions().mode() & 0o777;
    let destination = directory.path().join("permissions.kiraidx");
    index.save(&destination).unwrap();
    let created_mode = std::fs::metadata(&destination)
        .unwrap()
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(created_mode, ordinary_mode);
    std::fs::set_permissions(&destination, std::fs::Permissions::from_mode(0o640)).unwrap();
    index.save(&destination).unwrap();
    let replaced_mode = std::fs::metadata(&destination)
        .unwrap()
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(replaced_mode, 0o640);
    assert_eq!(
        std::fs::read(&fixture).unwrap(),
        std::fs::read(&destination).unwrap()
    );
}
