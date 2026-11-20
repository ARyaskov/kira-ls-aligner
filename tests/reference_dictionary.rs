use kira_ls_aligner::{
    index::{Index, IndexConfig},
    io::read_reference,
    types::{RefBases, RefSeq, Reference},
};

#[test]
fn fasta_rejects_empty_and_duplicate_canonical_names() {
    let dir = tempfile::tempdir().unwrap();
    let fasta = dir.path().join("invalid.fa");
    for (input, message) in [
        (
            ">chr1 copy_A\nACGT\n>chr1 copy_B\nTGCA\n",
            "duplicate reference contig name \"chr1\"",
        ),
        (">\nACGT\n", "empty reference contig name"),
    ] {
        std::fs::write(&fasta, input).unwrap();
        let error = read_reference(&fasta).unwrap_err();
        assert!(error.to_string().contains(message), "{error:#}");
    }
}

#[test]
fn index_load_canonicalizes_descriptions_and_rejects_collisions() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("old.kiraidx");
    let reference = Reference {
        sequences: ["chr1_copy_A", "chr2_copy_B"]
            .into_iter()
            .map(|name| RefSeq {
                name: name.into(),
                bases: RefBases::Owned(b"ACGT".to_vec()),
            })
            .collect(),
    };
    let cfg = IndexConfig {
        build_short: false,
        build_long: false,
        ..IndexConfig::default()
    };
    Index::try_build(reference, cfg)
        .unwrap()
        .save(&path)
        .unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    let name_offset = |name: &[u8]| {
        bytes
            .windows(name.len())
            .position(|segment| segment == name)
            .unwrap()
    };
    let first = name_offset(b"chr1_copy_A");
    let second = name_offset(b"chr2_copy_B");
    bytes[first..first + 11].copy_from_slice(b"chr1 copy_A");
    bytes[second..second + 11].copy_from_slice(b"chr2 copy_B");
    std::fs::write(&path, &bytes).unwrap();
    let loaded = Index::load(&path).unwrap();
    assert_eq!(loaded.reference.sequences[0].name, "chr1");
    assert_eq!(loaded.reference.sequences[1].name, "chr2");
    drop(loaded);

    bytes[second + 3] = b'1';
    std::fs::write(&path, &bytes).unwrap();
    let error = Index::load(&path).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("duplicate reference contig name \"chr1\""),
        "{error:#}"
    );
}
