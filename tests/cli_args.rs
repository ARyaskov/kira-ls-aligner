//! bwa-mem command-line compatibility: every documented flag parses with
//! bwa's letter and meaning, and an unmodified `bwa mem` invocation is
//! accepted.

use clap::Parser;
use kira_ls_aligner::cli::MemArgs;

fn parse(args: &[&str]) -> Result<MemArgs, clap::Error> {
    let mut argv = vec!["mem"];
    argv.extend_from_slice(args);
    MemArgs::try_parse_from(argv)
}

#[test]
fn unmodified_bwa_mem_command_line_parses() {
    let a = parse(&[
        "-t",
        "16",
        "-K",
        "100000000",
        "-Y",
        "-M",
        "-5",
        "-R",
        "@RG\\tID:x\\tSM:y",
        "-v",
        "1",
        "-w",
        "100",
        "-I",
        "350,50",
        "-A",
        "1",
        "-B",
        "4",
        "-O",
        "6,6",
        "-E",
        "1,1",
        "-L",
        "5,5",
        "-T",
        "30",
        "-h",
        "5,200",
        "-q",
        "-V",
        "-r",
        "1.5",
        "ref.fa",
        "r1.fq",
        "r2.fq",
    ])
    .expect("bwa mem command line must parse");
    assert_eq!(a.threads, 16);
    assert_eq!(a.band_width, Some(100));
    assert_eq!(
        a.window_len, None,
        "-w is the band width, not the minimizer window"
    );
    assert_eq!(a.insert_size.as_deref(), Some("350,50"));
    assert_eq!(a.gap_open, 6);
    assert_eq!(a.gap_extend, 1);
    assert_eq!(a.clip_penalty, 5);
    assert_eq!(a.xa_max, (5, 200));
    assert_eq!(a.verbosity, 1);
    assert_eq!(a.reads.len(), 2);
    let ignored = a.ignored_compat_flags();
    assert!(ignored.contains(&"-q") && ignored.contains(&"-V") && ignored.contains(&"-r"));
}

#[test]
fn verbosity_zero_is_accepted() {
    let a = parse(&["-v", "0", "ref.fa", "r.fq"]).unwrap();
    assert_eq!(a.verbosity, 0);
    assert!(parse(&["-v", "5", "ref.fa", "r.fq"]).is_err());
}

#[test]
fn minimizer_window_is_long_only() {
    let a = parse(&["--window-len", "7", "ref.fa", "r.fq"]).unwrap();
    assert_eq!(a.window_len, Some(7));
    assert_eq!(a.band_width, None);
}

#[test]
fn band_width_must_be_positive() {
    assert!(parse(&["-w", "0", "ref.fa", "r.fq"]).is_err());
}

#[test]
fn insert_window_keeps_the_legacy_form() {
    let a = parse(&["--insert-window", "100,600", "ref.fa", "r.fq"]).unwrap();
    assert_eq!(a.insert_window, "100,600");
    assert_eq!(a.insert_size, None);
}

#[test]
fn unknown_preset_is_rejected_and_bwa_long_read_presets_map_to_long() {
    assert!(parse(&["-x", "shrot", "ref.fa", "r.fq"]).is_err());
    for p in ["pacbio", "ont2d", "intractg", "LONG"] {
        let a = parse(&["-x", p, "ref.fa", "r.fq"]).unwrap();
        assert_eq!(a.preset, "long", "{p}");
    }
    assert_eq!(
        parse(&["-x", "splice:hq", "ref.fa", "r.fq"])
            .unwrap()
            .preset,
        "splice:hq"
    );
    assert_eq!(parse(&["ref.fa", "r.fq"]).unwrap().preset, "auto");
}

/// bwa-mem v0.7.17/master have no `-e`; it must be rejected like any other
/// unknown option, not accepted as a no-op.
#[test]
fn bwa_discard_exact_flag_is_rejected() {
    let err = parse(&["-e", "ref.fa", "r.fq"]).unwrap_err();
    assert_eq!(err.kind(), clap::error::ErrorKind::UnknownArgument);
}

/// `-c` is no longer an accepted-without-effect flag: it parses, carries its
/// value, and is dropped from the ignored-flags warning.
#[test]
fn bwa_max_occ_parses_and_is_not_ignored() {
    let a = parse(&["-c", "7", "ref.fa", "r.fq"]).unwrap();
    assert_eq!(a.compat_max_occ, Some(7));
    assert!(
        !a.ignored_compat_flags().contains(&"-c"),
        "-c must not be listed as accepted without effect"
    );
}

/// bwa-mem `-h INT[,INT]`: one INT sets BOTH caps; a pair sets the non-ALT
/// (first) and ALT (second) caps. Default mirrors upstream 5/200.
#[test]
fn xa_max_takes_one_or_two_values() {
    let a = parse(&["ref.fa", "r.fq"]).unwrap();
    assert_eq!(a.xa_max, (5, 200), "default mirrors bwa-mem -h 5,200");
    assert_eq!(
        parse(&["-h", "7", "ref.fa", "r.fq"]).unwrap().xa_max,
        (7, 7),
        "one INT must set both caps"
    );
    assert_eq!(
        parse(&["-h", "7,300", "ref.fa", "r.fq"]).unwrap().xa_max,
        (7, 300),
        "a pair sets first/second caps"
    );
}

#[test]
fn tiled_mode_rejects_unknown_output_format_before_opening_files() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_kira_ls_aligner"))
        .args([
            "mem",
            "--split-prefix",
            "chunks",
            "--emit",
            "pfa",
            "missing.fa",
            "missing.fq",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("unknown --emit value"), "{stderr}");
}
