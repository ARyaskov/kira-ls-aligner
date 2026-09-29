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
        "-t", "16", "-K", "100000000", "-Y", "-M", "-5", "-R", "@RG\\tID:x\\tSM:y",
        "-v", "1", "-w", "100", "-I", "350,50", "-A", "1", "-B", "4", "-O", "6,6",
        "-E", "1,1", "-L", "5,5", "-T", "30", "-h", "5,200", "-q", "-V", "-r", "1.5",
        "ref.fa", "r1.fq", "r2.fq",
    ])
    .expect("bwa mem command line must parse");
    assert_eq!(a.threads, 16);
    assert_eq!(a.band_width, Some(100));
    assert_eq!(a.window_len, None, "-w is the band width, not the minimizer window");
    assert_eq!(a.insert_size.as_deref(), Some("350,50"));
    assert_eq!(a.gap_open, 6);
    assert_eq!(a.gap_extend, 1);
    assert_eq!(a.clip_penalty, 5);
    assert_eq!(a.xa_max, 5);
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
    assert_eq!(parse(&["-x", "splice:hq", "ref.fa", "r.fq"]).unwrap().preset, "splice:hq");
    assert_eq!(parse(&["ref.fa", "r.fq"]).unwrap().preset, "auto");
}
