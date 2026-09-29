//! Registry of the `KIRA_*` tuning knobs.
//!
//! Every environment variable the aligner reads at run time is listed here
//! with its default and a one-line description. The registry is what
//! `--set` / `--config` validate against (a misspelt knob used to be
//! accepted silently and do nothing), what `kira_ls_aligner knobs` prints,
//! and what a unit test checks the source tree against in both directions:
//! a knob read somewhere but missing here, or listed here but read nowhere,
//! fails the build's tests.
//!
//! Knobs are read once through a `OnceLock` at the site that uses them; the
//! registry does not parse values, it only names them.

/// One tuning knob.
#[derive(Clone, Copy, Debug)]
pub struct Knob {
    pub name: &'static str,
    /// Default as the user would write it; `unset` for pure switches that
    /// act by being present.
    pub default: &'static str,
    pub doc: &'static str,
}

const fn knob(name: &'static str, default: &'static str, doc: &'static str) -> Knob {
    Knob { name, default, doc }
}

/// All run-time knobs, grouped by stage.
pub const KNOBS: &[Knob] = &[
    // ── input / execution ─────────────────────────────────────────────
    knob(
        "KIRA_STATS",
        "unset",
        "Per-stage timings and counters on stderr.",
    ),
    knob(
        "KIRA_PREFETCH",
        "1",
        "Parse the next batch on a producer thread while the current one aligns; 0 = serial.",
    ),
    knob(
        "KIRA_BGZF_THREADS",
        "2",
        "Workers inflating a BGZF input, per input file; 1 = single-threaded backend, 0 = default.",
    ),
    knob(
        "KIRA_HYBRID_DISABLE",
        "unset",
        "Force the homogeneous thread pool on hybrid (P/E core) CPUs.",
    ),
    knob(
        "KIRA_AFFINITY_DEBUG",
        "unset",
        "Log the detected core topology (Windows).",
    ),
    knob(
        "KIRA_VNNI_DISABLE",
        "unset",
        "Disable the AVX-VNNI INT8 SW kernel and use the i16 AVX2 path.",
    ),
    knob(
        "KIRA_GPU_DIAG",
        "unset",
        "Compare GPU spectral results against the CPU path for the first reads of each batch.",
    ),
    knob(
        "KIRA_GPU_DIAG_N",
        "5",
        "Reads per batch KIRA_GPU_DIAG compares.",
    ),
    // ── index ─────────────────────────────────────────────────────────
    knob(
        "KIRA_INDEX_USE_MPH",
        "1",
        "Build the PtrHash minimal perfect hash for bucket lookup; 0 = sorted-array lookup.",
    ),
    knob(
        "KIRA_INDEX_TMPDIR",
        "system temp dir",
        "Directory for the external (disk-backed) index build's run files.",
    ),
    knob(
        "KIRA_INDEX_RAM_MB",
        "half of RAM",
        "RAM budget for index building, in MB.",
    ),
    knob(
        "KIRA_HOT_CACHE_N",
        "50000",
        "Most-queried minimizer buckets pre-materialised per index table; 0 disables.",
    ),
    knob(
        "KIRA_HOT_CACHE_MAX_OCCS",
        "8000000",
        "Total occurrences the hot-bucket cache may hold.",
    ),
    // ── seeding / chaining ────────────────────────────────────────────
    knob(
        "KIRA_K_HITS",
        "16",
        "Occurrences sampled per read minimizer; overrides --seed-occ-cap.",
    ),
    knob(
        "KIRA_ANCHOR_CAP_K",
        "1",
        "Never require an anchor longer than the seed length k; 0 restores min_anchor_len.",
    ),
    knob(
        "KIRA_MATE_SEED",
        "1",
        "Bias seed-occurrence sampling toward copies with the mate nearby; 0 disables.",
    ),
    knob(
        "KIRA_CHAIN_SCAN_MULT",
        "4",
        "Multiplier on rmq_window for the raw-candidate predecessor scan in chaining.",
    ),
    knob(
        "KIRA_CHAIN_MAX_DIST",
        "500",
        "Chaining max_dist for the in-process (fused) short-read profile.",
    ),
    knob(
        "KIRA_CHAIN_GAP_OPEN",
        "5",
        "Chaining gap-open cost for the in-process short-read profile.",
    ),
    knob(
        "KIRA_CHAIN_GAP_EXTEND",
        "1",
        "Chaining gap-extend cost for the in-process short-read profile.",
    ),
    knob(
        "KIRA_MIN_CHAIN_RATIO",
        "0.4",
        "Min chain score ratio vs best for the in-process profile (kira mem: --min-chain-ratio).",
    ),
    knob(
        "KIRA_CHAIN_QUALITY_GATE",
        "0",
        "Require KIRA_CHAIN_QUALITY_PCT anchor coverage before a chain tries a fast path.",
    ),
    knob(
        "KIRA_CHAIN_QUALITY_PCT",
        "70",
        "Anchor-coverage percentage for KIRA_CHAIN_QUALITY_GATE.",
    ),
    knob(
        "KIRA_AMBIG_DIV",
        "5",
        "A runner-up chain within best/DIV marks the read ambiguous and buys a competing DP placement.",
    ),
    knob(
        "KIRA_SHORT_DPTOPK",
        "2",
        "Chains kept per short read for DP and pair promotion; overrides --dp-topk.",
    ),
    knob(
        "KIRA_MATE_GUIDE",
        "1",
        "Promote the candidate locus with a plausible mate partner when exactly one has one; 0 disables.",
    ),
    knob(
        "KIRA_TWOTIER",
        "1",
        "Rank ambiguous candidate loci by bounded Myers edit cost instead of chain score.",
    ),
    knob(
        "KIRA_TWOTIER_K",
        "5",
        "Candidate loci the two-tier search ranks per read.",
    ),
    knob(
        "KIRA_TWOTIER_MAPQ",
        "0",
        "Feed the two-tier Myers cost gap into MAPQ as the XS proxy.",
    ),
    knob(
        "KIRA_XS_MINRATIO",
        "0",
        "Floor on the chain-derived XS proxy, as a percentage of AS; 0 = off.",
    ),
    // ── alignment ─────────────────────────────────────────────────────
    knob(
        "KIRA_ALGO",
        "packed",
        "Fast-path aligner: packed (bit-parallel SWAR), spectral, wfa, sw.",
    ),
    knob(
        "KIRA_GAP_OPEN",
        "6",
        "Gap-open penalty for the in-process short-read profile (kira mem: -O).",
    ),
    knob(
        "KIRA_GAP_EXTEND",
        "1",
        "Gap-extend penalty for the in-process short-read profile (kira mem: -E).",
    ),
    knob(
        "KIRA_CLIP_PENALTY",
        "5",
        "Soft-clip penalty for the in-process short-read profile (kira mem: -L).",
    ),
    knob(
        "KIRA_BANDWIDTH",
        "50",
        "DP band width for the in-process short-read profile (kira mem: -w).",
    ),
    knob(
        "KIRA_XDROP",
        "50",
        "X-drop for the in-process short-read profile (kira mem: -d).",
    ),
    knob(
        "KIRA_DP_XDROP",
        "unset",
        "X-drop for the banded DP alone, decoupled from the prefilter's ungapped extension.",
    ),
    knob(
        "KIRA_ADAPTIVE_BAND",
        "1",
        "Re-centre the DP band on the previous row's running maximum; 0 pins it to the seed diagonal.",
    ),
    knob(
        "KIRA_ADAPTIVE_MAX_DRIFT",
        "16",
        "Max drift of the adaptive band centre from the seed diagonal, in bases.",
    ),
    knob(
        "KIRA_CERT_STRICT",
        "0",
        "Stricter spectral certification: also test 2-base shifts so disguised 1-2 bp indels go to WFA.",
    ),
    knob(
        "KIRA_UNGAP_TAIL",
        "1",
        "Check the read tail during the ungapped accept; 0 disables.",
    ),
    knob(
        "KIRA_UNGAP_MINID",
        "unset",
        "Override the ungapped-accept identity floor, x10000 (9850 = 98.5 %).",
    ),
    knob(
        "KIRA_UNGAP_GAP_CHECK",
        "0",
        "Gate the ungapped accept with a WFA score-only probe (catches disguised indels; +45 % align time).",
    ),
    knob(
        "KIRA_WFA_MAX_LEN",
        "300",
        "Longest read the WFA fast path takes.",
    ),
    knob(
        "KIRA_WFA_BUDGET_PCT",
        "25",
        "WFA error budget as a percentage of read length.",
    ),
    knob(
        "KIRA_WFA_ENDS_FREE",
        "0",
        "Ends-free allowance for WFA, in bases.",
    ),
    knob(
        "KIRA_WFA_LEAD",
        "0",
        "Leading reference text before the seed for WFA, in bases.",
    ),
    knob(
        "KIRA_WFA_ADAPTIVE",
        "unset",
        "d > 0 bounds the WFA2 wavefront to ~2d (near-linear, inexact).",
    ),
    knob(
        "KIRA_MYERS_BOUND_PCT",
        "15",
        "Myers reject bound: read_len * pct / 100 + KIRA_MYERS_BOUND_FLOOR.",
    ),
    knob(
        "KIRA_MYERS_BOUND_FLOOR",
        "4",
        "Additive floor of the Myers reject bound.",
    ),
    knob(
        "KIRA_SPECTRAL_MISM_PCT",
        "3",
        "Spectral sieve mismatch budget: read_len * pct / 100 + KIRA_SPECTRAL_MISM_FLOOR.",
    ),
    knob(
        "KIRA_SPECTRAL_MISM_FLOOR",
        "2",
        "Additive floor of the spectral mismatch budget.",
    ),
    knob(
        "KIRA_AC_DISABLE",
        "unset",
        "Aho-Corasick exact-match fast path: unset = auto, 0 = force on, 1 = force off.",
    ),
    knob(
        "KIRA_AC_MAX_REF_MB",
        "50",
        "Largest reference, in Mbp, for which the Aho-Corasick fast path is built.",
    ),
    knob(
        "KIRA_CGK_ENABLE",
        "0",
        "Enable the CGK-embedding rescue fallback.",
    ),
    knob("KIRA_LSH_ENABLE", "0", "Enable the LSH rescue fallback."),
    knob(
        "KIRA_LEFT_NORM",
        "1",
        "Left-normalize indel placement in emitted CIGARs; 0/off emits traceback-native CIGARs.",
    ),
    // ── pairing / rescue ──────────────────────────────────────────────
    knob(
        "KIRA_PAIR_PROMOTE",
        "1",
        "Promote the best concordant pair before rescue and mate-field assignment; 0 disables.",
    ),
    knob("KIRA_PAIR_BONUS", "30", "Concordant joint-score bonus."),
    knob(
        "KIRA_PAIR_BONUS_ALL",
        "0",
        "Give the concordance bonus to every concordant pair, not only the winner (ties -> MAPQ 0).",
    ),
    knob(
        "KIRA_PAIR_MAPQ",
        "0",
        "MAPQ floor applied to a promoted concordant primary; 0 = no boost.",
    ),
    knob(
        "KIRA_RESCUE_WIDE",
        "0",
        "After a banded mate-rescue attempt misses its bar, also run a full-window SW.",
    ),
    knob(
        "KIRA_RESCUE_NO_FAST",
        "unset",
        "Force mate rescue through the wide banded-SW path (skip the packed scan).",
    ),
    // ── MAPQ ──────────────────────────────────────────────────────────
    knob(
        "KIRA_MAPQ_BETA",
        "22.5",
        "Slope of the MAPQ score-gap posterior model.",
    ),
    knob(
        "KIRA_MAPQ_MULT",
        "0",
        "Multiplicity penalty slope on competing loci (bwa-mem uses 6.585); 0 = off.",
    ),
    knob(
        "KIRA_REPEAT_MAPQ",
        "1",
        "Cap MAPQ by seed repetitiveness; 0 disables.",
    ),
    knob(
        "KIRA_ID_MAPQ",
        "binary",
        "Identity-based MAPQ ceiling: 0/off, ramp (graded), anything else = binary.",
    ),
    knob(
        "KIRA_RESCUE_MAPQ_CAP",
        "30",
        "MAPQ ceiling for mate-rescue placements; 60 disables the cap.",
    ),
];

/// Environment variables that start with `KIRA_` but are not run-time knobs
/// (build-time inputs of `build.rs`, read with `env!`).
pub const NOT_KNOBS: &[&str] = &["KIRA_SPECTRAL_PTX", "KIRA_SPECTRAL_PTX_STUB"];

/// The registry entry for `name`, if it is a knob.
pub fn lookup(name: &str) -> Option<&'static Knob> {
    KNOBS.iter().find(|k| k.name == name)
}

pub fn is_known(name: &str) -> bool {
    lookup(name).is_some()
}

/// The registered knob closest to `name` (edit distance at most a third of
/// the name), for "did you mean" messages.
pub fn suggest(name: &str) -> Option<&'static str> {
    let upper = name.to_ascii_uppercase();
    let limit = (upper.len() / 3).max(2);
    KNOBS
        .iter()
        .map(|k| (levenshtein(&upper, k.name), k.name))
        .filter(|(d, _)| *d <= limit)
        .min_by_key(|(d, _)| *d)
        .map(|(_, n)| n)
}

/// `KIRA_*` variables set in the process environment that are not knobs.
pub fn unknown_in_env() -> Vec<String> {
    let mut v: Vec<String> = std::env::vars_os()
        .filter_map(|(k, _)| k.into_string().ok())
        .filter(|k| k.starts_with("KIRA_") && !is_known(k) && !NOT_KNOBS.contains(&k.as_str()))
        .collect();
    v.sort();
    v
}

/// The registry as an aligned text table.
pub fn render_table() -> String {
    let name_w = KNOBS.iter().map(|k| k.name.len()).max().unwrap_or(0);
    let def_w = KNOBS.iter().map(|k| k.default.len()).max().unwrap_or(0);
    let mut out = String::new();
    for k in KNOBS {
        out.push_str(&format!(
            "{:<name_w$}  {:<def_w$}  {}\n",
            k.name, k.default, k.doc
        ));
    }
    out
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ca != cb);
            cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::path::Path;

    /// Every `"KIRA_…"` string literal under `src/` (a knob name is always a
    /// quoted literal at its read site).
    fn literals_in_source() -> BTreeSet<String> {
        fn walk(dir: &Path, out: &mut BTreeSet<String>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    walk(&path, out);
                    continue;
                }
                if path.extension().is_none_or(|e| e != "rs") || path.ends_with("knobs.rs") {
                    continue;
                }
                let text = std::fs::read_to_string(&path).unwrap();
                let mut rest = text.as_str();
                while let Some(i) = rest.find("\"KIRA_") {
                    let name_start = &rest[i + 1..];
                    let end = name_start
                        .find(|c: char| !(c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'))
                        .unwrap_or(name_start.len());
                    // A bare "KIRA_" prefix (starts_with filters) is not a name.
                    if end > "KIRA_".len() && name_start[end..].starts_with('"') {
                        out.insert(name_start[..end].to_string());
                    }
                    rest = &rest[i + 1..];
                }
            }
        }
        let mut out = BTreeSet::new();
        walk(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src"), &mut out);
        out
    }

    #[test]
    fn registry_matches_the_source_tree() {
        let in_source: BTreeSet<String> = literals_in_source()
            .into_iter()
            .filter(|n| !NOT_KNOBS.contains(&n.as_str()))
            .collect();
        let registered: BTreeSet<String> = KNOBS.iter().map(|k| k.name.to_string()).collect();
        let unregistered: Vec<_> = in_source.difference(&registered).collect();
        let unused: Vec<_> = registered.difference(&in_source).collect();
        assert!(
            unregistered.is_empty(),
            "knobs read in src/ but missing from knobs::KNOBS: {unregistered:?}"
        );
        assert!(
            unused.is_empty(),
            "knobs in knobs::KNOBS that nothing reads: {unused:?}"
        );
    }

    #[test]
    fn names_are_unique_and_documented() {
        let mut seen = BTreeSet::new();
        for k in KNOBS {
            assert!(seen.insert(k.name), "duplicate knob {}", k.name);
            assert!(k.name.starts_with("KIRA_"));
            assert!(!k.doc.is_empty() && !k.default.is_empty(), "{}", k.name);
        }
    }

    #[test]
    fn suggestions_catch_typos() {
        assert_eq!(suggest("KIRA_BANDWITH"), Some("KIRA_BANDWIDTH"));
        assert_eq!(suggest("kira_left_norm"), Some("KIRA_LEFT_NORM"));
        assert_eq!(suggest("KIRA_SOMETHING_ELSE_ENTIRELY"), None);
    }
}
