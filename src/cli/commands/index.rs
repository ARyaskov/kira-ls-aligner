use std::time::Instant;

use anyhow::Result;

use crate::cli::IndexArgs;
use crate::index::{Index, IndexConfig};
use crate::io::read_reference;

pub fn cmd_index(args: IndexArgs) -> Result<()> {
    let preset = args.preset.to_lowercase();
    let (mut short_k, mut short_w, mut long_k, mut long_w) = match preset.as_str() {
        "short" => (19, 10, 19, 10),
        "long" => (15, 10, 15, 10),
        _ => (19, 10, 15, 10),
    };
    if let Some(k) = args.seed_len {
        short_k = k;
        long_k = k;
    }
    if let Some(w) = args.window_len {
        short_w = w;
        long_w = w;
    }

    let (build_short, build_long) = match args.only.to_lowercase().as_str() {
        "short" => (true, false),
        "long" => (false, true),
        "both" | "" => (true, true),
        other => {
            anyhow::bail!("--only must be one of `both`, `short`, `long` (got `{other}`)");
        }
    };

    let cfg = IndexConfig {
        short_k,
        short_w,
        long_k,
        long_w,
        max_occ: args.max_occ,
        build_short,
        build_long,
    };

    let wall_start = Instant::now();
    eprintln!(
        "[KIRA_INDEX] reading reference {}...",
        args.reference.display()
    );
    let t = Instant::now();
    let reference = read_reference(&args.reference)?;
    let n_seqs = reference.sequences.len();
    let total_bp: usize = reference.sequences.iter().map(|s| s.len(None)).sum();
    eprintln!(
        "[KIRA_INDEX] read {} sequence(s), {:.2} Mbp in {:.2}s",
        n_seqs,
        total_bp as f64 / 1e6,
        t.elapsed().as_secs_f64()
    );

    let t = Instant::now();
    let index = Index::build(reference, cfg);
    eprintln!(
        "[KIRA_INDEX] Index::build done in {:.2}s",
        t.elapsed().as_secs_f64()
    );

    let output = args.output.unwrap_or_else(|| {
        let mut p = args.reference.clone();
        p.set_extension("kiraidx");
        p
    });
    let t = Instant::now();
    eprintln!("[KIRA_INDEX] writing index to {}...", output.display());
    index.save(&output)?;
    eprintln!(
        "[KIRA_INDEX] save done in {:.2}s (total wall time: {:.2}s)",
        t.elapsed().as_secs_f64(),
        wall_start.elapsed().as_secs_f64()
    );
    Ok(())
}
