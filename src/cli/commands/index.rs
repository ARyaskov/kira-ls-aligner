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

    let cfg = IndexConfig {
        short_k,
        short_w,
        long_k,
        long_w,
        max_occ: args.max_occ,
    };

    let reference = read_reference(&args.reference)?;
    let index = Index::build(reference, cfg);

    let output = args.output.unwrap_or_else(|| {
        let mut p = args.reference.clone();
        p.set_extension("kiraidx");
        p
    });
    index.save(&output)?;
    eprintln!("[kira-ls-aligner] wrote index: {}", output.display());
    Ok(())
}
