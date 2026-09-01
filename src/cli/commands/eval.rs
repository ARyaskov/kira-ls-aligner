use std::fs::File;
use std::io::{stdin, BufReader, BufWriter, Write};

use anyhow::{Context, Result};

use crate::cli::EvalArgs;
use crate::eval::{evaluate, render_report, EvalConfig};

/// Run the `eval` subcommand: score a SAM file against truth-in-name read ids.
pub fn cmd_eval(args: EvalArgs) -> Result<()> {
    let mapq_thresholds: Vec<u8> = args
        .mapq_thresholds
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<u8>()
                .with_context(|| format!("invalid MAPQ threshold '{s}'"))
        })
        .collect::<Result<_>>()?;
    let cfg = EvalConfig {
        tolerance: args.tolerance,
        mapq_thresholds,
    };

    let reader: Box<dyn std::io::BufRead> = if args.sam.as_os_str() == "-" {
        Box::new(BufReader::new(stdin()))
    } else {
        Box::new(BufReader::new(File::open(&args.sam).with_context(
            || format!("open SAM file {}", args.sam.display()),
        )?))
    };

    let mut dump_writer: Option<BufWriter<File>> = match &args.dump_attribution {
        Some(path) => {
            Some(BufWriter::new(File::create(path).with_context(|| {
                format!("create attribution dump {}", path.display())
            })?))
        }
        None => None,
    };

    let counts = evaluate(reader, &cfg, dump_writer.as_mut())?;
    if let Some(w) = dump_writer.as_mut() {
        w.flush().context("flush attribution dump")?;
    }
    print!("{}", render_report(&counts));
    Ok(())
}
