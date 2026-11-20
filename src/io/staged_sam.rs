use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tempfile::TempPath;

use super::SamWriter;
use crate::types::Reference;

/// Keep a named SAM/PAF private until all input, workers and output have finished.
pub(crate) enum SamOutput {
    Stream,
    File {
        staged: TempPath,
        destination: PathBuf,
    },
}

impl SamOutput {
    pub(crate) fn open(path: Option<&Path>, reference: &Reference) -> Result<(SamWriter, Self)> {
        let Some(path) = path else {
            return Ok((SamWriter::new(None::<&Path>, reference)?, Self::Stream));
        };
        // Follow links as File::create does, including links to devices and
        // valid links whose final output file has not been created yet.
        let mut destination = path.to_owned();
        while destination.is_symlink() {
            match destination.canonicalize() {
                Ok(resolved) => {
                    destination = resolved;
                    break;
                }
                Err(e) if e.kind() == ErrorKind::NotFound => {
                    let target = destination.read_link().context("read SAM output symlink")?;
                    destination.pop();
                    destination.push(target);
                }
                Err(e) => return Err(e).context("resolve SAM output symlink"),
            }
        }
        let metadata = match destination.metadata() {
            Ok(meta) if meta.is_file() => {
                // Check the same access as a direct writer, without truncation.
                std::fs::OpenOptions::new()
                    .write(true)
                    .open(&destination)
                    .context("open existing SAM output")?;
                Some(meta)
            }
            Ok(_) => return Ok((SamWriter::new(Some(&destination), reference)?, Self::Stream)),
            Err(e) if e.kind() == ErrorKind::NotFound => None,
            Err(e) => return Err(e).context("inspect SAM output"),
        };
        let directory = destination
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let mut builder = tempfile::Builder::new();
        builder.prefix(".kira-text-");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            builder.permissions(std::fs::Permissions::from_mode(0o666));
        }
        let staged = builder
            .tempfile_in(directory)
            .context("create temporary SAM output")?;
        if let Some(meta) = metadata {
            let permissions = meta.permissions();
            #[cfg(unix)]
            let permissions = {
                use std::os::unix::fs::PermissionsExt;
                std::fs::Permissions::from_mode(permissions.mode() & 0o777)
            };
            staged
                .as_file()
                .set_permissions(permissions)
                .context("preserve SAM output permissions")?;
        }
        let (file, staged) = staged.into_parts();
        Ok((
            SamWriter::from_writer(Box::new(file), reference),
            Self::File {
                staged,
                destination,
            },
        ))
    }

    pub(crate) fn progress_path(&self) -> Option<&Path> {
        match self {
            Self::File { staged, .. } => Some(staged),
            Self::Stream => None,
        }
    }

    pub(crate) fn publish(self) -> Result<()> {
        if let Self::File {
            staged,
            destination,
        } = self
        {
            staged
                .persist(&destination)
                .with_context(|| format!("publish {}", destination.display()))?;
        }
        Ok(())
    }
}
