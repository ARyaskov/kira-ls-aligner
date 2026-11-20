use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use tempfile::TempPath;

/// The per-destination directory serializes publishers and holds the previous
/// generation until its replacement is complete. Never remove a nonempty
/// directory on error: it may hold the only copy of the previous generation.
struct PublicationDirectory(PathBuf);

impl Drop for PublicationDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir(&self.0);
    }
}

fn move_existing(path: &Path, backup: &Path) -> Result<bool> {
    match path.symlink_metadata() {
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(e).with_context(|| format!("inspect {}", path.display())),
        Ok(meta) => ensure!(
            meta.is_file() || meta.is_symlink(),
            "cannot replace non-file {}",
            path.display()
        ),
    }
    fs::rename(path, backup).with_context(|| format!("back up {}", path.display()))?;
    Ok(true)
}

pub(super) fn publish_bam(bam: TempPath, bai: Option<TempPath>, destination: &Path) -> Result<()> {
    let index_path = kira_bam::io::default_index_path(destination, false);
    let mut dir = destination.as_os_str().to_owned();
    dir.push(".kira-publishing");
    let dir = PathBuf::from(dir);
    fs::create_dir(&dir).with_context(|| format!(
        "reserve BAM publication directory {}; an existing directory may belong to a running or interrupted publication",
        dir.display()
    ))?;
    let guard = PublicationDirectory(dir);
    let old_index = guard.0.join("previous.bai");
    let old_bam = guard.0.join("previous.bam");

    // Detach the old index first. A reader opening the new BAM must never
    // discover the offsets from the previous generation at its default path.
    let had_index = move_existing(&index_path, &old_index)?;
    let mut saved_bam = false;
    let mut installed_bam = false;
    let publish = (|| -> Result<()> {
        if bai.is_some() {
            saved_bam = move_existing(destination, &old_bam)?;
        }
        bam.persist(destination).context("publish BAM output")?;
        installed_bam = true;
        if let Some(index) = bai {
            index.persist(&index_path).context("publish BAI")?;
        }
        Ok(())
    })();
    if let Err(error) = publish {
        let rollback = (|| -> Result<()> {
            if saved_bam {
                fs::rename(&old_bam, destination)?;
            } else if installed_bam {
                fs::remove_file(destination)?;
            }
            if had_index {
                fs::rename(&old_index, &index_path)?;
            }
            Ok(())
        })();
        if let Err(rollback_error) = rollback {
            return Err(error).context(format!(
                "could not restore the previous output ({rollback_error}); recovery directory: {}",
                guard.0.display()
            ));
        }
        return Err(error);
    }
    if saved_bam {
        fs::remove_file(&old_bam).context("remove previous BAM backup")?;
    }
    if had_index {
        fs::remove_file(&old_index).context("remove previous BAI backup")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failure_publishing_index_restores_previous_pair() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let destination = dir.path().join("out.bam");
        let index = kira_bam::io::default_index_path(&destination, false);
        fs::write(&destination, b"old BAM")?;
        fs::write(&index, b"old index")?;
        let bam = tempfile::NamedTempFile::new_in(dir.path())?.into_temp_path();
        fs::write(&bam, b"new BAM")?;
        let bai = tempfile::NamedTempFile::new_in(dir.path())?.into_temp_path();
        fs::remove_file(&bai)?;
        assert!(publish_bam(bam, Some(bai), &destination).is_err());
        assert_eq!(fs::read(&destination)?, b"old BAM");
        assert_eq!(fs::read(&index)?, b"old index");
        assert_eq!(fs::read_dir(dir.path())?.count(), 2);
        Ok(())
    }
}
