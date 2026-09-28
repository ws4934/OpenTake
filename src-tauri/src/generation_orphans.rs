//! Paid generation jobs whose project was closed before their state could be
//! written to it.
//!
//! A provider can accept a job after the project that submitted it was
//! replaced, and a synchronous provider (OpenAI, ElevenLabs) returns its
//! result in the submission itself, which no later poll can fetch again. Such
//! a job is recorded here, in application data, so reopening the project
//! resumes it (or finalizes the held result) instead of requiring a paid
//! retry, even after the app quit or restarted.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::durable_list::DurableJsonList;

const STORE_VERSION: u32 = 1;

/// A job id accepted by the provider that its project does not record yet.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct OrphanedGeneration {
    /// The local job id of the project's placeholders.
    pub job_id: String,
    pub provider_job_id: String,
    /// The bundle the job was working for (diagnostics only).
    pub project_path: String,
    /// Unix seconds when the job was recorded.
    pub recorded_at: u64,
    /// Results a synchronous provider returned with the submission, held
    /// under the store's `results/` directory in placeholder order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub held_results: Vec<HeldResult>,
}

/// One result file held for a job.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct HeldResult {
    /// File name under the store's `results/` directory.
    pub file: String,
    pub media_type: String,
    pub byte_size: u64,
}

pub(crate) struct OrphanedGenerationStore {
    list: DurableJsonList<OrphanedGeneration>,
    results_dir: PathBuf,
}

impl OrphanedGenerationStore {
    /// A store rooted at `root` (`<app data>/generation-orphans`).
    pub(crate) fn new(root: PathBuf) -> Self {
        Self {
            list: DurableJsonList::new(
                root.join("orphans.json"),
                "orphans",
                STORE_VERSION,
                "orphaned generation jobs",
            ),
            results_dir: root.join("results"),
        }
    }

    pub(crate) fn get(&self, job_id: &str) -> Result<Option<OrphanedGeneration>, String> {
        Ok(self
            .list
            .list()?
            .into_iter()
            .find(|entry| entry.job_id == job_id))
    }

    /// Record a job, replacing an earlier record of the same job (whose held
    /// files are removed unless the new record keeps them).
    pub(crate) fn record(&self, entry: OrphanedGeneration) -> Result<(), String> {
        let replaced = self.list.update(|entries| {
            let replaced = entries
                .iter()
                .position(|existing| existing.job_id == entry.job_id)
                .map(|index| entries.remove(index));
            entries.push(entry.clone());
            (true, replaced)
        })?;
        if let Some(replaced) = replaced {
            self.remove_held_files(
                replaced
                    .held_results
                    .iter()
                    .filter(|held| !entry.held_results.contains(held)),
            );
        }
        Ok(())
    }

    /// Forget a job once its project records it, with its held files.
    pub(crate) fn remove(&self, job_id: &str) -> Result<(), String> {
        let removed = self.list.update(|entries| {
            let removed = entries
                .iter()
                .position(|entry| entry.job_id == job_id)
                .map(|index| entries.remove(index));
            (removed.is_some(), removed)
        })?;
        if let Some(removed) = removed {
            self.remove_held_files(removed.held_results.iter());
        }
        Ok(())
    }

    /// Move a staged result file into the store. The file is fsynced before
    /// the caller records it.
    pub(crate) fn hold(&self, staged: &Path, media_type: &str) -> Result<HeldResult, String> {
        fs::create_dir_all(&self.results_dir)
            .map_err(|error| format!("create held generation results: {error}"))?;
        let file = format!("{}.result", uuid::Uuid::new_v4().simple());
        let destination = self.results_dir.join(&file);
        if fs::rename(staged, &destination).is_err() {
            // Staging may live on another volume.
            fs::copy(staged, &destination)
                .map_err(|error| format!("hold generation result: {error}"))?;
            let _ = fs::remove_file(staged);
        }
        // Flushing needs write access on Windows (`FlushFileBuffers`).
        let held = fs::OpenOptions::new()
            .write(true)
            .open(&destination)
            .and_then(|file| {
                file.sync_all()?;
                file.metadata()
            })
            .map_err(|error| format!("hold generation result: {error}"));
        let metadata = match held {
            Ok(metadata) => metadata,
            Err(error) => {
                let _ = fs::remove_file(&destination);
                return Err(error);
            }
        };
        crate::external_mcp::sync_parent_directory(&self.results_dir)
            .map_err(|error| format!("hold generation result: {error}"))?;
        Ok(HeldResult {
            file,
            media_type: media_type.to_string(),
            byte_size: metadata.len(),
        })
    }

    pub(crate) fn held_path(&self, held: &HeldResult) -> PathBuf {
        self.results_dir.join(&held.file)
    }

    fn remove_held_files<'a>(&self, held: impl Iterator<Item = &'a HeldResult>) {
        for held in held {
            let path = self.held_path(held);
            if let Err(error) = fs::remove_file(&path) {
                if error.kind() != std::io::ErrorKind::NotFound {
                    eprintln!(
                        "[generation] held result {} was not removed: {error}",
                        path.display()
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(job_id: &str, held_results: Vec<HeldResult>) -> OrphanedGeneration {
        OrphanedGeneration {
            job_id: job_id.into(),
            provider_job_id: format!("fal::{job_id}"),
            project_path: "/projects/A.opentake".into(),
            recorded_at: 1,
            held_results,
        }
    }

    #[test]
    fn records_survive_reopen_and_removal_deletes_held_files() {
        let root = tempfile::tempdir().unwrap();
        let store = OrphanedGenerationStore::new(root.path().join("orphans"));
        assert_eq!(store.get("job-1").unwrap(), None);

        let staged = root.path().join("staged.png");
        fs::write(&staged, b"png bytes").unwrap();
        let held = store.hold(&staged, "image/png").unwrap();
        assert!(!staged.exists());
        assert_eq!(held.byte_size, 9);
        store.record(record("job-1", vec![held.clone()])).unwrap();
        store.record(record("job-2", Vec::new())).unwrap();

        let reopened = OrphanedGenerationStore::new(root.path().join("orphans"));
        let found = reopened.get("job-1").unwrap().unwrap();
        assert_eq!(found.held_results, vec![held.clone()]);
        assert_eq!(fs::read(reopened.held_path(&held)).unwrap(), b"png bytes");

        reopened.remove("job-1").unwrap();
        assert_eq!(reopened.get("job-1").unwrap(), None);
        assert!(!reopened.held_path(&held).exists());
        assert!(reopened.get("job-2").unwrap().is_some());
        reopened.remove("job-unknown").unwrap();
    }

    #[test]
    fn a_new_record_of_the_same_job_replaces_the_old_one() {
        let root = tempfile::tempdir().unwrap();
        let store = OrphanedGenerationStore::new(root.path().to_path_buf());
        let staged = root.path().join("staged.mp3");
        fs::write(&staged, b"old").unwrap();
        let old = store.hold(&staged, "audio/mpeg").unwrap();
        store.record(record("job-1", vec![old.clone()])).unwrap();
        store.record(record("job-1", Vec::new())).unwrap();
        assert!(store.get("job-1").unwrap().unwrap().held_results.is_empty());
        assert!(!store.held_path(&old).exists());
    }
}
