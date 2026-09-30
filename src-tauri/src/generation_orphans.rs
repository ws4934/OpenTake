//! Paid generation jobs whose project was closed before their state could be
//! written to it.
//!
//! A provider can accept a job after the project that submitted it was
//! replaced, and a synchronous provider (OpenAI, ElevenLabs) returns its
//! result in the submission itself, which no later poll can fetch again. Such
//! a job is recorded here, in application data, so reopening the project
//! resumes it (or finalizes the held result) instead of requiring a paid
//! retry, even after the app quit or restarted.
//!
//! A submission is also recorded here, without a provider job id, from just
//! before it is sent until its project records the answer. If the app quits
//! or the project is replaced in between, or the answer never arrives,
//! reopening the project reports that the provider may have accepted and
//! billed the job instead of offering a plain retry.

use std::fs;
use std::path::{Path, PathBuf};

use cap_fs_ext::DirExt;
use serde::{Deserialize, Serialize};

use crate::durable_list::DurableJsonList;

const STORE_VERSION: u32 = 1;

/// A job id accepted by the provider that its project does not record yet.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct OrphanedGeneration {
    /// The local job id of the project's placeholders.
    pub job_id: String,
    /// `None` while the submission's outcome is unknown: it was sent, and
    /// the provider may have accepted and billed it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_job_id: Option<String>,
    /// The bundle the job was working for (diagnostics only).
    pub project_path: String,
    /// Unix seconds when the job was recorded.
    pub recorded_at: u64,
    /// Results a synchronous provider returned with the submission, held
    /// under the store's `results/` directory in placeholder order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub held_results: Vec<HeldResult>,
    /// Persisted before deletion so an interrupted cleanup can be retried.
    #[serde(default)]
    pub discarded: bool,
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
    /// Makes [`Self::record`] fail, as a full disk would.
    #[cfg(test)]
    pub(crate) fail_records: std::sync::atomic::AtomicBool,
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
            #[cfg(test)]
            fail_records: std::sync::atomic::AtomicBool::new(false),
        }
    }

    pub(crate) fn get(&self, job_id: &str) -> Result<Option<OrphanedGeneration>, String> {
        Ok(self
            .list()?
            .into_iter()
            .find(|entry| entry.job_id == job_id && !entry.discarded))
    }

    pub(crate) fn list(&self) -> Result<Vec<OrphanedGeneration>, String> {
        let entries = self.list.list_strict()?;
        for entry in &entries {
            Self::validate_entry(entry)?;
        }
        Ok(entries)
    }

    fn validate_entry(entry: &OrphanedGeneration) -> Result<(), String> {
        for held in &entry.held_results {
            if held.file.is_empty()
                || held.file == "."
                || held.file == ".."
                || held.file.contains(['/', '\\', ':', '\0'])
            {
                return Err("held generation result is not a file name".into());
            }
        }
        Ok(())
    }

    fn results_directory(&self) -> Result<cap_std::fs::Dir, String> {
        let root = self.results_dir.parent().expect("store root");
        let parent = root.parent().ok_or("generation store has no parent")?;
        let parent = cap_std::fs::Dir::open_ambient_dir(parent, cap_std::ambient_authority())
            .map_err(|e| format!("open generation store parent: {e}"))?;
        let root = parent
            .open_dir_nofollow(root.file_name().ok_or("generation store has no name")?)
            .map_err(|e| format!("open generation store: {e}"))?;
        root.open_dir_nofollow("results")
            .map_err(|e| format!("open held results: {e}"))
    }

    /// Record a job, replacing an earlier record of the same job (whose held
    /// files are removed unless the new record keeps them).
    pub(crate) fn record(&self, entry: OrphanedGeneration) -> Result<(), String> {
        Self::validate_entry(&entry)?;
        #[cfg(test)]
        if self.fail_records.load(std::sync::atomic::Ordering::SeqCst) {
            return Err("write orphaned generation jobs: no space left on device".to_string());
        }
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

    /// Forget a job and its held files. Failed cleanup retains a tombstone.
    pub(crate) fn remove(&self, job_id: &str) -> Result<(), String> {
        let Some(entry) = self
            .list()?
            .into_iter()
            .find(|entry| entry.job_id == job_id)
        else {
            return Ok(());
        };
        let directory = if entry.held_results.is_empty() {
            None
        } else {
            Some(self.results_directory()?)
        };
        self.list.update(|entries| {
            let entry = entries.iter_mut().find(|entry| entry.job_id == job_id);
            let changed = entry.as_ref().is_some_and(|entry| !entry.discarded);
            if let Some(entry) = entry {
                entry.discarded = true;
            }
            (changed, ())
        })?;
        if let Some(directory) = directory {
            for held in &entry.held_results {
                match directory.remove_file(&held.file) {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => return Err(format!("remove held generation result: {e}")),
                }
            }
        }
        self.list.update(|entries| {
            let old_len = entries.len();
            entries.retain(|entry| entry.job_id != job_id || !entry.discarded);
            (entries.len() != old_len, ())
        })?;
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
            provider_job_id: Some(format!("fal::{job_id}")),
            project_path: "/projects/A.opentake".into(),
            recorded_at: 1,
            held_results,
            discarded: false,
        }
    }

    #[test]
    fn management_preserves_and_reports_a_damaged_store() {
        let root = tempfile::tempdir().unwrap();
        let store = OrphanedGenerationStore::new(root.path().to_path_buf());
        let path = root.path().join("orphans.json");
        for bytes in [
            b"{".as_slice(),
            br#"{"version":99,"orphans":[]}"#.as_slice(),
        ] {
            fs::write(&path, bytes).unwrap();
            assert!(store.list().is_err());
            assert!(store.get("job").is_err());
            assert!(store.remove("job").is_err());
            assert_eq!(fs::read(&path).unwrap(), bytes);
            assert_eq!(fs::read_dir(root.path()).unwrap().count(), 1);
        }
    }

    #[test]
    fn failed_cleanup_remains_visible_and_can_be_retried() {
        let root = tempfile::tempdir().unwrap();
        let store = OrphanedGenerationStore::new(root.path().join("orphans"));
        let leaf = store.results_dir.join("held.result");
        fs::create_dir_all(&leaf).unwrap();
        store
            .record(record(
                "job",
                vec![HeldResult {
                    file: "held.result".into(),
                    media_type: "image/png".into(),
                    byte_size: 1,
                }],
            ))
            .unwrap();
        assert!(store.remove("job").is_err());
        assert!(store.list().unwrap()[0].discarded);
        assert!(store.get("job").unwrap().is_none());
        fs::remove_dir(&leaf).unwrap();
        store.remove("job").unwrap();
        assert!(store.list().unwrap().is_empty());
    }

    #[test]
    fn held_file_names_cannot_escape_the_store() {
        let root = tempfile::tempdir().unwrap();
        let store = OrphanedGenerationStore::new(root.path().join("orphans"));
        for file in ["../outside", "..\\outside", "C:outside", "/outside"] {
            assert!(store
                .record(record(
                    "job",
                    vec![HeldResult {
                        file: file.into(),
                        media_type: "image/png".into(),
                        byte_size: 1
                    }]
                ))
                .is_err());
        }
        assert!(store.list().unwrap().is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn cleanup_rejects_a_replaced_results_directory() {
        let root = tempfile::tempdir().unwrap();
        let store = OrphanedGenerationStore::new(root.path().join("orphans"));
        store
            .record(record(
                "job",
                vec![HeldResult {
                    file: "held.result".into(),
                    media_type: "image/png".into(),
                    byte_size: 1,
                }],
            ))
            .unwrap();
        let outside = root.path().join("outside");
        fs::create_dir(&outside).unwrap();
        fs::write(outside.join("held.result"), b"unrelated").unwrap();
        std::os::unix::fs::symlink(&outside, &store.results_dir).unwrap();
        assert!(store.remove("job").is_err());
        assert_eq!(fs::read(outside.join("held.result")).unwrap(), b"unrelated");
        assert!(!store.list().unwrap()[0].discarded);
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
    fn a_submission_with_an_unknown_outcome_is_recorded_without_a_provider_job_id() {
        let root = tempfile::tempdir().unwrap();
        let store = OrphanedGenerationStore::new(root.path().to_path_buf());
        let mut unknown = record("job-1", Vec::new());
        unknown.provider_job_id = None;
        store.record(unknown.clone()).unwrap();
        let text = fs::read_to_string(root.path().join("orphans.json")).unwrap();
        assert!(!text.contains("providerJobId"), "{text}");
        assert_eq!(
            OrphanedGenerationStore::new(root.path().to_path_buf())
                .get("job-1")
                .unwrap(),
            Some(unknown)
        );
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
