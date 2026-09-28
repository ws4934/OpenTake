//! Voice clones the user abandoned that OpenTake could not yet remove from
//! the provider.
//!
//! A cloned voice is consent-bound biometric data. When an enrollment is
//! cancelled, or its project record cannot be committed, the provider voice is
//! revoked straight away. If that revocation fails as well, the provider voice
//! id is queued here, in application data rather than in a project (there is
//! no project record to revoke it from), so Settings can list it and the user
//! can retry the removal.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

const STORE_VERSION: u32 = 1;

/// One provider voice whose removal is still owed to the user.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct PendingVoiceRevocation {
    pub provider: String,
    pub provider_voice_id: String,
    pub voice_name: String,
    /// Unix seconds when the removal was first queued.
    pub recorded_at: u64,
    /// The latest removal failure. Provider errors never carry credentials.
    pub last_error: String,
}

#[derive(Serialize, Deserialize)]
struct PersistedRevocations {
    version: u32,
    revocations: Vec<PendingVoiceRevocation>,
}

/// Durable queue of provider voices to revoke, written atomically.
pub struct VoiceRevocationStore {
    path: PathBuf,
    lock: Mutex<()>,
}

impl VoiceRevocationStore {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            lock: Mutex::new(()),
        }
    }

    pub fn list(&self) -> Result<Vec<PendingVoiceRevocation>, String> {
        let _guard = self
            .lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.read_locked()
    }

    /// Queue a voice, or refresh the failure of one already queued (keeping
    /// when it was first recorded).
    pub fn record(&self, entry: PendingVoiceRevocation) -> Result<(), String> {
        let _guard = self
            .lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut entries = self.read_locked()?;
        match entries.iter_mut().find(|existing| {
            existing.provider == entry.provider
                && existing.provider_voice_id == entry.provider_voice_id
        }) {
            Some(existing) => {
                existing.last_error = entry.last_error;
                if existing.voice_name.is_empty() {
                    existing.voice_name = entry.voice_name;
                }
            }
            None => entries.push(entry),
        }
        self.write_locked(&entries)
    }

    /// Drop a voice after the provider confirmed its removal. Returns whether
    /// it was queued.
    pub fn remove(&self, provider: &str, provider_voice_id: &str) -> Result<bool, String> {
        let _guard = self
            .lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut entries = self.read_locked()?;
        let before = entries.len();
        entries.retain(|entry| {
            !(entry.provider == provider && entry.provider_voice_id == provider_voice_id)
        });
        if entries.len() == before {
            return Ok(false);
        }
        self.write_locked(&entries)?;
        Ok(true)
    }

    fn read_locked(&self) -> Result<Vec<PendingVoiceRevocation>, String> {
        let bytes = match fs::read(&self.path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(format!("read pending voice removals: {error}")),
        };
        let persisted: PersistedRevocations = serde_json::from_slice(&bytes)
            .map_err(|error| format!("read pending voice removals: {error}"))?;
        if persisted.version != STORE_VERSION {
            return Err("unsupported pending voice removal list version".to_string());
        }
        Ok(persisted.revocations)
    }

    fn write_locked(&self, entries: &[PendingVoiceRevocation]) -> Result<(), String> {
        let parent = self
            .path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)
            .map_err(|error| format!("create pending voice removal directory: {error}"))?;
        let bytes = serde_json::to_vec_pretty(&PersistedRevocations {
            version: STORE_VERSION,
            revocations: entries.to_vec(),
        })
        .map_err(|error| format!("encode pending voice removals: {error}"))?;
        let staging = parent.join(format!(".voice-revocations.{}.tmp", uuid::Uuid::new_v4()));
        let result = (|| {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&staging)
                .map_err(|error| format!("stage pending voice removals: {error}"))?;
            file.write_all(&bytes)
                .and_then(|()| file.sync_all())
                .map_err(|error| format!("write pending voice removals: {error}"))?;
            drop(file);
            crate::external_mcp::replace_file_atomically(&staging, &self.path)
                .map_err(|error| format!("publish pending voice removals: {error}"))?;
            crate::external_mcp::sync_parent_directory(parent)
                .map_err(|error| format!("sync pending voice removals: {error}"))
        })();
        if result.is_err() {
            let _ = fs::remove_file(&staging);
        }
        result
    }
}

pub fn unix_now_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str, error: &str) -> PendingVoiceRevocation {
        PendingVoiceRevocation {
            provider: "elevenlabs".into(),
            provider_voice_id: id.into(),
            voice_name: "Narrator".into(),
            recorded_at: 100,
            last_error: error.into(),
        }
    }

    #[test]
    fn queued_voices_survive_reopen_update_in_place_and_leave_on_removal() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("app-data").join("voice-revocations.json");
        let store = VoiceRevocationStore::new(path.clone());
        assert!(store.list().unwrap().is_empty());

        store.record(entry("voice-a", "offline")).unwrap();
        store.record(entry("voice-b", "HTTP 500")).unwrap();
        let mut refreshed = entry("voice-a", "HTTP 503");
        refreshed.recorded_at = 999;
        store.record(refreshed).unwrap();

        let reopened = VoiceRevocationStore::new(path.clone());
        let listed = reopened.list().unwrap();
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].provider_voice_id, "voice-a");
        assert_eq!(listed[0].last_error, "HTTP 503");
        assert_eq!(listed[0].recorded_at, 100, "first queue time is kept");

        assert!(reopened.remove("elevenlabs", "voice-a").unwrap());
        assert!(!reopened.remove("elevenlabs", "voice-a").unwrap());
        assert_eq!(
            VoiceRevocationStore::new(path).list().unwrap(),
            vec![entry("voice-b", "HTTP 500")]
        );
        // Only the published file remains: no staging leftovers.
        let names = fs::read_dir(root.path().join("app-data"))
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["voice-revocations.json".to_string()]);
    }

    #[test]
    fn a_corrupt_queue_is_reported_instead_of_being_overwritten() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("voice-revocations.json");
        fs::write(&path, b"not json").unwrap();
        let store = VoiceRevocationStore::new(path.clone());
        assert!(store.list().is_err());
        assert!(store.record(entry("voice-a", "offline")).is_err());
        assert_eq!(fs::read(&path).unwrap(), b"not json");
    }
}
