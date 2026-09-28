//! Voice clones the user abandoned that OpenTake could not yet remove from
//! the provider.
//!
//! A cloned voice is consent-bound biometric data. When an enrollment is
//! cancelled, or its project record cannot be committed, the provider voice is
//! revoked straight away. If that revocation fails as well, the provider voice
//! id is queued here, in application data rather than in a project (there is
//! no project record to revoke it from), so Settings can list it and the user
//! can retry the removal.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::durable_list::DurableJsonList;

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

/// Durable queue of provider voices to revoke, written atomically. An
/// unreadable queue is set aside (kept for recovery) instead of blocking
/// every later record.
pub struct VoiceRevocationStore {
    list: DurableJsonList<PendingVoiceRevocation>,
}

impl VoiceRevocationStore {
    pub fn new(path: PathBuf) -> Self {
        Self {
            list: DurableJsonList::new(
                path,
                "revocations",
                STORE_VERSION,
                "pending voice removals",
            ),
        }
    }

    pub fn list(&self) -> Result<Vec<PendingVoiceRevocation>, String> {
        self.list.list()
    }

    /// Queue a voice, or refresh the failure of one already queued (keeping
    /// when it was first recorded).
    pub fn record(&self, entry: PendingVoiceRevocation) -> Result<(), String> {
        self.list.update(|entries| {
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
            (true, ())
        })
    }

    /// Drop a voice after the provider confirmed its removal. Returns whether
    /// it was queued.
    pub fn remove(&self, provider: &str, provider_voice_id: &str) -> Result<bool, String> {
        self.list.update(|entries| {
            let before = entries.len();
            entries.retain(|entry| {
                !(entry.provider == provider && entry.provider_voice_id == provider_voice_id)
            });
            let removed = entries.len() != before;
            (removed, removed)
        })
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
    use std::fs;

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
    fn a_corrupt_queue_is_set_aside_and_a_new_one_starts() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("voice-revocations.json");
        fs::write(&path, b"not json").unwrap();
        let store = VoiceRevocationStore::new(path.clone());
        assert!(store.list().unwrap().is_empty());
        store.record(entry("voice-a", "offline")).unwrap();
        assert_eq!(
            VoiceRevocationStore::new(path).list().unwrap(),
            vec![entry("voice-a", "offline")]
        );
        // The unreadable queue is kept for recovery.
        let kept = fs::read_dir(root.path())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("voice-revocations.json.unreadable-")
            })
            .collect::<Vec<_>>();
        assert_eq!(kept.len(), 1);
        assert_eq!(fs::read(&kept[0]).unwrap(), b"not json");

        // A queue from another format version is set aside the same way.
        let versioned = root.path().join("future.json");
        fs::write(&versioned, br#"{"version": 99, "revocations": []}"#).unwrap();
        let store = VoiceRevocationStore::new(versioned.clone());
        assert!(store.list().unwrap().is_empty());
        assert!(!versioned.exists());
    }
}
