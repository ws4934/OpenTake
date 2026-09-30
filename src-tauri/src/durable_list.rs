//! Small durable lists kept in application data (not in a project), written
//! atomically: a staged file is fsynced and renamed over the list.
//!
//! A list that cannot be read (corrupt, or written by another format version)
//! must not block every later record, so it is moved aside next to the list
//! (kept for recovery) and a new, empty list starts.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::de::DeserializeOwned;
use serde::Serialize;

pub(crate) struct DurableJsonList<T> {
    path: PathBuf,
    /// JSON key holding the entries next to `version`.
    key: &'static str,
    version: u32,
    /// What the list holds, for messages.
    label: &'static str,
    lock: Mutex<()>,
    entries: PhantomData<fn() -> T>,
}

impl<T: Serialize + DeserializeOwned + Clone> DurableJsonList<T> {
    pub(crate) fn new(path: PathBuf, key: &'static str, version: u32, label: &'static str) -> Self {
        Self {
            path,
            key,
            version,
            label,
            lock: Mutex::new(()),
            entries: PhantomData,
        }
    }

    pub(crate) fn list(&self) -> Result<Vec<T>, String> {
        let _guard = self
            .lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.read_locked(true)
    }

    /// A management view must expose damaged records instead of showing empty.
    pub(crate) fn list_strict(&self) -> Result<Vec<T>, String> {
        let _guard = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        self.read_locked(false)
    }

    /// Read, change and (when `change` says so) rewrite the list under one
    /// lock.
    pub(crate) fn update<R>(
        &self,
        change: impl FnOnce(&mut Vec<T>) -> (bool, R),
    ) -> Result<R, String> {
        let _guard = self
            .lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut entries = self.read_locked(true)?;
        let (write, result) = change(&mut entries);
        if write {
            self.write_locked(&entries)?;
        }
        Ok(result)
    }

    fn read_locked(&self, recover_invalid: bool) -> Result<Vec<T>, String> {
        let bytes = match fs::read(&self.path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(format!("read {}: {error}", self.label)),
        };
        match self.decode(&bytes) {
            Ok(entries) => Ok(entries),
            Err(reason) => {
                if !recover_invalid {
                    return Err(format!("{} could not be read: {reason}", self.label));
                }
                let kept = self.quarantine()?;
                eprintln!(
                    "[opentake] {} at {} could not be read ({reason}); it was kept as {} and a \
                     new list was started",
                    self.label,
                    self.path.display(),
                    kept.display()
                );
                Ok(Vec::new())
            }
        }
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<T>, String> {
        let mut value: serde_json::Value =
            serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
        let version = value
            .get("version")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| "missing version".to_string())?;
        if version != u64::from(self.version) {
            return Err(format!("unsupported version {version}"));
        }
        let entries = value
            .get_mut(self.key)
            .map(serde_json::Value::take)
            .ok_or_else(|| format!("missing {}", self.key))?;
        serde_json::from_value(entries).map_err(|error| error.to_string())
    }

    /// Move an unreadable list aside, next to it, and return where it went.
    fn quarantine(&self) -> Result<PathBuf, String> {
        let name = self
            .path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "list".to_string());
        let kept = self.path.with_file_name(format!(
            "{name}.unreadable-{}-{}",
            crate::voice_revocations::unix_now_seconds(),
            &uuid::Uuid::new_v4().simple().to_string()[..8]
        ));
        fs::rename(&self.path, &kept)
            .map_err(|error| format!("set aside unreadable {}: {error}", self.label))?;
        let _ = crate::external_mcp::sync_parent_directory(self.parent());
        Ok(kept)
    }

    fn parent(&self) -> &Path {
        self.path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
    }

    fn write_locked(&self, entries: &[T]) -> Result<(), String> {
        let parent = self.parent();
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {} directory: {error}", self.label))?;
        let mut document = serde_json::Map::new();
        document.insert("version".to_string(), self.version.into());
        document.insert(
            self.key.to_string(),
            serde_json::to_value(entries)
                .map_err(|error| format!("encode {}: {error}", self.label))?,
        );
        let bytes = serde_json::to_vec_pretty(&document)
            .map_err(|error| format!("encode {}: {error}", self.label))?;
        let staging = parent.join(format!(".{}.{}.tmp", self.key, uuid::Uuid::new_v4()));
        let result = (|| {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&staging)
                .map_err(|error| format!("stage {}: {error}", self.label))?;
            file.write_all(&bytes)
                .and_then(|()| file.sync_all())
                .map_err(|error| format!("write {}: {error}", self.label))?;
            drop(file);
            crate::external_mcp::replace_file_atomically(&staging, &self.path)
                .map_err(|error| format!("publish {}: {error}", self.label))?;
            crate::external_mcp::sync_parent_directory(parent)
                .map_err(|error| format!("sync {}: {error}", self.label))
        })();
        if result.is_err() {
            let _ = fs::remove_file(&staging);
        }
        result
    }
}
