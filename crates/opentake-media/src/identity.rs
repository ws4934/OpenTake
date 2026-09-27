//! Metadata-validated content hashes for render hot paths.
//!
//! Image textures are keyed by the source's SHA-256 so a replaced file never
//! reuses a stale texture, but hashing the whole file on every frame costs far
//! more than the frame budget. [`ContentHashCache`] hashes a file once and
//! reuses that hash while a cheap `fstat` identity ([`FileStamp`]) is
//! unchanged. The file is still opened and stat'ed on every lookup, so a
//! missing or unreadable source keeps failing immediately.

use std::collections::HashMap;
use std::fs::{File, Metadata};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::cancel::MediaCancelToken;
use crate::error::Result;
use crate::proxy::{file_sha256_file_cancellable, open_retained_regular_file};

/// Cheap file identity: size + modification time, plus device/inode and
/// status-change time on Unix. Any rewrite, replacement, or truncation changes
/// at least one field except a same-size rewrite within the filesystem's
/// timestamp granularity, which on Unix still bumps `ctime`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FileStamp {
    len: u64,
    modified: Option<SystemTime>,
    #[cfg(unix)]
    unix: (u64, u64, i64, i64),
}

impl FileStamp {
    pub fn of(metadata: &Metadata) -> Self {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        FileStamp {
            len: metadata.len(),
            modified: metadata.modified().ok(),
            #[cfg(unix)]
            unix: (
                metadata.dev(),
                metadata.ino(),
                metadata.ctime(),
                metadata.ctime_nsec(),
            ),
        }
    }
}

/// Path-keyed SHA-256 cache validated by [`FileStamp`] on every lookup.
#[derive(Default)]
pub struct ContentHashCache {
    entries: HashMap<PathBuf, (FileStamp, String)>,
    hashes: u64,
}

impl ContentHashCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// SHA-256 of the regular file at `path` (symlinks rejected, like
    /// [`crate::file_sha256`]), rehashing only when its identity changed.
    pub fn sha256(&mut self, path: &Path) -> Result<String> {
        let file = open_retained_regular_file(path)?;
        self.sha256_file(path, &file, &MediaCancelToken::new())
    }

    /// SHA-256 of an already-retained handle for `path`, rehashing only when
    /// the handle's identity changed.
    pub fn sha256_file(
        &mut self,
        path: &Path,
        file: &File,
        cancel: &MediaCancelToken,
    ) -> Result<String> {
        // Stamp before reading: a write racing the hash changes the stamp, so
        // the next lookup rehashes instead of trusting a torn read forever.
        let stamp = FileStamp::of(&file.metadata()?);
        if let Some((cached, hash)) = self.entries.get(path) {
            if *cached == stamp {
                return Ok(hash.clone());
            }
        }
        let hash = file_sha256_file_cancellable(file, cancel)?;
        self.hashes += 1;
        self.entries
            .insert(path.to_path_buf(), (stamp, hash.clone()));
        Ok(hash)
    }

    /// Number of full-file hashes computed so far.
    pub fn hashes(&self) -> u64 {
        self.hashes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unchanged_file_is_hashed_once() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("overlay.png");
        std::fs::write(&path, b"first").unwrap();
        let mut cache = ContentHashCache::new();
        let first = cache.sha256(&path).unwrap();
        for _ in 0..1000 {
            assert_eq!(cache.sha256(&path).unwrap(), first);
        }
        assert_eq!(cache.hashes(), 1);
        assert_eq!(first, crate::file_sha256(&path).unwrap());
    }

    #[test]
    fn modified_file_is_rehashed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("overlay.png");
        std::fs::write(&path, b"first").unwrap();
        let mut cache = ContentHashCache::new();
        let first = cache.sha256(&path).unwrap();
        std::fs::write(&path, b"second, longer").unwrap();
        let second = cache.sha256(&path).unwrap();
        assert_ne!(first, second);
        assert_eq!(second, crate::file_sha256(&path).unwrap());
        assert_eq!(cache.hashes(), 2);
    }

    #[test]
    fn missing_file_still_fails_after_caching() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("overlay.png");
        std::fs::write(&path, b"first").unwrap();
        let mut cache = ContentHashCache::new();
        cache.sha256(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert!(cache.sha256(&path).is_err());
    }
}
