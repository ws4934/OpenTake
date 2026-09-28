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
use std::io;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::cancel::MediaCancelToken;
use crate::error::Result;
use crate::proxy::{file_sha256_file_cancellable, open_retained_regular_file};

/// Cheap file identity: size + modification time, plus the file's identity
/// and change time (device, inode and `ctime` on Unix; volume serial number,
/// file index and `ChangeTime` on Windows). Any rewrite, replacement, or
/// truncation changes at least one field: a replacement is a different file
/// identity even when it preserves the size and modification time, and a
/// same-size in-place rewrite with a restored modification time still bumps
/// the change time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FileStamp {
    len: u64,
    modified: Option<SystemTime>,
    #[cfg(unix)]
    unix: (u64, u64, i64, i64),
    /// `(volume serial, file index, change time)` of the open handle.
    #[cfg(windows)]
    windows: (u32, u64, i64),
}

impl FileStamp {
    /// Stamp an open file. The identity describes the handle, so it stays
    /// stable across reopening the same file.
    pub fn of_file(file: &File) -> io::Result<Self> {
        let metadata = file.metadata()?;
        #[cfg(windows)]
        {
            Ok(Self::with_windows_identity(
                &metadata,
                windows_file_identity(file)?,
            ))
        }
        #[cfg(not(windows))]
        {
            Ok(Self::of_metadata(&metadata))
        }
    }

    /// Stamp the file at `path` (following symlinks, like `fs::metadata`).
    /// On Windows the file identity needs a handle, so the path is opened the
    /// way `fs::metadata` opens it: without read access and with backup
    /// semantics, so directories and read-locked files stamp too.
    pub fn of_path(path: &Path) -> io::Result<Self> {
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            use windows_sys::Win32::Storage::FileSystem::FILE_FLAG_BACKUP_SEMANTICS;
            let file = File::options()
                .access_mode(0)
                .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
                .open(path)?;
            Self::of_file(&file)
        }
        #[cfg(not(windows))]
        {
            // Never open: a FIFO would block. `stat` carries the identity.
            Ok(Self::of_metadata(&std::fs::metadata(path)?))
        }
    }

    #[cfg(not(windows))]
    fn of_metadata(metadata: &Metadata) -> Self {
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

    #[cfg(windows)]
    fn with_windows_identity(metadata: &Metadata, identity: (u32, u64, i64)) -> Self {
        FileStamp {
            len: metadata.len(),
            modified: metadata.modified().ok(),
            windows: identity,
        }
    }
}

/// Volume serial number, file index and change time of an open handle.
#[cfg(windows)]
fn windows_file_identity(file: &File) -> std::io::Result<(u32, u64, i64)> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::Storage::FileSystem::{
        FileBasicInfo, GetFileInformationByHandle, GetFileInformationByHandleEx,
        BY_HANDLE_FILE_INFORMATION, FILE_BASIC_INFO,
    };

    let handle = file.as_raw_handle() as HANDLE;
    let mut information = BY_HANDLE_FILE_INFORMATION::default();
    // SAFETY: `file` owns a live handle and `information` is writable.
    if unsafe { GetFileInformationByHandle(handle, std::ptr::addr_of_mut!(information)) } == 0 {
        return Err(std::io::Error::last_os_error());
    }
    let mut basic = FILE_BASIC_INFO::default();
    // SAFETY: `basic` is a writable `FILE_BASIC_INFO` of the size passed.
    if unsafe {
        GetFileInformationByHandleEx(
            handle,
            FileBasicInfo,
            std::ptr::addr_of_mut!(basic).cast(),
            std::mem::size_of::<FILE_BASIC_INFO>() as u32,
        )
    } == 0
    {
        return Err(std::io::Error::last_os_error());
    }
    let file_index =
        (u64::from(information.nFileIndexHigh) << 32) | u64::from(information.nFileIndexLow);
    Ok((
        information.dwVolumeSerialNumber,
        file_index,
        basic.ChangeTime,
    ))
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
        let stamp = FileStamp::of_file(file)?;
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
    fn same_size_rewrite_with_restored_mtime_is_rehashed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("overlay.png");
        std::fs::write(&path, b"first").unwrap();
        let modified = std::fs::metadata(&path).unwrap().modified().unwrap();
        let mut cache = ContentHashCache::new();
        let first = cache.sha256(&path).unwrap();

        // Rewrite in place with the same length, then restore the old mtime:
        // only the change time (Unix `ctime`, Windows `ChangeTime`) differs.
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(&path, b"other").unwrap();
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(modified)
            .unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().modified().unwrap(),
            modified
        );

        assert_ne!(cache.sha256(&path).unwrap(), first);
        assert_eq!(cache.hashes(), 2);
    }

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
    fn stamp_is_stable_across_reopen_and_changes_on_replacement() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("overlay.png");
        std::fs::write(&path, b"first").unwrap();
        let first = FileStamp::of_path(&path).unwrap();
        let handle = File::open(&path).unwrap();
        assert_eq!(FileStamp::of_file(&handle).unwrap(), first);
        assert_eq!(
            FileStamp::of_file(&File::open(&path).unwrap()).unwrap(),
            first
        );
        assert_eq!(FileStamp::of_path(&path).unwrap(), first);

        // A same-size replacement that restores the modification time is
        // still a different file.
        let modified = std::fs::metadata(&path).unwrap().modified().unwrap();
        let replacement = dir.path().join("replacement.png");
        std::fs::write(&replacement, b"other").unwrap();
        File::options()
            .write(true)
            .open(&replacement)
            .unwrap()
            .set_modified(modified)
            .unwrap();
        drop(handle);
        std::fs::rename(&replacement, &path).unwrap();
        let replaced = FileStamp::of_path(&path).unwrap();
        assert_ne!(replaced, first);

        let mut cache = ContentHashCache::new();
        std::fs::write(&replacement, b"first").unwrap();
        File::options()
            .write(true)
            .open(&replacement)
            .unwrap()
            .set_modified(modified)
            .unwrap();
        std::fs::rename(&replacement, &path).unwrap();
        let before = cache.sha256(&path).unwrap();
        std::fs::write(&replacement, b"other").unwrap();
        File::options()
            .write(true)
            .open(&replacement)
            .unwrap()
            .set_modified(modified)
            .unwrap();
        std::fs::rename(&replacement, &path).unwrap();
        assert_ne!(cache.sha256(&path).unwrap(), before, "stale hash reused");
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
