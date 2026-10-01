//! Global asset library — a cross-project store of "favorited" media that lives
//! outside any single OpenTake project (issue #54, part of #37 "全局可复用素材库").
//!
//! Layout under the library root (resolved cross-platform via [`dirs`], e.g.
//! `~/Library/Application Support/OpenTake/Library/` on macOS):
//! ```text
//! <root>/
//!   library.json          manifest: { version, entries: [LibraryEntry, …] }
//!   files/<hash><ext>      copy-on-favorite content, content-addressed
//!   files/.staging/*       unpublished content, reclaimed on startup
//!   .library.json.*.tmp    per-transaction temp, atomically committed/cleaned
//! ```
//!
//! Design choices:
//! - **copy-on-favorite**: favoriting copies the source file *into* the library
//!   so it survives the original being moved/deleted.
//! - **hash dedup**: the in-library filename is the SHA-256 of the file content,
//!   so favoriting the same bytes twice stores one copy and reuses it.
//! - **atomic manifest**: the manifest is written to a temp file and renamed,
//!   so a crash mid-write never leaves a truncated `library.json`. An in-process
//!   `Mutex` serializes read-modify-write so concurrent favorites from worker
//!   threads do not lose entries.
//!
//! The store takes its root as an explicit path so it stays testable; the Tauri
//! command layer (#55) constructs it from `app_data_dir`. [`default_library_dir`]
//! provides the `dirs`-based production default.

use std::collections::{HashMap, HashSet};
use std::ffi::{OsStr, OsString};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use cap_fs_ext::{ambient_authority, DirExt, FollowSymlinks, OpenOptionsFollowExt};
use cap_std::fs::{Dir, OpenOptions};
use same_file::Handle;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{MediaError, Result};

/// Manifest filename under the library root.
pub const MANIFEST_NAME: &str = "library.json";
/// Subdirectory holding the content-addressed copies.
pub const FILES_SUBDIR: &str = "files";
/// Hidden directory for favorite content that is not yet visible in the
/// library manifest.
const STAGING_SUBDIR: &str = ".staging";
/// On-disk manifest schema version (bumped on incompatible changes).
pub const MANIFEST_VERSION: u32 = 1;
/// Application directory name under the platform data dir.
const APP_DIR: &str = "OpenTake";
/// Library directory name under the application directory.
const LIBRARY_DIR: &str = "Library";
const STREAM_BUFFER_SIZE: usize = 64 * 1024;

#[cfg(test)]
std::thread_local! {
    static FAIL_MANIFEST_DIRECTORY_SYNC: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static FAIL_COMMITTED_BACKUP_CLEANUP: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(any(test, feature = "test-faults"))]
std::thread_local! {
    static FAIL_REMOVED_STORED_CLEANUP: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static FAIL_ATOMIC_CAPABILITY_REPLACE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static FAIL_REPAIR_STORED_COPY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Test-only fault injection for the post-commit remove cleanup boundary.
/// This is public only so the Tauri integration test can prove that project
/// mapping cleanup still completes after the global manifest commit point.
#[doc(hidden)]
#[cfg(any(test, feature = "test-faults"))]
pub fn fail_next_removed_stored_cleanup_for_test() {
    FAIL_REMOVED_STORED_CLEANUP.with(|fail| fail.set(true));
}

/// Test-only fault injection immediately before a capability-bound atomic
/// replacement. The already-existing canonical leaf must remain unchanged.
#[doc(hidden)]
#[cfg(any(test, feature = "test-faults"))]
pub fn fail_next_atomic_capability_replace_for_test() {
    FAIL_ATOMIC_CAPABILITY_REPLACE.with(|fail| fail.set(true));
}

/// Test-only fault injection for one stored-copy repair. This lets integration
/// tests exercise per-asset failure isolation without replacing a retained
/// directory, which Windows intentionally prevents while handles are open.
#[doc(hidden)]
#[cfg(any(test, feature = "test-faults"))]
pub fn fail_next_repair_stored_copy_for_test() {
    FAIL_REPAIR_STORED_COPY.with(|fail| fail.set(true));
}

#[cfg(test)]
std::thread_local! {
    static STORED_INDEX_SCANS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static STREAM_HASH_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Whether this thread currently holds the library write lock.
    static WRITE_LOCK_HELD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Whole-file hash/copy passes started while holding the write lock.
    static STREAMS_UNDER_WRITE_LOCK: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The library write lock. It serializes manifest read-modify-write and the
/// O(1) renames that publish content; whole-file hashing and copying happen
/// outside it so one large favorite cannot block every other library write.
struct WriteLockGuard<'a> {
    _guard: std::sync::MutexGuard<'a, ()>,
}

impl WriteLockGuard<'_> {
    fn acquire(lock: &Mutex<()>) -> WriteLockGuard<'_> {
        let guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        #[cfg(test)]
        WRITE_LOCK_HELD.with(|held| held.set(true));
        WriteLockGuard { _guard: guard }
    }
}

impl Drop for WriteLockGuard<'_> {
    fn drop(&mut self) {
        #[cfg(test)]
        WRITE_LOCK_HELD.with(|held| held.set(false));
    }
}

/// Content-addressed leaves found in `files/`, grouped by content id. Only
/// non-empty, nofollow regular files whose names carry a valid content id are
/// candidates; anything else (`.DS_Store`, `desktop.ini`, sync-tool metadata,
/// directories, links, truncated leftovers) is ignored.
struct StoredIndex {
    by_id: HashMap<String, Vec<OsString>>,
}

impl StoredIndex {
    /// The unique stored leaf for `id`. Several leaves claiming one id stay a
    /// fail-closed error for that id only; the rest of the library keeps
    /// working and reconciliation/favoriting resolve verified duplicates.
    fn lookup(&self, id: &str) -> Result<Option<&OsString>> {
        match self.by_id.get(id).map(Vec::as_slice) {
            None | Some([]) => Ok(None),
            Some([name]) => Ok(Some(name)),
            Some(_) => Err(MediaError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("multiple stored copies claim library id {id}"),
            ))),
        }
    }

    fn candidates(&self, id: &str) -> &[OsString] {
        self.by_id.get(id).map(Vec::as_slice).unwrap_or(&[])
    }
}

/// One favorited asset in the global library.
///
/// JSON is camelCase (`favoritedAt`) to match the frontend DTO (#37-B/#37-C).
/// Identity, type, and timestamp are required. Optional metadata may be absent,
/// but malformed or partial records fail closed instead of inventing values.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LibraryEntry {
    /// Content hash (SHA-256 hex) of the stored file — the library-internal id.
    pub id: String,
    /// Asset kind, e.g. `"video"`, `"audio"`, `"image"`. `type` in JSON.
    #[serde(rename = "type")]
    pub kind: String,
    /// Optional user category/tag for filtering; `None` when uncategorized.
    #[serde(default)]
    pub category: Option<String>,
    /// Unix epoch seconds when the asset was favorited.
    pub favorited_at: f64,
    /// Original source path the file was copied from (for display/back-ref).
    #[serde(default)]
    pub source: Option<String>,
    /// Optional thumbnail reference (path or data URI), filled by upper layers.
    #[serde(default)]
    pub thumb: Option<String>,
}

/// The persisted manifest: a version tag plus the entry list.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Manifest {
    version: u32,
    entries: Vec<LibraryEntry>,
}

fn validate_manifest(manifest: Manifest) -> Result<Manifest> {
    if manifest.version != MANIFEST_VERSION {
        return Err(MediaError::Other(anyhow::anyhow!(
            "unsupported library manifest version: {}",
            manifest.version
        )));
    }
    let mut ids = HashSet::new();
    for entry in &manifest.entries {
        let valid_id = entry.id.len() == 64
            && entry
                .id
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
        if !valid_id {
            return Err(MediaError::Other(anyhow::anyhow!(
                "invalid library content id: {}",
                entry.id
            )));
        }
        if !ids.insert(entry.id.as_str()) {
            return Err(MediaError::Other(anyhow::anyhow!(
                "duplicate library content id: {}",
                entry.id
            )));
        }
    }
    Ok(manifest)
}

fn decode_manifest(bytes: &[u8]) -> Result<Manifest> {
    validate_manifest(serde_json::from_slice(bytes)?)
}

/// Describes a file to favorite into the library. The id/stored path are derived
/// from the file content, so the caller only supplies the source and metadata.
#[derive(Clone, Debug)]
pub struct FavoriteRequest<'a> {
    /// Path to the source file to copy in.
    pub source: &'a Path,
    /// Asset kind (`"video"` / `"audio"` / `"image"` / …).
    pub kind: &'a str,
    /// Optional category/tag.
    pub category: Option<String>,
    /// Unix epoch seconds to record; the command layer passes the real clock.
    pub favorited_at: f64,
    /// Optional thumbnail reference.
    pub thumb: Option<String>,
}

/// Result of one favorite transaction. `created` reports whether publication
/// appended the entry to the manifest; callers must not use it as ownership
/// proof for a later compensating delete.
#[derive(Clone, Debug, PartialEq)]
pub struct FavoriteOutcome {
    /// The entry selected by the actual source bytes read by this transaction.
    pub entry: LibraryEntry,
    /// `true` only when this transaction appended `entry` to the manifest.
    pub created: bool,
}

/// A favorite prepared from one immutable source-byte snapshot. A new entry's
/// content stays under the hidden staging directory until [`LibraryStore`] is
/// explicitly asked to publish it. Dropping an unpublished preparation removes
/// its private stage best-effort; even if cleanup fails, it is never in Mine.
pub struct PreparedFavorite {
    entry: LibraryEntry,
    capabilities: Arc<LibraryCapabilities>,
    active_stages: Arc<Mutex<HashSet<OsString>>>,
    stage: Option<OwnedLeaf>,
    stored_name: Option<OsString>,
    final_leaf: Option<OwnedLeaf>,
}

impl PreparedFavorite {
    /// Entry identity derived from the exact source bytes read during prepare.
    pub fn entry(&self) -> &LibraryEntry {
        &self.entry
    }

    /// Whether publication is still required after the project mapping saves.
    pub fn needs_publish(&self) -> bool {
        self.stage.is_some() || self.final_leaf.is_some()
    }

    fn release_stage(&mut self) {
        let Some(stage) = self.stage.take() else {
            return;
        };
        self.active_stages
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&stage.name);
        drop(stage);
    }
}

impl Drop for PreparedFavorite {
    fn drop(&mut self) {
        self.release_stage();
        self.final_leaf.take();
    }
}

struct LibraryCapabilities {
    root: Dir,
    files: Dir,
    staging: Dir,
}

/// The global library store, rooted at a directory. Cloneable handles are not
/// provided; share one instance behind an `Arc` if multiple owners are needed.
pub struct LibraryStore {
    root: PathBuf,
    capabilities: std::result::Result<Arc<LibraryCapabilities>, String>,
    active_stages: Arc<Mutex<HashSet<OsString>>>,
    /// Serializes manifest read-modify-write across in-process threads.
    write_lock: Mutex<()>,
}

/// Cross-platform default library directory:
/// `<platform data dir>/OpenTake/Library`. Returns `None` only if the platform
/// data directory cannot be resolved (handled as an error by callers).
pub fn default_library_dir() -> Option<PathBuf> {
    dirs::data_dir().map(|d| d.join(APP_DIR).join(LIBRARY_DIR))
}

fn open_capabilities(root_path: &Path) -> std::io::Result<LibraryCapabilities> {
    let parent_path = root_path.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "library root must have a parent directory",
        )
    })?;
    let root_name = root_path.file_name().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "library root must have a final component",
        )
    })?;
    Dir::create_ambient_dir_all(parent_path, ambient_authority())?;
    let parent = Dir::open_ambient_dir(parent_path, ambient_authority())?;
    ensure_child_dir(&parent, root_name)?;
    let root = parent.open_dir_nofollow(root_name)?;
    ensure_child_dir(&root, FILES_SUBDIR)?;
    let files = root.open_dir_nofollow(FILES_SUBDIR)?;
    ensure_child_dir(&files, STAGING_SUBDIR)?;
    let staging = files.open_dir_nofollow(STAGING_SUBDIR)?;
    Ok(LibraryCapabilities {
        root,
        files,
        staging,
    })
}

fn ensure_child_dir(parent: &Dir, name: impl AsRef<Path>) -> std::io::Result<()> {
    match parent.create_dir(name.as_ref()) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(error) => Err(error),
    }
}

struct OwnedLeaf {
    name: OsString,
    handle: Handle,
    cleanup_directory: Option<Dir>,
}

impl OwnedLeaf {
    #[cfg(test)]
    fn create(dir: &Dir, name: impl AsRef<Path>) -> std::io::Result<Self> {
        Self::create_transaction(dir, name)
    }

    fn open(dir: &Dir, name: impl AsRef<Path>) -> std::io::Result<Self> {
        let name = name.as_ref().as_os_str().to_owned();
        let mut options = OpenOptions::new();
        options.read(true).follow(FollowSymlinks::No);
        #[cfg(windows)]
        {
            use cap_std::fs::OpenOptionsExt;
            options.share_mode(0x1 | 0x2);
        }
        let file = dir.open_with(&name, &options)?;
        if !file.metadata()?.is_file() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "library leaf is not a regular file",
            ));
        }
        let handle = Handle::from_file(file.into_std())?;
        Ok(Self {
            name,
            handle,
            cleanup_directory: None,
        })
    }

    fn open_writable(dir: &Dir, name: impl AsRef<Path>) -> std::io::Result<Self> {
        let name = name.as_ref().as_os_str().to_owned();
        let mut options = OpenOptions::new();
        options.read(true).write(true).follow(FollowSymlinks::No);
        #[cfg(windows)]
        {
            use cap_std::fs::OpenOptionsExt;
            options.share_mode(0x1 | 0x2);
        }
        let file = dir.open_with(&name, &options)?;
        if !file.metadata()?.is_file() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "library leaf is not a regular file",
            ));
        }
        Ok(Self {
            name,
            handle: Handle::from_file(file.into_std())?,
            cleanup_directory: None,
        })
    }

    fn create_transaction(dir: &Dir, name: impl AsRef<Path>) -> std::io::Result<Self> {
        let cleanup_directory = dir.try_clone()?;
        let name = name.as_ref().as_os_str().to_owned();
        let mut options = OpenOptions::new();
        options
            .read(true)
            .write(true)
            .create_new(true)
            .follow(FollowSymlinks::No);
        #[cfg(windows)]
        {
            use cap_std::fs::OpenOptionsExt;
            use windows_sys::Win32::Foundation::{GENERIC_READ, GENERIC_WRITE};
            use windows_sys::Win32::Storage::FileSystem::{
                DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE,
            };
            options
                .access_mode(GENERIC_READ | GENERIC_WRITE | DELETE)
                .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE);
        }
        let file = dir.open_with(&name, &options)?;
        Ok(Self {
            name,
            handle: Handle::from_file(file.into_std())?,
            cleanup_directory: Some(cleanup_directory),
        })
    }

    fn open_transaction(dir: &Dir, name: impl AsRef<Path>) -> std::io::Result<Self> {
        let name = name.as_ref().as_os_str().to_owned();
        let mut options = OpenOptions::new();
        options.read(true).write(true).follow(FollowSymlinks::No);
        #[cfg(unix)]
        {
            use cap_std::fs::OpenOptionsExt;
            // Inspect an unexpected special leaf without waiting on a FIFO.
            options.custom_flags(libc::O_NONBLOCK);
        }
        #[cfg(windows)]
        {
            use cap_std::fs::OpenOptionsExt;
            use windows_sys::Win32::Foundation::{GENERIC_READ, GENERIC_WRITE};
            use windows_sys::Win32::Storage::FileSystem::{
                DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE,
            };
            options
                .access_mode(GENERIC_READ | GENERIC_WRITE | DELETE)
                .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE);
        }
        let file = dir.open_with(&name, &options)?;
        if !file.metadata()?.is_file() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "library transaction leaf is not a regular file",
            ));
        }
        Ok(Self {
            name,
            handle: Handle::from_file(file.into_std())?,
            cleanup_directory: None,
        })
    }

    fn open_identity(dir: &Dir, name: impl AsRef<Path>) -> std::io::Result<Self> {
        let name = name.as_ref().as_os_str().to_owned();
        let mut options = OpenOptions::new();
        options.read(true).follow(FollowSymlinks::No);
        #[cfg(windows)]
        {
            use cap_std::fs::OpenOptionsExt;
            use windows_sys::Win32::Storage::FileSystem::{
                FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE,
            };
            options.share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE);
        }
        let file = dir.open_with(&name, &options)?;
        if !file.metadata()?.is_file() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "library identity leaf is not a regular file",
            ));
        }
        Ok(Self {
            name,
            handle: Handle::from_file(file.into_std())?,
            cleanup_directory: None,
        })
    }

    fn matches_name(&self, dir: &Dir) -> std::io::Result<bool> {
        let current = Self::open_identity(dir, &self.name)?;
        Ok(self.handle == current.handle)
    }

    fn rewind(&mut self) -> std::io::Result<()> {
        self.handle.as_file_mut().seek(SeekFrom::Start(0))?;
        Ok(())
    }

    fn sync_all(&self) -> std::io::Result<()> {
        self.handle.as_file().sync_all()
    }

    fn truncate_exact(&self) -> std::io::Result<()> {
        self.handle.as_file().set_len(0)?;
        self.handle.as_file().sync_all()
    }

    fn disarm_cleanup(&mut self) {
        self.cleanup_directory = None;
    }
}

impl Drop for OwnedLeaf {
    fn drop(&mut self) {
        if let Some(directory) = &self.cleanup_directory {
            if let Err(error) = destroy_owned_leaf(directory, self) {
                tracing::warn!(%error, "library: failed to clean an owned temporary leaf");
            }
        }
    }
}

fn destroy_owned_leaf(directory: &Dir, leaf: &OwnedLeaf) -> std::io::Result<()> {
    let truncate = if link_count(leaf.handle.as_file())? <= 1 {
        leaf.truncate_exact()
    } else {
        Ok(())
    };
    let remove = unlink_owned_leaf(directory, leaf).and_then(|()| sync_dir(directory));
    match (truncate, remove) {
        (Err(a), Err(b)) => Err(std::io::Error::other(format!("truncate: {a}; unlink: {b}"))),
        (Err(error), _) | (_, Err(error)) => Err(error),
        (Ok(()), Ok(())) => Ok(()),
    }
}

#[cfg(unix)]
fn unlink_owned_leaf(directory: &Dir, leaf: &OwnedLeaf) -> std::io::Result<()> {
    use std::os::fd::AsRawFd;
    use std::os::unix::ffi::OsStrExt;
    let name = std::ffi::CString::new(leaf.name.as_bytes())?;
    let mut retained = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: the owned file descriptor and writable stat buffer stay live.
    if unsafe { libc::fstat(leaf.handle.as_file().as_raw_fd(), retained.as_mut_ptr()) } < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: successful fstat initialized the complete stat buffer.
    let retained = unsafe { retained.assume_init() };
    let mut visible = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: the retained directory and single-component name are live;
    // nofollow inspects the entry itself and the output buffer is writable.
    if unsafe {
        libc::fstatat(
            directory.as_raw_fd(),
            name.as_ptr(),
            visible.as_mut_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    } < 0
    {
        let error = std::io::Error::last_os_error();
        return if error.kind() == std::io::ErrorKind::NotFound {
            Ok(())
        } else {
            Err(error)
        };
    }
    // SAFETY: successful fstatat initialized the stat buffer.
    let visible = unsafe { visible.assume_init() };
    if visible.st_dev != retained.st_dev || visible.st_ino != retained.st_ino {
        return Ok(());
    }
    // SAFETY: both authorities remain retained; only the identity-matched
    // leaf is removed, never a rebound pathname or another hard link's bytes.
    if unsafe { libc::unlinkat(directory.as_raw_fd(), name.as_ptr(), 0) } < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(windows)]
fn unlink_owned_leaf(_directory: &Dir, leaf: &OwnedLeaf) -> std::io::Result<()> {
    delete_transaction_leaf_by_handle(leaf)
}

#[cfg(not(any(unix, windows)))]
fn unlink_owned_leaf(_directory: &Dir, _leaf: &OwnedLeaf) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "retained library cleanup unavailable",
    ))
}

fn stream_hash_copy(reader: &mut impl Read, writer: &mut impl Write) -> std::io::Result<String> {
    #[cfg(test)]
    if WRITE_LOCK_HELD.with(std::cell::Cell::get) {
        STREAMS_UNDER_WRITE_LOCK.with(|count| count.set(count.get() + 1));
    }
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; STREAM_BUFFER_SIZE];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        writer.write_all(&buffer[..read])?;
    }
    Ok(hex_digest(hasher.finalize().as_slice()))
}

fn stream_hash(reader: &mut impl Read) -> std::io::Result<String> {
    #[cfg(test)]
    STREAM_HASH_CALLS.with(|count| count.set(count.get() + 1));
    stream_hash_copy(reader, &mut std::io::sink())
}

/// Number of hard links to the open file.
#[cfg(unix)]
fn link_count(file: &std::fs::File) -> std::io::Result<u64> {
    use std::os::unix::fs::MetadataExt;
    Ok(file.metadata()?.nlink())
}

/// Number of hard links to the open file.
#[cfg(windows)]
fn link_count(file: &std::fs::File) -> std::io::Result<u64> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        GetFileInformationByHandle, BY_HANDLE_FILE_INFORMATION,
    };
    let mut info = BY_HANDLE_FILE_INFORMATION::default();
    // SAFETY: `info` is a properly sized, writable BY_HANDLE_FILE_INFORMATION
    // and the file handle stays open for this synchronous call.
    let ok = unsafe { GetFileInformationByHandle(file.as_raw_handle(), &mut info) };
    if ok == 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(u64::from(info.nNumberOfLinks))
}

/// Make a rename inside `dir` durable before anything refers to it. Windows
/// (NTFS) journals directory updates and has no directory fsync.
fn sync_dir(dir: &Dir) -> std::io::Result<()> {
    // The capability itself may be a path-only descriptor that cannot be
    // synced; open the directory for reading through it instead.
    #[cfg(unix)]
    dir.open(".")?.sync_all()?;
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

fn read_nofollow(dir: &Dir, name: impl AsRef<Path>) -> std::io::Result<Vec<u8>> {
    let mut options = OpenOptions::new();
    options.read(true).follow(FollowSymlinks::No);
    let mut file = dir.open_with(name, &options)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn unique_atomic_artifact(target: &OsStr, suffix: &str) -> OsString {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let sequence = COUNTER.fetch_add(1, Ordering::Relaxed);
    OsString::from(format!(
        ".{}.{}.{sequence:020}.{suffix}",
        target.to_string_lossy(),
        std::process::id()
    ))
}

fn unique_manifest_artifact(suffix: &str) -> OsString {
    unique_atomic_artifact(OsStr::new(MANIFEST_NAME), suffix)
}

fn is_manifest_backup(name: &OsStr) -> bool {
    let text = name.to_string_lossy();
    text.starts_with(&format!(".{MANIFEST_NAME}.")) && text.ends_with(".backup")
}

fn rename_owned(root: &Dir, leaf: &mut OwnedLeaf, target: &Path) -> std::io::Result<()> {
    if !leaf.matches_name(root)? {
        return Err(std::io::Error::other(
            "library leaf identity changed before rename",
        ));
    }
    #[cfg(not(windows))]
    root.rename(&leaf.name, root, target)?;
    #[cfg(windows)]
    rename_transaction_leaf_by_handle(root, leaf, target, false)?;
    leaf.name = target.as_os_str().to_owned();
    if !leaf.matches_name(root)? {
        return Err(std::io::Error::other(
            "library leaf identity changed during rename",
        ));
    }
    Ok(())
}

/// Move a retained leaf from one directory capability to another on the same
/// filesystem without replacing an existing target on Windows. The leaf's
/// identity is checked on both sides; if the name was swapped underneath us
/// the foreign file that landed on `target` is moved back out of `to` (best
/// effort) and the move fails, so it can never be published as ours.
fn move_owned(from: &Dir, leaf: &mut OwnedLeaf, to: &Dir, target: &Path) -> std::io::Result<()> {
    if !leaf.matches_name(from)? {
        return Err(std::io::Error::other(
            "library leaf identity changed before move",
        ));
    }
    let cleanup_directory = leaf
        .cleanup_directory
        .as_ref()
        .map(|_| to.try_clone())
        .transpose()?;
    #[cfg(not(windows))]
    from.rename(&leaf.name, to, target)?;
    #[cfg(windows)]
    rename_transaction_leaf_by_handle(to, leaf, target, false)?;
    let source_name = std::mem::replace(&mut leaf.name, target.as_os_str().to_owned());
    leaf.cleanup_directory = cleanup_directory;
    if !leaf.matches_name(to)? {
        let quarantine = unique_atomic_artifact(&source_name, "quarantine");
        let _ = to.rename(target, from, Path::new(&quarantine));
        return Err(std::io::Error::other(
            "library leaf identity changed during move",
        ));
    }
    Ok(())
}

fn replace_owned(root: &Dir, leaf: &mut OwnedLeaf, target: &Path) -> std::io::Result<()> {
    if !leaf.matches_name(root)? {
        return Err(std::io::Error::other(
            "atomic leaf identity changed before replacement",
        ));
    }
    #[cfg(not(windows))]
    root.rename(&leaf.name, root, target)?;
    #[cfg(windows)]
    rename_transaction_leaf_by_handle(root, leaf, target, true)?;
    // The single replace syscall is the commit point. From here onward no
    // fallible diagnostic may convert a durable replacement into `Err`, because
    // callers restore their live state on any writer error.
    leaf.name = target.as_os_str().to_owned();
    Ok(())
}

#[cfg(windows)]
fn rename_transaction_leaf_by_handle(
    root: &Dir,
    leaf: &OwnedLeaf,
    target: &Path,
    replace_existing: bool,
) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Wdk::Storage::FileSystem::{
        FileRenameInformation, NtSetInformationFile, FILE_RENAME_INFORMATION,
        FILE_RENAME_INFORMATION_0,
    };
    use windows_sys::Win32::Foundation::RtlNtStatusToDosError;
    use windows_sys::Win32::System::IO::IO_STATUS_BLOCK;

    let mut components = target.components();
    let target_name = match (components.next(), components.next()) {
        (Some(std::path::Component::Normal(name)), None) => name,
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "manifest transaction target must be one relative leaf",
            ))
        }
    };
    let wide: Vec<u16> = target_name.encode_wide().collect();
    if wide.is_empty() || wide.contains(&0) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "manifest transaction target is empty or contains NUL",
        ));
    }
    let file_name_bytes = wide
        .len()
        .checked_mul(std::mem::size_of::<u16>())
        .and_then(|size| u32::try_from(size).ok())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "manifest transaction target is too long",
            )
        })?;
    // NtSetInformationFile requires a complete FILE_RENAME_INFORMATION header
    // followed by the variable-length UTF-16 name. Keep the SDK-sized trailing
    // FileName element/padding instead of passing only offset + payload.
    let info_size = std::mem::size_of::<FILE_RENAME_INFORMATION>()
        .checked_add(file_name_bytes as usize)
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "manifest rename allocation overflow",
            )
        })?;
    const _: () =
        assert!(std::mem::align_of::<usize>() >= std::mem::align_of::<FILE_RENAME_INFORMATION>());
    let word_size = std::mem::size_of::<usize>();
    let mut storage = vec![0_usize; info_size.div_ceil(word_size)];
    let info = storage.as_mut_ptr().cast::<FILE_RENAME_INFORMATION>();
    let info_size = u32::try_from(info_size).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "manifest rename buffer is too large",
        )
    })?;
    // SAFETY: `storage` is pointer-aligned and sized for the fixed
    // FILE_RENAME_INFORMATION header plus every UTF-16 code unit. Both raw
    // handles remain owned and open for the duration of the synchronous NT
    // call. The native contract explicitly models the retained RootDirectory
    // used here for a capability-relative leaf rename.
    // OwnedLeaf is opened without FILE_FLAG_OVERLAPPED, so this call completes
    // synchronously and cannot outlive the stack-owned IO_STATUS_BLOCK.
    let status = unsafe {
        (*info).Anonymous = FILE_RENAME_INFORMATION_0 {
            ReplaceIfExists: replace_existing,
        };
        (*info).RootDirectory = root.as_raw_handle();
        (*info).FileNameLength = file_name_bytes;
        std::ptr::copy_nonoverlapping(
            wide.as_ptr(),
            std::ptr::addr_of_mut!((*info).FileName).cast::<u16>(),
            wide.len(),
        );
        let mut io_status = IO_STATUS_BLOCK::default();
        NtSetInformationFile(
            leaf.handle.as_file().as_raw_handle(),
            &mut io_status,
            info.cast(),
            info_size,
            FileRenameInformation,
        )
    };
    if status < 0 {
        // SAFETY: converting the NTSTATUS returned by the immediately preceding
        // native call does not dereference memory or consume either live handle.
        let error = unsafe { RtlNtStatusToDosError(status) };
        return Err(std::io::Error::from_raw_os_error(error as i32));
    }
    Ok(())
}

fn commit_atomic_file(root: &Dir, target: &Path, tmp: &mut OwnedLeaf) -> std::io::Result<()> {
    let target_name = match target.components().collect::<Vec<_>>().as_slice() {
        [std::path::Component::Normal(name)] => *name,
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "atomic target must be one relative leaf",
            ))
        }
    };
    let mut backup = match OwnedLeaf::open_transaction(root, target) {
        Ok(mut canonical) => {
            let backup_name = unique_atomic_artifact(target_name, "backup");
            rename_owned(root, &mut canonical, Path::new(&backup_name))?;
            Some(canonical)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error),
    };
    if let Err(error) = rename_owned(root, tmp, target) {
        if let Some(backup) = backup.as_mut() {
            if let Err(restore) = rename_owned(root, backup, target) {
                return Err(std::io::Error::other(format!(
                    "{error}; atomic backup restore failed: {restore}"
                )));
            }
        }
        return Err(error);
    }
    if !tmp.matches_name(root)? {
        if let Ok(mut replacement) = OwnedLeaf::open_transaction(root, target) {
            let quarantine = unique_atomic_artifact(target_name, "quarantine");
            let _ = rename_owned(root, &mut replacement, Path::new(&quarantine));
        }
        if let Some(backup) = backup.as_mut() {
            let _ = rename_owned(root, backup, target);
        }
        return Err(std::io::Error::other(
            "atomic target identity changed during commit",
        ));
    }
    // The canonical rename is now verified. Disarm it before backup cleanup so
    // a cleanup error can never truncate the newly committed manifest on drop.
    tmp.disarm_cleanup();
    // The new name must be durable before deleting the last recovery copy.
    // The logical commit already happened: retain the backup and report the
    // flush failure without making the caller discard committed media.
    if let Err(error) = sync_committed_manifest_directory(root) {
        tracing::warn!(%error, "library: keeping backup after manifest directory sync failed");
        return Ok(());
    }
    if let Some(backup) = backup.as_ref() {
        // The commit point has passed. Cleanup must never turn a successful
        // publication into an error: the caller could otherwise discard the
        // newly manifest-owned content while the manifest is already durable.
        if let Err(error) = cleanup_committed_backup(root, backup) {
            tracing::warn!(%error, "library: committed backup cleanup failed");
        }
    }
    Ok(())
}

/// Atomically replace one relative leaf through a retained directory
/// capability. Publication is one rename operation, so a crash cannot strand
/// the canonical name between backup and publish. On Windows the retained temp
/// handle carries DELETE access and uses native `FILE_RENAME_INFORMATION`
/// with `ReplaceIfExists`.
pub fn write_atomic_capability_file(
    root: &Dir,
    target: impl AsRef<Path>,
    bytes: &[u8],
) -> std::io::Result<()> {
    let target = target.as_ref();
    let target_name = match target.components().collect::<Vec<_>>().as_slice() {
        [std::path::Component::Normal(name)] => *name,
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "atomic target must be one relative leaf",
            ))
        }
    };
    let tmp_name = unique_atomic_artifact(target_name, "tmp");
    let mut tmp = OwnedLeaf::create_transaction(root, &tmp_name)?;
    tmp.handle.as_file_mut().write_all(bytes)?;
    tmp.sync_all()?;
    #[cfg(any(test, feature = "test-faults"))]
    if FAIL_ATOMIC_CAPABILITY_REPLACE.with(|fail| fail.replace(false)) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "injected atomic capability replacement failure",
        ));
    }
    replace_owned(root, &mut tmp, target)?;
    tmp.disarm_cleanup();
    Ok(())
}

fn sync_committed_manifest_directory(root: &Dir) -> std::io::Result<()> {
    #[cfg(test)]
    if FAIL_MANIFEST_DIRECTORY_SYNC.with(|fail| fail.replace(false)) {
        return Err(std::io::Error::other(
            "injected manifest directory sync failure",
        ));
    }
    sync_dir(root)
}

fn cleanup_committed_backup(root: &Dir, backup: &OwnedLeaf) -> std::io::Result<()> {
    #[cfg(test)]
    if FAIL_COMMITTED_BACKUP_CLEANUP.with(|fail| fail.replace(false)) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "injected committed-backup cleanup failure",
        ));
    }
    // Unlink the old version without truncating a snapshot that a reader
    // already holds, or bytes reachable through a hard link.
    unlink_owned_leaf(root, backup)?;
    sync_dir(root)
}

#[cfg(windows)]
fn delete_transaction_leaf_by_handle(leaf: &OwnedLeaf) -> std::io::Result<()> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        FileDispositionInfo, SetFileInformationByHandle, FILE_DISPOSITION_INFO,
    };

    let disposition = FILE_DISPOSITION_INFO { DeleteFile: true };
    // SAFETY: `disposition` has the exact ABI type/size required by
    // FileDispositionInfo, and the retained transaction handle stays open for
    // the duration of this synchronous call.
    let deleted = unsafe {
        SetFileInformationByHandle(
            leaf.handle.as_file().as_raw_handle(),
            FileDispositionInfo,
            std::ptr::addr_of!(disposition).cast(),
            u32::try_from(std::mem::size_of::<FILE_DISPOSITION_INFO>())
                .expect("FILE_DISPOSITION_INFO size fits u32"),
        )
    };
    if deleted == 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

impl LibraryStore {
    /// Open (or lazily create) a store rooted at `root`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let capabilities = open_capabilities(&root)
            .map(Arc::new)
            .map_err(|error| format!("could not secure global library root: {error}"));
        LibraryStore {
            root,
            capabilities,
            active_stages: Arc::new(Mutex::new(HashSet::new())),
            write_lock: Mutex::new(()),
        }
    }

    /// Open a store at the platform-default library directory.
    pub fn open_default() -> Result<Self> {
        let root = default_library_dir().ok_or_else(|| {
            MediaError::Other(anyhow::anyhow!("could not resolve platform data directory"))
        })?;
        let store = LibraryStore::new(root);
        store.reconcile_storage()?;
        Ok(store)
    }

    /// The library root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn capabilities(&self) -> Result<&Arc<LibraryCapabilities>> {
        self.capabilities
            .as_ref()
            .map_err(|message| MediaError::Other(anyhow::anyhow!(message.clone())))
    }

    fn files_dir(&self) -> PathBuf {
        self.root.join(FILES_SUBDIR)
    }

    fn manifest_backups(&self) -> Result<Vec<OsString>> {
        let root = &self.capabilities()?.root;
        let mut backups = Vec::new();
        for entry in root.entries()? {
            let entry = entry?;
            let name = entry.file_name();
            if !is_manifest_backup(&name) {
                continue;
            }
            let file_type = entry.file_type()?;
            if !file_type.is_file() || file_type.is_symlink() {
                return Err(MediaError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "library manifest backup is not a nofollow regular file",
                )));
            }
            backups.push(name);
        }
        backups.sort();
        Ok(backups)
    }

    fn latest_valid_manifest_backup(&self) -> Result<Option<(OwnedLeaf, Manifest)>> {
        let root = &self.capabilities()?.root;
        let mut latest_error = None;
        for backup in self.manifest_backups()?.into_iter().rev() {
            let mut leaf = OwnedLeaf::open_transaction(root, &backup)?;
            let mut bytes = Vec::new();
            leaf.handle.as_file_mut().read_to_end(&mut bytes)?;
            match decode_manifest(&bytes) {
                Ok(manifest) => return Ok(Some((leaf, manifest))),
                Err(error) if latest_error.is_none() => latest_error = Some(error),
                Err(_) => {}
            }
        }
        match latest_error {
            Some(error) => Err(error),
            None => Ok(None),
        }
    }

    fn reconcile_manifest_artifacts(&self) -> Result<()> {
        let root = &self.capabilities()?.root;
        match root.symlink_metadata(MANIFEST_NAME) {
            Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => {
                // A regular canonical leaf can still be corrupt. Validate it
                // before removing any crash-recovery backup.
                let bytes = read_nofollow(root, MANIFEST_NAME)?;
                decode_manifest(&bytes)?;
                if let Ok(backups) = self.manifest_backups() {
                    for backup in backups {
                        if let Ok(backup) = OwnedLeaf::open_transaction(root, backup) {
                            if let Err(error) = cleanup_committed_backup(root, &backup) {
                                tracing::warn!(%error, "library: recovered backup cleanup failed");
                            }
                        }
                    }
                }
            }
            Ok(_) => {
                return Err(MediaError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "library manifest is not a nofollow regular file",
                )))
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                if let Some((mut backup, _manifest)) = self.latest_valid_manifest_backup()? {
                    rename_owned(root, &mut backup, Path::new(MANIFEST_NAME))?;
                }
            }
            Err(error) => return Err(MediaError::Io(error)),
        }
        Ok(())
    }

    fn stored_name(id: &str, source: &Path) -> OsString {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let sequence = COUNTER.fetch_add(1, Ordering::Relaxed);
        let safe_extension = source.extension().and_then(OsStr::to_str).filter(|value| {
            !value.is_empty()
                && value.len() <= 16
                && value.bytes().all(|byte| byte.is_ascii_alphanumeric())
        });
        match safe_extension {
            Some(extension) => OsString::from(format!(
                "{id}.{}.{sequence}.{extension}",
                std::process::id()
            )),
            None => OsString::from(format!("{id}.{}.{sequence}", std::process::id())),
        }
    }

    fn content_id_from_name(name: &OsStr) -> Option<String> {
        let text = name.to_str()?;
        let id = text.get(..64)?;
        if text.len() > 64 && text.as_bytes().get(64) != Some(&b'.') {
            return None;
        }
        let valid = id
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
        valid.then(|| id.to_string())
    }

    fn staging_name(source: &Path) -> OsString {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let sequence = COUNTER.fetch_add(1, Ordering::Relaxed);
        let extension = source
            .extension()
            .and_then(OsStr::to_str)
            .filter(|value| value.bytes().all(|byte| byte.is_ascii_alphanumeric()))
            .unwrap_or("bin");
        OsString::from(format!(
            ".incoming.{}.{}.{extension}.pending",
            std::process::id(),
            sequence
        ))
    }

    fn stored_index(&self) -> Result<StoredIndex> {
        #[cfg(test)]
        STORED_INDEX_SCANS.with(|count| count.set(count.get() + 1));
        let capabilities = self.capabilities()?;
        let mut by_id: HashMap<String, Vec<OsString>> = HashMap::new();
        for entry in capabilities.files.entries()? {
            let entry = entry?;
            let name = entry.file_name();
            if name.to_string_lossy().starts_with('.') {
                // `.staging`, Finder's `.DS_Store`, and other hidden metadata.
                continue;
            }
            let file_type = entry.file_type()?;
            if !file_type.is_file() || file_type.is_symlink() {
                tracing::debug!(
                    "library: ignoring non-regular entry in files/: {}",
                    name.to_string_lossy()
                );
                continue;
            }
            let Some(id) = Self::content_id_from_name(&name) else {
                tracing::debug!(
                    "library: ignoring foreign file in files/: {}",
                    name.to_string_lossy()
                );
                continue;
            };
            if entry.metadata()?.len() == 0 {
                continue;
            }
            by_id.entry(id).or_default().push(name);
        }
        for names in by_id.values_mut() {
            names.sort();
        }
        Ok(StoredIndex { by_id })
    }

    fn open_stored_leaf(&self, id: &str) -> Result<Option<OwnedLeaf>> {
        let index = self.stored_index()?;
        let Some(name) = index.lookup(id)? else {
            return Ok(None);
        };
        let capabilities = self.capabilities()?;
        Ok(Some(OwnedLeaf::open(&capabilities.files, name)?))
    }

    /// Hash every leaf claiming `id` through its own read-only handle.
    /// Mismatching leaves (bit rot, external edits, a crash-truncated write)
    /// are quarantined with [`Self::release_stored_leaf`], which hides them
    /// from [`Self::stored_index`]; they are never adopted. Returns the
    /// verified leaves, rewound, in name order.
    fn verify_candidates(&self, id: &str, names: &[OsString]) -> Result<Vec<OwnedLeaf>> {
        let files = &self.capabilities()?.files;
        let mut verified = Vec::new();
        for name in names {
            let mut leaf = match OwnedLeaf::open(files, name) {
                Ok(leaf) => leaf,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => return Err(error.into()),
            };
            let actual = stream_hash(leaf.handle.as_file_mut())?;
            if actual == id {
                leaf.rewind()?;
                verified.push(leaf);
            } else {
                tracing::warn!(
                    "library: quarantining stored copy {} (expected {id}, got {actual})",
                    name.to_string_lossy()
                );
                self.release_stored_leaf(leaf)?;
            }
        }
        Ok(verified)
    }

    /// Keep the first verified leaf and release every other one, so exactly
    /// one name claims the id afterwards.
    fn keep_first_verified(&self, id: &str, verified: Vec<OwnedLeaf>) -> Result<Option<OwnedLeaf>> {
        let mut verified = verified.into_iter();
        let kept = verified.next();
        for duplicate in verified {
            tracing::warn!(
                "library: releasing duplicate stored copy {} of {id}",
                duplicate.name.to_string_lossy()
            );
            self.release_stored_leaf(duplicate)?;
        }
        Ok(kept)
    }

    /// Stop `leaf` from claiming its content id without destroying data that
    /// is reachable elsewhere. A file with a single link is truncated and unlinked through
    /// a handle proven to be the same file (empty leaves are ignored by
    /// [`Self::stored_index`]). A file with several links — two names in
    /// `files/` for one file, or a hard link from outside the library — only
    /// loses this directory entry, so the other links keep their bytes. A name
    /// that no longer refers to the verified file is left alone.
    fn release_stored_leaf(&self, leaf: OwnedLeaf) -> Result<()> {
        let files = &self.capabilities()?.files;
        if link_count(leaf.handle.as_file())? <= 1 {
            match OwnedLeaf::open_writable(files, &leaf.name) {
                Ok(writable) if writable.handle == leaf.handle => {
                    writable.truncate_exact()?;
                }
                Ok(_) => return Ok(()),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
                // Not writable (for example a read-only file): unlink instead.
                Err(_) => {}
            }
        }
        match OwnedLeaf::open_identity(files, &leaf.name) {
            Ok(current) if current.handle == leaf.handle => {}
            Ok(_) => return Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error.into()),
        }
        let name = leaf.name.clone();
        // Windows cannot remove a name while this process holds the file
        // without delete sharing.
        #[cfg(windows)]
        drop(leaf);
        match files.remove_file(&name) {
            Ok(()) => sync_dir(files).map_err(Into::into),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
        }
    }

    /// Validate all retained stored leaves once and return their content ids.
    /// The result carries no path authority; consumers must reopen through this
    /// store and verify the exact handle they read.
    pub fn stored_ids_verified(&self) -> Result<HashSet<String>> {
        let index = self.stored_index()?;
        let manifest = self.load_manifest()?;
        let mut ids = HashSet::with_capacity(manifest.entries.len());
        for entry in manifest.entries {
            let id = entry.id;
            let Some(name) = index.lookup(&id)? else {
                continue;
            };
            let mut leaf = OwnedLeaf::open(&self.capabilities()?.files, name)?;
            let actual = stream_hash(leaf.handle.as_file_mut())?;
            if actual != id {
                return Err(MediaError::Other(anyhow::anyhow!(
                    "stored library content hash mismatch: expected {id}, got {actual}"
                )));
            }
            ids.insert(id);
        }
        Ok(ids)
    }

    /// Verify only content ids referenced by the current project. Listing the
    /// directory is still required to retain no-follow file authority, but
    /// unrelated library files are never opened or hashed during project open.
    pub fn stored_ids_verified_for(
        &self,
        requested: &HashSet<String>,
    ) -> Result<(HashSet<String>, HashMap<String, String>)> {
        if requested.is_empty() {
            return Ok((HashSet::new(), HashMap::new()));
        }
        let index = self.stored_index()?;
        let capabilities = self.capabilities()?;
        let mut ids = HashSet::with_capacity(requested.len());
        let mut failures = HashMap::new();
        for id in requested {
            let name = match index.lookup(id) {
                Ok(Some(name)) => name,
                Ok(None) => continue,
                Err(error) => {
                    failures.insert(id.clone(), error.to_string());
                    continue;
                }
            };
            let verified = (|| -> Result<()> {
                let mut leaf = OwnedLeaf::open(&capabilities.files, name)?;
                let actual = stream_hash(leaf.handle.as_file_mut())?;
                if actual != *id {
                    return Err(MediaError::Other(anyhow::anyhow!(
                        "stored library content hash mismatch: expected {id}, got {actual}"
                    )));
                }
                Ok(())
            })();
            match verified {
                Ok(()) => {
                    ids.insert(id.clone());
                }
                Err(error) => {
                    failures.insert(id.clone(), error.to_string());
                }
            }
        }
        Ok((ids, failures))
    }

    /// Stream one verified retained library leaf into `writer`. The source is
    /// opened no-follow through the retained `files` capability and its exact
    /// bytes must hash to `id`; no ambient library path is returned.
    pub fn copy_stored_verified(
        &self,
        id: &str,
        writer: &mut impl Write,
    ) -> Result<Option<OsString>> {
        let Some(mut leaf) = self.open_stored_leaf(id)? else {
            return Ok(None);
        };
        let actual = stream_hash_copy(leaf.handle.as_file_mut(), writer)?;
        if actual != id {
            return Err(MediaError::Other(anyhow::anyhow!(
                "stored library content changed while copying: expected {id}, got {actual}"
            )));
        }
        Ok(Some(leaf.name.clone()))
    }

    /// Return only the retained directory entry name for extension/display
    /// decisions. This is not path authority; callers must consume bytes via
    /// [`Self::copy_stored_verified`].
    pub fn stored_file_name(&self, id: &str) -> Result<Option<OsString>> {
        Ok(self.stored_index()?.lookup(id)?.cloned())
    }

    /// Validate storage after a crash while leaving unknown mutable names
    /// untouched. A strictly valid manifest is required first; unknown leaves
    /// (staging leftovers, foreign files such as `.DS_Store`, directories)
    /// stay hidden and never make the library unavailable. Known empty stored
    /// leaves are reclaimed without reading media payloads; see
    /// [`Self::resolve_duplicate_copies`] for duplicate claims.
    pub fn reconcile_storage(&self) -> Result<()> {
        let _guard = self.lock_writes();
        self.reconcile_manifest_artifacts()?;
        // A strictly valid manifest is required before anything else.
        self.load_manifest()?;
        let capabilities = self.capabilities()?;
        let mut cleanup_needs_sync = false;
        for entry in capabilities.files.entries()? {
            let entry = entry?;
            let name = entry.file_name();
            let file_type = entry.file_type()?;
            if Self::content_id_from_name(&name).is_none()
                || !file_type.is_file()
                || file_type.is_symlink()
                || entry.metadata()?.len() != 0
            {
                continue;
            }
            let cleanup = (|| -> std::io::Result<bool> {
                let leaf = OwnedLeaf::open_transaction(&capabilities.files, &name)?;
                if leaf.handle.as_file().metadata()?.len() == 0 {
                    unlink_owned_leaf(&capabilities.files, &leaf)?;
                    return Ok(true);
                }
                Ok(false)
            })();
            match cleanup {
                Ok(removed) => cleanup_needs_sync |= removed,
                Err(error) => {
                    tracing::warn!(%error, "library: could not reclaim an empty stored leaf")
                }
            }
        }
        if cleanup_needs_sync {
            if let Err(error) = sync_dir(&capabilities.files) {
                tracing::warn!(%error, "library: could not sync reclaimed stored leaves");
            }
        }
        let active_stages = self
            .active_stages
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        for entry in capabilities.staging.entries()? {
            let entry = entry?;
            let name = entry.file_name();
            if active_stages.contains(&name) {
                continue;
            }
            let file_type = entry.file_type()?;
            if !file_type.is_file() || file_type.is_symlink() {
                tracing::warn!(
                    "library: ignoring unexpected staging entry {}",
                    name.to_string_lossy()
                );
            }
            // Unknown startup leaves stay hidden. Reopening by this mutable
            // name for cleanup could target a replacement installed after the
            // directory enumeration, so reconciliation performs no destructive
            // action without an already-retained owner handle.
        }
        Ok(())
    }

    /// Resolve manifest-owned ids that several leaves claim (crash leftovers,
    /// copies restored by hand, hard links). Each candidate is hashed through
    /// its own handle: mismatching ones are quarantined and verified extras
    /// released, so lookups for those ids work again. This hashes whole
    /// files, so it runs without the write lock and is meant for a background
    /// thread after startup; favoriting such content resolves it the same way.
    pub fn resolve_duplicate_copies(&self) -> Result<usize> {
        let (manifest, index) = {
            let _guard = self.lock_writes();
            (self.load_manifest()?, self.stored_index()?)
        };
        let mut resolved = 0;
        for entry in &manifest.entries {
            let candidates = index.candidates(&entry.id);
            if candidates.len() < 2 {
                continue;
            }
            let verified = self.verify_candidates(&entry.id, candidates)?;
            self.keep_first_verified(&entry.id, verified)?;
            resolved += 1;
        }
        Ok(resolved)
    }

    fn lock_writes(&self) -> WriteLockGuard<'_> {
        WriteLockGuard::acquire(&self.write_lock)
    }

    /// Read the manifest, returning an empty one if it does not exist yet.
    fn load_manifest(&self) -> Result<Manifest> {
        let root = &self.capabilities()?.root;
        match read_nofollow(root, MANIFEST_NAME) {
            Ok(bytes) => decode_manifest(&bytes),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                if let Some((_backup, manifest)) = self.latest_valid_manifest_backup()? {
                    Ok(manifest)
                } else {
                    Ok(Manifest {
                        version: MANIFEST_VERSION,
                        entries: Vec::new(),
                    })
                }
            }
            Err(e) => Err(MediaError::Io(e)),
        }
    }

    /// Atomically persist the manifest: write a temp file, then rename over the
    /// real path. The rename is atomic on the same filesystem.
    fn store_manifest(&self, manifest: &Manifest) -> Result<()> {
        let root = &self.capabilities()?.root;
        let bytes = serde_json::to_vec_pretty(manifest)?;
        let tmp_name = unique_manifest_artifact("tmp");
        let mut tmp = OwnedLeaf::create_transaction(root, &tmp_name)?;
        tmp.handle.as_file_mut().write_all(&bytes)?;
        tmp.sync_all()?;
        commit_atomic_file(root, Path::new(MANIFEST_NAME), &mut tmp)?;
        tmp.disarm_cleanup();
        Ok(())
    }

    /// All entries currently in the library (manifest order).
    pub fn entries(&self) -> Result<Vec<LibraryEntry>> {
        Ok(self.load_manifest()?.entries)
    }

    /// Entries filtered by `category`. `Some(c)` keeps entries whose category
    /// equals `c`; `None` keeps only uncategorized entries.
    pub fn entries_in_category(&self, category: Option<&str>) -> Result<Vec<LibraryEntry>> {
        let want = category.map(|c| c.to_string());
        Ok(self
            .load_manifest()?
            .entries
            .into_iter()
            .filter(|e| e.category == want)
            .collect())
    }

    /// Whether an entry with this content id already exists.
    pub fn contains(&self, id: &str) -> Result<bool> {
        Ok(self.load_manifest()?.entries.iter().any(|e| e.id == id))
    }

    /// Compute the content-addressed id without changing the library. Used to
    /// migrate legacy project favorites that predate persisted library ids.
    pub fn content_id(&self, path: impl AsRef<Path>) -> Result<String> {
        let mut source = std::fs::File::open(path)?;
        Ok(stream_hash(&mut source)?)
    }

    /// Favorite a file: copy its bytes into the library (dedup by content hash)
    /// and record an entry. If the same content is already favorited, the
    /// existing entry is returned unchanged and no duplicate file is written.
    ///
    /// Preparation and publication each run under the in-process write lock;
    /// publication re-reads the manifest so concurrent favorites cannot clobber
    /// each other or create duplicate entries.
    pub fn favorite(&self, req: &FavoriteRequest<'_>) -> Result<LibraryEntry> {
        Ok(self.favorite_with_outcome(req)?.entry)
    }

    /// Favorite a file and report whether this transaction created its manifest
    /// entry. The source is read exactly once; ownership is never inferred from
    /// a separate preflight hash that could observe different bytes.
    pub fn favorite_with_outcome(&self, req: &FavoriteRequest<'_>) -> Result<FavoriteOutcome> {
        let prepared = self.prepare_favorite(req)?;
        self.publish_favorite(prepared)
    }

    /// Read/hash one source snapshot and prepare a favorite without publishing a
    /// new manifest entry. Existing entries are returned without staging; new
    /// content is durable but hidden until [`Self::publish_favorite`].
    ///
    /// The source is read exactly once, into a private stage on the same
    /// filesystem as `files/`. Verifying an existing stored copy (or a crash
    /// orphan) is a whole-file hash, so it runs outside the write lock.
    pub fn prepare_favorite(&self, req: &FavoriteRequest<'_>) -> Result<PreparedFavorite> {
        let capabilities = Arc::clone(self.capabilities()?);
        let staged_name = Self::staging_name(req.source);
        let (stage, id) = self.stage_bytes(&capabilities, &staged_name, |stage| {
            let mut source = std::fs::File::open(req.source)?;
            stream_hash_copy(&mut source, stage)
        })?;
        let mut prepared = PreparedFavorite {
            entry: LibraryEntry {
                id: id.clone(),
                kind: req.kind.to_string(),
                category: req.category.clone(),
                favorited_at: req.favorited_at,
                source: req.source.to_str().map(|s| s.to_string()),
                thumb: req.thumb.clone(),
            },
            capabilities,
            active_stages: Arc::clone(&self.active_stages),
            stage: Some(stage),
            stored_name: Some(Self::stored_name(&id, req.source)),
            final_leaf: None,
        };

        // Snapshot ownership and candidate names under the lock (cheap reads);
        // hash the candidates after releasing it.
        let (existing, candidates) = {
            let _guard = self.lock_writes();
            let existing = self
                .load_manifest()?
                .entries
                .into_iter()
                .find(|entry| entry.id == id);
            let candidates = self.stored_index()?.candidates(&id).to_vec();
            (existing, candidates)
        };
        let verified = self.verify_candidates(&id, &candidates)?;
        let kept = self.keep_first_verified(&id, verified)?;

        if let Some(existing) = existing {
            prepared.entry = existing;
            if kept.is_some() {
                // The durable copy is intact; nothing to publish.
                prepared.release_stage();
                prepared.stored_name = None;
            }
            // Otherwise keep the stage to repair a missing or quarantined copy.
            return Ok(prepared);
        }

        if let Some(orphan) = kept {
            // A verified crash orphan: adopt it instead of storing a second
            // copy. Mismatching orphans were quarantined and verified
            // duplicates released above.
            prepared.release_stage();
            prepared.stored_name = Some(orphan.name.clone());
            prepared.final_leaf = Some(orphan);
        }
        Ok(prepared)
    }

    /// Create a private stage, fill it through `fill` (which returns the
    /// content id of the bytes it wrote), and make it durable. The stage is
    /// registered as active until it is published or dropped.
    fn stage_bytes(
        &self,
        capabilities: &LibraryCapabilities,
        staged_name: &OsStr,
        fill: impl FnOnce(&mut std::fs::File) -> std::io::Result<String>,
    ) -> Result<(OwnedLeaf, String)> {
        self.active_stages
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(staged_name.to_owned());
        let staged = (|| -> std::io::Result<(OwnedLeaf, String)> {
            // Transaction access lets the stage be renamed by its own handle
            // into `files/` on Windows; other openers still cannot delete it.
            let mut stage = OwnedLeaf::create_transaction(&capabilities.staging, staged_name)?;
            let id = fill(stage.handle.as_file_mut())?;
            stage.sync_all()?;
            stage.rewind()?;
            Ok((stage, id))
        })();
        staged.map_err(|error| {
            self.active_stages
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(staged_name);
            MediaError::Io(error)
        })
    }

    /// A stage whose name no longer maps to the retained handle cannot be
    /// published by rename. Re-stage the retained, verified bytes under a new
    /// private name (outside the write lock) so rebinding the name can never
    /// change what is published.
    fn rebind_stage_if_moved(&self, prepared: &mut PreparedFavorite) -> Result<()> {
        let capabilities = Arc::clone(&prepared.capabilities);
        let Some(stage) = prepared.stage.as_mut() else {
            return Ok(());
        };
        let bound = match stage.matches_name(&capabilities.staging) {
            Ok(bound) => bound,
            // The name was moved away or deleted: re-stage the retained bytes.
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
            Err(error) => return Err(error.into()),
        };
        if bound {
            return Ok(());
        }
        stage.rewind()?;
        let expected = prepared.entry.id.clone();
        let fresh_name = Self::staging_name(Path::new(&stage.name));
        let (fresh, actual) = self.stage_bytes(&capabilities, &fresh_name, |fresh| {
            stream_hash_copy(stage.handle.as_file_mut(), fresh)
        })?;
        if actual != expected {
            drop(fresh);
            self.active_stages
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(&fresh_name);
            return Err(MediaError::Other(anyhow::anyhow!(
                "prepared favorite content changed before publication"
            )));
        }
        prepared.release_stage();
        prepared.stage = Some(fresh);
        Ok(())
    }

    /// Move the private stage into `files/` under its final content name. This
    /// is one rename on the same filesystem (the stage lives in `files/.staging`),
    /// so publication never copies or hashes the content again. The returned
    /// leaf still truncates its content on drop until the caller disarms it.
    fn publish_stage(&self, prepared: &mut PreparedFavorite) -> Result<OwnedLeaf> {
        let stored_name = prepared.stored_name.clone().ok_or_else(|| {
            MediaError::Other(anyhow::anyhow!("favorite preparation has no target"))
        })?;
        let mut stage = prepared.stage.take().ok_or_else(|| {
            MediaError::Other(anyhow::anyhow!(
                "favorite preparation was already published"
            ))
        })?;
        let staged_name = stage.name.clone();
        let moved = move_owned(
            &prepared.capabilities.staging,
            &mut stage,
            &prepared.capabilities.files,
            Path::new(&stored_name),
        );
        self.active_stages
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&staged_name);
        moved?;
        // The manifest will refer to this name: persist the directory entry
        // (and the removal from staging) first.
        sync_dir(&prepared.capabilities.files)?;
        sync_dir(&prepared.capabilities.staging)?;
        Ok(stage)
    }

    /// Publish a prepared favorite after its project mapping is durable. The
    /// manifest is re-read under the write lock so concurrent prepares dedup at
    /// publication time. The lock covers only the manifest and one rename: no
    /// whole-file hash or copy runs while it is held. RAII owns the staged or
    /// final copy until the manifest commit succeeds.
    pub fn publish_favorite(&self, mut prepared: PreparedFavorite) -> Result<FavoriteOutcome> {
        if !Arc::ptr_eq(self.capabilities()?, &prepared.capabilities) {
            return Err(MediaError::Other(anyhow::anyhow!(
                "favorite preparation belongs to a different library capability"
            )));
        }
        self.rebind_stage_if_moved(&mut prepared)?;
        let guard = self.lock_writes();
        let outcome = self.publish_locked(&mut prepared);
        // Close the retained stage/final handles before other writers can look
        // the content up: on Windows they carry DELETE access, which would make
        // a concurrent verification open fail with a sharing violation.
        drop(prepared);
        drop(guard);
        outcome
    }

    fn publish_locked(&self, prepared: &mut PreparedFavorite) -> Result<FavoriteOutcome> {
        let mut manifest = self.load_manifest()?;
        let manifest_before = manifest.clone();
        manifest.version = MANIFEST_VERSION;
        if let Some(existing) = manifest
            .entries
            .iter()
            .find(|entry| entry.id == prepared.entry.id)
            .cloned()
        {
            if prepared.stage.is_some() && self.stored_index()?.lookup(&existing.id)?.is_none() {
                let mut final_leaf = self.publish_stage(prepared)?;
                final_leaf.disarm_cleanup();
            }
            return Ok(FavoriteOutcome {
                entry: existing,
                created: false,
            });
        }

        if prepared.final_leaf.is_none() {
            let final_leaf = self.publish_stage(prepared)?;
            prepared.final_leaf = Some(final_leaf);
        }
        if !prepared
            .final_leaf
            .as_ref()
            .ok_or_else(|| MediaError::Other(anyhow::anyhow!("favorite has no retained leaf")))?
            .matches_name(&prepared.capabilities.files)?
        {
            return Err(MediaError::Other(anyhow::anyhow!(
                "stored favorite identity changed before manifest commit"
            )));
        }

        manifest.entries.push(prepared.entry.clone());
        self.store_manifest(&manifest)?;
        let final_leaf = prepared.final_leaf.as_mut().ok_or_else(|| {
            MediaError::Other(anyhow::anyhow!(
                "favorite publication lost its final handle"
            ))
        })?;
        if !final_leaf.matches_name(&prepared.capabilities.files)? {
            self.store_manifest(&manifest_before)?;
            return Err(MediaError::Other(anyhow::anyhow!(
                "stored favorite identity changed during manifest commit"
            )));
        }
        final_leaf.disarm_cleanup();
        prepared.final_leaf = None;
        prepared.stored_name = None;
        Ok(FavoriteOutcome {
            entry: prepared.entry.clone(),
            created: true,
        })
    }

    /// Restore a missing durable copy for an existing content id.
    ///
    /// The source is accepted only when its current bytes still hash to
    /// `expected_id`. Unlike [`Self::favorite`], this never creates a manifest
    /// entry, so a source that changed in place cannot create an orphan under a
    /// new id while leaving the project's original mapping unresolved.
    pub fn repair_stored_copy(&self, expected_id: &str, source: impl AsRef<Path>) -> Result<()> {
        #[cfg(any(test, feature = "test-faults"))]
        if FAIL_REPAIR_STORED_COPY.with(|fail| fail.replace(false)) {
            return Err(MediaError::Io(std::io::Error::other(
                "injected stored copy repair failure",
            )));
        }
        let source = source.as_ref();
        let existing = self
            .load_manifest()?
            .entries
            .into_iter()
            .find(|entry| entry.id == expected_id)
            .ok_or_else(|| {
                MediaError::Other(anyhow::anyhow!("unknown library entry: {expected_id}"))
            })?;
        let request = FavoriteRequest {
            source,
            kind: &existing.kind,
            category: existing.category.clone(),
            favorited_at: existing.favorited_at,
            thumb: existing.thumb.clone(),
        };
        let prepared = self.prepare_favorite(&request)?;
        let actual_id = prepared.entry().id.clone();
        if actual_id != expected_id {
            return Err(MediaError::Other(anyhow::anyhow!(
                "source content changed: expected {expected_id}, got {actual_id}"
            )));
        }
        self.publish_favorite(prepared)?;
        Ok(())
    }

    /// Compatibility/debug path to the stored copy, if present. This path is
    /// not authority: production consumers must read via retained handles such
    /// as [`Self::copy_stored_verified`].
    pub fn stored_path(&self, id: &str) -> Result<Option<PathBuf>> {
        Ok(self
            .stored_index()?
            .lookup(id)?
            .map(|name| self.files_dir().join(name)))
    }

    /// Compatibility/debug enumeration of absolute stored paths. Returned paths
    /// carry no authority and must not be used for preview, import, or deletion.
    pub fn stored_paths(&self) -> Result<HashMap<String, PathBuf>> {
        let index = self.stored_index()?;
        let mut paths = HashMap::with_capacity(index.by_id.len());
        for id in index.by_id.keys() {
            if let Some(name) = index.lookup(id)? {
                paths.insert(id.clone(), self.files_dir().join(name));
            }
        }
        Ok(paths)
    }

    /// Remove an entry from the manifest and delete its stored copy. Returns
    /// `true` if an entry was removed. Runs under the write lock.
    ///
    /// Removal does not verify content: a damaged copy (bit rot, external
    /// edit, crash leftover) must stay deletable. Every nofollow regular leaf
    /// claiming the id is opened before the commit and released after it.
    pub fn remove(&self, id: &str) -> Result<bool> {
        let _guard = self.lock_writes();

        let mut manifest = self.load_manifest()?;
        let before = manifest.entries.len();
        manifest.entries.retain(|e| e.id != id);
        if manifest.entries.len() == before {
            return Ok(false);
        }
        manifest.version = MANIFEST_VERSION;
        let files = &self.capabilities()?.files;
        let mut stored = Vec::new();
        for name in self.stored_index()?.candidates(id) {
            match OwnedLeaf::open(files, name) {
                Ok(leaf) => stored.push(leaf),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
        self.store_manifest(&manifest)?;
        // Commit the manifest first. A failed manifest write must not leave an
        // entry that still exists on disk pointing at a copy we already deleted.
        // A failed best-effort cleanup after the commit only leaves an orphaned
        // content-addressed file, which is safe and can be reclaimed later.
        for stored in stored {
            let cleanup = || -> std::io::Result<()> {
                #[cfg(any(test, feature = "test-faults"))]
                if FAIL_REMOVED_STORED_CLEANUP.with(|fail| fail.replace(false)) {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        "injected removed-stored cleanup failure",
                    ));
                }
                self.release_stored_leaf(stored)
                    .map_err(|error| std::io::Error::other(error.to_string()))
            };
            if let Err(error) = cleanup() {
                // The manifest is already committed, so the removal stands; only
                // the best-effort truncation of the content-addressed copy
                // failed, leaving an orphaned file that a later sweep can
                // reclaim. Report it instead of swallowing it silently.
                tracing::warn!(
                    "library remove {id}: stored copy cleanup failed after manifest commit: {error}"
                );
            }
        }
        Ok(true)
    }

    /// Set (or clear, with `None`) the category of the entry with `id`. Returns
    /// the updated entry, or `None` if no entry has that id. Runs under the write
    /// lock so it cannot race a concurrent favorite/remove. Used by the command
    /// layer's `library_categorize` (#55).
    pub fn set_category(&self, id: &str, category: Option<String>) -> Result<Option<LibraryEntry>> {
        let _guard = self.lock_writes();

        let mut manifest = self.load_manifest()?;
        let Some(entry) = manifest.entries.iter_mut().find(|e| e.id == id) else {
            return Ok(None);
        };
        entry.category = category;
        let updated = entry.clone();
        manifest.version = MANIFEST_VERSION;
        self.store_manifest(&manifest)?;
        Ok(Some(updated))
    }

    /// Rename a category: move every entry whose category equals `from` to `to`
    /// (`None` un-categorizes them). Returns the number of entries changed. Runs
    /// under the write lock. Used by the command layer's `library_rename` (#55).
    pub fn rename_category(&self, from: &str, to: Option<String>) -> Result<usize> {
        let _guard = self.lock_writes();

        let mut manifest = self.load_manifest()?;
        let mut changed = 0usize;
        for entry in manifest.entries.iter_mut() {
            if entry.category.as_deref() == Some(from) {
                entry.category = to.clone();
                changed += 1;
            }
        }
        if changed == 0 {
            return Ok(0);
        }
        manifest.version = MANIFEST_VERSION;
        self.store_manifest(&manifest)?;
        Ok(changed)
    }
}

/// SHA-256 of `bytes` as lowercase hex (the content id).
fn hex_digest(digest: &[u8]) -> String {
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::{Arc, Barrier};

    struct BoundedReader {
        remaining: usize,
        max_requested: usize,
        short_read: usize,
    }

    impl Read for BoundedReader {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            self.max_requested = self.max_requested.max(buffer.len());
            if self.remaining == 0 {
                return Ok(0);
            }
            let count = self.remaining.min(buffer.len()).min(self.short_read);
            buffer[..count].fill(0x5a);
            self.remaining -= count;
            Ok(count)
        }
    }

    #[derive(Default)]
    struct CountingWriter(u64);

    impl Write for CountingWriter {
        fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
            self.0 += buffer.len() as u64;
            Ok(buffer.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn src_file(dir: &Path, name: &str, content: &[u8]) -> PathBuf {
        let p = dir.join(name);
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(content).unwrap();
        p
    }

    fn nonempty_file_count(dir: &Path) -> usize {
        std::fs::read_dir(dir)
            .unwrap()
            .filter_map(std::result::Result::ok)
            .filter(|entry| entry.metadata().is_ok_and(|metadata| metadata.len() > 0))
            .count()
    }

    #[test]
    fn streaming_hash_copy_is_bounded_and_handles_short_reads() {
        let total = 32 * 1024 * 1024 + 17;
        let mut reader = BoundedReader {
            remaining: total,
            max_requested: 0,
            short_read: 997,
        };
        let mut writer = CountingWriter::default();

        let streamed = stream_hash_copy(&mut reader, &mut writer).unwrap();
        let mut expected_reader = BoundedReader {
            remaining: total,
            max_requested: 0,
            short_read: 4093,
        };
        let expected = stream_hash(&mut expected_reader).unwrap();

        assert_eq!(streamed, expected);
        assert_eq!(writer.0, total as u64);
        assert!(reader.max_requested <= STREAM_BUFFER_SIZE);
    }

    #[cfg(unix)]
    #[test]
    fn owned_leaf_cleanup_truncates_the_handle_without_touching_a_replacement() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = Dir::open_ambient_dir(tmp.path(), ambient_authority()).unwrap();
        let mut leaf = OwnedLeaf::create(&dir, "owned.tmp").unwrap();
        leaf.handle.as_file_mut().write_all(b"owned bytes").unwrap();
        leaf.sync_all().unwrap();
        std::fs::rename(tmp.path().join("owned.tmp"), tmp.path().join("moved.tmp")).unwrap();
        std::fs::write(tmp.path().join("owned.tmp"), b"replacement").unwrap();

        assert!(!leaf.matches_name(&dir).unwrap());
        drop(leaf);

        assert_eq!(
            std::fs::read(tmp.path().join("owned.tmp")).unwrap(),
            b"replacement"
        );
        assert_eq!(
            std::fs::metadata(tmp.path().join("moved.tmp"))
                .unwrap()
                .len(),
            0
        );
    }

    #[cfg(unix)]
    #[test]
    fn stage_rebinding_cannot_change_the_published_content() {
        let tmp = tempfile::tempdir().unwrap();
        let source = src_file(tmp.path(), "clip.mp4", b"trusted stage bytes");
        let store = LibraryStore::new(tmp.path().join("lib"));
        let prepared = store
            .prepare_favorite(&req(&source, "video", None))
            .unwrap();
        let stage_name = prepared.stage.as_ref().unwrap().name.clone();
        let stage_path = store.files_dir().join(STAGING_SUBDIR).join(&stage_name);
        let moved_stage = store.files_dir().join(STAGING_SUBDIR).join("moved.pending");
        std::fs::rename(&stage_path, &moved_stage).unwrap();
        std::fs::write(&stage_path, b"replacement stage bytes").unwrap();

        let outcome = store.publish_favorite(prepared).unwrap();
        let stored = store.stored_path(&outcome.entry.id).unwrap().unwrap();

        assert_eq!(std::fs::read(stored).unwrap(), b"trusted stage bytes");
        assert_eq!(
            std::fs::read(stage_path).unwrap(),
            b"replacement stage bytes"
        );
        assert_eq!(std::fs::metadata(moved_stage).unwrap().len(), 0);
    }

    #[cfg(windows)]
    #[test]
    fn owned_leaf_denies_windows_delete_sharing_while_retained() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = Dir::open_ambient_dir(tmp.path(), ambient_authority()).unwrap();
        let leaf = OwnedLeaf::create(&dir, "owned.tmp").unwrap();

        let rename = std::fs::rename(tmp.path().join("owned.tmp"), tmp.path().join("moved.tmp"));

        assert!(
            rename.is_err(),
            "retained leaf unexpectedly allowed replacement"
        );
        drop(leaf);
    }

    #[cfg(windows)]
    #[test]
    fn manifest_transaction_leaf_renames_by_its_delete_capable_handle() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = Dir::open_ambient_dir(tmp.path(), ambient_authority()).unwrap();
        let mut leaf = OwnedLeaf::create_transaction(&dir, "manifest.tmp").unwrap();
        leaf.handle.as_file_mut().write_all(b"manifest").unwrap();

        rename_owned(&dir, &mut leaf, Path::new("manifest.json")).unwrap();

        assert!(leaf.matches_name(&dir).unwrap());
        assert_eq!(
            std::fs::read(tmp.path().join("manifest.json")).unwrap(),
            b"manifest"
        );
    }

    fn req<'a>(source: &'a Path, kind: &'a str, category: Option<&str>) -> FavoriteRequest<'a> {
        FavoriteRequest {
            source,
            kind,
            category: category.map(|c| c.to_string()),
            favorited_at: 1_700_000_000.0,
            thumb: None,
        }
    }

    #[test]
    fn favorite_copies_file_and_writes_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        let lib = tmp.path().join("lib");
        let src = src_file(tmp.path(), "clip.mp4", b"hello world");

        let store = LibraryStore::new(&lib);
        let entry = store.favorite(&req(&src, "video", None)).unwrap();

        // File copied into the library under its content hash.
        let stored = store.stored_path(&entry.id).unwrap().unwrap();
        assert!(stored.exists());
        assert_eq!(std::fs::read(&stored).unwrap(), b"hello world");
        // Manifest persisted and reloads to the same single entry.
        assert!(lib.join(MANIFEST_NAME).exists());
        let entries = store.entries().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], entry);
        assert_eq!(entries[0].kind, "video");
    }

    #[test]
    fn dedup_same_content_does_not_duplicate() {
        let tmp = tempfile::tempdir().unwrap();
        let lib = tmp.path().join("lib");
        // Two different source paths, identical bytes.
        let a = src_file(tmp.path(), "a.mp4", b"same bytes");
        let b = src_file(tmp.path(), "b.mp4", b"same bytes");

        let store = LibraryStore::new(&lib);
        let first = store.favorite(&req(&a, "video", None)).unwrap();
        let second = store.favorite(&req(&b, "video", None)).unwrap();

        assert_eq!(first.id, second.id);
        // Only one manifest entry and one stored file.
        assert_eq!(store.entries().unwrap().len(), 1);
        assert_eq!(store.stored_paths().unwrap().len(), 1);
        // The kept entry is the first favorite (source a).
        assert_eq!(second.source.as_deref(), a.to_str());
    }

    #[test]
    fn favorite_outcome_ownership_uses_the_returned_content_snapshot() {
        let tmp = tempfile::tempdir().unwrap();
        let first_source = src_file(tmp.path(), "first.mp4", b"shared bytes");
        let returned_source = src_file(tmp.path(), "returned.mp4", b"shared bytes");
        let store = LibraryStore::new(tmp.path().join("lib"));

        let created = store
            .favorite_with_outcome(&req(&first_source, "video", None))
            .unwrap();
        let reused = store
            .favorite_with_outcome(&req(&returned_source, "video", None))
            .unwrap();

        assert!(created.created);
        assert!(!reused.created);
        assert_eq!(reused.entry, created.entry);
        assert_eq!(store.entries().unwrap(), vec![created.entry]);
    }

    #[test]
    fn content_id_matches_favorite_identity() {
        let tmp = tempfile::tempdir().unwrap();
        let source = src_file(tmp.path(), "clip.mp4", b"same identity");
        let store = LibraryStore::new(tmp.path().join("lib"));

        let id = store.content_id(&source).unwrap();
        let entry = store.favorite(&req(&source, "video", None)).unwrap();

        assert_eq!(id, entry.id);
    }

    #[test]
    fn favorite_repairs_missing_stored_copy() {
        let tmp = tempfile::tempdir().unwrap();
        let source = src_file(tmp.path(), "clip.mp4", b"repair me");
        let store = LibraryStore::new(tmp.path().join("lib"));
        let first = store.favorite(&req(&source, "video", None)).unwrap();
        let stored = store.stored_path(&first.id).unwrap().unwrap();
        std::fs::remove_file(&stored).unwrap();

        let repaired = store.favorite(&req(&source, "video", None)).unwrap();

        assert_eq!(repaired, first);
        let repaired_path = store.stored_path(&first.id).unwrap().unwrap();
        assert_eq!(std::fs::read(repaired_path).unwrap(), b"repair me");
        assert_eq!(store.entries().unwrap(), vec![first]);
    }

    #[test]
    fn prepared_favorite_stays_hidden_until_publish_and_drop_cleans_stage() {
        let tmp = tempfile::tempdir().unwrap();
        let source = src_file(tmp.path(), "clip.mp4", b"prepared bytes");
        let store = LibraryStore::new(tmp.path().join("lib"));

        let prepared = store
            .prepare_favorite(&req(&source, "video", None))
            .unwrap();
        let id = prepared.entry().id.clone();
        assert!(prepared.needs_publish());
        assert!(store.entries().unwrap().is_empty());
        assert!(store.stored_path(&id).unwrap().is_none());
        drop(prepared);
        assert!(store.entries().unwrap().is_empty());
        assert!(store.stored_path(&id).unwrap().is_none());
        assert_eq!(
            nonempty_file_count(&store.files_dir().join(STAGING_SUBDIR)),
            0
        );

        let prepared = store
            .prepare_favorite(&req(&source, "video", None))
            .unwrap();
        let outcome = store.publish_favorite(prepared).unwrap();
        assert!(outcome.created);
        assert_eq!(outcome.entry.id, id);
        assert_eq!(store.entries().unwrap(), vec![outcome.entry]);
        assert_eq!(
            std::fs::read(store.stored_path(&id).unwrap().unwrap()).unwrap(),
            b"prepared bytes"
        );
    }

    #[test]
    fn failed_manifest_publish_cleans_both_stage_and_unowned_target() {
        let tmp = tempfile::tempdir().unwrap();
        let source = src_file(tmp.path(), "clip.mp4", b"failed publish bytes");
        let store = LibraryStore::new(tmp.path().join("lib"));
        let prepared = store
            .prepare_favorite(&req(&source, "video", None))
            .unwrap();
        let id = prepared.entry().id.clone();
        std::fs::create_dir(store.root.join(MANIFEST_NAME)).unwrap();

        store
            .publish_favorite(prepared)
            .expect_err("blocked manifest commit must fail");

        std::fs::remove_dir(store.root.join(MANIFEST_NAME)).unwrap();
        assert!(store.entries().unwrap().is_empty());
        assert!(store.stored_path(&id).unwrap().is_none());
        assert!(store.stored_paths().unwrap().is_empty());
        assert_eq!(
            nonempty_file_count(&store.files_dir().join(STAGING_SUBDIR)),
            0
        );
    }

    #[test]
    fn restart_reconciliation_preserves_unknown_leaves_without_exposing_them() {
        let tmp = tempfile::tempdir().unwrap();
        let library_root = tmp.path().join("lib");
        let source = src_file(tmp.path(), "kept.mp4", b"kept bytes");
        let store = LibraryStore::new(&library_root);
        let kept = store.favorite(&req(&source, "video", None)).unwrap();
        let kept_path = store.stored_path(&kept.id).unwrap().unwrap();
        let staging_dir = store.files_dir().join(STAGING_SUBDIR);
        std::fs::create_dir_all(&staging_dir).unwrap();
        std::fs::write(staging_dir.join("crashed.pending"), b"staged orphan").unwrap();
        let orphan_path = store.files_dir().join(format!("{}.mov", "0".repeat(64)));
        std::fs::write(&orphan_path, b"published before crash").unwrap();

        let reopened = LibraryStore::new(&library_root);
        reopened.reconcile_storage().unwrap();

        assert!(kept_path.is_file());
        assert_eq!(
            std::fs::read(staging_dir.join("crashed.pending")).unwrap(),
            b"staged orphan"
        );
        assert_eq!(
            std::fs::read(&orphan_path).unwrap(),
            b"published before crash"
        );
        assert_eq!(reopened.entries().unwrap(), vec![kept]);
        assert_eq!(reopened.stored_ids_verified().unwrap().len(), 1);
    }

    #[test]
    fn project_scoped_verification_does_not_hash_unreferenced_library_content() {
        let tmp = tempfile::tempdir().unwrap();
        let store = LibraryStore::new(tmp.path().join("lib"));
        let entries: Vec<_> = (0..20)
            .map(|index| {
                let name = format!("clip-{index}.mp4");
                let source = src_file(tmp.path(), &name, name.as_bytes());
                store.favorite(&req(&source, "video", None)).unwrap()
            })
            .collect();
        std::fs::write(
            store.stored_path(&entries[1].id).unwrap().unwrap(),
            b"corrupted copy",
        )
        .unwrap();

        let requested = HashSet::from([entries[0].id.clone()]);
        STREAM_HASH_CALLS.with(|count| count.set(0));
        assert_eq!(
            store.stored_ids_verified_for(&requested).unwrap().0,
            requested
        );
        STREAM_HASH_CALLS.with(|count| assert_eq!(count.get(), 1));
        assert!(store.stored_ids_verified().is_err());
        let (verified, failures) = store
            .stored_ids_verified_for(&HashSet::from([
                entries[0].id.clone(),
                entries[1].id.clone(),
            ]))
            .unwrap();
        assert_eq!(verified, HashSet::from([entries[0].id.clone()]));
        assert!(failures[&entries[1].id].contains("hash mismatch"));
        STREAM_HASH_CALLS.with(|count| count.set(0));
        assert!(store
            .stored_ids_verified_for(&HashSet::new())
            .unwrap()
            .0
            .is_empty());
        STREAM_HASH_CALLS.with(|count| assert_eq!(count.get(), 0));
    }

    #[test]
    fn crash_orphan_is_reconciled_before_same_content_is_favorited() {
        let tmp = tempfile::tempdir().unwrap();
        let library_root = tmp.path().join("lib");
        let source = src_file(tmp.path(), "clip.mp4", b"crash-window bytes");
        let store = LibraryStore::new(&library_root);
        let id = store.content_id(&source).unwrap();
        let orphan = store.files_dir().join(format!("{id}.crashed.mp4"));
        std::fs::write(&orphan, b"crash-window bytes").unwrap();

        store.reconcile_storage().unwrap();
        assert_eq!(std::fs::read(&orphan).unwrap(), b"crash-window bytes");

        let entry = store.favorite(&req(&source, "video", None)).unwrap();
        assert_eq!(entry.id, id);
        assert_eq!(store.stored_ids_verified().unwrap(), HashSet::from([id]));
        assert_eq!(
            std::fs::read(store.stored_path(&entry.id).unwrap().unwrap()).unwrap(),
            b"crash-window bytes"
        );
    }

    #[test]
    fn mismatched_crash_orphan_is_quarantined_instead_of_becoming_owned() {
        let tmp = tempfile::tempdir().unwrap();
        let source = src_file(tmp.path(), "clip.mp4", b"trusted bytes");
        let store = LibraryStore::new(tmp.path().join("lib"));
        let id = store.content_id(&source).unwrap();
        let orphan = store.files_dir().join(format!("{id}.crashed.mp4"));
        std::fs::write(&orphan, b"untrusted bytes").unwrap();

        let entry = store.favorite(&req(&source, "video", None)).unwrap();

        assert_eq!(entry.id, id);
        assert_eq!(store.entries().unwrap(), vec![entry.clone()]);
        let stored = store.stored_path(&id).unwrap().unwrap();
        assert_ne!(stored, orphan, "the mismatched orphan must not be adopted");
        assert_eq!(std::fs::read(stored).unwrap(), b"trusted bytes");
        // The mismatching owned copy is removed rather than adopted.
        assert!(!orphan.exists());
        assert_eq!(store.stored_ids_verified().unwrap(), HashSet::from([id]));
    }

    #[test]
    fn duplicate_crash_orphans_adopt_one_verified_copy() {
        let tmp = tempfile::tempdir().unwrap();
        let source = src_file(tmp.path(), "clip.mp4", b"duplicate orphan bytes");
        let store = LibraryStore::new(tmp.path().join("lib"));
        let id = store.content_id(&source).unwrap();
        let first = store.files_dir().join(format!("{id}.first.mp4"));
        let second = store.files_dir().join(format!("{id}.second.mp4"));
        std::fs::write(&first, b"duplicate orphan bytes").unwrap();
        std::fs::write(&second, b"duplicate orphan bytes").unwrap();

        let entry = store.favorite(&req(&source, "video", None)).unwrap();

        assert_eq!(entry.id, id);
        assert_eq!(store.stored_path(&id).unwrap(), Some(first.clone()));
        assert_eq!(std::fs::read(&first).unwrap(), b"duplicate orphan bytes");
        assert!(!second.exists());
    }

    #[cfg(unix)]
    #[test]
    fn hard_linked_duplicates_never_lose_the_kept_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        let source = src_file(tmp.path(), "clip.mp4", b"linked bytes");
        let store = LibraryStore::new(tmp.path().join("lib"));
        let entry = store.favorite(&req(&source, "video", None)).unwrap();
        let kept = store.stored_path(&entry.id).unwrap().unwrap();
        // A second name for the same file, sorted after the kept one.
        let linked = store.files_dir().join(format!("{}.zz.mp4", entry.id));
        std::fs::hard_link(&kept, &linked).unwrap();

        store.resolve_duplicate_copies().unwrap();

        assert_eq!(std::fs::read(&kept).unwrap(), b"linked bytes");
        assert!(!linked.exists(), "only the extra name is removed");
        assert_eq!(store.stored_path(&entry.id).unwrap(), Some(kept.clone()));

        // A hard link from outside the library keeps its bytes on removal.
        let outside = tmp.path().join("outside-link.mp4");
        std::fs::hard_link(&kept, &outside).unwrap();
        assert!(store.remove(&entry.id).unwrap());
        assert_eq!(std::fs::read(&outside).unwrap(), b"linked bytes");
        assert!(!kept.exists());
    }

    #[cfg(unix)]
    #[test]
    fn a_read_only_stored_copy_can_be_refavorited_and_removed() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir().unwrap();
        let source = src_file(tmp.path(), "clip.mp4", b"read only bytes");
        let store = LibraryStore::new(tmp.path().join("lib"));
        let entry = store.favorite(&req(&source, "video", None)).unwrap();
        let stored = store.stored_path(&entry.id).unwrap().unwrap();
        std::fs::set_permissions(&stored, std::fs::Permissions::from_mode(0o444)).unwrap();

        let again = store
            .favorite_with_outcome(&req(&source, "video", None))
            .unwrap();
        assert!(!again.created);
        assert!(store.remove(&entry.id).unwrap());
        assert!(store.entries().unwrap().is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn reconciliation_rejects_a_symlinked_files_root_without_touching_its_target() {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::tempdir().unwrap();
        let library_root = tmp.path().join("lib");
        let external = tmp.path().join("external");
        std::fs::create_dir_all(&library_root).unwrap();
        std::fs::create_dir(&external).unwrap();
        let sentinel = external.join("sentinel.mp4");
        std::fs::write(&sentinel, b"must survive").unwrap();
        symlink(&external, library_root.join(FILES_SUBDIR)).unwrap();
        let store = LibraryStore::new(library_root);

        let error = store
            .reconcile_storage()
            .expect_err("symlinked files root must be rejected");

        assert!(error.to_string().contains("secure global library root"));
        assert_eq!(std::fs::read(&sentinel).unwrap(), b"must survive");
        assert_eq!(std::fs::read_dir(&external).unwrap().count(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn retained_files_capability_defeats_a_barrier_coordinated_root_swap() {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::tempdir().unwrap();
        let library_root = tmp.path().join("lib");
        let external = tmp.path().join("external");
        std::fs::create_dir(&external).unwrap();
        let sentinel = external.join("sentinel.mp4");
        std::fs::write(&sentinel, b"must survive").unwrap();
        let store = Arc::new(LibraryStore::new(&library_root));
        let retained_files = library_root.join("files.retained");
        std::fs::write(store.files_dir().join("orphan.mp4"), b"orphan").unwrap();
        let barrier = Arc::new(Barrier::new(2));
        let worker = {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                store.reconcile_storage()
            })
        };

        std::fs::rename(store.files_dir(), &retained_files).unwrap();
        symlink(&external, store.files_dir()).unwrap();
        barrier.wait();
        worker.join().unwrap().unwrap();

        assert_eq!(std::fs::read(&sentinel).unwrap(), b"must survive");
        assert_eq!(std::fs::read_dir(&external).unwrap().count(), 1);
        assert_eq!(
            std::fs::read(retained_files.join("orphan.mp4")).unwrap(),
            b"orphan"
        );
    }

    #[cfg(unix)]
    #[test]
    fn retained_root_capability_defeats_a_barrier_coordinated_namespace_swap() {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::tempdir().unwrap();
        let source = src_file(tmp.path(), "clip.mp4", b"retained root bytes");
        let library_root = tmp.path().join("lib");
        let retained_root = tmp.path().join("lib.retained");
        let external = tmp.path().join("external");
        std::fs::create_dir(&external).unwrap();
        let sentinel = external.join("sentinel.json");
        std::fs::write(&sentinel, b"must survive").unwrap();
        let store = Arc::new(LibraryStore::new(&library_root));
        let barrier = Arc::new(Barrier::new(2));
        let worker = {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                store.favorite(&req(&source, "video", None))
            })
        };

        std::fs::rename(&library_root, &retained_root).unwrap();
        symlink(&external, &library_root).unwrap();
        barrier.wait();
        let entry = worker.join().unwrap().unwrap();

        assert_eq!(std::fs::read(&sentinel).unwrap(), b"must survive");
        assert_eq!(std::fs::read_dir(&external).unwrap().count(), 1);
        assert!(retained_root.join(MANIFEST_NAME).is_file());
        assert!(std::fs::read_dir(retained_root.join(FILES_SUBDIR))
            .unwrap()
            .any(|item| item
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(&entry.id)));
    }

    #[cfg(unix)]
    #[test]
    fn retained_staging_capability_defeats_a_barrier_coordinated_namespace_swap() {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::tempdir().unwrap();
        let source = src_file(tmp.path(), "clip.mp4", b"retained staging bytes");
        let store = Arc::new(LibraryStore::new(tmp.path().join("lib")));
        let staging_path = store.files_dir().join(STAGING_SUBDIR);
        let retained_staging = tmp.path().join("staging.retained");
        let external = tmp.path().join("external-stage");
        std::fs::create_dir(&external).unwrap();
        let sentinel = external.join("sentinel.pending");
        std::fs::write(&sentinel, b"must survive").unwrap();
        let barrier = Arc::new(Barrier::new(2));
        let worker = {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                drop(
                    store
                        .prepare_favorite(&req(&source, "video", None))
                        .unwrap(),
                );
            })
        };

        std::fs::rename(&staging_path, &retained_staging).unwrap();
        symlink(&external, &staging_path).unwrap();
        barrier.wait();
        worker.join().unwrap();

        assert_eq!(std::fs::read(&sentinel).unwrap(), b"must survive");
        assert_eq!(std::fs::read_dir(&external).unwrap().count(), 1);
        assert_eq!(nonempty_file_count(&retained_staging), 0);
    }

    #[cfg(unix)]
    #[test]
    fn manifest_reads_never_follow_a_swapped_leaf_symlink() {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::tempdir().unwrap();
        let library_root = tmp.path().join("lib");
        let external_manifest = tmp.path().join("external.json");
        std::fs::write(&external_manifest, br#"{"version":1,"entries":[]}"#).unwrap();
        let store = LibraryStore::new(&library_root);
        symlink(&external_manifest, library_root.join(MANIFEST_NAME)).unwrap();

        store
            .entries()
            .expect_err("manifest leaf symlink must fail closed");

        assert_eq!(
            std::fs::read(&external_manifest).unwrap(),
            br#"{"version":1,"entries":[]}"#
        );
    }

    #[cfg(windows)]
    #[test]
    fn reconciliation_rejects_a_junctioned_files_root_without_touching_its_target() {
        let tmp = tempfile::tempdir().unwrap();
        let library_root = tmp.path().join("lib");
        let external = tmp.path().join("external");
        std::fs::create_dir_all(&library_root).unwrap();
        std::fs::create_dir(&external).unwrap();
        let sentinel = external.join("sentinel.mp4");
        std::fs::write(&sentinel, b"must survive").unwrap();
        let status = std::process::Command::new("cmd")
            .args(["/C", "mklink", "/J"])
            .arg(library_root.join(FILES_SUBDIR))
            .arg(&external)
            .status()
            .unwrap();
        assert!(status.success(), "create junction fixture");
        let store = LibraryStore::new(library_root);

        store
            .reconcile_storage()
            .expect_err("junctioned files root must be rejected");

        assert_eq!(std::fs::read(&sentinel).unwrap(), b"must survive");
        assert_eq!(std::fs::read_dir(&external).unwrap().count(), 1);
    }

    #[test]
    fn reconciliation_does_not_sweep_a_live_prepared_favorite() {
        let tmp = tempfile::tempdir().unwrap();
        let source = src_file(tmp.path(), "clip.mp4", b"live stage");
        let store = LibraryStore::new(tmp.path().join("lib"));
        let prepared = store
            .prepare_favorite(&req(&source, "video", None))
            .unwrap();

        store.reconcile_storage().unwrap();
        let outcome = store.publish_favorite(prepared).unwrap();

        assert!(outcome.created);
        assert!(store.stored_path(&outcome.entry.id).unwrap().is_some());
    }

    #[test]
    fn stored_lookup_rejects_a_directory_claiming_a_manifest_id() {
        let tmp = tempfile::tempdir().unwrap();
        let source = src_file(tmp.path(), "clip.mp4", b"directory collision");
        let store = LibraryStore::new(tmp.path().join("lib"));
        let entry = store.favorite(&req(&source, "video", None)).unwrap();
        std::fs::remove_file(store.stored_path(&entry.id).unwrap().unwrap()).unwrap();
        std::fs::create_dir(store.files_dir().join(format!("{}.mp4", entry.id))).unwrap();

        assert_eq!(
            store.stored_path(&entry.id).unwrap(),
            None,
            "a directory must not be accepted as durable content"
        );
        store.reconcile_storage().unwrap();
        // The entry has no durable copy; favoriting the source repairs it.
        let repaired = store.favorite(&req(&source, "video", None)).unwrap();
        assert_eq!(repaired, entry);
        let stored = store.stored_path(&entry.id).unwrap().unwrap();
        assert_eq!(std::fs::read(stored).unwrap(), b"directory collision");
    }

    #[test]
    fn stored_lookup_rejects_duplicate_extensions_for_one_content_id() {
        let tmp = tempfile::tempdir().unwrap();
        let source = src_file(tmp.path(), "clip.mp4", b"duplicate collision");
        let store = LibraryStore::new(tmp.path().join("lib"));
        let entry = store.favorite(&req(&source, "video", None)).unwrap();
        std::fs::write(
            store.files_dir().join(format!("{}.mov", entry.id)),
            b"duplicate",
        )
        .unwrap();

        let error = store
            .stored_paths()
            .expect_err("duplicate candidates must fail closed");
        assert!(error.to_string().contains("multiple stored copies"));

        // Duplicate resolution keeps only the copy that verifies.
        store.reconcile_storage().unwrap();
        assert_eq!(store.resolve_duplicate_copies().unwrap(), 1);
        let paths = store.stored_paths().unwrap();
        assert_eq!(
            std::fs::read(&paths[&entry.id]).unwrap(),
            b"duplicate collision"
        );
        assert!(!store.files_dir().join(format!("{}.mov", entry.id)).exists());
    }

    #[test]
    fn favorite_publishes_the_staged_inode_without_hashing_under_the_write_lock() {
        let tmp = tempfile::tempdir().unwrap();
        let source = src_file(
            tmp.path(),
            "clip.mp4",
            &vec![7_u8; 3 * STREAM_BUFFER_SIZE + 5],
        );
        let store = LibraryStore::new(tmp.path().join("lib"));
        STREAMS_UNDER_WRITE_LOCK.with(|count| count.set(0));

        let prepared = store
            .prepare_favorite(&req(&source, "video", None))
            .unwrap();
        let staged = Handle::from_file(
            prepared
                .stage
                .as_ref()
                .unwrap()
                .handle
                .as_file()
                .try_clone()
                .unwrap(),
        )
        .unwrap();
        let outcome = store.publish_favorite(prepared).unwrap();
        assert!(outcome.created);
        let stored = store.stored_path(&outcome.entry.id).unwrap().unwrap();
        // One copy: the published file is the very inode the source was
        // streamed into, moved by rename.
        assert_eq!(Handle::from_path(&stored).unwrap(), staged);
        // Windows: the cloned stage handle carries DELETE access; release it
        // like publication releases its own before other writers look it up.
        drop(staged);
        assert_eq!(
            nonempty_file_count(&store.files_dir().join(STAGING_SUBDIR)),
            0
        );

        // Re-favoriting, repairing and removing never hash or copy a whole
        // file while holding the global write lock either.
        store.favorite(&req(&source, "video", None)).unwrap();
        std::fs::remove_file(&stored).unwrap();
        store.favorite(&req(&source, "video", None)).unwrap();
        assert!(store.stored_path(&outcome.entry.id).unwrap().is_some());
        assert!(store.remove(&outcome.entry.id).unwrap());
        STREAMS_UNDER_WRITE_LOCK.with(|count| assert_eq!(count.get(), 0));
    }

    #[test]
    fn foreign_files_and_directories_in_files_do_not_break_the_library() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("lib");
        let store = LibraryStore::new(&root);
        let kept_source = src_file(tmp.path(), "kept.mp4", b"kept bytes");
        let kept = store.favorite(&req(&kept_source, "video", None)).unwrap();
        let files = store.files_dir();
        std::fs::write(files.join(".DS_Store"), b"finder metadata").unwrap();
        std::fs::write(files.join("desktop.ini"), b"[.ShellClassInfo]").unwrap();
        std::fs::write(files.join("Thumbs.db"), b"thumbs").unwrap();
        std::fs::write(files.join("note.txt"), b"a note").unwrap();
        std::fs::create_dir(files.join("Some Folder")).unwrap();
        std::fs::create_dir(files.join(STAGING_SUBDIR).join("nested")).unwrap();

        store.reconcile_storage().unwrap();
        let reopened = LibraryStore::new(&root);
        reopened.reconcile_storage().unwrap();

        let source = src_file(tmp.path(), "clip.mp4", b"new favorite bytes");
        let added = reopened.favorite(&req(&source, "video", None)).unwrap();
        let mut copied = Vec::new();
        assert!(reopened
            .copy_stored_verified(&kept.id, &mut copied)
            .unwrap()
            .is_some());
        assert_eq!(copied, b"kept bytes");
        assert_eq!(
            reopened.stored_ids_verified().unwrap(),
            HashSet::from([kept.id.clone(), added.id.clone()])
        );
        assert!(reopened.remove(&kept.id).unwrap());
        assert_eq!(reopened.entries().unwrap(), vec![added]);
        assert_eq!(
            std::fs::read(files.join(".DS_Store")).unwrap(),
            b"finder metadata"
        );
        assert!(files.join("Some Folder").is_dir());
    }

    #[test]
    fn a_damaged_stored_copy_can_still_be_removed() {
        let tmp = tempfile::tempdir().unwrap();
        let source = src_file(tmp.path(), "clip.mp4", b"original bytes");
        let store = LibraryStore::new(tmp.path().join("lib"));
        let entry = store.favorite(&req(&source, "video", None)).unwrap();
        let stored = store.stored_path(&entry.id).unwrap().unwrap();
        std::fs::write(&stored, b"Original bytes").unwrap();

        assert!(store.remove(&entry.id).unwrap());

        assert!(store.entries().unwrap().is_empty());
        assert!(!stored.exists());
    }

    #[test]
    fn refavoriting_repairs_a_damaged_stored_copy() {
        let tmp = tempfile::tempdir().unwrap();
        let source = src_file(tmp.path(), "clip.mp4", b"original bytes");
        let store = LibraryStore::new(tmp.path().join("lib"));
        let entry = store.favorite(&req(&source, "video", None)).unwrap();
        let damaged = store.stored_path(&entry.id).unwrap().unwrap();
        std::fs::write(&damaged, b"Original bytes").unwrap();

        let outcome = store
            .favorite_with_outcome(&req(&source, "video", None))
            .unwrap();

        assert!(!outcome.created);
        assert_eq!(outcome.entry, entry);
        assert!(!damaged.exists());
        let repaired = store.stored_path(&entry.id).unwrap().unwrap();
        assert_eq!(std::fs::read(repaired).unwrap(), b"original bytes");
        assert_eq!(
            store.stored_ids_verified().unwrap(),
            HashSet::from([entry.id])
        );
    }

    #[test]
    fn batch_stored_path_index_enumerates_the_directory_once() {
        let tmp = tempfile::tempdir().unwrap();
        let store = LibraryStore::new(tmp.path().join("lib"));
        for index in 0..32 {
            let source = src_file(
                tmp.path(),
                &format!("clip-{index}.mp4"),
                format!("bytes-{index}").as_bytes(),
            );
            store.favorite(&req(&source, "video", None)).unwrap();
        }
        STORED_INDEX_SCANS.with(|count| count.set(0));

        let paths = store.stored_paths().unwrap();

        assert_eq!(paths.len(), 32);
        STORED_INDEX_SCANS.with(|count| assert_eq!(count.get(), 1));
    }

    #[test]
    fn concurrent_prepared_favorites_dedup_when_published() {
        let tmp = tempfile::tempdir().unwrap();
        let source = src_file(tmp.path(), "clip.mp4", b"shared prepared bytes");
        let store = Arc::new(LibraryStore::new(tmp.path().join("lib")));
        let first = store
            .prepare_favorite(&req(&source, "video", None))
            .unwrap();
        let second = store
            .prepare_favorite(&req(&source, "video", None))
            .unwrap();
        let barrier = Arc::new(Barrier::new(2));
        let handles: Vec<_> = [first, second]
            .into_iter()
            .map(|prepared| {
                let store = Arc::clone(&store);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    store.publish_favorite(prepared).unwrap()
                })
            })
            .collect();
        let outcomes: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();

        assert_eq!(outcomes.iter().filter(|outcome| outcome.created).count(), 1);
        assert_eq!(store.entries().unwrap().len(), 1);
        assert_eq!(outcomes[0].entry.id, outcomes[1].entry.id);
        assert_eq!(
            nonempty_file_count(&store.files_dir().join(STAGING_SUBDIR)),
            0
        );
    }

    #[test]
    fn expected_id_repair_rejects_changed_source_without_creating_an_orphan() {
        let tmp = tempfile::tempdir().unwrap();
        let source = src_file(tmp.path(), "clip.mp4", b"original bytes");
        let store = LibraryStore::new(tmp.path().join("lib"));
        let entry = store.favorite(&req(&source, "video", None)).unwrap();
        let stored = store.stored_path(&entry.id).unwrap().unwrap();
        std::fs::remove_file(stored).unwrap();
        std::fs::write(&source, b"changed bytes").unwrap();

        let error = store
            .repair_stored_copy(&entry.id, &source)
            .expect_err("changed source must not repair a different content id");

        assert!(error.to_string().contains("source content changed"));
        assert_eq!(store.entries().unwrap(), vec![entry.clone()]);
        assert!(store.stored_path(&entry.id).unwrap().is_none());
        assert!(store.stored_paths().unwrap().is_empty());
    }

    #[test]
    fn expected_id_repair_restores_unchanged_source() {
        let tmp = tempfile::tempdir().unwrap();
        let source = src_file(tmp.path(), "clip.mp4", b"stable bytes");
        let store = LibraryStore::new(tmp.path().join("lib"));
        let entry = store.favorite(&req(&source, "video", None)).unwrap();
        std::fs::remove_file(store.stored_path(&entry.id).unwrap().unwrap()).unwrap();

        store.repair_stored_copy(&entry.id, &source).unwrap();

        assert_eq!(store.entries().unwrap(), vec![entry.clone()]);
        assert_eq!(
            std::fs::read(store.stored_path(&entry.id).unwrap().unwrap()).unwrap(),
            b"stable bytes"
        );
    }

    #[test]
    fn concurrent_favorites_do_not_lose_entries() {
        const THREADS: usize = 16;
        let tmp = tempfile::tempdir().unwrap();
        let store = Arc::new(LibraryStore::new(tmp.path().join("lib")));
        let barrier = Arc::new(Barrier::new(THREADS));
        let sources: Vec<_> = (0..THREADS)
            .map(|index| {
                src_file(
                    tmp.path(),
                    &format!("clip-{index}.mp4"),
                    format!("content-{index}").as_bytes(),
                )
            })
            .collect();

        let handles: Vec<_> = sources
            .into_iter()
            .map(|source| {
                let store = Arc::clone(&store);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    store.favorite(&req(&source, "video", None)).unwrap()
                })
            })
            .collect();
        let entries: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();

        let stored = store.entries().unwrap();
        assert_eq!(stored.len(), THREADS);
        assert!(entries
            .iter()
            .all(|entry| stored.iter().any(|stored| stored.id == entry.id)));
    }

    #[test]
    fn distinct_content_yields_distinct_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let lib = tmp.path().join("lib");
        let a = src_file(tmp.path(), "a.mp4", b"alpha");
        let b = src_file(tmp.path(), "b.mp4", b"beta");

        let store = LibraryStore::new(&lib);
        store.favorite(&req(&a, "video", None)).unwrap();
        store.favorite(&req(&b, "audio", None)).unwrap();

        assert_eq!(store.entries().unwrap().len(), 2);
    }

    #[test]
    fn category_filter_partitions_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let lib = tmp.path().join("lib");
        let a = src_file(tmp.path(), "a.mp4", b"a");
        let b = src_file(tmp.path(), "b.mp4", b"b");
        let c = src_file(tmp.path(), "c.mp4", b"c");

        let store = LibraryStore::new(&lib);
        store.favorite(&req(&a, "video", Some("broll"))).unwrap();
        store.favorite(&req(&b, "video", Some("broll"))).unwrap();
        store.favorite(&req(&c, "video", None)).unwrap();

        assert_eq!(store.entries_in_category(Some("broll")).unwrap().len(), 2);
        assert_eq!(store.entries_in_category(Some("music")).unwrap().len(), 0);
        // None keeps only uncategorized.
        let uncat = store.entries_in_category(None).unwrap();
        assert_eq!(uncat.len(), 1);
        assert_eq!(uncat[0].category, None);
    }

    #[test]
    fn missing_manifest_reads_as_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let store = LibraryStore::new(tmp.path().join("never_created"));
        assert!(store.entries().unwrap().is_empty());
        assert!(!store.contains("anything").unwrap());
    }

    #[test]
    fn contains_reflects_favorited_id() {
        let tmp = tempfile::tempdir().unwrap();
        let lib = tmp.path().join("lib");
        let a = src_file(tmp.path(), "a.mp4", b"payload");
        let store = LibraryStore::new(&lib);
        let e = store.favorite(&req(&a, "video", None)).unwrap();
        assert!(store.contains(&e.id).unwrap());
    }

    #[test]
    fn remove_deletes_entry_and_file() {
        let tmp = tempfile::tempdir().unwrap();
        let lib = tmp.path().join("lib");
        let a = src_file(tmp.path(), "a.mp4", b"removable");
        let store = LibraryStore::new(&lib);
        let e = store.favorite(&req(&a, "video", None)).unwrap();

        let stored = store.stored_path(&e.id).unwrap().unwrap();
        assert!(store.remove(&e.id).unwrap());
        assert!(
            !stored.exists(),
            "removal must unlink the stored copy, not leave a tombstone"
        );
        assert!(store.entries().unwrap().is_empty());
        assert!(store.stored_path(&e.id).unwrap().is_none());
        // Removing again is a no-op.
        assert!(!store.remove(&e.id).unwrap());
    }

    #[test]
    fn postcommit_stored_cleanup_failure_does_not_fail_removal() {
        let tmp = tempfile::tempdir().unwrap();
        let source = src_file(tmp.path(), "cleanup.mp4", b"cleanup boundary");
        let store = LibraryStore::new(tmp.path().join("lib"));
        let entry = store.favorite(&req(&source, "video", None)).unwrap();

        fail_next_removed_stored_cleanup_for_test();
        assert!(store.remove(&entry.id).unwrap());

        assert!(store.entries().unwrap().is_empty());
        assert!(!store.stored_ids_verified().unwrap().contains(&entry.id));
        assert!(!store.remove(&entry.id).unwrap());
        let refavorited = store.favorite(&req(&source, "video", None)).unwrap();
        assert_eq!(refavorited.id, entry.id);
        assert!(store.contains(&entry.id).unwrap());
    }

    #[test]
    fn remove_reports_failed_stored_copy_cleanup() {
        use tracing::field::Visit;
        use tracing::span::{Attributes, Id, Record};
        use tracing::{Event, Level, Metadata, Subscriber};

        struct Warnings(std::sync::Arc<std::sync::Mutex<Vec<String>>>);
        impl Visit for Warnings {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                if field.name() == "message" {
                    self.0.lock().unwrap().push(format!("{value:?}"));
                }
            }
        }
        impl Subscriber for Warnings {
            fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
                true
            }
            fn new_span(&self, _attrs: &Attributes<'_>) -> Id {
                Id::from_u64(1)
            }
            fn record(&self, _span: &Id, _values: &Record<'_>) {}
            fn record_follows_from(&self, _span: &Id, _follows: &Id) {}
            fn event(&self, event: &Event<'_>) {
                if *event.metadata().level() <= Level::WARN {
                    event.record(&mut Warnings(self.0.clone()));
                }
            }
            fn enter(&self, _span: &Id) {}
            fn exit(&self, _span: &Id) {}
        }

        let warnings = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured = warnings.clone();
        let (entry, messages) = tracing::subscriber::with_default(Warnings(warnings), || {
            let tmp = tempfile::tempdir().unwrap();
            let source = src_file(tmp.path(), "cleanup-report.mp4", b"reported cleanup");
            let store = LibraryStore::new(tmp.path().join("lib"));
            let entry = store.favorite(&req(&source, "video", None)).unwrap();

            // The `tracing::warn!` callsite inside `remove` caches its interest
            // globally on first execution. Under parallel test load another
            // test can reach it while no subscriber is installed, caching
            // "never interested". This first failing removal guarantees the
            // callsite is registered; rebuilding the interest cache then
            // re-evaluates it with this subscriber active, so the second
            // failing removal's warning is guaranteed to be delivered.
            fail_next_removed_stored_cleanup_for_test();
            assert!(store.remove(&entry.id).unwrap());
            assert!(store.entries().unwrap().is_empty());
            let refavorited = store.favorite(&req(&source, "video", None)).unwrap();
            assert_eq!(refavorited.id, entry.id);
            tracing::callsite::rebuild_interest_cache();
            fail_next_removed_stored_cleanup_for_test();
            assert!(store.remove(&refavorited.id).unwrap());
            // The manifest commit still stands even though the best-effort
            // truncation of the stored copy failed.
            assert!(store.entries().unwrap().is_empty());
            let messages = captured.lock().unwrap().clone();
            (entry, messages)
        });
        assert!(
            messages.iter().any(|message| {
                message.contains("stored copy cleanup failed") && message.contains(&entry.id)
            }),
            "expected a warning reporting the stored copy cleanup failure, got {messages:?}"
        );
    }

    #[test]
    fn failed_capability_replace_preserves_the_existing_canonical_leaf() {
        let tmp = tempfile::tempdir().unwrap();
        let root = Dir::open_ambient_dir(tmp.path(), ambient_authority()).unwrap();
        std::fs::write(tmp.path().join("media.json"), b"old manifest").unwrap();

        fail_next_atomic_capability_replace_for_test();
        let error = write_atomic_capability_file(&root, "media.json", b"new manifest")
            .expect_err("injected pre-commit failure");

        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
        assert_eq!(
            std::fs::read(tmp.path().join("media.json")).unwrap(),
            b"old manifest"
        );
    }

    #[test]
    fn removed_copy_is_unlinked_and_does_not_block_refavorite() {
        let tmp = tempfile::tempdir().unwrap();
        let source = src_file(tmp.path(), "again.mp4", b"favorite again");
        let store = LibraryStore::new(tmp.path().join("lib"));
        let first = store.favorite(&req(&source, "video", None)).unwrap();
        let first_path = store.stored_path(&first.id).unwrap().unwrap();

        assert!(store.remove(&first.id).unwrap());
        assert!(!first_path.exists());
        assert!(store.entries().unwrap().is_empty());
        assert!(store.stored_path(&first.id).unwrap().is_none());
        assert!(store.stored_ids_verified().unwrap().is_empty());

        let second = store.favorite(&req(&source, "video", None)).unwrap();
        assert_eq!(second.id, first.id);
        let second_path = store.stored_path(&second.id).unwrap().unwrap();
        assert_ne!(second_path, first_path);
        assert_eq!(std::fs::read(second_path).unwrap(), b"favorite again");
        assert_eq!(
            store.stored_ids_verified().unwrap(),
            HashSet::from([second.id])
        );
    }

    #[test]
    fn failed_remove_manifest_commit_preserves_entry_and_stored_copy() {
        let tmp = tempfile::tempdir().unwrap();
        let lib = tmp.path().join("lib");
        let source = src_file(tmp.path(), "a.mp4", b"must survive failed removal");
        let store = LibraryStore::new(&lib);
        let entry = store.favorite(&req(&source, "video", None)).unwrap();
        let stored = store.stored_path(&entry.id).unwrap().unwrap();
        let manifest_path = lib.join(MANIFEST_NAME);
        let manifest_before = std::fs::read(&manifest_path).unwrap();
        std::fs::remove_file(&manifest_path).unwrap();
        std::fs::create_dir(&manifest_path).unwrap();

        assert!(store.remove(&entry.id).is_err());
        std::fs::remove_dir(&manifest_path).unwrap();
        std::fs::write(&manifest_path, manifest_before).unwrap();
        assert!(store.contains(&entry.id).unwrap());
        assert_eq!(
            store.stored_path(&entry.id).unwrap().as_deref(),
            Some(stored.as_path())
        );
        assert!(stored.is_file());
    }

    #[test]
    fn stale_transaction_temp_does_not_block_a_later_manifest_commit() {
        let tmp = tempfile::tempdir().unwrap();
        let source = src_file(tmp.path(), "clip.mp4", b"fresh transaction");
        let store = LibraryStore::new(tmp.path().join("lib"));
        let legacy_tmp = store.root.join(format!("{MANIFEST_NAME}.tmp"));
        let unique_tmp = store.root.join(format!(".{MANIFEST_NAME}.0.{:020}.tmp", 7));
        std::fs::write(&legacy_tmp, b"crashed").unwrap();
        std::fs::write(&unique_tmp, b"crashed").unwrap();

        let entry = store.favorite(&req(&source, "video", None)).unwrap();

        assert_eq!(store.entries().unwrap(), vec![entry]);
        assert_eq!(std::fs::read(legacy_tmp).unwrap(), b"crashed");
        assert_eq!(std::fs::read(unique_tmp).unwrap(), b"crashed");
    }

    #[test]
    fn failed_directory_sync_keeps_backup_without_orphaning_committed_media() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("lib");
        let store = LibraryStore::new(&root);
        let first = src_file(tmp.path(), "first.mp4", b"first");
        let second = src_file(tmp.path(), "second.mp4", b"second");
        let first_entry = store.favorite(&req(&first, "video", None)).unwrap();
        FAIL_MANIFEST_DIRECTORY_SYNC.with(|fail| fail.set(true));
        let second_entry = store.favorite(&req(&second, "video", None)).unwrap();
        assert_eq!(store.entries().unwrap().len(), 2);
        assert_eq!(
            std::fs::read(store.stored_path(&second_entry.id).unwrap().unwrap()).unwrap(),
            b"second"
        );
        let backups = store.manifest_backups().unwrap();
        assert_eq!(
            backups.len(),
            1,
            "a failed namespace flush must preserve recovery"
        );
        let bytes = read_nofollow(&store.capabilities().unwrap().root, &backups[0]).unwrap();
        assert_eq!(decode_manifest(&bytes).unwrap().entries, [first_entry]);
        let reopened = LibraryStore::new(root);
        reopened.reconcile_storage().unwrap();
        assert!(reopened.manifest_backups().unwrap().is_empty());
        assert_eq!(reopened.entries().unwrap().len(), 2);
    }

    #[test]
    fn backup_cleanup_keeps_an_existing_reader_snapshot_intact() {
        let tmp = tempfile::tempdir().unwrap();
        let store = LibraryStore::new(tmp.path().join("lib"));
        let first = src_file(tmp.path(), "first.mp4", b"first version");
        store.favorite(&req(&first, "video", None)).unwrap();
        let path = store.root().join(MANIFEST_NAME);
        let expected = std::fs::read(&path).unwrap();
        let mut reader = std::fs::File::open(&path).unwrap();
        let second = src_file(tmp.path(), "second.mp4", b"second version");
        store.favorite(&req(&second, "video", None)).unwrap();
        let mut snapshot = Vec::new();
        reader.read_to_end(&mut snapshot).unwrap();
        assert_eq!(
            snapshot, expected,
            "unlink must not truncate an active reader"
        );
    }

    #[test]
    fn startup_reclaims_known_empty_copies_but_preserves_foreign_empty_files() {
        let tmp = tempfile::tempdir().unwrap();
        let store = LibraryStore::new(tmp.path().join("lib"));
        let source = src_file(tmp.path(), "keep.mp4", b"kept bytes");
        let entry = store.favorite(&req(&source, "video", None)).unwrap();
        let legacy = store
            .files_dir()
            .join(format!("{}.legacy.mp4", "0".repeat(64)));
        let foreign = store.files_dir().join("notes.txt");
        std::fs::write(&legacy, []).unwrap();
        std::fs::write(&foreign, []).unwrap();
        store.reconcile_storage().unwrap();
        assert!(!legacy.exists());
        assert!(foreign.exists());
        assert_eq!(
            std::fs::read(store.stored_path(&entry.id).unwrap().unwrap()).unwrap(),
            b"kept bytes"
        );
    }

    #[test]
    fn dropped_preparations_leave_no_staging_directory_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let store = LibraryStore::new(tmp.path().join("lib"));
        let source = src_file(tmp.path(), "unused.mp4", b"unpublished bytes");
        for _ in 0..8 {
            drop(
                store
                    .prepare_favorite(&req(&source, "video", None))
                    .unwrap(),
            );
        }
        assert_eq!(
            std::fs::read_dir(store.files_dir().join(STAGING_SUBDIR))
                .unwrap()
                .count(),
            0
        );
    }

    #[test]
    fn successful_manifest_commits_remove_retained_backups() {
        let tmp = tempfile::tempdir().unwrap();
        let store = LibraryStore::new(tmp.path().join("lib"));
        for index in 0..8 {
            let source = src_file(
                tmp.path(),
                &format!("clip-{index}.mp4"),
                format!("manifest version {index}").as_bytes(),
            );
            store.favorite(&req(&source, "video", None)).unwrap();
        }

        let root = store.capabilities().unwrap().root.entries().unwrap();
        let backups = root
            .filter_map(std::result::Result::ok)
            .filter(|entry| is_manifest_backup(&entry.file_name()))
            .collect::<Vec<_>>();
        assert!(backups.is_empty(), "committed backups must not accumulate");
        assert!(backups
            .iter()
            .all(|entry| entry.metadata().is_ok_and(|metadata| metadata.len() == 0)));
    }

    #[test]
    fn committed_backup_cleanup_failure_does_not_fail_or_orphan_publication() {
        let tmp = tempfile::tempdir().unwrap();
        let store = LibraryStore::new(tmp.path().join("lib"));
        let first_source = src_file(tmp.path(), "first.mp4", b"first manifest entry");
        let second_source = src_file(tmp.path(), "second.mp4", b"second manifest entry");
        store.favorite(&req(&first_source, "video", None)).unwrap();
        FAIL_COMMITTED_BACKUP_CLEANUP.with(|fail| fail.set(true));

        let second = store
            .favorite(&req(&second_source, "video", None))
            .expect("post-commit backup cleanup is best effort");

        assert!(store.contains(&second.id).unwrap());
        assert_eq!(store.entries().unwrap().len(), 2);
        assert_eq!(
            std::fs::read(store.stored_path(&second.id).unwrap().unwrap()).unwrap(),
            b"second manifest entry"
        );
    }

    #[test]
    fn startup_restores_a_retained_manifest_backup_when_canonical_is_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let source = src_file(tmp.path(), "clip.mp4", b"recover backup");
        let library_root = tmp.path().join("lib");
        let store = LibraryStore::new(&library_root);
        let entry = store.favorite(&req(&source, "video", None)).unwrap();
        let backup = library_root.join(format!(
            ".{MANIFEST_NAME}.{}.{:020}.backup",
            std::process::id(),
            42
        ));
        std::fs::rename(library_root.join(MANIFEST_NAME), &backup).unwrap();
        let reopened = LibraryStore::new(&library_root);

        reopened.reconcile_storage().unwrap();

        assert_eq!(reopened.entries().unwrap(), vec![entry]);
        assert!(library_root.join(MANIFEST_NAME).is_file());
        assert!(!backup.exists());
    }

    #[test]
    fn invalid_canonical_manifest_preserves_a_recoverable_backup() {
        let tmp = tempfile::tempdir().unwrap();
        let library_root = tmp.path().join("lib");
        let store = LibraryStore::new(&library_root);
        let backup = library_root.join(format!(
            ".{MANIFEST_NAME}.{}.{:020}.backup",
            std::process::id(),
            43
        ));
        std::fs::write(&backup, br#"{"version":1,"entries":[]}"#).unwrap();
        std::fs::write(library_root.join(MANIFEST_NAME), b"not json").unwrap();

        store
            .reconcile_storage()
            .expect_err("invalid canonical manifest must fail closed");

        assert!(backup.is_file());
        assert_eq!(
            std::fs::read(library_root.join(MANIFEST_NAME)).unwrap(),
            b"not json"
        );
    }

    #[test]
    fn semantic_manifest_errors_preserve_content_and_backups() {
        let tmp = tempfile::tempdir().unwrap();
        let source = src_file(tmp.path(), "clip.mp4", b"owned media bytes");
        let library_root = tmp.path().join("lib");
        let store = LibraryStore::new(&library_root);
        let entry = store.favorite(&req(&source, "video", None)).unwrap();
        let stored = store.stored_path(&entry.id).unwrap().unwrap();
        let stored_bytes = std::fs::read(&stored).unwrap();
        let canonical = library_root.join(MANIFEST_NAME);
        let valid_manifest = std::fs::read(&canonical).unwrap();
        let backup = library_root.join(format!(
            ".{MANIFEST_NAME}.{}.{:020}.backup",
            std::process::id(),
            91
        ));
        std::fs::write(&backup, &valid_manifest).unwrap();
        let duplicate = format!(
            r#"{{"version":1,"entries":[{{"id":"{0}","type":"video","favoritedAt":1.0}},{{"id":"{0}","type":"video","favoritedAt":2.0}}]}}"#,
            entry.id
        );
        let malformed = "0".repeat(63);
        let cases = [
            "{}".to_string(),
            r#"{"version":0,"entries":[]}"#.to_string(),
            r#"{"version":2,"entries":[],"futureField":true}"#.to_string(),
            duplicate,
            format!(
                r#"{{"version":1,"entries":[{{"id":"{malformed}","type":"video","favoritedAt":1.0}}]}}"#
            ),
            format!(
                r#"{{"version":1,"entries":[{{"id":"{}","favoritedAt":1.0}}]}}"#,
                entry.id
            ),
            format!(
                r#"{{"version":1,"entries":[{{"id":"{}","type":"video"}}]}}"#,
                entry.id
            ),
            format!(
                r#"{{"version":1,"entries":[{{"id":"{}","type":"video","favoritedAt":1.0,"unknown":true}}]}}"#,
                entry.id
            ),
            r#"{"version":1,"entries":[],"unknownField":"preserve me"}"#.to_string(),
        ];

        for invalid in cases {
            std::fs::write(&canonical, invalid.as_bytes()).unwrap();
            assert!(store.reconcile_storage().is_err(), "accepted {invalid}");
            assert!(store.set_category(&entry.id, Some("x".into())).is_err());
            assert_eq!(std::fs::read(&canonical).unwrap(), invalid.as_bytes());
            assert_eq!(std::fs::read(&backup).unwrap(), valid_manifest);
            assert_eq!(std::fs::read(&stored).unwrap(), stored_bytes);
        }
    }

    #[test]
    fn manifest_roundtrips_all_fields() {
        let tmp = tempfile::tempdir().unwrap();
        let lib = tmp.path().join("lib");
        let a = src_file(tmp.path(), "a.png", b"img");
        let store = LibraryStore::new(&lib);
        let r = FavoriteRequest {
            source: &a,
            kind: "image",
            category: Some("logo".to_string()),
            favorited_at: 1_718_900_000.5,
            thumb: Some("data:thumb".to_string()),
        };
        let e = store.favorite(&r).unwrap();
        // Reload from a fresh store instance to exercise full serde roundtrip.
        let reopened = LibraryStore::new(&lib);
        let got = reopened.entries().unwrap();
        assert_eq!(got, vec![e]);
        assert_eq!(got[0].thumb.as_deref(), Some("data:thumb"));
        assert_eq!(got[0].favorited_at, 1_718_900_000.5);
    }

    #[test]
    fn entry_json_uses_camelcase_and_type_key() {
        let e = LibraryEntry {
            id: "abc".to_string(),
            kind: "video".to_string(),
            category: None,
            favorited_at: 1.0,
            source: None,
            thumb: None,
        };
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains("\"type\":\"video\""));
        assert!(json.contains("\"favoritedAt\":1.0"));
    }

    #[test]
    fn set_category_updates_one_entry() {
        let tmp = tempfile::tempdir().unwrap();
        let lib = tmp.path().join("lib");
        let a = src_file(tmp.path(), "a.mp4", b"cat");
        let store = LibraryStore::new(&lib);
        let e = store.favorite(&req(&a, "video", None)).unwrap();

        let updated = store
            .set_category(&e.id, Some("broll".to_string()))
            .unwrap()
            .unwrap();
        assert_eq!(updated.category.as_deref(), Some("broll"));
        // Persisted across a fresh store instance.
        let got = LibraryStore::new(&lib).entries().unwrap();
        assert_eq!(got[0].category.as_deref(), Some("broll"));
        // Unknown id yields None.
        assert!(store.set_category("nope", None).unwrap().is_none());
        // Clearing works.
        store.set_category(&e.id, None).unwrap();
        assert_eq!(store.entries().unwrap()[0].category, None);
    }

    #[test]
    fn rename_category_moves_matching_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let lib = tmp.path().join("lib");
        let a = src_file(tmp.path(), "a.mp4", b"x");
        let b = src_file(tmp.path(), "b.mp4", b"y");
        let c = src_file(tmp.path(), "c.mp4", b"z");
        let store = LibraryStore::new(&lib);
        store.favorite(&req(&a, "video", Some("old"))).unwrap();
        store.favorite(&req(&b, "video", Some("old"))).unwrap();
        store.favorite(&req(&c, "video", Some("keep"))).unwrap();

        let changed = store
            .rename_category("old", Some("new".to_string()))
            .unwrap();
        assert_eq!(changed, 2);
        assert_eq!(store.entries_in_category(Some("new")).unwrap().len(), 2);
        assert_eq!(store.entries_in_category(Some("keep")).unwrap().len(), 1);
        // No match is a no-op.
        assert_eq!(store.rename_category("missing", None).unwrap(), 0);
    }

    #[test]
    fn default_library_dir_ends_with_app_and_library() {
        if let Some(dir) = default_library_dir() {
            assert!(dir.ends_with(Path::new(APP_DIR).join(LIBRARY_DIR)));
        }
    }
}
