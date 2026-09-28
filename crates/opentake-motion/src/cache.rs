//! Content-hash cache for rendered motion clips (docs/MOTION-GRAPHICS-PLUGIN.md
//! §2): the cache key is a SHA-256 over everything that affects the pixels —
//! the source (code or template id + params), fps, width, height, and the
//! transparency flag. Same inputs ⇒ same key ⇒ reuse the already-rendered frames;
//! change the source or any param and the key changes, so the next render misses
//! and recomputes. This is the standard content-addressed-cache pattern: the key
//! is path-independent and self-invalidating.
//!
//! The keying ([`content_hash`]) is **pure** and unit-tested with no filesystem.
//! [`MotionCache`] is the thin directory wrapper that maps a key to a folder and
//! reports hit/miss; the renderer writes frames into that folder.
//!
//! The cache is bounded: [`MotionCache::evict`] removes the least recently used
//! entries above a byte budget (and abandoned partial renders past an age
//! limit), and [`MotionCache::clear`] empties it. Both skip every directory a
//! [`MotionCachePin`] holds, so a render in progress, or frames still being
//! encoded or read, are never deleted underneath their owner.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use crate::error::{MotionError, MotionResult};
use crate::source::{MotionRenderRequest, MotionSource, ParamValue};

const COMPLETION_MARKER_FILE: &str = ".opentake-motion-complete-v4";
/// Written when a render starts in a directory. Frames next to it were
/// produced by the current capture pipeline, so an interrupted render may
/// resume from them; frames without it are never reused. Bump it together
/// with the completion marker whenever the captured pixels change.
const PARTIAL_MARKER_FILE: &str = ".opentake-motion-partial-v4";
/// Evicted entries are renamed to this prefix before deletion so a concurrent
/// cache lookup can never observe a half-deleted frame set.
const EVICTING_PREFIX: &str = ".evicting-";
static COMPLETION_MARKER_COUNTER: AtomicU64 = AtomicU64::new(0);
static EVICTION_COUNTER: AtomicU64 = AtomicU64::new(0);
static PINNED_DIRS: OnceLock<Mutex<HashMap<PathBuf, usize>>> = OnceLock::new();

/// Default byte budget for one Motion frame cache. Single-frame previews and
/// abandoned partial renders are rebuilt on demand, so the cache keeps only
/// the most recently used entries within this budget.
pub const DEFAULT_CACHE_MAX_BYTES: u64 = 512 * 1024 * 1024;

/// Default age after which an unfinished (not completed) render directory is
/// removed even when the cache is within its byte budget.
pub const DEFAULT_INCOMPLETE_RENDER_MAX_AGE: Duration = Duration::from_secs(24 * 60 * 60);

/// Bounds applied by [`MotionCache::evict`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MotionCacheLimits {
    /// Total bytes the cache may keep; least recently used entries beyond it
    /// are removed.
    pub max_bytes: u64,
    /// Unfinished render directories untouched for this long are removed.
    pub max_incomplete_age: Duration,
}

impl Default for MotionCacheLimits {
    fn default() -> Self {
        MotionCacheLimits {
            max_bytes: DEFAULT_CACHE_MAX_BYTES,
            max_incomplete_age: DEFAULT_INCOMPLETE_RENDER_MAX_AGE,
        }
    }
}

/// What an eviction or clear pass did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MotionCacheSweep {
    /// Cache entries (render directories) removed.
    pub removed_entries: usize,
    /// Bytes those entries held.
    pub removed_bytes: u64,
    /// Bytes still held by the remaining entries, pinned ones included.
    pub retained_bytes: u64,
}

/// Keeps one render directory out of [`MotionCache::evict`] and
/// [`MotionCache::clear`] while its frames are being written, encoded or read.
/// Pins are process-wide, so every renderer and cache handle sharing a
/// directory observes them. Dropping the pin releases it.
#[must_use = "dropping the pin immediately makes the directory evictable"]
#[derive(Debug)]
pub struct MotionCachePin {
    dir: Option<PathBuf>,
}

/// Compute the content hash (lowercase hex SHA-256) for a render request.
///
/// We feed a canonical, unambiguous byte stream into the hash — each field
/// length-prefixed or delimited so that `("a","bc")` and `("ab","c")` can't
/// collide. `BTreeMap` param ordering makes the template arm deterministic.
pub fn content_hash(req: &MotionRenderRequest) -> String {
    let mut hasher = Sha256::new();

    // Version the key format so a future change to what we hash invalidates old
    // entries instead of silently colliding.
    hasher.update(b"opentake-motion/v2\n");

    // Numeric/flags first (fixed-width, no ambiguity).
    hasher.update(b"fps=");
    hasher.update(req.fps.to_le_bytes());
    hasher.update(b";frames=");
    hasher.update(req.duration_frames.to_le_bytes());
    hasher.update(b";start=");
    hasher.update(req.start_frame.to_le_bytes());
    hasher.update(b";w=");
    hasher.update(req.width.to_le_bytes());
    hasher.update(b";h=");
    hasher.update(req.height.to_le_bytes());
    hasher.update(b";transparent=");
    hasher.update([req.transparent as u8]);
    hasher.update(b"\n");

    // The source.
    match &req.source {
        MotionSource::Code { html_css_js } => {
            hasher.update(b"source=code;len=");
            hasher.update((html_css_js.len() as u64).to_le_bytes());
            hasher.update(b";body=");
            hasher.update(html_css_js.as_bytes());
        }
        MotionSource::Template { id, params } => {
            hasher.update(b"source=template;id_len=");
            hasher.update((id.len() as u64).to_le_bytes());
            hasher.update(b";id=");
            hasher.update(id.as_bytes());
            hasher.update(b";params=");
            // BTreeMap iterates in sorted key order ⇒ deterministic.
            for (name, value) in params {
                hasher.update((name.len() as u64).to_le_bytes());
                hasher.update(name.as_bytes());
                hasher.update(b"=");
                hash_param_value(&mut hasher, value);
                hasher.update(b";");
            }
        }
    }

    hex::encode(hasher.finalize())
}

/// Fold one param value into the hash with a type tag so a string `"1"` and a
/// number `1` hash differently.
fn hash_param_value(hasher: &mut Sha256, value: &ParamValue) {
    match value {
        ParamValue::String(s) => {
            hasher.update(b"s:");
            hasher.update((s.len() as u64).to_le_bytes());
            hasher.update(s.as_bytes());
        }
        ParamValue::Number(n) => {
            hasher.update(b"n:");
            // Canonical bit pattern; normalize -0.0 to 0.0 so they don't diverge.
            let bits = if *n == 0.0 { 0.0f64 } else { *n }.to_bits();
            hasher.update(bits.to_le_bytes());
        }
        ParamValue::Bool(b) => {
            hasher.update(b"b:");
            hasher.update([*b as u8]);
        }
        ParamValue::Color(c) => {
            hasher.update(b"c:");
            // Hash colors case-insensitively (#ABC == #abc on the wire).
            let lower = c.to_ascii_lowercase();
            hasher.update((lower.len() as u64).to_le_bytes());
            hasher.update(lower.as_bytes());
        }
    }
}

/// A content-addressed frame cache rooted at a directory. Each render key maps to
/// `root/<hash>/`; the renderer fills that folder with frame files and the cache
/// reports whether it already exists & looks complete.
#[derive(Clone, Debug)]
pub struct MotionCache {
    root: PathBuf,
}

impl MotionCache {
    /// Create a cache rooted at `root` (not created on disk until
    /// [`MotionCache::ensure_dir`] / a render writes into it).
    pub fn new(root: impl Into<PathBuf>) -> Self {
        MotionCache { root: root.into() }
    }

    /// The cache root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The directory a given request's frames live in (`root/<hash>`).
    pub fn dir_for(&self, req: &MotionRenderRequest) -> PathBuf {
        self.root.join(content_hash(req))
    }

    /// The directory for an explicit hash key.
    pub fn dir_for_hash(&self, hash: &str) -> PathBuf {
        self.root.join(hash)
    }

    /// Whether a complete render for this request is already cached. "Complete"
    /// means the directory holds the exact expected frame set and an atomically
    /// published completion marker. A render that produced every frame but
    /// failed during browser shutdown therefore remains a miss.
    pub fn is_cached(&self, req: &MotionRenderRequest) -> bool {
        let dir = self.dir_for(req);
        completion_marker(&dir).is_file()
            && has_exact_frame_files(&dir, req.duration_frames as usize)
    }

    /// Create the cache directory for a request, returning its path.
    pub fn ensure_dir(&self, req: &MotionRenderRequest) -> MotionResult<PathBuf> {
        let dir = self.dir_for(req);
        std::fs::create_dir_all(&dir)?;
        Ok(dir)
    }

    /// Prepare a cache directory for a render. Removing the marker first
    /// makes every subsequent write fail-closed until completion is published.
    ///
    /// Frames of an interrupted render stay in place for [`Self::completed_prefix`]
    /// only when the directory carries the current partial-render marker;
    /// frames from any other pipeline version are removed first.
    pub(crate) fn begin_render(&self, req: &MotionRenderRequest) -> MotionResult<PathBuf> {
        let dir = self.ensure_dir(req)?;
        Self::remove_completion_marker(&dir)?;
        let marker = partial_marker(&dir);
        if marker.is_file() {
            touch_file(&marker);
        } else {
            remove_frame_files(&dir)?;
            publish_marker(&dir, PARTIAL_MARKER_FILE, b"opentake-motion-partial/v4\n")?;
        }
        Ok(dir)
    }

    /// Publish render completion with a write-sync-rename sequence so readers
    /// can never observe a partially written marker.
    pub(crate) fn mark_complete(dir: &Path) -> MotionResult<()> {
        publish_marker(dir, COMPLETION_MARKER_FILE, b"opentake-motion-cache/v4\n")
    }

    pub(crate) fn remove_completion_marker(dir: &Path) -> MotionResult<()> {
        remove_file_if_present(&completion_marker(dir))
    }

    /// Remove everything a render wrote into `dir`: frames and both markers.
    /// Used when a render fails in a way whose frames must not be resumed.
    #[cfg(any(feature = "chromium", test))]
    pub(crate) fn discard_render_output(dir: &Path) -> MotionResult<()> {
        remove_file_if_present(&completion_marker(dir))?;
        remove_frame_files(dir)?;
        remove_file_if_present(&partial_marker(dir))
    }

    /// Number of leading frames of `req` that an interrupted render already
    /// wrote completely into `dir`. The first missing or damaged frame is
    /// removed so it is rendered again; a render resumes from the returned
    /// index.
    #[cfg(any(feature = "chromium", test))]
    pub(crate) fn completed_prefix(dir: &Path, req: &MotionRenderRequest) -> MotionResult<usize> {
        if !partial_marker(dir).is_file() {
            return Ok(0);
        }
        for index in 0..req.duration_frames as usize {
            let frame = Self::frame_file(dir, index);
            if !frame_file_is_intact(&frame, req.width, req.height) {
                remove_file_if_present(&frame)?;
                return Ok(index);
            }
        }
        Ok(req.duration_frames as usize)
    }

    /// Record a cache hit so least-recently-used eviction keeps this entry.
    pub(crate) fn touch(dir: &Path) {
        touch_file(&completion_marker(dir));
    }

    /// The expected per-frame file path inside a render dir: zero-padded so
    /// lexical order == playback order (`frame_00000.png`).
    pub fn frame_file(dir: &Path, frame_index: usize) -> PathBuf {
        dir.join(format!("frame_{frame_index:05}.png"))
    }

    /// Keep the directory of `req` out of eviction and clearing until the
    /// returned pin is dropped or discarded.
    pub fn pin(&self, req: &MotionRenderRequest) -> MotionCachePin {
        MotionCachePin::new(self.dir_for(req))
    }

    /// Remove least recently used entries until the cache fits
    /// `limits.max_bytes`, and every unfinished render untouched for
    /// `limits.max_incomplete_age`. Pinned entries and unknown files are never
    /// removed. Returns the first removal failure after trying every candidate.
    pub fn evict(&self, limits: MotionCacheLimits) -> MotionResult<MotionCacheSweep> {
        let now = SystemTime::now();
        let mut failures = Vec::new();
        let mut entries = self.entries(&mut failures)?;
        let mut sweep = MotionCacheSweep {
            retained_bytes: entries.iter().map(|entry| entry.bytes).sum(),
            ..MotionCacheSweep::default()
        };
        entries.sort_by_key(|entry| entry.last_used);
        for entry in entries {
            let abandoned = !entry.complete
                && now
                    .duration_since(entry.last_used)
                    .is_ok_and(|age| age >= limits.max_incomplete_age);
            if sweep.retained_bytes <= limits.max_bytes && !abandoned {
                continue;
            }
            match self.remove_unpinned(&entry.path) {
                Ok(true) => {
                    sweep.removed_entries += 1;
                    sweep.removed_bytes += entry.bytes;
                    sweep.retained_bytes = sweep.retained_bytes.saturating_sub(entry.bytes);
                }
                Ok(false) => {}
                Err(error) => failures.push(format!("{}: {error}", entry.path.display())),
            }
        }
        sweep_result(sweep, failures)
    }

    /// Remove every unpinned entry and recreate the (empty) cache root. Pinned
    /// entries, which renders or encoders are using, stay in place.
    pub fn clear(&self) -> MotionResult<MotionCacheSweep> {
        let mut failures = Vec::new();
        let entries = self.entries(&mut failures)?;
        let mut sweep = MotionCacheSweep::default();
        for entry in entries {
            match self.remove_unpinned(&entry.path) {
                Ok(true) => {
                    sweep.removed_entries += 1;
                    sweep.removed_bytes += entry.bytes;
                }
                Ok(false) => sweep.retained_bytes += entry.bytes,
                Err(error) => {
                    sweep.retained_bytes += entry.bytes;
                    failures.push(format!("{}: {error}", entry.path.display()));
                }
            }
        }
        if let Err(error) = std::fs::create_dir_all(&self.root) {
            failures.push(format!("recreate {}: {error}", self.root.display()));
        }
        sweep_result(sweep, failures)
    }

    /// Scan the cache root: content-hash directories become eviction
    /// candidates; leftovers of an interrupted eviction are deleted here.
    fn entries(&self, failures: &mut Vec<String>) -> MotionResult<Vec<CacheEntry>> {
        let read = match std::fs::read_dir(&self.root) {
            Ok(read) => read,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error.into()),
        };
        let mut entries = Vec::new();
        for entry in read.flatten() {
            let path = entry.path();
            let Ok(metadata) = std::fs::symlink_metadata(&path) else {
                continue;
            };
            // Symlinks and plain files are never followed or removed.
            if !metadata.is_dir() {
                continue;
            }
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            if name.starts_with(EVICTING_PREFIX) {
                if let Err(error) = std::fs::remove_dir_all(&path) {
                    failures.push(format!("{}: {error}", path.display()));
                }
                continue;
            }
            if !is_content_hash(name) {
                continue;
            }
            // Completion is refreshed on every cache hit and a partial render
            // on every (re)start; entries without either predate both markers.
            let last_used = modified(&completion_marker(&path))
                .or_else(|| modified(&partial_marker(&path)))
                .or_else(|| metadata.modified().ok())
                .unwrap_or(UNIX_EPOCH);
            entries.push(CacheEntry {
                bytes: tree_bytes(&path),
                complete: completion_marker(&path).is_file(),
                last_used,
                path,
            });
        }
        Ok(entries)
    }

    /// Move an unpinned entry out of the cache namespace, then delete it. The
    /// rename happens under the pin lock, so a render that pins the entry
    /// afterwards simply starts from an empty directory.
    fn remove_unpinned(&self, dir: &Path) -> std::io::Result<bool> {
        let trash = {
            let pins = pinned_dirs();
            if pins.contains_key(dir) {
                return Ok(false);
            }
            let Some(trash) = trash_path(dir) else {
                return Ok(false);
            };
            match std::fs::rename(dir, &trash) {
                Ok(()) => trash,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
                Err(error) => return Err(error),
            }
        };
        std::fs::remove_dir_all(trash).map(|()| true)
    }
}

impl MotionCachePin {
    fn new(dir: PathBuf) -> Self {
        *pinned_dirs().entry(dir.clone()).or_insert(0) += 1;
        MotionCachePin { dir: Some(dir) }
    }

    /// The pinned render directory.
    pub fn dir(&self) -> &Path {
        self.dir
            .as_deref()
            .expect("a live cache pin always holds its directory")
    }

    /// Release the pin and delete the directory unless another pin still holds
    /// it, e.g. once its frames were encoded into a published video. Returns
    /// whether the directory was removed.
    pub fn discard(mut self) -> MotionResult<bool> {
        let dir = self
            .dir
            .take()
            .expect("a live cache pin always holds its directory");
        let trash = {
            let mut pins = pinned_dirs();
            if release(&mut pins, &dir) > 0 {
                return Ok(false);
            }
            let Some(trash) = trash_path(&dir) else {
                return Ok(false);
            };
            match std::fs::rename(&dir, &trash) {
                Ok(()) => trash,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
                Err(error) => return Err(error.into()),
            }
        };
        std::fs::remove_dir_all(trash)?;
        Ok(true)
    }
}

impl Drop for MotionCachePin {
    fn drop(&mut self) {
        if let Some(dir) = self.dir.take() {
            release(&mut pinned_dirs(), &dir);
        }
    }
}

struct CacheEntry {
    path: PathBuf,
    bytes: u64,
    complete: bool,
    last_used: SystemTime,
}

fn pinned_dirs() -> MutexGuard<'static, HashMap<PathBuf, usize>> {
    PINNED_DIRS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Drop one pin of `dir`, returning how many remain.
fn release(pins: &mut HashMap<PathBuf, usize>, dir: &Path) -> usize {
    let Some(count) = pins.get_mut(dir) else {
        return 0;
    };
    *count = count.saturating_sub(1);
    let remaining = *count;
    if remaining == 0 {
        pins.remove(dir);
    }
    remaining
}

/// A sibling name outside the content-hash namespace for an entry that is
/// about to be deleted.
fn trash_path(dir: &Path) -> Option<PathBuf> {
    let name = dir.file_name()?.to_str()?;
    let counter = EVICTION_COUNTER.fetch_add(1, Ordering::Relaxed);
    Some(dir.parent()?.join(format!(
        "{EVICTING_PREFIX}{name}-{pid}-{counter}",
        pid = std::process::id()
    )))
}

fn sweep_result(sweep: MotionCacheSweep, failures: Vec<String>) -> MotionResult<MotionCacheSweep> {
    if failures.is_empty() {
        Ok(sweep)
    } else {
        Err(MotionError::Io(std::io::Error::other(format!(
            "Motion frame cache cleanup failed: {}",
            failures.join("; ")
        ))))
    }
}

fn is_content_hash(name: &str) -> bool {
    name.len() == 64
        && name
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Best-effort recency update; a failure only makes the entry look older to
/// eviction.
fn touch_file(path: &Path) {
    if let Ok(file) = std::fs::OpenOptions::new().write(true).open(path) {
        let _ = file.set_modified(SystemTime::now());
    }
}

fn modified(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).ok()?.modified().ok()
}

/// Recursive size of the regular files under `path`; symlinks are neither
/// followed nor counted.
fn tree_bytes(path: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| {
            let path = entry.path();
            match std::fs::symlink_metadata(&path) {
                Ok(metadata) if metadata.is_dir() => tree_bytes(&path),
                Ok(metadata) if metadata.is_file() => metadata.len(),
                _ => 0,
            }
        })
        .sum()
}

/// Write `contents` to `dir/name` with a write-sync-rename sequence so
/// readers never observe a partially written marker.
fn publish_marker(dir: &Path, name: &str, contents: &[u8]) -> MotionResult<()> {
    let marker = dir.join(name);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let counter = COMPLETION_MARKER_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temporary = dir.join(format!(
        "{name}-{pid}-{nanos}-{counter}.tmp",
        pid = std::process::id()
    ));

    let result = (|| -> std::io::Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(contents)?;
        file.sync_all()?;
        drop(file);
        match std::fs::rename(&temporary, &marker) {
            Ok(()) => Ok(()),
            // Another identical renderer may have published the same key
            // concurrently. Its atomic marker is equivalent.
            Err(_) if marker.is_file() => Ok(()),
            Err(error) => Err(error),
        }
    })();
    let _ = std::fs::remove_file(&temporary);
    result.map_err(Into::into)
}

fn remove_file_if_present(path: &Path) -> MotionResult<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn remove_frame_files(dir: &Path) -> MotionResult<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("frame_") && name.ends_with(".png") {
            remove_file_if_present(&entry.path())?;
        }
    }
    Ok(())
}

/// A frame written by the renderer is an 8-bit RGBA PNG of the requested
/// size that ends with its `IEND` chunk. Frames are published by an atomic
/// rename, so this cheap header/trailer check rejects foreign files and a
/// frame torn by a crash without decoding pixels.
#[cfg(any(feature = "chromium", test))]
fn frame_file_is_intact(path: &Path, width: u32, height: u32) -> bool {
    use std::io::{Read, Seek, SeekFrom};

    const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    const IEND: [u8; 12] = [0, 0, 0, 0, b'I', b'E', b'N', b'D', 0xae, 0x42, 0x60, 0x82];
    let check = || -> std::io::Result<bool> {
        let mut file = std::fs::File::open(path)?;
        if !file.metadata()?.is_file() {
            return Ok(false);
        }
        let mut header = [0_u8; 26];
        file.read_exact(&mut header)?;
        let mut trailer = [0_u8; 12];
        file.seek(SeekFrom::End(-12))?;
        file.read_exact(&mut trailer)?;
        Ok(header[..8] == SIGNATURE
            && header[8..16] == [0, 0, 0, 13, b'I', b'H', b'D', b'R']
            && header[16..20] == width.to_be_bytes()
            && header[20..24] == height.to_be_bytes()
            && header[24] == 8
            && header[25] == 6
            && trailer == IEND)
    };
    check().unwrap_or(false)
}

fn completion_marker(dir: &Path) -> PathBuf {
    dir.join(COMPLETION_MARKER_FILE)
}

fn partial_marker(dir: &Path) -> PathBuf {
    dir.join(PARTIAL_MARKER_FILE)
}

fn has_exact_frame_files(dir: &Path, expected: usize) -> bool {
    count_frame_files(dir) == Some(expected)
        && (0..expected).all(|index| MotionCache::frame_file(dir, index).is_file())
}

/// Count `frame_*.png` files in a directory, or `None` if it doesn't exist.
fn count_frame_files(dir: &Path) -> Option<usize> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut n = 0usize;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("frame_") && name.ends_with(".png") {
            n += 1;
        }
    }
    Some(n)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn code_req(body: &str) -> MotionRenderRequest {
        MotionRenderRequest::new(MotionSource::code(body), 30, 60, 1920, 1080)
    }

    #[test]
    fn hash_is_64_hex_chars() {
        let h = content_hash(&code_req("<div/>"));
        assert_eq!(h.len(), 64);
        assert!(h.bytes().all(|b| b.is_ascii_hexdigit()));
    }

    #[test]
    fn hash_is_stable_for_identical_requests() {
        assert_eq!(
            content_hash(&code_req("<a/>")),
            content_hash(&code_req("<a/>"))
        );
    }

    #[test]
    fn hash_changes_with_source_body() {
        assert_ne!(
            content_hash(&code_req("<a/>")),
            content_hash(&code_req("<b/>"))
        );
    }

    #[test]
    fn hash_changes_with_size_fps_and_transparency() {
        let base = code_req("<x/>");
        let mut bigger = base.clone();
        bigger.width = 1280;
        assert_ne!(content_hash(&base), content_hash(&bigger));

        let mut faster = base.clone();
        faster.fps = 60;
        assert_ne!(content_hash(&base), content_hash(&faster));

        let opaque = base.clone().with_transparent(false);
        assert_ne!(content_hash(&base), content_hash(&opaque));

        let mut longer = base.clone();
        longer.duration_frames = 120;
        assert_ne!(content_hash(&base), content_hash(&longer));

        let offset = base.clone().with_start_frame(1);
        assert_ne!(content_hash(&base), content_hash(&offset));
    }

    #[test]
    fn template_param_order_does_not_affect_hash() {
        // BTreeMap canonicalizes order, so two insert orders hash identically.
        let mut a = BTreeMap::new();
        a.insert("b".to_string(), ParamValue::Number(2.0));
        a.insert("a".to_string(), ParamValue::String("x".into()));
        let mut b = BTreeMap::new();
        b.insert("a".to_string(), ParamValue::String("x".into()));
        b.insert("b".to_string(), ParamValue::Number(2.0));

        let ra = MotionRenderRequest::new(
            MotionSource::Template {
                id: "t".into(),
                params: a,
            },
            30,
            60,
            100,
            100,
        );
        let rb = MotionRenderRequest::new(
            MotionSource::Template {
                id: "t".into(),
                params: b,
            },
            30,
            60,
            100,
            100,
        );
        assert_eq!(content_hash(&ra), content_hash(&rb));
    }

    #[test]
    fn template_param_value_type_changes_hash() {
        let mut as_str = BTreeMap::new();
        as_str.insert("v".to_string(), ParamValue::String("1".into()));
        let mut as_num = BTreeMap::new();
        as_num.insert("v".to_string(), ParamValue::Number(1.0));

        let r1 = MotionRenderRequest::new(
            MotionSource::Template {
                id: "t".into(),
                params: as_str,
            },
            30,
            60,
            100,
            100,
        );
        let r2 = MotionRenderRequest::new(
            MotionSource::Template {
                id: "t".into(),
                params: as_num,
            },
            30,
            60,
            100,
            100,
        );
        assert_ne!(content_hash(&r1), content_hash(&r2));
    }

    #[test]
    fn code_and_template_with_same_string_do_not_collide() {
        let code = MotionRenderRequest::new(MotionSource::code("t"), 30, 60, 100, 100);
        let tmpl = MotionRenderRequest::new(MotionSource::template("t"), 30, 60, 100, 100);
        assert_ne!(content_hash(&code), content_hash(&tmpl));
    }

    #[test]
    fn dir_for_is_root_join_hash() {
        let cache = MotionCache::new("/cache/motion");
        let req = code_req("<x/>");
        let expected = PathBuf::from("/cache/motion").join(content_hash(&req));
        assert_eq!(cache.dir_for(&req), expected);
    }

    #[test]
    fn frame_file_is_zero_padded() {
        let p = MotionCache::frame_file(Path::new("/d"), 7);
        assert_eq!(p, PathBuf::from("/d/frame_00007.png"));
    }

    #[test]
    fn is_cached_false_for_missing_dir() {
        let cache = MotionCache::new("/definitely/not/here");
        assert!(!cache.is_cached(&code_req("<x/>")));
    }

    #[test]
    fn is_cached_true_only_when_frame_count_matches() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = MotionCache::new(tmp.path());
        let req = MotionRenderRequest::new(MotionSource::code("<x/>"), 30, 3, 64, 64);
        let dir = cache.ensure_dir(&req).unwrap();

        // No frames yet -> miss.
        assert!(!cache.is_cached(&req));

        // Two of three frames -> still a miss (partial render).
        std::fs::write(MotionCache::frame_file(&dir, 0), b"x").unwrap();
        std::fs::write(MotionCache::frame_file(&dir, 1), b"x").unwrap();
        assert!(!cache.is_cached(&req));

        // All three without a completion marker are still a miss: a renderer
        // may have produced every frame and then failed during shutdown.
        std::fs::write(MotionCache::frame_file(&dir, 2), b"x").unwrap();
        assert!(!cache.is_cached(&req));

        MotionCache::mark_complete(&dir).unwrap();
        assert!(cache.is_cached(&req));
        assert!(
            std::fs::read_dir(&dir)
                .unwrap()
                .flatten()
                .all(|entry| !entry.file_name().to_string_lossy().ends_with(".tmp")),
            "atomic publication must not leave a temporary marker"
        );

        // Beginning a new render invalidates completion before any frame is
        // touched, so a later renderer failure cannot reuse stale frames.
        assert_eq!(cache.begin_render(&req).unwrap(), dir);
        assert!(!cache.is_cached(&req));
    }

    #[test]
    fn completion_marker_requires_the_exact_expected_frame_set() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = MotionCache::new(tmp.path());
        let req = MotionRenderRequest::new(MotionSource::code("<x/>"), 30, 2, 64, 64);
        let dir = cache.ensure_dir(&req).unwrap();

        MotionCache::mark_complete(&dir).unwrap();
        assert!(!cache.is_cached(&req), "a marker alone is not a cache hit");

        std::fs::write(MotionCache::frame_file(&dir, 0), b"x").unwrap();
        std::fs::write(MotionCache::frame_file(&dir, 999), b"x").unwrap();
        assert!(!cache.is_cached(&req), "wrong frame names are not complete");

        std::fs::write(MotionCache::frame_file(&dir, 1), b"x").unwrap();
        assert!(!cache.is_cached(&req), "extra frame files are not complete");
    }

    #[test]
    fn completion_marker_schema_rejects_legacy_rendered_frames() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = MotionCache::new(tmp.path());
        let req = MotionRenderRequest::new(MotionSource::code("<x/>"), 30, 2, 64, 64);
        let dir = cache.ensure_dir(&req).unwrap();
        for index in 0..2 {
            std::fs::write(MotionCache::frame_file(&dir, index), b"legacy").unwrap();
        }
        std::fs::write(
            dir.join(".opentake-motion-complete-v3"),
            b"opentake-motion-cache/v3\n",
        )
        .unwrap();

        assert!(
            !cache.is_cached(&req),
            "a legacy capture marker must not validate the current renderer output"
        );
        MotionCache::mark_complete(&dir).unwrap();
        assert!(cache.is_cached(&req));
        assert_eq!(
            cache.dir_for(&req),
            dir,
            "cache directory contract is stable"
        );
    }

    fn write_frames(dir: &Path, count: usize, width: u32, height: u32) {
        for index in 0..count {
            std::fs::write(
                MotionCache::frame_file(dir, index),
                crate::renderer::encode_solid_rgba_png(width, height, [1, 2, 3, 255]),
            )
            .unwrap();
        }
    }

    /// A completed cache entry of roughly `bytes` whose last use was `age` ago.
    fn cached_entry(cache: &MotionCache, body: &str, bytes: usize, age: Duration) -> PathBuf {
        let req = MotionRenderRequest::new(MotionSource::code(body), 30, 1, 4, 4);
        let dir = cache.begin_render(&req).unwrap();
        std::fs::write(MotionCache::frame_file(&dir, 0), vec![0_u8; bytes]).unwrap();
        MotionCache::mark_complete(&dir).unwrap();
        let used = SystemTime::now() - age;
        for marker in [completion_marker(&dir), partial_marker(&dir)] {
            std::fs::OpenOptions::new()
                .write(true)
                .open(marker)
                .unwrap()
                .set_modified(used)
                .unwrap();
        }
        dir
    }

    #[test]
    fn eviction_keeps_the_most_recently_used_entries_within_the_byte_budget() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = MotionCache::new(tmp.path());
        let oldest = cached_entry(&cache, "<oldest/>", 1000, Duration::from_secs(400));
        let pinned_old = cached_entry(&cache, "<pinned/>", 1000, Duration::from_secs(300));
        let older = cached_entry(&cache, "<older/>", 1000, Duration::from_secs(200));
        let newer = cached_entry(&cache, "<newer/>", 1000, Duration::from_secs(100));
        let newest = cached_entry(&cache, "<newest/>", 1000, Duration::from_secs(1));
        let unknown = tmp.path().join("not-a-cache-entry");
        std::fs::create_dir(&unknown).unwrap();
        std::fs::write(unknown.join("keep.bin"), vec![0_u8; 5000]).unwrap();
        let pin = MotionCachePin::new(pinned_old.clone());

        let sweep = cache
            .evict(MotionCacheLimits {
                max_bytes: 3500,
                max_incomplete_age: Duration::from_secs(3600),
            })
            .unwrap();

        assert!(!oldest.exists() && !older.exists());
        assert!(
            pinned_old.is_dir(),
            "a pinned render directory is never evicted"
        );
        assert!(newer.is_dir() && newest.is_dir());
        assert!(
            unknown.join("keep.bin").is_file(),
            "unknown entries are never swept"
        );
        assert_eq!(sweep.removed_entries, 2);
        assert!(sweep.retained_bytes <= 3500, "{sweep:?}");
        assert!(
            std::fs::read_dir(tmp.path())
                .unwrap()
                .flatten()
                .all(|entry| !entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(EVICTING_PREFIX)),
            "eviction must not leave renamed entries behind"
        );

        drop(pin);
        cache
            .evict(MotionCacheLimits {
                max_bytes: 2500,
                max_incomplete_age: Duration::from_secs(3600),
            })
            .unwrap();
        assert!(
            !pinned_old.exists(),
            "a released pin makes the entry evictable"
        );
        assert!(newer.is_dir() && newest.is_dir());
    }

    #[test]
    fn eviction_removes_abandoned_partial_renders_by_age_and_keeps_recent_ones() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = MotionCache::new(tmp.path());
        let abandoned_req = MotionRenderRequest::new(MotionSource::code("<a/>"), 30, 4, 4, 4);
        let abandoned = cache.begin_render(&abandoned_req).unwrap();
        write_frames(&abandoned, 2, 4, 4);
        std::fs::OpenOptions::new()
            .write(true)
            .open(partial_marker(&abandoned))
            .unwrap()
            .set_modified(SystemTime::now() - Duration::from_secs(7200))
            .unwrap();
        let recent_req = MotionRenderRequest::new(MotionSource::code("<b/>"), 30, 4, 4, 4);
        let recent = cache.begin_render(&recent_req).unwrap();
        write_frames(&recent, 2, 4, 4);

        let sweep = cache
            .evict(MotionCacheLimits {
                max_bytes: u64::MAX,
                max_incomplete_age: Duration::from_secs(3600),
            })
            .unwrap();
        assert_eq!(sweep.removed_entries, 1);
        assert!(!abandoned.exists());
        assert_eq!(
            MotionCache::completed_prefix(&recent, &recent_req).unwrap(),
            2
        );
    }

    #[test]
    fn clear_skips_pinned_entries_and_recreates_the_root() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("motion-frames");
        let cache = MotionCache::new(&root);
        let idle = cached_entry(&cache, "<idle/>", 64, Duration::ZERO);
        let active_req = MotionRenderRequest::new(MotionSource::code("<active/>"), 30, 1, 4, 4);
        let active = cache.begin_render(&active_req).unwrap();
        let pin = cache.pin(&active_req);
        assert_eq!(pin.dir(), active);

        let sweep = cache.clear().unwrap();
        assert!(!idle.exists());
        assert!(active.is_dir(), "a render in progress keeps its directory");
        assert_eq!(sweep.removed_entries, 1);

        drop(pin);
        std::fs::remove_dir_all(&root).unwrap();
        assert_eq!(cache.clear().unwrap(), MotionCacheSweep::default());
        assert!(root.is_dir(), "clearing recreates the cache root");
    }

    #[test]
    fn discarding_a_pin_removes_its_directory_once_no_other_pin_holds_it() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = MotionCache::new(tmp.path());
        let req = MotionRenderRequest::new(MotionSource::code("<published/>"), 30, 1, 4, 4);
        let dir = cache.begin_render(&req).unwrap();
        write_frames(&dir, 1, 4, 4);
        MotionCache::mark_complete(&dir).unwrap();

        let publisher = cache.pin(&req);
        let reader = cache.pin(&req);
        assert!(
            !publisher.discard().unwrap(),
            "another pin still reads the frames"
        );
        assert!(cache.is_cached(&req));
        assert!(reader.discard().unwrap());
        assert!(!dir.exists());
        assert!(
            !cache.pin(&req).discard().unwrap(),
            "a missing directory is not an error"
        );
    }

    #[test]
    fn resumable_frames_stop_at_the_first_missing_or_damaged_frame() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = MotionCache::new(tmp.path());
        let req = MotionRenderRequest::new(MotionSource::code("<resume/>"), 30, 5, 6, 4);
        let dir = cache.begin_render(&req).unwrap();
        assert_eq!(MotionCache::completed_prefix(&dir, &req).unwrap(), 0);

        write_frames(&dir, 3, 6, 4);
        assert_eq!(MotionCache::completed_prefix(&dir, &req).unwrap(), 3);

        // A frame torn by a crash lacks its IEND trailer: it is removed and
        // rendered again.
        let torn = std::fs::read(MotionCache::frame_file(&dir, 1)).unwrap();
        std::fs::write(MotionCache::frame_file(&dir, 1), &torn[..torn.len() - 5]).unwrap();
        assert_eq!(MotionCache::completed_prefix(&dir, &req).unwrap(), 1);
        assert!(!MotionCache::frame_file(&dir, 1).exists());

        // A frame of another size never belongs to this request.
        write_frames(&dir, 2, 6, 4);
        std::fs::write(
            MotionCache::frame_file(&dir, 1),
            crate::renderer::encode_solid_rgba_png(4, 6, [0, 0, 0, 255]),
        )
        .unwrap();
        assert_eq!(MotionCache::completed_prefix(&dir, &req).unwrap(), 1);

        write_frames(&dir, 5, 6, 4);
        assert_eq!(MotionCache::completed_prefix(&dir, &req).unwrap(), 5);
        assert!(!cache.is_cached(&req), "frames alone are never a cache hit");
    }

    #[test]
    fn frames_without_the_partial_marker_are_never_resumed() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = MotionCache::new(tmp.path());
        let req = MotionRenderRequest::new(MotionSource::code("<legacy/>"), 30, 2, 6, 4);
        let dir = cache.ensure_dir(&req).unwrap();
        write_frames(&dir, 2, 6, 4);
        assert_eq!(MotionCache::completed_prefix(&dir, &req).unwrap(), 0);

        assert_eq!(cache.begin_render(&req).unwrap(), dir);
        assert!(!MotionCache::frame_file(&dir, 0).exists());
        write_frames(&dir, 1, 6, 4);
        assert_eq!(cache.begin_render(&req).unwrap(), dir);
        assert_eq!(MotionCache::completed_prefix(&dir, &req).unwrap(), 1);

        MotionCache::discard_render_output(&dir).unwrap();
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
    }
}
