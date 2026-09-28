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

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use crate::error::MotionResult;
use crate::source::{MotionRenderRequest, MotionSource, ParamValue};

const COMPLETION_MARKER_FILE: &str = ".opentake-motion-complete-v4";
/// Written when a render starts in a directory. Frames next to it were
/// produced by the current capture pipeline, so an interrupted render may
/// resume from them; frames without it are never reused. Bump it together
/// with the completion marker whenever the captured pixels change.
const PARTIAL_MARKER_FILE: &str = ".opentake-motion-partial-v4";
static COMPLETION_MARKER_COUNTER: AtomicU64 = AtomicU64::new(0);

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
        if !partial_marker(&dir).is_file() {
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

    /// The expected per-frame file path inside a render dir: zero-padded so
    /// lexical order == playback order (`frame_00000.png`).
    pub fn frame_file(dir: &Path, frame_index: usize) -> PathBuf {
        dir.join(format!("frame_{frame_index:05}.png"))
    }
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
