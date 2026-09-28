//! Source color resolution for HDR-aware frame decode.
//!
//! PQ/HLG video must be tone-mapped before it becomes RGBA8 (see
//! [`crate::color`]), so every decode needs the source's color signalling.
//! Callers that already know it — the media manifest records it at import —
//! pass a [`ColorHint::Known`] and no ffprobe runs. Otherwise the signalling is
//! probed once per file identity and kept in a small process-wide cache, so
//! decoding N frames of one file runs at most one ffprobe. The probe honours
//! the caller's cancellation token and a short deadline, and it waits
//! (boundedly and cancellably) for an ffprobe admission slot instead of
//! failing at once when other media work holds them all. A probe that still
//! fails is an error: decoding a PQ/HLG source as SDR would silently produce
//! wrong pixels. Only an input that ffprobe reads and rejects as media goes on
//! to the decoder without color, so the decoder reports why it is unreadable.

use std::collections::VecDeque;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use opentake_domain::MediaColorMetadata;

use crate::cancel::MediaCancelToken;
use crate::error::{MediaError, Result};
use crate::ff::ProbeTarget;
use crate::identity::FileStamp;

/// Upper bound for one decode-time color probe. Header probing a local file
/// takes milliseconds; the bound only stops a stuck helper.
const COLOR_PROBE_TIMEOUT: Duration = Duration::from_secs(10);
/// How long a decode-time probe queues for an ffprobe admission slot.
const COLOR_PROBE_ADMISSION_WAIT: Duration = Duration::from_secs(5);
/// Distinct sources whose probed signalling is remembered.
const COLOR_CACHE_CAPACITY: usize = 256;

/// What the caller knows about a source's color signalling.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ColorHint {
    /// Not known: probe it (at most once per file identity).
    #[default]
    Unknown,
    /// Known from durable metadata. `None` means the source reports no color
    /// signalling, which decodes as SDR. No probe runs.
    Known(Option<MediaColorMetadata>),
}

impl ColorHint {
    /// Hint from a media manifest entry's `color` field. Only `Some` is
    /// authoritative: manifests written before color was recorded also say
    /// `None`, so a missing value is still probed.
    pub fn from_manifest(color: Option<&MediaColorMetadata>) -> Self {
        match color {
            Some(color) => ColorHint::Known(Some(color.clone())),
            None => ColorHint::Unknown,
        }
    }
}

/// Identity under which probed signalling is cached. Any rewrite or
/// replacement changes the stamp (size/mtime, plus device/inode/ctime on
/// Unix), so a changed file is probed again.
#[derive(Clone, Debug, PartialEq, Eq)]
enum SourceIdentity {
    /// A pathname and the stamp of the file it named when probed.
    Path(PathBuf, FileStamp),
    /// A retained handle: its stamp plus the native file id (device/inode or
    /// volume/file index), which identifies the file without a pathname.
    Handle(FileStamp, String),
}

/// Bounded most-recently-used cache of probed signalling.
struct ColorCache {
    entries: VecDeque<(SourceIdentity, Option<MediaColorMetadata>)>,
}

impl ColorCache {
    const fn new() -> Self {
        ColorCache {
            entries: VecDeque::new(),
        }
    }

    fn get(&mut self, identity: &SourceIdentity) -> Option<Option<MediaColorMetadata>> {
        let index = self.entries.iter().position(|(key, _)| key == identity)?;
        // Most recently used entries live at the back.
        let entry = self.entries.remove(index)?;
        let color = entry.1.clone();
        self.entries.push_back(entry);
        Some(color)
    }

    fn insert(&mut self, identity: SourceIdentity, color: Option<MediaColorMetadata>) {
        if let Some(index) = self.entries.iter().position(|(key, _)| *key == identity) {
            self.entries.remove(index);
        }
        while self.entries.len() >= COLOR_CACHE_CAPACITY {
            self.entries.pop_front();
        }
        self.entries.push_back((identity, color));
    }
}

static COLOR_CACHE: Mutex<ColorCache> = Mutex::new(ColorCache::new());

fn with_cache<T>(action: impl FnOnce(&mut ColorCache) -> T) -> T {
    let mut cache = COLOR_CACHE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    action(&mut cache)
}

fn probe_color(
    identity: SourceIdentity,
    target: ProbeTarget<'_>,
    cancel: &MediaCancelToken,
) -> Result<Option<MediaColorMetadata>> {
    if let Some(color) = with_cache(|cache| cache.get(&identity)) {
        return Ok(color);
    }
    match crate::probe::probe_for_decode(
        target,
        cancel,
        COLOR_PROBE_TIMEOUT,
        COLOR_PROBE_ADMISSION_WAIT,
    ) {
        Ok(Some(probe)) => {
            with_cache(|cache| cache.insert(identity, probe.color.clone()));
            Ok(probe.color)
        }
        // ffprobe read the input and could not open it as media. The decoder
        // opens it the same way, so it fails too and reports the real reason;
        // no frame (and so no untone-mapped frame) can come out of it.
        Ok(None) => Ok(None),
        Err(MediaError::Cancelled) => Err(MediaError::Cancelled),
        Err(error) => Err(MediaError::Ffmpeg(format!(
            "source color probe failed: {error}"
        ))),
    }
}

/// Color signalling for decoding `path`.
///
/// Only ordinary files are probed. FIFOs and device inputs are valid FFmpeg
/// sources too; opening them once for ffprobe would consume or block the
/// stream before the actual cancellable decoder child is spawned, so they
/// decode without HDR conversion unless the caller supplies a hint. A path
/// that cannot be stat'ed is left to the decoder, which reports the failure.
pub(crate) fn resolve_path_color(
    path: &Path,
    hint: &ColorHint,
    cancel: &MediaCancelToken,
) -> Result<Option<MediaColorMetadata>> {
    if let ColorHint::Known(color) = hint {
        return Ok(color.clone());
    }
    if cancel.is_cancelled() {
        return Err(MediaError::Cancelled);
    }
    let Ok(metadata) = std::fs::metadata(path) else {
        return Ok(None);
    };
    if !metadata.is_file() {
        return Ok(None);
    }
    let identity = SourceIdentity::Path(path.to_path_buf(), FileStamp::of(&metadata));
    probe_color(identity, ProbeTarget::Path(path), cancel)
}

/// Color signalling for decoding an already-open regular file. The handle is
/// probed through `fd:`; no pathname is resolved.
pub(crate) fn resolve_file_color(
    file: &File,
    hint: &ColorHint,
    cancel: &MediaCancelToken,
) -> Result<Option<MediaColorMetadata>> {
    if let ColorHint::Known(color) = hint {
        return Ok(color.clone());
    }
    if cancel.is_cancelled() {
        return Err(MediaError::Cancelled);
    }
    let stamp = FileStamp::of(&file.metadata()?);
    let file_id = crate::proxy::source_file_stamp_file(file)?.file_id;
    probe_color(
        SourceIdentity::Handle(stamp, file_id),
        ProbeTarget::File(file),
        cancel,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hdr() -> MediaColorMetadata {
        MediaColorMetadata {
            primaries: Some("bt2020".into()),
            transfer: Some("smpte2084".into()),
            matrix: Some("bt2020nc".into()),
            range: Some("tv".into()),
        }
    }

    #[test]
    fn manifest_hint_trusts_recorded_color_but_probes_a_missing_one() {
        assert_eq!(
            ColorHint::from_manifest(Some(&hdr())),
            ColorHint::Known(Some(hdr()))
        );
        assert_eq!(ColorHint::from_manifest(None), ColorHint::Unknown);
    }

    #[test]
    fn known_hint_never_probes() {
        let before = crate::ff::test_seams::probe_requests();
        let color = resolve_path_color(
            Path::new("/definitely/missing/source.mp4"),
            &ColorHint::Known(Some(hdr())),
            &MediaCancelToken::new(),
        )
        .unwrap();
        assert_eq!(color, Some(hdr()));
        assert_eq!(crate::ff::test_seams::probe_requests(), before);
    }

    #[test]
    fn cache_is_bounded_and_keeps_recently_used_entries() {
        let stamp = FileStamp::of(&std::fs::metadata(std::env::temp_dir()).unwrap());
        let key = |index: usize| {
            SourceIdentity::Path(PathBuf::from(format!("/cache-test/{index}")), stamp)
        };
        let mut cache = ColorCache::new();
        cache.insert(key(0), Some(hdr()));
        for index in 1..COLOR_CACHE_CAPACITY {
            cache.insert(key(index), None);
        }
        // Touch the oldest entry so the next insertion evicts entry 1 instead.
        assert_eq!(cache.get(&key(0)), Some(Some(hdr())));
        cache.insert(key(COLOR_CACHE_CAPACITY), None);
        assert_eq!(cache.entries.len(), COLOR_CACHE_CAPACITY);
        assert_eq!(cache.get(&key(0)), Some(Some(hdr())));
        assert_eq!(cache.get(&key(1)), None);
        assert_eq!(cache.get(&key(COLOR_CACHE_CAPACITY)), Some(None));
    }
}
