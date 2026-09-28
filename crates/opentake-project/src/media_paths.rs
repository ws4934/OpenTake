//! The one portable-path rule every `media.json` reader and writer applies to
//! `.project` media and proxy paths.
//!
//! Writers refuse to persist a path the rule rejects, so a save can never
//! produce a manifest its own reader refuses. Readers no longer fail the whole
//! project on such a path either: the entry is kept but made offline, its
//! original string is retained verbatim for the next save, and the unsafe path
//! itself is never exposed to anything that could resolve it.

use std::borrow::Cow;

use opentake_domain::{
    is_safe_project_asset_relative_path, MediaManifest, MediaProxy, MediaSource,
};

use crate::error::{ProjectError, Result};
use crate::layout;

/// The source an entry with an unsafe project path carries in memory. An
/// empty path is never a file on any platform, so every resolver treats the
/// asset as offline (and the media panel offers relinking) while nothing can
/// join the original string onto a base directory.
pub(crate) fn offline_source() -> MediaSource {
    MediaSource::External {
        absolute_path: String::new(),
    }
}

/// The unsafe paths one manifest entry carried on disk, withheld from the
/// live manifest when the project was opened.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct QuarantinedMediaPaths {
    asset_id: String,
    /// The entry's live source right after opening. While the entry still
    /// carries it, the user has neither relinked nor replaced the source.
    live_source: MediaSource,
    /// The original unsafe `.project` relative path, if the source was unsafe.
    source: Option<String>,
    /// The original proxy, if its path was unsafe.
    proxy: Option<MediaProxy>,
}

/// Withhold every unsafe path of a freshly decoded manifest. Returns what was
/// withheld plus one file-qualified warning per affected entry.
pub(crate) fn quarantine_unsafe_media_paths(
    manifest: &mut MediaManifest,
) -> (Vec<QuarantinedMediaPaths>, Vec<String>) {
    let mut quarantined = Vec::new();
    let mut warnings = Vec::new();
    for entry in &mut manifest.entries {
        let source = match &entry.source {
            MediaSource::Project { relative_path }
                if !is_safe_project_asset_relative_path(relative_path) =>
            {
                Some(relative_path.clone())
            }
            _ => None,
        };
        let proxy = entry
            .proxy
            .take_if(|proxy| !is_safe_project_asset_relative_path(&proxy.relative_path));
        if source.is_none() && proxy.is_none() {
            continue;
        }
        if source.is_some() {
            entry.source = offline_source();
            warnings.push(format!(
                "{}:offline-media:{}",
                layout::MANIFEST_FILE,
                entry.id
            ));
        }
        if proxy.is_some() {
            warnings.push(format!(
                "{}:ignored-proxy:{}",
                layout::MANIFEST_FILE,
                entry.id
            ));
        }
        quarantined.push(QuarantinedMediaPaths {
            asset_id: entry.id.clone(),
            live_source: entry.source.clone(),
            source,
            proxy,
        });
    }
    (quarantined, warnings)
}

/// The manifest to persist for `live`: every path it carries must pass the
/// portable-path rule, then the withheld originals of entries the user has not
/// relinked are restored verbatim so saving never discards them.
pub(crate) fn manifest_for_write<'a>(
    live: &'a MediaManifest,
    quarantined: &[QuarantinedMediaPaths],
) -> Result<Cow<'a, MediaManifest>> {
    validate_manifest_paths(live)?;
    if quarantined.is_empty() {
        return Ok(Cow::Borrowed(live));
    }
    let mut manifest = live.clone();
    let mut consumed = vec![false; quarantined.len()];
    for entry in &mut manifest.entries {
        // An entry whose source changed since opening was relinked or
        // replaced: its withheld paths no longer describe it.
        let Some(index) = quarantined.iter().enumerate().position(|(index, record)| {
            !consumed[index] && record.asset_id == entry.id && record.live_source == entry.source
        }) else {
            continue;
        };
        consumed[index] = true;
        let record = &quarantined[index];
        if let Some(relative_path) = &record.source {
            entry.source = MediaSource::Project {
                relative_path: relative_path.clone(),
            };
        }
        if entry.proxy.is_none() {
            entry.proxy.clone_from(&record.proxy);
        }
    }
    Ok(Cow::Owned(manifest))
}

/// Refuse a manifest whose `.project` media or proxy paths are not portable
/// bundle-relative paths.
pub(crate) fn validate_manifest_paths(manifest: &MediaManifest) -> Result<()> {
    for entry in &manifest.entries {
        if let MediaSource::Project { relative_path } = &entry.source {
            if !is_safe_project_asset_relative_path(relative_path) {
                return Err(ProjectError::InvalidMediaManifest {
                    file: layout::MANIFEST_FILE,
                    reason: format!(
                        "project source for asset '{}' is not a safe bundle-relative path",
                        entry.id
                    ),
                });
            }
        }
        if let Some(proxy) = &entry.proxy {
            if !is_safe_project_asset_relative_path(&proxy.relative_path) {
                return Err(ProjectError::InvalidMediaManifest {
                    file: layout::MANIFEST_FILE,
                    reason: format!(
                        "proxy for asset '{}' is not a safe bundle-relative path",
                        entry.id
                    ),
                });
            }
        }
    }
    Ok(())
}
