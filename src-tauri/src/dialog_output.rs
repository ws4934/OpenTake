//! Save-dialog authorization and durable writes for user-chosen outputs.
//!
//! Commands that write user-chosen output (timeline interchange, subtitles,
//! extracted audio, rendered video, project bundles) accept only a path that
//! [`pick_save_path`] returned from a native save dialog for the same
//! [`SavePurpose`]. The backend records that result as an in-memory,
//! single-use [`SaveGrants`] entry; the asset-protocol scope is not consulted,
//! because it also holds read grants from open dialogs, folder imports, proxies
//! and thumbnails that must never authorize a write (#95).
//!
//! The extension is appended here rather than in the WebView: GTK save dialogs
//! never append one and the interchange dialogs pass no filters (see
//! `TitleBar.tsx`), so a path the frontend had extended would no longer match
//! the dialog result (#102).

use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use opentake_domain::NativePath;

#[cfg(windows)]
mod windows;
use serde::Deserialize;

pub(crate) const UNAPPROVED_OUTPUT: &str =
    "output path has not been approved by a native save dialog";

/// What a save-dialog result may be written as. A grant issued for one purpose
/// never authorizes another command.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SavePurpose {
    Project,
    Interchange,
    Subtitles,
    Video,
    ExtractAudio,
}

/// How long an unused save-dialog result stays valid.
const SAVE_GRANT_TTL: Duration = Duration::from_secs(30 * 60);
/// Unused grants kept at once; older ones are dropped first.
const MAX_SAVE_GRANTS: usize = 16;

struct SaveGrant {
    path: PathBuf,
    purpose: SavePurpose,
    issued: Instant,
}

/// Single-use save-dialog results, held only in memory.
#[derive(Default)]
pub struct SaveGrants {
    grants: Mutex<Vec<SaveGrant>>,
}

impl SaveGrants {
    pub(crate) fn issue(&self, path: &Path, purpose: SavePurpose) {
        let mut grants = self.grants.lock().unwrap_or_else(PoisonError::into_inner);
        grants.retain(|grant| grant.issued.elapsed() < SAVE_GRANT_TTL);
        if grants.len() >= MAX_SAVE_GRANTS {
            grants.remove(0);
        }
        grants.push(SaveGrant {
            path: path.components().collect(),
            purpose,
            issued: Instant::now(),
        });
    }

    /// Consume the grant for `path` and `purpose`, if one is live, and return
    /// the path exactly as the dialog returned it. Writers use that spelling,
    /// not the caller's, so a case-insensitive match can never redirect a write
    /// in a case-sensitive directory.
    fn take(&self, path: &Path, purpose: SavePurpose) -> Option<PathBuf> {
        let mut grants = self.grants.lock().unwrap_or_else(PoisonError::into_inner);
        grants.retain(|grant| grant.issued.elapsed() < SAVE_GRANT_TTL);
        let position = grants
            .iter()
            .position(|grant| grant.purpose == purpose && same_path(&grant.path, path))?;
        Some(grants.remove(position).path)
    }
}

fn same_path(left: &Path, right: &Path) -> bool {
    opentake_domain::native_path::identity_key(left)
        == opentake_domain::native_path::identity_key(right)
}

/// One filter of the native save dialog.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SaveDialogFilter {
    name: String,
    extensions: Vec<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenDialogOptions {
    title: Option<String>,
    default_path: Option<NativePath>,
    #[serde(default)]
    filters: Vec<SaveDialogFilter>,
    #[serde(default)]
    multiple: bool,
    #[serde(default)]
    directory: bool,
    #[serde(default)]
    recursive: bool,
    can_create_directories: Option<bool>,
}

#[derive(serde::Serialize)]
#[serde(untagged)]
pub enum OpenSelection {
    Multiple(Option<Vec<NativePath>>),
    Single(Option<NativePath>),
}

/// Intercept native results before plugin IPC serialization and scope grants.
/// The plugin's Rust picker preserves `PathBuf`; its JS command does not.
#[tauri::command]
pub async fn pick_open_paths(
    window: tauri::Window,
    options: OpenDialogOptions,
) -> Result<OpenSelection, String> {
    use tauri::Manager;
    let app = window.app_handle().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let directory = options.directory;
        let recursive = options.recursive;
        let multiple = options.multiple;
        let paths = pick_open_on_platform(&window, options)?;
        if let Some(paths) = &paths {
            let mut approved = paths.clone();
            for path in paths {
                let final_path = std::fs::canonicalize(path)
                    .map_err(|error| format!("selected path is unavailable: {error}"))?;
                if final_path != *path {
                    approved.push(final_path);
                }
            }
            crate::native_read_scope::allow_selections(&app, &approved, directory, recursive)?;
        }
        let paths = paths.map(|paths| paths.into_iter().map(NativePath::from).collect::<Vec<_>>());
        Ok(if multiple {
            OpenSelection::Multiple(paths)
        } else {
            OpenSelection::Single(paths.and_then(|mut paths| paths.pop()))
        })
    })
    .await
    .map_err(|error| format!("open dialog failed: {error}"))?
}

#[cfg(not(windows))]
fn pick_open_on_platform(
    window: &tauri::Window,
    options: OpenDialogOptions,
) -> Result<Option<Vec<PathBuf>>, String> {
    use tauri::Manager;
    use tauri_plugin_dialog::DialogExt;
    let mut builder = window.app_handle().dialog().file().set_parent(window);
    if let Some(title) = options.title {
        builder = builder.set_title(title);
    }
    if let Some(path) = options.default_path {
        let path = path.into_path_buf().map_err(str::to_owned)?;
        if path.is_dir() {
            builder = builder.set_directory(&path);
        } else {
            if let Some(parent) = path.parent() {
                builder = builder.set_directory(parent);
            }
            if let Some(name) = path.file_name().and_then(|name| name.to_str()) {
                builder = builder.set_file_name(name);
            }
        }
    }
    if let Some(can) = options.can_create_directories {
        builder = builder.set_can_create_directories(can);
    }
    for filter in options.filters {
        let extensions: Vec<_> = filter.extensions.iter().map(String::as_str).collect();
        builder = builder.add_filter(filter.name, &extensions);
    }
    let chosen = match (options.directory, options.multiple) {
        (true, true) => builder.blocking_pick_folders(),
        (true, false) => builder.blocking_pick_folder().map(|path| vec![path]),
        (false, true) => builder.blocking_pick_files(),
        (false, false) => builder.blocking_pick_file().map(|path| vec![path]),
    };
    chosen
        .map(|paths| {
            paths
                .into_iter()
                .map(|path| {
                    path.simplified()
                        .into_path()
                        .map_err(|error| format!("open dialog returned an unusable path: {error}"))
                })
                .collect()
        })
        .transpose()
}

#[cfg(windows)]
fn pick_open_on_platform(
    window: &tauri::Window,
    options: OpenDialogOptions,
) -> Result<Option<Vec<PathBuf>>, String> {
    // Windows Shell controls directory creation; this option is macOS-only.
    let _ = options.can_create_directories;
    let default_path = options
        .default_path
        .map(NativePath::into_path_buf)
        .transpose()
        .map_err(str::to_owned)?;
    windows::pick(
        window,
        windows::Kind::Open {
            directory: options.directory,
            multiple: options.multiple,
        },
        options.title,
        default_path,
        options.filters,
    )
}

/// `pick_save_path`: show the native save dialog off the UI thread and record
/// the chosen path as a single-use grant for `purpose`. Returns the path
/// exactly as the dialog returned it, or `None` when the user cancelled.
#[tauri::command]
pub async fn pick_save_path(
    window: tauri::Window,
    purpose: SavePurpose,
    title: Option<String>,
    default_path: Option<NativePath>,
    filters: Option<Vec<SaveDialogFilter>>,
) -> Result<Option<String>, String> {
    use tauri::Manager;
    let default_path = default_path
        .map(NativePath::into_path_buf)
        .transpose()
        .map_err(str::to_owned)?;
    let parent = window.clone();
    let chosen = tauri::async_runtime::spawn_blocking(move || {
        pick_save_on_platform(&parent, title, default_path, filters.unwrap_or_default())
    })
    .await
    .map_err(|error| format!("save dialog failed: {error}"))??;
    let Some(path) = chosen else {
        return Ok(None);
    };
    window.state::<SaveGrants>().issue(&path, purpose);
    Ok(Some(NativePath::from(path).to_wire()))
}

#[cfg(windows)]
fn pick_save_on_platform(
    window: &tauri::Window,
    title: Option<String>,
    default_path: Option<PathBuf>,
    filters: Vec<SaveDialogFilter>,
) -> Result<Option<PathBuf>, String> {
    Ok(
        windows::pick(window, windows::Kind::Save, title, default_path, filters)?
            .and_then(|mut paths| paths.pop()),
    )
}

#[cfg(not(windows))]
fn pick_save_on_platform(
    window: &tauri::Window,
    title: Option<String>,
    default_path: Option<PathBuf>,
    filters: Vec<SaveDialogFilter>,
) -> Result<Option<PathBuf>, String> {
    use tauri::Manager;
    use tauri_plugin_dialog::DialogExt;
    let mut builder = window.app_handle().dialog().file().set_parent(window);
    if let Some(title) = title {
        builder = builder.set_title(title);
    }
    if let Some(path) = default_path.filter(|path| !path.as_os_str().is_empty()) {
        let path: PathBuf = path.components().collect();
        if path.is_file() || !path.exists() {
            if let (Some(parent), Some(name)) = (path.parent(), path.file_name()) {
                if parent.components().count() > 0 {
                    builder = builder.set_directory(parent);
                }
                if let Some(name) = name.to_str() {
                    builder = builder.set_file_name(name);
                }
            } else {
                builder = builder.set_directory(&path);
            }
        } else {
            builder = builder.set_directory(&path);
        }
    }
    for filter in filters {
        let extensions: Vec<_> = filter.extensions.iter().map(String::as_str).collect();
        builder = builder.add_filter(filter.name, &extensions);
    }
    builder
        .blocking_save_file()
        .map(|path| {
            path.simplified()
                .into_path()
                .map_err(|error| format!("save dialog returned an unusable path: {error}"))
        })
        .transpose()
}

/// What a dialog result without an allowed extension turns into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ForeignExtension {
    /// Append the default extension (`cut.txt` becomes `cut.txt.xml`).
    Append,
    /// Refuse a result that already carries another extension, so the
    /// consumer's own "unsupported extension" error is what the user sees.
    Reject,
}

/// How a purpose's consumer recognizes its extensions.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OutputRule<'a> {
    /// Allowed extensions; the first one is appended when none is present.
    pub extensions: &'a [&'a str],
    /// Whether the consumer compares extensions ignoring ASCII case.
    pub ignore_case: bool,
    pub foreign: ForeignExtension,
}

/// A write target resolved from a save-dialog result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DialogOutput {
    pub path: PathBuf,
    /// The backend appended the extension, so the user never confirmed
    /// replacing `path`: it must be created, never overwritten.
    pub appended: bool,
}

/// Resolve the path a command may write for a save-dialog result `raw_path`.
///
/// `raw_path` must be absolute, free of `.`/`..` segments and carry a live
/// grant from [`pick_save_path`] for `purpose`; the grant is consumed. When its
/// extension is not exactly one of `extensions`, the first one is appended to
/// the file name, so the result stays in the directory the user picked. That
/// only happens for a new name: `raw_path` itself must not exist (a save result
/// without its extension is never an existing directory or file) and neither
/// may the appended path, because the dialog asked about replacing `raw_path`.
pub(crate) fn authorize_dialog_output(
    grants: &SaveGrants,
    raw_path: &str,
    purpose: SavePurpose,
    rule: OutputRule<'_>,
) -> Result<DialogOutput, String> {
    let default_extension = rule
        .extensions
        .first()
        .ok_or_else(|| "no output extension configured".to_string())?;
    if raw_path.is_empty() || raw_path.contains('\0') {
        return Err("output path is empty or contains a null byte".to_string());
    }
    let native = NativePath::from_wire(raw_path).map_err(str::to_owned)?;
    let raw = native.local_path().map_err(str::to_owned)?;
    if !raw.is_absolute() {
        return Err("output path must be absolute".to_string());
    }
    if raw
        .components()
        .any(|component| matches!(component, Component::ParentDir | Component::CurDir))
        || raw
            .as_os_str()
            .as_encoded_bytes()
            .last()
            .is_some_and(|byte| *byte == b'/' || (cfg!(windows) && *byte == b'\\'))
    {
        return Err("output path must name a file without relative segments".to_string());
    }
    if raw.file_name().is_none() {
        return Err("output path must name a file".to_string());
    }
    let Some(normalized) = grants.take(raw, purpose) else {
        return Err(UNAPPROVED_OUTPUT.to_string());
    };
    // Match extensions the way the purpose's consumer does: the project
    // bundle detection and the audio codec table are case-sensitive, video
    // presets are not.
    let extension = normalized
        .extension()
        .map(|extension| extension.to_string_lossy().into_owned());
    let has_extension = extension.as_deref().is_some_and(|extension| {
        rule.extensions.iter().any(|allowed| {
            if rule.ignore_case {
                allowed.eq_ignore_ascii_case(extension)
            } else {
                *allowed == extension
            }
        })
    });
    if has_extension {
        return Ok(DialogOutput {
            path: normalized,
            appended: false,
        });
    }
    if let (Some(extension), ForeignExtension::Reject) = (&extension, rule.foreign) {
        return Err(format!(
            "unsupported extension .{extension} (use {})",
            rule.extensions
                .iter()
                .map(|allowed| format!(".{allowed}"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    match std::fs::symlink_metadata(&normalized) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Ok(_) => {
            return Err(format!(
                "{} already exists and is not a .{default_extension} file",
                display_name(&normalized)
            ))
        }
        Err(error) => return Err(format!("cannot inspect output path: {error}")),
    }
    let mut name = normalized
        .file_name()
        .ok_or_else(|| "output path must name a file".to_string())?
        .to_os_string();
    name.push(".");
    name.push(default_extension);
    let resolved = normalized.with_file_name(name);
    match std::fs::symlink_metadata(&resolved) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(DialogOutput {
            path: resolved,
            appended: true,
        }),
        Ok(_) => Err(format!(
            "{} already exists; type the .{default_extension} extension in the save dialog to replace it",
            display_name(&resolved)
        )),
        Err(error) => Err(format!("cannot inspect output path: {error}")),
    }
}

/// The file name alone, so errors never echo the user's directory layout.
fn display_name(path: &Path) -> String {
    path.file_name().map_or_else(
        || "output".to_string(),
        |name| {
            name.to_str()
                .map(str::to_owned)
                .unwrap_or_else(|| format!("{name:?}"))
        },
    )
}

/// Reject an output target that exists but is not a plain file (a symlink, a
/// directory, a device or a Windows reparse point), so a write can never be
/// redirected elsewhere.
pub(crate) fn ensure_replaceable_regular_file(path: &Path) -> Result<(), String> {
    match std::fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("cannot inspect output path: {error}")),
        Ok(metadata) if metadata.file_type().is_file() && !is_reparse_point(&metadata) => Ok(()),
        Ok(_) => Err(format!(
            "{} exists and is not a regular file",
            display_name(path)
        )),
    }
}

#[cfg(windows)]
fn is_reparse_point(metadata: &std::fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;
    const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
    metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
}

#[cfg(not(windows))]
fn is_reparse_point(_metadata: &std::fs::Metadata) -> bool {
    false
}

/// Write `bytes` to `output` without following symlinks and without
/// truncating an existing file first: stage an exclusive temporary file in the
/// same directory, fsync it, then publish it. A confirmed target is replaced
/// by rename; an appended one is published without replacing anything, so a
/// file created there in the meantime is never overwritten.
pub(crate) fn write_file_atomically(output: &DialogOutput, bytes: &[u8]) -> Result<(), String> {
    let (staging, mut file) = create_staging(output)?;
    let written = file
        .write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|error| format!("write output: {error}"));
    drop(file);
    if let Err(error) = written {
        remove_staging(&staging);
        return Err(error);
    }
    publish_staged(&staging, output)
}

/// Create an exclusive, empty staging file beside `output` that keeps its
/// extension (tools such as ffmpeg pick the container from it). The caller
/// fills it and hands it to [`publish_staged`], or removes it on failure.
pub(crate) fn create_staging(output: &DialogOutput) -> Result<(PathBuf, std::fs::File), String> {
    let path = output.path.as_path();
    ensure_replaceable_regular_file(path)?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .ok_or_else(|| "output path has no parent directory".to_string())?;
    let stem = path
        .file_stem()
        .ok_or_else(|| "output path must name a file".to_string())?;
    create_staging_file(parent, stem, path.extension())
}

/// Publish a filled staging file at `output`: a confirmed target is replaced
/// by rename, an appended one only if nothing exists there. The staging file
/// is removed when publishing fails.
pub(crate) fn publish_staged(staging: &Path, output: &DialogOutput) -> Result<(), String> {
    let path = output.path.as_path();
    let published = if output.appended {
        publish_new(staging, path)
    } else {
        ensure_replaceable_regular_file(path).and_then(|()| {
            std::fs::rename(staging, path).map_err(|error| format!("replace output: {error}"))
        })
    };
    if let Err(error) = published {
        remove_staging(staging);
        return Err(error);
    }
    // The file is already in place; a failed directory sync only weakens
    // durability across a power loss, so report it without failing the save.
    if let Some(parent) = path.parent() {
        if let Err(error) = sync_parent_directory(parent) {
            eprintln!("[dialog-output] could not sync output directory: {error}");
        }
    }
    Ok(())
}

pub(crate) fn remove_staging_file(staging: &Path) {
    remove_staging(staging);
}

/// Publish `staging` at `path` only if nothing exists there. A hard link fails
/// atomically on an existing name; file systems without hard links fall back to
/// a checked rename.
fn publish_new(staging: &Path, path: &Path) -> Result<(), String> {
    match std::fs::hard_link(staging, path) {
        Ok(()) => {
            remove_staging(staging);
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Err(format!(
            "{} already exists; type the extension in the save dialog to replace it",
            display_name(path)
        )),
        Err(_) => match std::fs::symlink_metadata(path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                std::fs::rename(staging, path).map_err(|error| format!("create output: {error}"))
            }
            Ok(_) => Err(format!("{} already exists", display_name(path))),
            Err(error) => Err(format!("cannot inspect output path: {error}")),
        },
    }
}

fn remove_staging(staging: &Path) {
    if let Err(error) = std::fs::remove_file(staging) {
        if error.kind() != std::io::ErrorKind::NotFound {
            eprintln!("[dialog-output] could not remove a staging file: {error}");
        }
    }
}

fn create_staging_file(
    parent: &Path,
    stem: &std::ffi::OsStr,
    extension: Option<&std::ffi::OsStr>,
) -> Result<(PathBuf, std::fs::File), String> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut last_error = None;
    for _ in 0..16 {
        let mut name = std::ffi::OsString::from(".");
        name.push(stem);
        name.push(format!(
            ".{}-{}.tmp",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        if let Some(extension) = extension {
            name.push(".");
            name.push(extension);
        }
        let staging = parent.join(name);
        // `create_new` is O_CREAT|O_EXCL: it never follows or reuses an
        // existing entry, symlinks included.
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&staging)
        {
            Ok(file) => return Ok((staging, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                last_error = Some(error);
            }
            Err(error) => return Err(format!("create output: {error}")),
        }
    }
    Err(format!(
        "create output: {}",
        last_error.map_or_else(
            || "no staging name available".to_string(),
            |e| e.to_string()
        )
    ))
}

#[cfg(unix)]
fn sync_parent_directory(parent: &Path) -> std::io::Result<()> {
    std::fs::File::open(parent)?.sync_all()
}

#[cfg(not(unix))]
fn sync_parent_directory(_parent: &Path) -> std::io::Result<()> {
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn native_save_grants_never_authorize_a_replacement_name() {
        use std::os::unix::ffi::OsStrExt;
        let directory = tempfile::tempdir().unwrap();
        let native = directory
            .path()
            .join(std::ffi::OsStr::from_bytes(b"output-\xff.mp4"));
        let shadow = PathBuf::from(native.to_string_lossy().as_ref());
        let grants = granted(&native, SavePurpose::Video);
        assert_eq!(
            authorize_dialog_output(
                &grants,
                &NativePath::new(&shadow).to_wire(),
                SavePurpose::Video,
                rule(&["mp4"])
            )
            .unwrap_err(),
            UNAPPROVED_OUTPUT
        );
        let output = authorize_dialog_output(
            &grants,
            &NativePath::new(&native).to_wire(),
            SavePurpose::Video,
            rule(&["mp4"]),
        )
        .unwrap();
        assert_eq!(output.path, native);
        assert!(
            authorize_dialog_output(
                &grants,
                &NativePath::new(&output.path).to_wire(),
                SavePurpose::Video,
                rule(&["mp4"])
            )
            .is_err(),
            "the native grant remains single-use"
        );
    }

    /// A grant store holding one save-dialog result.
    pub(crate) fn granted(path: &Path, purpose: SavePurpose) -> SaveGrants {
        let grants = SaveGrants::default();
        grants.issue(path, purpose);
        grants
    }

    fn rule(extensions: &'static [&'static str]) -> OutputRule<'static> {
        OutputRule {
            extensions,
            ignore_case: false,
            foreign: ForeignExtension::Append,
        }
    }

    fn authorize(
        grants: &SaveGrants,
        path: &Path,
        extensions: &'static [&'static str],
    ) -> Result<DialogOutput, String> {
        authorize_dialog_output(
            grants,
            &path.to_string_lossy(),
            SavePurpose::Interchange,
            rule(extensions),
        )
    }

    #[test]
    fn unapproved_path_is_rejected_without_touching_the_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let target = dir.path().join("existing.srt");
        std::fs::write(&target, b"keep").expect("seed target");

        let error =
            authorize(&SaveGrants::default(), &target, &["srt"]).expect_err("unapproved path");

        assert_eq!(error, UNAPPROVED_OUTPUT);
        assert_eq!(std::fs::read(&target).expect("read target"), b"keep");
    }

    #[test]
    fn a_grant_is_single_use_and_bound_to_its_purpose() {
        let dir = tempfile::tempdir().expect("tempdir");
        let raw = dir.path().join("cut.xml");
        let grants = granted(&raw, SavePurpose::Interchange);

        assert_eq!(
            authorize_dialog_output(
                &grants,
                &raw.to_string_lossy(),
                SavePurpose::Video,
                rule(&["xml"])
            ),
            Err(UNAPPROVED_OUTPUT.to_string())
        );
        assert!(authorize(&grants, &raw, &["xml"]).is_ok());
        assert_eq!(
            authorize(&grants, &raw, &["xml"]),
            Err(UNAPPROVED_OUTPUT.to_string())
        );
    }

    #[test]
    fn approved_path_with_extension_is_returned_unchanged() {
        let dir = tempfile::tempdir().expect("tempdir");
        let raw = dir.path().join("cut.xml");
        let grants = granted(&raw, SavePurpose::Interchange);

        assert_eq!(
            authorize(&grants, &raw, &["xml"]),
            Ok(DialogOutput {
                path: raw,
                appended: false
            })
        );
    }

    #[test]
    fn approved_path_without_extension_gets_it_in_the_same_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let raw = dir.path().join("cut");
        let grants = granted(&raw, SavePurpose::Interchange);

        let resolved = authorize(&grants, &raw, &["xml"]).expect("approved");

        assert_eq!(resolved.path, dir.path().join("cut.xml"));
        assert!(resolved.appended);
    }

    #[test]
    fn extensions_match_case_sensitively() {
        let dir = tempfile::tempdir().expect("tempdir");
        let raw = dir.path().join("Vlog.OPENTAKE");
        let grants = granted(&raw, SavePurpose::Interchange);

        let resolved = authorize(&grants, &raw, &["opentake"]).expect("approved");

        assert_eq!(resolved.path, dir.path().join("Vlog.OPENTAKE.opentake"));
    }

    #[test]
    fn case_insensitive_rules_accept_an_upper_case_extension() {
        let dir = tempfile::tempdir().expect("tempdir");
        let raw = dir.path().join("Film.MP4");
        let grants = granted(&raw, SavePurpose::Interchange);

        let resolved = authorize_dialog_output(
            &grants,
            &raw.to_string_lossy(),
            SavePurpose::Interchange,
            OutputRule {
                extensions: &["mp4"],
                ignore_case: true,
                foreign: ForeignExtension::Append,
            },
        )
        .expect("approved");

        assert_eq!(resolved.path, raw);
        assert!(!resolved.appended);
    }

    #[test]
    fn a_reject_rule_refuses_a_foreign_extension() {
        let dir = tempfile::tempdir().expect("tempdir");
        let raw = dir.path().join("voice.ogg");
        let grants = granted(&raw, SavePurpose::Interchange);

        let error = authorize_dialog_output(
            &grants,
            &raw.to_string_lossy(),
            SavePurpose::Interchange,
            OutputRule {
                extensions: &["m4a", "mp3"],
                ignore_case: false,
                foreign: ForeignExtension::Reject,
            },
        )
        .expect_err("foreign extension");

        assert!(error.contains("unsupported extension .ogg"), "{error}");
    }

    #[test]
    fn staged_output_keeps_the_extension_and_publishes_once() {
        let dir = tempfile::tempdir().expect("tempdir");
        let output = DialogOutput {
            path: dir.path().join("voice.m4a"),
            appended: true,
        };

        let (staging, file) = create_staging(&output).expect("staging");
        drop(file);
        assert_eq!(staging.extension().and_then(|e| e.to_str()), Some("m4a"));
        std::fs::write(&staging, b"audio").expect("fill staging");
        publish_staged(&staging, &output).expect("publish");

        assert_eq!(std::fs::read(&output.path).expect("read"), b"audio");
        assert!(!staging.exists());
    }

    #[test]
    fn an_existing_directory_never_gets_an_extension_appended() {
        let dir = tempfile::tempdir().expect("tempdir");
        let footage = dir.path().join("Footage");
        std::fs::create_dir(&footage).expect("imported folder");
        let grants = granted(&footage, SavePurpose::Interchange);

        let error = authorize(&grants, &footage, &["srt"]).expect_err("existing raw path");

        assert!(error.contains("already exists"), "{error}");
        assert!(!dir.path().join("Footage.srt").exists());
    }

    #[test]
    fn appended_path_that_already_exists_is_not_overwritten() {
        let dir = tempfile::tempdir().expect("tempdir");
        let raw = dir.path().join("cut");
        let existing = dir.path().join("cut.edl");
        std::fs::write(&existing, b"keep").expect("seed existing export");
        let grants = granted(&raw, SavePurpose::Interchange);

        let error = authorize(&grants, &raw, &["edl"]).expect_err("unconfirmed overwrite");

        assert!(error.contains("already exists"), "{error}");
        assert!(!error.contains(&dir.path().to_string_lossy().into_owned()));
        assert_eq!(std::fs::read(&existing).expect("read existing"), b"keep");
    }

    #[test]
    fn appended_publish_never_replaces_a_file_created_meanwhile() {
        let dir = tempfile::tempdir().expect("tempdir");
        let raw = dir.path().join("cut");
        let grants = granted(&raw, SavePurpose::Interchange);
        let output = authorize(&grants, &raw, &["otio"]).expect("approved");
        std::fs::write(&output.path, b"raced").expect("concurrent writer");

        assert!(write_file_atomically(&output, b"export").is_err());
        assert_eq!(std::fs::read(&output.path).expect("read"), b"raced");
        assert_eq!(std::fs::read_dir(dir.path()).expect("list").count(), 1);
    }

    #[test]
    fn frontend_extended_path_no_longer_matches_the_dialog_result() {
        let dir = tempfile::tempdir().expect("tempdir");
        let raw = dir.path().join("Vlog");
        let grants = granted(&raw, SavePurpose::Interchange);

        assert_eq!(
            authorize(&grants, &dir.path().join("Vlog.opentake"), &["opentake"]),
            Err(UNAPPROVED_OUTPUT.to_string())
        );
    }

    #[test]
    fn parent_directory_changes_are_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let nested = dir.path().join("nested");
        std::fs::create_dir(&nested).expect("nested dir");
        let raw = nested.join("cut");
        let grants = granted(&raw, SavePurpose::Interchange);

        for attempt in [
            nested.join("..").join("nested").join("cut"),
            dir.path().join("cut"),
        ] {
            assert!(
                authorize(&grants, &attempt, &["xml"]).is_err(),
                "{} must be rejected",
                attempt.display()
            );
        }
        assert!(authorize(&grants, Path::new("cut"), &["xml"]).is_err());
        assert!(authorize(&grants, Path::new(""), &["xml"]).is_err());
    }

    #[test]
    fn atomic_write_creates_and_replaces_regular_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let target = DialogOutput {
            path: dir.path().join("cut.otio"),
            appended: false,
        };

        write_file_atomically(&target, b"first").expect("create");
        write_file_atomically(&target, b"second").expect("replace");

        assert_eq!(std::fs::read(&target.path).expect("read"), b"second");
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .expect("list")
            .map(|entry| entry.expect("entry").file_name())
            .filter(|name| name != "cut.otio")
            .collect();
        assert!(leftovers.is_empty(), "staging files left: {leftovers:?}");
    }

    #[test]
    fn atomic_write_rejects_a_directory_target() {
        let dir = tempfile::tempdir().expect("tempdir");
        let target = DialogOutput {
            path: dir.path().join("cut.xml"),
            appended: false,
        };
        std::fs::create_dir(&target.path).expect("directory target");

        assert!(write_file_atomically(&target, b"x").is_err());
        assert!(target.path.is_dir());
    }

    #[test]
    fn missing_directory_fails_without_creating_output() {
        let dir = tempfile::tempdir().expect("tempdir");
        let target = DialogOutput {
            path: dir.path().join("gone").join("cut.xml"),
            appended: false,
        };

        assert!(write_file_atomically(&target, b"x").is_err());
        assert!(!target.path.exists());
    }

    #[cfg(unix)]
    #[test]
    fn atomic_write_refuses_a_symlink_and_leaves_its_target_alone() {
        let dir = tempfile::tempdir().expect("tempdir");
        let victim = dir.path().join("victim.rc");
        std::fs::write(&victim, b"keep").expect("seed victim");
        let link = dir.path().join("cut.srt");
        std::os::unix::fs::symlink(&victim, &link).expect("symlink");
        let target = DialogOutput {
            path: link.clone(),
            appended: false,
        };

        assert!(write_file_atomically(&target, b"overwrite").is_err());
        assert_eq!(std::fs::read(&victim).expect("read victim"), b"keep");
        assert!(std::fs::symlink_metadata(&link)
            .expect("link")
            .file_type()
            .is_symlink());
    }
}
