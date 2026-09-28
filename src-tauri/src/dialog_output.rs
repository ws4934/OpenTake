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

    /// Consume the grant for exactly `path` and `purpose`, if one is live.
    fn take(&self, path: &Path, purpose: SavePurpose) -> bool {
        let mut grants = self.grants.lock().unwrap_or_else(PoisonError::into_inner);
        grants.retain(|grant| grant.issued.elapsed() < SAVE_GRANT_TTL);
        let position = grants
            .iter()
            .position(|grant| grant.purpose == purpose && same_path(&grant.path, path));
        position.map(|index| grants.remove(index)).is_some()
    }
}

#[cfg(windows)]
fn same_path(left: &Path, right: &Path) -> bool {
    let right: PathBuf = right.components().collect();
    left.to_string_lossy()
        .eq_ignore_ascii_case(&right.to_string_lossy())
}

#[cfg(not(windows))]
fn same_path(left: &Path, right: &Path) -> bool {
    let right: PathBuf = right.components().collect();
    left == right
}

/// One filter of the native save dialog.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SaveDialogFilter {
    name: String,
    extensions: Vec<String>,
}

/// `pick_save_path`: show the native save dialog off the UI thread and record
/// the chosen path as a single-use grant for `purpose`. Returns the path
/// exactly as the dialog returned it, or `None` when the user cancelled.
#[tauri::command]
pub async fn pick_save_path(
    window: tauri::Window,
    purpose: SavePurpose,
    title: Option<String>,
    default_path: Option<String>,
    filters: Option<Vec<SaveDialogFilter>>,
) -> Result<Option<String>, String> {
    use tauri::Manager;
    use tauri_plugin_dialog::DialogExt;
    let app = window.app_handle().clone();
    let parent = window.clone();
    let chosen = tauri::async_runtime::spawn_blocking(move || {
        let mut builder = app.dialog().file().set_parent(&parent);
        if let Some(title) = title {
            builder = builder.set_title(title);
        }
        if let Some(default_path) = default_path.filter(|path| !path.is_empty()) {
            // Mirrors tauri-plugin-dialog's `set_default_path` for desktop.
            let default_path: PathBuf = Path::new(&default_path).components().collect();
            if default_path.is_file() || !default_path.exists() {
                if let (Some(parent), Some(file_name)) =
                    (default_path.parent(), default_path.file_name())
                {
                    if parent.components().count() > 0 {
                        builder = builder.set_directory(parent);
                    }
                    builder = builder.set_file_name(file_name.to_string_lossy());
                } else {
                    builder = builder.set_directory(&default_path);
                }
            } else {
                builder = builder.set_directory(&default_path);
            }
        }
        for filter in filters.unwrap_or_default() {
            let extensions: Vec<&str> = filter.extensions.iter().map(String::as_str).collect();
            builder = builder.add_filter(filter.name, &extensions);
        }
        builder.blocking_save_file()
    })
    .await
    .map_err(|error| format!("save dialog failed: {error}"))?;
    let Some(chosen) = chosen else {
        return Ok(None);
    };
    let path = chosen
        .simplified()
        .into_path()
        .map_err(|error| format!("save dialog returned an unusable path: {error}"))?;
    window.state::<SaveGrants>().issue(&path, purpose);
    Ok(Some(path.to_string_lossy().into_owned()))
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
    extensions: &[&str],
) -> Result<DialogOutput, String> {
    let default_extension = extensions
        .first()
        .ok_or_else(|| "no output extension configured".to_string())?;
    if raw_path.is_empty() || raw_path.contains('\0') {
        return Err("output path is empty or contains a null byte".to_string());
    }
    let raw = Path::new(raw_path);
    if !raw.is_absolute() {
        return Err("output path must be absolute".to_string());
    }
    if raw
        .components()
        .any(|component| matches!(component, Component::ParentDir | Component::CurDir))
        || raw_path.ends_with(['/', std::path::MAIN_SEPARATOR])
    {
        return Err("output path must name a file without relative segments".to_string());
    }
    let Some(file_name) = raw.file_name() else {
        return Err("output path must name a file".to_string());
    };
    if !grants.take(raw, purpose) {
        return Err(UNAPPROVED_OUTPUT.to_string());
    }
    let normalized: PathBuf = raw.components().collect();
    // Exact-case match: consumers such as the project bundle detection and
    // the audio codec table compare extensions case-sensitively.
    let has_extension = raw
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extensions.contains(&extension));
    if has_extension {
        return Ok(DialogOutput {
            path: normalized,
            appended: false,
        });
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
    let mut name = file_name.to_os_string();
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
        |name| name.to_string_lossy().into_owned(),
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
    let path = output.path.as_path();
    ensure_replaceable_regular_file(path)?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .ok_or_else(|| "output path has no parent directory".to_string())?;
    let file_name = path
        .file_name()
        .ok_or_else(|| "output path must name a file".to_string())?;
    let (staging, mut file) = create_staging_file(parent, file_name)?;
    let written = file
        .write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|error| format!("write output: {error}"));
    drop(file);
    let published = written.and_then(|()| {
        if output.appended {
            publish_new(&staging, path)
        } else {
            ensure_replaceable_regular_file(path).and_then(|()| {
                std::fs::rename(&staging, path).map_err(|error| format!("replace output: {error}"))
            })
        }
    });
    if let Err(error) = published {
        remove_staging(&staging);
        return Err(error);
    }
    // The file is already in place; a failed directory sync only weakens
    // durability across a power loss, so report it without failing the save.
    if let Err(error) = sync_parent_directory(parent) {
        eprintln!("[dialog-output] could not sync output directory: {error}");
    }
    Ok(())
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
    file_name: &std::ffi::OsStr,
) -> Result<(PathBuf, std::fs::File), String> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut last_error = None;
    for _ in 0..16 {
        let mut name = std::ffi::OsString::from(".");
        name.push(file_name);
        name.push(format!(
            ".{}-{}.tmp",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
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

    /// A grant store holding one save-dialog result.
    pub(crate) fn granted(path: &Path, purpose: SavePurpose) -> SaveGrants {
        let grants = SaveGrants::default();
        grants.issue(path, purpose);
        grants
    }

    fn authorize(
        grants: &SaveGrants,
        path: &Path,
        extensions: &[&str],
    ) -> Result<DialogOutput, String> {
        authorize_dialog_output(
            grants,
            &path.to_string_lossy(),
            SavePurpose::Interchange,
            extensions,
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
                &["xml"]
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
