//! Authorization and durable writes for files the user picks in a native save
//! dialog.
//!
//! `tauri-plugin-dialog`'s `save` adds the exact path the user confirmed to the
//! asset-protocol scope. Commands that write user-chosen output (timeline
//! interchange, subtitles, extracted audio, rendered video, project bundles)
//! accept only such a path, exactly as the dialog returned it. The extension is
//! appended here rather than in the WebView: GTK save dialogs never append one
//! and the interchange dialogs pass no filters (see `TitleBar.tsx`), so a path
//! the frontend had extended would no longer match the dialog grant.

use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use tauri::scope::fs::Scope;

pub(crate) const UNAPPROVED_OUTPUT: &str =
    "output path has not been approved by a native file dialog";

/// Resolve the path a command may write for a dialog result `raw_path`.
///
/// `raw_path` must be absolute, free of `.`/`..` segments and hold an exact
/// file grant from a native dialog (recursive directory grants left by folder
/// imports and the static application roots do not count). When its extension
/// is not one of `extensions` (compared ASCII case-insensitively), the first
/// extension is appended to the file name, so the result always stays in the
/// directory the user picked. An appended path the user never confirmed must
/// not already exist: the dialog asked about overwriting `raw_path`, not it.
pub(crate) fn authorize_dialog_output(
    scope: &Scope,
    raw_path: &str,
    extensions: &[&str],
) -> Result<PathBuf, String> {
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
    if !crate::safe_asset_protocol::scope_has_exact_file_grant(scope, raw)
        || !crate::safe_asset_protocol::scope_allows_lexical_path(scope, raw)
    {
        return Err(UNAPPROVED_OUTPUT.to_string());
    }
    let has_extension = raw
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            extensions
                .iter()
                .any(|allowed| allowed.eq_ignore_ascii_case(extension))
        });
    let normalized: PathBuf = raw.components().collect();
    if has_extension {
        return Ok(normalized);
    }
    let mut name = file_name.to_os_string();
    name.push(".");
    name.push(default_extension);
    let resolved = normalized.with_file_name(name);
    match std::fs::symlink_metadata(&resolved) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(resolved),
        Ok(_) => Err(format!(
            "{} already exists; type the .{default_extension} extension in the save dialog to replace it",
            resolved.display()
        )),
        Err(error) => Err(format!("cannot inspect output path: {error}")),
    }
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
            path.display()
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

/// Write `bytes` to `path` without following symlinks and without truncating
/// an existing file first: stage an exclusive temporary file in the same
/// directory, fsync it, and rename it over `path` only once it is complete.
pub(crate) fn write_file_atomically(path: &Path, bytes: &[u8]) -> Result<(), String> {
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
    let published = written
        .and_then(|()| ensure_replaceable_regular_file(path))
        .and_then(|()| {
            std::fs::rename(&staging, path).map_err(|error| format!("replace output: {error}"))
        });
    if let Err(error) = published {
        if let Err(cleanup) = std::fs::remove_file(&staging) {
            eprintln!(
                "[dialog-output] could not remove staging file {}: {cleanup}",
                staging.display()
            );
        }
        return Err(error);
    }
    sync_parent_directory(parent).map_err(|error| format!("sync output directory: {error}"))
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
mod tests {
    use super::*;
    use tauri::Manager;

    fn scope_with_file(path: &Path) -> (tauri::App<tauri::test::MockRuntime>, Scope) {
        let app = tauri::test::mock_app();
        let scope = app.handle().asset_protocol_scope();
        scope.allow_file(path).expect("grant dialog file");
        (app, scope)
    }

    #[test]
    fn unapproved_path_is_rejected_without_touching_the_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let target = dir.path().join("existing.srt");
        std::fs::write(&target, b"keep").expect("seed target");
        let app = tauri::test::mock_app();
        let scope = app.handle().asset_protocol_scope();

        let error = authorize_dialog_output(&scope, &target.to_string_lossy(), &["srt"])
            .expect_err("unapproved path");

        assert_eq!(error, UNAPPROVED_OUTPUT);
        assert_eq!(std::fs::read(&target).expect("read target"), b"keep");
    }

    #[test]
    fn recursive_directory_grant_does_not_authorize_output() {
        let dir = tempfile::tempdir().expect("tempdir");
        let app = tauri::test::mock_app();
        let scope = app.handle().asset_protocol_scope();
        scope
            .allow_directory(dir.path(), true)
            .expect("folder import grant");

        let error = authorize_dialog_output(
            &scope,
            &dir.path().join("cut.xml").to_string_lossy(),
            &["xml"],
        )
        .expect_err("directory grants are not dialog output grants");
        assert_eq!(error, UNAPPROVED_OUTPUT);
    }

    #[test]
    fn approved_path_with_extension_is_returned_unchanged() {
        let dir = tempfile::tempdir().expect("tempdir");
        let raw = dir.path().join("cut.XML");
        let (_app, scope) = scope_with_file(&raw);

        let resolved =
            authorize_dialog_output(&scope, &raw.to_string_lossy(), &["xml"]).expect("approved");

        assert_eq!(resolved, raw);
    }

    #[test]
    fn approved_path_without_extension_gets_it_in_the_same_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let raw = dir.path().join("cut");
        let (_app, scope) = scope_with_file(&raw);

        let resolved =
            authorize_dialog_output(&scope, &raw.to_string_lossy(), &["xml"]).expect("approved");

        assert_eq!(resolved, dir.path().join("cut.xml"));
        assert_eq!(resolved.parent(), raw.parent());
    }

    #[test]
    fn a_foreign_extension_is_kept_and_the_required_one_appended() {
        let dir = tempfile::tempdir().expect("tempdir");
        let raw = dir.path().join("notes.txt");
        let (_app, scope) = scope_with_file(&raw);

        let resolved =
            authorize_dialog_output(&scope, &raw.to_string_lossy(), &["m4a", "mp3", "wav"])
                .expect("approved");

        assert_eq!(resolved, dir.path().join("notes.txt.m4a"));
    }

    #[test]
    fn appended_path_that_already_exists_is_not_overwritten() {
        let dir = tempfile::tempdir().expect("tempdir");
        let raw = dir.path().join("cut");
        let existing = dir.path().join("cut.edl");
        std::fs::write(&existing, b"keep").expect("seed existing export");
        let (_app, scope) = scope_with_file(&raw);

        let error = authorize_dialog_output(&scope, &raw.to_string_lossy(), &["edl"])
            .expect_err("unconfirmed overwrite");

        assert!(error.contains("already exists"), "{error}");
        assert_eq!(std::fs::read(&existing).expect("read existing"), b"keep");
    }

    #[test]
    fn frontend_extended_path_no_longer_matches_the_dialog_grant() {
        let dir = tempfile::tempdir().expect("tempdir");
        let raw = dir.path().join("Vlog");
        let (_app, scope) = scope_with_file(&raw);

        let error = authorize_dialog_output(
            &scope,
            &dir.path().join("Vlog.opentake").to_string_lossy(),
            &["opentake"],
        )
        .expect_err("only the dialog result is authorized");
        assert_eq!(error, UNAPPROVED_OUTPUT);
    }

    #[test]
    fn parent_directory_changes_are_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let nested = dir.path().join("nested");
        std::fs::create_dir(&nested).expect("nested dir");
        let raw = nested.join("cut");
        let (_app, scope) = scope_with_file(&raw);

        for attempt in [
            nested.join("..").join("nested").join("cut"),
            dir.path().join("cut"),
        ] {
            assert!(
                authorize_dialog_output(&scope, &attempt.to_string_lossy(), &["xml"]).is_err(),
                "{} must be rejected",
                attempt.display()
            );
        }
        assert!(authorize_dialog_output(&scope, "cut", &["xml"]).is_err());
        assert!(authorize_dialog_output(&scope, "", &["xml"]).is_err());
    }

    #[test]
    fn forbidden_path_is_rejected_even_with_an_exact_grant() {
        let dir = tempfile::tempdir().expect("tempdir");
        let raw = dir.path().join("cut.xml");
        let (_app, scope) = scope_with_file(&raw);
        scope.forbid_file(&raw).expect("forbid");

        assert_eq!(
            authorize_dialog_output(&scope, &raw.to_string_lossy(), &["xml"]),
            Err(UNAPPROVED_OUTPUT.to_string())
        );
    }

    #[test]
    fn atomic_write_creates_and_replaces_regular_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let target = dir.path().join("cut.otio");

        write_file_atomically(&target, b"first").expect("create");
        write_file_atomically(&target, b"second").expect("replace");

        assert_eq!(std::fs::read(&target).expect("read"), b"second");
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
        let target = dir.path().join("cut.xml");
        std::fs::create_dir(&target).expect("directory target");

        assert!(write_file_atomically(&target, b"x").is_err());
        assert!(target.is_dir());
    }

    #[test]
    fn missing_directory_fails_without_creating_output() {
        let dir = tempfile::tempdir().expect("tempdir");
        let missing_parent = dir.path().join("gone").join("cut.xml");

        assert!(write_file_atomically(&missing_parent, b"x").is_err());
        assert!(!missing_parent.exists());
    }

    #[cfg(unix)]
    #[test]
    fn atomic_write_refuses_a_symlink_and_leaves_its_target_alone() {
        let dir = tempfile::tempdir().expect("tempdir");
        let victim = dir.path().join("victim.rc");
        std::fs::write(&victim, b"keep").expect("seed victim");
        let link = dir.path().join("cut.srt");
        std::os::unix::fs::symlink(&victim, &link).expect("symlink");

        assert!(write_file_atomically(&link, b"overwrite").is_err());
        assert_eq!(std::fs::read(&victim).expect("read victim"), b"keep");
        assert!(std::fs::symlink_metadata(&link)
            .expect("link")
            .file_type()
            .is_symlink());
    }
}
