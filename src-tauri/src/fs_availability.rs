//! Non-materializing filesystem availability checks for UI-facing local paths.
//!
//! macOS File Provider placeholders report as ordinary files to `exists()` and
//! `is_file()`, but opening one may synchronously hydrate it from the network.
//! Tauri's asset protocol performs that open on the AppKit thread, so a broken
//! provider can otherwise freeze the whole window before an `<img>` `onError`
//! callback has a chance to run.

use std::fs::Metadata;
use std::path::Path;

#[cfg(target_os = "macos")]
use std::os::macos::fs::MetadataExt;
#[cfg(target_os = "windows")]
use std::os::windows::ffi::OsStrExt;
#[cfg(target_os = "windows")]
use std::os::windows::fs::MetadataExt;

#[cfg(target_os = "windows")]
use windows_sys::Win32::Foundation::{HANDLE, INVALID_HANDLE_VALUE};
#[cfg(target_os = "windows")]
use windows_sys::Win32::Storage::CloudFilters::CfGetPlaceholderStateFromAttributeTag;
#[cfg(target_os = "windows")]
use windows_sys::Win32::Storage::FileSystem::{
    FileAttributeTagInfo, FindClose, FindFirstFileW, GetFileInformationByHandleEx,
    FILE_ATTRIBUTE_TAG_INFO, FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_NO_RECALL,
    FILE_FLAG_OPEN_REPARSE_POINT, FILE_READ_ATTRIBUTES, FILE_SHARE_DELETE, FILE_SHARE_READ,
    FILE_SHARE_WRITE, WIN32_FIND_DATAW,
};

#[cfg(target_os = "macos")]
const MACOS_SF_DATALESS: u32 = 0x4000_0000;

#[cfg(target_os = "macos")]
const IOPOL_TYPE_VFS_MATERIALIZE_DATALESS_FILES: libc::c_int = 3;
#[cfg(target_os = "macos")]
const IOPOL_SCOPE_PROCESS: libc::c_int = 0;
#[cfg(target_os = "macos")]
const IOPOL_MATERIALIZE_DATALESS_FILES_OFF: libc::c_int = 1;

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn setiopolicy_np(io_type: libc::c_int, scope: libc::c_int, policy: libc::c_int)
        -> libc::c_int;
}

#[cfg(target_os = "macos")]
fn flags_are_dataless(flags: u32) -> bool {
    flags & MACOS_SF_DATALESS != 0
}

#[cfg(target_os = "macos")]
fn metadata_is_dataless(metadata: &Metadata) -> bool {
    flags_are_dataless(metadata.st_flags())
}

#[cfg(not(target_os = "macos"))]
fn metadata_is_dataless(_metadata: &Metadata) -> bool {
    #[cfg(target_os = "windows")]
    {
        windows_attributes_are_unavailable(_metadata.file_attributes())
    }
    #[cfg(not(target_os = "windows"))]
    {
        false
    }
}

#[cfg(any(target_os = "windows", test))]
const WINDOWS_UNAVAILABLE_ATTRIBUTES: u32 = 0x0000_1000 | 0x0004_0000 | 0x0040_0000;
#[cfg(any(target_os = "windows", test))]
const WINDOWS_PARTIAL_PLACEHOLDER_STATES: u32 = 0x10 | 0x20;

#[cfg(any(target_os = "windows", test))]
fn windows_attributes_are_unavailable(attributes: u32) -> bool {
    attributes & WINDOWS_UNAVAILABLE_ATTRIBUTES != 0
}

#[cfg(any(target_os = "windows", test))]
fn windows_placeholder_state_is_partial(state: u32) -> bool {
    state & WINDOWS_PARTIAL_PLACEHOLDER_STATES != 0
}

/// Win32 extended-length form (`\\?\`) of an absolute path that Win32 has
/// already normalized (`std::path::absolute` uses `GetFullPathNameW`; the
/// prefix turns normalization off). `FindFirstFileW` is not long-path aware
/// without the process opt-in, so paths of MAX_PATH (260) or more need it.
/// Returns `None` for paths that are already verbatim, device or relative.
#[cfg(any(target_os = "windows", test))]
fn windows_extended_length_path(absolute: &str) -> Option<String> {
    if absolute.starts_with(r"\\?\") || absolute.starts_with(r"\\.\") {
        return None;
    }
    if let Some(unc) = absolute.strip_prefix(r"\\") {
        return Some(format!(r"\\?\UNC\{unc}"));
    }
    let bytes = absolute.as_bytes();
    (bytes.len() >= 3 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' && bytes[2] == b'\\')
        .then(|| format!(r"\\?\{absolute}"))
}

#[cfg(target_os = "windows")]
fn windows_placeholder_is_partial(path: &Path) -> bool {
    let absolute = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    let query_path = absolute
        .to_str()
        .and_then(windows_extended_length_path)
        .map_or_else(|| absolute.clone().into_os_string(), Into::into);
    let mut path_wide = query_path.encode_wide().collect::<Vec<_>>();
    path_wide.push(0);
    let mut find_data = WIN32_FIND_DATAW::default();
    // SAFETY: `path_wide` is NUL-terminated and `find_data` is writable for
    // the duration of the Win32 calls. A successful find handle is closed once.
    let handle = unsafe { FindFirstFileW(path_wide.as_ptr(), &mut find_data) };
    if handle == INVALID_HANDLE_VALUE {
        // A path that vanished is unavailable. Any other failed directory
        // query is not evidence of a placeholder: ask the file itself, and
        // fail closed only when that attribute query fails as well.
        if std::io::Error::last_os_error().kind() == std::io::ErrorKind::NotFound {
            return true;
        }
        return windows_handle_placeholder_is_partial(path).unwrap_or(true);
    }
    // SAFETY: `handle` is the live search handle returned above.
    unsafe { FindClose(handle) };
    // SAFETY: this pure Cloud Files classifier accepts the attributes and
    // reparse tag returned in the same WIN32_FIND_DATAW snapshot.
    let state = unsafe {
        CfGetPlaceholderStateFromAttributeTag(find_data.dwFileAttributes, find_data.dwReserved0)
    };
    windows_placeholder_state_is_partial(state)
}

/// Classify through an attribute-only handle. Neither the access mask nor
/// `FILE_FLAG_OPEN_NO_RECALL` requests file data, so this cannot hydrate a
/// cloud placeholder; std opens long paths in their extended-length form.
#[cfg(target_os = "windows")]
fn windows_handle_placeholder_is_partial(path: &Path) -> std::io::Result<bool> {
    use std::os::windows::fs::OpenOptionsExt;
    use std::os::windows::io::AsRawHandle;

    let file = std::fs::OpenOptions::new()
        .access_mode(FILE_READ_ATTRIBUTES)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
        .custom_flags(
            FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_NO_RECALL,
        )
        .open(path)?;
    let mut info = FILE_ATTRIBUTE_TAG_INFO::default();
    // SAFETY: `file` owns a live handle and `info` is writable for its exact size.
    if unsafe {
        GetFileInformationByHandleEx(
            file.as_raw_handle() as HANDLE,
            FileAttributeTagInfo,
            std::ptr::addr_of_mut!(info).cast(),
            u32::try_from(std::mem::size_of::<FILE_ATTRIBUTE_TAG_INFO>()).unwrap_or(u32::MAX),
        )
    } == 0
    {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: the values come from the same attribute/tag query.
    let state =
        unsafe { CfGetPlaceholderStateFromAttributeTag(info.FileAttributes, info.ReparseTag) };
    Ok(windows_attributes_are_unavailable(info.FileAttributes)
        || windows_placeholder_state_is_partial(state))
}

#[cfg(not(target_os = "windows"))]
fn path_is_unavailable(_path: &Path, metadata: &Metadata) -> bool {
    metadata_is_dataless(metadata)
}

#[cfg(target_os = "windows")]
fn path_is_unavailable(path: &Path, metadata: &Metadata) -> bool {
    metadata_is_dataless(metadata) || windows_placeholder_is_partial(path)
}

/// Whether a path is an on-disk regular file that can be handed to the asset
/// protocol without implicitly asking macOS to download its contents.
pub(crate) fn is_materialized_regular_file(path: &Path) -> bool {
    std::fs::symlink_metadata(path)
        .ok()
        .is_some_and(|metadata| is_materialized_regular_file_metadata(path, &metadata))
}

pub(crate) fn is_materialized_regular_file_metadata(path: &Path, metadata: &Metadata) -> bool {
    metadata.is_file() && !metadata.file_type().is_symlink() && !path_is_unavailable(path, metadata)
}

/// Whether an existing path is a macOS File Provider placeholder whose bytes
/// are not currently resident. Metadata inspection itself does not hydrate it.
pub(crate) fn is_dataless(path: &Path) -> bool {
    std::fs::symlink_metadata(path)
        .ok()
        .is_some_and(|metadata| path_is_unavailable(path, &metadata))
}

/// Components read synchronously when a project opens. A dataless optional
/// component must also block the open because reading it would otherwise hang
/// or leave a partially decoded project.
pub(crate) fn project_bundle_has_dataless_components(bundle: &Path) -> bool {
    ["project.json", "media.json", "generation-log.json"]
        .into_iter()
        .map(|name| bundle.join(name))
        .any(|path| is_dataless(&path))
}

/// Disable implicit cloud hydration for this process. The OS will return a
/// prompt I/O error for a dataless file instead of blocking the AppKit thread;
/// explicit download/materialization remains the file provider's responsibility.
pub(crate) fn disable_implicit_dataless_materialization() -> std::io::Result<()> {
    #[cfg(target_os = "macos")]
    {
        // SAFETY: `setiopolicy_np` is a process-local macOS API. The constants
        // and signature are taken from `<sys/resource.h>`; no pointers cross FFI.
        let result = unsafe {
            setiopolicy_np(
                IOPOL_TYPE_VFS_MATERIALIZE_DATALESS_FILES,
                IOPOL_SCOPE_PROCESS,
                IOPOL_MATERIALIZE_DATALESS_FILES_OFF,
            )
        };
        if result != 0 {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinary_regular_file_is_materialized() {
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("thumbnail.jpg");
        std::fs::write(&file, b"jpeg").unwrap();

        assert!(is_materialized_regular_file(&file));
        assert!(!is_dataless(&file));
        assert!(!is_materialized_regular_file(directory.path()));
    }

    #[test]
    fn windows_cloud_file_flags_and_partial_states_fail_closed() {
        assert!(windows_attributes_are_unavailable(0x0000_1000));
        assert!(windows_attributes_are_unavailable(0x0004_0000));
        assert!(windows_attributes_are_unavailable(0x0040_0000));
        assert!(!windows_attributes_are_unavailable(0x0000_0020));
        assert!(windows_placeholder_state_is_partial(0x10));
        assert!(windows_placeholder_state_is_partial(0x20));
        assert!(!windows_placeholder_state_is_partial(0x08));
    }

    #[test]
    fn windows_extended_length_paths_cover_disk_and_unc_forms() {
        assert_eq!(
            windows_extended_length_path(r"C:\Media\My clip.mp4").as_deref(),
            Some(r"\\?\C:\Media\My clip.mp4")
        );
        assert_eq!(
            windows_extended_length_path(r"\\server\share\clip.mp4").as_deref(),
            Some(r"\\?\UNC\server\share\clip.mp4")
        );
        assert_eq!(windows_extended_length_path(r"\\?\C:\Media\clip.mp4"), None);
        assert_eq!(
            windows_extended_length_path(r"\\?\UNC\server\share\x"),
            None
        );
        assert_eq!(windows_extended_length_path(r"\\.\pipe\clip"), None);
        assert_eq!(windows_extended_length_path(r"Media\clip.mp4"), None);
        assert_eq!(windows_extended_length_path("C:"), None);
    }

    /// Paths of MAX_PATH or more used to fail the `FindFirstFileW` query and be
    /// reported as offline cloud placeholders.
    #[cfg(target_os = "windows")]
    #[test]
    fn windows_long_paths_are_materialized_regular_files() {
        let directory = tempfile::tempdir().unwrap();
        let mut parent = directory.path().to_path_buf();
        while parent.as_os_str().len() < 300 {
            parent.push("a".repeat(40));
        }
        std::fs::create_dir_all(&parent).unwrap();
        let file = parent.join("thumbnail.jpg");
        std::fs::write(&file, b"jpeg").unwrap();
        std::fs::write(parent.join("project.json"), b"{}").unwrap();
        assert!(file.as_os_str().len() > 260);

        assert!(is_materialized_regular_file(&file));
        assert!(!is_dataless(&file));
        assert_eq!(
            windows_handle_placeholder_is_partial(&file).ok(),
            Some(false)
        );
        assert!(!project_bundle_has_dataless_components(&parent));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_dataless_flag_detection_matches_the_sdk_contract() {
        assert_eq!(MACOS_SF_DATALESS, 0x4000_0000);
        assert!(flags_are_dataless(MACOS_SF_DATALESS));
        assert!(flags_are_dataless(MACOS_SF_DATALESS | 0x20));
        assert!(!flags_are_dataless(0));
        assert!(!flags_are_dataless(0x2000_0000));
    }
}
