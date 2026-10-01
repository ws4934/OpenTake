//! Per-workspace recovery receipts. The receipt lives beside the workspace,
//! survives a crash during directory cleanup, and is locked for the full job.

use super::{EncodeWorkspace, ENCODE_WORKSPACE_PREFIX};
#[cfg(not(windows))]
use cap_fs_ext::DirExt;
use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
use cap_std::fs::{Dir, OpenOptions};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

pub(super) const MARKER: &str = ".opentake-owner";
const SUFFIX: &str = ".recovery.json";
const MAX_RECEIPT: u64 = 4096;

#[derive(Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct FileKey {
    volume: u64,
    index: u64,
}

impl FileKey {
    #[cfg(unix)]
    fn read(file: &File) -> io::Result<Self> {
        use std::os::unix::fs::MetadataExt;
        let metadata = file.metadata()?;
        Ok(Self {
            volume: metadata.dev(),
            index: metadata.ino(),
        })
    }

    #[cfg(windows)]
    fn read(file: &File) -> io::Result<Self> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Storage::FileSystem::{
            GetFileInformationByHandle, BY_HANDLE_FILE_INFORMATION,
        };
        let mut info = BY_HANDLE_FILE_INFORMATION::default();
        // SAFETY: the retained handle and output buffer are valid for the call.
        if unsafe { GetFileInformationByHandle(file.as_raw_handle(), &mut info) } == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            volume: info.dwVolumeSerialNumber.into(),
            index: (u64::from(info.nFileIndexHigh) << 32) | u64::from(info.nFileIndexLow),
        })
    }

    #[cfg(not(any(unix, windows)))]
    fn read(_file: &File) -> io::Result<Self> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "workspace identity unavailable",
        ))
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    version: u8,
    directory: FileKey,
    marker: String,
}

pub(super) struct RecoveryRecord {
    #[cfg(not(windows))]
    parent: Dir,
    #[cfg(not(windows))]
    name: String,
    file: File,
}

fn receipt_options(create: bool) -> OpenOptions {
    let mut options = OpenOptions::new();
    options
        .read(true)
        .write(true)
        .create_new(create)
        .follow(FollowSymlinks::No);
    #[cfg(unix)]
    {
        use cap_std::fs::OpenOptionsExt;
        options.mode(0o600).custom_flags(libc::O_NONBLOCK);
    }
    #[cfg(windows)]
    {
        use cap_std::fs::OpenOptionsExt;
        use windows_sys::Win32::Foundation::{GENERIC_READ, GENERIC_WRITE};
        use windows_sys::Win32::Storage::FileSystem::{
            DELETE, FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE,
        };
        options
            .access_mode(GENERIC_READ | GENERIC_WRITE | DELETE)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE);
    }
    options
}

fn privately_owned(file: &File) -> io::Result<bool> {
    let metadata = file.metadata()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        // Recovery can conservatively decline foreign/mapped ownership without
        // deciding whether the original filesystem supports valid encoding.
        // SAFETY: geteuid has no arguments or memory access requirements.
        Ok(metadata.uid() == unsafe { libc::geteuid() } && metadata.mode() & 0o022 == 0)
    }
    #[cfg(not(unix))]
    {
        Ok(!super::metadata_is_link(&metadata))
    }
}

impl RecoveryRecord {
    pub(super) fn create(workspace: &EncodeWorkspace) -> io::Result<Self> {
        let parent = Dir::open_ambient_dir(
            workspace.path.parent().expect("workspace parent"),
            cap_std::ambient_authority(),
        )?;
        let marker = workspace
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| io::Error::other("workspace leaf is not ASCII"))?
            .to_owned();
        let name = format!("{marker}{SUFFIX}");
        let file = parent.open_with(&name, &receipt_options(true))?.into_std();
        let mut record = Self {
            #[cfg(not(windows))]
            parent: parent.try_clone()?,
            #[cfg(not(windows))]
            name,
            file,
        };
        let registration = (|| -> io::Result<()> {
            record.file.try_lock().map_err(io::Error::from)?;
            #[cfg(windows)]
            if let Err(error) = EncodeWorkspace::hide_file(&record.file) {
                tracing::warn!(%error, "could not hide encode recovery receipt");
            }
            let receipt = Receipt {
                version: 1,
                directory: FileKey::read(&workspace.directory)?,
                marker: marker.clone(),
            };
            record.file.write_all(&serde_json::to_vec(&receipt)?)?;
            record.file.sync_all()?;
            let directory = Dir::from_std_file(workspace.directory.try_clone()?);
            let mut marker_options = OpenOptions::new();
            marker_options
                .write(true)
                .create_new(true)
                .follow(FollowSymlinks::No);
            let mut owned_marker = directory.open_with(MARKER, &marker_options)?;
            owned_marker.write_all(marker.as_bytes())?;
            owned_marker.sync_all()?;
            #[cfg(unix)]
            {
                workspace.directory.sync_all()?;
                // A capability directory can be O_PATH on Linux. Open a
                // readable descriptor through it before flushing the entry.
                parent.open(".")?.sync_all()?;
            }
            Ok(())
        })();
        if let Err(error) = registration {
            if let Err(cleanup) = record.remove() {
                tracing::warn!(%cleanup, "failed to roll back encode recovery receipt");
            }
            return Err(error);
        }
        Ok(record)
    }

    pub(super) fn remove(&self) -> io::Result<()> {
        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawHandle;
            use windows_sys::Win32::Storage::FileSystem::{
                FileDispositionInfo, SetFileInformationByHandle, FILE_DISPOSITION_INFO,
            };
            let info = FILE_DISPOSITION_INFO { DeleteFile: true };
            // SAFETY: this DELETE-capable receipt handle and the SDK buffer
            // remain live; a rebound pathname is never opened for deletion.
            if unsafe {
                SetFileInformationByHandle(
                    self.file.as_raw_handle(),
                    FileDispositionInfo,
                    (&info as *const FILE_DISPOSITION_INFO).cast(),
                    std::mem::size_of_val(&info) as u32,
                )
            } == 0
            {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        }
        #[cfg(not(windows))]
        {
            let visible = match self.parent.open_with(&self.name, &receipt_options(false)) {
                Ok(file) => file.into_std(),
                Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
                Err(error) => return Err(error),
            };
            if FileKey::read(&visible)? != FileKey::read(&self.file)? {
                return Err(io::Error::other("workspace receipt was replaced"));
            }
            self.parent.remove_file(&self.name)
        }
    }

    pub(super) fn configure_child(&self, command: &mut std::process::Command) -> io::Result<()> {
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            use std::os::unix::process::CommandExt;
            let lease = self.file.try_clone()?;
            // SAFETY: fcntl is async-signal-safe. The closure owns the cloned
            // descriptor; only the forked child clears close-on-exec, so other
            // concurrent launches cannot accidentally inherit this lease.
            unsafe {
                command.pre_exec(move || {
                    let flags = libc::fcntl(lease.as_raw_fd(), libc::F_GETFD);
                    if flags < 0
                        || libc::fcntl(lease.as_raw_fd(), libc::F_SETFD, flags & !libc::FD_CLOEXEC)
                            < 0
                    {
                        return Err(io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
        }
        // Windows encoders are placed in kill-on-close Job Objects before
        // they run. Parent death terminates their entire process tree.
        #[cfg(not(unix))]
        let _ = command;
        Ok(())
    }
}

pub(super) fn recover(parent_path: &Path) -> io::Result<()> {
    let parent = Dir::open_ambient_dir(parent_path, cap_std::ambient_authority())?;
    for entry in parent.entries()? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name
            .to_str()
            .filter(|name| name.starts_with(ENCODE_WORKSPACE_PREFIX) && name.ends_with(SUFFIX))
        else {
            continue;
        };
        if let Err(error) = recover_one(&parent, parent_path, name) {
            tracing::warn!(%error, receipt = name, "could not recover encode workspace");
        }
    }
    Ok(())
}

fn recover_one(parent: &Dir, parent_path: &Path, name: &str) -> io::Result<()> {
    let mut file = parent.open_with(name, &receipt_options(false))?.into_std();
    if !file.metadata()?.is_file() || !privately_owned(&file)? {
        return Ok(());
    }
    match file.try_lock().map_err(io::Error::from) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(()),
        Err(error) => return Err(error),
    }
    if file.metadata()?.len() > MAX_RECEIPT {
        return Err(io::Error::other("oversized workspace receipt"));
    }
    let mut bytes = Vec::new();
    (&mut file).take(MAX_RECEIPT + 1).read_to_end(&mut bytes)?;
    let receipt: Receipt = serde_json::from_slice(&bytes)?;
    let leaf = name.strip_suffix(SUFFIX).expect("receipt suffix");
    if receipt.version != 1 || receipt.marker != leaf {
        return Err(io::Error::other("invalid workspace receipt"));
    }
    let record = RecoveryRecord {
        #[cfg(not(windows))]
        parent: parent.try_clone()?,
        #[cfg(not(windows))]
        name: name.to_owned(),
        file,
    };
    #[cfg(not(windows))]
    let directory = parent.open_dir_nofollow(leaf).map(Dir::into_std_file);
    #[cfg(windows)]
    let directory =
        super::open_directory_nofollow(&parent_path.join(leaf)).map_err(|error| match error {
            crate::MediaError::Io(error) => error,
            other => io::Error::other(other.to_string()),
        });
    let directory = match directory {
        Ok(directory) => directory,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return record.remove(),
        Err(error) => return Err(error),
    };
    if FileKey::read(&directory)? != receipt.directory || !privately_owned(&directory)? {
        return Ok(());
    }
    let owned = Dir::from_std_file(directory.try_clone()?);
    let entries: Vec<_> = owned.entries()?.collect::<io::Result<_>>()?;
    if !entries.is_empty() {
        let mut options = OpenOptions::new();
        options.read(true).follow(FollowSymlinks::No);
        #[cfg(unix)]
        {
            use cap_std::fs::OpenOptionsExt;
            options.custom_flags(libc::O_NONBLOCK);
        }
        let marker_file = owned.open_with(MARKER, &options)?;
        let metadata = marker_file.metadata()?;
        if !metadata.is_file() || metadata.len() > 256 {
            return Err(io::Error::other("invalid workspace ownership marker"));
        }
        let mut marker = Vec::new();
        marker_file.take(256).read_to_end(&mut marker)?;
        if marker != receipt.marker.as_bytes() {
            return Err(io::Error::other("workspace ownership marker changed"));
        }
        for entry in entries {
            if !entry.file_type()?.is_file()
                || !matches!(
                    entry.file_name().to_str(),
                    Some(
                        "video.mp4"
                            | "video.mov"
                            | "audio.pcm"
                            | "muxed.mp4"
                            | "muxed.mov"
                            | MARKER
                    )
                )
            {
                return Err(io::Error::other("workspace contains an unrecognized entry"));
            }
        }
    }
    // Drop uses the same handle-bound cleanup as a live encode. The receipt is
    // removed only after the workspace is gone, including crash-after-unlink.
    drop(EncodeWorkspace {
        directory,
        path: parent_path.join(leaf),
        recovery: Some(record),
        #[cfg(unix)]
        parent: parent.try_clone()?.into_std_file(),
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    fn crashed_workspace(parent: &Path) -> std::path::PathBuf {
        assert!(Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "encode::tests::next_encode_recovers_a_crashed_workspace_but_preserves_active_and_foreign_dirs", "--test-threads=1"])
            .env("OPENTAKE_TEST_CRASH_WORKSPACE_ROOT", parent)
            .status().unwrap().success());
        std::fs::read_dir(parent)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .find(|path| path.is_dir())
            .unwrap()
    }

    #[test]
    fn recovery_preserves_a_rebound_directory_even_with_a_copied_marker() {
        let parent = tempfile::tempdir().unwrap();
        let stale = crashed_workspace(parent.path());
        let marker = std::fs::read(stale.join(MARKER)).unwrap();
        let moved = parent.path().join("moved-owned-directory");
        std::fs::rename(&stale, &moved).unwrap();
        std::fs::create_dir(&stale).unwrap();
        std::fs::write(stale.join(MARKER), marker).unwrap();
        std::fs::write(stale.join("video.mp4"), b"foreign video").unwrap();
        recover(parent.path()).unwrap();
        assert_eq!(
            std::fs::read(stale.join("video.mp4")).unwrap(),
            b"foreign video"
        );
        assert_eq!(
            std::fs::read(moved.join("video.mp4")).unwrap(),
            b"unfinished encode"
        );
    }

    #[test]
    fn recovery_preserves_unrecognized_contents_and_modified_markers() {
        for extra_file in [true, false] {
            let parent = tempfile::tempdir().unwrap();
            let stale = crashed_workspace(parent.path());
            if extra_file {
                std::fs::write(stale.join("notes.txt"), b"user notes").unwrap();
            } else {
                std::fs::write(stale.join(MARKER), b"different owner").unwrap();
            }
            recover(parent.path()).unwrap();
            assert_eq!(
                std::fs::read(stale.join("video.mp4")).unwrap(),
                b"unfinished encode"
            );
        }
    }

    #[test]
    fn recovery_finishes_both_crash_windows_after_payload_cleanup() {
        for remove_directory in [false, true] {
            let parent = tempfile::tempdir().unwrap();
            let stale = crashed_workspace(parent.path());
            std::fs::remove_file(stale.join("video.mp4")).unwrap();
            std::fs::remove_file(stale.join(MARKER)).unwrap();
            if remove_directory {
                std::fs::remove_dir(&stale).unwrap();
            }
            recover(parent.path()).unwrap();
            assert_eq!(std::fs::read_dir(parent.path()).unwrap().count(), 0);
        }
    }

    #[cfg(unix)]
    #[test]
    fn receipt_symlinks_are_not_followed() {
        let parent = tempfile::tempdir().unwrap();
        let outside = parent.path().join("outside.json");
        std::fs::write(&outside, b"user data").unwrap();
        std::os::unix::fs::symlink(
            &outside,
            parent
                .path()
                .join(format!("{ENCODE_WORKSPACE_PREFIX}fake{SUFFIX}")),
        )
        .unwrap();
        recover(parent.path()).unwrap();
        assert_eq!(std::fs::read(outside).unwrap(), b"user data");
    }

    #[test]
    fn registration_failure_removes_only_its_new_receipt() {
        let parent = tempfile::tempdir().unwrap();
        let mut workspace = EncodeWorkspace::in_directory(parent.path()).unwrap();
        let original = workspace.recovery.take().unwrap();
        original.remove().unwrap();
        drop(original);
        std::fs::write(workspace.path.join(MARKER), b"existing marker").unwrap();
        assert!(RecoveryRecord::create(&workspace).is_err());
        assert_eq!(
            std::fs::read(workspace.path.join(MARKER)).unwrap(),
            b"existing marker"
        );
        assert_eq!(
            std::fs::read_dir(parent.path()).unwrap().count(),
            1,
            "failed registration must not accumulate unusable receipts"
        );
    }

    #[cfg(unix)]
    #[test]
    fn special_receipts_and_markers_cannot_block_recovery() {
        use std::os::unix::ffi::OsStrExt;
        let parent = tempfile::tempdir().unwrap();
        let stale = crashed_workspace(parent.path());
        std::fs::remove_file(stale.join(MARKER)).unwrap();
        for path in [
            stale.join(MARKER),
            parent
                .path()
                .join(format!("{ENCODE_WORKSPACE_PREFIX}pipe{SUFFIX}")),
        ] {
            let path = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
            // SAFETY: the generated path is a live NUL-terminated string.
            assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
        }
        recover(parent.path()).unwrap();
        assert_eq!(
            std::fs::read(stale.join("video.mp4")).unwrap(),
            b"unfinished encode"
        );
    }

    #[cfg(unix)]
    #[test]
    fn surviving_encoder_keeps_the_receipt_locked_after_its_parent_exits() {
        use std::os::unix::fs::PermissionsExt;
        use std::time::{Duration, Instant};
        const ROOT: &str = "OPENTAKE_TEST_ORPHAN_ENCODER_ROOT";
        if let Some(root) = std::env::var_os(ROOT) {
            let root = std::path::PathBuf::from(root);
            crate::ff::test_seams::override_ffmpeg(Some(root.join("helper").into_os_string()));
            let preset = super::super::ExportPreset::new(
                super::super::VideoCodec::H264,
                super::super::ExportResolution::P720,
            );
            let _encoder = super::super::VideoEncoder::new_in_workspace(
                &root.join("movie.mp4"),
                2,
                2,
                30,
                &preset,
            )
            .unwrap();
            let deadline = Instant::now() + Duration::from_secs(5);
            while !root.join("pid").is_file() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(5));
            }
            assert!(root.join("pid").is_file());
            std::process::exit(0);
        }
        let parent = tempfile::tempdir().unwrap();
        let helper = parent.path().join("helper");
        std::fs::write(
            &helper,
            format!(
                "#!/bin/sh\nprintf '%s' \"$$\" > '{}'\nexec sleep 60\n",
                parent.path().join("pid").display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&helper, std::fs::Permissions::from_mode(0o700)).unwrap();
        assert!(Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "encode::recovery::tests::surviving_encoder_keeps_the_receipt_locked_after_its_parent_exits", "--test-threads=1"])
            .env(ROOT, parent.path()).status().unwrap().success());
        let pid: u32 = std::fs::read_to_string(parent.path().join("pid"))
            .unwrap()
            .parse()
            .unwrap();
        let workspace = std::fs::read_dir(parent.path())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .find(|path| path.is_dir())
            .unwrap();
        recover(parent.path()).unwrap();
        let preserved_while_live = workspace.is_dir();
        assert!(Command::new("kill")
            .args(["-KILL", &pid.to_string()])
            .status()
            .unwrap()
            .success());
        let deadline = Instant::now() + Duration::from_secs(5);
        while workspace.exists() && Instant::now() < deadline {
            recover(parent.path()).unwrap();
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(
            preserved_while_live,
            "a live orphan must retain the workspace lease"
        );
        assert!(!workspace.exists(), "the orphan's exit releases recovery");
    }
}
