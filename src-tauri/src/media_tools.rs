//! Desktop media commands always use the bundled pair, including in development.

use std::ffi::OsString;
use std::path::Path;

pub fn pin() -> std::io::Result<()> {
    let executable = std::env::current_exe()?;
    let target = env!("OPENTAKE_BUILD_TARGET");
    let development =
        cfg!(debug_assertions).then(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("binaries"));
    for (key, tool) in [
        ("OPENTAKE_FFMPEG", "ffmpeg"),
        ("OPENTAKE_FFPROBE", "ffprobe"),
    ] {
        let path = tool_path(
            tool,
            &executable,
            target,
            development.as_deref(),
            std::env::var_os(key),
        );
        std::env::set_var(key, path);
    }
    Ok(())
}

/// File name (without the target suffix or extension) of a bundled tool.
///
/// Linux deb and rpm bundles install external binaries beside the executable
/// in the shared `/usr/bin`, where plain `ffmpeg`/`ffprobe` would collide
/// with the distribution's FFmpeg package. macOS app bundles and Windows
/// installation directories are private to OpenTake and keep the plain names.
/// `tauri.linux.conf.json`, `scripts/provision_ffmpeg_sidecars.py` and
/// `scripts/tests/packaged-sidecars-test.rb` use the same names.
fn sidecar_name(tool: &str, target: &str) -> String {
    if target.contains("linux") {
        format!("opentake-{tool}")
    } else {
        tool.to_owned()
    }
}

fn tool_path(
    tool: &str,
    executable: &Path,
    target: &str,
    development: Option<&Path>,
    override_path: Option<OsString>,
) -> OsString {
    let name = sidecar_name(tool, target);
    let extension = if target.contains("windows") {
        ".exe"
    } else {
        ""
    };
    let sibling = executable.with_file_name(format!("{name}{extension}"));
    let Some(dir) = development else {
        // A damaged package must report its missing tool, without consulting PATH.
        return sibling.into_os_string();
    };
    if let Some(path) = override_path {
        return path;
    }
    if std::fs::symlink_metadata(&sibling).is_ok_and(|metadata| metadata.is_file()) {
        return sibling.into_os_string();
    }
    dir.join(format!("{name}-{target}{extension}"))
        .into_os_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const TARGETS: [&str; 4] = [
        "aarch64-apple-darwin",
        "x86_64-apple-darwin",
        "x86_64-unknown-linux-gnu",
        "x86_64-pc-windows-msvc",
    ];

    fn packaged_name(tool: &str, target: &str) -> String {
        match target {
            "x86_64-unknown-linux-gnu" => format!("opentake-{tool}"),
            "x86_64-pc-windows-msvc" => format!("{tool}.exe"),
            _ => tool.to_owned(),
        }
    }

    #[test]
    fn development_uses_target_named_binaries_even_when_missing() {
        for target in TARGETS {
            for tool in ["ffmpeg", "ffprobe"] {
                let (stem, extension) = match target {
                    "x86_64-unknown-linux-gnu" => (format!("opentake-{tool}"), ""),
                    "x86_64-pc-windows-msvc" => (tool.to_owned(), ".exe"),
                    _ => (tool.to_owned(), ""),
                };
                assert_eq!(
                    tool_path(
                        tool,
                        Path::new("missing/app"),
                        target,
                        Some(Path::new("resources")),
                        None
                    ),
                    PathBuf::from("resources")
                        .join(format!("{stem}-{target}{extension}"))
                        .into_os_string(),
                );
            }
        }
    }

    #[test]
    fn explicit_development_override_wins() {
        let custom = OsString::from("custom-ffmpeg");
        assert_eq!(
            tool_path(
                "ffmpeg",
                Path::new("app"),
                "aarch64-apple-darwin",
                Some(Path::new("resources")),
                Some(custom.clone())
            ),
            custom
        );
    }

    #[test]
    fn debug_packages_use_regular_siblings() {
        let dir = std::env::temp_dir().join(format!("opentake-tool-paths-{}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let executable = dir.join("app");
        for target in ["aarch64-apple-darwin", "x86_64-unknown-linux-gnu"] {
            let sibling = dir.join(packaged_name("ffmpeg", target));
            std::fs::write(&sibling, b"fixture").unwrap();
            let development = Some(Path::new("resources"));
            assert_eq!(
                tool_path("ffmpeg", &executable, target, development, None),
                sibling.as_os_str(),
            );
            #[cfg(unix)]
            {
                std::fs::remove_file(&sibling).unwrap();
                std::os::unix::fs::symlink("missing", &sibling).unwrap();
                let stem = sidecar_name("ffmpeg", target);
                assert_eq!(
                    tool_path("ffmpeg", &executable, target, development, None),
                    Path::new("resources")
                        .join(format!("{stem}-{target}"))
                        .as_os_str(),
                );
            }
            std::fs::remove_file(&sibling).unwrap();
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn package_paths_ignore_development_overrides_and_missing_files() {
        for target in TARGETS {
            for tool in ["ffmpeg", "ffprobe"] {
                assert_eq!(
                    tool_path(
                        tool,
                        Path::new("package/app"),
                        target,
                        None,
                        Some(OsString::from("host-tool"))
                    ),
                    PathBuf::from("package")
                        .join(packaged_name(tool, target))
                        .into_os_string(),
                );
            }
        }
    }

    /// deb/rpm bundles install sidecars into `/usr/bin`; they must never own
    /// the distribution's `ffmpeg`/`ffprobe` paths.
    #[test]
    fn linux_packages_never_claim_the_system_tool_names() {
        for tool in ["ffmpeg", "ffprobe"] {
            let packaged = tool_path(
                tool,
                Path::new("/usr/bin/opentake"),
                "x86_64-unknown-linux-gnu",
                None,
                None,
            );
            assert_eq!(
                packaged,
                PathBuf::from(format!("/usr/bin/opentake-{tool}")).into_os_string()
            );
            assert_ne!(
                packaged,
                PathBuf::from("/usr/bin").join(tool).into_os_string()
            );
        }
    }
}
