//! Desktop media commands always use the bundled pair, including in development.

use std::ffi::OsString;
use std::path::Path;

pub fn pin() -> std::io::Result<()> {
    let executable = std::env::current_exe()?;
    let development = cfg!(debug_assertions).then_some((
        Path::new(env!("CARGO_MANIFEST_DIR")).join("binaries"),
        env!("OPENTAKE_BUILD_TARGET"),
    ));
    for (key, tool) in [
        ("OPENTAKE_FFMPEG", "ffmpeg"),
        ("OPENTAKE_FFPROBE", "ffprobe"),
    ] {
        let path = tool_path(
            tool,
            &executable,
            development
                .as_ref()
                .map(|(dir, target)| (dir.as_path(), *target)),
            std::env::var_os(key),
        );
        std::env::set_var(key, path);
    }
    Ok(())
}

fn tool_path(
    tool: &str,
    executable: &Path,
    development: Option<(&Path, &str)>,
    override_path: Option<OsString>,
) -> OsString {
    let extension = if cfg!(windows) { ".exe" } else { "" };
    let sibling = executable.with_file_name(format!("{tool}{extension}"));
    let Some((dir, target)) = development else {
        // A damaged package must report its missing tool, without consulting PATH.
        return sibling.into_os_string();
    };
    if let Some(path) = override_path {
        return path;
    }
    if std::fs::symlink_metadata(&sibling).is_ok_and(|metadata| metadata.is_file()) {
        return sibling.into_os_string();
    }
    let extension = if target.contains("windows") {
        ".exe"
    } else {
        ""
    };
    dir.join(format!("{tool}-{target}{extension}"))
        .into_os_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn development_uses_target_named_binaries_even_when_missing() {
        for target in [
            "aarch64-apple-darwin",
            "x86_64-apple-darwin",
            "x86_64-unknown-linux-gnu",
            "x86_64-pc-windows-msvc",
        ] {
            for tool in ["ffmpeg", "ffprobe"] {
                let extension = if target.contains("windows") {
                    ".exe"
                } else {
                    ""
                };
                assert_eq!(
                    tool_path(
                        tool,
                        Path::new("missing/app"),
                        Some((Path::new("resources"), target)),
                        None
                    ),
                    PathBuf::from("resources")
                        .join(format!("{tool}-{target}{extension}"))
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
                Some((Path::new("resources"), "aarch64-apple-darwin")),
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
        let extension = if cfg!(windows) { ".exe" } else { "" };
        let sibling = dir.join(format!("ffmpeg{extension}"));
        std::fs::write(&sibling, b"fixture").unwrap();
        let development = Some((Path::new("resources"), "aarch64-apple-darwin"));
        assert_eq!(
            tool_path("ffmpeg", &executable, development, None),
            sibling.as_os_str(),
        );
        #[cfg(unix)]
        {
            std::fs::remove_file(&sibling).unwrap();
            std::os::unix::fs::symlink("missing", &sibling).unwrap();
            assert_eq!(
                tool_path("ffmpeg", &executable, development, None),
                Path::new("resources/ffmpeg-aarch64-apple-darwin").as_os_str(),
            );
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn package_paths_ignore_development_overrides_and_missing_files() {
        let extension = if cfg!(windows) { ".exe" } else { "" };
        for tool in ["ffmpeg", "ffprobe"] {
            assert_eq!(
                tool_path(
                    tool,
                    Path::new("package/app"),
                    None,
                    Some(OsString::from("host-tool"))
                ),
                PathBuf::from("package")
                    .join(format!("{tool}{extension}"))
                    .into_os_string(),
            );
        }
    }
}
