//! Local media path → `file:` URL for the timeline-interchange exporters.
//!
//! XMEML `<pathurl>`, FCPXML `src` and OTIO `target_url` hold URLs, not paths,
//! so a path must be percent-encoded and, on Windows, rewritten from
//! `C:\dir\clip.mp4` to `/C:/dir/clip.mp4`. Upstream palmier-pro gets both from
//! Foundation (`URL.absoluteString`); this module is the cross-platform
//! equivalent.
//!
//! - Every byte outside the RFC 3986 unreserved set (`A-Z a-z 0-9 - . _ ~`) is
//!   written as `%XX`: what a URL reader would split or decode (space, `#`,
//!   `%`, `?`), every non-ASCII byte (UTF-8 on Windows, the raw and possibly
//!   non-UTF-8 bytes on Unix), and the sub-delimiters `'!$&()*+,;=`, which
//!   upstream's FCPXML `mediaSrc` encodes because DaVinci Resolve cannot relink
//!   them once they are written as XML entities (`&amp;`, `&apos;`).
//! - RFC 8089 forms: POSIX `/tmp/a b.mp4` → `file:///tmp/a%20b.mp4`; a drive
//!   path `C:\a b.mp4` (or `\\?\C:\a b.mp4`) → `file:///C:/a%20b.mp4`; a share
//!   `\\server\share\a.mp4` (or `\\?\UNC\server\share\a.mp4`) →
//!   `file://server/share/a.mp4`. A rooted path without a drive (`\a.mp4` on
//!   Windows) is written like a POSIX one.
//! - A path no `file:` URL can name — relative, or a `\\.\` device or other
//!   verbatim (`\\?\Volume{…}\`) root — becomes a relative reference of its
//!   encoded components.
//!
//! The platform's own [`Path::components`] splits a path into an [`Anchor`] and
//! name segments; everything after that is pure, so the Windows forms are unit
//! tested on every platform.

use std::borrow::Cow;
use std::ffi::OsStr;
use std::path::{Component, Path, Prefix};

/// RFC 8089 `file:` URL for `path` (see the module docs).
pub(crate) fn path_to_file_url(path: &Path) -> String {
    UrlPath::from_path(path).file_url()
}

/// XMEML `<pathurl>` for `path`. POSIX paths keep upstream's host form,
/// `file://localhost//` + `/abs/path` (so `file://localhost///abs/path`), which
/// Premiere and Resolve relink on macOS. A Windows drive path becomes
/// `file://localhost/C:/…`: the same `localhost` authority with the drive as
/// the first path segment, the shape of Premiere's own Windows export
/// (`file://localhost/C%3a/…`); upstream's extra slashes would put `//C:` in
/// front of the drive, which reads as a UNC share. Shares keep the standard
/// `file://server/share/…`.
pub(crate) fn xmeml_path_url(path: &Path) -> String {
    UrlPath::from_path(path).xmeml_path_url()
}

/// What a path is rooted at, as a `file:` URL sees it.
#[derive(Debug)]
enum Anchor<'a> {
    /// A rooted path without a drive or share: POSIX `/…` (`\…` on Windows).
    Root,
    /// A Windows drive: `C:\…` or `\\?\C:\…`.
    Drive(u8),
    /// A Windows share: `\\server\share\…` or `\\?\UNC\server\share\…`.
    Unc {
        server: Cow<'a, [u8]>,
        share: Cow<'a, [u8]>,
    },
    /// No `file:` URL form: a relative, `\\.\` device or other verbatim path.
    Relative,
}

/// A path as a URL sees it: its [`Anchor`] and the name segments below it.
struct UrlPath<'a> {
    anchor: Anchor<'a>,
    segments: Vec<Cow<'a, [u8]>>,
}

impl<'a> UrlPath<'a> {
    /// Split `path` with the platform's own parser. Prefixes only occur on
    /// Windows; `.` components are dropped and `..` is kept for the reader.
    fn from_path(path: &'a Path) -> Self {
        let mut anchor = Anchor::Relative;
        let mut segments = Vec::new();
        for (index, component) in path.components().enumerate() {
            match component {
                Component::Prefix(prefix) => match prefix.kind() {
                    Prefix::Disk(letter) | Prefix::VerbatimDisk(letter) => {
                        anchor = Anchor::Drive(letter);
                    }
                    Prefix::UNC(server, share) | Prefix::VerbatimUNC(server, share) => {
                        anchor = Anchor::Unc {
                            server: os_bytes(server),
                            share: os_bytes(share),
                        };
                    }
                    // Keep the root's text rather than dropping it silently.
                    Prefix::Verbatim(_) | Prefix::DeviceNS(_) => {
                        segments.push(os_bytes(prefix.as_os_str()));
                    }
                },
                Component::RootDir if index == 0 => anchor = Anchor::Root,
                Component::RootDir | Component::CurDir => {}
                Component::ParentDir => segments.push(Cow::Borrowed(&b".."[..])),
                Component::Normal(name) => segments.push(os_bytes(name)),
            }
        }
        UrlPath { anchor, segments }
    }

    fn file_url(&self) -> String {
        match &self.anchor {
            Anchor::Root | Anchor::Drive(_) => format!("file://{}", self.local_url_path()),
            Anchor::Unc { server, share } => {
                let mut url = String::from("file://");
                push_encoded(&mut url, server);
                url.push('/');
                push_encoded(&mut url, share);
                self.push_segments(&mut url);
                url
            }
            Anchor::Relative => {
                let mut reference = String::new();
                for (index, segment) in self.segments.iter().enumerate() {
                    if index > 0 {
                        reference.push('/');
                    }
                    push_encoded(&mut reference, segment);
                }
                reference
            }
        }
    }

    fn xmeml_path_url(&self) -> String {
        match &self.anchor {
            Anchor::Root => format!("file://localhost//{}", self.local_url_path()),
            Anchor::Drive(_) => format!("file://localhost{}", self.local_url_path()),
            Anchor::Unc { .. } | Anchor::Relative => self.file_url(),
        }
    }

    /// URL path of a [`Anchor::Root`] or [`Anchor::Drive`] path: `/tmp/a%20b`,
    /// `/C:/a%20b`; `/` and `/C:/` for the bare roots.
    fn local_url_path(&self) -> String {
        let mut path = String::new();
        if let Anchor::Drive(letter) = self.anchor {
            path.push('/');
            path.push(char::from(letter));
            path.push(':');
        }
        self.push_segments(&mut path);
        if self.segments.is_empty() {
            path.push('/');
        }
        path
    }

    fn push_segments(&self, out: &mut String) {
        for segment in &self.segments {
            out.push('/');
            push_encoded(out, segment);
        }
    }
}

/// The bytes of one path component: the raw bytes on Unix, so non-UTF-8 names
/// survive; UTF-8 elsewhere (unpaired UTF-16 surrogates on Windows, which no
/// URL can carry, become U+FFFD).
#[cfg(unix)]
fn os_bytes(name: &OsStr) -> Cow<'_, [u8]> {
    use std::os::unix::ffi::OsStrExt;
    Cow::Borrowed(name.as_bytes())
}

#[cfg(not(unix))]
fn os_bytes(name: &OsStr) -> Cow<'_, [u8]> {
    match name.to_string_lossy() {
        Cow::Borrowed(text) => Cow::Borrowed(text.as_bytes()),
        Cow::Owned(text) => Cow::Owned(text.into_bytes()),
    }
}

/// Append `bytes`, writing everything but RFC 3986 unreserved characters as
/// `%XX` (upper-case hex).
fn push_encoded(out: &mut String, bytes: &[u8]) {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    for &byte in bytes {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            out.push(char::from(byte));
        } else {
            out.push('%');
            out.push(char::from(HEX[usize::from(byte >> 4)]));
            out.push(char::from(HEX[usize::from(byte & 0x0F)]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn url_path(anchor: Anchor<'static>, segments: &[&'static str]) -> UrlPath<'static> {
        UrlPath {
            anchor,
            segments: segments
                .iter()
                .map(|segment| Cow::Borrowed(segment.as_bytes()))
                .collect(),
        }
    }

    // --- pure formatting, every platform ---

    #[test]
    fn only_unreserved_bytes_stay_literal() {
        let mut out = String::new();
        push_encoded(
            &mut out,
            b"AZaz09-._~ !\"#$%&'()*+,/:;<=>?@[\\]^`{|}\x00\x7f\xc3\xa9\xff",
        );
        assert_eq!(
            out,
            "AZaz09-._~%20%21%22%23%24%25%26%27%28%29%2A%2B%2C%2F%3A%3B%3C%3D%3E%3F%40\
             %5B%5C%5D%5E%60%7B%7C%7D%00%7F%C3%A9%FF"
        );
    }

    #[test]
    fn posix_paths_percent_encode_every_name() {
        let path = url_path(Anchor::Root, &["tmp", "Take #1 50% é.mp4"]);
        assert_eq!(
            path.file_url(),
            "file:///tmp/Take%20%231%2050%25%20%C3%A9.mp4"
        );
        assert_eq!(
            path.xmeml_path_url(),
            "file://localhost///tmp/Take%20%231%2050%25%20%C3%A9.mp4"
        );
        assert_eq!(url_path(Anchor::Root, &[]).file_url(), "file:///");
    }

    #[test]
    fn drive_paths_put_the_drive_first_in_the_url_path() {
        let path = url_path(Anchor::Drive(b'C'), &["Users", "me", "a b.mp4"]);
        assert_eq!(path.file_url(), "file:///C:/Users/me/a%20b.mp4");
        assert_eq!(
            path.xmeml_path_url(),
            "file://localhost/C:/Users/me/a%20b.mp4"
        );
        assert_eq!(url_path(Anchor::Drive(b'D'), &[]).file_url(), "file:///D:/");
    }

    #[test]
    fn unc_paths_put_the_server_in_the_authority() {
        let path = UrlPath {
            anchor: Anchor::Unc {
                server: Cow::Borrowed(b"server"),
                share: Cow::Borrowed(b"my share"),
            },
            segments: vec![Cow::Borrowed(b"a#1.mp4")],
        };
        assert_eq!(path.file_url(), "file://server/my%20share/a%231.mp4");
        assert_eq!(path.xmeml_path_url(), "file://server/my%20share/a%231.mp4");
    }

    #[test]
    fn unanchored_paths_become_relative_references() {
        let path = url_path(Anchor::Relative, &["media", "..", "a b.mp4"]);
        assert_eq!(path.file_url(), "media/../a%20b.mp4");
        assert_eq!(path.xmeml_path_url(), "media/../a%20b.mp4");
    }

    // --- the platform's own path parser ---

    #[test]
    fn rooted_paths_convert_on_every_platform() {
        // POSIX absolute, and drive-less rooted on Windows.
        assert_eq!(
            path_to_file_url(Path::new("/abs/./a b.mov")),
            "file:///abs/a%20b.mov"
        );
        assert_eq!(
            xmeml_path_url(Path::new("/abs/clip.mov")),
            "file://localhost///abs/clip.mov"
        );
        assert_eq!(
            path_to_file_url(Path::new("clips/a b.mp4")),
            "clips/a%20b.mp4"
        );
    }

    #[cfg(unix)]
    #[test]
    fn unix_names_are_encoded_from_their_raw_bytes() {
        use std::os::unix::ffi::OsStrExt;
        let path = Path::new(OsStr::from_bytes(b"/tmp/bad\xff\xfe name\\x.mp4"));
        assert_eq!(
            path_to_file_url(path),
            "file:///tmp/bad%FF%FE%20name%5Cx.mp4"
        );
    }

    #[cfg(unix)]
    #[test]
    fn unix_urls_round_trip_through_a_url_reader() {
        use std::os::unix::ffi::OsStrExt;
        let paths: [&[u8]; 5] = [
            "/tmp/Take #1 50% é & 中文.mp4".as_bytes(),
            b"/a/b?c#d/%41%zz;x=y&z@h:1",
            b"/q/sam's clip (1)!$*+,[x]^|`{}.mov",
            b"/tmp/bad\xff\xfe name.mp4",
            b"/",
        ];
        for raw in paths {
            let path = Path::new(OsStr::from_bytes(raw));
            for url in [path_to_file_url(path), xmeml_path_url(path)] {
                let parsed = url::Url::parse(&url).unwrap_or_else(|e| panic!("{url}: {e}"));
                assert_eq!(parsed.to_file_path().as_deref(), Ok(path), "{url}");
            }
        }
    }

    #[cfg(windows)]
    #[test]
    fn windows_paths_convert_to_rfc8089_urls() {
        let cases = [
            (r"C:\Users\me\a b.mp4", "file:///C:/Users/me/a%20b.mp4"),
            (
                r"C:\Take #1 50% é & 中文.mp4",
                "file:///C:/Take%20%231%2050%25%20%C3%A9%20%26%20%E4%B8%AD%E6%96%87.mp4",
            ),
            (r"\\?\C:\x.mp4", "file:///C:/x.mp4"),
            (r"\\server\share\a.mp4", "file://server/share/a.mp4"),
            (r"\\?\UNC\server\share\a.mp4", "file://server/share/a.mp4"),
        ];
        for (path, url) in cases {
            assert_eq!(path_to_file_url(Path::new(path)), url, "{path}");
        }
        assert_eq!(
            xmeml_path_url(Path::new(r"C:\Users\me\a b.mp4")),
            "file://localhost/C:/Users/me/a%20b.mp4"
        );
        assert_eq!(
            xmeml_path_url(Path::new(r"\\server\share\a.mp4")),
            "file://server/share/a.mp4"
        );
    }

    #[cfg(windows)]
    #[test]
    fn windows_urls_round_trip_through_a_url_reader() {
        for path in [
            r"C:\Users\me\Take #1 50% é & 中文.mp4",
            r"\\server\share\a b.mp4",
        ] {
            let path = Path::new(path);
            for url in [path_to_file_url(path), xmeml_path_url(path)] {
                let parsed = url::Url::parse(&url).unwrap_or_else(|e| panic!("{url}: {e}"));
                assert_eq!(parsed.to_file_path().as_deref(), Ok(path), "{url}");
            }
        }
    }
}
