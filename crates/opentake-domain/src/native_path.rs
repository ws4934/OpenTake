//! Lossless filesystem paths at JSON boundaries. Unicode paths retain their
//! existing string representation; other paths use a versioned platform tag.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::path::{Path, PathBuf};

const PREFIX: &str = "opentake-path-v1:";
const MAX_WIRE_BYTES: usize = 256 * 1024;
const HEX: &[u8; 16] = b"0123456789abcdef";

/// Decode an IPC path, requiring this platform's native representation.
pub fn decode(text: &str) -> Result<PathBuf, &'static str> {
    NativePath::from_wire(text)?.into_path_buf()
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NativePath(Representation);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Representation {
    Local(PathBuf),
    Foreign(String),
}

impl Default for NativePath {
    fn default() -> Self {
        Self::new(PathBuf::new())
    }
}

impl NativePath {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self(Representation::Local(path.into()))
    }

    /// A foreign platform's native filename stays in the manifest as offline
    /// media, retaining the original encoding for relinking or another platform.
    pub fn as_path(&self) -> Option<&Path> {
        match &self.0 {
            Representation::Local(path) => Some(path),
            Representation::Foreign(_) => None,
        }
    }

    pub fn local_path(&self) -> Result<&Path, &'static str> {
        self.as_path()
            .ok_or("native path encoding is unavailable on this platform")
    }

    pub fn into_path_buf(self) -> Result<PathBuf, &'static str> {
        match self.0 {
            Representation::Local(path) => Ok(path),
            Representation::Foreign(_) => {
                Err("native path encoding is unavailable on this platform")
            }
        }
    }

    pub fn to_wire(&self) -> String {
        let path = match &self.0 {
            Representation::Local(path) => path,
            Representation::Foreign(encoded) => return encoded.clone(),
        };
        if let Some(text) = path.to_str().filter(|text| !text.starts_with(PREFIX)) {
            return text.to_owned();
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            let bytes = path.as_os_str().as_bytes();
            let mut text = String::with_capacity(PREFIX.len() + 5 + bytes.len() * 2);
            text.push_str(PREFIX);
            text.push_str("unix:");
            for &byte in bytes {
                text.push(HEX[(byte >> 4) as usize] as char);
                text.push(HEX[(byte & 15) as usize] as char);
            }
            text
        }
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStrExt;
            let mut text = format!("{PREFIX}windows:");
            for unit in path.as_os_str().encode_wide() {
                for shift in [12, 8, 4, 0] {
                    text.push(HEX[((unit >> shift) & 15) as usize] as char);
                }
            }
            text
        }
    }

    pub fn from_wire(text: &str) -> Result<Self, &'static str> {
        if text.len() > MAX_WIRE_BYTES || text.contains('\0') {
            return Err("native path length or contents are invalid");
        }
        let Some(encoded) = text.strip_prefix(PREFIX) else {
            return Ok(Self::new(text));
        };
        if let Some(hex) = encoded.strip_prefix("unix:") {
            let units = decode_hex(hex, 2)?;
            #[cfg(unix)]
            {
                use std::os::unix::ffi::OsStringExt;
                return Ok(Self::new(std::ffi::OsString::from_vec(
                    units.into_iter().map(|unit| unit as u8).collect(),
                )));
            }
            #[cfg(not(unix))]
            {
                let _ = units;
                return Ok(Self(Representation::Foreign(text.to_owned())));
            }
        }
        if let Some(hex) = encoded.strip_prefix("windows:") {
            let units = decode_hex(hex, 4)?;
            #[cfg(windows)]
            {
                use std::os::windows::ffi::OsStringExt;
                return Ok(Self::new(std::ffi::OsString::from_wide(&units)));
            }
            #[cfg(not(windows))]
            {
                let _ = units;
                return Ok(Self(Representation::Foreign(text.to_owned())));
            }
        }
        Err("unknown native path encoding")
    }
}

fn decode_hex(text: &str, width: usize) -> Result<Vec<u16>, &'static str> {
    if text.is_empty() || !text.len().is_multiple_of(width) || !text.is_ascii() {
        return Err("native path encoding is malformed");
    }
    let units = text
        .as_bytes()
        .chunks_exact(width)
        .map(|chunk| {
            let text =
                std::str::from_utf8(chunk).map_err(|_| "native path encoding is malformed")?;
            let unit =
                u16::from_str_radix(text, 16).map_err(|_| "native path encoding is malformed")?;
            if unit == 0 {
                return Err("native paths cannot contain NUL");
            }
            Ok(unit)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(units)
}

impl Serialize for NativePath {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let text = self.to_wire();
        if text.len() > MAX_WIRE_BYTES
            || self
                .as_path()
                .is_some_and(|path| path.as_os_str().as_encoded_bytes().contains(&0))
        {
            return Err(serde::ser::Error::custom(
                "native path length or contents are invalid",
            ));
        }
        serializer.serialize_str(&text)
    }
}

impl<'de> Deserialize<'de> for NativePath {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        Self::from_wire(&text).map_err(serde::de::Error::custom)
    }
}

impl std::fmt::Display for NativePath {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            Representation::Local(path) => match path.to_str() {
                Some(text) => formatter.write_str(text),
                None => write!(formatter, "{:?}", path.as_os_str()),
            },
            Representation::Foreign(encoded) => formatter.write_str(encoded),
        }
    }
}
impl From<PathBuf> for NativePath {
    fn from(path: PathBuf) -> Self {
        Self::new(path)
    }
}
impl From<&Path> for NativePath {
    fn from(path: &Path) -> Self {
        Self::new(path)
    }
}
impl From<String> for NativePath {
    fn from(path: String) -> Self {
        Self::new(path)
    }
}
impl From<&str> for NativePath {
    fn from(path: &str) -> Self {
        Self::new(path)
    }
}

/// Lexical identity without replacing native bytes or UTF-16 units. Matches
/// the desktop authority's ASCII case-insensitive rule on Windows.
pub fn normalize(path: &Path) -> PathBuf {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::{OsStrExt, OsStringExt};
        let normalized: PathBuf = path.components().collect();
        let mut units: Vec<u16> = normalized.as_os_str().encode_wide().collect();
        if units.starts_with(&[92, 92, 63, 92])
            && matches!(path.components().next(), Some(std::path::Component::Prefix(prefix)) if matches!(prefix.kind(), std::path::Prefix::VerbatimDisk(_)))
        {
            units.drain(..4);
        }
        PathBuf::from(std::ffi::OsString::from_wide(&units))
    }
    #[cfg(not(windows))]
    {
        path.components().collect()
    }
}

pub fn identity_key(path: &Path) -> PathBuf {
    let path = normalize(path);
    #[cfg(windows)]
    {
        use std::os::windows::ffi::{OsStrExt, OsStringExt};
        let units: Vec<_> = path
            .as_os_str()
            .encode_wide()
            .map(|unit| {
                if (65..=90).contains(&unit) {
                    unit + 32
                } else {
                    unit
                }
            })
            .collect();
        PathBuf::from(std::ffi::OsString::from_wide(&units))
    }
    #[cfg(not(windows))]
    {
        path
    }
}

/// Serde adapter for fields that remain native `PathBuf`s inside the process.
pub mod path {
    use super::*;

    pub fn serialize<S: Serializer>(path: &Path, serializer: S) -> Result<S::Ok, S::Error> {
        NativePath::from(path).serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<PathBuf, D::Error> {
        NativePath::deserialize(deserializer)?
            .into_path_buf()
            .map_err(serde::de::Error::custom)
    }
}

/// Serde adapter for optional native filesystem paths.
pub mod optional_path {
    use super::*;

    pub fn serialize<S: Serializer>(
        path: &Option<PathBuf>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        path.as_deref().map(NativePath::from).serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<PathBuf>, D::Error> {
        Option::<NativePath>::deserialize(deserializer).and_then(|path| {
            path.map(NativePath::into_path_buf)
                .transpose()
                .map_err(serde::de::Error::custom)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn foreign_native_paths_remain_roundtrippable_without_becoming_local_names() {
        let encoded = if cfg!(windows) {
            "opentake-path-v1:unix:2f78ff"
        } else {
            "opentake-path-v1:windows:0043003a005cd800"
        };
        let path = NativePath::from_wire(encoded).unwrap();
        assert!(path.as_path().is_none());
        assert!(decode(encoded).is_err());
        let json = serde_json::to_string(&path).unwrap();
        assert_eq!(
            serde_json::from_str::<NativePath>(&json).unwrap().to_wire(),
            encoded
        );
    }

    #[test]
    fn unicode_paths_keep_the_existing_wire_representation() {
        let path = NativePath::new("/media/片段 🐎.mp4");
        assert_eq!(path.to_wire(), "/media/片段 🐎.mp4");
        let json = serde_json::to_string(&path).unwrap();
        assert_eq!(serde_json::from_str::<NativePath>(&json).unwrap(), path);
    }

    #[test]
    fn malformed_native_paths_are_rejected() {
        for value in [
            "opentake-path-v1:unix:",
            "opentake-path-v1:unix:0",
            "opentake-path-v1:unix:xx",
            "opentake-path-v1:unix:00",
            "opentake-path-v1:windows:0000",
            "opentake-path-v1:other:ff",
            "bad\0path",
        ] {
            assert!(NativePath::from_wire(value).is_err(), "{value:?}");
        }
        assert!(NativePath::from_wire(&"x".repeat(MAX_WIRE_BYTES + 1)).is_err());
    }

    #[test]
    fn a_literal_reserved_prefix_does_not_alias_an_encoded_path() {
        let path = NativePath::new("opentake-path-v1:unix:ff");
        assert_ne!(path.to_wire(), "opentake-path-v1:unix:ff");
        assert_eq!(NativePath::from_wire(&path.to_wire()).unwrap(), path);
    }

    #[cfg(unix)]
    #[test]
    fn raw_unix_bytes_survive_json_and_do_not_alias_replacement_characters() {
        use std::os::unix::ffi::OsStrExt;
        let path = NativePath::new(
            Path::new("/media").join(std::ffi::OsStr::from_bytes(b"clip-\xff.mp4")),
        );
        let json = serde_json::to_string(&path).unwrap();
        let restored: NativePath = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, path);
        assert_ne!(
            restored,
            NativePath::new(path.local_path().unwrap().to_string_lossy().as_ref())
        );
    }

    #[cfg(windows)]
    #[test]
    fn raw_windows_units_survive_json() {
        use std::os::windows::ffi::OsStringExt;
        let path = NativePath::new(std::ffi::OsString::from_wide(&[
            67, 58, 92, 0xd800, 46, 109, 112, 52,
        ]));
        let json = serde_json::to_string(&path).unwrap();
        assert_eq!(serde_json::from_str::<NativePath>(&json).unwrap(), path);
    }
}
