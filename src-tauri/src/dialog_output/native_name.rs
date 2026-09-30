//! Editable Unix filenames for GTK's Unicode text fields. Invalid bytes and
//! literal backslashes have distinct spellings; the chooser explains them.

use std::ffi::{OsStr, OsString};
use std::os::unix::ffi::{OsStrExt, OsStringExt};

const HEX: &[u8; 16] = b"0123456789abcdef";

pub(super) fn display(name: &OsStr) -> String {
    let mut text = String::new();
    for chunk in name.as_bytes().utf8_chunks() {
        for character in chunk.valid().chars() {
            if character == '\\' {
                text.push_str("\\\\");
            } else if character.is_control() {
                let mut encoded = [0; 4];
                for &byte in character.encode_utf8(&mut encoded).as_bytes() {
                    escape_byte(&mut text, byte);
                }
            } else {
                text.push(character);
            }
        }
        for &byte in chunk.invalid() {
            escape_byte(&mut text, byte);
        }
    }
    text
}

fn escape_byte(text: &mut String, byte: u8) {
    text.push_str("\\x");
    text.push(HEX[(byte >> 4) as usize] as char);
    text.push(HEX[(byte & 15) as usize] as char);
}

pub(super) fn parse(text: &str) -> Result<OsString, String> {
    let mut input = text.bytes();
    let mut bytes = Vec::with_capacity(text.len());
    while let Some(byte) = input.next() {
        if byte != b'\\' {
            bytes.push(byte);
            continue;
        }
        match input.next() {
            Some(b'\\') => bytes.push(b'\\'),
            Some(b'x') => {
                let high = input.next().and_then(|byte| (byte as char).to_digit(16));
                let low = input.next().and_then(|byte| (byte as char).to_digit(16));
                match (high, low) {
                    (Some(high), Some(low)) => bytes.push((high * 16 + low) as u8),
                    _ => return Err("filename byte escapes must use two hex digits".into()),
                }
            }
            _ => return Err("use \\xNN for filename bytes or \\\\ for a backslash".into()),
        }
    }
    if bytes.is_empty()
        || bytes == b"."
        || bytes == b".."
        || bytes.iter().any(|byte| matches!(byte, 0 | b'/'))
    {
        return Err("enter a filename without a directory separator or NUL".into());
    }
    Ok(OsString::from_vec(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn editable_names_preserve_native_bytes_and_literal_escapes() {
        for bytes in [
            b"clip-\xff.mp4".as_slice(),
            b"literal-\\xff.mp4",
            b"line\nbreak.mp4",
            b"incomplete-\xe2\x82",
            "片段 e\u{301}.mp4".as_bytes(),
        ] {
            let name = OsStr::from_bytes(bytes);
            assert_eq!(parse(&display(name)).unwrap(), name);
        }
        assert_ne!(
            display(OsStr::from_bytes(b"clip-\xff.mp4")),
            display(OsStr::new("clip-\\xff.mp4"))
        );
        assert_ne!(
            parse("clip-\\xff.mp4").unwrap(),
            parse("clip-�.mp4").unwrap()
        );
        assert_eq!(display(OsStr::new("片段.mp4")), "片段.mp4");
    }

    #[test]
    fn editable_names_reject_ambiguous_escapes_and_path_traversal() {
        for text in [
            "", ".", "..", "dir/file", "\\x2f", "\\x00", "\\", "\\n", "\\xG0", "\\x0",
        ] {
            assert!(parse(text).is_err(), "{text:?}");
        }
        assert_eq!(
            parse("clip-\\xFF.mp4").unwrap().as_bytes(),
            b"clip-\xff.mp4"
        );
    }
}
