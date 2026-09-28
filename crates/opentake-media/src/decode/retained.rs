//! Single-frame decode from a retained, already-open regular file.
//!
//! Strict project-cover capture holds pre-authorized handles so a source
//! cannot be swapped between authorization and decode; this path must never
//! reopen a pathname. The handle (a rewound clone of it) becomes ffmpeg's
//! stdin and is read through the `fd:` protocol, which keeps normal file seek
//! semantics for regular files. A pipe fed from the handle could not seek, so
//! every MP4/MOV whose `moov` index follows the media data failed and the
//! target could only be reached by decoding from the start of the file.
//!
//! ffmpeg-sidecar requires a piped stdin, so this path spawns ffmpeg itself,
//! contains it in a [`ProcessTree`], and parses its output directly: the frame
//! arrives as a self-describing PAM image on stdout, and the real pts of the
//! selected source frame comes from the `showinfo` lines on stderr, read with
//! a bounded line buffer.

use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::process::{Child, Command, Stdio};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use opentake_domain::MediaColorMetadata;

use super::frame::{retained_frame_args, showinfo_pts, target_micros, DisplayedPts, FrameRequest};
use crate::cancel::MediaCancelToken;
use crate::error::{MediaError, Result};
use crate::frame::RgbaFrame;
use crate::process_tree::{configure_command, ProcessTree};

const CHILD_POLL_INTERVAL: Duration = Duration::from_millis(5);
/// A PAM header is a handful of short `KEY value` lines.
const PAM_HEADER_LIMIT: usize = 1024;
/// Largest frame accepted from the decoder (a 16K RGBA frame is ~0.5 GiB).
const PAM_FRAME_LIMIT: usize = 1 << 30;
/// Longest stderr line retained for parsing; FFmpeg's own log lines are far
/// shorter, and the rest of an overlong line is discarded.
const LOG_LINE_LIMIT: usize = 16 * 1024;

pub(super) fn decode_retained_frame(
    file: &File,
    req: &FrameRequest,
    color: Option<&MediaColorMetadata>,
    cancel: &MediaCancelToken,
) -> Result<(f64, RgbaFrame)> {
    if cancel.is_cancelled() {
        return Err(MediaError::Cancelled);
    }
    let mut input = file.try_clone()?;
    input.seek(SeekFrom::Start(0))?;
    let mut command = Command::new(crate::ff::ffmpeg_path());
    command
        .args(retained_frame_args(req, color))
        .stdin(Stdio::from(input))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    configure_command(&mut command);
    let mut child = command
        .spawn()
        .map_err(|error| MediaError::Ffmpeg(format!("spawn: {error}")))?;
    crate::ff::record_helper_process();
    // Windows starts the configured child suspended until it is attached.
    let mut tree = match ProcessTree::attach(child.id()) {
        Ok(tree) => tree,
        Err(error) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(MediaError::Ffmpeg(format!("ffmpeg containment: {error}")));
        }
    };
    cancel.child_spawned();

    let (Some(stdout), Some(stderr)) = (child.stdout.take(), child.stderr.take()) else {
        terminate(&mut tree, &mut child);
        return Err(MediaError::Ffmpeg(
            "retained frame pipes missing".to_string(),
        ));
    };
    let target_us = target_micros(req);
    let stdout_cancel = cancel.clone();
    let frame_reader = match thread::Builder::new()
        .name("opentake-retained-frame".to_string())
        .spawn(move || {
            stdout_cancel.reader_started();
            let result = read_pam_frame(stdout);
            stdout_cancel.reader_finished();
            result
        }) {
        Ok(reader) => reader,
        Err(error) => {
            terminate(&mut tree, &mut child);
            return Err(MediaError::Ffmpeg(format!(
                "spawn retained frame reader: {error}"
            )));
        }
    };
    let stderr_cancel = cancel.clone();
    let log_reader = match thread::Builder::new()
        .name("opentake-retained-frame-log".to_string())
        .spawn(move || {
            stderr_cancel.reader_started();
            let result = read_displayed_pts(stderr, target_us);
            stderr_cancel.reader_finished();
            result
        }) {
        Ok(reader) => reader,
        Err(error) => {
            terminate(&mut tree, &mut child);
            let _ = frame_reader.join();
            return Err(MediaError::Ffmpeg(format!(
                "spawn retained frame log reader: {error}"
            )));
        }
    };

    loop {
        if cancel.checkpoint() {
            terminate(&mut tree, &mut child);
            let _ = join(frame_reader, "retained frame");
            let _ = join(log_reader, "retained frame log");
            return Err(MediaError::Cancelled);
        }
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => thread::sleep(CHILD_POLL_INTERVAL),
            Err(error) => {
                terminate(&mut tree, &mut child);
                let _ = join(frame_reader, "retained frame");
                let _ = join(log_reader, "retained frame log");
                return Err(MediaError::Io(error));
            }
        }
    }
    // Close the whole tree before draining: nothing else may keep the
    // inherited pipes open after ffmpeg itself exited.
    terminate(&mut tree, &mut child);
    let frame = join(frame_reader, "retained frame");
    let selection = join(log_reader, "retained frame log");
    if cancel.is_cancelled() {
        return Err(MediaError::Cancelled);
    }
    let selection = selection?;
    frame?
        .map(|frame| (selection.secs(), frame))
        .ok_or_else(|| MediaError::Decode(format!("no frame at {:.3}s", req.time_secs)))
}

fn terminate(tree: &mut ProcessTree, child: &mut Child) {
    if tree.terminate().is_ok() {
        tree.disarm();
    }
    let _ = child.kill();
    let _ = child.wait();
}

fn join<T>(reader: JoinHandle<Result<T>>, name: &str) -> Result<T> {
    reader
        .join()
        .map_err(|_| MediaError::Ffmpeg(format!("{name} reader panicked")))?
}

fn read_error(error: std::io::Error) -> MediaError {
    MediaError::Ffmpeg(format!("read retained frame: {error}"))
}

/// Read the single PAM image ffmpeg writes, then drain stdout. `Ok(None)`
/// means ffmpeg wrote nothing (it selected no frame).
fn read_pam_frame(stdout: impl Read) -> Result<Option<RgbaFrame>> {
    let mut reader = BufReader::new(stdout);
    let mut header = Vec::new();
    loop {
        let mut line = Vec::new();
        let read =
            read_bounded_line(&mut reader, &mut line, PAM_HEADER_LIMIT).map_err(read_error)?;
        if read == 0 {
            if header.is_empty() {
                return Ok(None);
            }
            return Err(MediaError::Decode(
                "truncated PAM header from ffmpeg".to_string(),
            ));
        }
        header.extend_from_slice(&line);
        if header.len() > PAM_HEADER_LIMIT {
            return Err(MediaError::Decode(
                "oversized PAM header from ffmpeg".to_string(),
            ));
        }
        if line.trim_ascii() == b"ENDHDR" {
            break;
        }
    }
    let (width, height) = parse_pam_header(&header)?;
    let bytes = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(4))
        .filter(|bytes| *bytes <= PAM_FRAME_LIMIT)
        .ok_or_else(|| MediaError::Decode(format!("PAM frame {width}x{height} is too large")))?;
    let mut rgba = Vec::new();
    rgba.try_reserve_exact(bytes)
        .map_err(|error| MediaError::Decode(format!("frame allocation {bytes}: {error}")))?;
    rgba.resize(bytes, 0);
    reader.read_exact(&mut rgba).map_err(|error| {
        if error.kind() == std::io::ErrorKind::UnexpectedEof {
            MediaError::Decode("truncated PAM frame from ffmpeg".to_string())
        } else {
            read_error(error)
        }
    })?;
    std::io::copy(&mut reader, &mut std::io::sink()).map_err(read_error)?;
    Ok(Some(RgbaFrame::new(width, height, rgba)))
}

/// Dimensions of an 8-bit RGBA PAM header (`P7`, `DEPTH 4`, `MAXVAL 255`).
fn parse_pam_header(header: &[u8]) -> Result<(u32, u32)> {
    let invalid = |detail: &str| MediaError::Decode(format!("invalid PAM header: {detail}"));
    let text = std::str::from_utf8(header).map_err(|_| invalid("not ASCII"))?;
    let mut lines = text.lines().map(str::trim).filter(|line| !line.is_empty());
    if lines.next() != Some("P7") {
        return Err(invalid("missing P7 magic"));
    }
    let (mut width, mut height, mut depth, mut maxval) = (None, None, None, None);
    for line in lines {
        let mut fields = line.split_whitespace();
        let (Some(key), value) = (fields.next(), fields.next()) else {
            continue;
        };
        let number = || value.and_then(|value| value.parse::<u32>().ok());
        match key {
            "WIDTH" => width = number(),
            "HEIGHT" => height = number(),
            "DEPTH" => depth = number(),
            "MAXVAL" => maxval = number(),
            _ => {}
        }
    }
    match (width, height, depth, maxval) {
        (Some(width), Some(height), Some(4), Some(255)) if width > 0 && height > 0 => {
            Ok((width, height))
        }
        _ => Err(invalid("expected a non-empty 8-bit RGBA image")),
    }
}

/// Follow ffmpeg's log for the `showinfo` pts lines that identify the frame
/// selected for `target_us`.
fn read_displayed_pts(stderr: impl Read, target_us: i64) -> Result<DisplayedPts> {
    let mut reader = BufReader::new(stderr);
    let mut selection = DisplayedPts::new(target_us);
    let mut line = Vec::new();
    loop {
        line.clear();
        let read = read_bounded_line(&mut reader, &mut line, LOG_LINE_LIMIT).map_err(read_error)?;
        if read == 0 {
            return Ok(selection);
        }
        if let Some(pts) = showinfo_pts(&String::from_utf8_lossy(&line)) {
            selection.observe(pts);
        }
    }
}

/// Consume one `\n`-terminated line (or the rest of the stream), keeping at
/// most `limit` bytes of it in `line`. Returns the bytes consumed; 0 at EOF.
fn read_bounded_line(
    reader: &mut impl BufRead,
    line: &mut Vec<u8>,
    limit: usize,
) -> std::io::Result<usize> {
    let mut consumed = 0;
    loop {
        let available = match reader.fill_buf() {
            Ok(available) => available,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };
        if available.is_empty() {
            return Ok(consumed);
        }
        let (length, complete) = match available.iter().position(|byte| *byte == b'\n') {
            Some(newline) => (newline + 1, true),
            None => (available.len(), false),
        };
        let keep = limit.saturating_sub(line.len()).min(length);
        line.extend_from_slice(&available[..keep]);
        reader.consume(length);
        consumed += length;
        if complete {
            return Ok(consumed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pam(width: u32, height: u32, pixel: [u8; 4]) -> Vec<u8> {
        let mut bytes = format!(
            "P7\nWIDTH {width}\nHEIGHT {height}\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n"
        )
        .into_bytes();
        for _ in 0..width * height {
            bytes.extend_from_slice(&pixel);
        }
        bytes
    }

    #[test]
    fn pam_frame_is_parsed_and_the_stream_drained() {
        let mut bytes = pam(3, 2, [1, 2, 3, 4]);
        bytes.extend_from_slice(b"trailing bytes are drained");
        let frame = read_pam_frame(bytes.as_slice()).unwrap().unwrap();
        assert_eq!((frame.width, frame.height), (3, 2));
        assert_eq!(frame.rgba, [1, 2, 3, 4].repeat(6));
    }

    #[test]
    fn empty_output_is_no_frame_but_damaged_output_is_an_error() {
        assert!(read_pam_frame(&b""[..]).unwrap().is_none());
        let complete = pam(2, 2, [9; 4]);
        let truncated = &complete[..complete.len() - 1];
        assert!(matches!(
            read_pam_frame(truncated),
            Err(MediaError::Decode(message)) if message.contains("truncated PAM frame")
        ));
        assert!(read_pam_frame(&b"P7\nWIDTH 2\n"[..]).is_err());
        let gray = b"P7\nWIDTH 1\nHEIGHT 1\nDEPTH 1\nMAXVAL 255\nENDHDR\n\x00";
        assert!(read_pam_frame(&gray[..]).is_err());
        let huge = b"P7\nWIDTH 100000\nHEIGHT 100000\nDEPTH 4\nMAXVAL 255\nENDHDR\n";
        assert!(read_pam_frame(&huge[..]).is_err());
        let endless = vec![b'x'; 4 * PAM_HEADER_LIMIT];
        assert!(read_pam_frame(endless.as_slice()).is_err());
    }

    #[test]
    fn log_selection_reads_bounded_lines() {
        let mut log = Vec::new();
        log.extend_from_slice(b"[Parsed_showinfo_1 @ 0x1] n:   0 pts:900000 pts_time:0.9\n");
        log.extend(std::iter::repeat_n(b'z', 3 * LOG_LINE_LIMIT));
        log.extend_from_slice(b"\n[Parsed_showinfo_1 @ 0x1] n:   1 pts:1033333 pts_time:1\n");
        let selection = read_displayed_pts(log.as_slice(), 1_000_000).unwrap();
        assert!((selection.secs() - 0.9).abs() < 1e-9);

        let mut reader = BufReader::with_capacity(8, &b"0123456789\nrest"[..]);
        let mut line = Vec::new();
        assert_eq!(read_bounded_line(&mut reader, &mut line, 4).unwrap(), 11);
        assert_eq!(line, b"0123");
        line.clear();
        assert_eq!(read_bounded_line(&mut reader, &mut line, 64).unwrap(), 4);
        assert_eq!(line, b"rest");
        line.clear();
        assert_eq!(read_bounded_line(&mut reader, &mut line, 64).unwrap(), 0);
    }

    /// Top-level MP4 box types in file order.
    fn top_level_boxes(bytes: &[u8]) -> Vec<String> {
        let mut boxes = Vec::new();
        let mut offset = 0;
        while offset + 8 <= bytes.len() {
            let size = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
            boxes.push(String::from_utf8_lossy(&bytes[offset + 4..offset + 8]).into_owned());
            if size < 8 {
                break;
            }
            offset += size;
        }
        boxes
    }

    #[test]
    fn retained_handle_decodes_an_index_at_end_mp4_like_its_path() {
        assert!(
            crate::ff::ffmpeg_available(),
            "retained decode test needs a runnable FFmpeg"
        );
        let temp = tempfile::tempdir().unwrap();
        let clip = temp.path().join("moov-at-end.mp4");
        let generated = std::process::Command::new(crate::ff::ffmpeg_path())
            .args(["-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi"])
            .args(["-i", "testsrc2=size=320x180:rate=30", "-t", "2"])
            .args(["-c:v", "mpeg4", "-q:v", "1"])
            .arg(&clip)
            .status()
            .unwrap();
        assert!(generated.success(), "generate moov-at-end fixture");
        let boxes = top_level_boxes(&std::fs::read(&clip).unwrap());
        let mdat = boxes.iter().position(|kind| kind == "mdat").unwrap();
        let moov = boxes.iter().position(|kind| kind == "moov").unwrap();
        assert!(
            mdat < moov,
            "fixture must keep its index at the end: {boxes:?}"
        );

        let request = FrameRequest {
            time_secs: 1.0,
            ..FrameRequest::default()
        };
        let cancel = MediaCancelToken::new();
        let mut file = File::open(&clip).unwrap();
        // The handle's position must not matter.
        file.seek(SeekFrom::End(0)).unwrap();
        let (retained_time, retained) =
            super::super::decode_frame_file_at_cancellable(&file, &request, &cancel)
                .expect("retained handle decodes an index-at-end MP4");
        let (path_time, by_path) =
            super::super::decode_frame_at_cancellable(&clip, &request, &cancel).unwrap();
        assert_eq!((retained.width, retained.height), (320, 180));
        assert_eq!(retained, by_path, "same pixels as the pathname decode");
        assert!((retained_time - path_time).abs() < 1e-9);
        assert!((retained_time - 1.0).abs() < 1e-6, "{retained_time}");
    }

    #[cfg(unix)]
    #[test]
    fn cancelling_a_retained_decode_kills_the_whole_ffmpeg_tree() {
        use std::os::unix::fs::PermissionsExt;
        use std::sync::mpsc;
        use std::time::Instant;

        let temp = tempfile::tempdir().unwrap();
        let pids = temp.path().join("pids");
        let script = temp.path().join("stuck-ffmpeg");
        std::fs::write(
            &script,
            format!(
                "#!/bin/sh\nsleep 60 &\nprintf 'child=%s\\nparent=%s\\n' \"$!\" \"$$\" > '{}'\nexec sleep 60\n",
                pids.display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).unwrap();
        let source = temp.path().join("source.mp4");
        std::fs::write(&source, b"regular file handed to the stuck decoder").unwrap();
        let file = File::open(&source).unwrap();

        let cancel = MediaCancelToken::new();
        let worker_cancel = cancel.clone();
        let (done_tx, done_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            crate::ff::test_seams::override_ffmpeg(Some(script.into_os_string()));
            let result = super::super::decode_frame_file_at_with_color_cancellable(
                &file,
                &FrameRequest::default(),
                &crate::decode::ColorHint::Known(None),
                &worker_cancel,
            );
            done_tx.send(result).unwrap();
        });
        let deadline = Instant::now() + Duration::from_secs(10);
        while !(pids.exists() && std::fs::read_to_string(&pids).unwrap().lines().count() == 2)
            && Instant::now() < deadline
        {
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(cancel.spawned_child_count(), 1, "decoder spawned");
        cancel.cancel();
        let result = done_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("cancelled retained decode returns promptly");
        worker.join().unwrap();
        assert!(matches!(result, Err(MediaError::Cancelled)), "{result:?}");
        assert_eq!(cancel.active_reader_count(), 0, "readers joined");

        let alive = |pid: &str| {
            Command::new("kill")
                .args(["-0", pid])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .unwrap()
                .success()
        };
        for pid in std::fs::read_to_string(&pids)
            .unwrap()
            .lines()
            .filter_map(|line| line.split_once('=').map(|(_, pid)| pid.to_string()))
        {
            // A killed background member may linger as a zombie until init
            // reaps it; allow a bounded grace period.
            let exit_deadline = Instant::now() + Duration::from_secs(5);
            while alive(&pid) && Instant::now() < exit_deadline {
                thread::sleep(Duration::from_millis(20));
            }
            assert!(
                !alive(&pid),
                "ffmpeg tree member {pid} survived cancellation"
            );
        }
    }
}
