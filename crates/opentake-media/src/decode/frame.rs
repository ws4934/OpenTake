//! Single/batch frame decode via the system ffmpeg CLI. Replaces upstream's
//! `AVAssetImageGenerator` (`MediaVisualCache`, `FrameSampler`, `MediaAsset`).
//!
//! `decode_frame_at` returns the frame on screen at a timestamp (the last frame
//! whose pts is `<= t`) as packed RGBA8, together with that frame's real pts.
//! `decode_frames_at` decodes a batch of ascending timestamps, de-duplicating
//! frames whose actual time does not advance (upstream's `t > lastTime` rule).
//!
//! The *scaling math* ([`fit_within`]) is a pure function and unit-tested; the
//! ffmpeg invocation requires the binary and is covered by ignore-by-default
//! integration tests.

use std::io::{Seek, SeekFrom};
use std::path::Path;
use std::thread;
use std::time::Duration;

use ffmpeg_sidecar::event::FfmpegEvent;
use image::ImageEncoder;
use rayon::prelude::*;

use crate::cancel::MediaCancelToken;
use crate::error::{MediaError, Result};
use crate::ff;
use crate::frame::RgbaFrame;

const FRAME_CHILD_POLL_INTERVAL: Duration = Duration::from_millis(5);

/// Source positions and blend factors within this distance of a whole frame
/// are that frame. Absorbs f64 error in `(n / target_fps) * source_fps`, so
/// equal rates never request a spurious interpolation.
const FRAME_POSITION_EPSILON: f64 = 1e-6;

/// A frame decode request.
#[derive(Clone, Debug)]
pub struct FrameRequest {
    /// Presentation time to sample. The decoded frame is the one displayed at
    /// this instant: the last frame with pts `<= time_secs`, or the first frame
    /// when the request precedes it.
    pub time_secs: f64,
    /// Upper bound box; the frame is scaled down to fit while preserving aspect
    /// ratio (never enlarged). `(0, 0)` disables scaling.
    pub max_size: (u32, u32),
    /// Apply container rotation (display matrix). Default true.
    pub apply_rotation: bool,
}

impl Default for FrameRequest {
    fn default() -> Self {
        FrameRequest {
            time_secs: 0.0,
            max_size: (0, 0),
            apply_rotation: true,
        }
    }
}

/// Source-frame reconstruction policy used when a timeline requests frames at
/// a different rate than the decoded asset. Optical flow is a deterministic
/// local motion-compensated path; it never implies a cloud/model dependency.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameInterpolationMode {
    Nearest,
    Blend,
    OpticalFlow,
}

/// Explicit recovery behavior when optical flow is unavailable on the current
/// device/runtime. The caller chooses quality, determinism, or fail-closed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameInterpolationFallback {
    Nearest,
    Blend,
    Error,
}

/// One target-rate sample mapped back into the source-frame interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameRateSample {
    pub timestamp_secs: f64,
    pub source_frame: u64,
    pub next_source_frame: u64,
    pub source_alpha: f64,
}

/// Result of one pair interpolation, including the effective mode after an
/// explicit unsupported-device fallback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameInterpolationResult {
    pub frame: RgbaFrame,
    pub mode_used: FrameInterpolationMode,
}

/// Map a finite source sequence onto a target frame rate while preserving both
/// endpoint timestamps exactly. Interior timestamps follow the target-rate
/// grid; the final sample is pinned to the source's final presentation time.
pub fn convert_frame_rate(
    source_frame_count: u64,
    source_fps: f64,
    target_fps: f64,
) -> Result<Vec<FrameRateSample>> {
    if source_frame_count == 0 {
        return Err(MediaError::Decode(
            "source_frame_count must be greater than zero".to_string(),
        ));
    }
    if !source_fps.is_finite() || source_fps <= 0.0 {
        return Err(MediaError::Decode(
            "source_fps must be finite and greater than zero".to_string(),
        ));
    }
    if !target_fps.is_finite() || target_fps <= 0.0 {
        return Err(MediaError::Decode(
            "target_fps must be finite and greater than zero".to_string(),
        ));
    }
    if source_frame_count == 1 {
        return Ok(vec![FrameRateSample {
            timestamp_secs: 0.0,
            source_frame: 0,
            next_source_frame: 0,
            source_alpha: 0.0,
        }]);
    }

    let source_last = source_frame_count - 1;
    let duration_secs = source_last as f64 / source_fps;
    let target_intervals = (duration_secs * target_fps).round().max(1.0) as u64;
    let mut samples = Vec::with_capacity(target_intervals as usize + 1);
    for output_frame in 0..=target_intervals {
        let timestamp_secs = if output_frame == target_intervals {
            duration_secs
        } else {
            (output_frame as f64 / target_fps).min(duration_secs)
        };
        let source_position = (timestamp_secs * source_fps).clamp(0.0, source_last as f64);
        let source_frame = source_position.floor() as u64;
        let next_source_frame = source_frame.saturating_add(1).min(source_last);
        let source_alpha = if source_frame == next_source_frame {
            0.0
        } else {
            source_position - source_frame as f64
        };
        samples.push(FrameRateSample {
            timestamp_secs,
            source_frame,
            next_source_frame,
            source_alpha,
        });
    }
    Ok(samples)
}

/// Source-frame endpoints `(first, next, alpha)` for project frame
/// `source_frame` on the `target_fps` timebase of an asset decoded at
/// `source_fps`. A position within [`FRAME_POSITION_EPSILON`] of a whole frame
/// snaps to it, giving `first == next` and `alpha == 0`.
pub fn source_frame_pair(source_frame: i64, target_fps: f64, source_fps: f64) -> (i64, i64, f64) {
    let timestamp = source_frame.max(0) as f64 / target_fps;
    let position = timestamp * source_fps;
    let whole = position.round();
    let position = if (position - whole).abs() < FRAME_POSITION_EPSILON {
        whole
    } else {
        position
    };
    let first = position.floor().max(0.0) as i64;
    let next = position.ceil().max(0.0) as i64;
    (first, next, position - first as f64)
}

/// Interpolate two equal-size RGBA frames at `alpha` in `[0, 1]`.
///
/// The optical-flow path estimates a deterministic local block-motion field,
/// warps both endpoints toward the requested instant, then blends the aligned
/// pixels. This traditional path is intentionally model-free and provides a
/// stable baseline for preview/export parity.
pub fn interpolate_frame_pair(
    first: &RgbaFrame,
    last: &RgbaFrame,
    alpha: f64,
    requested: FrameInterpolationMode,
    fallback: FrameInterpolationFallback,
    optical_flow_available: bool,
) -> Result<FrameInterpolationResult> {
    if first.width != last.width
        || first.height != last.height
        || first.rgba.len() != last.rgba.len()
    {
        return Err(MediaError::Decode(
            "interpolation frames must have identical dimensions".to_string(),
        ));
    }
    if !alpha.is_finite() {
        return Err(MediaError::Decode(
            "interpolation alpha must be finite".to_string(),
        ));
    }

    let mode_used = if requested == FrameInterpolationMode::OpticalFlow && !optical_flow_available {
        match fallback {
            FrameInterpolationFallback::Nearest => FrameInterpolationMode::Nearest,
            FrameInterpolationFallback::Blend => FrameInterpolationMode::Blend,
            FrameInterpolationFallback::Error => {
                return Err(MediaError::Decode(
                    "optical-flow interpolation is unavailable and fallback is Error".to_string(),
                ));
            }
        }
    } else {
        requested
    };

    let alpha = alpha.clamp(0.0, 1.0);
    let frame = if alpha <= FRAME_POSITION_EPSILON {
        first.clone()
    } else if alpha >= 1.0 - FRAME_POSITION_EPSILON {
        last.clone()
    } else {
        match mode_used {
            FrameInterpolationMode::Nearest => {
                if alpha < 0.5 {
                    first.clone()
                } else {
                    last.clone()
                }
            }
            FrameInterpolationMode::Blend => blend_frames(first, last, alpha),
            FrameInterpolationMode::OpticalFlow => optical_flow_frame(first, last, alpha),
        }
    };

    Ok(FrameInterpolationResult { frame, mode_used })
}

fn blend_frames(first: &RgbaFrame, last: &RgbaFrame, alpha: f64) -> RgbaFrame {
    let rgba = first
        .rgba
        .iter()
        .zip(&last.rgba)
        .map(|(&a, &b)| lerp_channel(a, b, alpha))
        .collect();
    RgbaFrame::new(first.width, first.height, rgba)
}

#[cfg(test)]
thread_local! {
    static OPTICAL_FLOW_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn optical_flow_frame(first: &RgbaFrame, last: &RgbaFrame, alpha: f64) -> RgbaFrame {
    #[cfg(test)]
    OPTICAL_FLOW_CALLS.with(|calls| calls.set(calls.get() + 1));
    let flow = estimate_block_motion(first, last);
    let mut rgba = vec![0; first.rgba.len()];
    // Rows are independent; each output byte is computed exactly as in a
    // sequential pass, so the result does not depend on scheduling.
    rgba.par_chunks_mut(first.width as usize * 4)
        .enumerate()
        .for_each(|(y, row)| {
            let y = y as u32;
            for x in 0..first.width {
                let (motion_x, motion_y) = flow.at(x, y);
                let from_first = sample_bilinear(
                    first,
                    x as f64 - alpha * motion_x,
                    y as f64 - alpha * motion_y,
                );
                let from_last = sample_bilinear(
                    last,
                    x as f64 + (1.0 - alpha) * motion_x,
                    y as f64 + (1.0 - alpha) * motion_y,
                );
                let offset = x as usize * 4;
                for channel in 0..4 {
                    row[offset + channel] =
                        lerp_channel(from_first[channel], from_last[channel], alpha);
                }
            }
        });
    RgbaFrame::new(first.width, first.height, rgba)
}

struct BlockMotionField {
    block_size: u32,
    columns: u32,
    rows: u32,
    vectors: Vec<(f64, f64)>,
}

impl BlockMotionField {
    fn at(&self, x: u32, y: u32) -> (f64, f64) {
        let column = (x / self.block_size).min(self.columns.saturating_sub(1));
        let row = (y / self.block_size).min(self.rows.saturating_sub(1));
        self.vectors[(row * self.columns + column) as usize]
    }
}

/// Estimate a deterministic local motion field with block matching. Bounded
/// search and per-block spatial sampling avoid the whole-frame distortion of a
/// single global translation vector without introducing a model dependency.
fn estimate_block_motion(first: &RgbaFrame, last: &RgbaFrame) -> BlockMotionField {
    let shortest = first.width.min(first.height).max(1);
    let block_size = shortest.min(32);
    let columns = first.width.div_ceil(block_size);
    let rows = first.height.div_ceil(block_size);
    let search_radius = (block_size / 2).clamp(1, 12) as i32;
    let sample_step = (block_size / 8).max(1);
    // Luma is computed once per pixel instead of once per candidate sample.
    let first_luma = luma_plane(first);
    let last_luma = luma_plane(last);
    let width = first.width as usize;

    // Blocks are independent and searched in the same candidate order as a
    // sequential pass, so the field does not depend on scheduling.
    let vectors = (0..columns * rows)
        .into_par_iter()
        .map(|block| {
            let start_x = (block % columns) * block_size;
            let start_y = (block / columns) * block_size;
            let end_x = (start_x + block_size).min(first.width);
            let end_y = (start_y + block_size).min(first.height);
            let mut best = (f64::INFINITY, i32::MAX, 0, 0);
            for dy in -search_radius..=search_radius {
                for dx in -search_radius..=search_radius {
                    let mut error = 0.0;
                    let mut samples = 0u32;
                    for y in (start_y..end_y).step_by(sample_step as usize) {
                        for x in (start_x..end_x).step_by(sample_step as usize) {
                            let target_x = x as i32 + dx;
                            let target_y = y as i32 + dy;
                            let target = if target_x < 0
                                || target_y < 0
                                || target_x >= last.width as i32
                                || target_y >= last.height as i32
                            {
                                255.0
                            } else {
                                last_luma[target_y as usize * width + target_x as usize]
                            };
                            error += (first_luma[y as usize * width + x as usize] - target).abs();
                            samples += 1;
                        }
                    }
                    let mean_error = error / samples.max(1) as f64;
                    let distance = dx * dx + dy * dy;
                    let candidate = (mean_error, distance, dy, dx);
                    if candidate < best {
                        best = candidate;
                    }
                }
            }
            (best.3 as f64, best.2 as f64)
        })
        .collect();

    BlockMotionField {
        block_size,
        columns,
        rows,
        vectors,
    }
}

fn luma_plane(frame: &RgbaFrame) -> Vec<f64> {
    frame
        .rgba
        .as_chunks::<4>()
        .0
        .iter()
        .map(|pixel| {
            let r = pixel[0] as f64;
            let g = pixel[1] as f64;
            let b = pixel[2] as f64;
            let a = pixel[3] as f64 / 255.0;
            (0.2126 * r + 0.7152 * g + 0.0722 * b) * a
        })
        .collect()
}

fn sample_bilinear(frame: &RgbaFrame, x: f64, y: f64) -> [u8; 4] {
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;
    let mut out = [0; 4];
    for (channel, value) in out.iter_mut().enumerate() {
        let p00 = sample_channel(frame, x0, y0, channel);
        let p10 = sample_channel(frame, x0 + 1, y0, channel);
        let p01 = sample_channel(frame, x0, y0 + 1, channel);
        let p11 = sample_channel(frame, x0 + 1, y0 + 1, channel);
        let top = p00 + (p10 - p00) * fx;
        let bottom = p01 + (p11 - p01) * fx;
        *value = (top + (bottom - top) * fy).round().clamp(0.0, 255.0) as u8;
    }
    out
}

fn sample_channel(frame: &RgbaFrame, x: i64, y: i64, channel: usize) -> f64 {
    if x < 0 || y < 0 || x >= frame.width as i64 || y >= frame.height as i64 {
        return if channel == 3 { 255.0 } else { 0.0 };
    }
    let offset = ((y as u32 * frame.width + x as u32) * 4) as usize;
    frame.rgba[offset + channel] as f64
}

fn lerp_channel(first: u8, last: u8, alpha: f64) -> u8 {
    (first as f64 + (last as f64 - first as f64) * alpha)
        .round()
        .clamp(0.0, 255.0) as u8
}

/// Scale `(w, h)` down to fit within `max` while preserving aspect ratio. Never
/// enlarges. A zero in either `max` dimension disables that bound. Mirrors
/// `AVAssetImageGenerator.maximumSize` semantics ("not larger than this box,
/// keep aspect ratio"). Output dimensions are at least 1.
pub fn fit_within(w: u32, h: u32, max: (u32, u32)) -> (u32, u32) {
    if w == 0 || h == 0 {
        return (w.max(1), h.max(1));
    }
    let (mw, mh) = max;
    let mut scale = 1.0f64;
    if mw > 0 {
        scale = scale.min(mw as f64 / w as f64);
    }
    if mh > 0 {
        scale = scale.min(mh as f64 / h as f64);
    }
    if scale >= 1.0 {
        return (w, h); // never enlarge
    }
    let nw = ((w as f64 * scale).round() as u32).max(1);
    let nh = ((h as f64 * scale).round() as u32).max(1);
    (nw, nh)
}

/// Build the ffmpeg arg list for decoding one frame to rawvideo RGBA on stdout.
/// Pure so the exact CLI contract is testable.
#[cfg(test)]
fn frame_args(path: &Path, req: &FrameRequest) -> Vec<String> {
    frame_args_with_color(path, req, None)
}

fn frame_args_with_color(
    path: &Path,
    req: &FrameRequest,
    color: Option<&opentake_domain::MediaColorMetadata>,
) -> Vec<String> {
    let time_secs = req.time_secs.max(0.0);
    let target_us = target_micros(req);
    let mut args: Vec<String> = Vec::new();
    if let Some(color) = color {
        args.extend(crate::color::hdr_decode_input_args(color));
    }
    // Keyframe seek at/before the target with source timestamps kept (relative
    // to the container start). Accurate seek is off because it drops every
    // frame before the target, including the one still on screen at it.
    args.extend(["-noaccurate_seek", "-copyts", "-start_at_zero", "-ss"].map(String::from));
    args.push(format!("{time_secs:.6}"));
    args.push("-i".into());
    args.push(path.to_string_lossy().into_owned());
    args.push("-frames:v".into());
    args.push("1".into());

    // Select the displayed frame before any conversion work: move timestamps
    // to microseconds (logged by showinfo so the real pts can be reported),
    // shift the target to 0, then `fps=1:start_time=0:round=up` emits the last
    // frame with pts <= 0 as its first output, padding with the first frame
    // when none precedes the target.
    let mut filters: Vec<String> = vec![
        format!("settb=1/{MICROS_PER_SEC}"),
        "showinfo=checksum=0".to_string(),
        format!("setpts=PTS-{target_us}"),
        "fps=fps=1:start_time=0:round=up".to_string(),
    ];
    if let Some(filter) = color.and_then(crate::color::hdr_tonemap_filter) {
        filters.push(filter);
    }
    if req.apply_rotation {
        // Honor the display matrix when transposing (ffmpeg applies it via the
        // autorotate behavior; the scale filter runs after rotation).
        // Nothing to add here — ffmpeg autorotates by default for the decoder.
    }
    if req.max_size.0 > 0 || req.max_size.1 > 0 {
        // Downscale-only, keep aspect: scale='min(iw,MW)':-2 style. We use
        // force_original_aspect_ratio=decrease against the box.
        let mw = if req.max_size.0 > 0 {
            req.max_size.0.to_string()
        } else {
            "iw".to_string()
        };
        let mh = if req.max_size.1 > 0 {
            req.max_size.1.to_string()
        } else {
            "ih".to_string()
        };
        filters.push(format!(
            "scale=w={mw}:h={mh}:force_original_aspect_ratio=decrease"
        ));
    }
    args.push("-vf".into());
    args.push(filters.join(","));
    args.push("-fps_mode".into());
    args.push("passthrough".into());
    args.push("-pix_fmt".into());
    args.push("rgba".into());
    args.push("-f".into());
    args.push("rawvideo".into());
    args.push("-".into());
    args
}

fn frame_args_for_input(
    input: &str,
    req: &FrameRequest,
    color: Option<&opentake_domain::MediaColorMetadata>,
) -> Vec<String> {
    let mut args = frame_args_with_color(Path::new(input), req, color);
    // A pipe cannot seek: decode from the start and let the display-frame
    // filters select the target.
    let seek_index = args
        .iter()
        .position(|argument| argument == "-ss")
        .expect("frame args always contain seek");
    args.drain(seek_index..seek_index + 2);
    args
}

const MICROS_PER_SEC: i64 = 1_000_000;

/// Microsecond pts from a `showinfo` frame line (`... n:   3 pts:1200000 ...`).
fn showinfo_pts(line: &str) -> Option<i64> {
    if !line.contains("Parsed_showinfo_") {
        return None;
    }
    let rest = line.split_once(" pts:")?.1.trim_start();
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || c == '-'))
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Read the first output frame and report the real pts of the source frame the
/// display-frame filters selected: the last logged pts `<= target_us`, else the
/// first one after it. Log lines are ordered, so once a frame after the target
/// has been logged the selection is final.
fn read_displayed_frame(
    events: impl IntoIterator<Item = FfmpegEvent>,
    target_us: i64,
) -> Option<(f64, RgbaFrame)> {
    let mut frame = None;
    let mut at_or_before: Option<i64> = None;
    let mut after: Option<i64> = None;
    for event in events {
        match event {
            FfmpegEvent::OutputFrame(output)
                if frame.is_none() && output.width > 0 && output.height > 0 =>
            {
                frame = Some(RgbaFrame::new(output.width, output.height, output.data));
            }
            FfmpegEvent::Log(_, line) => match showinfo_pts(&line) {
                Some(pts) if pts <= target_us => {
                    at_or_before = Some(at_or_before.map_or(pts, |seen| seen.max(pts)));
                }
                Some(pts) => after = Some(after.map_or(pts, |seen| seen.min(pts))),
                None => {}
            },
            _ => {}
        }
        if frame.is_some() && after.is_some() {
            break;
        }
    }
    let pts = at_or_before.or(after).unwrap_or(target_us);
    frame.map(|frame| (pts as f64 / MICROS_PER_SEC as f64, frame))
}

fn target_micros(req: &FrameRequest) -> i64 {
    (req.time_secs.max(0.0) * MICROS_PER_SEC as f64).round() as i64
}

/// Decode the frame displayed at `req.time_secs`, returning `(actual_secs,
/// frame)` where `actual_secs` is that source frame's presentation time.
pub fn decode_frame_at(path: &Path, req: &FrameRequest) -> Result<(f64, RgbaFrame)> {
    decode_frame_at_cancellable(path, req, &MediaCancelToken::new())
}

pub fn decode_frame_at_cancellable(
    path: &Path,
    req: &FrameRequest,
    cancel: &MediaCancelToken,
) -> Result<(f64, RgbaFrame)> {
    if cancel.is_cancelled() {
        return Err(MediaError::Cancelled);
    }
    // Probe color only for ordinary files. FIFOs/device inputs are valid FFmpeg
    // sources too; opening them once for ffprobe would consume or block the
    // stream before the actual cancellable decoder child is spawned.
    let color = path
        .metadata()
        .ok()
        .filter(|metadata| metadata.is_file())
        .and_then(|_| crate::probe::probe(path).ok())
        .and_then(|probe| probe.color);
    let mut child = ff::ffmpeg()
        .args(frame_args_with_color(path, req, color.as_ref()))
        .spawn()
        .map_err(|e| MediaError::Ffmpeg(format!("spawn: {e}")))?;
    cancel.child_spawned();

    let iter = match child.iter() {
        Ok(iter) => iter,
        Err(error) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(MediaError::Ffmpeg(format!("iter: {error}")));
        }
    };
    let reader_cancel = cancel.clone();
    let target_us = target_micros(req);
    let reader = match thread::Builder::new()
        .name("opentake-frame-events".to_string())
        .spawn(move || {
            reader_cancel.reader_started();
            let result = read_displayed_frame(iter, target_us);
            reader_cancel.reader_finished();
            result
        }) {
        Ok(reader) => reader,
        Err(error) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(MediaError::Ffmpeg(format!(
                "spawn frame event reader: {error}"
            )));
        }
    };

    loop {
        if cancel.checkpoint() {
            let _ = child.kill();
            let _ = child.wait();
            let _ = reader.join();
            return Err(MediaError::Cancelled);
        }
        if reader.is_finished() {
            // The reader stops once the frame is selected. Explicitly terminate
            // the single-frame command before joining so no producer remains
            // blocked trying to publish a later event into a dropped receiver.
            let _ = child.kill();
            let _ = child.wait();
            let result = reader
                .join()
                .map_err(|_| MediaError::Ffmpeg("frame event reader panicked".to_string()))?;
            if cancel.is_cancelled() {
                return Err(MediaError::Cancelled);
            }
            return result
                .ok_or_else(|| MediaError::Decode(format!("no frame at {:.3}s", req.time_secs)));
        }
        match child.as_inner_mut().try_wait() {
            Ok(Some(_)) => {
                let result = reader
                    .join()
                    .map_err(|_| MediaError::Ffmpeg("frame event reader panicked".to_string()))?;
                if cancel.is_cancelled() {
                    return Err(MediaError::Cancelled);
                }
                return result.ok_or_else(|| {
                    MediaError::Decode(format!("no frame at {:.3}s", req.time_secs))
                });
            }
            Ok(None) => thread::sleep(FRAME_CHILD_POLL_INTERVAL),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = reader.join();
                return Err(MediaError::Io(error));
            }
        }
    }
}

/// Decode from an already-open regular file. The retained handle is cloned,
/// rewound, and becomes ffmpeg's stdin (`fd:`); no pathname fallback occurs.
pub fn decode_frame_file_at_cancellable(
    file: &std::fs::File,
    req: &FrameRequest,
    cancel: &MediaCancelToken,
) -> Result<(f64, RgbaFrame)> {
    if cancel.is_cancelled() {
        return Err(MediaError::Cancelled);
    }
    let color = crate::probe::probe_file(file)
        .ok()
        .and_then(|probe| probe.color);
    let mut input = file.try_clone()?;
    input.seek(SeekFrom::Start(0))?;
    let mut child = ff::ffmpeg()
        .args(frame_args_for_input("fd:", req, color.as_ref()))
        .spawn()
        .map_err(|error| MediaError::Ffmpeg(format!("spawn: {error}")))?;
    cancel.child_spawned();
    let mut stdin = child
        .take_stdin()
        .ok_or_else(|| MediaError::Ffmpeg("retained frame stdin missing".to_string()))?;
    let feeder = thread::Builder::new()
        .name("opentake-retained-frame-input".to_string())
        .spawn(move || std::io::copy(&mut input, &mut stdin))
        .map_err(MediaError::Io)?;
    let result = decode_first_child_frame(&mut child, req, cancel);
    match feeder.join() {
        Ok(Ok(_)) => result,
        Ok(Err(error)) if error.kind() == std::io::ErrorKind::BrokenPipe => result,
        Ok(Err(error)) => Err(MediaError::Io(error)),
        Err(_) => Err(MediaError::Ffmpeg(
            "retained frame input feeder panicked".to_string(),
        )),
    }
}

fn decode_first_child_frame(
    child: &mut ffmpeg_sidecar::child::FfmpegChild,
    req: &FrameRequest,
    cancel: &MediaCancelToken,
) -> Result<(f64, RgbaFrame)> {
    let requested_time = req.time_secs;
    let iter = match child.iter() {
        Ok(iter) => iter,
        Err(error) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(MediaError::Ffmpeg(format!("iter: {error}")));
        }
    };
    let reader_cancel = cancel.clone();
    let target_us = target_micros(req);
    let reader = match thread::Builder::new()
        .name("opentake-retained-frame-events".to_string())
        .spawn(move || {
            reader_cancel.reader_started();
            let result = read_displayed_frame(iter, target_us);
            reader_cancel.reader_finished();
            result
        }) {
        Ok(reader) => reader,
        Err(error) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(MediaError::Ffmpeg(format!(
                "spawn frame event reader: {error}"
            )));
        }
    };
    loop {
        if cancel.checkpoint() {
            let _ = child.kill();
            let _ = child.wait();
            let _ = reader.join();
            return Err(MediaError::Cancelled);
        }
        if reader.is_finished() {
            let _ = child.kill();
            let _ = child.wait();
            let result = reader
                .join()
                .map_err(|_| MediaError::Ffmpeg("frame event reader panicked".to_string()))?;
            return result
                .ok_or_else(|| MediaError::Decode(format!("no frame at {requested_time:.3}s")));
        }
        match child.as_inner_mut().try_wait() {
            Ok(Some(_)) => {
                let result = reader
                    .join()
                    .map_err(|_| MediaError::Ffmpeg("frame event reader panicked".to_string()))?;
                return result.ok_or_else(|| {
                    MediaError::Decode(format!("no frame at {requested_time:.3}s"))
                });
            }
            Ok(None) => thread::sleep(FRAME_CHILD_POLL_INTERVAL),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = reader.join();
                return Err(MediaError::Io(error));
            }
        }
    }
}

/// Decode one cancellable frame and encode it as PNG bytes without publishing
/// a cache file. Project-scoped prewarm jobs stage these bytes and let their
/// epoch guard perform the final atomic rename.
pub fn decode_frame_png_cancellable(
    path: &Path,
    req: &FrameRequest,
    cancel: &MediaCancelToken,
) -> Result<(f64, Vec<u8>)> {
    let (actual, frame) = decode_frame_at_cancellable(path, req, cancel)?;
    let mut bytes = Vec::new();
    image::codecs::png::PngEncoder::new(&mut bytes)
        .write_image(
            &frame.rgba,
            frame.width,
            frame.height,
            image::ExtendedColorType::Rgba8,
        )
        .map_err(|error| MediaError::Encode(format!("png: {error}")))?;
    Ok((actual, bytes))
}

pub fn decode_frames_at_cancellable(
    path: &Path,
    times_secs: &[f64],
    base: &FrameRequest,
    cancel: &MediaCancelToken,
) -> Vec<Result<(f64, RgbaFrame)>> {
    let mut out = Vec::with_capacity(times_secs.len());
    let mut last_time = f64::NEG_INFINITY;
    for &time in times_secs {
        if cancel.checkpoint() {
            out.push(Err(MediaError::Cancelled));
            break;
        }
        let request = FrameRequest {
            time_secs: time,
            ..base.clone()
        };
        match decode_frame_at_cancellable(path, &request, cancel) {
            Ok((actual, frame)) if actual > last_time => {
                last_time = actual;
                out.push(Ok((actual, frame)));
            }
            Ok(_) | Err(MediaError::Decode(_)) => {}
            Err(error) => out.push(Err(error)),
        }
    }
    out
}

/// Decode a batch of ascending `times_secs`. De-duplicates frames whose decoded
/// timestamp does not strictly advance past the previous one (`t > lastTime`).
/// Returns `(actual_secs, frame)` pairs in ascending actual time. Frames that
/// fail to decode are skipped.
pub fn decode_frames_at(
    path: &Path,
    times_secs: &[f64],
    base: &FrameRequest,
) -> Vec<Result<(f64, RgbaFrame)>> {
    let mut out = Vec::with_capacity(times_secs.len());
    let mut last_time = f64::NEG_INFINITY;
    for &t in times_secs {
        let req = FrameRequest {
            time_secs: t,
            ..base.clone()
        };
        match decode_frame_at(path, &req) {
            Ok((actual, frame)) => {
                if actual <= last_time {
                    continue; // duplicate of an already-emitted keyframe
                }
                last_time = actual;
                out.push(Ok((actual, frame)));
            }
            Err(MediaError::Decode(_)) => continue, // skip undecodable point
            Err(e) => out.push(Err(e)),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::process::Command;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    fn two_tone(value: u8) -> RgbaFrame {
        let mut frame = RgbaFrame::black(8, 8);
        for pixel in frame.rgba.chunks_mut(8) {
            pixel[..3].copy_from_slice(&[value; 3]);
        }
        frame
    }

    #[test]
    fn near_endpoint_alpha_returns_endpoint_without_optical_flow() {
        let first = two_tone(40);
        let last = two_tone(200);
        OPTICAL_FLOW_CALLS.with(|calls| calls.set(0));
        for (alpha, expected) in [(1e-12, &first), (1.0 - 1e-12, &last)] {
            let result = interpolate_frame_pair(
                &first,
                &last,
                alpha,
                FrameInterpolationMode::OpticalFlow,
                FrameInterpolationFallback::Error,
                true,
            )
            .unwrap();
            assert_eq!(&result.frame, expected);
        }
        assert_eq!(OPTICAL_FLOW_CALLS.with(|calls| calls.get()), 0);

        interpolate_frame_pair(
            &first,
            &last,
            0.5,
            FrameInterpolationMode::OpticalFlow,
            FrameInterpolationFallback::Error,
            true,
        )
        .unwrap();
        assert_eq!(OPTICAL_FLOW_CALLS.with(|calls| calls.get()), 1);
    }

    #[test]
    fn equal_rates_map_every_frame_to_itself() {
        for fps in [24.0, 25.0, 30.0, 50.0, 60.0] {
            for n in 0..10_000 {
                assert_eq!(
                    source_frame_pair(n, fps, fps),
                    (n, n, 0.0),
                    "{fps} fps frame {n}"
                );
            }
        }
    }

    #[test]
    fn differing_rates_keep_real_fractional_positions() {
        // 24 fps source on a 30 fps timeline: frame 1 sits 0.8 into source 0.
        let (first, next, alpha) = source_frame_pair(1, 30.0, 24.0);
        assert_eq!((first, next), (0, 1));
        assert!((alpha - 0.8).abs() < 1e-9);
        assert_eq!(source_frame_pair(5, 30.0, 24.0), (4, 4, 0.0));
        assert_eq!(source_frame_pair(-3, 30.0, 24.0), (0, 0, 0.0));
    }

    // --- fit_within: pure scaling math ---

    #[test]
    fn cancelling_frame_decode_with_no_iterator_events_kills_child_and_joins_reader() {
        assert!(
            crate::ff::ffmpeg_available(),
            "required cancellation test needs a runnable FFmpeg"
        );
        let temp = tempfile::tempdir().expect("create frame cancellation fixture directory");
        let fifo = temp.path().join("blocking-video-input");
        let status = Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .expect("spawn mkfifo");
        assert!(
            status.success(),
            "mkfifo must create a blocking media input"
        );

        let cancel = MediaCancelToken::new();
        let worker_cancel = cancel.clone();
        let (done_tx, done_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let result =
                decode_frame_at_cancellable(&fifo, &FrameRequest::default(), &worker_cancel);
            done_tx.send(result).expect("publish frame decoder result");
        });

        let deadline = Instant::now() + Duration::from_secs(5);
        while cancel.spawned_child_count() == 0 && Instant::now() < deadline {
            std::thread::yield_now();
        }
        let spawned = cancel.spawned_child_count();
        cancel.cancel();
        assert_eq!(spawned, 1, "the test must cancel a live FFmpeg child");

        let result = done_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("cancellation must not wait for the first iterator event");
        assert!(matches!(result, Err(MediaError::Cancelled)));
        worker.join().expect("frame decoder worker must be reaped");
        assert_eq!(cancel.active_reader_count(), 0);
    }

    #[test]
    fn fit_within_no_box_keeps_size() {
        assert_eq!(fit_within(1920, 1080, (0, 0)), (1920, 1080));
    }

    #[test]
    fn fit_within_never_enlarges() {
        // box bigger than image → unchanged.
        assert_eq!(fit_within(100, 50, (1000, 1000)), (100, 50));
    }

    #[test]
    fn fit_within_scales_down_keeping_aspect() {
        // 1920x1080 into 120x68 box → width-limited: scale ~0.0625 → 120x68.
        let (w, h) = fit_within(1920, 1080, (120, 68));
        assert_eq!(w, 120);
        assert_eq!(h, 68);
    }

    #[test]
    fn fit_within_portrait_into_square_box() {
        // 1080x1920 into 512x512 → height-limited: scale 512/1920 → 288x512.
        let (w, h) = fit_within(1080, 1920, (512, 512));
        assert_eq!(h, 512);
        assert_eq!(w, 288);
    }

    #[test]
    fn fit_within_single_dim_box() {
        // only width bound (120), height unbounded.
        let (w, h) = fit_within(600, 300, (120, 0));
        assert_eq!(w, 120);
        assert_eq!(h, 60);
    }

    #[test]
    fn fit_within_min_one_pixel() {
        let (w, h) = fit_within(10000, 1, (5, 5));
        assert!(w >= 1 && h >= 1);
    }

    #[test]
    fn fit_within_zero_input() {
        assert_eq!(fit_within(0, 0, (10, 10)), (1, 1));
    }

    // --- frame_args: CLI contract ---

    #[test]
    fn frame_args_keyframe_seek_to_target_and_select_displayed_frame() {
        let req = FrameRequest {
            time_secs: 5.0,
            ..Default::default()
        };
        let args = frame_args(Path::new("/x.mp4"), &req);
        let ss = args.iter().position(|a| a == "-ss").unwrap();
        assert_eq!(args[ss + 1], "5.000000");
        for flag in ["-noaccurate_seek", "-copyts", "-start_at_zero"] {
            let at = args.iter().position(|a| a == flag).unwrap();
            assert!(at < ss, "{flag} must be an input option");
        }
        let vf = args.iter().position(|a| a == "-vf").unwrap();
        assert!(args[vf + 1].starts_with(
            "settb=1/1000000,showinfo=checksum=0,setpts=PTS-5000000,fps=fps=1:start_time=0:round=up"
        ));
        assert!(args.windows(2).any(|w| w == ["-fps_mode", "passthrough"]));

        let negative = frame_args(
            Path::new("/x.mp4"),
            &FrameRequest {
                time_secs: -0.5,
                ..Default::default()
            },
        );
        let ss0 = negative.iter().position(|a| a == "-ss").unwrap();
        assert_eq!(negative[ss0 + 1], "0.000000");
        assert!(negative.iter().any(|a| a.contains("setpts=PTS-0,")));
    }

    #[test]
    fn retained_input_args_drop_the_seek_but_keep_selection() {
        let args = frame_args_for_input(
            "fd:",
            &FrameRequest {
                time_secs: 2.0,
                ..Default::default()
            },
            None,
        );
        assert!(!args.iter().any(|a| a == "-ss"));
        assert!(args.windows(2).any(|w| w == ["-i", "fd:"]));
        assert!(args.iter().any(|a| a.contains("setpts=PTS-2000000")));
    }

    #[test]
    fn showinfo_pts_parses_padded_and_negative_values() {
        let padded =
            "[Parsed_showinfo_1 @ 0x1] [info] n:   0 pts: 966667 pts_time:0.966667 duration:1";
        assert_eq!(showinfo_pts(padded), Some(966_667));
        let negative = "[Parsed_showinfo_1 @ 0x1] [info] n:   2 pts:-33333 pts_time:-0.033333";
        assert_eq!(showinfo_pts(negative), Some(-33_333));
        assert_eq!(
            showinfo_pts("[Parsed_showinfo_1 @ 0x1] [info] config in time_base: 1/1000000"),
            None
        );
        assert_eq!(showinfo_pts("[h264 @ 0x2] [error] pts: 5"), None);
    }

    fn showinfo_event(pts: i64) -> FfmpegEvent {
        FfmpegEvent::Log(
            ffmpeg_sidecar::event::LogLevel::Info,
            format!("[Parsed_showinfo_1 @ 0x1] [info] n:   0 pts:{pts} pts_time:0"),
        )
    }

    fn output_event(value: u8) -> FfmpegEvent {
        FfmpegEvent::OutputFrame(ffmpeg_sidecar::event::OutputVideoFrame {
            width: 1,
            height: 1,
            pix_fmt: "rgba".to_string(),
            output_index: 0,
            data: vec![value, value, value, 255],
            frame_num: 0,
            timestamp: 0.0,
        })
    }

    #[test]
    fn displayed_frame_reports_real_pts_of_last_frame_at_or_before_target() {
        let (actual, frame) = read_displayed_frame(
            [
                showinfo_event(900_000),
                showinfo_event(966_667),
                output_event(7),
                showinfo_event(2_000_000),
            ],
            1_200_000,
        )
        .unwrap();
        assert!((actual - 0.966_667).abs() < 1e-9);
        assert_eq!(frame.rgba[0], 7);
    }

    #[test]
    fn displayed_frame_falls_back_to_first_frame_after_target() {
        let (actual, _) = read_displayed_frame(
            [
                output_event(1),
                showinfo_event(500_000),
                showinfo_event(600_000),
            ],
            0,
        )
        .unwrap();
        assert!((actual - 0.5).abs() < 1e-9);
        assert!(read_displayed_frame([showinfo_event(0)], 0).is_none());
    }

    #[test]
    fn frame_args_request_rgba_rawvideo_one_frame() {
        let args = frame_args(Path::new("/x.mp4"), &FrameRequest::default());
        assert!(args.windows(2).any(|w| w == ["-pix_fmt", "rgba"]));
        assert!(args.windows(2).any(|w| w == ["-f", "rawvideo"]));
        assert!(args.windows(2).any(|w| w == ["-frames:v", "1"]));
        assert_eq!(args.last().unwrap(), "-");
    }

    #[test]
    fn frame_args_adds_scale_filter_only_when_boxed() {
        let plain = frame_args(Path::new("/x.mp4"), &FrameRequest::default());
        assert!(!plain.iter().any(|a| a.contains("scale=")));

        let boxed = frame_args(
            Path::new("/x.mp4"),
            &FrameRequest {
                max_size: (120, 68),
                ..Default::default()
            },
        );
        let vf = boxed.iter().position(|a| a == "-vf").unwrap();
        assert!(boxed[vf + 1].contains("force_original_aspect_ratio=decrease"));
        assert!(boxed[vf + 1].contains("w=120"));
        assert!(boxed[vf + 1].contains("h=68"));
    }
}
