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

use std::path::Path;
use std::thread;
use std::time::Duration;

use ffmpeg_sidecar::event::FfmpegEvent;
use image::ImageEncoder;
use rayon::prelude::*;

use opentake_domain::MediaColorMetadata;

use crate::cancel::MediaCancelToken;
use crate::decode::source_color::{resolve_file_color, resolve_path_color, ColorHint};
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
    color: Option<&MediaColorMetadata>,
) -> Vec<String> {
    single_frame_args(path, req, color, FrameOutput::RawVideo)
}

/// Arguments for the retained-handle decoder: the handle is ffmpeg's stdin
/// and is read through `fd:`, which keeps normal file seek semantics, so the
/// keyframe seek stays an input option exactly as for a pathname. The frame
/// is written as a self-describing PAM image because this path parses
/// ffmpeg's output itself instead of through the sidecar event stream.
pub(super) fn retained_frame_args(
    req: &FrameRequest,
    color: Option<&MediaColorMetadata>,
) -> Vec<String> {
    let mut args: Vec<String> = ["-hide_banner", "-nostats", "-loglevel", "info"]
        .map(String::from)
        .to_vec();
    args.extend(single_frame_args(
        Path::new("fd:"),
        req,
        color,
        FrameOutput::Pam,
    ));
    args
}

/// Encoding of the RGBA frame(s) written to stdout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FrameOutput {
    /// Headerless RGBA, sized from the sidecar's parsed output metadata.
    RawVideo,
    /// Netpbm PAM (`P7`, `RGB_ALPHA`): the header carries the dimensions.
    Pam,
}

fn single_frame_args(
    path: &Path,
    req: &FrameRequest,
    color: Option<&MediaColorMetadata>,
    output: FrameOutput,
) -> Vec<String> {
    let time_secs = req.time_secs.max(0.0);
    let target_us = target_micros(req);
    let mut args = seek_input_args(path, time_secs, color);
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
    push_conversion_filters(&mut filters, req, color);
    push_rgba_output_args(&mut args, &filters, output);
    args
}

/// Input options shared by single-frame and batched decodes: a keyframe seek
/// at/before `time_secs` with source timestamps kept (relative to the
/// container start). Accurate seek is off because it drops every frame before
/// the target, including the one still on screen at it.
fn seek_input_args(path: &Path, time_secs: f64, color: Option<&MediaColorMetadata>) -> Vec<String> {
    let mut args: Vec<String> = Vec::new();
    if let Some(color) = color {
        args.extend(crate::color::hdr_decode_input_args(color));
    }
    args.extend(["-noaccurate_seek", "-copyts", "-start_at_zero", "-ss"].map(String::from));
    args.push(format!("{time_secs:.6}"));
    args.push("-i".into());
    args.push(path.to_string_lossy().into_owned());
    args
}

/// HDR tone mapping and the downscale box, applied after frame selection so
/// only emitted frames pay for them.
fn push_conversion_filters(
    filters: &mut Vec<String>,
    req: &FrameRequest,
    color: Option<&MediaColorMetadata>,
) {
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
}

fn push_rgba_output_args(args: &mut Vec<String>, filters: &[String], output: FrameOutput) {
    args.push("-vf".into());
    args.push(filters.join(","));
    args.push("-fps_mode".into());
    args.push("passthrough".into());
    args.push("-pix_fmt".into());
    args.push("rgba".into());
    match output {
        FrameOutput::RawVideo => args.extend(["-f", "rawvideo"].map(String::from)),
        FrameOutput::Pam => args.extend(["-c:v", "pam", "-f", "image2pipe"].map(String::from)),
    }
    args.push("-".into());
}

pub(super) const MICROS_PER_SEC: i64 = 1_000_000;

/// Real pts of the frame the display-frame filters select for a target: the
/// last logged pts `<= target_us`, else the first one after it (the padded
/// first frame), else the target itself.
#[derive(Clone, Copy, Debug)]
pub(super) struct DisplayedPts {
    target_us: i64,
    at_or_before: Option<i64>,
    after: Option<i64>,
}

impl DisplayedPts {
    pub(super) fn new(target_us: i64) -> Self {
        DisplayedPts {
            target_us,
            at_or_before: None,
            after: None,
        }
    }

    pub(super) fn observe(&mut self, pts: i64) {
        if pts <= self.target_us {
            self.at_or_before = Some(self.at_or_before.map_or(pts, |seen| seen.max(pts)));
        } else {
            self.after = Some(self.after.map_or(pts, |seen| seen.min(pts)));
        }
    }

    /// Log lines are ordered, so once a frame after the target has been
    /// logged the selection cannot change.
    fn is_final(&self) -> bool {
        self.after.is_some()
    }

    pub(super) fn secs(&self) -> f64 {
        self.at_or_before.or(self.after).unwrap_or(self.target_us) as f64 / MICROS_PER_SEC as f64
    }
}

/// Microsecond pts from a `showinfo` frame line (`... n:   3 pts:1200000 ...`).
pub(super) fn showinfo_pts(line: &str) -> Option<i64> {
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
    let mut selection = DisplayedPts::new(target_us);
    for event in events {
        match event {
            FfmpegEvent::OutputFrame(output)
                if frame.is_none() && output.width > 0 && output.height > 0 =>
            {
                frame = Some(RgbaFrame::new(output.width, output.height, output.data));
            }
            FfmpegEvent::Log(_, line) => {
                if let Some(pts) = showinfo_pts(&line) {
                    selection.observe(pts);
                }
            }
            _ => {}
        }
        if frame.is_some() && selection.is_final() {
            break;
        }
    }
    frame.map(|frame| (selection.secs(), frame))
}

pub(super) fn target_micros(req: &FrameRequest) -> i64 {
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
    decode_frame_at_with_color_cancellable(path, req, &ColorHint::Unknown, cancel)
}

/// [`decode_frame_at_cancellable`] with the caller's knowledge of the source's
/// color signalling. A [`ColorHint::Known`] (for example from the media
/// manifest) skips the color probe entirely; an unknown one is probed at most
/// once per file identity, and a failed probe fails the decode instead of
/// silently skipping HDR tone mapping.
pub fn decode_frame_at_with_color_cancellable(
    path: &Path,
    req: &FrameRequest,
    color: &ColorHint,
    cancel: &MediaCancelToken,
) -> Result<(f64, RgbaFrame)> {
    if cancel.is_cancelled() {
        return Err(MediaError::Cancelled);
    }
    let color = resolve_path_color(path, color, cancel)?;
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
    decode_frame_file_at_with_color_cancellable(file, req, &ColorHint::Unknown, cancel)
}

/// [`decode_frame_file_at_cancellable`] with the caller's color hint; see
/// [`decode_frame_at_with_color_cancellable`].
///
/// The handle itself (not a pipe fed from it) is ffmpeg's stdin, so the
/// decoder can seek: MP4/MOV files whose `moov` index follows the media data
/// decode like any path, and the keyframe seek happens on input instead of
/// decoding from the start of the file.
pub fn decode_frame_file_at_with_color_cancellable(
    file: &std::fs::File,
    req: &FrameRequest,
    color: &ColorHint,
    cancel: &MediaCancelToken,
) -> Result<(f64, RgbaFrame)> {
    if cancel.is_cancelled() {
        return Err(MediaError::Cancelled);
    }
    let color = resolve_file_color(file, color, cancel)?;
    super::retained::decode_retained_frame(file, req, color.as_ref(), cancel)
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
    decode_frames_at_with_color_cancellable(path, times_secs, base, &ColorHint::Unknown, cancel)
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
    decode_frames_at_with_color_cancellable(
        path,
        times_secs,
        base,
        &ColorHint::Unknown,
        &MediaCancelToken::new(),
    )
}

/// Batch decode with the caller's color hint. The source color is resolved
/// once for the whole batch. Dense runs of targets on a common time grid
/// (thumbnail strips, sampling cadences) are decoded by one forward ffmpeg
/// pass that selects the displayed frame at every target; isolated or
/// irregular targets fall back to one keyframe seek each.
pub fn decode_frames_at_with_color_cancellable(
    path: &Path,
    times_secs: &[f64],
    base: &FrameRequest,
    color: &ColorHint,
    cancel: &MediaCancelToken,
) -> Vec<Result<(f64, RgbaFrame)>> {
    let mut out = Vec::with_capacity(times_secs.len());
    if times_secs.is_empty() {
        return out;
    }
    if cancel.checkpoint() {
        out.push(Err(MediaError::Cancelled));
        return out;
    }
    let color = match resolve_path_color(path, color, cancel) {
        Ok(color) => color,
        Err(error) => {
            out.push(Err(error));
            return out;
        }
    };
    let known = ColorHint::Known(color.clone());
    let mut last_time = f64::NEG_INFINITY;
    for plan in plan_frame_batches(times_secs) {
        if cancel.checkpoint() {
            out.push(Err(MediaError::Cancelled));
            break;
        }
        let decoded = match plan {
            FramePlan::Single(index) => {
                let request = FrameRequest {
                    time_secs: times_secs[index],
                    ..base.clone()
                };
                vec![decode_frame_at_with_color_cancellable(
                    path, &request, &known, cancel,
                )]
            }
            FramePlan::Grid(run) => match decode_grid_run(path, base, color.as_ref(), &run, cancel)
            {
                Ok(frames) => frames.into_iter().map(Ok).collect(),
                Err(error) => vec![Err(error)],
            },
        };
        for result in decoded {
            match result {
                Ok((actual, frame)) if actual > last_time => {
                    last_time = actual;
                    out.push(Ok((actual, frame)));
                }
                // A duplicate of an already-emitted frame, or an undecodable
                // point: skip it.
                Ok(_) | Err(MediaError::Decode(_)) => {}
                Err(MediaError::Cancelled) => {
                    out.push(Err(MediaError::Cancelled));
                    return out;
                }
                Err(error) => out.push(Err(error)),
            }
        }
    }
    out
}

/// Consecutive batch targets at most this far apart share one forward
/// decode. Wider gaps seek per frame, which decodes less than running through
/// the gap.
const BATCH_MAX_GAP_US: i64 = 2 * MICROS_PER_SEC;
/// Targets decoded by one forward pass; bounds the selection expression.
const BATCH_MAX_TARGETS: usize = 240;
/// Finest grid a forward pass selects on; finer common steps (irregular
/// targets) fall back to per-frame seeks.
const BATCH_MIN_STEP_US: i64 = 1_000;
/// Grid slots one forward pass may step through.
const BATCH_MAX_SLOTS: i64 = 100_000;

/// How one batch target is decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
enum FramePlan {
    /// Keyframe-seek decode of `times[index]` alone.
    Single(usize),
    /// One forward pass over the next `slots.len()` targets.
    Grid(GridRun),
}

/// Consecutive ascending targets that lie on the grid
/// `origin_us + slot * step_us`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct GridRun {
    origin_us: i64,
    step_us: i64,
    /// Grid slot of each target: strictly increasing, starting at 0.
    slots: Vec<i64>,
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.abs()
}

/// Split `times` into forward-pass grid runs and single seeks, preserving
/// order. A run is a maximal sequence of strictly ascending, finite,
/// non-negative targets whose consecutive gaps are at most
/// [`BATCH_MAX_GAP_US`] and whose offsets share a grid step of at least
/// [`BATCH_MIN_STEP_US`].
fn plan_frame_batches(times: &[f64]) -> Vec<FramePlan> {
    let micros = times
        .iter()
        .map(|&time| {
            (time.is_finite() && time >= 0.0).then(|| (time * MICROS_PER_SEC as f64).round() as i64)
        })
        .collect::<Vec<_>>();
    let mut plans = Vec::new();
    let mut index = 0;
    while index < times.len() {
        if micros[index].is_none() {
            plans.push(FramePlan::Single(index));
            index += 1;
            continue;
        }
        let mut end = index + 1;
        while end < times.len() && end - index < BATCH_MAX_TARGETS {
            let (Some(previous), Some(next)) = (micros[end - 1], micros[end]) else {
                break;
            };
            if next <= previous || next - previous > BATCH_MAX_GAP_US {
                break;
            }
            end += 1;
        }
        match grid_run(&micros[index..end]) {
            Some(run) => plans.push(FramePlan::Grid(run)),
            None => plans.extend((index..end).map(FramePlan::Single)),
        }
        index = end;
    }
    plans
}

fn grid_run(micros: &[Option<i64>]) -> Option<GridRun> {
    if micros.len() < 2 {
        return None;
    }
    let origin_us = micros[0]?;
    let offsets = micros
        .iter()
        .map(|micros| micros.map(|micros| micros - origin_us))
        .collect::<Option<Vec<_>>>()?;
    let step_us = offsets[1..]
        .iter()
        .fold(0, |step, &offset| gcd(step, offset));
    if step_us < BATCH_MIN_STEP_US || offsets.last()? / step_us >= BATCH_MAX_SLOTS {
        return None;
    }
    Some(GridRun {
        origin_us,
        step_us,
        slots: offsets.iter().map(|offset| offset / step_us).collect(),
    })
}

/// `select` filter keeping only the grid slots that are targets, or `None`
/// when every slot is one.
fn grid_selection(slots: &[i64]) -> Option<String> {
    let contiguous = slots
        .iter()
        .enumerate()
        .all(|(index, &slot)| slot == index as i64);
    if contiguous {
        return None;
    }
    let mut terms = Vec::new();
    let mut index = 0;
    while index < slots.len() {
        let start = slots[index];
        let mut end = start;
        while index + 1 < slots.len() && slots[index + 1] == end + 1 {
            index += 1;
            end += 1;
        }
        terms.push(if start == end {
            format!("eq(n,{start})")
        } else {
            format!("between(n,{start},{end})")
        });
        index += 1;
    }
    Some(format!("select='{}'", terms.join("+")))
}

/// One forward decode emitting the frame displayed at every target of `run`.
/// The same selection as single-frame decode, generalized to a grid: shift
/// the first target to 0, let `fps` at the grid rate put the last frame with
/// pts <= each grid time into that slot, then keep only the target slots.
fn grid_frame_args(
    path: &Path,
    base: &FrameRequest,
    color: Option<&MediaColorMetadata>,
    run: &GridRun,
) -> Vec<String> {
    let origin_secs = run.origin_us as f64 / MICROS_PER_SEC as f64;
    let mut args = seek_input_args(path, origin_secs, color);
    args.push("-frames:v".into());
    args.push(run.slots.len().to_string());
    let mut filters: Vec<String> = vec![
        format!("settb=1/{MICROS_PER_SEC}"),
        "showinfo=checksum=0".to_string(),
        format!("setpts=PTS-{}", run.origin_us),
        format!(
            "fps=fps={MICROS_PER_SEC}/{}:start_time=0:round=up",
            run.step_us
        ),
    ];
    filters.extend(grid_selection(&run.slots));
    push_conversion_filters(&mut filters, base, color);
    push_rgba_output_args(&mut args, &filters, FrameOutput::RawVideo);
    args
}

/// Every output frame (up to `expected`) plus every logged source pts.
fn read_grid_frames(
    events: impl IntoIterator<Item = FfmpegEvent>,
    expected: usize,
) -> (Vec<RgbaFrame>, Vec<i64>) {
    let mut frames = Vec::with_capacity(expected);
    let mut pts = Vec::new();
    for event in events {
        match event {
            FfmpegEvent::OutputFrame(output)
                if frames.len() < expected && output.width > 0 && output.height > 0 =>
            {
                frames.push(RgbaFrame::new(output.width, output.height, output.data));
            }
            FfmpegEvent::Log(_, line) => pts.extend(showinfo_pts(&line)),
            _ => {}
        }
    }
    (frames, pts)
}

/// The real pts of the frame shown at `target_us`: the last logged pts at or
/// before it, else the first one after it (the padded first frame).
fn displayed_pts(sorted_pts: &[i64], target_us: i64) -> i64 {
    match sorted_pts.partition_point(|&pts| pts <= target_us) {
        0 => sorted_pts.first().copied().unwrap_or(target_us),
        after => sorted_pts[after - 1],
    }
}

fn decode_grid_run(
    path: &Path,
    base: &FrameRequest,
    color: Option<&MediaColorMetadata>,
    run: &GridRun,
    cancel: &MediaCancelToken,
) -> Result<Vec<(f64, RgbaFrame)>> {
    if cancel.is_cancelled() {
        return Err(MediaError::Cancelled);
    }
    let mut child = ff::ffmpeg()
        .args(grid_frame_args(path, base, color, run))
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
    let expected = run.slots.len();
    let reader = match thread::Builder::new()
        .name("opentake-frame-batch-events".to_string())
        .spawn(move || {
            reader_cancel.reader_started();
            let result = read_grid_frames(iter, expected);
            reader_cancel.reader_finished();
            result
        }) {
        Ok(reader) => reader,
        Err(error) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(MediaError::Ffmpeg(format!(
                "spawn frame batch event reader: {error}"
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
            break;
        }
        match child.as_inner_mut().try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => thread::sleep(FRAME_CHILD_POLL_INTERVAL),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = reader.join();
                return Err(MediaError::Io(error));
            }
        }
    }
    // Both pipes reach EOF once the command exits after its last frame; the
    // reader drains every remaining event, so joining it cannot block on a
    // producer. Reap the child explicitly afterwards.
    let joined = reader.join();
    let _ = child.kill();
    let _ = child.wait();
    let (frames, mut pts) =
        joined.map_err(|_| MediaError::Ffmpeg("frame batch event reader panicked".to_string()))?;
    if cancel.is_cancelled() {
        return Err(MediaError::Cancelled);
    }
    if frames.is_empty() {
        return Err(MediaError::Decode(format!(
            "no frame at {:.3}s",
            run.origin_us as f64 / MICROS_PER_SEC as f64
        )));
    }
    pts.sort_unstable();
    Ok(frames
        .into_iter()
        .zip(&run.slots)
        .map(|(frame, slot)| {
            let target_us = run.origin_us + slot * run.step_us;
            let actual = displayed_pts(&pts, target_us) as f64 / MICROS_PER_SEC as f64;
            (actual, frame)
        })
        .collect())
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
    fn retained_input_args_seek_on_input_and_emit_a_pam_frame() {
        let args = retained_frame_args(
            &FrameRequest {
                time_secs: 2.0,
                ..Default::default()
            },
            None,
        );
        let ss = args.iter().position(|a| a == "-ss").unwrap();
        let input = args.windows(2).position(|w| w == ["-i", "fd:"]).unwrap();
        assert!(ss < input, "the retained handle seeks like a pathname");
        assert_eq!(args[ss + 1], "2.000000");
        assert!(args.iter().any(|a| a.contains("setpts=PTS-2000000")));
        assert!(args.windows(2).any(|w| w == ["-c:v", "pam"]));
        assert!(args.windows(2).any(|w| w == ["-f", "image2pipe"]));
        assert!(!args.iter().any(|a| a == "rawvideo"));
        assert_eq!(args.last().unwrap(), "-");
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

    // --- batched decode planning ---

    fn grid(origin_us: i64, step_us: i64, slots: &[i64]) -> FramePlan {
        FramePlan::Grid(GridRun {
            origin_us,
            step_us,
            slots: slots.to_vec(),
        })
    }

    #[test]
    fn thumbnail_cadence_is_one_forward_pass() {
        assert_eq!(
            plan_frame_batches(&[0.0, 1.0, 2.0, 3.0, 4.0]),
            vec![grid(0, 1_000_000, &[0, 1, 2, 3, 4])]
        );
        // A sampling cadence offset from zero keeps its own origin.
        assert_eq!(
            plan_frame_batches(&[0.5, 2.5, 4.5]),
            vec![grid(500_000, 2_000_000, &[0, 1, 2])]
        );
    }

    #[test]
    fn subsets_of_a_grid_select_only_their_slots() {
        assert_eq!(
            plan_frame_batches(&[0.0, 1.0, 1.01]),
            vec![grid(0, 10_000, &[0, 100, 101])]
        );
        assert_eq!(grid_selection(&[0, 1, 2]), None);
        assert_eq!(
            grid_selection(&[0, 100, 101]).as_deref(),
            Some("select='eq(n,0)+between(n,100,101)'")
        );
    }

    #[test]
    fn sparse_irregular_or_unordered_targets_seek_per_frame() {
        // Gaps wider than the forward-pass limit.
        assert_eq!(
            plan_frame_batches(&[0.0, 10.0, 20.0]),
            (0..3).map(FramePlan::Single).collect::<Vec<_>>()
        );
        // No common grid step of at least a millisecond.
        assert_eq!(
            plan_frame_batches(&[0.0, 1.0 / 3.0, 2.0 / 3.0]),
            (0..3).map(FramePlan::Single).collect::<Vec<_>>()
        );
        // Non-finite, negative, and non-ascending targets end a run.
        assert_eq!(
            plan_frame_batches(&[f64::NAN, -1.0, 2.0, 1.0, 2.0, 3.0]),
            vec![
                FramePlan::Single(0),
                FramePlan::Single(1),
                FramePlan::Single(2),
                grid(1_000_000, 1_000_000, &[0, 1, 2]),
            ]
        );
    }

    #[test]
    fn long_batches_split_into_bounded_runs() {
        let times = (0..300).map(f64::from).collect::<Vec<_>>();
        let plans = plan_frame_batches(&times);
        assert_eq!(plans.len(), 2);
        let FramePlan::Grid(first) = &plans[0] else {
            panic!("expected a grid run: {plans:?}");
        };
        let FramePlan::Grid(second) = &plans[1] else {
            panic!("expected a grid run: {plans:?}");
        };
        assert_eq!(first.slots.len(), BATCH_MAX_TARGETS);
        assert_eq!(second.origin_us, BATCH_MAX_TARGETS as i64 * 1_000_000);
        assert_eq!(second.slots.len(), 300 - BATCH_MAX_TARGETS);
    }

    #[test]
    fn grid_args_seek_to_the_first_target_and_select_on_the_grid() {
        let run = GridRun {
            origin_us: 1_500_000,
            step_us: 500_000,
            slots: vec![0, 1, 4],
        };
        let base = FrameRequest {
            max_size: (120, 68),
            ..FrameRequest::default()
        };
        let args = grid_frame_args(Path::new("/x.mp4"), &base, None, &run);
        let ss = args.iter().position(|a| a == "-ss").unwrap();
        let input = args.iter().position(|a| a == "-i").unwrap();
        assert!(ss < input);
        assert_eq!(args[ss + 1], "1.500000");
        assert!(args.windows(2).any(|w| w == ["-frames:v", "3"]));
        let vf = args.iter().position(|a| a == "-vf").unwrap();
        assert_eq!(
            args[vf + 1],
            "settb=1/1000000,showinfo=checksum=0,setpts=PTS-1500000,\
             fps=fps=1000000/500000:start_time=0:round=up,\
             select='between(n,0,1)+eq(n,4)',\
             scale=w=120:h=68:force_original_aspect_ratio=decrease"
        );
    }

    #[test]
    fn batch_frames_report_the_real_pts_on_screen_at_each_target() {
        let pts = [900_000, 966_667, 1_033_333, 2_000_000];
        assert_eq!(displayed_pts(&pts, 1_000_000), 966_667);
        assert_eq!(displayed_pts(&pts, 1_500_000), 1_033_333);
        assert_eq!(displayed_pts(&pts, 0), 900_000);
        assert_eq!(displayed_pts(&[], 42), 42);

        let events = [
            showinfo_event(900_000),
            output_event(1),
            showinfo_event(966_667),
            output_event(2),
            output_event(3),
        ];
        let (frames, logged) = read_grid_frames(events, 2);
        assert_eq!(frames.len(), 2, "frames beyond the targets are ignored");
        assert_eq!(logged, vec![900_000, 966_667]);
    }

    // --- color probing (#47) ---

    fn run_ffmpeg(args: &[&str], output: &Path) -> bool {
        Command::new(crate::ff::ffmpeg_path())
            .args(["-hide_banner", "-loglevel", "error", "-y"])
            .args(args)
            .arg(output)
            .status()
            .is_ok_and(|status| status.success())
    }

    fn small_clip(path: &Path, seconds: u32) {
        let duration = seconds.to_string();
        assert!(
            run_ffmpeg(
                &[
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=64x36:rate=5",
                    "-t",
                    &duration,
                    "-c:v",
                    "mpeg4",
                ],
                path,
            ),
            "generate clip fixture"
        );
    }

    #[test]
    fn decoding_many_frames_of_one_file_probes_color_at_most_once() {
        assert!(
            crate::ff::ffmpeg_available(),
            "color probe accounting test needs a runnable FFmpeg"
        );
        let temp = tempfile::tempdir().unwrap();
        let clip = temp.path().join("frames.mp4");
        small_clip(&clip, 22);
        let probes = crate::ff::test_seams::probe_requests;

        let before = probes();
        for time_secs in [0.0, 1.0, 2.5, 4.0] {
            let request = FrameRequest {
                time_secs,
                max_size: (32, 18),
                ..FrameRequest::default()
            };
            decode_frame_at_cancellable(&clip, &request, &MediaCancelToken::new()).unwrap();
        }
        assert_eq!(probes() - before, 1, "single-frame decodes share one probe");

        // A caller that knows the signalling never probes.
        let other = temp.path().join("hinted.mp4");
        std::fs::copy(&clip, &other).unwrap();
        let before = probes();
        for time_secs in [0.0, 3.0] {
            let request = FrameRequest {
                time_secs,
                ..FrameRequest::default()
            };
            decode_frame_at_with_color_cancellable(
                &other,
                &request,
                &ColorHint::Known(None),
                &MediaCancelToken::new(),
            )
            .unwrap();
        }
        assert_eq!(probes(), before, "a known color hint skips ffprobe");

        // Twenty thumbnails of a not-yet-probed file: one ffprobe plus one
        // forward ffmpeg pass, instead of two processes per thumbnail.
        let fresh = temp.path().join("thumbnails.mp4");
        std::fs::copy(&clip, &fresh).unwrap();
        let times = (0..20).map(f64::from).collect::<Vec<_>>();
        let cancel = MediaCancelToken::new();
        let before = probes();
        let thumbs = decode_frames_at_cancellable(
            &fresh,
            &times,
            &FrameRequest {
                max_size: (32, 18),
                ..FrameRequest::default()
            },
            &cancel,
        );
        assert_eq!(thumbs.len(), 20);
        assert!(thumbs.iter().all(Result::is_ok));
        assert_eq!(probes() - before, 1);
        assert_eq!(cancel.spawned_child_count(), 2, "one ffprobe + one ffmpeg");
    }

    #[test]
    fn saturated_probe_admission_never_yields_untonemapped_hdr_frames() {
        assert!(
            crate::ff::ffmpeg_available(),
            "HDR admission test needs a runnable FFmpeg"
        );
        let temp = tempfile::tempdir().unwrap();
        let clip = temp.path().join("pq.mp4");
        let generated = run_ffmpeg(
            &[
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=160x90:rate=24",
                "-frames:v",
                "1",
                "-vf",
                "format=yuv420p10le",
                "-c:v",
                "libx265",
                "-preset",
                "ultrafast",
                "-x265-params",
                "log-level=error:hdr-opt=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc",
                "-color_primaries",
                "bt2020",
                "-color_trc",
                "smpte2084",
                "-colorspace",
                "bt2020nc",
            ],
            &clip,
        );
        if !generated {
            eprintln!("skip: this FFmpeg cannot encode a PQ HEVC fixture");
            return;
        }
        let request = FrameRequest {
            max_size: (160, 90),
            ..FrameRequest::default()
        };
        let cancel = MediaCancelToken::new();
        let hdr = crate::probe::probe(&clip).unwrap().color.unwrap();
        assert!(hdr.is_hdr());

        let (_, probed) = decode_frame_at_cancellable(&clip, &request, &cancel).unwrap();
        let (_, as_sdr) = decode_frame_at_with_color_cancellable(
            &clip,
            &request,
            &ColorHint::Known(None),
            &cancel,
        )
        .unwrap();
        if crate::color::hdr_tonemap_filter(&hdr).is_some() {
            assert_ne!(probed.rgba, as_sdr.rgba, "PQ frames must be tone-mapped");
        }

        crate::ff::test_seams::saturate_admission(true);
        crate::ff::test_seams::override_admission_wait(Some(Duration::from_millis(50)));
        let hinted = decode_frame_at_with_color_cancellable(
            &clip,
            &request,
            &ColorHint::Known(Some(hdr)),
            &cancel,
        );
        // A different identity is not cached, so it has to probe and cannot.
        let copy = temp.path().join("pq-copy.mp4");
        std::fs::copy(&clip, &copy).unwrap();
        let unknown = decode_frame_at_cancellable(&copy, &request, &cancel);
        crate::ff::test_seams::saturate_admission(false);
        crate::ff::test_seams::override_admission_wait(None);

        let (_, hinted) = hinted.expect("a hinted HDR decode needs no probe admission");
        assert_eq!(hinted.rgba, probed.rgba, "hinted decode is tone-mapped");
        match unknown {
            Err(MediaError::Ffmpeg(message)) => {
                assert!(message.contains("source color probe failed"), "{message}");
                assert!(message.contains("admission limit"), "{message}");
            }
            other => panic!("expected an explicit probe error, got {other:?}"),
        }
    }

    #[cfg(unix)]
    #[test]
    fn cancelling_during_the_color_probe_returns_promptly() {
        use std::os::unix::fs::PermissionsExt;

        let temp = tempfile::tempdir().unwrap();
        let entered = temp.path().join("probe-entered");
        let script = temp.path().join("stuck-ffprobe");
        std::fs::write(
            &script,
            format!("#!/bin/sh\ntouch '{}'\nexec sleep 60\n", entered.display()),
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).unwrap();
        let source = temp.path().join("source.mp4");
        std::fs::write(&source, b"any regular file is probed first").unwrap();

        let cancel = MediaCancelToken::new();
        let worker_cancel = cancel.clone();
        let (done_tx, done_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            crate::ff::test_seams::override_ffprobe(Some(script.into_os_string()));
            let result =
                decode_frame_at_cancellable(&source, &FrameRequest::default(), &worker_cancel);
            done_tx.send((result, Instant::now())).unwrap();
        });
        let deadline = Instant::now() + Duration::from_secs(10);
        while !entered.exists() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(entered.exists(), "the stuck ffprobe must be running");
        let cancelled_at = Instant::now();
        cancel.cancel();
        let (result, returned_at) = done_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("cancellation must not wait for the probe deadline");
        worker.join().unwrap();
        assert!(matches!(result, Err(MediaError::Cancelled)), "{result:?}");
        let latency = returned_at.duration_since(cancelled_at);
        eprintln!("color probe cancellation latency: {latency:?}");
        // Bounded by the probe poll interval plus cleanup, not the 10 s probe
        // deadline; generous for loaded CI runners.
        assert!(latency < Duration::from_secs(1), "{latency:?}");
        assert_eq!(cancel.spawned_child_count(), 1, "no decoder was spawned");
    }
}
