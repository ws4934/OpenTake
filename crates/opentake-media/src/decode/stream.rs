//! Continuous video frame decode for the Rust playback pipeline.
//!
//! This module is intentionally isolated from the existing seek-per-frame
//! preview path. It provides the first reusable building block for #53: one
//! worker thread runs ffmpeg forward, maps each output image to an integer
//! project-frame PTS, and pushes frames through a bounded queue.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use ffmpeg_sidecar::event::{FfmpegEvent, LogLevel, OutputVideoFrame};

use crate::error::{MediaError, Result};
use crate::ff;
use crate::frame::RgbaFrame;

/// Default number of decoded frames buffered between the ffmpeg worker and the
/// render/composite worker. At 30 fps this is roughly 250 ms of video.
pub const DEFAULT_VIDEO_STREAM_QUEUE_CAPACITY: usize = 8;

const BACKPRESSURE_SLEEP: Duration = Duration::from_millis(5);

/// Most recent ffmpeg error/fatal log lines attached to a stream failure.
const FAILURE_LOG_TAIL_LINES: usize = 20;

/// A continuous video decode request.
///
/// `start_frame` / `end_frame` are source-frame positions on the project
/// timeline fps timebase. `end_frame` is exclusive. The worker emits one RGBA
/// frame per project frame: frame `F` is the source frame displayed at
/// `F / timeline_fps` (the last one with pts `<= F / timeline_fps`), the same
/// rule single-frame decode uses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoStreamRequest {
    pub path: PathBuf,
    pub start_frame: i64,
    pub end_frame: Option<i64>,
    pub timeline_fps: i32,
    /// Upper bound box; `(0, 0)` disables scaling.
    pub max_size: (u32, u32),
    pub queue_capacity: usize,
    pub apply_rotation: bool,
}

impl VideoStreamRequest {
    pub fn new(path: impl Into<PathBuf>, timeline_fps: i32) -> Self {
        VideoStreamRequest {
            path: path.into(),
            start_frame: 0,
            end_frame: None,
            timeline_fps,
            max_size: (0, 0),
            queue_capacity: DEFAULT_VIDEO_STREAM_QUEUE_CAPACITY,
            apply_rotation: true,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.timeline_fps <= 0 {
            return Err(MediaError::Decode(format!(
                "timeline_fps must be > 0, got {}",
                self.timeline_fps
            )));
        }
        if self.start_frame < 0 {
            return Err(MediaError::Decode(format!(
                "start_frame must be >= 0, got {}",
                self.start_frame
            )));
        }
        if let Some(end_frame) = self.end_frame {
            if end_frame <= self.start_frame {
                return Err(MediaError::Decode(format!(
                    "end_frame must be > start_frame, got {end_frame} <= {}",
                    self.start_frame
                )));
            }
        }
        if self.queue_capacity == 0 {
            return Err(MediaError::Decode(
                "queue_capacity must be at least 1".to_string(),
            ));
        }
        Ok(())
    }

    fn start_secs(&self) -> f64 {
        frame_to_secs(self.start_frame, self.timeline_fps)
    }

    fn frame_limit(&self) -> Option<i64> {
        self.end_frame.map(|end| end - self.start_frame)
    }
}

/// One decoded source frame with an integer-frame PTS.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamVideoFrame {
    pub source_frame: i64,
    pub pts_secs: f64,
    pub frame: RgbaFrame,
}

/// Cloneable cooperative stop control for a decode worker.
#[derive(Clone, Debug)]
pub struct StreamDecodeControl {
    stop: Arc<AtomicBool>,
}

impl StreamDecodeControl {
    pub fn request_stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }

    pub fn is_stopped(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }
}

/// Handle for a spawned video decode worker.
///
/// Dropping the handle requests cooperative stop. Call [`VideoStream::join`] in
/// owners that need deterministic teardown before replacing a playback session.
pub struct VideoStream {
    receiver: Receiver<Result<StreamVideoFrame>>,
    control: StreamDecodeControl,
    worker: Option<JoinHandle<()>>,
}

impl VideoStream {
    pub fn receiver(&self) -> &Receiver<Result<StreamVideoFrame>> {
        &self.receiver
    }

    pub fn control(&self) -> StreamDecodeControl {
        self.control.clone()
    }

    pub fn request_stop(&self) {
        self.control.request_stop();
    }

    pub fn join(mut self) -> thread::Result<()> {
        self.request_stop();
        match self.worker.take() {
            Some(worker) => worker.join(),
            None => Ok(()),
        }
    }
}

impl Drop for VideoStream {
    fn drop(&mut self) {
        self.request_stop();
    }
}

/// Spawn a forward video decode worker.
///
/// This is not wired into preview playback yet. It is the media-side primitive
/// the render playback pipeline will consume in a later PR.
pub fn spawn_video_stream(req: VideoStreamRequest) -> Result<VideoStream> {
    req.validate()?;
    let (tx, rx) = sync_channel(req.queue_capacity);
    let control = StreamDecodeControl {
        stop: Arc::new(AtomicBool::new(false)),
    };
    let worker_control = control.clone();
    let worker = thread::Builder::new()
        .name("opentake-video-decode".to_string())
        .spawn(move || run_video_stream(req, tx, worker_control))
        .map_err(MediaError::Io)?;

    Ok(VideoStream {
        receiver: rx,
        control,
        worker: Some(worker),
    })
}

fn run_video_stream(
    req: VideoStreamRequest,
    tx: SyncSender<Result<StreamVideoFrame>>,
    control: StreamDecodeControl,
) {
    let color = crate::probe::probe(&req.path)
        .ok()
        .and_then(|probe| probe.color);
    let args = video_stream_args_with_color(&req, color.as_ref());
    let mut child = match ff::ffmpeg().args(args).spawn() {
        Ok(child) => child,
        Err(e) => {
            let _ = send_with_backpressure(
                &tx,
                Err(MediaError::Ffmpeg(format!("spawn: {e}"))),
                &control,
            );
            return;
        }
    };

    let iter = match child.iter() {
        Ok(iter) => iter,
        Err(e) => {
            let _ = send_with_backpressure(
                &tx,
                Err(MediaError::Ffmpeg(format!("iter: {e}"))),
                &control,
            );
            let _ = child.wait();
            return;
        }
    };

    let mut log_tail = VecDeque::with_capacity(FAILURE_LOG_TAIL_LINES);
    let mut fatal = false;
    let mut stopped = false;
    for event in iter {
        if control.is_stopped() {
            let _ = child.quit();
            stopped = true;
            break;
        }

        match event {
            FfmpegEvent::OutputFrame(frame) => {
                let decoded = stream_frame_from_output(&req, frame);
                if !send_with_backpressure(&tx, Ok(decoded), &control) {
                    let _ = child.quit();
                    stopped = true;
                    break;
                }
            }
            // This worker has a single rawvideo output, so the sidecar's
            // non-fatal `Error` uses (mismatched multi-output rates) cannot
            // occur here.
            FfmpegEvent::Error(e) | FfmpegEvent::Log(LogLevel::Fatal, e) => {
                fatal = true;
                push_log_tail(&mut log_tail, e);
            }
            // Decoders log recoverable bitstream damage at [error] and keep
            // going with error concealment; keep it for diagnostics only.
            FfmpegEvent::Log(LogLevel::Error, e) => push_log_tail(&mut log_tail, e),
            FfmpegEvent::Done => break,
            _ => {}
        }
    }

    let status = child.wait();
    if stopped {
        return;
    }
    let failure = match status {
        Err(e) => Some(format!("wait: {e}")),
        Ok(status) if !status.success() => Some(format!("ffmpeg exited with {status}")),
        Ok(_) if fatal => Some("ffmpeg reported a fatal error".to_string()),
        Ok(_) => None,
    };
    if let Some(reason) = failure {
        let message = if log_tail.is_empty() {
            reason
        } else {
            let tail = log_tail.into_iter().collect::<Vec<_>>().join(" | ");
            format!("{reason}: {tail}")
        };
        let _ = send_with_backpressure(&tx, Err(MediaError::Ffmpeg(message)), &control);
    }
}

fn push_log_tail(tail: &mut VecDeque<String>, line: String) {
    if tail.len() == FAILURE_LOG_TAIL_LINES {
        tail.pop_front();
    }
    tail.push_back(line);
}

fn send_with_backpressure(
    tx: &SyncSender<Result<StreamVideoFrame>>,
    mut item: Result<StreamVideoFrame>,
    control: &StreamDecodeControl,
) -> bool {
    loop {
        if control.is_stopped() {
            return false;
        }
        match tx.try_send(item) {
            Ok(()) => return true,
            Err(TrySendError::Disconnected(_)) => return false,
            Err(TrySendError::Full(returned)) => {
                item = returned;
                thread::sleep(BACKPRESSURE_SLEEP);
            }
        }
    }
}

fn stream_frame_from_output(req: &VideoStreamRequest, frame: OutputVideoFrame) -> StreamVideoFrame {
    // The filter chain emits one frame per grid slot starting exactly at
    // `start_frame`, so the output count is the slot offset.
    let source_frame = req.start_frame + i64::from(frame.frame_num);
    StreamVideoFrame {
        source_frame,
        pts_secs: frame_to_secs(source_frame, req.timeline_fps),
        frame: RgbaFrame::new(frame.width, frame.height, frame.data),
    }
}

fn frame_to_secs(frame: i64, fps: i32) -> f64 {
    frame.max(0) as f64 / fps.max(1) as f64
}

#[cfg(test)]
fn video_stream_args(req: &VideoStreamRequest) -> Vec<String> {
    video_stream_args_with_color(req, None)
}

fn video_stream_args_with_color(
    req: &VideoStreamRequest,
    color: Option<&opentake_domain::MediaColorMetadata>,
) -> Vec<String> {
    let mut args = Vec::new();
    // Keyframe seek with source timestamps kept (relative to the container
    // start) so the fps grid below is the absolute project-frame grid. Accurate
    // seek is off because it drops the frame still on screen at the start.
    args.extend(["-noaccurate_seek", "-copyts", "-start_at_zero", "-ss"].map(String::from));
    args.push(format!("{:.6}", req.start_secs()));
    if let Some(color) = color {
        args.extend(crate::color::hdr_decode_input_args(color));
    }
    if !req.apply_rotation {
        args.push("-noautorotate".to_string());
    }
    args.push("-i".to_string());
    args.push(path_to_string(&req.path));
    args.push("-map".to_string());
    args.push("0:v:0".to_string());
    args.push("-an".to_string());
    args.push("-sn".to_string());

    if let Some(frame_limit) = req.frame_limit() {
        args.push("-frames:v".to_string());
        args.push(frame_limit.to_string());
    }

    // `round=up` maps a source frame to the first slot at/after its pts, so
    // slot F shows the last frame with pts <= F / fps. The first decoded frame
    // is pulled back to half a slot before `start_frame` when it starts later
    // (late first frame), then `trim` drops the slots before `start_frame`.
    let lead_secs = (req.start_frame as f64 - 0.5).max(0.0) / req.timeline_fps as f64;
    let mut filters = vec![
        format!("setpts='if(eq(N,0),min(PTS,{lead_secs:.6}/TB),PTS)'"),
        format!("fps=fps={}:round=up", req.timeline_fps),
        format!("trim=start_pts={}", req.start_frame),
    ];
    if let Some(filter) = color.and_then(crate::color::hdr_tonemap_filter) {
        filters.push(filter);
    }
    if req.max_size.0 > 0 || req.max_size.1 > 0 {
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
    args.push("-vf".to_string());
    args.push(filters.join(","));
    args.push("-fps_mode".to_string());
    args.push("passthrough".to_string());
    args.push("-pix_fmt".to_string());
    args.push("rgba".to_string());
    args.push("-f".to_string());
    args.push("rawvideo".to_string());
    args.push("-".to_string());
    args
}

fn path_to_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> VideoStreamRequest {
        VideoStreamRequest {
            path: PathBuf::from("/x/clip.mp4"),
            start_frame: 60,
            end_frame: Some(65),
            timeline_fps: 30,
            max_size: (1280, 720),
            queue_capacity: 2,
            apply_rotation: true,
        }
    }

    fn black_output(frame_num: u32) -> OutputVideoFrame {
        OutputVideoFrame {
            width: 2,
            height: 1,
            pix_fmt: "rgba".to_string(),
            output_index: 0,
            data: vec![0, 0, 0, 255, 1, 1, 1, 255],
            frame_num,
            timestamp: frame_num as f32 / 30.0,
        }
    }

    #[test]
    fn request_defaults_to_bounded_queue_and_rotation() {
        let req = VideoStreamRequest::new("/x/clip.mp4", 30);
        assert_eq!(req.start_frame, 0);
        assert_eq!(req.end_frame, None);
        assert_eq!(req.queue_capacity, DEFAULT_VIDEO_STREAM_QUEUE_CAPACITY);
        assert!(req.apply_rotation);
    }

    #[test]
    fn request_validation_rejects_invalid_frame_clock() {
        let mut req = request();
        req.timeline_fps = 0;
        assert!(req
            .validate()
            .unwrap_err()
            .to_string()
            .contains("timeline_fps"));

        req = request();
        req.start_frame = -1;
        assert!(req
            .validate()
            .unwrap_err()
            .to_string()
            .contains("start_frame"));

        req = request();
        req.end_frame = Some(req.start_frame);
        assert!(req
            .validate()
            .unwrap_err()
            .to_string()
            .contains("end_frame"));

        req = request();
        req.queue_capacity = 0;
        assert!(req
            .validate()
            .unwrap_err()
            .to_string()
            .contains("queue_capacity"));
    }

    #[test]
    fn stream_args_seek_and_limit_are_integer_frame_based() {
        let args = video_stream_args(&request());
        let ss = args.iter().position(|arg| arg == "-ss").unwrap();
        assert_eq!(args[ss + 1], "2.000000");
        for flag in ["-noaccurate_seek", "-copyts", "-start_at_zero"] {
            let at = args.iter().position(|arg| arg == flag).unwrap();
            let input = args.iter().position(|arg| arg == "-i").unwrap();
            assert!(at < input, "{flag} must be an input option");
        }

        let frames = args.iter().position(|arg| arg == "-frames:v").unwrap();
        assert_eq!(args[frames + 1], "5");
    }

    #[test]
    fn stream_args_force_project_fps_rgba_rawvideo() {
        let args = video_stream_args(&request());
        let vf = args.iter().position(|arg| arg == "-vf").unwrap();
        assert!(args[vf + 1].starts_with(
            "setpts='if(eq(N,0),min(PTS,1.983333/TB),PTS)',fps=fps=30:round=up,trim=start_pts=60,"
        ));
        assert!(args[vf + 1].contains("force_original_aspect_ratio=decrease"));
        assert!(args.windows(2).any(|w| w == ["-fps_mode", "passthrough"]));
        assert!(args.windows(2).any(|w| w == ["-pix_fmt", "rgba"]));
        assert!(args.windows(2).any(|w| w == ["-f", "rawvideo"]));
        assert_eq!(args.last().unwrap(), "-");
    }

    #[test]
    fn stream_args_can_disable_autorotate() {
        let mut req = request();
        req.apply_rotation = false;
        let args = video_stream_args(&req);
        assert!(args.iter().any(|arg| arg == "-noautorotate"));
    }

    #[test]
    fn output_frame_maps_to_integer_source_frame_and_pts() {
        let got = stream_frame_from_output(&request(), black_output(3));
        assert_eq!(got.source_frame, 63);
        assert!((got.pts_secs - 2.1).abs() < 0.0001);
        assert_eq!(got.frame.width, 2);
        assert_eq!(got.frame.height, 1);
    }

    #[test]
    fn failure_log_tail_keeps_only_the_most_recent_lines() {
        let mut tail = VecDeque::new();
        for line in 0..FAILURE_LOG_TAIL_LINES + 5 {
            push_log_tail(&mut tail, line.to_string());
        }
        assert_eq!(tail.len(), FAILURE_LOG_TAIL_LINES);
        assert_eq!(tail.front().map(String::as_str), Some("5"));
        assert_eq!(
            tail.back().map(String::as_str),
            Some((FAILURE_LOG_TAIL_LINES + 4).to_string().as_str())
        );
    }

    #[test]
    fn bounded_send_stops_instead_of_waiting_forever() {
        let (tx, _rx) = sync_channel(1);
        let control = StreamDecodeControl {
            stop: Arc::new(AtomicBool::new(false)),
        };
        assert!(send_with_backpressure(
            &tx,
            Ok(stream_frame_from_output(&request(), black_output(0))),
            &control
        ));
        control.request_stop();
        assert!(!send_with_backpressure(
            &tx,
            Ok(stream_frame_from_output(&request(), black_output(1))),
            &control
        ));
    }
}
