//! Audio-track PCM extraction via the system ffmpeg CLI. Replaces upstream
//! `Transcription.extractAudioTrack` (`Transcription.swift:203-280`), which
//! decoded the first audio track to 16 kHz mono s16le.
//!
//! The canonical output for transcription is **16 kHz mono f32**; the buffer
//! always carries an f32 mono view for downstream consumers (whisper). The
//! `PcmFormat` selects the on-wire sample format ffmpeg emits.
//!
//! Decoded PCM is consumed as it streams out of FFmpeg: the stdout reader
//! hands whole sample frames to a [`PcmSink`] chunk by chunk, so callers keep
//! only what they need (one f32 buffer for [`extract_pcm`], per-bucket sums
//! for waveforms) and never a raw byte copy of the whole track next to it.
//!
//! The arg builder ([`pcm_args`]) and the sample conversions are pure and
//! unit-tested; the extraction itself requires ffmpeg.

use std::ffi::OsString;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{ChildStderr, ExitStatus};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::cancel::MediaCancelToken;
use crate::error::{MediaError, Result};
use crate::ff;
use crate::ff::SpawnCounted;
use crate::probe;

/// On-wire PCM sample format requested from ffmpeg.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcmFormat {
    S16Le,
    F32,
}

impl PcmFormat {
    /// ffmpeg `-f` rawvideo-equivalent codec/format token.
    fn ffmpeg_fmt(self) -> &'static str {
        match self {
            PcmFormat::S16Le => "s16le",
            PcmFormat::F32 => "f32le",
        }
    }
    pub(crate) fn bytes_per_sample(self) -> usize {
        match self {
            PcmFormat::S16Le => 2,
            PcmFormat::F32 => 4,
        }
    }

    /// One little-endian sample as a unit-range f32.
    fn sample(self, bytes: &[u8]) -> f32 {
        match self {
            PcmFormat::S16Le => i16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 32768.0,
            PcmFormat::F32 => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        }
    }
}

const CHILD_POLL_INTERVAL: Duration = Duration::from_millis(5);
/// Upper bound on the PCM one whole-track decode may deliver. A whole track's
/// decoded length is only estimated by the container (a VBR MP3 without a
/// Xing/VBRI header is sized from its first frames' bitrate, and audio may
/// outlast the video), so the output is read until EOF and only this ceiling
/// guards against runaway decoder output: about 18 hours of 16 kHz mono f32,
/// or 3 hours of 48 kHz stereo f32.
const WHOLE_TRACK_PCM_MAX_BYTES: u64 = 4 << 30;
const STDERR_DETAIL_LIMIT: usize = 64 * 1024;
const STDOUT_CHUNK_BYTES: usize = 64 * 1024;
const PCM_PROGRESS_TOTAL: usize = 4_000;
const PCM_DECODE_PROGRESS_END: usize = 3_000;

/// Byte-level progress reported while FFmpeg streams decoded PCM to stdout.
pub type PcmProgressCallback = Arc<dyn Fn(usize, usize) + Send + Sync>;

/// Consumer of decoded PCM, fed on the stdout reader thread while FFmpeg is
/// still decoding.
pub(crate) trait PcmSink: Send + 'static {
    /// Consume whole sample frames: `frames.len()` is a multiple of the frame
    /// size of the spec the decode was started with.
    fn push_frames(&mut self, frames: &[u8]) -> Result<()>;
}

/// Collects the mono f32 view: channels are averaged per frame.
#[derive(Debug)]
struct MonoF32Sink {
    spec: PcmSpec,
    samples: Vec<f32>,
}

impl MonoF32Sink {
    fn with_capacity(spec: PcmSpec, frames: usize) -> Result<Self> {
        let mut samples = Vec::new();
        samples
            .try_reserve_exact(frames)
            .map_err(|error| allocation_error(format!("mono f32 reserve {frames}: {error}")))?;
        Ok(MonoF32Sink { spec, samples })
    }
}

impl PcmSink for MonoF32Sink {
    fn push_frames(&mut self, frames: &[u8]) -> Result<()> {
        let bps = self.spec.format.bytes_per_sample();
        let channels = usize::from(self.spec.channels.max(1));
        let frame_bytes = bps * channels;
        let count = frames.len() / frame_bytes;
        self.samples
            .try_reserve(count)
            .map_err(|error| allocation_error(format!("mono f32 grow by {count}: {error}")))?;
        for frame in frames.chunks_exact(frame_bytes) {
            let mut sum = 0.0f32;
            for sample in frame.chunks_exact(bps) {
                sum += self.spec.format.sample(sample);
            }
            self.samples.push(sum / channels as f32);
        }
        Ok(())
    }
}

/// Collects interleaved f32 without folding channels.
pub(crate) struct InterleavedF32Sink {
    format: PcmFormat,
    samples: Vec<f32>,
}

impl InterleavedF32Sink {
    pub(crate) fn with_capacity(format: PcmFormat, samples: usize) -> Result<Self> {
        let mut buffer = Vec::new();
        buffer.try_reserve_exact(samples).map_err(|error| {
            allocation_error(format!("interleaved f32 reserve {samples}: {error}"))
        })?;
        Ok(InterleavedF32Sink {
            format,
            samples: buffer,
        })
    }

    pub(crate) fn into_samples(self) -> Vec<f32> {
        self.samples
    }
}

impl PcmSink for InterleavedF32Sink {
    fn push_frames(&mut self, frames: &[u8]) -> Result<()> {
        let bps = self.format.bytes_per_sample();
        let count = frames.len() / bps;
        self.samples.try_reserve(count).map_err(|error| {
            allocation_error(format!("interleaved f32 grow by {count}: {error}"))
        })?;
        self.samples.extend(
            frames
                .chunks_exact(bps)
                .map(|sample| self.format.sample(sample)),
        );
        Ok(())
    }
}

/// Re-frames arbitrary read chunks into whole sample frames for a sink; a
/// frame split across reads is carried to the next one.
pub(super) struct FrameAligner {
    frame_bytes: usize,
    carry: Vec<u8>,
}

impl FrameAligner {
    pub(super) fn new(frame_bytes: usize) -> Self {
        FrameAligner {
            frame_bytes,
            carry: Vec::with_capacity(frame_bytes),
        }
    }

    pub(super) fn feed(&mut self, mut data: &[u8], sink: &mut impl PcmSink) -> Result<()> {
        if !self.carry.is_empty() {
            let take = (self.frame_bytes - self.carry.len()).min(data.len());
            self.carry.extend_from_slice(&data[..take]);
            data = &data[take..];
            if self.carry.len() < self.frame_bytes {
                return Ok(());
            }
            sink.push_frames(&self.carry)?;
            self.carry.clear();
        }
        let whole = data.len() - data.len() % self.frame_bytes;
        if whole > 0 {
            sink.push_frames(&data[..whole])?;
        }
        // A trailing partial frame at EOF is dropped, as FFmpeg never ends
        // a PCM stream mid-frame on success.
        self.carry.extend_from_slice(&data[whole..]);
        Ok(())
    }
}

struct PipeReaders<S> {
    stdout: JoinHandle<Result<StdoutRead<S>>>,
    stderr: JoinHandle<Result<Vec<u8>>>,
}

struct StdoutRead<S> {
    sink: S,
    exceeded_cap: bool,
    total_read: usize,
}

fn audio_buffer_too_large(detail: impl std::fmt::Display) -> MediaError {
    MediaError::Decode(format!("audio_buffer_too_large: {detail}"))
}

fn allocation_error(detail: impl std::fmt::Display) -> MediaError {
    MediaError::Decode(format!("audio_allocation_failed: {detail}"))
}

fn validate_spec(spec: &PcmSpec) -> Result<()> {
    if spec.sample_rate == 0 || spec.channels == 0 {
        return Err(MediaError::Decode(
            "PCM sample rate and channel count must be non-zero".to_string(),
        ));
    }
    Ok(())
}

fn expected_pcm_bytes_for_duration(duration_secs: f64, spec: &PcmSpec) -> Result<usize> {
    validate_spec(spec)?;
    if !duration_secs.is_finite() {
        return Err(audio_buffer_too_large("non-finite duration"));
    }
    let frames = (duration_secs * f64::from(spec.sample_rate)).ceil();
    if frames > usize::MAX as f64 {
        return Err(audio_buffer_too_large("PCM frame count exceeds usize"));
    }
    let frame_bytes = usize::from(spec.channels)
        .checked_mul(spec.format.bytes_per_sample())
        .ok_or_else(|| audio_buffer_too_large("PCM frame byte count overflow"))?;
    (frames as usize)
        .checked_mul(frame_bytes)
        .ok_or_else(|| audio_buffer_too_large("PCM output byte count overflow"))
}

fn expected_pcm_bytes_for_range(range: (f64, f64), spec: &PcmSpec) -> Result<usize> {
    let (lo, hi) = range;
    expected_pcm_bytes_for_duration((hi - lo.max(0.0)).max(0.0), spec)
}

/// Seconds a whole-track decode is expected to produce: the longer of the
/// clip duration (video first) and the longest audio stream.
fn whole_track_estimate_secs(media: &probe::MediaProbe) -> f64 {
    let video_or_container = media.duration_secs;
    let audio = media.audio_duration_secs.unwrap_or(0.0);
    let estimate = video_or_container.max(audio);
    if estimate.is_finite() && estimate > 0.0 {
        estimate
    } else {
        0.0
    }
}

/// How much decoded PCM the stdout reader accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ReadLimit {
    /// Reading past this many bytes fails the decode with
    /// `audio_buffer_too_large`.
    cap: usize,
    /// Expected output size: the sink's initial reservation (its buffers
    /// still grow past it, up to `cap`) and the progress total.
    expected: usize,
}

/// Stream stdout into `sink` in bounded chunks. Only one read chunk and at
/// most one partial frame are buffered here; the sink owns everything else.
fn read_stdout<S: PcmSink>(
    mut stdout: impl Read,
    mut sink: S,
    frame_bytes: usize,
    limit: ReadLimit,
    cancel: MediaCancelToken,
    progress: Option<PcmProgressCallback>,
) -> Result<StdoutRead<S>> {
    cancel.reader_started();
    let result = (|| {
        let mut aligner = FrameAligner::new(frame_bytes);
        let mut exceeded_cap = false;
        let mut total_read = 0_usize;
        let mut accepted = 0_usize;
        let mut chunk = vec![0_u8; STDOUT_CHUNK_BYTES];
        loop {
            if cancel.is_cancelled() {
                return Err(MediaError::Cancelled);
            }
            let read = stdout
                .read(&mut chunk)
                .map_err(|error| MediaError::Ffmpeg(format!("read stdout: {error}")))?;
            if read == 0 {
                break;
            }
            total_read = total_read.saturating_add(read);
            if let Some(report) = &progress {
                report(total_read.min(limit.expected), limit.expected);
            }
            let retained = limit.cap.saturating_sub(accepted).min(read);
            aligner.feed(&chunk[..retained], &mut sink)?;
            accepted += retained;
            if retained < read {
                // Stop at the cap instead of draining a runaway stream: the
                // closed pipe ends the decoder, and the error is reported
                // once it has been reaped.
                exceeded_cap = true;
                break;
            }
        }
        Ok(StdoutRead {
            sink,
            exceeded_cap,
            total_read,
        })
    })();
    cancel.reader_finished();
    result
}

fn read_stderr(mut stderr: ChildStderr, cancel: MediaCancelToken) -> Result<Vec<u8>> {
    cancel.reader_started();
    let result = (|| {
        let mut detail = Vec::new();
        detail
            .try_reserve_exact(STDERR_DETAIL_LIMIT)
            .map_err(|error| allocation_error(format!("stderr reserve: {error}")))?;
        let mut chunk = [0_u8; 8 * 1024];
        loop {
            let read = stderr
                .read(&mut chunk)
                .map_err(|error| MediaError::Ffmpeg(format!("read stderr: {error}")))?;
            if read == 0 {
                break;
            }
            let retained = STDERR_DETAIL_LIMIT.saturating_sub(detail.len()).min(read);
            detail.extend_from_slice(&chunk[..retained]);
        }
        Ok(detail)
    })();
    cancel.reader_finished();
    result
}

fn join_reader<T>(handle: JoinHandle<Result<T>>, name: &str) -> Result<T> {
    handle
        .join()
        .map_err(|_| MediaError::Ffmpeg(format!("{name} reader panicked")))?
}

fn join_pipes<S>(readers: PipeReaders<S>) -> Result<(StdoutRead<S>, Vec<u8>)> {
    // Join both handles before propagating either failure. Returning after the
    // first failed join would detach the other pipe reader and could keep the
    // FFmpeg pipe (and its allocation) alive beyond this decode request.
    let stdout = join_reader(readers.stdout, "stdout");
    let stderr = join_reader(readers.stderr, "stderr");
    match (stdout, stderr) {
        (Ok(stdout), Ok(stderr)) => Ok((stdout, stderr)),
        (Err(error), _) | (_, Err(error)) => Err(error),
    }
}

fn terminate_child(child: &mut ffmpeg_sidecar::child::FfmpegChild) {
    let _ = child.kill();
    let _ = child.wait();
}

fn wait_for_pcm_child<S>(
    child: &mut ffmpeg_sidecar::child::FfmpegChild,
    readers: PipeReaders<S>,
    cancel: &MediaCancelToken,
) -> Result<(ExitStatus, StdoutRead<S>, Vec<u8>)> {
    loop {
        if cancel.checkpoint() {
            terminate_child(child);
            let _ = join_pipes(readers);
            return Err(MediaError::Cancelled);
        }
        let status = match child.as_inner_mut().try_wait() {
            Ok(status) => status,
            Err(error) => {
                terminate_child(child);
                let _ = join_pipes(readers);
                return Err(MediaError::Io(error));
            }
        };
        if let Some(status) = status {
            let (stdout, stderr) = join_pipes(readers)?;
            if cancel.is_cancelled() {
                return Err(MediaError::Cancelled);
            }
            return Ok((status, stdout, stderr));
        }
        thread::sleep(CHILD_POLL_INTERVAL);
    }
}

fn validate_pcm_output<S>(
    path: &Path,
    status: ExitStatus,
    stdout: StdoutRead<S>,
    stderr: Vec<u8>,
    reader_cap: usize,
) -> Result<S> {
    if stdout.exceeded_cap {
        return Err(audio_buffer_too_large(format!(
            "FFmpeg PCM output exceeded the {reader_cap}-byte limit after {} bytes",
            stdout.total_read
        )));
    }
    if !status.success() {
        let detail = String::from_utf8_lossy(&stderr);
        let suffix = if detail.trim().is_empty() {
            String::new()
        } else {
            format!(": {}", detail.trim())
        };
        return Err(MediaError::Ffmpeg(format!(
            "decode exited {status}{suffix}"
        )));
    }
    if stdout.total_read == 0 {
        return Err(MediaError::no_track("audio", path));
    }
    Ok(stdout.sink)
}

/// Requested PCM layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PcmSpec {
    pub sample_rate: u32,
    pub channels: u16,
    pub format: PcmFormat,
}

/// Decoded PCM. `samples_f32` is always a mono f32 view (downstream-friendly);
/// when the requested spec has multiple channels they are averaged into mono.
#[derive(Clone, Debug, PartialEq)]
pub struct PcmBuffer {
    pub spec: PcmSpec,
    pub samples_f32: Vec<f32>,
}

impl PcmBuffer {
    /// Duration in seconds implied by the mono sample count and sample rate.
    pub fn duration_secs(&self) -> f64 {
        if self.spec.sample_rate == 0 {
            return 0.0;
        }
        self.samples_f32.len() as f64 / self.spec.sample_rate as f64
    }
}

/// Build the ffmpeg arg list for decoding the first audio track to raw PCM on
/// stdout, honoring an optional `[lo, hi)` absolute-seconds range.
fn pcm_args(path: &Path, spec: &PcmSpec, range: Option<(f64, f64)>) -> Vec<OsString> {
    let mut args: Vec<OsString> = Vec::new();
    if let Some((lo, hi)) = range {
        args.push("-ss".into());
        args.push(format!("{:.6}", lo.max(0.0)).into());
        args.push("-to".into());
        args.push(format!("{hi:.6}").into());
    }
    args.push("-i".into());
    args.push(path.as_os_str().to_owned());
    args.push("-vn".into()); // drop video
    args.push("-ac".into());
    args.push(spec.channels.to_string().into());
    args.push("-ar".into());
    args.push(spec.sample_rate.to_string().into());
    args.push("-f".into());
    args.push(spec.format.ffmpeg_fmt().into());
    args.push("-".into());
    args
}

fn bounded_pcm_args(
    path: &Path,
    spec: &PcmSpec,
    range: Option<(f64, f64)>,
    max_frames: usize,
) -> Vec<OsString> {
    let mut args = pcm_args(path, spec, range);
    args.pop(); // stdout destination follows all output options
    args.extend([
        OsString::from("-af"),
        // AAC decoders may retain a padded final packet beyond the container's
        // duration. Enforce the same frame budget as the stdout reader, after
        // conversion to the requested sample rate, without relaxing that cap.
        format!(
            "aresample={},atrim=end_sample={max_frames}",
            spec.sample_rate
        )
        .into(),
        "-".into(),
    ]);
    args
}

/// Decode `path`'s first audio track to the requested PCM spec, returning a mono
/// f32 buffer. `range` is an absolute-seconds `[lo, hi)` window. Errors with
/// `NoTrack("audio", …)` when the file has no audio stream.
pub fn extract_pcm(path: &Path, spec: &PcmSpec, range: Option<(f64, f64)>) -> Result<PcmBuffer> {
    extract_pcm_cancellable(path, spec, range, &MediaCancelToken::new())
}

pub fn extract_pcm_cancellable(
    path: &Path,
    spec: &PcmSpec,
    range: Option<(f64, f64)>,
    cancel: &MediaCancelToken,
) -> Result<PcmBuffer> {
    extract_pcm_cancellable_with_progress(path, spec, range, cancel, None)
}

pub fn extract_pcm_cancellable_with_progress(
    path: &Path,
    spec: &PcmSpec,
    range: Option<(f64, f64)>,
    cancel: &MediaCancelToken,
    progress: Option<PcmProgressCallback>,
) -> Result<PcmBuffer> {
    let decode_progress = progress.as_ref().map(|report| {
        let report = Arc::clone(report);
        Arc::new(move |done: usize, total: usize| {
            let mapped = done
                .min(total.max(1))
                .saturating_mul(PCM_DECODE_PROGRESS_END)
                / total.max(1);
            report(mapped, PCM_PROGRESS_TOTAL);
        }) as PcmProgressCallback
    });
    let sink = decode_pcm_streaming(path, spec, range, cancel, decode_progress, |frames| {
        MonoF32Sink::with_capacity(*spec, frames)
    })?;
    if cancel.checkpoint() {
        return Err(MediaError::Cancelled);
    }
    if let Some(report) = &progress {
        report(PCM_PROGRESS_TOTAL, PCM_PROGRESS_TOTAL);
    }
    Ok(PcmBuffer {
        spec: *spec,
        samples_f32: sink.samples,
    })
}

/// Decode `path`'s first audio track as `spec` and stream it into the sink
/// built by `make_sink`, which receives the expected frame count: the exact
/// frame budget for a `range`, or the probe's estimate for a whole track.
pub(crate) fn decode_pcm_streaming<S: PcmSink>(
    path: &Path,
    spec: &PcmSpec,
    range: Option<(f64, f64)>,
    cancel: &MediaCancelToken,
    progress: Option<PcmProgressCallback>,
    make_sink: impl FnOnce(usize) -> Result<S>,
) -> Result<S> {
    let ceiling = usize::try_from(WHOLE_TRACK_PCM_MAX_BYTES).unwrap_or(usize::MAX);
    decode_pcm_streaming_with_ceiling(path, spec, range, cancel, progress, make_sink, ceiling)
}

fn decode_pcm_streaming_with_ceiling<S: PcmSink>(
    path: &Path,
    spec: &PcmSpec,
    range: Option<(f64, f64)>,
    cancel: &MediaCancelToken,
    progress: Option<PcmProgressCallback>,
    make_sink: impl FnOnce(usize) -> Result<S>,
    whole_track_ceiling: usize,
) -> Result<S> {
    if cancel.is_cancelled() {
        return Err(MediaError::Cancelled);
    }
    validate_spec(spec)?;
    // One probe per ordinary file: it rejects files without audio up front
    // and estimates how long a whole-track decode will be.
    let probed = if path.is_file() {
        let media = probe::probe(path)?;
        if !media.has_audio {
            return Err(MediaError::no_track("audio", path));
        }
        Some(media)
    } else {
        None
    };
    let frame_bytes = usize::from(spec.channels)
        .checked_mul(spec.format.bytes_per_sample())
        .ok_or_else(|| audio_buffer_too_large("PCM frame byte count overflow"))?;
    let (args, limit) = match range {
        Some(range) => {
            // Explicit ranges keep one frame of rounding slack and are trimmed
            // at the requested output sample rate.
            let expected = expected_pcm_bytes_for_range(range, spec)?;
            let cap = expected
                .checked_add(frame_bytes)
                .ok_or_else(|| audio_buffer_too_large("PCM reader cap overflow"))?;
            (
                bounded_pcm_args(path, spec, Some(range), expected / frame_bytes),
                ReadLimit { cap, expected },
            )
        }
        None => {
            // A whole track decodes to EOF. Container durations are estimates
            // that can be far too short, so they only size the initial buffer
            // and the progress total; the reader grows up to the ceiling.
            let estimate = probed.as_ref().map_or(0.0, whole_track_estimate_secs);
            // An estimate too large to represent reserves nothing up front.
            let expected = expected_pcm_bytes_for_duration(estimate, spec)
                .unwrap_or(0)
                .min(whole_track_ceiling);
            (
                pcm_args(path, spec, None),
                ReadLimit {
                    cap: whole_track_ceiling,
                    expected,
                },
            )
        }
    };
    let sink = make_sink(limit.expected / frame_bytes)?;
    let mut child = ff::ffmpeg_decode(args, path)?
        .spawn_counted()
        .map_err(|e| MediaError::Ffmpeg(format!("spawn: {e}")))?;
    cancel.child_spawned();
    let stdout = match child.take_stdout() {
        Some(stdout) => stdout,
        None => {
            terminate_child(&mut child);
            return Err(MediaError::Ffmpeg("FFmpeg stdout pipe missing".to_string()));
        }
    };
    let stderr = match child.take_stderr() {
        Some(stderr) => stderr,
        None => {
            terminate_child(&mut child);
            return Err(MediaError::Ffmpeg("FFmpeg stderr pipe missing".to_string()));
        }
    };
    let stdout_cancel = cancel.clone();
    let stderr_cancel = cancel.clone();
    let stdout_reader = match thread::Builder::new()
        .name("opentake-pcm-stdout".to_string())
        .spawn(move || read_stdout(stdout, sink, frame_bytes, limit, stdout_cancel, progress))
    {
        Ok(reader) => reader,
        Err(error) => {
            terminate_child(&mut child);
            return Err(MediaError::Ffmpeg(format!("spawn stdout reader: {error}")));
        }
    };
    let stderr_reader = match thread::Builder::new()
        .name("opentake-pcm-stderr".to_string())
        .spawn(move || read_stderr(stderr, stderr_cancel))
    {
        Ok(reader) => reader,
        Err(error) => {
            terminate_child(&mut child);
            let _ = join_reader(stdout_reader, "stdout");
            return Err(MediaError::Ffmpeg(format!("spawn stderr reader: {error}")));
        }
    };
    let readers = PipeReaders {
        stdout: stdout_reader,
        stderr: stderr_reader,
    };
    let (status, stdout, stderr) = wait_for_pcm_child(&mut child, readers, cancel)?;
    validate_pcm_output(path, status, stdout, stderr, limit.cap)
}

/// Converted chunks queued between a [`PcmStream`]'s stdout reader and its
/// consumer. Each holds at most one stdout read, so this bounds the decoded
/// audio held ahead of the consumer; FFmpeg then blocks on its full pipe.
const PCM_STREAM_QUEUE_CHUNKS: usize = 4;

/// Hands converted f32 chunks to a [`PcmStream`] consumer through a bounded
/// queue, blocking while the consumer is behind. [`PcmStream::shutdown`]
/// drops the receiver before joining, which fails a blocked send.
struct QueueSink {
    format: PcmFormat,
    sender: SyncSender<Vec<f32>>,
    stop: Arc<AtomicBool>,
    cancel: MediaCancelToken,
}

impl PcmSink for QueueSink {
    fn push_frames(&mut self, frames: &[u8]) -> Result<()> {
        let bps = self.format.bytes_per_sample();
        let mut chunk = Vec::new();
        chunk
            .try_reserve_exact(frames.len() / bps)
            .map_err(|error| allocation_error(format!("PCM stream chunk: {error}")))?;
        chunk.extend(
            frames
                .chunks_exact(bps)
                .map(|sample| self.format.sample(sample)),
        );
        if self.stop.load(Ordering::Acquire) || self.cancel.is_cancelled() {
            return Err(MediaError::Cancelled);
        }
        // Fails only once the consumer has shut the stream down.
        self.sender.send(chunk).map_err(|_| MediaError::Cancelled)
    }
}

/// One forward decode of a bounded source range, read incrementally: a single
/// FFmpeg process serves the whole range however many reads the caller makes,
/// so a caller that walks a long range in windows does not start a decoder
/// (and re-seek) per window. Samples are interleaved f32 in the requested
/// channel layout and rate.
///
/// Unlike [`extract_pcm`] this does not probe the input: the caller must know
/// the file has an audio track (without one, the decode fails). The caller's
/// token cancels the decode and counts its process; dropping the stream stops
/// and reaps the decoder.
pub struct PcmStream {
    path: PathBuf,
    channels: usize,
    cancel: MediaCancelToken,
    stop: Arc<AtomicBool>,
    child: Option<ffmpeg_sidecar::child::FfmpegChild>,
    stdout_reader: Option<JoinHandle<Result<(bool, usize)>>>,
    stderr_reader: Option<JoinHandle<Result<Vec<u8>>>>,
    /// `None` once shut down, which unblocks the stdout reader's send.
    receiver: Option<Receiver<Vec<f32>>>,
    pending: Vec<f32>,
    pending_offset: usize,
    reader_cap: usize,
    ended: bool,
}

impl PcmStream {
    /// Start decoding `range` (absolute source seconds, `[lo, hi)`) of
    /// `path`'s first audio track. The output is trimmed to the range's frame
    /// budget at the requested rate, as for a ranged [`extract_pcm`].
    pub fn open(
        path: &Path,
        spec: &PcmSpec,
        range: (f64, f64),
        cancel: &MediaCancelToken,
    ) -> Result<Self> {
        if cancel.is_cancelled() {
            return Err(MediaError::Cancelled);
        }
        validate_spec(spec)?;
        let channels = usize::from(spec.channels);
        let frame_bytes = channels
            .checked_mul(spec.format.bytes_per_sample())
            .ok_or_else(|| audio_buffer_too_large("PCM frame byte count overflow"))?;
        let expected = expected_pcm_bytes_for_range(range, spec)?;
        let cap = expected
            .checked_add(frame_bytes)
            .ok_or_else(|| audio_buffer_too_large("PCM reader cap overflow"))?;
        let limit = ReadLimit { cap, expected };
        let mut child = ff::ffmpeg_decode(
            bounded_pcm_args(path, spec, Some(range), expected / frame_bytes),
            path,
        )?
        .spawn_counted()
        .map_err(|e| MediaError::Ffmpeg(format!("spawn: {e}")))?;
        cancel.child_spawned();
        let Some(stdout) = child.take_stdout() else {
            terminate_child(&mut child);
            return Err(MediaError::Ffmpeg("FFmpeg stdout pipe missing".to_string()));
        };
        let Some(stderr) = child.take_stderr() else {
            terminate_child(&mut child);
            return Err(MediaError::Ffmpeg("FFmpeg stderr pipe missing".to_string()));
        };
        let (sender, receiver) = sync_channel(PCM_STREAM_QUEUE_CHUNKS);
        let stop = Arc::new(AtomicBool::new(false));
        let sink = QueueSink {
            format: spec.format,
            sender,
            stop: Arc::clone(&stop),
            cancel: cancel.clone(),
        };
        let stdout_cancel = cancel.clone();
        let stdout_reader = match thread::Builder::new()
            .name("opentake-pcm-stream".to_string())
            .spawn(move || {
                // Return only the summary: the sink, and with it the queue's
                // sender, drops here so the consumer sees the end of stream.
                read_stdout(stdout, sink, frame_bytes, limit, stdout_cancel, None)
                    .map(|read| (read.exceeded_cap, read.total_read))
            }) {
            Ok(reader) => reader,
            Err(error) => {
                terminate_child(&mut child);
                return Err(MediaError::Ffmpeg(format!("spawn stdout reader: {error}")));
            }
        };
        let stderr_cancel = cancel.clone();
        let stderr_reader = match thread::Builder::new()
            .name("opentake-pcm-stderr".to_string())
            .spawn(move || read_stderr(stderr, stderr_cancel))
        {
            Ok(reader) => reader,
            Err(error) => {
                stop.store(true, Ordering::Release);
                terminate_child(&mut child);
                let _ = join_reader(stdout_reader, "stdout");
                return Err(MediaError::Ffmpeg(format!("spawn stderr reader: {error}")));
            }
        };
        Ok(PcmStream {
            path: path.to_path_buf(),
            channels,
            cancel: cancel.clone(),
            stop,
            child: Some(child),
            stdout_reader: Some(stdout_reader),
            stderr_reader: Some(stderr_reader),
            receiver: Some(receiver),
            pending: Vec::new(),
            pending_offset: 0,
            reader_cap: cap,
            ended: false,
        })
    }

    /// Interleaved channels per frame.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Append up to `frames` decoded frames to `out`, waiting until that many
    /// are available or the decode ends. Returns the number appended; fewer
    /// than requested means the range is exhausted and FFmpeg exited cleanly.
    /// A failed or cancelled decode is an error.
    pub fn read(&mut self, frames: usize, out: &mut Vec<f32>) -> Result<usize> {
        let wanted = frames.saturating_mul(self.channels);
        out.try_reserve(wanted)
            .map_err(|error| allocation_error(format!("PCM stream read: {error}")))?;
        let mut appended = 0_usize;
        while appended < wanted {
            let available = self.pending.len() - self.pending_offset;
            if available > 0 {
                let take = available.min(wanted - appended);
                out.extend_from_slice(
                    &self.pending[self.pending_offset..self.pending_offset + take],
                );
                self.pending_offset += take;
                appended += take;
                continue;
            }
            if self.ended {
                break;
            }
            // Checked before every chunk: a fast decoder keeps the queue full,
            // so the timeout below may never fire.
            if self.cancel.is_cancelled() {
                self.shutdown();
                return Err(MediaError::Cancelled);
            }
            let Some(receiver) = self.receiver.as_ref() else {
                break;
            };
            match receiver.recv_timeout(CHILD_POLL_INTERVAL) {
                Ok(chunk) => {
                    self.pending = chunk;
                    self.pending_offset = 0;
                }
                Err(RecvTimeoutError::Timeout) => {
                    if self.cancel.checkpoint() {
                        self.shutdown();
                        return Err(MediaError::Cancelled);
                    }
                }
                Err(RecvTimeoutError::Disconnected) => self.finish()?,
            }
        }
        Ok(appended / self.channels)
    }

    /// The stdout reader is done: reap FFmpeg and check how the decode ended.
    fn finish(&mut self) -> Result<()> {
        self.ended = true;
        let stdout = match self.stdout_reader.take() {
            Some(reader) => join_reader(reader, "stdout"),
            None => Ok((false, 0)),
        };
        let Some(mut child) = self.child.take() else {
            return Err(MediaError::Ffmpeg("PCM stream decoder missing".to_string()));
        };
        let status = loop {
            if self.cancel.checkpoint() {
                terminate_child(&mut child);
                self.join_stderr();
                return Err(MediaError::Cancelled);
            }
            match child.as_inner_mut().try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) => thread::sleep(CHILD_POLL_INTERVAL),
                Err(error) => {
                    terminate_child(&mut child);
                    self.join_stderr();
                    return Err(MediaError::Io(error));
                }
            }
        };
        let stderr = match self.stderr_reader.take() {
            Some(reader) => join_reader(reader, "stderr"),
            None => Ok(Vec::new()),
        };
        let (exceeded_cap, total_read) = stdout?;
        let stderr = stderr?;
        if exceeded_cap {
            return Err(audio_buffer_too_large(format!(
                "FFmpeg PCM output exceeded the {}-byte limit after {total_read} bytes",
                self.reader_cap
            )));
        }
        if !status.success() {
            let detail = String::from_utf8_lossy(&stderr);
            let suffix = if detail.trim().is_empty() {
                String::new()
            } else {
                format!(": {}", detail.trim())
            };
            return Err(MediaError::Ffmpeg(format!(
                "decode of {} exited {status}{suffix}",
                self.path.display()
            )));
        }
        Ok(())
    }

    fn join_stderr(&mut self) {
        if let Some(reader) = self.stderr_reader.take() {
            let _ = join_reader(reader, "stderr");
        }
    }

    /// Stop the decoder and join both pipe readers.
    fn shutdown(&mut self) {
        self.ended = true;
        self.stop.store(true, Ordering::Release);
        // Fails a send the stdout reader is blocked in.
        self.receiver = None;
        if let Some(mut child) = self.child.take() {
            terminate_child(&mut child);
        }
        if let Some(reader) = self.stdout_reader.take() {
            let _ = join_reader(reader, "stdout");
        }
        self.join_stderr();
    }
}

impl Drop for PcmStream {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    use std::process::Command;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    use crate::MediaCancelToken;

    fn f32_mono_spec() -> PcmSpec {
        PcmSpec {
            sample_rate: 48_000,
            channels: 1,
            format: PcmFormat::F32,
        }
    }

    fn write_silence_wav(path: &Path, sample_rate: u32, samples: usize) {
        let data_len = samples.checked_mul(2).expect("wav data length") as u32;
        let mut wav = Vec::with_capacity(44 + data_len as usize);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_len).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16_u32.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        wav.extend_from_slice(&2_u16.to_le_bytes());
        wav.extend_from_slice(&16_u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_len.to_le_bytes());
        wav.resize(44 + data_len as usize, 0);
        std::fs::write(path, wav).expect("write wav fixture");
    }

    /// Mono s16 WAV whose sample `i` is `(i % 1000) * 16 - 8000`.
    fn write_ramp_wav(path: &Path, sample_rate: u32, samples: usize) {
        write_silence_wav(path, sample_rate, samples);
        let mut wav = std::fs::read(path).expect("read wav fixture");
        for (index, sample) in wav[44..].as_chunks_mut::<2>().0.iter_mut().enumerate() {
            let value = ((index % 1000) as i16) * 16 - 8000;
            sample.copy_from_slice(&value.to_le_bytes());
        }
        std::fs::write(path, wav).expect("write ramp wav fixture");
    }

    #[test]
    fn pcm_stream_reads_a_range_incrementally_with_one_decoder() {
        assert!(crate::ff::ffmpeg_available(), "requires runnable FFmpeg");
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("ramp.wav");
        write_ramp_wav(&input, 48_000, 96_000);
        let spec = f32_mono_spec();
        let range = (0.25, 1.75);
        let whole = extract_pcm(&input, &spec, Some(range)).unwrap().samples_f32;
        assert_eq!(whole.len(), 72_000);

        let cancel = MediaCancelToken::new();
        let before = crate::ff::test_seams::probe_requests();
        let mut stream = PcmStream::open(&input, &spec, range, &cancel).unwrap();
        let mut streamed = Vec::new();
        for frames in [1, 999, 4_096, 17, 30_000].into_iter().cycle() {
            if stream.read(frames, &mut streamed).unwrap() < frames {
                break;
            }
        }
        assert_eq!(streamed, whole, "streamed reads equal one ranged decode");
        assert_eq!(
            stream.read(10, &mut streamed).unwrap(),
            0,
            "range exhausted"
        );
        drop(stream);
        assert_eq!(cancel.spawned_child_count(), 1, "one decoder for the range");
        assert_eq!(
            crate::ff::test_seams::probe_requests(),
            before,
            "the caller already knows the source has audio"
        );
        assert_eq!(cancel.active_reader_count(), 0);
    }

    #[test]
    fn dropping_or_cancelling_a_pcm_stream_reaps_its_decoder() {
        assert!(crate::ff::ffmpeg_available(), "requires runnable FFmpeg");
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("long.wav");
        write_ramp_wav(&input, 48_000, 48_000 * 20);
        let spec = f32_mono_spec();

        let cancel = MediaCancelToken::new();
        let mut stream = PcmStream::open(&input, &spec, (0.0, 20.0), &cancel).unwrap();
        let mut first = Vec::new();
        assert_eq!(stream.read(1_000, &mut first).unwrap(), 1_000);
        let started = Instant::now();
        drop(stream);
        assert!(started.elapsed() < Duration::from_secs(5));
        assert_eq!(cancel.active_reader_count(), 0, "readers joined on drop");

        let cancel = MediaCancelToken::new();
        let mut stream = PcmStream::open(&input, &spec, (0.0, 20.0), &cancel).unwrap();
        let mut samples = Vec::new();
        assert_eq!(stream.read(1_000, &mut samples).unwrap(), 1_000);
        cancel.cancel();
        let result = loop {
            // Queued chunks may still drain before the cancellation lands.
            match stream.read(48_000, &mut samples) {
                Ok(0) => break Ok(0),
                Ok(_) => continue,
                Err(error) => break Err(error),
            }
        };
        assert!(matches!(result, Err(MediaError::Cancelled)), "{result:?}");
        drop(stream);
        assert_eq!(cancel.active_reader_count(), 0);
    }

    #[test]
    fn whole_track_decode_probes_the_source_once() {
        assert!(crate::ff::ffmpeg_available(), "requires runnable FFmpeg");
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("one-second.wav");
        write_silence_wav(&input, 16_000, 16_000);
        let spec = PcmSpec {
            sample_rate: 16_000,
            channels: 1,
            format: PcmFormat::F32,
        };

        let before = crate::ff::test_seams::probe_requests();
        let pcm = extract_pcm(&input, &spec, None).unwrap();

        assert_eq!(crate::ff::test_seams::probe_requests() - before, 1);
        assert_eq!(pcm.samples_f32.len(), 16_000);
    }

    #[test]
    fn whole_track_output_past_the_absolute_ceiling_is_rejected() {
        assert!(crate::ff::ffmpeg_available(), "requires runnable FFmpeg");
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("two-seconds.wav");
        write_silence_wav(&input, 48_000, 96_000);
        let cancel = MediaCancelToken::new();
        let started = Instant::now();

        let error = decode_pcm_streaming_with_ceiling(
            &input,
            &f32_mono_spec(),
            None,
            &cancel,
            None,
            |frames| MonoF32Sink::with_capacity(f32_mono_spec(), frames),
            64 * 1024,
        )
        .expect_err("output past the ceiling must fail");

        assert!(
            error.to_string().contains("audio_buffer_too_large"),
            "{error}"
        );
        // The reader stops at the ceiling and the decoder is reaped.
        assert!(started.elapsed() < Duration::from_secs(10));
        assert_eq!(cancel.spawned_child_count(), 1);
        assert_eq!(cancel.active_reader_count(), 0);
    }

    #[test]
    fn pre_cancelled_pcm_decode_does_not_spawn_ffmpeg() {
        let cancel = MediaCancelToken::new();
        cancel.cancel();

        let error = extract_pcm_cancellable(
            Path::new("/definitely/missing/pre-cancelled.wav"),
            &f32_mono_spec(),
            Some((0.0, 1.0)),
            &cancel,
        )
        .expect_err("pre-cancelled decode must fail before path probing or spawn");

        assert!(matches!(error, MediaError::Cancelled));
        assert_eq!(cancel.spawned_child_count(), 0);
    }

    #[test]
    fn output_frame_limit_trims_padding_after_resampling() {
        assert!(crate::ff::ffmpeg_available(), "requires runnable FFmpeg");
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("padded.wav");
        // Supply 2 seconds at 48k, but permit only the reported 1 second at
        // 16k. Trimming before resampling would return just 5,333 samples.
        write_silence_wav(&input, 48_000, 96_000);
        let spec = PcmSpec {
            sample_rate: 16_000,
            channels: 1,
            format: PcmFormat::F32,
        };
        let output = std::process::Command::new(crate::ff::ffmpeg_path())
            .args(bounded_pcm_args(&input, &spec, None, 16_000))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(output.stdout.len(), 16_000 * 4);
    }

    #[test]
    fn pcm_decode_reports_non_terminal_progress_for_multiple_stdout_chunks() {
        assert!(
            crate::ff::ffmpeg_available(),
            "required progress test needs a runnable FFmpeg"
        );
        let temp = tempfile::tempdir().expect("create progress fixture directory");
        let input = temp.path().join("two-seconds.wav");
        write_silence_wav(&input, 48_000, 96_000);
        let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
        let callback_observed = Arc::clone(&observed);
        let progress: PcmProgressCallback = Arc::new(move |done, total| {
            callback_observed
                .lock()
                .expect("progress lock")
                .push((done, total));
        });

        let pcm = extract_pcm_cancellable_with_progress(
            &input,
            &f32_mono_spec(),
            Some((0.0, 2.0)),
            &MediaCancelToken::new(),
            Some(progress),
        )
        .expect("decode progress fixture");

        let observed = observed.lock().expect("progress lock");
        assert_eq!(pcm.samples_f32.len(), 96_000);
        assert!(
            observed.len() > 1,
            "large decode must report multiple chunks"
        );
        assert!(observed.iter().any(|(done, total)| done < total));
        assert_eq!(
            observed.last(),
            Some(&(PCM_PROGRESS_TOTAL, PCM_PROGRESS_TOTAL))
        );
        assert!(observed.windows(2).all(|pair| pair[0].0 <= pair[1].0));
    }

    #[cfg(unix)]
    #[test]
    fn cancelling_running_pcm_decode_kills_child_and_reaps_readers() {
        assert!(
            crate::ff::ffmpeg_available(),
            "required cancellation test needs a runnable FFmpeg"
        );
        let temp = tempfile::tempdir().expect("create cancellation fixture directory");
        let fifo = temp.path().join("blocking-input.wav");
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
                extract_pcm_cancellable(&fifo, &f32_mono_spec(), Some((0.0, 30.0)), &worker_cancel);
            done_tx.send(result).expect("publish decoder result");
        });

        let deadline = Instant::now() + Duration::from_secs(5);
        while cancel.spawned_child_count() == 0 && Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert_eq!(
            cancel.spawned_child_count(),
            1,
            "the test must cancel a live FFmpeg child"
        );
        cancel.cancel();

        let result = done_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("cancelled decode must kill FFmpeg and join both pipe readers");
        assert!(matches!(result, Err(MediaError::Cancelled)));
        worker.join().expect("decoder worker must be reaped");
        assert_eq!(cancel.active_reader_count(), 0);
    }

    #[cfg(windows)]
    #[test]
    fn windows_cancelling_running_pcm_child_reaps_both_pipe_readers() {
        assert!(
            crate::ff::ffmpeg_available(),
            "required cancellation test needs a runnable FFmpeg"
        );
        let cancel = MediaCancelToken::new();
        let mut child = crate::ff::ffmpeg()
            .args([
                "-re",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=48000:cl=mono",
                "-t",
                "30",
                "-f",
                "f32le",
                "-",
            ])
            .spawn()
            .expect("spawn blocking PCM FFmpeg");
        cancel.child_spawned();
        let stdout = child.take_stdout().expect("PCM stdout");
        let stderr = child.take_stderr().expect("PCM stderr");
        let stdout_cancel = cancel.clone();
        let stderr_cancel = cancel.clone();
        let readers = PipeReaders {
            stdout: std::thread::spawn(move || {
                let limit = ReadLimit {
                    cap: 1024 * 1024,
                    expected: 1024 * 1024,
                };
                let sink = MonoF32Sink::with_capacity(f32_mono_spec(), 0)?;
                read_stdout(stdout, sink, 4, limit, stdout_cancel, None)
            }),
            stderr: std::thread::spawn(move || read_stderr(stderr, stderr_cancel)),
        };
        let worker_cancel = cancel.clone();
        let (done_tx, done_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let result = wait_for_pcm_child(&mut child, readers, &worker_cancel);
            let reaped = child
                .as_inner_mut()
                .try_wait()
                .expect("inspect cancelled PCM child")
                .is_some();
            done_tx
                .send((result.map(|_| ()), reaped))
                .expect("publish PCM cancellation");
        });

        let deadline = Instant::now() + Duration::from_secs(5);
        while cancel.active_reader_count() < 2 && Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert_eq!(cancel.active_reader_count(), 2);
        cancel.cancel();
        let (result, reaped) = done_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("cancelled PCM wait must return promptly");

        assert!(matches!(result, Err(MediaError::Cancelled)));
        assert!(reaped, "cancelled PCM child must be killed and waited");
        worker.join().expect("PCM cancellation worker joins");
        assert_eq!(cancel.active_reader_count(), 0);
    }

    /// Records each pushed chunk and cancels the decode after the first.
    struct CancellingSink {
        cancel: MediaCancelToken,
        pushes: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl PcmSink for CancellingSink {
        fn push_frames(&mut self, _frames: &[u8]) -> Result<()> {
            self.pushes
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.cancel.cancel();
            Ok(())
        }
    }

    #[test]
    fn cancellation_stops_the_streaming_conversion_between_chunks() {
        // Far more PCM than one read chunk, delivered in small reads.
        let raw = vec![0_u8; STDOUT_CHUNK_BYTES * 8];
        let stdout = std::io::BufReader::with_capacity(4096, raw.as_slice());
        let cancel = MediaCancelToken::new();
        let pushes = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let sink = CancellingSink {
            cancel: cancel.clone(),
            pushes: std::sync::Arc::clone(&pushes),
        };
        let limit = ReadLimit {
            cap: raw.len(),
            expected: raw.len(),
        };

        let result = read_stdout(stdout, sink, 4, limit, cancel.clone(), None);

        assert!(matches!(result, Err(MediaError::Cancelled)));
        assert_eq!(pushes.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(cancel.active_reader_count(), 0);
    }

    #[test]
    fn reader_failure_still_joins_the_other_started_reader() {
        let stdout = std::thread::spawn(|| -> Result<StdoutRead<MonoF32Sink>> {
            Err(MediaError::Ffmpeg(
                "deterministic stdout read failure".to_string(),
            ))
        });
        let (stderr_entered_tx, stderr_entered_rx) = mpsc::channel();
        let (stderr_release_tx, stderr_release_rx) = mpsc::channel();
        let stderr = std::thread::spawn(move || {
            stderr_entered_tx.send(()).expect("stderr reader entered");
            stderr_release_rx.recv().expect("release stderr reader");
            Ok(Vec::new())
        });
        let (done_tx, done_rx) = mpsc::channel();
        let joiner = std::thread::spawn(move || {
            done_tx
                .send(join_pipes(PipeReaders { stdout, stderr }))
                .expect("publish join result");
        });
        stderr_entered_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("stderr reader started");

        let early = done_rx.recv_timeout(Duration::from_millis(100)).ok();
        stderr_release_tx.send(()).expect("release stderr reader");
        let returned_early = early.is_some();
        let result = match early {
            Some(result) => result,
            None => done_rx
                .recv_timeout(Duration::from_secs(2))
                .expect("join waits for both readers"),
        };
        joiner.join().expect("join coordinator");

        assert!(
            !returned_early,
            "reader failure must not detach the other reader"
        );
        assert!(matches!(result, Err(MediaError::Ffmpeg(_))));
    }

    #[cfg(unix)]
    #[test]
    fn nonzero_ffmpeg_exit_with_partial_stdout_is_a_hard_error() {
        let status = Command::new("sh")
            .args(["-c", "exit 7"])
            .status()
            .expect("obtain deterministic nonzero status");
        let mut sink = MonoF32Sink::with_capacity(f32_mono_spec(), 1).unwrap();
        sink.push_frames(&[1, 2, 3, 4]).unwrap();
        let stdout = StdoutRead {
            sink,
            exceeded_cap: false,
            total_read: 4,
        };

        let error = validate_pcm_output(
            Path::new("/partial.wav"),
            status,
            stdout,
            b"decoder failed after producing bytes".to_vec(),
            8,
        )
        .expect_err("partial stdout must never mask a nonzero FFmpeg exit");

        assert!(matches!(error, MediaError::Ffmpeg(_)));
        assert!(error.to_string().contains("decoder failed"));
    }

    #[test]
    fn duration_from_mono_samples() {
        let b = PcmBuffer {
            spec: PcmSpec {
                sample_rate: 16_000,
                channels: 1,
                format: PcmFormat::F32,
            },
            samples_f32: vec![0.0; 32_000],
        };
        assert!((b.duration_secs() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn pcm_args_range_emits_ss_and_to() {
        let spec = PcmSpec {
            sample_rate: 16_000,
            channels: 1,
            format: PcmFormat::F32,
        };
        let args = pcm_args(Path::new("/a.mp4"), &spec, Some((1.5, 4.0)));
        let ss = args.iter().position(|a| a == "-ss").unwrap();
        assert_eq!(args[ss + 1], "1.500000");
        let to = args.iter().position(|a| a == "-to").unwrap();
        assert_eq!(args[to + 1], "4.000000");
        assert!(args.windows(2).any(|w| w == ["-ar", "16000"]));
        assert!(args.windows(2).any(|w| w == ["-ac", "1"]));
        assert!(args.windows(2).any(|w| w == ["-f", "f32le"]));
        assert!(args.iter().any(|a| a == "-vn"));
    }

    #[test]
    fn pcm_args_no_range_has_no_seek() {
        let spec = PcmSpec {
            sample_rate: 48_000,
            channels: 2,
            format: PcmFormat::S16Le,
        };
        let args = pcm_args(Path::new("/a.mp4"), &spec, None);
        assert!(!args.iter().any(|a| a == "-ss"));
        assert!(args.windows(2).any(|w| w == ["-f", "s16le"]));
        assert!(args.windows(2).any(|w| w == ["-ac", "2"]));
    }

    /// Mono f32 view of `bytes`, delivered in reads of `read_size` bytes.
    fn mono(bytes: &[u8], spec: &PcmSpec, read_size: usize) -> Vec<f32> {
        let frame_bytes = spec.format.bytes_per_sample() * usize::from(spec.channels);
        let mut sink = MonoF32Sink::with_capacity(*spec, 0).unwrap();
        let mut aligner = FrameAligner::new(frame_bytes);
        for read in bytes.chunks(read_size) {
            aligner.feed(read, &mut sink).unwrap();
        }
        sink.samples
    }

    #[test]
    fn raw_s16_mono_converts_to_unit_floats() {
        let spec = PcmSpec {
            sample_rate: 16_000,
            channels: 1,
            format: PcmFormat::S16Le,
        };
        // samples: 0, 16384 (~0.5), -32768 (-1.0)
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0i16.to_le_bytes());
        bytes.extend_from_slice(&16384i16.to_le_bytes());
        bytes.extend_from_slice(&(-32768i16).to_le_bytes());
        let out = mono(&bytes, &spec, 64);
        assert_eq!(out.len(), 3);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-3);
        assert!((out[2] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn raw_stereo_f32_averages_channels() {
        let spec = PcmSpec {
            sample_rate: 16_000,
            channels: 2,
            format: PcmFormat::F32,
        };
        // frame0: L=1.0 R=0.0 → 0.5 ; frame1: L=-0.5 R=0.5 → 0.0
        let mut bytes = Vec::new();
        for v in [1.0f32, 0.0, -0.5, 0.5] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let out = mono(&bytes, &spec, 64);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn raw_partial_trailing_frame_ignored() {
        let spec = PcmSpec {
            sample_rate: 16_000,
            channels: 1,
            format: PcmFormat::S16Le,
        };
        // 3 bytes = 1 full s16 sample + 1 stray byte → 1 sample.
        let out = mono(&[0, 0, 7], &spec, 64);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn frames_split_across_reads_convert_like_one_read() {
        let spec = PcmSpec {
            sample_rate: 48_000,
            channels: 3,
            format: PcmFormat::F32,
        };
        let bytes = (0..3 * 1000)
            .flat_map(|index| ((index as f32 * 0.37).sin()).to_le_bytes())
            .collect::<Vec<_>>();
        let whole = mono(&bytes, &spec, bytes.len());
        assert_eq!(whole.len(), 1000);
        for read_size in [1, 5, 11, 12, 4096] {
            assert_eq!(
                mono(&bytes, &spec, read_size),
                whole,
                "reads of {read_size}"
            );
        }
    }
}
