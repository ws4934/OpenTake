//! Audio master clock + cpal output for streaming playback (#63 / #160).
//!
//! The acceptance is "audio drives the playhead; video follows (dropping frames
//! to stay in sync)". [`try_build_clock`] realises that: when the timeline carries
//! sound it schedules bounded **interleaved stereo** windows at the cpal device
//! sample rate, plays them through a dedicated cpal output thread,
//! and exposes the device's frame position as [`AudioClock`] — the master clock
//! the render loop reads to pick its target video frame. A silent timeline falls
//! back to the wall-clock [`InstantClock`] PR1 ships.
//!
//! The cpal callback never blocks or allocates. A single producer decodes and
//! mixes fixed windows into a bounded channel; the callback uses only atomics
//! and non-blocking `try_recv`, emitting silence on underrun. Seek advances a
//! generation, cancels the old decode, and makes both producer and consumer
//! discard stale windows before audible output resumes.
//!
//! Stereo is mixed once and mapped to the device's channel count in the callback
//! (mono downmix / >2 zero-fill). The mixing math mirrors the proven export
//! mixdown (`export.rs`), parameterised by the device rate and done per channel.

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use crossbeam_channel::{bounded, Receiver as ChunkReceiver, Sender as ChunkSender};

use opentake_domain::{AudioDenoise, Clip, ClipType, Timeline};
use opentake_media::{
    decode_pcm_interleaved_cancellable, encode::mix::apply_true_peak_ceiling, MediaCancelToken,
    MediaError, PcmFormat, PcmSpec,
};

use crate::clip_audio::{
    clip_source_window_secs, keep_clip_reader, ClipAudioLayout, ClipAudioReader, PreviewDenoise,
};

pub(crate) use crate::clip_audio::ProfileScope;

use super::engine::{InstantClock, PlaybackClock};
use super::project::MediaInfo;

/// Default device sample rate when cpal can't report one (no device queried yet).
const FALLBACK_SAMPLE_RATE: u32 = 48_000;

/// The mix is always interleaved stereo; the callback maps it to the device's
/// channel count.
const MIX_CHANNELS: usize = 2;
const MIX_CANCEL_CHUNK_FRAMES: usize = 4 * 1024;
const STREAM_WINDOW_SECONDS: usize = 2;
const STREAM_WINDOW_CAPACITY: usize = 4;
const STREAM_SEND_POLL: Duration = Duration::from_millis(5);
const CALLBACK_START_TIMEOUT: Duration = Duration::from_secs(1);
const CALLBACK_POLL_INTERVAL: Duration = Duration::from_millis(5);
const AUDIO_CLOCK_STALL_TIMEOUT: Duration = Duration::from_millis(150);
/// How far (in video frames) recovered audio may trail the wall-clock
/// fallback and still take over as is. Video holds for at most this many
/// frames instead of discarding decoded audio, and a device whose callback
/// period is close to the stall timeout cannot loop stall → re-align → seek.
const REALIGN_TOLERANCE_FRAMES: i32 = 2;
const CALLBACKS_REQUIRED_FOR_LIVENESS: u64 = 2;
pub(super) const AUDIO_PREPARE_BUSY: &str = "audio_prepare_busy";

/// One CPAL device query, run on the `opentake-audio-device` thread.
type DeviceJob = Box<dyn FnOnce() + Send + 'static>;

/// Serializes CPAL device discovery on one process-lifetime thread.
///
/// CPAL 0.15's WASAPI backend caches its `IMMDeviceEnumerator` process-wide,
/// while COM initialization is thread-local. Rust's test harness (and Tokio in
/// production) may invoke playback setup from successive short-lived threads;
/// allowing the thread which first created the enumerator to exit can leave the
/// cached COM object with no live originating apartment and the next query can
/// terminate the process with `STATUS_ACCESS_VIOLATION`. Keeping discovery on a
/// dedicated thread both preserves that COM lifetime and prevents concurrent
/// default-device queries from racing. The default rate probe and the
/// default-device check on resume both run there. Opening a stream cannot:
/// `cpal::Stream` is `!Send`, so it is built on (and never leaves) the
/// session's own `opentake-audio` thread, as it was before device checks
/// existed.
static AUDIO_DEVICE_THREAD: OnceLock<Option<SyncSender<DeviceJob>>> = OnceLock::new();

struct AudioPrepareJob<T> {
    build: Box<dyn FnOnce() -> T + Send + 'static>,
    result: tokio::sync::oneshot::Sender<Result<T, String>>,
}

struct AudioPrepareOccupancy {
    occupied: AtomicBool,
    idle: tokio::sync::Notify,
}

impl AudioPrepareOccupancy {
    fn new() -> Self {
        Self {
            occupied: AtomicBool::new(false),
            idle: tokio::sync::Notify::new(),
        }
    }

    fn release(&self) {
        self.occupied.store(false, Ordering::Release);
        self.idle.notify_waiters();
    }
}

struct AudioPrepareOccupancyGuard(Arc<AudioPrepareOccupancy>);

impl Drop for AudioPrepareOccupancyGuard {
    fn drop(&mut self) {
        self.0.release();
    }
}

/// One persistent blocking worker with exactly one admitted audio-prepare job.
pub struct AudioPrepareWorker<T: Send + 'static> {
    sender: SyncSender<AudioPrepareJob<T>>,
    occupancy: Arc<AudioPrepareOccupancy>,
}

/// Owning admission for the single audio-prepare worker slot. Dropping an
/// unsubmitted permit releases the reservation; after submit, the worker owns
/// it until the build closure has fully returned.
#[must_use]
pub struct AudioPreparePermit<T: Send + 'static> {
    sender: SyncSender<AudioPrepareJob<T>>,
    occupancy: Arc<AudioPrepareOccupancy>,
    reserved: bool,
}

impl<T: Send + 'static> AudioPreparePermit<T> {
    pub fn submit(
        mut self,
        build: impl FnOnce() -> T + Send + 'static,
    ) -> Result<tokio::sync::oneshot::Receiver<Result<T, String>>, String> {
        let (result, receiver) = tokio::sync::oneshot::channel();
        let job = AudioPrepareJob {
            build: Box::new(build),
            result,
        };
        match self.sender.try_send(job) {
            Ok(()) => {
                self.reserved = false;
                Ok(receiver)
            }
            Err(TrySendError::Full(_)) => Err(AUDIO_PREPARE_BUSY.to_string()),
            Err(TrySendError::Disconnected(_)) => Err("audio_prepare_worker_stopped".to_string()),
        }
    }
}

impl<T: Send + 'static> Drop for AudioPreparePermit<T> {
    fn drop(&mut self) {
        if self.reserved {
            self.occupancy.release();
        }
    }
}

impl<T: Send + 'static> AudioPrepareWorker<T> {
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::sync_channel::<AudioPrepareJob<T>>(1);
        let occupancy = Arc::new(AudioPrepareOccupancy::new());
        let worker_occupancy = Arc::clone(&occupancy);
        let _ = thread::Builder::new()
            .name("opentake-audio-prepare".to_string())
            .spawn(move || {
                while let Ok(job) = receiver.recv() {
                    let occupancy = AudioPrepareOccupancyGuard(Arc::clone(&worker_occupancy));
                    let value = catch_unwind(AssertUnwindSafe(job.build))
                        .map_err(|_| "audio_prepare_job_panicked".to_string());
                    drop(occupancy);
                    let _ = job.result.send(value);
                }
            });
        Self { sender, occupancy }
    }

    pub fn try_reserve(&self) -> Result<AudioPreparePermit<T>, String> {
        self.occupancy
            .occupied
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| AUDIO_PREPARE_BUSY.to_string())?;
        Ok(AudioPreparePermit {
            sender: self.sender.clone(),
            occupancy: Arc::clone(&self.occupancy),
            reserved: true,
        })
    }

    pub fn try_submit(
        &self,
        build: impl FnOnce() -> T + Send + 'static,
    ) -> Result<tokio::sync::oneshot::Receiver<Result<T, String>>, String> {
        self.try_reserve()?.submit(build)
    }

    pub fn is_occupied(&self) -> bool {
        self.occupancy.occupied.load(Ordering::Acquire)
    }

    /// Wait for the admitted closure to return without polling a worker thread.
    pub async fn wait_until_idle(&self, timeout: Duration) -> bool {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let idle = self.occupancy.idle.notified();
            if !self.is_occupied() {
                return true;
            }
            if tokio::time::timeout_at(deadline, idle).await.is_err() {
                return !self.is_occupied();
            }
        }
    }
}

impl<T: Send + 'static> Default for AudioPrepareWorker<T> {
    fn default() -> Self {
        Self::new()
    }
}

fn audio_buffer_too_large(detail: impl std::fmt::Display) -> MediaError {
    MediaError::Decode(format!("audio_buffer_too_large: {detail}"))
}

fn audio_allocation_failed(detail: impl std::fmt::Display) -> MediaError {
    MediaError::Decode(format!("audio_allocation_failed: {detail}"))
}

struct AudioStreamControl {
    generation: AtomicU64,
    requested_start: AtomicU64,
    stopped: AtomicBool,
    underruns: AtomicU64,
    active_decode: Mutex<Option<MediaCancelToken>>,
    /// The first window failure since the last explicit seek, awaiting report.
    pending_error: Mutex<Option<String>>,
    /// Counts explicit (transport) seeks. A clock re-alignment also restarts
    /// the producer but is not a user action, so a clip that is known to be
    /// broken is not reported again for it.
    seek_epoch: AtomicU64,
    /// Seek epoch whose failure was already recorded (reported once per seek).
    reported_epoch: AtomicU64,
    /// `(generation, end)`: the producer has queued every window of that
    /// generation up to output frame `end`.
    buffered: Mutex<(u64, u64)>,
}

impl AudioStreamControl {
    fn new(start_frame: u64) -> Self {
        Self {
            generation: AtomicU64::new(0),
            requested_start: AtomicU64::new(start_frame),
            stopped: AtomicBool::new(false),
            underruns: AtomicU64::new(0),
            active_decode: Mutex::new(None),
            pending_error: Mutex::new(None),
            seek_epoch: AtomicU64::new(0),
            reported_epoch: AtomicU64::new(u64::MAX),
            buffered: Mutex::new((0, start_frame)),
        }
    }

    /// Record a window failure. Only the first failure after each explicit
    /// seek is kept, so a broken clip is reported once rather than every
    /// window or every clock re-alignment.
    fn record_error(&self, message: String) {
        let epoch = self.seek_epoch.load(Ordering::Acquire);
        if self.reported_epoch.swap(epoch, Ordering::AcqRel) == epoch {
            return;
        }
        *self
            .pending_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(message);
    }

    fn take_error(&self) -> Option<String> {
        self.pending_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
    }

    /// A transport seek: restart the producer at `start_frame` and forget a
    /// failure reported for the previous position.
    fn request_seek(&self, start_frame: u64) {
        self.seek_epoch.fetch_add(1, Ordering::AcqRel);
        self.pending_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        self.restart_at(start_frame);
    }

    /// Whether output frame `frame` of the current generation is already
    /// queued, so the consumer reaches it by skipping forward.
    fn is_buffered(&self, frame: u64) -> bool {
        let generation = self.generation.load(Ordering::Acquire);
        let (buffered_generation, end) = *self
            .buffered
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        buffered_generation == generation && frame < end
    }

    fn mark_buffered(&self, generation: u64, end: u64) {
        let mut buffered = self
            .buffered
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.generation.load(Ordering::Acquire) == generation {
            *buffered = (generation, end);
        }
    }

    /// Restart the producer at `start_frame` under a new generation.
    fn restart_at(&self, start_frame: u64) {
        self.requested_start.store(start_frame, Ordering::Release);
        let generation = self.generation.fetch_add(1, Ordering::AcqRel) + 1;
        *self
            .buffered
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = (generation, start_frame);
        if let Some(cancel) = self
            .active_decode
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
        {
            cancel.cancel();
        }
    }

    fn stop(&self) {
        self.stopped.store(true, Ordering::Release);
        if let Some(cancel) = self
            .active_decode
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
        {
            cancel.cancel();
        }
    }
}

#[derive(Debug)]
struct AudioStreamChunk {
    generation: u64,
    start_frame: u64,
    samples: Vec<f32>,
}

struct AudioStreamConsumer {
    receiver: ChunkReceiver<Result<AudioStreamChunk, MediaError>>,
    control: Arc<AudioStreamControl>,
    current: Option<AudioStreamChunk>,
    /// The producer reported an error or exited, so no later window arrives.
    terminated: bool,
}

impl AudioStreamConsumer {
    fn discard_stale(&mut self) {
        let generation = self.control.generation.load(Ordering::Acquire);
        if self
            .current
            .as_ref()
            .is_some_and(|chunk| chunk.generation != generation)
        {
            self.current = None;
        }
        if self.current.is_some() {
            return;
        }
        loop {
            match self.receiver.try_recv() {
                Ok(Ok(chunk)) if chunk.generation == generation => {
                    self.current = Some(chunk);
                    break;
                }
                Ok(Ok(_)) => {}
                // Producer failures are reported through `AudioStreamControl`;
                // the real-time callback must not log or block.
                Ok(Err(_)) => {
                    self.terminated = true;
                }
                Err(crossbeam_channel::TryRecvError::Empty) => break,
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    self.terminated = true;
                    break;
                }
            }
        }
    }

    fn ready_at(&mut self, frame: u64) -> bool {
        let generation = self.control.generation.load(Ordering::Acquire);
        let covers = |chunk: &AudioStreamChunk| {
            let frames = chunk.samples.len() / MIX_CHANNELS;
            chunk.generation == generation
                && frame >= chunk.start_frame
                && frame < chunk.start_frame.saturating_add(frames as u64)
        };
        // Skip every queued window that ends at or before `frame` (the clock
        // re-aligned forward within the buffer). Bounded by the channel
        // capacity and never blocks.
        loop {
            if self.current.as_ref().is_some_and(covers) {
                return true;
            }
            self.current = None;
            self.discard_stale();
            match self.current.as_ref() {
                Some(chunk) if !covers(chunk) && frame < chunk.start_frame => return false,
                Some(_) => {}
                None => return false,
            }
        }
    }

    fn sample_frame(&mut self, frame: u64) -> (f32, f32) {
        if self.ready_at(frame) {
            let chunk = self.current.as_ref().expect("ready chunk");
            let offset = (frame - chunk.start_frame) as usize * MIX_CHANNELS;
            return (chunk.samples[offset], chunk.samples[offset + 1]);
        }
        self.control.underruns.fetch_add(1, Ordering::Relaxed);
        (0.0, 0.0)
    }
}

enum PlaybackSamples {
    Buffered(Arc<Vec<f32>>),
    Streaming(AudioStreamConsumer),
}

/// Time source of [`AudioClock`]; injectable so stall/recovery tests run on
/// virtual time.
type ClockNow = Arc<dyn Fn() -> Instant + Send + Sync>;

/// Audio master clock: the playhead derives from the device frame position
/// (`pos`, in output audio frames), which the cpal callback advances in lock-step
/// with the sound the user hears — so video genuinely follows audio.
pub struct AudioClock {
    /// Output audio frames played so far (shared with the cpal callback).
    pos: Arc<AtomicU64>,
    /// Output device sample rate (Hz = frames/sec).
    rate: u32,
    /// Project fps (for `seek`, which has no fps argument).
    fps: i32,
    stream: Option<Arc<AudioStreamControl>>,
    progress: Mutex<AudioClockProgress>,
    now: ClockNow,
    /// The output muted by [`PlaybackClock::halt`].
    mute: Option<OutputMute>,
}

struct AudioClockProgress {
    observed_pos: u64,
    observed_underruns: u64,
    observed_at: Instant,
    fallback: Option<(Instant, i32)>,
    last_frame: i32,
}

impl AudioClock {
    fn new(
        pos: Arc<AtomicU64>,
        rate: u32,
        fps: i32,
        stream: Option<Arc<AudioStreamControl>>,
    ) -> Self {
        Self::with_time_source(pos, rate, fps, stream, Arc::new(Instant::now))
    }

    fn with_time_source(
        pos: Arc<AtomicU64>,
        rate: u32,
        fps: i32,
        stream: Option<Arc<AudioStreamControl>>,
        now: ClockNow,
    ) -> Self {
        let observed_pos = pos.load(Ordering::Acquire);
        let observed_underruns = stream
            .as_ref()
            .map_or(0, |control| control.underruns.load(Ordering::Acquire));
        let initial_frame = audio_position_frame(observed_pos, rate, fps);
        let observed_at = now();
        Self {
            pos,
            rate,
            fps,
            stream,
            progress: Mutex::new(AudioClockProgress {
                observed_pos,
                observed_underruns,
                observed_at,
                fallback: None,
                last_frame: initial_frame,
            }),
            now,
            mute: None,
        }
    }

    /// Let [`PlaybackClock::halt`] mute the output that drives this clock.
    fn muting(mut self, mute: OutputMute) -> Self {
        self.mute = Some(mute);
        self
    }
}

fn audio_position_frame(pos: u64, rate: u32, fps: i32) -> i32 {
    let fps = fps.max(1);
    ((pos as f64 / rate.max(1) as f64) * fps as f64) as i32
}

/// Output-frame position of timeline `frame`. Rounds (consistent with the clip
/// placement in `project_clip_audio_stereo`) so it round-trips through
/// [`audio_position_frame`] even when the device rate isn't a multiple of fps
/// (e.g. 44100 Hz @ 24 fps) — plain truncation would land a half-sample short.
fn frame_audio_position(frame: i32, rate: u32, fps: i32) -> u64 {
    ((frame.max(0) as f64 / fps.max(1) as f64) * rate as f64).round() as u64
}

impl PlaybackClock for AudioClock {
    fn frame(&self, fps: i32) -> i32 {
        let fps = if fps > 0 { fps } else { self.fps.max(1) };
        let pos = self.pos.load(Ordering::Acquire);
        let audio_frame = audio_position_frame(pos, self.rate, fps);
        let now = (self.now)();
        let mut progress = self
            .progress
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let underruns = self
            .stream
            .as_ref()
            .map_or(0, |control| control.underruns.load(Ordering::Acquire));
        let wall_frame = |origin: Instant, base_frame: i32| {
            let elapsed_frames =
                (now.saturating_duration_since(origin).as_secs_f64() * fps as f64) as i32;
            base_frame.saturating_add(elapsed_frames.max(0))
        };

        if pos != progress.observed_pos {
            if let Some((origin, base_frame)) = progress.fallback.take() {
                // Callbacks resumed after a stall. While they were silent the
                // video followed wall time, so the device position now trails
                // it by the stall length — and, both advancing at the same
                // rate, would never catch up (permanent A/V drift). Move audio
                // to the video position instead, dropping the stalled span.
                let target_frame = wall_frame(origin, base_frame).max(progress.last_frame);
                progress.observed_underruns = underruns;
                progress.observed_at = now;
                if audio_frame.saturating_add(REALIGN_TOLERANCE_FRAMES) >= target_frame {
                    // Close enough: audio takes over and video holds (never
                    // rewinds) until it catches up.
                    progress.observed_pos = pos;
                    progress.last_frame = progress.last_frame.max(audio_frame);
                    return progress.last_frame;
                }
                let target_pos = frame_audio_position(target_frame, self.rate, fps);
                self.pos.store(target_pos, Ordering::Release);
                if let Some(stream) = &self.stream {
                    // Windows already queued up to the target are skipped by
                    // the consumer; only restart decoding past the buffer.
                    if !stream.is_buffered(target_pos) {
                        stream.restart_at(target_pos);
                    }
                }
                progress.observed_pos = target_pos;
                progress.last_frame = target_frame;
                return target_frame;
            }
            progress.observed_pos = pos;
            progress.observed_underruns = underruns;
            progress.observed_at = now;
        } else if underruns != progress.observed_underruns {
            // The callback is alive but waiting for a decoded window. Hold the
            // video clock at the same frame until the sound can actually play.
            progress.observed_underruns = underruns;
            progress.observed_at = now;
        } else if progress.fallback.is_none()
            && now.saturating_duration_since(progress.observed_at) >= AUDIO_CLOCK_STALL_TIMEOUT
        {
            // The callback was proven live at startup/resume, but devices can be
            // interrupted later. Continue from the last monotonic frame on wall
            // time instead of rendering the same timeline frame forever.
            progress.fallback = Some((progress.observed_at, progress.last_frame.max(audio_frame)));
        }

        let candidate = match progress.fallback {
            Some((origin, base_frame)) => wall_frame(origin, base_frame),
            None => audio_frame,
        };
        progress.last_frame = progress.last_frame.max(candidate);
        progress.last_frame
    }

    fn seek(&self, frame: i32) {
        let pos = frame_audio_position(frame, self.rate, self.fps);
        let moved = self.pos.load(Ordering::Acquire) != pos;
        if moved {
            // Release pairs with the callback's AcqRel fetch_add so it observes the seek.
            self.pos.store(pos, Ordering::Release);
        }
        let mut progress = self
            .progress
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *progress = AudioClockProgress {
            observed_pos: pos,
            observed_underruns: self
                .stream
                .as_ref()
                .map_or(0, |control| control.underruns.load(Ordering::Acquire)),
            observed_at: (self.now)(),
            fallback: None,
            last_frame: frame.max(0),
        };
        drop(progress);
        if moved {
            if let Some(stream) = &self.stream {
                stream.request_seek(pos);
            }
        }
    }

    fn take_error(&self) -> Option<String> {
        self.stream
            .as_ref()
            .and_then(|control| control.take_error())
    }

    fn halt(&self) {
        if let Some(mute) = &self.mute {
            mute.halt();
        }
    }
}

/// The logical mute of one output, as the render loop's clock sees it.
#[derive(Clone)]
struct OutputMute {
    paused: Arc<AtomicBool>,
    /// Set when the render loop paused itself on a fatal failure; keeps a
    /// resume that is committing concurrently from unmuting. Cleared by the
    /// next `prepare_resume`, which precedes the render thread's resume.
    halted: Arc<AtomicBool>,
}

impl OutputMute {
    fn halt(&self) {
        self.halted.store(true, Ordering::Release);
        self.paused.store(true, Ordering::Release);
    }
}

/// A cloneable transport endpoint of one audio output thread. `pause` and
/// `mute` never wait; only `prepare_resume` blocks (for callback liveness or a
/// device rebuild), so callers run it without holding the playback slot lock.
#[derive(Clone)]
pub struct AudioControl {
    control_tx: Sender<AudioCmd>,
    paused: Arc<AtomicBool>,
    halted: Arc<AtomicBool>,
}

impl AudioControl {
    fn new(control_tx: Sender<AudioCmd>, paused: Arc<AtomicBool>) -> Self {
        Self {
            control_tx,
            paused,
            halted: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Mute output immediately. The hardware stream keeps running silently so
    /// a later resume can prove callback liveness without trusting an
    /// asynchronous backend play/pause acknowledgement.
    pub fn pause(&self) -> Result<(), String> {
        self.paused.store(true, Ordering::Release);
        let (reply, _acknowledgement) = mpsc::channel();
        self.control_tx
            .send(AudioCmd::Pause(reply))
            .map_err(|_| "audio thread exited before transport control".to_string())
    }

    /// Prove that callbacks continue while output remains logically muted,
    /// rebuilding the stream first when the default output device changed or
    /// the stream reported an error. The caller can then seek/resume the video
    /// clock before committing audible output.
    pub fn prepare_resume(&self) -> Result<(), String> {
        self.paused.store(true, Ordering::Release);
        self.halted.store(false, Ordering::Release);
        let (reply_tx, reply_rx) = mpsc::channel();
        self.control_tx
            .send(AudioCmd::Resume(reply_tx))
            .map_err(|_| "audio thread exited before transport control".to_string())?;
        reply_rx
            .recv()
            .map_err(|_| "audio thread exited during transport control".to_string())?
    }

    /// Commit a successfully prepared resume after the render clock has been
    /// positioned. The already-running callback begins consuming at `pos` on
    /// its next block. Stays muted when the render thread failed (and halted
    /// the output) after `prepare_resume`.
    pub fn commit_resume(&self) {
        if !self.halted.load(Ordering::Acquire) {
            self.paused.store(false, Ordering::Release);
        }
    }

    fn output_mute(&self) -> OutputMute {
        OutputMute {
            paused: Arc::clone(&self.paused),
            halted: Arc::clone(&self.halted),
        }
    }

    pub fn mute(&self) {
        self.paused.store(true, Ordering::Release);
    }
}

/// Owns the cpal output thread for a playback session. The cpal `Stream` is
/// `!Send` on macOS, so it lives entirely on that thread; this handle drives a
/// cooperative stop. Dropping it stops audio and joins the thread.
pub struct AudioPlayback {
    control: AudioControl,
    handle: Option<JoinHandle<()>>,
    stream_control: Option<Arc<AudioStreamControl>>,
    stream_producer: Option<JoinHandle<()>>,
}

enum AudioCmd {
    Pause(Sender<Result<(), String>>),
    Resume(Sender<Result<(), String>>),
    Stop,
}

/// Samples shared by successive output streams of one session: a stream
/// rebuilt for a new default device continues from the same consumer state.
type SharedSamples = Arc<Mutex<PlaybackSamples>>;

/// What one output stream needs from the session.
#[derive(Clone)]
struct OutputShared {
    samples: SharedSamples,
    pos: Arc<AtomicU64>,
    paused: Arc<AtomicBool>,
    callback_epoch: Arc<AtomicU64>,
    /// Set by the backend's error callback (device removed / invalidated).
    stream_error: Arc<AtomicBool>,
    /// Sample rate the session's audio was mixed at; a rebuilt stream on a new
    /// device must play at the same rate or pitch and clock would drift.
    rate: u32,
}

/// Opens output streams on the system's default device. Abstracted so device
/// changes can be simulated in tests.
trait OutputBackend {
    type Stream;
    /// Identity of the current default output device, when one exists.
    fn default_device_id(&self) -> Option<String>;
    /// Build and start a stream on the current default device, proving
    /// callback liveness, and return it with that device's identity.
    fn open(&self, shared: &OutputShared) -> Result<(Self::Stream, Option<String>), String>;
}

struct CpalBackend;

impl OutputBackend for CpalBackend {
    type Stream = cpal::Stream;

    fn default_device_id(&self) -> Option<String> {
        on_audio_device_thread(query_default_output_device_name)
    }

    fn open(&self, shared: &OutputShared) -> Result<(cpal::Stream, Option<String>), String> {
        build_and_play(shared)
    }
}

impl AudioPlayback {
    /// Start playing `buffer` (interleaved stereo, at the device rate) from `pos`.
    /// Returns `Err` if the device/stream can't be set up (caller falls back to
    /// the wall clock). Blocks until the stream is built so failures surface
    /// synchronously.
    fn start(
        buffer: Arc<Vec<f32>>,
        rate: u32,
        pos: Arc<AtomicU64>,
        paused: Arc<AtomicBool>,
    ) -> Result<Self, String> {
        let (control_tx, handle) = spawn_output(
            || CpalBackend,
            PlaybackSamples::Buffered(buffer),
            rate,
            pos,
            &paused,
        )?;
        Ok(AudioPlayback {
            control: AudioControl::new(control_tx, paused),
            handle: Some(handle),
            stream_control: None,
            stream_producer: None,
        })
    }

    fn start_stream(
        consumer: AudioStreamConsumer,
        stream_control: Arc<AudioStreamControl>,
        stream_producer: JoinHandle<()>,
        rate: u32,
        pos: Arc<AtomicU64>,
        paused: Arc<AtomicBool>,
    ) -> Result<Self, String> {
        match spawn_output(
            || CpalBackend,
            PlaybackSamples::Streaming(consumer),
            rate,
            pos,
            &paused,
        ) {
            Ok((control_tx, handle)) => Ok(Self {
                control: AudioControl::new(control_tx, paused),
                handle: Some(handle),
                stream_control: Some(stream_control),
                stream_producer: Some(stream_producer),
            }),
            Err(error) => {
                stream_control.stop();
                let _ = stream_producer.join();
                Err(error)
            }
        }
    }

    /// A cloneable endpoint for transport control outside the owner.
    pub fn control(&self) -> AudioControl {
        self.control.clone()
    }

    pub fn pause(&self) -> Result<(), String> {
        self.control.pause()
    }

    pub fn prepare_resume(&self) -> Result<(), String> {
        self.control.prepare_resume()
    }

    pub fn commit_resume(&self) {
        self.control.commit_resume();
    }

    pub fn mute(&self) {
        self.control.mute();
    }

    pub fn request_stop(mut self) -> Option<JoinHandle<()>> {
        self.mute();
        if let Some(control) = &self.stream_control {
            control.stop();
        }
        let _ = self.control.control_tx.send(AudioCmd::Stop);
        let audio = self.handle.take();
        let producer = self.stream_producer.take();
        match (audio, producer) {
            (Some(audio), Some(producer)) => {
                let joins = Arc::new(Mutex::new(Some((audio, producer))));
                let worker_joins = Arc::clone(&joins);
                match thread::Builder::new()
                    .name("opentake-audio-stop".to_string())
                    .spawn(move || {
                        if let Some((audio, producer)) = worker_joins
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .take()
                        {
                            let _ = audio.join();
                            let _ = producer.join();
                        }
                    }) {
                    Ok(handle) => Some(handle),
                    Err(_) => {
                        if let Some((audio, producer)) = joins
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .take()
                        {
                            let _ = audio.join();
                            let _ = producer.join();
                        }
                        None
                    }
                }
            }
            (Some(audio), None) => Some(audio),
            (None, Some(producer)) => Some(producer),
            (None, None) => None,
        }
    }

    #[cfg(test)]
    fn from_test_thread(
        control_tx: Sender<AudioCmd>,
        paused: &Arc<AtomicBool>,
        handle: JoinHandle<()>,
    ) -> Self {
        Self {
            control: AudioControl::new(control_tx, Arc::clone(paused)),
            handle: Some(handle),
            stream_control: None,
            stream_producer: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn test_stub() -> (Self, Arc<AtomicBool>, Receiver<()>) {
        let (control_tx, control_rx) = mpsc::channel();
        let (stopped_tx, stopped_rx) = mpsc::channel();
        let paused = Arc::new(AtomicBool::new(false));
        let handle = thread::spawn(move || {
            while let Ok(command) = control_rx.recv() {
                match command {
                    AudioCmd::Pause(reply) | AudioCmd::Resume(reply) => {
                        let _ = reply.send(Ok(()));
                    }
                    AudioCmd::Stop => {
                        let _ = stopped_tx.send(());
                        break;
                    }
                }
            }
        });
        (
            Self::from_test_thread(control_tx, &paused, handle),
            paused,
            stopped_rx,
        )
    }

    #[cfg(test)]
    pub(crate) fn test_failing_resume() -> (Self, Arc<AtomicBool>) {
        let (control_tx, control_rx) = mpsc::channel();
        let paused = Arc::new(AtomicBool::new(true));
        let handle = thread::spawn(move || {
            while let Ok(command) = control_rx.recv() {
                match command {
                    AudioCmd::Pause(reply) => {
                        let _ = reply.send(Ok(()));
                    }
                    AudioCmd::Resume(reply) => {
                        let _ = reply.send(Err("test audio callback unavailable".to_string()));
                    }
                    AudioCmd::Stop => break,
                }
            }
        });
        (Self::from_test_thread(control_tx, &paused, handle), paused)
    }

    /// An output thread whose resume liveness check blocks until released (a
    /// Bluetooth device that stopped calling back).
    #[cfg(test)]
    pub(crate) fn test_blocking_resume() -> (Self, Arc<AtomicBool>, Receiver<()>, Sender<()>) {
        let (control_tx, control_rx) = mpsc::channel();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let paused = Arc::new(AtomicBool::new(true));
        let handle = thread::spawn(move || {
            while let Ok(command) = control_rx.recv() {
                match command {
                    AudioCmd::Pause(reply) => {
                        let _ = reply.send(Ok(()));
                    }
                    AudioCmd::Resume(reply) => {
                        let _ = entered_tx.send(());
                        let _ = release_rx.recv();
                        let _ = reply.send(Ok(()));
                    }
                    AudioCmd::Stop => break,
                }
            }
        });
        (
            Self::from_test_thread(control_tx, &paused, handle),
            paused,
            entered_rx,
            release_tx,
        )
    }

    #[cfg(test)]
    pub(crate) fn test_blocking_stop() -> (Self, Arc<AtomicBool>, Receiver<()>, Sender<()>) {
        let (control_tx, control_rx) = mpsc::channel();
        let (stopped_tx, stopped_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let paused = Arc::new(AtomicBool::new(false));
        let handle = thread::spawn(move || {
            while let Ok(command) = control_rx.recv() {
                match command {
                    AudioCmd::Pause(reply) | AudioCmd::Resume(reply) => {
                        let _ = reply.send(Ok(()));
                    }
                    AudioCmd::Stop => {
                        let _ = stopped_tx.send(());
                        let _ = release_rx.recv();
                        break;
                    }
                }
            }
        });
        (
            Self::from_test_thread(control_tx, &paused, handle),
            paused,
            stopped_rx,
            release_tx,
        )
    }
}

impl Drop for AudioPlayback {
    fn drop(&mut self) {
        if let Some(control) = &self.stream_control {
            control.stop();
        }
        let _ = self.control.control_tx.send(AudioCmd::Stop);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.stream_producer.take() {
            let _ = handle.join();
        }
    }
}

/// Spawn the output thread for `samples` and wait until its first stream is
/// live. `backend` is constructed on that thread (cpal handles may be `!Send`).
fn spawn_output<B, F>(
    backend: F,
    samples: PlaybackSamples,
    rate: u32,
    pos: Arc<AtomicU64>,
    paused: &Arc<AtomicBool>,
) -> Result<(Sender<AudioCmd>, JoinHandle<()>), String>
where
    B: OutputBackend,
    F: FnOnce() -> B + Send + 'static,
{
    let (control_tx, control_rx) = mpsc::channel::<AudioCmd>();
    let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();
    let shared = OutputShared {
        samples: Arc::new(Mutex::new(samples)),
        pos,
        paused: Arc::clone(paused),
        callback_epoch: Arc::new(AtomicU64::new(0)),
        stream_error: Arc::new(AtomicBool::new(false)),
        rate,
    };
    let handle = thread::Builder::new()
        .name("opentake-audio".to_string())
        .spawn(move || audio_thread(backend(), shared, control_rx, ready_tx))
        .map_err(|e| format!("spawn audio thread: {e}"))?;
    match ready_rx.recv() {
        Ok(Ok(())) => Ok((control_tx, handle)),
        Ok(Err(error)) => {
            let _ = handle.join();
            Err(error)
        }
        Err(_) => {
            let _ = handle.join();
            Err("audio thread exited before init".to_string())
        }
    }
}

/// The audio thread: build + play the output stream, report the result, then park
/// (holding the `!Send` stream alive) until a stop is requested. A resume
/// rebuilds the stream on the current default device when the device changed
/// or the stream failed, so a retained session follows the system output.
fn audio_thread<B: OutputBackend>(
    backend: B,
    shared: OutputShared,
    control_rx: Receiver<AudioCmd>,
    ready_tx: Sender<Result<(), String>>,
) {
    let (mut stream, mut device_id) = match backend.open(&shared) {
        Ok((stream, device_id)) => (Some(stream), device_id),
        Err(error) => {
            let _ = ready_tx.send(Err(error));
            return;
        }
    };
    let _ = ready_tx.send(Ok(()));
    while let Ok(command) = control_rx.recv() {
        match command {
            AudioCmd::Pause(reply) => {
                // Logical pause is established by `paused=true` before this
                // message. Keep the hardware stream running muted so a later
                // resume can prove current callback liveness without trusting
                // an asynchronous backend play ack.
                let _ = reply.send(Ok(()));
            }
            AudioCmd::Resume(reply) => {
                let current_device = backend.default_device_id();
                let device_changed = current_device.is_some() && current_device != device_id;
                let result = if stream.is_none()
                    || device_changed
                    || shared.stream_error.load(Ordering::Acquire)
                {
                    // Release the old device before opening the new one.
                    drop(stream.take());
                    shared.stream_error.store(false, Ordering::Release);
                    backend.open(&shared).map(|(rebuilt, rebuilt_device)| {
                        stream = Some(rebuilt);
                        device_id = rebuilt_device;
                    })
                } else {
                    let before = shared.callback_epoch.load(Ordering::Acquire);
                    require_callback_after(&shared.callback_epoch, before, CALLBACK_START_TIMEOUT)
                };
                let _ = reply.send(result);
            }
            AudioCmd::Stop => break,
        }
    }
    drop(stream);
}

/// Acquire the default output device + config, build the typed output stream, and
/// start it. The returned `Stream` must stay alive on the calling thread.
fn build_and_play(shared: &OutputShared) -> Result<(cpal::Stream, Option<String>), String> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| "no default audio output device".to_string())?;
    let device_id = device.name().ok();
    let supported = output_config_at_rate(&device, shared.rate)?;
    let sample_format = supported.sample_format();
    let config: cpal::StreamConfig = supported.into();
    let stream = build_stream(sample_format, &device, &config, shared.clone())?;
    // A successful backend `play()` request does not guarantee that the output
    // callback is live. Installing an AudioClock before sustained callbacks can
    // freeze the playhead at frame zero forever. Prepared sessions start muted
    // for this handshake and keep the hardware stream running silently, so
    // resume never depends on an asynchronous backend play/pause transition.
    let before = shared.callback_epoch.load(Ordering::Acquire);
    stream.play().map_err(|e| format!("stream play: {e}"))?;
    require_callback_after(&shared.callback_epoch, before, CALLBACK_START_TIMEOUT)?;
    Ok((stream, device_id))
}

/// The device's default output config, or — when the device (for example one
/// the user just switched to) defaults to another rate — a supported config at
/// the rate the session's audio was mixed at.
///
/// On macOS, CPAL 0.15 opens a stream at a non-default rate by setting the
/// device's nominal sample rate (`kAudioDevicePropertyNominalSampleRate`),
/// which applies system-wide until something changes it back. Other apps on
/// that device then run at the session's rate; restarting playback after the
/// switch mixes at the device's new default and needs no change.
fn output_config_at_rate(
    device: &cpal::Device,
    rate: u32,
) -> Result<cpal::SupportedStreamConfig, String> {
    let default = device
        .default_output_config()
        .map_err(|e| format!("default output config: {e}"))?;
    if default.sample_rate().0 == rate {
        return Ok(default);
    }
    device
        .supported_output_configs()
        .map_err(|e| format!("supported output configs: {e}"))?
        .filter(|range| range.min_sample_rate().0 <= rate && rate <= range.max_sample_rate().0)
        .max_by_key(|range| {
            (
                range.channels() == default.channels(),
                range.sample_format() == default.sample_format(),
            )
        })
        .map(|range| range.with_sample_rate(cpal::SampleRate(rate)))
        .ok_or_else(|| format!("the default output device does not support {rate} Hz"))
}

fn require_callback_after(epoch: &AtomicU64, before: u64, timeout: Duration) -> Result<(), String> {
    let deadline = Instant::now() + timeout;
    while epoch.load(Ordering::Acquire).wrapping_sub(before) < CALLBACKS_REQUIRED_FOR_LIVENESS {
        if Instant::now() >= deadline {
            return Err(format!(
                "audio output callback did not remain live within {} ms",
                timeout.as_millis()
            ));
        }
        thread::sleep(CALLBACK_POLL_INTERVAL.min(timeout));
    }
    Ok(())
}

/// Dispatch on the device sample format to the typed stream builder.
fn build_stream(
    format: cpal::SampleFormat,
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    shared: OutputShared,
) -> Result<cpal::Stream, String> {
    // Cover every fixed-size cpal format (all satisfy SizedSample + FromSample<f32>)
    // so a non-F32 default device (I32 is common on Linux/Windows) still gets audio
    // instead of silently falling back to the wall clock.
    match format {
        cpal::SampleFormat::F32 => out_stream::<f32>(device, config, shared),
        cpal::SampleFormat::F64 => out_stream::<f64>(device, config, shared),
        cpal::SampleFormat::I8 => out_stream::<i8>(device, config, shared),
        cpal::SampleFormat::I16 => out_stream::<i16>(device, config, shared),
        cpal::SampleFormat::I32 => out_stream::<i32>(device, config, shared),
        cpal::SampleFormat::I64 => out_stream::<i64>(device, config, shared),
        cpal::SampleFormat::U8 => out_stream::<u8>(device, config, shared),
        cpal::SampleFormat::U16 => out_stream::<u16>(device, config, shared),
        cpal::SampleFormat::U32 => out_stream::<u32>(device, config, shared),
        cpal::SampleFormat::U64 => out_stream::<u64>(device, config, shared),
        other => Err(format!("unsupported cpal sample format: {other}")),
    }
}

/// Write one interleaved stereo `(left, right)` sample to a device output frame,
/// mapping to its channel count: mono = average, stereo = L/R, >2 = L/R then
/// silence. Pure (no I/O) so the mapping is unit-tested.
fn write_frame<T: cpal::Sample + FromSample<f32>>(frame: &mut [T], left: f32, right: f32) {
    match frame.len() {
        0 => {}
        1 => frame[0] = T::from_sample((left + right) * 0.5),
        _ => {
            frame[0] = T::from_sample(left);
            frame[1] = T::from_sample(right);
            for sample in frame[2..].iter_mut() {
                *sample = T::from_sample(0.0f32);
            }
        }
    }
}

/// Build an output stream whose callback maps the interleaved stereo mix to the
/// device channels and advances `pos` by the frames written — the lock-free
/// master-clock tick.
fn out_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    shared: OutputShared,
) -> Result<cpal::Stream, String>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = (config.channels as usize).max(1);
    let OutputShared {
        samples,
        pos,
        paused,
        callback_epoch,
        stream_error,
        rate: _,
    } = shared;
    let err_fn = move |e| {
        // Not the real-time callback: record the failure so the next resume
        // rebuilds the stream on the current default device.
        stream_error.store(true, Ordering::Release);
        eprintln!("[audio] stream error: {e}");
    };
    device
        .build_output_stream(
            config,
            move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
                callback_epoch.fetch_add(1, Ordering::Release);
                // Only one stream of a session is ever alive (a rebuild drops
                // the old one first), so this lock is uncontended; never wait.
                let mut samples = match samples.try_lock() {
                    Ok(samples) => samples,
                    Err(std::sync::TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
                    Err(std::sync::TryLockError::WouldBlock) => {
                        for sample in data.iter_mut() {
                            *sample = T::from_sample(0.0f32);
                        }
                        return;
                    }
                };
                let samples = &mut *samples;
                let out_frames = data.len() / channels;
                // Atomically claim this block's start frame and advance the master
                // clock. A concurrent `seek` (store) is honored on the next
                // callback; within a block we play from the claimed start.
                if paused.load(Ordering::Acquire) {
                    if let PlaybackSamples::Streaming(consumer) = samples {
                        consumer.discard_stale();
                    }
                    for sample in data.iter_mut() {
                        *sample = T::from_sample(0.0f32);
                    }
                    return;
                }
                let mut written = 0;
                while written < out_frames {
                    let Some((start, count)) =
                        claim_ready_audio_block(samples, &pos, out_frames - written)
                    else {
                        for sample in &mut data[written * channels..] {
                            *sample = T::from_sample(0.0f32);
                        }
                        return;
                    };
                    for (i, frame) in data[written * channels..(written + count) * channels]
                        .chunks_mut(channels)
                        .enumerate()
                    {
                        let audio_frame = start.saturating_add(i as u64);
                        let (left, right) = match &mut *samples {
                            PlaybackSamples::Buffered(buffer) => {
                                let base = usize::try_from(audio_frame)
                                    .ok()
                                    .and_then(|frame| frame.checked_mul(MIX_CHANNELS));
                                match base.filter(|base| base + 1 < buffer.len()) {
                                    Some(base) => (buffer[base], buffer[base + 1]),
                                    None => (0.0, 0.0),
                                }
                            }
                            PlaybackSamples::Streaming(consumer) => {
                                consumer.sample_frame(audio_frame)
                            }
                        };
                        write_frame(frame, left, right);
                    }
                    written += count;
                }
            },
            err_fn,
            None,
        )
        .map_err(|e| format!("build output stream: {e}"))
}

/// Keep the audio master clock at the first undecoded sample. Once the next
/// window arrives, playback resumes at that exact sample rather than dropping
/// the start of a clip while the callback is emitting silence.
fn claim_ready_audio_block(
    samples: &mut PlaybackSamples,
    pos: &AtomicU64,
    out_frames: usize,
) -> Option<(u64, usize)> {
    let mut count = out_frames;
    if let PlaybackSamples::Streaming(consumer) = samples {
        let start = pos.load(Ordering::Acquire);
        if !consumer.ready_at(start) {
            if !consumer.terminated {
                consumer.control.underruns.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            // A failed or finished producer can never fill the gap. Keep the
            // clock moving over silence instead of freezing playback.
            return Some((pos.fetch_add(count as u64, Ordering::AcqRel), count));
        }
        let chunk = consumer.current.as_ref().expect("ready chunk");
        let end = chunk
            .start_frame
            .saturating_add((chunk.samples.len() / MIX_CHANNELS) as u64);
        count = count.min((end - start) as usize);
    }
    Some((pos.fetch_add(count as u64, Ordering::AcqRel), count))
}

/// Query the default output device's sample rate (Hz), or `None` if unavailable.
fn default_output_rate() -> Option<u32> {
    on_audio_device_thread(query_default_output_rate)
}

/// Run `query` on the process-lifetime device thread and wait for its result.
/// A panicking query yields `None` and leaves the thread serving later ones.
fn on_audio_device_thread<T: Send + 'static>(
    query: impl FnOnce() -> Option<T> + Send + 'static,
) -> Option<T> {
    let worker = AUDIO_DEVICE_THREAD
        .get_or_init(|| {
            let (job_tx, job_rx) = mpsc::sync_channel::<DeviceJob>(1);
            thread::Builder::new()
                .name("opentake-audio-device".to_string())
                .spawn(move || run_device_jobs(job_rx))
                .ok()
                .map(|_| job_tx)
        })
        .as_ref()?;
    submit_device_query(worker, query)
}

fn submit_device_query<T: Send + 'static>(
    worker: &SyncSender<DeviceJob>,
    query: impl FnOnce() -> Option<T> + Send + 'static,
) -> Option<T> {
    let (reply_tx, reply_rx) = mpsc::sync_channel(1);
    worker
        .send(Box::new(move || {
            let result = catch_unwind(AssertUnwindSafe(query)).unwrap_or(None);
            let _ = reply_tx.send(result);
        }))
        .ok()?;
    reply_rx.recv().ok().flatten()
}

fn run_device_jobs(job_rx: Receiver<DeviceJob>) {
    while let Ok(job) = job_rx.recv() {
        job();
    }
}

/// Name of the current default output device. CPAL 0.15 exposes no stable
/// device id, so two identical devices (two headsets of one model) share a
/// name and a switch between them is not detected until the stream reports
/// an error.
fn query_default_output_device_name() -> Option<String> {
    cpal::default_host().default_output_device()?.name().ok()
}

fn query_default_output_rate() -> Option<u32> {
    let host = cpal::default_host();
    let device = host.default_output_device()?;
    let config = device.default_output_config().ok()?;
    Some(config.sample_rate().0)
}

/// One clip's decoded audio, placed on the output timeline as interleaved stereo
/// at the device rate, with its per-output-frame `volume_at` gain envelope.
struct StereoClip {
    /// Output audio-frame offset on the timeline (sample index = ×2).
    start_frame: usize,
    /// Interleaved stereo samples (length = 2 × frames).
    interleaved: Vec<f32>,
    /// Per-output-frame gain (length = frames; empty = unity throughout).
    gains: Vec<f32>,
    /// User true-peak ceiling. The mixer keeps the same codec reconstruction
    /// safety margin as export so native preview does not audition hotter peaks.
    true_peak_ceiling_dbtp: Option<f64>,
}

/// Decode one clip's visible audio window into a placed [`StereoClip`] at `rate`
/// (interleaved stereo). `None` when the clip contributes no audio.
fn project_clip_audio_stereo(
    clip: &Clip,
    media: &HashMap<String, MediaInfo>,
    timeline_fps: i32,
    rate: u32,
    cancel: &MediaCancelToken,
) -> Result<Option<StereoClip>, MediaError> {
    if clip.duration_frames <= 0 || timeline_fps <= 0 || rate == 0 {
        return Ok(None);
    }
    let Some(info) = media.get(&clip.media_ref) else {
        return Ok(None);
    };
    let Some((lo, hi)) = clip_source_window_secs(clip, timeline_fps) else {
        return Ok(None);
    };

    let spec = PcmSpec {
        sample_rate: rate,
        channels: MIX_CHANNELS as u16,
        format: PcmFormat::F32,
    };
    let interleaved =
        decode_pcm_interleaved_cancellable(&info.path, &spec, Some((lo, hi)), cancel)?;
    let interleaved =
        apply_preview_denoise(&interleaved, MIX_CHANNELS, rate, clip.audio_denoise, cancel)?;
    let frames = interleaved.len() / MIX_CHANNELS;
    if frames == 0 {
        return Ok(None);
    }

    let start_frame =
        ((clip.start_frame.max(0) as f64) / timeline_fps as f64 * rate as f64).round() as usize;
    let frames_per_tl_frame = rate as f64 / timeline_fps as f64;
    let mut gains = Vec::new();
    gains
        .try_reserve_exact(frames)
        .map_err(|error| audio_allocation_failed(format!("gain reserve {frames}: {error}")))?;
    let mut all_unity = true;
    for k in 0..frames {
        let tl_frame = clip.start_frame + (k as f64 / frames_per_tl_frame).floor() as i32;
        let g = clip.volume_at(tl_frame) as f32;
        if (g - 1.0).abs() > f32::EPSILON {
            all_unity = false;
        }
        gains.push(g);
    }

    Ok(Some(StereoClip {
        start_frame,
        interleaved,
        gains: if all_unity { Vec::new() } else { gains },
        true_peak_ceiling_dbtp: clip
            .loudness_normalization
            .map(|normalization| normalization.true_peak_ceiling_dbtp),
    }))
}

fn apply_preview_denoise(
    samples: &[f32],
    channels: usize,
    sample_rate: u32,
    config: Option<AudioDenoise>,
    cancel: &MediaCancelToken,
) -> Result<Vec<f32>, MediaError> {
    let Some(config) = config.filter(|config| config.preview_enabled) else {
        return Ok(samples.to_vec());
    };
    opentake_media::analysis::denoise_interleaved(
        samples,
        channels,
        sample_rate,
        config,
        cancel,
        None,
    )
    .map_err(|error| match error {
        opentake_media::analysis::DenoiseError::Cancelled => MediaError::Cancelled,
        other => MediaError::Decode(other.to_string()),
    })
}

/// Visit a placed stereo mix in bounded windows. Only one `window_frames`
/// scratch buffer is live at a time regardless of the total timeline extent.
fn mix_stereo_windows(
    clips: &[StereoClip],
    window_frames: usize,
    cancel: &MediaCancelToken,
    mut emit: impl FnMut(usize, &[f32]) -> Result<(), MediaError>,
) -> Result<(), MediaError> {
    if window_frames == 0 {
        return Err(MediaError::Decode(
            "audio mix window must contain at least one frame".to_string(),
        ));
    }
    let true_peak_ceiling_dbtp = clips
        .iter()
        .filter_map(|clip| clip.true_peak_ceiling_dbtp)
        .min_by(f64::total_cmp);
    let total_frames = clips
        .iter()
        .map(|c| {
            c.start_frame
                .checked_add(c.interleaved.len() / MIX_CHANNELS)
                .ok_or_else(|| audio_buffer_too_large("mix frame extent overflow"))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or(0);
    for window_start in (0..total_frames).step_by(window_frames) {
        if cancel.checkpoint() {
            return Err(MediaError::Cancelled);
        }
        let window_end = window_start.saturating_add(window_frames).min(total_frames);
        let window_samples = window_end
            .saturating_sub(window_start)
            .checked_mul(MIX_CHANNELS)
            .ok_or_else(|| audio_buffer_too_large("mix window sample count overflow"))?;
        let mut out = Vec::new();
        out.try_reserve_exact(window_samples).map_err(|error| {
            audio_allocation_failed(format!("mix window reserve {window_samples}: {error}"))
        })?;
        out.resize(window_samples, 0.0);
        for clip in clips {
            let clip_frames = clip.interleaved.len() / MIX_CHANNELS;
            let clip_end = clip.start_frame.saturating_add(clip_frames);
            let overlap_start = window_start.max(clip.start_frame);
            let overlap_end = window_end.min(clip_end);
            if overlap_start >= overlap_end {
                continue;
            }
            for chunk_start in (overlap_start..overlap_end).step_by(MIX_CANCEL_CHUNK_FRAMES) {
                if cancel.checkpoint() {
                    return Err(MediaError::Cancelled);
                }
                let chunk_end = chunk_start
                    .saturating_add(MIX_CANCEL_CHUNK_FRAMES)
                    .min(overlap_end);
                for timeline_frame in chunk_start..chunk_end {
                    let clip_frame = timeline_frame - clip.start_frame;
                    let gain = if clip.gains.is_empty() {
                        1.0
                    } else {
                        clip.gains[clip_frame]
                    };
                    let output = (timeline_frame - window_start) * MIX_CHANNELS;
                    let input = clip_frame * MIX_CHANNELS;
                    out[output] += clip.interleaved[input] * gain;
                    out[output + 1] += clip.interleaved[input + 1] * gain;
                }
            }
        }
        for value in &mut out {
            *value = value.clamp(-1.0, 1.0);
        }
        apply_true_peak_ceiling(&mut out, true_peak_ceiling_dbtp);
        emit(window_start, &out)?;
    }
    Ok(())
}

/// Sum placed stereo clips into one interleaved buffer, applying per-frame gains
/// and hard-limiting to [-1, 1] (mirrors the export mixdown, per channel).
fn mix_stereo(clips: &[StereoClip], cancel: &MediaCancelToken) -> Result<Vec<f32>, MediaError> {
    let mut out = Vec::new();
    mix_stereo_windows(
        clips,
        MIX_CANCEL_CHUNK_FRAMES,
        cancel,
        |_start_frame, samples| {
            out.try_reserve(samples.len()).map_err(|error| {
                audio_allocation_failed(format!("mix output reserve {}: {error}", samples.len()))
            })?;
            out.extend_from_slice(samples);
            Ok(())
        },
    )?;
    Ok(out)
}

fn timeline_audio_frames(timeline: &Timeline, rate: u32) -> Result<u64, MediaError> {
    if timeline.fps <= 0 || rate == 0 {
        return Ok(0);
    }
    let frames = timeline.total_frames().max(0) as u64;
    let numerator = frames
        .checked_mul(rate as u64)
        .ok_or_else(|| audio_buffer_too_large("streaming timeline frame extent overflow"))?;
    numerator
        .checked_add(timeline.fps as u64 / 2)
        .map(|rounded| rounded / timeline.fps as u64)
        .ok_or_else(|| audio_buffer_too_large("streaming timeline frame rounding overflow"))
}

/// Per-session preview audio state reused across mix windows: which sources
/// have audio (probed once per source, not once per window), the clip
/// readers of the window just mixed, which the next window continues so a
/// clip keeps one decoder and one resampler while it plays, and the owner of
/// the background profile passes the session asks for.
#[derive(Default)]
struct PreviewAudioSources {
    has_audio: HashMap<PathBuf, bool>,
    readers: Vec<PreviewReader>,
    profiles: ProfileScope,
}

/// An open clip reader, and the clip and source it reads.
struct PreviewReader {
    clip_id: String,
    path: PathBuf,
    layout: ClipAudioLayout,
    reader: ClipAudioReader,
}

impl PreviewAudioSources {
    fn new(profiles: ProfileScope) -> Self {
        PreviewAudioSources {
            profiles,
            ..PreviewAudioSources::default()
        }
    }

    fn has_audio(&mut self, path: &Path, cancel: &MediaCancelToken) -> Result<bool, MediaError> {
        if let Some(audible) = self.has_audio.get(path) {
            return Ok(*audible);
        }
        let audible = crate::clip_audio::source_has_audio(path, cancel)?;
        self.has_audio.insert(path.to_path_buf(), audible);
        Ok(audible)
    }

    /// Take the open reader that continues `clip` at clip frame `from` with
    /// the requested denoise, if there is one. Any other reader of the clip
    /// (an old layout, or a denoise that changed) is closed.
    fn take_reader(
        &mut self,
        clip_id: &str,
        path: &Path,
        layout: &ClipAudioLayout,
        from: usize,
        denoised: bool,
    ) -> Option<ClipAudioReader> {
        let mut taken = None;
        let mut index = 0;
        while index < self.readers.len() {
            let open = &self.readers[index];
            if open.clip_id != clip_id {
                index += 1;
                continue;
            }
            let open = self.readers.swap_remove(index);
            if taken.is_none()
                && open.path == path
                && open.layout == *layout
                && open.reader.position() == from
                && open.reader.is_denoised() == denoised
            {
                taken = Some(open.reader);
            }
        }
        taken
    }

    /// Close the readers a window over mix frames `[start, end)` cannot
    /// continue: their clip does not play there, or the reader is not at
    /// the window's first frame of the clip.
    fn close_readers_outside(&mut self, start: u64, end: u64) {
        self.readers.retain(|open| {
            let (clip_start, clip_end) = open.layout.span();
            let overlap_start = start.max(clip_start);
            overlap_start < end.min(clip_end)
                && open.reader.position() == open.layout.offset_of(overlap_start)
        });
    }

    /// Close every open reader (after a seek or a failed window).
    fn close_readers(&mut self) {
        self.readers.clear();
    }
}

/// How a mix window gets the noise profile of a denoised clip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProfileWait {
    /// Playback: never wait; play the clip undenoised until the background
    /// worker has its profile.
    Background,
    /// Compute (or wait for) the profile before mixing, as export does.
    #[cfg(test)]
    Block,
}

fn mix_timeline_window(
    timeline: &Timeline,
    media: &HashMap<String, MediaInfo>,
    rate: u32,
    window_start: u64,
    window_frames: usize,
    sources: &mut PreviewAudioSources,
    cancel: &MediaCancelToken,
) -> Result<Vec<f32>, MediaError> {
    mix_timeline_window_channels(
        timeline,
        media,
        rate,
        MIX_CHANNELS,
        window_start,
        window_frames,
        sources,
        ProfileWait::Background,
        cancel,
    )
}

/// Mix timeline mix frames `[window_start, window_start + window_frames)`.
/// Every clip is rendered through the export's clip-audio path (placement,
/// resampling and denoise are functions of the absolute clip position), so a
/// window's samples do not depend on where playback started (#16). A clip
/// that continues from the previous window keeps its reader, so consecutive
/// windows read one continuous decode, as export does. `channels` is the
/// preview's stereo except in the export-parity test.
#[allow(clippy::too_many_arguments)]
fn mix_timeline_window_channels(
    timeline: &Timeline,
    media: &HashMap<String, MediaInfo>,
    rate: u32,
    channels: usize,
    window_start: u64,
    window_frames: usize,
    sources: &mut PreviewAudioSources,
    profiles: ProfileWait,
    cancel: &MediaCancelToken,
) -> Result<Vec<f32>, MediaError> {
    let result = mix_window_with_readers(
        timeline,
        media,
        rate,
        channels,
        window_start,
        window_frames,
        sources,
        profiles,
        cancel,
    );
    if result.is_err() {
        // A failed or cancelled read leaves its reader mid-chunk.
        sources.close_readers();
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn mix_window_with_readers(
    timeline: &Timeline,
    media: &HashMap<String, MediaInfo>,
    rate: u32,
    channels: usize,
    window_start: u64,
    window_frames: usize,
    sources: &mut PreviewAudioSources,
    profiles: ProfileWait,
    cancel: &MediaCancelToken,
) -> Result<Vec<f32>, MediaError> {
    let sample_count = window_frames
        .checked_mul(channels)
        .ok_or_else(|| audio_buffer_too_large("streaming window sample count overflow"))?;
    let mut mixed = vec![0.0_f32; sample_count];
    let window_end = window_start.saturating_add(window_frames as u64);
    let audible_clips = || {
        timeline
            .tracks
            .iter()
            .filter(|track| !track.muted)
            .flat_map(|track| &track.clips)
            .filter(|clip| {
                matches!(clip.media_type, ClipType::Audio | ClipType::Video)
                    && clip.duration_frames > 0
            })
    };
    // Like export, the strictest ceiling of any clip applies to the whole
    // timeline, so it cannot change with the window alignment.
    let true_peak_ceiling_dbtp = audible_clips()
        .filter_map(|clip| {
            clip.loudness_normalization
                .map(|normalization| normalization.true_peak_ceiling_dbtp)
        })
        .min_by(f64::total_cmp);
    let mut samples = Vec::new();
    // Readers this window leaves mid-clip; they replace `sources.readers` at
    // the end, which closes the previous window's readers nothing continued.
    // At most `MAX_OPEN_CLIP_READERS` readers are open at any moment: the
    // previous window's (`sources.readers`), the continuing ones and one
    // reader of a clip beyond the cap, closed right after it is read.
    let mut continuing = Vec::new();
    sources.close_readers_outside(window_start, window_end);
    for clip in audible_clips() {
        if cancel.checkpoint() {
            return Err(MediaError::Cancelled);
        }
        let Some(info) = media.get(&clip.media_ref) else {
            continue;
        };
        let Some(layout) = ClipAudioLayout::new(clip, timeline.fps, rate) else {
            continue;
        };
        let (clip_start, clip_end) = layout.span();
        let overlap_start = window_start.max(clip_start);
        let overlap_end = window_end.min(clip_end);
        if overlap_start >= overlap_end || !sources.has_audio(&info.path, cancel)? {
            continue;
        }
        let config = clip.audio_denoise.filter(|config| config.preview_enabled);
        let denoise = match profiles {
            ProfileWait::Background => {
                match crate::clip_audio::preview_clip_denoise(
                    config,
                    &clip.id,
                    &layout,
                    &info.path,
                    channels,
                    &sources.profiles,
                )? {
                    PreviewDenoise::Ready(profile, config) => Some((profile, config)),
                    PreviewDenoise::Off | PreviewDenoise::Pending => None,
                }
            }
            #[cfg(test)]
            ProfileWait::Block => crate::clip_audio::clip_denoise(
                config, &layout, &info.path, channels, cancel, None,
            )?,
        };
        let from = layout.offset_of(overlap_start);
        let (mut reader, keep) =
            match sources.take_reader(&clip.id, &info.path, &layout, from, denoise.is_some()) {
                Some(reader) => (reader, true),
                None => {
                    let keep = keep_clip_reader(sources.readers.len() + continuing.len());
                    let reader =
                        ClipAudioReader::open(layout, &info.path, channels, from, denoise, cancel)?;
                    (reader, keep)
                }
            };
        let frames = (overlap_end - overlap_start) as usize;
        samples.clear();
        reader.read(frames, &mut samples)?;
        if overlap_end < clip_end && keep {
            continuing.push(PreviewReader {
                clip_id: clip.id.clone(),
                path: info.path.clone(),
                layout,
                reader,
            });
        } else {
            drop(reader);
        }
        let output_start = (overlap_start - window_start) as usize;
        for frame in 0..frames {
            let timeline_frame = crate::clip_audio::timeline_frame_at(
                overlap_start + frame as u64,
                timeline.fps,
                rate,
            );
            let gain = clip.volume_at(timeline_frame) as f32;
            let output = (output_start + frame) * channels;
            for channel in 0..channels {
                mixed[output + channel] += samples[frame * channels + channel] * gain;
            }
        }
    }
    // Keep the readers the next window continues; close the rest.
    sources.readers = continuing;
    for sample in &mut mixed {
        *sample = sample.clamp(-1.0, 1.0);
    }
    apply_true_peak_ceiling(&mut mixed, true_peak_ceiling_dbtp);
    Ok(mixed)
}

struct PreparedTimelineAudio {
    consumer: AudioStreamConsumer,
    control: Arc<AudioStreamControl>,
    producer: JoinHandle<()>,
}

fn send_stream_chunk(
    sender: &ChunkSender<Result<AudioStreamChunk, MediaError>>,
    mut chunk: Result<AudioStreamChunk, MediaError>,
    control: &AudioStreamControl,
    generation: u64,
) -> bool {
    loop {
        if control.stopped.load(Ordering::Acquire)
            || control.generation.load(Ordering::Acquire) != generation
        {
            return false;
        }
        match sender.try_send(chunk) {
            Ok(()) => return true,
            Err(crossbeam_channel::TrySendError::Full(returned)) => {
                chunk = returned;
                thread::sleep(STREAM_SEND_POLL);
            }
            Err(crossbeam_channel::TrySendError::Disconnected(_)) => return false,
        }
    }
}

/// Window geometry of one streaming audio producer.
struct ProducerWindows {
    rate: u32,
    /// First output frame after the synchronously prefilled window.
    next_frame: u64,
    total_frames: u64,
    window_frames: usize,
}

/// The `opentake-audio-fill` loop: mix consecutive windows into the bounded
/// channel, restarting at the requested position on every seek generation.
/// A window that fails to decode is reported once per generation and played
/// as silence, so one broken clip neither stops the producer (silencing the
/// rest of the session) nor freezes the audio master clock.
fn run_audio_producer(
    control: &AudioStreamControl,
    sender: &ChunkSender<Result<AudioStreamChunk, MediaError>>,
    windows: ProducerWindows,
    mut mix_window: impl FnMut(u64, usize, &MediaCancelToken) -> Result<Vec<f32>, MediaError>,
) {
    let ProducerWindows {
        rate,
        mut next_frame,
        total_frames,
        window_frames,
    } = windows;
    let mut generation = 0_u64;
    let mut window_cancel = MediaCancelToken::new();
    loop {
        if control.stopped.load(Ordering::Acquire) {
            break;
        }
        let observed = control.generation.load(Ordering::Acquire);
        if observed != generation {
            generation = observed;
            next_frame = control.requested_start.load(Ordering::Acquire);
        }
        if next_frame >= total_frames {
            thread::sleep(STREAM_SEND_POLL);
            continue;
        }
        let len = (total_frames - next_frame).min(window_frames as u64) as usize;
        // Clip readers stay open from window to window, so the windows of one
        // generation share a token; a seek or stop cancels it.
        if window_cancel.is_cancelled() {
            window_cancel = MediaCancelToken::new();
        }
        *control
            .active_decode
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(window_cancel.clone());
        if control.generation.load(Ordering::Acquire) != generation {
            window_cancel.cancel();
            control
                .active_decode
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .take();
            continue;
        }
        let result = mix_window(next_frame, len, &window_cancel);
        control
            .active_decode
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if control.generation.load(Ordering::Acquire) != generation {
            continue;
        }
        let samples = match result {
            Ok(samples) => samples,
            Err(MediaError::Cancelled) => continue,
            Err(error) => {
                control.record_error(format!(
                    "audio at {:.1} s could not be decoded and plays as silence: {error}",
                    next_frame as f64 / rate.max(1) as f64
                ));
                vec![0.0; len * MIX_CHANNELS]
            }
        };
        if send_stream_chunk(
            sender,
            Ok(AudioStreamChunk {
                generation,
                start_frame: next_frame,
                samples,
            }),
            control,
            generation,
        ) {
            next_frame = next_frame.saturating_add(len as u64);
            control.mark_buffered(generation, next_frame);
        }
    }
}

/// Prepare bounded timeline-audio scheduling at `rate`. The initial window is
/// mixed synchronously so decode failures surface before playback ownership is
/// published; all subsequent windows are produced on one bounded worker.
fn mix_timeline_stereo(
    timeline: &Timeline,
    media: &HashMap<String, MediaInfo>,
    rate: u32,
    start_frame: u64,
    profiles: &ProfileScope,
    cancel: &MediaCancelToken,
) -> Result<Option<PreparedTimelineAudio>, MediaError> {
    if timeline.fps <= 0 || rate == 0 {
        return Ok(None);
    }
    let has_candidates = timeline
        .tracks
        .iter()
        .filter(|track| !track.muted)
        .any(|track| {
            track.clips.iter().any(|clip| {
                matches!(clip.media_type, ClipType::Audio | ClipType::Video)
                    && clip.duration_frames > 0
                    && media.contains_key(&clip.media_ref)
            })
        });
    if !has_candidates {
        return Ok(None);
    }
    let total_frames = timeline_audio_frames(timeline, rate)?;
    if start_frame >= total_frames {
        return Ok(None);
    }
    let window_frames = (rate as usize)
        .checked_mul(STREAM_WINDOW_SECONDS)
        .ok_or_else(|| audio_buffer_too_large("streaming window frame overflow"))?;
    let first_len = (total_frames - start_frame).min(window_frames as u64) as usize;
    let mut sources = PreviewAudioSources::new(profiles.clone());
    let first_samples = mix_timeline_window(
        timeline,
        media,
        rate,
        start_frame,
        first_len,
        &mut sources,
        cancel,
    )?;
    let (sender, receiver) = bounded(STREAM_WINDOW_CAPACITY);
    sender
        .send(Ok(AudioStreamChunk {
            generation: 0,
            start_frame,
            samples: first_samples,
        }))
        .map_err(|_| MediaError::Decode("audio stream queue closed during prefill".to_string()))?;
    let control = Arc::new(AudioStreamControl::new(start_frame));
    control.mark_buffered(0, start_frame.saturating_add(first_len as u64));
    let producer_control = Arc::clone(&control);
    let producer_timeline = timeline.clone();
    let producer_media = media.clone();
    let producer = thread::Builder::new()
        .name("opentake-audio-fill".to_string())
        .spawn(move || {
            run_audio_producer(
                &producer_control,
                &sender,
                ProducerWindows {
                    rate,
                    next_frame: start_frame.saturating_add(first_len as u64),
                    total_frames,
                    window_frames,
                },
                |start, len, cancel| {
                    mix_timeline_window(
                        &producer_timeline,
                        &producer_media,
                        rate,
                        start,
                        len,
                        &mut sources,
                        cancel,
                    )
                    .map_err(|error| match error {
                        MediaError::Cancelled => MediaError::Cancelled,
                        error => MediaError::Decode(super::project::redact_media_paths(
                            &producer_media,
                            &error.to_string(),
                        )),
                    })
                },
            )
        })
        .map_err(|error| MediaError::Decode(format!("spawn audio fill worker: {error}")))?;
    Ok(Some(PreparedTimelineAudio {
        consumer: AudioStreamConsumer {
            receiver,
            control: Arc::clone(&control),
            current: None,
            terminated: false,
        },
        control,
        producer,
    }))
}

/// Production-facing clock construction with explicit media/allocation errors.
pub fn try_build_clock(
    timeline: &Timeline,
    media: &HashMap<String, MediaInfo>,
    fps: i32,
    start_frame: i32,
) -> Result<(Arc<dyn PlaybackClock>, Option<AudioPlayback>), MediaError> {
    build_clock_with_state(
        timeline,
        media,
        fps,
        start_frame,
        false,
        &ProfileScope::new(),
        &MediaCancelToken::new(),
    )
}

/// Prepare audio without starting the device clock. The retained playback
/// session resumes audio only after its first composited frame is buffered.
pub fn build_clock_paused(
    timeline: &Timeline,
    media: &HashMap<String, MediaInfo>,
    fps: i32,
    start_frame: i32,
) -> Result<(Arc<dyn PlaybackClock>, Option<AudioPlayback>), MediaError> {
    build_clock_paused_cancellable(
        timeline,
        media,
        fps,
        start_frame,
        &ProfileScope::new(),
        &MediaCancelToken::new(),
    )
}

/// [`build_clock_paused`] for a playback state: `profiles` owns the
/// background denoise-profile passes the preview asks for.
pub(crate) fn build_clock_paused_cancellable(
    timeline: &Timeline,
    media: &HashMap<String, MediaInfo>,
    fps: i32,
    start_frame: i32,
    profiles: &ProfileScope,
    cancel: &MediaCancelToken,
) -> Result<(Arc<dyn PlaybackClock>, Option<AudioPlayback>), MediaError> {
    build_clock_with_state(timeline, media, fps, start_frame, true, profiles, cancel)
}

fn build_clock_with_state(
    timeline: &Timeline,
    media: &HashMap<String, MediaInfo>,
    fps: i32,
    start_frame: i32,
    start_paused: bool,
    profiles: &ProfileScope,
    cancel: &MediaCancelToken,
) -> Result<(Arc<dyn PlaybackClock>, Option<AudioPlayback>), MediaError> {
    let rate = default_output_rate().unwrap_or(FALLBACK_SAMPLE_RATE);
    let start_audio_frame =
        ((start_frame.max(0) as f64 / fps.max(1) as f64) * rate as f64).round() as u64;
    let Some(prepared) =
        mix_timeline_stereo(timeline, media, rate, start_audio_frame, profiles, cancel)?
    else {
        return Ok((Arc::new(InstantClock::new(start_frame)), None));
    };
    let pos = Arc::new(AtomicU64::new(start_audio_frame));
    let paused = Arc::new(AtomicBool::new(start_paused));
    let clock = AudioClock::new(
        Arc::clone(&pos),
        rate,
        fps,
        Some(Arc::clone(&prepared.control)),
    );
    match AudioPlayback::start_stream(
        prepared.consumer,
        prepared.control,
        prepared.producer,
        rate,
        pos,
        paused,
    ) {
        Ok(audio) => {
            let clock = clock.muting(audio.control.output_mute());
            Ok((Arc::new(clock), Some(audio)))
        }
        Err(error) => {
            eprintln!("[audio] {error}; falling back to wall clock");
            Ok((Arc::new(InstantClock::new(start_frame)), None))
        }
    }
}

fn clock_from_mixed<F>(
    mixed: Vec<f32>,
    rate: u32,
    fps: i32,
    start_frame: i32,
    start_paused: bool,
    start: F,
) -> (Arc<dyn PlaybackClock>, Option<AudioPlayback>)
where
    F: FnOnce(Arc<Vec<f32>>, Arc<AtomicU64>, Arc<AtomicBool>) -> Result<AudioPlayback, String>,
{
    let buffer = Arc::new(mixed);
    let pos = Arc::new(AtomicU64::new(0));
    let paused = Arc::new(AtomicBool::new(start_paused));
    let clock = AudioClock::new(pos.clone(), rate, fps, None);
    clock.seek(start_frame); // begin playback at the current playhead

    match start(buffer, pos, paused) {
        Ok(audio) => {
            let clock = clock.muting(audio.control.output_mute());
            (Arc::new(clock), Some(audio))
        }
        Err(e) => {
            eprintln!("[audio] {e}; falling back to wall clock");
            (Arc::new(InstantClock::new(start_frame)), None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    use opentake_domain::{Clip, Track};
    use opentake_media::{MediaCancelToken, MediaError};

    fn audio_timeline(clips: Vec<Clip>) -> Timeline {
        let mut timeline = Timeline::new();
        timeline.fps = 30;
        let mut track = Track::new("a1", ClipType::Audio);
        track.clips = clips;
        timeline.tracks.push(track);
        timeline
    }

    fn audio_clip(id: &str, media_ref: &str, start_frame: i32, duration_frames: i32) -> Clip {
        let mut clip = Clip::new(id, media_ref, start_frame, duration_frames);
        clip.media_type = ClipType::Audio;
        clip.source_clip_type = ClipType::Audio;
        clip
    }

    #[test]
    fn audio_prepare_cancel_stops_before_decoding_next_clip() {
        assert!(
            opentake_media::ffmpeg_status::ffmpeg_available(),
            "required cancellation test needs a runnable FFmpeg"
        );
        let temp = tempfile::tempdir().expect("create audio cancellation fixtures");
        let first = temp.path().join("first.wav");
        let second = temp.path().join("second.wav");
        for fifo in [&first, &second] {
            let status = std::process::Command::new("mkfifo")
                .arg(fifo)
                .status()
                .expect("spawn mkfifo");
            assert!(status.success());
        }
        let timeline = audio_timeline(vec![
            audio_clip("c1", "m1", 0, 900),
            audio_clip("c2", "m2", 900, 900),
        ]);
        let media = HashMap::from([
            ("m1".to_string(), MediaInfo { path: first }),
            ("m2".to_string(), MediaInfo { path: second }),
        ]);
        let cancel = MediaCancelToken::new();
        let worker_cancel = cancel.clone();
        let (done_tx, done_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            done_tx
                .send(mix_timeline_stereo(
                    &timeline,
                    &media,
                    48_000,
                    0,
                    &ProfileScope::new(),
                    &worker_cancel,
                ))
                .expect("publish audio prepare result");
        });

        let deadline = Instant::now() + Duration::from_secs(5);
        while cancel.spawned_child_count() == 0 && Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert_eq!(cancel.spawned_child_count(), 1);
        cancel.cancel();
        let error = match done_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("cancelled audio prepare must return")
        {
            Err(error) => error,
            Ok(_) => panic!("cancelled audio prepare must fail"),
        };
        assert!(matches!(error, MediaError::Cancelled));
        worker.join().expect("join audio prepare worker");
        assert_eq!(
            cancel.spawned_child_count(),
            1,
            "second clip must not spawn"
        );
    }

    #[test]
    fn large_mix_observes_cancellation_between_chunks() {
        let cancel = MediaCancelToken::new();
        let worker_cancel = cancel.clone();
        let clip = StereoClip {
            start_frame: 0,
            interleaved: vec![0.25; 12_000_000],
            gains: Vec::new(),
            true_peak_ceiling_dbtp: None,
        };
        let (done_tx, done_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            done_tx
                .send(mix_stereo(&[clip], &worker_cancel))
                .expect("publish mix result");
        });
        let deadline = Instant::now() + Duration::from_secs(5);
        while cancel.checkpoint_count() == 0 && Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert!(
            cancel.checkpoint_count() > 0,
            "mix must enter a production chunk"
        );
        cancel.cancel();
        let result = done_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("chunked mix must observe cancellation");
        assert!(matches!(result, Err(MediaError::Cancelled)));
        worker.join().expect("join mix worker");
    }

    #[test]
    fn long_timeline_mix_has_constant_peak_allocation_and_matches_short_reference() {
        let near = StereoClip {
            start_frame: 0,
            interleaved: vec![0.6, -0.6, 0.5, 0.5],
            gains: Vec::new(),
            true_peak_ceiling_dbtp: None,
        };
        let far = StereoClip {
            start_frame: 48_000 * 60 * 60,
            interleaved: vec![0.25, -0.25],
            gains: Vec::new(),
            true_peak_ceiling_dbtp: None,
        };
        let reference = mix_stereo(
            &[StereoClip {
                start_frame: near.start_frame,
                interleaved: near.interleaved.clone(),
                gains: near.gains.clone(),
                true_peak_ceiling_dbtp: near.true_peak_ceiling_dbtp,
            }],
            &MediaCancelToken::new(),
        )
        .unwrap();
        let mut first_window = Vec::new();
        let mut peak_samples = 0;

        mix_stereo_windows(
            &[near, far],
            1024,
            &MediaCancelToken::new(),
            |start_frame, samples| {
                peak_samples = peak_samples.max(samples.len());
                if start_frame == 0 {
                    first_window.extend_from_slice(samples);
                }
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(&first_window[..reference.len()], reference);
        assert!(
            peak_samples <= 1024 * MIX_CHANNELS,
            "one-hour timeline must retain only one bounded mix window"
        );
    }

    #[test]
    fn streaming_consumer_discards_pre_seek_chunks_and_reports_underrun_as_silence() {
        let control = Arc::new(AudioStreamControl::new(0));
        let (sender, receiver) = bounded(4);
        sender
            .send(Ok(AudioStreamChunk {
                generation: 0,
                start_frame: 0,
                samples: vec![0.25, -0.25],
            }))
            .unwrap();
        let mut consumer = AudioStreamConsumer {
            receiver,
            control: Arc::clone(&control),
            current: None,
            terminated: false,
        };
        assert_eq!(consumer.sample_frame(0), (0.25, -0.25));

        control.request_seek(10);
        sender
            .send(Ok(AudioStreamChunk {
                generation: 0,
                start_frame: 1,
                samples: vec![0.5, 0.5],
            }))
            .unwrap();
        sender
            .send(Ok(AudioStreamChunk {
                generation: 1,
                start_frame: 10,
                samples: vec![0.75, -0.75],
            }))
            .unwrap();
        assert_eq!(consumer.sample_frame(10), (0.75, -0.75));
        assert_eq!(consumer.sample_frame(99), (0.0, 0.0));
        assert_eq!(control.underruns.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn prefilled_audio_survives_start_and_resume_at_the_same_frame() {
        let first_sample = 48_000;
        let control = Arc::new(AudioStreamControl::new(first_sample));
        let (sender, receiver) = bounded(4);
        sender
            .send(Ok(AudioStreamChunk {
                generation: 0,
                start_frame: first_sample,
                samples: vec![0.75, -0.75, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125],
            }))
            .unwrap();
        let pos = Arc::new(AtomicU64::new(first_sample));
        let clock = AudioClock::new(Arc::clone(&pos), 48_000, 30, Some(Arc::clone(&control)));
        let mut samples = PlaybackSamples::Streaming(AudioStreamConsumer {
            receiver,
            control: Arc::clone(&control),
            current: None,
            terminated: false,
        });

        clock.seek(30); // render thread initialization
        clock.seek(30); // first Resume
        assert_eq!(control.generation.load(Ordering::Acquire), 0);
        for expected in [
            [(0.75, -0.75), (0.5, -0.5)],
            [(0.25, -0.25), (0.125, -0.125)],
        ] {
            let (start, count) =
                claim_ready_audio_block(&mut samples, &pos, 2).expect("prefilled block");
            assert_eq!(count, 2);
            let PlaybackSamples::Streaming(consumer) = &mut samples else {
                unreachable!()
            };
            assert_eq!(consumer.sample_frame(start), expected[0]);
            assert_eq!(consumer.sample_frame(start + 1), expected[1]);
        }
        assert_eq!(control.underruns.load(Ordering::Acquire), 0);

        clock.seek(31); // pause at the next frame
        let paused_generation = control.generation.load(Ordering::Acquire);
        clock.seek(31); // resume at exactly that paused position
        assert_eq!(
            control.generation.load(Ordering::Acquire),
            paused_generation
        );
    }

    #[test]
    fn audio_decode_stall_preserves_the_first_unplayed_sample() {
        let control = Arc::new(AudioStreamControl::new(0));
        let (sender, receiver) = bounded(4);
        let pos = AtomicU64::new(0);
        let mut samples = PlaybackSamples::Streaming(AudioStreamConsumer {
            receiver,
            control: Arc::clone(&control),
            current: None,
            terminated: false,
        });

        assert_eq!(claim_ready_audio_block(&mut samples, &pos, 128), None);
        assert_eq!(pos.load(Ordering::Acquire), 0);
        sender
            .send(Ok(AudioStreamChunk {
                generation: 0,
                start_frame: 0,
                samples: vec![0.8, -0.8, 0.6, -0.6],
            }))
            .unwrap();
        assert_eq!(claim_ready_audio_block(&mut samples, &pos, 2), Some((0, 2)));
        let PlaybackSamples::Streaming(consumer) = &mut samples else {
            unreachable!()
        };
        assert_eq!(consumer.sample_frame(0), (0.8, -0.8));
        assert_eq!(consumer.sample_frame(1), (0.6, -0.6));
    }

    #[test]
    fn failed_audio_stream_keeps_the_clock_moving_with_silence() {
        let control = Arc::new(AudioStreamControl::new(0));
        let (sender, receiver) = bounded(4);
        sender
            .send(Err(MediaError::Decode("broken clip".into())))
            .unwrap();
        drop(sender);
        let pos = AtomicU64::new(0);
        let mut samples = PlaybackSamples::Streaming(AudioStreamConsumer {
            receiver,
            control,
            current: None,
            terminated: false,
        });

        assert_eq!(
            claim_ready_audio_block(&mut samples, &pos, 128),
            Some((0, 128))
        );
        assert_eq!(pos.load(Ordering::Acquire), 128);
        let PlaybackSamples::Streaming(consumer) = &mut samples else {
            unreachable!()
        };
        assert_eq!(consumer.sample_frame(0), (0.0, 0.0));
    }

    #[test]
    fn callback_crossing_a_window_boundary_waits_without_skipping_samples() {
        let control = Arc::new(AudioStreamControl::new(0));
        let (sender, receiver) = bounded(4);
        sender
            .send(Ok(AudioStreamChunk {
                generation: 0,
                start_frame: 0,
                samples: vec![0.5, 0.5, 0.5, 0.5],
            }))
            .unwrap();
        let pos = AtomicU64::new(0);
        let mut samples = PlaybackSamples::Streaming(AudioStreamConsumer {
            receiver,
            control,
            current: None,
            terminated: false,
        });

        assert_eq!(
            claim_ready_audio_block(&mut samples, &pos, 128),
            Some((0, 2))
        );
        assert_eq!(claim_ready_audio_block(&mut samples, &pos, 126), None);
        assert_eq!(pos.load(Ordering::Acquire), 2);
        sender
            .send(Ok(AudioStreamChunk {
                generation: 0,
                start_frame: 2,
                samples: vec![0.75, -0.75, 0.75, -0.75],
            }))
            .unwrap();
        assert_eq!(
            claim_ready_audio_block(&mut samples, &pos, 126),
            Some((2, 2))
        );
        let PlaybackSamples::Streaming(consumer) = &mut samples else {
            unreachable!()
        };
        assert_eq!(consumer.sample_frame(2), (0.75, -0.75));
    }

    #[test]
    fn paused_stream_drain_retains_the_next_current_generation_chunk() {
        let control = Arc::new(AudioStreamControl::new(0));
        let (sender, receiver) = bounded(4);
        for (start_frame, sample) in [(0, 0.25), (1, 0.75)] {
            sender
                .send(Ok(AudioStreamChunk {
                    generation: 0,
                    start_frame,
                    samples: vec![sample, -sample],
                }))
                .unwrap();
        }
        let mut consumer = AudioStreamConsumer {
            receiver,
            control,
            current: None,
            terminated: false,
        };

        consumer.discard_stale();
        consumer.discard_stale();

        assert_eq!(consumer.sample_frame(0), (0.25, -0.25));
        assert_eq!(consumer.sample_frame(1), (0.75, -0.75));
    }

    #[test]
    fn audio_decode_failure_is_not_silently_treated_as_silent_timeline() {
        let timeline = audio_timeline(vec![audio_clip("broken", "missing", 0, 30)]);
        let media = HashMap::from([(
            "missing".to_string(),
            MediaInfo {
                path: PathBuf::from("/definitely/missing/audio.wav"),
            },
        )]);

        let error = match mix_timeline_stereo(
            &timeline,
            &media,
            48_000,
            0,
            &ProfileScope::new(),
            &MediaCancelToken::new(),
        ) {
            Err(error) => error,
            Ok(_) => panic!("decode failure must propagate instead of producing an empty mix"),
        };

        assert!(!matches!(error, MediaError::Cancelled));
    }

    #[test]
    fn try_build_clock_propagates_missing_audio_error_without_panic() {
        let timeline = audio_timeline(vec![audio_clip("broken", "missing", 0, 30)]);
        let media = HashMap::from([(
            "missing".to_string(),
            MediaInfo {
                path: PathBuf::from("/definitely/missing/try-build-clock.wav"),
            },
        )]);

        let error = match try_build_clock(&timeline, &media, 30, 0) {
            Err(error) => error,
            Ok(_) => panic!("production clock entry must propagate media errors"),
        };

        assert!(!matches!(error, MediaError::Cancelled));
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn default_output_rate_survives_sequential_short_lived_callers() {
        for _ in 0..8 {
            std::thread::spawn(default_output_rate)
                .join()
                .expect("WASAPI rate probing must survive caller thread teardown");
        }
    }

    #[test]
    fn audio_device_thread_survives_a_query_panic() {
        let (job_tx, job_rx) = mpsc::sync_channel::<DeviceJob>(1);
        let worker = std::thread::spawn(move || run_device_jobs(job_rx));

        assert_eq!(
            submit_device_query(&job_tx, || -> Option<u32> {
                panic!("simulated CPAL query panic")
            }),
            None
        );
        assert_eq!(submit_device_query(&job_tx, || Some(44_100)), Some(44_100));
        assert_eq!(
            submit_device_query(&job_tx, || Some("speakers".to_string())),
            Some("speakers".to_string())
        );

        drop(job_tx);
        worker.join().expect("join audio device thread");
    }

    #[test]
    fn rapid_superseding_starts_never_exceed_one_audio_prepare_job() {
        let worker = AudioPrepareWorker::<usize>::new();
        let active = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let max_active = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let job_active = Arc::clone(&active);
        let job_max = Arc::clone(&max_active);
        let first = worker
            .try_submit(move || {
                let now = job_active.fetch_add(1, Ordering::AcqRel) + 1;
                job_max.fetch_max(now, Ordering::AcqRel);
                entered_tx.send(()).expect("announce first prepare");
                release_rx.recv().expect("release first prepare");
                job_active.fetch_sub(1, Ordering::AcqRel);
                1
            })
            .expect("first prepare admitted");
        entered_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("first prepare starts");

        for _ in 0..16 {
            assert!(
                worker.try_submit(|| 2).is_err(),
                "superseding starts must be busy while one prepare owns capacity"
            );
        }
        release_tx.send(()).expect("release first prepare");
        assert_eq!(
            first
                .blocking_recv()
                .expect("first result channel")
                .expect("first job succeeds"),
            1
        );
        assert_eq!(max_active.load(Ordering::Acquire), 1);
    }

    #[test]
    fn panicking_prepare_reports_error_releases_capacity_and_worker_survives() {
        let worker = AudioPrepareWorker::<usize>::new();
        let first = worker
            .try_reserve()
            .expect("reserve first prepare")
            .submit(|| panic!("deterministic prepare panic"))
            .expect("submit first prepare");

        let error = first
            .blocking_recv()
            .expect("worker must publish a panic result")
            .expect_err("panicking closure must be an explicit job error");
        assert_eq!(error, "audio_prepare_job_panicked");
        assert!(!worker.is_occupied());

        let second = worker
            .try_reserve()
            .expect("capacity recovered after unwind")
            .submit(|| 7)
            .expect("persistent worker accepts next job");
        assert_eq!(
            second
                .blocking_recv()
                .expect("second result channel")
                .expect("second job succeeds"),
            7
        );
    }

    #[test]
    fn cancelled_prepare_releases_capacity_only_after_worker_exits() {
        let worker = AudioPrepareWorker::<()>::new();
        let cancel = MediaCancelToken::new();
        let job_cancel = cancel.clone();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let first = worker
            .try_submit(move || {
                entered_tx.send(()).expect("announce prepare");
                while !job_cancel.is_cancelled() {
                    std::thread::yield_now();
                }
                release_rx
                    .recv()
                    .expect("hold cancelled worker before exit");
            })
            .expect("first prepare admitted");
        entered_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("prepare starts");
        cancel.cancel();

        assert!(worker.try_submit(|| ()).is_err());
        release_tx.send(()).expect("allow cancelled worker exit");
        first
            .blocking_recv()
            .expect("cancelled worker result channel")
            .expect("cancelled worker exits");
        let deadline = Instant::now() + Duration::from_secs(2);
        while worker.is_occupied() && Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert!(!worker.is_occupied());
        worker
            .try_submit(|| ())
            .expect("capacity releases only after exit")
            .blocking_recv()
            .expect("replacement result channel")
            .expect("replacement prepare completes");
    }

    #[test]
    fn audio_clock_frame_and_seek_round_trip() {
        let clock = AudioClock::new(Arc::new(AtomicU64::new(0)), 48_000, 30, None);
        // seek(30) → 30 frames = 1s = 48000 output frames → frame()==30.
        clock.seek(30);
        assert_eq!(clock.pos.load(Ordering::Relaxed), 48_000);
        assert_eq!(clock.frame(30), 30);

        // Half a second of frames → frame 15.
        let half_second = AudioClock::new(Arc::new(AtomicU64::new(24_000)), 48_000, 30, None);
        assert_eq!(half_second.frame(30), 15);
    }

    #[test]
    fn audio_clock_falls_back_to_wall_time_when_callbacks_stall_mid_playback() {
        let clock = AudioClock::new(Arc::new(AtomicU64::new(0)), 48_000, 100, None);

        assert_eq!(clock.frame(100), 0);
        std::thread::sleep(AUDIO_CLOCK_STALL_TIMEOUT + Duration::from_millis(20));

        assert!(
            clock.frame(100) >= 1,
            "a dead audio callback must not freeze the timeline forever"
        );
    }

    /// Virtual time for [`AudioClock`] tests.
    #[derive(Clone)]
    struct TestTime(Arc<Mutex<Instant>>);

    impl TestTime {
        fn new() -> Self {
            Self(Arc::new(Mutex::new(Instant::now())))
        }

        fn advance(&self, by: Duration) {
            *self.0.lock().unwrap() += by;
        }

        fn source(&self) -> ClockNow {
            let now = Arc::clone(&self.0);
            Arc::new(move || *now.lock().unwrap())
        }
    }

    /// Simulate the cpal callback: `ms` of linear output at `rate`, in 10 ms
    /// blocks, reading the clock after each block like the render loop does.
    fn play_linear(
        clock: &AudioClock,
        time: &TestTime,
        pos: &AtomicU64,
        rate: u64,
        ms: u64,
        frames: &mut Vec<i32>,
    ) {
        for _ in 0..ms / 10 {
            time.advance(Duration::from_millis(10));
            pos.fetch_add(rate / 100, Ordering::AcqRel);
            frames.push(clock.frame(30));
        }
    }

    fn stall(clock: &AudioClock, time: &TestTime, ms: u64, frames: &mut Vec<i32>) {
        for _ in 0..ms / 10 {
            time.advance(Duration::from_millis(10));
            frames.push(clock.frame(30));
        }
    }

    #[test]
    fn recovered_audio_callbacks_and_explicit_seeks_never_rewind_accidentally() {
        let time = TestTime::new();
        let pos = Arc::new(AtomicU64::new(0));
        let clock =
            AudioClock::with_time_source(Arc::clone(&pos), 48_000, 100, None, time.source());
        let mut frames = vec![clock.frame(100)];
        time.advance(AUDIO_CLOCK_STALL_TIMEOUT + Duration::from_millis(20));
        let fallback_frame = clock.frame(100);
        assert!(fallback_frame >= 1);

        // A recovering device advances linearly from where it stalled, behind
        // the wall fallback. It must neither pull the timeline backwards nor
        // stay on wall time forever.
        for _ in 0..10 {
            time.advance(Duration::from_millis(10));
            pos.fetch_add(480, Ordering::AcqRel);
            frames.push(clock.frame(100));
        }
        assert!(
            frames.windows(2).all(|pair| pair[0] <= pair[1]),
            "{frames:?}"
        );
        let fallback_left = clock.progress.lock().unwrap().fallback.is_none();
        assert!(
            fallback_left,
            "linear recovery must exit the wall-clock fallback"
        );
        let audio_frame = audio_position_frame(pos.load(Ordering::Acquire), 48_000, 100);
        assert!((clock.frame(100) - audio_frame).abs() <= 1);

        // A user/transport seek is authoritative and intentionally may move back.
        clock.seek(7);
        assert_eq!(clock.frame(100), 7);
    }

    #[test]
    fn stalled_then_linearly_recovered_callback_realigns_audio_to_video() {
        let time = TestTime::new();
        let pos = Arc::new(AtomicU64::new(0));
        let control = Arc::new(AudioStreamControl::new(0));
        let clock = AudioClock::with_time_source(
            Arc::clone(&pos),
            48_000,
            30,
            Some(Arc::clone(&control)),
            time.source(),
        );
        let mut frames = Vec::new();

        play_linear(&clock, &time, &pos, 48_000, 1_000, &mut frames);
        assert_eq!(*frames.last().unwrap(), 30);
        stall(&clock, &time, 300, &mut frames);
        assert!(
            *frames.last().unwrap() >= 38,
            "wall fallback keeps video moving during the stall: {frames:?}"
        );
        assert_eq!(control.generation.load(Ordering::Acquire), 0);

        // One callback block after recovery (one clock tick) the clock and the
        // audio position agree again, via exactly one stream seek.
        play_linear(&clock, &time, &pos, 48_000, 10, &mut frames);
        let audio_frame = audio_position_frame(pos.load(Ordering::Acquire), 48_000, 30);
        let video_frame = *frames.last().unwrap();
        assert!(
            (video_frame - audio_frame).abs() <= 1,
            "video {video_frame} vs audio {audio_frame}"
        );
        assert_eq!(control.generation.load(Ordering::Acquire), 1);
        assert_eq!(
            control.requested_start.load(Ordering::Acquire),
            pos.load(Ordering::Acquire)
        );

        // Afterwards audio drives the clock again, with no residual offset.
        play_linear(&clock, &time, &pos, 48_000, 500, &mut frames);
        let audio_frame = audio_position_frame(pos.load(Ordering::Acquire), 48_000, 30);
        assert_eq!(*frames.last().unwrap(), audio_frame);
        assert_eq!(control.generation.load(Ordering::Acquire), 1);
        assert!(
            frames.windows(2).all(|pair| pair[0] <= pair[1]),
            "the playhead never moves backwards: {frames:?}"
        );
    }

    #[test]
    fn a_render_failure_during_a_resume_handshake_keeps_audio_muted() {
        let (audio, paused, _stopped) = AudioPlayback::test_stub();
        let clock = AudioClock::new(Arc::new(AtomicU64::new(0)), 48_000, 30, None)
            .muting(audio.control().output_mute());

        audio.prepare_resume().expect("prepare");
        // The render thread, already resumed, fails before the commit.
        clock.halt();
        audio.commit_resume();
        assert!(
            paused.load(Ordering::Acquire),
            "a failed render keeps sound off"
        );

        // The next resume is a retry and unmutes normally.
        audio.prepare_resume().expect("prepare again");
        audio.commit_resume();
        assert!(!paused.load(Ordering::Acquire));
    }

    #[test]
    fn a_large_callback_block_after_a_stall_takes_over_without_a_seek() {
        // A device with a ~160 ms callback period: the clock falls back to
        // wall time between blocks, and each block catches audio up in one go.
        let time = TestTime::new();
        let pos = Arc::new(AtomicU64::new(0));
        let control = Arc::new(AudioStreamControl::new(0));
        let clock = AudioClock::with_time_source(
            Arc::clone(&pos),
            48_000,
            30,
            Some(Arc::clone(&control)),
            time.source(),
        );
        let mut frames = Vec::new();
        play_linear(&clock, &time, &pos, 48_000, 1_000, &mut frames);
        for _ in 0..5 {
            stall(&clock, &time, 160, &mut frames);
            pos.fetch_add(48_000 * 160 / 1_000, Ordering::AcqRel);
            frames.push(clock.frame(30));
        }
        assert_eq!(
            control.generation.load(Ordering::Acquire),
            0,
            "audio within the tolerance must not be re-seeked: {frames:?}"
        );
        assert!(
            frames.windows(2).all(|pair| pair[0] <= pair[1]),
            "{frames:?}"
        );
        let audio_frame = audio_position_frame(pos.load(Ordering::Acquire), 48_000, 30);
        assert!(
            (*frames.last().unwrap() - audio_frame).abs() <= REALIGN_TOLERANCE_FRAMES,
            "{frames:?} vs audio {audio_frame}"
        );
    }

    #[test]
    fn realignment_within_the_queued_audio_skips_forward_without_restarting() {
        let time = TestTime::new();
        let pos = Arc::new(AtomicU64::new(0));
        let control = Arc::new(AudioStreamControl::new(0));
        // Eight seconds are already queued for generation 0.
        control.mark_buffered(0, 8 * 48_000);
        let clock = AudioClock::with_time_source(
            Arc::clone(&pos),
            48_000,
            30,
            Some(Arc::clone(&control)),
            time.source(),
        );
        let mut frames = Vec::new();
        play_linear(&clock, &time, &pos, 48_000, 1_000, &mut frames);
        stall(&clock, &time, 300, &mut frames);
        play_linear(&clock, &time, &pos, 48_000, 10, &mut frames);

        let audio_frame = audio_position_frame(pos.load(Ordering::Acquire), 48_000, 30);
        assert!((*frames.last().unwrap() - audio_frame).abs() <= 1);
        assert_eq!(
            control.generation.load(Ordering::Acquire),
            0,
            "the queued windows are kept; the consumer skips to the new position"
        );
    }

    #[test]
    fn consumer_skips_queued_windows_before_a_forward_position() {
        let control = Arc::new(AudioStreamControl::new(0));
        let (sender, receiver) = bounded(STREAM_WINDOW_CAPACITY);
        for window in 0..3_u64 {
            sender
                .send(Ok(AudioStreamChunk {
                    generation: 0,
                    start_frame: window * 100,
                    samples: vec![window as f32; 100 * MIX_CHANNELS],
                }))
                .unwrap();
        }
        let mut consumer = AudioStreamConsumer {
            receiver,
            control: Arc::clone(&control),
            current: None,
            terminated: false,
        };
        assert_eq!(consumer.sample_frame(10), (0.0, 0.0));
        assert_eq!(consumer.sample_frame(250), (2.0, 2.0));
        assert_eq!(control.underruns.load(Ordering::Acquire), 0);
    }

    #[test]
    fn a_seek_forgets_an_unreported_failure_but_realignment_does_not_repeat_it() {
        let control = AudioStreamControl::new(0);
        control.record_error("clip-2 failed at 1.0 s".to_string());
        control.request_seek(500);
        assert_eq!(
            control.take_error(),
            None,
            "the error belonged to the old position"
        );

        control.record_error("clip-2 failed at 6.0 s".to_string());
        assert!(control.take_error().is_some());
        // A clock re-alignment restarts the producer, which hits the same clip.
        control.restart_at(700);
        control.record_error("clip-2 failed at 7.0 s".to_string());
        assert_eq!(
            control.take_error(),
            None,
            "reported once until the next seek"
        );
    }

    #[test]
    fn producer_reports_a_failed_window_once_and_recovers_after_seek() {
        let control = Arc::new(AudioStreamControl::new(0));
        let (sender, receiver) = bounded(STREAM_WINDOW_CAPACITY);
        let failing = Arc::new(AtomicBool::new(true));
        let producer_control = Arc::clone(&control);
        let producer_failing = Arc::clone(&failing);
        let producer = thread::spawn(move || {
            run_audio_producer(
                &producer_control,
                &sender,
                ProducerWindows {
                    rate: 100,
                    next_frame: 0,
                    total_frames: 10_000,
                    window_frames: 100,
                },
                |start, len, _cancel| {
                    // Every window from 100 on hits a broken clip.
                    if start >= 100 && producer_failing.load(Ordering::Acquire) {
                        return Err(MediaError::Decode("clip-2 decode failed".to_string()));
                    }
                    Ok(vec![0.5; len * MIX_CHANNELS])
                },
            )
        });

        let next_chunk = || {
            receiver
                .recv_timeout(Duration::from_secs(2))
                .expect("producer keeps producing")
                .expect("failures are silence, not stream errors")
        };
        let mut chunks = Vec::new();
        for _ in 0..3 {
            chunks.push(next_chunk());
        }
        assert_eq!(
            chunks
                .iter()
                .map(|chunk| chunk.start_frame)
                .collect::<Vec<_>>(),
            vec![0, 100, 200]
        );
        assert!(chunks[0].samples.iter().all(|sample| *sample == 0.5));
        assert!(chunks[1].samples.iter().all(|sample| *sample == 0.0));
        let error = control.take_error().expect("failure reported");
        assert!(error.contains("clip-2 decode failed"), "{error}");
        assert!(error.contains("1.0 s"), "{error}");
        assert_eq!(control.take_error(), None, "reported once per generation");

        // The source becomes decodable; a seek restarts the producer there and
        // the consumer plays real samples without underruns.
        failing.store(false, Ordering::Release);
        control.request_seek(500);
        let mut consumer = AudioStreamConsumer {
            receiver: receiver.clone(),
            control: Arc::clone(&control),
            current: None,
            terminated: false,
        };
        let deadline = Instant::now() + Duration::from_secs(2);
        while !consumer.ready_at(500) {
            assert!(Instant::now() < deadline, "post-seek window arrives");
            thread::sleep(Duration::from_millis(2));
        }
        let underruns = control.underruns.load(Ordering::Acquire);
        for frame in 500..600 {
            assert_eq!(consumer.sample_frame(frame), (0.5, 0.5));
        }
        assert_eq!(control.underruns.load(Ordering::Acquire), underruns);
        assert_eq!(control.take_error(), None);

        control.stop();
        drop(consumer);
        drop(receiver);
        producer.join().expect("producer exits on stop");
    }

    /// Output backend double: a "stream" is a thread that ticks the callback
    /// epoch until dropped; the default device id can change between resumes.
    struct FakeBackend {
        default_device: Arc<Mutex<String>>,
        opened: Arc<Mutex<Vec<String>>>,
        stream_error: Arc<Mutex<Option<Arc<AtomicBool>>>>,
    }

    struct FakeStream {
        stop: Arc<AtomicBool>,
    }

    impl Drop for FakeStream {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Release);
        }
    }

    impl OutputBackend for FakeBackend {
        type Stream = FakeStream;

        fn default_device_id(&self) -> Option<String> {
            Some(self.default_device.lock().unwrap().clone())
        }

        fn open(&self, shared: &OutputShared) -> Result<(FakeStream, Option<String>), String> {
            let device = self.default_device.lock().unwrap().clone();
            self.opened.lock().unwrap().push(device.clone());
            *self.stream_error.lock().unwrap() = Some(Arc::clone(&shared.stream_error));
            let stop = Arc::new(AtomicBool::new(false));
            let ticking = Arc::clone(&stop);
            let epoch = Arc::clone(&shared.callback_epoch);
            thread::spawn(move || {
                while !ticking.load(Ordering::Acquire) {
                    epoch.fetch_add(1, Ordering::Release);
                    thread::sleep(Duration::from_millis(1));
                }
            });
            let before = shared.callback_epoch.load(Ordering::Acquire);
            require_callback_after(&shared.callback_epoch, before, CALLBACK_START_TIMEOUT)?;
            Ok((FakeStream { stop }, Some(device)))
        }
    }

    #[test]
    fn retained_resume_rebuilds_the_stream_only_when_the_default_device_changes() {
        let default_device = Arc::new(Mutex::new("speakers".to_string()));
        let opened = Arc::new(Mutex::new(Vec::new()));
        let stream_error = Arc::new(Mutex::new(None));
        let backend = FakeBackend {
            default_device: Arc::clone(&default_device),
            opened: Arc::clone(&opened),
            stream_error: Arc::clone(&stream_error),
        };
        let paused = Arc::new(AtomicBool::new(true));
        let (control_tx, handle) = spawn_output(
            move || backend,
            PlaybackSamples::Buffered(Arc::new(vec![0.0; 96_000])),
            48_000,
            Arc::new(AtomicU64::new(0)),
            &paused,
        )
        .expect("fake output starts");
        let audio = AudioPlayback::from_test_thread(control_tx, &paused, handle);
        assert_eq!(*opened.lock().unwrap(), vec!["speakers"]);

        audio.pause().expect("pause");
        audio.prepare_resume().expect("resume on the same device");
        assert_eq!(
            opened.lock().unwrap().len(),
            1,
            "unchanged device keeps its stream"
        );

        audio.pause().expect("pause");
        *default_device.lock().unwrap() = "headphones".to_string();
        audio
            .prepare_resume()
            .expect("resume follows the new default device");
        assert_eq!(*opened.lock().unwrap(), vec!["speakers", "headphones"]);

        audio
            .prepare_resume()
            .expect("resume again on the same new device");
        assert_eq!(opened.lock().unwrap().len(), 2);

        // A stream error (device invalidated) forces a rebuild on next resume.
        stream_error
            .lock()
            .unwrap()
            .as_ref()
            .expect("backend saw the error flag")
            .store(true, Ordering::Release);
        audio
            .prepare_resume()
            .expect("resume rebuilds after a stream error");
        assert_eq!(opened.lock().unwrap().len(), 3);
        drop(audio);
    }

    #[test]
    fn callback_liveness_detects_epoch_advance() {
        let epoch = Arc::new(AtomicU64::new(4));
        let worker_epoch = Arc::clone(&epoch);
        let worker = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(10));
            worker_epoch.fetch_add(1, Ordering::Release);
            std::thread::sleep(Duration::from_millis(10));
            worker_epoch.fetch_add(1, Ordering::Release);
        });

        require_callback_after(&epoch, 4, Duration::from_millis(250))
            .expect("callback epoch should advance");
        worker.join().expect("callback worker");
    }

    #[test]
    fn callback_liveness_times_out_without_callback() {
        let epoch = AtomicU64::new(0);
        let error = require_callback_after(&epoch, 0, Duration::from_millis(15))
            .expect_err("missing callback must fail readiness");
        assert!(error.contains("did not remain live"));
    }

    #[test]
    fn callback_liveness_rejects_one_trailing_callback() {
        let epoch = Arc::new(AtomicU64::new(7));
        let trailing_epoch = Arc::clone(&epoch);
        let trailing = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(5));
            trailing_epoch.fetch_add(1, Ordering::Release);
        });

        require_callback_after(&epoch, 7, Duration::from_millis(25))
            .expect_err("one trailing callback must not prove resumed liveness");
        trailing.join().expect("trailing callback worker");
    }

    #[test]
    fn failed_audio_start_installs_advancing_wall_clock() {
        let (clock, audio) = clock_from_mixed(
            vec![0.0, 0.0],
            48_000,
            100,
            0,
            false,
            |_buffer, _pos, _paused| Err("callback readiness timeout".to_string()),
        );

        assert!(audio.is_none());
        std::thread::sleep(Duration::from_millis(25));
        assert!(
            clock.frame(100) >= 1,
            "fallback clock must keep playback live"
        );
    }

    #[test]
    fn successful_audio_start_retains_device_clock() {
        let (clock, audio) = clock_from_mixed(
            vec![0.0, 0.0],
            48_000,
            30,
            0,
            false,
            |_buffer, pos, _paused| {
                pos.store(48_000, Ordering::Release);
                Ok(AudioPlayback::test_stub().0)
            },
        );

        assert!(audio.is_some());
        assert_eq!(clock.frame(30), 30);
    }

    #[test]
    fn audio_clock_truncates_partial_frames() {
        let clock = AudioClock::new(Arc::new(AtomicU64::new(0)), 48_000, 30, None);
        // 1599 frames @ 48k, 30fps = 0.999 video frame → truncates to 0.
        clock.pos.store(1_599, Ordering::Relaxed);
        assert_eq!(clock.frame(30), 0);
        // 1600 frames = exactly one video frame.
        clock.pos.store(1_600, Ordering::Relaxed);
        assert_eq!(clock.frame(30), 1);
    }

    #[test]
    fn audio_clock_seek_round_trips_at_non_divisible_rate() {
        // 44100 Hz @ 24 fps: rate/fps = 1837.5 (not integer). seek (round) +
        // frame (truncate) must still land back on the same frame — a regression
        // guard for the truncate-only seek that reported frame-1 here.
        let clock = AudioClock::new(Arc::new(AtomicU64::new(0)), 44_100, 24, None);
        for f in [1, 7, 23, 100, 511] {
            clock.seek(f);
            assert_eq!(clock.frame(24), f, "seek({f}) must round-trip");
        }
    }

    #[test]
    fn streaming_timeline_extent_rounds_like_the_audio_clock_seek() {
        let mut timeline = Timeline::new();
        timeline.fps = 24;
        let mut track = Track::new("v1", ClipType::Video);
        track.clips.push(audio_clip("odd", "source", 0, 7));
        timeline.tracks.push(track);

        assert_eq!(timeline_audio_frames(&timeline, 44_100).unwrap(), 12_863);
    }

    #[test]
    fn clip_source_window_uses_timeline_fps() {
        let mut clip = Clip::new("c1", "asset-1", 0, 60);
        clip.trim_start_frame = 15;
        clip.speed = 1.0;
        let (lo, hi) = clip_source_window_secs(&clip, 30).expect("window");
        assert!((lo - 0.5).abs() < 1e-6);
        assert!((hi - 2.5).abs() < 1e-6);
    }

    #[test]
    fn project_clip_audio_stereo_skips_clip_without_media_entry() {
        let clip = Clip::new("c1", "missing", 0, 30);
        let media: HashMap<String, MediaInfo> = HashMap::new();
        assert!(
            project_clip_audio_stereo(&clip, &media, 30, 48_000, &MediaCancelToken::new())
                .expect("missing media is silent")
                .is_none()
        );
    }

    #[test]
    fn mix_timeline_stereo_empty_when_no_audio_clips() {
        let timeline = Timeline::new();
        let media: HashMap<String, MediaInfo> = HashMap::new();
        assert!(mix_timeline_stereo(
            &timeline,
            &media,
            48_000,
            0,
            &ProfileScope::new(),
            &MediaCancelToken::new()
        )
        .expect("empty timeline")
        .is_none());
    }

    #[test]
    fn mix_stereo_sums_placed_clips_and_clamps() {
        // Clip A at frame 0: 2 stereo frames [(0.6,-0.6),(0.5,0.5)].
        // Clip B at frame 1: 1 stereo frame (0.6,0.6) → overlaps A's frame 1.
        let a = StereoClip {
            start_frame: 0,
            interleaved: vec![0.6, -0.6, 0.5, 0.5],
            gains: Vec::new(),
            true_peak_ceiling_dbtp: None,
        };
        let b = StereoClip {
            start_frame: 1,
            interleaved: vec![0.6, 0.6],
            gains: Vec::new(),
            true_peak_ceiling_dbtp: None,
        };
        let out = mix_stereo(&[a, b], &MediaCancelToken::new()).expect("mix");
        assert_eq!(out.len(), 4); // 2 frames × 2 channels
                                  // frame 0 = A only.
        assert!((out[0] - 0.6).abs() < 1e-6);
        assert!((out[1] + 0.6).abs() < 1e-6);
        // frame 1 = A(0.5,0.5) + B(0.6,0.6) = (1.1,1.1) → clamped to (1.0,1.0).
        assert!((out[2] - 1.0).abs() < 1e-6);
        assert!((out[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mix_stereo_applies_per_frame_gain() {
        let c = StereoClip {
            start_frame: 0,
            interleaved: vec![1.0, 1.0, 1.0, 1.0],
            gains: vec![0.5, 0.25],
            true_peak_ceiling_dbtp: None,
        };
        let out = mix_stereo(&[c], &MediaCancelToken::new()).expect("mix");
        assert_eq!(out, vec![0.5, 0.5, 0.25, 0.25]);
    }

    #[test]
    fn mix_stereo_enforces_normalized_true_peak_with_codec_margin() {
        let clip = StereoClip {
            start_frame: 0,
            interleaved: vec![1.0, -1.0],
            gains: Vec::new(),
            true_peak_ceiling_dbtp: Some(-1.0),
        };
        let out = mix_stereo(&[clip], &MediaCancelToken::new()).expect("mix");
        let expected = 10.0_f32.powf(-3.0 / 20.0);
        assert!((out[0] - expected).abs() < 1e-6);
        assert!((out[1] + expected).abs() < 1e-6);
    }

    #[test]
    fn denoise_preview_uses_shared_processing_owner() {
        let config = opentake_domain::AudioDenoise {
            mode: opentake_domain::DenoiseMode::Adaptive,
            strength: 0.75,
            preview_enabled: true,
        };
        let input = vec![0.2, -0.1, 0.15, -0.05, 0.1, 0.0, 0.05, 0.05];
        let cancel = MediaCancelToken::new();
        let preview = apply_preview_denoise(&input, 2, 48_000, Some(config), &cancel)
            .expect("preview denoise");
        let shared =
            opentake_media::analysis::denoise_interleaved(&input, 2, 48_000, config, &cancel, None)
                .expect("shared denoise");
        assert_eq!(preview, shared);
    }

    /// Two noisy tones, the second denoised and faded in, over eight seconds.
    fn denoised_timeline(dir: &std::path::Path) -> (Timeline, HashMap<String, PathBuf>) {
        use crate::clip_audio::fixtures::{noisy_tone, write_wav};

        let first = dir.join("first.wav");
        let second = dir.join("second.wav");
        write_wav(&first, &noisy_tone(6.0, 440.0, 3));
        write_wav(&second, &noisy_tone(8.0, 550.0, 4));
        let mut speech = audio_clip("speech", "second", 60, 180);
        speech.trim_start_frame = 15;
        speech.fade_in_frames = 20;
        speech.audio_denoise = Some(AudioDenoise {
            mode: opentake_domain::DenoiseMode::Voice,
            strength: 0.8,
            preview_enabled: true,
        });
        let mut timeline = audio_timeline(vec![audio_clip("music", "first", 0, 150)]);
        let mut track = Track::new("a2", ClipType::Audio);
        track.clips.push(speech);
        timeline.tracks.push(track);
        let paths = HashMap::from([("first".to_string(), first), ("second".to_string(), second)]);
        (timeline, paths)
    }

    fn media_for(paths: &HashMap<String, PathBuf>) -> HashMap<String, MediaInfo> {
        paths
            .iter()
            .map(|(id, path)| (id.clone(), MediaInfo { path: path.clone() }))
            .collect()
    }

    #[test]
    fn preview_windows_render_the_same_samples_whatever_the_playback_start() {
        if !crate::clip_audio::fixtures::ffmpeg_ready() {
            eprintln!("skip: ffmpeg/ffprobe not available");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let (timeline, paths) = denoised_timeline(dir.path());
        let media = media_for(&paths);
        // Seeks inside the plain clip (0.7 s), inside the denoised clip
        // (2.08 s), and in the middle of an STFT hop (3.96 s), at the usual
        // device rate and at a Bluetooth hands-free rate, where the denoiser's
        // hop (128) and warm-up (2,560 frames) are shorter than the reader's
        // pre-roll.
        for (rate, starts) in [
            (48_000_u32, [33_600_u64, 100_000, 190_003]),
            (16_000, [11_200, 33_333, 63_335]),
        ] {
            let total = timeline_audio_frames(&timeline, rate).unwrap();
            assert_eq!(total, 8 * u64::from(rate));
            let window = rate as usize * STREAM_WINDOW_SECONDS;
            let cancel = MediaCancelToken::new();
            // Stream the whole timeline in two-second windows from `start`,
            // as the audio producer does after a play or seek there.
            let stream_from = |start: u64| {
                let mut sources = PreviewAudioSources::default();
                let mut out = Vec::new();
                let mut position = start;
                while position < total {
                    let len = (total - position).min(window as u64) as usize;
                    out.extend(
                        mix_timeline_window_channels(
                            &timeline,
                            &media,
                            rate,
                            MIX_CHANNELS,
                            position,
                            len,
                            &mut sources,
                            ProfileWait::Block,
                            &cancel,
                        )
                        .unwrap_or_else(|error| {
                            panic!("{rate} Hz window at {position} failed: {error}")
                        }),
                    );
                    position += len as u64;
                }
                out
            };
            let from_zero = stream_from(0);
            for start in starts {
                let seeked = stream_from(start);
                let offset = start as usize * MIX_CHANNELS;
                let max_difference = seeked
                    .iter()
                    .zip(&from_zero[offset..])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max);
                assert!(
                    max_difference < 1.0e-4,
                    "{rate} Hz playback from {start} differs by {max_difference}"
                );
            }
        }
    }

    #[test]
    fn preview_never_opens_more_clip_decoders_than_the_cap() {
        use crate::clip_audio::fixtures::{ffmpeg_ready, noisy_tone, write_wav};
        use crate::clip_audio::{reader_census, MAX_OPEN_CLIP_READERS};
        use opentake_media::ffmpeg_status::HelperProcessCount;

        if !ffmpeg_ready() {
            eprintln!("skip: ffmpeg/ffprobe not available");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("tone.wav");
        write_wav(&source, &noisy_tone(10.0, 440.0, 11));
        // Four more overlapping audible clips than the cap, entering at
        // staggered frames and all still playing in the third window.
        let extra = 4;
        let clips = (0..MAX_OPEN_CLIP_READERS + extra)
            .map(|index| {
                let start = index as i32 * 3;
                let mut clip = audio_clip(&format!("c{index}"), "tone", start, 180 - start);
                clip.trim_start_frame = index as i32;
                clip.volume = 0.05;
                clip
            })
            .collect::<Vec<_>>();
        let timeline = audio_timeline(clips);
        let paths = HashMap::from([("tone".to_string(), source)]);
        let media = media_for(&paths);
        let exported = crate::export::mix_timeline_audio_for_paths(&timeline, &paths)
            .unwrap()
            .expect("audible timeline");
        let rate = 48_000_u32;
        let total = timeline_audio_frames(&timeline, rate).unwrap() as usize;
        let window = rate as usize * STREAM_WINDOW_SECONDS;
        let census = reader_census::start();
        let processes = HelperProcessCount::start();
        let mut sources = PreviewAudioSources::default();
        let mut previewed = Vec::new();
        let mut position = 0;
        while position < total {
            let len = (total - position).min(window);
            previewed.extend(
                mix_timeline_window_channels(
                    &timeline,
                    &media,
                    rate,
                    1,
                    position as u64,
                    len,
                    &mut sources,
                    ProfileWait::Background,
                    &MediaCancelToken::new(),
                )
                .unwrap(),
            );
            assert!(
                census.live() < MAX_OPEN_CLIP_READERS,
                "readers kept between windows: {}",
                census.live()
            );
            position += len;
        }
        let spawned = processes.count();
        drop(processes);
        assert_eq!(
            census.peak(),
            MAX_OPEN_CLIP_READERS,
            "clip decoders open at once"
        );
        drop(sources);
        assert_eq!(census.live(), 0);
        // One probe, one decoder for each clip kept open, and one decoder per
        // window for each other clip.
        let windows = total.div_ceil(window);
        assert_eq!(windows, 3);
        let kept = MAX_OPEN_CLIP_READERS - 1;
        assert_eq!(spawned, 1 + kept + (extra + 1) * windows);
        // Readers opened for one window still continue the clip's samples.
        let max_difference = previewed
            .iter()
            .zip(&exported)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_difference < 1.0e-6,
            "preview differs from export by {max_difference}"
        );
    }

    #[test]
    fn preview_mix_matches_export_audio_with_the_same_channels() {
        if !crate::clip_audio::fixtures::ffmpeg_ready() {
            eprintln!("skip: ffmpeg/ffprobe not available");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let (timeline, paths) = denoised_timeline(dir.path());
        let media = media_for(&paths);
        let exported = crate::export::mix_timeline_audio_for_paths(&timeline, &paths)
            .unwrap()
            .expect("audible timeline");
        let rate = 48_000_u32;
        let total = timeline_audio_frames(&timeline, rate).unwrap() as usize;
        assert_eq!(exported.len(), total);
        // The preview mixes stereo; mixed as mono at the export rate it lands
        // on the export's samples, window by window.
        let cancel = MediaCancelToken::new();
        let mut sources = PreviewAudioSources::default();
        let mut previewed = Vec::new();
        let mut position = 0;
        while position < total {
            let len = (total - position).min(70_001);
            previewed.extend(
                mix_timeline_window_channels(
                    &timeline,
                    &media,
                    rate,
                    1,
                    position as u64,
                    len,
                    &mut sources,
                    ProfileWait::Block,
                    &cancel,
                )
                .unwrap(),
            );
            position += len;
        }
        let max_difference = previewed
            .iter()
            .zip(&exported)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_difference < 1.0e-5,
            "preview differs by {max_difference}"
        );
    }

    /// Wait until `ready` holds, failing after `timeout`.
    fn wait_for(timeout: Duration, what: &str, mut ready: impl FnMut() -> bool) {
        let deadline = Instant::now() + timeout;
        while !ready() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            thread::sleep(Duration::from_millis(10));
        }
    }

    /// Start playback preparation on another thread and wait (bounded) for it.
    fn prepare_playback(
        timeline: &Timeline,
        media: &HashMap<String, MediaInfo>,
        start: u64,
        profiles: &ProfileScope,
    ) -> PreparedTimelineAudio {
        let (timeline, media, profiles) = (timeline.clone(), media.clone(), profiles.clone());
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || {
            let prepared = mix_timeline_stereo(
                &timeline,
                &media,
                48_000,
                start,
                &profiles,
                &MediaCancelToken::new(),
            );
            let _ = sender.send(prepared);
        });
        receiver
            .recv_timeout(Duration::from_secs(30))
            .expect("playback prepare must not wait for the noise profile")
            .expect("prepare")
            .expect("audible timeline")
    }

    fn stop_playback(prepared: PreparedTimelineAudio) {
        prepared.control.stop();
        drop(prepared.consumer);
        prepared.producer.join().expect("audio producer");
    }

    #[test]
    fn playback_does_not_wait_for_a_denoise_profile_and_seeks_do_not_restart_it() {
        use crate::clip_audio::test_hooks;

        if !crate::clip_audio::fixtures::ffmpeg_ready() {
            eprintln!("skip: ffmpeg/ffprobe not available");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let (timeline, paths) = denoised_timeline(dir.path());
        let media = media_for(&paths);
        let denoised_source = paths["second"].clone();
        let mut undenoised = timeline.clone();
        for clip in undenoised
            .tracks
            .iter_mut()
            .flat_map(|track| &mut track.clips)
        {
            clip.audio_denoise = None;
        }
        let rate = 48_000_u32;
        let window = rate as usize * STREAM_WINDOW_SECONDS;
        let mix_undenoised = |start: u64| {
            mix_timeline_window_channels(
                &undenoised,
                &media,
                rate,
                MIX_CHANNELS,
                start,
                window,
                &mut PreviewAudioSources::default(),
                ProfileWait::Block,
                &MediaCancelToken::new(),
            )
            .unwrap()
        };

        // Hold the background pass: playback must start, seek and restart
        // while the profile is still being computed. The passes belong to
        // this playback state's scope, so other tests' project transitions
        // cannot cancel them.
        let profiles = ProfileScope::new();
        let hold = test_hooks::hold(&denoised_source);
        let start = 100_000;
        let first = prepare_playback(&timeline, &media, start, &profiles);
        let chunk = first
            .consumer
            .receiver
            .recv_timeout(Duration::from_secs(10))
            .expect("first window")
            .expect("first window mixed");
        assert_eq!(chunk.start_frame, start);
        assert!(
            chunk.samples == mix_undenoised(start),
            "until its profile is ready the clip plays undenoised"
        );
        assert_eq!(test_hooks::requests(&denoised_source), 1);

        // A seek cancels the window decode, not the profile pass.
        first.control.request_seek(150_000);
        let seeked = loop {
            let chunk = first
                .consumer
                .receiver
                .recv_timeout(Duration::from_secs(10))
                .expect("window after the seek")
                .expect("window mixed after the seek");
            if chunk.generation == 1 {
                break chunk;
            }
        };
        assert_eq!(seeked.start_frame, 150_000);
        // Stop and play again: a new session joins the queued pass.
        stop_playback(first);
        let second = prepare_playback(&timeline, &media, 150_000, &profiles);
        assert_eq!(test_hooks::requests(&denoised_source), 1, "one queued pass");
        assert_eq!(test_hooks::passes(&denoised_source), 0, "the pass is held");
        stop_playback(second);
        // Another playback state closing its project leaves the pass alone.
        ProfileScope::new().cancel();
        assert_eq!(test_hooks::requests(&denoised_source), 1);

        drop(hold);
        let speech = &timeline.tracks[1].clips[0];
        let layout = ClipAudioLayout::new(speech, timeline.fps, rate).unwrap();
        wait_for(Duration::from_secs(30), "the background profile", || {
            !crate::clip_audio::denoise_profile_pending(
                speech.audio_denoise,
                &layout,
                &denoised_source,
                MIX_CHANNELS,
            )
        });
        assert_eq!(test_hooks::passes(&denoised_source), 1, "one profile pass");
        assert_eq!(test_hooks::requests(&denoised_source), 1);

        // Once the profile is cached, playback denoises without another pass.
        let third = prepare_playback(&timeline, &media, start, &profiles);
        let chunk = third
            .consumer
            .receiver
            .recv_timeout(Duration::from_secs(10))
            .expect("first window")
            .expect("first window mixed");
        assert!(
            chunk.samples != mix_undenoised(start),
            "denoised once ready"
        );
        stop_playback(third);
        assert_eq!(test_hooks::passes(&denoised_source), 1);
    }

    #[test]
    fn preview_windows_continue_one_decode_like_export_for_resampled_aac() {
        use opentake_media::ffmpeg_status::HelperProcessCount;

        if !crate::clip_audio::fixtures::ffmpeg_ready() {
            eprintln!("skip: ffmpeg/ffprobe not available");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("tone.m4a");
        let status = std::process::Command::new(opentake_media::ffmpeg_status::ffmpeg_path())
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
            ])
            .arg("sine=frequency=440:sample_rate=44100:duration=7")
            .args(["-c:a", "aac", "-b:a", "128k"])
            .arg(&source)
            .status()
            .expect("run ffmpeg");
        if !status.success() {
            eprintln!("skip: ffmpeg could not encode the AAC fixture");
            return;
        }
        let mut clip = audio_clip("tone", "tone", 12, 160);
        clip.trim_start_frame = 7;
        clip.fade_in_frames = 10;
        let timeline = audio_timeline(vec![clip]);
        let paths = HashMap::from([("tone".to_string(), source)]);
        let media = media_for(&paths);
        let exported = crate::export::mix_timeline_audio_for_paths(&timeline, &paths)
            .unwrap()
            .expect("audible timeline");
        let rate = 48_000_u32;
        let total = exported.len();
        let window = rate as usize * STREAM_WINDOW_SECONDS;
        let processes = HelperProcessCount::start();
        let mut sources = PreviewAudioSources::default();
        let mut previewed = Vec::new();
        let mut position = 0;
        while position < total {
            let len = (total - position).min(window);
            previewed.extend(
                mix_timeline_window_channels(
                    &timeline,
                    &media,
                    rate,
                    1,
                    position as u64,
                    len,
                    &mut sources,
                    ProfileWait::Background,
                    &MediaCancelToken::new(),
                )
                .unwrap(),
            );
            position += len;
        }
        let spawned = processes.count();
        drop(processes);
        // Three windows over one clip: one audibility probe and one decoder.
        assert_eq!(spawned, 2, "helper processes for three preview windows");
        // One continuous 44.1 kHz AAC decode resampled once, as export reads
        // it: the window edges add nothing.
        let max_difference = previewed
            .iter()
            .zip(&exported)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_difference < 1.0e-6,
            "preview differs from export by {max_difference}"
        );
    }

    #[test]
    fn write_frame_maps_to_device_channels() {
        // Mono device: average L+R.
        let mut mono = [0.0f32; 1];
        write_frame(&mut mono, 1.0, -1.0);
        assert!((mono[0] - 0.0).abs() < 1e-6);

        // Stereo device: L/R passthrough.
        let mut stereo = [0.0f32; 2];
        write_frame(&mut stereo, 0.3, -0.4);
        assert_eq!(stereo, [0.3, -0.4]);

        // Surround device: L, R, then silence on the extra channels.
        let mut surround = [9.0f32; 4];
        write_frame(&mut surround, 0.3, -0.4);
        assert_eq!(surround, [0.3, -0.4, 0.0, 0.0]);
    }
}
