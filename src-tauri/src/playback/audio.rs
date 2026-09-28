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
//! The cpal callback never blocks, allocates, frees or prints. A single
//! producer decodes and mixes fixed windows into a bounded channel; the
//! callback uses only atomics, an uncontended `try_lock` and non-blocking
//! `try_recv`/`try_send`, emitting silence on underrun. Windows it has played
//! or found stale go back to the producer through a bounded recycle channel
//! (the producer frees them), and failures are recorded for the render loop,
//! never logged from the callback. Seek advances a generation, cancels the old
//! decode, and makes both producer and consumer discard stale windows before
//! audible output resumes.
//!
//! The clock reports what is audible, not what was handed to the device: each
//! callback publishes the frames it wrote plus the output latency derived from
//! its `OutputCallbackInfo` timestamps, and [`AudioClock`] subtracts that
//! delay from the device position (#69).
//!
//! Stereo is mixed once and mapped to the device's channel count in the callback
//! (mono downmix / >2 zero-fill). The mix itself is the export's: the render
//! plan's flattened audio clips (nested sequences included) read through the
//! shared clip readers with the same gain, true-peak ceiling and denoise
//! semantics ([`AudioPlanLike`]), parameterised by the device rate and done per
//! channel.

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use crossbeam_channel::{bounded, Receiver as ChunkReceiver, Sender as ChunkSender};

use opentake_domain::Timeline;
use opentake_media::{encode::mix::apply_true_peak_ceiling, MediaCancelToken, MediaError};
use opentake_render::{try_collect_audio_clips, AudioClipPlan};

use crate::clip_audio::{
    keep_clip_reader, AudioPlanLike, ClipAudioLayout, ClipAudioReader, PreviewDenoise,
};

pub(crate) use crate::clip_audio::ProfileScope;

use super::engine::{InstantClock, PlaybackClock};
use super::project::MediaInfo;

/// Default device sample rate when cpal can't report one (no device queried yet).
const FALLBACK_SAMPLE_RATE: u32 = 48_000;

/// The mix is always interleaved stereo; the callback maps it to the device's
/// channel count.
const MIX_CHANNELS: usize = 2;
const STREAM_WINDOW_SECONDS: usize = 2;
const STREAM_WINDOW_CAPACITY: usize = 4;
/// Window buffers the callback can hand back before the producer drains them:
/// every queued window plus the one playing and the one being sent, so a
/// `try_send` into the recycle channel never finds it full.
const RECYCLE_CAPACITY: usize = STREAM_WINDOW_CAPACITY + 2;
/// Longest wait for one query on the `opentake-audio-device` thread. A
/// driver that hangs there must not hang a resume: the query then answers
/// "unknown", which callers read as "the device did not change".
const DEVICE_QUERY_TIMEOUT: Duration = Duration::from_secs(2);
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
    /// Whether the output is muted: a paused session's producer closes its
    /// clip readers once its queue is full instead of holding them open.
    output: Arc<OutputState>,
    /// Window buffers the callback had to free itself because the recycle
    /// channel was gone or full. Stays zero in a healthy session.
    callback_frees: AtomicU64,
}

impl AudioStreamControl {
    #[cfg(test)]
    fn new(start_frame: u64) -> Self {
        Self::with_output(start_frame, Arc::new(OutputState::new(false)))
    }

    fn with_output(start_frame: u64, output: Arc<OutputState>) -> Self {
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
            output,
            callback_frees: AtomicU64::new(0),
        }
    }

    /// Record the failure of a window of `generation`. Only the first failure
    /// after each explicit seek is kept, so a broken clip is reported once
    /// rather than every window or every clock re-alignment. A window of a
    /// superseded generation reports nothing: the generation is checked under
    /// the lock [`Self::request_seek`] takes after advancing it, so a failure
    /// from before a seek can never be recorded under the seek's epoch.
    fn record_error(&self, generation: u64, message: String) {
        let mut pending = self
            .pending_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.generation.load(Ordering::Acquire) != generation {
            return;
        }
        let epoch = self.seek_epoch.load(Ordering::Acquire);
        if self.reported_epoch.swap(epoch, Ordering::AcqRel) == epoch {
            return;
        }
        *pending = Some(message);
    }

    fn take_error(&self) -> Option<String> {
        self.pending_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
    }

    /// A transport seek: restart the producer at `start_frame` and forget a
    /// failure reported for the previous position. The generation advances
    /// before the seek epoch, both under the error lock, so a window of the
    /// old position that fails meanwhile is dropped by [`Self::record_error`]
    /// instead of counting for the new one, and the first failure at the new
    /// position is never mistaken for an old one.
    fn request_seek(&self, start_frame: u64) {
        let mut pending = self
            .pending_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.restart_at(start_frame);
        self.seek_epoch.fetch_add(1, Ordering::AcqRel);
        pending.take();
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

/// The callback's end of the window stream. It never frees a window: one it
/// has played or found stale goes back to the producer through `recycle`.
struct AudioStreamConsumer {
    receiver: ChunkReceiver<AudioStreamChunk>,
    recycle: ChunkSender<Vec<f32>>,
    /// Buffers `recycle` had no room for, held until it has. Its capacity is
    /// reserved up front, so parking a buffer here does not allocate.
    parked: Vec<Vec<f32>>,
    control: Arc<AudioStreamControl>,
    current: Option<AudioStreamChunk>,
    /// The producer exited, so no later window arrives.
    terminated: bool,
}

impl AudioStreamConsumer {
    fn new(
        receiver: ChunkReceiver<AudioStreamChunk>,
        recycle: ChunkSender<Vec<f32>>,
        control: Arc<AudioStreamControl>,
    ) -> Self {
        Self {
            receiver,
            recycle,
            parked: Vec::with_capacity(RECYCLE_CAPACITY),
            control,
            current: None,
            terminated: false,
        }
    }

    /// Hand a window's buffer back to the producer without freeing it here.
    fn retire(&mut self, chunk: AudioStreamChunk) {
        while let Some(parked) = self.parked.pop() {
            if let Err(error) = self.recycle.try_send(parked) {
                self.parked.push(error.into_inner());
                break;
            }
        }
        let samples = match self.recycle.try_send(chunk.samples) {
            Ok(()) => return,
            Err(error) => error.into_inner(),
        };
        if self.parked.len() < self.parked.capacity() {
            self.parked.push(samples);
        } else {
            // Unreachable while the producer runs (the recycle channel holds
            // every buffer that can exist); counted so tests can prove it.
            self.control.callback_frees.fetch_add(1, Ordering::Relaxed);
            drop(samples);
        }
    }

    fn discard_stale(&mut self) {
        let generation = self.control.generation.load(Ordering::Acquire);
        if self
            .current
            .as_ref()
            .is_some_and(|chunk| chunk.generation != generation)
        {
            if let Some(stale) = self.current.take() {
                self.retire(stale);
            }
        }
        if self.current.is_some() {
            return;
        }
        loop {
            match self.receiver.try_recv() {
                Ok(chunk) if chunk.generation == generation => {
                    self.current = Some(chunk);
                    break;
                }
                Ok(stale) => self.retire(stale),
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
            if let Some(played) = self.current.take() {
                self.retire(played);
            }
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

/// Time source of [`AudioClock`]; injectable so stall/recovery tests run on
/// virtual time.
type ClockNow = Arc<dyn Fn() -> Instant + Send + Sync>;

/// Audio master clock: the playhead derives from the device frame position
/// (`pos`, in output audio frames), which the cpal callback advances as it
/// hands samples to the device, minus the output delay the callback measures
/// (`delay`) — so video follows the sound the user actually hears.
pub struct AudioClock {
    /// Output audio frames handed to the device so far (shared with the cpal
    /// callback).
    pos: Arc<AtomicU64>,
    /// Output frames between `pos` and the frame audible now, as the last
    /// callback measured them: the frames it wrote plus the device latency
    /// its `OutputCallbackInfo` timestamps report (#69).
    delay: Arc<AtomicU64>,
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
            delay: Arc::new(AtomicU64::new(0)),
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

    /// Read the output delay the output's callback measures.
    fn delayed_by(mut self, delay: Arc<AtomicU64>) -> Self {
        self.delay = delay;
        self
    }

    /// The output frame audible now: the device position minus the output
    /// delay, never below zero.
    fn audible_pos(&self, pos: u64) -> u64 {
        pos.saturating_sub(self.delay.load(Ordering::Acquire))
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
        // What is audible, not what was handed to the device: video must
        // not run ahead of the sound by the output latency. The playhead
        // still never moves back when the delay grows (`last_frame`).
        let audio_frame = audio_position_frame(self.audible_pos(pos), self.rate, fps);
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
                // Hand the device the samples that become audible at the
                // target once the output delay has elapsed.
                let target_pos = frame_audio_position(target_frame, self.rate, fps)
                    .saturating_add(self.delay.load(Ordering::Acquire));
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
            mute.0.halt();
        }
    }

    fn resumed(&self) {
        if let Some(mute) = &self.mute {
            mute.0.clear_halt();
        }
    }
}

const OUTPUT_MUTED: u8 = 1;
const OUTPUT_HALTED: u8 = 2;

/// The logical state of one output: muted (the callback plays silence) and
/// halted (the render loop paused itself on a fatal failure). One atomic
/// holds both, so a committing resume checks "not halted" and unmutes in one
/// step: a failure can never slip between the check and the unmute.
#[derive(Debug)]
pub(crate) struct OutputState(AtomicU8);

impl OutputState {
    fn new(muted: bool) -> Self {
        Self(AtomicU8::new(if muted { OUTPUT_MUTED } else { 0 }))
    }

    /// Whether the output plays silence.
    pub(crate) fn is_muted(&self) -> bool {
        self.0.load(Ordering::Acquire) & OUTPUT_MUTED != 0
    }

    fn mute(&self) {
        self.0.fetch_or(OUTPUT_MUTED, Ordering::AcqRel);
    }

    /// Mute and hold the output muted across a concurrently committing
    /// resume, until the render thread resumes again.
    fn halt(&self) {
        self.0
            .fetch_or(OUTPUT_MUTED | OUTPUT_HALTED, Ordering::AcqRel);
    }

    /// The render thread resumed: a failure it reported before no longer
    /// holds the output muted.
    fn clear_halt(&self) {
        self.0.fetch_and(!OUTPUT_HALTED, Ordering::AcqRel);
    }

    /// Unmute unless halted, atomically.
    fn unmute_unless_halted(&self) {
        let _ = self
            .0
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                (state & OUTPUT_HALTED == 0).then_some(state & !OUTPUT_MUTED)
            });
    }
}

/// The logical mute of one output, as the render loop's clock sees it.
#[derive(Clone)]
struct OutputMute(Arc<OutputState>);

/// A cloneable transport endpoint of one audio output thread. `pause` and
/// `mute` never wait; only `prepare_resume` blocks (for callback liveness or a
/// device rebuild), so callers run it without holding the playback slot lock.
#[derive(Clone)]
pub struct AudioControl {
    control_tx: Sender<AudioCmd>,
    output: Arc<OutputState>,
}

impl AudioControl {
    fn new(control_tx: Sender<AudioCmd>, output: Arc<OutputState>) -> Self {
        Self { control_tx, output }
    }

    /// Mute output immediately. The hardware stream keeps running silently so
    /// a later resume can prove callback liveness without trusting an
    /// asynchronous backend play/pause acknowledgement.
    pub fn pause(&self) -> Result<(), String> {
        self.output.mute();
        let (reply, _acknowledgement) = mpsc::channel();
        self.control_tx
            .send(AudioCmd::Pause(reply))
            .map_err(|_| "audio thread exited before transport control".to_string())
    }

    /// Prove that callbacks continue while output remains logically muted,
    /// rebuilding the stream first when the default output device changed or
    /// the stream reported an error. The caller can then seek/resume the video
    /// clock before committing audible output.
    ///
    /// A halt from an earlier failure is not cleared here but by the render
    /// thread when it processes this resume ([`PlaybackClock::resumed`]), so
    /// a failure the previous run reports while this handshake blocks still
    /// counts as old, and one the resumed run reports keeps the output muted.
    pub fn prepare_resume(&self) -> Result<(), String> {
        self.output.mute();
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
    /// its next block. Stays muted when the resumed render thread failed (and
    /// halted the output); the check and the unmute are one atomic step.
    pub fn commit_resume(&self) {
        self.output.unmute_unless_halted();
    }

    fn output_mute(&self) -> OutputMute {
        OutputMute(Arc::clone(&self.output))
    }

    pub fn mute(&self) {
        self.output.mute();
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
type SharedSamples = Arc<Mutex<AudioStreamConsumer>>;

/// What one output stream needs from the session.
#[derive(Clone)]
struct OutputShared {
    samples: SharedSamples,
    pos: Arc<AtomicU64>,
    /// The output delay each callback publishes for [`AudioClock`].
    delay: Arc<AtomicU64>,
    output: Arc<OutputState>,
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
    /// Play the prepared window stream through the default output device.
    /// Returns `Err` if the device/stream can't be set up (the caller falls
    /// back to the wall clock). Blocks until the stream is live so failures
    /// surface synchronously.
    fn start_stream(
        prepared: PreparedTimelineAudio,
        rate: u32,
        pos: Arc<AtomicU64>,
        delay: Arc<AtomicU64>,
    ) -> Result<Self, String> {
        let PreparedTimelineAudio {
            consumer,
            control: stream_control,
            producer: stream_producer,
        } = prepared;
        let output = Arc::clone(&stream_control.output);
        match spawn_output(|| CpalBackend, consumer, rate, pos, delay, &output) {
            Ok((control_tx, handle)) => Ok(Self {
                control: AudioControl::new(control_tx, output),
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
        output: &Arc<OutputState>,
        handle: JoinHandle<()>,
    ) -> Self {
        Self {
            control: AudioControl::new(control_tx, Arc::clone(output)),
            handle: Some(handle),
            stream_control: None,
            stream_producer: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn test_stub() -> (Self, Arc<OutputState>, Receiver<()>) {
        let (control_tx, control_rx) = mpsc::channel();
        let (stopped_tx, stopped_rx) = mpsc::channel();
        let paused = Arc::new(OutputState::new(false));
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
    pub(crate) fn test_failing_resume() -> (Self, Arc<OutputState>) {
        let (control_tx, control_rx) = mpsc::channel();
        let paused = Arc::new(OutputState::new(true));
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
    pub(crate) fn test_blocking_resume() -> (Self, Arc<OutputState>, Receiver<()>, Sender<()>) {
        let (control_tx, control_rx) = mpsc::channel();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let paused = Arc::new(OutputState::new(true));
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
    pub(crate) fn test_blocking_stop() -> (Self, Arc<OutputState>, Receiver<()>, Sender<()>) {
        let (control_tx, control_rx) = mpsc::channel();
        let (stopped_tx, stopped_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let paused = Arc::new(OutputState::new(false));
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
    samples: AudioStreamConsumer,
    rate: u32,
    pos: Arc<AtomicU64>,
    delay: Arc<AtomicU64>,
    output: &Arc<OutputState>,
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
        delay,
        output: Arc::clone(output),
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
                // Logical pause is established by muting the output before this
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
/// master-clock tick — and publishes the output delay to subtract from it.
fn out_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    shared: OutputShared,
) -> Result<cpal::Stream, String>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = (config.channels as usize).max(1);
    let device_rate = config.sample_rate.0;
    let OutputShared {
        samples,
        pos,
        delay,
        output,
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
            move |data: &mut [T], info: &cpal::OutputCallbackInfo| {
                callback_epoch.fetch_add(1, Ordering::Release);
                let latency = output_latency_frames(info.timestamp(), device_rate);
                // Only one stream of a session is ever alive (a rebuild drops
                // the old one first), so this lock is uncontended; never wait.
                let mut samples = match samples.try_lock() {
                    Ok(samples) => samples,
                    Err(std::sync::TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
                    Err(std::sync::TryLockError::WouldBlock) => {
                        silence(data);
                        delay.store(latency, Ordering::Release);
                        return;
                    }
                };
                let written = fill_output_block(&mut samples, &pos, &output, data, channels);
                // At this instant the first frame just written becomes audible
                // after `latency`, so the audible frame is `pos` minus what was
                // written and the latency.
                delay.store((written as u64).saturating_add(latency), Ordering::Release);
            },
            err_fn,
            None,
        )
        .map_err(|e| format!("build output stream: {e}"))
}

/// Output frames between the start of a callback's block and its playback,
/// from the callback's timestamps; zero when the backend reports none.
fn output_latency_frames(timestamp: cpal::OutputStreamTimestamp, rate: u32) -> u64 {
    latency_frames(timestamp.playback.duration_since(&timestamp.callback), rate)
}

fn latency_frames(latency: Option<Duration>, rate: u32) -> u64 {
    latency.map_or(0, |latency| {
        (latency.as_secs_f64() * f64::from(rate)) as u64
    })
}

fn silence<T: SizedSample + FromSample<f32>>(data: &mut [T]) {
    for sample in data.iter_mut() {
        *sample = T::from_sample(0.0f32);
    }
}

/// One callback's work, independent of the device sample type: play the
/// consumer's windows from `pos` into `data` (silence while muted or on
/// underrun) and return the output frames the clock advanced by.
fn fill_output_block<T: SizedSample + FromSample<f32>>(
    consumer: &mut AudioStreamConsumer,
    pos: &AtomicU64,
    output: &OutputState,
    data: &mut [T],
    channels: usize,
) -> usize {
    let out_frames = data.len() / channels;
    if output.is_muted() {
        consumer.discard_stale();
        silence(data);
        return 0;
    }
    // Atomically claim this block's start frame and advance the master clock.
    // A concurrent `seek` (store) is honored on the next callback; within a
    // block we play from the claimed start.
    let mut written = 0;
    while written < out_frames {
        let Some((start, count)) = claim_ready_audio_block(consumer, pos, out_frames - written)
        else {
            silence(&mut data[written * channels..]);
            return written;
        };
        for (i, frame) in data[written * channels..(written + count) * channels]
            .chunks_mut(channels)
            .enumerate()
        {
            let (left, right) = consumer.sample_frame(start.saturating_add(i as u64));
            write_frame(frame, left, right);
        }
        written += count;
    }
    written
}

/// Keep the audio master clock at the first undecoded sample. Once the next
/// window arrives, playback resumes at that exact sample rather than dropping
/// the start of a clip while the callback is emitting silence.
fn claim_ready_audio_block(
    consumer: &mut AudioStreamConsumer,
    pos: &AtomicU64,
    out_frames: usize,
) -> Option<(u64, usize)> {
    let start = pos.load(Ordering::Acquire);
    if !consumer.ready_at(start) {
        if !consumer.terminated {
            consumer.control.underruns.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        // A failed or finished producer can never fill the gap. Keep the
        // clock moving over silence instead of freezing playback.
        return Some((
            pos.fetch_add(out_frames as u64, Ordering::AcqRel),
            out_frames,
        ));
    }
    let chunk = consumer.current.as_ref().expect("ready chunk");
    let end = chunk
        .start_frame
        .saturating_add((chunk.samples.len() / MIX_CHANNELS) as u64);
    let count = out_frames.min((end - start) as usize);
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
    submit_device_query(worker, DEVICE_QUERY_TIMEOUT, query)
}

/// Run `query` on the device thread behind `worker`, waiting at most
/// `timeout` for the thread to take it and answer. A timeout answers `None`
/// ("unknown"): a resume then keeps its stream (the device did not change)
/// and a rate probe uses the fallback rate. A late answer is discarded.
fn submit_device_query<T: Send + 'static>(
    worker: &SyncSender<DeviceJob>,
    timeout: Duration,
    query: impl FnOnce() -> Option<T> + Send + 'static,
) -> Option<T> {
    let deadline = Instant::now() + timeout;
    let (reply_tx, reply_rx) = mpsc::sync_channel(1);
    let mut job: DeviceJob = Box::new(move || {
        let result = catch_unwind(AssertUnwindSafe(query)).unwrap_or(None);
        let _ = reply_tx.try_send(result);
    });
    // The queue holds one job; a hung query keeps it full.
    loop {
        match worker.try_send(job) {
            Ok(()) => break,
            Err(TrySendError::Full(returned)) if Instant::now() < deadline => {
                job = returned;
                thread::sleep(CALLBACK_POLL_INTERVAL);
            }
            Err(_) => return None,
        }
    }
    reply_rx
        .recv_timeout(deadline.saturating_duration_since(Instant::now()))
        .ok()
        .flatten()
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

/// An open clip reader, and the flattened clip and source it reads.
struct PreviewReader {
    /// Index of the clip in the session's flattened audio clips. Leaves of a
    /// nested sequence used by two compound clips share a clip id, so the
    /// index, not the id, names the reader's clip.
    clip: usize,
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

    /// Take the open reader that continues clip `clip` at clip frame `from`
    /// with the requested denoise, if there is one. Any other reader of the
    /// clip (an old layout, or a denoise that changed) is closed.
    fn take_reader(
        &mut self,
        clip: usize,
        path: &Path,
        layout: &ClipAudioLayout,
        from: usize,
        denoised: bool,
    ) -> Option<ClipAudioReader> {
        let mut taken = None;
        let mut index = 0;
        while index < self.readers.len() {
            let open = &self.readers[index];
            if open.clip != clip {
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

    /// Close every open reader (after a seek, a failed window, or while the
    /// session is paused).
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

/// What one playback session mixes: the render plan's audio clips (nested
/// sequences flattened, muted tracks dropped) at the timeline's fps.
struct PreviewMix<'a> {
    clips: &'a [AudioClipPlan],
    media: &'a HashMap<String, MediaInfo>,
    fps: i32,
    rate: u32,
}

fn mix_timeline_window(
    mix: &PreviewMix<'_>,
    window_start: u64,
    window_frames: usize,
    sources: &mut PreviewAudioSources,
    cancel: &MediaCancelToken,
) -> Result<Vec<f32>, MediaError> {
    mix_timeline_window_channels(
        mix,
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
/// preview's stereo except in the export-parity tests.
fn mix_timeline_window_channels(
    mix: &PreviewMix<'_>,
    channels: usize,
    window_start: u64,
    window_frames: usize,
    sources: &mut PreviewAudioSources,
    profiles: ProfileWait,
    cancel: &MediaCancelToken,
) -> Result<Vec<f32>, MediaError> {
    let result = mix_window_with_readers(
        mix,
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

fn mix_window_with_readers(
    mix: &PreviewMix<'_>,
    channels: usize,
    window_start: u64,
    window_frames: usize,
    sources: &mut PreviewAudioSources,
    profiles: ProfileWait,
    cancel: &MediaCancelToken,
) -> Result<Vec<f32>, MediaError> {
    let PreviewMix {
        clips,
        media,
        fps,
        rate,
    } = *mix;
    let sample_count = window_frames
        .checked_mul(channels)
        .ok_or_else(|| audio_buffer_too_large("streaming window sample count overflow"))?;
    let mut mixed = vec![0.0_f32; sample_count];
    let window_end = window_start.saturating_add(window_frames as u64);
    // Like export, the strictest ceiling of any clip (or of a compound clip
    // around it) applies to the whole timeline, so it cannot change with the
    // window alignment.
    let true_peak_ceiling_dbtp = clips
        .iter()
        .filter_map(AudioPlanLike::true_peak_ceiling_dbtp)
        .min_by(f64::total_cmp);
    let mut samples = Vec::new();
    // Readers this window leaves mid-clip; they replace `sources.readers` at
    // the end, which closes the previous window's readers nothing continued.
    // At most `MAX_OPEN_CLIP_READERS` readers are open at any moment: the
    // previous window's (`sources.readers`), the continuing ones and one
    // reader of a clip beyond the cap, closed right after it is read.
    let mut continuing = Vec::new();
    sources.close_readers_outside(window_start, window_end);
    for (index, plan) in clips.iter().enumerate() {
        if cancel.checkpoint() {
            return Err(MediaError::Cancelled);
        }
        let clip = plan.clip();
        if clip.duration_frames <= 0 {
            continue;
        }
        let Some(info) = media.get(&clip.media_ref) else {
            continue;
        };
        let Some(layout) = ClipAudioLayout::new(clip, fps, rate) else {
            continue;
        };
        let (clip_start, clip_end) = layout.span();
        let overlap_start = window_start.max(clip_start);
        let overlap_end = window_end.min(clip_end);
        if overlap_start >= overlap_end || !sources.has_audio(&info.path, cancel)? {
            continue;
        }
        let config = plan.audio_denoise().filter(|config| config.preview_enabled);
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
            match sources.take_reader(index, &info.path, &layout, from, denoise.is_some()) {
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
                clip: index,
                path: info.path.clone(),
                layout,
                reader,
            });
        } else {
            drop(reader);
        }
        let output_start = (overlap_start - window_start) as usize;
        for frame in 0..frames {
            let timeline_frame =
                crate::clip_audio::timeline_frame_at(overlap_start + frame as u64, fps, rate);
            // Compound clips multiply their volume into their leaves' gain.
            let gain = plan.volume_at(timeline_frame) as f32;
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

/// The producer's view of a session's mix.
trait WindowMixer {
    /// Mix output frames `[start, start + len)`.
    fn mix(
        &mut self,
        start: u64,
        len: usize,
        cancel: &MediaCancelToken,
    ) -> Result<Vec<f32>, MediaError>;

    /// Close every open clip reader; the next window reopens what it needs.
    fn close_readers(&mut self);
}

/// Free the window buffers the callback handed back.
fn drain_recycled(recycle: &ChunkReceiver<Vec<f32>>) {
    while let Ok(buffer) = recycle.try_recv() {
        drop(buffer);
    }
}

/// Queue `chunk`, waiting while the queue is full. Returns `false` when the
/// producer stopped or a newer generation made the chunk stale. While it
/// waits the producer frees recycled buffers, and a muted (paused) session
/// closes its clip readers: nothing is consumed until the session resumes,
/// so up to [`MAX_OPEN_CLIP_READERS`](crate::clip_audio::MAX_OPEN_CLIP_READERS)
/// FFmpeg processes would otherwise stay open for as long as it is paused.
fn send_stream_chunk(
    sender: &ChunkSender<AudioStreamChunk>,
    recycle: &ChunkReceiver<Vec<f32>>,
    mut chunk: AudioStreamChunk,
    control: &AudioStreamControl,
    mixer: &mut impl WindowMixer,
) -> bool {
    let generation = chunk.generation;
    loop {
        drain_recycled(recycle);
        if control.stopped.load(Ordering::Acquire)
            || control.generation.load(Ordering::Acquire) != generation
        {
            return false;
        }
        match sender.try_send(chunk) {
            Ok(()) => return true,
            Err(crossbeam_channel::TrySendError::Full(returned)) => {
                chunk = returned;
                if control.output.is_muted() {
                    mixer.close_readers();
                }
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
///
/// `first_window` is the token the prefilled first window was read with: the
/// readers it left open continue into the next windows, so it is this
/// generation's token and a seek or stop cancels it like any other.
fn run_audio_producer(
    control: &AudioStreamControl,
    sender: &ChunkSender<AudioStreamChunk>,
    recycle: &ChunkReceiver<Vec<f32>>,
    windows: ProducerWindows,
    mut mixer: impl WindowMixer,
    first_window: MediaCancelToken,
) {
    let ProducerWindows {
        rate,
        mut next_frame,
        total_frames,
        window_frames,
    } = windows;
    let mut generation = 0_u64;
    let mut window_cancel = first_window;
    loop {
        drain_recycled(recycle);
        if control.stopped.load(Ordering::Acquire) {
            break;
        }
        let observed = control.generation.load(Ordering::Acquire);
        if observed != generation {
            generation = observed;
            next_frame = control.requested_start.load(Ordering::Acquire);
        }
        if next_frame >= total_frames {
            if control.output.is_muted() {
                mixer.close_readers();
            }
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
        let result = mixer.mix(next_frame, len, &window_cancel);
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
                control.record_error(
                    generation,
                    format!(
                        "audio at {:.1} s could not be decoded and plays as silence: {error}",
                        next_frame as f64 / rate.max(1) as f64
                    ),
                );
                vec![0.0; len * MIX_CHANNELS]
            }
        };
        let chunk = AudioStreamChunk {
            generation,
            start_frame: next_frame,
            samples,
        };
        if send_stream_chunk(sender, recycle, chunk, control, &mut mixer) {
            next_frame = next_frame.saturating_add(len as u64);
            control.mark_buffered(generation, next_frame);
        }
    }
}

/// The production [`WindowMixer`]: the session's flattened clips, read
/// through its preview sources.
struct TimelineMixer {
    clips: Vec<AudioClipPlan>,
    media: HashMap<String, MediaInfo>,
    fps: i32,
    rate: u32,
    sources: PreviewAudioSources,
}

impl WindowMixer for TimelineMixer {
    fn mix(
        &mut self,
        start: u64,
        len: usize,
        cancel: &MediaCancelToken,
    ) -> Result<Vec<f32>, MediaError> {
        let mix = PreviewMix {
            clips: &self.clips,
            media: &self.media,
            fps: self.fps,
            rate: self.rate,
        };
        mix_timeline_window(&mix, start, len, &mut self.sources, cancel).map_err(
            |error| match error {
                MediaError::Cancelled => MediaError::Cancelled,
                error => MediaError::Decode(super::project::redact_media_paths(
                    &self.media,
                    &error.to_string(),
                )),
            },
        )
    }

    fn close_readers(&mut self) {
        self.sources.close_readers();
    }
}

/// Prepare bounded timeline-audio scheduling at `rate`. The initial window is
/// mixed synchronously so decode failures surface before playback ownership is
/// published; all subsequent windows are produced on one bounded worker.
///
/// The mix covers the render plan's flattened audio clips, so sound inside a
/// compound clip plays as it exports (#33). The first window is read with
/// `first_window`, which the producer adopts for the readers that window
/// leaves open: pass a token the session's prepare cancellation reaches (a
/// child of it); a seek or stop then cancels it too. `output` is the
/// session's output state, which starts muted for a prepared session.
fn mix_timeline_stereo(
    timeline: &Timeline,
    media: &HashMap<String, MediaInfo>,
    rate: u32,
    start_frame: u64,
    profiles: &ProfileScope,
    output: Arc<OutputState>,
    first_window: &MediaCancelToken,
) -> Result<Option<PreparedTimelineAudio>, MediaError> {
    if timeline.fps <= 0 || rate == 0 {
        return Ok(None);
    }
    let clips = try_collect_audio_clips(timeline).map_err(MediaError::Decode)?;
    let has_candidates = clips.iter().any(|plan| {
        let clip = plan.clip();
        clip.duration_frames > 0 && media.contains_key(&clip.media_ref)
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
    let mix = PreviewMix {
        clips: &clips,
        media,
        fps: timeline.fps,
        rate,
    };
    let first_samples =
        mix_timeline_window(&mix, start_frame, first_len, &mut sources, first_window)?;
    let (sender, receiver) = bounded(STREAM_WINDOW_CAPACITY);
    let (recycle_sender, recycle) = bounded(RECYCLE_CAPACITY);
    sender
        .send(AudioStreamChunk {
            generation: 0,
            start_frame,
            samples: first_samples,
        })
        .map_err(|_| MediaError::Decode("audio stream queue closed during prefill".to_string()))?;
    let control = Arc::new(AudioStreamControl::with_output(start_frame, output));
    control.mark_buffered(0, start_frame.saturating_add(first_len as u64));
    // Until the producer registers its own window token, a seek or stop
    // cancels the first window's readers through this one.
    *control
        .active_decode
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(first_window.clone());
    let producer_control = Arc::clone(&control);
    let mixer = TimelineMixer {
        clips,
        media: media.clone(),
        fps: timeline.fps,
        rate,
        sources,
    };
    let first_window = first_window.clone();
    let producer = thread::Builder::new()
        .name("opentake-audio-fill".to_string())
        .spawn(move || {
            run_audio_producer(
                &producer_control,
                &sender,
                &recycle,
                ProducerWindows {
                    rate,
                    next_frame: start_frame.saturating_add(first_len as u64),
                    total_frames,
                    window_frames,
                },
                mixer,
                first_window,
            )
        })
        .map_err(|error| MediaError::Decode(format!("spawn audio fill worker: {error}")))?;
    Ok(Some(PreparedTimelineAudio {
        consumer: AudioStreamConsumer::new(receiver, recycle_sender, Arc::clone(&control)),
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
    let output = Arc::new(OutputState::new(start_paused));
    // The prepare token reaches the first window's readers through this child,
    // which the session's seeks and stop cancel afterwards.
    let first_window = cancel.child();
    let Some(prepared) = mix_timeline_stereo(
        timeline,
        media,
        rate,
        start_audio_frame,
        profiles,
        output,
        &first_window,
    )?
    else {
        return Ok((Arc::new(InstantClock::new(start_frame)), None));
    };
    let pos = Arc::new(AtomicU64::new(start_audio_frame));
    let delay = Arc::new(AtomicU64::new(0));
    let clock = AudioClock::new(
        Arc::clone(&pos),
        rate,
        fps,
        Some(Arc::clone(&prepared.control)),
    )
    .delayed_by(Arc::clone(&delay));
    Ok(install_audio_clock(
        clock,
        AudioPlayback::start_stream(prepared, rate, pos, delay),
        start_frame,
    ))
}

/// The session's clock: the device clock when its output started, else the
/// wall clock from `start_frame` (a device that fails to start never
/// freezes playback).
fn install_audio_clock(
    clock: AudioClock,
    started: Result<AudioPlayback, String>,
    start_frame: i32,
) -> (Arc<dyn PlaybackClock>, Option<AudioPlayback>) {
    match started {
        Ok(audio) => {
            let clock = clock.muting(audio.control.output_mute());
            (Arc::new(clock), Some(audio))
        }
        Err(error) => {
            eprintln!("[audio] {error}; falling back to wall clock");
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

    use opentake_domain::{AudioDenoise, Clip, ClipType, Track};
    use opentake_media::{MediaCancelToken, MediaError};

    fn audio_timeline(clips: Vec<Clip>) -> Timeline {
        let mut timeline = Timeline::new();
        timeline.fps = 30;
        let mut track = Track::new("a1", ClipType::Audio);
        track.clips = clips;
        timeline.tracks.push(track);
        timeline
    }

    /// A consumer of `receiver` whose recycled buffers nobody drains.
    fn test_consumer(
        receiver: ChunkReceiver<AudioStreamChunk>,
        control: Arc<AudioStreamControl>,
    ) -> AudioStreamConsumer {
        let (recycle, _drained) = bounded(RECYCLE_CAPACITY);
        AudioStreamConsumer::new(receiver, recycle, control)
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
        // The prepare token reaches the first window through a child token.
        let cancel = MediaCancelToken::new();
        let first_window = cancel.child();
        let worker_window = first_window.clone();
        let (done_tx, done_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            done_tx
                .send(mix_timeline_stereo(
                    &timeline,
                    &media,
                    48_000,
                    0,
                    &ProfileScope::new(),
                    Arc::new(OutputState::new(true)),
                    &worker_window,
                ))
                .expect("publish audio prepare result");
        });

        let deadline = Instant::now() + Duration::from_secs(5);
        while first_window.spawned_child_count() == 0 && Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert_eq!(first_window.spawned_child_count(), 1);
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
            first_window.spawned_child_count(),
            1,
            "second clip must not spawn"
        );
    }

    #[test]
    fn streaming_consumer_discards_pre_seek_chunks_and_reports_underrun_as_silence() {
        let control = Arc::new(AudioStreamControl::new(0));
        let (sender, receiver) = bounded(4);
        sender
            .send(AudioStreamChunk {
                generation: 0,
                start_frame: 0,
                samples: vec![0.25, -0.25],
            })
            .unwrap();
        let mut consumer = test_consumer(receiver, Arc::clone(&control));
        assert_eq!(consumer.sample_frame(0), (0.25, -0.25));

        control.request_seek(10);
        sender
            .send(AudioStreamChunk {
                generation: 0,
                start_frame: 1,
                samples: vec![0.5, 0.5],
            })
            .unwrap();
        sender
            .send(AudioStreamChunk {
                generation: 1,
                start_frame: 10,
                samples: vec![0.75, -0.75],
            })
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
            .send(AudioStreamChunk {
                generation: 0,
                start_frame: first_sample,
                samples: vec![0.75, -0.75, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125],
            })
            .unwrap();
        let pos = Arc::new(AtomicU64::new(first_sample));
        let clock = AudioClock::new(Arc::clone(&pos), 48_000, 30, Some(Arc::clone(&control)));
        let mut samples = test_consumer(receiver, Arc::clone(&control));

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
            let consumer = &mut samples;
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
        let mut samples = test_consumer(receiver, Arc::clone(&control));

        assert_eq!(claim_ready_audio_block(&mut samples, &pos, 128), None);
        assert_eq!(pos.load(Ordering::Acquire), 0);
        sender
            .send(AudioStreamChunk {
                generation: 0,
                start_frame: 0,
                samples: vec![0.8, -0.8, 0.6, -0.6],
            })
            .unwrap();
        assert_eq!(claim_ready_audio_block(&mut samples, &pos, 2), Some((0, 2)));
        let consumer = &mut samples;
        assert_eq!(consumer.sample_frame(0), (0.8, -0.8));
        assert_eq!(consumer.sample_frame(1), (0.6, -0.6));
    }

    #[test]
    fn failed_audio_stream_keeps_the_clock_moving_with_silence() {
        let control = Arc::new(AudioStreamControl::new(0));
        let (sender, receiver) = bounded(4);
        // The producer exited: no later window arrives.
        drop(sender);
        let pos = AtomicU64::new(0);
        let mut samples = test_consumer(receiver, control);

        assert_eq!(
            claim_ready_audio_block(&mut samples, &pos, 128),
            Some((0, 128))
        );
        assert_eq!(pos.load(Ordering::Acquire), 128);
        let consumer = &mut samples;
        assert_eq!(consumer.sample_frame(0), (0.0, 0.0));
    }

    #[test]
    fn callback_crossing_a_window_boundary_waits_without_skipping_samples() {
        let control = Arc::new(AudioStreamControl::new(0));
        let (sender, receiver) = bounded(4);
        sender
            .send(AudioStreamChunk {
                generation: 0,
                start_frame: 0,
                samples: vec![0.5, 0.5, 0.5, 0.5],
            })
            .unwrap();
        let pos = AtomicU64::new(0);
        let mut samples = test_consumer(receiver, control);

        assert_eq!(
            claim_ready_audio_block(&mut samples, &pos, 128),
            Some((0, 2))
        );
        assert_eq!(claim_ready_audio_block(&mut samples, &pos, 126), None);
        assert_eq!(pos.load(Ordering::Acquire), 2);
        sender
            .send(AudioStreamChunk {
                generation: 0,
                start_frame: 2,
                samples: vec![0.75, -0.75, 0.75, -0.75],
            })
            .unwrap();
        assert_eq!(
            claim_ready_audio_block(&mut samples, &pos, 126),
            Some((2, 2))
        );
        let consumer = &mut samples;
        assert_eq!(consumer.sample_frame(2), (0.75, -0.75));
    }

    #[test]
    fn paused_stream_drain_retains_the_next_current_generation_chunk() {
        let control = Arc::new(AudioStreamControl::new(0));
        let (sender, receiver) = bounded(4);
        for (start_frame, sample) in [(0, 0.25), (1, 0.75)] {
            sender
                .send(AudioStreamChunk {
                    generation: 0,
                    start_frame,
                    samples: vec![sample, -sample],
                })
                .unwrap();
        }
        let mut consumer = test_consumer(receiver, control);

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
            Arc::new(OutputState::new(true)),
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
            submit_device_query(&job_tx, DEVICE_QUERY_TIMEOUT, || -> Option<u32> {
                panic!("simulated CPAL query panic")
            }),
            None
        );
        assert_eq!(
            submit_device_query(&job_tx, DEVICE_QUERY_TIMEOUT, || Some(44_100)),
            Some(44_100)
        );
        assert_eq!(
            submit_device_query(&job_tx, DEVICE_QUERY_TIMEOUT, || Some(
                "speakers".to_string()
            )),
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
        // The render thread resumes, then fails before the commit.
        clock.resumed();
        clock.halt();
        audio.commit_resume();
        assert!(paused.is_muted(), "a failed render keeps sound off");

        // The next resume is a retry and unmutes normally.
        audio.prepare_resume().expect("prepare again");
        clock.resumed();
        audio.commit_resume();
        assert!(!paused.is_muted());
    }

    #[test]
    fn a_failure_of_the_previous_run_does_not_mute_a_retried_resume() {
        let (audio, output, _stopped) = AudioPlayback::test_stub();
        let clock = AudioClock::new(Arc::new(AtomicU64::new(0)), 48_000, 30, None)
            .muting(audio.control().output_mute());

        // Play, pause, play again: the retried resume's handshake is under way
        // when the first run's render failure lands.
        audio.prepare_resume().expect("prepare the retry");
        clock.halt();
        // The render thread then processes the retried resume, which renders
        // (and would show video) again: its commit must unmute.
        clock.resumed();
        audio.commit_resume();
        assert!(!output.is_muted(), "video and sound resume together");
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
                .send(AudioStreamChunk {
                    generation: 0,
                    start_frame: window * 100,
                    samples: vec![window as f32; 100 * MIX_CHANNELS],
                })
                .unwrap();
        }
        let mut consumer = test_consumer(receiver, Arc::clone(&control));
        assert_eq!(consumer.sample_frame(10), (0.0, 0.0));
        assert_eq!(consumer.sample_frame(250), (2.0, 2.0));
        assert_eq!(control.underruns.load(Ordering::Acquire), 0);
    }

    #[test]
    fn a_seek_forgets_an_unreported_failure_but_realignment_does_not_repeat_it() {
        let control = AudioStreamControl::new(0);
        control.record_error(0, "clip-2 failed at 1.0 s".to_string());
        control.request_seek(500);
        assert_eq!(
            control.take_error(),
            None,
            "the error belonged to the old position"
        );

        let generation = control.generation.load(Ordering::Acquire);
        control.record_error(generation, "clip-2 failed at 6.0 s".to_string());
        assert!(control.take_error().is_some());
        // A clock re-alignment restarts the producer, which hits the same clip.
        control.restart_at(700);
        let generation = control.generation.load(Ordering::Acquire);
        control.record_error(generation, "clip-2 failed at 7.0 s".to_string());
        assert_eq!(
            control.take_error(),
            None,
            "reported once until the next seek"
        );
    }

    #[test]
    fn a_window_that_fails_across_a_seek_is_not_reported_for_the_new_position() {
        let control = AudioStreamControl::new(0);
        // The producer is mixing a window of generation 0 when the seek lands;
        // its failure arrives after the seek completed.
        let window_generation = control.generation.load(Ordering::Acquire);
        control.request_seek(500);
        control.record_error(window_generation, "clip-2 failed at 0.0 s".to_string());
        assert_eq!(control.take_error(), None, "a stale window reports nothing");

        // The first failure at the new position is still reported, once.
        let generation = control.generation.load(Ordering::Acquire);
        control.record_error(generation, "clip-3 failed at 5.0 s".to_string());
        assert_eq!(
            control.take_error().as_deref(),
            Some("clip-3 failed at 5.0 s")
        );
    }

    /// A [`WindowMixer`] around a closure, counting reader closes.
    struct FnMixer<F> {
        mix: F,
        closes: Arc<AtomicU64>,
    }

    impl<F> WindowMixer for FnMixer<F>
    where
        F: FnMut(u64, usize, &MediaCancelToken) -> Result<Vec<f32>, MediaError>,
    {
        fn mix(
            &mut self,
            start: u64,
            len: usize,
            cancel: &MediaCancelToken,
        ) -> Result<Vec<f32>, MediaError> {
            (self.mix)(start, len, cancel)
        }

        fn close_readers(&mut self) {
            self.closes.fetch_add(1, Ordering::AcqRel);
        }
    }

    fn fn_mixer<F>(mix: F) -> FnMixer<F>
    where
        F: FnMut(u64, usize, &MediaCancelToken) -> Result<Vec<f32>, MediaError>,
    {
        FnMixer {
            mix,
            closes: Arc::new(AtomicU64::new(0)),
        }
    }

    #[test]
    fn producer_reports_a_failed_window_once_and_recovers_after_seek() {
        let control = Arc::new(AudioStreamControl::new(0));
        let (sender, receiver) = bounded(STREAM_WINDOW_CAPACITY);
        let (recycle_sender, recycle) = bounded(RECYCLE_CAPACITY);
        let failing = Arc::new(AtomicBool::new(true));
        let producer_control = Arc::clone(&control);
        let producer_failing = Arc::clone(&failing);
        let producer = thread::spawn(move || {
            run_audio_producer(
                &producer_control,
                &sender,
                &recycle,
                ProducerWindows {
                    rate: 100,
                    next_frame: 0,
                    total_frames: 10_000,
                    window_frames: 100,
                },
                fn_mixer(|start, len, _cancel: &MediaCancelToken| {
                    // Every window from 100 on hits a broken clip.
                    if start >= 100 && producer_failing.load(Ordering::Acquire) {
                        return Err(MediaError::Decode("clip-2 decode failed".to_string()));
                    }
                    Ok(vec![0.5; len * MIX_CHANNELS])
                }),
                MediaCancelToken::new(),
            )
        });

        let next_chunk = || {
            receiver
                .recv_timeout(Duration::from_secs(2))
                .expect("producer keeps producing")
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
        let mut consumer =
            AudioStreamConsumer::new(receiver.clone(), recycle_sender, Arc::clone(&control));
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
        let paused = Arc::new(OutputState::new(true));
        let (_chunks, receiver) = bounded(STREAM_WINDOW_CAPACITY);
        let (control_tx, handle) = spawn_output(
            move || backend,
            test_consumer(receiver, Arc::new(AudioStreamControl::new(0))),
            48_000,
            Arc::new(AtomicU64::new(0)),
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
        let device = AudioClock::new(Arc::new(AtomicU64::new(0)), 48_000, 100, None);
        let (clock, audio) =
            install_audio_clock(device, Err("callback readiness timeout".to_string()), 0);

        assert!(audio.is_none());
        std::thread::sleep(Duration::from_millis(25));
        assert!(
            clock.frame(100) >= 1,
            "fallback clock must keep playback live"
        );
    }

    #[test]
    fn successful_audio_start_retains_device_clock() {
        let pos = Arc::new(AtomicU64::new(0));
        let device = AudioClock::new(Arc::clone(&pos), 48_000, 30, None);
        let (clock, audio) = install_audio_clock(device, Ok(AudioPlayback::test_stub().0), 0);
        pos.store(48_000, Ordering::Release);

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
        let (lo, hi) = crate::clip_audio::clip_source_window_secs(&clip, 30).expect("window");
        assert!((lo - 0.5).abs() < 1e-6);
        assert!((hi - 2.5).abs() < 1e-6);
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
            Arc::new(OutputState::new(true)),
            &MediaCancelToken::new()
        )
        .expect("empty timeline")
        .is_none());
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
        let clips = try_collect_audio_clips(&timeline).unwrap();
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
                            &PreviewMix {
                                clips: &clips,
                                media: &media,
                                fps: timeline.fps,
                                rate,
                            },
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
        let clips = try_collect_audio_clips(&timeline).unwrap();
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
                    &PreviewMix {
                        clips: &clips,
                        media: &media,
                        fps: timeline.fps,
                        rate,
                    },
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
        let clips = try_collect_audio_clips(&timeline).unwrap();
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
                    &PreviewMix {
                        clips: &clips,
                        media: &media,
                        fps: timeline.fps,
                        rate,
                    },
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
                Arc::new(OutputState::new(false)),
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
        let undenoised_clips = try_collect_audio_clips(&undenoised).unwrap();
        let mix_undenoised = |start: u64| {
            mix_timeline_window_channels(
                &PreviewMix {
                    clips: &undenoised_clips,
                    media: &media,
                    fps: undenoised.fps,
                    rate,
                },
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
            .expect("first window");
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
                .expect("window after the seek");
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
            .expect("first window");
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
        let clips = try_collect_audio_clips(&timeline).unwrap();
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
                    &PreviewMix {
                        clips: &clips,
                        media: &media,
                        fps: timeline.fps,
                        rate,
                    },
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

    // --- The production mix: flattened clips, gains, ceilings (#33, #69) ---

    const MIX_TEST_RATE: u32 = 48_000;
    /// Output frames per timeline frame at 30 fps and 48 kHz.
    const FRAME: usize = 1_600;

    /// A mono WAV holding `value` for `seconds`.
    fn dc_wav(dir: &std::path::Path, name: &str, value: f32, seconds: f32) -> PathBuf {
        let path = dir.join(format!("{name}.wav"));
        crate::clip_audio::fixtures::write_wav(
            &path,
            &vec![value; (seconds * MIX_TEST_RATE as f32) as usize],
        );
        path
    }

    /// Mix the whole timeline as preview does, in mono at 48 kHz (the export's
    /// channel layout), in windows of `window` frames.
    fn preview_mono(
        timeline: &Timeline,
        paths: &HashMap<String, PathBuf>,
        window: usize,
    ) -> Vec<f32> {
        let clips = try_collect_audio_clips(timeline).expect("flatten audio");
        let media = media_for(paths);
        let mix = PreviewMix {
            clips: &clips,
            media: &media,
            fps: timeline.fps,
            rate: MIX_TEST_RATE,
        };
        let total = timeline_audio_frames(timeline, MIX_TEST_RATE).unwrap() as usize;
        let mut sources = PreviewAudioSources::default();
        let mut out = Vec::with_capacity(total);
        let mut position = 0;
        while position < total {
            let len = (total - position).min(window);
            out.extend(
                mix_timeline_window_channels(
                    &mix,
                    1,
                    position as u64,
                    len,
                    &mut sources,
                    ProfileWait::Block,
                    &MediaCancelToken::new(),
                )
                .expect("mix window"),
            );
            position += len;
        }
        out
    }

    fn assert_span(samples: &[f32], range: std::ops::Range<usize>, expected: f32, what: &str) {
        let worst = samples[range.clone()]
            .iter()
            .map(|sample| (sample - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            worst < 1.0e-3,
            "{what}: samples {range:?} differ from {expected} by up to {worst}"
        );
    }

    /// A root timeline whose only clip is compound clip `compound` over a
    /// nested sequence holding `inner` on one audio track.
    fn compound_timeline(inner: Vec<Clip>, compound: Clip, nested_muted: bool) -> Timeline {
        let mut child = audio_timeline(inner);
        child.tracks[0].muted = nested_muted;
        let mut root = Timeline::new();
        root.fps = 30;
        root.nested_sequences
            .push(opentake_domain::NestedSequence::new(
                "sequence", "Sequence", child,
            ));
        let mut track = Track::new("v1", ClipType::Video);
        track.clips.push(compound);
        root.tracks.push(track);
        root
    }

    #[test]
    fn preview_mix_adds_clips_by_gain_clamps_and_skips_muted_tracks() {
        if !crate::clip_audio::fixtures::ffmpeg_ready() {
            eprintln!("skip: ffmpeg/ffprobe not available");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let paths = HashMap::from([
            ("half".to_string(), dc_wav(dir.path(), "half", 0.5, 2.0)),
            (
                "quarter".to_string(),
                dc_wav(dir.path(), "quarter", 0.25, 2.0),
            ),
            ("loud".to_string(), dc_wav(dir.path(), "loud", 0.9, 2.0)),
        ]);
        let mut timeline = audio_timeline(vec![audio_clip("a", "half", 0, 30)]);
        let mut quiet = audio_clip("b", "quarter", 15, 30);
        quiet.volume = 0.5;
        for (id, clips, muted) in [
            ("a2", vec![quiet], false),
            ("a3", vec![audio_clip("c", "loud", 0, 15)], false),
            ("muted", vec![audio_clip("m", "loud", 0, 45)], true),
        ] {
            let mut track = Track::new(id, ClipType::Audio);
            track.clips = clips;
            track.muted = muted;
            timeline.tracks.push(track);
        }

        let mixed = preview_mono(&timeline, &paths, 2 * MIX_TEST_RATE as usize);
        assert_eq!(mixed.len(), 45 * FRAME);
        // 0.5 + 0.9 clamps to 1; the muted track adds nothing anywhere.
        assert_span(&mixed, 0..15 * FRAME, 1.0, "clamped sum");
        assert_span(
            &mixed,
            15 * FRAME..30 * FRAME,
            0.5 + 0.25 * 0.5,
            "gained sum",
        );
        assert_span(&mixed, 30 * FRAME..45 * FRAME, 0.25 * 0.5, "gained clip");
    }

    #[test]
    fn preview_mix_applies_volume_envelopes_per_frame_across_window_boundaries() {
        if !crate::clip_audio::fixtures::ffmpeg_ready() {
            eprintln!("skip: ffmpeg/ffprobe not available");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let paths = HashMap::from([("half".to_string(), dc_wav(dir.path(), "half", 0.5, 2.0))]);
        let mut clip = audio_clip("fade", "half", 3, 40);
        clip.fade_in_frames = 12;
        clip.fade_out_frames = 9;
        let timeline = audio_timeline(vec![clip.clone()]);

        let whole = preview_mono(&timeline, &paths, 64 * FRAME);
        for frame in 0..timeline.total_frames() {
            let gain = if (3..43).contains(&frame) {
                clip.volume_at(frame) as f32
            } else {
                0.0
            };
            let start = frame as usize * FRAME;
            assert_span(
                &whole,
                start..start + FRAME,
                0.5 * gain,
                &format!("frame {frame}"),
            );
        }
        // Windows that start mid-frame land every sample in the same place.
        for window in [1_234, 70_001] {
            assert_eq!(preview_mono(&timeline, &paths, window), whole, "{window}");
        }
    }

    #[test]
    fn preview_mix_enforces_the_strictest_true_peak_ceiling() {
        if !crate::clip_audio::fixtures::ffmpeg_ready() {
            eprintln!("skip: ffmpeg/ffprobe not available");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let paths = HashMap::from([("loud".to_string(), dc_wav(dir.path(), "loud", 0.9, 1.0))]);
        let mut limited = audio_clip("limited", "loud", 0, 15);
        limited.loudness_normalization = Some(opentake_domain::LoudnessNormalization {
            target_lufs: -14.0,
            true_peak_ceiling_dbtp: -1.0,
            input_integrated_lufs: -14.0,
            input_true_peak_dbtp: -1.0,
            gain_db: 0.0,
            output_integrated_lufs: -14.0,
            output_true_peak_dbtp: -1.0,
        });
        // The ceiling of one clip applies to the whole timeline, as in export.
        let timeline = audio_timeline(vec![limited, audio_clip("free", "loud", 15, 15)]);

        let mixed = preview_mono(&timeline, &paths, 2 * MIX_TEST_RATE as usize);
        // -1 dBTP less the export's 2 dB codec margin.
        let ceiling = 10.0_f32.powf(-3.0 / 20.0);
        assert_span(&mixed, 0..30 * FRAME, ceiling, "limited to the ceiling");
    }

    #[test]
    fn preview_mix_skips_a_clip_without_a_media_entry() {
        let timeline = audio_timeline(vec![audio_clip("c1", "missing", 0, 30)]);
        let mixed = preview_mono(&timeline, &HashMap::new(), 2 * MIX_TEST_RATE as usize);
        assert_eq!(mixed.len(), 30 * FRAME);
        assert!(mixed.iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn sound_inside_a_compound_clip_plays_in_preview_as_it_exports() {
        if !crate::clip_audio::fixtures::ffmpeg_ready() {
            eprintln!("skip: ffmpeg/ffprobe not available");
            return;
        }
        use crate::clip_audio::fixtures::{noisy_tone, write_wav};
        let dir = tempfile::tempdir().unwrap();
        let tone = dir.path().join("tone.wav");
        write_wav(&tone, &noisy_tone(4.0, 440.0, 7));
        let paths = HashMap::from([("tone".to_string(), tone)]);
        let mut inner = audio_clip("inner", "tone", 6, 90);
        inner.trim_start_frame = 4;
        inner.fade_in_frames = 8;
        let mut compound = Clip::new_nested("compound", "sequence", 10, 70);
        compound.trim_start_frame = 12;
        compound.volume = 0.7;
        let timeline = compound_timeline(vec![inner], compound, false);

        let previewed = preview_mono(&timeline, &paths, 2 * MIX_TEST_RATE as usize);
        let exported = crate::export::mix_timeline_audio_for_paths(&timeline, &paths)
            .unwrap()
            .expect("export hears the compound clip");
        assert_eq!(previewed.len(), exported.len());
        let span = 10 * FRAME..80 * FRAME;
        let rms = (previewed[span.clone()]
            .iter()
            .map(|sample| sample * sample)
            .sum::<f32>()
            / span.len() as f32)
            .sqrt();
        assert!(rms > 0.05, "the compound clip is audible in preview: {rms}");
        let max_difference = previewed
            .iter()
            .zip(&exported)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_difference < 1.0e-5,
            "preview differs from export by {max_difference}"
        );
    }

    #[test]
    fn a_compound_clip_multiplies_its_volume_and_ceiling_into_its_leaves() {
        if !crate::clip_audio::fixtures::ffmpeg_ready() {
            eprintln!("skip: ffmpeg/ffprobe not available");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let paths = HashMap::from([("dc".to_string(), dc_wav(dir.path(), "dc", 0.8, 2.0))]);
        let mut inner = audio_clip("inner", "dc", 0, 60);
        inner.volume = 0.5;
        let mut compound = Clip::new_nested("compound", "sequence", 0, 60);
        compound.volume = 0.5;
        let timeline = compound_timeline(vec![inner.clone()], compound.clone(), false);
        let mixed = preview_mono(&timeline, &paths, 2 * MIX_TEST_RATE as usize);
        assert_span(&mixed, 0..60 * FRAME, 0.8 * 0.25, "0.5 x 0.5 = 0.25");

        // A ceiling on the compound clip limits the sound inside it.
        inner.volume = 1.0;
        compound.volume = 1.0;
        compound.loudness_normalization = Some(opentake_domain::LoudnessNormalization {
            target_lufs: -14.0,
            true_peak_ceiling_dbtp: -6.0,
            input_integrated_lufs: -14.0,
            input_true_peak_dbtp: -1.0,
            gain_db: 0.0,
            output_integrated_lufs: -14.0,
            output_true_peak_dbtp: -6.0,
        });
        let timeline = compound_timeline(vec![inner], compound, false);
        let mixed = preview_mono(&timeline, &paths, 2 * MIX_TEST_RATE as usize);
        assert_span(
            &mixed,
            0..60 * FRAME,
            10.0_f32.powf(-8.0 / 20.0),
            "compound ceiling",
        );
    }

    #[test]
    fn a_left_trimmed_compound_clip_reads_the_matching_source_position() {
        if !crate::clip_audio::fixtures::ffmpeg_ready() {
            eprintln!("skip: ffmpeg/ffprobe not available");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        // One value per source second: 0.1, 0.2, 0.3, 0.4.
        let steps = dir.path().join("steps.wav");
        crate::clip_audio::fixtures::write_wav(
            &steps,
            &(0..4 * MIX_TEST_RATE as usize)
                .map(|index| 0.1 * (index / MIX_TEST_RATE as usize + 1) as f32)
                .collect::<Vec<_>>(),
        );
        let paths = HashMap::from([("steps".to_string(), steps)]);
        let mut compound = Clip::new_nested("compound", "sequence", 0, 60);
        compound.trim_start_frame = 30;
        let timeline =
            compound_timeline(vec![audio_clip("inner", "steps", 0, 120)], compound, false);

        let mixed = preview_mono(&timeline, &paths, 2 * MIX_TEST_RATE as usize);
        // The compound's first second shows the sequence's second second.
        assert_span(&mixed, 100..30 * FRAME - 100, 0.2, "first root second");
        assert_span(
            &mixed,
            30 * FRAME + 100..60 * FRAME - 100,
            0.3,
            "second root second",
        );
    }

    #[test]
    fn a_muted_nested_track_is_silent() {
        let dir = tempfile::tempdir().unwrap();
        let paths = HashMap::from([("dc".to_string(), dc_wav(dir.path(), "dc", 0.8, 2.0))]);
        let timeline = compound_timeline(
            vec![audio_clip("inner", "dc", 0, 60)],
            Clip::new_nested("compound", "sequence", 0, 60),
            true,
        );
        assert!(try_collect_audio_clips(&timeline).unwrap().is_empty());
        assert!(preview_mono(&timeline, &paths, 2 * MIX_TEST_RATE as usize)
            .iter()
            .all(|sample| *sample == 0.0));
        let prepared = mix_timeline_stereo(
            &timeline,
            &media_for(&paths),
            MIX_TEST_RATE,
            0,
            &ProfileScope::new(),
            Arc::new(OutputState::new(true)),
            &MediaCancelToken::new(),
        )
        .expect("prepare");
        assert!(prepared.is_none(), "nothing audible, no audio stream");
    }

    #[test]
    fn a_timeline_whose_only_sound_is_inside_a_compound_clip_creates_an_audio_stream() {
        if !crate::clip_audio::fixtures::ffmpeg_ready() {
            eprintln!("skip: ffmpeg/ffprobe not available");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let paths = HashMap::from([("dc".to_string(), dc_wav(dir.path(), "dc", 0.8, 2.0))]);
        let timeline = compound_timeline(
            vec![audio_clip("inner", "dc", 0, 60)],
            Clip::new_nested("compound", "sequence", 0, 60),
            false,
        );
        let prepared = prepare_playback(&timeline, &media_for(&paths), 0, &ProfileScope::new());
        let first = prepared
            .consumer
            .receiver
            .recv_timeout(Duration::from_secs(10))
            .expect("first window");
        assert!(first.samples.iter().any(|sample| sample.abs() > 0.5));
        stop_playback(prepared);
    }

    // --- The realtime callback frees nothing (#69-2) ---

    #[test]
    fn the_callback_hands_every_played_or_stale_window_back_to_the_producer() {
        let control = Arc::new(AudioStreamControl::new(0));
        let (sender, receiver) = bounded(STREAM_WINDOW_CAPACITY);
        let (recycle_sender, recycle) = bounded(RECYCLE_CAPACITY);
        let mut consumer = AudioStreamConsumer::new(receiver, recycle_sender, Arc::clone(&control));
        let mut sent = Vec::new();
        let mut send = |generation: u64, start_frame: u64, value: f32| {
            let samples = vec![value; 4 * MIX_CHANNELS];
            sent.push(samples.as_ptr() as usize);
            sender
                .send(AudioStreamChunk {
                    generation,
                    start_frame,
                    samples,
                })
                .unwrap();
        };
        let pos = AtomicU64::new(0);
        let output = OutputState::new(false);
        let mut block = vec![0.0_f32; 6 * MIX_CHANNELS];

        // Play across a window boundary: the first window is retired.
        send(0, 0, 0.25);
        send(0, 4, 0.5);
        assert_eq!(
            fill_output_block(&mut consumer, &pos, &output, &mut block, MIX_CHANNELS),
            6
        );
        // A seek makes the playing window and a queued one stale.
        send(0, 8, 0.75);
        control.request_seek(100);
        send(1, 100, 1.0);
        output.mute();
        consumer.discard_stale();
        output.unmute_unless_halted();
        pos.store(100, Ordering::Release);
        assert_eq!(
            fill_output_block(
                &mut consumer,
                &pos,
                &output,
                &mut block[..4 * MIX_CHANNELS],
                MIX_CHANNELS
            ),
            4
        );

        let returned = std::iter::from_fn(|| recycle.try_recv().ok())
            .map(|buffer| buffer.as_ptr() as usize)
            .collect::<Vec<_>>();
        // Everything but the post-seek window it is still playing.
        assert_eq!(returned, sent[..3]);
        assert_eq!(control.callback_frees.load(Ordering::Acquire), 0);
    }

    #[test]
    fn a_full_recycle_channel_parks_windows_instead_of_freeing_them() {
        let control = Arc::new(AudioStreamControl::new(0));
        let (sender, receiver) = bounded(16);
        let (recycle_sender, recycle) = bounded(1);
        let mut consumer = AudioStreamConsumer::new(receiver, recycle_sender, Arc::clone(&control));
        for window in 0..4_u64 {
            sender
                .send(AudioStreamChunk {
                    generation: 0,
                    start_frame: window,
                    samples: vec![0.5; MIX_CHANNELS],
                })
                .unwrap();
        }
        // Skipping to the fourth window retires three: one fits the channel,
        // two wait parked until the producer drains it.
        assert!(consumer.ready_at(3));
        assert_eq!(consumer.parked.len(), 2);
        drain_recycled(&recycle);
        consumer.control.request_seek(10);
        consumer.discard_stale();
        assert_eq!(consumer.parked.len(), 2, "one parked window moved on");
        assert_eq!(control.callback_frees.load(Ordering::Acquire), 0);
    }

    // --- The clock follows what is audible (#69-3) ---

    #[test]
    fn the_clock_subtracts_the_output_delay_and_never_moves_back() {
        let pos = Arc::new(AtomicU64::new(0));
        let delay = Arc::new(AtomicU64::new(0));
        let clock =
            AudioClock::new(Arc::clone(&pos), 48_000, 30, None).delayed_by(Arc::clone(&delay));
        // One second handed to the device, 150 ms of it not audible yet
        // (for example a Bluetooth headset).
        pos.store(48_000, Ordering::Release);
        delay.store(7_200, Ordering::Release);
        assert_eq!(clock.frame(30), 25);
        // The device reports more latency: the playhead holds, never rewinds.
        delay.store(12_000, Ordering::Release);
        assert_eq!(clock.frame(30), 25);
        pos.store(48_000 + 1_600, Ordering::Release);
        assert_eq!(clock.frame(30), 25);
        // It advances again once the audible position passes it.
        pos.store(48_000 + 12_000, Ordering::Release);
        assert_eq!(clock.frame(30), 30);
        // A delay larger than the position saturates at the start.
        let early = AudioClock::new(Arc::new(AtomicU64::new(1_000)), 48_000, 30, None)
            .delayed_by(Arc::new(AtomicU64::new(9_000)));
        assert_eq!(early.frame(30), 0);
    }

    #[test]
    fn output_latency_converts_the_callback_timestamps_to_frames() {
        assert_eq!(latency_frames(None, 48_000), 0);
        assert_eq!(latency_frames(Some(Duration::from_millis(20)), 48_000), 960);
        assert_eq!(
            latency_frames(Some(Duration::from_millis(200)), 44_100),
            8_820
        );
    }

    #[test]
    fn a_muted_block_is_silent_and_does_not_advance_the_clock() {
        let control = Arc::new(AudioStreamControl::new(0));
        let (sender, receiver) = bounded(STREAM_WINDOW_CAPACITY);
        sender
            .send(AudioStreamChunk {
                generation: 0,
                start_frame: 0,
                samples: vec![0.5; 8 * MIX_CHANNELS],
            })
            .unwrap();
        let mut consumer = test_consumer(receiver, control);
        let pos = AtomicU64::new(0);
        let mut block = vec![1.0_f32; 4 * MIX_CHANNELS];
        let output = OutputState::new(true);
        assert_eq!(
            fill_output_block(&mut consumer, &pos, &output, &mut block, MIX_CHANNELS),
            0
        );
        assert!(block.iter().all(|sample| *sample == 0.0));
        assert_eq!(pos.load(Ordering::Acquire), 0);
        output.unmute_unless_halted();
        assert_eq!(
            fill_output_block(&mut consumer, &pos, &output, &mut block, MIX_CHANNELS),
            4
        );
        assert!(block.iter().all(|sample| *sample == 0.5));
    }

    // --- Device-thread waits are bounded (#214) ---

    #[test]
    fn a_hung_device_query_answers_unknown_after_the_timeout() {
        let (job_tx, job_rx) = mpsc::sync_channel::<DeviceJob>(1);
        let worker = std::thread::spawn(move || run_device_jobs(job_rx));
        let (release_tx, release_rx) = mpsc::channel::<()>();

        // A driver call that never returns while the resume waits on it.
        let hung = submit_device_query(&job_tx, Duration::from_millis(50), move || {
            let _ = release_rx.recv();
            Some("speakers".to_string())
        });
        assert_eq!(hung, None, "a timeout reads as \"device unchanged\"");
        // The queue behind it is full too; that also times out.
        assert_eq!(
            submit_device_query(&job_tx, Duration::from_millis(50), || Some(1_u32)),
            None
        );

        release_tx.send(()).unwrap();
        assert_eq!(
            submit_device_query(&job_tx, DEVICE_QUERY_TIMEOUT, || Some(44_100)),
            Some(44_100)
        );
        drop(job_tx);
        worker.join().expect("join audio device thread");
    }

    // --- Clip readers of a paused session (#209) ---

    #[test]
    fn a_paused_session_closes_its_clip_readers_once_its_queue_is_full() {
        let output = Arc::new(OutputState::new(true));
        let control = Arc::new(AudioStreamControl::with_output(0, Arc::clone(&output)));
        let (sender, receiver) = bounded(STREAM_WINDOW_CAPACITY);
        let (recycle_sender, recycle) = bounded(RECYCLE_CAPACITY);
        let mixed = Arc::new(AtomicU64::new(0));
        let mixer = fn_mixer({
            let mixed = Arc::clone(&mixed);
            move |_start: u64, len: usize, _cancel: &MediaCancelToken| {
                mixed.fetch_add(1, Ordering::AcqRel);
                Ok(vec![0.5; len * MIX_CHANNELS])
            }
        });
        let closes = Arc::clone(&mixer.closes);
        let producer_control = Arc::clone(&control);
        let producer = thread::spawn(move || {
            run_audio_producer(
                &producer_control,
                &sender,
                &recycle,
                ProducerWindows {
                    rate: 100,
                    next_frame: 0,
                    total_frames: 100_000,
                    window_frames: 100,
                },
                mixer,
                MediaCancelToken::new(),
            )
        });

        // Paused: the producer fills the queue, then releases its decoders.
        wait_for(Duration::from_secs(5), "readers closed", || {
            closes.load(Ordering::Acquire) > 0
        });
        let mixed_while_paused = mixed.load(Ordering::Acquire);
        assert_eq!(
            mixed_while_paused as usize,
            STREAM_WINDOW_CAPACITY + 1,
            "the queue plus the window waiting to be sent"
        );

        // Resumed: the session consumes and the producer reopens and mixes on.
        output.unmute_unless_halted();
        let mut consumer = AudioStreamConsumer::new(receiver, recycle_sender, Arc::clone(&control));
        let mut frame = 0;
        wait_for(Duration::from_secs(5), "mixing resumes", || {
            while consumer.ready_at(frame) {
                frame += 1;
            }
            mixed.load(Ordering::Acquire) > mixed_while_paused
        });
        let closes_while_playing = closes.load(Ordering::Acquire);
        for _ in 0..3 {
            while consumer.ready_at(frame) {
                frame += 1;
            }
            thread::sleep(STREAM_SEND_POLL * 4);
        }
        assert_eq!(
            closes.load(Ordering::Acquire),
            closes_while_playing,
            "a playing session keeps its readers"
        );

        control.stop();
        drop(consumer);
        producer.join().expect("producer exits on stop");
    }

    #[test]
    fn a_seek_or_stop_interrupts_the_readers_the_first_window_left_open() {
        for stop in [false, true] {
            // The prepare token reaches the prefilled window through a child;
            // the readers that window left open read the next window with it.
            let prepare = MediaCancelToken::new();
            let first_window = prepare.child();
            let control = Arc::new(AudioStreamControl::new(0));
            let (sender, _receiver) = bounded(STREAM_WINDOW_CAPACITY);
            let (_recycle_sender, recycle) = bounded(RECYCLE_CAPACITY);
            let (entered_tx, entered_rx) = mpsc::channel();
            let (outcome_tx, outcome_rx) = mpsc::channel();
            let expected = first_window.clone();
            let mixer = fn_mixer(move |start: u64, _len: usize, cancel: &MediaCancelToken| {
                if start == 100 {
                    // The second window: continue the first window's reader.
                    let _ = outcome_tx.send(cancel.same_instance(&expected));
                    let _ = entered_tx.send(());
                    let deadline = Instant::now() + Duration::from_secs(5);
                    while !cancel.is_cancelled() && Instant::now() < deadline {
                        thread::sleep(Duration::from_millis(1));
                    }
                    return Err(MediaError::Cancelled);
                }
                Ok(vec![0.0; 100 * MIX_CHANNELS])
            });
            let producer_control = Arc::clone(&control);
            let producer_window = first_window.clone();
            let producer = thread::spawn(move || {
                run_audio_producer(
                    &producer_control,
                    &sender,
                    &recycle,
                    ProducerWindows {
                        rate: 100,
                        next_frame: 100,
                        total_frames: 400,
                        window_frames: 100,
                    },
                    mixer,
                    producer_window,
                )
            });
            assert!(
                outcome_rx.recv_timeout(Duration::from_secs(5)).unwrap(),
                "the second window reads with the first window's token"
            );
            entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            if stop {
                control.stop();
            } else {
                control.request_seek(300);
            }
            assert!(
                first_window.is_cancelled(),
                "stop={stop}: the read is interrupted"
            );
            assert!(!prepare.is_cancelled(), "the session itself goes on");
            control.stop();
            producer.join().expect("producer exits");
        }
    }
}
