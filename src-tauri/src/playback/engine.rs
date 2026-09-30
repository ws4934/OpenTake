//! The playback render loop + its dedicated thread (#53).
//!
//! A single thread owns a wgpu device and drives the whole "read clock → build
//! the frame plan → pull/decode each clip's frame → composite → hand the frame
//! to a sink → broadcast the playhead" cycle. Keeping it on one thread is a hard
//! requirement: the compositor's textures are `Rc` (not `Send`), and wgpu's
//! device/queue must be touched from one thread. The thread creates its **own**
//! [`RenderDevice`] and never touches the preview's `RenderState`, so playback and
//! the paused-frame `composite_frame` path never contend.
//!
//! The clock, frame sink, and error sink are traits so the loop logic is
//! decoupled from cpal / the JPEG transport / Tauri: tests supply in-memory
//! implementations (and a stub renderer), production the cpal master clock, the
//! off-thread JPEG sink, and the `playback_error` emitter.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use opentake_domain::Timeline;
use opentake_media::MediaCancelToken;
use opentake_render::{
    try_build_render_plan, Compositor, DecodedFrame, FramePlan, RenderDevice, RenderPlan,
    RenderSize, TextureSource,
};

use super::project::{ManifestMetrics, MediaInfo, TextInfo};
use super::resolver::{PlaybackResolverState, StreamingResolver};

const REAPER_CAPACITY: usize = 2;

struct ReapJob(Vec<JoinHandle<()>>);

struct ReaperInner {
    sender: mpsc::SyncSender<ReapJob>,
    outstanding: Arc<AtomicUsize>,
}

/// One persistent join worker with at most two outstanding teardown jobs.
#[derive(Clone)]
pub struct BoundedReaper {
    inner: Arc<ReaperInner>,
}

pub struct ReapPermit {
    reaper: BoundedReaper,
    active: bool,
}

impl BoundedReaper {
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::sync_channel::<ReapJob>(REAPER_CAPACITY);
        let inner = Arc::new(ReaperInner {
            sender,
            outstanding: Arc::new(AtomicUsize::new(0)),
        });
        let worker_outstanding = Arc::clone(&inner.outstanding);
        let _ = thread::Builder::new()
            .name("opentake-playback-reaper".to_string())
            .spawn(move || {
                while let Ok(ReapJob(handles)) = receiver.recv() {
                    for handle in handles {
                        let _ = handle.join();
                    }
                    worker_outstanding.fetch_sub(1, Ordering::AcqRel);
                }
            });
        Self { inner }
    }

    pub fn try_reserve(&self) -> Result<ReapPermit, String> {
        self.inner
            .outstanding
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |outstanding| {
                (outstanding < REAPER_CAPACITY).then_some(outstanding + 1)
            })
            .map_err(|_| "playback_teardown_busy".to_string())?;
        Ok(ReapPermit {
            reaper: self.clone(),
            active: true,
        })
    }

    #[cfg(test)]
    pub(crate) fn wait_until_idle(&self, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        while self.inner.outstanding.load(Ordering::Acquire) != 0 && Instant::now() < deadline {
            thread::yield_now();
        }
        assert_eq!(self.inner.outstanding.load(Ordering::Acquire), 0);
    }

    #[cfg(test)]
    pub(crate) fn outstanding_count(&self) -> usize {
        self.inner.outstanding.load(Ordering::Acquire)
    }
}

impl Default for BoundedReaper {
    fn default() -> Self {
        Self::new()
    }
}

impl ReapPermit {
    pub fn enqueue(mut self, handles: Vec<JoinHandle<()>>) -> Result<(), String> {
        match self.reaper.inner.sender.try_send(ReapJob(handles)) {
            Ok(()) => {
                self.active = false;
                Ok(())
            }
            Err(error) => Err(format!("playback reaper enqueue failed: {error}")),
        }
    }
}

impl Drop for ReapPermit {
    fn drop(&mut self) {
        if self.active {
            self.reaper.inner.outstanding.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

/// Drives the playback playhead. The audio master clock (cpal) implements this in
/// PR2; PR1 uses [`InstantClock`] (wall-clock) and the no-audio fallback.
pub trait PlaybackClock: Send + Sync {
    /// The target timeline frame *now*, given the project fps.
    fn frame(&self, fps: i32) -> i32;
    /// Reset the clock so `frame()` resumes counting from `frame`.
    fn seek(&self, frame: i32);
    /// A non-fatal error of the clock's audio source (for example a window that
    /// failed to decode and plays as silence), reported at most once until the
    /// next seek. The render thread forwards it as a `playback_error`.
    fn take_error(&self) -> Option<String> {
        None
    }
    /// The render loop paused itself after a fatal failure: silence the
    /// clock's audio output until the render thread resumes and that resume
    /// commits. A no-op for clocks without sound.
    fn halt(&self) {}
    /// The render thread processed a resume: a [`Self::halt`] from before it
    /// no longer keeps the output muted, one after it still does. Called on
    /// the render thread, so the order against its own failures is exact.
    fn resumed(&self) {}
}

/// Receives each composited frame. Production: [`super::transport::MjpegSink`],
/// which encodes and publishes on its own thread, so both calls must return
/// without waiting for encoding.
pub trait FrameSink: Send + Sync {
    /// Hand off the composited image for timeline `frame`.
    fn push_frame(&self, frame: i32, image: DecodedFrame);
    /// The final timeline frame failed to render: announce the terminal tick
    /// without new pixels so the front end can end its transport.
    fn push_terminal(&self, frame: i32);
    /// The render thread moved the playhead (a seek): drop frames handed off
    /// before this call, keeping publication open for the ones that follow.
    /// Test doubles that do not queue frames can keep the no-op.
    fn invalidate(&self) {}
}

/// Why playback stopped producing frames or sound (`playback_error.code`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub enum PlaybackFailureCode {
    /// A video layer could not be decoded or its decoder stopped early.
    VideoDecode,
    /// An image, text, Lottie or LUT layer could not be materialized.
    Materialization,
    /// The compositor or GPU readback failed.
    Render,
    /// A timeline audio window could not be decoded; it plays as silence.
    AudioDecode,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaybackFailure {
    pub frame: i32,
    pub code: PlaybackFailureCode,
    /// Names media by asset id; absolute paths are redacted.
    pub message: String,
    /// `true` when the engine paused itself and the transport must stop.
    pub fatal: bool,
}

/// Reports playback failures after startup. Production emits `playback_error`.
pub trait PlaybackErrorSink: Send + Sync {
    fn report(&self, failure: PlaybackFailure);
}

/// A classified render failure of one frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderFailure {
    pub code: PlaybackFailureCode,
    pub message: String,
}

impl RenderFailure {
    fn new(code: PlaybackFailureCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

/// What the render loop drives: the GPU [`RenderLoop`] in production, a stub in
/// the loop's unit tests.
trait FrameRenderer {
    fn total_frames(&self) -> i32;
    fn fps(&self) -> i32;
    fn render(&mut self, target: i32) -> Result<DecodedFrame, RenderFailure>;
    /// Restart decoding at the next rendered position (seek / moved resume).
    fn seek(&mut self);
}

/// Control messages to the render thread.
pub enum PlaybackCmd {
    /// Freeze at `frame` while retaining GPU and decoder state.
    Pause(i32, mpsc::Sender<()>),
    /// Resume a retained session from `frame`.
    Resume(i32, mpsc::Sender<()>),
    /// Wake the render thread to consume the newest coalesced seek.
    Seek,
    /// Stop the loop and tear down (streams stop cooperatively).
    Stop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SeekRequest {
    frame: i32,
    generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SeekSubmission {
    should_wake: bool,
}

#[derive(Default)]
struct SeekMailboxState {
    generation: u64,
    pending: Option<SeekRequest>,
    wake_queued: bool,
}

/// One-slot newest-wins mailbox. Twenty rapid seeks overwrite one pending
/// request and enqueue at most one control wake; the generation also lets the
/// render thread discard pixels completed after a newer seek arrived.
#[derive(Default)]
struct SeekMailbox(Mutex<SeekMailboxState>);

impl SeekMailbox {
    fn submit(&self, frame: i32) -> SeekSubmission {
        let mut state = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.generation = state.generation.wrapping_add(1);
        state.pending = Some(SeekRequest {
            frame,
            generation: state.generation,
        });
        let should_wake = !state.wake_queued;
        state.wake_queued = true;
        SeekSubmission { should_wake }
    }

    fn take(&self) -> Option<SeekRequest> {
        let mut state = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let request = state.pending.take();
        state.wake_queued = false;
        request
    }

    fn generation(&self) -> u64 {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .generation
    }

    /// A queued seek has not updated the clock yet. Sample it only after
    /// that seek has been taken and applied by the control loop.
    fn render_generation(&self) -> Option<u64> {
        let state = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.pending.is_none().then_some(state.generation)
    }

    fn is_current(&self, generation: u64) -> bool {
        self.generation() == generation
    }
}

/// Integer target frame from a base frame plus elapsed time. Truncates (matching
/// the `secondsToFrame = Int(secs*fps)` port rule), never rounds. `fps <= 0`
/// falls back to 30 (the project default) to stay defined.
fn frame_at_elapsed(base_frame: i32, elapsed_secs: f64, fps: i32) -> i32 {
    let fps = if fps > 0 { fps } else { 30 };
    base_frame + (elapsed_secs.max(0.0) * fps as f64) as i32
}

fn next_render_deadline(previous: Instant, period: Duration, now: Instant) -> Instant {
    // Keep the cadence when a timer wakes late. A render that misses the next
    // deadline starts its successor immediately so the media clock can catch up.
    (previous + period).max(now)
}

/// Clamp the clock's frame to the drawable range and decide whether playback has
/// reached the end. Returns `(target, done)`: `target` is the frame to render,
/// `done` is true once the clock hits the last frame (→ auto-stop). Pure so the
/// loop's termination boundary is unit-tested.
pub(super) fn loop_step(clock_frame: i32, total: i32) -> (i32, bool) {
    let last = total.max(1) - 1;
    (clock_frame.clamp(0, last), clock_frame >= last)
}

/// Wall-clock playback clock: the PR1 driver and the no-audio fallback. Advances
/// the playhead by real elapsed time from the last `seek` (or construction).
pub struct InstantClock {
    /// `(origin, base_frame)`: `frame()` = `base_frame + elapsed_since(origin)`.
    inner: Mutex<(Instant, i32)>,
}

impl InstantClock {
    pub fn new(start_frame: i32) -> Self {
        InstantClock {
            inner: Mutex::new((Instant::now(), start_frame)),
        }
    }
}

impl PlaybackClock for InstantClock {
    fn frame(&self, fps: i32) -> i32 {
        // Recover from a poisoned lock rather than panicking on the render thread.
        let guard = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        let (origin, base) = *guard;
        frame_at_elapsed(base, origin.elapsed().as_secs_f64(), fps)
    }

    fn seek(&self, frame: i32) {
        let mut guard = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        *guard = (Instant::now(), frame);
    }
}

/// The GPU-backed render loop: owns the device, the (frame-independent)
/// [`RenderPlan`], and the streaming resolver state. One instance lives for a
/// whole playback session on the render thread. Exposed (with `render_frame`) so
/// a GPU+ffmpeg integration test can drive it deterministically without the
/// thread/clock.
pub struct RenderLoop {
    device: opentake_render::wgpu::Device,
    queue: opentake_render::wgpu::Queue,
    compositor: Compositor,
    timeline: Timeline,
    plan: RenderPlan,
    render_size: RenderSize,
    state: PlaybackResolverState,
    last_video_sources: HashMap<String, i64>,
    reversed_clips: HashSet<String>,
}

fn active_video_sources(frame_plan: &FramePlan) -> HashMap<String, i64> {
    frame_plan
        .draws
        .iter()
        .filter_map(|draw| match draw.source {
            TextureSource::Decoded { .. } => Some((draw.clip_id.to_string(), draw.source_frame)),
            _ => None,
        })
        .collect()
}

/// Clips whose forward stream must restart because their source frame moved
/// backwards since the previous frame. Reversed clips descend by design and
/// are served by reverse windows, so they never count as a rewind.
fn rewound_video_sources<'a>(
    previous: &HashMap<String, i64>,
    current: &'a HashMap<String, i64>,
    reversed: &HashSet<String>,
) -> Vec<&'a str> {
    current
        .iter()
        .filter(|(clip_id, source_frame)| {
            !reversed.contains(*clip_id)
                && previous
                    .get(*clip_id)
                    .is_some_and(|prev| *source_frame < prev)
        })
        .map(|(clip_id, _)| clip_id.as_str())
        .collect()
}

fn reversed_video_clips(plan: &RenderPlan) -> HashSet<String> {
    plan.clip_plans
        .iter()
        .filter(|clip| clip.reversed && matches!(clip.source, TextureSource::Decoded { .. }))
        .map(|clip| clip.clip_id.clone())
        .collect()
}

impl RenderLoop {
    /// Build the render loop: acquire a GPU device, build the render plan from the
    /// timeline (same `build_render_plan` the preview/export use), and prime the
    /// resolver state. Returns `Err` (never panics) when no GPU is available.
    pub fn new(
        timeline: Timeline,
        media: HashMap<String, MediaInfo>,
        text: HashMap<String, TextInfo>,
        sizes: HashMap<String, (u32, u32)>,
        render_size: RenderSize,
    ) -> Result<Self, String> {
        Self::new_with_cancel(
            timeline,
            media,
            text,
            sizes,
            render_size,
            MediaCancelToken::new(),
            None,
        )
    }

    fn new_with_cancel(
        timeline: Timeline,
        media: HashMap<String, MediaInfo>,
        text: HashMap<String, TextInfo>,
        sizes: HashMap<String, (u32, u32)>,
        render_size: RenderSize,
        cancel: MediaCancelToken,
        project_dir: Option<PathBuf>,
    ) -> Result<Self, String> {
        let dev = RenderDevice::try_new().map_err(|e| format!("no GPU device: {e}"))?;
        let compositor = Compositor::new(&dev.device);
        let metrics = ManifestMetrics { sizes };
        let plan = try_build_render_plan(&timeline, render_size, &metrics)
            .map_err(|error| format!("invalid timeline graph: {error}"))?;
        let project_root = project_dir
            .map(opentake_project::ProjectRoot::open)
            .transpose()
            .map_err(|error| format!("open project LUT storage: {error}"))?;
        let mut state = PlaybackResolverState::new_with_project_root(
            media,
            text,
            plan.fps,
            (render_size.width, render_size.height),
            cancel,
            project_root,
        );
        let reversed_clips = reversed_video_clips(&plan);
        state.set_reversed_clips(reversed_clips.clone());
        Ok(RenderLoop {
            device: dev.device,
            queue: dev.queue,
            compositor,
            timeline,
            plan,
            render_size,
            state,
            last_video_sources: HashMap::new(),
            reversed_clips,
        })
    }

    pub fn total_frames(&self) -> i32 {
        self.plan.total_frames
    }

    pub fn fps(&self) -> i32 {
        self.plan.fps
    }

    /// Composite a single frame at `target`: reconcile the streams to this frame,
    /// then run the same compositor pixel path as the preview/export.
    pub fn render_frame(&mut self, target: i32) -> Result<DecodedFrame, String> {
        self.render_classified(target)
            .map_err(|failure| failure.message)
    }

    /// [`Self::render_frame`] with the failure classified for `playback_error`
    /// and absolute media paths redacted from the message.
    pub fn render_classified(&mut self, target: i32) -> Result<DecodedFrame, RenderFailure> {
        let frame_plan = self.plan.frame(&self.timeline, target);
        let current_video_sources = active_video_sources(&frame_plan);
        self.state.reset_streams(rewound_video_sources(
            &self.last_video_sources,
            &current_video_sources,
            &self.reversed_clips,
        ));
        let mut resolver = StreamingResolver::new(&self.device, &self.queue, &mut self.state);
        if let Err(error) = resolver.sync_active(&frame_plan) {
            drop(resolver);
            return Err(RenderFailure::new(
                PlaybackFailureCode::VideoDecode,
                self.state.redact_media_paths(&error),
            ));
        }
        let composite = self.compositor.render_to_rgba(
            &self.device,
            &self.queue,
            self.render_size,
            &frame_plan,
            &mut resolver,
        );
        drop(resolver);
        if let Some(error) = self.state.take_materialization_error() {
            return Err(RenderFailure::new(
                PlaybackFailureCode::Materialization,
                self.state.redact_media_paths(&format!(
                    "playback materialization failed at frame {target}: {error}"
                )),
            ));
        }
        self.last_video_sources = current_video_sources;
        composite.map_err(|error| {
            RenderFailure::new(
                PlaybackFailureCode::Render,
                format!("composite render failed at frame {target}: {error}"),
            )
        })
    }

    /// Restart all decode streams (used on seek): the next `render_frame` re-spawns
    /// each visible clip's stream at its new target source frame, and clips whose
    /// decoder failed are retried.
    pub fn seek(&mut self) {
        self.state.clear_streams();
        self.last_video_sources.clear();
    }
}

impl FrameRenderer for RenderLoop {
    fn total_frames(&self) -> i32 {
        self.plan.total_frames
    }

    fn fps(&self) -> i32 {
        self.plan.fps
    }

    fn render(&mut self, target: i32) -> Result<DecodedFrame, RenderFailure> {
        self.render_classified(target)
    }

    fn seek(&mut self) {
        RenderLoop::seek(self);
    }
}

/// An exact resume can continue the existing decoder streams. An explicit
/// seek has already cleared them; a resume elsewhere invalidates old data.
fn resume_decode_streams(paused_frame: Option<i32>, requested_frame: i32, mut reset: impl FnMut()) {
    if paused_frame != Some(requested_frame) {
        reset();
    }
}

/// A cloneable control endpoint of one render thread. Every call only enqueues
/// work except [`Self::resume`], which waits for the render thread to adopt the
/// new position; callers must never hold a lock the main thread needs while
/// waiting on it (#42).
#[derive(Clone)]
pub struct EngineControl {
    control_tx: mpsc::Sender<PlaybackCmd>,
    seek_mailbox: Arc<SeekMailbox>,
    pause_requested: Arc<AtomicBool>,
    pressure: PressureSlot,
}

/// Playback/export pressure held while the engine plays, so background
/// inference yields (#43). The command layer fills it before a start or
/// resume handshake, tagged with that resume's token; it is emptied by a
/// pause (before any fallible step), by a stop, and by the render thread
/// itself when it halts at the end of the timeline or on a render failure and
/// when it exits for any reason. The render thread does not release on a
/// queued `Pause`: the control side already did, and a resume issued after
/// that pause may have installed a newer guard by the time it is processed.
type PressureSlot = Arc<Mutex<Option<(u64, opentake_media::ExportPauseGuard)>>>;

fn release_pressure_slot(slot: &PressureSlot) {
    slot.lock().unwrap_or_else(|p| p.into_inner()).take();
}

/// Empties the pressure slot when the render thread returns or unwinds.
struct ReleasePressureOnExit(PressureSlot);

impl Drop for ReleasePressureOnExit {
    fn drop(&mut self) {
        release_pressure_slot(&self.0);
    }
}

impl EngineControl {
    fn new(control_tx: mpsc::Sender<PlaybackCmd>) -> Self {
        Self {
            control_tx,
            seek_mailbox: Arc::new(SeekMailbox::default()),
            pause_requested: Arc::new(AtomicBool::new(false)),
            pressure: PressureSlot::default(),
        }
    }

    /// Hold playback pressure for the resume identified by `token` (resume
    /// tokens only grow). At most one guard is held per engine: a newer
    /// resume replaces an older one's guard, and an older resume never
    /// replaces a newer one's.
    pub(crate) fn hold_pressure(&self, token: u64, guard: opentake_media::ExportPauseGuard) {
        let mut slot = self.pressure.lock().unwrap_or_else(|p| p.into_inner());
        if slot.as_ref().is_none_or(|(held, _)| *held <= token) {
            *slot = Some((token, guard));
        }
    }

    /// Release the guard only if the resume identified by `token` still owns
    /// it (a superseded or failed handshake must not drop a newer resume's).
    pub(crate) fn release_pressure_for(&self, token: u64) {
        let mut slot = self.pressure.lock().unwrap_or_else(|p| p.into_inner());
        if slot.as_ref().is_some_and(|(held, _)| *held == token) {
            *slot = None;
        }
    }

    /// Release playback pressure unconditionally (pause, stop).
    pub(crate) fn release_pressure(&self) {
        release_pressure_slot(&self.pressure);
    }

    /// Seek the running engine to `frame`.
    pub fn seek(&self, frame: i32) {
        if self.seek_mailbox.submit(frame).should_wake {
            let _ = self.control_tx.send(PlaybackCmd::Seek);
        }
    }

    pub fn pause(&self, frame: i32) -> Result<(), String> {
        self.release_pressure();
        self.pause_requested.store(true, Ordering::Release);
        let (reply, _acknowledgement) = mpsc::channel();
        if self
            .control_tx
            .send(PlaybackCmd::Pause(frame, reply))
            .is_err()
        {
            self.pause_requested.store(false, Ordering::Release);
            return Err("playback render thread exited before control".to_string());
        }
        Ok(())
    }

    pub fn resume(&self, frame: i32) -> Result<(), String> {
        let (reply_tx, reply_rx) = mpsc::channel();
        self.control_tx
            .send(PlaybackCmd::Resume(frame, reply_tx))
            .map_err(|_| "playback render thread exited before control".to_string())?;
        reply_rx
            .recv()
            .map_err(|_| "playback render thread exited during control".to_string())
    }

    fn request_stop(&self) {
        let _ = self.control_tx.send(PlaybackCmd::Stop);
    }
}

/// Ordered transport commands observed by a render-thread test double.
#[cfg(test)]
pub(crate) type CommandLog = Arc<Mutex<Vec<String>>>;

/// Owns the playback render thread and a control channel to it. Dropping (or
/// `stop`) requests a cooperative shutdown.
pub struct PlaybackEngine {
    control: EngineControl,
    handle: Option<JoinHandle<()>>,
    cancel: MediaCancelToken,
}

impl PlaybackEngine {
    /// Spawn the render thread. The GPU device is created **inside** the thread
    /// (so nothing non-`Send` crosses the boundary); on GPU-acquire failure the
    /// thread logs and exits, leaving this handle inert.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        timeline: Timeline,
        media: HashMap<String, MediaInfo>,
        text: HashMap<String, TextInfo>,
        sizes: HashMap<String, (u32, u32)>,
        render_size: RenderSize,
        clock: Arc<dyn PlaybackClock>,
        sink: Arc<dyn FrameSink>,
        errors: Arc<dyn PlaybackErrorSink>,
    ) -> Result<Self, String> {
        Self::spawn_internal(
            timeline,
            media,
            text,
            sizes,
            render_size,
            clock,
            sink,
            errors,
            None,
            None,
            None,
            MediaCancelToken::new(),
        )
    }

    /// Spawn the GPU thread, render and buffer its first complete frame, then
    /// return a paused handle. The caller installs the authoritative session
    /// before `resume` makes that buffered frame observable. Waiting for the
    /// render-thread handshake is synchronous, so async command callers must run
    /// this constructor on a blocking worker.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn_ready(
        timeline: Timeline,
        media: HashMap<String, MediaInfo>,
        text: HashMap<String, TextInfo>,
        sizes: HashMap<String, (u32, u32)>,
        render_size: RenderSize,
        clock: Arc<dyn PlaybackClock>,
        sink: Arc<dyn FrameSink>,
        errors: Arc<dyn PlaybackErrorSink>,
        start_frame: i32,
    ) -> Result<Self, String> {
        Self::spawn_ready_cancellable(
            timeline,
            media,
            text,
            sizes,
            render_size,
            clock,
            sink,
            errors,
            start_frame,
            MediaCancelToken::new(),
        )
    }

    /// Prepare the first exact frame with a caller-owned session token. The
    /// playback coordinator keeps this token reachable until installation, so
    /// project/timeline invalidation can cancel a blocked initial bootstrap
    /// before a [`PlaybackEngine`] handle exists.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn_ready_cancellable(
        timeline: Timeline,
        media: HashMap<String, MediaInfo>,
        text: HashMap<String, TextInfo>,
        sizes: HashMap<String, (u32, u32)>,
        render_size: RenderSize,
        clock: Arc<dyn PlaybackClock>,
        sink: Arc<dyn FrameSink>,
        errors: Arc<dyn PlaybackErrorSink>,
        start_frame: i32,
        cancel: MediaCancelToken,
    ) -> Result<Self, String> {
        Self::spawn_ready_cancellable_with_project(
            timeline,
            media,
            text,
            sizes,
            render_size,
            clock,
            sink,
            errors,
            start_frame,
            cancel,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn spawn_ready_cancellable_with_project(
        timeline: Timeline,
        media: HashMap<String, MediaInfo>,
        text: HashMap<String, TextInfo>,
        sizes: HashMap<String, (u32, u32)>,
        render_size: RenderSize,
        clock: Arc<dyn PlaybackClock>,
        sink: Arc<dyn FrameSink>,
        errors: Arc<dyn PlaybackErrorSink>,
        start_frame: i32,
        cancel: MediaCancelToken,
        project_dir: Option<PathBuf>,
    ) -> Result<Self, String> {
        let (ready_tx, ready_rx) = mpsc::channel();
        let engine = Self::spawn_internal(
            timeline,
            media,
            text,
            sizes,
            render_size,
            clock,
            sink,
            errors,
            project_dir,
            Some(start_frame.max(0)),
            Some(ready_tx),
            cancel,
        )?;
        engine.await_ready(ready_rx)
    }

    fn await_ready(self, ready_rx: mpsc::Receiver<Result<(), String>>) -> Result<Self, String> {
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(self),
            Ok(Err(error)) => {
                self.stop();
                Err(error)
            }
            Err(_) => {
                self.stop();
                Err("playback thread exited before the first frame".to_string())
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn spawn_internal(
        timeline: Timeline,
        media: HashMap<String, MediaInfo>,
        text: HashMap<String, TextInfo>,
        sizes: HashMap<String, (u32, u32)>,
        render_size: RenderSize,
        clock: Arc<dyn PlaybackClock>,
        sink: Arc<dyn FrameSink>,
        errors: Arc<dyn PlaybackErrorSink>,
        project_dir: Option<PathBuf>,
        initial_frame: Option<i32>,
        startup: Option<mpsc::Sender<Result<(), String>>>,
        cancel: MediaCancelToken,
    ) -> Result<Self, String> {
        let render_cancel = cancel.clone();
        Self::spawn_with(
            move || {
                RenderLoop::new_with_cancel(
                    timeline,
                    media,
                    text,
                    sizes,
                    render_size,
                    render_cancel,
                    project_dir,
                )
            },
            RenderOutputs {
                clock,
                sink,
                errors,
            },
            initial_frame,
            startup,
            cancel,
        )
    }

    /// Spawn the render thread around a renderer built on that thread (the GPU
    /// device and its `Rc` textures never cross threads).
    fn spawn_with<R, B>(
        build: B,
        outputs: RenderOutputs,
        initial_frame: Option<i32>,
        mut startup: Option<mpsc::Sender<Result<(), String>>>,
        cancel: MediaCancelToken,
    ) -> Result<Self, String>
    where
        R: FrameRenderer,
        B: FnOnce() -> Result<R, String> + Send + 'static,
    {
        let (tx, rx) = mpsc::channel();
        let control = EngineControl::new(tx);
        let loop_control = LoopControl {
            rx,
            seek_mailbox: Arc::clone(&control.seek_mailbox),
            pause_requested: Arc::clone(&control.pause_requested),
            pressure: Arc::clone(&control.pressure),
            cancel: cancel.clone(),
        };
        let exit_pressure = Arc::clone(&control.pressure);
        let handle = thread::Builder::new()
            .name("opentake-playback-render".to_string())
            .spawn(move || {
                let _release_pressure = ReleasePressureOnExit(exit_pressure);
                let renderer = match build() {
                    Ok(renderer) => renderer,
                    Err(error) => {
                        eprintln!("[playback] {error}");
                        if let Some(tx) = startup.take() {
                            let _ = tx.send(Err(error));
                        }
                        return;
                    }
                };
                run_render_loop(renderer, outputs, loop_control, initial_frame, startup);
            })
            .map_err(|e| format!("spawn playback thread: {e}"))?;
        Ok(PlaybackEngine {
            control,
            handle: Some(handle),
            cancel,
        })
    }

    /// A cloneable control endpoint usable without owning the engine.
    pub fn control(&self) -> EngineControl {
        self.control.clone()
    }

    /// Seek the running engine to `frame`.
    pub fn seek(&self, frame: i32) {
        self.control.seek(frame);
    }

    pub fn pause(&self, frame: i32) -> Result<(), String> {
        self.control.pause(frame)
    }

    pub fn resume(&self, frame: i32) -> Result<(), String> {
        self.control.resume(frame)
    }

    /// Stop the engine and join the render thread.
    pub fn stop(mut self) {
        self.cancel.cancel();
        self.control.request_stop();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }

    pub fn request_stop(mut self) -> Option<JoinHandle<()>> {
        self.cancel.cancel();
        self.control.request_stop();
        self.handle.take()
    }

    #[cfg(test)]
    fn from_test_thread(
        control_tx: mpsc::Sender<PlaybackCmd>,
        handle: JoinHandle<()>,
    ) -> (Self, Arc<SeekMailbox>) {
        let control = EngineControl::new(control_tx);
        let mailbox = Arc::clone(&control.seek_mailbox);
        (
            Self {
                control,
                handle: Some(handle),
                cancel: MediaCancelToken::new(),
            },
            mailbox,
        )
    }

    #[cfg(test)]
    pub(crate) fn test_stub() -> (Self, mpsc::Receiver<()>) {
        let (control_tx, control_rx) = mpsc::channel();
        let (stopped_tx, stopped_rx) = mpsc::channel();
        let handle = thread::spawn(move || {
            while let Ok(command) = control_rx.recv() {
                match command {
                    PlaybackCmd::Stop => {
                        let _ = stopped_tx.send(());
                        break;
                    }
                    PlaybackCmd::Pause(_, reply) | PlaybackCmd::Resume(_, reply) => {
                        let _ = reply.send(());
                    }
                    PlaybackCmd::Seek => {}
                }
            }
        });
        (Self::from_test_thread(control_tx, handle).0, stopped_rx)
    }

    #[cfg(test)]
    pub(crate) fn test_resume_observer(
        audio_output: Arc<super::audio::OutputState>,
    ) -> (Self, mpsc::Receiver<bool>, mpsc::Receiver<()>) {
        let (control_tx, control_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        let (stopped_tx, stopped_rx) = mpsc::channel();
        let handle = thread::spawn(move || {
            while let Ok(command) = control_rx.recv() {
                match command {
                    PlaybackCmd::Stop => {
                        let _ = stopped_tx.send(());
                        break;
                    }
                    PlaybackCmd::Pause(_, reply) => {
                        let _ = reply.send(());
                    }
                    PlaybackCmd::Resume(_, reply) => {
                        let paused = audio_output.is_muted();
                        let _ = resume_tx.send(paused);
                        let _ = reply.send(());
                    }
                    PlaybackCmd::Seek => {}
                }
            }
        });
        (
            Self::from_test_thread(control_tx, handle).0,
            resume_rx,
            stopped_rx,
        )
    }

    #[cfg(test)]
    pub(crate) fn test_blocking_pause() -> (Self, mpsc::Receiver<i32>, mpsc::Sender<()>) {
        let (control_tx, control_rx) = mpsc::channel();
        let (pause_tx, pause_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let handle = thread::spawn(move || {
            while let Ok(command) = control_rx.recv() {
                match command {
                    PlaybackCmd::Pause(frame, reply) => {
                        let _ = pause_tx.send(frame);
                        let _ = release_rx.recv();
                        let _ = reply.send(());
                    }
                    PlaybackCmd::Resume(_, reply) => {
                        let _ = reply.send(());
                    }
                    PlaybackCmd::Seek => {}
                    PlaybackCmd::Stop => break,
                }
            }
        });
        (
            Self::from_test_thread(control_tx, handle).0,
            pause_rx,
            release_tx,
        )
    }

    /// A render-thread stand-in whose `Resume` barrier blocks until released
    /// (an in-flight 4K frame). It records the ordered transport commands it
    /// observes: `pause:N`, `resume:N`, and `seek:N` for each consumed seek.
    #[cfg(test)]
    pub(crate) fn test_blocking_resume() -> (Self, mpsc::Receiver<i32>, mpsc::Sender<()>, CommandLog)
    {
        let (control_tx, control_rx) = mpsc::channel();
        let (resume_seen_tx, resume_seen_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let log = Arc::new(Mutex::new(Vec::new()));
        let thread_log = Arc::clone(&log);
        let (mailbox_tx, mailbox_rx) = mpsc::channel::<Arc<SeekMailbox>>();
        let handle = thread::spawn(move || {
            let Ok(mailbox) = mailbox_rx.recv() else {
                return;
            };
            let record = |entry: String| {
                thread_log
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .push(entry);
            };
            while let Ok(command) = control_rx.recv() {
                match command {
                    PlaybackCmd::Pause(frame, reply) => {
                        record(format!("pause:{frame}"));
                        let _ = reply.send(());
                    }
                    PlaybackCmd::Resume(frame, reply) => {
                        let _ = resume_seen_tx.send(frame);
                        let _ = release_rx.recv();
                        record(format!("resume:{frame}"));
                        let _ = reply.send(());
                    }
                    PlaybackCmd::Seek => {
                        if let Some(request) = mailbox.take() {
                            record(format!("seek:{}", request.frame));
                        }
                    }
                    PlaybackCmd::Stop => break,
                }
            }
        });
        let (engine, mailbox) = Self::from_test_thread(control_tx, handle);
        mailbox_tx.send(mailbox).expect("hand mailbox to stub");
        (engine, resume_seen_rx, release_tx, log)
    }
}

impl Drop for PlaybackEngine {
    fn drop(&mut self) {
        // Best-effort cooperative stop if the caller didn't `stop()` explicitly.
        self.cancel.cancel();
        self.control.request_stop();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Where the render loop sends its time base reads, frames and failures.
struct RenderOutputs {
    clock: Arc<dyn PlaybackClock>,
    sink: Arc<dyn FrameSink>,
    errors: Arc<dyn PlaybackErrorSink>,
}

/// The render thread's end of [`EngineControl`].
struct LoopControl {
    rx: mpsc::Receiver<PlaybackCmd>,
    seek_mailbox: Arc<SeekMailbox>,
    pause_requested: Arc<AtomicBool>,
    pressure: PressureSlot,
    /// The session's media cancellation. Once it fires the session is being
    /// torn down: a render it interrupted is a stop, never a failure.
    cancel: MediaCancelToken,
}

/// The render thread body: render frames paced at the project fps until the
/// clock reaches the end or a `Stop` arrives. A render failure after startup
/// pauses the loop and is reported once: as a terminal tick on the final frame
/// (the front end ends its transport), otherwise as a fatal `playback_error`.
fn run_render_loop<R: FrameRenderer>(
    mut renderer: R,
    outputs: RenderOutputs,
    control: LoopControl,
    initial_frame: Option<i32>,
    mut startup: Option<mpsc::Sender<Result<(), String>>>,
) {
    let RenderOutputs {
        clock,
        sink,
        errors,
    } = outputs;
    let LoopControl {
        rx,
        seek_mailbox,
        pause_requested,
        pressure,
        cancel,
    } = control;
    let total = renderer.total_frames();
    let fps = renderer.fps();
    if total <= 0 {
        if let Some(tx) = startup.take() {
            let _ = tx.send(Err("playback timeline has no drawable frames".to_string()));
        }
        return;
    }
    if let Some(frame) = initial_frame {
        clock.seek(frame);
    }
    let frame_dur = Duration::from_secs_f64(1.0 / fps.max(1) as f64);
    let mut frame_deadline = Instant::now();
    let mut paused = false;
    let mut paused_frame = initial_frame;
    let mut buffered_first: Option<(i32, DecodedFrame)> = None;
    // Set when a render failure paused the loop: the next resume restarts
    // decoding (clearing the renderer's negative cache) even at the same frame.
    let mut retry_on_resume = false;

    loop {
        if paused {
            match rx.recv() {
                Ok(PlaybackCmd::Pause(frame, reply)) => {
                    clock.seek(frame);
                    resume_decode_streams(paused_frame, frame, || renderer.seek());
                    if paused_frame != Some(frame) {
                        buffered_first = None;
                    }
                    paused_frame = Some(frame);
                    let _ = reply.send(());
                }
                Ok(PlaybackCmd::Resume(frame, reply)) => {
                    clock.seek(frame);
                    clock.resumed();
                    if std::mem::take(&mut retry_on_resume) {
                        renderer.seek();
                    } else {
                        resume_decode_streams(paused_frame, frame, || renderer.seek());
                    }
                    if let Some((buffered_frame, image)) = buffered_first.take() {
                        sink.push_frame(buffered_frame, image);
                    }
                    paused = false;
                    frame_deadline = Instant::now();
                    pause_requested.store(false, Ordering::Release);
                    let _ = reply.send(());
                }
                Ok(PlaybackCmd::Seek) => {
                    if let Some(request) = seek_mailbox.take() {
                        sink.invalidate();
                        clock.seek(request.frame);
                        renderer.seek();
                        paused_frame = Some(request.frame);
                        buffered_first = None;
                    }
                }
                Ok(PlaybackCmd::Stop) | Err(_) => return,
            }
            continue;
        }

        // Drain pending control messages first.
        loop {
            match rx.try_recv() {
                Ok(PlaybackCmd::Pause(frame, reply)) => {
                    clock.seek(frame);
                    paused = true;
                    paused_frame = Some(frame);
                    let _ = reply.send(());
                    break;
                }
                Ok(PlaybackCmd::Resume(frame, reply)) => {
                    let current_frame = clock.frame(fps);
                    if current_frame != frame {
                        sink.invalidate();
                    }
                    clock.seek(frame);
                    clock.resumed();
                    frame_deadline = Instant::now();
                    resume_decode_streams(Some(current_frame), frame, || renderer.seek());
                    pause_requested.store(false, Ordering::Release);
                    let _ = reply.send(());
                }
                Ok(PlaybackCmd::Seek) => {
                    if let Some(request) = seek_mailbox.take() {
                        // Frames rendered before the seek may still be queued
                        // or encoding; they must not move the playhead back.
                        sink.invalidate();
                        clock.seek(request.frame);
                        renderer.seek();
                    }
                }
                Ok(PlaybackCmd::Stop) => return,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }

        if paused {
            continue;
        }

        // The seek generation is read before the clock: a seek submitted
        // after this read makes the frame stale however the clock read raced
        // it, so a frame picked from a pre-seek clock never publishes.
        let Some(render_generation) = seek_mailbox.render_generation() else {
            thread::yield_now();
            continue;
        };
        let (clamped, done) = loop_step(clock.frame(fps), total);
        if let Some(message) = clock.take_error() {
            errors.report(PlaybackFailure {
                frame: clamped,
                code: PlaybackFailureCode::AudioDecode,
                message,
                fatal: false,
            });
        }
        let rendered = renderer.render(clamped);

        if startup.is_none() && cancel.is_cancelled() {
            // Teardown cancelled the session's decodes (an edit, a project
            // switch or a stop); a `Stop` follows. Whatever the render
            // returned, it is not the user's failure to hear about. (During
            // startup the typed cause still goes to the waiting caller.)
            return;
        }
        if pause_requested.load(Ordering::Acquire) || !seek_mailbox.is_current(render_generation) {
            continue;
        }
        match rendered {
            Ok(frame) => {
                if let Some(tx) = startup.take() {
                    if tx.send(Ok(())).is_err() {
                        return;
                    }
                    buffered_first = Some((clamped, frame));
                    paused = true;
                    paused_frame = Some(clamped);
                } else {
                    sink.push_frame(clamped, frame);
                }
            }
            Err(failure) => {
                if let Some(tx) = startup.take() {
                    let _ = tx.send(Err(failure.message));
                    return;
                }
                // Hold the last published frame instead of retrying (and
                // respawning decoders) every tick, and silence audio: the
                // transport is stopped until a resume or seek retries.
                paused = true;
                paused_frame = Some(clamped);
                retry_on_resume = true;
                clock.halt();
                // Halted until a resume: background inference may run.
                release_pressure_slot(&pressure);
                if done {
                    eprintln!("[playback] final frame {clamped}: {}", failure.message);
                    sink.push_terminal(clamped);
                } else {
                    errors.report(PlaybackFailure {
                        frame: clamped,
                        code: failure.code,
                        message: failure.message,
                        fatal: true,
                    });
                }
                continue;
            }
        }

        // Auto-stop once the clock reaches the final frame (#53: end → stop).
        if done || paused {
            if done {
                // Reaching the end is a pause the command layer never sees.
                release_pressure_slot(&pressure);
            }
            paused = true;
            continue;
        }

        // A relative sleep accumulates timer wakeup delays on every frame.
        // Anchor ticks to the cadence; retain immediate catch-up for overruns.
        let now = Instant::now();
        frame_deadline = next_render_deadline(frame_deadline, frame_dur, now);
        if frame_deadline > now {
            thread::sleep(frame_deadline - now);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn late_timer_wakeups_do_not_accumulate_in_render_cadence() {
        let origin = Instant::now();
        let period = Duration::from_millis(10);
        let mut deadline = origin;
        for tick in 1..=100 {
            let completed = origin + period * (tick - 1) + Duration::from_millis(6);
            deadline = next_render_deadline(deadline, period, completed);
            assert_eq!(deadline, origin + period * tick);
        }
    }

    #[test]
    fn an_overrun_keeps_immediate_clock_catchup() {
        let origin = Instant::now();
        let completed = origin + Duration::from_millis(40);
        assert_eq!(
            next_render_deadline(origin, Duration::from_millis(33), completed),
            completed
        );
    }

    use std::sync::atomic::AtomicI32;

    use opentake_media::MediaError;

    /// Advances one frame per read, like a clock running at the render rate.
    struct SteppingClock {
        next: AtomicI32,
        audio_errors: Mutex<Vec<String>>,
        halts: AtomicI32,
    }

    impl SteppingClock {
        fn new() -> Self {
            Self {
                next: AtomicI32::new(0),
                audio_errors: Mutex::new(Vec::new()),
                halts: AtomicI32::new(0),
            }
        }
    }

    impl PlaybackClock for SteppingClock {
        fn frame(&self, _fps: i32) -> i32 {
            self.next.fetch_add(1, Ordering::AcqRel)
        }

        fn seek(&self, frame: i32) {
            self.next.store(frame, Ordering::Release);
        }

        fn take_error(&self) -> Option<String> {
            self.audio_errors.lock().unwrap().pop()
        }

        fn halt(&self) {
            self.halts.fetch_add(1, Ordering::AcqRel);
        }
    }

    /// A render at `frame` that blocks until `cancel` fires, then fails the
    /// way an interrupted decode does.
    struct BlockAt {
        frame: i32,
        cancel: MediaCancelToken,
        entered: mpsc::Sender<()>,
    }

    /// A render at `frame` that signals `entered` and then waits for
    /// `release` (a slow 4K or long-GOP frame), then succeeds.
    struct HoldAt {
        frame: i32,
        entered: mpsc::Sender<()>,
        release: Mutex<mpsc::Receiver<()>>,
    }

    struct StubRenderer {
        total: i32,
        fail_at: Option<i32>,
        renders: Arc<Mutex<Vec<i32>>>,
        seeks: Arc<AtomicI32>,
        block_at: Option<BlockAt>,
        hold_at: Option<HoldAt>,
    }

    impl FrameRenderer for StubRenderer {
        fn total_frames(&self) -> i32 {
            self.total
        }

        fn fps(&self) -> i32 {
            1_000
        }

        fn render(&mut self, target: i32) -> Result<DecodedFrame, RenderFailure> {
            self.renders.lock().unwrap().push(target);
            if let Some(hold) = self.hold_at.as_ref().filter(|hold| hold.frame == target) {
                let _ = hold.entered.send(());
                let _ = hold
                    .release
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(5));
            }
            if let Some(block) = self.block_at.as_ref().filter(|block| block.frame == target) {
                let _ = block.entered.send(());
                while !block.cancel.is_cancelled() {
                    thread::sleep(Duration::from_millis(1));
                }
                return Err(RenderFailure::new(
                    PlaybackFailureCode::VideoDecode,
                    format!("clip-1 at frame {target}: {}", MediaError::Cancelled),
                ));
            }
            if self.fail_at == Some(target) {
                return Err(RenderFailure::new(
                    PlaybackFailureCode::VideoDecode,
                    format!("clip-1 decode failed before source frame {target}"),
                ));
            }
            Ok(DecodedFrame::new(1, 1, vec![0, 0, 0, 255], false))
        }

        fn seek(&mut self) {
            self.seeks.fetch_add(1, Ordering::AcqRel);
        }
    }

    #[derive(Default)]
    struct RecordingSink {
        frames: Mutex<Vec<i32>>,
        terminals: Mutex<Vec<i32>>,
        /// Every push and invalidation in order: `Some(frame)` or `None`.
        events: Mutex<Vec<Option<i32>>>,
    }

    impl FrameSink for RecordingSink {
        fn push_frame(&self, frame: i32, _image: DecodedFrame) {
            self.frames.lock().unwrap().push(frame);
            self.events.lock().unwrap().push(Some(frame));
        }

        fn push_terminal(&self, frame: i32) {
            self.terminals.lock().unwrap().push(frame);
        }

        fn invalidate(&self) {
            self.events.lock().unwrap().push(None);
        }
    }

    #[derive(Default)]
    struct RecordingErrors(Mutex<Vec<PlaybackFailure>>);

    impl PlaybackErrorSink for RecordingErrors {
        fn report(&self, failure: PlaybackFailure) {
            self.0.lock().unwrap().push(failure);
        }
    }

    struct StubRun {
        engine: PlaybackEngine,
        renders: Arc<Mutex<Vec<i32>>>,
        seeks: Arc<AtomicI32>,
        sink: Arc<RecordingSink>,
        errors: Arc<RecordingErrors>,
        clock: Arc<SteppingClock>,
    }

    fn run_stub(total: i32, fail_at: Option<i32>) -> StubRun {
        run_stub_with(total, fail_at, None, MediaCancelToken::new())
    }

    fn run_stub_with(
        total: i32,
        fail_at: Option<i32>,
        block_at: Option<BlockAt>,
        cancel: MediaCancelToken,
    ) -> StubRun {
        run_stub_full(total, fail_at, block_at, None, cancel)
    }

    fn run_stub_full(
        total: i32,
        fail_at: Option<i32>,
        block_at: Option<BlockAt>,
        hold_at: Option<HoldAt>,
        cancel: MediaCancelToken,
    ) -> StubRun {
        let renders = Arc::new(Mutex::new(Vec::new()));
        let seeks = Arc::new(AtomicI32::new(0));
        let sink = Arc::new(RecordingSink::default());
        let errors = Arc::new(RecordingErrors::default());
        let clock = Arc::new(SteppingClock::new());
        let renderer = StubRenderer {
            total,
            fail_at,
            renders: Arc::clone(&renders),
            seeks: Arc::clone(&seeks),
            block_at,
            hold_at,
        };
        let engine = PlaybackEngine::spawn_with(
            move || Ok(renderer),
            RenderOutputs {
                clock: clock.clone(),
                sink: sink.clone(),
                errors: errors.clone(),
            },
            None,
            None,
            cancel,
        )
        .expect("spawn stub render loop");
        StubRun {
            engine,
            renders,
            seeks,
            sink,
            errors,
            clock,
        }
    }

    fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(3);
        while !done() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn render_thread_releases_playback_pressure_whenever_it_stops_playing() {
        let pressure = opentake_media::ExportPause::new();

        // An explicit pause releases at once.
        let run = run_stub(1_000_000, None);
        run.engine.control().hold_pressure(1, pressure.guard());
        run.engine.pause(0).expect("pause");
        assert!(!pressure.is_active(), "pause releases pressure");

        // Resuming until the end of the timeline releases without a pause.
        let run = run_stub(50, None);
        wait_until("the loop to reach the end", || {
            run.renders.lock().unwrap().contains(&49)
        });
        run.engine.control().hold_pressure(2, pressure.guard());
        run.engine.resume(0).expect("resume from the start");
        wait_until("the end-of-timeline release", || !pressure.is_active());

        // A render failure halts the loop and releases.
        let failing = run_stub(100, Some(5));
        wait_until("the failing render", || {
            failing.errors.0.lock().unwrap().len() == 1
        });
        failing.engine.control().hold_pressure(3, pressure.guard());
        failing.engine.resume(5).expect("retry resume");
        wait_until("the failure-halt release", || !pressure.is_active());

        // Stopping (the render thread exits) releases.
        failing.engine.control().hold_pressure(4, pressure.guard());
        failing.engine.stop();
        assert!(!pressure.is_active(), "thread exit releases pressure");
        run.engine.stop();
    }

    #[test]
    fn a_stale_queued_pause_does_not_drop_a_newer_resumes_pressure() {
        let pressure = opentake_media::ExportPause::new();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let run = run_stub_full(
            1_000_000,
            None,
            None,
            Some(HoldAt {
                frame: 3,
                entered: entered_tx,
                release: Mutex::new(release_rx),
            }),
            MediaCancelToken::new(),
        );
        let control = run.engine.control();
        control.hold_pressure(1, pressure.guard());
        entered_rx
            .recv_timeout(Duration::from_secs(3))
            .expect("a slow render is in flight");

        // Pause while the render runs: the control side releases at once and
        // the render thread sees the `Pause` only after the render returns.
        control.pause(3).expect("pause");
        assert!(!pressure.is_active());
        // Play again before that: the new resume holds its own guard and
        // waits for the render thread to adopt the resume.
        control.hold_pressure(2, pressure.guard());
        let resumer = {
            let control = control.clone();
            thread::spawn(move || control.resume(3))
        };
        release_tx.send(()).expect("finish the slow render");
        resumer
            .join()
            .expect("join resumer")
            .expect("resume after the stale pause");

        assert!(
            pressure.is_active(),
            "the stale pause must not drop the newer resume's guard"
        );
        run.engine.stop();
        assert!(!pressure.is_active());
    }

    #[test]
    fn an_older_resume_cannot_replace_or_release_a_newer_resumes_pressure() {
        let pressure = opentake_media::ExportPause::new();
        let run = run_stub(1_000_000, None);
        let control = run.engine.control();
        control.hold_pressure(5, pressure.guard());
        control.hold_pressure(4, opentake_media::ExportPause::new().guard());
        control.release_pressure_for(4);
        assert!(pressure.is_active(), "resume 5 still owns the guard");
        control.release_pressure_for(5);
        assert!(!pressure.is_active());
        run.engine.stop();
    }

    #[test]
    fn mid_playback_render_failure_pauses_and_reports_exactly_once() {
        let run = run_stub(100, Some(5));
        wait_until("the failing render", || {
            run.renders.lock().unwrap().contains(&5)
        });
        // Give a runaway loop time to retry before asserting it did not.
        thread::sleep(Duration::from_millis(100));

        assert_eq!(*run.renders.lock().unwrap(), vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(*run.sink.frames.lock().unwrap(), vec![0, 1, 2, 3, 4]);
        assert!(run.sink.terminals.lock().unwrap().is_empty());
        let errors = run.errors.0.lock().unwrap().clone();
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert_eq!(errors[0].frame, 5);
        assert_eq!(errors[0].code, PlaybackFailureCode::VideoDecode);
        assert!(errors[0].fatal);
        assert!(errors[0].message.contains("clip-1"));

        // Resuming is an explicit retry of the same frame.
        run.engine.resume(5).expect("retry resume");
        wait_until("the retried render", || {
            run.errors.0.lock().unwrap().len() == 2
        });
        run.engine.stop();
    }

    #[test]
    fn a_fatal_failure_silences_audio_and_resume_retries_with_fresh_decoders() {
        let run = run_stub(100, Some(5));
        wait_until("the failing render", || {
            run.errors.0.lock().unwrap().len() == 1
        });
        assert_eq!(run.clock.halts.load(Ordering::Acquire), 1);
        let seeks_before = run.seeks.load(Ordering::Acquire);

        // Resuming at the very frame that failed must not be served from the
        // renderer's negative cache: it restarts decoding first.
        run.engine.resume(5).expect("retry resume");
        wait_until("the retried render", || {
            run.errors.0.lock().unwrap().len() == 2
        });
        assert_eq!(run.seeks.load(Ordering::Acquire), seeks_before + 1);
        assert_eq!(run.clock.halts.load(Ordering::Acquire), 2);
        run.engine.stop();
    }

    #[test]
    fn a_render_interrupted_by_session_teardown_reports_nothing() {
        let cancel = MediaCancelToken::new();
        let (entered_tx, entered_rx) = mpsc::channel();
        let run = run_stub_with(
            100,
            None,
            Some(BlockAt {
                frame: 3,
                cancel: cancel.clone(),
                entered: entered_tx,
            }),
            cancel,
        );
        entered_rx
            .recv_timeout(Duration::from_secs(3))
            .expect("the render reaches the blocking decode");

        // Teardown cancels the session's decodes before the loop sees `Stop`.
        run.engine.stop();

        assert!(
            run.errors.0.lock().unwrap().is_empty(),
            "a cancelled session must not report a failure: {:?}",
            run.errors.0.lock().unwrap()
        );
        assert_eq!(run.clock.halts.load(Ordering::Acquire), 0);
        assert!(run.sink.terminals.lock().unwrap().is_empty());
    }

    #[test]
    fn a_seek_while_playing_invalidates_frames_pushed_before_it() {
        let run = run_stub(1_000_000, None);
        wait_until("playback to run", || {
            run.sink.frames.lock().unwrap().len() > 3
        });
        run.engine.seek(500_000);
        wait_until("a frame after the seek", || {
            run.sink
                .frames
                .lock()
                .unwrap()
                .iter()
                .any(|frame| *frame >= 500_000)
        });
        run.engine.stop();

        let events = run.sink.events.lock().unwrap().clone();
        let invalidated = events
            .iter()
            .position(Option::is_none)
            .expect("the seek invalidates queued frames");
        assert!(
            events[..invalidated]
                .iter()
                .flatten()
                .all(|frame| *frame < 500_000),
            "{events:?}"
        );
        assert!(
            events[invalidated + 1..]
                .iter()
                .flatten()
                .all(|frame| *frame >= 500_000),
            "every frame after the invalidation follows the seek: {events:?}"
        );
    }

    #[test]
    fn final_frame_failure_publishes_one_terminal_tick_instead_of_hanging() {
        let run = run_stub(6, Some(5));
        wait_until("the terminal tick", || {
            !run.sink.terminals.lock().unwrap().is_empty()
        });
        thread::sleep(Duration::from_millis(100));

        assert_eq!(*run.sink.terminals.lock().unwrap(), vec![5]);
        assert_eq!(*run.sink.frames.lock().unwrap(), vec![0, 1, 2, 3, 4]);
        assert_eq!(*run.renders.lock().unwrap(), vec![0, 1, 2, 3, 4, 5]);
        assert!(run.errors.0.lock().unwrap().is_empty());
        run.engine.stop();
    }

    #[test]
    fn successful_final_frame_is_pushed_and_the_loop_stops() {
        let run = run_stub(4, None);
        wait_until("the final frame", || {
            run.sink.frames.lock().unwrap().contains(&3)
        });
        thread::sleep(Duration::from_millis(50));
        assert_eq!(*run.sink.frames.lock().unwrap(), vec![0, 1, 2, 3]);
        assert!(run.sink.terminals.lock().unwrap().is_empty());
        run.engine.stop();
    }

    #[test]
    fn audio_source_errors_are_forwarded_as_non_fatal_failures() {
        let run = run_stub(1_000_000, None);
        run.clock
            .audio_errors
            .lock()
            .unwrap()
            .push("audio window at 2.0 s failed".to_string());
        wait_until("the audio report", || {
            !run.errors.0.lock().unwrap().is_empty()
        });
        let failure = run.errors.0.lock().unwrap()[0].clone();
        assert_eq!(failure.code, PlaybackFailureCode::AudioDecode);
        assert!(!failure.fatal);
        let rendered = run.renders.lock().unwrap().len();
        wait_until("rendering to continue", || {
            run.renders.lock().unwrap().len() > rendered + 5
        });
        assert_eq!(run.errors.0.lock().unwrap().len(), 1);
        run.engine.stop();
    }

    #[test]
    fn pause_request_does_not_wait_for_an_inflight_render_to_finish() {
        let (engine, pause_seen, release_pause) = PlaybackEngine::test_blocking_pause();
        let (result_tx, result_rx) = mpsc::channel();
        let caller = thread::spawn(move || {
            let result = engine.pause(73);
            let _ = result_tx.send(result);
        });

        assert_eq!(
            pause_seen
                .recv_timeout(Duration::from_secs(1))
                .expect("pause reaches render thread"),
            73
        );
        // Give the caller thread a bounded scheduling window. An immediate
        // try_recv races the caller's result send against the render fixture's
        // pause_seen send and flakes under loaded CI runners even though
        // `pause` has already returned without waiting for the render reply.
        let returned_before_render_release = result_rx.recv_timeout(Duration::from_secs(1));
        release_pause
            .send(())
            .expect("release synthetic inflight render");
        caller.join().expect("join pause caller");

        assert!(
            matches!(returned_before_render_release, Ok(Ok(()))),
            "pause must acknowledge after enqueueing, not after the slow render finishes"
        );
    }

    #[test]
    fn bounded_reaper_rejects_new_start_when_teardown_backlog_is_full() {
        let reaper = BoundedReaper::new();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let release_rx = Arc::new(Mutex::new(release_rx));

        for _ in 0..2 {
            let permit = reaper
                .try_reserve()
                .expect("two teardown jobs fit the bounded reaper");
            let job_release = Arc::clone(&release_rx);
            let handle = thread::spawn(move || {
                job_release
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .recv()
                    .expect("release teardown handle");
            });
            permit
                .enqueue(vec![handle])
                .expect("enqueue teardown handles");
        }

        assert!(
            reaper.try_reserve().is_err(),
            "a third start must be rejected while two teardowns are outstanding"
        );
        release_tx.send(()).expect("release first teardown");
        release_tx.send(()).expect("release second teardown");
        reaper.wait_until_idle(Duration::from_secs(2));
    }

    #[test]
    fn frame_at_elapsed_truncates_not_rounds() {
        // 0.999 frames of elapsed time is still frame 0 (truncate toward zero).
        assert_eq!(frame_at_elapsed(0, 0.999 / 30.0, 30), 0);
        // Exactly one frame's worth advances by one.
        assert_eq!(frame_at_elapsed(0, 1.0 / 30.0, 30), 1);
        // 2.5 frames -> 2 (no rounding up).
        assert_eq!(frame_at_elapsed(0, 2.5 / 30.0, 30), 2);
    }

    #[test]
    fn frame_at_elapsed_applies_base_offset() {
        assert_eq!(frame_at_elapsed(100, 1.0, 30), 130);
    }

    #[test]
    fn a_pending_seek_cannot_be_tagged_as_a_render_from_the_old_clock() {
        let mailbox = SeekMailbox::default();
        assert_eq!(mailbox.render_generation(), Some(0));
        mailbox.submit(25);
        mailbox.submit(40);
        assert_eq!(mailbox.render_generation(), None);
        let request = mailbox
            .take()
            .expect("the latest seek must be applied first");
        assert_eq!(request.frame, 40);
        assert_eq!(mailbox.render_generation(), Some(request.generation));
    }

    #[test]
    fn loop_step_clamps_and_flags_end() {
        assert_eq!(loop_step(5, 100), (5, false));
        assert_eq!(loop_step(99, 100), (99, true)); // last frame → done
        assert_eq!(loop_step(150, 100), (99, true)); // past end → clamp + done
        assert_eq!(loop_step(-5, 100), (0, false)); // negative → clamp to 0
        assert_eq!(loop_step(0, 1), (0, true)); // single-frame timeline
    }

    #[test]
    fn frame_at_elapsed_clamps_negative_elapsed_and_bad_fps() {
        assert_eq!(frame_at_elapsed(10, -5.0, 30), 10);
        // fps <= 0 falls back to 30, so one second is 30 frames.
        assert_eq!(frame_at_elapsed(0, 1.0, 0), 30);
    }

    #[test]
    fn instant_clock_seek_resets_base_frame() {
        let clock = InstantClock::new(0);
        clock.seek(500);
        // Immediately after a seek, ~no time has elapsed, so we're at the base.
        let f = clock.frame(30);
        assert!(
            (500..=501).contains(&f),
            "expected ~500 right after seek, got {f}"
        );
    }

    #[test]
    fn resuming_paused_frame_preserves_decoders_but_moving_clears_them() {
        let mut decoder_starts = 1;
        let mut streams = HashMap::from([("visible_clip".to_string(), decoder_starts)]);

        // A paused frame already has an active decode stream. Resume at the
        // same frame must use it rather than launching another process.
        resume_decode_streams(Some(30), 30, || streams.clear());
        streams
            .entry("visible_clip".to_string())
            .or_insert_with(|| {
                decoder_starts += 1;
                decoder_starts
            });
        assert_eq!(decoder_starts, 1);
        assert_eq!(streams["visible_clip"], 1);

        // A different requested position still discards the old stream.
        resume_decode_streams(Some(30), 90, || streams.clear());
        streams
            .entry("visible_clip".to_string())
            .or_insert_with(|| {
                decoder_starts += 1;
                decoder_starts
            });
        assert_eq!(decoder_starts, 2);
        assert_eq!(streams["visible_clip"], 2);
    }

    #[test]
    fn rapid_seek_mailbox_keeps_only_the_latest_frame_and_one_wake() {
        let mailbox = SeekMailbox::default();
        let wake_count = (0..20)
            .filter(|frame| mailbox.submit(*frame).should_wake)
            .count();

        assert_eq!(wake_count, 1);
        assert_eq!(
            mailbox.take(),
            Some(SeekRequest {
                frame: 19,
                generation: 20
            })
        );
        assert_eq!(mailbox.take(), None);
    }

    /// A clock whose read at `trigger` races a seek to `target`: the seek is
    /// submitted while the read is under way and the pre-seek frame returned.
    struct SeekRacingClock {
        next: AtomicI32,
        trigger: i32,
        target: i32,
        control: std::sync::OnceLock<EngineControl>,
    }

    impl PlaybackClock for SeekRacingClock {
        fn frame(&self, _fps: i32) -> i32 {
            let frame = self.next.fetch_add(1, Ordering::AcqRel);
            if frame == self.trigger {
                let deadline = Instant::now() + Duration::from_secs(3);
                while self.control.get().is_none() && Instant::now() < deadline {
                    thread::yield_now();
                }
                self.control
                    .get()
                    .expect("engine control installed")
                    .seek(self.target);
            }
            frame
        }

        fn seek(&self, frame: i32) {
            self.next.store(frame, Ordering::Release);
        }
    }

    #[test]
    fn a_frame_picked_from_a_pre_seek_clock_never_publishes() {
        let clock = Arc::new(SeekRacingClock {
            next: AtomicI32::new(0),
            trigger: 3,
            target: 500,
            control: std::sync::OnceLock::new(),
        });
        let sink = Arc::new(RecordingSink::default());
        let renders = Arc::new(Mutex::new(Vec::new()));
        let renderer = StubRenderer {
            total: 1_000,
            fail_at: None,
            renders: Arc::clone(&renders),
            seeks: Arc::new(AtomicI32::new(0)),
            block_at: None,
            hold_at: None,
        };
        let engine = PlaybackEngine::spawn_with(
            move || Ok(renderer),
            RenderOutputs {
                clock: clock.clone(),
                sink: sink.clone(),
                errors: Arc::new(RecordingErrors::default()),
            },
            None,
            None,
            MediaCancelToken::new(),
        )
        .expect("spawn stub render loop");
        let _ = clock.control.set(engine.control());

        wait_until("post-seek frames", || {
            sink.frames
                .lock()
                .unwrap()
                .iter()
                .any(|frame| *frame >= 510)
        });
        engine.stop();
        let frames = sink.frames.lock().unwrap().clone();
        assert!(
            !frames.contains(&3),
            "the frame read while the seek landed is stale: {frames:?}"
        );
        assert!(frames.iter().all(|frame| *frame < 3 || *frame >= 500));
    }

    #[test]
    fn a_new_seek_generation_supersedes_an_inflight_render() {
        let mailbox = SeekMailbox::default();
        mailbox.submit(10);
        let rendering = mailbox.take().expect("first seek");
        mailbox.submit(99);

        assert!(!mailbox.is_current(rendering.generation));
        assert_eq!(
            mailbox.take(),
            Some(SeekRequest {
                frame: 99,
                generation: 2
            })
        );
    }

    #[test]
    fn rewinding_active_video_source_requests_stream_reset() {
        static NO_MASKS: [opentake_domain::Mask; 0] = [];
        static NO_EFFECTS: [opentake_domain::Effect; 0] = [];
        let previous = HashMap::from([("clip-1".to_string(), 7)]);
        let source = TextureSource::Decoded {
            media_ref: "asset".to_string(),
        };
        let current = active_video_sources(&FramePlan {
            clear_rgba: [0.0, 0.0, 0.0, 1.0],
            draws: vec![opentake_render::LayerDraw {
                source: &source,
                source_frame: 2,
                affine: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                nat_size: (1.0, 1.0),
                crop_uv: (0.0, 0.0, 1.0, 1.0),
                opacity: 1.0,
                clip_id: "clip-1",
                color_grade: None,
                lut: None,
                chroma_key: None,
                masks: &NO_MASKS,
                effects: &NO_EFFECTS,
            }],
        });

        assert_eq!(
            rewound_video_sources(&previous, &current, &HashSet::new()),
            vec!["clip-1"]
        );
    }

    #[test]
    fn forward_active_video_source_keeps_stream_state() {
        static NO_MASKS: [opentake_domain::Mask; 0] = [];
        static NO_EFFECTS: [opentake_domain::Effect; 0] = [];
        let previous = HashMap::from([("clip-1".to_string(), 2)]);
        let source = TextureSource::Decoded {
            media_ref: "asset".to_string(),
        };
        let current = active_video_sources(&FramePlan {
            clear_rgba: [0.0, 0.0, 0.0, 1.0],
            draws: vec![opentake_render::LayerDraw {
                source: &source,
                source_frame: 7,
                affine: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                nat_size: (1.0, 1.0),
                crop_uv: (0.0, 0.0, 1.0, 1.0),
                opacity: 1.0,
                clip_id: "clip-1",
                color_grade: None,
                lut: None,
                chroma_key: None,
                masks: &NO_MASKS,
                effects: &NO_EFFECTS,
            }],
        });

        assert!(rewound_video_sources(&previous, &current, &HashSet::new()).is_empty());
    }

    #[test]
    fn only_the_rewound_clip_restarts_its_stream() {
        let previous = HashMap::from([("rewound".to_string(), 7), ("forward".to_string(), 3)]);
        let current = HashMap::from([("rewound".to_string(), 2), ("forward".to_string(), 4)]);

        assert_eq!(
            rewound_video_sources(&previous, &current, &HashSet::new()),
            vec!["rewound"]
        );
    }

    #[test]
    fn reversed_clip_never_restarts_streams_across_sixty_frames() {
        use opentake_domain::{Clip, ClipType, Track};

        let mut timeline = Timeline::new();
        timeline.fps = 30;
        let mut forward_track = Track::new("t1", ClipType::Video);
        forward_track
            .clips
            .push(Clip::new("forward", "asset-forward", 0, 60));
        let mut reversed_track = Track::new("t2", ClipType::Video);
        let mut reversed = Clip::new("reversed", "asset-reversed", 0, 60);
        reversed.reversed = true;
        reversed_track.clips.push(reversed);
        timeline.tracks.push(forward_track);
        timeline.tracks.push(reversed_track);
        let metrics = ManifestMetrics {
            sizes: HashMap::from([
                ("asset-forward".to_string(), (64, 36)),
                ("asset-reversed".to_string(), (64, 36)),
            ]),
        };
        let plan = try_build_render_plan(&timeline, RenderSize::new(64, 36), &metrics)
            .expect("two-track plan");
        let reversed_clips = reversed_video_clips(&plan);
        assert_eq!(reversed_clips, HashSet::from(["reversed".to_string()]));

        let mut previous = HashMap::new();
        for frame in 0..60 {
            let current = active_video_sources(&plan.frame(&timeline, frame));
            assert_eq!(current.len(), 2, "both clips draw at frame {frame}");
            if frame > 0 {
                assert!(current["reversed"] < previous["reversed"]);
                assert!(current["forward"] > previous["forward"]);
            }
            assert!(
                rewound_video_sources(&previous, &current, &reversed_clips).is_empty(),
                "frame {frame} must not restart any stream"
            );
            previous = current;
        }
    }
}
