//! Streaming texture resolver for continuous playback (#53).
//!
//! Where the preview's [`crate::render::composite_frame`] resolves each video
//! layer with a fresh seek-per-frame `decode_frame_at` (correct but far too slow
//! for real-time multi-track playback), this resolver keeps **one forward
//! [`VideoStream`] per active clip** and pulls frames out of each clip's bounded
//! queue to match the frame the compositor is asking for. Sequential decode (no
//! per-frame seek) is the whole point — that is what makes high-bitrate / ProRes
//! playback smooth.
//!
//! ## Two-part shape (why a persistent state + a transient resolver)
//! The compositor's [`TextureResolver`] trait hands `resolve()` only a
//! `(&TextureSource, source_frame)` — **no `clip_id`**. But stream *lifecycle*
//! must be keyed by clip id (a split clip, or the same asset reused twice, needs
//! its own decode position). So lifecycle can't live inside `resolve()`.
//!
//! Instead the render thread owns the persistent [`PlaybackResolverState`]
//! (the per-clip streams + the static image/text caches), and each frame wraps it
//! in a transient [`StreamingResolver`] that borrows the wgpu device + the state.
//! Before compositing, the thread calls [`StreamingResolver::sync_active`] with
//! the frame's [`FramePlan`]: that adds/stops streams by `clip_id`, advances each
//! to its target `source_frame`, and pre-uploads the matching textures into a
//! per-frame lookup keyed `v:{media_ref}:{source_frame}`. Multiple clips may
//! share that key; when one decoder is behind, the exact-frame candidate wins
//! over its stale fallback regardless of draw order. `resolve()` then degrades
//! to a table lookup for video, the static cache for image / text, and the
//! content-hash + internal-frame LRU for Lottie.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;
use std::sync::mpsc::{RecvTimeoutError, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use opentake_domain::LutReference;
use opentake_media::decode::{
    spawn_video_stream, StreamVideoFrame, VideoStream, VideoStreamRequest,
};
use opentake_media::{
    decode_frame_at_cancellable, ContentHashCache, FrameRequest, MediaCancelToken, MediaError,
};
use opentake_project::ProjectRoot;
use opentake_render::gpu::texture::upload_rgba;
use opentake_render::wgpu;
use opentake_render::{
    rasterize_text_layer, CosmicTextRasterizer, DecodedFrame, FramePlan, GpuLutTexture, GpuTexture,
    TextRasterRequest, TextRasterizer, TextureCache, TextureResolver, TextureSource,
};

use crate::render::LottieMaterializer;

use super::project::{MediaInfo, TextInfo};

/// Per-frame texture cache size for static (image / text) layers. Video frames
/// are NOT cached here — they live in each clip's stream and are uploaded per
/// frame. Bounds VRAM for the static layers.
const STATIC_CACHE_CAP: usize = 64;

/// Source frames decoded per reverse-playback window. A reversed clip's source
/// frame descends every tick, which a forward decoder cannot follow, so it is
/// served from short forward-decoded windows instead: one decode process per
/// window rather than one per frame.
const REVERSE_WINDOW_FRAMES: i64 = 16;
/// Start the next decode as soon as the current window is buffered. Long-GOP
/// seeks need the full window's playback time, rather than half of it.
const REVERSE_PREFETCH_FRAMES: i64 = REVERSE_WINDOW_FRAMES;
const REVERSE_BOOTSTRAP_TIMEOUT: Duration = Duration::from_secs(10);

/// One active video clip's continuous-decode state. Created when a clip first
/// appears in a frame plan, dropped (after a cooperative stop) when it leaves.
struct ClipStream {
    decoder: ClipDecoder,
    /// Most recently uploaded texture, reused when decode falls behind the
    /// target ("drop video, keep the clock moving").
    cached_tex: Rc<GpuTexture>,
    /// Source-frame identity of `cached_tex`. This may trail the requested
    /// target while the decoder catches up.
    cached_source_frame: i64,
}

enum ClipDecoder {
    /// The forward ffmpeg decode worker for this clip's source, plus a frame
    /// pulled off the queue that is *ahead* of the current target, held for a
    /// future tick instead of being discarded (slow-motion / dup frames).
    Forward {
        stream: VideoStream,
        pending: Option<StreamVideoFrame>,
    },
    /// A reversed clip: descending source frames from bounded windows.
    Reverse {
        windows: ReverseWindows<VideoStream>,
        request: VideoStreamRequest,
    },
}

impl ClipStream {
    fn new(decoder: ClipDecoder, cached_tex: Rc<GpuTexture>, cached_source_frame: i64) -> Self {
        ClipStream {
            decoder,
            cached_tex,
            cached_source_frame,
        }
    }

    fn request_stop(&self) {
        match &self.decoder {
            ClipDecoder::Forward { stream, .. } => stream.request_stop(),
            ClipDecoder::Reverse { windows, .. } => windows.request_stop(),
        }
    }

    /// Advance this clip's decoder to `target`, uploading and caching the
    /// matched frame. Cold bootstrap is completed synchronously before
    /// construction; subsequent calls are non-blocking and retain `cached_tex`
    /// when decode falls behind.
    fn advance(
        &mut self,
        target: i64,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), String> {
        let cached_source_frame = self.cached_source_frame;
        let next = match &mut self.decoder {
            ClipDecoder::Forward { stream, pending } => {
                let rx = stream.receiver();
                drain_to_target(
                    pending,
                    || classify_stream_pull(rx.try_recv(), cached_source_frame, target),
                    target,
                )?
            }
            ClipDecoder::Reverse { windows, request } => {
                windows.advance(target, cached_source_frame, |start, end| {
                    spawn_reverse_window(request, start, end)
                })?
            }
        };
        if let Some(vf) = next {
            self.cached_source_frame = vf.source_frame;
            let decoded = DecodedFrame::new(vf.frame.width, vf.frame.height, vf.frame.rgba, false);
            let tex = upload_rgba(device, queue, &decoded, false, Some("playback-src"));
            self.cached_tex = Rc::new(tex);
        }
        Ok(())
    }
}

/// Non-blocking frame supply of one decode worker.
trait FrameSupply {
    fn try_pull(&self) -> Result<Result<StreamVideoFrame, MediaError>, TryRecvError>;
    fn request_stop(&self);
}

impl FrameSupply for VideoStream {
    fn try_pull(&self) -> Result<Result<StreamVideoFrame, MediaError>, TryRecvError> {
        self.receiver().try_recv()
    }

    fn request_stop(&self) {
        VideoStream::request_stop(self);
    }
}

fn spawn_reverse_window(
    template: &VideoStreamRequest,
    start: i64,
    end: i64,
) -> Result<VideoStream, String> {
    let mut req = template.clone();
    req.start_frame = start;
    req.end_frame = Some(end);
    req.queue_capacity = REVERSE_WINDOW_FRAMES as usize;
    spawn_video_stream(req).map_err(|error| {
        format!("playback reverse window [{start}, {end}) failed to start: {error}")
    })
}

/// Prepare the first picture and its earlier frames in one bounded decode.
/// An exact single-frame seek followed by another window seek would replay the
/// same GOP and leave the first playback ticks with no earlier picture ready.
fn bootstrap_reverse_window(
    request: &VideoStreamRequest,
    target: i64,
    cancel: &MediaCancelToken,
) -> Result<(ReverseWindows<VideoStream>, StreamVideoFrame), String> {
    let start = (target + 1 - REVERSE_WINDOW_FRAMES).max(0);
    let stream = spawn_reverse_window(request, start, target + 1)?;
    let deadline = Instant::now() + REVERSE_BOOTSTRAP_TIMEOUT;
    let mut frames = BTreeMap::new();
    loop {
        if cancel.is_cancelled() {
            stream.request_stop();
            return Err("playback reverse bootstrap cancelled".into());
        }
        if Instant::now() >= deadline {
            stream.request_stop();
            return Err("playback reverse bootstrap timed out".into());
        }
        match stream.receiver().recv_timeout(Duration::from_millis(25)) {
            Ok(Ok(frame)) if frame.source_frame == target => {
                return Ok((
                    ReverseWindows {
                        frames,
                        inflight: Some(stream),
                        coverage: (start, target + 1),
                    },
                    frame,
                ));
            }
            Ok(Ok(frame)) if (start..target).contains(&frame.source_frame) => {
                frames.insert(frame.source_frame, frame);
            }
            Ok(Ok(frame)) => {
                return Err(format!(
                    "playback reverse bootstrap skipped frame {target} (next {})",
                    frame.source_frame
                ));
            }
            Ok(Err(error)) => return Err(format!("playback reverse bootstrap failed: {error}")),
            Err(RecvTimeoutError::Disconnected) => {
                return Err(format!(
                    "playback reverse bootstrap ended before frame {target}"
                ));
            }
            Err(RecvTimeoutError::Timeout) => {}
        }
    }
}

/// Serves a reversed clip's descending source frames. Each window decodes
/// source frames `[start, end)` forward in the background; the render thread
/// only drains finished frames and hands them out in descending order, and
/// requests the next earlier window before the buffered ones run out. At most
/// one window decodes at a time, so decode processes scale with the number of
/// windows played, never with the number of frames.
struct ReverseWindows<S> {
    /// Decoded frames at or below the current target.
    frames: BTreeMap<i64, StreamVideoFrame>,
    inflight: Option<S>,
    /// Source frames `[low, high)` shown or requested by the current chain.
    coverage: (i64, i64),
}

impl<S: FrameSupply> ReverseWindows<S> {
    /// Start a chain at the synchronously bootstrapped `source_frame`.
    #[cfg(test)]
    fn new(source_frame: i64) -> Self {
        ReverseWindows {
            frames: BTreeMap::new(),
            inflight: None,
            coverage: (source_frame, source_frame + 1),
        }
    }

    fn request_stop(&self) {
        if let Some(supply) = &self.inflight {
            supply.request_stop();
        }
    }

    fn restart(&mut self) {
        if let Some(supply) = self.inflight.take() {
            supply.request_stop();
        }
        self.frames.clear();
    }

    /// Return the decoded frame at `target` when it is ready (`None` reuses the
    /// displayed `cached_source_frame`), spawning windows through `spawn`.
    fn advance(
        &mut self,
        target: i64,
        cached_source_frame: i64,
        mut spawn: impl FnMut(i64, i64) -> Result<S, String>,
    ) -> Result<Option<StreamVideoFrame>, String> {
        if let Some(supply) = &self.inflight {
            loop {
                match supply.try_pull() {
                    Ok(Ok(frame)) => {
                        self.frames.insert(frame.source_frame, frame);
                    }
                    Ok(Err(error)) => {
                        return Err(format!(
                            "playback reverse decode failed before source frame {target}: {error}"
                        ));
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        self.inflight = None;
                        break;
                    }
                }
            }
        }
        // Reverse playback only descends: frames above the target were shown.
        drop(self.frames.split_off(&(target + 1)));
        let hit = self.frames.remove(&target);
        let (low, high) = self.coverage;
        if !(low..high).contains(&target) {
            // Outside the requested chain (decode fell a whole window behind,
            // or the target moved up): restart the chain at the target.
            self.restart();
            if cached_source_frame == target {
                self.coverage = (target, target + 1);
            } else {
                let start = (target + 1 - REVERSE_WINDOW_FRAMES).max(0);
                self.inflight = Some(spawn(start, target + 1)?);
                self.coverage = (start, target + 1);
                return Ok(None);
            }
        }
        self.coverage.1 = target + 1;
        let low = self.coverage.0;
        if self.inflight.is_none() && low > 0 && target - low < REVERSE_PREFETCH_FRAMES {
            let start = (low - REVERSE_WINDOW_FRAMES).max(0);
            self.inflight = Some(spawn(start, low)?);
            self.coverage.0 = start;
        }
        Ok(hit)
    }
}

fn classify_stream_pull(
    result: Result<Result<StreamVideoFrame, MediaError>, TryRecvError>,
    cached_source_frame: i64,
    target: i64,
) -> Result<Option<StreamVideoFrame>, String> {
    match result {
        Ok(Ok(frame)) => Ok(Some(frame)),
        Ok(Err(error)) => Err(format!(
            "playback continuous decode failed before source frame {target}: {error}"
        )),
        Err(TryRecvError::Empty) => Ok(None),
        Err(TryRecvError::Disconnected) if cached_source_frame >= target => Ok(None),
        Err(TryRecvError::Disconnected) => Err(format!(
            "playback continuous decode ended at source frame {cached_source_frame} before requested source frame {target}"
        )),
    }
}

fn ensure_stream_with<'a, S, E>(
    streams: &'a mut HashMap<String, S>,
    clip_id: &str,
    create: impl FnOnce() -> Result<S, E>,
) -> Result<&'a mut S, E> {
    use std::collections::hash_map::Entry;

    match streams.entry(clip_id.to_string()) {
        Entry::Occupied(entry) => Ok(entry.into_mut()),
        Entry::Vacant(entry) => Ok(entry.insert(create()?)),
    }
}

/// A clip whose decoder could not start or ended early. It is not retried
/// until a seek (or a new session), so a broken source does not respawn
/// ffprobe/ffmpeg on every render tick.
#[derive(Clone, Debug, PartialEq, Eq)]
struct StreamFailure {
    media_ref: String,
    message: String,
}

/// [`ensure_stream_with`] behind the negative cache: a recorded failure of the
/// same clip and media answers immediately; a new failure is recorded.
fn ensure_stream_or_cached_failure<'a, S>(
    streams: &'a mut HashMap<String, S>,
    failures: &mut HashMap<String, StreamFailure>,
    clip_id: &str,
    media_ref: &str,
    create: impl FnOnce() -> Result<S, String>,
) -> Result<&'a mut S, String> {
    if let Some(failure) = failures.get(clip_id) {
        if failure.media_ref == media_ref {
            return Err(failure.message.clone());
        }
    }
    ensure_stream_with(streams, clip_id, create).inspect_err(|message| {
        failures.insert(
            clip_id.to_string(),
            StreamFailure {
                media_ref: media_ref.to_string(),
                message: message.clone(),
            },
        );
    })
}

fn bootstrap_frame_request(
    source_frame: i64,
    timeline_fps: i32,
    render_box: (u32, u32),
) -> FrameRequest {
    FrameRequest {
        time_secs: source_frame.max(0) as f64 / timeline_fps.max(1) as f64,
        max_size: render_box,
        apply_rotation: true,
    }
}

/// Pure drain decision: pick the queued frame to display at `target`, discarding
/// stale (behind-target) frames and stashing an ahead-of-target frame in
/// `pending` for a later tick. Returns `Some(frame)` when a frame *at* `target`
/// is available (caller uploads it), or `None` to reuse the cached texture
/// (decode is behind, or the only available frame is still ahead).
///
/// `pull` is the non-blocking queue read (`try_recv`); it returns `None` when the
/// queue is momentarily empty — the render loop never blocks on decode.
fn drain_to_target<E>(
    pending: &mut Option<StreamVideoFrame>,
    mut pull: impl FnMut() -> Result<Option<StreamVideoFrame>, E>,
    target: i64,
) -> Result<Option<StreamVideoFrame>, E> {
    // A frame stashed on a previous tick takes priority over the live queue.
    if let Some(p) = pending.take() {
        if p.source_frame == target {
            return Ok(Some(p));
        }
        if p.source_frame > target {
            *pending = Some(p); // still ahead: keep it, reuse cache this tick
            return Ok(None);
        }
        // p.source_frame < target: stale, drop and fall through to the queue.
    }
    while let Some(f) = pull()? {
        if f.source_frame < target {
            continue; // behind target: discard (fast-forward / normal advance)
        }
        if f.source_frame == target {
            return Ok(Some(f));
        }
        // Over-pulled past the target: stash for a later tick, reuse cache now.
        *pending = Some(f);
        return Ok(None);
    }
    Ok(None)
}

fn should_replace_frame_texture(current_exact: bool, candidate_exact: bool) -> bool {
    candidate_exact && !current_exact
}

struct FrameTexture {
    texture: Rc<GpuTexture>,
    exact: bool,
}

/// The render-thread-owned persistent state behind the streaming resolver: the
/// per-clip decode streams plus the static (image / text) texture cache. Lives
/// for the whole playback session and is wrapped in a transient
/// [`StreamingResolver`] each frame.
pub struct PlaybackResolverState {
    /// Active video streams, keyed by **clip id** (NOT media_ref): a split clip
    /// or a reused asset needs an independent decode position.
    streams: HashMap<String, ClipStream>,
    /// Clip ids whose source frames play in reverse (decoded in windows).
    reversed_clips: HashSet<String>,
    /// Image, text, and bounded Lottie-frame textures (persistent across frames).
    static_cache: TextureCache,
    lottie: LottieMaterializer,
    /// Image content hashes, revalidated by file identity on every lookup.
    content_hashes: ContentHashCache,
    text_rasterizer: Box<dyn TextRasterizer>,
    media: HashMap<String, MediaInfo>,
    text: HashMap<String, TextInfo>,
    timeline_fps: i32,
    /// Decode / raster downscale box (matches the playback render size).
    render_box: (u32, u32),
    cancel: MediaCancelToken,
    project_root: Option<ProjectRoot>,
    lut_cache: HashMap<String, Arc<GpuLutTexture>>,
    materialization_error: Option<String>,
    /// Negative cache of clips whose decoder failed, keyed by clip id.
    failed_streams: HashMap<String, StreamFailure>,
}

impl PlaybackResolverState {
    pub fn new(
        media: HashMap<String, MediaInfo>,
        text: HashMap<String, TextInfo>,
        timeline_fps: i32,
        render_box: (u32, u32),
        cancel: MediaCancelToken,
    ) -> Self {
        Self::new_with_project_root(media, text, timeline_fps, render_box, cancel, None)
    }

    pub fn new_with_project_root(
        media: HashMap<String, MediaInfo>,
        text: HashMap<String, TextInfo>,
        timeline_fps: i32,
        render_box: (u32, u32),
        cancel: MediaCancelToken,
        project_root: Option<ProjectRoot>,
    ) -> Self {
        PlaybackResolverState {
            streams: HashMap::new(),
            reversed_clips: HashSet::new(),
            static_cache: TextureCache::new(STATIC_CACHE_CAP),
            lottie: LottieMaterializer::new(),
            content_hashes: ContentHashCache::new(),
            text_rasterizer: Box::new(CosmicTextRasterizer::new()),
            media,
            text,
            timeline_fps,
            render_box,
            cancel,
            project_root,
            lut_cache: HashMap::new(),
            materialization_error: None,
            failed_streams: HashMap::new(),
        }
    }

    /// Stop and drop every active stream (used on seek: streams restart at the
    /// new position on the next `sync_active`, and failed clips are retried).
    /// Cooperative stop is requested; the worker threads are reaped in the
    /// background via `Drop`, never joined on the render thread.
    pub fn clear_streams(&mut self) {
        for (_, cs) in self.streams.drain() {
            cs.request_stop();
        }
        self.failed_streams.clear();
    }

    /// Replace every authorized media path in `message` with its file name, so
    /// errors shown to the user (and events) never carry absolute paths.
    pub fn redact_media_paths(&self, message: &str) -> String {
        super::project::redact_media_paths(&self.media, message)
    }

    /// Stop and drop only the given clips' streams (a source that moved
    /// backwards restarts at its new position; other clips keep decoding).
    pub fn reset_streams<'a>(&mut self, clip_ids: impl IntoIterator<Item = &'a str>) {
        for clip_id in clip_ids {
            if let Some(cs) = self.streams.remove(clip_id) {
                cs.request_stop();
            }
        }
    }

    /// Mark the clips whose source frames descend during playback so they are
    /// served by reverse windows instead of a forward stream.
    pub fn set_reversed_clips(&mut self, clip_ids: HashSet<String>) {
        self.reversed_clips = clip_ids;
    }

    /// Swap the text backend (tests use a rasterizer that returns `None`).
    #[cfg(test)]
    fn set_text_rasterizer(&mut self, rasterizer: Box<dyn TextRasterizer>) {
        self.text_rasterizer = rasterizer;
    }

    /// Take the first materialization failure recorded during the current
    /// frame. The render loop turns this into an explicit playback error instead
    /// of silently dropping the Lottie layer.
    pub fn take_materialization_error(&mut self) -> Option<String> {
        self.materialization_error.take()
    }

    fn fail_materialization<T>(&mut self, message: impl Into<String>) -> Option<T> {
        if self.materialization_error.is_none() {
            self.materialization_error = Some(message.into());
        }
        None
    }
}

/// One video layer's decode target for a frame: which clip, which asset, and the
/// integer source frame the plan asked for.
struct VideoTarget {
    clip_id: String,
    media_ref: String,
    source_frame: i64,
}

/// Extract the per-clip video decode targets from a frame plan (the `Decoded`
/// layers). Image / text / Lottie layers carry no stream.
fn video_targets(plan: &FramePlan) -> Vec<VideoTarget> {
    plan.draws
        .iter()
        .filter_map(|d| match d.source {
            TextureSource::Decoded { media_ref } => Some(VideoTarget {
                clip_id: d.clip_id.to_string(),
                media_ref: media_ref.clone(),
                source_frame: d.source_frame,
            }),
            _ => None,
        })
        .collect()
}

/// A transient, per-frame [`TextureResolver`] over the persistent
/// [`PlaybackResolverState`] and the render thread's wgpu device. Built fresh
/// each frame; `sync_active` must be called before handing it to the compositor.
pub struct StreamingResolver<'d, 's> {
    device: &'d wgpu::Device,
    queue: &'d wgpu::Queue,
    state: &'s mut PlaybackResolverState,
    /// Per-frame video lookup, keyed `v:{media_ref}:{source_frame}`. Filled by
    /// `sync_active`, read by `resolve`.
    frame_tex: HashMap<String, FrameTexture>,
}

impl<'d, 's> StreamingResolver<'d, 's> {
    pub fn new(
        device: &'d wgpu::Device,
        queue: &'d wgpu::Queue,
        state: &'s mut PlaybackResolverState,
    ) -> Self {
        StreamingResolver {
            device,
            queue,
            state,
            frame_tex: HashMap::new(),
        }
    }

    /// Reconcile the active video streams with this frame's plan and pre-upload
    /// each clip's current texture. Must run before `render_to_rgba`.
    ///
    /// 1. Stop streams whose clip is no longer on screen.
    /// 2. Prepare the exact target and a stream for each newly-visible clip.
    ///    Reversed clips also buffer their first window before playback starts.
    /// 3. Advance every active stream to its target and stash the resulting
    ///    texture in the per-frame lookup.
    pub fn sync_active(&mut self, plan: &FramePlan) -> Result<(), String> {
        let targets = video_targets(plan);
        let active_ids: HashSet<&str> = targets.iter().map(|t| t.clip_id.as_str()).collect();

        // 1. Drop streams for clips that left the frame.
        self.state.streams.retain(|id, cs| {
            if active_ids.contains(id.as_str()) {
                true
            } else {
                cs.request_stop();
                false
            }
        });

        // 2 + 3. Spawn missing streams, advance all, collect textures. Textures
        // are gathered into a local Vec first so `frame_tex` is not borrowed
        // while `state.streams` is.
        let mut uploaded: Vec<(String, Rc<GpuTexture>, bool)> = Vec::with_capacity(targets.len());
        for t in &targets {
            let media_path = self
                .state
                .media
                .get(&t.media_ref)
                .map(|info| info.path.clone())
                .ok_or_else(|| format!("playback bootstrap media not found: {}", t.media_ref))?;
            let timeline_fps = self.state.timeline_fps;
            let render_box = self.state.render_box;
            let reversed = self.state.reversed_clips.contains(&t.clip_id);
            ensure_stream_or_cached_failure(
                &mut self.state.streams,
                &mut self.state.failed_streams,
                &t.clip_id,
                &t.media_ref,
                || {
                    let mut req = VideoStreamRequest::new(media_path, timeline_fps);
                    req.max_size = render_box;
                    let (decoder, frame) = if reversed {
                        let (windows, first) =
                            bootstrap_reverse_window(&req, t.source_frame, &self.state.cancel)?;
                        (
                            ClipDecoder::Reverse {
                                windows,
                                request: req,
                            },
                            first.frame,
                        )
                    } else {
                        let request =
                            bootstrap_frame_request(t.source_frame, timeline_fps, render_box);
                        let (_, frame) = decode_frame_at_cancellable(&req.path, &request, &self.state.cancel)
                            .map_err(|error| format!(
                                "playback bootstrap decode failed for {} at source frame {}: {error}",
                                t.media_ref, t.source_frame
                            ))?;
                        req.start_frame = t.source_frame.max(0).saturating_add(1);
                        let stream = spawn_video_stream(req).map_err(|error| {
                            format!(
                            "playback bootstrap stream failed for {} at source frame {}: {error}",
                            t.media_ref, t.source_frame
                        )
                        })?;
                        (
                            ClipDecoder::Forward {
                                stream,
                                pending: None,
                            },
                            frame,
                        )
                    };
                    let decoded = DecodedFrame::new(frame.width, frame.height, frame.rgba, false);
                    let texture = Rc::new(upload_rgba(
                        self.device,
                        self.queue,
                        &decoded,
                        false,
                        Some("playback-bootstrap"),
                    ));
                    Ok(ClipStream::new(decoder, texture, t.source_frame))
                },
            )?;
            if let Some(cs) = self.state.streams.get_mut(&t.clip_id) {
                if let Err(message) = cs.advance(t.source_frame, self.device, self.queue) {
                    if let Some(failed) = self.state.streams.remove(&t.clip_id) {
                        failed.request_stop();
                    }
                    self.state.failed_streams.insert(
                        t.clip_id.clone(),
                        StreamFailure {
                            media_ref: t.media_ref.clone(),
                            message: message.clone(),
                        },
                    );
                    return Err(message);
                }
                uploaded.push((
                    format!("v:{}:{}", t.media_ref, t.source_frame),
                    cs.cached_tex.clone(),
                    cs.cached_source_frame == t.source_frame,
                ));
            }
        }

        self.frame_tex.clear();
        for (key, texture, exact) in uploaded {
            use std::collections::hash_map::Entry;
            match self.frame_tex.entry(key) {
                Entry::Vacant(entry) => {
                    entry.insert(FrameTexture { texture, exact });
                }
                Entry::Occupied(mut entry)
                    if should_replace_frame_texture(entry.get().exact, exact) =>
                {
                    entry.insert(FrameTexture { texture, exact });
                }
                Entry::Occupied(_) => {}
            }
        }
        Ok(())
    }

    /// Decode (once) and cache a static image layer, mirroring the preview
    /// resolver's image path.
    fn resolve_image(&mut self, media_ref: &str) -> Option<Rc<GpuTexture>> {
        let Some(info) = self.state.media.get(media_ref) else {
            return self
                .state
                .fail_materialization(format!("image source {media_ref} is unauthorized"));
        };
        let content_hash = match self.state.content_hashes.sha256(&info.path) {
            Ok(hash) => hash,
            Err(error) => {
                return self.state.fail_materialization(format!(
                    "image source {media_ref} is unavailable: {error}"
                ));
            }
        };
        let key = format!("i:{content_hash}");
        if let Some(tex) = self.state.static_cache.get(&key) {
            return Some(tex);
        }
        let req = FrameRequest {
            time_secs: 0.0,
            max_size: self.state.render_box,
            apply_rotation: true,
        };
        let (_actual, frame) =
            match decode_frame_at_cancellable(&info.path, &req, &self.state.cancel) {
                Ok(result) => result,
                Err(error) => {
                    return self.state.fail_materialization(format!(
                        "image source {media_ref} decode failed: {error}"
                    ));
                }
            };
        let decoded = DecodedFrame::new(frame.width, frame.height, frame.rgba, false);
        let tex = upload_rgba(
            self.device,
            self.queue,
            &decoded,
            false,
            Some("playback-image"),
        );
        Some(self.state.static_cache.insert(key, tex))
    }

    /// Rasterize (once) and cache a text layer, mirroring the preview resolver's
    /// text path (premultiplied RGBA box, composited above video).
    fn resolve_text(&mut self, clip_id: &str) -> Option<Rc<GpuTexture>> {
        let key = format!("t:{clip_id}");
        if let Some(tex) = self.state.static_cache.get(&key) {
            return Some(tex);
        }
        let Some(info) = self.state.text.get(clip_id) else {
            return self
                .state
                .fail_materialization(format!("text clip {clip_id} has no raster input"));
        };
        let req = TextRasterRequest {
            clip_id,
            content: &info.content,
            style: &info.style,
            box_norm: info.box_norm,
            canvas: self.state.render_box,
        };
        // A blank text clip draws nothing; only a non-blank one that yields no
        // pixels fails playback (#180).
        let frame = match rasterize_text_layer(self.state.text_rasterizer.as_ref(), &req) {
            Ok(Some(frame)) => frame,
            Ok(None) => return None,
            Err(error) => {
                return self
                    .state
                    .fail_materialization(format!("text clip {clip_id} {error}"));
            }
        };
        let tex = upload_rgba(
            self.device,
            self.queue,
            &frame,
            false,
            Some("playback-text"),
        );
        Some(self.state.static_cache.insert(key, tex))
    }
}

impl TextureResolver for StreamingResolver<'_, '_> {
    fn resolve(&mut self, source: &TextureSource, source_frame: i64) -> Option<Rc<GpuTexture>> {
        match source {
            // Video: pre-uploaded by `sync_active`. The compositor does not pass
            // clip id here, so duplicate media/source keys share the best
            // candidate selected above (exact beats stale). A miss returns None
            // and the compositor skips the layer for this frame.
            TextureSource::Decoded { media_ref } => self
                .frame_tex
                .get(&format!("v:{media_ref}:{source_frame}"))
                .map(|candidate| candidate.texture.clone()),
            TextureSource::Image { media_ref } => self.resolve_image(media_ref),
            TextureSource::Text { clip_id } => self.resolve_text(clip_id),
            TextureSource::Lottie { media_ref } => {
                let Some(info) = self.state.media.get(media_ref) else {
                    return self.state.fail_materialization(format!(
                        "Lottie source {media_ref} is unauthorized"
                    ));
                };
                match self.state.lottie.resolve(
                    self.device,
                    self.queue,
                    &mut self.state.static_cache,
                    &info.path,
                    source_frame,
                    self.state.render_box,
                    "playback-lottie",
                ) {
                    Ok(texture) => Some(texture),
                    Err(error) => {
                        eprintln!("[playback] {error}");
                        self.state.materialization_error = Some(error);
                        None
                    }
                }
            }
        }
    }

    fn resolve_lut(
        &mut self,
        reference: &LutReference,
    ) -> Result<Option<Arc<GpuLutTexture>>, opentake_render::RenderError> {
        if let Some(cached) = self.state.lut_cache.get(&reference.id) {
            return Ok(Some(cached.clone()));
        }
        let resolved = crate::lut::resolve_project_lut(
            self.state.project_root.as_ref(),
            reference,
            self.device,
            self.queue,
            "playback-lut",
        )?;
        if let Some(texture) = &resolved {
            self.state
                .lut_cache
                .insert(reference.id.clone(), texture.clone());
        }
        Ok(resolved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentake_media::RgbaFrame;
    use opentake_render::plan::LayerDraw;
    use opentake_render::{Compositor, FramePlan, RenderDevice, RenderSize};
    use std::io::Write;

    fn lottie_fixture(first: &str, second: &str) -> String {
        format!(
            r##"{{
  "v":"5.5.2","fr":2,"ip":0,"op":2,"w":16,"h":16,"ddd":0,"assets":[],
  "layers":[
    {{"ddd":0,"ind":1,"ty":4,"nm":"first",
      "ks":{{"o":{{"a":0,"k":100}},"r":{{"a":0,"k":0}},"p":{{"a":0,"k":[8,8,0]}},
             "a":{{"a":0,"k":[8,8,0]}},"s":{{"a":0,"k":[100,100,100]}}}},
      "shapes":[
        {{"ty":"rc","d":1,"s":{{"a":0,"k":[16,16]}},"p":{{"a":0,"k":[8,8]}},"r":{{"a":0,"k":0}}}},
        {{"ty":"fl","c":{{"a":0,"k":{first}}},"o":{{"a":0,"k":100}},"r":1}}
      ],
      "ao":0,"ip":0,"op":1,"st":0,"bm":0}},
    {{"ddd":0,"ind":2,"ty":4,"nm":"second",
      "ks":{{"o":{{"a":0,"k":100}},"r":{{"a":0,"k":0}},"p":{{"a":0,"k":[8,8,0]}},
             "a":{{"a":0,"k":[8,8,0]}},"s":{{"a":0,"k":[100,100,100]}}}},
      "shapes":[
        {{"ty":"rc","d":1,"s":{{"a":0,"k":[16,16]}},"p":{{"a":0,"k":[8,8]}},"r":{{"a":0,"k":0}}}},
        {{"ty":"fl","c":{{"a":0,"k":{second}}},"o":{{"a":0,"k":100}},"r":1}}
      ],
      "ao":0,"ip":1,"op":2,"st":0,"bm":0}}
  ]
}}"##
        )
    }

    fn composite_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: Rc<GpuTexture>,
    ) -> DecodedFrame {
        struct FixedResolver(Rc<GpuTexture>);
        impl TextureResolver for FixedResolver {
            fn resolve(
                &mut self,
                _source: &TextureSource,
                _source_frame: i64,
            ) -> Option<Rc<GpuTexture>> {
                Some(self.0.clone())
            }
        }

        let source = TextureSource::Lottie {
            media_ref: "fixture".into(),
        };
        let draw = LayerDraw {
            source: &source,
            source_frame: 0,
            affine: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            nat_size: (16.0, 16.0),
            crop_uv: (0.0, 0.0, 1.0, 1.0),
            opacity: 1.0,
            clip_id: "fixture-clip",
            color_grade: None,
            lut: None,
            chroma_key: None,
            masks: &[],
            effects: &[],
        };
        let plan = FramePlan {
            clear_rgba: [0.0, 0.0, 0.0, 1.0],
            draws: vec![draw],
        };
        Compositor::new(device)
            .render_to_rgba(
                device,
                queue,
                RenderSize::new(16, 16),
                &plan,
                &mut FixedResolver(texture),
            )
            .expect("composite Lottie texture")
    }

    #[test]
    fn lottie_cache_lifecycle_frame_modulo_and_preview_export_parity() {
        let Ok(dev) = RenderDevice::try_new() else {
            return;
        };
        let temp = tempfile::tempdir().expect("temp Lottie fixture");
        let path = temp.path().join("two-frame.json");
        std::fs::write(&path, lottie_fixture("[1,0,0,1]", "[0,1,0,1]"))
            .expect("write Lottie fixture");

        let mut preview = crate::render::LottieMaterializer::new();
        let mut preview_cache = TextureCache::new(8);
        let first = preview
            .resolve(
                &dev.device,
                &dev.queue,
                &mut preview_cache,
                &path,
                0,
                (16, 16),
                "preview-lottie",
            )
            .expect("preview frame zero");
        let second = preview
            .resolve(
                &dev.device,
                &dev.queue,
                &mut preview_cache,
                &path,
                1,
                (16, 16),
                "preview-lottie",
            )
            .expect("preview frame one");
        let wrapped = preview
            .resolve(
                &dev.device,
                &dev.queue,
                &mut preview_cache,
                &path,
                2,
                (16, 16),
                "preview-lottie",
            )
            .expect("preview wrapped frame");
        assert!(Rc::ptr_eq(&first, &wrapped), "frame 2 must wrap to frame 0");
        assert!(!Rc::ptr_eq(&first, &second));

        let first_pixels = composite_texture(&dev.device, &dev.queue, first.clone());
        let second_pixels = composite_texture(&dev.device, &dev.queue, second);
        let first_center = &first_pixels.rgba[(8 * 16 + 8) * 4..][..4];
        let second_center = &second_pixels.rgba[(8 * 16 + 8) * 4..][..4];
        assert!(
            first_center[0] > 200 && first_center[1] < 30,
            "{first_center:?}"
        );
        assert!(
            second_center[1] > 200 && second_center[0] < 30,
            "{second_center:?}"
        );

        // Export owns an independent materializer/cache on the same source, but
        // must produce byte-identical pixels for the same internal frame.
        let mut export = crate::render::LottieMaterializer::new();
        let mut export_cache = TextureCache::new(8);
        let export_first = export
            .resolve(
                &dev.device,
                &dev.queue,
                &mut export_cache,
                &path,
                0,
                (16, 16),
                "export-lottie",
            )
            .expect("export frame zero");
        assert_eq!(
            first_pixels,
            composite_texture(&dev.device, &dev.queue, export_first)
        );

        // Recreating the device-owned lifecycle drops GPU handles while
        // retaining deterministic output on the next materialization.
        let mut rebuilt = crate::render::LottieMaterializer::new();
        let mut rebuilt_cache = TextureCache::new(8);
        let rebuilt_first = rebuilt
            .resolve(
                &dev.device,
                &dev.queue,
                &mut rebuilt_cache,
                &path,
                0,
                (16, 16),
                "rebuilt-lottie",
            )
            .expect("rebuilt frame zero");
        assert!(!Rc::ptr_eq(&first, &rebuilt_first));
        assert_eq!(
            first_pixels,
            composite_texture(&dev.device, &dev.queue, rebuilt_first)
        );

        // Same path, changed bytes: the content hash must invalidate the old
        // texture rather than reusing a stale media-ref-only key.
        std::fs::write(&path, lottie_fixture("[0,0,1,1]", "[1,1,0,1]"))
            .expect("replace Lottie fixture");
        let changed = preview
            .resolve(
                &dev.device,
                &dev.queue,
                &mut preview_cache,
                &path,
                0,
                (16, 16),
                "preview-lottie",
            )
            .expect("changed frame zero");
        assert!(!Rc::ptr_eq(&first, &changed));
        let changed_pixels = composite_texture(&dev.device, &dev.queue, changed);
        let changed_center = &changed_pixels.rgba[(8 * 16 + 8) * 4..][..4];
        assert!(
            changed_center[2] > 200 && changed_center[0] < 30,
            "{changed_center:?}"
        );

        // Invalid documents must become a typed frame failure owned by the
        // playback loop, never a successful frame with a silently missing layer.
        let invalid_path = temp.path().join("invalid-lottie.json");
        std::fs::write(&invalid_path, b"{not valid lottie}").expect("write invalid fixture");
        let mut media = HashMap::new();
        media.insert("invalid".into(), MediaInfo { path: invalid_path });
        let mut state = PlaybackResolverState::new(
            media,
            HashMap::new(),
            30,
            (16, 16),
            MediaCancelToken::new(),
        );
        let invalid_source = TextureSource::Lottie {
            media_ref: "invalid".into(),
        };
        let mut resolver = StreamingResolver::new(&dev.device, &dev.queue, &mut state);
        assert!(resolver.resolve(&invalid_source, 0).is_none());
        drop(resolver);
        let error = state
            .take_materialization_error()
            .expect("playback must retain the materialization failure");
        assert!(error.contains("parse Lottie document"), "{error}");
    }

    #[test]
    fn lottie_container_preview_and_export_materialize_identical_pixels() {
        let Ok(dev) = RenderDevice::try_new() else {
            return;
        };
        let temp = tempfile::tempdir().expect("temp Lottie container fixture");
        let path = temp.path().join("container.lottie");
        let file = std::fs::File::create(&path).expect("create Lottie container");
        let mut archive = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        archive
            .start_file("animations/main.json", options)
            .expect("create Lottie animation entry");
        archive
            .write_all(lottie_fixture("[1,0,0,1]", "[0,1,0,1]").as_bytes())
            .expect("write Lottie animation entry");
        archive.finish().expect("finish Lottie container");

        let mut preview = crate::render::LottieMaterializer::new();
        let mut preview_cache = TextureCache::new(4);
        let preview_frame = preview
            .resolve(
                &dev.device,
                &dev.queue,
                &mut preview_cache,
                &path,
                0,
                (16, 16),
                "preview-lottie-container",
            )
            .expect("materialize Lottie container for preview");

        let mut export = crate::render::LottieMaterializer::new();
        let mut export_cache = TextureCache::new(4);
        let export_frame = export
            .resolve(
                &dev.device,
                &dev.queue,
                &mut export_cache,
                &path,
                0,
                (16, 16),
                "export-lottie-container",
            )
            .expect("materialize Lottie container for export");

        let preview_pixels = composite_texture(&dev.device, &dev.queue, preview_frame);
        let export_pixels = composite_texture(&dev.device, &dev.queue, export_frame);
        assert_eq!(preview_pixels, export_pixels);
        let center = &preview_pixels.rgba[(8 * 16 + 8) * 4..][..4];
        assert!(center[0] > 200 && center[1] < 30, "{center:?}");
    }

    #[test]
    fn unchanged_image_is_hashed_once_across_playback_frames() {
        let Ok(dev) = RenderDevice::try_new() else {
            return;
        };
        if !opentake_media::ffmpeg_status::ffmpeg_available() {
            return;
        }
        let temp = tempfile::tempdir().expect("image fixture");
        let path = temp.path().join("overlay.png");
        image::RgbaImage::from_pixel(4, 4, image::Rgba([255, 0, 0, 255]))
            .save(&path)
            .expect("write first image");
        let mut media = HashMap::new();
        media.insert("overlay".into(), MediaInfo { path: path.clone() });
        let mut state = PlaybackResolverState::new(
            media,
            HashMap::new(),
            30,
            (16, 16),
            MediaCancelToken::new(),
        );
        let source = TextureSource::Image {
            media_ref: "overlay".into(),
        };
        let mut resolver = StreamingResolver::new(&dev.device, &dev.queue, &mut state);
        let first = resolver.resolve(&source, 0).expect("first image texture");
        for frame in 1..1000 {
            let again = resolver.resolve(&source, frame).expect("cached image");
            assert!(Rc::ptr_eq(&first, &again));
        }
        assert_eq!(resolver.state.content_hashes.hashes(), 1);

        image::RgbaImage::from_pixel(8, 8, image::Rgba([0, 0, 255, 255]))
            .save(&path)
            .expect("replace image");
        let replaced = resolver.resolve(&source, 1000).expect("replaced image");
        assert!(!Rc::ptr_eq(&first, &replaced));
        assert_eq!(resolver.state.content_hashes.hashes(), 2);
        drop(resolver);
        assert!(state.take_materialization_error().is_none());
    }

    #[test]
    fn lottie_document_is_read_once_until_replaced() {
        let Ok(dev) = RenderDevice::try_new() else {
            return;
        };
        let temp = tempfile::tempdir().expect("Lottie fixture");
        let path = temp.path().join("overlay.json");
        std::fs::write(&path, lottie_fixture("[1,0,0,1]", "[0,0,1,1]")).expect("write Lottie");
        let mut media = HashMap::new();
        media.insert("overlay".into(), MediaInfo { path: path.clone() });
        let mut state = PlaybackResolverState::new(
            media,
            HashMap::new(),
            30,
            (16, 16),
            MediaCancelToken::new(),
        );
        let source = TextureSource::Lottie {
            media_ref: "overlay".into(),
        };
        let mut resolver = StreamingResolver::new(&dev.device, &dev.queue, &mut state);
        for frame in 0..1000 {
            resolver.resolve(&source, frame).expect("Lottie frame");
        }
        assert_eq!(resolver.state.lottie.document_reads(), 1);

        // Different length, so the identity changes even on coarse mtimes.
        std::fs::write(&path, lottie_fixture("[0,1,0,1]", "[0,1,0,1.0]")).expect("replace Lottie");
        let replaced = resolver.resolve(&source, 0).expect("replaced Lottie frame");
        assert_eq!(resolver.state.lottie.document_reads(), 2);
        let pixels = composite_texture(&dev.device, &dev.queue, replaced);
        let center = &pixels.rgba[(8 * 16 + 8) * 4..][..4];
        assert!(center[1] > 200 && center[0] < 30, "{center:?}");
        drop(resolver);
        assert!(state.take_materialization_error().is_none());
    }

    #[test]
    fn missing_image_retains_a_materialization_error_instead_of_becoming_a_skipped_layer() {
        let Ok(dev) = RenderDevice::try_new() else {
            return;
        };
        let temp = tempfile::tempdir().expect("missing image fixture");
        let missing_path = temp.path().join("gone.png");
        let mut media = HashMap::new();
        media.insert("missing-image".into(), MediaInfo { path: missing_path });
        let mut state = PlaybackResolverState::new(
            media,
            HashMap::new(),
            30,
            (16, 16),
            MediaCancelToken::new(),
        );
        let source = TextureSource::Image {
            media_ref: "missing-image".into(),
        };
        let mut resolver = StreamingResolver::new(&dev.device, &dev.queue, &mut state);

        assert!(resolver.resolve(&source, 0).is_none());
        drop(resolver);
        let error = state
            .take_materialization_error()
            .expect("missing image must retain an explicit materialization failure");
        assert!(error.contains("image source missing-image"), "{error}");
    }

    #[test]
    fn missing_text_retains_a_materialization_error_instead_of_becoming_a_skipped_layer() {
        let Ok(dev) = RenderDevice::try_new() else {
            return;
        };
        let mut state = PlaybackResolverState::new(
            HashMap::new(),
            HashMap::new(),
            30,
            (16, 16),
            MediaCancelToken::new(),
        );
        let source = TextureSource::Text {
            clip_id: "missing-text".into(),
        };
        let mut resolver = StreamingResolver::new(&dev.device, &dev.queue, &mut state);

        assert!(resolver.resolve(&source, 0).is_none());
        drop(resolver);
        let error = state
            .take_materialization_error()
            .expect("missing text must retain an explicit materialization failure");
        assert!(error.contains("text clip missing-text"), "{error}");
    }

    #[test]
    fn blank_text_draws_nothing_while_a_missing_raster_still_fails_playback() {
        let Ok(dev) = RenderDevice::try_new() else {
            assert!(
                std::env::var_os("OPENTAKE_REQUIRE_GPU").is_none(),
                "playback resolver qualification requires a GPU adapter"
            );
            eprintln!("skip: no GPU adapter available");
            return;
        };
        let text_info = |content: &str, box_norm| TextInfo {
            content: content.to_string(),
            style: opentake_domain::TextStyle::default(),
            box_norm,
        };
        let full = (0.0, 0.0, 1.0, 1.0);
        let text = HashMap::from([
            ("empty".to_string(), text_info("", full)),
            (
                "flat".to_string(),
                text_info("hidden", (0.0, 0.0, 1.0, 0.0)),
            ),
            ("spaces".to_string(), text_info(" \t ", full)),
            ("visible".to_string(), text_info("visible", full)),
        ]);
        let mut state =
            PlaybackResolverState::new(HashMap::new(), text, 30, (64, 64), MediaCancelToken::new());
        // Returns `None` for every request, like a broken text backend.
        state.set_text_rasterizer(Box::new(opentake_render::NullTextRasterizer));
        let source = |clip_id: &str| TextureSource::Text {
            clip_id: clip_id.to_string(),
        };
        let mut resolver = StreamingResolver::new(&dev.device, &dev.queue, &mut state);
        for blank in ["empty", "flat"] {
            assert!(resolver.resolve(&source(blank), 0).is_none());
        }
        drop(resolver);
        assert_eq!(state.take_materialization_error(), None);

        // Whitespace still paints its box, so it reaches the rasterizer.
        for drawn in ["spaces", "visible"] {
            let mut resolver = StreamingResolver::new(&dev.device, &dev.queue, &mut state);
            assert!(resolver.resolve(&source(drawn), 0).is_none());
            drop(resolver);
            let error = state
                .take_materialization_error()
                .expect("a missing raster for drawn text must fail playback");
            assert!(
                error.contains(&format!("text clip {drawn} rasterization failed")),
                "{error}"
            );
        }
    }

    #[test]
    fn bootstrap_request_targets_exact_source_frame() {
        let request = bootstrap_frame_request(17, 25, (640, 360));

        assert_eq!(request.time_secs, 17.0 / 25.0);
        assert_eq!(request.max_size, (640, 360));
        assert!(request.apply_rotation);
    }

    fn vf(source_frame: i64) -> StreamVideoFrame {
        StreamVideoFrame {
            source_frame,
            pts_secs: source_frame as f64 / 30.0,
            frame: RgbaFrame::new(1, 1, vec![0, 0, 0, 255]),
        }
    }

    /// A `pull` closure draining a fixed queue in order.
    fn queue_pull(
        frames: Vec<StreamVideoFrame>,
    ) -> impl FnMut() -> Result<Option<StreamVideoFrame>, ()> {
        let mut it = frames.into_iter();
        move || Ok(it.next())
    }

    #[test]
    fn drain_exact_hit_returns_target_frame() {
        let mut pending = None;
        let got = drain_to_target(&mut pending, queue_pull(vec![vf(5)]), 5).unwrap();
        assert_eq!(got.map(|f| f.source_frame), Some(5));
        assert!(pending.is_none());
    }

    #[test]
    fn drain_discards_frames_behind_target() {
        let mut pending = None;
        // Normal forward advance: 3 and 4 are stale, 5 is the target.
        let got = drain_to_target(&mut pending, queue_pull(vec![vf(3), vf(4), vf(5)]), 5).unwrap();
        assert_eq!(got.map(|f| f.source_frame), Some(5));
        assert!(pending.is_none());
    }

    #[test]
    fn drain_stashes_ahead_frame_and_reuses_cache() {
        let mut pending = None;
        // Only a future frame is available (slow-mo / dup): reuse cache now, keep 7.
        let got = drain_to_target(&mut pending, queue_pull(vec![vf(7)]), 5).unwrap();
        assert!(got.is_none());
        assert_eq!(pending.as_ref().map(|f| f.source_frame), Some(7));
    }

    #[test]
    fn drain_consumes_pending_when_target_catches_up() {
        let mut pending = Some(vf(7));
        // Queue empty; target now equals the stashed frame -> use it.
        let got = drain_to_target(&mut pending, queue_pull(vec![]), 7).unwrap();
        assert_eq!(got.map(|f| f.source_frame), Some(7));
        assert!(pending.is_none());
    }

    #[test]
    fn drain_keeps_pending_while_still_ahead() {
        let mut pending = Some(vf(8));
        let got = drain_to_target(&mut pending, queue_pull(vec![]), 5).unwrap();
        assert!(got.is_none());
        assert_eq!(pending.as_ref().map(|f| f.source_frame), Some(8));
    }

    #[test]
    fn drain_drops_stale_pending_then_pulls_target() {
        let mut pending = Some(vf(2));
        let got = drain_to_target(&mut pending, queue_pull(vec![vf(5)]), 5).unwrap();
        assert_eq!(got.map(|f| f.source_frame), Some(5));
        assert!(pending.is_none());
    }

    #[test]
    fn drain_empty_queue_reuses_cache() {
        let mut pending = None;
        let got = drain_to_target(&mut pending, queue_pull(vec![]), 5).unwrap();
        assert!(got.is_none());
        assert!(pending.is_none());
    }

    #[test]
    fn one_clip_uses_one_decoder_source() {
        let mut streams = HashMap::new();
        let mut decoder_invocations = 0;
        let stream = ensure_stream_with(&mut streams, "clip-1", || {
            decoder_invocations += 1;
            Ok::<_, ()>("decoder-1")
        })
        .expect("create cold decoder");
        assert_eq!(*stream, "decoder-1");

        let same = ensure_stream_with(&mut streams, "clip-1", || {
            decoder_invocations += 1;
            Ok::<_, ()>("duplicate-decoder")
        })
        .expect("reuse decoder");
        assert_eq!(*same, "decoder-1");
        assert_eq!(
            decoder_invocations, 1,
            "one clip must have one cold decoder source"
        );
    }

    #[test]
    fn exact_frame_texture_wins_same_key_collision_in_either_draw_order() {
        assert!(should_replace_frame_texture(false, true));
        assert!(!should_replace_frame_texture(true, false));
        assert!(!should_replace_frame_texture(true, true));
        assert!(!should_replace_frame_texture(false, false));
    }

    #[test]
    fn drain_propagates_stream_failure_instead_of_freezing_cache() {
        let mut pending = None;
        let error = drain_to_target(
            &mut pending,
            || Err::<Option<StreamVideoFrame>, _>("decoder failed"),
            9,
        )
        .expect_err("continuous decoder failure must propagate");

        assert_eq!(error, "decoder failed");
    }

    struct FakeSupply {
        rx: std::sync::mpsc::Receiver<Result<StreamVideoFrame, MediaError>>,
        stopped: Rc<std::cell::Cell<usize>>,
    }

    impl FrameSupply for FakeSupply {
        fn try_pull(&self) -> Result<Result<StreamVideoFrame, MediaError>, TryRecvError> {
            self.rx.try_recv()
        }

        fn request_stop(&self) {
            self.stopped.set(self.stopped.get() + 1);
        }
    }

    /// A window supply that already decoded `[start, end)` (instant decoder).
    fn finished_window(start: i64, end: i64, stopped: &Rc<std::cell::Cell<usize>>) -> FakeSupply {
        let (tx, rx) = std::sync::mpsc::channel();
        for frame in start..end {
            tx.send(Ok(vf(frame))).expect("queue window frame");
        }
        FakeSupply {
            rx,
            stopped: stopped.clone(),
        }
    }

    #[test]
    fn reverse_prefetch_uses_the_full_buffered_window_as_decode_lead() {
        let stopped = Rc::new(std::cell::Cell::new(0));
        let mut spawned = Vec::new();
        let mut windows = ReverseWindows::new(63);
        for target in [63, 62] {
            windows
                .advance(target, 63, |start, end| {
                    spawned.push((start, end));
                    Ok(finished_window(start, end, &stopped))
                })
                .unwrap();
        }
        assert_eq!(spawned, [(47, 63), (31, 47)]);
        assert!(windows.frames.len() < REVERSE_WINDOW_FRAMES as usize);
        assert_eq!(stopped.get(), 0);
    }

    #[test]
    fn reverse_windows_serve_sixty_descending_frames_with_bounded_spawns() {
        let stopped = Rc::new(std::cell::Cell::new(0));
        let mut spawned = Vec::new();
        let mut windows = ReverseWindows::new(59);
        let mut cached = 59;
        for target in (0..60).rev() {
            let hit = windows
                .advance(target, cached, |start, end| {
                    spawned.push((start, end));
                    Ok(finished_window(start, end, &stopped))
                })
                .expect("advance reverse windows");
            if target < 59 {
                assert_eq!(
                    hit.as_ref().map(|frame| frame.source_frame),
                    Some(target),
                    "prefetched window must hold source frame {target}"
                );
            }
            if let Some(frame) = hit {
                cached = frame.source_frame;
            }
        }

        let windows_played = 60_usize.div_ceil(REVERSE_WINDOW_FRAMES as usize);
        assert!(
            spawned.len() <= windows_played + 2,
            "60 reversed frames spawned {} windows: {spawned:?}",
            spawned.len()
        );
        assert_eq!(spawned, vec![(43, 59), (27, 43), (11, 27), (0, 11)]);
        assert_eq!(stopped.get(), 0, "no window is abandoned while in step");
    }

    #[test]
    fn lagging_reverse_decode_restarts_once_per_window_not_per_frame() {
        let stopped = Rc::new(std::cell::Cell::new(0));
        let mut spawned = 0;
        // Senders stay alive and never deliver: every window lags forever.
        let mut senders = Vec::new();
        let mut windows = ReverseWindows::new(59);
        for target in (0..60).rev() {
            let hit = windows
                .advance(target, 59, |_, _| {
                    spawned += 1;
                    let (tx, rx) = std::sync::mpsc::channel();
                    senders.push(tx);
                    Ok(FakeSupply {
                        rx,
                        stopped: stopped.clone(),
                    })
                })
                .expect("advance lagging reverse windows");
            assert!(hit.is_none());
        }

        let windows_played = 60_usize.div_ceil(REVERSE_WINDOW_FRAMES as usize);
        assert!(
            spawned <= windows_played + 2,
            "lagging decode spawned {spawned} windows over 60 frames"
        );
        assert_eq!(stopped.get(), spawned - 1, "each superseded window stops");
    }

    #[test]
    fn reverse_windows_restart_at_a_target_above_the_chain() {
        let stopped = Rc::new(std::cell::Cell::new(0));
        let mut spawned = Vec::new();
        let mut windows = ReverseWindows::new(20);
        let mut spawn = |start, end| {
            spawned.push((start, end));
            Ok(finished_window(start, end, &stopped))
        };
        windows.advance(20, 20, &mut spawn).expect("bootstrap");
        windows.advance(19, 20, &mut spawn).expect("descend");
        let hit = windows.advance(40, 19, &mut spawn).expect("jump up");

        assert!(hit.is_none());
        assert_eq!(spawned.last(), Some(&(25, 41)));
    }

    #[test]
    fn reverse_window_decode_failure_propagates() {
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(Err(MediaError::Decode("broken window".to_string())))
            .expect("queue failure");
        let mut supply = Some(FakeSupply {
            rx,
            stopped: Rc::new(std::cell::Cell::new(0)),
        });
        let mut windows = ReverseWindows::new(10);
        windows
            .advance(10, 10, |_, _| Ok(supply.take().expect("one window")))
            .expect("first advance spawns the failing window");
        let error = windows
            .advance(9, 10, |_, _| panic!("no new window while one is in flight"))
            .expect_err("reverse decode failure must propagate");
        assert!(error.contains("broken window"), "{error}");
    }

    #[test]
    fn stream_pull_surfaces_worker_error_and_premature_disconnect() {
        let decode_error = classify_stream_pull(
            Ok(Err(MediaError::Decode("broken stream".to_string()))),
            4,
            5,
        )
        .expect_err("worker error must propagate");
        assert!(decode_error.contains("broken stream"));

        let disconnect_error = classify_stream_pull(Err(TryRecvError::Disconnected), 4, 5)
            .expect_err("disconnect before the target must propagate");
        assert!(disconnect_error.contains("ended at source frame 4"));

        assert!(classify_stream_pull(Err(TryRecvError::Disconnected), 5, 5)
            .expect("normal EOF may retain the exact final frame")
            .is_none());
        assert!(classify_stream_pull(Err(TryRecvError::Empty), 4, 5)
            .expect("an empty live queue may temporarily reuse cache")
            .is_none());
    }

    #[test]
    fn failed_bootstrap_runs_once_until_a_seek_clears_the_negative_cache() {
        let mut streams: HashMap<String, u32> = HashMap::new();
        let mut failures = HashMap::new();
        let mut factory_calls = 0;
        for _ in 0..2 {
            let error = ensure_stream_or_cached_failure(
                &mut streams,
                &mut failures,
                "clip-1",
                "asset-1",
                || {
                    factory_calls += 1;
                    Err::<u32, _>("playback bootstrap decode failed for asset-1".to_string())
                },
            )
            .expect_err("bootstrap fails");
            assert!(error.contains("asset-1"));
        }
        assert_eq!(
            factory_calls, 1,
            "a failed bootstrap must not respawn per tick"
        );

        // Another clip, or the same clip bound to different media, still builds.
        assert_eq!(
            *ensure_stream_or_cached_failure(
                &mut streams,
                &mut failures,
                "clip-1",
                "asset-2",
                || Ok(7)
            )
            .expect("replacement media builds"),
            7
        );

        // A seek (`clear_streams`) clears the negative cache and retries.
        streams.clear();
        failures.clear();
        ensure_stream_or_cached_failure(&mut streams, &mut failures, "clip-1", "asset-1", || {
            factory_calls += 1;
            Ok(1)
        })
        .expect("retried after seek");
        assert_eq!(factory_calls, 2);
    }

    #[test]
    fn seek_clears_recorded_stream_failures() {
        let mut state = PlaybackResolverState::new(
            HashMap::new(),
            HashMap::new(),
            30,
            (64, 36),
            MediaCancelToken::new(),
        );
        state.failed_streams.insert(
            "clip-1".to_string(),
            StreamFailure {
                media_ref: "asset-1".to_string(),
                message: "broken".to_string(),
            },
        );
        state.reset_streams(["clip-1"]);
        assert_eq!(state.failed_streams.len(), 1, "a rewind keeps the failure");
        state.clear_streams();
        assert!(state.failed_streams.is_empty());
    }

    #[test]
    fn user_visible_errors_name_files_not_absolute_paths() {
        let path = std::env::temp_dir().join("private-dir").join("take-3.mov");
        let state = PlaybackResolverState::new(
            HashMap::from([("asset-1".to_string(), MediaInfo { path: path.clone() })]),
            HashMap::new(),
            30,
            (64, 36),
            MediaCancelToken::new(),
        );
        let message = format!("ffmpeg: {}: Invalid data found", path.display());
        let redacted = state.redact_media_paths(&message);
        assert_eq!(redacted, "ffmpeg: take-3.mov: Invalid data found");
    }
}
