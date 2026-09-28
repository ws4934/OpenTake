//! Clip audio rendering shared by preview playback and export (#3, #16).
//!
//! A clip's audio is a pure function of its clip-relative position at the mix
//! rate: frame `k` of the clip reads its decoded source window at `k * ratio`
//! source frames (linear interpolation, `ratio` = consumed / timeline
//! frames), and a denoised clip is filtered with one noise profile estimated
//! from its whole source window. Nothing depends on where a mix window or a
//! playback session starts, so preview and export produce the same samples for
//! the same timeline position.
//!
//! [`ClipAudioReader`] serves consecutive clip frames from one forward
//! [`PcmStream`]. Export and preview playback keep one reader per audible clip
//! for as long as the clip plays (one decoder per clip, not per window), up to
//! [`MAX_OPEN_CLIP_READERS`] at a time. A reader opened mid-clip (after a
//! seek) starts a denoise warm-up before its first frame, which converges on
//! the uninterrupted result.
//!
//! Noise profiles are computed once per clip layout and cached. Export waits
//! for them; preview asks a background worker and plays the clip undenoised
//! until its profile is ready, so a profile pass never delays playback.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
#[cfg(feature = "playback-engine")]
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, SystemTime};

use opentake_domain::{AudioDenoise, Clip};
use opentake_media::analysis::{
    denoise_stream_start, denoise_warmup_frames, DenoiseError, DenoiseProfile,
    DenoiseProfileBuilder, DenoiseStream,
};
use opentake_media::{MediaCancelToken, MediaError, PcmFormat, PcmSpec, PcmStream};
use opentake_render::AudioClipPlan;

/// One audible clip of a mix and the gain chain around it: a top-level clip,
/// or a leaf of a nested sequence flattened by the render plan
/// ([`AudioClipPlan`]), whose volume multiplies its compound ancestors', whose
/// true-peak ceiling is the strictest along the chain and whose denoise is the
/// nearest setting. Export and preview mix through this one definition, so a
/// compound clip sounds the same in both.
pub(crate) trait AudioPlanLike {
    fn clip(&self) -> &Clip;
    fn volume_at(&self, frame: i32) -> f64;
    fn true_peak_ceiling_dbtp(&self) -> Option<f64>;
    fn audio_denoise(&self) -> Option<AudioDenoise>;
}

impl AudioPlanLike for Clip {
    fn clip(&self) -> &Clip {
        self
    }

    fn volume_at(&self, frame: i32) -> f64 {
        Clip::volume_at(self, frame)
    }

    fn true_peak_ceiling_dbtp(&self) -> Option<f64> {
        self.loudness_normalization
            .map(|normalization| normalization.true_peak_ceiling_dbtp)
    }

    fn audio_denoise(&self) -> Option<AudioDenoise> {
        self.audio_denoise
    }
}

impl AudioPlanLike for AudioClipPlan {
    fn clip(&self) -> &Clip {
        &self.clip
    }

    fn volume_at(&self, frame: i32) -> f64 {
        AudioClipPlan::volume_at(self, frame)
    }

    fn true_peak_ceiling_dbtp(&self) -> Option<f64> {
        std::iter::once(&self.gain_clip)
            .chain(self.compound_ancestors.iter())
            .filter_map(|clip| {
                clip.loudness_normalization
                    .map(|normalization| normalization.true_peak_ceiling_dbtp)
            })
            .min_by(f64::total_cmp)
    }

    fn audio_denoise(&self) -> Option<AudioDenoise> {
        std::iter::once(&self.gain_clip)
            .chain(self.compound_ancestors.iter())
            .find_map(|clip| clip.audio_denoise)
    }
}

/// Source frames pulled from the decoder per read.
const SOURCE_READ_FRAMES: usize = 8 * 1024;
/// Clip frames resampled (and fed to the denoiser) per step.
const RENDER_CHUNK_FRAMES: usize = 4 * 1024;
/// Distinct clip noise profiles remembered across preview windows, playback
/// sessions and exports (a few KB each), least recently used evicted first.
const PROFILE_CACHE_CAPACITY: usize = 256;
/// Background profile passes waiting at once; the oldest is dropped beyond
/// this (a later window asks again if its clip still plays).
#[cfg(feature = "playback-engine")]
const PROFILE_QUEUE_CAPACITY: usize = 64;
/// How often a caller waiting for another caller's profile pass checks its
/// own cancellation.
const PROFILE_WAIT_POLL: Duration = Duration::from_millis(50);
/// Clip decoders one mix has open at once. Each is an FFmpeg process with
/// three pipes. A mix keeps at most one fewer open across windows (see
/// [`keep_clip_reader`]); further clips are decoded by a reader opened for one
/// window and closed right after it is read, which keeps a dense timeline well
/// inside the default open-file limit (256 on macOS).
pub(crate) const MAX_OPEN_CLIP_READERS: usize = 16;

/// Whether a mix that already keeps `kept` readers open across windows may
/// keep one more. The last slot under [`MAX_OPEN_CLIP_READERS`] stays free
/// for the per-window reader of a clip beyond the cap.
pub(crate) fn keep_clip_reader(kept: usize) -> bool {
    kept + 1 < MAX_OPEN_CLIP_READERS
}
/// Clip frames a reader opened mid-clip decodes and discards before its first
/// frame. A decode that starts at a seek point differs from a continuous one
/// for its first few milliseconds (codec priming such as AAC's overlapped
/// first frame, and the resampler's filter start); the pre-roll moves that
/// difference out of the frames the reader returns. A denoised reader starts
/// at least this far back too (its warm-up is shorter below 32 kHz), aligned
/// down to the denoiser's hop grid.
const DECODE_PREROLL_FRAMES: usize = 4_096;

/// Timeline mix frame at the start of timeline frame `frame` (rounded, as the
/// audio clock seeks). Negative before the timeline starts.
fn mix_frame_at(frame: i32, timeline_fps: i32, rate: u32) -> i64 {
    ((frame as f64 / timeline_fps as f64) * rate as f64).round() as i64
}

/// Timeline frame containing mix frame `position`, computed exactly so a gain
/// envelope switches frames on the same sample in every mixer.
pub(crate) fn timeline_frame_at(position: u64, timeline_fps: i32, rate: u32) -> i32 {
    let frame =
        u128::from(position) * u128::from(timeline_fps.max(1) as u32) / u128::from(rate.max(1));
    i32::try_from(frame).unwrap_or(i32::MAX)
}

/// Source-media window `[lo, hi)` seconds a clip consumes (trim + speed).
pub(crate) fn clip_source_window_secs(clip: &Clip, timeline_fps: i32) -> Option<(f64, f64)> {
    if clip.duration_frames <= 0 || timeline_fps <= 0 {
        return None;
    }
    let fps = timeline_fps as f64;
    let lo = clip.trim_start_frame.max(0) as f64 / fps;
    let consumed = clip.source_frames_consumed().max(0);
    if consumed == 0 {
        return None;
    }
    Some((lo, lo + consumed as f64 / fps))
}

/// Where a clip's audio sits on the timeline and in its source, at one mix rate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ClipAudioLayout {
    /// Timeline mix frame of the clip's first frame.
    start: i64,
    /// Clip length in mix frames.
    len: usize,
    source_lo: f64,
    source_hi: f64,
    /// Source frames (at the mix rate) consumed per clip frame.
    ratio: f64,
    rate: u32,
}

impl ClipAudioLayout {
    /// `None` when the clip contributes no audio frames.
    pub(crate) fn new(clip: &Clip, timeline_fps: i32, rate: u32) -> Option<Self> {
        if clip.duration_frames <= 0 || timeline_fps <= 0 || rate == 0 {
            return None;
        }
        let (source_lo, source_hi) = clip_source_window_secs(clip, timeline_fps)?;
        let start = mix_frame_at(clip.start_frame, timeline_fps, rate);
        let end = mix_frame_at(
            clip.start_frame.saturating_add(clip.duration_frames),
            timeline_fps,
            rate,
        );
        let len = usize::try_from(end - start).ok().filter(|len| *len > 0)?;
        Some(ClipAudioLayout {
            start,
            len,
            source_lo,
            source_hi,
            ratio: f64::from(clip.source_frames_consumed()) / f64::from(clip.duration_frames),
            rate,
        })
    }

    /// Timeline mix frames `[start, end)` the clip covers, from frame 0 on.
    pub(crate) fn span(&self) -> (u64, u64) {
        let end = self.start + self.len as i64;
        (self.start.max(0) as u64, end.max(0) as u64)
    }

    /// Clip-relative frame of timeline mix frame `position` (inside the span).
    pub(crate) fn offset_of(&self, position: u64) -> usize {
        (position as i64 - self.start) as usize
    }

    /// Clip length in mix frames.
    pub(crate) fn len(&self) -> usize {
        self.len
    }
}

fn denoise_error(error: DenoiseError) -> MediaError {
    match error {
        DenoiseError::Cancelled => MediaError::Cancelled,
        other => MediaError::Decode(format!("audio denoise failed: {other}")),
    }
}

/// Whether `path` has an audio track. Only regular files are probed; a pipe or
/// other special file goes straight to the decoder, which reports failures.
pub(crate) fn source_has_audio(path: &Path, cancel: &MediaCancelToken) -> Result<bool, MediaError> {
    if !path.is_file() {
        return Ok(true);
    }
    Ok(opentake_media::probe::probe_cancellable(path, cancel)?.has_audio)
}

/// Consecutive frames of one clip's audio, resampled onto the clip's frame
/// grid and optionally denoised, decoded by one forward [`PcmStream`].
pub(crate) struct ClipAudioReader {
    layout: ClipAudioLayout,
    channels: usize,
    source: Option<PcmStream>,
    /// Decoded source frames (interleaved) from source frame `source_start`.
    source_buffer: VecDeque<f32>,
    source_start: usize,
    /// No more source frames will arrive; later frames read as silence.
    source_done: bool,
    /// Next clip frame to resample.
    raw_position: usize,
    denoise: Option<DenoiseStream>,
    /// Denoised frames already emitted by the denoiser, not yet read.
    denoised: VecDeque<f32>,
    /// Next clip frame [`ClipAudioReader::read`] returns.
    position: usize,
    cancel: MediaCancelToken,
    #[cfg(test)]
    _census: Option<reader_census::Entry>,
}

impl ClipAudioReader {
    /// Serve clip frames from `from` on. A reader opened mid-clip starts
    /// decoding a pre-roll (a denoised one a denoise warm-up) before `from`,
    /// so its output matches a reader that started at the clip's first frame.
    pub(crate) fn open(
        layout: ClipAudioLayout,
        path: &Path,
        channels: usize,
        from: usize,
        denoise: Option<(DenoiseProfile, AudioDenoise)>,
        cancel: &MediaCancelToken,
    ) -> Result<Self, MediaError> {
        Self::open_with_preroll(
            layout,
            path,
            channels,
            from,
            denoise,
            DECODE_PREROLL_FRAMES,
            cancel,
        )
    }

    fn open_with_preroll(
        layout: ClipAudioLayout,
        path: &Path,
        channels: usize,
        from: usize,
        denoise: Option<(DenoiseProfile, AudioDenoise)>,
        preroll: usize,
        cancel: &MediaCancelToken,
    ) -> Result<Self, MediaError> {
        if from > layout.len || channels == 0 || channels > usize::from(u16::MAX) {
            return Err(MediaError::Decode(format!(
                "clip audio read at {from} of {} frames x {channels} channels",
                layout.len
            )));
        }
        let start = if denoise.is_some() {
            // The denoiser must start on its hop grid. Below 32 kHz its
            // warm-up is shorter than the pre-roll, so reach back by the
            // longer of the two and align down to the grid from there.
            let extra = preroll.saturating_sub(denoise_warmup_frames(layout.rate));
            denoise_stream_start(layout.rate, from.saturating_sub(extra))
        } else {
            from.saturating_sub(preroll)
        };
        let source_start = (start as f64 * layout.ratio).floor() as usize;
        let source_from = layout.source_lo + source_start as f64 / f64::from(layout.rate);
        let spec = PcmSpec {
            sample_rate: layout.rate,
            channels: channels as u16,
            format: PcmFormat::F32,
        };
        let source = if source_from < layout.source_hi {
            Some(PcmStream::open(
                path,
                &spec,
                (source_from, layout.source_hi),
                cancel,
            )?)
        } else {
            None
        };
        let denoise = match denoise {
            Some((profile, config)) => {
                Some(DenoiseStream::new(profile, config, start).map_err(denoise_error)?)
            }
            None => None,
        };
        let mut reader = ClipAudioReader {
            layout,
            channels,
            source_done: source.is_none(),
            source,
            source_buffer: VecDeque::new(),
            source_start,
            raw_position: start,
            denoise,
            denoised: VecDeque::new(),
            position: start,
            cancel: cancel.clone(),
            #[cfg(test)]
            _census: reader_census::enter(),
        };
        if start < from {
            // Pre-roll and denoise warm-up: frames before `from` only settle
            // the decoder, the resampler and the denoiser.
            let mut discard = Vec::new();
            reader.read(from - start, &mut discard)?;
        }
        Ok(reader)
    }

    /// The next clip frame [`ClipAudioReader::read`] returns.
    #[cfg(feature = "playback-engine")]
    pub(crate) fn position(&self) -> usize {
        self.position
    }

    /// Whether this reader denoises its output.
    #[cfg(feature = "playback-engine")]
    pub(crate) fn is_denoised(&self) -> bool {
        self.denoise.is_some()
    }

    /// Append the next `frames` clip frames (interleaved) to `out`.
    pub(crate) fn read(&mut self, frames: usize, out: &mut Vec<f32>) -> Result<(), MediaError> {
        if frames > self.layout.len - self.position {
            return Err(MediaError::Decode(format!(
                "clip audio read of {frames} frames past the clip end at {}",
                self.layout.len
            )));
        }
        let wanted = frames * self.channels;
        out.try_reserve(wanted)
            .map_err(|error| MediaError::Decode(format!("clip audio buffer: {error}")))?;
        if self.denoise.is_none() {
            self.resample(frames, out)?;
            self.position += frames;
            return Ok(());
        }
        let mut raw = Vec::new();
        let mut emitted = Vec::new();
        while self.denoised.len() < wanted && self.raw_position < self.layout.len {
            let step = RENDER_CHUNK_FRAMES.min(self.layout.len - self.raw_position);
            raw.clear();
            self.resample(step, &mut raw)?;
            emitted.clear();
            if let Some(stream) = self.denoise.as_mut() {
                stream
                    .push(&raw, &mut emitted, &self.cancel)
                    .map_err(denoise_error)?;
            }
            self.denoised.extend(emitted.iter().copied());
        }
        if self.denoised.len() < wanted {
            return Err(MediaError::Decode(
                "clip audio denoiser ended before the clip".to_string(),
            ));
        }
        out.extend(self.denoised.drain(..wanted));
        self.position += frames;
        Ok(())
    }

    /// Resample the next `frames` clip frames from the source.
    fn resample(&mut self, frames: usize, out: &mut Vec<f32>) -> Result<(), MediaError> {
        for _ in 0..frames {
            let source = self.raw_position as f64 * self.layout.ratio;
            let index = source.floor() as usize;
            let fraction = (source - index as f64) as f32;
            self.fill_source(index + 1)?;
            for channel in 0..self.channels {
                let a = self.source_sample(index, channel);
                let value = if fraction == 0.0 {
                    a
                } else {
                    let b = self.source_sample(index + 1, channel);
                    a + (b - a) * fraction
                };
                out.push(value);
            }
            self.raw_position += 1;
        }
        // Keep only the frames the next clip frame can still reach.
        let next = (self.raw_position as f64 * self.layout.ratio).floor() as usize;
        let stale = next.saturating_sub(self.source_start);
        let buffered = self.source_buffer.len() / self.channels;
        if stale > SOURCE_READ_FRAMES && stale <= buffered {
            self.source_buffer.drain(..stale * self.channels);
            self.source_start += stale;
        }
        Ok(())
    }

    /// Decode until source frame `index` is buffered or the source ends.
    fn fill_source(&mut self, index: usize) -> Result<(), MediaError> {
        let mut chunk = Vec::new();
        while !self.source_done
            && self.source_start + self.source_buffer.len() / self.channels <= index
        {
            let Some(stream) = self.source.as_mut() else {
                self.source_done = true;
                break;
            };
            chunk.clear();
            let read = stream.read(SOURCE_READ_FRAMES, &mut chunk)?;
            self.source_buffer.extend(chunk.iter().copied());
            if read < SOURCE_READ_FRAMES {
                // The window is exhausted: reap the decoder now.
                self.source_done = true;
                self.source = None;
            }
        }
        Ok(())
    }

    /// A decoded source sample, or silence past the end of the source.
    fn source_sample(&self, index: usize, channel: usize) -> f32 {
        index
            .checked_sub(self.source_start)
            .and_then(|offset| {
                self.source_buffer
                    .get(offset * self.channels + channel)
                    .copied()
            })
            .unwrap_or(0.0)
    }
}

/// Validated denoise settings for a clip, or `None` when the clip is not
/// denoised (zero strength is a bit-exact bypass).
fn active_denoise(config: Option<AudioDenoise>) -> Result<Option<AudioDenoise>, MediaError> {
    let Some(config) = config.filter(|config| config.strength != 0.0) else {
        return Ok(None);
    };
    config
        .validate()
        .map_err(|error| denoise_error(DenoiseError::InvalidConfig(error.to_string())))?;
    Ok(Some(config))
}

/// The denoiser input for a clip reader: `None` when the clip is not denoised,
/// else the validated settings and the clip's noise profile, computed now if
/// it is not cached (`progress` reports the profile pass in clip frames).
pub(crate) fn clip_denoise(
    config: Option<AudioDenoise>,
    layout: &ClipAudioLayout,
    path: &Path,
    channels: usize,
    cancel: &MediaCancelToken,
    progress: Option<&dyn Fn(usize)>,
) -> Result<Option<(DenoiseProfile, AudioDenoise)>, MediaError> {
    let Some(config) = active_denoise(config)? else {
        return Ok(None);
    };
    let profile = clip_denoise_profile(layout, path, channels, cancel, progress)?;
    Ok(Some((DenoiseProfile::clone(&profile), config)))
}

/// What preview playback should do with a clip's denoise right now.
#[cfg(feature = "playback-engine")]
#[derive(Debug)]
pub(crate) enum PreviewDenoise {
    /// The clip is not denoised in preview.
    Off,
    /// The profile is still being computed in the background (or cannot be):
    /// play the clip undenoised meanwhile.
    Pending,
    Ready(DenoiseProfile, AudioDenoise),
}

/// The owner of preview's background profile passes (one per playback
/// state). Cancelling a scope, when its project closes, cancels only the
/// passes it requested; other owners' passes keep running.
#[cfg(feature = "playback-engine")]
#[derive(Clone)]
pub(crate) struct ProfileScope(Arc<ScopeInner>);

#[cfg(feature = "playback-engine")]
struct ScopeInner {
    id: u64,
    cancel: Mutex<MediaCancelToken>,
}

#[cfg(feature = "playback-engine")]
impl ProfileScope {
    pub(crate) fn new() -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        ProfileScope(Arc::new(ScopeInner {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            cancel: Mutex::new(MediaCancelToken::new()),
        }))
    }

    fn id(&self) -> u64 {
        self.0.id
    }

    fn token(&self) -> MediaCancelToken {
        self.0
            .cancel
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Cancel this scope's queued and running passes (its project is
    /// closing) and forget the passes that failed for it, so the next
    /// project retries them. Cached profiles stay: they are keyed by source
    /// file version.
    pub(crate) fn cancel(&self) {
        {
            let mut token = self
                .0
                .cancel
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            token.cancel();
            *token = MediaCancelToken::new();
        }
        let service = profiles();
        let mut store = service.lock();
        store.drop_queued(|job| job.scope == self.id());
        store.failed.retain(|(scope, _)| *scope != self.id());
        drop(store);
        service.finished.notify_all();
    }
}

#[cfg(feature = "playback-engine")]
impl Default for ProfileScope {
    fn default() -> Self {
        ProfileScope::new()
    }
}

/// Preview's denoise input for a clip. Never runs a profile pass on the
/// caller's thread: a missing profile is queued on the background worker (once
/// per layout, however many windows or playback sessions ask) and the clip
/// plays undenoised until it is ready. A newer request for the same clip (a
/// trim changed its layout) replaces the queued one.
#[cfg(feature = "playback-engine")]
pub(crate) fn preview_clip_denoise(
    config: Option<AudioDenoise>,
    clip_id: &str,
    layout: &ClipAudioLayout,
    path: &Path,
    channels: usize,
    scope: &ProfileScope,
) -> Result<PreviewDenoise, MediaError> {
    let Some(config) = active_denoise(config)? else {
        return Ok(PreviewDenoise::Off);
    };
    let key = ProfileKey::new(layout, path, channels);
    let service = profiles();
    let mut store = service.lock();
    if let Some(profile) = store.cached(&key) {
        return Ok(PreviewDenoise::Ready(
            DenoiseProfile::clone(&profile),
            config,
        ));
    }
    let job = ProfileJob {
        key,
        clip_id: clip_id.to_string(),
        layout: *layout,
        path: path.to_path_buf(),
        channels,
        scope: scope.id(),
        cancel: scope.token(),
    };
    let running = service.start_worker();
    let superseded = store.request(job, running);
    drop(store);
    if superseded {
        service.finished.notify_all();
    }
    service.queued.notify_one();
    Ok(PreviewDenoise::Pending)
}

/// Whether a clip's noise profile still needs a pass (not cached).
pub(crate) fn denoise_profile_pending(
    config: Option<AudioDenoise>,
    layout: &ClipAudioLayout,
    path: &Path,
    channels: usize,
) -> bool {
    if !matches!(active_denoise(config), Ok(Some(_))) {
        return false;
    }
    let key = ProfileKey::new(layout, path, channels);
    profiles().lock().cached(&key).is_none()
}

/// Cache key for a clip's noise profile: the source file (and its version),
/// the source window, and the frame grid it is rendered on.
#[derive(Clone, Debug, PartialEq)]
struct ProfileKey {
    path: PathBuf,
    file_len: u64,
    modified: Option<SystemTime>,
    source_lo: u64,
    source_hi: u64,
    ratio: u64,
    len: usize,
    rate: u32,
    channels: usize,
}

impl ProfileKey {
    fn new(layout: &ClipAudioLayout, path: &Path, channels: usize) -> Self {
        let metadata = std::fs::metadata(path).ok();
        ProfileKey {
            path: path.to_path_buf(),
            file_len: metadata.as_ref().map_or(0, std::fs::Metadata::len),
            modified: metadata.and_then(|metadata| metadata.modified().ok()),
            source_lo: layout.source_lo.to_bits(),
            source_hi: layout.source_hi.to_bits(),
            ratio: layout.ratio.to_bits(),
            len: layout.len,
            rate: layout.rate,
            channels,
        }
    }
}

struct ProfileJob {
    key: ProfileKey,
    /// The clip that asked; a newer request for it replaces this one.
    #[cfg(feature = "playback-engine")]
    clip_id: String,
    #[cfg(feature = "playback-engine")]
    layout: ClipAudioLayout,
    #[cfg(feature = "playback-engine")]
    path: PathBuf,
    #[cfg(feature = "playback-engine")]
    channels: usize,
    /// The [`ProfileScope`] that asked, and its cancellation.
    #[cfg(feature = "playback-engine")]
    scope: u64,
    #[cfg(feature = "playback-engine")]
    cancel: MediaCancelToken,
}

struct ProfileStore {
    /// Least recently used first.
    cache: VecDeque<(ProfileKey, Arc<DenoiseProfile>)>,
    /// Passes running or queued, by any caller: a second caller for the same
    /// key waits for (or, in preview, skips) the first instead of decoding
    /// the clip again.
    in_flight: Vec<ProfileKey>,
    /// Background passes waiting for the worker, oldest first; it runs the
    /// newest first.
    queue: VecDeque<ProfileJob>,
    /// Background passes that failed, by scope; preview plays these clips
    /// undenoised instead of retrying every window. Cleared when the scope's
    /// project closes.
    failed: Vec<(u64, ProfileKey)>,
}

impl ProfileStore {
    fn new() -> Self {
        ProfileStore {
            cache: VecDeque::new(),
            in_flight: Vec::new(),
            queue: VecDeque::new(),
            failed: Vec::new(),
        }
    }

    /// The cached profile for `key`, now the most recently used.
    fn cached(&mut self, key: &ProfileKey) -> Option<Arc<DenoiseProfile>> {
        let index = self.cache.iter().position(|(cached, _)| cached == key)?;
        let entry = self.cache.remove(index)?;
        let profile = Arc::clone(&entry.1);
        self.cache.push_back(entry);
        Some(profile)
    }

    /// Queue a background pass unless one for its key is in flight or failed
    /// in its scope, replacing the queued passes of the same clip. Without a
    /// worker the pass counts as failed, so the clip plays undenoised.
    /// Returns whether queued passes were dropped (waiters must re-check).
    #[cfg(feature = "playback-engine")]
    fn request(&mut self, job: ProfileJob, worker_running: bool) -> bool {
        if self.in_flight.contains(&job.key)
            || self
                .failed
                .iter()
                .any(|(scope, key)| *scope == job.scope && *key == job.key)
        {
            return false;
        }
        if !worker_running {
            eprintln!(
                "[audio] denoise profile worker is not running; {} plays undenoised",
                job.path.display()
            );
            self.failed.push((job.scope, job.key));
            return false;
        }
        let mut superseded =
            self.drop_queued(|queued| queued.scope == job.scope && queued.clip_id == job.clip_id);
        while self.queue.len() >= PROFILE_QUEUE_CAPACITY {
            if let Some(oldest) = self.queue.pop_front() {
                self.in_flight.retain(|key| *key != oldest.key);
                superseded = true;
            }
        }
        #[cfg(test)]
        test_hooks::record_request(&job.path);
        self.in_flight.push(job.key.clone());
        self.queue.push_back(job);
        superseded
    }

    /// Drop the queued passes `drop` selects. Returns whether any were.
    fn drop_queued(&mut self, drop: impl Fn(&ProfileJob) -> bool) -> bool {
        let before = self.queue.len();
        let in_flight = &mut self.in_flight;
        self.queue.retain(|job| {
            if drop(job) {
                in_flight.retain(|key| *key != job.key);
                false
            } else {
                true
            }
        });
        self.queue.len() != before
    }

    /// Record the end of a pass for `key`, caching its profile if it succeeded.
    fn finish(&mut self, key: &ProfileKey, profile: Option<&Arc<DenoiseProfile>>) {
        self.in_flight.retain(|queued| queued != key);
        if let Some(profile) = profile {
            self.failed.retain(|(_, failed)| failed != key);
            self.cache.retain(|(cached, _)| cached != key);
            while self.cache.len() >= PROFILE_CACHE_CAPACITY {
                self.cache.pop_front();
            }
            self.cache.push_back((key.clone(), Arc::clone(profile)));
        }
    }
}

struct ProfileService {
    store: Mutex<ProfileStore>,
    /// Signalled whenever a pass ends or leaves the queue.
    finished: Condvar,
    /// Signalled whenever a pass is queued.
    #[cfg(feature = "playback-engine")]
    queued: Condvar,
    /// Whether the worker thread started.
    #[cfg(feature = "playback-engine")]
    worker: OnceLock<bool>,
}

impl ProfileService {
    fn lock(&self) -> MutexGuard<'_, ProfileStore> {
        self.store
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Start the background worker on first use; false if it could not
    /// start. One worker runs the passes one at a time, so preview never
    /// runs more than one extra decode however many denoised clips it plays.
    #[cfg(feature = "playback-engine")]
    fn start_worker(&'static self) -> bool {
        *self.worker.get_or_init(|| {
            match std::thread::Builder::new()
                .name("opentake-denoise-profile".to_string())
                .spawn(move || run_profile_worker(self))
            {
                Ok(_) => true,
                Err(error) => {
                    eprintln!("[audio] could not start the denoise profile worker: {error}");
                    false
                }
            }
        })
    }
}

fn profiles() -> &'static ProfileService {
    static SERVICE: OnceLock<ProfileService> = OnceLock::new();
    SERVICE.get_or_init(|| ProfileService {
        store: Mutex::new(ProfileStore::new()),
        finished: Condvar::new(),
        #[cfg(feature = "playback-engine")]
        queued: Condvar::new(),
        #[cfg(feature = "playback-engine")]
        worker: OnceLock::new(),
    })
}

#[cfg(feature = "playback-engine")]
fn run_profile_worker(service: &'static ProfileService) {
    loop {
        let job = {
            let mut store = service.lock();
            loop {
                if let Some(job) = store.queue.pop_back() {
                    break job;
                }
                store = service
                    .queued
                    .wait(store)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
            }
        };
        #[cfg(test)]
        test_hooks::wait_while_held(&job.path, &job.cancel);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            compute_profile(&job.layout, &job.path, job.channels, &job.cancel, None)
        }))
        .unwrap_or_else(|_| {
            Err(MediaError::Decode(
                "denoise profile pass panicked".to_string(),
            ))
        });
        let mut store = service.lock();
        store.finish(&job.key, result.as_ref().ok());
        match &result {
            Ok(_) | Err(MediaError::Cancelled) => {}
            Err(error) => {
                // Preview keeps playing the clip undenoised; export computes
                // the profile itself and reports the error.
                eprintln!(
                    "[audio] denoise profile for {} failed: {error}",
                    job.path.display()
                );
                store.failed.push((job.scope, job.key.clone()));
            }
        }
        drop(store);
        service.finished.notify_all();
    }
}

/// The noise profile of a clip's whole rendered audio, estimated once from
/// its entire source window (one extra decode) and cached by source version.
/// Waits for a pass another caller already runs for the same clip, and takes
/// over one that only waits in the background queue; `progress` reports this
/// caller's own pass in clip frames.
pub(crate) fn clip_denoise_profile(
    layout: &ClipAudioLayout,
    path: &Path,
    channels: usize,
    cancel: &MediaCancelToken,
    progress: Option<&dyn Fn(usize)>,
) -> Result<Arc<DenoiseProfile>, MediaError> {
    let key = ProfileKey::new(layout, path, channels);
    let service = profiles();
    let mut store = service.lock();
    loop {
        if let Some(profile) = store.cached(&key) {
            return Ok(profile);
        }
        store.drop_queued(|job| job.key == key);
        if !store.in_flight.contains(&key) {
            store.in_flight.push(key.clone());
            break;
        }
        if cancel.checkpoint() {
            return Err(MediaError::Cancelled);
        }
        store = service
            .finished
            .wait_timeout(store, PROFILE_WAIT_POLL)
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .0;
    }
    drop(store);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        compute_profile(layout, path, channels, cancel, progress)
    }));
    let mut store = service.lock();
    store.finish(
        &key,
        result.as_ref().ok().and_then(|result| result.as_ref().ok()),
    );
    drop(store);
    service.finished.notify_all();
    result.unwrap_or_else(|panic| std::panic::resume_unwind(panic))
}

fn compute_profile(
    layout: &ClipAudioLayout,
    path: &Path,
    channels: usize,
    cancel: &MediaCancelToken,
    progress: Option<&dyn Fn(usize)>,
) -> Result<Arc<DenoiseProfile>, MediaError> {
    #[cfg(all(test, feature = "playback-engine"))]
    test_hooks::record_pass(path);
    let mut reader = ClipAudioReader::open(*layout, path, channels, 0, None, cancel)?;
    let mut builder =
        DenoiseProfileBuilder::new(channels, layout.rate, layout.len).map_err(denoise_error)?;
    let mut chunk = Vec::new();
    let mut done = 0;
    while done < layout.len {
        let step = RENDER_CHUNK_FRAMES.min(layout.len - done);
        chunk.clear();
        reader.read(step, &mut chunk)?;
        builder.push(&chunk, cancel).map_err(denoise_error)?;
        done += step;
        if let Some(progress) = progress {
            progress(done);
        }
    }
    Ok(Arc::new(builder.finish(cancel).map_err(denoise_error)?))
}

/// Test seam counting the clip readers open at once: a census started on a
/// thread counts the readers that thread opens until they drop, wherever
/// they drop.
#[cfg(test)]
pub(crate) mod reader_census {
    use std::cell::RefCell;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[derive(Default)]
    struct Counts {
        live: AtomicUsize,
        peak: AtomicUsize,
    }

    thread_local! {
        static CENSUS: RefCell<Option<Arc<Counts>>> = const { RefCell::new(None) };
    }

    /// Counts this thread's readers until dropped.
    pub(crate) struct Census(Arc<Counts>);

    pub(crate) fn start() -> Census {
        let counts = Arc::new(Counts::default());
        CENSUS.with(|slot| *slot.borrow_mut() = Some(Arc::clone(&counts)));
        Census(counts)
    }

    impl Census {
        /// Readers open now.
        pub(crate) fn live(&self) -> usize {
            self.0.live.load(Ordering::SeqCst)
        }

        /// The most readers open at once since the census started.
        pub(crate) fn peak(&self) -> usize {
            self.0.peak.load(Ordering::SeqCst)
        }
    }

    impl Drop for Census {
        fn drop(&mut self) {
            let _ = CENSUS.try_with(|slot| slot.borrow_mut().take());
        }
    }

    /// One counted reader.
    pub(crate) struct Entry(Arc<Counts>);

    pub(super) fn enter() -> Option<Entry> {
        let counts = CENSUS
            .try_with(|slot| slot.borrow().clone())
            .ok()
            .flatten()?;
        let live = counts.live.fetch_add(1, Ordering::SeqCst) + 1;
        counts.peak.fetch_max(live, Ordering::SeqCst);
        Some(Entry(counts))
    }

    impl Drop for Entry {
        fn drop(&mut self) {
            self.0.live.fetch_sub(1, Ordering::SeqCst);
        }
    }
}

/// Test seams for the background profile worker: count passes and requests
/// per source path, and hold a path's background pass until released.
#[cfg(all(test, feature = "playback-engine"))]
pub(crate) mod test_hooks {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::{Condvar, Mutex};
    use std::time::Duration;

    use opentake_media::MediaCancelToken;

    static PASSES: Mutex<Option<HashMap<PathBuf, usize>>> = Mutex::new(None);
    static REQUESTS: Mutex<Option<HashMap<PathBuf, usize>>> = Mutex::new(None);
    static HELD: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());
    static RELEASED: Condvar = Condvar::new();

    fn bump(map: &Mutex<Option<HashMap<PathBuf, usize>>>, path: &Path) {
        *map.lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get_or_insert_with(HashMap::new)
            .entry(path.to_path_buf())
            .or_default() += 1;
    }

    fn get(map: &Mutex<Option<HashMap<PathBuf, usize>>>, path: &Path) -> usize {
        map.lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .and_then(|counts| counts.get(path).copied())
            .unwrap_or(0)
    }

    pub(super) fn record_pass(path: &Path) {
        bump(&PASSES, path);
    }

    pub(super) fn record_request(path: &Path) {
        bump(&REQUESTS, path);
    }

    /// Profile passes run for `path` (by any caller).
    pub(crate) fn passes(path: &Path) -> usize {
        get(&PASSES, path)
    }

    /// Background passes queued for `path`.
    pub(crate) fn requests(path: &Path) -> usize {
        get(&REQUESTS, path)
    }

    /// Hold background passes for `path` until the guard drops.
    pub(crate) struct Hold(PathBuf);

    pub(crate) fn hold(path: &Path) -> Hold {
        HELD.lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(path.to_path_buf());
        Hold(path.to_path_buf())
    }

    impl Drop for Hold {
        fn drop(&mut self) {
            HELD.lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .retain(|held| *held != self.0);
            RELEASED.notify_all();
        }
    }

    pub(super) fn wait_while_held(path: &Path, cancel: &MediaCancelToken) {
        let mut held = HELD.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        while held.iter().any(|held| held == path) && !cancel.is_cancelled() {
            held = RELEASED
                .wait_timeout(held, Duration::from_millis(20))
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .0;
        }
    }
}

/// A mono 48 kHz test signal (a sine plus a little white noise) and a WAV
/// writer for it, shared by the preview and export audio tests.
#[cfg(test)]
pub(crate) mod fixtures {
    use std::path::Path;

    pub(crate) const RATE: u32 = 48_000;

    pub(crate) fn noisy_tone(seconds: f32, frequency: f32, seed: u32) -> Vec<f32> {
        let mut state = seed.max(1);
        (0..(seconds * RATE as f32) as usize)
            .map(|index| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let white = (state as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32;
                let time = index as f32 / RATE as f32;
                (std::f32::consts::TAU * frequency * time).sin() * 0.3 + white * 0.04
            })
            .collect()
    }

    /// Write `samples` as a mono 16-bit PCM WAV at [`RATE`].
    pub(crate) fn write_wav(path: &Path, samples: &[f32]) {
        let data = samples
            .iter()
            .flat_map(|sample| ((sample.clamp(-1.0, 1.0) * 32_767.0).round() as i16).to_le_bytes())
            .collect::<Vec<_>>();
        let mut wav = Vec::with_capacity(44 + data.len());
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16_u32.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&RATE.to_le_bytes());
        wav.extend_from_slice(&(RATE * 2).to_le_bytes());
        wav.extend_from_slice(&2_u16.to_le_bytes());
        wav.extend_from_slice(&16_u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
        wav.extend_from_slice(&data);
        std::fs::write(path, wav).expect("write WAV fixture");
    }

    /// Encode a 440 Hz sine (amplitude 0.125) of `seconds` at `rate` with
    /// `codec`; false when this FFmpeg build cannot.
    pub(crate) fn encode_sine(path: &Path, codec: &str, rate: u32, seconds: u32) -> bool {
        std::process::Command::new(opentake_media::ffmpeg_status::ffmpeg_path())
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
            ])
            .arg(format!(
                "sine=frequency=440:sample_rate={rate}:duration={seconds}"
            ))
            .args(["-c:a", codec, "-b:a", "128k"])
            .arg(path)
            .status()
            .is_ok_and(|status| status.success())
    }

    pub(crate) fn ffmpeg_ready() -> bool {
        opentake_media::ffmpeg_status::ffmpeg_available()
            && opentake_media::ffmpeg_status::ffprobe_available()
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::{encode_sine, ffmpeg_ready, noisy_tone, write_wav, RATE};
    use super::*;
    use opentake_domain::DenoiseMode;
    #[cfg(feature = "playback-engine")]
    use std::time::Duration;

    fn clip(start_frame: i32, duration_frames: i32) -> Clip {
        Clip::new("clip", "media", start_frame, duration_frames)
    }

    #[test]
    fn layout_places_clips_on_absolute_mix_frames() {
        let mut speed_two = clip(45, 30);
        speed_two.trim_start_frame = 15;
        speed_two.speed = 2.0;
        let layout = ClipAudioLayout::new(&speed_two, 30, RATE).unwrap();
        assert_eq!(layout.span(), (72_000, 120_000));
        assert_eq!(layout.len, 48_000);
        assert_eq!(layout.offset_of(96_000), 24_000);
        assert_eq!((layout.source_lo, layout.source_hi), (0.5, 2.5));
        assert_eq!(layout.ratio, 2.0);

        // A clip starting before the timeline keeps its source alignment and
        // is only audible from frame 0 on.
        let early = ClipAudioLayout::new(&clip(-15, 30), 30, RATE).unwrap();
        assert_eq!(early.span(), (0, 24_000));
        assert_eq!(early.offset_of(0), 24_000);

        assert!(ClipAudioLayout::new(&clip(0, 0), 30, RATE).is_none());
        let mut frozen = clip(0, 30);
        frozen.speed = 0.0;
        assert!(ClipAudioLayout::new(&frozen, 30, RATE).is_none());
    }

    #[test]
    fn timeline_frame_lookup_is_exact_on_frame_boundaries() {
        // 1_600 / 48_000 * 30 rounds to 0.999…; the gain lookup must not.
        assert_eq!(timeline_frame_at(1_599, 30, RATE), 0);
        assert_eq!(timeline_frame_at(1_600, 30, RATE), 1);
        assert_eq!(timeline_frame_at(44_099, 24, 44_100), 23);
        assert_eq!(timeline_frame_at(44_100, 24, 44_100), 24);
    }

    fn read_all(
        layout: ClipAudioLayout,
        path: &Path,
        from: usize,
        steps: &[usize],
        denoise: Option<(DenoiseProfile, AudioDenoise)>,
    ) -> Vec<f32> {
        let cancel = MediaCancelToken::new();
        let mut reader = ClipAudioReader::open(layout, path, 1, from, denoise, &cancel).unwrap();
        let mut out = Vec::new();
        let mut position = from;
        for step in steps.iter().copied().cycle() {
            let step = step.min(layout.len - position);
            if step == 0 {
                break;
            }
            reader.read(step, &mut out).unwrap();
            position += step;
        }
        out
    }

    fn test_profile() -> Arc<DenoiseProfile> {
        let cancel = MediaCancelToken::new();
        let mut builder = DenoiseProfileBuilder::new(1, RATE, 4_096).unwrap();
        builder.push(&vec![0.01; 4_096], &cancel).unwrap();
        Arc::new(builder.finish(&cancel).unwrap())
    }

    fn test_key(frames: i32) -> ProfileKey {
        let layout = ClipAudioLayout::new(&clip(0, frames), 30, RATE).unwrap();
        ProfileKey::new(
            &layout,
            Path::new("/nonexistent/opentake-profile-test.wav"),
            1,
        )
    }

    #[cfg(feature = "playback-engine")]
    fn test_job(clip_id: &str, frames: i32, scope: u64) -> ProfileJob {
        let layout = ClipAudioLayout::new(&clip(0, frames), 30, RATE).unwrap();
        let path = PathBuf::from("/nonexistent/opentake-profile-test.wav");
        ProfileJob {
            key: ProfileKey::new(&layout, &path, 1),
            clip_id: clip_id.to_string(),
            layout,
            path,
            channels: 1,
            scope,
            cancel: MediaCancelToken::new(),
        }
    }

    #[test]
    fn profile_cache_evicts_the_least_recently_used_profile() {
        let mut store = ProfileStore::new();
        let profile = test_profile();
        let keys = (1..=PROFILE_CACHE_CAPACITY as i32 + 1)
            .map(test_key)
            .collect::<Vec<_>>();
        for key in &keys[..PROFILE_CACHE_CAPACITY] {
            store.finish(key, Some(&profile));
        }
        // Reading the oldest entry makes it the most recently used.
        assert!(store.cached(&keys[0]).is_some());
        store.finish(&keys[PROFILE_CACHE_CAPACITY], Some(&profile));
        assert_eq!(store.cache.len(), PROFILE_CACHE_CAPACITY);
        assert!(store.cached(&keys[0]).is_some(), "recently read entry kept");
        assert!(
            store.cached(&keys[1]).is_none(),
            "least recently used evicted"
        );
        assert!(store.cached(&keys[PROFILE_CACHE_CAPACITY]).is_some());
    }

    #[test]
    #[cfg(feature = "playback-engine")]
    fn a_newer_request_for_a_clip_replaces_its_queued_pass_and_runs_first() {
        let mut store = ProfileStore::new();
        let old_layout = test_job("speech", 90, 7);
        let old_key = old_layout.key.clone();
        assert!(!store.request(old_layout, true));
        assert!(!store.request(test_job("music", 60, 7), true));
        // A trim gives the clip a new layout: its stale pass leaves the queue.
        let trimmed = test_job("speech", 80, 7);
        let trimmed_key = trimmed.key.clone();
        assert!(store.request(trimmed, true), "the stale pass was dropped");
        assert!(!store.in_flight.contains(&old_key));
        assert_eq!(store.queue.len(), 2);
        assert_eq!(
            store.queue.back().map(|job| &job.key),
            Some(&trimmed_key),
            "the worker takes the newest request first"
        );
        // Another scope's request for the same clip id is its own.
        assert!(!store.request(test_job("speech", 70, 8), true));
        assert_eq!(store.queue.len(), 3);
        // The queue is bounded; the oldest requests go first.
        for frames in 100..100 + PROFILE_QUEUE_CAPACITY as i32 {
            store.request(test_job(&format!("c{frames}"), frames, 7), true);
        }
        assert_eq!(store.queue.len(), PROFILE_QUEUE_CAPACITY);
        assert_eq!(store.in_flight.len(), PROFILE_QUEUE_CAPACITY);
        assert!(!store.in_flight.contains(&trimmed_key));
    }

    #[test]
    #[cfg(feature = "playback-engine")]
    fn preview_plays_undenoised_when_the_profile_worker_is_not_running() {
        let mut store = ProfileStore::new();
        let job = test_job("speech", 90, 3);
        let key = job.key.clone();
        assert!(!store.request(job, false));
        assert!(store.queue.is_empty());
        assert!(!store.in_flight.contains(&key));
        assert!(store.failed.contains(&(3, key.clone())));
        // Later windows do not ask again for this scope.
        store.request(test_job("speech", 90, 3), true);
        assert!(store.queue.is_empty());
    }

    #[test]
    #[cfg(feature = "playback-engine")]
    fn cancelling_a_profile_scope_leaves_other_scopes_passes_running() {
        if !ffmpeg_ready() {
            eprintln!("skip: ffmpeg/ffprobe not available");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("scoped.wav");
        write_wav(&path, &noisy_tone(3.0, 250.0, 21));
        let layout = ClipAudioLayout::new(&clip(0, 60), 30, RATE).unwrap();
        let config = Some(AudioDenoise {
            mode: DenoiseMode::Voice,
            strength: 0.6,
            preview_enabled: true,
        });
        let key = ProfileKey::new(&layout, &path, 1);
        let in_flight = || profiles().lock().in_flight.contains(&key);
        let wait_until = |what: &str, ready: &dyn Fn() -> bool| {
            let deadline = std::time::Instant::now() + Duration::from_secs(30);
            while !ready() {
                assert!(std::time::Instant::now() < deadline, "timed out: {what}");
                std::thread::sleep(Duration::from_millis(10));
            }
        };
        let hold = test_hooks::hold(&path);
        let scope = ProfileScope::new();
        let request = || preview_clip_denoise(config, "speech", &layout, &path, 1, &scope).unwrap();
        assert!(matches!(request(), PreviewDenoise::Pending));
        assert_eq!(test_hooks::requests(&path), 1);

        // Another owner's project transition does not touch this pass.
        ProfileScope::new().cancel();
        assert!(in_flight());
        assert!(matches!(request(), PreviewDenoise::Pending));
        assert_eq!(test_hooks::requests(&path), 1);

        // This owner's transition cancels it (queued or held), and the next
        // project asks again.
        scope.cancel();
        wait_until("the cancelled pass to end", &|| !in_flight());
        assert!(matches!(request(), PreviewDenoise::Pending));
        assert_eq!(test_hooks::requests(&path), 2);
        drop(hold);
        wait_until("the profile", &|| {
            !denoise_profile_pending(config, &layout, &path, 1)
        });
        assert!(matches!(request(), PreviewDenoise::Ready(..)));
    }

    #[test]
    fn reader_opened_mid_clip_pre_rolls_compressed_and_resampled_sources() {
        if !ffmpeg_ready() {
            eprintln!("skip: ffmpeg/ffprobe not available");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        for (name, codec) in [("tone.m4a", "aac"), ("tone.mp3", "libmp3lame")] {
            let path = dir.path().join(name);
            if !encode_sine(&path, codec, 44_100, 8) {
                eprintln!("skip {name}: ffmpeg could not encode it");
                continue;
            }
            let mut source = clip(0, 200);
            source.trim_start_frame = 7;
            let layout = ClipAudioLayout::new(&source, 30, RATE).unwrap();
            let whole = read_all(layout, &path, 0, &[layout.len], None);
            let cancel = MediaCancelToken::new();
            for from in [96_000, 123_457, 250_001] {
                let difference = |preroll| {
                    let mut reader = ClipAudioReader::open_with_preroll(
                        layout, &path, 1, from, None, preroll, &cancel,
                    )
                    .unwrap();
                    let mut out = Vec::new();
                    reader.read(24_000, &mut out).unwrap();
                    out.iter()
                        .zip(&whole[from..])
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0_f32, f32::max)
                };
                // A decode started at `from` itself is off by a large step
                // (decoder priming); the pre-roll leaves only a sub-sample
                // resampler phase difference (the tone peaks at 0.125).
                let pre_rolled = difference(DECODE_PREROLL_FRAMES);
                assert!(
                    pre_rolled < 5.0e-3,
                    "{name} from {from}: pre-rolled reader differs by {pre_rolled}"
                );
                assert!(
                    difference(0) > 10.0 * pre_rolled,
                    "{name} from {from}: the pre-roll is what makes the difference"
                );
            }
        }
    }

    #[test]
    fn denoised_reader_opened_mid_clip_matches_at_every_mix_rate() {
        if !ffmpeg_ready() {
            eprintln!("skip: ffmpeg/ffprobe not available");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("speech.wav");
        write_wav(&path, &noisy_tone(5.0, 310.0, 13));
        let config = AudioDenoise {
            mode: DenoiseMode::Voice,
            strength: 0.7,
            preview_enabled: true,
        };
        let cancel = MediaCancelToken::new();
        // Below 32 kHz the denoiser's hop is 128 and its warm-up (2,560
        // frames) is shorter than the reader's pre-roll; every start must
        // still land on the hop grid.
        for rate in [8_000_u32, 16_000, 22_050, 24_000, 48_000] {
            let layout = ClipAudioLayout::new(&clip(0, 120), 30, rate).unwrap();
            let profile = clip_denoise_profile(&layout, &path, 1, &cancel, None).unwrap();
            let denoised = |from, steps: &[usize]| {
                read_all(
                    layout,
                    &path,
                    from,
                    steps,
                    Some((DenoiseProfile::clone(&profile), config)),
                )
            };
            let reference = denoised(0, &[layout.len]);
            for from in [
                1,
                777,
                DECODE_PREROLL_FRAMES + 1,
                5_003,
                layout.len / 2 + 37,
            ] {
                assert_ne!(from % 128, 0);
                // Opening off the hop grid used to fail here below 32 kHz.
                let restarted = denoised(from, &[3_001]);
                let max_difference = restarted
                    .iter()
                    .zip(&reference[from..])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max);
                // Rates that divide the 48 kHz source decode exactly after a
                // seek; 22.05 kHz restarts FFmpeg's resampler at another
                // phase, which undenoised readers show just the same.
                let tolerance = if 48_000 % rate == 0 { 1.0e-6 } else { 0.05 };
                assert!(
                    max_difference < tolerance,
                    "{rate} Hz reader from {from} differs by {max_difference}"
                );
            }
        }
    }

    #[test]
    fn reader_output_depends_only_on_the_clip_position() {
        if !ffmpeg_ready() {
            eprintln!("skip: ffmpeg/ffprobe not available");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tone.wav");
        write_wav(&path, &noisy_tone(6.0, 330.0, 7));
        let mut retimed = clip(0, 90);
        retimed.trim_start_frame = 12;
        retimed.speed = 1.5;
        let config = AudioDenoise {
            mode: DenoiseMode::Voice,
            strength: 0.7,
            preview_enabled: true,
        };
        for source in [clip(0, 150), retimed] {
            let layout = ClipAudioLayout::new(&source, 30, RATE).unwrap();
            let cancel = MediaCancelToken::new();
            let whole = read_all(layout, &path, 0, &[layout.len], None);
            assert_eq!(whole.len(), layout.len);
            assert_eq!(read_all(layout, &path, 0, &[1, 4_095, 17_000], None), whole);
            let from = 37_123;
            assert_eq!(
                read_all(layout, &path, from, &[9_999], None),
                whole[from..],
                "a reader opened mid-clip continues the same samples"
            );

            let profile = clip_denoise_profile(&layout, &path, 1, &cancel, None).unwrap();
            let denoised = |from, steps: &[usize]| {
                read_all(
                    layout,
                    &path,
                    from,
                    steps,
                    Some((DenoiseProfile::clone(&profile), config)),
                )
            };
            let reference = denoised(0, &[layout.len]);
            assert_eq!(denoised(0, &[96_000, 5, 20_000]), reference);
            let restarted = denoised(from, &[96_000]);
            let max_difference = restarted
                .iter()
                .zip(&reference[from..])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                max_difference < 1.0e-6,
                "restart differs by {max_difference}"
            );
            assert_ne!(reference, whole, "denoise changed the signal");
        }
    }
}
