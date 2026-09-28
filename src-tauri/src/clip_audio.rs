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
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, SystemTime};

use opentake_domain::{AudioDenoise, Clip};
use opentake_media::analysis::{
    denoise_stream_start, DenoiseError, DenoiseProfile, DenoiseProfileBuilder, DenoiseStream,
};
use opentake_media::{MediaCancelToken, MediaError, PcmFormat, PcmSpec, PcmStream};

/// Source frames pulled from the decoder per read.
const SOURCE_READ_FRAMES: usize = 8 * 1024;
/// Clip frames resampled (and fed to the denoiser) per step.
const RENDER_CHUNK_FRAMES: usize = 4 * 1024;
/// Distinct clip noise profiles remembered across preview windows, playback
/// sessions and exports.
const PROFILE_CACHE_CAPACITY: usize = 32;
/// How often a caller waiting for another caller's profile pass checks its
/// own cancellation.
const PROFILE_WAIT_POLL: Duration = Duration::from_millis(50);
/// Clip decoders one mix keeps open across windows. Each is an FFmpeg process
/// with three pipes; beyond this many overlapping clips, further clips are
/// decoded by a reader opened for one window and closed again, which keeps a
/// dense timeline well inside the default open-file limit (256 on macOS).
pub(crate) const MAX_OPEN_CLIP_READERS: usize = 16;

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
}

impl ClipAudioReader {
    /// Serve clip frames from `from` on. A denoised reader starts decoding a
    /// warm-up before `from`, so its output matches a reader that started at
    /// the clip's first frame.
    pub(crate) fn open(
        layout: ClipAudioLayout,
        path: &Path,
        channels: usize,
        from: usize,
        denoise: Option<(DenoiseProfile, AudioDenoise)>,
        cancel: &MediaCancelToken,
    ) -> Result<Self, MediaError> {
        if from > layout.len || channels == 0 || channels > usize::from(u16::MAX) {
            return Err(MediaError::Decode(format!(
                "clip audio read at {from} of {} frames x {channels} channels",
                layout.len
            )));
        }
        let start = if denoise.is_some() {
            denoise_stream_start(layout.rate, from)
        } else {
            from
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
        };
        if start < from {
            // Warm-up: frames before `from` only settle the denoiser.
            let mut discard = Vec::new();
            reader.read(from - start, &mut discard)?;
        }
        Ok(reader)
    }

    /// The next clip frame [`ClipAudioReader::read`] returns.
    pub(crate) fn position(&self) -> usize {
        self.position
    }

    /// Whether this reader denoises its output.
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
#[derive(Debug)]
pub(crate) enum PreviewDenoise {
    /// The clip is not denoised in preview.
    Off,
    /// The profile is still being computed in the background: play the clip
    /// undenoised meanwhile.
    Pending,
    Ready(DenoiseProfile, AudioDenoise),
}

/// Preview's denoise input for a clip. Never runs a profile pass on the
/// caller's thread: a missing profile is queued on the background worker (once
/// per layout, however many windows or playback sessions ask) and the clip
/// plays undenoised until it is ready.
pub(crate) fn preview_clip_denoise(
    config: Option<AudioDenoise>,
    layout: &ClipAudioLayout,
    path: &Path,
    channels: usize,
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
    if store.in_flight.contains(&key) || store.failed.contains(&key) {
        return Ok(PreviewDenoise::Pending);
    }
    let job = ProfileJob {
        key: key.clone(),
        layout: *layout,
        path: path.to_path_buf(),
        channels,
        cancel: store.background.clone(),
    };
    store.in_flight.push(key);
    #[cfg(test)]
    test_hooks::record_request(path);
    let sent = match service.worker() {
        Some(worker) => worker.send(job).map_err(|mpsc::SendError(job)| job.key),
        None => Err(job.key),
    };
    if let Err(key) = sent {
        store.in_flight.retain(|queued| *queued != key);
        return Err(MediaError::Decode(
            "denoise profile worker is not running".to_string(),
        ));
    }
    Ok(PreviewDenoise::Pending)
}

/// Cancel queued and running background profile passes (the project is
/// closing) and forget passes that failed, so the next project retries them.
/// Cached profiles stay: they are keyed by source file version.
pub(crate) fn cancel_background_profiles() {
    let mut store = profiles().lock();
    store.background.cancel();
    store.background = MediaCancelToken::new();
    store.failed.clear();
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
    layout: ClipAudioLayout,
    path: PathBuf,
    channels: usize,
    cancel: MediaCancelToken,
}

struct ProfileStore {
    cache: VecDeque<(ProfileKey, Arc<DenoiseProfile>)>,
    /// Passes running or queued, by any caller: a second caller for the same
    /// key waits for (or, in preview, skips) the first instead of decoding
    /// the clip again.
    in_flight: Vec<ProfileKey>,
    /// Background passes that failed; preview plays these clips undenoised
    /// instead of retrying every window. Cleared when the project closes.
    failed: Vec<ProfileKey>,
    /// Cancels background passes; replaced when the project closes.
    background: MediaCancelToken,
}

impl ProfileStore {
    fn cached(&self, key: &ProfileKey) -> Option<Arc<DenoiseProfile>> {
        self.cache
            .iter()
            .find(|(cached, _)| cached == key)
            .map(|(_, profile)| Arc::clone(profile))
    }

    /// Record the end of a pass for `key`, caching its profile if it succeeded.
    fn finish(&mut self, key: &ProfileKey, profile: Option<&Arc<DenoiseProfile>>) {
        self.in_flight.retain(|queued| queued != key);
        if let Some(profile) = profile {
            self.failed.retain(|failed| failed != key);
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
    /// Signalled whenever a pass ends.
    finished: Condvar,
    /// `None` when the worker thread could not be started.
    worker: OnceLock<Option<Mutex<Sender<ProfileJob>>>>,
}

impl ProfileService {
    fn lock(&self) -> MutexGuard<'_, ProfileStore> {
        self.store
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The background worker's queue, starting the worker on first use. One
    /// worker runs the passes one at a time, so preview never runs more than
    /// one extra decode however many denoised clips it plays.
    fn worker(&self) -> Option<Sender<ProfileJob>> {
        let worker = self.worker.get_or_init(|| {
            let (sender, receiver) = mpsc::channel();
            match std::thread::Builder::new()
                .name("opentake-denoise-profile".to_string())
                .spawn(move || run_profile_worker(receiver))
            {
                Ok(_) => Some(Mutex::new(sender)),
                Err(error) => {
                    eprintln!("[audio] could not start the denoise profile worker: {error}");
                    None
                }
            }
        });
        worker.as_ref().map(|sender| {
            sender
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone()
        })
    }
}

fn profiles() -> &'static ProfileService {
    static SERVICE: OnceLock<ProfileService> = OnceLock::new();
    SERVICE.get_or_init(|| ProfileService {
        store: Mutex::new(ProfileStore {
            cache: VecDeque::new(),
            in_flight: Vec::new(),
            failed: Vec::new(),
            background: MediaCancelToken::new(),
        }),
        finished: Condvar::new(),
        worker: OnceLock::new(),
    })
}

fn run_profile_worker(jobs: Receiver<ProfileJob>) {
    for job in jobs {
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
        let service = profiles();
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
                store.failed.push(job.key.clone());
            }
        }
        drop(store);
        service.finished.notify_all();
    }
}

/// The noise profile of a clip's whole rendered audio, estimated once from
/// its entire source window (one extra decode) and cached by source version.
/// Waits for a pass another caller already runs for the same clip; `progress`
/// reports this caller's own pass in clip frames.
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
    #[cfg(test)]
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

/// Test seams for the background profile worker: count passes and requests
/// per source path, and hold a path's background pass until released.
#[cfg(test)]
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

    pub(crate) fn ffmpeg_ready() -> bool {
        opentake_media::ffmpeg_status::ffmpeg_available()
            && opentake_media::ffmpeg_status::ffprobe_available()
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::{ffmpeg_ready, noisy_tone, write_wav, RATE};
    use super::*;
    use opentake_domain::DenoiseMode;

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
