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
//! [`PcmStream`]. Export keeps one reader per audible clip for as long as the
//! clip is in range (one decoder per clip, not per window). Preview mixes
//! independent windows and opens a reader per window that starts a denoise
//! warm-up before the window, which converges on the uninterrupted result.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

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

/// The denoiser input for a clip reader: `None` when the clip is not denoised
/// (zero strength is a bit-exact bypass), else the validated settings and the
/// clip's noise profile.
pub(crate) fn clip_denoise(
    config: Option<AudioDenoise>,
    layout: &ClipAudioLayout,
    path: &Path,
    channels: usize,
    cancel: &MediaCancelToken,
) -> Result<Option<(DenoiseProfile, AudioDenoise)>, MediaError> {
    let Some(config) = config.filter(|config| config.strength != 0.0) else {
        return Ok(None);
    };
    config
        .validate()
        .map_err(|error| denoise_error(DenoiseError::InvalidConfig(error.to_string())))?;
    let profile = clip_denoise_profile(layout, path, channels, cancel)?;
    Ok(Some((DenoiseProfile::clone(&profile), config)))
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

static PROFILE_CACHE: Mutex<VecDeque<(ProfileKey, Arc<DenoiseProfile>)>> =
    Mutex::new(VecDeque::new());

/// The noise profile of a clip's whole rendered audio, estimated once from
/// its entire source window (one extra decode) and cached by source version.
pub(crate) fn clip_denoise_profile(
    layout: &ClipAudioLayout,
    path: &Path,
    channels: usize,
    cancel: &MediaCancelToken,
) -> Result<Arc<DenoiseProfile>, MediaError> {
    let metadata = std::fs::metadata(path).ok();
    let key = ProfileKey {
        path: path.to_path_buf(),
        file_len: metadata.as_ref().map_or(0, std::fs::Metadata::len),
        modified: metadata.and_then(|metadata| metadata.modified().ok()),
        source_lo: layout.source_lo.to_bits(),
        source_hi: layout.source_hi.to_bits(),
        ratio: layout.ratio.to_bits(),
        len: layout.len,
        rate: layout.rate,
        channels,
    };
    let cached = PROFILE_CACHE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .iter()
        .find(|(cached, _)| *cached == key)
        .map(|(_, profile)| Arc::clone(profile));
    if let Some(profile) = cached {
        return Ok(profile);
    }
    let mut reader = ClipAudioReader::open(*layout, path, channels, 0, None, cancel)?;
    let mut builder =
        DenoiseProfileBuilder::new(channels, layout.rate, layout.len).map_err(denoise_error)?;
    let mut chunk = Vec::new();
    let mut remaining = layout.len;
    while remaining > 0 {
        let step = RENDER_CHUNK_FRAMES.min(remaining);
        chunk.clear();
        reader.read(step, &mut chunk)?;
        builder.push(&chunk, cancel).map_err(denoise_error)?;
        remaining -= step;
    }
    let profile = Arc::new(builder.finish(cancel).map_err(denoise_error)?);
    let mut cache = PROFILE_CACHE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    cache.retain(|(cached, _)| *cached != key);
    while cache.len() >= PROFILE_CACHE_CAPACITY {
        cache.pop_front();
    }
    cache.push_back((key, Arc::clone(&profile)));
    Ok(profile)
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

            let profile = clip_denoise_profile(&layout, &path, 1, &cancel).unwrap();
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
