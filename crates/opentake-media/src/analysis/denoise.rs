//! Deterministic local STFT noise suppression shared by preview and export.
//!
//! The suppressor estimates one noise spectrum per channel from the whole
//! signal (a [`DenoiseProfile`]) and then filters the signal frame by frame
//! ([`DenoiseStream`]). Splitting the two lets callers process a long clip in
//! windows without the per-window artifacts of re-estimating noise and
//! restarting gain smoothing at every window edge: a stream fed the whole
//! signal in chunks of any size produces exactly what [`denoise_interleaved`]
//! produces for the whole buffer, and a stream started a
//! [`denoise_warmup_frames`]-long pre-roll before a position matches it from
//! that position on.

use std::collections::VecDeque;
use std::sync::Arc;

use opentake_domain::{AudioDenoise, DenoiseMode};
use rustfft::{num_complex::Complex32, Fft, FftPlanner};

use crate::MediaCancelToken;

pub type DenoiseProgressCallback = Arc<dyn Fn(usize, usize) + Send + Sync>;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum DenoiseError {
    #[error("denoise_invalid_config: {0}")]
    InvalidConfig(String),
    #[error("denoise_cancelled")]
    Cancelled,
}

/// Noise-estimate frames sampled from one signal (strided over long signals).
const MAX_NOISE_ESTIMATE_WINDOWS: usize = 512;
/// Hops a restarted stream needs before its temporal gain smoothing agrees
/// with an uninterrupted stream. The smoothing keeps a quarter of the previous
/// gain per hop, so any initial difference shrinks below 0.25^16 (2e-10).
const WARMUP_HOPS: usize = 16;
/// Frames [`denoise_interleaved`] feeds its stream at a time, bounding the
/// stream's input buffer for long signals.
const ONE_SHOT_CHUNK_FRAMES: usize = 64 * 1024;

/// STFT frame length and hop for a sample rate.
fn framing(sample_rate: u32) -> (usize, usize) {
    let frame_len = if sample_rate >= 32_000 { 1_024 } else { 512 };
    (frame_len, frame_len / 4)
}

/// STFT frames the suppressor runs over a signal of `frames` samples.
fn frame_count(frames: usize, frame_len: usize, hop: usize) -> usize {
    if frames <= frame_len {
        1
    } else {
        1 + (frames - 1) / hop
    }
}

fn validate_layout(channels: usize, sample_rate: u32) -> Result<(), DenoiseError> {
    if channels == 0 || channels > 8 || sample_rate < 8_000 {
        return Err(DenoiseError::InvalidConfig(
            "channels, sample rate, or interleaving is unsupported".to_string(),
        ));
    }
    Ok(())
}

fn hann_window(frame_len: usize) -> Vec<f32> {
    (0..frame_len)
        .map(|index| {
            let phase = std::f32::consts::TAU * index as f32 / frame_len as f32;
            0.5 - 0.5 * phase.cos()
        })
        .collect()
}

/// Samples a stream must be fed before `position` so its output from
/// `position` on matches a stream that started at the beginning of the signal
/// (the overlap-add is complete and the gain smoothing has converged).
pub fn denoise_warmup_frames(sample_rate: u32) -> usize {
    let (frame_len, hop) = framing(sample_rate);
    frame_len + WARMUP_HOPS * hop
}

/// The frame a stream should start at to serve `position` exactly: the
/// warm-up before it, aligned down to the STFT hop grid of the signal.
pub fn denoise_stream_start(sample_rate: u32, position: usize) -> usize {
    let (_, hop) = framing(sample_rate);
    let start = position.saturating_sub(denoise_warmup_frames(sample_rate));
    start - start % hop
}

/// Per-channel noise statistics of one whole signal: the noise power of each
/// STFT bin (the 15th percentile over up to 512 frames spread across the
/// signal) and the input peak the output is held to. Depends only on the
/// signal, not on the strength or mode, so one profile serves every config.
#[derive(Clone, Debug, PartialEq)]
pub struct DenoiseProfile {
    sample_rate: u32,
    channels: usize,
    frames: usize,
    noise_power: Vec<Vec<f32>>,
    input_peak: Vec<f32>,
}

impl DenoiseProfile {
    /// Estimate the profile of a whole interleaved signal held in memory.
    pub fn from_interleaved(
        samples: &[f32],
        channels: usize,
        sample_rate: u32,
        cancel: &MediaCancelToken,
    ) -> Result<Self, DenoiseError> {
        if channels == 0 || !samples.len().is_multiple_of(channels) {
            return Err(DenoiseError::InvalidConfig(
                "channels, sample rate, or interleaving is unsupported".to_string(),
            ));
        }
        let mut builder =
            DenoiseProfileBuilder::new(channels, sample_rate, samples.len() / channels)?;
        builder.push(samples, cancel)?;
        builder.finish(cancel)
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Length of the profiled signal in frames (samples per channel).
    pub fn frames(&self) -> usize {
        self.frames
    }
}

/// Builds a [`DenoiseProfile`] from a signal delivered in chunks, holding at
/// most one STFT frame of samples plus the sampled bin powers.
pub struct DenoiseProfileBuilder {
    sample_rate: u32,
    channels: usize,
    frames: usize,
    frame_len: usize,
    windows: usize,
    stride: usize,
    window: Vec<f32>,
    forward: Arc<dyn Fft<f32>>,
    /// Next sampled frame index and the absolute position its samples start at.
    next_frame: usize,
    /// Per channel: samples from the next sampled frame's start onward.
    pending: Vec<Vec<f32>>,
    /// Absolute position of the next pushed frame.
    received: usize,
    powers: Vec<Vec<Vec<f32>>>,
    input_peak: Vec<f32>,
    spectrum: Vec<Complex32>,
}

impl DenoiseProfileBuilder {
    /// Expect a signal of exactly `frames` frames of `channels` interleaved
    /// channels. The frame count fixes how the noise frames are sampled.
    pub fn new(channels: usize, sample_rate: u32, frames: usize) -> Result<Self, DenoiseError> {
        validate_layout(channels, sample_rate)?;
        let (frame_len, hop) = framing(sample_rate);
        let windows = frame_count(frames, frame_len, hop);
        let stride = windows.div_ceil(MAX_NOISE_ESTIMATE_WINDOWS).max(1);
        let bins = frame_len / 2 + 1;
        let sampled = windows.div_ceil(stride);
        let mut planner = FftPlanner::<f32>::new();
        Ok(DenoiseProfileBuilder {
            sample_rate,
            channels,
            frames,
            frame_len,
            windows,
            stride,
            window: hann_window(frame_len),
            forward: planner.plan_fft_forward(frame_len),
            next_frame: 0,
            pending: vec![Vec::with_capacity(frame_len); channels],
            received: 0,
            powers: vec![vec![Vec::with_capacity(sampled); bins]; channels],
            input_peak: vec![0.0; channels],
            spectrum: vec![Complex32::new(0.0, 0.0); frame_len],
        })
    }

    fn hop(&self) -> usize {
        self.frame_len / 4
    }

    fn frame_start(&self) -> usize {
        self.next_frame * self.hop()
    }

    /// Feed the next interleaved samples of the signal.
    pub fn push(&mut self, samples: &[f32], cancel: &MediaCancelToken) -> Result<(), DenoiseError> {
        if !samples.len().is_multiple_of(self.channels) {
            return Err(DenoiseError::InvalidConfig(
                "channels, sample rate, or interleaving is unsupported".to_string(),
            ));
        }
        let frames = samples.len() / self.channels;
        if self.received.saturating_add(frames) > self.frames {
            return Err(DenoiseError::InvalidConfig(
                "denoise profile received more samples than announced".to_string(),
            ));
        }
        for (offset, frame) in samples.chunks_exact(self.channels).enumerate() {
            let position = self.received + offset;
            for (channel, &value) in frame.iter().enumerate() {
                self.input_peak[channel] = self.input_peak[channel].max(value.abs());
                if self.next_frame < self.windows && position >= self.frame_start() {
                    self.pending[channel].push(value);
                }
            }
            if self.next_frame < self.windows && self.pending[0].len() == self.frame_len {
                self.analyze_frame(cancel)?;
            }
        }
        self.received += frames;
        Ok(())
    }

    /// Record the sampled frame held in `pending` and move to the next one.
    fn analyze_frame(&mut self, cancel: &MediaCancelToken) -> Result<(), DenoiseError> {
        if cancel.checkpoint() {
            return Err(DenoiseError::Cancelled);
        }
        let bins = self.frame_len / 2 + 1;
        for channel in 0..self.channels {
            load_window(&self.pending[channel], &self.window, 0, &mut self.spectrum);
            self.forward.process(&mut self.spectrum);
            for bin in 0..bins {
                self.powers[channel][bin].push(self.spectrum[bin].norm_sqr());
            }
        }
        let next = self.next_frame + self.stride;
        let advance = (next - self.next_frame) * self.hop();
        for pending in &mut self.pending {
            pending.drain(..advance.min(pending.len()));
        }
        self.next_frame = next;
        Ok(())
    }

    /// Finish the estimate once the whole signal has been pushed.
    pub fn finish(mut self, cancel: &MediaCancelToken) -> Result<DenoiseProfile, DenoiseError> {
        if self.received != self.frames {
            return Err(DenoiseError::InvalidConfig(format!(
                "denoise profile expected {} frames, received {}",
                self.frames, self.received
            )));
        }
        // Frames reaching past the end of the signal are zero-padded.
        while self.next_frame < self.windows {
            self.analyze_frame(cancel)?;
        }
        let noise_power = self
            .powers
            .into_iter()
            .map(|channel| {
                channel
                    .into_iter()
                    .map(|mut values| {
                        values.sort_by(f32::total_cmp);
                        let index =
                            ((values.len().saturating_sub(1)) as f32 * 0.15).round() as usize;
                        values.get(index).copied().unwrap_or(0.0).max(1.0e-12)
                    })
                    .collect()
            })
            .collect();
        Ok(DenoiseProfile {
            sample_rate: self.sample_rate,
            channels: self.channels,
            frames: self.frames,
            noise_power,
            input_peak: self
                .input_peak
                .into_iter()
                .map(|peak| peak.min(1.0))
                .collect(),
        })
    }
}

/// Frame-by-frame noise suppression of one profiled signal, fed in chunks.
///
/// Output is emitted in order as soon as it is final (up to one STFT frame
/// behind the input) and covers exactly the fed positions. A stream started at
/// the beginning of the signal reproduces [`denoise_interleaved`] bit for bit
/// however the input is chunked. A stream started at a later hop-aligned
/// position (see [`denoise_stream_start`]) matches it from that start plus
/// [`denoise_warmup_frames`] on; its output before that is warm-up.
pub struct DenoiseStream {
    profile: DenoiseProfile,
    bypass: bool,
    oversubtraction: f32,
    floor_gain: f32,
    frame_len: usize,
    hop: usize,
    windows: usize,
    edge_span: usize,
    window: Vec<f32>,
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
    /// Absolute position of the next frame to process (a hop multiple).
    frame_base: usize,
    next_frame: usize,
    /// Absolute position of the next input sample.
    received: usize,
    /// Per channel: raw input from `frame_base` on.
    input: Vec<VecDeque<f32>>,
    /// Per channel: overlap-add sums for `frame_base..frame_base + frame_len`.
    acc: Vec<Vec<f32>>,
    norm: Vec<Vec<f32>>,
    prior_gain: Vec<Vec<f32>>,
    raw_gain: Vec<f32>,
    spectrum: Vec<Complex32>,
}

impl DenoiseStream {
    /// Start filtering the profiled signal at absolute frame `start`, which
    /// must lie on the STFT hop grid ([`denoise_stream_start`]).
    pub fn new(
        profile: DenoiseProfile,
        config: AudioDenoise,
        start: usize,
    ) -> Result<Self, DenoiseError> {
        config
            .validate()
            .map_err(|error| DenoiseError::InvalidConfig(error.to_string()))?;
        validate_layout(profile.channels, profile.sample_rate)?;
        let (frame_len, hop) = framing(profile.sample_rate);
        if !start.is_multiple_of(hop) || start > profile.frames {
            return Err(DenoiseError::InvalidConfig(format!(
                "denoise stream start {start} is not a hop boundary within the signal"
            )));
        }
        let strength = config.strength as f32;
        let oversubtraction = match config.mode {
            DenoiseMode::Adaptive => 1.0 + 4.5 * strength,
            DenoiseMode::Voice => 1.0 + 6.0 * strength,
        };
        let bins = frame_len / 2 + 1;
        let channels = profile.channels;
        let windows = frame_count(profile.frames, frame_len, hop);
        let edge_span = (frame_len / 2).min(profile.frames.saturating_sub(1)).max(1);
        let mut planner = FftPlanner::<f32>::new();
        Ok(DenoiseStream {
            bypass: config.strength == 0.0,
            oversubtraction,
            floor_gain: 1.0 - 0.92 * strength,
            frame_len,
            hop,
            windows,
            edge_span,
            window: hann_window(frame_len),
            forward: planner.plan_fft_forward(frame_len),
            inverse: planner.plan_fft_inverse(frame_len),
            frame_base: start,
            next_frame: start / hop,
            received: start,
            input: vec![VecDeque::with_capacity(frame_len * 2); channels],
            acc: vec![vec![0.0; frame_len]; channels],
            norm: vec![vec![0.0; frame_len]; channels],
            prior_gain: vec![vec![1.0; bins]; channels],
            raw_gain: vec![1.0; bins],
            spectrum: vec![Complex32::new(0.0, 0.0); frame_len],
            profile,
        })
    }

    /// Absolute position of the next input frame.
    pub fn position(&self) -> usize {
        self.received
    }

    /// Feed the next interleaved samples and append every output sample that
    /// became final to `out`. Once the signal's last frame is fed, all of its
    /// remaining output is emitted.
    pub fn push(
        &mut self,
        samples: &[f32],
        out: &mut Vec<f32>,
        cancel: &MediaCancelToken,
    ) -> Result<(), DenoiseError> {
        self.push_with_progress(samples, out, cancel, &mut || {})
    }

    fn push_with_progress(
        &mut self,
        samples: &[f32],
        out: &mut Vec<f32>,
        cancel: &MediaCancelToken,
        on_frame: &mut dyn FnMut(),
    ) -> Result<(), DenoiseError> {
        let channels = self.profile.channels;
        if !samples.len().is_multiple_of(channels) {
            return Err(DenoiseError::InvalidConfig(
                "channels, sample rate, or interleaving is unsupported".to_string(),
            ));
        }
        let frames = samples.len() / channels;
        if self.received.saturating_add(frames) > self.profile.frames {
            return Err(DenoiseError::InvalidConfig(
                "denoise stream received more samples than its profile covers".to_string(),
            ));
        }
        if cancel.checkpoint() {
            return Err(DenoiseError::Cancelled);
        }
        if self.bypass {
            self.received += frames;
            out.extend_from_slice(samples);
            return Ok(());
        }
        for frame in samples.chunks_exact(channels) {
            for (channel, &value) in frame.iter().enumerate() {
                self.input[channel].push_back(value);
            }
        }
        self.received += frames;
        while self.next_frame < self.windows
            && (self.received >= self.frame_base + self.frame_len
                || self.received == self.profile.frames)
        {
            self.process_frame(out, cancel)?;
            on_frame();
        }
        Ok(())
    }

    /// Filter the frame at `frame_base`, then emit the samples no later frame
    /// overlaps (all remaining samples after the signal's last frame).
    fn process_frame(
        &mut self,
        out: &mut Vec<f32>,
        cancel: &MediaCancelToken,
    ) -> Result<(), DenoiseError> {
        if cancel.checkpoint() {
            return Err(DenoiseError::Cancelled);
        }
        let frame_len = self.frame_len;
        let bins = frame_len / 2 + 1;
        let channels = self.profile.channels;
        for channel in 0..channels {
            for (index, complex) in self.spectrum.iter_mut().enumerate() {
                let value = self.input[channel].get(index).copied().unwrap_or(0.0);
                *complex = Complex32::new(value * self.window[index], 0.0);
            }
            self.forward.process(&mut self.spectrum);
            let noise_power = &self.profile.noise_power[channel];
            for ((raw_gain, bin), noise) in self
                .raw_gain
                .iter_mut()
                .zip(&self.spectrum[..bins])
                .zip(noise_power)
            {
                let power = bin.norm_sqr().max(1.0e-12);
                let clean_ratio = (1.0 - self.oversubtraction * noise / power).max(0.0);
                *raw_gain = clean_ratio.sqrt().max(self.floor_gain);
            }
            for (bin, prior_gain) in self.prior_gain[channel].iter_mut().enumerate() {
                let lo = bin.saturating_sub(1);
                let hi = (bin + 1).min(bins - 1);
                let frequency_smoothed =
                    self.raw_gain[lo..=hi].iter().sum::<f32>() / (hi - lo + 1) as f32;
                let gain =
                    (*prior_gain * 0.25 + frequency_smoothed * 0.75).clamp(self.floor_gain, 1.0);
                *prior_gain = gain;
                self.spectrum[bin] *= gain;
                if bin > 0 && bin < frame_len / 2 {
                    self.spectrum[frame_len - bin] *= gain;
                }
            }
            self.inverse.process(&mut self.spectrum);
            let acc = &mut self.acc[channel];
            let norm = &mut self.norm[channel];
            for index in 0..frame_len {
                if self.frame_base + index >= self.profile.frames {
                    break;
                }
                let weight = self.window[index];
                acc[index] += self.spectrum[index].re / frame_len as f32 * weight;
                norm[index] += weight * weight;
            }
        }
        self.next_frame += 1;
        let last = self.next_frame == self.windows;
        let settled = if last {
            self.profile.frames - self.frame_base
        } else {
            self.hop
        };
        self.emit(settled, out);
        if !last {
            let hop = self.hop;
            for channel in 0..channels {
                let consumed = hop.min(self.input[channel].len());
                self.input[channel].drain(..consumed);
                self.acc[channel].copy_within(hop.., 0);
                self.acc[channel][frame_len - hop..].fill(0.0);
                self.norm[channel].copy_within(hop.., 0);
                self.norm[channel][frame_len - hop..].fill(0.0);
            }
            self.frame_base += hop;
        }
        Ok(())
    }

    /// Normalize, edge-blend and append the first `count` settled samples.
    fn emit(&self, count: usize, out: &mut Vec<f32>) {
        let frames = self.profile.frames;
        let channels = self.profile.channels;
        out.reserve(count * channels);
        for offset in 0..count {
            let position = self.frame_base + offset;
            // A centered STFT would normally pad both ends before analysis.
            // Keep the implementation allocation-bounded by crossfading the
            // unpadded edge into the processed signal instead. The peak guard
            // prevents low Hann-normalization weights from creating a click or
            // a new peak.
            let edge_distance = position.min(frames - 1 - position);
            let processed_mix = (edge_distance as f32 / self.edge_span as f32).min(1.0);
            for channel in 0..channels {
                let sample = self.input[channel][offset];
                let weight = self.norm[channel][offset];
                let normalized = if weight > 1.0e-6 {
                    self.acc[channel][offset] / weight
                } else {
                    sample
                };
                let peak = self.profile.input_peak[channel];
                let value = (sample * (1.0 - processed_mix) + normalized * processed_mix)
                    .clamp(-peak, peak);
                out.push(value.clamp(-1.0, 1.0));
            }
        }
    }
}

/// Process interleaved PCM without mutating the source. A zero-strength config
/// is a bit-exact bypass. Each channel is transformed independently so stereo
/// placement is preserved, while all call sites share identical parameters and
/// math. Equivalent to profiling the whole buffer ([`DenoiseProfile`]) and
/// running one [`DenoiseStream`] over it.
pub fn denoise_interleaved(
    samples: &[f32],
    channels: usize,
    sample_rate: u32,
    config: AudioDenoise,
    cancel: &MediaCancelToken,
    progress: Option<DenoiseProgressCallback>,
) -> Result<Vec<f32>, DenoiseError> {
    config
        .validate()
        .map_err(|error| DenoiseError::InvalidConfig(error.to_string()))?;
    if channels == 0
        || channels > 8
        || sample_rate < 8_000
        || !samples.len().is_multiple_of(channels)
    {
        return Err(DenoiseError::InvalidConfig(
            "channels, sample rate, or interleaving is unsupported".to_string(),
        ));
    }
    if cancel.checkpoint() {
        return Err(DenoiseError::Cancelled);
    }
    if samples.is_empty() || config.strength == 0.0 {
        return Ok(samples.to_vec());
    }

    let (frame_len, hop) = framing(sample_rate);
    let windows = frame_count(samples.len() / channels, frame_len, hop);
    let total_steps = windows.saturating_mul(2).max(1);
    let profile = DenoiseProfile::from_interleaved(samples, channels, sample_rate, cancel)?;
    if let Some(report) = &progress {
        report(windows.min(total_steps), total_steps);
    }
    let mut completed = windows;
    let mut stream = DenoiseStream::new(profile, config, 0)?;
    let mut output = Vec::with_capacity(samples.len());
    for chunk in samples.chunks(ONE_SHOT_CHUNK_FRAMES * channels) {
        stream.push_with_progress(chunk, &mut output, cancel, &mut || {
            completed = completed.saturating_add(1);
            if let Some(report) = &progress {
                report(completed.min(total_steps), total_steps);
            }
        })?;
    }
    if let Some(report) = progress {
        report(total_steps, total_steps);
    }
    Ok(output)
}

fn load_window(samples: &[f32], window: &[f32], start: usize, target: &mut [Complex32]) {
    for (index, complex) in target.iter_mut().enumerate() {
        let value = samples.get(start + index).copied().unwrap_or(0.0);
        *complex = Complex32::new(value * window[index], 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48_000;

    fn config(strength: f64) -> AudioDenoise {
        AudioDenoise {
            mode: DenoiseMode::Voice,
            strength,
            preview_enabled: true,
        }
    }

    /// Steady white noise plus a sine, interleaved over `channels`.
    fn noisy_sine(frames: usize, channels: usize, sample_rate: u32) -> Vec<f32> {
        let mut state = 0x1234_5678_u32;
        let mut samples = Vec::with_capacity(frames * channels);
        for index in 0..frames {
            let time = index as f32 / sample_rate as f32;
            let tone = (std::f32::consts::TAU * 440.0 * time).sin() * 0.3;
            for channel in 0..channels {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let white = (state as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32;
                samples.push(tone * (1.0 - 0.3 * channel as f32) + white * 0.05);
            }
        }
        samples
    }

    fn stream_in_chunks(
        samples: &[f32],
        channels: usize,
        sample_rate: u32,
        denoise: AudioDenoise,
        chunk_frames: usize,
    ) -> Vec<f32> {
        let cancel = MediaCancelToken::new();
        let mut builder =
            DenoiseProfileBuilder::new(channels, sample_rate, samples.len() / channels).unwrap();
        for chunk in samples.chunks(chunk_frames * channels) {
            builder.push(chunk, &cancel).unwrap();
        }
        let profile = builder.finish(&cancel).unwrap();
        let mut stream = DenoiseStream::new(profile, denoise, 0).unwrap();
        let mut out = Vec::new();
        for chunk in samples.chunks(chunk_frames * channels) {
            stream.push(chunk, &mut out, &cancel).unwrap();
        }
        out
    }

    #[test]
    fn stream_refactor_reproduces_the_previous_one_shot_bit_for_bit() {
        let cancel = MediaCancelToken::new();
        // Short (one frame), exact frame, strided noise estimate (over 512
        // frames), stereo, and the 512-sample framing below 32 kHz.
        for (frames, channels, sample_rate) in [
            (700, 1, RATE),
            (1_024, 1, RATE),
            (150_000, 1, RATE),
            (20_000, 2, RATE),
            (9_000, 1, 16_000),
        ] {
            let samples = noisy_sine(frames, channels, sample_rate);
            for denoise in [
                config(0.8),
                AudioDenoise {
                    mode: DenoiseMode::Adaptive,
                    ..config(0.35)
                },
            ] {
                let expected = reference_denoise_interleaved(
                    &samples,
                    channels,
                    sample_rate,
                    denoise,
                    &cancel,
                    None,
                )
                .unwrap();
                let actual =
                    denoise_interleaved(&samples, channels, sample_rate, denoise, &cancel, None)
                        .unwrap();
                assert!(
                    actual == expected,
                    "{frames} frames x {channels} ch @ {sample_rate} Hz diverged"
                );
            }
        }
    }

    #[test]
    fn streaming_two_second_windows_matches_whole_signal_processing() {
        // Six seconds of steady white noise plus a sine, streamed in the
        // two-second windows preview and export mix in.
        let samples = noisy_sine(6 * RATE as usize, 1, RATE);
        let denoise = config(0.9);
        let whole = denoise_interleaved(&samples, 1, RATE, denoise, &MediaCancelToken::new(), None)
            .unwrap();
        let streamed = stream_in_chunks(&samples, 1, RATE, denoise, 2 * RATE as usize);
        assert_eq!(streamed.len(), whole.len());
        let max_difference = streamed
            .iter()
            .zip(&whole)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_difference < 1.0e-3, "max difference {max_difference}");
        assert_eq!(streamed, whole, "chunking never changes the result");

        // Residual noise (output minus the clean tone) per 50 ms block: the
        // blocks at the 2 s and 4 s window boundaries look like every other
        // block instead of carrying an unsuppressed burst.
        let block = RATE as usize / 20;
        let residual_db = |start: usize| {
            let energy = (start..start + block)
                .map(|index| {
                    let time = index as f32 / RATE as f32;
                    let clean = (std::f32::consts::TAU * 440.0 * time).sin() * 0.3;
                    f64::from(streamed[index] - clean).powi(2)
                })
                .sum::<f64>();
            10.0 * (energy / block as f64).log10()
        };
        let interior = (2..118)
            .map(|index| residual_db(index * block))
            .collect::<Vec<_>>();
        let mut sorted = interior.clone();
        sorted.sort_by(f64::total_cmp);
        let median = sorted[sorted.len() / 2];
        for boundary in [2 * RATE as usize, 4 * RATE as usize] {
            for start in [boundary - block, boundary - block / 2, boundary] {
                let level = residual_db(start);
                assert!(
                    (level - median).abs() < 0.5,
                    "residual at {start} is {level:.2} dB vs median {median:.2} dB"
                );
            }
        }
    }

    #[test]
    fn a_stream_started_after_warmup_matches_the_uninterrupted_stream() {
        let samples = noisy_sine(4 * RATE as usize, 2, RATE);
        let denoise = config(0.7);
        let cancel = MediaCancelToken::new();
        let whole = denoise_interleaved(&samples, 2, RATE, denoise, &cancel, None).unwrap();
        let profile = DenoiseProfile::from_interleaved(&samples, 2, RATE, &cancel).unwrap();
        for position in [1_000, 48_000, 100_003, 4 * RATE as usize - 700] {
            let start = denoise_stream_start(RATE, position);
            assert!(
                start <= position && position - start >= denoise_warmup_frames(RATE).min(position)
            );
            let mut stream = DenoiseStream::new(profile.clone(), denoise, start).unwrap();
            let mut out = Vec::new();
            for chunk in samples[start * 2..].chunks(7_777 * 2) {
                stream.push(chunk, &mut out, &cancel).unwrap();
            }
            assert_eq!(out.len(), samples.len() - start * 2);
            let from = (position - start) * 2;
            let max_difference = out[from..]
                .iter()
                .zip(&whole[position * 2..])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                max_difference < 1.0e-6,
                "restart for {position} differs by {max_difference}"
            );
        }
    }

    #[test]
    fn zero_strength_stream_is_a_bit_exact_bypass() {
        let samples = noisy_sine(5_000, 1, RATE);
        let bypass = stream_in_chunks(&samples, 1, RATE, config(0.0), 333);
        assert_eq!(bypass, samples);
    }

    #[test]
    fn stream_rejects_off_grid_starts_and_overlong_input() {
        let samples = noisy_sine(4_000, 1, RATE);
        let cancel = MediaCancelToken::new();
        let profile = DenoiseProfile::from_interleaved(&samples, 1, RATE, &cancel).unwrap();
        assert!(DenoiseStream::new(profile.clone(), config(0.5), 100).is_err());
        let mut stream = DenoiseStream::new(profile, config(0.5), 256).unwrap();
        let mut out = Vec::new();
        assert!(stream.push(&samples, &mut out, &cancel).is_err());
        let mut builder = DenoiseProfileBuilder::new(1, RATE, 10).unwrap();
        assert!(builder.push(&samples, &cancel).is_err());
    }

    // The single-pass implementation this module replaced, kept to pin the
    // streaming refactor to the exact previous output.
    /// Process interleaved PCM without mutating the source. A zero-strength config
    /// is a bit-exact bypass. Each channel is transformed independently so stereo
    /// placement is preserved, while all call sites share identical parameters and
    /// math.
    fn reference_denoise_interleaved(
        samples: &[f32],
        channels: usize,
        sample_rate: u32,
        config: AudioDenoise,
        cancel: &MediaCancelToken,
        progress: Option<DenoiseProgressCallback>,
    ) -> Result<Vec<f32>, DenoiseError> {
        config
            .validate()
            .map_err(|error| DenoiseError::InvalidConfig(error.to_string()))?;
        if channels == 0
            || channels > 8
            || sample_rate < 8_000
            || !samples.len().is_multiple_of(channels)
        {
            return Err(DenoiseError::InvalidConfig(
                "channels, sample rate, or interleaving is unsupported".to_string(),
            ));
        }
        if cancel.checkpoint() {
            return Err(DenoiseError::Cancelled);
        }
        if samples.is_empty() || config.strength == 0.0 {
            return Ok(samples.to_vec());
        }

        let frame_len = if sample_rate >= 32_000 { 1_024 } else { 512 };
        let hop = frame_len / 4;
        let audio_frames = samples.len() / channels;
        let windows = if audio_frames <= frame_len {
            1
        } else {
            1 + (audio_frames - 1) / hop
        };
        let total_steps = channels.saturating_mul(windows).saturating_mul(2).max(1);
        let mut completed = 0usize;

        let window = (0..frame_len)
            .map(|index| {
                let phase = std::f32::consts::TAU * index as f32 / frame_len as f32;
                0.5 - 0.5 * phase.cos()
            })
            .collect::<Vec<_>>();
        let mut planner = FftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(frame_len);
        let inverse = planner.plan_fft_inverse(frame_len);
        let mut output = vec![0.0_f32; samples.len()];

        for channel in 0..channels {
            let mono = samples
                .iter()
                .skip(channel)
                .step_by(channels)
                .copied()
                .collect::<Vec<_>>();
            let processed = reference_process_channel(
                &mono,
                &window,
                hop,
                windows,
                config,
                cancel,
                &progress,
                total_steps,
                &mut completed,
                &forward,
                &inverse,
            )?;
            for (frame, value) in processed.into_iter().enumerate() {
                output[frame * channels + channel] = value.clamp(-1.0, 1.0);
            }
        }

        if let Some(report) = progress {
            report(total_steps, total_steps);
        }
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn reference_process_channel(
        samples: &[f32],
        window: &[f32],
        hop: usize,
        windows: usize,
        config: AudioDenoise,
        cancel: &MediaCancelToken,
        progress: &Option<DenoiseProgressCallback>,
        total_steps: usize,
        completed: &mut usize,
        forward: &Arc<dyn Fft<f32>>,
        inverse: &Arc<dyn Fft<f32>>,
    ) -> Result<Vec<f32>, DenoiseError> {
        let frame_len = window.len();
        let bins = frame_len / 2 + 1;
        const MAX_NOISE_ESTIMATE_WINDOWS: usize = 512;
        let estimate_stride = windows.div_ceil(MAX_NOISE_ESTIMATE_WINDOWS).max(1);
        let mut powers = (0..bins)
            .map(|_| Vec::with_capacity(windows.min(MAX_NOISE_ESTIMATE_WINDOWS)))
            .collect::<Vec<_>>();
        let mut spectrum = vec![Complex32::new(0.0, 0.0); frame_len];

        for frame_index in 0..windows {
            if cancel.checkpoint() {
                return Err(DenoiseError::Cancelled);
            }
            reference_load_window(samples, window, frame_index * hop, &mut spectrum);
            forward.process(&mut spectrum);
            if frame_index.is_multiple_of(estimate_stride) {
                for bin in 0..bins {
                    powers[bin].push(spectrum[bin].norm_sqr());
                }
            }
            reference_report_step(progress, total_steps, completed);
        }

        let noise_power = powers
            .into_iter()
            .map(|mut values| {
                values.sort_by(f32::total_cmp);
                let index = ((values.len().saturating_sub(1)) as f32 * 0.15).round() as usize;
                values[index].max(1.0e-12)
            })
            .collect::<Vec<_>>();
        let strength = config.strength as f32;
        let oversubtraction = match config.mode {
            DenoiseMode::Adaptive => 1.0 + 4.5 * strength,
            DenoiseMode::Voice => 1.0 + 6.0 * strength,
        };
        let floor_gain = 1.0 - 0.92 * strength;
        let mut prior_gain = vec![1.0_f32; bins];
        let mut raw_gain = vec![1.0_f32; bins];
        let mut out = vec![0.0_f32; samples.len()];
        let mut norm = vec![0.0_f32; samples.len()];

        for frame_index in 0..windows {
            if cancel.checkpoint() {
                return Err(DenoiseError::Cancelled);
            }
            let start = frame_index * hop;
            reference_load_window(samples, window, start, &mut spectrum);
            forward.process(&mut spectrum);
            for bin in 0..bins {
                let power = spectrum[bin].norm_sqr().max(1.0e-12);
                let clean_ratio = (1.0 - oversubtraction * noise_power[bin] / power).max(0.0);
                raw_gain[bin] = clean_ratio.sqrt().max(floor_gain);
            }
            for bin in 0..bins {
                let lo = bin.saturating_sub(1);
                let hi = (bin + 1).min(bins - 1);
                let frequency_smoothed =
                    raw_gain[lo..=hi].iter().sum::<f32>() / (hi - lo + 1) as f32;
                let gain =
                    (prior_gain[bin] * 0.25 + frequency_smoothed * 0.75).clamp(floor_gain, 1.0);
                prior_gain[bin] = gain;
                spectrum[bin] *= gain;
                if bin > 0 && bin < frame_len / 2 {
                    spectrum[frame_len - bin] *= gain;
                }
            }
            inverse.process(&mut spectrum);
            for index in 0..frame_len {
                let output_index = start + index;
                if output_index >= out.len() {
                    break;
                }
                let weight = window[index];
                out[output_index] += spectrum[index].re / frame_len as f32 * weight;
                norm[output_index] += weight * weight;
            }
            reference_report_step(progress, total_steps, completed);
        }

        let input_peak = samples
            .iter()
            .map(|sample| sample.abs())
            .fold(0.0_f32, f32::max)
            .min(1.0);
        let edge_span = (frame_len / 2).min(samples.len().saturating_sub(1)).max(1);
        for (index, (value, weight)) in out.iter_mut().zip(norm).enumerate() {
            let normalized = if weight > 1.0e-6 {
                *value / weight
            } else {
                samples[index]
            };
            // A centered STFT would normally pad both ends before analysis. Keep
            // the implementation allocation-bounded by crossfading the unpadded
            // edge into the processed signal instead. The peak guard prevents
            // low Hann-normalization weights from creating a click or a new peak.
            let edge_distance = index.min(samples.len() - 1 - index);
            let processed_mix = (edge_distance as f32 / edge_span as f32).min(1.0);
            *value = (samples[index] * (1.0 - processed_mix) + normalized * processed_mix)
                .clamp(-input_peak, input_peak);
        }
        Ok(out)
    }

    fn reference_load_window(
        samples: &[f32],
        window: &[f32],
        start: usize,
        target: &mut [Complex32],
    ) {
        for (index, complex) in target.iter_mut().enumerate() {
            let value = samples.get(start + index).copied().unwrap_or(0.0);
            *complex = Complex32::new(value * window[index], 0.0);
        }
    }

    fn reference_report_step(
        progress: &Option<DenoiseProgressCallback>,
        total: usize,
        completed: &mut usize,
    ) {
        *completed = completed.saturating_add(1);
        if let Some(report) = progress {
            report((*completed).min(total), total);
        }
    }
}
