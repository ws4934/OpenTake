//! Interleaved (multi-channel) PCM decode for streaming-playback audio
//! (#160 / #63).
//!
//! `pcm.rs`'s [`extract_pcm`](crate::decode::extract_pcm) averages every channel
//! into a mono f32 view (right for transcription/waveform). Playback wants real
//! stereo, so this decodes the audio track and keeps the channels **interleaved**
//! — it deliberately does NOT reuse `raw_to_mono_f32`.
//!
//! This is the one-shot preload form (decode a clip's window up front); the
//! chunked / background streaming form is the remaining half of #160.

#[cfg(test)]
use std::ffi::OsString;
use std::path::Path;

use crate::cancel::MediaCancelToken;
#[cfg(test)]
use crate::decode::pcm::PcmFormat;
use crate::decode::pcm::{decode_pcm_streaming, InterleavedF32Sink, PcmSpec};
use crate::error::Result;

/// Build the ffmpeg args to decode the first audio track to raw interleaved PCM
/// on stdout, honoring an optional `[lo, hi)` absolute-seconds range. Mirrors
/// `pcm::pcm_args` but is kept self-contained (no shared mono path).
#[cfg(test)]
fn interleaved_args(path: &Path, spec: &PcmSpec, range: Option<(f64, f64)>) -> Vec<OsString> {
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
    args.push(
        match spec.format {
            PcmFormat::F32 => "f32le",
            PcmFormat::S16Le => "s16le",
        }
        .into(),
    );
    args.push("-".into());
    args
}

/// Decode `path`'s first audio track to interleaved f32 at the requested spec
/// (channels preserved). `range` is an absolute-seconds `[lo, hi)` window. Errors
/// with `NoTrack("audio", …)` when the file has no audio stream.
pub fn decode_pcm_interleaved(
    path: &Path,
    spec: &PcmSpec,
    range: Option<(f64, f64)>,
) -> Result<Vec<f32>> {
    decode_pcm_interleaved_cancellable(path, spec, range, &MediaCancelToken::new())
}

/// Channels stay interleaved and unfolded (the playback mixer pans/sums per
/// channel later). Samples are converted as FFmpeg streams them, so only the
/// f32 result is ever held, not a raw byte copy of the track beside it.
pub fn decode_pcm_interleaved_cancellable(
    path: &Path,
    spec: &PcmSpec,
    range: Option<(f64, f64)>,
    cancel: &MediaCancelToken,
) -> Result<Vec<f32>> {
    let channels = usize::from(spec.channels);
    let sink = decode_pcm_streaming(path, spec, range, cancel, None, |frames| {
        InterleavedF32Sink::with_capacity(spec.format, frames.saturating_mul(channels))
    })?;
    Ok(sink.into_samples())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(channels: u16, format: PcmFormat) -> PcmSpec {
        PcmSpec {
            sample_rate: 48_000,
            channels,
            format,
        }
    }

    #[test]
    fn args_request_interleaved_stereo_f32le_with_range() {
        let args = interleaved_args(
            Path::new("/a.mp4"),
            &spec(2, PcmFormat::F32),
            Some((1.0, 2.0)),
        );
        assert!(args.windows(2).any(|w| w == ["-ac", "2"]));
        assert!(args.windows(2).any(|w| w == ["-ar", "48000"]));
        assert!(args.windows(2).any(|w| w == ["-f", "f32le"]));
        assert!(args.iter().any(|a| a == "-vn"));
        let ss = args.iter().position(|a| a == "-ss").unwrap();
        assert_eq!(args[ss + 1], "1.000000");
        let to = args.iter().position(|a| a == "-to").unwrap();
        assert_eq!(args[to + 1], "2.000000");
        assert_eq!(args.last().unwrap(), "-");
    }

    #[test]
    fn args_have_no_seek_without_range() {
        let args = interleaved_args(Path::new("/a.mp4"), &spec(2, PcmFormat::F32), None);
        assert!(!args.iter().any(|a| a == "-ss"));
    }

    /// Interleaved f32 of `bytes` streamed through the decode's framing.
    fn interleaved(bytes: &[u8], spec: &PcmSpec) -> Vec<f32> {
        let frame_bytes = spec.format.bytes_per_sample() * usize::from(spec.channels);
        let mut sink = InterleavedF32Sink::with_capacity(spec.format, 0).unwrap();
        let mut aligner = crate::decode::pcm::FrameAligner::new(frame_bytes);
        for read in bytes.chunks(3) {
            aligner.feed(read, &mut sink).unwrap();
        }
        sink.into_samples()
    }

    #[test]
    fn f32_interleaved_keeps_channels_unfolded() {
        // Stereo: (L=1.0 R=-1.0), (L=0.5 R=0.0) — NOT averaged to mono.
        let mut bytes = Vec::new();
        for v in [1.0f32, -1.0, 0.5, 0.0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let out = interleaved(&bytes, &spec(2, PcmFormat::F32));
        assert_eq!(out, vec![1.0, -1.0, 0.5, 0.0]);
    }

    #[test]
    fn s16_interleaved_converts_to_unit_floats_unfolded() {
        let mut bytes = Vec::new();
        for v in [0i16, 16384, -32768, 0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let out = interleaved(&bytes, &spec(2, PcmFormat::S16Le));
        assert_eq!(out.len(), 4);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-3);
        assert!((out[2] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn trailing_partial_sample_is_ignored() {
        // 5 bytes of f32 = 1 full sample + 1 stray byte → 1 sample.
        let out = interleaved(&[0, 0, 0, 63, 7], &spec(1, PcmFormat::F32));
        assert_eq!(out.len(), 1);
    }
}
