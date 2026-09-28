//! Waveform generation: ffmpeg audio decode → streamed mono f32 → RMS
//! downsample → normalized `0=loud, 1=silence` buckets, with an optional
//! `.waveform` disk cache.
//!
//! Replaces upstream's `DSWaveformImage` dependency (`MediaVisualCache.swift`).
//! The count formula and normalization are byte-for-byte intent-compatible; the
//! exact bucket *values* are visually equivalent but not bit-identical to the
//! third-party analyzer (SPEC §4.3 — waveform is a UI affordance, not a
//! frame-level edit quantity).
//!
//! Decoding goes through the same `ffmpeg` CLI backend as probe/thumbnail/PCM
//! extraction (`crate::extract_pcm`) rather than a separate pure-Rust decoder:
//! a dedicated decoder only covered a subset of containers/codecs (e.g. it could
//! not decode the audio track inside many `.mov` files or non-AAC codecs), so a
//! clip whose `media_ref` pointed at such a source rendered with NO waveform
//! while its thumbnail/probe (ffmpeg) worked fine. Sharing one backend makes the
//! waveform succeed for everything ffmpeg can read.

mod dsp;
pub mod store;

pub use dsp::{
    rms_downsample_normalized, waveform_sample_count, BUCKETS_PER_SECOND, MAX_BUCKETS, MIN_BUCKETS,
};

use std::path::Path;

use dsp::RmsBuckets;

use crate::cache_key::visual_file_identity_key;
use crate::decode::pcm::{decode_pcm_streaming, PcmSink};
use crate::error::Result;
use crate::{MediaCancelToken, PcmFormat, PcmSpec};

/// Sample rate for waveform decode. The exact rate is immaterial — the signal is
/// RMS-downsampled to a fixed bucket count derived from duration — so a single
/// modest mono rate keeps the decode cheap.
const WAVEFORM_SAMPLE_RATE: u32 = 22_050;

/// Generate normalized waveform buckets for `path` (no caching).
/// Bucket count follows [`waveform_sample_count`]. Decodes the first audio track
/// to mono f32 via ffmpeg; errors (propagated) when there is no audio track.
pub fn waveform(path: &Path, duration_secs: f64) -> Result<Vec<f32>> {
    waveform_cancellable(path, duration_secs, &MediaCancelToken::new())
}

/// Waveform RMS accumulated straight from the decoder's f32 mono output.
struct WaveformSink(RmsBuckets);

impl PcmSink for WaveformSink {
    fn push_frames(&mut self, frames: &[u8]) -> Result<()> {
        for sample in frames.chunks_exact(4) {
            self.0.push(f32::from_le_bytes([
                sample[0], sample[1], sample[2], sample[3],
            ]));
        }
        Ok(())
    }
}

/// The decoded samples stream into per-bucket RMS sums, so memory stays
/// bounded by the bucket count however long the source is. Buckets are laid
/// out over the requested duration's exact frame budget; see [`RmsBuckets`].
pub fn waveform_cancellable(
    path: &Path,
    duration_secs: f64,
    cancel: &MediaCancelToken,
) -> Result<Vec<f32>> {
    let spec = PcmSpec {
        sample_rate: WAVEFORM_SAMPLE_RATE,
        channels: 1,
        format: PcmFormat::F32,
    };
    let count = waveform_sample_count(duration_secs);
    let sink = decode_pcm_streaming(
        path,
        &spec,
        Some((0.0, duration_secs)),
        cancel,
        None,
        |expected_frames| Ok(WaveformSink(RmsBuckets::new(count, expected_frames))),
    )?;
    Ok(sink.0.finish())
}

/// Compute a cancellable waveform and serialize the cache payload without
/// publishing it. Project-scoped prewarm jobs hand these bytes to their epoch
/// guard for a same-directory staged rename.
pub fn waveform_cache_bytes_cancellable(
    path: &Path,
    duration_secs: f64,
    cancel: &MediaCancelToken,
) -> Result<Vec<u8>> {
    let samples = waveform_cancellable(path, duration_secs, cancel)?;
    if cancel.checkpoint() {
        return Err(crate::MediaError::Cancelled);
    }
    Ok(waveform_cache_bytes(&samples))
}

fn waveform_cache_bytes(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len().saturating_mul(4));
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

/// Like [`waveform`] but reads/writes the `.waveform` disk cache under
/// `<cache_root>/MediaVisualCache/<key>.waveform`.
pub fn waveform_cached(cache_root: &Path, path: &Path, duration_secs: f64) -> Result<Vec<f32>> {
    waveform_cached_cancellable(cache_root, path, duration_secs, &MediaCancelToken::new())
}

pub fn waveform_cached_cancellable(
    cache_root: &Path,
    path: &Path,
    duration_secs: f64,
    cancel: &MediaCancelToken,
) -> Result<Vec<f32>> {
    if let Some(key) = visual_file_identity_key(path) {
        if let Some(cached) = store::load_waveform(cache_root, &key) {
            return Ok(cached);
        }
        let samples = waveform_cancellable(path, duration_secs, cancel)?;
        let _ = store::save_waveform(cache_root, &key, &samples);
        return Ok(samples);
    }
    waveform_cancellable(path, duration_secs, cancel)
}

#[cfg(test)]
mod tests {
    use super::waveform_cache_bytes;

    #[test]
    fn prewarm_waveform_cache_bytes_are_little_endian() {
        assert_eq!(
            waveform_cache_bytes(&[1.0, -0.25]),
            vec![0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0xbe]
        );
    }
}
