//! Shared JPEG encoder for preview frames (#9, #14).
//!
//! Playback frames and paused still frames both paint onto an opaque canvas, so
//! they are JPEG-encoded straight from the compositor's RGBA readback. The
//! `jpeg-encoder` crate reads RGBA directly (alpha ignored) and has an AVX2 path,
//! which avoids the per-frame RGBA→RGB repack and runs several times faster
//! than `image`'s baseline encoder at the same quality.

use opentake_render::DecodedFrame;

/// JPEG quality for preview frames (0–100). 75 is visually clean for a preview
/// while keeping each frame small enough for a 30–60 fps loopback transport.
pub const PREVIEW_JPEG_QUALITY: u8 = 75;

/// Encode `frame` as a baseline JPEG, appending to `out`. The caller owns `out`
/// so a long-lived encoder can reuse its allocation across frames.
pub fn encode_rgba_jpeg(frame: &DecodedFrame, out: &mut Vec<u8>) -> Result<(), String> {
    let width = u16::try_from(frame.width)
        .map_err(|_| format!("jpeg width {} exceeds 65535", frame.width))?;
    let height = u16::try_from(frame.height)
        .map_err(|_| format!("jpeg height {} exceeds 65535", frame.height))?;
    jpeg_encoder::Encoder::new(out, PREVIEW_JPEG_QUALITY)
        .encode(&frame.rgba, width, height, jpeg_encoder::ColorType::Rgba)
        .map_err(|error| format!("jpeg encode: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// Gradient plus deterministic noise: a worst case for DCT coding that is
    /// closer to real footage than a flat colour.
    fn noisy_frame(width: u32, height: u32) -> DecodedFrame {
        frame_with_noise(width, height, 0x3f)
    }

    fn frame_with_noise(width: u32, height: u32, noise_mask: u32) -> DecodedFrame {
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        let mut seed = 0x2545_f491_u32;
        for y in 0..height {
            for x in 0..width {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                let noise = (seed & noise_mask) as u8;
                rgba.extend_from_slice(&[
                    ((x * 255 / width) as u8).wrapping_add(noise),
                    ((y * 255 / height) as u8).wrapping_add(noise / 2),
                    (((x + y) * 255 / (width + height)) as u8).wrapping_sub(noise),
                    255,
                ]);
            }
        }
        DecodedFrame::new(width, height, rgba, false)
    }

    /// The encoder this module replaced: `image` 0.25 baseline JPEG after an
    /// RGBA→RGB repack. Kept only as the "before" side of the benchmark.
    fn encode_with_image_crate(frame: &DecodedFrame) -> Vec<u8> {
        let mut rgb = Vec::with_capacity((frame.width * frame.height * 3) as usize);
        for px in frame.rgba.as_chunks::<4>().0 {
            rgb.extend_from_slice(&px[..3]);
        }
        let mut out = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, PREVIEW_JPEG_QUALITY)
            .encode(
                &rgb,
                frame.width,
                frame.height,
                image::ExtendedColorType::Rgb8,
            )
            .expect("image crate jpeg encode");
        out
    }

    #[test]
    fn encodes_a_decodable_jpeg_with_the_source_dimensions() {
        let frame = noisy_frame(33, 17);
        let mut out = Vec::new();
        encode_rgba_jpeg(&frame, &mut out).expect("encode");
        assert_eq!(&out[..2], &[0xFF, 0xD8], "JPEG SOI marker");
        let decoded = image::load_from_memory(&out).expect("decode jpeg");
        assert_eq!((decoded.width(), decoded.height()), (33, 17));
    }

    #[test]
    fn reused_output_buffer_is_appended_to_not_reallocated() {
        let frame = noisy_frame(64, 36);
        let mut out = Vec::new();
        encode_rgba_jpeg(&frame, &mut out).expect("first encode");
        let first = out.clone();
        let capacity = out.capacity();
        out.clear();
        encode_rgba_jpeg(&frame, &mut out).expect("second encode");
        assert_eq!(out, first, "encoding is deterministic");
        assert_eq!(out.capacity(), capacity, "the cleared buffer is reused");
    }

    #[test]
    fn oversized_frames_are_rejected_instead_of_truncated() {
        let frame = DecodedFrame::new(70_000, 1, vec![0; 70_000 * 4], false);
        assert!(encode_rgba_jpeg(&frame, &mut Vec::new())
            .expect_err("width above u16")
            .contains("65535"));
    }

    fn median(mut samples: Vec<Duration>) -> Duration {
        samples.sort();
        samples[samples.len() / 2]
    }

    /// Release benchmark for issues #9 and #14. Run with
    /// `cargo test --release -p opentake-tauri --lib jpeg::tests::bench -- --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark; run explicitly in release"]
    fn bench_preview_jpeg_encoders() {
        for (content, noise_mask, width, height) in [
            ("noisy", 0x3f, 1280, 720),
            ("smooth", 0x03, 1280, 720),
            ("noisy", 0x3f, 1920, 1080),
            ("smooth", 0x03, 1920, 1080),
        ] {
            let frame = frame_with_noise(width, height, noise_mask);
            let mut before = Vec::new();
            let mut after = Vec::new();
            let mut before_len = 0;
            let mut after_len = 0;
            let mut out = Vec::new();
            for _ in 0..15 {
                let start = Instant::now();
                before_len = encode_with_image_crate(&frame).len();
                before.push(start.elapsed());

                out.clear();
                let start = Instant::now();
                encode_rgba_jpeg(&frame, &mut out).expect("encode");
                after.push(start.elapsed());
                after_len = out.len();
            }
            println!(
                "{width}x{height} {content}: image 0.25 {:.1} ms / {} KB -> jpeg-encoder {:.1} ms / {} KB",
                median(before).as_secs_f64() * 1e3,
                before_len / 1024,
                median(after).as_secs_f64() * 1e3,
                after_len / 1024,
            );
        }
    }
}
