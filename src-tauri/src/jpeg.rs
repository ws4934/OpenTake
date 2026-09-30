//! Shared JPEG encoder for preview frames (#9, #14).
//!
//! Playback frames and paused still frames both paint onto an opaque canvas, so
//! they are JPEG-encoded straight from the compositor's RGBA readback. The
//! TurboJPEG reads RGBA directly (alpha ignored) and uses native SIMD on both
//! x86 and ARM. Each encoding thread retains one compressor; there is no shared
//! process-wide encoder lock or per-frame RGBA→RGB repack.

use opentake_render::DecodedFrame;
use std::cell::RefCell;

thread_local! {
    static ENCODER: RefCell<Option<turbojpeg::Compressor>> = const { RefCell::new(None) };
}

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
    if width == 0 || height == 0 {
        return Err("jpeg dimensions must be nonzero".into());
    }
    let width = usize::from(width);
    let height = usize::from(height);
    let pitch = width * 4;
    let expected = pitch.checked_mul(height).ok_or("jpeg RGBA size overflow")?;
    if frame.rgba.len() != expected {
        return Err("jpeg RGBA length does not match its dimensions".into());
    }
    let bound = turbojpeg::compressed_buf_len(width, height, turbojpeg::Subsamp::Sub2x2)
        .map_err(|error| format!("jpeg buffer size: {error}"))?;
    let offset = out.len();
    let required = offset
        .checked_add(bound)
        .ok_or("jpeg output size overflow")?;
    ENCODER.with(|cached| {
        let mut cached = cached
            .try_borrow_mut()
            .map_err(|_| "jpeg encoder is already in use")?;
        if cached.is_none() {
            let mut encoder =
                turbojpeg::Compressor::new().map_err(|error| format!("jpeg init: {error}"))?;
            encoder
                .set_quality(i32::from(PREVIEW_JPEG_QUALITY))
                .map_err(|error| format!("jpeg quality: {error}"))?;
            encoder
                .set_subsamp(turbojpeg::Subsamp::Sub2x2)
                .map_err(|error| format!("jpeg sampling: {error}"))?;
            *cached = Some(encoder);
        }
        // A borrowed, conservatively sized buffer cannot be reallocated by the
        // native encoder. Rust owns and reuses the allocation across frames.
        out.resize(required, 0);
        let encoded = cached
            .as_mut()
            .expect("encoder initialized")
            .compress_to_slice(
                turbojpeg::Image {
                    pixels: &frame.rgba,
                    width,
                    height,
                    pitch,
                    format: turbojpeg::PixelFormat::RGBA,
                },
                &mut out[offset..],
            );
        match encoded {
            Ok(len) => {
                out.truncate(offset + len);
                Ok(())
            }
            Err(error) => {
                out.truncate(offset);
                Err(format!("jpeg encode: {error}"))
            }
        }
    })
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

    #[test]
    fn malformed_frame_is_rejected_without_panicking_or_changing_output() {
        let frame = DecodedFrame {
            width: 2,
            height: 1,
            rgba: vec![0; 7],
            premultiplied: false,
        };
        let mut out = b"prefix".to_vec();
        assert!(encode_rgba_jpeg(&frame, &mut out)
            .unwrap_err()
            .contains("RGBA length"));
        assert_eq!(out, b"prefix");
    }

    #[test]
    fn reused_native_encoder_handles_alternating_dimensions_and_keeps_prefix() {
        for (width, height) in [(33, 17), (128, 72), (7, 19), (33, 17)] {
            let frame = noisy_frame(width, height);
            let mut out = b"prefix".to_vec();
            encode_rgba_jpeg(&frame, &mut out).unwrap();
            assert_eq!(&out[..6], b"prefix");
            let decoded = image::load_from_memory(&out[6..]).unwrap();
            assert_eq!((decoded.width(), decoded.height()), (width, height));
        }
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
                "{width}x{height} {content}: image 0.25 {:.3} ms / {} KB -> preview encoder {:.3} ms / {} KB",
                median(before).as_secs_f64() * 1e3,
                before_len / 1024,
                median(after).as_secs_f64() * 1e3,
                after_len / 1024,
            );
        }
    }
}
