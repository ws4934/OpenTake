//! Media-source contracts (SPEC §5.3). `opentake-render` DEFINES these traits;
//! `opentake-media` (or the caller) IMPLEMENTS them. This keeps the render crate
//! free of any decode/filesystem dependency: the plan builder only asks for a
//! source's intrinsic size / orientation, and the compositor only
//! asks for decoded pixels on demand.
//!
//! `media_ref` resolution (ref -> path) is the caller's job (upstream's
//! `MediaResolver`); render never touches the filesystem.

/// A decoded frame as packed RGBA8 (row-major, top-left origin), structurally
/// identical to `opentake_media::RgbaFrame`. Defined here so the render crate
/// does not depend on `opentake-media`; the integrating layer converts between
/// the two trivially (same field layout). SPEC §5.3.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DecodedFrame {
    pub width: u32,
    pub height: u32,
    /// `rgba.len() == width * height * 4`.
    pub rgba: Vec<u8>,
    /// Whether the RGB is already premultiplied by alpha.
    pub premultiplied: bool,
}

impl DecodedFrame {
    pub fn new(width: u32, height: u32, rgba: Vec<u8>, premultiplied: bool) -> Self {
        debug_assert_eq!(rgba.len(), width as usize * height as usize * 4);
        DecodedFrame {
            width,
            height,
            rgba,
            premultiplied,
        }
    }

    /// The same pixels with straight (non-premultiplied) alpha, for consumers
    /// such as FFmpeg's `rgba` input that read color independently of alpha.
    /// The compositor blends and reads back premultiplied color; convert only
    /// at that boundary. A straight frame is returned unchanged.
    pub fn into_straight_alpha(mut self) -> Self {
        if self.premultiplied {
            unpremultiply_rgba(&mut self.rgba);
            self.premultiplied = false;
        }
        self
    }
}

/// Undo premultiplication in place: `c = min(255, round(c * 255 / a))`, and a
/// fully transparent pixel becomes transparent black. Opaque pixels keep their
/// bytes, and color that rounding pushed above its alpha saturates at 255.
pub fn unpremultiply_rgba(rgba: &mut [u8]) {
    for px in rgba.as_chunks_mut::<4>().0 {
        match px[3] {
            0 => px[..3].fill(0),
            u8::MAX => {}
            alpha => {
                let alpha = u32::from(alpha);
                for channel in &mut px[..3] {
                    let straight = (u32::from(*channel) * 255 + alpha / 2) / alpha;
                    *channel = straight.min(255) as u8;
                }
            }
        }
    }
}

/// Source intrinsic size / orientation, queried once while building the plan
/// (pure metadata lookups; no decoding).
pub trait SourceMetrics {
    /// Video: decoded frame size; image: pixel size; Lottie: canvas size.
    /// Mirrors upstream `imageNativeSize` (L90) / `naturalSize`.
    fn natural_size(&self, media_ref: &str) -> Option<(u32, u32)>;

    /// Container display matrix -> row-major 6-tuple `[a, b, c, d, tx, ty]`
    /// (identity when absent). Mirrors upstream `preferredTransform` (L169).
    fn preferred_transform(&self, _media_ref: &str) -> [f64; 6] {
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    }

    /// Lottie internal frame count (used for the modulo wrap in SPEC §4.3).
    fn lottie_frame_count(&self, media_ref: &str) -> Option<i64> {
        let _ = media_ref;
        None
    }
}

/// Per-frame pixel supply, pulled lazily by the compositor while rendering.
pub trait FrameProvider {
    /// Pixels for `media_ref` at `source_frame` (SPEC §2.5). Preview: decode to
    /// the nearest keyframe and drop forward; export: decode sequentially.
    fn decoded_frame(&self, media_ref: &str, source_frame: i64) -> Option<DecodedFrame>;

    /// Image pixels (single frame; straight or premultiplied per
    /// [`DecodedFrame::premultiplied`], mirrors upstream `createPixelBuffer`, L101).
    fn image_pixels(&self, media_ref: &str) -> Option<DecodedFrame>;

    /// Lottie internal-frame raster (premultiplied RGBA).
    fn lottie_frame(&self, media_ref: &str, frame: i64) -> Option<DecodedFrame>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpremultiply_restores_straight_color_without_overflow() {
        let mut rgba = [
            128, 128, 128, 128, // 50% white
            7, 3, 1, 0, // transparent: color is meaningless
            10, 20, 30, 255, // opaque: unchanged
            64, 0, 32, 64, // 25% (255, 0, 128)
            200, 130, 129, 128, // rounding put color above alpha
        ];
        unpremultiply_rgba(&mut rgba);
        assert_eq!(
            rgba,
            [
                255, 255, 255, 128, //
                0, 0, 0, 0, //
                10, 20, 30, 255, //
                255, 0, 128, 64, //
                255, 255, 255, 128,
            ]
        );
    }

    #[test]
    fn straight_alpha_conversion_only_touches_premultiplied_frames() {
        let premultiplied = DecodedFrame::new(1, 1, vec![128, 128, 128, 128], true);
        let straight = premultiplied.into_straight_alpha();
        assert_eq!(straight.rgba, vec![255, 255, 255, 128]);
        assert!(!straight.premultiplied);
        // Already straight: returned as is, never divided a second time.
        assert_eq!(straight.clone().into_straight_alpha(), straight);
    }

    #[test]
    fn decoded_frame_holds_shape() {
        let f = DecodedFrame::new(2, 1, vec![1, 2, 3, 4, 5, 6, 7, 8], false);
        assert_eq!(f.width, 2);
        assert_eq!(f.height, 1);
        assert_eq!(f.rgba.len(), 8);
        assert!(!f.premultiplied);
    }

    /// A `SourceMetrics` using only the defaulted methods still compiles and
    /// returns the documented identity / None.
    struct MinimalMetrics;
    impl SourceMetrics for MinimalMetrics {
        fn natural_size(&self, _r: &str) -> Option<(u32, u32)> {
            Some((100, 50))
        }
    }

    #[test]
    fn source_metrics_defaults() {
        let m = MinimalMetrics;
        assert_eq!(m.natural_size("x"), Some((100, 50)));
        assert_eq!(m.preferred_transform("x"), [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(m.lottie_frame_count("x"), None);
    }
}
