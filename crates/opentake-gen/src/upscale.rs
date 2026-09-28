//! Output targets for resolution-targeted video upscalers.
//!
//! Replicate's `topazlabs/video-upscale` has no scale-factor input. Its input
//! schema (`https://replicate.com/topazlabs/video-upscale/api/schema`) is
//! `video`, `target_resolution` (`720p`, `1080p` or `4k`; default `1080p`) and
//! `target_fps` (an integer from 15 to 60; default 30). A request that sends
//! only the video therefore always produces 1080p at 30 fps. OpenTake asks for
//! the smallest target above the source's short side and keeps the source
//! frame rate whenever the model accepts it.

use std::ops::RangeInclusive;

/// One `target_resolution` value and the frame size it produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpscaleResolution {
    /// Wire value sent to the provider.
    pub value: &'static str,
    /// Label used in the generated asset name.
    pub label: &'static str,
    /// Short side of the produced frame, in pixels.
    pub short_side: u32,
}

/// Targets the model accepts, smallest first.
pub const VIDEO_UPSCALE_RESOLUTIONS: [UpscaleResolution; 3] = [
    UpscaleResolution {
        value: "720p",
        label: "720p",
        short_side: 720,
    },
    UpscaleResolution {
        value: "1080p",
        label: "1080p",
        short_side: 1080,
    },
    UpscaleResolution {
        value: "4k",
        label: "4K",
        short_side: 2160,
    },
];

/// The model's default target, which applies to requests that sent no
/// `target_resolution` (jobs submitted before targets were sent).
pub const VIDEO_UPSCALE_DEFAULT_RESOLUTION: &str = "1080p";

/// Frame rates the model accepts for `target_fps`.
pub const VIDEO_UPSCALE_FPS: RangeInclusive<u32> = 15..=60;

/// The model's default `target_fps`.
pub const VIDEO_UPSCALE_DEFAULT_FPS: u32 = 30;

/// Encoders round odd sizes to even ones (or pad to a block size), so a
/// result's short side may fall this far below the target.
const SHORT_SIDE_TOLERANCE: u32 = 2;

/// Largest relative difference between the result's and the source's aspect
/// ratio (rounding, codec padding, anamorphic storage).
const ASPECT_RATIO_TOLERANCE: f64 = 0.01;

/// What to request for one source video.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoUpscalePlan {
    pub resolution: UpscaleResolution,
    /// `None` when the source frame rate is unknown; the model default applies.
    pub fps: Option<u32>,
    /// User-facing note when the result frame rate will differ from the source.
    pub fps_warning: Option<String>,
}

/// Why a source cannot be upscaled by a resolution-targeted model.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum VideoUpscaleError {
    #[error("the upscale source has no known frame size; re-import it and try again")]
    UnknownSourceSize,
    #[error(
        "the upscale source is already {0} pixels on its short side; the largest upscale target is 4K (2160)"
    )]
    NoLargerTarget(u32),
}

/// Look up a `target_resolution` wire value.
pub fn video_upscale_resolution(value: &str) -> Option<UpscaleResolution> {
    VIDEO_UPSCALE_RESOLUTIONS
        .iter()
        .copied()
        .find(|resolution| resolution.value.eq_ignore_ascii_case(value))
}

/// Choose the smallest target above the source's short side, and the source
/// frame rate clamped to what the model accepts.
pub fn plan_video_upscale(
    width: u32,
    height: u32,
    fps: Option<f64>,
) -> Result<VideoUpscalePlan, VideoUpscaleError> {
    if width == 0 || height == 0 {
        return Err(VideoUpscaleError::UnknownSourceSize);
    }
    let short_side = width.min(height);
    let resolution = VIDEO_UPSCALE_RESOLUTIONS
        .iter()
        .copied()
        .find(|resolution| resolution.short_side > short_side)
        .ok_or(VideoUpscaleError::NoLargerTarget(short_side))?;
    let (fps, fps_warning) = match fps.filter(|fps| fps.is_finite() && *fps > 0.0) {
        None => (
            None,
            Some(format!(
                "the source frame rate is unknown, so the upscaler's default \
                 {VIDEO_UPSCALE_DEFAULT_FPS} fps applies"
            )),
        ),
        Some(source) => {
            let rounded = source.round().min(f64::from(u32::MAX)) as u32;
            let clamped = rounded.clamp(*VIDEO_UPSCALE_FPS.start(), *VIDEO_UPSCALE_FPS.end());
            let warning = (clamped != rounded).then(|| {
                format!(
                    "the source is {} fps but the upscaler accepts {}-{} fps, so the result \
                     will be {clamped} fps",
                    (source * 1000.0).round() / 1000.0,
                    VIDEO_UPSCALE_FPS.start(),
                    VIDEO_UPSCALE_FPS.end()
                )
            });
            (Some(clamped), warning)
        }
    };
    Ok(VideoUpscalePlan {
        resolution,
        fps,
        fps_warning,
    })
}

/// The aspect-preserving frame size whose short side is the target's.
pub fn expected_upscale_size(
    source_width: u32,
    source_height: u32,
    target: UpscaleResolution,
) -> Option<(u32, u32)> {
    if source_width == 0 || source_height == 0 {
        return None;
    }
    let short = source_width.min(source_height);
    let long = source_width.max(source_height);
    let scaled_long =
        (u64::from(long) * u64::from(target.short_side) + u64::from(short) / 2) / u64::from(short);
    let scaled_long = u32::try_from(scaled_long).ok()?;
    Some(if source_width >= source_height {
        (scaled_long, target.short_side)
    } else {
        (target.short_side, scaled_long)
    })
}

/// Whether a probed result is plausibly what the request asked for. The
/// result is paid for before it is checked, so the check only rejects a
/// result that is clearly something else: the source's aspect ratio (within
/// about 1 %, which also fixes the orientation) and a short side of at least
/// the target's. A larger result is accepted.
pub fn upscale_result_matches(
    source_width: u32,
    source_height: u32,
    target: UpscaleResolution,
    result_width: u32,
    result_height: u32,
) -> bool {
    if source_width == 0 || source_height == 0 || result_width == 0 || result_height == 0 {
        return false;
    }
    let source_aspect = f64::from(source_width) / f64::from(source_height);
    let result_aspect = f64::from(result_width) / f64::from(result_height);
    ((result_aspect - source_aspect).abs() / source_aspect) <= ASPECT_RATIO_TOLERANCE
        && result_width.min(result_height) + SHORT_SIDE_TOLERANCE >= target.short_side
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_the_smallest_target_above_the_short_side() {
        let cases = [
            ((640, 360), "720p"),
            ((960, 540), "720p"),
            ((1280, 720), "1080p"),
            ((720, 1280), "1080p"),
            ((1920, 1080), "4k"),
            ((1080, 1920), "4k"),
            ((2048, 1080), "4k"),
        ];
        for ((width, height), expected) in cases {
            let plan = plan_video_upscale(width, height, Some(30.0)).unwrap();
            assert_eq!(plan.resolution.value, expected, "{width}x{height}");
        }
    }

    #[test]
    fn refuses_sources_at_or_above_the_largest_target_and_unknown_sizes() {
        assert_eq!(
            plan_video_upscale(3840, 2160, Some(30.0)),
            Err(VideoUpscaleError::NoLargerTarget(2160))
        );
        assert_eq!(
            plan_video_upscale(0, 1080, Some(30.0)),
            Err(VideoUpscaleError::UnknownSourceSize)
        );
    }

    #[test]
    fn keeps_the_source_frame_rate_and_warns_only_when_clamped() {
        let plan = plan_video_upscale(1280, 720, Some(24.0)).unwrap();
        assert_eq!(plan.fps, Some(24));
        assert!(plan.fps_warning.is_none());
        let ntsc = plan_video_upscale(1280, 720, Some(29.97)).unwrap();
        assert_eq!(ntsc.fps, Some(30));
        assert!(ntsc.fps_warning.is_none());
        let fast = plan_video_upscale(1280, 720, Some(120.0)).unwrap();
        assert_eq!(fast.fps, Some(60));
        assert_eq!(
            fast.fps_warning.as_deref(),
            Some("the source is 120 fps but the upscaler accepts 15-60 fps, so the result will be 60 fps")
        );
        let slow = plan_video_upscale(1280, 720, Some(8.0)).unwrap();
        assert_eq!(slow.fps, Some(15));
        assert!(slow.fps_warning.is_some());
        let unknown = plan_video_upscale(1280, 720, None).unwrap();
        assert_eq!(unknown.fps, None);
        assert!(unknown.fps_warning.is_some());
    }

    #[test]
    fn accepts_results_with_the_source_aspect_and_at_least_the_target_size() {
        let full_hd = video_upscale_resolution("1080p").unwrap();
        let uhd = video_upscale_resolution("4k").unwrap();
        let hd = video_upscale_resolution("720p").unwrap();
        assert_eq!(
            expected_upscale_size(1280, 720, full_hd),
            Some((1920, 1080))
        );
        for (source, target, result) in [
            ((1280, 720), full_hd, (1920, 1080)),
            // Padded to a codec block size.
            ((1280, 720), full_hd, (1920, 1088)),
            // Larger than asked for.
            ((1280, 720), full_hd, (2560, 1440)),
            ((1920, 1080), uhd, (3840, 2160)),
            // Portrait, DCI and ultra-wide sources.
            ((1080, 1920), uhd, (2160, 3840)),
            ((720, 1280), full_hd, (1080, 1920)),
            ((2048, 1080), uhd, (4096, 2160)),
            ((2560, 1080), uhd, (5120, 2160)),
            // 854x480 scales to 1281x720; encoders round the long side.
            ((854, 480), hd, (1280, 720)),
            ((854, 480), hd, (1282, 720)),
            // An odd short side rounded down.
            ((854, 480), hd, (1280, 718)),
        ] {
            assert!(
                upscale_result_matches(source.0, source.1, target, result.0, result.1),
                "{source:?} -> {result:?}"
            );
        }
        for (source, target, result) in [
            // Rotated.
            ((1280, 720), full_hd, (1080, 1920)),
            // Not upscaled to the target.
            ((1920, 1080), uhd, (1920, 1080)),
            ((1280, 720), full_hd, (1280, 720)),
            // Another aspect ratio (4:3, 5 % wider).
            ((1280, 720), full_hd, (1440, 1080)),
            ((1280, 720), full_hd, (2016, 1080)),
            ((0, 720), full_hd, (1920, 1080)),
        ] {
            assert!(
                !upscale_result_matches(source.0, source.1, target, result.0, result.1),
                "{source:?} -> {result:?}"
            );
        }
        assert_eq!(video_upscale_resolution("8k"), None);
    }
}
