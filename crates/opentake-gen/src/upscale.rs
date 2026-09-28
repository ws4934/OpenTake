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

/// Encoders round odd sizes to even ones, so the long side of an
/// aspect-preserving result may differ from the exact product by this much.
const LONG_SIDE_TOLERANCE: u32 = 2;

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

/// Whether a probed result is the size the request asked for: the same
/// orientation, exactly the target short side, and the source aspect ratio.
pub fn upscale_result_matches(
    source_width: u32,
    source_height: u32,
    target: UpscaleResolution,
    result_width: u32,
    result_height: u32,
) -> bool {
    let Some((expected_width, expected_height)) =
        expected_upscale_size(source_width, source_height, target)
    else {
        return false;
    };
    if source_width >= source_height {
        result_height == expected_height
            && result_width.abs_diff(expected_width) <= LONG_SIDE_TOLERANCE
    } else {
        result_width == expected_width
            && result_height.abs_diff(expected_height) <= LONG_SIDE_TOLERANCE
    }
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
    fn accepts_only_the_requested_target_size() {
        let full_hd = video_upscale_resolution("1080p").unwrap();
        let uhd = video_upscale_resolution("4k").unwrap();
        assert_eq!(
            expected_upscale_size(1280, 720, full_hd),
            Some((1920, 1080))
        );
        assert!(upscale_result_matches(1280, 720, full_hd, 1920, 1080));
        assert!(!upscale_result_matches(1280, 720, full_hd, 2560, 1440));
        assert!(!upscale_result_matches(1280, 720, full_hd, 1920, 1088));
        assert!(!upscale_result_matches(1280, 720, full_hd, 1080, 1920));
        assert!(upscale_result_matches(1920, 1080, uhd, 3840, 2160));
        assert!(!upscale_result_matches(1920, 1080, uhd, 1920, 1080));
        assert!(upscale_result_matches(1080, 1920, uhd, 2160, 3840));
        // 854x480 scales to 1281x720; encoders may round the long side.
        let hd = video_upscale_resolution("720p").unwrap();
        assert!(upscale_result_matches(854, 480, hd, 1280, 720));
        assert!(upscale_result_matches(854, 480, hd, 1282, 720));
        assert!(!upscale_result_matches(854, 480, hd, 1290, 720));
        assert_eq!(video_upscale_resolution("8k"), None);
    }
}
