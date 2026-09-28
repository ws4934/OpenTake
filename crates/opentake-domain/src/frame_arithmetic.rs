//! The frame arithmetic every stored clip satisfies.
//!
//! Frames are `i32`. Edits derive end frames, visible source extents and
//! retimed durations from a clip's start, duration, trims and speed; this is
//! the one check that all of them stay representable. The editing layer
//! validates every command against it, and the project loader refuses a
//! timeline that breaks it.

use crate::clip::Clip;
use crate::clip_type::ClipType;

/// The first frame-arithmetic rule a clip, or a clip about to be created,
/// breaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameArithmeticError {
    /// An audio or video clip has a negative trim. Image and text trims may
    /// reach past the source.
    NegativeTrim,
    /// `startFrame < 0` or `durationFrames < 1`.
    StartOrDuration,
    /// `speed` is not finite, or not positive.
    Speed,
    /// `startFrame + durationFrames` overflows.
    EndOverflow,
    /// `durationFrames + trimStartFrame + trimEndFrame` overflows.
    TrimOverflow,
    /// The visible source extent `round(durationFrames * speed)` is outside
    /// `0..=i32::MAX`.
    SourceExtentRange,
    /// `trimStartFrame` plus the visible source extent overflows.
    TrimStartExtent,
    /// `trimEndFrame` plus the visible source extent overflows.
    TrimEndExtent,
    /// Both trims plus the visible source extent overflow.
    SourceExtent,
}

impl FrameArithmeticError {
    /// The broken rule, worded for a user-facing error.
    pub fn message(self) -> &'static str {
        match self {
            Self::NegativeTrim => "trim frames must be >= 0 for audio/video clips",
            Self::StartOrDuration => "startFrame must be >= 0 and durationFrames >= 1",
            Self::Speed => "speed must be finite and > 0",
            Self::EndOverflow => "startFrame + durationFrames overflows",
            Self::TrimOverflow => "durationFrames + trim frames overflows",
            Self::SourceExtentRange => "visible source-frame extent is out of range",
            Self::TrimStartExtent => "trimStart source-frame extent overflows",
            Self::TrimEndExtent => "trimEnd source-frame extent overflows",
            Self::SourceExtent => "source-frame extent overflows",
        }
    }
}

impl std::fmt::Display for FrameArithmeticError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for FrameArithmeticError {}

/// Check the start, duration, trims and speed of a clip-shaped value, with
/// trims of any sign. Returns the end frame.
pub fn frame_arithmetic(
    start_frame: i32,
    duration_frames: i32,
    trim_start_frame: i32,
    trim_end_frame: i32,
    speed: f64,
) -> Result<i32, FrameArithmeticError> {
    use FrameArithmeticError as E;
    if start_frame < 0 || duration_frames < 1 {
        return Err(E::StartOrDuration);
    }
    if !speed.is_finite() || speed <= 0.0 {
        return Err(E::Speed);
    }
    let end_frame = start_frame
        .checked_add(duration_frames)
        .ok_or(E::EndOverflow)?;
    duration_frames
        .checked_add(trim_start_frame)
        .and_then(|value| value.checked_add(trim_end_frame))
        .ok_or(E::TrimOverflow)?;
    let consumed = (duration_frames as f64 * speed).round();
    if !(0.0..=i32::MAX as f64).contains(&consumed) {
        return Err(E::SourceExtentRange);
    }
    let consumed = consumed as i32;
    trim_start_frame
        .checked_add(consumed)
        .ok_or(E::TrimStartExtent)?;
    trim_end_frame
        .checked_add(consumed)
        .ok_or(E::TrimEndExtent)?;
    trim_start_frame
        .checked_add(consumed)
        .and_then(|value| value.checked_add(trim_end_frame))
        .ok_or(E::SourceExtent)?;
    Ok(end_frame)
}

/// [`frame_arithmetic`] for a clip of `media_type`: audio and video trims
/// must also be non-negative. Returns the end frame.
pub fn clip_frame_arithmetic(
    start_frame: i32,
    duration_frames: i32,
    trim_start_frame: i32,
    trim_end_frame: i32,
    speed: f64,
    media_type: ClipType,
) -> Result<i32, FrameArithmeticError> {
    if !matches!(media_type, ClipType::Image | ClipType::Text)
        && (trim_start_frame < 0 || trim_end_frame < 0)
    {
        return Err(FrameArithmeticError::NegativeTrim);
    }
    frame_arithmetic(
        start_frame,
        duration_frames,
        trim_start_frame,
        trim_end_frame,
        speed,
    )
}

impl Clip {
    /// [`clip_frame_arithmetic`] of this clip at its own start frame. Returns
    /// its end frame.
    pub fn frame_arithmetic(&self) -> Result<i32, FrameArithmeticError> {
        clip_frame_arithmetic(
            self.start_frame,
            self.duration_frames,
            self.trim_start_frame,
            self.trim_end_frame,
            self.speed,
            self.media_type,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use FrameArithmeticError as E;

    fn video(start: i32, duration: i32, trim_start: i32, trim_end: i32, speed: f64) -> Clip {
        let mut clip = Clip::new("c", "m", start, duration);
        clip.trim_start_frame = trim_start;
        clip.trim_end_frame = trim_end;
        clip.speed = speed;
        clip
    }

    #[test]
    fn a_well_formed_clip_reports_its_end_frame() {
        assert_eq!(video(10, 20, 3, 4, 2.0).frame_arithmetic(), Ok(30));
    }

    #[test]
    fn each_rule_is_reported_by_the_first_value_breaking_it() {
        let max = i32::MAX;
        let cases = [
            (video(0, 10, -1, 0, 1.0), E::NegativeTrim),
            (video(-1, 10, 0, 0, 1.0), E::StartOrDuration),
            (video(0, 0, 0, 0, 1.0), E::StartOrDuration),
            (video(0, 10, 0, 0, f64::NAN), E::Speed),
            (video(0, 10, 0, 0, 0.0), E::Speed),
            (video(max, 1, 0, 0, 1.0), E::EndOverflow),
            (video(0, 10, max, 0, 1.0), E::TrimOverflow),
            (video(0, 10, 0, 0, f64::MAX), E::SourceExtentRange),
            (video(0, 10, max - 12, 0, 2.0), E::TrimStartExtent),
            (video(0, 10, 0, max - 12, 2.0), E::TrimEndExtent),
            (
                video(0, 10, (max - 20) / 2, (max - 20) / 2, 4.0),
                E::SourceExtent,
            ),
        ];
        for (clip, error) in cases {
            assert_eq!(clip.frame_arithmetic(), Err(error), "{clip:?}");
        }
    }

    #[test]
    fn image_and_text_trims_may_be_negative() {
        for media_type in [ClipType::Image, ClipType::Text] {
            let mut clip = video(0, 10, -5, -7, 1.0);
            clip.media_type = media_type;
            assert_eq!(clip.frame_arithmetic(), Ok(10));
        }
        assert_eq!(frame_arithmetic(0, 10, -5, -7, 1.0), Ok(10));
    }

    #[test]
    fn errors_read_as_the_rule_they_break() {
        assert_eq!(
            E::StartOrDuration.to_string(),
            "startFrame must be >= 0 and durationFrames >= 1"
        );
        assert_eq!(
            E::NegativeTrim.to_string(),
            "trim frames must be >= 0 for audio/video clips"
        );
    }
}
