//! Persisted, editable video-stabilization solution.
//!
//! The track is deliberately separate from the user's authored position/scale/
//! rotation keyframes. Renderers compose both tracks, so applying or resetting
//! stabilization never destroys manual animation or source media identity.

use serde::{Deserialize, Serialize};

use crate::clip::Clip;

fn default_strength() -> f64 {
    1.0
}

fn default_model_version() -> u32 {
    1
}

#[derive(Clone, Copy, PartialEq, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StabilizationTransform {
    pub translation_x: f64,
    pub translation_y: f64,
    pub rotation_degrees: f64,
}

#[derive(Clone, Copy, PartialEq, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StabilizationKeyframe {
    pub frame: i32,
    pub translation_x: f64,
    pub translation_y: f64,
    pub rotation_degrees: f64,
}

/// Which source frame a clip shows at each clip-relative frame, before the
/// renderer rounds: the source frame (counted at the timeline frame rate) at
/// clip-relative frame `r` is `origin + step * r`. Mirrors the renderer's
/// `source_frame_index` for decoded video, where a reversed clip runs backwards
/// from the last source frame it consumes.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SourceMapping {
    pub origin: f64,
    pub step: f64,
}

impl SourceMapping {
    /// `None` when the clip's speed cannot map frames.
    pub fn of(clip: &Clip) -> Option<Self> {
        if !clip.speed.is_finite() || clip.speed <= 0.0 {
            return None;
        }
        let trim = f64::from(clip.trim_start_frame);
        if clip.reversed {
            let consumed = (f64::from(clip.duration_frames.max(1)) * clip.speed)
                .round()
                .max(1.0);
            Some(Self {
                origin: trim + consumed - 1.0,
                step: -clip.speed,
            })
        } else {
            Some(Self {
                origin: trim,
                step: clip.speed,
            })
        }
    }
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StabilizationTrack {
    pub model: String,
    #[serde(default = "default_model_version")]
    pub model_version: u32,
    pub source_identity: String,
    #[serde(default = "default_strength")]
    pub strength: f64,
    #[serde(default)]
    pub crop_margin: f64,
    #[serde(default)]
    pub keyframes: Vec<StabilizationKeyframe>,
}

impl StabilizationTrack {
    /// Linearly sample the correction track at one clip-relative frame.
    pub fn sample(&self, frame: i32) -> StabilizationTransform {
        let Some(first) = self.keyframes.first() else {
            return StabilizationTransform::default();
        };
        let strength = self.strength.clamp(0.0, 1.0);
        let raw = if frame <= first.frame {
            keyframe_transform(*first)
        } else if let Some(last) = self.keyframes.last().filter(|last| frame >= last.frame) {
            keyframe_transform(*last)
        } else {
            let pair = self
                .keyframes
                .windows(2)
                .find(|pair| frame >= pair[0].frame && frame <= pair[1].frame)
                .expect("a sorted stabilization track covers an interior sample");
            let span = (pair[1].frame - pair[0].frame).max(1) as f64;
            let t = (frame - pair[0].frame) as f64 / span;
            StabilizationTransform {
                translation_x: lerp(pair[0].translation_x, pair[1].translation_x, t),
                translation_y: lerp(pair[0].translation_y, pair[1].translation_y, t),
                rotation_degrees: lerp(pair[0].rotation_degrees, pair[1].rotation_degrees, t),
            }
        };
        StabilizationTransform {
            translation_x: raw.translation_x * strength,
            translation_y: raw.translation_y * strength,
            rotation_degrees: raw.rotation_degrees * strength,
        }
    }

    /// The same corrections, moved with the source frames they were measured
    /// on after an edit changed which source frame the clip shows at each
    /// clip-relative frame (a head trim, slip, split, speed or direction change,
    /// or a timeline frame-rate change). `fps_ratio` is the new timeline frame
    /// rate divided by the old one.
    ///
    /// Keyframes land on the nearest whole frame. When several land on one
    /// frame (a speed-up), the one measured closest to it is kept, and a
    /// solution that collapses onto a single frame keeps it as a constant
    /// correction. `None` when the result does not fit whole `i32` frames.
    pub fn rebased(&self, from: SourceMapping, to: SourceMapping, fps_ratio: f64) -> Option<Self> {
        if !fps_ratio.is_finite()
            || fps_ratio <= 0.0
            || !to.step.is_finite()
            || to.step == 0.0
            || !to.origin.is_finite()
        {
            return None;
        }
        let mut placed = Vec::with_capacity(self.keyframes.len());
        for keyframe in &self.keyframes {
            let source = (from.origin + from.step * f64::from(keyframe.frame)) * fps_ratio;
            let exact = (source - to.origin) / to.step;
            let frame = exact.round();
            if !frame.is_finite() || frame < f64::from(i32::MIN) || frame > f64::from(i32::MAX) {
                return None;
            }
            let frame = frame as i32;
            let error = (exact - f64::from(frame)).abs();
            placed.push((frame, error, StabilizationKeyframe { frame, ..*keyframe }));
        }
        // Reversal flips the order; the closest sample wins a shared frame.
        placed.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
        placed.dedup_by_key(|(frame, _, _)| *frame);
        let mut keyframes: Vec<StabilizationKeyframe> = placed
            .into_iter()
            .map(|(_, _, keyframe)| keyframe)
            .collect();
        if keyframes.len() == 1 && self.keyframes.len() > 1 {
            let only = keyframes[0];
            keyframes.push(StabilizationKeyframe {
                frame: only.frame.checked_add(1)?,
                ..only
            });
        }
        Some(Self {
            model: self.model.clone(),
            model_version: self.model_version,
            source_identity: self.source_identity.clone(),
            strength: self.strength,
            crop_margin: self.crop_margin,
            keyframes,
        })
    }

    /// Conservative uniform zoom needed to keep every output corner covered.
    /// `aspect_ratio` is output width / height.
    pub fn crop_scale(&self, aspect_ratio: f64) -> f64 {
        let aspect = aspect_ratio.max(1e-6);
        let required = self
            .keyframes
            .iter()
            .map(|keyframe| {
                let correction = self.sample(keyframe.frame);
                coverage_scale(correction, aspect)
            })
            .fold(1.0_f64, f64::max);
        required + self.crop_margin.max(0.0) * 2.0
    }

    pub fn guarantees_coverage(&self, aspect_ratio: f64) -> bool {
        let scale = self.crop_scale(aspect_ratio);
        self.keyframes.iter().all(|keyframe| {
            scale + 1e-12 >= coverage_scale(self.sample(keyframe.frame), aspect_ratio.max(1e-6))
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.model.trim().is_empty() || self.model_version == 0 {
            return Err("stabilization model and version are required".to_string());
        }
        if self.source_identity.trim().is_empty() {
            return Err("stabilization source identity is required".to_string());
        }
        if !(0.0..=1.0).contains(&self.strength) || !self.strength.is_finite() {
            return Err("stabilization strength must be finite and within 0..=1".to_string());
        }
        if !(0.0..=0.5).contains(&self.crop_margin) || !self.crop_margin.is_finite() {
            return Err("stabilization crop margin must be finite and within 0..=0.5".to_string());
        }
        if self.keyframes.len() < 2 {
            return Err("stabilization requires at least two keyframes".to_string());
        }
        let mut previous = None;
        for keyframe in &self.keyframes {
            if previous.is_some_and(|frame| keyframe.frame <= frame) {
                return Err("stabilization keyframes must be strictly increasing".to_string());
            }
            if !keyframe.translation_x.is_finite()
                || !keyframe.translation_y.is_finite()
                || !keyframe.rotation_degrees.is_finite()
            {
                return Err("stabilization keyframes must be finite".to_string());
            }
            previous = Some(keyframe.frame);
        }
        Ok(())
    }
}

fn keyframe_transform(keyframe: StabilizationKeyframe) -> StabilizationTransform {
    StabilizationTransform {
        translation_x: keyframe.translation_x,
        translation_y: keyframe.translation_y,
        rotation_degrees: keyframe.rotation_degrees,
    }
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

fn coverage_scale(correction: StabilizationTransform, aspect: f64) -> f64 {
    let radians = correction.rotation_degrees.to_radians();
    let (sin, cos) = radians.sin_cos();
    let sin = sin.abs();
    let cos = cos.abs();
    let translation_x = correction.translation_x.abs();
    let translation_y = correction.translation_y.abs();
    let cover_width = cos + sin / aspect + 2.0 * (translation_x + translation_y / aspect);
    let cover_height = cos + sin * aspect + 2.0 * (translation_y + translation_x * aspect);
    cover_width.max(cover_height).max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_scales_correction_by_editable_strength() {
        let track = StabilizationTrack {
            model: "test".into(),
            model_version: 1,
            source_identity: "asset".into(),
            strength: 0.5,
            crop_margin: 0.0,
            keyframes: vec![
                StabilizationKeyframe::default(),
                StabilizationKeyframe {
                    frame: 10,
                    translation_x: 0.2,
                    translation_y: -0.1,
                    rotation_degrees: 4.0,
                },
            ],
        };
        let sample = track.sample(5);
        assert!((sample.translation_x - 0.05).abs() < 1e-12);
        assert!((sample.translation_y + 0.025).abs() < 1e-12);
        assert!((sample.rotation_degrees - 1.0).abs() < 1e-12);
        assert!(track.guarantees_coverage(16.0 / 9.0));
    }

    /// One keyframe per source frame `0..count`, translation `0.001 * frame`.
    fn per_frame_track(count: i32) -> StabilizationTrack {
        StabilizationTrack {
            model: "test".into(),
            model_version: 1,
            source_identity: "asset".into(),
            strength: 1.0,
            crop_margin: 0.0,
            keyframes: (0..count)
                .map(|frame| StabilizationKeyframe {
                    frame,
                    translation_x: 0.001 * f64::from(frame),
                    ..StabilizationKeyframe::default()
                })
                .collect(),
        }
    }

    fn frames(track: &StabilizationTrack) -> Vec<(i32, f64)> {
        track
            .keyframes
            .iter()
            .map(|keyframe| (keyframe.frame, (keyframe.translation_x * 1000.0).round()))
            .collect()
    }

    fn clip(trim_start: i32, duration: i32, speed: f64, reversed: bool) -> Clip {
        let mut clip = Clip::new("c", "asset", 0, duration);
        clip.trim_start_frame = trim_start;
        clip.speed = speed;
        clip.reversed = reversed;
        clip
    }

    fn mapping(clip: &Clip) -> SourceMapping {
        SourceMapping::of(clip).unwrap()
    }

    #[test]
    fn source_mapping_matches_forward_and_reversed_playback() {
        assert_eq!(
            mapping(&clip(10, 30, 2.0, false)),
            SourceMapping {
                origin: 10.0,
                step: 2.0
            }
        );
        // Reversed playback starts on the last consumed frame: 10 + 60 - 1.
        assert_eq!(
            mapping(&clip(10, 30, 2.0, true)),
            SourceMapping {
                origin: 69.0,
                step: -2.0
            }
        );
        assert!(SourceMapping::of(&clip(0, 30, 0.0, false)).is_none());
        assert!(SourceMapping::of(&clip(0, 30, f64::NAN, false)).is_none());
    }

    #[test]
    fn rebase_follows_head_trims_and_reversal() {
        let track = per_frame_track(4);
        let original = mapping(&clip(0, 4, 1.0, false));
        assert_eq!(track.rebased(original, original, 1.0).unwrap(), track);

        let head_trimmed = track
            .rebased(original, mapping(&clip(2, 2, 1.0, false)), 1.0)
            .unwrap();
        assert_eq!(
            frames(&head_trimmed),
            [(-2, 0.0), (-1, 1.0), (0, 2.0), (1, 3.0)]
        );

        let reversed = track
            .rebased(original, mapping(&clip(0, 4, 1.0, true)), 1.0)
            .unwrap();
        assert_eq!(frames(&reversed), [(0, 3.0), (1, 2.0), (2, 1.0), (3, 0.0)]);
        assert!(reversed.validate().is_ok());
    }

    #[test]
    fn rebase_keeps_the_closest_sample_when_a_speed_up_merges_frames() {
        let track = per_frame_track(5);
        let original = mapping(&clip(0, 5, 1.0, false));
        // At 2x clip frame r shows source frame 2r: odd samples fall between
        // frames and lose to the exact even ones.
        let faster = track
            .rebased(original, mapping(&clip(0, 3, 2.0, false)), 1.0)
            .unwrap();
        assert_eq!(frames(&faster), [(0, 0.0), (1, 2.0), (2, 4.0)]);
        assert!(faster.validate().is_ok());

        // A solution that collapses onto one frame stays a valid, constant one.
        let two = per_frame_track(2);
        let collapsed = two
            .rebased(original, mapping(&clip(0, 1, 100.0, false)), 1.0)
            .unwrap();
        assert_eq!(frames(&collapsed), [(0, 0.0), (1, 0.0)]);
        assert!(collapsed.validate().is_ok());
    }

    #[test]
    fn rebase_scales_with_the_timeline_frame_rate() {
        let track = per_frame_track(3);
        let original = mapping(&clip(0, 3, 1.0, false));
        // 30 -> 60 fps: source frame s is now counted as 2s.
        let doubled = track.rebased(original, original, 2.0).unwrap();
        assert_eq!(frames(&doubled), [(0, 0.0), (2, 1.0), (4, 2.0)]);
    }

    #[test]
    fn rebase_rejects_frames_outside_i32() {
        let track = per_frame_track(2);
        let original = mapping(&clip(0, 2, 1.0, false));
        let far = SourceMapping {
            origin: -1.0e12,
            step: 1.0,
        };
        assert!(track.rebased(original, far, 1.0).is_none());
        assert!(track.rebased(original, original, f64::NAN).is_none());
    }
}
