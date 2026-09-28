//! Keyframe model and sampling. 1:1 port of upstream `Keyframe.swift` plus the
//! `split_keyframe_track` model invariant (lifted from `EditorViewModel`).
//!
//! Frames stored in a track are **clip-relative offsets**. Sampling clamps at the
//! endpoints (no extrapolation) and picks the interpolation mode from the *left*
//! keyframe's `interpolation_out`. `smoothstep(t) = t*t*(3 - 2t)`.

use serde::{Deserialize, Serialize};

use crate::transform::Crop;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Interpolation {
    Linear,
    Hold,
    Smooth,
}

impl Interpolation {
    pub const ALL: [Interpolation; 3] = [
        Interpolation::Linear,
        Interpolation::Hold,
        Interpolation::Smooth,
    ];
}

/// `smoothstep(t) = t*t*(3 - 2t)`. Matches upstream exactly (no clamping of `t`;
/// callers pass an already-normalized `t`).
#[inline]
pub fn smoothstep(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}

/// Linear interpolation between keyframe values. `f64`, [`AnimPair`], and
/// [`Crop`](crate::transform::Crop) implement this.
pub trait KeyframeInterpolatable: Sized {
    fn keyframe_interpolate(a: Self, b: Self, t: f64) -> Self;
}

impl KeyframeInterpolatable for f64 {
    fn keyframe_interpolate(a: f64, b: f64, t: f64) -> f64 {
        a + (b - a) * t
    }
}

/// Two-component keyframe value used for position `(x, y)` and scale `(w, h)`.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
pub struct AnimPair {
    pub a: f64,
    pub b: f64,
}

impl AnimPair {
    /// Persisted keys owned by the pair-valued keyframe wire schema.
    pub const WIRE_FIELDS: &'static [&'static str] = &["a", "b"];
    pub const SCALAR_WIRE_FIELDS: &'static [&'static str] = Self::WIRE_FIELDS;

    pub fn new(a: f64, b: f64) -> Self {
        AnimPair { a, b }
    }
}

impl KeyframeInterpolatable for AnimPair {
    fn keyframe_interpolate(from: AnimPair, to: AnimPair, t: f64) -> AnimPair {
        AnimPair {
            a: f64::keyframe_interpolate(from.a, to.a, t),
            b: f64::keyframe_interpolate(from.b, to.b, t),
        }
    }
}

// `Crop` interpolates component-wise. Mirrors upstream
// `extension Crop: KeyframeInterpolatable` from `Keyframe.swift`.
impl KeyframeInterpolatable for Crop {
    fn keyframe_interpolate(a: Crop, b: Crop, t: f64) -> Crop {
        Crop {
            left: f64::keyframe_interpolate(a.left, b.left, t),
            top: f64::keyframe_interpolate(a.top, b.top, t),
            right: f64::keyframe_interpolate(a.right, b.right, t),
            bottom: f64::keyframe_interpolate(a.bottom, b.bottom, t),
        }
    }
}

/// Keyframe values whose components can be checked for finiteness before a
/// track is stored (see [`KeyframeTrack::validate`]).
pub trait FiniteKeyframeValue {
    /// `true` when every component is a finite number.
    fn all_finite(&self) -> bool;
}

impl FiniteKeyframeValue for f64 {
    fn all_finite(&self) -> bool {
        self.is_finite()
    }
}

impl FiniteKeyframeValue for AnimPair {
    fn all_finite(&self) -> bool {
        self.a.is_finite() && self.b.is_finite()
    }
}

impl FiniteKeyframeValue for Crop {
    fn all_finite(&self) -> bool {
        [self.left, self.top, self.right, self.bottom]
            .iter()
            .all(|value| value.is_finite())
    }
}

fn default_interpolation_out() -> Interpolation {
    Interpolation::Smooth
}

#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Keyframe<V> {
    pub frame: i32,
    pub value: V,
    #[serde(default = "default_interpolation_out")]
    pub interpolation_out: Interpolation,
}

impl<V> Keyframe<V> {
    /// Persisted keys owned by a keyframe entry.
    pub const WIRE_FIELDS: &'static [&'static str] = &["frame", "value", "interpolationOut"];
    pub const SCALAR_WIRE_FIELDS: &'static [&'static str] = &["frame", "interpolationOut"];
    pub const VALUE_WIRE_FIELD: &'static str = "value";

    /// New keyframe with the upstream default `interpolation_out = .smooth`.
    pub fn new(frame: i32, value: V) -> Self {
        Keyframe {
            frame,
            value,
            interpolation_out: Interpolation::Smooth,
        }
    }

    pub fn with_interpolation(frame: i32, value: V, interpolation_out: Interpolation) -> Self {
        Keyframe {
            frame,
            value,
            interpolation_out,
        }
    }
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct KeyframeTrack<V> {
    #[serde(default = "Vec::new")]
    pub keyframes: Vec<Keyframe<V>>,
}

impl<V> Default for KeyframeTrack<V> {
    fn default() -> Self {
        KeyframeTrack {
            keyframes: Vec::new(),
        }
    }
}

impl<V> KeyframeTrack<V> {
    /// Persisted keys owned by a keyframe track container.
    pub const WIRE_FIELDS: &'static [&'static str] = &["keyframes"];
    pub const KEYFRAMES_WIRE_FIELD: &'static str = "keyframes";

    pub fn new() -> Self {
        KeyframeTrack::default()
    }

    pub fn from_keyframes(keyframes: Vec<Keyframe<V>>) -> Self {
        KeyframeTrack { keyframes }
    }

    /// A track is active iff it holds at least one keyframe.
    pub fn is_active(&self) -> bool {
        !self.keyframes.is_empty()
    }

    /// Restore the storage invariant that [`Self::sample`] and [`Self::upsert`]
    /// rely on: keyframes strictly ascending by frame, one per frame. The sort
    /// is stable, so the **last** keyframe given for a duplicate frame wins.
    /// 1:1 port of upstream `ToolExecutor.sortAndDedupe`.
    pub fn normalized(mut self) -> Self {
        self.keyframes.sort_by_key(|kf| kf.frame);
        let mut keyframes: Vec<Keyframe<V>> = Vec::with_capacity(self.keyframes.len());
        for kf in self.keyframes {
            match keyframes.last_mut() {
                Some(last) if last.frame == kf.frame => *last = kf,
                _ => keyframes.push(kf),
            }
        }
        KeyframeTrack { keyframes }
    }

    /// Insert or replace a keyframe, keeping `keyframes` sorted ascending by
    /// frame. Replaces in place when a keyframe already exists at `kf.frame`.
    pub fn upsert(&mut self, kf: Keyframe<V>) {
        if let Some(i) = self.keyframes.iter().position(|k| k.frame == kf.frame) {
            self.keyframes[i] = kf;
        } else {
            let at = self
                .keyframes
                .iter()
                .position(|k| k.frame > kf.frame)
                .unwrap_or(self.keyframes.len());
            self.keyframes.insert(at, kf);
        }
    }

    /// Remove every keyframe at `frame`.
    pub fn remove(&mut self, frame: i32) {
        self.keyframes.retain(|k| k.frame != frame);
    }

    /// Move a keyframe from `old_frame` to `new_frame`. No-op if no keyframe sits
    /// at `old_frame`; **abandoned** if the destination is already occupied
    /// (matching upstream `move(from:to:)`).
    pub fn move_keyframe(&mut self, old_frame: i32, new_frame: i32) {
        let Some(i) = self.keyframes.iter().position(|k| k.frame == old_frame) else {
            return;
        };
        if new_frame != old_frame && self.keyframes.iter().any(|k| k.frame == new_frame) {
            return;
        }
        let mut kf = self.keyframes.remove(i);
        kf.frame = new_frame;
        self.upsert(kf);
    }
}

impl<V: FiniteKeyframeValue> KeyframeTrack<V> {
    /// Check a replacement track against a clip of `duration_frames`: every
    /// frame must lie in the clip-relative span `[0, duration_frames]` (the
    /// closed range [`Clip::clamp_keyframes_to_duration`](crate::Clip) keeps)
    /// and every value must be finite. Errors name the offending
    /// `keyframes[index]` in the track's current order.
    pub fn validate(&self, duration_frames: i32) -> Result<(), String> {
        for (index, kf) in self.keyframes.iter().enumerate() {
            if kf.frame < 0 || kf.frame > duration_frames {
                return Err(format!(
                    "keyframes[{index}]: frame {} is outside the clip (0...{duration_frames})",
                    kf.frame
                ));
            }
            if !kf.value.all_finite() {
                return Err(format!("keyframes[{index}]: values must be finite"));
            }
        }
        Ok(())
    }
}

impl<V: KeyframeInterpolatable + Clone> KeyframeTrack<V> {
    /// Sample the curve at clip-relative `frame`. Clamps to the first/last value
    /// outside the keyframe span (no extrapolation). Inside a span, the *left*
    /// keyframe's `interpolation_out` selects hold / linear / smooth. Frame
    /// distances are computed in `i64`, so persisted extreme frames cannot
    /// overflow.
    pub fn sample(&self, frame: i32, fallback: V) -> V {
        if self.keyframes.is_empty() {
            return fallback;
        }
        if self.keyframes.len() == 1 {
            return self.keyframes[0].value.clone();
        }
        if frame <= self.keyframes[0].frame {
            return self.keyframes[0].value.clone();
        }
        let last = self.keyframes.last().expect("non-empty");
        if frame >= last.frame {
            return last.value.clone();
        }

        let Some(b_idx) = self.keyframes.iter().position(|k| k.frame > frame) else {
            return last.value.clone();
        };
        let a = &self.keyframes[b_idx - 1];
        let b = &self.keyframes[b_idx];
        let raw = (i64::from(frame) - i64::from(a.frame)) as f64
            / (i64::from(b.frame) - i64::from(a.frame)) as f64;
        match a.interpolation_out {
            Interpolation::Hold => a.value.clone(),
            Interpolation::Linear => V::keyframe_interpolate(a.value.clone(), b.value.clone(), raw),
            Interpolation::Smooth => {
                V::keyframe_interpolate(a.value.clone(), b.value.clone(), smoothstep(raw))
            }
        }
    }

    /// Drop everything before clip-relative `offset` and make `offset` the new
    /// frame 0, for a clip whose head was cut by `offset` frames. A boundary
    /// keyframe sampled at `offset` keeps the curve continuous and carries the
    /// interpolation of the segment it cuts, so linear and hold spans keep
    /// their shape. `None` for an inactive track. 1:1 port of upstream
    /// `KeyframeTrack.rebased(by:fallback:)`.
    pub fn rebased(&self, offset: i32, fallback: V) -> Option<Self> {
        if !self.is_active() {
            return None;
        }
        let boundary = self.sample(offset, fallback);
        let mut keyframes: Vec<Keyframe<V>> = self
            .keyframes
            .iter()
            .filter(|k| k.frame >= offset)
            .map(|k| {
                Keyframe::with_interpolation(
                    k.frame.saturating_sub(offset),
                    k.value.clone(),
                    k.interpolation_out,
                )
            })
            .collect();
        if keyframes.first().map(|k| k.frame) != Some(0) {
            let interpolation = self
                .keyframes
                .iter()
                .rev()
                .find(|k| k.frame < offset)
                .map_or(Interpolation::Smooth, |k| k.interpolation_out);
            keyframes.insert(0, Keyframe::with_interpolation(0, boundary, interpolation));
        }
        Some(KeyframeTrack::from_keyframes(keyframes))
    }
}

/// Splits a keyframe track at `split_offset` (clip-relative), keeping both halves
/// continuous by inserting a boundary keyframe sampled at the cut. Returns the
/// track unchanged on both sides when it is empty/inactive. Model invariant
/// lifted from upstream `EditorViewModel.splitKeyframeTrack`, whose right half
/// is [`KeyframeTrack::rebased`].
pub fn split_keyframe_track<V: KeyframeInterpolatable + Clone>(
    track: Option<&KeyframeTrack<V>>,
    split_offset: i32,
    fallback: V,
) -> (Option<KeyframeTrack<V>>, Option<KeyframeTrack<V>>) {
    let Some(track) = track.filter(|t| t.is_active()) else {
        return (track.cloned(), track.cloned());
    };
    let boundary = track.sample(split_offset, fallback.clone());

    let mut left_kfs: Vec<Keyframe<V>> = track
        .keyframes
        .iter()
        .filter(|k| k.frame <= split_offset)
        .cloned()
        .collect();
    if left_kfs.last().map(|k| k.frame) != Some(split_offset) {
        left_kfs.push(Keyframe::new(split_offset, boundary));
    }

    (
        if left_kfs.is_empty() {
            None
        } else {
            Some(KeyframeTrack::from_keyframes(left_kfs))
        },
        track.rebased(split_offset, fallback),
    )
}

/// Identifies which clip property an inspector lane / stamp button drives.
/// 1:1 port of `AnimatableProperty`. `display_name` is pure-UI and omitted.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AnimatableProperty {
    Opacity,
    Position,
    Scale,
    Rotation,
    Crop,
    Volume,
}

impl AnimatableProperty {
    pub const ALL: [AnimatableProperty; 6] = [
        AnimatableProperty::Opacity,
        AnimatableProperty::Position,
        AnimatableProperty::Scale,
        AnimatableProperty::Rotation,
        AnimatableProperty::Crop,
        AnimatableProperty::Volume,
    ];
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::Crop;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }

    #[test]
    fn smoothstep_endpoints_and_mid() {
        approx(smoothstep(0.0), 0.0);
        approx(smoothstep(1.0), 1.0);
        approx(smoothstep(0.5), 0.5);
        approx(smoothstep(0.25), 0.25 * 0.25 * (3.0 - 0.5));
    }

    #[test]
    fn keyframe_default_interpolation_is_smooth() {
        let kf = Keyframe::new(0, 1.0);
        assert_eq!(kf.interpolation_out, Interpolation::Smooth);
    }

    #[test]
    fn keyframe_missing_interpolation_decodes_to_smooth() {
        let kf: Keyframe<f64> = serde_json::from_str(r#"{"frame":3,"value":0.5}"#).unwrap();
        assert_eq!(kf.interpolation_out, Interpolation::Smooth);
        assert_eq!(kf.frame, 3);
    }

    #[test]
    fn upsert_keeps_sorted_and_replaces() {
        let mut t = KeyframeTrack::<f64>::new();
        t.upsert(Keyframe::new(10, 1.0));
        t.upsert(Keyframe::new(0, 0.0));
        t.upsert(Keyframe::new(5, 0.5));
        assert_eq!(
            t.keyframes.iter().map(|k| k.frame).collect::<Vec<_>>(),
            [0, 5, 10]
        );
        // Replace in place.
        t.upsert(Keyframe::new(5, 0.9));
        assert_eq!(t.keyframes.len(), 3);
        approx(t.keyframes[1].value, 0.9);
    }

    #[test]
    fn remove_drops_matching_frame() {
        let mut t = KeyframeTrack::<f64>::from_keyframes(vec![
            Keyframe::new(0, 0.0),
            Keyframe::new(5, 0.5),
        ]);
        t.remove(5);
        assert_eq!(t.keyframes.len(), 1);
        t.remove(99); // no-op
        assert_eq!(t.keyframes.len(), 1);
    }

    #[test]
    fn move_keyframe_abandons_on_occupied_target() {
        let mut t = KeyframeTrack::<f64>::from_keyframes(vec![
            Keyframe::new(0, 0.0),
            Keyframe::new(5, 0.5),
        ]);
        // target 5 occupied -> abandoned, nothing changes
        t.move_keyframe(0, 5);
        assert_eq!(
            t.keyframes.iter().map(|k| k.frame).collect::<Vec<_>>(),
            [0, 5]
        );
        approx(t.keyframes[0].value, 0.0);
        // free target -> moves and re-sorts
        t.move_keyframe(0, 10);
        assert_eq!(
            t.keyframes.iter().map(|k| k.frame).collect::<Vec<_>>(),
            [5, 10]
        );
    }

    #[test]
    fn move_keyframe_noop_when_source_missing() {
        let mut t = KeyframeTrack::<f64>::from_keyframes(vec![Keyframe::new(0, 0.0)]);
        t.move_keyframe(99, 100);
        assert_eq!(t.keyframes.iter().map(|k| k.frame).collect::<Vec<_>>(), [0]);
    }

    #[test]
    fn sample_empty_returns_fallback() {
        let t = KeyframeTrack::<f64>::new();
        approx(t.sample(5, 0.42), 0.42);
    }

    #[test]
    fn sample_single_keyframe_returns_its_value() {
        let t = KeyframeTrack::<f64>::from_keyframes(vec![Keyframe::new(3, 0.7)]);
        approx(t.sample(0, 0.0), 0.7);
        approx(t.sample(100, 0.0), 0.7);
    }

    #[test]
    fn sample_clamps_at_endpoints() {
        let t = KeyframeTrack::<f64>::from_keyframes(vec![
            Keyframe::new(10, 0.0),
            Keyframe::new(20, 1.0),
        ]);
        // before first
        approx(t.sample(0, 9.9), 0.0);
        // after last
        approx(t.sample(99, 9.9), 1.0);
        // exactly first / last
        approx(t.sample(10, 0.0), 0.0);
        approx(t.sample(20, 0.0), 1.0);
    }

    #[test]
    fn sample_linear_branch() {
        let t = KeyframeTrack::<f64>::from_keyframes(vec![
            Keyframe::with_interpolation(0, 0.0, Interpolation::Linear),
            Keyframe::new(10, 1.0),
        ]);
        approx(t.sample(5, 0.0), 0.5);
        approx(t.sample(2, 0.0), 0.2);
    }

    #[test]
    fn sample_hold_branch_uses_left_value() {
        let t = KeyframeTrack::<f64>::from_keyframes(vec![
            Keyframe::with_interpolation(0, 0.0, Interpolation::Hold),
            Keyframe::new(10, 1.0),
        ]);
        approx(t.sample(5, 0.0), 0.0);
        approx(t.sample(9, 0.0), 0.0);
    }

    #[test]
    fn sample_smooth_branch_uses_smoothstep() {
        let t = KeyframeTrack::<f64>::from_keyframes(vec![
            Keyframe::with_interpolation(0, 0.0, Interpolation::Smooth),
            Keyframe::new(10, 1.0),
        ]);
        // raw=0.5 -> smoothstep(0.5)=0.5
        approx(t.sample(5, 0.0), 0.5);
        // raw=0.2 -> smoothstep(0.2)=0.104
        approx(t.sample(2, 0.0), smoothstep(0.2));
    }

    #[test]
    fn sample_interpolation_taken_from_left_keyframe() {
        // left=hold means the whole span holds, regardless of right's mode.
        let t = KeyframeTrack::<f64>::from_keyframes(vec![
            Keyframe::with_interpolation(0, 0.0, Interpolation::Hold),
            Keyframe::with_interpolation(10, 1.0, Interpolation::Linear),
            Keyframe::with_interpolation(20, 2.0, Interpolation::Smooth),
        ]);
        approx(t.sample(5, 0.0), 0.0); // first span holds
        approx(t.sample(15, 0.0), 1.5); // second span linear
    }

    #[test]
    fn animpair_interpolation() {
        let p =
            AnimPair::keyframe_interpolate(AnimPair::new(0.0, 10.0), AnimPair::new(1.0, 20.0), 0.5);
        approx(p.a, 0.5);
        approx(p.b, 15.0);
    }

    #[test]
    fn animpair_track_sample() {
        let t = KeyframeTrack::<AnimPair>::from_keyframes(vec![
            Keyframe::with_interpolation(0, AnimPair::new(0.0, 0.0), Interpolation::Linear),
            Keyframe::new(10, AnimPair::new(1.0, 2.0)),
        ]);
        let s = t.sample(5, AnimPair::new(0.0, 0.0));
        approx(s.a, 0.5);
        approx(s.b, 1.0);
    }

    #[test]
    fn crop_track_sample() {
        let t = KeyframeTrack::<Crop>::from_keyframes(vec![
            Keyframe::with_interpolation(
                0,
                Crop {
                    left: 0.0,
                    top: 0.0,
                    right: 0.0,
                    bottom: 0.0,
                },
                Interpolation::Linear,
            ),
            Keyframe::new(
                10,
                Crop {
                    left: 0.2,
                    top: 0.4,
                    right: 0.0,
                    bottom: 0.0,
                },
            ),
        ]);
        let s = t.sample(5, Crop::default());
        approx(s.left, 0.1);
        approx(s.top, 0.2);
    }

    #[test]
    fn split_inactive_track_returns_input_both_sides() {
        let (l, r) = split_keyframe_track::<f64>(None, 5, 0.0);
        assert!(l.is_none() && r.is_none());
        let empty = KeyframeTrack::<f64>::new();
        let (l2, r2) = split_keyframe_track(Some(&empty), 5, 0.0);
        // inactive -> returned as-is (the empty track on both sides)
        assert_eq!(l2, Some(empty.clone()));
        assert_eq!(r2, Some(empty));
    }

    #[test]
    fn split_inserts_boundary_keyframes_and_rebases_right() {
        let track = KeyframeTrack::<f64>::from_keyframes(vec![
            Keyframe::with_interpolation(0, 0.0, Interpolation::Linear),
            Keyframe::with_interpolation(10, 1.0, Interpolation::Linear),
        ]);
        let (left, right) = split_keyframe_track(Some(&track), 5, 0.0);
        let left = left.unwrap();
        let right = right.unwrap();
        // left: kf at 0 and boundary at 5 (value 0.5 from linear sample)
        assert_eq!(
            left.keyframes.iter().map(|k| k.frame).collect::<Vec<_>>(),
            [0, 5]
        );
        approx(left.keyframes[1].value, 0.5);
        // right: rebased to 0 (boundary) and 5 (was 10)
        assert_eq!(
            right.keyframes.iter().map(|k| k.frame).collect::<Vec<_>>(),
            [0, 5]
        );
        approx(right.keyframes[0].value, 0.5);
        approx(right.keyframes[1].value, 1.0);
    }

    #[test]
    fn split_on_existing_keyframe_no_duplicate_boundary() {
        let track = KeyframeTrack::<f64>::from_keyframes(vec![
            Keyframe::new(0, 0.0),
            Keyframe::new(5, 0.5),
            Keyframe::new(10, 1.0),
        ]);
        let (left, right) = split_keyframe_track(Some(&track), 5, 0.0);
        let left = left.unwrap();
        let right = right.unwrap();
        assert_eq!(
            left.keyframes.iter().map(|k| k.frame).collect::<Vec<_>>(),
            [0, 5]
        );
        assert_eq!(
            right.keyframes.iter().map(|k| k.frame).collect::<Vec<_>>(),
            [0, 5]
        );
    }

    #[test]
    fn normalized_sorts_by_frame_and_the_last_duplicate_wins() {
        let t = KeyframeTrack::<f64>::from_keyframes(vec![
            Keyframe::with_interpolation(30, 1.0, Interpolation::Linear),
            Keyframe::with_interpolation(0, 0.0, Interpolation::Linear),
            Keyframe::with_interpolation(30, 0.8, Interpolation::Hold),
            Keyframe::with_interpolation(0, 0.25, Interpolation::Linear),
        ])
        .normalized();
        assert_eq!(
            t.keyframes
                .iter()
                .map(|k| (k.frame, k.value, k.interpolation_out))
                .collect::<Vec<_>>(),
            [
                (0, 0.25, Interpolation::Linear),
                (30, 0.8, Interpolation::Hold),
            ]
        );
        // The sorted track samples the ramp instead of the stale first row.
        approx(t.sample(15, 9.9), 0.25 + (0.8 - 0.25) * 0.5);
    }

    #[test]
    fn validate_rejects_frames_outside_the_clip_and_non_finite_values() {
        let valid = KeyframeTrack::<f64>::from_keyframes(vec![
            Keyframe::new(0, 0.0),
            Keyframe::new(30, 1.0),
        ]);
        assert!(valid.validate(30).is_ok());
        assert!(valid.validate(29).unwrap_err().contains("keyframes[1]"));
        let negative = KeyframeTrack::<f64>::from_keyframes(vec![Keyframe::new(-1, 0.0)]);
        assert!(negative.validate(30).is_err());
        let non_finite = KeyframeTrack::<f64>::from_keyframes(vec![Keyframe::new(3, f64::NAN)]);
        assert!(non_finite.validate(30).unwrap_err().contains("finite"));
        let pair = KeyframeTrack::from_keyframes(vec![Keyframe::new(
            0,
            AnimPair::new(0.0, f64::INFINITY),
        )]);
        assert!(pair.validate(30).is_err());
        let crop = KeyframeTrack::from_keyframes(vec![Keyframe::new(
            0,
            Crop {
                left: f64::NEG_INFINITY,
                ..Crop::default()
            },
        )]);
        assert!(crop.validate(30).is_err());
    }

    #[test]
    fn sample_between_extreme_frames_does_not_overflow() {
        let t = KeyframeTrack::<f64>::from_keyframes(vec![
            Keyframe::with_interpolation(i32::MIN, 0.0, Interpolation::Linear),
            Keyframe::new(i32::MAX, 1.0),
        ]);
        let mid = t.sample(0, 9.9);
        assert!((mid - 0.5).abs() < 1e-6, "{mid}");
        approx(t.sample(i32::MIN, 9.9), 0.0);
        approx(t.sample(i32::MAX, 9.9), 1.0);
    }

    #[test]
    fn rebased_boundary_carries_the_cut_segments_interpolation() {
        let t = KeyframeTrack::<f64>::from_keyframes(vec![
            Keyframe::with_interpolation(0, 0.0, Interpolation::Linear),
            Keyframe::with_interpolation(100, 1.0, Interpolation::Linear),
        ]);
        let rebased = t.rebased(20, 9.9).unwrap();
        assert_eq!(
            rebased
                .keyframes
                .iter()
                .map(|k| (k.frame, k.interpolation_out))
                .collect::<Vec<_>>(),
            [(0, Interpolation::Linear), (80, Interpolation::Linear)]
        );
        approx(rebased.keyframes[0].value, 0.2);
        // The linear ramp keeps its shape: original frame 50 == rebased 30.
        approx(rebased.sample(30, 9.9), t.sample(50, 9.9));

        let hold = KeyframeTrack::<f64>::from_keyframes(vec![
            Keyframe::with_interpolation(0, 0.3, Interpolation::Hold),
            Keyframe::new(10, 1.0),
        ])
        .rebased(4, 9.9)
        .unwrap();
        assert_eq!(hold.keyframes[0].interpolation_out, Interpolation::Hold);
        approx(hold.sample(5, 9.9), 0.3);
    }

    #[test]
    fn rebased_keeps_a_keyframe_on_the_offset_and_defaults_before_the_track() {
        let t = KeyframeTrack::<f64>::from_keyframes(vec![
            Keyframe::new(0, 0.0),
            Keyframe::with_interpolation(20, 0.5, Interpolation::Hold),
            Keyframe::new(40, 1.0),
        ]);
        let on_keyframe = t.rebased(20, 9.9).unwrap();
        assert_eq!(
            on_keyframe
                .keyframes
                .iter()
                .map(|k| (k.frame, k.value, k.interpolation_out))
                .collect::<Vec<_>>(),
            [
                (0, 0.5, Interpolation::Hold),
                (20, 1.0, Interpolation::Smooth)
            ]
        );

        let late = KeyframeTrack::<f64>::from_keyframes(vec![Keyframe::with_interpolation(
            50,
            0.7,
            Interpolation::Linear,
        )])
        .rebased(20, 9.9)
        .unwrap();
        assert_eq!(
            late.keyframes
                .iter()
                .map(|k| (k.frame, k.value, k.interpolation_out))
                .collect::<Vec<_>>(),
            [
                (0, 0.7, Interpolation::Smooth),
                (30, 0.7, Interpolation::Linear)
            ]
        );
        assert!(KeyframeTrack::<f64>::new().rebased(20, 9.9).is_none());
    }

    #[test]
    fn split_right_half_keeps_the_cut_segments_interpolation() {
        let track = KeyframeTrack::<f64>::from_keyframes(vec![
            Keyframe::with_interpolation(0, 0.0, Interpolation::Linear),
            Keyframe::with_interpolation(10, 1.0, Interpolation::Linear),
        ]);
        let (_, right) = split_keyframe_track(Some(&track), 5, 0.0);
        let right = right.unwrap();
        assert_eq!(right.keyframes[0].interpolation_out, Interpolation::Linear);
        approx(right.sample(2, 9.9), track.sample(7, 9.9));
    }

    #[test]
    fn keyframe_track_roundtrip_json() {
        let t = KeyframeTrack::<f64>::from_keyframes(vec![
            Keyframe::with_interpolation(0, 0.0, Interpolation::Hold),
            Keyframe::new(7, 1.0),
        ]);
        let json = serde_json::to_string(&t).unwrap();
        assert!(json.contains("\"interpolationOut\":\"hold\""));
        let back: KeyframeTrack<f64> = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }
}
