//! Internal editing operations — the building blocks [`crate::command`] composes
//! into transactions. Each is a direct port of an `EditorViewModel` method,
//! stripped of AppKit/undo glue: they mutate the `Timeline` (or `MediaManifest`)
//! in place, and the command layer snapshots/commits around them.

pub mod clear_region;
pub mod duplicate;
pub mod folders;
pub mod linking;
pub mod move_clips;
pub mod place;
pub mod ripple;
pub mod settings;
pub mod split;
pub mod swap;
pub mod tracks;
pub mod trim;

/// Whether `clip`'s frame arithmetic is representable: the domain rule
/// [`opentake_domain::clip_frame_arithmetic`], shared by every op that
/// refuses to touch a malformed timeline.
pub(crate) fn clip_arithmetic_is_safe(clip: &opentake_domain::Clip) -> bool {
    clip.frame_arithmetic().is_ok()
}

pub use clear_region::clear_region;
pub(crate) use clear_region::{clear_region_validated, remove_clips};
pub use duplicate::duplicate_clips;
pub use folders::{
    create_folder, delete_folder, delete_media, move_to_folder, rename_folder, rename_media,
};
pub use linking::{
    expand_to_link_group, link_index, linked_partner_ids, partner_moves,
    timing_propagation_partners,
};
pub use move_clips::{move_clips, ClipMove};
pub(crate) use place::place_clip_validated;
pub use place::{place_clip, sort_clips, PlaceSpec};
pub use ripple::{
    apply_shifts, ripple_delete, ripple_delete_ranges_on_track, ripple_insert, validate_shifts,
    RippleOutcome, RippleRangesReport,
};
pub use settings::set_timeline_settings;
pub use split::{split_clip, split_single_clip};
pub use swap::{swap_clip_positions, swap_tracks};
pub use tracks::{
    available_audio_track_index, insert_track, prune_empty_tracks, remove_tracks,
    resolve_or_create_audio_track, zones, ZoneLayout,
};
pub use trim::{trim_clip_internal, trim_clips, trim_values, TrimEdge};

#[cfg(test)]
mod tests {
    use super::clip_arithmetic_is_safe;
    use opentake_domain::{Clip, ClipType};

    /// The guard each op carried its own copy of before they shared the
    /// domain rule.
    fn previous_clip_arithmetic_is_safe(clip: &Clip) -> bool {
        if clip.start_frame < 0
            || clip.duration_frames < 1
            || (!matches!(clip.media_type, ClipType::Image | ClipType::Text)
                && (clip.trim_start_frame < 0 || clip.trim_end_frame < 0))
            || !clip.speed.is_finite()
            || clip.speed <= 0.0
            || clip.start_frame.checked_add(clip.duration_frames).is_none()
            || clip
                .duration_frames
                .checked_add(clip.trim_start_frame)
                .and_then(|value| value.checked_add(clip.trim_end_frame))
                .is_none()
        {
            return false;
        }
        let consumed = (clip.duration_frames as f64 * clip.speed).round();
        if !(0.0..=i32::MAX as f64).contains(&consumed) {
            return false;
        }
        let consumed = consumed as i32;
        clip.trim_start_frame.checked_add(consumed).is_some()
            && clip.trim_end_frame.checked_add(consumed).is_some()
            && clip
                .trim_start_frame
                .checked_add(consumed)
                .and_then(|value| value.checked_add(clip.trim_end_frame))
                .is_some()
    }

    #[test]
    fn shared_guard_matches_the_per_op_copies() {
        let trims = [i32::MIN, -4, 0, 5, i32::MAX - 8, i32::MAX];
        for start in [-1, 0, 7, i32::MAX - 3, i32::MAX] {
            for duration in [-2, 0, 1, 9, i32::MAX - 1, i32::MAX] {
                for trim_start in trims {
                    for trim_end in trims {
                        for speed in [f64::NAN, -1.0, 0.0, 1e-300, 0.5, 1.0, 2.5, 1e12] {
                            for kind in [
                                ClipType::Video,
                                ClipType::Audio,
                                ClipType::Image,
                                ClipType::Text,
                            ] {
                                let mut clip = Clip::new("c", "m", start, duration);
                                clip.trim_start_frame = trim_start;
                                clip.trim_end_frame = trim_end;
                                clip.speed = speed;
                                clip.media_type = kind;
                                assert_eq!(
                                    clip_arithmetic_is_safe(&clip),
                                    previous_clip_arithmetic_is_safe(&clip),
                                    "{clip:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
