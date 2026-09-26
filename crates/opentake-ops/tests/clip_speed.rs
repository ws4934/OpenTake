//! Retime preserves animation and treats an adjacent chain as one edit.
use opentake_domain::{
    Clip, ClipType, Interpolation, Keyframe, KeyframeTrack, MediaManifest, Timeline, Track,
};
use opentake_ops::{apply, ClipProperties, EditCommand, EditorState, SeqIdGen};

fn animated(id: &str, start: i32) -> Clip {
    let mut clip = Clip::new(id, "asset", start, 30);
    let mut track = KeyframeTrack::new();
    track.upsert(Keyframe::with_interpolation(0, 0.0, Interpolation::Linear));
    track.upsert(Keyframe::with_interpolation(30, 1.0, Interpolation::Linear));
    clip.opacity_track = Some(track.clone());
    clip.volume_track = Some(track);
    clip.fade_in_frames = 10;
    clip.fade_out_frames = 20;
    clip
}

fn state(clips: Vec<Clip>) -> EditorState {
    let mut timeline = Timeline::new();
    let mut track = Track::new("video", ClipType::Video);
    track.clips = clips;
    timeline.tracks.push(track);
    EditorState::new(timeline, MediaManifest::new())
}

fn edit(state: &mut EditorState, command: EditCommand) {
    apply(state, command, &SeqIdGen::new("speed-")).unwrap();
}

fn speed_command(ids: &[&str], speed: f64, ripple: bool) -> EditCommand {
    EditCommand::SetClipSpeed {
        clip_ids: ids.iter().map(|id| (*id).into()).collect(),
        speed,
        ripple,
    }
}

fn keyframes(clip: &Clip) -> Vec<(i32, f64)> {
    clip.opacity_track
        .as_ref()
        .unwrap()
        .keyframes
        .iter()
        .map(|key| (key.frame, key.value))
        .collect()
}

#[test]
fn speeding_up_and_slowing_down_rescales_animation_and_adjacent_chain_atomically() {
    for (speed, duration) in [(2.0, 15), (0.5, 60)] {
        let mut state = state(vec![animated("a", 0), animated("b", 30), animated("c", 60)]);
        let before = state.timeline.clone();
        edit(&mut state, speed_command(&["a"], speed, true));
        let clips = &state.timeline.tracks[0].clips;
        assert_eq!(clips[0].speed, speed);
        assert_eq!(clips[0].duration_frames, duration);
        assert_eq!(keyframes(&clips[0]), [(0, 0.0), (duration, 1.0)]);
        assert_eq!(
            clips[0].volume_track.as_ref().unwrap().keyframes[1].frame,
            duration
        );
        assert_eq!(clips[1].start_frame, duration);
        assert_eq!(clips[2].start_frame, duration + 30);
        assert_eq!(keyframes(&clips[1]), [(0, 0.0), (30, 1.0)]);
        assert_eq!(clips[0].fade_in_frames, 10);
        assert_eq!(clips[0].fade_out_frames, if speed == 2.0 { 5 } else { 20 });
        let after = state.timeline.clone();
        assert_eq!(state.version(), 1);
        edit(&mut state, EditCommand::Undo);
        assert_eq!(state.timeline, before);
        assert!(!state.can_undo());
        edit(&mut state, EditCommand::Redo);
        assert_eq!(state.timeline, after);
    }
}

#[test]
fn linked_audio_retimes_once_and_moves_its_own_following_chain() {
    for (speed, duration) in [(2.0, 15), (0.5, 60)] {
        let mut state = state(vec![animated("video-a", 0), animated("video-b", 30)]);
        state.timeline.tracks[0].clips[0].link_group_id = Some("av".into());
        let mut audio = Track::new("audio", ClipType::Audio);
        audio.clips = vec![animated("audio-a", 0), animated("audio-b", 30)];
        for clip in &mut audio.clips {
            clip.media_type = ClipType::Audio;
            clip.source_clip_type = ClipType::Audio;
        }
        audio.clips[0].link_group_id = Some("av".into());
        state.timeline.tracks.push(audio);
        let before = state.timeline.clone();
        // Directly selecting the linked partner as well must not double-retime.
        edit(
            &mut state,
            speed_command(&["video-a", "audio-a", "video-a"], speed, true),
        );
        for track in &state.timeline.tracks {
            assert_eq!(track.clips[0].duration_frames, duration);
            assert_eq!(track.clips[1].start_frame, duration);
            assert_eq!(keyframes(&track.clips[0]), [(0, 0.0), (duration, 1.0)]);
        }
        edit(&mut state, EditCommand::Undo);
        assert_eq!(state.timeline, before);
        // Selecting only video must also reach audio through link propagation.
        edit(&mut state, speed_command(&["video-a"], speed, true));
        assert_eq!(state.timeline.tracks[1].clips[0].duration_frames, duration);
        assert_eq!(state.timeline.tracks[1].clips[1].start_frame, duration);
    }
}

#[test]
fn multiple_targets_use_current_positions_in_caller_order() {
    for ids in [["a", "b"], ["b", "a"]] {
        let mut state = state(vec![animated("a", 0), animated("b", 30), animated("c", 60)]);
        edit(&mut state, speed_command(&ids, 2.0, true));
        let clips = &state.timeline.tracks[0].clips;
        assert_eq!(
            clips
                .iter()
                .map(|clip| (clip.start_frame, clip.duration_frames))
                .collect::<Vec<_>>(),
            [(0, 15), (15, 15), (30, 30)]
        );
        assert_eq!(state.version(), 1);
    }
}

#[test]
fn gap_and_explicit_non_ripple_mode_leave_other_clips_in_place() {
    for (next_start, ripple) in [(45, true), (30, false)] {
        let mut state = state(vec![animated("a", 0), animated("b", next_start)]);
        edit(&mut state, speed_command(&["a"], 2.0, ripple));
        assert_eq!(state.timeline.tracks[0].clips[0].duration_frames, 15);
        assert_eq!(state.timeline.tracks[0].clips[1].start_frame, next_start);
    }
}

fn assert_rejected(state: &mut EditorState, command: EditCommand) {
    let before = state.timeline.clone();
    let manifest = state.manifest.clone();
    let version = state.version();
    let history = (state.can_undo(), state.can_redo());
    assert!(apply(state, command, &SeqIdGen::new("speed-")).is_err());
    assert_eq!(state.timeline, before);
    assert_eq!(state.manifest, manifest);
    assert_eq!(state.version(), version);
    assert_eq!((state.can_undo(), state.can_redo()), history);
}

#[test]
fn growing_into_an_unrelated_clip_refuses_without_overwrite_or_history_changes() {
    for (clips, ripple) in [
        (vec![animated("a", 0), animated("gap", 45)], true),
        (
            vec![animated("a", 0), animated("b", 30), animated("gap", 75)],
            true,
        ),
        (vec![animated("a", 0), animated("b", 30)], false),
    ] {
        let mut state = state(clips);
        assert_rejected(&mut state, speed_command(&["a"], 0.5, ripple));
    }
}

#[test]
fn a_later_target_collision_rolls_back_earlier_targets_and_preserves_redo() {
    let mut state = state(vec![
        animated("a", 0),
        animated("b", 30),
        animated("gap", 100),
    ]);
    edit(&mut state, speed_command(&["gap"], 2.0, true));
    edit(&mut state, EditCommand::Undo);
    let before = state.timeline.clone();
    assert_rejected(&mut state, speed_command(&["a", "b"], 0.5, true));
    assert_eq!(state.timeline, before);
    edit(&mut state, EditCommand::Redo);
    assert_eq!(state.timeline.tracks[0].clips[2].duration_frames, 15);
}

#[test]
fn invalid_speed_missing_targets_nested_retime_and_frame_overflow_are_atomic() {
    let mut state = state(vec![animated("a", 0)]);
    for speed in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::MIN_POSITIVE] {
        assert_rejected(&mut state, speed_command(&["a"], speed, true));
    }
    assert_rejected(&mut state, speed_command(&[], 2.0, true));
    assert_rejected(&mut state, speed_command(&["a", "missing"], 2.0, true));
    state.timeline.tracks[0].clips[0].nested_sequence_id = Some("nested".into());
    assert_rejected(&mut state, speed_command(&["a"], 2.0, true));
    state.timeline.tracks[0].clips[0].nested_sequence_id = None;
    state.timeline.tracks[0].clips[0].start_frame = i32::MAX - 60;
    state.timeline.tracks[0]
        .clips
        .push(animated("b", i32::MAX - 30));
    assert_rejected(&mut state, speed_command(&["a"], 0.5, true));
}

#[test]
fn unchanged_speed_does_not_clear_redo_or_create_an_undo_step() {
    let mut state = state(vec![animated("a", 0), animated("b", 30)]);
    edit(&mut state, speed_command(&["a"], 2.0, true));
    edit(&mut state, EditCommand::Undo);
    let version = state.version();
    let before = state.timeline.clone();
    edit(&mut state, speed_command(&["a"], 1.0, true));
    assert_eq!(state.timeline, before);
    assert_eq!(state.version(), version);
    assert!(!state.can_undo());
    assert!(state.can_redo());
}

#[test]
fn agent_speed_properties_rescale_animation_but_do_not_ripple() {
    let mut state = state(vec![animated("a", 0), animated("b", 30)]);
    edit(
        &mut state,
        EditCommand::SetClipProperties {
            clip_ids: vec!["a".into()],
            properties: Box::new(ClipProperties {
                speed: Some(2.0),
                ..Default::default()
            }),
        },
    );
    let clips = &state.timeline.tracks[0].clips;
    assert_eq!(clips[0].duration_frames, 15);
    assert_eq!(keyframes(&clips[0]), [(0, 0.0), (15, 1.0)]);
    assert_eq!(clips[1].start_frame, 30);
}

#[test]
fn agent_explicit_duration_rescales_animation_without_moving_other_clips() {
    let mut state = state(vec![animated("a", 0), animated("b", 30)]);
    edit(
        &mut state,
        EditCommand::SetClipProperties {
            clip_ids: vec!["a".into()],
            properties: Box::new(ClipProperties {
                duration_frames: Some(15),
                ..Default::default()
            }),
        },
    );
    let clips = &state.timeline.tracks[0].clips;
    assert_eq!(keyframes(&clips[0]), [(0, 0.0), (15, 1.0)]);
    assert_eq!(clips[1].start_frame, 30);
}
