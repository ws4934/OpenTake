use opentake_domain::{
    Clip, ClipType, MediaManifest, MediaManifestEntry, MediaSource, Timeline, Track, Transition,
    TransitionKind,
};
use opentake_ops::{apply, ClipEntry, EditCommand, EditorState, FrameRange, SeqIdGen};

fn state(clips: Vec<Clip>) -> EditorState {
    let mut track = Track::new("video", ClipType::Video);
    track.clips = clips;
    let mut timeline = Timeline::new();
    timeline.tracks.push(track);
    let mut manifest = MediaManifest::new();
    // Placement commands require their media in the manifest.
    manifest.entries.push(MediaManifestEntry {
        id: "still".into(),
        name: "still".into(),
        kind: ClipType::Image,
        source: MediaSource::External {
            absolute_path: "/still.png".into(),
        },
        duration: 1.0,
        generation_input: None,
        source_width: None,
        source_height: None,
        source_fps: None,
        has_audio: None,
        color: None,
        proxy: None,
        folder_id: None,
        cached_remote_url: None,
        cached_remote_url_expires_at: None,
    });
    EditorState::new(timeline, manifest)
}

fn assert_no_overlap(timeline: &Timeline) {
    for track in &timeline.tracks {
        let mut clips: Vec<_> = track.clips.iter().collect();
        clips.sort_by_key(|clip| clip.start_frame);
        for pair in clips.windows(2) {
            assert!(pair[0].end_frame() <= pair[1].start_frame, "{pair:?}");
        }
    }
}

#[test]
fn extending_trim_overwrites_neighbors_and_one_undo_restores_them() {
    for (reversed, neighbor_duration) in [(false, 100), (false, 40), (true, 100)] {
        let mut clip = Clip::new("a", "asset", 0, 100);
        clip.reversed = reversed;
        if reversed {
            clip.trim_start_frame = 50;
        } else {
            clip.trim_end_frame = 50;
        }
        let mut state = state(vec![clip, Clip::new("b", "asset", 100, neighbor_duration)]);
        let before = state.timeline.clone();
        let ids = SeqIdGen::default();
        apply(
            &mut state,
            EditCommand::TrimClips {
                edits: vec![("a".into(), 0, 0)],
            },
            &ids,
        )
        .unwrap();
        assert_no_overlap(&state.timeline);
        assert_eq!(state.timeline.tracks[0].clips[0].end_frame(), 150);
        if neighbor_duration == 40 {
            assert_eq!(state.timeline.tracks[0].clips.len(), 1);
        } else {
            let neighbor = state.timeline.tracks[0]
                .clips
                .iter()
                .find(|clip| clip.id == "b")
                .unwrap();
            assert_eq!((neighbor.start_frame, neighbor.end_frame()), (150, 200));
        }
        assert_eq!(state.undo_depth(), 1);
        apply(&mut state, EditCommand::Undo, &ids).unwrap();
        assert_eq!(state.timeline, before);
    }
}

#[test]
fn image_left_extension_trims_the_previous_clip() {
    let mut image = Clip::new("image", "asset", 100, 50);
    image.media_type = ClipType::Image;
    let mut state = state(vec![Clip::new("previous", "asset", 0, 100), image]);
    let ids = SeqIdGen::default();
    apply(
        &mut state,
        EditCommand::TrimClips {
            edits: vec![("image".into(), -30, 0)],
        },
        &ids,
    )
    .unwrap();
    assert_no_overlap(&state.timeline);
    assert_eq!(state.timeline.tracks[0].clips[0].end_frame(), 70);
    assert_eq!(state.timeline.tracks[0].clips[1].start_frame, 70);
}

#[test]
fn batch_trim_preserves_targets_and_refuses_conflicting_final_bounds() {
    let mut first = Clip::new("a", "asset", 0, 100);
    first.trim_end_frame = 50;
    let mut state = state(vec![first, Clip::new("b", "asset", 100, 100)]);
    let before = state.timeline.clone();
    let ids = SeqIdGen::default();
    let error = apply(
        &mut state,
        EditCommand::TrimClips {
            edits: vec![("a".into(), 0, 0), ("b".into(), 0, 0)],
        },
        &ids,
    );
    assert!(error.is_err());
    assert_eq!(state.timeline, before);
    assert_eq!(state.undo_depth(), 0);
    assert_eq!(ids.count(), 0);
    apply(
        &mut state,
        EditCommand::TrimClips {
            edits: vec![("a".into(), 0, 0), ("b".into(), 50, 0)],
        },
        &ids,
    )
    .unwrap();
    assert_eq!(state.timeline.tracks[0].clips.len(), 2);
    assert_no_overlap(&state.timeline);
    apply(&mut state, EditCommand::Undo, &ids).unwrap();
    assert_eq!(state.timeline, before);
}

#[test]
fn linked_video_audio_trim_clears_both_tracks_and_undo_restores_the_pair() {
    let mut timeline = Timeline::new();
    for (id, kind) in [("video", ClipType::Video), ("audio", ClipType::Audio)] {
        let mut clip = Clip::new(id, "asset", 0, 100);
        clip.media_type = kind;
        clip.source_clip_type = kind;
        clip.link_group_id = Some("pair".into());
        clip.trim_end_frame = 50;
        let mut neighbor = Clip::new(format!("{id}-neighbor"), "asset", 100, 100);
        neighbor.media_type = kind;
        neighbor.source_clip_type = kind;
        let mut track = Track::new(id, kind);
        track.clips = vec![clip, neighbor];
        timeline.tracks.push(track);
    }
    let before = timeline.clone();
    let mut state = EditorState::from_timeline(timeline);
    let ids = SeqIdGen::default();
    apply(
        &mut state,
        EditCommand::TrimClips {
            edits: vec![("video".into(), 0, 0), ("audio".into(), 0, 0)],
        },
        &ids,
    )
    .unwrap();
    assert_no_overlap(&state.timeline);
    for track in &state.timeline.tracks {
        assert_eq!(track.clips.len(), 2);
        assert_eq!(track.clips[0].link_group_id.as_deref(), Some("pair"));
        assert_eq!(track.clips[0].end_frame(), 150);
        assert_eq!(track.clips[1].start_frame, 150);
    }
    assert_eq!(state.undo_depth(), 1);
    apply(&mut state, EditCommand::Undo, &ids).unwrap();
    assert_eq!(state.timeline, before);
}

#[test]
fn repeated_target_edits_only_clear_the_final_expansion() {
    let mut clip = Clip::new("a", "asset", 0, 100);
    clip.trim_end_frame = 50;
    let mut state = state(vec![clip, Clip::new("b", "asset", 100, 100)]);
    let before = state.timeline.clone();
    let ids = SeqIdGen::default();
    let result = apply(
        &mut state,
        EditCommand::TrimClips {
            edits: vec![("a".into(), 0, 0), ("a".into(), 0, 50)],
        },
        &ids,
    )
    .unwrap();
    assert!(!result.changed);
    assert_eq!(state.timeline, before);
    assert_eq!(state.undo_depth(), 0);
    assert_eq!(ids.count(), 0);
}

fn transition_state() -> EditorState {
    let mut from = Clip::new("a", "asset", 0, 100);
    from.transition_out = Some(Transition {
        from_clip_id: "a".into(),
        to_clip_id: "b".into(),
        kind: TransitionKind::CrossDissolve,
        duration_frames: 10,
    });
    state(vec![
        from,
        Clip::new("b", "asset", 100, 40),
        Clip::new("c", "asset", 140, 40),
    ])
}

fn entry() -> ClipEntry {
    ClipEntry {
        media_ref: "still".into(),
        media_type: ClipType::Image,
        source_clip_type: ClipType::Image,
        track_index: 0,
        start_frame: 20,
        duration_frames: 10,
        trim_start_frame: None,
        trim_end_frame: None,
        has_audio: false,
        add_linked_audio: false,
        transform: None,
    }
}

#[test]
fn outgoing_transition_follows_the_surviving_tail_in_every_split_path() {
    let commands = [
        EditCommand::SplitClip {
            clip_id: "a".into(),
            at_frame: 20,
        },
        EditCommand::SplitClips {
            clip_ids: vec!["a".into()],
            at_frame: 20,
        },
        EditCommand::AddClips {
            entries: vec![entry()],
        },
        EditCommand::InsertClips {
            track_index: 0,
            at_frame: 20,
            entries: vec![entry()],
        },
        EditCommand::FreezeFrame {
            clip_id: "a".into(),
            at_frame: 20,
            duration_frames: 10,
            media_ref: "still".into(),
        },
        EditCommand::RippleDeleteRanges {
            track_index: 0,
            ranges: vec![FrameRange::new(20, 30)],
        },
    ];
    for command in commands {
        let mut state = transition_state();
        let before = state.timeline.clone();
        let ids = SeqIdGen::new("split-");
        apply(&mut state, command.clone(), &ids).unwrap();
        let clips = &state.timeline.tracks[0].clips;
        let owners: Vec<_> = clips
            .iter()
            .filter(|clip| clip.transition_out.is_some())
            .collect();
        assert_eq!(owners.len(), 1, "{command:?}");
        let tail = owners[0];
        assert_ne!(tail.id, "a", "{command:?}");
        let transition = tail.transition_out.as_ref().unwrap();
        assert_eq!(transition.from_clip_id, tail.id);
        assert_eq!(transition.to_clip_id, "b");
        assert_eq!(
            tail.end_frame(),
            clips
                .iter()
                .find(|clip| clip.id == "b")
                .unwrap()
                .start_frame
        );
        assert_no_overlap(&state.timeline);
        apply(&mut state, EditCommand::Undo, &ids).unwrap();
        assert_eq!(state.timeline, before, "{command:?}");
    }
}

#[test]
fn ripple_deletion_removes_dangling_transitions_and_undo_restores_them() {
    for command in [
        EditCommand::RippleDeleteClips {
            clip_ids: vec!["b".into()],
        },
        EditCommand::RippleDeleteRanges {
            track_index: 0,
            ranges: vec![FrameRange::new(100, 140)],
        },
    ] {
        let mut state = transition_state();
        let before = state.timeline.clone();
        let ids = SeqIdGen::default();
        apply(&mut state, command.clone(), &ids).unwrap();
        assert!(
            state.timeline.tracks[0]
                .clips
                .iter()
                .all(|clip| clip.transition_out.is_none()),
            "{command:?}"
        );
        apply(&mut state, EditCommand::Undo, &ids).unwrap();
        assert_eq!(state.timeline, before);
    }
}
