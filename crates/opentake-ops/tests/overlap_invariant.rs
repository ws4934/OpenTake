//! Same-track overlap invariant (issue #26, item 1): swaps, batch moves,
//! duplicates and pastes must never leave two non-text clips overlapping on
//! one track. Text clips may overlap by design (the render plan exempts them).

use opentake_domain::{
    Clip, ClipType, MediaManifest, MediaManifestEntry, MediaSource, Timeline, Track,
};
use opentake_ops::command::PasteClipEntry;
use opentake_ops::{
    apply, ClipMove, ClipProperties, EditCommand, EditError, EditorState, SeqIdGen,
};

fn clip(id: &str, start: i32, duration: i32) -> Clip {
    Clip::new(id, "asset", start, duration)
}

fn text(id: &str, start: i32, duration: i32) -> Clip {
    let mut clip = clip(id, start, duration);
    clip.media_type = ClipType::Text;
    clip.source_clip_type = ClipType::Text;
    clip.media_ref.clear();
    clip
}

/// Every non-text clip references the video asset `asset`, so pasted copies
/// validate against the manifest.
fn state(tracks: Vec<Track>) -> EditorState {
    let mut timeline = Timeline::new();
    timeline.tracks = tracks;
    let mut manifest = MediaManifest::new();
    manifest.entries.push(MediaManifestEntry {
        id: "asset".into(),
        name: "asset".into(),
        kind: ClipType::Video,
        source: MediaSource::External {
            absolute_path: "/abs/asset.mp4".into(),
        },
        duration: 600.0,
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

fn video_track(id: &str, clips: Vec<Clip>) -> Track {
    let mut track = Track::new(id, ClipType::Video);
    track.clips = clips;
    track
}

/// The first two non-text clips on one track that overlap.
fn overlap(timeline: &Timeline) -> Option<(String, String, String)> {
    for track in &timeline.tracks {
        let mut clips: Vec<&Clip> = track
            .clips
            .iter()
            .filter(|clip| clip.media_type != ClipType::Text)
            .collect();
        clips.sort_by_key(|clip| clip.start_frame);
        let mut furthest: Option<&Clip> = None;
        for clip in clips {
            if let Some(previous) = furthest {
                if clip.start_frame < previous.end_frame() {
                    return Some((track.id.clone(), previous.id.clone(), clip.id.clone()));
                }
                if clip.end_frame() <= previous.end_frame() {
                    continue;
                }
            }
            furthest = Some(clip);
        }
    }
    None
}

fn span(state: &EditorState, id: &str) -> (i32, i32) {
    let clip = state
        .timeline
        .tracks
        .iter()
        .flat_map(|track| &track.clips)
        .find(|clip| clip.id == id)
        .unwrap_or_else(|| panic!("clip {id} exists"));
    (clip.start_frame, clip.end_frame())
}

fn assert_rejected_without_change(state: &mut EditorState, command: EditCommand) -> String {
    let before = state.timeline.clone();
    let (version, depth) = (state.version(), state.undo_depth());
    let ids = SeqIdGen::new("rejected-");
    let error = apply(state, command, &ids).expect_err("command must be rejected");
    assert_eq!(state.timeline, before);
    assert_eq!((state.version(), state.undo_depth()), (version, depth));
    assert_eq!(ids.count(), 0);
    error.to_string()
}

#[test]
fn same_track_swap_of_different_lengths_is_refused() {
    // The issue's reproduction: b would land on [0,30) over a's new [10,20).
    let mut state = state(vec![video_track(
        "v",
        vec![clip("a", 0, 10), clip("b", 10, 30)],
    )]);
    let before = state.timeline.clone();
    let result = apply(
        &mut state,
        EditCommand::SwapClips {
            a: "a".into(),
            b: "b".into(),
        },
        &SeqIdGen::default(),
    )
    .unwrap();
    assert!(!result.changed);
    assert_eq!(state.timeline, before);
    assert_eq!(state.undo_depth(), 0);
}

#[test]
fn same_track_swap_that_fits_exchanges_start_frames() {
    let mut state = state(vec![video_track(
        "v",
        vec![clip("a", 0, 10), clip("b", 50, 30)],
    )]);
    let result = apply(
        &mut state,
        EditCommand::SwapClips {
            a: "a".into(),
            b: "b".into(),
        },
        &SeqIdGen::default(),
    )
    .unwrap();
    assert!(result.changed);
    assert_eq!(span(&state, "a"), (50, 60));
    assert_eq!(span(&state, "b"), (0, 30));
}

#[test]
fn duplicating_a_batch_before_frame_zero_keeps_the_copies_apart() {
    // The issue's reproduction: both copies used to land on [0,30).
    let mut state = state(vec![video_track(
        "v",
        vec![clip("a", 0, 30), clip("b", 30, 30)],
    )]);
    let result = apply(
        &mut state,
        EditCommand::DuplicateClips {
            clip_ids: vec!["a".into(), "b".into()],
            offset_frames: -100,
            target_track_indexes: vec![0, 0],
        },
        &SeqIdGen::new("copy-"),
    )
    .unwrap();
    assert_eq!(overlap(&state.timeline), None);
    let copies: Vec<(i32, i32)> = result
        .affected_clip_ids
        .iter()
        .map(|id| span(&state, id))
        .collect();
    assert_eq!(copies, vec![(0, 30), (30, 60)]);
}

#[test]
fn moving_a_batch_before_frame_zero_clamps_the_group_not_each_clip() {
    // The issue's reproduction: a and b both used to become [0,30).
    let mut state = state(vec![video_track(
        "v",
        vec![clip("a", 0, 30), clip("b", 30, 30), clip("c", 100, 10)],
    )]);
    apply(
        &mut state,
        EditCommand::MoveClips {
            moves: vec![
                ClipMove {
                    clip_id: "a".into(),
                    to_track: 0,
                    to_frame: -40,
                },
                ClipMove {
                    clip_id: "b".into(),
                    to_track: 0,
                    to_frame: -10,
                },
            ],
        },
        &SeqIdGen::default(),
    )
    .unwrap();
    assert_eq!(overlap(&state.timeline), None);
    assert_eq!(span(&state, "a"), (0, 30));
    assert_eq!(span(&state, "b"), (30, 60));
    assert_eq!(span(&state, "c"), (100, 110));
}

#[test]
fn moving_two_clips_onto_overlapping_frames_is_rejected() {
    let mut state = state(vec![video_track(
        "v",
        vec![clip("a", 0, 30), clip("b", 30, 30)],
    )]);
    let message = assert_rejected_without_change(
        &mut state,
        EditCommand::MoveClips {
            moves: vec![
                ClipMove {
                    clip_id: "a".into(),
                    to_track: 0,
                    to_frame: 100,
                },
                ClipMove {
                    clip_id: "b".into(),
                    to_track: 0,
                    to_frame: 110,
                },
            ],
        },
    );
    assert!(
        message.contains("moves[0]") && message.contains("moves[1]"),
        "{message}"
    );
}

#[test]
fn duplicates_landing_on_each_other_are_rejected() {
    let mut state = state(vec![
        video_track("top", vec![clip("a", 0, 30)]),
        video_track("bottom", vec![clip("b", 10, 30)]),
    ]);
    let message = assert_rejected_without_change(
        &mut state,
        EditCommand::DuplicateClips {
            clip_ids: vec!["a".into(), "b".into()],
            offset_frames: 200,
            target_track_indexes: vec![0, 0],
        },
    );
    assert!(
        message.contains("clip a") && message.contains("clip b"),
        "{message}"
    );
}

#[test]
fn pasting_entries_that_overlap_on_one_track_is_rejected() {
    let mut state = state(vec![video_track("v", vec![clip("a", 0, 30)])]);
    let entry = |id: &str, start: i32| PasteClipEntry {
        clip: clip(id, 0, 30),
        target_track_id: "v".into(),
        start_frame: start,
    };
    let message = assert_rejected_without_change(
        &mut state,
        EditCommand::PasteClips {
            entries: vec![entry("x", 100), entry("y", 120)],
        },
    );
    assert!(
        message.contains("entries[0]") && message.contains("entries[1]"),
        "{message}"
    );
}

#[test]
fn text_clips_may_still_overlap() {
    let mut state = state(vec![video_track(
        "v",
        vec![text("t1", 0, 30), text("t2", 30, 30)],
    )]);
    apply(
        &mut state,
        EditCommand::MoveClips {
            moves: vec![
                ClipMove {
                    clip_id: "t1".into(),
                    to_track: 0,
                    to_frame: 100,
                },
                ClipMove {
                    clip_id: "t2".into(),
                    to_track: 0,
                    to_frame: 110,
                },
            ],
        },
        &SeqIdGen::default(),
    )
    .unwrap();
    assert_eq!(span(&state, "t1"), (100, 130));
    assert_eq!(span(&state, "t2"), (110, 140));
}

#[test]
fn an_edit_that_would_overlap_a_neighbor_is_rolled_back() {
    // Lengthening `a` over `b` has no overwrite semantics, so the transaction
    // refuses the overlapping result instead of hiding `b` in the render.
    let mut state = state(vec![video_track(
        "v",
        vec![clip("a", 0, 30), clip("b", 30, 30)],
    )]);
    let message = assert_rejected_without_change(
        &mut state,
        EditCommand::SetClipProperties {
            clip_ids: vec!["a".into()],
            properties: Box::new(ClipProperties {
                duration_frames: Some(45),
                ..Default::default()
            }),
        },
    );
    assert!(message.contains("overlapping"), "{message}");
}

#[test]
fn an_existing_overlap_does_not_block_other_edits_on_its_track() {
    // Overlaps saved by an older build stay editable; only new ones are refused.
    let mut state = state(vec![video_track(
        "v",
        vec![clip("a", 0, 30), clip("b", 20, 30), clip("c", 100, 10)],
    )]);
    apply(
        &mut state,
        EditCommand::MoveClips {
            moves: vec![ClipMove {
                clip_id: "c".into(),
                to_track: 0,
                to_frame: 200,
            }],
        },
        &SeqIdGen::default(),
    )
    .unwrap();
    assert_eq!(span(&state, "c"), (200, 210));
}

// ---- property test -------------------------------------------------------------

/// SplitMix64, so the random sequences are reproducible.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn range(&mut self, low: i32, high: i32) -> i32 {
        low + self.below((high - low + 1) as usize) as i32
    }
}

fn random_state(rng: &mut Rng) -> EditorState {
    let mut tracks = Vec::new();
    let mut next = 0;
    for index in 0..1 + rng.below(3) {
        let mut clips = Vec::new();
        let mut cursor = rng.range(0, 20);
        for _ in 0..rng.below(6) {
            cursor += rng.range(0, 12);
            let duration = rng.range(1, 30);
            let mut clip = if rng.below(5) == 0 {
                text(&format!("c{next}"), cursor, duration)
            } else {
                clip(&format!("c{next}"), cursor, duration)
            };
            if rng.below(4) == 0 {
                clip.link_group_id = Some(format!("g{}", rng.below(3)));
            }
            next += 1;
            cursor += duration;
            clips.push(clip);
        }
        tracks.push(video_track(&format!("t{index}"), clips));
    }
    state(tracks)
}

fn clip_ids(state: &EditorState) -> Vec<String> {
    state
        .timeline
        .tracks
        .iter()
        .flat_map(|track| &track.clips)
        .map(|clip| clip.id.clone())
        .collect()
}

/// Up to `count` distinct ids drawn from `ids`.
fn pick_ids(rng: &mut Rng, ids: &[String], count: usize) -> Vec<String> {
    let mut chosen: Vec<String> = Vec::new();
    for _ in 0..count {
        let id = &ids[rng.below(ids.len())];
        if !chosen.contains(id) {
            chosen.push(id.clone());
        }
    }
    chosen
}

fn find<'a>(state: &'a EditorState, id: &str) -> &'a Clip {
    state
        .timeline
        .tracks
        .iter()
        .flat_map(|track| &track.clips)
        .find(|clip| clip.id == id)
        .unwrap()
}

fn random_command(rng: &mut Rng, state: &EditorState) -> Option<EditCommand> {
    let ids = clip_ids(state);
    let tracks = state.timeline.tracks.len();
    if ids.is_empty() || tracks == 0 {
        return None;
    }
    Some(match rng.below(4) {
        0 => EditCommand::SwapClips {
            a: ids[rng.below(ids.len())].clone(),
            b: ids[rng.below(ids.len())].clone(),
        },
        1 => {
            let delta = rng.range(-60, 60);
            let count = 1 + rng.below(3);
            let moves = pick_ids(rng, &ids, count)
                .into_iter()
                .map(|clip_id| {
                    let to_frame = if rng.below(3) == 0 {
                        rng.range(-30, 150)
                    } else {
                        find(state, &clip_id).start_frame + delta
                    };
                    ClipMove {
                        clip_id,
                        to_track: rng.below(tracks),
                        to_frame,
                    }
                })
                .collect();
            EditCommand::MoveClips { moves }
        }
        2 => {
            let count = 1 + rng.below(3);
            let clip_ids = pick_ids(rng, &ids, count);
            let target_track_indexes = clip_ids.iter().map(|_| rng.below(tracks)).collect();
            EditCommand::DuplicateClips {
                clip_ids,
                offset_frames: rng.range(-80, 80),
                target_track_indexes,
            }
        }
        _ => {
            let count = 1 + rng.below(3);
            let entries = pick_ids(rng, &ids, count)
                .into_iter()
                .map(|id| PasteClipEntry {
                    clip: find(state, &id).clone(),
                    target_track_id: state.timeline.tracks[rng.below(tracks)].id.clone(),
                    start_frame: rng.range(0, 150),
                })
                .collect();
            EditCommand::PasteClips { entries }
        }
    })
}

#[test]
fn random_swaps_moves_duplicates_and_pastes_never_overlap_a_track() {
    let mut rng = Rng(26);
    let mut outcomes = [0usize; 2];
    for _ in 0..300 {
        let mut state = random_state(&mut rng);
        let ids = SeqIdGen::new("n-");
        for _ in 0..8 {
            let Some(command) = random_command(&mut rng, &state) else {
                break;
            };
            let described = format!("{command:?}");
            match apply(&mut state, command, &ids) {
                Ok(_) => outcomes[0] += 1,
                Err(EditError::Invalid(message) | EditError::Refused(message)) => {
                    // Each command refuses its own overlapping batch before
                    // the transaction boundary has to roll it back.
                    assert!(
                        !message.contains("overlapping on track"),
                        "{described} reached the transaction backstop: {message}"
                    );
                    outcomes[1] += 1;
                }
            }
            assert_eq!(overlap(&state.timeline), None, "after {described}");
        }
    }
    assert!(outcomes[0] > 500 && outcomes[1] > 50, "{outcomes:?}");
}
