//! Dissolving a compound clip is a lossless regroup: every frame must draw the
//! same layers, in the same order, from the same source frames, with the same
//! geometry and opacity as the compound drew them.

use opentake_domain::{
    AnimPair, Clip, ClipType, Interpolation, Keyframe, KeyframeTrack, Timeline, Track,
};
use opentake_ops::{apply, EditCommand, EditorState, SeqIdGen};
use opentake_render::{try_build_render_plan, RenderSize, SourceMetrics, TextureSource};

struct Metrics;

impl SourceMetrics for Metrics {
    fn natural_size(&self, _media_ref: &str) -> Option<(u32, u32)> {
        Some((64, 36))
    }
}

fn video_track(id: &str, clips: Vec<Clip>) -> Track {
    let mut track = Track::new(id, ClipType::Video);
    track.clips = clips;
    track
}

fn linear<V>(keyframes: Vec<(i32, V)>) -> KeyframeTrack<V> {
    KeyframeTrack::from_keyframes(
        keyframes
            .into_iter()
            .map(|(frame, value)| Keyframe::with_interpolation(frame, value, Interpolation::Linear))
            .collect(),
    )
}

type Draw = (TextureSource, i64, [f64; 6], (f64, f64, f64, f64), f64);

fn draws(timeline: &Timeline, frames: std::ops::Range<i32>) -> Vec<Vec<Draw>> {
    let plan = try_build_render_plan(timeline, RenderSize::new(64, 36), &Metrics).unwrap();
    frames
        .map(|frame| {
            plan.frame(timeline, frame)
                .draws
                .iter()
                .map(|draw| {
                    (
                        draw.source.clone(),
                        draw.source_frame,
                        draw.affine,
                        draw.crop_uv,
                        draw.opacity,
                    )
                })
                .collect()
        })
        .collect()
}

fn assert_same_picture(before: &[Vec<Draw>], after: &[Vec<Draw>]) {
    assert_eq!(before.len(), after.len());
    for (index, (before, after)) in before.iter().zip(after).enumerate() {
        assert_eq!(
            before.len(),
            after.len(),
            "frame offset {index}: layer count"
        );
        for (before, after) in before.iter().zip(after) {
            assert_eq!(before.0, after.0, "frame offset {index}: layer order");
            assert_eq!(before.1, after.1, "frame offset {index}: source frame");
            for (b, a) in before.2.iter().zip(after.2) {
                assert!((b - a).abs() < 1e-9, "frame offset {index}: affine");
            }
            assert!((before.3 .0 - after.3 .0).abs() < 1e-9);
            assert!((before.3 .2 - after.3 .2).abs() < 1e-9);
            assert!(
                (before.4 - after.4).abs() < 1e-9,
                "frame offset {index}: opacity {} != {}",
                before.4,
                after.4
            );
        }
    }
}

#[test]
fn dissolving_a_trimmed_compound_keeps_every_frame() {
    // Two animated child lanes; the compound window cuts 20 frames from the
    // head and 15 from the tail, including both fades of the title.
    let mut title = Clip::new("title", "asset-title", 0, 100);
    title.opacity_track = Some(linear(vec![(0, 0.0), (100, 1.0)]));
    title.position_track = Some(linear(vec![
        (0, AnimPair::new(0.0, 0.0)),
        (100, AnimPair::new(0.5, 0.25)),
    ]));
    title.fade_in_frames = 10;
    title.fade_out_frames = 10;
    let mut broll = Clip::new("broll", "asset-broll", 0, 100);
    broll.trim_start_frame = 7;
    broll.scale_track = Some(linear(vec![
        (0, AnimPair::new(1.0, 1.0)),
        (100, AnimPair::new(0.5, 0.5)),
    ]));
    let mut child = Timeline::new();
    child.tracks = vec![
        video_track("child-title", vec![title]),
        video_track("child-broll", vec![broll]),
    ];

    let mut root = Timeline::new();
    root.tracks = vec![
        video_track("top", vec![]),
        video_track("background", vec![Clip::new("bg", "asset-bg", 0, 120)]),
    ];
    let mut state = EditorState::from_timeline(root);
    let ids = SeqIdGen::new("dissolve-render-");
    let compound = apply(
        &mut state,
        EditCommand::CreateNestedSequence {
            name: "Scene".into(),
            timeline: child,
            track_index: 0,
            start_frame: 10,
            duration_frames: 100,
        },
        &ids,
    )
    .unwrap()
    .affected_clip_ids[0]
        .clone();
    apply(
        &mut state,
        EditCommand::TrimClips {
            edits: vec![(compound.clone(), 20, 15)],
        },
        &ids,
    )
    .unwrap();
    let before = draws(&state.timeline, 20..110);
    assert_eq!(
        before[20].len(),
        3,
        "background, b-roll and title at frame 40"
    );

    apply(
        &mut state,
        EditCommand::DissolveNestedSequence { clip_id: compound },
        &ids,
    )
    .unwrap();
    assert!(state
        .timeline
        .tracks
        .iter()
        .flat_map(|track| &track.clips)
        .all(|clip| clip.nested_sequence_id.is_none()));

    assert_same_picture(&before, &draws(&state.timeline, 20..110));
}
