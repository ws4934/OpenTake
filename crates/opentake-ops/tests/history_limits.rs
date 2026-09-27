//! History is bounded and checkpoint costs do not multiply with its depth.
use opentake_domain::{Clip, ClipType, Timeline, Track};
use opentake_ops::{apply, ClipMove, EditCommand, EditorState, SeqIdGen};
use std::time::{Duration, Instant};

fn state(clip_count: usize) -> EditorState {
    let mut timeline = Timeline::new();
    let mut track = Track::new("video", ClipType::Video);
    for index in 0..clip_count {
        track.clips.push(Clip::new(
            format!("clip-{index}"),
            "asset",
            index as i32 * 1000,
            30,
        ));
    }
    timeline.tracks.push(track);
    EditorState::from_timeline(timeline)
}

fn move_to(state: &mut EditorState, frame: i32) {
    apply(
        state,
        EditCommand::MoveClips {
            moves: vec![ClipMove {
                clip_id: "clip-0".into(),
                to_track: 0,
                to_frame: frame,
            }],
        },
        &SeqIdGen::new("history-"),
    )
    .unwrap();
}

#[test]
fn history_retains_the_latest_200_transactions_and_redoes_them_in_order() {
    let mut state = state(1);
    for frame in 1..=210 {
        move_to(&mut state, frame);
    }
    assert_eq!(state.undo_depth(), 200);
    let ids = SeqIdGen::new("history-");
    for frame in (10..210).rev() {
        apply(&mut state, EditCommand::Undo, &ids).unwrap();
        assert_eq!(state.timeline.tracks[0].clips[0].start_frame, frame);
    }
    assert!(!state.can_undo());
    for frame in 11..=210 {
        apply(&mut state, EditCommand::Redo, &ids).unwrap();
        assert_eq!(state.timeline.tracks[0].clips[0].start_frame, frame);
    }
    assert!(!state.can_redo());
    assert_eq!(state.undo_depth(), 200);
}

fn median(samples: &mut [Duration]) -> Duration {
    samples.sort();
    samples[samples.len() / 2]
}

#[test]
#[ignore = "controlled release benchmark: cargo test --release -p opentake-ops --test history_limits -- --ignored --nocapture"]
#[allow(clippy::assertions_on_constants)]
fn release_history_latency_is_document_sized() {
    // Keep the benchmark discoverable in debug without accepting debug timings.
    assert!(!cfg!(debug_assertions), "run this benchmark with --release");
    let ids = SeqIdGen::new("history-");
    for depth in [150, 1] {
        let mut state = state(1000);
        for frame in 1..=depth {
            move_to(&mut state, frame);
        }
        let mut undo = Vec::new();
        let mut redo = Vec::new();
        let mut checkpoint = Vec::new();
        for _ in 0..7 {
            let start = Instant::now();
            apply(&mut state, EditCommand::Undo, &ids).unwrap();
            undo.push(start.elapsed());
            let start = Instant::now();
            apply(&mut state, EditCommand::Redo, &ids).unwrap();
            redo.push(start.elapsed());
            let start = Instant::now();
            let saved = std::hint::black_box(state.clone());
            checkpoint.push(start.elapsed());
            drop(saved);
        }
        let undo = median(&mut undo);
        let redo = median(&mut redo);
        let checkpoint = median(&mut checkpoint);
        println!("1000 clips depth={depth}: median Undo={undo:?}, Redo={redo:?}, checkpoint={checkpoint:?}");
        assert!(
            undo < Duration::from_millis(10),
            "Undo at depth {depth}: {undo:?}"
        );
        assert!(
            redo < Duration::from_millis(10),
            "Redo at depth {depth}: {redo:?}"
        );
        assert!(
            checkpoint < Duration::from_millis(20),
            "checkpoint at depth {depth}: {checkpoint:?}"
        );
    }
}
