//! History changes only the manifest fields owned by the editing transaction.
use opentake_domain::{
    ClipType, GenerationInput, GenerationJobStatus, MediaManifest, MediaManifestEntry, MediaProxy,
    MediaSource, Timeline, Track,
};
use opentake_ops::command::RenameEntry;
use opentake_ops::{apply, ClipEntry, EditCommand, EditorState, SeqIdGen};

fn media(id: &str) -> MediaManifestEntry {
    MediaManifestEntry {
        id: id.into(),
        name: format!("{id}.mp4"),
        kind: ClipType::Video,
        source: MediaSource::Project {
            relative_path: format!("media/{id}.mp4"),
        },
        duration: 1.0,
        generation_input: None,
        source_width: Some(1920),
        source_height: Some(1080),
        source_fps: Some(30.0),
        has_audio: Some(false),
        color: None,
        proxy: None,
        folder_id: None,
        cached_remote_url: None,
        cached_remote_url_expires_at: None,
    }
}

fn state() -> EditorState {
    let mut timeline = Timeline::new();
    timeline.tracks.push(Track::new("video", ClipType::Video));
    let mut manifest = MediaManifest::new();
    manifest.entries.push(media("source"));
    EditorState::new(timeline, manifest)
}

fn clip_entry(id: &str) -> ClipEntry {
    ClipEntry {
        media_ref: id.into(),
        media_type: ClipType::Video,
        source_clip_type: ClipType::Video,
        track_index: 0,
        start_frame: 0,
        duration_frames: 30,
        trim_start_frame: None,
        trim_end_frame: None,
        has_audio: false,
        add_linked_audio: false,
        transform: None,
    }
}

fn edit(state: &mut EditorState, command: EditCommand) {
    apply(state, command, &SeqIdGen::new("history-")).unwrap();
}

fn add_clip(state: &mut EditorState) {
    edit(
        state,
        EditCommand::AddClips {
            entries: vec![clip_entry("source")],
        },
    );
}

fn ids(state: &EditorState) -> Vec<&str> {
    state
        .manifest
        .entries
        .iter()
        .map(|entry| entry.id.as_str())
        .collect()
}

#[test]
fn moving_a_clip_then_importing_never_removes_the_import_on_undo() {
    let mut state = state();
    add_clip(&mut state);
    let clip_id = state.timeline.tracks[0].clips[0].id.clone();
    edit(
        &mut state,
        EditCommand::MoveClips {
            moves: vec![opentake_ops::ClipMove {
                clip_id,
                to_track: 0,
                to_frame: 60,
            }],
        },
    );
    state.manifest.entries.push(media("imported"));
    for _ in 0..3 {
        edit(&mut state, EditCommand::Undo);
        assert_eq!(state.timeline.tracks[0].clips[0].start_frame, 0);
        assert_eq!(ids(&state), ["source", "imported"]);
        edit(&mut state, EditCommand::Redo);
        assert_eq!(state.timeline.tracks[0].clips[0].start_frame, 60);
        assert_eq!(ids(&state), ["source", "imported"]);
    }
}

#[test]
fn timeline_undo_and_redo_preserve_imports_made_on_either_side() {
    let mut state = state();
    add_clip(&mut state);
    state.manifest.entries.push(media("imported-before-undo"));
    edit(&mut state, EditCommand::Undo);
    assert!(state.timeline.tracks[0].clips.is_empty());
    assert_eq!(ids(&state), ["source", "imported-before-undo"]);
    state.manifest.entries.push(media("imported-before-redo"));
    let manifest = state.manifest.clone();
    for _ in 0..3 {
        let result = apply(&mut state, EditCommand::Redo, &SeqIdGen::new("redo-")).unwrap();
        assert!(result.timeline_changed);
        assert!(!result.manifest_changed);
        assert_eq!(state.timeline.tracks[0].clips.len(), 1);
        assert_eq!(state.manifest, manifest);
        edit(&mut state, EditCommand::Undo);
        assert_eq!(state.manifest, manifest);
    }
    // A fresh edit may clear redo, but must not destroy the imported entries.
    add_clip(&mut state);
    assert!(!state.can_redo());
    assert_eq!(state.manifest, manifest);
}

#[test]
fn unrelated_history_keeps_latest_relink_proxy_and_generation_results() {
    let mut state = state();
    add_clip(&mut state);
    let entry = &mut state.manifest.entries[0];
    entry.source = MediaSource::Project {
        relative_path: "media/final.mp4".into(),
    };
    entry.duration = 3.0;
    entry.source_width = Some(1280);
    entry.source_height = Some(720);
    entry.source_fps = Some(24.0);
    entry.has_audio = Some(true);
    entry.proxy = Some(MediaProxy {
        relative_path: "media/proxy.mp4".into(),
        source_sha256: "a".repeat(64),
        source_stamp: None,
        width: 640,
        height: 360,
    });
    entry.generation_input = Some(GenerationInput {
        status: Some(GenerationJobStatus::Ready),
        ..Default::default()
    });
    entry.cached_remote_url = Some("https://example.test/result.mp4".into());
    entry.cached_remote_url_expires_at = Some(100.0);
    let manifest = state.manifest.clone();
    edit(&mut state, EditCommand::Undo);
    assert_eq!(state.manifest, manifest);
    edit(&mut state, EditCommand::Redo);
    assert_eq!(state.manifest, manifest);
}

#[test]
fn registration_stays_undoable_without_removing_later_imports() {
    let mut state = state();
    edit(
        &mut state,
        EditCommand::RegisterMediaAndAddClip {
            media: media("registered"),
            entry: clip_entry("registered"),
            auto_track: false,
        },
    );
    state.manifest.entries.push(media("imported"));
    // Redo must recover the actual entry removed by Undo, not its old registration value.
    state.manifest.entries[1].proxy = Some(MediaProxy {
        relative_path: "media/proxy.mp4".into(),
        source_sha256: "b".repeat(64),
        source_stamp: None,
        width: 640,
        height: 360,
    });
    let expected = state.manifest.clone();
    for _ in 0..3 {
        edit(&mut state, EditCommand::Undo);
        assert_eq!(ids(&state), ["source", "imported"]);
        assert!(state.timeline.tracks[0].clips.is_empty());
        edit(&mut state, EditCommand::Redo);
        assert_eq!(state.manifest, expected);
        assert_eq!(state.timeline.tracks[0].clips.len(), 1);
    }
}

#[test]
fn explicit_deletion_restores_original_order_and_keeps_later_imports() {
    let mut state = state();
    state
        .manifest
        .entries
        .extend([media("second"), media("third")]);
    edit(
        &mut state,
        EditCommand::DeleteMedia {
            asset_ids: vec!["source".into(), "third".into()],
        },
    );
    state.manifest.entries.push(media("later"));
    for _ in 0..3 {
        edit(&mut state, EditCommand::Undo);
        assert_eq!(ids(&state), ["source", "second", "third", "later"]);
        edit(&mut state, EditCommand::Redo);
        assert_eq!(ids(&state), ["second", "later"]);
    }
}

#[test]
fn folder_and_rename_history_only_reverts_edited_fields() {
    let mut state = state();
    edit(
        &mut state,
        EditCommand::CreateFolder {
            name: "Folder".into(),
            parent_folder_id: None,
        },
    );
    let folder_id = state.manifest.folders[0].id.clone();
    edit(
        &mut state,
        EditCommand::MoveToFolder {
            asset_ids: vec!["source".into()],
            folder_id: Some(folder_id.clone()),
        },
    );
    edit(
        &mut state,
        EditCommand::RenameMedia {
            entries: vec![RenameEntry {
                id: "source".into(),
                name: "Edited".into(),
            }],
        },
    );
    state.manifest.entries[0].source = MediaSource::Project {
        relative_path: "media/relinked.mp4".into(),
    };
    state.manifest.set_favorites(&["source".into()], true);
    state
        .manifest
        .favorite_library_ids
        .insert("source".into(), "library-id".into());
    state.manifest.entries.push(media("imported"));
    let relinked = state.manifest.entries[0].source.clone();
    for _ in 0..3 {
        edit(&mut state, EditCommand::Undo);
        assert_eq!(state.manifest.entries[0].source, relinked);
        assert!(state.manifest.is_favorite("source"));
        assert_eq!(state.manifest.favorite_library_ids["source"], "library-id");
        assert_eq!(ids(&state), ["source", "imported"]);
    }
    assert!(state.manifest.folders.is_empty());
    assert_eq!(state.manifest.entries[0].folder_id, None);
    assert_eq!(state.manifest.entries[0].name, "source.mp4");
    for _ in 0..3 {
        edit(&mut state, EditCommand::Redo);
    }
    assert_eq!(state.manifest.entries[0].folder_id, Some(folder_id));
    assert_eq!(state.manifest.entries[0].name, "Edited");
    assert_eq!(state.manifest.entries[0].source, relinked);
    assert!(state.manifest.is_favorite("source"));
}

#[test]
fn undo_folder_creation_rehomes_later_imports_instead_of_hiding_them() {
    let mut state = state();
    edit(
        &mut state,
        EditCommand::CreateFolder {
            name: "Generated".into(),
            parent_folder_id: None,
        },
    );
    let folder_id = state.manifest.folders[0].id.clone();
    let mut imported = media("later-import");
    imported.folder_id = Some(folder_id.clone());
    state.manifest.entries.push(imported);
    for _ in 0..3 {
        edit(&mut state, EditCommand::Undo);
        assert!(state.manifest.folders.is_empty());
        assert_eq!(ids(&state), ["source", "later-import"]);
        assert_eq!(state.manifest.entries[1].folder_id, None);
        edit(&mut state, EditCommand::Redo);
        assert_eq!(state.manifest.folders[0].id, folder_id);
        assert_eq!(state.manifest.entries[1].folder_id, Some(folder_id.clone()));
    }
}

#[test]
fn later_out_of_band_write_to_the_same_field_wins_over_history() {
    let mut state = state();
    edit(
        &mut state,
        EditCommand::RenameMedia {
            entries: vec![RenameEntry {
                id: "source".into(),
                name: "Edited".into(),
            }],
        },
    );
    state.manifest.entries[0].name = "Newer external name".into();
    let manifest = state.manifest.clone();
    edit(&mut state, EditCommand::Undo);
    assert_eq!(state.manifest, manifest);
    assert!(!state.can_redo());
}
