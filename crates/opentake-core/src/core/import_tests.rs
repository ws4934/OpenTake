//! Batch imports are out-of-band catalog changes, not thousands of edits.
use super::*;
use std::time::{Duration, Instant};

fn project() -> (tempfile::TempDir, AppCore, PathBuf) {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("Import.opentake");
    let core = AppCore::new();
    core.save_project(Some(path.clone())).unwrap();
    (root, core, path)
}

fn file(index: usize, folder: Option<PreparedMediaFolderRef>) -> PreparedMediaImportOp {
    PreparedMediaImportOp::ImportFile {
        path: PathBuf::from(format!("/fixture/clip-{index}.mp4")),
        name: format!("clip {index}"),
        probe: ProbedMedia {
            duration_secs: 1.0,
            has_audio: true,
            ..Default::default()
        },
        folder,
    }
}

fn folder(key: u64, parent: Option<PreparedMediaFolderRef>) -> PreparedMediaImportOp {
    PreparedMediaImportOp::CreateFolder {
        key,
        name: format!("folder {key}"),
        parent,
    }
}

fn import(
    core: &AppCore,
    path: &Path,
    plan: Vec<PreparedMediaImportOp>,
) -> Result<Vec<CommittedMediaImport>> {
    core.import_media_batch_for_project_persisted(core.runtime_snapshot().project_epoch, path, plan)
}

#[test]
fn folders_and_files_preserve_prior_undo_and_redo_without_adding_history() {
    let (_root, core, path) = project();
    core.apply(EditCommand::CreateFolder {
        name: "earlier edit".into(),
        parent_folder_id: None,
    })
    .unwrap();
    core.apply(EditCommand::CreateFolder {
        name: "redo edit".into(),
        parent_folder_id: None,
    })
    .unwrap();
    core.undo().unwrap();
    let before = core.lock().editor.checkpoint_editor_state();
    let result = import(
        &core,
        &path,
        vec![
            folder(1, None),
            folder(2, Some(PreparedMediaFolderRef::Planned(1))),
            file(0, Some(PreparedMediaFolderRef::Planned(2))),
        ],
    )
    .unwrap();
    let state = core.lock().editor.checkpoint_editor_state();
    assert_eq!(state.version(), before.version());
    assert_eq!(state.undo_depth(), before.undo_depth());
    assert_eq!(
        state.undo_transaction_version(),
        before.undo_transaction_version()
    );
    assert_eq!(state.can_redo(), before.can_redo());
    assert_eq!(result.len(), 1);
    let imported = core.media();
    core.redo().unwrap();
    core.undo().unwrap();
    core.undo().unwrap();
    assert_eq!(core.media().entries, imported.entries);
    assert_eq!(core.media().folders.len(), 2);
    assert!(!core.can_undo());
    let reopened = AppCore::new();
    reopened.open_project(&path).unwrap();
    assert_eq!(reopened.media(), imported);
}

#[test]
fn nonexistent_existing_folder_and_parent_reject_the_whole_batch() {
    for invalid_parent in [false, true] {
        let (_root, core, path) = project();
        let before = core.lock().editor.checkpoint_editor_state();
        let missing = Some(PreparedMediaFolderRef::Existing("missing".into()));
        let invalid = if invalid_parent {
            folder(2, missing)
        } else {
            file(1, missing)
        };
        let result = import(
            &core,
            &path,
            vec![
                folder(1, None),
                file(0, Some(PreparedMediaFolderRef::Planned(1))),
                invalid,
            ],
        );
        assert!(
            result.is_err(),
            "unknown folder must not create a dangling reference"
        );
        let after = core.lock().editor.checkpoint_editor_state();
        assert_eq!(after.manifest, before.manifest);
        assert_eq!(after.timeline, before.timeline);
        assert_eq!(after.version(), before.version());
        assert_eq!(after.undo_depth(), before.undo_depth());
        assert_eq!(after.can_redo(), before.can_redo());
        let reopened = AppCore::new();
        reopened.open_project(&path).unwrap();
        assert_eq!(reopened.media(), before.manifest);
    }
}

#[test]
fn late_invalid_file_preserves_document_version_and_owned_history() {
    let (_root, core, path) = project();
    core.apply(EditCommand::CreateFolder {
        name: "kept".into(),
        parent_folder_id: None,
    })
    .unwrap();
    core.undo().unwrap();
    let before = core.lock().editor.checkpoint_editor_state();
    let mut plan = vec![folder(1, None)];
    plan.extend((0..1000).map(|index| file(index, Some(PreparedMediaFolderRef::Planned(1)))));
    plan.push(PreparedMediaImportOp::ImportFile {
        path: PathBuf::from("/fixture/unsupported.exe"),
        name: "invalid".into(),
        probe: ProbedMedia::default(),
        folder: None,
    });
    assert!(import(&core, &path, plan).is_err());
    let after = core.lock().editor.checkpoint_editor_state();
    assert_eq!(after.manifest, before.manifest);
    assert_eq!(after.timeline, before.timeline);
    assert_eq!(after.version(), before.version());
    assert_eq!(after.undo_depth(), before.undo_depth());
    assert_eq!(after.can_redo(), before.can_redo());
    core.redo().unwrap();
    assert_eq!(core.media_counts(), (0, 1));
}

#[test]
fn repeated_sources_reuse_ids_metadata_and_apply_the_requested_folder() {
    let (_root, core, path) = project();
    let first = import(&core, &path, vec![file(0, None)])
        .unwrap()
        .remove(0)
        .entry;
    let result = import(
        &core,
        &path,
        vec![
            folder(1, None),
            file(0, Some(PreparedMediaFolderRef::Planned(1))),
            file(0, None),
            file(1, None),
            file(1, None),
        ],
    )
    .unwrap();
    assert_eq!(core.media_count(), 2);
    assert_eq!(result[0].entry.id, first.id);
    assert_eq!(result[1].entry.id, first.id);
    assert_eq!(result[0].entry.folder_id, result[1].entry.folder_id);
    assert!(result[0].entry.folder_id.is_some());
    assert_eq!(result[2].entry.id, result[3].entry.id);
    assert_eq!(core.version(), 0);
    assert!(!core.can_undo());
}

#[test]
fn invalid_planned_folder_keys_and_empty_names_are_atomic() {
    for suffix in [
        vec![folder(1, None)],
        vec![file(0, Some(PreparedMediaFolderRef::Planned(9)))],
        vec![PreparedMediaImportOp::CreateFolder {
            key: 2,
            name: String::new(),
            parent: None,
        }],
    ] {
        let (_root, core, path) = project();
        let mut plan = vec![folder(1, None)];
        plan.extend(suffix);
        assert!(import(&core, &path, plan).is_err());
        assert_eq!(core.media_counts(), (0, 0));
        assert_eq!(core.version(), 0);
        assert!(!core.can_undo());
    }
}

#[test]
fn final_identity_failure_only_retracts_the_new_entry_and_preserves_duplicates() {
    let (_root, core, path) = project();
    import(
        &core,
        &path,
        vec![
            folder(1, None),
            file(0, Some(PreparedMediaFolderRef::Planned(1))),
        ],
    )
    .unwrap();
    let before = core.media();
    for index in [0, 1] {
        let result = core.lock().editor.import_media_file_checked(
            PathBuf::from(format!("/fixture/clip-{index}.mp4")),
            "rejected-id",
            "rejected name",
            &ProbedMedia::default(),
            || Err(CoreError::Media("retained file identity changed".into())),
        );
        assert!(result.is_err());
        assert_eq!(core.media(), before);
    }
}

#[test]
fn batch_and_single_stems_share_provenance_and_invalid_stems_roll_back() {
    let (_root, core, path) = project();
    let original = import(&core, &path, vec![file(0, None)])
        .unwrap()
        .remove(0)
        .entry;
    let provenance = crate::session::DerivedStemProvenance {
        source_asset_id: original.id.clone(),
        source_sha256: "a".repeat(64),
        execution: "local:demucs".into(),
        model_sha256: Some("b".repeat(64)),
        stem: "vocals".into(),
    };
    let probe = ProbedMedia {
        duration_secs: 2.0,
        has_audio: true,
        ..Default::default()
    };
    let single = core
        .lock()
        .editor
        .import_derived_stem_file(
            "/fixture/single.wav",
            "single-id",
            "single",
            &probe,
            provenance.clone(),
        )
        .unwrap();
    let result = import(
        &core,
        &path,
        vec![PreparedMediaImportOp::ImportDerivedStem {
            path: PathBuf::from("/fixture/batch.wav"),
            name: "batch".into(),
            probe: probe.clone(),
            provenance: provenance.clone(),
        }],
    )
    .unwrap();
    assert_eq!(result[0].entry.generation_input, single.generation_input);
    assert_eq!(core.media().entries[0], original);
    assert_eq!(core.version(), 0);
    assert!(!core.can_undo());
    let before = core.media();
    for invalid in [
        crate::session::DerivedStemProvenance {
            source_asset_id: "missing".into(),
            ..provenance.clone()
        },
        crate::session::DerivedStemProvenance {
            source_sha256: "not a digest".into(),
            ..provenance.clone()
        },
        crate::session::DerivedStemProvenance {
            execution: "missing-provider".into(),
            ..provenance.clone()
        },
    ] {
        assert!(import(
            &core,
            &path,
            vec![
                folder(1, None),
                file(1, None),
                PreparedMediaImportOp::ImportDerivedStem {
                    path: PathBuf::from("/fixture/invalid.wav"),
                    name: "invalid".into(),
                    probe: probe.clone(),
                    provenance: invalid,
                }
            ]
        )
        .is_err());
        assert_eq!(core.media(), before);
        assert_eq!(core.version(), 0);
        assert!(!core.can_undo());
    }
}

#[test]
#[ignore = "controlled release benchmark: cargo test --release -p opentake-core release_batch_import -- --ignored --nocapture"]
#[allow(clippy::assertions_on_constants)]
fn release_batch_import_is_linear_and_does_not_retain_catalog_snapshots() {
    assert!(!cfg!(debug_assertions), "run this benchmark with --release");
    for (count, with_folder) in [(2000, true), (5000, false)] {
        let (_root, core, path) = project();
        let mut plan = Vec::with_capacity(count + 1);
        if with_folder {
            plan.push(folder(1, None));
        }
        plan.extend((0..count).map(|index| {
            file(
                index,
                with_folder.then_some(PreparedMediaFolderRef::Planned(1)),
            )
        }));
        let started = Instant::now();
        let results = import(&core, &path, plan).unwrap();
        let elapsed = started.elapsed();
        let state = core.lock().editor.checkpoint_editor_state();
        let manifest_bytes = serde_json::to_vec(&state.manifest).unwrap().len();
        println!("batch import count={count} folder={with_folder} elapsed={elapsed:?} manifest_bytes={manifest_bytes} undo_depth={} version={}", state.undo_depth(), state.version());
        assert_eq!(results.len(), count);
        assert_eq!(state.manifest.entries.len(), count);
        assert_eq!(state.undo_depth(), 0);
        assert_eq!(state.version(), 0);
        // No per-file historical catalogs; live catalog plus the transient
        // indexes/results stay proportional to the manifest, not its square.
        assert!(manifest_bytes < 20 * 1024 * 1024);
        assert!(
            elapsed < Duration::from_millis(200),
            "batch import: {elapsed:?}"
        );
    }
}
