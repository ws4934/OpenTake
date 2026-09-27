use opentake_core::{AppCore, EditCommand, ProbedMedia};
use opentake_project::Project;

// Exercise the desktop's production coordination without linking Tauri.
#[path = "../../../src-tauri/src/close_coordinator.rs"]
mod close_coordinator;
#[path = "../../../src-tauri/src/instance_lock.rs"]
mod instance_lock;

#[test]
fn exit_save_persists_latest_edits_and_manifest_only_imports_without_touching_cover() {
    let storage = tempfile::tempdir().unwrap();
    let bundle = storage.path().join("work.opentake");
    Project::new(&bundle).save().unwrap();
    let cover = bundle.join("thumbnail.jpg");
    std::fs::write(&cover, b"existing cover").unwrap();
    let core = AppCore::new();
    core.open_project(&bundle).unwrap();
    core.apply(EditCommand::SetTimelineSettings {
        fps: 30,
        width: 1280,
        height: 720,
    })
    .unwrap();
    let version = core.project_revision().version;
    let media = storage.path().join("source.mp4");
    std::fs::write(&media, b"fixture").unwrap();
    core.import_media_file(&media, "Imported", &ProbedMedia::default())
        .unwrap();
    assert_eq!(core.project_revision().version, version);

    assert_eq!(
        core.save_project_before_exit_if(|| true).unwrap(),
        Some(bundle.clone())
    );

    let reopened = Project::open(&bundle).unwrap();
    assert_eq!(
        (reopened.timeline.width, reopened.timeline.height),
        (1280, 720)
    );
    assert_eq!(reopened.manifest.entries.len(), 1);
    assert_eq!(std::fs::read(cover).unwrap(), b"existing cover");
}

#[test]
fn refused_exit_save_keeps_the_previous_disk_document_and_current_edits() {
    let storage = tempfile::tempdir().unwrap();
    let bundle = storage.path().join("work.opentake");
    Project::new(&bundle).save().unwrap();
    let core = AppCore::new();
    core.open_project(&bundle).unwrap();
    let before = std::fs::read(bundle.join("project.json")).unwrap();
    core.apply(EditCommand::SetTimelineSettings {
        fps: 30,
        width: 1280,
        height: 720,
    })
    .unwrap();

    assert!(core.save_project_before_exit_if(|| false).is_err());

    assert_eq!(std::fs::read(bundle.join("project.json")).unwrap(), before);
    assert_eq!(core.get_timeline().timeline.width, 1280);
    core.save_project_before_exit_if(|| true).unwrap();
    assert_eq!(Project::open(bundle).unwrap().timeline.width, 1280);
}

#[test]
fn exit_save_skips_missing_and_read_only_projects() {
    let core = AppCore::new();
    assert_eq!(
        core.save_project_before_exit_if(|| panic!("no write expected"))
            .unwrap(),
        None
    );
    let storage = tempfile::tempdir().unwrap();
    let bundle = storage.path().join("future.opentake");
    Project::new(&bundle).save().unwrap();
    let path = bundle.join("project.json");
    let mut document: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    document["futureFeature"] = serde_json::json!(true);
    let before = serde_json::to_vec(&document).unwrap();
    std::fs::write(&path, &before).unwrap();
    core.open_project(&bundle).unwrap();
    assert!(core.get_timeline().compatibility.is_read_only());

    assert_eq!(
        core.save_project_before_exit_if(|| panic!("no write expected"))
            .unwrap(),
        None
    );

    assert_eq!(std::fs::read(path).unwrap(), before);
}
