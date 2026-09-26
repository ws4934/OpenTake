use std::fs;

use opentake_core::{
    AppCore, GenerationStateUpdate, PreparedGenerationJob, PreparedGenerationOutput, ProbedMedia,
};
use opentake_domain::{
    ClipType, GenerationInput, GenerationJobStatus, MediaManifestEntry, MediaSource,
};
use opentake_ops::{ClipEntry, EditCommand};
use opentake_project::Project;

fn saved_project() -> (tempfile::TempDir, std::path::PathBuf) {
    let temp = tempfile::tempdir().unwrap();
    let bundle = temp.path().join("Generation.opentake");
    let mut project = Project::new(&bundle);
    project.manifest.entries.push(MediaManifestEntry {
        id: "source-image".to_string(),
        name: "source.png".to_string(),
        kind: ClipType::Image,
        source: MediaSource::Project {
            relative_path: "media/source.png".to_string(),
        },
        duration: 0.0,
        generation_input: None,
        source_width: Some(4),
        source_height: Some(3),
        source_fps: None,
        has_audio: Some(false),
        color: None,
        proxy: None,
        folder_id: None,
        cached_remote_url: None,
        cached_remote_url_expires_at: None,
    });
    project.save().unwrap();
    fs::create_dir_all(bundle.join("media")).unwrap();
    fs::write(bundle.join("media/source.png"), b"source-bytes").unwrap();
    fs::write(bundle.join("thumbnail.jpg"), b"cover").unwrap();
    (temp, bundle)
}

fn upscale_plan() -> PreparedGenerationJob {
    PreparedGenerationJob {
        name: "Upscaled source".to_string(),
        kind: ClipType::Image,
        folder_id: None,
        provider: "fal".to_string(),
        input: GenerationInput {
            prompt: String::new(),
            model: "fal:fixture-upscaler".to_string(),
            duration: 0,
            aspect_ratio: String::new(),
            ..Default::default()
        },
        output_count: 1,
        source_asset_id: Some("source-image".to_string()),
        source_clip_id: Some("source-clip".to_string()),
        estimated_cost_credits: Some(12),
        created_at: Some(800_000_000.0),
    }
}

fn update(status: GenerationJobStatus, progress: Option<f64>) -> GenerationStateUpdate {
    GenerationStateUpdate {
        status,
        progress,
        error_code: None,
        provider_job_id: None,
        cost_credits: None,
        created_at: Some(800_000_001.0),
    }
}

fn add_history_clip(core: &AppCore) {
    core.apply(EditCommand::AddClipsAutoTrack {
        entries: vec![ClipEntry {
            media_ref: "source-image".into(),
            media_type: ClipType::Image,
            source_clip_type: ClipType::Image,
            track_index: 0,
            start_frame: 0,
            duration_frames: 30,
            trim_start_frame: None,
            trim_end_frame: None,
            has_audio: false,
            add_linked_audio: false,
            transform: None,
        }],
    })
    .unwrap();
}

fn history_image_probe() -> ProbedMedia {
    ProbedMedia {
        duration_secs: 0.0,
        width: Some(8),
        height: Some(6),
        fps: None,
        has_audio: false,
        color: None,
    }
}

fn finish_history_generation(
    core: &AppCore,
    epoch: u64,
    bundle: &std::path::Path,
    job_id: &str,
    asset_id: &str,
) -> String {
    for (status, progress) in [
        (GenerationJobStatus::Generating, 0.2),
        (GenerationJobStatus::Downloading, 0.8),
        (GenerationJobStatus::Finalizing, 0.9),
    ] {
        core.update_generation_job_for_project(
            epoch,
            bundle,
            job_id,
            update(status, Some(progress)),
        )
        .unwrap();
    }
    let leaf = format!("{asset_id}.png");
    let relative_path = format!("media/{leaf}");
    let bytes = b"paid-result-fixture";
    core.finalize_generation_output_with_media_for_project(
        epoch,
        bundle,
        PreparedGenerationOutput {
            asset_id: asset_id.into(),
            relative_path: relative_path.clone(),
            probe: history_image_probe(),
            created_at: Some(800_000_002.0),
        },
        &leaf,
        bytes.len() as u64,
        &mut std::io::Cursor::new(bytes),
    )
    .unwrap();
    relative_path
}

#[test]
fn undo_after_import_and_generation_keeps_placeholders_finalizable_and_durable() {
    let (temp, bundle) = saved_project();
    let core = AppCore::new();
    core.open_project(&bundle).unwrap();
    add_history_clip(&core);
    let imported_path = temp.path().join("imported.png");
    fs::write(&imported_path, b"imported-image").unwrap();
    let imported = core
        .import_media_file(&imported_path, "Imported", &history_image_probe())
        .unwrap();
    let epoch = core.runtime_snapshot().project_epoch;
    let job = core
        .begin_generation_job_for_project(epoch, &bundle, upscale_plan())
        .unwrap();
    let asset_id = &job.placeholder_asset_ids[0];

    let undone = core.undo().unwrap();
    assert!(undone.timeline_changed);
    assert!(!undone.manifest_changed);
    let manifest = core.media();
    assert!(manifest.entries.iter().any(|entry| entry.id == imported.id));
    let placeholder = manifest
        .entries
        .iter()
        .find(|entry| &entry.id == asset_id)
        .unwrap();
    assert_eq!(
        placeholder.generation_input.as_ref().unwrap().status,
        Some(GenerationJobStatus::Queued)
    );

    let relative_path = finish_history_generation(&core, epoch, &bundle, &job.job_id, asset_id);
    core.save_project(None).unwrap();
    let reopened = AppCore::new();
    reopened.open_project(&bundle).unwrap();
    let manifest = reopened.media();
    assert!(manifest.entries.iter().any(|entry| entry.id == imported.id));
    let output = manifest
        .entries
        .iter()
        .find(|entry| &entry.id == asset_id)
        .unwrap();
    assert_eq!(
        output.generation_input.as_ref().unwrap().status,
        Some(GenerationJobStatus::Ready)
    );
    assert_eq!(
        output.source,
        MediaSource::Project {
            relative_path: relative_path.clone()
        }
    );
    assert_eq!(
        fs::read(bundle.join(relative_path)).unwrap(),
        b"paid-result-fixture"
    );
}

#[test]
fn undo_of_edit_during_generation_never_reverts_a_ready_result_to_pending() {
    let (_temp, bundle) = saved_project();
    let core = AppCore::new();
    core.open_project(&bundle).unwrap();
    let epoch = core.runtime_snapshot().project_epoch;
    let job = core
        .begin_generation_job_for_project(epoch, &bundle, upscale_plan())
        .unwrap();
    let asset_id = &job.placeholder_asset_ids[0];
    add_history_clip(&core);
    let relative_path = finish_history_generation(&core, epoch, &bundle, &job.job_id, asset_id);
    let ready = core.media();

    for _ in 0..3 {
        core.undo().unwrap();
        assert_eq!(core.media(), ready);
        core.redo().unwrap();
        assert_eq!(core.media(), ready);
    }
    core.undo().unwrap();
    core.save_project(None).unwrap();
    let reopened = AppCore::new();
    reopened.open_project(&bundle).unwrap();
    let manifest = reopened.media();
    let output = manifest
        .entries
        .iter()
        .find(|entry| &entry.id == asset_id)
        .unwrap();
    assert_eq!(
        output.generation_input.as_ref().unwrap().status,
        Some(GenerationJobStatus::Ready)
    );
    assert_eq!(output.source, MediaSource::Project { relative_path });
}

#[test]
fn placeholders_job_events_and_finalized_output_survive_restart() {
    let (_temp, bundle) = saved_project();
    let core = AppCore::new();
    core.open_project(&bundle).unwrap();
    let runtime = core.runtime_snapshot();

    let committed = core
        .begin_generation_job_for_project(runtime.project_epoch, &bundle, upscale_plan())
        .unwrap();
    assert_eq!(committed.placeholder_asset_ids.len(), 1);
    let asset_id = committed.placeholder_asset_ids[0].clone();

    let queued = Project::open(&bundle).unwrap();
    let placeholder = queued
        .manifest
        .entries
        .iter()
        .find(|entry| entry.id == asset_id)
        .unwrap();
    let input = placeholder.generation_input.as_ref().unwrap();
    assert_eq!(input.status, Some(GenerationJobStatus::Queued));
    assert_eq!(input.source_asset_id.as_deref(), Some("source-image"));
    assert_eq!(input.source_clip_id.as_deref(), Some("source-clip"));
    assert_eq!(input.estimated_cost_credits, Some(12));
    assert_eq!(fs::read(bundle.join("thumbnail.jpg")).unwrap(), b"cover");
    assert_eq!(
        queued.generation_log.as_ref().unwrap().entries[0].status,
        Some(GenerationJobStatus::Queued)
    );

    let mut running = update(GenerationJobStatus::Generating, Some(0.2));
    running.provider_job_id = Some("fal::fixture-job".to_string());
    core.update_generation_job_for_project(
        runtime.project_epoch,
        &bundle,
        &committed.job_id,
        running,
    )
    .unwrap();
    let mut downloading = update(GenerationJobStatus::Downloading, Some(0.8));
    downloading.cost_credits = Some(11);
    core.update_generation_job_for_project(
        runtime.project_epoch,
        &bundle,
        &committed.job_id,
        downloading,
    )
    .unwrap();
    core.update_generation_job_for_project(
        runtime.project_epoch,
        &bundle,
        &committed.job_id,
        update(GenerationJobStatus::Finalizing, Some(0.9)),
    )
    .unwrap();

    let media_leaf = format!("{asset_id}.png");
    let relative_path = format!("media/{media_leaf}");
    let mut generated_media = std::io::Cursor::new(b"upscaled-bytes");
    core.finalize_generation_output_with_media_for_project(
        runtime.project_epoch,
        &bundle,
        PreparedGenerationOutput {
            asset_id: asset_id.clone(),
            relative_path: relative_path.clone(),
            probe: ProbedMedia {
                duration_secs: 0.0,
                width: Some(8),
                height: Some(6),
                fps: None,
                has_audio: false,
                color: None,
            },
            created_at: Some(800_000_002.0),
        },
        &media_leaf,
        14,
        &mut generated_media,
    )
    .unwrap();

    let reopened = AppCore::new();
    reopened.open_project(&bundle).unwrap();
    let media = reopened.media();
    let source = media
        .entries
        .iter()
        .find(|entry| entry.id == "source-image")
        .unwrap();
    assert_eq!(source.source_width, Some(4));
    assert_eq!(source.source_height, Some(3));
    assert_eq!(
        fs::read(bundle.join("media/source.png")).unwrap(),
        b"source-bytes"
    );

    let output = media
        .entries
        .iter()
        .find(|entry| entry.id == asset_id)
        .unwrap();
    assert_eq!(output.source_width, Some(8));
    assert_eq!(output.source_height, Some(6));
    assert_eq!(
        output.source,
        MediaSource::Project {
            relative_path: relative_path.clone()
        }
    );
    assert_eq!(
        output.generation_input.as_ref().unwrap().status,
        Some(GenerationJobStatus::Ready)
    );
    assert_eq!(
        fs::read(bundle.join(&relative_path)).unwrap(),
        b"upscaled-bytes"
    );
    assert_eq!(fs::read(bundle.join("thumbnail.jpg")).unwrap(), b"cover");

    let log = reopened.generation_log();
    assert_eq!(log.entries.len(), 5);
    assert_eq!(log.total_credits(), 11);
    assert_eq!(
        log.entries.last().and_then(|entry| entry.status),
        Some(GenerationJobStatus::Ready)
    );
    assert!(log.entries.iter().all(|entry| {
        let json = serde_json::to_string(entry).unwrap();
        !json.contains("source-bytes") && !json.contains("https://")
    }));
}

#[test]
fn invalid_progress_and_error_codes_do_not_mutate_the_durable_job() {
    let (_temp, bundle) = saved_project();
    let core = AppCore::new();
    core.open_project(&bundle).unwrap();
    let runtime = core.runtime_snapshot();
    let committed = core
        .begin_generation_job_for_project(runtime.project_epoch, &bundle, upscale_plan())
        .unwrap();
    let before = fs::read(bundle.join("media.json")).unwrap();

    let invalid = GenerationStateUpdate {
        status: GenerationJobStatus::Generating,
        progress: Some(f64::NAN),
        error_code: None,
        provider_job_id: None,
        cost_credits: None,
        created_at: None,
    };
    assert!(core
        .update_generation_job_for_project(
            runtime.project_epoch,
            &bundle,
            &committed.job_id,
            invalid,
        )
        .is_err());
    assert_eq!(fs::read(bundle.join("media.json")).unwrap(), before);

    assert!(core
        .fail_generation_output_for_project(
            runtime.project_epoch,
            &bundle,
            &committed.placeholder_asset_ids[0],
            "provider leaked /private/path",
            None,
        )
        .is_err());
    assert_eq!(fs::read(bundle.join("media.json")).unwrap(), before);
}

#[test]
fn cancelling_a_partially_finalized_job_preserves_ready_outputs() {
    let (_temp, bundle) = saved_project();
    let core = AppCore::new();
    core.open_project(&bundle).unwrap();
    let runtime = core.runtime_snapshot();
    let mut plan = upscale_plan();
    plan.output_count = 2;
    let committed = core
        .begin_generation_job_for_project(runtime.project_epoch, &bundle, plan)
        .unwrap();
    let ready_id = committed.placeholder_asset_ids[0].clone();
    let cancelled_id = committed.placeholder_asset_ids[1].clone();

    for (status, progress) in [
        (GenerationJobStatus::Generating, Some(0.2)),
        (GenerationJobStatus::Downloading, Some(0.8)),
        (GenerationJobStatus::Finalizing, Some(0.9)),
    ] {
        core.update_generation_job_for_project(
            runtime.project_epoch,
            &bundle,
            &committed.job_id,
            update(status, progress),
        )
        .unwrap();
    }

    let relative_path = format!("media/{ready_id}.png");
    fs::write(bundle.join(&relative_path), b"ready-output").unwrap();
    core.finalize_generation_output_for_project(
        runtime.project_epoch,
        &bundle,
        PreparedGenerationOutput {
            asset_id: ready_id.clone(),
            relative_path: relative_path.clone(),
            probe: ProbedMedia {
                duration_secs: 0.0,
                width: Some(8),
                height: Some(6),
                fps: None,
                has_audio: false,
                color: None,
            },
            created_at: Some(800_000_002.0),
        },
    )
    .unwrap();
    core.cancel_generation_output_for_project(
        runtime.project_epoch,
        &bundle,
        &ready_id,
        Some(800_000_003.0),
    )
    .unwrap();
    core.cancel_generation_output_for_project(
        runtime.project_epoch,
        &bundle,
        &cancelled_id,
        Some(800_000_003.0),
    )
    .unwrap();

    let reopened = Project::open(&bundle).unwrap();
    let ready = reopened
        .manifest
        .entries
        .iter()
        .find(|entry| entry.id == ready_id)
        .unwrap();
    assert_eq!(
        ready.generation_input.as_ref().unwrap().status,
        Some(GenerationJobStatus::Ready)
    );
    assert_eq!(
        ready.source,
        MediaSource::Project {
            relative_path: relative_path.clone(),
        }
    );
    assert_eq!(
        fs::read(bundle.join(relative_path)).unwrap(),
        b"ready-output"
    );

    let cancelled = reopened
        .manifest
        .entries
        .iter()
        .find(|entry| entry.id == cancelled_id)
        .unwrap();
    assert_eq!(
        cancelled.generation_input.as_ref().unwrap().status,
        Some(GenerationJobStatus::Cancelled)
    );
    assert!(matches!(
        cancelled.source,
        MediaSource::Project { ref relative_path } if relative_path.ends_with(".pending")
    ));
    assert!(!bundle.join(format!("media/{cancelled_id}.png")).exists());
}
