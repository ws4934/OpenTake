use std::fs;

use opentake_core::{
    AppCore, CoreEvent, GenerationStateUpdate, PreparedGenerationJob, PreparedGenerationOutput,
    ProbedMedia,
};
use opentake_domain::{
    ClipType, GenerationInput, GenerationJobStatus, MediaManifestEntry, MediaSource,
};
use opentake_ops::{ClipEntry, EditCommand};
use opentake_project::{Project, ProjectRoot};

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

fn running_poll() -> GenerationStateUpdate {
    let mut running = update(GenerationJobStatus::Generating, Some(0.5));
    running.provider_job_id = Some("fal::fixture-job".to_string());
    running
}

/// Sibling names the bundle publisher uses while replacing the bundle (stage,
/// backup, journal, lock). Component writes must not add any of them.
fn publication_artifacts(bundle: &std::path::Path) -> Vec<String> {
    let prefix = format!(".{}", bundle.file_name().unwrap().to_string_lossy());
    fs::read_dir(bundle.parent().unwrap())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with(&prefix))
        .collect()
}

#[test]
fn repeated_running_polls_neither_log_nor_write_nor_rotate_the_bundle_root() {
    let (_temp, bundle) = saved_project();
    let core = AppCore::new();
    core.open_project(&bundle).unwrap();
    let epoch = core.runtime_snapshot().project_epoch;
    let job = core
        .begin_generation_job_for_project(epoch, &bundle, upscale_plan())
        .unwrap();
    let mut submitted = update(GenerationJobStatus::Generating, Some(0.15));
    submitted.provider_job_id = Some("fal::fixture-job".to_string());
    core.update_generation_job_for_project(epoch, &bundle, &job.job_id, submitted)
        .unwrap();

    let rows = core.generation_log().entries.len();
    let artifacts = publication_artifacts(&bundle);
    let authority = core.project_asset_authority().unwrap();
    let identity = ProjectRoot::open(&bundle).unwrap().stable_identity();
    let manifest_bytes = fs::read(bundle.join("media.json")).unwrap();
    let log_bytes = fs::read(bundle.join("generation-log.json")).unwrap();
    let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let recorded = events.clone();
    core.subscribe(move |event| recorded.lock().unwrap().push(event.clone()));

    for _ in 0..10 {
        assert_eq!(
            core.update_generation_job_for_project(epoch, &bundle, &job.job_id, running_poll())
                .unwrap(),
            1
        );
    }

    assert_eq!(core.generation_log().entries.len(), rows);
    assert_eq!(core.project_asset_authority().unwrap(), authority);
    assert_eq!(
        ProjectRoot::open(&bundle).unwrap().stable_identity(),
        identity
    );
    assert_eq!(fs::read(bundle.join("media.json")).unwrap(), manifest_bytes);
    assert_eq!(
        fs::read(bundle.join("generation-log.json")).unwrap(),
        log_bytes
    );
    assert_eq!(publication_artifacts(&bundle), artifacts);
    let placeholder = core
        .media()
        .entries
        .into_iter()
        .find(|entry| entry.id == job.placeholder_asset_ids[0])
        .unwrap();
    assert_eq!(placeholder.generation_input.unwrap().progress, Some(0.5));
    // Only the first poll moved progress; it is announced but never saved.
    let events = events.lock().unwrap();
    assert_eq!(events.len(), 1, "{events:?}");
    assert!(matches!(events[0], CoreEvent::MediaChanged { .. }));
}

#[test]
fn identical_updates_append_no_rows_while_each_transition_appends_one() {
    let (_temp, bundle) = saved_project();
    let core = AppCore::new();
    core.open_project(&bundle).unwrap();
    let epoch = core.runtime_snapshot().project_epoch;
    let mut plan = upscale_plan();
    plan.output_count = 2;
    let job = core
        .begin_generation_job_for_project(epoch, &bundle, plan)
        .unwrap();
    assert_eq!(core.generation_log().entries.len(), 2);

    for _ in 0..50 {
        core.update_generation_job_for_project(epoch, &bundle, &job.job_id, running_poll())
            .unwrap();
    }
    assert_eq!(core.generation_log().entries.len(), 4);

    let mut downloading = update(GenerationJobStatus::Downloading, Some(0.8));
    downloading.cost_credits = Some(7);
    core.update_generation_job_for_project(epoch, &bundle, &job.job_id, downloading)
        .unwrap();
    for _ in 0..5 {
        core.update_generation_job_for_project(
            epoch,
            &bundle,
            &job.job_id,
            update(GenerationJobStatus::Downloading, Some(0.8)),
        )
        .unwrap();
    }
    let log = Project::open(&bundle).unwrap().generation_log.unwrap();
    assert_eq!(log.entries.len(), 6);
    assert_eq!(log.total_credits(), 7);
    assert_eq!(log, core.generation_log());
}

#[test]
fn retained_root_identity_survives_polls_transitions_and_media_finalization() {
    let (_temp, bundle) = saved_project();
    let held = ProjectRoot::open(&bundle).unwrap();
    let artifacts = publication_artifacts(&bundle);
    #[cfg(unix)]
    let source_inode = {
        use std::os::unix::fs::MetadataExt;
        fs::metadata(bundle.join("media/source.png")).unwrap().ino()
    };
    let core = AppCore::new();
    core.open_project(&bundle).unwrap();
    let epoch = core.runtime_snapshot().project_epoch;
    let still_bound = || {
        core.ensure_project_root_identity_for_project(epoch, &bundle, held.identity())
            .expect("a retained root must stay bound to the open session")
    };

    let job = core
        .begin_generation_job_for_project(epoch, &bundle, upscale_plan())
        .unwrap();
    still_bound();
    for _ in 0..5 {
        core.update_generation_job_for_project(epoch, &bundle, &job.job_id, running_poll())
            .unwrap();
        still_bound();
    }
    core.update_generation_job_for_project(
        epoch,
        &bundle,
        &job.job_id,
        update(GenerationJobStatus::Downloading, Some(0.8)),
    )
    .unwrap();
    still_bound();

    let asset_id = &job.placeholder_asset_ids[0];
    let leaf = format!("{asset_id}.png");
    core.finalize_generation_output_with_media_for_project(
        epoch,
        &bundle,
        PreparedGenerationOutput {
            asset_id: asset_id.clone(),
            relative_path: format!("media/{leaf}"),
            probe: history_image_probe(),
            created_at: Some(800_000_002.0),
        },
        &leaf,
        6,
        &mut std::io::Cursor::new(b"result"),
    )
    .unwrap();
    still_bound();

    assert_eq!(publication_artifacts(&bundle), artifacts);
    assert_eq!(
        fs::read(bundle.join("media").join(&leaf)).unwrap(),
        b"result"
    );
    assert_eq!(fs::read(bundle.join("thumbnail.jpg")).unwrap(), b"cover");
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        assert_eq!(
            fs::metadata(bundle.join("media/source.png")).unwrap().ino(),
            source_inode,
            "existing media must not be copied"
        );
    }
    let reopened = Project::open(&bundle).unwrap();
    let output = reopened
        .manifest
        .entries
        .iter()
        .find(|entry| &entry.id == asset_id)
        .unwrap();
    assert_eq!(
        output.generation_input.as_ref().unwrap().status,
        Some(GenerationJobStatus::Ready)
    );
    assert_eq!(reopened.generation_log.unwrap().entries.len(), 4);
}

#[cfg(unix)]
#[test]
fn failed_manifest_commit_rolls_back_the_published_media_leaf() {
    let (_temp, bundle) = saved_project();
    let core = AppCore::new();
    core.open_project(&bundle).unwrap();
    let epoch = core.runtime_snapshot().project_epoch;
    let job = core
        .begin_generation_job_for_project(epoch, &bundle, upscale_plan())
        .unwrap();
    for status in [
        GenerationJobStatus::Generating,
        GenerationJobStatus::Downloading,
    ] {
        core.update_generation_job_for_project(epoch, &bundle, &job.job_id, update(status, None))
            .unwrap();
    }
    let before = core.media();
    let rows = core.generation_log().entries.len();
    // A non-empty directory at `media.json` makes its atomic replace fail.
    fs::remove_file(bundle.join("media.json")).unwrap();
    fs::create_dir_all(bundle.join("media.json/blocker")).unwrap();

    let asset_id = &job.placeholder_asset_ids[0];
    let leaf = format!("{asset_id}.png");
    core.finalize_generation_output_with_media_for_project(
        epoch,
        &bundle,
        PreparedGenerationOutput {
            asset_id: asset_id.clone(),
            relative_path: format!("media/{leaf}"),
            probe: history_image_probe(),
            created_at: None,
        },
        &leaf,
        6,
        &mut std::io::Cursor::new(b"result"),
    )
    .expect_err("the manifest commit point failed");

    assert!(!bundle.join("media").join(&leaf).exists());
    assert_eq!(
        fs::read_dir(bundle.join("media")).unwrap().count(),
        1,
        "only the pre-existing source remains"
    );
    assert_eq!(core.media(), before);
    assert_eq!(core.generation_log().entries.len(), rows);
}

#[test]
fn unflushed_manifest_commit_keeps_the_finalized_output_and_its_media_leaf() {
    let (_temp, bundle) = saved_project();
    let core = AppCore::new();
    core.open_project(&bundle).unwrap();
    let epoch = core.runtime_snapshot().project_epoch;
    let job = core
        .begin_generation_job_for_project(epoch, &bundle, upscale_plan())
        .unwrap();
    for status in [
        GenerationJobStatus::Generating,
        GenerationJobStatus::Downloading,
    ] {
        core.update_generation_job_for_project(epoch, &bundle, &job.job_id, update(status, None))
            .unwrap();
    }
    let rows = core.generation_log().entries.len();
    let asset_id = &job.placeholder_asset_ids[0];
    let leaf = format!("{asset_id}.png");

    // The media leaf and log flushes succeed; the manifest's commit flush
    // fails after its rename.
    opentake_project::bundle::test_hooks::fail_directory_sync_after(2);
    let error = core
        .finalize_generation_output_with_media_for_project(
            epoch,
            &bundle,
            PreparedGenerationOutput {
                asset_id: asset_id.clone(),
                relative_path: format!("media/{leaf}"),
                probe: history_image_probe(),
                created_at: None,
            },
            &leaf,
            6,
            &mut std::io::Cursor::new(b"result"),
        )
        .expect_err("the unconfirmed manifest flush must be reported");

    assert!(error.is_committed(), "{error:?}");
    assert_eq!(
        fs::read(bundle.join("media").join(&leaf)).unwrap(),
        b"result"
    );
    let live = core
        .media()
        .entries
        .into_iter()
        .find(|entry| &entry.id == asset_id)
        .unwrap();
    assert_eq!(
        live.generation_input.as_ref().unwrap().status,
        Some(GenerationJobStatus::Ready)
    );
    assert_eq!(core.generation_log().entries.len(), rows + 1);
    let persisted = Project::open(&bundle).unwrap();
    let persisted_entry = persisted
        .manifest
        .entries
        .iter()
        .find(|entry| &entry.id == asset_id)
        .unwrap();
    assert_eq!(persisted_entry, &live);
    assert_eq!(persisted.generation_log.unwrap(), core.generation_log());
}

#[test]
fn generation_log_retention_keeps_a_heavily_used_project_editable() {
    use opentake_project::{GenerationLog, GenerationLogEntry, GENERATION_LOG_RETENTION_BYTES};
    const READ_LIMIT: u64 = 16 * 1024 * 1024;

    let (_temp, bundle) = saved_project();
    // A long-lived project whose finished jobs each logged every lifecycle
    // step: queued -> generating -> downloading -> finalizing -> ready.
    let mut entries = Vec::new();
    for job in 0..4_800_u64 {
        let job_id = format!("{job:08x}-0000-4000-8000-000000000000");
        let asset_id = format!("{job:08x}-0000-4000-8000-00000000a55e");
        for (step, status) in [
            GenerationJobStatus::Queued,
            GenerationJobStatus::Generating,
            GenerationJobStatus::Downloading,
            GenerationJobStatus::Finalizing,
            GenerationJobStatus::Ready,
        ]
        .into_iter()
        .enumerate()
        {
            entries.push(GenerationLogEntry::job_event(
                format!("{job:08x}-{step:04x}-4000-8000-000000000001"),
                job_id.clone(),
                "fal:fixture-upscaler",
                (status == GenerationJobStatus::Ready).then_some(12),
                "fal",
                Some(format!("fal::{job_id}")),
                asset_id.clone(),
                status,
                Some(step as f64 / 4.0),
                None,
                Some(800_000_000.0 + job as f64),
                Some("source-image".to_string()),
                Some("source-clip".to_string()),
            ));
        }
    }
    let log = GenerationLog {
        version: 1,
        entries,
    };
    let credits = log.total_credits();
    let mut project = Project::open(&bundle).unwrap();
    project.generation_log = Some(log);
    project.save().unwrap();
    let seeded = fs::metadata(bundle.join("generation-log.json"))
        .unwrap()
        .len();
    assert!(
        seeded > GENERATION_LOG_RETENTION_BYTES as u64 && seeded < READ_LIMIT,
        "fixture log is {seeded} bytes"
    );

    let core = AppCore::new();
    core.open_project(&bundle).unwrap();
    let epoch = core.runtime_snapshot().project_epoch;
    let job = core
        .begin_generation_job_for_project(epoch, &bundle, upscale_plan())
        .unwrap();

    let retained = fs::metadata(bundle.join("generation-log.json"))
        .unwrap()
        .len();
    assert!(
        retained <= GENERATION_LOG_RETENTION_BYTES as u64,
        "retained log is {retained} bytes"
    );
    let reopened = Project::open(&bundle).unwrap();
    assert!(!reopened.compatibility().is_read_only());
    let reopened_log = reopened.generation_log.unwrap();
    assert_eq!(reopened_log, core.generation_log());
    assert_eq!(reopened_log.total_credits(), credits);
    assert!(reopened_log.entries.iter().any(|row| {
        row.job_id.as_deref() == Some(job.job_id.as_str())
            && row.status == Some(GenerationJobStatus::Queued)
    }));

    let reopened_core = AppCore::new();
    reopened_core.open_project(&bundle).unwrap();
    add_history_clip(&reopened_core);
    reopened_core
        .save_project(None)
        .expect("the reopened project stays editable");
}

/// Acceptance timing for issue #76: with 256 MiB under `media/`, one durable
/// transition and every concurrent session read stay under 50 ms because no
/// media byte is copied. Ignored by default because fsync latency on shared
/// CI disks is not a stable timing source; run it explicitly with
/// `cargo test -p opentake-core --test generation_persistence -- --ignored`.
#[test]
#[ignore]
fn durable_transition_with_large_media_stays_fast_and_does_not_block_readers() {
    use std::io::Write;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    let (_temp, bundle) = saved_project();
    let mut large = fs::File::create(bundle.join("media/large.bin")).unwrap();
    let chunk = vec![0x5a_u8; 1024 * 1024];
    for _ in 0..256 {
        large.write_all(&chunk).unwrap();
    }
    large.sync_all().unwrap();
    drop(large);
    let core = AppCore::new();
    core.open_project(&bundle).unwrap();
    let epoch = core.runtime_snapshot().project_epoch;
    let job = core
        .begin_generation_job_for_project(epoch, &bundle, upscale_plan())
        .unwrap();

    let running = std::sync::Arc::new(AtomicBool::new(true));
    let reader_core = core.clone();
    let reader_running = running.clone();
    let reader = std::thread::spawn(move || {
        let mut worst = Duration::ZERO;
        while reader_running.load(Ordering::Relaxed) {
            let started = Instant::now();
            let _ = reader_core.get_timeline();
            worst = worst.max(started.elapsed());
        }
        worst
    });
    let mut slowest = Duration::ZERO;
    for status in [
        GenerationJobStatus::Generating,
        GenerationJobStatus::Downloading,
        GenerationJobStatus::Finalizing,
    ] {
        let started = Instant::now();
        core.update_generation_job_for_project(epoch, &bundle, &job.job_id, update(status, None))
            .unwrap();
        slowest = slowest.max(started.elapsed());
    }
    running.store(false, Ordering::Relaxed);
    let worst_read = reader.join().unwrap();
    eprintln!("slowest durable transition {slowest:?}, slowest concurrent read {worst_read:?}");
    assert!(slowest < Duration::from_millis(50), "{slowest:?}");
    assert!(worst_read < Duration::from_millis(50), "{worst_read:?}");
}

#[test]
fn job_updates_leave_the_ready_outputs_of_a_partly_finalized_job_alone() {
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
    let pending_id = committed.placeholder_asset_ids[1].clone();
    for (status, progress) in [
        (GenerationJobStatus::Generating, Some(0.2)),
        (GenerationJobStatus::Downloading, Some(0.8)),
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
            probe: history_image_probe(),
            created_at: Some(800_000_002.0),
        },
    )
    .unwrap();

    // A job interrupted here is resumed from its provider job id, and
    // finalization moves the job to Downloading again while output 0 is
    // already Ready.
    let mut resumed = update(GenerationJobStatus::Downloading, Some(0.85));
    resumed.provider_job_id = Some("fal::resumed".to_string());
    assert_eq!(
        core.update_generation_job_for_project(
            runtime.project_epoch,
            &bundle,
            &committed.job_id,
            resumed,
        )
        .unwrap(),
        1
    );

    let reopened = Project::open(&bundle).unwrap();
    let input = |id: &str| {
        reopened
            .manifest
            .entries
            .iter()
            .find(|entry| entry.id == id)
            .unwrap()
            .generation_input
            .clone()
            .unwrap()
    };
    let ready = input(&ready_id);
    assert_eq!(ready.status, Some(GenerationJobStatus::Ready));
    assert_ne!(ready.provider_job_id.as_deref(), Some("fal::resumed"));
    let pending = input(&pending_id);
    assert_eq!(pending.status, Some(GenerationJobStatus::Downloading));
    assert_eq!(pending.provider_job_id.as_deref(), Some("fal::resumed"));

    // A job whose outputs are all terminal still refuses to restart.
    core.fail_generation_output_for_project(
        runtime.project_epoch,
        &bundle,
        &pending_id,
        "GENERATION_DOWNLOAD_FAILED",
        None,
    )
    .unwrap();
    assert!(core
        .update_generation_job_for_project(
            runtime.project_epoch,
            &bundle,
            &committed.job_id,
            update(GenerationJobStatus::Downloading, Some(0.8)),
        )
        .is_err());
}
