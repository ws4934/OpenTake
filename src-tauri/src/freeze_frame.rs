//! Background capture ownership for the synchronous, revision-bound edit API.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use opentake_core::{AppCore, CmdError, ProjectRevision};
use tauri::{AppHandle, Manager, Runtime};

use crate::commands::EditRequest;
use crate::media::prewarm::{PrewarmKind, PrewarmScheduler};
use crate::render::PreparedFreezeFrame;

const MAX_PREPARATIONS: usize = 32;
const PREPARATION_TTL: Duration = Duration::from_secs(60);

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FreezeBinding {
    pub revision: ProjectRevision,
    pub project_path: Option<PathBuf>,
    pub clip_id: String,
    pub at_frame: i32,
    pub duration_frames: i32,
}

struct Preparation {
    binding: FreezeBinding,
    capture: PreparedFreezeFrame,
    expires: Instant,
}

#[derive(Clone, Default)]
pub(crate) struct FreezeFramePreparations(Arc<Mutex<HashMap<String, Preparation>>>);

impl FreezeFramePreparations {
    pub fn new() -> Self {
        let store = Self::default();
        let weak = Arc::downgrade(&store.0);
        tauri::async_runtime::spawn(async move {
            loop {
                tokio::time::sleep(PREPARATION_TTL).await;
                let Some(inner) = weak.upgrade() else { break };
                let now = Instant::now();
                inner
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .retain(|_, preparation| preparation.expires > now);
            }
        });
        store
    }

    pub fn invalidate(&self) {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).clear();
    }

    pub fn insert(
        &self,
        binding: FreezeBinding,
        capture: PreparedFreezeFrame,
    ) -> Result<String, String> {
        let mut preparations = self.0.lock().unwrap_or_else(|p| p.into_inner());
        let now = Instant::now();
        preparations.retain(|_, preparation| preparation.expires > now);
        if preparations.len() >= MAX_PREPARATIONS {
            return Err("too many uncommitted freeze-frame preparations".into());
        }
        let ticket = uuid::Uuid::new_v4().to_string();
        preparations.insert(
            ticket.clone(),
            Preparation {
                binding,
                capture,
                expires: now + PREPARATION_TTL,
            },
        );
        Ok(ticket)
    }

    pub fn take(
        &self,
        ticket: &str,
        binding: &FreezeBinding,
    ) -> Result<PreparedFreezeFrame, String> {
        let prepared = self
            .0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(ticket)
            .ok_or_else(|| {
                "freeze-frame preparation is missing, expired or already used".to_string()
            })?;
        if prepared.expires <= Instant::now() || prepared.binding != *binding {
            return Err("freeze-frame preparation does not match this edit".into());
        }
        Ok(prepared.capture)
    }
}

fn validation(message: String) -> CmdError {
    CmdError {
        code: "validation".into(),
        message,
        params: Default::default(),
    }
}

#[tauri::command]
pub async fn prepare_freeze_frame<R: Runtime>(
    app: AppHandle<R>,
    command: EditRequest,
    expected_project_epoch: u64,
    expected_timeline_version: u64,
    expected_project_path: Option<String>,
) -> Result<String, CmdError> {
    let EditRequest::FreezeFrame {
        clip_id,
        at_frame,
        duration_frames,
    } = command
    else {
        return Err(validation(
            "only freezeFrame has a capture preparation".into(),
        ));
    };
    let binding = FreezeBinding {
        revision: ProjectRevision {
            project_epoch: expected_project_epoch,
            version: expected_timeline_version,
        },
        project_path: expected_project_path.map(PathBuf::from),
        clip_id,
        at_frame,
        duration_frames,
    };
    let scheduler = app.state::<PrewarmScheduler>().inner().clone();
    let handle = app.clone();
    let work_binding = binding.clone();
    let capture = scheduler
        .request(
            expected_project_epoch,
            PrewarmKind::TimelineVisuals,
            format!("freeze:{}", uuid::Uuid::new_v4()),
            move |context| {
                let core = handle.state::<AppCore>();
                let _activity = crate::updater::begin_mutating_activity(
                    &handle.state::<crate::updater::InstallAdmissionGate>(),
                )?;
                let snapshot = {
                    // The identity lease protects only the snapshot, never decoding.
                    let _identity = core.lock_project_identity_workflow();
                    core.ensure_project_mutable()
                        .map_err(|error| error.to_string())?;
                    let snapshot = core.runtime_snapshot();
                    context.ensure_project(snapshot.project_epoch)?;
                    if snapshot.project_epoch != work_binding.revision.project_epoch
                        || snapshot.version != work_binding.revision.version
                        || snapshot.project_dir != work_binding.project_path
                    {
                        return Err(
                            "project or timeline changed before freeze-frame capture".into()
                        );
                    }
                    crate::commands::validate_freeze_frame_request(
                        &core,
                        &work_binding.clip_id,
                        work_binding.at_frame,
                        work_binding.duration_frames,
                    )?;
                    snapshot
                };
                crate::render::capture_freeze_frame(
                    &core,
                    &handle.state::<crate::render::RenderState>(),
                    &handle.state::<crate::media::MediaState>(),
                    snapshot,
                    &work_binding.clip_id,
                    work_binding.at_frame,
                    &context.cancel_token(),
                )
            },
        )
        .await
        .map_err(validation)?;
    // The ticket exposes no filesystem path or manifest data to IPC clients.
    app.state::<FreezeFramePreparations>()
        .insert(binding, capture)
        .map_err(validation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentake_core::{EditCommand, ProbedMedia};
    use opentake_domain::ClipType;
    use opentake_ops::command::ClipEntry;

    fn fixture(root: &std::path::Path) -> (AppCore, FreezeBinding, PreparedFreezeFrame, PathBuf) {
        let core = AppCore::new();
        core.save_project(Some(root.join("Original.opentake")))
            .unwrap();
        let source = root.join("source.mp4");
        std::fs::write(&source, b"source fixture").unwrap();
        let media = core
            .import_media_file(&source, "source", &ProbedMedia::default())
            .unwrap();
        core.apply(EditCommand::InsertTrack {
            kind: ClipType::Video,
            at: None,
        })
        .unwrap();
        let result = core
            .apply(EditCommand::AddClips {
                entries: vec![ClipEntry {
                    media_ref: media.id,
                    media_type: ClipType::Video,
                    source_clip_type: ClipType::Video,
                    track_index: 0,
                    start_frame: 0,
                    duration_frames: 60,
                    trim_start_frame: None,
                    trim_end_frame: None,
                    has_audio: false,
                    add_linked_audio: false,
                    transform: None,
                }],
            })
            .unwrap();
        let binding = FreezeBinding {
            revision: core.project_revision(),
            project_path: core.project_dir(),
            clip_id: result.affected_clip_ids[0].clone(),
            at_frame: 15,
            duration_frames: 30,
        };
        let path = root.join("freeze.png");
        std::fs::write(&path, b"owned capture fixture").unwrap();
        let media = core
            .prepare_media_file_entry(&path, "freeze", &ProbedMedia::default())
            .unwrap();
        let capture = PreparedFreezeFrame::fixture(path.clone(), media);
        (core, binding, capture, path)
    }

    fn wait_for_cleanup(path: &std::path::Path) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while path.exists() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(!path.exists(), "uncommitted capture was not cleaned up");
    }

    fn commit(
        core: &AppCore,
        store: &FreezeFramePreparations,
        binding: &FreezeBinding,
        ticket: String,
    ) -> Result<opentake_core::EditResultDto, CmdError> {
        let app = tauri::test::mock_builder()
            .manage(core.clone())
            .manage(store.clone())
            .manage(crate::updater::InstallAdmissionGate::default())
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .unwrap();
        crate::commands::edit_apply(
            app.state::<AppCore>(),
            app.state::<FreezeFramePreparations>(),
            app.state::<crate::updater::InstallAdmissionGate>(),
            EditRequest::FreezeFrame {
                clip_id: binding.clip_id.clone(),
                at_frame: binding.at_frame,
                duration_frames: binding.duration_frames,
            },
            Some(ticket),
            binding.revision.project_epoch,
            binding.revision.version,
            binding
                .project_path
                .as_ref()
                .map(|path| path.to_string_lossy().into_owned()),
        )
    }

    #[test]
    fn prepared_capture_is_one_use_and_survives_commit_undo_redo() {
        let root = tempfile::tempdir().unwrap();
        let (core, binding, capture, path) = fixture(root.path());
        let store = FreezeFramePreparations::default();
        let ticket = store.insert(binding.clone(), capture).unwrap();
        assert!(
            commit(&core, &store, &binding, ticket.clone())
                .unwrap()
                .changed
        );
        let revision = core.project_revision();
        assert!(commit(&core, &store, &binding, ticket).is_err());
        assert_eq!(core.project_revision(), revision);
        core.undo().unwrap();
        core.redo().unwrap();
        assert!(
            path.is_file(),
            "committed capture belongs to the document and history"
        );
    }

    #[test]
    fn stale_capture_cannot_edit_a_changed_project_or_leave_a_file() {
        for change in ["timeline", "new_project", "save_as"] {
            let root = tempfile::tempdir().unwrap();
            let (core, binding, capture, path) = fixture(root.path());
            let store = FreezeFramePreparations::default();
            let ticket = store.insert(binding.clone(), capture).unwrap();
            match change {
                "timeline" => {
                    core.apply(EditCommand::InsertTrack {
                        kind: ClipType::Video,
                        at: None,
                    })
                    .unwrap();
                }
                "new_project" => {
                    core.new_project();
                }
                "save_as" => {
                    core.save_project(Some(root.path().join("Other.opentake")))
                        .unwrap();
                }
                _ => unreachable!(),
            }
            let before = core.runtime_snapshot();
            assert!(commit(&core, &store, &binding, ticket).is_err(), "{change}");
            let after = core.runtime_snapshot();
            assert_eq!(after.timeline, before.timeline, "{change}");
            assert_eq!(after.media, before.media, "{change}");
            assert_eq!(after.version, before.version, "{change}");
            assert_eq!(after.project_epoch, before.project_epoch, "{change}");
            wait_for_cleanup(&path);
        }
    }

    #[test]
    fn ticket_cannot_authorize_different_capture_inputs_and_expires() {
        for mismatch in [
            "clip",
            "frame",
            "duration",
            "epoch",
            "version",
            "path",
            "expiry",
            "invalidate",
        ] {
            let root = tempfile::tempdir().unwrap();
            let (_core, mut binding, capture, path) = fixture(root.path());
            let store = FreezeFramePreparations::default();
            let ticket = store.insert(binding.clone(), capture).unwrap();
            match mismatch {
                "clip" => binding.clip_id = "different".into(),
                "frame" => binding.at_frame += 1,
                "duration" => binding.duration_frames += 1,
                "epoch" => binding.revision.project_epoch += 1,
                "version" => binding.revision.version += 1,
                "path" => binding.project_path = Some(root.path().join("Other.opentake")),
                "expiry" => {
                    store.0.lock().unwrap().get_mut(&ticket).unwrap().expires = Instant::now()
                }
                "invalidate" => store.invalidate(),
                _ => unreachable!(),
            }
            assert!(store.take(&ticket, &binding).is_err(), "{mismatch}");
            wait_for_cleanup(&path);
        }
    }
}
