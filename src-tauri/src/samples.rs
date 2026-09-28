//! Sample-project command and progress events.

mod service;

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};

use service::SampleProjectService;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct SampleProgress {
    slug: String,
    completed: usize,
    total: usize,
}

#[tauri::command]
pub async fn sample_project_materialize(app: AppHandle, slug: String) -> Result<String, String> {
    let activity = crate::updater::begin_mutating_activity(
        &app.state::<crate::updater::InstallAdmissionGate>(),
    )?;
    let backend = crate::account::configured_backend_url()?;
    let projects = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("resolve sample project storage: {error}"))?
        .join("Projects");
    let core = app.state::<opentake_core::AppCore>().inner().clone();
    let progress_app = app.clone();
    let progress_slug = slug.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let _activity = activity;
        let service = SampleProjectService::new(projects)?;
        let progress = |fraction: f64| {
            let total = 10_000;
            let _ = progress_app.emit(
                "sample-materialization-progress",
                SampleProgress {
                    slug: progress_slug.clone(),
                    completed: (fraction * total as f64).round() as usize,
                    total,
                },
            );
        };
        // The open project may hold unsaved edits the on-disk identity cannot
        // see, so it is never handed out as an unmodified copy.
        let in_use = |bundle: &std::path::Path| {
            core.runtime_snapshot()
                .project_dir
                .is_some_and(|open| same_path(&open, bundle))
        };
        match backend {
            Some(backend) => service.materialize(&backend, &slug, in_use, progress),
            None => service.materialize_builtin(&slug, in_use, progress),
        }
        .map(|path| path.to_string_lossy().into_owned())
    })
    .await
    .map_err(|error| format!("sample materialization task failed: {error}"))?
}

fn same_path(left: &std::path::Path, right: &std::path::Path) -> bool {
    left == right
        || matches!(
            (std::fs::canonicalize(left), std::fs::canonicalize(right)),
            (Ok(left), Ok(right)) if left == right
        )
}

#[cfg(test)]
mod tests {
    use super::service::SampleProjectService;
    use opentake_core::AppCore;

    #[test]
    fn a_copy_opened_and_quit_without_edits_is_reused() {
        let storage = tempfile::tempdir().unwrap();
        let service = SampleProjectService::new(storage.path().to_path_buf()).unwrap();
        let first = service
            .materialize_builtin("quick-tutorial", |_| false, |_| {})
            .unwrap();

        // Open the copy and quit through the desktop's exit save, then open
        // and save it explicitly: neither may count as a user edit.
        let core = AppCore::new();
        core.open_project(&first).unwrap();
        core.save_project_before_exit_if(|| true).unwrap();
        core.save_project(None).unwrap();
        core.new_project();

        let second = service
            .materialize_builtin("quick-tutorial", |_| false, |_| {})
            .unwrap();
        assert_eq!(second, first);
        assert_eq!(std::fs::read_dir(storage.path()).unwrap().count(), 1);

        // A real edit saved by the core is never handed out again.
        core.open_project(&first).unwrap();
        core.apply(opentake_core::EditCommand::SetTimelineSettings {
            fps: 25,
            width: 640,
            height: 360,
        })
        .unwrap();
        core.save_project(None).unwrap();
        core.new_project();
        let third = service
            .materialize_builtin("quick-tutorial", |_| false, |_| {})
            .unwrap();
        assert_ne!(third, first);
    }
}
