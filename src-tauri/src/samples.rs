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
        match backend {
            Some(backend) => service.materialize(&backend, &slug, progress),
            None => service.materialize_builtin(&slug, progress),
        }
        .map(|path| path.to_string_lossy().into_owned())
    })
    .await
    .map_err(|error| format!("sample materialization task failed: {error}"))?
}
