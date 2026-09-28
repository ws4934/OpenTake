use std::time::Duration;

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, Runtime};
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};

use crate::close_coordinator::{CloseCoordinator, CloseIntent, FailedCloseChoice};
use crate::instance_lock::InstanceLock;

/// Event asking the WebView to offer Save As / Don't Save / Cancel after the
/// save before a close or quit failed.
const CLOSE_SAVE_FAILED_EVENT: &str = "close_save_failed";
/// How long the WebView has to claim that prompt before the native fallback
/// shows it instead (for example when the WebView's listener never registered).
const PROMPT_CLAIM_TIMEOUT: Duration = Duration::from_secs(3);

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CloseSaveFailed {
    id: u64,
    intent: &'static str,
    message: String,
}

fn update_prevents_exit<R: Runtime>(app: &AppHandle<R>) -> bool {
    app.try_state::<crate::updater::UpdateCoordinator>()
        .is_some_and(|coordinator| coordinator.prevents_user_exit())
}

pub(crate) fn request_close(app: &AppHandle, intent: CloseIntent) {
    if update_prevents_exit(app) {
        return;
    }
    let Some(coordinator) = app.try_state::<CloseCoordinator>() else {
        return;
    };
    if !coordinator.request(intent) {
        return;
    }
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        let saved = crate::commands::save_current_project_before_exit(app.clone()).await;
        match app.state::<CloseCoordinator>().finish(saved) {
            Ok(intent) => perform_close(&app, intent),
            Err((intent, error)) => offer_failed_close_choice(&app, intent, error.message),
        }
    });
}

fn perform_close(app: &AppHandle, intent: CloseIntent) {
    match intent {
        CloseIntent::Exit => app.exit(0),
        CloseIntent::Hide => {
            let result = (|| {
                let window = app
                    .get_webview_window("main")
                    .ok_or_else(|| "main window is unavailable".to_string())?;
                window.hide().map_err(|error| error.to_string())?;
                app.emit("go_home", ()).map_err(|error| error.to_string())
            })();
            if let Err(error) = result {
                report_failure(app, &error);
            }
        }
    }
}

/// A failing save (deleted bundle, unplugged volume, full disk) must not trap
/// the user: offer Save As / Don't Save / Cancel instead of only cancelling.
/// Save As needs the WebView's save dialog, which also grants the chosen
/// path, so the WebView shows the choice; a native Don't Save / Keep Open
/// prompt takes over if the WebView does not claim it in time or cannot show
/// it.
fn offer_failed_close_choice(app: &AppHandle, intent: CloseIntent, error: String) {
    eprintln!(
        "[lifecycle] save before {} failed: {error}",
        intent.as_str()
    );
    if let Err(show_error) = show_main_window(app) {
        eprintln!("[lifecycle] restore window: {show_error}");
    }
    let failure = app
        .state::<CloseCoordinator>()
        .record_failure(intent, error);
    let event = CloseSaveFailed {
        id: failure.id,
        intent: intent.as_str(),
        message: failure.message.clone(),
    };
    if let Err(emit_error) = app.emit(CLOSE_SAVE_FAILED_EVENT, event) {
        eprintln!("[lifecycle] offer failed-save choice: {emit_error}");
        show_native_failed_close_prompt(app, failure.id);
        return;
    }
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        tokio::time::sleep(PROMPT_CLAIM_TIMEOUT).await;
        show_native_failed_close_prompt(&app, failure.id);
    });
}

/// Show failure `id` natively unless a prompt already owns it or it was
/// answered or superseded. It carries its own error, never a newer one's.
fn show_native_failed_close_prompt(app: &AppHandle, id: u64) {
    let Some(failure) = app.state::<CloseCoordinator>().claim_prompt(id) else {
        return;
    };
    let app = app.clone();
    app.dialog()
        .message(format!(
            "工程未能保存。不保存关闭会丢弃未保存的修改，但不会删除磁盘上的任何文件。\nThe project could not be saved. Closing without saving discards the unsaved changes but deletes nothing on disk.\n\n{}",
            failure.message
        ))
        .title("OpenTake")
        .kind(MessageDialogKind::Warning)
        .buttons(MessageDialogButtons::OkCancelCustom(
            "不保存关闭 / Close Without Saving".into(),
            "保持打开 / Keep Open".into(),
        ))
        .show(move |close_without_saving| {
            let choice = if close_without_saving {
                FailedCloseChoice::Discard
            } else {
                FailedCloseChoice::Cancel
            };
            if let Err(error) = resolve_failed_close_choice(&app, id, choice) {
                report_failure(&app, &error);
            }
        });
}

fn resolve_failed_close_choice(
    app: &AppHandle,
    id: u64,
    choice: FailedCloseChoice,
) -> Result<(), String> {
    let coordinator = app.state::<CloseCoordinator>();
    if choice == FailedCloseChoice::Native {
        if coordinator.release_prompt(id).is_some() {
            show_native_failed_close_prompt(app, id);
        }
        return Ok(());
    }
    let Some(failure) = coordinator.take_failure(id) else {
        return Ok(()); // Superseded by a newer close request or already chosen.
    };
    match choice {
        FailedCloseChoice::Cancel | FailedCloseChoice::Native => Ok(()),
        FailedCloseChoice::Retry | FailedCloseChoice::Discard if update_prevents_exit(app) => {
            Err("an update is being installed; close again when it finishes".into())
        }
        FailedCloseChoice::Retry => {
            request_close(app, failure.intent);
            Ok(())
        }
        FailedCloseChoice::Discard => match failure.intent {
            CloseIntent::Exit => {
                app.exit(0);
                Ok(())
            }
            CloseIntent::Hide => {
                // A hidden window keeps the process and its session alive, so
                // the declined edits must leave the core now: otherwise a later
                // exit or project-switch save would write them after all.
                let app = app.clone();
                tauri::async_runtime::spawn(async move {
                    match discard_open_session(&app).await {
                        Ok(()) => perform_close(&app, CloseIntent::Hide),
                        Err(error) => {
                            report_failure(&app, &format!("close without saving: {error}"))
                        }
                    }
                });
                Ok(())
            }
        },
    }
}

/// Replace the open project with a new, unsaved session, exactly as File >
/// New does (stopping playback for the old project first). Nothing on disk is
/// written or deleted; the WebView resets on the resulting `project_opened`.
async fn discard_open_session<R: Runtime>(app: &AppHandle<R>) -> Result<(), String> {
    crate::commands::project_new(app.clone(), None)
        .await
        .map(|_| ())
        .map_err(|error| error.to_string())
}

/// The WebView claims failed-save prompt `id` it was sent. `false` means the
/// native fallback already showed it or it is no longer waiting.
#[tauri::command]
pub(crate) fn lifecycle_claim_failed_close(app: AppHandle, id: u64) -> bool {
    app.state::<CloseCoordinator>().claim_prompt(id).is_some()
}

/// Apply the user's choice for failed close or quit save `id`.
#[tauri::command]
pub(crate) fn lifecycle_resolve_failed_close(
    app: AppHandle,
    id: u64,
    choice: FailedCloseChoice,
) -> Result<(), String> {
    resolve_failed_close_choice(&app, id, choice)
}

fn show_main_window(app: &AppHandle) -> Result<(), String> {
    let window = app
        .get_webview_window("main")
        .ok_or_else(|| "main window is unavailable".to_string())?;
    window.show().map_err(|error| error.to_string())?;
    window.unminimize().map_err(|error| error.to_string())?;
    window.set_focus().map_err(|error| error.to_string())
}

fn report_failure(app: &AppHandle, error: &str) {
    eprintln!("[lifecycle] {error}");
    if let Err(show_error) = show_main_window(app) {
        eprintln!("[lifecycle] restore window: {show_error}");
    }
    // Native UI remains available even if WebView listeners failed to register.
    app.dialog()
        .message(format!(
            "操作未完成，工程仍保持打开。请修复问题后重试。\nThe operation was cancelled; your project remains open. Resolve the error and retry.\n\n{error}"
        ))
        .title("OpenTake")
        .kind(MessageDialogKind::Error)
        .show(|_| {});
}

/// A private filesystem signal avoids opening an unauthenticated local server
/// merely to focus the existing process. The process lock covers all shared
/// stores and is acquired before any of them are initialized.
pub(crate) fn watch_duplicate_launches(app: AppHandle) -> std::io::Result<()> {
    std::thread::Builder::new()
        .name("opentake-activation".into())
        .spawn(move || loop {
            std::thread::sleep(Duration::from_millis(500));
            match app.state::<InstanceLock>().take_activation() {
                Ok(false) => {}
                Ok(true) => {
                    if let Err(error) = show_main_window(&app) {
                        report_failure(&app, &error);
                    }
                }
                Err(error) => {
                    report_failure(&app, &format!("receive duplicate launch: {error}"));
                    break;
                }
            }
        })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentake_core::{AppCore, EditCommand};

    fn mock_app() -> tauri::App<tauri::test::MockRuntime> {
        let core = AppCore::new();
        let epoch = core.project_revision().project_epoch;
        let builder = tauri::test::mock_builder()
            .manage(core)
            .manage(crate::updater::InstallAdmissionGate::default())
            .manage(crate::commands::ProjectLifecycleCoordinator::default())
            .manage(crate::media::prewarm::PrewarmScheduler::new(epoch));
        #[cfg(feature = "playback-engine")]
        let builder = builder.manage(crate::playback::PlaybackState::new());
        builder
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .expect("build managed mock app")
    }

    #[test]
    fn dont_save_drops_the_edits_so_no_later_save_writes_them() {
        let dir = tempfile::tempdir().unwrap();
        let bundle = dir.path().join("Unplugged.opentake");
        let app = mock_app();
        let core = app.state::<AppCore>();
        core.save_project(Some(bundle.clone())).unwrap();
        let saved = std::fs::read(bundle.join(opentake_project::layout::TIMELINE_FILE)).unwrap();
        core.apply(EditCommand::SetTimelineSettings {
            fps: 25,
            width: 640,
            height: 360,
        })
        .unwrap();

        tauri::async_runtime::block_on(discard_open_session(app.handle())).unwrap();

        assert_eq!(core.runtime_snapshot().project_dir, None);
        // The exit save and the next project boundary find nothing to write.
        assert_eq!(core.save_project_before_exit_if(|| true).unwrap(), None);
        assert_eq!(
            std::fs::read(bundle.join(opentake_project::layout::TIMELINE_FILE)).unwrap(),
            saved,
            "the declined edits never reach the bundle"
        );
        assert!(bundle.is_dir(), "Don't Save deletes nothing");
    }
}
