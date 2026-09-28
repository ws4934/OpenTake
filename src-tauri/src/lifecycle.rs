use std::time::Duration;

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};
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
    intent: &'static str,
    message: String,
}

pub(crate) fn request_close(app: &AppHandle, intent: CloseIntent) {
    if app
        .try_state::<crate::updater::UpdateCoordinator>()
        .is_some_and(|coordinator| coordinator.prevents_user_exit())
    {
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
            Err((intent, error)) => offer_failed_close_choice(&app, intent, &error.message),
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
/// Save As needs the WebView's file dialog, which also approves the chosen
/// path, so the WebView shows the choice; a native Don't Save / Cancel prompt
/// takes over if the WebView does not claim it in time.
fn offer_failed_close_choice(app: &AppHandle, intent: CloseIntent, error: &str) {
    eprintln!(
        "[lifecycle] save before {} failed: {error}",
        intent.as_str()
    );
    if let Err(show_error) = show_main_window(app) {
        eprintln!("[lifecycle] restore window: {show_error}");
    }
    app.state::<CloseCoordinator>().record_failure(intent);
    let event = CloseSaveFailed {
        intent: intent.as_str(),
        message: error.to_string(),
    };
    if let Err(emit_error) = app.emit(CLOSE_SAVE_FAILED_EVENT, event) {
        eprintln!("[lifecycle] offer failed-save choice: {emit_error}");
        show_native_failed_close_prompt(app, error);
        return;
    }
    let app = app.clone();
    let error = error.to_string();
    tauri::async_runtime::spawn(async move {
        tokio::time::sleep(PROMPT_CLAIM_TIMEOUT).await;
        show_native_failed_close_prompt(&app, &error);
    });
}

fn show_native_failed_close_prompt(app: &AppHandle, error: &str) {
    if app.state::<CloseCoordinator>().claim_prompt().is_none() {
        return; // The WebView owns the prompt, or the choice was already made.
    }
    let app = app.clone();
    app.dialog()
        .message(format!(
            "工程未能保存。不保存关闭会丢弃未保存的修改，但不会删除磁盘上的任何文件。\nThe project could not be saved. Closing without saving discards the unsaved changes but deletes nothing on disk.\n\n{error}"
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
            if let Err(error) = resolve_failed_close_choice(&app, choice) {
                report_failure(&app, &error);
            }
        });
}

fn resolve_failed_close_choice(app: &AppHandle, choice: FailedCloseChoice) -> Result<(), String> {
    let Some(intent) = app.state::<CloseCoordinator>().take_failure() else {
        return Ok(()); // Superseded by a newer close request or already chosen.
    };
    match choice {
        FailedCloseChoice::Cancel => Ok(()),
        FailedCloseChoice::Retry => {
            request_close(app, intent);
            Ok(())
        }
        FailedCloseChoice::Discard => {
            if app
                .try_state::<crate::updater::UpdateCoordinator>()
                .is_some_and(|coordinator| coordinator.prevents_user_exit())
            {
                return Err("an update is being installed; try again when it finishes".into());
            }
            // Only the in-memory edits are dropped: nothing on disk is deleted.
            perform_close(app, intent);
            Ok(())
        }
    }
}

/// The WebView claims the failed-save prompt it was sent. `None` means the
/// native fallback already showed it or nothing is waiting.
#[tauri::command]
pub(crate) fn lifecycle_claim_failed_close(app: AppHandle) -> Option<&'static str> {
    app.state::<CloseCoordinator>()
        .claim_prompt()
        .map(CloseIntent::as_str)
}

/// Apply the user's choice after a failed close or quit save.
#[tauri::command]
pub(crate) fn lifecycle_resolve_failed_close(
    app: AppHandle,
    choice: FailedCloseChoice,
) -> Result<(), String> {
    resolve_failed_close_choice(&app, choice)
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
