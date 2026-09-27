use std::time::Duration;

use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_dialog::{DialogExt, MessageDialogKind};

use crate::close_coordinator::{CloseCoordinator, CloseIntent};
use crate::instance_lock::InstanceLock;

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
        let action = app.state::<CloseCoordinator>().finish(saved);
        match action {
            Err(error) => report_failure(&app, &error.message),
            Ok(CloseIntent::Exit) => app.exit(0),
            Ok(CloseIntent::Hide) => {
                let result = (|| {
                    let window = app
                        .get_webview_window("main")
                        .ok_or_else(|| "main window is unavailable".to_string())?;
                    window.hide().map_err(|error| error.to_string())?;
                    app.emit("go_home", ()).map_err(|error| error.to_string())
                })();
                if let Err(error) = result {
                    report_failure(&app, &error);
                }
            }
        }
    });
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
