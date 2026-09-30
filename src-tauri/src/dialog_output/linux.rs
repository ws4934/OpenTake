//! GTK's filename APIs preserve Unix bytes. RFD's GTK backend converts those
//! C strings to UTF-8 and discards filenames that fail that conversion.

use std::path::{Path, PathBuf};

use gtk::prelude::*;

use super::{PickerKind, SaveDialogFilter};

pub(super) fn pick(
    window: &tauri::Window,
    kind: PickerKind,
    title: Option<String>,
    default_path: Option<PathBuf>,
    filters: Vec<SaveDialogFilter>,
    can_create_directories: Option<bool>,
) -> Result<Option<Vec<PathBuf>>, String> {
    if title.as_ref().is_some_and(|text| text.contains('\0'))
        || filters.iter().any(|filter| {
            filter.name.contains('\0') || filter.extensions.iter().any(|text| text.contains('\0'))
        })
    {
        return Err("native dialog text contains NUL".into());
    }
    let default_is_directory = default_path.as_ref().is_some_and(|path| path.is_dir());
    let (send, receive) = std::sync::mpsc::sync_channel(1);
    let parent = window.clone();
    window
        .run_on_main_thread(move || {
            let result = (|| {
                let parent = parent.gtk_window().map_err(|error| error.to_string())?;
                let dialog = build_dialog(
                    Some(parent.upcast_ref()),
                    kind,
                    title.as_deref(),
                    default_path.as_deref(),
                    default_is_directory,
                    &filters,
                    can_create_directories,
                )?;
                let response = dialog.run();
                let result = if response == gtk::ResponseType::Accept {
                    let paths = selected_paths(&dialog, kind);
                    if paths.is_empty() {
                        Err("native file dialog returned no filesystem selection".into())
                    } else {
                        Ok(Some(paths))
                    }
                } else {
                    Ok(None)
                };
                dialog.destroy();
                result
            })();
            // A disconnected receiver means the command stopped waiting; the
            // dialog is closed and no selection is authorized.
            let _ = send.send(result);
        })
        .map_err(|error| format!("cannot show native file dialog: {error}"))?;
    receive
        .recv()
        .map_err(|_| "native file dialog did not return a result".to_string())?
}

fn build_dialog(
    parent: Option<&gtk::Window>,
    kind: PickerKind,
    title: Option<&str>,
    default_path: Option<&Path>,
    default_is_directory: bool,
    filters: &[SaveDialogFilter],
    can_create_directories: Option<bool>,
) -> Result<gtk::FileChooserNative, String> {
    let action = match kind {
        PickerKind::Open {
            directory: true, ..
        } => gtk::FileChooserAction::SelectFolder,
        PickerKind::Open { .. } => gtk::FileChooserAction::Open,
        PickerKind::Save => gtk::FileChooserAction::Save,
    };
    let dialog = gtk::FileChooserNative::new(title, parent, action, None, None);
    dialog.set_modal(true);
    dialog.set_local_only(true);
    dialog.set_select_multiple(matches!(kind, PickerKind::Open { multiple: true, .. }));
    dialog.set_do_overwrite_confirmation(matches!(kind, PickerKind::Save));
    if let Some(can) = can_create_directories {
        dialog.set_create_folders(can);
    }
    if let Some(path) = default_path.filter(|path| !path.as_os_str().is_empty()) {
        if default_is_directory {
            if !dialog.set_current_folder(path) {
                return Err("cannot select the requested dialog directory".into());
            }
        } else {
            if let Some(parent) = path.parent().filter(|path| !path.as_os_str().is_empty()) {
                if !dialog.set_current_folder(parent) {
                    return Err("cannot select the requested dialog directory".into());
                }
            }
            if let Some(name) = path.file_name().and_then(|name| name.to_str()) {
                if matches!(kind, PickerKind::Save) {
                    dialog.set_current_name(name);
                } else {
                    dialog.set_filename(path);
                }
            } else if !dialog.set_filename(path) {
                return Err("native dialog cannot select the requested filename".into());
            }
        }
    }
    for filter in filters {
        let native = gtk::FileFilter::new();
        native.set_name(Some(&filter.name));
        for extension in &filter.extensions {
            native.add_pattern(&format!("*.{extension}"));
        }
        dialog.add_filter(native);
    }
    Ok(dialog)
}

fn selected_paths(dialog: &gtk::FileChooserNative, kind: PickerKind) -> Vec<PathBuf> {
    match kind {
        PickerKind::Open { multiple: true, .. } => dialog.filenames(),
        _ => dialog.filename().into_iter().collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::ffi::OsStrExt;

    #[test]
    #[ignore = "requires GTK display; qualified under xvfb in Linux CI"]
    fn native_gtk_selections_preserve_original_names() {
        gtk::init().expect("GTK display is required for native dialog qualification");
        let temp = tempfile::tempdir().unwrap();
        let directory = temp
            .path()
            .join(std::ffi::OsStr::from_bytes(b"directory-\xfe"));
        std::fs::create_dir(&directory).unwrap();
        let file = directory.join(std::ffi::OsStr::from_bytes(b"clip-\xff.mp4"));
        let shadow = directory.join("clip-\u{fffd}.mp4");
        std::fs::write(&file, b"original").unwrap();
        std::fs::write(&shadow, b"shadow").unwrap();

        let kind = PickerKind::Open {
            directory: false,
            multiple: true,
        };
        let chooser = build_dialog(
            None,
            kind,
            Some("Native path test"),
            Some(&directory),
            true,
            &[],
            None,
        )
        .unwrap();
        chooser.show();
        assert!(chooser.select_filename(&file));
        assert!(chooser.select_filename(&shadow));
        let expected: std::collections::HashSet<_> = [file.clone(), shadow].into_iter().collect();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let context = gtk::glib::MainContext::default();
        loop {
            while context.pending() {
                context.iteration(false);
            }
            let paths: std::collections::HashSet<_> =
                selected_paths(&chooser, kind).into_iter().collect();
            if paths == expected {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "native GTK selection mismatch: {paths:?}"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        chooser.destroy();

        for kind in [
            PickerKind::Open {
                directory: false,
                multiple: false,
            },
            PickerKind::Save,
        ] {
            let chooser = build_dialog(None, kind, None, Some(&file), false, &[], None).unwrap();
            assert_eq!(selected_paths(&chooser, kind), vec![file.clone()]);
            chooser.destroy();
        }
        let kind = PickerKind::Open {
            directory: true,
            multiple: false,
        };
        let chooser = build_dialog(None, kind, None, Some(&directory), true, &[], None).unwrap();
        assert_eq!(selected_paths(&chooser, kind), vec![directory]);
        chooser.destroy();
    }

    #[test]
    fn gtk_file_objects_preserve_original_bytes_and_distinct_aliases() {
        let path = Path::new("/tmp").join(std::ffi::OsStr::from_bytes(b"clip-\xff.mp4"));
        let file = gtk::gio::File::for_path(&path);
        assert_eq!(file.path().unwrap(), path);
        assert!(file.uri().ends_with("clip-%FF.mp4"));
        assert_ne!(
            file.path().unwrap(),
            PathBuf::from(path.to_string_lossy().as_ref())
        );
    }
}
