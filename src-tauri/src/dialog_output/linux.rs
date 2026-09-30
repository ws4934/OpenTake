//! GTK's filename APIs preserve Unix bytes. RFD's GTK backend converts those
//! C strings to UTF-8 and discards filenames that fail that conversion.

use std::path::{Path, PathBuf};

use gtk::prelude::*;

use super::{PickerKind, SaveDialogFilter};

struct Picker {
    dialog: gtk::FileChooserNative,
    name: Option<gtk::Entry>,
}

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
    let chosen = on_main_thread(window, move |parent| {
        let picker = build_dialog(
            Some(parent),
            kind,
            title.as_deref(),
            default_path.as_deref(),
            default_is_directory,
            &filters,
            can_create_directories,
        )?;
        let response = picker.dialog.run();
        let result = if response == gtk::ResponseType::Accept {
            selected_paths(&picker, kind).and_then(|paths| {
                if paths.is_empty() {
                    Err("native file dialog returned no filesystem selection".into())
                } else {
                    Ok(Some(paths))
                }
            })
        } else {
            Ok(None)
        };
        picker.dialog.destroy();
        result
    })?;
    if matches!(kind, PickerKind::Save) {
        if let Some(path) = chosen.as_ref().and_then(|paths| paths.first()) {
            // Inspect the actual byte-preserving target off the GTK thread.
            // GTK's SAVE action confirms a Unicode display-name alias instead.
            match std::fs::symlink_metadata(path) {
                Ok(metadata) if metadata.is_dir() => {
                    return Err("the selected output is a directory".into());
                }
                Ok(_) => {
                    let name = super::native_name::display(path.as_os_str());
                    if !on_main_thread(window, move |parent| {
                        let dialog = overwrite_dialog(Some(parent), &name);
                        let response = dialog.run();
                        dialog.close();
                        Ok(response == gtk::ResponseType::Accept)
                    })? {
                        return Ok(None);
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(format!("cannot inspect the selected output: {error}")),
            }
        }
    }
    Ok(chosen)
}

fn on_main_thread<T: Send + 'static>(
    window: &tauri::Window,
    task: impl FnOnce(&gtk::Window) -> Result<T, String> + Send + 'static,
) -> Result<T, String> {
    let (send, receive) = std::sync::mpsc::sync_channel(1);
    let parent = window.clone();
    window
        .run_on_main_thread(move || {
            let result = parent
                .gtk_window()
                .map_err(|error| error.to_string())
                .and_then(|parent| task(parent.upcast_ref()));
            // A disconnected receiver means the command stopped waiting; no
            // returned selection is authorized in that case.
            let _ = send.send(result);
        })
        .map_err(|error| format!("cannot show native file dialog: {error}"))?;
    receive
        .recv()
        .map_err(|_| "native file dialog did not return a result".to_string())?
}

fn overwrite_dialog(parent: Option<&gtk::Window>, name: &str) -> gtk::MessageDialog {
    let dialog = gtk::MessageDialog::new(
        parent,
        gtk::DialogFlags::MODAL,
        gtk::MessageType::Question,
        gtk::ButtonsType::None,
        &format!("Replace the existing file?\n{name}"),
    );
    dialog.add_buttons(&[
        ("_Cancel", gtk::ResponseType::Cancel),
        ("_Replace", gtk::ResponseType::Accept),
    ]);
    dialog.set_default_response(gtk::ResponseType::Cancel);
    dialog
}

fn build_dialog(
    parent: Option<&gtk::Window>,
    kind: PickerKind,
    title: Option<&str>,
    default_path: Option<&Path>,
    default_is_directory: bool,
    filters: &[SaveDialogFilter],
    can_create_directories: Option<bool>,
) -> Result<Picker, String> {
    let action = match kind {
        PickerKind::Open {
            directory: true, ..
        } => gtk::FileChooserAction::SelectFolder,
        PickerKind::Open { .. } => gtk::FileChooserAction::Open,
        // GTK SAVE builds a target from its UTF-8 display name, even when the
        // selected GFile has original bytes. Choose a native directory and an
        // explicitly escaped filename in one dialog instead.
        PickerKind::Save => gtk::FileChooserAction::SelectFolder,
    };
    let dialog = gtk::FileChooserNative::new(title, parent, action, None, None);
    dialog.set_modal(true);
    dialog.set_local_only(true);
    dialog.set_select_multiple(matches!(kind, PickerKind::Open { multiple: true, .. }));
    if let Some(can) = can_create_directories {
        dialog.set_create_folders(can);
    }
    let name = if matches!(kind, PickerKind::Save) {
        let entry = gtk::Entry::new();
        let suggested = default_path
            .filter(|_| !default_is_directory)
            .and_then(Path::file_name)
            .unwrap_or_else(|| std::ffi::OsStr::new("Untitled"));
        entry.set_text(&super::native_name::display(suggested));
        entry.set_width_chars(40);
        entry.set_tooltip_text(Some(
            "Use \\xNN for filename bytes and \\\\ for a literal backslash.",
        ));
        let fields = gtk::Box::new(gtk::Orientation::Vertical, 6);
        let label = gtk::Label::new(Some("File name"));
        label.set_xalign(0.0);
        fields.pack_start(&label, false, false, 0);
        fields.pack_start(&entry, false, false, 0);
        fields.show_all();
        dialog.set_extra_widget(&fields);
        dialog.set_accept_label(Some("_Save"));
        Some(entry)
    } else {
        None
    };
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
            if !matches!(kind, PickerKind::Save) && !dialog.set_filename(path) {
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
    Ok(Picker { dialog, name })
}

fn selected_paths(picker: &Picker, kind: PickerKind) -> Result<Vec<PathBuf>, String> {
    if let Some(name) = &picker.name {
        let name = super::native_name::parse(name.text().as_str())?;
        return Ok(picker
            .dialog
            .filename()
            .map(|directory| directory.join(name))
            .into_iter()
            .collect());
    }
    Ok(match kind {
        PickerKind::Open { multiple: true, .. } => picker.dialog.filenames(),
        _ => picker.dialog.filename().into_iter().collect(),
    })
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
        chooser.dialog.show();
        assert!(chooser.dialog.select_filename(&file));
        wait_for_paths(&chooser, kind, [file.clone()].into_iter().collect());
        // GTK's programmatic select_filename moves the cursor and replaces
        // the current selection. Select all on the visible fallback dialog,
        // as a user would, to exercise the native multiple-result getter.
        let visible = gtk::Window::list_toplevels()
            .into_iter()
            .filter_map(|window| window.downcast::<gtk::FileChooserDialog>().ok())
            .find(|dialog| dialog.is_visible())
            .expect("visible GTK file chooser");
        visible.select_all();
        let expected: std::collections::HashSet<_> = [file.clone(), shadow].into_iter().collect();
        wait_for_paths(&chooser, kind, expected);
        chooser.dialog.destroy();

        for kind in [
            PickerKind::Open {
                directory: false,
                multiple: false,
            },
            PickerKind::Save,
        ] {
            let chooser = build_dialog(None, kind, None, Some(&file), false, &[], None).unwrap();
            chooser.dialog.show();
            wait_for_paths(&chooser, kind, [file.clone()].into_iter().collect());
            chooser.dialog.destroy();
        }
        let proposed = directory.join(std::ffi::OsStr::from_bytes(b"new-\xfd.mp4"));
        let chooser = build_dialog(
            None,
            PickerKind::Save,
            None,
            Some(&proposed),
            false,
            &[],
            None,
        )
        .unwrap();
        chooser.dialog.show();
        wait_for_paths(&chooser, PickerKind::Save, [proposed].into_iter().collect());
        let entry = chooser.name.as_ref().unwrap();
        entry.set_text("clip-�.mp4");
        assert_eq!(
            selected_paths(&chooser, PickerKind::Save).unwrap(),
            vec![directory.join("clip-�.mp4")]
        );
        entry.set_text("literal-\\\\xff.mp4");
        assert_eq!(
            selected_paths(&chooser, PickerKind::Save).unwrap(),
            vec![directory.join("literal-\\xff.mp4")]
        );
        entry.set_text("\\x2f");
        assert!(selected_paths(&chooser, PickerKind::Save).is_err());
        chooser.dialog.destroy();

        for response in [gtk::ResponseType::Cancel, gtk::ResponseType::Accept] {
            let dialog =
                overwrite_dialog(None, &super::super::native_name::display(file.as_os_str()));
            let responder = dialog.clone();
            gtk::glib::idle_add_local_once(move || responder.response(response));
            assert_eq!(dialog.run(), response);
            dialog.close();
        }
        let kind = PickerKind::Open {
            directory: true,
            multiple: false,
        };
        let chooser = build_dialog(None, kind, None, Some(&directory), true, &[], None).unwrap();
        chooser.dialog.show();
        wait_for_paths(&chooser, kind, [directory].into_iter().collect());
        chooser.dialog.destroy();
    }

    fn wait_for_paths(
        chooser: &Picker,
        kind: PickerKind,
        expected: std::collections::HashSet<PathBuf>,
    ) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let context = gtk::glib::MainContext::default();
        loop {
            // A busy GTK source must not prevent the timeout from being checked.
            for _ in 0..32 {
                if !context.pending() {
                    break;
                }
                context.iteration(false);
            }
            let paths: std::collections::HashSet<_> =
                selected_paths(chooser, kind).unwrap().into_iter().collect();
            if paths == expected {
                return;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "native GTK selection mismatch: {paths:?}, expected {expected:?}"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
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
