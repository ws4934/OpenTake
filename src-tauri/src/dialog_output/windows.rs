//! IFileDialog paths stay UTF-16. RFD's Windows result conversion requires
//! valid Unicode and panics for unpaired surrogates.

use std::ffi::{OsStr, OsString};
use std::os::windows::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};

use ::windows::core::{Interface, PCWSTR};
use ::windows::Win32::Foundation::HWND;
use ::windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CoTaskMemFree, CoUninitialize, CLSCTX_INPROC_SERVER,
    COINIT_APARTMENTTHREADED,
};
use ::windows::Win32::UI::Shell::{
    Common::COMDLG_FILTERSPEC, FileOpenDialog, FileSaveDialog, IFileDialog, IFileOpenDialog,
    IFileSaveDialog, IShellItem, SHCreateItemFromParsingName, FOS_ALLOWMULTISELECT,
    FOS_DONTADDTORECENT, FOS_FILEMUSTEXIST, FOS_FORCEFILESYSTEM, FOS_NOCHANGEDIR,
    FOS_OVERWRITEPROMPT, FOS_PATHMUSTEXIST, FOS_PICKFOLDERS, SIGDN_FILESYSPATH,
};

use super::{PickerKind as Kind, SaveDialogFilter};

struct Apartment;
impl Apartment {
    fn enter() -> ::windows::core::Result<Self> {
        // SAFETY: this dedicated dialog thread has no prior COM apartment.
        unsafe {
            CoInitializeEx(None, COINIT_APARTMENTTHREADED).ok()?;
        }
        Ok(Self)
    }
}
impl Drop for Apartment {
    fn drop(&mut self) {
        // SAFETY: one successful CoInitializeEx is balanced on its own thread,
        // after all COM dialog objects have been dropped.
        unsafe {
            CoUninitialize();
        }
    }
}

pub(super) fn pick(
    window: &tauri::Window,
    kind: Kind,
    title: Option<String>,
    default_path: Option<PathBuf>,
    filters: Vec<SaveDialogFilter>,
) -> Result<Option<Vec<PathBuf>>, String> {
    let owner = window.hwnd().map_err(|error| error.to_string())?.0 as usize;
    // Tokio blocking workers may already use MTA. Each dialog gets one STA
    // thread, which lives until the native dialog closes and is then joined.
    std::thread::Builder::new()
        .name("native-file-dialog".into())
        .spawn(move || {
            run(owner, kind, title, default_path, filters)
                .map_err(|error| format!("native file dialog failed: {error}"))
        })
        .map_err(|error| format!("cannot start native file dialog: {error}"))?
        .join()
        .map_err(|_| "native file dialog worker panicked".to_string())?
}

fn wide(text: &OsStr) -> Vec<u16> {
    text.encode_wide().chain(std::iter::once(0)).collect()
}

fn shell_item(path: &Path) -> ::windows::core::Result<IShellItem> {
    let path = wide(path.as_os_str());
    // SAFETY: the native path is NUL-terminated for the call. The shell item
    // owns the returned shell item independently of this buffer.
    unsafe { SHCreateItemFromParsingName(PCWSTR(path.as_ptr()), None) }
}

fn shell_path(item: &IShellItem) -> ::windows::core::Result<PathBuf> {
    // SAFETY: GetDisplayName allocates a NUL-terminated UTF-16 buffer with the
    // COM task allocator. Copy its units, then release that buffer exactly once.
    unsafe {
        let pointer = item.GetDisplayName(SIGDN_FILESYSPATH)?;
        let mut length = 0;
        while *pointer.0.add(length) != 0 {
            length += 1;
        }
        let path = PathBuf::from(OsString::from_wide(std::slice::from_raw_parts(
            pointer.0, length,
        )));
        CoTaskMemFree(Some(pointer.0.cast()));
        Ok(path)
    }
}

fn run(
    owner: usize,
    kind: Kind,
    title: Option<String>,
    default_path: Option<PathBuf>,
    filters: Vec<SaveDialogFilter>,
) -> ::windows::core::Result<Option<Vec<PathBuf>>> {
    let _apartment = Apartment::enter()?;
    // SAFETY: all interfaces and input buffers stay on this initialized STA.
    // IFileDialog copies dialog settings; filter buffers also remain live until
    // Show returns. The owner handle belongs to the calling Tauri window.
    unsafe {
        let open: Option<IFileOpenDialog> = match kind {
            Kind::Open { .. } => Some(CoCreateInstance(
                &FileOpenDialog,
                None,
                CLSCTX_INPROC_SERVER,
            )?),
            Kind::Save => None,
        };
        let dialog: IFileDialog = match &open {
            Some(open) => open.cast()?,
            None => {
                let save: IFileSaveDialog =
                    CoCreateInstance(&FileSaveDialog, None, CLSCTX_INPROC_SERVER)?;
                save.cast()?
            }
        };
        let mut options = dialog.GetOptions()?
            | FOS_FORCEFILESYSTEM
            | FOS_NOCHANGEDIR
            | FOS_DONTADDTORECENT
            | FOS_PATHMUSTEXIST;
        match kind {
            Kind::Open {
                directory,
                multiple,
            } => {
                options |= FOS_FILEMUSTEXIST;
                if directory {
                    options |= FOS_PICKFOLDERS;
                }
                if multiple {
                    options |= FOS_ALLOWMULTISELECT;
                }
            }
            Kind::Save => options |= FOS_OVERWRITEPROMPT,
        }
        dialog.SetOptions(options)?;
        if let Some(title) = title {
            let title = wide(OsStr::new(&title));
            dialog.SetTitle(PCWSTR(title.as_ptr()))?;
        }
        if let Some(path) = default_path.filter(|path| !path.as_os_str().is_empty()) {
            if path.is_dir() {
                dialog.SetFolder(&shell_item(&path)?)?;
            } else {
                if let Some(parent) = path
                    .parent()
                    .filter(|parent| !parent.as_os_str().is_empty())
                {
                    dialog.SetFolder(&shell_item(parent)?)?;
                }
                if let Some(name) = path.file_name() {
                    let name = wide(name);
                    dialog.SetFileName(PCWSTR(name.as_ptr()))?;
                }
            }
        }
        let filter_texts: Vec<_> = filters
            .iter()
            .map(|filter| {
                let pattern = filter
                    .extensions
                    .iter()
                    .map(|extension| {
                        if extension == "*" {
                            "*.*".into()
                        } else {
                            format!("*.{extension}")
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(";");
                (wide(OsStr::new(&filter.name)), wide(OsStr::new(&pattern)))
            })
            .collect();
        let specs: Vec<_> = filter_texts
            .iter()
            .map(|(name, pattern)| COMDLG_FILTERSPEC {
                pszName: PCWSTR(name.as_ptr()),
                pszSpec: PCWSTR(pattern.as_ptr()),
            })
            .collect();
        if !specs.is_empty()
            && !matches!(
                kind,
                Kind::Open {
                    directory: true,
                    ..
                }
            )
        {
            dialog.SetFileTypes(&specs)?;
        }
        if let Err(error) = dialog.Show(Some(HWND(owner as *mut _))) {
            if error.code().0 as u32 == 0x800704c7 {
                return Ok(None);
            }
            return Err(error);
        }
        let paths = if matches!(kind, Kind::Open { multiple: true, .. }) {
            let items = open
                .as_ref()
                .expect("multiple selections use IFileOpenDialog")
                .GetResults()?;
            (0..items.GetCount()?)
                .map(|index| shell_path(&items.GetItemAt(index)?))
                .collect::<::windows::core::Result<Vec<_>>>()?
        } else {
            vec![shell_path(&dialog.GetResult()?)?]
        };
        Ok(Some(paths))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_item_results_preserve_unpaired_utf16_units() {
        let _apartment = Apartment::enter().unwrap();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join(OsString::from_wide(&[
            99, 108, 105, 112, 45, 0xd800, 46, 112, 110, 103,
        ]));
        std::fs::write(&path, b"native name").unwrap();
        let returned = shell_path(&shell_item(&path).unwrap()).unwrap();
        assert_eq!(returned.file_name(), path.file_name());
        assert_eq!(std::fs::read(&returned).unwrap(), b"native name");
        assert_eq!(
            std::fs::canonicalize(&returned).unwrap(),
            std::fs::canonicalize(&path).unwrap()
        );
        assert_ne!(returned, PathBuf::from(path.to_string_lossy().as_ref()));
    }
}
