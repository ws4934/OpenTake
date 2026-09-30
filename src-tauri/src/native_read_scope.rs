//! Native grants supplement Tauri's Unicode-only glob scope. Raw file names
//! never enter `Scope::allow_file`, which would also grant their lossy alias.
//! Directory grants retain the native root so non-Unicode descendants can be
//! authorized without matching a replacement-character spelling.

use std::collections::HashSet;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

use opentake_domain::native_path::identity_key;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager, Runtime};

const MAX_SCOPE_BYTES: u64 = 8 * 1024 * 1024;
const MAX_GRANTS: usize = 10_000;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
enum Grant {
    File {
        #[serde(with = "opentake_domain::native_path::path")]
        path: PathBuf,
    },
    Directory {
        #[serde(with = "opentake_domain::native_path::path")]
        path: PathBuf,
        recursive: bool,
    },
    DenyFile {
        #[serde(with = "opentake_domain::native_path::path")]
        path: PathBuf,
    },
}

impl Grant {
    fn path(&self) -> &Path {
        match self {
            Self::File { path } | Self::Directory { path, .. } | Self::DenyFile { path } => path,
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    version: u32,
    grants: Vec<Grant>,
}

#[derive(Default)]
pub(crate) struct NativeScopeSnapshot {
    files: HashSet<PathBuf>,
    directories: Vec<(PathBuf, bool)>,
    denied: HashSet<PathBuf>,
}

impl NativeScopeSnapshot {
    fn from_grants(grants: &[Grant]) -> Self {
        let mut snapshot = Self::default();
        for grant in grants {
            let key = identity_key(grant.path());
            match grant {
                Grant::File { .. } => {
                    snapshot.files.insert(key);
                }
                Grant::Directory { recursive, .. } => snapshot.directories.push((key, *recursive)),
                Grant::DenyFile { .. } => {
                    snapshot.denied.insert(key);
                }
            }
        }
        snapshot
    }

    pub(crate) fn forbids(&self, path: &Path) -> bool {
        self.denied.contains(&identity_key(path))
    }

    pub(crate) fn has_file(&self, path: &Path) -> bool {
        self.files.contains(&identity_key(path))
    }

    pub(crate) fn allows(&self, path: &Path) -> bool {
        if path.to_str().is_some() || self.forbids(path) {
            return false;
        }
        let key = identity_key(path);
        self.files.contains(&key)
            || self.directories.iter().any(|(root, recursive)| {
                let Ok(relative) = key.strip_prefix(root) else {
                    return false;
                };
                let components: Vec<_> = relative.components().collect();
                (*recursive || components.len() <= 1)
                    && components.iter().all(|component| {
                        matches!(component, Component::Normal(_))
                            && (cfg!(windows)
                                || component.as_os_str().as_encoded_bytes().first() != Some(&b'.'))
                    })
            })
    }
}

struct ScopeState {
    grants: Vec<Grant>,
    snapshot: Arc<NativeScopeSnapshot>,
}

pub(crate) struct NativeReadScope {
    ledger: Option<PathBuf>,
    state: Mutex<ScopeState>,
}

impl NativeReadScope {
    pub(crate) fn load(path: PathBuf) -> Result<Self, String> {
        let grants = match std::fs::File::open(&path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(error) => return Err(format!("cannot read native file grants: {error}")),
            Ok(file) => {
                let mut bytes = Vec::new();
                file.take(MAX_SCOPE_BYTES + 1)
                    .read_to_end(&mut bytes)
                    .map_err(|error| format!("cannot read native file grants: {error}"))?;
                if bytes.len() as u64 > MAX_SCOPE_BYTES {
                    return Err("native file grants exceed the supported size".into());
                }
                let document: Document = serde_json::from_slice(&bytes)
                    .map_err(|error| format!("native file grants are invalid: {error}"))?;
                if document.version != 1 || document.grants.len() > MAX_GRANTS {
                    return Err("native file grants use an unsupported format or size".into());
                }
                for grant in &document.grants {
                    validate_path(grant.path())?;
                }
                document.grants
            }
        };
        Ok(Self::new(Some(path), grants))
    }

    fn new(ledger: Option<PathBuf>, grants: Vec<Grant>) -> Self {
        let snapshot = Arc::new(NativeScopeSnapshot::from_grants(&grants));
        Self {
            ledger,
            state: Mutex::new(ScopeState { grants, snapshot }),
        }
    }

    fn issue(&self, grant: Grant) -> Result<(), String> {
        self.issue_many(std::iter::once(grant))
    }

    fn issue_many(&self, incoming: impl IntoIterator<Item = Grant>) -> Result<(), String> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| "native file grants lock failed")?;
        let mut seen: HashSet<_> = state.grants.iter().cloned().collect();
        let mut grants = state.grants.clone();
        for grant in incoming {
            validate_path(grant.path())?;
            if seen.insert(grant.clone()) {
                grants.push(grant);
            }
        }
        if grants.len() == state.grants.len() {
            return Ok(());
        }
        if grants.len() > MAX_GRANTS {
            return Err("too many native file grants".into());
        }
        if let Some(path) = &self.ledger {
            let document = Document {
                version: 1,
                grants: grants.clone(),
            };
            let bytes = serde_json::to_vec(&document).map_err(|error| error.to_string())?;
            if bytes.len() as u64 > MAX_SCOPE_BYTES {
                return Err("native file grants exceed the supported size".into());
            }
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
            }
            crate::dialog_output::write_file_atomically(
                &crate::dialog_output::DialogOutput {
                    path: path.clone(),
                    appended: false,
                },
                &bytes,
            )?;
        }
        state.snapshot = Arc::new(NativeScopeSnapshot::from_grants(&grants));
        state.grants = grants;
        Ok(())
    }

    fn snapshot(&self) -> Arc<NativeScopeSnapshot> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .snapshot
            .clone()
    }
}

fn validate_path(path: &Path) -> Result<(), String> {
    if !path.is_absolute()
        || path.as_os_str().as_encoded_bytes().contains(&0)
        || path
            .components()
            .any(|part| matches!(part, Component::CurDir | Component::ParentDir))
    {
        return Err("native file grant must use an absolute path without relative segments".into());
    }
    Ok(())
}

pub(crate) fn snapshot<R: Runtime>(app: &AppHandle<R>) -> Arc<NativeScopeSnapshot> {
    app.try_state::<NativeReadScope>().map_or_else(
        || Arc::new(NativeScopeSnapshot::default()),
        |scope| scope.snapshot(),
    )
}

pub(crate) fn allow_file<R: Runtime>(app: &AppHandle<R>, path: &Path) -> Result<(), String> {
    if path.to_str().is_some() {
        return app
            .asset_protocol_scope()
            .allow_file(path)
            .map_err(|error| error.to_string());
    }
    app.try_state::<NativeReadScope>()
        .ok_or("native file grants are unavailable")?
        .issue(Grant::File {
            path: path.to_path_buf(),
        })
}

/// One dialog selection persists all native results in one publication.
pub(crate) fn allow_selections<R: Runtime>(
    app: &AppHandle<R>,
    paths: &[PathBuf],
    directory: bool,
    recursive: bool,
) -> Result<(), String> {
    for path in paths {
        validate_path(path)?;
    }
    let native = paths
        .iter()
        .filter(|path| path.to_str().is_none())
        .map(|path| {
            if directory {
                Grant::Directory {
                    path: path.clone(),
                    recursive,
                }
            } else {
                Grant::File { path: path.clone() }
            }
        })
        .collect::<Vec<_>>();
    if !native.is_empty() {
        app.try_state::<NativeReadScope>()
            .ok_or("native file grants are unavailable")?
            .issue_many(native)?;
    }
    for path in paths.iter().filter(|path| path.to_str().is_some()) {
        if directory {
            app.asset_protocol_scope().allow_directory(path, recursive)
        } else {
            app.asset_protocol_scope().allow_file(path)
        }
        .map_err(|error| error.to_string())?;
    }
    Ok(())
}

pub(crate) fn allow_directory<R: Runtime>(
    app: &AppHandle<R>,
    path: &Path,
    recursive: bool,
) -> Result<(), String> {
    if path.to_str().is_some() {
        return app
            .asset_protocol_scope()
            .allow_directory(path, recursive)
            .map_err(|error| error.to_string());
    }
    app.try_state::<NativeReadScope>()
        .ok_or("native file grants are unavailable")?
        .issue(Grant::Directory {
            path: path.to_path_buf(),
            recursive,
        })
}

pub(crate) fn forbid_file<R: Runtime>(app: &AppHandle<R>, path: &Path) -> Result<(), String> {
    if path.to_str().is_some() {
        let scope = app.asset_protocol_scope();
        scope.forbid_file(path).map_err(|error| error.to_string())?;
        // Persisted-scope saves on PathAllowed; deny precedence is retained.
        return scope.allow_file(path).map_err(|error| error.to_string());
    }
    app.try_state::<NativeReadScope>()
        .ok_or("native file grants are unavailable")?
        .issue(Grant::DenyFile {
            path: path.to_path_buf(),
        })
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::ffi::OsStrExt;

    #[test]
    fn original_names_survive_restart_without_authorizing_lossy_aliases() {
        let temp = tempfile::tempdir().unwrap();
        let ledger = temp.path().join("grants.json");
        let raw = temp
            .path()
            .join(std::ffi::OsStr::from_bytes(b"clip-\xff.png"));
        let shadow = PathBuf::from(raw.to_string_lossy().as_ref());
        let scope = NativeReadScope::load(ledger.clone()).unwrap();
        scope.issue(Grant::File { path: raw.clone() }).unwrap();
        let reopened = NativeReadScope::load(ledger.clone()).unwrap();
        assert!(reopened.snapshot().allows(&raw));
        assert!(!reopened.snapshot().allows(&shadow));
        reopened
            .issue(Grant::DenyFile { path: raw.clone() })
            .unwrap();
        assert!(!NativeReadScope::load(ledger)
            .unwrap()
            .snapshot()
            .allows(&raw));
    }

    #[test]
    fn directory_grants_preserve_depth_and_hidden_file_boundaries() {
        let raw = Path::new("/approved").join(std::ffi::OsStr::from_bytes(b"clip-\xff.png"));
        let scope = NativeReadScope::new(
            None,
            vec![Grant::Directory {
                path: "/approved".into(),
                recursive: false,
            }],
        );
        assert!(scope.snapshot().allows(&raw));
        assert!(!scope
            .snapshot()
            .allows(&Path::new("/other").join(raw.file_name().unwrap())));
        assert!(!scope
            .snapshot()
            .allows(&Path::new("/approved/nested").join(raw.file_name().unwrap())));
        assert!(!scope
            .snapshot()
            .allows(&Path::new("/approved/.hidden").join(raw.file_name().unwrap())));
    }

    #[test]
    fn corrupt_grants_remain_explicit_and_untouched() {
        let temp = tempfile::tempdir().unwrap();
        let ledger = temp.path().join("grants.json");
        std::fs::write(&ledger, b"broken").unwrap();
        assert!(NativeReadScope::load(ledger.clone()).is_err());
        assert_eq!(std::fs::read(ledger).unwrap(), b"broken");
    }
}
