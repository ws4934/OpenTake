//! Bounded, off-main-thread delivery of user-approved local media.
//!
//! Tauri's built-in asset protocol opens the path again after its scope check.
//! A File Provider update or hostile local replacement can therefore turn a
//! previously regular file into a FIFO, symlink, device, or cloud placeholder
//! and block the WebView/AppKit thread. This protocol authorizes requests from
//! in-memory state only and leaves every file operation to a pool of isolated
//! helper processes: a helper opens with no-recall/non-blocking platform flags
//! and reports the retained handle's final path and identity, the parent
//! authorizes that identity before any byte is read, and only bounded bodies
//! are served.

use opentake_domain::NativePath;
use std::collections::HashSet;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use http_range::HttpRange;
use opentake_core::{AppCore, MediaAuthorityRevision, ProjectAssetAuthority};
use opentake_project::{ProjectRoot, ProjectRootIdentity};
use percent_encoding::percent_decode;
use serde::{Deserialize, Serialize};
use tauri::http::header::{
    ACCEPT_RANGES, ACCESS_CONTROL_ALLOW_HEADERS, ACCESS_CONTROL_ALLOW_METHODS,
    ACCESS_CONTROL_ALLOW_ORIGIN, ACCESS_CONTROL_EXPOSE_HEADERS, CONTENT_LENGTH, CONTENT_RANGE,
    CONTENT_TYPE, ETAG, IF_RANGE, RANGE, RETRY_AFTER,
};
use tauri::http::{Method, Request, Response, StatusCode};
use tauri::scope::fs::Scope;
use tauri::{AppHandle, Manager, Runtime, UriSchemeResponder};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tokio::sync::Semaphore;

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
#[cfg(target_os = "macos")]
use std::os::{fd::AsRawFd, macos::fs::MetadataExt};

const MAX_CONCURRENT_READS: usize = 4;
/// Requests admitted to wait for a worker. A burst beyond the workers (a media
/// panel full of thumbnails while a video streams) waits instead of failing.
const MAX_QUEUED_READS: usize = 256;
/// How long an admitted request may wait for a worker before a 504.
const QUEUE_DEADLINE: Duration = Duration::from_secs(20);
const MAX_PATH_BYTES: usize = 32_768;
const MAX_FULL_BODY_BYTES: u64 = 32 * 1024 * 1024;
const MAX_FULL_IMAGE_BODY_BYTES: u64 = 128 * 1024 * 1024;
/// Largest body served for one Range request. Media elements stream with
/// open-ended ranges, so this sets the request rate while playing: at 4 MiB a
/// 50 Mbit/s source needs about 1.5 requests per second. Memory budget: at
/// most `MAX_CONCURRENT_READS` range bodies are in flight, each held once by
/// the helper and once by the parent, so 4 x 4 MiB x 2 = 32 MiB.
const MAX_RANGE_BYTES: u64 = 4 * 1024 * 1024;
const IO_DEADLINE: Duration = Duration::from_secs(5);
const REAP_DEADLINE: Duration = Duration::from_secs(1);
const MAX_HELPER_REQUEST_BYTES: usize = 256 * 1024;
const MAX_HELPER_METADATA_BYTES: usize = 256 * 1024;
const MAX_HELPER_BODY_BYTES: usize = MAX_FULL_IMAGE_BODY_BYTES as usize;
const HELPER_ARG: &str = "--opentake-internal-safe-asset-helper-v1";
const HELPER_TOKEN_ENV: &str = "OPENTAKE_INTERNAL_ASSET_TOKEN";
const HELPER_PARENT_ENV: &str = "OPENTAKE_INTERNAL_ASSET_PARENT_PID";

mod helper;
mod pool;
mod scope;

pub(crate) use helper::run_helper_if_requested;
use helper::{
    authorize_opened_asset, isolated_response_to_http, HelperProjectAuthority, HelperRequest,
    IsolatedHelperError, IsolatedResponse,
};
pub(crate) use pool::{retire_helper_pool, shutdown_helper_pool};
use pool::{HelperOutcome, HelperPool};
#[cfg(test)]
pub(crate) use scope::asset_scope_snapshot_captures;
pub(crate) use scope::{asset_scope_snapshot, ScopeSnapshot};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScopeOnlyKind {
    HomeThumbnail,
    ApplicationOwned,
}

/// In-memory authorization of a requested path outside the current bundle.
/// It is derived from the cached scope snapshot and external media index, and
/// re-derived after the helper opened the file and again before publication.
#[derive(Clone, Debug, Eq, PartialEq)]
enum NonProjectAssetAuthority {
    ScopeOnly {
        kind: ScopeOnlyKind,
        requested_path: PathBuf,
    },
    ProjectMedia {
        project_epoch: u64,
        requested_path: PathBuf,
    },
}

impl NonProjectAssetAuthority {
    fn requested_path(&self) -> &Path {
        match self {
            Self::ScopeOnly { requested_path, .. } | Self::ProjectMedia { requested_path, .. } => {
                requested_path
            }
        }
    }
}

#[cfg(all(test, unix))]
use helper::{actual_parent_process_id, parent_is_same_executable};
#[cfg(test)]
use helper::{
    open_helper_asset, opened_metadata, serve_helper_asset, OpenedMetadata, WireIoErrorKind,
};
#[cfg(test)]
use pool::bounded_reap;
#[cfg(all(test, unix))]
use pool::terminate_or_quarantine;

#[derive(Clone)]
pub(crate) struct SafeAssetProtocol {
    worker_permits: Arc<Semaphore>,
    queued_permits: Arc<Semaphore>,
    pool: Arc<HelperPool>,
}

impl Default for SafeAssetProtocol {
    fn default() -> Self {
        Self::with_pool(HelperPool::shared())
    }
}

impl SafeAssetProtocol {
    fn with_pool(pool: Arc<HelperPool>) -> Self {
        Self {
            worker_permits: Arc::new(Semaphore::new(MAX_CONCURRENT_READS)),
            queued_permits: Arc::new(Semaphore::new(MAX_QUEUED_READS)),
            pool,
        }
    }

    pub(crate) fn respond<R: Runtime>(
        &self,
        app: AppHandle<R>,
        request: Request<Vec<u8>>,
        responder: UriSchemeResponder,
    ) {
        let protocol = self.clone();
        tauri::async_runtime::spawn(async move {
            responder.respond(protocol.serve(&app, request).await);
        });
    }

    /// Admission: up to `MAX_QUEUED_READS` requests wait (bounded by
    /// `QUEUE_DEADLINE`) for one of `MAX_CONCURRENT_READS` workers.
    async fn serve<R: Runtime>(
        &self,
        app: &AppHandle<R>,
        request: Request<Vec<u8>>,
    ) -> Response<Vec<u8>> {
        let Ok(_queued_permit) = self.queued_permits.clone().try_acquire_owned() else {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "local asset workers are busy",
                Some((RETRY_AFTER, "1")),
            );
        };
        let _worker_permit =
            match tokio::time::timeout(QUEUE_DEADLINE, self.worker_permits.clone().acquire_owned())
                .await
            {
                Ok(Ok(permit)) => permit,
                Ok(Err(_)) => {
                    return error_response(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "local asset service is shutting down",
                        None,
                    );
                }
                Err(_) => {
                    return error_response(
                        StatusCode::GATEWAY_TIMEOUT,
                        "local asset worker queue timed out",
                        Some((RETRY_AFTER, "1")),
                    );
                }
            };
        response_for_request(app, request, &self.pool).await
    }
}

async fn response_for_request<R: Runtime>(
    app: &AppHandle<R>,
    request: Request<Vec<u8>>,
    pool: &Arc<HelperPool>,
) -> Response<Vec<u8>> {
    if request.method() == Method::OPTIONS {
        return secure_response_builder(StatusCode::NO_CONTENT)
            .header(ACCESS_CONTROL_ALLOW_METHODS, "GET, HEAD, OPTIONS")
            .header(ACCESS_CONTROL_ALLOW_HEADERS, "Range, If-Range")
            .body(Vec::new())
            .expect("static response headers");
    }
    if request.method() != Method::GET && request.method() != Method::HEAD {
        return error_response(StatusCode::METHOD_NOT_ALLOWED, "GET and HEAD only", None);
    }
    let path = match decode_request_path(request.uri().path()) {
        Ok(path) => path,
        Err(message) => return error_response(StatusCode::BAD_REQUEST, message, None),
    };
    let core = app.state::<AppCore>();
    let scope = asset_scope_snapshot(app);
    let project_authority = match project_request_authority(&core, &scope, &path) {
        Ok(authority) => authority,
        Err(response) => return *response,
    };
    // A retained current-project root is itself the authority for nested
    // relative assets. External files and the Home thumbnail exception still
    // require an exact/runtime scope grant before any helper is involved.
    let non_project_authority = if project_authority.is_none() {
        match non_project_asset_authority(app, &core, &scope, &path) {
            Some(authority) => Some(authority),
            None => {
                return error_response(
                    StatusCode::FORBIDDEN,
                    "local asset path is not approved",
                    None,
                );
            }
        }
    } else {
        None
    };
    let token = pool::random_token();
    let helper_request = HelperRequest {
        token: token.clone(),
        parent_pid: std::process::id(),
        path: NativePath::from(path).to_wire(),
        head_only: request.method() == Method::HEAD,
        range: request
            .headers()
            .get(RANGE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned),
        if_range: request
            .headers()
            .get(IF_RANGE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned),
        project: project_authority
            .as_ref()
            .map(HelperProjectAuthority::from_core),
    };
    let outcome = pool
        .exchange(&helper_request, |opened| {
            authorize_opened_asset(
                app,
                &core,
                project_authority.as_ref(),
                non_project_authority.as_ref(),
                &token,
                opened,
            )
        })
        .await;
    match outcome {
        Ok(HelperOutcome::OpenFailed(opened)) => opened.error_kind.map_or_else(
            || error_response(StatusCode::BAD_GATEWAY, "local asset helper failed", None),
            |kind| kind.response(),
        ),
        Ok(HelperOutcome::Refused(response)) => *response,
        Ok(HelperOutcome::Served { opened, response }) => isolated_response_to_http(
            app,
            &core,
            project_authority.as_ref(),
            non_project_authority.as_ref(),
            &token,
            &opened,
            response,
        ),
        Err(IsolatedHelperError::TimedOut) => error_response(
            StatusCode::GATEWAY_TIMEOUT,
            "local asset I/O timed out",
            Some((RETRY_AFTER, "1")),
        ),
        Err(IsolatedHelperError::Degraded) => error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "local asset isolation is degraded",
            Some((RETRY_AFTER, "5")),
        ),
        Err(IsolatedHelperError::Io | IsolatedHelperError::InvalidResponse) => error_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            "local asset is unavailable",
            None,
        ),
    }
}

fn project_request_authority(
    core: &AppCore,
    scope: &ScopeSnapshot,
    path: &Path,
) -> Result<Option<ProjectAssetAuthority>, Box<Response<Vec<u8>>>> {
    let Some(bundle_path) = opentake_ancestor(path) else {
        return Ok(None);
    };
    if let Some(authority) = core.project_asset_authority() {
        if paths_equal_for_authority(&authority.project_path, &bundle_path) {
            return Ok(Some(authority));
        }
    }
    if is_home_thumbnail_exception(scope, path, &bundle_path) {
        return Ok(None);
    }
    Err(Box::new(error_response(
        StatusCode::FORBIDDEN,
        "project assets require the current retained project authority",
        None,
    )))
}

fn opentake_ancestor(path: &Path) -> Option<PathBuf> {
    path.ancestors()
        .skip(1)
        .find(|ancestor| {
            ancestor.file_name().is_some_and(|name| {
                #[cfg(windows)]
                {
                    name.as_encoded_bytes()
                        .to_ascii_lowercase()
                        .ends_with(b".opentake")
                }
                #[cfg(not(windows))]
                {
                    name.as_encoded_bytes().ends_with(b".opentake")
                }
            })
        })
        .map(normalized_path)
}

fn normalized_path(path: &Path) -> PathBuf {
    opentake_domain::native_path::normalize(path)
}

fn paths_equal_for_authority(left: &Path, right: &Path) -> bool {
    opentake_domain::native_path::identity_key(left)
        == opentake_domain::native_path::identity_key(right)
}

fn relative_to_authority(path: &Path, root: &Path) -> Option<PathBuf> {
    let path = normalized_path(path);
    let root = normalized_path(root);
    let path_components = path.components().collect::<Vec<_>>();
    let root_components = root.components().collect::<Vec<_>>();
    if root_components.len() >= path_components.len() {
        return None;
    }
    let matches_root =
        path_components
            .iter()
            .zip(&root_components)
            .all(|(path_component, root_component)| {
                #[cfg(target_os = "windows")]
                {
                    paths_equal_for_authority(
                        Path::new(path_component.as_os_str()),
                        Path::new(root_component.as_os_str()),
                    )
                }
                #[cfg(not(target_os = "windows"))]
                {
                    path_component == root_component
                }
            });
    matches_root.then(|| {
        let mut relative = PathBuf::new();
        for component in &path_components[root_components.len()..] {
            relative.push(component.as_os_str());
        }
        relative
    })
}

fn is_home_thumbnail_exception(scope: &ScopeSnapshot, path: &Path, bundle_path: &Path) -> bool {
    paths_equal_for_authority(path, &bundle_path.join(HOME_THUMBNAIL_FILE))
        && scope.has_exact_file_grant(path)
}

/// Runtime dialog grants are persisted by Tauri. Keep the configured
/// application-owned cache/data/resource roots available, but require every
/// other external media path to remain referenced by the current project.
/// This also closes stale recursive directory grants from folder imports, not
/// just exact file grants, without mutating persisted scope state.
///
/// Lexical only: no file is opened here. The helper reports the retained
/// handle's final path, which [`non_project_final_path_is_authorized`] checks
/// before any byte is read.
fn non_project_asset_authority<R: Runtime>(
    app: &AppHandle<R>,
    core: &AppCore,
    scope: &ScopeSnapshot,
    path: &Path,
) -> Option<NonProjectAssetAuthority> {
    let normalized = normalized_path(path);
    if !scope.allows(&normalized) {
        return None;
    }
    if opentake_ancestor(&normalized)
        .is_some_and(|bundle| is_home_thumbnail_exception(scope, &normalized, bundle.as_path()))
    {
        return Some(NonProjectAssetAuthority::ScopeOnly {
            kind: ScopeOnlyKind::HomeThumbnail,
            requested_path: normalized,
        });
    }
    if application_owned_asset_roots(app)
        .iter()
        .any(|root| path_is_at_or_below(&normalized, root))
    {
        return Some(NonProjectAssetAuthority::ScopeOnly {
            kind: ScopeOnlyKind::ApplicationOwned,
            requested_path: normalized,
        });
    }
    let index = external_media_index(core);
    (index.revision.has_project_dir && index.paths.contains(&authority_key(&normalized))).then(
        || NonProjectAssetAuthority::ProjectMedia {
            project_epoch: index.revision.project_epoch,
            requested_path: normalized,
        },
    )
}

/// Authorize the final path of the handle a helper opened for `expected`,
/// exactly as the requested path was: in scope lexically, of the same kind,
/// and never inside a `.opentake` bundle other than as its Home thumbnail.
fn non_project_final_path_is_authorized<R: Runtime>(
    app: &AppHandle<R>,
    scope: &ScopeSnapshot,
    expected: &NonProjectAssetAuthority,
    final_path: &Path,
) -> bool {
    let bundle = opentake_ancestor(final_path);
    let home_thumbnail = bundle
        .as_deref()
        .is_some_and(|bundle| is_home_thumbnail_exception(scope, final_path, bundle));
    if !scope.allows(final_path) || (bundle.is_some() && !home_thumbnail) {
        return false;
    }
    match expected {
        NonProjectAssetAuthority::ScopeOnly {
            kind: ScopeOnlyKind::HomeThumbnail,
            ..
        } => home_thumbnail,
        NonProjectAssetAuthority::ScopeOnly {
            kind: ScopeOnlyKind::ApplicationOwned,
            ..
        } => application_owned_asset_roots(app)
            .iter()
            .any(|root| path_is_at_or_below(final_path, root)),
        NonProjectAssetAuthority::ProjectMedia { .. } => true,
    }
}

type AuthorityKey = PathBuf;

fn authority_key(path: &Path) -> AuthorityKey {
    opentake_domain::native_path::identity_key(path)
}

/// External media paths of one manifest revision, keyed for O(1) lookups.
struct ExternalMediaIndex {
    revision: MediaAuthorityRevision,
    paths: HashSet<AuthorityKey>,
}

/// Recently built indexes. Editor generations are unique across every core
/// in the process, so indexes of different cores never alias.
static EXTERNAL_MEDIA_INDEXES: Mutex<Vec<Arc<ExternalMediaIndex>>> = Mutex::new(Vec::new());
const EXTERNAL_MEDIA_INDEX_CACHE: usize = 4;

/// The external media authorization index for the core's current manifest.
/// Rebuilt only after the manifest (or project) changed; a request costs one
/// session-lock revision read and one hash lookup instead of a deep runtime
/// snapshot and a linear scan.
fn external_media_index(core: &AppCore) -> Arc<ExternalMediaIndex> {
    let revision = core.media_authority_revision();
    if let Some(index) = EXTERNAL_MEDIA_INDEXES
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .iter()
        .find(|index| index.revision == revision)
    {
        return index.clone();
    }
    let (revision, paths) = core.external_media_paths();
    let index = Arc::new(ExternalMediaIndex {
        revision,
        paths: paths.iter().map(|path| authority_key(path)).collect(),
    });
    let mut indexes = EXTERNAL_MEDIA_INDEXES
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    indexes.retain(|cached| cached.revision != revision);
    if indexes.len() >= EXTERNAL_MEDIA_INDEX_CACHE {
        indexes.remove(0);
    }
    indexes.push(index.clone());
    index
}

pub(crate) fn application_owned_asset_roots<R: Runtime>(app: &AppHandle<R>) -> Vec<PathBuf> {
    let resolver = app.path();
    let mut roots = Vec::with_capacity(3);
    if let Ok(path) = resolver.app_cache_dir() {
        roots.push(normalized_path(&path));
    }
    if let Ok(path) = resolver.app_data_dir() {
        roots.push(normalized_path(&path.join("OpenTake/Library")));
    }
    if let Ok(path) = resolver.resource_dir() {
        roots.push(normalized_path(&path));
    }
    roots
}

fn path_is_at_or_below(path: &Path, root: &Path) -> bool {
    paths_equal_for_authority(path, root) || relative_to_authority(path, root).is_some()
}

fn decode_request_path(uri_path: &str) -> Result<PathBuf, &'static str> {
    let encoded = uri_path
        .strip_prefix('/')
        .ok_or("local asset URL must contain an absolute path")?;
    let decoded = percent_decode(encoded.as_bytes())
        .decode_utf8()
        .map_err(|_| "local asset path is not valid UTF-8")?;
    if decoded.is_empty()
        || decoded.len() > MAX_PATH_BYTES * 4 + 64
        || decoded.as_bytes().contains(&0)
    {
        return Err("local asset path length is invalid");
    }
    let path = NativePath::from_wire(decoded.as_ref())?.into_path_buf()?;
    if path.as_os_str().as_encoded_bytes().len() > MAX_PATH_BYTES {
        return Err("local asset path length is invalid");
    }
    if !path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                std::path::Component::CurDir | std::path::Component::ParentDir
            )
        })
        || !platform_path_is_local(&path)
    {
        return Err("local asset path must be on a supported local volume");
    }
    Ok(path)
}

#[cfg(test)]
fn serve_open_file(
    path: &Path,
    scope: Option<&Scope>,
    head_only: bool,
    range: Option<&tauri::http::HeaderValue>,
) -> std::io::Result<Response<Vec<u8>>> {
    let (file, final_path) = open_retained_regular_file(path)?;
    if scope.is_some_and(|scope| !scope_allows_lexical_path(scope, &final_path)) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "the opened asset resolves outside its approved scope",
        ));
    }
    serve_opened_file(file, &final_path, head_only, range, None)
}

fn serve_opened_file(
    mut file: File,
    final_path: &Path,
    head_only: bool,
    range: Option<&tauri::http::HeaderValue>,
    if_range: Option<&tauri::http::HeaderValue>,
) -> std::io::Result<Response<Vec<u8>>> {
    let metadata = file.metadata()?;
    let length = metadata.len();
    let etag = retained_file_etag(&file, &metadata)?;
    let mime = mime_guess::from_path(final_path).first_or_octet_stream();
    if !allowed_media_mime(mime.essence_str()) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "only inert media content may use the local asset protocol",
        ));
    }

    let mut builder = secure_response_builder(StatusCode::OK)
        .header(ACCEPT_RANGES, "bytes")
        .header(ETAG, &etag)
        .header(CONTENT_TYPE, mime.essence_str());

    let range = range.filter(|_| if_range.is_none_or(|value| value.as_bytes() == etag.as_bytes()));
    if let Some(range) = range {
        let range = range
            .to_str()
            .ok()
            .and_then(|value| HttpRange::parse(value, length).ok())
            .and_then(|ranges| (ranges.len() == 1).then(|| ranges[0]));
        let Some(range) = range else {
            return Ok(range_not_satisfiable(length));
        };
        if range.start >= length || range.length == 0 {
            return Ok(range_not_satisfiable(length));
        }
        let read_length = range.length.min(MAX_RANGE_BYTES);
        let end = range.start.saturating_add(read_length).saturating_sub(1);
        builder = builder
            .status(StatusCode::PARTIAL_CONTENT)
            .header(
                CONTENT_RANGE,
                format!("bytes {}-{end}/{length}", range.start),
            )
            .header(ACCESS_CONTROL_EXPOSE_HEADERS, "content-range, etag")
            .header(CONTENT_LENGTH, read_length);
        if head_only {
            return Ok(builder.body(Vec::new()).expect("static response headers"));
        }
        file.seek(SeekFrom::Start(range.start))?;
        let body = read_exact_bounded(&mut file, read_length)?;
        ensure_retained_file_identity(&file, &etag)?;
        return Ok(builder.body(body).expect("static response headers"));
    }

    builder = builder.header(CONTENT_LENGTH, length);
    if head_only {
        return Ok(builder.body(Vec::new()).expect("static response headers"));
    }

    let full_body_limit = if mime.type_().as_str() == "image" {
        MAX_FULL_IMAGE_BODY_BYTES
    } else {
        MAX_FULL_BODY_BYTES
    };
    if length > full_body_limit {
        return Ok(error_response(
            StatusCode::PAYLOAD_TOO_LARGE,
            "local asset requires a byte-range request",
            None,
        ));
    }
    let body = read_exact_bounded(&mut file, length)?;
    ensure_retained_file_identity(&file, &etag)?;
    Ok(builder.body(body).expect("static response headers"))
}

fn allowed_media_mime(mime: &str) -> bool {
    (mime.starts_with("image/") && mime != "image/svg+xml")
        || mime.starts_with("audio/")
        || mime.starts_with("video/")
        || mime == "application/octet-stream"
}

fn read_exact_bounded(file: &mut File, length: u64) -> std::io::Result<Vec<u8>> {
    let capacity = usize::try_from(length).map_err(|_| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "asset range is too large")
    })?;
    let mut body = Vec::with_capacity(capacity);
    file.take(length).read_to_end(&mut body)?;
    if body.len() != capacity {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "asset changed while its retained handle was being read",
        ));
    }
    Ok(body)
}

fn ensure_retained_file_identity(file: &File, expected_etag: &str) -> std::io::Result<()> {
    let metadata = file.metadata()?;
    if retained_file_etag(file, &metadata)? != expected_etag {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "asset identity changed while its retained handle was being read",
        ));
    }
    Ok(())
}

#[cfg(unix)]
fn retained_file_etag(_file: &File, metadata: &std::fs::Metadata) -> std::io::Result<String> {
    use std::os::unix::fs::MetadataExt as _;

    Ok(format!(
        "\"{:x}-{:x}-{:x}-{:x}-{:x}-{:x}-{:x}\"",
        metadata.dev(),
        metadata.ino(),
        metadata.len(),
        metadata.mtime(),
        metadata.mtime_nsec(),
        metadata.ctime(),
        metadata.ctime_nsec(),
    ))
}

#[cfg(target_os = "windows")]
fn retained_file_etag(file: &File, metadata: &std::fs::Metadata) -> std::io::Result<String> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::Storage::FileSystem::{
        GetFileInformationByHandle, BY_HANDLE_FILE_INFORMATION,
    };

    let mut information = BY_HANDLE_FILE_INFORMATION::default();
    // SAFETY: `file` owns a live handle and `information` is writable.
    if unsafe {
        GetFileInformationByHandle(
            file.as_raw_handle() as HANDLE,
            std::ptr::addr_of_mut!(information),
        )
    } == 0
    {
        return Err(std::io::Error::last_os_error());
    }
    let file_index =
        (u64::from(information.nFileIndexHigh) << 32) | u64::from(information.nFileIndexLow);
    Ok(format!(
        "\"{:x}-{:x}-{:x}-{:x}{:08x}\"",
        information.dwVolumeSerialNumber,
        file_index,
        metadata.len(),
        information.ftLastWriteTime.dwHighDateTime,
        information.ftLastWriteTime.dwLowDateTime,
    ))
}

pub(crate) fn open_retained_regular_file(path: &Path) -> std::io::Result<(File, PathBuf)> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options
            .share_mode(
                windows_sys::Win32::Storage::FileSystem::FILE_SHARE_READ
                    | windows_sys::Win32::Storage::FileSystem::FILE_SHARE_WRITE
                    | windows_sys::Win32::Storage::FileSystem::FILE_SHARE_DELETE,
            )
            .custom_flags(windows_sys::Win32::Storage::FileSystem::FILE_FLAG_OPEN_NO_RECALL);
    }
    let file = options.open(path)?;
    let final_path = validate_opened_resident_regular_file(&file)?;
    Ok((file, final_path))
}

fn validate_opened_resident_regular_file(file: &File) -> std::io::Result<PathBuf> {
    let metadata = file.metadata()?;
    if !metadata.is_file() || retained_metadata_is_unavailable(file, &metadata) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "asset is not a resident regular file on a local volume",
        ));
    }
    let final_path = retained_final_path(file)?;
    if !platform_path_is_local(&final_path) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "asset resolves outside a supported local volume",
        ));
    }
    Ok(final_path)
}

const HOME_THUMBNAIL_FILE: &str = "thumbnail.jpg";

/// Validate one exact Home thumbnail grant without expanding it to a recursive
/// bundle scope.
///
/// The cover is opened through a retained no-follow root for its `.opentake`
/// bundle, so only the `thumbnail.jpg` leaf of a real bundle directory
/// qualifies: a symlinked bundle or leaf is rejected. Ancestor directories may
/// be symlinks, junctions or `subst` drives (macOS `/tmp`, linked volumes), so
/// the requested path is not compared with the returned one. The returned path
/// comes from the retained file handle and must itself have the Home cover
/// shape that the asset protocol re-checks for every request.
pub(crate) fn validate_resident_home_thumbnail(thumbnail: &Path) -> std::io::Result<PathBuf> {
    let not_a_cover = || {
        std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "path is not the thumbnail of a .opentake bundle",
        )
    };
    let bundle = opentake_ancestor(thumbnail).ok_or_else(not_a_cover)?;
    if !paths_equal_for_authority(thumbnail, &bundle.join(HOME_THUMBNAIL_FILE)) {
        return Err(not_a_cover());
    }
    let root = ProjectRoot::open(&bundle).map_err(std::io::Error::other)?;
    let file = root
        .open_asset_file(Path::new(HOME_THUMBNAIL_FILE))
        .map_err(std::io::Error::other)?;
    let final_path = validate_opened_resident_regular_file(&file)?;
    let final_bundle = opentake_ancestor(&final_path).ok_or_else(not_a_cover)?;
    if !paths_equal_for_authority(&final_path, &final_bundle.join(HOME_THUMBNAIL_FILE)) {
        return Err(not_a_cover());
    }
    Ok(final_path)
}

#[cfg(test)]
pub(crate) fn scope_allows_lexical_path(scope: &Scope, path: &Path) -> bool {
    let normalized: PathBuf = path.components().collect();
    let options = scope_match_options();
    if scope
        .forbidden_patterns()
        .iter()
        .any(|pattern| pattern.matches_path_with(&normalized, options))
    {
        return false;
    }
    scope
        .allowed_patterns()
        .iter()
        .any(|pattern| pattern.matches_path_with(&normalized, options))
}

fn scope_match_options() -> glob::MatchOptions {
    glob::MatchOptions {
        #[cfg(target_os = "windows")]
        case_sensitive: false,
        #[cfg(not(target_os = "windows"))]
        case_sensitive: true,
        require_literal_separator: true,
        #[cfg(unix)]
        require_literal_leading_dot: true,
        #[cfg(not(unix))]
        require_literal_leading_dot: false,
    }
}

#[cfg(target_os = "macos")]
fn retained_final_path(file: &File) -> std::io::Result<PathBuf> {
    use std::ffi::CStr;
    use std::os::unix::ffi::OsStrExt;

    let mut buffer = vec![0_i8; libc::PATH_MAX as usize];
    // SAFETY: `buffer` is writable for PATH_MAX bytes and `file` owns a live fd.
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETPATH, buffer.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: F_GETPATH succeeded and wrote a NUL-terminated pathname.
    let bytes = unsafe { CStr::from_ptr(buffer.as_ptr()) }.to_bytes();
    Ok(PathBuf::from(std::ffi::OsStr::from_bytes(bytes)))
}

#[cfg(all(unix, not(target_os = "macos")))]
fn retained_final_path(file: &File) -> std::io::Result<PathBuf> {
    use std::os::fd::AsRawFd;

    std::fs::read_link(format!("/proc/self/fd/{}", file.as_raw_fd()))
}

#[cfg(target_os = "windows")]
fn retained_final_path(file: &File) -> std::io::Result<PathBuf> {
    use std::ffi::OsString;
    use std::os::windows::{ffi::OsStringExt, io::AsRawHandle};
    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::Storage::FileSystem::{
        GetFinalPathNameByHandleW, FILE_NAME_NORMALIZED, VOLUME_NAME_DOS,
    };

    let handle = file.as_raw_handle() as HANDLE;
    let mut buffer = vec![0_u16; 32_768];
    // SAFETY: `handle` is live and `buffer` is writable for its declared length.
    let mut length = unsafe {
        GetFinalPathNameByHandleW(
            handle,
            buffer.as_mut_ptr(),
            u32::try_from(buffer.len()).unwrap_or(u32::MAX),
            FILE_NAME_NORMALIZED | VOLUME_NAME_DOS,
        )
    };
    if length == 0 {
        return Err(std::io::Error::last_os_error());
    }
    if usize::try_from(length).unwrap_or(usize::MAX) >= buffer.len() {
        buffer.resize(usize::try_from(length).unwrap_or(MAX_PATH_BYTES) + 1, 0);
        // SAFETY: same live handle; resized buffer is writable for its declared length.
        length = unsafe {
            GetFinalPathNameByHandleW(
                handle,
                buffer.as_mut_ptr(),
                u32::try_from(buffer.len()).unwrap_or(u32::MAX),
                FILE_NAME_NORMALIZED | VOLUME_NAME_DOS,
            )
        };
        if length == 0 || usize::try_from(length).unwrap_or(usize::MAX) >= buffer.len() {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok(PathBuf::from(OsString::from_wide(
        &buffer[..usize::try_from(length).unwrap_or(0)],
    )))
}

#[cfg(target_os = "macos")]
fn retained_metadata_is_unavailable(file: &File, metadata: &std::fs::Metadata) -> bool {
    const SF_DATALESS: u32 = 0x4000_0000;
    if metadata.st_flags() & SF_DATALESS != 0 {
        return true;
    }
    let mut stat = std::mem::MaybeUninit::<libc::statfs>::uninit();
    // SAFETY: `stat` points to writable storage and `file` owns a live fd.
    let result = unsafe { libc::fstatfs(file.as_raw_fd(), stat.as_mut_ptr()) };
    if result != 0 {
        return true;
    }
    // SAFETY: fstatfs returned success and initialized the structure.
    let stat = unsafe { stat.assume_init() };
    stat.f_flags & u32::try_from(libc::MNT_LOCAL).unwrap_or(u32::MAX) == 0
}

#[cfg(target_os = "windows")]
fn retained_metadata_is_unavailable(file: &File, _metadata: &std::fs::Metadata) -> bool {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::Storage::CloudFilters::{
        CfGetPlaceholderStateFromAttributeTag, CF_PLACEHOLDER_STATE_PARTIAL,
        CF_PLACEHOLDER_STATE_PARTIALLY_ON_DISK,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        FileAttributeTagInfo, GetFileInformationByHandleEx, FILE_ATTRIBUTE_OFFLINE,
        FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS, FILE_ATTRIBUTE_RECALL_ON_OPEN,
        FILE_ATTRIBUTE_TAG_INFO,
    };

    let mut info = FILE_ATTRIBUTE_TAG_INFO::default();
    // SAFETY: `file` owns a live handle and `info` is writable for its exact size.
    if unsafe {
        GetFileInformationByHandleEx(
            file.as_raw_handle() as HANDLE,
            FileAttributeTagInfo,
            std::ptr::addr_of_mut!(info).cast(),
            u32::try_from(std::mem::size_of::<FILE_ATTRIBUTE_TAG_INFO>()).unwrap_or(u32::MAX),
        )
    } == 0
    {
        return true;
    }
    if info.FileAttributes
        & (FILE_ATTRIBUTE_OFFLINE
            | FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS
            | FILE_ATTRIBUTE_RECALL_ON_OPEN)
        != 0
    {
        return true;
    }
    // Fully hydrated Cloud Files remain reparse points. Reject only partial
    // placeholder states; FILE_FLAG_OPEN_NO_RECALL above prevents this open
    // from silently hydrating the file.
    // SAFETY: the values come from the retained handle's attribute/tag query.
    let placeholder_state =
        unsafe { CfGetPlaceholderStateFromAttributeTag(info.FileAttributes, info.ReparseTag) };
    placeholder_state & (CF_PLACEHOLDER_STATE_PARTIAL | CF_PLACEHOLDER_STATE_PARTIALLY_ON_DISK) != 0
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
fn retained_metadata_is_unavailable(_file: &File, _metadata: &std::fs::Metadata) -> bool {
    false
}

#[cfg(target_os = "windows")]
fn platform_path_is_local(path: &Path) -> bool {
    use std::path::{Component, Prefix};
    use windows_sys::Win32::Storage::FileSystem::GetDriveTypeW;

    // DRIVE_* constants are not exported from Win32::Storage::FileSystem in
    // windows-sys 0.61; define locally like safe_fs/windows.rs.
    const DRIVE_REMOVABLE: u32 = 2;
    const DRIVE_FIXED: u32 = 3;
    let Some(Component::Prefix(prefix)) = path.components().next() else {
        return false;
    };
    let letter = match prefix.kind() {
        Prefix::Disk(letter) | Prefix::VerbatimDisk(letter) => letter,
        _ => return false,
    };
    let root = [u16::from(letter), u16::from(b':'), u16::from(b'\\'), 0];
    // SAFETY: `root` is a valid NUL-terminated drive-root string.
    matches!(
        unsafe { GetDriveTypeW(root.as_ptr()) },
        DRIVE_FIXED | DRIVE_REMOVABLE
    )
}

#[cfg(not(target_os = "windows"))]
fn platform_path_is_local(_path: &Path) -> bool {
    true
}

mod response;
#[cfg(test)]
use response::asset_origin;
use response::{error_response, range_not_satisfiable, secure_response_builder};

#[cfg(test)]
#[path = "safe_asset_protocol/tests.rs"]
mod tests;
