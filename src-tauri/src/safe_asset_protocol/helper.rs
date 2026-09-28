use super::*;

/// Written by a helper once it authenticated its parent and is ready for
/// requests.
pub(super) const HELPER_HANDSHAKE: &[u8] = b"OTAH\x00\x00\x00\x02";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct HelperRequest {
    /// Random per-request token echoed by both replies.
    pub(super) token: String,
    pub(super) parent_pid: u32,
    pub(super) path: String,
    pub(super) head_only: bool,
    pub(super) range: Option<String>,
    pub(super) if_range: Option<String>,
    pub(super) project: Option<HelperProjectAuthority>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct HelperProjectAuthority {
    pub(super) project_epoch: u64,
    pub(super) project_path: String,
    pub(super) root_identity: ProjectRootIdentity,
}

impl HelperProjectAuthority {
    pub(super) fn from_core(authority: &ProjectAssetAuthority) -> Option<Self> {
        Some(Self {
            project_epoch: authority.project_epoch,
            project_path: authority.project_path.to_str()?.to_owned(),
            root_identity: authority.root_identity,
        })
    }
}

/// Parent-to-helper frames, each a big-endian `u32` length plus JSON.
#[derive(Debug, Serialize, Deserialize)]
pub(super) enum HelperFrame {
    /// Open the requested file; `session` must equal the helper's process
    /// secret from its environment.
    Request {
        session: String,
        request: HelperRequest,
    },
    /// The opened identity is authorized: serve the request from it.
    Proceed { token: String },
    /// The opened identity is not authorized: read nothing and close it.
    Abort { token: String },
}

/// Helper-to-parent frames: a big-endian `u32` metadata length, the JSON
/// metadata and, for [`HelperReply::Served`], `body_length` body bytes.
#[derive(Debug, Serialize, Deserialize)]
pub(super) enum HelperReply {
    Opened(OpenedMetadata),
    Served(HelperResponseMetadata),
}

/// Identity of the file a helper opened, reported before it reads any byte.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct OpenedMetadata {
    pub(super) token: String,
    pub(super) final_path: Option<String>,
    pub(super) etag: Option<String>,
    pub(super) project_root_identity: Option<ProjectRootIdentity>,
    pub(super) error_kind: Option<WireIoErrorKind>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct HelperResponseMetadata {
    pub(super) token: String,
    pub(super) final_path: Option<String>,
    pub(super) project_root_identity: Option<ProjectRootIdentity>,
    pub(super) status: u16,
    pub(super) headers: Vec<(String, String)>,
    pub(super) body_length: u64,
    pub(super) error_kind: Option<WireIoErrorKind>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(super) enum WireIoErrorKind {
    NotFound,
    PermissionDenied,
    InvalidInput,
    Other,
}

impl WireIoErrorKind {
    fn from_error(error: &std::io::Error) -> Self {
        match error.kind() {
            std::io::ErrorKind::NotFound => Self::NotFound,
            std::io::ErrorKind::PermissionDenied => Self::PermissionDenied,
            std::io::ErrorKind::InvalidInput => Self::InvalidInput,
            _ => Self::Other,
        }
    }

    pub(super) fn response(self) -> Response<Vec<u8>> {
        let status = match self {
            Self::NotFound => StatusCode::NOT_FOUND,
            Self::PermissionDenied | Self::InvalidInput => StatusCode::FORBIDDEN,
            Self::Other => StatusCode::UNPROCESSABLE_ENTITY,
        };
        error_response(status, "local asset is unavailable", None)
    }
}

pub(super) struct IsolatedResponse {
    pub(super) metadata: HelperResponseMetadata,
    pub(super) body: Vec<u8>,
}

#[derive(Debug)]
pub(super) enum IsolatedHelperError {
    TimedOut,
    Degraded,
    Io,
    InvalidResponse,
}

pub(super) async fn write_helper_frame<W: tokio::io::AsyncWrite + Unpin>(
    writer: &mut W,
    frame: &HelperFrame,
) -> Result<(), IsolatedHelperError> {
    let encoded = serde_json::to_vec(frame).map_err(|_| IsolatedHelperError::InvalidResponse)?;
    if encoded.is_empty() || encoded.len() > MAX_HELPER_REQUEST_BYTES {
        return Err(IsolatedHelperError::InvalidResponse);
    }
    let mut framed = Vec::with_capacity(encoded.len() + 4);
    framed.extend_from_slice(&(encoded.len() as u32).to_be_bytes());
    framed.extend_from_slice(&encoded);
    writer
        .write_all(&framed)
        .await
        .map_err(|_| IsolatedHelperError::Io)?;
    writer.flush().await.map_err(|_| IsolatedHelperError::Io)
}

/// Read one reply whose body may not exceed `max_body` bytes.
pub(super) async fn read_helper_reply<R: tokio::io::AsyncRead + Unpin>(
    reader: &mut R,
    max_body: usize,
) -> Result<(HelperReply, Vec<u8>), IsolatedHelperError> {
    let mut metadata_length = [0_u8; 4];
    reader
        .read_exact(&mut metadata_length)
        .await
        .map_err(|_| IsolatedHelperError::InvalidResponse)?;
    let metadata_length = u32::from_be_bytes(metadata_length) as usize;
    if metadata_length == 0 || metadata_length > MAX_HELPER_METADATA_BYTES {
        return Err(IsolatedHelperError::InvalidResponse);
    }
    let mut metadata_bytes = vec![0_u8; metadata_length];
    reader
        .read_exact(&mut metadata_bytes)
        .await
        .map_err(|_| IsolatedHelperError::InvalidResponse)?;
    let reply: HelperReply = serde_json::from_slice(&metadata_bytes)
        .map_err(|_| IsolatedHelperError::InvalidResponse)?;
    let body_length = match &reply {
        HelperReply::Opened(_) => 0,
        HelperReply::Served(metadata) => usize::try_from(metadata.body_length)
            .map_err(|_| IsolatedHelperError::InvalidResponse)?,
    };
    if body_length > max_body.min(MAX_HELPER_BODY_BYTES) {
        return Err(IsolatedHelperError::InvalidResponse);
    }
    let mut body = vec![0_u8; body_length];
    reader
        .read_exact(&mut body)
        .await
        .map_err(|_| IsolatedHelperError::InvalidResponse)?;
    Ok((reply, body))
}

/// Authorize the identity a helper opened before it reads any byte. Only
/// in-memory state is consulted: the cached scope snapshot, the external media
/// index and the retained project authority. The parent never opens the file.
pub(super) fn authorize_opened_asset<R: Runtime>(
    app: &AppHandle<R>,
    core: &AppCore,
    expected_project: Option<&ProjectAssetAuthority>,
    expected_non_project: Option<&NonProjectAssetAuthority>,
    token: &str,
    opened: &OpenedMetadata,
) -> Result<(), Box<Response<Vec<u8>>>> {
    if opened.token != token {
        return Err(Box::new(error_response(
            StatusCode::BAD_GATEWAY,
            "local asset helper failed",
            None,
        )));
    }
    if let Some(kind) = opened.error_kind {
        return Err(Box::new(kind.response()));
    }
    let (Some(final_path), Some(_)) = (opened.final_path.as_deref(), opened.etag.as_deref()) else {
        return Err(Box::new(error_response(
            StatusCode::BAD_GATEWAY,
            "local asset helper failed",
            None,
        )));
    };
    let final_path = Path::new(final_path);
    if let Some(expected) = expected_project {
        // The helper re-opened the bundle, required this exact retained root
        // identity and opened every asset component no-follow beneath it, so
        // the identity is the authority. The handle's final path is resolved
        // by the OS (symlinked ancestors such as macOS `/tmp`, junctions and
        // `subst` drives), so it cannot be compared with the opened path.
        if opened.project_root_identity != Some(expected.root_identity)
            || !core.project_asset_authority_matches(expected)
        {
            return Err(Box::new(error_response(
                StatusCode::FORBIDDEN,
                "project asset authority changed during the read",
                None,
            )));
        }
        return Ok(());
    }
    let Some(expected) = expected_non_project else {
        return Err(Box::new(outside_scope()));
    };
    if opened.project_root_identity.is_some() {
        return Err(Box::new(outside_scope()));
    }
    let scope = asset_scope_snapshot(app);
    // Re-derive the requested path's authority from current in-memory state:
    // a scope revocation, project switch or manifest removal since admission
    // must not publish bytes under the admission-time decision.
    if non_project_asset_authority(app, core, &scope, expected.requested_path()).as_ref()
        != Some(expected)
    {
        return Err(Box::new(outside_scope()));
    }
    if !non_project_final_path_is_authorized(app, &scope, expected, final_path) {
        return Err(Box::new(outside_scope()));
    }
    Ok(())
}

fn outside_scope() -> Response<Vec<u8>> {
    error_response(
        StatusCode::FORBIDDEN,
        "the opened asset resolves outside its approved scope",
        None,
    )
}

/// Publish a served helper response only if it is the identity that was
/// authorized before the read and that authorization still holds.
pub(super) fn isolated_response_to_http<R: Runtime>(
    app: &AppHandle<R>,
    core: &AppCore,
    expected_project: Option<&ProjectAssetAuthority>,
    expected_non_project: Option<&NonProjectAssetAuthority>,
    token: &str,
    opened: &OpenedMetadata,
    isolated: IsolatedResponse,
) -> Response<Vec<u8>> {
    let metadata = isolated.metadata;
    if metadata.token != token || metadata.body_length != isolated.body.len() as u64 {
        return error_response(StatusCode::BAD_GATEWAY, "local asset helper failed", None);
    }
    if let Some(kind) = metadata.error_kind {
        return kind.response();
    }
    let response_etag = metadata
        .headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("etag"))
        .map(|(_, value)| value.as_str());
    let success = (200..300).contains(&metadata.status);
    // The served bytes must come from the identity reported before the read:
    // same handle path, same root, and the same file identity (etag).
    if metadata.final_path.is_none()
        || metadata.final_path != opened.final_path
        || metadata.project_root_identity != opened.project_root_identity
        || (success && response_etag.is_none())
        || response_etag.is_some_and(|etag| Some(etag) != opened.etag.as_deref())
    {
        return error_response(
            StatusCode::FORBIDDEN,
            "local asset identity changed during the read",
            None,
        );
    }
    // Acquire the project-transition lease before the final authorization
    // comparison. Once held, the current bundle/external-media authority cannot
    // rotate between this check and byte publication. Only in-memory state is
    // compared under the lease; no file is opened.
    let requires_project_lease = expected_project.is_some()
        || matches!(
            expected_non_project,
            Some(NonProjectAssetAuthority::ProjectMedia { .. })
        );
    let _identity_lease = requires_project_lease.then(|| core.lock_project_identity_workflow());
    if let Err(response) = authorize_opened_asset(
        app,
        core,
        expected_project,
        expected_non_project,
        token,
        opened,
    ) {
        return *response;
    }

    let status = StatusCode::from_u16(metadata.status).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut builder = secure_response_builder(status);
    for (name, value) in metadata.headers {
        let Ok(name) = tauri::http::HeaderName::from_bytes(name.as_bytes()) else {
            return error_response(StatusCode::BAD_GATEWAY, "local asset helper failed", None);
        };
        let Ok(value) = tauri::http::HeaderValue::from_str(&value) else {
            return error_response(StatusCode::BAD_GATEWAY, "local asset helper failed", None);
        };
        builder = builder.header(name, value);
    }
    builder.body(isolated.body).unwrap_or_else(|_| {
        error_response(StatusCode::BAD_GATEWAY, "local asset helper failed", None)
    })
}

/// Run the undocumented asset reader mode before Tauri starts. The random
/// session secret and actual parent PID must agree across env and stdin.
#[doc(hidden)]
pub(crate) fn run_helper_if_requested() -> bool {
    if std::env::args_os().nth(1).as_deref() != Some(std::ffi::OsStr::new(HELPER_ARG)) {
        return false;
    }
    let exit_code = run_helper_stdio().map_or(1, |()| 0);
    std::process::exit(exit_code);
}

/// Authenticate the parent, then serve framed requests until stdin closes.
pub(super) fn run_helper_stdio() -> std::io::Result<()> {
    let expected_session = std::env::var(HELPER_TOKEN_ENV)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::PermissionDenied, "missing token"))?;
    let expected_parent = std::env::var(HELPER_PARENT_ENV)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::PermissionDenied, "missing parent")
        })?;
    ensure_helper_parent(expected_parent)?;
    if !parent_is_same_executable(expected_parent)? {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "helper parent executable mismatch",
        ));
    }
    let mut stdin = std::io::stdin().lock();
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(HELPER_HANDSHAKE)?;
    stdout.flush()?;
    loop {
        // EOF at a frame boundary is the parent's normal way to retire us.
        let Some(frame) = read_helper_frame(&mut stdin)? else {
            return Ok(());
        };
        let HelperFrame::Request { session, request } = frame else {
            return Err(protocol_error("expected a helper request"));
        };
        if session != expected_session || request.parent_pid != expected_parent {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "helper authentication failed",
            ));
        }
        // A reparented helper no longer serves the process that started it.
        ensure_helper_parent(expected_parent)?;
        let opened = open_helper_asset(&request);
        let metadata = opened_metadata(&request, &opened);
        // The parent answers only an identity reported without an error;
        // after an error it sends the next request instead of a decision.
        let decision_follows = metadata.error_kind.is_none();
        write_helper_reply(&mut stdout, &HelperReply::Opened(metadata), &[])?;
        let Ok(opened) = opened else {
            continue;
        };
        if !decision_follows {
            continue;
        }
        match read_helper_frame(&mut stdin)? {
            Some(HelperFrame::Proceed { token }) if token == request.token => {
                let response = serve_helper_asset(&request, opened);
                write_helper_reply(
                    &mut stdout,
                    &HelperReply::Served(response.metadata),
                    &response.body,
                )?;
            }
            Some(HelperFrame::Abort { token }) if token == request.token => {}
            _ => return Err(protocol_error("expected a decision for the open request")),
        }
    }
}

fn ensure_helper_parent(expected_parent: u32) -> std::io::Result<()> {
    if expected_parent != actual_parent_process_id()? {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "helper parent mismatch",
        ));
    }
    Ok(())
}

fn protocol_error(message: &'static str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message)
}

pub(super) fn read_helper_frame(input: &mut impl Read) -> std::io::Result<Option<HelperFrame>> {
    let mut length = [0_u8; 4];
    let mut filled = 0;
    while filled < length.len() {
        match input.read(&mut length[filled..]) {
            Ok(0) if filled == 0 => return Ok(None),
            Ok(0) => return Err(protocol_error("truncated helper frame")),
            Ok(read) => filled += read,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    let length = u32::from_be_bytes(length) as usize;
    if length == 0 || length > MAX_HELPER_REQUEST_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "helper request is too large",
        ));
    }
    let mut encoded = vec![0_u8; length];
    input.read_exact(&mut encoded)?;
    serde_json::from_slice(&encoded)
        .map(Some)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error))
}

pub(super) fn write_helper_reply(
    output: &mut impl Write,
    reply: &HelperReply,
    body: &[u8],
) -> std::io::Result<()> {
    let metadata = serde_json::to_vec(reply)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    if metadata.len() > MAX_HELPER_METADATA_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "helper metadata is too large",
        ));
    }
    output.write_all(&(metadata.len() as u32).to_be_bytes())?;
    output.write_all(&metadata)?;
    output.write_all(body)?;
    output.flush()
}

#[cfg(unix)]
pub(super) fn actual_parent_process_id() -> std::io::Result<u32> {
    // SAFETY: getppid has no preconditions.
    Ok(unsafe { libc::getppid() } as u32)
}

#[cfg(target_os = "linux")]
pub(super) fn parent_is_same_executable(parent_pid: u32) -> std::io::Result<bool> {
    same_file::is_same_file(
        std::fs::read_link(format!("/proc/{parent_pid}/exe"))?,
        std::env::current_exe()?,
    )
}

#[cfg(all(unix, not(any(target_os = "linux", target_os = "macos"))))]
pub(super) fn parent_is_same_executable(_parent_pid: u32) -> std::io::Result<bool> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "asset helper parent identity is unsupported on this platform",
    ))
}

#[cfg(target_os = "macos")]
pub(super) fn parent_is_same_executable(parent_pid: u32) -> std::io::Result<bool> {
    use std::ffi::c_void;
    use std::os::unix::ffi::OsStrExt;

    const PROC_PIDPATHINFO_MAXSIZE: usize = 4096;
    #[link(name = "proc")]
    unsafe extern "C" {
        fn proc_pidpath(pid: libc::c_int, buffer: *mut c_void, buffersize: u32) -> libc::c_int;
    }
    let mut buffer = vec![0_u8; PROC_PIDPATHINFO_MAXSIZE];
    // SAFETY: `buffer` is writable for the declared size and parent_pid names
    // the helper's live parent.
    let length = unsafe {
        proc_pidpath(
            parent_pid as libc::c_int,
            buffer.as_mut_ptr().cast(),
            buffer.len() as u32,
        )
    };
    if length <= 0 {
        return Err(std::io::Error::last_os_error());
    }
    buffer.truncate(length as usize);
    same_file::is_same_file(
        PathBuf::from(std::ffi::OsStr::from_bytes(&buffer)),
        std::env::current_exe()?,
    )
}

#[cfg(target_os = "windows")]
pub(super) fn actual_parent_process_id() -> std::io::Result<u32> {
    use std::ffi::c_void;

    #[repr(C)]
    struct ProcessBasicInformation {
        reserved1: *mut c_void,
        peb_base_address: *mut c_void,
        reserved2: [*mut c_void; 2],
        unique_process_id: usize,
        inherited_from_unique_process_id: usize,
    }
    #[link(name = "ntdll")]
    unsafe extern "system" {
        fn NtQueryInformationProcess(
            process_handle: isize,
            process_information_class: u32,
            process_information: *mut c_void,
            process_information_length: u32,
            return_length: *mut u32,
        ) -> i32;
    }
    let mut information = std::mem::MaybeUninit::<ProcessBasicInformation>::uninit();
    let mut returned = 0_u32;
    // SAFETY: -1 is the documented current-process pseudo-handle and the
    // output buffer is writable for its exact declared size.
    let status = unsafe {
        NtQueryInformationProcess(
            -1_isize,
            0,
            information.as_mut_ptr().cast(),
            u32::try_from(std::mem::size_of::<ProcessBasicInformation>()).unwrap_or(u32::MAX),
            &mut returned,
        )
    };
    if status < 0 {
        return Err(std::io::Error::other(format!(
            "NtQueryInformationProcess failed with NTSTATUS {status:#x}"
        )));
    }
    // SAFETY: the successful syscall initialized the output structure.
    let information = unsafe { information.assume_init() };
    u32::try_from(information.inherited_from_unique_process_id)
        .map_err(|_| std::io::Error::other("parent process ID is out of range"))
}

#[cfg(target_os = "windows")]
pub(super) fn parent_is_same_executable(parent_pid: u32) -> std::io::Result<bool> {
    use std::ffi::c_void;
    use std::os::windows::ffi::OsStringExt;

    const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn OpenProcess(desired_access: u32, inherit_handle: i32, process_id: u32) -> *mut c_void;
        fn QueryFullProcessImageNameW(
            process: *mut c_void,
            flags: u32,
            executable_name: *mut u16,
            size: *mut u32,
        ) -> i32;
        fn CloseHandle(object: *mut c_void) -> i32;
    }

    // SAFETY: the access mask is read-only and parent_pid names a live process.
    let process = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, parent_pid) };
    if process.is_null() {
        return Err(std::io::Error::last_os_error());
    }
    let mut buffer = vec![0_u16; 32_768];
    let mut length = buffer.len() as u32;
    // SAFETY: `process` is live and `buffer` is writable for `length` UTF-16 units.
    let queried =
        unsafe { QueryFullProcessImageNameW(process, 0, buffer.as_mut_ptr(), &mut length) };
    let query_error = (queried == 0).then(std::io::Error::last_os_error);
    // SAFETY: `process` was returned by OpenProcess and is closed exactly once.
    let _ = unsafe { CloseHandle(process) };
    if let Some(error) = query_error {
        return Err(error);
    }
    buffer.truncate(length as usize);
    same_file::is_same_file(
        PathBuf::from(std::ffi::OsString::from_wide(&buffer)),
        std::env::current_exe()?,
    )
}

/// A file the helper opened and validated but has not read yet.
pub(super) struct OpenedAsset {
    file: File,
    final_path: PathBuf,
    etag: String,
    project_root_identity: Option<ProjectRootIdentity>,
}

pub(super) fn open_helper_asset(request: &HelperRequest) -> std::io::Result<OpenedAsset> {
    let path = PathBuf::from(&request.path);
    let (file, final_path, project_root_identity) = if let Some(project) = &request.project {
        let project_path = PathBuf::from(&project.project_path);
        let relative = relative_to_authority(&path, &project_path).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "project asset is outside the retained root",
            )
        })?;
        if relative
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "project asset relative path is invalid",
            ));
        }
        let root = ProjectRoot::open(&project_path).map_err(std::io::Error::other)?;
        if root.stable_identity() != project.root_identity {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "project root identity changed before asset read",
            ));
        }
        let file = root
            .open_asset_file(&relative)
            .map_err(std::io::Error::other)?;
        let final_path = validate_opened_resident_regular_file(&file)?;
        (file, final_path, Some(root.stable_identity()))
    } else {
        let (file, final_path) = open_retained_regular_file(&path)?;
        (file, final_path, None)
    };
    let metadata = file.metadata()?;
    let etag = retained_file_etag(&file, &metadata)?;
    Ok(OpenedAsset {
        file,
        final_path,
        etag,
        project_root_identity,
    })
}

pub(super) fn opened_metadata(
    request: &HelperRequest,
    opened: &std::io::Result<OpenedAsset>,
) -> OpenedMetadata {
    match opened {
        Ok(opened) => OpenedMetadata {
            token: request.token.clone(),
            final_path: opened.final_path.to_str().map(str::to_owned),
            etag: Some(opened.etag.clone()),
            project_root_identity: opened.project_root_identity,
            // A non-UTF-8 final path cannot be authorized lexically.
            error_kind: opened
                .final_path
                .to_str()
                .is_none()
                .then_some(WireIoErrorKind::PermissionDenied),
        },
        Err(error) => OpenedMetadata {
            token: request.token.clone(),
            final_path: None,
            etag: None,
            project_root_identity: None,
            error_kind: Some(WireIoErrorKind::from_error(error)),
        },
    }
}

pub(super) fn serve_helper_asset(request: &HelperRequest, opened: OpenedAsset) -> IsolatedResponse {
    let range = request
        .range
        .as_deref()
        .and_then(|value| tauri::http::HeaderValue::from_str(value).ok());
    let if_range = request
        .if_range
        .as_deref()
        .and_then(|value| tauri::http::HeaderValue::from_str(value).ok());
    let OpenedAsset {
        file,
        final_path,
        project_root_identity,
        ..
    } = opened;
    match serve_opened_file(
        file,
        &final_path,
        request.head_only,
        range.as_ref(),
        if_range.as_ref(),
    ) {
        Ok(response) => {
            let (parts, body) = response.into_parts();
            let headers = parts
                .headers
                .iter()
                .filter_map(|(name, value)| {
                    value
                        .to_str()
                        .ok()
                        .map(|value| (name.as_str().to_owned(), value.to_owned()))
                })
                .collect();
            IsolatedResponse {
                metadata: HelperResponseMetadata {
                    token: request.token.clone(),
                    final_path: final_path.to_str().map(str::to_owned),
                    project_root_identity,
                    status: parts.status.as_u16(),
                    headers,
                    body_length: body.len() as u64,
                    error_kind: None,
                },
                body,
            }
        }
        Err(error) => IsolatedResponse {
            metadata: HelperResponseMetadata {
                token: request.token.clone(),
                final_path: None,
                project_root_identity: None,
                status: 0,
                headers: Vec::new(),
                body_length: 0,
                error_kind: Some(WireIoErrorKind::from_error(&error)),
            },
            body: Vec::new(),
        },
    }
}
