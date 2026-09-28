//! Official Codex CLI integration.
//!
//! OpenTake never reads, copies, or stores Codex credentials. Authentication is
//! delegated to the user-installed official CLI (`codex login`), and Agent
//! turns run through `codex exec` using that CLI's existing ChatGPT session.

use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
#[cfg(all(test, unix))]
use std::process::Command;
use std::process::{ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime};

use base64::Engine as _;
use opentake_agent::chat::{ChatTurnGate, ToolCall};
use opentake_agent::mcp::dispatch::Dispatcher;
use opentake_agent::plugin::registry::PluginRegistry;
use opentake_agent::tools::result::Block;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tauri::State;
use tokio::io::{
    AsyncBufRead, AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader,
};

const MINIMUM_CODEX_VERSION: (u64, u64, u64) = (0, 146, 0);
const CODEX_TURN_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const CODEX_AUTH_TIMEOUT: Duration = Duration::from_secs(15);
const CODEX_LOGOUT_TIMEOUT: Duration = Duration::from_secs(20);
const CODEX_LOGIN_SESSION_TIMEOUT: Duration = Duration::from_secs(15 * 60);
/// How long a user-cancelled turn waits for in-flight tool dispatches to
/// observe cancellation before it stops waiting for the MCP endpoint.
const CANCELLED_TURN_CLEANUP_GRACE: Duration = Duration::from_secs(5);
const CANCEL_POLL_INTERVAL: Duration = Duration::from_millis(50);
const MAX_JSONL_LINE_BYTES: usize = 1024 * 1024;
const MAX_STDOUT_BYTES: usize = 16 * 1024 * 1024;
const MAX_STDERR_CAPTURE_BYTES: usize = 64 * 1024;
const MAX_PROBE_CAPTURE_BYTES: usize = 16 * 1024;
const MAX_FINAL_TEXT_BYTES: usize = 256 * 1024;
const MAX_TOOL_CALLS: usize = 512;
/// Bytes kept from the start of an oversized JSONL line to identify its item.
const OVERSIZED_LINE_HEAD_BYTES: usize = 4096;
const MAX_TOOL_RESULT_BLOCKS: usize = 64;
const MAX_TOOL_RESULT_IMAGE_BASE64_BYTES: usize = 1024 * 1024;
const CODEX_MCP_BEARER_ENV: &str = "OPENTAKE_CODEX_MCP_BEARER_TOKEN";
const CODEX_CLEANUP_RESERVE: Duration = Duration::from_secs(2);

#[derive(Clone)]
struct LoginController {
    id: u64,
    cancel: Arc<AtomicBool>,
    completion: Arc<LoginCompletion>,
}

#[derive(Default)]
struct LoginCompletion {
    done: AtomicBool,
    notify: tokio::sync::Notify,
}

impl LoginCompletion {
    fn finish(&self) {
        self.done.store(true, Ordering::Release);
        self.notify.notify_waiters();
    }

    async fn wait_until(&self, deadline: tokio::time::Instant) -> Result<(), String> {
        loop {
            let notified = self.notify.notified();
            if self.done.load(Ordering::Acquire) {
                return Ok(());
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                return Err("Codex login cleanup timed out".to_string());
            }
        }
    }
}

#[derive(Default)]
pub struct CodexAuthState {
    login_process: Mutex<Option<LoginController>>,
    next_login_id: AtomicU64,
}

impl Drop for CodexAuthState {
    fn drop(&mut self) {
        if let Ok(process) = self.login_process.get_mut() {
            if let Some(controller) = process.take() {
                controller.cancel.store(true, Ordering::Release);
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ExecutableIdentity {
    canonical_path: PathBuf,
    byte_len: u64,
    modified: Option<SystemTime>,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
}

impl ExecutableIdentity {
    fn capture(path: &Path) -> Option<Self> {
        let canonical_path = std::fs::canonicalize(path).ok()?;
        let metadata = std::fs::metadata(path).ok()?;
        if !metadata.is_file() {
            return None;
        }
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        Some(Self {
            canonical_path,
            byte_len: metadata.len(),
            modified: metadata.modified().ok(),
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
        })
    }

    fn is_current(&self, path: &Path) -> bool {
        Self::capture(path).as_ref() == Some(self)
    }
}

#[derive(Clone, Debug)]
struct VerifiedCodex {
    path: PathBuf,
    version: String,
    identity: ExecutableIdentity,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CodexAuthStatus {
    pub available: bool,
    pub authenticated: bool,
    pub auth_method: Option<String>,
    pub version: Option<String>,
    pub login_in_progress: bool,
    pub message: String,
}

impl CodexAuthStatus {
    fn unavailable() -> Self {
        Self {
            available: false,
            authenticated: false,
            auth_method: None,
            version: None,
            login_in_progress: false,
            message: "Official Codex CLI was not found".into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodexTurnError {
    Cancelled,
    Unavailable,
    IncompatibleCli,
    NotAuthenticated,
    McpStart,
    StrictConfigRejected,
    Timeout,
    Protocol,
    /// Codex reported that the turn failed (`turn.failed`), which covers
    /// expired sign-ins.
    ProviderFailed,
    /// The Codex process could not be started, lost its stdin, or exited
    /// unsuccessfully without reporting a turn failure.
    CliFailed,
}

#[derive(Debug)]
pub struct CodexTurnOutput {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
}

pub(crate) struct CodexTurnContext {
    pub dispatcher: Arc<Dispatcher>,
    pub registry: Arc<RwLock<PluginRegistry>>,
    pub gate: Arc<dyn ChatTurnGate>,
    pub cancel: Arc<AtomicBool>,
}

/// Return candidate locations without trusting a packaged app's truncated PATH
/// alone. `OPENTAKE_CODEX` is an explicit administrator/developer override.
fn candidate_paths() -> Vec<PathBuf> {
    let executable = if cfg!(windows) { "codex.exe" } else { "codex" };
    let mut candidates = Vec::new();

    if let Some(explicit) = std::env::var_os("OPENTAKE_CODEX") {
        candidates.push(PathBuf::from(explicit));
    }
    if let Some(path) = std::env::var_os("PATH") {
        candidates.extend(std::env::split_paths(&path).map(|dir| dir.join(executable)));
    }

    for dir in [
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/opt/local/bin",
        "/usr/bin",
    ] {
        candidates.push(PathBuf::from(dir).join(executable));
    }

    if let Some(home) = home_dir() {
        candidates.push(home.join(".local/bin").join(executable));
        candidates.push(home.join(".volta/bin").join(executable));
        candidates.push(home.join(".cargo/bin").join(executable));

        let nvm_nodes = home.join(".nvm/versions/node");
        if let Ok(entries) = std::fs::read_dir(nvm_nodes) {
            let mut versions = entries
                .flatten()
                .map(|entry| entry.path())
                .collect::<Vec<_>>();
            versions.sort_by(|a, b| b.file_name().cmp(&a.file_name()));
            candidates.extend(
                versions
                    .into_iter()
                    .map(|version| version.join("bin").join(executable)),
            );
        }
    }

    #[cfg(windows)]
    if let Some(local_app_data) = std::env::var_os("LOCALAPPDATA") {
        candidates.push(
            PathBuf::from(local_app_data)
                .join("Programs/OpenAI/Codex/bin")
                .join(executable),
        );
    }

    let mut seen = std::collections::HashSet::new();
    candidates.retain(|path| seen.insert(path.clone()));
    candidates
}

fn home_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        std::env::var_os("USERPROFILE").map(PathBuf::from)
    }
    #[cfg(not(windows))]
    {
        std::env::var_os("HOME").map(PathBuf::from)
    }
}

/// The `(major, minor, patch)` of a `codex-cli X.Y.Z[-pre][+build]` banner.
fn codex_version_triple(version: &str) -> Option<(u64, u64, u64)> {
    let raw = version.strip_prefix("codex-cli ")?;
    let core = raw.split(['-', '+']).next().unwrap_or(raw);
    let mut parts = core.split('.');
    let parsed = (
        parts.next()?.parse::<u64>().ok()?,
        parts.next()?.parse::<u64>().ok()?,
        parts.next()?.parse::<u64>().ok()?,
    );
    parts.next().is_none().then_some(parsed)
}

fn supported_codex_version(version: &str) -> bool {
    codex_version_triple(version).is_some_and(|triple| triple >= MINIMUM_CODEX_VERSION)
}

fn parsed_codex_version(stdout: &[u8]) -> Option<String> {
    let version = String::from_utf8_lossy(stdout).trim().to_string();
    version.starts_with("codex-cli ").then_some(version)
}

fn parse_login_status(text: &str) -> (bool, Option<String>) {
    let normalized = text.trim();
    let lower = normalized.to_ascii_lowercase();
    if !lower.contains("logged in") || lower.contains("not logged in") {
        return (false, None);
    }
    let reported_method = normalized
        .split_once("using")
        .map(|(_, value)| value.trim().trim_end_matches('.'));
    let method = match reported_method.map(str::to_ascii_lowercase).as_deref() {
        Some("chatgpt") => Some("ChatGPT".to_string()),
        Some("an api key" | "api key") => Some("API key".to_string()),
        _ => None,
    };
    (true, method)
}

fn redacted_url(raw: &str) -> String {
    let Ok(mut url) = reqwest::Url::parse(raw) else {
        return "[redacted-url]".to_string();
    };
    if url.set_username("").is_err() || url.set_password(None).is_err() {
        return "[redacted-url]".to_string();
    }
    url.set_query(None);
    url.set_fragment(None);
    url.to_string()
}

fn redacted_inline_bytes(encoded: &str) -> Value {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .unwrap_or_else(|_| encoded.as_bytes().to_vec());
    let digest = Sha256::digest(&bytes);
    serde_json::json!({
        "byteLength": bytes.len(),
        "sha256": format!("{digest:x}"),
        "redacted": true,
    })
}

fn redacted_tool_args(tool_name: &str, args: Value) -> Value {
    let leaf_name = tool_name.rsplit(['.', '/']).next().unwrap_or(tool_name);
    if leaf_name != "import_media" && !leaf_name.ends_with("__import_media") {
        return args;
    }
    let Value::Object(mut args) = args else {
        return serde_json::json!({ "redacted": true });
    };
    let Some(source) = args.get_mut("source") else {
        return Value::Object(args);
    };
    let Value::Object(source) = source else {
        *source = serde_json::json!({ "redacted": true });
        return Value::Object(args);
    };
    if let Some(url) = source.get_mut("url") {
        *url = match url.as_str() {
            Some(raw) => Value::String(redacted_url(raw)),
            None => Value::String("[redacted-url]".to_string()),
        };
    }
    if let Some(bytes) = source.get_mut("bytes") {
        *bytes = match bytes.as_str() {
            Some(encoded) => redacted_inline_bytes(encoded),
            None => serde_json::json!({ "redacted": true }),
        };
    }
    Value::Object(args)
}

fn login_is_running(state: &CodexAuthState) -> Result<bool, String> {
    let mut process = state.login_process.lock().map_err(|e| e.to_string())?;
    let Some(controller) = process.as_ref() else {
        return Ok(false);
    };
    if controller.completion.done.load(Ordering::Acquire) {
        process.take();
        Ok(false)
    } else {
        Ok(true)
    }
}

fn remove_login_controller(state: &CodexAuthState, id: u64) -> Result<(), String> {
    let mut process = state.login_process.lock().map_err(|e| e.to_string())?;
    if process
        .as_ref()
        .is_some_and(|controller| controller.id == id)
    {
        process.take();
    }
    Ok(())
}

async fn cancel_login_until(
    state: &CodexAuthState,
    deadline: tokio::time::Instant,
) -> Result<(), String> {
    let controller = state
        .login_process
        .lock()
        .map_err(|e| e.to_string())?
        .take();
    let Some(controller) = controller else {
        return Ok(());
    };
    controller.cancel.store(true, Ordering::Release);
    controller.completion.wait_until(deadline).await
}

fn auth_probe_error(context: &str, error: CodexTurnError) -> String {
    match error {
        CodexTurnError::Timeout => format!("{context} timed out"),
        CodexTurnError::Cancelled => format!("{context} was cancelled"),
        _ => format!("{context} failed"),
    }
}

async fn auth_status_until(
    state: &CodexAuthState,
    deadline: tokio::time::Instant,
) -> Result<(CodexAuthStatus, Option<VerifiedCodex>), String> {
    let login_in_progress = login_is_running(state)?;
    let cancel = AtomicBool::new(false);
    let codex = match discover_codex_until(&cancel, deadline)
        .await
        .map_err(|error| auth_probe_error("Codex version check", error))?
    {
        CodexDiscovery::Supported(codex) => codex,
        CodexDiscovery::Incompatible { version } => {
            return Ok((CodexDiscovery::incompatible_status(version), None));
        }
        CodexDiscovery::NotFound => return Ok((CodexAuthStatus::unavailable(), None)),
    };

    if login_in_progress {
        return Ok((
            CodexAuthStatus {
                available: true,
                authenticated: false,
                auth_method: None,
                version: Some(codex.version.clone()),
                login_in_progress: true,
                message: "Waiting for official Codex browser login".into(),
            },
            Some(codex),
        ));
    }

    let output = run_verified_probe(&codex, &["login", "status"], &cancel, deadline)
        .await
        .map_err(|error| auth_probe_error("Codex login status check", error))?;
    let combined = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let (authenticated, auth_method) = parse_login_status(&combined);
    Ok((
        CodexAuthStatus {
            available: true,
            authenticated,
            auth_method,
            version: Some(codex.version.clone()),
            login_in_progress: false,
            message: if authenticated {
                "Codex is signed in".into()
            } else {
                "Codex is not signed in".into()
            },
        },
        Some(codex),
    ))
}

struct LoginCompletionGuard(Arc<LoginCompletion>);

impl Drop for LoginCompletionGuard {
    fn drop(&mut self) {
        self.0.finish();
    }
}

async fn run_login_process(
    codex: VerifiedCodex,
    cancel: Arc<AtomicBool>,
    ready: tokio::sync::oneshot::Sender<Result<(), String>>,
) {
    if !codex.identity.is_current(&codex.path) {
        let _ = ready.send(Err("Codex executable changed before login".to_string()));
        return;
    }
    let mut command = tokio::process::Command::new(&codex.path);
    command
        .arg("login")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    opentake_media::process_tree::configure_command(command.as_std_mut());
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(_) => {
            let _ = ready.send(Err("could not start official Codex login".to_string()));
            return;
        }
    };
    let child_id = match child.id() {
        Some(child_id) => child_id,
        None => {
            let _ = child.start_kill();
            let _ = ready.send(Err("could not contain official Codex login".to_string()));
            return;
        }
    };
    let mut tree = match opentake_media::process_tree::ProcessTree::attach(child_id) {
        Ok(tree) => tree,
        Err(_) => {
            let _ = child.start_kill();
            let cleanup_deadline = tokio::time::Instant::now() + CODEX_CLEANUP_RESERVE;
            let _ = tokio::time::timeout_at(cleanup_deadline, child.wait()).await;
            let _ = ready.send(Err("could not contain official Codex login".to_string()));
            return;
        }
    };
    if ready.send(Ok(())).is_err() {
        let cleanup_deadline = tokio::time::Instant::now() + CODEX_CLEANUP_RESERVE;
        let _ = terminate_and_reap_until(&mut child, &mut tree, cleanup_deadline).await;
        return;
    }

    let session_deadline = tokio::time::Instant::now() + CODEX_LOGIN_SESSION_TIMEOUT;
    let cancellation = wait_for_cancel(cancel.as_ref());
    tokio::pin!(cancellation);
    let selected = tokio::select! {
        status = child.wait() => status.map_err(|_| CodexTurnError::CliFailed),
        _ = &mut cancellation => Err(CodexTurnError::Cancelled),
        _ = tokio::time::sleep_until(session_deadline) => Err(CodexTurnError::Timeout),
    };
    match selected {
        Ok(_) => {
            if tree.terminate().is_ok() {
                tree.disarm();
            }
        }
        Err(_) => {
            let cleanup_deadline = tokio::time::Instant::now() + CODEX_CLEANUP_RESERVE;
            let _ = terminate_and_reap_until(&mut child, &mut tree, cleanup_deadline).await;
        }
    }
}

async fn start_login_until(
    state: &CodexAuthState,
    codex: VerifiedCodex,
    deadline: tokio::time::Instant,
    activity: crate::updater::ActivityLease,
) -> Result<(), String> {
    let id = state.next_login_id.fetch_add(1, Ordering::AcqRel) + 1;
    let cancel = Arc::new(AtomicBool::new(false));
    let completion = Arc::new(LoginCompletion::default());
    let controller = LoginController {
        id,
        cancel: cancel.clone(),
        completion: completion.clone(),
    };
    {
        let mut process = state.login_process.lock().map_err(|e| e.to_string())?;
        if process
            .as_ref()
            .is_some_and(|active| !active.completion.done.load(Ordering::Acquire))
        {
            return Err("Codex login is already in progress".to_string());
        }
        *process = Some(controller.clone());
    }

    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
    let thread_completion = completion.clone();
    let spawn = std::thread::Builder::new()
        .name("codex-login".to_string())
        .spawn(move || {
            // The command returns after startup while the browser login child
            // keeps running. Let the worker own update admission until the
            // contained child has actually exited and been reaped.
            let _activity = activity;
            let _guard = LoginCompletionGuard(thread_completion);
            let runtime = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(_) => {
                    let _ = ready_tx.send(Err("could not start Codex login runtime".to_string()));
                    return;
                }
            };
            runtime.block_on(run_login_process(codex, cancel, ready_tx));
        });
    if spawn.is_err() {
        completion.finish();
        remove_login_controller(state, id)?;
        return Err("could not start Codex login worker".to_string());
    }

    match tokio::time::timeout_at(deadline, ready_rx).await {
        Ok(Ok(Ok(()))) => Ok(()),
        Ok(Ok(Err(error))) => {
            let _ = completion.wait_until(deadline).await;
            remove_login_controller(state, id)?;
            Err(error)
        }
        Ok(Err(_)) => {
            let _ = completion.wait_until(deadline).await;
            remove_login_controller(state, id)?;
            Err("Codex login worker stopped before startup".to_string())
        }
        Err(_) => {
            controller.cancel.store(true, Ordering::Release);
            Err("Codex login startup timed out".to_string())
        }
    }
}

#[tauri::command]
pub async fn codex_auth_status(
    state: State<'_, CodexAuthState>,
) -> Result<CodexAuthStatus, String> {
    let deadline = tokio::time::Instant::now() + CODEX_AUTH_TIMEOUT;
    auth_status_until(&state, deadline)
        .await
        .map(|(status, _)| status)
}

#[tauri::command]
pub async fn codex_login_start(
    state: State<'_, CodexAuthState>,
    admission: State<'_, crate::updater::InstallAdmissionGate>,
) -> Result<CodexAuthStatus, String> {
    let activity = crate::updater::begin_mutating_activity(&admission)?;
    let deadline = tokio::time::Instant::now() + CODEX_AUTH_TIMEOUT;
    let (current, codex) = auth_status_until(&state, deadline).await?;
    if current.authenticated || current.login_in_progress {
        return Ok(current);
    }
    let Some(codex) = codex else {
        // Not found or too old: `current` already says which.
        return Ok(current);
    };
    start_login_until(&state, codex.clone(), deadline, activity).await?;
    Ok(CodexAuthStatus {
        available: true,
        authenticated: false,
        auth_method: None,
        version: Some(codex.version),
        login_in_progress: true,
        message: "Waiting for official Codex browser login".into(),
    })
}

#[tauri::command]
pub async fn codex_login_cancel(
    state: State<'_, CodexAuthState>,
) -> Result<CodexAuthStatus, String> {
    let deadline = tokio::time::Instant::now() + CODEX_AUTH_TIMEOUT;
    cancel_login_until(&state, deadline).await?;
    auth_status_until(&state, deadline)
        .await
        .map(|(status, _)| status)
}

#[tauri::command]
pub async fn codex_logout(
    state: State<'_, CodexAuthState>,
    admission: State<'_, crate::updater::InstallAdmissionGate>,
) -> Result<CodexAuthStatus, String> {
    let _activity = crate::updater::begin_mutating_activity(&admission)?;
    let deadline = tokio::time::Instant::now() + CODEX_LOGOUT_TIMEOUT;
    cancel_login_until(&state, deadline).await?;
    let cancel = AtomicBool::new(false);
    let codex = match discover_codex_until(&cancel, deadline)
        .await
        .map_err(|error| auth_probe_error("Codex version check", error))?
    {
        CodexDiscovery::Supported(codex) => codex,
        CodexDiscovery::Incompatible { version } => {
            return Ok(CodexDiscovery::incompatible_status(version));
        }
        CodexDiscovery::NotFound => return Ok(CodexAuthStatus::unavailable()),
    };
    let output = run_verified_probe(&codex, &["logout"], &cancel, deadline)
        .await
        .map_err(|error| auth_probe_error("Codex logout", error))?;
    if !output.status.success() {
        return Err("Codex logout failed".to_string());
    }
    auth_status_until(&state, deadline)
        .await
        .map(|(status, _)| status)
}

#[derive(Debug, PartialEq, Eq)]
enum ExecEvent {
    Ignored,
    AgentMessage(String),
    ToolChanged(String),
    /// A tool call beyond [`MAX_TOOL_CALLS`] whose display copy is not kept.
    ToolOmitted(String),
    TurnFailed,
}

/// Build the bounded display copy of a Codex MCP tool result for the chat
/// history. The model already received the full result through MCP, so the
/// display limits degrade the copy (truncating or replacing blocks with a
/// note) instead of failing the turn. Only structurally malformed results are
/// protocol errors.
fn normalized_codex_tool_result(item: &Value, failed: bool) -> Result<Value, CodexTurnError> {
    if failed {
        return Ok(serde_json::json!({ "status": "failed" }));
    }
    let Some(content) = item.get("result").and_then(|result| result.get("content")) else {
        return Ok(serde_json::json!({ "status": "completed" }));
    };
    let content = content.as_array().ok_or(CodexTurnError::Protocol)?;
    if content.is_empty() {
        return Ok(serde_json::json!({ "status": "completed" }));
    }

    // Reserve the last block for the omission note when there are too many.
    let kept = if content.len() > MAX_TOOL_RESULT_BLOCKS {
        MAX_TOOL_RESULT_BLOCKS - 1
    } else {
        content.len()
    };
    let mut blocks = Vec::with_capacity(kept + 1);
    for content_block in content {
        let block = normalized_codex_content_block(content_block)?;
        if blocks.len() < kept {
            blocks.push(block);
        }
    }
    if content.len() > kept {
        blocks.push(Block::text(format!(
            "[OpenTake: {} more tool result blocks were omitted from the chat history; the model received the full result.]",
            content.len() - kept
        )));
    }
    Ok(serde_json::json!({ "content": blocks }))
}

fn normalized_codex_content_block(content_block: &Value) -> Result<Block, CodexTurnError> {
    let block_type = content_block
        .get("type")
        .and_then(Value::as_str)
        .ok_or(CodexTurnError::Protocol)?;
    match block_type {
        "text" => {
            let text = content_block
                .get("text")
                .and_then(Value::as_str)
                .ok_or(CodexTurnError::Protocol)?;
            Ok(Block::text(bounded_tool_result_text(text)))
        }
        "image" => {
            let base64 = content_block
                .get("data")
                .and_then(Value::as_str)
                .ok_or(CodexTurnError::Protocol)?;
            let media_type = content_block
                .get("mimeType")
                .and_then(Value::as_str)
                .ok_or(CodexTurnError::Protocol)?;
            if base64.is_empty()
                || base64.len() > MAX_TOOL_RESULT_IMAGE_BASE64_BYTES
                || !matches!(
                    media_type,
                    "image/png" | "image/jpeg" | "image/webp" | "image/gif"
                )
                || base64::engine::general_purpose::STANDARD
                    .decode(base64)
                    .is_err()
            {
                return Ok(Block::text(format!(
                    "[OpenTake: an image ({}, {} base64 bytes) was omitted from the chat history.]",
                    display_label(media_type),
                    base64.len()
                )));
            }
            Ok(Block::image(base64, media_type))
        }
        other => Ok(Block::text(format!(
            "[OpenTake: a {} block was omitted from the chat history.]",
            display_label(other)
        ))),
    }
}

/// Truncate a text block to the display limit at a UTF-8 boundary and say so,
/// naming the original length and hash so the copy stays truthful.
fn bounded_tool_result_text(text: &str) -> String {
    if text.len() <= MAX_FINAL_TEXT_BYTES {
        return text.to_owned();
    }
    let digest = Sha256::digest(text.as_bytes());
    let note = format!(
        "\n[OpenTake: truncated for the chat history; the original text was {} bytes, sha256 {digest:x}. The model received the full result.]",
        text.len(),
    );
    let keep = text.floor_char_boundary(MAX_FINAL_TEXT_BYTES.saturating_sub(note.len()));
    let mut bounded = String::with_capacity(keep + note.len());
    bounded.push_str(&text[..keep]);
    bounded.push_str(&note);
    bounded
}

/// Truncate an oversized final reply at a UTF-8 boundary and say so.
fn bounded_reply_text(text: &str) -> String {
    if text.len() <= MAX_FINAL_TEXT_BYTES {
        return text.to_owned();
    }
    let note = format!(
        "\n\n[OpenTake: the reply was truncated for display; it was {} bytes.]",
        text.len()
    );
    let keep = text.floor_char_boundary(MAX_FINAL_TEXT_BYTES.saturating_sub(note.len()));
    format!("{}{note}", &text[..keep])
}

/// A short, printable rendering of an untrusted type label for a note.
fn display_label(label: &str) -> String {
    let clean = label
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '/' | '-' | '_' | '.' | '+'))
        .take(64)
        .collect::<String>();
    if clean.is_empty() {
        "unknown".to_string()
    } else {
        clean
    }
}

fn parse_exec_event(
    line: &str,
    tool_calls: &mut HashMap<String, ToolCall>,
) -> Result<ExecEvent, CodexTurnError> {
    let event: Value = serde_json::from_str(line).map_err(|_| CodexTurnError::Protocol)?;
    let event_type = event
        .get("type")
        .and_then(Value::as_str)
        .ok_or(CodexTurnError::Protocol)?;
    if event_type == "turn.failed" {
        return Ok(ExecEvent::TurnFailed);
    }
    if event_type != "item.started" && event_type != "item.completed" {
        return Ok(ExecEvent::Ignored);
    }
    let item = event.get("item").ok_or(CodexTurnError::Protocol)?;
    match item.get("type").and_then(Value::as_str) {
        Some("agent_message") if event_type == "item.completed" => {
            let text = item
                .get("text")
                .and_then(Value::as_str)
                .ok_or(CodexTurnError::Protocol)?;
            Ok(ExecEvent::AgentMessage(bounded_reply_text(text)))
        }
        Some("mcp_tool_call") => {
            let id = item
                .get("id")
                .and_then(Value::as_str)
                .ok_or(CodexTurnError::Protocol)?
                .to_string();
            if !tool_calls.contains_key(&id) && tool_calls.len() >= MAX_TOOL_CALLS {
                // The call still ran; only its chat-history copy is dropped.
                return Ok(ExecEvent::ToolOmitted(id));
            }
            let existed = tool_calls.contains_key(&id);
            let previous_result = tool_calls
                .get(&id)
                .and_then(|call| call.result.as_ref())
                .cloned();
            let name = item
                .get("tool")
                .and_then(Value::as_str)
                .ok_or(CodexTurnError::Protocol)?
                .to_string();
            let args = item
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));
            let args = redacted_tool_args(&name, args);
            let failed = if event_type == "item.completed" {
                let result_error = match item.get("result").and_then(|result| result.get("isError"))
                {
                    Some(Value::Bool(value)) => *value,
                    Some(_) => return Err(CodexTurnError::Protocol),
                    None => false,
                };
                item.get("error").is_some_and(|value| !value.is_null()) || result_error
            } else {
                false
            };
            let mut call = tool_calls
                .remove(&id)
                .unwrap_or_else(|| ToolCall::request(id.clone(), name, args));
            if event_type == "item.completed" {
                call.is_error = Some(failed);
                call.result = Some(normalized_codex_tool_result(item, failed)?);
            }
            let changed = !existed || previous_result.as_ref() != call.result.as_ref();
            tool_calls.insert(id.clone(), call);
            Ok(if changed {
                ExecEvent::ToolChanged(id)
            } else {
                ExecEvent::Ignored
            })
        }
        _ => Ok(ExecEvent::Ignored),
    }
}

fn push_config(args: &mut Vec<OsString>, value: impl Into<OsString>) {
    args.push(OsString::from("-c"));
    args.push(value.into());
}

fn loopback_no_proxy(existing: Option<OsString>) -> OsString {
    let required = "127.0.0.1,localhost,::1,[::1]";
    match existing.filter(|value| !value.is_empty()) {
        Some(existing) => {
            let mut combined = existing;
            combined.push(",");
            combined.push(required);
            combined
        }
        None => OsString::from(required),
    }
}

fn build_exec_args(endpoint_url: &str, isolated_cwd: &Path) -> Vec<OsString> {
    let mut args = vec![
        OsString::from("exec"),
        OsString::from("--strict-config"),
        OsString::from("--json"),
        OsString::from("--ephemeral"),
        OsString::from("--ignore-user-config"),
        OsString::from("--ignore-rules"),
        OsString::from("--sandbox"),
        OsString::from("read-only"),
        OsString::from("--skip-git-repo-check"),
        OsString::from("--color"),
        OsString::from("never"),
        OsString::from("-C"),
        isolated_cwd.as_os_str().to_owned(),
    ];
    push_config(
        &mut args,
        format!("mcp_servers.opentake.url=\"{endpoint_url}\""),
    );
    push_config(
        &mut args,
        format!("mcp_servers.opentake.bearer_token_env_var=\"{CODEX_MCP_BEARER_ENV}\""),
    );
    for config in [
        "mcp_servers.opentake.required=true",
        "mcp_servers.opentake.default_tools_approval_mode=\"approve\"",
        "approval_policy=\"never\"",
        "agents.enabled=false",
        "skills.include_instructions=false",
        "apps._default.enabled=false",
        "features.apps=false",
        "features.auth_elicitation=false",
        "features.browser_use=false",
        "features.in_app_browser=false",
        "features.code_mode_host=false",
        "features.computer_use=false",
        "features.goals=false",
        "features.hooks=false",
        "features.image_generation=false",
        "features.memories=false",
        "features.multi_agent=false",
        "features.personality=false",
        "features.plugins=false",
        "features.remote_plugin=false",
        "features.request_permissions_tool=false",
        "features.shell_tool=false",
        "features.skill_search=false",
        "features.tool_call_mcp_elicitation=false",
        "features.tool_suggest=false",
        "features.unified_exec=false",
        "features.workspace_dependencies=false",
        "web_search=\"disabled\"",
    ] {
        push_config(&mut args, config);
    }
    args.push(OsString::from("-"));
    args
}

#[derive(Debug, PartialEq, Eq)]
enum JsonlLine {
    Text(String),
    /// A line longer than [`MAX_JSONL_LINE_BYTES`], drained without being
    /// buffered; carries its byte length and whether it was the reply.
    Oversized {
        len: usize,
        agent_message: bool,
    },
}

/// Partial-line state kept outside the read future, so dropping the future
/// (it races the cancel poll) never loses or misframes a line.
#[derive(Default)]
struct JsonlLineBuffer {
    bytes: Vec<u8>,
    /// Bytes drained so far from a line already known to be oversized.
    oversized: Option<usize>,
    /// The first [`OVERSIZED_LINE_HEAD_BYTES`] of that line.
    oversized_head: Vec<u8>,
}

/// Whether the head of an oversized JSONL line is a completed `agent_message`
/// item (Codex serializes the item type before its text). Only the event's
/// own `type` and its `item.type` count, never a `"type"` inside a string or
/// another nested value such as tool arguments.
fn oversized_line_is_agent_message(head: &[u8]) -> bool {
    let (event_type, item_type) = jsonl_head_types(head);
    event_type.as_deref() == Some("item.completed") && item_type.as_deref() == Some("agent_message")
}

/// Read the top-level `type` and the `item.type` string values from the
/// first bytes of a JSONL event, which may end in the middle of a token.
fn jsonl_head_types(head: &[u8]) -> (Option<String>, Option<String>) {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Scope {
        Root,
        Item,
        Other,
    }
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Expect {
        Key,
        Colon,
        Value,
        Comma,
    }
    struct Object {
        scope: Scope,
        expect: Expect,
        key: String,
    }
    // `None` entries are arrays.
    let mut stack: Vec<Option<Object>> = Vec::new();
    let mut event_type = None;
    let mut item_type = None;
    let mut bytes = head.iter().copied().peekable();
    while let Some(byte) = bytes.next() {
        match byte {
            b'"' => {
                let mut text = Vec::new();
                let mut closed = false;
                while let Some(byte) = bytes.next() {
                    match byte {
                        b'"' => {
                            closed = true;
                            break;
                        }
                        b'\\' => match bytes.next() {
                            Some(escaped) => text.push(escaped),
                            None => break,
                        },
                        _ => text.push(byte),
                    }
                }
                if !closed {
                    break;
                }
                let text = String::from_utf8_lossy(&text).into_owned();
                if let Some(Some(object)) = stack.last_mut() {
                    match object.expect {
                        Expect::Key => {
                            object.key = text;
                            object.expect = Expect::Colon;
                        }
                        Expect::Value => {
                            if object.key == "type" {
                                match object.scope {
                                    Scope::Root => event_type = Some(text),
                                    Scope::Item => item_type = Some(text),
                                    Scope::Other => {}
                                }
                            }
                            object.expect = Expect::Comma;
                        }
                        Expect::Colon | Expect::Comma => break,
                    }
                }
            }
            b'{' => {
                let scope = match stack.last_mut() {
                    None => Scope::Root,
                    Some(Some(parent)) => {
                        let scope = if parent.scope == Scope::Root && parent.key == "item" {
                            Scope::Item
                        } else {
                            Scope::Other
                        };
                        parent.expect = Expect::Comma;
                        scope
                    }
                    Some(None) => Scope::Other,
                };
                stack.push(Some(Object {
                    scope,
                    expect: Expect::Key,
                    key: String::new(),
                }));
            }
            b'[' => {
                if stack.is_empty() {
                    break;
                }
                if let Some(Some(parent)) = stack.last_mut() {
                    parent.expect = Expect::Comma;
                }
                stack.push(None);
            }
            b'}' | b']' => {
                stack.pop();
                if stack.is_empty() {
                    break;
                }
            }
            b':' => {
                if let Some(Some(object)) = stack.last_mut() {
                    object.expect = Expect::Value;
                }
            }
            b',' => {
                if let Some(Some(object)) = stack.last_mut() {
                    object.expect = Expect::Key;
                }
            }
            byte if byte.is_ascii_whitespace() => {}
            _ => {
                // A number, `true`, `false` or `null`: consume the scalar.
                while bytes.peek().is_some_and(|next| {
                    !matches!(next, b',' | b'}' | b']') && !next.is_ascii_whitespace()
                }) {
                    bytes.next();
                }
                if let Some(Some(object)) = stack.last_mut() {
                    object.expect = Expect::Comma;
                }
            }
        }
    }
    (event_type, item_type)
}

/// Cancel-safe: bytes consumed from `reader` stay in `buffer` until a full
/// line is returned, so dropping this future inside `select!` and calling it
/// again with the same buffer resumes the partial line instead of losing it.
async fn read_bounded_line<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    buffer: &mut JsonlLineBuffer,
) -> Result<Option<JsonlLine>, CodexTurnError> {
    loop {
        let available = reader
            .fill_buf()
            .await
            .map_err(|_| CodexTurnError::Protocol)?;
        if available.is_empty() {
            if buffer.bytes.is_empty() && buffer.oversized.is_none() {
                return Ok(None);
            }
            break;
        }
        let newline = available.iter().position(|byte| *byte == b'\n');
        let take = newline.unwrap_or(available.len());
        if let Some(skipped) = buffer.oversized.as_mut() {
            *skipped = skipped.saturating_add(take);
            if *skipped > MAX_STDOUT_BYTES {
                // An unterminated line cannot outlast the stdout budget.
                return Err(CodexTurnError::Protocol);
            }
        } else if buffer.bytes.len().saturating_add(take) > MAX_JSONL_LINE_BYTES {
            // Drain the rest of the line without buffering it; the caller
            // decides whether an unreadable line is fatal. The head of the line
            // is kept so the caller can tell which item it was.
            buffer.oversized = Some(buffer.bytes.len().saturating_add(take));
            let mut head = std::mem::take(&mut buffer.bytes);
            head.extend_from_slice(&available[..take.min(OVERSIZED_LINE_HEAD_BYTES)]);
            head.truncate(OVERSIZED_LINE_HEAD_BYTES);
            buffer.oversized_head = head;
        } else {
            buffer.bytes.extend_from_slice(&available[..take]);
        }
        reader.consume(take + usize::from(newline.is_some()));
        if newline.is_some() {
            break;
        }
    }
    if let Some(len) = buffer.oversized.take() {
        let head = std::mem::take(&mut buffer.oversized_head);
        return Ok(Some(JsonlLine::Oversized {
            len,
            agent_message: oversized_line_is_agent_message(&head),
        }));
    }
    if buffer.bytes.last() == Some(&b'\r') {
        buffer.bytes.pop();
    }
    String::from_utf8(std::mem::take(&mut buffer.bytes))
        .map(|line| Some(JsonlLine::Text(line)))
        .map_err(|_| CodexTurnError::Protocol)
}

async fn drain_bounded_capture<R: AsyncRead + Unpin>(mut reader: R, limit: usize) -> Vec<u8> {
    let mut captured = Vec::with_capacity(limit);
    let mut chunk = [0_u8; 8192];
    loop {
        match reader.read(&mut chunk).await {
            Ok(0) | Err(_) => return captured,
            Ok(read) => {
                let remaining = limit.saturating_sub(captured.len());
                captured.extend_from_slice(&chunk[..read.min(remaining)]);
            }
        }
    }
}

async fn drain_stderr<R: AsyncRead + Unpin>(reader: R) -> Vec<u8> {
    drain_bounded_capture(reader, MAX_STDERR_CAPTURE_BYTES).await
}

fn work_deadline(deadline: tokio::time::Instant) -> tokio::time::Instant {
    deadline
        .checked_sub(CODEX_CLEANUP_RESERVE)
        .unwrap_or(deadline)
}

async fn join_capture_until(
    task: Option<tokio::task::JoinHandle<Vec<u8>>>,
    deadline: tokio::time::Instant,
) -> Result<Vec<u8>, CodexTurnError> {
    let Some(mut task) = task else {
        return Ok(Vec::new());
    };
    match tokio::time::timeout_at(deadline, &mut task).await {
        Ok(Ok(bytes)) => Ok(bytes),
        Ok(Err(_)) => Ok(Vec::new()),
        Err(_) => {
            task.abort();
            Err(CodexTurnError::Timeout)
        }
    }
}

async fn terminate_and_reap_until(
    child: &mut tokio::process::Child,
    tree: &mut opentake_media::process_tree::ProcessTree,
    deadline: tokio::time::Instant,
) -> Result<ExitStatus, CodexTurnError> {
    let _ = tree.terminate();
    let _ = child.start_kill();
    let result = tokio::time::timeout_at(deadline, child.wait()).await;
    tree.disarm();
    match result {
        Ok(Ok(status)) => Ok(status),
        Ok(Err(_)) => Err(CodexTurnError::CliFailed),
        Err(_) => Err(CodexTurnError::Timeout),
    }
}

struct ProbeOutput {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

async fn wait_for_cancel(cancel: &AtomicBool) {
    let mut poll = tokio::time::interval(CANCEL_POLL_INTERVAL);
    poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        poll.tick().await;
        if cancel.load(Ordering::Acquire) {
            return;
        }
    }
}

async fn run_probe(
    path: &Path,
    args: &[&str],
    cancel: &AtomicBool,
    deadline: tokio::time::Instant,
) -> Result<ProbeOutput, CodexTurnError> {
    let mut command = tokio::process::Command::new(path);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    opentake_media::process_tree::configure_command(command.as_std_mut());
    let mut child = command.spawn().map_err(|_| CodexTurnError::CliFailed)?;
    let child_id = child.id().ok_or(CodexTurnError::CliFailed)?;
    let mut tree = match opentake_media::process_tree::ProcessTree::attach(child_id) {
        Ok(tree) => tree,
        Err(_) => {
            let _ = child.start_kill();
            let _ = tokio::time::timeout_at(deadline, child.wait()).await;
            return Err(CodexTurnError::CliFailed);
        }
    };
    let stdout_task = child
        .stdout
        .take()
        .map(|stdout| tokio::spawn(drain_bounded_capture(stdout, MAX_PROBE_CAPTURE_BYTES)));
    let stderr_task = child
        .stderr
        .take()
        .map(|stderr| tokio::spawn(drain_bounded_capture(stderr, MAX_PROBE_CAPTURE_BYTES)));
    let cancellation = wait_for_cancel(cancel);
    tokio::pin!(cancellation);
    let deadline_wait = tokio::time::sleep_until(work_deadline(deadline));
    tokio::pin!(deadline_wait);
    let selected = tokio::select! {
        status = child.wait() => status.map_err(|_| CodexTurnError::CliFailed),
        _ = &mut cancellation => Err(CodexTurnError::Cancelled),
        _ = &mut deadline_wait => Err(CodexTurnError::Timeout),
    };
    let status = match selected {
        Ok(status) => {
            // Kill descendants that inherited either capture pipe before the
            // drain tasks are joined.
            let _ = tree.terminate();
            tree.disarm();
            status
        }
        Err(error) => {
            let _ = terminate_and_reap_until(&mut child, &mut tree, deadline).await;
            let _ = join_capture_until(stdout_task, deadline).await;
            let _ = join_capture_until(stderr_task, deadline).await;
            return Err(error);
        }
    };
    let stdout = join_capture_until(stdout_task, deadline).await?;
    let stderr = join_capture_until(stderr_task, deadline).await?;
    Ok(ProbeOutput {
        status,
        stdout,
        stderr,
    })
}

/// What Codex CLI discovery found. An installed but too old CLI is reported
/// as such, so the user is asked to update it rather than to install Codex.
enum CodexDiscovery {
    Supported(VerifiedCodex),
    /// No supported candidate; `version` is the newest parsable one found.
    Incompatible {
        version: String,
    },
    NotFound,
}

impl CodexDiscovery {
    fn incompatible_status(version: String) -> CodexAuthStatus {
        CodexAuthStatus {
            available: false,
            authenticated: false,
            auth_method: None,
            message: format!(
                "Official Codex CLI {version} is too old; update it to {}.{}.{} or newer",
                MINIMUM_CODEX_VERSION.0, MINIMUM_CODEX_VERSION.1, MINIMUM_CODEX_VERSION.2
            ),
            version: Some(version),
            login_in_progress: false,
        }
    }
}

async fn discover_codex_until(
    cancel: &AtomicBool,
    deadline: tokio::time::Instant,
) -> Result<CodexDiscovery, CodexTurnError> {
    discover_codex_among(candidate_paths(), cancel, deadline).await
}

/// Use the first supported candidate; otherwise report the newest version
/// that was found but is too old.
async fn discover_codex_among(
    candidates: Vec<PathBuf>,
    cancel: &AtomicBool,
    deadline: tokio::time::Instant,
) -> Result<CodexDiscovery, CodexTurnError> {
    let mut newest_incompatible: Option<((u64, u64, u64), String)> = None;
    for path in candidates {
        let Some(identity) = ExecutableIdentity::capture(&path) else {
            continue;
        };
        let probe = match run_probe(&path, &["--version"], cancel, deadline).await {
            Ok(probe) => probe,
            Err(CodexTurnError::Cancelled) => return Err(CodexTurnError::Cancelled),
            Err(CodexTurnError::Timeout) => return Err(CodexTurnError::Timeout),
            Err(_) => continue,
        };
        if !probe.status.success() || !identity.is_current(&path) {
            continue;
        }
        let Some(version) = parsed_codex_version(&probe.stdout) else {
            continue;
        };
        if supported_codex_version(&version) {
            return Ok(CodexDiscovery::Supported(VerifiedCodex {
                path,
                version,
                identity,
            }));
        }
        if let Some(triple) = codex_version_triple(&version) {
            if newest_incompatible
                .as_ref()
                .is_none_or(|(newest, _)| triple > *newest)
            {
                newest_incompatible = Some((triple, version));
            }
        }
    }
    Ok(match newest_incompatible {
        Some((_, version)) => CodexDiscovery::Incompatible { version },
        None => CodexDiscovery::NotFound,
    })
}

async fn run_verified_probe(
    codex: &VerifiedCodex,
    args: &[&str],
    cancel: &AtomicBool,
    deadline: tokio::time::Instant,
) -> Result<ProbeOutput, CodexTurnError> {
    if !codex.identity.is_current(&codex.path) {
        return Err(CodexTurnError::Unavailable);
    }
    let output = run_probe(&codex.path, args, cancel, deadline).await?;
    if !codex.identity.is_current(&codex.path) {
        return Err(CodexTurnError::Unavailable);
    }
    Ok(output)
}

async fn write_prompt_with_lifecycle<W: AsyncWrite + Unpin>(
    mut stdin: W,
    prompt: &str,
    endpoint: &opentake_agent::mcp::server::EphemeralMcpEndpoint,
    cancel: &AtomicBool,
    deadline: tokio::time::Instant,
) -> Result<(), CodexTurnError> {
    let write = async {
        stdin.write_all(prompt.as_bytes()).await?;
        stdin.shutdown().await
    };
    tokio::pin!(write);
    let mut poll = tokio::time::interval(CANCEL_POLL_INTERVAL);
    poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let deadline = tokio::time::sleep_until(deadline);
    tokio::pin!(deadline);

    loop {
        tokio::select! {
            result = &mut write => {
                return result.map_err(|_| CodexTurnError::CliFailed);
            }
            _ = poll.tick() => {
                if cancel.load(Ordering::Acquire) {
                    return Err(CodexTurnError::Cancelled);
                }
            }
            _ = endpoint.stopped() => return Err(CodexTurnError::McpStart),
            _ = &mut deadline => return Err(CodexTurnError::Timeout),
        }
    }
}

/// When to stop waiting for the MCP endpoint's tool calls to drain. After a
/// user cancel the dispatches observe their tokens and end promptly, so a
/// straggler gets only a short grace. Internal failures keep the turn
/// deadline, but a Stop pressed while they drain switches to the same short
/// grace. Giving up detaches the endpoint (see
/// `EphemeralMcpEndpoint::close_or_detach`), which keeps the turn's saved
/// error or timeout reply instead of cancelling the whole turn.
async fn endpoint_drain_give_up(
    deadline: tokio::time::Instant,
    user_cancelled: bool,
    cancel: &AtomicBool,
) {
    if user_cancelled {
        tokio::time::sleep_until(endpoint_close_deadline(
            deadline,
            tokio::time::Instant::now(),
            true,
        ))
        .await;
        return;
    }
    tokio::select! {
        () = tokio::time::sleep_until(deadline) => {}
        () = wait_for_cancel(cancel) => {
            tokio::time::sleep_until(endpoint_close_deadline(
                deadline,
                tokio::time::Instant::now(),
                true,
            ))
            .await;
        }
    }
}

/// Deadline for draining the MCP endpoint: the short grace after a user
/// cancel, the turn deadline otherwise.
fn endpoint_close_deadline(
    deadline: tokio::time::Instant,
    now: tokio::time::Instant,
    user_cancelled: bool,
) -> tokio::time::Instant {
    if user_cancelled {
        deadline.min(now + CANCELLED_TURN_CLEANUP_GRACE)
    } else {
        deadline
    }
}

/// Close the endpoint after Codex could not be started, bounded like the
/// normal drain.
async fn close_endpoint_after_failure(
    endpoint: opentake_agent::mcp::server::EphemeralMcpEndpoint,
    cancel: &AtomicBool,
    deadline: tokio::time::Instant,
) {
    let user_cancelled = cancel.load(Ordering::Acquire);
    let _ = endpoint
        .close_or_detach(endpoint_drain_give_up(deadline, user_cancelled, cancel))
        .await;
}

fn strict_config_rejected(stderr: &[u8]) -> bool {
    let text = String::from_utf8_lossy(stderr).to_ascii_lowercase();
    text.contains("strict config")
        || text.contains("unknown configuration")
        || text.contains("unknown config")
        || text.contains("unknown field")
}

async fn consume_exec_stream<R, F>(
    stdout: R,
    endpoint: &opentake_agent::mcp::server::EphemeralMcpEndpoint,
    context: &CodexTurnContext,
    deadline: tokio::time::Instant,
    on_tool_call: &mut F,
) -> StreamEnd
where
    R: AsyncRead + Unpin,
    F: FnMut(ToolCall),
{
    let mut reader = BufReader::new(stdout);
    let mut line_buffer = JsonlLineBuffer::default();
    let mut stream = ExecStreamState::default();
    let mut poll = tokio::time::interval(CANCEL_POLL_INTERVAL);
    poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let deadline = tokio::time::sleep_until(deadline);
    tokio::pin!(deadline);

    loop {
        tokio::select! {
            line = read_bounded_line(&mut reader, &mut line_buffer) => {
                let line = match line {
                    Ok(Some(line)) => line,
                    Ok(None) => break,
                    Err(error) => return StreamEnd::Failed(error),
                };
                match stream.accept_line(line) {
                    Ok(Some(call)) => on_tool_call(call),
                    Ok(None) => {}
                    Err(error) => return StreamEnd::Failed(error),
                }
            }
            _ = poll.tick() => {
                if context.cancel.load(Ordering::Acquire) {
                    return StreamEnd::Failed(CodexTurnError::Cancelled);
                }
            }
            _ = endpoint.stopped() => return StreamEnd::Failed(CodexTurnError::McpStart),
            _ = &mut deadline => return StreamEnd::Failed(CodexTurnError::Timeout),
        }
    }
    StreamEnd::Eof(stream.finish())
}

/// How reading Codex's JSONL stream ended.
enum StreamEnd {
    /// Codex closed stdout; the turn's result, which may still be an error
    /// (no reply). Codex is exiting, so its exit status decides the rest.
    Eof(Result<CodexTurnOutput, CodexTurnError>),
    /// A failure found while Codex still runs; Codex is killed.
    Failed(CodexTurnError),
}

/// Codex JSONL stream state, separated from the process plumbing so a whole
/// stream can be exercised in tests.
#[derive(Default)]
struct ExecStreamState {
    stdout_bytes: usize,
    final_text: Option<String>,
    tool_calls: HashMap<String, ToolCall>,
    omitted_tool_calls: std::collections::HashSet<String>,
    oversized_lines: usize,
}

impl ExecStreamState {
    /// Consume one line and return a tool call whose display copy changed.
    fn accept_line(&mut self, line: JsonlLine) -> Result<Option<ToolCall>, CodexTurnError> {
        let line = match line {
            JsonlLine::Text(line) => line,
            JsonlLine::Oversized { len, agent_message } => {
                self.count_stdout(len)?;
                if agent_message {
                    self.final_text = Some(format!(
                        "[OpenTake: Codex's reply ({len} bytes) was too large to display.]"
                    ));
                } else {
                    self.oversized_lines += 1;
                }
                return Ok(None);
            }
        };
        self.count_stdout(line.len())?;
        match parse_exec_event(&line, &mut self.tool_calls)? {
            ExecEvent::AgentMessage(text) => self.final_text = Some(text),
            ExecEvent::TurnFailed => return Err(CodexTurnError::ProviderFailed),
            ExecEvent::ToolChanged(id) => {
                let call = self.tool_calls.get(&id).ok_or(CodexTurnError::Protocol)?;
                return Ok(Some(call.clone()));
            }
            ExecEvent::ToolOmitted(id) => {
                self.omitted_tool_calls.insert(id);
            }
            ExecEvent::Ignored => {}
        }
        Ok(None)
    }

    fn count_stdout(&mut self, line_len: usize) -> Result<(), CodexTurnError> {
        self.stdout_bytes = self.stdout_bytes.saturating_add(line_len).saturating_add(1);
        if self.stdout_bytes > MAX_STDOUT_BYTES {
            return Err(CodexTurnError::Protocol);
        }
        Ok(())
    }

    fn finish(self) -> Result<CodexTurnOutput, CodexTurnError> {
        let text = self.final_text.filter(|text| !text.trim().is_empty());
        let Some(mut text) = text else {
            return Err(CodexTurnError::Protocol);
        };
        let mut tool_calls = self.tool_calls.into_values().collect::<Vec<_>>();
        if self.oversized_lines > 0 {
            // Codex completes every call it starts, so a call without a
            // completion lost it to a line above the reader's bound.
            for call in &mut tool_calls {
                if call.result.is_none() {
                    call.result = Some(serde_json::json!({
                        "status": "omitted",
                        "content": [Block::text(
                            "[OpenTake: this tool result was too large to keep in the chat history; the model received the full result.]",
                        )],
                    }));
                }
            }
        }
        tool_calls.sort_by(|a, b| a.id.cmp(&b.id));
        if !self.omitted_tool_calls.is_empty() {
            text.push_str(&format!(
                "\n\n_OpenTake kept the first {MAX_TOOL_CALLS} tool calls of this turn in the chat history; {} more were not recorded._",
                self.omitted_tool_calls.len()
            ));
        }
        Ok(CodexTurnOutput { text, tool_calls })
    }
}

pub async fn run_agent_turn<F>(
    context: CodexTurnContext,
    prompt: &str,
    on_tool_call: F,
) -> Result<CodexTurnOutput, CodexTurnError>
where
    F: FnMut(ToolCall),
{
    run_agent_turn_among(candidate_paths(), context, prompt, on_tool_call).await
}

async fn run_agent_turn_among<F>(
    candidates: Vec<PathBuf>,
    context: CodexTurnContext,
    prompt: &str,
    on_tool_call: F,
) -> Result<CodexTurnOutput, CodexTurnError>
where
    F: FnMut(ToolCall),
{
    let deadline = tokio::time::Instant::now() + CODEX_TURN_TIMEOUT;
    let path = match discover_codex_among(candidates, context.cancel.as_ref(), deadline).await? {
        CodexDiscovery::Supported(codex) => codex.path,
        CodexDiscovery::Incompatible { .. } => return Err(CodexTurnError::IncompatibleCli),
        CodexDiscovery::NotFound => return Err(CodexTurnError::Unavailable),
    };
    let output = run_probe(
        &path,
        &["login", "status"],
        context.cancel.as_ref(),
        deadline,
    )
    .await
    .map_err(|error| match error {
        CodexTurnError::Cancelled | CodexTurnError::Timeout => error,
        _ => CodexTurnError::Unavailable,
    })?;
    let login_text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    if !output.status.success() || !parse_login_status(&login_text).0 {
        return Err(CodexTurnError::NotAuthenticated);
    }

    run_agent_turn_with_executable_until(&path, context, prompt, on_tool_call, deadline).await
}

#[cfg(all(test, unix))]
pub(crate) async fn run_agent_turn_with_executable<F>(
    path: &Path,
    context: CodexTurnContext,
    prompt: &str,
    on_tool_call: F,
) -> Result<CodexTurnOutput, CodexTurnError>
where
    F: FnMut(ToolCall),
{
    let deadline = tokio::time::Instant::now() + CODEX_TURN_TIMEOUT;
    run_agent_turn_with_executable_until(path, context, prompt, on_tool_call, deadline).await
}

pub(crate) async fn run_agent_turn_with_executable_until<F>(
    path: &Path,
    context: CodexTurnContext,
    prompt: &str,
    mut on_tool_call: F,
    deadline: tokio::time::Instant,
) -> Result<CodexTurnOutput, CodexTurnError>
where
    F: FnMut(ToolCall),
{
    let isolated_cwd = tempfile::tempdir().map_err(|_| CodexTurnError::CliFailed)?;
    let endpoint = crate::mcp::spawn(
        context.dispatcher.clone(),
        context.registry.clone(),
        context.gate.clone(),
    )
    .await
    .map_err(|_| CodexTurnError::McpStart)?;
    let args = build_exec_args(endpoint.url(), isolated_cwd.path());
    let no_proxy =
        loopback_no_proxy(std::env::var_os("NO_PROXY").or_else(|| std::env::var_os("no_proxy")));
    let mut command = tokio::process::Command::new(path);
    command
        .args(args)
        .current_dir(isolated_cwd.path())
        .env_remove("PWD")
        .env_remove("OLDPWD")
        .env_remove("INIT_CWD")
        .env_remove("npm_config_local_prefix")
        .env_remove("CARGO_MANIFEST_DIR")
        .env_remove("CARGO_WORKSPACE_DIR")
        .env(CODEX_MCP_BEARER_ENV, endpoint.bearer_token())
        .env("NO_PROXY", &no_proxy)
        .env("no_proxy", &no_proxy)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    opentake_media::process_tree::configure_command(command.as_std_mut());
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(_) => {
            context.gate.request_dispatch_cancel();
            close_endpoint_after_failure(endpoint, context.cancel.as_ref(), deadline).await;
            drop(isolated_cwd);
            return Err(CodexTurnError::CliFailed);
        }
    };
    let child_id = match child.id() {
        Some(child_id) => child_id,
        None => {
            context.gate.request_dispatch_cancel();
            let _ = child.start_kill();
            let _ = tokio::time::timeout_at(deadline, child.wait()).await;
            close_endpoint_after_failure(endpoint, context.cancel.as_ref(), deadline).await;
            drop(isolated_cwd);
            return Err(CodexTurnError::CliFailed);
        }
    };
    let mut tree = match opentake_media::process_tree::ProcessTree::attach(child_id) {
        Ok(tree) => tree,
        Err(_) => {
            context.gate.request_dispatch_cancel();
            let _ = child.start_kill();
            let _ = tokio::time::timeout_at(deadline, child.wait()).await;
            close_endpoint_after_failure(endpoint, context.cancel.as_ref(), deadline).await;
            drop(isolated_cwd);
            return Err(CodexTurnError::CliFailed);
        }
    };

    let stderr_task = child
        .stderr
        .take()
        .map(|stderr| tokio::spawn(drain_stderr(stderr)));
    let stdout = child.stdout.take();
    let operation_deadline = work_deadline(deadline);
    let prompt_result = match child.stdin.take() {
        Some(stdin) => {
            write_prompt_with_lifecycle(
                stdin,
                prompt,
                &endpoint,
                context.cancel.as_ref(),
                operation_deadline,
            )
            .await
        }
        None => Err(CodexTurnError::CliFailed),
    };
    let mut outcome = prompt_result.err().map(Err);
    let mut status: Option<std::io::Result<ExitStatus>> = None;
    if outcome.is_none() {
        if let Some(stdout) = stdout {
            let consume = consume_exec_stream(
                stdout,
                &endpoint,
                &context,
                operation_deadline,
                &mut on_tool_call,
            );
            tokio::pin!(consume);
            let wait = child.wait();
            tokio::pin!(wait);
            let mut poll = tokio::time::interval(CANCEL_POLL_INTERVAL);
            poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            let wait_deadline = tokio::time::sleep_until(operation_deadline);
            tokio::pin!(wait_deadline);
            // After EOF Codex is exiting: wait for its status (so an exit
            // failure is reported as such) unless the turn is stopped.
            let mut reached_eof = false;
            let mut stopped = false;
            loop {
                tokio::select! {
                    end = &mut consume, if outcome.is_none() => match end {
                        StreamEnd::Eof(result) => {
                            reached_eof = true;
                            outcome = Some(result);
                        }
                        StreamEnd::Failed(error) => outcome = Some(Err(error)),
                    },
                    waited = &mut wait, if status.is_none() => {
                        status = Some(waited);
                        // The immediate CLI may have left descendants holding the
                        // JSONL/stderr pipes. Terminate them so EOF is observable.
                        let _ = tree.terminate();
                        tree.disarm();
                    }
                    _ = poll.tick(), if outcome.is_none() || reached_eof => {
                        if context.cancel.load(Ordering::Acquire) {
                            outcome = Some(Err(CodexTurnError::Cancelled));
                            stopped = true;
                        }
                    }
                    _ = endpoint.stopped(), if outcome.is_none() || reached_eof => {
                        outcome = Some(Err(CodexTurnError::McpStart));
                        stopped = true;
                    }
                    _ = &mut wait_deadline, if outcome.is_none() || status.is_none() => {
                        outcome = Some(Err(CodexTurnError::Timeout));
                        stopped = true;
                    }
                }
                if outcome.is_some() && status.is_some() {
                    break;
                }
                if stopped || (!reached_eof && outcome.as_ref().is_some_and(Result::is_err)) {
                    break;
                }
            }
        } else {
            outcome = Some(Err(CodexTurnError::CliFailed));
        }
    }
    let mut outcome = outcome.unwrap_or(Err(CodexTurnError::Protocol));
    // Whether Codex exited by itself; otherwise it is still running and is
    // killed below, and its exit status reflects that kill.
    let exited_on_its_own = status.is_some();
    let wait_failed = match status.as_ref() {
        Some(Ok(status)) => !status.success(),
        Some(Err(_)) => true,
        None => false,
    };
    if wait_failed && outcome.is_ok() {
        outcome = Err(CodexTurnError::CliFailed);
    }
    let outcome_was_cancelled = matches!(&outcome, Err(CodexTurnError::Cancelled));
    let externally_cancelled = context.cancel.load(Ordering::Acquire);
    let requested_cleanup_cancel = outcome.is_err() || externally_cancelled;
    if requested_cleanup_cancel {
        if outcome_was_cancelled || externally_cancelled {
            context.gate.request_cancel();
        } else {
            context.gate.request_dispatch_cancel();
        }
        let _ = tree.terminate();
        let _ = child.start_kill();
    }
    let status = match status {
        Some(Ok(status)) => Ok(status),
        Some(Err(_)) | None => terminate_and_reap_until(&mut child, &mut tree, deadline).await,
    };
    let endpoint_result = endpoint
        .close_or_detach(endpoint_drain_give_up(
            deadline,
            outcome_was_cancelled || externally_cancelled,
            context.cancel.as_ref(),
        ))
        .await;
    if matches!(
        endpoint_result,
        Err(opentake_agent::mcp::server::EphemeralMcpError::Detached)
    ) && outcome.is_ok()
    {
        outcome = Err(CodexTurnError::Timeout);
    }
    let stderr = match join_capture_until(stderr_task, deadline).await {
        Ok(stderr) => stderr,
        Err(error) => {
            if outcome.is_ok() {
                outcome = Err(error);
            }
            Vec::new()
        }
    };
    drop(isolated_cwd);

    if outcome_was_cancelled
        || externally_cancelled
        || (!requested_cleanup_cancel && context.cancel.load(Ordering::Acquire))
    {
        return Err(CodexTurnError::Cancelled);
    }
    let status = match status {
        Ok(status) => status,
        Err(error) => return Err(error),
    };
    if !status.success()
        && !matches!(
            outcome,
            Err(CodexTurnError::Cancelled | CodexTurnError::Timeout | CodexTurnError::McpStart)
        )
    {
        outcome = Err(if strict_config_rejected(&stderr) {
            CodexTurnError::StrictConfigRejected
        } else {
            match outcome {
                // Keep the reported turn failure (and its sign-in hint).
                Err(CodexTurnError::ProviderFailed) => CodexTurnError::ProviderFailed,
                // A failure found while Codex still ran (for example a
                // protocol error) caused the kill; the kill's exit status
                // must not relabel it.
                Err(error) if !exited_on_its_own => error,
                _ => CodexTurnError::CliFailed,
            }
        });
    }
    if endpoint_result.is_err() && outcome.is_ok() {
        context.gate.request_dispatch_cancel();
        outcome = Err(CodexTurnError::McpStart);
    }
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentake_agent::chat::{ChatMessage, ChatSession};
    #[cfg(unix)]
    use opentake_agent::mcp::core_handle::{AppCoreHandle, CoreHandle};
    #[cfg(unix)]
    use opentake_agent::tools::result::ToolResult;
    #[cfg(unix)]
    use opentake_core::AppCore;

    #[cfg(unix)]
    struct TestTurnGate;

    #[cfg(unix)]
    impl ChatTurnGate for TestTurnGate {
        fn timeline(&self, dispatcher: &Dispatcher) -> Option<opentake_domain::Timeline> {
            Some(dispatcher.timeline())
        }

        fn dispatch(&self, dispatcher: &Dispatcher, name: &str, args: Value) -> Option<ToolResult> {
            Some(dispatcher.dispatch(name, args))
        }
    }

    #[cfg(unix)]
    fn turn_context(cancel: Arc<AtomicBool>) -> CodexTurnContext {
        let registry = Arc::new(RwLock::new(PluginRegistry::with_builtins()));
        let handle: Arc<dyn CoreHandle> = Arc::new(AppCoreHandle::new(AppCore::new()));
        let dispatcher = Arc::new(Dispatcher::new(handle, registry.clone()));
        CodexTurnContext {
            dispatcher,
            registry,
            gate: Arc::new(TestTurnGate),
            cancel,
        }
    }

    #[cfg(unix)]
    fn fake_codex_script(root: &Path, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;

        let path = root.join("fake-codex");
        std::fs::write(&path, format!("#!/bin/sh\nset -eu\n{body}\n")).unwrap();
        let mut permissions = std::fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&path, permissions).unwrap();
        path
    }

    #[cfg(unix)]
    fn process_exists(pid: &str) -> bool {
        Command::new("kill")
            .args(["-0", pid])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    }

    #[cfg(unix)]
    struct CapturedEndpoint {
        url: String,
        token: String,
    }

    #[cfg(unix)]
    fn captured_endpoint_and_cwd(capture: &str) -> (CapturedEndpoint, PathBuf) {
        let cwd = capture
            .lines()
            .find_map(|line| line.strip_prefix("cwd="))
            .map(PathBuf::from)
            .expect("captured isolated cwd");
        let url = capture
            .lines()
            .find(|line| line.contains("mcp_servers.opentake.url="))
            .and_then(|line| line.split('"').nth(1))
            .expect("captured dynamic endpoint");
        assert!(
            url.starts_with("http://127.0.0.1:") && url.ends_with("/mcp"),
            "{url}"
        );
        let token = capture
            .lines()
            .find_map(|line| line.strip_prefix("token="))
            .expect("captured bearer token");
        assert_eq!(token.len(), 64);
        (
            CapturedEndpoint {
                url: url.to_string(),
                token: token.to_string(),
            },
            cwd,
        )
    }

    /// Whether an MCP endpoint at `endpoint.url` accepts `endpoint.token`.
    /// A released ephemeral port can be bound again at once by a parallel
    /// test, so a refused connection cannot be required after cleanup;
    /// instead nothing there may accept the turn's bearer token, which only
    /// the turn's own endpoint ever held.
    #[cfg(unix)]
    async fn endpoint_accepts_token(endpoint: &CapturedEndpoint) -> bool {
        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap();
        let initialize = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": { "name": "closed-endpoint-check", "version": "0" }
            }
        });
        client
            .post(&endpoint.url)
            .bearer_auth(&endpoint.token)
            .header("accept", "application/json, text/event-stream")
            .json(&initialize)
            .send()
            .await
            .is_ok_and(|response| response.status().is_success())
    }

    #[cfg(unix)]
    async fn assert_endpoint_closed(endpoint: &CapturedEndpoint) {
        assert!(
            !endpoint_accepts_token(endpoint).await,
            "the turn's MCP endpoint still accepts its token at {}",
            endpoint.url
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn closed_endpoint_check_detects_a_live_endpoint() {
        let context = turn_context(Arc::new(AtomicBool::new(false)));
        let live = crate::mcp::spawn(context.dispatcher, context.registry, context.gate)
            .await
            .unwrap();
        let endpoint = CapturedEndpoint {
            url: live.url().to_string(),
            token: live.bearer_token().to_string(),
        };
        assert!(endpoint_accepts_token(&endpoint).await);
        let wrong_token = CapturedEndpoint {
            url: endpoint.url.clone(),
            token: "0".repeat(64),
        };
        assert!(!endpoint_accepts_token(&wrong_token).await);
        live.close().await.unwrap();
        assert_endpoint_closed(&endpoint).await;
    }

    #[test]
    fn parses_chatgpt_and_api_login_without_exposing_credentials() {
        assert_eq!(
            parse_login_status("Logged in using ChatGPT\n"),
            (true, Some("ChatGPT".into()))
        );
        assert_eq!(
            parse_login_status("Logged in using an API key."),
            (true, Some("API key".into()))
        );
        assert_eq!(
            parse_login_status("Logged in using token=must-not-surface"),
            (true, None)
        );
        assert_eq!(parse_login_status("Not logged in"), (false, None));
    }

    #[test]
    fn requires_the_verified_codex_cli_baseline() {
        assert!(!supported_codex_version("codex-cli 0.145.9"));
        assert!(supported_codex_version("codex-cli 0.146.0"));
        assert!(supported_codex_version("codex-cli 1.0.0"));
        assert!(!supported_codex_version("codex-cli unknown"));
        assert!(!supported_codex_version("other 0.146.0"));
    }

    #[test]
    fn exec_args_are_strict_dynamic_and_use_stdin_in_an_isolated_cwd() {
        let isolated = Path::new("/private/tmp/opentake-codex-turn");
        let endpoint = "http://127.0.0.1:43127/mcp";
        let args = build_exec_args(endpoint, isolated);
        let rendered = args
            .iter()
            .map(|arg| arg.to_string_lossy())
            .collect::<Vec<_>>()
            .join("\n");

        assert_eq!(args.first(), Some(&OsString::from("exec")));
        assert_eq!(args.last(), Some(&OsString::from("-")));
        assert!(rendered.contains("--strict-config"));
        assert!(rendered.contains("--ignore-user-config"));
        assert!(rendered.contains("--ignore-rules"));
        assert!(rendered.contains("--sandbox\nread-only"));
        assert!(rendered.contains(endpoint));
        assert!(rendered.contains(&format!(
            "mcp_servers.opentake.bearer_token_env_var=\"{CODEX_MCP_BEARER_ENV}\""
        )));
        assert!(!rendered.contains("not-a-real-secret"));
        assert!(rendered.contains(isolated.to_string_lossy().as_ref()));
        assert!(rendered.contains("approval_policy=\"never\""));
        assert!(rendered.contains("features.shell_tool=false"));
        assert!(rendered.contains("features.unified_exec=false"));
        assert!(rendered.contains("features.multi_agent=false"));
        assert!(rendered.contains("apps._default.enabled=false"));
        assert!(rendered.contains("web_search=\"disabled\""));
        assert!(!rendered.contains("127.0.0.1:19789"));
        assert!(!rendered.contains("tools.view_image"));
        assert!(!rendered.contains("secret-project-path"));
        let exec = args.iter().position(|arg| arg == "exec").unwrap();
        let ignore = args
            .iter()
            .position(|arg| arg == "--ignore-user-config")
            .unwrap();
        assert!(ignore > exec, "exec-only flag must follow the subcommand");
    }

    #[test]
    fn parses_codex_jsonl_agent_and_mcp_events() {
        let mut calls = HashMap::new();
        assert_eq!(
            parse_exec_event(
                r#"{"type":"item.started","item":{"id":"item_1","type":"mcp_tool_call","tool":"get_timeline","arguments":{}}}"#,
                &mut calls,
            ),
            Ok(ExecEvent::ToolChanged("item_1".into()))
        );
        assert_eq!(calls["item_1"].name, "get_timeline");
        assert_eq!(calls["item_1"].result, None);

        assert_eq!(
            parse_exec_event(
                r#"{"type":"item.completed","item":{"id":"item_1","type":"mcp_tool_call","tool":"get_timeline","arguments":{},"result":{},"error":null}}"#,
                &mut calls,
            ),
            Ok(ExecEvent::ToolChanged("item_1".into()))
        );
        assert_eq!(calls["item_1"].is_error, Some(false));
        assert_eq!(
            parse_exec_event(
                r#"{"type":"item.completed","item":{"id":"item_1","type":"mcp_tool_call","tool":"get_timeline","arguments":{},"result":{},"error":null}}"#,
                &mut calls,
            ),
            Ok(ExecEvent::Ignored),
            "duplicate events must not emit duplicate tool-call updates"
        );
        assert_eq!(
            parse_exec_event(
                r#"{"type":"item.completed","item":{"id":"item_2","type":"agent_message","text":"1280 × 720"}}"#,
                &mut calls,
            ),
            Ok(ExecEvent::AgentMessage("1280 × 720".into()))
        );
    }

    #[test]
    fn preserves_bounded_codex_mcp_text_and_raster_result_content() {
        let mut calls = HashMap::new();
        let event = serde_json::json!({
            "type": "item.completed",
            "item": {
                "id": "clear-timeline",
                "type": "mcp_tool_call",
                "tool": "remove_clips",
                "arguments": { "clipIds": ["clip-1"] },
                "result": {
                    "content": [
                        { "type": "text", "text": "Removed 1 clip" },
                        {
                            "type": "image",
                            "data": "iVBORw0KGgo=",
                            "mimeType": "image/png"
                        }
                    ],
                    "structuredContent": { "privatePath": "/must/not/persist" }
                },
                "error": null
            }
        });

        assert_eq!(
            parse_exec_event(&event.to_string(), &mut calls),
            Ok(ExecEvent::ToolChanged("clear-timeline".into()))
        );
        assert_eq!(
            calls["clear-timeline"].result,
            Some(serde_json::json!({
                "content": [
                    { "kind": "text", "text": "Removed 1 clip" },
                    {
                        "kind": "image",
                        "base64": "iVBORw0KGgo=",
                        "mediaType": "image/png"
                    }
                ]
            }))
        );
        assert!(!serde_json::to_string(&calls)
            .unwrap()
            .contains("privatePath"));
        assert_eq!(
            parse_exec_event(&event.to_string(), &mut calls),
            Ok(ExecEvent::Ignored),
            "an exact rich-result retry must remain idempotent"
        );
    }

    fn tool_result_item(content: Value) -> Value {
        serde_json::json!({
            "id": "big",
            "type": "mcp_tool_call",
            "tool": "get_transcript",
            "arguments": {},
            "result": { "content": content },
            "error": null
        })
    }

    fn result_texts(result: &Value) -> Vec<String> {
        result["content"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|block| block["text"].as_str().map(str::to_owned))
            .collect()
    }

    #[test]
    fn oversized_text_tool_result_is_truncated_with_its_length_and_hash() {
        // Multi-byte characters straddle the cut point.
        let text = "词".repeat(300 * 1024 / 3);
        let item = tool_result_item(serde_json::json!([{ "type": "text", "text": text }]));
        let result = normalized_codex_tool_result(&item, false).unwrap();
        let texts = result_texts(&result);
        assert_eq!(texts.len(), 1);
        let shown = &texts[0];
        assert!(shown.len() <= MAX_FINAL_TEXT_BYTES);
        assert!(text.starts_with(shown.split("\n[OpenTake: truncated").next().unwrap()));
        assert!(shown.contains(&format!("{} bytes", text.len())));
        let digest = Sha256::digest(text.as_bytes());
        assert!(shown.contains(&format!("sha256 {digest:x}")));

        let exact = "x".repeat(MAX_FINAL_TEXT_BYTES);
        let item = tool_result_item(serde_json::json!([{ "type": "text", "text": exact }]));
        let result = normalized_codex_tool_result(&item, false).unwrap();
        assert_eq!(result_texts(&result), vec![exact]);
    }

    #[test]
    fn excess_blocks_unknown_types_and_invalid_images_become_notes() {
        let blocks = (0..65)
            .map(|index| serde_json::json!({ "type": "text", "text": format!("block {index}") }))
            .collect::<Vec<_>>();
        let result =
            normalized_codex_tool_result(&tool_result_item(Value::Array(blocks)), false).unwrap();
        let texts = result_texts(&result);
        assert_eq!(texts.len(), MAX_TOOL_RESULT_BLOCKS);
        assert_eq!(texts[MAX_TOOL_RESULT_BLOCKS - 2], "block 62");
        assert!(texts[MAX_TOOL_RESULT_BLOCKS - 1].contains("2 more tool result blocks"));

        let oversized_image = "A".repeat(MAX_TOOL_RESULT_IMAGE_BASE64_BYTES + 4);
        let item = tool_result_item(serde_json::json!([
            { "type": "audio", "data": "AAAA", "mimeType": "audio/wav" },
            { "type": "resource", "resource": { "uri": "file:///private" } },
            { "type": "image", "data": oversized_image, "mimeType": "image/png" },
            { "type": "image", "data": "AAAA", "mimeType": "image/svg+xml" },
            { "type": "image", "data": "not base64!", "mimeType": "image/png" },
        ]));
        let result = normalized_codex_tool_result(&item, false).unwrap();
        let texts = result_texts(&result);
        assert_eq!(texts.len(), 5);
        assert!(texts[0].contains("audio block was omitted"));
        assert!(texts[1].contains("resource block was omitted"));
        assert!(!serde_json::to_string(&result).unwrap().contains("private"));
        assert!(texts[2].contains("image (image/png"));
        assert!(texts[3].contains("image/svg+xml"));
        assert!(texts[4].contains("omitted"));
    }

    #[test]
    fn structurally_malformed_tool_results_remain_protocol_errors() {
        for content in [
            serde_json::json!("not an array"),
            serde_json::json!([{ "type": "text" }]),
            serde_json::json!([{ "type": "text", "text": 5 }]),
            serde_json::json!([{ "text": "no type" }]),
            serde_json::json!([{ "type": "image", "mimeType": "image/png" }]),
        ] {
            assert_eq!(
                normalized_codex_tool_result(&tool_result_item(content.clone()), false),
                Err(CodexTurnError::Protocol),
                "{content}"
            );
        }
    }

    fn jsonl(value: Value) -> JsonlLine {
        JsonlLine::Text(value.to_string())
    }

    #[test]
    fn stream_with_oversized_tool_results_completes_the_turn() {
        let mut stream = ExecStreamState::default();
        let started = serde_json::json!({
            "type": "item.started",
            "item": { "id": "big", "type": "mcp_tool_call", "tool": "get_transcript", "arguments": {} }
        });
        assert!(stream.accept_line(jsonl(started)).unwrap().is_some());
        let big = serde_json::json!({
            "type": "item.completed",
            "item": tool_result_item(serde_json::json!([
                { "type": "text", "text": "y".repeat(300 * 1024) }
            ]))
        });
        let emitted = stream.accept_line(jsonl(big)).unwrap().unwrap();
        assert_eq!(emitted.is_error, Some(false));
        // A completion line beyond the reader's bound is skipped, not fatal.
        let started = serde_json::json!({
            "type": "item.started",
            "item": { "id": "t2", "type": "mcp_tool_call", "tool": "inspect_timeline", "arguments": {} }
        });
        stream.accept_line(jsonl(started)).unwrap();
        assert!(stream
            .accept_line(JsonlLine::Oversized {
                len: MAX_JSONL_LINE_BYTES + 1,
                agent_message: false,
            })
            .unwrap()
            .is_none());
        let message = serde_json::json!({
            "type": "item.completed",
            "item": { "id": "m", "type": "agent_message", "text": "Done." }
        });
        stream.accept_line(jsonl(message)).unwrap();

        let output = stream.finish().unwrap();
        assert_eq!(output.text, "Done.");
        assert_eq!(output.tool_calls.len(), 2);
        let big = &output.tool_calls[0];
        assert_eq!(big.id, "big");
        assert!(serde_json::to_string(&big.result)
            .unwrap()
            .contains("truncated"));
        let skipped = &output.tool_calls[1];
        assert_eq!(skipped.id, "t2");
        assert_eq!(skipped.result.as_ref().unwrap()["status"], "omitted");
    }

    #[tokio::test]
    async fn an_unterminated_oversized_line_fails_at_the_stdout_budget() {
        let data = vec![b'x'; MAX_STDOUT_BYTES + 2];
        let mut reader = BufReader::new(data.as_slice());
        let mut buffer = JsonlLineBuffer::default();
        assert_eq!(
            read_bounded_line(&mut reader, &mut buffer).await,
            Err(CodexTurnError::Protocol)
        );
    }

    #[tokio::test]
    async fn an_oversized_reply_line_is_identified_and_replaces_the_reply() {
        // Codex writes the item type before its text.
        let text = "r".repeat(MAX_JSONL_LINE_BYTES);
        let line = format!(
            r#"{{"type":"item.completed","item":{{"id":"m2","type":"agent_message","text":"{text}"}}}}"#
        );
        let data = format!("{line}\n").into_bytes();
        let mut reader = BufReader::new(data.as_slice());
        let mut buffer = JsonlLineBuffer::default();
        let read = read_bounded_line(&mut reader, &mut buffer)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            read,
            JsonlLine::Oversized {
                len: line.len(),
                agent_message: true,
            }
        );

        let mut stream = ExecStreamState::default();
        let earlier = serde_json::json!({
            "type": "item.completed",
            "item": { "id": "m1", "type": "agent_message", "text": "Working on it." }
        });
        stream.accept_line(jsonl(earlier)).unwrap();
        stream.accept_line(read).unwrap();
        let output = stream.finish().unwrap();
        assert!(
            output.text.contains("too large to display"),
            "{}",
            output.text
        );
        assert!(!output.text.contains("Working on it."));
    }

    #[test]
    fn oversized_reply_detection_keys_on_the_item_type() {
        let reply =
            br#"{"type":"item.completed","item":{"id":"m","type":"agent_message","text":"abc"#;
        assert!(oversized_line_is_agent_message(reply));
        // A tool call whose arguments mention an agent_message is not a reply.
        let tool = br#"{"type":"item.completed","item":{"id":"t","type":"mcp_tool_call","tool":"add_texts","arguments":{"type":"agent_message","text":"a \"type\":\"agent_message\" b"#;
        assert!(!oversized_line_is_agent_message(tool));
        // Neither is a string value containing the marker ahead of the item.
        let note = br#"{"note":"{\"type\":\"agent_message\"}","type":"item.completed","item":{"id":"t","type":"mcp_tool_call","result":"#;
        assert!(!oversized_line_is_agent_message(note));
        // A nested object's type is not the item's type.
        let nested = br#"{"type":"item.completed","item":{"meta":{"type":"agent_message"},"id":"t","type":"mcp_tool_call","#;
        assert!(!oversized_line_is_agent_message(nested));
        // A started (not completed) message is not the reply.
        let started =
            br#"{"type":"item.started","item":{"id":"m","type":"agent_message","text":"abc"#;
        assert!(!oversized_line_is_agent_message(started));
        // Whitespace, numbers and arrays before the type are skipped.
        let spaced = br#"{ "seq" : 12 , "tags" : [1, {"type":"agent_message"}], "type" : "item.completed" , "item" : { "id" : "m" , "type" : "agent_message" , "text" : "#;
        assert!(oversized_line_is_agent_message(spaced));
        // A head that ends inside the item type decides nothing.
        let cut = br#"{"type":"item.completed","item":{"id":"m","type":"agent_mes"#;
        assert!(!oversized_line_is_agent_message(cut));
    }

    #[test]
    fn a_reply_above_the_display_limit_is_truncated_with_a_note() {
        let text = "é".repeat(MAX_FINAL_TEXT_BYTES / 2 + 10);
        let event = serde_json::json!({
            "type": "item.completed",
            "item": { "id": "m", "type": "agent_message", "text": text }
        });
        let mut calls = HashMap::new();
        let Ok(ExecEvent::AgentMessage(shown)) = parse_exec_event(&event.to_string(), &mut calls)
        else {
            panic!("an oversized reply is not a protocol error");
        };
        assert!(shown.len() <= MAX_FINAL_TEXT_BYTES);
        assert!(shown.contains(&format!("it was {} bytes", text.len())));
    }

    #[test]
    fn only_a_user_cancel_shortens_the_endpoint_drain() {
        let now = tokio::time::Instant::now();
        let deadline = now + Duration::from_secs(600);
        assert_eq!(
            endpoint_close_deadline(deadline, now, true),
            now + CANCELLED_TURN_CLEANUP_GRACE
        );
        assert_eq!(endpoint_close_deadline(deadline, now, false), deadline);
        let soon = now + Duration::from_secs(1);
        assert_eq!(endpoint_close_deadline(soon, now, true), soon);
    }

    #[tokio::test(start_paused = true)]
    async fn a_stop_during_an_internal_failure_drain_switches_to_the_short_grace() {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(1800);
        let cancel = Arc::new(AtomicBool::new(false));
        let give_up = endpoint_drain_give_up(deadline, false, cancel.as_ref());
        tokio::pin!(give_up);
        assert!(
            tokio::time::timeout(Duration::from_secs(600), &mut give_up)
                .await
                .is_err(),
            "an internal failure keeps waiting for its tool calls"
        );
        cancel.store(true, Ordering::Release);
        let stopped = tokio::time::Instant::now();
        tokio::time::timeout(
            CANCELLED_TURN_CLEANUP_GRACE + Duration::from_secs(1),
            &mut give_up,
        )
        .await
        .expect("a Stop bounds the drain by the short grace");
        assert!(tokio::time::Instant::now() - stopped >= CANCELLED_TURN_CLEANUP_GRACE);

        // A user cancel uses the short grace from the start; the deadline
        // still bounds both.
        let started = tokio::time::Instant::now();
        endpoint_drain_give_up(deadline, true, &AtomicBool::new(false)).await;
        assert_eq!(
            tokio::time::Instant::now() - started,
            CANCELLED_TURN_CLEANUP_GRACE
        );
        let near = tokio::time::Instant::now() + Duration::from_secs(1);
        endpoint_drain_give_up(near, false, &AtomicBool::new(false)).await;
        assert_eq!(tokio::time::Instant::now(), near);
    }

    #[test]
    fn tool_calls_beyond_the_display_cap_are_counted_not_fatal() {
        let mut stream = ExecStreamState::default();
        for index in 0..MAX_TOOL_CALLS + 3 {
            for event_type in ["item.started", "item.completed"] {
                let event = serde_json::json!({
                    "type": event_type,
                    "item": {
                        "id": format!("call-{index:04}"),
                        "type": "mcp_tool_call",
                        "tool": "get_timeline",
                        "arguments": {},
                        "result": {},
                        "error": null
                    }
                });
                stream.accept_line(jsonl(event)).unwrap();
            }
        }
        let message = serde_json::json!({
            "type": "item.completed",
            "item": { "id": "m", "type": "agent_message", "text": "Done." }
        });
        stream.accept_line(jsonl(message)).unwrap();
        let output = stream.finish().unwrap();
        assert_eq!(output.tool_calls.len(), MAX_TOOL_CALLS);
        assert!(output.text.starts_with("Done."));
        assert!(output.text.contains("3 more were not recorded"));
    }

    #[test]
    fn codex_mcp_error_marker_is_strict_and_error_content_is_redacted() {
        const PRIVATE_SENTINEL: &str = "PRIVATE_CODEX_MCP_ERROR_SENTINEL";
        let malformed = serde_json::json!({
            "type": "item.completed",
            "item": {
                "id": "malformed-error",
                "type": "mcp_tool_call",
                "tool": "remove_clips",
                "arguments": {},
                "result": {
                    "isError": "true",
                    "content": [{ "type": "text", "text": PRIVATE_SENTINEL }]
                },
                "error": null
            }
        });
        let mut calls = HashMap::new();
        assert_eq!(
            parse_exec_event(&malformed.to_string(), &mut calls),
            Err(CodexTurnError::Protocol)
        );
        assert!(calls.is_empty());

        for (index, item_error) in [
            Value::Null,
            serde_json::json!({ "message": PRIVATE_SENTINEL }),
        ]
        .into_iter()
        .enumerate()
        {
            let event = serde_json::json!({
                "type": "item.completed",
                "item": {
                    "id": format!("private-error-{index}"),
                    "type": "mcp_tool_call",
                    "tool": "remove_clips",
                    "arguments": {},
                    "result": {
                        "isError": item_error.is_null(),
                        "content": [{ "type": "text", "text": PRIVATE_SENTINEL }]
                    },
                    "error": item_error
                }
            });
            assert!(matches!(
                parse_exec_event(&event.to_string(), &mut calls),
                Ok(ExecEvent::ToolChanged(_))
            ));
        }
        let persisted = serde_json::to_string(&calls).unwrap();
        assert!(!persisted.contains(PRIVATE_SENTINEL));
        assert!(calls.values().all(|call| {
            call.result == Some(serde_json::json!({ "status": "failed" }))
                && call.is_error == Some(true)
        }));
    }

    #[test]
    fn codex_import_args_are_redacted_before_events_blocks_and_session_json() {
        const URL_TOKEN: &str = "CODEX_SENTINEL_URL_TOKEN";
        const INLINE_BYTES: &str = "Q09ERVhfU0VOVElORUxfQkFTRTY0";
        let mut calls = HashMap::new();
        let event = serde_json::json!({
            "type": "item.started",
            "item": {
                "id": "secret-import",
                "type": "mcp_tool_call",
                "tool": "mcp__opentake__import_media",
                "arguments": {
                    "source": {
                        "url": format!(
                            "https://user:password@example.invalid/media.mp4?token={URL_TOKEN}#{URL_TOKEN}"
                        ),
                        "bytes": INLINE_BYTES,
                        "mimeType": "video/mp4"
                    }
                }
            }
        });
        assert_eq!(
            parse_exec_event(&event.to_string(), &mut calls),
            Ok(ExecEvent::ToolChanged("secret-import".into()))
        );

        let call = calls.remove("secret-import").unwrap();
        assert_eq!(
            call.args["source"]["url"],
            "https://example.invalid/media.mp4"
        );
        assert_eq!(call.args["source"]["bytes"]["byteLength"], 21);
        assert_eq!(call.args["source"]["bytes"]["redacted"], true);
        assert_eq!(
            call.args["source"]["bytes"]["sha256"]
                .as_str()
                .unwrap()
                .len(),
            64
        );

        let emitted = serde_json::to_string(&call).unwrap();
        assert!(!emitted.contains(URL_TOKEN));
        assert!(!emitted.contains(INLINE_BYTES));
        assert!(!emitted.contains("user:password"));

        let message = ChatMessage::assistant("imported", vec![call]);
        let blocks = serde_json::to_string(&message.blocks).unwrap();
        assert!(!blocks.contains(URL_TOKEN));
        assert!(!blocks.contains(INLINE_BYTES));

        let mut session = ChatSession::new("provider-switch");
        session.provider = Some("codex".into());
        session.messages.push(message);
        session.provider = Some("openai".into());
        let persisted = serde_json::to_string(&session).unwrap();
        assert!(!persisted.contains(URL_TOKEN));
        assert!(!persisted.contains(INLINE_BYTES));
        assert!(!persisted.contains("user:password"));
    }

    #[test]
    fn rejects_non_json_and_recognizes_safe_terminal_events() {
        let mut calls = HashMap::new();
        assert_eq!(
            parse_exec_event("diagnostic", &mut calls),
            Err(CodexTurnError::Protocol)
        );
        assert_eq!(
            parse_exec_event(r#"{"type":"thread.started"}"#, &mut calls),
            Ok(ExecEvent::Ignored)
        );
        assert_eq!(
            parse_exec_event(
                r#"{"type":"turn.failed","error":{"message":"private"}}"#,
                &mut calls,
            ),
            Ok(ExecEvent::TurnFailed)
        );
        assert!(calls.is_empty());
    }

    #[tokio::test]
    async fn bounded_jsonl_reader_drains_oversized_lines_without_buffering_them() {
        let mut data = vec![b'x'; MAX_JSONL_LINE_BYTES + 1];
        data.extend_from_slice(b"\n{\"type\":\"thread.started\"}\n");
        let mut reader = BufReader::new(data.as_slice());
        let mut buffer = JsonlLineBuffer::default();
        assert_eq!(
            read_bounded_line(&mut reader, &mut buffer).await,
            Ok(Some(JsonlLine::Oversized {
                len: MAX_JSONL_LINE_BYTES + 1,
                agent_message: false,
            }))
        );
        assert!(buffer.bytes.capacity() <= MAX_JSONL_LINE_BYTES);
        assert!(buffer.oversized_head.is_empty());
        assert_eq!(
            read_bounded_line(&mut reader, &mut buffer).await,
            Ok(Some(JsonlLine::Text(r#"{"type":"thread.started"}"#.into())))
        );
        assert_eq!(read_bounded_line(&mut reader, &mut buffer).await, Ok(None));
    }

    /// Same shape as `consume_exec_stream`: a fresh read future races the
    /// cancel-poll tick on every loop iteration.
    async fn read_line_racing_cancel_poll<R: AsyncBufRead + Unpin>(
        reader: &mut R,
        buffer: &mut JsonlLineBuffer,
    ) -> Result<Option<JsonlLine>, CodexTurnError> {
        let mut poll = tokio::time::interval(CANCEL_POLL_INTERVAL);
        loop {
            tokio::select! {
                line = read_bounded_line(reader, buffer) => return line,
                _ = poll.tick() => {}
            }
        }
    }

    async fn read_lines_written_in_delayed_segments(
        bytes: Vec<u8>,
        splits: Vec<usize>,
        lines: usize,
    ) -> Vec<JsonlLine> {
        let (mut writer, stdout) = tokio::io::duplex(64 * 1024);
        let feeder = tokio::spawn(async move {
            let mut start = 0;
            for end in splits.into_iter().chain([bytes.len()]) {
                writer.write_all(&bytes[start..end]).await.unwrap();
                start = end;
                tokio::time::sleep(CANCEL_POLL_INTERVAL * 2 + Duration::from_millis(20)).await;
            }
        });
        let mut reader = BufReader::new(stdout);
        let mut buffer = JsonlLineBuffer::default();
        let mut read = Vec::new();
        for _ in 0..lines {
            read.push(
                read_line_racing_cancel_poll(&mut reader, &mut buffer)
                    .await
                    .unwrap()
                    .unwrap(),
            );
        }
        feeder.await.unwrap();
        read
    }

    async fn read_line_written_in_delayed_segments(
        bytes: Vec<u8>,
        splits: Vec<usize>,
    ) -> Option<String> {
        match read_lines_written_in_delayed_segments(bytes, splits, 1)
            .await
            .pop()
        {
            Some(JsonlLine::Text(line)) => Some(line),
            other => panic!("expected one text line, got {other:?}"),
        }
    }

    #[tokio::test(start_paused = true)]
    async fn bounded_jsonl_reader_keeps_oversized_framing_across_cancel_ticks() {
        let mut bytes = vec![b'x'; MAX_JSONL_LINE_BYTES + 10];
        bytes.extend_from_slice(b"\n{\"type\":\"thread.started\"}\n");
        let splits = vec![MAX_JSONL_LINE_BYTES / 2, MAX_JSONL_LINE_BYTES + 5];
        let read = read_lines_written_in_delayed_segments(bytes, splits, 2).await;
        assert_eq!(
            read,
            vec![
                JsonlLine::Oversized {
                    len: MAX_JSONL_LINE_BYTES + 10,
                    agent_message: false,
                },
                JsonlLine::Text(r#"{"type":"thread.started"}"#.into()),
            ]
        );
    }

    #[tokio::test(start_paused = true)]
    async fn bounded_jsonl_reader_keeps_a_large_line_split_across_cancel_ticks() {
        let text = "x".repeat(100 * 1024);
        let line = format!(
            r#"{{"type":"item.completed","item":{{"id":"m1","type":"agent_message","text":"{text}"}}}}"#
        );
        let mut bytes = line.clone().into_bytes();
        bytes.push(b'\n');
        let splits = vec![30, 40 * 1024, 80 * 1024];
        let read = read_line_written_in_delayed_segments(bytes, splits)
            .await
            .unwrap();
        assert_eq!(read, line);
        let mut calls = HashMap::new();
        assert_eq!(
            parse_exec_event(&read, &mut calls),
            Ok(ExecEvent::AgentMessage(text))
        );
    }

    #[tokio::test(start_paused = true)]
    async fn bounded_jsonl_reader_splits_at_any_byte_including_crlf_and_utf8() {
        let line = r#"{"type":"thread.started","note":"你好"}"#;
        let bytes = format!("{line}\r\n").into_bytes();
        for split in 1..bytes.len() {
            let read = read_line_written_in_delayed_segments(bytes.clone(), vec![split]).await;
            assert_eq!(read.as_deref(), Some(line), "split at byte {split}");
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancelled_cli_probe_kills_and_reaps_the_probe_process() {
        let root = tempfile::tempdir().unwrap();
        let script = fake_codex_script(
            root.path(),
            r#"
hold="$(dirname "$0")/probe-hold"
mkfifo "$hold"
sleep 60 &
descendant="$!"
printf 'pid=%s\ndescendant=%s\n' "$$" "$descendant" > "$(dirname "$0")/probe-capture"
: > "$(dirname "$0")/probe-ready"
exec 3<> "$hold"
IFS= read -r ignored <&3
"#,
        );
        let cancel = Arc::new(AtomicBool::new(false));
        let task_cancel = cancel.clone();
        let task_script = script.clone();
        let task = tokio::spawn(async move {
            run_probe(
                &task_script,
                &["--version"],
                task_cancel.as_ref(),
                tokio::time::Instant::now() + Duration::from_secs(60),
            )
            .await
        });
        let ready = root.path().join("probe-ready");
        tokio::time::timeout(Duration::from_secs(5), async {
            while !ready.exists() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("probe child reached its blocking point");
        let capture = std::fs::read_to_string(root.path().join("probe-capture")).unwrap();
        let pid = capture
            .lines()
            .find_map(|line| line.strip_prefix("pid="))
            .expect("captured probe pid")
            .to_string();
        let descendant = capture
            .lines()
            .find_map(|line| line.strip_prefix("descendant="))
            .expect("captured probe descendant")
            .to_string();

        cancel.store(true, Ordering::Release);
        let result = tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .expect("probe cancellation completed")
            .expect("probe task joined");
        assert!(matches!(result, Err(CodexTurnError::Cancelled)));
        assert!(!process_exists(&pid), "probe child must be reaped");
        assert!(
            !process_exists(&descendant),
            "probe descendant inheriting capture pipes must be killed"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn fake_cli_success_closes_endpoint_waits_child_and_removes_tempdir() {
        let root = tempfile::tempdir().unwrap();
        let script = fake_codex_script(
            root.path(),
            r#"
capture="$(dirname "$0")/capture"
{
  sleep 60 &
  descendant="$!"
  printf 'cwd=%s\n' "$PWD"
  printf 'token=%s\n' "${OPENTAKE_CODEX_MCP_BEARER_TOKEN:-}"
  printf 'token_length=%s\n' "${#OPENTAKE_CODEX_MCP_BEARER_TOKEN}"
  printf 'descendant=%s\n' "$descendant"
  printf 'arg=%s\n' "$@"
  IFS= read -r prompt || true
  printf 'prompt=%s\n' "$prompt"
} > "$capture"
printf '%s\n' '{"type":"thread.started"}'
printf '%s\n' '{"type":"item.completed","item":{"id":"message_1","type":"agent_message","text":"finished"}}'
printf '%s\n' 'child-waited' > "$(dirname "$0")/finished"
"#,
        );
        let result = run_agent_turn_with_executable(
            &script,
            turn_context(Arc::new(AtomicBool::new(false))),
            "prompt over stdin",
            |_| {},
        )
        .await
        .expect("fake Codex turn succeeds");
        assert_eq!(result.text, "finished");
        assert!(root.path().join("finished").exists());

        let capture = std::fs::read_to_string(root.path().join("capture")).unwrap();
        let (endpoint, cwd) = captured_endpoint_and_cwd(&capture);
        let descendant = capture
            .lines()
            .find_map(|line| line.strip_prefix("descendant="))
            .expect("captured inherited-pipe descendant");
        assert!(capture.contains("prompt=prompt over stdin"));
        assert!(capture.contains("token_length=64"));
        assert!(
            !process_exists(descendant),
            "successful turn must kill descendants that retain JSONL pipes"
        );
        assert!(!cwd.exists(), "isolated cwd removed only after cleanup");
        assert_endpoint_closed(&endpoint).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn fake_cli_cancel_interrupts_blocked_stdin_and_cleans_up_everything() {
        let root = tempfile::tempdir().unwrap();
        let script = fake_codex_script(
            root.path(),
            r#"
capture="$(dirname "$0")/capture"
hold="$(dirname "$0")/hold"
mkfifo "$hold"
{
  printf 'cwd=%s\n' "$PWD"
  printf 'token=%s\n' "${OPENTAKE_CODEX_MCP_BEARER_TOKEN:-}"
  printf 'pid=%s\n' "$$"
  printf 'arg=%s\n' "$@"
} > "$capture"
: > "$(dirname "$0")/ready"
printf '%s\n' '{"type":"thread.started"}'
exec 3<> "$hold"
IFS= read -r ignored <&3
"#,
        );
        let cancel = Arc::new(AtomicBool::new(false));
        let task_cancel = cancel.clone();
        let task_script = script.clone();
        let prompt = "x".repeat(2 * 1024 * 1024);
        let task = tokio::spawn(async move {
            run_agent_turn_with_executable(&task_script, turn_context(task_cancel), &prompt, |_| {})
                .await
        });
        let capture_path = root.path().join("capture");
        let ready_path = root.path().join("ready");
        tokio::time::timeout(Duration::from_secs(5), async {
            while !ready_path.exists() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("fake child reached its blocking point");
        let capture = std::fs::read_to_string(&capture_path).unwrap();
        let pid = capture
            .lines()
            .find_map(|line| line.strip_prefix("pid="))
            .expect("captured child pid")
            .to_string();
        let (endpoint, cwd) = captured_endpoint_and_cwd(&capture);
        cancel.store(true, Ordering::Release);
        let result = tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .expect("cancel cleanup completed")
            .expect("runner task joined");
        assert_eq!(result.unwrap_err(), CodexTurnError::Cancelled);
        assert!(!cwd.exists());
        assert_endpoint_closed(&endpoint).await;
        assert!(
            !Command::new("kill")
                .args(["-0", &pid])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .unwrap()
                .success(),
            "child must be reaped before the runner returns"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn fake_cli_deadline_kills_descendant_and_finishes_within_absolute_deadline() {
        let root = tempfile::tempdir().unwrap();
        let script = fake_codex_script(
            root.path(),
            r#"
capture="$(dirname "$0")/deadline-capture"
sleep 60 &
descendant="$!"
printf 'descendant=%s\n' "$descendant" > "$capture"
printf '%s\n' '{"type":"thread.started"}'
sleep 60
"#,
        );
        // The runner reserves two seconds for bounded cleanup. Leave three
        // seconds for CLI probes/spawn so this process-tree assertion remains
        // meaningful under a parallel, CPU-contended test suite.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        let started = tokio::time::Instant::now();
        let result = run_agent_turn_with_executable_until(
            &script,
            turn_context(Arc::new(AtomicBool::new(false))),
            "deadline",
            |_| {},
            deadline,
        )
        .await;

        assert_eq!(result.unwrap_err(), CodexTurnError::Timeout);
        assert!(
            tokio::time::Instant::now().duration_since(started) <= Duration::from_secs(6),
            "cleanup exceeded its absolute deadline by an unbounded amount"
        );
        let capture = std::fs::read_to_string(root.path().join("deadline-capture")).unwrap();
        let descendant = capture
            .trim()
            .strip_prefix("descendant=")
            .expect("captured deadline descendant");
        assert!(
            !process_exists(descendant),
            "deadline cleanup must kill the descendant process tree"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn fake_cli_strict_failure_is_structured_and_fully_cleaned_up() {
        let root = tempfile::tempdir().unwrap();
        let script = fake_codex_script(
            root.path(),
            r#"
capture="$(dirname "$0")/capture"
{
  printf 'cwd=%s\n' "$PWD"
  printf 'token=%s\n' "${OPENTAKE_CODEX_MCP_BEARER_TOKEN:-}"
  printf 'arg=%s\n' "$@"
  IFS= read -r prompt || true
} > "$capture"
printf '%s\n' 'unknown configuration key containing private/path/token' >&2
exit 2
"#,
        );
        let result = run_agent_turn_with_executable(
            &script,
            turn_context(Arc::new(AtomicBool::new(false))),
            "do not expose failures",
            |_| {},
        )
        .await;
        assert_eq!(
            result.as_ref().unwrap_err(),
            &CodexTurnError::StrictConfigRejected
        );

        let capture = std::fs::read_to_string(root.path().join("capture")).unwrap();
        let (endpoint, cwd) = captured_endpoint_and_cwd(&capture);
        assert!(!cwd.exists());
        assert_endpoint_closed(&endpoint).await;
        assert!(!format!("{result:?}").contains("private/path/token"));
    }

    #[cfg(unix)]
    fn fake_codex_with_version(root: &Path, name: &str, version: &str) -> PathBuf {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        fake_codex_script(
            &dir,
            &format!("if [ \"${{1:-}}\" = \"--version\" ]; then echo 'codex-cli {version}'; exit 0; fi\nexit 3"),
        )
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn an_old_codex_is_reported_as_incompatible_not_missing() {
        let root = tempfile::tempdir().unwrap();
        let old = fake_codex_with_version(root.path(), "old", "0.145.9");
        let older = fake_codex_with_version(root.path(), "older", "0.100.0");
        let new = fake_codex_with_version(root.path(), "new", "0.146.0");
        let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
        let cancel = AtomicBool::new(false);

        let found = discover_codex_among(vec![older.clone(), old.clone()], &cancel, deadline)
            .await
            .unwrap();
        let CodexDiscovery::Incompatible { version } = found else {
            panic!("an old CLI must be reported as incompatible");
        };
        assert_eq!(
            version, "codex-cli 0.145.9",
            "the newest old version is named"
        );
        let status = CodexDiscovery::incompatible_status(version);
        assert!(!status.available);
        assert_eq!(status.version.as_deref(), Some("codex-cli 0.145.9"));
        assert!(status.message.contains("0.145.9"), "{}", status.message);
        assert!(
            status.message.contains("update it to 0.146.0"),
            "{}",
            status.message
        );

        let found = discover_codex_among(vec![old.clone(), new.clone()], &cancel, deadline)
            .await
            .unwrap();
        let CodexDiscovery::Supported(codex) = found else {
            panic!("the supported candidate must win over an older one");
        };
        assert_eq!(codex.path, new);
        assert!(matches!(
            discover_codex_among(vec![root.path().join("missing")], &cancel, deadline)
                .await
                .unwrap(),
            CodexDiscovery::NotFound
        ));

        let result = run_agent_turn_among(
            vec![old],
            turn_context(Arc::new(AtomicBool::new(false))),
            "prompt",
            |_| {},
        )
        .await;
        assert_eq!(result.unwrap_err(), CodexTurnError::IncompatibleCli);
        let result = run_agent_turn_among(
            Vec::new(),
            turn_context(Arc::new(AtomicBool::new(false))),
            "prompt",
            |_| {},
        )
        .await;
        assert_eq!(result.unwrap_err(), CodexTurnError::Unavailable);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_codex_executable_that_cannot_start_is_a_cli_failure() {
        let root = tempfile::tempdir().unwrap();
        let result = run_agent_turn_with_executable(
            &root.path().join("missing-codex"),
            turn_context(Arc::new(AtomicBool::new(false))),
            "prompt",
            |_| {},
        )
        .await;
        assert_eq!(result.unwrap_err(), CodexTurnError::CliFailed);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_protocol_error_keeps_its_kind_after_codex_is_killed() {
        let root = tempfile::tempdir().unwrap();
        let script = fake_codex_script(
            root.path(),
            r#"
IFS= read -r prompt || true
printf '%s\n' '{"type":"thread.started"}'
printf '%s\n' 'not json'
sleep 60
"#,
        );
        let result = run_agent_turn_with_executable(
            &script,
            turn_context(Arc::new(AtomicBool::new(false))),
            "prompt",
            |_| {},
        )
        .await;
        assert_eq!(result.unwrap_err(), CodexTurnError::Protocol);

        // Codex that fails by itself is still a CLI failure.
        let script = fake_codex_script(
            root.path(),
            r#"
IFS= read -r prompt || true
printf '%s\n' '{"type":"thread.started"}'
exit 4
"#,
        );
        let result = run_agent_turn_with_executable(
            &script,
            turn_context(Arc::new(AtomicBool::new(false))),
            "prompt",
            |_| {},
        )
        .await;
        assert_eq!(result.unwrap_err(), CodexTurnError::CliFailed);
    }
}
