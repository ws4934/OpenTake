//! Thin internal helpers for driving bundled or development `ffmpeg`/`ffprobe`
//! binaries.
//!
//! We deliberately do **not** link libav*: the local toolchain is ffmpeg 8.1
//! (libavcodec 62) which the C-binding crates do not support, and pkg-config is
//! absent. ffmpeg-sidecar shells out to binaries on `PATH`; these helpers wrap
//! binary discovery and one-shot ffprobe JSON queries so the higher-level decode
//! modules stay readable.
//!
//! Packaged binaries live beside the OpenTake executable. Environment overrides
//! `OPENTAKE_FFMPEG` / `OPENTAKE_FFPROBE` remain available to tests and
//! development tools; the desktop shell pins them to the bundled sidecars before
//! media initialization.

use std::cell::RefCell;
use std::ffi::OsString;
use std::future::Future;
use std::io::{Seek, SeekFrom};
use std::path::{Path, PathBuf};
#[cfg(all(test, unix))]
use std::process::Command;
use std::process::{ExitStatus, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use ffmpeg_sidecar::command::FfmpegCommand;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command as TokioCommand;
use tokio::task::{JoinHandle, JoinSet, LocalSet};
use tokio::time::{Instant as TokioInstant, MissedTickBehavior};

const FFPROBE_TIMEOUT: Duration = Duration::from_secs(30);
const FFPROBE_CAPTURE_MAX: usize = 8 * 1024 * 1024;
const FFPROBE_POLL_INTERVAL: Duration = Duration::from_millis(10);
const FFPROBE_READ_BUFFER_SIZE: usize = 16 * 1024;
const FFPROBE_MAX_IN_FLIGHT: usize = 8;
const FFPROBE_CLEANUP_MAX: Duration = Duration::from_millis(250);

struct FfprobeOutput {
    status: ExitStatus,
    stdout: Vec<u8>,
}

enum ProbeCaptureError {
    LimitReached,
    Read(std::io::Error),
}

enum ProbeCapture {
    Stdout(Result<Vec<u8>, ProbeCaptureError>),
    Stderr(Result<Vec<u8>, ProbeCaptureError>),
}

enum ProbeInput {
    Path(PathBuf),
    File(std::fs::File),
}

struct ProbeSpec {
    executable: OsString,
    input: ProbeInput,
    cancel: crate::MediaCancelToken,
    operation_deadline: Instant,
    api_deadline: Instant,
}

enum ProbeExecutorRequest {
    Run {
        spec: ProbeSpec,
        response: SyncSender<crate::error::Result<FfprobeOutput>>,
    },
    Available {
        executable: OsString,
        operation_deadline: Instant,
        api_deadline: Instant,
        response: SyncSender<crate::error::Result<bool>>,
    },
    #[cfg(test)]
    Seam {
        operation_deadline: Instant,
        never_returns: bool,
        response: SyncSender<crate::error::Result<()>>,
    },
}

struct ProbeExecutor {
    sender: tokio::sync::mpsc::Sender<ProbeExecutorRequest>,
}

struct ProbeAdmission;

static FFPROBE_ACTIVE: AtomicUsize = AtomicUsize::new(0);
static FFPROBE_EXECUTOR: OnceLock<Result<ProbeExecutor, String>> = OnceLock::new();

/// Saturation is transient and says nothing about the input, so callers get
/// [`MediaError::Busy`](crate::error::MediaError::Busy) rather than an ffmpeg
/// error they could take for an unreadable file (#63).
fn admission_limit_error() -> crate::error::MediaError {
    crate::error::MediaError::Busy("ffprobe admission limit reached".to_string())
}

impl ProbeAdmission {
    fn try_acquire() -> Option<Self> {
        #[cfg(test)]
        if test_seams::admission_saturated() || test_seams::take_saturated_attempt() {
            return None;
        }
        let mut active = FFPROBE_ACTIVE.load(Ordering::Acquire);
        loop {
            if active >= FFPROBE_MAX_IN_FLIGHT {
                return None;
            }
            match FFPROBE_ACTIVE.compare_exchange_weak(
                active,
                active + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some(Self),
                Err(observed) => active = observed,
            }
        }
    }

    fn acquire() -> crate::error::Result<Self> {
        Self::try_acquire().ok_or_else(admission_limit_error)
    }

    /// Wait for a free slot for at most `wait`, polling `cancel`. Decode-time
    /// probes queue here instead of failing the moment other media work holds
    /// every slot.
    fn acquire_waiting(
        cancel: &crate::MediaCancelToken,
        wait: Duration,
    ) -> crate::error::Result<Self> {
        #[cfg(test)]
        let wait = test_seams::admission_wait_override().unwrap_or(wait);
        let deadline = Instant::now().checked_add(wait);
        loop {
            if let Some(admission) = Self::try_acquire() {
                return Ok(admission);
            }
            if cancel.checkpoint() {
                return Err(crate::error::MediaError::Cancelled);
            }
            let now = Instant::now();
            let Some(remaining) = deadline
                .map(|deadline| deadline.saturating_duration_since(now))
                .filter(|remaining| !remaining.is_zero())
            else {
                return Err(admission_limit_error());
            };
            std::thread::sleep(FFPROBE_POLL_INTERVAL.min(remaining));
        }
    }
}

impl Drop for ProbeAdmission {
    fn drop(&mut self) {
        FFPROBE_ACTIVE.fetch_sub(1, Ordering::AcqRel);
    }
}

async fn read_probe_capture(
    mut pipe: impl AsyncRead + Unpin,
) -> Result<Vec<u8>, ProbeCaptureError> {
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; FFPROBE_READ_BUFFER_SIZE];
    loop {
        match pipe.read(&mut buffer).await {
            Ok(0) => return Ok(bytes),
            Ok(read) => {
                let remaining = FFPROBE_CAPTURE_MAX - bytes.len();
                if read >= remaining {
                    return Err(ProbeCaptureError::LimitReached);
                }
                bytes.extend_from_slice(&buffer[..read]);
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(ProbeCaptureError::Read(error)),
        }
    }
}

fn finish_probe_capture(
    stream: &str,
    capture: Result<Vec<u8>, ProbeCaptureError>,
) -> crate::error::Result<Vec<u8>> {
    match capture {
        Ok(bytes) => Ok(bytes),
        Err(ProbeCaptureError::LimitReached) => Err(crate::error::MediaError::Ffmpeg(format!(
            "ffprobe {stream} exceeded its bounded capture"
        ))),
        Err(ProbeCaptureError::Read(error)) => Err(crate::error::MediaError::Ffmpeg(format!(
            "ffprobe {stream} capture read: {error}"
        ))),
    }
}

fn timeout_error() -> crate::error::MediaError {
    crate::error::MediaError::Ffmpeg("ffprobe timed out".to_string())
}

fn probe_deadlines(timeout: Duration) -> crate::error::Result<(Instant, Instant)> {
    let now = Instant::now();
    let api_deadline = now.checked_add(timeout).ok_or_else(|| {
        crate::error::MediaError::Ffmpeg("ffprobe timeout is too large".to_string())
    })?;
    let cleanup_reserve = (timeout / 4).min(FFPROBE_CLEANUP_MAX);
    let operation_deadline = api_deadline
        .checked_sub(cleanup_reserve)
        .unwrap_or(api_deadline);
    Ok((operation_deadline, api_deadline))
}

fn terminate_tree(tree: &mut Option<crate::process_tree::ProcessTree>) {
    let Some(tree) = tree.as_mut() else {
        return;
    };
    if tree.terminate().is_ok() {
        tree.disarm();
    }
}

fn bounded_cleanup_deadline(api_deadline: TokioInstant) -> TokioInstant {
    api_deadline.min(TokioInstant::now() + FFPROBE_CLEANUP_MAX)
}

async fn finish_cleanup_before_deadline<W, O, E>(
    deadline: TokioInstant,
    wait: W,
    stdout: O,
    stderr: E,
) -> bool
where
    W: Future,
    O: Future,
    E: Future,
{
    tokio::time::timeout_at(deadline, async move {
        let _ = tokio::join!(wait, stdout, stderr);
    })
    .await
    .is_ok()
}

async fn terminate_running_ffprobe(
    child: &mut tokio::process::Child,
    tree: &mut Option<crate::process_tree::ProcessTree>,
    stdout_worker: JoinHandle<()>,
    stderr_worker: JoinHandle<()>,
    deadline: TokioInstant,
) {
    terminate_tree(tree);
    let _ = child.start_kill();
    stdout_worker.abort();
    stderr_worker.abort();
    let _ =
        finish_cleanup_before_deadline(deadline, child.wait(), stdout_worker, stderr_worker).await;
}

async fn reap_uncontained_child(child: &mut tokio::process::Child, deadline: TokioInstant) {
    let _ = child.start_kill();
    let _ = tokio::time::timeout_at(deadline, child.wait()).await;
}

async fn ffprobe_available_async(
    executable: OsString,
    operation_deadline: Instant,
    api_deadline: Instant,
) -> crate::error::Result<bool> {
    let operation_deadline = TokioInstant::from_std(operation_deadline);
    let api_deadline = TokioInstant::from_std(api_deadline);
    if TokioInstant::now() >= operation_deadline {
        return Err(timeout_error());
    }
    let mut command = TokioCommand::new(executable);
    command
        .arg("-version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    crate::process_tree::configure_command(command.as_std_mut());
    command.kill_on_drop(true);
    let mut child = command
        .spawn()
        .map_err(|error| crate::error::MediaError::Ffmpeg(format!("ffprobe spawn: {error}")))?;
    let child_id = child.id().ok_or_else(|| {
        crate::error::MediaError::Ffmpeg("ffprobe process id missing".to_string())
    })?;
    let mut tree = match crate::process_tree::ProcessTree::attach(child_id) {
        Ok(tree) => Some(tree),
        Err(error) => {
            reap_uncontained_child(&mut child, bounded_cleanup_deadline(api_deadline)).await;
            return Err(crate::error::MediaError::Ffmpeg(format!(
                "ffprobe containment: {error}"
            )));
        }
    };
    match tokio::time::timeout_at(operation_deadline, child.wait()).await {
        Ok(Ok(status)) => {
            terminate_tree(&mut tree);
            Ok(status.success())
        }
        Ok(Err(error)) => {
            terminate_tree(&mut tree);
            let _ = child.start_kill();
            let _ =
                tokio::time::timeout_at(bounded_cleanup_deadline(api_deadline), child.wait()).await;
            Err(crate::error::MediaError::Ffmpeg(format!(
                "ffprobe wait: {error}"
            )))
        }
        Err(_) => {
            terminate_tree(&mut tree);
            let _ = child.start_kill();
            let _ =
                tokio::time::timeout_at(bounded_cleanup_deadline(api_deadline), child.wait()).await;
            Err(timeout_error())
        }
    }
}

async fn run_ffprobe_async(spec: ProbeSpec) -> crate::error::Result<FfprobeOutput> {
    if spec.cancel.checkpoint() {
        return Err(crate::error::MediaError::Cancelled);
    }
    let operation_deadline = TokioInstant::from_std(spec.operation_deadline);
    let api_deadline = TokioInstant::from_std(spec.api_deadline);
    if TokioInstant::now() >= operation_deadline {
        return Err(timeout_error());
    }

    let mut command = TokioCommand::new(spec.executable);
    command.args([
        "-v",
        "quiet",
        "-of",
        "json",
        "-show_streams",
        "-show_format",
    ]);
    match spec.input {
        ProbeInput::File(mut input) => {
            input.seek(SeekFrom::Start(0)).map_err(|error| {
                crate::error::MediaError::Ffmpeg(format!("ffprobe input rewind: {error}"))
            })?;
            command.arg("fd:").stdin(Stdio::from(input));
        }
        ProbeInput::Path(path) => {
            command.arg(path).stdin(Stdio::null());
        }
    }
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    crate::process_tree::configure_command(command.as_std_mut());
    command.kill_on_drop(true);
    let mut child = command
        .spawn()
        .map_err(|error| crate::error::MediaError::Ffmpeg(format!("ffprobe spawn: {error}")))?;
    spec.cancel.child_spawned();
    let child_id = child.id().ok_or_else(|| {
        crate::error::MediaError::Ffmpeg("ffprobe process id missing".to_string())
    })?;
    let mut tree = match crate::process_tree::ProcessTree::attach(child_id) {
        Ok(tree) => Some(tree),
        Err(error) => {
            reap_uncontained_child(&mut child, bounded_cleanup_deadline(api_deadline)).await;
            return Err(crate::error::MediaError::Ffmpeg(format!(
                "ffprobe containment: {error}"
            )));
        }
    };

    let Some(stdout) = child.stdout.take() else {
        let empty_worker = tokio::task::spawn_local(async {});
        let second_empty_worker = tokio::task::spawn_local(async {});
        terminate_running_ffprobe(
            &mut child,
            &mut tree,
            empty_worker,
            second_empty_worker,
            bounded_cleanup_deadline(api_deadline),
        )
        .await;
        return Err(crate::error::MediaError::Ffmpeg(
            "ffprobe stdout pipe missing".to_string(),
        ));
    };
    let Some(stderr) = child.stderr.take() else {
        let empty_worker = tokio::task::spawn_local(async {});
        let second_empty_worker = tokio::task::spawn_local(async {});
        terminate_running_ffprobe(
            &mut child,
            &mut tree,
            empty_worker,
            second_empty_worker,
            bounded_cleanup_deadline(api_deadline),
        )
        .await;
        return Err(crate::error::MediaError::Ffmpeg(
            "ffprobe stderr pipe missing".to_string(),
        ));
    };

    let (capture_tx, mut capture_rx) = tokio::sync::mpsc::channel(2);
    let stdout_tx = capture_tx.clone();
    let stdout_worker = tokio::task::spawn_local(async move {
        let capture = read_probe_capture(stdout).await;
        let _ = stdout_tx.send(ProbeCapture::Stdout(capture)).await;
    });
    let stderr_worker = tokio::task::spawn_local(async move {
        let capture = read_probe_capture(stderr).await;
        let _ = capture_tx.send(ProbeCapture::Stderr(capture)).await;
    });

    let mut status = None;
    let mut stdout = None;
    let mut stderr = None;
    let mut drain_deadline = api_deadline;
    let mut cancel_poll = tokio::time::interval(FFPROBE_POLL_INTERVAL);
    cancel_poll.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let failure = loop {
        if status.is_some() && stdout.is_some() && stderr.is_some() {
            break None;
        }
        let phase_deadline = if status.is_some() {
            drain_deadline
        } else {
            operation_deadline
        };
        tokio::select! {
            wait = child.wait(), if status.is_none() => {
                match wait {
                    Ok(exit_status) => {
                        status = Some(exit_status);
                        // A malicious override can leave descendants holding the
                        // inherited pipes. Close the whole tree before draining.
                        terminate_tree(&mut tree);
                        drain_deadline = bounded_cleanup_deadline(api_deadline);
                    }
                    Err(error) => break Some(crate::error::MediaError::Ffmpeg(
                        format!("ffprobe wait: {error}"),
                    )),
                }
            }
            capture = capture_rx.recv(), if stdout.is_none() || stderr.is_none() => {
                match capture {
                    Some(ProbeCapture::Stdout(result)) => match finish_probe_capture("stdout", result) {
                        Ok(bytes) => stdout = Some(bytes),
                        Err(error) => break Some(error),
                    },
                    Some(ProbeCapture::Stderr(result)) => match finish_probe_capture("stderr", result) {
                        Ok(bytes) => stderr = Some(bytes),
                        Err(error) => break Some(error),
                    },
                    None => break Some(crate::error::MediaError::Ffmpeg(
                        "ffprobe capture worker stopped unexpectedly".to_string(),
                    )),
                }
            }
            _ = cancel_poll.tick() => {
                if spec.cancel.checkpoint() {
                    break Some(crate::error::MediaError::Cancelled);
                }
            }
            _ = tokio::time::sleep_until(phase_deadline) => {
                break Some(timeout_error());
            }
        }
    };

    if let Some(error) = failure {
        terminate_running_ffprobe(
            &mut child,
            &mut tree,
            stdout_worker,
            stderr_worker,
            bounded_cleanup_deadline(api_deadline),
        )
        .await;
        return Err(error);
    }

    // Both workers have delivered their bounded buffers and the immediate
    // process has been reaped. Their final send/return cannot retain a pipe.
    drop(stdout_worker);
    drop(stderr_worker);
    Ok(FfprobeOutput {
        status: status.expect("loop requires an exit status"),
        stdout: stdout.expect("loop requires stdout"),
    })
}

async fn execute_probe_request(request: ProbeExecutorRequest) {
    match request {
        ProbeExecutorRequest::Run { spec, response } => {
            let _ = response.try_send(run_ffprobe_async(spec).await);
        }
        ProbeExecutorRequest::Available {
            executable,
            operation_deadline,
            api_deadline,
            response,
        } => {
            let _ = response.try_send(
                ffprobe_available_async(executable, operation_deadline, api_deadline).await,
            );
        }
        #[cfg(test)]
        ProbeExecutorRequest::Seam {
            operation_deadline,
            never_returns,
            response,
        } => {
            let result = if never_returns {
                let deadline = TokioInstant::from_std(operation_deadline);
                let completed = finish_cleanup_before_deadline(
                    deadline,
                    std::future::pending::<()>(),
                    std::future::pending::<()>(),
                    std::future::pending::<()>(),
                )
                .await;
                if completed {
                    Ok(())
                } else {
                    Err(timeout_error())
                }
            } else {
                Ok(())
            };
            let _ = response.try_send(result);
        }
    }
}

async fn probe_executor_loop(mut receiver: tokio::sync::mpsc::Receiver<ProbeExecutorRequest>) {
    let mut tasks = JoinSet::new();
    loop {
        if tasks.len() >= FFPROBE_MAX_IN_FLIGHT {
            let _ = tasks.join_next().await;
            continue;
        }
        tokio::select! {
            completed = tasks.join_next(), if !tasks.is_empty() => {
                let _ = completed;
            }
            request = receiver.recv() => {
                let Some(request) = request else {
                    while tasks.join_next().await.is_some() {}
                    return;
                };
                tasks.spawn_local(execute_probe_request(request));
            }
        }
    }
}

fn initialize_probe_executor() -> Result<ProbeExecutor, String> {
    let (sender, receiver) = tokio::sync::mpsc::channel(FFPROBE_MAX_IN_FLIGHT);
    std::thread::Builder::new()
        .name("ffprobe-runtime".to_string())
        .spawn(move || {
            let runtime = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(_) => return,
            };
            let local = LocalSet::new();
            local.block_on(&runtime, probe_executor_loop(receiver));
        })
        .map_err(|error| format!("ffprobe runtime thread: {error}"))?;
    Ok(ProbeExecutor { sender })
}

fn probe_executor() -> crate::error::Result<&'static ProbeExecutor> {
    match FFPROBE_EXECUTOR.get_or_init(initialize_probe_executor) {
        Ok(executor) => Ok(executor),
        Err(error) => Err(crate::error::MediaError::Ffmpeg(error.clone())),
    }
}

fn receive_probe_response<T>(
    receiver: &Receiver<crate::error::Result<T>>,
    cancel: Option<&crate::MediaCancelToken>,
    api_deadline: Instant,
) -> crate::error::Result<T> {
    let mut cancelled = false;
    let mut response_deadline = api_deadline;
    loop {
        if !cancelled && cancel.is_some_and(crate::MediaCancelToken::checkpoint) {
            cancelled = true;
            response_deadline = response_deadline.min(
                Instant::now()
                    .checked_add(FFPROBE_CLEANUP_MAX)
                    .unwrap_or(response_deadline),
            );
        }
        let now = Instant::now();
        if now >= response_deadline {
            if let Ok(result) = receiver.try_recv() {
                return if cancelled {
                    Err(crate::error::MediaError::Cancelled)
                } else {
                    result
                };
            }
            return if cancelled {
                Err(crate::error::MediaError::Cancelled)
            } else {
                Err(timeout_error())
            };
        }
        let wait = FFPROBE_POLL_INTERVAL.min(response_deadline.saturating_duration_since(now));
        match receiver.recv_timeout(wait) {
            Ok(result) => {
                return if cancelled {
                    Err(crate::error::MediaError::Cancelled)
                } else {
                    result
                };
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                return Err(crate::error::MediaError::Ffmpeg(
                    "ffprobe runtime stopped unexpectedly".to_string(),
                ));
            }
        }
    }
}

/// Run one bounded ffprobe. `admission_wait` of zero fails at once when every
/// admission slot is taken; a positive wait queues (cancellably) for a slot.
fn run_ffprobe(
    executable: &std::ffi::OsStr,
    input_path: Option<&Path>,
    input_file: Option<&std::fs::File>,
    cancel: &crate::MediaCancelToken,
    timeout: Duration,
    admission_wait: Duration,
) -> crate::error::Result<FfprobeOutput> {
    #[cfg(test)]
    test_seams::record_probe_request();
    if cancel.checkpoint() {
        return Err(crate::error::MediaError::Cancelled);
    }
    record_helper_process();
    let queued_at = Instant::now();
    let _admission = if admission_wait.is_zero() {
        ProbeAdmission::acquire()?
    } else {
        ProbeAdmission::acquire_waiting(cancel, admission_wait)?
    };
    // Time spent queueing counts against `timeout`, so a queued probe never
    // takes longer than its caller's bound.
    let timeout = timeout.saturating_sub(queued_at.elapsed());
    let (operation_deadline, api_deadline) = probe_deadlines(timeout)?;
    let input = match (input_path, input_file) {
        (Some(path), None) => ProbeInput::Path(path.to_path_buf()),
        (None, Some(file)) => ProbeInput::File(file.try_clone().map_err(|error| {
            crate::error::MediaError::Ffmpeg(format!("ffprobe input clone: {error}"))
        })?),
        _ => {
            return Err(crate::error::MediaError::Ffmpeg(
                "ffprobe input missing".to_string(),
            ));
        }
    };
    let (response_tx, response_rx) = std::sync::mpsc::sync_channel(1);
    let request = ProbeExecutorRequest::Run {
        spec: ProbeSpec {
            executable: executable.to_os_string(),
            input,
            cancel: cancel.clone(),
            operation_deadline,
            api_deadline,
        },
        response: response_tx,
    };
    probe_executor()?
        .sender
        .try_send(request)
        .map_err(|error| {
            crate::error::MediaError::Ffmpeg(format!("ffprobe runtime unavailable: {error}"))
        })?;
    receive_probe_response(&response_rx, Some(cancel), api_deadline)
}

#[cfg(test)]
fn run_probe_executor_seam(never_returns: bool, timeout: Duration) -> crate::error::Result<()> {
    let _admission = ProbeAdmission::acquire()?;
    let (operation_deadline, api_deadline) = probe_deadlines(timeout)?;
    let (response_tx, response_rx) = std::sync::mpsc::sync_channel(1);
    probe_executor()?
        .sender
        .try_send(ProbeExecutorRequest::Seam {
            operation_deadline,
            never_returns,
            response: response_tx,
        })
        .map_err(|error| {
            crate::error::MediaError::Ffmpeg(format!("ffprobe runtime unavailable: {error}"))
        })?;
    receive_probe_response(&response_rx, None, api_deadline)
}

fn sidecar_filename(binary: &str) -> String {
    if cfg!(windows) {
        format!("{binary}.exe")
    } else {
        binary.to_string()
    }
}

/// Return a regular, non-symlink sidecar next to `executable`.
///
/// Keeping this pure helper separate makes the packaged-path security boundary
/// deterministic to test without mutating the process executable or PATH.
pub fn packaged_sidecar_beside(executable: &Path, binary: &str) -> Option<PathBuf> {
    let parent = executable.parent()?;
    let candidate = parent.join(sidecar_filename(binary));
    let metadata = std::fs::symlink_metadata(&candidate).ok()?;
    if metadata.file_type().is_file() && !metadata.file_type().is_symlink() {
        Some(candidate)
    } else {
        None
    }
}

/// Find a verified-by-the-package-manager sidecar beside the current executable.
/// Runtime code still checks that the path is a regular file; the build/package
/// pipeline owns its pinned SHA-256 and version verification.
pub fn packaged_sidecar_path(binary: &str) -> Option<PathBuf> {
    let executable = std::env::current_exe().ok()?;
    packaged_sidecar_beside(&executable, binary)
}

/// Resolve one CLI tool without mutating global process state in tests.
///
/// An explicit override always wins, then a regular non-symlink packaged
/// sidecar beside the executable, and finally the platform command name on
/// `PATH`.
fn resolve_cli_path(
    override_path: Option<OsString>,
    executable: Option<&Path>,
    binary: &str,
) -> OsString {
    override_path
        .or_else(|| {
            executable
                .and_then(|path| packaged_sidecar_beside(path, binary))
                .map(PathBuf::into_os_string)
        })
        .unwrap_or_else(|| OsString::from(binary))
}

/// Path to `ffmpeg`: explicit development override, packaged sidecar, then PATH.
pub fn ffmpeg_path() -> OsString {
    #[cfg(test)]
    if let Some(path) = test_seams::ffmpeg_override() {
        return path;
    }
    let executable = std::env::current_exe().ok();
    resolve_cli_path(
        std::env::var_os("OPENTAKE_FFMPEG"),
        executable.as_deref(),
        "ffmpeg",
    )
}

/// Path to `ffprobe`: explicit development override, packaged sidecar, then PATH.
pub fn ffprobe_path() -> OsString {
    #[cfg(test)]
    if let Some(path) = test_seams::ffprobe_override() {
        return path;
    }
    let executable = std::env::current_exe().ok();
    resolve_cli_path(
        std::env::var_os("OPENTAKE_FFPROBE"),
        executable.as_deref(),
        "ffprobe",
    )
}

/// A fresh `FfmpegCommand` bound to [`ffmpeg_path`]. Every caller spawns the
/// command it builds, so this is where FFmpeg helper processes are counted.
pub fn ffmpeg() -> FfmpegCommand {
    record_helper_process();
    FfmpegCommand::new_with_path(ffmpeg_path())
}

thread_local! {
    static HELPER_PROCESS_COUNT: RefCell<Option<Arc<AtomicUsize>>> = const { RefCell::new(None) };
}

/// Counts the FFmpeg and ffprobe processes an operation starts: those started
/// on this thread while the count is alive, including the ones started by
/// media workers (such as video decode streams) that this thread spawns.
/// Other threads are not counted, so parallel tests do not disturb each other.
/// Nested counts restore the outer one when dropped.
///
/// This is an observation seam for tests that bound how many helper processes
/// an operation needs (for example one decoder per clip rather than one per
/// frame). It counts processes launched through this module's `ffmpeg()`
/// builder and ffprobe runner.
#[doc(hidden)]
pub struct HelperProcessCount {
    counter: Arc<AtomicUsize>,
    outer: Option<Arc<AtomicUsize>>,
}

impl HelperProcessCount {
    pub fn start() -> Self {
        let counter = Arc::new(AtomicUsize::new(0));
        let outer = HELPER_PROCESS_COUNT.with(|slot| slot.replace(Some(Arc::clone(&counter))));
        HelperProcessCount { counter, outer }
    }

    /// Helper processes started so far.
    pub fn count(&self) -> usize {
        self.counter.load(Ordering::Acquire)
    }
}

impl Drop for HelperProcessCount {
    fn drop(&mut self) {
        let outer = self.outer.take();
        HELPER_PROCESS_COUNT.with(|slot| *slot.borrow_mut() = outer);
    }
}

fn record_helper_process() {
    HELPER_PROCESS_COUNT.with(|slot| {
        if let Some(counter) = slot.borrow().as_ref() {
            counter.fetch_add(1, Ordering::AcqRel);
        }
    });
}

/// The calling thread's active count, for a worker thread it spawns to adopt
/// with [`adopt_helper_process_count`].
pub(crate) fn inherited_helper_process_count() -> Option<Arc<AtomicUsize>> {
    HELPER_PROCESS_COUNT.with(|slot| slot.borrow().clone())
}

/// Count this (worker) thread's helper processes toward its spawner's count.
pub(crate) fn adopt_helper_process_count(counter: Option<Arc<AtomicUsize>>) {
    HELPER_PROCESS_COUNT.with(|slot| *slot.borrow_mut() = counter);
}

/// Whether `ffmpeg` is runnable (`-version` exits 0). Used by tests/integration
/// to skip when the binary is unavailable, keeping the default test run green on
/// machines without ffmpeg.
pub fn ffmpeg_available() -> bool {
    opentake_process_tree::background_command(ffmpeg_path())
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Whether `ffprobe` is runnable.
pub fn ffprobe_available() -> bool {
    run_ffprobe_availability(&ffprobe_path(), FFPROBE_TIMEOUT).unwrap_or(false)
}

fn run_ffprobe_availability(
    executable: &std::ffi::OsStr,
    timeout: Duration,
) -> crate::error::Result<bool> {
    let _admission = ProbeAdmission::acquire()?;
    let (operation_deadline, api_deadline) = probe_deadlines(timeout)?;
    let (response_tx, response_rx) = std::sync::mpsc::sync_channel(1);
    probe_executor()?
        .sender
        .try_send(ProbeExecutorRequest::Available {
            executable: executable.to_os_string(),
            operation_deadline,
            api_deadline,
            response: response_tx,
        })
        .map_err(|error| {
            crate::error::MediaError::Ffmpeg(format!("ffprobe runtime unavailable: {error}"))
        })?;
    receive_probe_response(&response_rx, None, api_deadline)
}

/// Run `ffprobe -of json -show_streams -show_format <path>` and return parsed
/// JSON. Zero decoding — header/stream parameters only.
pub fn ffprobe_json(path: &std::path::Path) -> crate::error::Result<serde_json::Value> {
    let executable = ffprobe_path();
    let out = run_ffprobe(
        &executable,
        Some(path),
        None,
        &crate::MediaCancelToken::new(),
        FFPROBE_TIMEOUT,
        Duration::ZERO,
    )?;
    if !out.status.success() {
        return Err(crate::error::MediaError::Ffmpeg(format!(
            "ffprobe exited {}",
            out.status
        )));
    }
    serde_json::from_slice(&out.stdout)
        .map_err(|e| crate::error::MediaError::Ffmpeg(format!("ffprobe json: {e}")))
}

/// Input of a queued probe: a pathname, or an already-open regular file
/// probed through `fd:` without resolving any pathname.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ProbeTarget<'a> {
    Path(&'a Path),
    File(&'a std::fs::File),
}

/// Outcome of a probe that ran to completion.
pub(crate) enum QueuedProbe {
    /// ffprobe read the input and reported its streams and format.
    Parsed(serde_json::Value),
    /// ffprobe ran and rejected the input as unreadable media (its error exit
    /// status 1). Unlike a helper failure this is a verdict on the input.
    Rejected,
}

/// Probe on behalf of a caller that is about to decode: the caller's
/// cancellation token stops the helper, `timeout` bounds it, and a saturated
/// admission limit is waited out for at most `admission_wait` rather than
/// failing immediately.
pub(crate) fn ffprobe_json_queued(
    target: ProbeTarget<'_>,
    cancel: &crate::MediaCancelToken,
    timeout: Duration,
    admission_wait: Duration,
) -> crate::error::Result<QueuedProbe> {
    let executable = ffprobe_path();
    let (path, file) = match target {
        ProbeTarget::Path(path) => (Some(path), None),
        ProbeTarget::File(file) => (None, Some(file)),
    };
    let out = run_ffprobe(&executable, path, file, cancel, timeout, admission_wait)?;
    if out.status.code() == Some(1) {
        return Ok(QueuedProbe::Rejected);
    }
    if !out.status.success() {
        return Err(crate::error::MediaError::Ffmpeg(format!(
            "ffprobe exited {}",
            out.status
        )));
    }
    serde_json::from_slice(&out.stdout)
        .map(QueuedProbe::Parsed)
        .map_err(|e| crate::error::MediaError::Ffmpeg(format!("ffprobe json: {e}")))
}

/// Probe an already-open regular file without resolving an ambient pathname.
/// ffprobe's `fd:` protocol keeps normal file seek semantics (unlike `pipe:`),
/// which is required by containers whose metadata lives near the end.
pub fn ffprobe_json_file(file: &std::fs::File) -> crate::error::Result<serde_json::Value> {
    // Uncancellable callers must not block on the queue: a saturated limit
    // fails at once with `MediaError::Busy`.
    ffprobe_json_file_with_wait(
        file,
        &crate::MediaCancelToken::new(),
        FFPROBE_TIMEOUT,
        Duration::ZERO,
    )
}

/// Cancellable probe of an open file, used by imports and proxy builds that
/// run off the UI thread: a saturated admission limit is waited out
/// (cancellably, for up to half of `timeout`) instead of failing at once (#63).
pub fn ffprobe_json_file_cancellable(
    file: &std::fs::File,
    cancel: &crate::MediaCancelToken,
    timeout: Duration,
) -> crate::error::Result<serde_json::Value> {
    ffprobe_json_file_with_wait(file, cancel, timeout, timeout / 2)
}

fn ffprobe_json_file_with_wait(
    file: &std::fs::File,
    cancel: &crate::MediaCancelToken,
    timeout: Duration,
    admission_wait: Duration,
) -> crate::error::Result<serde_json::Value> {
    let executable = ffprobe_path();
    let out = run_ffprobe(
        &executable,
        None,
        Some(file),
        cancel,
        timeout,
        admission_wait,
    )?;
    if !out.status.success() {
        return Err(crate::error::MediaError::Ffmpeg(format!(
            "ffprobe fd input exited {}",
            out.status
        )));
    }
    serde_json::from_slice(&out.stdout)
        .map_err(|e| crate::error::MediaError::Ffmpeg(format!("ffprobe json: {e}")))
}

/// Per-thread fault and observation seams for helper tests. The admission
/// counter, probe runtime, and caches are process-wide and tests run in
/// parallel, so every seam applies only to the calling thread (admission,
/// executable resolution, and request accounting all happen on the thread
/// that asks for the probe).
#[cfg(test)]
pub(crate) mod test_seams {
    use std::cell::{Cell, RefCell};
    use std::ffi::OsString;
    use std::time::Duration;

    thread_local! {
        static ADMISSION_SATURATED: Cell<bool> = const { Cell::new(false) };
        static ADMISSION_SATURATED_ATTEMPTS: Cell<usize> = const { Cell::new(0) };
        static ADMISSION_WAIT: Cell<Option<Duration>> = const { Cell::new(None) };
        static PROBE_REQUESTS: Cell<usize> = const { Cell::new(0) };
        static FFPROBE_OVERRIDE: RefCell<Option<OsString>> = const { RefCell::new(None) };
        static FFMPEG_OVERRIDE: RefCell<Option<OsString>> = const { RefCell::new(None) };
    }

    /// Make every admission attempt on this thread behave as if all slots
    /// were taken by other work.
    pub(crate) fn saturate_admission(saturated: bool) {
        ADMISSION_SATURATED.with(|cell| cell.set(saturated));
    }

    pub(crate) fn admission_saturated() -> bool {
        ADMISSION_SATURATED.with(Cell::get)
    }

    /// Reject the next `attempts` admission attempts on this thread, as if a
    /// slot were freed by other work only afterwards.
    pub(crate) fn saturate_admission_attempts(attempts: usize) {
        ADMISSION_SATURATED_ATTEMPTS.with(|cell| cell.set(attempts));
    }

    pub(crate) fn take_saturated_attempt() -> bool {
        ADMISSION_SATURATED_ATTEMPTS.with(|cell| {
            let remaining = cell.get();
            cell.set(remaining.saturating_sub(1));
            remaining > 0
        })
    }

    /// Shorten (or restore with `None`) the queued-admission wait on this thread.
    pub(crate) fn override_admission_wait(wait: Option<Duration>) {
        ADMISSION_WAIT.with(|cell| cell.set(wait));
    }

    pub(crate) fn admission_wait_override() -> Option<Duration> {
        ADMISSION_WAIT.with(Cell::get)
    }

    pub(crate) fn record_probe_request() {
        PROBE_REQUESTS.with(|cell| cell.set(cell.get() + 1));
    }

    /// Number of ffprobe requests this thread has made so far.
    pub(crate) fn probe_requests() -> usize {
        PROBE_REQUESTS.with(Cell::get)
    }

    /// Resolve `ffprobe` to `path` for probes requested by this thread. Only
    /// Unix tests stand in a shell script for the executable.
    #[cfg(unix)]
    pub(crate) fn override_ffprobe(path: Option<OsString>) {
        FFPROBE_OVERRIDE.with(|cell| *cell.borrow_mut() = path);
    }

    pub(crate) fn ffprobe_override() -> Option<OsString> {
        FFPROBE_OVERRIDE.with(|cell| cell.borrow().clone())
    }

    /// Resolve `ffmpeg` to `path` for commands built on this thread (Unix
    /// tests only, like `override_ffprobe`).
    #[cfg(unix)]
    pub(crate) fn override_ffmpeg(path: Option<OsString>) {
        FFMPEG_OVERRIDE.with(|cell| *cell.borrow_mut() = path);
    }

    pub(crate) fn ffmpeg_override() -> Option<OsString> {
        FFMPEG_OVERRIDE.with(|cell| cell.borrow().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saturated_admission_queues_until_a_slot_frees() {
        test_seams::saturate_admission_attempts(3);
        let admission = ProbeAdmission::acquire_waiting(
            &crate::MediaCancelToken::new(),
            Duration::from_secs(5),
        );
        test_seams::saturate_admission_attempts(0);
        assert!(admission.is_ok(), "a queued probe must get the freed slot");
    }

    #[test]
    fn saturated_admission_reports_busy_not_a_media_error() {
        test_seams::saturate_admission(true);
        let immediate = ProbeAdmission::acquire();
        let waited = ProbeAdmission::acquire_waiting(
            &crate::MediaCancelToken::new(),
            Duration::from_millis(60),
        );
        test_seams::saturate_admission(false);
        assert!(matches!(immediate, Err(crate::MediaError::Busy(_))));
        assert!(matches!(waited, Err(crate::MediaError::Busy(_))));
    }

    #[test]
    fn cancelling_a_queued_admission_returns_promptly() {
        let cancel = crate::MediaCancelToken::new();
        let canceller = cancel.clone();
        let started = Instant::now();
        let trigger = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            canceller.cancel();
        });
        test_seams::saturate_admission(true);
        let result = ProbeAdmission::acquire_waiting(&cancel, Duration::from_secs(30));
        test_seams::saturate_admission(false);
        trigger.join().unwrap();
        assert!(matches!(result, Err(crate::MediaError::Cancelled)));
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn helper_process_count_follows_the_thread_and_its_adopting_workers() {
        let outer = HelperProcessCount::start();
        let _ = ffmpeg();
        {
            let inner = HelperProcessCount::start();
            let _ = ffmpeg();
            let inherited = inherited_helper_process_count();
            std::thread::spawn(move || {
                adopt_helper_process_count(inherited);
                let _ = ffmpeg();
            })
            .join()
            .unwrap();
            // A thread that did not adopt the count is not counted.
            std::thread::spawn(|| {
                let _ = ffmpeg();
            })
            .join()
            .unwrap();
            assert_eq!(inner.count(), 2);
        }
        let _ = ffmpeg();
        assert_eq!(outer.count(), 2, "the outer count resumes after the inner");
        drop(outer);
        assert!(inherited_helper_process_count().is_none());
    }

    #[cfg(unix)]
    fn assert_capture_limit_kills_tree(output_redirection: &str, expected_stream: &str) {
        use std::os::unix::fs::PermissionsExt;

        let temp = tempfile::tempdir().unwrap();
        let script = temp.path().join("fake-ffprobe");
        let capture = temp.path().join("pids");
        std::fs::write(
            &script,
            format!(
                "#!/bin/sh\nsleep 60 &\nprintf 'parent=%s\\nchild=%s\\n' \"$$\" \"$!\" > '{}'\ndd if=/dev/zero bs=1048576 count=9 {output_redirection}\nwait\n",
                capture.display()
            ),
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&script).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&script, permissions).unwrap();

        let input = tempfile::tempfile().unwrap();
        // The probe executor is a single background thread shared by every
        // ffprobe test. Under parallel test load that thread can be starved long
        // enough for the wall-clock operation deadline to win over the bounded
        // capture, which yields "ffprobe timed out" before the capture limit has
        // a chance to fire — such an attempt proves nothing about the capture
        // limit. Retry starved attempts; only an attempt that actually ran the
        // probe can carry the assertions below, which are unchanged.
        let mut result = None;
        for _ in 0..3 {
            let attempt = run_ffprobe(
                script.as_os_str(),
                None,
                Some(&input),
                &crate::MediaCancelToken::new(),
                Duration::from_secs(3),
                Duration::ZERO,
            );
            let starved = matches!(
                &attempt,
                Err(crate::MediaError::Ffmpeg(message)) if message == "ffprobe timed out"
            ) || matches!(&attempt, Err(crate::MediaError::Busy(_)));
            if !starved {
                result = Some(attempt);
                break;
            }
        }
        let result = result.expect("every attempt was starved past the operation deadline");
        let Err(crate::MediaError::Ffmpeg(message)) = result else {
            panic!("expected bounded capture error");
        };
        assert!(
            message.contains(expected_stream) && message.contains("bounded capture"),
            "unexpected capture error: {message}"
        );

        let pids = std::fs::read_to_string(capture).unwrap();
        for pid in pids
            .lines()
            .filter_map(|line| line.split_once('=').map(|(_, pid)| pid))
        {
            let exit_deadline = Instant::now() + Duration::from_secs(2);
            while Command::new("kill")
                .args(["-0", pid])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .unwrap()
                .success()
                && Instant::now() < exit_deadline
            {
                std::thread::sleep(Duration::from_millis(10));
            }
            assert!(
                !Command::new("kill")
                    .args(["-0", pid])
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .unwrap()
                    .success(),
                "ffprobe process tree member {pid} survived capture overflow"
            );
        }
    }

    #[test]
    fn env_override_is_respected_for_ffmpeg() {
        let temp = tempfile::tempdir().unwrap();
        let executable = temp.path().join(if cfg!(windows) {
            "opentake.exe"
        } else {
            "opentake"
        });
        std::fs::write(&executable, b"app").unwrap();
        std::fs::write(temp.path().join(sidecar_filename("ffmpeg")), b"sidecar").unwrap();

        assert_eq!(
            resolve_cli_path(
                Some(OsString::from("/opt/opentake/custom-ffmpeg")),
                Some(&executable),
                "ffmpeg",
            ),
            OsString::from("/opt/opentake/custom-ffmpeg"),
        );
    }

    #[test]
    fn default_ffprobe_is_ffprobe() {
        assert_eq!(
            resolve_cli_path(None, None, "ffprobe"),
            OsString::from("ffprobe")
        );
    }

    #[test]
    fn nonreturning_wait_and_drain_are_abandoned_without_poisoning_executor() {
        let started = Instant::now();
        for _ in 0..4 {
            let result = run_probe_executor_seam(true, Duration::from_millis(80));
            let Err(crate::MediaError::Ffmpeg(message)) = result else {
                panic!("expected the injected cleanup deadline to expire");
            };
            assert!(message.contains("timed out"));
        }
        run_probe_executor_seam(false, Duration::from_secs(1))
            .expect("healthy work must run after four abandoned cleanup futures");
        // The four 80ms abandoned cleanups plus the healthy run finish in well
        // under a second locally; the bound only guards against a cleanup
        // deadline that never fires at all, so keep it generous for loaded CI.
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "nonreturning cleanup work escaped its deadline"
        );
    }

    #[test]
    fn synchronous_dispatch_is_safe_inside_an_existing_tokio_runtime() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            run_probe_executor_seam(false, Duration::from_secs(1))
                .expect("sync dispatch must use the independent ffprobe runtime");
        });
    }

    #[cfg(unix)]
    #[test]
    fn availability_probe_uses_deadline_and_process_tree_containment() {
        use std::os::unix::fs::PermissionsExt;

        let temp = tempfile::tempdir().unwrap();
        let script = temp.path().join("fake-ffprobe-version");
        let capture = temp.path().join("version-pids");
        std::fs::write(
            &script,
            format!(
                "#!/bin/sh\nsleep 60 &\nprintf 'parent=%s\\nchild=%s\\n' \"$$\" \"$!\" > '{}'\nwait\n",
                capture.display()
            ),
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&script).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&script, permissions).unwrap();

        let started = Instant::now();
        let result = run_ffprobe_availability(script.as_os_str(), Duration::from_secs(1));
        assert!(started.elapsed() < Duration::from_secs(2));
        let Err(crate::MediaError::Ffmpeg(message)) = result else {
            panic!("expected availability timeout");
        };
        assert!(message.contains("timed out"));

        // The probe executor is single-threaded; if this request dequeues after
        // its operation deadline, the pre-spawn deadline check returns "timed
        // out" without ever spawning, so no capture file exists and there is no
        // process tree to assert on. The "timed out" message above already
        // asserts the probe failed closed. A capture file that exists but cannot
        // be read is a real test failure.
        let pids = match std::fs::read_to_string(&capture) {
            Ok(pids) => Some(pids),
            Err(_) if !capture.exists() => None,
            Err(error) => panic!("failed to read ffprobe capture file: {error}"),
        };
        if let Some(pids) = pids {
            for pid in pids
                .lines()
                .filter_map(|line| line.split_once('=').map(|(_, pid)| pid))
            {
                // A SIGKILLed member may linger as a zombie until reaped by
                // init; poll with a bounded deadline (mirrors
                // assert_capture_limit_kills_tree) so CI scheduling latency
                // cannot flake the assertion.
                let exit_deadline = Instant::now() + Duration::from_secs(5);
                while Command::new("kill")
                    .args(["-0", pid])
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .unwrap()
                    .success()
                    && Instant::now() < exit_deadline
                {
                    std::thread::sleep(Duration::from_millis(50));
                }
                assert!(
                    !Command::new("kill")
                        .args(["-0", pid])
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .status()
                        .unwrap()
                        .success(),
                    "ffprobe availability process tree member {pid} survived"
                );
            }
        }
    }

    #[test]
    fn packaged_sidecar_must_be_regular_and_beside_executable() {
        let temp = tempfile::tempdir().unwrap();
        let executable = temp.path().join(if cfg!(windows) {
            "opentake.exe"
        } else {
            "opentake"
        });
        std::fs::write(&executable, b"app").unwrap();
        let sidecar = temp.path().join(sidecar_filename("ffmpeg"));
        std::fs::write(&sidecar, b"sidecar").unwrap();

        assert_eq!(
            packaged_sidecar_beside(&executable, "ffmpeg"),
            Some(sidecar.clone())
        );
        assert_eq!(
            resolve_cli_path(None, Some(&executable), "ffmpeg"),
            sidecar.into_os_string()
        );
        assert_eq!(packaged_sidecar_beside(&executable, "ffprobe"), None);
    }

    #[cfg(unix)]
    #[test]
    fn stdout_capture_limit_terminates_ffprobe_tree_immediately() {
        assert_capture_limit_kills_tree("2>/dev/null", "stdout");
    }

    #[cfg(unix)]
    #[test]
    fn stderr_capture_limit_terminates_ffprobe_tree_immediately() {
        assert_capture_limit_kills_tree("1>&2 2>/dev/null", "stderr");
    }

    #[cfg(unix)]
    #[test]
    fn cancellable_ffprobe_kills_descendants_that_inherit_helper_resources() {
        use std::os::unix::fs::PermissionsExt;
        use std::sync::mpsc;

        let temp = tempfile::tempdir().unwrap();
        let script = temp.path().join("fake-ffprobe");
        let capture = temp.path().join("pids");
        std::fs::write(
            &script,
            format!(
                "#!/bin/sh\nsleep 60 &\nprintf 'parent=%s\\nchild=%s\\n' \"$$\" \"$!\" > '{}'\nwait\n",
                capture.display()
            ),
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&script).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&script, permissions).unwrap();
        let input = tempfile::tempfile().unwrap();
        let cancel = crate::MediaCancelToken::new();
        let worker_cancel = cancel.clone();
        let worker_script = script.clone();
        let (done_tx, done_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let result = run_ffprobe(
                worker_script.as_os_str(),
                None,
                Some(&input),
                &worker_cancel,
                Duration::from_secs(60),
                Duration::ZERO,
            );
            done_tx.send(result).unwrap();
        });
        let ready_deadline = Instant::now() + Duration::from_secs(5);
        while (!capture.exists() || cancel.spawned_child_count() == 0)
            && Instant::now() < ready_deadline
        {
            std::thread::yield_now();
        }
        assert!(capture.exists(), "fake ffprobe entered");
        cancel.cancel();
        let result = done_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("cancelled ffprobe returned");
        assert!(matches!(result, Err(crate::MediaError::Cancelled)));
        worker.join().unwrap();

        let pids = std::fs::read_to_string(capture).unwrap();
        for pid in pids
            .lines()
            .filter_map(|line| line.split_once('=').map(|(_, pid)| pid))
        {
            // SIGKILLed members may linger as zombies until init reaps them;
            // poll with a bounded deadline (mirrors
            // assert_capture_limit_kills_tree) so CI scheduling latency
            // cannot flake the assertion.
            let exit_deadline = Instant::now() + Duration::from_secs(5);
            while Command::new("kill")
                .args(["-0", pid])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .unwrap()
                .success()
                && Instant::now() < exit_deadline
            {
                std::thread::sleep(Duration::from_millis(50));
            }
            assert!(
                !Command::new("kill")
                    .args(["-0", pid])
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .unwrap()
                    .success(),
                "ffprobe process tree member {pid} survived"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn packaged_sidecar_rejects_symlink() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let executable = temp.path().join("opentake");
        let outside = temp.path().join("outside");
        let sidecar = temp.path().join("ffmpeg");
        std::fs::write(&executable, b"app").unwrap();
        std::fs::write(&outside, b"untrusted").unwrap();
        symlink(&outside, &sidecar).unwrap();

        assert_eq!(packaged_sidecar_beside(&executable, "ffmpeg"), None);
    }
}
