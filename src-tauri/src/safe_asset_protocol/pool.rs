//! Bounded, reused pool of isolated asset-reader processes.
//!
//! Each helper is a contained copy of the application binary started in
//! helper mode (see `helper.rs`). It authenticates its parent once, then
//! serves any number of length-prefixed requests, each carrying its own random
//! token. A helper that times out, violates the protocol or answers for the
//! wrong request is killed and replaced; one that cannot be reaped keeps its
//! process slot in quarantine, so a stuck volume can never grow the number of
//! live processes beyond [`HELPER_POOL_SIZE`].

use super::helper::{
    read_helper_reply, write_helper_frame, HelperFrame, HelperReply, OpenedMetadata,
    HELPER_HANDSHAKE,
};
use super::*;
use opentake_media::process_tree::ProcessTree;
use std::ffi::OsString;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::OwnedSemaphorePermit;

/// Live plus quarantined helper processes. Equal to the number of concurrent
/// reads, so every admitted read can own one helper.
pub(super) const HELPER_POOL_SIZE: usize = MAX_CONCURRENT_READS;
/// Idle helpers exit after this long without a request.
const HELPER_IDLE_TIMEOUT: Duration = Duration::from_secs(30);
/// Starting a process can take seconds on a loaded machine (or while an
/// antivirus scans the executable); this bounds spawn plus authentication.
const HELPER_HANDSHAKE_DEADLINE: Duration = Duration::from_secs(10);

static SHARED_POOL: OnceLock<Arc<HelperPool>> = OnceLock::new();

/// How to start one helper process.
pub(super) struct HelperLauncher {
    pub(super) program: Option<PathBuf>,
    pub(super) args: Vec<OsString>,
    pub(super) env: Vec<(OsString, OsString)>,
    /// Bytes the child may print before its handshake. Production helpers
    /// must start with the handshake; the unit-test harness prints a banner.
    pub(super) max_preamble_bytes: usize,
}

impl HelperLauncher {
    fn application() -> Self {
        Self {
            program: None,
            args: vec![HELPER_ARG.into()],
            env: Vec::new(),
            max_preamble_bytes: 0,
        }
    }
}

pub(super) struct HelperPool {
    launcher: HelperLauncher,
    /// One permit per live or quarantined process.
    process_slots: Arc<Semaphore>,
    idle: Mutex<Vec<PooledHelper>>,
    /// Bumped by [`HelperPool::retire_all`]; helpers from an older generation
    /// are killed instead of being returned to `idle`.
    generation: AtomicU64,
    spawned: AtomicUsize,
    sweeper_scheduled: AtomicBool,
    deadline: Duration,
}

struct PooledHelper {
    child: Child,
    tree: ProcessTree,
    stdin: ChildStdin,
    stdout: ChildStdout,
    /// Process-wide secret shared through the environment; every request
    /// frame must repeat it.
    session: String,
    slot: OwnedSemaphorePermit,
    generation: u64,
    idle_since: Instant,
}

/// Result of one request/authorization/serve exchange with a helper.
pub(super) enum HelperOutcome<E> {
    /// The helper could not open a resident regular file; nothing was read.
    OpenFailed(OpenedMetadata),
    /// The parent refused the opened identity; nothing was read.
    Refused(E),
    /// The parent authorized the opened identity and the helper served it.
    Served {
        opened: OpenedMetadata,
        response: IsolatedResponse,
    },
}

impl HelperPool {
    /// The application's pool, shared by both asset schemes and reaped by the
    /// project-switch and exit hooks.
    pub(super) fn shared() -> Arc<Self> {
        SHARED_POOL
            .get_or_init(|| Arc::new(Self::new(HelperLauncher::application(), IO_DEADLINE)))
            .clone()
    }

    pub(super) fn new(launcher: HelperLauncher, deadline: Duration) -> Self {
        Self::with_slots(launcher, deadline, HELPER_POOL_SIZE)
    }

    pub(super) fn with_slots(launcher: HelperLauncher, deadline: Duration, slots: usize) -> Self {
        Self {
            launcher,
            process_slots: Arc::new(Semaphore::new(slots)),
            idle: Mutex::new(Vec::new()),
            generation: AtomicU64::new(0),
            spawned: AtomicUsize::new(0),
            sweeper_scheduled: AtomicBool::new(false),
            deadline,
        }
    }

    /// Test hook: helper processes started by this pool.
    #[cfg(test)]
    pub(super) fn spawned(&self) -> usize {
        self.spawned.load(Ordering::Relaxed)
    }

    /// Test hook: process slots neither live nor quarantined.
    #[cfg(test)]
    pub(super) fn available_slots(&self) -> usize {
        self.process_slots.available_permits()
    }

    /// Test hook: process ids of helpers waiting for a request.
    #[cfg(test)]
    pub(super) fn idle_helper_ids(&self) -> Vec<u32> {
        self.lock_idle()
            .iter()
            .filter_map(|helper| helper.child.id())
            .collect()
    }

    /// Test hook: retire every idle helper and wait for the bounded reap.
    #[cfg(all(test, unix))]
    pub(super) async fn retire_idle_for_test(&self) {
        self.generation.fetch_add(1, Ordering::AcqRel);
        self.retire_idle_helpers(true).await;
    }

    fn lock_idle(&self) -> std::sync::MutexGuard<'_, Vec<PooledHelper>> {
        self.idle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Send one request, let `authorize` inspect the opened file identity
    /// before any byte is read, and collect the served response.
    pub(super) async fn exchange<E>(
        self: &Arc<Self>,
        request: &HelperRequest,
        authorize: impl FnOnce(&OpenedMetadata) -> Result<(), E>,
    ) -> Result<HelperOutcome<E>, IsolatedHelperError> {
        let mut helper = self.checkout().await?;
        let result = tokio::time::timeout(
            self.deadline,
            exchange_with(&mut helper, request, authorize),
        )
        .await;
        match result {
            Ok(Ok(outcome)) => {
                self.checkin(helper);
                Ok(outcome)
            }
            Ok(Err(error)) => {
                retire(helper).await;
                Err(error)
            }
            Err(_) => {
                retire(helper).await;
                Err(IsolatedHelperError::TimedOut)
            }
        }
    }

    async fn checkout(self: &Arc<Self>) -> Result<PooledHelper, IsolatedHelperError> {
        let generation = self.generation.load(Ordering::Acquire);
        let mut expired = Vec::new();
        let reusable = {
            let mut idle = self.lock_idle();
            let mut reusable = None;
            while let Some(helper) = idle.pop() {
                if helper.generation == generation
                    && helper.idle_since.elapsed() < HELPER_IDLE_TIMEOUT
                {
                    reusable = Some(helper);
                    break;
                }
                expired.push(helper);
            }
            reusable
        };
        for helper in expired {
            retire(helper).await;
        }
        if let Some(helper) = reusable {
            return Ok(helper);
        }
        // Reserve the quarantine capacity before spawning. Once every slot is
        // held by a live or quarantined helper no further process is created.
        let slot = self
            .process_slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| IsolatedHelperError::Degraded)?;
        self.spawn(slot, generation).await
    }

    async fn spawn(
        &self,
        slot: OwnedSemaphorePermit,
        generation: u64,
    ) -> Result<PooledHelper, IsolatedHelperError> {
        let program = match &self.launcher.program {
            Some(program) => program.clone(),
            None => std::env::current_exe().map_err(|_| IsolatedHelperError::Io)?,
        };
        let session = random_token();
        let mut command = std::process::Command::new(program);
        command
            .args(&self.launcher.args)
            .envs(self.launcher.env.iter().map(|(key, value)| (key, value)))
            .env(HELPER_TOKEN_ENV, &session)
            .env(HELPER_PARENT_ENV, std::process::id().to_string())
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null());
        opentake_media::process_tree::configure_command(&mut command);
        let mut command = Command::from(command);
        command.kill_on_drop(true);
        let mut child = command.spawn().map_err(|_| IsolatedHelperError::Io)?;
        self.spawned.fetch_add(1, Ordering::Relaxed);
        // Windows starts the configured child suspended; attaching places it
        // in its kill-on-close job before it runs.
        let tree = match child.id().map(ProcessTree::attach) {
            Some(Ok(tree)) => tree,
            _ => {
                let _ = child.start_kill();
                terminate_or_quarantine(child, None, slot).await;
                return Err(IsolatedHelperError::Io);
            }
        };
        let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
            let _ = tree.terminate();
            let _ = child.start_kill();
            terminate_or_quarantine(child, Some(tree), slot).await;
            return Err(IsolatedHelperError::Io);
        };
        let mut helper = PooledHelper {
            child,
            tree,
            stdin,
            stdout,
            session,
            slot,
            generation,
            idle_since: Instant::now(),
        };
        let handshake = tokio::time::timeout(
            HELPER_HANDSHAKE_DEADLINE,
            read_handshake(&mut helper.stdout, self.launcher.max_preamble_bytes),
        )
        .await;
        if !matches!(handshake, Ok(Ok(()))) {
            retire(helper).await;
            return Err(IsolatedHelperError::Io);
        }
        Ok(helper)
    }

    fn checkin(self: &Arc<Self>, mut helper: PooledHelper) {
        if helper.generation != self.generation.load(Ordering::Acquire) {
            tokio::spawn(retire(helper));
            return;
        }
        helper.idle_since = Instant::now();
        self.lock_idle().push(helper);
        if self.sweeper_scheduled.swap(true, Ordering::AcqRel) {
            return;
        }
        // One sweeper per pool runs while any helper is idle.
        let pool = self.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(HELPER_IDLE_TIMEOUT).await;
                pool.retire_idle_helpers(false).await;
                if !pool.lock_idle().is_empty() {
                    continue;
                }
                pool.sweeper_scheduled.store(false, Ordering::Release);
                // A helper checked in after the emptiness test but before the
                // flag was cleared would otherwise never be swept.
                if pool.lock_idle().is_empty()
                    || pool.sweeper_scheduled.swap(true, Ordering::AcqRel)
                {
                    break;
                }
            }
        });
    }

    async fn retire_idle_helpers(&self, all: bool) {
        let retired = {
            let mut idle = self.lock_idle();
            let (retired, kept): (Vec<_>, Vec<_>) = idle
                .drain(..)
                .partition(|helper| all || helper.idle_since.elapsed() >= HELPER_IDLE_TIMEOUT);
            *idle = kept;
            retired
        };
        for helper in retired {
            retire(helper).await;
        }
    }

    /// Kill every idle helper and make busy ones exit when they return. Used
    /// when the project changes, so no helper outlives the authority it was
    /// started under.
    pub(super) fn retire_all(self: &Arc<Self>) {
        self.generation.fetch_add(1, Ordering::AcqRel);
        let pool = self.clone();
        tauri::async_runtime::spawn(async move { pool.retire_idle_helpers(true).await });
    }

    /// Synchronous shutdown for application exit: kill every idle helper's
    /// process tree now. Busy helpers are killed when their exchange returns,
    /// and `kill_on_drop` plus stdin EOF stop any that outlive the runtime.
    pub(super) fn shutdown(&self) {
        self.generation.fetch_add(1, Ordering::AcqRel);
        let helpers = std::mem::take(&mut *self.lock_idle());
        for mut helper in helpers {
            let _ = helper.tree.terminate();
            let _ = helper.child.start_kill();
        }
    }
}

async fn exchange_with<E>(
    helper: &mut PooledHelper,
    request: &HelperRequest,
    authorize: impl FnOnce(&OpenedMetadata) -> Result<(), E>,
) -> Result<HelperOutcome<E>, IsolatedHelperError> {
    write_helper_frame(
        &mut helper.stdin,
        &HelperFrame::Request {
            session: helper.session.clone(),
            request: request.clone(),
        },
    )
    .await?;
    let (reply, body) = read_helper_reply(&mut helper.stdout).await?;
    let HelperReply::Opened(opened) = reply else {
        return Err(IsolatedHelperError::InvalidResponse);
    };
    if opened.token != request.token || !body.is_empty() {
        return Err(IsolatedHelperError::InvalidResponse);
    }
    if opened.error_kind.is_some() {
        return Ok(HelperOutcome::OpenFailed(opened));
    }
    if let Err(refusal) = authorize(&opened) {
        write_helper_frame(
            &mut helper.stdin,
            &HelperFrame::Abort {
                token: request.token.clone(),
            },
        )
        .await?;
        return Ok(HelperOutcome::Refused(refusal));
    }
    write_helper_frame(
        &mut helper.stdin,
        &HelperFrame::Proceed {
            token: request.token.clone(),
        },
    )
    .await?;
    let (reply, body) = read_helper_reply(&mut helper.stdout).await?;
    let HelperReply::Served(metadata) = reply else {
        return Err(IsolatedHelperError::InvalidResponse);
    };
    if metadata.token != request.token {
        return Err(IsolatedHelperError::InvalidResponse);
    }
    Ok(HelperOutcome::Served {
        opened,
        response: IsolatedResponse { metadata, body },
    })
}

async fn read_handshake(
    stdout: &mut ChildStdout,
    max_preamble_bytes: usize,
) -> Result<(), IsolatedHelperError> {
    let mut window = Vec::with_capacity(HELPER_HANDSHAKE.len());
    let mut skipped = 0_usize;
    loop {
        let mut byte = [0_u8; 1];
        stdout
            .read_exact(&mut byte)
            .await
            .map_err(|_| IsolatedHelperError::Io)?;
        window.push(byte[0]);
        if window.len() > HELPER_HANDSHAKE.len() {
            window.remove(0);
            skipped += 1;
            if skipped > max_preamble_bytes {
                return Err(IsolatedHelperError::InvalidResponse);
            }
        }
        if window == HELPER_HANDSHAKE {
            return Ok(());
        }
    }
}

/// Kill one helper's process tree and reap it within [`REAP_DEADLINE`]. A
/// process stuck in an uninterruptible kernel wait keeps its slot (and its
/// armed containment) in a background task until it finally exits.
async fn retire(helper: PooledHelper) {
    let PooledHelper {
        child,
        tree,
        stdin,
        stdout,
        slot,
        ..
    } = helper;
    drop(stdin);
    drop(stdout);
    let _ = tree.terminate();
    terminate_or_quarantine(child, Some(tree), slot).await;
}

pub(super) async fn terminate_or_quarantine(
    mut child: Child,
    tree: Option<ProcessTree>,
    process_slot: OwnedSemaphorePermit,
) {
    let _ = child.start_kill();
    // Never turn the helper deadline into another unbounded wait;
    // kill_on_drop and the armed tree remain if this bounded reap expires.
    if bounded_reap(child.wait(), REAP_DEADLINE).await {
        // The group leader was reaped: its id may be reused, so release the
        // containment without signalling the group again.
        if let Some(mut tree) = tree {
            tree.disarm();
        }
        return;
    }
    tokio::spawn(async move {
        let _process_slot = process_slot;
        let _ = child.wait().await;
        if let Some(mut tree) = tree {
            tree.disarm();
        }
    });
}

pub(super) async fn bounded_reap<F>(wait: F, deadline: Duration) -> bool
where
    F: std::future::Future<Output = std::io::Result<std::process::ExitStatus>>,
{
    tokio::time::timeout(deadline, wait).await.is_ok()
}

pub(super) fn random_token() -> String {
    format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    )
}

/// Reap every pooled helper when the application exits.
pub(crate) fn shutdown_helper_pool() {
    if let Some(pool) = SHARED_POOL.get() {
        pool.shutdown();
    }
}

/// Retire every pooled helper after the open project changed.
pub(crate) fn retire_helper_pool() {
    if let Some(pool) = SHARED_POOL.get() {
        pool.retire_all();
    }
}
