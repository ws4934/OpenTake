//! Production asynchronous generation bridge shared by MCP and in-app Chat.
//!
//! Durable placeholder state is committed before provider submission. Provider
//! keys stay in the OS keychain; signed result URLs and provider diagnostics are
//! never persisted. Terminal downloads are probed, then streamed into the same
//! complete-bundle publication that makes the original placeholder ready.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::generation_orphans::{OrphanedGeneration, OrphanedGenerationStore};
use base64::Engine as _;
use futures_util::StreamExt;
use opentake_agent::mcp::generation::{
    finalize_terminal_outputs, DownloadedGenerationArtifact, FinishedOutput,
    GenerationArtifactDownloader, GenerationBridge, GenerationFinalizationStore, GenerationRequest,
    GenerationSubmission,
};
use opentake_agent::tools::args::{
    GenerateAudioArgs, GenerateImageArgs, GenerateVideoArgs, UpscaleMediaArgs,
};
use opentake_core::{
    AppCore, GenerationStateUpdate, PreparedGenerationJob, PreparedGenerationOutput, ProbedMedia,
};
use opentake_domain::{ClipType, GenerationInput, GenerationJobStatus, MediaResolver, Timeline};
use opentake_gen::catalog::cost::cost_for_input;
use opentake_gen::upscale::VIDEO_UPSCALE_DEFAULT_RESOLUTION;
use opentake_gen::{
    build_params, plan_video_upscale, upscale_result_matches, video_upscale_resolution, Catalog,
    CatalogEntry, ElevenLabsAdapter, FalAdapter, GenClient, GenError, GenerationJob,
    GenerationParams, JobStatus, KeyStore, KeyringStore, ModelKind, ModelRoute, OpenAiAdapter,
    ProviderKey, ProviderRegistry, ReplicateAdapter, ReqwestTransport, StaticToken, UiCapabilities,
    WatchEvent,
};
use opentake_media::{MediaCancelToken, MediaEngine};

const RESULT_BYTES_MAX: u64 = 1024 * 1024 * 1024;
const DATA_URL_ENCODED_MAX: usize = 512 * 1024 * 1024;
const RESULT_REDIRECT_MAX: usize = 5;

/// Failure code for a submission whose answer never arrived: the provider may
/// or may not have accepted (and billed) the job, so the user must check the
/// provider console before retrying. Nothing is resubmitted automatically.
const SUBMIT_OUTCOME_UNKNOWN: &str = "GENERATION_SUBMIT_OUTCOME_UNKNOWN";

/// Bound on waiting for a submission to answer. Transport deadlines normally
/// end a submission first; synchronous media providers are allowed
/// `opentake_gen::transport::MEDIA_REQUEST_TIMEOUT` for the call itself.
const SUBMIT_OUTCOME_TIMEOUT: Duration = Duration::from_secs(11 * 60);

/// Bound on holding the results of a synchronous provider for a job whose
/// project was replaced (usually a local `data:` URL decode). When it runs
/// out the downloads are cancelled and what was held is recorded.
const HOLD_RESULTS_TIMEOUT: Duration = Duration::from_secs(15 * 60);

/// Failure code for a job whose project could not record its state.
const STATE_PERSIST_FAILED: &str = "GENERATION_STATE_PERSIST_FAILED";

/// Failure code for a status poll the provider answered with a final error
/// (an unknown job, a refused request): polling again cannot help.
const PROVIDER_POLL_REFUSED: &str = "GENERATION_PROVIDER_POLL_REFUSED";

/// Failure codes of a job that ended on this side, or while its status was
/// polled, after the provider accepted it: the provider job may still hold
/// its result, so a retry polls and finalizes it again instead of paying
/// for a new one. `GENERATION_PROVIDER_POLL_FAILED` is what earlier versions
/// wrote when a poll failed or its budget ran out.
const RESUMABLE_FAILURES: &[&str] = &[
    STATE_PERSIST_FAILED,
    "GENERATION_PROVIDER_POLL_FAILED",
    "GENERATION_RATE_LIMITED",
    "GENERATION_AUTH_FAILED",
    "GENERATION_RECOVERY_AUTH_UNAVAILABLE",
    "GENERATION_DOWNLOAD_FAILED",
    "GENERATION_FINALIZE_FAILED",
    "GENERATION_FINALIZE_TASK_FAILED",
];

/// Failure code for a job a retry resumed that failed again: the next retry
/// resubmits it (after the cost confirmation) instead of resuming it again.
const RESUME_FAILED: &str = "GENERATION_RESUME_FAILED";

/// How often one session resumes polling a job whose watch was interrupted
/// (retry budget or deadline) before leaving it for the next reopen.
const INTERRUPTED_RESUMES_MAX: u32 = 5;

/// Runtime bounds of background jobs (tests shorten them).
#[derive(Clone, Copy, Debug)]
struct GenerationTimings {
    submit_timeout: Duration,
    /// Replaces the per-kind polling deadline.
    watch_deadline: Option<Duration>,
    /// Wait before polling an interrupted job again.
    resume_delay: Duration,
    /// Replaces [`HOLD_RESULTS_TIMEOUT`].
    hold_timeout: Duration,
}

impl Default for GenerationTimings {
    fn default() -> Self {
        Self {
            submit_timeout: SUBMIT_OUTCOME_TIMEOUT,
            watch_deadline: None,
            resume_delay: Duration::from_secs(2 * 60),
            hold_timeout: HOLD_RESULTS_TIMEOUT,
        }
    }
}

impl GenerationTimings {
    /// How long one session polls a job before leaving it for recovery.
    fn watch_deadline(&self, kind: Option<ModelKind>) -> Duration {
        self.watch_deadline.unwrap_or(match kind {
            Some(ModelKind::Image | ModelKind::Audio) => Duration::from_secs(20 * 60),
            Some(ModelKind::Upscale) => Duration::from_secs(90 * 60),
            Some(ModelKind::Video) | None => Duration::from_secs(60 * 60),
        })
    }
}

/// Why a background job stopped before it finalized.
#[derive(Debug, PartialEq, Eq)]
enum JobStop {
    /// Persist this fixed code on every nonterminal placeholder.
    Failed(String),
    /// The user cancelled: nonterminal placeholders become Cancelled.
    Cancelled,
    /// Polling stopped while the provider job may still finish. Nothing is
    /// written: the placeholders keep Generating and the provider job id, and
    /// `recover_current_project` resumes them (upstream "retry on reopen").
    Detached,
}

impl From<String> for JobStop {
    fn from(code: String) -> Self {
        if code == "GENERATION_CANCELLED" {
            JobStop::Cancelled
        } else {
            JobStop::Failed(code)
        }
    }
}

/// How long a job waits for a project identity transition to be resolved
/// (rebound or detached) after one of its writes failed because of it.
const TRANSITION_SETTLE_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Default)]
struct GenerationRuntime {
    /// Jobs working for the open project. Each holds an install lease.
    jobs: Mutex<HashMap<String, ActiveGenerationJob>>,
    /// Tasks detached from a project that was replaced, by job id, until they
    /// exit. A reopened project hands their jobs over only after that, so one
    /// job never has two tasks. Each keeps its install lease: a detached task
    /// may still be waiting for a paid submission's answer and must record
    /// it before an update installs. Lock after `jobs`.
    exiting: Mutex<HashMap<String, ExitingTask>>,
    /// Serializes registering a new job with recovery: `submit` commits the
    /// placeholders and registers the job under it, and recovery reads the
    /// project and claims its jobs under it, so recovery never finds a
    /// placeholder whose task is not registered yet. Lock before `jobs`.
    registration: Mutex<()>,
    /// Orphaned jobs whose durable record failed, kept for this session.
    orphaned_submissions: Mutex<HashMap<String, OrphanedGeneration>>,
    /// Interrupted watches resumed per job in this session.
    resumes: Mutex<HashMap<String, u32>>,
    /// Failed jobs a retry resumed, until they end.
    retry_resumed: Mutex<HashSet<String>>,
    next_task: std::sync::atomic::AtomicU64,
    /// Finalization leases and completions, keyed by project and job: a
    /// Save As copy of a generating job is finalized once per bundle.
    terminal_leases: Mutex<BTreeSet<String>>,
    completed: Mutex<BTreeSet<String>>,
}

struct ActiveGenerationJob {
    cancel: MediaCancelToken,
    binding: Arc<JobBinding>,
    task: u64,
    admission: crate::updater::ActivityLease,
}

impl ActiveGenerationJob {
    /// Detach the job from its project and keep its task's lease until the
    /// task exits.
    fn detach(self) -> ExitingTask {
        self.binding.detach();
        self.cancel.cancel();
        ExitingTask {
            task: self.task,
            _admission: self.admission,
        }
    }
}

struct ExitingTask {
    task: u64,
    _admission: crate::updater::ActivityLease,
}

/// The project a background job writes to: the identity it was started or
/// recovered in. Save As rebinds it to the new bundle, which carries the same
/// placeholders; replacing the project detaches it, after which the job
/// writes nothing and leaves its durable state for recovery.
struct JobBinding {
    state: Mutex<BindingState>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BindingState {
    project_epoch: u64,
    project_dir: PathBuf,
    detached: bool,
    /// Bumped by every rebind or detach, so a writer whose write failed can
    /// tell whether the transition that caused it has been resolved.
    revision: u64,
}

impl JobBinding {
    fn new(project_epoch: u64, project_dir: PathBuf) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(BindingState {
                project_epoch,
                project_dir,
                detached: false,
                revision: 0,
            }),
        })
    }

    fn state(&self) -> BindingState {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn rebind(&self, project_epoch: u64, project_dir: PathBuf) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !state.detached {
            state.project_epoch = project_epoch;
            state.project_dir = project_dir;
            state.revision += 1;
        }
    }

    fn detach(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !state.detached {
            state.detached = true;
            state.revision += 1;
        }
    }

    fn is_detached(&self) -> bool {
        self.state().detached
    }
}

/// A write for a bound job could not be made.
#[derive(Debug, PartialEq, Eq)]
enum BoundWriteError {
    /// The job no longer belongs to the open project.
    Detached,
    Failed(String),
}

enum WriteStep<T> {
    Done(Result<T, BoundWriteError>),
    /// The open project changed under the write; retry once the binding has
    /// moved past this revision.
    Wait(u64),
}

#[derive(Clone)]
pub(crate) struct TauriGenerationBridge {
    core: AppCore,
    engine: Arc<MediaEngine>,
    staging_root: PathBuf,
    runtime: Arc<GenerationRuntime>,
    clients: Arc<dyn GenerationClientFactory>,
    admission: crate::updater::InstallAdmissionGate,
    timings: GenerationTimings,
    orphans: Arc<OrphanedGenerationStore>,
}

trait GenerationClientFactory: Send + Sync {
    fn configured_byok_prefixes(&self) -> BTreeSet<String>;
    fn has_managed_credential(&self) -> bool;
    fn build(&self, provider: &str, managed: bool) -> Result<GenClient, String>;
}

struct ProductionGenerationClientFactory;

impl GenerationClientFactory for ProductionGenerationClientFactory {
    fn configured_byok_prefixes(&self) -> BTreeSet<String> {
        let store = KeyringStore::new();
        [
            ProviderKey::Fal,
            ProviderKey::Replicate,
            ProviderKey::OpenAI,
            ProviderKey::ElevenLabs,
        ]
        .into_iter()
        .filter_map(|key| {
            (&store as &dyn KeyStore)
                .load_key(key)
                .ok()
                .flatten()
                .map(|_| key.prefix().to_string())
        })
        .collect()
    }

    fn has_managed_credential(&self) -> bool {
        crate::account::generation_credential()
            .ok()
            .flatten()
            .is_some()
    }

    fn build(&self, provider: &str, managed: bool) -> Result<GenClient, String> {
        build_client(provider, managed)
    }
}

struct PreparedDispatch {
    plan: PreparedGenerationJob,
    references: Vec<PreparedReference>,
    timeline_span: Option<PreparedTimelineSpan>,
    requires_source_video: bool,
    model_kind: ModelKind,
    managed: bool,
    /// Output frame rate for a resolution-targeted upscale. The persisted input
    /// has no frame-rate field, so it travels with the dispatch only.
    upscale_target_fps: Option<u32>,
    /// Notes returned with the accepted submission.
    warnings: Vec<String>,
}

struct PreparedTimelineSpan {
    timeline: Timeline,
    manifest: opentake_domain::MediaManifest,
    project_dir: Option<PathBuf>,
    start_frame: i32,
    end_frame: i32,
}

#[derive(Clone)]
struct PreparedReference {
    path: PathBuf,
    fallback: &'static str,
    trim_range: Option<(f64, f64)>,
}

struct StagedCleanup {
    path: PathBuf,
    armed: bool,
}

impl StagedCleanup {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn preserve(mut self) {
        self.armed = false;
    }
}

impl Drop for StagedCleanup {
    fn drop(&mut self) {
        if self.armed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

impl PreparedReference {
    fn whole(path: PathBuf, fallback: &'static str) -> Self {
        Self {
            path,
            fallback,
            trim_range: None,
        }
    }
}

/// `orphans_root` must be in application data: it holds paid jobs that no
/// open project records yet.
pub(crate) fn build_bridge(
    core: AppCore,
    cache_root: PathBuf,
    models_dir: PathBuf,
    orphans_root: PathBuf,
    admission: crate::updater::InstallAdmissionGate,
) -> Arc<TauriGenerationBridge> {
    Arc::new(TauriGenerationBridge {
        core,
        engine: Arc::new(MediaEngine::new(cache_root.clone(), models_dir)),
        staging_root: cache_root.join("generation-staging"),
        runtime: Arc::new(GenerationRuntime::default()),
        clients: Arc::new(ProductionGenerationClientFactory),
        admission,
        timings: GenerationTimings::default(),
        orphans: Arc::new(OrphanedGenerationStore::new(orphans_root)),
    })
}

#[cfg(test)]
fn build_bridge_with_clients(
    core: AppCore,
    cache_root: PathBuf,
    models_dir: PathBuf,
    clients: Arc<dyn GenerationClientFactory>,
) -> Arc<TauriGenerationBridge> {
    build_bridge_with_clients_and_admission(
        core,
        cache_root,
        models_dir,
        clients,
        crate::updater::InstallAdmissionGate::default(),
    )
}

#[cfg(test)]
fn build_bridge_with_clients_and_admission(
    core: AppCore,
    cache_root: PathBuf,
    models_dir: PathBuf,
    clients: Arc<dyn GenerationClientFactory>,
    admission: crate::updater::InstallAdmissionGate,
) -> Arc<TauriGenerationBridge> {
    build_bridge_with_timings(
        core,
        cache_root,
        models_dir,
        clients,
        admission,
        GenerationTimings::default(),
    )
}

#[cfg(test)]
fn build_bridge_with_timings(
    core: AppCore,
    cache_root: PathBuf,
    models_dir: PathBuf,
    clients: Arc<dyn GenerationClientFactory>,
    admission: crate::updater::InstallAdmissionGate,
    timings: GenerationTimings,
) -> Arc<TauriGenerationBridge> {
    Arc::new(TauriGenerationBridge {
        core,
        engine: Arc::new(MediaEngine::new(cache_root.clone(), models_dir)),
        staging_root: cache_root.join("generation-staging"),
        runtime: Arc::new(GenerationRuntime::default()),
        clients,
        admission,
        timings,
        // A restarted app (a new bridge on the same cache) finds the records.
        orphans: Arc::new(OrphanedGenerationStore::new(
            cache_root.join("app-data").join("generation-orphans"),
        )),
    })
}

/// How a recovery found a job.
enum RecoveryClaim {
    /// A task already works on this job for the open project.
    Running,
    /// A task bound to a replaced project has not exited yet.
    Handover,
    /// The update installer holds admission.
    Refused,
    /// Registered for a new task.
    Claimed {
        cancel: MediaCancelToken,
        binding: Arc<JobBinding>,
        task: u64,
    },
}

impl TauriGenerationBridge {
    /// Follow project identity transitions (see `on_project_identity_transition`).
    /// The listener holds the bridge weakly: the bridge holds the core.
    pub(crate) fn follow_project_identity(self: &Arc<Self>) {
        let bridge = Arc::downgrade(self);
        self.core
            .subscribe_project_identity_transition(move |pending| {
                if let Some(bridge) = bridge.upgrade() {
                    bridge.on_project_identity_transition(pending);
                }
            });
    }

    /// Runs synchronously inside the core's transition announcement, so it
    /// only updates bookkeeping and signals tasks; it never waits for them.
    ///
    /// Until the old identity is replaced (`pending`), jobs keep writing to
    /// it. Afterwards a job whose project was saved under a new path is
    /// rebound to the new bundle, which carries the same placeholders, and
    /// keeps running. A job whose project was replaced is detached: it is not
    /// cancelled at the provider and writes nothing more to the project; its
    /// placeholders keep Generating and the provider job id on disk. A task
    /// still waiting for a submission's answer records an accepted job (and
    /// the results of a synchronous provider) in application data before it
    /// exits, and keeps its install lease until then. Recovery then resumes
    /// whatever the open project owns.
    fn on_project_identity_transition(&self, pending: bool) {
        if pending {
            return;
        }
        let (project_epoch, project_dir) = self.session_identity();
        {
            let mut jobs = self
                .runtime
                .jobs
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let mut replaced = Vec::new();
            for (job_id, job) in jobs.iter() {
                let state = job.binding.state();
                if state.project_epoch == project_epoch
                    && project_dir.as_deref() == Some(state.project_dir.as_path())
                {
                    continue;
                }
                match &project_dir {
                    Some(dir) if state.project_epoch == project_epoch => {
                        job.binding.rebind(project_epoch, dir.clone());
                    }
                    _ => replaced.push(job_id.clone()),
                }
            }
            let mut exiting = self
                .runtime
                .exiting
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for job_id in replaced {
                if let Some(job) = jobs.remove(&job_id) {
                    exiting.insert(job_id, job.detach());
                }
            }
        }
        let bridge = self.clone();
        tauri::async_runtime::spawn_blocking(move || {
            bridge.recover_current_project();
        });
    }

    /// Register a task for `job_id` in the open project unless one exists.
    fn claim_job(&self, job_id: &str, project_epoch: u64, project_dir: &Path) -> RecoveryClaim {
        let mut jobs = self
            .runtime
            .jobs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(job) = jobs.get(job_id) {
            let state = job.binding.state();
            if !state.detached
                && state.project_epoch == project_epoch
                && state.project_dir == project_dir
            {
                return RecoveryClaim::Running;
            }
            // A task still bound elsewhere must exit before this project
            // takes the job over.
            if let Some(job) = jobs.remove(job_id) {
                self.runtime
                    .exiting
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .insert(job_id.to_string(), job.detach());
            }
            return RecoveryClaim::Handover;
        }
        if self
            .runtime
            .exiting
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains_key(job_id)
        {
            return RecoveryClaim::Handover;
        }
        let Ok(admission) = self.admission.begin_activity() else {
            return RecoveryClaim::Refused;
        };
        let cancel = MediaCancelToken::new();
        let binding = JobBinding::new(project_epoch, project_dir.to_path_buf());
        let task = self
            .runtime
            .next_task
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        jobs.insert(
            job_id.to_string(),
            ActiveGenerationJob {
                cancel: cancel.clone(),
                binding: binding.clone(),
                task,
                admission,
            },
        );
        RecoveryClaim::Claimed {
            cancel,
            binding,
            task,
        }
    }

    /// Forget a task that ended, unless its job has moved on to another task.
    fn finish_task(&self, job_id: &str, task: u64) {
        let mut jobs = self
            .runtime
            .jobs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if jobs.get(job_id).is_some_and(|job| job.task == task) {
            jobs.remove(job_id);
        }
        let mut exiting = self
            .runtime
            .exiting
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if exiting
            .get(job_id)
            .is_some_and(|exiting| exiting.task == task)
        {
            exiting.remove(job_id);
        }
    }

    /// Resume `job_id` for the open project once the task of the replaced
    /// project has exited (it may still be waiting for a submission answer).
    fn schedule_handover(&self, job_id: String) {
        let bridge = self.clone();
        let wait =
            self.timings.submit_timeout + self.timings.hold_timeout + TRANSITION_SETTLE_TIMEOUT * 2;
        tauri::async_runtime::spawn(async move {
            let deadline = tokio::time::Instant::now() + wait;
            while bridge
                .runtime
                .exiting
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .contains_key(&job_id)
            {
                if tokio::time::Instant::now() >= deadline {
                    eprintln!(
                        "[generation] job {job_id}: handover gave up waiting for the previous \
                         task; reopen the project to resume it"
                    );
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            let recovery = bridge.clone();
            if let Err(error) = tauri::async_runtime::spawn_blocking(move || {
                recovery.recover_jobs(Some(&job_id));
            })
            .await
            {
                eprintln!("[generation] handover recovery failed: {error}");
            }
        });
    }

    /// The open project's epoch and bundle, read under one session lock.
    fn session_identity(&self) -> (u64, Option<PathBuf>) {
        match self.core.project_asset_authority() {
            Some(authority) => (authority.project_epoch, Some(authority.project_path)),
            None => (self.core.project_revision().project_epoch, None),
        }
    }

    fn session_is(&self, state: &BindingState) -> bool {
        let (project_epoch, project_dir) = self.session_identity();
        !state.detached
            && project_epoch == state.project_epoch
            && project_dir.as_deref() == Some(state.project_dir.as_path())
    }

    fn try_bound_write<T>(
        &self,
        binding: &JobBinding,
        write: &mut dyn FnMut(u64, &Path) -> opentake_core::Result<T>,
    ) -> WriteStep<T> {
        let state = binding.state();
        if state.detached {
            return WriteStep::Done(Err(BoundWriteError::Detached));
        }
        match write(state.project_epoch, &state.project_dir) {
            Ok(value) => WriteStep::Done(Ok(value)),
            Err(error) if self.session_is(&state) => {
                WriteStep::Done(Err(BoundWriteError::Failed(error.to_string())))
            }
            // The project changed under the write: wait for the transition
            // to rebind (Save As) or detach (replacement) this job.
            Err(_) => WriteStep::Wait(state.revision),
        }
    }

    /// Write for a bound job, following a Save As to the new bundle. The
    /// write persists (and fsyncs) project state, so it runs on a blocking
    /// thread rather than on an async worker.
    async fn bound_write<T, W>(
        &self,
        binding: &Arc<JobBinding>,
        write: W,
    ) -> Result<T, BoundWriteError>
    where
        T: Send + 'static,
        W: FnMut(&AppCore, u64, &Path) -> opentake_core::Result<T> + Send + 'static,
    {
        let deadline = tokio::time::Instant::now() + TRANSITION_SETTLE_TIMEOUT;
        let mut write = write;
        loop {
            let bridge = self.clone();
            let worker_binding = Arc::clone(binding);
            let (returned, step) = tokio::task::spawn_blocking(move || {
                let core = bridge.core.clone();
                let step = bridge
                    .try_bound_write(&worker_binding, &mut |epoch, dir| write(&core, epoch, dir));
                (write, step)
            })
            .await
            .map_err(|error| {
                BoundWriteError::Failed(format!("generation state worker failed: {error}"))
            })?;
            write = returned;
            match step {
                WriteStep::Done(result) => return result,
                WriteStep::Wait(revision) => {
                    while binding.state().revision == revision {
                        if tokio::time::Instant::now() >= deadline {
                            return Err(BoundWriteError::Detached);
                        }
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                }
            }
        }
    }

    /// [`Self::bound_write`] for blocking code (finalization).
    fn bound_write_blocking<T>(
        &self,
        binding: &JobBinding,
        mut write: impl FnMut(u64, &Path) -> opentake_core::Result<T>,
    ) -> Result<T, BoundWriteError> {
        let deadline = std::time::Instant::now() + TRANSITION_SETTLE_TIMEOUT;
        loop {
            match self.try_bound_write(binding, &mut write) {
                WriteStep::Done(result) => return result,
                WriteStep::Wait(revision) => {
                    while binding.state().revision == revision {
                        if std::time::Instant::now() >= deadline {
                            return Err(BoundWriteError::Detached);
                        }
                        std::thread::sleep(Duration::from_millis(10));
                    }
                }
            }
        }
    }

    /// A provider accepted a job that its project could not record: the
    /// project was replaced first, or writing the provider job id failed.
    /// Record the provider job id, and the results a synchronous provider
    /// already returned, in application data, so the job is resumed (in this
    /// session, or on reopen even after a restart) instead of requiring a
    /// paid retry. Runs before the task exits, under its install lease: when
    /// holding the results takes too long, their downloads are cancelled and
    /// what was held is still recorded before this returns.
    async fn keep_accepted_job(
        &self,
        binding: &Arc<JobBinding>,
        job_id: &str,
        provider_job_id: String,
        terminal: Option<GenerationJob>,
    ) {
        let bridge = self.clone();
        let project_path = binding.state().project_dir.display().to_string();
        let worker_job_id = job_id.to_string();
        let cancel = MediaCancelToken::new();
        let worker_cancel = cancel.clone();
        let mut worker = tokio::task::spawn_blocking(move || {
            bridge.keep_accepted_job_blocking(
                &worker_job_id,
                provider_job_id,
                project_path,
                terminal,
                &worker_cancel,
            )
        });
        let kept = tokio::select! {
            kept = &mut worker => kept,
            () = tokio::time::sleep(self.timings.hold_timeout) => {
                eprintln!(
                    "[generation] job {job_id}: holding its results timed out; keeping what \
                     was held"
                );
                cancel.cancel();
                worker.await
            }
        };
        match kept {
            Ok(Ok(())) => {}
            Err(error) => eprintln!(
                "[generation] job {job_id}: GENERATION_SUBMIT_ORPHANED: keeping its accepted job \
                 failed: {error}"
            ),
            Ok(Err(error)) => eprintln!(
                "[generation] job {job_id}: GENERATION_SUBMIT_ORPHANED: its accepted job could \
                 not be kept: {error}"
            ),
        }
    }

    fn keep_accepted_job_blocking(
        &self,
        job_id: &str,
        provider_job_id: String,
        project_path: String,
        terminal: Option<GenerationJob>,
        cancel: &MediaCancelToken,
    ) -> Result<(), String> {
        let mut record = OrphanedGeneration {
            job_id: job_id.to_string(),
            provider_job_id: Some(provider_job_id),
            project_path,
            recorded_at: crate::voice_revocations::unix_now_seconds(),
            held_results: Vec::new(),
        };
        let mut hold_error = None;
        if let Some(job) = terminal {
            // The provider cannot be asked for these results again. Only the
            // hold timeout cancels this: the user's cancellation was of the
            // project, not of the paid result.
            let (held, error) =
                self.hold_results(job.result_urls.as_deref().unwrap_or_default(), cancel);
            record.held_results = held;
            hold_error = error;
        }
        // Replaces the submission's unknown-outcome record.
        let recorded = self.orphans.record(record.clone());
        if recorded.is_err() {
            // At least this session can still resume it.
            self.runtime
                .orphaned_submissions
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .insert(job_id.to_string(), record);
        }
        eprintln!(
            "[generation] job {job_id}: accepted, but its project did not record it; it is \
             resumed from application data"
        );
        recorded?;
        match hold_error {
            Some(error) => Err(format!("its results could not all be held: {error}")),
            None => Ok(()),
        }
    }

    /// Download (or decode) a terminal job's results into the orphan store,
    /// in placeholder order. Holding stops at the first failure and returns
    /// the results held before it, which the caller records.
    fn hold_results(
        &self,
        urls: &[String],
        cancel: &MediaCancelToken,
    ) -> (Vec<crate::generation_orphans::HeldResult>, Option<String>) {
        let mut held = Vec::with_capacity(urls.len());
        let downloader =
            match SecureResultDownloader::new(self.staging_root.clone(), cancel.clone()) {
                Ok(downloader) => downloader,
                Err(error) => return (held, Some(error)),
            };
        for (index, url) in urls.iter().enumerate() {
            let artifact = match downloader.download(&format!("held-{index}"), url) {
                Ok(artifact) => artifact,
                Err(error) => return (held, Some(error)),
            };
            match self.orphans.hold(&artifact.path, &artifact.media_type) {
                Ok(kept) => held.push(kept),
                Err(error) => {
                    let _ = std::fs::remove_file(&artifact.path);
                    return (held, Some(error));
                }
            }
        }
        (held, None)
    }

    /// Record, before a submission is sent, that its outcome is unknown
    /// until its project records the answer. Nothing has been paid yet, so
    /// a job whose record cannot be written is not sent.
    async fn record_pending_submission(
        &self,
        binding: &Arc<JobBinding>,
        job_id: &str,
    ) -> Result<(), JobStop> {
        let orphans = Arc::clone(&self.orphans);
        let record = OrphanedGeneration {
            job_id: job_id.to_string(),
            provider_job_id: None,
            project_path: binding.state().project_dir.display().to_string(),
            recorded_at: crate::voice_revocations::unix_now_seconds(),
            held_results: Vec::new(),
        };
        let recorded = tokio::task::spawn_blocking(move || orphans.record(record))
            .await
            .map_err(|error| format!("generation state worker failed: {error}"))
            .and_then(|recorded| recorded);
        recorded.map_err(|error| {
            eprintln!(
                "[generation] job {job_id}: not submitted: its pending submission could not be \
                 recorded: {error}"
            );
            JobStop::Failed(STATE_PERSIST_FAILED.to_string())
        })
    }

    /// The orphan record of `job_id`, kept in memory or durable. A record
    /// is kept in memory only after its durable write failed, so it is newer
    /// than the durable one, which may still be the pending submission.
    fn orphan_record(&self, job_id: &str) -> Option<OrphanedGeneration> {
        let kept = self
            .runtime
            .orphaned_submissions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(job_id)
            .cloned();
        if kept.is_some() {
            return kept;
        }
        match self.orphans.get(job_id) {
            Ok(record) => record,
            Err(error) => {
                eprintln!("[generation] job {job_id}: orphaned jobs could not be read: {error}");
                None
            }
        }
    }

    /// Forget an orphan record once the project records the job, or its
    /// held results can no longer be finalized.
    fn forget_orphan(&self, job_id: &str) {
        self.runtime
            .orphaned_submissions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(job_id);
        if let Err(error) = self.orphans.remove(job_id) {
            eprintln!("[generation] job {job_id}: its orphan record was not removed: {error}");
        }
    }

    /// [`Self::forget_orphan`] from async code: it rewrites the store with
    /// fsync and deletes held files, so it runs on a blocking thread.
    async fn forget_orphan_off_thread(&self, job_id: &str) {
        let bridge = self.clone();
        let worker_job_id = job_id.to_string();
        if let Err(error) =
            tokio::task::spawn_blocking(move || bridge.forget_orphan(&worker_job_id)).await
        {
            eprintln!("[generation] job {job_id}: its orphan record was not removed: {error}");
        }
    }

    /// Whether any task still works, including a task detached from a
    /// replaced project that has not exited yet.
    pub(crate) fn has_active(&self) -> bool {
        let jobs = self
            .runtime
            .jobs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        !jobs.is_empty()
            || !self
                .runtime
                .exiting
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .is_empty()
    }

    /// Detach every task working for the open project before an update
    /// installs, as replacing the project does: nothing is cancelled at the
    /// provider, an accepted job (and a synchronous provider's results) is
    /// kept in application data, and each task keeps its install lease until
    /// it exits. Returns how many tasks still have to exit, detached ones
    /// included. The jobs resume when the project is next opened; if the
    /// update has not installed after `resume_delay`, the open project
    /// resumes them.
    pub(crate) fn detach_all_active(&self) -> usize {
        let detached = {
            let mut jobs = self
                .runtime
                .jobs
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let mut exiting = self
                .runtime
                .exiting
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let detached = jobs.len();
            for (job_id, job) in jobs.drain() {
                exiting.insert(job_id, job.detach());
            }
            detached
        };
        if detached > 0 {
            let bridge = self.clone();
            tauri::async_runtime::spawn(async move {
                tokio::time::sleep(bridge.timings.resume_delay).await;
                let recovery = bridge.clone();
                if let Err(error) =
                    tauri::async_runtime::spawn_blocking(move || recovery.recover_current_project())
                        .await
                {
                    eprintln!("[generation] resuming jobs detached for an update failed: {error}");
                }
            });
        }
        self.runtime
            .exiting
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    /// Cancel a running job, or one whose polling was interrupted and left
    /// for recovery (no task works on it, so its placeholders in the open
    /// project are cancelled directly). Nothing is cancelled at the provider.
    pub(crate) fn cancel(&self, job_id: &str) -> Result<bool, String> {
        let running = self
            .runtime
            .jobs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(job_id)
            .is_some_and(|job| {
                job.cancel.cancel();
                true
            });
        if running {
            return Ok(true);
        }
        self.cancel_idle_job(job_id)
    }

    /// Persists project state, so it holds an install lease like any job.
    fn cancel_idle_job(&self, job_id: &str) -> Result<bool, String> {
        let _admission = self.admission.begin_activity()?;
        let snapshot = self.core.runtime_snapshot();
        let Some(project_dir) = snapshot.project_dir.as_deref() else {
            return Ok(false);
        };
        let placeholder_ids = snapshot
            .media
            .entries
            .iter()
            .filter(|entry| {
                entry.generation_input.as_ref().is_some_and(|input| {
                    input.job_id.as_deref() == Some(job_id)
                        && matches!(
                            input.status,
                            Some(
                                GenerationJobStatus::Queued
                                    | GenerationJobStatus::Generating
                                    | GenerationJobStatus::Downloading
                                    | GenerationJobStatus::Finalizing
                            )
                        )
                })
            })
            .map(|entry| entry.id.clone())
            .collect::<Vec<_>>();
        if placeholder_ids.is_empty() {
            return Ok(false);
        }
        self.cancel_nonterminal_outputs(snapshot.project_epoch, project_dir, &placeholder_ids);
        // A result held for this job will never be finalized now.
        if self.orphan_record(job_id).is_some() {
            self.forget_orphan(job_id);
        }
        Ok(true)
    }

    /// Whether retrying `job_id` resumes its accepted provider job, which
    /// costs nothing, instead of submitting a new one.
    pub(crate) fn retry_resumes(&self, job_id: &str) -> bool {
        resumable_job(&self.core.runtime_snapshot().media, job_id).is_some()
    }

    /// Retry a failed or cancelled job. A job the provider accepted that
    /// failed on this side is polled and finalized again (see
    /// [`RESUMABLE_FAILURES`]); any other job is submitted again, which needs
    /// a fresh cost authorization.
    pub(crate) fn retry(
        &self,
        job_id: &str,
        cost_authorized: bool,
    ) -> Result<GenerationSubmission, String> {
        let snapshot = self.core.runtime_snapshot();
        if let Some(placeholder_asset_ids) = resumable_job(&snapshot.media, job_id) {
            return self.resume_failed_job(&snapshot, job_id, placeholder_asset_ids);
        }
        if !cost_authorized {
            return Err("cost authorization is required before retry".to_string());
        }
        let outputs = snapshot
            .media
            .entries
            .iter()
            .filter(|entry| {
                entry
                    .generation_input
                    .as_ref()
                    .and_then(|input| input.job_id.as_deref())
                    == Some(job_id)
            })
            .collect::<Vec<_>>();
        let first = outputs
            .first()
            .ok_or_else(|| "generation job does not exist".to_string())?;
        if outputs.iter().any(|entry| {
            !matches!(
                entry
                    .generation_input
                    .as_ref()
                    .and_then(|input| input.status),
                Some(GenerationJobStatus::Failed | GenerationJobStatus::Cancelled)
            )
        }) {
            return Err("only a failed or cancelled generation can be retried".to_string());
        }
        let input = first
            .generation_input
            .as_ref()
            .ok_or_else(|| "generation provenance is missing".to_string())?;
        let catalog = Catalog::builtin();
        let model = catalog
            .entries()
            .iter()
            .find(|entry| entry.id == input.model)
            .ok_or_else(|| "generation model is no longer available".to_string())?;
        let request = match model.kind {
            ModelKind::Video => {
                let frames = input.image_url_asset_ids.clone().unwrap_or_default();
                GenerationRequest::Video(GenerateVideoArgs {
                    cost_authorized: Some(true),
                    prompt: input.prompt.clone(),
                    name: Some(first.name.clone()),
                    model: Some(input.model.clone()),
                    duration: Some(input.duration),
                    aspect_ratio: Some(input.aspect_ratio.clone()),
                    resolution: input.resolution.clone(),
                    start_frame_media_ref: frames.first().cloned(),
                    end_frame_media_ref: frames.get(1).cloned(),
                    source_video_media_ref: input.source_asset_id.clone(),
                    source_clip_id: input.source_clip_id.clone(),
                    reference_image_media_refs: input.reference_image_asset_ids.clone(),
                    reference_video_media_refs: input.reference_video_asset_ids.clone(),
                    reference_audio_media_refs: input.reference_audio_asset_ids.clone(),
                    folder_id: first.folder_id.clone(),
                })
            }
            ModelKind::Image => GenerationRequest::Image(GenerateImageArgs {
                cost_authorized: Some(true),
                prompt: input.prompt.clone(),
                name: Some(first.name.clone()),
                model: Some(input.model.clone()),
                aspect_ratio: Some(input.aspect_ratio.clone()),
                resolution: input.resolution.clone(),
                quality: input.quality.clone(),
                num_images: Some(outputs.len() as i32),
                reference_media_refs: input.reference_image_asset_ids.clone(),
                folder_id: first.folder_id.clone(),
            }),
            ModelKind::Audio => GenerationRequest::Audio(GenerateAudioArgs {
                cost_authorized: Some(true),
                prompt: Some(input.prompt.clone()),
                name: Some(first.name.clone()),
                model: Some(input.model.clone()),
                voice: input.voice.clone(),
                lyrics: input.lyrics.clone(),
                style_instructions: input.style_instructions.clone(),
                instrumental: input.instrumental,
                duration: Some(input.duration),
                video_source_start_frame: input.source_start_frame,
                video_source_end_frame: input.source_end_frame,
                video_source_media_ref: input.source_asset_id.clone(),
                folder_id: first.folder_id.clone(),
            }),
            ModelKind::Upscale => GenerationRequest::Upscale(UpscaleMediaArgs {
                cost_authorized: Some(true),
                media_ref: input
                    .source_asset_id
                    .clone()
                    .ok_or_else(|| "upscale source provenance is missing".to_string())?,
                model: Some(input.model.clone()),
                source_clip_id: input.source_clip_id.clone(),
            }),
        };
        self.submit(request, &MediaCancelToken::new())
    }

    fn resume_failed_job(
        &self,
        snapshot: &opentake_core::ProjectRuntimeSnapshot,
        job_id: &str,
        placeholder_asset_ids: Vec<String>,
    ) -> Result<GenerationSubmission, String> {
        let project_dir = snapshot
            .project_dir
            .as_deref()
            .ok_or_else(|| "no project is open".to_string())?;
        self.core
            .resume_generation_job_for_project(
                snapshot.project_epoch,
                project_dir,
                job_id,
                Some(now_apple_reference_seconds()),
            )
            .map_err(|error| error.to_string())?;
        self.runtime
            .resumes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(job_id);
        self.runtime
            .retry_resumed
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(job_id.to_string());
        self.recover_jobs(Some(job_id));
        Ok(GenerationSubmission {
            job_id: job_id.to_string(),
            placeholder_asset_ids,
            status: "generating".to_string(),
            warnings: Vec::new(),
        })
    }

    /// Resume provider polling for durable non-terminal jobs after a project is
    /// opened. A queued record without a provider id is deliberately failed and
    /// exposed for explicit retry: resubmitting it automatically could create a
    /// second paid job if the process died between provider acceptance and the
    /// durable id write. A job whose task is still bound to a replaced project
    /// is handed over once that task has exited.
    pub(crate) fn recover_current_project(&self) -> usize {
        self.recover_jobs(None)
    }

    fn recover_jobs(&self, only: Option<&str>) -> usize {
        #[derive(Default)]
        struct RecoveryJob {
            provider: String,
            model: String,
            provider_job_id: Option<String>,
            placeholders: Vec<(usize, String)>,
            has_active_output: bool,
        }

        // Read the project and claim its jobs while no submission is between
        // committing its placeholders and registering its task.
        let registration = self
            .runtime
            .registration
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let snapshot = self.core.runtime_snapshot();
        let Some(project_dir) = snapshot.project_dir.clone() else {
            return 0;
        };
        let mut recoverable = HashMap::<String, RecoveryJob>::new();
        for entry in &snapshot.media.entries {
            let Some(input) = entry.generation_input.as_ref() else {
                continue;
            };
            let Some(job_id) = input.job_id.as_ref() else {
                continue;
            };
            if only.is_some_and(|only| only != job_id) {
                continue;
            }
            let job = recoverable.entry(job_id.clone()).or_default();
            if job.provider.is_empty() {
                job.provider = input.provider.clone().unwrap_or_default();
            }
            if job.model.is_empty() {
                job.model = input.model.clone();
            }
            if job.provider_job_id.is_none() {
                job.provider_job_id = input.provider_job_id.clone();
            }
            job.placeholders
                .push((input.output_index.unwrap_or(usize::MAX), entry.id.clone()));
            if matches!(
                input.status,
                Some(
                    GenerationJobStatus::Queued
                        | GenerationJobStatus::Generating
                        | GenerationJobStatus::Downloading
                        | GenerationJobStatus::Finalizing
                )
            ) {
                job.has_active_output = true;
            }
        }
        let mut claimed = Vec::new();
        let mut resumed = 0;
        for (job_id, job) in recoverable {
            if !job.has_active_output {
                continue;
            }
            match self.claim_job(&job_id, snapshot.project_epoch, &project_dir) {
                RecoveryClaim::Running | RecoveryClaim::Refused => {}
                RecoveryClaim::Handover => {
                    self.schedule_handover(job_id);
                    resumed += 1;
                }
                RecoveryClaim::Claimed {
                    cancel,
                    binding,
                    task,
                } => claimed.push((job_id, job, cancel, binding, task)),
            }
        }
        drop(registration);

        for (job_id, mut job, cancel, binding, task) in claimed {
            job.placeholders.sort_by_key(|(index, _)| *index);
            let placeholder_ids = job
                .placeholders
                .into_iter()
                .map(|(_, asset_id)| asset_id)
                .collect::<Vec<_>>();
            let orphan = self.orphan_record(&job_id);
            let accepted_id = orphan
                .as_ref()
                .and_then(|orphan| orphan.provider_job_id.clone());
            let provider_job_id = match (job.provider_job_id, orphan.as_ref(), accepted_id) {
                (Some(provider_job_id), _, _) => provider_job_id,
                // Sent, but its answer was never recorded (the app quit or
                // the project closed first, or no answer arrived): the
                // provider may have accepted and billed it.
                (None, Some(_), None) => {
                    match self.core.update_generation_job_for_project(
                        snapshot.project_epoch,
                        &project_dir,
                        &job_id,
                        GenerationStateUpdate {
                            status: GenerationJobStatus::Failed,
                            progress: None,
                            error_code: Some(SUBMIT_OUTCOME_UNKNOWN.to_string()),
                            provider_job_id: None,
                            cost_credits: None,
                            created_at: Some(now_apple_reference_seconds()),
                        },
                    ) {
                        Ok(_) => self.forget_orphan(&job_id),
                        Err(error) => eprintln!(
                            "[generation] job {job_id}: unknown submission outcome was not \
                             persisted: {error}"
                        ),
                    }
                    self.finish_task(&job_id, task);
                    continue;
                }
                // Accepted after the project was closed: record it before
                // polling. The orphan record stays until this succeeded.
                (None, Some(_), Some(provider_job_id)) => {
                    if let Err(error) = self.core.update_generation_job_for_project(
                        snapshot.project_epoch,
                        &project_dir,
                        &job_id,
                        GenerationStateUpdate {
                            status: GenerationJobStatus::Generating,
                            progress: Some(0.15),
                            error_code: None,
                            provider_job_id: Some(provider_job_id.clone()),
                            cost_credits: None,
                            created_at: Some(now_apple_reference_seconds()),
                        },
                    ) {
                        eprintln!(
                            "[generation] job {job_id}: accepted provider job id was not \
                             persisted: {error}"
                        );
                        self.finish_task(&job_id, task);
                        continue;
                    }
                    provider_job_id
                }
                // No submission was recorded as sent (or the provider
                // refused it), so a retry does not pay twice.
                (None, None, _) => {
                    if let Err(error) = self.core.update_generation_job_for_project(
                        snapshot.project_epoch,
                        &project_dir,
                        &job_id,
                        GenerationStateUpdate {
                            status: GenerationJobStatus::Failed,
                            progress: None,
                            error_code: Some("GENERATION_RESTART_RETRY_REQUIRED".to_string()),
                            provider_job_id: None,
                            cost_credits: None,
                            created_at: Some(now_apple_reference_seconds()),
                        },
                    ) {
                        eprintln!(
                            "[generation] job {job_id}: recovery failure was not persisted: {error}"
                        );
                    }
                    self.finish_task(&job_id, task);
                    continue;
                }
            };
            let held = orphan
                .filter(|orphan| !orphan.held_results.is_empty())
                .map(|orphan| orphan.held_results);
            if held.is_none() && self.orphan_record(&job_id).is_some() {
                // The project records the provider job id now.
                self.forget_orphan(&job_id);
            }
            if job.provider.is_empty() {
                self.fail_nonterminal_outputs(
                    snapshot.project_epoch,
                    &project_dir,
                    &placeholder_ids,
                    "GENERATION_RECOVERY_STATE_INVALID",
                );
                self.finish_task(&job_id, task);
                continue;
            }
            let bridge = self.clone();
            if let Some(held) = held {
                // A synchronous provider's result, held because no poll can
                // fetch it again.
                tauri::async_runtime::spawn(async move {
                    bridge
                        .run_held_job(
                            binding,
                            task,
                            job_id,
                            placeholder_ids,
                            provider_job_id,
                            held,
                            cancel,
                        )
                        .await;
                });
                resumed += 1;
                continue;
            }
            let managed = !provider_job_id.starts_with(&format!("{}::", job.provider));
            let deadline = self
                .timings
                .watch_deadline(Catalog::builtin().by_id(&job.model).map(|model| model.kind));
            tauri::async_runtime::spawn(async move {
                bridge
                    .run_recovered_job(
                        binding,
                        task,
                        job_id,
                        placeholder_ids,
                        job.provider,
                        managed,
                        provider_job_id,
                        deadline,
                        cancel,
                    )
                    .await;
            });
            resumed += 1;
        }
        resumed
    }

    /// Whether the open project records every placeholder as finished. The
    /// core rolls back a failed write, so this is what the project holds.
    fn outputs_terminal(&self, placeholder_ids: &[String]) -> bool {
        let media = self.core.media();
        placeholder_ids.iter().all(|asset_id| {
            media
                .entries
                .iter()
                .find(|entry| entry.id == *asset_id)
                .and_then(|entry| entry.generation_input.as_ref())
                .and_then(|input| input.status)
                .is_some_and(|status| {
                    matches!(
                        status,
                        GenerationJobStatus::Ready
                            | GenerationJobStatus::Failed
                            | GenerationJobStatus::Cancelled
                    )
                })
        })
    }

    fn fail_nonterminal_outputs(
        &self,
        project_epoch: u64,
        project_dir: &Path,
        placeholder_ids: &[String],
        code: &str,
    ) {
        let snapshot = self.core.runtime_snapshot();
        for asset_id in placeholder_ids {
            let terminal = snapshot
                .media
                .entries
                .iter()
                .find(|entry| entry.id == *asset_id)
                .and_then(|entry| entry.generation_input.as_ref())
                .and_then(|input| input.status)
                .is_some_and(|status| {
                    matches!(
                        status,
                        GenerationJobStatus::Ready
                            | GenerationJobStatus::Failed
                            | GenerationJobStatus::Cancelled
                    )
                });
            if !terminal {
                if let Err(error) = self.core.fail_generation_output_for_project(
                    project_epoch,
                    project_dir,
                    asset_id,
                    code,
                    Some(now_apple_reference_seconds()),
                ) {
                    eprintln!(
                        "[generation] output {asset_id}: failure {code} was not persisted: {error}"
                    );
                }
            }
        }
    }

    fn cancel_nonterminal_outputs(
        &self,
        project_epoch: u64,
        project_dir: &Path,
        placeholder_ids: &[String],
    ) {
        for asset_id in placeholder_ids {
            if let Err(error) = self.core.cancel_generation_output_for_project(
                project_epoch,
                project_dir,
                asset_id,
                Some(now_apple_reference_seconds()),
            ) {
                eprintln!(
                    "[generation] output {asset_id}: cancellation was not persisted: {error}"
                );
            }
        }
    }

    /// Fail a bound job's nonterminal placeholders wherever its project now
    /// lives. A detached job writes nothing.
    async fn fail_bound_outputs(
        &self,
        binding: &Arc<JobBinding>,
        placeholder_ids: &[String],
        code: &str,
    ) {
        for asset_id in placeholder_ids {
            let terminal = self
                .core
                .media()
                .entries
                .iter()
                .find(|entry| entry.id == *asset_id)
                .and_then(|entry| entry.generation_input.as_ref())
                .and_then(|input| input.status)
                .is_some_and(|status| {
                    matches!(
                        status,
                        GenerationJobStatus::Ready
                            | GenerationJobStatus::Failed
                            | GenerationJobStatus::Cancelled
                    )
                });
            if terminal {
                continue;
            }
            let (write_asset_id, write_code) = (asset_id.clone(), code.to_string());
            match self
                .bound_write(binding, move |core, project_epoch, project_dir| {
                    core.fail_generation_output_for_project(
                        project_epoch,
                        project_dir,
                        &write_asset_id,
                        &write_code,
                        Some(now_apple_reference_seconds()),
                    )
                })
                .await
            {
                Ok(()) => {}
                Err(BoundWriteError::Detached) => return,
                Err(BoundWriteError::Failed(error)) => eprintln!(
                    "[generation] output {asset_id}: failure {code} was not persisted: {error}"
                ),
            }
        }
    }

    async fn cancel_bound_outputs(&self, binding: &Arc<JobBinding>, placeholder_ids: &[String]) {
        for asset_id in placeholder_ids {
            let write_asset_id = asset_id.clone();
            match self
                .bound_write(binding, move |core, project_epoch, project_dir| {
                    core.cancel_generation_output_for_project(
                        project_epoch,
                        project_dir,
                        &write_asset_id,
                        Some(now_apple_reference_seconds()),
                    )
                })
                .await
            {
                Ok(()) => {}
                Err(BoundWriteError::Detached) => return,
                Err(BoundWriteError::Failed(error)) => eprintln!(
                    "[generation] output {asset_id}: cancellation was not persisted: {error}"
                ),
            }
        }
    }

    fn configured_byok_prefixes(&self) -> BTreeSet<String> {
        self.clients.configured_byok_prefixes()
    }

    fn has_managed_credential(&self) -> bool {
        self.clients.has_managed_credential()
    }

    fn prepare(&self, request: GenerationRequest) -> Result<PreparedDispatch, String> {
        let snapshot = self.core.runtime_snapshot();
        snapshot
            .project_dir
            .as_deref()
            .ok_or_else(|| "Save the project before starting generation".to_string())?;
        let configured = self.configured_byok_prefixes();
        let managed_available = self.has_managed_credential();
        let catalog = Catalog::builtin();

        match request {
            GenerationRequest::Video(args) => {
                let entry = select_model(
                    &catalog,
                    ModelKind::Video,
                    args.model.as_deref(),
                    &configured,
                    managed_available,
                )?;
                let UiCapabilities::Video(caps) = &entry.ui_capabilities else {
                    return Err("selected model has invalid video capabilities".to_string());
                };
                let duration = args
                    .duration
                    .map(|value| value.max(0) as u32)
                    .or_else(|| caps.durations.first().copied())
                    .unwrap_or(0);
                if !caps.durations.is_empty() && !caps.durations.contains(&duration) {
                    return Err("duration is not supported by the selected model".to_string());
                }
                let aspect_ratio = args
                    .aspect_ratio
                    .clone()
                    .or_else(|| caps.aspect_ratios.first().cloned())
                    .unwrap_or_default();
                if !caps.aspect_ratios.is_empty() && !caps.aspect_ratios.contains(&aspect_ratio) {
                    return Err("aspectRatio is not supported by the selected model".to_string());
                }
                validate_choice(
                    "resolution",
                    args.resolution.as_deref(),
                    caps.resolutions.as_deref(),
                )?;

                let frames = [
                    args.start_frame_media_ref.clone(),
                    args.end_frame_media_ref.clone(),
                ]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
                if args.start_frame_media_ref.is_some() && !caps.supports_first_frame {
                    return Err("selected model does not support a first frame".to_string());
                }
                if args.end_frame_media_ref.is_some() && !caps.supports_last_frame {
                    return Err("selected model does not support a last frame".to_string());
                }
                let image_refs = args.reference_image_media_refs.clone().unwrap_or_default();
                let video_refs = args.reference_video_media_refs.clone().unwrap_or_default();
                let audio_refs = args.reference_audio_media_refs.clone().unwrap_or_default();
                validate_reference_count("image", image_refs.len(), caps.max_reference_images)?;
                validate_reference_count("video", video_refs.len(), caps.max_reference_videos)?;
                validate_reference_count("audio", audio_refs.len(), caps.max_reference_audios)?;
                if let Some(max) = caps.max_total_references {
                    if image_refs.len() + video_refs.len() + audio_refs.len() > max as usize {
                        return Err(
                            "too many combined references for the selected model".to_string()
                        );
                    }
                }
                if caps.frames_and_references_exclusive
                    && !frames.is_empty()
                    && (!image_refs.is_empty() || !video_refs.is_empty() || !audio_refs.is_empty())
                {
                    return Err(
                        "frames and references are mutually exclusive for this model".to_string(),
                    );
                }
                if caps.requires_source_video && args.source_video_media_ref.is_none() {
                    return Err("selected model requires sourceVideoMediaRef".to_string());
                }
                if caps.requires_reference_image && image_refs.is_empty() {
                    return Err("selected model requires an image reference".to_string());
                }
                let source_trim = validate_source_clip(
                    &snapshot.timeline,
                    args.source_clip_id.as_deref(),
                    args.source_video_media_ref.as_deref(),
                )?;

                let mut references = Vec::new();
                if caps.requires_source_video {
                    if let Some(source) = args.source_video_media_ref.as_deref() {
                        references.push(PreparedReference {
                            path: resolve_media(&snapshot, source, ClipType::Video)?,
                            fallback: "video",
                            trim_range: source_trim,
                        });
                    }
                } else {
                    for media_ref in &frames {
                        references.push(PreparedReference::whole(
                            resolve_media(&snapshot, media_ref, ClipType::Image)?,
                            "image",
                        ));
                    }
                }
                for media_ref in &image_refs {
                    references.push(PreparedReference::whole(
                        resolve_media(&snapshot, media_ref, ClipType::Image)?,
                        "image",
                    ));
                }
                for media_ref in &video_refs {
                    references.push(PreparedReference::whole(
                        resolve_media(&snapshot, media_ref, ClipType::Video)?,
                        "video",
                    ));
                }
                for media_ref in &audio_refs {
                    references.push(PreparedReference::whole(
                        resolve_media(&snapshot, media_ref, ClipType::Audio)?,
                        "audio",
                    ));
                }

                let provider = provider_prefix(&entry.id)?;
                let input = GenerationInput {
                    prompt: args.prompt.clone(),
                    model: entry.id.clone(),
                    duration: duration as i32,
                    aspect_ratio,
                    resolution: args.resolution.clone(),
                    image_url_asset_ids: (!frames.is_empty()).then_some(frames),
                    reference_image_asset_ids: (!image_refs.is_empty()).then_some(image_refs),
                    reference_video_asset_ids: (!video_refs.is_empty()).then_some(video_refs),
                    reference_audio_asset_ids: (!audio_refs.is_empty()).then_some(audio_refs),
                    generate_audio: Some(true),
                    ..Default::default()
                };
                let estimated_cost_credits = cost_for_input(entry, &input);
                Ok(PreparedDispatch {
                    plan: PreparedGenerationJob {
                        name: display_name(args.name.as_deref(), &args.prompt, "Generated video"),
                        kind: ClipType::Video,
                        folder_id: args.folder_id,
                        provider: provider.clone(),
                        input,
                        output_count: 1,
                        source_asset_id: args.source_video_media_ref,
                        source_clip_id: args.source_clip_id,
                        estimated_cost_credits,
                        created_at: Some(now_apple_reference_seconds()),
                    },
                    references,
                    timeline_span: None,
                    requires_source_video: caps.requires_source_video,
                    model_kind: ModelKind::Video,
                    managed: !configured.contains(&provider) && managed_available,
                    upscale_target_fps: None,
                    warnings: Vec::new(),
                })
            }
            GenerationRequest::Image(args) => {
                let entry = select_model(
                    &catalog,
                    ModelKind::Image,
                    args.model.as_deref(),
                    &configured,
                    managed_available,
                )?;
                let UiCapabilities::Image(caps) = &entry.ui_capabilities else {
                    return Err("selected model has invalid image capabilities".to_string());
                };
                validate_choice(
                    "aspectRatio",
                    args.aspect_ratio.as_deref(),
                    Some(&caps.aspect_ratios),
                )?;
                validate_choice(
                    "resolution",
                    args.resolution.as_deref(),
                    caps.resolutions.as_deref(),
                )?;
                validate_choice(
                    "quality",
                    args.quality.as_deref(),
                    caps.qualities.as_deref(),
                )?;
                let refs = args.reference_media_refs.clone().unwrap_or_default();
                if !refs.is_empty() && !caps.supports_image_reference {
                    return Err("selected model does not support image references".to_string());
                }
                if refs.len() > caps.max_images as usize {
                    return Err("too many image references for the selected model".to_string());
                }
                let references = refs
                    .iter()
                    .map(|media_ref| {
                        resolve_media(&snapshot, media_ref, ClipType::Image)
                            .map(|path| PreparedReference::whole(path, "image"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let output_count = args.num_images.unwrap_or(1).clamp(1, 4) as usize;
                let provider = provider_prefix(&entry.id)?;
                let input = GenerationInput {
                    prompt: args.prompt.clone(),
                    model: entry.id.clone(),
                    duration: 0,
                    aspect_ratio: args
                        .aspect_ratio
                        .clone()
                        .or_else(|| caps.aspect_ratios.first().cloned())
                        .unwrap_or_default(),
                    resolution: args.resolution.clone(),
                    quality: args.quality.clone(),
                    num_images: Some(output_count as i32),
                    reference_image_asset_ids: (!refs.is_empty()).then_some(refs),
                    ..Default::default()
                };
                let estimated_cost_credits = cost_for_input(entry, &input);
                Ok(PreparedDispatch {
                    plan: PreparedGenerationJob {
                        name: display_name(args.name.as_deref(), &args.prompt, "Generated image"),
                        kind: ClipType::Image,
                        folder_id: args.folder_id,
                        provider: provider.clone(),
                        input,
                        output_count,
                        source_asset_id: None,
                        source_clip_id: None,
                        estimated_cost_credits,
                        created_at: Some(now_apple_reference_seconds()),
                    },
                    references,
                    timeline_span: None,
                    requires_source_video: false,
                    model_kind: ModelKind::Image,
                    managed: !configured.contains(&provider) && managed_available,
                    upscale_target_fps: None,
                    warnings: Vec::new(),
                })
            }
            GenerationRequest::Audio(args) => {
                let entry = select_model(
                    &catalog,
                    ModelKind::Audio,
                    args.model.as_deref(),
                    &configured,
                    managed_available,
                )?;
                let UiCapabilities::Audio(caps) = &entry.ui_capabilities else {
                    return Err("selected model has invalid audio capabilities".to_string());
                };
                let prompt = args.prompt.clone().unwrap_or_default();
                if prompt.chars().count() < caps.min_prompt_length as usize {
                    return Err("prompt is too short for the selected audio model".to_string());
                }
                if args.lyrics.is_some() && !caps.supports_lyrics {
                    return Err("selected model does not support lyrics".to_string());
                }
                if args.instrumental == Some(true) && !caps.supports_instrumental {
                    return Err("selected model does not support instrumental mode".to_string());
                }
                if args.style_instructions.is_some() && !caps.supports_style_instructions {
                    return Err("selected model does not support style instructions".to_string());
                }
                if let Some(voice) = args.voice.as_deref() {
                    if !caps.supports_voice(voice) {
                        return Err(format!("voice '{voice}' is not supported by this model"));
                    }
                }
                let timeline_span = match (
                    args.video_source_start_frame,
                    args.video_source_end_frame,
                ) {
                    (None, None) => None,
                    (Some(start), Some(end)) => {
                        if args.video_source_media_ref.is_some() {
                            return Err(
                                "timeline span and videoSourceMediaRef are mutually exclusive"
                                    .to_string(),
                            );
                        }
                        let total = snapshot.timeline.total_frames();
                        if start < 0 || end <= start || end > total {
                            return Err(
                                "video source frame range is outside the timeline".to_string()
                            );
                        }
                        Some(PreparedTimelineSpan {
                            timeline: snapshot.timeline.clone(),
                            manifest: snapshot.media.clone(),
                            project_dir: snapshot.project_dir.clone(),
                            start_frame: start,
                            end_frame: end,
                        })
                    }
                    _ => return Err(
                        "videoSourceStartFrame and videoSourceEndFrame must be provided together"
                            .to_string(),
                    ),
                };
                if (timeline_span.is_some() || args.video_source_media_ref.is_some())
                    && !caps
                        .inputs
                        .as_ref()
                        .is_some_and(|inputs| inputs.iter().any(|input| input == "video"))
                {
                    return Err("selected audio model does not support a video source".to_string());
                }
                let references = match args.video_source_media_ref.as_deref() {
                    Some(media_ref) => vec![PreparedReference::whole(
                        resolve_media(&snapshot, media_ref, ClipType::Video)?,
                        "video",
                    )],
                    None => Vec::new(),
                };
                let provider = provider_prefix(&entry.id)?;
                let input = GenerationInput {
                    prompt: prompt.clone(),
                    model: entry.id.clone(),
                    duration: args.duration.unwrap_or_else(|| {
                        timeline_span
                            .as_ref()
                            .map(|span| {
                                ((span.end_frame - span.start_frame) as f64
                                    / span.timeline.fps.max(1) as f64)
                                    .ceil() as i32
                            })
                            .unwrap_or(0)
                    }),
                    aspect_ratio: String::new(),
                    voice: args.voice,
                    lyrics: args.lyrics,
                    style_instructions: args.style_instructions,
                    instrumental: args.instrumental,
                    reference_video_asset_ids: args
                        .video_source_media_ref
                        .clone()
                        .map(|id| vec![id]),
                    source_start_frame: args.video_source_start_frame,
                    source_end_frame: args.video_source_end_frame,
                    ..Default::default()
                };
                let estimated_cost_credits = cost_for_input(entry, &input);
                Ok(PreparedDispatch {
                    plan: PreparedGenerationJob {
                        name: display_name(args.name.as_deref(), &prompt, "Generated audio"),
                        kind: ClipType::Audio,
                        folder_id: args.folder_id,
                        provider: provider.clone(),
                        input,
                        output_count: 1,
                        source_asset_id: args.video_source_media_ref,
                        source_clip_id: None,
                        estimated_cost_credits,
                        created_at: Some(now_apple_reference_seconds()),
                    },
                    references,
                    timeline_span,
                    requires_source_video: false,
                    model_kind: ModelKind::Audio,
                    managed: !configured.contains(&provider) && managed_available,
                    upscale_target_fps: None,
                    warnings: Vec::new(),
                })
            }
            GenerationRequest::Upscale(args) => {
                let source = snapshot
                    .media
                    .entries
                    .iter()
                    .find(|entry| entry.id == args.media_ref)
                    .ok_or_else(|| "upscale source asset does not exist".to_string())?;
                if !matches!(source.kind, ClipType::Image | ClipType::Video) {
                    return Err("upscale source must be image or video".to_string());
                }
                let source_trim = validate_source_clip(
                    &snapshot.timeline,
                    args.source_clip_id.as_deref(),
                    Some(&args.media_ref),
                )?;
                let entry = select_model(
                    &catalog,
                    ModelKind::Upscale,
                    args.model.as_deref(),
                    &configured,
                    managed_available,
                )?;
                let UiCapabilities::Upscale(caps) = &entry.ui_capabilities else {
                    return Err("selected model has invalid upscale capabilities".to_string());
                };
                let source_kind = if source.kind == ClipType::Image {
                    "image"
                } else {
                    "video"
                };
                if !caps.supported_types.iter().any(|kind| kind == source_kind) {
                    return Err("selected upscaler does not support the source type".to_string());
                }
                // A resolution-targeted video upscaler takes an output size,
                // not a scale factor: ask for the smallest target above the
                // source and keep its frame rate (refusing before anything
                // is paid for when no larger target exists).
                let targets = caps.target_resolutions.as_deref().unwrap_or_default();
                let (resolution, upscale_target_fps, warnings, size_label) =
                    if source.kind == ClipType::Video && !targets.is_empty() {
                        let plan = plan_video_upscale(
                            positive_dimension(source.source_width),
                            positive_dimension(source.source_height),
                            source.source_fps,
                        )
                        .map_err(|error| error.to_string())?;
                        if !targets
                            .iter()
                            .any(|target| target.eq_ignore_ascii_case(plan.resolution.value))
                        {
                            return Err(format!(
                                "the selected upscaler cannot produce {}",
                                plan.resolution.label
                            ));
                        }
                        (
                            Some(plan.resolution.value.to_string()),
                            plan.fps,
                            plan.fps_warning.into_iter().collect(),
                            plan.resolution.label,
                        )
                    } else {
                        (None, None, Vec::new(), "2x")
                    };
                let source_path = resolve_media(&snapshot, &args.media_ref, source.kind)?;
                let provider = provider_prefix(&entry.id)?;
                let input = GenerationInput {
                    prompt: String::new(),
                    model: entry.id.clone(),
                    duration: source.duration.max(0.0).round() as i32,
                    aspect_ratio: String::new(),
                    resolution,
                    source_asset_id: Some(args.media_ref.clone()),
                    source_clip_id: args.source_clip_id.clone(),
                    ..Default::default()
                };
                let estimated_cost_credits = cost_for_input(entry, &input);
                Ok(PreparedDispatch {
                    plan: PreparedGenerationJob {
                        name: format!("{} {size_label}", source.name),
                        kind: source.kind,
                        folder_id: source.folder_id.clone(),
                        provider: provider.clone(),
                        input,
                        output_count: 1,
                        source_asset_id: Some(args.media_ref),
                        source_clip_id: args.source_clip_id,
                        estimated_cost_credits,
                        created_at: Some(now_apple_reference_seconds()),
                    },
                    references: vec![PreparedReference {
                        path: source_path,
                        fallback: source_kind,
                        trim_range: source_trim,
                    }],
                    timeline_span: None,
                    requires_source_video: false,
                    model_kind: ModelKind::Upscale,
                    managed: !configured.contains(&provider) && managed_available,
                    upscale_target_fps,
                    warnings,
                })
            }
        }
    }

    async fn run_job(
        self,
        binding: Arc<JobBinding>,
        task: u64,
        local_job_id: String,
        placeholder_ids: Vec<String>,
        prepared: PreparedDispatch,
        cancel: MediaCancelToken,
    ) {
        let mut accepted = AcceptedSubmission::default();
        let mut result = self
            .run_job_inner(
                &binding,
                &local_job_id,
                &placeholder_ids,
                &prepared,
                &cancel,
                &mut accepted,
            )
            .await;
        let detached = binding.is_detached();
        let outcome_unknown =
            matches!(&result, Err(JobStop::Failed(code)) if code == SUBMIT_OUTCOME_UNKNOWN);
        // Whether the pending-submission record stays for recovery.
        let mut keep_record = false;
        if result.is_err() {
            match accepted.provider_job_id.take() {
                // A paid job its project did not record (it was replaced, or
                // the write failed), or whose results no later poll can fetch
                // while it is detached, is kept in application data.
                Some(provider_job_id)
                    if !accepted.recorded || (detached && accepted.terminal.is_some()) =>
                {
                    self.keep_accepted_job(
                        &binding,
                        &local_job_id,
                        provider_job_id,
                        accepted.terminal.take(),
                    )
                    .await;
                    keep_record = true;
                    if !detached {
                        // Resumed from the record like an interrupted watch,
                        // instead of failing a job the provider accepted.
                        result = Err(JobStop::Detached);
                    }
                }
                Some(_) => {}
                // The provider may have accepted the job, and its project can
                // no longer be told: recovery reports the unknown outcome.
                None if detached && outcome_unknown => {
                    eprintln!(
                        "[generation] job {local_job_id}: {SUBMIT_OUTCOME_UNKNOWN} after its \
                         project was closed; reopening the project reports it"
                    );
                    keep_record = accepted.pending_recorded;
                }
                None => {}
            }
        }
        self.settle_job(&binding, &local_job_id, &placeholder_ids, result, &cancel)
            .await;
        // Settled (or definitely not paid): the project holds the outcome,
        // unless it was replaced before the unknown outcome was written.
        if accepted.pending_recorded && !keep_record && !(outcome_unknown && binding.is_detached())
        {
            self.forget_orphan_off_thread(&local_job_id).await;
        }
        self.finish_task(&local_job_id, task);
    }

    /// Persist how a job ended. A job detached from a replaced project writes
    /// nothing: its durable state belongs to recovery. Otherwise a user
    /// cancellation wins over a failure it caused, except for an unknown
    /// submission outcome, which the user must see because the provider may
    /// have billed the job. A job whose polling was interrupted keeps its
    /// state and is polled again after a while.
    async fn settle_job(
        &self,
        binding: &Arc<JobBinding>,
        local_job_id: &str,
        placeholder_ids: &[String],
        result: Result<(), JobStop>,
        cancel: &MediaCancelToken,
    ) {
        if binding.is_detached() {
            if result.is_err() {
                eprintln!(
                    "[generation] job {local_job_id}: left its project; its state is kept for \
                     recovery"
                );
            }
            return;
        }
        let resumed_by_retry = {
            let mut resumed = self
                .runtime
                .retry_resumed
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if matches!(result, Err(JobStop::Detached)) {
                resumed.contains(local_job_id)
            } else {
                resumed.remove(local_job_id)
            }
        };
        let result = match result {
            // A resume that fails the same way would repeat forever; the
            // next retry submits the job again.
            Err(JobStop::Failed(code))
                if resumed_by_retry && RESUMABLE_FAILURES.contains(&code.as_str()) =>
            {
                eprintln!("[generation] job {local_job_id}: resumed job failed again ({code})");
                Err(JobStop::Failed(RESUME_FAILED.to_string()))
            }
            result => result,
        };
        match result {
            Ok(()) => {}
            Err(JobStop::Failed(code)) if code == SUBMIT_OUTCOME_UNKNOWN => {
                self.fail_bound_outputs(binding, placeholder_ids, &code)
                    .await;
            }
            Err(_) if cancel.is_cancelled() => {
                self.cancel_bound_outputs(binding, placeholder_ids).await;
            }
            Err(JobStop::Cancelled) => {
                self.cancel_bound_outputs(binding, placeholder_ids).await;
            }
            Err(JobStop::Failed(code)) => {
                self.fail_bound_outputs(binding, placeholder_ids, &code)
                    .await;
            }
            Err(JobStop::Detached) => self.schedule_resume(local_job_id),
        }
    }

    /// Poll an interrupted job again after `resume_delay`, a bounded number
    /// of times per session; it stays recoverable on reopen either way.
    fn schedule_resume(&self, job_id: &str) {
        let attempt = {
            let mut resumes = self
                .runtime
                .resumes
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let attempt = resumes.entry(job_id.to_string()).or_insert(0);
            *attempt += 1;
            *attempt
        };
        if attempt > INTERRUPTED_RESUMES_MAX {
            eprintln!(
                "[generation] job {job_id}: polling stopped before the provider finished; it \
                 resumes when the project is reopened"
            );
            return;
        }
        eprintln!(
            "[generation] job {job_id}: polling stopped before the provider finished; resuming \
             in {} s",
            self.timings.resume_delay.as_secs()
        );
        let bridge = self.clone();
        let job_id = job_id.to_string();
        tauri::async_runtime::spawn(async move {
            tokio::time::sleep(bridge.timings.resume_delay).await;
            let recovery = bridge.clone();
            if let Err(error) = tauri::async_runtime::spawn_blocking(move || {
                recovery.recover_jobs(Some(&job_id));
            })
            .await
            {
                eprintln!("[generation] resuming an interrupted job failed: {error}");
            }
        });
    }

    async fn run_job_inner(
        &self,
        binding: &Arc<JobBinding>,
        local_job_id: &str,
        placeholder_ids: &[String],
        prepared: &PreparedDispatch,
        cancel: &MediaCancelToken,
        accepted: &mut AcceptedSubmission,
    ) -> Result<(), JobStop> {
        cancelled(cancel)?;
        let client = self
            .clients
            .build(&prepared.plan.provider, prepared.managed)
            .map_err(|_| JobStop::Failed("GENERATION_AUTH_FAILED".to_string()))?;
        let mut references = prepared.references.clone();
        let timeline_cleanup = if let Some(span) = prepared.timeline_span.as_ref() {
            std::fs::create_dir_all(&self.staging_root)
                .map_err(|_| "GENERATION_SOURCE_PREPROCESS_FAILED".to_string())?;
            let destination = self.staging_root.join(format!(
                "{local_job_id}-{}.timeline.mp4",
                uuid::Uuid::new_v4()
            ));
            let timeline = span.timeline.clone();
            let manifest = span.manifest.clone();
            let project_dir = span.project_dir.clone();
            let start_frame = span.start_frame;
            let end_frame = span.end_frame;
            let output = destination.clone();
            // Export has its own final success boundary; committing its child
            // token must not make the later upload/download workflow immune
            // to cancellation on the parent generation token.
            let export_cancel = cancel.child();
            tokio::task::spawn_blocking(move || {
                crate::export::run_export_with_control(
                    &timeline,
                    &manifest,
                    &project_dir,
                    &crate::export::ExportRequest {
                        out_path: output.to_string_lossy().into_owned(),
                        codec: crate::export::ExportCodec::default(),
                        quality: crate::export::ExportQuality::default(),
                    },
                    crate::export::ExportRunOptions {
                        external_cancel: Some(export_cancel),
                        frame_range: Some((start_frame, end_frame)),
                        ..crate::export::ExportRunOptions::default()
                    },
                )
            })
            .await
            .map_err(|_| "GENERATION_SOURCE_PREPROCESS_FAILED".to_string())?
            .map_err(|error| {
                if error == crate::export::CANCELLED_SENTINEL {
                    "GENERATION_CANCELLED".to_string()
                } else {
                    "GENERATION_SOURCE_PREPROCESS_FAILED".to_string()
                }
            })?;
            references.push(PreparedReference::whole(destination.clone(), "video"));
            Some(StagedCleanup::new(destination))
        } else {
            None
        };
        let mut uploaded = Vec::with_capacity(references.len());
        for reference in &references {
            cancelled(cancel)?;
            let staged_trim = if let Some((start, end)) = reference.trim_range {
                let destination = self
                    .staging_root
                    .join(format!("{local_job_id}-{}.trim.mp4", uuid::Uuid::new_v4()));
                let source = reference.path.clone();
                let output = destination.clone();
                let trim_cancel = cancel.clone();
                tokio::task::spawn_blocking(move || {
                    opentake_media::trim_video_range(&source, &output, start, end, &trim_cancel)
                })
                .await
                .map_err(|_| "GENERATION_SOURCE_PREPROCESS_FAILED".to_string())?
                .map_err(|error| {
                    if matches!(error, opentake_media::MediaError::Cancelled) {
                        "GENERATION_CANCELLED".to_string()
                    } else {
                        "GENERATION_SOURCE_PREPROCESS_FAILED".to_string()
                    }
                })?;
                Some(StagedCleanup::new(destination))
            } else {
                None
            };
            let upload_path = staged_trim
                .as_ref()
                .map_or(reference.path.as_path(), |staged| staged.path.as_path());
            let content_type = opentake_gen::content_type_for(upload_path, reference.fallback);
            let upload = async {
                if prepared.managed {
                    client.upload_reference(upload_path, &content_type).await
                } else {
                    client
                        .upload_reference_via(&prepared.plan.provider, upload_path, &content_type)
                        .await
                }
            };
            // An upload has no provider-side effect worth keeping, so a
            // cancellation abandons it at once instead of waiting on the
            // network.
            let uploaded_url = tokio::select! {
                result = upload => result,
                () = wait_for_cancel(cancel) => return Err(JobStop::Cancelled),
            };
            drop(staged_trim);
            let uploaded_url = uploaded_url.map_err(|error| {
                JobStop::Failed(generation_provider_error_code(
                    &error,
                    "GENERATION_REFERENCE_UPLOAD_FAILED",
                ))
            })?;
            uploaded.push(uploaded_url);
        }
        drop(timeline_cleanup);
        cancelled(cancel)?;
        let mut params = build_params(
            &prepared.plan.input,
            &uploaded,
            prepared.model_kind,
            prepared.requires_source_video,
        );
        if let GenerationParams::Upscale(upscale) = &mut params {
            upscale.target_fps = prepared.upscale_target_fps;
        }
        // Until the project records the answer, the submission's outcome is
        // unknown; a durable record says so if the app quits or the project
        // closes first.
        self.record_pending_submission(binding, local_job_id)
            .await?;
        accepted.pending_recorded = true;
        // A submission is not raced against cancellation: once sent, the
        // provider may accept and bill it. Its answer is awaited within a
        // bound, and an accepted job id is persisted before a cancellation is
        // honored, so no paid job is left without a record.
        // A synchronous provider (OpenAI, ElevenLabs) answers with the
        // finished job: its results exist only in this answer.
        let submission = async {
            if prepared.managed {
                client
                    .submit(&prepared.plan.input.model, params, Some(local_job_id))
                    .await
                    .map(|id| (id, None))
            } else {
                client
                    .submit_byok_job(&prepared.plan.input.model, params)
                    .await
                    .map(|job| {
                        let terminal = (job.status == JobStatus::Succeeded).then(|| job.clone());
                        (job.id, terminal)
                    })
            }
        };
        let (provider_job_id, terminal) =
            match tokio::time::timeout(self.timings.submit_timeout, submission).await {
                Err(_) => return Err(JobStop::Failed(SUBMIT_OUTCOME_UNKNOWN.to_string())),
                Ok(Err(error)) => return Err(JobStop::Failed(submit_error_code(&error))),
                Ok(Ok(submitted)) => submitted,
            };
        accepted.provider_job_id = Some(provider_job_id.clone());
        accepted.terminal = terminal.clone();
        let update = GenerationStateUpdate {
            status: GenerationJobStatus::Generating,
            progress: Some(0.15),
            error_code: None,
            provider_job_id: Some(provider_job_id.clone()),
            cost_credits: None,
            created_at: Some(now_apple_reference_seconds()),
        };
        let write_job_id = local_job_id.to_string();
        match self
            .bound_write(binding, move |core, project_epoch, project_dir| {
                core.update_generation_job_for_project(
                    project_epoch,
                    project_dir,
                    &write_job_id,
                    update.clone(),
                )
            })
            .await
        {
            Ok(_) => accepted.recorded = true,
            // The project was replaced while the provider accepted the job,
            // or could not record it: the caller keeps the accepted id.
            Err(BoundWriteError::Detached) => return Err(JobStop::Detached),
            Err(BoundWriteError::Failed(_)) => {
                return Err(JobStop::Failed(STATE_PERSIST_FAILED.to_string()))
            }
        }
        // The project holds the provider job id now.
        self.forget_orphan_off_thread(local_job_id).await;
        accepted.pending_recorded = false;
        cancelled(cancel)?;

        if let Some(job) = terminal {
            let staging_root = self.staging_root.clone();
            let download_cancel = cancel.clone();
            return self
                .finalize_succeeded(
                    binding,
                    local_job_id,
                    placeholder_ids,
                    &provider_job_id,
                    job.cost_credits,
                    job.result_urls.unwrap_or_default(),
                    move || {
                        SecureResultDownloader::new(staging_root, download_cancel)
                            .map(|downloader| Box::new(downloader) as Box<ResultDownloader>)
                    },
                )
                .await;
        }
        self.watch_and_finalize(
            binding,
            local_job_id,
            placeholder_ids,
            client,
            &provider_job_id,
            self.timings.watch_deadline(Some(prepared.model_kind)),
            cancel,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn watch_and_finalize(
        &self,
        binding: &Arc<JobBinding>,
        local_job_id: &str,
        placeholder_ids: &[String],
        client: GenClient,
        provider_job_id: &str,
        deadline: Duration,
        cancel: &MediaCancelToken,
    ) -> Result<(), JobStop> {
        let stream = client.watch(provider_job_id, deadline);
        futures_util::pin_mut!(stream);
        loop {
            cancelled(cancel)?;
            let event = tokio::select! {
                event = stream.next() => event,
                () = wait_for_cancel(cancel) => return Err(JobStop::Cancelled),
            };
            let job = match event {
                Some(WatchEvent::Snapshot(job)) => job,
                Some(WatchEvent::Retrying {
                    attempt,
                    delay,
                    error,
                }) => {
                    // No URL or credential: only the failure kind is logged.
                    eprintln!(
                        "[generation] job {local_job_id}: status poll failed ({}); retry \
                         {attempt} in {} ms",
                        error.kind_label(),
                        delay.as_millis()
                    );
                    continue;
                }
                Some(WatchEvent::Failed(error)) => {
                    return Err(JobStop::Failed(generation_provider_error_code(
                        &error,
                        PROVIDER_POLL_REFUSED,
                    )))
                }
                Some(WatchEvent::Interrupted(reason)) => {
                    eprintln!(
                        "[generation] job {local_job_id}: polling stopped before the provider \
                         finished ({reason:?})"
                    );
                    return Err(JobStop::Detached);
                }
                // The watch always ends with one of the events above; if it
                // ever does not, the job must stay recoverable.
                None => return Err(JobStop::Detached),
            };
            match job.status {
                JobStatus::Queued => {}
                JobStatus::Running => {
                    // Identical polls are no-ops in the core and progress-only
                    // polls stay in memory, so this never rewrites the bundle.
                    // A failure here does not end the job, but is not silent.
                    let running = GenerationStateUpdate {
                        status: GenerationJobStatus::Generating,
                        progress: Some(0.5),
                        error_code: None,
                        provider_job_id: Some(provider_job_id.to_string()),
                        cost_credits: None,
                        created_at: Some(now_apple_reference_seconds()),
                    };
                    let write_job_id = local_job_id.to_string();
                    match self
                        .bound_write(binding, move |core, project_epoch, project_dir| {
                            core.update_generation_job_for_project(
                                project_epoch,
                                project_dir,
                                &write_job_id,
                                running.clone(),
                            )
                        })
                        .await
                    {
                        Ok(_) => {}
                        Err(BoundWriteError::Detached) => return Err(JobStop::Detached),
                        Err(BoundWriteError::Failed(error)) => eprintln!(
                            "[generation] job {local_job_id}: progress was not persisted: {error}"
                        ),
                    }
                }
                JobStatus::Failed => {
                    return Err(JobStop::Failed("GENERATION_PROVIDER_FAILED".to_string()))
                }
                JobStatus::Succeeded => {
                    let staging_root = self.staging_root.clone();
                    let download_cancel = cancel.clone();
                    return self
                        .finalize_succeeded(
                            binding,
                            local_job_id,
                            placeholder_ids,
                            provider_job_id,
                            job.cost_credits,
                            job.result_urls.unwrap_or_default(),
                            move || {
                                SecureResultDownloader::new(staging_root, download_cancel)
                                    .map(|downloader| Box::new(downloader) as Box<ResultDownloader>)
                            },
                        )
                        .await;
                }
            }
        }
    }

    /// Download a succeeded job's results and publish them into its
    /// placeholders. `downloader` is built on the blocking finalization
    /// thread.
    #[allow(clippy::too_many_arguments)]
    async fn finalize_succeeded(
        &self,
        binding: &Arc<JobBinding>,
        local_job_id: &str,
        placeholder_ids: &[String],
        provider_job_id: &str,
        cost_credits: Option<i64>,
        urls: Vec<String>,
        downloader: impl FnOnce() -> Result<Box<ResultDownloader>, String> + Send + 'static,
    ) -> Result<(), JobStop> {
        let downloading = GenerationStateUpdate {
            status: GenerationJobStatus::Downloading,
            progress: Some(0.8),
            error_code: None,
            provider_job_id: Some(provider_job_id.to_string()),
            cost_credits,
            created_at: Some(now_apple_reference_seconds()),
        };
        let write_job_id = local_job_id.to_string();
        self.bound_write(binding, move |core, project_epoch, project_dir| {
            core.update_generation_job_for_project(
                project_epoch,
                project_dir,
                &write_job_id,
                downloading.clone(),
            )
        })
        .await
        .map_err(|error| match error {
            BoundWriteError::Detached => JobStop::Detached,
            BoundWriteError::Failed(_) => JobStop::Failed(STATE_PERSIST_FAILED.to_string()),
        })?;
        let store = TauriFinalizationStore {
            bridge: self.clone(),
            binding: Arc::clone(binding),
            lease: Mutex::new(None),
        };
        let terminal_job_id = local_job_id.to_string();
        let terminal_placeholder_ids = placeholder_ids.to_vec();
        tokio::task::spawn_blocking(move || {
            let downloader = downloader()?;
            finalize_terminal_outputs(
                &store,
                downloader.as_ref(),
                &terminal_job_id,
                &terminal_placeholder_ids,
                &urls,
            )
        })
        .await
        .map_err(|_| JobStop::Failed("GENERATION_FINALIZE_TASK_FAILED".to_string()))?
        .map_err(|error| {
            if error == "GENERATION_CANCELLED" {
                JobStop::Cancelled
            } else {
                JobStop::Failed("GENERATION_FINALIZE_FAILED".to_string())
            }
        })?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_recovered_job(
        self,
        binding: Arc<JobBinding>,
        task: u64,
        local_job_id: String,
        placeholder_ids: Vec<String>,
        provider: String,
        managed: bool,
        provider_job_id: String,
        deadline: Duration,
        cancel: MediaCancelToken,
    ) {
        let result = match self.clients.build(&provider, managed) {
            Ok(client) => {
                self.watch_and_finalize(
                    &binding,
                    &local_job_id,
                    &placeholder_ids,
                    client,
                    &provider_job_id,
                    deadline,
                    &cancel,
                )
                .await
            }
            Err(_) => Err(JobStop::Failed(
                "GENERATION_RECOVERY_AUTH_UNAVAILABLE".to_string(),
            )),
        };
        self.settle_job(&binding, &local_job_id, &placeholder_ids, result, &cancel)
            .await;
        self.finish_task(&local_job_id, task);
    }

    /// Finalize a synchronous provider's result held in application data
    /// for a job that was accepted after its project closed.
    #[allow(clippy::too_many_arguments)]
    async fn run_held_job(
        self,
        binding: Arc<JobBinding>,
        task: u64,
        local_job_id: String,
        placeholder_ids: Vec<String>,
        provider_job_id: String,
        held: Vec<crate::generation_orphans::HeldResult>,
        cancel: MediaCancelToken,
    ) {
        let urls = (0..held.len()).map(held_result_url).collect::<Vec<_>>();
        let files = held
            .iter()
            .map(|held| (self.orphans.held_path(held), held.media_type.clone()))
            .collect::<Vec<_>>();
        let staging_root = self.staging_root.clone();
        let download_cancel = cancel.clone();
        let result = self
            .finalize_succeeded(
                &binding,
                &local_job_id,
                &placeholder_ids,
                &provider_job_id,
                None,
                urls,
                move || {
                    std::fs::create_dir_all(&staging_root)
                        .map_err(|_| "generation staging directory is unavailable".to_string())?;
                    Ok(Box::new(HeldResultDownloader {
                        files,
                        staging_root,
                        cancel: download_cancel,
                    }) as Box<ResultDownloader>)
                },
            )
            .await;
        let interrupted = matches!(result, Err(JobStop::Detached));
        self.settle_job(&binding, &local_job_id, &placeholder_ids, result, &cancel)
            .await;
        // The held results are kept while the job can still be finalized
        // from them: it left its project, was interrupted, or its end state
        // was not saved, so its placeholders still wait for a result the
        // provider cannot return again. Once the project records every
        // output as finished, they are removed.
        if !interrupted && !binding.is_detached() && self.outputs_terminal(&placeholder_ids) {
            self.forget_orphan_off_thread(&local_job_id).await;
        }
        self.finish_task(&local_job_id, task);
    }
}

/// What a submission's provider accepted, for keeping a job whose project
/// was replaced before the job was recorded or finalized.
#[derive(Default)]
struct AcceptedSubmission {
    provider_job_id: Option<String>,
    /// The finished job a synchronous provider answered with.
    terminal: Option<GenerationJob>,
    /// Whether the open project recorded the provider job id.
    recorded: bool,
    /// Whether the pending-submission record is in the orphan store.
    pending_recorded: bool,
}

type ResultDownloader = dyn GenerationArtifactDownloader + Send;

/// Placeholder URL of held result `index`: finalization only accepts HTTPS
/// or `data:` URLs, and [`HeldResultDownloader`] maps this one to the file.
fn held_result_url(index: usize) -> String {
    format!("https://held-result.opentake.invalid/{index}")
}

/// Serves results held in application data. Each download is a staging
/// copy, so the held file survives until the job is finalized.
struct HeldResultDownloader {
    files: Vec<(PathBuf, String)>,
    staging_root: PathBuf,
    cancel: MediaCancelToken,
}

impl GenerationArtifactDownloader for HeldResultDownloader {
    fn download(&self, asset_id: &str, url: &str) -> Result<DownloadedGenerationArtifact, String> {
        cancelled(&self.cancel)?;
        let (source, media_type) = url
            .strip_prefix("https://held-result.opentake.invalid/")
            .and_then(|index| index.parse::<usize>().ok())
            .and_then(|index| self.files.get(index))
            .ok_or_else(|| "held generation result is unknown".to_string())?;
        let path = self
            .staging_root
            .join(format!("{asset_id}-{}.download", uuid::Uuid::new_v4()));
        let cleanup = StagedCleanup::new(path.clone());
        let mut input = std::fs::File::open(source)
            .map_err(|_| "held generation result is unavailable".to_string())?;
        let mut output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|_| "generation staging file creation failed".to_string())?;
        let byte_size = std::io::copy(&mut input, &mut output)
            .and_then(|size| output.sync_all().map(|()| size))
            .map_err(|_| "generation staging write failed".to_string())?;
        cleanup.preserve();
        Ok(DownloadedGenerationArtifact {
            path,
            media_type: media_type.clone(),
            byte_size,
        })
    }
}

impl GenerationBridge for TauriGenerationBridge {
    fn can_generate(&self) -> bool {
        !self.configured_byok_prefixes().is_empty() || self.has_managed_credential()
    }

    fn submit(
        &self,
        request: GenerationRequest,
        cancel: &MediaCancelToken,
    ) -> Result<GenerationSubmission, String> {
        cancelled(cancel)?;
        let admission = self.admission.begin_activity()?;
        let prepared = self.prepare(request)?;
        let snapshot = self.core.runtime_snapshot();
        let project_dir = snapshot
            .project_dir
            .clone()
            .ok_or_else(|| "Save the project before starting generation".to_string())?;
        // Register the job before recovery can see its placeholders: a
        // recovery (after a Save As, say) would otherwise fail the Queued
        // placeholders while this task goes on to submit and pay.
        let registration = self
            .runtime
            .registration
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let committed = self
            .core
            .begin_generation_job_for_project(
                snapshot.project_epoch,
                &project_dir,
                prepared.plan.clone(),
            )
            .map_err(|error| error.to_string())?;
        let background_cancel = MediaCancelToken::new();
        let binding = JobBinding::new(snapshot.project_epoch, project_dir);
        let task = self
            .runtime
            .next_task
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.runtime
            .jobs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(
                committed.job_id.clone(),
                ActiveGenerationJob {
                    cancel: background_cancel.clone(),
                    binding: binding.clone(),
                    task,
                    admission,
                },
            );
        drop(registration);
        let bridge = self.clone();
        let job_id = committed.job_id.clone();
        let placeholder_ids = committed.placeholder_asset_ids.clone();
        let warnings = prepared.warnings.clone();
        tauri::async_runtime::spawn(async move {
            bridge
                .run_job(
                    binding,
                    task,
                    job_id,
                    placeholder_ids,
                    prepared,
                    background_cancel,
                )
                .await;
        });
        Ok(GenerationSubmission {
            job_id: committed.job_id,
            placeholder_asset_ids: committed.placeholder_asset_ids,
            status: "queued".to_string(),
            warnings,
        })
    }
}

/// Finalization writes follow the job's binding (a Save As during the
/// download commits into the new bundle); a detached job commits nothing.
struct TauriFinalizationStore {
    bridge: TauriGenerationBridge,
    binding: Arc<JobBinding>,
    /// The lease key claimed by this finalization.
    lease: Mutex<Option<String>>,
}

impl TauriFinalizationStore {
    /// Leases and completions are per bundle: a Save As copy of a
    /// generating job is its own placeholder set.
    fn key(&self, job_id: &str) -> String {
        format!("{}\n{job_id}", self.binding.state().project_dir.display())
    }

    fn write<T>(
        &self,
        write: impl FnMut(u64, &Path) -> opentake_core::Result<T>,
    ) -> Result<T, String> {
        self.bridge
            .bound_write_blocking(&self.binding, write)
            .map_err(|error| match error {
                BoundWriteError::Detached => "generation job left its project".to_string(),
                BoundWriteError::Failed(error) => error,
            })
    }
}

impl GenerationFinalizationStore for TauriFinalizationStore {
    fn claim_terminal(&self, job_id: &str) -> Result<bool, String> {
        if self.binding.is_detached() {
            return Err("generation job left its project".to_string());
        }
        let key = self.key(job_id);
        if self
            .bridge
            .runtime
            .completed
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains(&key)
        {
            return Ok(false);
        }
        let claimed = self
            .bridge
            .runtime
            .terminal_leases
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(key.clone());
        if claimed {
            *self
                .lease
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(key);
        }
        Ok(claimed)
    }

    fn release_terminal(&self, job_id: &str) -> Result<(), String> {
        let key = self
            .lease
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
            .unwrap_or_else(|| self.key(job_id));
        self.bridge
            .runtime
            .terminal_leases
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&key);
        Ok(())
    }

    fn finalize_output(
        &self,
        asset_id: &str,
        artifact: DownloadedGenerationArtifact,
    ) -> Result<(), String> {
        let snapshot = self.bridge.core.runtime_snapshot();
        let entry = snapshot
            .media
            .entries
            .iter()
            .find(|entry| entry.id == asset_id)
            .ok_or_else(|| "generation placeholder disappeared".to_string())?;
        let probe = self
            .bridge
            .engine
            .probe(&artifact.path)
            .map_err(|_| "downloaded generation result could not be probed".to_string())?;
        let actual_kind = if probe.has_video {
            if probe.duration_secs > 0.0 {
                ClipType::Video
            } else {
                ClipType::Image
            }
        } else if probe.has_audio {
            ClipType::Audio
        } else {
            return Err("downloaded generation result has no supported stream".to_string());
        };
        if actual_kind != entry.kind {
            return Err("downloaded generation result has the wrong media type".to_string());
        }
        if let Some(input) = entry.generation_input.as_ref().filter(|input| {
            Catalog::builtin()
                .by_id(&input.model)
                .is_some_and(|model| model.kind == ModelKind::Upscale)
        }) {
            validate_upscale_result(&snapshot.media, input, probe.width, probe.height)?;
        }
        let extension = result_extension(
            &artifact.media_type,
            actual_kind,
            probe.format_name.as_deref(),
        )?;
        let leaf = format!("{asset_id}.{extension}");
        let output = PreparedGenerationOutput {
            asset_id: asset_id.to_string(),
            relative_path: format!("media/{leaf}"),
            probe: ProbedMedia {
                duration_secs: probe.duration_secs,
                width: probe.width.map(|value| value as i32),
                height: probe.height.map(|value| value as i32),
                fps: probe.fps,
                has_audio: probe.has_audio,
                color: probe.color.clone(),
            },
            created_at: Some(now_apple_reference_seconds()),
        };
        self.write(|project_epoch, project_dir| {
            // Reopened per attempt: a retry after a Save As streams again.
            let mut source = std::fs::File::open(&artifact.path).map_err(|error| {
                opentake_core::CoreError::Media(format!(
                    "generation staging result is unavailable: {error}"
                ))
            })?;
            self.bridge
                .core
                .finalize_generation_output_with_media_for_project(
                    project_epoch,
                    project_dir,
                    output.clone(),
                    &leaf,
                    artifact.byte_size,
                    &mut source,
                )
        })?;
        if let Err(error) = std::fs::remove_file(&artifact.path) {
            eprintln!("[generation] staging file was not removed: {error}");
        }
        Ok(())
    }

    fn fail_output(&self, asset_id: &str, code: &str) -> Result<(), String> {
        self.write(|project_epoch, project_dir| {
            self.bridge.core.fail_generation_output_for_project(
                project_epoch,
                project_dir,
                asset_id,
                code,
                Some(now_apple_reference_seconds()),
            )
        })
    }

    fn finished_output(&self, asset_id: &str) -> Result<Option<FinishedOutput>, String> {
        let status = self
            .bridge
            .core
            .media()
            .entries
            .iter()
            .find(|entry| entry.id == asset_id)
            .and_then(|entry| entry.generation_input.as_ref())
            .and_then(|input| input.status);
        Ok(match status {
            Some(GenerationJobStatus::Ready) => Some(FinishedOutput::Succeeded),
            Some(GenerationJobStatus::Failed | GenerationJobStatus::Cancelled) => {
                Some(FinishedOutput::Failed)
            }
            _ => None,
        })
    }

    fn complete_job(&self, job_id: &str, _succeeded: usize, _failed: usize) -> Result<(), String> {
        self.release_terminal(job_id)?;
        self.bridge
            .runtime
            .completed
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(self.key(job_id));
        Ok(())
    }
}

/// Resolves a result host to its addresses (injected in tests).
type ResultLookup = dyn Fn(
        String,
        u16,
    ) -> futures_util::future::BoxFuture<'static, std::io::Result<Vec<std::net::SocketAddr>>>
    + Send
    + Sync;

/// Downloads provider results from provider-chosen URLs. Before each request
/// (every redirect hop included) the host is resolved, every address must be
/// public (the same policy as `source.url` imports), and the connection is
/// pinned to the checked addresses so the name cannot resolve elsewhere.
struct SecureResultDownloader {
    staging_root: PathBuf,
    cancel: MediaCancelToken,
    /// Runs DNS lookups with a timeout and cancellation from blocking code.
    dns: tokio::runtime::Runtime,
    lookup: Arc<ResultLookup>,
    /// Which URLs and addresses may be fetched (tests widen both to reach a
    /// local server).
    url_policy: fn(&str) -> Result<reqwest::Url, String>,
    address_policy: fn(std::net::IpAddr) -> bool,
}

impl SecureResultDownloader {
    fn new(staging_root: PathBuf, cancel: MediaCancelToken) -> Result<Self, String> {
        Self::with_lookup(
            staging_root,
            cancel,
            Arc::new(|host, port| Box::pin(crate::public_net::system_lookup(host, port))),
        )
    }

    fn with_lookup(
        staging_root: PathBuf,
        cancel: MediaCancelToken,
        lookup: Arc<ResultLookup>,
    ) -> Result<Self, String> {
        std::fs::create_dir_all(&staging_root)
            .map_err(|_| "generation staging directory is unavailable".to_string())?;
        let dns = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|_| "generation result resolver initialization failed".to_string())?;
        Ok(Self {
            staging_root,
            cancel,
            dns,
            lookup,
            url_policy: validate_result_https_url,
            address_policy: crate::public_net::public_ip,
        })
    }

    /// A client for one request to `url` that can only connect to the
    /// checked public addresses of its host, and those addresses (the IP of
    /// an address literal).
    fn pinned_client(
        &self,
        url: &reqwest::Url,
    ) -> Result<(reqwest::blocking::Client, Vec<std::net::IpAddr>), String> {
        let lookup = Arc::clone(&self.lookup);
        let (host, pinned) = self
            .dns
            .block_on(crate::public_net::resolve_target_with_policy(
                url,
                &self.cancel,
                move |host, port| lookup(host, port),
                self.address_policy,
            ))
            .map_err(|error| match error {
                crate::public_net::PublicTargetError::Cancelled => {
                    "GENERATION_CANCELLED".to_string()
                }
                crate::public_net::PublicTargetError::NonPublicAddress => {
                    "generation result host is not a public address".to_string()
                }
                _ => "generation result host could not be resolved".to_string(),
            })?;
        let mut builder = reqwest::blocking::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(15))
            .timeout(Duration::from_secs(10 * 60));
        if !pinned.is_empty() {
            // Every checked address, so a dead edge falls back to the next.
            builder = builder.resolve_to_addrs(&host, &pinned);
        }
        let expected = crate::public_net::expected_peers(&host, &pinned);
        let client = builder
            .build()
            .map_err(|_| "generation result client initialization failed".to_string())?;
        Ok((client, expected))
    }

    fn write_staging(
        &self,
        asset_id: &str,
        media_type: String,
        bytes: &[u8],
    ) -> Result<DownloadedGenerationArtifact, String> {
        if bytes.len() as u64 > RESULT_BYTES_MAX {
            return Err("generation result exceeds the download limit".to_string());
        }
        let path = self
            .staging_root
            .join(format!("{asset_id}-{}.download", uuid::Uuid::new_v4()));
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        let mut file = options
            .open(&path)
            .map_err(|_| "generation staging file creation failed".to_string())?;
        let cleanup = StagedCleanup::new(path.clone());
        file.write_all(bytes)
            .and_then(|()| file.sync_all())
            .map_err(|_| "generation staging write failed".to_string())?;
        cleanup.preserve();
        Ok(DownloadedGenerationArtifact {
            path,
            media_type,
            byte_size: bytes.len() as u64,
        })
    }
}

impl GenerationArtifactDownloader for SecureResultDownloader {
    fn download(
        &self,
        asset_id: &str,
        raw_url: &str,
    ) -> Result<DownloadedGenerationArtifact, String> {
        cancelled(&self.cancel)?;
        if raw_url.starts_with("data:") {
            if raw_url.len() > DATA_URL_ENCODED_MAX {
                return Err("generation data URL exceeds the download limit".to_string());
            }
            let (header, encoded) = raw_url
                .split_once(',')
                .ok_or_else(|| "generation data URL is malformed".to_string())?;
            let media_type = header
                .strip_prefix("data:")
                .and_then(|value| value.strip_suffix(";base64"))
                .filter(|value| {
                    value.starts_with("image/")
                        || value.starts_with("audio/")
                        || value.starts_with("video/")
                })
                .ok_or_else(|| "generation data URL media type is unsupported".to_string())?;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .map_err(|_| "generation data URL base64 is invalid".to_string())?;
            return self.write_staging(asset_id, media_type.to_string(), &bytes);
        }

        let mut current = (self.url_policy)(raw_url)?;
        for redirect_count in 0..=RESULT_REDIRECT_MAX {
            cancelled(&self.cancel)?;
            let (client, expected) = self.pinned_client(&current)?;
            let mut response = client
                .get(current.clone())
                .send()
                .map_err(|_| "generation result download failed".to_string())?;
            // Defense in depth: the connection must have gone to one of the
            // checked addresses; an unknown peer is refused.
            if !crate::public_net::peer_is_expected(
                response.remote_addr(),
                &expected,
                self.address_policy,
            ) {
                return Err(
                    "generation result connection did not reach the checked public address"
                        .to_string(),
                );
            }
            if response.status().is_redirection() {
                if redirect_count == RESULT_REDIRECT_MAX {
                    return Err("generation result exceeded redirect limit".to_string());
                }
                let location = response
                    .headers()
                    .get(reqwest::header::LOCATION)
                    .and_then(|value| value.to_str().ok())
                    .ok_or_else(|| "generation result redirect is invalid".to_string())?;
                current = (self.url_policy)(
                    current
                        .join(location)
                        .map_err(|_| "generation result redirect is invalid".to_string())?
                        .as_str(),
                )?;
                continue;
            }
            if !response.status().is_success() {
                return Err("generation result download returned an error".to_string());
            }
            if response
                .content_length()
                .is_some_and(|length| length > RESULT_BYTES_MAX)
            {
                return Err("generation result exceeds the download limit".to_string());
            }
            let media_type = response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.split(';').next())
                .unwrap_or("application/octet-stream")
                .trim()
                .to_ascii_lowercase();
            let path = self
                .staging_root
                .join(format!("{asset_id}-{}.download", uuid::Uuid::new_v4()));
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(|_| "generation staging file creation failed".to_string())?;
            let cleanup = StagedCleanup::new(path.clone());
            let mut total = 0_u64;
            let mut buffer = [0_u8; 64 * 1024];
            loop {
                cancelled(&self.cancel)?;
                let count = response
                    .read(&mut buffer)
                    .map_err(|_| "generation result stream failed".to_string())?;
                if count == 0 {
                    break;
                }
                total = total.saturating_add(count as u64);
                if total > RESULT_BYTES_MAX {
                    return Err("generation result exceeds the download limit".to_string());
                }
                file.write_all(&buffer[..count])
                    .map_err(|_| "generation staging write failed".to_string())?;
            }
            file.sync_all()
                .map_err(|_| "generation staging write failed".to_string())?;
            cleanup.preserve();
            return Ok(DownloadedGenerationArtifact {
                path,
                media_type,
                byte_size: total,
            });
        }
        Err("generation result download failed".to_string())
    }
}

/// Reuse the production generation downloader for advanced provider workflows.
/// The destination must be a fresh explicit file path chosen by the caller.
pub(crate) fn secure_download_generation_result(
    staging_root: PathBuf,
    cancel: MediaCancelToken,
    raw_url: &str,
    destination: &Path,
) -> Result<(String, u64), String> {
    let downloader = SecureResultDownloader::new(staging_root, cancel)?;
    let artifact = downloader.download("advanced", raw_url)?;
    let mut source = std::fs::File::open(&artifact.path)
        .map_err(|_| "generation staging result disappeared".to_string())?;
    let mut destination_file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)
        .map_err(|_| "generated destination already exists or is unavailable".to_string())?;
    let copy_result =
        std::io::copy(&mut source, &mut destination_file).and_then(|_| destination_file.sync_all());
    let _ = std::fs::remove_file(&artifact.path);
    if copy_result.is_err() {
        let _ = std::fs::remove_file(destination);
        return Err("generated result could not be committed to staging".to_string());
    }
    Ok((artifact.media_type, artifact.byte_size))
}

fn build_client(provider: &str, managed: bool) -> Result<GenClient, String> {
    if managed {
        let (backend, token) = crate::account::generation_credential()?
            .ok_or_else(|| "managed generation credential is unavailable".to_string())?;
        let base = reqwest::Url::parse(&(backend + "/"))
            .map_err(|_| "managed generation backend is invalid".to_string())?;
        return Ok(GenClient::managed(base, Arc::new(StaticToken(token))));
    }
    let store = KeyringStore::new();
    let key = match provider {
        "fal" => ProviderKey::Fal,
        "replicate" => ProviderKey::Replicate,
        "openai" => ProviderKey::OpenAI,
        "elevenlabs" => ProviderKey::ElevenLabs,
        _ => return Err("generation provider is unsupported".to_string()),
    };
    let secret = (&store as &dyn KeyStore)
        .load_key(key)
        .map_err(|_| "generation provider key could not be loaded".to_string())?
        .ok_or_else(|| "generation provider key is not configured".to_string())?;
    let transport = Arc::new(ReqwestTransport::new());
    let registry = match provider {
        "fal" => ProviderRegistry::new().with(Arc::new(FalAdapter::new(transport, secret))),
        "replicate" => {
            ProviderRegistry::new().with(Arc::new(ReplicateAdapter::new(transport, secret)))
        }
        "openai" => ProviderRegistry::new().with(Arc::new(OpenAiAdapter::new(transport, secret))),
        "elevenlabs" => {
            ProviderRegistry::new().with(Arc::new(ElevenLabsAdapter::new(transport, secret)))
        }
        _ => unreachable!(),
    };
    Ok(GenClient::byok(registry, Catalog::builtin()))
}

/// The placeholders of `job_id` when a retry resumes it: every output is
/// Failed or Ready, the failed ones name the same accepted provider job and
/// failed with a [`RESUMABLE_FAILURES`] code, and that provider job can be
/// polled again. A synchronous vendor's (OpenAI, ElevenLabs) result lives
/// only in the memory of the client that submitted it, so its job is
/// submitted again instead.
fn resumable_job(media: &opentake_domain::MediaManifest, job_id: &str) -> Option<Vec<String>> {
    let mut placeholders = Vec::new();
    let mut provider_job_id = None;
    for entry in &media.entries {
        let Some(input) = entry.generation_input.as_ref() else {
            continue;
        };
        if input.job_id.as_deref() != Some(job_id) {
            continue;
        }
        placeholders.push(entry.id.clone());
        match input.status {
            Some(GenerationJobStatus::Ready) => continue,
            Some(GenerationJobStatus::Failed) => {}
            _ => return None,
        }
        let resumable = input
            .error_code
            .as_deref()
            .is_some_and(|code| RESUMABLE_FAILURES.contains(&code));
        let id = input.provider_job_id.as_deref()?;
        if !resumable || *provider_job_id.get_or_insert(id) != id {
            return None;
        }
    }
    let provider_job_id = provider_job_id?;
    let synchronous = ["openai::", "elevenlabs::"]
        .iter()
        .any(|prefix| provider_job_id.starts_with(prefix));
    (!synchronous).then_some(placeholders)
}

fn generation_provider_error_code(error: &GenError, fallback: &str) -> String {
    match error {
        GenError::Unauthenticated | GenError::NotConfigured => "GENERATION_AUTH_FAILED",
        GenError::InsufficientCredits(_) => "GENERATION_INSUFFICIENT_CREDITS",
        GenError::Api { status: 429, .. } => "GENERATION_RATE_LIMITED",
        GenError::UploadTooLarge { .. } => "GENERATION_REFERENCE_TOO_LARGE",
        _ => fallback,
    }
    .to_string()
}

/// A submission that may have been accepted (a reset or timed-out
/// connection, a gateway error, or a 2xx answer this client cannot use) has
/// an unknown outcome; a connection that was never established, or an HTTP
/// refusal, is a plain failure.
fn submit_error_code(error: &GenError) -> String {
    if error.submission_outcome_unknown() {
        SUBMIT_OUTCOME_UNKNOWN.to_string()
    } else {
        generation_provider_error_code(error, "GENERATION_SUBMIT_FAILED")
    }
}

fn select_model<'a>(
    catalog: &'a Catalog,
    kind: ModelKind,
    requested: Option<&str>,
    configured: &BTreeSet<String>,
    managed: bool,
) -> Result<&'a CatalogEntry, String> {
    if let Some(requested) = requested {
        let entry = catalog
            .by_id(requested)
            .ok_or_else(|| "generation model does not exist".to_string())?;
        if entry.kind != kind {
            return Err("generation model has the wrong media kind".to_string());
        }
        let provider = provider_prefix(&entry.id)?;
        if !managed && !configured.contains(&provider) {
            return Err("selected model provider is not configured".to_string());
        }
        return Ok(entry);
    }
    catalog
        .entries()
        .iter()
        .find(|entry| {
            entry.kind == kind
                && (managed
                    || provider_prefix(&entry.id)
                        .ok()
                        .is_some_and(|provider| configured.contains(&provider)))
        })
        .ok_or_else(|| "no configured provider supports this generation type".to_string())
}

fn resolve_media(
    snapshot: &opentake_core::ProjectRuntimeSnapshot,
    media_ref: &str,
    expected_kind: ClipType,
) -> Result<PathBuf, String> {
    let entry = snapshot
        .media
        .entries
        .iter()
        .find(|entry| entry.id == media_ref)
        .ok_or_else(|| format!("referenced media does not exist: {media_ref}"))?;
    if entry.kind != expected_kind {
        return Err(format!("referenced media has the wrong type: {media_ref}"));
    }
    let path = MediaResolver::new(&snapshot.media, snapshot.project_dir.as_deref())
        .expected_path(media_ref)
        .ok_or_else(|| format!("referenced media cannot be resolved: {media_ref}"))?;
    if !path.is_file() {
        return Err(format!("referenced media is offline: {media_ref}"));
    }
    Ok(path)
}

fn validate_source_clip(
    timeline: &Timeline,
    clip_id: Option<&str>,
    media_ref: Option<&str>,
) -> Result<Option<(f64, f64)>, String> {
    let Some(clip_id) = clip_id else {
        return Ok(None);
    };
    let clip = timeline
        .tracks
        .iter()
        .flat_map(|track| &track.clips)
        .find(|clip| clip.id == clip_id)
        .ok_or_else(|| "sourceClipId does not exist".to_string())?;
    if media_ref.is_some_and(|media_ref| clip.media_ref != media_ref) {
        return Err("sourceClipId does not reference the requested media".to_string());
    }
    if timeline.fps <= 0 || clip.duration_frames <= 0 {
        return Err("sourceClipId has no valid visible source range".to_string());
    }
    let start = clip.trim_start_frame.max(0) as f64 / timeline.fps as f64;
    let consumed = clip.source_frames_consumed();
    if consumed <= 0 {
        return Err("sourceClipId has no valid visible source range".to_string());
    }
    Ok(Some((start, start + consumed as f64 / timeline.fps as f64)))
}

fn validate_choice(
    field: &str,
    value: Option<&str>,
    allowed: Option<&[String]>,
) -> Result<(), String> {
    if let (Some(value), Some(allowed)) = (value, allowed) {
        if !allowed.is_empty() && !allowed.iter().any(|candidate| candidate == value) {
            return Err(format!("{field} is not supported by the selected model"));
        }
    }
    Ok(())
}

fn validate_reference_count(label: &str, count: usize, max: u32) -> Result<(), String> {
    if count > max as usize {
        Err(format!(
            "too many {label} references for the selected model"
        ))
    } else {
        Ok(())
    }
}

fn provider_prefix(model: &str) -> Result<String, String> {
    ModelRoute::parse(model)
        .map(|route| route.prefix)
        .map_err(|_| "generation model id is invalid".to_string())
}

fn positive_dimension(value: Option<i32>) -> u32 {
    value
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(0)
}

/// Accept an upscale only at the size its request asked for. A
/// resolution-targeted upscaler renders the source aspect ratio at (at
/// least) the target short side; a job persisted without a target got the
/// provider default. Scale-factor upscales keep the exact 2x contract.
fn validate_upscale_result(
    media: &opentake_domain::MediaManifest,
    input: &GenerationInput,
    width: Option<u32>,
    height: Option<u32>,
) -> Result<(), String> {
    let resolution_targeted =
        Catalog::builtin()
            .by_id(&input.model)
            .is_some_and(|model| match &model.ui_capabilities {
                UiCapabilities::Upscale(caps) => caps
                    .target_resolutions
                    .as_ref()
                    .is_some_and(|targets| !targets.is_empty()),
                _ => false,
            });
    let source_id = input
        .source_asset_id
        .as_deref()
        .ok_or_else(|| "upscale source provenance is missing".to_string())?;
    let source = media
        .entries
        .iter()
        .find(|source| source.id == source_id)
        .ok_or_else(|| "upscale source disappeared".to_string())?;
    let (Some(source_width), Some(source_height), Some(width), Some(height)) = (
        source
            .source_width
            .and_then(|value| u32::try_from(value).ok()),
        source
            .source_height
            .and_then(|value| u32::try_from(value).ok()),
        width,
        height,
    ) else {
        return Ok(());
    };
    if source.kind == ClipType::Video && (input.resolution.is_some() || resolution_targeted) {
        let requested = input
            .resolution
            .as_deref()
            .unwrap_or(VIDEO_UPSCALE_DEFAULT_RESOLUTION);
        let target = video_upscale_resolution(requested)
            .ok_or_else(|| "upscale target resolution is not supported".to_string())?;
        if !upscale_result_matches(source_width, source_height, target, width, height) {
            return Err(format!(
                "upscale result {width}x{height} is not the requested {} size",
                target.label
            ));
        }
    } else if u64::from(width) != u64::from(source_width) * 2
        || u64::from(height) != u64::from(source_height) * 2
    {
        return Err("upscale result is not exactly 2x the source dimensions".to_string());
    }
    Ok(())
}

fn display_name(requested: Option<&str>, prompt: &str, fallback: &str) -> String {
    requested
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| {
            let value = prompt.trim().chars().take(30).collect::<String>();
            (!value.is_empty()).then_some(value)
        })
        .unwrap_or_else(|| fallback.to_string())
}

fn result_extension(
    media_type: &str,
    kind: ClipType,
    format_name: Option<&str>,
) -> Result<&'static str, String> {
    if let Some(format) = format_name {
        let formats = format.split(',').collect::<BTreeSet<_>>();
        let detected = match kind {
            ClipType::Image if formats.contains("png_pipe") => Some("png"),
            ClipType::Image if formats.contains("jpeg_pipe") || formats.contains("image2") => {
                Some("jpg")
            }
            ClipType::Image if formats.contains("webp_pipe") => Some("webp"),
            ClipType::Video if formats.contains("mov") || formats.contains("mp4") => Some("mp4"),
            ClipType::Audio if formats.contains("mp3") => Some("mp3"),
            ClipType::Audio if formats.contains("wav") => Some("wav"),
            ClipType::Audio if formats.contains("mov") || formats.contains("mp4") => Some("m4a"),
            _ => None,
        };
        if let Some(extension) = detected {
            return Ok(extension);
        }
    }
    match media_type {
        "image/png" => Ok("png"),
        "image/jpeg" | "image/jpg" => Ok("jpg"),
        "image/webp" => Ok("webp"),
        "video/mp4" => Ok("mp4"),
        "video/quicktime" => Ok("mov"),
        "audio/mpeg" => Ok("mp3"),
        "audio/wav" | "audio/x-wav" => Ok("wav"),
        "audio/mp4" => Ok("m4a"),
        "application/octet-stream" => match kind {
            ClipType::Image => Ok("png"),
            ClipType::Video => Ok("mp4"),
            ClipType::Audio => Ok("mp3"),
            ClipType::Text | ClipType::Lottie => {
                Err("generated media type is unsupported".to_string())
            }
        },
        _ => Err("generation result content type is unsupported".to_string()),
    }
}

fn validate_result_https_url(raw: &str) -> Result<reqwest::Url, String> {
    let url =
        reqwest::Url::parse(raw).map_err(|_| "generation result URL is invalid".to_string())?;
    if url.scheme() != "https"
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.port().is_some_and(|port| port != 443)
    {
        return Err("generation result URL is not an accepted HTTPS URL".to_string());
    }
    let host = url.host_str().unwrap_or_default().to_ascii_lowercase();
    if host == "localhost"
        || host.ends_with(".localhost")
        || host.ends_with(".local")
        || host.ends_with(".internal")
        || crate::public_net::literal_host_ip(&host)
            .is_some_and(|address| !crate::public_net::public_ip(address))
    {
        return Err("generation result URL host is not public".to_string());
    }
    Ok(url)
}

fn cancelled(cancel: &MediaCancelToken) -> Result<(), String> {
    if cancel.is_cancelled() {
        Err("GENERATION_CANCELLED".to_string())
    } else {
        Ok(())
    }
}

async fn wait_for_cancel(cancel: &MediaCancelToken) {
    while !cancel.is_cancelled() {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

fn now_apple_reference_seconds() -> f64 {
    const APPLE_REFERENCE_UNIX_OFFSET: f64 = 978_307_200.0;
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64() - APPLE_REFERENCE_UNIX_OFFSET)
        .unwrap_or(0.0)
}

/// Cancelling a job whose polling was interrupted persists its placeholders,
/// so this runs off the UI thread.
#[tauri::command]
pub async fn generation_cancel(
    bridge: tauri::State<'_, Arc<TauriGenerationBridge>>,
    job_id: String,
) -> Result<bool, String> {
    let bridge = Arc::clone(&bridge);
    tauri::async_runtime::spawn_blocking(move || bridge.cancel(&job_id))
        .await
        .map_err(|error| format!("generation cancel worker failed: {error}"))?
}

/// Whether a retry of `job_id` resumes its accepted provider job (no cost)
/// rather than submitting it again (after a cost confirmation).
#[tauri::command]
pub async fn generation_retry_resumes(
    bridge: tauri::State<'_, Arc<TauriGenerationBridge>>,
    job_id: String,
) -> Result<bool, String> {
    let bridge = Arc::clone(&bridge);
    tauri::async_runtime::spawn_blocking(move || bridge.retry_resumes(&job_id))
        .await
        .map_err(|error| format!("generation retry worker failed: {error}"))
}

/// Retrying commits new placeholders to the project (or writes the resumed
/// job), so it runs off the UI thread.
#[tauri::command]
pub async fn generation_retry(
    bridge: tauri::State<'_, Arc<TauriGenerationBridge>>,
    job_id: String,
    cost_authorized: bool,
) -> Result<GenerationSubmission, String> {
    let bridge = Arc::clone(&bridge);
    tauri::async_runtime::spawn_blocking(move || bridge.retry(&job_id, cost_authorized))
        .await
        .map_err(|error| format!("generation retry worker failed: {error}"))?
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    use image::{DynamicImage, ImageFormat};
    use opentake_agent::mcp::core_handle::AppCoreHandle;
    use opentake_agent::mcp::dispatch::Dispatcher;
    use opentake_agent::plugin::registry::PluginRegistry;
    use opentake_gen::{AuthMode, FalAdapter, HttpResponse, Method, MockTransport};
    use opentake_project::{GenerationLog, Project};
    use serde_json::json;
    use std::sync::RwLock;

    #[derive(Clone)]
    struct FixtureClients {
        client: GenClient,
    }

    impl GenerationClientFactory for FixtureClients {
        fn configured_byok_prefixes(&self) -> BTreeSet<String> {
            BTreeSet::from([
                "fal".to_string(),
                "openai".to_string(),
                "replicate".to_string(),
            ])
        }

        fn has_managed_credential(&self) -> bool {
            false
        }

        fn build(&self, _provider: &str, _managed: bool) -> Result<GenClient, String> {
            Ok(self.client.clone())
        }
    }

    fn fixture_client_with_interval(mock: &MockTransport, interval: Duration) -> GenClient {
        fixture_client_with_transport(mock, Arc::new(mock.clone())).with_poll_interval(interval)
    }

    fn fixture_client_with_transport(
        mock: &MockTransport,
        transport: Arc<dyn opentake_gen::HttpTransport>,
    ) -> GenClient {
        let fal = FalAdapter::new(transport.clone(), "fixture-secret").with_base("https://mockfal");
        let openai =
            OpenAiAdapter::new(transport.clone(), "fixture-secret").with_base("https://mockoai/v1");
        let replicate =
            ReplicateAdapter::new(transport, "fixture-secret").with_base("https://mockrep/v1");
        GenClient::with_transport(
            AuthMode::Byok {
                registry: ProviderRegistry::new()
                    .with(Arc::new(fal))
                    .with(Arc::new(openai))
                    .with(Arc::new(replicate)),
                catalog: Catalog::builtin(),
            },
            Arc::new(mock.clone()),
        )
    }

    /// Scripted poll failures retry at once instead of after seconds.
    fn quick_poll_policy(retry_budget: u32) -> opentake_gen::PollPolicy {
        opentake_gen::PollPolicy {
            interval: Duration::ZERO,
            retry_base: Duration::from_millis(1),
            retry_max: Duration::from_millis(2),
            retry_after_max: Duration::from_millis(2),
            retry_budget,
            poll_timeout: Duration::from_secs(5),
        }
    }

    fn quick_retry_client(mock: &MockTransport, retry_budget: u32) -> GenClient {
        fixture_client_with_transport(mock, Arc::new(mock.clone()))
            .with_poll_policy(quick_poll_policy(retry_budget))
    }

    /// Delays the listed requests before the mock answers, like a slow
    /// provider.
    /// Holds requests to one URL until the test releases them, so a test can
    /// act while a submission is in flight without depending on timing.
    struct GatedTransport {
        mock: MockTransport,
        gated_url: String,
        /// Set when the gated request has been sent.
        sent: Arc<std::sync::atomic::AtomicBool>,
        release: Arc<tokio::sync::Notify>,
    }

    impl GatedTransport {
        fn new(mock: &MockTransport, gated_url: &str) -> Arc<Self> {
            Arc::new(Self {
                mock: mock.clone(),
                gated_url: gated_url.to_string(),
                sent: Arc::default(),
                release: Arc::default(),
            })
        }

        fn is_sent(&self) -> bool {
            self.sent.load(std::sync::atomic::Ordering::SeqCst)
        }

        /// Let the held request (or the next one) answer.
        fn release(&self) {
            self.release.notify_one();
        }
    }

    // The trait is declared with `async_trait`; this is its expanded form.
    impl opentake_gen::HttpTransport for GatedTransport {
        fn send<'life0, 'async_trait>(
            &'life0 self,
            request: opentake_gen::HttpRequest,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Result<HttpResponse, GenError>>
                    + Send
                    + 'async_trait,
            >,
        >
        where
            'life0: 'async_trait,
            Self: 'async_trait,
        {
            Box::pin(async move {
                if request.url == self.gated_url {
                    self.sent.store(true, std::sync::atomic::Ordering::SeqCst);
                    self.release.notified().await;
                }
                self.mock.send(request).await
            })
        }
    }

    fn placeholder_input(core: &AppCore, asset_id: &str) -> GenerationInput {
        core.media()
            .entries
            .into_iter()
            .find(|entry| entry.id == asset_id)
            .and_then(|entry| entry.generation_input)
            .unwrap()
    }

    async fn wait_until(what: &str, mut ready: impl FnMut() -> bool) {
        // Generous for loaded CI runners; tests synchronize on events.
        let deadline = std::time::Instant::now() + Duration::from_secs(60);
        while !ready() {
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting: {what}"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    fn submit_fixture_image(bridge: &TauriGenerationBridge) -> GenerationSubmission {
        bridge
            .submit(
                GenerationRequest::Image(GenerateImageArgs {
                    cost_authorized: Some(true),
                    prompt: "network fixture".to_string(),
                    model: Some("fal:flux-pro".to_string()),
                    aspect_ratio: Some("1:1".to_string()),
                    num_images: Some(1),
                    ..Default::default()
                }),
                &MediaCancelToken::new(),
            )
            .unwrap()
    }

    const FLUX_SUBMIT: &str = "https://mockfal/fal-ai/flux-pro/v1.1";

    fn fixture_client(mock: &MockTransport) -> GenClient {
        fixture_client_with_interval(mock, Duration::ZERO)
    }

    fn saved_core() -> (tempfile::TempDir, PathBuf, AppCore) {
        let temp = tempfile::tempdir().unwrap();
        let bundle = temp.path().join("Generation.opentake");
        let mut project = Project::new(&bundle);
        project.generation_log = Some(GenerationLog::new());
        project.save().unwrap();
        let core = AppCore::new();
        core.open_project(&bundle).unwrap();
        (temp, bundle, core)
    }

    fn image_plan() -> PreparedGenerationJob {
        PreparedGenerationJob {
            name: "Recovered image".to_string(),
            kind: ClipType::Image,
            folder_id: None,
            provider: "fal".to_string(),
            input: GenerationInput {
                prompt: "fixture".to_string(),
                model: "fal:flux-pro".to_string(),
                duration: 0,
                aspect_ratio: "1:1".to_string(),
                num_images: Some(1),
                ..Default::default()
            },
            output_count: 1,
            source_asset_id: None,
            source_clip_id: None,
            estimated_cost_credits: None,
            created_at: Some(800_000_000.0),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_update_detaches_a_paid_submission_and_installs_only_after_it_is_kept() {
        let (_temp, bundle, core) = saved_core();
        let mock = moving_job_mock();
        let transport = GatedTransport::new(&mock, FLUX_SUBMIT);
        let admission = crate::updater::InstallAdmissionGate::default();
        let bridge = following_bridge(&core, &bundle, transport.clone(), &mock, admission.clone());
        let submitted = submit_fixture_image(&bridge);
        let asset_id = submitted.placeholder_asset_ids[0].clone();
        wait_until("the submission to be sent", || transport.is_sent()).await;

        // The update detaches the job instead of cancelling it, and cannot
        // install while the paid submission still waits for its answer.
        assert!(bridge.has_active());
        assert_eq!(bridge.detach_all_active(), 1);
        assert!(bridge.has_active());
        assert!(admission.begin_install().is_err());
        transport.release();
        wait_until("the detached task to exit", || !bridge.has_active()).await;
        drop(admission.begin_install().unwrap());
        assert_eq!(bridge.detach_all_active(), 0);
        let orphan = bridge.orphans.get(&submitted.job_id).unwrap().unwrap();
        assert_eq!(orphan.provider_job_id.as_deref(), Some(MOVING_PROVIDER_JOB));
        let input = on_disk_input(&bundle, &asset_id);
        assert_ne!(input.status, Some(GenerationJobStatus::Cancelled));
        assert_eq!(input.error_code, None);

        // The update installs and the app restarts.
        drop(bridge);
        drop(core);
        mock.on(
            Method::Get,
            PROJECT_STATUS,
            200,
            json!({"status": "COMPLETED"}),
        );
        let restarted = AppCore::new();
        let bridge = following_bridge(
            &restarted,
            &bundle,
            Arc::new(mock.clone()),
            &mock,
            crate::updater::InstallAdmissionGate::default(),
        );
        restarted.open_project(&bundle).unwrap();
        bridge.recover_current_project();
        let ready = wait_for_ready_model(&restarted, "fal:flux-pro").await;
        assert_eq!(ready.id, asset_id);
        assert_eq!(count_calls(&mock, FLUX_SUBMIT), 1, "never resubmitted");
        assert_eq!(bridge.orphans.get(&submitted.job_id).unwrap(), None);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn jobs_detached_for_an_update_that_did_not_install_resume() {
        let (_temp, bundle, core) = saved_core();
        let mock = moving_job_mock();
        let (cache, models) = runtime_dirs(&bundle);
        let client = fixture_client(&mock).with_poll_policy(opentake_gen::PollPolicy {
            interval: Duration::from_millis(5),
            ..quick_poll_policy(20)
        });
        let bridge = build_bridge_with_timings(
            core.clone(),
            cache,
            models,
            Arc::new(FixtureClients { client }),
            crate::updater::InstallAdmissionGate::default(),
            GenerationTimings {
                resume_delay: Duration::from_millis(300),
                ..GenerationTimings::default()
            },
        );
        let submitted = submit_fixture_image(&bridge);
        let asset_id = submitted.placeholder_asset_ids[0].clone();
        wait_until("the job to be polled", || {
            count_calls(&mock, PROJECT_STATUS) > 0
        })
        .await;

        assert_eq!(bridge.detach_all_active(), 1);
        wait_until("the detached task to exit", || !bridge.has_active()).await;
        let input = on_disk_input(&bundle, &asset_id);
        assert_eq!(input.status, Some(GenerationJobStatus::Generating));
        assert_eq!(input.provider_job_id.as_deref(), Some(MOVING_PROVIDER_JOB));

        // The update did not install: the open project resumes the job.
        mock.on(
            Method::Get,
            PROJECT_STATUS,
            200,
            json!({"status": "COMPLETED"}),
        );
        let ready = wait_for_ready_model(&core, "fal:flux-pro").await;
        assert_eq!(ready.id, asset_id);
        assert_eq!(count_calls(&mock, FLUX_SUBMIT), 1, "never resubmitted");
    }

    #[test]
    fn generation_cannot_submit_after_update_install_claims_admission() {
        let (_temp, bundle, core) = saved_core();
        let mock = MockTransport::new();
        let (cache, models) = runtime_dirs(&bundle);
        let admission = crate::updater::InstallAdmissionGate::default();
        let bridge = build_bridge_with_clients_and_admission(
            core,
            cache,
            models,
            Arc::new(FixtureClients {
                client: fixture_client(&mock),
            }),
            admission.clone(),
        );
        let _install = admission.begin_install().unwrap();

        assert_eq!(
            bridge
                .submit(
                    GenerationRequest::Image(GenerateImageArgs {
                        prompt: "must not submit".to_string(),
                        ..Default::default()
                    }),
                    &MediaCancelToken::new(),
                )
                .unwrap_err(),
            "app update installation is in progress"
        );
        assert!(bridge.runtime.jobs.lock().unwrap().is_empty());
    }

    fn runtime_dirs(bundle: &Path) -> (PathBuf, PathBuf) {
        let root = bundle.parent().unwrap();
        (root.join("cache"), root.join("models"))
    }

    fn png_bytes(width: u32, height: u32) -> Vec<u8> {
        let mut bytes = Cursor::new(Vec::new());
        DynamicImage::new_rgba8(width, height)
            .write_to(&mut bytes, ImageFormat::Png)
            .unwrap();
        bytes.into_inner()
    }

    fn png_data_url_for(width: u32, height: u32) -> String {
        format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(png_bytes(width, height))
        )
    }

    fn png_data_url() -> String {
        png_data_url_for(2, 2)
    }

    fn wav_bytes() -> Vec<u8> {
        let sample_rate = 8_000_u32;
        let sample_count = sample_rate / 10;
        let data_size = sample_count * 2;
        let mut bytes = Vec::with_capacity((44 + data_size) as usize);
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_size).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_size.to_le_bytes());
        bytes.resize((44 + data_size) as usize, 0);
        bytes
    }

    fn mp4_bytes(directory: &Path, width: u32, height: u32) -> Vec<u8> {
        let path = directory.join(format!("generation-fixture-{width}x{height}.mp4"));
        let status = std::process::Command::new("ffmpeg")
            .args(["-loglevel", "error", "-y", "-f", "lavfi", "-i"])
            .arg(format!("color=c=black:s={width}x{height}:d=0.1:r=10"))
            .args(["-pix_fmt", "yuv420p"])
            .arg(&path)
            .status()
            .unwrap();
        assert!(status.success());
        std::fs::read(path).unwrap()
    }

    fn mp4_data_url_for(directory: &Path, width: u32, height: u32) -> String {
        let encoded =
            base64::engine::general_purpose::STANDARD.encode(mp4_bytes(directory, width, height));
        format!("data:video/mp4;base64,{encoded}")
    }

    fn mp4_data_url(directory: &Path) -> String {
        mp4_data_url_for(directory, 16, 16)
    }

    fn saved_core_with_source_video() -> (tempfile::TempDir, PathBuf, AppCore) {
        saved_core_with_source_videos(&[("source-video", 16, 16, 10.0)])
    }

    /// Source videos are tiny real files; the manifest carries the probed
    /// size and frame rate that upscale planning reads.
    fn saved_core_with_source_videos(
        sources: &[(&str, i32, i32, f64)],
    ) -> (tempfile::TempDir, PathBuf, AppCore) {
        let temp = tempfile::tempdir().unwrap();
        let bundle = temp.path().join("Upscale.opentake");
        let mut project = Project::new(&bundle);
        project.generation_log = Some(GenerationLog::new());
        std::fs::create_dir_all(bundle.join("media")).unwrap();
        for (id, width, height, fps) in sources {
            project
                .manifest
                .entries
                .push(opentake_domain::MediaManifestEntry {
                    id: id.to_string(),
                    name: format!("{id}.mp4"),
                    kind: ClipType::Video,
                    source: opentake_domain::MediaSource::Project {
                        relative_path: format!("media/{id}.mp4"),
                    },
                    duration: 0.1,
                    generation_input: None,
                    source_width: Some(*width),
                    source_height: Some(*height),
                    source_fps: Some(*fps),
                    has_audio: Some(false),
                    color: None,
                    proxy: None,
                    folder_id: None,
                    cached_remote_url: None,
                    cached_remote_url_expires_at: None,
                });
        }
        project.save().unwrap();
        let fixture = mp4_bytes(temp.path(), 16, 16);
        for (id, ..) in sources {
            std::fs::write(bundle.join(format!("media/{id}.mp4")), &fixture).unwrap();
        }
        let core = AppCore::new();
        core.open_project(&bundle).unwrap();
        (temp, bundle, core)
    }

    async fn wait_for_ready_model(
        core: &AppCore,
        model: &str,
    ) -> opentake_domain::MediaManifestEntry {
        // Generous for loaded CI runners (probing and fsyncs are slow there).
        let deadline = std::time::Instant::now() + Duration::from_secs(60);
        while std::time::Instant::now() < deadline {
            if let Some(entry) = core.media().entries.into_iter().find(|entry| {
                entry.generation_input.as_ref().is_some_and(|input| {
                    input.model == model && input.status == Some(GenerationJobStatus::Ready)
                })
            }) {
                return entry;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!(
            "generation did not become ready for {model}: {:?}",
            core.media().entries
        );
    }

    #[test]
    fn result_url_validation_rejects_local_and_private_network_targets() {
        assert!(validate_result_https_url("https://cdn.example.com/result.png").is_ok());
        for url in [
            "https://localhost/result.png",
            "https://service.local/result.png",
            "https://127.0.0.1/result.png",
            "https://10.0.0.1/result.png",
            "https://169.254.1.1/result.png",
            "https://[::1]/result.png",
            "https://[fc00::1]/result.png",
            "https://[::ffff:127.0.0.1]/result.png",
            "https://[2001:db8::1]/result.png",
        ] {
            assert!(
                validate_result_https_url(url).is_err(),
                "private result URL accepted: {url}"
            );
        }
    }

    #[test]
    fn restart_before_provider_id_requires_explicit_retry_without_resubmission() {
        let (_temp, bundle, core) = saved_core();
        let committed = core
            .begin_generation_job_for_project(1, &bundle, image_plan())
            .unwrap();
        // Model a real process restart. A second independent AppCore retaining
        // the old bundle at the same time is not a supported runtime state and
        // prevents same-target directory publication on Windows.
        drop(core);
        let reopened = AppCore::new();
        reopened.open_project(&bundle).unwrap();
        let mock = MockTransport::new();
        let (cache, models) = runtime_dirs(&bundle);
        let bridge = build_bridge_with_clients(
            reopened.clone(),
            cache,
            models,
            Arc::new(FixtureClients {
                client: fixture_client(&mock),
            }),
        );

        assert_eq!(bridge.recover_current_project(), 0);
        let persisted = reopened.media();
        let input = persisted
            .entries
            .iter()
            .find(|entry| entry.id == committed.placeholder_asset_ids[0])
            .unwrap()
            .generation_input
            .as_ref()
            .unwrap();
        assert_eq!(input.status, Some(GenerationJobStatus::Failed));
        assert_eq!(
            input.error_code.as_deref(),
            Some("GENERATION_RESTART_RETRY_REQUIRED")
        );
        assert!(mock.calls().is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn restart_with_provider_id_resumes_and_finalizes_offline_fixture() {
        let (_temp, bundle, core) = saved_core();
        let runtime = core.runtime_snapshot();
        let committed = core
            .begin_generation_job_for_project(runtime.project_epoch, &bundle, image_plan())
            .unwrap();
        core.update_generation_job_for_project(
            runtime.project_epoch,
            &bundle,
            &committed.job_id,
            GenerationStateUpdate {
                status: GenerationJobStatus::Generating,
                progress: Some(0.25),
                error_code: None,
                provider_job_id: Some("fal::flux-pro|recover-1".to_string()),
                cost_credits: None,
                created_at: Some(800_000_001.0),
            },
        )
        .unwrap();

        // Model a real process restart before opening the same bundle again.
        drop(core);
        let reopened = AppCore::new();
        let reopened_snapshot = reopened.open_project(&bundle).unwrap();
        let mock = MockTransport::new();
        mock.on(
            Method::Get,
            "https://mockfal/flux-pro/requests/recover-1/status",
            200,
            json!({"status": "COMPLETED"}),
        );
        mock.on(
            Method::Get,
            "https://mockfal/flux-pro/requests/recover-1",
            200,
            json!({"images": [{"url": png_data_url()}]}),
        );
        let (cache, models) = runtime_dirs(&bundle);
        let bridge = build_bridge_with_clients(
            reopened.clone(),
            cache,
            models,
            Arc::new(FixtureClients {
                client: fixture_client(&mock),
            }),
        );

        assert_eq!(bridge.recover_current_project(), 1);
        let asset_id = committed.placeholder_asset_ids[0].clone();
        for _ in 0..100 {
            let status = reopened
                .media()
                .entries
                .iter()
                .find(|entry| entry.id == asset_id)
                .and_then(|entry| entry.generation_input.as_ref())
                .and_then(|input| input.status);
            if status == Some(GenerationJobStatus::Ready) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let entry = reopened
            .media()
            .entries
            .into_iter()
            .find(|entry| entry.id == asset_id)
            .unwrap();
        assert_eq!(
            entry
                .generation_input
                .as_ref()
                .and_then(|input| input.status),
            Some(GenerationJobStatus::Ready)
        );
        assert_eq!(entry.source_width, Some(2));
        assert_eq!(entry.source_height, Some(2));
        assert!(MediaResolver::new(&reopened.media(), Some(&bundle))
            .expected_path(&entry.id)
            .unwrap()
            .is_file());
        assert_eq!(
            reopened.project_revision().project_epoch,
            reopened_snapshot.project_epoch
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn production_dispatch_path_persists_and_finalizes_ordered_mock_results() {
        let (_temp, bundle, core) = saved_core();
        let mock = MockTransport::new();
        mock.on(
            Method::Post,
            "https://mockfal/fal-ai/flux-pro/v1.1",
            200,
            json!({"request_id": "dispatch-1", "status": "IN_QUEUE"}),
        );
        mock.on(
            Method::Get,
            "https://mockfal/fal-ai/flux-pro/requests/dispatch-1/status",
            200,
            json!({"status": "COMPLETED"}),
        );
        mock.on(
            Method::Get,
            "https://mockfal/fal-ai/flux-pro/requests/dispatch-1",
            200,
            json!({"images": [{"url": png_data_url()}, {"url": png_data_url()}]}),
        );
        let (cache, models) = runtime_dirs(&bundle);
        let bridge = build_bridge_with_clients(
            core.clone(),
            cache,
            models,
            Arc::new(FixtureClients {
                client: fixture_client(&mock),
            }),
        );
        let dispatcher = Dispatcher::with_bridges(
            Arc::new(AppCoreHandle::new(core.clone())),
            Arc::new(RwLock::new(PluginRegistry::new())),
            None,
            Some(bridge),
        );

        let unauthorized = dispatcher.dispatch(
            "generate_image",
            json!({
                "costAuthorized": false,
                "prompt": "ordered fixture",
                "model": "fal:flux-pro",
                "numImages": 2
            }),
        );
        assert!(unauthorized.is_error);
        assert!(mock.calls().is_empty());

        let accepted = dispatcher.dispatch(
            "generate_image",
            json!({
                "costAuthorized": true,
                "prompt": "ordered fixture",
                "model": "fal:flux-pro",
                "aspectRatio": "1:1",
                "numImages": 2
            }),
        );
        assert!(!accepted.is_error, "{}", accepted.text_joined());
        for _ in 0..100 {
            let generated = core
                .media()
                .entries
                .into_iter()
                .filter(|entry| entry.generation_input.is_some())
                .collect::<Vec<_>>();
            if generated.len() == 2
                && generated.iter().all(|entry| {
                    entry
                        .generation_input
                        .as_ref()
                        .and_then(|input| input.status)
                        == Some(GenerationJobStatus::Ready)
                })
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let mut generated = core
            .media()
            .entries
            .into_iter()
            .filter(|entry| entry.generation_input.is_some())
            .collect::<Vec<_>>();
        generated.sort_by_key(|entry| {
            entry
                .generation_input
                .as_ref()
                .and_then(|input| input.output_index)
        });
        assert_eq!(generated.len(), 2);
        assert!(
            generated.iter().all(|entry| {
                entry
                    .generation_input
                    .as_ref()
                    .and_then(|input| input.status)
                    == Some(GenerationJobStatus::Ready)
                    && entry.source_width == Some(2)
                    && entry.source_height == Some(2)
            }),
            "generated outputs: {generated:?}"
        );
        assert_eq!(
            generated
                .iter()
                .filter_map(|entry| entry
                    .generation_input
                    .as_ref()
                    .and_then(|input| input.output_index))
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert!(generated.iter().all(|entry| {
            MediaResolver::new(&core.media(), Some(&bundle))
                .expected_path(&entry.id)
                .is_some_and(|path| path.is_file())
        }));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn fifty_running_polls_never_republish_the_bundle_before_ready() {
        let (_temp, bundle, core) = saved_core();
        let held = opentake_project::ProjectRoot::open(&bundle).unwrap();
        let authority = core.project_asset_authority().unwrap();
        let siblings = |bundle: &Path| {
            let prefix = format!(".{}", bundle.file_name().unwrap().to_string_lossy());
            let mut names = std::fs::read_dir(bundle.parent().unwrap())
                .unwrap()
                .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
                .filter(|name| name.starts_with(&prefix))
                .collect::<Vec<_>>();
            names.sort();
            names
        };
        let siblings_before = siblings(&bundle);
        let mock = MockTransport::new();
        mock.on(
            Method::Post,
            "https://mockfal/fal-ai/flux-pro/v1.1",
            200,
            json!({"request_id": "poll-1", "status": "IN_QUEUE"}),
        );
        let mut polls = vec![(200, json!({"status": "IN_PROGRESS"})); 50];
        polls.push((200, json!({"status": "COMPLETED"})));
        mock.on_sequence(
            Method::Get,
            "https://mockfal/fal-ai/flux-pro/requests/poll-1/status",
            polls,
        );
        mock.on(
            Method::Get,
            "https://mockfal/fal-ai/flux-pro/requests/poll-1",
            200,
            json!({"images": [{"url": png_data_url()}]}),
        );
        let (cache, models) = runtime_dirs(&bundle);
        let bridge = build_bridge_with_clients(
            core.clone(),
            cache,
            models,
            Arc::new(FixtureClients {
                client: fixture_client(&mock),
            }),
        );

        let submitted = bridge
            .submit(
                GenerationRequest::Image(GenerateImageArgs {
                    cost_authorized: Some(true),
                    prompt: "polled fixture".to_string(),
                    model: Some("fal:flux-pro".to_string()),
                    aspect_ratio: Some("1:1".to_string()),
                    num_images: Some(1),
                    ..Default::default()
                }),
                &MediaCancelToken::new(),
            )
            .unwrap();
        let ready = wait_for_ready_model(&core, "fal:flux-pro").await;

        assert_eq!(ready.id, submitted.placeholder_asset_ids[0]);
        let status_polls = mock
            .calls()
            .iter()
            .filter(|call| call.url.ends_with("/requests/poll-1/status"))
            .count();
        assert!(status_polls >= 51, "{status_polls} status polls");
        // Zero complete-bundle publications: the root keeps its identity and
        // no stage/backup/journal sibling ever appeared.
        assert_eq!(core.project_asset_authority().unwrap(), authority);
        core.ensure_project_root_identity_for_project(
            authority.project_epoch,
            &bundle,
            held.identity(),
        )
        .unwrap();
        assert_eq!(siblings(&bundle), siblings_before);
        // Queued, Generating (provider id), Downloading, Ready: the 50
        // Running polls add no audit rows.
        let persisted = Project::open(&bundle).unwrap();
        assert_eq!(persisted.generation_log.unwrap().entries.len(), 4);
        assert_eq!(
            persisted
                .manifest
                .entries
                .iter()
                .find(|entry| entry.id == ready.id)
                .and_then(|entry| entry.generation_input.as_ref())
                .and_then(|input| input.status),
            Some(GenerationJobStatus::Ready)
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn configured_provider_smoke_covers_video_audio_and_upscale() {
        let (video_temp, video_bundle, video_core) = saved_core();
        let video_mock = MockTransport::new();
        video_mock.on(
            Method::Post,
            "https://mockfal/fal-ai/kling-video/v2.5-turbo/pro/text-to-video",
            200,
            json!({"request_id": "video-1", "status": "IN_QUEUE"}),
        );
        video_mock.on(
            Method::Get,
            "https://mockfal/fal-ai/kling-video/requests/video-1/status",
            200,
            json!({"status": "COMPLETED"}),
        );
        video_mock.on(
            Method::Get,
            "https://mockfal/fal-ai/kling-video/requests/video-1",
            200,
            json!({"video": {"url": mp4_data_url(video_temp.path())}}),
        );
        let (cache, models) = runtime_dirs(&video_bundle);
        let video_bridge = build_bridge_with_clients(
            video_core.clone(),
            cache,
            models,
            Arc::new(FixtureClients {
                client: fixture_client(&video_mock),
            }),
        );
        let video_dispatcher = Dispatcher::with_bridges(
            Arc::new(AppCoreHandle::new(video_core.clone())),
            Arc::new(RwLock::new(PluginRegistry::new())),
            None,
            Some(video_bridge),
        );
        let video_result = video_dispatcher.dispatch(
            "generate_video",
            json!({
                "costAuthorized": true,
                "prompt": "fixture video",
                "model": "fal:kling-video",
                "duration": 5,
                "aspectRatio": "16:9",
                "resolution": "720p"
            }),
        );
        assert!(!video_result.is_error, "{}", video_result.text_joined());
        let video = wait_for_ready_model(&video_core, "fal:kling-video").await;
        assert_eq!(video.kind, ClipType::Video);
        assert_eq!(video.source_width, Some(16));
        assert_eq!(video.source_height, Some(16));

        let (_audio_temp, audio_bundle, audio_core) = saved_core();
        let audio_mock = MockTransport::new();
        let mut audio_response = HttpResponse::new(200, wav_bytes());
        audio_response
            .headers
            .push(("Content-Type".to_string(), "audio/wav".to_string()));
        audio_mock.on_raw(
            Method::Post,
            "https://mockoai/v1/audio/speech",
            audio_response,
        );
        let (cache, models) = runtime_dirs(&audio_bundle);
        let audio_bridge = build_bridge_with_clients(
            audio_core.clone(),
            cache,
            models,
            Arc::new(FixtureClients {
                client: fixture_client(&audio_mock),
            }),
        );
        let audio_dispatcher = Dispatcher::with_bridges(
            Arc::new(AppCoreHandle::new(audio_core.clone())),
            Arc::new(RwLock::new(PluginRegistry::new())),
            None,
            Some(audio_bridge),
        );
        let audio_result = audio_dispatcher.dispatch(
            "generate_audio",
            json!({
                "costAuthorized": true,
                "prompt": "fixture speech",
                "model": "openai:tts-1",
                "voice": "alloy"
            }),
        );
        assert!(!audio_result.is_error, "{}", audio_result.text_joined());
        let audio = wait_for_ready_model(&audio_core, "openai:tts-1").await;
        assert_eq!(audio.kind, ClipType::Audio);
        assert_eq!(audio.has_audio, Some(true));
        assert!(audio.duration > 0.0);

        // The catalog's upscaler only accepts video and renders a target
        // resolution: a 16x16 source asks for 720p, so the result is 720x720.
        let (upscale_temp, upscale_bundle, upscale_core) = saved_core_with_source_video();
        let source_before = std::fs::read(upscale_bundle.join("media/source-video.mp4")).unwrap();
        let upscale_mock = MockTransport::new();
        upscale_mock.on(
            Method::Post,
            "https://mockrep/v1/files",
            200,
            json!({"urls": {"get": "https://fixtures.invalid/source.mp4"}}),
        );
        upscale_mock.on(
            Method::Post,
            "https://mockrep/v1/models/topazlabs/video-upscale/predictions",
            200,
            json!({"id": "upscale-1", "status": "starting"}),
        );
        upscale_mock.on(
            Method::Get,
            "https://mockrep/v1/predictions/upscale-1",
            200,
            json!({
                "id": "upscale-1",
                "status": "succeeded",
                "output": mp4_data_url_for(upscale_temp.path(), 720, 720)
            }),
        );
        let (cache, models) = runtime_dirs(&upscale_bundle);
        let upscale_bridge = build_bridge_with_clients(
            upscale_core.clone(),
            cache,
            models,
            Arc::new(FixtureClients {
                client: fixture_client(&upscale_mock),
            }),
        );
        let upscale_dispatcher = Dispatcher::with_bridges(
            Arc::new(AppCoreHandle::new(upscale_core.clone())),
            Arc::new(RwLock::new(PluginRegistry::new())),
            None,
            Some(upscale_bridge),
        );
        let upscale_result = upscale_dispatcher.dispatch(
            "upscale_media",
            json!({
                "costAuthorized": true,
                "mediaRef": "source-video",
                "model": "replicate:topaz-upscale"
            }),
        );
        assert!(!upscale_result.is_error, "{}", upscale_result.text_joined());
        let upscale = wait_for_ready_model(&upscale_core, "replicate:topaz-upscale").await;
        assert_eq!(upscale.kind, ClipType::Video);
        assert_eq!(upscale.name, "source-video.mp4 720p");
        assert_eq!(upscale.source_width, Some(720));
        assert_eq!(upscale.source_height, Some(720));
        assert_eq!(
            std::fs::read(upscale_bundle.join("media/source-video.mp4")).unwrap(),
            source_before
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn production_dispatch_cancel_terminalizes_without_importing_media() {
        let (_temp, bundle, core) = saved_core();
        let mock = MockTransport::new();
        mock.on(
            Method::Post,
            "https://mockfal/fal-ai/flux-pro/v1.1",
            200,
            json!({"request_id": "cancel-1", "status": "IN_QUEUE"}),
        );
        mock.on(
            Method::Get,
            "https://mockfal/fal-ai/flux-pro/requests/cancel-1/status",
            200,
            json!({"status": "IN_QUEUE"}),
        );
        let (cache, models) = runtime_dirs(&bundle);
        let bridge = build_bridge_with_clients(
            core.clone(),
            cache,
            models,
            Arc::new(FixtureClients {
                client: fixture_client_with_interval(&mock, Duration::from_secs(2)),
            }),
        );
        let dispatcher = Dispatcher::with_bridges(
            Arc::new(AppCoreHandle::new(core.clone())),
            Arc::new(RwLock::new(PluginRegistry::new())),
            None,
            Some(bridge.clone()),
        );
        let accepted = dispatcher.dispatch(
            "generate_image",
            json!({
                "costAuthorized": true,
                "prompt": "cancel fixture",
                "model": "fal:flux-pro",
                "aspectRatio": "1:1"
            }),
        );
        assert!(!accepted.is_error, "{}", accepted.text_joined());
        // Bounded, so a submission failure fails the test instead of hanging it.
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let (job_id, asset_id) = loop {
            if let Some(entry) = core
                .media()
                .entries
                .into_iter()
                .find(|entry| entry.generation_input.is_some())
            {
                let input = entry.generation_input.as_ref().unwrap();
                if input.provider_job_id.is_some() {
                    break (input.job_id.clone().unwrap(), entry.id);
                }
            }
            assert!(
                std::time::Instant::now() < deadline,
                "generation job never reached the provider: {:?}",
                core.media().entries
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        };
        assert!(bridge.cancel(&job_id).unwrap());
        for _ in 0..100 {
            let status = core
                .media()
                .entries
                .iter()
                .find(|entry| entry.id == asset_id)
                .and_then(|entry| entry.generation_input.as_ref())
                .and_then(|input| input.status);
            if status == Some(GenerationJobStatus::Cancelled) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let entry = core
            .media()
            .entries
            .into_iter()
            .find(|entry| entry.id == asset_id)
            .unwrap();
        assert_eq!(
            entry
                .generation_input
                .as_ref()
                .and_then(|input| input.status),
            Some(GenerationJobStatus::Cancelled)
        );
        assert!(MediaResolver::new(&core.media(), Some(&bundle))
            .expected_path(&entry.id)
            .is_some_and(|path| !path.exists()));
    }

    #[test]
    fn upscale_finalization_is_exactly_two_x_and_preserves_source_bytes() {
        let temp = tempfile::tempdir().unwrap();
        let bundle = temp.path().join("Upscale.opentake");
        let source_bytes = png_bytes(3, 2);
        let mut project = Project::new(&bundle);
        project
            .manifest
            .entries
            .push(opentake_domain::MediaManifestEntry {
                id: "source-image".to_string(),
                name: "source.png".to_string(),
                kind: ClipType::Image,
                source: opentake_domain::MediaSource::Project {
                    relative_path: "media/source.png".to_string(),
                },
                duration: 0.0,
                generation_input: None,
                source_width: Some(3),
                source_height: Some(2),
                source_fps: None,
                has_audio: Some(false),
                color: None,
                proxy: None,
                folder_id: None,
                cached_remote_url: None,
                cached_remote_url_expires_at: None,
            });
        project.save().unwrap();
        std::fs::create_dir_all(bundle.join("media")).unwrap();
        std::fs::write(bundle.join("media/source.png"), &source_bytes).unwrap();
        let core = AppCore::new();
        let snapshot = core.open_project(&bundle).unwrap();
        let mock = MockTransport::new();
        let (cache, models) = runtime_dirs(&bundle);
        let bridge = build_bridge_with_clients(
            core.clone(),
            cache.clone(),
            models,
            Arc::new(FixtureClients {
                client: fixture_client(&mock),
            }),
        );
        let mut plan = image_plan();
        plan.provider = "replicate".to_string();
        plan.input.model = "replicate:topaz-upscale".to_string();
        plan.source_asset_id = Some("source-image".to_string());
        let committed = core
            .begin_generation_job_for_project(snapshot.project_epoch, &bundle, plan)
            .unwrap();
        core.update_generation_job_for_project(
            snapshot.project_epoch,
            &bundle,
            &committed.job_id,
            GenerationStateUpdate {
                status: GenerationJobStatus::Generating,
                progress: Some(0.5),
                error_code: None,
                provider_job_id: Some("replicate::fixture".to_string()),
                cost_credits: None,
                created_at: None,
            },
        )
        .unwrap();
        core.update_generation_job_for_project(
            snapshot.project_epoch,
            &bundle,
            &committed.job_id,
            GenerationStateUpdate {
                status: GenerationJobStatus::Downloading,
                progress: Some(0.8),
                error_code: None,
                provider_job_id: None,
                cost_credits: None,
                created_at: None,
            },
        )
        .unwrap();
        std::fs::create_dir_all(&cache).unwrap();
        let staged = cache.join("upscale.png");
        let staged_bytes = png_bytes(6, 4);
        std::fs::write(&staged, &staged_bytes).unwrap();
        let store = TauriFinalizationStore {
            bridge: bridge.as_ref().clone(),
            binding: JobBinding::new(snapshot.project_epoch, bundle.clone()),
            lease: Mutex::new(None),
        };
        store
            .finalize_output(
                &committed.placeholder_asset_ids[0],
                DownloadedGenerationArtifact {
                    path: staged,
                    media_type: "image/png".to_string(),
                    byte_size: staged_bytes.len() as u64,
                },
            )
            .unwrap();
        let media = core.media();
        let source = media
            .entries
            .iter()
            .find(|entry| entry.id == "source-image")
            .unwrap();
        let output = media
            .entries
            .iter()
            .find(|entry| entry.id == committed.placeholder_asset_ids[0])
            .unwrap();
        assert_eq!(
            (source.source_width, source.source_height),
            (Some(3), Some(2))
        );
        assert_eq!(
            (output.source_width, output.source_height),
            (Some(6), Some(4))
        );
        assert_eq!(
            std::fs::read(bundle.join("media/source.png")).unwrap(),
            source_bytes
        );
    }

    async fn wait_for_job_status(
        core: &AppCore,
        job_id: &str,
        expected: GenerationJobStatus,
    ) -> opentake_domain::GenerationInput {
        let deadline = std::time::Instant::now() + Duration::from_secs(60);
        loop {
            if let Some(input) = core
                .media()
                .entries
                .into_iter()
                .filter_map(|entry| entry.generation_input)
                .find(|input| input.job_id.as_deref() == Some(job_id))
                .filter(|input| input.status == Some(expected))
            {
                return input;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "job {job_id} never reached {expected:?}: {:?}",
                core.media().entries
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn topaz_upscale_requests_the_next_target_at_the_source_frame_rate() {
        let (_temp, bundle, core) = saved_core_with_source_videos(&[
            ("hd-24", 1280, 720, 24.0),
            ("full-hd-60", 1920, 1080, 60.0),
        ]);
        let mock = MockTransport::new();
        mock.on(
            Method::Post,
            "https://mockrep/v1/files",
            200,
            json!({"urls": {"get": "https://fixtures.invalid/source.mp4"}}),
        );
        mock.on_sequence(
            Method::Post,
            "https://mockrep/v1/models/topazlabs/video-upscale/predictions",
            vec![
                (201, json!({"id": "upscale-hd", "status": "starting"})),
                (201, json!({"id": "upscale-full-hd", "status": "starting"})),
            ],
        );
        for id in ["upscale-hd", "upscale-full-hd"] {
            mock.on(
                Method::Get,
                format!("https://mockrep/v1/predictions/{id}"),
                200,
                json!({"id": id, "status": "failed", "error": "fixture stop"}),
            );
        }
        let (cache, models) = runtime_dirs(&bundle);
        let bridge = build_bridge_with_clients(
            core.clone(),
            cache,
            models,
            Arc::new(FixtureClients {
                client: fixture_client(&mock),
            }),
        );
        let mut placeholders = Vec::new();
        for source in ["hd-24", "full-hd-60"] {
            let submitted = bridge
                .submit(
                    GenerationRequest::Upscale(UpscaleMediaArgs {
                        cost_authorized: Some(true),
                        media_ref: source.to_string(),
                        model: Some("replicate:topaz-upscale".to_string()),
                        source_clip_id: None,
                    }),
                    &MediaCancelToken::new(),
                )
                .unwrap();
            assert!(submitted.warnings.is_empty(), "{:?}", submitted.warnings);
            let failed =
                wait_for_job_status(&core, &submitted.job_id, GenerationJobStatus::Failed).await;
            assert_eq!(
                failed.error_code.as_deref(),
                Some("GENERATION_PROVIDER_FAILED")
            );
            placeholders.push(submitted.placeholder_asset_ids[0].clone());
        }
        let inputs = mock
            .calls()
            .into_iter()
            .filter(|call| call.url.ends_with("/topazlabs/video-upscale/predictions"))
            .map(|call| match call.body {
                opentake_gen::Body::Json(body) => body["input"].clone(),
                other => panic!("expected a JSON prediction, got {other:?}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            inputs,
            vec![
                json!({
                    "video": "https://fixtures.invalid/source.mp4",
                    "target_resolution": "1080p",
                    "target_fps": 24
                }),
                json!({
                    "video": "https://fixtures.invalid/source.mp4",
                    "target_resolution": "4k",
                    "target_fps": 60
                }),
            ]
        );
        let media = core.media();
        let named = |asset_id: &str| {
            let entry = media
                .entries
                .iter()
                .find(|entry| entry.id == asset_id)
                .unwrap();
            (
                entry.name.clone(),
                entry
                    .generation_input
                    .as_ref()
                    .and_then(|input| input.resolution.clone()),
            )
        };
        assert_eq!(
            named(&placeholders[0]),
            ("hd-24.mp4 1080p".to_string(), Some("1080p".to_string()))
        );
        assert_eq!(
            named(&placeholders[1]),
            ("full-hd-60.mp4 4K".to_string(), Some("4k".to_string()))
        );
    }

    #[test]
    fn topaz_upscale_refuses_sources_without_a_larger_target_before_paying() {
        let (_temp, bundle, core) = saved_core_with_source_videos(&[("uhd", 3840, 2160, 30.0)]);
        let mock = MockTransport::new();
        let (cache, models) = runtime_dirs(&bundle);
        let bridge = build_bridge_with_clients(
            core.clone(),
            cache,
            models,
            Arc::new(FixtureClients {
                client: fixture_client(&mock),
            }),
        );
        let error = bridge
            .submit(
                GenerationRequest::Upscale(UpscaleMediaArgs {
                    cost_authorized: Some(true),
                    media_ref: "uhd".to_string(),
                    model: Some("replicate:topaz-upscale".to_string()),
                    source_clip_id: None,
                }),
                &MediaCancelToken::new(),
            )
            .unwrap_err();
        assert!(error.contains("largest upscale target"), "{error}");
        assert!(mock.calls().is_empty());
        assert!(core
            .media()
            .entries
            .iter()
            .all(|entry| entry.generation_input.is_none()));
    }

    #[test]
    fn topaz_upscale_warns_when_the_frame_rate_is_clamped() {
        let (_temp, bundle, core) = saved_core_with_source_videos(&[("fast", 1280, 720, 120.0)]);
        let mock = MockTransport::new();
        let (cache, models) = runtime_dirs(&bundle);
        let bridge = build_bridge_with_clients(
            core,
            cache,
            models,
            Arc::new(FixtureClients {
                client: fixture_client(&mock),
            }),
        );
        let prepared = bridge
            .prepare(GenerationRequest::Upscale(UpscaleMediaArgs {
                cost_authorized: Some(true),
                media_ref: "fast".to_string(),
                model: Some("replicate:topaz-upscale".to_string()),
                source_clip_id: None,
            }))
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(prepared.upscale_target_fps, Some(60));
        assert_eq!(prepared.plan.input.resolution.as_deref(), Some("1080p"));
        assert_eq!(
            prepared.warnings,
            vec![
                "the source is 120 fps but the upscaler accepts 15-60 fps, so the result will be 60 fps"
                    .to_string()
            ]
        );
    }

    #[test]
    fn upscale_finalization_accepts_the_requested_target_aspect_and_size() {
        let source = |id: &str, kind: ClipType, width: i32, height: i32| {
            opentake_domain::MediaManifestEntry {
                id: id.to_string(),
                name: id.to_string(),
                kind,
                source: opentake_domain::MediaSource::Project {
                    relative_path: format!("media/{id}"),
                },
                duration: 1.0,
                generation_input: None,
                source_width: Some(width),
                source_height: Some(height),
                source_fps: Some(30.0),
                has_audio: Some(false),
                color: None,
                proxy: None,
                folder_id: None,
                cached_remote_url: None,
                cached_remote_url_expires_at: None,
            }
        };
        let media = opentake_domain::MediaManifest {
            entries: vec![
                source("hd", ClipType::Video, 1280, 720),
                source("portrait", ClipType::Video, 1080, 1920),
                source("dci", ClipType::Video, 2048, 1080),
                source("wide", ClipType::Video, 2560, 1080),
                source("still", ClipType::Image, 3, 2),
            ],
            ..Default::default()
        };
        let input = |source: &str, resolution: Option<&str>| GenerationInput {
            model: "replicate:topaz-upscale".to_string(),
            source_asset_id: Some(source.to_string()),
            resolution: resolution.map(str::to_string),
            ..Default::default()
        };
        let check = |input: &GenerationInput, width: u32, height: u32| {
            validate_upscale_result(&media, input, Some(width), Some(height))
        };
        let full_hd = input("hd", Some("1080p"));
        for (width, height) in [(1920, 1080), (1920, 1088), (2560, 1440)] {
            assert!(check(&full_hd, width, height).is_ok(), "{width}x{height}");
        }
        for (width, height) in [(1280, 720), (1080, 1920), (1440, 1080)] {
            assert!(check(&full_hd, width, height).is_err(), "{width}x{height}");
        }
        let uhd_portrait = input("portrait", Some("4k"));
        assert!(check(&uhd_portrait, 2160, 3840).is_ok());
        assert!(check(&uhd_portrait, 3840, 2160).is_err());
        assert!(check(&input("dci", Some("4k")), 4096, 2160).is_ok());
        assert!(check(&input("wide", Some("4k")), 5120, 2160).is_ok());
        assert!(check(&input("wide", Some("4k")), 3840, 2160).is_err());
        // Jobs submitted before targets were sent got the provider default.
        let legacy = input("hd", None);
        assert!(check(&legacy, 1920, 1080).is_ok());
        assert!(check(&legacy, 1280, 720).is_err());
        assert!(check(&input("hd", Some("8k")), 1920, 1080).is_err());
        let still = input("still", None);
        assert!(check(&still, 6, 4).is_ok());
        assert!(check(&still, 9, 6).is_err());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn transient_poll_failures_are_retried_and_the_result_downloads_once() {
        let (_temp, bundle, core) = saved_core();
        let mock = MockTransport::new();
        let status = "https://mockfal/fal-ai/flux-pro/requests/flaky-1/status";
        let result = "https://mockfal/fal-ai/flux-pro/requests/flaky-1";
        mock.on(
            Method::Post,
            FLUX_SUBMIT,
            200,
            json!({"request_id": "flaky-1", "status": "IN_QUEUE"}),
        );
        mock.on(Method::Get, status, 200, json!({"status": "IN_PROGRESS"}));
        mock.on_transport_error(Method::Get, status, "connection reset by peer");
        mock.on(Method::Get, status, 503, json!({}));
        mock.on(
            Method::Get,
            status,
            429,
            json!({"error": {"code": "rate_limited", "message": "slow down"}}),
        );
        mock.on(Method::Get, status, 200, json!({"status": "IN_PROGRESS"}));
        mock.on(Method::Get, status, 200, json!({"status": "COMPLETED"}));
        mock.on(
            Method::Get,
            result,
            200,
            json!({"images": [{"url": png_data_url()}]}),
        );
        let (cache, models) = runtime_dirs(&bundle);
        let bridge = build_bridge_with_clients(
            core.clone(),
            cache,
            models,
            Arc::new(FixtureClients {
                client: quick_retry_client(&mock, 5),
            }),
        );

        let submitted = submit_fixture_image(&bridge);
        let ready = wait_for_ready_model(&core, "fal:flux-pro").await;
        assert_eq!(ready.id, submitted.placeholder_asset_ids[0]);
        wait_until("the job to finish", || !bridge.has_active()).await;
        let calls = mock.calls();
        assert_eq!(calls.iter().filter(|call| call.url == status).count(), 6);
        assert_eq!(calls.iter().filter(|call| call.url == result).count(), 1);
        let log = Project::open(&bundle).unwrap().generation_log.unwrap();
        assert!(log
            .entries
            .iter()
            .all(|entry| entry.status != Some(GenerationJobStatus::Failed)));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn exhausted_poll_retries_leave_the_job_recoverable() {
        let (_temp, bundle, core) = saved_core();
        let mock = MockTransport::new();
        let status = "https://mockfal/fal-ai/flux-pro/requests/offline-1/status";
        mock.on(
            Method::Post,
            FLUX_SUBMIT,
            200,
            json!({"request_id": "offline-1", "status": "IN_QUEUE"}),
        );
        mock.on_connect_error(Method::Get, status, "network is unreachable");
        mock.on(
            Method::Get,
            "https://mockfal/fal-ai/flux-pro/requests/offline-1",
            200,
            json!({"images": [{"url": png_data_url()}]}),
        );
        let (cache, models) = runtime_dirs(&bundle);
        let admission = crate::updater::InstallAdmissionGate::default();
        let bridge = build_bridge_with_clients_and_admission(
            core.clone(),
            cache,
            models,
            Arc::new(FixtureClients {
                client: quick_retry_client(&mock, 3),
            }),
            admission.clone(),
        );

        let submitted = submit_fixture_image(&bridge);
        let asset_id = submitted.placeholder_asset_ids[0].clone();
        wait_until("the job to stop polling", || !bridge.has_active()).await;
        assert_eq!(
            mock.calls()
                .iter()
                .filter(|call| call.url == status)
                .count(),
            4,
            "one poll plus three retries"
        );
        let input = placeholder_input(&core, &asset_id);
        assert_eq!(input.status, Some(GenerationJobStatus::Generating));
        assert_eq!(input.error_code, None);
        assert_eq!(
            input.provider_job_id.as_deref(),
            Some("fal::fal-ai/flux-pro/v1.1|offline-1")
        );
        let persisted = Project::open(&bundle).unwrap();
        let on_disk = persisted
            .manifest
            .entries
            .iter()
            .find(|entry| entry.id == asset_id)
            .and_then(|entry| entry.generation_input.as_ref())
            .unwrap();
        assert_eq!(on_disk.status, Some(GenerationJobStatus::Generating));
        assert!(on_disk.provider_job_id.is_some());
        // The interrupted job holds no install lease.
        drop(admission.begin_install().unwrap());

        // The network is back: recovery takes the same provider job over,
        // without submitting again.
        mock.on(Method::Get, status, 200, json!({"status": "COMPLETED"}));
        assert_eq!(bridge.recover_current_project(), 1);
        let ready = wait_for_ready_model(&core, "fal:flux-pro").await;
        assert_eq!(ready.id, asset_id);
        assert_eq!(
            mock.calls()
                .iter()
                .filter(|call| call.url == FLUX_SUBMIT)
                .count(),
            1
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_job_left_for_recovery_can_still_be_cancelled() {
        let (_temp, bundle, core) = saved_core();
        let mock = MockTransport::new();
        mock.on(
            Method::Post,
            FLUX_SUBMIT,
            200,
            json!({"request_id": "stuck-1", "status": "IN_QUEUE"}),
        );
        mock.on(
            Method::Get,
            "https://mockfal/fal-ai/flux-pro/requests/stuck-1/status",
            502,
            json!({}),
        );
        let (cache, models) = runtime_dirs(&bundle);
        let bridge = build_bridge_with_clients(
            core.clone(),
            cache,
            models,
            Arc::new(FixtureClients {
                client: quick_retry_client(&mock, 1),
            }),
        );
        let submitted = submit_fixture_image(&bridge);
        wait_until("the job to stop polling", || !bridge.has_active()).await;
        assert!(bridge.cancel(&submitted.job_id).unwrap());
        assert_eq!(
            placeholder_input(&core, &submitted.placeholder_asset_ids[0]).status,
            Some(GenerationJobStatus::Cancelled)
        );
        assert!(!bridge.cancel(&submitted.job_id).unwrap());
        assert!(!bridge.cancel("unknown-job").unwrap());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_recovered_job_survives_a_failed_first_poll() {
        let (_temp, bundle, core) = saved_core();
        let runtime = core.runtime_snapshot();
        let committed = core
            .begin_generation_job_for_project(runtime.project_epoch, &bundle, image_plan())
            .unwrap();
        core.update_generation_job_for_project(
            runtime.project_epoch,
            &bundle,
            &committed.job_id,
            GenerationStateUpdate {
                status: GenerationJobStatus::Generating,
                progress: Some(0.25),
                error_code: None,
                provider_job_id: Some("fal::flux-pro|offline-open".to_string()),
                cost_credits: None,
                created_at: None,
            },
        )
        .unwrap();
        drop(core);
        let reopened = AppCore::new();
        reopened.open_project(&bundle).unwrap();
        let mock = MockTransport::new();
        let status = "https://mockfal/flux-pro/requests/offline-open/status";
        mock.on_connect_error(Method::Get, status, "offline");
        mock.on(Method::Get, status, 200, json!({"status": "COMPLETED"}));
        mock.on(
            Method::Get,
            "https://mockfal/flux-pro/requests/offline-open",
            200,
            json!({"images": [{"url": png_data_url()}]}),
        );
        let (cache, models) = runtime_dirs(&bundle);
        let bridge = build_bridge_with_clients(
            reopened.clone(),
            cache,
            models,
            Arc::new(FixtureClients {
                client: quick_retry_client(&mock, 5),
            }),
        );

        assert_eq!(bridge.recover_current_project(), 1);
        let ready = wait_for_ready_model(&reopened, "fal:flux-pro").await;
        assert_eq!(ready.id, committed.placeholder_asset_ids[0]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cancelling_a_hung_upload_ends_the_job_promptly() {
        let (_temp, bundle, core) = saved_core_with_source_videos(&[("clip", 640, 360, 30.0)]);
        let mock = MockTransport::new();
        mock.on_pending(Method::Post, "https://mockrep/v1/files");
        let (cache, models) = runtime_dirs(&bundle);
        let admission = crate::updater::InstallAdmissionGate::default();
        let bridge = build_bridge_with_clients_and_admission(
            core.clone(),
            cache,
            models,
            Arc::new(FixtureClients {
                client: fixture_client(&mock),
            }),
            admission.clone(),
        );
        let submitted = bridge
            .submit(
                GenerationRequest::Upscale(UpscaleMediaArgs {
                    cost_authorized: Some(true),
                    media_ref: "clip".to_string(),
                    model: Some("replicate:topaz-upscale".to_string()),
                    source_clip_id: None,
                }),
                &MediaCancelToken::new(),
            )
            .unwrap();
        wait_until("the upload to start", || {
            mock.calls()
                .iter()
                .any(|call| call.url == "https://mockrep/v1/files")
        })
        .await;
        match &mock.calls().last().unwrap().body {
            opentake_gen::Body::File { len, .. } => assert!(*len > 0),
            other => panic!("the reference must stream from disk, got {other:?}"),
        }

        // The upload never completes, so the job exiting at all shows that
        // the cancellation abandoned it.
        assert!(bridge.cancel(&submitted.job_id).unwrap());
        wait_until("the cancelled job to exit", || !bridge.has_active()).await;
        assert_eq!(
            placeholder_input(&core, &submitted.placeholder_asset_ids[0]).status,
            Some(GenerationJobStatus::Cancelled)
        );
        drop(admission.begin_install().unwrap());
        assert!(!mock
            .calls()
            .iter()
            .any(|call| call.url.contains("/predictions")));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_job_uploading_when_its_project_is_replaced_asks_for_a_retry_on_reopen() {
        let (temp, bundle, core) = saved_core_with_source_videos(&[("clip", 640, 360, 30.0)]);
        let other = saved_bundle(temp.path(), "Other.opentake");
        let mock = MockTransport::new();
        mock.on_pending(Method::Post, "https://mockrep/v1/files");
        let admission = crate::updater::InstallAdmissionGate::default();
        let bridge = following_bridge(
            &core,
            &bundle,
            Arc::new(mock.clone()),
            &mock,
            admission.clone(),
        );
        let submitted = bridge
            .submit(
                GenerationRequest::Upscale(UpscaleMediaArgs {
                    cost_authorized: Some(true),
                    media_ref: "clip".to_string(),
                    model: Some("replicate:topaz-upscale".to_string()),
                    source_clip_id: None,
                }),
                &MediaCancelToken::new(),
            )
            .unwrap();
        let asset_id = submitted.placeholder_asset_ids[0].clone();
        wait_until("the upload to start", || {
            mock.calls()
                .iter()
                .any(|call| call.url == "https://mockrep/v1/files")
        })
        .await;

        // Nothing was submitted, so the upload is abandoned and nothing is
        // kept for the job.
        core.open_project(&other).unwrap();
        wait_until("the detached task to exit", || !bridge.has_active()).await;
        drop(admission.begin_install().unwrap());
        assert_eq!(bridge.orphan_record(&submitted.job_id), None);
        let input = on_disk_input(&bundle, &asset_id);
        assert_eq!(input.status, Some(GenerationJobStatus::Queued));
        assert_eq!(input.provider_job_id, None);

        core.open_project(&bundle).unwrap();
        let failed =
            wait_for_job_status(&core, &submitted.job_id, GenerationJobStatus::Failed).await;
        assert_eq!(
            failed.error_code.as_deref(),
            Some("GENERATION_RESTART_RETRY_REQUIRED")
        );
        let reopened = on_disk_input(&bundle, &asset_id);
        assert_eq!(reopened.status, Some(GenerationJobStatus::Failed));
        assert_eq!(reopened.source_asset_id.as_deref(), Some("clip"));
        assert!(
            core.media().entries.iter().any(|entry| entry.id == "clip"),
            "the source is kept"
        );
        assert!(!bridge.retry_resumes(&submitted.job_id));
        assert!(!mock
            .calls()
            .iter()
            .any(|call| call.url.contains("/predictions")));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cancelling_during_submission_keeps_the_accepted_provider_job() {
        let (_temp, bundle, core) = saved_core();
        let mock = MockTransport::new();
        mock.on(
            Method::Post,
            FLUX_SUBMIT,
            200,
            json!({"request_id": "late-1", "status": "IN_QUEUE"}),
        );
        let transport = GatedTransport::new(&mock, FLUX_SUBMIT);
        let (cache, models) = runtime_dirs(&bundle);
        let bridge = build_bridge_with_clients(
            core.clone(),
            cache,
            models,
            Arc::new(FixtureClients {
                client: fixture_client_with_transport(&mock, transport.clone()),
            }),
        );
        let submitted = submit_fixture_image(&bridge);
        wait_until("the submission to be sent", || transport.is_sent()).await;
        assert!(bridge.cancel(&submitted.job_id).unwrap());
        assert!(
            mock.calls().is_empty(),
            "the cancellation arrived while the provider was still answering"
        );
        assert!(bridge.has_active(), "the job waits for the answer");
        transport.release();
        wait_until("the job to exit", || !bridge.has_active()).await;

        let input = placeholder_input(&core, &submitted.placeholder_asset_ids[0]);
        assert_eq!(input.status, Some(GenerationJobStatus::Cancelled));
        assert_eq!(
            input.provider_job_id.as_deref(),
            Some("fal::fal-ai/flux-pro/v1.1|late-1"),
            "the accepted job is recorded before the cancellation"
        );
        assert!(!mock.calls().iter().any(|call| call.url.contains("/status")));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_submission_that_never_answers_fails_with_an_unknown_outcome() {
        for cancel_while_waiting in [false, true] {
            let (_temp, bundle, core) = saved_core();
            let mock = MockTransport::new();
            mock.on_pending(Method::Post, FLUX_SUBMIT);
            let (cache, models) = runtime_dirs(&bundle);
            let admission = crate::updater::InstallAdmissionGate::default();
            let bridge = build_bridge_with_timings(
                core.clone(),
                cache,
                models,
                Arc::new(FixtureClients {
                    client: fixture_client(&mock),
                }),
                admission.clone(),
                GenerationTimings {
                    submit_timeout: Duration::from_millis(300),
                    ..GenerationTimings::default()
                },
            );
            let submitted = submit_fixture_image(&bridge);
            wait_until("the submission to be sent", || {
                mock.calls().iter().any(|call| call.url == FLUX_SUBMIT)
            })
            .await;
            if cancel_while_waiting {
                assert!(bridge.cancel(&submitted.job_id).unwrap());
            }
            wait_until("the job to exit", || !bridge.has_active()).await;
            let input = placeholder_input(&core, &submitted.placeholder_asset_ids[0]);
            assert_eq!(input.status, Some(GenerationJobStatus::Failed));
            assert_eq!(input.error_code.as_deref(), Some(SUBMIT_OUTCOME_UNKNOWN));
            assert_eq!(input.provider_job_id, None);
            drop(admission.begin_install().unwrap());
            assert_eq!(mock.call_count(), 1, "nothing is resubmitted");
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_reset_submission_is_an_unknown_outcome_but_a_refused_connection_is_not() {
        for (connect_failure, expected) in [
            (false, SUBMIT_OUTCOME_UNKNOWN),
            (true, "GENERATION_SUBMIT_FAILED"),
        ] {
            let (_temp, bundle, core) = saved_core();
            let mock = MockTransport::new();
            if connect_failure {
                mock.on_connect_error(Method::Post, FLUX_SUBMIT, "connection refused");
            } else {
                mock.on_transport_error(Method::Post, FLUX_SUBMIT, "connection reset");
            }
            let (cache, models) = runtime_dirs(&bundle);
            let bridge = build_bridge_with_clients(
                core.clone(),
                cache,
                models,
                Arc::new(FixtureClients {
                    client: fixture_client(&mock),
                }),
            );
            let submitted = submit_fixture_image(&bridge);
            let failed =
                wait_for_job_status(&core, &submitted.job_id, GenerationJobStatus::Failed).await;
            assert_eq!(failed.error_code.as_deref(), Some(expected));
        }
    }

    #[test]
    fn result_downloads_refuse_private_resolutions_before_connecting() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();
        let staging = tempfile::tempdir().unwrap();
        for answer in [
            vec!["127.0.0.1"],
            vec!["10.0.0.1"],
            vec!["64:ff9b::a00:1"],
            vec!["2002:a00:1::1"],
            vec!["93.184.216.34", "127.0.0.1"],
        ] {
            let addresses = answer
                .iter()
                .map(|ip| std::net::SocketAddr::new(ip.parse().unwrap(), port))
                .collect::<Vec<_>>();
            let downloader = SecureResultDownloader::with_lookup(
                staging.path().to_path_buf(),
                MediaCancelToken::new(),
                Arc::new(move |host, _port| {
                    assert_eq!(host, "cdn.example.test");
                    let addresses = addresses.clone();
                    Box::pin(async move { Ok(addresses) })
                }),
            )
            .unwrap();
            let error = downloader
                .download("asset", "https://cdn.example.test/result.png")
                .unwrap_err();
            assert_eq!(
                error, "generation result host is not a public address",
                "{answer:?}"
            );
        }
        // Nothing ever connected to the local listener.
        assert_eq!(
            listener.accept().unwrap_err().kind(),
            std::io::ErrorKind::WouldBlock
        );
        assert_eq!(std::fs::read_dir(staging.path()).unwrap().count(), 0);
    }

    /// Serve one canned HTTP response per connection on a loopback port and
    /// return the request heads the server read.
    fn serve_http(responses: Vec<Vec<u8>>) -> (u16, std::thread::JoinHandle<Vec<String>>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let mut heads = Vec::new();
            for response in responses {
                let (mut stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(30)))
                    .unwrap();
                let mut head = Vec::new();
                let mut byte = [0_u8; 1];
                while !head.ends_with(b"\r\n\r\n") {
                    match stream.read(&mut byte) {
                        Ok(1) => head.push(byte[0]),
                        _ => break,
                    }
                }
                heads.push(String::from_utf8_lossy(&head).to_ascii_lowercase());
                stream.write_all(&response).unwrap();
            }
            heads
        });
        (port, server)
    }

    /// A downloader that may reach a loopback server over plain HTTP; every
    /// other address stays refused.
    fn loopback_downloader(
        staging: &Path,
        answers: Vec<(&'static str, Vec<std::net::SocketAddr>)>,
    ) -> SecureResultDownloader {
        let mut downloader = SecureResultDownloader::with_lookup(
            staging.to_path_buf(),
            MediaCancelToken::new(),
            Arc::new(move |host, _port| {
                let answer = answers
                    .iter()
                    .find(|(name, _)| *name == host)
                    .map(|(_, addresses)| addresses.clone())
                    .unwrap_or_default();
                Box::pin(async move { Ok(answer) })
            }),
        )
        .unwrap();
        downloader.url_policy =
            |raw| reqwest::Url::parse(raw).map_err(|_| "generation result URL is invalid".into());
        downloader.address_policy = |ip| ip.is_loopback();
        downloader
    }

    #[test]
    fn result_downloads_connect_to_the_pinned_addresses_only() {
        let body = b"pinned result bytes".to_vec();
        let mut response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\n\
             Connection: close\r\n\r\n",
            body.len()
        )
        .into_bytes();
        response.extend_from_slice(&body);
        let (port, server) = serve_http(vec![response]);
        let staging = tempfile::tempdir().unwrap();
        // The name resolves nowhere but through the checked answer, whose
        // first address is dead: the connection falls back to the next.
        let downloader = loopback_downloader(
            staging.path(),
            vec![(
                "cdn.pinned.test",
                vec![
                    std::net::SocketAddr::new("::1".parse().unwrap(), port),
                    std::net::SocketAddr::new("127.0.0.1".parse().unwrap(), port),
                ],
            )],
        );
        let artifact = downloader
            .download(
                "asset",
                &format!("http://cdn.pinned.test:{port}/result.png"),
            )
            .unwrap();
        assert_eq!(artifact.media_type, "image/png");
        assert_eq!(std::fs::read(&artifact.path).unwrap(), body);
        let heads = server.join().unwrap();
        assert!(
            heads[0].contains(&format!("host: cdn.pinned.test:{port}")),
            "{heads:?}"
        );
    }

    #[test]
    fn a_redirect_to_a_private_address_is_refused_before_connecting() {
        let private = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        private.set_nonblocking(true).unwrap();
        let private_port = private.local_addr().unwrap().port();
        let redirect = format!(
            "HTTP/1.1 302 Found\r\nLocation: http://internal.pinned.test:{private_port}/x\r\n\
             Content-Length: 0\r\nConnection: close\r\n\r\n"
        )
        .into_bytes();
        let (port, server) = serve_http(vec![redirect]);
        let staging = tempfile::tempdir().unwrap();
        let downloader = loopback_downloader(
            staging.path(),
            vec![
                (
                    "cdn.pinned.test",
                    vec![std::net::SocketAddr::new(
                        "127.0.0.1".parse().unwrap(),
                        port,
                    )],
                ),
                (
                    "internal.pinned.test",
                    vec![std::net::SocketAddr::new(
                        "10.0.0.7".parse().unwrap(),
                        private_port,
                    )],
                ),
            ],
        );
        let error = downloader
            .download(
                "asset",
                &format!("http://cdn.pinned.test:{port}/result.png"),
            )
            .unwrap_err();
        assert_eq!(error, "generation result host is not a public address");
        assert_eq!(server.join().unwrap().len(), 1);
        assert_eq!(
            private.accept().unwrap_err().kind(),
            std::io::ErrorKind::WouldBlock,
            "the redirect target was never contacted"
        );
        assert_eq!(std::fs::read_dir(staging.path()).unwrap().count(), 0);
    }

    const OPENAI_IMAGES: &str = "https://mockoai/v1/images/generations";

    fn submit_openai_image(bridge: &TauriGenerationBridge) -> GenerationSubmission {
        bridge
            .submit(
                GenerationRequest::Image(GenerateImageArgs {
                    cost_authorized: Some(true),
                    prompt: "synchronous fixture".to_string(),
                    model: Some("openai:gpt-image-1".to_string()),
                    num_images: Some(1),
                    ..Default::default()
                }),
                &MediaCancelToken::new(),
            )
            .unwrap()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_synchronous_result_is_held_when_its_project_closes_and_finalized_after_a_restart() {
        let (temp, bundle, core) = saved_core();
        let other = saved_bundle(temp.path(), "Other.opentake");
        let mock = MockTransport::new();
        let png = png_data_url();
        let b64 = png.split_once(',').unwrap().1.to_string();
        mock.on(
            Method::Post,
            OPENAI_IMAGES,
            200,
            json!({"created": 7, "data": [{"b64_json": b64}]}),
        );
        let transport = GatedTransport::new(&mock, OPENAI_IMAGES);
        let admission = crate::updater::InstallAdmissionGate::default();
        let bridge = following_bridge(&core, &bundle, transport.clone(), &mock, admission.clone());
        let submitted = submit_openai_image(&bridge);
        let asset_id = submitted.placeholder_asset_ids[0].clone();
        wait_until("the submission to be sent", || transport.is_sent()).await;

        core.open_project(&other).unwrap();
        transport.release();
        wait_until("the old task to exit", || !bridge.has_active()).await;
        drop(admission.begin_install().unwrap());
        let orphan = bridge.orphans.get(&submitted.job_id).unwrap().unwrap();
        assert_eq!(orphan.held_results.len(), 1);
        let held = bridge.orphans.held_path(&orphan.held_results[0]);
        assert!(held.is_file());

        drop(bridge);
        drop(core);
        let restarted = AppCore::new();
        let bridge = following_bridge(
            &restarted,
            &bundle,
            Arc::new(mock.clone()),
            &mock,
            crate::updater::InstallAdmissionGate::default(),
        );
        restarted.open_project(&bundle).unwrap();
        bridge.recover_current_project();
        let ready = wait_for_ready_model(&restarted, "openai:gpt-image-1").await;
        assert_eq!(ready.id, asset_id);
        wait_until("the job to finish", || !bridge.has_active()).await;
        assert_eq!(count_calls(&mock, OPENAI_IMAGES), 1, "paid once");
        assert!(
            !mock.calls().iter().any(|call| call.method == Method::Get),
            "a synchronous job is never polled"
        );
        assert_eq!(bridge.orphans.get(&submitted.job_id).unwrap(), None);
        assert!(!held.exists());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_synchronous_result_is_finalized_without_polling() {
        let (_temp, bundle, core) = saved_core();
        let mock = MockTransport::new();
        let png = png_data_url();
        let b64 = png.split_once(',').unwrap().1.to_string();
        mock.on(
            Method::Post,
            OPENAI_IMAGES,
            200,
            json!({"created": 8, "data": [{"b64_json": b64}]}),
        );
        let (cache, models) = runtime_dirs(&bundle);
        let bridge = build_bridge_with_clients(
            core.clone(),
            cache,
            models,
            Arc::new(FixtureClients {
                client: fixture_client(&mock),
            }),
        );
        let submitted = submit_openai_image(&bridge);
        let ready = wait_for_ready_model(&core, "openai:gpt-image-1").await;
        assert_eq!(ready.id, submitted.placeholder_asset_ids[0]);
        assert_eq!(mock.call_count(), 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_interrupted_watch_is_resumed_in_the_same_session() {
        let (_temp, bundle, core) = saved_core();
        let mock = moving_job_mock();
        mock.on_connect_error(Method::Get, PROJECT_STATUS, "offline");
        mock.on_connect_error(Method::Get, PROJECT_STATUS, "offline");
        mock.on(
            Method::Get,
            PROJECT_STATUS,
            200,
            json!({"status": "COMPLETED"}),
        );
        let (cache, models) = runtime_dirs(&bundle);
        let bridge = build_bridge_with_timings(
            core.clone(),
            cache,
            models,
            Arc::new(FixtureClients {
                client: quick_retry_client(&mock, 1),
            }),
            crate::updater::InstallAdmissionGate::default(),
            GenerationTimings {
                resume_delay: Duration::from_millis(20),
                ..GenerationTimings::default()
            },
        );
        let submitted = submit_fixture_image(&bridge);
        let ready = wait_for_ready_model(&core, "fal:flux-pro").await;
        assert_eq!(ready.id, submitted.placeholder_asset_ids[0]);
        assert_eq!(count_calls(&mock, FLUX_SUBMIT), 1, "never resubmitted");
        assert_eq!(count_calls(&mock, PROJECT_RESULT), 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_accepted_but_unusable_submission_has_an_unknown_outcome() {
        for (status, body, expected) in [
            (
                200,
                b"{\"status\":\"IN_QUEUE\"}".to_vec(),
                SUBMIT_OUTCOME_UNKNOWN,
            ),
            (
                200,
                b"<html>gateway</html>".to_vec(),
                SUBMIT_OUTCOME_UNKNOWN,
            ),
            (502, b"{}".to_vec(), SUBMIT_OUTCOME_UNKNOWN),
            (504, b"{}".to_vec(), SUBMIT_OUTCOME_UNKNOWN),
            (500, b"{}".to_vec(), "GENERATION_SUBMIT_FAILED"),
        ] {
            let (_temp, bundle, core) = saved_core();
            let mock = MockTransport::new();
            mock.on_raw(Method::Post, FLUX_SUBMIT, HttpResponse::new(status, body));
            let (cache, models) = runtime_dirs(&bundle);
            let bridge = build_bridge_with_clients(
                core.clone(),
                cache,
                models,
                Arc::new(FixtureClients {
                    client: fixture_client(&mock),
                }),
            );
            let submitted = submit_fixture_image(&bridge);
            let failed =
                wait_for_job_status(&core, &submitted.job_id, GenerationJobStatus::Failed).await;
            assert_eq!(failed.error_code.as_deref(), Some(expected), "{status}");
            assert_eq!(mock.call_count(), 1, "nothing is resubmitted");
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn recovery_never_fails_a_job_that_is_being_registered() {
        let (_temp, bundle, core) = saved_core();
        let mock = MockTransport::new();
        mock.on_pending(Method::Post, FLUX_SUBMIT);
        let (cache, models) = runtime_dirs(&bundle);
        let bridge = build_bridge_with_clients(
            core.clone(),
            cache,
            models,
            Arc::new(FixtureClients {
                client: fixture_client(&mock),
            }),
        );
        // The placeholders' commit announces itself before `submit` returns:
        // start a recovery right then and give it every chance to run
        // before the job is registered.
        let recoveries = Arc::new(Mutex::new(Vec::new()));
        {
            let bridge = Arc::downgrade(&bridge);
            let recoveries = Arc::clone(&recoveries);
            core.subscribe(move |event| {
                if !matches!(event, opentake_core::CoreEvent::MediaChanged { .. }) {
                    return;
                }
                let Some(bridge) = bridge.upgrade() else {
                    return;
                };
                let (done, finished) = std::sync::mpsc::channel();
                recoveries.lock().unwrap().push(std::thread::spawn(move || {
                    bridge.recover_current_project();
                    let _ = done.send(());
                }));
                let _ = finished.recv_timeout(Duration::from_millis(200));
            });
        }
        let submitted = submit_fixture_image(&bridge);
        for recovery in std::mem::take(&mut *recoveries.lock().unwrap()) {
            recovery.join().unwrap();
        }
        let input = placeholder_input(&core, &submitted.placeholder_asset_ids[0]);
        assert_eq!(input.status, Some(GenerationJobStatus::Queued));
        assert_eq!(input.error_code, None);
        assert!(bridge.has_active(), "the submission is still running");
    }

    /// A second saved project to switch to.
    fn saved_bundle(root: &Path, name: &str) -> PathBuf {
        let bundle = root.join(name);
        let mut project = Project::new(&bundle);
        project.generation_log = Some(GenerationLog::new());
        project.save().unwrap();
        bundle
    }

    fn on_disk_input(bundle: &Path, asset_id: &str) -> GenerationInput {
        Project::open(bundle)
            .unwrap()
            .manifest
            .entries
            .into_iter()
            .find(|entry| entry.id == asset_id)
            .and_then(|entry| entry.generation_input)
            .unwrap()
    }

    /// A bridge that follows project identity like the desktop app, polling
    /// often enough for scripted transitions.
    fn following_bridge(
        core: &AppCore,
        bundle: &Path,
        transport: Arc<dyn opentake_gen::HttpTransport>,
        mock: &MockTransport,
        admission: crate::updater::InstallAdmissionGate,
    ) -> Arc<TauriGenerationBridge> {
        let (cache, models) = runtime_dirs(bundle);
        let client = fixture_client_with_transport(mock, transport).with_poll_policy(
            opentake_gen::PollPolicy {
                interval: Duration::from_millis(5),
                ..quick_poll_policy(20)
            },
        );
        let bridge = build_bridge_with_clients_and_admission(
            core.clone(),
            cache,
            models,
            Arc::new(FixtureClients { client }),
            admission,
        );
        bridge.follow_project_identity();
        bridge
    }

    const PROJECT_STATUS: &str = "https://mockfal/fal-ai/flux-pro/requests/moving-1/status";
    const PROJECT_RESULT: &str = "https://mockfal/fal-ai/flux-pro/requests/moving-1";
    const MOVING_PROVIDER_JOB: &str = "fal::fal-ai/flux-pro/v1.1|moving-1";

    fn moving_job_mock() -> MockTransport {
        let mock = MockTransport::new();
        mock.on(
            Method::Post,
            FLUX_SUBMIT,
            200,
            json!({"request_id": "moving-1", "status": "IN_QUEUE"}),
        );
        mock.on(
            Method::Get,
            PROJECT_STATUS,
            200,
            json!({"status": "IN_PROGRESS"}),
        );
        mock.on(
            Method::Get,
            PROJECT_RESULT,
            200,
            json!({"images": [{"url": png_data_url()}]}),
        );
        mock
    }

    fn count_calls(mock: &MockTransport, url: &str) -> usize {
        mock.calls().iter().filter(|call| call.url == url).count()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn save_as_while_generating_finishes_in_the_new_bundle() {
        let (temp, bundle, core) = saved_core();
        let mock = moving_job_mock();
        let bridge = following_bridge(
            &core,
            &bundle,
            Arc::new(mock.clone()),
            &mock,
            crate::updater::InstallAdmissionGate::default(),
        );
        let submitted = submit_fixture_image(&bridge);
        let asset_id = submitted.placeholder_asset_ids[0].clone();
        wait_until("the job to poll", || {
            count_calls(&mock, PROJECT_STATUS) >= 2
        })
        .await;

        let copy = temp.path().join("Copy.opentake");
        core.save_project(Some(copy.clone())).unwrap();
        mock.on(
            Method::Get,
            PROJECT_STATUS,
            200,
            json!({"status": "COMPLETED"}),
        );

        let ready = wait_for_ready_model(&core, "fal:flux-pro").await;
        assert_eq!(ready.id, asset_id);
        wait_until("the job to finish", || !bridge.has_active()).await;
        assert_eq!(count_calls(&mock, PROJECT_RESULT), 1, "downloaded once");
        assert_eq!(core.project_dir().as_deref(), Some(copy.as_path()));
        let saved = on_disk_input(&copy, &asset_id);
        assert_eq!(saved.status, Some(GenerationJobStatus::Ready));
        assert!(MediaResolver::new(&core.media(), Some(&copy))
            .expected_path(&asset_id)
            .unwrap()
            .is_file());
        // The original bundle keeps the state it had when it was saved away.
        let original = on_disk_input(&bundle, &asset_id);
        assert_eq!(original.status, Some(GenerationJobStatus::Generating));
        assert_eq!(
            original.provider_job_id.as_deref(),
            Some(MOVING_PROVIDER_JOB)
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn switching_projects_detaches_the_job_and_reopening_hands_it_over() {
        let (temp, bundle, core) = saved_core();
        let other = saved_bundle(temp.path(), "Other.opentake");
        let mock = moving_job_mock();
        let admission = crate::updater::InstallAdmissionGate::default();
        let bridge = following_bridge(
            &core,
            &bundle,
            Arc::new(mock.clone()),
            &mock,
            admission.clone(),
        );
        let submitted = submit_fixture_image(&bridge);
        let asset_id = submitted.placeholder_asset_ids[0].clone();
        wait_until("the job to poll", || {
            count_calls(&mock, PROJECT_STATUS) >= 2
        })
        .await;

        core.open_project(&other).unwrap();
        // The old project's job was not cancelled at the provider: its
        // bundle keeps Generating and the id. Its polling task exits at once
        // and then releases its install lease.
        wait_until("the old task to exit", || !bridge.has_active()).await;
        drop(admission.begin_install().unwrap());
        let parked = on_disk_input(&bundle, &asset_id);
        assert_eq!(parked.status, Some(GenerationJobStatus::Generating));
        assert_eq!(parked.provider_job_id.as_deref(), Some(MOVING_PROVIDER_JOB));
        assert_eq!(
            bridge.orphans.get(&submitted.job_id).unwrap(),
            None,
            "the project records the job already"
        );
        let polls_while_away = count_calls(&mock, PROJECT_STATUS);
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(
            count_calls(&mock, PROJECT_STATUS),
            polls_while_away,
            "nothing polls for a closed project"
        );

        mock.on(
            Method::Get,
            PROJECT_STATUS,
            200,
            json!({"status": "COMPLETED"}),
        );
        core.open_project(&bundle).unwrap();
        let ready = wait_for_ready_model(&core, "fal:flux-pro").await;
        assert_eq!(ready.id, asset_id);
        wait_until("the job to finish", || !bridge.has_active()).await;
        assert_eq!(count_calls(&mock, PROJECT_RESULT), 1, "downloaded once");
        assert_eq!(count_calls(&mock, FLUX_SUBMIT), 1, "never resubmitted");
        assert_eq!(
            on_disk_input(&bundle, &asset_id).status,
            Some(GenerationJobStatus::Ready)
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn save_as_before_the_submission_answers_records_the_job_in_the_new_bundle() {
        let (temp, bundle, core) = saved_core();
        let mock = moving_job_mock();
        let transport = GatedTransport::new(&mock, FLUX_SUBMIT);
        let bridge = following_bridge(
            &core,
            &bundle,
            transport.clone(),
            &mock,
            crate::updater::InstallAdmissionGate::default(),
        );
        let submitted = submit_fixture_image(&bridge);
        let asset_id = submitted.placeholder_asset_ids[0].clone();
        wait_until("the submission to be sent", || transport.is_sent()).await;

        let copy = temp.path().join("Copy.opentake");
        core.save_project(Some(copy.clone())).unwrap();
        assert!(mock.calls().is_empty(), "Save As ran before the answer");
        transport.release();
        wait_until("the accepted job to be recorded", || {
            placeholder_input(&core, &asset_id)
                .provider_job_id
                .is_some()
        })
        .await;
        mock.on(
            Method::Get,
            PROJECT_STATUS,
            200,
            json!({"status": "COMPLETED"}),
        );

        wait_for_ready_model(&core, "fal:flux-pro").await;
        let saved = on_disk_input(&copy, &asset_id);
        assert_eq!(saved.status, Some(GenerationJobStatus::Ready));
        assert_eq!(saved.provider_job_id.as_deref(), Some(MOVING_PROVIDER_JOB));
        let original = on_disk_input(&bundle, &asset_id);
        assert_eq!(original.status, Some(GenerationJobStatus::Queued));
        assert_eq!(count_calls(&mock, FLUX_SUBMIT), 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_job_accepted_after_its_project_closed_resumes_after_a_restart() {
        let (temp, bundle, core) = saved_core();
        let other = saved_bundle(temp.path(), "Other.opentake");
        let mock = moving_job_mock();
        let transport = GatedTransport::new(&mock, FLUX_SUBMIT);
        let admission = crate::updater::InstallAdmissionGate::default();
        let bridge = following_bridge(&core, &bundle, transport.clone(), &mock, admission.clone());
        let submitted = submit_fixture_image(&bridge);
        let asset_id = submitted.placeholder_asset_ids[0].clone();
        wait_until("the submission to be sent", || transport.is_sent()).await;

        core.open_project(&other).unwrap();
        // The detached task still waits for a paid answer: an update may not
        // install before the answer is recorded.
        assert!(bridge.has_active());
        assert!(admission.begin_install().is_err());
        transport.release();
        wait_until("the old task to exit", || !bridge.has_active()).await;
        drop(admission.begin_install().unwrap());
        assert_eq!(
            count_calls(&mock, FLUX_SUBMIT),
            1,
            "the provider accepted it"
        );
        assert!(
            on_disk_input(&bundle, &asset_id).provider_job_id.is_none(),
            "the closed project could not record it"
        );
        let orphan = bridge.orphans.get(&submitted.job_id).unwrap().unwrap();
        assert_eq!(orphan.provider_job_id.as_deref(), Some(MOVING_PROVIDER_JOB));

        // The app quits and restarts: nothing survives but disk.
        drop(bridge);
        drop(core);
        mock.on(
            Method::Get,
            PROJECT_STATUS,
            200,
            json!({"status": "COMPLETED"}),
        );
        let restarted = AppCore::new();
        let bridge = following_bridge(
            &restarted,
            &bundle,
            Arc::new(mock.clone()),
            &mock,
            crate::updater::InstallAdmissionGate::default(),
        );
        restarted.open_project(&bundle).unwrap();
        bridge.recover_current_project();
        let ready = wait_for_ready_model(&restarted, "fal:flux-pro").await;
        assert_eq!(ready.id, asset_id);
        let reopened = on_disk_input(&bundle, &asset_id);
        assert_eq!(reopened.status, Some(GenerationJobStatus::Ready));
        assert_eq!(
            reopened.provider_job_id.as_deref(),
            Some(MOVING_PROVIDER_JOB)
        );
        assert_eq!(count_calls(&mock, FLUX_SUBMIT), 1, "never resubmitted");
        assert_eq!(bridge.orphans.get(&submitted.job_id).unwrap(), None);
    }

    #[test]
    fn recovery_hands_a_job_over_only_after_the_replaced_task_exits() {
        let (_temp, bundle, core) = saved_core();
        let mock = MockTransport::new();
        let (cache, models) = runtime_dirs(&bundle);
        let bridge = build_bridge_with_clients(
            core.clone(),
            cache,
            models,
            Arc::new(FixtureClients {
                client: fixture_client(&mock),
            }),
        );
        let epoch = core.project_revision().project_epoch;
        let RecoveryClaim::Claimed { cancel, task, .. } = bridge.claim_job("job-1", epoch, &bundle)
        else {
            panic!("a free job is claimed");
        };
        assert!(matches!(
            bridge.claim_job("job-1", epoch, &bundle),
            RecoveryClaim::Running
        ));
        // Another identity finds the task bound elsewhere and must wait.
        assert!(matches!(
            bridge.claim_job("job-1", epoch + 1, &bundle),
            RecoveryClaim::Handover
        ));
        assert!(cancel.is_cancelled());
        // The replaced task keeps its install lease until it exits.
        assert!(bridge.has_active());
        assert!(matches!(
            bridge.claim_job("job-1", epoch + 1, &bundle),
            RecoveryClaim::Handover
        ));
        bridge.finish_task("job-1", task);
        assert!(!bridge.has_active());
        assert!(matches!(
            bridge.claim_job("job-1", epoch + 1, &bundle),
            RecoveryClaim::Claimed { .. }
        ));
    }

    async fn assert_submit_failure(status: u16, body: serde_json::Value, expected_code: &str) {
        let (_temp, bundle, core) = saved_core();
        let mock = MockTransport::new();
        mock.on(
            Method::Post,
            "https://mockfal/fal-ai/flux-pro/v1.1",
            status,
            body,
        );
        let (cache, models) = runtime_dirs(&bundle);
        let bridge = build_bridge_with_clients(
            core.clone(),
            cache,
            models,
            Arc::new(FixtureClients {
                client: fixture_client(&mock),
            }),
        );
        let dispatcher = Dispatcher::with_bridges(
            Arc::new(AppCoreHandle::new(core.clone())),
            Arc::new(RwLock::new(PluginRegistry::new())),
            None,
            Some(bridge),
        );
        let accepted = dispatcher.dispatch(
            "generate_image",
            json!({
                "costAuthorized": true,
                "prompt": "failure fixture",
                "model": "fal:flux-pro",
                "aspectRatio": "1:1"
            }),
        );
        assert!(!accepted.is_error, "{}", accepted.text_joined());
        for _ in 0..100 {
            let terminal = core
                .media()
                .entries
                .iter()
                .find_map(|entry| entry.generation_input.as_ref())
                .is_some_and(|input| input.status == Some(GenerationJobStatus::Failed));
            if terminal {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let input = core
            .media()
            .entries
            .into_iter()
            .find_map(|entry| entry.generation_input)
            .unwrap();
        assert_eq!(input.status, Some(GenerationJobStatus::Failed));
        assert_eq!(input.error_code.as_deref(), Some(expected_code));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn production_dispatch_maps_auth_and_rate_limit_to_safe_fixed_codes() {
        assert_submit_failure(
            401,
            json!({"error": {"code": "unauthenticated", "message": "private"}}),
            "GENERATION_AUTH_FAILED",
        )
        .await;
        assert_submit_failure(
            429,
            json!({"error": {"code": "rate_limited", "message": "private"}}),
            "GENERATION_RATE_LIMITED",
        )
        .await;
    }

    /// A committed fal image job the provider accepted as
    /// [`MOVING_PROVIDER_JOB`], failed with `code`.
    fn failed_accepted_job(
        core: &AppCore,
        bundle: &Path,
        provider_job_id: &str,
        code: &str,
    ) -> (String, String) {
        let epoch = core.runtime_snapshot().project_epoch;
        let committed = core
            .begin_generation_job_for_project(epoch, bundle, image_plan())
            .unwrap();
        core.update_generation_job_for_project(
            epoch,
            bundle,
            &committed.job_id,
            GenerationStateUpdate {
                status: GenerationJobStatus::Generating,
                progress: Some(0.15),
                error_code: None,
                provider_job_id: Some(provider_job_id.to_string()),
                cost_credits: None,
                created_at: None,
            },
        )
        .unwrap();
        let asset_id = committed.placeholder_asset_ids[0].clone();
        core.fail_generation_output_for_project(epoch, bundle, &asset_id, code, None)
            .unwrap();
        (committed.job_id, asset_id)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn retrying_an_accepted_job_that_failed_locally_resumes_it_without_paying_again() {
        let (_temp, bundle, core) = saved_core();
        let mock = moving_job_mock();
        mock.on(
            Method::Get,
            PROJECT_STATUS,
            200,
            json!({"status": "COMPLETED"}),
        );
        let bridge = outcome_bridge(&core, &bundle, &mock);
        let (job_id, asset_id) = failed_accepted_job(
            &core,
            &bundle,
            MOVING_PROVIDER_JOB,
            "GENERATION_DOWNLOAD_FAILED",
        );

        assert!(bridge.retry_resumes(&job_id));
        // Nothing is paid, so no cost authorization is needed.
        let resumed = bridge.retry(&job_id, false).unwrap();
        assert_eq!(resumed.job_id, job_id);
        assert_eq!(resumed.placeholder_asset_ids, vec![asset_id.clone()]);
        let ready = wait_for_job_status(&core, &job_id, GenerationJobStatus::Ready).await;
        assert_eq!(ready.provider_job_id.as_deref(), Some(MOVING_PROVIDER_JOB));
        assert_eq!(
            on_disk_input(&bundle, &asset_id).status,
            Some(GenerationJobStatus::Ready)
        );
        assert_eq!(count_calls(&mock, FLUX_SUBMIT), 0, "never resubmitted");
        assert!(count_calls(&mock, PROJECT_STATUS) >= 1);
        wait_until("the job to finish", || !bridge.has_active()).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_resumed_job_that_fails_again_is_submitted_again_by_the_next_retry() {
        let (_temp, bundle, core) = saved_core();
        let mock = moving_job_mock();
        mock.on(
            Method::Get,
            PROJECT_STATUS,
            401,
            json!({"error": {"code": "unauthenticated", "message": "private"}}),
        );
        let bridge = outcome_bridge(&core, &bundle, &mock);
        let (job_id, asset_id) =
            failed_accepted_job(&core, &bundle, MOVING_PROVIDER_JOB, STATE_PERSIST_FAILED);

        bridge.retry(&job_id, false).unwrap();
        let failed = wait_for_job_status(&core, &job_id, GenerationJobStatus::Failed).await;
        assert_eq!(failed.error_code.as_deref(), Some(RESUME_FAILED));
        wait_until("the job to finish", || !bridge.has_active()).await;
        assert_eq!(
            on_disk_input(&bundle, &asset_id).error_code.as_deref(),
            Some(RESUME_FAILED)
        );
        assert!(!bridge.retry_resumes(&job_id));
        assert!(
            bridge.retry(&job_id, false).is_err(),
            "a resubmission needs a cost authorization"
        );
        assert_eq!(count_calls(&mock, FLUX_SUBMIT), 0);
    }

    #[test]
    fn retry_submits_again_when_the_provider_job_cannot_be_resumed() {
        let (_temp, bundle, core) = saved_core();
        let mock = MockTransport::new();
        let bridge = outcome_bridge(&core, &bundle, &mock);
        // The provider refused or failed the job.
        let (refused, _) = failed_accepted_job(
            &core,
            &bundle,
            MOVING_PROVIDER_JOB,
            "GENERATION_PROVIDER_FAILED",
        );
        assert!(!bridge.retry_resumes(&refused));
        let (unavailable, _) =
            failed_accepted_job(&core, &bundle, MOVING_PROVIDER_JOB, PROVIDER_POLL_REFUSED);
        assert!(!bridge.retry_resumes(&unavailable));
        // A synchronous vendor's result cannot be fetched again.
        let (synchronous, _) = failed_accepted_job(
            &core,
            &bundle,
            "openai::image-1",
            "GENERATION_DOWNLOAD_FAILED",
        );
        assert!(!bridge.retry_resumes(&synchronous));
        // Nothing was accepted.
        let epoch = core.runtime_snapshot().project_epoch;
        let never_sent = core
            .begin_generation_job_for_project(epoch, &bundle, image_plan())
            .unwrap();
        core.fail_generation_output_for_project(
            epoch,
            &bundle,
            &never_sent.placeholder_asset_ids[0],
            STATE_PERSIST_FAILED,
            None,
        )
        .unwrap();
        assert!(!bridge.retry_resumes(&never_sent.job_id));
        for job_id in [&refused, &unavailable, &synchronous, &never_sent.job_id] {
            assert!(bridge.retry(job_id, false).is_err());
        }
        assert!(mock.calls().is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn retry_requires_fresh_cost_authorization_and_creates_a_new_job() {
        let (_temp, bundle, core) = saved_core();
        let runtime = core.runtime_snapshot();
        let failed = core
            .begin_generation_job_for_project(runtime.project_epoch, &bundle, image_plan())
            .unwrap();
        core.update_generation_job_for_project(
            runtime.project_epoch,
            &bundle,
            &failed.job_id,
            GenerationStateUpdate {
                status: GenerationJobStatus::Failed,
                progress: None,
                error_code: Some("GENERATION_PROVIDER_FAILED".to_string()),
                provider_job_id: None,
                cost_credits: None,
                created_at: None,
            },
        )
        .unwrap();
        let mock = MockTransport::new();
        mock.on(
            Method::Post,
            "https://mockfal/fal-ai/flux-pro/v1.1",
            200,
            json!({"request_id": "retry-1", "status": "IN_QUEUE"}),
        );
        mock.on(
            Method::Get,
            "https://mockfal/fal-ai/flux-pro/requests/retry-1/status",
            200,
            json!({"status": "COMPLETED"}),
        );
        mock.on(
            Method::Get,
            "https://mockfal/fal-ai/flux-pro/requests/retry-1",
            200,
            json!({"images": [{"url": png_data_url()}]}),
        );
        let (cache, models) = runtime_dirs(&bundle);
        let bridge = build_bridge_with_clients(
            core.clone(),
            cache,
            models,
            Arc::new(FixtureClients {
                client: fixture_client(&mock),
            }),
        );
        assert!(bridge.retry(&failed.job_id, false).is_err());
        assert!(mock.calls().is_empty());
        let retried = bridge.retry(&failed.job_id, true).unwrap();
        assert_ne!(retried.job_id, failed.job_id);
        for _ in 0..100 {
            let ready = core.media().entries.iter().any(|entry| {
                entry.generation_input.as_ref().is_some_and(|input| {
                    input.job_id.as_deref() == Some(retried.job_id.as_str())
                        && input.status == Some(GenerationJobStatus::Ready)
                })
            });
            if ready {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let statuses = core
            .media()
            .entries
            .iter()
            .filter_map(|entry| entry.generation_input.as_ref())
            .map(|input| (input.job_id.clone().unwrap(), input.status.unwrap()))
            .collect::<HashMap<_, _>>();
        assert_eq!(
            statuses.get(&failed.job_id),
            Some(&GenerationJobStatus::Failed)
        );
        assert_eq!(
            statuses.get(&retried.job_id),
            Some(&GenerationJobStatus::Ready)
        );
    }

    fn outcome_bridge(
        core: &AppCore,
        bundle: &Path,
        mock: &MockTransport,
    ) -> Arc<TauriGenerationBridge> {
        let (cache, models) = runtime_dirs(bundle);
        build_bridge_with_clients(
            core.clone(),
            cache,
            models,
            Arc::new(FixtureClients {
                client: fixture_client(mock),
            }),
        )
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_unknown_outcome_after_its_project_closed_is_reported_on_reopen() {
        let (temp, bundle, core) = saved_core();
        let other = saved_bundle(temp.path(), "Other.opentake");
        let mock = MockTransport::new();
        mock.on_raw(
            Method::Post,
            FLUX_SUBMIT,
            HttpResponse::new(504, b"{}".to_vec()),
        );
        let transport = GatedTransport::new(&mock, FLUX_SUBMIT);
        let admission = crate::updater::InstallAdmissionGate::default();
        let bridge = following_bridge(&core, &bundle, transport.clone(), &mock, admission);
        let submitted = submit_fixture_image(&bridge);
        let asset_id = submitted.placeholder_asset_ids[0].clone();
        wait_until("the submission to be sent", || transport.is_sent()).await;

        core.open_project(&other).unwrap();
        transport.release();
        wait_until("the old task to exit", || !bridge.has_active()).await;
        let record = bridge.orphans.get(&submitted.job_id).unwrap().unwrap();
        assert_eq!(record.provider_job_id, None);

        drop(bridge);
        drop(core);
        let restarted = AppCore::new();
        restarted.open_project(&bundle).unwrap();
        let bridge = outcome_bridge(&restarted, &bundle, &mock);
        assert_eq!(bridge.recover_current_project(), 0);
        let input = on_disk_input(&bundle, &asset_id);
        assert_eq!(input.status, Some(GenerationJobStatus::Failed));
        assert_eq!(input.error_code.as_deref(), Some(SUBMIT_OUTCOME_UNKNOWN));
        assert_eq!(bridge.orphans.get(&submitted.job_id).unwrap(), None);
        assert_eq!(count_calls(&mock, FLUX_SUBMIT), 1, "nothing is resubmitted");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn quitting_while_a_detached_submission_waits_reports_an_unknown_outcome() {
        let (temp, bundle, core) = saved_core();
        let other = saved_bundle(temp.path(), "Other.opentake");
        let mock = moving_job_mock();
        let transport = GatedTransport::new(&mock, FLUX_SUBMIT);
        let bridge = following_bridge(
            &core,
            &bundle,
            transport.clone(),
            &mock,
            crate::updater::InstallAdmissionGate::default(),
        );
        let submitted = submit_fixture_image(&bridge);
        let asset_id = submitted.placeholder_asset_ids[0].clone();
        wait_until("the submission to be sent", || transport.is_sent()).await;
        core.open_project(&other).unwrap();

        // The app quits while the detached task still waits for the answer:
        // it never records anything more.
        let restarted = AppCore::new();
        restarted.open_project(&bundle).unwrap();
        let recovered = outcome_bridge(&restarted, &bundle, &mock);
        assert_eq!(recovered.recover_current_project(), 0);
        let input = on_disk_input(&bundle, &asset_id);
        assert_eq!(input.status, Some(GenerationJobStatus::Failed));
        assert_eq!(input.error_code.as_deref(), Some(SUBMIT_OUTCOME_UNKNOWN));
        assert!(bridge.has_active(), "the old task never answered");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_accepted_job_its_open_project_could_not_record_is_resumed() {
        let (_temp, bundle, core) = saved_core();
        let mock = moving_job_mock();
        mock.on(
            Method::Get,
            PROJECT_STATUS,
            200,
            json!({"status": "COMPLETED"}),
        );
        let transport = GatedTransport::new(&mock, FLUX_SUBMIT);
        let (cache, models) = runtime_dirs(&bundle);
        let client = fixture_client_with_transport(&mock, transport.clone());
        let bridge = build_bridge_with_timings(
            core.clone(),
            cache,
            models,
            Arc::new(FixtureClients { client }),
            crate::updater::InstallAdmissionGate::default(),
            GenerationTimings {
                resume_delay: Duration::from_millis(500),
                ..GenerationTimings::default()
            },
        );
        let submitted = submit_fixture_image(&bridge);
        let asset_id = submitted.placeholder_asset_ids[0].clone();
        wait_until("the submission to be sent", || transport.is_sent()).await;
        // A non-empty directory at `media.json` makes the next write fail.
        std::fs::remove_file(bundle.join("media.json")).unwrap();
        std::fs::create_dir_all(bundle.join("media.json/blocker")).unwrap();
        transport.release();
        wait_until("the accepted job to be kept", || {
            bridge
                .orphan_record(&submitted.job_id)
                .is_some_and(|record| {
                    record.provider_job_id.as_deref() == Some(MOVING_PROVIDER_JOB)
                })
        })
        .await;
        std::fs::remove_dir_all(bundle.join("media.json")).unwrap();

        let ready = wait_for_ready_model(&core, "fal:flux-pro").await;
        assert_eq!(ready.id, asset_id);
        assert_eq!(
            on_disk_input(&bundle, &asset_id).provider_job_id.as_deref(),
            Some(MOVING_PROVIDER_JOB)
        );
        assert_eq!(count_calls(&mock, FLUX_SUBMIT), 1, "never resubmitted");
        wait_until("the job to finish", || !bridge.has_active()).await;
        assert_eq!(bridge.orphan_record(&submitted.job_id), None);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_accepted_id_kept_only_in_memory_is_resumed_over_its_pending_record() {
        let (_temp, bundle, core) = saved_core();
        let mock = moving_job_mock();
        mock.on(
            Method::Get,
            PROJECT_STATUS,
            200,
            json!({"status": "COMPLETED"}),
        );
        let transport = GatedTransport::new(&mock, FLUX_SUBMIT);
        let (cache, models) = runtime_dirs(&bundle);
        let client = fixture_client_with_transport(&mock, transport.clone());
        let bridge = build_bridge_with_timings(
            core.clone(),
            cache,
            models,
            Arc::new(FixtureClients { client }),
            crate::updater::InstallAdmissionGate::default(),
            GenerationTimings {
                resume_delay: Duration::from_millis(500),
                ..GenerationTimings::default()
            },
        );
        let submitted = submit_fixture_image(&bridge);
        let asset_id = submitted.placeholder_asset_ids[0].clone();
        wait_until("the submission to be sent", || transport.is_sent()).await;
        // Neither the project nor the orphan store can record the accepted
        // id: it is kept only in memory, behind the pending record.
        bridge
            .orphans
            .fail_records
            .store(true, std::sync::atomic::Ordering::SeqCst);
        std::fs::remove_file(bundle.join("media.json")).unwrap();
        std::fs::create_dir_all(bundle.join("media.json/blocker")).unwrap();
        transport.release();
        wait_until("the accepted job to be kept in memory", || {
            bridge
                .runtime
                .orphaned_submissions
                .lock()
                .unwrap()
                .contains_key(&submitted.job_id)
        })
        .await;
        let durable = bridge.orphans.get(&submitted.job_id).unwrap().unwrap();
        assert_eq!(durable.provider_job_id, None, "the pending record remains");
        assert_eq!(
            bridge
                .orphan_record(&submitted.job_id)
                .and_then(|record| record.provider_job_id)
                .as_deref(),
            Some(MOVING_PROVIDER_JOB)
        );
        std::fs::remove_dir_all(bundle.join("media.json")).unwrap();

        let ready = wait_for_ready_model(&core, "fal:flux-pro").await;
        assert_eq!(ready.id, asset_id);
        let input = on_disk_input(&bundle, &asset_id);
        assert_eq!(input.provider_job_id.as_deref(), Some(MOVING_PROVIDER_JOB));
        assert_eq!(input.error_code, None);
        assert_eq!(count_calls(&mock, FLUX_SUBMIT), 1, "never resubmitted");
        wait_until("the job to finish", || !bridge.has_active()).await;
        assert_eq!(bridge.orphan_record(&submitted.job_id), None);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_finished_submission_leaves_no_pending_record() {
        let (_temp, bundle, core) = saved_core();
        let mock = MockTransport::new();
        mock.on_raw(
            Method::Post,
            FLUX_SUBMIT,
            HttpResponse::new(504, b"{}".to_vec()),
        );
        let bridge = outcome_bridge(&core, &bundle, &mock);
        let submitted = submit_fixture_image(&bridge);
        wait_for_job_status(&core, &submitted.job_id, GenerationJobStatus::Failed).await;
        wait_until("the job to finish", || !bridge.has_active()).await;
        assert_eq!(bridge.orphan_record(&submitted.job_id), None);
    }

    const HELD_PROVIDER_JOB: &str = "openai::held-1";

    /// An OpenAI job with `outputs` placeholders whose provider job id the
    /// project records, and a held result for each placeholder.
    fn held_openai_job(
        core: &AppCore,
        bundle: &Path,
        bridge: &TauriGenerationBridge,
        outputs: usize,
    ) -> (
        String,
        Vec<String>,
        Vec<crate::generation_orphans::HeldResult>,
    ) {
        let epoch = core.runtime_snapshot().project_epoch;
        let mut plan = image_plan();
        plan.provider = "openai".to_string();
        plan.input.model = "openai:gpt-image-1".to_string();
        plan.output_count = outputs;
        let committed = core
            .begin_generation_job_for_project(epoch, bundle, plan)
            .unwrap();
        core.update_generation_job_for_project(
            epoch,
            bundle,
            &committed.job_id,
            GenerationStateUpdate {
                status: GenerationJobStatus::Generating,
                progress: Some(0.15),
                error_code: None,
                provider_job_id: Some(HELD_PROVIDER_JOB.to_string()),
                cost_credits: None,
                created_at: None,
            },
        )
        .unwrap();
        let staging = bundle.parent().unwrap().join("to-hold");
        std::fs::create_dir_all(&staging).unwrap();
        let held = (0..outputs)
            .map(|index| {
                let staged = staging.join(format!("{index}.png"));
                std::fs::write(&staged, png_bytes(4 + index as u32, 4)).unwrap();
                bridge.orphans.hold(&staged, "image/png").unwrap()
            })
            .collect::<Vec<_>>();
        bridge
            .orphans
            .record(OrphanedGeneration {
                job_id: committed.job_id.clone(),
                provider_job_id: Some(HELD_PROVIDER_JOB.to_string()),
                project_path: bundle.display().to_string(),
                recorded_at: 1,
                held_results: held.clone(),
            })
            .unwrap();
        (committed.job_id, committed.placeholder_asset_ids, held)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn held_results_finish_a_partly_finalized_job_and_keep_its_ready_output() {
        let (_temp, bundle, core) = saved_core();
        let mock = MockTransport::new();
        let bridge = outcome_bridge(&core, &bundle, &mock);
        let (job_id, placeholders, held) = held_openai_job(&core, &bundle, &bridge, 2);
        // Output 0 was finalized before the job left its project.
        let epoch = core.runtime_snapshot().project_epoch;
        core.update_generation_job_for_project(
            epoch,
            &bundle,
            &job_id,
            GenerationStateUpdate {
                status: GenerationJobStatus::Downloading,
                progress: Some(0.8),
                error_code: None,
                provider_job_id: Some(HELD_PROVIDER_JOB.to_string()),
                cost_credits: None,
                created_at: None,
            },
        )
        .unwrap();
        let first_leaf = format!("{}.png", placeholders[0]);
        let first_bytes = png_bytes(9, 9);
        core.finalize_generation_output_with_media_for_project(
            epoch,
            &bundle,
            PreparedGenerationOutput {
                asset_id: placeholders[0].clone(),
                relative_path: format!("media/{first_leaf}"),
                probe: ProbedMedia {
                    duration_secs: 0.0,
                    width: Some(9),
                    height: Some(9),
                    fps: None,
                    has_audio: false,
                    color: None,
                },
                created_at: None,
            },
            &first_leaf,
            first_bytes.len() as u64,
            &mut std::io::Cursor::new(first_bytes.clone()),
        )
        .unwrap();

        assert_eq!(bridge.recover_current_project(), 1);
        wait_until("the job to finish", || !bridge.has_active()).await;
        let first = on_disk_input(&bundle, &placeholders[0]);
        assert_eq!(first.status, Some(GenerationJobStatus::Ready));
        assert_eq!(
            std::fs::read(bundle.join("media").join(&first_leaf)).unwrap(),
            first_bytes,
            "the finalized output keeps its result"
        );
        let second = on_disk_input(&bundle, &placeholders[1]);
        assert_eq!(
            second.status,
            Some(GenerationJobStatus::Ready),
            "{second:?}"
        );
        let second_entry = core
            .media()
            .entries
            .into_iter()
            .find(|entry| entry.id == placeholders[1])
            .unwrap();
        assert_eq!(second_entry.source_width, Some(5), "held result 1 is used");
        assert_eq!(bridge.orphans.get(&job_id).unwrap(), None);
        assert!(held
            .iter()
            .all(|held| !bridge.orphans.held_path(held).exists()));
        assert!(mock.calls().is_empty(), "nothing is paid again");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn held_results_are_removed_when_their_job_fails() {
        let (_temp, bundle, core) = saved_core();
        let mock = MockTransport::new();
        let bridge = outcome_bridge(&core, &bundle, &mock);
        let (job_id, placeholders, held) = held_openai_job(&core, &bundle, &bridge, 1);
        // Finalization cannot stage anything.
        std::fs::create_dir_all(bridge.staging_root.parent().unwrap()).unwrap();
        std::fs::write(&bridge.staging_root, b"not a directory").unwrap();

        assert_eq!(bridge.recover_current_project(), 1);
        wait_until("the job to finish", || !bridge.has_active()).await;
        let input = on_disk_input(&bundle, &placeholders[0]);
        assert_eq!(input.status, Some(GenerationJobStatus::Failed));
        assert_eq!(bridge.orphans.get(&job_id).unwrap(), None);
        assert!(!bridge.orphans.held_path(&held[0]).exists());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn held_results_are_kept_until_their_job_end_is_saved() {
        let (_temp, bundle, core) = saved_core();
        let mock = MockTransport::new();
        let bridge = outcome_bridge(&core, &bundle, &mock);
        let (job_id, placeholders, held) = held_openai_job(&core, &bundle, &bridge, 1);
        // Neither the result nor the failure can be saved.
        std::fs::remove_file(bundle.join("media.json")).unwrap();
        std::fs::create_dir_all(bundle.join("media.json/blocker")).unwrap();

        assert_eq!(bridge.recover_current_project(), 1);
        wait_until("the job to finish", || !bridge.has_active()).await;
        let status = core
            .media()
            .entries
            .into_iter()
            .find(|entry| entry.id == placeholders[0])
            .and_then(|entry| entry.generation_input)
            .and_then(|input| input.status);
        assert_eq!(status, Some(GenerationJobStatus::Generating));
        assert!(bridge.orphans.get(&job_id).unwrap().is_some());
        assert!(bridge.orphans.held_path(&held[0]).is_file());

        std::fs::remove_dir_all(bundle.join("media.json")).unwrap();
        assert_eq!(bridge.recover_current_project(), 1);
        wait_until("the job to finish", || !bridge.has_active()).await;
        let input = on_disk_input(&bundle, &placeholders[0]);
        assert_eq!(input.status, Some(GenerationJobStatus::Ready), "{input:?}");
        assert_eq!(bridge.orphans.get(&job_id).unwrap(), None);
        assert!(!bridge.orphans.held_path(&held[0]).exists());
        assert!(mock.calls().is_empty(), "nothing is paid again");
    }

    #[test]
    fn results_held_before_a_failed_hold_are_recorded() {
        let (_temp, bundle, core) = saved_core();
        let mock = MockTransport::new();
        let bridge = outcome_bridge(&core, &bundle, &mock);
        let job = GenerationJob::succeeded(
            HELD_PROVIDER_JOB,
            vec![png_data_url(), "data:image/png;base64,@@@".to_string()],
        );
        let kept = bridge.keep_accepted_job_blocking(
            "job-partial",
            HELD_PROVIDER_JOB.to_string(),
            bundle.display().to_string(),
            Some(job),
            &MediaCancelToken::new(),
        );
        assert!(kept.is_err());
        let record = bridge.orphans.get("job-partial").unwrap().unwrap();
        assert_eq!(record.held_results.len(), 1);
        assert!(bridge.orphans.held_path(&record.held_results[0]).is_file());
        let results_dir = bridge.orphans.held_path(&record.held_results[0]);
        assert_eq!(
            std::fs::read_dir(results_dir.parent().unwrap())
                .unwrap()
                .count(),
            1,
            "no held file is left without a record"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_timed_out_hold_is_recorded_before_the_task_exits() {
        let (_temp, bundle, core) = saved_core();
        let mock = MockTransport::new();
        let (cache, models) = runtime_dirs(&bundle);
        let bridge = build_bridge_with_timings(
            core.clone(),
            cache,
            models,
            Arc::new(FixtureClients {
                client: fixture_client(&mock),
            }),
            crate::updater::InstallAdmissionGate::default(),
            GenerationTimings {
                hold_timeout: Duration::ZERO,
                ..GenerationTimings::default()
            },
        );
        let binding = JobBinding::new(core.runtime_snapshot().project_epoch, bundle.clone());
        binding.detach();
        bridge
            .keep_accepted_job(
                &binding,
                "job-slow",
                HELD_PROVIDER_JOB.to_string(),
                Some(GenerationJob::succeeded(
                    HELD_PROVIDER_JOB,
                    vec![png_data_url()],
                )),
            )
            .await;
        let record = bridge.orphans.get("job-slow").unwrap();
        assert_eq!(
            record.and_then(|record| record.provider_job_id).as_deref(),
            Some(HELD_PROVIDER_JOB)
        );
    }
}
