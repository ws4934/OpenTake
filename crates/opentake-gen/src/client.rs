//! `GenClient` — the top-level generation client. Two auth modes with an
//! identical call surface (axiom A7): `Bearer` (managed proxy + billing) and
//! `Byok` (local direct-to-vendor via provider adapters + static catalog).
//!
//! - `list_models` — managed: GET /v1/models; BYOK: built-in static catalog.
//! - `submit` — managed: POST /v1/generations; BYOK: route to an adapter.
//! - `get` — single status snapshot.
//! - `watch` — poll until terminal, replicating the upstream `runJob` loop
//!   (`GenerationService.swift:338-361`): only succeeded/failed stop the stream.
//!   Transient poll failures are retried with backoff; when the retry budget
//!   or the deadline runs out the watch ends as interrupted, never as failed,
//!   so a paid job stays recoverable (upstream "retry on reopen").
//! - `sign_upload` / `upload_reference` — managed: presigned PUT; BYOK: adapter.

use crate::catalog::{Catalog, CatalogEntry, ModelKind};
use crate::error::{map_http_response, GenError};
use crate::job::GenerationJob;
use crate::params::GenerationParams;
use crate::provider::ProviderRegistry;
use crate::transport::{file_upload_body, HttpRequest, HttpTransport, Method, ReqwestTransport};
use async_trait::async_trait;
use futures_util::stream::Stream;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

/// Timing of [`GenClient::watch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PollPolicy {
    /// Delay between polls while the job is queued or running.
    pub interval: Duration,
    /// Delay after the first transient poll failure, doubled for each further
    /// consecutive failure (with jitter).
    pub retry_base: Duration,
    /// Largest delay between retries.
    pub retry_max: Duration,
    /// Largest server-requested `Retry-After` delay honored.
    pub retry_after_max: Duration,
    /// Consecutive transient failures tolerated before the watch stops as
    /// interrupted.
    pub retry_budget: u32,
    /// Upper bound on one status poll; a slower poll counts as a transient
    /// failure.
    pub poll_timeout: Duration,
}

impl Default for PollPolicy {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(2),
            retry_base: Duration::from_secs(2),
            retry_max: Duration::from_secs(60),
            retry_after_max: Duration::from_secs(5 * 60),
            // With 2s doubling to 60s this rides out about 25 minutes of
            // continuous failures.
            retry_budget: 30,
            poll_timeout: Duration::from_secs(90),
        }
    }
}

impl PollPolicy {
    /// The wait before retry `attempt` (1-based): the server's `Retry-After`
    /// when given, otherwise exponential backoff with equal jitter.
    pub fn retry_delay(&self, attempt: u32, retry_after: Option<Duration>) -> Duration {
        if let Some(retry_after) = retry_after {
            return retry_after.min(self.retry_after_max);
        }
        let exponent = attempt.saturating_sub(1).min(20);
        let backoff = self
            .retry_base
            .saturating_mul(1_u32 << exponent)
            .min(self.retry_max);
        let half = backoff / 2;
        half + jitter_up_to(backoff - half)
    }
}

/// A uniformly spread duration in `0..=span`, from the standard library's
/// per-instance random hasher keys (no RNG dependency needed for jitter).
fn jitter_up_to(span: Duration) -> Duration {
    use std::hash::{BuildHasher, Hasher};
    let nanos = u64::try_from(span.as_nanos()).unwrap_or(u64::MAX);
    if nanos == 0 {
        return Duration::ZERO;
    }
    let random = std::collections::hash_map::RandomState::new()
        .build_hasher()
        .finish();
    Duration::from_nanos(random % nanos.saturating_add(1))
}

/// Why [`GenClient::watch`] stopped before a terminal provider status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatchInterruption {
    /// Consecutive transient poll failures used up the retry budget.
    RetryBudgetExhausted,
    /// The watch deadline passed while the job was still pending.
    DeadlineExceeded,
}

/// One observation from [`GenClient::watch`]. The stream ends after a
/// terminal `Snapshot`, a `Failed` or an `Interrupted` event.
#[derive(Debug)]
pub enum WatchEvent {
    /// A provider status snapshot; a terminal status ends the stream.
    Snapshot(GenerationJob),
    /// A transient poll failure that is retried after `delay`.
    Retrying {
        attempt: u32,
        delay: Duration,
        error: GenError,
    },
    /// A final poll error: authentication, credits, other 4xx, or an
    /// unusable response.
    Failed(GenError),
    /// Polling stopped before the provider reported a terminal status. The
    /// job may still finish at the provider, so it must stay recoverable.
    Interrupted(WatchInterruption),
}

struct WatchState {
    client: GenClient,
    job_id: String,
    policy: PollPolicy,
    deadline: tokio::time::Instant,
    failures: u32,
    wait: Duration,
    done: bool,
}

/// Asynchronously provides a Bearer token (managed mode). UI injects this,
/// reusing any OIDC provider (axiom A6).
#[async_trait]
pub trait TokenProvider: Send + Sync {
    async fn bearer_token(&self) -> Result<String, GenError>;
}

/// A static-token provider (tests / simple deployments).
pub struct StaticToken(pub String);

#[async_trait]
impl TokenProvider for StaticToken {
    async fn bearer_token(&self) -> Result<String, GenError> {
        Ok(self.0.clone())
    }
}

/// Authentication / routing mode. Both expose the same call surface.
pub enum AuthMode {
    /// Managed: all calls go through a self-hosted proxy that holds vendor keys
    /// and bills usage.
    Bearer {
        base_url: url::Url,
        token_provider: Arc<dyn TokenProvider>,
    },
    /// BYOK: local direct-to-vendor; catalog is the built-in static one.
    Byok {
        registry: ProviderRegistry,
        catalog: Catalog,
    },
}

/// Ticket returned by `sign_upload` (managed presigned-upload flow).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct UploadTicket {
    #[serde(rename = "uploadUrl")]
    pub upload_url: String,
    #[serde(rename = "publicUrl")]
    pub public_url: String,
}

/// Submit result envelope from the proxy (`{"jobId": "..."}`).
#[derive(Debug, serde::Deserialize)]
struct SubmitResult {
    #[serde(rename = "jobId")]
    job_id: String,
}

struct GenClientInner {
    mode: AuthMode,
    http: Arc<dyn HttpTransport>,
    poll_policy: PollPolicy,
}

/// The generation client. Cheap to clone.
#[derive(Clone)]
pub struct GenClient {
    inner: Arc<GenClientInner>,
}

impl GenClient {
    /// Managed-mode client backed by `reqwest`.
    pub fn managed(base_url: url::Url, token_provider: Arc<dyn TokenProvider>) -> Self {
        Self::with_transport(
            AuthMode::Bearer {
                base_url,
                token_provider,
            },
            Arc::new(ReqwestTransport::new()),
        )
    }

    /// BYOK-mode client. `registry` carries the provider adapters; `catalog` is
    /// the static model catalog (typically `Catalog::builtin()`).
    pub fn byok(registry: ProviderRegistry, catalog: Catalog) -> Self {
        // BYOK adapters carry their own transport; this top-level one is unused
        // for vendor calls. A reqwest transport is a safe default.
        Self::with_transport(
            AuthMode::Byok { registry, catalog },
            Arc::new(ReqwestTransport::new()),
        )
    }

    /// Construct with an explicit transport (tests inject `MockTransport`).
    pub fn with_transport(mode: AuthMode, http: Arc<dyn HttpTransport>) -> Self {
        Self {
            inner: Arc::new(GenClientInner {
                mode,
                http,
                poll_policy: PollPolicy::default(),
            }),
        }
    }

    /// Override the `watch` poll interval (tests use zero to run instantly).
    pub fn with_poll_interval(mut self, interval: Duration) -> Self {
        Arc::get_mut(&mut self.inner)
            .expect("with_poll_interval must be called before cloning")
            .poll_policy
            .interval = interval;
        self
    }

    /// Override the whole `watch` timing (poll interval, retries, limits).
    pub fn with_poll_policy(mut self, policy: PollPolicy) -> Self {
        Arc::get_mut(&mut self.inner)
            .expect("with_poll_policy must be called before cloning")
            .poll_policy = policy;
        self
    }

    pub fn poll_policy(&self) -> PollPolicy {
        self.inner.poll_policy
    }

    fn proxy_parts(&self) -> Result<(&url::Url, &Arc<dyn TokenProvider>), GenError> {
        match &self.inner.mode {
            AuthMode::Bearer {
                base_url,
                token_provider,
            } => Ok((base_url, token_provider)),
            AuthMode::Byok { .. } => Err(GenError::NotConfigured),
        }
    }

    async fn bearer_header(&self) -> Result<(String, String), GenError> {
        let (_, tp) = self.proxy_parts()?;
        let token = tp.bearer_token().await?;
        Ok(("Authorization".to_string(), format!("Bearer {token}")))
    }

    fn endpoint(base: &url::Url, path: &str) -> Result<String, GenError> {
        Ok(base.join(path)?.to_string())
    }

    /// Model catalog. Managed: GET /v1/models. BYOK: built-in static catalog.
    pub async fn list_models(&self) -> Result<Vec<CatalogEntry>, GenError> {
        match &self.inner.mode {
            AuthMode::Byok { catalog, .. } => Ok(catalog.entries().to_vec()),
            AuthMode::Bearer { base_url, .. } => {
                let url = Self::endpoint(base_url, "v1/models")?;
                let (hk, hv) = self.bearer_header().await?;
                let resp = self
                    .inner
                    .http
                    .send(HttpRequest::get(url).header(hk, hv))
                    .await?;
                if !resp.is_success() {
                    return Err(map_http_response(&resp));
                }
                resp.json()
            }
        }
    }

    /// Mint a presigned upload ticket (managed only). Replicates
    /// `uploads:generateUploadTicket` via object-storage presign (gen-SPEC §3.4).
    pub async fn sign_upload(&self, content_type: &str) -> Result<UploadTicket, GenError> {
        let (base, _) = self.proxy_parts()?;
        let url = Self::endpoint(base, "v1/uploads/sign")?;
        let (hk, hv) = self.bearer_header().await?;
        let resp = self
            .inner
            .http
            .send(
                HttpRequest::post(url)
                    .header(hk, hv)
                    .json(serde_json::json!({ "contentType": content_type })),
            )
            .await?;
        if !resp.is_success() {
            return Err(map_http_response(&resp));
        }
        resp.json()
    }

    /// Upload a reference file -> public URL. Managed: sign then PUT bytes, use
    /// the returned `publicUrl` (gen-SPEC §3.4). BYOK: delegate to the adapter
    /// for `model_prefix` (vendors differ in upload support).
    pub async fn upload_reference(
        &self,
        path: &Path,
        content_type: &str,
    ) -> Result<String, GenError> {
        match &self.inner.mode {
            AuthMode::Byok { .. } => Err(GenError::Other(anyhow::anyhow!(
                "BYOK upload_reference requires a provider; use upload_reference_via"
            ))),
            AuthMode::Bearer { .. } => {
                // Size-check before minting a ticket; the bytes stream from
                // disk during the PUT.
                let body = file_upload_body(path, content_type).await?;
                let ticket = self.sign_upload(content_type).await?;
                let resp = self
                    .inner
                    .http
                    .send(HttpRequest::new(Method::Put, ticket.upload_url).file(body))
                    .await?;
                if !resp.is_success() {
                    return Err(map_http_response(&resp));
                }
                Ok(ticket.public_url)
            }
        }
    }

    /// BYOK reference upload via the adapter selected by `model_prefix`.
    pub async fn upload_reference_via(
        &self,
        model_prefix: &str,
        path: &Path,
        content_type: &str,
    ) -> Result<String, GenError> {
        match &self.inner.mode {
            AuthMode::Byok { registry, .. } => {
                let (adapter, _) = registry.route(&format!("{model_prefix}:_"))?;
                adapter.upload(path, content_type).await
            }
            AuthMode::Bearer { .. } => self.upload_reference(path, content_type).await,
        }
    }

    /// Submit a job, returning the job id. Managed: POST /v1/generations. BYOK:
    /// route to an adapter and submit.
    pub async fn submit(
        &self,
        model: &str,
        params: GenerationParams,
        project_id: Option<&str>,
    ) -> Result<String, GenError> {
        match &self.inner.mode {
            AuthMode::Byok { registry, catalog } => {
                let (adapter, mut route) = registry.route(model)?;
                if let Some(vendor_model) = catalog
                    .entries()
                    .iter()
                    .find(|entry| entry.id == model)
                    .and_then(|entry| entry.vendor_model.as_deref())
                {
                    route.vendor_model = vendor_model.to_owned();
                }
                let job = adapter.submit(&route, &params).await?;
                Ok(job.id)
            }
            AuthMode::Bearer { base_url, .. } => {
                let url = Self::endpoint(base_url, "v1/generations")?;
                let (hk, hv) = self.bearer_header().await?;
                let mut body = serde_json::json!({
                    "model": model,
                    "params": params,
                });
                if let Some(pid) = project_id {
                    body["projectId"] = serde_json::json!(pid);
                }
                let resp = self
                    .inner
                    .http
                    .send(HttpRequest::post(url).header(hk, hv).json(body))
                    .await?;
                if !resp.is_success() {
                    return Err(map_http_response(&resp));
                }
                let r: SubmitResult = resp.json()?;
                Ok(r.job_id)
            }
        }
    }

    /// Single status snapshot. Managed: GET /v1/generations/:id. BYOK: adapter
    /// poll. `model` is required under BYOK to select the adapter.
    pub async fn get(&self, job_id: &str) -> Result<GenerationJob, GenError> {
        match &self.inner.mode {
            AuthMode::Byok { registry, .. } => {
                // BYOK job ids are adapter-internal; the prefix is encoded by the
                // caller convention "<prefix>::<vendorJobId>" when needed. For
                // single-adapter setups we try each adapter is unnecessary, so we
                // require the prefixed form here.
                let (prefix, vendor_job) = split_byok_job_id(job_id)?;
                let (adapter, _) = registry.route(&format!("{prefix}:_"))?;
                adapter.poll(vendor_job).await
            }
            AuthMode::Bearer { base_url, .. } => {
                let url = Self::endpoint(base_url, &format!("v1/generations/{job_id}"))?;
                let (hk, hv) = self.bearer_header().await?;
                let resp = self
                    .inner
                    .http
                    .send(HttpRequest::get(url).header(hk, hv))
                    .await?;
                if !resp.is_success() {
                    return Err(map_http_response(&resp));
                }
                resp.json()
            }
        }
    }

    /// BYOK convenience: submit and return a prefixed job id usable with
    /// `watch_byok` / `get` (the prefix lets `get` re-select the adapter).
    pub async fn submit_byok(
        &self,
        model: &str,
        params: GenerationParams,
    ) -> Result<String, GenError> {
        let route_prefix = crate::provider::ModelRoute::parse(model)?.prefix;
        let vendor_job = self.submit(model, params, None).await?;
        Ok(format!("{route_prefix}::{vendor_job}"))
    }

    /// Subscribe to a job until it reaches a terminal state, polling at the
    /// configured interval. Replicates the upstream subscription loop:
    /// queued/running continue, succeeded/failed terminate.
    ///
    /// A transient poll failure (network, HTTP 408/429/5xx) is reported as
    /// [`WatchEvent::Retrying`] and retried after backoff or `Retry-After`. A
    /// final error ends the stream with [`WatchEvent::Failed`]. When the retry
    /// budget or `deadline` runs out first, the stream ends with
    /// [`WatchEvent::Interrupted`]: the provider job may still complete.
    pub fn watch(&self, job_id: &str, deadline: Duration) -> impl Stream<Item = WatchEvent> + Send {
        let state = WatchState {
            client: self.clone(),
            job_id: job_id.to_string(),
            policy: self.inner.poll_policy,
            deadline: tokio::time::Instant::now() + deadline,
            failures: 0,
            wait: Duration::ZERO,
            done: false,
        };
        futures_util::stream::unfold(state, |mut state| async move {
            if state.done {
                return None;
            }
            if state.wait.is_zero() {
                // Never spin a zero-interval watch without letting other
                // tasks (cancellation, persistence) run.
                tokio::task::yield_now().await;
            } else {
                let wake = (tokio::time::Instant::now() + state.wait).min(state.deadline);
                tokio::time::sleep_until(wake).await;
            }
            let now = tokio::time::Instant::now();
            if now >= state.deadline {
                state.done = true;
                return Some((
                    WatchEvent::Interrupted(WatchInterruption::DeadlineExceeded),
                    state,
                ));
            }
            let poll_timeout = state.policy.poll_timeout.min(state.deadline - now);
            let result =
                match tokio::time::timeout(poll_timeout, state.client.get(&state.job_id)).await {
                    Ok(result) => result,
                    Err(_) if tokio::time::Instant::now() >= state.deadline => {
                        state.done = true;
                        return Some((
                            WatchEvent::Interrupted(WatchInterruption::DeadlineExceeded),
                            state,
                        ));
                    }
                    Err(_) => Err(GenError::Transport("status poll timed out".to_string())),
                };
            match result {
                Ok(job) => {
                    state.failures = 0;
                    state.done = job.status.is_terminal();
                    state.wait = state.policy.interval;
                    Some((WatchEvent::Snapshot(job), state))
                }
                Err(error) if error.is_transient() => {
                    state.failures += 1;
                    if state.failures > state.policy.retry_budget {
                        state.done = true;
                        return Some((
                            WatchEvent::Interrupted(WatchInterruption::RetryBudgetExhausted),
                            state,
                        ));
                    }
                    let delay = state
                        .policy
                        .retry_delay(state.failures, error.retry_after());
                    state.wait = delay;
                    Some((
                        WatchEvent::Retrying {
                            attempt: state.failures,
                            delay,
                            error,
                        },
                        state,
                    ))
                }
                Err(error) => {
                    state.done = true;
                    Some((WatchEvent::Failed(error), state))
                }
            }
        })
    }
}

/// Split a BYOK prefixed job id `"<prefix>::<vendorJobId>"`.
fn split_byok_job_id(job_id: &str) -> Result<(&str, &str), GenError> {
    job_id
        .split_once("::")
        .filter(|(p, v)| !p.is_empty() && !v.is_empty())
        .ok_or_else(|| {
            GenError::Other(anyhow::anyhow!(
                "BYOK job id must be '<prefix>::<vendorJobId>', got '{job_id}'"
            ))
        })
}

/// Compute a `canGenerate` signal (gen-SPEC §5.3). Managed: a token is
/// obtainable. BYOK: at least one adapter is registered.
pub async fn can_generate(client: &GenClient) -> bool {
    match &client.inner.mode {
        AuthMode::Bearer { token_provider, .. } => token_provider.bearer_token().await.is_ok(),
        AuthMode::Byok { registry, .. } => ["fal", "replicate", "openai", "elevenlabs"]
            .iter()
            .any(|p| registry.has_prefix(p)),
    }
}

/// Filter a catalog list by kind (mirrors the proxy `?type=` filter).
pub fn filter_by_kind(entries: &[CatalogEntry], kind: ModelKind) -> Vec<CatalogEntry> {
    entries.iter().filter(|e| e.kind == kind).cloned().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::JobStatus;
    use crate::params::{AudioParams, ImageParams, UpscaleParams, VideoParams};
    use crate::provider::{ElevenLabsAdapter, FalAdapter, ReplicateAdapter};
    use crate::transport::{HttpResponse, MockTransport};
    use futures_util::StreamExt;
    use serde_json::json;

    fn byok_client(mock: &MockTransport) -> GenClient {
        let fal =
            FalAdapter::new(Arc::new(mock.clone()), "fal-secret").with_base("https://mockfal");
        let registry = ProviderRegistry::new().with(Arc::new(fal));
        GenClient::with_transport(
            AuthMode::Byok {
                registry,
                catalog: Catalog::builtin(),
            },
            Arc::new(mock.clone()),
        )
        .with_poll_interval(Duration::ZERO)
    }

    fn managed_client(mock: &MockTransport) -> GenClient {
        GenClient::with_transport(
            AuthMode::Bearer {
                base_url: url::Url::parse("https://proxy.test/").unwrap(),
                token_provider: Arc::new(StaticToken("jwt-abc".into())),
            },
            Arc::new(mock.clone()),
        )
        .with_poll_interval(Duration::ZERO)
    }

    const WATCH_DEADLINE: Duration = Duration::from_secs(60);

    /// Fast retries for scripted failures.
    fn quick_policy() -> PollPolicy {
        PollPolicy {
            interval: Duration::ZERO,
            retry_base: Duration::from_millis(1),
            retry_max: Duration::from_millis(4),
            retry_after_max: Duration::from_millis(5),
            retry_budget: 5,
            poll_timeout: Duration::from_millis(200),
        }
    }

    /// The snapshots of a watch that must contain nothing else.
    fn snapshots(events: Vec<WatchEvent>) -> Vec<GenerationJob> {
        events
            .into_iter()
            .map(|event| match event {
                WatchEvent::Snapshot(job) => job,
                other => panic!("unexpected watch event {other:?}"),
            })
            .collect()
    }

    #[tokio::test]
    async fn byok_list_models_returns_builtin_catalog() {
        let client = byok_client(&MockTransport::new());
        let models = client.list_models().await.unwrap();
        assert!(!models.is_empty());
        assert!(models.iter().any(|m| m.id == "fal:flux-pro"));
    }

    #[tokio::test]
    async fn builtin_replicate_models_use_official_endpoints() {
        let mock = MockTransport::new();
        let registry = ProviderRegistry::new().with(Arc::new(
            ReplicateAdapter::new(Arc::new(mock.clone()), "token").with_base("https://mockrep/v1"),
        ));
        let client = GenClient::byok(registry, Catalog::builtin());
        for (id, endpoint, params) in [
            (
                "replicate:seedance-1-pro",
                "https://mockrep/v1/models/bytedance/seedance-1-pro/predictions",
                GenerationParams::Video(VideoParams {
                    prompt: "scene".into(),
                    duration: 5,
                    aspect_ratio: "16:9".into(),
                    ..Default::default()
                }),
            ),
            (
                "replicate:topaz-upscale",
                "https://mockrep/v1/models/topazlabs/video-upscale/predictions",
                GenerationParams::Upscale(UpscaleParams {
                    source_url: "https://x/video.mp4".into(),
                    duration_seconds: 5,
                    target_resolution: Some("4k".into()),
                    target_fps: Some(60),
                }),
            ),
        ] {
            mock.on(
                Method::Post,
                endpoint,
                201,
                json!({"id":"p","status":"starting"}),
            );
            client.submit_byok(id, params).await.unwrap();
            let call = mock.last_call().unwrap();
            assert_eq!(call.url, endpoint);
            let crate::transport::Body::Json(body) = call.body else {
                panic!("expected JSON");
            };
            assert!(body.get("version").is_none());
            if id == "replicate:topaz-upscale" {
                assert_eq!(
                    body["input"],
                    json!({
                        "video": "https://x/video.mp4",
                        "target_resolution": "4k",
                        "target_fps": 60
                    })
                );
            }
        }
    }

    #[tokio::test]
    async fn builtin_elevenlabs_uses_documented_tts_model_and_voice_id() {
        let mock = MockTransport::new();
        let voice = "21m00Tcm4TlvDq8ikWAM";
        mock.on_raw(
            Method::Post,
            format!("https://mockel/v1/text-to-speech/{voice}"),
            HttpResponse::new(200, b"speech".to_vec()),
        );
        let registry = ProviderRegistry::new().with(Arc::new(
            ElevenLabsAdapter::new(Arc::new(mock.clone()), "key").with_base("https://mockel/v1"),
        ));
        let client = GenClient::byok(registry, Catalog::builtin());
        let mut params = AudioParams::new("hello", false);
        params.voice = Some("rachel".into());
        client
            .submit_byok(
                "elevenlabs:eleven-multilingual-v2",
                GenerationParams::Audio(params),
            )
            .await
            .unwrap();
        let call = mock.last_call().unwrap();
        assert_eq!(
            call.url,
            format!("https://mockel/v1/text-to-speech/{voice}")
        );
        let crate::transport::Body::Json(body) = call.body else {
            panic!("expected JSON");
        };
        assert_eq!(body["model_id"], "eleven_multilingual_v2");
    }

    #[tokio::test]
    async fn byok_submit_then_watch_to_succeeded() {
        let mock = MockTransport::new();
        // submit
        mock.on(
            Method::Post,
            "https://mockfal/fal-ai/flux-pro/v1.1",
            200,
            json!({"request_id": "req-7", "status": "IN_QUEUE"}),
        );
        // poll status: queued -> running -> completed
        mock.on_sequence(
            Method::Get,
            "https://mockfal/fal-ai/flux-pro/requests/req-7/status",
            vec![
                (200, json!({"status": "IN_QUEUE"})),
                (200, json!({"status": "IN_PROGRESS"})),
                (200, json!({"status": "COMPLETED"})),
            ],
        );
        // terminal result fetch
        mock.on(
            Method::Get,
            "https://mockfal/fal-ai/flux-pro/requests/req-7",
            200,
            json!({"images": [{"url": "https://out/final.png"}]}),
        );

        let client = byok_client(&mock);
        let params = GenerationParams::Image(ImageParams::new("a cat", "1:1", 1));
        let job_id = client.submit_byok("fal:flux-pro", params).await.unwrap();
        assert!(job_id.starts_with("fal::"));

        let jobs = snapshots(client.watch(&job_id, WATCH_DEADLINE).collect().await);
        let statuses: Vec<JobStatus> = jobs.iter().map(|job| job.status).collect();
        assert_eq!(
            statuses,
            vec![JobStatus::Queued, JobStatus::Running, JobStatus::Succeeded]
        );
        let last = jobs.last().unwrap();
        assert_eq!(last.result_urls, Some(vec!["https://out/final.png".into()]));
    }

    #[tokio::test]
    async fn watch_stops_immediately_on_terminal_first_poll() {
        let mock = MockTransport::new();
        mock.on(
            Method::Get,
            "https://mockfal/m/requests/r/status",
            200,
            json!({"status": "COMPLETED"}),
        );
        mock.on(
            Method::Get,
            "https://mockfal/m/requests/r",
            200,
            json!({"video": {"url": "https://out/v.mp4"}}),
        );
        let client = byok_client(&mock);
        let jobs = snapshots(client.watch("fal::m|r", WATCH_DEADLINE).collect().await);
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].status, JobStatus::Succeeded);
    }

    #[tokio::test]
    async fn watch_stops_on_failed() {
        let mock = MockTransport::new();
        mock.on(
            Method::Get,
            "https://mockfal/m/requests/r/status",
            200,
            json!({"status": "FAILED", "error": "bad"}),
        );
        let client = byok_client(&mock);
        let jobs = snapshots(client.watch("fal::m|r", WATCH_DEADLINE).collect().await);
        assert_eq!(jobs.len(), 1);
        let job = &jobs[0];
        assert_eq!(job.status, JobStatus::Failed);
        assert_eq!(job.error_message.as_deref(), Some("bad"));
    }

    #[tokio::test]
    async fn managed_list_models_hits_proxy_with_bearer() {
        let mock = MockTransport::new();
        mock.on(
            Method::Get,
            "https://proxy.test/v1/models",
            200,
            json!([{
                "id": "fal:x", "kind": "image", "displayName": "X",
                "allowedEndpoints": [], "responseShape": "images",
                "uiCapabilities": {"aspectRatios": ["1:1"], "supportsImageReference": false, "maxImages": 1}
            }]),
        );
        let client = managed_client(&mock);
        let models = client.list_models().await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "fal:x");
        assert!(mock
            .last_call()
            .unwrap()
            .headers
            .iter()
            .any(|(k, v)| k == "Authorization" && v == "Bearer jwt-abc"));
    }

    #[tokio::test]
    async fn managed_submit_returns_job_id() {
        let mock = MockTransport::new();
        mock.on(
            Method::Post,
            "https://proxy.test/v1/generations",
            200,
            json!({"jobId": "server-job-1"}),
        );
        let client = managed_client(&mock);
        let params = GenerationParams::Image(ImageParams::new("x", "1:1", 1));
        let id = client
            .submit("fal:flux-pro", params, Some("proj-1"))
            .await
            .unwrap();
        assert_eq!(id, "server-job-1");
        match mock.last_call().unwrap().body {
            crate::transport::Body::Json(v) => {
                assert_eq!(v["model"], "fal:flux-pro");
                assert_eq!(v["projectId"], "proj-1");
                assert_eq!(v["params"]["kind"], "image");
            }
            _ => panic!("expected json"),
        }
    }

    #[tokio::test]
    async fn managed_get_then_watch() {
        let mock = MockTransport::new();
        mock.on_sequence(
            Method::Get,
            "https://proxy.test/v1/generations/job-9",
            vec![
                (200, json!({"id": "job-9", "status": "queued"})),
                (200, json!({"id": "job-9", "status": "running"})),
                (
                    200,
                    json!({"id": "job-9", "status": "succeeded", "resultUrls": ["https://out/a.png"]}),
                ),
            ],
        );
        let client = managed_client(&mock);
        let snapshot = client.get("job-9").await.unwrap();
        assert_eq!(snapshot.status, JobStatus::Queued);
        // watch continues from the next poll
        let jobs = snapshots(client.watch("job-9", WATCH_DEADLINE).collect().await);
        // first watch poll returns running (2nd seq), then succeeded (3rd)
        assert_eq!(
            jobs.iter().map(|job| job.status).collect::<Vec<_>>(),
            vec![JobStatus::Running, JobStatus::Succeeded]
        );
    }

    #[tokio::test]
    async fn managed_sign_upload_and_upload_reference() {
        let mock = MockTransport::new();
        mock.on(
            Method::Post,
            "https://proxy.test/v1/uploads/sign",
            200,
            json!({"uploadUrl": "https://put.test/key", "publicUrl": "https://cdn.test/key"}),
        );
        let ticket = managed_client(&mock)
            .sign_upload("image/png")
            .await
            .unwrap();
        assert_eq!(ticket.public_url, "https://cdn.test/key");
        match mock.last_call().unwrap().body {
            crate::transport::Body::Json(v) => assert_eq!(v["contentType"], "image/png"),
            _ => panic!("expected json"),
        }
    }

    #[tokio::test]
    async fn managed_error_envelope_maps() {
        let mock = MockTransport::new();
        mock.on(
            Method::Get,
            "https://proxy.test/v1/models",
            401,
            json!({"error": {"code": "unauthenticated", "message": "no"}}),
        );
        let client = managed_client(&mock);
        assert!(matches!(
            client.list_models().await,
            Err(GenError::Unauthenticated)
        ));
    }

    #[tokio::test]
    async fn byok_sign_upload_is_not_configured() {
        let client = byok_client(&MockTransport::new());
        assert!(matches!(
            client.sign_upload("image/png").await,
            Err(GenError::NotConfigured)
        ));
    }

    #[tokio::test]
    async fn can_generate_byok_true_with_adapter() {
        let client = byok_client(&MockTransport::new());
        assert!(can_generate(&client).await);
    }

    #[tokio::test]
    async fn can_generate_managed_true_with_token() {
        let client = managed_client(&MockTransport::new());
        assert!(can_generate(&client).await);
    }

    #[test]
    fn filter_by_kind_works() {
        let cat = Catalog::builtin();
        let imgs = filter_by_kind(cat.entries(), ModelKind::Image);
        assert!(!imgs.is_empty());
        assert!(imgs.iter().all(|e| e.kind == ModelKind::Image));
    }

    const RETRY_JOB: &str = "https://proxy.test/v1/generations/job-r";

    #[tokio::test]
    async fn watch_retries_transient_failures_until_the_job_succeeds() {
        let mock = MockTransport::new();
        let mut rate_limited = HttpResponse::new(
            429,
            br#"{"error":{"code":"rate_limited","message":"slow down"}}"#.to_vec(),
        );
        rate_limited
            .headers
            .push(("Retry-After".into(), "0".into()));
        mock.on(
            Method::Get,
            RETRY_JOB,
            200,
            json!({"id": "job-r", "status": "running"}),
        );
        mock.on_transport_error(Method::Get, RETRY_JOB, "connection reset by peer");
        mock.on(Method::Get, RETRY_JOB, 503, json!({}));
        mock.on_raw(Method::Get, RETRY_JOB, rate_limited);
        mock.on(
            Method::Get,
            RETRY_JOB,
            200,
            json!({"id": "job-r", "status": "running"}),
        );
        mock.on(
            Method::Get,
            RETRY_JOB,
            200,
            json!({"id": "job-r", "status": "succeeded", "resultUrls": ["https://out/r.mp4"]}),
        );
        let client = managed_client(&mock).with_poll_policy(quick_policy());

        let events: Vec<WatchEvent> = client.watch("job-r", WATCH_DEADLINE).collect().await;

        let summary = events
            .iter()
            .map(|event| match event {
                WatchEvent::Snapshot(job) => format!("{:?}", job.status),
                WatchEvent::Retrying { attempt, error, .. } => {
                    format!("retry {attempt} {}", error.kind_label())
                }
                other => format!("{other:?}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            summary,
            vec![
                "Running",
                "retry 1 transport",
                "retry 2 http 503",
                "retry 3 http 429",
                "Running",
                "Succeeded"
            ]
        );
        let WatchEvent::Retrying { delay, .. } = &events[3] else {
            unreachable!()
        };
        assert_eq!(*delay, Duration::ZERO, "Retry-After is honored");
        assert_eq!(mock.call_count(), 6);
    }

    #[tokio::test]
    async fn watch_stops_at_once_on_a_final_error() {
        let mock = MockTransport::new();
        mock.on(
            Method::Get,
            RETRY_JOB,
            401,
            json!({"error": {"code": "unauthenticated", "message": "no"}}),
        );
        let client = managed_client(&mock).with_poll_policy(quick_policy());
        let events: Vec<WatchEvent> = client.watch("job-r", WATCH_DEADLINE).collect().await;
        assert!(matches!(
            events.as_slice(),
            [WatchEvent::Failed(GenError::Unauthenticated)]
        ));
        assert_eq!(mock.call_count(), 1);
    }

    #[tokio::test]
    async fn watch_is_interrupted_not_failed_when_retries_run_out() {
        let mock = MockTransport::new();
        mock.on(Method::Get, RETRY_JOB, 502, json!({}));
        let client = managed_client(&mock).with_poll_policy(PollPolicy {
            retry_budget: 3,
            ..quick_policy()
        });
        let events: Vec<WatchEvent> = client.watch("job-r", WATCH_DEADLINE).collect().await;
        assert_eq!(events.len(), 4, "{events:?}");
        assert!(events[..3]
            .iter()
            .all(|event| matches!(event, WatchEvent::Retrying { .. })));
        assert!(matches!(
            events[3],
            WatchEvent::Interrupted(WatchInterruption::RetryBudgetExhausted)
        ));
        assert_eq!(mock.call_count(), 4);
    }

    #[tokio::test]
    async fn watch_is_interrupted_not_failed_when_the_deadline_passes() {
        let mock = MockTransport::new();
        mock.on(
            Method::Get,
            RETRY_JOB,
            200,
            json!({"id": "job-r", "status": "running"}),
        );
        let client = managed_client(&mock).with_poll_policy(PollPolicy {
            interval: Duration::from_millis(5),
            ..quick_policy()
        });
        let started = std::time::Instant::now();
        let events: Vec<WatchEvent> = client
            .watch("job-r", Duration::from_millis(60))
            .collect()
            .await;
        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(matches!(
            events.last(),
            Some(WatchEvent::Interrupted(WatchInterruption::DeadlineExceeded))
        ));
        assert!(events[..events.len() - 1].iter().all(|event| matches!(
            event,
            WatchEvent::Snapshot(job) if job.status == JobStatus::Running
        )));
    }

    #[tokio::test]
    async fn a_hung_poll_is_retried_after_the_poll_timeout() {
        let mock = MockTransport::new();
        mock.on_pending(Method::Get, RETRY_JOB);
        mock.on(
            Method::Get,
            RETRY_JOB,
            200,
            json!({"id": "job-r", "status": "succeeded", "resultUrls": ["https://out/r.png"]}),
        );
        let client = managed_client(&mock).with_poll_policy(PollPolicy {
            poll_timeout: Duration::from_millis(20),
            ..quick_policy()
        });
        let events: Vec<WatchEvent> = client.watch("job-r", WATCH_DEADLINE).collect().await;
        assert!(matches!(
            &events[0],
            WatchEvent::Retrying {
                attempt: 1,
                error: GenError::Transport(_),
                ..
            }
        ));
        assert!(matches!(
            &events[1],
            WatchEvent::Snapshot(job) if job.status == JobStatus::Succeeded
        ));
    }

    #[test]
    fn retry_delays_back_off_with_jitter_up_to_the_cap() {
        let policy = PollPolicy::default();
        for (attempt, full) in [(1, 2), (2, 4), (3, 8), (4, 16), (5, 32), (6, 60), (30, 60)] {
            let full = Duration::from_secs(full);
            for _ in 0..20 {
                let delay = policy.retry_delay(attempt, None);
                assert!(delay >= full / 2 && delay <= full, "{attempt}: {delay:?}");
            }
        }
        assert_eq!(
            policy.retry_delay(1, Some(Duration::from_secs(90))),
            Duration::from_secs(90)
        );
        assert_eq!(
            policy.retry_delay(1, Some(Duration::from_secs(3600))),
            policy.retry_after_max
        );
    }

    #[tokio::test]
    async fn managed_upload_streams_the_file_from_disk() {
        let mock = MockTransport::new();
        mock.on(
            Method::Post,
            "https://proxy.test/v1/uploads/sign",
            200,
            json!({"uploadUrl": "https://put.test/key", "publicUrl": "https://cdn.test/key"}),
        );
        mock.on(Method::Put, "https://put.test/key", 200, json!({}));
        let dir = tempfile_dir();
        let path = dir.join("reference.png");
        std::fs::write(&path, b"png-bytes").unwrap();

        let url = managed_client(&mock)
            .upload_reference(&path, "image/png")
            .await
            .unwrap();

        assert_eq!(url, "https://cdn.test/key");
        let put = mock.last_call().unwrap();
        assert_eq!(put.method, Method::Put);
        match put.body {
            crate::transport::Body::File {
                content_type,
                path: sent,
                len,
            } => {
                assert_eq!(content_type, "image/png");
                assert_eq!(sent, path);
                assert_eq!(len, 9);
            }
            other => panic!("expected a streamed file body, got {other:?}"),
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn oversized_references_are_refused_before_anything_is_sent() {
        // A sparse file: no disk space is used for its apparent size.
        let dir = tempfile_dir();
        let path = dir.join("huge.mp4");
        std::fs::File::create(&path)
            .unwrap()
            .set_len(crate::transport::UPLOAD_BYTES_MAX + 1)
            .unwrap();
        let mock = MockTransport::new();
        let error = managed_client(&mock)
            .upload_reference(&path, "video/mp4")
            .await
            .unwrap_err();
        assert!(
            matches!(error, GenError::UploadTooLarge { .. }),
            "{error:?}"
        );
        let fal = FalAdapter::new(Arc::new(mock.clone()), "key").with_base("https://mockfal");
        assert!(matches!(
            crate::provider::ProviderAdapter::upload(&fal, &path, "video/mp4").await,
            Err(GenError::UploadTooLarge { .. })
        ));
        assert_eq!(mock.call_count(), 0, "no ticket and no upload request");
        std::fs::remove_dir_all(dir).unwrap();
    }

    fn tempfile_dir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "opentake-gen-client-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn split_byok_job_id_validation() {
        assert_eq!(split_byok_job_id("fal::m|r").unwrap(), ("fal", "m|r"));
        assert!(split_byok_job_id("noseparator").is_err());
        assert!(split_byok_job_id("::x").is_err());
    }
}
