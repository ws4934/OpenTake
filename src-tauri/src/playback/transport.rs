//! Loopback JPEG frame transport for streaming playback (#64, #9, #80).
//!
//! The render thread ([`super::engine`]) composites frames and hands each to a
//! [`super::engine::FrameSink`]. [`MjpegSink`] does not encode on the render
//! thread: it drops the RGBA frame into a one-slot, newest-wins mailbox and a
//! per-session encoder thread JPEG-encodes it, stores it as the session's
//! latest frame and emits the matching `playback_frame` event. The WebView
//! requests `GET /frame` with the event's session identity for each event, so
//! an event is only ever emitted after its JPEG is retrievable.
//!
//! Security: the server binds `127.0.0.1:<random port>`, requires a loopback
//! `Host` carrying that port (so a DNS-rebinding page cannot read frames through
//! its own hostname), and rejects any request carrying a non-loopback `Origin`.
//! A frame is only served to a request naming the exact playback session.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};

use axum::extract::{Query, Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use tauri::{AppHandle, Emitter};

use opentake_render::DecodedFrame;

use super::engine::{FrameSink, PlaybackErrorSink, PlaybackFailure, PlaybackFailureCode};
use super::session::PlaybackIdentity;

/// The loopback preview server: a bound port plus the latest encoded frame of
/// the current session. The axum task is spawned on the Tauri async runtime and
/// shuts down when the process exits. Managed as Tauri state so
/// `get_preview_endpoint` and the sink can reach it.
pub struct PreviewServer {
    port: u16,
    latest: LatestFrameStore,
}

/// Shared axum state: the latest encoded frame for the polling `/frame` route.
#[derive(Clone)]
struct ServerState {
    latest: LatestFrameStore,
}

#[derive(Clone, Debug)]
struct LatestFrame {
    identity: PlaybackIdentity,
    frame: i32,
    jpeg: Bytes,
}

#[derive(Clone, Default)]
struct LatestFrameStore(Arc<RwLock<Option<LatestFrame>>>);

impl LatestFrameStore {
    fn publish(&self, identity: PlaybackIdentity, frame: i32, jpeg: Bytes) {
        *self
            .0
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(LatestFrame {
            identity,
            frame,
            jpeg,
        });
    }

    fn lookup(&self, query: &FrameQuery) -> Option<Bytes> {
        let latest = self
            .0
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        latest.as_ref().and_then(|latest| {
            // The store keeps only the newest published frame. The front end
            // issues one `<img>` request per `playback_frame` event; on a slow
            // render (4K/multitrack) the engine publishes faster than the
            // event→IPC→React→DOM→HTTP round trip, so an exact frame+sequence
            // match would 204 for almost every frame and freeze the preview.
            // Serve the newest frame for any query of the same session that is
            // not newer than what has been published (a future frame means the
            // request raced ahead of its own event and must be dropped).
            (latest.identity.project_epoch == query.project_epoch
                && latest.identity.timeline_version == query.timeline_version
                && latest.identity.session_id == query.session_id
                && query.frame <= latest.frame)
                .then(|| latest.jpeg.clone())
        })
    }

    fn clear_session(&self, identity: &PlaybackIdentity) {
        let mut latest = self
            .0
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if latest
            .as_ref()
            .is_some_and(|latest| &latest.identity == identity)
        {
            *latest = None;
        }
    }
}

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct FrameQuery {
    project_epoch: u64,
    timeline_version: u64,
    session_id: String,
    frame: i32,
    /// Sent by the front end with every request; [`LatestFrameStore::lookup`]
    /// serves the newest frame whatever the sequence, so it is only parsed.
    #[allow(dead_code)]
    sequence: u64,
}

impl FrameQuery {
    #[cfg(test)]
    fn new(
        project_epoch: u64,
        timeline_version: u64,
        session_id: impl Into<String>,
        frame: i32,
        sequence: u64,
    ) -> Self {
        Self {
            project_epoch,
            timeline_version,
            session_id: session_id.into(),
            frame,
            sequence,
        }
    }

    fn valid(&self) -> bool {
        self.frame >= 0
            && PlaybackIdentity::new(
                self.project_epoch,
                self.timeline_version,
                self.session_id.clone(),
            )
            .is_ok()
    }
}

/// A failure report deferred while publication was closed.
pub(crate) type HeldReport = Box<dyn FnOnce() + Send>;

#[derive(Default)]
struct GateState {
    open: bool,
    /// Bumped by every close and every invalidation. Work captured under an
    /// older epoch (a frame queued before a pause or a seek) can never publish
    /// after a later reopen.
    epoch: u64,
    /// The first fatal failure (or first non-fatal report) while closed. A cancelled project
    /// transition reopens the gate of a session that kept running and
    /// reports it then; any other reopen (a resume, which retries the
    /// render) or the session's teardown drops it.
    held: Option<(bool, HeldReport)>,
}

#[derive(Clone, Default)]
pub struct PublicationGate(Arc<Mutex<GateState>>);

impl PublicationGate {
    pub fn open() -> Self {
        let gate = Self::default();
        gate.lock().open = true;
        gate
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, GateState> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn close(&self) {
        let mut state = self.lock();
        state.open = false;
        state.epoch = state.epoch.wrapping_add(1);
    }

    /// Reopen for a resume. A failure held while closed is dropped: the
    /// resume retries the render, which reports again if it still fails.
    pub fn reopen(&self) {
        let held = {
            let mut state = self.lock();
            state.open = true;
            state.held.take()
        };
        drop(held);
    }

    /// Reopen after a cancelled project transition, which closed the gate
    /// without pausing the session: report the failure the session raised
    /// meanwhile instead of losing it.
    pub fn reopen_after_transition(&self) {
        let held = {
            let mut state = self.lock();
            state.open = true;
            state.held.take()
        };
        if let Some((_, report)) = held {
            report();
        }
    }

    /// Run `report` now while publication is open; otherwise keep it (the
    /// first fatal one, otherwise the first report) for [`Self::reopen_after_transition`].
    pub(crate) fn report_or_hold(&self, fatal: bool, report: HeldReport) {
        {
            let mut state = self.lock();
            if !state.open {
                if state
                    .held
                    .as_ref()
                    .is_none_or(|(held_fatal, _)| fatal && !held_fatal)
                {
                    state.held = Some((fatal, report));
                }
                return;
            }
        }
        report();
    }

    /// Retire every frame captured so far without closing publication: the
    /// playhead moved, so queued and in-flight frames are stale while the
    /// frames rendered after this call publish normally.
    pub fn invalidate(&self) {
        let mut state = self.lock();
        state.epoch = state.epoch.wrapping_add(1);
    }

    #[cfg(test)]
    pub(crate) fn is_open(&self) -> bool {
        self.lock().open
    }

    /// The epoch to tag work with, or `None` while publication is closed.
    fn open_epoch(&self) -> Option<u64> {
        let state = self.lock();
        state.open.then_some(state.epoch)
    }

    fn with_epoch<T>(&self, epoch: u64, publish: impl FnOnce() -> T) -> Option<T> {
        let state = self.lock();
        if !state.open || state.epoch != epoch {
            return None;
        }
        Some(publish())
    }
}

/// Encodes one RGBA frame into the provided (reused) buffer.
pub(crate) type EncodeFn = fn(&DecodedFrame, &mut Vec<u8>) -> Result<(), String>;

/// Receives each committed publication; production emits `playback_frame`.
pub type PublishFn = Arc<dyn Fn(PlaybackFramePublication) + Send + Sync>;

impl PreviewServer {
    /// Start the frame server on a random loopback port. Must run inside the
    /// Tauri async runtime (call via `tauri::async_runtime::block_on` in setup).
    pub async fn start() -> Result<Arc<Self>, String> {
        let latest = LatestFrameStore::default();

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .map_err(|e| format!("preview server bind: {e}"))?;
        let port = listener
            .local_addr()
            .map_err(|e| format!("preview server local_addr: {e}"))?
            .port();

        let state = ServerState {
            latest: latest.clone(),
        };
        tauri::async_runtime::spawn(async move {
            let app = axum::Router::new()
                .route("/frame", axum::routing::get(frame_handler))
                .layer(axum::middleware::from_fn(
                    move |request: Request, next: Next| async move {
                        if !request_is_local(request.headers(), port) {
                            return (StatusCode::FORBIDDEN, "non-loopback preview request denied")
                                .into_response();
                        }
                        next.run(request).await
                    },
                ))
                .with_state(state);
            if let Err(e) = axum::serve(listener, app).await {
                eprintln!("[preview] server error: {e}");
            }
        });

        Ok(Arc::new(Self { port, latest }))
    }

    /// The single-frame poll URL (`GET /frame` -> latest JPEG). WKWebView's
    /// secure `tauri://` context blocks plain `ws://` to loopback as mixed
    /// content, while a passive `<img>` load over loopback http is allowed. The
    /// `playback_frame` event drives one `<img>` reload per published frame.
    pub fn endpoint_frame(&self) -> String {
        format!("http://127.0.0.1:{}/frame", self.port)
    }

    /// A frame sink for one playback session. Frames are encoded on a
    /// dedicated thread; each committed frame is handed to `on_publish`.
    pub fn sink(
        &self,
        identity: PlaybackIdentity,
        gate: PublicationGate,
        last_frame: i32,
        on_publish: PublishFn,
    ) -> MjpegSink {
        self.sink_with_encoder(
            identity,
            gate,
            last_frame,
            on_publish,
            crate::jpeg::encode_rgba_jpeg,
        )
    }

    pub(crate) fn sink_with_encoder(
        &self,
        identity: PlaybackIdentity,
        gate: PublicationGate,
        last_frame: i32,
        on_publish: PublishFn,
        encode: EncodeFn,
    ) -> MjpegSink {
        MjpegSink::spawn(
            EncodedFramePublication {
                identity,
                gate,
                latest: self.latest.clone(),
                sequence: Arc::new(AtomicU64::new(0)),
                last_frame,
            },
            on_publish,
            encode,
        )
    }

    pub fn clear_session(&self, identity: &PlaybackIdentity) {
        self.latest.clear_session(identity);
    }
}

/// `Host` guard against DNS rebinding: the request must name the loopback
/// interface and this server's port, exactly as the WebView's `<img>` does.
fn host_is_local(host: &str, port: u16) -> bool {
    let Some((name, host_port)) = host.rsplit_once(':') else {
        return false;
    };
    host_port.parse::<u16>().ok() == Some(port)
        && (name == "127.0.0.1" || name == "[::1]" || name.eq_ignore_ascii_case("localhost"))
}

fn request_is_local(headers: &HeaderMap, port: u16) -> bool {
    let host_ok = headers
        .get(axum::http::header::HOST)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|host| host_is_local(host, port));
    host_ok && origin_is_allowed(headers)
}

/// `Origin` defence-in-depth: allow requests with no `Origin` (a plain `<img>`
/// load omits it) or a loopback / Tauri-webview origin; reject anything else.
fn origin_is_allowed(headers: &HeaderMap) -> bool {
    match headers.get(axum::http::header::ORIGIN) {
        None => true,
        Some(value) => match value.to_str() {
            Ok(origin) => origin_value_is_allowed(origin),
            Err(_) => false,
        },
    }
}

fn origin_value_is_allowed(origin: &str) -> bool {
    let Ok(uri) = origin.parse::<axum::http::Uri>() else {
        return false;
    };
    if uri.path() != "/" || uri.query().is_some() {
        return false;
    }
    let (Some(scheme), Some(host)) = (uri.scheme_str(), uri.host()) else {
        return false;
    };
    matches!(
        (scheme, host),
        ("http", "127.0.0.1")
            | ("http", "localhost")
            | ("https", "localhost")
            | ("tauri", "localhost")
            | ("http", "tauri.localhost")
            | ("http", "[::1]")
            | ("https", "[::1]")
    )
}

/// `/frame`: one session-scoped composited JPEG. The preview requests the exact
/// project epoch, timeline version, session id, frame, and publication sequence;
/// a mismatch returns 204. WKWebView permits this passive loopback image request
/// while blocking plain `ws://` in the secure `tauri://` context.
async fn frame_handler(
    State(state): State<ServerState>,
    Query(query): Query<FrameQuery>,
) -> Response {
    if !query.valid() {
        return (StatusCode::NO_CONTENT, "").into_response();
    }
    match state.latest.lookup(&query) {
        Some(jpeg) => (
            [
                (axum::http::header::CONTENT_TYPE, "image/jpeg".to_string()),
                (
                    axum::http::header::CACHE_CONTROL,
                    "no-store, no-cache, must-revalidate".to_string(),
                ),
            ],
            jpeg,
        )
            .into_response(),
        None => (StatusCode::NO_CONTENT, "").into_response(),
    }
}

/// The commit coordinator shared by the encoder thread and the exact-frame HTTP
/// store: a frame becomes observable (stored, then announced) in one step, and
/// only while the session's publication gate is still open under the epoch the
/// frame was queued in.
#[derive(Clone)]
pub struct EncodedFramePublication {
    identity: PlaybackIdentity,
    gate: PublicationGate,
    latest: LatestFrameStore,
    sequence: Arc<AtomicU64>,
    last_frame: i32,
}

impl EncodedFramePublication {
    /// Store `jpeg` (when present) as the session's latest frame and return the
    /// event payload announcing it. `None` pixels announce a terminal tick that
    /// has no new image: `/frame` keeps serving the last good frame and the
    /// front end's exhausted-terminal path ends the transport.
    fn commit(
        &self,
        epoch: u64,
        frame: i32,
        jpeg: Option<Bytes>,
    ) -> Option<PlaybackFramePublication> {
        self.gate.with_epoch(epoch, || {
            let sequence = self.sequence.fetch_add(1, Ordering::AcqRel) + 1;
            let terminal = frame >= self.last_frame;
            if let Some(jpeg) = jpeg {
                self.latest.publish(self.identity.clone(), frame, jpeg);
            }
            PlaybackFramePublication::new(self.identity.clone(), frame, sequence, terminal)
        })
    }
}

enum EncodeJob {
    Frame { frame: i32, image: DecodedFrame },
    Terminal { frame: i32 },
}

struct QueuedJob {
    epoch: u64,
    job: EncodeJob,
}

#[derive(Default)]
struct EncoderSlot {
    job: Option<QueuedJob>,
    shutdown: bool,
}

#[derive(Default)]
struct EncoderMailbox {
    slot: Mutex<EncoderSlot>,
    ready: Condvar,
}

impl EncoderMailbox {
    fn lock(&self) -> std::sync::MutexGuard<'_, EncoderSlot> {
        self.slot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Newest wins: an unencoded frame is replaced rather than queued, so a
    /// slow encode never builds latency or back-pressures the render thread.
    fn put(&self, job: QueuedJob) {
        let mut slot = self.lock();
        if slot.shutdown {
            return;
        }
        slot.job = Some(job);
        drop(slot);
        self.ready.notify_one();
    }

    fn take(&self) -> Option<QueuedJob> {
        let mut slot = self.lock();
        loop {
            if slot.shutdown {
                return None;
            }
            if let Some(job) = slot.job.take() {
                return Some(job);
            }
            slot = self
                .ready
                .wait(slot)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    fn shutdown(&self) {
        let mut slot = self.lock();
        slot.shutdown = true;
        slot.job = None;
        drop(slot);
        self.ready.notify_all();
    }
}

/// Stops the encoder thread once the last sink clone is dropped.
struct EncoderWorker {
    mailbox: Arc<EncoderMailbox>,
}

impl Drop for EncoderWorker {
    fn drop(&mut self) {
        self.mailbox.shutdown();
    }
}

/// Encodes and commits frames for one session. Runs on the encoder thread, or
/// synchronously through [`MjpegSink::publish_now`].
struct FramePublisher {
    publication: EncodedFramePublication,
    on_publish: PublishFn,
    encode: EncodeFn,
    #[cfg(test)]
    before_encode: Mutex<Option<Box<dyn FnOnce() + Send>>>,
}

impl FramePublisher {
    fn publish(
        &self,
        queued: QueuedJob,
        scratch: &mut Vec<u8>,
    ) -> Option<PlaybackFramePublication> {
        let (frame, jpeg) = match queued.job {
            EncodeJob::Frame { frame, image } => {
                #[cfg(test)]
                {
                    let hook = self.before_encode.lock().unwrap().take();
                    if let Some(hook) = hook {
                        hook();
                    }
                }
                scratch.clear();
                if let Err(error) = (self.encode)(&image, scratch) {
                    eprintln!("[preview] frame {frame} {error}");
                    return None;
                }
                (frame, Some(Bytes::copy_from_slice(scratch)))
            }
            EncodeJob::Terminal { frame } => (frame, None),
        };
        let publication = self.publication.commit(queued.epoch, frame, jpeg)?;
        (self.on_publish)(publication.clone());
        Some(publication)
    }
}

/// A [`FrameSink`] that JPEG-encodes composited frames off the render thread
/// and publishes them to the loopback `/frame` route. Dropping an unencoded
/// frame in favour of a newer one is intentional: playback never blocks on
/// the transport.
#[derive(Clone)]
pub struct MjpegSink {
    publisher: Arc<FramePublisher>,
    mailbox: Arc<EncoderMailbox>,
    _worker: Arc<EncoderWorker>,
}

impl MjpegSink {
    fn spawn(
        publication: EncodedFramePublication,
        on_publish: PublishFn,
        encode: EncodeFn,
    ) -> Self {
        let publisher = Arc::new(FramePublisher {
            publication,
            on_publish,
            encode,
            #[cfg(test)]
            before_encode: Mutex::new(None),
        });
        let mailbox = Arc::new(EncoderMailbox::default());
        let worker_publisher = Arc::clone(&publisher);
        let worker_mailbox = Arc::clone(&mailbox);
        if let Err(error) = std::thread::Builder::new()
            .name("opentake-playback-encode".to_string())
            .spawn(move || {
                let mut scratch = Vec::new();
                while let Some(job) = worker_mailbox.take() {
                    worker_publisher.publish(job, &mut scratch);
                }
            })
        {
            // Without an encoder no frame can be published; the session still
            // runs and the preview keeps the idle still.
            eprintln!("[preview] spawn frame encoder: {error}");
            mailbox.shutdown();
        }
        Self {
            publisher,
            _worker: Arc::new(EncoderWorker {
                mailbox: Arc::clone(&mailbox),
            }),
            mailbox,
        }
    }

    /// Encode and commit `image` on the calling thread, bypassing the mailbox.
    /// Integration tests use this to observe each publication deterministically.
    pub fn publish_now(
        &self,
        frame: i32,
        image: &DecodedFrame,
    ) -> Option<PlaybackFramePublication> {
        let epoch = self.publisher.publication.gate.open_epoch()?;
        self.publisher.publish(
            QueuedJob {
                epoch,
                job: EncodeJob::Frame {
                    frame,
                    image: image.clone(),
                },
            },
            &mut Vec::new(),
        )
    }

    fn enqueue(&self, job: EncodeJob) {
        if let Some(epoch) = self.publisher.publication.gate.open_epoch() {
            self.mailbox.put(QueuedJob { epoch, job });
        }
    }
}

impl FrameSink for MjpegSink {
    fn push_frame(&self, frame: i32, image: DecodedFrame) {
        self.enqueue(EncodeJob::Frame { frame, image });
    }

    fn push_terminal(&self, frame: i32) {
        self.enqueue(EncodeJob::Terminal { frame });
    }

    fn invalidate(&self) {
        // Called on the render thread, so it is ordered before that thread's
        // next push: the frame in the mailbox and the one being encoded fail
        // their epoch check at commit, the next pushed frame passes.
        self.publisher.publication.gate.invalidate();
    }
}

/// Playhead frame number broadcast to the front end, so it can move the
/// playhead / timecode and request the matching JPEG from `/frame`.
#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlaybackFramePublication {
    project_epoch: u64,
    timeline_version: u64,
    session_id: String,
    frame: i32,
    sequence: u64,
    terminal: bool,
}

impl PlaybackFramePublication {
    fn new(identity: PlaybackIdentity, frame: i32, sequence: u64, terminal: bool) -> Self {
        Self {
            project_epoch: identity.project_epoch,
            timeline_version: identity.timeline_version,
            session_id: identity.session_id,
            frame,
            sequence,
            terminal,
        }
    }

    pub fn frame(&self) -> i32 {
        self.frame
    }

    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn terminal(&self) -> bool {
        self.terminal
    }
}

/// Emits each committed publication as a Tauri `playback_frame` event.
pub fn tauri_frame_publisher(app: AppHandle) -> PublishFn {
    Arc::new(move |publication| {
        let _ = app.emit("playback_frame", publication);
    })
}

/// `playback_error` payload: which session failed, where, and why. Messages
/// name media by asset id, never by absolute path.
#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlaybackErrorEvent {
    project_epoch: u64,
    timeline_version: u64,
    session_id: String,
    frame: i32,
    code: PlaybackFailureCode,
    message: String,
    fatal: bool,
}

impl PlaybackErrorEvent {
    fn new(identity: &PlaybackIdentity, failure: PlaybackFailure) -> Self {
        Self {
            project_epoch: identity.project_epoch,
            timeline_version: identity.timeline_version,
            session_id: identity.session_id.clone(),
            frame: failure.frame,
            code: failure.code,
            message: failure.message,
            fatal: failure.fatal,
        }
    }
}

/// A [`PlaybackErrorSink`] that emits a Tauri `playback_error` event while
/// the session's publication gate is open. A paused, stopped or replaced
/// session reports nothing: its failures (for example a decode cancelled by
/// the teardown itself) are not the running transport's to show. A failure
/// raised during a project transition that is then cancelled is reported
/// when the gate reopens.
pub struct TauriPlaybackErrorEmitter {
    app: AppHandle,
    identity: PlaybackIdentity,
    gate: PublicationGate,
}

impl TauriPlaybackErrorEmitter {
    pub fn new(app: AppHandle, identity: PlaybackIdentity, gate: PublicationGate) -> Self {
        Self {
            app,
            identity,
            gate,
        }
    }
}

impl PlaybackErrorSink for TauriPlaybackErrorEmitter {
    fn report(&self, failure: PlaybackFailure) {
        let fatal = failure.fatal;
        let event = PlaybackErrorEvent::new(&self.identity, failure);
        let app = self.app.clone();
        self.gate.report_or_hold(
            fatal,
            Box::new(move || {
                let _ = app.emit("playback_error", event);
            }),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{HeaderMap, HeaderValue};
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    fn identity(epoch: u64, version: u64, session: &str) -> PlaybackIdentity {
        PlaybackIdentity::new(epoch, version, session).expect("valid identity")
    }

    fn recording_publisher() -> (PublishFn, mpsc::Receiver<PlaybackFramePublication>) {
        let (tx, rx) = mpsc::channel();
        let tx = Mutex::new(tx);
        (
            Arc::new(move |publication| {
                let _ = tx
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .send(publication);
            }),
            rx,
        )
    }

    fn test_sink(
        latest: &LatestFrameStore,
        identity: PlaybackIdentity,
        gate: PublicationGate,
        last_frame: i32,
        on_publish: PublishFn,
        encode: EncodeFn,
    ) -> MjpegSink {
        MjpegSink::spawn(
            EncodedFramePublication {
                identity,
                gate,
                latest: latest.clone(),
                sequence: Arc::new(AtomicU64::new(0)),
                last_frame,
            },
            on_publish,
            encode,
        )
    }

    fn solid(width: u32, height: u32) -> DecodedFrame {
        DecodedFrame::new(
            width,
            height,
            vec![200; (width * height * 4) as usize],
            false,
        )
    }

    fn slow_encode(frame: &DecodedFrame, out: &mut Vec<u8>) -> Result<(), String> {
        std::thread::sleep(Duration::from_millis(150));
        crate::jpeg::encode_rgba_jpeg(frame, out)
    }

    fn hold_first_encode(
        sink: &MjpegSink,
    ) -> (
        std::sync::mpsc::Receiver<()>,
        std::sync::mpsc::SyncSender<()>,
    ) {
        let (started_tx, started) = std::sync::mpsc::sync_channel(1);
        let (release, release_rx) = std::sync::mpsc::sync_channel(1);
        *sink.publisher.before_encode.lock().unwrap() = Some(Box::new(move || {
            started_tx.send(()).expect("test waits for the encoder");
            release_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("test releases the held encoder job");
        }));
        (started, release)
    }

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.insert(*name, HeaderValue::from_str(value).unwrap());
        }
        map
    }

    #[test]
    fn origin_guard_allows_missing_and_loopback_origins() {
        let empty = HeaderMap::new();
        assert!(origin_is_allowed(&empty), "no Origin (plain <img>) allowed");

        for ok in [
            "http://127.0.0.1:1420",
            "http://localhost:1420",
            "tauri://localhost",
            "http://tauri.localhost",
        ] {
            assert!(
                origin_is_allowed(&headers(&[("origin", ok)])),
                "{ok} should be allowed"
            );
        }
    }

    #[test]
    fn origin_guard_rejects_remote_origin() {
        assert!(!origin_is_allowed(&headers(&[(
            "origin",
            "http://evil.example.com"
        )])));
    }

    #[test]
    fn host_guard_accepts_only_loopback_names_with_the_server_port() {
        for ok in [
            "127.0.0.1:4100",
            "localhost:4100",
            "LOCALHOST:4100",
            "[::1]:4100",
        ] {
            assert!(request_is_local(&headers(&[("host", ok)]), 4100), "{ok}");
        }
        for bad in [
            "attacker.example:4100",
            "127.0.0.1.attacker.example:4100",
            "127.0.0.1:4101",
            "127.0.0.1",
            "localhost",
            "",
        ] {
            assert!(!request_is_local(&headers(&[("host", bad)]), 4100), "{bad}");
        }
        assert!(!request_is_local(&HeaderMap::new(), 4100), "missing Host");
        assert!(!request_is_local(
            &headers(&[
                ("host", "127.0.0.1:4100"),
                ("origin", "http://evil.example")
            ]),
            4100
        ));
    }

    #[test]
    fn playhead_event_carries_session_revision_sequence_and_terminal() {
        let dto = PlaybackFramePublication::new(identity(7, 11, "session-42"), 123, 9, true);

        assert_eq!(
            serde_json::to_value(dto).expect("serialize"),
            serde_json::json!({
                "projectEpoch": 7,
                "timelineVersion": 11,
                "sessionId": "session-42",
                "frame": 123,
                "sequence": 9,
                "terminal": true,
            })
        );
    }

    #[test]
    fn playback_error_event_is_camel_case_and_session_scoped() {
        let event = PlaybackErrorEvent::new(
            &identity(3, 4, "session-err"),
            PlaybackFailure {
                frame: 17,
                code: PlaybackFailureCode::VideoDecode,
                message: "clip-1 decode failed".to_string(),
                fatal: true,
            },
        );
        assert_eq!(
            serde_json::to_value(event).expect("serialize"),
            serde_json::json!({
                "projectEpoch": 3,
                "timelineVersion": 4,
                "sessionId": "session-err",
                "frame": 17,
                "code": "videoDecode",
                "message": "clip-1 decode failed",
                "fatal": true,
            })
        );
    }

    #[test]
    fn frame_route_never_serves_another_session_latest() {
        let latest = LatestFrameStore::default();
        let identity = identity(3, 5, "current");
        latest.publish(identity.clone(), 18, Bytes::from_static(b"jpeg"));

        // Session, project epoch and timeline version stay hard boundaries.
        assert!(latest
            .lookup(&FrameQuery::new(3, 5, "stale", 18, 4))
            .is_none());
        assert!(latest
            .lookup(&FrameQuery::new(2, 5, "current", 18, 4))
            .is_none());
        assert!(latest
            .lookup(&FrameQuery::new(3, 4, "current", 18, 4))
            .is_none());
        // A request that raced ahead of its own publication is dropped.
        assert!(latest
            .lookup(&FrameQuery::new(3, 5, "current", 19, 4))
            .is_none());
        // Exact frame matches serve; a stale sequence within the same session
        // and an older frame both resolve to the newest published frame, so a
        // slow front-end round trip cannot freeze the preview on a fast engine.
        assert_eq!(
            latest.lookup(&FrameQuery::new(3, 5, "current", 18, 4)),
            Some(Bytes::from_static(b"jpeg"))
        );
        assert_eq!(
            latest.lookup(&FrameQuery::new(3, 5, "current", 18, 3)),
            Some(Bytes::from_static(b"jpeg"))
        );
        assert_eq!(
            latest.lookup(&FrameQuery::new(3, 5, "current", 17, 1)),
            Some(Bytes::from_static(b"jpeg"))
        );
    }

    #[test]
    fn slow_consumer_keeps_receiving_the_newest_published_frame() {
        // A 4K/multitrack render that falls behind publishes several frames
        // before the front end's request for the first one arrives. Every
        // request must still resolve to the current newest frame, otherwise the
        // preview freezes on the idle still (the reported bug).
        let latest = LatestFrameStore::default();
        let identity = identity(1, 2, "session-9");
        for frame in 0..60 {
            latest.publish(identity.clone(), frame, Bytes::from_static(b"jpeg"));
        }
        for requested in 0..60 {
            assert_eq!(
                latest.lookup(&FrameQuery::new(
                    1,
                    2,
                    "session-9",
                    requested,
                    requested as u64 + 1
                )),
                Some(Bytes::from_static(b"jpeg")),
                "stale frame {requested} must resolve to the newest published frame"
            );
        }
        // Future frames still race ahead of publication and are rejected.
        assert!(latest
            .lookup(&FrameQuery::new(1, 2, "session-9", 60, 61))
            .is_none());
    }

    #[test]
    fn push_frame_hands_off_without_waiting_for_a_slow_encoder() {
        let latest = LatestFrameStore::default();
        let (on_publish, published) = recording_publisher();
        let sink = test_sink(
            &latest,
            identity(1, 1, "slow-encoder"),
            PublicationGate::open(),
            1_000,
            on_publish,
            slow_encode,
        );
        let frames: Vec<DecodedFrame> = (0..8).map(|_| solid(1280, 720)).collect();
        let mut handoffs = Vec::new();
        for (index, image) in frames.into_iter().enumerate() {
            let start = Instant::now();
            sink.push_frame(index as i32, image);
            handoffs.push(start.elapsed());
        }
        handoffs.sort();
        // Each encode takes 150 ms; a synchronous encode would make every
        // hand-off at least that long.
        assert!(
            handoffs[handoffs.len() / 2] < Duration::from_millis(1),
            "median render-thread hand-off must stay under 1 ms: {handoffs:?}"
        );
        assert!(handoffs[handoffs.len() - 1] < Duration::from_millis(50));

        // Newest wins: the last frame is always published, stale ones dropped.
        let mut frames_seen = Vec::new();
        while let Ok(publication) = published.recv_timeout(Duration::from_secs(2)) {
            frames_seen.push(publication.frame());
            if publication.frame() == 7 {
                break;
            }
        }
        assert_eq!(frames_seen.last(), Some(&7));
        assert!(
            frames_seen.len() < 8,
            "stale frames are skipped: {frames_seen:?}"
        );
    }

    #[test]
    fn every_announced_frame_is_already_retrievable() {
        let latest = LatestFrameStore::default();
        let session = identity(4, 6, "announce-after-store");
        let (tx, rx) = mpsc::channel();
        let tx = Mutex::new(tx);
        let lookup_store = latest.clone();
        let sink = test_sink(
            &latest,
            session.clone(),
            PublicationGate::open(),
            40,
            Arc::new(move |publication: PlaybackFramePublication| {
                let served = lookup_store
                    .lookup(&FrameQuery::new(
                        4,
                        6,
                        "announce-after-store",
                        publication.frame(),
                        publication.sequence(),
                    ))
                    .is_some();
                let _ = tx.lock().unwrap().send((publication.frame(), served));
            }),
            crate::jpeg::encode_rgba_jpeg,
        );
        for frame in 0..=40 {
            sink.push_frame(frame, solid(8, 8));
            std::thread::sleep(Duration::from_millis(1));
        }
        let mut last = -1;
        while last != 40 {
            let (frame, served) = rx
                .recv_timeout(Duration::from_secs(2))
                .expect("terminal frame is published");
            assert!(served, "frame {frame} was announced before it was stored");
            last = frame;
        }
    }

    #[test]
    fn frames_queued_before_a_pause_never_publish_after_resume() {
        let latest = LatestFrameStore::default();
        let (on_publish, published) = recording_publisher();
        let gate = PublicationGate::open();
        let sink = test_sink(
            &latest,
            identity(2, 2, "pause-epoch"),
            gate.clone(),
            1_000,
            on_publish,
            crate::jpeg::encode_rgba_jpeg,
        );
        let (started, release) = hold_first_encode(&sink);
        // Frame 1 occupies the encoder; frame 2 waits in the mailbox.
        sink.push_frame(1, solid(4, 4));
        started
            .recv_timeout(Duration::from_secs(5))
            .expect("first frame is held in the encoder");
        sink.push_frame(2, solid(4, 4));
        gate.close();
        gate.reopen();
        sink.push_frame(30, solid(4, 4));
        release.send(()).unwrap();

        let publication = published
            .recv_timeout(Duration::from_secs(5))
            .expect("replacement frame is published");
        assert_eq!(publication.frame(), 30, "pre-pause frames must not publish");
        assert!(
            published.try_recv().is_err(),
            "only the replacement is queued"
        );
    }

    #[test]
    fn frames_queued_before_a_seek_never_publish_after_it() {
        let latest = LatestFrameStore::default();
        let (on_publish, published) = recording_publisher();
        let gate = PublicationGate::open();
        let sink = test_sink(
            &latest,
            identity(2, 3, "seek-epoch"),
            gate.clone(),
            1_000,
            on_publish,
            crate::jpeg::encode_rgba_jpeg,
        );
        let (started, release) = hold_first_encode(&sink);
        // Frame 100 occupies the encoder; frame 101 waits in the mailbox.
        sink.push_frame(100, solid(4, 4));
        started
            .recv_timeout(Duration::from_secs(5))
            .expect("first frame is held in the encoder");
        sink.push_frame(101, solid(4, 4));
        // The render thread consumes a seek to 500 while playing.
        sink.invalidate();
        assert!(gate.is_open(), "a seek keeps publication open");
        sink.push_frame(500, solid(4, 4));
        sink.push_frame(501, solid(4, 4));
        release.send(()).unwrap();

        let publication = published
            .recv_timeout(Duration::from_secs(5))
            .expect("replacement frame is published");
        assert!(
            publication.frame() >= 500,
            "pre-seek frame published after the seek: {}",
            publication.frame()
        );
        assert_eq!(publication.frame(), 501);
        assert!(
            published.try_recv().is_err(),
            "only the replacement is queued"
        );
    }

    /// A report that records `message` into `reports` when it runs.
    fn recording_report(
        reports: &Arc<Mutex<Vec<&'static str>>>,
        message: &'static str,
    ) -> HeldReport {
        let reports = Arc::clone(reports);
        Box::new(move || {
            reports
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(message)
        })
    }

    fn reported(reports: &Arc<Mutex<Vec<&'static str>>>) -> Vec<&'static str> {
        reports
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    #[test]
    fn playback_errors_are_dropped_once_the_session_gate_closes() {
        let gate = PublicationGate::open();
        let reports = Arc::new(Mutex::new(Vec::new()));
        gate.report_or_hold(true, recording_report(&reports, "open"));
        assert_eq!(reported(&reports), ["open"]);
        gate.close();
        gate.report_or_hold(true, recording_report(&reports, "while paused"));
        assert_eq!(
            reported(&reports),
            ["open"],
            "a paused or torn-down session reports nothing"
        );
        // A resume retries the render instead of replaying the old failure.
        gate.reopen();
        assert_eq!(reported(&reports), ["open"]);
        gate.report_or_hold(true, recording_report(&reports, "resumed"));
        assert_eq!(reported(&reports), ["open", "resumed"]);
    }

    #[test]
    fn a_failure_during_a_cancelled_project_transition_is_reported_when_it_reopens() {
        let gate = PublicationGate::open();
        let reports = Arc::new(Mutex::new(Vec::new()));
        // A project transition closes the gate without pausing the session,
        // whose render then genuinely fails.
        gate.close();
        gate.report_or_hold(true, recording_report(&reports, "first failure"));
        gate.report_or_hold(true, recording_report(&reports, "second failure"));
        assert!(reported(&reports).is_empty());
        gate.reopen_after_transition();
        assert_eq!(reported(&reports), ["first failure"]);
        gate.reopen_after_transition();
        assert_eq!(reported(&reports), ["first failure"], "reported once");
    }

    #[test]
    fn a_held_fatal_failure_replaces_an_earlier_nonfatal_report() {
        let gate = PublicationGate::open();
        let reports = Arc::new(Mutex::new(Vec::new()));
        gate.close();
        gate.report_or_hold(false, recording_report(&reports, "audio warning"));
        gate.report_or_hold(true, recording_report(&reports, "fatal render failure"));
        gate.report_or_hold(false, recording_report(&reports, "later warning"));
        gate.reopen_after_transition();
        assert_eq!(reported(&reports), ["fatal render failure"]);
    }

    #[test]
    fn old_session_mailbox_frame_is_never_published_after_a_new_session_starts() {
        let latest = LatestFrameStore::default();
        let old_identity = identity(5, 1, "old-session");
        let (old_publish, old_published) = recording_publisher();
        let old_gate = PublicationGate::open();
        let old_sink = test_sink(
            &latest,
            old_identity.clone(),
            old_gate.clone(),
            100,
            old_publish,
            slow_encode,
        );
        old_sink.push_frame(10, solid(4, 4));
        std::thread::sleep(Duration::from_millis(20));
        old_sink.push_frame(11, solid(4, 4));

        // Session teardown closes the old gate before the replacement starts.
        old_gate.close();
        drop(old_sink);
        let new_identity = identity(5, 1, "new-session");
        let (new_publish, new_published) = recording_publisher();
        let new_sink = test_sink(
            &latest,
            new_identity.clone(),
            PublicationGate::open(),
            100,
            new_publish,
            crate::jpeg::encode_rgba_jpeg,
        );
        new_sink.push_frame(3, solid(4, 4));
        assert_eq!(
            new_published
                .recv_timeout(Duration::from_secs(2))
                .expect("new session publishes")
                .frame(),
            3
        );
        std::thread::sleep(Duration::from_millis(300));
        assert!(old_published.try_recv().is_err());
        assert!(latest
            .lookup(&FrameQuery::new(5, 1, "old-session", 0, 0))
            .is_none());
        assert!(latest
            .lookup(&FrameQuery::new(5, 1, "new-session", 3, 1))
            .is_some());
    }

    #[test]
    fn terminal_without_pixels_is_announced_but_keeps_the_last_good_jpeg() {
        let latest = LatestFrameStore::default();
        let (on_publish, published) = recording_publisher();
        let sink = test_sink(
            &latest,
            identity(8, 8, "terminal-failed"),
            PublicationGate::open(),
            9,
            on_publish,
            crate::jpeg::encode_rgba_jpeg,
        );
        let first = sink
            .publish_now(8, &solid(4, 4))
            .expect("frame 8 published");
        assert!(!first.terminal());
        let _ = published.recv_timeout(Duration::from_secs(1));
        sink.push_terminal(9);
        let terminal = published
            .recv_timeout(Duration::from_secs(2))
            .expect("terminal tick published");
        assert_eq!((terminal.frame(), terminal.terminal()), (9, true));
        assert_eq!(terminal.sequence(), first.sequence() + 1);
        // The terminal frame itself has no JPEG, so the front end's exhausted
        // path runs; earlier frames still resolve to the last good JPEG.
        assert!(latest
            .lookup(&FrameQuery::new(
                8,
                8,
                "terminal-failed",
                9,
                terminal.sequence()
            ))
            .is_none());
        assert!(latest
            .lookup(&FrameQuery::new(
                8,
                8,
                "terminal-failed",
                8,
                first.sequence()
            ))
            .is_some());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn preview_server_rejects_rebinding_hosts_and_serves_no_stream_routes() {
        use std::io::{Read, Write};

        let server = PreviewServer::start().await.expect("start preview server");
        let address = server
            .endpoint_frame()
            .strip_prefix("http://")
            .and_then(|rest| rest.strip_suffix("/frame"))
            .expect("loopback endpoint")
            .to_string();
        let status = |path: &str, host: &str, extra: &str| -> u16 {
            let mut stream = std::net::TcpStream::connect(&address).expect("connect");
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .expect("read timeout");
            write!(
                stream,
                "GET {path} HTTP/1.1\r\nHost: {host}\r\n{extra}Connection: close\r\n\r\n"
            )
            .expect("write request");
            let mut response = Vec::new();
            stream.read_to_end(&mut response).expect("read response");
            String::from_utf8_lossy(&response)
                .split_whitespace()
                .nth(1)
                .expect("status")
                .parse()
                .expect("numeric status")
        };
        let port = address.rsplit(':').next().unwrap();
        let frame = "/frame?projectEpoch=1&timelineVersion=1&sessionId=s&frame=0&sequence=1";

        assert_eq!(status(frame, &format!("attacker.example:{port}"), ""), 403);
        assert_eq!(status(frame, &address, ""), 204);
        assert_eq!(status(frame, &format!("localhost:{port}"), ""), 204);
        assert_eq!(
            status(frame, &address, "Origin: http://attacker.example\r\n"),
            403
        );
        assert_eq!(status("/stream", &address, ""), 404);
        assert_eq!(status("/ws", &address, ""), 404);
        assert_eq!(
            status("/stream", &format!("attacker.example:{port}"), ""),
            403
        );
    }
}
