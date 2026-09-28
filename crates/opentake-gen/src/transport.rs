//! HTTP transport abstraction. Every network call in this crate flows through
//! `HttpTransport`, never `reqwest` directly. Production uses `ReqwestTransport`;
//! tests use `MockTransport`, which serves canned responses keyed by request and
//! makes the whole suite fully offline — no socket is ever opened.
//!
//! `ReqwestTransport` bounds every request: connections time out, each request
//! has a total deadline (stretched for file uploads by their size), local
//! files are streamed from disk instead of being read into memory, and
//! response bodies are read chunk by chunk up to a ceiling.

use crate::error::GenError;
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Ceiling for a buffered control-plane response (JSON submit, status and
/// upload replies). A larger body fails as a transport error.
pub const CONTROL_RESPONSE_BYTES_MAX: u64 = 8 * 1024 * 1024;

/// Ceiling for synchronous media responses (OpenAI images and speech,
/// ElevenLabs audio) that are buffered and re-encoded as `data:` URLs. Base64
/// grows the data by a third, so the resulting URL stays within the desktop
/// downloader's 512 MiB `data:` URL limit.
pub const MEDIA_RESPONSE_BYTES_MAX: u64 = 256 * 1024 * 1024;

/// Largest local file a reference upload streams. Checked before the file is
/// opened for reading.
pub const UPLOAD_BYTES_MAX: u64 = 1024 * 1024 * 1024;

/// Total deadline for a synchronous media request (OpenAI images and speech,
/// ElevenLabs audio), which returns only once the media is generated.
pub const MEDIA_REQUEST_TIMEOUT: Duration = Duration::from_secs(10 * 60);

/// HTTP method, kept minimal to what the adapters and client need.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Get,
    Post,
    Put,
}

impl Method {
    pub fn as_str(self) -> &'static str {
        match self {
            Method::Get => "GET",
            Method::Post => "POST",
            Method::Put => "PUT",
        }
    }
}

/// Request body: JSON, raw bytes (with content-type), a local file, or none.
#[derive(Debug, Clone)]
pub enum Body {
    Empty,
    Json(serde_json::Value),
    Bytes {
        content_type: String,
        data: Vec<u8>,
    },
    /// A local file streamed from disk with an exact `Content-Length`. Only
    /// the metadata lives in the request; the bytes are read while sending.
    File {
        content_type: String,
        path: PathBuf,
        len: u64,
    },
}

/// A transport-agnostic HTTP request.
#[derive(Debug, Clone)]
pub struct HttpRequest {
    pub method: Method,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Body,
    /// Response body ceiling; `None` means [`CONTROL_RESPONSE_BYTES_MAX`].
    pub max_response_bytes: Option<u64>,
    /// Total deadline replacing the transport's control-plane default.
    pub timeout: Option<Duration>,
}

impl HttpRequest {
    pub fn new(method: Method, url: impl Into<String>) -> Self {
        Self {
            method,
            url: url.into(),
            headers: Vec::new(),
            body: Body::Empty,
            max_response_bytes: None,
            timeout: None,
        }
    }

    pub fn get(url: impl Into<String>) -> Self {
        Self::new(Method::Get, url)
    }

    pub fn post(url: impl Into<String>) -> Self {
        Self::new(Method::Post, url)
    }

    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    pub fn json(mut self, value: serde_json::Value) -> Self {
        self.body = Body::Json(value);
        self
    }

    pub fn bytes(mut self, content_type: impl Into<String>, data: Vec<u8>) -> Self {
        self.body = Body::Bytes {
            content_type: content_type.into(),
            data,
        };
        self
    }

    /// Stream an already size-checked local file (see [`file_upload_body`]).
    pub fn file(mut self, body: Body) -> Self {
        debug_assert!(matches!(body, Body::File { .. }));
        self.body = body;
        self
    }

    /// Allow a larger response body than the control-plane default.
    pub fn max_response_bytes(mut self, limit: u64) -> Self {
        self.max_response_bytes = Some(limit);
        self
    }

    /// Give this request its own total deadline.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// A synchronous media generation: a large response, generated while
    /// the request is open.
    pub fn media_response(self) -> Self {
        self.max_response_bytes(MEDIA_RESPONSE_BYTES_MAX)
            .timeout(MEDIA_REQUEST_TIMEOUT)
    }
}

/// Describe a local reference file for a streaming upload, refusing files
/// over [`UPLOAD_BYTES_MAX`] before any byte is read.
pub async fn file_upload_body(path: &Path, content_type: &str) -> Result<Body, GenError> {
    file_upload_body_with_limit(path, content_type, UPLOAD_BYTES_MAX).await
}

pub(crate) async fn file_upload_body_with_limit(
    path: &Path,
    content_type: &str,
    limit: u64,
) -> Result<Body, GenError> {
    let metadata = tokio::fs::metadata(path)
        .await
        .map_err(|error| GenError::Other(anyhow::anyhow!("read upload file: {error}")))?;
    if !metadata.is_file() {
        return Err(GenError::Other(anyhow::anyhow!(
            "upload source is not a regular file"
        )));
    }
    if metadata.len() > limit {
        return Err(GenError::UploadTooLarge {
            len: metadata.len(),
            limit,
        });
    }
    Ok(Body::File {
        content_type: content_type.to_string(),
        path: path.to_path_buf(),
        len: metadata.len(),
    })
}

/// A transport-agnostic HTTP response.
#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl HttpResponse {
    pub fn new(status: u16, body: Vec<u8>) -> Self {
        Self {
            status,
            headers: Vec::new(),
            body,
        }
    }

    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    /// Parse the body as JSON into `T`.
    pub fn json<T: serde::de::DeserializeOwned>(&self) -> Result<T, GenError> {
        serde_json::from_slice(&self.body).map_err(GenError::from)
    }

    /// Find a response header value (case-insensitive).
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

/// The transport contract. Implementors perform a single request/response.
#[async_trait]
pub trait HttpTransport: Send + Sync {
    async fn send(&self, req: HttpRequest) -> Result<HttpResponse, GenError>;
}

/// Time and size bounds applied by [`ReqwestTransport`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransportLimits {
    /// Establishing a connection (DNS, TCP, TLS).
    pub connect_timeout: Duration,
    /// Total time for one control-plane request, response body included.
    pub request_timeout: Duration,
    /// Slowest upload throughput a file upload is given time for, on top of
    /// `request_timeout`.
    pub upload_min_bytes_per_second: u64,
    /// Upper bound on the total time of one file upload.
    pub upload_timeout_max: Duration,
    /// TCP keepalive, so a silently dropped peer is noticed.
    pub tcp_keepalive: Duration,
}

impl Default for TransportLimits {
    fn default() -> Self {
        Self {
            connect_timeout: Duration::from_secs(15),
            request_timeout: Duration::from_secs(60),
            upload_min_bytes_per_second: 64 * 1024,
            upload_timeout_max: Duration::from_secs(6 * 60 * 60),
            tcp_keepalive: Duration::from_secs(30),
        }
    }
}

impl TransportLimits {
    /// Total deadline for one request: an explicit request deadline wins,
    /// and uploads get extra time for their size.
    pub fn timeout_for_request(&self, request: &HttpRequest) -> Duration {
        request
            .timeout
            .unwrap_or_else(|| self.timeout_for(&request.body))
    }

    /// Total deadline for a body: uploads get extra time for their size.
    pub fn timeout_for(&self, body: &Body) -> Duration {
        match body {
            Body::File { len, .. } => {
                let transfer = Duration::from_secs(*len / self.upload_min_bytes_per_second.max(1));
                (self.request_timeout + transfer).min(self.upload_timeout_max)
            }
            _ => self.request_timeout,
        }
    }
}

/// Production transport backed by `reqwest`.
pub struct ReqwestTransport {
    client: reqwest::Client,
    limits: TransportLimits,
}

impl ReqwestTransport {
    /// A client with connect and keepalive timeouts; every request also gets
    /// a total deadline from [`TransportLimits`].
    pub fn new() -> Self {
        Self::with_limits(TransportLimits::default())
    }

    pub fn with_limits(limits: TransportLimits) -> Self {
        let client = reqwest::Client::builder()
            .connect_timeout(limits.connect_timeout)
            .tcp_keepalive(limits.tcp_keepalive)
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self { client, limits }
    }

    /// Use a caller-built client. Per-request deadlines and body ceilings
    /// from the default [`TransportLimits`] still apply.
    pub fn with_client(client: reqwest::Client) -> Self {
        Self {
            client,
            limits: TransportLimits::default(),
        }
    }

    /// A caller-built client with explicit limits.
    pub fn with_client_and_limits(client: reqwest::Client, limits: TransportLimits) -> Self {
        Self { client, limits }
    }

    pub fn limits(&self) -> TransportLimits {
        self.limits
    }
}

impl Default for ReqwestTransport {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert a reqwest failure without its URL (it can carry signed query
/// parameters). A connect failure means the request never reached the server.
fn map_reqwest_error(error: reqwest::Error) -> GenError {
    let connect = error.is_connect();
    let timeout = error.is_timeout();
    let error = error.without_url();
    let mut message = error.to_string();
    let mut source = std::error::Error::source(&error);
    while let Some(cause) = source {
        message.push_str(": ");
        message.push_str(&cause.to_string());
        source = cause.source();
    }
    if timeout && !message.contains("timed out") {
        message.push_str(" (timed out)");
    }
    if connect {
        GenError::Connect(message)
    } else {
        GenError::Transport(message)
    }
}

#[async_trait]
impl HttpTransport for ReqwestTransport {
    async fn send(&self, req: HttpRequest) -> Result<HttpResponse, GenError> {
        let method = match req.method {
            Method::Get => reqwest::Method::GET,
            Method::Post => reqwest::Method::POST,
            Method::Put => reqwest::Method::PUT,
        };
        let limit = req.max_response_bytes.unwrap_or(CONTROL_RESPONSE_BYTES_MAX);
        let mut builder = self
            .client
            .request(method, &req.url)
            .timeout(self.limits.timeout_for_request(&req));
        for (k, v) in &req.headers {
            builder = builder.header(k, v);
        }
        builder = match req.body {
            Body::Empty => builder,
            Body::Json(v) => builder.json(&v),
            Body::Bytes { content_type, data } => {
                builder.header("Content-Type", content_type).body(data)
            }
            Body::File {
                content_type,
                path,
                len,
            } => {
                let file = tokio::fs::File::open(&path).await.map_err(|error| {
                    GenError::Other(anyhow::anyhow!("open upload file: {error}"))
                })?;
                let actual = file
                    .metadata()
                    .await
                    .map_err(|error| GenError::Other(anyhow::anyhow!("read upload file: {error}")))?
                    .len();
                if actual != len {
                    return Err(GenError::Other(anyhow::anyhow!(
                        "upload file changed size before it was sent"
                    )));
                }
                // An explicit length keeps presigned PUTs (which reject
                // chunked bodies) working while the bytes stream from disk.
                builder
                    .header("Content-Type", content_type)
                    .header("Content-Length", len.to_string())
                    .body(reqwest::Body::from(file))
            }
        };

        let mut resp = builder.send().await.map_err(map_reqwest_error)?;
        let status = resp.status().as_u16();
        let headers = resp
            .headers()
            .iter()
            .filter_map(|(k, v)| {
                v.to_str()
                    .ok()
                    .map(|s| (k.as_str().to_string(), s.to_string()))
            })
            .collect();
        if resp.content_length().is_some_and(|length| length > limit) {
            return Err(GenError::Transport(format!(
                "response body exceeds the {limit}-byte limit"
            )));
        }
        let mut body = Vec::new();
        while let Some(chunk) = resp.chunk().await.map_err(map_reqwest_error)? {
            if body.len() as u64 + chunk.len() as u64 > limit {
                return Err(GenError::Transport(format!(
                    "response body exceeds the {limit}-byte limit"
                )));
            }
            body.extend_from_slice(&chunk);
        }
        Ok(HttpResponse {
            status,
            headers,
            body,
        })
    }
}

/// One canned reply in a `MockTransport` script.
#[derive(Clone)]
enum Canned {
    Response(HttpResponse),
    /// A failure after the request may have reached the server.
    Transport(String),
    /// A failure before the request reached the server.
    Connect(String),
    /// The request never completes (a hung connection).
    Pending,
}

/// Offline transport for tests. Two modes (composable):
///
/// - **Keyed map**: exact `"METHOD url"` -> response. Good for stable endpoints.
/// - **Sequence per key**: pops responses in order so `submit` then repeated
///   `poll` can return queued -> running -> succeeded.
///
/// Replies can also be scripted transport failures or requests that never
/// complete. Every sent request is recorded in `calls` for assertions (a file
/// body only as its metadata). No I/O ever occurs.
#[derive(Clone, Default)]
pub struct MockTransport {
    inner: std::sync::Arc<MockInner>,
}

#[derive(Default)]
struct MockInner {
    routes: std::sync::Mutex<HashMap<String, Vec<Canned>>>,
    calls: std::sync::Mutex<Vec<HttpRequest>>,
    fallback: std::sync::Mutex<Option<HttpResponse>>,
}

impl MockTransport {
    pub fn new() -> Self {
        Self::default()
    }

    fn key(method: Method, url: &str) -> String {
        format!("{} {}", method.as_str(), url)
    }

    /// Register a single response for `method url`. Repeated registration on the
    /// same key appends to that key's sequence.
    pub fn on(
        &self,
        method: Method,
        url: impl AsRef<str>,
        status: u16,
        body: serde_json::Value,
    ) -> &Self {
        self.on_raw(
            method,
            url,
            HttpResponse::new(status, serde_json::to_vec(&body).unwrap()),
        )
    }

    /// Register a raw response (arbitrary body/headers).
    pub fn on_raw(&self, method: Method, url: impl AsRef<str>, response: HttpResponse) -> &Self {
        self.push(method, url.as_ref(), Canned::Response(response))
    }

    /// Script a transport failure after the request may have been delivered
    /// (reset connection, timeout, body read error).
    pub fn on_transport_error(
        &self,
        method: Method,
        url: impl AsRef<str>,
        message: impl Into<String>,
    ) -> &Self {
        self.push(method, url.as_ref(), Canned::Transport(message.into()))
    }

    /// Script a connection failure: the request never reached the server.
    pub fn on_connect_error(
        &self,
        method: Method,
        url: impl AsRef<str>,
        message: impl Into<String>,
    ) -> &Self {
        self.push(method, url.as_ref(), Canned::Connect(message.into()))
    }

    /// Script a request that never completes, like a half-open connection.
    pub fn on_pending(&self, method: Method, url: impl AsRef<str>) -> &Self {
        self.push(method, url.as_ref(), Canned::Pending)
    }

    fn push(&self, method: Method, url: &str, reply: Canned) -> &Self {
        let key = Self::key(method, url);
        self.inner
            .routes
            .lock()
            .unwrap()
            .entry(key)
            .or_default()
            .push(reply);
        self
    }

    /// Register a sequence of JSON responses for `method url`, served in order.
    pub fn on_sequence(
        &self,
        method: Method,
        url: impl AsRef<str>,
        steps: Vec<(u16, serde_json::Value)>,
    ) -> &Self {
        for (status, body) in steps {
            self.on(method, url.as_ref(), status, body);
        }
        self
    }

    /// Set a catch-all response for any unmatched request.
    pub fn fallback(&self, status: u16, body: serde_json::Value) -> &Self {
        *self.inner.fallback.lock().unwrap() = Some(HttpResponse::new(
            status,
            serde_json::to_vec(&body).unwrap(),
        ));
        self
    }

    /// Number of requests sent so far.
    pub fn call_count(&self) -> usize {
        self.inner.calls.lock().unwrap().len()
    }

    /// Snapshot of all recorded requests.
    pub fn calls(&self) -> Vec<HttpRequest> {
        self.inner.calls.lock().unwrap().clone()
    }

    /// The most recently recorded request, if any.
    pub fn last_call(&self) -> Option<HttpRequest> {
        self.inner.calls.lock().unwrap().last().cloned()
    }
}

#[async_trait]
impl HttpTransport for MockTransport {
    async fn send(&self, req: HttpRequest) -> Result<HttpResponse, GenError> {
        self.inner.calls.lock().unwrap().push(req.clone());
        let key = Self::key(req.method, &req.url);
        let reply = {
            let mut routes = self.inner.routes.lock().unwrap();
            routes.get_mut(&key).and_then(|queue| {
                if queue.len() > 1 {
                    // Sequence: pop the front, keep the rest.
                    Some(queue.remove(0))
                } else {
                    // Single entry: serve it repeatedly (sticky).
                    queue.first().cloned()
                }
            })
        };
        match reply {
            Some(Canned::Response(response)) => {
                let limit = req.max_response_bytes.unwrap_or(CONTROL_RESPONSE_BYTES_MAX);
                if response.body.len() as u64 > limit {
                    return Err(GenError::Transport(format!(
                        "response body exceeds the {limit}-byte limit"
                    )));
                }
                Ok(response)
            }
            Some(Canned::Transport(message)) => Err(GenError::Transport(message)),
            Some(Canned::Connect(message)) => Err(GenError::Connect(message)),
            Some(Canned::Pending) => std::future::pending().await,
            None => {
                if let Some(fb) = self.inner.fallback.lock().unwrap().clone() {
                    return Ok(fb);
                }
                Err(GenError::Transport(format!("no mock route for {key}")))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn mock_serves_keyed_response() {
        let m = MockTransport::new();
        m.on(Method::Get, "https://x/a", 200, json!({"ok": true}));
        let resp = m.send(HttpRequest::get("https://x/a")).await.unwrap();
        assert_eq!(resp.status, 200);
        assert_eq!(
            resp.json::<serde_json::Value>().unwrap(),
            json!({"ok":true})
        );
        assert_eq!(m.call_count(), 1);
    }

    #[tokio::test]
    async fn mock_serves_sequence_then_sticks_on_last() {
        let m = MockTransport::new();
        m.on_sequence(
            Method::Get,
            "https://x/poll",
            vec![
                (200, json!({"status":"queued"})),
                (200, json!({"status":"running"})),
                (200, json!({"status":"succeeded"})),
            ],
        );
        let s1: serde_json::Value = m
            .send(HttpRequest::get("https://x/poll"))
            .await
            .unwrap()
            .json()
            .unwrap();
        let s2: serde_json::Value = m
            .send(HttpRequest::get("https://x/poll"))
            .await
            .unwrap()
            .json()
            .unwrap();
        let s3: serde_json::Value = m
            .send(HttpRequest::get("https://x/poll"))
            .await
            .unwrap()
            .json()
            .unwrap();
        let s4: serde_json::Value = m
            .send(HttpRequest::get("https://x/poll"))
            .await
            .unwrap()
            .json()
            .unwrap();
        assert_eq!(s1["status"], "queued");
        assert_eq!(s2["status"], "running");
        assert_eq!(s3["status"], "succeeded");
        // Last entry is sticky.
        assert_eq!(s4["status"], "succeeded");
    }

    #[tokio::test]
    async fn mock_records_request_details() {
        let m = MockTransport::new();
        m.on(Method::Post, "https://x/submit", 200, json!({}));
        let req = HttpRequest::post("https://x/submit")
            .header("Authorization", "Key abc")
            .json(json!({"prompt": "hi"}));
        m.send(req).await.unwrap();
        let last = m.last_call().unwrap();
        assert_eq!(last.method, Method::Post);
        assert!(last
            .headers
            .iter()
            .any(|(k, v)| k == "Authorization" && v == "Key abc"));
        match last.body {
            Body::Json(v) => assert_eq!(v["prompt"], "hi"),
            _ => panic!("expected json body"),
        }
    }

    #[tokio::test]
    async fn mock_unmatched_route_errors() {
        let m = MockTransport::new();
        let r = m.send(HttpRequest::get("https://x/none")).await;
        assert!(matches!(r, Err(GenError::Transport(_))));
    }

    #[tokio::test]
    async fn mock_scripts_failures_hangs_and_response_limits() {
        let m = MockTransport::new();
        m.on_transport_error(Method::Get, "https://x/a", "reset");
        m.on_connect_error(Method::Get, "https://x/a", "refused");
        m.on(Method::Get, "https://x/a", 200, json!({"ok": true}));
        assert!(matches!(
            m.send(HttpRequest::get("https://x/a")).await,
            Err(GenError::Transport(message)) if message == "reset"
        ));
        assert!(matches!(
            m.send(HttpRequest::get("https://x/a")).await,
            Err(GenError::Connect(message)) if message == "refused"
        ));
        assert_eq!(
            m.send(HttpRequest::get("https://x/a"))
                .await
                .unwrap()
                .status,
            200
        );

        m.on_pending(Method::Get, "https://x/hang");
        assert!(tokio::time::timeout(
            Duration::from_millis(20),
            m.send(HttpRequest::get("https://x/hang"))
        )
        .await
        .is_err());

        m.on_raw(
            Method::Get,
            "https://x/big",
            HttpResponse::new(200, vec![0; 2048]),
        );
        let limited = HttpRequest::get("https://x/big").max_response_bytes(1024);
        assert!(matches!(m.send(limited).await, Err(GenError::Transport(_))));
        let allowed = HttpRequest::get("https://x/big").max_response_bytes(4096);
        assert_eq!(m.send(allowed).await.unwrap().body.len(), 2048);
    }

    #[tokio::test]
    async fn file_bodies_are_size_checked_before_any_read() {
        let dir = std::env::temp_dir().join(format!(
            "opentake-gen-upload-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("clip.mp4");
        std::fs::write(&path, [7_u8; 64]).unwrap();

        assert!(matches!(
            file_upload_body_with_limit(&path, "video/mp4", 63).await,
            Err(GenError::UploadTooLarge { len: 64, limit: 63 })
        ));
        match file_upload_body_with_limit(&path, "video/mp4", 64)
            .await
            .unwrap()
        {
            Body::File {
                content_type,
                path: body_path,
                len,
            } => {
                assert_eq!(content_type, "video/mp4");
                assert_eq!(body_path, path);
                assert_eq!(len, 64);
            }
            other => panic!("expected a file body, got {other:?}"),
        }
        assert!(file_upload_body(&dir, "video/mp4").await.is_err());
        assert!(file_upload_body(&dir.join("missing.mp4"), "video/mp4")
            .await
            .is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn upload_deadlines_grow_with_the_file_size_up_to_the_cap() {
        let limits = TransportLimits::default();
        assert_eq!(limits.timeout_for(&Body::Empty), limits.request_timeout);
        let file = |len| Body::File {
            content_type: "video/mp4".into(),
            path: PathBuf::from("clip.mp4"),
            len,
        };
        assert_eq!(
            limits.timeout_for(&file(64 * 1024 * 60)),
            limits.request_timeout + Duration::from_secs(60)
        );
        assert_eq!(
            limits.timeout_for(&file(u64::MAX)),
            limits.upload_timeout_max
        );
        let media = HttpRequest::post("https://x/speech").media_response();
        assert_eq!(limits.timeout_for_request(&media), MEDIA_REQUEST_TIMEOUT);
        assert_eq!(media.max_response_bytes, Some(MEDIA_RESPONSE_BYTES_MAX));
    }

    /// A transport that never goes through a system proxy, with short limits.
    fn local_transport(request_timeout: Duration) -> ReqwestTransport {
        let limits = TransportLimits {
            connect_timeout: Duration::from_secs(2),
            request_timeout,
            ..TransportLimits::default()
        };
        let client = reqwest::Client::builder()
            .no_proxy()
            .connect_timeout(limits.connect_timeout)
            .build()
            .unwrap();
        ReqwestTransport::with_client_and_limits(client, limits)
    }

    /// Serve one connection on a local listener with `respond`, returning the
    /// raw request bytes the server read.
    fn serve_once(
        respond: impl FnOnce(&mut std::net::TcpStream) + Send + 'static,
    ) -> (String, std::thread::JoinHandle<Vec<u8>>) {
        use std::io::Read;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 4096];
            // Read the head, then any body announced by Content-Length.
            while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                let count = stream.read(&mut buffer).unwrap_or(0);
                if count == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..count]);
            }
            let head_end = request
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|index| index + 4)
                .unwrap_or(request.len());
            let head = String::from_utf8_lossy(&request[..head_end]).to_ascii_lowercase();
            let body_len = head
                .lines()
                .find_map(|line| line.strip_prefix("content-length:"))
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(0);
            while request.len() < head_end + body_len {
                let count = stream.read(&mut buffer).unwrap_or(0);
                if count == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..count]);
            }
            respond(&mut stream);
            request
        });
        (url, handle)
    }

    #[tokio::test]
    async fn reqwest_transport_times_out_when_the_server_never_answers() {
        let (url, server) = serve_once(|_stream| {
            // Accept, read the request, and never write a response.
            std::thread::sleep(Duration::from_secs(2));
        });
        let transport = local_transport(Duration::from_millis(300));
        let started = std::time::Instant::now();
        let error = transport.send(HttpRequest::get(url)).await.unwrap_err();
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "{:?}",
            started.elapsed()
        );
        match error {
            GenError::Transport(message) => {
                assert!(message.contains("timed out"), "{message}");
                assert!(!message.contains("127.0.0.1"), "URL leaked: {message}");
            }
            other => panic!("expected a transport timeout, got {other:?}"),
        }
        server.join().unwrap();
    }

    #[tokio::test]
    async fn reqwest_transport_rejects_oversized_response_bodies() {
        use std::io::Write;
        // Announced length over the limit.
        let (url, server) = serve_once(|stream| {
            let body = vec![b'x'; 4096];
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n",
                body.len()
            )
            .unwrap();
            let _ = stream.write_all(&body);
        });
        let transport = local_transport(Duration::from_secs(5));
        let error = transport
            .send(HttpRequest::get(url).max_response_bytes(1024))
            .await
            .unwrap_err();
        assert!(
            matches!(&error, GenError::Transport(message) if message.contains("exceeds")),
            "{error:?}"
        );
        server.join().unwrap();

        // A chunked body with no length is counted while it streams.
        let (url, server) = serve_once(|stream| {
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
                .unwrap();
            for _ in 0..4 {
                let _ = write!(stream, "400\r\n{}\r\n", "y".repeat(1024));
            }
            let _ = stream.write_all(b"0\r\n\r\n");
        });
        let error = transport
            .send(HttpRequest::get(url).max_response_bytes(2048))
            .await
            .unwrap_err();
        assert!(
            matches!(&error, GenError::Transport(message) if message.contains("exceeds")),
            "{error:?}"
        );
        server.join().unwrap();
    }

    #[tokio::test]
    async fn reqwest_transport_reports_a_refused_connection_as_never_sent() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        drop(listener);
        let error = local_transport(Duration::from_secs(5))
            .send(HttpRequest::get(url))
            .await
            .unwrap_err();
        assert!(matches!(error, GenError::Connect(_)), "{error:?}");
        assert!(error.is_transient());
    }

    #[tokio::test]
    async fn reqwest_transport_streams_file_bodies_with_an_exact_length() {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!(
            "opentake-gen-stream-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("reference.bin");
        let bytes = (0..200_000_u32)
            .map(|value| value as u8)
            .collect::<Vec<_>>();
        std::fs::write(&path, &bytes).unwrap();
        let (url, server) = serve_once(|stream| {
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")
                .unwrap();
        });
        let body = file_upload_body(&path, "application/octet-stream")
            .await
            .unwrap();
        let response = local_transport(Duration::from_secs(5))
            .send(HttpRequest::new(Method::Put, url).file(body))
            .await
            .unwrap();
        assert_eq!(response.status, 200);
        let request = server.join().unwrap();
        let head_end = request
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .unwrap()
            + 4;
        let head = String::from_utf8_lossy(&request[..head_end]).to_ascii_lowercase();
        assert!(head.starts_with("put / http/1.1"), "{head}");
        assert!(head.contains("content-length: 200000"), "{head}");
        assert!(!head.contains("transfer-encoding"), "{head}");
        assert_eq!(&request[head_end..], bytes.as_slice());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn mock_fallback_used_when_no_route() {
        let m = MockTransport::new();
        m.fallback(404, json!({"error":{"code":"not_found","message":"x"}}));
        let r = m.send(HttpRequest::get("https://x/none")).await.unwrap();
        assert_eq!(r.status, 404);
    }
}
