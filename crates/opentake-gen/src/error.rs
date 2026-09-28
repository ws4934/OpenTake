//! Error type for the generation client. Replicates the upstream error-code
//! contract (axiom A6): `401/unauthenticated`, `402/insufficient_credits`, plus
//! a structured `{"error":{code,message}}` envelope. Internal failures flow as
//! `anyhow::Error`; boundary layers convert to `Err(String)` via `to_string()`.

/// Errors surfaced by `GenClient` and the provider adapters.
#[derive(Debug, thiserror::Error)]
pub enum GenError {
    /// Backend (managed proxy) or a required provider key is not configured.
    /// Upstream: `GenerationBackendError.notConfigured`.
    #[error("backend not configured")]
    NotConfigured,

    /// 401 / code "unauthenticated" — `PalmierClientError` 401 mapping.
    #[error("sign in to continue")]
    Unauthenticated,

    /// 402 / code "insufficient_credits".
    #[error("{0}")]
    InsufficientCredits(String),

    /// Transport-level failure after the connection was established (reset,
    /// timeout, non-HTTP response, body read). The request may have reached
    /// the server.
    #[error("transport error: {0}")]
    Transport(String),

    /// The connection could not be established (DNS, TCP connect, TLS), so
    /// the request never reached the server.
    #[error("connection failed: {0}")]
    Connect(String),

    /// The provider answered with a response or job identity this client
    /// cannot use (missing ids, foreign URLs, a synchronous job that is no
    /// longer cached). Repeating the same request cannot fix it.
    #[error("unusable provider response: {0}")]
    Protocol(String),

    /// A reference file is larger than the upload limit; nothing was read.
    #[error("upload file is {len} bytes, above the {limit}-byte limit")]
    UploadTooLarge { len: u64, limit: u64 },

    /// Structured API error parsed from the `{"error":{code,message}}` envelope,
    /// or synthesized from an HTTP status when no envelope is present.
    #[error("api error {status} [{code}]: {message}")]
    Api {
        status: u16,
        code: String,
        message: String,
        /// The server's `Retry-After` delay, when it sent one in seconds.
        retry_after: Option<std::time::Duration>,
    },

    /// Any other internal error (serde, IO, keyring, logic).
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl From<serde_json::Error> for GenError {
    fn from(e: serde_json::Error) -> Self {
        GenError::Other(anyhow::Error::new(e))
    }
}

impl From<keyring::Error> for GenError {
    fn from(e: keyring::Error) -> Self {
        GenError::Other(anyhow::Error::new(e))
    }
}

impl From<url::ParseError> for GenError {
    fn from(e: url::ParseError) -> Self {
        GenError::Protocol(e.to_string())
    }
}

impl GenError {
    /// Whether the same request may succeed when repeated later: network
    /// failures and HTTP 408, 429 and 5xx. Authentication, credits, other 4xx
    /// and unusable responses are final.
    pub fn is_transient(&self) -> bool {
        match self {
            GenError::Transport(_) | GenError::Connect(_) => true,
            GenError::Api { status, .. } => matches!(status, 408 | 429 | 500..=599),
            _ => false,
        }
    }

    /// The server-requested delay before retrying, if any.
    pub fn retry_after(&self) -> Option<std::time::Duration> {
        match self {
            GenError::Api { retry_after, .. } => *retry_after,
            _ => None,
        }
    }

    /// A short, credential- and URL-free label for logs.
    pub fn kind_label(&self) -> String {
        match self {
            GenError::NotConfigured => "not configured".to_string(),
            GenError::Unauthenticated => "unauthenticated".to_string(),
            GenError::InsufficientCredits(_) => "insufficient credits".to_string(),
            GenError::Transport(_) => "transport".to_string(),
            GenError::Connect(_) => "connect".to_string(),
            GenError::Protocol(_) => "protocol".to_string(),
            GenError::UploadTooLarge { .. } => "upload too large".to_string(),
            GenError::Api { status, .. } => format!("http {status}"),
            GenError::Other(_) => "other".to_string(),
        }
    }
}

/// Shape of the `{"error":{"code","message"}}` envelope returned by the proxy
/// and (when present) by upstream vendors. Replicates `BackendErrorEnvelope`.
#[derive(Debug, serde::Deserialize)]
pub(crate) struct ErrorEnvelope {
    pub error: ErrorEnvelopeInner,
}

#[derive(Debug, serde::Deserialize)]
pub(crate) struct ErrorEnvelopeInner {
    #[serde(default)]
    pub code: String,
    #[serde(default)]
    pub message: String,
}

/// Map a non-2xx HTTP response to a `GenError`. Replicates `assertHTTPOK`
/// (`GenerationBackend.swift:76-90`) + `PalmierClientError.from`
/// (`PalmierClient.swift:80-91`): parse the envelope first, prefer `code`, then
/// fall back to status (401 -> Unauthenticated, 402 -> InsufficientCredits).
pub(crate) fn map_http_error(status: u16, body: &[u8]) -> GenError {
    let parsed: Option<ErrorEnvelope> = serde_json::from_slice(body).ok();
    let (code, message) = match parsed {
        Some(env) => (env.error.code, env.error.message),
        None => (
            String::new(),
            String::from_utf8_lossy(body).trim().to_string(),
        ),
    };

    // Prefer explicit code, then fall back to HTTP status.
    if code == "unauthenticated" || status == 401 {
        return GenError::Unauthenticated;
    }
    if code == "insufficient_credits" || status == 402 {
        let msg = if message.is_empty() {
            "insufficient credits".to_string()
        } else {
            message
        };
        return GenError::InsufficientCredits(msg);
    }

    GenError::Api {
        status,
        code,
        message,
        retry_after: None,
    }
}

/// [`map_http_error`] for a whole response, keeping a `Retry-After` delay
/// given in seconds (an HTTP-date is ignored; the caller's backoff applies).
pub(crate) fn map_http_response(response: &crate::transport::HttpResponse) -> GenError {
    let mut error = map_http_error(response.status, &response.body);
    if let GenError::Api { retry_after, .. } = &mut error {
        *retry_after = response
            .header("Retry-After")
            .and_then(|value| value.trim().parse::<u64>().ok())
            .map(std::time::Duration::from_secs);
    }
    error
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_401_to_unauthenticated() {
        let e = map_http_error(401, b"{}");
        assert!(matches!(e, GenError::Unauthenticated));
    }

    #[test]
    fn maps_code_unauthenticated_even_on_400() {
        let body = br#"{"error":{"code":"unauthenticated","message":"nope"}}"#;
        let e = map_http_error(400, body);
        assert!(matches!(e, GenError::Unauthenticated));
    }

    #[test]
    fn maps_402_to_insufficient_credits_with_message() {
        let body = br#"{"error":{"code":"insufficient_credits","message":"out of credits"}}"#;
        match map_http_error(402, body) {
            GenError::InsufficientCredits(m) => assert_eq!(m, "out of credits"),
            other => panic!("expected InsufficientCredits, got {other:?}"),
        }
    }

    #[test]
    fn maps_402_with_empty_body_to_default_message() {
        // Empty body -> no message available -> default text.
        let e = map_http_error(402, b"");
        match e {
            GenError::InsufficientCredits(m) => assert_eq!(m, "insufficient credits"),
            other => panic!("expected InsufficientCredits, got {other:?}"),
        }
    }

    #[test]
    fn maps_402_with_plain_text_body_passes_it_through() {
        // Non-JSON body is surfaced as the message (server text passthrough).
        let e = map_http_error(402, b"please add funds");
        match e {
            GenError::InsufficientCredits(m) => assert_eq!(m, "please add funds"),
            other => panic!("expected InsufficientCredits, got {other:?}"),
        }
    }

    #[test]
    fn maps_500_with_envelope_to_api() {
        let body = br#"{"error":{"code":"server_error","message":"boom"}}"#;
        match map_http_error(500, body) {
            GenError::Api {
                status,
                code,
                message,
                ..
            } => {
                assert_eq!(status, 500);
                assert_eq!(code, "server_error");
                assert_eq!(message, "boom");
            }
            other => panic!("expected Api, got {other:?}"),
        }
    }

    #[test]
    fn maps_unknown_status_without_envelope_to_api_with_body_message() {
        match map_http_error(418, b"i am a teapot") {
            GenError::Api {
                status,
                code,
                message,
                ..
            } => {
                assert_eq!(status, 418);
                assert_eq!(code, "");
                assert_eq!(message, "i am a teapot");
            }
            other => panic!("expected Api, got {other:?}"),
        }
    }

    #[test]
    fn transient_errors_are_network_failures_and_retryable_statuses() {
        assert!(GenError::Transport("reset".into()).is_transient());
        assert!(GenError::Connect("refused".into()).is_transient());
        for status in [408, 429, 500, 502, 503, 504] {
            assert!(map_http_error(status, b"").is_transient(), "{status}");
        }
        for final_error in [
            map_http_error(400, b""),
            map_http_error(401, b""),
            map_http_error(402, b""),
            map_http_error(404, b""),
            map_http_error(422, b""),
            GenError::NotConfigured,
            GenError::Protocol("bad id".into()),
            GenError::UploadTooLarge { len: 2, limit: 1 },
        ] {
            assert!(!final_error.is_transient(), "{final_error:?}");
        }
    }

    #[test]
    fn retry_after_seconds_are_kept_from_the_response() {
        let mut response = crate::transport::HttpResponse::new(429, b"{}".to_vec());
        response.headers.push(("retry-after".into(), " 7 ".into()));
        let error = map_http_response(&response);
        assert_eq!(error.retry_after(), Some(std::time::Duration::from_secs(7)));
        let mut dated = crate::transport::HttpResponse::new(503, Vec::new());
        dated
            .headers
            .push(("Retry-After".into(), "Wed, 21 Oct 2015 07:28:00 GMT".into()));
        assert_eq!(map_http_response(&dated).retry_after(), None);
    }

    #[test]
    fn error_display_is_user_facing() {
        assert_eq!(GenError::Unauthenticated.to_string(), "sign in to continue");
        assert_eq!(
            GenError::NotConfigured.to_string(),
            "backend not configured"
        );
    }
}
