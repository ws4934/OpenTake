//! Explicit, privacy-bounded error telemetry initialization.
//!
//! No SDK client is created unless a non-empty, valid DSN is supplied by the
//! packaged build or the current process environment. Event preprocessing drops
//! request/user payloads and redacts paths and credential-shaped text.

use std::borrow::Cow;
use std::fmt;
use std::sync::{Arc, LazyLock};

use regex::Regex;
use sentry::protocol::{Event, Stacktrace};
use sentry::types::Dsn;

const ENV_DSN: &str = "OPENTAKE_SENTRY_DSN";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TelemetrySource {
    Packaged,
    Environment,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TelemetryInitStatus {
    Disabled,
    InvalidConfiguration,
    StartFailed,
    Started(TelemetrySource),
}

pub struct TelemetryOptions {
    pub dsn: Dsn,
    pub source: TelemetrySource,
    pub send_default_pii: bool,
    pub capture_failed_requests: bool,
    pub traces_sample_rate: f32,
}

impl fmt::Debug for TelemetryOptions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TelemetryOptions")
            .field("dsn", &"[REDACTED]")
            .field("source", &self.source)
            .field("send_default_pii", &self.send_default_pii)
            .field("capture_failed_requests", &self.capture_failed_requests)
            .field("traces_sample_rate", &self.traces_sample_rate)
            .finish()
    }
}

pub fn init_telemetry_with(
    packaged_dsn: Option<&str>,
    environment_dsn: Option<&str>,
    start: impl FnOnce(&TelemetryOptions) -> Result<(), ()>,
) -> TelemetryInitStatus {
    let candidate = match environment_dsn {
        Some(value) => non_empty(value).map(|dsn| (dsn, TelemetrySource::Environment)),
        None => packaged_dsn
            .and_then(non_empty)
            .map(|dsn| (dsn, TelemetrySource::Packaged)),
    };
    let Some((raw_dsn, source)) = candidate else {
        return TelemetryInitStatus::Disabled;
    };
    let Ok(dsn) = raw_dsn.parse::<Dsn>() else {
        return TelemetryInitStatus::InvalidConfiguration;
    };
    let options = TelemetryOptions {
        dsn,
        source,
        send_default_pii: false,
        capture_failed_requests: false,
        traces_sample_rate: 0.1,
    };
    if start(&options).is_err() {
        TelemetryInitStatus::StartFailed
    } else {
        TelemetryInitStatus::Started(source)
    }
}

fn non_empty(value: &str) -> Option<&str> {
    let value = value.trim();
    (!value.is_empty()).then_some(value)
}

pub struct TelemetryRuntime {
    _guard: Option<sentry::ClientInitGuard>,
    pub status: TelemetryInitStatus,
}

pub fn init_telemetry() -> TelemetryRuntime {
    let packaged_dsn = option_env!("OPENTAKE_PACKAGED_SENTRY_DSN");
    let mut guard = None;
    let status = init_telemetry_from_environment(
        packaged_dsn,
        |name| std::env::var(name),
        |options| {
            guard = Some(start_sentry(options));
            Ok(())
        },
    );
    TelemetryRuntime {
        _guard: guard,
        status,
    }
}

fn init_telemetry_from_environment(
    packaged_dsn: Option<&str>,
    read: impl FnOnce(&str) -> Result<String, std::env::VarError>,
    start: impl FnOnce(&TelemetryOptions) -> Result<(), ()>,
) -> TelemetryInitStatus {
    let environment_dsn = match read(ENV_DSN) {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => return TelemetryInitStatus::InvalidConfiguration,
    };
    init_telemetry_with(packaged_dsn, environment_dsn.as_deref(), start)
}

fn start_sentry(options: &TelemetryOptions) -> sentry::ClientInitGuard {
    let release = format!(
        "opentake@{}+{}",
        env!("CARGO_PKG_VERSION"),
        option_env!("OPENTAKE_BUILD_NUMBER").unwrap_or("?")
    );
    let environment = if cfg!(debug_assertions) {
        "development"
    } else {
        "production"
    };
    let mut client_options =
        sentry::ClientOptions::default().traces_sample_rate(options.traces_sample_rate);
    client_options.dsn = Some(options.dsn.clone());
    client_options.release = Some(Cow::Owned(release));
    client_options.environment = Some(Cow::Borrowed(environment));
    client_options.attach_stacktrace = true;
    client_options.send_default_pii = options.send_default_pii;
    client_options.max_request_body_size = sentry::MaxRequestBodySize::None;
    client_options.enable_logs = false;
    client_options.enable_metrics = false;
    client_options.auto_session_tracking = false;
    client_options.before_send = Some(Arc::new(|event| Some(scrub_event(event))));
    client_options.before_breadcrumb = Some(Arc::new(|mut breadcrumb| {
        breadcrumb.message = breadcrumb
            .message
            .map(|value| redact_sensitive_text(&value));
        breadcrumb.data.clear();
        Some(breadcrumb)
    }));
    sentry::init(client_options)
}

fn scrub_event(mut event: Event<'static>) -> Event<'static> {
    event.user = None;
    event.request = None;
    event.server_name = None;
    event.extra.clear();
    event.message = event.message.map(|value| redact_sensitive_text(&value));
    event.culprit = event.culprit.map(|value| redact_sensitive_text(&value));
    event.transaction = event.transaction.map(|value| redact_sensitive_text(&value));
    for value in event.tags.values_mut() {
        *value = redact_sensitive_text(value);
    }
    if let Some(entry) = &mut event.logentry {
        entry.message = redact_sensitive_text(&entry.message);
        entry.params.clear();
    }
    for breadcrumb in &mut event.breadcrumbs {
        breadcrumb.message = breadcrumb
            .message
            .take()
            .map(|value| redact_sensitive_text(&value));
        breadcrumb.data.clear();
    }
    for exception in &mut event.exception {
        exception.value = exception
            .value
            .take()
            .map(|value| redact_sensitive_text(&value));
        scrub_stacktrace(exception.stacktrace.as_mut());
        scrub_stacktrace(exception.raw_stacktrace.as_mut());
    }
    scrub_stacktrace(event.stacktrace.as_mut());
    for thread in &mut event.threads {
        thread.name = thread
            .name
            .take()
            .map(|value| redact_sensitive_text(&value));
        scrub_stacktrace(thread.stacktrace.as_mut());
        scrub_stacktrace(thread.raw_stacktrace.as_mut());
    }
    event
}

fn scrub_stacktrace(stacktrace: Option<&mut Stacktrace>) {
    let Some(stacktrace) = stacktrace else {
        return;
    };
    for frame in &mut stacktrace.frames {
        frame.abs_path = None;
        frame.package = frame
            .package
            .take()
            .map(|value| redact_sensitive_text(&value));
        frame.pre_context.clear();
        frame.context_line = None;
        frame.post_context.clear();
        frame.vars.clear();
    }
    stacktrace.registers.clear();
}

pub fn redact_sensitive_text(input: &str) -> String {
    static REDACTORS: LazyLock<[(Regex, &str); 6]> = LazyLock::new(|| {
        [
            // Headers can contain schemes, whitespace and multiple attributes.
            (r#"(?im)\b((?:proxy[-_])?authorization|x[-_][a-z0-9_-]*key)["']?(?:\s*[:=]\s*|\s+)[^\r\n]+"#, "${1}=[REDACTED]"),
            (r"(?i)([?&][a-z0-9_.-]*(?:token|key|secret|sig|auth)[a-z0-9_.-]*=)[^&\s#]+", "${1}[REDACTED]"),
            (r#"(?i)\b(?P<key>[a-z0-9_.-]*(?:api[_-]?key|token|secret|password|passwd|pwd|credential|auth)[a-z0-9_.-]*)["']?\s*[:=]\s*(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s"',;&}]+)"#, "${key}=[REDACTED]"),
            (r#"(?i)\b(bearer|basic)\s+[^\s"',;<>]+"#, "${1} [REDACTED]"),
            (r"\b(?:sk-[A-Za-z0-9_-]+|r8_[A-Za-z0-9_-]+)\b", "[REDACTED]"),
            // Paths may contain spaces, quotes and punctuation. Remove the
            // line suffix rather than guessing where a private name ends.
            (r#"(?im)(?:(?P<prefix>^|[\s"'(=,\[{])(?:[a-z]:[\\/]|\\\\|~/|/)|(?:\b[a-z]:[\\/]|\\\\|/(?:Users|home)/))[^\r\n]+"#, "${prefix}[PATH]"),
        ].map(|(pattern, replacement)| (Regex::new(pattern).expect("valid telemetry redaction pattern"), replacement))
    });
    let mut output = input.to_owned();
    for (pattern, replacement) in &*REDACTORS {
        output = pattern.replace_all(&output, *replacement).into_owned();
    }
    output
}

#[cfg(test)]
mod tests {
    use super::{init_telemetry_from_environment, redact_sensitive_text, TelemetryInitStatus};

    #[test]
    fn generic_sentry_dsn_does_not_enable_reporting() {
        let status = init_telemetry_from_environment(
            None,
            |name| match name {
                "SENTRY_DSN" => Ok("https://other-project@example.com/1".into()),
                _ => Err(std::env::VarError::NotPresent),
            },
            |_| panic!("generic SENTRY_DSN must not start telemetry"),
        );
        assert_eq!(status, TelemetryInitStatus::Disabled);
    }

    #[test]
    fn redacts_unix_windows_and_credential_shapes() {
        let text = redact_sensitive_text(
            r#"/home/alice/project C:\Users\alice\project password:hunter2 Bearer token-value"#,
        );
        assert_eq!(text, "[PATH]");
    }
}
