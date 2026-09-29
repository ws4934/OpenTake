use std::cell::RefCell;

use opentake_tauri_lib::telemetry::{
    init_telemetry_with, redact_sensitive_text, TelemetryInitStatus, TelemetrySource,
};

#[test]
fn starts_only_with_explicit_packaged_or_environment_dsn() {
    let starts = RefCell::new(Vec::new());
    let start = |options: &opentake_tauri_lib::telemetry::TelemetryOptions| {
        starts.borrow_mut().push((
            options.source,
            options.dsn.to_string(),
            options.send_default_pii,
            options.capture_failed_requests,
            options.traces_sample_rate,
        ));
        Ok(())
    };

    assert_eq!(
        init_telemetry_with(None, None, start),
        TelemetryInitStatus::Disabled
    );
    assert_eq!(
        init_telemetry_with(Some("  "), Some(""), start),
        TelemetryInitStatus::Disabled
    );
    assert!(starts.borrow().is_empty());

    assert_eq!(
        init_telemetry_with(Some("https://public@example.com/1"), None, start),
        TelemetryInitStatus::Started(TelemetrySource::Packaged)
    );
    assert_eq!(
        init_telemetry_with(
            Some("https://packaged@example.com/1"),
            Some("https://environment@example.com/2"),
            start,
        ),
        TelemetryInitStatus::Started(TelemetrySource::Environment)
    );
    assert_eq!(starts.borrow().len(), 2);
    assert_eq!(starts.borrow()[0].0, TelemetrySource::Packaged);
    assert_eq!(starts.borrow()[1].0, TelemetrySource::Environment);
    assert_eq!(starts.borrow()[1].1, "https://environment:@example.com/2");
    assert!(!starts.borrow()[1].2);
    assert!(!starts.borrow()[1].3);
    assert_eq!(starts.borrow()[1].4, 0.1);

    assert_eq!(
        init_telemetry_with(None, Some("not a dsn"), start),
        TelemetryInitStatus::InvalidConfiguration
    );
    assert_eq!(starts.borrow().len(), 2);

    let redacted = redact_sensitive_text(
        "failed /Users/alice/private.otproj api_key=sk-secret token=abc123 Bearer xyz",
    );
    assert!(!redacted.contains("/Users/alice"));
    assert!(!redacted.contains("sk-secret"));
    assert!(!redacted.contains("abc123"));
    assert!(!redacted.contains("xyz"));
}

#[test]
fn explicit_empty_environment_dsn_disables_packaged_reporting() {
    for environment in ["", "  \n\t"] {
        assert_eq!(
            init_telemetry_with(
                Some("https://packaged@example.com/1"),
                Some(environment),
                |_| { panic!("an explicit opt-out must not start telemetry") }
            ),
            TelemetryInitStatus::Disabled
        );
    }
}

#[test]
fn sensitive_values_are_removed_from_common_panic_formats() {
    for (input, forbidden) in [
        ("password: hunter2", "hunter2"),
        (r#"{"api_key": "sk-live-123"}"#, "sk-live-123"),
        ("https://api.example.com/v1?access_token=abc123", "abc123"),
        (
            "https://api.example.com/v1?signature=abc123&format=json",
            "abc123",
        ),
        ("Authorization: Basic dXNlcjpwYXNz", "dXNlcjpwYXNz"),
        ("Proxy-Authorization: Bearer hidden-token", "hidden-token"),
        (
            "invalid x-api-key sk-ant-api03-SECRETSECRET",
            "sk-ant-api03",
        ),
        ("OPENAI_API_KEY=sk-proj-abc", "sk-proj-abc"),
        ("client_secret=abc123", "abc123"),
        ("refresh_token=abc", "abc"),
        ("token = abc123", "abc123"),
        (r#"{"password":"a secret with \"quotes\""}"#, "quotes"),
        (
            r"open C:\Users\John Smith\Videos\Wedding.mov failed",
            "Wedding",
        ),
        (
            "/Users/alice/My Projects/Client Secret Cut.opentake",
            "Client Secret Cut",
        ),
        (r"\\fileserver\share\Client\cut.mov", "fileserver"),
        (
            "/Users/alice/My [Client] (private)/Client Secret.mov",
            "Client Secret",
        ),
        (
            "standalone sk-live-123 and r8_provider_secret",
            "r8_provider_secret",
        ),
        ("Bearer opaque-token", "opaque-token"),
    ] {
        let scrubbed = redact_sensitive_text(input);
        assert!(
            !scrubbed.contains(forbidden),
            "credential or private path was retained"
        );
    }
    let path = redact_sensitive_text(r"open C:\Users\John Smith\Videos\Wedding.mov failed");
    assert!(!path.contains("John Smith"));
}

#[test]
fn ordinary_diagnostics_keep_their_text_and_whitespace() {
    for input in [
        "failed to decode frame 42",
        "decode failed: EOF\nplease retry\tframe 42",
        "request failed: https://api.example.com/v1/status",
    ] {
        assert_eq!(redact_sensitive_text(input), input);
    }
}
