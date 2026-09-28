//! Append-only AI generation audit log. Port of upstream `GenerationLog` /
//! `GenerationLogEntry` (`Editor/ViewModel/EditorViewModel+Cost.swift`),
//! persisted as `generation-log.json`.
//!
//! Two upstream tolerances are preserved verbatim:
//! - `version` defaults to `1` (the struct default and the missing-key
//!   fallback are both `1`, unlike `MediaManifest` whose default is 2 but
//!   fallback is 1).
//! - A row's cost migrates from the legacy dollar field: when `costCredits` is
//!   absent but `cost` (USD, a float) is present,
//!   `costCredits = ceil(cost * 100)` (Swift `(dollars * 100).rounded(.up)`).
//!
//! Dates: like the domain crate, `created_at` is Apple-reference-date seconds
//! (`f64`) — upstream's `JSONEncoder` default `Date` encoding. The
//! project/render layer converts to/from wall-clock time.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use opentake_domain::GenerationJobStatus;

/// Pretty-printed size [`GenerationLog::enforce_retention`] keeps the log
/// within: half the 16 MiB limit the `generation-log.json` reader enforces, so
/// the log cannot grow into a component its own reader refuses.
pub const GENERATION_LOG_RETENTION_BYTES: usize = 8 * 1024 * 1024;

/// Row id and model of the synthetic row that retention folds the oldest
/// finished rows into. Carrying their billed credits keeps
/// [`GenerationLog::total_credits`] unchanged.
const RETENTION_SUMMARY_ID: &str = "opentake:retention-summary";
const RETENTION_SUMMARY_MODEL: &str = "opentake:retention-summary";

fn default_version() -> i64 {
    1
}

/// The whole log. 1:1 with upstream `GenerationLog`.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct GenerationLog {
    /// Schema version. Defaults to 1 and falls back to 1 when missing. `i64` to
    /// match the width of upstream's Swift `Int` (64-bit on arm64).
    #[serde(default = "default_version")]
    pub version: i64,
    /// One row per AI generation, in append order.
    #[serde(default)]
    pub entries: Vec<GenerationLogEntry>,
}

impl Default for GenerationLog {
    fn default() -> Self {
        GenerationLog {
            version: 1,
            entries: Vec::new(),
        }
    }
}

impl GenerationLog {
    /// An empty log (`version = 1`).
    pub fn new() -> Self {
        GenerationLog::default()
    }

    /// Sum of `cost_credits` across rows (treating `None` as 0). Mirrors
    /// upstream `totalGenerationCost`.
    pub fn total_credits(&self) -> i64 {
        self.entries
            .iter()
            .map(|e| e.cost_credits.unwrap_or(0))
            .sum()
    }

    /// Drop the cost-free intermediate lifecycle rows (`generating`,
    /// `downloading`, `finalizing`) of every generation output whose latest
    /// recorded status is terminal (`ready`, `failed` or `cancelled`).
    ///
    /// Submission (`queued`), outcome and billed rows are kept, as are the
    /// rows of outputs that are still running and rows without a job identity
    /// (legacy and upstream audit rows). Returns the number of removed rows.
    pub fn compact_finished_jobs(&mut self) -> usize {
        let finished = finished_outputs(&self.entries);
        let intermediate = self
            .entries
            .iter()
            .map(|row| {
                row.cost_credits.is_none()
                    && matches!(
                        row.status,
                        Some(
                            GenerationJobStatus::Generating
                                | GenerationJobStatus::Downloading
                                | GenerationJobStatus::Finalizing
                        )
                    )
                    && output_key(row).is_some_and(|key| finished.contains(&key))
            })
            .collect::<Vec<_>>();
        let before = self.entries.len();
        let mut index = 0;
        self.entries.retain(|_| {
            index += 1;
            !intermediate[index - 1]
        });
        before - self.entries.len()
    }

    /// Keep the pretty-printed log, as it is persisted, within `max_bytes`.
    ///
    /// A log within the budget is left untouched. Otherwise finished jobs are
    /// compacted first ([`Self::compact_finished_jobs`]); if the log is still
    /// larger than half the budget, the oldest rows of finished outputs and
    /// legacy rows are folded into one leading summary row that carries their
    /// billed credits and latest date, so [`Self::total_credits`] does not
    /// change. Rows of outputs that are still running are never removed.
    /// Returns the number of removed rows.
    pub fn enforce_retention(&mut self, max_bytes: usize) -> usize {
        if pretty_len(self).0 <= max_bytes {
            return 0;
        }
        let mut removed = self.compact_finished_jobs();
        let target = max_bytes / 2;
        loop {
            let size = pretty_len(self).0;
            if size <= target {
                break;
            }
            let folded = self.fold_oldest_finished_rows(size - target);
            if folded == 0 {
                break;
            }
            removed += folded;
        }
        removed
    }

    /// Fold the oldest finished or legacy rows, estimated to free at least
    /// `excess` bytes, into the leading summary row. Returns the number of
    /// rows folded.
    fn fold_oldest_finished_rows(&mut self, mut excess: usize) -> usize {
        let finished = finished_outputs(&self.entries);
        let candidates = self
            .entries
            .iter()
            .map(|row| output_key(row).is_none_or(|key| finished.contains(&key)))
            .collect::<Vec<_>>();
        let mut summary: Option<GenerationLogEntry> = None;
        let mut kept = Vec::with_capacity(self.entries.len());
        let mut folded = 0;
        for (row, candidate) in self.entries.drain(..).zip(candidates) {
            if row.id == RETENTION_SUMMARY_ID && row.model == RETENTION_SUMMARY_MODEL {
                summary = Some(row);
                continue;
            }
            if excess == 0 || !candidate {
                kept.push(row);
                continue;
            }
            let (row_bytes, row_newlines) = pretty_len(&row);
            // Each row sits two levels deep in the pretty-printed log: every
            // line gains four spaces of indentation, plus a `,\n` separator.
            excess = excess.saturating_sub(row_bytes + 4 * (row_newlines + 1) + 2);
            let summary = summary.get_or_insert_with(|| {
                GenerationLogEntry::new(RETENTION_SUMMARY_ID, RETENTION_SUMMARY_MODEL, None, None)
            });
            if let Some(credits) = row.cost_credits {
                summary.cost_credits =
                    Some(summary.cost_credits.unwrap_or(0).saturating_add(credits));
            }
            if let Some(created_at) = row.created_at {
                summary.created_at = Some(
                    summary
                        .created_at
                        .map_or(created_at, |latest| latest.max(created_at)),
                );
            }
            folded += 1;
        }
        self.entries = summary.into_iter().chain(kept).collect();
        folded
    }
}

/// Outputs whose latest recorded lifecycle status is terminal.
fn finished_outputs(entries: &[GenerationLogEntry]) -> HashSet<(&str, &str)> {
    let mut latest = HashMap::new();
    for row in entries {
        if let (Some(key), Some(status)) = (output_key(row), row.status) {
            latest.insert(key, status);
        }
    }
    latest
        .into_iter()
        .filter(|(_, status)| {
            matches!(
                status,
                GenerationJobStatus::Ready
                    | GenerationJobStatus::Failed
                    | GenerationJobStatus::Cancelled
            )
        })
        .map(|(key, _)| key)
        .collect()
}

/// The generation output a lifecycle row belongs to.
fn output_key(row: &GenerationLogEntry) -> Option<(&str, &str)> {
    Some((row.job_id.as_deref()?, row.asset_id.as_deref()?))
}

/// Pretty-printed byte and newline count of `value`, without allocating it.
fn pretty_len(value: &impl Serialize) -> (usize, usize) {
    struct Counter {
        bytes: usize,
        newlines: usize,
    }
    impl std::io::Write for Counter {
        fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
            self.bytes += buffer.len();
            self.newlines += buffer.iter().filter(|byte| **byte == b'\n').count();
            Ok(buffer.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut counter = Counter {
        bytes: 0,
        newlines: 0,
    };
    // Encoding these plain values cannot fail; a failure would resurface as
    // the save's own encoding error, so measure it as empty here.
    match serde_json::to_writer_pretty(&mut counter, value) {
        Ok(()) => (counter.bytes, counter.newlines),
        Err(_) => (0, 0),
    }
}

/// One row in the project activity log. 1:1 with upstream `GenerationLogEntry`.
///
/// `id` is required on the wire when written by OpenTake, but tolerated as
/// missing on read (upstream synthesizes a UUID). An explicitly stored empty
/// string remains empty. `model` is required; `cost_credits` and `created_at`
/// are optional.
#[derive(Clone, PartialEq, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationLogEntry {
    /// Stable row id. Missing/null values synthesize a UUID; explicit empty
    /// strings remain empty.
    pub id: String,
    /// Model identifier used for the generation.
    pub model: String,
    /// Cost in credits. `None` when unknown. `i64` to match the width of
    /// upstream's Swift `Int` (64-bit on arm64).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost_credits: Option<i64>,
    /// Apple-reference-date seconds. `None` when unknown.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<f64>,
    /// Provider-neutral durable job identity. Never a signed URL or credential.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_job_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub asset_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<GenerationJobStatus>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<f64>,
    /// Fixed application-owned code only; provider diagnostic text is private.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_asset_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_clip_id: Option<String>,
}

impl GenerationLogEntry {
    /// Construct a row.
    pub fn new(
        id: impl Into<String>,
        model: impl Into<String>,
        cost_credits: Option<i64>,
        created_at: Option<f64>,
    ) -> Self {
        GenerationLogEntry {
            id: id.into(),
            model: model.into(),
            cost_credits,
            created_at,
            job_id: None,
            provider: None,
            provider_job_id: None,
            asset_id: None,
            status: None,
            progress: None,
            error_code: None,
            source_asset_id: None,
            source_clip_id: None,
        }
    }

    /// Construct one append-only job lifecycle event without provider secrets.
    #[allow(clippy::too_many_arguments)]
    pub fn job_event(
        id: impl Into<String>,
        job_id: impl Into<String>,
        model: impl Into<String>,
        cost_credits: Option<i64>,
        provider: impl Into<String>,
        provider_job_id: Option<String>,
        asset_id: impl Into<String>,
        status: GenerationJobStatus,
        progress: Option<f64>,
        error_code: Option<String>,
        created_at: Option<f64>,
        source_asset_id: Option<String>,
        source_clip_id: Option<String>,
    ) -> Self {
        Self {
            id: id.into(),
            job_id: Some(job_id.into()),
            model: model.into(),
            provider: Some(provider.into()),
            provider_job_id,
            asset_id: Some(asset_id.into()),
            status: Some(status),
            progress,
            error_code,
            cost_credits,
            created_at,
            source_asset_id,
            source_clip_id,
        }
    }
}

impl<'de> Deserialize<'de> for GenerationLogEntry {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Capture both the new `costCredits` and the legacy `cost` (USD float),
        // matching upstream's hand-written decoder.
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct Raw {
            id: Option<String>,
            model: String,
            cost_credits: Option<i64>,
            created_at: Option<f64>,
            // Legacy: dollars as a float. Only consulted when costCredits is absent.
            cost: Option<f64>,
            job_id: Option<String>,
            provider: Option<String>,
            provider_job_id: Option<String>,
            asset_id: Option<String>,
            status: Option<GenerationJobStatus>,
            progress: Option<f64>,
            error_code: Option<String>,
            source_asset_id: Option<String>,
            source_clip_id: Option<String>,
        }
        let raw = Raw::deserialize(deserializer)?;
        let cost_credits = match raw.cost_credits {
            Some(c) => Some(c),
            None => raw
                .cost
                // Swift: Int((dollars * 100).rounded(.up)) — ceil toward +inf.
                .map(|dollars| (dollars * 100.0).ceil() as i64),
        };
        Ok(GenerationLogEntry {
            id: raw.id.unwrap_or_else(|| Uuid::new_v4().to_string()),
            model: raw.model,
            cost_credits,
            created_at: raw.created_at,
            job_id: raw.job_id,
            provider: raw.provider,
            provider_job_id: raw.provider_job_id,
            asset_id: raw.asset_id,
            status: raw.status,
            progress: raw.progress,
            error_code: raw.error_code,
            source_asset_id: raw.source_asset_id,
            source_clip_id: raw.source_clip_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_and_new_version_is_one() {
        assert_eq!(GenerationLog::default().version, 1);
        assert_eq!(GenerationLog::new().version, 1);
        assert!(GenerationLog::new().entries.is_empty());
    }

    #[test]
    fn missing_version_falls_back_to_one() {
        let log: GenerationLog = serde_json::from_str(r#"{"entries":[]}"#).unwrap();
        assert_eq!(log.version, 1);
        let log2: GenerationLog = serde_json::from_str("{}").unwrap();
        assert_eq!(log2.version, 1);
        assert!(log2.entries.is_empty());
    }

    #[test]
    fn entry_roundtrip_camel_case() {
        let e = GenerationLogEntry::new("row-1", "veo-3", Some(250), Some(700_000_000.0));
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains("\"costCredits\":250"));
        assert!(json.contains("\"createdAt\":700000000.0"));
        assert!(json.contains("\"model\":\"veo-3\""));
        let back: GenerationLogEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(e, back);
    }

    #[test]
    fn entry_omits_none_fields() {
        let e = GenerationLogEntry::new("row-2", "m", None, None);
        let json = serde_json::to_string(&e).unwrap();
        assert!(!json.contains("costCredits"));
        assert!(!json.contains("createdAt"));
        // id and model always present
        assert!(json.contains("\"id\":\"row-2\""));
    }

    #[test]
    fn legacy_cost_dollars_migrates_to_credits_ceil() {
        // 1.23 USD -> ceil(123.0) = 123
        let e: GenerationLogEntry =
            serde_json::from_str(r#"{"id":"a","model":"m","cost":1.23}"#).unwrap();
        assert_eq!(e.cost_credits, Some(123));
        // 0.005 USD -> ceil(0.5) = 1 (rounds up, never truncates)
        let e2: GenerationLogEntry =
            serde_json::from_str(r#"{"id":"b","model":"m","cost":0.005}"#).unwrap();
        assert_eq!(e2.cost_credits, Some(1));
        // exact: 2.00 USD -> 200
        let e3: GenerationLogEntry = serde_json::from_str(r#"{"model":"m","cost":2.0}"#).unwrap();
        assert_eq!(e3.cost_credits, Some(200));
    }

    #[test]
    fn cost_credits_wins_over_legacy_cost() {
        // When both present, costCredits is authoritative (upstream consults
        // legacy `cost` only when costCredits is absent).
        let e: GenerationLogEntry =
            serde_json::from_str(r#"{"model":"m","costCredits":7,"cost":99.0}"#).unwrap();
        assert_eq!(e.cost_credits, Some(7));
    }

    #[test]
    fn missing_or_null_id_gets_uuid_but_explicit_empty_is_preserved() {
        let e: GenerationLogEntry = serde_json::from_str(r#"{"model":"m"}"#).unwrap();
        let id = Uuid::parse_str(&e.id).expect("missing generation id gets UUID");
        assert_eq!(id.get_version_num(), 4);
        assert_eq!(e.cost_credits, None);
        assert_eq!(e.created_at, None);

        let null: GenerationLogEntry = serde_json::from_str(r#"{"id":null,"model":"m"}"#).unwrap();
        assert_eq!(Uuid::parse_str(&null.id).unwrap().get_version_num(), 4);

        let empty: GenerationLogEntry = serde_json::from_str(r#"{"id":"","model":"m"}"#).unwrap();
        assert_eq!(empty.id, "");
    }

    #[test]
    fn total_credits_sums_treating_none_as_zero() {
        let log = GenerationLog {
            version: 1,
            entries: vec![
                GenerationLogEntry::new("a", "m", Some(100), None),
                GenerationLogEntry::new("b", "m", None, None),
                GenerationLogEntry::new("c", "m", Some(50), None),
            ],
        };
        assert_eq!(log.total_credits(), 150);
    }

    fn job_row(
        id: usize,
        job: &str,
        asset: &str,
        status: GenerationJobStatus,
        cost_credits: Option<i64>,
    ) -> GenerationLogEntry {
        GenerationLogEntry::job_event(
            format!("row-{id}"),
            job,
            "fal:fixture-model",
            cost_credits,
            "fal",
            Some(format!("fal::{job}")),
            asset,
            status,
            None,
            None,
            Some(800_000_000.0 + id as f64),
            None,
            None,
        )
    }

    fn statuses(log: &GenerationLog) -> Vec<(Option<&str>, Option<GenerationJobStatus>)> {
        log.entries
            .iter()
            .map(|row| (row.job_id.as_deref(), row.status))
            .collect()
    }

    #[test]
    fn compaction_keeps_submissions_outcomes_and_billed_rows_of_finished_outputs() {
        use GenerationJobStatus as Status;
        let mut log = GenerationLog {
            version: 1,
            entries: vec![
                GenerationLogEntry::new("legacy", "veo-3", Some(40), Some(1.0)),
                job_row(1, "done", "a", Status::Queued, None),
                job_row(2, "done", "a", Status::Generating, None),
                job_row(3, "done", "a", Status::Downloading, Some(7)),
                job_row(4, "done", "a", Status::Finalizing, None),
                job_row(5, "done", "a", Status::Ready, None),
                job_row(6, "retry", "b", Status::Queued, None),
                job_row(7, "retry", "b", Status::Generating, None),
                job_row(8, "retry", "b", Status::Failed, None),
                job_row(9, "retry", "b", Status::Queued, None),
                job_row(10, "retry", "b", Status::Generating, None),
            ],
        };
        let credits = log.total_credits();

        assert_eq!(log.compact_finished_jobs(), 2);

        assert_eq!(
            statuses(&log),
            [
                (None, None),
                (Some("done"), Some(Status::Queued)),
                (Some("done"), Some(Status::Downloading)),
                (Some("done"), Some(Status::Ready)),
                (Some("retry"), Some(Status::Queued)),
                (Some("retry"), Some(Status::Generating)),
                (Some("retry"), Some(Status::Failed)),
                (Some("retry"), Some(Status::Queued)),
                (Some("retry"), Some(Status::Generating)),
            ]
        );
        assert_eq!(log.total_credits(), credits);
    }

    #[test]
    fn retention_leaves_a_log_within_its_budget_untouched() {
        let mut log = GenerationLog {
            version: 1,
            entries: vec![
                job_row(1, "job", "a", GenerationJobStatus::Queued, None),
                job_row(2, "job", "a", GenerationJobStatus::Generating, None),
                job_row(3, "job", "a", GenerationJobStatus::Ready, Some(3)),
            ],
        };
        let before = log.clone();

        assert_eq!(log.enforce_retention(GENERATION_LOG_RETENTION_BYTES), 0);
        assert_eq!(log, before);
    }

    #[test]
    fn retention_folds_the_oldest_finished_rows_and_preserves_credits() {
        use GenerationJobStatus as Status;
        let mut entries = vec![GenerationLogEntry::new("legacy", "veo-3", Some(40), None)];
        for job in 0..200 {
            let name = format!("job-{job}");
            entries.push(job_row(job * 3, &name, "a", Status::Queued, None));
            entries.push(job_row(job * 3 + 1, &name, "a", Status::Generating, None));
            entries.push(job_row(job * 3 + 2, &name, "a", Status::Ready, Some(2)));
        }
        entries.push(job_row(9_000, "running", "r", Status::Queued, None));
        entries.push(job_row(9_001, "running", "r", Status::Generating, None));
        let mut log = GenerationLog {
            version: 1,
            entries,
        };
        let credits = log.total_credits();
        let max_bytes = 32 * 1024;
        assert!(pretty_len(&log).0 > max_bytes);

        let removed = log.enforce_retention(max_bytes);

        assert!(removed > 200, "compaction alone cannot fit the budget");
        assert!(pretty_len(&log).0 <= max_bytes / 2);
        assert_eq!(log.total_credits(), credits);
        let summary = &log.entries[0];
        assert_eq!(summary.id, RETENTION_SUMMARY_ID);
        assert_eq!(summary.job_id, None);
        // The newest finished job and the running job survive intact.
        let last_job = log
            .entries
            .iter()
            .filter(|row| row.job_id.as_deref() == Some("job-199"))
            .map(|row| row.status)
            .collect::<Vec<_>>();
        assert_eq!(last_job, [Some(Status::Queued), Some(Status::Ready)]);
        let running = log
            .entries
            .iter()
            .filter(|row| row.job_id.as_deref() == Some("running"))
            .count();
        assert_eq!(running, 2);

        // A second pass merges into the same summary instead of adding one.
        for job in 200..400 {
            let name = format!("job-{job}");
            log.entries
                .push(job_row(job * 3, &name, "a", Status::Queued, None));
            log.entries
                .push(job_row(job * 3 + 2, &name, "a", Status::Ready, Some(2)));
        }
        let credits = log.total_credits();
        log.enforce_retention(max_bytes);
        assert_eq!(log.total_credits(), credits);
        assert_eq!(
            log.entries
                .iter()
                .filter(|row| row.id == RETENTION_SUMMARY_ID)
                .count(),
            1
        );
        assert!(pretty_len(&log).0 <= max_bytes / 2);
    }

    #[test]
    fn retention_never_removes_rows_of_running_outputs() {
        let mut log = GenerationLog {
            version: 1,
            entries: (0..400)
                .map(|row| {
                    job_row(
                        row,
                        &format!("running-{row}"),
                        "a",
                        GenerationJobStatus::Generating,
                        None,
                    )
                })
                .collect(),
        };
        let before = log.clone();

        assert_eq!(log.enforce_retention(4 * 1024), 0);
        assert_eq!(log, before);
    }

    #[test]
    fn retention_budget_stays_below_the_reader_limit() {
        const {
            assert!(
                GENERATION_LOG_RETENTION_BYTES
                    < crate::project_root::GENERATION_LOG_COMPONENT_MAX_BYTES
            )
        };
    }

    #[test]
    fn full_log_roundtrip() {
        let log = GenerationLog {
            version: 1,
            entries: vec![GenerationLogEntry::new("a", "veo-3", Some(250), Some(1.0))],
        };
        let json = serde_json::to_string(&log).unwrap();
        let back: GenerationLog = serde_json::from_str(&json).unwrap();
        assert_eq!(log, back);
    }
}
