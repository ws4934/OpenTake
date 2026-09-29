//! Process-wide playback/export pressure (`ExportPause`) shared by every
//! `MediaEngine` in the app. Export and native playback hold a guard while they
//! run; the bounded inference worker (visual indexing, transcription, semantic
//! search) waits at job boundaries while any guard is alive, mirroring upstream
//! `ExportService.isExporting`.
//!
//! There is exactly one counter per process: the UI's `MediaState` engine, the
//! MCP media bridge, generation, job-local engines and the inference worker all
//! observe the same instance, so no caller can bind the worker to a private
//! counter that nothing ever raises.

use std::path::PathBuf;
use std::sync::OnceLock;

use opentake_media::{ExportPause, ExportPauseGuard, MediaEngine};

/// The single process-wide pressure counter.
pub(crate) fn process_export_pause() -> ExportPause {
    static PAUSE: OnceLock<ExportPause> = OnceLock::new();
    PAUSE.get_or_init(ExportPause::new).clone()
}

/// A production engine rooted at `cache_root`/`models_dir` that shares the
/// process-wide pressure counter.
pub(crate) fn production_media_engine(
    cache_root: impl Into<PathBuf>,
    models_dir: impl Into<PathBuf>,
) -> MediaEngine {
    MediaEngine::new(cache_root, models_dir).with_export_pause(process_export_pause())
}

/// Hold playback/export pressure for the lifetime of one export run (ordinary
/// export, save-range and save-clip). Dropping the guard on success, failure,
/// cancellation or unwind releases it.
pub(crate) fn export_guard() -> ExportPauseGuard {
    let guard = export_pause_for_current_thread().guard();
    #[cfg(test)]
    test_support::observe_export_guard();
    guard
}

#[cfg(not(test))]
fn export_pause_for_current_thread() -> ExportPause {
    process_export_pause()
}

#[cfg(test)]
fn export_pause_for_current_thread() -> ExportPause {
    test_support::OVERRIDE
        .with(|slot| slot.borrow().clone())
        .unwrap_or_else(process_export_pause)
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::cell::RefCell;

    use opentake_media::ExportPause;

    thread_local! {
        pub(super) static OVERRIDE: RefCell<Option<ExportPause>> = const { RefCell::new(None) };
        static OBSERVER: RefCell<Option<Box<dyn Fn()>>> = const { RefCell::new(None) };
    }

    /// Route export guards taken on this thread to `pause` and call `observer`
    /// right after each one is acquired, so a test can inspect the counter
    /// while an export is running without racing parallel tests.
    pub(crate) fn isolate_export_pressure(pause: ExportPause, observer: impl Fn() + 'static) {
        OVERRIDE.with(|slot| *slot.borrow_mut() = Some(pause));
        OBSERVER.with(|slot| *slot.borrow_mut() = Some(Box::new(observer)));
    }

    pub(super) fn observe_export_guard() {
        OBSERVER.with(|slot| {
            if let Some(observer) = slot.borrow().as_ref() {
                observer();
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_isolated_export(
        control: Option<&crate::export::ExportControl>,
        out_path: &std::path::Path,
    ) -> (Result<crate::export::ExportSummary, String>, bool, bool) {
        use std::cell::Cell;
        use std::rc::Rc;

        let pause = ExportPause::new();
        let observed = Rc::new(Cell::new(false));
        let seen = observed.clone();
        let observed_pause = pause.clone();
        test_support::isolate_export_pressure(pause.clone(), move || {
            seen.set(observed_pause.is_active());
        });
        let result = crate::export::run_export_with_control(
            &opentake_domain::Timeline::new(),
            &opentake_domain::MediaManifest::default(),
            &None,
            &crate::export::ExportRequest {
                out_path: out_path.to_string_lossy().into_owned(),
                codec: crate::export::ExportCodec::H264,
                quality: crate::export::ExportQuality::P720,
            },
            crate::export::ExportRunOptions {
                control,
                ..Default::default()
            },
        );
        (result, observed.get(), pause.is_active())
    }

    #[test]
    fn export_holds_pressure_while_running_and_releases_it_when_cancelled() {
        let temp = tempfile::tempdir().unwrap();
        let control = crate::export::ExportControl::default();
        let _lease = control.try_begin("pressure-cancel").unwrap();
        assert!(control.request_cancel("pressure-cancel"));
        let (result, active_during, active_after) =
            run_isolated_export(Some(&control), &temp.path().join("out.mp4"));
        assert_eq!(result.unwrap_err(), crate::export::CANCELLED_SENTINEL);
        assert!(active_during, "export must raise pressure before any work");
        assert!(!active_after, "cancelled export must release pressure");
    }

    #[test]
    fn export_releases_pressure_when_it_fails() {
        let temp = tempfile::tempdir().unwrap();
        // An unsupported container fails preset resolution after the guard.
        let (result, active_during, active_after) =
            run_isolated_export(None, &temp.path().join("out.unsupported"));
        assert!(result.is_err());
        assert!(active_during);
        assert!(!active_after, "failed export must release pressure");
    }

    #[test]
    fn production_worker_does_not_start_background_jobs_during_export() {
        use opentake_media::ort_worker::{JobKind, JobPriority, JobRequest, JobState};
        use std::time::{Duration, Instant};

        let worker = crate::search::production_index_worker();
        let guard = process_export_pause().guard();
        let job = worker
            .submit(
                JobRequest::new(
                    JobKind::Index,
                    "pressure-test",
                    format!("pressure-test-{:?}", Instant::now()),
                    JobPriority::Background,
                ),
                |_, _| Ok(1usize),
            )
            .unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(job.state(), JobState::Queued);
        drop(guard);
        // Resumes once pressure clears (other tests may briefly hold the
        // shared counter, which only delays this).
        assert_eq!(job.wait(), Ok(1));
    }

    #[test]
    fn production_engines_share_the_process_counter() {
        let a = production_media_engine("/a", "/m");
        let b = production_media_engine("/b", "/m");
        assert!(a.export_pause().ptr_eq(&b.export_pause()));
        assert!(a.export_pause().ptr_eq(&process_export_pause()));
    }
}
