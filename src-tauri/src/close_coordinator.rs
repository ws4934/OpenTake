//! One save flight shared by window close and user Quit, and the choice that
//! follows a failed save.
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, PoisonError};

use serde::Deserialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum CloseIntent {
    Hide = 1,
    Exit = 2,
}

impl CloseIntent {
    fn from_bits(bits: u8) -> Option<Self> {
        match bits {
            1 => Some(Self::Hide),
            2 => Some(Self::Exit),
            _ => None,
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Hide => "hide",
            Self::Exit => "exit",
        }
    }
}

/// What the user chose after the save before a close or quit failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) enum FailedCloseChoice {
    /// The project was saved elsewhere (Save As): save again and close.
    Retry,
    /// Close without saving. Unsaved edits are dropped; nothing is deleted.
    Discard,
    /// Keep the project open.
    Cancel,
    /// The WebView could not show its prompt: hand it to the native one.
    Native,
}

/// One close or quit whose save failed and that waits for the user's choice.
/// The id distinguishes it from any later failure, so a stale prompt, timer
/// or answer never acts on a newer one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FailedClose {
    pub(crate) id: u64,
    pub(crate) intent: CloseIntent,
    pub(crate) message: String,
}

#[derive(Default)]
struct FailedState {
    next_id: u64,
    /// The waiting failure and whether a prompt (WebView or native) shows it.
    pending: Option<(FailedClose, bool)>,
}

#[derive(Default)]
pub(crate) struct CloseCoordinator {
    pending: AtomicU8,
    failed: Mutex<FailedState>,
}

impl CloseCoordinator {
    fn failed(&self) -> std::sync::MutexGuard<'_, FailedState> {
        // The state is updated in single assignments, so it stays consistent
        // even if a holder panicked.
        self.failed.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Start one save, or promote an in-flight window close to Quit. A new
    /// request supersedes a failed close still waiting for its choice.
    pub(crate) fn request(&self, intent: CloseIntent) -> bool {
        let admitted = self.pending.fetch_max(intent as u8, Ordering::AcqRel) == 0;
        if admitted {
            self.failed().pending = None;
        }
        admitted
    }

    /// A failed save releases the flight for retry and never yields an action;
    /// the error carries the intent so the caller can offer a choice.
    pub(crate) fn finish<E>(&self, saved: Result<(), E>) -> Result<CloseIntent, (CloseIntent, E)> {
        let intent = CloseIntent::from_bits(self.pending.swap(0, Ordering::AcqRel))
            .expect("only an admitted close flight can finish");
        saved.map(|()| intent).map_err(|error| (intent, error))
    }

    /// Remember a close whose save failed until the user chooses what to do.
    pub(crate) fn record_failure(&self, intent: CloseIntent, message: String) -> FailedClose {
        let mut state = self.failed();
        state.next_id += 1;
        let failure = FailedClose {
            id: state.next_id,
            intent,
            message,
        };
        state.pending = Some((failure.clone(), false));
        failure
    }

    /// Let exactly one prompt (the WebView's, or the native fallback when the
    /// WebView never answers) show failure `id`, if it is still waiting.
    pub(crate) fn claim_prompt(&self, id: u64) -> Option<FailedClose> {
        match &mut self.failed().pending {
            Some((failure, claimed)) if failure.id == id && !*claimed => {
                *claimed = true;
                Some(failure.clone())
            }
            _ => None,
        }
    }

    /// Return a claimed failure to the unclaimed state, so the native prompt
    /// can show it after the WebView failed to.
    pub(crate) fn release_prompt(&self, id: u64) -> Option<FailedClose> {
        match &mut self.failed().pending {
            Some((failure, claimed)) if failure.id == id && *claimed => {
                *claimed = false;
                Some(failure.clone())
            }
            _ => None,
        }
    }

    /// Consume failure `id`; `None` when it is no longer waiting (a newer
    /// close request superseded it, or it was already answered).
    pub(crate) fn take_failure(&self, id: u64) -> Option<FailedClose> {
        let mut state = self.failed();
        match &state.pending {
            Some((failure, _)) if failure.id == id => {
                state.pending.take().map(|(failure, _)| failure)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_requests_share_a_save_and_quit_promotes_close() {
        let coordinator = CloseCoordinator::default();
        assert!(coordinator.request(CloseIntent::Hide));
        assert!(!coordinator.request(CloseIntent::Hide));
        assert!(!coordinator.request(CloseIntent::Exit));
        assert!(!coordinator.request(CloseIntent::Hide));
        assert_eq!(coordinator.finish::<()>(Ok(())), Ok(CloseIntent::Exit));
    }

    #[test]
    fn failure_and_timeout_refuse_hide_or_exit_and_allow_retry() {
        for intent in [CloseIntent::Hide, CloseIntent::Exit] {
            for error in ["disk full", "save timed out"] {
                let coordinator = CloseCoordinator::default();
                assert!(coordinator.request(intent));
                assert_eq!(coordinator.finish(Err(error)), Err((intent, error)));
                assert!(coordinator.request(intent));
                assert_eq!(coordinator.finish::<()>(Ok(())), Ok(intent));
            }
        }
    }

    #[test]
    fn failed_close_is_prompted_once_and_resolved_once() {
        let coordinator = CloseCoordinator::default();
        assert_eq!(coordinator.claim_prompt(1), None);
        assert_eq!(coordinator.take_failure(1), None);

        let failure = coordinator.record_failure(CloseIntent::Exit, "disk full".into());
        assert_eq!(failure.message, "disk full");
        assert_eq!(coordinator.claim_prompt(failure.id), Some(failure.clone()));
        assert_eq!(
            coordinator.claim_prompt(failure.id),
            None,
            "one prompt at a time"
        );
        assert_eq!(coordinator.take_failure(failure.id), Some(failure.clone()));
        assert_eq!(
            coordinator.take_failure(failure.id),
            None,
            "a choice applies once"
        );
        assert_eq!(coordinator.claim_prompt(failure.id), None);
    }

    #[test]
    fn a_released_prompt_can_be_claimed_again() {
        let coordinator = CloseCoordinator::default();
        let failure = coordinator.record_failure(CloseIntent::Hide, "unplugged".into());
        assert_eq!(
            coordinator.release_prompt(failure.id),
            None,
            "not claimed yet"
        );
        assert!(coordinator.claim_prompt(failure.id).is_some());
        assert_eq!(
            coordinator.release_prompt(failure.id),
            Some(failure.clone())
        );
        assert_eq!(coordinator.claim_prompt(failure.id), Some(failure));
    }

    #[test]
    fn stale_prompts_timers_and_answers_never_act_on_a_newer_failure() {
        let coordinator = CloseCoordinator::default();
        let old = coordinator.record_failure(CloseIntent::Hide, "old error".into());
        let new = coordinator.record_failure(CloseIntent::Exit, "new error".into());
        assert_ne!(old.id, new.id);
        assert_eq!(coordinator.claim_prompt(old.id), None);
        assert_eq!(coordinator.release_prompt(old.id), None);
        assert_eq!(coordinator.take_failure(old.id), None);
        assert_eq!(coordinator.claim_prompt(new.id), Some(new.clone()));
        assert_eq!(coordinator.take_failure(new.id), Some(new));
    }

    #[test]
    fn a_new_close_request_supersedes_a_waiting_failure() {
        let coordinator = CloseCoordinator::default();
        assert!(coordinator.request(CloseIntent::Hide));
        let (intent, message) = coordinator.finish(Err("volume unplugged")).unwrap_err();
        let failure = coordinator.record_failure(intent, message.into());
        assert!(coordinator.request(CloseIntent::Exit));
        assert_eq!(coordinator.take_failure(failure.id), None);
        assert_eq!(coordinator.finish::<()>(Ok(())), Ok(CloseIntent::Exit));
    }

    #[test]
    fn choices_use_camel_case_ipc_names() {
        assert_eq!(CloseIntent::Hide.as_str(), "hide");
        assert_eq!(CloseIntent::Exit.as_str(), "exit");
        for (raw, choice) in [
            ("\"retry\"", FailedCloseChoice::Retry),
            ("\"discard\"", FailedCloseChoice::Discard),
            ("\"cancel\"", FailedCloseChoice::Cancel),
            ("\"native\"", FailedCloseChoice::Native),
        ] {
            assert_eq!(
                serde_json::from_str::<FailedCloseChoice>(raw).unwrap(),
                choice
            );
        }
    }
}
