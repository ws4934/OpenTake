//! One save flight shared by window close and user Quit, and the choice that
//! follows a failed save.
use std::sync::atomic::{AtomicU8, Ordering};

use serde::Deserialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum CloseIntent {
    Hide = 1,
    Exit = 2,
}

impl CloseIntent {
    fn from_bits(bits: u8) -> Option<Self> {
        match bits & INTENT_BITS {
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
}

const INTENT_BITS: u8 = 0b11;
/// Set once the WebView or the native fallback has taken over the prompt.
const PROMPT_CLAIMED: u8 = 0b100;

#[derive(Default)]
pub(crate) struct CloseCoordinator {
    pending: AtomicU8,
    /// The intent of a close whose save failed and awaits the user's choice,
    /// plus [`PROMPT_CLAIMED`] once one prompt shows it.
    failed: AtomicU8,
}

impl CloseCoordinator {
    /// Start one save, or promote an in-flight window close to Quit. A new
    /// request supersedes a failed close still waiting for its choice.
    pub(crate) fn request(&self, intent: CloseIntent) -> bool {
        let admitted = self.pending.fetch_max(intent as u8, Ordering::AcqRel) == 0;
        if admitted {
            self.failed.store(0, Ordering::Release);
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
    pub(crate) fn record_failure(&self, intent: CloseIntent) {
        self.failed.store(intent as u8, Ordering::Release);
    }

    /// Let exactly one prompt (the WebView's, or the native fallback when the
    /// WebView never answers) show the pending choice.
    pub(crate) fn claim_prompt(&self) -> Option<CloseIntent> {
        let current = self.failed.load(Ordering::Acquire);
        let intent = CloseIntent::from_bits(current)?;
        if current & PROMPT_CLAIMED != 0 {
            return None;
        }
        self.failed
            .compare_exchange(
                current,
                current | PROMPT_CLAIMED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .ok()
            .map(|_| intent)
    }

    /// Consume the pending failed close; `None` when nothing is waiting (for
    /// example after a newer close request superseded it).
    pub(crate) fn take_failure(&self) -> Option<CloseIntent> {
        CloseIntent::from_bits(self.failed.swap(0, Ordering::AcqRel))
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
        assert_eq!(coordinator.claim_prompt(), None);
        assert_eq!(coordinator.take_failure(), None);

        coordinator.record_failure(CloseIntent::Exit);
        assert_eq!(coordinator.claim_prompt(), Some(CloseIntent::Exit));
        assert_eq!(coordinator.claim_prompt(), None, "one prompt at a time");
        assert_eq!(coordinator.take_failure(), Some(CloseIntent::Exit));
        assert_eq!(coordinator.take_failure(), None, "a choice applies once");
        assert_eq!(coordinator.claim_prompt(), None);
    }

    #[test]
    fn a_new_close_request_supersedes_a_waiting_failure() {
        let coordinator = CloseCoordinator::default();
        assert!(coordinator.request(CloseIntent::Hide));
        let (intent, _) = coordinator.finish(Err("volume unplugged")).unwrap_err();
        coordinator.record_failure(intent);
        assert!(coordinator.request(CloseIntent::Exit));
        assert_eq!(coordinator.take_failure(), None);
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
        ] {
            assert_eq!(
                serde_json::from_str::<FailedCloseChoice>(raw).unwrap(),
                choice
            );
        }
    }
}
