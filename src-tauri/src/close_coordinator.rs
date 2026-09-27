//! One save flight shared by window close and user Quit.
use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum CloseIntent {
    Hide = 1,
    Exit = 2,
}

#[derive(Default)]
pub(crate) struct CloseCoordinator {
    pending: AtomicU8,
}

impl CloseCoordinator {
    /// Start one save, or promote an in-flight window close to Quit.
    pub(crate) fn request(&self, intent: CloseIntent) -> bool {
        self.pending.fetch_max(intent as u8, Ordering::AcqRel) == 0
    }

    /// A failed save releases the flight for retry and never yields an action.
    pub(crate) fn finish<E>(&self, saved: Result<(), E>) -> Result<CloseIntent, E> {
        let intent = match self.pending.swap(0, Ordering::AcqRel) {
            1 => CloseIntent::Hide,
            2 => CloseIntent::Exit,
            _ => unreachable!("only an admitted close flight can finish"),
        };
        saved.map(|()| intent)
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
                assert_eq!(coordinator.finish(Err(error)), Err(error));
                assert!(coordinator.request(intent));
                assert_eq!(coordinator.finish::<()>(Ok(())), Ok(intent));
            }
        }
    }
}
