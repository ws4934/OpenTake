//! Cached answer to "can this install start a generation?" for the Agent's
//! tool catalog (issue #6). `can_generate` runs on every tool dispatch, every
//! `tools/list` and every chat turn, and answering it reads several keychain
//! items, which is synchronous IPC to the OS credential store. The answer is
//! kept until a generation credential changes (BYOK key saved or deleted,
//! account sign-in or sign-out, backend URL change) or a short TTL expires,
//! so a keychain that was locked and is unlocked later is noticed too.
//! Submitting a generation still reads the keychain directly.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Bumped whenever a generation credential may have changed.
static CREDENTIALS_EPOCH: AtomicU64 = AtomicU64::new(0);

const AVAILABILITY_TTL: Duration = Duration::from_secs(10);

/// Drop every cached availability answer; call after changing a BYOK key,
/// the account credential or the account backend URL.
pub(crate) fn invalidate() {
    CREDENTIALS_EPOCH.fetch_add(1, Ordering::AcqRel);
}

pub(crate) struct GenerationAvailabilityCache {
    epoch: &'static AtomicU64,
    ttl: Duration,
    cached: Mutex<Option<CachedAvailability>>,
}

struct CachedAvailability {
    epoch: u64,
    read_at: Instant,
    available: bool,
}

impl GenerationAvailabilityCache {
    pub(crate) fn new() -> Self {
        Self::with_epoch(&CREDENTIALS_EPOCH, AVAILABILITY_TTL)
    }

    pub(crate) fn with_epoch(epoch: &'static AtomicU64, ttl: Duration) -> Self {
        Self {
            epoch,
            ttl,
            cached: Mutex::new(None),
        }
    }

    /// The cached answer, or `read()` when there is none for the current
    /// credential epoch or it is older than the TTL. The lock is held while
    /// reading, so concurrent callers share one keychain read.
    pub(crate) fn get(&self, read: impl FnOnce() -> bool) -> bool {
        let mut cached = self
            .cached
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Load the epoch before reading: a change during the read leaves a
        // stale epoch behind, so the next call reads again.
        let epoch = self.epoch.load(Ordering::Acquire);
        if let Some(cached) = cached.as_ref() {
            if cached.epoch == epoch && cached.read_at.elapsed() < self.ttl {
                return cached.available;
            }
        }
        let available = read();
        *cached = Some(CachedAvailability {
            epoch,
            read_at: Instant::now(),
            available,
        });
        available
    }
}

#[cfg(test)]
pub(crate) fn test_epoch() -> &'static AtomicU64 {
    Box::leak(Box::new(AtomicU64::new(0)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn reads_once_until_invalidated() {
        let epoch = test_epoch();
        let cache = GenerationAvailabilityCache::with_epoch(epoch, Duration::from_secs(3600));
        let reads = AtomicUsize::new(0);
        let read = |value| {
            reads.fetch_add(1, Ordering::SeqCst);
            value
        };
        assert!(!cache.get(|| read(false)));
        assert!(!cache.get(|| read(true)), "the cached answer is kept");
        assert_eq!(reads.load(Ordering::SeqCst), 1);

        epoch.fetch_add(1, Ordering::AcqRel);
        assert!(cache.get(|| read(true)), "a credential change rereads");
        assert!(cache.get(|| read(false)));
        assert_eq!(reads.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn an_expired_answer_is_read_again() {
        let cache = GenerationAvailabilityCache::with_epoch(test_epoch(), Duration::ZERO);
        assert!(!cache.get(|| false));
        assert!(cache.get(|| true));
    }

    #[test]
    fn the_production_cache_follows_invalidate() {
        let cache = GenerationAvailabilityCache::new();
        let before = CREDENTIALS_EPOCH.load(Ordering::Acquire);
        cache.get(|| false);
        invalidate();
        assert!(CREDENTIALS_EPOCH.load(Ordering::Acquire) > before);
        assert!(cache.get(|| true));
    }
}
