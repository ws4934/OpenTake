//! Bounded, copy-on-write history metadata with immutable shared entries.
//!
//! Checkpoints retain two Arc handles, not N document clones. A speculative
//! edit copies at most LIMIT Arc pointers; popping materializes only its one
//! selected entry. Keeping whole stacks shared also makes rollback exact when
//! an attempted batch evicts old history or clears redo before failing.
use std::{collections::VecDeque, sync::Arc};

use super::HistoryEntry;

pub(super) const LIMIT: usize = 200;

#[derive(Clone, Debug, Default)]
pub(super) struct HistoryStack {
    entries: Arc<VecDeque<Arc<HistoryEntry>>>,
}

impl HistoryStack {
    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(super) fn last(&self) -> Option<&HistoryEntry> {
        self.entries.back().map(Arc::as_ref)
    }

    pub(super) fn push(&mut self, entry: HistoryEntry) {
        let entries = Arc::make_mut(&mut self.entries);
        entries.push_back(Arc::new(entry));
        if entries.len() > LIMIT {
            entries.pop_front();
        }
    }

    pub(super) fn pop(&mut self) -> Option<HistoryEntry> {
        let entry = Arc::make_mut(&mut self.entries).pop_back()?;
        Some(Arc::unwrap_or_clone(entry))
    }

    pub(super) fn clear(&mut self) {
        self.entries = Arc::default();
    }
}

#[cfg(test)]
mod tests {
    use super::super::ManifestDelta;
    use super::*;
    use opentake_domain::{MediaManifest, Timeline};

    fn entry(version: u64) -> HistoryEntry {
        HistoryEntry {
            timeline: Timeline::new(),
            manifest: ManifestDelta::between(&MediaManifest::new(), &MediaManifest::new()),
            action_name: "Edit".into(),
            transaction_version: version,
        }
    }

    #[test]
    fn checkpoint_shares_stacks_and_entries_until_mutation() {
        let mut history = HistoryStack::default();
        history.push(entry(1));
        let saved = history.clone();
        assert!(Arc::ptr_eq(&saved.entries, &history.entries));
        history.push(entry(2));
        assert!(!Arc::ptr_eq(&saved.entries, &history.entries));
        assert!(Arc::ptr_eq(&saved.entries[0], &history.entries[0]));
        assert_eq!(saved.len(), 1);
        assert_eq!(history.pop().unwrap().transaction_version, 2);
        history.clear();
        assert_eq!(saved.last().unwrap().transaction_version, 1);
    }
}
