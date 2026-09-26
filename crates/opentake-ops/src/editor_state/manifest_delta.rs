//! A history step owns only the manifest changes made by its command.
//!
//! Imports, relinking, proxies and generation update the live manifest outside
//! command history. Replacing a manifest snapshot would roll those writes back.
use std::collections::{HashMap, HashSet};

use opentake_domain::{MediaFolder, MediaManifest, MediaManifestEntry};

#[derive(Clone, Debug)]
pub(super) struct ManifestDelta {
    entries: CollectionDelta<MediaManifestEntry>,
    folders: CollectionDelta<MediaFolder>,
}

impl ManifestDelta {
    /// Build a directional patch from `from` to `to`, retaining only changed items.
    pub(super) fn between(from: &MediaManifest, to: &MediaManifest) -> Self {
        // Exhaustive classification: format version and favorite mirrors belong
        // to persistence/library state, not to the editor's undo history.
        let MediaManifest {
            entries,
            folders,
            version: _,
            favorites: _,
            favorite_library_ids: _,
        } = to;
        Self {
            entries: CollectionDelta::between(&from.entries, entries),
            folders: CollectionDelta::between(&from.folders, folders),
        }
    }

    pub(super) fn apply(&self, manifest: &mut MediaManifest) {
        self.entries.apply(&mut manifest.entries);
        self.folders.apply(&mut manifest.folders);
        // A later import or generation may now live in a folder that this
        // history step removes. Preserve the asset and expose it at the root,
        // rather than leaving an invisible dangling folder reference. The
        // inverse delta records this reparenting so Redo can restore it.
        for entry in &mut manifest.entries {
            if entry
                .folder_id
                .as_ref()
                .is_some_and(|id| self.folders.removed.contains(id))
            {
                entry.folder_id = None;
            }
        }
        for folder in &mut manifest.folders {
            if folder
                .parent_folder_id
                .as_ref()
                .is_some_and(|id| self.folders.removed.contains(id))
            {
                folder.parent_folder_id = None;
            }
        }
    }
}

#[derive(Clone, Debug)]
struct CollectionDelta<T> {
    removed: HashSet<String>,
    inserted: Vec<(usize, T)>,
    changed: Vec<(T, T)>,
}

trait HistoryItem: Clone + PartialEq {
    fn id(&self) -> &str;
    fn restore_fields(&mut self, from: &Self, to: &Self);
}

impl<T: HistoryItem> CollectionDelta<T> {
    fn between(from: &[T], to: &[T]) -> Self {
        let from_by_id: HashMap<_, _> = from.iter().map(|item| (item.id(), item)).collect();
        let to_ids: HashSet<_> = to.iter().map(HistoryItem::id).collect();
        let removed = from
            .iter()
            .filter(|item| !to_ids.contains(item.id()))
            .map(|item| item.id().to_owned())
            .collect();
        let mut inserted = Vec::new();
        let mut changed = Vec::new();
        for (index, item) in to.iter().enumerate() {
            match from_by_id.get(item.id()) {
                Some(previous) if *previous != item => {
                    changed.push(((*previous).clone(), item.clone()));
                }
                None => inserted.push((index, item.clone())),
                _ => {}
            }
        }
        Self {
            removed,
            inserted,
            changed,
        }
    }

    fn apply(&self, items: &mut Vec<T>) {
        if !self.removed.is_empty() {
            items.retain(|item| !self.removed.contains(item.id()));
        }
        if !self.inserted.is_empty() {
            let existing: HashSet<_> = items.iter().map(|item| item.id().to_owned()).collect();
            let mut remaining = std::mem::take(items).into_iter();
            let mut restored = Vec::with_capacity(remaining.len() + self.inserted.len());
            // Insertions are in original-index order. Merge in one pass instead
            // of repeatedly shifting the entire library with Vec::insert.
            for (index, item) in &self.inserted {
                if existing.contains(item.id()) {
                    continue;
                }
                while restored.len() < *index {
                    let Some(current) = remaining.next() else {
                        break;
                    };
                    restored.push(current);
                }
                restored.push(item.clone());
            }
            restored.extend(remaining);
            *items = restored;
        }
        if !self.changed.is_empty() {
            let changes: HashMap<_, _> = self
                .changed
                .iter()
                .map(|pair| (pair.0.id(), pair))
                .collect();
            for item in items {
                if let Some(change) = changes.get(item.id()) {
                    item.restore_fields(&change.0, &change.1);
                }
            }
        }
    }
}

fn restore_field<T: Clone + PartialEq>(live: &mut T, from: &T, to: &T) {
    // Do not touch fields this command did not change. If a later out-of-band
    // write also changed the same field, that newer value wins over history.
    if from != to && live == from {
        live.clone_from(to);
    }
}

macro_rules! history_item {
    ($item:ty, $($field:ident),+ $(,)?) => {
        impl HistoryItem for $item {
            fn id(&self) -> &str { &self.id }

            fn restore_fields(&mut self, from: &Self, to: &Self) {
                // No `..`: a new domain field must be classified here at compile
                // time, rather than silently regressing undo/data preservation.
                let Self { id: _, $($field: _,)+ } = to;
                $(restore_field(&mut self.$field, &from.$field, &to.$field);)+
            }
        }
    };
}

history_item!(
    MediaManifestEntry,
    name,
    kind,
    source,
    duration,
    generation_input,
    source_width,
    source_height,
    source_fps,
    has_audio,
    color,
    proxy,
    folder_id,
    cached_remote_url,
    cached_remote_url_expires_at,
);
history_item!(MediaFolder, name, parent_folder_id);
