//! Linear-time catalog admission for one already-probed import plan.
//! The caller owns the session lock, one rollback checkpoint and persistence.
//! Catalog imports (including their folders) never edit timeline/history state.
use std::collections::{HashMap, HashSet};

use super::*;
use crate::core::{CommittedMediaImport, PreparedMediaFolderRef, PreparedMediaImportOp};
use opentake_domain::MediaFolder;

struct ImportIndex {
    sources: HashMap<MediaSource, usize>,
    assets: HashMap<String, usize>,
    folders: HashSet<String>,
    planned_folders: HashMap<u64, String>,
}

impl ImportIndex {
    fn new(manifest: &MediaManifest) -> Self {
        let mut sources = HashMap::with_capacity(manifest.entries.len());
        let mut assets = HashMap::with_capacity(manifest.entries.len());
        for (index, entry) in manifest.entries.iter().enumerate() {
            // Match the existing single-import first-match policy, even when an
            // older project already contains duplicate sources or identities.
            sources.entry(entry.source.clone()).or_insert(index);
            assets.entry(entry.id.clone()).or_insert(index);
        }
        Self {
            sources,
            assets,
            folders: manifest
                .folders
                .iter()
                .map(|folder| folder.id.clone())
                .collect(),
            planned_folders: HashMap::new(),
        }
    }

    fn folder(&self, target: Option<PreparedMediaFolderRef>) -> Result<Option<String>> {
        let id = match target {
            None => return Ok(None),
            Some(PreparedMediaFolderRef::Existing(id)) => id,
            Some(PreparedMediaFolderRef::Planned(key)) => {
                self.planned_folders.get(&key).cloned().ok_or_else(|| {
                    CoreError::Media(format!("prepared folder key not found: {key}"))
                })?
            }
        };
        if !self.folders.contains(&id) {
            return Err(CoreError::Media(format!(
                "import folder does not exist: {id}"
            )));
        }
        Ok(Some(id))
    }

    fn insert(&mut self, manifest: &mut MediaManifest, entry: MediaManifestEntry) -> usize {
        if let Some(&index) = self.sources.get(&entry.source) {
            return index;
        }
        let index = manifest.entries.len();
        self.sources.insert(entry.source.clone(), index);
        self.assets.entry(entry.id.clone()).or_insert(index);
        manifest.entries.push(entry);
        index
    }
}

impl EditorSession {
    /// Called only by AppCore's checkpointed, identity-bound import transaction.
    /// A failure may leave partial catalog changes for that transaction to roll
    /// back; no public importer can bypass the durable commit/rollback boundary.
    pub(crate) fn import_prepared_media(
        &mut self,
        plan: Vec<PreparedMediaImportOp>,
        ids: &dyn IdGen,
    ) -> Result<Vec<CommittedMediaImport>> {
        self.ensure_mutable()?;
        let mut index = ImportIndex::new(&self.state.manifest);
        let mut imports = Vec::with_capacity(plan.len());
        for operation in plan {
            match operation {
                PreparedMediaImportOp::CreateFolder { key, name, parent } => {
                    if index.planned_folders.contains_key(&key) {
                        return Err(CoreError::Media(format!(
                            "duplicate prepared folder key: {key}"
                        )));
                    }
                    if name.is_empty() {
                        return Err(opentake_ops::EditError::Invalid(
                            "folder name is required".into(),
                        )
                        .into());
                    }
                    let parent_folder_id = index.folder(parent)?;
                    let id = ids.next_id();
                    if !index.folders.insert(id.clone()) {
                        return Err(CoreError::Media(format!(
                            "import folder id already exists: {id}"
                        )));
                    }
                    self.state.manifest.folders.push(MediaFolder {
                        id: id.clone(),
                        name,
                        parent_folder_id,
                    });
                    index.planned_folders.insert(key, id);
                }
                PreparedMediaImportOp::ImportFile {
                    path,
                    name,
                    probe,
                    folder,
                } => {
                    let folder_id = index.folder(folder)?;
                    let entry =
                        self.prepare_media_file_entry(&path, ids.next_id(), name, &probe)?;
                    let position = index.insert(&mut self.state.manifest, entry);
                    let target = &mut self.state.manifest.entries[position];
                    // An absent target preserves the existing entry's folder,
                    // just like an ordinary re-import of the same source.
                    if let Some(folder_id) = folder_id {
                        target.folder_id = Some(folder_id);
                    }
                    imports.push(CommittedMediaImport {
                        path,
                        entry: target.clone(),
                    });
                }
                PreparedMediaImportOp::ImportDerivedStem {
                    path,
                    name,
                    probe,
                    provenance,
                } => {
                    let source_index = index
                        .assets
                        .get(&provenance.source_asset_id)
                        .copied()
                        .ok_or_else(|| {
                            CoreError::Media(format!(
                                "stem source asset does not exist: {}",
                                provenance.source_asset_id
                            ))
                        })?;
                    let generation_input = stem_generation_input(
                        &self.state.manifest.entries[source_index],
                        provenance,
                        &probe,
                    )?;
                    let entry =
                        self.prepare_media_file_entry(&path, ids.next_id(), name, &probe)?;
                    let position = index.insert(&mut self.state.manifest, entry);
                    let target = &mut self.state.manifest.entries[position];
                    target.generation_input = Some(generation_input);
                    imports.push(CommittedMediaImport {
                        path,
                        entry: target.clone(),
                    });
                }
            }
        }
        Ok(imports)
    }
}

/// Validate all fallible provenance fields before a standalone or batch import
/// mutates the catalog; both entry points persist the same non-secret schema.
pub(super) fn stem_generation_input(
    source: &MediaManifestEntry,
    provenance: DerivedStemProvenance,
    probe: &ProbedMedia,
) -> Result<GenerationInput> {
    if !matches!(source.kind, ClipType::Audio | ClipType::Video)
        || !source.has_audio.unwrap_or(source.kind == ClipType::Audio)
    {
        return Err(CoreError::Media("stem source asset has no audio".into()));
    }
    if !valid_sha256(&provenance.source_sha256)
        || provenance
            .model_sha256
            .as_deref()
            .is_some_and(|digest| !valid_sha256(digest))
    {
        return Err(CoreError::Media(
            "stem provenance checksum is invalid".into(),
        ));
    }
    let (provider, model) = provenance
        .execution
        .split_once(':')
        .ok_or_else(|| CoreError::Media("stem execution must be '<provider>:<model>'".into()))?;
    if !safe_provider_prefix(provider) || model.trim().is_empty() {
        return Err(CoreError::Media(
            "stem execution provider or model is invalid".into(),
        ));
    }
    let output_index = match provenance.stem.as_str() {
        "vocals" => 0,
        "accompaniment" => 1,
        _ => {
            return Err(CoreError::Media(
                "stem kind must be vocals or accompaniment".into(),
            ))
        }
    };
    Ok(GenerationInput {
        prompt: format!("stem:{}", provenance.stem),
        model: model.to_string(),
        duration: probe.duration_secs.max(0.0).round() as i32,
        aspect_ratio: "audio".to_string(),
        quality: provenance
            .model_sha256
            .map(|digest| format!("model-sha256:{digest}")),
        reference_audio_urls: Some(vec![format!("sha256:{}", provenance.source_sha256)]),
        provider: Some(provider.to_string()),
        status: Some(GenerationJobStatus::Ready),
        progress: Some(1.0),
        output_index: Some(output_index),
        source_asset_id: Some(provenance.source_asset_id),
        ..GenerationInput::default()
    })
}
