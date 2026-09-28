//! The `.opentake` directory bundle: in-memory [`Project`] plus
//! [`Project::open`] / [`Project::save`].
//!
//! Port of `VideoProject`'s persistence (`Project/VideoProject.swift`), minus
//! the AppKit `NSDocument` / `FileWrapper` machinery. A bundle is a plain
//! directory; we read and write its files by path.
//!
//! Read semantics match upstream `read(from:)`:
//! - `project.json` is mandatory; absence is [`ProjectError::MissingTimeline`]
//!   (upstream throws `fileReadCorruptFile`).
//! - `media.json`, if present, is parsed strictly; a parse failure is an error
//!   (upstream throws `fileReadCorruptFile`).
//! - `generation-log.json`, if present, is parsed leniently; a parse failure
//!   yields an in-memory `None` recovery (upstream `try?`) plus a compatibility
//!   blocker, so the damaged bytes remain readable but cannot be overwritten.
//!
//! Write semantics follow the architecture note "assemble an in-memory
//! snapshot, then write atomically": each JSON component is written to a
//! sibling temp file and renamed into place, so a crash never leaves a
//! half-written `project.json`. `save` owns only the JSON components (and the
//! thumbnail when held); it never creates or deletes `media/`,
//! `chat-sessions/`, or `motion-documents/`, which their owning layers manage
//! out-of-band.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use opentake_domain::{MediaManifest, Timeline};
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::compatibility;
use crate::error::{ProjectError, Result};
use crate::gen_log::{GenerationLog, GenerationLogEntry};
use crate::layout;
use crate::{is_safe_project_asset_relative_path, ProjectRoot};

#[cfg(any(test, feature = "test-hooks"))]
thread_local! {
    static FAIL_FINAL_MANIFEST_WRITE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(feature = "test-hooks")]
#[doc(hidden)]
pub mod test_hooks {
    pub fn fail_next_final_manifest_write() {
        super::FAIL_FINAL_MANIFEST_WRITE.with(|fail| {
            assert!(!fail.replace(true), "previous test left the failure armed");
        });
    }

    /// Let `successes` more directory flushes on this thread succeed, then
    /// fail the next one after its rename has already committed.
    pub fn fail_directory_sync_after(successes: usize) {
        crate::project_root::sync_hooks::fail_directory_sync_after(successes);
    }

    /// Directory flushes attempted on this thread so far.
    pub fn directory_syncs() -> usize {
        crate::project_root::sync_hooks::directory_syncs()
    }
}

/// Persisted schema details this build cannot safely write back.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProjectCompatibility {
    blockers: Vec<String>,
}

fn validate_manifest_paths(manifest: &MediaManifest) -> Result<()> {
    for entry in &manifest.entries {
        if let opentake_domain::MediaSource::Project { relative_path } = &entry.source {
            if !is_safe_project_asset_relative_path(relative_path) {
                return Err(ProjectError::InvalidMediaManifest {
                    file: layout::MANIFEST_FILE,
                    reason: format!(
                        "project source for asset '{}' is not a safe bundle-relative path",
                        entry.id
                    ),
                });
            }
        }
        if let Some(proxy) = &entry.proxy {
            if !is_safe_project_asset_relative_path(&proxy.relative_path) {
                return Err(ProjectError::InvalidMediaManifest {
                    file: layout::MANIFEST_FILE,
                    reason: format!(
                        "proxy for asset '{}' is not a safe bundle-relative path",
                        entry.id
                    ),
                });
            }
        }
    }
    Ok(())
}

impl ProjectCompatibility {
    /// Whether saving would discard data this build does not understand.
    pub fn is_read_only(&self) -> bool {
        !self.blockers.is_empty()
    }

    /// Sorted, file-qualified reasons the project is compatibility read-only.
    pub fn blockers(&self) -> &[String] {
        &self.blockers
    }

    fn extend(&mut self, blockers: impl IntoIterator<Item = String>) {
        self.blockers.extend(blockers);
        self.blockers.sort();
        self.blockers.dedup();
    }

    /// Refuse a write that would discard unknown persisted data.
    pub fn ensure_writable(&self) -> Result<()> {
        if self.is_read_only() {
            return Err(ProjectError::CompatibilityReadOnly {
                blockers: self.blockers.clone(),
            });
        }
        Ok(())
    }
}

/// Requested mutation for the optional project cover in an explicit save.
///
/// `Preserve` is the compatibility/default behavior for ordinary saves,
/// `Replace` commits newly captured JPEG bytes, and `Remove` represents the
/// authoritative result that the project has no visible cover content.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ThumbnailUpdate {
    #[default]
    Preserve,
    Replace(Vec<u8>),
    Remove,
}

/// An opened `.opentake` project: the bundle path plus its decoded components.
///
/// Media files referenced by `manifest` live under the bundle's `media/`
/// directory (`.project` sources) or at absolute paths (`.external`); they are
/// not loaded into this struct. Chat sessions and the thumbnail are likewise
/// left on disk, except for an optional in-memory `thumbnail` that `save` will
/// persist when set.
#[derive(Clone, Debug)]
pub struct Project {
    /// Absolute path to the bundle directory (`…/Name.opentake`).
    pub bundle_path: PathBuf,
    /// The timeline (`project.json`).
    pub timeline: Timeline,
    /// The media manifest (`media.json`). Defaults to empty when the file was
    /// absent.
    pub manifest: MediaManifest,
    /// The generation log (`generation-log.json`). `None` when the file was
    /// absent or failed to parse; the latter also makes compatibility read-only.
    pub generation_log: Option<GenerationLog>,
    /// Optional cover bytes to write on the next save. `None` preserves an
    /// existing on-disk cover, matching the original public API.
    pub thumbnail: Option<Vec<u8>>,
    compatibility: ProjectCompatibility,
}

impl Project {
    /// Create a fresh, empty project rooted at `bundle_path` (not yet written).
    pub fn new(bundle_path: impl Into<PathBuf>) -> Self {
        Self::new_with_compatibility(bundle_path, ProjectCompatibility::default())
    }

    /// Create a project while preserving compatibility state from an opened bundle.
    pub fn new_with_compatibility(
        bundle_path: impl Into<PathBuf>,
        compatibility: ProjectCompatibility,
    ) -> Self {
        Project {
            bundle_path: bundle_path.into(),
            timeline: Timeline::new(),
            manifest: MediaManifest::new(),
            generation_log: None,
            thumbnail: None,
            compatibility,
        }
    }

    /// Compatibility state detected while opening the persisted components.
    pub fn compatibility(&self) -> &ProjectCompatibility {
        &self.compatibility
    }

    /// Reconstruct the legacy generation audit rows carried only by manifest
    /// entries saved before `generation-log.json` existed.
    ///
    /// Canonically identical [`opentake_domain::GenerationInput`] snapshots
    /// represent one generation even when it produced multiple assets. The full
    /// SHA-256 provenance digest supplies a fixed-size, deterministic synthetic
    /// row id. Canonical keys also impose a total row order, so manifest ordering
    /// cannot perturb saved bytes.
    /// Legacy manifests contain no trustworthy billed-cost field, so seeded rows
    /// keep `cost_credits = None` instead of applying a mutable pricing catalog
    /// retroactively.
    pub fn seed_generation_log_from_assets(&self) -> Result<GenerationLog> {
        let mut seeds = BTreeMap::<Vec<u8>, (String, Option<f64>)>::new();

        for entry in &self.manifest.entries {
            let Some(provenance) = &entry.generation_input else {
                continue;
            };
            let canonical_key = serde_json::to_vec(provenance)
                .map_err(|error| ProjectError::json(layout::MANIFEST_FILE, error))?;
            seeds
                .entry(canonical_key)
                .or_insert_with(|| (provenance.model.clone(), provenance.created_at));
        }

        Ok(GenerationLog {
            version: 1,
            entries: seeds
                .into_iter()
                .map(|(canonical_key, (model, created_at))| {
                    GenerationLogEntry::new(
                        format!("legacy-generation:{}", sha256_hex(&canonical_key)),
                        model,
                        None,
                        created_at,
                    )
                })
                .collect(),
        })
    }

    /// Open the `.opentake` bundle at `path`.
    ///
    /// Returns [`ProjectError::NotABundle`] if `path` is not a directory,
    /// [`ProjectError::MissingTimeline`] if `project.json` is absent, and
    /// [`ProjectError::Json`] if `project.json` or `media.json` fails to parse.
    /// A malformed `generation-log.json` opens as a compatibility read-only
    /// recovery with no decoded log.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let root = ProjectRoot::open(path)?;
        Self::open_from_root(&root)
    }

    /// Decode every project component from one retained root capability.
    ///
    /// Persisted compatibility is applied by each component's deserializer:
    /// missing optional fields receive their legacy defaults, while explicit
    /// schema versions are preserved. Saving the returned project therefore
    /// writes the decoded state without silently promoting legacy versions.
    pub fn open_from_root(root: &ProjectRoot) -> Result<Self> {
        Self::open_from_root_with_hook(root, |_| {})
    }

    fn open_from_root_with_hook(
        root: &ProjectRoot,
        mut after_component: impl FnMut(&str),
    ) -> Result<Self> {
        let bundle = root.path();
        let timeline_bytes = root.read_optional(layout::TIMELINE_FILE)?.ok_or_else(|| {
            ProjectError::MissingTimeline {
                file: layout::TIMELINE_FILE,
                bundle: bundle.to_path_buf(),
            }
        })?;
        let (mut timeline, timeline_blockers, timeline_document) =
            decode_component::<Timeline>(&timeline_bytes, layout::TIMELINE_FILE)?;
        compatibility::repair_timeline_ids(&mut timeline, &timeline_document);
        timeline
            .validate_nested_sequences()
            .map_err(|reason| ProjectError::InvalidTimeline {
                file: layout::TIMELINE_FILE,
                reason,
            })?;
        after_component(layout::TIMELINE_FILE);
        let mut compatibility = ProjectCompatibility::default();
        compatibility.extend(timeline_blockers);

        // media.json: strict when present, empty default when absent.
        let manifest = if let Some(bytes) = root.read_optional(layout::MANIFEST_FILE)? {
            let (manifest, blockers, _) =
                decode_component::<MediaManifest>(&bytes, layout::MANIFEST_FILE)?;
            compatibility.extend(blockers);
            validate_manifest_paths(&manifest)?;
            manifest
        } else {
            MediaManifest::new()
        };
        after_component(layout::MANIFEST_FILE);

        // generation-log.json: lenient read recovery — a parse error degrades
        // to None but records a blocker so no save can overwrite the bytes.
        let generation_log = match root.read_optional(layout::GENERATION_LOG_FILE) {
            Ok(Some(bytes)) => {
                match decode_component::<GenerationLog>(&bytes, layout::GENERATION_LOG_FILE) {
                    Ok((log, blockers, _)) => {
                        compatibility.extend(blockers);
                        Some(log)
                    }
                    Err(_) => {
                        compatibility.extend([format!(
                            "{}:invalid-or-unreadable",
                            layout::GENERATION_LOG_FILE
                        )]);
                        None
                    }
                }
            }
            Ok(None) => None,
            Err(_) => {
                compatibility.extend([format!(
                    "{}:invalid-or-unreadable",
                    layout::GENERATION_LOG_FILE
                )]);
                None
            }
        };
        after_component(layout::GENERATION_LOG_FILE);

        Ok(Project {
            bundle_path: bundle.to_path_buf(),
            timeline,
            manifest,
            generation_log,
            thumbnail: None,
            compatibility,
        })
    }

    /// Write this project's JSON components into [`Self::bundle_path`].
    ///
    /// Creates the bundle directory if needed. Always (re)writes `project.json`
    /// and `media.json`; writes `generation-log.json` when a log is held and
    /// `thumbnail.jpg` when [`Self::thumbnail`] is set. Each file is written
    /// atomically (temp file + rename), ordering the timeline and manifest by
    /// asset additions/removals so a published timeline never references a
    /// missing manifest entry. Existing `media/` and `chat-sessions/`
    /// directories are left untouched.
    pub fn save(&self) -> Result<()> {
        let encoded = EncodedProject::prepare(self)?;
        if let Some(root) = ProjectRoot::open_optional(&self.bundle_path)? {
            encoded.write_to(&root)
        } else {
            let publisher = ProjectRoot::begin_replace(&self.bundle_path)?;
            encoded.write_to(publisher.stage()).map_err(unpublished)?;
            publisher.publish().map(|_| ())
        }
    }

    /// Persist only `media.json` through one atomic replacement.
    ///
    /// Media-library workflows mutate no timeline, generation log, thumbnail,
    /// or bundled media bytes. Restricting their durable commit to this one
    /// component prevents a later unrelated component failure from turning an
    /// error result into a partially saved manifest.
    pub fn save_manifest(&self) -> Result<()> {
        self.compatibility.ensure_writable()?;
        let manifest = encode_component(layout::MANIFEST_FILE, &self.manifest)?;
        let root = ProjectRoot::create(&self.bundle_path)?;
        root.write_atomic(layout::MANIFEST_FILE, &manifest)
    }

    /// Persist only `media.json` through a retained bundle root.
    pub fn save_manifest_to_root(&self, root: &ProjectRoot) -> Result<()> {
        self.compatibility.ensure_writable()?;
        let manifest = encode_component(layout::MANIFEST_FILE, &self.manifest)?;
        root.write_atomic(layout::MANIFEST_FILE, &manifest)
    }

    /// Persist `generation-log.json` (when held) and then `media.json` in
    /// place through a retained bundle root.
    ///
    /// Generation state lives in exactly these two components. Each is replaced
    /// atomically and the root directory keeps its identity, so no other
    /// component or `media/` byte is copied. `media.json` is written last and
    /// is the commit point: a failure or crash between the two writes leaves
    /// the previous manifest with at most extra trailing log rows describing a
    /// transition the manifest does not yet show. Job state is always taken
    /// from the manifest and the log is an append-only audit trail, so the
    /// worst outcome is that a resumed job records that transition twice.
    pub fn save_manifest_and_generation_log_to_root(&self, root: &ProjectRoot) -> Result<()> {
        self.compatibility.ensure_writable()?;
        let manifest = encode_component(layout::MANIFEST_FILE, &self.manifest)?;
        let log = self
            .generation_log
            .as_ref()
            .map(|log| encode_component(layout::GENERATION_LOG_FILE, log))
            .transpose()?;
        let mut writes = ComponentWrites::default();
        if let Some(log) = &log {
            writes.write(root, layout::GENERATION_LOG_FILE, log)?;
        }
        writes.write(root, layout::MANIFEST_FILE, &manifest)?;
        writes.finish()
    }

    /// Like [`Self::save`] but targets an explicit `bundle` directory (used by
    /// the archiver to stage a self-contained copy). Does not mutate `self`.
    pub fn save_to(&self, bundle: impl AsRef<Path>) -> Result<()> {
        let encoded = EncodedProject::prepare(self)?;
        let publisher = ProjectRoot::begin_replace(bundle.as_ref())?;
        encoded.write_to(publisher.stage()).map_err(unpublished)?;
        publisher.publish().map(|_| ())
    }

    /// Persist this snapshot exclusively through `root` authority.
    pub fn save_to_root(&self, root: &ProjectRoot) -> Result<()> {
        EncodedProject::prepare(self)?.write_to(root)
    }

    /// Persist this snapshot with an explicit optional-cover mutation.
    ///
    /// This additive API keeps [`Self::thumbnail`] source-compatible while
    /// allowing authoritative callers to distinguish preserve from removal.
    pub fn save_to_root_with_thumbnail_update(
        &self,
        root: &ProjectRoot,
        thumbnail: ThumbnailUpdate,
    ) -> Result<()> {
        EncodedProject::prepare_with_thumbnail_update(self, thumbnail)?.write_to(root)
    }

    /// Publish a complete fresh sibling bundle and return the exact root that
    /// became visible. Sessions adopt this retained authority only after the
    /// directory publication commit succeeds.
    pub fn publish_complete_to(
        &self,
        bundle: impl AsRef<Path>,
        media_source: Option<&ProjectRoot>,
    ) -> Result<ProjectRoot> {
        self.publish_complete_to_with_thumbnail_update(
            bundle,
            media_source,
            self.thumbnail
                .clone()
                .map_or(ThumbnailUpdate::Preserve, ThumbnailUpdate::Replace),
        )
    }

    /// Publish a complete fresh sibling with an explicit optional-cover
    /// mutation while retaining all other bundle components.
    pub fn publish_complete_to_with_thumbnail_update(
        &self,
        bundle: impl AsRef<Path>,
        media_source: Option<&ProjectRoot>,
        thumbnail: ThumbnailUpdate,
    ) -> Result<ProjectRoot> {
        let preserve_thumbnail = matches!(thumbnail, ThumbnailUpdate::Preserve);
        let encoded = EncodedProject::prepare_with_thumbnail_update(self, thumbnail)?;
        let publisher = ProjectRoot::begin_replace(bundle.as_ref())?;
        encoded.write_to(publisher.stage()).map_err(unpublished)?;
        if let Some(source) = media_source {
            source.copy_media_to(publisher.stage())?;
            source.copy_chat_sessions_to(publisher.stage())?;
            source.copy_motion_documents_to(publisher.stage())?;
            if preserve_thumbnail {
                source
                    .copy_thumbnail_to(publisher.stage())
                    .map_err(unpublished)?;
            }
        }
        publisher.publish()
    }
}

/// A write into an unpublished stage commits nothing: only the stage's
/// publication rename does. An unconfirmed flush there is an ordinary failure
/// that aborts publication and discards the stage.
fn unpublished(error: ProjectError) -> ProjectError {
    match error {
        ProjectError::DurabilityUnconfirmed { path, source } => ProjectError::io(path, source),
        error => error,
    }
}

/// The component replacements of one logical save. A replacement whose
/// directory flush failed has still committed, so the save carries on to its
/// commit point and reports the unconfirmed flush only after the remaining
/// components are written; any other failure stops the save immediately.
#[derive(Default)]
struct ComponentWrites {
    unconfirmed: Option<ProjectError>,
}

impl ComponentWrites {
    fn write(&mut self, root: &ProjectRoot, name: &str, bytes: &[u8]) -> Result<()> {
        match root.write_atomic(name, bytes) {
            Err(error @ ProjectError::DurabilityUnconfirmed { .. }) => {
                self.unconfirmed.get_or_insert(error);
                Ok(())
            }
            result => result,
        }
    }

    fn finish(self) -> Result<()> {
        self.unconfirmed.map_or(Ok(()), Err)
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

struct EncodedProject {
    timeline: Vec<u8>,
    manifest: Vec<u8>,
    generation_log: Option<Vec<u8>>,
    thumbnail: ThumbnailUpdate,
}

impl EncodedProject {
    /// Produce the exact byte snapshot before any destination path is created.
    fn prepare(project: &Project) -> Result<Self> {
        let thumbnail = project
            .thumbnail
            .clone()
            .map_or(ThumbnailUpdate::Preserve, ThumbnailUpdate::Replace);
        Self::prepare_with_thumbnail_update(project, thumbnail)
    }

    fn prepare_with_thumbnail_update(
        project: &Project,
        thumbnail: ThumbnailUpdate,
    ) -> Result<Self> {
        project.compatibility.ensure_writable()?;
        project
            .timeline
            .validate_nested_sequences()
            .map_err(|reason| ProjectError::InvalidTimeline {
                file: layout::TIMELINE_FILE,
                reason,
            })?;
        Ok(Self {
            timeline: encode_component(layout::TIMELINE_FILE, &project.timeline)?,
            manifest: encode_component(layout::MANIFEST_FILE, &project.manifest)?,
            generation_log: project
                .generation_log
                .as_ref()
                .map(|log| encode_component(layout::GENERATION_LOG_FILE, log))
                .transpose()?,
            thumbnail,
        })
    }

    fn write_to(&self, root: &ProjectRoot) -> Result<()> {
        let current_manifest = match root.read_optional(layout::MANIFEST_FILE)? {
            Some(bytes) => decode_component::<MediaManifest>(&bytes, layout::MANIFEST_FILE)?.0,
            None => MediaManifest::default(),
        };
        let next_manifest =
            decode_component::<MediaManifest>(&self.manifest, layout::MANIFEST_FILE)?.0;
        let current_ids: HashSet<&str> = current_manifest
            .entries
            .iter()
            .map(|entry| entry.id.as_str())
            .collect();
        let next_ids: HashSet<&str> = next_manifest
            .entries
            .iter()
            .map(|entry| entry.id.as_str())
            .collect();
        let adds_assets = next_ids.iter().any(|id| !current_ids.contains(id));
        let removes_assets = current_ids.iter().any(|id| !next_ids.contains(id));
        let mut writes = ComponentWrites::default();

        if removes_assets && !adds_assets {
            self.write_non_manifest_components(root, &mut writes)?;
            writes.write(root, layout::TIMELINE_FILE, &self.timeline)?;
            write_final_manifest(root, &mut writes, &self.manifest)
                .map_err(ProjectError::partial_commit)?;
            return writes.finish();
        }

        if adds_assets && removes_assets {
            let mut transition_manifest = next_manifest.clone();
            for entry in &current_manifest.entries {
                if !next_ids.contains(entry.id.as_str()) {
                    transition_manifest.entries.push(entry.clone());
                }
            }
            let transition_folder_ids: HashSet<String> = transition_manifest
                .folders
                .iter()
                .map(|folder| folder.id.clone())
                .collect();
            for folder in &current_manifest.folders {
                if !transition_folder_ids.contains(&folder.id) {
                    transition_manifest.folders.push(folder.clone());
                }
            }
            validate_manifest_paths(&transition_manifest)?;
            let transition_bytes = encode_component(layout::MANIFEST_FILE, &transition_manifest)?;
            writes.write(root, layout::MANIFEST_FILE, &transition_bytes)?;
            self.write_non_manifest_components(root, &mut writes)?;
            writes.write(root, layout::TIMELINE_FILE, &self.timeline)?;
            write_final_manifest(root, &mut writes, &self.manifest)
                .map_err(ProjectError::partial_commit)?;
            return writes.finish();
        }

        writes.write(root, layout::MANIFEST_FILE, &self.manifest)?;
        self.write_non_manifest_components(root, &mut writes)?;
        // New clip references become visible only after their manifest entries.
        // A failed timeline replacement can leave extra, unreferenced entries.
        writes.write(root, layout::TIMELINE_FILE, &self.timeline)?;
        writes.finish()
    }

    fn write_non_manifest_components(
        &self,
        root: &ProjectRoot,
        writes: &mut ComponentWrites,
    ) -> Result<()> {
        if let Some(log) = &self.generation_log {
            writes.write(root, layout::GENERATION_LOG_FILE, log)?;
        }
        match &self.thumbnail {
            ThumbnailUpdate::Preserve => {}
            ThumbnailUpdate::Replace(thumbnail) => {
                writes.write(root, layout::THUMBNAIL_FILE, thumbnail)?;
            }
            ThumbnailUpdate::Remove => root.remove_optional_component(layout::THUMBNAIL_FILE)?,
        }
        Ok(())
    }
}

fn write_final_manifest(
    root: &ProjectRoot,
    writes: &mut ComponentWrites,
    manifest: &[u8],
) -> Result<()> {
    #[cfg(any(test, feature = "test-hooks"))]
    if FAIL_FINAL_MANIFEST_WRITE.with(|fail| fail.replace(false)) {
        return Err(ProjectError::io(
            root.path().join(layout::MANIFEST_FILE),
            std::io::Error::other("injected final manifest failure"),
        ));
    }
    writes.write(root, layout::MANIFEST_FILE, manifest)
}

fn encode_component<T: Serialize>(file_name: &str, value: &T) -> Result<Vec<u8>> {
    serde_json::to_vec_pretty(value).map_err(|error| ProjectError::json(file_name, error))
}

fn decode_component<T: DeserializeOwned>(
    bytes: &[u8],
    file: &str,
) -> Result<(T, Vec<String>, Value)> {
    let document: Value =
        serde_json::from_slice(bytes).map_err(|error| ProjectError::json(file, error))?;

    // The normal path performs one formal decode and no Track.clips probes or
    // document clone. Only a failed timeline decode enters the narrow upstream
    // Track.clips fallback.
    let initial = deserialize_with_ignored(bytes, file, &document);
    let (value, mut ignored, failed_tracks) = match initial {
        Ok((value, ignored)) => (value, ignored, Vec::new()),
        Err(initial_error) if file == layout::TIMELINE_FILE => {
            let Some(fallback) = compatibility::prepare_timeline_fallback(&document) else {
                return Err(ProjectError::json(file, initial_error));
            };
            let normalized = serde_json::to_vec(&fallback.normalized)
                .map_err(|error| ProjectError::json(file, error))?;
            let (value, ignored) = deserialize_with_ignored(&normalized, file, &document)
                .map_err(|error| ProjectError::json(file, error))?;
            (value, ignored, fallback.failed_tracks)
        }
        Err(error) => return Err(ProjectError::json(file, error)),
    };

    if file == layout::TIMELINE_FILE {
        compatibility::scan_timeline(&document, file, &failed_tracks, &mut ignored);
    }
    ignored.sort();
    ignored.dedup();
    Ok((value, ignored, document))
}

fn deserialize_with_ignored<T: DeserializeOwned>(
    bytes: &[u8],
    file: &str,
    document: &Value,
) -> std::result::Result<(T, Vec<String>), serde_json::Error> {
    let mut decoder = serde_json::Deserializer::from_slice(bytes);
    let mut ignored = Vec::new();
    let value = serde_ignored::deserialize(&mut decoder, |path| {
        ignored.push(format!(
            "{file}:{}",
            compatibility::canonical_ignored_path(&path, document)
        ));
    })?;
    decoder.end()?;
    Ok((value, ignored))
}

/// Copy a source bundle's `media/` directory into `dest_bundle`, recursively,
/// preserving the relative layout — the port of upstream `mediaDirWrapper`
/// (`Project/VideoProject.swift:112-117`), which folds the whole `media/`
/// directory into the saved package on every save/save-as. Save-as builds the
/// new bundle at a fresh path; without this, project-internal media
/// ([`MediaSource::Project`](opentake_domain::MediaSource) relative paths — AI
/// output, pasted, captured stills) is left behind and every reference silently
/// dangles.
///
/// Contract:
/// - **Missing source `media/`** → no-op `Ok(())` (upstream returns `nil` from
///   `mediaDirWrapper` when the dir doesn't exist; nothing to carry).
/// - **Same-path save** (source and dest bundle are the same directory) → no-op,
///   so autosave never copies `media/` onto itself.
/// - **Existing destination `media/`** → fail without modifying it. Complete
///   Save As replacement is owned by [`Project::publish_complete_to`], which
///   publishes a fresh whole-bundle sibling through the backup/recovery state
///   machine rather than deleting a live media tree.
pub fn copy_media_dir(source_bundle: &Path, dest_bundle: &Path) -> Result<()> {
    if source_bundle == dest_bundle {
        return Ok(());
    }
    let source = ProjectRoot::open(source_bundle)?;
    if !source.has_media_tree()? {
        return Ok(());
    }
    let destination = ProjectRoot::create(dest_bundle)?;
    source.copy_media_to(&destination)
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentake_domain::{Clip, ClipType, MediaManifestEntry, MediaSource, Track};
    use std::fs;

    /// A per-call-unique scratch dir under the system temp dir, removed on drop.
    struct TmpDir(PathBuf);
    impl TmpDir {
        fn new(tag: &str) -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static N: AtomicU64 = AtomicU64::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let p = std::env::temp_dir()
                .join(format!("opentake-bundle-{tag}-{}-{n}", std::process::id()));
            let _ = fs::remove_dir_all(&p);
            fs::create_dir_all(&p).unwrap();
            TmpDir(p)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TmpDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn failed_manifest_cleanup_keeps_the_committed_timeline_and_reports_partial_commit() {
        let tmp = TmpDir::new("partial-delete-commit");
        let bundle = tmp.path().join("Delete.opentake");
        let mut project = Project::new(&bundle);
        project
            .timeline
            .tracks
            .push(Track::new("V1", ClipType::Video));
        project.timeline.tracks[0]
            .clips
            .push(Clip::new("clip-1", "asset-1", 0, 30));
        project.manifest.entries.push(MediaManifestEntry {
            id: "asset-1".into(),
            name: "source.mp4".into(),
            kind: ClipType::Video,
            source: MediaSource::Project {
                relative_path: "media/source.mp4".into(),
            },
            duration: 1.0,
            generation_input: None,
            source_width: None,
            source_height: None,
            source_fps: None,
            has_audio: Some(false),
            color: None,
            proxy: None,
            folder_id: None,
            cached_remote_url: None,
            cached_remote_url_expires_at: None,
        });
        project.save().unwrap();

        let mut deleted = Project::open(&bundle).unwrap();
        deleted.timeline.tracks[0].clips.clear();
        deleted.manifest.entries.clear();
        FAIL_FINAL_MANIFEST_WRITE.with(|fail| {
            assert!(!fail.replace(true), "previous test left the failure armed");
        });

        let error = deleted.save().expect_err("manifest cleanup must fail");
        assert!(error.is_partial_commit());
        assert!(error.to_string().contains("timeline was committed"));

        let reopened = Project::open(&bundle).unwrap();
        assert!(reopened.timeline.tracks[0].clips.is_empty());
        assert!(reopened
            .manifest
            .entries
            .iter()
            .any(|entry| entry.id == "asset-1"));
    }

    fn video_entry(id: &str, relative_path: &str) -> MediaManifestEntry {
        MediaManifestEntry {
            id: id.into(),
            name: format!("{id}.mp4"),
            kind: ClipType::Video,
            source: MediaSource::Project {
                relative_path: relative_path.into(),
            },
            duration: 1.0,
            generation_input: None,
            source_width: None,
            source_height: None,
            source_fps: None,
            has_audio: Some(false),
            color: None,
            proxy: None,
            folder_id: None,
            cached_remote_url: None,
            cached_remote_url_expires_at: None,
        }
    }

    #[test]
    fn unflushed_component_does_not_stop_the_save_before_its_commit_point() {
        let tmp = TmpDir::new("deferred-durability");
        let bundle = tmp.path().join("Deferred.opentake");
        let mut project = Project::new(&bundle);
        project.save().unwrap();
        let root = ProjectRoot::open(&bundle).unwrap();
        project.timeline.fps = 48;
        project
            .manifest
            .entries
            .push(video_entry("asset-1", "media/asset-1.mp4"));

        // The manifest (written before the timeline commit) is not flushed.
        crate::project_root::sync_hooks::fail_directory_sync_after(0);
        let error = project
            .save_to_root(&root)
            .expect_err("the unconfirmed manifest flush must be reported");

        assert!(
            matches!(error, ProjectError::DurabilityUnconfirmed { .. }),
            "{error:?}"
        );
        assert!(error.is_partial_commit());
        let reopened = Project::open(&bundle).unwrap();
        assert_eq!(
            reopened.timeline.fps, 48,
            "the save reached its commit point"
        );
        assert_eq!(reopened.manifest.entries.len(), 1);
    }

    #[test]
    fn unflushed_stage_component_aborts_publication_without_a_commit() {
        let tmp = TmpDir::new("stage-durability");
        let source = tmp.path().join("Source.opentake");
        let destination = tmp.path().join("Copy.opentake");
        let mut project = Project::new(&source);
        project
            .manifest
            .entries
            .push(video_entry("asset-1", "media/asset-1.mp4"));

        // Journal and stage marker flushes succeed; the staged manifest does not.
        crate::project_root::sync_hooks::fail_directory_sync_after(2);
        let error = project
            .save_to(&destination)
            .expect_err("an unflushed stage must not be published");

        assert!(!error.is_partial_commit(), "{error:?}");
        assert!(!destination.exists());
        // Only the persistent transaction lock remains: no stage or journal.
        let artifacts = fs::read_dir(tmp.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.starts_with(".Copy.opentake.opentake-"))
            .collect::<Vec<_>>();
        assert_eq!(artifacts, [".Copy.opentake.opentake-lock"]);
    }

    fn tree_receipt(root: &Path) -> Vec<(PathBuf, Option<Vec<u8>>)> {
        fn visit(base: &Path, path: &Path, receipt: &mut Vec<(PathBuf, Option<Vec<u8>>)>) {
            let mut entries = fs::read_dir(path)
                .unwrap()
                .map(std::result::Result::unwrap)
                .collect::<Vec<_>>();
            entries.sort_by_key(std::fs::DirEntry::file_name);
            for entry in entries {
                let path = entry.path();
                let relative = path.strip_prefix(base).unwrap().to_path_buf();
                if entry.file_type().unwrap().is_dir() {
                    receipt.push((relative, None));
                    visit(base, &path, receipt);
                } else {
                    receipt.push((relative, Some(fs::read(path).unwrap())));
                }
            }
        }

        let mut receipt = Vec::new();
        visit(root, root, &mut receipt);
        receipt
    }

    #[cfg(unix)]
    #[test]
    fn retained_root_prevents_component_mixing_during_ambient_aba() {
        let tmp = TmpDir::new("open-aba");
        let projects = tmp.path().join("projects");
        let retained = tmp.path().join("projects-retained");
        let replacement = tmp.path().join("projects-replacement");
        let bundle = projects.join("A.opentake");
        let replacement_bundle = replacement.join("A.opentake");
        let mut original = Project::new(&bundle);
        original.timeline.fps = 24;
        original.manifest.favorites.push("from-original".into());
        original.save().unwrap();
        let mut other = Project::new(&replacement_bundle);
        other.timeline.fps = 60;
        other.manifest.favorites.push("from-replacement".into());
        other.save().unwrap();
        let root = ProjectRoot::open(&bundle).unwrap();

        let opened = Project::open_from_root_with_hook(&root, |component| {
            if component == layout::TIMELINE_FILE {
                fs::rename(&projects, &retained).unwrap();
                fs::rename(&replacement, &projects).unwrap();
            } else if component == layout::MANIFEST_FILE {
                fs::rename(&projects, &replacement).unwrap();
                fs::rename(&retained, &projects).unwrap();
            }
        })
        .unwrap();

        assert_eq!(opened.timeline.fps, 24);
        assert_eq!(opened.manifest.favorites, ["from-original"]);
    }

    #[cfg(unix)]
    #[test]
    fn final_bundle_symlink_is_rejected() {
        use std::os::unix::fs::symlink;

        let tmp = TmpDir::new("root-symlink");
        let real = tmp.path().join("Real.opentake");
        Project::new(&real).save().unwrap();
        let link = tmp.path().join("Link.opentake");
        symlink(&real, &link).unwrap();

        assert!(matches!(
            Project::open(link),
            Err(ProjectError::NotABundle(_))
        ));
    }

    #[test]
    fn project_open_rejects_unsafe_project_media_and_proxy_paths() {
        for (index, unsafe_path) in [
            "../private.mov",
            "media/../../private.mov",
            "/private.mov",
            r"C:\private.mov",
            "C:private.mov",
        ]
        .into_iter()
        .enumerate()
        {
            let tmp = TmpDir::new(&format!("unsafe-media-path-{index}"));
            let bundle = tmp.path().join("Unsafe.opentake");
            Project::new(&bundle).save().unwrap();
            let manifest = serde_json::json!({
                "version": 2,
                "entries": [{
                    "id": "asset-1",
                    "name": "clip.mov",
                    "type": "video",
                    "source": { "project": { "relativePath": unsafe_path } },
                    "duration": 1.0
                }],
                "folders": []
            });
            fs::write(
                bundle.join(layout::MANIFEST_FILE),
                serde_json::to_vec(&manifest).unwrap(),
            )
            .unwrap();

            assert!(Project::open(&bundle).is_err());

            let mut proxy_manifest = manifest;
            proxy_manifest["entries"][0]["source"] = serde_json::json!({
                "project": { "relativePath": "media/valid.mov" }
            });
            proxy_manifest["entries"][0]["proxy"] = serde_json::json!({
                "relativePath": unsafe_path,
                "sourceSha256": "00",
                "width": 320,
                "height": 180
            });
            fs::write(
                bundle.join(layout::MANIFEST_FILE),
                serde_json::to_vec(&proxy_manifest).unwrap(),
            )
            .unwrap();
            assert!(Project::open(&bundle).is_err());
        }
    }

    #[test]
    fn project_open_accepts_nested_project_media_path() {
        let tmp = TmpDir::new("safe-media-path");
        let bundle = tmp.path().join("Safe.opentake");
        Project::new(&bundle).save().unwrap();
        let manifest = serde_json::json!({
            "version": 2,
            "entries": [{
                "id": "asset-1",
                "name": "clip.mov",
                "type": "video",
                "source": { "project": { "relativePath": "media/nested/clip.mov" } },
                "duration": 1.0
            }],
            "folders": []
        });
        fs::write(
            bundle.join(layout::MANIFEST_FILE),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();

        assert_eq!(
            Project::open(&bundle).unwrap().manifest.entries[0].id,
            "asset-1"
        );
    }

    #[cfg(unix)]
    #[test]
    fn retained_root_save_never_writes_an_ambient_replacement() {
        let tmp = TmpDir::new("save-aba");
        let projects = tmp.path().join("projects");
        let retained = tmp.path().join("projects-retained");
        let bundle = projects.join("A.opentake");
        let mut original = Project::new(&bundle);
        original.timeline.fps = 24;
        original.save().unwrap();
        let root = ProjectRoot::open(&bundle).unwrap();
        fs::rename(&projects, &retained).unwrap();
        let mut replacement = Project::new(&bundle);
        replacement.timeline.fps = 60;
        replacement.save().unwrap();

        original.timeline.fps = 48;
        original.save_to_root(&root).unwrap();

        assert_eq!(Project::open(&bundle).unwrap().timeline.fps, 60);
        assert_eq!(
            Project::open(retained.join("A.opentake"))
                .unwrap()
                .timeline
                .fps,
            48
        );
    }

    #[test]
    fn complete_publish_replaces_an_existing_bundle_with_fresh_media() {
        let tmp = TmpDir::new("complete-replace");
        let source = tmp.path().join("Source.opentake");
        let target = tmp.path().join("Target.opentake");
        let mut project = Project::new(&source);
        project.timeline.fps = 48;
        project.save().unwrap();
        fs::create_dir_all(source.join("media/nested")).unwrap();
        fs::write(source.join("media/nested/clip.bin"), b"fresh media").unwrap();
        let source_root = ProjectRoot::open(&source).unwrap();
        fs::create_dir_all(target.join("media")).unwrap();
        fs::write(target.join("project.json"), b"old timeline").unwrap();
        fs::write(target.join("media/stale.bin"), b"stale media").unwrap();
        fs::write(target.join("stale.txt"), b"stale component").unwrap();

        let published = project
            .publish_complete_to(&target, Some(&source_root))
            .expect("complete Save As publication");

        assert_eq!(
            Project::open_from_root(&published).unwrap().timeline.fps,
            48
        );
        assert_eq!(
            fs::read(target.join("media/nested/clip.bin")).unwrap(),
            b"fresh media"
        );
        assert!(!target.join("media/stale.bin").exists());
        assert!(!target.join("stale.txt").exists());
    }

    #[test]
    fn complete_publish_carries_project_chat_sessions_across_save_as() {
        let tmp = TmpDir::new("complete-chat-sessions");
        let source = tmp.path().join("Source.opentake");
        let target = tmp.path().join("Target.opentake");
        let project = Project::new(&source);
        project.save().unwrap();
        fs::create_dir_all(source.join("chat-sessions")).unwrap();
        fs::write(
            source.join("chat-sessions/chat-1.json"),
            br#"{"id":"chat-1","messages":[]}"#,
        )
        .unwrap();
        let source_root = ProjectRoot::open(&source).unwrap();

        project
            .publish_complete_to(&target, Some(&source_root))
            .expect("Save As must carry project-local conversations");

        assert_eq!(
            fs::read(target.join("chat-sessions/chat-1.json")).unwrap(),
            br#"{"id":"chat-1","messages":[]}"#
        );
    }

    #[test]
    fn complete_publish_carries_motion_documents_across_save_as() {
        let tmp = TmpDir::new("complete-motion-documents");
        let source = tmp.path().join("Source.opentake");
        let target = tmp.path().join("Target.opentake");
        let project = Project::new(&source);
        project.save().unwrap();
        fs::create_dir_all(source.join("motion-documents/rev-document")).unwrap();
        fs::write(
            source.join("motion-documents/catalog.json"),
            br#"{"schemaVersion":1,"documents":{}}"#,
        )
        .unwrap();
        fs::write(
            source.join("motion-documents/rev-document/index.html"),
            b"<main>Motion Studio</main>",
        )
        .unwrap();
        let source_root = ProjectRoot::open(&source).unwrap();

        project
            .publish_complete_to(&target, Some(&source_root))
            .expect("Save As must carry project-local motion documents");

        assert_eq!(
            fs::read(target.join("motion-documents/catalog.json")).unwrap(),
            br#"{"schemaVersion":1,"documents":{}}"#
        );
        assert_eq!(
            fs::read(target.join("motion-documents/rev-document/index.html")).unwrap(),
            b"<main>Motion Studio</main>"
        );
    }

    #[test]
    fn generation_components_are_written_in_place_without_touching_other_components() {
        let tmp = TmpDir::new("generation-components");
        let target = tmp.path().join("Project.opentake");
        let mut project = Project::new(&target);
        project.timeline.fps = 24;
        project.save().unwrap();
        fs::create_dir_all(target.join("media")).unwrap();
        fs::write(target.join("media/clip.bin"), b"media").unwrap();
        fs::write(target.join("thumbnail.jpg"), b"cover").unwrap();
        let timeline_before = fs::read(target.join(layout::TIMELINE_FILE)).unwrap();
        let root = ProjectRoot::open(&target).unwrap();
        let identity = root.stable_identity();

        project.timeline.fps = 48;
        project.manifest.favorites.push("generated".into());
        project.generation_log = Some(GenerationLog {
            version: 1,
            entries: vec![GenerationLogEntry::new("row-1", "fal:model", None, None)],
        });
        project
            .save_manifest_and_generation_log_to_root(&root)
            .unwrap();

        assert_eq!(
            ProjectRoot::open(&target).unwrap().stable_identity(),
            identity
        );
        assert!(root.is_current_namespace().unwrap());
        assert_eq!(
            fs::read(target.join(layout::TIMELINE_FILE)).unwrap(),
            timeline_before
        );
        assert_eq!(fs::read(target.join("media/clip.bin")).unwrap(), b"media");
        assert_eq!(fs::read(target.join("thumbnail.jpg")).unwrap(), b"cover");
        let reopened = Project::open_from_root(&root).unwrap();
        assert_eq!(reopened.timeline.fps, 24);
        assert_eq!(reopened.manifest.favorites, ["generated"]);
        assert_eq!(reopened.generation_log.unwrap().entries.len(), 1);
    }

    #[test]
    fn explicit_thumbnail_removal_deletes_only_the_retained_optional_component() {
        let tmp = TmpDir::new("remove-thumbnail");
        let target = tmp.path().join("Project.opentake");
        let mut project = Project::new(&target);
        project.thumbnail = Some(b"cover".to_vec());
        project.save().unwrap();
        fs::write(target.join("keep.bin"), b"keep").unwrap();

        let root = ProjectRoot::open(&target).unwrap();
        project
            .save_to_root_with_thumbnail_update(&root, ThumbnailUpdate::Remove)
            .expect("thumbnail removal is a valid save");

        assert!(!target.join("thumbnail.jpg").exists());
        assert_eq!(fs::read(target.join("keep.bin")).unwrap(), b"keep");
        assert!(target.join("project.json").is_file());
        assert!(target.join("media.json").is_file());
    }

    #[test]
    fn staged_generated_media_is_invisible_until_published_and_committed() {
        let tmp = TmpDir::new("staged-generated-media");
        let target = tmp.path().join("Project.opentake");
        Project::new(&target).save().unwrap();
        fs::create_dir_all(target.join("media")).unwrap();
        fs::write(target.join("media/source.bin"), b"source").unwrap();
        let root = ProjectRoot::open(&target).unwrap();
        let identity = root.stable_identity();

        let staged = root
            .stage_media_leaf("output.bin", 9, &mut std::io::Cursor::new(b"generated"))
            .expect("generated media stages under the live media directory");
        assert_eq!(staged.root_identity(), identity);
        assert!(!target.join("media/output.bin").exists());
        staged.publish().unwrap().commit();

        assert_eq!(
            fs::read(target.join("media/source.bin")).unwrap(),
            b"source"
        );
        assert_eq!(
            fs::read(target.join("media/output.bin")).unwrap(),
            b"generated"
        );
        assert_eq!(fs::read_dir(target.join("media")).unwrap().count(), 2);
        assert_eq!(
            ProjectRoot::open(&target).unwrap().stable_identity(),
            identity
        );
    }

    #[test]
    fn uncommitted_published_media_rolls_back_unless_it_replaced_a_leftover() {
        let tmp = TmpDir::new("published-generated-media-rollback");
        let target = tmp.path().join("Project.opentake");
        Project::new(&target).save().unwrap();
        let root = ProjectRoot::open(&target).unwrap();
        let before = tree_receipt(&target);

        let fresh = root
            .stage_media_leaf("output.bin", 5, &mut std::io::Cursor::new(b"fresh"))
            .unwrap()
            .publish()
            .unwrap();
        assert!(target.join("media/output.bin").is_file());
        drop(fresh);
        assert!(!target.join("media/output.bin").exists());
        fs::remove_dir(target.join("media")).unwrap();
        assert_eq!(tree_receipt(&target), before);

        fs::create_dir_all(target.join("media")).unwrap();
        fs::write(target.join("media/output.bin"), b"leftover").unwrap();
        let replacement = root
            .stage_media_leaf("output.bin", 11, &mut std::io::Cursor::new(b"replacement"))
            .unwrap()
            .publish()
            .unwrap();
        drop(replacement);
        assert_eq!(
            fs::read(target.join("media/output.bin")).unwrap(),
            b"replacement"
        );
    }

    #[test]
    fn generated_media_stream_failure_preserves_the_live_bundle_byte_exact() {
        struct FailingReader(bool);

        impl std::io::Read for FailingReader {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                if self.0 {
                    return Err(std::io::Error::other("injected media read failure"));
                }
                self.0 = true;
                let bytes = b"partial";
                buffer[..bytes.len()].copy_from_slice(bytes);
                Ok(bytes.len())
            }
        }

        let tmp = TmpDir::new("staged-generated-media-failure");
        let target = tmp.path().join("Project.opentake");
        let project = Project::new(&target);
        project.save().unwrap();
        fs::create_dir_all(target.join("media")).unwrap();
        fs::write(target.join("media/source.bin"), b"source").unwrap();
        let before = tree_receipt(&target);
        let root = ProjectRoot::open(&target).unwrap();

        root.stage_media_leaf("output.bin", 14, &mut FailingReader(false))
            .expect_err("a failed generated media stream must abort staging");
        root.stage_media_leaf("output.bin", 14, &mut std::io::Cursor::new(b"short"))
            .expect_err("a truncated generated media stream must abort staging");

        assert_eq!(tree_receipt(&target), before);
        assert!(!target.join("media/output.bin").exists());
    }

    #[cfg(unix)]
    #[test]
    fn media_copy_failure_leaves_an_existing_target_tree_byte_exact() {
        use std::os::unix::fs::symlink;

        let tmp = TmpDir::new("complete-copy-failure");
        let source = tmp.path().join("Source.opentake");
        let target = tmp.path().join("Target.opentake");
        let project = Project::new(&source);
        project.save().unwrap();
        fs::create_dir_all(source.join("media")).unwrap();
        symlink(
            tmp.path().join("outside"),
            source.join("media/refused-link"),
        )
        .unwrap();
        let source_root = ProjectRoot::open(&source).unwrap();
        fs::create_dir_all(target.join("media/nested")).unwrap();
        fs::write(target.join("project.json"), b"old timeline").unwrap();
        fs::write(target.join("media/nested/clip.bin"), b"old media").unwrap();
        let before = tree_receipt(&target);

        project
            .publish_complete_to(&target, Some(&source_root))
            .expect_err("a symlink in retained project media must fail closed");

        assert_eq!(tree_receipt(&target), before);
        assert!(!fs::read_dir(tmp.path()).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".Target.opentake.opentake-stage")
        }));
        assert!(!tmp
            .path()
            .join(".Target.opentake.opentake-journal")
            .exists());
        assert!(!tmp.path().join(".Target.opentake.opentake-backup").exists());
    }

    #[test]
    fn copy_media_dir_mirrors_nested_layout() {
        let tmp = TmpDir::new("nested");
        let src = tmp.path().join("Src.opentake");
        let dst = tmp.path().join("Dst.opentake");
        let src_media = layout::media_dir(&src);
        fs::create_dir_all(src_media.join("sub")).unwrap();
        fs::write(src_media.join("a.png"), b"AAA").unwrap();
        fs::write(src_media.join("sub").join("b.mov"), b"BBBB").unwrap();

        copy_media_dir(&src, &dst).unwrap();

        assert_eq!(fs::read(dst.join("media").join("a.png")).unwrap(), b"AAA");
        assert_eq!(
            fs::read(dst.join("media").join("sub").join("b.mov")).unwrap(),
            b"BBBB"
        );
    }

    #[test]
    fn copy_media_dir_missing_source_is_noop() {
        let tmp = TmpDir::new("missing");
        let src = tmp.path().join("Src.opentake"); // no media/ under it
        let dst = tmp.path().join("Dst.opentake");
        fs::create_dir_all(&src).unwrap();

        copy_media_dir(&src, &dst).unwrap();
        assert!(!dst.join("media").exists());
    }

    #[test]
    fn copy_media_dir_same_path_is_noop() {
        let tmp = TmpDir::new("same");
        let bundle = tmp.path().join("Same.opentake");
        let media = layout::media_dir(&bundle);
        fs::create_dir_all(&media).unwrap();
        fs::write(media.join("keep.png"), b"KEEP").unwrap();

        // Source == dest: must not touch (delete/replace) the existing media/.
        copy_media_dir(&bundle, &bundle).unwrap();
        assert_eq!(fs::read(media.join("keep.png")).unwrap(), b"KEEP");
    }

    #[test]
    fn copy_media_dir_refuses_existing_dest_media_without_changes() {
        let tmp = TmpDir::new("replace");
        let src = tmp.path().join("Src.opentake");
        let dst = tmp.path().join("Dst.opentake");
        fs::create_dir_all(layout::media_dir(&src)).unwrap();
        fs::write(layout::media_dir(&src).join("new.png"), b"NEW").unwrap();
        // Pre-existing stale file in the destination media/ that is NOT in the
        // source; a full swap must not leave it behind.
        fs::create_dir_all(layout::media_dir(&dst)).unwrap();
        fs::write(layout::media_dir(&dst).join("stale.png"), b"OLD").unwrap();

        let before = tree_receipt(&dst);
        copy_media_dir(&src, &dst).expect_err("legacy media-only copy must not delete live media");

        assert_eq!(tree_receipt(&dst), before);
        assert!(!dst.join("media").join("new.png").exists());
    }
}
