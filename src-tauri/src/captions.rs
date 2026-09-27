//! The Captions-tab command: `generate_captions`.
//!
//! The UI-facing sibling of the `add_captions` MCP tool. Both run the SAME pure
//! pipeline (`opentake_media::caption_specs` for packing/timing, then
//! `EditCommand::AddCaptions` to place atomically); this command is what the
//! React Captions tab calls, mirroring upstream `EditorViewModel.generateCaptions`
//! (`EditorViewModel+Captions.swift:97-117`) driving `CaptionTab`.
//!
//! Flow: resolve caption-eligible clips (all, a track, or a clip selection);
//! transcribe each unique source (cached, language hint bypasses the cache);
//! auto-pick the dominant spoken track when the source is "auto"; build caption
//! specs with the pure builder using this timeline's canvas for text-fit and the
//! per-line transform; place them as one undoable "Generate Captions" action.
//!
//! DTOs are camelCase (`web/src/lib/types.ts` contract; the repo's #1 bug class),
//! with a serde round-trip test.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter};

use opentake_core::dto::EditResultDto;
use opentake_core::{AppCore, ProjectRevision, ProjectRuntimeSnapshot};
use opentake_domain::{Clip, ClipType, MediaManifest, TextLayout, TextStyle, Transform};
use opentake_media::{
    caption_specs, dominant_speech_track, CaptionCase, CaptionTarget, MediaCancelToken,
    MediaEngine, TranscriptionResult,
};
use opentake_ops::{CaptionEntry, EditCommand};
use tauri::State;

use crate::media::MediaState;

/// Caption style/placement defaults, 1:1 with upstream `AppTheme.Caption`
/// (`UI/AppTheme.swift:239-249`).
const DEFAULT_FONT_SIZE: f64 = 48.0;
const DEFAULT_CENTER_X: f64 = 0.5;
const DEFAULT_CENTER_Y: f64 = 0.9;
const MAX_TEXT_WIDTH_RATIO: f64 = 0.9;

/// Which clips to caption (mirrors the Captions tab's source selector). `Auto`
/// captions every eligible clip and then keeps the dominant spoken track; `Track`
/// captions one track; `Clips` captions a specific selection.
#[derive(Clone, Debug, Deserialize, PartialEq, Default)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum CaptionSource {
    /// All eligible audio, then narrowed to the dominant spoken track.
    #[default]
    Auto,
    /// Only clips on the track with this id.
    #[serde(rename_all = "camelCase")]
    Track { track_id: String },
    /// Only these clip ids.
    #[serde(rename_all = "camelCase")]
    Clips { clip_ids: Vec<String> },
}

/// Letter case on the wire (`auto`/`upper`/`lower`), mapped onto [`CaptionCase`].
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum CaptionCaseDto {
    #[default]
    Auto,
    Upper,
    Lower,
}

impl From<CaptionCaseDto> for CaptionCase {
    fn from(c: CaptionCaseDto) -> Self {
        match c {
            CaptionCaseDto::Auto => CaptionCase::Auto,
            CaptionCaseDto::Upper => CaptionCase::Upper,
            CaptionCaseDto::Lower => CaptionCase::Lower,
        }
    }
}

/// The Captions-tab request (mirror of upstream `CaptionRequest`). Style is the
/// full [`TextStyle`] (font/size/color/background/…); placement is a normalized
/// canvas center. `language` is an optional BCP-47/ISO-639 hint.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptionRequestDto {
    #[serde(default)]
    pub source: CaptionSource,
    #[serde(default)]
    pub style: Option<TextStyle>,
    #[serde(default)]
    pub center_x: Option<f64>,
    #[serde(default)]
    pub center_y: Option<f64>,
    #[serde(default)]
    pub text_case: CaptionCaseDto,
    #[serde(default)]
    pub censor_profanity: bool,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub operation_id: Option<String>,
}

/// Result of a caption Generate: the edit outcome plus a caption count for the UI.
#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GenerateCaptionsResult {
    /// The underlying edit result (version bump, affected clip ids, …).
    pub edit: EditResultDto,
    /// How many caption clips were placed (0 when no speech was detected).
    pub caption_count: usize,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CaptionGenerationProgress {
    operation_id: String,
    completed: usize,
    total: usize,
    fraction: f64,
}

type CaptionProgressCallback = Arc<dyn Fn(usize, usize, f64) + Send + Sync>;

struct ActiveCaptionGeneration {
    id: String,
    cancel: MediaCancelToken,
    notify: Option<tokio::sync::oneshot::Sender<()>>,
}

#[derive(Clone, Default)]
pub struct CaptionGenerationState {
    active: Arc<Mutex<Option<ActiveCaptionGeneration>>>,
}

struct CaptionGenerationGuard {
    state: CaptionGenerationState,
    id: String,
}

impl CaptionGenerationState {
    fn begin(
        &self,
        id: String,
    ) -> Result<
        (
            MediaCancelToken,
            tokio::sync::oneshot::Receiver<()>,
            CaptionGenerationGuard,
        ),
        String,
    > {
        let mut active = self.active.lock().unwrap_or_else(|e| e.into_inner());
        if active.is_some() {
            return Err("caption generation is already running".into());
        }
        let cancel = MediaCancelToken::new();
        let (notify, receiver) = tokio::sync::oneshot::channel();
        *active = Some(ActiveCaptionGeneration {
            id: id.clone(),
            cancel: cancel.clone(),
            notify: Some(notify),
        });
        Ok((
            cancel,
            receiver,
            CaptionGenerationGuard {
                state: self.clone(),
                id,
            },
        ))
    }

    fn cancel(&self, id: &str) -> bool {
        let mut active = self.active.lock().unwrap_or_else(|e| e.into_inner());
        let Some(entry) = active.as_mut().filter(|entry| entry.id == id) else {
            return false;
        };
        entry.cancel.cancel();
        if entry.cancel.is_cancelled() {
            if let Some(notify) = entry.notify.take() {
                let _ = notify.send(());
            }
            return true;
        }
        false
    }
}

impl Drop for CaptionGenerationGuard {
    fn drop(&mut self) {
        let mut active = self.state.active.lock().unwrap_or_else(|e| e.into_inner());
        if active.as_ref().is_some_and(|entry| entry.id == self.id) {
            *active = None;
        }
    }
}

#[tauri::command]
pub fn cancel_caption_generation(
    state: State<'_, CaptionGenerationState>,
    operation_id: String,
) -> bool {
    state.cancel(&operation_id)
}

/// `generate_captions`: transcribe the selected source and place styled caption
/// clips on a fresh top track, as one undoable action. Errors surface as a
/// `Result::Err(String)` for the UI to show (model-not-installed guides the user
/// to `download_transcribe_model`). Returns `caption_count == 0` (not an error)
/// when nothing was captionable / no speech was found, matching upstream's empty
/// return.
fn begin_caption_generation(
    admission: &crate::updater::InstallAdmissionGate,
) -> Result<crate::updater::ActivityLease, String> {
    admission.begin_activity()
}

#[tauri::command]
pub async fn generate_captions(
    app: AppHandle,
    core: State<'_, AppCore>,
    media: State<'_, MediaState>,
    admission: State<'_, crate::updater::InstallAdmissionGate>,
    state: State<'_, CaptionGenerationState>,
    request: CaptionRequestDto,
) -> Result<GenerateCaptionsResult, String> {
    use opentake_media::ort_worker::{JobKind, JobPriority, JobRequest, WorkerError};

    let activity = begin_caption_generation(&admission)?;
    let operation_id = request
        .operation_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let (cancel, cancelled, guard) = state.begin(operation_id.clone())?;
    let snapshot = core.runtime_snapshot();
    let core = core.inner().clone();
    let cache_root = media.engine().cache_root().to_path_buf();
    let models_dir = media.engine().models_dir().to_path_buf();
    let worker = crate::search::production_index_worker(media.engine().export_pause());
    let job = worker
        .submit(
            JobRequest::new(
                JobKind::Transcribe,
                format!(
                    "{}@{}",
                    opentake_media::DEFAULT_WHISPER_MODEL.file_name,
                    opentake_media::DEFAULT_WHISPER_MODEL.sha1
                ),
                format!("captions:{operation_id}"),
                JobPriority::Interactive,
            ),
            move |_, worker_cancel| {
                let _activity = activity;
                let _guard = guard;
                if worker_cancel.is_cancelled() || cancel.is_cancelled() {
                    return Err(WorkerError::Cancelled);
                }
                let engine = MediaEngine::new(cache_root, models_dir);
                let progress: CaptionProgressCallback = Arc::new(move |done, total, part| {
                    let _ = app.emit(
                        "captions://progress",
                        CaptionGenerationProgress {
                            operation_id: operation_id.clone(),
                            completed: done,
                            total,
                            fraction: (done as f64 + part) / total.max(1) as f64,
                        },
                    );
                });
                generate_captions_blocking(&core, &engine, snapshot, request, &cancel, progress)
                    .map_err(|error| {
                        if cancel.is_cancelled() {
                            WorkerError::Cancelled
                        } else {
                            WorkerError::Job(error)
                        }
                    })
            },
        )
        .map_err(|error| error.to_string())?;
    let wait = tauri::async_runtime::spawn_blocking(move || job.wait());
    tokio::select! {
        result = wait => result.map_err(|error| error.to_string())?.map_err(|error| error.to_string()),
        _ = cancelled => Err("caption generation cancelled".into()),
    }
}

fn generate_captions_blocking(
    core: &AppCore,
    engine: &MediaEngine,
    snapshot: ProjectRuntimeSnapshot,
    request: CaptionRequestDto,
    cancel: &MediaCancelToken,
    on_progress: CaptionProgressCallback,
) -> Result<GenerateCaptionsResult, String> {
    let revision = ProjectRevision {
        project_epoch: snapshot.project_epoch,
        version: snapshot.version,
    };
    let timeline = &snapshot.timeline;
    let manifest = &snapshot.media;
    let fps = timeline.fps;

    // Style + placement (defaults: 48-pt caption near the bottom, white).
    let mut style = request.style.unwrap_or_else(|| TextStyle {
        font_size: DEFAULT_FONT_SIZE,
        ..TextStyle::default()
    });
    if style.font_size <= 0.0 {
        style.font_size = DEFAULT_FONT_SIZE;
    }
    let center_x = request.center_x.unwrap_or(DEFAULT_CENTER_X);
    let center_y = request.center_y.unwrap_or(DEFAULT_CENTER_Y);
    let case: CaptionCase = request.text_case.into();

    // Resolve the requested language against the backend's supported set.
    let language = match request.language.as_deref() {
        None => None,
        Some(lang) => Some(opentake_media::match_language(lang).ok_or_else(|| {
            format!("on-device transcription does not support language '{lang}'.")
        })?),
    };

    // Caption-eligible clips for the chosen source (each with its track id).
    let auto_detect = matches!(request.source, CaptionSource::Auto);
    let eligible = eligible_targets(timeline, manifest, &request.source);
    if eligible.is_empty() {
        if cancel.is_cancelled() {
            return Err("caption generation cancelled".into());
        }
        return Ok(GenerateCaptionsResult {
            edit: unchanged_edit(&snapshot.version),
            caption_count: 0,
        });
    }

    // Transcribe each unique source once. Skip-don't-fail per source (a missing
    // file / decode error / model-not-installed skips just that clip); if EVERY
    // source failed with the same reason, surface it (so "model not installed"
    // reaches the UI instead of a silent empty result).
    //
    // A language hint OR profanity masking makes the transcript differ from the
    // shared auto-detect cache, so those variants transcribe directly with the
    // options threaded to the backend (upstream bypasses the cache for option
    // variants, `EditorViewModel+Captions.swift:127`). The plain case uses the
    // caching convenience so repeats are instant. `censor_profanity` is honored
    // here so it takes effect if/when the whisper backend gains masking (today it
    // is a no-op in the backend, matching upstream's transcription-level boundary).
    let uses_options = language.is_some() || request.censor_profanity;
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let sources = eligible
        .iter()
        .filter(|t| seen.insert(t.media_ref.clone()))
        .map(|t| {
            (
                t.media_ref.clone(),
                crate::transcribe::resolve_asset_from_snapshot(&snapshot, &t.media_ref),
            )
        })
        .collect::<Vec<_>>();
    let source_progress = Arc::clone(&on_progress);
    let transcripts = transcribe_unique_sources(
        sources,
        cancel,
        |path| {
            (!uses_options)
                .then(|| {
                    opentake_media::transcribe::cache::cached_on_disk(engine.cache_root(), path)
                })
                .flatten()
        },
        || crate::transcribe::load_backend(engine),
        |path, _is_video, backend, completed, total| {
            let progress = Arc::clone(&source_progress);
            let last_percent = Arc::new(std::sync::atomic::AtomicU32::new(u32::MAX));
            let opts = opentake_media::TranscribeOptions {
                preferred_language: language.clone(),
                censor_profanity: request.censor_profanity,
                cancel: Some(cancel.clone()),
                progress: Some(opentake_media::transcribe::TranscriptionProgress(Arc::new(
                    move |part: f64| {
                        use std::sync::atomic::Ordering;
                        let percent = (part.clamp(0.0, 1.0) * 100.0).floor() as u32;
                        if last_percent.swap(percent, Ordering::Relaxed) != percent {
                            progress(completed, total, part);
                        }
                    },
                ))),
                ..Default::default()
            };
            let result = opentake_media::transcribe::transcribe_file(path, backend, &opts)
                .map_err(|error| error.to_string())?;
            if !uses_options && !cancel.is_cancelled() {
                crate::transcribe::persist_full_transcript(engine.cache_root(), path, &result);
            }
            Ok(result)
        },
        |done, total| on_progress(done, total, 0.0),
    )?;
    if transcripts.is_empty() {
        if cancel.is_cancelled() {
            return Err("caption generation cancelled".into());
        }
        return Ok(GenerateCaptionsResult {
            edit: unchanged_edit(&snapshot.version),
            caption_count: 0,
        });
    }

    // Build caption targets (clip + track id + resolved transcript).
    let targets: Vec<CaptionTarget<'_>> = eligible
        .iter()
        .map(|t| CaptionTarget {
            clip_id: t.clip.id.clone(),
            track_id: t.track_id.clone(),
            clip: t.clip,
            transcript: transcripts.get(&t.media_ref),
        })
        .collect();

    // Auto source: keep only the dominant spoken track.
    let targets: Vec<CaptionTarget<'_>> = if auto_detect {
        match dominant_speech_track(&targets, fps) {
            Some(winner) => targets
                .into_iter()
                .filter(|t| t.track_id == winner)
                .collect(),
            None => {
                if cancel.is_cancelled() {
                    return Err("caption generation cancelled".into());
                }
                return Ok(GenerateCaptionsResult {
                    edit: unchanged_edit(&snapshot.version),
                    caption_count: 0,
                });
            }
        }
    } else {
        targets
    };

    // Build specs via the pure builder. `fits` + the per-line transform use this
    // timeline's canvas (upstream `captionLineFits` / `captionTransform`).
    let group_id = new_caption_group_id();
    let canvas_w = timeline.width.max(1) as f64;
    let canvas_h = timeline.height.max(1) as f64;
    let max_text_w = canvas_w * MAX_TEXT_WIDTH_RATIO;
    let fits = |line: &str| {
        let (w, _) = TextLayout::natural_size(line, &style, f64::MAX, canvas_h);
        w <= max_text_w
    };
    let specs = caption_specs(&targets, fps, case, &group_id, &fits);
    if specs.is_empty() {
        if cancel.is_cancelled() {
            return Err("caption generation cancelled".into());
        }
        return Ok(GenerateCaptionsResult {
            edit: unchanged_edit(&snapshot.version),
            caption_count: 0,
        });
    }

    let entries: Vec<CaptionEntry> = specs
        .into_iter()
        .map(|s| {
            let (w, h) = TextLayout::natural_size(&s.content, &style, max_text_w, canvas_h);
            let transform = Transform {
                center_x,
                center_y,
                width: w / canvas_w,
                height: h / canvas_h,
                ..Transform::default()
            };
            CaptionEntry {
                start_frame: s.start_frame,
                duration_frames: s.duration_frames,
                content: s.content,
                text_style: style.clone(),
                transform,
                caption_group_id: s.caption_group_id,
            }
        })
        .collect();

    let count = entries.len();
    // Place atomically only if the project and version still match the snapshot
    // used for media resolution and caption layout.
    if !cancel.try_commit() {
        return Err("caption generation cancelled".into());
    }
    let edit = apply_captions_at_revision(core, revision, entries)?;
    Ok(GenerateCaptionsResult {
        edit,
        caption_count: count,
    })
}

fn apply_captions_at_revision(
    core: &AppCore,
    revision: ProjectRevision,
    entries: Vec<CaptionEntry>,
) -> Result<EditResultDto, String> {
    core.apply_at_revision(revision, EditCommand::AddCaptions { entries })
        .map(EditResultDto::from)
        .map_err(|error| error.to_string())
}

/// Resolve each distinct source once. The backend is loaded lazily on the first
/// cache miss and kept alive across all sources (including a failed source).
/// The closures make loading, cache hits and cancellation testable without a
/// Whisper model or a desktop runtime.
type CaptionSource = (String, Result<(PathBuf, bool), String>);

fn transcribe_unique_sources<B>(
    sources: Vec<CaptionSource>,
    cancel: &MediaCancelToken,
    mut cached: impl FnMut(&Path) -> Option<TranscriptionResult>,
    mut load: impl FnMut() -> Result<B, String>,
    mut transcribe: impl FnMut(&Path, bool, &B, usize, usize) -> Result<TranscriptionResult, String>,
    mut progress: impl FnMut(usize, usize),
) -> Result<HashMap<String, TranscriptionResult>, String> {
    let total = sources.len();
    progress(0, total);
    let mut backend: Option<Result<B, String>> = None;
    let mut transcripts = HashMap::new();
    let mut first_error = None;
    for (completed, (id, source)) in sources.into_iter().enumerate() {
        if cancel.is_cancelled() {
            return Err("caption generation cancelled".into());
        }
        let result = source.and_then(|(path, is_video)| {
            if let Some(result) = cached(&path) {
                return Ok(result);
            }
            let backend = backend.get_or_insert_with(&mut load);
            match backend {
                Ok(backend) => transcribe(&path, is_video, backend, completed, total),
                Err(error) => Err(error.clone()),
            }
        });
        if cancel.is_cancelled() {
            return Err("caption generation cancelled".into());
        }
        match result {
            Ok(result) => {
                transcripts.insert(id, result);
            }
            Err(error) => {
                first_error.get_or_insert(error);
            }
        }
        progress(completed + 1, total);
    }
    if transcripts.is_empty() {
        if let Some(error) = first_error {
            return Err(error);
        }
    }
    Ok(transcripts)
}

/// One caption-eligible clip located on the timeline: the clip + its track id +
/// its source `media_ref`.
struct EligibleTarget<'a> {
    clip: &'a Clip,
    track_id: String,
    media_ref: String,
}

/// Caption-eligible clips for the chosen [`CaptionSource`], mirroring upstream
/// `captionTargets(in:)` (`EditorViewModel+Captions.swift:80-89`): keep
/// audio/video clips whose asset can be transcribed, but drop a **video** clip
/// whose link group also has a linked **audio** clip (that audio partner is
/// transcribed instead). `Track` restricts to one track; `Clips` to a selection.
fn eligible_targets<'a>(
    timeline: &'a opentake_domain::Timeline,
    manifest: &MediaManifest,
    source: &CaptionSource,
) -> Vec<EligibleTarget<'a>> {
    // Link groups that contain at least one audio clip anywhere.
    let audio_link_groups: std::collections::BTreeSet<&str> = timeline
        .tracks
        .iter()
        .flat_map(|t| &t.clips)
        .filter(|c| c.media_type == ClipType::Audio)
        .filter_map(|c| c.link_group_id.as_deref())
        .collect();

    let want_track: Option<&str> = match source {
        CaptionSource::Track { track_id } => Some(track_id.as_str()),
        _ => None,
    };
    let want_clips: Option<std::collections::BTreeSet<&str>> = match source {
        CaptionSource::Clips { clip_ids } => Some(clip_ids.iter().map(String::as_str).collect()),
        _ => None,
    };

    let mut out = Vec::new();
    for track in &timeline.tracks {
        if let Some(tid) = want_track {
            if track.id != tid {
                continue;
            }
        }
        for clip in &track.clips {
            if let Some(clips) = &want_clips {
                if !clips.contains(clip.id.as_str()) {
                    continue;
                }
            }
            if !can_transcribe(clip, manifest) {
                continue;
            }
            if clip.media_type == ClipType::Video {
                if let Some(gid) = clip.link_group_id.as_deref() {
                    if audio_link_groups.contains(gid) {
                        continue;
                    }
                }
            }
            out.push(EligibleTarget {
                clip,
                track_id: track.id.clone(),
                media_ref: clip.media_ref.clone(),
            });
        }
    }
    out.sort_by_key(|t| t.clip.start_frame);
    out
}

/// Whether a clip can be transcribed, mirroring upstream `captionCanTranscribe`:
/// media type must be video/audio, and (when the asset is known) it must be audio
/// or a video WITH an audio track. Unknown assets are permissively eligible.
fn can_transcribe(clip: &Clip, manifest: &MediaManifest) -> bool {
    if !matches!(clip.media_type, ClipType::Video | ClipType::Audio) {
        return false;
    }
    match manifest.entries.iter().find(|e| e.id == clip.media_ref) {
        None => true,
        Some(entry) => {
            entry.kind == ClipType::Audio
                || (entry.kind == ClipType::Video && entry.has_audio.unwrap_or(false))
        }
    }
}

/// The "nothing changed" edit result (no caption track created). Mirrors the
/// shape of an `EditResult` for a no-op so the UI's version stays put.
fn unchanged_edit(version: &u64) -> EditResultDto {
    EditResultDto {
        changed: false,
        action_name: "Generate Captions".into(),
        affected_clip_ids: Vec::new(),
        timeline_version: *version,
        summary: String::new(),
    }
}

/// Mint a fresh caption-group id (upstream `UUID().uuidString`) without a uuid
/// dependency: a process-wide counter plus a nanosecond timestamp. Opaque; only
/// used for group membership (subtitle export + caption-group style sync).
fn new_caption_group_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("cap-{nanos:x}-{n:x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentake_domain::{MediaManifestEntry, MediaSource, Timeline, Track};

    fn transcript(text: &str) -> TranscriptionResult {
        TranscriptionResult {
            text: text.into(),
            language: Some("en".into()),
            segments: Vec::new(),
            words: Vec::new(),
        }
    }

    #[test]
    fn caption_sources_reuse_one_backend_and_continue_after_a_failed_source() {
        let sources = (0..3)
            .map(|index| {
                (
                    format!("source-{index}"),
                    Ok((PathBuf::from(format!("source-{index}.wav")), false)),
                )
            })
            .collect();
        let mut loads = 0;
        let mut progress = Vec::new();
        let results = transcribe_unique_sources(
            sources,
            &MediaCancelToken::new(),
            |_| None, // an options variant bypasses the shared transcript cache
            || {
                loads += 1;
                Ok(())
            },
            |path, _, _, _, _| {
                if path == Path::new("source-1.wav") {
                    Err("bad audio".into())
                } else {
                    Ok(transcript(&path.to_string_lossy()))
                }
            },
            |done, total| progress.push((done, total)),
        )
        .unwrap();
        assert_eq!(loads, 1);
        assert_eq!(results.len(), 2);
        assert_eq!(progress, [(0, 3), (1, 3), (2, 3), (3, 3)]);
    }

    #[test]
    fn caption_cancel_during_transcription_stops_before_commit() {
        let state = CaptionGenerationState::default();
        let (cancel, mut notified, guard) = state.begin("op-1".into()).unwrap();
        assert!(!state.cancel("unrelated"));
        let result = transcribe_unique_sources(
            vec![("source".into(), Ok((PathBuf::from("speech.wav"), false)))],
            &cancel,
            |_| None,
            || Ok(()),
            |_, _, _, _, _| {
                assert!(state.cancel("op-1"));
                Ok(transcript("speech"))
            },
            |_, _| {},
        );
        assert!(result.unwrap_err().contains("cancelled"));
        assert!(!cancel.try_commit());
        assert!(notified.try_recv().is_ok());
        drop(guard);
        assert!(!state.cancel("op-1"));
    }

    fn entry(id: &str, kind: ClipType, has_audio: bool) -> MediaManifestEntry {
        MediaManifestEntry {
            id: id.into(),
            name: id.into(),
            kind,
            source: MediaSource::External {
                absolute_path: format!("/{id}"),
            },
            duration: 1.0,
            generation_input: None,
            source_width: None,
            source_height: None,
            source_fps: None,
            has_audio: Some(has_audio),
            color: None,
            proxy: None,
            folder_id: None,
            cached_remote_url: None,
            cached_remote_url_expires_at: None,
        }
    }

    #[test]
    fn request_dto_deserializes_camelcase() {
        // The Captions tab sends camelCase; every multi-word field must decode.
        let req: CaptionRequestDto = serde_json::from_str(
            r#"{"source":{"kind":"clips","clipIds":["c1","c2"]},
                "centerX":0.5,"centerY":0.9,"textCase":"upper",
                "censorProfanity":true,"language":"es"}"#,
        )
        .expect("camelCase request");
        assert_eq!(
            req.source,
            CaptionSource::Clips {
                clip_ids: vec!["c1".into(), "c2".into()]
            }
        );
        assert_eq!(req.center_y, Some(0.9));
        assert_eq!(req.text_case, CaptionCaseDto::Upper);
        assert!(req.censor_profanity);
        assert_eq!(req.language.as_deref(), Some("es"));
    }

    #[test]
    fn request_dto_defaults_to_auto_source() {
        let req: CaptionRequestDto = serde_json::from_str("{}").expect("empty request");
        assert_eq!(req.source, CaptionSource::Auto);
        assert_eq!(req.text_case, CaptionCaseDto::Auto);
        assert!(!req.censor_profanity);
    }

    #[test]
    fn caption_generation_is_rejected_while_update_install_owns_admission() {
        let admission = crate::updater::InstallAdmissionGate::default();
        let install = admission.begin_install().unwrap();
        assert!(begin_caption_generation(&admission).is_err());
        drop(install);
        assert!(begin_caption_generation(&admission).is_ok());
    }

    #[test]
    fn result_serializes_camelcase() {
        let r = GenerateCaptionsResult {
            edit: unchanged_edit(&3),
            caption_count: 2,
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("\"captionCount\":2"));
        assert!(json.contains("\"timelineVersion\":3"));
    }

    fn sample_caption_entry() -> CaptionEntry {
        CaptionEntry {
            start_frame: 0,
            duration_frames: 24,
            content: "snapshot caption".into(),
            text_style: TextStyle::default(),
            transform: Transform::default(),
            caption_group_id: "snapshot-group".into(),
        }
    }

    #[test]
    fn caption_commit_rejects_stale_project_revision() {
        let current = AppCore::new();
        let revision = current.project_revision();
        apply_captions_at_revision(&current, revision, vec![sample_caption_entry()])
            .expect("valid caption edit applies at its source revision");
        assert_eq!(current.get_timeline().timeline.tracks.len(), 1);

        let stale = AppCore::new();
        let revision = stale.project_revision();
        stale.new_project();
        let before = stale.runtime_snapshot();
        let error = apply_captions_at_revision(&stale, revision, vec![sample_caption_entry()])
            .expect_err("caption result from the replaced project must not commit");
        assert_eq!(error, "project changed while preparing a deferred edit");
        let after = stale.runtime_snapshot();
        assert_eq!(after.project_epoch, before.project_epoch);
        assert_eq!(after.version, before.version);
        assert_eq!(after.timeline, before.timeline);
        assert_eq!(after.media, before.media);
    }

    fn tl_with_audio() -> Timeline {
        let mut tl = Timeline::new();
        let mut vt = Track::new("v", ClipType::Video);
        // A silent video clip (has_audio=false asset) — not eligible.
        vt.clips.push(Clip::new("v-silent", "vid", 0, 60));
        tl.tracks.push(vt);
        let mut at = Track::new("a", ClipType::Audio);
        let mut ac = Clip::new("a1", "aud", 0, 60);
        ac.media_type = ClipType::Audio;
        at.clips.push(ac);
        tl.tracks.push(at);
        tl
    }

    fn manifest_with_audio() -> MediaManifest {
        let mut m = MediaManifest::new();
        m.entries.push(entry("vid", ClipType::Video, false));
        m.entries.push(entry("aud", ClipType::Audio, true));
        m
    }

    #[test]
    fn eligible_auto_keeps_audio_drops_silent_video() {
        let tl = tl_with_audio();
        let m = manifest_with_audio();
        let targets = eligible_targets(&tl, &m, &CaptionSource::Auto);
        let ids: Vec<&str> = targets.iter().map(|t| t.clip.id.as_str()).collect();
        assert_eq!(ids, vec!["a1"]);
        assert_eq!(targets[0].track_id, "a");
    }

    #[test]
    fn eligible_track_scopes_to_one_track() {
        let tl = tl_with_audio();
        let m = manifest_with_audio();
        let targets = eligible_targets(
            &tl,
            &m,
            &CaptionSource::Track {
                track_id: "a".into(),
            },
        );
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].clip.id, "a1");
        // The (silent) video track is excluded by the track filter.
        let none = eligible_targets(
            &tl,
            &m,
            &CaptionSource::Track {
                track_id: "v".into(),
            },
        );
        assert!(none.is_empty());
    }

    #[test]
    fn eligible_clips_scopes_to_selection() {
        let tl = tl_with_audio();
        let m = manifest_with_audio();
        let targets = eligible_targets(
            &tl,
            &m,
            &CaptionSource::Clips {
                clip_ids: vec!["a1".into()],
            },
        );
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].clip.id, "a1");
    }

    #[test]
    fn eligible_drops_video_with_linked_audio() {
        let mut tl = Timeline::new();
        let mut vt = Track::new("v", ClipType::Video);
        let mut vc = Clip::new("v1", "vid_a", 0, 60);
        vc.link_group_id = Some("grp".into());
        vt.clips.push(vc);
        tl.tracks.push(vt);
        let mut at = Track::new("a", ClipType::Audio);
        let mut ac = Clip::new("a1", "aud", 0, 60);
        ac.media_type = ClipType::Audio;
        ac.link_group_id = Some("grp".into());
        at.clips.push(ac);
        tl.tracks.push(at);
        let mut m = MediaManifest::new();
        m.entries.push(entry("vid_a", ClipType::Video, true));
        m.entries.push(entry("aud", ClipType::Audio, true));
        let targets = eligible_targets(&tl, &m, &CaptionSource::Auto);
        let ids: Vec<&str> = targets.iter().map(|t| t.clip.id.as_str()).collect();
        assert!(!ids.contains(&"v1"), "linked video should be dropped");
        assert!(ids.contains(&"a1"));
    }
}
