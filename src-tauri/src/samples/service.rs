//! Remote sample-project materialization.
//!
//! An open reuses an earlier copy of the same sample only while that copy's
//! content identity still matches the one recorded when it was published, so
//! a copy the user edited is never handed out again; otherwise it creates an
//! independent project copy in durable application data. The complete bundle
//! is validated before publication; failures remove only directories owned by
//! that attempt, never an existing editable project.

use std::collections::HashSet;
use std::fs;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use opentake_domain::{Clip, ClipType, MediaManifest, TextStyle, Timeline, Track};
use opentake_project::{layout, Project};
use reqwest::blocking::{Client, Response};
use reqwest::redirect::Policy;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

const MAX_RESOLVE_BYTES: u64 = 4 * 1024 * 1024;
const MAX_DOWNLOAD_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
/// Written next to each published copy's bundle (outside it) to identify the
/// sample and the copy's content at publication.
const COPY_RECORD_FILE: &str = "sample-copy.json";
const MAX_COPY_RECORD_BYTES: u64 = 64 * 1024;
const COPY_RECORD_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SampleCopyRecord {
    version: u32,
    slug: String,
    /// SHA-256 of the resolved sample the copy was built from.
    source: String,
    /// The bundle directory name inside the copy's directory.
    bundle: String,
    /// [`bundle_content_identity`] right after publication.
    content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SampleDownload {
    id: String,
    relative_path: String,
    url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SampleChatDownload {
    name: String,
    url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ResolvedSample {
    title: String,
    project: Value,
    manifest: Value,
    #[serde(default)]
    generation_log: Option<Value>,
    #[serde(default)]
    poster_url: Option<String>,
    #[serde(default)]
    downloads: Vec<SampleDownload>,
    #[serde(default)]
    chat: Vec<SampleChatDownload>,
}

struct StagingDirectory {
    path: PathBuf,
    armed: bool,
}

impl Drop for StagingDirectory {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

pub(super) struct SampleProjectService {
    storage_root: PathBuf,
    client: Client,
}

impl SampleProjectService {
    pub(super) fn new(storage_root: PathBuf) -> Result<Self, String> {
        Ok(Self {
            storage_root,
            client: Client::builder()
                .redirect(sample_redirect_policy())
                .timeout(REQUEST_TIMEOUT)
                .build()
                .map_err(|error| format!("configure sample HTTP client: {error}"))?,
        })
    }

    /// `in_use` names bundles that must not be reused, such as the open
    /// project, whose unsaved edits the on-disk identity cannot see.
    pub(super) fn materialize(
        &self,
        backend_url: &str,
        slug: &str,
        in_use: impl Fn(&Path) -> bool,
        on_progress: impl FnMut(f64),
    ) -> Result<PathBuf, String> {
        validate_slug(slug)?;
        let mut endpoint = validated_network_url(backend_url)?;
        endpoint.set_path("/v1/samples/resolve");
        endpoint.set_query(None);
        endpoint.query_pairs_mut().append_pair("slug", slug);
        let response = self
            .client
            .get(endpoint.clone())
            .send()
            .map_err(|error| format!("resolve sample {slug}: {error}"))?;
        let bytes = read_bounded_response(response, MAX_RESOLVE_BYTES, "sample metadata")?;
        let resolved: ResolvedSample = serde_json::from_slice(&bytes)
            .map_err(|error| format!("decode sample {slug}: {error}"))?;
        self.materialize_resolved(
            slug,
            resolved,
            |download, target| {
                let url = validated_network_url(&download.url)?;
                if !can_follow_sample_redirect(&endpoint, &endpoint, &url) {
                    return Err("sample download URL is outside the authorized transport or loopback origin".into());
                }
                self.download_file(download, target)
            },
            in_use,
            on_progress,
        )
    }

    pub(super) fn materialize_builtin(
        &self,
        slug: &str,
        in_use: impl Fn(&Path) -> bool,
        on_progress: impl FnMut(f64),
    ) -> Result<PathBuf, String> {
        self.materialize_resolved(
            slug,
            builtin_sample(slug)?,
            |_, _| Err("built-in sample unexpectedly requested a download".into()),
            in_use,
            on_progress,
        )
    }

    fn materialize_resolved(
        &self,
        slug: &str,
        resolved: ResolvedSample,
        mut download_file: impl FnMut(&SampleDownload, &Path) -> Result<(), String>,
        in_use: impl Fn(&Path) -> bool,
        mut on_progress: impl FnMut(f64),
    ) -> Result<PathBuf, String> {
        validate_slug(slug)?;
        fs::create_dir_all(&self.storage_root)
            .map_err(|error| format!("create sample projects directory: {error}"))?;
        let source = sample_source_identity(slug, &resolved)?;
        if let Some(existing) = self.reusable_copy(slug, &source, &in_use) {
            on_progress(0.0);
            on_progress(1.0);
            return Ok(existing);
        }
        let stage_root = self
            .storage_root
            .join(format!(".{slug}.{}.tmp", uuid::Uuid::new_v4()));
        fs::create_dir(&stage_root).map_err(|error| format!("create sample stage: {error}"))?;
        let stage = StagingDirectory {
            path: stage_root,
            armed: true,
        };
        let bundle = stage
            .path
            .join(format!("{}.opentake", safe_name(&resolved.title)));
        fs::create_dir_all(bundle.join(layout::MEDIA_DIR))
            .map_err(|error| format!("create sample bundle: {error}"))?;
        write_json(&bundle.join(layout::TIMELINE_FILE), &resolved.project)?;
        write_json(&bundle.join(layout::MANIFEST_FILE), &resolved.manifest)?;
        if let Some(log) = &resolved.generation_log {
            write_json(&bundle.join(layout::GENERATION_LOG_FILE), log)?;
        }

        let mut downloads = resolved.downloads;
        downloads.extend(resolved.chat.into_iter().map(|chat| SampleDownload {
            id: chat.name.clone(),
            relative_path: format!("chat-sessions/{}", chat.name),
            url: chat.url,
        }));
        if let Some(url) = resolved.poster_url {
            downloads.push(SampleDownload {
                id: "poster".into(),
                relative_path: layout::THUMBNAIL_FILE.into(),
                url,
            });
        }
        validate_downloads(&downloads)?;
        let total = downloads.len().max(1);
        on_progress(0.0);
        if downloads.is_empty() {
            on_progress(1.0);
        }
        for (index, download) in downloads.iter().enumerate() {
            let target = safe_target(&bundle, &download.relative_path)?;
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)
                    .map_err(|error| format!("create sample download directory: {error}"))?;
            }
            download_file(download, &target)?;
            on_progress((index + 1) as f64 / total as f64);
        }

        Project::open(&bundle).map_err(|error| format!("validate sample bundle: {error}"))?;
        let published_root = self
            .storage_root
            .join(format!("{slug}-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&published_root)
            .map_err(|error| format!("create sample project directory: {error}"))?;
        let mut publication = StagingDirectory {
            path: published_root,
            armed: true,
        };
        let name = bundle.file_name().ok_or("sample bundle has no filename")?;
        let destination = publication.path.join(name);
        fs::rename(&bundle, &destination)
            .map_err(|error| format!("publish sample project: {error}"))?;
        publication.armed = false;
        // The record only enables reuse; a copy without one is still a
        // complete project, and the next open creates a fresh copy.
        if let Err(error) = write_copy_record(&publication.path, slug, &source, name) {
            eprintln!("[samples] sample copy will not be reused: {error}");
        }
        Ok(destination)
    }

    /// An earlier copy of this exact sample whose content is unchanged since
    /// publication. Unreadable, mismatched or in-use candidates are skipped.
    fn reusable_copy(
        &self,
        slug: &str,
        source: &str,
        in_use: &impl Fn(&Path) -> bool,
    ) -> Option<PathBuf> {
        let prefix = format!("{slug}-");
        let mut candidates = fs::read_dir(&self.storage_root)
            .ok()?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .is_some_and(|name| name.starts_with(&prefix))
                    && entry.file_type().is_ok_and(|kind| kind.is_dir())
            })
            .map(|entry| entry.path())
            .collect::<Vec<_>>();
        candidates.sort();
        candidates.into_iter().find_map(|directory| {
            let record = read_copy_record(&directory)?;
            if record.version != COPY_RECORD_VERSION
                || record.slug != slug
                || record.source != source
            {
                return None;
            }
            let bundle = directory.join(&record.bundle);
            if in_use(&bundle) {
                return None;
            }
            (bundle_content_identity(&bundle).ok()? == record.content).then_some(bundle)
        })
    }

    fn download_file(&self, download: &SampleDownload, target: &Path) -> Result<(), String> {
        let url = validated_network_url(&download.url)?;
        let mut response = self
            .client
            .get(url)
            .send()
            .map_err(|error| format!("download sample file {}: {error}", download.id))?;
        validate_response(&response, MAX_DOWNLOAD_BYTES, &download.id)?;
        let temp = target.with_extension(format!("download-{}.tmp", uuid::Uuid::new_v4()));
        let result = (|| {
            let mut file = fs::File::create(&temp)
                .map_err(|error| format!("create sample file {}: {error}", download.id))?;
            let mut copied = 0_u64;
            let mut buffer = [0_u8; 128 * 1024];
            loop {
                let read = response
                    .read(&mut buffer)
                    .map_err(|error| format!("read sample file {}: {error}", download.id))?;
                if read == 0 {
                    break;
                }
                copied = copied.saturating_add(read as u64);
                if copied > MAX_DOWNLOAD_BYTES {
                    return Err(format!(
                        "{}: response exceeds {} bytes",
                        download.id, MAX_DOWNLOAD_BYTES
                    ));
                }
                file.write_all(&buffer[..read])
                    .map_err(|error| format!("write sample file {}: {error}", download.id))?;
            }
            file.sync_all()
                .map_err(|error| format!("sync sample file {}: {error}", download.id))?;
            fs::rename(&temp, target)
                .map_err(|error| format!("publish sample file {}: {error}", download.id))
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temp);
        }
        result
    }
}

fn builtin_sample(slug: &str) -> Result<ResolvedSample, String> {
    let (title, cards): (&str, &[&str]) = match slug {
        "product-demo" => (
            "OpenTake Product Demo",
            &[
                "Welcome to OpenTake",
                "Import media, trim clips, and add captions",
                "Export when your story is ready",
            ],
        ),
        "quick-tutorial" => (
            "OpenTake Quick Tutorial",
            &[
                "1. Import media from the Media panel",
                "2. Drag clips onto the timeline and trim their edges",
                "3. Press Space to preview, then Export",
            ],
        ),
        "template-project" => ("OpenTake Template", &[]),
        _ => return Err(format!("unknown built-in sample: {slug}")),
    };
    let mut timeline = Timeline::new();
    timeline.settings_configured = true;
    if !cards.is_empty() {
        let mut track = Track::new("sample-text", ClipType::Text);
        for (index, content) in cards.iter().enumerate() {
            let mut clip = Clip::new(format!("sample-text-{index}"), "", index as i32 * 120, 120);
            clip.media_type = ClipType::Text;
            clip.source_clip_type = ClipType::Text;
            clip.text_content = Some((*content).into());
            clip.text_style = Some(TextStyle::default());
            track.clips.push(clip);
        }
        timeline.tracks.push(track);
    }
    Ok(ResolvedSample {
        title: title.into(),
        project: serde_json::to_value(timeline)
            .map_err(|error| format!("encode built-in sample timeline: {error}"))?,
        manifest: serde_json::to_value(MediaManifest::new())
            .map_err(|error| format!("encode built-in sample manifest: {error}"))?,
        generation_log: None,
        poster_url: None,
        downloads: vec![],
        chat: vec![],
    })
}

fn sample_source_identity(slug: &str, resolved: &ResolvedSample) -> Result<String, String> {
    let encoded =
        serde_json::to_vec(resolved).map_err(|error| format!("encode sample identity: {error}"))?;
    let mut hasher = Sha256::new();
    hasher.update(slug.as_bytes());
    hasher.update([0]);
    hasher.update(&encoded);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Content identity of a sample copy: the timeline, manifest and generation
/// log as the project layer reads them (so a save that only re-encodes them
/// keeps the identity), plus the path and bytes of every other file. Only the
/// cover thumbnail, which saving regenerates, is left out. A symlink or any
/// other non-regular entry fails, so such a copy is never reused.
fn bundle_content_identity(bundle: &Path) -> Result<String, String> {
    let kind = fs::symlink_metadata(bundle)
        .map_err(|error| format!("inspect sample copy: {error}"))?
        .file_type();
    if !kind.is_dir() {
        return Err("sample copy is not a directory".into());
    }
    let project = Project::open(bundle).map_err(|error| format!("open sample copy: {error}"))?;
    let mut hasher = Sha256::new();
    for (label, encoded) in [
        ("timeline", serde_json::to_vec(&project.timeline)),
        ("manifest", serde_json::to_vec(&project.manifest)),
        (
            "generation-log",
            serde_json::to_vec(&project.generation_log),
        ),
    ] {
        let encoded = encoded.map_err(|error| format!("encode sample {label}: {error}"))?;
        hasher.update(label.as_bytes());
        hasher.update((encoded.len() as u64).to_le_bytes());
        hasher.update(&encoded);
    }
    let mut files = Vec::new();
    collect_bundle_files(bundle, "", &mut files)?;
    files.sort();
    for relative in files {
        if matches!(
            relative.as_str(),
            layout::TIMELINE_FILE
                | layout::MANIFEST_FILE
                | layout::GENERATION_LOG_FILE
                | layout::THUMBNAIL_FILE
        ) {
            continue;
        }
        hasher.update(b"file");
        hasher.update((relative.len() as u64).to_le_bytes());
        hasher.update(relative.as_bytes());
        let mut file = fs::File::open(bundle.join(&relative))
            .map_err(|error| format!("read sample copy {relative}: {error}"))?;
        let mut content = Sha256::new();
        std::io::copy(&mut file, &mut content)
            .map_err(|error| format!("read sample copy {relative}: {error}"))?;
        hasher.update(content.finalize());
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn collect_bundle_files(
    directory: &Path,
    prefix: &str,
    files: &mut Vec<String>,
) -> Result<(), String> {
    for entry in fs::read_dir(directory).map_err(|error| format!("list sample copy: {error}"))? {
        let entry = entry.map_err(|error| format!("list sample copy: {error}"))?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| "sample copy has a non-UTF-8 file name".to_string())?;
        let relative = format!("{prefix}{name}");
        let kind = entry
            .file_type()
            .map_err(|error| format!("inspect sample copy {relative}: {error}"))?;
        if kind.is_dir() {
            collect_bundle_files(&entry.path(), &format!("{relative}/"), files)?;
        } else if kind.is_file() {
            files.push(relative);
        } else {
            return Err(format!("sample copy holds a non-regular entry: {relative}"));
        }
    }
    Ok(())
}

fn write_copy_record(
    directory: &Path,
    slug: &str,
    source: &str,
    bundle_name: &std::ffi::OsStr,
) -> Result<(), String> {
    let bundle = bundle_name
        .to_str()
        .ok_or("sample bundle name is not UTF-8")?
        .to_string();
    let record = SampleCopyRecord {
        version: COPY_RECORD_VERSION,
        slug: slug.to_string(),
        source: source.to_string(),
        content: bundle_content_identity(&directory.join(&bundle))?,
        bundle,
    };
    let bytes = serde_json::to_vec_pretty(&record)
        .map_err(|error| format!("encode sample copy record: {error}"))?;
    let temp = directory.join(format!(".{COPY_RECORD_FILE}.{}.tmp", uuid::Uuid::new_v4()));
    let result = (|| {
        let mut file = fs::File::create_new(&temp)
            .map_err(|error| format!("create sample copy record: {error}"))?;
        file.write_all(&bytes)
            .and_then(|()| file.sync_all())
            .map_err(|error| format!("write sample copy record: {error}"))?;
        fs::rename(&temp, directory.join(COPY_RECORD_FILE))
            .map_err(|error| format!("publish sample copy record: {error}"))
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

fn read_copy_record(directory: &Path) -> Option<SampleCopyRecord> {
    let file = fs::File::open(directory.join(COPY_RECORD_FILE)).ok()?;
    let mut bytes = Vec::new();
    file.take(MAX_COPY_RECORD_BYTES + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() as u64 > MAX_COPY_RECORD_BYTES {
        return None;
    }
    let record: SampleCopyRecord = serde_json::from_slice(&bytes).ok()?;
    // The bundle must be one plain child of the copy's own directory.
    let mut components = Path::new(&record.bundle).components();
    let single =
        matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none();
    (single && record.bundle.ends_with(".opentake")).then_some(record)
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("encode {}: {error}", path.display()))?;
    fs::write(path, bytes).map_err(|error| format!("write {}: {error}", path.display()))
}

fn validate_slug(slug: &str) -> Result<(), String> {
    if slug.is_empty()
        || slug.len() > 80
        || !slug
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
    {
        return Err("sample slug must contain only ASCII letters, digits, '-' or '_'".into());
    }
    Ok(())
}

fn safe_name(value: &str) -> String {
    let clean = value
        .chars()
        .map(|character| match character {
            '/' | '\\' | ':' => ' ',
            other => other,
        })
        .collect::<String>();
    let trimmed = clean.trim();
    if trimmed.is_empty() {
        "Sample".into()
    } else {
        trimmed.chars().take(120).collect()
    }
}

fn validate_downloads(downloads: &[SampleDownload]) -> Result<(), String> {
    let mut paths = HashSet::new();
    for download in downloads {
        validated_network_url(&download.url)?;
        let normalized = Path::new(&download.relative_path);
        if normalized.as_os_str().is_empty()
            || normalized
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err(format!("unsafe sample path: {}", download.relative_path));
        }
        if !paths.insert(download.relative_path.clone()) {
            return Err(format!("duplicate sample path: {}", download.relative_path));
        }
    }
    Ok(())
}

fn safe_target(root: &Path, relative: &str) -> Result<PathBuf, String> {
    let path = Path::new(relative);
    if path
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!("unsafe sample path: {relative}"));
    }
    Ok(root.join(path))
}

fn sample_redirect_policy() -> Policy {
    Policy::custom(|attempt| {
        if attempt.previous().len() > 5 {
            return attempt.error("sample redirect limit exceeded");
        }
        let (Some(initial), Some(previous)) =
            (attempt.previous().first(), attempt.previous().last())
        else {
            return attempt.error("sample redirect has no initial URL");
        };
        if can_follow_sample_redirect(initial, previous, attempt.url()) {
            attempt.follow()
        } else {
            attempt.error("sample redirect must preserve HTTPS and the authorized loopback origin")
        }
    })
}

fn is_loopback_url(url: &reqwest::Url) -> bool {
    url.host_str().is_some_and(|host| {
        host.trim_end_matches('.').eq_ignore_ascii_case("localhost")
            || host
                .trim_start_matches('[')
                .trim_end_matches(']')
                .parse::<std::net::IpAddr>()
                .is_ok_and(|address| match address {
                    std::net::IpAddr::V4(address) => address.is_loopback(),
                    std::net::IpAddr::V6(address) => {
                        address.is_loopback()
                            || address
                                .to_ipv4_mapped()
                                .is_some_and(|address| address.is_loopback())
                    }
                })
    })
}

fn sample_network_url_allowed(url: &reqwest::Url) -> bool {
    url.host_str().is_some()
        && url.username().is_empty()
        && url.password().is_none()
        && (url.scheme() == "https" || (url.scheme() == "http" && is_loopback_url(url)))
}

fn can_follow_sample_redirect(
    initial: &reqwest::Url,
    previous: &reqwest::Url,
    next: &reqwest::Url,
) -> bool {
    sample_network_url_allowed(initial)
        && sample_network_url_allowed(next)
        && !(previous.scheme() == "https" && next.scheme() != "https")
        && (!is_loopback_url(next) || initial.origin() == next.origin())
}

fn validated_network_url(raw: &str) -> Result<reqwest::Url, String> {
    let url = reqwest::Url::parse(raw).map_err(|error| format!("invalid sample URL: {error}"))?;
    if !sample_network_url_allowed(&url) {
        return Err("sample URL must use HTTPS or loopback HTTP without credentials".into());
    }
    Ok(url)
}

fn read_bounded_response(mut response: Response, max: u64, label: &str) -> Result<Vec<u8>, String> {
    validate_response(&response, max, label)?;
    let mut bytes = Vec::new();
    response
        .by_ref()
        .take(max + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read {label}: {error}"))?;
    if bytes.len() as u64 > max {
        return Err(format!("{label}: response exceeds {max} bytes"));
    }
    Ok(bytes)
}

fn validate_response(response: &Response, max: u64, label: &str) -> Result<(), String> {
    validated_network_url(response.url().as_str())?;
    if !response.status().is_success() {
        return Err(format!(
            "{label}: server returned HTTP {}",
            response.status()
        ));
    }
    if response.content_length().is_some_and(|length| length > max) {
        return Err(format!("{label}: response exceeds {max} bytes"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentake_domain::{MediaManifest, Timeline};

    #[test]
    fn redirect_policy_preserves_https_credentials_and_loopback_authority() {
        for (initial, next, allowed) in [
            ("https://samples.example/a", "https://cdn.example/b", true),
            ("https://samples.example/a", "http://plain.example/b", false),
            ("https://samples.example/a", "http://127.0.0.1/b", false),
            ("https://samples.example/a", "https://localhost/b", false),
            (
                "https://samples.example/a",
                "https://user:secret@cdn.example/b",
                false,
            ),
            ("http://127.0.0.1:8000/a", "http://127.0.0.1:8000/b", true),
            ("http://127.0.0.1:8000/a", "http://127.0.0.1:8001/b", false),
        ] {
            assert_eq!(
                can_follow_sample_redirect(
                    &reqwest::Url::parse(initial).unwrap(),
                    &reqwest::Url::parse(initial).unwrap(),
                    &reqwest::Url::parse(next).unwrap()
                ),
                allowed,
                "{initial} -> {next}"
            );
        }
        assert!(validated_network_url("http://[::ffff:127.0.0.1]/a").is_ok());
        assert!(validated_network_url("http://localhost./a").is_ok());
        assert!(validated_network_url("http://remote.example/a").is_err());
        assert!(!can_follow_sample_redirect(
            &reqwest::Url::parse("http://127.0.0.1:8000/start").unwrap(),
            &reqwest::Url::parse("https://cdn.example/file").unwrap(),
            &reqwest::Url::parse("http://127.0.0.1:8000/finish").unwrap(),
        ));
    }

    #[test]
    fn opening_a_sample_again_preserves_the_edited_current_copy() {
        let storage = tempfile::tempdir().unwrap();
        let service = SampleProjectService::new(storage.path().to_path_buf()).unwrap();
        let current = service
            .materialize_builtin("template-project", |_| false, |_| {})
            .unwrap();
        let timeline_path = current.join(layout::TIMELINE_FILE);
        let mut timeline: Timeline =
            serde_json::from_slice(&fs::read(&timeline_path).unwrap()).unwrap();
        timeline.width = 1300;
        write_json(&timeline_path, &serde_json::to_value(timeline).unwrap()).unwrap();
        let saved = fs::read(&timeline_path).unwrap();
        let fresh = service
            .materialize_builtin("template-project", |_| false, |_| {})
            .unwrap();
        assert_ne!(current, fresh);
        assert_eq!(fs::read(&timeline_path).unwrap(), saved);
        Project::open(&current).unwrap();
        Project::open(&fresh).unwrap();
        // The unedited fresh copy is the one reused from now on.
        let again = service
            .materialize_builtin("template-project", |_| false, |_| {})
            .unwrap();
        assert_eq!(again, fresh);
        assert_eq!(fs::read(&timeline_path).unwrap(), saved);
    }

    #[test]
    fn unmodified_copy_is_reused_after_an_open_and_save() {
        let storage = tempfile::tempdir().unwrap();
        let service = SampleProjectService::new(storage.path().to_path_buf()).unwrap();
        let first = service
            .materialize_builtin("quick-tutorial", |_| false, |_| {})
            .unwrap();
        // Opening and saving without an edit re-encodes the documents and
        // writes the cover, which must not count as a user edit. (This module
        // is also compiled into an opentake-project test, so it saves through
        // the project layer that the core's save uses.)
        let mut project = Project::open(&first).unwrap();
        project.thumbnail = Some(b"regenerated cover".to_vec());
        project.save().unwrap();
        let mut progress = Vec::new();
        let second = service
            .materialize_builtin("quick-tutorial", |_| false, |value| progress.push(value))
            .unwrap();
        assert_eq!(second, first);
        assert_eq!(progress, vec![0.0, 1.0]);
        assert_eq!(fs::read_dir(storage.path()).unwrap().count(), 1);

        // A different sample never reuses this copy, nor does the open project.
        let other = service
            .materialize_builtin("product-demo", |_| false, |_| {})
            .unwrap();
        assert_ne!(other, first);
        let in_use = service
            .materialize_builtin("quick-tutorial", |bundle| bundle == first, |_| {})
            .unwrap();
        assert_ne!(in_use, first);
        assert_eq!(fs::read_dir(storage.path()).unwrap().count(), 3);
    }

    #[test]
    fn copy_with_changed_media_or_record_is_not_reused() {
        let storage = tempfile::tempdir().unwrap();
        let service = SampleProjectService::new(storage.path().to_path_buf()).unwrap();
        let sample = ResolvedSample {
            title: "Media sample".into(),
            project: serde_json::to_value(Timeline::new()).unwrap(),
            manifest: serde_json::to_value(MediaManifest::new()).unwrap(),
            generation_log: None,
            poster_url: None,
            downloads: vec![SampleDownload {
                id: "clip".into(),
                relative_path: "media/clip.bin".into(),
                url: "https://samples.example/clip".into(),
            }],
            chat: vec![],
        };
        let materialize = |downloads: &mut usize| {
            service
                .materialize_resolved(
                    "media-sample",
                    sample.clone(),
                    |_, target| {
                        *downloads += 1;
                        fs::write(target, b"sample bytes").map_err(|error| error.to_string())
                    },
                    |_| false,
                    |_| {},
                )
                .unwrap()
        };
        let mut downloads = 0;
        let first = materialize(&mut downloads);
        assert_eq!(materialize(&mut downloads), first);
        assert_eq!(downloads, 1, "a reused copy downloads nothing");

        // Same size, same name, different bytes: a user edit, never reused.
        let media = first.join("media/clip.bin");
        fs::write(&media, b"edited bytes").unwrap();
        let second = materialize(&mut downloads);
        assert_ne!(second, first);
        assert_eq!(fs::read(&media).unwrap(), b"edited bytes");

        // A new file inside the bundle (a chat session, say) is an edit too.
        fs::write(second.join("notes.txt"), b"mine").unwrap();
        let third = materialize(&mut downloads);
        assert_ne!(third, second);
        assert!(second.join("notes.txt").is_file());

        // A record naming a bundle outside its own directory is ignored.
        let record_path = third.parent().unwrap().join(COPY_RECORD_FILE);
        let mut record: SampleCopyRecord =
            serde_json::from_slice(&fs::read(&record_path).unwrap()).unwrap();
        record.bundle = "../elsewhere.opentake".into();
        fs::write(&record_path, serde_json::to_vec(&record).unwrap()).unwrap();
        let fourth = materialize(&mut downloads);
        assert_ne!(fourth, third);
        assert_eq!(downloads, 4);
        assert_eq!(materialize(&mut downloads), fourth);
    }

    fn serve_once(
        listener: std::net::TcpListener,
        response: String,
    ) -> std::thread::JoinHandle<bool> {
        listener.set_nonblocking(true).unwrap();
        std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            while std::time::Instant::now() < deadline {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        stream
                            .set_read_timeout(Some(Duration::from_secs(1)))
                            .unwrap();
                        let mut request = [0; 4096];
                        assert!(stream.read(&mut request).unwrap() > 0);
                        stream.write_all(response.as_bytes()).unwrap();
                        return true;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(error) => panic!("accept fixture request: {error}"),
                }
            }
            false
        })
    }

    fn local_client() -> Client {
        Client::builder()
            .no_proxy()
            .redirect(sample_redirect_policy())
            .timeout(Duration::from_secs(2))
            .build()
            .unwrap()
    }

    #[test]
    fn redirect_to_another_loopback_origin_is_rejected_before_connecting() {
        let source = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let source_address = source.local_addr().unwrap();
        let target = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let target_address = target.local_addr().unwrap();
        let redirect = serve_once(source, format!(
            "HTTP/1.1 302 Found\r\nLocation: http://{target_address}/other-service\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        ));
        let destination = serve_once(
            target,
            "HTTP/1.1 200 OK\r\nContent-Length: 6\r\nConnection: close\r\n\r\nunsafe".into(),
        );
        let storage = tempfile::tempdir().unwrap();
        let mut service = SampleProjectService::new(storage.path().into()).unwrap();
        service.client = local_client();
        let output = storage.path().join("output.bin");
        let result = service.download_file(
            &SampleDownload {
                id: "fixture".into(),
                relative_path: "output.bin".into(),
                url: format!("http://{source_address}/download"),
            },
            &output,
        );
        assert!(
            redirect.join().unwrap(),
            "fixture must receive the first request"
        );
        let contacted_destination = destination.join().unwrap();
        assert!(result.is_err(), "unexpected redirect success: {result:?}");
        assert!(
            !contacted_destination,
            "redirect contacted another local service"
        );
        assert!(!output.exists());
        assert_eq!(fs::read_dir(storage.path()).unwrap().count(), 0);
    }

    #[test]
    fn local_metadata_endpoint_materializes_a_complete_project() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let metadata = serde_json::json!({"title":"Remote sample", "project":Timeline::new(), "manifest":MediaManifest::new()}).to_string();
        let server = serve_once(
            listener,
            format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{metadata}",
                metadata.len()
            ),
        );
        let storage = tempfile::tempdir().unwrap();
        let mut service = SampleProjectService::new(storage.path().into()).unwrap();
        service.client = local_client();
        let result = service.materialize(&format!("http://{address}"), "remote", |_| false, |_| {});
        assert!(server.join().unwrap());
        Project::open(result.unwrap()).unwrap();
    }

    #[test]
    fn metadata_cannot_direct_downloads_to_another_local_service() {
        let source = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let source_address = source.local_addr().unwrap();
        let target = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let target_address = target.local_addr().unwrap();
        let metadata = serde_json::json!({
            "title": "Unsafe sample",
            "project": Timeline::new(),
            "manifest": MediaManifest::new(),
            "downloads": [{
                "id": "fixture",
                "relativePath": "media/fixture.bin",
                "url": format!("http://{target_address}/other-service"),
            }],
        })
        .to_string();
        let resolver = serve_once(
            source,
            format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{metadata}",
                metadata.len()
            ),
        );
        let destination = serve_once(
            target,
            "HTTP/1.1 200 OK\r\nContent-Length: 6\r\nConnection: close\r\n\r\nunsafe".into(),
        );
        let storage = tempfile::tempdir().unwrap();
        let mut service = SampleProjectService::new(storage.path().into()).unwrap();
        service.client = local_client();

        let result = service.materialize(
            &format!("http://{source_address}"),
            "unsafe",
            |_| false,
            |_| {},
        );

        assert!(resolver.join().unwrap());
        let contacted_destination = destination.join().unwrap();
        assert!(result.is_err());
        assert!(!contacted_destination);
        assert_eq!(fs::read_dir(storage.path()).unwrap().count(), 0);
    }

    #[test]
    fn failed_materialization_rolls_back_entire_sample_directory() {
        let cache = tempfile::tempdir().unwrap();
        let service = SampleProjectService::new(cache.path().to_path_buf()).unwrap();
        let sample = ResolvedSample {
            title: "Rollback demo".into(),
            project: serde_json::to_value(Timeline::new()).unwrap(),
            manifest: serde_json::to_value(MediaManifest::new()).unwrap(),
            generation_log: None,
            poster_url: None,
            downloads: vec![
                SampleDownload {
                    id: "first".into(),
                    relative_path: "media/first.bin".into(),
                    url: "https://samples.example/first".into(),
                },
                SampleDownload {
                    id: "broken".into(),
                    relative_path: "media/broken.bin".into(),
                    url: "https://samples.example/broken".into(),
                },
            ],
            chat: vec![],
        };

        let error = service
            .materialize_resolved(
                "rollback-demo",
                sample,
                |download, target| {
                    if download.id == "broken" {
                        return Err("fixture download failed".into());
                    }
                    fs::write(target, b"complete bytes").map_err(|error| error.to_string())
                },
                |_| false,
                |_| {},
            )
            .unwrap_err();

        assert!(error.contains("fixture download failed"), "{error}");
        assert!(!cache.path().join("rollback-demo").exists());
        assert_eq!(fs::read_dir(cache.path()).unwrap().count(), 0);
    }

    #[test]
    fn successful_materialization_publishes_a_valid_bundle_and_completes_progress() {
        let cache = tempfile::tempdir().unwrap();
        let service = SampleProjectService::new(cache.path().to_path_buf()).unwrap();
        let sample = ResolvedSample {
            title: "Starter / sample".into(),
            project: serde_json::to_value(Timeline::new()).unwrap(),
            manifest: serde_json::to_value(MediaManifest::new()).unwrap(),
            generation_log: None,
            poster_url: None,
            downloads: vec![],
            chat: vec![],
        };
        let mut progress = Vec::new();

        let bundle = service
            .materialize_resolved(
                "starter",
                sample,
                |_, _| panic!("empty sample must not download"),
                |_| false,
                |value| progress.push(value),
            )
            .unwrap();

        assert_eq!(progress, vec![0.0, 1.0]);
        assert_eq!(
            bundle.file_name().and_then(|name| name.to_str()),
            Some("Starter   sample.opentake")
        );
        Project::open(bundle).unwrap();
        assert_eq!(fs::read_dir(cache.path()).unwrap().count(), 1);
    }

    #[test]
    fn built_in_tutorial_is_offline_and_contains_editing_steps() {
        let cache = tempfile::tempdir().unwrap();
        let service = SampleProjectService::new(cache.path().to_path_buf()).unwrap();

        let bundle = service
            .materialize_builtin("quick-tutorial", |_| false, |_| {})
            .unwrap();
        let project = Project::open(bundle).unwrap();

        let text = project.timeline.tracks[0]
            .clips
            .iter()
            .filter_map(|clip| clip.text_content.as_deref())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(text.contains("Import media"));
        assert!(text.contains("Press Space"));
        assert_eq!(project.timeline.total_frames(), 360);
    }
}
