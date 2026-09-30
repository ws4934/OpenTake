//! Video encoding back end for `opentake-render`'s export path. The wgpu
//! compositor produces RGBA frames; this encoder pipes them to the system ffmpeg
//! CLI and muxes them (with an optional audio track) into a container.
//!
//! `opentake-render` decides the (even) frame size, applies BT.709 instructions,
//! and resolves keyframe ramps; this crate only encodes already-composited
//! frames (SPEC §2.4 / §8.2). The arg builder ([`encode_args`]) is pure and
//! unit-tested; the encode itself requires ffmpeg.

pub mod mix;
pub mod preset;

pub use mix::{mix_clips, mono_f32_to_s16le, ClipAudio, MIX_SAMPLE_RATE};
pub use preset::{even_dimension, ExportPreset, ExportResolution, VideoCodec};

use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{ChildStderr, ChildStdout, ExitStatus};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::cancel::MediaCancelToken;
use crate::decode::pcm::{PcmBuffer, PcmFormat, PcmSpec};
use crate::error::{MediaError, Result};
use crate::ff::SpawnCounted;
use crate::frame::RgbaFrame;

/// Build the ffmpeg arg list for encoding a raw-RGBA frame stream (read from
/// stdin) to `out` with `preset`. Pure so the CLI contract is testable.
///
/// Layout: `-f rawvideo -pix_fmt rgba -s {w}x{h} -r {fps} -i -` for video,
/// followed by codec/pixfmt/color args, then `out`.
fn encode_args(out: &Path, w: u32, h: u32, fps: i32, preset: &ExportPreset) -> Vec<OsString> {
    let mut args: Vec<OsString> = Vec::new();
    args.push("-y".into()); // overwrite
                            // Raw video input from stdin.
    args.push("-f".into());
    args.push("rawvideo".into());
    args.push("-pix_fmt".into());
    args.push("rgba".into());
    args.push("-s".into());
    args.push(format!("{w}x{h}").into());
    args.push("-r".into());
    args.push(fps.to_string().into());
    args.push("-i".into());
    args.push("-".into());

    // Video codec + pixel format.
    args.push("-c:v".into());
    args.push(preset.vcodec_arg().into());
    args.push("-pix_fmt".into());
    args.push(preset.pix_fmt_arg().into());
    if preset.codec == VideoCodec::ProRes4444 {
        args.push("-profile:v".into());
        args.push("4444".into());
    }
    args.extend(preset.color_args().into_iter().map(OsString::from));

    args.push(out.as_os_str().to_owned());
    args
}

/// Build the ffmpeg arg list for the second mux pass: take the already-encoded
/// (audio-less) video at `video_in` and a raw mono `s16le` PCM stream at
/// `pcm_in`, copy the video stream untouched, encode the audio with `acodec`,
/// and write the muxed container to `out`. Pure so the CLI contract is testable.
///
/// The video's length is authoritative. `apad` pads audio that ends early
/// with silence and `-t` (the encoded frames' duration) trims a longer tail,
/// so the output is never cut to the audio's length nor extended past the
/// last frame. `-shortest` cannot be used for this: it cut the video to short
/// audio, and combined with `apad` and a copied video stream it never ends.
/// Without video frames the audio is muxed as-is.
fn mux_args(
    video_in: &Path,
    pcm_in: &Path,
    out: &Path,
    sample_rate: u32,
    acodec: &str,
    video_duration: Option<&str>,
) -> Vec<OsString> {
    let mut args: Vec<OsString> = vec![
        "-y".into(),
        // Input 0: the encoded video (audio-less).
        "-i".into(),
        video_in.as_os_str().to_owned(),
        // Input 1: raw mono s16le PCM (the mixed audio).
        "-f".into(),
        "s16le".into(),
        "-ar".into(),
        sample_rate.to_string().into(),
        "-ac".into(),
        "1".into(),
        "-i".into(),
        pcm_in.as_os_str().to_owned(),
        // Copy the video stream verbatim; (re-)encode the audio.
        "-c:v".into(),
        "copy".into(),
        "-c:a".into(),
        acodec.into(),
    ];
    if let Some(duration) = video_duration {
        args.extend(["-af".into(), "apad".into(), "-t".into(), duration.into()]);
    }
    args.push(out.as_os_str().to_owned());
    args
}

/// Exact duration of `frames` at `fps`, rounded up to the microsecond so a
/// `-t` bound never drops the last frame. `None` without frames or rate.
fn video_duration_arg(frames: u64, fps: i32) -> Option<String> {
    let fps = u64::try_from(fps).ok().filter(|fps| *fps > 0)?;
    if frames == 0 {
        return None;
    }
    let micros = u128::from(frames)
        .checked_mul(1_000_000)?
        .div_ceil(u128::from(fps));
    Some(format!("{}.{:06}", micros / 1_000_000, micros % 1_000_000))
}

const ENCODE_POLL_INTERVAL: Duration = Duration::from_millis(5);
const OUTPUT_COPY_CHUNK: usize = 64 * 1024;
const ENCODE_PROGRESS_TOTAL: usize = 1_000;
const FIRST_PASS_END: usize = 100;
const PCM_WRITE_END: usize = 700;
const MUX_WAIT_START: usize = 800;
const MUX_COPY_START: usize = 900;
/// Bytes of FFmpeg's stderr kept for an encode or mux error message.
const STDERR_TAIL_BYTES: usize = 4 * 1024;
/// Lines of that tail quoted in the error message.
const STDERR_TAIL_LINES: usize = 12;
/// Name prefix of the private encode workspace next to the output.
pub const ENCODE_WORKSPACE_PREFIX: &str = ".opentake-encode-";

pub type EncodeProgressCallback = dyn Fn(usize, usize);

/// A streaming RGBA → video encoder. FFmpeg writes only inside a private
/// private workspace and never reopens the final pathname. The finished file is either
/// copied into the caller's retained output file ([`VideoEncoder::finish`])
/// or handed over in place for the caller to publish by rename
/// ([`VideoEncoder::finish_in_workspace`]).
pub struct VideoEncoder {
    child: ffmpeg_sidecar::child::FfmpegChild,
    stdin: Option<std::process::ChildStdin>,
    output_pump: Option<JoinHandle<Result<()>>>,
    stderr_pump: Option<JoinHandle<Result<Vec<u8>>>>,
    /// The end of the first pass's stderr, once its pump has been joined.
    stderr_tail: Vec<u8>,
    expected_frame_bytes: usize,
    fps: i32,
    /// Frames fully written to the first pass; they fix the output duration.
    frames_written: u64,
    /// `None` only after [`VideoEncoder::finish_in_workspace`] handed it over.
    workspace: Option<EncodeWorkspace>,
    first_pass: PathBuf,
    /// `None` for an encoder whose result is published from the workspace.
    output: Option<File>,
    acodec: &'static str,
    pending_audio: Option<PendingAudio>,
    child_reaped: bool,
}

/// Cleanup uses retained directory handles so a replaced pathname is left alone.
struct EncodeWorkspace {
    /// Retained no-follow handle of the directory.
    directory: File,
    path: PathBuf,
    #[cfg(unix)]
    parent: File,
}

impl EncodeWorkspace {
    fn next_to(out_hint: &Path) -> Result<Self> {
        let parent = match out_hint.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent,
            _ => Path::new("."),
        };
        Self::in_directory(parent)
    }

    fn in_directory(parent: &Path) -> Result<Self> {
        #[cfg(unix)]
        let parent_handle = open_directory_nofollow(parent)?;
        let mut builder = tempfile::Builder::new();
        builder.prefix(ENCODE_WORKSPACE_PREFIX);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            builder.permissions(std::fs::Permissions::from_mode(0o700));
        }
        // Exclusive mkdir establishes ownership; mount-specific uid mappings
        // and chmod support must not decide whether encoding is permitted.
        // Disable TempDir's recursive path cleanup before retaining the handle.
        let path = builder.tempdir_in(parent).map_err(MediaError::Io)?.keep();
        let directory = open_directory_nofollow(&path)?;
        let workspace = Self {
            directory,
            path,
            #[cfg(unix)]
            parent: parent_handle,
        };
        // Visibility is cosmetic: a volume that rejects DOS attributes must
        // not make an otherwise valid export fail or weaken its retained lease.
        #[cfg(windows)]
        if let Err(error) = workspace.hide_directory() {
            tracing::warn!(%error, "could not hide the encode workspace");
        }
        Ok(workspace)
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn remove_file(&self, name: &std::ffi::OsStr) -> std::io::Result<()> {
        cap_std::fs::Dir::from_std_file(self.directory.try_clone()?).remove_file(name)
    }

    fn cleanup(&self) -> std::io::Result<()> {
        // FFmpeg creates only these leaves. Never recurse into an unexpected
        // directory or resolve the ambient workspace path during cleanup.
        for name in [
            "video.mp4",
            "video.mov",
            "audio.pcm",
            "muxed.mp4",
            "muxed.mov",
        ] {
            match self.remove_file(std::ffi::OsStr::new(name)) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error),
            }
        }
        self.remove_directory()
    }
}

impl Drop for EncodeWorkspace {
    fn drop(&mut self) {
        if let Err(error) = self.cleanup() {
            tracing::warn!(path = %self.path.display(), %error, "failed to clean encode workspace");
        }
    }
}

#[cfg(unix)]
impl EncodeWorkspace {
    fn remove_directory(&self) -> std::io::Result<()> {
        use std::ffi::CString;
        use std::os::fd::AsRawFd;
        use std::os::unix::ffi::OsStrExt;

        let name = CString::new(self.path.file_name().expect("workspace leaf").as_bytes())?;
        let mut visible = std::mem::MaybeUninit::<libc::stat>::uninit();
        // SAFETY: the retained parent descriptor is live, and both buffers are
        // valid. AT_SYMLINK_NOFOLLOW checks the named object itself.
        if unsafe {
            libc::fstatat(
                self.parent.as_raw_fd(),
                name.as_ptr(),
                visible.as_mut_ptr(),
                libc::AT_SYMLINK_NOFOLLOW,
            )
        } < 0
        {
            let error = std::io::Error::last_os_error();
            return if error.kind() == std::io::ErrorKind::NotFound {
                Ok(())
            } else {
                Err(error)
            };
        }
        // SAFETY: successful fstatat initialized the complete stat buffer.
        let visible = unsafe { visible.assume_init() };
        let mut retained = std::mem::MaybeUninit::<libc::stat>::uninit();
        // SAFETY: the descriptor and output buffer stay valid for the call.
        if unsafe { libc::fstat(self.directory.as_raw_fd(), retained.as_mut_ptr()) } < 0 {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: successful fstat initialized the complete stat buffer.
        let retained = unsafe { retained.assume_init() };
        if visible.st_dev != retained.st_dev || visible.st_ino != retained.st_ino {
            return Ok(());
        }
        // SAFETY: the parent is retained and the single-component name still
        // identifies the owned directory. Only an empty directory is removed.
        if unsafe { libc::unlinkat(self.parent.as_raw_fd(), name.as_ptr(), libc::AT_REMOVEDIR) } < 0
        {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }
}

#[cfg(windows)]
impl EncodeWorkspace {
    fn hide_directory(&self) -> std::io::Result<()> {
        use std::os::windows::fs::MetadataExt;
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Storage::FileSystem::{
            FileBasicInfo, SetFileInformationByHandle, FILE_ATTRIBUTE_HIDDEN,
            FILE_ATTRIBUTE_NORMAL, FILE_BASIC_INFO,
        };
        let attributes = self.directory.metadata()?.file_attributes();
        if attributes & FILE_ATTRIBUTE_HIDDEN != 0 {
            return Ok(());
        }
        let info = FILE_BASIC_INFO {
            FileAttributes: (attributes & !FILE_ATTRIBUTE_NORMAL) | FILE_ATTRIBUTE_HIDDEN,
            // Zero timestamps preserve the existing values.
            ..Default::default()
        };
        // SAFETY: the live attribute handle and SDK buffer have the required
        // access, layout and lifetime for this synchronous call.
        if unsafe {
            SetFileInformationByHandle(
                self.directory.as_raw_handle(),
                FileBasicInfo,
                (&info as *const FILE_BASIC_INFO).cast(),
                std::mem::size_of::<FILE_BASIC_INFO>() as u32,
            )
        } == 0
        {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }

    fn remove_directory(&self) -> std::io::Result<()> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Storage::FileSystem::{
            FileDispositionInfo, SetFileInformationByHandle, FILE_DISPOSITION_INFO,
        };
        let info = FILE_DISPOSITION_INFO { DeleteFile: true };
        // SAFETY: the DELETE-capable directory handle and SDK buffer stay
        // valid for the call. Deletion completes when the retained handle closes.
        if unsafe {
            SetFileInformationByHandle(
                self.directory.as_raw_handle(),
                FileDispositionInfo,
                (&info as *const FILE_DISPOSITION_INFO).cast(),
                std::mem::size_of::<FILE_DISPOSITION_INFO>() as u32,
            )
        } == 0
        {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }
}

#[cfg(not(any(unix, windows)))]
impl EncodeWorkspace {
    fn remove_directory(&self) -> std::io::Result<()> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "retained encode directory cleanup is unsupported",
        ))
    }
}

/// A finished encode left in its workspace for the caller to publish with a
/// single rename, so no second copy of the movie is ever written. The
/// workspace, and the file if it is still there, are removed when this is
/// dropped; once the file has been renamed away only the empty workspace
/// remains to remove.
pub struct EncodedFile {
    /// Retained handle of the finished file: readable, writable and, on
    /// Windows, opened with `DELETE` access so it can be renamed and deleted
    /// through the handle.
    file: File,
    name: OsString,
    workspace: EncodeWorkspace,
}

impl EncodedFile {
    /// The retained handle of the finished file.
    pub fn file(&self) -> &File {
        &self.file
    }

    /// The retained no-follow handle of the workspace directory holding it.
    pub fn directory(&self) -> &File {
        &self.workspace.directory
    }

    /// Path of the workspace directory.
    pub fn directory_path(&self) -> &Path {
        self.workspace.path()
    }

    /// The file's single-component name inside the workspace.
    pub fn name(&self) -> &std::ffi::OsStr {
        &self.name
    }

    /// Full path of the finished file inside the workspace.
    pub fn path(&self) -> PathBuf {
        self.workspace.path().join(&self.name)
    }
}

struct PendingAudio {
    path: PathBuf,
    spec: PcmSpec,
    sample_count: u64,
}

impl VideoEncoder {
    /// Start an encoder writing to `out`. `w`/`h` must already be even.
    pub fn new(out: &Path, w: u32, h: u32, fps: i32, preset: &ExportPreset) -> Result<Self> {
        let output = Self::open_output_file(out)?;
        Self::new_with_file(out, output, w, h, fps, preset)
    }

    /// Start an encoder whose result stays in a private workspace next to
    /// `out_hint` (same directory, so the same volume) until
    /// [`VideoEncoder::finish_in_workspace`] hands it over for publishing by
    /// rename. `out_hint` itself is never opened.
    pub fn new_in_workspace(
        out_hint: &Path,
        w: u32,
        h: u32,
        fps: i32,
        preset: &ExportPreset,
    ) -> Result<Self> {
        Self::start(out_hint, None, w, h, fps, preset)
    }

    /// Open and truncate a regular, non-link output file without following a
    /// symlink. Callers that need an identity-safe cleanup guard can retain
    /// this handle and pass a clone into `new_with_file`.
    pub fn open_output_file(out: &Path) -> Result<File> {
        reject_link_output(out)?;
        open_output_nofollow(out)
    }

    /// Start an encoder whose finished bytes [`VideoEncoder::finish`] copies
    /// into `output`, a retained handle to `out_hint`. Intermediates stay in
    /// system temporary storage, outside project bundles copied by Save As.
    pub fn new_with_file(
        out_hint: &Path,
        mut output: File,
        w: u32,
        h: u32,
        fps: i32,
        preset: &ExportPreset,
    ) -> Result<Self> {
        output.set_len(0).map_err(MediaError::Io)?;
        output.seek(SeekFrom::Start(0)).map_err(MediaError::Io)?;
        Self::start(out_hint, Some(output), w, h, fps, preset)
    }

    fn start(
        out_hint: &Path,
        output: Option<File>,
        w: u32,
        h: u32,
        fps: i32,
        preset: &ExportPreset,
    ) -> Result<Self> {
        let workspace = if output.is_some() {
            EncodeWorkspace::in_directory(&std::env::temp_dir())?
        } else {
            EncodeWorkspace::next_to(out_hint)?
        };
        let extension = if matches!(preset.codec, VideoCodec::ProRes422 | VideoCodec::ProRes4444) {
            "mov"
        } else {
            "mp4"
        };
        let first_pass = workspace.path().join(format!("video.{extension}"));
        // Private leaves are ASCII; the native working directory avoids
        // Windows FFmpeg's lossy argv conversion for parent directories.
        let mut command = ffmpeg_sidecar::command::FfmpegCommand::new_with_path(
            crate::ff::ffmpeg_workspace_path()?,
        );
        command
            .args(encode_args(
                Path::new(&format!("video.{extension}")),
                w,
                h,
                fps,
                preset,
            ))
            .as_inner_mut()
            .current_dir(workspace.path());
        let mut child = command
            .spawn_counted()
            .map_err(|e| MediaError::Encode(format!("spawn: {e}")))?;
        let stdin = child.take_stdin();
        let stdout = child.take_stdout().ok_or_else(|| {
            terminate_child(&mut child);
            MediaError::Encode("encoder stdout pipe missing".to_string())
        })?;
        let stderr = child.take_stderr().ok_or_else(|| {
            terminate_child(&mut child);
            MediaError::Encode("encoder stderr pipe missing".to_string())
        })?;
        let output_pump = match thread::Builder::new()
            .name("opentake-encoder-stdout".to_string())
            .spawn(move || drain_stdout(stdout))
        {
            Ok(pump) => pump,
            Err(error) => {
                terminate_child(&mut child);
                return Err(MediaError::Encode(format!(
                    "spawn encoder output pump for {}: {error}",
                    out_hint.display()
                )));
            }
        };
        let stderr_pump = match thread::Builder::new()
            .name("opentake-encoder-stderr".to_string())
            .spawn(move || drain_stderr(stderr))
        {
            Ok(pump) => pump,
            Err(error) => {
                terminate_child(&mut child);
                let _ = join_named_pump(output_pump, "encoder output");
                return Err(MediaError::Encode(format!(
                    "spawn encoder stderr pump: {error}"
                )));
            }
        };
        Ok(VideoEncoder {
            child,
            stdin,
            output_pump: Some(output_pump),
            stderr_pump: Some(stderr_pump),
            stderr_tail: Vec::new(),
            expected_frame_bytes: w as usize * h as usize * 4,
            fps,
            frames_written: 0,
            workspace: Some(workspace),
            first_pass,
            output,
            acodec: preset.acodec_arg(),
            pending_audio: None,
            child_reaped: false,
        })
    }

    /// Push one composited frame. The frame's byte length must match the
    /// encoder's configured dimensions.
    pub fn push_frame(&mut self, rgba: &RgbaFrame) -> Result<()> {
        if rgba.rgba.len() != self.expected_frame_bytes {
            return Err(MediaError::Encode(format!(
                "frame size mismatch: got {} bytes, expected {}",
                rgba.rgba.len(),
                self.expected_frame_bytes
            )));
        }
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| MediaError::Encode("encoder stdin closed".into()))?;
        if let Err(error) = stdin.write_all(&rgba.rgba) {
            // A closed pipe usually means FFmpeg rejected its arguments or
            // input and exited; its stderr says why.
            self.reap_child();
            return Err(MediaError::Encode(with_stderr_tail(
                format!("write frame: {error}"),
                &self.stderr_tail,
            )));
        }
        self.frames_written += 1;
        Ok(())
    }

    /// Record one complete mixed-down mono audio buffer. Internally this uses
    /// the same file-backed chunk sink as long-timeline export, so the encoder
    /// never retains the caller's `Vec` until `finish`.
    pub fn push_audio(&mut self, pcm: PcmBuffer) -> Result<()> {
        if let Some(pending) = self.pending_audio.take() {
            let _ = std::fs::remove_file(pending.path);
        }
        if pcm.samples_f32.is_empty() {
            return Ok(());
        }
        self.push_audio_chunk(pcm.spec, &pcm.samples_f32, &MediaCancelToken::new())
    }

    /// Append one bounded mono f32 chunk to the private PCM spool used by the
    /// final mux. The spool is file-backed, cancellable, and requires every
    /// chunk to keep the same PCM contract.
    pub fn push_audio_chunk(
        &mut self,
        spec: PcmSpec,
        samples: &[f32],
        cancel: &MediaCancelToken,
    ) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }
        if spec.channels != 1 || spec.format != PcmFormat::F32 || spec.sample_rate == 0 {
            return Err(MediaError::Encode(
                "streamed audio must be mono f32 at a positive sample rate".to_string(),
            ));
        }
        let path = self.workspace_path()?.join("audio.pcm");
        let mut output = if let Some(pending) = &self.pending_audio {
            if pending.spec != spec {
                return Err(MediaError::Encode(
                    "streamed audio format changed between chunks".to_string(),
                ));
            }
            OpenOptions::new()
                .append(true)
                .open(&pending.path)
                .map_err(MediaError::Io)?
        } else {
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(MediaError::Io)?
        };
        for chunk in samples.chunks(OUTPUT_COPY_CHUNK / 2) {
            if cancel.checkpoint() {
                return Err(MediaError::Cancelled);
            }
            output
                .write_all(&mix::mono_f32_to_s16le(chunk))
                .map_err(MediaError::Io)?;
        }
        output.flush().map_err(MediaError::Io)?;
        let appended = u64::try_from(samples.len())
            .map_err(|_| MediaError::Encode("streamed audio sample count overflow".to_string()))?;
        match &mut self.pending_audio {
            Some(pending) => {
                pending.sample_count =
                    pending.sample_count.checked_add(appended).ok_or_else(|| {
                        MediaError::Encode("streamed audio sample count overflow".to_string())
                    })?;
            }
            None => {
                self.pending_audio = Some(PendingAudio {
                    path,
                    spec,
                    sample_count: appended,
                });
            }
        }
        Ok(())
    }

    /// Abort a mid-stream encode (e.g. a user cancel): kill the ffmpeg child and
    /// wait for it to exit, so the caller can safely remove the (now-closed)
    /// partial output file. `std::process::Child`'s own `Drop` does **not** kill
    /// or wait — a plain `drop(encoder)` would orphan the ffmpeg process, which
    /// could still be writing `out_path` at the moment the caller deletes it.
    /// Best-effort: the child may have already exited on its own.
    pub fn abort(mut self) {
        self.reap_child();
    }

    /// Finish an encoder started with [`VideoEncoder::new`] or
    /// [`VideoEncoder::new_with_file`]: its output file receives the result.
    pub fn finish(mut self) -> Result<()> {
        self.finish_into_output(&MediaCancelToken::new(), None, None)
    }

    pub fn finish_cancellable(
        mut self,
        cancel: &MediaCancelToken,
        progress: Option<&EncodeProgressCallback>,
    ) -> Result<()> {
        self.finish_into_output(cancel, progress, None)
    }

    /// Finish an encoder started with [`VideoEncoder::new_in_workspace`] and
    /// hand over the synced result in its workspace, without copying it.
    pub fn finish_in_workspace(
        mut self,
        cancel: &MediaCancelToken,
        progress: Option<&EncodeProgressCallback>,
    ) -> Result<EncodedFile> {
        if self.output.is_some() {
            return Err(MediaError::Encode(
                "an encoder with an output file finishes into that file".to_string(),
            ));
        }
        let progress = DedupedProgress::new(progress);
        let finished = self.finish_passes(cancel, &progress, None)?;
        let workspace = self
            .workspace
            .as_ref()
            .ok_or_else(|| MediaError::Encode("encode workspace is missing".to_string()))?;
        let name = finished
            .file_name()
            .ok_or_else(|| MediaError::Encode("encoded file has no name".to_string()))?
            .to_os_string();
        let file = open_encoded_file(&workspace.directory, workspace.path(), &name)?;
        file.sync_all().map_err(MediaError::Io)?;
        if cancel.checkpoint() {
            return Err(MediaError::Cancelled);
        }
        let workspace = self
            .workspace
            .take()
            .ok_or_else(|| MediaError::Encode("encode workspace is missing".to_string()))?;
        progress.report(ENCODE_PROGRESS_TOTAL);
        Ok(EncodedFile {
            file,
            name,
            workspace,
        })
    }

    fn finish_into_output(
        &mut self,
        cancel: &MediaCancelToken,
        progress: Option<&EncodeProgressCallback>,
        mux_wait_hook: Option<&dyn Fn()>,
    ) -> Result<()> {
        if self.output.is_none() {
            return Err(MediaError::Encode(
                "an encoder without an output file finishes in its workspace".to_string(),
            ));
        }
        let progress = DedupedProgress::new(progress);
        let finished = self.finish_passes(cancel, &progress, mux_wait_hook)?;
        let copy_start = if finished == self.first_pass {
            FIRST_PASS_END
        } else {
            MUX_COPY_START
        };
        self.copy_file_to_output(&finished, cancel, &progress, copy_start)?;
        if cancel.checkpoint() {
            return Err(MediaError::Cancelled);
        }
        let output = self.output_file()?;
        output.flush().map_err(MediaError::Io)?;
        output.sync_all().map_err(MediaError::Io)?;
        progress.report(ENCODE_PROGRESS_TOTAL);
        Ok(())
    }

    /// Run the first pass to completion and mux any audio. Returns the
    /// finished file in the workspace: the first pass, or the mux result
    /// (after the first pass and the PCM spool it replaces are removed).
    fn finish_passes(
        &mut self,
        cancel: &MediaCancelToken,
        progress: &DedupedProgress<'_>,
        mux_wait_hook: Option<&dyn Fn()>,
    ) -> Result<PathBuf> {
        let status = self.wait_for_child(cancel, progress)?;
        if !status.success() {
            return Err(MediaError::Encode(with_stderr_tail(
                format!("ffmpeg exited {status}"),
                &self.stderr_tail,
            )));
        }
        progress.report(FIRST_PASS_END);

        let finished = match self.pending_audio.take() {
            Some(audio) => {
                let muxed = self.mux_audio(&audio, cancel, progress, mux_wait_hook)?;
                // Free the intermediate files before the result is copied or
                // published, so the workspace holds one movie at a time.
                let workspace = self.workspace.as_ref().expect("encoder workspace");
                workspace
                    .remove_file(self.first_pass.file_name().expect("first-pass leaf"))
                    .map_err(MediaError::Io)?;
                workspace
                    .remove_file(audio.path.file_name().expect("audio leaf"))
                    .map_err(MediaError::Io)?;
                muxed
            }
            None => self.first_pass.clone(),
        };
        if cancel.checkpoint() {
            return Err(MediaError::Cancelled);
        }
        Ok(finished)
    }

    fn workspace_path(&self) -> Result<&Path> {
        self.workspace
            .as_ref()
            .map(EncodeWorkspace::path)
            .ok_or_else(|| MediaError::Encode("encode workspace is missing".to_string()))
    }

    fn output_file(&mut self) -> Result<&mut File> {
        self.output
            .as_mut()
            .ok_or_else(|| MediaError::Encode("encoder has no output file".to_string()))
    }

    fn wait_for_child(
        &mut self,
        cancel: &MediaCancelToken,
        progress: &DedupedProgress<'_>,
    ) -> Result<ExitStatus> {
        self.stdin.take();
        let mut polls = 0_usize;
        loop {
            if cancel.checkpoint() {
                terminate_child(&mut self.child);
                self.child_reaped = true;
                let _ = self.join_output_pump();
                return Err(MediaError::Cancelled);
            }
            match self.child.as_inner_mut().try_wait() {
                Ok(Some(status)) => {
                    self.child_reaped = true;
                    self.join_output_pump()?;
                    return Ok(status);
                }
                Ok(None) => {
                    polls = polls.saturating_add(1);
                    if polls.is_multiple_of(20) {
                        progress.report((polls / 20).min(FIRST_PASS_END - 1));
                    }
                    thread::sleep(ENCODE_POLL_INTERVAL);
                }
                Err(error) => {
                    self.reap_child();
                    return Err(MediaError::Io(error));
                }
            }
        }
    }

    fn join_output_pump(&mut self) -> Result<()> {
        let output = self
            .output_pump
            .take()
            .map(|pump| join_named_pump(pump, "encoder output"))
            .unwrap_or(Ok(()));
        let stderr = match self.stderr_pump.take() {
            Some(pump) => join_named_pump(pump, "encoder stderr").map(|tail| {
                self.stderr_tail = tail;
            }),
            None => Ok(()),
        };
        output.and(stderr)
    }

    fn reap_child(&mut self) {
        self.stdin.take();
        if !self.child_reaped {
            terminate_child(&mut self.child);
            self.child_reaped = true;
        }
        let _ = self.join_output_pump();
    }

    #[cfg(all(test, unix))]
    fn child_id(&mut self) -> u32 {
        self.child.as_inner_mut().id()
    }

    fn copy_file_to_output(
        &mut self,
        source_path: &Path,
        cancel: &MediaCancelToken,
        progress: &DedupedProgress<'_>,
        progress_start: usize,
    ) -> Result<()> {
        let mut source = File::open(source_path).map_err(MediaError::Io)?;
        source.seek(SeekFrom::Start(0)).map_err(MediaError::Io)?;
        let output = self.output_file()?;
        output.set_len(0).map_err(MediaError::Io)?;
        output.seek(SeekFrom::Start(0)).map_err(MediaError::Io)?;
        let total = source.metadata().map_err(MediaError::Io)?.len().max(1);
        let mut copied = 0_u64;
        let mut chunk = [0_u8; OUTPUT_COPY_CHUNK];
        loop {
            if cancel.checkpoint() {
                return Err(MediaError::Cancelled);
            }
            let read = source.read(&mut chunk).map_err(MediaError::Io)?;
            if read == 0 {
                break;
            }
            output.write_all(&chunk[..read]).map_err(MediaError::Io)?;
            copied = copied.saturating_add(read as u64);
            let mapped = progress_start
                + ((copied.min(total) * (ENCODE_PROGRESS_TOTAL - progress_start) as u64) / total)
                    as usize;
            progress.report(mapped.min(ENCODE_PROGRESS_TOTAL - 1));
        }
        Ok(())
    }

    fn mux_audio(
        &mut self,
        audio: &PendingAudio,
        cancel: &MediaCancelToken,
        progress: &DedupedProgress<'_>,
        mux_wait_hook: Option<&dyn Fn()>,
    ) -> Result<PathBuf> {
        progress.report(PCM_WRITE_END);

        let mux_path = self.workspace_path()?.join(
            if self
                .first_pass
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("mov"))
            {
                "muxed.mov"
            } else {
                "muxed.mp4"
            },
        );
        let video_duration = video_duration_arg(self.frames_written, self.fps);
        let args = mux_args(
            Path::new(self.first_pass.file_name().expect("private video leaf")),
            Path::new(audio.path.file_name().expect("private audio leaf")),
            Path::new(mux_path.file_name().expect("private mux leaf")),
            audio.spec.sample_rate,
            self.acodec,
            video_duration.as_deref(),
        );
        let mut command = ffmpeg_sidecar::command::FfmpegCommand::new_with_path(
            crate::ff::ffmpeg_workspace_path()?,
        );
        command
            .args(args)
            .as_inner_mut()
            .current_dir(self.workspace_path()?);
        let mut child = command
            .spawn_counted()
            .map_err(|e| MediaError::Encode(format!("mux spawn: {e}")))?;
        let stdout = child.take_stdout().ok_or_else(|| {
            terminate_child(&mut child);
            MediaError::Encode("mux stdout pipe missing".to_string())
        })?;
        let stderr = child.take_stderr().ok_or_else(|| {
            terminate_child(&mut child);
            MediaError::Encode("mux stderr pipe missing".to_string())
        })?;
        let pump = match thread::Builder::new()
            .name("opentake-mux-stdout".to_string())
            .spawn(move || drain_stdout(stdout))
        {
            Ok(pump) => pump,
            Err(error) => {
                terminate_child(&mut child);
                return Err(MediaError::Encode(format!(
                    "spawn mux output pump: {error}"
                )));
            }
        };
        let stderr_pump = match thread::Builder::new()
            .name("opentake-mux-stderr".to_string())
            .spawn(move || drain_stderr(stderr))
        {
            Ok(pump) => pump,
            Err(error) => {
                terminate_child(&mut child);
                let _ = join_named_pump(pump, "mux output");
                return Err(MediaError::Encode(format!(
                    "spawn mux stderr pump: {error}"
                )));
            }
        };
        progress.report(MUX_WAIT_START);
        let (status, stderr_tail) = wait_external_child(
            &mut child,
            pump,
            stderr_pump,
            cancel,
            progress,
            mux_wait_hook,
        )?;
        if !status.success() {
            return Err(MediaError::Encode(with_stderr_tail(
                format!("ffmpeg mux exited {status}"),
                &stderr_tail,
            )));
        }
        Ok(mux_path)
    }
}

fn reject_link_output(out: &Path) -> Result<()> {
    let metadata = match std::fs::symlink_metadata(out) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(MediaError::Io(error)),
    };
    #[cfg(not(windows))]
    let is_link = metadata.file_type().is_symlink();
    #[cfg(windows)]
    let is_link = {
        use std::os::windows::fs::MetadataExt;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
        metadata.file_type().is_symlink()
            || metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
    };
    if is_link {
        return Err(MediaError::Encode(
            "refusing to encode through a symlink or reparse point".to_string(),
        ));
    }
    if !metadata.is_file() {
        return Err(MediaError::Encode(
            "encoder output must be a regular file".to_string(),
        ));
    }
    Ok(())
}

fn open_output_nofollow(out: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }
    let file = options.open(out).map_err(MediaError::Io)?;
    let metadata = file.metadata().map_err(MediaError::Io)?;
    #[cfg(not(windows))]
    let is_link = metadata.file_type().is_symlink();
    #[cfg(windows)]
    let is_link = {
        use std::os::windows::fs::MetadataExt;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
        metadata.file_type().is_symlink()
            || metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
    };
    if is_link || !metadata.is_file() {
        return Err(MediaError::Encode(
            "encoder output must be a regular non-link file".to_string(),
        ));
    }
    Ok(file)
}

impl Drop for VideoEncoder {
    fn drop(&mut self) {
        self.reap_child();
    }
}

fn drain_stdout(mut stdout: ChildStdout) -> Result<()> {
    let mut bytes = [0_u8; OUTPUT_COPY_CHUNK];
    loop {
        let read = stdout.read(&mut bytes).map_err(MediaError::Io)?;
        if read == 0 {
            return Ok(());
        }
    }
}

/// Read stderr to its end, keeping only its last [`STDERR_TAIL_BYTES`].
fn drain_stderr(mut stderr: ChildStderr) -> Result<Vec<u8>> {
    let mut tail = std::collections::VecDeque::with_capacity(STDERR_TAIL_BYTES);
    let mut bytes = [0_u8; 8 * 1024];
    loop {
        let read = stderr.read(&mut bytes).map_err(MediaError::Io)?;
        if read == 0 {
            return Ok(tail.into());
        }
        let kept = &bytes[read.saturating_sub(STDERR_TAIL_BYTES)..read];
        let overflow = (tail.len() + kept.len()).saturating_sub(STDERR_TAIL_BYTES);
        tail.drain(..overflow);
        tail.extend(kept);
    }
}

/// `message`, followed by the last lines FFmpeg wrote to stderr, if any.
fn with_stderr_tail(message: String, stderr_tail: &[u8]) -> String {
    let text = String::from_utf8_lossy(stderr_tail);
    let lines: Vec<&str> = text
        .split(['\r', '\n'])
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    if lines.is_empty() {
        return message;
    }
    let diagnostic = |line: &str| {
        ["[error]", "[fatal]", "[warning]"]
            .iter()
            .any(|level| line.contains(level))
    };
    let mut quoted: Vec<&str> = lines
        .iter()
        .rev()
        .copied()
        .filter(|line| diagnostic(line))
        .take(STDERR_TAIL_LINES)
        .collect();
    quoted.reverse();
    let mut context: Vec<&str> = lines
        .iter()
        .rev()
        .copied()
        .filter(|line| !diagnostic(line))
        .take(STDERR_TAIL_LINES - quoted.len())
        .collect();
    context.reverse();
    quoted.extend(context);
    format!("{message}: {}", quoted.join(" | "))
}

/// Forwards encode progress, dropping repeats of the last reported value:
/// a large copy reports every 64 KiB, but only per-mille changes matter.
struct DedupedProgress<'a> {
    callback: Option<&'a EncodeProgressCallback>,
    last: std::cell::Cell<Option<usize>>,
}

impl<'a> DedupedProgress<'a> {
    fn new(callback: Option<&'a EncodeProgressCallback>) -> Self {
        DedupedProgress {
            callback,
            last: std::cell::Cell::new(None),
        }
    }

    fn report(&self, done: usize) {
        let done = done.min(ENCODE_PROGRESS_TOTAL);
        if self.last.replace(Some(done)) != Some(done) {
            report_progress(self.callback, done);
        }
    }
}

fn open_directory_nofollow(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_SHARE_READ: u32 = 0x1;
        const FILE_SHARE_WRITE: u32 = 0x2;
        const DELETE: u32 = 0x0001_0000;
        const GENERIC_READ: u32 = 0x8000_0000;
        const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        // Without delete sharing the workspace cannot be renamed or replaced
        // while it is retained. Attribute access lets this same retained handle
        // hide the directory without reopening it with different access rights.
        options
            .access_mode(GENERIC_READ | DELETE | FILE_WRITE_ATTRIBUTES)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
            .custom_flags(FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT);
    }
    let directory = options.open(path).map_err(MediaError::Io)?;
    let metadata = directory.metadata().map_err(MediaError::Io)?;
    if metadata_is_link(&metadata) || !metadata.is_dir() {
        return Err(MediaError::Encode(
            "encode workspace must be a real directory".to_string(),
        ));
    }
    Ok(directory)
}

/// Open the finished `name` inside the retained workspace without following
/// a link, with the access a caller needs to verify, rename and delete it.
#[cfg(unix)]
fn open_encoded_file(directory: &File, _path: &Path, name: &std::ffi::OsStr) -> Result<File> {
    use std::ffi::CString;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::ffi::OsStrExt;

    let name = CString::new(name.as_bytes())
        .map_err(|_| MediaError::Encode("encoded file name contains NUL".to_string()))?;
    // SAFETY: `directory` is a live descriptor and `name` a NUL-terminated
    // single component; a returned descriptor is owned by the File below.
    let descriptor = unsafe {
        libc::openat(
            directory.as_raw_fd(),
            name.as_ptr(),
            libc::O_RDWR | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if descriptor < 0 {
        return Err(MediaError::Io(std::io::Error::last_os_error()));
    }
    // SAFETY: `openat` returned a new descriptor that nothing else owns.
    let file = unsafe { File::from_raw_fd(descriptor) };
    ensure_regular_file(file)
}

#[cfg(not(unix))]
fn open_encoded_file(_directory: &File, path: &Path, name: &std::ffi::OsStr) -> Result<File> {
    let mut options = OpenOptions::new();
    options.read(true).write(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const DELETE: u32 = 0x0001_0000;
        const FILE_WRITE_ATTRIBUTES: u32 = 0x0000_0100;
        const GENERIC_READ: u32 = 0x8000_0000;
        const GENERIC_WRITE: u32 = 0x4000_0000;
        const FILE_SHARE_READ: u32 = 0x1;
        const FILE_SHARE_WRITE: u32 = 0x2;
        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        // No delete sharing: nothing else can rename or replace the file
        // while it is retained; the holder renames it through this handle.
        options
            .access_mode(GENERIC_READ | GENERIC_WRITE | DELETE)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
            .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }
    let file = options.open(path.join(name)).map_err(MediaError::Io)?;
    ensure_regular_file(file)
}

fn ensure_regular_file(file: File) -> Result<File> {
    let metadata = file.metadata().map_err(MediaError::Io)?;
    if metadata_is_link(&metadata) || !metadata.is_file() {
        return Err(MediaError::Encode(
            "encoded output must be a regular non-link file".to_string(),
        ));
    }
    Ok(file)
}

fn metadata_is_link(metadata: &std::fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
        metadata.file_type().is_symlink()
            || metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
    }
    #[cfg(not(windows))]
    {
        metadata.file_type().is_symlink()
    }
}

#[cfg(test)]
fn write_pcm_s16le_cancellable(
    samples: &[f32],
    destination: &mut File,
    cancel: &MediaCancelToken,
    progress: Option<&EncodeProgressCallback>,
    checkpoint_hook: Option<&dyn Fn(usize)>,
) -> Result<()> {
    let progress = DedupedProgress::new(progress);
    destination.set_len(0).map_err(MediaError::Io)?;
    destination
        .seek(SeekFrom::Start(0))
        .map_err(MediaError::Io)?;
    for (chunk_index, chunk) in samples.chunks(OUTPUT_COPY_CHUNK / 2).enumerate() {
        let done = chunk_index.saturating_mul(OUTPUT_COPY_CHUNK / 2);
        if let Some(hook) = checkpoint_hook {
            hook(done);
        }
        if cancel.checkpoint() {
            return Err(MediaError::Cancelled);
        }
        let bytes = mix::mono_f32_to_s16le(chunk);
        destination.write_all(&bytes).map_err(MediaError::Io)?;
        let completed = (done + chunk.len()).min(samples.len());
        let span = PCM_WRITE_END - FIRST_PASS_END;
        let mapped = FIRST_PASS_END + completed.saturating_mul(span) / samples.len().max(1);
        progress.report(mapped);
    }
    if cancel.checkpoint() {
        return Err(MediaError::Cancelled);
    }
    Ok(())
}

/// Wait for the mux child; returns its exit status and the end of its stderr.
fn wait_external_child(
    child: &mut ffmpeg_sidecar::child::FfmpegChild,
    output_pump: JoinHandle<Result<()>>,
    stderr_pump: JoinHandle<Result<Vec<u8>>>,
    cancel: &MediaCancelToken,
    progress: &DedupedProgress<'_>,
    wait_hook: Option<&dyn Fn()>,
) -> Result<(ExitStatus, Vec<u8>)> {
    if let Some(hook) = wait_hook {
        hook();
    }
    let mut polls = 0_usize;
    loop {
        if cancel.checkpoint() {
            terminate_child(child);
            let _ = join_named_pump(output_pump, "mux output");
            let _ = join_named_pump(stderr_pump, "mux stderr");
            return Err(MediaError::Cancelled);
        }
        match child.as_inner_mut().try_wait() {
            Ok(Some(status)) => {
                let output = join_named_pump(output_pump, "mux output");
                let stderr = join_named_pump(stderr_pump, "mux stderr");
                output?;
                return Ok((status, stderr?));
            }
            Ok(None) => {
                polls = polls.saturating_add(1);
                if polls.is_multiple_of(20) {
                    progress.report(
                        MUX_WAIT_START + (polls / 20).min(MUX_COPY_START - MUX_WAIT_START - 1),
                    );
                }
                thread::sleep(ENCODE_POLL_INTERVAL);
            }
            Err(error) => {
                terminate_child(child);
                let _ = join_named_pump(output_pump, "mux output");
                let _ = join_named_pump(stderr_pump, "mux stderr");
                return Err(MediaError::Io(error));
            }
        }
    }
}

fn join_named_pump<T>(pump: JoinHandle<Result<T>>, name: &str) -> Result<T> {
    pump.join()
        .map_err(|_| MediaError::Encode(format!("{name} pump panicked")))?
}

fn terminate_child(child: &mut ffmpeg_sidecar::child::FfmpegChild) {
    let _ = child.kill();
    let _ = child.wait();
}

fn report_progress(progress: Option<&EncodeProgressCallback>, done: usize) {
    if let Some(report) = progress {
        report(done.min(ENCODE_PROGRESS_TOTAL), ENCODE_PROGRESS_TOTAL);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    #[cfg(unix)]
    fn process_is_running(pid: u32) -> bool {
        std::process::Command::new("ps")
            .args(["-p", &pid.to_string()])
            .status()
            .is_ok_and(|status| status.success())
    }

    #[test]
    fn audio_chunks_spool_incrementally_without_retaining_the_timeline_mix() {
        assert!(crate::ff::ffmpeg_available(), "test requires FFmpeg");
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("chunked.mp4");
        let preset = ExportPreset::new(VideoCodec::H264, ExportResolution::P720);
        let mut encoder = VideoEncoder::new(&output, 2, 2, 1, &preset).unwrap();
        encoder
            .push_frame(&RgbaFrame::new(2, 2, vec![0; 2 * 2 * 4]))
            .unwrap();
        let spec = PcmSpec {
            sample_rate: 48_000,
            channels: 1,
            format: PcmFormat::F32,
        };
        let cancel = MediaCancelToken::new();
        encoder
            .push_audio_chunk(spec, &[0.25, -0.25], &cancel)
            .unwrap();
        encoder
            .push_audio_chunk(spec, &[0.5, -0.5], &cancel)
            .unwrap();
        let pending = encoder.pending_audio.as_ref().unwrap();
        assert_eq!(pending.sample_count, 4);
        assert_eq!(std::fs::metadata(&pending.path).unwrap().len(), 8);

        encoder.finish().unwrap();
        assert!(output.is_file());
    }

    fn workspace_entries(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn failed_encode_reports_ffmpeg_stderr() {
        assert!(crate::ff::ffmpeg_available(), "test requires FFmpeg");
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("odd.mp4");
        let preset = ExportPreset::new(VideoCodec::H264, ExportResolution::P720);
        // libx264 with 4:2:0 chroma rejects an odd width when it opens.
        let mut encoder = VideoEncoder::new(&output, 15, 16, 30, &preset).unwrap();
        let frame = RgbaFrame::new(15, 16, vec![0; 15 * 16 * 4]);
        let error = (0..64)
            .find_map(|_| encoder.push_frame(&frame).err())
            .unwrap_or_else(|| encoder.finish().unwrap_err());
        let message = error.to_string();
        assert!(
            message.contains("divisible by 2"),
            "the error quotes FFmpeg's reason: {message}"
        );
        assert!(message.len() < STDERR_TAIL_BYTES + 256, "{message}");
    }

    #[test]
    fn stderr_tail_is_bounded_and_keeps_the_end() {
        let long: Vec<u8> = (0..500)
            .flat_map(|line| format!("line {line}\n").into_bytes())
            .collect();
        let message = with_stderr_tail("ffmpeg exited 1".to_string(), &long);
        assert!(message.starts_with("ffmpeg exited 1: "));
        assert!(message.ends_with("line 499"));
        assert_eq!(message.matches(" | ").count(), STDERR_TAIL_LINES - 1);
        assert_eq!(with_stderr_tail("x".to_string(), b"\n \n"), "x");
    }

    #[test]
    fn stderr_tail_keeps_diagnostics_ahead_of_carriage_return_progress() {
        let mut stderr = b"[error] cannot write movie\n[warning] disk is full\n".to_vec();
        for frame in 0..30 {
            stderr.extend(format!("[info] frame={frame}\r").bytes());
        }
        let message = with_stderr_tail("ffmpeg exited 1".into(), &stderr);
        assert!(message.starts_with(
            "ffmpeg exited 1: [error] cannot write movie | [warning] disk is full | "
        ));
        assert!(message.ends_with("[info] frame=29"));
        assert_eq!(message.matches(" | ").count(), STDERR_TAIL_LINES - 1);
    }

    #[cfg(unix)]
    #[test]
    fn workspace_cleanup_preserves_a_replacement_directory() {
        let temp = tempfile::tempdir().unwrap();
        let workspace = EncodeWorkspace::next_to(&temp.path().join("movie.mp4")).unwrap();
        let path = workspace.path().to_path_buf();
        std::fs::write(path.join("audio.pcm"), b"our spool").unwrap();
        let moved = temp.path().join("moved-workspace");
        std::fs::rename(&path, &moved).unwrap();
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("foreign.txt"), b"another writer").unwrap();
        drop(workspace);

        assert_eq!(
            std::fs::read(path.join("foreign.txt")).unwrap(),
            b"another writer"
        );
        assert!(!moved.join("audio.pcm").exists());
    }

    #[test]
    fn copy_mode_keeps_encode_intermediates_outside_the_project_bundle() {
        assert!(crate::ff::ffmpeg_available(), "test requires FFmpeg");
        let project = tempfile::tempdir().unwrap();
        let media = project.path().join("media");
        std::fs::create_dir(&media).unwrap();
        let output = media.join("range.mp4");
        let preset = ExportPreset::new(VideoCodec::H264, ExportResolution::P720);
        let mut encoder = VideoEncoder::new(&output, 16, 16, 30, &preset).unwrap();
        let workspace = encoder.workspace_path().unwrap().to_path_buf();
        assert!(!workspace.starts_with(project.path()));
        encoder.push_frame(&RgbaFrame::black(16, 16)).unwrap();
        encoder.finish().unwrap();
        assert!(!workspace.exists());
        assert_eq!(workspace_entries(&media), ["range.mp4"]);
        assert!(crate::probe::probe(&output).unwrap().has_video);
    }

    #[cfg(unix)]
    #[test]
    fn workspace_encode_is_published_by_rename_next_to_the_output() {
        use std::os::unix::fs::MetadataExt;
        assert!(crate::ff::ffmpeg_available(), "test requires FFmpeg");
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("movie.mp4");
        std::fs::write(&output, b"previous movie").unwrap();
        let preset = ExportPreset::new(VideoCodec::H264, ExportResolution::P720);
        let mut encoder = VideoEncoder::new_in_workspace(&output, 16, 16, 30, &preset).unwrap();
        for _ in 0..3 {
            encoder.push_frame(&RgbaFrame::black(16, 16)).unwrap();
        }
        let spec = PcmSpec {
            sample_rate: 48_000,
            channels: 1,
            format: PcmFormat::F32,
        };
        encoder
            .push_audio_chunk(spec, &[0.1; 4_800], &MediaCancelToken::new())
            .unwrap();
        let reports = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let recorded = std::rc::Rc::clone(&reports);
        let record = move |done: usize, _total: usize| recorded.borrow_mut().push(done);
        let encoded = encoder
            .finish_in_workspace(&MediaCancelToken::new(), Some(&record))
            .unwrap();

        // The workspace sits in the output's directory, holds only the
        // finished movie, and the previous movie is untouched.
        assert_eq!(encoded.directory_path().parent(), Some(temp.path()));
        assert!(encoded
            .directory_path()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with(ENCODE_WORKSPACE_PREFIX));
        assert_eq!(
            workspace_entries(encoded.directory_path()),
            [encoded.name().to_string_lossy()]
        );
        assert_eq!(std::fs::read(&output).unwrap(), b"previous movie");
        assert_eq!(reports.borrow().last(), Some(&ENCODE_PROGRESS_TOTAL));
        assert!(reports.borrow().windows(2).all(|pair| pair[0] < pair[1]));

        let encoded_inode = encoded.file().metadata().unwrap().ino();
        std::fs::rename(encoded.path(), &output).unwrap();
        let workspace = encoded.directory_path().to_path_buf();
        drop(encoded);
        assert_eq!(std::fs::metadata(&output).unwrap().ino(), encoded_inode);
        assert!(!workspace.exists(), "the empty workspace is removed");
        assert_eq!(workspace_entries(temp.path()), ["movie.mp4"]);
        let probe = crate::probe::probe(&output).unwrap();
        assert!(probe.has_video && probe.has_audio);
    }

    #[test]
    fn copy_mode_and_aborted_encodes_leave_no_workspace() {
        assert!(crate::ff::ffmpeg_available(), "test requires FFmpeg");
        let temp = tempfile::tempdir().unwrap();
        let preset = ExportPreset::new(VideoCodec::H264, ExportResolution::P720);

        let copied = temp.path().join("copied.mp4");
        let mut encoder = VideoEncoder::new(&copied, 16, 16, 30, &preset).unwrap();
        encoder.push_frame(&RgbaFrame::black(16, 16)).unwrap();
        encoder.finish().unwrap();
        assert_eq!(workspace_entries(temp.path()), ["copied.mp4"]);

        let aborted = temp.path().join("aborted.mp4");
        let mut encoder = VideoEncoder::new_in_workspace(&aborted, 16, 16, 30, &preset).unwrap();
        encoder.push_frame(&RgbaFrame::black(16, 16)).unwrap();
        encoder.abort();
        assert_eq!(workspace_entries(temp.path()), ["copied.mp4"]);

        let dropped = VideoEncoder::new_in_workspace(&aborted, 16, 16, 30, &preset)
            .unwrap()
            .finish_in_workspace(&MediaCancelToken::new(), None)
            .unwrap();
        drop(dropped);
        assert_eq!(workspace_entries(temp.path()), ["copied.mp4"]);
    }

    #[test]
    fn large_output_copy_reports_each_progress_value_once() {
        assert!(crate::ff::ffmpeg_available(), "test requires FFmpeg");
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("copy.mp4");
        let preset = ExportPreset::new(VideoCodec::H264, ExportResolution::P720);
        let mut encoder = VideoEncoder::new(&output, 16, 16, 30, &preset).unwrap();
        // A sparse 128 MiB source: 2,048 chunk copies.
        let source = temp.path().join("large.bin");
        File::create(&source)
            .unwrap()
            .set_len(128 * 1024 * 1024)
            .unwrap();
        let reports = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let recorded = std::rc::Rc::clone(&reports);
        let record = move |done: usize, _total: usize| recorded.borrow_mut().push(done);
        let progress = DedupedProgress::new(Some(&record));
        encoder
            .copy_file_to_output(&source, &MediaCancelToken::new(), &progress, MUX_COPY_START)
            .unwrap();
        let reports = reports.borrow().clone();
        assert!(
            reports.len() <= ENCODE_PROGRESS_TOTAL - MUX_COPY_START,
            "{} reports",
            reports.len()
        );
        assert!(reports.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(reports.last(), Some(&(ENCODE_PROGRESS_TOTAL - 1)));
        encoder.abort();
    }

    #[cfg(unix)]
    #[test]
    fn dropping_live_encoder_reaps_child_and_releases_output() {
        assert!(
            crate::ff::ffmpeg_available(),
            "encoder lifecycle test requires FFmpeg"
        );
        let temp = tempfile::tempdir().expect("encoder temp dir");
        let output = temp.path().join("live.mp4");
        let preset = ExportPreset::new(VideoCodec::H264, ExportResolution::P720);
        let mut encoder = VideoEncoder::new(&output, 16, 16, 30, &preset).expect("start encoder");
        let pid = encoder.child_id();

        assert!(process_is_running(pid), "encoder child must be live");
        drop(encoder);

        assert!(
            !process_is_running(pid),
            "drop must kill and wait for FFmpeg"
        );
        if output.exists() {
            std::fs::remove_file(&output).expect("reaped encoder releases output file");
        }
    }

    #[cfg(unix)]
    #[test]
    fn encoder_rejects_preexisting_output_symlink() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().expect("encoder temp dir");
        let outside = temp.path().join("outside.mp4");
        std::fs::write(&outside, b"keep").expect("outside fixture");
        let output = temp.path().join("linked.mp4");
        symlink(&outside, &output).expect("output symlink");
        let preset = ExportPreset::new(VideoCodec::H264, ExportResolution::P720);

        let error = VideoEncoder::new(&output, 16, 16, 30, &preset)
            .err()
            .expect("encoder must reject output links");

        assert!(error.to_string().contains("symlink"));
        assert_eq!(std::fs::read(&outside).expect("outside readable"), b"keep");
    }

    #[test]
    fn encode_args_declare_rawvideo_stdin_input() {
        let preset = ExportPreset::new(VideoCodec::H264, ExportResolution::P1080);
        let args = encode_args(Path::new("/out.mp4"), 1920, 1080, 30, &preset);
        // input is rawvideo rgba from stdin at the right size/fps.
        assert!(args.windows(2).any(|w| w == ["-f", "rawvideo"]));
        assert!(args.windows(2).any(|w| w == ["-pix_fmt", "rgba"]));
        assert!(args.windows(2).any(|w| w == ["-s", "1920x1080"]));
        assert!(args.windows(2).any(|w| w == ["-r", "30"]));
        assert!(args.windows(2).any(|w| w == ["-i", "-"]));
        assert_eq!(args.last().unwrap(), "/out.mp4");
    }

    #[test]
    fn encode_args_use_preset_codec_and_color() {
        let preset = ExportPreset::new(VideoCodec::H265, ExportResolution::P720);
        let args = encode_args(Path::new("/o.mp4"), 1280, 720, 24, &preset);
        assert!(args.windows(2).any(|w| w == ["-c:v", "libx265"]));
        assert!(args.windows(2).any(|w| w == ["-pix_fmt", "yuv420p"]));
        assert!(args.windows(2).any(|w| w == ["-colorspace", "bt709"]));
    }

    #[test]
    fn encode_args_prores_pixfmt_and_bt709_delivery_tags() {
        let preset = ExportPreset::new(VideoCodec::ProRes422, ExportResolution::P2160);
        let args = encode_args(Path::new("/o.mov"), 3840, 2160, 30, &preset);
        assert!(args.windows(2).any(|w| w == ["-c:v", "prores_ks"]));
        assert!(args.windows(2).any(|w| w == ["-pix_fmt", "yuv422p10le"]));
        assert!(args.windows(2).any(|w| w == ["-colorspace", "bt709"]));
        assert!(args.iter().any(|arg| arg
            .to_str()
            .unwrap()
            .contains("setparams=color_primaries=bt709")));
    }

    #[test]
    fn encode_args_prores_4444_preserve_alpha() {
        let preset = ExportPreset::new(VideoCodec::ProRes4444, ExportResolution::P1080);
        let args = encode_args(Path::new("/matte.mov"), 1920, 1080, 30, &preset);
        assert!(args.windows(2).any(|w| w == ["-c:v", "prores_ks"]));
        assert!(args.windows(2).any(|w| w == ["-pix_fmt", "yuva444p10le"]));
        assert!(args.windows(2).any(|w| w == ["-profile:v", "4444"]));
    }

    #[test]
    fn mux_args_copy_video_and_encode_audio() {
        let args = mux_args(
            Path::new("/v.mp4"),
            Path::new("/a.pcm"),
            Path::new("/out.mp4"),
            48_000,
            "aac",
            Some("4.000000"),
        );
        // video input first, then the raw s16le PCM input declared with rate/ch.
        assert!(args.windows(2).any(|w| w == ["-i", "/v.mp4"]));
        assert!(args.windows(2).any(|w| w == ["-f", "s16le"]));
        assert!(args.windows(2).any(|w| w == ["-ar", "48000"]));
        assert!(args.windows(2).any(|w| w == ["-ac", "1"]));
        assert!(args.windows(2).any(|w| w == ["-i", "/a.pcm"]));
        // copy the video stream, encode audio with the preset codec.
        assert!(args.windows(2).any(|w| w == ["-c:v", "copy"]));
        assert!(args.windows(2).any(|w| w == ["-c:a", "aac"]));
        // Silence pads short audio; the video's duration bounds the output.
        assert!(args.windows(2).any(|w| w == ["-af", "apad"]));
        assert!(args.windows(2).any(|w| w == ["-t", "4.000000"]));
        assert!(!args.iter().any(|a| a == "-shortest"));
        assert_eq!(args.last().unwrap(), "/out.mp4");
    }

    #[test]
    fn mux_args_without_video_frames_mux_audio_as_is() {
        let args = mux_args(
            Path::new("/v.mp4"),
            Path::new("/a.pcm"),
            Path::new("/out.mp4"),
            48_000,
            "aac",
            None,
        );
        assert!(!args
            .iter()
            .any(|a| a == "apad" || a == "-t" || a == "-shortest"));
    }

    #[test]
    fn video_duration_covers_every_pushed_frame() {
        assert_eq!(video_duration_arg(120, 30).as_deref(), Some("4.000000"));
        // 100 / 24 s = 4.1666…: rounded up so the last frame stays inside.
        assert_eq!(video_duration_arg(100, 24).as_deref(), Some("4.166667"));
        assert_eq!(video_duration_arg(1, 60).as_deref(), Some("0.016667"));
        assert_eq!(video_duration_arg(0, 30), None);
        assert_eq!(video_duration_arg(10, 0), None);
    }

    #[test]
    fn mux_args_threads_prores_lpcm_codec() {
        let args = mux_args(
            Path::new("/v.mov"),
            Path::new("/a.pcm"),
            Path::new("/out.mov"),
            48_000,
            "pcm_s16le",
            Some("1.000000"),
        );
        assert!(args.windows(2).any(|w| w == ["-c:a", "pcm_s16le"]));
    }

    /// `(video frames, audio seconds)` of an encoded file, counted by decoding.
    fn stream_lengths(path: &Path) -> (u64, f64) {
        let output = std::process::Command::new(crate::ff::ffprobe_path())
            .args(["-v", "error", "-count_frames", "-of", "json"])
            .args(["-show_entries", "stream=codec_type,nb_read_frames,duration"])
            .arg(path)
            .output()
            .expect("run ffprobe");
        assert!(output.status.success(), "ffprobe {}", path.display());
        let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        let streams = json["streams"].as_array().unwrap();
        let of_type = |kind: &str| {
            streams
                .iter()
                .find(|stream| stream["codec_type"] == kind)
                .unwrap_or_else(|| panic!("no {kind} stream"))
        };
        let frames = of_type("video")["nb_read_frames"]
            .as_str()
            .unwrap()
            .parse()
            .unwrap();
        let audio = of_type("audio")["duration"]
            .as_str()
            .unwrap()
            .parse()
            .unwrap();
        (frames, audio)
    }

    fn encode_with_audio(output: &Path, frames: usize, fps: i32, audio_secs: f64) {
        let preset = ExportPreset::new(VideoCodec::H264, ExportResolution::P720);
        let mut encoder = VideoEncoder::new(output, 16, 16, fps, &preset).unwrap();
        for index in 0..frames {
            let value = (index % 256) as u8;
            encoder
                .push_frame(&RgbaFrame::new(16, 16, [value, 0, 0, 255].repeat(16 * 16)))
                .unwrap();
        }
        let spec = PcmSpec {
            sample_rate: 48_000,
            channels: 1,
            format: PcmFormat::F32,
        };
        let samples = (audio_secs * 48_000.0) as usize;
        encoder
            .push_audio(PcmBuffer {
                spec,
                samples_f32: (0..samples)
                    .map(|index| (index as f32 * 0.0575).sin() * 0.5)
                    .collect(),
            })
            .unwrap();
        encoder.finish().unwrap();
    }

    #[test]
    fn short_audio_is_padded_instead_of_cutting_the_video() {
        assert!(crate::ff::ffmpeg_available(), "test requires FFmpeg");
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("short-audio.mp4");

        encode_with_audio(&output, 120, 30, 2.5);

        let (frames, audio_secs) = stream_lengths(&output);
        assert_eq!(frames, 120, "every encoded frame survives the mux");
        assert!(
            (audio_secs - 4.0).abs() <= 1.0 / 30.0,
            "audio {audio_secs} s"
        );
    }

    #[test]
    fn long_audio_still_ends_with_the_video() {
        assert!(crate::ff::ffmpeg_available(), "test requires FFmpeg");
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("long-audio.mp4");

        encode_with_audio(&output, 120, 30, 5.0);

        let (frames, audio_secs) = stream_lengths(&output);
        assert_eq!(frames, 120);
        assert!(
            (audio_secs - 4.0).abs() <= 1.0 / 30.0,
            "audio {audio_secs} s"
        );
    }

    #[test]
    fn cancellation_inside_mux_pcm_write_stops_the_actual_loop() {
        let temp = tempfile::tempdir().expect("PCM temp dir");
        let output = temp.path().join("audio.pcm");
        let samples = vec![0.25_f32; (OUTPUT_COPY_CHUNK / 2) * 4];
        let expected_full_len = samples.len() * 2;
        let cancel = MediaCancelToken::new();
        let worker_cancel = cancel.clone();
        let worker_output = output.clone();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let mut destination = File::create(&worker_output).expect("create PCM destination");
            let hook = move |done: usize| {
                if done == OUTPUT_COPY_CHUNK / 2 {
                    entered_tx.send(()).expect("PCM checkpoint entered");
                    release_rx.recv().expect("release PCM checkpoint");
                }
            };
            write_pcm_s16le_cancellable(
                &samples,
                &mut destination,
                &worker_cancel,
                None,
                Some(&hook),
            )
        });

        entered_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("actual PCM write loop reached its second chunk");
        cancel.cancel();
        release_tx.send(()).expect("release PCM loop");
        let result = worker.join().expect("PCM writer joins");

        assert!(matches!(result, Err(MediaError::Cancelled)));
        let partial_len = std::fs::metadata(&output)
            .expect("partial PCM exists")
            .len() as usize;
        assert!(partial_len > 0);
        assert!(partial_len < expected_full_len);
    }

    fn assert_cancelling_mux_wait_reaps_child() {
        assert!(
            crate::ff::ffmpeg_available(),
            "mux cancellation test requires FFmpeg"
        );
        let mut child = crate::ff::ffmpeg()
            .args([
                "-re",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=48000:cl=mono",
                "-t",
                "30",
                "-f",
                "null",
                "-",
            ])
            .spawn()
            .expect("spawn blocking mux-like FFmpeg");
        let stdout = child.take_stdout().expect("FFmpeg stdout");
        let stderr = child.take_stderr().expect("FFmpeg stderr");
        let output_pump = std::thread::spawn(move || drain_stdout(stdout));
        let stderr_pump = std::thread::spawn(move || drain_stderr(stderr));
        let cancel = MediaCancelToken::new();
        let worker_cancel = cancel.clone();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let hook = move || {
                entered_tx.send(()).expect("mux wait entered");
                release_rx.recv().expect("release mux wait");
            };
            let result = wait_external_child(
                &mut child,
                output_pump,
                stderr_pump,
                &worker_cancel,
                &DedupedProgress::new(None),
                Some(&hook),
            );
            let reaped = child
                .as_inner_mut()
                .try_wait()
                .expect("inspect mux child")
                .is_some();
            (result, reaped)
        });

        entered_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("mux wait reached cancellation barrier");
        cancel.cancel();
        release_tx.send(()).expect("release mux wait");
        let (result, reaped) = worker.join().expect("mux wait worker joins");

        assert!(matches!(result, Err(MediaError::Cancelled)));
        assert!(reaped, "cancelled mux child must be killed and waited");
    }

    #[cfg(not(windows))]
    #[test]
    fn cancelling_mux_wait_reaps_child() {
        assert_cancelling_mux_wait_reaps_child();
    }

    #[cfg(windows)]
    #[test]
    fn windows_cancelling_mux_wait_reaps_child() {
        assert_cancelling_mux_wait_reaps_child();
    }
    #[cfg(windows)]
    #[test]
    fn windows_workspace_is_hidden_and_keeps_its_retained_identity() {
        use std::os::windows::ffi::OsStringExt;
        use std::os::windows::fs::MetadataExt;
        use windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_HIDDEN;

        let temp = tempfile::tempdir().unwrap();
        let parent = temp
            .path()
            .join(std::ffi::OsString::from_wide(&[100, 105, 114, 45, 0xd800]));
        std::fs::create_dir(&parent).unwrap();
        let workspace = EncodeWorkspace::in_directory(&parent).unwrap();
        let path = workspace.path().to_path_buf();
        let attributes = workspace.directory.metadata().unwrap().file_attributes();
        assert_ne!(
            attributes & FILE_ATTRIBUTE_HIDDEN,
            0,
            "private workspace must be hidden on Windows"
        );
        assert!(
            std::fs::rename(&path, parent.join("moved")).is_err(),
            "the original no-delete-sharing lease must still prevent rebinding"
        );
        drop(workspace);
        assert!(
            !path.exists(),
            "the hidden workspace is still cleaned up by its retained handle"
        );
    }
}
