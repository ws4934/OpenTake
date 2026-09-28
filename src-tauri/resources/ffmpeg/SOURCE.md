# Bundled FFmpeg

OpenTake packages checksum-pinned FFmpeg and FFprobe binaries. The exact
per-platform download URL, archive member (when applicable), archive SHA-256,
extracted-binary SHA-256, and reported version are recorded in
`scripts/ffmpeg-sidecars.lock.json`.

The Apple Silicon pair is the FFmpeg 7.0 arm64 build published by
[OSXExperts](https://www.osxexperts.net/). Its reported configure line enables
GPL components such as x264/x265 but does **not** enable FFmpeg's `nonfree`
option. Both programs report GNU GPL terms through `-L`. The provisioning script
fails closed when a binary reports `--enable-nonfree` or says it is not legally
redistributable. Other target records remain pinned to the
[`eugeneware/ffmpeg-static` b6.1.1 release](https://github.com/eugeneware/ffmpeg-static/releases/tag/b6.1.1)
and are subject to the same fail-closed runtime license check on their native
build hosts.

FFmpeg is distributed under GPL-compatible terms; OpenTake itself is
GPL-3.0-or-later and packages its complete GPL license and this source notice.

The Linux x86_64 assets in that release are John Van Sickle static builds of
FFmpeg **7.0.2**, despite the release tag being `b6.1.1`. Their extracted-binary
hashes and reported version are pinned separately in the lock.

Tauri dev/build automatically provisions only the selected target (the Tauri
hook target, explicit `--target`, or native Rust host). Cached binaries are
verified again; missing or corrupt downloads fail the hook. Supported targets:
macOS arm64/x86_64, Windows x86_64 MSVC, and Linux x86_64 GNU. Binaries stay in
`src-tauri/binaries/` locally and are gitignored. Tauri `externalBin` packages
the selected pair beside the application executable; development uses that
pair or the target-named source binaries, never implicit host/PATH discovery.
On Linux the pair is named `opentake-ffmpeg` and `opentake-ffprobe`, because
deb and rpm bundles install it into the shared `/usr/bin`, next to the
distribution's own `ffmpeg` package.
Debug-only `OPENTAKE_FFMPEG` / `OPENTAKE_FFPROBE` overrides remain available.

FFmpeg source tags used by the locked target assets:
https://github.com/FFmpeg/FFmpeg/tree/n7.0 and
https://github.com/FFmpeg/FFmpeg/tree/n6.1.1 and
https://github.com/FFmpeg/FFmpeg/tree/n7.0.2
