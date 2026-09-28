# Bundled FFmpeg

OpenTake packages checksum-pinned FFmpeg and FFprobe binaries. The exact
per-platform download URL, archive member (when applicable), archive SHA-256,
extracted-binary SHA-256, and reported version are recorded in
`scripts/ffmpeg-sidecars.lock.json`.

## Download locations

The primary download location is the
[`ffmpeg-sidecars-v1` prerelease](https://github.com/ws4934/OpenTake/releases/tag/ffmpeg-sidecars-v1)
of this repository. It holds byte-identical copies of the pinned upstream
files (the original archive or raw binary, unchanged, named
`<tool>-<target>-<sha256[:16]>` plus the upstream extension, where the hash
prefix is the pinned SHA-256 of that file) and is listed as the
`mirror_urls` of each lock record. The upstream `url` of each record remains
the provenance of the file and the fallback: `scripts/provision_ffmpeg_sidecars.py`
tries the mirror first and moves to the upstream URL on a network error, an
HTTP error or a checksum mismatch. Every source is held to the same archive and
binary SHA-256 pins, so a mirror can never change what is packaged.

The `Mirror FFmpeg sidecars` workflow
(`.github/workflows/mirror-ffmpeg-sidecars.yml`, run manually from `main`)
fills the release with `scripts/mirror_ffmpeg_sidecars.py`: it downloads every
file from its upstream URL, verifies its pins (the archive and the extracted
binary SHA-256 for a zip, the binary SHA-256 for a raw binary) and uploads the
missing assets. The script only appends: an existing asset with the same
SHA-256 is kept, one with different bytes fails the run, and it never deletes
or replaces an asset. GitHub does not enforce this; anyone with write access
to the repository can replace or delete a release asset. That is harmless
only because every download, from the mirror or upstream, is checked against
the pins in the lock, so a replaced asset fails verification and the
provisioner falls back to the next source. Changing a pin changes the asset
name, so the new `mirror_urls` must be updated with it (the unit tests and the
mirror workflow reject a mismatch) and the workflow run again.

## Builds and licences

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
