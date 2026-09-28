from __future__ import annotations

import hashlib
import http.client
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest import mock
import zipfile


ROOT = Path(
    os.environ.get("OPENTAKE_REPOSITORY_ROOT", Path(__file__).resolve().parents[2])
).resolve()
MODULE_PATH = Path(
    os.environ.get(
        "OPENTAKE_PROVISIONER_PATH",
        ROOT / "scripts" / "provision_ffmpeg_sidecars.py",
    )
).resolve()
SPEC = importlib.util.spec_from_file_location("provision_ffmpeg_sidecars", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
provisioner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(provisioner)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ProvisionFfmpegSidecarsTests(unittest.TestCase):
    def test_cli_target_precedence_is_explicit_then_tauri_then_host(self) -> None:
        cases = [
            (["--target", "aarch64-apple-darwin"], "x86_64-unknown-linux-gnu", "aarch64-apple-darwin"),
            ([], "x86_64-unknown-linux-gnu", "x86_64-unknown-linux-gnu"),
            ([], "", "x86_64-pc-windows-msvc"),
        ]
        for arguments, hook_target, expected in cases:
            with self.subTest(arguments=arguments, hook_target=hook_target):
                with (
                    tempfile.TemporaryDirectory() as directory,
                    mock.patch.object(provisioner, "BIN_DIR", Path(directory)),
                    mock.patch.object(provisioner.sys, "argv", ["provisioner", *arguments]),
                    mock.patch.dict(os.environ, {"TAURI_ENV_TARGET_TRIPLE": hook_target}),
                    mock.patch.object(provisioner, "host_target", return_value="x86_64-pc-windows-msvc") as host,
                    mock.patch.object(provisioner, "provision") as install,
                ):
                    self.assertEqual(provisioner.main(), 0)
                self.assertEqual([call.args[2] for call in install.call_args_list], [expected, expected])
                self.assertEqual(host.call_count, int(not arguments and not hook_target))

    def test_repository_root_can_be_bound_when_tooling_runs_outside_checkout(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            expected_root = Path(directory).resolve()
            with mock.patch.dict(
                os.environ,
                {"OPENTAKE_REPOSITORY_ROOT": str(expected_root)},
            ):
                isolated_spec = importlib.util.spec_from_file_location(
                    "isolated_provision_ffmpeg_sidecars", MODULE_PATH
                )
                assert isolated_spec is not None and isolated_spec.loader is not None
                isolated = importlib.util.module_from_spec(isolated_spec)
                isolated_spec.loader.exec_module(isolated)

            self.assertEqual(isolated.ROOT, expected_root)
            self.assertEqual(
                isolated.BIN_DIR, expected_root / "src-tauri" / "binaries"
            )

    def test_provision_publishes_an_unexecuted_copy_when_windows_locks_images(
        self,
    ) -> None:
        binary = b"redistributable ffmpeg"
        expected_sha = digest(binary)
        record = {
            "url": "https://example.invalid/ffmpeg.exe",
            "sha256": expected_sha,
            "version": "7.0",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary_dir = root / "src-tauri" / "binaries"
            binary_dir.mkdir(parents=True)
            executed_paths: set[Path] = set()
            real_replace = os.replace

            def download_fixture(_url: str, path: Path) -> None:
                path.write_bytes(binary)

            def lock_executed_image(
                path: Path, _expected_sha: str, _version: str
            ) -> None:
                executed_paths.add(path)

            def windows_replace(source: Path, destination: Path) -> None:
                if source in executed_paths:
                    raise PermissionError(
                        32,
                        "The process cannot access the file because it is being used "
                        "by another process",
                        str(source),
                    )
                real_replace(source, destination)

            with (
                mock.patch.object(provisioner, "ROOT", root),
                mock.patch.object(provisioner, "BIN_DIR", binary_dir),
                mock.patch.object(provisioner, "download", download_fixture),
                mock.patch.object(provisioner, "verify", lock_executed_image),
                mock.patch.object(provisioner.os, "replace", windows_replace),
            ):
                provisioner.provision(
                    "ffmpeg", record, "x86_64-pc-windows-msvc"
                )

            final_path = binary_dir / "ffmpeg-x86_64-pc-windows-msvc.exe"
            self.assertEqual(final_path.read_bytes(), binary)
            self.assertTrue(executed_paths)
            self.assertTrue(
                all(binary_dir not in path.parents for path in executed_paths)
            )

    def test_cached_sidecar_is_probed_only_from_a_system_temporary_copy(self) -> None:
        binary = b"redistributable ffmpeg"
        expected_sha = digest(binary)
        record = {
            "url": "https://example.invalid/ffmpeg.exe",
            "sha256": expected_sha,
            "version": "7.0",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary_dir = root / "src-tauri" / "binaries"
            binary_dir.mkdir(parents=True)
            final_path = binary_dir / "ffmpeg-x86_64-pc-windows-msvc.exe"
            final_path.write_bytes(binary)
            executed_paths: list[Path] = []

            def record_probe(
                path: Path, _expected_sha: str, _version: str
            ) -> None:
                executed_paths.append(path)

            with (
                mock.patch.object(provisioner, "ROOT", root),
                mock.patch.object(provisioner, "BIN_DIR", binary_dir),
                mock.patch.object(provisioner, "verify", record_probe),
            ):
                provisioner.provision(
                    "ffmpeg", record, "x86_64-pc-windows-msvc"
                )

            self.assertEqual(final_path.read_bytes(), binary)
            self.assertEqual(len(executed_paths), 1)
            self.assertNotEqual(executed_paths[0], final_path)
            self.assertNotIn(binary_dir, executed_paths[0].parents)

    def test_materializes_only_the_checksum_pinned_zip_member(self) -> None:
        binary = b"redistributable ffmpeg"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "sidecar.zip"
            destination = root / "ffmpeg"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("ffmpeg", binary)
                archive.writestr("ignored", b"not selected")
            record = {
                "archive": {
                    "format": "zip",
                    "member": "ffmpeg",
                    "sha256": provisioner.sha256(archive_path),
                }
            }

            provisioner.materialize_download(record, archive_path, destination)

            self.assertEqual(destination.read_bytes(), binary)

    def provision_with_downloads(
        self, record: dict[str, object], remaining: list[bytes], delays: list[float]
    ) -> Path:
        """Provision a Linux sidecar whose successive downloads pop their bytes
        from the front of `remaining`; every back-off delay is appended to
        `delays`. Returns the published path."""

        def download_fixture(_url: str, path: Path) -> None:
            path.write_bytes(remaining.pop(0))

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        binary_dir = root / "src-tauri" / "binaries"
        binary_dir.mkdir(parents=True)
        with (
            mock.patch.object(provisioner, "ROOT", root),
            mock.patch.object(provisioner, "BIN_DIR", binary_dir),
            mock.patch.object(provisioner, "download", download_fixture),
            mock.patch.object(provisioner, "verify", lambda *_args: None),
            mock.patch.object(provisioner.time, "sleep", delays.append),
        ):
            provisioner.provision("ffmpeg", record, "x86_64-unknown-linux-gnu")
            return provisioner.destination("ffmpeg", "x86_64-unknown-linux-gnu")

    def test_download_with_the_wrong_bytes_is_fetched_again(self) -> None:
        binary = b"pinned ffmpeg"
        record = {
            "url": "https://example.invalid/ffmpeg",
            "sha256": digest(binary),
            "version": "7.0",
        }

        remaining = [b"replaced upstream", binary]
        delays: list[float] = []
        published = self.provision_with_downloads(record, remaining, delays)

        self.assertEqual(published.read_bytes(), binary)
        self.assertEqual(remaining, [])
        self.assertEqual(delays, [1])

    def test_archive_with_the_wrong_bytes_is_fetched_again(self) -> None:
        binary = b"pinned ffprobe"

        def archive_bytes(member: bytes) -> bytes:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "sidecar.zip"
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr("ffmpeg", member)
                return path.read_bytes()

        pinned_archive = archive_bytes(binary)
        record = {
            "url": "https://example.invalid/ffmpeg.zip",
            "sha256": digest(binary),
            "version": "7.0",
            "archive": {
                "format": "zip",
                "member": "ffmpeg",
                "sha256": digest(pinned_archive),
            },
        }

        remaining = [archive_bytes(b"other build"), pinned_archive]
        published = self.provision_with_downloads(record, remaining, [])

        self.assertEqual(published.read_bytes(), binary)
        self.assertEqual(remaining, [])

    def test_repeated_checksum_mismatch_fails_after_bounded_downloads(self) -> None:
        record = {
            "url": "https://example.invalid/ffmpeg",
            "sha256": digest(b"pinned ffmpeg"),
            "version": "7.0",
        }
        attempts = provisioner.DOWNLOAD_CHECKSUM_ATTEMPTS
        remaining = [b"wrong"] * (attempts + 1)
        delays: list[float] = []

        with self.assertRaisesRegex(
            RuntimeError, f"checksum mismatch.*after {attempts} downloads"
        ):
            self.provision_with_downloads(record, remaining, delays)

        self.assertEqual(len(remaining), 1, "one download per attempt")
        self.assertEqual(delays, list(range(1, attempts)))

    def provision_from_sources(
        self,
        record: dict[str, object],
        responses: dict[str, list[object]],
        verified: list[tuple[bytes, str, str]] | None = None,
    ) -> tuple[Path, list[str]]:
        """Provision a Linux sidecar where each download of a URL pops the next
        response for it: bytes are written, an exception is raised. Each call
        of `verify` appends the checked bytes, SHA-256 and version to
        `verified`. Returns the published path and the downloaded URLs in
        order."""
        requested: list[str] = []
        self.client_error_retries: dict[str, bool] = {}
        checks = [] if verified is None else verified

        def download_fixture(
            url: str, path: Path, *, retry_client_errors: bool = True
        ) -> None:
            requested.append(url)
            self.client_error_retries[url] = retry_client_errors
            response = responses[url].pop(0)
            if isinstance(response, BaseException):
                raise response
            assert isinstance(response, bytes)
            path.write_bytes(response)

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        binary_dir = root / "src-tauri" / "binaries"
        binary_dir.mkdir(parents=True)
        with (
            mock.patch.object(provisioner, "ROOT", root),
            mock.patch.object(provisioner, "BIN_DIR", binary_dir),
            mock.patch.object(provisioner, "download", download_fixture),
            mock.patch.object(
                provisioner,
                "verify",
                lambda path, sha, version: checks.append(
                    (path.read_bytes(), sha, version)
                ),
            ),
            mock.patch.object(provisioner.time, "sleep", lambda _delay: None),
            mock.patch.object(provisioner.sys, "stderr", mock.MagicMock()),
        ):
            provisioner.provision("ffmpeg", record, "x86_64-unknown-linux-gnu")
            return (
                provisioner.destination("ffmpeg", "x86_64-unknown-linux-gnu"),
                requested,
            )

    MIRROR = "https://mirror.invalid/ffmpeg-x86_64-unknown-linux-gnu"
    SECOND_MIRROR = "https://second-mirror.invalid/ffmpeg"
    UPSTREAM = "https://upstream.invalid/ffmpeg-linux-x64"
    PINNED = b"pinned ffmpeg"

    def mirrored_record(self, *mirrors: str) -> dict[str, object]:
        return {
            "url": self.UPSTREAM,
            "mirror_urls": list(mirrors),
            "sha256": digest(self.PINNED),
            "version": "7.0",
        }

    def test_mirror_is_downloaded_before_upstream(self) -> None:
        record = self.mirrored_record(self.MIRROR, self.SECOND_MIRROR)
        verified: list[tuple[bytes, str, str]] = []
        published, requested = self.provision_from_sources(
            record, {self.MIRROR: [self.PINNED]}, verified
        )

        self.assertEqual(published.read_bytes(), self.PINNED)
        self.assertEqual(requested, [self.MIRROR])
        self.assertEqual(verified, [(self.PINNED, digest(self.PINNED), "7.0")])
        self.assertEqual(
            provisioner.download_sources(record),
            [self.MIRROR, self.SECOND_MIRROR, self.UPSTREAM],
        )

    def test_network_error_falls_back_to_the_next_source(self) -> None:
        published, requested = self.provision_from_sources(
            self.mirrored_record(self.MIRROR, self.SECOND_MIRROR),
            {
                self.MIRROR: [RuntimeError("download failed after 4 attempts")],
                self.SECOND_MIRROR: [ConnectionResetError("reset by peer")],
                self.UPSTREAM: [self.PINNED],
            },
        )

        self.assertEqual(published.read_bytes(), self.PINNED)
        self.assertEqual(requested, [self.MIRROR, self.SECOND_MIRROR, self.UPSTREAM])

    def test_missing_mirror_release_falls_back_to_upstream(self) -> None:
        published, requested = self.provision_from_sources(
            self.mirrored_record(self.MIRROR),
            {
                self.MIRROR: [RuntimeError("download failed: HTTP 404")],
                self.UPSTREAM: [self.PINNED],
            },
        )

        self.assertEqual(published.read_bytes(), self.PINNED)
        self.assertEqual(requested, [self.MIRROR, self.UPSTREAM])
        self.assertEqual(
            self.client_error_retries, {self.MIRROR: False, self.UPSTREAM: True}
        )

    def test_mirror_404_is_not_retried_but_upstream_and_network_errors_are(
        self,
    ) -> None:
        not_found = urllib.error.HTTPError(self.MIRROR, 404, "Not Found", {}, None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "download"
            with (
                mock.patch.object(
                    provisioner.urllib.request, "urlopen", side_effect=not_found
                ) as urlopen,
                mock.patch.object(provisioner.time, "sleep") as sleep,
            ):
                with self.assertRaisesRegex(RuntimeError, "HTTP 404"):
                    provisioner.download(
                        self.MIRROR, path, retry_client_errors=False
                    )
            self.assertEqual(urlopen.call_count, 1)
            sleep.assert_not_called()

            with (
                mock.patch.object(
                    provisioner.urllib.request, "urlopen", side_effect=not_found
                ) as urlopen,
                mock.patch.object(provisioner.time, "sleep"),
            ):
                with self.assertRaisesRegex(RuntimeError, "after 4 attempts.*404"):
                    provisioner.download(self.UPSTREAM, path)
            self.assertEqual(
                urlopen.call_count, provisioner.DOWNLOAD_NETWORK_ATTEMPTS
            )

            with (
                mock.patch.object(
                    provisioner.urllib.request,
                    "urlopen",
                    side_effect=urllib.error.URLError("connection refused"),
                ) as urlopen,
                mock.patch.object(provisioner.time, "sleep"),
            ):
                with self.assertRaisesRegex(RuntimeError, "after 4 attempts"):
                    provisioner.download(self.MIRROR, path)
            self.assertEqual(
                urlopen.call_count, provisioner.DOWNLOAD_NETWORK_ATTEMPTS
            )
            self.assertFalse(path.exists())

    def test_body_cut_short_is_retried_and_then_falls_back(self) -> None:
        class TruncatedResponse:
            def __enter__(self) -> "TruncatedResponse":
                return self

            def __exit__(self, *_exc: object) -> None:
                return None

            def read(self, _size: int) -> bytes:
                raise http.client.IncompleteRead(b"partial", 1024)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "download"
            with (
                mock.patch.object(
                    provisioner.urllib.request,
                    "urlopen",
                    side_effect=lambda *_args, **_kwargs: TruncatedResponse(),
                ) as urlopen,
                mock.patch.object(provisioner.time, "sleep"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "after 4 attempts.*IncompleteRead"
                ):
                    provisioner.download(
                        self.MIRROR, path, retry_client_errors=False
                    )
            self.assertEqual(
                urlopen.call_count, provisioner.DOWNLOAD_NETWORK_ATTEMPTS
            )
            self.assertFalse(path.exists())

    def test_checksum_mismatch_falls_back_to_the_next_source(self) -> None:
        attempts = provisioner.DOWNLOAD_CHECKSUM_ATTEMPTS
        published, requested = self.provision_from_sources(
            self.mirrored_record(self.MIRROR),
            {
                self.MIRROR: [b"stale mirror"] * attempts,
                self.UPSTREAM: [self.PINNED],
            },
        )

        self.assertEqual(published.read_bytes(), self.PINNED)
        self.assertEqual(requested, [self.MIRROR] * attempts + [self.UPSTREAM])

    def test_failure_names_every_source_when_all_fail(self) -> None:
        attempts = provisioner.DOWNLOAD_CHECKSUM_ATTEMPTS
        with self.assertRaises(RuntimeError) as raised:
            self.provision_from_sources(
                self.mirrored_record(self.MIRROR, self.SECOND_MIRROR),
                {
                    self.MIRROR: [RuntimeError("download failed: HTTP 404")],
                    self.SECOND_MIRROR: [TimeoutError("timed out")],
                    self.UPSTREAM: [b"replaced upstream"] * attempts,
                },
            )

        message = str(raised.exception)
        self.assertIn("every download source failed for ffmpeg", message)
        self.assertIn(f"{self.MIRROR}: download failed: HTTP 404", message)
        self.assertIn(f"{self.SECOND_MIRROR}: timed out", message)
        self.assertRegex(
            message,
            f"{self.UPSTREAM}: download checksum mismatch.*after {attempts} downloads",
        )

    def test_mirror_zip_with_a_bad_archive_hash_falls_back(self) -> None:
        def archive_bytes(*members: tuple[str, bytes]) -> bytes:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "sidecar.zip"
                with zipfile.ZipFile(path, "w") as archive:
                    for name, data in members:
                        archive.writestr(name, data)
                return path.read_bytes()

        pinned_archive = archive_bytes(("ffmpeg", self.PINNED))
        # Same pinned member, but not the pinned archive: rejected by the
        # archive pin before the member is ever extracted.
        other_archive = archive_bytes(("ffmpeg", self.PINNED), ("extra", b"x"))
        record = {
            **self.mirrored_record(self.MIRROR + ".zip"),
            "url": self.UPSTREAM + ".zip",
            "archive": {
                "format": "zip",
                "member": "ffmpeg",
                "sha256": digest(pinned_archive),
            },
        }
        attempts = provisioner.DOWNLOAD_CHECKSUM_ATTEMPTS
        published, requested = self.provision_from_sources(
            record,
            {
                self.MIRROR + ".zip": [other_archive] * attempts,
                self.UPSTREAM + ".zip": [pinned_archive],
            },
        )

        self.assertEqual(published.read_bytes(), self.PINNED)
        self.assertEqual(
            requested, [self.MIRROR + ".zip"] * attempts + [self.UPSTREAM + ".zip"]
        )

    def test_nonfree_check_runs_on_a_mirror_sourced_file(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        binary_dir = root / "src-tauri" / "binaries"
        binary_dir.mkdir(parents=True)
        requested: list[str] = []

        def download_fixture(url: str, path: Path, **_kwargs: object) -> None:
            requested.append(url)
            path.write_bytes(self.PINNED)

        with (
            mock.patch.object(provisioner, "ROOT", root),
            mock.patch.object(provisioner, "BIN_DIR", binary_dir),
            mock.patch.object(provisioner, "download", download_fixture),
            mock.patch.object(
                provisioner.subprocess,
                "check_output",
                return_value="ffmpeg version 7.0\nconfiguration: --enable-nonfree\n",
            ) as metadata,
        ):
            with self.assertRaisesRegex(RuntimeError, "nonfree sidecar"):
                provisioner.provision(
                    "ffmpeg",
                    self.mirrored_record(self.MIRROR),
                    "x86_64-unknown-linux-gnu",
                )
            final_path = provisioner.destination("ffmpeg", "x86_64-unknown-linux-gnu")

        self.assertEqual(requested, [self.MIRROR])
        metadata.assert_called_once()
        self.assertFalse(final_path.exists())
        self.assertEqual(list(binary_dir.iterdir()), [])

    def test_record_without_mirrors_downloads_only_upstream(self) -> None:
        record = self.mirrored_record()
        del record["mirror_urls"]
        published, requested = self.provision_from_sources(
            record, {self.UPSTREAM: [self.PINNED]}
        )

        self.assertEqual(published.read_bytes(), self.PINNED)
        self.assertEqual(requested, [self.UPSTREAM])
        self.assertEqual(provisioner.download_sources(record), [self.UPSTREAM])

    def test_malformed_mirror_urls_are_rejected(self) -> None:
        cases = [
            ("https://mirror.invalid/ffmpeg", "sidecar lock mirror_urls must be a list"),
            ([""], "sidecar lock mirror_urls must be non-empty strings"),
            ([None], "sidecar lock mirror_urls must be non-empty strings"),
        ]
        for mirrors, message in cases:
            with self.subTest(mirrors=mirrors):
                record = {"url": self.UPSTREAM, "mirror_urls": mirrors}
                with self.assertRaises(RuntimeError) as raised:
                    provisioner.download_sources(record)
                self.assertEqual(str(raised.exception), message)
        with self.assertRaises(RuntimeError) as raised:
            provisioner.download_sources({"mirror_urls": []})
        self.assertEqual(
            str(raised.exception), "sidecar lock record requires a string url"
        )

    def test_rejects_archive_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "sidecar.zip"
            destination = root / "ffmpeg"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("ffmpeg", b"unexpected")
            record = {
                "archive": {
                    "format": "zip",
                    "member": "ffmpeg",
                    "sha256": "0" * 64,
                }
            }

            with self.assertRaisesRegex(RuntimeError, "archive checksum mismatch"):
                provisioner.materialize_download(record, archive_path, destination)

            self.assertFalse(destination.exists())

    def test_verify_rejects_nonfree_or_unredistributable_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "ffmpeg"
            binary.write_bytes(b"fixture")
            expected_sha = digest(b"fixture")

            with mock.patch.object(
                provisioner.subprocess,
                "check_output",
                return_value="ffmpeg version 7.0\nconfiguration: --enable-nonfree\n",
            ):
                with self.assertRaisesRegex(RuntimeError, "nonfree sidecar"):
                    provisioner.verify(binary, expected_sha, "7.0")

            with mock.patch.object(
                provisioner.subprocess,
                "check_output",
                side_effect=[
                    "ffmpeg version 7.0\nconfiguration: --enable-gpl\n",
                    "This version is not legally redistributable.\n",
                ],
            ):
                with self.assertRaisesRegex(RuntimeError, "sidecar license"):
                    provisioner.verify(binary, expected_sha, "7.0")

            with mock.patch.object(
                provisioner.subprocess,
                "check_output",
                side_effect=[
                    "ffmpeg version 7.0\nconfiguration: --enable-gpl\n",
                    "GNU General Public License version 3 or later\n",
                ],
            ) as metadata:
                provisioner.verify(binary, expected_sha, "7.0")
                self.assertEqual(metadata.call_count, 2)

    def test_linux_sidecars_never_take_the_distribution_tool_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            binary_dir = Path(directory)
            with mock.patch.object(provisioner, "BIN_DIR", binary_dir):
                cases = [
                    ("ffmpeg", "x86_64-unknown-linux-gnu", "opentake-ffmpeg-x86_64-unknown-linux-gnu"),
                    ("ffprobe", "x86_64-unknown-linux-gnu", "opentake-ffprobe-x86_64-unknown-linux-gnu"),
                    ("ffmpeg", "aarch64-apple-darwin", "ffmpeg-aarch64-apple-darwin"),
                    ("ffprobe", "x86_64-apple-darwin", "ffprobe-x86_64-apple-darwin"),
                    ("ffmpeg", "x86_64-pc-windows-msvc", "ffmpeg-x86_64-pc-windows-msvc.exe"),
                ]
                for tool, target, name in cases:
                    with self.subTest(tool=tool, target=target):
                        self.assertEqual(
                            provisioner.destination(tool, target), binary_dir / name
                        )

    def test_tauri_external_binaries_match_the_provisioned_names(self) -> None:
        for config_name, target in (
            ("tauri.linux.conf.json", "x86_64-unknown-linux-gnu"),
            ("tauri.macos.conf.json", "aarch64-apple-darwin"),
            ("tauri.windows.conf.json", "x86_64-pc-windows-msvc"),
        ):
            with self.subTest(config=config_name):
                config = json.loads((ROOT / "src-tauri" / config_name).read_text())
                self.assertEqual(
                    config["bundle"]["externalBin"],
                    [
                        f"binaries/{provisioner.sidecar_name(tool, target)}"
                        for tool in ("ffmpeg", "ffprobe")
                    ],
                )

    def test_apple_silicon_lock_pins_archive_and_binary_hashes(self) -> None:
        lock = json.loads((ROOT / "scripts" / "ffmpeg-sidecars.lock.json").read_text())
        self.assertEqual(lock["schema"], "opentake-ffmpeg-sidecars-v2")
        target = lock["targets"]["aarch64-apple-darwin"]
        for tool in ("ffmpeg", "ffprobe"):
            record = target[tool]
            self.assertEqual(record["version"], "7.0")
            self.assertRegex(record["sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(record["archive"]["format"], "zip")
            self.assertEqual(record["archive"]["member"], tool)
            self.assertRegex(record["archive"]["sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
