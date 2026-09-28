from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
import sys
import zipfile


ROOT = Path(
    os.environ.get("OPENTAKE_REPOSITORY_ROOT", Path(__file__).resolve().parents[2])
).resolve()
MODULE_PATH = ROOT / "scripts" / "mirror_ffmpeg_sidecars.py"
SPEC = importlib.util.spec_from_file_location("mirror_ffmpeg_sidecars", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
mirror = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mirror)

sys.path.insert(0, str(ROOT / "scripts"))
from workflow_yaml import parse_workflow_yaml  # noqa: E402

WORKFLOW_PATH = ROOT / ".github" / "workflows" / "mirror-ffmpeg-sidecars.yml"
TAG = "ffmpeg-sidecars-v1"
REPOSITORY = "ws4934/OpenTake"


def lock() -> dict[str, object]:
    return json.loads((ROOT / "scripts" / "ffmpeg-sidecars.lock.json").read_text())


class FakeGh:
    """Records `gh` invocations and serves a release held in memory."""

    def __init__(self, release: dict[str, object] | None, assets: dict[str, bytes]):
        self.release = release
        self.assets = dict(assets)
        self.calls: list[list[str]] = []

    def __call__(self, arguments: list[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(arguments)
        command = arguments[:2]

        def done(stdout: str = "", code: int = 0, stderr: str = ""):
            return subprocess.CompletedProcess(arguments, code, stdout, stderr)

        if command == ["release", "view"]:
            if self.release is None:
                return done(code=1, stderr="release not found")
            assets = [{"name": name} for name in self.assets]
            return done(json.dumps({**self.release, "assets": assets}))
        if command == ["release", "create"]:
            self.release = {
                "tagName": arguments[2],
                "isDraft": False,
                "isPrerelease": "--prerelease" in arguments,
            }
            return done()
        if command == ["release", "download"]:
            name = arguments[arguments.index("--pattern") + 1]
            directory = Path(arguments[arguments.index("--dir") + 1])
            (directory / name).write_bytes(self.assets[name])
            return done()
        if command == ["release", "upload"]:
            path = Path(arguments[3])
            if path.name in self.assets:
                return done(code=1, stderr="asset already exists")
            self.assets[path.name] = path.read_bytes()
            return done()
        raise AssertionError(f"unexpected gh call: {arguments}")


class MirrorFfmpegSidecarsTests(unittest.TestCase):
    def test_every_lock_record_mirrors_the_asset_the_workflow_uploads(self) -> None:
        data = lock()
        records = mirror.lock_records(data)
        self.assertEqual(len(records), 8)
        names = set()
        for target, tool, record in records:
            with self.subTest(target=target, tool=tool):
                name = mirror.asset_name(target, tool, record)
                names.add(name)
                archive = record.get("archive")
                pin = archive["sha256"] if archive else record["sha256"]
                suffix = ".zip" if archive else ""
                self.assertEqual(name, f"{tool}-{target}-{pin[:16]}{suffix}")
                self.assertEqual(
                    record["mirror_urls"],
                    [
                        f"https://github.com/{REPOSITORY}/releases/download/"
                        f"{TAG}/{name}"
                    ],
                )
                self.assertEqual(
                    name.endswith(".zip"), record.get("archive") is not None
                )
        self.assertEqual(len(names), len(records), "asset names are unique")
        mirror.check_lock_mirrors(data, REPOSITORY, TAG)

    def test_lock_check_rejects_a_record_pointing_elsewhere(self) -> None:
        data = lock()
        record = data["targets"]["x86_64-unknown-linux-gnu"]["ffmpeg"]
        record["mirror_urls"] = ["https://example.invalid/ffmpeg"]
        with self.assertRaisesRegex(RuntimeError, "mirror_urls\\[0\\]"):
            mirror.check_lock_mirrors(data, REPOSITORY, TAG)
        del record["mirror_urls"]
        with self.assertRaisesRegex(RuntimeError, "has no mirror_urls"):
            mirror.check_lock_mirrors(data, REPOSITORY, TAG)

    def test_lock_check_rejects_a_pin_bump_that_keeps_the_old_mirror(self) -> None:
        for target, tool, pin_path in (
            ("x86_64-unknown-linux-gnu", "ffmpeg", ("sha256",)),
            ("aarch64-apple-darwin", "ffprobe", ("archive", "sha256")),
        ):
            with self.subTest(target=target, tool=tool):
                data = lock()
                record = data["targets"][target][tool]
                pinned = record
                for key in pin_path[:-1]:
                    pinned = pinned[key]
                pinned[pin_path[-1]] = "f" * 64
                with self.assertRaisesRegex(RuntimeError, "mirror_urls\\[0\\]"):
                    mirror.check_lock_mirrors(data, REPOSITORY, TAG)

    def test_lock_check_rejects_a_later_mirror_with_a_stale_name(self) -> None:
        data = lock()
        record = data["targets"]["x86_64-apple-darwin"]["ffmpeg"]
        record["mirror_urls"].append(
            "https://other.invalid/ffmpeg-x86_64-apple-darwin-0000000000000000"
        )
        with self.assertRaisesRegex(RuntimeError, "does not name the pinned asset"):
            mirror.check_lock_mirrors(data, REPOSITORY, TAG)

    def test_workflow_writes_only_from_main_with_job_scoped_permission(self) -> None:
        workflow = parse_workflow_yaml(WORKFLOW_PATH.read_text(encoding="utf-8"))
        self.assertEqual(list(workflow["on"]), ["workflow_dispatch"])
        self.assertEqual(workflow["permissions"], {"contents": "read"})
        self.assertEqual(list(workflow["jobs"]), ["mirror"])
        job = workflow["jobs"]["mirror"]
        self.assertEqual(job["if"], "github.ref == 'refs/heads/main'")
        self.assertEqual(job["permissions"], {"contents": "write"})
        checkout = job["steps"][0]
        self.assertEqual(checkout["with"], {"persist-credentials": False})

    def test_workflow_uses_the_script_tag_and_never_marks_latest(self) -> None:
        text = WORKFLOW_PATH.read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", text)
        self.assertNotIn("pull_request", text)
        self.assertIn("python3 -B scripts/mirror_ffmpeg_sidecars.py", text)
        self.assertEqual(mirror.MIRROR_TAG, TAG)
        self.assertEqual(mirror.MIRROR_REPOSITORY, REPOSITORY)
        for line in text.splitlines():
            if "uses:" in line:
                self.assertRegex(line, r"uses: [\w./-]+@[0-9a-f]{40}\b")

    def publish(self, gh: FakeGh, files: dict[str, bytes]) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory)
            paths = {}
            for name, data in files.items():
                path = work_dir / name
                path.write_bytes(data)
                paths[name] = path
            mirror.publish(paths, REPOSITORY, TAG, "0" * 40, "notes", work_dir, gh)

    def test_missing_release_is_created_as_a_prerelease_that_is_not_latest(
        self,
    ) -> None:
        gh = FakeGh(None, {})
        self.publish(gh, {"ffmpeg-a.zip": b"a", "ffprobe-a.zip": b"b"})

        create = next(call for call in gh.calls if call[:2] == ["release", "create"])
        self.assertIn("--prerelease", create)
        self.assertIn("--latest=false", create)
        self.assertEqual(gh.assets, {"ffmpeg-a.zip": b"a", "ffprobe-a.zip": b"b"})

    def test_identical_assets_are_kept_and_missing_ones_uploaded(self) -> None:
        release = {"tagName": TAG, "isDraft": False, "isPrerelease": True}
        gh = FakeGh(release, {"ffmpeg-a.zip": b"a"})
        self.publish(gh, {"ffmpeg-a.zip": b"a", "ffprobe-a.zip": b"b"})

        uploads = [call for call in gh.calls if call[:2] == ["release", "upload"]]
        self.assertEqual([Path(call[3]).name for call in uploads], ["ffprobe-a.zip"])
        self.assertFalse(any(call[:2] == ["release", "create"] for call in gh.calls))
        self.assertFalse(any("--clobber" in call for call in gh.calls))
        self.assertFalse(any("delete" in call or "delete-asset" in call for call in gh.calls))

    def test_existing_asset_with_different_bytes_fails_without_replacing(self) -> None:
        release = {"tagName": TAG, "isDraft": False, "isPrerelease": True}
        gh = FakeGh(release, {"ffmpeg-a.zip": b"other"})
        with self.assertRaisesRegex(RuntimeError, "refusing to replace"):
            self.publish(gh, {"ffmpeg-a.zip": b"a"})

        self.assertEqual(gh.assets, {"ffmpeg-a.zip": b"other"})
        self.assertFalse(any(call[:2] == ["release", "upload"] for call in gh.calls))

    def test_release_that_is_not_a_prerelease_is_rejected(self) -> None:
        gh = FakeGh({"tagName": TAG, "isDraft": False, "isPrerelease": False}, {})
        with self.assertRaisesRegex(RuntimeError, "published prerelease"):
            self.publish(gh, {"ffmpeg-a.zip": b"a"})

    def test_upstream_archive_is_kept_unchanged_after_verification(self) -> None:
        binary = b"pinned ffprobe"
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory)
            source = work_dir / "source.zip"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("ffprobe", binary)
            archive_bytes = source.read_bytes()
            record = {
                "url": "https://upstream.invalid/ffprobe7arm.zip",
                "sha256": hashlib.sha256(binary).hexdigest(),
                "version": "7.0",
                "archive": {
                    "format": "zip",
                    "member": "ffprobe",
                    "sha256": hashlib.sha256(archive_bytes).hexdigest(),
                },
            }
            assets = work_dir / "assets"
            assets.mkdir()
            with mock.patch.object(
                mirror.provisioner,
                "download",
                lambda _url, path: path.write_bytes(archive_bytes),
            ):
                path = mirror.fetch_upstream(
                    "aarch64-apple-darwin", "ffprobe", record, assets
                )

            self.assertEqual(
                path.name,
                "ffprobe-aarch64-apple-darwin-"
                f"{hashlib.sha256(archive_bytes).hexdigest()[:16]}.zip",
            )
            self.assertEqual(path.read_bytes(), archive_bytes)
            self.assertEqual([item.name for item in assets.iterdir()], [path.name])

    def test_upstream_raw_binary_is_fetched_again_and_kept_unchanged(self) -> None:
        binary = b"pinned ffmpeg"
        pin = hashlib.sha256(binary).hexdigest()
        record = {
            "url": "https://upstream.invalid/b6.1.1/ffmpeg-linux-x64",
            "sha256": pin,
            "version": "6.1.1",
        }
        responses = [b"replaced upstream", binary]
        requested: list[str] = []

        def download_fixture(url: str, path: Path) -> None:
            requested.append(url)
            path.write_bytes(responses.pop(0))

        with tempfile.TemporaryDirectory() as directory:
            assets = Path(directory)
            with (
                mock.patch.object(mirror.provisioner, "download", download_fixture),
                mock.patch.object(mirror.provisioner.time, "sleep"),
                mock.patch.object(mirror.provisioner.sys, "stderr", mock.MagicMock()),
            ):
                path = mirror.fetch_upstream(
                    "x86_64-unknown-linux-gnu", "ffmpeg", record, assets
                )

            self.assertEqual(path.name, f"ffmpeg-x86_64-unknown-linux-gnu-{pin[:16]}")
            self.assertEqual(path.read_bytes(), binary)
            self.assertEqual([item.name for item in assets.iterdir()], [path.name])
        self.assertEqual(requested, [record["url"]] * 2)
        self.assertEqual(responses, [])


if __name__ == "__main__":
    unittest.main()
