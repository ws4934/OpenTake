#!/usr/bin/env python3
"""Mirror the checksum-pinned FFmpeg sidecar downloads into a release.

Every file named by `scripts/ffmpeg-sidecars.lock.json` is downloaded from its
upstream `url`, verified against the lock (archive SHA-256 when the record has
an archive, extracted-binary SHA-256 always) and uploaded unchanged as an
asset of the `ffmpeg-sidecars-v1` prerelease. The lock's `mirror_urls` point at
those assets, so the provisioner downloads from the mirror first.

The release is append-only: an existing asset with the same bytes is skipped,
an existing asset with different bytes fails the run, and nothing is ever
deleted or replaced. `.github/workflows/mirror-ffmpeg-sidecars.yml` runs this
script; `--dry-run` downloads and verifies without touching any release.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
from typing import Callable
import urllib.parse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provision_ffmpeg_sidecars as provisioner  # noqa: E402


MIRROR_REPOSITORY = "ws4934/OpenTake"
MIRROR_TAG = "ffmpeg-sidecars-v1"
SOURCE_NOTICE = "src-tauri/resources/ffmpeg/SOURCE.md"
TOOLS = ("ffmpeg", "ffprobe")

GhRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def asset_name(target: str, tool: str, record: dict[str, object]) -> str:
    """Deterministic release asset name: `<tool>-<target>` plus the extension
    of the upstream file (`.zip` for an archive, none for a raw binary)."""
    url = record.get("url")
    if not isinstance(url, str):
        raise RuntimeError(f"invalid sidecar lock record for {tool}/{target}")
    suffix = PurePosixPath(urllib.parse.urlsplit(url).path).suffix
    return f"{tool}-{target}{suffix}"


def mirror_url(
    target: str,
    tool: str,
    record: dict[str, object],
    repository: str = MIRROR_REPOSITORY,
    tag: str = MIRROR_TAG,
) -> str:
    return (
        f"https://github.com/{repository}/releases/download/{tag}/"
        f"{asset_name(target, tool, record)}"
    )


def lock_records(lock: dict[str, object]) -> list[tuple[str, str, dict[str, object]]]:
    targets = lock.get("targets")
    if not isinstance(targets, dict) or not targets:
        raise RuntimeError("sidecar lock has no targets")
    records = []
    for target, tools in sorted(targets.items()):
        for tool in TOOLS:
            record = tools.get(tool) if isinstance(tools, dict) else None
            if not isinstance(record, dict):
                raise RuntimeError(f"sidecar lock is missing {tool}/{target}")
            records.append((target, tool, record))
    return records


def check_lock_mirrors(lock: dict[str, object], repository: str, tag: str) -> None:
    """Fail unless every record lists this release's asset as its first
    mirror, so the uploaded names are exactly the ones the provisioner uses."""
    for target, tool, record in lock_records(lock):
        expected = mirror_url(target, tool, record, repository, tag)
        mirrors = record.get("mirror_urls")
        if not isinstance(mirrors, list) or not mirrors:
            raise RuntimeError(f"{tool}/{target} has no mirror_urls")
        first = mirrors[0]
        if not isinstance(first, str) or first.lower() != expected.lower():
            raise RuntimeError(
                f"{tool}/{target} mirror_urls[0] is {first!r}, expected {expected!r}"
            )


def fetch_upstream(
    target: str, tool: str, record: dict[str, object], directory: Path
) -> Path:
    """Download the upstream file into `directory/<asset name>` and verify it
    the way the provisioner does; return the unchanged downloaded file."""
    expected_sha = record.get("sha256")
    url = record.get("url")
    if not isinstance(expected_sha, str) or not isinstance(url, str):
        raise RuntimeError(f"invalid sidecar lock record for {tool}/{target}")
    original = directory / asset_name(target, tool, record)
    extracted = directory / f".{original.name}.extracted"
    if record.get("archive") is None:
        provisioner.fetch_from_source(
            tool, record, url, expected_sha, extracted, original
        )
    else:
        provisioner.fetch_from_source(
            tool, record, url, expected_sha, original, extracted
        )
        extracted.unlink()
    print(f"verified {original.name} from {url}")
    return original


def release_notes(lock: dict[str, object], repository: str) -> str:
    lines = [
        "Byte-identical mirror of the checksum-pinned FFmpeg sidecar downloads.",
        "",
        "`scripts/provision_ffmpeg_sidecars.py` downloads these assets first and falls",
        "back to the upstream URLs. Provenance, build configuration and licence terms:",
        f"https://github.com/{repository}/blob/main/{SOURCE_NOTICE}",
        "",
        "| Asset | Upstream URL | SHA-256 of the asset |",
        "|---|---|---|",
    ]
    for target, tool, record in lock_records(lock):
        archive = record.get("archive")
        asset_sha = archive.get("sha256") if isinstance(archive, dict) else record.get("sha256")
        lines.append(
            f"| `{asset_name(target, tool, record)}` | {record.get('url')} | `{asset_sha}` |"
        )
    lines += [
        "",
        "This prerelease is append-only and is never marked as the latest release.",
    ]
    return "\n".join(lines) + "\n"


def run_gh(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["gh", *arguments], capture_output=True, text=True)


def checked(result: subprocess.CompletedProcess[str], action: str) -> str:
    if result.returncode != 0:
        raise RuntimeError(f"{action} failed: {result.stderr.strip()}")
    return result.stdout


def publish(
    files: dict[str, Path],
    repository: str,
    tag: str,
    target_commit: str,
    notes: str,
    work_dir: Path,
    gh: GhRunner = run_gh,
) -> None:
    """Create the prerelease when it is missing and upload the missing assets.
    Existing assets are compared by SHA-256 and never deleted or replaced."""
    view = ["release", "view", tag, "--repo", repository, "--json",
            "tagName,isDraft,isPrerelease,assets"]
    existing = gh(view)
    if existing.returncode != 0:
        if "not found" not in existing.stderr.lower():
            raise RuntimeError(f"release lookup failed: {existing.stderr.strip()}")
        notes_path = work_dir / "release-notes.md"
        notes_path.write_text(notes, encoding="utf-8")
        checked(
            gh([
                "release", "create", tag, "--repo", repository,
                "--target", target_commit,
                "--title", "FFmpeg sidecar mirror (v1)",
                "--notes-file", str(notes_path),
                "--prerelease", "--latest=false",
            ]),
            f"creating release {tag}",
        )
        existing = gh(view)
    release = json.loads(checked(existing, f"reading release {tag}"))
    if release.get("tagName") != tag or release.get("isDraft") or not release.get("isPrerelease"):
        raise RuntimeError(f"release {tag} must be a published prerelease")
    present = {asset["name"] for asset in release.get("assets", [])}

    for name, path in sorted(files.items()):
        expected_sha = provisioner.sha256(path)
        if name in present:
            download_dir = work_dir / "existing" / name
            download_dir.mkdir(parents=True)
            checked(
                gh([
                    "release", "download", tag, "--repo", repository,
                    "--pattern", name, "--dir", str(download_dir),
                ]),
                f"downloading existing asset {name}",
            )
            actual_sha = provisioner.sha256(download_dir / name)
            if actual_sha != expected_sha:
                raise RuntimeError(
                    f"existing asset {name} differs from the pinned file: "
                    f"{actual_sha} != {expected_sha}; refusing to replace it"
                )
            print(f"kept {name} (sha256 {expected_sha})")
            continue
        checked(
            gh(["release", "upload", tag, str(path), "--repo", repository]),
            f"uploading {name}",
        )
        print(f"uploaded {name} (sha256 {expected_sha})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", MIRROR_REPOSITORY))
    parser.add_argument("--tag", default=MIRROR_TAG)
    parser.add_argument("--target-commit", default=os.environ.get("GITHUB_SHA"))
    parser.add_argument("--dry-run", action="store_true", help="download and verify only")
    args = parser.parse_args()

    lock = json.loads(provisioner.LOCK_PATH.read_text(encoding="utf-8"))
    check_lock_mirrors(lock, args.repository, args.tag)
    with tempfile.TemporaryDirectory(prefix="opentake-sidecar-mirror-") as directory:
        work_dir = Path(directory)
        assets_dir = work_dir / "assets"
        assets_dir.mkdir()
        files = {
            path.name: path
            for target, tool, record in lock_records(lock)
            for path in [fetch_upstream(target, tool, record, assets_dir)]
        }
        if args.dry_run:
            return 0
        if not args.target_commit:
            raise RuntimeError("--target-commit (or GITHUB_SHA) is required")
        publish(
            files,
            args.repository,
            args.tag,
            args.target_commit,
            release_notes(lock, args.repository),
            work_dir,
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
