#!/usr/bin/env python3
"""Select the Rust crates a pull request affects, for the native PR checks.

The lightweight PR gate (pr.yml) builds only a few crates on Linux. This script
decides whether `.github/workflows/pr-native.yml` must build and test the
workspace on Linux, Windows and macOS, and which packages to test: every
workspace crate with a changed file, plus every crate that depends on one.

Usage:
  python3 scripts/pr_native_scope.py --base <sha> --head <sha>
  python3 scripts/pr_native_scope.py --files crates/opentake-media/src/lib.rs

Prints `native=true|false` and `packages=<-p flags>` lines, suitable for
appending to "$GITHUB_OUTPUT".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

# Changes to these paths can affect every crate: dependency resolution, the
# toolchain, the bundled FFmpeg the build script embeds, or this check itself.
GLOBAL_PATHS = (
    "Cargo.lock",
    "Cargo.toml",
    "rust-toolchain.toml",
    ".cargo/",
    ".github/workflows/pr-native.yml",
    "scripts/pr_native_scope.py",
    "scripts/provision_ffmpeg_sidecars.py",
    "scripts/ffmpeg-sidecars.lock.json",
)


def workspace_graph(metadata: dict) -> tuple[dict[str, PurePosixPath], dict[str, set[str]]]:
    """Return (package -> directory relative to the root, package -> workspace dependents)."""
    root = PurePosixPath(Path(metadata["workspace_root"]).as_posix())
    members = set(metadata["workspace_members"])
    packages = [package for package in metadata["packages"] if package["id"] in members]
    names = {package["name"] for package in packages}
    directories = {
        package["name"]: PurePosixPath(Path(package["manifest_path"]).parent.as_posix()).relative_to(root)
        for package in packages
    }
    dependents: dict[str, set[str]] = {name: set() for name in names}
    for package in packages:
        for dependency in package["dependencies"]:
            if dependency["name"] in names:
                dependents[dependency["name"]].add(package["name"])
    return directories, dependents


def owning_package(path: str, directories: dict[str, PurePosixPath]) -> str | None:
    """The workspace package whose directory contains `path` (the deepest one wins)."""
    candidate = PurePosixPath(path)
    owners = [
        (len(directory.parts), name)
        for name, directory in directories.items()
        if directory.parts and candidate.parts[: len(directory.parts)] == directory.parts
    ]
    return max(owners)[1] if owners else None


def affected_packages(files: list[str], metadata: dict) -> list[str]:
    directories, dependents = workspace_graph(metadata)
    if any(path == entry or (entry.endswith("/") and path.startswith(entry)) for path in files for entry in GLOBAL_PATHS):
        return sorted(directories)
    changed = {owner for path in files if (owner := owning_package(path, directories))}
    affected = set(changed)
    pending = list(changed)
    while pending:
        for dependent in dependents[pending.pop()]:
            if dependent not in affected:
                affected.add(dependent)
                pending.append(dependent)
    return sorted(affected)


def changed_files(base: str, head: str) -> list[str]:
    output = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...{head}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line for line in output.splitlines() if line]


def cargo_metadata() -> dict:
    output = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps", "--locked"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base", help="base commit of the pull request")
    source.add_argument("--files", nargs="*", help="changed paths, relative to the repository root")
    parser.add_argument("--head", default="HEAD", help="head commit of the pull request")
    args = parser.parse_args(argv)

    files = args.files if args.files is not None else changed_files(args.base, args.head)
    packages = affected_packages(files, cargo_metadata())
    print(f"native={'true' if packages else 'false'}")
    print(f"packages={' '.join(f'-p {name}' for name in packages)}")
    print(f"affected: {', '.join(packages) or 'none'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
