#!/usr/bin/env python3
"""Unit tests for scripts/pr_native_scope.py."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pr_native_scope as scope  # noqa: E402

ROOT = "/work/repo"


def package(name: str, directory: str, *dependencies: str) -> dict:
    return {
        "id": f"{name} 1.0.0",
        "name": name,
        "manifest_path": f"{ROOT}/{directory}/Cargo.toml",
        "dependencies": [{"name": dependency} for dependency in dependencies] + [{"name": "serde"}],
    }


METADATA = {
    "workspace_root": ROOT,
    "workspace_members": [
        "domain 1.0.0",
        "media 1.0.0",
        "media-extra 1.0.0",
        "core 1.0.0",
        "app 1.0.0",
    ],
    "packages": [
        package("domain", "crates/domain"),
        package("media", "crates/media", "domain"),
        package("media-extra", "crates/media/extra", "media"),
        package("core", "crates/core", "domain"),
        package("app", "app", "media", "core"),
    ],
}


class AffectedPackagesTests(unittest.TestCase):
    def affected(self, *files: str) -> list[str]:
        return scope.affected_packages(list(files), METADATA)

    def test_changed_crate_includes_its_dependents(self) -> None:
        self.assertEqual(self.affected("crates/media/src/lib.rs"), ["app", "media", "media-extra"])
        self.assertEqual(self.affected("app/src/main.rs"), ["app"])

    def test_leaf_dependency_change_reaches_every_dependent(self) -> None:
        self.assertEqual(
            self.affected("crates/domain/src/clip.rs"),
            ["app", "core", "domain", "media", "media-extra"],
        )

    def test_nested_crate_owns_its_files(self) -> None:
        self.assertEqual(self.affected("crates/media/extra/src/lib.rs"), ["media-extra"])

    def test_files_outside_crates_select_nothing(self) -> None:
        self.assertEqual(self.affected("web/src/App.tsx", "README.md", "docs/guide.md"), [])
        self.assertEqual(self.affected(), [])

    def test_workspace_wide_inputs_select_every_crate(self) -> None:
        everything = ["app", "core", "domain", "media", "media-extra"]
        for path in (
            "Cargo.lock",
            "Cargo.toml",
            ".cargo/config.toml",
            ".github/workflows/pr-native.yml",
            "scripts/ffmpeg-sidecars.lock.json",
        ):
            self.assertEqual(self.affected(path), everything, path)

    def test_crate_manifest_is_owned_by_its_crate(self) -> None:
        self.assertEqual(self.affected("crates/core/Cargo.toml"), ["app", "core"])


class MainTests(unittest.TestCase):
    def test_outputs_github_step_outputs(self) -> None:
        import contextlib
        import io
        from unittest import mock

        output = io.StringIO()
        with mock.patch.object(scope, "cargo_metadata", return_value=METADATA), contextlib.redirect_stdout(
            output
        ), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(scope.main(["--files", "crates/core/src/lib.rs"]), 0)
        self.assertEqual(output.getvalue(), "native=true\npackages=-p app -p core\n")

        output = io.StringIO()
        with mock.patch.object(scope, "cargo_metadata", return_value=METADATA), contextlib.redirect_stdout(
            output
        ), contextlib.redirect_stderr(io.StringIO()):
            scope.main(["--files", "web/src/App.tsx"])
        self.assertEqual(output.getvalue(), "native=false\npackages=\n")


if __name__ == "__main__":
    unittest.main()
