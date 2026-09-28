from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest


ROOT = Path(
    os.environ.get("OPENTAKE_REPOSITORY_ROOT", Path(__file__).resolve().parents[2])
).resolve()
sys.path.insert(0, str(ROOT / "scripts"))
from workflow_yaml import parse_workflow_yaml  # noqa: E402

WORKFLOWS = ROOT / ".github" / "workflows"
CACHE_PIN = "0057852bfaa89a56745cba8c7296529d2fc39830"
RESTORE = f"actions/cache/restore@{CACHE_PIN}"
SAVE = f"actions/cache/save@{CACHE_PIN}"
LOCK_HASH = (
    "${{ hashFiles('scripts/ffmpeg-sidecars.lock.json', "
    "'scripts/provision_ffmpeg_sidecars.py') }}"
)
SAVE_WITH = {
    "path": "src-tauri/binaries",
    "key": "${{ steps.sidecar-cache.outputs.cache-primary-key }}",
}


def workflow(name: str) -> dict[str, object]:
    return parse_workflow_yaml((WORKFLOWS / name).read_text(encoding="utf-8"))


def step_index(steps: list[dict[str, object]], name: str) -> int:
    matches = [index for index, step in enumerate(steps) if step.get("name") == name]
    if len(matches) != 1:
        raise AssertionError(f"expected one step named {name!r}, found {len(matches)}")
    return matches[0]


class FfmpegSidecarCacheWorkflowTests(unittest.TestCase):
    def assert_restore_provision_save(
        self, steps: list[dict[str, object]], provision: str, key: str
    ) -> None:
        restore = step_index(steps, "Restore pinned FFmpeg sidecars")
        save = step_index(steps, "Save pinned FFmpeg sidecars")
        provisioned = step_index(steps, provision)
        self.assertEqual(steps[restore]["uses"], RESTORE)
        self.assertEqual(steps[restore]["id"], "sidecar-cache")
        self.assertEqual(
            steps[restore]["with"], {"path": "src-tauri/binaries", "key": key}
        )
        self.assertEqual(save, provisioned + 1, "save right after provisioning")
        self.assertLess(restore, provisioned)
        self.assertEqual(steps[save]["uses"], SAVE)
        self.assertEqual(
            steps[save]["if"], "steps.sidecar-cache.outputs.cache-hit != 'true'"
        )
        self.assertEqual(steps[save]["with"], SAVE_WITH)
        self.assertFalse(
            any(str(step.get("uses", "")).startswith("actions/cache@")
                and step.get("with", {}).get("path") == "src-tauri/binaries"
                for step in steps),
            "the combined cache action saves only when the whole job succeeds",
        )

    def test_pr_workflow_restores_and_saves_the_linux_sidecars(self) -> None:
        steps = workflow("pr.yml")["jobs"]["rust"]["steps"]
        self.assert_restore_provision_save(
            steps,
            "Provision pinned FFmpeg",
            f"ffmpeg-sidecars-${{{{ runner.os }}}}-x86_64-unknown-linux-gnu-{LOCK_HASH}",
        )

    def test_native_workflow_restores_and_saves_the_host_sidecars(self) -> None:
        steps = workflow("pr-native.yml")["jobs"]["native"]["steps"]
        self.assert_restore_provision_save(
            steps,
            "Provision pinned FFmpeg",
            "ffmpeg-sidecars-${{ runner.os }}-${{ steps.host.outputs.triple }}-"
            + LOCK_HASH,
        )

    def test_main_seeds_the_keys_the_pull_request_workflows_restore(self) -> None:
        seed = workflow("ffmpeg-sidecar-cache.yml")
        self.assertEqual(
            seed["on"]["push"],
            {
                "branches": ["main"],
                "paths": [
                    "scripts/ffmpeg-sidecars.lock.json",
                    "scripts/provision_ffmpeg_sidecars.py",
                    ".github/workflows/ffmpeg-sidecar-cache.yml",
                ],
            },
        )
        self.assertNotIn("pull_request", seed["on"])
        self.assertEqual(seed["permissions"], {"contents": "read"})
        job = seed["jobs"]["seed"]
        self.assertEqual(job["if"], "github.ref == 'refs/heads/main'")
        self.assertNotIn("permissions", job)
        self.assertEqual(job["steps"][0]["with"], {"persist-credentials": False})
        # The runners and Rust host targets of pr.yml (Linux) and pr-native.yml.
        native = workflow("pr-native.yml")["jobs"]["native"]["strategy"]["matrix"]["os"]
        self.assertEqual(
            job["strategy"]["matrix"]["include"],
            [
                {"os": native[0], "target": "x86_64-unknown-linux-gnu"},
                {"os": native[1], "target": "x86_64-pc-windows-msvc"},
                {"os": native[2], "target": "aarch64-apple-darwin"},
            ],
        )
        self.assert_restore_provision_save(
            job["steps"],
            "Provision pinned FFmpeg",
            f"ffmpeg-sidecars-${{{{ runner.os }}}}-${{{{ matrix.target }}}}-{LOCK_HASH}",
        )

    def test_every_action_is_pinned_by_sha(self) -> None:
        for name in ("pr.yml", "pr-native.yml", "ffmpeg-sidecar-cache.yml"):
            with self.subTest(workflow=name):
                for job in workflow(name)["jobs"].values():
                    for step in job.get("steps", []):
                        if "uses" in step:
                            self.assertRegex(step["uses"], r"^[\w./-]+@[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()
