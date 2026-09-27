"""Test desktop tool resolution without compiling Tauri or the workspace."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class MediaToolsTests(unittest.TestCase):
    def test_standalone_rust_resolver(self) -> None:
        target = subprocess.check_output(
            ["rustc", "--print", "host-tuple"], text=True
        ).strip()
        environment = {
            **os.environ,
            "CARGO_MANIFEST_DIR": str(ROOT / "src-tauri"),
            "OPENTAKE_BUILD_TARGET": target,
        }
        with tempfile.TemporaryDirectory(prefix="opentake-media-tools-") as directory:
            executable = Path(directory) / ("tests.exe" if os.name == "nt" else "tests")
            subprocess.run(
                ["rustc", "--edition=2021", "--test", "-D", "warnings",
                 str(ROOT / "src-tauri/src/media_tools.rs"), "-o", str(executable)],
                env=environment, check=True,
            )
            subprocess.run([str(executable)], check=True)
