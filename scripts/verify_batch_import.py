#!/usr/bin/env python3
"""Run issue #35's named regressions and release benchmark, rejecting empty runs.

This is a focused core gate, not a substitute for workspace/Tauri or GUI tests.
Build output is isolated from other checkouts; evidence records the exact HEAD.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile

PREFIX = "core::import_tests::"
REGRESSIONS = tuple(PREFIX + name for name in (
    "folders_and_files_preserve_prior_undo_and_redo_without_adding_history",
    "nonexistent_existing_folder_and_parent_reject_the_whole_batch",
    "late_invalid_file_preserves_document_version_and_owned_history",
    "repeated_sources_reuse_ids_metadata_and_apply_the_requested_folder",
    "invalid_planned_folder_keys_and_empty_names_are_atomic",
    "final_identity_failure_only_retracts_the_new_entry_and_preserves_duplicates",
    "batch_and_single_stems_share_provenance_and_invalid_stems_roll_back",
))
BENCHMARK = PREFIX + "release_batch_import_is_linear_and_does_not_retain_catalog_snapshots"
RESULT = re.compile(r"^test (\S+) \.\.\. (ok|FAILED|ignored(?:,.*)?)$", re.MULTILINE)
SUMMARY = re.compile(
    r"^test result: (ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;",
    re.MULTILINE,
)
MEASUREMENT = re.compile(
    r"batch import count=(\d+) folder=(true|false) "
    r"elapsed=(\d+(?:\.\d+)?)(ns|µs|μs|us|ms|s) "
    r"manifest_bytes=(\d+) undo_depth=(\d+) version=(\d+)"
)
SCALE = {"ns": 1e-9, "µs": 1e-6, "μs": 1e-6, "us": 1e-6, "ms": 1e-3, "s": 1.0}


def verify_tests(output: str, expected: tuple[str, ...]) -> None:
    """Require every named test and exactly one successful libtest summary."""
    records = RESULT.findall(output)
    wanted = [(name, "ok") for name in expected]
    if sorted(records) != sorted(wanted):
        raise ValueError("named tests missing, duplicated, ignored, unexpected, or failed")
    summaries = SUMMARY.findall(output)
    if summaries != [("ok", str(len(expected)), "0", "0")]:
        raise ValueError("expected one nonempty successful libtest summary")


def verify_benchmark(output: str) -> list[dict[str, object]]:
    verify_tests(output, (BENCHMARK,))
    rows = []
    for count, folder, duration, unit, size, undo, version in MEASUREMENT.findall(output):
        elapsed = float(duration) * SCALE[unit]
        if not 0 <= elapsed < 0.2 or not 0 < int(size) < 20 * 1024 * 1024:
            raise ValueError("release benchmark exceeded time or catalog-size limit")
        if int(undo) != 0 or int(version) != 0:
            raise ValueError("batch import changed undo depth or document version")
        rows.append({"count": int(count), "folder": folder == "true",
                     "elapsed_seconds": elapsed, "manifest_bytes": int(size),
                     "undo_depth": int(undo), "version": int(version)})
    if sorted((row["count"], row["folder"]) for row in rows) != [(2000, True), (5000, False)]:
        raise ValueError("expected exactly the 2000/folder and 5000/no-folder measurements")
    return rows


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def verify_checkout(repo: Path, expected_sha: str) -> None:
    if git(repo, "rev-parse", "HEAD") != expected_sha:
        raise ValueError("HEAD does not match the expected immutable commit")
    if git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("verification requires a clean checkout")


def run(command: list[str], repo: Path, env: dict[str, str], log: Path) -> str:
    print("+ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=repo, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            encoding="utf-8", errors="replace", check=False)
    log.write_text(result.stdout, encoding="utf-8")
    print(result.stdout, end="", flush=True)
    result.check_returncode()
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha", required=True, help="full 40-hex candidate commit")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="new evidence directory outside the checkout")
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-fA-F]{40}", args.expected_sha) is None:
        parser.error("--expected-sha must be an immutable 40-hex commit")
    repo = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if output == repo or repo in output.parents:
        parser.error("--output-dir must be outside the checkout")
    output.mkdir(parents=True, exist_ok=False)
    sha = args.expected_sha.lower()
    evidence: dict[str, object] = {
        "candidate": sha, "platform": platform.platform(), "status": "failed",
        "scope": "issue-35 core regressions and release benchmark only",
        "memory_evidence": "catalog size and undo depth; not process RSS",
    }
    try:
        verify_checkout(repo, sha)
        # Never reuse target artifacts from another commit or worktree.
        with tempfile.TemporaryDirectory(prefix="opentake-batch-import-") as target:
            env = dict(os.environ, CARGO_TARGET_DIR=target,
                       CARGO_INCREMENTAL="0", CARGO_TERM_COLOR="never")
            run(["cargo", "--version"], repo, env, output / "toolchain.log")
            regression = run([
                "cargo", "test", "--locked", "-p", "opentake-core", "--lib", PREFIX,
                "--", "--skip", BENCHMARK, "--test-threads=1", "--color=never",
            ], repo, env, output / "regressions.log")
            verify_tests(regression, REGRESSIONS)
            evidence["regressions_passed"] = len(REGRESSIONS)
            benchmark = run([
                "cargo", "test", "--locked", "--release", "-p", "opentake-core",
                "--lib", BENCHMARK, "--", "--exact", "--ignored", "--show-output",
                "--test-threads=1", "--color=never",
            ], repo, env, output / "benchmark.log")
            evidence["measurements"] = verify_benchmark(benchmark)
        verify_checkout(repo, sha)
        evidence["status"] = "passed"
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        evidence["error"] = str(error)
        print(f"Verification failed: {error}", flush=True)
    finally:
        (output / "result.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Evidence: {output}", flush=True)
    return 0 if evidence["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
