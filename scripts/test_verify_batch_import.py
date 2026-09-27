"""Synthetic harness-output tests; these do not run or certify the Rust code."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import verify_batch_import as gate


def output(names):
    return "\n".join(f"test {name} ... ok" for name in names) + (
        f"\ntest result: ok. {len(names)} passed; 0 failed; 0 ignored; 0 measured; 10 filtered out; finished in 0.01s\n"
    )


def benchmark():
    return output((gate.BENCHMARK,)) + (
        "batch import count=2000 folder=true elapsed=10.5ms manifest_bytes=800000 undo_depth=0 version=0\n"
        "batch import count=5000 folder=false elapsed=20.5ms manifest_bytes=1800000 undo_depth=0 version=0\n"
    )


class EvidenceTests(unittest.TestCase):
    def test_all_named_regressions_pass(self):
        gate.verify_tests(output(gate.REGRESSIONS), gate.REGRESSIONS)

    def test_empty_filtered_run_fails(self):
        with self.assertRaises(ValueError):
            gate.verify_tests(output(()), gate.REGRESSIONS)

    def test_missing_wrong_duplicate_ignored_or_failed_tests_fail(self):
        good = output(gate.REGRESSIONS)
        for bad in (output(gate.REGRESSIONS[:-1]),
                    good.replace(gate.REGRESSIONS[0], "old::test"),
                    good + f"test {gate.REGRESSIONS[0]} ... ok\n",
                    good.replace("... ok", "... ignored", 1),
                    good.replace("... ok", "... FAILED", 1)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                gate.verify_tests(bad, gate.REGRESSIONS)

    def test_missing_duplicate_or_failed_summary_fails(self):
        good = output(gate.REGRESSIONS)
        for bad in (good.split("test result:")[0], good + good,
                    good.replace("0 failed", "1 failed"),
                    good.replace("test result: ok", "test result: FAILED")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                gate.verify_tests(bad, gate.REGRESSIONS)

    def test_benchmark_reports_both_sizes(self):
        rows = gate.verify_benchmark(benchmark())
        self.assertEqual([row["count"] for row in rows], [2000, 5000])
        self.assertAlmostEqual(rows[0]["elapsed_seconds"], 0.0105)

    def test_benchmark_accepts_rust_duration_units(self):
        for duration in ("10500000ns", "10500µs", "10500μs", "10500us", "0.0105s"):
            with self.subTest(duration=duration):
                rows = gate.verify_benchmark(benchmark().replace("10.5ms", duration))
                self.assertAlmostEqual(rows[0]["elapsed_seconds"], 0.0105)

    def test_missing_duplicate_or_wrong_measurements_fail(self):
        good = benchmark()
        for bad in (output((gate.BENCHMARK,)),
                    good.rsplit("batch import count=5000", 1)[0],
                    good + good[good.index("batch import count=2000"):],
                    good.replace("count=5000", "count=4000"),
                    good.replace("folder=true", "folder=false")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                gate.verify_benchmark(bad)

    def test_limits_are_not_relaxed(self):
        for before, after in (("10.5ms", "200ms"), ("10.5ms", "1s"),
                              ("800000", str(20 * 1024 * 1024)),
                              ("800000", "0"), ("undo_depth=0", "undo_depth=1"),
                              ("version=0", "version=1")):
            with self.subTest(after=after), self.assertRaises(ValueError):
                gate.verify_benchmark(benchmark().replace(before, after))

    def test_benchmark_cannot_use_regression_output(self):
        with self.assertRaises(ValueError):
            gate.verify_benchmark(output(gate.REGRESSIONS))

    def test_exact_clean_checkout_is_required(self):
        sha = "a" * 40
        with patch.object(gate, "git", side_effect=[sha, ""]):
            gate.verify_checkout(Path("."), sha)
        for replies in (["b" * 40], [sha, " M src/lib.rs"], [sha, "?? stale.rs"]):
            with self.subTest(replies=replies), patch.object(gate, "git", side_effect=replies):
                with self.assertRaises(ValueError):
                    gate.verify_checkout(Path("."), sha)

    def test_nonzero_process_exit_fails_even_with_success_text(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "run.log"
            result = subprocess.CompletedProcess(["cargo"], 1, output(gate.REGRESSIONS))
            with patch.object(gate.subprocess, "run", return_value=result), patch("builtins.print"):
                with self.assertRaises(subprocess.CalledProcessError):
                    gate.run(["cargo"], Path(directory), dict(os.environ), log)
            self.assertEqual(log.read_text(), result.stdout)


if __name__ == "__main__":
    unittest.main()
