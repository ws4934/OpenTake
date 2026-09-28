#!/usr/bin/env python3
"""Unit tests for scripts/check_pr_message.py."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_pr_message as check  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GOOD_TITLE = "fix(export): keep the old movie until the new export succeeds (#28)"
PARTIAL_TITLE = "perf(export): stream forward video layers during export (partial #28)"
GOOD_BODY = """## Linked issues
Closes #28

## Summary
- Export writes to an exclusive temporary file and replaces the destination only after verification.

## Root cause
The destination was opened with O_TRUNC and deleted on cancel, so re-exporting lost the old movie.

## Implementation notes
- Windows uses `ReplaceFileW`; Unix uses `renameat`.

## Testing
- [x] `cargo test --locked -p opentake-tauri export` (18 passed)
- [x] Manual check on Windows 11: exported over an existing movie, then cancelled; the old movie stayed.

## Risk and rollback
Low; revert this commit.
"""
FOLLOW_UPS = "\n## Follow-ups\n- Stream audio layers as well.\n"


def run_main(argv: list[str], github_actions: str = "") -> tuple[int, str]:
    output = io.StringIO()
    with mock.patch.dict(os.environ, {"GITHUB_ACTIONS": github_actions}), contextlib.redirect_stdout(output):
        code = check.main(argv)
    return code, output.getvalue()


def has_error(errors: list[str], text: str) -> bool:
    return any(text in error for error in errors)


class HeaderTests(unittest.TestCase):
    def test_accepts_conventional_english_title(self) -> None:
        self.assertEqual(check.check_header(GOOD_TITLE), [])

    def test_accepts_multiple_scopes_breaking_change_and_partial_suffix(self) -> None:
        self.assertEqual(check.check_header("feat(agent,web)!: require host approval for paid generation"), [])
        self.assertEqual(check.check_header("perf(export): stream forward video layers (partial #3)"), [])
        self.assertEqual(check.check_header("fix(tauri): unblock saves during proxy creation (#64, #97)"), [])
        self.assertEqual(check.check_header("docs: add agent message conventions"), [])

    def test_rejects_title_without_type(self) -> None:
        errors = check.check_header("Preserve existing movie until replacement export succeeds (#28)")
        self.assertTrue(has_error(errors, "<type>(<scope>): <summary>"))

    def test_rejects_unknown_type_and_uppercase_scope(self) -> None:
        self.assertTrue(check.check_header("bugfix(export): keep the existing movie"))
        self.assertTrue(check.check_header("fix(Export): keep the existing movie"))
        self.assertTrue(check.check_header("fix(export, web): keep the existing movie"))

    def test_rejects_non_english_summary(self) -> None:
        errors = check.check_header("fix(export): 替换导出成功前保留原有成片 (#28)")
        self.assertTrue(has_error(errors, "English"))

    def test_rejects_trailing_period_long_title_and_wip(self) -> None:
        self.assertTrue(check.check_header("fix(export): keep the existing movie."))
        self.assertTrue(check.check_header("fix(export): keep the existing movie. (#28)"))
        self.assertTrue(check.check_header("fix(export): " + "x" * 80))
        self.assertTrue(check.check_header("fix(export): WIP keep the existing movie"))


class BodyTests(unittest.TestCase):
    def test_accepts_filled_template(self) -> None:
        self.assertEqual(check.check_body(GOOD_BODY, GOOD_TITLE), [])

    def test_accepts_minimal_description_without_optional_sections(self) -> None:
        body = "## Summary\n- Add the conventions.\n\n## Testing\nRan the checker unit tests: 30 passed.\n"
        self.assertEqual(check.check_body(body, "docs: add agent message conventions"), [])

    def test_rejects_untouched_template(self) -> None:
        template = (ROOT / ".github" / "pull_request_template.md").read_text(encoding="utf-8")
        errors = check.check_body(template, GOOD_TITLE)
        self.assertTrue(has_error(errors, "'## Summary' is empty"))
        self.assertTrue(has_error(errors, "'## Root cause' is empty"))
        self.assertTrue(has_error(errors, "'## Implementation notes' is empty; fill it in or delete it"))
        self.assertTrue(has_error(errors, "'## Linked issues' needs"))
        self.assertTrue(has_error(errors, "'## Testing' must not contain unchecked items"))
        self.assertTrue(has_error(errors, "template placeholder"))

    def test_rejects_missing_sections(self) -> None:
        errors = check.check_body("Fixes the export bug.")
        self.assertEqual(sum("is missing the" in error for error in errors), 2)
        errors = check.check_body("Fixes the export bug.", PARTIAL_TITLE)
        for name in ("Summary", "Testing", "Root cause", "Linked issues", "Follow-ups"):
            self.assertTrue(has_error(errors, f"missing the '## {name}' section"), name)

    def test_root_cause_is_required_only_for_fix_and_perf(self) -> None:
        without = GOOD_BODY.replace(
            "## Root cause\nThe destination was opened with O_TRUNC and deleted on cancel, "
            "so re-exporting lost the old movie.\n\n",
            "",
        )
        self.assertTrue(has_error(check.check_body(without, GOOD_TITLE), "'## Root cause'"))
        self.assertEqual(check.check_body(without, "refactor(export): split the export module (#28)"), [])

    def test_rejects_placeholder_and_empty_sections(self) -> None:
        for extra in ("## Follow-ups\nNone\n", "## Follow-ups\n- N/A\n", "## Notes\nTBD.\n", "## Notes\n—\n"):
            errors = check.check_body(GOOD_BODY + "\n" + extra, GOOD_TITLE)
            self.assertTrue(has_error(errors, "instead of writing a placeholder"), extra)
        errors = check.check_body(GOOD_BODY + "\n## Follow-ups\n<!-- nothing -->\n-\n", GOOD_TITLE)
        self.assertTrue(has_error(errors, "'## Follow-ups' is empty; fill it in or delete it"))
        errors = check.check_body(GOOD_BODY.replace("Low; revert this commit.", ""), GOOD_TITLE)
        self.assertTrue(has_error(errors, "'## Risk and rollback' is empty"))

    def test_rejects_attribution_and_session_links(self) -> None:
        for footer in (
            "\U0001f916 Generated with [Claude Code](https://claude.com/claude-code)",
            "https://claude.ai/code/session_016nSUJNm1Jjh5fAFhQrWBxE",
            "Codex task: https://chatgpt.com/codex/tasks/task_e_123",
        ):
            errors = check.check_body(GOOD_BODY + "\n" + footer + "\n", GOOD_TITLE)
            self.assertTrue(has_error(errors, "tool attribution"), footer)
        prose = GOOD_BODY.replace("Low; revert this commit.", "Frames generated by the renderer are unchanged.")
        self.assertEqual(check.check_body(prose, GOOD_TITLE), [])

    def test_rejects_non_english_prose_but_allows_code(self) -> None:
        chinese = GOOD_BODY.replace("Low; revert this commit.", "风险较低，可直接回滚。")
        self.assertTrue(has_error(check.check_body(chinese, GOOD_TITLE), "English"))
        with_code = GOOD_BODY.replace(
            "Low; revert this commit.", "Low. The UI label `导出完成` is unchanged.\n\n```text\n导出失败\n```"
        )
        self.assertEqual(check.check_body(with_code, GOOD_TITLE), [])

    def test_linked_issues_match_the_title(self) -> None:
        self.assertEqual(check.check_body(GOOD_BODY.replace("Closes #28", "Refs #3"), "docs: explain export"), [])
        errors = check.check_body(GOOD_BODY.replace("Closes #28", "See the export issue"))
        self.assertTrue(has_error(errors, "'## Linked issues' needs"))
        errors = check.check_body(GOOD_BODY.replace("Closes #28", "None — documentation only"))
        self.assertTrue(has_error(errors, "'## Linked issues' needs"))
        errors = check.check_body(GOOD_BODY.replace("Closes #28", "Refs #28"), GOOD_TITLE)
        self.assertTrue(has_error(errors, "link it with 'Closes #28'"))
        errors = check.check_body(GOOD_BODY, "fix(tauri): unblock saves during proxy creation (#28, #97)")
        self.assertTrue(has_error(errors, "must reference #97"))

    def test_partial_fix_uses_refs_and_follow_ups(self) -> None:
        partial = GOOD_BODY.replace("Closes #28", "Refs #28") + FOLLOW_UPS
        self.assertEqual(check.check_body(partial, PARTIAL_TITLE), [])
        errors = check.check_body(GOOD_BODY + FOLLOW_UPS, PARTIAL_TITLE)
        self.assertTrue(has_error(errors, "partial fix must link #28 with 'Refs #28'"))
        errors = check.check_body(GOOD_BODY.replace("Closes #28", "Refs #28"), PARTIAL_TITLE)
        self.assertTrue(has_error(errors, "missing the '## Follow-ups' section"))

    def test_testing_lists_only_what_was_run(self) -> None:
        unchecked = GOOD_BODY.replace("- [x] Manual check", "- [ ] Manual check")
        self.assertTrue(has_error(check.check_body(unchecked, GOOD_TITLE), "unchecked items"))
        described = GOOD_BODY.replace(
            "- [x] `cargo test --locked -p opentake-tauri export` (18 passed)",
            "Ran the export test suite locally: 42 passed.",
        )
        self.assertEqual(check.check_body(described, GOOD_TITLE), [])

    def test_ignores_html_comments(self) -> None:
        body = GOOD_BODY.replace("## Summary\n", "## Summary\n<!-- 中文注释 None -->\n")
        self.assertEqual(check.check_body(body, GOOD_TITLE), [])


class CommitMessageTests(unittest.TestCase):
    def test_accepts_valid_message_with_trailers(self) -> None:
        message = GOOD_TITLE + "\n\nExplain why.\n\nCo-Authored-By: Someone <someone@example.com>\n"
        self.assertEqual(check.check_commit_message(message), [])

    def test_rejects_session_links(self) -> None:
        message = GOOD_TITLE + "\n\nExplain why.\n\nClaude-Session: https://claude.ai/code/session_01\n"
        self.assertTrue(has_error(check.check_commit_message(message), "tool attribution"))

    def test_skips_merge_fixup_and_revert_commits(self) -> None:
        for subject in ("Merge branch 'main' into fix/28", "fixup! fix(export): keep", 'Revert "fix(export): keep"'):
            self.assertEqual(check.check_commit_message(subject + "\n"), [])

    def test_requires_blank_line_and_english_body(self) -> None:
        self.assertTrue(check.check_commit_message(GOOD_TITLE + "\nno blank line\n"))
        self.assertTrue(check.check_commit_message(GOOD_TITLE + "\n\n修复导出。\n"))

    def test_ignores_git_comment_lines(self) -> None:
        message = "# Please enter the commit message\n" + GOOD_TITLE + "\n# 中文注释\n"
        self.assertEqual(check.check_commit_message(message), [])


class MainTests(unittest.TestCase):
    def write(self, directory: str, name: str, content: str) -> str:
        path = Path(directory) / name
        path.write_text(content, encoding="utf-8")
        return str(path)

    def event(self, directory: str, **pull_request: object) -> str:
        payload = {"pull_request": {"title": GOOD_TITLE, "body": GOOD_BODY, "draft": False, "user": {"type": "User"}}}
        payload["pull_request"].update(pull_request)
        return self.write(directory, "event.json", json.dumps(payload))

    def test_event_pass_and_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(run_main(["--event", self.event(directory)])[0], 0)
            code, output = run_main(["--event", self.event(directory, title="Fix export")])
            self.assertEqual(code, 1)
            self.assertIn("error:", output)

    def test_draft_only_warns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            code, output = run_main(["--event", self.event(directory, title="Fix export", draft=True)])
            self.assertEqual(code, 0)
            self.assertIn("warning:", output)

    def test_github_actions_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            code, output = run_main(["--event", self.event(directory, title="Fix export")], github_actions="true")
            self.assertEqual(code, 1)
            self.assertIn("::error title=PR message::", output)

    def test_bot_pull_requests_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            code, _ = run_main(["--event", self.event(directory, title="Bump x", user={"type": "Bot"})])
            self.assertEqual(code, 0)

    def test_title_and_commit_modes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            body = self.write(directory, "pr.md", GOOD_BODY)
            self.assertEqual(run_main(["--title", GOOD_TITLE, "--body-file", body])[0], 0)
            commit = self.write(directory, "COMMIT_EDITMSG", "Fix export\n")
            self.assertEqual(run_main(["--commit-msg-file", commit])[0], 1)


if __name__ == "__main__":
    unittest.main()
