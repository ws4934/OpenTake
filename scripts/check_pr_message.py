#!/usr/bin/env python3
"""Check PR titles, PR descriptions and commit messages against the conventions.

The rules are documented in .agent/message-conventions.md.

Usage:
  python3 scripts/check_pr_message.py --event "$GITHUB_EVENT_PATH"
  python3 scripts/check_pr_message.py --title "fix(export): ..." --body-file pr.md
  python3 scripts/check_pr_message.py --commit-msg-file .git/COMMIT_EDITMSG
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

TYPES = ("feat", "fix", "perf", "refactor", "test", "docs", "build", "ci", "chore", "revert")
HEADER = re.compile(
    r"^(?P<type>[A-Za-z]+)"
    r"(?:\((?P<scope>[^()]*)\))?"
    r"(?P<breaking>!)?"
    r": (?P<summary>\S.*)$"
)
SCOPE = re.compile(r"^[a-z0-9][a-z0-9-]*(?:,[a-z0-9][a-z0-9-]*)*$")
ISSUE_SUFFIX = re.compile(r"\s*\((?P<partial>partial )?(?P<issues>#\d+(?:, #\d+)*)\)$")
MAX_HEADER = 72
# Ideographs, kana, hangul and full-width punctuation: anything outside code
# that is not English text.
NON_ENGLISH = re.compile(
    r"[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uff00-\uffef]"
)
COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
INLINE_CODE = re.compile(r"(`+).*?\1")
HEADING = re.compile(r"^##\s+(?P<name>.+?)\s*#*\s*$")
EMPTY_LINE = re.compile(r"^\s*(?:[-*+]\s*(?:\[[ xX]\]\s*)?)?$")
PLACEHOLDER = re.compile(
    r"^\s*(?:[-*+]\s*)?(?:none|n/?a|tbd|todo|nothing|not applicable|[-\u2013\u2014]+)\s*\.?\s*$",
    re.IGNORECASE,
)
TEMPLATE_PLACEHOLDER = re.compile(r"<(?:crate|platform, steps, result)>")
CHECKED = re.compile(r"^\s*[-*+]\s*\[[xX]\]")
UNCHECKED = re.compile(r"^\s*[-*+]\s*\[ \]")
ISSUE_REFERENCE = re.compile(
    r"\b(?P<keyword>close[sd]?|fix(?:e[sd])?|resolve[sd]?|refs?)\s+#(?P<number>\d+)", re.IGNORECASE
)
# GitHub closes the issue after a closing keyword anywhere in a description.
CLOSING_REFERENCE = re.compile(
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?):?\s+(?:[\w.-]+/[\w.-]+)?#\d+", re.IGNORECASE
)
# Tool attribution footers and agent session or task links.
ATTRIBUTION = re.compile(
    r"generated (?:with|by) \[?(?:claude|codex|chatgpt|openai|copilot|cursor|gemini)"
    r"|claude\.ai/code/session|chatgpt\.com/codex/tasks",
    re.IGNORECASE,
)
ROOT_CAUSE_TYPES = ("fix", "perf")
EXEMPT_COMMIT_PREFIXES = ("Merge ", "fixup! ", "squash! ", "amend! ", 'Revert "')
TEMPLATE_HINT = "see .github/pull_request_template.md and .agent/message-conventions.md"


def prose_lines(text: str) -> list[tuple[int, str]]:
    """Return (line number, text) pairs outside comments, code blocks and inline code."""
    text = COMMENT.sub(lambda match: "\n" * match.group(0).count("\n"), text)
    fence: str | None = None
    lines = []
    for number, line in enumerate(text.splitlines(), 1):
        marker = FENCE.match(line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            continue
        if fence is None:
            lines.append((number, INLINE_CODE.sub("", line)))
    return lines


def non_english_errors(text: str, what: str) -> list[str]:
    for number, line in prose_lines(text):
        if NON_ENGLISH.search(line):
            return [
                f"{what} must be written in English; non-English text is allowed only inside "
                f"inline code or code blocks (line {number}: {line.strip()[:60]!r})"
            ]
    return []


def check_header(header: str, what: str = "PR title") -> list[str]:
    errors = []
    if len(header) > MAX_HEADER:
        errors.append(f"{what} is {len(header)} characters long; the limit is {MAX_HEADER}")
    if re.search(r"\bWIP\b", header, re.IGNORECASE):
        errors.append(f"{what} must not contain WIP; use a draft PR instead")
    match = HEADER.match(header)
    if not match:
        errors.append(
            f"{what} must look like '<type>(<scope>): <summary>', for example "
            "'fix(export): keep the old movie until the new export succeeds (#28)'"
        )
        return errors + non_english_errors(header, what)
    if match["type"] not in TYPES:
        errors.append(f"{what} type {match['type']!r} must be one of: {', '.join(TYPES)}")
    scope = match["scope"]
    if scope is not None and not SCOPE.match(scope):
        errors.append(
            f"{what} scope {scope!r} must be lowercase names separated by commas without spaces"
        )
    summary = ISSUE_SUFFIX.sub("", match["summary"]).rstrip()
    if not summary:
        errors.append(f"{what} needs a summary after the colon")
    elif summary.endswith((".", "\u3002")):
        errors.append(f"{what} summary must not end with a period")
    return errors + non_english_errors(header, what)


def attribution_errors(text: str, what: str) -> list[str]:
    text = COMMENT.sub(lambda match: "\n" * match.group(0).count("\n"), text)
    for number, line in enumerate(text.splitlines(), 1):
        if ATTRIBUTION.search(line):
            return [
                f"{what} must not contain tool attribution or agent session links "
                f"(line {number}: {line.strip()[:60]!r})"
            ]
    return []


def stray_closing_errors(body: str) -> list[str]:
    """Closing keywords outside '## Linked issues' still close their issues on merge."""
    errors = []
    in_linked = False
    for number, line in prose_lines(body):
        heading = HEADING.match(line)
        if heading:
            in_linked = heading["name"].strip().lower() == "linked issues"
            continue
        if in_linked:
            continue
        for match in CLOSING_REFERENCE.finditer(line):
            errors.append(
                f"{match.group(0)!r} outside '## Linked issues' closes that issue when the PR "
                f"merges (line {number}); link issues only in '## Linked issues' or reword it"
            )
    return errors


def sections(body: str) -> dict[str, list[str]]:
    """Map '## ' headings, as written, to their content lines (code included, comments removed)."""
    found: dict[str, list[str]] = {}
    current: list[str] | None = None
    fence: str | None = None
    for line in COMMENT.sub("", body).splitlines():
        marker = FENCE.match(line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
        heading = HEADING.match(line) if fence is None and not marker else None
        if heading:
            current = found.setdefault(heading["name"].strip(), [])
        elif current is not None:
            current.append(line)
    return found


def filled(lines: list[str]) -> list[str]:
    return [line for line in lines if not EMPTY_LINE.match(line)]


def check_body(body: str, title: str = "") -> list[str]:
    """Check a PR description; the title decides which conditional sections are required."""
    errors = non_english_errors(body, "PR description") + attribution_errors(body, "PR description")
    errors += stray_closing_errors(body)
    header = HEADER.match(title)
    suffix = ISSUE_SUFFIX.search(title)
    partial = bool(suffix and suffix["partial"])
    found = sections(body)
    by_key = {name.lower(): lines for name, lines in found.items()}

    required = {"Summary": "always", "Testing": "always"}
    if header and header["type"] in ROOT_CAUSE_TYPES:
        required["Root cause"] = f"required for a {header['type']} title"
    if suffix:
        required["Linked issues"] = "required when the title references an issue"
    if partial:
        required["Follow-ups"] = "required for a partial fix"
    for name, reason in required.items():
        if name.lower() not in by_key:
            hint = TEMPLATE_HINT if reason == "always" else reason
            errors.append(f"PR description is missing the '## {name}' section ({hint})")

    for name, lines in found.items():
        content = filled(lines)
        if not content:
            advice = "" if name.lower() in map(str.lower, required) else "; fill it in or delete it"
            errors.append(f"PR description section '## {name}' is empty{advice}")
        elif all(PLACEHOLDER.match(line) for line in content):
            errors.append(
                f"PR description section '## {name}' only says {content[0].strip()!r}; "
                "delete sections that do not apply instead of writing a placeholder"
            )
        elif any(TEMPLATE_PLACEHOLDER.search(line) for line in content):
            errors.append(f"PR description section '## {name}' still contains a template placeholder")

    linked = filled(by_key.get("linked issues", []))
    closing: set[str] = set()
    related: set[str] = set()
    for match in ISSUE_REFERENCE.finditer("\n".join(linked)):
        (related if match["keyword"].lower().startswith("ref") else closing).add(match["number"])
    if linked and not closing | related:
        errors.append(
            "'## Linked issues' needs 'Closes #N' (complete fix) or 'Refs #N' (partial or related "
            "change); delete the section when there is no issue"
        )
    for number in re.findall(r"\d+", suffix["issues"]) if suffix else ():
        if number not in closing | related:
            errors.append(f"'## Linked issues' must reference #{number} from the title")
        elif partial and number in closing:
            errors.append(
                f"a partial fix must link #{number} with 'Refs #{number}'; a closing keyword would "
                "close the issue on merge"
            )
        elif not partial and number not in closing:
            errors.append(
                f"the title marks #{number} as fixed: link it with 'Closes #{number}', or end the "
                f"title with '(partial #{number})'"
            )

    if any(UNCHECKED.match(line) for line in by_key.get("testing", [])):
        errors.append(
            "'## Testing' must not contain unchecked items; check what you ran ('- [x]') and "
            "delete the rest"
        )
    return errors


def check_pull_request(title: str, body: str) -> list[str]:
    return check_header(title, "PR title") + check_body(body or "", title)


def check_commit_message(message: str) -> list[str]:
    lines = [line.rstrip() for line in message.splitlines() if not line.startswith("#")]
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return ["commit message is empty"]
    subject = lines[0]
    if subject.startswith(EXEMPT_COMMIT_PREFIXES):
        return []
    errors = check_header(subject, "commit subject")
    if len(lines) > 1 and lines[1].strip():
        errors.append("separate the commit subject from the body with a blank line")
    if len(lines) > 2:
        errors += non_english_errors("\n".join(lines[2:]), "commit body")
    return errors + attribution_errors("\n".join(lines), "commit message")


def report(errors: list[str], *, warn_only: bool) -> int:
    annotate = os.environ.get("GITHUB_ACTIONS") == "true"
    level = "warning" if warn_only else "error"
    for error in errors:
        print(f"::{level} title=PR message::{error}" if annotate else f"{level}: {error}")
    if errors and warn_only:
        print("draft PR: these problems must be fixed before the PR is marked ready for review")
    return 1 if errors and not warn_only else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--event", type=Path, help="GitHub pull_request event payload")
    source.add_argument("--title", help="PR title to check")
    source.add_argument("--commit-msg-file", type=Path, help="commit message file to check")
    parser.add_argument("--body-file", type=Path, help="PR description to check with --title")
    args = parser.parse_args(argv)

    if args.commit_msg_file:
        message = args.commit_msg_file.read_text(encoding="utf-8")
        return report(check_commit_message(message), warn_only=False)

    if args.title is not None:
        body = args.body_file.read_text(encoding="utf-8") if args.body_file else ""
        errors = check_pull_request(args.title, body)
        if not errors:
            print("PR message follows the conventions")
        return report(errors, warn_only=False)

    event = json.loads(args.event.read_text(encoding="utf-8"))
    pull_request = event.get("pull_request") or {}
    if (pull_request.get("user") or {}).get("type") == "Bot":
        print("skipping PR opened by a bot account")
        return 0
    errors = check_pull_request(pull_request.get("title") or "", pull_request.get("body") or "")
    if not errors:
        print("PR message follows the conventions")
    return report(errors, warn_only=bool(pull_request.get("draft")))


if __name__ == "__main__":
    sys.exit(main())
