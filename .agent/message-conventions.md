# Message conventions: commits, branches, issues, pull requests, reviews

> Applies to every contributor, including AI agents such as Codex and Claude Code.
> Automated checks: `.github/workflows/pr-message.yml` validates PR titles and descriptions; the optional local hook `.githooks/commit-msg` validates commit messages. Both use `scripts/check_pr_message.py`.

## 1. Language

| Artifact | Language |
|---|---|
| PR title and description | **English only** |
| Commit message (subject and body) | **English only** |
| PR review comments and replies | English (preferred) |
| Issue title and body | English preferred; Chinese is accepted. Use one language per issue. |

- In English-only artifacts, non-English text may appear only inside inline code or code blocks (UI strings, test data, quoted logs). Link issues by number instead of quoting non-English titles.
- Keep code identifiers, paths, commands, configuration keys, log and error text, protocol fields, and third-party names verbatim.
- No tool attribution in PR titles and descriptions, issues or review comments: do not name the tool or model that produced the change, and do not add "Generated with …" lines or agent session and task links (for example `claude.ai/code/session_…` or `chatgpt.com/codex/tasks/…`). Delete them when a tool appends them. Commit messages may keep a standard `Co-authored-by:` trailer but must not contain session links.
- Write only what carries information. Delete a section that does not apply instead of filling it with `None`, `N/A`, `TBD` or `-`.

## 2. Types and scopes

Commit subjects and PR titles share the Conventional Commits header `<type>(<scope>): <summary>`.

| type | Use for |
|---|---|
| `feat` | New feature |
| `fix` | Bug fix |
| `perf` | Performance improvement without behavior change |
| `refactor` | Restructuring without behavior change |
| `test` | Tests only |
| `docs` | Documentation or conventions only |
| `build` | Build system, dependencies, packaging |
| `ci` | CI workflows |
| `chore` | Other maintenance |
| `revert` | Reverting a previous commit |

The scope is optional. Use lowercase module names; separate several scopes with commas and no spaces (`agent,web`). Recommended scopes:

| scope | Area |
|---|---|
| `domain`, `ops`, `project`, `core` | the matching `crates/opentake-*` crate |
| `media`, `render`, `motion`, `agent`, `gen`, `process` | the matching `crates/opentake-*` crate (`process` = `opentake-process-tree`) |
| `playback`, `export` | playback engine and export in `src-tauri` |
| `tauri` | the rest of `src-tauri`: commands, lifecycle, asset protocol |
| `library`, `search` | global media library, semantic search |
| `web` | the `web/` frontend |
| `ci`, `deps`, `release`, `docs`, `scripts` | infrastructure |

## 3. Commit messages

```text
<type>(<scope>): <summary>

<body: why the change is needed (root cause or motivation), what changed,
and the impact or risk>

Closes #28
```

- Summary: English, imperative mood ("keep", "reject", "stream"), lowercase first word unless it is a proper noun or acronym, no trailing period. Aim for 50 characters; the hard limit is 72.
- The summary may end with an issue reference: `(#28)`, `(#64, #97)` or `(partial #3)`.
- Separate the subject from the body with one blank line; wrap body lines at about 72 columns.
- Breaking change: add `!` after the type or scope and explain it in a `BREAKING CHANGE:` paragraph.
- Revert: `revert: revert "<original subject>"`, and name the reverted SHA and the reason in the body.
- Exempt: merge commits created while syncing `main`, and `fixup!`/`squash!` commits that are squashed before merging.

Good:

```text
fix(export): keep the old movie until the new export succeeds (#28)

Export opened the destination with O_TRUNC and deleted it on cancel or
failure, so re-exporting over an existing file lost the previous movie.
Write to an exclusive temporary file in the same directory and atomically
replace the destination only after verification succeeds.

Closes #28
```

Bad: `Fix export bug` (no type), `fix: update.` (no information, trailing period), `WIP`, any non-English summary.

## 4. Branch names

- Format: `<type>/<issue>-<short-kebab-description>`, for example `fix/28-export-atomic` or `perf/3-export-stream`.
- Agents may add a tool prefix but must keep the issue number: `codex/28-export-atomic`, `claude/28-export-atomic`.
- Maintenance without an issue: `chore/<description>`, `docs/<description>`.
- Never push directly to `main`.

## 5. Issues

### 5.1 Before filing

- Search existing issues, including closed ones, to avoid duplicates.
- Do not publish exploitation details of a serious, exploitable vulnerability; contact the maintainers privately first.

### 5.2 Title

Format: `[<priority>][<area>] <summary>`. Describe the symptom, not the fix.

- Bug: `[P1][export] Export silently skips missing layers and reports success`
- Enhancement: `[P2][frontend] Sort media panel items by duration` (type label `enhancement`)
- Chinese issues use the same shape, for example `[P1][导出] 导出遇到缺失图层时静默跳过并报告成功`.

### 5.3 Priority

| Priority | Criteria |
|---|---|
| P0 | Data loss, crash, or exploitable security flaw that is easy to trigger |
| P1 | Broken core feature, frozen UI, severe performance problem |
| P2 | Edge-case bug, moderate performance problem, robustness issue |
| P3 | Minor issue, optimization, code quality (may be grouped into one "collection" issue) |

### 5.4 Area and labels

Every issue needs one type label (`bug`, `performance`, `security`, `enhancement`, `documentation`), one priority label (`P0`–`P3`) and at least one area or platform label.

| Area in the title (English / Chinese) | Label |
|---|---|
| `ops-domain` / 编辑内核 | `area:ops-domain` |
| `core-project` / 工程持久化 | `area:core-project` |
| `media` / 媒体解码 | `area:media` |
| `render` / 合成渲染 | `area:render` |
| `playback` / 播放引擎 | `area:playback` |
| `export` / 导出 | `area:export` |
| `tauri-shell` / 桌面壳层 | `area:tauri-shell` |
| `agent-mcp` / Agent/MCP | `area:agent-mcp` |
| `generation` / AI生成 | `area:generation` |
| `motion` / Motion | `area:motion` |
| `library-search` / 素材库/检索 | `area:library-search` |
| `frontend` / 前端 | `area:frontend` |
| `windows`, `linux` / Windows, Linux | `platform:windows`, `platform:linux` |

### 5.5 Body

Use a template from `.github/ISSUE_TEMPLATE/`. Bug, performance and security issues contain these sections (Chinese issues may use the equivalent Chinese headings):

- `## Summary`: the symptom, the root cause when known, and the consequence, in two to five sentences.
- `## Code locations`: `path/to/file.rs:line` entries with the baseline commit (for example `main@6831d38`), so readers can relocate them by symbol after the code moves.
- `## Reproduction`: numbered steps plus expected and actual results.
- `## Impact`: user-visible consequences and data, performance or security impact.
- `## Suggested fix`: concrete functions or approaches; explain trade-offs between alternatives.
- `## Acceptance criteria`: executable tests or measurable thresholds, as a checklist.
- Optional `## Notes for implementers`: files to read first, related issues, upstream semantics to keep.

Enhancements contain `## Background and goal`, `## Proposed approach` and `## Acceptance criteria`. Delete optional sections that have nothing to say.

Evidence: state at the top how the problem was verified (code reading, reproduction, benchmark) and mark speculation as "needs confirmation".

### 5.6 Maintenance and closing

- File problems found during a fix as new issues instead of widening the original.
- Partial fix: the PR uses `Refs #N`; after merging, comment on the issue with what is done and what remains, and keep it open.
- Closing: let `Closes #N` in a merged PR close the issue. A manual close states the reason: fixed by a PR, duplicate of an issue, or no longer applicable.

## 6. Pull requests

### 6.1 Title

- Same header as a commit subject: `<type>(<scope>): <summary>`, English, at most 72 characters. The PR title becomes the squash-merge commit subject.
- A partial fix ends with `(partial #N)`, for example `perf(export): stream forward video layers during export (partial #3)`.
- Never write `WIP`; use a draft PR instead.

### 6.2 Description

Start from `.github/pull_request_template.md` and write in English. Every section that remains must carry real information: delete sections that do not apply, never leave one empty or write `None`, `N/A`, `TBD` or `-` in it, and do not append tool attribution or session links (§1).

- `## Linked issues` (whenever the PR relates to an issue; required when the title references one): `Closes #N` for a complete fix, `Refs #N` for a partial or related change, one per line. Every issue in the title appears here; a `(partial #N)` title links it with `Refs #N`, never with a closing keyword. Delete the section when there is no issue.
- `## Summary` (required): user-visible changes and the main code changes.
- `## Root cause` (required for `fix` and `perf`; delete it for other types): why the problem happened.
- `## Implementation notes` (optional): key design decisions and trade-offs; justify any departure from upstream palmier-pro semantics.
- `## Testing` (required): only what was actually run, with results. Check each item you ran and delete the others; manual checks name the platform and steps. Changes to crates that PR CI does not build (see [workflow.md](workflow.md) §3) must include the local full-workspace results, or explain why they could not be run.
- `## Risk and rollback` (optional): what could break (data, compatibility, platforms) and how to roll back.
- `## Follow-ups` (required for a partial fix; optional otherwise): the remaining work, item by item.

Example of a complete fix, with the optional sections deleted:

```markdown
## Linked issues
Closes #28

## Summary
- Export writes to an exclusive temporary file next to the destination and replaces the destination only after verification succeeds.

## Root cause
`export_video` opened the destination with `O_TRUNC` and deleted it on cancel or failure, so exporting over an existing movie lost it.

## Testing
- [x] `cargo test --locked -p opentake-tauri export -- --test-threads=1` (18 passed)
- [x] `cargo clippy --workspace --all-targets --locked -- -D warnings`
```

### 6.3 Process rules

- One PR solves one issue (or one tightly related group). Do not mix in unrelated refactors, formatting or dependency upgrades.
- Keep a PR in draft while it is incomplete, unverified or waiting on another PR; mark it ready for review afterwards.
- Merge only when PR checks and the PR message check pass and review feedback is resolved. Never skip or delete tests to make CI pass.
- Merge method: squash merge, with the PR title as the commit subject.
- Branch cleanup: delete the head branch as soon as its PR is merged, on GitHub and in your clone. The repository setting "Automatically delete head branches" does this on merge; otherwise use `gh pr merge --squash --delete-branch` or the "Delete branch" button. When a PR is closed without merging, delete its branch too unless the work will continue in a new PR. Fork branches belong to their authors.
- Syncing with `main`: rebase or merge on your own branch; only merge (never force-push) on someone else's branch.
- Changing someone else's PR: prefer review comments; if you push to it directly, leave a PR comment describing the change.

### 6.4 Review comments

- Point to the location, the problem, the reason and a suggestion.
- When replying, say "Fixed in `<commit>`" or explain why the code stays as it is.

## 7. Command reference

```bash
# File an issue
gh issue create \
  --title "[P1][export] Export silently skips missing layers and reports success" \
  --label bug --label P1 --label area:export \
  --body-file issue.md

# Open a PR (draft by default; mark ready after verification)
gh pr create --draft --base main \
  --title "fix(export): keep the old movie until the new export succeeds (#28)" \
  --body-file pr.md

# Squash-merge a PR and delete its branch
gh pr merge 123 --squash --delete-branch

# Drop local branches whose remote branch is gone
git fetch --prune
git for-each-ref --format='%(refname:short) %(upstream:track)' refs/heads \
  | awk '$2 == "[gone]" {print $1}' | xargs -r git branch -D

# Check a PR title and description locally
python3 scripts/check_pr_message.py \
  --title "fix(export): keep the old movie until the new export succeeds (#28)" \
  --body-file pr.md

# Enable the commit message hook (once per clone)
git config core.hooksPath .githooks
```

## 8. What the checks enforce

- `.github/workflows/pr-message.yml` runs when a PR is opened, edited, synchronized, reopened or marked ready for review. Draft PRs only get warnings; a non-compliant PR fails once it is ready for review.
- PR title: Conventional Commits header with a known type, lowercase scope, an English summary, at most 72 characters, no trailing period, no `WIP`.
- PR description:
  - `## Summary` and `## Testing` exist; so does `## Root cause` for a `fix` or `perf` title, `## Linked issues` when the title references an issue, and `## Follow-ups` for a `(partial #N)` title.
  - No section is empty or holds only a placeholder such as `None`, `N/A`, `TBD` or `-`.
  - `## Linked issues` contains `Closes`/`Fixes`/`Resolves #N` or `Refs #N`, references every issue in the title, and uses only `Refs` for the issues of a partial fix.
  - `## Testing` has no unchecked items and contains a checked item or a written description.
  - No tool attribution lines, agent session links or leftover template placeholders such as `<crate>`; no non-English text outside inline code and code blocks.
- Commit messages (local hook): the same header rules, a blank line after the subject, an English body and no agent session links.
- PRs opened by bot accounts are skipped.
