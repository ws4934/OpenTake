# OpenTake agent guide

This file is the entry point for every AI agent working in this repository (Codex reads it automatically; Claude Code imports it through `CLAUDE.md`). Human contributors follow the same rules; see [CONTRIBUTING.md](CONTRIBUTING.md).

OpenTake is a cross-platform AI video editor: a Rust workspace (`crates/`), a Tauri 2 desktop shell (`src-tauri/`) and a React/TypeScript frontend (`web/`), with FFmpeg sidecars, wgpu compositing and an MCP-based agent surface.

## Required reading

| When you are about to… | Read |
|---|---|
| write a commit, file an issue, open or review a PR | [.agent/message-conventions.md](.agent/message-conventions.md) |
| take an issue from branch to merge, or decide what to verify | [.agent/workflow.md](.agent/workflow.md) |
| change code | [.agent/engineering.md](.agent/engineering.md) |

## Hard rules

These apply even if you have not opened the files above.

1. **PR titles, PR descriptions and commit messages are English only.** Issues are preferably English; Chinese is accepted.
2. Commit subjects and PR titles use `<type>(<scope>): <summary>` — for example `fix(export): keep the old movie until the new export succeeds (#28)` — at most 72 characters, no trailing period, no `WIP`.
3. Issue titles use `[P0-P3][<area>] <summary>` and carry a type label, a priority label and an area or platform label.
4. PR descriptions start from `.github/pull_request_template.md` and keep only sections with real content: delete sections that do not apply instead of writing `None`, `N/A` or `TBD`. Use `Closes #N` for a complete fix; use `Refs #N` plus `(partial #N)` in the title and a `## Follow-ups` list for a partial fix.
5. No tool attribution ("Generated with …" lines) or agent session links in PRs, issues, review comments or commit messages; a standard `Co-authored-by:` commit trailer is fine.
6. PR CI picks its checks from the change: `pr.yml` builds the core crates and the web frontend on Linux, and `pr-native.yml` lints the whole workspace and tests every affected crate on Linux, Windows and macOS when Rust code changes. Tests that need a GPU, an audio device or a real browser skip on CI runners; verify those paths yourself and record the actual commands and results in the PR.
7. Open PRs as drafts until they are complete and verified. One PR per issue. Never push to `main`, never force-push someone else's branch, never skip or delete tests to get CI green.
8. Squash-merge, then delete the head branch on GitHub and locally.

## Quick commands

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --locked -p <crate> -- --test-threads=1
pnpm -C web install --frozen-lockfile && pnpm -C web build && pnpm -C web test
python3 scripts/check_pr_message.py --title "<pr title>" --body-file pr.md
```
