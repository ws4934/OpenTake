# Agent workflow: from issue to merge

> Applies to every contributor, including AI agents. Message formats are defined in [message-conventions.md](message-conventions.md); coding rules in [engineering.md](engineering.md).

## 1. Picking up an issue

1. Read the whole issue, its comments and linked issues. Line numbers in an issue refer to its baseline commit; relocate the code by symbol name on the current `main`.
2. Confirm the problem still exists on the latest `main`. If it does not, or the issue is wrong, comment on the issue with the evidence instead of forcing a change.
3. Create a branch from the latest `main` (naming rules: [message-conventions.md](message-conventions.md) §4).

## 2. Implementing

- Write a test that reproduces the problem first, then make it pass.
- Keep the change focused on the issue; file anything else you notice as a new issue.
- If you can only fix part of the issue, say so in the PR title (`(partial #N)`) and list the rest under `## Follow-ups`.

## 3. Verification matrix

PR CI chooses its checks from the files a pull request changes:

- `.github/workflows/pr.yml` (every PR, Linux): `cargo fmt --all --check`; clippy and tests for `opentake-domain`, `opentake-ops`, `opentake-project`, `opentake-process-tree` and `opentake-core`; script unit tests and the bundled FFmpeg smoke test; web `pnpm -C web install --frozen-lockfile`, `pnpm -C web build`, `pnpm -C web test`.
- `.github/workflows/pr-native.yml` (Linux, Windows and macOS, when Rust code changes): `scripts/pr_native_scope.py` selects every workspace crate with a changed file plus the crates that depend on it (all crates when `Cargo.lock`, the toolchain or the FFmpeg pin changes); each runner then runs `cargo clippy --workspace --all-targets --locked -- -D warnings` and `cargo test --locked` for the selected crates.
- `.github/workflows/ci.yml` and the `*-qualification.yml` workflows run the release-time and hardware-dependent checks on tags or manual dispatch.

Tests that need a GPU, an audio device or a real browser skip on CI runners, and a skipped test is not a passed test. For those paths, and for fast feedback before pushing, run the same commands locally and paste the results into `## Testing`:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --locked -p <affected crate> -- --test-threads=1
```

Notes:

- Linux needs the Tauri system packages used by `.github/workflows/ci.yml` (`libwebkit2gtk-4.1-dev`, `libgtk-3-dev`, `libayatana-appindicator3-dev`, `librsvg2-dev`, `libasound2-dev`, `libsoup-3.0-dev`, `pkg-config`, `cmake`, `clang`).
- `ort-sys` downloads a prebuilt ONNX Runtime during the build. Where that download is blocked, `ORT_SKIP_DOWNLOAD=1` is enough for `cargo check`/`cargo clippy`; tests that link `ort` need `ORT_LIB_LOCATION` pointing at a matching ONNX Runtime (1.23.x).
- GPU, FFmpeg and audio integration tests skip themselves when the environment lacks the device or binary. A skipped test is not a passed test; report it as skipped.
- Platform-specific code (`#[cfg(windows)]`, `#[cfg(target_os = "macos")]`) must still compile on the other platforms. If you cannot run it on its target platform, write "needs verification on <platform>" in `## Testing`.
- Frontend changes: `pnpm -C web build` includes type checking; there is no separate lint script.

## 4. Opening the pull request

1. Push the branch and open a **draft** PR with an English title and a description built from the template (sections that do not apply deleted, no attribution footer or session link).
2. Check the message locally: `python3 scripts/check_pr_message.py --title "..." --body-file pr.md`.
3. Wait for CI. Fix every failure; never skip, disable or delete tests to get a green run. An error in code you did not touch that also fails on `main` belongs in its own issue and PR; mention it in `## Testing`.
4. Mark the PR ready for review once CI passes and the local verification above is recorded.

## 5. Review and merge

- Address every review comment with a fix or a written reason (see [message-conventions.md](message-conventions.md) §6.4).
- Merge with squash; the PR title becomes the commit subject.
- Delete the head branch right after the merge, on GitHub and locally (`git fetch --prune`); see [message-conventions.md](message-conventions.md) §6.3.
- After merging a partial fix, comment on the issue with what remains and keep it open. After a complete fix, check that `Closes #N` closed the issue.
