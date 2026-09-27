# Contributing to OpenTake

Contributions are welcome. Discuss large changes in an issue before implementing them.

Humans and AI agents follow the same conventions, which are maintained in one place:

- [AGENTS.md](AGENTS.md) — entry point and hard rules
- [.agent/message-conventions.md](.agent/message-conventions.md) — commits, branches, issues, pull requests and reviews
- [.agent/workflow.md](.agent/workflow.md) — from issue to merge, and what to verify locally
- [.agent/engineering.md](.agent/engineering.md) — architecture invariants and coding rules

In short:

- PR titles, PR descriptions and commit messages are written in English. Issues are preferably English; Chinese is accepted.
- Commit subjects and PR titles use `<type>(<scope>): <summary>`, for example `fix(export): keep the old movie until the new export succeeds (#28)`.
- Issue titles use `[P0-P3][<area>] <summary>`; use the templates under `.github/ISSUE_TEMPLATE/`.
- PR descriptions start from `.github/pull_request_template.md`; delete the sections that do not apply rather than writing `None` or `N/A`. The `PR message` check enforces the title and description format.
- No tool attribution lines or agent session links in PRs, issues or commits.
- Enable the local commit message check once per clone with `git config core.hooksPath .githooks`.

## Validation

PR CI builds only the lightweight crates and the web frontend. Run the checks that match your change from the repository root and record the commands you actually ran in the PR:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --locked -p <crate> -- --test-threads=1
pnpm -C web install --frozen-lockfile
pnpm -C web build
pnpm -C web test
```

Native playback, GPU, audio, provider and installer changes need evidence from the relevant platform; a browser fallback run does not validate the desktop app.

## License

Contributions are licensed under the repository's GPL-3.0-or-later license. Keep upstream attribution and third-party notices intact.
