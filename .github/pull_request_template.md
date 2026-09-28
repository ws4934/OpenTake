<!--
Title: <type>(<scope>): <summary>, in English, at most 72 characters.
Example: fix(export): keep the old movie until the new export succeeds (#28)
Partial fix: end the title with (partial #N), link the issue with "Refs #N"
and add a "## Follow-ups" section that lists the remaining work.

Write in English. Keep only sections that carry real information: delete the
ones that do not apply instead of writing None, N/A or TBD, and delete the
checks you did not run. No tool attribution lines or agent session links.
Keep the PR in draft until it is complete and verified.
Conventions: .agent/message-conventions.md
-->

## Linked issues
<!-- Complete fix: Closes #123. Partial or related change: Refs #123. One per line. Delete this section when there is no issue. -->
Closes #

## Summary
<!-- User-visible changes and the main code changes -->
-

## Root cause
<!-- fix and perf only: why the problem happened. Delete this section for other types. -->

## Implementation notes
<!-- Optional: key design decisions and trade-offs; justify any departure from upstream palmier-pro semantics -->

## Testing
<!--
Check what you actually ran and add the results; delete the rest. CI tests the
affected crates on Linux, Windows and macOS; name what it cannot cover (GPU,
audio device, real browser, installer) and how you verified it.
-->
- [ ] `cargo fmt --all --check`
- [ ] `cargo clippy --workspace --all-targets --locked -- -D warnings`
- [ ] `cargo test --locked -p <crate>`
- [ ] `pnpm -C web build && pnpm -C web test`
- [ ] Manual check: <platform, steps, result>

## Risk and rollback
<!-- Optional: what could break (data, compatibility, platforms) and how to roll back -->
