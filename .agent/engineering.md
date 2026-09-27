# Engineering conventions

> Rules for changing code in this repository. Applies to human contributors and AI agents alike.

## Architecture invariants

- Rust owns the authoritative `Timeline` and media manifest. The web frontend keeps a read-only mirror, version numbers and UI state only.
- Every edit — from the UI, the agent or MCP — goes through an `EditCommand` in `opentake-ops`. Undo and redo live in Rust; the frontend never builds its own authoritative model.
- Editing algorithms (overwrite, ripple, snap, trim, split) stay pure functions. `opentake-domain` performs no file or network I/O.
- Preview, playback and export share the RenderPlan and compositing semantics. When you change one path, check the other two.
- Timelines use integer frames. Seconds-to-frames conversions follow the upstream semantics (`Int(seconds * fps)` truncates toward zero; do not switch it to rounding). Keyframes are stored clip-relative; public editing APIs use absolute timeline frames and convert explicitly at the boundary.

## Threads, async and processes

- In Tauri 2 a `#[tauri::command]` that is not `async` runs on the main (WebView UI) thread. Anything that may take longer than about 16 ms — file I/O, decoding, hashing, network, GPU readback, model inference — belongs in an `async` command that hands the work to `tauri::async_runtime::spawn_blocking` or a bounded background worker.
- Do not perform blocking I/O directly in async code, do not hold a `std::sync` lock across long operations, and never hold a lock across `.await`.
- Long-running work must be cancellable (`MediaCancelToken` or an equivalent) and must stop when the open project changes.
- Start child processes through `opentake-process-tree` (`configure_command` / `background_command`) so they are contained, reaped and do not open console windows on Windows.

## Errors, IPC and persistence

- IPC field names are camelCase. Change the Rust DTO, `web/src/lib/types.ts` and every caller together.
- Tauri command boundaries convert errors to `Err(String)`. Do not swallow errors that affect user data (`let _ =`, `.ok()`, empty `catch`); the user must see failures.
- Persist with atomic writes (temporary file, fsync, rename). A failed operation must not leave memory and disk out of sync.

## Upstream parity

- Editing logic and UI behavior are ported one-to-one from upstream palmier-pro. Any intentional departure from upstream semantics must be justified in the PR's `## Implementation notes`.

## Dependencies and formatting

- Verify the real API of a new dependency (read its source in the cargo registry) instead of guessing. Keep the lockfiles consistent so `--locked` builds succeed.
- Run `cargo fmt --all` before committing Rust code. The frontend has no lint script; `pnpm -C web build` includes type checking.
- Do not commit binaries or generated artifacts. FFmpeg sidecars are provisioned by `scripts/provision_ffmpeg_sidecars.py`.
