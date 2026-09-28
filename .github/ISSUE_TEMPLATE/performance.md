---
name: Performance problem
about: Slow operations, frozen UI, excessive memory or disk use. Title format "[P0-P3][area] summary".
title: "[P?][area] "
labels: performance
---

<!--
Conventions: .agent/message-conventions.md §5. English is preferred; Chinese is accepted (keep one language per issue).
Delete optional sections that have nothing to say instead of writing None or N/A; no tool attribution lines or agent session links.
Add a priority label (P0-P3) and an area or platform label.
-->

> Baseline: `main@<sha>` · Verified by: <code reading / profiling / benchmark>

## Summary

## Code locations
- `path/to/file.rs:line` — hot path or blocking call

## Measurements
<!-- Input size, environment, numbers before the fix, how they were measured -->

## Reproduction
1.

## Impact

## Suggested fix

## Acceptance criteria
- [ ] <measurable threshold, e.g. "import of 2000 files takes < 200 ms">

## Notes for implementers
<!-- Optional: files to read first, related issues, upstream semantics to keep -->
