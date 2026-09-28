/**
 * Read-only timeline mirror (SPEC §10.1). Updated ONLY by `timeline_changed` ->
 * `get_timeline`. The UI never mutates `timeline` directly — every edit is an
 * `edit_apply` command to Rust, whose event triggers a re-fetch.
 */

import { create } from "zustand";
import type { RuntimeTimelineSnapshot, Timeline } from "../lib/types";

/** Freeze `value` and everything it references, in place. A frozen object is
 *  taken to be frozen all the way down (this store only freezes whole trees),
 *  so its subtree is skipped. */
function deepFreeze<T>(value: T): T {
  if (value === null || typeof value !== "object" || Object.isFrozen(value)) return value;
  if (Array.isArray(value)) {
    for (const item of value) deepFreeze(item);
  } else {
    for (const key in value) deepFreeze(value[key]);
  }
  return Object.freeze(value);
}

const EMPTY_TIMELINE: Timeline = deepFreeze({
  fps: 30,
  width: 1920,
  height: 1080,
  settingsConfigured: false,
  tracks: [],
});

/** The revision each accepted mirror was fetched at. Core bumps the version
 *  with every committed change, so a later snapshot at that revision carries
 *  the same timeline. Keyed by the mirror object, so the empty placeholder and
 *  a timeline set directly on the store never match a snapshot. */
const mirrorRevisions = new WeakMap<Timeline, { projectEpoch: number; version: number }>();

function sameStrings(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

interface ProjectState {
  /** Monotonic authority boundary for snapshot/path writes; never reset. */
  snapshotMutationRevision: number;
  projectEpoch: number;
  timelineVersion: number;
  timeline: Timeline;
  projectPath: string | null;
  compatibilityReadOnly: boolean;
  compatibilityBlockers: string[];
  /** Document version last persisted to disk; `timelineVersion` ahead of this
   *  means there are unsaved edits (drives autosave / the dirty state). */
  lastSavedVersion: number;
  /** Wall-clock time (ms) of the last completed bundle write — set by the
   *  explicit save promises and by the `project_saved` event (which also fires
   *  for core-internal saves like the media manifest). `null` until the first
   *  save of the current session. */
  lastSavedAt: number | null;
  canUndo: boolean;
  canRedo: boolean;
  /** Replace the whole authoritative snapshot. Same-project versions and
   * project epochs may only advance. The store takes ownership of an accepted
   * timeline and deep-freezes it in place, so callers must hand over a fresh
   * object (an IPC payload, a fallback clone). A snapshot at the revision the
   * mirror already holds keeps the mirror object, and changes nothing at all
   * when its path and compatibility match too. */
  replaceProjectSnapshot: (snapshot: RuntimeTimelineSnapshot) => void;
  clearProjectSnapshot: () => void;
  setProjectPath: (path: string | null) => void;
  setHistory: (canUndo: boolean, canRedo: boolean) => void;
  /** Mark the current version as persisted (called after a successful save / on
   *  open, so a freshly opened project is not considered dirty). */
  markSaved: (version?: number) => void;
  /** Record a bundle write observed via the `project_saved` event. The event
   *  carries no document version, so unlike `markSaved` this never touches the
   *  dirty-state version — only the last-saved timestamp. */
  recordSaveCompleted: () => void;
}

export const useProjectStore = create<ProjectState>((set) => ({
  snapshotMutationRevision: 0,
  projectEpoch: 0,
  timelineVersion: 0,
  timeline: EMPTY_TIMELINE,
  projectPath: null,
  compatibilityReadOnly: false,
  compatibilityBlockers: [],
  lastSavedVersion: 0,
  lastSavedAt: null,
  canUndo: false,
  canRedo: false,
  replaceProjectSnapshot: (snapshot) =>
    set((state) => {
      if (
        snapshot.projectEpoch < state.projectEpoch ||
        (snapshot.projectEpoch === state.projectEpoch && snapshot.version < state.timelineVersion)
      ) {
        return state;
      }
      const projectChanged = state.projectEpoch !== snapshot.projectEpoch;
      const mirrored = mirrorRevisions.get(state.timeline);
      const sameTimeline =
        mirrored?.projectEpoch === snapshot.projectEpoch &&
        mirrored.version === snapshot.version &&
        state.projectEpoch === snapshot.projectEpoch &&
        state.timelineVersion === snapshot.version;
      if (
        sameTimeline &&
        state.projectPath === snapshot.projectPath &&
        state.compatibilityReadOnly === snapshot.compatibilityReadOnly &&
        sameStrings(state.compatibilityBlockers, snapshot.compatibilityBlockers)
      ) {
        return state;
      }
      let timeline = state.timeline;
      if (!sameTimeline) {
        timeline = deepFreeze(snapshot.timeline);
        mirrorRevisions.set(timeline, {
          projectEpoch: snapshot.projectEpoch,
          version: snapshot.version,
        });
      }
      return {
        snapshotMutationRevision: state.snapshotMutationRevision + 1,
        projectEpoch: snapshot.projectEpoch,
        timelineVersion: snapshot.version,
        timeline,
        projectPath: snapshot.projectPath,
        compatibilityReadOnly: snapshot.compatibilityReadOnly,
        compatibilityBlockers: snapshot.compatibilityBlockers,
        ...(projectChanged
          ? {
              lastSavedVersion: snapshot.version,
              lastSavedAt: null,
              canUndo: false,
              canRedo: false,
            }
          : {}),
      };
    }),
  clearProjectSnapshot: () =>
    set((state) => ({
      snapshotMutationRevision: state.snapshotMutationRevision + 1,
      projectEpoch: 0,
      timelineVersion: 0,
      timeline: EMPTY_TIMELINE,
      projectPath: null,
      compatibilityReadOnly: false,
      compatibilityBlockers: [],
      lastSavedVersion: 0,
      lastSavedAt: null,
      canUndo: false,
      canRedo: false,
    })),
  setProjectPath: (projectPath) =>
    set((state) => ({
      snapshotMutationRevision: state.snapshotMutationRevision + 1,
      projectPath,
    })),
  setHistory: (canUndo, canRedo) => set({ canUndo, canRedo }),
  markSaved: (version) =>
    set((state) => ({
      lastSavedVersion: version ?? state.timelineVersion,
      lastSavedAt: Date.now(),
    })),
  recordSaveCompleted: () => set({ lastSavedAt: Date.now() }),
}));
