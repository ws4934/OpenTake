/**
 * Edit serialization: a strict core emulation rejects every edit whose
 * identity is not the current document version (like Rust `StaleProject`),
 * and never emits `timeline_changed`, so the mirror only advances through
 * explicit refreshes.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ProjectEditIdentity, Timeline } from "../lib/types";

const core = vi.hoisted(() => ({
  epoch: 1,
  version: 5,
  opacity: 1,
  clipStart: 10,
  sentVersions: [] as number[],
  getTimelineCalls: 0,
  failGetTimeline: 0,
}));

vi.mock("../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/api")>();
  const check = (expected: ProjectEditIdentity) => {
    core.sentVersions.push(expected.timelineVersion);
    if (expected.projectEpoch !== core.epoch || expected.timelineVersion !== core.version) {
      throw new actual.TauriCommandError("staleProject", "stale project edit identity");
    }
  };
  const result = (actionName: string) => ({
    changed: true,
    actionName,
    affectedClipIds: [],
    timelineVersion: core.version,
    summary: "",
  });
  return {
    ...actual,
    isTauri: true,
    editApply: async (
      command: { type: string; properties?: { opacity?: number }; moves?: Array<{ toFrame: number }> },
      expected: ProjectEditIdentity,
    ) => {
      await Promise.resolve();
      check(expected);
      if (command.type === "setClipProperties") core.opacity = command.properties?.opacity ?? core.opacity;
      if (command.type === "moveClips") core.clipStart = command.moves?.[0]?.toFrame ?? core.clipStart;
      core.version += 1;
      return result(command.type);
    },
    undo: async (expected: ProjectEditIdentity) => {
      await Promise.resolve();
      check(expected);
      core.version += 1;
      return result("undo");
    },
    redo: async (expected: ProjectEditIdentity) => {
      await Promise.resolve();
      check(expected);
      core.version += 1;
      return result("redo");
    },
    getTimeline: async () => {
      core.getTimelineCalls += 1;
      if (core.failGetTimeline > 0) {
        core.failGetTimeline -= 1;
        throw new Error("transient get_timeline failure");
      }
      return {
        timeline: timelineAt(core.clipStart, core.opacity),
        projectEpoch: core.epoch,
        version: core.version,
        projectPath: null,
        compatibilityReadOnly: false,
        compatibilityBlockers: [],
      };
    },
    canUndo: async () => true,
    canRedo: async () => false,
  };
});

function timelineAt(clipStart: number, opacity: number): Timeline {
  return {
    fps: 30,
    width: 1920,
    height: 1080,
    settingsConfigured: true,
    tracks: [
      {
        id: "v1",
        type: "video",
        muted: false,
        hidden: false,
        syncLocked: false,
        clips: [
          {
            id: "clip-1",
            mediaRef: "media-1",
            mediaType: "video",
            sourceClipType: "video",
            startFrame: clipStart,
            durationFrames: 30,
            trimStartFrame: 0,
            trimEndFrame: 30,
            speed: 1,
            volume: 1,
            fadeInFrames: 0,
            fadeOutFrames: 0,
            fadeInInterpolation: "linear",
            fadeOutInterpolation: "linear",
            opacity,
            transform: {
              centerX: 0.5,
              centerY: 0.5,
              width: 1,
              height: 1,
              rotation: 0,
              flipHorizontal: false,
              flipVertical: false,
            },
            crop: { left: 0, top: 0, right: 0, bottom: 0 },
          },
        ],
      },
    ],
  };
}

import {
  moveClips,
  nudgeSelectedClips,
  redo,
  runTimelineEdit,
  setClipProperties,
  undo,
} from "./editActions";
import { useEditorUiStore } from "./uiStore";
import { useProjectStore } from "./projectStore";

let epochs = 0;

beforeEach(() => {
  // A fresh project per test: the queue remembers the last committed edit.
  core.epoch = ++epochs;
  core.version = 5;
  core.opacity = 1;
  core.clipStart = 10;
  core.sentVersions = [];
  core.getTimelineCalls = 0;
  core.failGetTimeline = 0;
  useProjectStore.getState().clearProjectSnapshot();
  useProjectStore.getState().replaceProjectSnapshot({
    timeline: timelineAt(core.clipStart, core.opacity),
    version: core.version,
    projectEpoch: core.epoch,
    projectPath: null,
    compatibilityReadOnly: false,
    compatibilityBlockers: [],
  });
  useEditorUiStore.setState({ toast: null, selectedClipIds: new Set(), activeNestedSequenceId: null });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("edit queue", () => {
  it("applies two rapid undos in order (5 -> 6 -> 7)", async () => {
    await expect(Promise.all([undo(), undo()])).resolves.toEqual([undefined, undefined]);
    expect(core.sentVersions).toEqual([5, 6]);
    expect(core.version).toBe(7);
  });

  it("applies undo then redo fired together", async () => {
    await Promise.all([undo(), redo()]);
    expect(core.sentVersions).toEqual([5, 6]);
    expect(core.version).toBe(7);
  });

  it("resolves three concurrent property edits and keeps the last value", async () => {
    await expect(
      Promise.all([
        setClipProperties(["clip-1"], { opacity: 0.2 }),
        setClipProperties(["clip-1"], { opacity: 0.4 }),
        setClipProperties(["clip-1"], { opacity: 0.6 }),
      ]),
    ).resolves.toHaveLength(3);
    expect(core.sentVersions).toEqual([5, 6, 7]);
    expect(core.opacity).toBe(0.6);
  });

  it("plans each nudge from the previous nudge's committed position", async () => {
    useEditorUiStore.setState({ selectedClipIds: new Set(["clip-1"]) });
    await Promise.all([nudgeSelectedClips(1), nudgeSelectedClips(1), nudgeSelectedClips(1)]);
    expect(core.clipStart).toBe(13);
  });

  it("applies different concurrent edits in order", async () => {
    const results = await Promise.allSettled([
      moveClips([{ clipId: "clip-1", toTrack: 0, toFrame: 20 }]),
      setClipProperties(["clip-1"], { opacity: 0.5 }),
    ]);
    expect(results.map((r) => r.status)).toEqual(["fulfilled", "fulfilled"]);
    expect(core.clipStart).toBe(20);
    expect(core.opacity).toBe(0.5);
  });

  it("drops an edit queued before the project changed", async () => {
    const first = setClipProperties(["clip-1"], { opacity: 0.3 });
    const second = setClipProperties(["clip-1"], { opacity: 0.9 });
    // Another project replaces this one while the queue is still draining.
    useProjectStore.setState({ projectEpoch: core.epoch + 100, projectPath: "/other.opentake" });
    await first.catch(() => undefined);
    await expect(second).rejects.toThrow(/project changed/);
    expect(core.opacity).not.toBe(0.9);
  });
});

describe("stale identity recovery", () => {
  // A failed `timeline_changed` refresh leaves the mirror behind core.
  function advanceCoreBehindMirror(): void {
    core.version += 1;
  }

  it("resyncs after a rejected edit so the next edit succeeds", async () => {
    advanceCoreBehindMirror();

    await expect(moveClips([{ clipId: "clip-1", toTrack: 0, toFrame: 40 }])).rejects.toMatchObject({
      code: "staleProject",
    });
    expect(core.getTimelineCalls).toBeGreaterThan(0);
    expect(useProjectStore.getState().timelineVersion).toBe(6);

    await moveClips([{ clipId: "clip-1", toTrack: 0, toFrame: 40 }]);
    expect(core.clipStart).toBe(40);
  });

  it("retries an absolute property edit once against the refreshed mirror", async () => {
    advanceCoreBehindMirror();

    await setClipProperties(["clip-1"], { opacity: 0.25 });

    expect(core.sentVersions).toEqual([5, 6]);
    expect(core.opacity).toBe(0.25);
  });

  it("retries undo after a stale rejection", async () => {
    advanceCoreBehindMirror();
    await undo();
    expect(core.sentVersions).toEqual([5, 6]);
    expect(core.version).toBe(7);
  });

  it("reports the rejection when the resync itself fails", async () => {
    advanceCoreBehindMirror();
    core.failGetTimeline = 1;
    await expect(setClipProperties(["clip-1"], { opacity: 0.25 })).rejects.toMatchObject({
      code: "staleProject",
    });
    await setClipProperties(["clip-1"], { opacity: 0.25 });
    expect(core.opacity).toBe(0.25);
  });
});

describe("runTimelineEdit", () => {
  it("turns a rejected gesture edit into a toast without an unhandled rejection", async () => {
    const unhandled = vi.fn();
    process.on("unhandledRejection", unhandled);
    try {
      runTimelineEdit(Promise.reject(new Error("ripple collision")));
      await new Promise((resolve) => setTimeout(resolve, 0));
      expect(useEditorUiStore.getState().toast?.message).toContain("ripple collision");
      expect(unhandled).not.toHaveBeenCalled();
    } finally {
      process.off("unhandledRejection", unhandled);
    }
  });
});
