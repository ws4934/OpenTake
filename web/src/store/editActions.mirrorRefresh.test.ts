/**
 * Mirror refreshes after an edit under the desktop shell: core announces every
 * commit with `timeline_changed`, and the event-driven refresh is the only
 * `get_timeline` an edit needs, whether the event arrives before the edit
 * result, after it, or never.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Clip, ClipEntryReq, ProjectEditIdentity, Timeline } from "../lib/types";

type EventTiming = "beforeResult" | "afterResult" | "lost";

const core = vi.hoisted(() => ({
  epoch: 1,
  version: 1,
  clipIds: ["clip-1", "clip-2"] as string[],
  eventTiming: "beforeResult" as "beforeResult" | "afterResult" | "lost",
  getTimelineCalls: 0,
  failGetTimeline: 0,
  editTypes: [] as string[],
  onTimelineChanged: null as null | ((projectEpoch: number, version: number) => unknown),
}));

vi.mock("../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/api")>();
  const macrotask = () => new Promise<void>((resolve) => setTimeout(resolve, 0));
  return {
    ...actual,
    isTauri: true,
    editApply: async (command: { type: string; clipIds?: string[] }, expected: ProjectEditIdentity) => {
      await macrotask();
      if (expected.projectEpoch !== core.epoch || expected.timelineVersion !== core.version) {
        throw new actual.TauriCommandError("staleProject", "stale project edit identity");
      }
      core.editTypes.push(command.type);
      core.version += 1;
      const added = `clip-v${core.version}`;
      if (command.type === "removeClips") {
        core.clipIds = core.clipIds.filter((id) => !command.clipIds?.includes(id));
      } else if (command.type !== "moveClips") {
        core.clipIds = [...core.clipIds, added];
      }
      const { epoch, version } = core;
      const announce = () => void core.onTimelineChanged?.(epoch, version);
      if (core.eventTiming === "beforeResult") announce();
      if (core.eventTiming === "afterResult") setTimeout(announce, 0);
      return {
        changed: true,
        actionName: command.type,
        affectedClipIds: command.type === "removeClips" || command.type === "moveClips" ? [] : [added],
        timelineVersion: version,
        summary: "",
      };
    },
    getTimeline: async () => {
      core.getTimelineCalls += 1;
      const snapshot = {
        timeline: timelineOf(core.clipIds),
        projectEpoch: core.epoch,
        version: core.version,
        projectPath: null,
        compatibilityReadOnly: false,
        compatibilityBlockers: [],
      };
      await macrotask();
      if (core.failGetTimeline > 0) {
        core.failGetTimeline -= 1;
        throw new Error("get_timeline unavailable");
      }
      return snapshot;
    },
    canUndo: async () => true,
    canRedo: async () => false,
    onTimelineChanged: async (handler: (projectEpoch: number, version: number) => unknown) => {
      core.onTimelineChanged = handler;
      return () => {
        core.onTimelineChanged = null;
      };
    },
    onProjectOpened: async () => () => {},
    onProjectSaved: async () => () => {},
  };
});

vi.mock("./mediaStore", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./mediaStore")>()),
  refreshMedia: async () => true,
  resetProjectMediaState: () => {},
}));

function clipAt(id: string, index: number): Clip {
  return {
    id,
    mediaRef: "media-1",
    mediaType: "video",
    sourceClipType: "video",
    startFrame: index * 40,
    durationFrames: 30,
    trimStartFrame: 0,
    trimEndFrame: 0,
    speed: 1,
    volume: 1,
    fadeInFrames: 0,
    fadeOutFrames: 0,
    fadeInInterpolation: "linear",
    fadeOutInterpolation: "linear",
    opacity: 1,
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
  };
}

function timelineOf(clipIds: string[]): Timeline {
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
        clips: clipIds.map(clipAt),
      },
    ],
  };
}

import {
  addTextClip,
  deleteSelectedClips,
  insertClips,
  nudgeSelectedClips,
  rippleDeleteSelectedClips,
} from "./editActions";
import { awaitMirrorVersion, startSync, stopSync } from "./sync";
import { useEditorUiStore } from "./uiStore";
import { useProjectStore } from "./projectStore";

let epochs = 0;

async function startProject(eventTiming: EventTiming): Promise<void> {
  core.epoch = ++epochs;
  core.version = 1;
  core.clipIds = ["clip-1", "clip-2"];
  core.eventTiming = eventTiming;
  core.failGetTimeline = 0;
  core.editTypes = [];
  await startSync();
  expect(useProjectStore.getState()).toMatchObject({
    projectEpoch: core.epoch,
    timelineVersion: 1,
  });
  core.getTimelineCalls = 0;
}

const entry: ClipEntryReq = {
  mediaRef: "media-1",
  mediaType: "video",
  sourceClipType: "video",
  trackIndex: 0,
  startFrame: 0,
  durationFrames: 30,
};

beforeEach(() => {
  useProjectStore.getState().clearProjectSnapshot();
  useEditorUiStore.setState({
    toast: null,
    selectedClipIds: new Set(),
    activeNestedSequenceId: null,
    activeFrame: 0,
    currentFrame: 0,
  });
});

afterEach(() => {
  stopSync();
});

describe("edit mirror refresh under Tauri", () => {
  it.each<EventTiming>(["beforeResult", "afterResult", "lost"])(
    "fetches the inserted revision once when its event arrives %s",
    async (timing) => {
      await startProject(timing);

      await insertClips(0, 0, [entry]);

      expect(core.getTimelineCalls).toBe(1);
      expect(useProjectStore.getState().timelineVersion).toBe(2);
      expect(useEditorUiStore.getState().selectedClipIds).toEqual(new Set(["clip-v2"]));
    },
  );

  it.each<EventTiming>(["beforeResult", "afterResult", "lost"])(
    "shows a toolbar text clip with one fetch when its event arrives %s",
    async (timing) => {
      await startProject(timing);

      await addTextClip();

      expect(core.editTypes).toEqual(["addTextsAutoTrack"]);
      expect(core.getTimelineCalls).toBe(1);
      expect(useProjectStore.getState().timelineVersion).toBe(2);
      expect(useEditorUiStore.getState().selectedClipIds).toEqual(new Set(["clip-v2"]));
    },
  );

  it.each<EventTiming>(["beforeResult", "afterResult", "lost"])(
    "removes deleted clips from the mirror with one fetch when the event arrives %s",
    async (timing) => {
      await startProject(timing);
      useEditorUiStore.setState({ selectedClipIds: new Set(["clip-1"]) });

      await deleteSelectedClips();

      expect(core.getTimelineCalls).toBe(1);
      const clips = useProjectStore.getState().timeline.tracks[0]!.clips;
      expect(clips.map((clip) => clip.id)).toEqual(["clip-2"]);
      expect(useEditorUiStore.getState().toast).toBeNull();
    },
  );

  it("does not fetch again for an edit whose event refresh already landed", async () => {
    await startProject("beforeResult");
    useEditorUiStore.setState({ selectedClipIds: new Set(["clip-2"]) });
    await rippleDeleteSelectedClips();
    expect(core.getTimelineCalls).toBe(1);

    await awaitMirrorVersion({ projectEpoch: core.epoch, version: 2 });

    expect(core.getTimelineCalls).toBe(1);
  });

  it("fetches each queued nudge's revision once", async () => {
    await startProject("beforeResult");
    useEditorUiStore.setState({ selectedClipIds: new Set(["clip-1"]) });

    await Promise.all([nudgeSelectedClips(1), nudgeSelectedClips(1), nudgeSelectedClips(1)]);
    await awaitMirrorVersion({ projectEpoch: core.epoch, version: 4 });

    expect(core.editTypes).toEqual(["moveClips", "moveClips", "moveClips"]);
    expect(core.getTimelineCalls).toBe(3);
    expect(useProjectStore.getState().timelineVersion).toBe(4);
  });

  it("falls back to a forced refresh when the event refresh cannot converge", async () => {
    await startProject("lost");
    core.failGetTimeline = 2;

    await insertClips(0, 0, [entry]);

    // Two event-refresh attempts, then the forced refresh that succeeds.
    expect(core.getTimelineCalls).toBe(3);
    expect(useProjectStore.getState().timelineVersion).toBe(2);
    expect(useEditorUiStore.getState().toast?.message).toContain("get_timeline unavailable");
  });

  it("rejects when neither the event refresh nor the forced refresh succeeds", async () => {
    await startProject("lost");
    core.failGetTimeline = 3;

    await expect(insertClips(0, 0, [entry])).rejects.toThrow("get_timeline unavailable");
    expect(useProjectStore.getState().timelineVersion).toBe(1);
  });

  it("forces a refresh when the event sync is not running", async () => {
    await startProject("lost");
    stopSync();

    await insertClips(0, 0, [entry]);

    expect(core.getTimelineCalls).toBe(1);
    expect(useProjectStore.getState().timelineVersion).toBe(2);
  });
});
