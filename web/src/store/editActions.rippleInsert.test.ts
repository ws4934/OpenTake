/**
 * A ⌘/Ctrl media drop ripple-inserts into the timeline shown in the editor.
 * Inside a compound clip that is the nested sequence, whose tracks, frame rate
 * and canvas can differ from the root timeline's.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MediaItem, Timeline } from "../lib/types";

const fixtures = vi.hoisted(() => {
  const track = (id: string, type: "video" | "audio") => ({
    id,
    type,
    muted: false,
    hidden: false,
    syncLocked: true,
    clips: [],
  });
  // The root's first track is audio; the compound clip's first track is video,
  // at another frame rate and a portrait canvas.
  const nested = {
    fps: 24,
    width: 1080,
    height: 1920,
    settingsConfigured: true,
    tracks: [track("nested-video", "video")],
  };
  const root = {
    fps: 30,
    width: 1920,
    height: 1080,
    settingsConfigured: true,
    tracks: [track("root-audio", "audio"), track("root-video", "video")],
    nestedSequences: [{ id: "seq-1", name: "Compound", timeline: nested }],
  };
  return { root, nested, version: 1, commands: [] as unknown[] };
});

vi.mock("../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/api")>();
  return {
    ...actual,
    isTauri: true,
    editApply: async (command: unknown) => {
      fixtures.commands.push(command);
      fixtures.version += 1;
      return {
        changed: true,
        actionName: "Insert Clips",
        affectedClipIds: [],
        timelineVersion: fixtures.version,
        summary: "",
      };
    },
    getTimeline: async () => ({
      timeline: structuredClone(fixtures.root),
      projectEpoch: 1,
      version: fixtures.version,
      projectPath: "/projects/Compound.opentake",
      compatibilityReadOnly: false,
      compatibilityBlockers: [],
    }),
    canUndo: async () => true,
    canRedo: async () => false,
  };
});

import { buildMediaInsertPlan, rippleInsertMediaAt } from "./editActions";
import { useProjectStore } from "./projectStore";
import { useEditorUiStore } from "./uiStore";

const video: MediaItem = {
  id: "shot",
  name: "shot.mp4",
  type: "video",
  duration: 2,
  width: 1920,
  height: 1080,
  hasAudio: false,
};

function openProject(activeNestedSequenceId: string | null): void {
  useProjectStore.getState().replaceProjectSnapshot({
    timeline: structuredClone(fixtures.root) as Timeline,
    projectEpoch: 1,
    version: fixtures.version,
    projectPath: "/projects/Compound.opentake",
    compatibilityReadOnly: false,
    compatibilityBlockers: [],
  });
  useEditorUiStore.setState({ activeNestedSequenceId });
}

beforeEach(() => {
  fixtures.version = 1;
  fixtures.commands = [];
  useProjectStore.getState().clearProjectSnapshot();
});

afterEach(() => {
  useEditorUiStore.setState({ activeNestedSequenceId: null });
});

describe("rippleInsertMediaAt", () => {
  it("plans a drop inside a compound clip against the nested sequence", async () => {
    openProject("seq-1");

    expect(rippleInsertMediaAt(video, 12, 0)).toBe(true);
    await vi.waitFor(() => expect(fixtures.commands).toHaveLength(1));

    const expected = buildMediaInsertPlan(fixtures.nested as Timeline, video, 12, 0);
    expect(expected).not.toBeNull();
    // 2 s at the nested sequence's 24 fps, on its video track 0.
    expect(expected!.trackIndex).toBe(0);
    expect(expected!.entries[0].durationFrames).toBe(48);
    expect(fixtures.commands[0]).toEqual({
      type: "editNestedSequence",
      sequenceId: "seq-1",
      command: {
        type: "insertClips",
        trackIndex: expected!.trackIndex,
        atFrame: 12,
        entries: expected!.entries,
      },
    });
    // The root timeline would have chosen another track, length and fit.
    const fromRoot = buildMediaInsertPlan(fixtures.root as Timeline, video, 12, 0);
    expect(fromRoot!.trackIndex).toBe(1);
    expect(fromRoot!.entries[0].durationFrames).toBe(60);
    expect(fromRoot!.entries[0].transform).not.toEqual(expected!.entries[0].transform);
  });

  it("plans a root drop against the root timeline", async () => {
    openProject(null);

    expect(rippleInsertMediaAt(video, 12, 1)).toBe(true);
    await vi.waitFor(() => expect(fixtures.commands).toHaveLength(1));

    const expected = buildMediaInsertPlan(fixtures.root as Timeline, video, 12, 1);
    expect(fixtures.commands[0]).toEqual({
      type: "insertClips",
      trackIndex: 1,
      atFrame: 12,
      entries: expected!.entries,
    });
  });

  it("falls back to the overwrite add when the shown timeline has no compatible track", () => {
    openProject("seq-1");
    const audio: MediaItem = { ...video, id: "voice", name: "voice.wav", type: "audio" };

    // The compound clip has no audio track, although the root does.
    expect(rippleInsertMediaAt(audio, 0, 0)).toBe(false);
    expect(fixtures.commands).toEqual([]);
  });
});
