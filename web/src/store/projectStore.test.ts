import { beforeEach, describe, expect, it } from "vitest";
import type { Clip, RuntimeTimelineSnapshot, Timeline } from "../lib/types";
import { useProjectStore } from "./projectStore";

function clip(id: string, startFrame: number): Clip {
  return {
    id,
    mediaRef: "media-1",
    mediaType: "video",
    sourceClipType: "video",
    startFrame,
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
    positionTrack: {
      keyframes: [{ frame: 0, value: { a: 0.1, b: 0.2 }, interpolationOut: "smooth" }],
    },
    effects: [{ name: "blur", params: { radius: 2 }, enabled: true }],
    masks: [
      {
        shape: { kind: "poly", points: [{ x: 0, y: 0 }, { x: 1, y: 1 }] },
        feather: 0,
        invert: false,
      },
    ],
  };
}

/** A fresh timeline per call, the way `get_timeline` hands one over. */
function timeline(clipStart = 0): Timeline {
  return {
    fps: 30,
    width: 1920,
    height: 1080,
    settingsConfigured: true,
    nestedSequences: [
      {
        id: "sequence-1",
        name: "Compound",
        timeline: {
          fps: 30,
          width: 1920,
          height: 1080,
          settingsConfigured: true,
          tracks: [
            {
              id: "nested-v1",
              type: "video",
              muted: false,
              hidden: false,
              syncLocked: true,
              clips: [clip("nested-clip", 0)],
            },
          ],
        },
      },
    ],
    tracks: [
      {
        id: "v1",
        type: "video",
        muted: false,
        hidden: false,
        syncLocked: true,
        clips: [clip("clip-1", clipStart), clip("clip-2", clipStart + 30)],
      },
    ],
  };
}

function snapshot(overrides: Partial<RuntimeTimelineSnapshot> = {}): RuntimeTimelineSnapshot {
  return {
    timeline: timeline(),
    projectEpoch: 3,
    version: 7,
    projectPath: "/tmp/project.opentake",
    compatibilityReadOnly: false,
    compatibilityBlockers: [],
    ...overrides,
  };
}

function expectDeepFrozen(value: unknown, path = "timeline"): void {
  if (value === null || typeof value !== "object") return;
  expect(Object.isFrozen(value), path).toBe(true);
  for (const [key, child] of Object.entries(value)) expectDeepFrozen(child, `${path}.${key}`);
}

beforeEach(() => {
  useProjectStore.getState().clearProjectSnapshot();
});

describe("project snapshot mirror", () => {
  it("deep-freezes an accepted timeline so no reference can mutate the mirror", () => {
    const handedOver = snapshot();
    useProjectStore.getState().replaceProjectSnapshot(handedOver);

    const mirror = useProjectStore.getState().timeline;
    expect(mirror).toEqual(timeline());
    expectDeepFrozen(mirror);
    const firstClip = mirror.tracks[0]!.clips[0]!;
    expect(() => {
      firstClip.startFrame = 99;
    }).toThrow(TypeError);
    expect(() => {
      firstClip.effects![0]!.params.radius = 9;
    }).toThrow(TypeError);
    expect(() => {
      mirror.nestedSequences![0]!.timeline.tracks[0]!.clips.push(clip("extra", 90));
    }).toThrow(TypeError);
    // A reference the caller kept is no back door into the mirror (whether the
    // write is refused or lands on a private copy).
    try {
      handedOver.timeline.tracks[0]!.clips[1]!.opacity = 0;
    } catch (error) {
      expect(error).toBeInstanceOf(TypeError);
    }
    expect(useProjectStore.getState().timeline).toEqual(timeline());
  });

  it("keeps the mirror object for a second snapshot of the same revision", () => {
    useProjectStore.getState().replaceProjectSnapshot(snapshot());
    const before = useProjectStore.getState();
    let notifications = 0;
    const unsubscribe = useProjectStore.subscribe(() => {
      notifications += 1;
    });

    useProjectStore.getState().replaceProjectSnapshot(snapshot());
    unsubscribe();

    const after = useProjectStore.getState();
    expect(Object.is(before.timeline, after.timeline)).toBe(true);
    expect(after).toBe(before);
    expect(after.snapshotMutationRevision).toBe(before.snapshotMutationRevision);
    expect(notifications).toBe(0);
  });

  it("updates the path and compatibility of a same-revision snapshot without a new timeline", () => {
    useProjectStore.getState().replaceProjectSnapshot(snapshot());
    const before = useProjectStore.getState();

    useProjectStore.getState().replaceProjectSnapshot(
      snapshot({
        projectPath: "/tmp/saved-as.opentake",
        compatibilityReadOnly: true,
        compatibilityBlockers: ["timeline.futureField"],
      }),
    );

    const after = useProjectStore.getState();
    expect(after.timeline).toBe(before.timeline);
    expect(after.projectPath).toBe("/tmp/saved-as.opentake");
    expect(after.compatibilityReadOnly).toBe(true);
    expect(after.compatibilityBlockers).toEqual(["timeline.futureField"]);
    expect(after.snapshotMutationRevision).toBe(before.snapshotMutationRevision + 1);
  });

  it("replaces the timeline for a newer version and ignores an older one", () => {
    useProjectStore.getState().replaceProjectSnapshot(snapshot());
    const first = useProjectStore.getState().timeline;

    useProjectStore.getState().replaceProjectSnapshot(
      snapshot({ version: 8, timeline: timeline(12) }),
    );
    const second = useProjectStore.getState().timeline;
    expect(second).not.toBe(first);
    expect(second.tracks[0]!.clips[0]!.startFrame).toBe(12);
    expectDeepFrozen(second);

    useProjectStore.getState().replaceProjectSnapshot(
      snapshot({ version: 7, timeline: timeline(40) }),
    );
    expect(useProjectStore.getState().timeline).toBe(second);
    expect(useProjectStore.getState().timelineVersion).toBe(8);
  });

  it("replaces the empty placeholder even when the first snapshot has its revision", () => {
    // A fresh core and the browser fallback both start at epoch 0, version 0.
    useProjectStore.getState().replaceProjectSnapshot(
      snapshot({ projectEpoch: 0, version: 0, projectPath: null }),
    );
    expect(useProjectStore.getState().timeline.tracks).toHaveLength(1);

    useProjectStore.getState().clearProjectSnapshot();
    expect(useProjectStore.getState().timeline.tracks).toEqual([]);
    useProjectStore.getState().replaceProjectSnapshot(
      snapshot({ projectEpoch: 0, version: 0, projectPath: null }),
    );
    expect(useProjectStore.getState().timeline.tracks).toHaveLength(1);
  });

  it("replaces a timeline that was not committed from a snapshot at that revision", () => {
    useProjectStore.getState().replaceProjectSnapshot(snapshot());
    const direct = timeline(60);
    useProjectStore.setState({ timeline: direct });

    useProjectStore.getState().replaceProjectSnapshot(snapshot());

    expect(useProjectStore.getState().timeline).not.toBe(direct);
    expect(useProjectStore.getState().timeline.tracks[0]!.clips[0]!.startFrame).toBe(0);
  });
});
