import { describe, expect, it } from "vitest";
import { currentTimelineOf } from "./timelineScope";
import type { Timeline } from "./types";

function timeline(fps: number, over: Partial<Timeline> = {}): Timeline {
  return { fps, width: 1920, height: 1080, settingsConfigured: true, tracks: [], ...over };
}

describe("currentTimelineOf", () => {
  const nested = timeline(24);
  const root = timeline(30, { nestedSequences: [{ id: "seq-1", name: "Compound", timeline: nested }] });

  it("returns the open nested sequence's timeline", () => {
    expect(currentTimelineOf(root, "seq-1")).toBe(nested);
  });

  it("returns the root outside a nested sequence and for a sequence that no longer exists", () => {
    expect(currentTimelineOf(root, null)).toBe(root);
    expect(currentTimelineOf(root, "dissolved")).toBe(root);
    expect(currentTimelineOf(timeline(30), "seq-1")).toEqual(timeline(30));
  });
});
