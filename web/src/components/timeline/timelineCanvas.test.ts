/**
 * Which clips `paintTimeline` hands to `drawClip`, in which order and with which
 * rects and badges, recorded against a no-op 2D context.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { Clip, Timeline, Track } from "../../lib/types";
import { LAYOUT, RANGE, TRACK_SIZE } from "../../lib/theme";

interface DrawnClip {
  id: string;
  rect: { x: number; y: number; width: number; height: number };
  ghost: boolean;
  isSelected: boolean;
  linkOffset: number | null | undefined;
}

const drawn = vi.hoisted(() => ({ clips: [] as DrawnClip[] }));

vi.mock("./clipRenderer", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./clipRenderer")>();
  return {
    ...actual,
    drawClip: (...args: Parameters<typeof actual.drawClip>) => {
      const [, clip, rect, opts] = args;
      drawn.clips.push({
        id: clip.id,
        rect: { ...rect },
        ghost: opts.ghost === true,
        isSelected: opts.isSelected,
        linkOffset: opts.linkOffset,
      });
      actual.drawClip(...args);
    },
  };
});

import { paintTimeline, type PaintState } from "./timelineCanvas";

function clip(id: string, startFrame: number, durationFrames: number, over: Partial<Clip> = {}): Clip {
  return {
    id,
    mediaRef: `${id}-media`,
    mediaType: "video",
    sourceClipType: "video",
    startFrame,
    durationFrames,
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
    ...over,
  };
}

function track(id: string, type: Track["type"], clips: Clip[]): Track {
  return { id, type, muted: false, hidden: false, syncLocked: true, clips };
}

/** Transition, link groups (lead, trailing partner, three-clip group, tie) and
 *  an unlinked clip, on four 50 px rows starting at y = 84. */
function timeline(): Timeline {
  return {
    fps: 30,
    width: 1920,
    height: 1080,
    settingsConfigured: true,
    tracks: [
      track("v2", "video", [
        clip("t1", 0, 40, {
          transitionOut: { fromClipId: "t1", toClipId: "t2", kind: "crossDissolve", durationFrames: 10 },
        }),
        clip("t2", 40, 40),
        clip("far", 5_000, 40),
      ]),
      track("v1", "video", [
        clip("vid", 10, 60, { linkGroupId: "L1" }),
        clip("vid2", 100, 50, { linkGroupId: "L2" }),
        clip("tie-a", 200, 20, { linkGroupId: "L3" }),
      ]),
      track("a1", "audio", [
        clip("aud", 13, 60, { mediaType: "audio", sourceClipType: "audio", linkGroupId: "L1" }),
        clip("aud2", 95, 50, { mediaType: "audio", sourceClipType: "audio", linkGroupId: "L2" }),
        clip("tie-b", 200, 20, { mediaType: "audio", sourceClipType: "audio", linkGroupId: "L3" }),
      ]),
      track("a2", "audio", [
        clip("music", 0, 300, { mediaType: "audio", sourceClipType: "audio" }),
        clip("aud3", 21, 40, { mediaType: "audio", sourceClipType: "audio", linkGroupId: "L1" }),
      ]),
    ],
  };
}

function noopCtx(): { ctx: CanvasRenderingContext2D; fills: string[] } {
  const fills: string[] = [];
  let fillStyle = "";
  const stub = {
    set fillStyle(value: string) { fillStyle = value; },
    get fillStyle() { return fillStyle; },
    strokeStyle: "",
    lineWidth: 1,
    font: "",
    textAlign: "left",
    textBaseline: "alphabetic",
    globalAlpha: 1,
    setTransform() {},
    clearRect() {},
    fillRect() {},
    strokeRect() {},
    fillText() {},
    beginPath() {},
    moveTo() {},
    lineTo() {},
    stroke() {},
    fill() { fills.push(fillStyle); },
    setLineDash() {},
    save() {},
    restore() {},
    rect() {},
    arc() {},
    arcTo() {},
    closePath() {},
    clip() {},
    measureText() { return { width: 10 }; },
    drawImage() {},
  };
  return { ctx: stub as unknown as CanvasRenderingContext2D, fills };
}

function state(over: Partial<PaintState> = {}): PaintState {
  return {
    timeline: timeline(),
    pixelsPerFrame: 2,
    trackHeights: {},
    selectedClipIds: new Set(["vid"]),
    dpr: 1,
    width: 12_000,
    height: 600,
    firstAudioIndex: 2,
    scrollLeft: 0,
    scrollTop: 0,
    viewWidth: 1_000,
    viewHeight: 600,
    waveforms: new Map(),
    thumbnails: new Map(),
    missingMediaRefs: new Set(),
    emptyLabel: "",
    ...over,
  };
}

const ROW_TOP = LAYOUT.rulerHeight + LAYOUT.dropZoneHeight;
const ROW = TRACK_SIZE.defaultHeight;

function rect(row: number, startFrame: number, durationFrames: number, ppf = 2) {
  return { x: startFrame * ppf, y: ROW_TOP + row * ROW + 2, width: durationFrames * ppf, height: ROW - 4 };
}

beforeEach(() => {
  drawn.clips = [];
});

describe("paintTimeline clip pass", () => {
  it("draws every clip in the viewport in track order with its link offset", () => {
    const { ctx, fills } = noopCtx();

    paintTimeline(ctx, state());

    expect(drawn.clips).toEqual([
      { id: "t1", rect: rect(0, 0, 40), ghost: false, isSelected: false, linkOffset: null },
      { id: "t2", rect: rect(0, 40, 40), ghost: false, isSelected: false, linkOffset: null },
      { id: "vid", rect: rect(1, 10, 60), ghost: false, isSelected: true, linkOffset: null },
      { id: "vid2", rect: rect(1, 100, 50), ghost: false, isSelected: false, linkOffset: 5 },
      { id: "tie-a", rect: rect(1, 200, 20), ghost: false, isSelected: false, linkOffset: null },
      { id: "aud", rect: rect(2, 13, 60), ghost: false, isSelected: false, linkOffset: 3 },
      { id: "aud2", rect: rect(2, 95, 50), ghost: false, isSelected: false, linkOffset: null },
      { id: "tie-b", rect: rect(2, 200, 20), ghost: false, isSelected: false, linkOffset: null },
      { id: "music", rect: rect(3, 0, 300), ghost: false, isSelected: false, linkOffset: null },
      { id: "aud3", rect: rect(3, 21, 40), ghost: false, isSelected: false, linkOffset: 11 },
    ]);
    // One cross-dissolve marker, on the t1 -> t2 cut.
    expect(fills.filter((style) => style === RANGE.edge)).toHaveLength(1);
  });

  it("draws move ghosts at their live rows and trim ghosts in place", () => {
    const { ctx } = noopCtx();

    paintTimeline(ctx, state({
      drag: {
        kind: "move",
        ids: new Set(["vid", "aud"]),
        deltaFrames: 10,
        trackDelta: 1,
        leadTrackIndex: 1,
      },
    }));
    const moved = drawn.clips.filter((entry) => entry.ghost);
    expect(moved).toEqual([
      { id: "vid", rect: rect(2, 20, 60), ghost: true, isSelected: true, linkOffset: null },
      { id: "aud", rect: rect(3, 23, 60), ghost: true, isSelected: false, linkOffset: 3 },
    ]);

    drawn.clips = [];
    paintTimeline(ctx, state({
      drag: {
        kind: "trim",
        clipId: "vid",
        edge: "left",
        deltaFrames: 5,
        propagateToLinked: true,
        linkGroupId: "L1",
      },
    }));
    expect(drawn.clips.filter((entry) => entry.ghost).map((entry) => [entry.id, entry.rect])).toEqual([
      ["vid", { ...rect(1, 10, 60), x: 30, width: 110 }],
      ["aud", { ...rect(2, 13, 60), x: 36, width: 110 }],
      ["aud3", { ...rect(3, 21, 40), x: 52, width: 70 }],
    ]);
  });

  it("skips the clips of rows scrolled out of view", () => {
    const { ctx } = noopCtx();

    // Rows 1 and 2 (y 134-234) meet the 146-216 view; row 0 ends 12 px above
    // it and row 3 starts 18 px below it.
    paintTimeline(ctx, state({ scrollTop: 146, viewHeight: 70 }));

    expect(drawn.clips.map((entry) => entry.id)).toEqual([
      "vid",
      "vid2",
      "tie-a",
      "aud",
      "aud2",
      "tie-b",
    ]);
  });

  it("still draws a row whose envelope dots can reach into the view", () => {
    const { ctx } = noopCtx();

    // Row 2 ends at y 234, 3 px above the view: a volume dot on its body's
    // bottom edge spills past the row into view.
    paintTimeline(ctx, state({ scrollTop: 237, viewHeight: 40 }));

    expect(drawn.clips.map((entry) => entry.id)).toEqual([
      "aud",
      "aud2",
      "tie-b",
      "music",
      "aud3",
    ]);
  });

  it("draws move and swap ghosts carried onto visible rows from rows out of view", () => {
    const { ctx } = noopCtx();

    // Rows 0 and 1 end above the 190-290 view.
    paintTimeline(ctx, state({
      scrollTop: 190,
      viewHeight: 100,
      drag: {
        kind: "move",
        ids: new Set(["t1"]),
        deltaFrames: 0,
        trackDelta: 2,
        leadTrackIndex: 0,
        swap: { clipId: "vid2", toTrackIndex: 3, toFrame: 100 },
      },
    }));

    expect(drawn.clips.map((entry) => [entry.id, entry.ghost, entry.rect])).toEqual([
      ["t1", true, rect(2, 0, 40)],
      ["vid2", true, rect(3, 100, 50)],
      ["aud", false, rect(2, 13, 60)],
      ["aud2", false, rect(2, 95, 50)],
      ["tie-b", false, rect(2, 200, 20)],
      ["music", false, rect(3, 0, 300)],
      ["aud3", false, rect(3, 21, 40)],
    ]);
  });
});
