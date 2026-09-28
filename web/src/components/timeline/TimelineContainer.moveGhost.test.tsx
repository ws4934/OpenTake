// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { clipRect } from "../../lib/geometry";
import { LAYOUT } from "../../lib/theme";
import type { Clip, Timeline } from "../../lib/types";
import * as edit from "../../store/editActions";
import { useProjectStore } from "../../store/projectStore";
import { useEditorUiStore } from "../../store/uiStore";

const paint = vi.hoisted(() => ({ calls: [] as Array<{ drag?: unknown }> }));

vi.mock("./timelineCanvas", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./timelineCanvas")>();
  return {
    ...actual,
    paintTimeline: (_ctx: unknown, options: { drag?: unknown }) => {
      paint.calls.push(options);
    },
  };
});

import { TimelineContainer } from "./TimelineContainer";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
  .IS_REACT_ACT_ENVIRONMENT = true;

function videoClip(id: string, startFrame: number, durationFrames: number): Clip {
  return {
    id,
    mediaRef: `media-${id}`,
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
  };
}

const early = videoClip("early", 20, 30);
const lead = videoClip("lead", 100, 50);
const timeline: Timeline = {
  fps: 30,
  width: 1920,
  height: 1080,
  settingsConfigured: true,
  tracks: [
    { id: "v1", name: "V1", type: "video", muted: false, hidden: false, syncLocked: false, clips: [early, lead] },
  ],
};

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  paint.calls = [];
  vi.stubGlobal("ResizeObserver", class {
    observe() {}
    disconnect() {}
  });
  vi.spyOn(HTMLElement.prototype, "setPointerCapture").mockImplementation(() => {});
  vi.spyOn(HTMLElement.prototype, "releasePointerCapture").mockImplementation(() => {});
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(() =>
    ({ left: 0, top: 0, width: 1200, height: 500, right: 1200, bottom: 500 }) as DOMRect,
  );
  vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockReturnValue(LAYOUT.trackHeaderWidth + 1000);
  vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockReturnValue(500);
  // Only the clip canvas paints through the (captured) paintTimeline; the
  // ruler canvas stays context-less as in plain happy-dom.
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(function (
    this: HTMLCanvasElement,
  ) {
    return this === container.querySelectorAll("canvas")[0]
      ? ({} as unknown as CanvasRenderingContext2D)
      : null;
  } as unknown as HTMLCanvasElement["getContext"]);
  useProjectStore.setState({ timeline, projectEpoch: 1, compatibilityReadOnly: false });
  useEditorUiStore.setState({
    zoomScale: 1,
    minZoomScale: 0.01,
    activeFrame: 5_000,
    isPlaying: false,
    isScrubbing: false,
    selectedClipIds: new Set([early.id, lead.id]),
    selectedGap: null,
    selectedTimelineRange: null,
    trackDisplayHeights: {},
    toolMode: "pointer",
    scrollLeft: 0,
    scrollTop: 0,
  });
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  act(() => root.render(<TimelineContainer />));
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  useProjectStore.getState().clearProjectSnapshot();
});

it("shows the group-floor clamp in the ghost that the move commits", () => {
  const moveClips = vi.spyOn(edit, "moveClips").mockResolvedValue();
  const rect = clipRect(timeline, 0, lead, 1, {});
  const y = rect.y + rect.height / 2;
  const canvas = container.querySelectorAll("canvas")[0]!;
  const pointer = (type: string, frame: number) =>
    act(() => {
      canvas.dispatchEvent(
        new PointerEvent(type, {
          bubbles: true,
          button: 0,
          buttons: type === "pointerup" ? 0 : 1,
          clientX: LAYOUT.trackHeaderWidth + frame,
          clientY: y,
          pointerId: 1,
        }),
      );
    });

  pointer("pointerdown", 125);
  pointer("pointermove", 95);
  pointer("pointermove", 65);

  const ghost = paint.calls.at(-1)?.drag as { kind: string; deltaFrames: number } | undefined;
  expect(ghost).toMatchObject({ kind: "move", deltaFrames: -early.startFrame });

  pointer("pointerup", 65);
  expect(moveClips).toHaveBeenCalledOnce();
  expect(moveClips.mock.calls[0]?.[0]).toEqual(
    expect.arrayContaining([
      expect.objectContaining({ clipId: lead.id, toFrame: lead.startFrame + ghost!.deltaFrames }),
      expect.objectContaining({ clipId: early.id, toFrame: 0 }),
    ]),
  );
});
