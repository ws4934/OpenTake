// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { LAYOUT } from "../../lib/theme";
import type { Clip, Timeline } from "../../lib/types";
import { useProjectStore } from "../../store/projectStore";
import { useEditorUiStore } from "../../store/uiStore";

const paint = vi.hoisted(() => ({ timeline: 0, ruler: 0 }));

vi.mock("./timelineCanvas", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./timelineCanvas")>();
  return {
    ...actual,
    paintTimeline: () => {
      paint.timeline += 1;
    },
  };
});

vi.mock("./rulerCanvas", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./rulerCanvas")>();
  return {
    ...actual,
    paintRuler: () => {
      paint.ruler += 1;
    },
  };
});

import { TimelineContainer } from "./TimelineContainer";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
  .IS_REACT_ACT_ENVIRONMENT = true;

function videoClip(id: string, startFrame: number): Clip {
  return {
    id,
    mediaRef: `media-${id}`,
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
  };
}

const timeline: Timeline = {
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
      clips: [videoClip("a", 0), videoClip("b", 40)],
    },
  ],
};

const noopContext = { save() {}, restore() {}, clearRect() {} };

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  paint.timeline = 0;
  paint.ruler = 0;
  vi.stubGlobal("ResizeObserver", class {
    observe() {}
    disconnect() {}
  });
  vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockReturnValue(LAYOUT.trackHeaderWidth + 1000);
  vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockReturnValue(500);
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
    noopContext as unknown as CanvasRenderingContext2D,
  );
  useProjectStore.setState({ timeline, projectEpoch: 1, compatibilityReadOnly: false });
  useEditorUiStore.setState({
    zoomScale: 1,
    minZoomScale: 0.01,
    activeFrame: 0,
    isPlaying: false,
    isScrubbing: false,
    selectedClipIds: new Set(),
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
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  useProjectStore.getState().clearProjectSnapshot();
});

it("keeps the canvas backing stores across repaints of an unchanged viewport", () => {
  const widthWrites = vi.spyOn(HTMLCanvasElement.prototype, "width", "set");
  const heightWrites = vi.spyOn(HTMLCanvasElement.prototype, "height", "set");
  act(() => root.render(<TimelineContainer />));
  const [content, ruler] = Array.from(container.querySelectorAll("canvas"));
  expect(paint.timeline).toBeGreaterThan(0);
  expect(paint.ruler).toBeGreaterThan(0);
  expect(content?.width).toBe(Math.ceil(1000 * (window.devicePixelRatio || 1)));
  expect(ruler?.height).toBe(Math.ceil(LAYOUT.rulerHeight * (window.devicePixelRatio || 1)));
  widthWrites.mockClear();
  heightWrites.mockClear();
  const paintsBefore = paint.timeline;
  const rulerPaintsBefore = paint.ruler;

  act(() => useEditorUiStore.setState({ selectedClipIds: new Set(["a"]) }));
  act(() => useEditorUiStore.setState({ scrollLeft: 12 }));
  act(() => useEditorUiStore.setState({ zoomScale: 2 }));

  expect(paint.timeline).toBeGreaterThanOrEqual(paintsBefore + 3);
  expect(paint.ruler).toBeGreaterThanOrEqual(rulerPaintsBefore + 2);
  expect(widthWrites).not.toHaveBeenCalled();
  expect(heightWrites).not.toHaveBeenCalled();
});
