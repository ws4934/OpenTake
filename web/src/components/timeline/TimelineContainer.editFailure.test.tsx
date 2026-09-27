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
import { TimelineContainer } from "./TimelineContainer";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
  .IS_REACT_ACT_ENVIRONMENT = true;

const clip: Clip = {
  id: "clip-1",
  mediaRef: "media-1",
  mediaType: "video",
  sourceClipType: "video",
  startFrame: 0,
  durationFrames: 90,
  trimStartFrame: 0,
  trimEndFrame: 90,
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

const timeline: Timeline = {
  fps: 30,
  width: 1920,
  height: 1080,
  settingsConfigured: true,
  tracks: [
    { id: "v1", name: "V1", type: "video", muted: false, hidden: false, syncLocked: false, clips: [clip] },
  ],
};

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  vi.stubGlobal("ResizeObserver", class {
    observe() {}
    disconnect() {}
  });
  vi.spyOn(HTMLElement.prototype, "setPointerCapture").mockImplementation(() => {});
  vi.spyOn(HTMLElement.prototype, "releasePointerCapture").mockImplementation(() => {});
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(() =>
    ({ left: 0, top: 0, width: 1000, height: 500, right: 1000, bottom: 500 }) as DOMRect,
  );
  useProjectStore.setState({ timeline, projectEpoch: 1, compatibilityReadOnly: false });
  useEditorUiStore.setState({
    zoomScale: 1,
    minZoomScale: 0.01,
    activeFrame: 200,
    isPlaying: false,
    isScrubbing: false,
    selectedClipIds: new Set(),
    selectedGap: null,
    selectedTimelineRange: null,
    trackDisplayHeights: {},
    toolMode: "pointer",
    toast: null,
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

it("toasts a rejected clip drag instead of leaking an unhandled rejection", async () => {
  const moveClips = vi.spyOn(edit, "moveClips").mockRejectedValue(new Error("ripple collision"));
  const unhandled = vi.fn();
  const onUnhandled = (event: PromiseRejectionEvent) => unhandled(event.reason);
  window.addEventListener("unhandledrejection", onUnhandled);
  try {
    const rect = clipRect(timeline, 0, clip, 1, {});
    const y = rect.y + rect.height / 2;
    const x = (frame: number) => LAYOUT.trackHeaderWidth + frame;
    const canvas = container.querySelectorAll("canvas")[0];
    const pointer = (type: string, frame: number) =>
      new PointerEvent(type, {
        bubbles: true,
        button: 0,
        buttons: type === "pointerup" ? 0 : 1,
        clientX: x(frame),
        clientY: y,
        pointerId: 1,
      });
    await act(async () => canvas.dispatchEvent(pointer("pointerdown", 45)));
    await act(async () => canvas.dispatchEvent(pointer("pointermove", 60)));
    await act(async () => canvas.dispatchEvent(pointer("pointermove", 75)));
    await act(async () => canvas.dispatchEvent(pointer("pointerup", 75)));
    await act(async () => new Promise((resolve) => setTimeout(resolve, 0)));

    expect(moveClips).toHaveBeenCalledOnce();
    expect(useEditorUiStore.getState().toast?.message).toContain("ripple collision");
    expect(unhandled).not.toHaveBeenCalled();
  } finally {
    window.removeEventListener("unhandledrejection", onUnhandled);
  }
});
