// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { clipRect } from "../../lib/geometry";
import { CLIP, LAYOUT, TRIM } from "../../lib/theme";
import type { Clip, Timeline } from "../../lib/types";
import * as edit from "../../store/editActions";
import { useProjectStore } from "../../store/projectStore";
import { useEditorUiStore } from "../../store/uiStore";
import { TimelineContainer } from "./TimelineContainer";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
  .IS_REACT_ACT_ENVIRONMENT = true;

const audio: Clip = {
  id: "audio-1",
  mediaRef: "media-1",
  mediaType: "audio",
  sourceClipType: "audio",
  startFrame: 100,
  durationFrames: 100,
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
  volumeTrack: { keyframes: [{ frame: 0, value: 0.5, interpolationOut: "linear" }] },
};

const timeline: Timeline = {
  fps: 30,
  width: 1920,
  height: 1080,
  settingsConfigured: true,
  tracks: [
    { id: "a1", name: "A1", type: "audio", muted: false, hidden: false, syncLocked: false, clips: [audio] },
  ],
};

let container: HTMLDivElement;
let root: Root;
let moveKeyframe: ReturnType<typeof vi.spyOn>;
let stampKeyframe: ReturnType<typeof vi.spyOn>;

function envelope(zoom: number) {
  const rect = clipRect(timeline, 0, audio, zoom, {});
  const pixelsPerFrame = (rect.width - 2 * TRIM.handleWidth) / audio.durationFrames;
  const baseX = rect.x + TRIM.handleWidth;
  const bodyTop = rect.y + CLIP.labelBarHeight;
  const bodyHeight = rect.height - CLIP.labelBarHeight;
  return {
    xFor: (frame: number) => baseX + frame * pixelsPerFrame,
    dotY: bodyTop + bodyHeight * 0.5,
    lineFreeY: bodyTop + bodyHeight * 0.9,
    pixelsPerFrame,
  };
}

function pointer(type: string, docX: number, docY: number, metaKey = false) {
  const canvas = container.querySelectorAll("canvas")[0]!;
  act(() => {
    canvas.dispatchEvent(
      new PointerEvent(type, {
        bubbles: true,
        button: 0,
        buttons: type === "pointerup" ? 0 : 1,
        clientX: LAYOUT.trackHeaderWidth + docX,
        clientY: docY,
        pointerId: 1,
        metaKey,
      }),
    );
  });
}

function renderAt(zoom: number) {
  act(() => useEditorUiStore.setState({ zoomScale: zoom }));
}

beforeEach(() => {
  vi.stubGlobal("ResizeObserver", class {
    observe() {}
    disconnect() {}
  });
  vi.spyOn(HTMLElement.prototype, "setPointerCapture").mockImplementation(() => {});
  vi.spyOn(HTMLElement.prototype, "releasePointerCapture").mockImplementation(() => {});
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(() =>
    ({ left: 0, top: 0, width: 4000, height: 500, right: 4000, bottom: 500 }) as DOMRect,
  );
  moveKeyframe = vi.spyOn(edit, "moveKeyframe").mockResolvedValue();
  stampKeyframe = vi.spyOn(edit, "stampKeyframe").mockResolvedValue();
  useProjectStore.setState({ timeline, projectEpoch: 1, compatibilityReadOnly: false });
  useEditorUiStore.setState({
    zoomScale: 1,
    minZoomScale: 0.01,
    activeFrame: 5_000,
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
  act(() => root.render(<TimelineContainer />));
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  useProjectStore.getState().clearProjectSnapshot();
});

describe("volume keyframe drag", () => {
  it.each([0.1, 1, 4])("ignores a 1px jitter on a keyframe at zoom %s", (zoom) => {
    renderAt(zoom);
    const { xFor, dotY } = envelope(zoom);
    pointer("pointerdown", xFor(0), dotY);
    pointer("pointermove", xFor(0) + 1, dotY);
    pointer("pointerup", xFor(0) + 1, dotY);

    expect(moveKeyframe).not.toHaveBeenCalled();
  });

  it.each([1, 4])("moves the keyframe to the frame drawn under the cursor at zoom %s", (zoom) => {
    renderAt(zoom);
    const { xFor, dotY } = envelope(zoom);
    pointer("pointerdown", xFor(0), dotY);
    pointer("pointermove", xFor(10), dotY);
    pointer("pointermove", xFor(20), dotY);
    pointer("pointerup", xFor(20), dotY);

    expect(moveKeyframe).toHaveBeenCalledOnce();
    expect(moveKeyframe.mock.calls[0]?.slice(0, 4)).toEqual([audio.id, "volume", 100, 120]);
  });

  it("stamps a Cmd-click keyframe at the frame drawn under the cursor", () => {
    const { xFor, lineFreeY } = envelope(1);
    pointer("pointerdown", xFor(30), lineFreeY, true);
    pointer("pointerup", xFor(30), lineFreeY, true);

    expect(stampKeyframe).toHaveBeenCalledExactlyOnceWith(audio.id, "volume", 130);
  });
});
