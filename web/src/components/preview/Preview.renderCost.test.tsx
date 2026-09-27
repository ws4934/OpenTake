// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { Timeline } from "../../lib/types";

const renders = vi.hoisted(() => ({ frameBuffer: 0 }));

vi.mock("./RustFrameBuffer.tsx", () => ({
  RustFrameBuffer: () => {
    renders.frameBuffer += 1;
    return null;
  },
}));

import * as playbackRoute from "./playbackRoute";
import { useEditorUiStore } from "../../store/uiStore";
import { useProjectStore } from "../../store/projectStore";
import { Preview } from "./Preview";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT =
  true;

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
      clips: [
        {
          id: "clip-1",
          mediaRef: "media-1",
          mediaType: "video",
          sourceClipType: "video",
          startFrame: 0,
          durationFrames: 300,
          trimStartFrame: 0,
          trimEndFrame: 300,
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
        },
      ],
    },
  ],
};

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  vi.stubGlobal("ResizeObserver", class {
    observe() {}
    disconnect() {}
  });
  useProjectStore.setState({ timeline, projectEpoch: 1, timelineVersion: 1 });
  useEditorUiStore.setState({
    previewTabIds: [],
    previewTabHistory: [],
    previewActiveTabId: "timeline",
    previewMediaId: null,
    activeFrame: 10,
    currentFrame: 10,
    isPlaying: false,
    isScrubbing: false,
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

it("re-renders the timeline preview once per whole frame without re-resolving the route", async () => {
  await act(async () => root.render(<Preview />));
  expect(renders.frameBuffer).toBeGreaterThan(0);
  renders.frameBuffer = 0;
  const resolveRoute = vi.spyOn(playbackRoute, "resolveTimelinePlaybackRoute");

  // 60 fractional playback ticks across two whole-frame boundaries (10 -> 12).
  for (let tick = 1; tick <= 60; tick += 1) {
    await act(async () => useEditorUiStore.getState().setActiveFrame(10 + tick * (2 / 60)));
  }

  expect(renders.frameBuffer).toBe(2);
  expect(resolveRoute).not.toHaveBeenCalled();
});
