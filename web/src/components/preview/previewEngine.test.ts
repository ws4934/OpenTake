import React, { StrictMode, act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const nativeApiHarness = vi.hoisted(() => {
  const harness = {
    activeListeners: 0,
    deferred: false,
    listenerAttempts: [] as Array<{
      resolve(): void;
      reject(): void;
    }>,
    order: [] as string[],
    unlistenCalls: 0,
    getPreviewEndpoint: vi.fn(),
    onPlaybackFrame: vi.fn(),
    onPlaybackError: vi.fn(),
    activeErrorListeners: 0,
    playbackStart: vi.fn(),
    playbackPause: vi.fn(),
    playbackSeek: vi.fn(),
    playbackStop: vi.fn(),
  };
  const unlisten = () => {
    harness.activeListeners -= 1;
    harness.unlistenCalls += 1;
  };
  harness.onPlaybackFrame.mockImplementation(() => {
    if (!harness.deferred) {
      harness.activeListeners += 1;
      harness.order.push("listener-ready");
      return Promise.resolve(unlisten);
    }
    return new Promise<() => void>((resolve, reject) => {
      let settled = false;
      harness.listenerAttempts.push({
        resolve() {
          if (settled) return;
          settled = true;
          harness.activeListeners += 1;
          harness.order.push("listener-ready");
          resolve(unlisten);
        },
        reject() {
          if (settled) return;
          settled = true;
          harness.order.push("listener-rejected");
          reject(new Error("listener registration rejected"));
        },
      });
    });
  });
  harness.onPlaybackError.mockImplementation(() => {
    harness.activeErrorListeners += 1;
    return Promise.resolve(() => {
      harness.activeErrorListeners -= 1;
    });
  });
  harness.playbackStart.mockImplementation(async () => {
    harness.order.push("start");
  });
  harness.playbackPause.mockResolvedValue(undefined);
  harness.playbackSeek.mockResolvedValue(undefined);
  harness.playbackStop.mockResolvedValue(undefined);
  harness.getPreviewEndpoint.mockResolvedValue("http://127.0.0.1:43123/frame");
  return harness;
});

vi.mock("../../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../lib/api")>();
  return {
    ...actual,
    isTauri: true,
    getPreviewEndpoint: nativeApiHarness.getPreviewEndpoint,
    onPlaybackFrame: nativeApiHarness.onPlaybackFrame,
    onPlaybackError: nativeApiHarness.onPlaybackError,
    playbackStart: nativeApiHarness.playbackStart,
    playbackPause: nativeApiHarness.playbackPause,
    playbackSeek: nativeApiHarness.playbackSeek,
    playbackStop: nativeApiHarness.playbackStop,
  };
});

import * as previewEngine from "./previewEngine";
import { pausedSeekToleranceSec, previewElementKey, shouldSyncPausedMediaToFrame } from "./previewEngine";
import type { ActiveMedia } from "./timelinePlayback";
import type { Clip, ClipType, Timeline, Track } from "../../lib/types";
import { useProjectStore } from "../../store/projectStore";
import { useEditorUiStore } from "../../store/uiStore";
import { nativePlaybackController, publishNativePlaybackFrame } from "./nativePlaybackSession";
import * as playbackRoute from "./playbackRoute";

function installReactHost(): Element {
  const document = {
    nodeType: 9,
    defaultView: globalThis,
    addEventListener() {},
    removeEventListener() {},
    activeElement: null,
    documentElement: null as unknown,
  };
  const container = {
    nodeType: 1,
    nodeName: "DIV",
    tagName: "DIV",
    namespaceURI: "http://www.w3.org/1999/xhtml",
    ownerDocument: document,
    addEventListener() {},
    removeEventListener() {},
    appendChild() {},
    removeChild() {},
    firstChild: null,
  };
  document.documentElement = container;
  vi.stubGlobal("window", globalThis);
  vi.stubGlobal("document", document);
  vi.stubGlobal("HTMLIFrameElement", function HTMLIFrameElement() {});
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("getSelection", () => null);
  return container as unknown as Element;
}

function PlaybackHookHarness(): null {
  previewEngine.useTimelinePlaybackEngine();
  return null;
}

async function mountPlaybackHook(strict = false): Promise<Root> {
  const root = createRoot(installReactHost());
  await act(async () => {
    root.render(
      strict
        ? React.createElement(StrictMode, null, React.createElement(PlaybackHookHarness))
        : React.createElement(PlaybackHookHarness),
    );
    await Promise.resolve();
  });
  return root;
}

async function unmountPlaybackHook(root: Root): Promise<void> {
  await act(async () => {
    root.unmount();
    await Promise.resolve();
    await Promise.resolve();
  });
}

beforeEach(async () => {
  await nativePlaybackController.stopCurrent();
  nativeApiHarness.activeListeners = 0;
  nativeApiHarness.deferred = false;
  nativeApiHarness.listenerAttempts = [];
  nativeApiHarness.order = [];
  nativeApiHarness.unlistenCalls = 0;
  nativeApiHarness.getPreviewEndpoint.mockReset().mockResolvedValue(
    "http://127.0.0.1:43123/frame",
  );
  nativeApiHarness.onPlaybackFrame.mockClear();
  nativeApiHarness.onPlaybackError.mockClear();
  nativeApiHarness.activeErrorListeners = 0;
  nativeApiHarness.playbackStart.mockClear();
  nativeApiHarness.playbackPause.mockClear();
  nativeApiHarness.playbackSeek.mockClear();
  nativeApiHarness.playbackStop.mockClear();
});

afterEach(() => vi.unstubAllGlobals());

function clip(over: Partial<Clip> & { id: string; mediaType: ClipType }): Clip {
  return {
    id: over.id,
    mediaRef: over.mediaRef ?? "asset",
    mediaType: over.mediaType,
    sourceClipType: over.mediaType,
    startFrame: over.startFrame ?? 0,
    durationFrames: over.durationFrames ?? 100,
    trimStartFrame: over.trimStartFrame ?? 0,
    trimEndFrame: over.trimEndFrame ?? 0,
    speed: over.speed ?? 1,
    volume: over.volume ?? 1,
    fadeInFrames: 0,
    fadeOutFrames: 0,
    fadeInInterpolation: "smooth",
    fadeOutInterpolation: "smooth",
    opacity: over.opacity ?? 1,
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

function track(over: Partial<Track> & { id: string; type: ClipType; clips: Clip[] }): Track {
  return {
    id: over.id,
    type: over.type,
    muted: over.muted ?? false,
    hidden: over.hidden ?? false,
    syncLocked: over.syncLocked ?? true,
    clips: over.clips,
  };
}

function timeline(tracks: Track[]): Timeline {
  return { fps: 30, width: 1920, height: 1080, settingsConfigured: true, tracks };
}

function rustTimeline(overrides: Partial<Clip> = {}): Timeline {
  return timeline([
    track({
      id: "text-track",
      type: "text",
      clips: [clip({ id: "text-clip", mediaType: "text", ...overrides })],
    }),
  ]);
}

function temporalCompositorTimeline(): Timeline {
  return timeline([
    track({
      id: "text-track",
      type: "text",
      clips: [
        clip({
          id: "text-remap",
          mediaType: "text",
          reversed: true,
          speed: 1.5,
        }),
      ],
    }),
  ]);
}

describe("shouldSyncPausedMediaToFrame", () => {
  it("registers one listener before start across a StrictMode cleanup and remount", async () => {
    nativeApiHarness.deferred = true;
    useProjectStore.setState({ projectEpoch: 4, timelineVersion: 7, timeline: rustTimeline() });
    useEditorUiStore.setState({
      activeFrame: 0,
      currentFrame: 0,
      isPlaying: true,
      isScrubbing: false,
      rustEngineFailed: false,
    });

    const root = await mountPlaybackHook(true);

    expect(nativeApiHarness.onPlaybackFrame).toHaveBeenCalledTimes(1);
    expect(nativeApiHarness.activeListeners).toBe(0);
    expect(nativeApiHarness.playbackStart).not.toHaveBeenCalled();

    await act(async () => {
      nativeApiHarness.listenerAttempts[0]?.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(nativeApiHarness.onPlaybackFrame).toHaveBeenCalledTimes(1);
    expect(nativeApiHarness.activeListeners).toBe(1);
    expect(nativeApiHarness.unlistenCalls).toBe(0);
    expect(nativeApiHarness.playbackStart).toHaveBeenCalled();
    expect(nativeApiHarness.order[0]).toBe("listener-ready");
    expect(nativeApiHarness.order.slice(1).every((entry) => entry === "start")).toBe(true);

    await unmountPlaybackHook(root);
    expect(nativeApiHarness.activeListeners).toBe(0);
    expect(nativeApiHarness.unlistenCalls).toBe(1);
  });

  it("registers the playback error listener once across play, pause and play", async () => {
    useProjectStore.setState({ projectEpoch: 4, timelineVersion: 7, timeline: rustTimeline() });
    useEditorUiStore.setState({
      activeFrame: 0,
      currentFrame: 0,
      isPlaying: true,
      isScrubbing: false,
      rustEngineFailed: false,
    });

    const root = await mountPlaybackHook();
    expect(nativeApiHarness.playbackStart).toHaveBeenCalledTimes(1);
    for (const playing of [false, true, false, true]) {
      await act(async () => {
        useEditorUiStore.getState().setPlaying(playing);
        await Promise.resolve();
        await Promise.resolve();
      });
    }

    expect(nativeApiHarness.playbackStart).toHaveBeenCalledTimes(3);
    expect(nativeApiHarness.onPlaybackError).toHaveBeenCalledTimes(1);
    expect(nativeApiHarness.activeErrorListeners).toBe(1);
    await unmountPlaybackHook(root);
    expect(nativeApiHarness.activeErrorListeners).toBe(0);
  });

  it("retires the native playback identity as soon as scrubbing begins", async () => {
    useProjectStore.setState({ projectEpoch: 4, timelineVersion: 7, timeline: rustTimeline() });
    useEditorUiStore.setState({
      activeFrame: 12,
      currentFrame: 12,
      isPlaying: true,
      isScrubbing: false,
      rustEngineFailed: false,
    });
    const root = await mountPlaybackHook();
    const started = nativePlaybackController.currentIdentity();
    expect(started).not.toBeNull();

    await act(async () => {
      useEditorUiStore.getState().setScrubbing(true);
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(nativeApiHarness.playbackStop).toHaveBeenCalledWith(started);
    expect(nativePlaybackController.currentIdentity()).toBeNull();
    await unmountPlaybackHook(root);
  });

  it("re-registers the native frame listener when PLAY retries after registration rejection", async () => {
    nativeApiHarness.deferred = true;
    useProjectStore.setState({ projectEpoch: 4, timelineVersion: 7, timeline: rustTimeline() });
    useEditorUiStore.setState({
      activeFrame: 0,
      currentFrame: 0,
      isPlaying: true,
      isScrubbing: false,
      rustEngineFailed: false,
    });

    const root = await mountPlaybackHook();

    expect(nativeApiHarness.onPlaybackFrame).toHaveBeenCalledTimes(1);
    expect(nativeApiHarness.playbackStart).not.toHaveBeenCalled();

    await act(async () => {
      nativeApiHarness.listenerAttempts[0]?.reject();
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(useEditorUiStore.getState().isPlaying).toBe(false);

    await act(async () => {
      useEditorUiStore.getState().setPlaying(true);
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(nativeApiHarness.onPlaybackFrame).toHaveBeenCalledTimes(2);
    expect(nativeApiHarness.playbackStart).not.toHaveBeenCalled();

    await act(async () => {
      nativeApiHarness.listenerAttempts[1]?.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(nativeApiHarness.activeListeners).toBe(1);
    expect(nativeApiHarness.playbackStart).toHaveBeenCalledTimes(1);
    expect(nativeApiHarness.order).toEqual([
      "listener-rejected",
      "listener-ready",
      "start",
    ]);

    await unmountPlaybackHook(root);
    expect(nativeApiHarness.activeListeners).toBe(0);
    expect(nativeApiHarness.unlistenCalls).toBe(1);
  });

  it("keeps one pending registration across a rapid remount and replaces it after rejection", async () => {
    nativeApiHarness.deferred = true;
    useProjectStore.setState({ projectEpoch: 4, timelineVersion: 7, timeline: timeline([]) });
    useEditorUiStore.setState({
      activeFrame: 0,
      currentFrame: 0,
      isPlaying: false,
      isScrubbing: false,
      rustEngineFailed: false,
    });

    const firstRoot = await mountPlaybackHook();
    expect(nativeApiHarness.onPlaybackFrame).toHaveBeenCalledTimes(1);
    await unmountPlaybackHook(firstRoot);

    const secondRoot = await mountPlaybackHook();
    expect(nativeApiHarness.onPlaybackFrame).toHaveBeenCalledTimes(1);
    await act(async () => {
      nativeApiHarness.listenerAttempts[0]?.reject();
      await Promise.resolve();
      await Promise.resolve();
    });
    await unmountPlaybackHook(secondRoot);

    const thirdRoot = await mountPlaybackHook();
    expect(nativeApiHarness.onPlaybackFrame).toHaveBeenCalledTimes(2);
    await act(async () => {
      nativeApiHarness.listenerAttempts[1]?.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(nativeApiHarness.activeListeners).toBe(1);
    expect(nativeApiHarness.unlistenCalls).toBe(0);
    await unmountPlaybackHook(thirdRoot);
    expect(nativeApiHarness.activeListeners).toBe(0);
    expect(nativeApiHarness.unlistenCalls).toBe(1);
  });

  it("starts native playback for a temporal compositor timeline once the capability route resolves", async () => {
    useProjectStore.setState({
      projectEpoch: 4,
      timelineVersion: 7,
      timeline: temporalCompositorTimeline(),
    });
    useEditorUiStore.setState({
      activeFrame: 0,
      currentFrame: 0,
      isPlaying: true,
      isScrubbing: false,
      rustEngineFailed: false,
    });

    const root = await mountPlaybackHook();

    expect(useEditorUiStore.getState().isPlaying).toBe(true);
    expect(nativeApiHarness.playbackStart).toHaveBeenCalledTimes(1);
    await unmountPlaybackHook(root);
  });

  it("stops transport instead of starting native playback when the temporal compositor capability is unavailable", async () => {
    nativeApiHarness.getPreviewEndpoint.mockResolvedValue(null);
    useProjectStore.setState({
      projectEpoch: 4,
      timelineVersion: 7,
      timeline: temporalCompositorTimeline(),
    });
    useEditorUiStore.setState({
      activeFrame: 0,
      currentFrame: 0,
      isPlaying: true,
      isScrubbing: false,
      rustEngineFailed: false,
    });

    const root = await mountPlaybackHook();
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(useEditorUiStore.getState().isPlaying).toBe(false);
    expect(nativeApiHarness.playbackStart).not.toHaveBeenCalled();
    await unmountPlaybackHook(root);
  });

  it("uses WebKit without invoking native playback when the endpoint capability is absent", async () => {
    nativeApiHarness.getPreviewEndpoint.mockResolvedValue(null);
    vi.stubGlobal("requestAnimationFrame", vi.fn(() => 1));
    vi.stubGlobal("cancelAnimationFrame", vi.fn());
    useProjectStore.setState({
      projectEpoch: 4,
      timelineVersion: 7,
      timeline: timeline([
        track({
          id: "v1",
          type: "video",
          clips: [clip({ id: "base", mediaType: "video" })],
        }),
        track({
          id: "v2",
          type: "video",
          clips: [clip({ id: "overlay", mediaType: "video" })],
        }),
      ]),
    });
    useEditorUiStore.setState({
      activeFrame: 0,
      currentFrame: 0,
      isPlaying: true,
      isScrubbing: false,
      rustEngineFailed: false,
    });

    const root = await mountPlaybackHook();
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(nativeApiHarness.playbackStart).not.toHaveBeenCalled();
    expect(useEditorUiStore.getState().isPlaying).toBe(true);
    expect(requestAnimationFrame).toHaveBeenCalled();
    await unmountPlaybackHook(root);
  });

  it("does not seek on the play-to-pause edge", () => {
    expect(
      shouldSyncPausedMediaToFrame({
        isPlaying: false,
        isScrubbing: false,
        wasPlaying: true,
        wasScrubbing: false,
      }),
    ).toBe(false);
  });

  it("syncs DOM media only after transport is already settled paused", () => {
    expect(
      shouldSyncPausedMediaToFrame({
        isPlaying: false,
        isScrubbing: false,
        wasPlaying: false,
        wasScrubbing: false,
      }),
    ).toBe(true);
    expect(
      shouldSyncPausedMediaToFrame({
        isPlaying: true,
        isScrubbing: false,
        wasPlaying: false,
        wasScrubbing: false,
      }),
    ).toBe(false);
    expect(
      shouldSyncPausedMediaToFrame({
        isPlaying: false,
        isScrubbing: true,
        wasPlaying: false,
        wasScrubbing: false,
      }),
    ).toBe(false);
  });
});

describe("handleNativePlaybackError", () => {
  const failure = {
    projectEpoch: 1,
    timelineVersion: 2,
    sessionId: "session-1",
    frame: 30,
    code: "videoDecode" as const,
    message: "clip-1 decode failed",
    fatal: true,
  };

  it("stops the transport and tells the user when the engine paused on a fatal error", () => {
    useEditorUiStore.setState({ isPlaying: true, toast: null });
    previewEngine.handleNativePlaybackError(failure, { acceptError: () => true });

    const ui = useEditorUiStore.getState();
    expect(ui.isPlaying).toBe(false);
    expect(ui.toast?.message).toContain("clip-1 decode failed");
  });

  it("keeps playing but reports a non-fatal audio error", () => {
    useEditorUiStore.setState({ isPlaying: true, toast: null });
    previewEngine.handleNativePlaybackError(
      { ...failure, code: "audioDecode", fatal: false, message: "audio at 2.0 s" },
      { acceptError: () => true },
    );

    const ui = useEditorUiStore.getState();
    expect(ui.isPlaying).toBe(true);
    expect(ui.toast?.message).toContain("audio at 2.0 s");
  });

  it("ignores errors of another or paused session", () => {
    useEditorUiStore.setState({ isPlaying: true, toast: null });
    previewEngine.handleNativePlaybackError(failure, { acceptError: () => false });

    expect(useEditorUiStore.getState().isPlaying).toBe(true);
    expect(useEditorUiStore.getState().toast).toBeNull();
  });
});

describe("pausedSeekToleranceSec", () => {
  it("allows sub-frame pause differences without forcing a seek", () => {
    expect(pausedSeekToleranceSec(30)).toBeGreaterThan(0.5 / 30);
    expect(pausedSeekToleranceSec(0)).toBe(pausedSeekToleranceSec(30));
  });

  it("scales by speed because frame rounding expands source-time tolerance", () => {
    const fastTolerance = (pausedSeekToleranceSec as (fps: number, speed?: number) => number)(30, 2);
    expect(fastTolerance).toBeGreaterThan(1 / 30);
  });
});

describe("pausedPlayheadFrameFromFrozenVideo", () => {
  it("maps a frozen video element clock back to the timeline frame", () => {
    const fn = (
      previewEngine as {
        pausedPlayheadFrameFromFrozenVideo?: (
          media: ActiveMedia | null,
          currentTimeSec: number,
          fps: number,
        ) => number | null;
      }
    ).pausedPlayheadFrameFromFrozenVideo;
    const media = {
      trackIndex: 0,
      track: { id: "v1" },
      clip: {
        id: "clip-1",
        startFrame: 100,
        trimStartFrame: 30,
        speed: 2,
        mediaRef: "camera-a",
        mediaType: "video",
      },
    } as ActiveMedia;

    expect(typeof fn).toBe("function");
    expect(fn?.(media, 2, 30)).toBe(115);
  });

  it("never rewinds the authoritative playhead to a lagging decoded frame", () => {
    const fn = (
      previewEngine as {
        settlePausedPlayheadFrame?: (activeFrame: number, frozenFrame: number | null) => number;
      }
    ).settlePausedPlayheadFrame;

    expect(typeof fn).toBe("function");
    expect(fn?.(120, 114)).toBe(120);
    expect(fn?.(120, 123)).toBe(123);
    expect(fn?.(120, null)).toBe(120);
  });
});

describe("transportAcceptsNativePlayhead", () => {
  it("rejects native ticks immediately after pause and while scrubbing", () => {
    const fn = (
      previewEngine as {
        transportAcceptsNativePlayhead?: (isPlaying: boolean, isScrubbing: boolean) => boolean;
      }
    ).transportAcceptsNativePlayhead;

    expect(fn?.(true, false)).toBe(true);
    expect(fn?.(false, false)).toBe(false);
    expect(fn?.(true, true)).toBe(false);
  });
});

describe("activeVideoForPausedSnap", () => {
  it("uses the topmost active video even when an image layer is above it", () => {
    const fn = (
      previewEngine as {
        activeVideoForPausedSnap?: (timeline: Timeline, frame: number) => ActiveMedia | null;
      }
    ).activeVideoForPausedSnap;
    const tl = timeline([
      track({ id: "v1", type: "video", clips: [clip({ id: "video", mediaType: "video" })] }),
      track({ id: "v2", type: "video", clips: [clip({ id: "image", mediaType: "image" })] }),
    ]);

    expect(typeof fn).toBe("function");
    expect(fn?.(tl, 10)?.clip.id).toBe("video");
  });
});

describe("shouldSeekPlayingFollower", () => {
  it("forces a seek when a reused playback key switches clip identity", () => {
    const fn = (
      previewEngine as {
        shouldSeekPlayingFollower?: (args: {
          previousClipId: string | null;
          currentClipId: string;
          currentTimeSec: number;
          desiredTimeSec: number;
        }) => boolean;
      }
    ).shouldSeekPlayingFollower;

    expect(typeof fn).toBe("function");
    expect(
      fn?.({
        previousClipId: "left-half",
        currentClipId: "right-half",
        currentTimeSec: 1.05,
        desiredTimeSec: 1.1,
      }),
    ).toBe(true);
  });
});

describe("WebKit playback transport", () => {
  it("drives a registered media element through play and pause from the owning clock", async () => {
    const tl = timeline([
      track({
        id: "v1",
        type: "video",
        clips: [clip({ id: "clip-1", mediaRef: "asset", mediaType: "video" })],
      }),
    ]);
    const active = {
      trackIndex: 0,
      track: tl.tracks[0],
      clip: tl.tracks[0].clips[0],
    } as ActiveMedia;
    const key = previewElementKey(active);
    let paused = true;
    const play = vi.fn(async () => {
      paused = false;
    });
    const pause = vi.fn(() => {
      paused = true;
    });
    const element = {
      currentTime: 0,
      muted: false,
      volume: 1,
      isConnected: false,
      get paused() {
        return paused;
      },
      play,
      pause,
      load: vi.fn(),
      removeAttribute: vi.fn(),
    } as unknown as HTMLMediaElement;
    const rafCallbacks = new Map<number, FrameRequestCallback>();
    let nextRafId = 1;
    vi.stubGlobal(
      "requestAnimationFrame",
      vi.fn((callback: FrameRequestCallback) => {
        const id = nextRafId++;
        rafCallbacks.set(id, callback);
        return id;
      }),
    );
    vi.stubGlobal(
      "cancelAnimationFrame",
      vi.fn((id: number) => {
        rafCallbacks.delete(id);
      }),
    );
    vi.stubGlobal("localStorage", {
      getItem: () => "0",
      setItem() {},
      removeItem() {},
    });
    useProjectStore.setState({ timeline: tl, timelineVersion: 1 });
    useEditorUiStore.setState({
      currentFrame: 0,
      activeFrame: 0,
      isPlaying: false,
      isScrubbing: false,
      rustEngineFailed: true,
    });
    previewEngine.previewElements.set(key, element);

    useEditorUiStore.getState().togglePlay();
    const root = await mountPlaybackHook();
    const firstFrame = rafCallbacks.values().next().value as FrameRequestCallback | undefined;
    expect(firstFrame).toBeTypeOf("function");
    await act(async () => {
      firstFrame?.(16);
      await Promise.resolve();
    });
    expect(useEditorUiStore.getState().isPlaying).toBe(true);
    expect(play).toHaveBeenCalled();

    await act(async () => {
      useEditorUiStore.getState().togglePlay();
      await Promise.resolve();
    });
    expect(useEditorUiStore.getState().isPlaying).toBe(false);
    expect(pause).toHaveBeenCalled();
    expect(previewEngine).not.toHaveProperty("setWebKitMediaPlayback");

    await unmountPlaybackHook(root);
    previewEngine.previewElements.remove(key);
  });
});

describe("nested sequence transport", () => {
  const SEQUENCE_ID = "seq-1";

  /** A 300-frame root timeline holding a compound clip whose sequence is 60
   *  frames long. The root and nested clips use different media. */
  function compoundProject(
    root: Timeline = timeline([
      track({
        id: "root-v1",
        type: "video",
        clips: [clip({ id: "root-clip", mediaRef: "root-asset", mediaType: "video", durationFrames: 300 })],
      }),
    ]),
    nested: Timeline = timeline([
      track({
        id: "nested-v1",
        type: "video",
        clips: [clip({ id: "nested-clip", mediaRef: "nested-asset", mediaType: "video", durationFrames: 60 })],
      }),
    ]),
  ): Timeline {
    return { ...root, nestedSequences: [{ id: SEQUENCE_ID, name: "Compound", timeline: nested }] };
  }

  function mediaElement() {
    let paused = true;
    return {
      currentTime: 0,
      muted: false,
      volume: 1,
      isConnected: true,
      get paused() {
        return paused;
      },
      play: vi.fn(async () => {
        paused = false;
      }),
      pause: vi.fn(() => {
        paused = true;
      }),
      load: vi.fn(),
      removeAttribute: vi.fn(),
    };
  }

  function elementKey(tl: Timeline, trackIndex = 0): string {
    const active = { trackIndex, track: tl.tracks[trackIndex], clip: tl.tracks[trackIndex].clips[0] };
    return previewElementKey(active as ActiveMedia);
  }

  function stubAnimationFrames(): Map<number, FrameRequestCallback> {
    const callbacks = new Map<number, FrameRequestCallback>();
    let nextId = 1;
    vi.stubGlobal(
      "requestAnimationFrame",
      vi.fn((callback: FrameRequestCallback) => {
        const id = nextId++;
        callbacks.set(id, callback);
        return id;
      }),
    );
    vi.stubGlobal("cancelAnimationFrame", vi.fn((id: number) => callbacks.delete(id)));
    return callbacks;
  }

  async function runNextFrame(callbacks: Map<number, FrameRequestCallback>, ts: number) {
    const [id, callback] = callbacks.entries().next().value as [number, FrameRequestCallback];
    callbacks.delete(id);
    await act(async () => {
      callback(ts);
      await Promise.resolve();
    });
  }

  async function playInside(project: Timeline, activeFrame: number): Promise<Root> {
    useProjectStore.setState({ projectEpoch: 5, timelineVersion: 3, timeline: project });
    useEditorUiStore.setState({
      activeNestedSequenceId: SEQUENCE_ID,
      activeFrame,
      currentFrame: activeFrame,
      isPlaying: true,
      isScrubbing: false,
      rustEngineFailed: false,
      webkitPlaybackFailedRevision: null,
    });
    const root = await mountPlaybackHook();
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    return root;
  }

  afterEach(() => {
    useEditorUiStore.setState({ activeNestedSequenceId: null, isPlaying: false });
  });

  it("stops at the last frame of the open sequence, not the root timeline", async () => {
    const frames = stubAnimationFrames();
    const root = await playInside(compoundProject(), 58);

    await runNextFrame(frames, 1_000);
    expect(useEditorUiStore.getState().isPlaying).toBe(true);
    await runNextFrame(frames, 1_000 + (2 * 1_000) / 30);

    expect(useEditorUiStore.getState()).toMatchObject({ isPlaying: false, currentFrame: 59 });
    await unmountPlaybackHook(root);
  });

  it("plays and seeks the open sequence's media elements, not the root's", async () => {
    const frames = stubAnimationFrames();
    const project = compoundProject();
    const nestedElement = mediaElement();
    const rootElement = mediaElement();
    const nestedKey = elementKey(project.nestedSequences![0]!.timeline);
    const rootKey = elementKey(project);
    previewEngine.previewElements.set(nestedKey, nestedElement as unknown as HTMLMediaElement);
    previewEngine.previewElements.set(rootKey, rootElement as unknown as HTMLMediaElement);

    const root = await playInside(project, 30);
    await runNextFrame(frames, 1_000);

    expect(nestedElement.play).toHaveBeenCalled();
    expect(nestedElement.currentTime).toBeCloseTo(1, 5);
    expect(rootElement.play).not.toHaveBeenCalled();

    // An external seek while playing repositions the nested element.
    await act(async () => useEditorUiStore.getState().setCurrentFrame(45));
    await runNextFrame(frames, 1_050);
    expect(nestedElement.currentTime).toBeCloseTo(1.5, 5);
    expect(rootElement.currentTime).toBe(0);

    await unmountPlaybackHook(root);
    previewEngine.previewElements.remove(nestedKey);
    previewEngine.previewElements.remove(rootKey);
  });

  it("plays a nested sequence on WebKit even when the root needs the native engine", async () => {
    stubAnimationFrames();

    const root = await playInside(compoundProject(temporalCompositorTimeline()), 0);

    expect(nativeApiHarness.playbackStart).not.toHaveBeenCalled();
    expect(useEditorUiStore.getState().isPlaying).toBe(true);
    expect(requestAnimationFrame).toHaveBeenCalled();
    await unmountPlaybackHook(root);
  });

  it("stops a nested sequence that only the native engine could play", async () => {
    stubAnimationFrames();

    const root = await playInside(compoundProject(undefined, temporalCompositorTimeline()), 0);

    expect(nativeApiHarness.playbackStart).not.toHaveBeenCalled();
    expect(useEditorUiStore.getState().isPlaying).toBe(false);
    await unmountPlaybackHook(root);
  });

  it("resolves the route again when the nested sequence closes", async () => {
    stubAnimationFrames();
    const root = await playInside(compoundProject(temporalCompositorTimeline()), 0);
    expect(nativeApiHarness.playbackStart).not.toHaveBeenCalled();

    await act(async () => {
      useEditorUiStore.setState({ activeNestedSequenceId: null });
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(nativeApiHarness.playbackStart).toHaveBeenCalledTimes(1);
    await unmountPlaybackHook(root);
  });
});

describe("previewElements release", () => {
  // This suite runs without a DOM; the registry only needs the media API.
  function mediaElement(isConnected: boolean) {
    const attributes = new Map([["src", "asset://clip.mov"]]);
    return {
      isConnected,
      pause: vi.fn(),
      load: vi.fn(),
      getAttribute: (name: string) => attributes.get(name) ?? null,
      removeAttribute: (name: string) => {
        attributes.delete(name);
      },
    };
  }

  it("drops a removed element's source so its decoder is freed now", async () => {
    const video = mediaElement(false);
    previewEngine.previewElements.set("release-key", video as unknown as HTMLMediaElement);

    previewEngine.previewElements.remove("release-key");
    await Promise.resolve();

    expect(previewEngine.previewElements.get("release-key")).toBeNull();
    expect(video.pause).toHaveBeenCalled();
    expect(video.getAttribute("src")).toBeNull();
    expect(video.load).toHaveBeenCalledOnce();
  });

  it("leaves an element that is still mounted alone", async () => {
    const video = mediaElement(true);
    previewEngine.previewElements.set("mounted-key", video as unknown as HTMLMediaElement);

    previewEngine.previewElements.remove("mounted-key");
    await Promise.resolve();

    expect(video.getAttribute("src")).toBe("asset://clip.mov");
    expect(video.load).not.toHaveBeenCalled();
  });
});

describe("previewElementKey", () => {
  it("keeps linked video and audio elements separate even when clip ids match", () => {
    const base = {
      trackIndex: 0,
      track: { id: "v1" },
      clip: { id: "id-5", mediaRef: "asset", mediaType: "video" },
    } as ActiveMedia;
    const linkedAudio = {
      trackIndex: 1,
      track: { id: "a1" },
      clip: { id: "id-5", mediaRef: "asset", mediaType: "audio" },
    } as ActiveMedia;

    expect(previewElementKey(base)).not.toBe(previewElementKey(linkedAudio));
  });

  it("reuses the same element for adjacent split clips on the same track and source", () => {
    const left = {
      trackIndex: 0,
      track: { id: "v1" },
      clip: { id: "left-half", mediaRef: "interview", mediaType: "video" },
    } as ActiveMedia;
    const right = {
      trackIndex: 0,
      track: { id: "v1" },
      clip: { id: "right-half", mediaRef: "interview", mediaType: "video" },
    } as ActiveMedia;

    expect(previewElementKey(left)).toBe(previewElementKey(right));
  });

  it("keeps different source media separate on the same track", () => {
    const first = {
      trackIndex: 0,
      track: { id: "v1" },
      clip: { id: "a", mediaRef: "camera-a", mediaType: "video" },
    } as ActiveMedia;
    const second = {
      trackIndex: 0,
      track: { id: "v1" },
      clip: { id: "b", mediaRef: "camera-b", mediaType: "video" },
    } as ActiveMedia;

    expect(previewElementKey(first)).not.toBe(previewElementKey(second));
  });

  it("keeps the same source separate across tracks for picture-in-picture", () => {
    const base = {
      trackIndex: 0,
      track: { id: "v1" },
      clip: { id: "main", mediaRef: "interview", mediaType: "video" },
    } as ActiveMedia;
    const pip = {
      trackIndex: 1,
      track: { id: "v2" },
      clip: { id: "pip", mediaRef: "interview", mediaType: "video" },
    } as ActiveMedia;

    expect(previewElementKey(base)).not.toBe(previewElementKey(pip));
  });
});

describe("playback engine subscription cost", () => {
  async function startNativePlayback(): Promise<{ root: Root; renders: () => number }> {
    useProjectStore.setState({
      projectEpoch: 4,
      timelineVersion: 7,
      timeline: temporalCompositorTimeline(),
    });
    useEditorUiStore.setState({
      activeFrame: 10,
      currentFrame: 10,
      isPlaying: true,
      isScrubbing: false,
      rustEngineFailed: false,
    });
    let count = 0;
    function CountingHost(): null {
      count += 1;
      previewEngine.useTimelinePlaybackEngine();
      return null;
    }
    const root = createRoot(installReactHost());
    await act(async () => {
      root.render(React.createElement(CountingHost));
      await Promise.resolve();
    });
    expect(nativeApiHarness.playbackStart).toHaveBeenCalledTimes(1);
    return { root, renders: () => count };
  }

  it("re-renders its host once per whole frame and resolves the route once per revision", async () => {
    const { root, renders } = await startNativePlayback();
    const resolveRoute = vi.spyOn(playbackRoute, "resolveTimelinePlaybackGate");
    const rendersBefore = renders();

    // 60 fractional playback ticks across two whole-frame boundaries (10 -> 12).
    for (let tick = 1; tick <= 60; tick += 1) {
      await act(async () => useEditorUiStore.getState().setActiveFrame(10 + tick * (2 / 60)));
    }

    expect(renders() - rendersBefore).toBe(2);
    expect(resolveRoute).not.toHaveBeenCalled();

    await act(async () => useProjectStore.setState({ timelineVersion: 8 }));
    const perRevision = resolveRoute.mock.calls.length;
    expect(perRevision).toBeGreaterThan(0);
    expect(perRevision).toBeLessThanOrEqual(2);

    resolveRoute.mockRestore();
    await unmountPlaybackHook(root);
  });

  it("still forwards an external seek to the native engine while playing", async () => {
    const { root } = await startNativePlayback();
    const identity = nativePlaybackController.currentIdentity();
    expect(identity).not.toBeNull();
    await act(async () =>
      publishNativePlaybackFrame({ ...identity!, frame: 12, sequence: 1, terminal: false }),
    );
    expect(useEditorUiStore.getState().activeFrame).toBe(12);
    expect(nativeApiHarness.playbackSeek).not.toHaveBeenCalled();

    await act(async () => useEditorUiStore.getState().setCurrentFrame(90));

    expect(nativeApiHarness.playbackSeek).toHaveBeenCalledWith(identity, 90);
    await unmountPlaybackHook(root);
  });
});
