// @vitest-environment happy-dom

/**
 * The Preview play button and the Space shortcut gate timeline playback
 * separately; for every combination of shown timeline, nested sequence, engine
 * failure, WebKit decode failure and native capability they must agree.
 */
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { Clip, Timeline } from "../../lib/types";

const probe = vi.hoisted(() => ({
  capability: { checked: true, available: true, endpoint: null as string | null },
}));

vi.mock("./previewEngine", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./previewEngine")>()),
  useRustPlaybackCapability: () => probe.capability,
  lastRustPlaybackCapability: () => probe.capability,
}));

vi.mock("./RustFrameBuffer.tsx", () => ({ RustFrameBuffer: () => null }));

import { t } from "../../i18n";
import { useKeyboardShortcuts } from "../../hooks/useKeyboardShortcuts";
import { useEditorUiStore } from "../../store/uiStore";
import { useProjectStore } from "../../store/projectStore";
import { Preview } from "./Preview";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT =
  true;

function clip(id: string, over: Partial<Clip> = {}): Clip {
  return {
    id,
    mediaRef: `${id}-media`,
    mediaType: "video",
    sourceClipType: "video",
    startFrame: 0,
    durationFrames: 90,
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

function timelineOf(...tracks: Clip[][]): Timeline {
  return {
    fps: 30,
    width: 1920,
    height: 1080,
    settingsConfigured: true,
    tracks: tracks.map((clips, index) => ({
      id: `track-${index}-${clips[0]?.id ?? "empty"}`,
      type: clips[0]?.mediaType ?? "video",
      muted: false,
      hidden: false,
      syncLocked: false,
      clips,
    })),
  };
}

/** Timelines that WebKit plays, that need the native compositor, that need the
 *  native video stack, and that nothing can play. */
const CONTENT: Record<string, () => Timeline> = {
  video: () => timelineOf([clip("plain")]),
  title: () => timelineOf([clip("title", { mediaType: "text", sourceClipType: "text" })]),
  videoStack: () => timelineOf([clip("upper")], [clip("lower")]),
  unknownEffect: () =>
    timelineOf([clip("fx", { effects: [{ name: "not-an-effect", params: {}, enabled: true }] })]),
};

const CAPABILITIES = {
  available: { checked: true, available: true, endpoint: "http://127.0.0.1:43123/frame" },
  unavailable: { checked: true, available: false, endpoint: null },
  probing: { checked: false, available: false, endpoint: null },
};

interface GateCase {
  content: keyof typeof CONTENT;
  nested: boolean;
  rustEngineFailed: boolean;
  forceRust: boolean;
  capability: keyof typeof CAPABILITIES;
}

const CASES: GateCase[] = [];
for (const content of Object.keys(CONTENT)) {
  for (const nested of [false, true]) {
    for (const rustEngineFailed of [false, true]) {
      for (const forceRust of [false, true]) {
        for (const capability of Object.keys(CAPABILITIES) as Array<keyof typeof CAPABILITIES>) {
          CASES.push({ content, nested, rustEngineFailed, forceRust, capability });
        }
      }
    }
  }
}

function KeyboardHost(): null {
  useKeyboardShortcuts();
  return null;
}

let container: HTMLDivElement;
let root: Root;

beforeEach(async () => {
  vi.stubGlobal("ResizeObserver", class {
    observe() {}
    disconnect() {}
  });
  useEditorUiStore.setState({
    view: "editor",
    settingsOpen: false,
    exportDialogOpen: false,
    saveAsProgress: null,
    projectSettingsPrompt: null,
    pendingSwapClipId: null,
    previewTabIds: [],
    previewTabHistory: [],
    previewActiveTabId: "timeline",
    previewMediaId: null,
    focusedPanel: "timeline",
    activeFrame: 0,
    currentFrame: 0,
    isPlaying: false,
    isScrubbing: false,
  });
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () =>
    root.render(
      <>
        <Preview />
        <KeyboardHost />
      </>,
    ),
  );
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  vi.unstubAllGlobals();
  useEditorUiStore.setState({
    activeNestedSequenceId: null,
    isPlaying: false,
    rustEngineFailed: false,
    webkitPlaybackFailedRevision: null,
  });
  useProjectStore.getState().clearProjectSnapshot();
});

async function decide(gateCase: GateCase): Promise<{ button: boolean; space: boolean }> {
  const shown = CONTENT[gateCase.content]!();
  // Inside a compound clip the root holds content nothing can play, so a gate
  // that read the root instead of the open sequence would refuse.
  const rootTimeline = gateCase.nested ? CONTENT.unknownEffect!() : shown;
  await act(async () => {
    probe.capability = { ...CAPABILITIES[gateCase.capability] };
    useProjectStore.setState({
      projectEpoch: 2,
      timelineVersion: 9,
      timeline: gateCase.nested
        ? { ...rootTimeline, nestedSequences: [{ id: "seq-1", name: "Compound", timeline: shown }] }
        : rootTimeline,
    });
    useEditorUiStore.setState({
      activeNestedSequenceId: gateCase.nested ? "seq-1" : null,
      rustEngineFailed: gateCase.rustEngineFailed,
      webkitPlaybackFailedRevision: gateCase.forceRust ? "2:9" : null,
      isPlaying: false,
    });
  });
  const playButton = container.querySelector<HTMLButtonElement>(
    `button[title="${t("preview.playPause")}"]`,
  );
  expect(playButton).not.toBeNull();
  const button = !playButton!.disabled;
  await act(async () => {
    window.dispatchEvent(
      new KeyboardEvent("keydown", { key: " ", code: "Space", bubbles: true, cancelable: true }),
    );
  });
  const space = useEditorUiStore.getState().isPlaying;
  await act(async () => useEditorUiStore.setState({ isPlaying: false }));
  return { button, space };
}

it("gates timeline playback identically for the play button and Space", async () => {
  const mismatches: string[] = [];
  let allowed = 0;
  for (const gateCase of CASES) {
    const { button, space } = await decide(gateCase);
    if (button !== space) mismatches.push(`${JSON.stringify(gateCase)}: button ${button}, Space ${space}`);
    if (button) allowed += 1;
  }

  expect(mismatches).toEqual([]);
  // Both outcomes occur, so agreement is not vacuous.
  expect(allowed).toBeGreaterThan(0);
  expect(allowed).toBeLessThan(CASES.length);
});

it("follows the open sequence rather than the root timeline", async () => {
  expect(
    await decide({
      content: "video",
      nested: true,
      rustEngineFailed: false,
      forceRust: false,
      capability: "available",
    }),
  ).toEqual({ button: true, space: true });
  expect(
    await decide({
      content: "title",
      nested: true,
      rustEngineFailed: false,
      forceRust: false,
      capability: "available",
    }),
  ).toEqual({ button: false, space: false });
});
