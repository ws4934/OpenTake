// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

const renders = vi.hoisted(() => ({ editor: 0, titleBar: 0 }));

vi.mock("./store/sync", () => ({ startSync: vi.fn(async () => {}), stopSync: vi.fn() }));
vi.mock("./store/mediaStore", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./store/mediaStore")>()),
  startMediaSync: vi.fn(async () => {}),
  stopMediaSync: vi.fn(),
}));
vi.mock("./store/libraryStore", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./store/libraryStore")>()),
  startLibrarySync: vi.fn(async () => {}),
  stopLibrarySync: vi.fn(),
}));
vi.mock("./lib/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./lib/api")>()),
  onGoHome: vi.fn(async () => () => {}),
}));
vi.mock("./hooks/useKeyboardShortcuts", () => ({ useKeyboardShortcuts: vi.fn() }));
vi.mock("./hooks/useAutosave", () => ({ useAutosave: vi.fn() }));
// The real playback engine stays mounted: it is the per-frame subscriber.
vi.mock("./components/shell/TitleBar", () => ({
  TitleBar: () => {
    renders.titleBar += 1;
    return null;
  },
}));
vi.mock("./components/shell/EditorSplit", () => ({
  EditorSplit: () => {
    renders.editor += 1;
    return null;
  },
}));
vi.mock("./components/shell/ViewMenu", () => ({ ApplicationMenuBridge: () => null }));
vi.mock("./components/shell/ExportDialog", () => ({ ExportDialog: () => null }));
vi.mock("./components/shell/SaveAsProgress", () => ({ SaveAsProgress: () => null }));
vi.mock("./components/shell/ProjectSettingsMismatchDialog", () => ({
  ProjectSettingsMismatchDialog: () => null,
}));
vi.mock("./components/shell/CompatibilityBanner", () => ({ CompatibilityBanner: () => null }));
vi.mock("./components/settings/UpdateDialog", () => ({ UpdateCenter: () => null }));

import App from "./App";
import { useEditorUiStore } from "./store/uiStore";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT =
  true;

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  useEditorUiStore.setState({
    view: "editor",
    settingsOpen: false,
    toast: null,
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
});

it("does not re-render the editor tree when the playhead advances", async () => {
  await act(async () => root.render(<App />));
  expect(renders.editor).toBeGreaterThan(0);
  renders.editor = 0;
  renders.titleBar = 0;

  for (let tick = 1; tick <= 60; tick += 1) {
    await act(async () => useEditorUiStore.getState().setActiveFrame(10 + tick * (2 / 60)));
  }

  expect(renders.editor).toBe(0);
  expect(renders.titleBar).toBe(0);
});
