// @vitest-environment happy-dom

import { act, useRef, useState } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MediaItem } from "../../lib/types";

vi.mock("../../lib/asset", () => ({
  assetUrl: (path: string | null | undefined) => (path ? `asset://${path}` : null),
}));

vi.mock("../../lib/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../lib/api")>()),
  previewPoster: vi.fn(async () => null),
}));

vi.mock("./previewEngine", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./previewEngine")>()),
  useRustPlaybackCapability: () => ({ checked: true, available: false, endpoint: null }),
}));

import { useEditorUiStore } from "../../store/uiStore";
import { useMediaStore } from "../../store/mediaStore";
import { MediaPreview, Preview } from "./Preview";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT =
  true;

function item(id: string, type: "audio" | "video"): MediaItem {
  return { id, name: id, type, duration: 4, hasAudio: true, favorite: false, path: `/${id}` };
}

let container: HTMLDivElement;
let root: Root;
let pause: ReturnType<typeof vi.spyOn>;
let play: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  pause = vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
  play = vi.spyOn(HTMLMediaElement.prototype, "play").mockImplementation(async () => {});
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  vi.restoreAllMocks();
});

// Wired exactly like Preview: the callbacks are real parent state setters, so
// every play/timeupdate re-renders the parent (and MediaPreview).
let harnessRef: { current: HTMLMediaElement | null } = { current: null };
function Harness({ media }: { media: MediaItem }) {
  const mediaRef = useRef<HTMLMediaElement | null>(null);
  harnessRef = mediaRef;
  const [, setTime] = useState(0);
  const [, setDuration] = useState(0);
  const [, setPlaying] = useState(false);
  return (
    <MediaPreview
      item={media}
      projectEpoch={1}
      mediaRef={mediaRef}
      onTime={setTime}
      onDuration={setDuration}
      onPlayingChange={setPlaying}
    />
  );
}

describe("MediaPreview media element ref", () => {
  it.each(["audio", "video"] as const)(
    "keeps a playing %s element playing across parent re-renders",
    async (type) => {
      await act(async () => root.render(<Harness media={item("a", type)} />));
      const el = container.querySelector(type)!;
      expect(harnessRef.current).toBe(el);

      await act(async () => {
        el.dispatchEvent(new Event("play"));
      });
      for (let i = 1; i <= 3; i += 1) {
        await act(async () => {
          Object.defineProperty(el, "currentTime", { value: i, configurable: true });
          el.dispatchEvent(new Event("timeupdate"));
        });
      }

      expect(pause).not.toHaveBeenCalled();
      expect(harnessRef.current).toBe(el);
    },
  );

  it.each(["audio", "video"] as const)(
    "points the ref at the rendered %s element after switching items",
    async (type) => {
      await act(async () => root.render(<Harness media={item("a", type)} />));
      await act(async () => root.render(<Harness media={item("b", type)} />));

      const el = container.querySelector(type);
      expect(el?.getAttribute("src")).toBe("asset:///b");
      expect(harnessRef.current).toBe(el);
    },
  );
});

describe("Preview media transport", () => {
  it("toggles the current audio element after switching preview items", async () => {
    useMediaStore.setState({ items: [item("a", "audio"), item("b", "audio")] });
    useEditorUiStore.setState({
      previewTabIds: ["a"],
      previewTabHistory: ["a"],
      previewActiveTabId: "media_a",
      previewMediaId: "a",
    });
    await act(async () => root.render(<Preview />));
    const first = container.querySelector("audio")!;
    // Mounting resets the transport (one pause); only playback matters here.
    pause.mockClear();
    await act(async () => {
      first.dispatchEvent(new Event("play"));
    });
    expect(pause).not.toHaveBeenCalled();

    await act(async () =>
      useEditorUiStore.setState({
        previewTabIds: ["b"],
        previewTabHistory: ["b"],
        previewActiveTabId: "media_b",
        previewMediaId: "b",
      }),
    );
    const current = container.querySelector("audio")!;
    expect(current.getAttribute("src")).toBe("asset:///b");
    play.mockClear();
    Object.defineProperty(current, "paused", { value: true, configurable: true });

    await act(async () => useEditorUiStore.getState().requestMediaPreviewToggle());

    expect(play).toHaveBeenCalledTimes(1);
    expect(play.mock.instances[0]).toBe(current);
  });
});
