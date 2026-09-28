// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { LibraryEntry } from "../../lib/libraryApi";

vi.mock("../../lib/asset", () => ({
  assetUrl: (path: string | null | undefined) => (path ? `asset://${path}` : null),
}));

import { LibraryEntryCard } from "./LibraryView";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT =
  true;

const observers: Array<(entries: Array<Partial<IntersectionObserverEntry>>) => void> = [];
let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  observers.length = 0;
  vi.stubGlobal(
    "IntersectionObserver",
    class {
      constructor(callback: (entries: Array<Partial<IntersectionObserverEntry>>) => void) {
        observers.push(callback);
      }
      observe() {}
      disconnect() {}
    },
  );
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const entry: LibraryEntry = {
  id: "clip-hash",
  type: "video",
  category: null,
  favoritedAt: 1,
  source: "/offline/original.mov",
  storedPath: "/global/library/clip-hash.mov",
  thumb: "blob:https://example.test/clip",
};

it("unmounts and releases a card's video once it scrolls out of view", async () => {
  await act(async () => root.render(<LibraryEntryCard entry={entry} />));
  const notify = (isIntersecting: boolean) =>
    act(async () => observers.forEach((callback) => callback([{ isIntersecting }])));

  expect(container.querySelector("video")).toBeNull();
  await notify(true);
  const video = container.querySelector("video")!;
  expect(video).not.toBeNull();
  const load = vi.spyOn(video, "load").mockImplementation(() => {});

  await notify(false);
  await act(async () => Promise.resolve());
  expect(container.querySelector("video")).toBeNull();
  expect(video.getAttribute("src")).toBeNull();
  expect(load).toHaveBeenCalledOnce();

  await notify(true);
  expect(container.querySelector("video")).not.toBeNull();

  // Several crossings queued in one callback: the newest entry wins.
  await act(async () =>
    observers.forEach((callback) => callback([{ isIntersecting: true }, { isIntersecting: false }])),
  );
  expect(container.querySelector("video")).toBeNull();
});

it("keeps an image thumbnail once it has been seen", async () => {
  await act(async () => root.render(<LibraryEntryCard entry={{ ...entry, type: "image" }} />));
  const notify = (isIntersecting: boolean) =>
    act(async () => observers.forEach((callback) => callback([{ isIntersecting }])));

  expect(container.querySelector("img")).toBeNull();
  await notify(true);
  const image = container.querySelector("img");
  expect(image).not.toBeNull();
  await notify(false);
  expect(container.querySelector("img")).toBe(image);
});
