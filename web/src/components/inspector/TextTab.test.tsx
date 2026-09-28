// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Clip, Timeline } from "../../lib/types";
import { useEditorUiStore } from "../../store/uiStore";
import { useProjectStore } from "../../store/projectStore";
import * as edit from "../../store/editActions";
import { t } from "../../i18n";
import { Inspector } from "./Inspector";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT =
  true;

function textClip(id: string, startFrame: number): Clip {
  return {
    id,
    mediaRef: `media-${id}`,
    mediaType: "text",
    sourceClipType: "text",
    startFrame,
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
    textContent: id,
  };
}

function timelineWith(...clips: Clip[]): Timeline {
  return {
    fps: 30,
    width: 1920,
    height: 1080,
    settingsConfigured: true,
    tracks: [
      { id: "t1", name: "Text", type: "video", muted: false, hidden: false, syncLocked: false, clips },
    ],
  };
}

const a = textClip("text-a", 0);
const b = textClip("text-b", 90);

let container: HTMLDivElement;
let root: Root;
let setClipProperties: ReturnType<typeof vi.spyOn>;

function colorInput(): HTMLInputElement {
  const input = container.querySelector<HTMLInputElement>(
    `input[type="color"][aria-label="${t("inspector.field.textColor")}"]`,
  );
  if (!input) throw new Error("missing text color input");
  return input;
}

function pick(input: HTMLInputElement, hex: string) {
  act(() => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set?.call(input, hex);
    input.dispatchEvent(new Event("input", { bubbles: true }));
    // WebKit also fires `change` for every color the panel reports.
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });
}

function lastTextStyle() {
  return (setClipProperties.mock.calls.at(-1)?.[1] as { textStyle?: { color: unknown; alignment?: string } })
    .textStyle;
}

beforeEach(() => {
  vi.useFakeTimers();
  setClipProperties = vi.spyOn(edit, "setClipProperties").mockResolvedValue(undefined);
  useProjectStore.setState({ timeline: timelineWith(a, b), projectPath: "/tmp/text.opentake" });
  useEditorUiStore.setState({ selectedClipIds: new Set([a.id]), inspectorTab: "text", toast: null });
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  act(() => root.render(<Inspector />));
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
  vi.useRealTimers();
  vi.restoreAllMocks();
  useEditorUiStore.setState({ selectedClipIds: new Set(), inspectorTab: "video", toast: null });
  useProjectStore.getState().clearProjectSnapshot();
});

describe("TextTab color picker", () => {
  it("previews picked colors locally and commits the last one once on blur", () => {
    const input = colorInput();
    pick(input, "#ff0000");
    pick(input, "#00ff00");
    pick(input, "#0000ff");

    expect(setClipProperties).not.toHaveBeenCalled();
    expect(colorInput().value).toBe("#0000ff");

    act(() => input.dispatchEvent(new FocusEvent("focusout", { bubbles: true })));
    expect(setClipProperties).toHaveBeenCalledOnce();
    expect(setClipProperties.mock.calls[0]?.[0]).toEqual([a.id]);
    expect(lastTextStyle()?.color).toEqual({ r: 0, g: 0, b: 1, a: 1 });
  });

  it("commits once the picker has been idle for the upstream debounce interval", () => {
    const input = colorInput();
    pick(input, "#ff0000");
    act(() => vi.advanceTimersByTime(300));
    pick(input, "#808080");
    act(() => vi.advanceTimersByTime(399));
    expect(setClipProperties).not.toHaveBeenCalled();

    act(() => vi.advanceTimersByTime(1));
    expect(setClipProperties).toHaveBeenCalledOnce();
    expect(lastTextStyle()?.color).toEqual({ r: 128 / 255, g: 128 / 255, b: 128 / 255, a: 1 });
  });

  it("lands a pending color on the clip it was picked for when the selection switches", () => {
    pick(colorInput(), "#ff0000");
    act(() => useEditorUiStore.setState({ selectedClipIds: new Set([b.id]) }));

    expect(setClipProperties).toHaveBeenCalledOnce();
    expect(setClipProperties.mock.calls[0]?.[0]).toEqual([a.id]);
    expect(lastTextStyle()?.color).toEqual({ r: 1, g: 0, b: 0, a: 1 });
    act(() => vi.advanceTimersByTime(1_000));
    expect(setClipProperties).toHaveBeenCalledOnce();
  });

  it("folds a pending color into an immediate style edit", () => {
    pick(colorInput(), "#ff0000");
    const alignLeft = container.querySelector<HTMLButtonElement>(
      `button[aria-label="${t("inspector.align.left")}"]`,
    )!;
    act(() => alignLeft.click());
    act(() => vi.advanceTimersByTime(1_000));

    expect(setClipProperties).toHaveBeenCalledOnce();
    expect(lastTextStyle()).toMatchObject({ color: { r: 1, g: 0, b: 0, a: 1 }, alignment: "left" });
  });

  it("keeps a pending color when a refresh resets the local style before a discrete edit", () => {
    pick(colorInput(), "#ff0000");
    // A refresh from an earlier edit replaces the mirrored style mid-pick.
    act(() =>
      useProjectStore.setState({
        timeline: timelineWith({ ...a, textStyle: { fontName: "Helvetica", fontSize: 96 } as never }, b),
      }),
    );
    act(() =>
      container
        .querySelector<HTMLButtonElement>(`button[aria-label="${t("inspector.align.left")}"]`)!
        .click(),
    );

    expect(setClipProperties).toHaveBeenCalledOnce();
    expect(lastTextStyle()).toMatchObject({
      color: { r: 1, g: 0, b: 0, a: 1 },
      alignment: "left",
      fontName: "Helvetica",
    });
  });

  it("lands a pending color before a history command runs", () => {
    const hold = vi.spyOn(edit, "holdGestureCommit");
    pick(colorInput(), "#00ff00");
    expect(hold).toHaveBeenCalledOnce();

    act(() => hold.mock.calls[0]![0]());
    expect(setClipProperties).toHaveBeenCalledOnce();
    expect(lastTextStyle()?.color).toEqual({ r: 0, g: 1, b: 0, a: 1 });
    act(() => vi.advanceTimersByTime(1_000));
    expect(setClipProperties).toHaveBeenCalledOnce();
  });

  it("reports a rejected color commit and restores the committed color", async () => {
    setClipProperties.mockRejectedValue(
      Object.assign(new Error("project changed"), { code: "staleProject" }),
    );
    const input = colorInput();
    pick(input, "#ff0000");
    act(() => input.dispatchEvent(new FocusEvent("focusout", { bubbles: true })));
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(useEditorUiStore.getState().toast?.message).toContain("project changed");
    expect(colorInput().value).toBe("#ffffff");
  });
});
