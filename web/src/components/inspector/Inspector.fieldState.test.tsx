// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Clip, Mask, Timeline } from "../../lib/types";
import { useEditorUiStore } from "../../store/uiStore";
import { useMediaStore } from "../../store/mediaStore";
import { useProjectStore } from "../../store/projectStore";
import * as edit from "../../store/editActions";
import { t } from "../../i18n";
import { Inspector } from "./Inspector";
import { ScrubbableNumberField } from "./ScrubbableNumberField";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT =
  true;

function clip(id: string, startFrame: number, overrides: Partial<Clip> = {}): Clip {
  return {
    id,
    mediaRef: `media-${id}`,
    mediaType: "video",
    sourceClipType: "video",
    startFrame,
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
    ...overrides,
  };
}

function timelineWith(...clips: Clip[]): Timeline {
  return {
    fps: 30,
    width: 1920,
    height: 1080,
    settingsConfigured: true,
    tracks: [
      {
        id: "video-1",
        name: "Video 1",
        type: "video",
        muted: false,
        hidden: false,
        syncLocked: false,
        clips,
      },
    ],
  };
}

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  vi.spyOn(HTMLElement.prototype, "setPointerCapture").mockImplementation(() => {});
  vi.spyOn(HTMLElement.prototype, "releasePointerCapture").mockImplementation(() => {});
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  vi.restoreAllMocks();
  useEditorUiStore.setState({ selectedClipIds: new Set(), inspectorTab: "video" });
  useMediaStore.setState({ items: [], folders: [], importing: false, error: null });
  useProjectStore.getState().clearProjectSnapshot();
});

function field(label: string): HTMLElement {
  const el = container.querySelector<HTMLElement>(`[role="spinbutton"][aria-label="${label}"]`);
  if (!el) throw new Error(`missing field ${label}`);
  return el;
}

async function drag(label: string, dx: number): Promise<void> {
  const el = field(label);
  const pointer = (type: string, clientX: number) =>
    el.dispatchEvent(new PointerEvent(type, { bubbles: true, pointerId: 1, clientX }));
  await act(async () => pointer("pointerdown", 100));
  await act(async () => pointer("pointermove", 100 + dx));
  await act(async () => pointer("pointerup", 100 + dx));
}

async function typeInto(input: HTMLInputElement | HTMLTextAreaElement, value: string): Promise<void> {
  const proto = Object.getPrototypeOf(input) as object;
  await act(async () => {
    Object.getOwnPropertyDescriptor(proto, "value")!.set!.call(input, value);
    input.dispatchEvent(new InputEvent("input", { bubbles: true }));
  });
}

describe("ScrubbableNumberField latest callbacks", () => {
  it("drags through the onChange of the latest render", async () => {
    const first = vi.fn();
    const second = vi.fn();
    const render = (onChange: (v: number) => void) =>
      root.render(
        <ScrubbableNumberField
          ariaLabel="Value"
          value={0}
          min={-10}
          max={10}
          sensitivity={0.1}
          format={(v) => v.toFixed(1)}
          onChange={onChange}
          onCommit={() => {}}
        />,
      );
    await act(async () => render(first));
    await act(async () => render(second));
    await drag("Value", 20);
    expect(first).not.toHaveBeenCalled();
    expect(second).toHaveBeenCalledWith(2);
  });
});

describe("mask fields keep earlier edits", () => {
  async function renderMask(kind: Mask["shape"]["kind"]) {
    const a = clip("clip-a", 0);
    useProjectStore.setState({ timeline: timelineWith(a), projectPath: "/tmp/demo.opentake" });
    useEditorUiStore.setState({ selectedClipIds: new Set([a.id]), inspectorTab: "video" });
    const setMasks = vi.spyOn(edit, "setMasks").mockResolvedValue();
    await act(async () => root.render(<Inspector />));
    const section = [...container.querySelectorAll("section")].find((s) =>
      s.textContent?.includes(t("inspector.section.mask")),
    )!;
    await act(async () => section.querySelector<HTMLInputElement>('input[type="checkbox"]')!.click());
    if (kind !== "circle") {
      const select = section.querySelector<HTMLSelectElement>("select")!;
      select.value = kind;
      await act(async () => select.dispatchEvent(new Event("change", { bubbles: true })));
    }
    return setMasks;
  }
  const lastMask = (setMasks: ReturnType<typeof vi.spyOn>): Mask =>
    (setMasks.mock.calls.at(-1)![1] as Mask[])[0];

  it("circle: dragging Radius X keeps the committed Center X", async () => {
    const setMasks = await renderMask("circle");
    await drag(t("inspector.field.centerX"), 40);
    await drag(t("inspector.field.radiusX"), -100);
    const mask = lastMask(setMasks);
    if (mask.shape.kind !== "circle") throw new Error("expected circle");
    expect(mask.shape.center.x).toBeCloseTo(0.7);
    expect(mask.shape.radius.x).toBeCloseTo(1);
  });

  it("linear: dragging Normal X keeps the committed Point X", async () => {
    const setMasks = await renderMask("linear");
    await drag(t("inspector.field.pointX"), 40);
    await drag(t("inspector.field.normalX"), -100);
    const mask = lastMask(setMasks);
    if (mask.shape.kind !== "linear") throw new Error("expected linear");
    expect(mask.shape.point.x).toBeCloseTo(0.7);
    expect(mask.shape.normal.x).toBeCloseTo(0.5);
  });

  it("poly: dragging P2 X keeps the committed P1 X", async () => {
    const setMasks = await renderMask("poly");
    await drag("P1 X", 40);
    await drag("P2 X", -40);
    const mask = lastMask(setMasks);
    if (mask.shape.kind !== "poly") throw new Error("expected poly");
    expect(mask.shape.points[0].x).toBeCloseTo(0.45);
    expect(mask.shape.points[1].x).toBeCloseTo(0.55);
  });

  it("transform: dragging Scale X keeps the committed Offset X", async () => {
    const setMasks = await renderMask("circle");
    await drag(t("inspector.mask.offsetX"), 40);
    await drag(t("inspector.mask.scaleX"), 40);
    const mask = lastMask(setMasks);
    expect(mask.transform?.offset.x).toBeCloseTo(0.2);
    expect(mask.transform?.scale.x).toBeCloseTo(1.2);
  });
});

describe("editing across a selection switch", () => {
  it("commits a numeric draft to the clip it was typed for", async () => {
    const a = clip("clip-a", 0);
    const b = clip("clip-b", 90);
    const properties = vi.spyOn(edit, "setClipProperties").mockResolvedValue(undefined);
    useProjectStore.setState({ timeline: timelineWith(a, b) });
    useEditorUiStore.setState({ selectedClipIds: new Set([a.id]), inspectorTab: "video" });
    await act(async () => root.render(<Inspector />));

    const label = t("inspector.field.opacity");
    await act(async () =>
      field(label).dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true })),
    );
    const input = container.querySelector<HTMLInputElement>(`input[aria-label="${label}"]`)!;
    await typeInto(input, "50");
    await act(async () => useEditorUiStore.setState({ selectedClipIds: new Set([b.id]) }));
    await act(async () => input.dispatchEvent(new FocusEvent("focusout", { bubbles: true })));

    expect(properties).toHaveBeenCalledExactlyOnceWith([a.id], { opacity: 0.5 });
  });

  it("saves TextTab edits to the clip they were typed for", async () => {
    const a = clip("text-a", 0, { mediaType: "text", sourceClipType: "text", textContent: "Alpha" });
    const b = clip("text-b", 90, { mediaType: "text", sourceClipType: "text", textContent: "Beta" });
    const properties = vi.spyOn(edit, "setClipProperties").mockResolvedValue(undefined);
    useProjectStore.setState({ timeline: timelineWith(a, b) });
    useEditorUiStore.setState({ selectedClipIds: new Set([a.id]), inspectorTab: "text" });
    await act(async () => root.render(<Inspector />));

    const textarea = container.querySelector<HTMLTextAreaElement>(
      `textarea[aria-label="${t("inspector.section.text")}"]`,
    )!;
    await typeInto(textarea, "Alpha edited");
    await act(async () => useEditorUiStore.setState({ selectedClipIds: new Set([b.id]) }));
    await act(async () => textarea.dispatchEvent(new FocusEvent("focusout", { bubbles: true })));

    expect(properties).toHaveBeenCalledExactlyOnceWith([a.id], { textContent: "Alpha edited" });
    expect(
      container.querySelector<HTMLTextAreaElement>(`textarea[aria-label="${t("inspector.section.text")}"]`)!
        .value,
    ).toBe("Beta");
  });

  it("still commits a TextTab edit on blur without a switch", async () => {
    const a = clip("text-a", 0, { mediaType: "text", sourceClipType: "text", textContent: "Alpha" });
    const properties = vi.spyOn(edit, "setClipProperties").mockResolvedValue(undefined);
    useProjectStore.setState({ timeline: timelineWith(a) });
    useEditorUiStore.setState({ selectedClipIds: new Set([a.id]), inspectorTab: "text" });
    await act(async () => root.render(<Inspector />));

    const textarea = container.querySelector<HTMLTextAreaElement>("textarea")!;
    await typeInto(textarea, "Alpha edited");
    await act(async () => textarea.dispatchEvent(new FocusEvent("focusout", { bubbles: true })));
    await act(async () => root.unmount());
    root = createRoot(container);

    expect(properties).toHaveBeenCalledExactlyOnceWith([a.id], { textContent: "Alpha edited" });
  });
});
