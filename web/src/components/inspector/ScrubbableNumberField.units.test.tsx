// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ScrubbableNumberField } from "./ScrubbableNumberField";
import { decibelUnits, percentUnits } from "./inspectorUnits";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
  .IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;
let onCommit: ReturnType<typeof vi.fn>;

beforeEach(() => {
  onCommit = vi.fn();
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
});

async function typeAndCommit(
  props: {
    value: number;
    min: number;
    max: number;
    format: (v: number) => string;
    parse?: (text: string) => number | null;
    suffix?: string;
  },
  text: string,
): Promise<void> {
  await act(async () =>
    root.render(
      <ScrubbableNumberField ariaLabel="Field" sensitivity={0.01} onCommit={onCommit} {...props} />,
    ),
  );
  const display = container.querySelector<HTMLElement>("[role='spinbutton']")!;
  await act(async () => {
    display.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
  });
  const input = container.querySelector<HTMLInputElement>("input[aria-label='Field']")!;
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set?.call(input, text);
    input.dispatchEvent(new InputEvent("input", { bubbles: true }));
  });
  await act(async () => {
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
  });
}

describe("typed values are parsed in display units", () => {
  it("opens the editor with the displayed percent", async () => {
    await act(async () =>
      root.render(
        <ScrubbableNumberField
          ariaLabel="Field"
          value={0.25}
          min={0}
          max={1}
          sensitivity={0.005}
          format={percentUnits.format}
          parse={percentUnits.parse}
          suffix="%"
          onCommit={onCommit}
        />,
      ),
    );
    const display = container.querySelector<HTMLElement>("[role='spinbutton']")!;
    await act(async () => {
      display.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
    });
    expect(container.querySelector<HTMLInputElement>("input")!.value).toBe("25");
  });

  it("commits 50% opacity as 0.5", async () => {
    await typeAndCommit(
      { value: 1, min: 0, max: 1, format: percentUnits.format, parse: percentUnits.parse, suffix: "%" },
      "50",
    );
    expect(onCommit).toHaveBeenCalledExactlyOnceWith(0.5);
  });

  it("commits 150% scale as 1.5 instead of clamping to the 1000% ceiling", async () => {
    await typeAndCommit(
      { value: 1, min: 0.01, max: 10, format: percentUnits.format, parse: percentUnits.parse, suffix: "%" },
      "150%",
    );
    expect(onCommit).toHaveBeenCalledExactlyOnceWith(1.5);
  });

  it("clamps after converting back to raw units", async () => {
    await typeAndCommit(
      { value: 1, min: 0, max: 1, format: percentUnits.format, parse: percentUnits.parse, suffix: "%" },
      "150",
    );
    expect(onCommit).toHaveBeenCalledExactlyOnceWith(1);
  });

  it("commits -6 dB as linear gain ~0.501", async () => {
    await typeAndCommit(
      { value: 1, min: 0, max: 4, format: decibelUnits.format, parse: decibelUnits.parse, suffix: " dB" },
      "-6",
    );
    expect(onCommit).toHaveBeenCalledOnce();
    expect(onCommit.mock.calls[0][0]).toBeCloseTo(0.501, 3);
  });

  it.each(["-∞", "-inf", "-∞ dB", "-Infinity"])("commits %s as silence", async (text) => {
    await typeAndCommit(
      { value: 1, min: 0, max: 4, format: decibelUnits.format, parse: decibelUnits.parse, suffix: " dB" },
      text,
    );
    expect(onCommit).toHaveBeenCalledExactlyOnceWith(0);
  });

  it("clamps decibels above the linear maximum", async () => {
    await typeAndCommit(
      { value: 1, min: 0, max: 4, format: decibelUnits.format, parse: decibelUnits.parse, suffix: " dB" },
      "30",
    );
    expect(onCommit).toHaveBeenCalledExactlyOnceWith(4);
  });

  it("ignores text that is not a number", async () => {
    await typeAndCommit(
      { value: 1, min: 0, max: 1, format: percentUnits.format, parse: percentUnits.parse, suffix: "%" },
      "abc",
    );
    await typeAndCommit(
      { value: 1, min: 0, max: 1, format: percentUnits.format, parse: percentUnits.parse, suffix: "%" },
      "",
    );
    expect(onCommit).not.toHaveBeenCalled();
  });

  it("keeps raw-unit fields without a parser unchanged", async () => {
    await typeAndCommit({ value: 0, min: -180, max: 180, format: (v) => v.toFixed(1), suffix: "°" }, "45°");
    expect(onCommit).toHaveBeenCalledExactlyOnceWith(45);
  });
});

describe("inspector units round-trip", () => {
  it.each([0.01, 0.5, 1, 1.5, 10])("percent %s", (v) => {
    expect(percentUnits.parse(percentUnits.format(v))).toBeCloseTo(v, 6);
  });

  it.each([0.25, 0.5, 1, 2, 4])("decibels %s", (v) => {
    expect(decibelUnits.parse(decibelUnits.format(v))).toBeCloseTo(v, 1);
  });
});
