// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useEditorUiStore } from "../../store/uiStore";
import * as edit from "../../store/editActions";
import { ScrubbableNumberField } from "./ScrubbableNumberField";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
  .IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;
let onChange: ReturnType<typeof vi.fn>;
let onCommit: ReturnType<typeof vi.fn>;

function renderField(disabled = false): void {
  root.render(
    <ScrubbableNumberField
      ariaLabel="Opacity"
      value={5}
      min={0}
      max={10}
      sensitivity={1}
      format={(value) => value.toFixed(1)}
      suffix="%"
      disabled={disabled}
      onChange={onChange}
      onCommit={onCommit}
    />,
  );
}

async function enterTextMode(): Promise<HTMLInputElement> {
  const display = container.querySelector<HTMLElement>("[role='spinbutton']")!;
  await act(async () => {
    display.dispatchEvent(new PointerEvent("pointerdown", {
      bubbles: true,
      pointerId: 1,
      clientX: 10,
    }));
    display.dispatchEvent(new PointerEvent("pointerup", {
      bubbles: true,
      pointerId: 1,
      clientX: 10,
    }));
  });
  return container.querySelector<HTMLInputElement>("input[aria-label='Opacity']")!;
}

async function setInput(input: HTMLInputElement, value: string): Promise<void> {
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set?.call(input, value);
    input.dispatchEvent(new InputEvent("input", { bubbles: true }));
  });
}

beforeEach(() => {
  onChange = vi.fn();
  onCommit = vi.fn();
  vi.spyOn(HTMLElement.prototype, "setPointerCapture").mockImplementation(() => {});
  vi.spyOn(HTMLElement.prototype, "releasePointerCapture").mockImplementation(() => {});
  vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
    callback(0);
    return 1;
  });
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  act(() => renderField());
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

it("control-481c7d66573516a6 numeric text-entry mode", async () => {
  let input = await enterTextMode();
  expect(document.activeElement).toBe(input);
  await setInput(input, "12,5%");
  await act(async () => {
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
  });
  expect(onCommit).toHaveBeenCalledTimes(1);
  expect(onCommit).toHaveBeenCalledWith(10);
  expect(container.querySelector("input")).toBeNull();
  expect(document.activeElement).toBe(container.querySelector("[role='spinbutton']"));

  onCommit.mockClear();
  input = await enterTextMode();
  await setInput(input, "not-a-number");
  await act(async () => input.dispatchEvent(new FocusEvent("focusout", { bubbles: true })));
  expect(onCommit).not.toHaveBeenCalled();
  expect(container.querySelector("input")).toBeNull();

  input = await enterTextMode();
  await setInput(input, "7");
  await act(async () => {
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  });
  expect(onCommit).not.toHaveBeenCalled();
  expect(document.activeElement).toBe(container.querySelector("[role='spinbutton']"));
});

it("control-3e4fc80f4dde046e pointer-scrubbable numeric value", async () => {
  const display = container.querySelector<HTMLElement>("[role='spinbutton']")!;
  await act(async () => {
    display.dispatchEvent(new PointerEvent("pointerdown", {
      bubbles: true,
      pointerId: 2,
      clientX: 10,
    }));
    display.dispatchEvent(new PointerEvent("pointermove", {
      bubbles: true,
      pointerId: 2,
      clientX: 14,
    }));
    display.dispatchEvent(new PointerEvent("pointerup", {
      bubbles: true,
      pointerId: 2,
      clientX: 14,
    }));
  });
  expect(onChange).toHaveBeenCalledWith(9);
  expect(onCommit).toHaveBeenCalledTimes(1);
  expect(onCommit).toHaveBeenCalledWith(9);

  onChange.mockClear();
  onCommit.mockClear();
  await act(async () => {
    display.dispatchEvent(new PointerEvent("pointerdown", {
      bubbles: true,
      pointerId: 3,
      clientX: 10,
    }));
    display.dispatchEvent(new PointerEvent("pointermove", {
      bubbles: true,
      pointerId: 3,
      clientX: 14,
      shiftKey: true,
    }));
    display.dispatchEvent(new PointerEvent("pointercancel", {
      bubbles: true,
      pointerId: 3,
    }));
    display.dispatchEvent(new PointerEvent("pointerup", {
      bubbles: true,
      pointerId: 3,
      clientX: 14,
    }));
  });
  expect(onChange).toHaveBeenCalledWith(10);
  expect(onCommit).not.toHaveBeenCalled();
  expect(container.querySelector("input")).toBeNull();

  onChange.mockClear();
  await act(async () => {
    display.dispatchEvent(new PointerEvent("pointerdown", {
      bubbles: true,
      pointerId: 4,
      clientX: 10,
    }));
    display.dispatchEvent(new PointerEvent("pointermove", {
      bubbles: true,
      pointerId: 4,
      clientX: 20,
      metaKey: true,
    }));
    display.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  });
  expect(onChange).toHaveBeenCalledWith(6);
  expect(onCommit).not.toHaveBeenCalled();
  expect(document.activeElement).toBe(display);
});

it("cancels text and pointer commits when the field becomes disabled", async () => {
  const input = await enterTextMode();
  await setInput(input, "7");
  await act(async () => renderField(true));
  expect(container.querySelector("input")).toBeNull();
  expect(container.querySelector("[role='spinbutton']")?.getAttribute("aria-disabled")).toBe("true");
  expect(onCommit).not.toHaveBeenCalled();

  await act(async () => renderField(false));
  const display = container.querySelector<HTMLElement>("[role='spinbutton']")!;
  await act(async () => {
    display.dispatchEvent(new PointerEvent("pointerdown", {
      bubbles: true,
      pointerId: 9,
      clientX: 10,
    }));
    display.dispatchEvent(new PointerEvent("pointermove", {
      bubbles: true,
      pointerId: 9,
      clientX: 14,
    }));
  });
  expect(onChange).toHaveBeenCalledWith(9);
  await act(async () => renderField(true));
  await act(async () => {
    display.dispatchEvent(new PointerEvent("pointerup", {
      bubbles: true,
      pointerId: 9,
      clientX: 14,
    }));
  });
  expect(onCommit).not.toHaveBeenCalled();
  expect(container.querySelector("input")).toBeNull();
});

it("leaves IME candidate keys in text-entry mode to the input method", async () => {
  const input = await enterTextMode();
  await setInput(input, "８");
  await act(async () => {
    input.dispatchEvent(new Event("compositionstart", { bubbles: true }));
    input.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Escape", isComposing: true, keyCode: 229, bubbles: true }),
    );
    input.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Enter", isComposing: true, keyCode: 229, bubbles: true }),
    );
    input.dispatchEvent(new Event("compositionend", { bubbles: true }));
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", keyCode: 229, bubbles: true }));
  });
  expect(onCommit).not.toHaveBeenCalled();
  expect(container.querySelector("input[aria-label='Opacity']")).toBe(input);

  await setInput(input, "8");
  await act(async () => {
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", keyCode: 13, bubbles: true }));
  });
  expect(onCommit).toHaveBeenCalledExactlyOnceWith(8);
});

describe("arrow-key stepping", () => {
  function renderWide(
    commit: (value: number) => void | Promise<unknown> = onCommit,
    value = 5,
  ): HTMLElement {
    act(() =>
      root.render(
        <ScrubbableNumberField
          ariaLabel="Size"
          value={value}
          min={0}
          max={100}
          sensitivity={0.5}
          format={(v) => v.toFixed(1)}
          onChange={onChange}
          onCommit={commit}
        />,
      ),
    );
    return container.querySelector<HTMLElement>("[role='spinbutton']")!;
  }

  function key(target: HTMLElement, type: "keydown" | "keyup", key: string, repeat = false) {
    act(() => {
      target.dispatchEvent(new KeyboardEvent(type, { key, repeat, bubbles: true, cancelable: true }));
    });
  }

  afterEach(() => {
    useEditorUiStore.setState({ toast: null });
  });

  it("previews a held arrow key and commits the accumulated value once on release", () => {
    const display = renderWide();
    display.focus();
    for (let press = 0; press < 10; press++) key(display, "keydown", "ArrowUp", press > 0);

    expect(onCommit).not.toHaveBeenCalled();
    expect(onChange).toHaveBeenLastCalledWith(10);
    expect(display.textContent).toBe("10.0");

    key(display, "keyup", "ArrowUp");
    expect(onCommit).toHaveBeenCalledExactlyOnceWith(10);
    expect(display.textContent).toBe("5.0");
  });

  it("steps quick taps from the committed value while the mirror lags behind", async () => {
    const commits: Array<{ value: number; resolve: () => void }> = [];
    const commit = vi.fn((value: number) =>
      new Promise<void>((resolve) => {
        commits.push({ value, resolve });
      }),
    );
    const display = renderWide(commit);
    for (let tap = 0; tap < 4; tap++) {
      key(display, "keydown", "ArrowDown");
      key(display, "keyup", "ArrowDown");
    }
    expect(commits.map((entry) => entry.value)).toEqual([4.5, 4, 3.5, 3]);
    expect(display.textContent).toBe("3.0");

    // The edits land and the mirror refreshes: the next tap steps from it.
    await act(async () => commits.forEach((entry) => entry.resolve()));
    renderWide(commit, 3);
    key(display, "keydown", "ArrowUp");
    key(display, "keyup", "ArrowUp");
    expect(commits.at(-1)?.value).toBe(3.5);
  });

  it("commits through the onCommit of the render the key gesture started in", () => {
    const first = vi.fn();
    const second = vi.fn();
    const display = renderWide(first);
    key(display, "keydown", "ArrowUp");
    // e.g. the playhead moved while the key is held: a new target frame.
    renderWide(second);
    key(display, "keydown", "ArrowUp", true);
    key(display, "keyup", "ArrowUp");

    expect(first).toHaveBeenCalledExactlyOnceWith(6);
    expect(second).not.toHaveBeenCalled();
  });

  it("commits a held key on blur, on unmount and before history, and cancels it on Escape", () => {
    const hold = vi.spyOn(edit, "holdGestureCommit");
    let display = renderWide();
    display.focus();
    key(display, "keydown", "ArrowUp");
    act(() => display.blur());
    expect(onCommit).toHaveBeenCalledExactlyOnceWith(5.5);

    onCommit.mockClear();
    onChange.mockClear();
    key(display, "keydown", "ArrowUp");
    key(display, "keydown", "Escape");
    key(display, "keyup", "ArrowUp");
    expect(onCommit).not.toHaveBeenCalled();
    expect(onChange).toHaveBeenLastCalledWith(5);
    expect(display.textContent).toBe("5.0");

    // Undo/redo flush the held gesture first (editActions.holdGestureCommit).
    key(display, "keydown", "ArrowDown");
    act(() => hold.mock.calls.at(-1)![0]());
    expect(onCommit).toHaveBeenCalledExactlyOnceWith(4.5);
    key(display, "keyup", "ArrowDown");
    expect(onCommit).toHaveBeenCalledOnce();

    onCommit.mockClear();
    display = renderWide();
    key(display, "keydown", "ArrowUp");
    key(display, "keydown", "ArrowUp", true);
    act(() => root.unmount());
    expect(onCommit).toHaveBeenCalledExactlyOnceWith(6);
    root = createRoot(container);
  });

  it("reports a rejected commit as an edit-failure toast and shows the mirror again", async () => {
    const stale = Object.assign(new Error("project changed"), { code: "staleProject" });
    const unhandled = vi.fn();
    const onUnhandled = (event: PromiseRejectionEvent) => unhandled(event.reason);
    window.addEventListener("unhandledrejection", onUnhandled);
    try {
      const display = renderWide(() => Promise.reject(stale));
      key(display, "keydown", "ArrowUp");
      key(display, "keyup", "ArrowUp");
      expect(display.textContent).toBe("5.5");
      await act(async () => new Promise((resolve) => setTimeout(resolve, 0)));
      expect(useEditorUiStore.getState().toast?.message).toContain("project changed");
      expect(display.textContent).toBe("5.0");
      expect(unhandled).not.toHaveBeenCalled();
    } finally {
      window.removeEventListener("unhandledrejection", onUnhandled);
    }
  });
});
