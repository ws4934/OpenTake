// @vitest-environment happy-dom

import { act, createElement } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it, vi } from "vitest";
import { useEditorUiStore } from "../store/uiStore";
import { useUpdateStore } from "../store/updateStore";
import * as edit from "../store/editActions";
import {
  DOCUMENTED_SHORTCUT_ROWS,
  handleAgentPanelKeyDown,
  handleProjectSaveKeyDown,
  handleTransportSpaceKeyDown,
  resolveDocumentedShortcut,
  shouldHandleTransportSpaceKey,
  useKeyboardShortcuts,
} from "./useKeyboardShortcuts";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT =
  true;

function event(overrides: Partial<KeyboardEvent> = {}): KeyboardEvent {
  return {
    code: "Space",
    metaKey: false,
    ctrlKey: false,
    altKey: false,
    target: null,
    ...overrides,
  } as KeyboardEvent;
}

describe("keyboard transport Space shortcut", () => {
  it.each(["number", "date", "time", "datetime-local", "month", "week"])(
    "leaves editing and transport keys to %s inputs",
    (type) => {
      const input = document.createElement("input");
      input.type = type;
      const context = {
        view: "editor" as const, blocked: false, focusedPanel: "timeline" as const,
        compatibilityReadOnly: false, cropEditingActive: false,
      };
      for (const code of ["Backspace", "Delete", "Space"]) {
        for (const shiftKey of [false, true]) {
          expect(resolveDocumentedShortcut(event({ code, shiftKey, target: input }), context)).toBeNull();
        }
      }
      for (const code of ["KeyA", "KeyC", "KeyV", "KeyX", "KeyZ", "KeyY"]) {
        for (const modifier of [{ metaKey: true }, { ctrlKey: true }]) {
          expect(resolveDocumentedShortcut(event({ code, target: input, ...modifier }), context)).toBeNull();
          expect(resolveDocumentedShortcut(event({ code, shiftKey: true, target: input, ...modifier }), context)).toBeNull();
        }
      }
      expect(shouldHandleTransportSpaceKey(event({ target: input }), "editor")).toBe(false);
    },
  );

  it("handles plain Space in the editor", () => {
    expect(shouldHandleTransportSpaceKey(event(), "editor")).toBe(true);
  });

  it("handles Space from every non-editable editor panel surface", () => {
    for (const panel of ["media", "preview", "inspector", "timeline", "agent"]) {
      const surface = document.createElement("section");
      surface.dataset.panel = panel;
      expect(
        shouldHandleTransportSpaceKey(event({ target: surface }), "editor"),
        panel,
      ).toBe(true);
    }

    const contentEditable = document.createElement("div");
    contentEditable.contentEditable = "true";
    const plaintextEditable = document.createElement("div");
    plaintextEditable.contentEditable = "plaintext-only";
    for (const editable of [
      document.createElement("input"),
      document.createElement("textarea"),
      contentEditable,
      plaintextEditable,
    ]) {
      expect(
        shouldHandleTransportSpaceKey(event({ target: editable }), "editor"),
      ).toBe(false);
    }
  });

  it("does not suppress Space keyup outside the editor", () => {
    expect(shouldHandleTransportSpaceKey(event(), "home")).toBe(false);
  });

  it("does not suppress modified Space keyup", () => {
    expect(shouldHandleTransportSpaceKey(event({ metaKey: true }), "editor")).toBe(false);
  });

  it("claims Space from media tiles and every non-text native control", () => {
    const button = document.createElement("button");
    const buttonLabel = document.createElement("span");
    button.append(buttonLabel);
    const slider = document.createElement("input");
    slider.type = "range";
    const select = document.createElement("select");
    const mediaCard = document.createElement("div");
    mediaCard.dataset.mediaTile = "true";
    const context = {
      view: "editor" as const,
      blocked: false,
      focusedPanel: "timeline" as const,
      compatibilityReadOnly: false,
      cropEditingActive: false,
    };

    for (const control of [buttonLabel, slider, select, mediaCard]) {
      expect(shouldHandleTransportSpaceKey(event({ target: control }), "editor")).toBe(true);
      expect(resolveDocumentedShortcut(event({ target: control }), context)).toEqual({
        type: "transport",
      });
    }
    expect(
      resolveDocumentedShortcut(event({ code: "Escape", target: buttonLabel }), context),
    ).toEqual({ type: "escape" });
    expect(
      resolveDocumentedShortcut(event({ code: "Backquote", target: buttonLabel }), context),
    ).toEqual({ type: "maximize" });
    expect(
      resolveDocumentedShortcut(
        event({ code: "KeyS", metaKey: true, target: buttonLabel }),
        context,
      ),
    ).toEqual({ type: "application", id: "save" });
  });

  it("toggles playback synchronously on Space keydown", () => {
    let toggles = 0;
    const e = event({
      preventDefault: () => {},
      stopPropagation: () => {},
    } as Partial<KeyboardEvent>);

    const handled = handleTransportSpaceKeyDown(e, {
      view: "editor",
      previewMediaId: null,
      timelinePlaybackAllowed: true,
      requestMediaPreviewToggle: () => {},
      togglePlay: () => {
        toggles += 1;
      },
    });

    expect(handled).toBe(true);
    expect(toggles).toBe(1);
  });

  it("does not toggle repeatedly while Space is held", () => {
    let toggles = 0;
    const e = event({
      repeat: true,
      preventDefault: () => {},
      stopPropagation: () => {},
    } as Partial<KeyboardEvent>);

    const handled = handleTransportSpaceKeyDown(e, {
      view: "editor",
      previewMediaId: null,
      timelinePlaybackAllowed: true,
      requestMediaPreviewToggle: () => {},
      togglePlay: () => {
        toggles += 1;
      },
    });

    expect(handled).toBe(true);
    expect(toggles).toBe(0);
  });

  it("blocks at the global capture listener even when later blockers registered afterward", async () => {
    const requestMediaPreviewToggle = vi.fn();
    useEditorUiStore.setState({
      view: "editor",
      settingsOpen: false,
      exportDialogOpen: false,
      saveAsProgress: null,
      projectSettingsPrompt: null,
      pendingSwapClipId: null,
      previewMediaId: "asset-1",
      requestMediaPreviewToggle,
    });
    useUpdateStore.setState({ phase: "downloading", dialogOpen: true });
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    const Harness = () => {
      useKeyboardShortcuts();
      return null;
    };
    await act(async () => root.render(createElement(Harness)));

    const laterCaptureListener = vi.fn();
    window.addEventListener("keydown", laterCaptureListener, true);
    const space = new KeyboardEvent("keydown", {
      key: " ",
      code: "Space",
      bubbles: true,
      cancelable: true,
    });
    await act(async () => window.dispatchEvent(space));

    expect(space.defaultPrevented).toBe(true);
    expect(requestMediaPreviewToggle).not.toHaveBeenCalled();
    expect(laterCaptureListener).not.toHaveBeenCalled();
    window.removeEventListener("keydown", laterCaptureListener, true);
    await act(async () => root.unmount());
    container.remove();
    useUpdateStore.setState({ phase: "idle", dialogOpen: false });
  });

  it("consumes native-control Space once without activating the local control", () => {
    let toggles = 0;
    let prevented = 0;
    let stopped = 0;
    const button = document.createElement("button");
    const dispatch = (repeat: boolean) =>
      handleTransportSpaceKeyDown(
        event({
          target: button,
          repeat,
          preventDefault: () => {
            prevented += 1;
          },
          stopPropagation: () => {
            stopped += 1;
          },
        }),
        {
          view: "editor",
          previewMediaId: null,
          timelinePlaybackAllowed: true,
          requestMediaPreviewToggle: () => {},
          togglePlay: () => {
            toggles += 1;
          },
        },
      );

    expect(dispatch(false)).toBe(true);
    expect(dispatch(true)).toBe(true);
    expect({ toggles, prevented, stopped }).toEqual({
      toggles: 1,
      prevented: 2,
      stopped: 2,
    });
  });

  it("does not intercept Space while an IME composition is active", () => {
    const composing = event({ isComposing: true });
    const context = {
      view: "editor" as const,
      blocked: false,
      focusedPanel: "preview" as const,
      compatibilityReadOnly: false,
      cropEditingActive: false,
    };

    expect(shouldHandleTransportSpaceKey(composing, "editor")).toBe(false);
    expect(resolveDocumentedShortcut(composing, context)).toBeNull();
  });

  it("does not start timeline playback from Space when route is Unsupported", () => {
    let toggles = 0;
    const e = event({
      preventDefault: () => {},
      stopPropagation: () => {},
    } as Partial<KeyboardEvent>);

    const handled = handleTransportSpaceKeyDown(e, {
      view: "editor",
      previewMediaId: null,
      timelinePlaybackAllowed: false,
      requestMediaPreviewToggle: () => {},
      togglePlay: () => {
        toggles += 1;
      },
    });

    expect(handled).toBe(true);
    expect(toggles).toBe(0);
  });

  it("does not export stale focus-release or keyup-suppression helpers", async () => {
    const shortcuts = await import("./useKeyboardShortcuts");

    expect("releaseTransportSpaceFocus" in shortcuts).toBe(false);
    expect("suppressTransportSpaceKeyUp" in shortcuts).toBe(false);
  });
});

describe("native keyboard-control ownership", () => {
  it("reserves arrow keys for separators and range inputs", () => {
    const separator = document.createElement("div");
    separator.setAttribute("role", "separator");
    const range = document.createElement("input");
    range.type = "range";
    const context = {
      view: "editor" as const,
      blocked: false,
      focusedPanel: "timeline" as const,
      compatibilityReadOnly: false,
      cropEditingActive: false,
    };

    for (const target of [separator, range]) {
      expect(
        resolveDocumentedShortcut(event({ code: "ArrowLeft", target }), context),
      ).toBeNull();
      expect(
        resolveDocumentedShortcut(event({ code: "ArrowRight", target }), context),
      ).toBeNull();
      expect(resolveDocumentedShortcut(event({ target }), context)).toEqual({
        type: "transport",
      });
    }
  });
});

describe("project save shortcut", () => {
  it("prevents the native shortcut but ignores repeated KeyS events", () => {
    let saves = 0;
    let prevented = 0;
    const handled = handleProjectSaveKeyDown(
      event({
        code: "KeyS",
        metaKey: true,
        repeat: true,
        preventDefault: () => {
          prevented += 1;
        },
      }),
      () => {
        saves += 1;
      },
    );

    expect(handled).toBe(true);
    expect(prevented).toBe(1);
    expect(saves).toBe(0);
  });

  it("complete_documented_shortcut_table", () => {
    expect(DOCUMENTED_SHORTCUT_ROWS).toEqual([
      "transport",
      "timeline-arrows",
      "media-arrows",
      "delete",
      "tools",
      "range",
      "trim",
      "maximize",
      "return-escape",
      "undo-redo",
      "clipboard",
      "split",
      "file",
      "panels",
      "layouts",
      "fullscreen",
      "help",
      "settings",
    ]);

    const context = {
      view: "editor" as const,
      blocked: false,
      focusedPanel: "timeline" as const,
      compatibilityReadOnly: false,
      cropEditingActive: false,
    };
    expect(resolveDocumentedShortcut(event({ code: "BracketLeft" }), context)).toEqual({
      type: "trimStart",
    });
    expect(resolveDocumentedShortcut(event({ code: "BracketRight" }), context)).toEqual({
      type: "trimEnd",
    });
    expect(
      resolveDocumentedShortcut(
        event({ code: "KeyS", metaKey: true, shiftKey: true }),
        context,
      ),
    ).toEqual({ type: "application", id: "saveAs" });
    expect(
      resolveDocumentedShortcut(event({ code: "Slash", ctrlKey: true, shiftKey: true }), context),
    ).toEqual({ type: "application", id: "shortcuts" });
  });
});

describe("Agent panel shortcut", () => {
  it("supports both macOS and Windows modifiers without repeating", () => {
    for (const modifiers of [{ metaKey: true }, { ctrlKey: true }]) {
      let toggles = 0;
      let prevented = 0;
      const handled = handleAgentPanelKeyDown(
        event({
          code: "KeyA",
          altKey: true,
          ...modifiers,
          preventDefault: () => {
            prevented += 1;
          },
        }),
        "editor",
        () => {
          toggles += 1;
        },
      );
      const repeatHandled = handleAgentPanelKeyDown(
        event({
          code: "KeyA",
          altKey: true,
          repeat: true,
          ...modifiers,
          preventDefault: () => {},
        }),
        "editor",
        () => {
          toggles += 1;
        },
      );

      expect(handled).toBe(true);
      expect(repeatHandled).toBe(true);
      expect(prevented).toBe(1);
      expect(toggles).toBe(1);
    }
  });

  it("does not consume the shortcut outside the editor", () => {
    expect(
      handleAgentPanelKeyDown(
        event({ code: "KeyA", metaKey: true, altKey: true }),
        "home",
        () => {
          throw new Error("must not toggle");
        },
      ),
    ).toBe(false);
  });
});

describe("keyboard edit failures", () => {
  it("toasts a rejected ⌘Z instead of leaking an unhandled rejection", async () => {
    const undo = vi.spyOn(edit, "undo").mockRejectedValue(new Error("history locked"));
    useEditorUiStore.setState({
      view: "editor",
      settingsOpen: false,
      exportDialogOpen: false,
      saveAsProgress: null,
      projectSettingsPrompt: null,
      pendingSwapClipId: null,
      focusedPanel: "timeline",
      toast: null,
    });
    const unhandled = vi.fn();
    const onUnhandled = (event: PromiseRejectionEvent) => unhandled(event.reason);
    window.addEventListener("unhandledrejection", onUnhandled);
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    const Harness = () => {
      useKeyboardShortcuts();
      return null;
    };
    try {
      await act(async () => root.render(createElement(Harness)));
      await act(async () => {
        window.dispatchEvent(
          new KeyboardEvent("keydown", { code: "KeyZ", key: "z", metaKey: true, bubbles: true, cancelable: true }),
        );
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      expect(undo).toHaveBeenCalledOnce();
      expect(useEditorUiStore.getState().toast?.message).toContain("history locked");
      expect(unhandled).not.toHaveBeenCalled();
    } finally {
      window.removeEventListener("unhandledrejection", onUnhandled);
      await act(async () => root.unmount());
      container.remove();
      undo.mockRestore();
    }
  });
});
