// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

const native = vi.hoisted(() => {
  type Item = {
    id: string;
    predefined?: string;
    accelerator?: string;
    text?: string;
    children: Item[];
    setEnabled: ReturnType<typeof vi.fn>;
    setText: ReturnType<typeof vi.fn>;
    setChecked: ReturnType<typeof vi.fn>;
    setAsAppMenu: ReturnType<typeof vi.fn>;
  };
  const state = { menu: null as Item | null, menus: [] as Item[], predefined: [] as string[] };
  function item(options: { id?: string; items?: Item[]; item?: string; accelerator?: string; text?: string }): Item {
    const handle: Item = {
      id: options.id ?? options.item ?? "predefined",
      predefined: options.item,
      accelerator: options.accelerator,
      text: options.text,
      children: options.items ?? [],
      setEnabled: vi.fn().mockResolvedValue(undefined),
      setText: vi.fn().mockResolvedValue(undefined),
      setChecked: vi.fn().mockResolvedValue(undefined),
      setAsAppMenu: vi.fn(async () => { state.menu = handle; }),
    };
    if (typeof options.item === "string") state.predefined.push(options.item);
    if (!options.id && options.items) state.menus.push(handle);
    return handle;
  }
  return { state, item };
});

vi.mock("@tauri-apps/api/menu", () => ({
  Menu: { new: async (options: Parameters<typeof native.item>[0]) => native.item(options) },
  Submenu: { new: async (options: Parameters<typeof native.item>[0]) => native.item(options) },
  MenuItem: { new: async (options: Parameters<typeof native.item>[0]) => native.item(options) },
  CheckMenuItem: { new: async (options: Parameters<typeof native.item>[0]) => native.item(options) },
  PredefinedMenuItem: { new: async (options: Parameters<typeof native.item>[0]) => native.item(options) },
}));
vi.mock("../../lib/api", async (importOriginal) => ({
  ...await importOriginal<typeof import("../../lib/api")>(),
  isTauri: true,
}));

import { ApplicationMenuBridge } from "./ViewMenu";
import { useEditorUiStore } from "../../store/uiStore";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let root: Root | undefined;
let host: HTMLDivElement | undefined;

afterEach(async () => {
  await act(async () => root?.unmount());
  host?.remove();
  root = undefined;
  host = undefined;
  native.state.menu = null;
  native.state.menus = [];
  native.state.predefined = [];
  vi.restoreAllMocks();
});

async function mount(platform: string) {
  vi.spyOn(navigator, "platform", "get").mockReturnValue(platform);
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
  await act(async () => root!.render(<ApplicationMenuBridge />));
  await vi.waitFor(() => expect(native.state.menu).not.toBeNull());
}

describe("native text editing menu", () => {
  it("routes focused macOS inputs and textareas to the native responder menu", async () => {
    await mount("MacIntel");
    const documentMenu = native.state.menu!;
    for (const tag of ["input", "textarea"] as const) {
      const input = document.createElement(tag);
      host!.append(input);
      await act(async () => input.focus());
      await vi.waitFor(() => expect(native.state.menu?.children[2]?.id).toBe("editText"));
      expect(native.state.menu!.children[2]!.children.map(({ predefined }) => predefined)).toEqual([
        "Undo", "Redo", "Separator", "Cut", "Copy", "Paste", "SelectAll",
      ]);
      await act(async () => input.blur());
      await vi.waitFor(() => expect(native.state.menu).toBe(documentMenu));
      expect(documentMenu.children.map(({ id }) => id)).toEqual(["app", "file", "edit", "view", "help"]);
      input.remove();
    }
  });

  it("preserves the complete previous menu when native replacement fails", async () => {
    await mount("MacIntel");
    const previous = native.state.menu!;
    const textMenu = native.state.menus.find((menu) => menu.children[2]?.id === "editText")!;
    const reported = vi.spyOn(console, "error").mockImplementation(() => {});
    textMenu.setAsAppMenu.mockRejectedValueOnce(new Error("native menu unavailable"));
    const input = document.createElement("input");
    host!.append(input);
    await act(async () => input.focus());
    await vi.waitFor(() => expect(reported).toHaveBeenCalledOnce());
    expect(native.state.menu).toBe(previous);
    expect(previous.children.map(({ id }) => id)).toEqual(["app", "file", "edit", "view", "help"]);
    expect(useEditorUiStore.getState().toast).not.toBeNull();
    await act(async () => { input.blur(); input.focus(); });
    await vi.waitFor(() => expect(native.state.menu).toBe(textMenu));
  });

  it("keeps the document menu and browser text handling on Windows", async () => {
    await mount("Win32");
    const menu = native.state.menu!;
    const documentMenu = menu.children[2]!;
    const input = document.createElement("input");
    host!.append(input);
    await act(async () => input.focus());
    await vi.waitFor(() => expect(documentMenu.children[0]!.setEnabled).toHaveBeenLastCalledWith(false));
    expect(menu.children[2]).toBe(documentMenu);
    expect(native.state.predefined).not.toContain("Undo");
    const copy = documentMenu.children.find((item) => item.id === "copy")!;
    expect(copy.accelerator).toBeUndefined();
    expect(copy.text).toContain("\tCtrl+C");
    await vi.waitFor(() => expect(copy.setText).toHaveBeenLastCalledWith(
      expect.stringContaining("\tCtrl+C"),
    ));
  });
});
