// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

const native = vi.hoisted(() => {
  type Item = {
    id: string;
    predefined?: string;
    children: Item[];
    setEnabled: ReturnType<typeof vi.fn>;
    setText: ReturnType<typeof vi.fn>;
    setChecked: ReturnType<typeof vi.fn>;
    remove: (item: Item) => Promise<void>;
    insert: (item: Item, index: number) => Promise<void>;
    setAsAppMenu: () => Promise<void>;
  };
  const state = { menu: null as Item | null, predefined: [] as string[] };
  function item(options: { id?: string; items?: Item[]; item?: string }): Item {
    const handle: Item = {
      id: options.id ?? options.item ?? "predefined",
      predefined: options.item,
      children: options.items ?? [],
      setEnabled: vi.fn().mockResolvedValue(undefined),
      setText: vi.fn().mockResolvedValue(undefined),
      setChecked: vi.fn().mockResolvedValue(undefined),
      remove: async (removed) => {
        const index = handle.children.indexOf(removed);
        if (index < 0) throw new Error("menu item is absent");
        handle.children.splice(index, 1);
      },
      insert: async (added, index) => { handle.children.splice(index, 0, added); },
      setAsAppMenu: async () => { state.menu = handle; },
    };
    if (typeof options.item === "string") state.predefined.push(options.item);
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

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let root: Root | undefined;
let host: HTMLDivElement | undefined;

afterEach(async () => {
  await act(async () => root?.unmount());
  host?.remove();
  root = undefined;
  host = undefined;
  native.state.menu = null;
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
    const menu = native.state.menu!;
    const documentMenu = menu.children[2]!;
    for (const tag of ["input", "textarea"] as const) {
      const input = document.createElement(tag);
      host!.append(input);
      await act(async () => input.focus());
      await vi.waitFor(() => expect(menu.children[2]?.id).toBe("editText"));
      expect(menu.children[2]!.children.map(({ predefined }) => predefined)).toEqual([
        "Undo", "Redo", "Separator", "Cut", "Copy", "Paste", "SelectAll",
      ]);
      await act(async () => input.blur());
      await vi.waitFor(() => expect(menu.children[2]).toBe(documentMenu));
      expect(menu.children.map(({ id }) => id)).toEqual(["app", "file", "edit", "view", "help"]);
      input.remove();
    }
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
  });
});
