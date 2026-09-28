// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  chatSend: vi.fn(),
}));

vi.mock("../../i18n", () => ({
  useT: () => (key: string) => key,
}));

vi.mock("../../lib/api", () => ({
  isTauri: true,
  chatCancel: vi.fn(async () => {}),
  chatHistory: vi.fn(async () => []),
  chatHistoryAuthoritative: vi.fn(async () => []),
  chatSend: apiMocks.chatSend,
  chatSessionSetOpen: vi.fn(async (sessionId: string, isOpen: boolean) => ({
    id: sessionId,
    messages: [],
    createdAt: 1,
    isOpen,
  })),
  chatSessions: vi.fn(async () => [
    { id: "chat-ime", messages: [], createdAt: 1, isOpen: true },
  ]),
  onChatDelta: vi.fn(async () => () => {}),
  onChatToolCall: vi.fn(async () => () => {}),
  onChatDone: vi.fn(async () => () => {}),
}));

import { useChatStore } from "../../store/chatStore";
import { useProjectStore } from "../../store/projectStore";
import { useSettingsStore } from "../../store/settingsStore";
import { AgentPanel } from "./AgentPanel";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT =
  true;

let root: Root | null = null;
let container: HTMLDivElement | null = null;

function textarea(): HTMLTextAreaElement {
  const element = container?.querySelector<HTMLTextAreaElement>("textarea");
  if (!element) throw new Error("missing composer");
  return element;
}

async function type(value: string): Promise<void> {
  const element = textarea();
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set?.call(
      element,
      value,
    );
    element.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

async function keyDown(init: KeyboardEventInit): Promise<KeyboardEvent> {
  const event = new KeyboardEvent("keydown", { bubbles: true, cancelable: true, ...init });
  await act(async () => {
    textarea().dispatchEvent(event);
    await Promise.resolve();
  });
  return event;
}

async function composition(type: "compositionstart" | "compositionend"): Promise<void> {
  await act(async () => {
    textarea().dispatchEvent(new Event(type, { bubbles: true }));
  });
}

beforeEach(async () => {
  useProjectStore.setState({ projectEpoch: 7, projectPath: "/tmp/Ime.opentake" });
  useSettingsStore.setState({ byokProvider: "anthropic" });
  useChatStore.setState({
    sessionId: "stale-session",
    messages: [],
    streaming: false,
    streamingId: null,
    sessionMessages: {},
    sessionOrder: [],
    historyResyncRequests: {},
    resyncingSessionIds: {},
    resyncSessionOrder: [],
    projectEpoch: null,
    projectPath: null,
    composerDraft: null,
  });
  apiMocks.chatSend.mockReset();
  apiMocks.chatSend.mockResolvedValue(undefined);
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => {
    root?.render(<AgentPanel />);
    await Promise.resolve();
    await Promise.resolve();
  });
  await type("剪辑");
});

afterEach(async () => {
  vi.restoreAllMocks();
  if (root) await act(async () => root?.unmount());
  container?.remove();
  root = null;
  container = null;
});

describe("AgentPanel composer IME handling", () => {
  it("leaves an Enter that confirms a composing candidate to the IME", async () => {
    const event = await keyDown({ key: "Enter", isComposing: true, keyCode: 229 });

    expect(apiMocks.chatSend).not.toHaveBeenCalled();
    expect(event.defaultPrevented).toBe(false);
  });

  it("ignores the WebKit confirmation keydown that reports keyCode 229 only", async () => {
    const event = await keyDown({ key: "Enter", keyCode: 229 });

    expect(apiMocks.chatSend).not.toHaveBeenCalled();
    expect(event.defaultPrevented).toBe(false);
  });

  it("confirms on WebKit's compositionend-first order, then sends on the next Enter", async () => {
    await composition("compositionstart");
    await composition("compositionend");
    const confirming = await keyDown({ key: "Enter", keyCode: 229 });
    expect(apiMocks.chatSend).not.toHaveBeenCalled();
    expect(confirming.defaultPrevented).toBe(false);

    const sending = await keyDown({ key: "Enter", keyCode: 13 });
    expect(sending.defaultPrevented).toBe(true);
    expect(apiMocks.chatSend).toHaveBeenCalledOnce();
    expect(apiMocks.chatSend.mock.calls[0]?.[1]).toBe("剪辑");
  });

  it("ignores keys while composing even when the keydown carries no IME flag", async () => {
    await composition("compositionstart");
    await keyDown({ key: "Enter", keyCode: 13 });
    expect(apiMocks.chatSend).not.toHaveBeenCalled();
  });

  it("sends once for Chromium's Korean IME, which repeats Enter after the syllable commits", async () => {
    await composition("compositionstart");
    await keyDown({ key: "Enter", isComposing: true, keyCode: 229 });
    await composition("compositionend");
    await keyDown({ key: "Enter", keyCode: 13 });

    expect(apiMocks.chatSend).toHaveBeenCalledOnce();
  });

  it("sends on a plain Enter and keeps Shift+Enter for a new line", async () => {
    const newline = await keyDown({ key: "Enter", keyCode: 13, shiftKey: true });
    expect(newline.defaultPrevented).toBe(false);
    expect(apiMocks.chatSend).not.toHaveBeenCalled();

    const send = await keyDown({ key: "Enter", keyCode: 13 });
    expect(send.defaultPrevented).toBe(true);
    expect(apiMocks.chatSend).toHaveBeenCalledOnce();
  });
});
