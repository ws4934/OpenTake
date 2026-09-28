// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ChatMessage, ChatSession } from "../../lib/types";

const apiMocks = vi.hoisted(() => ({
  chatSend: vi.fn(),
  sessions: [] as ChatSession[],
}));

vi.mock("../../i18n", () => ({
  useT: () => (key: string) => key,
}));

vi.mock("../../lib/types", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../lib/types")>();
  return { ...actual, isBoundedAgentContentBlock: vi.fn(actual.isBoundedAgentContentBlock) };
});

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
  chatSessions: vi.fn(async () => apiMocks.sessions),
  onChatDelta: vi.fn(async () => () => {}),
  onChatToolCall: vi.fn(async () => () => {}),
  onChatDone: vi.fn(async () => () => {}),
}));

import { isBoundedAgentContentBlock } from "../../lib/types";
import { useChatStore } from "../../store/chatStore";
import { useProjectStore } from "../../store/projectStore";
import { useSettingsStore } from "../../store/settingsStore";
import { AgentPanel } from "./AgentPanel";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT =
  true;

const SESSION_ID = "chat-tools";
/** A 512 KiB PNG, base64-encoded. */
const LARGE_IMAGE_BASE64 = "A".repeat(Math.ceil((512 * 1024) / 3) * 4);
const validateBlock = vi.mocked(isBoundedAgentContentBlock);

let root: Root | null = null;
let container: HTMLDivElement | null = null;

function toolTurn(index: number): ChatMessage[] {
  return [
    {
      id: `user-${index}`,
      role: "user",
      content: `Show frame ${index}`,
      toolCalls: [],
      createdAt: index * 10,
    },
    {
      id: `assistant-${index}`,
      role: "assistant",
      content: `Frame ${index}`,
      toolCalls: [],
      blocks: [
        {
          type: "toolUse",
          id: `tool-${index}`,
          name: "inspect_media",
          input: { frame: index },
          result: {
            content: [
              { kind: "text", text: `Frame ${index} inspected` },
              { kind: "image", mediaType: "image/png", base64: LARGE_IMAGE_BASE64 },
            ],
          },
        },
        { type: "text", text: `Frame ${index}` },
      ],
      createdAt: index * 10 + 1,
    },
  ];
}

async function renderPanel(sessions: ChatSession[]): Promise<void> {
  apiMocks.sessions = sessions;
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => {
    root?.render(<AgentPanel />);
    await Promise.resolve();
    await Promise.resolve();
  });
}

function transcript(): HTMLDivElement {
  const element = container?.querySelector(".agent-message")?.parentElement;
  if (!(element instanceof HTMLDivElement)) throw new Error("missing transcript");
  return element;
}

/** happy-dom has no layout: every rendered turn is 100 px tall in a 300 px viewport. */
function stubLayout(element: HTMLDivElement): void {
  Object.defineProperty(element, "scrollHeight", {
    configurable: true,
    get: () => element.childElementCount * 100,
  });
  Object.defineProperty(element, "clientHeight", { configurable: true, value: 300 });
}

async function scrollTo(element: HTMLDivElement, scrollTop: number): Promise<void> {
  await act(async () => {
    element.scrollTop = scrollTop;
    element.dispatchEvent(new Event("scroll"));
  });
}

async function streamReply(messageId: string, text: string): Promise<void> {
  await act(async () => {
    const chat = useChatStore.getState();
    chat.beginMessage(SESSION_ID, messageId);
    chat.appendBlockDelta(SESSION_ID, messageId, 0, 0, text);
  });
}

async function newTurn(index: number): Promise<void> {
  const current = useChatStore.getState().sessionMessages[SESSION_ID] ?? [];
  await act(async () => {
    useChatStore.getState().setMessagesForSession(SESSION_ID, [...current, ...toolTurn(index)]);
  });
}

beforeEach(() => {
  useProjectStore.setState({ projectEpoch: 12, projectPath: "/tmp/Tools.opentake" });
  useSettingsStore.setState({ byokProvider: "anthropic" });
  useChatStore.setState({
    sessionId: "stale-session",
    messages: [],
    streaming: false,
    streamingId: null,
    sessionMessages: {},
    sessionOrder: [],
    drafts: {},
    draftOrder: [],
    blockedMessageKeys: {},
    blockedMessageOrder: [],
    historyResyncRequests: {},
    resyncingSessionIds: {},
    resyncSessionOrder: [],
    deletedSessionIds: {},
    deletedSessionOrder: [],
    projectEpoch: null,
    projectPath: null,
    projectGeneration: 0,
    sessionVersions: {},
    composerDraft: null,
  });
  apiMocks.chatSend.mockReset();
  apiMocks.chatSend.mockResolvedValue(undefined);
  validateBlock.mockClear();
});

afterEach(async () => {
  if (root) await act(async () => root?.unmount());
  container?.remove();
  root = null;
  container = null;
});

describe("AgentPanel tool history rendering", () => {
  const history = Array.from({ length: 10 }, (_, index) => toolTurn(index)).flat();

  it("does not validate collapsed tool results while a reply streams", async () => {
    await renderPanel([{ id: SESSION_ID, messages: history, createdAt: 1, isOpen: true }]);
    expect(container?.querySelectorAll("[data-tool-activity]")).toHaveLength(10);
    expect(validateBlock).not.toHaveBeenCalled();

    await streamReply("assistant-streaming", "Checking");
    await act(async () => {
      useChatStore.getState().appendBlockDelta(SESSION_ID, "assistant-streaming", 1, 0, " the cut");
    });

    expect(container?.textContent).toContain("Checking the cut");
    expect(validateBlock).not.toHaveBeenCalled();
  });

  it("validates an opened result once and not again for later tokens", async () => {
    await renderPanel([{ id: SESSION_ID, messages: history, createdAt: 1, isOpen: true }]);
    const trigger = container!.querySelector<HTMLButtonElement>("[data-tool-activity-trigger]")!;

    await act(async () => trigger.click());
    const region = document.getElementById(trigger.getAttribute("aria-controls")!);
    expect(region?.textContent).toContain("Frame 0 inspected");
    expect(region?.querySelector("img")?.getAttribute("src")).toBe(
      `data:image/png;base64,${LARGE_IMAGE_BASE64}`,
    );
    expect(validateBlock).toHaveBeenCalledTimes(1);

    await streamReply("assistant-streaming", "Checking");

    expect(validateBlock).toHaveBeenCalledTimes(1);
  });
});

describe("AgentPanel transcript scrolling", () => {
  const history = Array.from({ length: 3 }, (_, index) => toolTurn(index)).flat();

  it("keeps the reader's position when output arrives while they read earlier turns", async () => {
    await renderPanel([{ id: SESSION_ID, messages: history, createdAt: 1, isOpen: true }]);
    const element = transcript();
    stubLayout(element);
    await scrollTo(element, 100);

    await streamReply("assistant-streaming", "Checking");
    await newTurn(3);

    expect(element.scrollTop).toBe(100);
  });

  it("follows new output while the reader is at the bottom", async () => {
    await renderPanel([{ id: SESSION_ID, messages: history, createdAt: 1, isOpen: true }]);
    const element = transcript();
    stubLayout(element);
    // 600 px of turns in a 300 px viewport: within 48 px of the bottom.
    await scrollTo(element, 260);

    await newTurn(3);

    expect(element.scrollHeight).toBe(800);
    expect(element.scrollTop).toBe(800);
  });

  it("resumes following once the reader scrolls back to the bottom", async () => {
    await renderPanel([{ id: SESSION_ID, messages: history, createdAt: 1, isOpen: true }]);
    const element = transcript();
    stubLayout(element);
    await scrollTo(element, 0);
    await newTurn(3);
    expect(element.scrollTop).toBe(0);

    await scrollTo(element, 500);
    await newTurn(4);

    expect(element.scrollTop).toBe(1000);
  });

  it("brings the reader's own message into view", async () => {
    await renderPanel([{ id: SESSION_ID, messages: history, createdAt: 1, isOpen: true }]);
    const element = transcript();
    stubLayout(element);
    await scrollTo(element, 0);
    const composer = container!.querySelector<HTMLTextAreaElement>("textarea")!;
    await act(async () => {
      Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set?.call(
        composer,
        "Trim the intro",
      );
      composer.dispatchEvent(new Event("input", { bubbles: true }));
    });

    await act(async () => {
      composer.dispatchEvent(
        new KeyboardEvent("keydown", { key: "Enter", keyCode: 13, bubbles: true, cancelable: true }),
      );
      await Promise.resolve();
    });

    expect(apiMocks.chatSend).toHaveBeenCalledOnce();
    expect(element.scrollTop).toBe(element.scrollHeight);
    expect(element.scrollTop).toBe(700);
  });

  it("opens another chat at its latest message", async () => {
    const other: ChatMessage[] = toolTurn(7);
    await renderPanel([
      { id: SESSION_ID, messages: history, createdAt: 2, isOpen: true },
      { id: "chat-other", messages: other, createdAt: 1, isOpen: true },
    ]);
    const element = transcript();
    stubLayout(element);
    await scrollTo(element, 0);

    const tabs = Array.from(container!.querySelectorAll<HTMLButtonElement>('[role="tab"]'));
    await act(async () => tabs[1]!.click());

    expect(useChatStore.getState().sessionId).toBe("chat-other");
    expect(element.scrollTop).toBe(element.scrollHeight);
    expect(element.scrollTop).toBe(200);
  });
});
