import {
  memo,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
} from "react";
import {
  ChevronDown,
  ChevronRight,
  Plus,
  Send,
  Settings as SettingsIcon,
  Square,
  Trash2,
  Wrench,
  X,
} from "lucide-react";
import { useT } from "../../i18n";
import {
  chatCancel,
  chatHistoryAuthoritative,
  chatSend,
  chatSessionDelete,
  chatSessionSetOpen,
  chatSessions,
  isTauri,
  onChatDelta,
  onChatDone,
  onChatToolCall,
  type ChatStreamDecodeFailure,
  type ChatStreamIdentity,
} from "../../lib/api";
import {
  MAX_CHAT_IMAGE_BASE64_CHARS,
  type AgentContentBlock,
  type AgentToolResultContentBlock,
  type ChatMessage,
  type ChatSession,
  isBoundedAgentContentBlock,
} from "../../lib/types";
import { useSettingsStore } from "../../store/settingsStore";
import { mintSessionId, useChatStore } from "../../store/chatStore";
import { useEditorUiStore } from "../../store/uiStore";
import { useProjectStore } from "../../store/projectStore";
import { useImeComposition } from "../../hooks/useImeComposition";
import { Reveal } from "../ui/Reveal";

const NO_KEY_HINT = /Settings|设置|API key/i;
const HISTORY_RESYNC_RETRY_BASE_MS = 250;
const HISTORY_RESYNC_RETRY_MAX_MS = 4_000;
const HISTORY_RESYNC_RETRY_MAX_EXPONENT = 4;
/** The transcript keeps following new output while its bottom edge is within
 *  this distance; scrolling further up stops following until the reader
 *  returns to the bottom. */
const FOLLOW_LATEST_THRESHOLD_PX = 48;

export function AgentPanel() {
  const t = useT();
  const provider = useSettingsStore((state) => state.byokProvider);
  const setSettingsOpen = useEditorUiStore((state) => state.setSettingsOpen);
  const projectEpoch = useProjectStore((state) => state.projectEpoch);
  const projectPath = useProjectStore((state) => state.projectPath);

  const sessionId = useChatStore((state) => state.sessionId);
  const messages = useChatStore((state) => state.messages);
  const streaming = useChatStore((state) => state.streaming);
  const pushUser = useChatStore((state) => state.pushUser);
  const beginMessage = useChatStore((state) => state.beginMessage);
  const appendBlockDelta = useChatStore((state) => state.appendBlockDelta);
  const upsertBlock = useChatStore((state) => state.upsertBlock);
  const finalize = useChatStore((state) => state.finalize);
  const requestHistoryResync = useChatStore((state) => state.requestHistoryResync);
  const takeHistoryResyncRequest = useChatStore((state) => state.takeHistoryResyncRequest);
  const rescheduleHistoryResync = useChatStore((state) => state.rescheduleHistoryResync);
  const historyResyncRequests = useChatStore((state) => state.historyResyncRequests);
  const selectedSessionResyncing = useChatStore(
    (state) => state.resyncingSessionIds[state.sessionId] === true,
  );
  const setMessagesForSession = useChatStore((state) => state.setMessagesForSession);
  const installSessionSnapshot = useChatStore((state) => state.installSessionSnapshot);
  const deleteSession = useChatStore((state) => state.deleteSession);
  const resetProject = useChatStore((state) => state.resetProject);
  const reset = useChatStore((state) => state.reset);
  const composerDraft = useChatStore((state) => state.composerDraft);
  const setComposerDraft = useChatStore((state) => state.setComposerDraft);

  const [input, setInput] = useState("");
  const [pendingSessionId, setPendingSessionId] = useState<string | null>(null);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  // Closed conversations are not shown as tabs but still occupy the project's
  // chat storage, so the panel offers to delete them.
  const [closedSessionIds, setClosedSessionIds] = useState<string[]>([]);
  // Failed chat deletions stay visible until dismissed or the project changes.
  const [chatError, setChatError] = useState<string | null>(null);
  const sessionsRef = useRef<ChatSession[]>([]);
  const inputRef = useRef("");
  const resyncProjectRef = useRef<Record<string, { projectEpoch: number; projectPath: string }>>({});
  const tabMutationRef = useRef<Promise<void>>(Promise.resolve());
  const scrollRef = useRef<HTMLDivElement>(null);
  const followLatestRef = useRef(true);
  const followedSessionRef = useRef(sessionId);
  const mountedRef = useRef(true);
  const turnLocked = streaming || pendingSessionId === sessionId;
  const interactionLocked = turnLocked || selectedSessionResyncing;
  const ime = useImeComposition();

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    if (composerDraft === null) return;
    inputRef.current = composerDraft;
    setInput(composerDraft);
    setComposerDraft(null);
  }, [composerDraft, setComposerDraft]);

  useEffect(() => () => {
    setComposerDraft(inputRef.current || null);
  }, [setComposerDraft]);

  function commitSessions(next: ChatSession[]) {
    sessionsRef.current = next;
    setSessions(next);
  }

  function updateSessions(update: (current: ChatSession[]) => ChatSession[]) {
    commitSessions(update(sessionsRef.current));
  }

  function enqueueTabMutation(operation: () => Promise<void>) {
    const pending = tabMutationRef.current.then(operation, operation);
    tabMutationRef.current = pending.then(
      () => undefined,
      () => undefined,
    );
  }

  useEffect(() => {
    let disposed = false;
    const unsubscribers: Array<() => void> = [];
    const install = (subscription: Promise<() => void>) => {
      void subscription
        .then((unsubscribe) => {
          if (disposed) unsubscribe();
          else unsubscribers.push(unsubscribe);
        })
        .catch(() => {});
    };
    const matchesProject = (event: ChatStreamIdentity) => {
      const project = useProjectStore.getState();
      return event.projectEpoch === project.projectEpoch && event.projectPath === project.projectPath;
    };
    const begin = (event: ChatStreamIdentity) => {
      const chat = useChatStore.getState();
      if (chat.projectEpoch !== event.projectEpoch || chat.projectPath !== event.projectPath) {
        resetProject(mintSessionId(), event.projectEpoch, event.projectPath);
      }
      resyncProjectRef.current[event.sessionId] = {
        projectEpoch: event.projectEpoch,
        projectPath: event.projectPath,
      };
      beginMessage(event.sessionId, event.messageId);
      setPendingSessionId((current) => current === event.sessionId ? null : current);
    };
    const malformed = (failure: ChatStreamDecodeFailure) => {
      if (!failure.sessionId) return;
      const chat = useChatStore.getState();
      const isKnownSession = failure.sessionId === chat.sessionId ||
        sessionsRef.current.some((session) => session.id === failure.sessionId);
      if (!isKnownSession) return;
      const project = useProjectStore.getState();
      if (project.projectPath) {
        resyncProjectRef.current[failure.sessionId] = {
          projectEpoch: project.projectEpoch,
          projectPath: project.projectPath,
        };
      }
      setPendingSessionId((current) => current === failure.sessionId ? null : current);
      requestHistoryResync(failure.sessionId, failure.messageId, failure.reason);
    };

    install(onChatDelta((event) => {
      if (!matchesProject(event)) return;
      begin(event);
      appendBlockDelta(
        event.sessionId,
        event.messageId,
        event.sequence,
        event.blockIndex,
        event.delta,
      );
    }, malformed));
    install(onChatToolCall((event) => {
      if (!matchesProject(event)) return;
      begin(event);
      upsertBlock(
        event.sessionId,
        event.messageId,
        event.sequence,
        event.blockIndex,
        event.block,
      );
    }, malformed));
    install(onChatDone((event) => {
      if (!matchesProject(event)) return;
      begin(event);
      finalize(event.sessionId, event.messageId, event.sequence, event.message);
    }, malformed));

    return () => {
      disposed = true;
      unsubscribers.splice(0).forEach((unsubscribe) => unsubscribe());
    };
  }, [appendBlockDelta, beginMessage, finalize, requestHistoryResync, resetProject, upsertBlock]);

  useEffect(() => {
    if (!projectPath || Object.keys(historyResyncRequests).length === 0) return;
    const request = takeHistoryResyncRequest();
    if (!request) return;
    const requestProject = resyncProjectRef.current[request.sessionId];
    delete resyncProjectRef.current[request.sessionId];
    if (
      requestProject &&
      (requestProject.projectEpoch !== projectEpoch || requestProject.projectPath !== projectPath)
    ) {
      return;
    }
    const loadingEpoch = requestProject?.projectEpoch ?? projectEpoch;
    const loadingPath = requestProject?.projectPath ?? projectPath;
    const chat = useChatStore.getState();
    const loadingGeneration = chat.projectGeneration;
    const loadingSessionVersion = chat.sessionVersions[request.sessionId] ?? 0;
    void chatHistoryAuthoritative(request.sessionId, loadingEpoch, loadingPath)
      .then((history) => {
        const installed = installSessionSnapshot(
          request.sessionId,
          history,
          loadingGeneration,
          loadingSessionVersion,
        );
        if (!installed) {
          rescheduleHistoryResync(request, loadingGeneration);
          return;
        }
        const project = useProjectStore.getState();
        if (
          !mountedRef.current ||
          project.projectEpoch !== loadingEpoch ||
          project.projectPath !== loadingPath
        ) {
          return;
        }
        updateSessions((current) => current.map((session) =>
          session.id === request.sessionId ? { ...session, messages: history } : session,
        ));
      })
      .catch(() => {
        const retryAttempt = Math.min(
          (request.retryAttempt ?? 0) + 1,
          HISTORY_RESYNC_RETRY_MAX_EXPONENT + 1,
        );
        const retryDelay = Math.min(
          HISTORY_RESYNC_RETRY_BASE_MS * (2 ** (retryAttempt - 1)),
          HISTORY_RESYNC_RETRY_MAX_MS,
        );
        // Re-sync belongs to the project-scoped store, not this panel mount.
        // The store action rejects stale projects and deleted session identities.
        window.setTimeout(() => {
          rescheduleHistoryResync({ ...request, retryAttempt }, loadingGeneration);
        }, retryDelay);
      });
  }, [
    historyResyncRequests,
    installSessionSnapshot,
    projectEpoch,
    projectPath,
    rescheduleHistoryResync,
    takeHistoryResyncRequest,
  ]);

  useEffect(() => {
    const previousChat = useChatStore.getState();
    const sameProject = previousChat.projectEpoch === projectEpoch &&
      previousChat.projectPath === projectPath;
    const previousSessionId = sameProject ? previousChat.sessionId : null;
    const freshSessionId = mintSessionId();
    const projectChanged = resetProject(freshSessionId, projectEpoch, projectPath);
    if (projectChanged) {
      resyncProjectRef.current = {};
      setPendingSessionId(null);
      inputRef.current = "";
      setInput("");
    }
    commitSessions([]);
    setClosedSessionIds([]);
    setChatError(null);
    if (!isTauri || !projectPath) return;
    let disposed = false;
    const loadingEpoch = projectEpoch;
    const loadingPath = projectPath;
    const loadingChat = useChatStore.getState();
    const loadingGeneration = loadingChat.projectGeneration;
    const loadingSessionVersions = { ...loadingChat.sessionVersions };
    void chatSessions(loadingEpoch, loadingPath)
      .then((projectSessions) => {
        if (disposed) return;
        const project = useProjectStore.getState();
        const chat = useChatStore.getState();
        if (
          project.projectEpoch !== loadingEpoch ||
          project.projectPath !== loadingPath ||
          chat.projectGeneration !== loadingGeneration
        ) {
          return;
        }
        const openSessions = projectSessions.filter((session) => session.isOpen !== false);
        setClosedSessionIds(
          projectSessions
            .filter((session) => session.isOpen === false)
            .map((session) => session.id),
        );
        const mergedSessions = openSessions.map((session) => {
          if (!useChatStore.getState().resyncingSessionIds[session.id]) {
            installSessionSnapshot(
              session.id,
              session.messages,
              loadingGeneration,
              loadingSessionVersions[session.id] ?? 0,
            );
          }
          return {
            ...session,
            messages: useChatStore.getState().sessionMessages[session.id] ?? session.messages,
          };
        });
        commitSessions(mergedSessions);
        const latest = mergedSessions.find((session) => session.id === previousSessionId) ??
          mergedSessions[0];
        if (latest) {
          reset(latest.id);
        } else {
          const emptySessionId = useChatStore.getState().sessionId;
          const optimistic: ChatSession = {
            id: emptySessionId,
            messages: [],
            createdAt: Date.now(),
            isOpen: true,
          };
          commitSessions([optimistic]);
          enqueueTabMutation(async () => {
            if (disposed) return;
            try {
              const created = await chatSessionSetOpen(
                emptySessionId,
                true,
                loadingEpoch,
                loadingPath,
              );
              if (disposed) return;
              const currentProject = useProjectStore.getState();
              if (
                currentProject.projectEpoch === loadingEpoch &&
                currentProject.projectPath === loadingPath &&
                useChatStore.getState().sessionId === emptySessionId
              ) {
                updateSessions((current) =>
                  current.map((session) =>
                    session.id === emptySessionId ? created : session,
                  ),
                );
              }
            } catch {
              // Keep the local empty tab; sending a message will surface a
              // project persistence error through the normal chat path.
            }
          });
        }
      })
      .catch(() => {});
    return () => {
      disposed = true;
    };
  }, [installSessionSnapshot, projectEpoch, projectPath, reset, resetProject]);

  useEffect(() => {
    updateSessions((current) =>
      current.map((session) =>
        session.id === sessionId ? { ...session, messages } : session,
      ),
    );
  }, [messages, sessionId]);

  useEffect(() => {
    const element = scrollRef.current;
    if (!element) return;
    // Another chat opens at its latest message.
    if (followedSessionRef.current !== sessionId) {
      followedSessionRef.current = sessionId;
      followLatestRef.current = true;
    }
    if (followLatestRef.current) element.scrollTop = element.scrollHeight;
  }, [messages, sessionId]);

  const openSettings = useCallback(() => setSettingsOpen(true), [setSettingsOpen]);
  const turns = useMemo(() => groupConversationMessages(messages), [messages]);

  async function send() {
    const text = input.trim();
    if (!text || interactionLocked || !projectPath) return;
    const sendingEpoch = projectEpoch;
    const sendingPath = projectPath;
    const sendingSessionId = sessionId;
    inputRef.current = "";
    setInput("");
    // The reader's own message always brings the transcript to its end.
    followLatestRef.current = true;
    pushUser(text);
    setPendingSessionId(sendingSessionId);
    try {
      await chatSend(sendingSessionId, text, provider, sendingEpoch, sendingPath);
    } catch (error) {
      const project = useProjectStore.getState();
      const chat = useChatStore.getState();
      if (
        project.projectEpoch !== sendingEpoch ||
        project.projectPath !== sendingPath ||
        chat.sessionId !== sendingSessionId
      ) {
        return;
      }
      const errorText = `⚠️ ${error instanceof Error ? error.message : String(error)}`;
      const errorMessage: ChatMessage = {
        id: `assistant-local-error-${Date.now()}`,
        role: "assistant",
        content: errorText,
        toolCalls: [],
        blocks: [{ type: "text", text: errorText }],
        createdAt: Date.now(),
      };
      setMessagesForSession(
        sendingSessionId,
        [...(chat.sessionMessages[sendingSessionId] ?? []), errorMessage],
      );
      setPendingSessionId((current) => current === sendingSessionId ? null : current);
    }
  }

  function cancel() {
    if (!projectPath) return;
    void chatCancel(sessionId, projectEpoch, projectPath).catch(() => {});
  }

  function openSession(session: ChatSession) {
    if (interactionLocked || session.id === sessionId || !projectPath) return;
    const storedMessages = useChatStore.getState().sessionMessages[session.id];
    if (!storedMessages) setMessagesForSession(session.id, session.messages);
    reset(session.id);
  }

  function newChat() {
    if (interactionLocked || !projectPath) return;
    const openingEpoch = projectEpoch;
    const openingPath = projectPath;
    enqueueTabMutation(() => createNewChatNow(openingEpoch, openingPath));
  }

  async function createNewChatNow(openingEpoch: number, openingPath: string) {
    const project = useProjectStore.getState();
    if (project.projectEpoch !== openingEpoch || project.projectPath !== openingPath) return;
    const createdId = mintSessionId();
    const optimistic: ChatSession = {
      id: createdId,
      messages: [],
      createdAt: Date.now(),
      isOpen: true,
    };
    reset(createdId);
    updateSessions((current) => [optimistic, ...current]);
    try {
      const persisted = await chatSessionSetOpen(
        createdId,
        true,
        openingEpoch,
        openingPath,
      );
      const currentProject = useProjectStore.getState();
      if (
        currentProject.projectEpoch === openingEpoch &&
        currentProject.projectPath === openingPath
      ) {
        updateSessions((current) =>
          current.map((session) => (session.id === createdId ? persisted : session)),
        );
      }
    } catch {
      // Keep the reversible local tab available; the next message surfaces any
      // project persistence failure through the existing chat error path.
    }
  }

  function closeChat(session: ChatSession) {
    if (interactionLocked || !projectPath) return;
    const closingEpoch = projectEpoch;
    const closingPath = projectPath;
    enqueueTabMutation(() => closeChatNow(session.id, closingEpoch, closingPath));
  }

  async function closeChatNow(
    closingSessionId: string,
    closingEpoch: number,
    closingPath: string,
  ) {
    const before = useProjectStore.getState();
    if (before.projectEpoch !== closingEpoch || before.projectPath !== closingPath) return;
    try {
      await chatSessionSetOpen(closingSessionId, false, closingEpoch, closingPath);
    } catch {
      return;
    }
    const project = useProjectStore.getState();
    if (project.projectEpoch !== closingEpoch || project.projectPath !== closingPath) return;
    setClosedSessionIds((current) =>
      current.includes(closingSessionId) ? current : [...current, closingSessionId],
    );
    await removeTab(closingSessionId, closingEpoch, closingPath);
  }

  async function removeTab(removedSessionId: string, epoch: number, path: string) {
    const remaining = sessionsRef.current.filter(
      (candidate) => candidate.id !== removedSessionId,
    );
    commitSessions(remaining);
    deleteSession(removedSessionId);
    if (removedSessionId !== useChatStore.getState().sessionId) return;
    const next = remaining[0];
    if (next) {
      const storedMessages = useChatStore.getState().sessionMessages[next.id];
      if (!storedMessages) setMessagesForSession(next.id, next.messages);
      reset(next.id);
    } else {
      await createNewChatNow(epoch, path);
    }
  }

  function isCurrentProject(epoch: number, path: string): boolean {
    const project = useProjectStore.getState();
    return project.projectEpoch === epoch && project.projectPath === path;
  }

  function deleteChat(session: ChatSession) {
    if (interactionLocked || !projectPath) return;
    if (!window.confirm(t("agent.deleteChatConfirm"))) return;
    const deletingEpoch = projectEpoch;
    const deletingPath = projectPath;
    enqueueTabMutation(async () => {
      const before = useProjectStore.getState();
      if (before.projectEpoch !== deletingEpoch || before.projectPath !== deletingPath) return;
      try {
        await chatSessionDelete(session.id, deletingEpoch, deletingPath);
      } catch (error) {
        // The backend refused (for example a running turn); keep the tab and
        // say why, unless another project is open by now.
        if (!isCurrentProject(deletingEpoch, deletingPath)) return;
        setChatError(`${t("agent.deleteChatFailed")} ${errorText(error)}`);
        return;
      }
      if (!isCurrentProject(deletingEpoch, deletingPath)) return;
      setChatError(null);
      await removeTab(session.id, deletingEpoch, deletingPath);
    });
  }

  function deleteClosedChats() {
    if (interactionLocked || !projectPath || closedSessionIds.length === 0) return;
    if (!window.confirm(t("agent.deleteClosedChatsConfirm"))) return;
    const deletingEpoch = projectEpoch;
    const deletingPath = projectPath;
    const deleting = [...closedSessionIds];
    enqueueTabMutation(async () => {
      let failed = 0;
      let lastError: unknown = null;
      for (const closedId of deleting) {
        if (!isCurrentProject(deletingEpoch, deletingPath)) return;
        try {
          await chatSessionDelete(closedId, deletingEpoch, deletingPath);
        } catch (error) {
          failed += 1;
          lastError = error;
          continue;
        }
        if (!isCurrentProject(deletingEpoch, deletingPath)) return;
        setClosedSessionIds((current) => current.filter((id) => id !== closedId));
      }
      if (!isCurrentProject(deletingEpoch, deletingPath)) return;
      setChatError(
        failed === 0
          ? null
          : `${t("agent.deleteClosedChatsFailed")} ${failed}/${deleting.length}: ${errorText(lastError)}`,
      );
    });
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    // Enter that confirms an IME candidate belongs to the input method.
    if (ime.isComposingKeyDown(event)) return;
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void send();
    }
  }

  return (
    <div
      style={{
        height: "100%",
        width: "100%",
        display: "flex",
        flexDirection: "column",
        minHeight: 0,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "var(--space-sm) var(--space-md)",
          borderBottom: "var(--bw-hairline) solid var(--border-subtle)",
          flexShrink: 0,
        }}
      >
        <span style={{ fontSize: "var(--fs-sm)", fontWeight: 600, color: "var(--text-primary)" }}>
          {t("agent.title")}
        </span>
        <div style={{ display: "inline-flex", alignItems: "center", gap: 2 }}>
          {closedSessionIds.length > 0 && (
            <button
              type="button"
              onClick={() => deleteClosedChats()}
              disabled={interactionLocked || !projectPath}
              title={t("agent.deleteClosedChats")}
              aria-label={t("agent.deleteClosedChats")}
              className="hover-area"
              style={{
                width: 26,
                height: 26,
                display: "inline-flex",
                alignItems: "center",
                justifyContent: "center",
                borderRadius: "var(--radius-sm)",
                color: "var(--text-secondary)",
                opacity: interactionLocked || !projectPath ? 0.4 : 1,
              }}
            >
              <Trash2 size={14} />
            </button>
          )}
          <button
            type="button"
            onClick={() => void newChat()}
            disabled={interactionLocked || !projectPath}
            title={t("agent.newTab")}
            aria-label={t("agent.newTab")}
            className="hover-area"
            style={{
              width: 26,
              height: 26,
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              borderRadius: "var(--radius-sm)",
              color: "var(--text-secondary)",
              opacity: interactionLocked || !projectPath ? 0.4 : 1,
            }}
          >
            <Plus size={14} />
          </button>
        </div>
      </div>

      <div
        role="tablist"
        aria-label={t("agent.tabs")}
        style={{
          display: "flex",
          gap: 2,
          overflowX: "auto",
          padding: "var(--space-xs) var(--space-sm)",
          borderBottom: "var(--bw-hairline) solid var(--border-subtle)",
          flexShrink: 0,
        }}
      >
        {sessions.map((session, index) => {
          const title = sessionTitle(session, `${t("agent.newChat")} ${index + 1}`);
          const active = session.id === sessionId;
          return (
            <div
              key={session.id}
              style={{
                display: "inline-flex",
                alignItems: "center",
                minWidth: 0,
                borderRadius: "var(--radius-sm)",
                background: active ? "var(--bg-elevated)" : "transparent",
              }}
            >
              <button
                type="button"
                role="tab"
                aria-selected={active}
                aria-label={title}
                disabled={interactionLocked}
                onClick={() => openSession(session)}
                style={{
                  maxWidth: 120,
                  height: 24,
                  padding: "0 4px 0 var(--space-sm)",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  color: active ? "var(--text-primary)" : "var(--text-muted)",
                  fontSize: "var(--fs-xs)",
                }}
              >
                {title}
              </button>
              <button
                type="button"
                aria-label={`${t("agent.closeTab")} ${title}`}
                disabled={interactionLocked}
                onClick={() => void closeChat(session)}
                className="hover-area"
                style={{
                  width: 24,
                  height: 24,
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  borderRadius: "var(--radius-xs)",
                  color: "var(--text-muted)",
                  opacity: interactionLocked ? 0.4 : 1,
                }}
              >
                <X size={11} />
              </button>
              <button
                type="button"
                aria-label={`${t("agent.deleteChat")} ${title}`}
                title={t("agent.deleteChat")}
                disabled={interactionLocked}
                onClick={() => deleteChat(session)}
                className="hover-area"
                style={{
                  width: 24,
                  height: 24,
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  borderRadius: "var(--radius-xs)",
                  color: "var(--text-muted)",
                  opacity: interactionLocked ? 0.4 : 1,
                }}
              >
                <Trash2 size={11} />
              </button>
            </div>
          );
        })}
      </div>

      {chatError && (
        <div
          role="alert"
          style={{
            display: "flex",
            alignItems: "flex-start",
            gap: "var(--space-xs)",
            padding: "var(--space-xs) var(--space-md)",
            borderBottom: "var(--bw-hairline) solid var(--border-subtle)",
            color: "var(--text-primary)",
            fontSize: "var(--fs-xs)",
            flexShrink: 0,
          }}
        >
          <span style={{ flex: 1, minWidth: 0, overflowWrap: "anywhere" }}>{chatError}</span>
          <button
            type="button"
            aria-label={t("agent.dismissError")}
            onClick={() => setChatError(null)}
            className="hover-area"
            style={{
              width: 24,
              height: 24,
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              borderRadius: "var(--radius-xs)",
              color: "var(--text-muted)",
              flexShrink: 0,
            }}
          >
            <X size={11} />
          </button>
        </div>
      )}

      <div
        ref={scrollRef}
        onScroll={(event) => {
          const element = event.currentTarget;
          followLatestRef.current =
            element.scrollHeight - element.scrollTop - element.clientHeight <=
            FOLLOW_LATEST_THRESHOLD_PX;
        }}
        style={{
          flex: 1,
          minHeight: 0,
          overflowY: "auto",
          padding: "var(--space-md)",
          display: "flex",
          flexDirection: "column",
          gap: "var(--space-sm)",
        }}
      >
        {messages.length === 0 && !streaming && (
          <div
            style={{
              color: "var(--text-muted)",
              fontSize: "var(--fs-sm)",
              textAlign: "center",
              marginTop: "var(--space-lg)",
              padding: "0 var(--space-md)",
            }}
          >
            {isTauri ? t("agent.empty") : t("agent.desktopOnly")}
          </div>
        )}
        {turns.map((turnMessages) => (
          <ConversationMessage
            key={turnMessages[0].id}
            messages={turnMessages}
            onOpenSettings={openSettings}
          />
        ))}
      </div>

      <div
        style={{
          borderTop: "var(--bw-hairline) solid var(--border-subtle)",
          padding: "var(--space-sm) var(--space-md)",
          display: "flex",
          gap: "var(--space-sm)",
          alignItems: "flex-end",
          flexShrink: 0,
        }}
      >
        <textarea
          className="agent-composer__input"
          value={input}
          onChange={(event) => {
            inputRef.current = event.target.value;
            setInput(event.target.value);
          }}
          onKeyDown={onKeyDown}
          {...ime.compositionHandlers}
          placeholder={t("agent.inputPlaceholder")}
          aria-label={t("agent.inputPlaceholder")}
          disabled={!isTauri || interactionLocked}
          rows={1}
          style={{
            flex: 1,
            resize: "none",
            border: "var(--bw-thin) solid var(--border-subtle)",
            borderRadius: "var(--radius-sm)",
            padding: "var(--space-sm) var(--space-md)",
            fontFamily: "inherit",
            fontSize: "var(--fs-sm)",
            color: "var(--text-primary)",
            background: "var(--bg-elevated)",
            minHeight: 34,
            maxHeight: 120,
            opacity: !isTauri ? 0.6 : 1,
          }}
        />
        {turnLocked ? (
          <button
            type="button"
            onClick={cancel}
            title={t("agent.cancel")}
            aria-label={t("agent.cancel")}
            className="agent-composer__action"
            style={iconButtonStyle("var(--accent-spotlight)", "#fff")}
          >
            <Square size={14} />
          </button>
        ) : (
          <button
            type="button"
            onClick={() => void send()}
            disabled={!isTauri || interactionLocked || !input.trim()}
            title={t("agent.send")}
            aria-label={t("agent.send")}
            className="agent-composer__action"
            style={{
              ...iconButtonStyle("var(--accent-primary)", "#111"),
              opacity: isTauri && !interactionLocked && input.trim() ? 1 : 0.4,
              cursor: isTauri && !interactionLocked && input.trim() ? "pointer" : "not-allowed",
            }}
          >
            <Send size={14} />
          </button>
        )}
      </div>
    </div>
  );
}

function errorText(error: unknown): string {
  if (error instanceof Error) return error.message;
  return typeof error === "string" ? error : String(error);
}

function sessionTitle(session: ChatSession, fallback: string): string {
  const firstUserMessage = session.messages.find(
    (message) => message.role === "user" && authoritativeMessageText(message).trim().length > 0,
  );
  if (!firstUserMessage) return fallback;
  const compact = authoritativeMessageText(firstUserMessage).trim().replace(/\s+/g, " ");
  return compact.length > 20 ? `${compact.slice(0, 20)}…` : compact;
}

type ConversationMessageProps = (
  | { message: ChatMessage; messages?: never }
  | { message?: never; messages: ChatMessage[] }
) & {
  onOpenSettings: () => void;
};

/** Chat store updates replace only the message that changed, so a turn whose
 *  messages are all the same objects renders the same output. */
function sameMessages(
  left: readonly ChatMessage[] | undefined,
  right: readonly ChatMessage[] | undefined,
): boolean {
  if (left === right) return true;
  if (!left || !right || left.length !== right.length) return false;
  return left.every((message, index) => message === right[index]);
}

/** Streaming re-renders the panel for every token; earlier turns skip it. */
export const ConversationMessage = memo(
  ConversationMessageView,
  (previous, next) =>
    previous.message === next.message &&
    sameMessages(previous.messages, next.messages) &&
    previous.onOpenSettings === next.onOpenSettings,
);

function ConversationMessageView({
  message,
  messages,
  onOpenSettings,
}: ConversationMessageProps) {
  const t = useT();
  const turnMessages = messages ?? (message ? [message] : []);
  const firstMessage = turnMessages[0];
  if (!firstMessage) return null;
  const isUser = turnMessages.length === 1 && firstMessage.role === "user";
  const guided = turnMessages.some(
    (candidate) => candidate.role === "assistant" &&
      NO_KEY_HINT.test(authoritativeMessageText(candidate)),
  );

  return (
    <div
      className={`agent-message ${isUser ? "agent-message--user" : "agent-message--assistant"}`}
    >
      {isUser
        ? <div className="agent-message__user-surface">
            {authoritativeMessageText(firstMessage)}
          </div>
        : <AssistantTurn messages={turnMessages} />}
      {guided && (
        <button
          type="button"
          onClick={onOpenSettings}
          className="hover-area"
          style={{
            alignSelf: "flex-start",
            display: "inline-flex",
            alignItems: "center",
            gap: 4,
            height: 26,
            padding: "0 var(--space-sm)",
            borderRadius: "var(--radius-sm)",
            border: "var(--bw-thin) solid var(--border-subtle)",
            color: "var(--text-secondary)",
            fontSize: "var(--fs-xs)",
          }}
        >
          <SettingsIcon size={12} />
          {t("agent.openSettings")}
        </button>
      )}
    </div>
  );
}

function authoritativeMessageText(message: ChatMessage): string {
  if (message.blocks === undefined) return message.content;
  return message.blocks
    .flatMap((block) => block.type === "text" ? [block.text] : [])
    .join("");
}

function groupConversationMessages(messages: ChatMessage[]): ChatMessage[][] {
  const groups: ChatMessage[][] = [];
  messages.forEach((message) => {
    const previous = groups[groups.length - 1];
    if (message.role !== "user" && previous && previous[0].role !== "user") {
      previous.push(message);
    } else {
      groups.push([message]);
    }
  });
  return groups;
}

type ToolActivityBlock = Exclude<AgentContentBlock, { type: "text" }>;

type AssistantTurnProps =
  | { message: ChatMessage; messages?: never }
  | { message?: never; messages: ChatMessage[] };

export const AssistantTurn = memo(
  AssistantTurnView,
  (previous, next) =>
    previous.message === next.message && sameMessages(previous.messages, next.messages),
);

function AssistantTurnView({ message, messages }: AssistantTurnProps) {
  const turnMessages = messages ?? (message ? [message] : []);
  const toolNames = new Map<string, string>();
  turnMessages.forEach((candidate) => {
    candidate.blocks?.forEach((block) => {
      if (block.type === "toolUse") toolNames.set(block.id, block.name);
    });
  });
  const entries = turnMessages.flatMap((candidate) => {
    if (candidate.blocks !== undefined) {
      return candidate.blocks.map((block, messageBlockIndex) => ({
        block,
        key: `${candidate.id}-${messageBlockIndex}`,
      }));
    }
    const legacyBlocks: AgentContentBlock[] = [];
    if (candidate.content) legacyBlocks.push({ type: "text", text: candidate.content });
    legacyBlocks.push(...candidate.toolCalls.map((toolCall) => ({
      type: "toolUse" as const,
      id: toolCall.id,
      name: toolCall.name,
      input: toolCall.args,
      result: toolCall.result,
      isError: toolCall.isError,
    })));
    return legacyBlocks.map((block, messageBlockIndex) => ({
      block,
      key: `${candidate.id}-legacy-${messageBlockIndex}`,
    }));
  });

  return (
    <div className="agent-assistant-turn" data-assistant-turn>
      {entries.map(({ block, key }, index) => {
        if (block.type === "text") {
          return (
            <div
              className="agent-assistant-turn__text"
              data-agent-block-index={index}
              data-agent-block-type="text"
              key={key}
            >
              {block.text}
            </div>
          );
        }
        return (
          <InlineToolActivity
            block={block}
            dataBlockIndex={index}
            key={key}
            toolName={block.type === "toolResult" ? toolNames.get(block.toolUseId) : undefined}
          />
        );
      })}
    </div>
  );
}

export const InlineToolActivity = memo(function InlineToolActivity({
  block,
  dataBlockIndex,
  toolName,
}: {
  block: ToolActivityBlock;
  dataBlockIndex?: number;
  toolName?: string;
}) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const reactId = useId();
  const disclosureId = `agent-tool-${reactId.replace(/:/g, "")}`;
  const statusId = `${disclosureId}-status`;
  const triggerRef = useRef<HTMLButtonElement>(null);
  const isError = block.isError === true;
  const pending = block.type === "toolUse" && block.result === undefined && !isError;
  const status = isError ? "error" : pending ? "running" : "complete";
  const statusLabel = t(
    status === "error"
      ? "agent.toolFailed"
      : status === "running"
        ? "agent.toolRunning"
        : "agent.toolComplete",
  );
  const label = block.type === "toolUse"
    ? block.name
    : toolName ?? t("agent.toolResult");

  return (
    <div
      className="agent-tool-activity"
      data-agent-block-index={dataBlockIndex}
      data-agent-block-type={block.type}
      data-status={status}
      data-tool-activity
      onKeyDown={(event) => {
        if (event.key !== "Escape" || !open) return;
        event.preventDefault();
        event.stopPropagation();
        setOpen(false);
        triggerRef.current?.focus();
      }}
    >
      <div className="agent-tool-activity__summary">
        <button
          type="button"
          aria-controls={disclosureId}
          aria-describedby={statusId}
          aria-expanded={open}
          aria-label={label}
          data-tool-activity-trigger
          onClick={() => setOpen((value) => !value)}
          className="agent-tool-activity__trigger"
          ref={triggerRef}
        >
          {open
            ? <ChevronDown aria-hidden="true" size={12} />
            : <ChevronRight aria-hidden="true" size={12} />}
          <Wrench aria-hidden="true" size={12} />
          <span className="agent-tool-activity__name">{label}</span>
        </button>
        <span
          aria-atomic="true"
          aria-live="polite"
          className="agent-tool-activity__status"
          id={statusId}
          role="status"
        >
          {statusLabel}
        </span>
      </div>
      <Reveal id={disclosureId} open={open} role="group">
        <ToolActivityDetails block={block} label={label} />
      </Reveal>
    </div>
  );
});

type ToolUseResultView =
  | { kind: "json"; text: string }
  | { kind: "content"; content: AgentToolResultContentBlock[] }
  | { kind: "unavailable" };

/** The body of an open tool disclosure. `Reveal` renders it only while the
 *  disclosure is shown, so a collapsed block never serializes its payload or
 *  runs the bounded-content validation over an image-sized result. */
const ToolActivityDetails = memo(function ToolActivityDetails({
  block,
  label,
}: {
  block: ToolActivityBlock;
  label: string;
}) {
  const t = useT();
  const args = useMemo(
    () => (block.type === "toolUse" ? prettyJson(block.input) : ""),
    [block],
  );
  const result = useMemo((): ToolUseResultView | null => {
    if (block.type !== "toolUse" || block.result === undefined) return null;
    const content = codexMcpResultContent(block.result);
    if (content === undefined) return { kind: "json", text: prettyJson(block.result) };
    return content === null ? { kind: "unavailable" } : { kind: "content", content };
  }, [block]);

  return (
    <div className="agent-tool-activity__details">
      {block.type === "toolUse"
        ? <>
            <ToolDetail label={t("agent.toolArgs")} value={args} />
            {result?.kind === "json" &&
              <ToolDetail label={t("agent.toolResult")} value={result.text} />}
            {result?.kind === "unavailable" &&
              <span className="agent-tool-activity__image-error">
                {t("agent.toolResultUnavailable")}
              </span>}
            {result?.kind === "content" &&
              <ToolResultContents content={result.content} label={label} />}
          </>
        : <ToolResultContents content={block.content} label={label} />}
    </div>
  );
});

function codexMcpResultContent(
  value: unknown,
): AgentToolResultContentBlock[] | null | undefined {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return undefined;
  if (!Object.prototype.hasOwnProperty.call(value, "content")) return undefined;
  const candidate: AgentContentBlock = {
    type: "toolResult",
    toolUseId: "codex-result",
    content: (value as Record<string, unknown>).content as AgentToolResultContentBlock[],
  };
  return isBoundedAgentContentBlock(candidate) && candidate.type === "toolResult"
    ? candidate.content
    : null;
}

function ToolResultContents({
  content,
  label,
}: {
  content: AgentToolResultContentBlock[];
  label: string;
}) {
  const t = useT();
  return content.map((contentBlock, index) => {
    if (contentBlock.kind === "text") {
      return (
        <ToolDetail
          key={index}
          label={t("agent.toolResult")}
          value={contentBlock.text}
        />
      );
    }
    const source = safeRasterDataUri(contentBlock.mediaType, contentBlock.base64);
    return source
      ? <img
          alt={t("agent.toolImageAlt", { tool: label })}
          className="agent-tool-activity__image"
          key={index}
          src={source}
        />
      : <span className="agent-tool-activity__image-error" key={index}>
          {t("agent.toolImageUnavailable")}
        </span>;
  });
}

function ToolDetail({ label, value }: { label: string; value: string }) {
  return (
    <div className="agent-tool-activity__detail">
      <div className="agent-tool-activity__detail-label">{label}</div>
      <pre>{value}</pre>
    </div>
  );
}

function prettyJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2) ?? String(value);
  } catch {
    return String(value);
  }
}

const SAFE_RASTER_MEDIA_TYPES = new Set(["image/png", "image/jpeg", "image/webp", "image/gif"]);
const BASE64_PATTERN = /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/;

function safeRasterDataUri(mediaType: string, base64: string): string | null {
  const normalizedMediaType = mediaType.trim().toLowerCase();
  if (
    !SAFE_RASTER_MEDIA_TYPES.has(normalizedMediaType) ||
    base64.length === 0 ||
    base64.length > MAX_CHAT_IMAGE_BASE64_CHARS ||
    !BASE64_PATTERN.test(base64)
  ) {
    return null;
  }
  return `data:${normalizedMediaType};base64,${base64}`;
}

function iconButtonStyle(background: string, color: string): CSSProperties {
  return {
    width: 34,
    height: 34,
    border: "none",
    borderRadius: "var(--radius-sm)",
    background,
    color,
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    cursor: "pointer",
    flexShrink: 0,
  };
}
