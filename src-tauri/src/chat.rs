//! In-app chat commands (HANDOFF §3.3, P1). Thin Tauri surface over the
//! agent's [`ChatLoop`]: `chat_send` spawns a turn, streaming `chat_delta` /
//! `chat_tool_call` / `chat_done` events as the loop runs; `chat_history`
//! returns the current message log; `chat_cancel` stops a running turn.
//!
//! The chat loop shares the SAME dispatcher shape the MCP server uses: live
//! [`AppCore`] handle, the same workflow registry scan, the same media bridge,
//! and the same BYOK key-store boundary. That keeps tool availability and tool
//! behavior consistent between the panel and the external MCP surface.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use serde::Serialize;
use tauri::{AppHandle, Emitter, State};

use opentake_agent::chat::{
    next_message_id, AgentContentBlock, ChatLoop, ChatMessage, ChatSession, ChatSessionStore,
    ChatTurn, ChatTurnGate, EmitLoop, LlmError, LoopError, LoopEvent, Role, ToolCall,
};
use opentake_agent::mcp::advanced::AdvancedWorkflowBridge;
use opentake_agent::mcp::core_handle::{AppCoreHandle, CoreHandle};
use opentake_agent::mcp::dispatch::Dispatcher;
use opentake_agent::mcp::generation::GenerationBridge;
use opentake_agent::mcp::motion::MotionBridge;
use opentake_agent::plugin::registry::PluginRegistry;
use opentake_agent::tools::result::ToolResult;
use opentake_gen::{KeyStore, KeyringStore};

use opentake_core::AppCore;

/// Managed state: one [`ChatLoop`] over the shared core + plugin registry, a
/// map of live sessions, and a map of cancel flags for in-flight turns.
#[derive(Clone)]
pub struct ChatState {
    core: AppCore,
    dispatcher: Arc<Dispatcher>,
    registry: Arc<RwLock<PluginRegistry>>,
    loop_: ChatLoop,
    sessions: Arc<Mutex<HashMap<SessionKey, ChatSession>>>,
    turns: Arc<Mutex<TurnRegistry>>,
    persistence: Arc<Mutex<()>>,
    admission: crate::updater::InstallAdmissionGate,
}

/// Immutable handles for long-lived MCP sessions to enter the exact same
/// dispatcher and workflow registry used by in-app Agent chat.
#[derive(Clone)]
#[allow(dead_code)] // Task 4's listener consumes both handles.
pub(crate) struct ExternalMcpComponents {
    pub(crate) dispatcher: Arc<Dispatcher>,
    pub(crate) registry: Arc<RwLock<PluginRegistry>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct SessionKey {
    project_epoch: u64,
    project_dir: PathBuf,
    session_id: String,
}

fn agent_undo_scope(key: &SessionKey) -> String {
    format!(
        "opentake:chat:{}:{}:{}",
        key.project_epoch,
        key.project_dir.to_string_lossy(),
        key.session_id
    )
}

struct TurnCancel {
    requested: Arc<AtomicBool>,
    media: opentake_media::MediaCancelToken,
    phase: Mutex<TurnPhase>,
    completion: tokio::sync::watch::Sender<TurnCompletion>,
}

#[derive(Clone, Debug)]
enum TurnCompletion {
    Pending,
    Terminal(Result<Vec<ChatMessage>, String>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TurnPhase {
    Running,
    Cancelled,
    Finalizing,
}

impl TurnCancel {
    fn new() -> Self {
        let (completion, _) = tokio::sync::watch::channel(TurnCompletion::Pending);
        Self {
            requested: Arc::new(AtomicBool::new(false)),
            media: opentake_media::MediaCancelToken::new(),
            phase: Mutex::new(TurnPhase::Running),
            completion,
        }
    }

    fn request(&self) {
        let mut phase = self
            .phase
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if *phase != TurnPhase::Running {
            return;
        }
        *phase = TurnPhase::Cancelled;
        self.requested.store(true, Ordering::Release);
        self.media.cancel();
    }

    fn begin_finalization(&self) -> bool {
        let mut phase = self
            .phase
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if *phase != TurnPhase::Running {
            return false;
        }
        *phase = TurnPhase::Finalizing;
        true
    }

    fn complete(&self, result: Result<Vec<ChatMessage>, String>) {
        self.completion
            .send_replace(TurnCompletion::Terminal(result));
    }

    async fn wait_completion(&self) -> Result<Vec<ChatMessage>, String> {
        let mut completion = self.completion.subscribe();
        loop {
            if let TurnCompletion::Terminal(result) = &*completion.borrow() {
                return result.clone();
            }
            if completion.changed().await.is_err() {
                return Err("Agent turn completion channel closed".into());
            }
        }
    }
}

#[derive(Default)]
struct TurnRegistry {
    running: HashMap<SessionKey, Arc<TurnCancel>>,
    transition_depth: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TurnFinalization {
    Committed,
    Cancelled,
}

enum AuthoritativeHistoryState {
    Wait(Arc<TurnCancel>),
    Ready(Vec<ChatMessage>),
}

#[derive(Clone)]
struct ChatProjectContext {
    project_epoch: u64,
    project_dir: PathBuf,
    store: Arc<ChatSessionStore>,
}

impl ChatProjectContext {
    fn key(&self, session_id: &str) -> SessionKey {
        SessionKey {
            project_epoch: self.project_epoch,
            project_dir: self.project_dir.clone(),
            session_id: session_id.to_string(),
        }
    }
}

impl ChatState {
    pub(crate) fn external_mcp_components(&self) -> ExternalMcpComponents {
        ExternalMcpComponents {
            dispatcher: self.dispatcher.clone(),
            registry: self.registry.clone(),
        }
    }

    fn project_turn_gate(
        &self,
        project: &ChatProjectContext,
        session_key: &SessionKey,
        cancel: Arc<TurnCancel>,
    ) -> Arc<dyn ChatTurnGate> {
        Arc::new(ProjectTurnGate {
            state: self.clone(),
            project: project.clone(),
            cancel,
            undo_scope: agent_undo_scope(session_key),
        })
    }

    #[cfg(test)]
    pub(crate) fn project_turn_gate_for_test(&self, session_id: &str) -> Arc<dyn ChatTurnGate> {
        let project = self.project_context().expect("saved test project");
        self.put_project_session(&project, ChatSession::new(session_id))
            .expect("persist test chat session");
        let session_key = project.key(session_id);
        self.project_turn_gate(&project, &session_key, Arc::new(TurnCancel::new()))
    }

    #[cfg(test)]
    pub fn new(
        core: AppCore,
        workflows_dir: PathBuf,
        cache_root: PathBuf,
        models_dir: PathBuf,
    ) -> Self {
        Self::new_inner(
            core,
            workflows_dir,
            cache_root,
            models_dir,
            None,
            None,
            None,
            None,
            crate::updater::InstallAdmissionGate::default(),
        )
    }

    #[cfg(test)]
    fn new_with_admission(
        core: AppCore,
        workflows_dir: PathBuf,
        cache_root: PathBuf,
        models_dir: PathBuf,
        admission: crate::updater::InstallAdmissionGate,
    ) -> Self {
        Self::new_inner(
            core,
            workflows_dir,
            cache_root,
            models_dir,
            None,
            None,
            None,
            None,
            admission,
        )
    }

    /// Build the state in `setup`: a dispatcher over the live core + workflow
    /// registry + the same media bridge the desktop MCP server uses.
    #[allow(clippy::too_many_arguments)] // explicit dependency-injection boundary
    pub fn new_with_capabilities(
        core: AppCore,
        workflows_dir: PathBuf,
        cache_root: PathBuf,
        models_dir: PathBuf,
        generation_bridge: Arc<dyn GenerationBridge>,
        motion_bridge: Arc<dyn MotionBridge>,
        advanced_bridge: Arc<dyn AdvancedWorkflowBridge>,
        motion_document_notify: crate::mcp::MotionDocumentNotifier,
        admission: crate::updater::InstallAdmissionGate,
    ) -> Self {
        Self::new_inner(
            core,
            workflows_dir,
            cache_root,
            models_dir,
            Some(generation_bridge),
            Some(motion_bridge),
            Some(advanced_bridge),
            Some(motion_document_notify),
            admission,
        )
    }

    #[allow(clippy::too_many_arguments)] // keeps optional production bridges explicit in tests
    fn new_inner(
        core: AppCore,
        workflows_dir: PathBuf,
        cache_root: PathBuf,
        models_dir: PathBuf,
        generation_bridge: Option<Arc<dyn GenerationBridge>>,
        motion_bridge: Option<Arc<dyn MotionBridge>>,
        advanced_bridge: Option<Arc<dyn AdvancedWorkflowBridge>>,
        motion_document_notify: Option<crate::mcp::MotionDocumentNotifier>,
        admission: crate::updater::InstallAdmissionGate,
    ) -> Self {
        let handle: Arc<dyn CoreHandle> = Arc::new(AppCoreHandle::new(core.clone()));
        let registry = Arc::new(RwLock::new(crate::mcp::build_registry(&workflows_dir)));
        let bridge = crate::mcp::build_media_bridge(core.clone(), cache_root.clone(), models_dir);
        let motion_documents = crate::mcp::build_motion_document_bridge(
            core.clone(),
            cache_root,
            motion_document_notify,
        );
        let dispatcher = Arc::new(
            Dispatcher::with_all_capability_bridges(
                handle,
                registry.clone(),
                Some(bridge),
                generation_bridge,
                motion_bridge,
                advanced_bridge,
            )
            .with_motion_document_bridge(Some(motion_documents)),
        );
        let store: Arc<dyn KeyStore> = Arc::new(KeyringStore::new());
        let sessions = Arc::new(Mutex::new(HashMap::new()));
        let turns = Arc::new(Mutex::new(TurnRegistry::default()));
        let state = ChatState {
            core,
            dispatcher: dispatcher.clone(),
            registry: registry.clone(),
            loop_: ChatLoop::new(dispatcher, registry, store),
            sessions: sessions.clone(),
            turns: turns.clone(),
            persistence: Arc::new(Mutex::new(())),
            admission,
        };
        let transition_turns = turns.clone();
        state
            .core
            .subscribe_project_identity_transition(move |pending| {
                if let Ok(mut turns) = transition_turns.lock() {
                    if pending {
                        turns.transition_depth = turns.transition_depth.saturating_add(1);
                        for cancel in turns.running.values() {
                            cancel.request();
                        }
                    } else {
                        turns.transition_depth = turns.transition_depth.saturating_sub(1);
                    }
                }
            });
        state.core.subscribe(move |event| {
            let (project_epoch, project_dir) = match event {
                opentake_core::CoreEvent::ProjectOpened {
                    path,
                    project_epoch,
                    ..
                }
                | opentake_core::CoreEvent::ProjectSaved {
                    path,
                    project_epoch,
                } => (
                    *project_epoch,
                    (!path.is_empty()).then(|| PathBuf::from(path)),
                ),
                _ => return,
            };
            if let Ok(mut turns) = turns.lock() {
                turns.running.retain(|key, cancel| {
                    let current = key.project_epoch == project_epoch
                        && project_dir.as_ref() == Some(&key.project_dir);
                    if !current {
                        cancel.request();
                        cancel.complete(Err("stale Agent chat project identity".into()));
                    }
                    current
                });
            }
            if let Ok(mut cached) = sessions.lock() {
                cached.retain(|key, _| {
                    key.project_epoch == project_epoch
                        && project_dir.as_ref() == Some(&key.project_dir)
                });
            }
        });
        state
    }

    fn project_context(&self) -> Result<ChatProjectContext, String> {
        let _identity = self.core.lock_project_identity_workflow();
        let snapshot = self.core.runtime_snapshot();
        let project_dir = snapshot
            .project_dir
            .ok_or_else(|| "save the project before starting an Agent chat".to_string())?;
        let store = Arc::new(ChatSessionStore::open(&project_dir).map_err(|e| e.to_string())?);
        self.core
            .ensure_project_root_identity_for_project(
                snapshot.project_epoch,
                &project_dir,
                store.root().identity(),
            )
            .map_err(|e| e.to_string())?;
        Ok(ChatProjectContext {
            project_epoch: snapshot.project_epoch,
            project_dir,
            store,
        })
    }

    fn ensure_project_context(&self, project: &ChatProjectContext) -> Result<(), String> {
        self.core
            .ensure_project_root_identity_for_project(
                project.project_epoch,
                &project.project_dir,
                project.store.root().identity(),
            )
            .map_err(|e| e.to_string())
    }

    fn project_context_for(
        &self,
        expected_project_epoch: u64,
        expected_project_path: &str,
    ) -> Result<ChatProjectContext, String> {
        let project = self.project_context()?;
        if project.project_epoch != expected_project_epoch
            || project.project_dir.as_path() != std::path::Path::new(expected_project_path)
        {
            return Err("stale Agent chat project identity".to_string());
        }
        Ok(project)
    }

    /// Snapshot a session for a running turn. The map entry stays in place so
    /// `chat_history` can still return the last persisted state while the turn
    /// is in flight.
    fn take_project_session(
        &self,
        project: &ChatProjectContext,
        session_id: &str,
    ) -> Result<ChatSession, String> {
        let _identity = self.core.lock_project_identity_workflow();
        self.ensure_project_context(project)?;
        let key = project.key(session_id);
        if let Some(session) = self
            .sessions
            .lock()
            .map_err(|e| e.to_string())?
            .get(&key)
            .cloned()
        {
            return Ok(session);
        }
        Ok(project
            .store
            .load(session_id)
            .map_err(|e| e.to_string())?
            .unwrap_or_else(|| ChatSession::new(session_id.to_string())))
    }

    fn take_open_project_session_for_turn(
        &self,
        project: &ChatProjectContext,
        session_id: &str,
    ) -> Result<ChatSession, String> {
        let session = self.take_project_session(project, session_id)?;
        if session.is_open {
            Ok(session)
        } else {
            Err("the Agent chat tab is closed".into())
        }
    }

    /// Atomically persist a session only while its original project identity
    /// still owns the core. Stale turn completion cannot publish into A or B.
    fn put_project_session(
        &self,
        project: &ChatProjectContext,
        session: ChatSession,
    ) -> Result<(), String> {
        let _identity = self.core.lock_project_identity_workflow();
        self.put_project_session_with_identity_held(project, session)
    }

    fn put_project_session_with_identity_held(
        &self,
        project: &ChatProjectContext,
        session: ChatSession,
    ) -> Result<(), String> {
        self.ensure_project_context(project)?;
        let _persistence = self.persistence.lock().map_err(|e| e.to_string())?;
        project.store.save(&session).map_err(|e| e.to_string())?;
        let key = project.key(&session.id);
        self.sessions
            .lock()
            .map_err(|e| e.to_string())?
            .insert(key, session);
        Ok(())
    }

    /// Linearize whole-turn cancellation against the durable Codex terminal
    /// commit. Holding the turn registry excludes `chat_cancel` and project
    /// transition hooks; `TurnCancel::begin_finalization` covers cancellation
    /// sources that hold the token directly. The identity read lease then keeps
    /// the accepted project stable through the session write.
    fn finalize_project_turn(
        &self,
        project: &ChatProjectContext,
        key: &SessionKey,
        owner: &Arc<TurnCancel>,
        session: ChatSession,
    ) -> Result<TurnFinalization, String> {
        let turns = self.turns.lock().map_err(|e| e.to_string())?;
        let owns_turn = turns
            .running
            .get(key)
            .is_some_and(|registered| Arc::ptr_eq(registered, owner));
        if !owns_turn || turns.transition_depth > 0 || !owner.begin_finalization() {
            return Ok(TurnFinalization::Cancelled);
        }

        let _identity = self.core.lock_project_identity_workflow();
        let persisted = self.put_project_session_with_identity_held(project, session);
        persisted?;
        Ok(TurnFinalization::Committed)
    }

    fn list_project_sessions(
        &self,
        project: &ChatProjectContext,
    ) -> Result<Vec<ChatSession>, String> {
        let _identity = self.core.lock_project_identity_workflow();
        self.ensure_project_context(project)?;
        let _persistence = self.persistence.lock().map_err(|e| e.to_string())?;
        project.store.list().map_err(|e| e.to_string())
    }

    fn set_project_session_open(
        &self,
        project: &ChatProjectContext,
        session_id: &str,
        is_open: bool,
    ) -> Result<ChatSession, String> {
        let key = project.key(session_id);
        let turns = self.turns.lock().map_err(|e| e.to_string())?;
        if turns.running.contains_key(&key) {
            return Err("finish or cancel the running turn before changing its tab state".into());
        }
        let mut session = self.take_project_session(project, session_id)?;
        session.is_open = is_open;
        self.put_project_session(project, session.clone())?;
        drop(turns);
        Ok(session)
    }

    /// Permanently delete one conversation of the current project. A session
    /// with a running turn is refused. The file is removed before the cached
    /// copy, so a failed delete leaves memory and disk unchanged.
    fn delete_project_session(
        &self,
        project: &ChatProjectContext,
        session_id: &str,
    ) -> Result<(), String> {
        let key = project.key(session_id);
        let turns = self.turns.lock().map_err(|e| e.to_string())?;
        if turns.running.contains_key(&key) {
            return Err("finish or cancel the running turn before deleting this chat".into());
        }
        let _identity = self.core.lock_project_identity_workflow();
        self.ensure_project_context(project)?;
        let _persistence = self.persistence.lock().map_err(|e| e.to_string())?;
        project
            .store
            .delete(session_id)
            .map_err(|e| e.to_string())?;
        self.sessions
            .lock()
            .map_err(|e| e.to_string())?
            .remove(&key);
        drop(turns);
        Ok(())
    }

    fn reserve_turn(
        &self,
        key: SessionKey,
        cancel: Arc<TurnCancel>,
    ) -> Result<crate::updater::ActivityLease, String> {
        let admission = self.admission.begin_activity()?;
        let mut turns = self.turns.lock().map_err(|e| e.to_string())?;
        if turns.transition_depth > 0 {
            cancel.request();
            return Err("project transition is in progress".into());
        }
        match turns.running.entry(key) {
            Entry::Vacant(entry) => {
                entry.insert(cancel);
                Ok(admission)
            }
            Entry::Occupied(_) => Err("a turn is already running on this session".into()),
        }
    }

    /// Request that the running turn of `key` stop (`chat_cancel`).
    fn cancel_turn(&self, key: &SessionKey) -> Result<(), String> {
        let turns = self.turns.lock().map_err(|e| e.to_string())?;
        if let Some(flag) = turns.running.get(key) {
            flag.request();
        }
        Ok(())
    }

    fn release_turn(&self, key: &SessionKey) {
        if let Ok(mut turns) = self.turns.lock() {
            if let Some(owner) = turns.running.remove(key) {
                owner.complete(Err(
                    "Agent turn ended without a durable terminal snapshot".into()
                ));
            }
        }
    }

    fn complete_turn(&self, key: &SessionKey, owner: &Arc<TurnCancel>) {
        let terminal = self
            .sessions
            .lock()
            .map_err(|error| error.to_string())
            .and_then(|sessions| {
                sessions
                    .get(key)
                    .map(|session| session.messages.clone())
                    .ok_or_else(|| "Agent turn completed without persisted history".to_string())
            });
        if let Ok(mut turns) = self.turns.lock() {
            let owns_turn = turns
                .running
                .get(key)
                .is_some_and(|registered| Arc::ptr_eq(registered, owner));
            if owns_turn {
                turns.running.remove(key);
                owner.complete(terminal);
            }
        }
    }

    async fn authoritative_project_history(
        &self,
        expected_project_epoch: u64,
        expected_project_path: &str,
        session_id: &str,
    ) -> Result<Vec<ChatMessage>, String> {
        let project = self.project_context_for(expected_project_epoch, expected_project_path)?;
        let key = project.key(session_id);
        match self.authoritative_history_state(&project, &key, session_id)? {
            AuthoritativeHistoryState::Wait(owner) => {
                let messages = owner.wait_completion().await?;
                self.ensure_project_context(&project)?;
                Ok(messages)
            }
            AuthoritativeHistoryState::Ready(messages) => Ok(messages),
        }
    }

    fn authoritative_history_state(
        &self,
        project: &ChatProjectContext,
        key: &SessionKey,
        session_id: &str,
    ) -> Result<AuthoritativeHistoryState, String> {
        // Exclude a new turn only for this synchronous authoritative read. No
        // runtime mutex escapes this helper or crosses an async suspension.
        let turns = self.turns.lock().map_err(|e| e.to_string())?;
        if let Some(owner) = turns.running.get(key).cloned() {
            return Ok(AuthoritativeHistoryState::Wait(owner));
        }
        let messages = self.take_project_session(project, session_id)?.messages;
        Ok(AuthoritativeHistoryState::Ready(messages))
    }
}

// MARK: - Event payloads (camelCase, mirror front-end types.ts)

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct BlockDeltaPayload {
    project_epoch: u64,
    project_path: String,
    session_id: String,
    message_id: String,
    sequence: u64,
    block_index: usize,
    delta: String,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct BlockUpsertPayload {
    project_epoch: u64,
    project_path: String,
    session_id: String,
    message_id: String,
    sequence: u64,
    block_index: usize,
    block: AgentContentBlock,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call: Option<ToolCall>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct DonePayload {
    project_epoch: u64,
    project_path: String,
    session_id: String,
    message_id: String,
    sequence: u64,
    message: ChatMessage,
}

#[derive(Default)]
struct StreamSequenceGate {
    next_by_message: HashMap<String, u64>,
}

impl StreamSequenceGate {
    fn accept(&mut self, message_id: &str, sequence: u64) -> bool {
        let expected = self
            .next_by_message
            .entry(message_id.to_string())
            .or_insert(0);
        if sequence != *expected {
            return false;
        }
        *expected = expected.saturating_add(1);
        true
    }

    fn next(&self, message_id: &str) -> u64 {
        self.next_by_message.get(message_id).copied().unwrap_or(0)
    }
}

#[derive(Default)]
struct MessageEventSequence(u64);

impl MessageEventSequence {
    fn take(&mut self) -> u64 {
        let sequence = self.0;
        self.0 = self.0.saturating_add(1);
        sequence
    }

    fn next(&self) -> u64 {
        self.0
    }
}

fn terminal_message_is_allowed(expected_id: &str, message: &ChatMessage) -> bool {
    if message.id != expected_id {
        return false;
    }
    match message.role {
        opentake_agent::chat::Role::Assistant => {
            message.tool_call_id.is_none()
                && message.tool_is_error.is_none()
                && message.blocks.iter().all(|block| {
                    matches!(
                        block,
                        AgentContentBlock::Text { .. } | AgentContentBlock::ToolUse { .. }
                    )
                })
        }
        opentake_agent::chat::Role::Tool => {
            let Some(tool_call_id) = message.tool_call_id.as_deref() else {
                return false;
            };
            !tool_call_id.is_empty()
                && message.tool_calls.is_empty()
                && !message.blocks.is_empty()
                && message.blocks.iter().all(|block| {
                    matches!(
                        block,
                        AgentContentBlock::ToolResult {
                            tool_use_id,
                            is_error,
                            ..
                        } if tool_use_id == tool_call_id
                            && *is_error == message.tool_is_error
                    )
                })
        }
        opentake_agent::chat::Role::System | opentake_agent::chat::Role::User => false,
    }
}

/// Adapt `AppHandle::emit` to the loop's [`EmitLoop`] trait. Each loop event
/// becomes a Tauri event the front end listens for.
struct AppEmitter {
    app: AppHandle,
    state: ChatState,
    project: ChatProjectContext,
    sequences: Mutex<StreamSequenceGate>,
}

impl AppEmitter {
    fn accept_sequence(&self, message_id: &str, sequence: u64) -> bool {
        let accepted = self
            .sequences
            .lock()
            .map(|mut gate| gate.accept(message_id, sequence))
            .unwrap_or(false);
        accepted
    }

    fn next_sequence(&self, message_id: &str) -> u64 {
        self.sequences
            .lock()
            .map(|gate| gate.next(message_id))
            .unwrap_or(0)
    }
}

impl EmitLoop for AppEmitter {
    fn emit(&self, event: LoopEvent) {
        let _identity = self.state.core.lock_project_identity_workflow();
        if self.state.ensure_project_context(&self.project).is_err() {
            return;
        }
        match event {
            LoopEvent::BlockDelta {
                session_id,
                message_id,
                sequence,
                block_index,
                delta,
            } => {
                if !self.accept_sequence(&message_id, sequence) {
                    return;
                }
                let _ = self.app.emit(
                    "chat_delta",
                    BlockDeltaPayload {
                        project_epoch: self.project.project_epoch,
                        project_path: self.project.project_dir.to_string_lossy().into_owned(),
                        session_id,
                        message_id,
                        sequence,
                        block_index,
                        delta,
                    },
                );
            }
            LoopEvent::BlockUpsert {
                session_id,
                message_id,
                sequence,
                block_index,
                block,
            } => {
                if !self.accept_sequence(&message_id, sequence) {
                    return;
                }
                let _ = self.app.emit(
                    "chat_tool_call",
                    BlockUpsertPayload {
                        project_epoch: self.project.project_epoch,
                        project_path: self.project.project_dir.to_string_lossy().into_owned(),
                        session_id,
                        message_id,
                        sequence,
                        block_index,
                        tool_call: match &block {
                            AgentContentBlock::ToolUse {
                                id,
                                name,
                                input,
                                result,
                                is_error,
                            } => Some(ToolCall {
                                id: id.clone(),
                                name: name.clone(),
                                args: input.clone(),
                                result: result.clone(),
                                is_error: *is_error,
                            }),
                            _ => None,
                        },
                        block,
                    },
                );
            }
            LoopEvent::Done {
                session_id,
                message_id,
                sequence,
                message,
            } => {
                if !terminal_message_is_allowed(&message_id, &message) {
                    return;
                }
                if !self.accept_sequence(&message_id, sequence) {
                    return;
                }
                let _ = self.app.emit(
                    "chat_done",
                    DonePayload {
                        project_epoch: self.project.project_epoch,
                        project_path: self.project.project_dir.to_string_lossy().into_owned(),
                        session_id,
                        message_id,
                        sequence,
                        message,
                    },
                );
            }
        }
    }
}

/// Binds every Context Signal snapshot and complete tool dispatch to the
/// project accepted by `chat_send`. Admission runs under the identity read
/// lease; media-heavy tool bodies and captures run from `finish_dispatch`
/// after it is released, pinned to the admitted project, and a project
/// transition cancels them through the turn's token instead of waiting.
struct ProjectTurnGate {
    state: ChatState,
    project: ChatProjectContext,
    cancel: Arc<TurnCancel>,
    undo_scope: String,
}

impl ProjectTurnGate {
    fn with_current_project<T>(&self, operation: impl FnOnce() -> T) -> Option<T> {
        let _identity = self.state.core.lock_project_identity_workflow();
        if self.cancel.requested.load(Ordering::Acquire)
            || self.state.ensure_project_context(&self.project).is_err()
        {
            self.cancel.request();
            return None;
        }
        Some(operation())
    }
}

impl ChatTurnGate for ProjectTurnGate {
    fn timeline(&self, dispatcher: &Dispatcher) -> Option<opentake_domain::Timeline> {
        self.with_current_project(|| dispatcher.timeline())
    }

    fn dispatch(
        &self,
        dispatcher: &Dispatcher,
        name: &str,
        args: serde_json::Value,
    ) -> Option<ToolResult> {
        let receipt = self.with_current_project(|| {
            dispatcher.dispatch_cancellable_scoped_deferred(
                &self.undo_scope,
                name,
                args,
                &self.cancel.media,
            )
        })?;
        let result = dispatcher.finish_dispatch(receipt, &self.cancel.media);
        self.with_current_project(|| ())?;
        Some(result)
    }

    fn request_cancel(&self) {
        self.cancel.request();
    }

    fn request_dispatch_cancel(&self) {
        self.cancel.media.cancel();
    }
}

/// User-facing text for a failed official Codex turn. Only a provider-reported
/// turn failure (which covers expired sign-ins) points at the login status.
fn codex_turn_failure_message(error: crate::codex::CodexTurnError) -> &'static str {
    use crate::codex::CodexTurnError;
    match error {
        CodexTurnError::McpStart => {
            "the OpenTake tool server for official Codex stopped during the turn; try again"
        }
        CodexTurnError::Timeout => {
            "the official Codex turn took too long and was stopped; try a smaller request"
        }
        CodexTurnError::Protocol => {
            "OpenTake could not read the official Codex output; update Codex CLI and try again"
        }
        CodexTurnError::CliFailed => {
            "the official Codex CLI stopped unexpectedly; try again, and update Codex CLI if it keeps failing"
        }
        CodexTurnError::ProviderFailed => {
            "official Codex reported that the turn failed; check the Codex login status and try again"
        }
        CodexTurnError::Cancelled => "the official Codex turn was cancelled",
        CodexTurnError::Unavailable => "official Codex CLI was not found",
        CodexTurnError::IncompatibleCli | CodexTurnError::StrictConfigRejected => {
            "the installed official Codex CLI is not compatible with this OpenTake version"
        }
        CodexTurnError::NotAuthenticated => "official Codex is not signed in",
    }
}

fn truncate_for_codex_prompt(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars();
    let truncated = chars.by_ref().take(max_chars).collect::<String>();
    if chars.next().is_some() {
        format!("{truncated}…")
    } else {
        truncated
    }
}

fn codex_turn_prompt(session: &ChatSession, user_text: &str) -> String {
    let mut history = session
        .messages
        .iter()
        .rev()
        .skip(1)
        .filter_map(|message| match message.role {
            Role::User => Some(format!(
                "User: {}",
                truncate_for_codex_prompt(&message.content, 2_000)
            )),
            Role::Assistant if !message.content.trim().is_empty() => Some(format!(
                "Assistant: {}",
                truncate_for_codex_prompt(&message.content, 2_000)
            )),
            _ => None,
        })
        .take(6)
        .collect::<Vec<_>>();
    history.reverse();
    let history = history.join("\n");
    format!(
        "You are the official Codex Agent embedded in OpenTake, a desktop video editor. \
Use only the `opentake` MCP server for reading or changing the current project. \
Do not use shell commands, direct file edits, web search, plugins, or other MCP servers. \
Treat the current saved OpenTake project as the sole editing target. Explain the completed result concisely.\n\n\
Recent conversation:\n{history}\n\nCurrent user request:\n{}",
        truncate_for_codex_prompt(user_text, 20_000)
    )
}

// MARK: - Commands

/// `chat_send`: spawn a chat turn. Returns immediately; the turn streams via
/// `chat_delta` / `chat_tool_call` / `chat_done` events.
#[tauri::command]
pub async fn chat_send(
    app: AppHandle,
    state: State<'_, ChatState>,
    session_id: String,
    text: String,
    chat_provider: String,
    expected_project_epoch: u64,
    expected_project_path: String,
) -> Result<(), String> {
    let project = state.project_context_for(expected_project_epoch, &expected_project_path)?;
    let session_key = project.key(&session_id);
    let turn_cancel = Arc::new(TurnCancel::new());
    let turn_admission = state.reserve_turn(session_key.clone(), turn_cancel.clone())?;
    let cancel = turn_cancel.requested.clone();

    let mut session = match state.take_open_project_session_for_turn(&project, &session_id) {
        Ok(session) => session,
        Err(error) => {
            state.release_turn(&session_key);
            return Err(error);
        }
    };
    session.provider = Some(chat_provider.clone());
    session.messages.push(ChatMessage::user(text.clone()));
    if let Err(error) = state.put_project_session(&project, session.clone()) {
        state.release_turn(&session_key);
        return Err(error);
    }
    let state_clone = state.inner().clone();
    let sid = session_id.clone();
    let first_message_id = next_message_id();

    tauri::async_runtime::spawn(async move {
        let _turn_admission = turn_admission;
        let emitter = AppEmitter {
            app: app.clone(),
            state: state_clone.clone(),
            project: project.clone(),
            sequences: Mutex::new(StreamSequenceGate::default()),
        };
        let turn_owner = turn_cancel.clone();
        let gate = state_clone.project_turn_gate(&project, &session_key, turn_cancel);
        let is_codex = chat_provider == "codex";
        let mut codex_final: Option<ChatMessage> = None;
        let mut codex_sequence = MessageEventSequence::default();
        let result = if is_codex {
            session.provider = Some("codex".into());
            session.model = Some("official-codex-default".into());
            let prompt = codex_turn_prompt(&session, &text);
            let context = crate::codex::CodexTurnContext {
                dispatcher: state_clone.dispatcher.clone(),
                registry: state_clone.registry.clone(),
                gate: gate.clone(),
                cancel: cancel.clone(),
            };
            let mut draft = ChatMessage::assistant_blocks_with_id(&first_message_id, Vec::new());
            match crate::codex::run_agent_turn(context, &prompt, |tool_call| {
                let block_index = draft.upsert_tool_use(tool_call);
                if let Some(block) = draft.blocks.get(block_index).cloned() {
                    emitter.emit(LoopEvent::BlockUpsert {
                        session_id: sid.clone(),
                        message_id: first_message_id.clone(),
                        sequence: codex_sequence.take(),
                        block_index,
                        block,
                    });
                }
            })
            .await
            {
                Ok(output) => {
                    for tool_call in output.tool_calls {
                        let block_index = draft.upsert_tool_use(tool_call);
                        if let Some(block) = draft.blocks.get(block_index).cloned() {
                            emitter.emit(LoopEvent::BlockUpsert {
                                session_id: sid.clone(),
                                message_id: first_message_id.clone(),
                                sequence: codex_sequence.take(),
                                block_index,
                                block,
                            });
                        }
                    }
                    let block_index = draft.append_text_delta(&output.text);
                    session.messages.push(draft.clone());
                    emitter.emit(LoopEvent::BlockDelta {
                        session_id: sid.clone(),
                        message_id: first_message_id.clone(),
                        sequence: codex_sequence.take(),
                        block_index,
                        delta: output.text,
                    });
                    codex_final = Some(draft);
                    Ok(first_message_id.clone())
                }
                Err(crate::codex::CodexTurnError::Cancelled) => Err(LoopError::cancelled(
                    &first_message_id,
                    codex_sequence.next(),
                )),
                Err(crate::codex::CodexTurnError::Unavailable) => {
                    let guide = "Official Codex CLI was not found. Install Codex, then return to Settings → AI and choose Official Codex / ChatGPT.".to_string();
                    let message =
                        ChatMessage::assistant_with_id(&first_message_id, &guide, Vec::new());
                    session.messages.push(message.clone());
                    emitter.emit(LoopEvent::BlockDelta {
                        session_id: sid.clone(),
                        message_id: first_message_id.clone(),
                        sequence: codex_sequence.take(),
                        block_index: 0,
                        delta: guide,
                    });
                    codex_final = Some(message);
                    Ok(first_message_id.clone())
                }
                Err(crate::codex::CodexTurnError::NotAuthenticated) => {
                    let guide = "Codex is not signed in. Open Settings → AI, choose Official Codex / ChatGPT, and sign in with ChatGPT.".to_string();
                    let message =
                        ChatMessage::assistant_with_id(&first_message_id, &guide, Vec::new());
                    session.messages.push(message.clone());
                    emitter.emit(LoopEvent::BlockDelta {
                        session_id: sid.clone(),
                        message_id: first_message_id.clone(),
                        sequence: codex_sequence.take(),
                        block_index: 0,
                        delta: guide,
                    });
                    codex_final = Some(message);
                    Ok(first_message_id.clone())
                }
                Err(crate::codex::CodexTurnError::IncompatibleCli)
                | Err(crate::codex::CodexTurnError::StrictConfigRejected) => {
                    let guide = "The installed official Codex CLI is not compatible with this OpenTake Beta. Update Codex CLI to version 0.146.0 or newer, then try again.".to_string();
                    let message =
                        ChatMessage::assistant_with_id(&first_message_id, &guide, Vec::new());
                    session.messages.push(message.clone());
                    emitter.emit(LoopEvent::BlockDelta {
                        session_id: sid.clone(),
                        message_id: first_message_id.clone(),
                        sequence: codex_sequence.take(),
                        block_index: 0,
                        delta: guide,
                    });
                    codex_final = Some(message);
                    Ok(first_message_id.clone())
                }
                Err(
                    error @ (crate::codex::CodexTurnError::McpStart
                    | crate::codex::CodexTurnError::Timeout
                    | crate::codex::CodexTurnError::Protocol
                    | crate::codex::CodexTurnError::ProviderFailed
                    | crate::codex::CodexTurnError::CliFailed),
                ) => Err(LoopError::llm(
                    LlmError::Provider(codex_turn_failure_message(error).into()),
                    &first_message_id,
                    codex_sequence.next(),
                )),
            }
        } else {
            state_clone
                .loop_
                .run_turn_gated(
                    &mut session,
                    chat_provider,
                    text,
                    ChatTurn {
                        first_message_id: first_message_id.clone(),
                        cancel,
                        gate,
                    },
                    &emitter,
                )
                .await
        };

        if is_codex {
            let (terminal, terminal_sequence) = match result {
                Ok(_) => (codex_final, codex_sequence.next()),
                Err(LoopError::Cancelled {
                    message_id,
                    sequence,
                }) => (
                    Some(ChatMessage::assistant_with_id(
                        message_id,
                        String::new(),
                        Vec::new(),
                    )),
                    sequence,
                ),
                Err(error) => {
                    let sequence = error.sequence();
                    let message = ChatMessage::assistant_with_id(
                        error.message_id(),
                        format!("⚠️ {error}"),
                        Vec::new(),
                    );
                    session.messages.push(message.clone());
                    (Some(message), sequence)
                }
            };
            match state_clone.finalize_project_turn(
                &project,
                &session_key,
                &turn_owner,
                session.clone(),
            ) {
                Ok(TurnFinalization::Committed) => {
                    if let Some(message) = terminal {
                        emitter.emit(LoopEvent::Done {
                            session_id: sid.clone(),
                            message_id: message.id.clone(),
                            sequence: terminal_sequence,
                            message,
                        });
                    }
                }
                Ok(TurnFinalization::Cancelled) => {
                    let message = terminal.unwrap_or_else(|| {
                        ChatMessage::assistant_with_id(&first_message_id, String::new(), Vec::new())
                    });
                    emitter.emit(LoopEvent::Done {
                        session_id: sid.clone(),
                        message_id: message.id.clone(),
                        sequence: terminal_sequence,
                        message,
                    });
                }
                Err(error) => {
                    let message_id = terminal
                        .as_ref()
                        .map(|message| message.id.as_str())
                        .unwrap_or(&first_message_id);
                    let message = ChatMessage::assistant_with_id(
                        message_id,
                        format!("⚠️ Chat history could not be saved: {error}"),
                        Vec::new(),
                    );
                    emitter.emit(LoopEvent::Done {
                        session_id: sid.clone(),
                        message_id: message.id.clone(),
                        sequence: terminal_sequence,
                        message,
                    });
                }
            }
            state_clone.complete_turn(&session_key, &turn_owner);
            return;
        }

        match &result {
            Err(LoopError::Cancelled {
                message_id,
                sequence,
            }) => {
                let message = ChatMessage::assistant_with_id(message_id, String::new(), Vec::new());
                emitter.emit(LoopEvent::Done {
                    session_id: sid.clone(),
                    message_id: message.id.clone(),
                    sequence: *sequence,
                    message,
                });
            }
            Err(e) => {
                let msg =
                    ChatMessage::assistant_with_id(e.message_id(), format!("⚠️ {e}"), Vec::new());
                emitter.emit(LoopEvent::Done {
                    session_id: sid.clone(),
                    message_id: msg.id.clone(),
                    sequence: e.sequence(),
                    message: msg.clone(),
                });
                session.messages.push(msg);
            }
            Ok(_) => {}
        }

        if let Err(error) = state_clone.put_project_session(&project, session) {
            let message_id = match &result {
                Ok(message_id) => message_id.as_str(),
                Err(error) => error.message_id(),
            };
            let message = ChatMessage::assistant_with_id(
                message_id,
                format!("⚠️ Chat history could not be saved: {error}"),
                Vec::new(),
            );
            emitter.emit(LoopEvent::Done {
                session_id: sid.clone(),
                message_id: message.id.clone(),
                sequence: emitter.next_sequence(&message.id),
                message,
            });
        }
        state_clone.complete_turn(&session_key, &turn_owner);
    });

    Ok(())
}

/// `chat_history`: return the current message log for a session. Empty when
/// the session doesn't exist yet.
#[tauri::command]
pub fn chat_history(
    state: State<'_, ChatState>,
    session_id: String,
    expected_project_epoch: u64,
    expected_project_path: String,
) -> Result<Vec<ChatMessage>, String> {
    let project = state.project_context_for(expected_project_epoch, &expected_project_path)?;
    Ok(state.take_project_session(&project, &session_id)?.messages)
}

/// Return only a terminal durable snapshot. If this exact project/session has
/// an active turn, wait for its terminal event boundary without holding the
/// turn registry mutex across the async suspension.
#[tauri::command]
pub async fn chat_history_authoritative(
    state: State<'_, ChatState>,
    session_id: String,
    expected_project_epoch: u64,
    expected_project_path: String,
) -> Result<Vec<ChatMessage>, String> {
    state
        .inner()
        .clone()
        .authoritative_project_history(expected_project_epoch, &expected_project_path, &session_id)
        .await
}

/// `chat_sessions`: newest-first persistent conversations for the project.
#[tauri::command]
pub fn chat_sessions(
    state: State<'_, ChatState>,
    expected_project_epoch: u64,
    expected_project_path: String,
) -> Result<Vec<ChatSession>, String> {
    let project = state.project_context_for(expected_project_epoch, &expected_project_path)?;
    state.list_project_sessions(&project)
}

/// Persist whether a conversation is represented by an open Agent tab. Opening
/// a new id creates an empty project-local session immediately, so tabs survive
/// restart even before their first message.
#[tauri::command]
pub fn chat_session_set_open(
    state: State<'_, ChatState>,
    session_id: String,
    is_open: bool,
    expected_project_epoch: u64,
    expected_project_path: String,
) -> Result<ChatSession, String> {
    let _activity = crate::updater::begin_mutating_activity(&state.admission)?;
    let project = state.project_context_for(expected_project_epoch, &expected_project_path)?;
    state.set_project_session_open(&project, &session_id, is_open)
}

/// `chat_session_delete`: permanently delete a conversation of the expected
/// project. Refused while that conversation has a running turn.
#[tauri::command]
pub async fn chat_session_delete(
    state: State<'_, ChatState>,
    session_id: String,
    expected_project_epoch: u64,
    expected_project_path: String,
) -> Result<(), String> {
    let _activity = crate::updater::begin_mutating_activity(&state.admission)?;
    let state = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let project = state.project_context_for(expected_project_epoch, &expected_project_path)?;
        state.delete_project_session(&project, &session_id)
    })
    .await
    .map_err(|error| error.to_string())?
}

/// `chat_cancel`: request a running turn stop at the next boundary. No-op when
/// no turn is running.
#[tauri::command]
pub fn chat_cancel(
    state: State<'_, ChatState>,
    session_id: String,
    expected_project_epoch: u64,
    expected_project_path: String,
) -> Result<(), String> {
    state.cancel_turn(&SessionKey {
        project_epoch: expected_project_epoch,
        project_dir: PathBuf::from(expected_project_path),
        session_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codex_turn_failures_have_distinct_messages_and_only_provider_failure_blames_login() {
        use crate::codex::CodexTurnError;
        let failures = [
            CodexTurnError::McpStart,
            CodexTurnError::Timeout,
            CodexTurnError::Protocol,
            CodexTurnError::ProviderFailed,
            CodexTurnError::CliFailed,
        ];
        let messages = failures.map(codex_turn_failure_message);
        for (index, message) in messages.iter().enumerate() {
            assert_eq!(
                messages.iter().filter(|other| other == &message).count(),
                1,
                "{message}"
            );
            assert_eq!(
                message.contains("login"),
                failures[index] == CodexTurnError::ProviderFailed,
                "{message}"
            );
        }
    }

    struct RedactionMediaBridge;

    impl opentake_agent::mcp::media_bridge::MediaBridge for RedactionMediaBridge {
        fn inspect_media(
            &self,
            request: &opentake_agent::mcp::media_bridge::InspectMediaRequest,
        ) -> Result<
            opentake_agent::mcp::media_bridge::InspectMediaResult,
            opentake_agent::mcp::media_bridge::BridgeError,
        > {
            Ok(opentake_agent::mcp::media_bridge::InspectMediaResult {
                frames: vec![opentake_agent::mcp::media_bridge::InspectedMediaFrame {
                    timestamp_seconds: 0.0,
                    bytes: vec![0xff, 0xd8, 0xff, 0xe0],
                    media_type: "image/jpeg".into(),
                }],
                overview_timestamps: Vec::new(),
                duration_seconds: 3.0,
                width: Some(1920),
                height: Some(1080),
                fps: Some(30.0),
                has_audio: request.kind == opentake_domain::ClipType::Video,
                byte_size: 4,
                transcript: None,
                transcription_unavailable: false,
            })
        }
    }

    struct BlockingTimelineResultBridge {
        capture_started: std::sync::mpsc::Sender<()>,
        release_capture: Mutex<std::sync::mpsc::Receiver<()>>,
    }

    impl opentake_agent::mcp::media_bridge::MediaBridge for BlockingTimelineResultBridge {
        fn visible_timeline_clip_count(
            &self,
            timeline: &opentake_domain::Timeline,
        ) -> Result<usize, opentake_agent::mcp::media_bridge::BridgeError> {
            Ok(timeline
                .tracks
                .iter()
                .flat_map(|track| &track.clips)
                .filter(|clip| clip.duration_frames > 0)
                .count())
        }

        fn capture_timeline_result(
            &self,
            _request: &opentake_agent::mcp::media_bridge::TimelineResultCaptureRequest,
            _cancel: &opentake_media::MediaCancelToken,
        ) -> Result<
            opentake_agent::tools::result::Block,
            opentake_agent::mcp::media_bridge::BridgeError,
        > {
            self.capture_started.send(()).unwrap();
            // Longer than the test's save window, so the test releases the
            // capture before this gives up.
            self.release_capture
                .lock()
                .unwrap()
                .recv_timeout(std::time::Duration::from_secs(10))
                .expect("test must release the blocked timeline capture");
            Ok(opentake_agent::tools::result::Block::image(
                "iVBORw0KGgo=",
                "image/png",
            ))
        }
    }

    struct RedactionAdvancedBridge;

    const MOTION_PRIVATE_RENDERER: &str = "PRIVATE_CHAT_MOTION_RENDERER";
    const MOTION_PRIVATE_VERSION: &str = "PRIVATE_CHAT_MOTION_RENDERER_VERSION";
    const MOTION_PRIVATE_OUTPUT: &str = "/Users/private/chat-motion-output.mp4";
    const MOTION_PRIVATE_HASH: &str = "PRIVATE_CHAT_MOTION_CONTENT_HASH";
    const MOTION_PRIVATE_ACTION: &str = "PRIVATE_CHAT_MOTION_ACTION";

    impl opentake_agent::mcp::advanced::AdvancedWorkflowBridge for RedactionAdvancedBridge {
        fn supported_tools(&self) -> Vec<opentake_agent::tools::names::ToolName> {
            vec![opentake_agent::tools::names::ToolName::GenerateAvatar]
        }

        fn execute(
            &self,
            _request: opentake_agent::mcp::advanced::AdvancedWorkflowRequest,
            _cancel: &opentake_media::MediaCancelToken,
        ) -> Result<
            opentake_agent::mcp::advanced::AdvancedWorkflowCommit,
            opentake_agent::mcp::advanced::AdvancedWorkflowError,
        > {
            Ok(opentake_agent::mcp::advanced::AdvancedWorkflowCommit {
                result: serde_json::json!({
                    "status": "completed",
                    "assetId": "avatar-safe-asset",
                    "imported": true,
                    "previewPath": "/Users/private/avatar-preview.mov",
                    "signedUrl": "https://provider.invalid/avatar?token=SIGNED_AVATAR_SECRET",
                    "providerRequestId": "PRIVATE_PROVIDER_REQUEST_ID",
                    "prompt": "PRIVATE_ADVANCED_PROMPT",
                    "errors": [{"message": "provider failed with sk-private-avatar-key"}],
                }),
                action_name: None,
            })
        }
    }

    struct RedactionMotionBridge;

    fn redaction_motion_commit(
        clip_id: String,
        asset_id: &str,
    ) -> opentake_agent::mcp::motion::MotionCommit {
        opentake_agent::mcp::motion::MotionCommit {
            clip_id,
            asset_id: asset_id.into(),
            content_hash: MOTION_PRIVATE_HASH.into(),
            action_name: MOTION_PRIVATE_ACTION.into(),
            output: opentake_agent::mcp::motion::MotionOutputMetadata {
                renderer: MOTION_PRIVATE_RENDERER.into(),
                renderer_version: MOTION_PRIVATE_VERSION.into(),
                output_file: MOTION_PRIVATE_OUTPUT.into(),
                fps: 30.0,
                width: 1920,
                height: 1080,
                duration_frames: 90,
                duration_seconds: 3.0,
                content_hash: MOTION_PRIVATE_HASH.into(),
            },
            source_document: None,
        }
    }

    impl opentake_agent::mcp::motion::MotionBridge for RedactionMotionBridge {
        fn can_render_motion(&self) -> bool {
            true
        }

        fn add(
            &self,
            _request: opentake_agent::mcp::motion::AddMotionRequest,
            _cancel: &opentake_media::MediaCancelToken,
        ) -> Result<
            opentake_agent::mcp::motion::MotionCommit,
            opentake_agent::mcp::motion::MotionBridgeError,
        > {
            Ok(redaction_motion_commit(
                "motion-safe-add-clip".into(),
                "motion-safe-add-asset",
            ))
        }

        fn edit(
            &self,
            request: opentake_agent::mcp::motion::EditMotionRequest,
            _cancel: &opentake_media::MediaCancelToken,
        ) -> Result<
            opentake_agent::mcp::motion::MotionCommit,
            opentake_agent::mcp::motion::MotionBridgeError,
        > {
            Ok(redaction_motion_commit(
                request.clip_id,
                "motion-safe-edit-asset",
            ))
        }
    }

    #[test]
    fn take_and_put_session_round_trips() {
        let temp = tempfile::tempdir().unwrap();
        let core = AppCore::new();
        core.save_project(Some(temp.path().join("RoundTrip.opentake")))
            .unwrap();
        let state = ChatState::new(
            core,
            temp.path().join("no-workflows"),
            temp.path().join("chat-cache"),
            temp.path().join("chat-models"),
        );
        let project = state.project_context().unwrap();
        state
            .put_project_session(&project, ChatSession::new("s1"))
            .unwrap();
        let taken = state.take_project_session(&project, "s1").unwrap();
        assert_eq!(taken.id, "s1");
        let second = state.take_project_session(&project, "s1").unwrap();
        assert_eq!(second.id, "s1");
        assert!(second.messages.is_empty());
    }

    #[test]
    fn history_snapshot_remains_visible_while_turn_owns_a_clone() {
        let temp = tempfile::tempdir().unwrap();
        let core = AppCore::new();
        core.save_project(Some(temp.path().join("Snapshot.opentake")))
            .unwrap();
        let state = ChatState::new(
            core,
            temp.path().join("no-workflows"),
            temp.path().join("chat-cache"),
            temp.path().join("chat-models"),
        );
        let project = state.project_context().unwrap();
        let mut session = ChatSession::new("s1");
        session.messages.push(ChatMessage::user("hello"));
        state
            .put_project_session(&project, session.clone())
            .unwrap();

        let running_copy = state.take_project_session(&project, "s1").unwrap();
        assert_eq!(running_copy.messages.len(), 1);

        let key = project.key("s1");
        let history = state
            .sessions
            .lock()
            .unwrap()
            .get(&key)
            .cloned()
            .unwrap()
            .messages;
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].content, "hello");
    }

    #[test]
    fn event_payloads_serialize_block_addresses_in_camel_case() {
        let payload = BlockDeltaPayload {
            project_epoch: 7,
            project_path: "/tmp/A.opentake".into(),
            session_id: "sess-1".into(),
            message_id: "assistant-1".into(),
            sequence: 0,
            block_index: 2,
            delta: "hi".into(),
        };
        let json = serde_json::to_value(payload).unwrap();
        assert_eq!(json["projectEpoch"], 7);
        assert_eq!(json["projectPath"], "/tmp/A.opentake");
        assert_eq!(json["sessionId"], "sess-1");
        assert_eq!(json["messageId"], "assistant-1");
        assert_eq!(json["sequence"], 0);
        assert_eq!(json["blockIndex"], 2);
        assert_eq!(json["delta"], "hi");

        let message = ChatMessage::assistant_with_id("assistant-1", "done", Vec::new());
        let done = DonePayload {
            project_epoch: 7,
            project_path: "/tmp/A.opentake".into(),
            session_id: "sess-1".into(),
            message_id: message.id.clone(),
            sequence: 1,
            message,
        };
        let json = serde_json::to_value(done).unwrap();
        assert_eq!(json["sessionId"], "sess-1");
        assert_eq!(json["messageId"], "assistant-1");
        assert_eq!(json["sequence"], 1);
        assert_eq!(json["message"]["role"], "assistant");
        assert_eq!(json["message"]["id"], json["messageId"]);
    }

    #[test]
    fn terminal_contract_allows_only_matching_nonempty_tool_result_messages() {
        let assistant = ChatMessage::assistant_with_id("assistant-1", "done", Vec::new());
        assert!(terminal_message_is_allowed("assistant-1", &assistant));

        let tool = ChatMessage::tool_result("call-1", serde_json::json!({"summary": "ok"}));
        assert!(terminal_message_is_allowed(&tool.id, &tool));

        let mut empty = tool.clone();
        empty.blocks.clear();
        assert!(!terminal_message_is_allowed(&empty.id, &empty));

        let mut wrong_block = tool.clone();
        wrong_block.blocks = vec![AgentContentBlock::Text { text: "ok".into() }];
        assert!(!terminal_message_is_allowed(&wrong_block.id, &wrong_block));

        let mut mismatched = tool.clone();
        mismatched.tool_call_id = Some("call-other".into());
        assert!(!terminal_message_is_allowed(&mismatched.id, &mismatched));

        let mut mismatched_error = tool.clone();
        mismatched_error.tool_is_error = Some(true);
        assert!(!terminal_message_is_allowed(
            &mismatched_error.id,
            &mismatched_error,
        ));

        let mut assistant_with_tool_result = assistant;
        assistant_with_tool_result.blocks = tool.blocks;
        assert!(!terminal_message_is_allowed(
            &assistant_with_tool_result.id,
            &assistant_with_tool_result,
        ));
    }

    #[test]
    fn tool_done_payload_preserves_terminal_identity_and_sequence() {
        let message = ChatMessage::tool_result("call-1", serde_json::json!({"summary": "ok"}));
        let payload = DonePayload {
            project_epoch: 7,
            project_path: "/tmp/A.opentake".into(),
            session_id: "sess-1".into(),
            message_id: message.id.clone(),
            sequence: 1,
            message,
        };

        let json = serde_json::to_value(payload).unwrap();
        assert_eq!(json["sequence"], 1);
        assert_eq!(json["messageId"], json["message"]["id"]);
        assert_eq!(json["message"]["role"], "tool");
        assert_eq!(json["message"]["toolCallId"], "call-1");
        assert_eq!(json["message"]["blocks"][0]["type"], "toolResult");
        assert_eq!(json["message"]["blocks"][0]["toolUseId"], "call-1");
    }

    #[test]
    fn stream_sequence_gate_rejects_duplicates_and_gaps_per_message() {
        let mut gate = StreamSequenceGate::default();

        assert!(gate.accept("message-a", 0));
        assert!(!gate.accept("message-a", 0));
        assert!(!gate.accept("message-a", 2));
        assert!(gate.accept("message-a", 1));
        assert!(gate.accept("message-b", 0));
    }

    #[test]
    fn block_upsert_payload_retains_beta4_tool_call_decoder_fields() {
        let block = AgentContentBlock::ToolUse {
            id: "call-1".into(),
            name: "split_clip".into(),
            input: serde_json::json!({"clipId": "c1"}),
            result: Some(serde_json::json!({"ok": true})),
            is_error: Some(false),
        };
        let payload = BlockUpsertPayload {
            project_epoch: 7,
            project_path: "/tmp/A.opentake".into(),
            session_id: "sess-1".into(),
            message_id: "assistant-1".into(),
            sequence: 3,
            block_index: 1,
            tool_call: Some(ToolCall {
                id: "call-1".into(),
                name: "split_clip".into(),
                args: serde_json::json!({"clipId": "c1"}),
                result: Some(serde_json::json!({"ok": true})),
                is_error: Some(false),
            }),
            block: block.clone(),
        };

        let wire = serde_json::to_value(payload).unwrap();

        assert_eq!(wire["messageId"], "assistant-1");
        assert_eq!(wire["sequence"], 3);
        assert_eq!(wire["blockIndex"], 1);
        assert_eq!(wire["block"], serde_json::to_value(block).unwrap());
        assert_eq!(wire["toolCall"]["id"], "call-1");
        assert_eq!(wire["toolCall"]["args"]["clipId"], "c1");
    }

    #[test]
    fn codex_prompt_keeps_recent_context_and_limits_unbounded_history() {
        let mut session = ChatSession::new("codex-prompt");
        session.messages.push(ChatMessage::user("earlier request"));
        session
            .messages
            .push(ChatMessage::assistant("earlier result", Vec::new()));
        session.messages.push(ChatMessage::user("x".repeat(30_000)));

        let prompt = codex_turn_prompt(&session, &"当前请求".repeat(10_000));
        assert!(prompt.contains("User: earlier request"));
        assert!(prompt.contains("Assistant: earlier result"));
        assert!(prompt.contains("Current user request:"));
        assert!(prompt.chars().count() < 21_000);
        assert!(prompt.ends_with('…'));
    }

    #[test]
    fn project_chat_session_reloads_from_the_bundle_after_state_recreation() {
        let temp = tempfile::tempdir().unwrap();
        let bundle = temp.path().join("Persist.opentake");
        let core = AppCore::new();
        core.save_project(Some(bundle.clone())).unwrap();
        let state = ChatState::new(
            core.clone(),
            temp.path().join("no-workflows"),
            temp.path().join("chat-cache"),
            temp.path().join("chat-models"),
        );
        let project = state.project_context().unwrap();
        let mut session = ChatSession::new("chat-persisted");
        session.messages.push(ChatMessage::user("remember me"));
        state.put_project_session(&project, session).unwrap();
        drop(state);

        let reopened = ChatState::new(
            core,
            temp.path().join("no-workflows"),
            temp.path().join("chat-cache"),
            temp.path().join("chat-models"),
        );
        let project = reopened.project_context().unwrap();
        let loaded = reopened
            .take_project_session(&project, "chat-persisted")
            .unwrap();
        assert_eq!(loaded.messages.len(), 1);
        assert_eq!(loaded.messages[0].content, "remember me");
        assert_eq!(reopened.list_project_sessions(&project).unwrap().len(), 1);
    }

    #[test]
    fn sensitive_media_and_advanced_results_never_reach_persisted_chat_history() {
        use opentake_agent::mcp::convert::safe_tool_result_for_llm;
        use opentake_domain::{
            ClipType, GenerationInput, GenerationJobStatus, MediaManifestEntry, MediaSource,
        };

        const ABSOLUTE_PATH: &str = "/Users/private/launch-cut.mov";
        const SIGNED_RESULT_URL: &str =
            "https://cdn.example.invalid/result.mov?token=SIGNED_RESULT_SECRET";
        const SIGNED_REFERENCE_URL: &str =
            "https://cdn.example.invalid/reference.png?token=SIGNED_REFERENCE_SECRET";
        const SIGNED_IMAGE_URL: &str =
            "https://cdn.example.invalid/input.png?token=SIGNED_IMAGE_SECRET";
        const PRIVATE_PROMPT: &str = "PRIVATE_PROVIDER_PROMPT_SECRET";
        const ADVANCED_PATH: &str = "/Users/private/avatar-preview.mov";
        const ADVANCED_URL_SECRET: &str = "SIGNED_AVATAR_SECRET";
        const ADVANCED_PROVIDER_ID: &str = "PRIVATE_PROVIDER_REQUEST_ID";
        const ADVANCED_PROMPT: &str = "PRIVATE_ADVANCED_PROMPT";
        const ADVANCED_KEY: &str = "sk-private-avatar-key";

        let temp = tempfile::tempdir().unwrap();
        let bundle = temp.path().join("Redaction.opentake");
        let mut project = opentake_project::Project::new(bundle.clone());
        project.manifest.entries.push(MediaManifestEntry {
            id: "generated-asset".into(),
            name: "Generated clip".into(),
            kind: ClipType::Video,
            source: MediaSource::External {
                absolute_path: ABSOLUTE_PATH.into(),
            },
            duration: 3.0,
            generation_input: Some(GenerationInput {
                prompt: PRIVATE_PROMPT.into(),
                model: "provider-model".into(),
                duration: 3,
                aspect_ratio: "16:9".into(),
                image_urls: Some(vec![SIGNED_IMAGE_URL.into()]),
                reference_image_urls: Some(vec![SIGNED_REFERENCE_URL.into()]),
                status: Some(GenerationJobStatus::Ready),
                ..GenerationInput::default()
            }),
            source_width: Some(1920),
            source_height: Some(1080),
            source_fps: Some(30.0),
            has_audio: Some(true),
            color: None,
            proxy: None,
            folder_id: None,
            cached_remote_url: Some(SIGNED_RESULT_URL.into()),
            cached_remote_url_expires_at: Some(123_456.0),
        });
        project.save().unwrap();

        let core = AppCore::new();
        core.open_project(bundle.clone()).unwrap();
        let state = ChatState::new(
            core,
            temp.path().join("no-workflows"),
            temp.path().join("chat-cache"),
            temp.path().join("chat-models"),
        );
        let project = state.project_context().unwrap();
        let result = state
            .dispatcher
            .dispatch("get_media", serde_json::json!({}));
        assert!(!result.is_error, "{}", result.text_joined());

        let inspect_dispatcher = Dispatcher::with_bridge(
            Arc::new(AppCoreHandle::new(state.core.clone())),
            Arc::new(RwLock::new(PluginRegistry::new())),
            Some(Arc::new(RedactionMediaBridge)),
        );
        let inspect_result = inspect_dispatcher.dispatch(
            "inspect_media",
            serde_json::json!({"mediaRef": "generated-asset"}),
        );
        assert!(!inspect_result.is_error, "{}", inspect_result.text_joined());

        let advanced_dispatcher = Dispatcher::with_all_capability_bridges(
            Arc::new(AppCoreHandle::new(state.core.clone())),
            Arc::new(RwLock::new(PluginRegistry::new())),
            None,
            None,
            None,
            Some(Arc::new(RedactionAdvancedBridge)),
        );
        let advanced_result = advanced_dispatcher.dispatch(
            "generate_avatar",
            serde_json::json!({
                "portraitMediaRef": "generated-asset",
                "audioMediaRef": "generated-asset",
                "consentId": "consent-safe",
                "costAuthorized": true
            }),
        );
        assert!(
            !advanced_result.is_error,
            "{}",
            advanced_result.text_joined()
        );

        let motion_dispatcher = Dispatcher::with_capability_bridges(
            Arc::new(AppCoreHandle::new(state.core.clone())),
            Arc::new(RwLock::new(PluginRegistry::new())),
            None,
            None,
            Some(Arc::new(RedactionMotionBridge)),
        );
        let add_motion_result = motion_dispatcher.dispatch(
            "add_motion_graphic",
            serde_json::json!({
                "source": {"code": "export default {}"},
                "startFrame": 0,
                "durationFrames": 90
            }),
        );
        assert!(
            !add_motion_result.is_error,
            "{}",
            add_motion_result.text_joined()
        );
        let edit_motion_result = motion_dispatcher.dispatch(
            "edit_motion_graphic",
            serde_json::json!({
                "clipId": "motion-safe-edit-clip",
                "code": "export default {}"
            }),
        );
        assert!(
            !edit_motion_result.is_error,
            "{}",
            edit_motion_result.text_joined()
        );

        let mut session = ChatSession::new("get-media-redaction");
        session.messages.push(ChatMessage::tool_result_blocks(
            "call-get-media",
            result.content.clone(),
            safe_tool_result_for_llm(&result),
            false,
        ));
        session.messages.push(ChatMessage::tool_result_blocks(
            "call-inspect-media",
            inspect_result.content.clone(),
            safe_tool_result_for_llm(&inspect_result),
            false,
        ));
        session.messages.push(ChatMessage::tool_result_blocks(
            "call-generate-avatar",
            advanced_result.content.clone(),
            safe_tool_result_for_llm(&advanced_result),
            false,
        ));
        session.messages.push(ChatMessage::tool_result_blocks(
            "call-add-motion",
            add_motion_result.content.clone(),
            safe_tool_result_for_llm(&add_motion_result),
            false,
        ));
        session.messages.push(ChatMessage::tool_result_blocks(
            "call-edit-motion",
            edit_motion_result.content.clone(),
            safe_tool_result_for_llm(&edit_motion_result),
            false,
        ));
        state.put_project_session(&project, session).unwrap();

        let persisted =
            std::fs::read_to_string(bundle.join("chat-sessions/get-media-redaction.json")).unwrap();
        for secret in [
            ABSOLUTE_PATH,
            SIGNED_RESULT_URL,
            SIGNED_REFERENCE_URL,
            SIGNED_IMAGE_URL,
            PRIVATE_PROMPT,
            ADVANCED_PATH,
            ADVANCED_URL_SECRET,
            ADVANCED_PROVIDER_ID,
            ADVANCED_PROMPT,
            ADVANCED_KEY,
            MOTION_PRIVATE_RENDERER,
            MOTION_PRIVATE_VERSION,
            MOTION_PRIVATE_OUTPUT,
            MOTION_PRIVATE_HASH,
            MOTION_PRIVATE_ACTION,
        ] {
            assert!(
                !persisted.contains(secret),
                "persisted chat history leaked {secret}: {persisted}"
            );
        }
        assert!(persisted.contains("generated-asset"));
        assert!(persisted.contains("Generated clip"));
        assert!(persisted.contains("avatar-safe-asset"));
        assert!(persisted.contains("motion-safe-add-clip"));
        assert!(persisted.contains("motion-safe-add-asset"));
        assert!(persisted.contains("motion-safe-edit-clip"));
        assert!(persisted.contains("motion-safe-edit-asset"));
    }

    #[test]
    fn session_open_state_persists_across_state_recreation() {
        let temp = tempfile::tempdir().unwrap();
        let bundle = temp.path().join("Tabs.opentake");
        let core = AppCore::new();
        core.save_project(Some(bundle)).unwrap();
        let state = ChatState::new(
            core.clone(),
            temp.path().join("no-workflows"),
            temp.path().join("chat-cache"),
            temp.path().join("chat-models"),
        );
        let project = state.project_context().unwrap();

        let opened = state
            .set_project_session_open(&project, "chat-tab", true)
            .unwrap();
        assert!(opened.is_open);
        let closed = state
            .set_project_session_open(&project, "chat-tab", false)
            .unwrap();
        assert!(!closed.is_open);
        assert!(state
            .take_open_project_session_for_turn(&project, "chat-tab")
            .is_err());

        let recreated = ChatState::new(
            core,
            temp.path().join("no-workflows-2"),
            temp.path().join("chat-cache-2"),
            temp.path().join("chat-models-2"),
        );
        let recreated_project = recreated.project_context().unwrap();
        let sessions = recreated.list_project_sessions(&recreated_project).unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].id, "chat-tab");
        assert!(!sessions[0].is_open);
    }

    #[test]
    fn deleting_a_session_removes_it_but_never_one_with_a_running_turn() {
        let temp = tempfile::tempdir().unwrap();
        let bundle = temp.path().join("Delete.opentake");
        let core = AppCore::new();
        core.save_project(Some(bundle.clone())).unwrap();
        let state = ChatState::new(
            core.clone(),
            temp.path().join("no-workflows"),
            temp.path().join("chat-cache"),
            temp.path().join("chat-models"),
        );
        let project = state.project_context().unwrap();
        for id in ["chat-old", "chat-busy"] {
            let mut session = ChatSession::new(id);
            session.messages.push(ChatMessage::user("hello"));
            state.put_project_session(&project, session).unwrap();
        }

        let busy = project.key("chat-busy");
        let _lease = state
            .reserve_turn(busy.clone(), Arc::new(TurnCancel::new()))
            .unwrap();
        let refused = state
            .delete_project_session(&project, "chat-busy")
            .expect_err("a running turn keeps its session");
        assert!(refused.contains("running turn"), "{refused}");
        state.release_turn(&busy);

        state.delete_project_session(&project, "chat-old").unwrap();
        assert!(!bundle.join("chat-sessions/chat-old.json").exists());
        let listed = state
            .list_project_sessions(&project)
            .unwrap()
            .into_iter()
            .map(|session| session.id)
            .collect::<Vec<_>>();
        assert_eq!(listed, ["chat-busy"]);
        assert!(state
            .take_project_session(&project, "chat-old")
            .unwrap()
            .messages
            .is_empty());

        core.new_project();
        core.save_project(Some(temp.path().join("Other.opentake")))
            .unwrap();
        assert!(state.delete_project_session(&project, "chat-busy").is_err());
        assert!(bundle.join("chat-sessions/chat-busy.json").exists());
    }

    #[test]
    fn stale_project_turn_cannot_overwrite_the_previous_bundle() {
        let temp = tempfile::tempdir().unwrap();
        let bundle = temp.path().join("A.opentake");
        let core = AppCore::new();
        core.save_project(Some(bundle.clone())).unwrap();
        let state = ChatState::new(
            core.clone(),
            temp.path().join("no-workflows"),
            temp.path().join("chat-cache"),
            temp.path().join("chat-models"),
        );
        let project_a = state.project_context().unwrap();
        let mut baseline = ChatSession::new("chat-stale");
        baseline.messages.push(ChatMessage::user("A baseline"));
        state
            .put_project_session(&project_a, baseline.clone())
            .unwrap();

        core.new_project();
        let bundle_b = temp.path().join("B.opentake");
        core.save_project(Some(bundle_b)).unwrap();
        baseline
            .messages
            .push(ChatMessage::assistant("stale result", vec![]));
        assert!(state.put_project_session(&project_a, baseline).is_err());

        let disk = opentake_agent::chat::ChatSessionStore::open(bundle)
            .unwrap()
            .load("chat-stale")
            .unwrap()
            .unwrap();
        assert_eq!(disk.messages.len(), 1);
        assert_eq!(disk.messages[0].content, "A baseline");
    }

    /// Blocks `transcribe_sources_cancellable` until its token is cancelled.
    struct CancelAwaitingTranscriptBridge {
        entered: Mutex<std::sync::mpsc::Sender<()>>,
    }

    impl opentake_agent::mcp::media_bridge::MediaBridge for CancelAwaitingTranscriptBridge {
        fn transcribe_sources_cancellable(
            &self,
            _sources: &[opentake_agent::mcp::media_bridge::TranscriptSource],
            cancel: &opentake_media::MediaCancelToken,
        ) -> Result<
            Vec<opentake_agent::mcp::media_bridge::TranscriptSourceResult>,
            opentake_agent::mcp::media_bridge::BridgeError,
        > {
            let _ = self.entered.lock().unwrap().send(());
            let started = std::time::Instant::now();
            while !cancel.is_cancelled() && started.elapsed() < std::time::Duration::from_secs(10) {
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            Err(opentake_agent::mcp::media_bridge::cancelled_bridge_error())
        }
    }

    /// Ignores cancellation: blocks `transcribe_sources_cancellable` until
    /// the test releases it (or a long safety timeout passes).
    #[cfg(unix)]
    struct StubbornTranscriptBridge {
        entered: Mutex<std::sync::mpsc::Sender<()>>,
        release: Mutex<std::sync::mpsc::Receiver<()>>,
    }

    #[cfg(unix)]
    impl opentake_agent::mcp::media_bridge::MediaBridge for StubbornTranscriptBridge {
        fn transcribe_sources_cancellable(
            &self,
            _sources: &[opentake_agent::mcp::media_bridge::TranscriptSource],
            _cancel: &opentake_media::MediaCancelToken,
        ) -> Result<
            Vec<opentake_agent::mcp::media_bridge::TranscriptSourceResult>,
            opentake_agent::mcp::media_bridge::BridgeError,
        > {
            let _ = self.entered.lock().unwrap().send(());
            let _ = self
                .release
                .lock()
                .unwrap()
                .recv_timeout(std::time::Duration::from_secs(60));
            Err(opentake_agent::mcp::media_bridge::cancelled_bridge_error())
        }
    }

    /// A saved project whose timeline holds one video clip with audio, so
    /// `get_transcript` reaches the media bridge.
    #[cfg(unix)]
    fn project_with_a_talking_clip(root: &std::path::Path) -> AppCore {
        let core = AppCore::new();
        core.save_project(Some(root.join("A.opentake"))).unwrap();
        core.apply(
            opentake_ops::command::EditCommand::RegisterMediaAndAddClip {
                media: opentake_domain::MediaManifestEntry {
                    id: "talk".into(),
                    name: "talk.mov".into(),
                    kind: opentake_domain::ClipType::Video,
                    source: opentake_domain::MediaSource::Project {
                        relative_path: "media/talk.mov".into(),
                    },
                    duration: 10.0,
                    generation_input: None,
                    source_width: Some(1920),
                    source_height: Some(1080),
                    source_fps: Some(30.0),
                    has_audio: Some(true),
                    color: None,
                    proxy: None,
                    folder_id: None,
                    cached_remote_url: None,
                    cached_remote_url_expires_at: None,
                },
                entry: opentake_ops::command::ClipEntry {
                    media_ref: "talk".into(),
                    media_type: opentake_domain::ClipType::Video,
                    source_clip_type: opentake_domain::ClipType::Video,
                    track_index: 0,
                    start_frame: 0,
                    duration_frames: 300,
                    trim_start_frame: None,
                    trim_end_frame: None,
                    has_audio: true,
                    add_linked_audio: false,
                    transform: None,
                },
                auto_track: true,
            },
        )
        .unwrap();
        core
    }

    /// A fake official Codex CLI that reads the prompt, opens an MCP session
    /// on the turn's private endpoint with curl, calls `get_transcript`, and
    /// then runs `after_call` (with the call still in flight when
    /// `background` is set).
    #[cfg(unix)]
    fn fake_codex_calling_get_transcript(
        root: &std::path::Path,
        background: bool,
        after_call: &str,
    ) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;

        let call_suffix = if background { " &" } else { "" };
        let script = format!(
            r#"#!/bin/sh
set -u
url=""
for arg in "$@"; do
  case "$arg" in
    mcp_servers.opentake.url=*) url="${{arg#mcp_servers.opentake.url=\"}}"; url="${{url%\"}}" ;;
  esac
done
IFS= read -r prompt || true
printf '%s\n' '{{"type":"thread.started"}}'
auth="Authorization: Bearer $OPENTAKE_CODEX_MCP_BEARER_TOKEN"
session="$(curl -sS --max-time 10 -D - -o /dev/null -H "$auth" \
  -H 'content-type: application/json' -H 'accept: application/json, text/event-stream' \
  --data '{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":"2025-06-18","capabilities":{{}},"clientInfo":{{"name":"fake-codex","version":"0"}}}}}}' \
  "$url" | tr -d '\r' | sed -n 's/^[Mm][Cc][Pp]-[Ss][Ee][Ss][Ss][Ii][Oo][Nn]-[Ii][Dd]: //p')"
curl -sS --max-time 10 -o /dev/null -H "$auth" -H "mcp-session-id: $session" \
  -H 'mcp-protocol-version: 2025-06-18' \
  -H 'content-type: application/json' -H 'accept: application/json, text/event-stream' \
  --data '{{"jsonrpc":"2.0","method":"notifications/initialized"}}' "$url"
curl -sS -N -o /dev/null -H "$auth" -H "mcp-session-id: $session" \
  -H 'mcp-protocol-version: 2025-06-18' \
  -H 'content-type: application/json' -H 'accept: application/json, text/event-stream' \
  --data '{{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{{"name":"get_transcript","arguments":{{}}}}}}' \
  "$url"{call_suffix}
{after_call}
"#
        );
        let path = root.join("fake-codex");
        std::fs::write(&path, script).unwrap();
        let mut permissions = std::fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&path, permissions).unwrap();
        path
    }

    /// A reserved Codex turn on a saved project whose dispatcher uses
    /// `bridge` for transcription.
    #[cfg(unix)]
    struct CodexTurnFixture {
        state: ChatState,
        project: ChatProjectContext,
        key: SessionKey,
        cancel: Arc<TurnCancel>,
        _lease: crate::updater::ActivityLease,
        context: crate::codex::CodexTurnContext,
    }

    #[cfg(unix)]
    fn codex_turn_fixture(
        root: &std::path::Path,
        bridge: Arc<dyn opentake_agent::mcp::media_bridge::MediaBridge>,
    ) -> CodexTurnFixture {
        let core = project_with_a_talking_clip(root);
        let state = ChatState::new(
            core.clone(),
            root.join("no-workflows"),
            root.join("chat-cache"),
            root.join("chat-models"),
        );
        let project = state.project_context().unwrap();
        let key = project.key("chat-codex");
        let cancel = Arc::new(TurnCancel::new());
        let lease = state.reserve_turn(key.clone(), cancel.clone()).unwrap();
        let gate = state.project_turn_gate(&project, &key, cancel.clone());
        let handle: Arc<dyn CoreHandle> = Arc::new(AppCoreHandle::new(core));
        let registry = Arc::new(RwLock::new(crate::mcp::build_registry(
            &root.join("no-workflows"),
        )));
        let dispatcher = Arc::new(Dispatcher::with_bridge(
            handle,
            registry.clone(),
            Some(bridge),
        ));
        let context = crate::codex::CodexTurnContext {
            dispatcher,
            registry,
            gate,
            cancel: cancel.requested.clone(),
        };
        CodexTurnFixture {
            state,
            project,
            key,
            cancel,
            _lease: lease,
            context,
        }
    }

    #[cfg(unix)]
    async fn wait_for_bridge_entry(entered: std::sync::mpsc::Receiver<()>) {
        tokio::task::spawn_blocking(move || {
            entered
                .recv_timeout(std::time::Duration::from_secs(20))
                .expect("the fake Codex CLI called get_transcript")
        })
        .await
        .unwrap();
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn codex_chat_cancel_ends_a_turn_with_a_long_transcription_within_two_seconds() {
        let temp = tempfile::tempdir().unwrap();
        let script = fake_codex_calling_get_transcript(temp.path(), false, "sleep 60");
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let fixture = codex_turn_fixture(
            temp.path(),
            Arc::new(CancelAwaitingTranscriptBridge {
                entered: Mutex::new(entered_tx),
            }),
        );
        let context = fixture.context;
        let turn = tokio::spawn(async move {
            crate::codex::run_agent_turn_with_executable(&script, context, "caption it", |_| {})
                .await
        });
        wait_for_bridge_entry(entered_rx).await;

        let started = std::time::Instant::now();
        fixture.state.cancel_turn(&fixture.key).unwrap();
        let result = tokio::time::timeout(std::time::Duration::from_secs(10), turn)
            .await
            .expect("the cancelled turn ended")
            .unwrap();
        let elapsed = started.elapsed();
        assert_eq!(result.unwrap_err(), crate::codex::CodexTurnError::Cancelled);
        assert!(fixture.cancel.media.is_cancelled());
        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "chat_cancel took {elapsed:?}"
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn codex_stop_during_an_internal_failure_drain_takes_effect() {
        let temp = tempfile::tempdir().unwrap();
        // The tool call is still running when the CLI reports garbage, so
        // the turn fails internally and drains its endpoint.
        let script = fake_codex_calling_get_transcript(
            temp.path(),
            true,
            "sleep 1\nprintf '%s\\n' 'not json'\nsleep 60",
        );
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let fixture = codex_turn_fixture(
            temp.path(),
            Arc::new(StubbornTranscriptBridge {
                entered: Mutex::new(entered_tx),
                release: Mutex::new(release_rx),
            }),
        );
        let context = fixture.context;
        let turn = tokio::spawn(async move {
            crate::codex::run_agent_turn_with_executable(&script, context, "caption it", |_| {})
                .await
        });
        wait_for_bridge_entry(entered_rx).await;
        // The protocol error has cancelled only the dispatch; the stubborn
        // transcription keeps the drain waiting on the 30-minute deadline.
        let drain_started = tokio::time::timeout(std::time::Duration::from_secs(10), async {
            while !fixture.cancel.media.is_cancelled() {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await;
        assert!(drain_started.is_ok(), "the internal failure began draining");
        assert!(!fixture.cancel.requested.load(Ordering::Acquire));

        fixture.state.cancel_turn(&fixture.key).unwrap();
        let result = tokio::time::timeout(std::time::Duration::from_secs(20), turn)
            .await
            .expect("Stop during the drain took effect")
            .unwrap();
        assert!(result.is_err());
        // The Stop wins: the turn is no longer eligible to commit its reply.
        assert!(!fixture.cancel.begin_finalization());
        let _ = release_tx.send(());
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn codex_deadline_keeps_the_timeout_reply_when_a_tool_call_outlives_the_drain() {
        let temp = tempfile::tempdir().unwrap();
        let script = fake_codex_calling_get_transcript(temp.path(), false, "sleep 60");
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let fixture = codex_turn_fixture(
            temp.path(),
            Arc::new(StubbornTranscriptBridge {
                entered: Mutex::new(entered_tx),
                release: Mutex::new(release_rx),
            }),
        );
        let context = fixture.context;
        // Leave the fake CLI time to start and call the tool; the runner
        // keeps the last two seconds for cleanup.
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(8);
        let turn = tokio::spawn(async move {
            crate::codex::run_agent_turn_with_executable_until(
                &script,
                context,
                "caption it",
                |_| {},
                deadline,
            )
            .await
        });
        wait_for_bridge_entry(entered_rx).await;
        let result = tokio::time::timeout(std::time::Duration::from_secs(30), turn)
            .await
            .expect("the turn ended at its deadline")
            .unwrap();
        assert_eq!(result.unwrap_err(), crate::codex::CodexTurnError::Timeout);
        // The stuck tool call was only asked to stop; the turn itself was not
        // cancelled, so its timeout reply is still committed.
        assert!(fixture.cancel.media.is_cancelled());
        assert!(!fixture.cancel.requested.load(Ordering::Acquire));
        let mut session = ChatSession::new("chat-codex");
        session
            .messages
            .push(ChatMessage::assistant("timed out", vec![]));
        assert!(matches!(
            fixture.state.finalize_project_turn(
                &fixture.project,
                &fixture.key,
                &fixture.cancel,
                session
            ),
            Ok(TurnFinalization::Committed)
        ));
        let _ = release_tx.send(());
    }

    #[test]
    fn byok_turn_transcription_is_cancelled_by_a_project_switch_without_blocking_it() {
        let temp = tempfile::tempdir().unwrap();
        let core = AppCore::new();
        core.save_project(Some(temp.path().join("A.opentake")))
            .unwrap();
        core.apply(
            opentake_ops::command::EditCommand::RegisterMediaAndAddClip {
                media: opentake_domain::MediaManifestEntry {
                    id: "talk".into(),
                    name: "talk.mov".into(),
                    kind: opentake_domain::ClipType::Video,
                    source: opentake_domain::MediaSource::Project {
                        relative_path: "media/talk.mov".into(),
                    },
                    duration: 10.0,
                    generation_input: None,
                    source_width: Some(1920),
                    source_height: Some(1080),
                    source_fps: Some(30.0),
                    has_audio: Some(true),
                    color: None,
                    proxy: None,
                    folder_id: None,
                    cached_remote_url: None,
                    cached_remote_url_expires_at: None,
                },
                entry: opentake_ops::command::ClipEntry {
                    media_ref: "talk".into(),
                    media_type: opentake_domain::ClipType::Video,
                    source_clip_type: opentake_domain::ClipType::Video,
                    track_index: 0,
                    start_frame: 0,
                    duration_frames: 300,
                    trim_start_frame: None,
                    trim_end_frame: None,
                    has_audio: true,
                    add_linked_audio: false,
                    transform: None,
                },
                auto_track: true,
            },
        )
        .unwrap();
        let state = ChatState::new(
            core.clone(),
            temp.path().join("no-workflows"),
            temp.path().join("chat-cache"),
            temp.path().join("chat-models"),
        );
        let project = state.project_context().unwrap();
        let cancel = Arc::new(TurnCancel::new());
        let _lease = state
            .reserve_turn(project.key("chat-byok"), cancel.clone())
            .unwrap();
        let gate = ProjectTurnGate {
            state: state.clone(),
            project,
            cancel: cancel.clone(),
            undo_scope: "test:byok-transcript".into(),
        };
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let handle: Arc<dyn CoreHandle> = Arc::new(AppCoreHandle::new(core.clone()));
        let registry = Arc::new(RwLock::new(crate::mcp::build_registry(
            &temp.path().join("no-workflows"),
        )));
        let dispatcher = Arc::new(Dispatcher::with_bridge(
            handle,
            registry,
            Some(Arc::new(CancelAwaitingTranscriptBridge {
                entered: Mutex::new(entered_tx),
            })),
        ));
        let dispatch = std::thread::spawn(move || {
            gate.dispatch(&dispatcher, "get_transcript", serde_json::json!({}))
        });
        entered_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("transcription started");

        let started = std::time::Instant::now();
        core.new_project();
        let switched_in = started.elapsed();
        let result = dispatch.join().expect("dispatch thread joined");
        let finished_in = started.elapsed();

        assert!(cancel.media.is_cancelled());
        assert!(result.is_none(), "a stale-project result is dropped");
        assert!(
            switched_in < std::time::Duration::from_secs(1),
            "{switched_in:?}"
        );
        assert!(
            finished_in < std::time::Duration::from_secs(1),
            "{finished_in:?}"
        );
    }

    #[test]
    fn stale_project_turn_gate_cannot_dispatch_into_the_replacement_project() {
        let temp = tempfile::tempdir().unwrap();
        let core = AppCore::new();
        core.save_project(Some(temp.path().join("A.opentake")))
            .unwrap();
        let state = ChatState::new(
            core.clone(),
            temp.path().join("no-workflows"),
            temp.path().join("chat-cache"),
            temp.path().join("chat-models"),
        );
        let project_a = state.project_context().unwrap();
        let cancel = Arc::new(TurnCancel::new());
        let gate = ProjectTurnGate {
            state,
            project: project_a,
            cancel: cancel.clone(),
            undo_scope: "test:stale-project".into(),
        };
        let handle: Arc<dyn CoreHandle> = Arc::new(AppCoreHandle::new(core.clone()));
        let registry = Arc::new(RwLock::new(crate::mcp::build_registry(
            &temp.path().join("no-workflows"),
        )));
        let dispatcher = Dispatcher::new(handle, registry);

        let current = gate
            .dispatch(
                &dispatcher,
                "create_folder",
                serde_json::json!({"name": "A folder"}),
            )
            .expect("the captured project is initially current");
        assert!(!current.is_error, "{}", current.text_joined());

        core.new_project();
        core.save_project(Some(temp.path().join("B.opentake")))
            .unwrap();
        assert!(gate
            .dispatch(
                &dispatcher,
                "create_folder",
                serde_json::json!({"name": "stale folder"}),
            )
            .is_none());
        assert!(cancel.requested.load(Ordering::Relaxed));
        assert!(cancel.media.is_cancelled());
        assert!(core.media().folders.is_empty());
    }

    #[test]
    fn project_turn_gate_releases_identity_lease_before_timeline_result_capture() {
        let temp = tempfile::tempdir().unwrap();
        let core = AppCore::new();
        core.save_project(Some(temp.path().join("A.opentake")))
            .unwrap();
        let state = ChatState::new(
            core.clone(),
            temp.path().join("no-workflows"),
            temp.path().join("chat-cache"),
            temp.path().join("chat-models"),
        );
        let cancel = Arc::new(TurnCancel::new());
        let gate = ProjectTurnGate {
            project: state.project_context().unwrap(),
            state,
            cancel,
            undo_scope: "test:deferred-timeline-capture".into(),
        };
        let handle: Arc<dyn CoreHandle> = Arc::new(AppCoreHandle::new(core.clone()));
        let registry = Arc::new(RwLock::new(crate::mcp::build_registry(
            &temp.path().join("no-workflows"),
        )));
        let (capture_started_tx, capture_started_rx) = std::sync::mpsc::channel();
        let (release_capture_tx, release_capture_rx) = std::sync::mpsc::channel();
        let dispatcher = Dispatcher::with_bridge(
            handle,
            registry,
            Some(Arc::new(BlockingTimelineResultBridge {
                capture_started: capture_started_tx,
                release_capture: Mutex::new(release_capture_rx),
            })),
        );

        let add_result = dispatcher.dispatch(
            "add_texts",
            serde_json::json!({
                "entries": [
                    {"startFrame": 0, "durationFrames": 30, "content": "visible"}
                ]
            }),
        );
        assert!(!add_result.is_error, "{}", add_result.text_joined());
        let clip_id = dispatcher.timeline().tracks[0].clips[0].id.clone();

        let dispatch_thread = std::thread::spawn(move || {
            gate.dispatch(
                &dispatcher,
                "remove_clips",
                serde_json::json!({"clipIds": [clip_id]}),
            )
        });
        capture_started_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("remove-last-clip should start timeline capture");

        let replacement_bundle = temp.path().join("B.opentake");
        let (transitioned_tx, transitioned_rx) = std::sync::mpsc::channel();
        let transition_core = core.clone();
        let transition_thread = std::thread::spawn(move || {
            let result = transition_core.save_project(Some(replacement_bundle));
            transitioned_tx.send(result).unwrap();
        });
        // A capture that held the identity lease would block this save until
        // the capture is released, so a generous window still catches it
        // while leaving a slow runner time to write the new bundle.
        let transition_during_capture =
            transitioned_rx.recv_timeout(std::time::Duration::from_secs(5));
        let transitioned_while_capture_blocked = transition_during_capture.is_ok();

        release_capture_tx.send(()).unwrap();
        let dispatch_result = dispatch_thread.join().unwrap();
        let transition_result = match transition_during_capture {
            Ok(result) => result,
            Err(_) => transitioned_rx
                .recv_timeout(std::time::Duration::from_secs(2))
                .expect("project transition should finish after capture is released"),
        };
        transition_thread.join().unwrap();
        transition_result.unwrap();

        assert!(
            transitioned_while_capture_blocked,
            "timeline capture must not hold the project identity workflow lease"
        );
        assert!(
            dispatch_result.is_none(),
            "a result captured for the replaced project must be discarded"
        );
    }

    #[test]
    fn reserving_a_turn_is_atomic_per_project_session() {
        let temp = tempfile::tempdir().unwrap();
        let core = AppCore::new();
        let state = ChatState::new(
            core,
            temp.path().join("no-workflows"),
            temp.path().join("chat-cache"),
            temp.path().join("chat-models"),
        );
        let key = SessionKey {
            project_epoch: 7,
            project_dir: temp.path().join("A.opentake"),
            session_id: "chat-one".into(),
        };

        state
            .reserve_turn(key.clone(), Arc::new(TurnCancel::new()))
            .unwrap();
        let error = state
            .reserve_turn(key, Arc::new(TurnCancel::new()))
            .expect_err("a second turn must not replace the first cancel token");

        assert!(error.contains("already running"));
        assert_eq!(state.turns.lock().unwrap().running.len(), 1);
    }

    #[test]
    fn chat_turn_admission_spans_reservation_and_rejects_turns_during_install() {
        let temp = tempfile::tempdir().unwrap();
        let core = AppCore::new();
        let admission = crate::updater::InstallAdmissionGate::default();
        let state = ChatState::new_with_admission(
            core,
            temp.path().join("no-workflows"),
            temp.path().join("chat-cache"),
            temp.path().join("chat-models"),
            admission.clone(),
        );
        let key = SessionKey {
            project_epoch: 7,
            project_dir: temp.path().join("A.opentake"),
            session_id: "chat-admission".into(),
        };

        let turn_admission = state
            .reserve_turn(key.clone(), Arc::new(TurnCancel::new()))
            .unwrap();
        assert!(admission.begin_install().is_err());
        state.release_turn(&key);
        drop(turn_admission);

        let install = admission.begin_install().unwrap();
        assert!(state
            .reserve_turn(key.clone(), Arc::new(TurnCancel::new()))
            .is_err());
        drop(install);
        let resumed = state
            .reserve_turn(key.clone(), Arc::new(TurnCancel::new()))
            .unwrap();
        state.release_turn(&key);
        drop(resumed);
    }

    #[test]
    fn project_turn_gate_request_cancel_cancels_the_whole_turn() {
        let temp = tempfile::tempdir().unwrap();
        let core = AppCore::new();
        core.save_project(Some(temp.path().join("Project.opentake")))
            .unwrap();
        let state = ChatState::new(
            core,
            temp.path().join("no-workflows"),
            temp.path().join("chat-cache"),
            temp.path().join("chat-models"),
        );
        let cancel = Arc::new(TurnCancel::new());
        let gate = ProjectTurnGate {
            project: state.project_context().unwrap(),
            state,
            cancel: cancel.clone(),
            undo_scope: "test:cancel".into(),
        };

        gate.request_dispatch_cancel();
        assert!(!cancel.requested.load(Ordering::Acquire));
        assert!(cancel.media.is_cancelled());

        gate.request_cancel();

        assert!(cancel.requested.load(Ordering::Acquire));
        assert!(cancel.media.is_cancelled());
    }

    #[test]
    fn codex_terminal_commit_linearizes_with_whole_turn_cancellation() {
        let temp = tempfile::tempdir().unwrap();
        let core = AppCore::new();
        core.save_project(Some(temp.path().join("Project.opentake")))
            .unwrap();
        let state = ChatState::new(
            core,
            temp.path().join("no-workflows"),
            temp.path().join("chat-cache"),
            temp.path().join("chat-models"),
        );
        let project = state.project_context().unwrap();

        let mut cancelled_session = ChatSession::new("cancel-wins");
        cancelled_session
            .messages
            .push(ChatMessage::user("request"));
        state
            .put_project_session(&project, cancelled_session.clone())
            .unwrap();
        let cancelled_owner = Arc::new(TurnCancel::new());
        let cancelled_key = project.key(&cancelled_session.id);
        state
            .reserve_turn(cancelled_key.clone(), cancelled_owner.clone())
            .unwrap();
        cancelled_owner.request();
        cancelled_session
            .messages
            .push(ChatMessage::assistant("must not commit", Vec::new()));
        assert_eq!(
            state
                .finalize_project_turn(
                    &project,
                    &cancelled_key,
                    &cancelled_owner,
                    cancelled_session,
                )
                .unwrap(),
            TurnFinalization::Cancelled
        );
        let persisted = state.take_project_session(&project, "cancel-wins").unwrap();
        assert_eq!(persisted.messages.len(), 1);

        let mut committed_session = ChatSession::new("commit-wins");
        committed_session
            .messages
            .push(ChatMessage::user("request"));
        state
            .put_project_session(&project, committed_session.clone())
            .unwrap();
        let committed_owner = Arc::new(TurnCancel::new());
        let committed_key = project.key(&committed_session.id);
        state
            .reserve_turn(committed_key.clone(), committed_owner.clone())
            .unwrap();
        committed_session
            .messages
            .push(ChatMessage::assistant("committed", Vec::new()));
        assert_eq!(
            state
                .finalize_project_turn(
                    &project,
                    &committed_key,
                    &committed_owner,
                    committed_session,
                )
                .unwrap(),
            TurnFinalization::Committed
        );
        committed_owner.request();
        assert!(!committed_owner.requested.load(Ordering::Acquire));
        let persisted = state.take_project_session(&project, "commit-wins").unwrap();
        assert_eq!(persisted.messages.len(), 2);
        assert_eq!(persisted.messages[1].content, "committed");
    }

    #[test]
    fn authoritative_history_waits_until_the_exact_session_terminal_boundary() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        runtime.block_on(async {
            let temp = tempfile::tempdir().unwrap();
            let core = AppCore::new();
            let project_path = temp.path().join("Project.opentake");
            core.save_project(Some(project_path.clone())).unwrap();
            let state = ChatState::new(
                core,
                temp.path().join("no-workflows"),
                temp.path().join("chat-cache"),
                temp.path().join("chat-models"),
            );
            let project = state.project_context().unwrap();
            let mut session = ChatSession::new("chat-authoritative");
            session.messages.push(ChatMessage::user("request"));
            state.put_project_session(&project, session.clone()).unwrap();
            let owner = Arc::new(TurnCancel::new());
            let key = project.key(&session.id);
            let _turn_admission = state
                .reserve_turn(key.clone(), owner.clone())
                .unwrap();

            let expected_path = project_path.to_string_lossy().into_owned();
            let session_id = session.id.clone();
            let mut history = Box::pin(state.authoritative_project_history(
                project.project_epoch,
                &expected_path,
                &session_id,
            ));
            assert!(tokio::time::timeout(
                std::time::Duration::from_millis(10),
                history.as_mut(),
            )
            .await
            .is_err());

            session
                .messages
                .push(ChatMessage::assistant("final reply", Vec::new()));
            assert_eq!(
                state
                    .finalize_project_turn(&project, &key, &owner, session)
                    .unwrap(),
                TurnFinalization::Committed,
            );
            assert!(tokio::time::timeout(
                std::time::Duration::from_millis(10),
                history.as_mut(),
            )
            .await
            .is_err(), "durable commit alone must not precede the terminal event");

            state.complete_turn(&key, &owner);
            let next_owner = Arc::new(TurnCancel::new());
            let _next_turn_admission = state
                .reserve_turn(key.clone(), next_owner)
                .expect("a new turn may start after the observed turn completes");
            let result = tokio::time::timeout(std::time::Duration::from_secs(1), history).await;
            state.release_turn(&key);
            let messages = result
                .expect("authoritative history must return the observed turn instead of waiting for its successor")
                .unwrap();
            assert_eq!(messages.len(), 2);
            assert_eq!(messages[1].content, "final reply");
        });
    }

    #[test]
    fn save_as_cancels_and_purges_the_previous_project_turn() {
        let temp = tempfile::tempdir().unwrap();
        let core = AppCore::new();
        let source = temp.path().join("A.opentake");
        let target = temp.path().join("B.opentake");
        core.save_project(Some(source)).unwrap();
        let state = ChatState::new(
            core.clone(),
            temp.path().join("no-workflows"),
            temp.path().join("chat-cache"),
            temp.path().join("chat-models"),
        );
        let project_a = state.project_context().unwrap();
        state
            .put_project_session(&project_a, ChatSession::new("chat-same"))
            .unwrap();
        let old_cancel = Arc::new(TurnCancel::new());
        state
            .reserve_turn(project_a.key("chat-same"), old_cancel.clone())
            .unwrap();

        core.save_project(Some(target)).unwrap();

        assert!(old_cancel.requested.load(Ordering::Relaxed));
        assert!(old_cancel.media.is_cancelled());
        assert!(state.turns.lock().unwrap().running.is_empty());
        assert!(state.sessions.lock().unwrap().is_empty());
        let project_b = state.project_context().unwrap();
        state
            .reserve_turn(project_b.key("chat-same"), Arc::new(TurnCancel::new()))
            .expect("same session id in the Save As target is independent");
    }

    #[test]
    fn save_as_cancels_an_active_tool_before_waiting_for_its_identity_lease() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("Source.opentake");
        let target = temp.path().join("Target.opentake");
        let core = AppCore::new();
        core.save_project(Some(source)).unwrap();
        let state = ChatState::new(
            core.clone(),
            temp.path().join("no-workflows"),
            temp.path().join("chat-cache"),
            temp.path().join("chat-models"),
        );
        let project = state.project_context().unwrap();
        let cancel = Arc::new(TurnCancel::new());
        state
            .reserve_turn(project.key("chat-active"), cancel.clone())
            .unwrap();
        let gate = ProjectTurnGate {
            state,
            project,
            cancel: cancel.clone(),
            undo_scope: "test:save-as".into(),
        };
        let (tool_started_tx, tool_started_rx) = std::sync::mpsc::channel();
        let (tool_finished_tx, tool_finished_rx) = std::sync::mpsc::channel();

        let tool = std::thread::spawn(move || {
            gate.with_current_project(|| {
                tool_started_tx.send(()).unwrap();
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
                while !cancel.media.is_cancelled() && std::time::Instant::now() < deadline {
                    std::thread::yield_now();
                }
                if cancel.media.is_cancelled() {
                    tool_finished_tx.send(()).unwrap();
                }
            })
        });
        tool_started_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("the simulated media tool must hold the project identity lease");

        let saver = std::thread::spawn(move || core.save_project(Some(target)));

        tool_finished_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("Save As must cancel the active tool before waiting for the write lease");
        assert!(tool.join().unwrap().is_some());
        saver.join().unwrap().unwrap();
    }

    #[test]
    fn save_as_rejects_a_turn_registered_after_the_cancel_sweep() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("Source.opentake");
        let target = temp.path().join("Target.opentake");
        let core = AppCore::new();
        core.save_project(Some(source)).unwrap();
        let state = ChatState::new(
            core.clone(),
            temp.path().join("no-workflows"),
            temp.path().join("chat-cache"),
            temp.path().join("chat-models"),
        );
        let project = state.project_context().unwrap();
        let (hook_entered_tx, hook_entered_rx) = std::sync::mpsc::channel();
        let hook_release = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        let hook_release_for_listener = hook_release.clone();
        core.subscribe_project_identity_transition(move |pending| {
            if !pending {
                return;
            }
            hook_entered_tx.send(()).unwrap();
            let (released, wake) = &*hook_release_for_listener;
            let mut released = released.lock().unwrap();
            while !*released {
                released = wake.wait(released).unwrap();
            }
        });

        let saver = std::thread::spawn(move || core.save_project(Some(target)));
        hook_entered_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("Save As must reach the pre-transition hook");

        let late_cancel = Arc::new(TurnCancel::new());
        let reserve = state.reserve_turn(project.key("chat-too-late"), late_cancel.clone());
        let (released, wake) = &*hook_release;
        *released.lock().unwrap() = true;
        wake.notify_all();
        saver.join().unwrap().unwrap();

        assert!(
            reserve.is_err(),
            "a pending transition must reject new turns"
        );
        assert!(late_cancel.requested.load(Ordering::Relaxed));
        assert!(late_cancel.media.is_cancelled());
        let current = state.project_context().unwrap();
        state
            .reserve_turn(current.key("chat-after-save"), Arc::new(TurnCancel::new()))
            .expect("the transition flag must clear after Save As completes");
    }

    #[test]
    fn overlapping_save_as_transitions_keep_new_turns_blocked_until_both_finish() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("Source.opentake");
        let target_a = temp.path().join("Target-A.opentake");
        let target_b = temp.path().join("Target-B.opentake");
        let core = AppCore::new();
        core.save_project(Some(source)).unwrap();
        let state = ChatState::new(
            core.clone(),
            temp.path().join("no-workflows"),
            temp.path().join("chat-cache"),
            temp.path().join("chat-models"),
        );
        let true_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (first_entered_tx, first_entered_rx) = std::sync::mpsc::channel();
        let (second_entered_tx, second_entered_rx) = std::sync::mpsc::channel();
        let first_release = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        let true_count_for_listener = true_count.clone();
        let first_release_for_listener = first_release.clone();
        core.subscribe_project_identity_transition(move |pending| {
            if !pending {
                return;
            }
            let index = true_count_for_listener.fetch_add(1, Ordering::Relaxed);
            if index == 0 {
                first_entered_tx.send(()).unwrap();
                let (released, wake) = &*first_release_for_listener;
                let mut released = released.lock().unwrap();
                while !*released {
                    released = wake.wait(released).unwrap();
                }
            } else {
                second_entered_tx.send(()).unwrap();
            }
        });

        let core_a = core.clone();
        let saver_a = std::thread::spawn(move || core_a.save_project(Some(target_a)));
        first_entered_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("the first transition must pause before its writer");
        let core_b = core.clone();
        let saver_b = std::thread::spawn(move || core_b.save_project(Some(target_b)));
        second_entered_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("the second transition must overlap the first");
        saver_b.join().unwrap().unwrap();

        let current = state.project_context().unwrap();
        let late_cancel = Arc::new(TurnCancel::new());
        let reserve = state.reserve_turn(current.key("chat-between-saves"), late_cancel.clone());
        let (released, wake) = &*first_release;
        *released.lock().unwrap() = true;
        wake.notify_all();
        saver_a.join().unwrap().unwrap();

        assert!(
            reserve.is_err(),
            "one completed transition must not clear another transition's pending state"
        );
        assert!(late_cancel.requested.load(Ordering::Relaxed));
        assert!(late_cancel.media.is_cancelled());
        let current = state.project_context().unwrap();
        state
            .reserve_turn(current.key("chat-after-both"), Arc::new(TurnCancel::new()))
            .expect("new turns may resume after both transitions finish");
    }

    #[test]
    fn failed_save_as_clears_the_transition_flag() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("Source.opentake");
        let invalid_target = temp.path().join("not-a-bundle");
        let core = AppCore::new();
        core.save_project(Some(source)).unwrap();
        let state = ChatState::new(
            core.clone(),
            temp.path().join("no-workflows"),
            temp.path().join("chat-cache"),
            temp.path().join("chat-models"),
        );
        let project = state.project_context().unwrap();
        std::fs::write(&invalid_target, b"regular file").unwrap();

        core.save_project(Some(invalid_target))
            .expect_err("Save As to a regular file must fail");

        assert_eq!(state.turns.lock().unwrap().transition_depth, 0);
        state
            .reserve_turn(
                project.key("chat-after-failure"),
                Arc::new(TurnCancel::new()),
            )
            .expect("a failed transition must not block later Agent turns");
    }
}
