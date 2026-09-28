//! Project-local chat-session persistence (`agent-SPEC.md` §5.8).

use std::path::Path;

use opentake_project::{ProjectError, ProjectRoot};
use serde_json::Value;

use crate::chat::ChatSession;

const MAX_SESSION_BYTES: usize = 8 * 1024 * 1024;
const MAX_SESSION_FILES: usize = 256;
const MAX_AGGREGATE_SESSION_BYTES: usize = 32 * 1024 * 1024;
/// Start of the note that replaces a tool-result image in a saved session.
const IMAGE_OMITTED_PREFIX: &str = "[Image omitted from the saved chat history";

#[derive(Debug, thiserror::Error)]
pub enum ChatSessionStoreError {
    #[error("invalid chat session id: {0}")]
    InvalidSessionId(String),
    #[error("chat session `{requested}` contains mismatched id `{stored}`")]
    MismatchedSessionId { requested: String, stored: String },
    #[error("chat session exceeds the {MAX_SESSION_BYTES}-byte limit; continue in a new chat")]
    TooLarge,
    #[error(
        "project exceeds the {MAX_SESSION_FILES}-session limit; delete old Agent chats to continue"
    )]
    TooManySessions,
    #[error(
        "project chat history exceeds the {MAX_AGGREGATE_SESSION_BYTES}-byte limit; delete old Agent chats to continue"
    )]
    AggregateTooLarge,
    #[error(transparent)]
    Project(#[from] ProjectError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

/// Retained no-follow authority for one project's `chat-sessions/` directory.
/// Every save uses a sibling temp file + atomic rename inside that authority.
pub struct ChatSessionStore {
    root: ProjectRoot,
}

impl ChatSessionStore {
    pub fn open(project_dir: impl AsRef<Path>) -> Result<Self, ChatSessionStoreError> {
        Ok(Self {
            root: ProjectRoot::open(project_dir)?,
        })
    }

    pub fn root(&self) -> &ProjectRoot {
        &self.root
    }

    pub fn load(&self, session_id: &str) -> Result<Option<ChatSession>, ChatSessionStoreError> {
        Ok(self
            .load_with_size(session_id)?
            .map(|(session, _byte_len)| session))
    }

    fn load_with_size(
        &self,
        session_id: &str,
    ) -> Result<Option<(ChatSession, usize)>, ChatSessionStoreError> {
        let file_name = session_file_name(session_id)?;
        let Some(bytes) = self.root.read_chat_session(&file_name, MAX_SESSION_BYTES)? else {
            return Ok(None);
        };
        let session: ChatSession = serde_json::from_slice(&bytes)?;
        if session.id != session_id {
            return Err(ChatSessionStoreError::MismatchedSessionId {
                requested: session_id.to_string(),
                stored: session.id,
            });
        }
        let byte_len = bytes.len();
        Ok(Some((session, byte_len)))
    }

    pub fn save(&self, session: &ChatSession) -> Result<(), ChatSessionStoreError> {
        self.save_with_limits(session, MAX_SESSION_FILES, MAX_AGGREGATE_SESSION_BYTES)
    }

    fn save_with_limits(
        &self,
        session: &ChatSession,
        max_files: usize,
        max_aggregate_bytes: usize,
    ) -> Result<(), ChatSessionStoreError> {
        let file_name = session_file_name(&session.id)?;
        let bytes = persisted_session_bytes(session)?;
        if bytes.len() > MAX_SESSION_BYTES {
            return Err(ChatSessionStoreError::TooLarge);
        }
        if bytes.len() > max_aggregate_bytes {
            return Err(ChatSessionStoreError::AggregateTooLarge);
        }
        // Sizes come from directory metadata: a save never reads the other
        // sessions' contents.
        let files = self.session_file_sizes(max_files)?;
        let target_exists = files
            .iter()
            .any(|(existing, _len)| existing.to_string_lossy() == file_name);
        if !target_exists && files.len() >= max_files {
            return Err(ChatSessionStoreError::TooManySessions);
        }
        let mut aggregate_bytes = bytes.len() as u64;
        for (existing, len) in files {
            let existing = existing
                .into_string()
                .map_err(|_| ChatSessionStoreError::InvalidSessionId("non-UTF-8 leaf".into()))?;
            if existing == file_name || !existing.ends_with(".json") {
                continue;
            }
            aggregate_bytes = aggregate_bytes
                .checked_add(len)
                .ok_or(ChatSessionStoreError::AggregateTooLarge)?;
            if aggregate_bytes > max_aggregate_bytes as u64 {
                return Err(ChatSessionStoreError::AggregateTooLarge);
            }
        }
        self.root.write_chat_session_atomic(&file_name, &bytes)?;
        Ok(())
    }

    /// Permanently remove one persisted session. The unlink is atomic; a
    /// session that was never saved is already absent. Callers own the policy
    /// of which sessions may be deleted (for example not one with a running
    /// turn).
    pub fn delete(&self, session_id: &str) -> Result<(), ChatSessionStoreError> {
        let file_name = session_file_name(session_id)?;
        self.root.remove_chat_session(&file_name)?;
        Ok(())
    }

    pub fn list(&self) -> Result<Vec<ChatSession>, ChatSessionStoreError> {
        self.list_with_limits(MAX_SESSION_FILES, MAX_AGGREGATE_SESSION_BYTES)
    }

    fn list_with_limits(
        &self,
        max_files: usize,
        max_aggregate_bytes: usize,
    ) -> Result<Vec<ChatSession>, ChatSessionStoreError> {
        let mut sessions = Vec::new();
        let mut aggregate_bytes = 0usize;
        let files = self.session_files(max_files)?;
        for file_name in files {
            let file_name = file_name
                .into_string()
                .map_err(|_| ChatSessionStoreError::InvalidSessionId("non-UTF-8 leaf".into()))?;
            let Some(session_id) = file_name.strip_suffix(".json") else {
                continue;
            };
            validate_session_id(session_id)?;
            if let Some((session, byte_len)) = self.load_with_size(session_id)? {
                aggregate_bytes = aggregate_bytes
                    .checked_add(byte_len)
                    .ok_or(ChatSessionStoreError::AggregateTooLarge)?;
                if aggregate_bytes > max_aggregate_bytes {
                    return Err(ChatSessionStoreError::AggregateTooLarge);
                }
                sessions.push(session);
            }
        }
        sessions.sort_by(|left, right| {
            right
                .created_at
                .cmp(&left.created_at)
                .then_with(|| right.id.cmp(&left.id))
        });
        Ok(sessions)
    }

    fn session_files(
        &self,
        max_files: usize,
    ) -> Result<Vec<std::ffi::OsString>, ChatSessionStoreError> {
        self.root
            .list_chat_session_files(max_files)
            .map_err(map_listing_error)
    }

    fn session_file_sizes(
        &self,
        max_files: usize,
    ) -> Result<Vec<(std::ffi::OsString, u64)>, ChatSessionStoreError> {
        self.root
            .list_chat_session_file_sizes(max_files)
            .map_err(map_listing_error)
    }
}

fn map_listing_error(error: ProjectError) -> ChatSessionStoreError {
    if error.to_string().contains("entry limit") {
        ChatSessionStoreError::TooManySessions
    } else {
        ChatSessionStoreError::Project(error)
    }
}

/// Compact on-disk form of a session.
///
/// * A message with `blocks` is reloaded by re-deriving its flat compatibility
///   fields (`content`, `toolCalls`, `toolCallId`, `toolIsError`) from those
///   blocks, so persisting them only duplicated every tool result.
/// * Base64 images in tool results are replaced by a short text note. They
///   are display copies or one-turn model context; the model can inspect the
///   media again. The live in-memory session keeps them.
///
/// Older files that still carry these fields load unchanged.
fn persisted_session_bytes(session: &ChatSession) -> Result<Vec<u8>, serde_json::Error> {
    let mut value = serde_json::to_value(session)?;
    if let Some(messages) = value.get_mut("messages").and_then(Value::as_array_mut) {
        for message in messages {
            let Some(message) = message.as_object_mut() else {
                continue;
            };
            let Some(blocks) = message.get_mut("blocks").and_then(Value::as_array_mut) else {
                continue;
            };
            for block in blocks.iter_mut() {
                omit_block_images(block);
            }
            for derived in ["content", "toolCalls", "toolCallId", "toolIsError"] {
                message.remove(derived);
            }
        }
    }
    serde_json::to_vec(&value)
}

fn omit_block_images(block: &mut Value) {
    let content = match block.get("type").and_then(Value::as_str) {
        Some("toolUse") => block
            .get_mut("result")
            .and_then(|result| result.get_mut("content")),
        Some("toolResult") => block.get_mut("content"),
        _ => None,
    };
    let Some(content) = content.and_then(Value::as_array_mut) else {
        return;
    };
    for item in content {
        if item.get("kind").and_then(Value::as_str) != Some("image") {
            continue;
        }
        let media_type = item
            .get("mediaType")
            .and_then(Value::as_str)
            .filter(|media_type| {
                media_type.len() <= 64
                    && media_type
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || b"/+-.".contains(&byte))
            })
            .unwrap_or("image");
        let byte_len = item
            .get("base64")
            .and_then(Value::as_str)
            .map(|base64| base64.trim_end_matches('=').len() * 3 / 4)
            .unwrap_or(0);
        *item = serde_json::json!({
            "kind": "text",
            "text": format!("{IMAGE_OMITTED_PREFIX} ({media_type}, {byte_len} bytes).]"),
        });
    }
}

fn session_file_name(session_id: &str) -> Result<String, ChatSessionStoreError> {
    validate_session_id(session_id)?;
    Ok(format!("{session_id}.json"))
}

fn validate_session_id(session_id: &str) -> Result<(), ChatSessionStoreError> {
    if session_id.is_empty()
        || session_id.len() > 128
        || !session_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(ChatSessionStoreError::InvalidSessionId(
            session_id.to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::{ChatMessage, ChatSession, ToolCall};
    use crate::tools::result::Block;

    #[test]
    fn session_round_trips_atomically_and_lists_newest_first() {
        let temp = tempfile::tempdir().unwrap();
        let bundle = temp.path().join("Chat.opentake");
        std::fs::create_dir(&bundle).unwrap();
        let store = ChatSessionStore::open(&bundle).unwrap();

        let mut older = ChatSession::new("chat-older");
        older.created_at = 10;
        older.messages.push(ChatMessage::user("first"));
        store.save(&older).unwrap();
        let mut newer = ChatSession::new("chat-newer");
        newer.created_at = 20;
        newer.messages.push(ChatMessage::user("second"));
        store.save(&newer).unwrap();

        assert_eq!(store.load("chat-older").unwrap().unwrap().messages.len(), 1);
        assert_eq!(
            store
                .list()
                .unwrap()
                .into_iter()
                .map(|session| session.id)
                .collect::<Vec<_>>(),
            ["chat-newer", "chat-older"]
        );
        let entries = std::fs::read_dir(bundle.join("chat-sessions"))
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect::<Vec<_>>();
        assert_eq!(entries.len(), 2);
        assert!(entries
            .iter()
            .all(|name| !name.to_string_lossy().contains(".tmp")));
    }

    fn temp_store() -> (tempfile::TempDir, std::path::PathBuf, ChatSessionStore) {
        let temp = tempfile::tempdir().unwrap();
        let bundle = temp.path().join("Chat.opentake");
        std::fs::create_dir(&bundle).unwrap();
        let store = ChatSessionStore::open(&bundle).unwrap();
        (temp, bundle, store)
    }

    fn codex_image_call(index: usize, base64: &str) -> ToolCall {
        let mut call = ToolCall::request(
            format!("call-{index}"),
            "inspect_timeline",
            serde_json::json!({}),
        );
        call.is_error = Some(false);
        call.result = Some(serde_json::json!({
            "content": [
                { "kind": "text", "text": format!("frame {index}") },
                { "kind": "image", "base64": base64, "mediaType": "image/jpeg" }
            ]
        }));
        call
    }

    #[test]
    fn codex_image_results_are_saved_once_as_compact_placeholders() {
        let (_temp, bundle, store) = temp_store();
        let image = "A".repeat(300 * 1024);
        let mut session = ChatSession::new("images");
        for index in 0..50 {
            session.messages.push(ChatMessage::assistant(
                "",
                vec![codex_image_call(index, &image)],
            ));
        }
        store.save(&session).unwrap();

        let on_disk = std::fs::read(bundle.join("chat-sessions/images.json")).unwrap();
        assert!(on_disk.len() < 64 * 1024, "{} bytes", on_disk.len());
        let text = String::from_utf8(on_disk).unwrap();
        assert!(!text.contains("AAAA"));
        assert!(!text.contains("toolCalls"));

        let loaded = store.load("images").unwrap().unwrap();
        assert_eq!(loaded.messages.len(), 50);
        let message = &loaded.messages[7];
        assert_eq!(message.tool_calls.len(), 1, "legacy fields are re-derived");
        let result = message.tool_calls[0].result.as_ref().unwrap();
        assert_eq!(result["content"][0]["text"], "frame 7");
        assert_eq!(result["content"][1]["kind"], "text");
        let note = result["content"][1]["text"].as_str().unwrap();
        assert!(note.starts_with(IMAGE_OMITTED_PREFIX), "{note}");
        assert!(note.contains("image/jpeg, 230400 bytes"), "{note}");
    }

    #[test]
    fn byok_tool_result_images_are_replaced_and_tool_messages_reload() {
        let (_temp, _bundle, store) = temp_store();
        let mut session = ChatSession::new("byok");
        let mut tool = ChatMessage::user("");
        tool.role = crate::chat::Role::Tool;
        tool.blocks = vec![crate::chat::AgentContentBlock::ToolResult {
            tool_use_id: "call-1".into(),
            content: vec![
                Block::text("{\"frames\":1}"),
                Block::image("iVBORw0KGgo=", "image/png"),
            ],
            is_error: Some(false),
        }];
        tool.refresh_legacy_fields();
        session.messages.push(tool);
        store.save(&session).unwrap();

        let loaded = store.load("byok").unwrap().unwrap();
        let message = &loaded.messages[0];
        assert_eq!(message.tool_call_id.as_deref(), Some("call-1"));
        assert_eq!(message.tool_is_error, Some(false));
        let crate::chat::AgentContentBlock::ToolResult { content, .. } = &message.blocks[0] else {
            panic!("expected a tool result block");
        };
        assert_eq!(content[0], Block::text("{\"frames\":1}"));
        assert!(
            matches!(&content[1], Block::Text { text } if text.starts_with(IMAGE_OMITTED_PREFIX))
        );
    }

    #[test]
    fn legacy_pretty_files_with_tool_calls_and_images_still_load_and_migrate() {
        let (_temp, bundle, store) = temp_store();
        let legacy = serde_json::json!({
            "id": "legacy",
            "createdAt": 5,
            "isOpen": false,
            "messages": [{
                "id": "m1",
                "role": "assistant",
                "content": "Here is the frame.",
                "toolCalls": [{
                    "id": "call-1",
                    "name": "inspect_timeline",
                    "args": {},
                    "result": { "content": [
                        { "kind": "image", "base64": "iVBORw0KGgo=", "mediaType": "image/png" }
                    ] },
                    "isError": false
                }],
                "blocks": [
                    { "type": "text", "text": "Here is the frame." },
                    {
                        "type": "toolUse",
                        "id": "call-1",
                        "name": "inspect_timeline",
                        "input": {},
                        "result": { "content": [
                            { "kind": "image", "base64": "iVBORw0KGgo=", "mediaType": "image/png" }
                        ] },
                        "isError": false
                    }
                ],
                "createdAt": 6
            }]
        });
        std::fs::create_dir(bundle.join("chat-sessions")).unwrap();
        std::fs::write(
            bundle.join("chat-sessions/legacy.json"),
            serde_json::to_vec_pretty(&legacy).unwrap(),
        )
        .unwrap();

        let loaded = store.load("legacy").unwrap().unwrap();
        assert!(!loaded.is_open);
        assert_eq!(loaded.messages[0].content, "Here is the frame.");
        assert_eq!(
            loaded.messages[0].tool_calls[0].result.as_ref().unwrap()["content"][0]["kind"],
            "image"
        );

        store.save(&loaded).unwrap();
        let migrated = store.load("legacy").unwrap().unwrap();
        assert_eq!(migrated.messages[0].content, "Here is the frame.");
        assert_eq!(migrated.messages[0].blocks.len(), 2);
        assert_eq!(
            migrated.messages[0].tool_calls[0].result.as_ref().unwrap()["content"][0]["kind"],
            "text"
        );
    }

    #[test]
    fn delete_removes_a_session_and_restores_room_under_the_aggregate_limit() {
        let (_temp, _bundle, store) = temp_store();
        let mut old = ChatSession::new("old");
        old.is_open = false;
        old.messages.push(ChatMessage::user("x".repeat(600)));
        store.save_with_limits(&old, 8, 1024).unwrap();

        let mut current = ChatSession::new("current");
        current.messages.push(ChatMessage::user("y".repeat(600)));
        let error = store
            .save_with_limits(&current, 8, 1024)
            .expect_err("aggregate limit");
        assert!(matches!(error, ChatSessionStoreError::AggregateTooLarge));
        assert!(error.to_string().contains("delete old Agent chats"));

        store.delete("old").unwrap();
        assert!(store.load("old").unwrap().is_none());
        store.save_with_limits(&current, 8, 1024).unwrap();
        assert_eq!(
            store
                .list()
                .unwrap()
                .into_iter()
                .map(|session| session.id)
                .collect::<Vec<_>>(),
            ["current"]
        );
        store.delete("never-saved").unwrap();
        assert!(matches!(
            store.delete("../current"),
            Err(ChatSessionStoreError::InvalidSessionId(_))
        ));
    }

    #[cfg(unix)]
    #[test]
    fn saving_counts_other_sessions_from_metadata_without_reading_them() {
        use std::os::unix::fs::PermissionsExt;

        let (_temp, bundle, store) = temp_store();
        store.save(&ChatSession::new("other")).unwrap();
        let other = bundle.join("chat-sessions/other.json");
        let other_len = std::fs::metadata(&other).unwrap().len() as usize;
        std::fs::set_permissions(&other, std::fs::Permissions::from_mode(0o000)).unwrap();
        if std::fs::File::open(&other).is_ok() {
            // Privileged users bypass file modes; nothing to observe here.
            return;
        }

        let current = ChatSession::new("current");
        let current_len = persisted_session_bytes(&current).unwrap().len();
        store
            .save_with_limits(&current, 8, current_len + other_len)
            .expect("an unreadable sibling is sized from metadata");
        assert!(matches!(
            store.save_with_limits(&current, 8, current_len + other_len - 1),
            Err(ChatSessionStoreError::AggregateTooLarge)
        ));
        std::fs::set_permissions(&other, std::fs::Permissions::from_mode(0o600)).unwrap();
    }

    #[test]
    fn rejects_traversal_and_a_mismatched_persisted_id() {
        let temp = tempfile::tempdir().unwrap();
        let bundle = temp.path().join("Chat.opentake");
        std::fs::create_dir(&bundle).unwrap();
        let store = ChatSessionStore::open(&bundle).unwrap();

        assert!(matches!(
            store.load("../escape"),
            Err(ChatSessionStoreError::InvalidSessionId(_))
        ));
        store.save(&ChatSession::new("safe")).unwrap();
        let path = bundle.join("chat-sessions/safe.json");
        let mut wrong = ChatSession::new("other");
        wrong.created_at = 1;
        std::fs::write(path, serde_json::to_vec(&wrong).unwrap()).unwrap();
        assert!(matches!(
            store.load("safe"),
            Err(ChatSessionStoreError::MismatchedSessionId { .. })
        ));
    }

    #[cfg(unix)]
    #[test]
    fn refuses_a_symlinked_chat_sessions_directory() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let bundle = temp.path().join("Chat.opentake");
        let outside = temp.path().join("outside");
        std::fs::create_dir(&bundle).unwrap();
        std::fs::create_dir(&outside).unwrap();
        symlink(&outside, bundle.join("chat-sessions")).unwrap();
        let store = ChatSessionStore::open(&bundle).unwrap();

        assert!(store.save(&ChatSession::new("safe")).is_err());
        assert!(std::fs::read_dir(outside).unwrap().next().is_none());
    }

    #[test]
    fn listing_enforces_file_and_aggregate_limits() {
        let temp = tempfile::tempdir().unwrap();
        let bundle = temp.path().join("Chat.opentake");
        std::fs::create_dir(&bundle).unwrap();
        let store = ChatSessionStore::open(&bundle).unwrap();
        store.save(&ChatSession::new("one")).unwrap();
        store.save(&ChatSession::new("two")).unwrap();

        assert!(matches!(
            store.list_with_limits(1, MAX_AGGREGATE_SESSION_BYTES),
            Err(ChatSessionStoreError::TooManySessions)
        ));
        assert!(matches!(
            store.list_with_limits(MAX_SESSION_FILES, 1),
            Err(ChatSessionStoreError::AggregateTooLarge)
        ));
    }

    #[test]
    fn successful_saves_always_remain_listable_with_the_same_limits() {
        let temp = tempfile::tempdir().unwrap();
        let bundle = temp.path().join("Chat.opentake");
        std::fs::create_dir(&bundle).unwrap();
        let store = ChatSessionStore::open(&bundle).unwrap();
        let mut first = ChatSession::new("one");
        first.messages.push(ChatMessage::user("small"));
        store.save_with_limits(&first, 1, 1024).unwrap();
        assert_eq!(store.list_with_limits(1, 1024).unwrap().len(), 1);

        assert!(matches!(
            store.save_with_limits(&ChatSession::new("two"), 1, 1024),
            Err(ChatSessionStoreError::TooManySessions)
        ));
        first.messages.push(ChatMessage::user("replacement"));
        store.save_with_limits(&first, 1, 1024).unwrap();
        assert_eq!(store.list_with_limits(1, 1024).unwrap().len(), 1);

        assert!(matches!(
            store.save_with_limits(&first, 1, 1),
            Err(ChatSessionStoreError::AggregateTooLarge)
        ));
        assert_eq!(store.list_with_limits(1, 1024).unwrap().len(), 1);
    }
}
