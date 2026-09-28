//! Short-id system: outbound shortening / inbound expansion (`agent-SPEC.md`
//! §3). 1:1 port of upstream `ToolExecutor+ShortId.swift`.
//!
//! Entity ids are full UUIDs (~36 chars) and dominate large `get_timeline` /
//! `get_transcript` payloads. We emit the shortest project-unique prefix
//! (≥ 8 chars) and accept any prefix back: tools always run on full ids
//! (resolved on input), and every text response has its known ids shortened on
//! the way out. The system prompt instructs the model to pass prefixes back
//! verbatim (`prompt::base`).
//!
//! The id universe is kept sorted ([`IdUniverse`]) so every query is a binary
//! search: the id sharing the longest prefix with an id is one of its sorted
//! neighbours, and the ids starting with a prefix form one contiguous run.

use std::collections::HashMap;
use std::sync::OnceLock;

use opentake_domain::{MediaManifest, Timeline};
use regex::Regex;

use crate::tools::errors::ToolError;
use crate::tools::result::{Block, ToolResult};

/// Minimum prefix length (upstream `idPrefixFloor`).
const ID_PREFIX_FLOOR: usize = 8;

/// Scalar argument keys whose string value is an id prefix to expand
/// (upstream `scalarIdKeys` plus OpenTake-only tool keys).
const SCALAR_ID_KEYS: &[&str] = &[
    "clipId",
    "sourceClipId",
    "mediaRef",
    "startFrameMediaRef",
    "endFrameMediaRef",
    "sourceVideoMediaRef",
    "videoSourceMediaRef",
    "folderId",
    "parentFolderId",
    "beatClipId",
    "beatMediaRef",
    "referenceMediaRef",
    "narrationMediaRef",
    "portraitMediaRef",
    "audioMediaRef",
    "referenceAudioMediaRef",
];

/// Array argument keys whose string elements are id prefixes to expand
/// (upstream `arrayIdKeys` plus OpenTake-only tool keys).
const ARRAY_ID_KEYS: &[&str] = &[
    "clipIds",
    "assetIds",
    "folderIds",
    "referenceMediaRefs",
    "referenceImageMediaRefs",
    "referenceVideoMediaRefs",
    "referenceAudioMediaRefs",
    "captionClipIds",
];

fn uuid_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}")
            .expect("valid uuid regex")
    })
}

/// Every entity id the agent can see or name back, sorted (byte order) and
/// deduplicated so each short-id query is a binary search instead of a scan of
/// the whole set. One universe serves both directions (upstream
/// `currentIdUniverse`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IdUniverse {
    sorted: Vec<String>,
}

impl IdUniverse {
    /// Build a universe from any id collection; duplicates collapse.
    pub fn new(ids: impl IntoIterator<Item = String>) -> Self {
        let mut sorted: Vec<String> = ids.into_iter().collect();
        sorted.sort_unstable();
        sorted.dedup();
        Self { sorted }
    }

    /// Number of distinct ids.
    pub fn len(&self) -> usize {
        self.sorted.len()
    }

    /// Whether the universe holds no ids.
    pub fn is_empty(&self) -> bool {
        self.sorted.is_empty()
    }

    /// Whether `id` is a known id.
    pub fn contains(&self, id: &str) -> bool {
        self.position(id).is_some()
    }

    /// The shortest prefix (≥ 8 chars, or the whole id when shorter) of the
    /// known `id` that no other id shares; `None` for an unknown id.
    pub fn short_id(&self, id: &str) -> Option<&str> {
        self.position(id).map(|index| self.short_id_at(index))
    }

    fn position(&self, id: &str) -> Option<usize> {
        self.sorted
            .binary_search_by(|probe| probe.as_str().cmp(id))
            .ok()
    }

    /// Upstream `shortIdMap` for one id: the longest prefix it shares with any
    /// other id is the longest it shares with a sorted neighbour, so the answer
    /// is `min(len, max(floor, shared + 1))` chars. Byte order maps to the
    /// same neighbours for char counts because UTF-8 preserves code-point
    /// order and a longer common byte prefix never holds fewer common chars.
    fn short_id_at(&self, index: usize) -> &str {
        let id = self.sorted[index].as_str();
        let previous = index.checked_sub(1).map(|i| self.sorted[i].as_str());
        let next = self.sorted.get(index + 1).map(String::as_str);
        let shared = [previous, next]
            .into_iter()
            .flatten()
            .map(|other| common_prefix_chars(id, other))
            .max()
            .unwrap_or(0);
        prefix_chars(id, ID_PREFIX_FLOOR.max(shared + 1))
    }

    /// Every id starting with `prefix`: one contiguous run of the sorted ids,
    /// beginning with `prefix` itself when it is a known id.
    fn with_prefix(&self, prefix: &str) -> &[String] {
        let start = self.sorted.partition_point(|id| id.as_str() < prefix);
        let len = self.sorted[start..].partition_point(|id| id.starts_with(prefix));
        &self.sorted[start..start + len]
    }
}

/// Every entity id the agent can see or name back, collected from the timeline
/// and media manifest. One universe serves both directions (upstream
/// `currentIdUniverse`). The signal/context layer feeds the live timeline and
/// manifest in here.
pub fn current_id_universe(timeline: &Timeline, manifest: &MediaManifest) -> IdUniverse {
    let mut ids = Vec::new();
    for track in &timeline.tracks {
        if !track.id.is_empty() {
            ids.push(track.id.clone());
        }
        for clip in &track.clips {
            if !clip.id.is_empty() {
                ids.push(clip.id.clone());
            }
            if let Some(g) = &clip.caption_group_id {
                ids.push(g.clone());
            }
            if let Some(g) = &clip.link_group_id {
                ids.push(g.clone());
            }
        }
    }
    for entry in &manifest.entries {
        if !entry.id.is_empty() {
            ids.push(entry.id.clone());
        }
    }
    for folder in &manifest.folders {
        if !folder.id.is_empty() {
            ids.push(folder.id.clone());
        }
    }
    IdUniverse::new(ids)
}

/// Map each id to its shortest prefix (≥ 8 chars) that no other id shares.
/// Same output as upstream `shortIdMap`, in `O(N·L)` over the sorted universe.
/// Lengths count chars, so a non-ASCII id (should not happen) never splits a
/// code point.
pub fn short_id_map(universe: &IdUniverse) -> HashMap<String, String> {
    (0..universe.len())
        .map(|index| {
            (
                universe.sorted[index].clone(),
                universe.short_id_at(index).to_string(),
            )
        })
        .collect()
}

/// Number of leading chars `a` and `b` share.
fn common_prefix_chars(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

/// The first `n` chars of `s`, or all of `s` when it is shorter.
fn prefix_chars(s: &str, n: usize) -> &str {
    s.char_indices().nth(n).map_or(s, |(end, _)| &s[..end])
}

/// Replace every known full UUID in the result's text blocks with its short
/// prefix. Unknown UUIDs (e.g. embedded in a filename) pass through untouched.
/// 1:1 port of `shorteningIds`. Done on the post-run state so newly created ids
/// in summaries are shortened too (`agent-SPEC.md` §3.3). `universe` is only
/// built when a text block contains a UUID, and only the UUIDs found are
/// looked up.
pub fn shorten_ids(result: ToolResult, universe: impl FnOnce() -> IdUniverse) -> ToolResult {
    let re = uuid_regex();
    let has_uuid = result
        .content
        .iter()
        .any(|block| matches!(block, Block::Text { text } if re.is_match(text)));
    if !has_uuid {
        return result;
    }
    let universe = universe();
    let content = result
        .content
        .into_iter()
        .map(|block| match block {
            Block::Text { text } => {
                let replaced = re
                    .replace_all(&text, |caps: &regex::Captures<'_>| {
                        let m = caps.get(0).expect("group 0").as_str();
                        universe.short_id(m).unwrap_or(m).to_string()
                    })
                    .into_owned();
                Block::Text { text: replaced }
            }
            other => other,
        })
        .collect();
    ToolResult {
        content,
        is_error: result.is_error,
        llm_error: result.llm_error,
    }
}

/// Expand id-prefix arguments back to full ids before a tool runs. Throws on an
/// ambiguous prefix; leaves unknown values untouched so the tool emits its own
/// not-found error. 1:1 port of `expandingIdPrefixes`. Recurses through nested
/// objects/arrays so `entries[].mediaRef`, `moves[].clipId` etc. are covered.
pub fn expand_id_prefixes(
    args: &serde_json::Value,
    universe: &IdUniverse,
) -> Result<serde_json::Value, ToolError> {
    expand_value(args, universe)
}

fn expand_value(
    value: &serde_json::Value,
    universe: &IdUniverse,
) -> Result<serde_json::Value, ToolError> {
    match value {
        serde_json::Value::Object(map) => {
            let mut out = serde_json::Map::with_capacity(map.len());
            for (key, v) in map {
                let new_v = if SCALAR_ID_KEYS.contains(&key.as_str()) {
                    if let serde_json::Value::String(s) = v {
                        serde_json::Value::String(expand_one(s, universe)?)
                    } else {
                        expand_value(v, universe)?
                    }
                } else if ARRAY_ID_KEYS.contains(&key.as_str()) {
                    if let serde_json::Value::Array(arr) = v {
                        let mut new_arr = Vec::with_capacity(arr.len());
                        for el in arr {
                            if let serde_json::Value::String(s) = el {
                                new_arr.push(serde_json::Value::String(expand_one(s, universe)?));
                            } else {
                                new_arr.push(expand_value(el, universe)?);
                            }
                        }
                        serde_json::Value::Array(new_arr)
                    } else {
                        expand_value(v, universe)?
                    }
                } else {
                    expand_value(v, universe)?
                };
                out.insert(key.clone(), new_v);
            }
            Ok(serde_json::Value::Object(out))
        }
        serde_json::Value::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for el in arr {
                out.push(expand_value(el, universe)?);
            }
            Ok(serde_json::Value::Array(out))
        }
        other => Ok(other.clone()),
    }
}

/// Expand one prefix: full id passes through, a unique prefix resolves, an
/// unknown value passes through (tool reports not-found), an ambiguous prefix
/// errors. 1:1 port of `expandOne`, as two binary searches.
fn expand_one(reference: &str, universe: &IdUniverse) -> Result<String, ToolError> {
    let matches = universe.with_prefix(reference);
    // A known id sorts first among the ids it prefixes.
    if matches.first().is_some_and(|first| first == reference) {
        return Ok(reference.to_string());
    }
    match matches {
        [only] => Ok(only.clone()),
        [] => Ok(reference.to_string()),
        _ => Err(ToolError::new(format!(
            "Ambiguous id '{reference}' matches {} items; re-read with get_timeline or get_media for current ids.",
            matches.len()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::time::{Duration, Instant};

    // Two UUIDs sharing the first 12 chars, then diverging at index 14/15.
    const A: &str = "11111111-1111-aaaa-0000-000000000000";
    const B: &str = "11111111-1111-bbbb-0000-000000000000";
    // A UUID unique from the very floor.
    const C: &str = "22222222-2222-2222-2222-222222222222";

    fn universe(items: &[&str]) -> IdUniverse {
        IdUniverse::new(items.iter().map(|s| s.to_string()))
    }

    #[test]
    fn unique_id_shortens_to_floor() {
        let u = universe(&[C]);
        let map = short_id_map(&u);
        assert_eq!(map[C], &C[..8]); // exactly the 8-char floor
    }

    #[test]
    fn shared_prefix_extends_until_unique() {
        // A and B share "11111111-1111-" (14 chars), diverge at index 14
        // ('a' vs 'b'). Shortest unique prefix is 15 chars.
        let u = universe(&[A, B]);
        let map = short_id_map(&u);
        assert_eq!(map[A], &A[..15]);
        assert_eq!(map[B], &B[..15]);
        assert_ne!(map[A], map[B]);
    }

    #[test]
    fn expand_unique_prefix_resolves_to_full() {
        let u = universe(&[C]);
        let got = expand_one(&C[..10], &u).unwrap();
        assert_eq!(got, C);
    }

    #[test]
    fn expand_full_id_passes_through() {
        let u = universe(&[C]);
        assert_eq!(expand_one(C, &u).unwrap(), C);
    }

    #[test]
    fn expand_unknown_passes_through() {
        let u = universe(&[C]);
        let got = expand_one("ffffffff", &u).unwrap();
        assert_eq!(got, "ffffffff"); // tool reports not-found itself
    }

    #[test]
    fn expand_ambiguous_prefix_errors() {
        let u = universe(&[A, B]);
        // "11111111" matches both A and B.
        let err = expand_one("11111111", &u).unwrap_err();
        assert!(
            err.message.contains("Ambiguous id '11111111'"),
            "{}",
            err.message
        );
        assert!(err.message.contains("matches 2 items"), "{}", err.message);
    }

    #[test]
    fn shorten_replaces_known_uuid_in_text() {
        let u = universe(&[C]);
        let r = ToolResult::ok(format!("clip {C} added"));
        let out = shorten_ids(r, || u);
        assert_eq!(out.text_joined(), format!("clip {} added", &C[..8]));
    }

    #[test]
    fn shorten_leaves_unknown_uuid_untouched() {
        // A filename-embedded UUID not in the universe.
        let other = "99999999-9999-9999-9999-999999999999";
        let u = universe(&[C]);
        let r = ToolResult::ok(format!("file {other}.mp4"));
        let out = shorten_ids(r, || u);
        assert_eq!(out.text_joined(), format!("file {other}.mp4"));
    }

    #[test]
    fn expand_recurses_into_nested_entries() {
        let u = universe(&[C]);
        let args = serde_json::json!({
            "entries": [{"mediaRef": &C[..9], "startFrame": 0}]
        });
        let out = expand_id_prefixes(&args, &u).unwrap();
        assert_eq!(out["entries"][0]["mediaRef"], serde_json::json!(C));
        assert_eq!(out["entries"][0]["startFrame"], serde_json::json!(0));
    }

    #[test]
    fn expand_array_id_keys() {
        let u = universe(&[A, B]);
        let args = serde_json::json!({"clipIds": [&A[..15], &B[..15]]});
        let out = expand_id_prefixes(&args, &u).unwrap();
        assert_eq!(out["clipIds"][0], serde_json::json!(A));
        assert_eq!(out["clipIds"][1], serde_json::json!(B));
    }

    #[test]
    fn expand_ambiguous_in_array_errors() {
        let u = universe(&[A, B]);
        let args = serde_json::json!({"clipIds": ["11111111"]});
        let err = expand_id_prefixes(&args, &u).unwrap_err();
        assert!(err.message.contains("Ambiguous"), "{}", err.message);
    }

    #[test]
    fn universe_collects_from_timeline_and_manifest() {
        use opentake_domain::{Clip, ClipType, MediaFolder, MediaManifest, Timeline, Track};
        let mut tl = Timeline::new();
        let mut t = Track::new("track-1", ClipType::Video);
        let mut c = Clip::new("clip-1", "asset-1", 0, 30);
        c.link_group_id = Some("link-1".into());
        c.caption_group_id = Some("cap-1".into());
        t.clips.push(c);
        tl.tracks.push(t);
        let mut m = MediaManifest::new();
        m.folders.push(MediaFolder::new("folder-1", "B-Roll"));
        let ids = current_id_universe(&tl, &m);
        for want in ["track-1", "clip-1", "link-1", "cap-1", "folder-1"] {
            assert!(ids.contains(want), "missing {want}");
        }
    }

    #[test]
    fn expand_opentake_tool_keys_with_uuid_prefixes() {
        let u = universe(&[A, B, C]);
        for key in [
            "beatClipId",
            "beatMediaRef",
            "referenceMediaRef",
            "narrationMediaRef",
            "portraitMediaRef",
            "audioMediaRef",
            "referenceAudioMediaRef",
        ] {
            let out = expand_id_prefixes(&serde_json::json!({ key: &A[..15] }), &u).unwrap();
            assert_eq!(out[key], serde_json::json!(A), "{key}");
            let err = expand_id_prefixes(&serde_json::json!({ key: "11111111" }), &u).unwrap_err();
            assert!(err.message.contains("Ambiguous id"), "{key}");
        }
        let out = expand_id_prefixes(
            &serde_json::json!({"captionClipIds": [&A[..15], &B[..15]]}),
            &u,
        )
        .unwrap();
        assert_eq!(out["captionClipIds"], serde_json::json!([A, B]));
        let err = expand_id_prefixes(&serde_json::json!({"captionClipIds": ["11111111"]}), &u)
            .unwrap_err();
        assert!(err.message.contains("Ambiguous id"));

        let nested = serde_json::json!({"segments": [{"narrationMediaRef": &C[..9]}]});
        let out = expand_id_prefixes(&nested, &u).unwrap();
        assert_eq!(
            out["segments"][0]["narrationMediaRef"],
            serde_json::json!(C)
        );
    }

    /// Keys that end like an id but name something outside the timeline/media
    /// id universe, so outbound shortening never produces a prefix for them.
    const NON_UNIVERSE_ID_KEYS: &[&str] = &[
        "consentId",
        "voiceId",
        "templateId",
        "workflowId",
        "maskId",
        "documentId",
    ];

    fn allowed_keys_in(source: &str) -> Vec<String> {
        const MARKER: &str = "ALLOWED_KEYS: &'static [&'static str] = &[";
        let mut keys = Vec::new();
        for block in source.split(MARKER).skip(1) {
            let list = &block[..block.find(']').expect("closed ALLOWED_KEYS list")];
            keys.extend(
                list.split(',')
                    .map(|item| item.trim().trim_matches('"'))
                    .filter(|key| !key.is_empty())
                    .map(str::to_string),
            );
        }
        keys
    }

    #[test]
    fn every_id_shaped_tool_argument_key_is_expanded_or_exempt() {
        let keys: Vec<String> = [
            include_str!("args.rs"),
            include_str!("../mcp/motion_documents.rs"),
            include_str!("../mcp/dispatch.rs"),
        ]
        .into_iter()
        .flat_map(allowed_keys_in)
        .collect();
        assert!(keys.iter().any(|key| key == "beatClipId"));
        for key in &keys {
            let key = key.as_str();
            if NON_UNIVERSE_ID_KEYS.contains(&key) {
                continue;
            }
            if key.ends_with("Ids") || key.ends_with("MediaRefs") {
                assert!(
                    ARRAY_ID_KEYS.contains(&key),
                    "{key} missing from ARRAY_ID_KEYS"
                );
            } else if key.ends_with("Id") || key.ends_with("MediaRef") || key == "mediaRef" {
                assert!(
                    SCALAR_ID_KEYS.contains(&key),
                    "{key} missing from SCALAR_ID_KEYS"
                );
            }
        }
    }

    // MARK: - Equivalence with the previous quadratic implementation (#78)

    /// The previous `short_id_map` (a scan of the whole set per candidate
    /// prefix), kept verbatim as the oracle for the sorted implementation.
    fn short_id_map_oracle(ids: &HashSet<String>) -> HashMap<String, String> {
        let take_chars = |s: &str, n: usize| -> String { s.chars().take(n).collect() };
        let mut out = HashMap::with_capacity(ids.len());
        for id in ids {
            let char_len = id.chars().count();
            let mut len = ID_PREFIX_FLOOR.min(char_len);
            while len < char_len {
                let prefix = take_chars(id, len);
                let collides = ids
                    .iter()
                    .any(|other| other != id && other.starts_with(&prefix));
                if collides {
                    len += 1;
                } else {
                    break;
                }
            }
            out.insert(id.clone(), take_chars(id, len));
        }
        out
    }

    /// The previous linear `expand_one`, kept verbatim as the oracle.
    fn expand_one_oracle(reference: &str, universe: &HashSet<String>) -> Result<String, ToolError> {
        if universe.contains(reference) {
            return Ok(reference.to_string());
        }
        let matches: Vec<&String> = universe
            .iter()
            .filter(|id| id.starts_with(reference))
            .collect();
        match matches.len() {
            1 => Ok(matches[0].clone()),
            0 => Ok(reference.to_string()),
            n => Err(ToolError::new(format!(
                "Ambiguous id '{reference}' matches {n} items; re-read with get_timeline or get_media for current ids."
            ))),
        }
    }

    /// The previous `shorten_ids`, kept verbatim as the oracle.
    fn shorten_ids_oracle(result: ToolResult, ids: &HashSet<String>) -> ToolResult {
        let map = short_id_map_oracle(ids);
        if map.is_empty() {
            return result;
        }
        let re = uuid_regex();
        let content = result
            .content
            .into_iter()
            .map(|block| match block {
                Block::Text { text } => {
                    let replaced = re
                        .replace_all(&text, |caps: &regex::Captures<'_>| {
                            let m = caps.get(0).expect("group 0").as_str();
                            map.get(m).cloned().unwrap_or_else(|| m.to_string())
                        })
                        .into_owned();
                    Block::Text { text: replaced }
                }
                other => other,
            })
            .collect();
        ToolResult {
            content,
            is_error: result.is_error,
            llm_error: result.llm_error,
        }
    }

    /// Deterministic SplitMix64, so the randomized checks are reproducible.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }

        fn pick(&mut self, items: &[char]) -> char {
            items[self.below(items.len())]
        }
    }

    const HEX: &[char] = &[
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
    ];

    fn random_uuid(rng: &mut Rng) -> String {
        (0..36)
            .map(|i| {
                if matches!(i, 8 | 13 | 18 | 23) {
                    '-'
                } else {
                    rng.pick(HEX)
                }
            })
            .collect()
    }

    /// `base` with every char from index `at` on redrawn (dashes kept), so the
    /// two UUIDs share a long prefix.
    fn uuid_sibling(rng: &mut Rng, base: &str, at: usize) -> String {
        base.chars()
            .enumerate()
            .map(|(i, c)| if i < at || c == '-' { c } else { rng.pick(HEX) })
            .collect()
    }

    fn random_word(rng: &mut Rng, alphabet: &[char], max_len: usize) -> String {
        let len = rng.below(max_len + 1);
        (0..len).map(|_| rng.pick(alphabet)).collect()
    }

    /// A random id set mixing UUIDs with shared long prefixes, ids that prefix
    /// other ids, short and empty ids, non-UUID names and non-ASCII chars.
    fn random_ids(rng: &mut Rng) -> Vec<String> {
        let mut ids: Vec<String> = Vec::new();
        for _ in 0..rng.below(40) {
            let id = match rng.below(7) {
                0 => random_uuid(rng),
                1 if !ids.is_empty() => {
                    let base = ids[rng.below(ids.len())].clone();
                    if base.chars().count() == 36 {
                        let at = 8 + rng.below(28);
                        uuid_sibling(rng, &base, at)
                    } else {
                        random_uuid(rng)
                    }
                }
                2 => random_word(rng, &['a', 'b', '-'], 12),
                3 => random_word(rng, &['a', 'é', '日', '-'], 10),
                4 => format!("clip-{}", rng.below(120)),
                5 if !ids.is_empty() => {
                    // A strict prefix of an existing id.
                    let base = ids[rng.below(ids.len())].clone();
                    let keep = rng.below(base.chars().count() + 1);
                    base.chars().take(keep).collect()
                }
                _ if !ids.is_empty() => {
                    // An existing id extended by a few chars.
                    let base = ids[rng.below(ids.len())].clone();
                    base + &random_word(rng, &['a', '0', '-', 'é'], 4)
                }
                _ => random_uuid(rng),
            };
            ids.push(id);
        }
        ids
    }

    /// References to expand: every id, random prefixes of ids, random words
    /// and UUIDs (mostly unknown), and the empty string.
    fn random_references(rng: &mut Rng, ids: &[String]) -> Vec<String> {
        let mut refs = vec![String::new()];
        for id in ids {
            refs.push(id.clone());
            let keep = rng.below(id.chars().count() + 1);
            refs.push(id.chars().take(keep).collect());
        }
        for _ in 0..8 {
            refs.push(random_word(rng, &['a', 'b', '-', 'é'], 6));
            refs.push(random_uuid(rng));
        }
        refs
    }

    #[test]
    fn sorted_short_id_map_matches_the_quadratic_oracle() {
        let mut rng = Rng(0x5eed_0078);
        for round in 0..500 {
            let ids = random_ids(&mut rng);
            let set: HashSet<String> = ids.iter().cloned().collect();
            let universe = IdUniverse::new(ids);
            assert_eq!(universe.len(), set.len());
            assert_eq!(
                short_id_map(&universe),
                short_id_map_oracle(&set),
                "round {round}: {set:?}"
            );
        }
    }

    #[test]
    fn sorted_expand_one_matches_the_linear_oracle() {
        let mut rng = Rng(0x0e8a_2d78);
        let (mut unique, mut ambiguous, mut unknown) = (0, 0, 0);
        for round in 0..500 {
            let ids = random_ids(&mut rng);
            let set: HashSet<String> = ids.iter().cloned().collect();
            let universe = IdUniverse::new(ids.iter().cloned());
            for reference in random_references(&mut rng, &ids) {
                let got = expand_one(&reference, &universe).map_err(|e| e.message);
                let want = expand_one_oracle(&reference, &set).map_err(|e| e.message);
                assert_eq!(got, want, "round {round}: {reference:?} in {set:?}");
                match want {
                    Err(_) => ambiguous += 1,
                    Ok(full) if full != reference || set.contains(&reference) => unique += 1,
                    Ok(_) => unknown += 1,
                }
            }
        }
        // The generator must exercise all three outcomes.
        assert!(unique > 100 && ambiguous > 100 && unknown > 100);
    }

    #[test]
    fn sorted_shorten_ids_matches_the_oracle() {
        let mut rng = Rng(0x5401_7e17);
        for round in 0..300 {
            let mut ids = random_ids(&mut rng);
            // Give every set a few UUIDs that share long prefixes.
            let base = random_uuid(&mut rng);
            for _ in 0..3 {
                let at = 8 + rng.below(28);
                ids.push(uuid_sibling(&mut rng, &base, at));
            }
            let set: HashSet<String> = ids.iter().cloned().collect();
            let mut text = String::from("summary ");
            for id in &ids {
                text.push_str(id);
                text.push(' ');
            }
            // Unknown and differently cased UUIDs pass through untouched.
            text.push_str(&random_uuid(&mut rng));
            text.push(' ');
            text.push_str(&base.to_uppercase());
            let result = ToolResult::blocks(vec![
                Block::text(text),
                Block::image("AAAA", "image/png"),
                Block::text("no ids here"),
            ]);
            let got = shorten_ids(result.clone(), || IdUniverse::new(ids.iter().cloned()));
            assert_eq!(got, shorten_ids_oracle(result, &set), "round {round}");
        }
    }

    #[test]
    fn shorten_ids_skips_the_universe_when_no_text_has_a_uuid() {
        let result = ToolResult::blocks(vec![
            Block::text("Moved 2 clip(s): clip-1, clip-2"),
            Block::image("AAAA", "image/png"),
        ]);
        let out = shorten_ids(result.clone(), || {
            panic!("the id universe must not be built for a result without UUIDs")
        });
        assert_eq!(out, result);
    }

    #[test]
    fn expand_prefers_an_exact_id_that_prefixes_other_ids() {
        let u = universe(&["clip-1", "clip-10", "clip-11"]);
        assert_eq!(expand_one("clip-1", &u).unwrap(), "clip-1");
        let err = expand_one("clip", &u).unwrap_err();
        assert!(err.message.contains("matches 3 items"), "{}", err.message);
        assert_eq!(expand_one("clip-10", &u).unwrap(), "clip-10");
        assert_eq!(expand_one("clip-2", &u).unwrap(), "clip-2");
    }

    fn median(samples: &mut [Duration]) -> Duration {
        samples.sort();
        samples[samples.len() / 2]
    }

    #[test]
    #[ignore = "controlled release benchmark: cargo test --release -p opentake-agent --lib short_id_release_benchmark -- --ignored --nocapture"]
    #[allow(clippy::assertions_on_constants)]
    fn short_id_release_benchmark() {
        // Keep the benchmark discoverable in debug without accepting debug timings.
        assert!(!cfg!(debug_assertions), "run this benchmark with --release");
        let mut rng = Rng(0x1d5_0078);
        let ids: Vec<String> = (0..10_000).map(|_| random_uuid(&mut rng)).collect();
        let set: HashSet<String> = ids.iter().cloned().collect();
        let prefixes: Vec<&str> = ids[..1000].iter().map(|id| &id[..13]).collect();
        let text = ids[..10].join(" ");

        let start = Instant::now();
        let old_map = short_id_map_oracle(&set);
        let old_map_time = start.elapsed();
        let start = Instant::now();
        let old_shortened = shorten_ids_oracle(ToolResult::ok(text.clone()), &set);
        let old_shorten_time = start.elapsed();
        let start = Instant::now();
        let old_expanded: Vec<_> = prefixes
            .iter()
            .map(|prefix| expand_one_oracle(prefix, &set).unwrap())
            .collect();
        let old_expand_time = start.elapsed();

        let (mut map_times, mut shorten_times, mut expand_times) = (vec![], vec![], vec![]);
        for _ in 0..7 {
            let start = Instant::now();
            let map = short_id_map(&IdUniverse::new(ids.iter().cloned()));
            map_times.push(start.elapsed());
            assert_eq!(map, old_map);

            let start = Instant::now();
            let shortened = shorten_ids(ToolResult::ok(text.clone()), || {
                IdUniverse::new(ids.iter().cloned())
            });
            shorten_times.push(start.elapsed());
            assert_eq!(shortened, old_shortened);

            let universe = IdUniverse::new(ids.iter().cloned());
            let start = Instant::now();
            let expanded: Vec<_> = prefixes
                .iter()
                .map(|prefix| expand_one(prefix, &universe).unwrap())
                .collect();
            expand_times.push(start.elapsed());
            assert_eq!(expanded, old_expanded);
        }
        let map_time = median(&mut map_times);
        let shorten_time = median(&mut shorten_times);
        let expand_time = median(&mut expand_times);
        println!(
            "10000 UUIDs: short_id_map old={old_map_time:?} new={map_time:?}; \
             shorten_ids(10 UUIDs) old={old_shorten_time:?} new={shorten_time:?}; \
             expand 1000 prefixes old={old_expand_time:?} new={expand_time:?}"
        );
        assert!(
            map_time < Duration::from_millis(20),
            "short_id_map {map_time:?}"
        );
        assert!(
            shorten_time < Duration::from_millis(20),
            "shorten_ids {shorten_time:?}"
        );
        assert!(
            expand_time < Duration::from_millis(5),
            "expand {expand_time:?}"
        );
    }
}
