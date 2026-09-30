//! Cached, locally matched copies of the asset-protocol scope patterns.
//!
//! `Scope::allowed_patterns()` / `forbidden_patterns()` clone every pattern
//! under Tauri's mutex, and every imported file adds an allowed pattern, so
//! calling them per path check made authorization cost grow with the number
//! of imports (and quadratically for per-entry loops). The allowed set is
//! cloned once per scope change and matched locally: patterns that are escaped
//! literal paths (all exact-file grants) are looked up in a hash set, real
//! globs are still evaluated by `glob` with the protocol's match options.
//!
//! The forbidden set is small (revoked proxy files) and is cloned fresh for
//! every snapshot. Tauri delivers scope events asynchronously when another
//! emit is in progress, so an event-invalidated deny set could still allow a
//! path after `forbid_file` returned; a fresh deny set honours a revocation as
//! soon as it returns. A stale allowed set can only deny a newly granted path.

use super::*;
use glob::Pattern;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

/// Local matcher equivalent to [`scope_allows_lexical_path`] plus the exact
/// file grant test used by the Home thumbnail exception.
pub(crate) struct ScopeSnapshot {
    allowed: Arc<AllowedPatterns>,
    forbidden: PatternSet,
    native: Arc<crate::native_read_scope::NativeScopeSnapshot>,
}

/// The allowed half of a snapshot, shared between requests until the scope
/// changes.
struct AllowedPatterns {
    patterns: PatternSet,
    /// Allowed pattern texts, keyed like `scope_has_exact_file_grant` compares
    /// them (ASCII case-insensitively on Windows).
    exact: HashSet<String>,
}

impl AllowedPatterns {
    fn capture(scope: &Scope) -> Self {
        let allowed = scope.allowed_patterns();
        let exact = allowed
            .iter()
            .map(|pattern| exact_grant_key(pattern.as_str()))
            .collect();
        Self {
            patterns: PatternSet::new(allowed),
            exact,
        }
    }
}

impl ScopeSnapshot {
    /// Clone the scope's two pattern sets once, bypassing the cache.
    #[cfg(test)]
    pub(crate) fn capture(scope: &Scope) -> Self {
        Self {
            allowed: Arc::new(AllowedPatterns::capture(scope)),
            forbidden: PatternSet::new(scope.forbidden_patterns()),
            native: Arc::new(crate::native_read_scope::NativeScopeSnapshot::default()),
        }
    }

    /// Same decision as [`scope_allows_lexical_path`]: forbidden patterns take
    /// precedence, then any allowed pattern must match the normalized path.
    pub(crate) fn allows(&self, path: &Path) -> bool {
        let normalized: PathBuf = path.components().collect();
        !self.forbids(&normalized)
            && (self.allowed.patterns.matches(&normalized) || self.native.allows(&normalized))
    }

    pub(crate) fn forbids(&self, path: &Path) -> bool {
        let normalized: PathBuf = path.components().collect();
        self.native.forbids(&normalized)
            || self.forbidden.matches(&normalized)
            || (normalized.to_str().is_none() && self.forbidden.native_globs.is_none())
    }

    /// Same decision as `scope_has_exact_file_grant`: an allowed pattern whose
    /// text is exactly the escaped normalized path, not a directory glob.
    pub(crate) fn has_exact_file_grant(&self, path: &Path) -> bool {
        self.native.has_file(path)
            || normalized_path(path).to_str().is_some_and(|text| {
                self.allowed
                    .exact
                    .contains(&exact_grant_key(&Pattern::escape(text)))
            })
    }

    /// Test hook: whether two snapshots share one cached allowed set.
    #[cfg(test)]
    pub(crate) fn shares_allowed_patterns_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.allowed, &other.allowed)
    }
}

struct PatternSet {
    literals: HashSet<String>,
    globs: Vec<Pattern>,
    native_globs: Option<Vec<Pattern>>,
}

impl PatternSet {
    fn new(patterns: HashSet<Pattern>) -> Self {
        let mut literals = HashSet::new();
        let mut globs = Vec::new();
        for pattern in patterns {
            match literal_pattern_text(pattern.as_str()) {
                Some(literal) => {
                    literals.insert(literal_match_key(&literal));
                }
                None => globs.push(pattern),
            }
        }
        // Glob syntax/separators are ASCII. A native byte/unit per character
        // preserves Unicode literal prefixes without replacing invalid names.
        let native_globs = globs.iter().map(native_pattern).collect::<Option<Vec<_>>>();
        Self {
            literals,
            globs,
            native_globs,
        }
    }

    fn matches(&self, normalized: &Path) -> bool {
        let Some(text) = normalized.to_str() else {
            let native = PathBuf::from(native_match_text(normalized.as_os_str()));
            return self.native_globs.as_ref().is_some_and(|patterns| {
                patterns
                    .iter()
                    .any(|pattern| pattern.matches_path_with(&native, scope_match_options()))
            });
        };
        let options = scope_match_options();
        self.literals.contains(&literal_match_key(text))
            || self
                .globs
                .iter()
                .any(|pattern| pattern.matches_path_with(normalized, options))
    }
}

fn native_pattern(pattern: &Pattern) -> Option<Pattern> {
    // Unicode classes cannot be expanded into independent native units: [é]
    // would become a choice of its UTF-8 bytes rather than their sequence.
    // Recognize class boundaries exactly as glob does, including []] and [!]].
    let chars: Vec<_> = pattern.as_str().chars().collect();
    let mut index = 0;
    while index < chars.len() {
        if chars[index] != '[' {
            index += 1;
            continue;
        }
        let start = index + 1 + usize::from(chars.get(index + 1) == Some(&'!'));
        let end = start + 1 + chars.get(start + 1..)?.iter().position(|ch| *ch == ']')?;
        if chars[start..end].iter().any(|ch| {
            if cfg!(windows) {
                ch.len_utf16() > 1
            } else {
                ch.len_utf8() > 1
            }
        }) {
            return None;
        }
        index = end + 1;
    }
    Pattern::new(&native_match_text(std::ffi::OsStr::new(pattern.as_str()))).ok()
}

#[cfg(unix)]
fn native_match_text(text: &std::ffi::OsStr) -> String {
    use std::os::unix::ffi::OsStrExt;
    text.as_bytes()
        .iter()
        .map(|byte| char::from(*byte))
        .collect()
}

#[cfg(windows)]
fn native_match_text(text: &std::ffi::OsStr) -> String {
    use std::os::windows::ffi::OsStrExt;
    text.encode_wide()
        .map(|unit| {
            if unit < 128 {
                char::from(unit as u8)
            } else {
                char::from_u32(0x10000 + u32::from(unit))
                    .expect("mapped UTF-16 units are Unicode scalar values")
            }
        })
        .collect()
}

/// The literal path a pattern matches, when it contains no wildcard.
///
/// Mirrors `glob::Pattern::new`: `?`, `*` and any bracket expression are
/// wildcards, except the single-character classes `[?]`, `[*]`, `[[]` and
/// `[]]` that `Pattern::escape` emits, which match exactly that character.
/// Anything else is left to `glob` itself.
fn literal_pattern_text(pattern: &str) -> Option<String> {
    let chars = pattern.chars().collect::<Vec<_>>();
    let mut literal = String::with_capacity(pattern.len());
    let mut index = 0;
    while index < chars.len() {
        match chars[index] {
            '?' | '*' => return None,
            '[' => {
                let escaped = chars.get(index + 1).copied()?;
                if !matches!(escaped, '?' | '*' | '[' | ']') || chars.get(index + 2) != Some(&']') {
                    return None;
                }
                literal.push(escaped);
                index += 3;
            }
            character => {
                literal.push(character);
                index += 1;
            }
        }
    }
    Some(literal)
}

/// Canonical form under `glob`'s character equality for the protocol's match
/// options: on Windows separators are interchangeable and ASCII letters
/// compare case-insensitively; elsewhere comparison is exact.
#[cfg(target_os = "windows")]
fn literal_match_key(text: &str) -> String {
    text.chars()
        .map(|character| {
            if std::path::is_separator(character) {
                '\\'
            } else {
                character.to_ascii_lowercase()
            }
        })
        .collect()
}

#[cfg(not(target_os = "windows"))]
fn literal_match_key(text: &str) -> String {
    text.to_owned()
}

#[cfg(target_os = "windows")]
fn exact_grant_key(text: &str) -> String {
    text.to_ascii_lowercase()
}

#[cfg(not(target_os = "windows"))]
fn exact_grant_key(text: &str) -> String {
    text.to_owned()
}

/// Per-app cache of the allowed half of the asset-protocol scope snapshot.
///
/// Tauri scopes only grow (`allow_*` / `forbid_*`) and emit an event after
/// every change. A listener bumps `generation`; readers load the generation
/// before capturing, so allowed patterns are never cached under a generation
/// newer than the patterns they contain. A late event only delays a grant.
struct ScopeSnapshotCache {
    generation: Arc<AtomicU64>,
    cached: Mutex<Option<(u64, Arc<AllowedPatterns>)>>,
    captures: AtomicUsize,
}

static SCOPE_CACHE_CREATION: Mutex<()> = Mutex::new(());

fn scope_snapshot_cache<R: Runtime>(app: &AppHandle<R>) -> tauri::State<'_, ScopeSnapshotCache> {
    if let Some(cache) = app.try_state::<ScopeSnapshotCache>() {
        return cache;
    }
    let _creation = SCOPE_CACHE_CREATION
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if app.try_state::<ScopeSnapshotCache>().is_none() {
        let generation = Arc::new(AtomicU64::new(0));
        let listener_generation = generation.clone();
        app.asset_protocol_scope().listen(move |_| {
            listener_generation.fetch_add(1, Ordering::AcqRel);
        });
        app.manage(ScopeSnapshotCache {
            generation,
            cached: Mutex::new(None),
            captures: AtomicUsize::new(0),
        });
    }
    app.state::<ScopeSnapshotCache>()
}

/// The current asset-protocol scope as a local matcher. The large allowed
/// set is cloned only after the scope changed; the small forbidden set is
/// read fresh so that a revocation is honoured as soon as it returns.
pub(crate) fn asset_scope_snapshot<R: Runtime>(app: &AppHandle<R>) -> ScopeSnapshot {
    let scope = app.asset_protocol_scope();
    ScopeSnapshot {
        allowed: cached_allowed_patterns(app, &scope),
        forbidden: PatternSet::new(scope.forbidden_patterns()),
        native: crate::native_read_scope::snapshot(app),
    }
}

fn cached_allowed_patterns<R: Runtime>(app: &AppHandle<R>, scope: &Scope) -> Arc<AllowedPatterns> {
    let cache = scope_snapshot_cache(app);
    let generation = cache.generation.load(Ordering::Acquire);
    {
        let cached = cache
            .cached
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some((cached_generation, allowed)) = cached.as_ref() {
            if *cached_generation == generation {
                return allowed.clone();
            }
        }
    }
    let allowed = Arc::new(AllowedPatterns::capture(scope));
    cache.captures.fetch_add(1, Ordering::Relaxed);
    let mut cached = cache
        .cached
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if cached
        .as_ref()
        .is_none_or(|(cached_generation, _)| *cached_generation <= generation)
    {
        *cached = Some((generation, allowed.clone()));
    }
    allowed
}

/// Test hook: how many times this app's allowed pattern set was cloned.
#[cfg(test)]
pub(crate) fn asset_scope_snapshot_captures<R: Runtime>(app: &AppHandle<R>) -> usize {
    scope_snapshot_cache(app).captures.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_patterns_keep_literals_but_reject_multi_unit_classes() {
        for pattern in ["/é/[a-z]*", "/é/[!a-z]*", "/é/[[]box[]]/**/*", "/é/[]]/*"] {
            assert!(
                native_pattern(&Pattern::new(pattern).unwrap()).is_some(),
                "{pattern}"
            );
        }
        for pattern in ["/approved/[🐎]*", "/approved/[!🐎]*"] {
            assert!(
                native_pattern(&Pattern::new(pattern).unwrap()).is_none(),
                "{pattern}"
            );
        }
        for pattern in ["/approved/[é]*", "/approved/[!é]*"] {
            assert_eq!(
                native_pattern(&Pattern::new(pattern).unwrap()).is_some(),
                cfg!(windows)
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn unicode_classes_never_authorize_their_partial_utf8_bytes() {
        use std::os::unix::ffi::OsStrExt;
        let native = Path::new("/approved").join(std::ffi::OsStr::from_bytes(b"\xc3\xff.mp4"));
        let pattern = Pattern::new("/approved/[é]*").unwrap();
        let set = PatternSet::new(HashSet::from([pattern]));
        assert!(set.matches(Path::new("/approved/é.mp4")));
        assert!(!set.matches(&native));

        let app = tauri::test::mock_app();
        let scope = app.handle().asset_protocol_scope();
        scope.allow_directory("/approved", true).unwrap();
        let mut snapshot = ScopeSnapshot::capture(&scope);
        snapshot.forbidden = set;
        assert!(
            snapshot.forbids(&native),
            "unsupported deny classes fail closed"
        );
    }

    #[cfg(windows)]
    #[test]
    fn unicode_classes_never_authorize_an_unpaired_surrogate() {
        use std::os::windows::ffi::OsStringExt;
        let mut units: Vec<_> = r"C:\approved\".encode_utf16().collect();
        units.push(0xd83d);
        units.extend(".mp4".encode_utf16());
        let native = PathBuf::from(std::ffi::OsString::from_wide(&units));
        let set = PatternSet::new(HashSet::from([Pattern::new(r"C:\approved\[🐎]*").unwrap()]));
        assert!(set.matches(Path::new(r"C:\approved\🐎.mp4")));
        assert!(!set.matches(&native));
    }

    #[cfg(unix)]
    #[test]
    fn directory_patterns_match_native_names_without_replacement_aliases() {
        use std::os::unix::ffi::OsStrExt;
        let app = tauri::test::mock_app();
        let scope = app.handle().asset_protocol_scope();
        let root = Path::new("/approved/片段");
        let native = root.join(std::ffi::OsStr::from_bytes(b"clip-\xff.mp4"));
        let shadow = PathBuf::from(native.to_string_lossy().as_ref());
        scope.allow_directory(root, true).unwrap();
        scope.forbid_file(&shadow).unwrap();
        let snapshot = asset_scope_snapshot(app.handle());
        assert!(snapshot.allows(&native));
        assert!(!snapshot.allows(&shadow));
        scope.forbid_directory(root, true).unwrap();
        assert!(!asset_scope_snapshot(app.handle()).allows(&native));

        let bad_parent = Path::new("/approved").join(std::ffi::OsStr::from_bytes(b"parent-\xff"));
        scope
            .allow_directory(PathBuf::from(bad_parent.to_string_lossy().as_ref()), true)
            .unwrap();
        assert!(!asset_scope_snapshot(app.handle()).allows(&bad_parent.join("clip.mp4")));
    }

    #[test]
    fn literal_detection_matches_glob_escape_and_rejects_wildcards() {
        for path in [
            "/media/clip.mp4",
            "/media/[take 1]/a?b*c.mp4",
            "/media/]odd[.mov",
            "C:\\Media\\Clip.mp4",
        ] {
            assert_eq!(
                literal_pattern_text(&Pattern::escape(path)).as_deref(),
                Some(path)
            );
        }
        for glob in ["/media/*", "/media/**", "/media/a?.mp4", "/media/[ab].mp4"] {
            assert_eq!(literal_pattern_text(glob), None, "{glob}");
        }
    }

    #[test]
    fn snapshot_matches_the_cloning_scope_check_exactly() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let app = tauri::test::mock_app();
        let scope = app.handle().asset_protocol_scope();
        scope.allow_file(root.join("exact.mp4")).unwrap();
        scope.allow_file(root.join("[odd] a*b?.mp4")).unwrap();
        scope.allow_directory(root.join("folder"), false).unwrap();
        scope.allow_directory(root.join("tree"), true).unwrap();
        scope.forbid_file(root.join("tree/forbidden.mp4")).unwrap();
        scope
            .forbid_directory(root.join("tree/blocked"), true)
            .unwrap();

        let snapshot = ScopeSnapshot::capture(&scope);
        let candidates = [
            "exact.mp4",
            "Exact.mp4",
            "exact.mp4.bak",
            "[odd] a*b?.mp4",
            "[odd] aXbY.mp4",
            "folder/a.mp4",
            "folder/.hidden.mp4",
            "folder/nested/a.mp4",
            "tree/a/b/c.mp4",
            "tree/forbidden.mp4",
            "tree/blocked/a.mp4",
            "tree/.dot/a.mp4",
            "other.mp4",
        ];
        for candidate in candidates {
            let path = root.join(candidate);
            assert_eq!(
                snapshot.allows(&path),
                scope_allows_lexical_path(&scope, &path),
                "{candidate}"
            );
        }
        assert!(snapshot.has_exact_file_grant(&root.join("exact.mp4")));
        assert!(!snapshot.has_exact_file_grant(&root.join("folder/a.mp4")));
    }

    #[test]
    fn cached_snapshot_clones_patterns_once_and_follows_every_scope_change() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("clip.mp4");
        let app = tauri::test::mock_app();
        let scope = app.handle().asset_protocol_scope();
        scope.allow_file(&path).unwrap();

        let first = asset_scope_snapshot(app.handle());
        let captures = asset_scope_snapshot_captures(app.handle());
        for _ in 0..100 {
            assert!(first.shares_allowed_patterns_with(&asset_scope_snapshot(app.handle())));
        }
        assert_eq!(asset_scope_snapshot_captures(app.handle()), captures);
        assert!(first.allows(&path));

        // Revocation (deny precedence) must be visible to the next check.
        scope.forbid_file(&path).unwrap();
        assert!(!asset_scope_snapshot(app.handle()).allows(&path));
        let other = directory.path().join("other.mp4");
        assert!(!asset_scope_snapshot(app.handle()).allows(&other));
        scope.allow_file(&other).unwrap();
        assert!(asset_scope_snapshot(app.handle()).allows(&other));
    }

    #[test]
    fn a_revocation_queued_behind_a_running_emit_is_honoured_when_forbid_returns() {
        let directory = tempfile::tempdir().unwrap();
        let revoked = directory.path().join("revoked.mp4");
        let slow = directory.path().join("slow-grant.mp4");
        let app = tauri::test::mock_app();
        let scope = app.handle().asset_protocol_scope();
        scope.allow_file(&revoked).unwrap();
        assert!(asset_scope_snapshot(app.handle()).allows(&revoked));

        // A listener that stalls one emit (like persisted-scope rewriting its
        // file) until released. Tauri queues every other event meanwhile.
        let (entered_sender, entered) = std::sync::mpsc::channel::<()>();
        let (release, released) = std::sync::mpsc::channel::<()>();
        let entered_sender = Mutex::new(entered_sender);
        let released = Mutex::new(released);
        let slow_path = slow.clone();
        scope.listen(move |event| {
            if matches!(event, tauri::scope::fs::Event::PathAllowed(path) if *path == slow_path) {
                let _ = entered_sender.lock().unwrap().send(());
                let _ = released.lock().unwrap().recv();
            }
        });
        let forbid_delivered = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let recorded = forbid_delivered.clone();
        let revoked_path = revoked.clone();
        scope.listen(move |event| {
            if matches!(event, tauri::scope::fs::Event::PathForbidden(path) if *path == revoked_path) {
                recorded.store(true, Ordering::Release);
            }
        });
        let emitting_scope = scope.clone();
        let emitter = std::thread::spawn(move || emitting_scope.allow_file(&slow).unwrap());
        entered.recv().unwrap();

        scope.forbid_file(&revoked).unwrap();
        assert!(
            !forbid_delivered.load(Ordering::Acquire),
            "the forbid event is still queued behind the stalled emit"
        );
        assert!(
            !asset_scope_snapshot(app.handle()).allows(&revoked),
            "a revocation must apply as soon as forbid_file returns"
        );

        release.send(()).unwrap();
        emitter.join().unwrap();
        assert!(forbid_delivered.load(Ordering::Acquire));
        assert!(!asset_scope_snapshot(app.handle()).allows(&revoked));
    }
}
