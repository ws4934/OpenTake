//! Cached, locally matched copies of the asset-protocol scope patterns.
//!
//! `Scope::allowed_patterns()` / `forbidden_patterns()` clone every pattern
//! under Tauri's mutex, and every imported file adds an allowed pattern, so
//! calling them per path check made authorization cost grow with the number
//! of imports (and quadratically for per-entry loops). A [`ScopeSnapshot`] is
//! taken once per scope change and matched locally: patterns that are escaped
//! literal paths (all exact-file grants) are looked up in a hash set, real
//! globs are still evaluated by `glob` with the protocol's match options.

use super::*;
use glob::Pattern;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

/// Local matcher equivalent to [`scope_allows_lexical_path`] plus the exact
/// file grant test used by the Home thumbnail exception.
pub(crate) struct ScopeSnapshot {
    allowed: PatternSet,
    forbidden: PatternSet,
    /// Allowed pattern texts, keyed like `scope_has_exact_file_grant` compares
    /// them (ASCII case-insensitively on Windows).
    exact_allowed: HashSet<String>,
}

impl ScopeSnapshot {
    /// Clone the scope's two pattern sets exactly once.
    pub(crate) fn capture(scope: &Scope) -> Self {
        let allowed = scope.allowed_patterns();
        let forbidden = scope.forbidden_patterns();
        let exact_allowed = allowed
            .iter()
            .map(|pattern| exact_grant_key(pattern.as_str()))
            .collect();
        Self {
            allowed: PatternSet::new(allowed),
            forbidden: PatternSet::new(forbidden),
            exact_allowed,
        }
    }

    /// Same decision as [`scope_allows_lexical_path`]: forbidden patterns take
    /// precedence, then any allowed pattern must match the normalized path.
    pub(crate) fn allows(&self, path: &Path) -> bool {
        let normalized: PathBuf = path.components().collect();
        !self.forbidden.matches(&normalized) && self.allowed.matches(&normalized)
    }

    /// Same decision as `scope_has_exact_file_grant`: an allowed pattern whose
    /// text is exactly the escaped normalized path, not a directory glob.
    pub(crate) fn has_exact_file_grant(&self, path: &Path) -> bool {
        let escaped = Pattern::escape(normalized_path(path).to_string_lossy().as_ref());
        self.exact_allowed.contains(&exact_grant_key(&escaped))
    }
}

struct PatternSet {
    literals: HashSet<String>,
    globs: Vec<Pattern>,
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
        Self { literals, globs }
    }

    fn matches(&self, normalized: &Path) -> bool {
        // `Pattern::matches_path_with` never matches a non-UTF-8 path either.
        let Some(text) = normalized.to_str() else {
            return false;
        };
        let options = scope_match_options();
        self.literals.contains(&literal_match_key(text))
            || self
                .globs
                .iter()
                .any(|pattern| pattern.matches_path_with(normalized, options))
    }
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

/// Per-app cache of the asset-protocol scope snapshot.
///
/// Tauri scopes only grow (`allow_*` / `forbid_*`) and emit an event after
/// every change. A listener bumps `generation`; readers load the generation
/// before capturing, so a snapshot is never cached under a generation newer
/// than the patterns it contains.
struct ScopeSnapshotCache {
    generation: Arc<AtomicU64>,
    cached: Mutex<Option<(u64, Arc<ScopeSnapshot>)>>,
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

/// The current asset-protocol scope as a local matcher. Pattern sets are
/// cloned only after the scope changed, never per path check.
pub(crate) fn asset_scope_snapshot<R: Runtime>(app: &AppHandle<R>) -> Arc<ScopeSnapshot> {
    let cache = scope_snapshot_cache(app);
    let generation = cache.generation.load(Ordering::Acquire);
    {
        let cached = cache
            .cached
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some((cached_generation, snapshot)) = cached.as_ref() {
            if *cached_generation == generation {
                return snapshot.clone();
            }
        }
    }
    let snapshot = Arc::new(ScopeSnapshot::capture(&app.asset_protocol_scope()));
    cache.captures.fetch_add(1, Ordering::Relaxed);
    let mut cached = cache
        .cached
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if cached
        .as_ref()
        .is_none_or(|(cached_generation, _)| *cached_generation <= generation)
    {
        *cached = Some((generation, snapshot.clone()));
    }
    snapshot
}

/// Test hook: how many times this app's scope pattern sets were cloned.
#[cfg(test)]
pub(crate) fn asset_scope_snapshot_captures<R: Runtime>(app: &AppHandle<R>) -> usize {
    scope_snapshot_cache(app).captures.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

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
            assert!(Arc::ptr_eq(&first, &asset_scope_snapshot(app.handle())));
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
}
