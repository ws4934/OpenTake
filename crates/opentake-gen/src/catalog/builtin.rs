//! Built-in static catalog. Under BYOK, `list_models()` returns this catalog
//! compiled into the binary (no backend required), with the same structure as
//! the managed `/v1/models` response so UI/agent behave identically (axiom A5).
//! Entry ids remain stable project-facing identifiers. `vendorModel` carries
//! the provider's API identifier independently; pricing is omitted under BYOK.

use super::entry::CatalogEntry;

/// The catalog JSON embedded at compile time.
const BUILTIN_CATALOG_JSON: &str = include_str!("builtin_catalog.json");

/// Parse and return the built-in catalog. Panics only on a malformed embedded
/// asset, which is a compile-time-shipped file and thus a programmer error.
pub fn builtin_catalog() -> Vec<CatalogEntry> {
    serde_json::from_str(BUILTIN_CATALOG_JSON).expect("builtin catalog must parse")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::entry::ModelKind;
    use std::collections::HashSet;

    #[test]
    fn builtin_catalog_parses() {
        let cat = builtin_catalog();
        assert!(!cat.is_empty());
    }

    #[test]
    fn all_ids_are_prefixed_and_unique() {
        let cat = builtin_catalog();
        let mut seen = HashSet::new();
        for e in &cat {
            assert!(e.id.contains(':'), "id {} must be prefix:vendorModel", e.id);
            assert!(seen.insert(e.id.clone()), "duplicate id {}", e.id);
        }
    }

    #[test]
    fn covers_all_four_kinds() {
        let cat = builtin_catalog();
        let kinds: HashSet<ModelKind> = cat.iter().map(|e| e.kind).collect();
        assert!(kinds.contains(&ModelKind::Image));
        assert!(kinds.contains(&ModelKind::Video));
        assert!(kinds.contains(&ModelKind::Audio));
        assert!(kinds.contains(&ModelKind::Upscale));
    }

    #[test]
    fn covers_all_four_providers() {
        let cat = builtin_catalog();
        let prefixes: HashSet<&str> = cat.iter().filter_map(|e| e.id.split(':').next()).collect();
        for p in ["fal", "replicate", "openai", "elevenlabs"] {
            assert!(prefixes.contains(p), "missing provider {p}");
        }
    }

    #[test]
    fn byok_routes_use_documented_vendor_identifiers() {
        // fal.ai model pages, Replicate official model pages, and ElevenLabs
        // /docs/api-reference/{text-to-speech/convert,music/compose}.
        let entries = builtin_catalog();
        for (id, vendor) in [
            ("fal:flux-pro", "fal-ai/flux-pro/v1.1"),
            ("fal:flux-kontext", "fal-ai/flux-pro/kontext/text-to-image"),
            (
                "fal:kling-video",
                "fal-ai/kling-video/v2.5-turbo/pro/text-to-video",
            ),
            ("replicate:seedance-1-pro", "bytedance/seedance-1-pro"),
            ("replicate:topaz-upscale", "topazlabs/video-upscale"),
            (
                "elevenlabs:eleven-multilingual-v2",
                "eleven_multilingual_v2",
            ),
            ("elevenlabs:eleven-music", "music_v1"),
        ] {
            let entry = entries.iter().find(|entry| entry.id == id).unwrap();
            assert_eq!(entry.vendor_model.as_deref(), Some(vendor), "{id}");
        }
        let image = entries
            .iter()
            .find(|entry| entry.id == "openai:gpt-image-1")
            .unwrap();
        let super::super::entry::UiCapabilities::Image(caps) = &image.ui_capabilities else {
            panic!("image model must have image capabilities");
        };
        assert!(!caps.supports_image_reference);
        let eleven = entries
            .iter()
            .find(|entry| entry.id == "elevenlabs:eleven-multilingual-v2")
            .unwrap();
        let super::super::entry::UiCapabilities::Audio(caps) = &eleven.ui_capabilities else {
            panic!("ElevenLabs TTS must have audio capabilities");
        };
        assert!(caps.supports_voice("RACHEL"));
        assert!(!caps.supports_voice("../voices"));
        assert!(!caps.supports_style_instructions);
    }
}
