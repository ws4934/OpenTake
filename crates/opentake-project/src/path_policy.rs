//! Portable validation for paths stored inside `.opentake` bundles.
//!
//! The rule lives in `opentake-domain` so the pure manifest writer
//! (`MediaAsset::to_manifest_entry`) applies exactly the same check as every
//! bundle reader and writer in this crate.

pub use opentake_domain::is_safe_project_asset_relative_path;
