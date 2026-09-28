//! Per-bridge caches for the advanced video workflows.
//!
//! Matting and object removal key their outputs by the source's SHA-256, and
//! the Agent advertises `generate_matte` only while the RVM model verifies.
//! Both used to re-read whole files on every call: a multi-gigabyte source on
//! every run (even a cache hit), and the ~15 MB model on every tool dispatch.
//! These caches remember a digest for as long as the file's
//! [`MediaSourceStamp`] (size, modification time and file identity) is
//! unchanged.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

use opentake_domain::MediaSourceStamp;
use opentake_media::analysis::{matting_model_path, verify_rvm_model, InstalledMattingModel};
use opentake_media::{
    file_sha256_with_stamp_cancellable, source_file_stamp, MediaCancelToken, MediaError,
};

/// Distinct sources remembered before the cache starts over.
const MAX_SOURCE_HASHES: usize = 256;

/// Model file stamp (`None` when absent) and the verification it produced.
type CachedModelVerification = (
    Option<MediaSourceStamp>,
    Result<InstalledMattingModel, String>,
);

#[derive(Default)]
pub(crate) struct AdvancedWorkflowCaches {
    source_hashes: Mutex<HashMap<PathBuf, (MediaSourceStamp, String)>>,
    rvm_model: Mutex<Option<CachedModelVerification>>,
    #[cfg(test)]
    source_hash_reads: AtomicUsize,
    #[cfg(test)]
    rvm_model_verifications: AtomicUsize,
}

impl AdvancedWorkflowCaches {
    /// SHA-256 of `path`, reusing the previous digest while the file's stamp
    /// is unchanged. Hashing observes `cancel` between reads.
    pub(crate) fn source_sha256(
        &self,
        path: &Path,
        cancel: &MediaCancelToken,
    ) -> Result<String, MediaError> {
        let stamp = source_file_stamp(path)?;
        if let Some((cached, digest)) = self
            .source_hashes
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(path)
        {
            if *cached == stamp {
                return Ok(digest.clone());
            }
        }
        #[cfg(test)]
        self.source_hash_reads.fetch_add(1, Ordering::SeqCst);
        // `None` means the file changed while it was read; that digest
        // describes no stable content.
        let (stamp, digest) =
            file_sha256_with_stamp_cancellable(path, cancel)?.ok_or_else(|| {
                MediaError::Io(std::io::Error::other(
                    "the source changed while it was being hashed; try again",
                ))
            })?;
        let mut hashes = self
            .source_hashes
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if hashes.len() >= MAX_SOURCE_HASHES && !hashes.contains_key(path) {
            hashes.clear();
        }
        hashes.insert(path.to_path_buf(), (stamp, digest.clone()));
        Ok(digest)
    }

    /// Verified RVM model, re-verified only when the model file is installed,
    /// replaced or deleted (its stamp changes).
    pub(crate) fn rvm_model(&self, models_dir: &Path) -> Result<InstalledMattingModel, String> {
        // `None` while the model is absent (or not a regular file), so an
        // install, replacement or delete always changes the key.
        let stamp = source_file_stamp(&matting_model_path(models_dir)).ok();
        let mut cached = self
            .rvm_model
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some((cached_stamp, result)) = cached.as_ref() {
            if *cached_stamp == stamp {
                return result.clone();
            }
        }
        #[cfg(test)]
        self.rvm_model_verifications.fetch_add(1, Ordering::SeqCst);
        let result = verify_rvm_model(models_dir).map_err(|error| error.to_string());
        *cached = Some((stamp, result.clone()));
        result
    }

    #[cfg(test)]
    pub(crate) fn source_hash_reads(&self) -> usize {
        self.source_hash_reads.load(Ordering::SeqCst)
    }

    #[cfg(test)]
    pub(crate) fn rvm_model_verifications(&self) -> usize {
        self.rvm_model_verifications.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_digest_is_reused_until_the_file_changes() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source.mov");
        std::fs::write(&source, b"first source bytes").unwrap();
        let caches = AdvancedWorkflowCaches::default();
        let cancel = MediaCancelToken::new();

        let first = caches.source_sha256(&source, &cancel).unwrap();
        let second = caches.source_sha256(&source, &cancel).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            caches.source_hash_reads(),
            1,
            "a cache hit must not re-read"
        );
        assert_eq!(first, opentake_media::file_sha256(&source).unwrap());

        std::fs::write(&source, b"replaced with different bytes").unwrap();
        let third = caches.source_sha256(&source, &cancel).unwrap();
        assert_ne!(first, third);
        assert_eq!(caches.source_hash_reads(), 2);
    }

    #[test]
    fn cancelled_source_hash_returns_cancelled_and_caches_nothing() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source.mov");
        std::fs::write(&source, vec![7_u8; 256 * 1024]).unwrap();
        let caches = AdvancedWorkflowCaches::default();
        let cancel = MediaCancelToken::new();
        cancel.cancel();

        let error = caches
            .source_sha256(&source, &cancel)
            .expect_err("a cancelled hash must stop");
        assert!(matches!(error, MediaError::Cancelled), "{error:?}");
        caches
            .source_sha256(&source, &MediaCancelToken::new())
            .unwrap();
        assert_eq!(caches.source_hash_reads(), 2);
    }

    #[test]
    fn model_verification_runs_once_until_the_model_changes() {
        let temp = tempfile::tempdir().unwrap();
        let caches = AdvancedWorkflowCaches::default();

        for _ in 0..100 {
            assert!(caches.rvm_model(temp.path()).is_err());
        }
        assert_eq!(caches.rvm_model_verifications(), 1);

        // "Installing" a model (here an invalid one) invalidates the result.
        let model = matting_model_path(temp.path());
        std::fs::create_dir_all(model.parent().unwrap()).unwrap();
        std::fs::write(&model, b"not the model").unwrap();
        for _ in 0..100 {
            let error = caches.rvm_model(temp.path()).unwrap_err();
            assert!(error.contains("size_mismatch"), "{error}");
        }
        assert_eq!(caches.rvm_model_verifications(), 2);

        std::fs::remove_file(&model).unwrap();
        assert!(caches
            .rvm_model(temp.path())
            .unwrap_err()
            .contains("not_installed"));
        assert_eq!(caches.rvm_model_verifications(), 3);
    }
}
