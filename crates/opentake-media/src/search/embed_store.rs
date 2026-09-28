//! `EmbeddingStore` — per-asset frame embeddings on disk in the `PALMEMB1`
//! binary format, byte-compatible with upstream
//! (`Search/Indexing/EmbeddingStore.swift`). Vectors are f16 on disk, f32 in
//! memory.
//!
//! Layout (all little-endian, unaligned):
//! ```text
//! "PALMEMB1"            8 bytes ASCII magic
//! u32 headerLen        4 bytes
//! JSON(Header)         headerLen bytes
//! count rows, each rowBytes = 24 + dim*2:
//!     f64 time
//!     f64 shotStart
//!     f64 shotEnd
//!     dim × f16
//! ```

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

use byteorder::{ByteOrder, LittleEndian};
use half::f16;
use serde::{Deserialize, Serialize};

use crate::cache_key::file_identity_key;
use crate::error::{MediaError, Result};

/// Magic bytes prefixing every `.embed` file.
pub const MAGIC: &[u8; 8] = b"PALMEMB1";
/// Cache subdirectory (kept identical to upstream).
pub const CACHE_SUBDIR: &str = "Embeddings";

/// Embedding store header (JSON, camelCase to match upstream).
#[derive(Serialize, Deserialize, PartialEq, Eq, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Header {
    pub model: String,
    pub model_version: i32,
    pub sampler_version: i32,
    pub dim: usize,
    pub count: usize,
}

/// One indexed frame's metadata (the f32 vector lives in `AssetIndex.vectors`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Row {
    pub time: f64,
    pub shot_start: f64,
    pub shot_end: f64,
}

/// A loaded index: header, per-row metadata, and a flat `count*dim` f32 vector
/// block (row-major) ready for matrix·vector ranking.
#[derive(Clone, Debug, PartialEq)]
pub struct AssetIndex {
    pub header: Header,
    pub rows: Vec<Row>,
    pub vectors: Vec<f32>,
}

fn embed_path(cache_root: &Path, key: &str) -> PathBuf {
    cache_root.join(CACHE_SUBDIR).join(format!("{key}.embed"))
}

fn row_bytes(dim: usize) -> usize {
    3 * 8 + dim * 2
}

/// Cache key for `path` (`file_identity_key` with 32 hex chars).
pub fn key(path: &Path) -> Option<String> {
    file_identity_key(path)
}

/// Serialize an index to the `PALMEMB1` byte layout. Pure (no IO) so the exact
/// bytes are testable; `save` wraps this with an atomic file write.
pub fn encode(header: &Header, rows: &[Row], vectors: &[f32]) -> Result<Vec<u8>> {
    if rows.len() != header.count || vectors.len() != header.count * header.dim {
        return Err(MediaError::StoreCorrupt);
    }
    let json = serde_json::to_vec(header).map_err(|_| MediaError::StoreCorrupt)?;
    let mut out = Vec::with_capacity(8 + 4 + json.len() + header.count * row_bytes(header.dim));
    out.extend_from_slice(MAGIC);
    let mut len4 = [0u8; 4];
    LittleEndian::write_u32(&mut len4, json.len() as u32);
    out.extend_from_slice(&len4);
    out.extend_from_slice(&json);

    let mut buf8 = [0u8; 8];
    let mut buf2 = [0u8; 2];
    for (i, row) in rows.iter().enumerate() {
        for v in [row.time, row.shot_start, row.shot_end] {
            LittleEndian::write_f64(&mut buf8, v);
            out.extend_from_slice(&buf8);
        }
        for d in 0..header.dim {
            let h = f16::from_f32(vectors[i * header.dim + d]);
            LittleEndian::write_u16(&mut buf2, h.to_bits());
            out.extend_from_slice(&buf2);
        }
    }
    Ok(out)
}

/// Parse the `PALMEMB1` byte layout into an [`AssetIndex`]. Strict length
/// validation: any mismatch → [`MediaError::StoreCorrupt`].
pub fn decode(data: &[u8]) -> Result<AssetIndex> {
    if data.len() < MAGIC.len() + 4 || &data[..MAGIC.len()] != MAGIC {
        return Err(MediaError::StoreCorrupt);
    }
    let mut offset = MAGIC.len();
    let header_len = LittleEndian::read_u32(&data[offset..offset + 4]) as usize;
    offset += 4;
    if data.len() < offset + header_len {
        return Err(MediaError::StoreCorrupt);
    }
    let header: Header = serde_json::from_slice(&data[offset..offset + header_len])
        .map_err(|_| MediaError::StoreCorrupt)?;
    offset += header_len;

    let rb = row_bytes(header.dim);
    if data.len() != offset + header.count * rb {
        return Err(MediaError::StoreCorrupt);
    }

    let mut rows = Vec::with_capacity(header.count);
    let mut vectors = vec![0.0f32; header.count * header.dim];
    for i in 0..header.count {
        let base = offset + i * rb;
        let time = LittleEndian::read_f64(&data[base..base + 8]);
        let shot_start = LittleEndian::read_f64(&data[base + 8..base + 16]);
        let shot_end = LittleEndian::read_f64(&data[base + 16..base + 24]);
        if !(time.is_finite() && shot_start.is_finite() && shot_end.is_finite()) {
            return Err(MediaError::StoreCorrupt);
        }
        rows.push(Row {
            time,
            shot_start,
            shot_end,
        });
        for d in 0..header.dim {
            let off = base + 24 + d * 2;
            let bits = LittleEndian::read_u16(&data[off..off + 2]);
            let value = f16::from_bits(bits).to_f32();
            // Encoders only produce finite vectors; NaN/inf here means a
            // damaged or foreign file, which must be re-indexed, not ranked.
            if !value.is_finite() {
                return Err(MediaError::StoreCorrupt);
            }
            vectors[i * header.dim + d] = value;
        }
    }
    Ok(AssetIndex {
        header,
        rows,
        vectors,
    })
}

/// Read just the header from a `.embed` file (cheap currency check).
pub fn header(cache_root: &Path, key: &str) -> Option<Header> {
    let data = std::fs::read(embed_path(cache_root, key)).ok()?;
    if data.len() < MAGIC.len() + 4 || &data[..MAGIC.len()] != MAGIC {
        return None;
    }
    let header_len = LittleEndian::read_u32(&data[MAGIC.len()..MAGIC.len() + 4]) as usize;
    let start = MAGIC.len() + 4;
    if data.len() < start + header_len {
        return None;
    }
    serde_json::from_slice(&data[start..start + header_len]).ok()
}

/// True when an on-disk index matches `(model, model_version, sampler_version)`.
pub fn is_current(
    cache_root: &Path,
    key: &str,
    model: &str,
    model_version: i32,
    sampler_version: i32,
) -> bool {
    match header(cache_root, key) {
        Some(h) => {
            h.model == model
                && h.model_version == model_version
                && h.sampler_version == sampler_version
        }
        None => false,
    }
}

/// Load a full index from `<cache_root>/Embeddings/<key>.embed`.
pub fn load(cache_root: &Path, key: &str) -> Result<AssetIndex> {
    let (index, _) = read_index(&embed_path(cache_root, key))?;
    Ok(index)
}

/// Upper bound for decoded indexes retained in memory by [`load_cached`].
pub const INDEX_CACHE_BYTES: usize = 256 * 1024 * 1024;

/// Load an index through the process-wide decoded-index cache. A hit costs one
/// `stat`; the file is read and decoded again only when its identity (size,
/// mtime and, on Unix, device/inode) changed, when [`save`] or [`clear_all`]
/// invalidated it, or after LRU eviction under [`INDEX_CACHE_BYTES`].
pub fn load_cached(cache_root: &Path, key: &str) -> Result<Arc<AssetIndex>> {
    let path = embed_path(cache_root, key);
    let current = FileStamp::of(&std::fs::metadata(&path)?);
    if let Some(index) = index_cache().lock_cache().get(&path, &current) {
        return Ok(index);
    }
    let (index, stamp) = read_index(&path)?;
    let index = Arc::new(index);
    index_cache()
        .lock_cache()
        .insert(path, stamp, index.clone(), INDEX_CACHE_BYTES);
    Ok(index)
}

/// Atomically write an index. Creates the cache subdirectory if needed. Each
/// writer uses its own temporary file, flushed to disk before it replaces the
/// final name, so concurrent writers (another app instance, a future parallel
/// indexer) can never interleave bytes into one file.
pub fn save(
    cache_root: &Path,
    key: &str,
    header: &Header,
    rows: &[Row],
    vectors: &[f32],
) -> Result<()> {
    let bytes = encode(header, rows, vectors)?;
    let dir = cache_root.join(CACHE_SUBDIR);
    std::fs::create_dir_all(&dir)?;
    let final_path = dir.join(format!("{key}.embed"));
    let tmp_path = dir.join(format!("{key}.{}.embed.tmp", unique_suffix()));
    let written = (|| -> Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&tmp_path, &final_path)?;
        Ok(())
    })();
    if written.is_err() {
        let _ = std::fs::remove_file(&tmp_path);
    }
    index_cache().lock_cache().remove(&final_path);
    written
}

fn unique_suffix() -> String {
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos())
        .unwrap_or(0);
    format!(
        "{}-{}-{nanos:x}",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    )
}

/// Read and decode one index file, returning the identity of the exact handle
/// that was read so a concurrent replacement cannot be cached under a stale
/// stamp.
fn read_index(path: &Path) -> Result<(AssetIndex, FileStamp)> {
    let mut file = std::fs::File::open(path)?;
    let stamp = FileStamp::of(&file.metadata()?);
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;
    Ok((decode(&data)?, stamp))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FileStamp {
    len: u64,
    modified: Option<SystemTime>,
    #[cfg(unix)]
    dev: u64,
    #[cfg(unix)]
    ino: u64,
}

impl FileStamp {
    fn of(metadata: &std::fs::Metadata) -> Self {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        FileStamp {
            len: metadata.len(),
            modified: metadata.modified().ok(),
            #[cfg(unix)]
            dev: metadata.dev(),
            #[cfg(unix)]
            ino: metadata.ino(),
        }
    }
}

struct CachedIndex {
    stamp: FileStamp,
    index: Arc<AssetIndex>,
    bytes: usize,
    last_used: u64,
}

#[derive(Default)]
struct IndexCache {
    entries: HashMap<PathBuf, CachedIndex>,
    bytes: usize,
    clock: u64,
}

impl IndexCache {
    fn get(&mut self, path: &Path, stamp: &FileStamp) -> Option<Arc<AssetIndex>> {
        self.clock += 1;
        let clock = self.clock;
        match self.entries.get_mut(path) {
            Some(entry) if entry.stamp == *stamp => {
                entry.last_used = clock;
                Some(entry.index.clone())
            }
            Some(_) => {
                self.remove(path);
                None
            }
            None => None,
        }
    }

    fn insert(&mut self, path: PathBuf, stamp: FileStamp, index: Arc<AssetIndex>, limit: usize) {
        self.remove(&path);
        let bytes = index.vectors.len() * std::mem::size_of::<f32>()
            + index.rows.len() * std::mem::size_of::<Row>();
        if bytes > limit {
            return;
        }
        while self.bytes + bytes > limit {
            let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(path, _)| path.clone())
            else {
                break;
            };
            self.remove(&oldest);
        }
        self.clock += 1;
        self.bytes += bytes;
        self.entries.insert(
            path,
            CachedIndex {
                stamp,
                index,
                bytes,
                last_used: self.clock,
            },
        );
    }

    fn remove(&mut self, path: &Path) {
        if let Some(entry) = self.entries.remove(path) {
            self.bytes -= entry.bytes;
        }
    }

    fn remove_under(&mut self, dir: &Path) {
        let stale: Vec<PathBuf> = self
            .entries
            .keys()
            .filter(|path| path.starts_with(dir))
            .cloned()
            .collect();
        for path in stale {
            self.remove(&path);
        }
    }
}

struct SharedIndexCache(Mutex<IndexCache>);

impl SharedIndexCache {
    fn lock_cache(&self) -> std::sync::MutexGuard<'_, IndexCache> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

fn index_cache() -> &'static SharedIndexCache {
    static CACHE: OnceLock<SharedIndexCache> = OnceLock::new();
    CACHE.get_or_init(|| SharedIndexCache(Mutex::new(IndexCache::default())))
}

/// Remove the entire embeddings cache directory.
pub fn clear_all(cache_root: &Path) -> Result<()> {
    let dir = cache_root.join(CACHE_SUBDIR);
    index_cache().lock_cache().remove_under(&dir);
    if dir.exists() {
        std::fs::remove_dir_all(dir)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header_dim2() -> Header {
        Header {
            model: "siglip2-base-patch16-256".into(),
            model_version: 1,
            sampler_version: 1,
            dim: 2,
            count: 2,
        }
    }

    #[test]
    fn encode_starts_with_magic_and_header_len() {
        let h = header_dim2();
        let rows = vec![
            Row {
                time: 0.0,
                shot_start: 0.0,
                shot_end: 1.0,
            },
            Row {
                time: 1.0,
                shot_start: 1.0,
                shot_end: 2.0,
            },
        ];
        let vectors = vec![0.5, -0.5, 1.0, 0.0];
        let bytes = encode(&h, &rows, &vectors).unwrap();
        assert_eq!(&bytes[..8], MAGIC);
        // total = 8 + 4 + json + 2*(24 + 2*2)
        let json_len = LittleEndian::read_u32(&bytes[8..12]) as usize;
        assert_eq!(bytes.len(), 8 + 4 + json_len + 2 * (24 + 4));
    }

    #[test]
    fn encode_decode_roundtrip_f16_quantized() {
        let h = header_dim2();
        let rows = vec![
            Row {
                time: 0.0,
                shot_start: 0.0,
                shot_end: 1.5,
            },
            Row {
                time: 2.25,
                shot_start: 1.5,
                shot_end: 3.0,
            },
        ];
        let vectors = vec![0.5f32, -0.25, 1.0, 0.125];
        let bytes = encode(&h, &rows, &vectors).unwrap();
        let idx = decode(&bytes).unwrap();
        assert_eq!(idx.header, h);
        assert_eq!(idx.rows, rows);
        // f16 round-trip is exact for these dyadic values.
        for (a, b) in vectors.iter().zip(idx.vectors.iter()) {
            assert_eq!(*a, *b);
        }
    }

    #[test]
    fn row_bytes_for_dim768_is_1560() {
        assert_eq!(row_bytes(768), 24 + 768 * 2);
        assert_eq!(row_bytes(768), 1560);
    }

    #[test]
    fn decode_rejects_bad_magic() {
        let mut bytes = vec![0u8; 20];
        bytes[..8].copy_from_slice(b"NOTMAGIC");
        assert!(matches!(decode(&bytes), Err(MediaError::StoreCorrupt)));
    }

    #[test]
    fn decode_rejects_truncation() {
        let h = header_dim2();
        let rows = vec![
            Row {
                time: 0.0,
                shot_start: 0.0,
                shot_end: 1.0,
            },
            Row {
                time: 1.0,
                shot_start: 1.0,
                shot_end: 2.0,
            },
        ];
        let vectors = vec![0.5, -0.5, 1.0, 0.0];
        let mut bytes = encode(&h, &rows, &vectors).unwrap();
        bytes.truncate(bytes.len() - 1);
        assert!(matches!(decode(&bytes), Err(MediaError::StoreCorrupt)));
    }

    #[test]
    fn decode_rejects_extra_trailing_bytes() {
        let h = header_dim2();
        let rows = vec![Row {
            time: 0.0,
            shot_start: 0.0,
            shot_end: 1.0,
        }];
        let mut h2 = h.clone();
        h2.count = 1;
        let vectors = vec![0.5, -0.5];
        let mut bytes = encode(&h2, &rows, &vectors).unwrap();
        bytes.push(0xFF);
        assert!(matches!(decode(&bytes), Err(MediaError::StoreCorrupt)));
    }

    #[test]
    fn encode_rejects_count_vector_mismatch() {
        let h = header_dim2(); // count 2, dim 2 → needs 4 floats
        let rows = vec![Row {
            time: 0.0,
            shot_start: 0.0,
            shot_end: 1.0,
        }]; // only 1 row
        assert!(matches!(
            encode(&h, &rows, &[0.0; 4]),
            Err(MediaError::StoreCorrupt)
        ));
    }

    #[test]
    fn save_load_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let h = header_dim2();
        let rows = vec![
            Row {
                time: 0.0,
                shot_start: 0.0,
                shot_end: 1.0,
            },
            Row {
                time: 1.0,
                shot_start: 1.0,
                shot_end: 2.0,
            },
        ];
        let vectors = vec![0.25f32, 0.5, 0.75, 1.0];
        save(dir.path(), "k", &h, &rows, &vectors).unwrap();
        let idx = load(dir.path(), "k").unwrap();
        assert_eq!(idx.header, h);
        assert_eq!(idx.rows, rows);
    }

    #[test]
    fn is_current_checks_version_triple() {
        let dir = tempfile::tempdir().unwrap();
        let h = header_dim2();
        save(
            dir.path(),
            "k",
            &h,
            &[
                Row {
                    time: 0.0,
                    shot_start: 0.0,
                    shot_end: 1.0,
                },
                Row {
                    time: 1.0,
                    shot_start: 1.0,
                    shot_end: 2.0,
                },
            ],
            &[0.0; 4],
        )
        .unwrap();
        assert!(is_current(
            dir.path(),
            "k",
            "siglip2-base-patch16-256",
            1,
            1
        ));
        assert!(!is_current(dir.path(), "k", "other-model", 1, 1));
        assert!(!is_current(
            dir.path(),
            "k",
            "siglip2-base-patch16-256",
            2,
            1
        ));
        assert!(!is_current(
            dir.path(),
            "k",
            "siglip2-base-patch16-256",
            1,
            2
        ));
        assert!(!is_current(
            dir.path(),
            "missing",
            "siglip2-base-patch16-256",
            1,
            1
        ));
    }

    #[test]
    fn clear_all_removes_directory() {
        let dir = tempfile::tempdir().unwrap();
        let h = header_dim2();
        save(
            dir.path(),
            "k",
            &h,
            &[
                Row {
                    time: 0.0,
                    shot_start: 0.0,
                    shot_end: 1.0,
                },
                Row {
                    time: 1.0,
                    shot_start: 1.0,
                    shot_end: 2.0,
                },
            ],
            &[0.0; 4],
        )
        .unwrap();
        assert!(dir.path().join(CACHE_SUBDIR).exists());
        clear_all(dir.path()).unwrap();
        assert!(!dir.path().join(CACHE_SUBDIR).exists());
    }

    fn one_row_index(value: f32) -> (Header, Vec<Row>, Vec<f32>) {
        let header = Header {
            model: "m".into(),
            model_version: 1,
            sampler_version: 1,
            dim: 2,
            count: 1,
        };
        let rows = vec![Row {
            time: 0.0,
            shot_start: 0.0,
            shot_end: 1.0,
        }];
        (header, rows, vec![value, 0.25])
    }

    #[test]
    fn decode_rejects_non_finite_vectors_and_rows() {
        let (header, rows, _) = one_row_index(0.0);
        for vector in [vec![f32::NAN, 0.0], vec![f32::INFINITY, 0.0]] {
            let bytes = encode(&header, &rows, &vector).unwrap();
            assert!(matches!(decode(&bytes), Err(MediaError::StoreCorrupt)));
        }
        let nan_time = [Row {
            time: f64::NAN,
            shot_start: 0.0,
            shot_end: 1.0,
        }];
        let bytes = encode(&header, &nan_time, &[0.5, 0.5]).unwrap();
        assert!(matches!(decode(&bytes), Err(MediaError::StoreCorrupt)));
    }

    #[test]
    fn concurrent_saves_leave_one_decodable_index_and_no_temporaries() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let writers: Vec<_> = [0.5f32, -0.5]
            .into_iter()
            .map(|value| {
                let root = root.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    let (header, rows, vectors) = one_row_index(value);
                    barrier.wait();
                    for _ in 0..50 {
                        save(&root, "same-key", &header, &rows, &vectors).unwrap();
                    }
                })
            })
            .collect();
        for writer in writers {
            writer.join().unwrap();
        }
        let loaded = load(&root, "same-key").unwrap();
        assert!(loaded.vectors == vec![0.5, 0.25] || loaded.vectors == vec![-0.5, 0.25]);
        let names: Vec<_> = std::fs::read_dir(root.join(CACHE_SUBDIR))
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        assert_eq!(names, vec!["same-key.embed".to_string()]);
    }

    #[test]
    fn cached_load_reuses_decoded_index_until_reindexed() {
        let dir = tempfile::tempdir().unwrap();
        let (header, rows, vectors) = one_row_index(0.5);
        save(dir.path(), "cached", &header, &rows, &vectors).unwrap();

        let first = load_cached(dir.path(), "cached").unwrap();
        let second = load_cached(dir.path(), "cached").unwrap();
        // The second query is served from memory: no new read/decode happened.
        assert!(Arc::ptr_eq(&first, &second));

        let (_, _, updated) = one_row_index(-0.5);
        save(dir.path(), "cached", &header, &rows, &updated).unwrap();
        let reindexed = load_cached(dir.path(), "cached").unwrap();
        assert!(!Arc::ptr_eq(&first, &reindexed));
        assert_eq!(reindexed.vectors, vec![-0.5, 0.25]);

        clear_all(dir.path()).unwrap();
        assert!(load_cached(dir.path(), "cached").is_err());
    }

    #[test]
    fn index_cache_evicts_least_recently_used_under_its_byte_limit() {
        let mut cache = IndexCache::default();
        let stamp = FileStamp::of(&std::fs::metadata(std::env::temp_dir()).unwrap());
        let (header, rows, vectors) = one_row_index(0.5);
        let index = Arc::new(AssetIndex {
            header,
            rows,
            vectors,
        });
        let size = 2 * 4 + std::mem::size_of::<Row>();
        cache.insert("a".into(), stamp, index.clone(), size * 2);
        cache.insert("b".into(), stamp, index.clone(), size * 2);
        assert!(cache.get(Path::new("a"), &stamp).is_some());
        cache.insert("c".into(), stamp, index.clone(), size * 2);
        assert!(cache.get(Path::new("b"), &stamp).is_none());
        assert!(cache.get(Path::new("a"), &stamp).is_some());
        assert!(cache.get(Path::new("c"), &stamp).is_some());
        assert_eq!(cache.bytes, size * 2);
    }
}
