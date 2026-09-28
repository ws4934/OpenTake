//! Texture upload + a small content-hash LRU cache (SPEC §4.4).
//!
//! Images / text / Lottie frames are keyed by an opaque content hash (computed
//! by the caller — render adds no hashing dependency) and capped by an LRU, by
//! entry count and optionally by bytes, so VRAM doesn't grow unbounded. Video
//! frames are NOT long-lived here; the compositor uploads the current frame on
//! demand.

use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use crate::source::DecodedFrame;
use opentake_domain::CubeLut;

/// A GPU texture plus a bindable view, reference-counted so a cache entry and an
/// in-flight draw can share it.
pub struct GpuTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
}

/// A filterable 3D LUT texture plus its bindable D3 view.
pub struct GpuLutTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub size: u32,
    pub domain_min: [f32; 3],
    pub domain_max: [f32; 3],
}

/// Upload a validated `.cube` table as RGBA16F. The alpha lane is padding.
pub fn upload_lut_3d(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    lut: &CubeLut,
    label: Option<&str>,
) -> GpuLutTexture {
    upload_lut_table_3d(
        device,
        queue,
        lut.size(),
        lut.domain_min(),
        lut.domain_max(),
        lut.table(),
        label,
    )
}

pub(crate) fn upload_lut_table_3d(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    lut_size: u32,
    domain_min: [f32; 3],
    domain_max: [f32; 3],
    table: &[[f32; 3]],
    label: Option<&str>,
) -> GpuLutTexture {
    let extent = wgpu::Extent3d {
        width: lut_size,
        height: lut_size,
        depth_or_array_layers: lut_size,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label,
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let mut rgba = Vec::<u16>::with_capacity(table.len() * 4);
    for value in table {
        rgba.extend(value.map(|channel| half::f16::from_f32(channel).to_bits()));
        rgba.push(half::f16::ONE.to_bits());
    }
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&rgba),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(lut_size * 8),
            rows_per_image: Some(lut_size),
        },
        extent,
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D3),
        ..Default::default()
    });
    GpuLutTexture {
        texture,
        view,
        size: lut_size,
        domain_min,
        domain_max,
    }
}

/// Upload a [`DecodedFrame`] as a premultiplied-alpha RGBA8 texture.
///
/// Every texture the compositor samples is premultiplied: a straight-alpha
/// frame (`premultiplied == false`, e.g. FFmpeg `rgba` output) is premultiplied
/// here so bilinear filtering never blends the color of fully transparent
/// texels into visible edges. Opaque frames upload unchanged. `srgb` selects the texture format: `Rgba8UnormSrgb` makes the sampler return
/// linear values (hardware sRGB decode); `Rgba8Unorm` keeps raw bytes. The PoC
/// composites in the sRGB non-linear domain (SPEC §3.7), so callers pass
/// `srgb = false` to sample raw encoded bytes and blend them directly.
pub fn upload_rgba(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    frame: &DecodedFrame,
    srgb: bool,
    label: Option<&str>,
) -> GpuTexture {
    let format = if srgb {
        wgpu::TextureFormat::Rgba8UnormSrgb
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    };
    let size = wgpu::Extent3d {
        width: frame.width,
        height: frame.height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let rgba = if frame.premultiplied {
        Cow::Borrowed(frame.rgba.as_slice())
    } else {
        premultiply_rgba(&frame.rgba)
    };
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &rgba,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(frame.width * 4),
            rows_per_image: Some(frame.height),
        },
        size,
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    GpuTexture {
        texture,
        view,
        width: frame.width,
        height: frame.height,
    }
}

/// Premultiply straight RGBA8 (`c = round(c * a / 255)`), borrowing the input
/// untouched when every pixel is opaque.
fn premultiply_rgba(rgba: &[u8]) -> Cow<'_, [u8]> {
    let pixels = rgba.as_chunks::<4>().0;
    if pixels.iter().all(|px| px[3] == u8::MAX) {
        return Cow::Borrowed(rgba);
    }
    let mut out = rgba.to_vec();
    for px in out.as_chunks_mut::<4>().0 {
        let a = u16::from(px[3]);
        for c in &mut px[..3] {
            *c = ((u16::from(*c) * a + 127) / 255) as u8;
        }
    }
    Cow::Owned(out)
}

/// GPU bytes held by a texture: `width * height * layers * texel size` of its
/// base level (4 bytes per texel for the RGBA8 formats the renderer uploads).
fn texture_bytes(tex: &GpuTexture) -> u64 {
    let texture = &tex.texture;
    let texel = texture.format().block_copy_size(None).unwrap_or(4);
    u64::from(texture.width())
        * u64::from(texture.height())
        * u64::from(texture.depth_or_array_layers())
        * u64::from(texel)
}

/// Key bookkeeping behind [`TextureCache`], generic over the value so the
/// eviction policy is testable without a GPU device.
struct LruBudget<V> {
    capacity: usize,
    byte_budget: u64,
    bytes: u64,
    peak_bytes: u64,
    map: HashMap<String, (V, u64)>,
    order: VecDeque<String>,
}

impl<V: Clone> LruBudget<V> {
    fn new(capacity: usize, byte_budget: u64) -> Self {
        LruBudget {
            capacity: capacity.max(1),
            byte_budget,
            bytes: 0,
            peak_bytes: 0,
            map: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn get(&mut self, key: &str) -> Option<V> {
        let value = self.map.get(key).map(|(value, _)| value.clone())?;
        self.touch(key);
        Some(value)
    }

    /// Insert (or replace) `key`. Least-recently-used entries are evicted
    /// before the new entry is added, until one more entry fits under the
    /// count cap and `cost` more bytes fit under the byte budget. An entry that
    /// alone exceeds the budget is still stored, as the only entry, so a single
    /// oversized frame keeps rendering from the cache.
    fn insert(&mut self, key: String, value: V, cost: u64) {
        if let Some((_, old_cost)) = self.map.remove(&key) {
            self.bytes -= old_cost;
            if let Some(pos) = self.order.iter().position(|k| *k == key) {
                self.order.remove(pos);
            }
        }
        while !self.order.is_empty()
            && (self.map.len() >= self.capacity
                || self.bytes.saturating_add(cost) > self.byte_budget)
        {
            let evict = self.order.pop_front().expect("order is non-empty");
            if let Some((_, evicted_cost)) = self.map.remove(&evict) {
                self.bytes -= evicted_cost;
            }
        }
        self.map.insert(key.clone(), (value, cost));
        self.order.push_back(key);
        self.bytes += cost;
        self.peak_bytes = self.peak_bytes.max(self.bytes);
    }

    fn touch(&mut self, key: &str) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            let k = self.order.remove(pos).expect("position valid");
            self.order.push_back(k);
        }
    }
}

/// LRU cache mapping an opaque content-hash key to a shared [`GpuTexture`].
///
/// Two limits apply: an entry count cap and a byte budget (each entry costs
/// its texture's `width * height * 4` bytes for RGBA8). Inserting evicts
/// least-recently-used entries until both hold. The one exception is a single
/// texture larger than the whole budget: it is kept as the only entry, so
/// [`TextureCache::bytes`] can exceed the budget only in that case.
///
/// Eviction only drops the cache's handle; a texture stays alive while a
/// caller or an in-flight draw still holds its `Rc`.
pub struct TextureCache {
    inner: LruBudget<Rc<GpuTexture>>,
}

impl TextureCache {
    /// New cache holding at most `capacity` textures (>= 1), with no byte
    /// budget.
    pub fn new(capacity: usize) -> Self {
        Self::with_byte_budget(capacity, u64::MAX)
    }

    /// New cache holding at most `capacity` textures (>= 1) and at most
    /// `byte_budget` bytes of texture data (see the type docs for an entry
    /// that alone exceeds the budget).
    pub fn with_byte_budget(capacity: usize, byte_budget: u64) -> Self {
        TextureCache {
            inner: LruBudget::new(capacity, byte_budget),
        }
    }

    pub fn len(&self) -> usize {
        self.inner.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.map.is_empty()
    }

    /// Maximum number of entries.
    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    /// Byte budget (`u64::MAX` when the cache was built without one).
    pub fn byte_budget(&self) -> u64 {
        self.inner.byte_budget
    }

    /// Bytes of texture data currently held by cache entries.
    pub fn bytes(&self) -> u64 {
        self.inner.bytes
    }

    /// Highest [`TextureCache::bytes`] value reached since creation.
    pub fn peak_bytes(&self) -> u64 {
        self.inner.peak_bytes
    }

    /// Fetch a cached texture, marking it most-recently-used.
    pub fn get(&mut self, key: &str) -> Option<Rc<GpuTexture>> {
        self.inner.get(key)
    }

    /// Insert (or replace) a texture, first evicting least-recently-used
    /// entries until it fits under the count cap and the byte budget. Returns
    /// the shared handle.
    pub fn insert(&mut self, key: impl Into<String>, tex: GpuTexture) -> Rc<GpuTexture> {
        let cost = texture_bytes(&tex);
        let rc = Rc::new(tex);
        self.inner.insert(key.into(), rc.clone(), cost);
        rc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // GpuTexture needs a device, so the eviction policy is tested through the
    // generic `LruBudget` bookkeeping; `tests/texture_cache_budget.rs` covers
    // the same invariants with real textures.
    #[test]
    fn capacity_floored_at_one() {
        let c = TextureCache::new(0);
        assert_eq!(c.capacity(), 1);
        assert_eq!(c.byte_budget(), u64::MAX);
        assert!(c.is_empty());
        assert_eq!((c.bytes(), c.peak_bytes()), (0, 0));
    }

    const UHD_BYTES: u64 = 3840 * 2160 * 4;
    const BUDGET: u64 = 512 * 1024 * 1024;

    fn keys(lru: &LruBudget<u32>) -> Vec<&str> {
        lru.order.iter().map(String::as_str).collect()
    }

    #[test]
    fn byte_budget_holds_after_every_insert() {
        let mut lru = LruBudget::new(64, BUDGET);
        for i in 0..100u32 {
            lru.insert(format!("k{i}"), i, UHD_BYTES);
            assert!(lru.bytes <= BUDGET, "insert {i}: {} bytes", lru.bytes);
            assert!(lru.peak_bytes <= BUDGET);
            assert!(lru.map.len() <= 64);
            assert_eq!(lru.bytes, lru.map.len() as u64 * UHD_BYTES);
        }
        // 16 UHD frames (~506 MiB) fit in 512 MiB; a 17th does not.
        assert_eq!(lru.map.len(), 16);
        assert_eq!(lru.get("k99"), Some(99));
        assert_eq!(lru.get("k84"), Some(84));
        assert_eq!(lru.get("k83"), None);
    }

    #[test]
    fn count_cap_still_applies_under_a_large_budget() {
        let mut lru = LruBudget::new(3, u64::MAX);
        for i in 0..10u32 {
            lru.insert(format!("k{i}"), i, 1);
            assert!(lru.map.len() <= 3);
        }
        assert_eq!(keys(&lru), ["k7", "k8", "k9"]);
        assert_eq!(lru.bytes, 3);
    }

    #[test]
    fn eviction_follows_recency_not_insertion_order() {
        let mut lru = LruBudget::new(64, 3 * 100);
        lru.insert("a".into(), 1, 100);
        lru.insert("b".into(), 2, 100);
        lru.insert("c".into(), 3, 100);
        assert_eq!(lru.get("a"), Some(1)); // a becomes most recent
        lru.insert("d".into(), 4, 100); // evicts b, the least recent
        assert_eq!(keys(&lru), ["c", "a", "d"]);
        assert_eq!(lru.get("b"), None);
        // A bigger entry evicts as many LRU entries as it needs.
        lru.insert("e".into(), 5, 200);
        assert_eq!(keys(&lru), ["d", "e"]);
        assert_eq!(lru.bytes, 300);
    }

    #[test]
    fn replacing_a_key_updates_its_cost_and_recency() {
        let mut lru = LruBudget::new(64, 300);
        lru.insert("a".into(), 1, 100);
        lru.insert("b".into(), 2, 100);
        lru.insert("a".into(), 3, 150);
        assert_eq!(keys(&lru), ["b", "a"]);
        assert_eq!(lru.bytes, 250);
        assert_eq!(lru.get("a"), Some(3));
        // Growing `b` past the budget evicts `a`, not `b` itself.
        lru.insert("b".into(), 4, 200);
        assert_eq!(keys(&lru), ["b"]);
        assert_eq!(lru.bytes, 200);
    }

    #[test]
    fn oversized_entry_is_kept_alone() {
        let mut lru = LruBudget::new(64, 100);
        lru.insert("small".into(), 1, 60);
        lru.insert("huge".into(), 2, 500);
        assert_eq!(keys(&lru), ["huge"]);
        assert_eq!(lru.bytes, 500);
        assert_eq!(lru.get("huge"), Some(2));
        // The next entry that fits displaces it and restores the invariant.
        lru.insert("next".into(), 3, 60);
        assert_eq!(keys(&lru), ["next"]);
        assert_eq!(lru.bytes, 60);
        assert_eq!(lru.peak_bytes, 500);
    }

    #[test]
    fn premultiply_scales_color_by_alpha() {
        let straight = [
            255, 255, 255, 128, 255, 0, 0, 64, 0, 255, 0, 0, 10, 20, 30, 255,
        ];
        assert_eq!(
            premultiply_rgba(&straight).as_ref(),
            &[128, 128, 128, 128, 64, 0, 0, 64, 0, 0, 0, 0, 10, 20, 30, 255]
        );
    }

    #[test]
    fn premultiply_borrows_opaque_frames_unchanged() {
        let opaque = [1, 2, 3, 255, 250, 128, 0, 255];
        let out = premultiply_rgba(&opaque);
        assert!(matches!(out, Cow::Borrowed(_)));
        assert_eq!(out.as_ref(), &opaque);
    }
}
