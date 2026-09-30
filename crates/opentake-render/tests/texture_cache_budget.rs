//! `TextureCache` byte budget with real GPU textures (issue #69): entries are
//! charged their texture's actual size, both the byte budget and the entry cap
//! hold after every insert, and hits return the cached handle unchanged.
//!
//! GPU tests skip when no adapter exists unless `OPENTAKE_REQUIRE_GPU` is set,
//! in which case a missing adapter fails the test.

use std::rc::Rc;

use opentake_render::gpu::texture::upload_rgba;
use opentake_render::source::DecodedFrame;
use opentake_render::wgpu;
use opentake_render::{GpuTexture, RenderDevice, TextureCache};

const UHD_BYTES: u64 = 3840 * 2160 * 4;
const BUDGET: u64 = 512 * 1024 * 1024;

fn device_or_skip(test: &str) -> Option<RenderDevice> {
    match RenderDevice::try_new() {
        Ok(d) => Some(d),
        Err(e) => {
            assert!(
                std::env::var_os("OPENTAKE_REQUIRE_GPU").is_none(),
                "{test}: OPENTAKE_REQUIRE_GPU is set but no GPU device is available ({e})"
            );
            eprintln!("[skip] {test}: no GPU device ({e})");
            None
        }
    }
}

/// An RGBA8 texture of the given size; its contents are never sampled here.
fn blank_texture(device: &wgpu::Device, width: u32, height: u32) -> GpuTexture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("budget test"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    GpuTexture {
        texture,
        view: view.into(),
        width,
        height,
    }
}

#[test]
fn uhd_textures_stay_under_byte_budget() {
    let Some(dev) = device_or_skip("uhd_textures_stay_under_byte_budget") else {
        return;
    };
    // Export's entry cap: 64 UHD frames would be ~2.1 GB without a budget.
    let mut cache = TextureCache::with_byte_budget(64, BUDGET);
    assert_eq!(cache.byte_budget(), BUDGET);
    for i in 0..100 {
        let handle = cache.insert(format!("frame{i}"), blank_texture(&dev.device, 3840, 2160));
        assert_eq!((handle.width, handle.height), (3840, 2160));
        assert!(
            cache.bytes() <= BUDGET,
            "insert {i}: {} bytes",
            cache.bytes()
        );
        assert!(cache.peak_bytes() <= BUDGET);
        assert!(cache.len() <= 64);
        assert_eq!(cache.bytes(), cache.len() as u64 * UHD_BYTES);
    }
    assert_eq!(cache.len(), 16);
    assert_eq!(cache.peak_bytes(), 16 * UHD_BYTES);
    // The 16 most recent frames survive; older ones were evicted in order.
    assert!(cache.get("frame83").is_none());
    for i in 84..100 {
        assert!(
            cache.get(&format!("frame{i}")).is_some(),
            "frame{i} evicted"
        );
    }
}

#[test]
fn byte_budget_respects_recency_and_count_cap() {
    let Some(dev) = device_or_skip("byte_budget_respects_recency_and_count_cap") else {
        return;
    };
    // Room for three 64x64 textures by bytes, four by count.
    let mut cache = TextureCache::with_byte_budget(4, 3 * 64 * 64 * 4);
    for key in ["a", "b", "c"] {
        cache.insert(key, blank_texture(&dev.device, 64, 64));
    }
    assert!(cache.get("a").is_some()); // `b` is now least recent
    cache.insert("d", blank_texture(&dev.device, 64, 64));
    assert!(cache.get("b").is_none());
    assert_eq!(cache.len(), 3);

    // Count cap still applies when the bytes would fit.
    let mut capped = TextureCache::with_byte_budget(2, u64::MAX);
    for key in ["a", "b", "c"] {
        capped.insert(key, blank_texture(&dev.device, 8, 8));
    }
    assert_eq!(capped.len(), 2);
    assert!(capped.get("a").is_none());
    assert_eq!(capped.bytes(), 2 * 8 * 8 * 4);

    // A texture larger than the whole budget is kept as the only entry.
    cache.insert("huge", blank_texture(&dev.device, 256, 256));
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.bytes(), 256 * 256 * 4);
    assert!(cache.get("huge").is_some());
}

#[test]
fn cache_hits_return_the_inserted_texture() {
    let Some(dev) = device_or_skip("cache_hits_return_the_inserted_texture") else {
        return;
    };
    let frame = DecodedFrame::new(2, 1, vec![10, 20, 30, 255, 40, 50, 60, 255], true);
    let mut cache = TextureCache::with_byte_budget(8, 1024);
    let inserted = cache.insert(
        "still",
        upload_rgba(&dev.device, &dev.queue, &frame, false, None),
    );
    let hit = cache.get("still").expect("cache hit");
    assert!(Rc::ptr_eq(&inserted, &hit));
    assert_eq!((hit.width, hit.height), (2, 1));
    assert_eq!(cache.bytes(), 2 * 4);
    assert!(cache.get("missing").is_none());
}
