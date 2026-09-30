//! Compositor GPU resource reuse (issue #119): the render target, readback
//! buffer and per-layer uniform buffers are cached across frames and rebuilt
//! only when the canvas size changes, without changing composite pixels.
//!
//! GPU tests skip when no adapter exists unless `OPENTAKE_REQUIRE_GPU` is set,
//! in which case a missing adapter fails the test.

use std::collections::HashMap;
use std::rc::Rc;
use std::time::Instant;

use opentake_domain::{Clip, ClipType, Point, Timeline, Track, Transform};
use opentake_render::gpu::texture::upload_rgba;
use opentake_render::source::DecodedFrame;
use opentake_render::wgpu;
use opentake_render::{
    build_render_plan, Compositor, CompositorResourceStats, GpuTexture, RenderDevice, RenderSize,
    SourceMetrics, TextureResolver, TextureSource,
};

const LAYERS: usize = 4;

struct Metrics;
impl SourceMetrics for Metrics {
    fn natural_size(&self, _r: &str) -> Option<(u32, u32)> {
        Some((1920, 1080))
    }
}

/// Resolves `asset<i>` to a gradient texture uploaded once up front.
struct PreloadedResolver {
    textures: HashMap<String, Rc<GpuTexture>>,
}

impl PreloadedResolver {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32) -> Self {
        let textures = (0..LAYERS)
            .map(|i| {
                let mut rgba = vec![0u8; (width * height * 4) as usize];
                for (p, px) in rgba.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    let x = (p as u32 % width) * 255 / width.max(1);
                    let y = (p as u32 / width) * 255 / height.max(1);
                    *px = [x as u8, y as u8, (i as u8).wrapping_mul(60), 255];
                }
                let frame = DecodedFrame::new(width, height, rgba, true);
                let tex = upload_rgba(device, queue, &frame, false, Some("layer"));
                (format!("asset{i}"), Rc::new(tex))
            })
            .collect();
        PreloadedResolver { textures }
    }
}

impl TextureResolver for PreloadedResolver {
    fn resolve(&mut self, source: &TextureSource, _frame: i64) -> Option<Rc<GpuTexture>> {
        match source {
            TextureSource::Decoded { media_ref } | TextureSource::Image { media_ref } => {
                self.textures.get(media_ref).cloned()
            }
            _ => None,
        }
    }
}

/// Four overlapping, partially transparent video layers so every draw
/// contributes to the blended result.
fn four_layer_timeline(size: RenderSize) -> Timeline {
    let mut tl = Timeline::new();
    tl.fps = 30;
    tl.width = size.width as i32;
    tl.height = size.height as i32;
    for i in 0..LAYERS {
        let mut clip = Clip::new(format!("c{i}"), format!("asset{i}"), 0, 10);
        let scale = 1.0 - i as f64 * 0.2;
        let offset = i as f64 * 0.08;
        clip.transform = Transform::from_top_left(
            Point {
                x: offset,
                y: offset,
            },
            scale,
            scale,
        );
        clip.opacity = 0.6;
        let mut track = Track::new(format!("t{i}"), ClipType::Video);
        track.clips.push(clip);
        tl.tracks.push(track);
    }
    tl
}

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

fn render(
    compositor: &Compositor,
    dev: &RenderDevice,
    resolver: &mut PreloadedResolver,
    size: RenderSize,
) -> Vec<u8> {
    let tl = four_layer_timeline(size);
    let plan = build_render_plan(&tl, size, &Metrics);
    let fp = plan.frame(&tl, 0);
    assert_eq!(fp.draws.len(), LAYERS);
    compositor
        .render_to_rgba(&dev.device, &dev.queue, size, &fp, resolver)
        .expect("render")
        .rgba
}

/// Timing probe for issue #119, not run by default and without wall-clock
/// assertions: composites a 1920x1080 frame with four layers N times and
/// prints the mean per-frame time. Run with
/// `cargo test --release -p opentake-render --test compositor_reuse -- --ignored --nocapture`.
#[test]
#[ignore = "timing probe; run manually with --ignored --nocapture"]
fn composite_1080p_four_layers_timing() {
    const N: u32 = 50;
    let Some(dev) = device_or_skip("composite_1080p_four_layers_timing") else {
        return;
    };
    let size = RenderSize {
        width: 1920,
        height: 1080,
    };
    let compositor = Compositor::new(&dev.device);
    let mut resolver = PreloadedResolver::new(&dev.device, &dev.queue, 1920, 1080);
    let tl = four_layer_timeline(size);
    let plan = build_render_plan(&tl, size, &Metrics);
    let fp = plan.frame(&tl, 0);
    assert_eq!(fp.draws.len(), LAYERS);

    // Warm-up frame: pipeline compilation and texture uploads are not measured.
    compositor
        .render_to_rgba(&dev.device, &dev.queue, size, &fp, &mut resolver)
        .expect("warm-up render");

    let start = Instant::now();
    for _ in 0..N {
        let frame = compositor
            .render_to_rgba(&dev.device, &dev.queue, size, &fp, &mut resolver)
            .expect("render");
        std::hint::black_box(frame);
    }
    let mean = start.elapsed() / N;
    println!("composite_1080p_four_layers_timing: {N} frames, mean {mean:?} per frame");
}

const SMALL: RenderSize = RenderSize {
    width: 64,
    height: 36,
};
// 128 px * 4 bytes = 512: rows need no padding, unlike SMALL's 256-byte rows.
const ALIGNED: RenderSize = RenderSize {
    width: 128,
    height: 72,
};

#[test]
fn warm_cache_creates_targets_once_per_size() {
    let Some(dev) = device_or_skip("warm_cache_creates_targets_once_per_size") else {
        return;
    };
    let compositor = Compositor::new(&dev.device);
    let mut resolver = PreloadedResolver::new(&dev.device, &dev.queue, 32, 18);
    assert_eq!(
        compositor.resource_stats(),
        CompositorResourceStats::default()
    );

    for _ in 0..100 {
        render(&compositor, &dev, &mut resolver, SMALL);
    }
    let warm = compositor.resource_stats();
    assert_eq!(warm.render_targets_created, 1);
    assert_eq!(warm.readback_buffers_created, 1);
    assert_eq!(warm.uniform_buffers_created, LAYERS);
    assert_eq!(warm.bind_groups_created, LAYERS);

    // A size change rebuilds the size-keyed targets exactly once; the uniform
    // pool is size-independent and is kept.
    for _ in 0..10 {
        render(&compositor, &dev, &mut resolver, ALIGNED);
    }
    let resized = compositor.resource_stats();
    assert_eq!(resized.render_targets_created, 2);
    assert_eq!(resized.readback_buffers_created, 2);
    assert_eq!(resized.uniform_buffers_created, LAYERS);
    assert_eq!(resized.bind_groups_created, LAYERS);
}

#[test]
fn warm_cache_frames_match_first_frame_pixels() {
    let Some(dev) = device_or_skip("warm_cache_frames_match_first_frame_pixels") else {
        return;
    };
    let compositor = Compositor::new(&dev.device);
    let mut resolver = PreloadedResolver::new(&dev.device, &dev.queue, 32, 18);

    let first = render(&compositor, &dev, &mut resolver, SMALL);
    assert_eq!(first.len(), (SMALL.width * SMALL.height * 4) as usize);
    for i in 0..20 {
        assert_eq!(
            render(&compositor, &dev, &mut resolver, SMALL),
            first,
            "warm frame {i} differs from the first frame"
        );
    }

    // Another size, then back: each matches a cold compositor's output.
    let aligned = render(&compositor, &dev, &mut resolver, ALIGNED);
    let cold_aligned = render(&Compositor::new(&dev.device), &dev, &mut resolver, ALIGNED);
    assert_eq!(
        aligned, cold_aligned,
        "resized frame differs from a cold render"
    );
    assert_eq!(
        render(&compositor, &dev, &mut resolver, SMALL),
        first,
        "returning to the original size changed the pixels"
    );
    assert_eq!(compositor.resource_stats().render_targets_created, 3);
}

#[test]
fn concurrent_renders_on_one_compositor_match() {
    let Some(dev) = device_or_skip("concurrent_renders_on_one_compositor_match") else {
        return;
    };
    let compositor = Compositor::new(&dev.device);
    let mut resolver = PreloadedResolver::new(&dev.device, &dev.queue, 32, 18);
    let expected = render(&compositor, &dev, &mut resolver, SMALL);

    // A render that finds the cache busy falls back to transient resources;
    // either way every frame must be identical.
    std::thread::scope(|scope| {
        for _ in 0..2 {
            scope.spawn(|| {
                let mut resolver = PreloadedResolver::new(&dev.device, &dev.queue, 32, 18);
                for _ in 0..20 {
                    assert_eq!(render(&compositor, &dev, &mut resolver, SMALL), expected);
                }
            });
        }
    });
}

/// The Tauri shell keeps compositors in shared state and on worker threads;
/// the frame cache must not make `Compositor` `!Send` or `!Sync`.
#[test]
fn compositor_stays_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Compositor>();
}

#[test]
fn changed_sources_and_empty_frames_invalidate_binding_slots() {
    let Some(dev) = device_or_skip("changed_sources_and_empty_frames_invalidate_binding_slots")
    else {
        return;
    };
    let compositor = Compositor::new(&dev.device);
    let mut resolver = PreloadedResolver::new(&dev.device, &dev.queue, 32, 18);
    let first = render(&compositor, &dev, &mut resolver, SMALL);
    let old_view = std::sync::Arc::downgrade(&resolver.textures["asset0"].view);
    let replacement = DecodedFrame::new(32, 18, [255, 0, 255, 255].repeat(32 * 18), true);
    resolver.textures.insert(
        "asset0".into(),
        Rc::new(upload_rgba(
            &dev.device,
            &dev.queue,
            &replacement,
            false,
            None,
        )),
    );
    assert!(
        old_view.upgrade().is_none(),
        "cache must not retain the source wrapper"
    );
    let changed = render(&compositor, &dev, &mut resolver, SMALL);
    assert_ne!(changed, first);
    assert_eq!(
        changed,
        render(&Compositor::new(&dev.device), &dev, &mut resolver, SMALL)
    );
    assert_eq!(compositor.resource_stats().bind_groups_created, LAYERS + 1);

    let empty = Timeline::new();
    let plan = build_render_plan(&empty, SMALL, &Metrics);
    compositor
        .render_to_rgba(
            &dev.device,
            &dev.queue,
            SMALL,
            &plan.frame(&empty, 0),
            &mut resolver,
        )
        .unwrap();
    assert_eq!(render(&compositor, &dev, &mut resolver, SMALL), changed);
    assert_eq!(
        compositor.resource_stats().bind_groups_created,
        2 * LAYERS + 1
    );
}

#[test]
fn cached_bindings_still_apply_changed_uniforms() {
    let Some(dev) = device_or_skip("cached_bindings_still_apply_changed_uniforms") else {
        return;
    };
    let compositor = Compositor::new(&dev.device);
    let mut resolver = PreloadedResolver::new(&dev.device, &dev.queue, 32, 18);
    let first = render(&compositor, &dev, &mut resolver, SMALL);
    let mut timeline = four_layer_timeline(SMALL);
    timeline.tracks[0].clips[0].opacity = 0.1;
    let plan = build_render_plan(&timeline, SMALL, &Metrics);
    let frame = plan.frame(&timeline, 0);
    let warm = compositor
        .render_to_rgba(&dev.device, &dev.queue, SMALL, &frame, &mut resolver)
        .unwrap();
    let cold = Compositor::new(&dev.device)
        .render_to_rgba(&dev.device, &dev.queue, SMALL, &frame, &mut resolver)
        .unwrap();
    assert_ne!(warm.rgba, first);
    assert_eq!(warm.rgba, cold.rgba);
    assert_eq!(compositor.resource_stats().bind_groups_created, LAYERS);
}
