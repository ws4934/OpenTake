//! Straight-alpha sources (FFmpeg `rgba`: PNG overlays, alpha video) must be
//! premultiplied at upload so they blend as `a*C + (1-a)*D` and bilinear
//! filtering never bleeds the color of fully transparent texels (#88).
//!
//! Skips gracefully (eprintln + early return) when no GPU device is available.

use std::rc::Rc;

use opentake_domain::{Clip, ClipType, Point, Timeline, Track, Transform};
use opentake_render::gpu::texture::upload_rgba;
use opentake_render::source::DecodedFrame;
use opentake_render::wgpu;
use opentake_render::{
    build_render_plan, Compositor, GpuTexture, RenderDevice, RenderSize, SourceMetrics,
    TextureResolver, TextureSource,
};

/// Odd width so the middle pixel center samples exactly halfway across a 2x1
/// texture stretched over the canvas.
const RS: RenderSize = RenderSize {
    width: 3,
    height: 3,
};

struct Metrics;
impl SourceMetrics for Metrics {
    fn natural_size(&self, _r: &str) -> Option<(u32, u32)> {
        Some((RS.width, RS.height))
    }
}

/// Resolves draws in plan order (bottom first) to the given frames.
struct Frames<'d> {
    device: &'d wgpu::Device,
    queue: &'d wgpu::Queue,
    frames: Vec<DecodedFrame>,
    next: usize,
}

impl TextureResolver for Frames<'_> {
    fn resolve(&mut self, _source: &TextureSource, _frame: i64) -> Option<Rc<GpuTexture>> {
        let frame = self.frames.get(self.next)?;
        self.next += 1;
        Some(Rc::new(upload_rgba(
            self.device,
            self.queue,
            frame,
            false,
            Some("straight-alpha"),
        )))
    }
}

fn solid(rgba: [u8; 4], premultiplied: bool) -> DecodedFrame {
    DecodedFrame::new(2, 2, rgba.repeat(4), premultiplied)
}

/// Full-canvas clips, listed top layer first (upstream track 0 is topmost).
fn timeline(layers: usize) -> Timeline {
    let mut tl = Timeline::new();
    tl.fps = 30;
    tl.width = RS.width as i32;
    tl.height = RS.height as i32;
    for i in 0..layers {
        let mut clip = Clip::new(format!("c{i}"), "asset", 0, 10);
        clip.transform = Transform::from_top_left(Point { x: 0.0, y: 0.0 }, 1.0, 1.0);
        let mut track = Track::new(format!("t{i}"), ClipType::Image);
        track.clips.push(clip);
        tl.tracks.push(track);
    }
    tl
}

/// Composite `frames` (bottom layer first) over the opaque black clear and
/// return the center pixel, or `None` when no GPU device is available.
fn center_pixel(test: &str, frames: Vec<DecodedFrame>) -> Option<[u8; 4]> {
    let dev = match RenderDevice::try_new() {
        Ok(dev) => dev,
        Err(e) => {
            eprintln!("[skip] {test}: no GPU device ({e})");
            return None;
        }
    };
    let tl = timeline(frames.len());
    let plan = build_render_plan(&tl, RS, &Metrics);
    let fp = plan.frame(&tl, 0);
    assert_eq!(fp.draws.len(), frames.len());
    let mut resolver = Frames {
        device: &dev.device,
        queue: &dev.queue,
        frames,
        next: 0,
    };
    let out = Compositor::new(&dev.device)
        .render_to_rgba(&dev.device, &dev.queue, RS, &fp, &mut resolver)
        .expect("render");
    let i = ((RS.height / 2) * RS.width + RS.width / 2) as usize * 4;
    Some([
        out.rgba[i],
        out.rgba[i + 1],
        out.rgba[i + 2],
        out.rgba[i + 3],
    ])
}

fn assert_near(actual: [u8; 4], expected: [u8; 4], tolerance: u8) {
    for (a, e) in actual.iter().zip(expected) {
        assert!(
            a.abs_diff(e) <= tolerance,
            "got {actual:?}, want {expected:?}"
        );
    }
}

#[test]
fn straight_half_white_over_black_is_half_grey() {
    let Some(px) = center_pixel(
        "straight_half_white_over_black_is_half_grey",
        vec![solid([255, 255, 255, 128], false)],
    ) else {
        return;
    };
    assert_near(px, [128, 128, 128, 255], 2);
}

#[test]
fn straight_quarter_red_over_white_tints_pink() {
    let Some(px) = center_pixel(
        "straight_quarter_red_over_white_tints_pink",
        vec![
            solid([255, 255, 255, 255], false),
            solid([255, 0, 0, 64], false),
        ],
    ) else {
        return;
    };
    assert_near(px, [255, 191, 191, 255], 2);
}

#[test]
fn transparent_texel_color_does_not_bleed_into_edges() {
    // Left texel: fully transparent but green; right texel: opaque red. The
    // center pixel samples halfway between them.
    let edge = DecodedFrame::new(2, 1, vec![0, 255, 0, 0, 255, 0, 0, 255], false);
    let Some(px) = center_pixel(
        "transparent_texel_color_does_not_bleed_into_edges",
        vec![edge],
    ) else {
        return;
    };
    assert!(px[1] <= 2, "green bled into the edge: {px:?}");
    assert_near(px, [128, 0, 0, 255], 2);
}

#[test]
fn premultiplied_frames_are_not_premultiplied_twice() {
    let Some(px) = center_pixel(
        "premultiplied_frames_are_not_premultiplied_twice",
        vec![solid([128, 128, 128, 128], true)],
    ) else {
        return;
    };
    assert_near(px, [128, 128, 128, 255], 2);
}

#[test]
fn opaque_straight_frames_composite_unchanged() {
    let Some(px) = center_pixel(
        "opaque_straight_frames_composite_unchanged",
        vec![solid([10, 120, 250, 255], false)],
    ) else {
        return;
    };
    assert_eq!(px, [10, 120, 250, 255]);
}
