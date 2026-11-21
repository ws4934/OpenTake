//! Forced-compositing readback latency probe (issue #259).
//!
//! Composites deterministic multi-track frames at 1080p and 4K and reports,
//! per frame, the CPU time spent preparing and submitting the frame, the
//! time spent in `PendingReadback::finish` (GPU wait, map and copy), and the
//! end-to-end latency from submission to pixels, as median and P95, plus the
//! throughput and the compositor's resource counts. Each fixture runs with
//! one frame in flight (preview/capture) and with the two-frame pipeline
//! export uses. The probe makes no wall-clock assertions; it only checks that
//! both schedules produce identical pixels. Run with
//! `cargo test --release -p opentake-render --test readback_latency -- --ignored --nocapture --test-threads=1`.
//! `OPENTAKE_PROBE_FRAMES` overrides the measured frame count (default 60).
//!
//! GPU tests skip when no adapter exists unless `OPENTAKE_REQUIRE_GPU` is set,
//! in which case a missing adapter fails the test.

use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::time::{Duration, Instant};

use opentake_domain::{Clip, ClipType, Point, Timeline, Track, Transform};
use opentake_render::gpu::compositor::{PendingReadback, TextureInterpolationConfig};
use opentake_render::gpu::texture::upload_rgba;
use opentake_render::source::DecodedFrame;
use opentake_render::wgpu;
use opentake_render::{
    build_render_plan, Compositor, CompositorResourceStats, GpuTexture, RenderDevice, RenderPlan,
    RenderSize, SourceMetrics, TextureResolver, TextureSource,
};

const LAYERS: usize = 6;
const WARMUP_FRAMES: usize = 3;

struct Metrics(RenderSize);
impl SourceMetrics for Metrics {
    fn natural_size(&self, _r: &str) -> Option<(u32, u32)> {
        Some((self.0.width, self.0.height))
    }
}

struct PreloadedResolver {
    textures: HashMap<String, Rc<GpuTexture>>,
}

impl PreloadedResolver {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, size: RenderSize) -> Self {
        let (width, height) = (size.width, size.height);
        let textures = (0..LAYERS)
            .map(|i| {
                let mut rgba = vec![0u8; (width * height * 4) as usize];
                for (p, px) in rgba.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    let x = (p as u32 % width) * 255 / width;
                    let y = (p as u32 / width) * 255 / height;
                    *px = [x as u8, y as u8, (i as u8).wrapping_mul(40), 255];
                }
                let frame = DecodedFrame::new(width, height, rgba, true);
                let tex = upload_rgba(device, queue, &frame, false, Some("probe layer"));
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

/// Full-canvas base track plus overlapping, translucent, scaled overlays, so
/// no frame can take the direct-source path and every draw is blended.
fn multi_track_timeline(size: RenderSize) -> Timeline {
    let mut tl = Timeline::new();
    tl.fps = 30;
    tl.width = size.width as i32;
    tl.height = size.height as i32;
    for i in 0..LAYERS {
        let mut clip = Clip::new(format!("c{i}"), format!("asset{i}"), 0, 10_000);
        if i > 0 {
            let scale = 1.0 - i as f64 * 0.12;
            let offset = i as f64 * 0.05;
            clip.transform = Transform::from_top_left(
                Point {
                    x: offset,
                    y: offset,
                },
                scale,
                scale,
            );
            clip.opacity = 0.65;
        }
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

fn adapter_description() -> String {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .map_or_else(
        || "unknown adapter".to_string(),
        |adapter| {
            let info = adapter.get_info();
            format!(
                "{} ({:?}, {:?}, driver {} {})",
                info.name, info.backend, info.device_type, info.driver, info.driver_info
            )
        },
    )
}

fn probe_frames() -> usize {
    std::env::var("OPENTAKE_PROBE_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(60)
}

struct Fixture<'a> {
    compositor: &'a Compositor,
    dev: &'a RenderDevice,
    size: RenderSize,
    timeline: Timeline,
    plan: RenderPlan,
}

impl<'a> Fixture<'a> {
    fn submit(&self, resolver: &mut PreloadedResolver, frame: i32) -> PendingReadback<'a> {
        let frame_plan = self.plan.frame(&self.timeline, frame);
        assert_eq!(frame_plan.draws.len(), LAYERS);
        self.compositor
            .render_pending_with_interpolation(
                &self.dev.device,
                &self.dev.queue,
                self.size,
                &frame_plan,
                resolver,
                TextureInterpolationConfig::passthrough(),
            )
            .expect("submit composite")
    }
}

struct Sample {
    submit: Duration,
    finish: Duration,
    latency: Duration,
}

struct Run {
    samples: Vec<Sample>,
    elapsed: Duration,
    first_frame: Vec<u8>,
    stats: CompositorResourceStats,
}

/// Composite `frames` frames keeping at most `depth` submissions in flight,
/// finishing the oldest before submitting past the bound (export's schedule
/// for `depth == 2`, preview/capture's for `depth == 1`).
fn run(
    fixture: &Fixture<'_>,
    resolver: &mut PreloadedResolver,
    frames: usize,
    depth: usize,
) -> Run {
    let mut samples = Vec::with_capacity(frames);
    let mut in_flight: VecDeque<(Instant, PendingReadback<'_>)> = VecDeque::new();
    let mut first_frame = None;
    let mut finish_one = |in_flight: &mut VecDeque<(Instant, PendingReadback<'_>)>,
                          samples: &mut Vec<Sample>,
                          submits: &VecDeque<Duration>| {
        let (submitted_at, pending) = in_flight.pop_front().expect("frame in flight");
        let start = Instant::now();
        let frame = pending.finish().expect("finish composite");
        let done = Instant::now();
        let submit = submits[samples.len()];
        samples.push(Sample {
            submit,
            finish: done - start,
            latency: done - submitted_at,
        });
        if first_frame.is_none() {
            first_frame = Some(frame.rgba);
        }
    };
    let mut submits = VecDeque::with_capacity(frames);
    let start = Instant::now();
    for f in 0..frames {
        if in_flight.len() == depth {
            finish_one(&mut in_flight, &mut samples, &submits);
        }
        let submitted_at = Instant::now();
        let pending = fixture.submit(resolver, f as i32);
        submits.push_back(submitted_at.elapsed());
        in_flight.push_back((submitted_at, pending));
    }
    while !in_flight.is_empty() {
        finish_one(&mut in_flight, &mut samples, &submits);
    }
    let elapsed = start.elapsed();
    drop(finish_one);
    Run {
        samples,
        elapsed,
        first_frame: first_frame.expect("at least one frame"),
        stats: fixture.compositor.resource_stats(),
    }
}

fn percentile(values: &mut [Duration], p: f64) -> Duration {
    values.sort_unstable();
    let rank = ((values.len() as f64 - 1.0) * p).round() as usize;
    values[rank]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn report(label: &str, run: &Run) {
    let column = |pick: fn(&Sample) -> Duration| -> (f64, f64) {
        let mut values: Vec<Duration> = run.samples.iter().map(pick).collect();
        (
            ms(percentile(&mut values, 0.5)),
            ms(percentile(&mut values, 0.95)),
        )
    };
    let (submit_p50, submit_p95) = column(|s| s.submit);
    let (finish_p50, finish_p95) = column(|s| s.finish);
    let (latency_p50, latency_p95) = column(|s| s.latency);
    let fps = run.samples.len() as f64 / run.elapsed.as_secs_f64();
    println!(
        "{label}: frames={} submit_ms p50={submit_p50:.2} p95={submit_p95:.2} \
         finish_ms p50={finish_p50:.2} p95={finish_p95:.2} \
         latency_ms p50={latency_p50:.2} p95={latency_p95:.2} fps={fps:.1} \
         targets={} readbacks={} uniforms={} bind_groups={}",
        run.samples.len(),
        run.stats.render_targets_created,
        run.stats.readback_buffers_created,
        run.stats.uniform_buffers_created,
        run.stats.bind_groups_created,
    );
}

fn probe(name: &str, size: RenderSize) {
    let Some(dev) = device_or_skip(name) else {
        return;
    };
    println!("{name}: adapter {}", adapter_description());
    let frames = probe_frames();
    let mut resolver = PreloadedResolver::new(&dev.device, &dev.queue, size);
    let timeline = multi_track_timeline(size);

    let mut reference = None;
    for depth in [1, 2] {
        let compositor = Compositor::new(&dev.device);
        let fixture = Fixture {
            compositor: &compositor,
            dev: &dev,
            size,
            plan: build_render_plan(&timeline, size, &Metrics(size)),
            timeline: timeline.clone(),
        };
        // Warm-up: pipeline compilation and first allocations are not measured.
        run(&fixture, &mut resolver, WARMUP_FRAMES, depth);
        let measured = run(&fixture, &mut resolver, frames, depth);
        report(
            &format!("{name} {}x{} depth={depth}", size.width, size.height),
            &measured,
        );
        assert!(measured.stats.render_targets_created <= depth);
        match &reference {
            None => reference = Some(measured.first_frame),
            Some(expected) => assert!(
                *expected == measured.first_frame,
                "depth {depth} changed composite pixels"
            ),
        }
    }
}

#[test]
#[ignore = "timing probe; run manually with --ignored --nocapture"]
fn forced_composite_readback_1080p() {
    probe(
        "forced_composite_readback_1080p",
        RenderSize {
            width: 1920,
            height: 1080,
        },
    );
}

#[test]
#[ignore = "timing probe; run manually with --ignored --nocapture"]
fn forced_composite_readback_4k() {
    probe(
        "forced_composite_readback_4k",
        RenderSize {
            width: 3840,
            height: 2160,
        },
    );
}
