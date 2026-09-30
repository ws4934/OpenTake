//! Fail-closed live HTTP transport integration for native playback.
#![cfg(feature = "playback-engine")]

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use opentake_domain::{
    Clip, ClipType, MediaManifest, MediaManifestEntry, MediaSource, Timeline, Track,
};
use opentake_media::{decode_frame_at, FrameRequest, RgbaFrame};
use opentake_render::{build_render_plan, source_frame_index, DecodedFrame, RenderSize};
use opentake_tauri_lib::playback::session::PlaybackIdentity;
use opentake_tauri_lib::playback::transport::PublicationGate;
use opentake_tauri_lib::playback::{
    project_media, project_text, ManifestMetrics, PreviewServer, RenderLoop,
};

struct HttpHead {
    status: u16,
    headers: BTreeMap<String, String>,
}

fn port_of(endpoint: &str) -> u16 {
    endpoint
        .rsplit(':')
        .next()
        .and_then(|tail| tail.split('/').next())
        .and_then(|port| port.parse().ok())
        .expect("endpoint carries a port")
}

fn start_server() -> Arc<PreviewServer> {
    tauri::async_runtime::block_on(PreviewServer::start())
        .expect("preview server must bind its loopback port")
}

fn make_distinct_cfr_video(path: &Path, w: u32, h: u32, fps: u32, frames: u32) {
    let output = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            &format!("testsrc2=size={w}x{h}:rate={fps}"),
            "-frames:v",
            &frames.to_string(),
            "-c:v",
            "libx264",
            "-g",
            &frames.to_string(),
            "-keyint_min",
            &frames.to_string(),
            "-sc_threshold",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-fps_mode",
            "cfr",
            "-y",
        ])
        .arg(path)
        .output()
        .expect("required ffmpeg must start for transport parity fixture");
    assert!(
        output.status.success(),
        "generate transport parity fixture: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn external_entry(id: &str, path: &Path, w: i32, h: i32, fps: f64) -> MediaManifestEntry {
    MediaManifestEntry {
        id: id.into(),
        name: format!("{id}.mp4"),
        kind: ClipType::Video,
        source: MediaSource::External {
            absolute_path: path.into(),
        },
        duration: 1.0,
        generation_input: None,
        source_width: Some(w),
        source_height: Some(h),
        source_fps: Some(fps),
        has_audio: Some(false),
        color: None,
        proxy: None,
        folder_id: None,
        cached_remote_url: None,
        cached_remote_url_expires_at: None,
    }
}

fn build_plan(
    timeline: &Timeline,
    manifest: &MediaManifest,
    render_size: RenderSize,
) -> opentake_render::RenderPlan {
    let (sizes, _) = project_media(manifest, &None);
    let metrics = ManifestMetrics { sizes };
    build_render_plan(timeline, render_size, &metrics)
}

fn require_render_loop(
    timeline: Timeline,
    manifest: &MediaManifest,
    render_size: RenderSize,
) -> RenderLoop {
    let (sizes, media) = project_media(manifest, &None);
    let text = project_text(&timeline);
    RenderLoop::new(timeline, media, text, sizes, render_size)
        .unwrap_or_else(|error| panic!("render loop init failed: {error}"))
}

fn decode_exact_source_frame(
    path: &Path,
    source_frame: i64,
    fps: u32,
    width: u32,
    height: u32,
) -> RgbaFrame {
    let request = FrameRequest {
        time_secs: source_frame as f64 / fps as f64,
        max_size: (width, height),
        apply_rotation: true,
    };
    let (_, frame) = decode_frame_at(path, &request)
        .unwrap_or_else(|error| panic!("decode exact source frame {source_frame}: {error}"));
    frame
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn open_request(port: u16, path: &str, extra_headers: &str) -> (TcpStream, HttpHead, Vec<u8>) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect loopback");
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .expect("set read timeout");
    let request = format!(
        "GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n{extra_headers}Connection: close\r\n\r\n"
    );
    stream.write_all(request.as_bytes()).expect("write request");

    let mut received = Vec::new();
    let header_end = loop {
        if let Some(index) = find_bytes(&received, b"\r\n\r\n") {
            break index + 4;
        }
        let mut chunk = [0u8; 1024];
        let count = stream.read(&mut chunk).expect("read response head");
        assert!(count > 0, "connection closed before response head");
        received.extend_from_slice(&chunk[..count]);
        assert!(received.len() <= 64 * 1024, "response head is unbounded");
    };
    let head_text = std::str::from_utf8(&received[..header_end]).expect("ASCII response head");
    let mut lines = head_text.split("\r\n");
    let status = lines
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|status| status.parse().ok())
        .expect("numeric HTTP status");
    let mut headers = BTreeMap::new();
    for line in lines.filter(|line| !line.is_empty()) {
        let (name, value) = line.split_once(':').expect("well-formed response header");
        headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
    }
    let remainder = received.split_off(header_end);
    (stream, HttpHead { status, headers }, remainder)
}

fn finite_get(port: u16, path: &str, extra_headers: &str) -> (HttpHead, Vec<u8>) {
    let (mut stream, head, mut body) = open_request(port, path, extra_headers);
    let content_length = head
        .headers
        .get("content-length")
        .map(|value| value.parse::<usize>().expect("numeric Content-Length"))
        .unwrap_or_else(|| {
            assert_eq!(head.status, 204, "finite response needs Content-Length");
            0
        });
    assert!(
        body.len() <= content_length,
        "received more than declared Content-Length"
    );
    let already_read = body.len();
    body.resize(content_length, 0);
    stream
        .read_exact(&mut body[already_read..])
        .expect("read complete Content-Length body");
    (head, body)
}

fn frame_path(identity: &PlaybackIdentity, frame: i32, sequence: u64) -> String {
    format!(
        "/frame?projectEpoch={}&timelineVersion={}&sessionId={}&frame={frame}&sequence={sequence}",
        identity.project_epoch, identity.timeline_version, identity.session_id
    )
}

fn solid_frame(width: u32, height: u32, rgb: [u8; 3]) -> DecodedFrame {
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for _ in 0..width * height {
        rgba.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
    }
    DecodedFrame::new(width, height, rgba, false)
}

#[test]
fn frame_route_transitions_from_204_to_valid_200_jpeg() {
    let server = start_server();
    let port = port_of(&server.endpoint_frame());
    let identity = PlaybackIdentity::new(7, 11, "session-frame-transition").unwrap();
    let path = frame_path(&identity, 4, 1);
    let (empty, body) = finite_get(port, &path, "");
    assert_eq!(empty.status, 204);
    assert!(body.is_empty());

    let sink = server.sink(
        identity.clone(),
        PublicationGate::open(),
        20,
        Arc::new(|_| {}),
    );
    sink.publish_now(4, &solid_frame(3, 2, [220, 20, 20]))
        .expect("commit frame");

    let (ready, jpeg) = finite_get(port, &path, "");
    assert_eq!(ready.status, 200);
    assert_eq!(
        ready.headers.get("content-type").map(String::as_str),
        Some("image/jpeg")
    );
    let image = image::load_from_memory(&jpeg).expect("decode complete JPEG body");
    assert_eq!((image.width(), image.height()), (3, 2));
}

#[test]
fn frame_route_returns_complete_decodable_jpeg_body() {
    let server = start_server();
    let port = port_of(&server.endpoint_frame());
    let identity = PlaybackIdentity::new(3, 5, "session-complete-body").unwrap();
    let sink = server.sink(
        identity.clone(),
        PublicationGate::open(),
        30,
        Arc::new(|_| {}),
    );
    sink.publish_now(9, &solid_frame(5, 4, [10, 200, 40]))
        .expect("commit frame");

    let (head, jpeg) = finite_get(port, &frame_path(&identity, 9, 1), "");
    assert_eq!(head.status, 200);
    assert_eq!(
        head.headers["content-length"].parse::<usize>().unwrap(),
        jpeg.len()
    );
    let image = image::load_from_memory(&jpeg).expect("decode complete JPEG body");
    assert_eq!((image.width(), image.height()), (5, 4));
}

#[test]
fn frame_route_rejects_cross_origin() {
    let server = start_server();
    let identity = PlaybackIdentity::new(1, 0, "session-origin").unwrap();
    let (head, _) = finite_get(
        port_of(&server.endpoint_frame()),
        &frame_path(&identity, 0, 1),
        "Origin: http://127.0.0.1.evil.example\r\n",
    );
    assert_eq!(head.status, 403);
}

#[test]
fn frame_route_returns_204_for_wrong_session_identity() {
    let server = start_server();
    let port = port_of(&server.endpoint_frame());
    let identity = PlaybackIdentity::new(2, 8, "session-current").unwrap();
    let sink = server.sink(
        identity.clone(),
        PublicationGate::open(),
        10,
        Arc::new(|_| {}),
    );
    sink.publish_now(6, &solid_frame(2, 2, [80, 90, 100]))
        .expect("commit frame");

    let wrong = PlaybackIdentity::new(2, 8, "session-replaced").unwrap();
    let (head, body) = finite_get(port, &frame_path(&wrong, 6, 1), "");
    assert_eq!(head.status, 204);
    assert!(body.is_empty());
}

#[test]
fn stream_routes_are_not_served() {
    let server = start_server();
    let port = port_of(&server.endpoint_frame());
    for path in ["/stream", "/ws"] {
        let (head, _) = finite_get(port, path, "");
        assert_eq!(head.status, 404, "{path} must not exist");
    }
}

#[test]
fn frame_route_rejects_a_rebinding_host() {
    let server = start_server();
    let port = port_of(&server.endpoint_frame());
    let identity = PlaybackIdentity::new(1, 0, "session-rebinding").unwrap();
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect loopback");
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .expect("set read timeout");
    write!(
        stream,
        "GET {} HTTP/1.1\r\nHost: attacker.example:{port}\r\nConnection: close\r\n\r\n",
        frame_path(&identity, 0, 1)
    )
    .expect("write request");
    let mut response = Vec::new();
    stream.read_to_end(&mut response).expect("read response");
    assert!(
        response.starts_with(b"HTTP/1.1 403"),
        "{}",
        String::from_utf8_lossy(&response)
    );
}

#[test]
fn frame_route_first_publication_matches_fractional_speed_plan() {
    let dir = tempfile::tempdir().expect("fixture tempdir");
    let src = dir.path().join("speed-15-cfr.mp4");
    let (w, h, fps, frames) = (160u32, 90u32, 12u32, 12u32);
    make_distinct_cfr_video(&src, w, h, fps, frames);

    let mut timeline = Timeline::new();
    timeline.fps = fps as i32;
    let mut track = Track::new("t1", ClipType::Video);
    let mut clip = Clip::new("clip-1", "asset-1", 0, 6);
    clip.trim_start_frame = 2;
    clip.speed = 1.5;
    track.clips.push(clip);
    timeline.tracks.push(track);

    let mut manifest = MediaManifest::new();
    manifest.entries.push(external_entry(
        "asset-1", &src, w as i32, h as i32, fps as f64,
    ));

    let render_size = RenderSize::new(w, h);
    let plan = build_plan(&timeline, &manifest, render_size);
    let clip_plan = &plan.clip_plans[0];
    let mut render_loop = require_render_loop(timeline, &manifest, render_size);

    let server = start_server();
    let port = port_of(&server.endpoint_frame());
    let identity = PlaybackIdentity::new(19, 27, "session-speed-15").unwrap();
    let targets = [0, 3, 5];
    let last_frame = *targets.last().expect("terminal target");
    let sink = server.sink(
        identity.clone(),
        PublicationGate::open(),
        last_frame,
        Arc::new(|_| {}),
    );
    let mut emitted = Vec::new();
    let mut mapped_sources = Vec::new();

    for target in targets {
        let source_frame = source_frame_index(clip_plan, target);
        mapped_sources.push(source_frame);
        let expected = decode_exact_source_frame(&src, source_frame, fps, w, h);
        let frame = render_loop
            .render_frame(target)
            .expect("render target frame");
        assert_eq!((frame.width, frame.height), (w, h));
        assert_eq!(
            frame.rgba, expected.rgba,
            "first render for timeline frame {target} must match plan source frame {source_frame}"
        );
        let event = sink
            .publish_now(target, &frame)
            .expect("commit playback frame");
        let payload = serde_json::to_value(&event).expect("serialize playback publication");
        let sequence = payload["sequence"]
            .as_u64()
            .expect("publication sequence is numeric");
        let (head, jpeg) = finite_get(port, &frame_path(&identity, target, sequence), "");
        assert_eq!(head.status, 200, "published frame {target} should resolve");
        let image = image::load_from_memory(&jpeg).expect("decode published playback JPEG");
        assert_eq!((image.width(), image.height()), (w, h));
        emitted.push(payload);
    }

    assert_eq!(mapped_sources, vec![2, 7, 10]);
    assert_eq!(
        emitted
            .iter()
            .map(|payload| payload["frame"].as_i64().expect("frame integer"))
            .collect::<Vec<_>>(),
        vec![0, 3, 5]
    );
    assert_eq!(
        emitted
            .iter()
            .map(|payload| payload["sequence"].as_u64().expect("sequence integer"))
            .collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    assert_eq!(
        emitted
            .iter()
            .map(|payload| payload["terminal"].as_bool().expect("terminal bool"))
            .collect::<Vec<_>>(),
        vec![false, false, true]
    );
}

#[test]
fn frame_route_first_publication_matches_reversed_plan() {
    let dir = tempfile::tempdir().expect("fixture tempdir");
    let src = dir.path().join("reversed-cfr.mp4");
    let (w, h, fps, frames) = (160u32, 90u32, 12u32, 12u32);
    make_distinct_cfr_video(&src, w, h, fps, frames);

    let mut timeline = Timeline::new();
    timeline.fps = fps as i32;
    let mut track = Track::new("t1", ClipType::Video);
    let mut clip = Clip::new("clip-1", "asset-1", 0, 6);
    clip.trim_start_frame = 2;
    clip.reversed = true;
    track.clips.push(clip);
    timeline.tracks.push(track);

    let mut manifest = MediaManifest::new();
    manifest.entries.push(external_entry(
        "asset-1", &src, w as i32, h as i32, fps as f64,
    ));

    let render_size = RenderSize::new(w, h);
    let plan = build_plan(&timeline, &manifest, render_size);
    let clip_plan = &plan.clip_plans[0];
    let mut render_loop = require_render_loop(timeline, &manifest, render_size);

    let server = start_server();
    let port = port_of(&server.endpoint_frame());
    let identity = PlaybackIdentity::new(19, 28, "session-reversed").unwrap();
    let targets = [0, 5];
    let last_frame = *targets.last().expect("terminal target");
    let sink = server.sink(
        identity.clone(),
        PublicationGate::open(),
        last_frame,
        Arc::new(|_| {}),
    );
    let mut emitted = Vec::new();
    let mut mapped_sources = Vec::new();

    for target in targets {
        let source_frame = source_frame_index(clip_plan, target);
        mapped_sources.push(source_frame);
        let expected = decode_exact_source_frame(&src, source_frame, fps, w, h);
        let frame = render_loop
            .render_frame(target)
            .expect("render target frame");
        assert_eq!((frame.width, frame.height), (w, h));
        assert_eq!(
            frame.rgba, expected.rgba,
            "first reversed render for timeline frame {target} must match plan source frame {source_frame}"
        );
        let event = sink
            .publish_now(target, &frame)
            .expect("commit reversed playback frame");
        let payload =
            serde_json::to_value(&event).expect("serialize reversed playback publication");
        let sequence = payload["sequence"]
            .as_u64()
            .expect("publication sequence is numeric");
        let (head, jpeg) = finite_get(port, &frame_path(&identity, target, sequence), "");
        assert_eq!(
            head.status, 200,
            "published reversed frame {target} should resolve"
        );
        let image = image::load_from_memory(&jpeg).expect("decode reversed playback JPEG");
        assert_eq!((image.width(), image.height()), (w, h));
        emitted.push(payload);
    }

    assert_eq!(mapped_sources, vec![7, 2]);
    assert_eq!(
        emitted
            .iter()
            .map(|payload| payload["frame"].as_i64().expect("frame integer"))
            .collect::<Vec<_>>(),
        vec![0, 5]
    );
    assert_eq!(
        emitted
            .iter()
            .map(|payload| payload["sequence"].as_u64().expect("sequence integer"))
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
    assert_eq!(
        emitted
            .iter()
            .map(|payload| payload["terminal"].as_bool().expect("terminal bool"))
            .collect::<Vec<_>>(),
        vec![false, true]
    );
}
