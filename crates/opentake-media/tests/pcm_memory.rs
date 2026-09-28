//! Peak heap use of PCM extraction and waveform generation.
//!
//! This binary installs a counting global allocator, so it holds a single
//! test: concurrently running tests would pollute each other's peaks.

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use opentake_media::ffmpeg_status::{ffmpeg_available, ffmpeg_path, ffprobe_available};
use opentake_media::{extract_pcm, waveform, PcmFormat, PcmSpec};

struct CountingAllocator;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(bytes: usize) {
    let live = LIVE.fetch_add(bytes, Ordering::SeqCst) + bytes;
    PEAK.fetch_max(live, Ordering::SeqCst);
}

// SAFETY: every call forwards to the system allocator unchanged; the
// wrapper only records sizes.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            grew(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            grew(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let moved = unsafe { System.realloc(ptr, layout, new_size) };
        if !moved.is_null() {
            if new_size >= layout.size() {
                grew(new_size - layout.size());
            } else {
                LIVE.fetch_sub(layout.size() - new_size, Ordering::SeqCst);
            }
        }
        moved
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Run `work` and report how far the heap grew above its starting level.
fn peak_growth<T>(work: impl FnOnce() -> T) -> (T, usize) {
    let base = LIVE.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);
    let result = work();
    (result, PEAK.load(Ordering::SeqCst).saturating_sub(base))
}

fn generate(output: &Path, source: &str, codec: &[&str]) {
    let status = Command::new(ffmpeg_path())
        .args(["-v", "error", "-y", "-f", "lavfi", "-i", source])
        .args(codec)
        .arg(output)
        .status()
        .expect("spawn ffmpeg");
    assert!(status.success(), "generate {}", output.display());
}

const MIB: usize = 1024 * 1024;

#[test]
fn pcm_extraction_and_waveforms_keep_one_copy_or_less_in_memory() {
    if !ffmpeg_available() || !ffprobe_available() {
        eprintln!("skip: ffmpeg not available");
        return;
    }
    let dir = tempfile::tempdir().unwrap();

    // A minute of 48 kHz mono f32 is 11.5 MB of samples. The decoded bytes are
    // converted as they stream in, so the peak stays near that one buffer
    // instead of a raw byte copy plus the f32 copy.
    let minute = dir.path().join("minute.wav");
    generate(
        &minute,
        "sine=frequency=440:sample_rate=48000:duration=60",
        &["-c:a", "pcm_s16le"],
    );
    let spec = PcmSpec {
        sample_rate: 48_000,
        channels: 1,
        format: PcmFormat::F32,
    };
    let (pcm, peak) = peak_growth(|| extract_pcm(&minute, &spec, Some((0.0, 60.0))).unwrap());
    let output = pcm.samples_f32.len() * 4;
    eprintln!(
        "extract_pcm 60 s: output {:.1} MiB, heap peak {:.1} MiB",
        output as f64 / MIB as f64,
        peak as f64 / MIB as f64
    );
    assert_eq!(pcm.samples_f32.len(), 60 * 48_000);
    assert!(
        peak < output + MIB,
        "peak {peak} bytes for {output} bytes of samples"
    );
    drop(pcm);

    // An hour of audio: 79 million samples at the waveform rate (317 MB as
    // f32) reduce to 20 000 buckets while they stream.
    let hour = dir.path().join("hour.mp2");
    generate(
        &hour,
        "sine=frequency=440:sample_rate=16000:duration=3600",
        &["-c:a", "mp2", "-b:a", "32k"],
    );
    let (buckets, peak) = peak_growth(|| waveform(&hour, 3600.0).unwrap());
    eprintln!(
        "waveform 1 h: heap peak {:.2} MiB",
        peak as f64 / MIB as f64
    );
    assert_eq!(buckets.len(), 20_000);
    assert!(buckets.iter().any(|value| *value < 0.5), "tone is loud");
    assert!(peak < 16 * MIB, "peak {peak} bytes");
}
