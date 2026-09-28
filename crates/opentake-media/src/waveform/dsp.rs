//! Pure waveform DSP: the sample-count formula and the RMS downsample +
//! normalization. No IO, no decoding — fully unit-testable.

/// Buckets per second of audio (upstream `MediaVisualCache.waveformSampleCount`).
pub const BUCKETS_PER_SECOND: f64 = 150.0;
/// Lower bound on bucket count.
pub const MIN_BUCKETS: usize = 4000;
/// Hard upper bound on bucket count.
pub const MAX_BUCKETS: usize = 20_000;

/// Number of normalized waveform buckets for a clip of `duration` seconds.
///
/// Verbatim port of `waveformSampleCount` (`MediaVisualCache.swift:186-190`):
/// - non-finite or `<= 0` duration → [`MIN_BUCKETS`]
/// - `duration >= MAX_BUCKETS / BUCKETS_PER_SECOND` (≈133.3 s) → [`MAX_BUCKETS`]
/// - otherwise → `max(MIN_BUCKETS, floor(duration * BUCKETS_PER_SECOND))`
pub fn waveform_sample_count(duration: f64) -> usize {
    if !duration.is_finite() || duration <= 0.0 {
        return MIN_BUCKETS;
    }
    if duration >= MAX_BUCKETS as f64 / BUCKETS_PER_SECOND {
        return MAX_BUCKETS;
    }
    MIN_BUCKETS.max((duration * BUCKETS_PER_SECOND) as usize)
}

/// Downsample mono `samples` into `count` normalized buckets, **0 = loud,
/// 1 = silence** (upstream's inverted convention,
/// `MediaVisualCache.swift:11`).
///
/// Each bucket holds the RMS of its slice of samples. The RMS envelope is scaled
/// to the loudest bucket (full-scale normalization) and then inverted:
/// `out = 1 - rms_bucket / peak_rms`. A fully silent input yields all-ones; a
/// full-scale input yields values near zero.
///
/// `count == 0` → empty. Fewer samples than buckets still produces `count`
/// values (empty buckets are treated as silence → `1.0`).
pub fn rms_downsample_normalized(samples: &[f32], count: usize) -> Vec<f32> {
    if count == 0 {
        return Vec::new();
    }
    if samples.is_empty() {
        // No audio data decoded: report full silence.
        return vec![1.0; count];
    }

    let n = samples.len();
    let mut rms = vec![0.0f32; count];
    for (bucket, slot) in rms.iter_mut().enumerate() {
        // Half-open slice [lo, hi) for this bucket, spreading samples evenly.
        let lo = bucket * n / count;
        let hi = ((bucket + 1) * n / count).max(lo + 1).min(n);
        let slice = &samples[lo..hi];
        let mut sum_sq = 0.0f64;
        for &s in slice {
            sum_sq += (s as f64) * (s as f64);
        }
        *slot = (sum_sq / slice.len() as f64).sqrt() as f32;
    }

    // Full-scale normalization against the loudest bucket, then invert.
    let peak = rms.iter().copied().fold(0.0f32, f32::max);
    if peak <= f32::EPSILON {
        return vec![1.0; count];
    }
    for v in rms.iter_mut() {
        let amp = (*v / peak).clamp(0.0, 1.0);
        *v = 1.0 - amp;
    }
    rms
}

/// Streaming form of [`rms_downsample_normalized`]: buckets laid out for an
/// expected sample count, accumulated as samples arrive, in `O(count)` memory.
///
/// Bucket `b` spans the same half-open slice `[b * n / count, hi)` that
/// [`rms_downsample_normalized`] uses for `n` samples, with `n` fixed to
/// `expected` up front, and sums squares in the same order, so a stream of
/// exactly `expected` samples yields the same values. A shorter stream leaves
/// its trailing buckets silent (the audio ended before the requested
/// duration) instead of stretching the audio across all of them; samples past
/// `expected` are ignored.
pub struct RmsBuckets {
    count: usize,
    expected: usize,
    sum_squares: Vec<f64>,
    lengths: Vec<usize>,
    /// Index of the next sample to arrive.
    next: usize,
    /// First bucket whose span has not ended before `next`.
    first_open: usize,
}

impl RmsBuckets {
    pub fn new(count: usize, expected: usize) -> Self {
        RmsBuckets {
            count,
            expected,
            sum_squares: vec![0.0; count],
            lengths: vec![0; count],
            next: 0,
            first_open: 0,
        }
    }

    fn lo(&self, bucket: usize) -> usize {
        bucket * self.expected / self.count
    }

    fn hi(&self, bucket: usize) -> usize {
        ((bucket + 1) * self.expected / self.count)
            .max(self.lo(bucket) + 1)
            .min(self.expected)
    }

    pub fn push(&mut self, sample: f32) {
        let index = self.next;
        self.next += 1;
        if index >= self.expected {
            return;
        }
        while self.first_open < self.count && self.hi(self.first_open) <= index {
            self.first_open += 1;
        }
        // Several buckets share a sample only when there are fewer samples
        // than buckets.
        let square = (sample as f64) * (sample as f64);
        let mut bucket = self.first_open;
        while bucket < self.count && self.lo(bucket) <= index {
            if index < self.hi(bucket) {
                self.sum_squares[bucket] += square;
                self.lengths[bucket] += 1;
            }
            bucket += 1;
        }
    }

    /// Normalized buckets, `0 = loud, 1 = silence`, as
    /// [`rms_downsample_normalized`] reports them.
    pub fn finish(self) -> Vec<f32> {
        let mut rms = self
            .sum_squares
            .iter()
            .zip(&self.lengths)
            .map(|(&sum, &length)| {
                if length == 0 {
                    0.0
                } else {
                    (sum / length as f64).sqrt() as f32
                }
            })
            .collect::<Vec<f32>>();
        let peak = rms.iter().copied().fold(0.0f32, f32::max);
        if peak <= f32::EPSILON {
            return vec![1.0; self.count];
        }
        for v in rms.iter_mut() {
            let amp = (*v / peak).clamp(0.0, 1.0);
            *v = 1.0 - amp;
        }
        rms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- waveform_sample_count: boundary table ---

    #[test]
    fn count_zero_or_negative_or_nan_is_min() {
        assert_eq!(waveform_sample_count(0.0), MIN_BUCKETS);
        assert_eq!(waveform_sample_count(-5.0), MIN_BUCKETS);
        assert_eq!(waveform_sample_count(f64::NAN), MIN_BUCKETS);
        assert_eq!(waveform_sample_count(f64::INFINITY), MIN_BUCKETS); // not finite
    }

    #[test]
    fn count_one_second_is_min_floor() {
        // 1 * 150 = 150 < 4000 → clamps up to MIN_BUCKETS.
        assert_eq!(waveform_sample_count(1.0), MIN_BUCKETS);
    }

    #[test]
    fn count_mid_range_is_duration_times_150() {
        // 100 s → 15000 buckets (between min and max).
        assert_eq!(waveform_sample_count(100.0), 15_000);
        // 30 s → 4500.
        assert_eq!(waveform_sample_count(30.0), 4_500);
    }

    #[test]
    fn count_at_and_above_cap_is_max() {
        let cap = MAX_BUCKETS as f64 / BUCKETS_PER_SECOND; // ≈133.333
        assert_eq!(waveform_sample_count(cap), MAX_BUCKETS); // boundary is inclusive (>=)
        assert_eq!(waveform_sample_count(cap + 0.001), MAX_BUCKETS);
        assert_eq!(waveform_sample_count(1000.0), MAX_BUCKETS);
    }

    #[test]
    fn count_just_below_cap_is_not_max() {
        let just_below = MAX_BUCKETS as f64 / BUCKETS_PER_SECOND - 1.0; // ~132.3s
        let c = waveform_sample_count(just_below);
        assert!(c < MAX_BUCKETS);
        assert_eq!(c, (just_below * BUCKETS_PER_SECOND) as usize);
    }

    // --- rms_downsample_normalized ---

    #[test]
    fn downsample_zero_count_is_empty() {
        assert!(rms_downsample_normalized(&[0.1, 0.2], 0).is_empty());
    }

    #[test]
    fn downsample_empty_input_is_full_silence() {
        let out = rms_downsample_normalized(&[], 5);
        assert_eq!(out, vec![1.0; 5]);
    }

    #[test]
    fn downsample_full_silence_is_all_ones() {
        let silent = vec![0.0f32; 1000];
        let out = rms_downsample_normalized(&silent, 10);
        assert_eq!(out.len(), 10);
        for v in out {
            assert!((v - 1.0).abs() < 1e-6, "silence must map to ~1.0, got {v}");
        }
    }

    #[test]
    fn downsample_full_scale_sine_is_near_zero() {
        // A full-amplitude tone: loudest bucket → ~0 after inversion.
        let mut s = Vec::with_capacity(2000);
        for i in 0..2000 {
            s.push((i as f32 * 0.3).sin());
        }
        let out = rms_downsample_normalized(&s, 20);
        let min = out.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(min < 0.2, "loudest bucket should be near 0, got {min}");
    }

    #[test]
    fn downsample_monotonic_loudness_inversion() {
        // First half quiet, second half loud → first buckets ~1, last buckets ~0.
        let mut s = vec![0.01f32; 1000];
        s.extend(std::iter::repeat_n(1.0f32, 1000));
        let out = rms_downsample_normalized(&s, 4);
        assert_eq!(out.len(), 4);
        // quiet region (high value) > loud region (low value)
        assert!(out[0] > out[3], "quiet→loud must be decreasing: {out:?}");
        assert!(out[3] < 0.1);
    }

    #[test]
    fn downsample_produces_exact_count_even_when_undersampled() {
        // 3 samples, 10 buckets — must still return 10 values.
        let out = rms_downsample_normalized(&[1.0, 0.0, 1.0], 10);
        assert_eq!(out.len(), 10);
    }

    #[test]
    fn downsample_values_in_unit_range() {
        let mut s = Vec::new();
        for i in 0..500 {
            s.push(((i as f32) / 500.0) * 2.0 - 1.0); // ramp -1..1
        }
        let out = rms_downsample_normalized(&s, 16);
        for v in out {
            assert!((0.0..=1.0).contains(&v), "out of [0,1]: {v}");
        }
    }

    // --- RmsBuckets: streaming equivalence ---

    /// Deterministic pseudo-random signal with loud and quiet stretches.
    fn signal(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|index| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let noise = (state % 2001) as f32 / 1000.0 - 1.0;
                let envelope = if (index / 997) % 3 == 0 { 0.05 } else { 0.9 };
                noise * envelope
            })
            .collect()
    }

    fn streamed(samples: &[f32], count: usize, expected: usize) -> Vec<f32> {
        let mut buckets = RmsBuckets::new(count, expected);
        for &sample in samples {
            buckets.push(sample);
        }
        buckets.finish()
    }

    fn max_error(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn streaming_rms_matches_the_whole_buffer_downsample() {
        for (n, count) in [
            (10_000, 4_000),
            (88_200, 4_000),
            (1_000_003, 20_000),
            (3, 10),
            (4_000, 4_000),
            (1, 4_000),
        ] {
            let samples = signal(n, n as u64 + 1);
            let error = max_error(
                &streamed(&samples, count, n),
                &rms_downsample_normalized(&samples, count),
            );
            assert!(error < 1e-4, "n={n} count={count} error={error}");
        }
        assert_eq!(streamed(&[], 4_000, 0), vec![1.0; 4_000]);
        assert!(streamed(&[0.5], 0, 1).is_empty());
    }

    #[test]
    fn streaming_rms_handles_a_slightly_short_or_long_stream() {
        let expected = 22_050 * 3;
        let samples = signal(expected, 7);
        let exact = streamed(&samples, 4_000, expected);

        // A few samples short (decoder padding): only the tail bucket changes,
        // by little.
        let short = streamed(&samples[..expected - 5], 4_000, expected);
        assert_eq!(short.len(), 4_000);
        assert!(max_error(&short[..3_999], &exact[..3_999]) < 1e-4);
        assert!(short.iter().all(|v| (0.0..=1.0).contains(v)));

        // Samples past the expected count are ignored.
        let mut long = samples.clone();
        long.extend(std::iter::repeat_n(1.0, 50));
        assert_eq!(streamed(&long, 4_000, expected), exact);

        // Audio that ends early leaves its missing tail silent.
        let half = streamed(&samples[..expected / 2], 4_000, expected);
        assert!(half[3_000..].iter().all(|v| *v == 1.0));
        assert!(half[..1_990].iter().any(|v| *v < 0.5));
    }
}
