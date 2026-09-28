//! SMPTE timecode strings for the timeline-interchange exporters (XMEML, EDL).
//!
//! A frame index becomes `HH:MM:SS:FF` at an integer timebase. NTSC sources
//! (29.97 / 59.94 fps, timebase 30 / 60) usually count in **drop-frame**
//! timecode, which keeps the label close to wall-clock time by skipping the
//! first `drop` frame *numbers* — 2 at timebase 30, 4 at 60 — of every minute
//! except each tenth. Only labels are skipped, never frames, so a ten-minute
//! block holds `per_minute * 10 + drop` frames, where
//! `per_minute = fps * 60 - drop`: 17 982 frames at 29.97 and 35 964 at 59.94.
//!
//! [`format_timecode`] is upstream palmier-pro `XMLExporter.formatTimecode` as
//! fixed in palmier-pro #361 (4b87afe). In SMPTE notation
//! ([`DropFrameSeparator::FramesOnly`]) it is the exact inverse of
//! `opentake_media::parse_smpte_timecode`, which reads source timecode tags.

/// How a drop-frame timecode string separates its fields. Non-drop timecode is
/// always written `HH:MM:SS:FF`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DropFrameSeparator {
    /// `HH:MM:SS;FF` — SMPTE notation (CMX3600 EDL, ffprobe tags): only the
    /// separator before the frames field marks drop-frame.
    FramesOnly,
    /// `HH;MM;SS;FF` — Final Cut Pro 7 XML (XMEML): every separator is `;`.
    All,
}

/// Frame index → SMPTE timecode at the integer timebase `fps`.
///
/// `drop_frame` selects drop-frame numbering. It only exists for timebases that
/// are a multiple of 30 (29.97 → 30, 59.94 → 60), so for any other timebase
/// the request is ignored and the timecode is written non-drop, the same way
/// `opentake_media::parse_smpte_timecode` reads a `;` on such a rate. Negative
/// frames clamp to 0, hours do not wrap at 24, and `fps <= 0` yields
/// `00:00:00:00` (upstream's guard).
pub fn format_timecode(
    frame: i32,
    fps: i32,
    drop_frame: bool,
    separator: DropFrameSeparator,
) -> String {
    if fps <= 0 {
        return "00:00:00:00".to_string();
    }
    let drop_frame = drop_frame && fps % 30 == 0;
    let (hh, mm, ss, ff) = timecode_fields(frame, fps, drop_frame);
    match (drop_frame, separator) {
        (false, _) => format!("{hh:02}:{mm:02}:{ss:02}:{ff:02}"),
        (true, DropFrameSeparator::FramesOnly) => format!("{hh:02}:{mm:02}:{ss:02};{ff:02}"),
        (true, DropFrameSeparator::All) => format!("{hh:02};{mm:02};{ss:02};{ff:02}"),
    }
}

/// `(hours, minutes, seconds, frames)` label of `frame` at `fps > 0`.
///
/// Drop-frame advances the frame count by the labels skipped before it — `drop`
/// for each of the nine shortened minutes of every whole ten-minute block, plus
/// `drop` for each whole shortened minute of the current block — and then
/// splits it like a non-drop count. Computed in `i64` so no frame overflows.
fn timecode_fields(frame: i32, fps: i32, drop_frame: bool) -> (i64, i64, i64, i64) {
    let fps = i64::from(fps);
    let mut f = i64::from(frame.max(0));
    if drop_frame {
        // Upstream `round(fps * 0.066666)`: 2 at 30, 4 at 60.
        let drop = (fps as f64 * 0.066_666).round() as i64;
        let per_minute = fps * 60 - drop;
        let per_ten_minutes = per_minute * 10 + drop;
        let (blocks, into_block) = (f / per_ten_minutes, f % per_ten_minutes);
        f += drop * 9 * blocks;
        if into_block > drop {
            f += drop * ((into_block - drop) / per_minute);
        }
    }
    (
        f / (fps * 3600),
        (f / (fps * 60)) % 60,
        (f / fps) % 60,
        f % fps,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use DropFrameSeparator::{All, FramesOnly};

    #[test]
    fn drop_frame_labels_match_smpte_reference_values() {
        // 29.97 (timebase 30): two labels dropped per minute except each tenth.
        assert_eq!(format_timecode(1799, 30, true, All), "00;00;59;29");
        assert_eq!(format_timecode(1800, 30, true, All), "00;01;00;02");
        assert_eq!(format_timecode(17_981, 30, true, All), "00;09;59;29");
        assert_eq!(format_timecode(17_982, 30, true, All), "00;10;00;00");
        // The tenth minute keeps all its labels; the eleventh drops two again.
        assert_eq!(format_timecode(19_781, 30, true, All), "00;10;59;29");
        assert_eq!(format_timecode(19_782, 30, true, All), "00;11;00;02");
        assert_eq!(format_timecode(42_966, 30, true, All), "00;23;53;18");
        assert_eq!(format_timecode(1_552_158, 30, true, All), "14;23;10;12");
        assert_eq!(
            format_timecode(24 * 6 * 17_982, 30, true, All),
            "24;00;00;00"
        );
        // 59.94 (timebase 60): four labels dropped per minute except each tenth.
        assert_eq!(format_timecode(3599, 60, true, All), "00;00;59;59");
        assert_eq!(format_timecode(3600, 60, true, All), "00;01;00;04");
        assert_eq!(format_timecode(35_963, 60, true, All), "00;09;59;59");
        assert_eq!(format_timecode(35_964, 60, true, All), "00;10;00;00");
    }

    #[test]
    fn drop_frame_separator_styles() {
        assert_eq!(format_timecode(1800, 30, true, All), "00;01;00;02");
        assert_eq!(format_timecode(1800, 30, true, FramesOnly), "00:01:00;02");
        // Non-drop timecode is `:`-separated in either style.
        assert_eq!(format_timecode(1800, 30, false, All), "00:01:00:00");
        assert_eq!(format_timecode(1800, 30, false, FramesOnly), "00:01:00:00");
    }

    #[test]
    fn non_drop_labels_count_plain_frames() {
        assert_eq!(format_timecode(0, 25, false, All), "00:00:00:00");
        assert_eq!(format_timecode(1_688_098, 25, false, All), "18:45:23:23");
        assert_eq!(format_timecode(24 * 3600, 24, false, All), "01:00:00:00");
        let f = 30 * 3600 + 30 * 60 + 30 + 1;
        assert_eq!(format_timecode(f, 30, false, All), "01:01:01:01");
    }

    #[test]
    fn drop_frame_needs_a_drop_frame_timebase() {
        // There is no drop-frame count at 24 / 25 fps: written non-drop.
        assert_eq!(format_timecode(1500, 25, true, All), "00:01:00:00");
        assert_eq!(format_timecode(1440, 24, true, FramesOnly), "00:01:00:00");
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        assert_eq!(format_timecode(100, 0, false, All), "00:00:00:00");
        assert_eq!(format_timecode(100, -30, true, All), "00:00:00:00");
        assert_eq!(format_timecode(-5, 30, true, All), "00;00;00;00");
        assert_eq!(format_timecode(i32::MAX, 30, true, All), "19904;00;42;19");
    }

    /// Next drop-frame label after `label`, by counting: every label exists
    /// except the first `drop` frame numbers of each minute not divisible by 10.
    fn next_label(label: (i64, i64, i64, i64), fps: i64, drop: i64) -> (i64, i64, i64, i64) {
        let (mut hh, mut mm, mut ss, mut ff) = label;
        ff += 1;
        if ff == fps {
            ff = 0;
            ss += 1;
        }
        if ss == 60 {
            ss = 0;
            mm += 1;
        }
        if mm == 60 {
            mm = 0;
            hh += 1;
        }
        if ss == 0 && ff == 0 && mm % 10 != 0 {
            ff = drop;
        }
        (hh, mm, ss, ff)
    }

    #[test]
    fn every_frame_of_24_hours_gets_the_next_valid_drop_frame_label() {
        for (fps, drop) in [(30, 2), (60, 4)] {
            let frames_per_day = 24 * 6 * (i64::from(fps) * 600 - 9 * drop);
            let mut label = (0, 0, 0, 0);
            for frame in 0..frames_per_day {
                let frame = i32::try_from(frame).unwrap();
                assert_eq!(
                    timecode_fields(frame, fps, true),
                    label,
                    "frame {frame} @ {fps}"
                );
                label = next_label(label, i64::from(fps), drop);
            }
            assert_eq!(label, (24, 0, 0, 0));
        }
    }
}
