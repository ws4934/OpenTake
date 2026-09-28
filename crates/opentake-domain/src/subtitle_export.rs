//! Subtitle export — pure logic for serializing caption clips to SubRip (`.srt`)
//! and WebVTT (`.vtt`) strings. Part of the ADVANCED-FEATURES.md D-tier slice
//! (see #29). Zero IO: every function returns a `String`; nothing touches the
//! filesystem.
//!
//! Conventions carried over from the rest of this crate:
//! - Frames are `i32`. A clip spans the half-open range `[start_frame, end_frame)`.
//! - Frame -> milliseconds uses `(frame * 1000) / fps` with `fps` floored at 1
//!   so a malformed `fps == 0` cannot divide by zero.
//! - SRT timestamps use `HH:MM:SS,mmm` (comma); VTT uses `HH:MM:SS.mmm` (dot).
//!
//! A "caption" clip is any clip carrying a `caption_group_id` *and* `text_content`.
//! Cues are flattened (no shared/deviant style folding — that lives in the
//! encode-timeline compaction layer), sorted by `start_frame` (ties broken by
//! `id` for stable, deterministic output), numbered from 1, and emitted in order.
//!
//! Caption text is free-form, but both formats give some text a structural
//! meaning: a blank line ends a cue and a line containing `-->` reads as a
//! timing line; WebVTT additionally parses `<` as a tag and `&` as a character
//! reference. Text is therefore sanitized so every caption stays one complete
//! cue — see [`SubtitleCue::text`] for the format-independent part, and
//! [`export_srt`] / [`export_vtt`] for the per-format escaping.

use crate::timeline::Timeline;

/// One exported subtitle entry, resolved to absolute timeline frames.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SubtitleCue {
    /// 1-based sequence number in output order.
    pub index: usize,
    /// Inclusive start frame on the timeline.
    pub start_frame: i32,
    /// Exclusive end frame on the timeline (`clip.end_frame()`).
    pub end_frame: i32,
    /// Caption text normalized to fit in a single cue: `\r\n` and lone `\r`
    /// become `\n`, trailing whitespace is removed from every line, blank and
    /// whitespace-only lines are dropped (a blank line would end the cue), and
    /// the result is trimmed. Single newlines between lines are preserved; it
    /// is never empty. Format-specific escaping happens at serialization.
    pub text: String,
}

/// Collect every caption clip across all tracks into a flat, ordered cue list.
///
/// A clip qualifies when it has a `caption_group_id` and `text_content` that is
/// still non-empty after normalization (see [`SubtitleCue::text`]). Clips are
/// sorted by `start_frame` (then `id` for a stable tie-break) and numbered from
/// 1, so skipping an empty caption keeps the numbering contiguous. [`export_srt`]
/// and [`export_vtt`] serialize exactly this list, so its length is the number
/// of cues in either file.
pub fn collect_caption_cues(timeline: &Timeline) -> Vec<SubtitleCue> {
    let mut captions: Vec<(i32, &str, i32, String)> = Vec::new();
    for track in &timeline.tracks {
        for clip in &track.clips {
            if clip.caption_group_id.is_none() {
                continue;
            }
            let Some(text) = clip.text_content.as_ref() else {
                continue;
            };
            let text = normalize_cue_text(text);
            if text.is_empty() {
                continue;
            }
            captions.push((clip.start_frame, &clip.id, clip.end_frame(), text));
        }
    }
    // Sort by start frame; break ties on id so output is deterministic.
    captions.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)));

    captions
        .into_iter()
        .enumerate()
        .map(|(i, (start, _id, end, text))| SubtitleCue {
            index: i + 1,
            start_frame: start,
            end_frame: end,
            text,
        })
        .collect()
}

/// The format-independent cleanup documented on [`SubtitleCue::text`]. Returns
/// an empty string when nothing but whitespace remains.
fn normalize_cue_text(text: &str) -> String {
    let unified = text.replace("\r\n", "\n").replace('\r', "\n");
    let lines: Vec<&str> = unified
        .split('\n')
        .map(str::trim_end)
        .filter(|line| !line.is_empty())
        .collect();
    lines.join("\n").trim().to_string()
}

/// SubRip cue text. SRT has no escape syntax, so every `-->` is shortened to
/// `->` (repeatedly, since `--->` would otherwise re-form one) and no text line
/// can read as a timing line. `<` and `&` pass through unchanged: SRT players
/// treat simple tags such as `<i>` as styling, and SRT cannot escape them.
fn srt_cue_text(text: &str) -> String {
    let mut text = text.to_string();
    while text.contains("-->") {
        text = text.replace("-->", "->");
    }
    text
}

/// WebVTT cue text: `&`, `<` and `>` become character references (`&` first so
/// the others are not double-escaped). This keeps `<` from opening a tag and
/// `&` from starting a reference, and since `-->` cannot survive `>` escaping,
/// the text keeps its original characters when a player renders it.
fn vtt_cue_text(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Frame -> milliseconds. `fps` is floored at 1 to avoid division by zero.
fn frame_to_ms(frame: i32, fps: i32) -> i64 {
    (frame as i64 * 1000) / fps.max(1) as i64
}

/// Split a non-negative millisecond count into `(hours, minutes, seconds, millis)`.
fn split_ms(ms: i64) -> (i64, i64, i64, i64) {
    let ms = ms.max(0);
    let hours = ms / 3_600_000;
    let minutes = (ms % 3_600_000) / 60_000;
    let seconds = (ms % 60_000) / 1_000;
    let millis = ms % 1_000;
    (hours, minutes, seconds, millis)
}

/// SubRip timestamp: `HH:MM:SS,mmm` (comma before milliseconds).
fn format_timestamp_srt(ms: i64) -> String {
    let (h, m, s, milli) = split_ms(ms);
    format!("{h:02}:{m:02}:{s:02},{milli:03}")
}

/// WebVTT timestamp: `HH:MM:SS.mmm` (dot before milliseconds).
fn format_timestamp_vtt(ms: i64) -> String {
    let (h, m, s, milli) = split_ms(ms);
    format!("{h:02}:{m:02}:{s:02}.{milli:03}")
}

/// Serialize all caption cues to a SubRip (`.srt`) document.
///
/// Each cue is `{index}\n{start} --> {end}\n{text}\n\n`, with any `-->` in the
/// text shortened to `->`. An empty timeline (no caption clips) yields an empty
/// string.
pub fn export_srt(timeline: &Timeline) -> String {
    let fps = timeline.fps;
    let mut out = String::new();
    for cue in collect_caption_cues(timeline) {
        let start = format_timestamp_srt(frame_to_ms(cue.start_frame, fps));
        let end = format_timestamp_srt(frame_to_ms(cue.end_frame, fps));
        out.push_str(&format!(
            "{}\n{} --> {}\n{}\n\n",
            cue.index,
            start,
            end,
            srt_cue_text(&cue.text)
        ));
    }
    out
}

/// Serialize all caption cues to a WebVTT (`.vtt`) document.
///
/// The document always opens with `WEBVTT\n\n`. Each cue follows as
/// `{start} --> {end}\n{text}\n\n` (the optional numeric cue id is omitted),
/// with `&`, `<` and `>` in the text escaped as character references. An empty
/// timeline yields just the `WEBVTT\n\n` header.
pub fn export_vtt(timeline: &Timeline) -> String {
    let fps = timeline.fps;
    let mut out = String::from("WEBVTT\n\n");
    for cue in collect_caption_cues(timeline) {
        let start = format_timestamp_vtt(frame_to_ms(cue.start_frame, fps));
        let end = format_timestamp_vtt(frame_to_ms(cue.end_frame, fps));
        out.push_str(&format!(
            "{} --> {}\n{}\n\n",
            start,
            end,
            vtt_cue_text(&cue.text)
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::Clip;
    use crate::clip_type::ClipType;
    use crate::timeline::{Timeline, Track};

    /// Build a caption clip: text + caption_group_id set, media_type Text.
    fn caption(id: &str, group: &str, start: i32, dur: i32, text: &str) -> Clip {
        let mut c = Clip::new(id, "caption", start, dur);
        c.media_type = ClipType::Text;
        c.caption_group_id = Some(group.to_string());
        c.text_content = Some(text.to_string());
        c
    }

    fn timeline_with(fps: i32, clips: Vec<Clip>) -> Timeline {
        let mut tl = Timeline::new();
        tl.fps = fps;
        let mut t = Track::new("t-cap", ClipType::Text);
        t.clips = clips;
        tl.tracks.push(t);
        tl
    }

    // --- frame -> timestamp ---

    #[test]
    fn frame_to_ms_basic_and_fps_floor() {
        assert_eq!(frame_to_ms(30, 30), 1000);
        assert_eq!(frame_to_ms(45, 30), 1500);
        // fps == 0 must not panic; floored to 1.
        assert_eq!(frame_to_ms(5, 0), 5000);
    }

    #[test]
    fn srt_timestamp_uses_comma() {
        assert_eq!(format_timestamp_srt(1000), "00:00:01,000");
        assert_eq!(format_timestamp_srt(3_661_500), "01:01:01,500");
    }

    #[test]
    fn vtt_timestamp_uses_dot() {
        assert_eq!(format_timestamp_vtt(1000), "00:00:01.000");
        assert_eq!(format_timestamp_vtt(3_661_500), "01:01:01.500");
    }

    // --- empty timeline ---

    #[test]
    fn empty_timeline_srt_is_empty_string() {
        let tl = Timeline::new();
        assert_eq!(export_srt(&tl), "");
    }

    #[test]
    fn empty_timeline_vtt_is_header_only() {
        let tl = Timeline::new();
        assert_eq!(export_vtt(&tl), "WEBVTT\n\n");
    }

    // --- single caption ---

    #[test]
    fn single_caption_srt_block() {
        // fps=30, start=30 (1.0s), dur=30 -> end frame 60 (2.0s).
        let tl = timeline_with(30, vec![caption("c1", "g1", 30, 30, "Hello")]);
        assert_eq!(
            export_srt(&tl),
            "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n"
        );
    }

    #[test]
    fn single_caption_vtt_block() {
        let tl = timeline_with(30, vec![caption("c1", "g1", 30, 30, "Hello")]);
        assert_eq!(
            export_vtt(&tl),
            "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHello\n\n"
        );
    }

    // --- ordering / indexing ---

    #[test]
    fn out_of_order_clips_sorted_by_start_with_sequential_index() {
        let tl = timeline_with(
            30,
            vec![
                caption("c3", "g1", 90, 30, "Third"),
                caption("c1", "g1", 0, 30, "First"),
                caption("c2", "g1", 30, 30, "Second"),
            ],
        );
        let cues = collect_caption_cues(&tl);
        assert_eq!(cues.len(), 3);
        assert_eq!(cues[0].index, 1);
        assert_eq!(cues[0].text, "First");
        assert_eq!(cues[1].index, 2);
        assert_eq!(cues[1].text, "Second");
        assert_eq!(cues[2].index, 3);
        assert_eq!(cues[2].text, "Third");
        // start frames strictly increasing.
        assert!(cues[0].start_frame < cues[1].start_frame);
        assert!(cues[1].start_frame < cues[2].start_frame);
    }

    #[test]
    fn equal_start_frames_break_tie_on_id() {
        let tl = timeline_with(
            30,
            vec![
                caption("b", "g1", 30, 30, "B"),
                caption("a", "g1", 30, 30, "A"),
            ],
        );
        let cues = collect_caption_cues(&tl);
        assert_eq!(cues[0].text, "A");
        assert_eq!(cues[1].text, "B");
    }

    // --- multi-line text ---

    #[test]
    fn multiline_text_preserves_newlines_in_both_formats() {
        let tl = timeline_with(30, vec![caption("c1", "g1", 30, 30, "Line one\nLine two")]);
        assert!(export_srt(&tl).contains("Line one\nLine two"));
        assert!(export_vtt(&tl).contains("Line one\nLine two"));
    }

    // --- filtering ---

    #[test]
    fn non_caption_clip_is_excluded() {
        // A plain text clip without caption_group_id must not appear.
        let mut plain = Clip::new("p1", "asset", 0, 30);
        plain.media_type = ClipType::Text;
        plain.text_content = Some("not a caption".to_string());
        let tl = timeline_with(30, vec![plain, caption("c1", "g1", 30, 30, "Caption")]);
        let cues = collect_caption_cues(&tl);
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "Caption");
    }

    #[test]
    fn caption_without_text_content_is_excluded() {
        let mut c = Clip::new("c1", "caption", 0, 30);
        c.media_type = ClipType::Text;
        c.caption_group_id = Some("g1".to_string());
        // text_content stays None.
        let tl = timeline_with(30, vec![c]);
        assert!(collect_caption_cues(&tl).is_empty());
    }

    #[test]
    fn empty_or_whitespace_text_is_skipped() {
        let tl = timeline_with(
            30,
            vec![
                caption("c1", "g1", 0, 30, "   "),
                caption("c2", "g1", 30, 30, "Real"),
            ],
        );
        let cues = collect_caption_cues(&tl);
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "Real");
        assert_eq!(cues[0].index, 1);
    }

    // --- separator distinction ---

    #[test]
    fn srt_uses_comma_vtt_uses_dot_for_same_cue() {
        let tl = timeline_with(30, vec![caption("c1", "g1", 30, 30, "X")]);
        let srt = export_srt(&tl);
        let vtt = export_vtt(&tl);
        assert!(srt.contains("00:00:01,000"));
        assert!(!srt.contains("00:00:01.000"));
        assert!(vtt.contains("00:00:01.000"));
        assert!(!vtt.contains("00:00:01,000"));
    }

    // --- cross-track collection ---

    #[test]
    fn captions_from_multiple_tracks_are_merged_and_ordered() {
        let mut tl = Timeline::new();
        tl.fps = 30;
        let mut t1 = Track::new("t1", ClipType::Text);
        t1.clips = vec![caption("c2", "g1", 60, 30, "Second")];
        let mut t2 = Track::new("t2", ClipType::Text);
        t2.clips = vec![caption("c1", "g2", 0, 30, "First")];
        tl.tracks.push(t1);
        tl.tracks.push(t2);
        let cues = collect_caption_cues(&tl);
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[0].text, "First");
        assert_eq!(cues[1].text, "Second");
    }

    // --- fps == 0 robustness through the full export path ---

    #[test]
    fn fps_zero_does_not_panic() {
        let tl = timeline_with(0, vec![caption("c1", "g1", 30, 30, "X")]);
        let srt = export_srt(&tl);
        let vtt = export_vtt(&tl);
        assert!(srt.contains("X"));
        assert!(vtt.starts_with("WEBVTT"));
    }

    // --- cue text sanitizing ---

    /// Minimal reader for the documents this module writes. Splits `doc` into
    /// blank-line separated blocks and returns each cue's text lines, asserting
    /// the shape a strict parser needs: a numeric index (SRT only), exactly one
    /// timing line, then non-blank text lines that never contain `-->` or a
    /// carriage return.
    fn parse_cues(doc: &str, srt: bool) -> Vec<Vec<String>> {
        let body = if srt {
            doc
        } else {
            doc.strip_prefix("WEBVTT\n\n").expect("WebVTT header")
        };
        body.split("\n\n")
            .filter(|block| !block.is_empty())
            .map(|block| {
                let mut lines = block.split('\n');
                if srt {
                    let index = lines.next().unwrap_or_default();
                    assert!(
                        index.parse::<usize>().is_ok(),
                        "block without a cue index {block:?} in {doc:?}"
                    );
                }
                let timing = lines.next().unwrap_or_default();
                assert!(
                    timing.contains(" --> "),
                    "block without a timing line {block:?} in {doc:?}"
                );
                let text: Vec<String> = lines.map(str::to_string).collect();
                assert!(!text.is_empty(), "cue without text in {doc:?}");
                for line in &text {
                    assert!(!line.trim().is_empty(), "blank cue line in {doc:?}");
                    assert!(!line.contains("-->"), "timing arrow in cue text {line:?}");
                    assert!(!line.contains('\r'), "carriage return in cue text {line:?}");
                }
                text
            })
            .collect()
    }

    #[test]
    fn blank_lines_inside_a_caption_do_not_split_the_cue() {
        let tl = timeline_with(
            30,
            vec![
                caption("c1", "g1", 30, 30, "Line one\n\nLine two"),
                caption("c2", "g1", 60, 30, "Next"),
            ],
        );
        let expected = vec![vec!["Line one", "Line two"], vec!["Next"]];
        assert_eq!(parse_cues(&export_srt(&tl), true), expected);
        assert_eq!(parse_cues(&export_vtt(&tl), false), expected);

        let cues = collect_caption_cues(&tl);
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[0].text, "Line one\nLine two");
    }

    #[test]
    fn timing_arrows_and_markup_cannot_break_a_cue() {
        let tl = timeline_with(30, vec![caption("c1", "g1", 30, 30, "a <b & c --> d")]);
        // SubRip has no escape syntax: the arrow is shortened, markup is kept.
        assert_eq!(
            parse_cues(&export_srt(&tl), true),
            vec![vec!["a <b & c -> d"]]
        );
        // WebVTT escapes `&`, `<` and `>`; escaping `>` also defuses the arrow.
        assert_eq!(
            parse_cues(&export_vtt(&tl), false),
            vec![vec!["a &lt;b &amp; c --&gt; d"]]
        );
    }

    #[test]
    fn longer_arrows_cannot_reassemble_a_timing_arrow_in_srt() {
        let tl = timeline_with(
            30,
            vec![
                caption("c1", "g1", 0, 30, "--->"),
                caption("c2", "g1", 30, 30, "a ----> b -->--> c"),
            ],
        );
        assert_eq!(
            parse_cues(&export_srt(&tl), true),
            vec![vec!["->"], vec!["a -> b ->-> c"]]
        );
        assert_eq!(
            parse_cues(&export_vtt(&tl), false),
            vec![vec!["---&gt;"], vec!["a ----&gt; b --&gt;--&gt; c"]]
        );
    }

    #[test]
    fn line_endings_and_whitespace_only_lines_are_normalized() {
        let tl = timeline_with(
            30,
            vec![caption(
                "c1",
                "g1",
                30,
                30,
                "  One\r\nTwo\rThree  \n \t \r\n\r\nFour\t\n",
            )],
        );
        let expected = vec![vec!["One", "Two", "Three", "Four"]];
        let srt = export_srt(&tl);
        let vtt = export_vtt(&tl);
        assert!(!srt.contains('\r') && !vtt.contains('\r'));
        assert_eq!(parse_cues(&srt, true), expected);
        assert_eq!(parse_cues(&vtt, false), expected);
        assert_eq!(collect_caption_cues(&tl)[0].text, "One\nTwo\nThree\nFour");
    }

    #[test]
    fn caption_empty_after_sanitizing_is_skipped_and_numbering_stays_contiguous() {
        let tl = timeline_with(
            30,
            vec![
                caption("c1", "g1", 0, 30, "First"),
                caption("c2", "g1", 30, 30, "\r\n \r\n\t"),
                caption("c3", "g1", 60, 30, "Third"),
            ],
        );
        let cues = collect_caption_cues(&tl);
        assert_eq!(
            cues.iter()
                .map(|c| (c.index, c.text.as_str()))
                .collect::<Vec<_>>(),
            vec![(1, "First"), (2, "Third")]
        );
        let srt = export_srt(&tl);
        assert!(srt.starts_with("1\n00:00:00,000 --> 00:00:01,000\nFirst\n\n2\n"));
        assert_eq!(parse_cues(&srt, true).len(), cues.len());
        assert_eq!(parse_cues(&export_vtt(&tl), false).len(), cues.len());
    }
}
