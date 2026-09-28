//! Text rasterization interface (SPEC §4.2). Upstream renders text via
//! `CATextLayer` + the CoreAnimationTool; OpenTake rasterizes each text clip to a
//! premultiplied-RGBA texture that composites like any other layer.
//!
//! Full glyph layout (cosmic-text) + raster (tiny-skia / Vello), with shadow /
//! border / background / alignment / word-wrap, lands in the advanced/motion
//! phase. This module defines the trait boundary now and ships a null
//! implementation that returns `None` (never `todo!()` / `unimplemented!()`), so
//! the compositor can already route text clips and tests never trip an
//! unimplemented panic.

use std::borrow::Cow;

use opentake_domain::{Clip, TextStyle};

use crate::source::DecodedFrame;

/// Inputs needed to rasterize one text clip at a given canvas size.
#[derive(Clone, PartialEq, Debug)]
pub struct TextRasterRequest<'a> {
    pub clip_id: &'a str,
    pub content: &'a str,
    pub style: &'a TextStyle,
    /// Normalized text box on the canvas (top-left x/y, width/height in 0..1).
    pub box_norm: (f64, f64, f64, f64),
    /// Canvas pixel size.
    pub canvas: (u32, u32),
}

/// Largest text box we will rasterize (px per side). Bounds the CPU/RAM cost of a
/// degenerate transform; real text boxes are well under this.
pub(crate) const MAX_TEXT_BOX_SIDE: u32 = 8192;

/// Box pixel size from the normalized box + canvas, clamped to sane bounds, or
/// `None` when the box is under one pixel on either axis (or not finite).
/// The box comes from `clip.transform` (top-left + width/height) — i.e. upstream
/// `layer.frame = (tl.x*W, tl.y*H, transform.width*W, transform.height*H)` at
/// `TextLayerController.applyStyle` L157-163 — **not** `TextLayout.naturalSize`
/// (that measures glyph bounds for clip placement only; the shadow padding
/// `12*2` and `+4` slack live there, not in the rasterizer's box).
pub(crate) fn text_box_pixels(
    box_norm: (f64, f64, f64, f64),
    canvas: (u32, u32),
) -> Option<(u32, u32)> {
    let (_, _, bw, bh) = box_norm;
    let w = (bw * canvas.0 as f64).round();
    let h = (bh * canvas.1 as f64).round();
    if !(w.is_finite() && h.is_finite()) || w < 1.0 || h < 1.0 {
        return None;
    }
    Some((
        (w as u32).min(MAX_TEXT_BOX_SIDE),
        (h as u32).min(MAX_TEXT_BOX_SIDE),
    ))
}

/// Whether a text request has nothing to draw: its content is empty or
/// whitespace-only, or its box is under one pixel at the request canvas.
///
/// A blank text clip is valid timeline content (clearing the Inspector's text
/// field commits `textContent: ""`), so resolvers skip it like an invisible
/// layer instead of treating it as a materialization failure. Every
/// [`TextRasterizer`] may return `None` for a blank request; `None` for a
/// request that is *not* blank means rasterization failed.
pub fn is_blank_text(request: &TextRasterRequest<'_>) -> bool {
    request.content.trim().is_empty() || text_box_pixels(request.box_norm, request.canvas).is_none()
}

/// The rasterizer input a text clip projects to: its content and style.
///
/// A missing `text_content` is blank text, not a broken clip: the Inspector
/// shows it as an empty field and the render plan already treats it as a
/// source with nothing to draw. It projects to empty content, so resolvers skip
/// it exactly like `""`. Blank text never reaches a rasterizer, so it needs no
/// style either (the default stands in). A non-blank clip without a style has
/// no raster input (`None`), which resolvers report as a materialization
/// failure.
pub fn text_clip_raster_input(clip: &Clip) -> Option<(&str, Cow<'_, TextStyle>)> {
    let content = clip.text_content.as_deref().unwrap_or_default();
    let style = match &clip.text_style {
        Some(style) => Cow::Borrowed(style),
        None if content.trim().is_empty() => Cow::Owned(TextStyle::default()),
        None => return None,
    };
    Some((content, style))
}

/// Why a non-blank text layer produced no pixels (see [`rasterize_text_layer`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TextLayerError {
    /// The rasterizer returned `None` for a request with something to draw.
    #[error("rasterization failed")]
    Failed,
    /// The rasterizer panicked.
    #[error("rasterization panicked")]
    Panicked,
}

/// The text-layer contract shared by the export, paused-preview and playback
/// resolvers: `Ok(None)` when the request is blank ([`is_blank_text`]) and the
/// layer draws nothing, `Ok(Some(frame))` for a rasterized box, and an error
/// when a non-blank request yields no frame or the rasterizer panics. The
/// rasterizer is not called for a blank request.
pub fn rasterize_text_layer(
    rasterizer: &dyn TextRasterizer,
    request: &TextRasterRequest<'_>,
) -> Result<Option<DecodedFrame>, TextLayerError> {
    if is_blank_text(request) {
        return Ok(None);
    }
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        rasterizer.rasterize(request)
    })) {
        Ok(Some(frame)) => Ok(Some(frame)),
        Ok(None) => Err(TextLayerError::Failed),
        Err(_) => Err(TextLayerError::Panicked),
    }
}

/// Rasterizes a text clip to a premultiplied-RGBA [`DecodedFrame`].
pub trait TextRasterizer {
    /// Render the request, or `None` if text rendering is unavailable in this
    /// build (the null backend) or the request is blank ([`is_blank_text`]).
    /// Resolvers go through [`rasterize_text_layer`], which never asks for a
    /// blank request and treats any other `None` as a failure.
    fn rasterize(&self, request: &TextRasterRequest<'_>) -> Option<DecodedFrame>;
}

/// Placeholder backend: produces no texture. Lets the pipeline compile, route
/// text clips, and run end-to-end without a glyph engine. Replaced by the
/// cosmic-text backend in a later phase.
#[derive(Clone, Copy, Debug, Default)]
pub struct NullTextRasterizer;

impl TextRasterizer for NullTextRasterizer {
    fn rasterize(&self, _request: &TextRasterRequest<'_>) -> Option<DecodedFrame> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request<'a>(
        content: &'a str,
        style: &'a TextStyle,
        box_norm: (f64, f64, f64, f64),
    ) -> TextRasterRequest<'a> {
        TextRasterRequest {
            clip_id: "t0",
            content,
            style,
            box_norm,
            canvas: (1920, 1080),
        }
    }

    /// Records whether it was asked to rasterize and answers with `frame`.
    struct Recording {
        called: std::cell::Cell<bool>,
        frame: Option<DecodedFrame>,
    }

    impl TextRasterizer for Recording {
        fn rasterize(&self, _request: &TextRasterRequest<'_>) -> Option<DecodedFrame> {
            self.called.set(true);
            self.frame.clone()
        }
    }

    struct Panicking;

    impl TextRasterizer for Panicking {
        fn rasterize(&self, _request: &TextRasterRequest<'_>) -> Option<DecodedFrame> {
            panic!("rasterizer bug");
        }
    }

    #[test]
    fn blank_text_covers_empty_whitespace_and_sub_pixel_boxes() {
        let style = TextStyle::default();
        let full = (0.0, 0.0, 1.0, 1.0);
        assert!(is_blank_text(&request("", &style, full)));
        assert!(is_blank_text(&request(" \t\n\u{3000}", &style, full)));
        // 0.0002 * 1920 = 0.38 px rounds to no pixel at this canvas.
        assert!(is_blank_text(&request(
            "hi",
            &style,
            (0.0, 0.0, 0.0002, 0.5)
        )));
        assert!(is_blank_text(&request(
            "hi",
            &style,
            (0.0, 0.0, 0.5, f64::NAN)
        )));
        assert!(!is_blank_text(&request("hi", &style, full)));
        // 0.0006 * 1920 = 1.15 px rounds to one pixel and still draws.
        assert!(!is_blank_text(&request(
            "hi",
            &style,
            (0.0, 0.0, 0.0006, 0.5)
        )));
    }

    #[test]
    fn blank_layer_draws_nothing_without_calling_the_rasterizer() {
        let style = TextStyle::default();
        let rasterizer = Recording {
            called: std::cell::Cell::new(false),
            frame: None,
        };
        for content in ["", "   "] {
            let blank = request(content, &style, (0.0, 0.0, 1.0, 1.0));
            assert_eq!(rasterize_text_layer(&rasterizer, &blank), Ok(None));
        }
        let sliver = request("hi", &style, (0.0, 0.0, 0.0, 0.5));
        assert_eq!(rasterize_text_layer(&rasterizer, &sliver), Ok(None));
        assert!(!rasterizer.called.get());
    }

    #[test]
    fn non_blank_layer_without_a_frame_or_with_a_panic_fails() {
        let style = TextStyle::default();
        let visible = request("hello", &style, (0.1, 0.1, 0.8, 0.2));
        assert_eq!(
            rasterize_text_layer(&NullTextRasterizer, &visible),
            Err(TextLayerError::Failed)
        );
        assert_eq!(
            rasterize_text_layer(&Panicking, &visible),
            Err(TextLayerError::Panicked)
        );
        let frame = DecodedFrame::new(1, 1, vec![255; 4], true);
        let rasterizer = Recording {
            called: std::cell::Cell::new(false),
            frame: Some(frame.clone()),
        };
        assert_eq!(rasterize_text_layer(&rasterizer, &visible), Ok(Some(frame)));
    }

    #[test]
    fn missing_content_projects_as_blank_text_and_needs_no_style() {
        let mut clip = Clip::new("t0", "", 0, 30);
        clip.text_style = Some(TextStyle::default());
        let (content, _) = text_clip_raster_input(&clip).expect("styled text projects");
        assert_eq!(content, "");

        clip.text_style = None;
        for blank in [None, Some(String::new()), Some("  ".to_string())] {
            clip.text_content = blank;
            let (content, style) = text_clip_raster_input(&clip).expect("blank text projects");
            assert!(content.trim().is_empty());
            assert_eq!(*style, TextStyle::default());
        }

        clip.text_content = Some("visible".to_string());
        assert!(
            text_clip_raster_input(&clip).is_none(),
            "non-blank text without a style has no raster input"
        );
    }

    #[test]
    fn null_rasterizer_returns_none_without_panicking() {
        let style = TextStyle::default();
        let req = TextRasterRequest {
            clip_id: "t0",
            content: "hello",
            style: &style,
            box_norm: (0.1, 0.1, 0.8, 0.2),
            canvas: (1920, 1080),
        };
        assert!(NullTextRasterizer.rasterize(&req).is_none());
    }
}
