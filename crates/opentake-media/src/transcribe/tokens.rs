//! Whisper token bytes → segment text and word rows.
//!
//! Whisper's byte-level BPE may split one UTF-8 character across tokens: a CJK
//! ideograph or an emoji often arrives as two or three tokens that are each
//! invalid UTF-8 on their own. Reading every token as a string therefore
//! drops those characters from the word rows, and reading a segment whose
//! boundary falls inside a character fails outright. Here raw bytes are
//! accumulated until they form complete characters: a word spans the first to
//! the last token that contributed bytes to it, a character split across a
//! segment boundary is carried into the next segment, and only bytes that can
//! never become valid are rendered with U+FFFD.
//!
//! Pure and backend-independent, so it is tested without whisper.cpp.

use super::{is_non_speech_marker, TranscriptionWord};

/// One decoded whisper token.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RawToken {
    /// The token's raw text bytes, possibly a fragment of a character.
    pub bytes: Vec<u8>,
    /// Control and timestamp tokens (id >= end-of-text) carry no text.
    pub special: bool,
    /// Token start/end in whisper centiseconds, when the backend reports them.
    pub t0: Option<i64>,
    pub t1: Option<i64>,
}

/// One segment as whisper reports it: raw text bytes, centisecond bounds, and
/// its tokens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RawSegment {
    pub text: Vec<u8>,
    pub t0: i64,
    pub t1: i64,
    pub tokens: Vec<RawToken>,
}

/// A segment with valid text and its word rows.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AssembledSegment {
    pub text: String,
    pub t0: i64,
    pub t1: i64,
    pub words: Vec<TranscriptionWord>,
}

/// whisper segment and token times are in centiseconds (1/100 s).
pub(crate) fn cs_to_secs(cs: i64) -> f64 {
    cs as f64 / 100.0
}

/// Length of the prefix of `bytes` that does not end inside a character:
/// everything except a trailing, still-completable UTF-8 sequence. Invalid
/// bytes inside the prefix are left for a lossy conversion.
fn complete_prefix_len(bytes: &[u8]) -> usize {
    let mut offset = 0;
    loop {
        match std::str::from_utf8(&bytes[offset..]) {
            Ok(_) => return bytes.len(),
            Err(error) => match error.error_len() {
                // The rest is the start of a character whose remaining bytes
                // have not arrived yet.
                None => return offset + error.valid_up_to(),
                Some(invalid) => offset += error.valid_up_to() + invalid,
            },
        }
    }
}

/// Builds word rows from token bytes, one row per token except where a
/// character spans tokens.
#[derive(Debug, Default)]
pub(crate) struct WordAssembler {
    pending: Vec<u8>,
    start: Option<i64>,
    end: Option<i64>,
    words: Vec<TranscriptionWord>,
}

impl WordAssembler {
    pub(crate) fn push(&mut self, token: &RawToken) {
        if token.special {
            return;
        }
        if self.pending.is_empty() {
            self.start = token.t0;
        }
        self.pending.extend_from_slice(&token.bytes);
        self.end = token.t1;
        let complete = complete_prefix_len(&self.pending);
        if complete == self.pending.len() {
            let text = String::from_utf8_lossy(&self.pending).into_owned();
            self.emit(&text);
            self.pending.clear();
        }
    }

    /// Words completed so far. A character still waiting for its remaining
    /// bytes stays pending, so it can complete in the next segment.
    pub(crate) fn take_words(&mut self) -> Vec<TranscriptionWord> {
        std::mem::take(&mut self.words)
    }

    /// Every remaining word; bytes that never completed a character are
    /// rendered lossily instead of being dropped.
    pub(crate) fn finish(&mut self) -> Vec<TranscriptionWord> {
        if !self.pending.is_empty() {
            let text = String::from_utf8_lossy(&self.pending).into_owned();
            self.emit(&text);
            self.pending.clear();
        }
        self.take_words()
    }

    fn emit(&mut self, text: &str) {
        let trimmed = text.trim();
        // Skip blanks, text-rendered special tokens (whisper wraps them in
        // `[_..]` / `<|..|>`), and a token that is itself a whole non-speech
        // marker (a short marker like "[MUSIC]" can decode as one token).
        if trimmed.is_empty()
            || trimmed.starts_with("[_")
            || (trimmed.starts_with("<|") && trimmed.ends_with("|>"))
            || is_non_speech_marker(trimmed)
        {
            return;
        }
        self.words.push(TranscriptionWord {
            text: trimmed.to_string(),
            start: self.start.map(cs_to_secs),
            end: self.end.map(cs_to_secs),
        });
    }
}

/// Token bytes of one segment → word rows (see [`WordAssembler`]).
#[cfg(test)]
pub(crate) fn assemble_words(tokens: &[RawToken]) -> Vec<TranscriptionWord> {
    let mut assembler = WordAssembler::default();
    for token in tokens {
        assembler.push(token);
    }
    assembler.finish()
}

/// Turn whisper's raw segments into valid text and word rows. A character
/// split by a segment boundary moves, text and word alike, into the next
/// segment; only bytes that can never form a character become U+FFFD.
pub(crate) fn assemble_segments(segments: Vec<RawSegment>) -> Vec<AssembledSegment> {
    let mut text_carry = Vec::new();
    let mut words = WordAssembler::default();
    let mut assembled = Vec::with_capacity(segments.len());
    for segment in segments {
        text_carry.extend_from_slice(&segment.text);
        let complete = complete_prefix_len(&text_carry);
        let text = String::from_utf8_lossy(&text_carry[..complete]).into_owned();
        text_carry.drain(..complete);
        for token in &segment.tokens {
            words.push(token);
        }
        assembled.push(AssembledSegment {
            text,
            t0: segment.t0,
            t1: segment.t1,
            words: words.take_words(),
        });
    }
    if let Some(last) = assembled.last_mut() {
        last.text.push_str(&String::from_utf8_lossy(&text_carry));
        last.words.extend(words.finish());
    }
    assembled
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token(bytes: &[u8], t0: i64, t1: i64) -> RawToken {
        RawToken {
            bytes: bytes.to_vec(),
            special: false,
            t0: Some(t0),
            t1: Some(t1),
        }
    }

    fn special(text: &str) -> RawToken {
        RawToken {
            bytes: text.as_bytes().to_vec(),
            special: true,
            t0: None,
            t1: None,
        }
    }

    fn texts(words: &[TranscriptionWord]) -> Vec<&str> {
        words.iter().map(|word| word.text.as_str()).collect()
    }

    #[test]
    fn a_character_split_across_tokens_is_one_word_spanning_them() {
        // 中 = E4 B8 AD, 文 = E6 96 87.
        let words = assemble_words(&[
            token(&[0xE4, 0xB8], 100, 110),
            token(&[0xAD], 110, 125),
            token(&[0xE6, 0x96, 0x87], 125, 140),
        ]);
        assert_eq!(texts(&words), ["中", "文"]);
        assert_eq!(words[0].start, Some(1.0));
        assert_eq!(words[0].end, Some(1.25));
        assert_eq!(words[1].start, Some(1.25));
        assert_eq!(words[1].end, Some(1.4));
    }

    #[test]
    fn a_token_that_finishes_one_character_and_starts_another_merges_them() {
        let words = assemble_words(&[
            token(&[0xE4, 0xB8], 0, 10),
            token(&[0xAD, 0xE6], 10, 20),
            token(&[0x96, 0x87], 20, 30),
        ]);
        assert_eq!(texts(&words), ["中文"]);
        assert_eq!(words[0].start, Some(0.0));
        assert_eq!(words[0].end, Some(0.3));
    }

    #[test]
    fn an_incomplete_tail_is_rendered_lossily_without_failing() {
        let words = assemble_words(&[token(b" ok", 0, 5), token(&[0xE4, 0xB8], 5, 9)]);
        assert_eq!(texts(&words), ["ok", "\u{FFFD}"]);
        assert_eq!(words[1].start, Some(0.05));
        // Bytes that can never be valid do not hold up later characters.
        let words = assemble_words(&[token(&[0xFF], 0, 1), token("好".as_bytes(), 1, 2)]);
        assert_eq!(texts(&words), ["\u{FFFD}", "好"]);
    }

    #[test]
    fn special_and_marker_tokens_are_filtered_after_assembly() {
        let words = assemble_words(&[
            special("[_BEG_]"),
            token(&[0xE4, 0xB8], 0, 5),
            // A timestamp token between the fragments does not break them.
            special("[_TT_5]"),
            token(&[0xAD], 5, 10),
            token(b" [MUSIC]", 10, 20),
            token(b" ", 20, 21),
            token(b"<|endoftext|>", 21, 22),
            token(b"[_TT_50]", 22, 23),
        ]);
        assert_eq!(texts(&words), ["中"]);
    }

    #[test]
    fn words_joined_without_spaces_reproduce_the_segment_text() {
        let segment = "Hello, 世界! 今天很好 😀 ok";
        // Split the text into byte-level tokens that cut characters apart.
        let bytes = segment.as_bytes();
        let cuts = [0, 6, 7, 9, 12, 15, 17, 19, 22, 25, 28, 30, 33, bytes.len()];
        let tokens = cuts
            .windows(2)
            .enumerate()
            .map(|(index, cut)| token(&bytes[cut[0]..cut[1]], index as i64, index as i64 + 1))
            .collect::<Vec<_>>();
        assert!(
            tokens
                .iter()
                .any(|token| std::str::from_utf8(&token.bytes).is_err()),
            "the fixture must split characters"
        );
        let words = assemble_words(&tokens);
        let without_spaces = |text: &str| text.split_whitespace().collect::<String>();
        let joined = words
            .iter()
            .map(|word| without_spaces(&word.text))
            .collect::<String>();
        assert_eq!(joined, without_spaces(segment));
        assert!(words.iter().all(|word| word.start < word.end));
    }

    #[test]
    fn a_character_split_by_a_segment_boundary_moves_to_the_next_segment() {
        let first = RawSegment {
            text: [b"one ".as_slice(), &[0xE4, 0xB8]].concat(),
            t0: 0,
            t1: 100,
            tokens: vec![token(b" one", 0, 50), token(&[0xE4, 0xB8], 90, 100)],
        };
        let second = RawSegment {
            text: [[0xAD].as_slice(), " two".as_bytes()].concat(),
            t0: 100,
            t1: 200,
            tokens: vec![token(&[0xAD], 100, 110), token(b" two", 120, 200)],
        };
        let assembled = assemble_segments(vec![first, second]);
        assert_eq!(assembled[0].text, "one ");
        assert_eq!(texts(&assembled[0].words), ["one"]);
        assert_eq!(assembled[1].text, "中 two");
        assert_eq!(texts(&assembled[1].words), ["中", "two"]);
        assert_eq!(assembled[1].words[0].start, Some(0.9));

        // A character that never completes ends the last segment lossily.
        let open = RawSegment {
            text: [b"x ".as_slice(), &[0xF0, 0x9F]].concat(),
            t0: 0,
            t1: 10,
            tokens: vec![token(b"x", 0, 5), token(&[0xF0, 0x9F], 5, 10)],
        };
        let assembled = assemble_segments(vec![open]);
        assert_eq!(assembled[0].text, "x \u{FFFD}");
        assert_eq!(texts(&assembled[0].words), ["x", "\u{FFFD}"]);
    }
}
