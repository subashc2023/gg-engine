//! Formatted text without a per-frame allocation.
//!
//! An immediate-mode UI formats numbers every frame — a frame time, an entity
//! count, a bound key — and the obvious shape for that is a `Vec<String>` built
//! with `format!`, which is one allocation per row per frame and the exact thing
//! M13's steady-state allocation criterion forbids (§6).
//!
//! So rows are written into one buffer that is cleared and refilled, and a
//! caller keeps a [`Span`] instead of a `String`. After the first few frames the
//! buffer stops growing and the whole layer allocates nothing.

use std::fmt::Write;

/// A run of text inside a [`Scratch`]. Invalidated by the next
/// [`Scratch::clear`], which is why it is a plain index pair and not a handle
/// with a lifetime: the borrow checker cannot see a frame boundary, and a type
/// that pretended otherwise would cost every call site a lifetime parameter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Span {
    at: u32,
    len: u32,
}

impl Span {
    /// A span of nothing — what a row that failed to format reads as.
    pub const EMPTY: Span = Span { at: 0, len: 0 };

    /// Characters in the span, as bytes. ASCII in practice; a caller wanting a
    /// column count should measure the string, not this.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Whether it holds nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// One frame's formatted text, reused across frames.
#[derive(Default)]
pub struct Scratch {
    text: String,
}

impl Scratch {
    /// Drop this frame's text, keeping the capacity — which is the whole point.
    pub fn clear(&mut self) {
        self.text.clear();
    }

    /// Format a row and return where it landed.
    ///
    /// Takes `format_args!` rather than being a macro so the arguments are
    /// borrowed and never collected into a temporary: `scratch.line(format_args!
    /// ("{ms:>6.2} ms"))` writes straight into the buffer.
    pub fn line(&mut self, args: std::fmt::Arguments<'_>) -> Span {
        let at = self.text.len();
        // A `String` write cannot fail; a partial one would still be described
        // correctly by the length below, so there is nothing to report.
        let _ = self.text.write_fmt(args);
        match (u32::try_from(at), u32::try_from(self.text.len() - at)) {
            (Ok(at), Ok(len)) => Span { at, len },
            // 4 GiB of overlay text in one frame is a bug upstream, not here.
            _ => Span::EMPTY,
        }
    }

    /// What `span` holds, or `""` for a span from another frame that no longer
    /// fits. Never panics: a stale span is a wrong row, not a crash.
    #[must_use]
    pub fn get(&self, span: Span) -> &str {
        let (at, end) = (span.at as usize, span.at as usize + span.len as usize);
        self.text.get(at..end).unwrap_or_default()
    }

    /// Bytes held. A budget check, not a layout measure.
    #[must_use]
    pub fn len(&self) -> usize {
        self.text.len()
    }

    /// Whether nothing has been written this frame.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn rows_come_back_in_the_order_and_shape_they_were_written() {
        let mut scratch = Scratch::default();
        let a = scratch.line(format_args!("{:>6.2} ms", 16.6667));
        let b = scratch.line(format_args!("tick {:>8}", 1234u64));
        assert_eq!(scratch.get(a), " 16.67 ms");
        assert_eq!(scratch.get(b), "tick     1234");
        assert_eq!(a.len(), 9);
        assert!(!a.is_empty() && !scratch.is_empty());
    }

    /// The reuse claim: clearing keeps the buffer, so a steady frame writes the
    /// same rows into the same bytes and never asks the allocator.
    #[test]
    fn a_settled_frame_reuses_the_buffer() {
        let mut scratch = Scratch::default();
        for tick in 0..64u64 {
            scratch.clear();
            scratch.line(format_args!("tick {tick:>8}"));
        }
        let capacity = scratch.text.capacity();
        for tick in 0..64u64 {
            scratch.clear();
            scratch.line(format_args!("tick {tick:>8}"));
        }
        assert_eq!(scratch.text.capacity(), capacity);
    }

    /// A span outlives its frame as a value and must not outlive it as a
    /// meaning. Reading one back after a shorter frame is a wrong row at worst
    /// — never a panic, because a UI that crashes on a stale row is worse than
    /// one that draws a blank.
    #[test]
    fn a_stale_span_reads_empty_rather_than_panicking() {
        let mut scratch = Scratch::default();
        scratch.line(format_args!("a long first row"));
        let stale = scratch.line(format_args!("and a second"));
        scratch.clear();
        scratch.line(format_args!("short"));
        assert_eq!(scratch.get(stale), "");
        assert_eq!(scratch.get(Span::EMPTY), "");
    }
}
