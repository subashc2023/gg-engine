//! Just enough JSON to *read* (§6 M16), for `claude --output-format stream-json`.
//!
//! # Why not `serde_json`
//!
//! This crate's manifest claims no dependencies, and the claim is about the
//! *reader* of a published record being able to link this without linking a
//! stack. A parser beside the writer this module already sits next to keeps
//! that true, and costs one testable file — the round-trip test at the bottom
//! reads [`crate::Journal::to_json`] back, which no dependency would have.
//!
//! Deliberately partial: numbers are kept as their source text because nothing
//! here reads one, so a float never enters a crate the §1.4 membrane would have
//! opinions about. Duplicate keys keep the first. Depth is bounded, because the
//! input is a subprocess's stdout and a stack overflow is not a parse error.

/// A parsed value.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Json {
    Null,
    Bool(bool),
    /// The source text, unparsed — see the module note.
    Num(String),
    Str(String),
    List(Vec<Json>),
    /// Insertion order, not sorted: a `HashMap` is banned tree-wide (§3) and an
    /// object here is read by key a handful of times, never iterated for output.
    Map(Vec<(String, Json)>),
}

/// Nesting past this is malformed for our purposes. `claude`'s events are four
/// or five deep; the bound is against a truncated or hostile line, not them.
const MAX_DEPTH: usize = 32;

impl Json {
    /// Parse one whole value. `None` if the text is not one — including trailing
    /// junk, which for a JSONL line means the line was torn.
    pub(crate) fn parse(text: &str) -> Option<Json> {
        let mut p = Parser {
            b: text.as_bytes(),
            i: 0,
        };
        let value = p.value(0)?;
        p.space();
        p.done().then_some(value)
    }

    /// The value at `key`, for a map. `None` for every other variant, so a
    /// caller chains `get` without matching first.
    pub(crate) fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Map(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// The string, for a string.
    pub(crate) fn str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    /// The elements, for a list — empty for anything else, so a caller can
    /// iterate a field that may be absent without a match.
    pub(crate) fn list(&self) -> &[Json] {
        match self {
            Json::List(items) => items,
            _ => &[],
        }
    }

    /// Whether this is `true`.
    pub(crate) fn is_true(&self) -> bool {
        *self == Json::Bool(true)
    }
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn done(&self) -> bool {
        self.i == self.b.len()
    }

    fn space(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }

    /// Consume `c` if it is next.
    fn eat(&mut self, c: u8) -> bool {
        let hit = self.peek() == Some(c);
        if hit {
            self.i += 1;
        }
        hit
    }

    fn literal(&mut self, word: &str) -> bool {
        let end = self.i + word.len();
        let hit = self.b.get(self.i..end) == Some(word.as_bytes());
        if hit {
            self.i = end;
        }
        hit
    }

    fn value(&mut self, depth: usize) -> Option<Json> {
        if depth > MAX_DEPTH {
            return None;
        }
        self.space();
        match self.peek()? {
            b'{' => self.map(depth),
            b'[' => self.list(depth),
            b'"' => {
                self.i += 1;
                self.string().map(Json::Str)
            }
            b't' => self.literal("true").then_some(Json::Bool(true)),
            b'f' => self.literal("false").then_some(Json::Bool(false)),
            b'n' => self.literal("null").then_some(Json::Null),
            _ => self.number(),
        }
    }

    fn map(&mut self, depth: usize) -> Option<Json> {
        self.i += 1; // `{`
        let mut entries: Vec<(String, Json)> = Vec::new();
        self.space();
        if self.eat(b'}') {
            return Some(Json::Map(entries));
        }
        loop {
            self.space();
            if !self.eat(b'"') {
                return None;
            }
            let key = self.string()?;
            self.space();
            if !self.eat(b':') {
                return None;
            }
            let value = self.value(depth + 1)?;
            // First wins. JSON permits duplicates and says nothing about which
            // is meant; picking one deliberately beats an accident of insertion.
            if !entries.iter().any(|(k, _)| *k == key) {
                entries.push((key, value));
            }
            self.space();
            if self.eat(b',') {
                continue;
            }
            return self.eat(b'}').then_some(Json::Map(entries));
        }
    }

    fn list(&mut self, depth: usize) -> Option<Json> {
        self.i += 1; // `[`
        let mut items = Vec::new();
        self.space();
        if self.eat(b']') {
            return Some(Json::List(items));
        }
        loop {
            items.push(self.value(depth + 1)?);
            self.space();
            if self.eat(b',') {
                continue;
            }
            return self.eat(b']').then_some(Json::List(items));
        }
    }

    /// The opening quote is already consumed.
    fn string(&mut self) -> Option<String> {
        let mut out = String::new();
        loop {
            let c = self.peek()?;
            self.i += 1;
            match c {
                b'"' => return Some(out),
                b'\\' => {
                    let esc = self.peek()?;
                    self.i += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => out.push(self.escaped_char()?),
                        _ => return None,
                    }
                }
                // A raw control character is invalid JSON, and letting one
                // through would put a bare newline in a transcript line.
                c if c < 0x20 => return None,
                // Everything else is UTF-8 the source `&str` already validated,
                // so bytes copy through and only the boundary needs finding.
                _ => {
                    let start = self.i - 1;
                    while self.peek().is_some_and(|n| (0x80..0xc0).contains(&n)) {
                        self.i += 1;
                    }
                    out.push_str(core::str::from_utf8(self.b.get(start..self.i)?).ok()?);
                }
            }
        }
    }

    /// A `\uXXXX`, plus its low surrogate where it has one. The `u` is consumed.
    fn escaped_char(&mut self) -> Option<char> {
        let hi = self.hex4()?;
        // A high surrogate is only half a character; the pair encodes one above
        // the BMP, which is every emoji an answer might carry.
        if (0xd800..0xdc00).contains(&hi) {
            if !(self.eat(b'\\') && self.eat(b'u')) {
                return None;
            }
            let lo = self.hex4()?;
            if !(0xdc00..0xe000).contains(&lo) {
                return None;
            }
            let combined = 0x10000 + ((hi - 0xd800) << 10) + (lo - 0xdc00);
            return char::from_u32(combined);
        }
        char::from_u32(hi)
    }

    fn hex4(&mut self) -> Option<u32> {
        let text = core::str::from_utf8(self.b.get(self.i..self.i + 4)?).ok()?;
        let value = u32::from_str_radix(text, 16).ok()?;
        self.i += 4;
        Some(value)
    }

    /// Kept as text (see the module note), but still validated: `01` and `-`
    /// are not numbers, and accepting them would let a torn line parse.
    fn number(&mut self) -> Option<Json> {
        let start = self.i;
        self.eat(b'-');
        let digits = self.digits();
        if digits == 0 {
            return None;
        }
        // No leading zeros, per the grammar — `0` alone is fine.
        if self.b.get(start) == Some(&b'0') && digits > 1 {
            return None;
        }
        if self.eat(b'.') && self.digits() == 0 {
            return None;
        }
        if self.eat(b'e') || self.eat(b'E') {
            let _ = self.eat(b'+') || self.eat(b'-');
            if self.digits() == 0 {
                return None;
            }
        }
        Some(Json::Num(
            core::str::from_utf8(self.b.get(start..self.i)?)
                .ok()?
                .to_owned(),
        ))
    }

    fn digits(&mut self) -> usize {
        let start = self.i;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.i += 1;
        }
        self.i - start
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn an_event_line_reads_by_key() {
        let line = r#"{"type":"assistant","message":{"content":[{"type":"text","text":"hi"}]}}"#;
        let json = Json::parse(line).unwrap();
        assert_eq!(json.get("type").and_then(Json::str), Some("assistant"));
        let blocks = json.get("message").and_then(|m| m.get("content")).unwrap();
        assert_eq!(blocks.list().len(), 1);
        assert_eq!(blocks.list()[0].get("text").and_then(Json::str), Some("hi"));
        // An absent key on a present map, and a key on a non-map, are both None
        // rather than a panic — every read in `ask.rs` is a chain of them.
        assert!(json.get("nope").is_none());
        assert!(blocks.get("type").is_none());
    }

    #[test]
    fn escapes_survive_including_a_windows_path_and_an_astral_char() {
        let json =
            Json::parse(r#"{"p":"C:\\dev\\a.rs","e":"\ud83d\ude80","c":"a\nb\u0001"}"#).unwrap();
        assert_eq!(json.get("p").and_then(Json::str), Some(r"C:\dev\a.rs"));
        assert_eq!(json.get("e").and_then(Json::str), Some("🚀"));
        assert_eq!(json.get("c").and_then(Json::str), Some("a\nb\u{1}"));
    }

    /// The half-line case that made this a parser rather than a `find`: stdout
    /// is a pipe, and a torn line must read as "not yet" and never as a value.
    #[test]
    fn a_torn_or_malformed_line_is_not_a_value() {
        for bad in [
            r#"{"type":"assistant","message":{"content":[{"te"#,
            r#"{"a":1}{"b":2}"#,
            r#"{"a":01}"#,
            r#"{"a":-}"#,
            r#"{"a":"unterminated}"#,
            r#"{"a":"\q"}"#,
            r#"{"a":"\ud83d"}"#,
            "{\"a\":\"raw\nnewline\"}",
            "",
            "  ",
        ] {
            assert_eq!(Json::parse(bad), None, "{bad}");
        }
    }

    #[test]
    fn the_empty_container_and_the_bare_values_parse() {
        assert_eq!(Json::parse("{}"), Some(Json::Map(Vec::new())));
        assert_eq!(Json::parse(" [ ] "), Some(Json::List(Vec::new())));
        assert_eq!(Json::parse("null"), Some(Json::Null));
        assert!(Json::parse("true").unwrap().is_true());
        assert_eq!(
            Json::parse("-0.5e+3"),
            Some(Json::Num("-0.5e+3".to_owned()))
        );
        // `list()` on a non-list is empty rather than a panic, same argument as
        // `get` above.
        assert!(Json::parse("7").unwrap().list().is_empty());
    }

    #[test]
    fn nesting_past_the_bound_is_refused_rather_than_recursed() {
        let deep = format!("{}1{}", "[".repeat(200), "]".repeat(200));
        assert_eq!(Json::parse(&deep), None);
    }

    /// The reader against this crate's own writer. Nothing else in the tree can
    /// check `push_str_json`'s escaping against a parser — which is the concrete
    /// thing the no-dependency claim was costing until this file existed.
    #[test]
    fn it_reads_back_what_this_crate_writes() {
        let mut journal = crate::Journal::new("demo-09-orbit", "tier-dev");
        journal.at_tick(2421);
        journal.recording(std::path::PathBuf::from(r"C:\dev\GGEngine\out.ggrp"));
        journal.record(crate::Seam {
            tick: 2421,
            outcome: crate::Outcome::Refused {
                kind: "Adoption",
                detail: "field \"spin\" retyped\u{1}, path C:\\dev".into(),
            },
            code_before: "aa".into(),
            code_after: None,
            state_before: "11".into(),
            state_after: "11".into(),
            entities: 4,
            changes: Vec::new(),
            load_ms: Some(7),
            save_to_swap_ms: None,
        });
        let json = Json::parse(&journal.to_json()).expect("this crate's own output parses");
        assert_eq!(json.get("tick"), Some(&Json::Num("2421".to_owned())));
        assert_eq!(
            json.get("recording").and_then(Json::str),
            Some(r"C:\dev\GGEngine\out.ggrp")
        );
        let seam = &json.get("seams").unwrap().list()[0];
        assert_eq!(seam.get("code_after"), Some(&Json::Null));
        assert_eq!(
            seam.get("detail").and_then(Json::str),
            Some("field \"spin\" retyped\u{1}, path C:\\dev")
        );
    }
}
