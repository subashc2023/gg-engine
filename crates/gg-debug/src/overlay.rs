//! The debug overlay (§4.8): frame stats, per-pass GPU milliseconds, memory,
//! and the CVar console — immediate mode, on `gg-ui`.
//!
//! §6 M13's acceptance test: [`Scratch`] and [`Stack`] hold every row, so the
//! layer either does this cheaply or it is overbuilt — `tests/no_alloc` is the
//! criterion in its concrete form.
//!
//! It reads what the frame already measured rather than measuring again: the
//! pass rows are the same [`PassTiming`]s Tracy's GPU zones are fed, so the two
//! views of one frame cannot disagree.
//!
//! The console line takes keys directly, and consuming a *press* is the whole
//! of its input handling. Releases are never consumed, so a key held when the
//! console opened still reaches the action map and cannot latch. It stays
//! keyboard-only deliberately: `gg_ui::Router` exists now, but the overlay has
//! nothing to hit-test, and routing a panel with no clickable widget in it
//! would be the speculative build §4.9 keeps out.

use crate::capture;
use crate::console;
use gg_core::{CVar, CVarError, cvar};
use gg_input::Key;
use gg_render::ui::UiVertex;
use gg_rhi::{MemoryUse, PassTiming};
use gg_ui::{DrawList, Rect, Scratch, Span, Stack, font};

/// Whether the overlay draws at all.
pub static SHOW: CVar = CVar::new_bool("d.overlay", true, "draw the debug overlay");
/// Integer pixel scale. Integer because the font is a bitmap: a fractional
/// scale under a nearest sampler drops and doubles rows of a 7-pixel glyph.
pub static SCALE: CVar = CVar::new_int("d.scale", 2, "debug overlay pixel scale");

/// Register the overlay's knobs.
///
/// # Errors
///
/// A name already taken.
pub fn register() -> Result<(), CVarError> {
    cvar::register_all(&[&SHOW, &SCALE])
}

/// What one frame has to say for itself.
pub struct Stats<'a> {
    /// Target size in physical pixels.
    pub extent: (u32, u32),
    /// The sim tick that last ran.
    pub tick: u64,
    /// Per-pass GPU milliseconds from the last retired frame — empty in a build
    /// without timestamps, which is why the row says so rather than showing 0.
    pub passes: &'a [PassTiming],
    /// Live device allocations.
    pub memory: MemoryUse,
    /// The frame's luminance distribution, darkest bucket first, or `None` when
    /// `r.histogram` is off (§6 M11). Buckets are log2 luminance over twelve
    /// stops; the renderer decides the range and this only draws it.
    pub luminance: Option<&'a [u32; gg_render::luminance::BINS]>,
}

/// Frames kept for the average — two seconds at 60 Hz, long enough that the
/// number stops flickering and short enough to still react to a stall.
const HISTORY: usize = 120;

/// Panel fill: dark and translucent, so the scene stays readable behind it.
const PANEL: u32 = 0xc00c_1016;
const TEXT: u32 = 0xffd8_e0e8;
const DIM: u32 = 0xff8a_94a0;
const ACCENT: u32 = 0xff7f_d0a0;
/// Inset between a panel's edge and its text.
const PAD: f32 = 4.0;
/// The histogram's plot area, in unscaled cells. One cell per bucket, so the
/// width is the bucket count and a bar is exactly one column — no resampling
/// between the data and the picture. The *cell* the chart occupies is this or
/// its label, whichever is wider; the bars stay one column each either way.
const CHART_WIDTH: f32 = gg_render::luminance::BINS as f32;
const CHART_HEIGHT: f32 = 20.0;
/// The chart's axis, such as it is. A const because the panel is measured
/// against it — a label wider than the bars it names would otherwise be the one
/// thing in the panel the clip cuts.
const LUMA_LABEL: &str = "luma  dark ....... 0EV ... bright";
/// Bytes to mebibytes, the unit every GPU tool reports in.
const MIB: f64 = 1024.0 * 1024.0;

/// The overlay's own state across frames: the frame-time window it averages,
/// the console's line and scrollback, and the two buffers this frame's rows are
/// rebuilt into. Everything else is rebuilt per frame.
pub struct Overlay {
    list: DrawList,
    scratch: Scratch,
    /// This frame's rows: where the text landed in `scratch`, and its colour.
    /// Cleared and refilled, never reallocated — the `Vec<String>` it replaces
    /// was the overlay's whole allocation budget.
    rows: Vec<(Span, u32)>,
    frame_ms: [f32; HISTORY],
    at: usize,
    seen: usize,
    last: Option<std::time::Instant>,
    console: Console,
}

impl Default for Overlay {
    fn default() -> Self {
        Overlay {
            list: DrawList::default(),
            scratch: Scratch::default(),
            rows: Vec::new(),
            frame_ms: [0.0; HISTORY],
            at: 0,
            seen: 0,
            last: None,
            console: Console::default(),
        }
    }
}

impl Overlay {
    /// Whether the console is taking keys — what tells a host that the next
    /// keystroke is text and not a verb.
    pub fn console_open(&self) -> bool {
        self.console.open
    }

    /// Offer a key. `true` means the overlay consumed it and the caller must
    /// not feed it to the action map; only presses are ever consumed.
    pub fn key(&mut self, key: Key, pressed: bool) -> bool {
        if !pressed {
            return false;
        }
        match key {
            // The console's key everywhere, and the reason `text` filters it:
            // the keystroke that opens the console also produces a character.
            Key::Backquote => {
                self.console.open = !self.console.open;
                true
            }
            // Two keys, one switch — F3 is the one hands already reach for.
            Key::F1 | Key::F3 => {
                SHOW.set_bool(!SHOW.bool());
                true
            }
            // Arms rather than captures: what RenderDoc records is the *next*
            // frame, and the one this keystroke landed in is already half built.
            Key::F11 => {
                capture::request(1);
                true
            }
            Key::Enter if self.console.open => {
                self.console.submit();
                true
            }
            Key::Backspace if self.console.open => {
                self.console.line.pop();
                true
            }
            Key::Escape if self.console.open => {
                self.console.open = false;
                true
            }
            // Everything else while typing: swallowed, so `wireframe 1` does
            // not also walk the camera.
            _ => self.console.open,
        }
    }

    /// Offer a character the platform produced. Ignored unless the console is
    /// open, and control characters never reach the line.
    pub fn text(&mut self, c: char) {
        if self.console.open && c != '`' && !c.is_control() {
            self.console.line.push(c);
        }
    }

    /// Build this frame's geometry. Empty when `d.overlay` is off, which is
    /// also how the renderer skips the pass entirely.
    pub fn build(&mut self, stats: &Stats<'_>) -> &[UiVertex] {
        let now = std::time::Instant::now();
        if let Some(last) = self.last.replace(now) {
            self.frame_ms[self.at] = now.duration_since(last).as_secs_f32() * 1e3;
            self.at = (self.at + 1) % HISTORY;
            self.seen = (self.seen + 1).min(HISTORY);
        }
        self.list.clear();
        if !SHOW.bool() {
            return self.list.vertices();
        }
        let scale = SCALE.int().clamp(1, 8) as f32;
        self.list.push_transform((0.0, 0.0), scale);
        // Everything below is authored in unscaled cells; the transform is the
        // only place the pixel scale exists.
        let logical = (stats.extent.0 as f32 / scale, stats.extent.1 as f32 / scale);
        self.stats_panel(stats);
        if self.console.open {
            self.console_panel(logical);
        }
        self.list.pop_transform();
        self.list.vertices()
    }

    fn stats_panel(&mut self, stats: &Stats<'_>) {
        let (average, worst) = self.frame_window();
        let Overlay {
            list,
            scratch,
            rows,
            ..
        } = self;
        let line = DrawList::line_height();
        scratch.clear();
        rows.clear();
        rows.push((
            scratch.line(format_args!(
                "{average:>6.2} ms  {:>3.0} fps",
                1e3 / average.max(1e-3)
            )),
            ACCENT,
        ));
        rows.push((scratch.line(format_args!("worst   {worst:>6.2} ms")), DIM));
        rows.push((
            scratch.line(format_args!("tick    {:>8}", stats.tick)),
            TEXT,
        ));
        match stats.passes.is_empty() {
            // Not "0.000": a build without the query pool measured nothing, and
            // a zero would read as a pass that cost nothing (§4.8).
            true => rows.push((scratch.line(format_args!("gpu     (no timestamps)")), DIM)),
            false => {
                let total: f32 = stats.passes.iter().map(|p| p.gpu_ms).sum();
                rows.push((scratch.line(format_args!("gpu     {total:>6.3} ms")), TEXT));
                for pass in stats.passes {
                    let row = scratch.line(format_args!(" {:<14}{:>6.3}", pass.name, pass.gpu_ms));
                    rows.push((row, DIM));
                }
            }
        }
        rows.push((
            scratch.line(format_args!(
                "mem  {:>6.1}M {} buf {} img",
                stats.memory.total_bytes() as f64 / MIB,
                stats.memory.buffers,
                stats.memory.images
            )),
            TEXT,
        ));
        // Only under RenderDoc: a row that always read "unavailable" would be a
        // row about the tool rather than about the frame.
        if capture::available() {
            let row = scratch.line(format_args!("rdoc  {:>7}  F11", capture::count()));
            rows.push((row, DIM));
        }

        // Measured, then placed. The panel is behind the rows and so must be
        // emitted before them, and it cannot be sized until they exist — which
        // is the two-pass shape `Stack::rewind` is for.
        let chart = stats.luminance.map(|_| {
            let width = CHART_WIDTH.max(DrawList::width(LUMA_LABEL));
            (width, CHART_HEIGHT + line)
        });
        let mut stack = Stack::vertical((PAD * 2.0, PAD * 2.0), 0.0);
        for (span, _) in rows.iter() {
            stack.push(DrawList::width(scratch.get(*span)), line);
        }
        if let Some((w, h)) = chart {
            stack.push(w, h);
        }
        let panel = stack.content().inset(-PAD);
        list.rect(panel, PANEL);
        // Clipped to what it was measured against, so anything that draws wider
        // than it measured is cut rather than running out over the scene.
        list.push_clip(panel);
        stack.rewind();
        for (span, color) in rows.iter() {
            let text = scratch.get(*span);
            let cell = stack.push(DrawList::width(text), line);
            list.text(cell.x, cell.y, text, *color);
        }
        if let (Some(bins), Some((w, h))) = (stats.luminance, chart) {
            histogram(list, bins, stack.push(w, h));
        }
        list.pop_clip();
    }

    fn console_panel(&mut self, logical: (f32, f32)) {
        let Overlay { list, console, .. } = self;
        let line = DrawList::line_height();
        // The one panel that is *not* sized to its contents: it is pinned to the
        // bottom edge and spans the window, so its box is known up front.
        let height = (console.output.len() as f32 + 1.0) * line + PAD * 2.0;
        let panel = Rect::new(PAD, logical.1 - height - PAD, logical.0 - PAD * 2.0, height);
        list.rect(panel, PANEL);
        list.push_clip(panel);
        let mut stack = Stack::vertical((panel.x + PAD, panel.y + PAD), 0.0);
        for text in &console.output {
            let cell = stack.push(DrawList::width(text), line);
            list.text(cell.x, cell.y, text, DIM);
        }

        let cell = stack.push(0.0, line);
        let mut pen = Stack::horizontal((cell.x, cell.y), 0.0);
        for (text, color) in [("> ", ACCENT), (console.line.as_str(), TEXT)] {
            let cell = pen.push(DrawList::width(text), line);
            list.text(cell.x, cell.y, text, color);
        }
        // A block cursor rather than a blinking one: a blink is a clock the
        // overlay would have to own, and a screenshot of a blinked-out cursor
        // is a bug report about a missing cursor.
        list.rect(pen.push(font::GLYPH.0 as f32, line - 1.0), ACCENT);
        list.pop_clip();
    }

    /// Mean and worst of the frames on record.
    fn frame_window(&self) -> (f32, f32) {
        if self.seen == 0 {
            return (0.0, 0.0);
        }
        let seen = &self.frame_ms[..self.seen];
        (
            seen.iter().sum::<f32>() / self.seen as f32,
            seen.iter().copied().fold(0.0, f32::max),
        )
    }
}

/// The luminance histogram (§6 M11's exit row): one bar per bucket, darkest on
/// the left, normalized to the tallest, inside the cell the stack reserved.
///
/// Normalized to the tallest rather than to the sample count, and that is what
/// makes it readable: a frame is usually dominated by one bucket, and a chart
/// scaled to the total would show that bucket and a flat line.
fn histogram(list: &mut DrawList, bins: &[u32; gg_render::luminance::BINS], at: Rect) {
    let line = DrawList::line_height();
    let tallest = bins.iter().copied().max().unwrap_or(0).max(1) as f32;
    list.text(at.x, at.y, LUMA_LABEL, DIM);
    let top = at.y + line;
    for (i, &count) in bins.iter().enumerate() {
        let height = (count as f32 / tallest * CHART_HEIGHT).max(f32::from(count > 0));
        let bar = Rect::new(at.x + i as f32, top + CHART_HEIGHT - height, 1.0, height);
        list.rect(bar, ACCENT);
    }
    // The bucket a luminance of 1.0 falls in — the reference an exposure
    // decision is made against. A tick rather than a number, because the
    // question the chart answers is "where is the mass", not "how much".
    let zero_ev = gg_render::luminance::BINS * 2 / 3;
    list.rect(
        Rect::new(at.x + zero_ev as f32, top, 1.0, CHART_HEIGHT),
        DIM,
    );
}

/// Lines the console keeps on screen. Deliberately short: the full history is
/// in the log, and a console that covers the game it is inspecting is worse
/// than one that scrolls.
const OUTPUT: usize = 10;

/// Not on [`Scratch`]: scrollback outlives the frame that produced it, which is
/// the one thing a span into a per-frame buffer cannot do. It allocates when a
/// human submits a line and never per frame.
#[derive(Default)]
struct Console {
    open: bool,
    line: String,
    output: std::collections::VecDeque<String>,
}

impl Console {
    fn submit(&mut self) {
        let line = std::mem::take(&mut self.line);
        if line.trim().is_empty() {
            return;
        }
        self.push(format!("> {line}"));
        for reply in console::run(&line) {
            self.push(reply);
        }
    }

    fn push(&mut self, text: String) {
        self.output.push_back(text);
        while self.output.len() > OUTPUT {
            self.output.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// The registry is process-wide and these tests share one, so registration
    /// happens once however many of them need it.
    fn registered() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| register().expect("the overlay's knobs are unclaimed"));
    }

    fn stats() -> Stats<'static> {
        Stats {
            extent: (1280, 720),
            tick: 42,
            passes: &[],
            memory: MemoryUse::default(),
            luminance: None,
        }
    }

    /// The console is a mode, and the mode is what decides whether a keystroke
    /// is a verb or a letter. A press consumed here never reaches the action
    /// map; a *release* never is, so a key held across the toggle still clears.
    #[test]
    fn opening_the_console_claims_presses_and_never_releases() {
        let mut overlay = Overlay::default();
        assert!(!overlay.key(Key::W, true), "closed: the game's key");
        assert!(overlay.key(Key::Backquote, true));
        assert!(overlay.console_open());
        assert!(overlay.key(Key::W, true), "open: swallowed");
        assert!(!overlay.key(Key::W, false), "a release always gets through");
        assert!(overlay.key(Key::Escape, true));
        assert!(!overlay.console_open());
    }

    /// The keystroke that opens the console also produces a backquote; typing
    /// it into the line is the classic bug and the filter is in `text`.
    #[test]
    fn the_toggle_key_never_lands_in_the_line() {
        let mut overlay = Overlay::default();
        overlay.key(Key::Backquote, true);
        overlay.text('`');
        overlay.text('\u{8}');
        overlay.text('r');
        assert_eq!(overlay.console.line, "r");
        // And nothing types while it is closed.
        overlay.key(Key::Escape, true);
        overlay.text('x');
        assert_eq!(overlay.console.line, "r");
    }

    #[test]
    fn a_submitted_line_runs_and_its_reply_is_kept() {
        registered();
        let mut overlay = Overlay::default();
        overlay.key(Key::Backquote, true);
        for c in "d.scale 3".chars() {
            overlay.text(c);
        }
        overlay.key(Key::Enter, true);
        assert_eq!(SCALE.int(), 3);
        assert!(overlay.console.line.is_empty());
        assert!(
            overlay.console.output.iter().any(|l| l.contains("d.scale")),
            "{:?}",
            overlay.console.output
        );
        SCALE.reset();
    }

    /// The output ring is bounded: a console left open under a chatty command
    /// must not grow a panel taller than the screen.
    #[test]
    fn the_output_ring_is_bounded() {
        let mut console = Console::default();
        for i in 0..OUTPUT * 3 {
            console.push(format!("line {i}"));
        }
        assert_eq!(console.output.len(), OUTPUT);
        assert_eq!(console.output.back().map(String::as_str), Some("line 29"));
    }

    #[test]
    fn the_cvar_switches_the_whole_layer_off() {
        let mut overlay = Overlay::default();
        assert!(!overlay.build(&stats()).is_empty());
        SHOW.set_bool(false);
        assert!(overlay.build(&stats()).is_empty());
        // F1 and F3 are the same switch, so a key and the CVar cannot disagree.
        assert!(overlay.key(Key::F1, true));
        assert!(SHOW.bool());
        assert!(!overlay.build(&stats()).is_empty());
        assert!(overlay.key(Key::F3, true));
        assert!(!SHOW.bool());
        assert!(overlay.build(&stats()).is_empty());
        SHOW.reset();
    }

    /// The histogram is drawn only when there is one, and the panel grows to
    /// hold it. A chart clipped to a panel sized for text would be a chart
    /// nobody can see, which is the failure this shape is chosen to avoid.
    #[test]
    fn the_histogram_appears_only_when_the_frame_measured_one() {
        let mut overlay = Overlay::default();
        let without = overlay.build(&stats()).len();

        let mut bins = [0u32; gg_render::luminance::BINS];
        bins[20] = 100;
        bins[21] = 40;
        let with = Stats {
            luminance: Some(&bins),
            ..stats()
        };
        let count = overlay.build(&with).len();
        assert!(count > without, "{count} vertices vs {without}");

        // Every bar stays inside the panel it was measured against, which the
        // clip would otherwise hide rather than fix.
        let extent = with.extent;
        for vertex in overlay.build(&with) {
            assert!(
                vertex.pos[0] <= extent.0 as f32 && vertex.pos[1] <= extent.1 as f32,
                "{:?} is off a {extent:?} target",
                vertex.pos
            );
        }
    }

    /// The panel is sized to its contents and the clip is taken from the *same*
    /// measurement, so a row that got longer widens the panel rather than being
    /// cut by it. A measure pass and a place pass that disagree is the failure
    /// this shape can have and the flat `PAD + i * line` arithmetic could not.
    #[test]
    fn a_long_row_widens_the_panel_instead_of_being_clipped_by_it() {
        // A fresh overlay per build: with no frame on record the timing rows
        // format identically, so the pass name is the only difference between
        // the two and the vertex counts are exactly comparable.
        let count = |passes: &[PassTiming]| {
            let mut overlay = Overlay::default();
            overlay.build(&Stats { passes, ..stats() }).len()
        };
        // Exactly the column width, so the longer name adds characters rather
        // than eating the padding that would have been there anyway.
        let short = timings("forward-opaque", 0.5);
        let long = timings("forward-opaque-and-then-some", 0.5);
        let extra = (long[0].name.len() - short[0].name.len()) * 6;
        assert_eq!(
            count(&long),
            count(&short) + extra,
            "six vertices per glyph and not one dropped"
        );
    }

    /// Every quad the overlay emits stays on the target it was told about — a
    /// panel that ran off the bottom would be a panel nobody can read.
    #[test]
    fn nothing_is_drawn_outside_the_target() {
        let mut overlay = Overlay::default();
        overlay.key(Key::Backquote, true);
        for c in "list".chars() {
            overlay.text(c);
        }
        overlay.key(Key::Enter, true);
        let passes = timings("forward-opaque", 0.5);
        let stats = Stats {
            passes: &passes,
            ..stats()
        };
        let extent = stats.extent;
        for vertex in overlay.build(&stats) {
            assert!(
                vertex.pos[0] >= 0.0
                    && vertex.pos[1] >= 0.0
                    && vertex.pos[0] <= extent.0 as f32
                    && vertex.pos[1] <= extent.1 as f32,
                "{:?} is off a {extent:?} target",
                vertex.pos
            );
        }
    }

    fn timings(name: &str, gpu_ms: f32) -> [PassTiming; 1] {
        [PassTiming {
            name: name.to_owned(),
            gpu_ms,
            begin: 0,
            end: 1,
        }]
    }
}
