//! The debug overlay (§4.8): frame stats, per-pass GPU milliseconds, memory,
//! and the CVar console — immediate mode, on our own renderer, over the draw
//! layer in [`crate::draw`].
//!
//! It reads what the frame already measured rather than measuring again: the
//! pass rows are the same [`PassTiming`]s Tracy's GPU zones are fed, so the two
//! views of one frame cannot disagree.
//!
//! The console line takes keys directly, and consuming a *press* is the whole
//! of its input handling — hit-testing, focus and capture are M13's (§4.8).
//! Releases are never consumed, so a key held when the console opened still
//! reaches the action map and cannot latch.

use crate::capture;
use crate::console;
use crate::draw::{DrawList, Rect};
use crate::font;
use gg_core::{CVar, CVarError, cvar};
use gg_input::Key;
use gg_render::ui::{Coverage, UiVertex};
use gg_rhi::{MemoryUse, PassTiming};

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

/// The coverage atlas the UI pass cuts the overlay's glyphs from — hand it to
/// `Renderer::set_ui_atlas` once. Expanded on first call and kept, because the
/// renderer may be rebuilt (a device loss, a second window) and re-expanding
/// 4.5 KiB per bring-up is not worth a second copy of the bytes.
pub fn atlas() -> Coverage<'static> {
    static TEXELS: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    Coverage {
        extent: font::EXTENT,
        texels: TEXELS.get_or_init(font::atlas),
    }
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
/// between the data and the picture.
const CHART_WIDTH: f32 = gg_render::luminance::BINS as f32;
const CHART_HEIGHT: f32 = 20.0;

/// The overlay's own state across frames: the frame-time window it averages
/// and the console's line and scrollback. Everything else is rebuilt per frame.
pub struct Overlay {
    list: DrawList,
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
            Key::F1 => {
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
        let line = DrawList::line_height();
        let mut rows: Vec<(String, u32)> = Vec::with_capacity(stats.passes.len() + 4);
        let (average, worst) = self.frame_window();
        rows.push((
            format!("{:>6.2} ms  {:>3.0} fps", average, 1e3 / average.max(1e-3)),
            ACCENT,
        ));
        rows.push((format!("worst   {worst:>6.2} ms"), DIM));
        rows.push((format!("tick    {:>8}", stats.tick), TEXT));
        match stats.passes.is_empty() {
            // Not "0.000": a build without the query pool measured nothing, and
            // a zero would read as a pass that cost nothing (§4.8).
            true => rows.push(("gpu     (no timestamps)".to_owned(), DIM)),
            false => {
                let total: f32 = stats.passes.iter().map(|p| p.gpu_ms).sum();
                rows.push((format!("gpu     {total:>6.3} ms"), TEXT));
                for pass in stats.passes {
                    rows.push((format!(" {:<14}{:>6.3}", pass.name, pass.gpu_ms), DIM));
                }
            }
        }
        rows.push((
            format!(
                "mem  {:>7} {} buf {} img",
                mib(stats.memory.total_bytes()),
                stats.memory.buffers,
                stats.memory.images
            ),
            TEXT,
        ));
        // Only under RenderDoc: a row that always read "unavailable" would be a
        // row about the tool rather than about the frame.
        if capture::available() {
            rows.push((format!("rdoc  {:>7}  F11", capture::count()), DIM));
        }

        let chart = stats.luminance.map_or(0.0, |_| CHART_HEIGHT + line);
        let width = rows
            .iter()
            .map(|(text, _)| DrawList::width(text))
            .fold(CHART_WIDTH, f32::max);
        let panel = Rect::new(
            PAD,
            PAD,
            width + PAD * 2.0,
            rows.len() as f32 * line + chart + PAD * 2.0,
        );
        self.list.rect(panel, PANEL);
        // Clipped to the panel it was measured against: a pass name longer than
        // the widest row would otherwise run out over the scene.
        self.list.push_clip(panel);
        for (i, (text, color)) in rows.iter().enumerate() {
            self.list
                .text(PAD * 2.0, PAD * 2.0 + i as f32 * line, text, *color);
        }
        if let Some(bins) = stats.luminance {
            self.histogram(bins, PAD * 2.0, PAD * 2.0 + rows.len() as f32 * line);
        }
        self.list.pop_clip();
    }

    /// The luminance histogram (§6 M11's exit row): one bar per bucket, darkest
    /// on the left, normalized to the tallest.
    ///
    /// Normalized to the tallest rather than to the sample count, and that is
    /// what makes it readable: a frame is usually dominated by one bucket, and a
    /// chart scaled to the total would show that bucket and a flat line.
    fn histogram(&mut self, bins: &[u32; gg_render::luminance::BINS], x: f32, y: f32) {
        let line = DrawList::line_height();
        let tallest = bins.iter().copied().max().unwrap_or(0).max(1) as f32;
        self.list
            .text(x, y, "luma  dark ....... 0EV ... bright", DIM);
        let top = y + line;
        for (i, &count) in bins.iter().enumerate() {
            let height = (count as f32 / tallest * CHART_HEIGHT).max(f32::from(count > 0));
            let bar = Rect::new(x + i as f32, top + CHART_HEIGHT - height, 1.0, height);
            self.list.rect(bar, ACCENT);
        }
        // The bucket a luminance of 1.0 falls in — the reference an exposure
        // decision is made against. A tick rather than a number, because the
        // question the chart answers is "where is the mass", not "how much".
        let zero_ev = gg_render::luminance::BINS * 2 / 3;
        self.list
            .rect(Rect::new(x + zero_ev as f32, top, 1.0, CHART_HEIGHT), DIM);
    }

    fn console_panel(&mut self, logical: (f32, f32)) {
        let line = DrawList::line_height();
        let rows = self.console.output.len() as f32 + 1.0;
        let height = rows * line + PAD * 2.0;
        let panel = Rect::new(PAD, logical.1 - height - PAD, logical.0 - PAD * 2.0, height);
        self.list.rect(panel, PANEL);
        self.list.push_clip(panel);
        let x = panel.x + PAD;
        let mut y = panel.y + PAD;
        for text in &self.console.output {
            self.list.text(x, y, text, DIM);
            y += line;
        }
        // A block cursor rather than a blinking one: a blink is a clock the
        // overlay would have to own, and a screenshot of a blinked-out cursor
        // is a bug report about a missing cursor.
        let pen = self.list.text(x, y, "> ", ACCENT);
        let pen = self.list.text(pen, y, &self.console.line, TEXT);
        self.list
            .rect(Rect::new(pen, y, font::GLYPH.0 as f32, line - 1.0), ACCENT);
        self.list.pop_clip();
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

/// Bytes as mebibytes, one decimal — the unit every GPU tool reports in.
fn mib(bytes: u64) -> String {
    format!("{:.1}M", bytes as f64 / (1024.0 * 1024.0))
}

/// Lines the console keeps on screen. Deliberately short: the full history is
/// in the log, and a console that covers the game it is inspecting is worse
/// than one that scrolls.
const OUTPUT: usize = 10;

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
        // F1 is the same switch, so the key and the CVar cannot disagree.
        overlay.key(Key::F1, true);
        assert!(SHOW.bool());
        assert!(!overlay.build(&stats()).is_empty());
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
        let passes = [PassTiming {
            name: "forward-opaque".to_owned(),
            gpu_ms: 0.5,
            begin: 0,
            end: 1,
        }];
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
}
