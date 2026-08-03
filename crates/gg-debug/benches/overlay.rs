//! §6 M13's "no frame-time regression" clause, measured (§4.11).
//!
//! M8 had no frame-time baseline for the overlay, so there is no recorded
//! number to regress from. What there *is* is a known shape: M8 built its rows
//! as `format!`ed `String`s pushed into a fresh `Vec` and folded to find the
//! widest; M13 builds them into a reused [`Scratch`] and measures with
//! [`Stack`]. That substitution is the entire difference between the two, so it
//! is what this measures — both shapes, over the same row set, in one process.
//!
//! No benchmark framework, for `gg-ecs`'s reason: the number asserted is a
//! *ratio* between two bodies timed on the same machine in the same run, which
//! makes timer quality nearly irrelevant. Absolute nanoseconds are printed for
//! the record and the whole-frame cost is printed beside them, so the reader
//! can see what fraction of a frame the panel is at all.
//!
//! Run: `cargo bench -p gg-debug` (or `cargo xtask bench`, the nightly tier).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use gg_debug::overlay::{Overlay, Stats};
use gg_rhi::{MemoryUse, PassTiming};
use gg_ui::{DrawList, Scratch, Span, Stack};

/// Frames per measurement, and repeats. The minimum of the repeats is reported:
/// noise on a desktop is one-sided.
const FRAMES: u32 = 200;
const REPS: u32 = 15;

/// Inset and colour, copied from the overlay so the two shapes lay out the same
/// panel. They are the overlay's private constants; a bench that reached for
/// them would be a reason to make them public, which they are not.
const PAD: f32 = 4.0;
const TEXT: u32 = 0xffd8_e0e8;

fn passes() -> Vec<PassTiming> {
    ["forward-opaque", "shadow", "resolve", "tonemap", "ui"]
        .iter()
        .enumerate()
        .map(|(i, name)| PassTiming {
            name: (*name).to_owned(),
            gpu_ms: 0.4 + i as f32 * 0.15,
            begin: i as i64,
            end: i as i64 + 1,
        })
        .collect()
}

fn measure(name: &str, mut body: impl FnMut()) -> Duration {
    for _ in 0..3 {
        body(); // warm the caches and the branch predictors
    }
    let mut best = Duration::MAX;
    for _ in 0..REPS {
        let start = Instant::now();
        body();
        best = best.min(start.elapsed());
    }
    let per_frame = best.as_secs_f64() * 1e9 / f64::from(FRAMES);
    println!("{name:<28} {best:>9.3?}  {per_frame:>9.1} ns/frame");
    best
}

/// The row set both shapes build — the stats panel's, with the numbers moving
/// so every row reformats. A fixed string would settle after one frame and
/// measure a memcpy.
///
/// Handed over as `fmt::Arguments` rather than as strings, which is the whole
/// point: the same formatting reaches a fresh `String` on one side and a
/// [`Scratch`] span on the other, and nothing else differs. Building the rows
/// once and copying them into both would charge M13 for M8's allocation and
/// measure nothing.
fn emit_rows(tick: u64, passes: &[PassTiming], f: &mut dyn FnMut(std::fmt::Arguments<'_>, u32)) {
    let ms = 16.0 + (tick % 13) as f32 / 8.0;
    f(format_args!("{ms:>6.2} ms  {:>3.0} fps", 1e3 / ms), TEXT);
    f(format_args!("worst   {:>6.2} ms", ms + 4.0), TEXT);
    f(format_args!("tick    {tick:>8}"), TEXT);
    f(format_args!("gpu     {:>6.3} ms", ms / 4.0), TEXT);
    for pass in passes {
        f(format_args!(" {:<14}{:>6.3}", pass.name, pass.gpu_ms), TEXT);
    }
    f(
        format_args!("mem  {:>6.1}M {} buf {} img", 128.5, 41, 17),
        TEXT,
    );
}

/// M8's shape: a fresh `Vec<(String, u32)>` every frame, and a fold over it to
/// find the widest row, because there was no [`Stack`] to ask.
fn m8(list: &mut DrawList, tick: u64, passes: &[PassTiming]) {
    let mut rows: Vec<(String, u32)> = Vec::new();
    emit_rows(tick, passes, &mut |args, color| {
        rows.push((std::fmt::format(args), color));
    });
    let line = DrawList::line_height();
    let widest = rows
        .iter()
        .map(|(text, _)| DrawList::width(text))
        .fold(0.0f32, f32::max);
    let height = rows.len() as f32 * line;
    list.rect(
        gg_ui::Rect::new(PAD, PAD, widest + PAD * 2.0, height + PAD * 2.0),
        0xc00c_1016,
    );
    for (i, (text, color)) in rows.iter().enumerate() {
        list.text(PAD * 2.0, PAD * 2.0 + i as f32 * line, text, *color);
    }
}

/// M13's shape: spans in a reused [`Scratch`], measured by a [`Stack`] that is
/// walked twice — once to size the panel, once to place the rows.
fn m13(
    list: &mut DrawList,
    scratch: &mut Scratch,
    rows: &mut Vec<(Span, u32)>,
    tick: u64,
    passes: &[PassTiming],
) {
    scratch.clear();
    rows.clear();
    let line = DrawList::line_height();
    emit_rows(tick, passes, &mut |args, color| {
        rows.push((scratch.line(args), color));
    });
    let mut stack = Stack::vertical((PAD * 2.0, PAD * 2.0), 0.0);
    for (span, _) in rows.iter() {
        stack.push(DrawList::width(scratch.get(*span)), line);
    }
    list.rect(stack.content().inset(-PAD), 0xc00c_1016);
    stack.rewind();
    for (span, color) in rows.iter() {
        let cell = stack.push(0.0, line);
        list.text(cell.x, cell.y, scratch.get(*span), *color);
    }
}

fn main() {
    let passes = passes();
    let bins = [7u32; gg_render::luminance::BINS];

    let mut list = DrawList::default();
    let old = measure("m8: String rows + fold", || {
        for tick in 0..u64::from(FRAMES) {
            list.clear();
            m8(&mut list, tick, &passes);
            black_box(list.vertices().len());
        }
    });

    let mut scratch = Scratch::default();
    let mut rows: Vec<(Span, u32)> = Vec::new();
    let new = measure("m13: Scratch spans + Stack", || {
        for tick in 0..u64::from(FRAMES) {
            list.clear();
            m13(&mut list, &mut scratch, &mut rows, tick, &passes);
            black_box(list.vertices().len());
        }
    });

    // The whole panel, console and histogram included, for scale. Not asserted:
    // it is the number a reader wants when deciding whether any of this matters,
    // and an absolute timing gate on a desktop is a flake generator.
    let mut overlay = Overlay::default();
    let whole = measure("m13: the whole overlay frame", || {
        for tick in 0..u64::from(FRAMES) {
            let stats = Stats {
                extent: (1920, 1080),
                tick,
                passes: &passes,
                memory: MemoryUse::default(),
                luminance: Some(&bins),
            };
            black_box(overlay.build(&stats).len());
        }
    });

    let ratio = new.as_secs_f64() / old.as_secs_f64();
    let share = whole.as_secs_f64() * 1e3 / f64::from(FRAMES) / 16.667;
    println!(
        "\nm13/m8 {ratio:.3}x   whole overlay frame is {:.3}% of a 60 Hz frame",
        share * 100.0
    );

    // `--json` for `xtask bench --record`, which archives this run (§4.11).
    // Emitted before the assertion below so the numbers reach the archive of a
    // run that ends red as well — the reader wants to see how far.
    if std::env::args().any(|a| a == "--json") {
        let ns = |d: Duration| d.as_secs_f64() * 1e9 / f64::from(FRAMES);
        println!(
            "{{\"frames\":{FRAMES},\"reps\":{REPS},\"ns_per_frame\":{{\"m8_rows\":{:.1},\
             \"m13_rows\":{:.1},\"whole_overlay\":{:.1}}},\"ratios\":{{\"m13_over_m8\":{ratio:.4},\
             \"frame_share_60hz\":{share:.6}}}}}",
            ns(old),
            ns(new),
            ns(whole),
        );
    }

    // The criterion, as a ratio. A tenth of slack for a noisy desk: what this
    // has to catch is a shape that costs *more*, not a percent of jitter.
    assert!(
        ratio <= 1.10,
        "the M13 row shape is {ratio:.3}x the M8 one — §6 M13 asks for no frame-time regression"
    );
}
