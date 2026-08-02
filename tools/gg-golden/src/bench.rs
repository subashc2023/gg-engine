//! §4.11's frame macro: N warmup + M measured frames of the engine's own v1
//! pass list, reporting CPU frame percentiles and per-pass GPU milliseconds.
//!
//! It lives here because this is the only binary that renders headlessly by
//! linkage (§1.5) — the windowed shell's frame loop cannot be measured by an
//! automated tier at all — and because the per-pass numbers come from the same
//! `pass_timings()` the overlay draws and Tracy's zones read (§4.8), so a
//! regression seen here is the one a profile would show.
//!
//! **The CPU number includes the readback the screen does not pay.** An
//! offscreen frame ends in a copy to a mapped buffer and a `Vec` of the pixels;
//! there is no no-readback path through `OffscreenRenderer` and inventing one
//! would be a second frame path to keep honest. It is a constant, so the
//! percentiles still track regressions — but it is wall clock for an offscreen
//! frame, not a frame budget, and the readback pass's own GPU row is itemised
//! below so its share is visible rather than assumed.

use std::time::Instant;

use gg_extract::Extracted;
use gg_render::{OffscreenRenderer, View};

/// 720p rather than the golden scenes' 320x180: this measures passes, and a
/// frame small enough to be free measures the submit instead.
pub const EXTENT: (u32, u32) = (1280, 720);

const CLEAR: [f32; 4] = [0.02, 0.02, 0.03, 1.0];

/// One pass's accumulated device time. A `Vec` with a linear probe rather than a
/// map: pass counts are single digits, and `HashMap` is banned (§3).
struct Pass {
    name: String,
    sum_ms: f64,
    max_ms: f64,
    frames: u32,
}

pub fn run(extracted: &Extracted, frames: usize, json: bool) -> anyhow::Result<()> {
    anyhow::ensure!(frames > 0, "a measured run of zero frames measures nothing");
    // Enough to have paged in the pipelines, warmed the allocator and let the
    // clock boost settle; a tenth of the run, floored, so a smoke run still
    // warms up at all.
    let warmup = (frames / 10).max(4);

    let mut renderer = OffscreenRenderer::new(EXTENT)?;
    let device = renderer.device().chosen.clone();
    let view = View::default();
    for _ in 0..warmup {
        renderer.frame(extracted, &view, CLEAR, &[])?;
    }

    let mut cpu_ms = Vec::with_capacity(frames);
    let mut passes: Vec<Pass> = Vec::new();
    for _ in 0..frames {
        let start = Instant::now();
        renderer.frame(extracted, &view, CLEAR, &[])?;
        cpu_ms.push(start.elapsed().as_secs_f64() * 1e3);
        for timing in renderer.pass_timings() {
            let ms = f64::from(timing.gpu_ms);
            match passes.iter_mut().find(|p| p.name == timing.name) {
                Some(pass) => {
                    pass.sum_ms += ms;
                    pass.max_ms = pass.max_ms.max(ms);
                    pass.frames += 1;
                }
                None => passes.push(Pass {
                    name: timing.name.clone(),
                    sum_ms: ms,
                    max_ms: ms,
                    frames: 1,
                }),
            }
        }
    }

    let report = renderer.shutdown();
    anyhow::ensure!(
        report.clean(),
        "unclean bench run: {} validation message(s), {} leak(s) — a measurement taken through a \
         leaking device is not a measurement (§4.3)",
        report.validation_messages,
        report.leaked_allocations.len(),
    );

    cpu_ms.sort_by(f64::total_cmp);
    let stats = Percentiles::of(&cpu_ms);
    if json {
        print!("{}", encode(&device, frames, warmup, stats, &passes));
    } else {
        human(&device, frames, warmup, stats, &passes);
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Percentiles {
    min: f64,
    p50: f64,
    p95: f64,
    p99: f64,
}

impl Percentiles {
    /// Nearest-rank over an ascending slice — no interpolation, so every number
    /// printed is a frame that actually happened.
    fn of(sorted: &[f64]) -> Self {
        let at = |p: f64| -> f64 {
            #[expect(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "frame counts are thousands at most; the rank is an index"
            )]
            let rank = (p / 100.0 * sorted.len() as f64).ceil() as usize;
            sorted
                .get(rank.saturating_sub(1).min(sorted.len().saturating_sub(1)))
                .copied()
                .unwrap_or(f64::NAN)
        };
        Percentiles {
            min: at(0.0),
            p50: at(50.0),
            p95: at(95.0),
            p99: at(99.0),
        }
    }
}

fn human(device: &str, frames: usize, warmup: usize, cpu: Percentiles, passes: &[Pass]) {
    println!(
        "gg-golden bench — v1 pass list, {}x{}, {frames} measured frames after {warmup} warmup\n  \
         device {device}\n",
        EXTENT.0, EXTENT.1
    );
    println!(
        "  cpu frame (readback included)   p50 {:>7.3} ms   p95 {:>7.3} ms   p99 {:>7.3} ms   min \
         {:>7.3} ms",
        cpu.p50, cpu.p95, cpu.p99, cpu.min
    );
    if passes.is_empty() {
        println!("  no per-pass GPU timings — built without `gpu-timings` (§4.8)");
        return;
    }
    println!("\n  {:<28} {:>10} {:>10}", "pass", "mean ms", "max ms");
    for pass in passes {
        println!(
            "  {:<28} {:>10.4} {:>10.4}",
            pass.name,
            pass.sum_ms / f64::from(pass.frames),
            pass.max_ms
        );
    }
    let total: f64 = passes.iter().map(|p| p.sum_ms / f64::from(p.frames)).sum();
    println!("  {:<28} {total:>10.4}", "gpu total");
}

/// §4.11 says the frame macro reports as JSON, so it does — hand-written,
/// because one flat record is not worth a serialization dependency in a crate
/// whose whole job is pixels.
fn encode(device: &str, frames: usize, warmup: usize, cpu: Percentiles, passes: &[Pass]) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    // Infallible: writing to a String cannot fail.
    let _ = write!(
        out,
        "{{\"scene\":\"boxes\",\"extent\":[{},{}],\"device\":\"{}\",\"frames\":{frames},\
         \"warmup\":{warmup},\"cpu_frame_ms\":{{\"p50\":{:.4},\"p95\":{:.4},\"p99\":{:.4},\
         \"min\":{:.4}}},\"passes\":[",
        EXTENT.0,
        EXTENT.1,
        escape(device),
        cpu.p50,
        cpu.p95,
        cpu.p99,
        cpu.min
    );
    for (i, pass) in passes.iter().enumerate() {
        let _ = write!(
            out,
            "{}{{\"name\":\"{}\",\"mean_ms\":{:.4},\"max_ms\":{:.4}}}",
            if i == 0 { "" } else { "," },
            escape(&pass.name),
            pass.sum_ms / f64::from(pass.frames),
            pass.max_ms
        );
    }
    out.push_str("]}");
    out
}

/// Device names come from a driver, so they are the one string here nobody
/// controls.
fn escape(text: &str) -> String {
    text.chars()
        .flat_map(|c| match c {
            '"' => vec!['\\', '"'],
            '\\' => vec!['\\', '\\'],
            c if c.is_control() => vec![' '],
            c => vec![c],
        })
        .collect()
}
