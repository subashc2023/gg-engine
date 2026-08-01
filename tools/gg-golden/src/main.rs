//! `gg-golden` v0 (§4.10): offscreen render → readback → PNG →
//! exact/tolerance compare, wired as a CI gate — visual regression testing
//! from the first triangle. Headless **by linkage**: this binary never links
//! gg-platform/winit, and gate 7's symbol-absence check proves it stays that
//! way. v1 (M7) grows replay-driven scenes, the perceptual gate, and the HTML
//! report on this spine.
//!
//! Usage:
//!   gg-golden run  [scene]     compare scenes against checked-in references
//!   gg-golden bless [scene]    (re)write references — a deliberate, reviewed
//!                              act; image diffs belong in the PR (§4.10)

mod compare;
mod png_io;

use compare::Policy;
use std::path::PathBuf;

/// RGBA8 pixels plus their extent — what a scene render produces.
type Render = anyhow::Result<(Vec<u8>, (u32, u32))>;

/// One golden scene: how to render it and how strictly to judge it.
struct Scene {
    name: &'static str,
    policy: Policy,
    render: fn() -> Render,
}

/// The v0 roster. Scenes register here; M7 replaces this with replay-driven
/// discovery.
const SCENES: &[Scene] = &[Scene {
    name: "triangle",
    // Lavapipe is deterministic on one box, but edge rasterization may move a
    // pixel across driver updates: tolerate nothing per-channel beyond 2, and
    // at most 16 stray pixels of a 640x360 frame (§4.10 per-test config).
    policy: Policy {
        tolerance: 2,
        max_diff_pixels: 16,
    },
    render: render_triangle,
}];

/// Render demo 01's scene — the same SPIR-V and push constants the demo draws
/// with (§4.10: the golden guards the demo, not a lookalike).
fn render_triangle() -> Render {
    let extent = demo_01_triangle::GOLDEN_EXTENT;
    let mut rhi = gg_rhi::OffscreenRhi::new(extent)?;
    tracing::info!(device = %rhi.device_report().chosen, "offscreen device");
    let pipeline = rhi.create_pipeline(&demo_01_triangle::pipeline_desc())?;
    let push = demo_01_triangle::push_for_extent(extent);
    let draw = gg_rhi::DrawSpec {
        pipeline,
        push_constants: bytemuck::bytes_of(&push),
        vertex_count: demo_01_triangle::VERTEX_COUNT,
    };
    let pixels = rhi.render(demo_01_triangle::CLEAR, Some(&draw))?;
    let report = rhi.shutdown();
    anyhow::ensure!(
        report.clean(),
        "unclean render: {} validation message(s), {} leak(s) {:?} (§4.3, §5.4)",
        report.validation_messages,
        report.leaked_allocations.len(),
        report.leaked_allocations,
    );
    Ok((pixels, extent))
}

/// Reference sets are per-backend (§4.10): software and hardware rasterizers
/// legitimately differ, and the two lavapipe pins (per OS, §5.4) do too.
fn backend_id() -> anyhow::Result<String> {
    // A tiny bring-up just to read the device name would be wasteful; scenes
    // already log it. For the reference key, one probe context is fine at v0
    // scale (one scene) — revisit when scene count makes it matter.
    let rhi = gg_rhi::OffscreenRhi::new((4, 4))?;
    let chosen = rhi.device_report().chosen.clone();
    drop(rhi.shutdown());
    let driver = if chosen.to_lowercase().contains("llvmpipe") {
        "lavapipe".to_string()
    } else {
        chosen
            .to_lowercase()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
            .collect::<String>()
            .split('-')
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("-")
    };
    let os = if cfg!(windows) { "windows" } else { "linux" };
    Ok(format!("{driver}-{os}"))
}

/// `<workspace>/tests/gg-images` (§3 layout): references live with the tests,
/// under the §4.10 size budget.
fn references_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .map(|root| root.join("tests/gg-images"))
        .unwrap_or_else(|| PathBuf::from("tests/gg-images"))
}

fn run(filter: Option<&str>) -> anyhow::Result<()> {
    let backend = backend_id()?;
    let root = references_root().join(&backend);
    let mut failures = Vec::new();
    let mut ran = 0usize;

    for scene in SCENES {
        if filter.is_some_and(|f| f != scene.name) {
            continue;
        }
        ran += 1;
        let (actual, extent) = (scene.render)()?;
        let reference_path = root.join(format!("{}.png", scene.name));
        if !reference_path.exists() {
            failures.push(format!(
                "{}: no reference for backend `{backend}` at {} — render verified clean; \
                 run `gg-golden bless {}` on this machine and review the image into the PR",
                scene.name,
                reference_path.display(),
                scene.name
            ));
            continue;
        }
        let (reference, ref_extent) = png_io::read(&reference_path)?;
        if ref_extent != extent {
            failures.push(format!(
                "{}: reference is {}x{}, render is {}x{}",
                scene.name, ref_extent.0, ref_extent.1, extent.0, extent.1
            ));
            continue;
        }
        let comparison = compare::compare(&actual, &reference, scene.policy)?;
        if comparison.passes(scene.policy) {
            tracing::info!(
                scene = scene.name,
                diff_pixels = comparison.diff_pixels,
                max_delta = comparison.max_delta,
                "golden pass"
            );
        } else {
            // Failure artifacts: actual + heatmap beside the build products.
            let out_dir = PathBuf::from("target/golden").join(scene.name);
            let actual_path = out_dir.join("actual.png");
            let heatmap_path = out_dir.join("diff-heatmap.png");
            png_io::write(&actual_path, &actual, extent)?;
            png_io::write(
                &heatmap_path,
                &compare::heatmap(&comparison, scene.policy.tolerance),
                extent,
            )?;
            failures.push(format!(
                "{}: {} differing pixel(s) (max channel delta {}) against tolerance {}/{} — \
                 see {} and {}",
                scene.name,
                comparison.diff_pixels,
                comparison.max_delta,
                scene.policy.tolerance,
                scene.policy.max_diff_pixels,
                actual_path.display(),
                heatmap_path.display(),
            ));
        }
    }

    anyhow::ensure!(ran > 0, "no scene matched the filter");
    anyhow::ensure!(
        failures.is_empty(),
        "golden suite failed:\n{}",
        failures.join("\n")
    );
    println!("gg-golden: {ran} scene(s) pass against `{backend}` references");
    Ok(())
}

fn bless(filter: Option<&str>) -> anyhow::Result<()> {
    let backend = backend_id()?;
    let root = references_root().join(&backend);
    let mut blessed = 0usize;
    for scene in SCENES {
        if filter.is_some_and(|f| f != scene.name) {
            continue;
        }
        let (actual, extent) = (scene.render)()?;
        let path = root.join(format!("{}.png", scene.name));
        png_io::write(&path, &actual, extent)?;
        println!(
            "gg-golden: blessed {} — a deliberate, reviewed act; the image diff belongs in the PR (§4.10)",
            path.display()
        );
        blessed += 1;
    }
    anyhow::ensure!(blessed > 0, "no scene matched the filter");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let filter = args.get(1).map(String::as_str);
    match args.first().map(String::as_str) {
        Some("run") => run(filter),
        Some("bless") => bless(filter),
        _ => anyhow::bail!("usage: gg-golden <run|bless> [scene]"),
    }
}
