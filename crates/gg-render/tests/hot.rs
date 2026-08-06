//! §9's "any shader edit is on screen in under half a second", proven as the
//! *engine's* property rather than demo 01's (§4.4): the same
//! [`OffscreenRenderer`] the golden suite drives, its shader source copied
//! aside and edited mid-run. No rebuild, no codegen — the next frames must
//! show the edit; a broken edit and a layout-changing edit must keep
//! last-good pixels; restoring the source must bring the baseline back. The
//! shell is windowed and therefore manual (§1.5), so this is where its shader
//! path is proven — [`gg_render::Renderer::frame`] polls the same watcher.
//!
//! Feature-gated: the watcher and the in-process compiler exist only under
//! `hot-reload`, so the nightly tier runs this with the feature named
//! (`ci.rs::gpu_tests`), the way `gg-debug --features tracy` gets its own leg.

#![cfg(feature = "hot-reload")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;
use std::time::{Duration, Instant};

use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenRenderer, View};

const EXTENT: (u32, u32) = (64, 64);

/// Linear mid-green: distinct from the hijack colour on every channel, so the
/// swap is unmistakable at one pixel.
const CLEAR: [f32; 4] = [0.05, 0.20, 0.05, 1.0];

/// How long an edit gets to reach the pixels before the gate calls the path
/// rotted. The §4.4 budget is 500 ms; the slack is for a loaded CI box, not
/// for the mechanism — the measured time is printed either way.
const PATIENCE: Duration = Duration::from_secs(20);

/// How long the negative cases render before "nothing changed" counts as
/// proven. Long enough for the watcher event and the recompile to have
/// happened and been refused; the positive phases show that pipeline is fast.
const REFUSAL_SOAK: Duration = Duration::from_secs(3);

fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap().flatten() {
        let target = to.join(entry.file_name());
        match entry.path().is_dir() {
            true => copy_tree(&entry.path(), &target),
            false => {
                std::fs::copy(entry.path(), &target).unwrap();
            }
        }
    }
}

/// One empty-world frame: nothing draws, so the post pass's tonemap of the
/// clear colour is every pixel — the smallest scene whose pixels still cross
/// the edited shader.
fn frame(renderer: &mut OffscreenRenderer) -> Vec<u8> {
    let view = View::default();
    let mut extracted = Extracted::default();
    extracted.clear(sim::DVec3::ZERO, view.frustum(renderer.view_extent()));
    renderer
        .frame(&extracted, &view, CLEAR, &[])
        .unwrap()
        .pixels
}

fn center(pixels: &[u8]) -> [u8; 4] {
    let at = ((EXTENT.1 / 2 * EXTENT.0 + EXTENT.0 / 2) * 4) as usize;
    pixels[at..at + 4].try_into().unwrap()
}

/// Render until the pixels leave `reference`, or `PATIENCE` runs out. Returns
/// the last frame and whether it moved.
fn settle(renderer: &mut OffscreenRenderer, reference: &[u8]) -> (Vec<u8>, bool) {
    let start = Instant::now();
    loop {
        let pixels = frame(renderer);
        if pixels != reference {
            return (pixels, true);
        }
        if start.elapsed() > PATIENCE {
            return (pixels, false);
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Render for `REFUSAL_SOAK` and insist the pixels never leave `reference`.
fn hold(renderer: &mut OffscreenRenderer, reference: &[u8], case: &str) {
    let start = Instant::now();
    while start.elapsed() < REFUSAL_SOAK {
        let pixels = frame(renderer);
        assert_eq!(
            pixels, reference,
            "{case}: the last-good pipelines were not kept"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn a_shader_edit_lands_without_a_rebuild_and_a_bad_one_keeps_last_good() {
    // A copy, never the tree: a gate that edited `shaders/` would be a gate
    // that dirtied the working copy to run.
    let dir = std::env::temp_dir().join(format!("gg-shader-hot-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    copy_tree(
        Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders")),
        &dir,
    );
    // SAFETY: read once, at `OffscreenRenderer::new` below; nextest gives each
    // test its own process, so no other thread touches the environment.
    unsafe { std::env::set_var("GG_SHADER_SRC", &dir) };

    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();
    let baseline = frame(&mut renderer);

    // A body edit: retire the tonemap's fragment entry by name and append a
    // magenta one under the old name — anchored on the signature rather than
    // on a body line, so a reformatted shader does not rot the gate.
    let post = dir.join("post.slang");
    let source = std::fs::read_to_string(&post).unwrap();
    assert!(
        source.contains("float4 fs_main("),
        "post.slang lost its fs_main anchor"
    );
    let hijacked = format!(
        "{}\n[shader(\"fragment\")]\nfloat4 fs_main(VOut i) : SV_Target\n{{\n    \
         return float4(1.0, 0.0, 1.0, 1.0);\n}}\n",
        source.replace("float4 fs_main(", "float4 fs_retired(")
    );
    let edited_at = Instant::now();
    std::fs::write(&post, &hijacked).unwrap();
    let (pixels, moved) = settle(&mut renderer, &baseline);
    assert!(moved, "the shader edit never reached the pixels (§4.4)");
    let px = center(&pixels);
    assert!(
        px[0] > 200 && px[1] < 50 && px[2] > 200,
        "the pixels moved but not to the edit's magenta: {px:?}"
    );
    // Printed rather than asserted at 500 ms: the §4.4 budget is measured on
    // the desk (demo 01's save-to-screen log and the shell's), and a loaded CI
    // box failing the gate on wall clock would gate the weather. PATIENCE is
    // the rot detector.
    println!(
        "shader edit on screen in {} ms (§4.4 budget: 500)",
        edited_at.elapsed().as_millis()
    );
    let magenta = pixels;

    // A broken edit: the compiler refuses, the last-good pipelines keep
    // drawing, and the session survives to be edited again.
    std::fs::write(&post, "this is not slang\n").unwrap();
    hold(&mut renderer, &magenta, "broken edit");

    // A layout edit: compiles clean, but the push-constant block no longer
    // matches what this build's codegen froze — swapping it in would corrupt
    // every draw, so it must be refused the same way (§4.4 codegen).
    let grown = hijacked.replace("uint reserved;", "uint reserved;\n    float4 grown;");
    assert!(
        grown != hijacked,
        "post.slang lost its `uint reserved;` anchor"
    );
    std::fs::write(&post, &grown).unwrap();
    hold(&mut renderer, &magenta, "push-constant layout edit");

    // And back: the original source restores the baseline, which also proves
    // the watcher outlived both refusals.
    std::fs::write(&post, &source).unwrap();
    let (pixels, moved) = settle(&mut renderer, &magenta);
    assert!(
        moved,
        "restoring the source never brought the baseline back"
    );
    assert_eq!(pixels, baseline, "the restored shader drew something else");

    let _ = std::fs::remove_dir_all(&dir);
}
