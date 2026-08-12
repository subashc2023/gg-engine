//! The two antialiasing modes (§6 M21) — `r.aa`'s post-process edge pass and
//! `r.msaa`'s multisampled scene pass — proven offscreen where a test can read
//! the pixels — the shell is windowed and therefore manual (§1.5).
//!
//! Every mode answers the same two claims, which fail in opposite directions:
//! a flat region must come out **bit-identical** to the unfiltered frame, and a
//! silhouette must come out different. One without the other is satisfied by a
//! pass that does nothing, or by a pass that blurs the whole picture — and both
//! of those have shipped in real engines under these names.
//!
//! MSAA answers a third that FXAA cannot: the count it *reports* must be one
//! the device actually advertises. It is the only mode here whose request may
//! be reduced rather than refused, and a renderer that reduced it silently
//! would leave an operator judging 8× by a 4× picture.
//!
//! One [`OffscreenRenderer`] for the whole file, because a device bring-up is
//! what an offscreen suite pays for and the pixels are nearly free (§6 M20
//! item 11).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_ecs::World;
use gg_ecs::boundary::{Light, Renderable};
use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenRenderer, View};
use gg_rhi::Samples;

const EXTENT: (u32, u32) = (96, 96);

/// Linear and dark, so the box's lit faces are the only bright thing and every
/// edge in the frame is one these passes are meant to find.
const CLEAR: [f32; 4] = [0.02, 0.02, 0.03, 1.0];

/// Corners the box never reaches, whatever the mode does to its silhouette.
const FLAT: [(u32, u32); 4] = [(1, 1), (2, 3), (EXTENT.0 - 2, 1), (1, EXTENT.1 - 2)];

fn pixel(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
    let at = ((y * EXTENT.0 + x) * 4) as usize;
    pixels[at..at + 4].try_into().unwrap()
}

/// One box, rotated off both axes so its silhouette is a set of *diagonals* —
/// the staircase these passes exist for. An axis-aligned box would be the one
/// subject with no aliasing to remove.
fn world() -> World {
    let mut world = World::new();
    world.register::<Renderable>().unwrap();
    world.register::<Light>().unwrap();
    let mut box_ = Renderable::boxed(
        sim::DVec3::new(0.0, 0.0, -4.0),
        sim::Vec3::splat(1.1),
        0x00d0_c8b0,
    );
    // Off both axes at once, so no face is parallel to a scanline.
    box_.rotation = sim::DQuat::from_axis_angle(sim::DVec3::new(0.0, 1.0, 0.0), 0.6).mul(
        sim::DQuat::from_axis_angle(sim::DVec3::new(1.0, 0.0, 0.0), 0.35),
    );
    let entity = world.spawn();
    world.insert(entity, box_).unwrap();
    let sun = world.spawn();
    world
        .insert(
            sun,
            Light::sun(sim::Vec3::new(0.3, -0.5, -1.0), 0x00ff_ffff, 4.0),
        )
        .unwrap();
    world
}

/// One frame in one mode, through the renderer the caller already brought up.
fn render(renderer: &mut OffscreenRenderer, world: &World, aa: bool, samples: Samples) -> Vec<u8> {
    let view = View {
        aa,
        samples,
        ..View::default()
    };
    let mut extracted = Extracted::default();
    extracted.clear(sim::DVec3::ZERO, view.frustum(renderer.view_extent()));
    extracted.append::<Renderable>(world).unwrap();
    extracted.append_lights(world).unwrap();
    renderer
        .frame(&extracted, &view, CLEAR, &[])
        .unwrap()
        .pixels
}

/// Every pixel that differs between two frames, and by how much at worst.
fn difference(a: &[u8], b: &[u8]) -> (usize, u8) {
    let (mut moved, mut worst) = (0usize, 0u8);
    for y in 0..EXTENT.1 {
        for x in 0..EXTENT.0 {
            let (l, r) = (pixel(a, x, y), pixel(b, x, y));
            if l != r {
                moved += 1;
                for channel in 0..3 {
                    worst = worst.max(l[channel].abs_diff(r[channel]));
                }
            }
        }
    }
    (moved, worst)
}

/// Both directions in one measurement, because either alone is satisfied by
/// something wrong: a corner the box never reaches must come back
/// **bit-identical** (a filter that touched flat colour would move every
/// reference by a rounding step), and the silhouette must move — bounded above
/// as well, since a pass that changed a quarter of the frame is a blur.
fn assert_antialiases(off: &[u8], on: &[u8], mode: &str) {
    for (x, y) in FLAT {
        assert_eq!(
            pixel(on, x, y),
            pixel(off, x, y),
            "{mode} moved flat clear colour at ({x}, {y})"
        );
    }
    let (moved, worst) = difference(off, on);
    let total = (EXTENT.0 * EXTENT.1) as usize;
    println!("{mode} moved {moved}/{total} pixels, worst channel delta {worst}");
    assert!(
        moved > total / 100,
        "{mode} changed {moved} of {total} pixels — it is not reaching the edges"
    );
    assert!(
        moved < total / 4,
        "{mode} changed {moved} of {total} pixels — that is a blur, not an edge pass"
    );
    // A softened staircase is a real move and not a rounding one: an off-by-one
    // in a tap offset, or a resolve that averaged one sample, would show up here
    // as a frame differing everywhere by a single level.
    assert!(
        worst > 8,
        "{mode}'s biggest change was {worst} levels — too small to be an edge"
    );
}

/// Both modes are off by default, which is the claim every blessed golden
/// (§4.10) rests on: the frame the suite renders today is the frame it was
/// blessed against. Free of a device on purpose — this is the assertion that
/// would catch an AA leaking into the default path, and it should cost nothing
/// to keep.
#[test]
fn the_default_frame_is_the_unfiltered_one() {
    let view = View::default();
    assert!(!view.aa, "`r.aa` is not off by default");
    assert_eq!(view.samples, Samples::X1, "`r.msaa` is not off by default");
}

/// Both modes against the same unfiltered frame, and against each other.
///
/// 4× rather than 8× because it is the count both backends in the execution
/// matrix advertise — pinned lavapipe's set is `[1, 4, 8]` and the desk's 4090
/// is `[1, 2, 4, 8]`, so 4× is the one an automated tier can prove on either
/// (§8 residual: 2× is reachable only on the real GPU).
#[test]
fn each_mode_softens_a_silhouette_and_leaves_flat_colour_alone() {
    let world = world();
    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();
    let off = render(&mut renderer, &world, false, Samples::X1);
    let fxaa = render(&mut renderer, &world, true, Samples::X1);
    let msaa = render(&mut renderer, &world, false, Samples::X4);

    assert_antialiases(&off, &fxaa, "fxaa");
    assert_antialiases(&off, &msaa, "msaa 4x");

    // And they are not each other: two modes that produced the same pixels
    // would be one mode wired twice, which a pair of "differs from off" claims
    // would happily pass.
    let (moved, _) = difference(&fxaa, &msaa);
    assert!(
        moved > 0,
        "fxaa and msaa produced identical frames — one of them is not wired"
    );
}

/// The count the renderer *reports* is one the device actually does, for every
/// count asked — the claim that separates a granted reduction from a silent
/// one, and the one that catches clamping to a maximum instead of testing
/// membership. Pinned lavapipe is exactly the device that punishes the
/// difference: it advertises 1, 4 and 8 and **not** 2, so asking for 2× there
/// must land on 1× rather than on a count no driver claimed.
#[test]
fn a_count_the_device_does_not_have_is_reduced_to_one_it_does() {
    let world = world();
    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();
    let advertised = renderer.device().samples.clone();
    assert!(
        advertised.contains(&Samples::X1),
        "every device does 1x; this one reports {advertised:?}"
    );
    for asked in Samples::ALL {
        render(&mut renderer, &world, false, asked);
        let granted = renderer.samples();
        assert!(
            advertised.contains(&granted),
            "asked {}x and rendered {}x, which this device does not advertise ({advertised:?})",
            asked.count(),
            granted.count()
        );
        assert!(
            granted <= asked,
            "asked {}x and got {}x — a reduction, never an increase",
            asked.count(),
            granted.count()
        );
    }
}

/// What MSAA does to the *graph*, which is the half the pixels cannot show:
/// the forward pass draws into a multisampled attachment and resolves into the
/// single-sample one, and every pass after it reads that same single-sample
/// image under either mode.
///
/// This is also what makes MSAA visible in `--dump-render-graph` where FXAA is
/// not: FXAA is a push constant and moves no edge, so a dump cannot show it.
#[test]
fn the_forward_pass_resolves_and_everything_after_it_reads_one_image() {
    let mut rhi = gg_rhi::OffscreenRhi::new(EXTENT).unwrap();
    let mut pool = gg_render::graph::Transients::default();
    {
        let mut frame = pool.frame(&mut rhi, EXTENT).unwrap();
        let backbuffer = frame.backbuffer();
        let scene = frame
            .color_at(
                "scene.color",
                EXTENT,
                gg_rhi::ImageFormat::Rgba16F,
                Samples::X1,
            )
            .unwrap();
        let scene_ms = frame
            .color_at(
                "scene.color.ms",
                EXTENT,
                gg_rhi::ImageFormat::Rgba16F,
                Samples::X4,
            )
            .unwrap();
        let depth = frame.depth_at("scene.depth", EXTENT, Samples::X4).unwrap();

        // A multisampled attachment has no bindless slot, which is the type-level
        // reason nothing downstream can sample it by accident.
        assert!(
            frame.texture(scene_ms).is_none(),
            "a multisample attachment must not be registered for sampling"
        );
        assert!(
            frame.texture(scene).is_some(),
            "the resolve target is sampled"
        );

        let samples = [scene];
        let declared = gg_render::scene_graph(&gg_render::SceneFrame {
            backbuffer,
            scene,
            scene_ms: Some(scene_ms),
            depth,
            shadows: &[],
            lamps: None,
            clear: CLEAR,
            viewport: None,
            prepass_draws: &[],
            shadow_draws: &[],
            lamp_draws: &[],
            forward_draws: &[],
            post_draws: &[],
            samples: &samples,
            forward_samples: &[],
        });
        let compiled = frame.compile(&declared).unwrap();
        let dump = compiled.dump();
        println!("{dump}");

        // The multisample attachment is written by exactly one pass and read by
        // none: a second `ColorWrite` on it, or any `SampledRead`, would mean
        // something downstream picked up the samples instead of the average.
        assert_eq!(
            dump.matches("barrier scene.color.ms: None -> ColorWrite")
                .count(),
            1,
            "the forward pass is the only writer of the multisample attachment"
        );
        assert!(
            !dump.contains("scene.color.ms: ColorWrite -> SampledRead"),
            "nothing may sample the multisample attachment — the resolve is what is read"
        );
        // And the resolve target is written by the same pass and sampled by the
        // post one, which is the single-sample path an unmultisampled frame has.
        assert!(
            dump.contains("barrier scene.color: None -> ColorWrite"),
            "the resolve target is an output of the forward pass:\n{dump}"
        );
        assert!(
            dump.contains("barrier scene.color: ColorWrite -> SampledRead"),
            "the post pass reads the resolve target:\n{dump}"
        );
    }
    pool.destroy(&mut rhi).unwrap();
    assert!(rhi.shutdown().clean(), "no leaks, no validation messages");
}
