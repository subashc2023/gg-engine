//! `gg-golden` v1 (§4.10, M7): offscreen render → readback → PNG → two gates,
//! wired as a CI gate — visual regression testing from the first triangle.
//! Headless **by linkage**: this binary never links gg-platform/winit, and gate
//! 7's symbol-absence check proves it stays that way.
//!
//! Usage:
//!   gg-golden run   [scene]    compare scenes against checked-in references
//!   gg-golden bless [scene]    (re)write references — a deliberate, reviewed
//!                              act; image diffs belong in the PR (§4.10)
//!   gg-golden graph [scene]    print each scene's render graph (§4.5's
//!                              `--dump-render-graph`)
//!   gg-golden verify-gates     prove both gates can fail and can forgive
//!   gg-golden chaos [seed]     render chaos streams' terminal frames (§5.11)
//!   gg-golden capture [scene]  render under RenderDoc, write a `.rdc` (§4.8)
//!   gg-golden bench [--json] [--frames N]   §4.11's frame macro
//!   gg-golden load  [pack]     time a pack from mapped to resident (§6 M9)

mod bench;
mod compare;
mod png_io;
mod report;

use compare::{Comparison, Policy, Verdict};
use gg_render::graph::{Declared, Transients, readback_pass};
use gg_rhi::{BufferDesc, BufferKind, OffscreenRhi};
use std::path::PathBuf;

/// What a scene render produces: RGBA8 pixels, their extent, and the graph
/// that drew them.
///
/// The dump rides along with the pixels rather than being regenerated on
/// request, which is what makes §6 M6's "matches the executed order" a property
/// of one object instead of a claim about two code paths.
struct Capture {
    pixels: Vec<u8>,
    extent: (u32, u32),
    graph: String,
}

type Render = anyhow::Result<Capture>;

/// The buffer §4.5's readback pass copies a frame into. The harness owns this
/// and nothing else about a scene's graph: what it renders is the scene's own
/// declaration list with one pass appended, which is what makes the golden
/// guard the demo's *frame* rather than a lookalike of it (§4.10).
fn readback_buffer(
    rhi: &mut OffscreenRhi,
    extent: (u32, u32),
) -> anyhow::Result<gg_rhi::BufferHandle> {
    Ok(rhi.create_buffer(&BufferDesc {
        name: "golden.readback",
        size: u64::from(extent.0) * u64::from(extent.1) * 4,
        kind: BufferKind::Readback,
    })?)
}

/// Every scene ends here: a reference rendered by a device that complained or
/// leaked is not a reference, so teardown accounting is part of the render, not
/// a separate check (§4.3, §5.4). One home so the citation and the wording
/// cannot drift between scenes.
fn ensure_clean(report: &gg_rhi::ShutdownReport) -> anyhow::Result<()> {
    anyhow::ensure!(
        report.clean(),
        "unclean render: {} validation message(s), {} leak(s) {:?} (§4.3, §5.4)",
        report.validation_messages,
        report.leaked_allocations.len(),
        report.leaked_allocations,
    );
    Ok(())
}

/// One golden scene: how to render it and how strictly to judge it.
struct Scene {
    name: &'static str,
    policy: Policy,
    render: fn() -> Render,
}

/// The roster. Nineteen scenes — seven demos, the engine's own v1 pass list, two
/// replay-driven captures, the UI layer, and three the harness builds itself to
/// put a lighting feature on its own against nothing else (§6 M26-M28) — each
/// with its own policy, because "how strictly" is a property of what the frame
/// contains and not of the harness (§4.10 per-test config).
const SCENES: &[Scene] = &[
    Scene {
        name: "triangle",
        // Lavapipe is deterministic on one box, but edge rasterization may move
        // a pixel across driver updates: tolerate nothing per-channel beyond 2,
        // and at most 16 stray pixels of a 640x360 frame (§4.10 per-test config).
        policy: Policy {
            tolerance: 2,
            max_diff_pixels: 16,
            benign_delta: 4,
            max_dssim: 0.02,
            max_bias: 0.25,
        },
        render: render_triangle,
    },
    Scene {
        name: "mesh",
        // Looser than the triangle by design: this frame has three silhouette
        // edges per face, a BC7 decoder whose interpolation is fixed-point but
        // whose *filtering* is not bit-specified, and a depth test deciding
        // pixels along every crease. Per-channel 3 still catches a wrong
        // texture index, a lost transfer, or a flipped depth comparison, which
        // is what this scene is for.
        policy: Policy {
            tolerance: 3,
            max_diff_pixels: 256,
            benign_delta: 6,
            max_dssim: 0.03,
            max_bias: 0.25,
        },
        render: render_mesh,
    },
    Scene {
        name: "mesh-far",
        // The same frame, simulated 10^12 m from the origin (§4.2.1). It is
        // judged against its *own* reference rather than against `mesh`, so the
        // gate catches a regression in the narrowing itself; the claim that the
        // two frames are the same picture is a demo unit test on the clip-space
        // corners, where a sub-pixel difference is measurable instead of
        // rounded away. Same policy as `mesh`: nothing about the distance is
        // supposed to make the image harder to reproduce, and if it does, that
        // is the finding.
        policy: Policy {
            tolerance: 3,
            max_diff_pixels: 256,
            benign_delta: 6,
            max_dssim: 0.03,
            max_bias: 0.25,
        },
        render: render_mesh_far,
    },
    Scene {
        name: "boxes",
        // The engine's own v1 pass list (§4.5), which the shell runs and no
        // automated tier can otherwise see: depth prepass, forward opaque into
        // an offscreen attachment, fullscreen post onto the target. Flat colours
        // over hard silhouettes — the diffuse term is the only smooth thing in
        // the frame, so a tolerance of 2 is plenty and anything looser would
        // stop noticing a wrong normal.
        policy: Policy {
            tolerance: 2,
            max_diff_pixels: 64,
            benign_delta: 4,
            max_dssim: 0.02,
            max_bias: 0.25,
        },
        render: render_boxes,
    },
    Scene {
        name: "boxes-occluded",
        // The same three boxes from an angle that makes them overlap. This is
        // the scene that judges *depth*: a flipped comparison, a lost prepass or
        // a depth attachment shared between frames in flight (§6 M6) reorders
        // which colour wins along every crease, and nothing else in the roster
        // would notice.
        policy: Policy {
            tolerance: 2,
            max_diff_pixels: 64,
            benign_delta: 4,
            max_dssim: 0.02,
            max_bias: 0.25,
        },
        render: render_boxes_occluded,
    },
    Scene {
        name: "mesh-replay",
        // The curated replay, played back through the sim, captured deep into
        // the script (§4.10's replay-driven playback). Judged as strictly as
        // `mesh` — it is the same mesh and the same shader; what differs is that
        // the pose arrived through 330 ticks of action state, so a divergence in
        // the input path lands here as a picture rather than only as a hash.
        policy: Policy {
            tolerance: 3,
            max_diff_pixels: 256,
            benign_delta: 6,
            max_dssim: 0.03,
            max_bias: 0.25,
        },
        render: render_mesh_replay,
    },
    Scene {
        name: "chaos-witness",
        // §5.11's chaos generator, gated as a *picture* rather than only as a
        // hash (§6 M7). One seed, not all eight: every seed drives the same code
        // and the hash baseline already covers all of them across three
        // architectures, so eight references would buy repetition and charge a
        // re-bless for it. `gg-golden chaos <seed>` renders any of the others on
        // demand, which is what a divergence actually needs.
        policy: Policy {
            tolerance: 3,
            max_diff_pixels: 256,
            benign_delta: 6,
            max_dssim: 0.03,
            max_bias: 0.25,
        },
        render: render_chaos_witness,
    },
    Scene {
        name: "hall",
        // Demo 04's pack (§4.6): geometry, a tiled BC7 base-colour map and a
        // material factor, none of which is written in any Rust source — the
        // frame comes out of `ggc`. Judged like `mesh` and for the same reason:
        // BC7 filtering is not bit-specified, and per-channel 3 still catches a
        // lost mip level, a wrong sampler, or a scene node placed by the CPU
        // instead of by the file. It is also the only scene in the roster whose
        // reference moves when an *asset* changes rather than when code does.
        policy: Policy {
            tolerance: 3,
            max_diff_pixels: 256,
            benign_delta: 6,
            max_dssim: 0.03,
            max_bias: 0.25,
        },
        render: render_hall,
    },
    Scene {
        name: "field",
        // Demo 05's ten thousand objects (§6 M10): four meshes, four materials,
        // a hundred hubs and a transform hierarchy the *host* composed. Judged
        // like `hall` — same pack pipeline, same BC7 filtering — but what it
        // guards that no other scene does is the batcher and the sort key: a
        // batch that dropped its tail, a key that ordered materials wrongly, or
        // an instance array staged at the wrong offset all land here as pixels.
        policy: Policy {
            tolerance: 3,
            max_diff_pixels: 256,
            benign_delta: 6,
            max_dssim: 0.03,
            max_bias: 0.25,
        },
        render: render_field,
    },
    Scene {
        name: "atrium",
        // §6 M11's lit scene: metal-rough shading over all four maps, a sun that
        // casts a shadow map, four point lights, and the tonemapper on the way
        // out. Judged like `hall` on the per-channel side — the same BC7
        // filtering is under it — but the **perceptual** gate is what carries
        // the weight here, exactly as the exit row says: this frame is smooth
        // gradients over curved surfaces, where a driver's rounding moves many
        // more pixels by a little than a hard-edged frame does, and a wrong
        // light direction or a lost shadow moves *structure* instead.
        policy: Policy {
            tolerance: 4,
            max_diff_pixels: 512,
            benign_delta: 8,
            max_dssim: 0.03,
            max_bias: 0.2,
        },
        render: render_atrium,
    },
    Scene {
        name: "atrium-noon",
        // The same room with the sun a quarter of its sweep on, which puts the
        // pillars' shadows across the floor at a different angle and the
        // spheres' highlights on their other sides. Two lit references rather
        // than one because a single sun angle can hide a shadow projection that
        // is wrong in only one axis — the classic symptom of a light-space basis
        // built from a fixed world up.
        policy: Policy {
            tolerance: 4,
            max_diff_pixels: 512,
            benign_delta: 8,
            max_dssim: 0.03,
            max_bias: 0.2,
        },
        render: render_atrium_noon,
    },
    Scene {
        name: "mirror",
        // §6 M27's subject, and the only scene in the roster lit by *nothing but*
        // the environment: no sun, no lamps, no ambient worth the name. Every
        // photon in this frame came through the prefiltered chain, so a chain
        // that stopped being sampled renders a black frame rather than a
        // slightly different one — which is the property a reference is for.
        //
        // Judged on the perceptual side like the atrium: it is five smooth
        // spheres of curved gradient, where a driver's rounding moves many
        // pixels a little. What it must not forgive is *structure* — the window
        // panes reflected in the mirror end, which is exactly what the three-band
        // convolution this replaced could not produce at all.
        policy: Policy {
            tolerance: 4,
            max_diff_pixels: 512,
            benign_delta: 8,
            max_dssim: 0.03,
            max_bias: 0.2,
        },
        render: render_mirror,
    },
    Scene {
        name: "volumes",
        // §6 M28's subject: two environments and the fade between them. Same
        // policy as `mirror` and for the same reasons — smooth curved gradients
        // over a large fraction of the frame — with one difference that matters
        // here. The bias bound is tight, because the failure this scene exists
        // to catch is *the wrong environment winning*, and one sky standing in
        // for another is a whole-frame level shift long before it is a
        // structural one.
        policy: Policy {
            tolerance: 4,
            max_diff_pixels: 512,
            benign_delta: 8,
            max_dssim: 0.03,
            max_bias: 0.1,
        },
        render: render_volumes,
    },
    Scene {
        name: "parallax",
        // §6 M29's subject, and it is deliberately *one* environment: the camera
        // stands inside a single bounded room, so nothing here blends and what
        // moves can only be the correction. Judged as `mirror` is — the same
        // chain reflected in the same smooth metal — because the failure it
        // catches is structural rather than a level shift: without the
        // correction every ball reflects the same picture from the same
        // directions, and the windows land at the same place on all five.
        policy: Policy {
            tolerance: 4,
            max_diff_pixels: 512,
            benign_delta: 8,
            max_dssim: 0.03,
            max_bias: 0.2,
        },
        render: render_parallax,
    },
    Scene {
        name: "lanterns",
        // §6 M30's subject: ninety-six point lights, three times the cap this
        // roster lived under until now, arranged so that most of them reach no
        // part of most of the frame. What it grades is the *assignment* — a
        // froxel list that under-includes shows here as a pool of light with a
        // straight edge across it, which is a structural failure and not a level
        // one.
        //
        // Judged like `atrium`, whose subject it shares: many smooth falloff
        // gradients over flat surfaces, where a driver's rounding moves a great
        // many pixels a little. The bias bound is the atrium's rather than the
        // volumes' — a missing lamp darkens a patch, and a whole-frame level
        // shift is not the failure mode.
        policy: Policy {
            tolerance: 4,
            max_diff_pixels: 512,
            benign_delta: 8,
            max_dssim: 0.03,
            max_bias: 0.2,
        },
        render: render_lanterns,
    },
    Scene {
        name: "lampshade",
        // §6 M31's subject, and it is the one thing a *sun's* shadow can never
        // show: three pillars around one low lamp throw shadows that **diverge**
        // — each one widening away from its own pillar, each pointing a
        // different way, none of them parallel. A cascade's are parallel by
        // construction, so a lookup that had quietly fallen back to the sun's
        // path would render this frame as three stripes going the same way.
        //
        // It also grades the two failure directions the offscreen test measures
        // in numbers: a lamp that shadows nothing (three lit pillars on an even
        // floor) and one that shadows everything (a dark frame, or the acne band
        // at each pillar's foot that a wrong bias makes).
        //
        // Judged tightly. Nothing here is a smooth falloff over a huge surface
        // the way the atrium is — the subject is a hard boundary in a small
        // frame, and a hard boundary that moves is exactly what must fail.
        policy: Policy {
            tolerance: 2,
            max_diff_pixels: 128,
            benign_delta: 4,
            max_dssim: 0.03,
            max_bias: 0.15,
        },
        render: render_lampshade,
    },
    Scene {
        name: "ui-overlay",
        // §6 M13's acceptance test as a picture: the real debug overlay, frozen.
        // Flat fills over a translucent panel and a bitmap font under a nearest
        // sampler — there is nothing in this frame a driver is entitled to round
        // differently, so it is judged as strictly as the roster allows. What it
        // catches is a clip that stopped cutting, a panel that stopped fitting
        // its rows, and a histogram that stopped being normalized.
        policy: Policy {
            tolerance: 1,
            max_diff_pixels: 16,
            benign_delta: 2,
            max_dssim: 0.02,
            max_bias: 0.25,
        },
        render: render_ui_overlay,
    },
    Scene {
        name: "editor",
        // §6 M15's third exit row: the editor is a golden-image subject like any
        // other scene. Flat fills and a nearest-sampled bitmap font at an exact
        // ×2 fit, so nothing here is a driver's to round — judged as strictly as
        // `ui-overlay` and catching the same class of thing one panel wider: a
        // clip that stopped cutting, a row that stopped fitting, a selection
        // highlight that stopped following the selection.
        policy: Policy {
            tolerance: 1,
            max_diff_pixels: 16,
            benign_delta: 2,
            max_dssim: 0.02,
            max_bias: 0.25,
        },
        render: render_editor,
    },
    Scene {
        name: "ui-text",
        // The text-heavy scene the M13 exit row names. Every glyph edge is
        // eight-bit coverage the CPU produced, so unlike every other scene here
        // the *rasterizer* is under test and not only the driver. It turns out
        // to be portable — the Windows and Linux lavapipe references are byte
        // for byte the same file — which is worth knowing precisely because
        // nothing enforces it: zeno's outline rasterizer is ordinary `f32` work
        // and sits nowhere near §1.4's membrane. Judged like the mesh scenes all
        // the same: per-channel 3 forgives a rounded edge and still catches a
        // wrong slot, a stale atlas, or an advance that drifted.
        policy: Policy {
            tolerance: 3,
            max_diff_pixels: 256,
            benign_delta: 6,
            max_dssim: 0.03,
            max_bias: 0.25,
        },
        render: render_ui_text,
    },
    Scene {
        name: "tetris",
        // §6 M18's playfield. Flat fills and the bitmap font at an exact ×1 fit
        // — the whole game is UI (§4.9), so there is nothing in this frame a
        // driver may round and it is judged as strictly as `ui-overlay`. What it
        // catches is the class of thing no unit test can: a board off centre, a
        // panel over the well, a cell colour that stopped being its piece's, and
        // a label clipped by a rectangle that was wide enough yesterday.
        policy: Policy {
            tolerance: 1,
            max_diff_pixels: 16,
            benign_delta: 2,
            max_dssim: 0.02,
            max_bias: 0.25,
        },
        render: render_tetris,
    },
    Scene {
        name: "platformer",
        // §6 M20's level under the orthographic camera — the roster's only
        // ortho subject, and its world is the checked-in `scene.ggsave` seen
        // through the `Eye` the scene itself holds, so an authored level moves
        // this reference rather than diverging from it. What it guards that no
        // unit test can: the ortho projection as a picture (equal-width slabs
        // at every depth), the box-shaped cascade fit over a flat playfield
        // (§6 M15.3's open question), and the sixth frustum plane's restraint —
        // `verify-gates` proves the same plane can *fail* by pulling it inside
        // the playfield. Judged like `mesh`: flat colours, but every contact
        // edge carries the shadow kernel's texels.
        policy: Policy {
            tolerance: 3,
            max_diff_pixels: 256,
            benign_delta: 6,
            max_dssim: 0.03,
            max_bias: 0.25,
        },
        render: render_platformer,
    },
    Scene {
        name: "chart",
        // §6 M24's material chart under the environment that lights it — demo
        // 12's own room, its own `chart()` and its own lights, framed so that
        // sky, floor and all five metallic rows are in one picture.
        //
        // Twenty-five *spheres* since §6 M26, which is what makes it a gate on
        // the BRDF rather than on three flat normals: a box shows one point of
        // the specular lobe per face, so a roughness that drifted moved a few
        // face tones and a curved highlight that collapsed moved nothing at
        // all. The grid also gates `unit_sphere`'s winding — a sphere wound
        // inside out is a hole, not a subtle delta — and the normal transform
        // beside it, which was a rotation alone while only boxes existed.
        //
        // It is the roster's only image-based-lighting subject, and what it
        // guards exists nowhere else: the spherical-harmonic irradiance (a
        // wrong band factor or a transposed basis is a room lit from the wrong
        // side), the split-sum specular (a metal row that went black is a
        // Fresnel term lost), the skybox's own gradient *and* its agreement
        // with what the metals reflect, and the depth test that masks it —
        // a skybox drawn over the scene rather than behind it is the most
        // recognizable way to get that pass wrong and it cannot survive here.
        //
        // Judged like `atrium`: smooth shading gradients across every face,
        // where a per-channel budget alone would forgive a curve that moved.
        policy: Policy {
            tolerance: 3,
            max_diff_pixels: 256,
            benign_delta: 6,
            max_dssim: 0.03,
            max_bias: 0.25,
        },
        render: render_chart,
    },
    Scene {
        name: "shooter",
        // §6 M37's room from inside it — the roster's only first-person
        // subject, and the first picture of the scene the `ao` and `bounce`
        // instruments have been measuring all along.
        //
        // What it guards that `chart` beside it cannot: `chart` frames the
        // material grid from four metres square on, which is a studio shot. This
        // is the eye at [`START`](demo_12_shooter::START) with the player's own
        // lift and pose, so the picture is the whole room down its long axis —
        // the two casting lamps at gameplay distance rather than filling the
        // frame, the shelter's bounded environment (§6 M28) beside the open sky
        // instead of alone, and the twelve target spots the course deals from,
        // which is the only reference that would notice a spot moving inside a
        // wall.
        //
        // No HUD: demo 12's overlay is built inside `present`, behind the ABI
        // this binary cannot reach, and restating the crosshair's rects here
        // would be the second table §4.10 forbids — the overlay path is already
        // gated by `ui-overlay`, `ui-text` and `tetris`.
        //
        // Judged like `chart`: smooth gradients across every face, plus a
        // shadow kernel on every contact edge.
        policy: Policy {
            tolerance: 3,
            max_diff_pixels: 256,
            benign_delta: 6,
            max_dssim: 0.03,
            max_bias: 0.25,
        },
        render: render_shooter,
    },
    Scene {
        name: "orbit",
        // §6 M38's map at the moment the transfer is handed to the star — the
        // roster's only subject whose positions are astronomical, and the only
        // picture of §1.4's membrane doing the thing it was built at M5 to do.
        //
        // What it guards that no neighbour can: 195 balls whose half-extents
        // span four orders of magnitude in one frame, positioned from `f64`
        // metres up to 2.3e11 and narrowed only at the camera-relative seam. A
        // ring is 64 samples of one conic, so a `state_at` that drifted is a
        // scatter rather than a shade, and the two planet rings plus the
        // transfer between them are the same closed form asked three questions.
        //
        // Judged like `mesh`: a lit sphere is a gradient, and 195 of them
        // against a dark field is exactly where a per-channel budget alone
        // would forgive a light that moved.
        policy: Policy {
            tolerance: 3,
            max_diff_pixels: 256,
            benign_delta: 6,
            max_dssim: 0.03,
            max_bias: 0.25,
        },
        render: render_orbit,
    },
];

/// Demo 12's room with §6 M24's material chart standing in the middle of it.
///
/// The world is dealt from the demo's own tables — [`ROOM`](demo_12_shooter::ROOM),
/// [`chart`](demo_12_shooter::chart), the sun, the two lamps and the `Sky` — rather
/// than from `bootstrap`, which is a system behind the ABI this binary cannot
/// reach (demo 04's `gg_game!` holds the `extern "C"` names). Same data, so an
/// edit to the chart's steps or the sky's intensity moves this reference; a
/// second copy of the *numbers* is what §4.10 forbids and there is none.
///
/// The eye is the harness's, and deliberately not the player's: `START` faces
/// the stairs, and a reference for image-based lighting has to hold the sky, the
/// floor and all twenty-five samples at once.
/// A row of spheres lit only by the compiled environment (§6 M27).
///
/// Deliberately austere: five balls, one base colour, smoothness running from
/// mirror to matte, and **no light entity at all**. The atrium proves the chain
/// composes with everything else in a lit room; this proves it is the thing
/// doing the lighting, and separating the two is what makes a regression here
/// readable rather than a change in a frame with six contributions to it.
///
/// The balls are metal. A dielectric would show mostly its diffuse lobe, which
/// is the SH half of this feature and was already gated at §6 M24 — the half
/// that is new is the specular one, and a conductor has nothing else.
fn render_mirror() -> Render {
    use gg_ecs::World;
    use gg_ecs::boundary::{Renderable, Sky};
    use gg_math::sim;

    let extent = BOXES_EXTENT;
    let pack = atrium_pack()?;
    let mut world = World::new();
    world.register::<Renderable>()?;
    world.register::<Sky>()?;
    let sky = world.spawn();
    world.insert(
        sky,
        Sky::image(demo_06_lit::SKY, demo_06_lit::SKY_INTENSITY),
    )?;
    // Five, across the whole smoothness axis. Large and close: the chain is 256
    // texels around a whole sphere, so a ball has to fill a good part of the
    // frame before its reflection has texels to resolve — a small mirror is a
    // correct reflection nobody can read.
    for i in 0..5 {
        let smoothness = 1.0 - i as f32 * 0.22;
        let ball = world.spawn();
        world.insert(
            ball,
            Renderable::ball(
                sim::DVec3::new(-2.0 + f64::from(i) * 1.0, 0.0, 0.0),
                0.45,
                0x00d8_d8dc,
            )
            .surfaced(smoothness, 1.0),
        )?;
    }

    let mut renderer = gg_render::OffscreenRenderer::new(extent)?;
    tracing::info!(device = %renderer.device().chosen, "offscreen device");
    renderer.open_pack(&pack)?;
    let view = gg_render::View::default();
    let eye = sim::DVec3::new(0.0, 0.0, 3.4);
    let mut extracted = gg_extract::Extracted::default();
    for _ in 0..HALL_FRAMES {
        extracted.clear(eye, view.frustum(extent));
        extracted.append::<Renderable>(&world)?;
        extracted.append_lights(&world)?;
        let _capture = gg_debug::capture::frame();
        renderer.frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])?;
    }
    // After the stream, because a probe renders whatever is resident and a
    // field gathered over a half-loaded pack is a field of the fallback (§6 M36).
    let frame = gathered(&mut renderer, |r| {
        let _capture = gg_debug::capture::frame();
        Ok(r.frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])?)
    })?;
    let pending = renderer
        .pack()
        .map_or(0, gg_render::content::Content::pending);
    anyhow::ensure!(
        pending == 0,
        "the environment was still streaming after {HALL_FRAMES} frames ({pending} pending) — a          half-resident chain is a reference of the fallback, not of the feature"
    );
    // No light casts, so no cascade is fitted and no shadow pass runs. Asserted
    // rather than assumed: a stray directional light would light these spheres
    // directly and quietly turn this into a second atrium.
    anyhow::ensure!(
        !frame.order.iter().any(|name| name.starts_with("shadow")),
        "nothing in this scene casts, so no shadow pass may run: {:?}",
        frame.order
    );
    ensure_clean(&renderer.shutdown())?;
    Ok(Capture {
        pixels: frame.pixels,
        extent,
        graph: frame.dump,
    })
}

/// Two environments and the seam between them (§6 M28).
///
/// The one scene in the roster with more than one sky, and it is built so the
/// failure modes separate. A **panorama** bounded to the left half of the frame,
/// a strongly tinted **gradient** unbounded behind everything, and a fade band
/// down the middle: the balls are metal, so what they show is the specular half
/// — a compiled chain on the left, a three-band convolution on the right, and
/// both at once in the band. The slab under them is chalk, so what it shows is
/// the diffuse half, which is where a blend that popped rather than faded would
/// be a line across the floor nobody could miss.
///
/// No lights at all, `mirror`'s reason: every photon in the frame came through
/// an environment, so a selection that broke renders black or renders one sky
/// everywhere, and neither is a small difference.
fn render_volumes() -> Render {
    use gg_ecs::World;
    use gg_ecs::boundary::{Renderable, Sky};
    use gg_math::sim;

    let extent = BOXES_EXTENT;
    let pack = atrium_pack()?;
    let mut world = World::new();
    world.register::<Renderable>()?;
    world.register::<Sky>()?;

    // The room, bounded to x < -0.5 with a 2 m fade — wide enough that the band
    // covers several balls, because a seam narrower than a ball is a seam this
    // frame cannot show.
    let room = world.spawn();
    world.insert(
        room,
        Sky::image(demo_06_lit::SKY, demo_06_lit::SKY_INTENSITY).within(
            sim::DVec3::new(-3.5, 0.0, 0.0),
            sim::Vec3::new(3.0, 4.0, 4.0),
            2.0,
        ),
    )?;
    // The outdoors: a gradient, and deliberately nothing like the room. Its
    // colours are not `daylight`'s — a blend is only legible against a sky the
    // other one could not be mistaken for.
    let outside = world.spawn();
    world.insert(
        outside,
        Sky {
            zenith: 0x00ff_7a2a,
            horizon: 0x00ff_c060,
            ground: 0x0060_2010,
            ..Sky::daylight(0.35)
        },
    )?;

    // Metal, and smooth: the specular half is the one that reads the chain.
    for i in 0..7 {
        let ball = world.spawn();
        world.insert(
            ball,
            Renderable::ball(
                sim::DVec3::new(-3.0 + f64::from(i) * 1.0, 0.35, 0.0),
                0.42,
                0x00d8_d8dc,
            )
            .surfaced(0.85, 1.0),
        )?;
    }
    // The floor, chalk — the diffuse half, and the only thing in frame wide
    // enough to show the fade as a gradient rather than as an edge.
    let floor = world.spawn();
    world.insert(
        floor,
        Renderable::boxed(
            sim::DVec3::new(0.0, -0.4, 0.0),
            sim::Vec3::new(6.0, 0.1, 3.0),
            0x00c8_c4bc,
        ),
    )?;

    let mut renderer = gg_render::OffscreenRenderer::new(extent)?;
    tracing::info!(device = %renderer.device().chosen, "offscreen device");
    renderer.open_pack(&pack)?;
    let view = gg_render::View::default();
    let eye = sim::DVec3::new(0.0, 1.2, 5.2);
    let mut extracted = gg_extract::Extracted::default();
    for _ in 0..HALL_FRAMES {
        extracted.clear(eye, view.frustum(extent));
        extracted.append::<Renderable>(&world)?;
        extracted.append_lights(&world)?;
        // The order the shader composites in, asserted where it is decided: the
        // bounded room first, the unbounded sky last. Reversed, every fragment
        // would take the world's sky and the room would never be reached.
        anyhow::ensure!(
            extracted.skies.len() == 2
                && !extracted.skies[0].unbounded()
                && extracted.skies[1].unbounded(),
            "the room must composite before the world it stands in: {:?}",
            extracted.skies
        );
        let _capture = gg_debug::capture::frame();
        renderer.frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])?;
    }
    // After the stream, because a probe renders whatever is resident and a
    // field gathered over a half-loaded pack is a field of the fallback (§6 M36).
    let frame = gathered(&mut renderer, |r| {
        let _capture = gg_debug::capture::frame();
        Ok(r.frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])?)
    })?;
    let pending = renderer
        .pack()
        .map_or(0, gg_render::content::Content::pending);
    anyhow::ensure!(
        pending == 0,
        "the environment was still streaming after {HALL_FRAMES} frames ({pending} pending)"
    );
    anyhow::ensure!(
        !frame.order.iter().any(|name| name.starts_with("shadow")),
        "nothing in this scene casts, so no shadow pass may run: {:?}",
        frame.order
    );
    ensure_clean(&renderer.shutdown())?;
    Ok(Capture {
        pixels: frame.pixels,
        extent,
        graph: frame.dump,
    })
}

/// An environment that is a *place* rather than a direction (§6 M29).
///
/// One sky, bounded, with the camera well inside it — so no blending happens and
/// the only thing this frame can be about is the parallax correction. Five
/// mirror balls spread across the room and a smooth metal floor under them:
/// each ball is a different distance from each wall, so with the correction the
/// bright window strips land at a different place on every one and the floor
/// shows the walls converging. Without it, an environment is infinitely far
/// away, every ball reflects the same directions of the same picture, and the
/// five come out near-identical — which is what the reference refuses.
///
/// No lights, `mirror`'s reason: every photon here came through the chain.
fn render_parallax() -> Render {
    use gg_ecs::World;
    use gg_ecs::boundary::{Renderable, Sky};
    use gg_math::sim;

    let extent = BOXES_EXTENT;
    let pack = atrium_pack()?;
    let mut world = World::new();
    world.register::<Renderable>()?;
    world.register::<Sky>()?;

    // The room, and the camera stands in it. Wider than it is tall, so the
    // near and far walls are at different distances from every ball — a cube
    // would make the correction symmetric and half of it invisible.
    const CENTRE: sim::DVec3 = sim::DVec3::new(0.0, 1.0, 0.0);
    const HALF: sim::Vec3 = sim::Vec3::new(6.0, 3.0, 7.0);
    let room = world.spawn();
    world.insert(
        room,
        Sky::image(demo_06_lit::SKY, demo_06_lit::SKY_INTENSITY).within(CENTRE, HALF, 1.0),
    )?;

    // Mirror-grade, which is the roughness the correction is most legible at:
    // a lobe wide enough to blur the windows would blur the parallax with them.
    for i in 0..5 {
        let ball = world.spawn();
        world.insert(
            ball,
            Renderable::ball(
                sim::DVec3::new(-4.0 + f64::from(i) * 2.0, 0.55, 0.0),
                0.55,
                0x00d8_d8dc,
            )
            .surfaced(0.96, 1.0),
        )?;
    }
    // Smooth metal rather than the chalk `volumes` uses: a floor that reflects
    // is the widest surface in the frame, and a correction that is wrong shows
    // there as walls that do not meet the ones behind them.
    let floor = world.spawn();
    world.insert(
        floor,
        Renderable::boxed(
            sim::DVec3::new(0.0, -0.1, 0.0),
            sim::Vec3::new(7.0, 0.1, 5.0),
            0x0090_9498,
        )
        .surfaced(0.88, 1.0),
    )?;

    let mut renderer = gg_render::OffscreenRenderer::new(extent)?;
    tracing::info!(device = %renderer.device().chosen, "offscreen device");
    renderer.open_pack(&pack)?;
    let view = gg_render::View::default();
    let eye = sim::DVec3::new(0.0, 1.4, 6.0);
    let mut extracted = gg_extract::Extracted::default();
    for _ in 0..HALL_FRAMES {
        extracted.clear(eye, view.frustum(extent));
        extracted.append::<Renderable>(&world)?;
        extracted.append_lights(&world)?;
        // What makes this scene about the correction and nothing else: one sky,
        // bounded, and the camera fully inside it. A camera that drifted into
        // the fade would put a blend in the picture and this reference would
        // start grading two features at once.
        anyhow::ensure!(
            extracted.skies.len() == 1
                && !extracted.skies[0].unbounded()
                && extracted.skies[0].weight_at(gg_math::render::Vec3::ZERO) == 1.0,
            "one bounded sky, and the camera inside it: {:?}",
            extracted.skies
        );
        let _capture = gg_debug::capture::frame();
        renderer.frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])?;
    }
    // After the stream, because a probe renders whatever is resident and a
    // field gathered over a half-loaded pack is a field of the fallback (§6 M36).
    let frame = gathered(&mut renderer, |r| {
        let _capture = gg_debug::capture::frame();
        Ok(r.frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])?)
    })?;
    let pending = renderer
        .pack()
        .map_or(0, gg_render::content::Content::pending);
    anyhow::ensure!(
        pending == 0,
        "the environment was still streaming after {HALL_FRAMES} frames ({pending} pending)"
    );
    anyhow::ensure!(
        !frame.order.iter().any(|name| name.starts_with("shadow")),
        "nothing in this scene casts, so no shadow pass may run: {:?}",
        frame.order
    );
    ensure_clean(&renderer.shutdown())?;
    Ok(Capture {
        pixels: frame.pixels,
        extent,
        graph: frame.dump,
    })
}

/// A hall of lanterns — more lights than the engine could carry before §6 M30.
///
/// Ninety-six of them, alternating down two walls, each with a range short
/// enough that it lights its own bay and not the next one. That is the property
/// the frame is for: a fragment in the far bay must be shaded by the two or
/// three lamps that reach it and by none of the ninety-three that do not, and
/// the reference is what "and by none of the others" looks like. Assignment that
/// under-includes puts a straight edge through a pool of light where a froxel
/// boundary runs; assignment that over-includes costs time and changes no pixel,
/// which is why this scene grades correctness and `gg-tools lights` grades cost.
///
/// The sun is deliberately absent and the ambient is low: a lit floor here has
/// to have been lit by a lamp.
fn render_lanterns() -> Render {
    use gg_ecs::World;
    use gg_ecs::boundary::{Light, Renderable};
    use gg_math::sim;

    let extent = BOXES_EXTENT;
    let mut world = World::new();
    world.register::<Renderable>()?;
    world.register::<Light>()?;

    /// Three times the pre-M30 cap, and the number the assertions below are
    /// written against — a scene that quietly lost lamps would otherwise still
    /// render something plausible.
    const LANTERNS: usize = 96;
    /// Metres a lamp reaches zero at. Long enough to cross the hall and short
    /// against its length, which is the arrangement that puts the selectivity on
    /// the *depth* axis — where a screen tile cannot help and a froxel can.
    const REACH: f32 = 3.0;

    let floor = world.spawn();
    world.insert(
        floor,
        Renderable::boxed(
            sim::DVec3::new(0.0, -0.1, -30.0),
            sim::Vec3::new(3.3, 0.1, 40.0),
            0x0086_8a8e,
        )
        .surfaced(0.7, 0.0),
    )?;
    for side in [-1.0, 1.0] {
        let wall = world.spawn();
        world.insert(
            wall,
            Renderable::boxed(
                sim::DVec3::new(side * 2.9, 2.2, -30.0),
                sim::Vec3::new(0.2, 2.4, 40.0),
                0x008e_8a82,
            )
            .surfaced(0.8, 0.0),
        )?;
        // A rail down each wall at lamp height, so a lamp has something near it
        // to fall off across — a falloff read on a distant floor is a gradient,
        // and a falloff read on the surface beside it is a shape.
        let rail = world.spawn();
        world.insert(
            rail,
            Renderable::boxed(
                sim::DVec3::new(side * 2.3, 0.7, -30.0),
                sim::Vec3::new(0.15, 0.15, 40.0),
                0x00b0_a898,
            )
            .surfaced(0.6, 0.0),
        )?;
    }
    for i in 0..LANTERNS {
        let side = if i.is_multiple_of(2) { -1.0 } else { 1.0 };
        let lamp = world.spawn();
        world.insert(
            lamp,
            Light::point(
                sim::DVec3::new(side * 2.5, 1.35, -1.5 - f64::from(i as u32 / 2) * 1.2),
                // Three colours down the hall, so a lamp assigned to the wrong
                // froxel is the wrong *hue* there and not merely the wrong
                // brightness — a difference a per-channel gate sees at once.
                match i % 3 {
                    0 => 0x00ff_c890,
                    1 => 0x0090_c8ff,
                    _ => 0x00c0_ffa0,
                },
                7.0,
                REACH,
            ),
        )?;
    }

    let mut renderer = gg_render::OffscreenRenderer::new(extent)?;
    tracing::info!(device = %renderer.device().chosen, "offscreen device");
    let view = gg_render::View::default();
    let eye = sim::DVec3::new(0.0, 1.5, 4.0);
    let mut extracted = gg_extract::Extracted::default();
    extracted.clear(eye, view.frustum(extent));
    extracted.append::<Renderable>(&world)?;
    extracted.append_lights(&world)?;
    // The scene's own claim, checked before the picture is: every lamp survived
    // the frustum and the cap. Without this the reference could be blessed from
    // a frame that had quietly dropped sixty of them and would then pass forever.
    anyhow::ensure!(
        extracted.lights.len() == LANTERNS && extracted.lights_dropped == 0,
        "{} of {LANTERNS} lamps reached the frame ({} dropped)",
        extracted.lights.len(),
        extracted.lights_dropped
    );
    let frame = gathered(&mut renderer, |r| {
        let _capture = gg_debug::capture::frame();
        Ok(r.frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])?)
    })?;
    // And the claim that makes it a §6 M30 scene rather than a scene with a lot
    // of lights in it: the busiest froxel holds a small fraction of them, so the
    // picture was produced by *selecting* and not by looping the frame.
    let load = renderer.cluster_load();
    // On the *total* rather than on the worst froxel, which is the number that
    // says how much of the frame's loop was removed. The worst froxel here is a
    // far one — log slices are thick at depth, so the froxel around the
    // vanishing point holds a third of the hall — and bounding it would be
    // bounding the distribution rather than the selection.
    anyhow::ensure!(
        load.dropped == 0
            && load.pairs * 40 < load.froxels * LANTERNS
            && load.worst * 2 < LANTERNS as u32,
        "the grid is not selecting here: {load:?} of {LANTERNS}"
    );
    tracing::info!(worst = load.worst, pairs = load.pairs, "froxel occupancy");
    // No *sun* — nothing here is directional, so no cascade may be fitted — but
    // since §6 M31 the lamps themselves cast, and the atlas pass having run is
    // what makes the rail shadows across this floor the engine's answer rather
    // than a gradient that happens to darken there.
    anyhow::ensure!(
        !frame.order.iter().any(|name| name.starts_with("shadow"))
            && frame.order.iter().any(|name| name == "lamp-shadows"),
        "this scene has no sun and four casting lamps: {:?}",
        frame.order
    );
    ensure_clean(&renderer.shutdown())?;
    Ok(Capture {
        pixels: frame.pixels,
        extent,
        graph: frame.dump,
    })
}

/// §6 M31: one lamp, three pillars, and shadows that diverge.
fn render_lampshade() -> Render {
    use gg_ecs::World;
    use gg_ecs::boundary::{Light, Renderable};
    use gg_math::sim;

    let extent = BOXES_EXTENT;
    let mut world = World::new();
    world.register::<Renderable>()?;
    world.register::<Light>()?;

    /// Where the lamp hangs. **Below the pillars' tops**, which is what makes
    /// each shadow run outward along the floor and widen as it goes — a lamp
    /// above them would throw three short stubs and grade almost nothing.
    const LAMP_HEIGHT: f64 = 1.15;
    /// Half-height of a pillar; its top is twice this.
    const PILLAR: f64 = 0.9;
    /// Three, at three unrelated bearings from the lamp. Not a ring: equal
    /// spacing about a central light is a symmetry a mirrored lookup would
    /// survive, and §6 M30's grid shipped mirrored.
    const PILLARS: [(f64, f64); 3] = [(-1.9, -0.5), (1.5, -1.8), (2.0, 1.3)];

    let floor = world.spawn();
    world.insert(
        floor,
        Renderable::boxed(
            sim::DVec3::new(0.0, -0.1, 0.0),
            sim::Vec3::new(7.0, 0.1, 7.0),
            0x009a_9a96,
        )
        .surfaced(0.75, 0.0),
    )?;
    // A back wall, so the frame holds a *vertical* surface too: a shadow's
    // shape on the floor and its shape climbing a wall fail differently, and a
    // face-selection bug shows on the second while the first still looks right.
    let wall = world.spawn();
    world.insert(
        wall,
        Renderable::boxed(
            sim::DVec3::new(0.0, 1.6, -4.2),
            sim::Vec3::new(7.0, 1.7, 0.15),
            0x008e_9298,
        )
        .surfaced(0.8, 0.0),
    )?;
    for (x, z) in PILLARS {
        let pillar = world.spawn();
        world.insert(
            pillar,
            Renderable::boxed(
                sim::DVec3::new(x, PILLAR, z),
                sim::Vec3::new(0.22, PILLAR as f32, 0.22),
                0x00b0_a898,
            )
            .surfaced(0.6, 0.0),
        )?;
    }
    // Off centre in both axes, for `PILLARS`' reason.
    let lamp = world.spawn();
    world.insert(
        lamp,
        Light::point(
            sim::DVec3::new(0.35, LAMP_HEIGHT, 0.2),
            0x00ff_f0d8,
            26.0,
            14.0,
        ),
    )?;

    let mut renderer = gg_render::OffscreenRenderer::new(extent)?;
    tracing::info!(device = %renderer.device().chosen, "offscreen device");
    let view = gg_render::View {
        pitch: -0.42,
        ..gg_render::View::default()
    };
    let eye = sim::DVec3::new(0.0, 3.4, 5.6);
    let mut extracted = gg_extract::Extracted::default();
    extracted.clear(eye, view.frustum(extent));
    extracted.append::<Renderable>(&world)?;
    extracted.append_lights(&world)?;
    anyhow::ensure!(
        extracted.lights.len() == 1 && extracted.lights_dropped == 0,
        "the one lamp this scene is about did not reach the frame: {} light(s), {} dropped",
        extracted.lights.len(),
        extracted.lights_dropped
    );
    let frame = gathered(&mut renderer, |r| {
        let _capture = gg_debug::capture::frame();
        Ok(r.frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])?)
    })?;
    // The scene's own claim: every shadow here is a *lamp's*. No directional
    // light exists, so a cascade pass running at all would mean the reference
    // was blessed from a frame whose shadows came from somewhere else.
    anyhow::ensure!(
        frame.order.iter().any(|name| name == "lamp-shadows")
            && !frame.order.iter().any(|name| name.starts_with("shadow")),
        "this scene's shadows must be the lamp's and only the lamp's: {:?}",
        frame.order
    );
    ensure_clean(&renderer.shutdown())?;
    Ok(Capture {
        pixels: frame.pixels,
        extent,
        graph: frame.dump,
    })
}

/// Demo 12's room seen from where the player spawns in it (§6 M37).
///
/// Every entity is the demo's own data — [`ROOM`](demo_12_shooter::ROOM),
/// [`chart`](demo_12_shooter::chart), [`SPOTS`](demo_12_shooter::SPOTS), the
/// sun, the lamps, the sky and [`shelter_sky`](demo_12_shooter::shelter_sky) —
/// dealt here rather than by `bootstrap`, which is a system behind the ABI this
/// binary cannot reach. Same tables, so a slab that moves or a spot that is
/// re-placed moves this reference instead of diverging from it.
///
/// The course is dealt **whole**: a target at every spot, not the three a round
/// holds. Which three is a draw off `Range::rng` and a reference must not
/// depend on one, and the full table is also the only thing that would catch a
/// spot re-placed inside a wall.
///
/// The eye is the *player's*, unlike `chart`'s: `START` with the body's own
/// [`EYE_LIFT`](demo_12_shooter::EYE_LIFT) and the pose
/// [`at_start`](demo_12_shooter::START) opens on — level, facing the stairs.
/// That is what makes this the first-person subject rather than a second studio
/// shot of the same room.
fn render_shooter() -> Render {
    use demo_12_shooter as shooter;
    use gg_ecs::World;
    use gg_ecs::boundary::{Light, Renderable, Sky};

    let extent = BOXES_EXTENT;
    let mut world = World::new();
    world.register::<Renderable>()?;
    world.register::<Light>()?;
    world.register::<Sky>()?;
    for (position, half_extent, color) in shooter::ROOM {
        let slab = world.spawn();
        world.insert(slab, Renderable::boxed(*position, *half_extent, *color))?;
    }
    for (at, smoothness, metallic) in shooter::chart() {
        let ball = world.spawn();
        world.insert(
            ball,
            Renderable::ball(at, shooter::CHART_RADIUS, shooter::CHART_INK)
                .surfaced(smoothness, metallic),
        )?;
    }
    for at in shooter::SPOTS {
        let target = world.spawn();
        world.insert(
            target,
            Renderable::boxed(
                *at,
                gg_math::sim::Vec3::splat(shooter::TARGET_HALF),
                shooter::TARGET_INK,
            )
            .surfaced(shooter::TARGET_SMOOTHNESS, 0.0),
        )?;
    }
    let sun = world.spawn();
    world.insert(
        sun,
        Light::sun(shooter::SUN, shooter::SUN_INK, shooter::SUN_INTENSITY),
    )?;
    for at in shooter::LAMPS {
        let lamp = world.spawn();
        world.insert(
            lamp,
            Light::point(
                at,
                shooter::LAMP_INK,
                shooter::LAMP_INTENSITY,
                shooter::LAMP_RANGE,
            ),
        )?;
    }
    let sky = world.spawn();
    world.insert(sky, Sky::daylight(shooter::SKY_INTENSITY))?;
    let shelter = world.spawn();
    world.insert(shelter, shooter::shelter_sky())?;

    let eye = gg_math::sim::DVec3::new(
        shooter::START.x,
        shooter::START.y + shooter::EYE_LIFT,
        shooter::START.z,
    );
    // Yaw and pitch zero, which is `at_start`'s pose: level, down -z, into the
    // room. `View::default()` is already that camera, so nothing is overridden.
    let view = gg_render::View::default();
    let mut extracted = gg_extract::Extracted::default();
    extracted.clear(eye, view.frustum(extent));
    extracted.append::<Renderable>(&world)?;
    extracted.append_lights(&world)?;
    extracted.cast_shadows(view.caster_reach(extent));
    let mut renderer = gg_render::OffscreenRenderer::new(extent)?;
    tracing::info!(
        device = %renderer.device().chosen,
        culled = extracted.culled,
        "offscreen device"
    );
    let frame = gathered(&mut renderer, |r| {
        let _capture = gg_debug::capture::frame();
        Ok(r.frame(&extracted, &view, [0.02, 0.02, 0.03, 1.0], &[])?)
    })?;
    ensure_clean(&renderer.shutdown())?;
    Ok(Capture {
        pixels: frame.pixels,
        extent,
        graph: frame.dump,
    })
}

fn render_chart() -> Render {
    use demo_12_shooter as shooter;
    use gg_ecs::World;
    use gg_ecs::boundary::{Light, Renderable, Sky};

    let extent = BOXES_EXTENT;
    let mut world = World::new();
    world.register::<Renderable>()?;
    world.register::<Light>()?;
    world.register::<Sky>()?;
    for (position, half_extent, color) in shooter::ROOM {
        let slab = world.spawn();
        world.insert(slab, Renderable::boxed(*position, *half_extent, *color))?;
    }
    for (at, smoothness, metallic) in shooter::chart() {
        let ball = world.spawn();
        world.insert(
            ball,
            Renderable::ball(at, shooter::CHART_RADIUS, shooter::CHART_INK)
                .surfaced(smoothness, metallic),
        )?;
    }
    let sun = world.spawn();
    world.insert(
        sun,
        Light::sun(shooter::SUN, shooter::SUN_INK, shooter::SUN_INTENSITY),
    )?;
    for at in shooter::LAMPS {
        let lamp = world.spawn();
        world.insert(
            lamp,
            Light::point(
                at,
                shooter::LAMP_INK,
                shooter::LAMP_INTENSITY,
                shooter::LAMP_RANGE,
            ),
        )?;
    }
    let sky = world.spawn();
    world.insert(sky, Sky::daylight(shooter::SKY_INTENSITY))?;

    // Square on to the grid at its own height, four metres out (§6 M26): at the
    // default 1.0 rad that frames 4.4 m of a 3.4 m chart, so every ball is in
    // shot with margin and none is at the edge where a wide lens would stretch
    // its highlight. Level rather than pitched — the horizon and the middle
    // sample land on the same line, and the frame still holds floor below and
    // sky above the 4 m wall, which is what keeps the skybox gated.
    let eye = gg_math::sim::DVec3::new(0.0, shooter::CHART_CENTRE.y, shooter::CHART_CENTRE.z + 4.0);
    let view = gg_render::View::default();
    let mut extracted = gg_extract::Extracted::default();
    extracted.clear(eye, view.frustum(extent));
    extracted.append::<Renderable>(&world)?;
    extracted.append_lights(&world)?;
    extracted.cast_shadows(view.caster_reach(extent));
    let mut renderer = gg_render::OffscreenRenderer::new(extent)?;
    tracing::info!(
        device = %renderer.device().chosen,
        culled = extracted.culled,
        "offscreen device"
    );
    let frame = gathered(&mut renderer, |r| {
        let _capture = gg_debug::capture::frame();
        Ok(r.frame(&extracted, &view, [0.02, 0.02, 0.03, 1.0], &[])?)
    })?;
    ensure_clean(&renderer.shutdown())?;
    Ok(Capture {
        pixels: frame.pixels,
        extent,
        graph: frame.dump,
    })
}

/// Render demo 02's scene — the same buffers, the same upload path through
/// the transfer queue, and the same bindless texture index the demo draws with
/// (§4.10: the golden guards the demo, not a lookalike). Tick 0 is the frozen
/// pose; the mesh's rotation is a pure function of it (§2, Sim time row).
fn render_mesh() -> Render {
    render_mesh_from(gg_math::sim::DVec3::new(0.0, 0.0, 0.0))
}

/// The same scene with the whole world — camera and cube together — translated
/// to [`demo_02_mesh::sim::FAR_ORIGIN`]. If subtract-then-narrow works, this is
/// the same picture; if anything narrows before the subtraction, it is a mess.
fn render_mesh_far() -> Render {
    render_mesh_from(demo_02_mesh::sim::FAR_ORIGIN)
}

fn render_mesh_from(origin: gg_math::sim::DVec3) -> Render {
    render_mesh_of(demo_02_mesh::sim::Sim::new_at(0, origin)?)
}

/// §4.10's replay-driven playback: the curated determinism replay (§5.6) drives
/// demo 02's own sim to [`REPLAY_TICK`], and *that* frame is the reference.
///
/// Every other mesh scene renders tick 0 — a pose the sim reaches by doing
/// nothing, which proves the draw and not the loop feeding it. This one has
/// flown, strafed, turned and spawned first, so the frame answers for the action
/// map, the fixed-point axes and the spawn order as well as for the pixels. The
/// replay is the same file the hash gate replays, so a divergence shows up as
/// both a wrong hash and a wrong picture.
fn render_mesh_replay() -> Render {
    let path = demo_02_mesh::gate::replay_path(demo_02_mesh::gate::CURATED);
    let replay = gg_input::Replay::decode(&std::fs::read(&path)?)?;
    let (sim, _) = demo_02_mesh::sim::run(&replay, REPLAY_TICK, None)?;
    render_mesh_of(sim)
}

/// Deep enough into the curated script to have flown, strafed, turned *and*
/// spawned (§5.6's phases are 100 ticks each), shallow enough to leave the tail
/// of the replay as headroom for a longer capture later.
const REPLAY_TICK: u64 = 330;

/// The chaos seed demo 02's own churn assertion witnesses with. Sharing it means
/// the seed proven to actually move the world is the seed with a picture.
const CHAOS_WITNESS: u64 = 8;

fn render_chaos_witness() -> Render {
    render_chaos(CHAOS_WITNESS)
}

/// A chaos stream's terminal frame (§5.11 + §4.10). The generator is the gate's
/// own, so the world this draws is the world the hash baseline checkpointed —
/// a divergence is visible here as a misplaced cube rather than only as a
/// different number.
fn render_chaos(seed: u64) -> Render {
    let replay = demo_02_mesh::gate::chaos_replay(seed, demo_02_mesh::gate::CHAOS_TICKS);
    let (sim, _) = demo_02_mesh::sim::run(&replay, demo_02_mesh::gate::CHAOS_TICKS, None)?;
    render_mesh_of(sim)
}

/// `gg-golden chaos [seed]` — render a chaos seed's terminal frame beside the
/// build products. Diagnosis, not a gate: the gated seed is the `chaos-witness`
/// scene, and this is how the other seven get a picture when one of them is the
/// one that diverged.
fn chaos(filter: Option<&str>) -> anyhow::Result<()> {
    let seeds: Vec<u64> = match filter {
        Some(arg) => vec![arg.parse()?],
        None => demo_02_mesh::gate::CHAOS_SEEDS.to_vec(),
    };
    for seed in seeds {
        let capture = render_chaos(seed)?;
        let path = artifacts_root().join(format!("chaos-{seed}.png"));
        png_io::write(&path, &capture.pixels, capture.extent)?;
        println!("gg-golden: chaos seed {seed} → {}", path.display());
    }
    Ok(())
}

fn render_mesh_of(sim: demo_02_mesh::sim::Sim) -> Render {
    let extent = demo_02_mesh::GOLDEN_EXTENT;
    let mut rhi = OffscreenRhi::new(extent)?;
    tracing::info!(
        device = %rhi.device_report().chosen,
        transfer_crosses_families = rhi.transfer_crosses_queue_families(),
        "offscreen device"
    );
    let scene = demo_02_mesh::upload(&mut rhi)?;
    // The demo's own extract stage: the golden guards the whole path from ECS
    // state to push constants, not just the draw at the end of it.
    let mut extracted = gg_extract::Extracted::default();
    let camera = demo_02_mesh::extract(&sim, &mut extracted)?;
    anyhow::ensure!(
        !extracted.instances.is_empty(),
        "demo 02's sim extracted no cube at tick {}",
        sim.tick_count()
    );
    // One draw per cube, exactly as the demo does it — the pushes outlive the
    // `DrawSpec`s that borrow them.
    let pushes: Vec<_> = extracted
        .instances
        .iter()
        .map(|instance| demo_02_mesh::push_for(&camera, extent, instance, &scene))
        .collect();
    let draws: Vec<gg_rhi::DrawSpec<'_>> = pushes
        .iter()
        .map(|push| gg_rhi::DrawSpec {
            pipeline: scene.pipeline,
            push_constants: bytemuck::bytes_of(push),
            count: scene.index_count,
            index_buffer: Some(scene.indices),
            indirect: None,
            depth_bias: None,
            viewport: None,
        })
        .collect();

    let dest = readback_buffer(&mut rhi, extent)?;
    let mut transients = Transients::default();
    let mut frame = transients.frame(&mut rhi, extent)?;
    let backbuffer = frame.backbuffer();
    let depth = frame.depth("scene.depth")?;
    let into = frame.readback_buffer("golden.readback", dest);
    let mut declared: Vec<Declared<'_>> = demo_02_mesh::declare(backbuffer, depth, &draws).into();
    declared.push(readback_pass(backbuffer, into));
    let compiled = frame.compile(&declared)?;
    let graph = compiled.dump();
    {
        // Scoped, so the capture closes before `shutdown` destroys the device it
        // was opened against (§4.8). Inert unless `capture` armed it.
        let _capture = gg_debug::capture::frame();
        rhi.execute(&compiled.passes())?;
    }
    let pixels = rhi.map_buffer(dest)?.to_vec();

    ensure_clean(&rhi.shutdown())?;
    Ok(Capture {
        pixels,
        extent,
        graph,
    })
}

/// The extent the v1 pass-list scenes capture at. Smaller than the demo scenes:
/// the box silhouettes are what these judge, and 640x360 of flat colour costs
/// reference bytes without adding evidence (§4.10's size budget).
const BOXES_EXTENT: (u32, u32) = (320, 180);

/// Three boxes, declared the way a game declares them: ordinary components in a
/// `World`, read back through the same typed query the shell's extract uses.
/// Demo 04's pack, as `cargo xtask assets` compiles it. Build output, never
/// checked in — so a missing one is a missing *step*, and says so.
const HALL_PACK: &str = "target/assets/04-scene.ggpack";
const FIELD_PACK: &str = "target/assets/05-many.ggpack";
const ATRIUM_PACK: &str = "target/assets/06-lit.ggpack";

/// §6 M9's exit row: a pack must be on the device within this. Measured from
/// `open` to the frame where nothing is pending, which is the span a player
/// spends looking at an empty room.
const LOAD_BUDGET: std::time::Duration = std::time::Duration::from_millis(500);

/// Frames the loader is given before the clock is called stopped. Generous
/// rather than tight: what is being measured is wall time, and a frame count
/// only bounds how long this is willing to wait for it.
const LOAD_FRAMES: usize = 4096;

/// `gg-golden boot [pack]` — what a cold start spends before the first picture
/// (§6 M25), broken down rather than totalled.
///
/// **Not a launch time, and the difference is named**: this process is already
/// running when the clock starts, so what it excludes is process creation, the
/// dynamic loader, and — because §1.5 forbids an automated tier a window —
/// window creation and the swapchain. What it *does* cover is the part that
/// dominates and the part we wrote: device bring-up, every pipeline, the pack,
/// and the first frame. A windowed launch is the manual measurement, the way
/// `bench --record` is.
///
/// The pack is optional because the two questions are different: without one
/// this is the engine's own floor, and with one it is a level's.
fn boot(pack: Option<&str>) -> anyhow::Result<()> {
    let started = std::time::Instant::now();
    let mut renderer = gg_render::OffscreenRenderer::new(bench::EXTENT)?;
    let mut extracted = gg_extract::Extracted::default();
    let mut frames = 0;

    let path = pack.map(std::path::PathBuf::from);
    if let Some(path) = &path {
        renderer.open_pack(path)?;
        let scenes: Vec<u64> = renderer
            .pack()
            .map(gg_render::content::Content::scene_ids)
            .unwrap_or_default();
        for (index, asset) in scenes.iter().enumerate() {
            extracted.models.push(gg_extract::Instance {
                entity: gg_ecs::Entity::from_bits(index as u64 + 1),
                offset: gg_math::render::Vec3::ZERO,
                rotation: gg_math::render::Quat::IDENTITY,
                half_extent: gg_math::render::Vec3::splat(1.0),
                color: 0x00ff_ffff,
                surface: (0.0, 0.0),
                // Unbounded, for `load`'s reason: an id culled before it streams
                // stops the clock by never asking.
                radius: f32::INFINITY,
                asset: *asset,
                // Unread on the model path: the mesh is the shape.
                shape: gg_extract::shape::BOX,
            });
        }
    }

    // Frames until the picture is complete: with a pack that means resident,
    // and without one the first frame *is* the boot.
    let wanted = path.is_some();
    loop {
        frames += 1;
        renderer.frame(
            &extracted,
            &gg_render::View::default(),
            [0.0, 0.0, 0.0, 1.0],
            &[],
        )?;
        let ready = !wanted
            || renderer
                .pack()
                .and_then(gg_render::content::Content::ready_at)
                .is_some();
        if ready || frames >= LOAD_FRAMES {
            break;
        }
    }
    let total = started.elapsed();

    let mut zones: Vec<bench::Zone> = Vec::new();
    bench::accumulate(&mut zones, &gg_core::zone::take());
    let report = renderer.shutdown();
    anyhow::ensure!(report.clean(), "unclean boot run: {report:?} (§4.3)");

    println!(
        "gg-golden boot — {} to first frame over {frames} frame(s)\n  pack {}",
        format_ms(total),
        path.as_ref()
            .map_or("(none)".into(), |p| p.display().to_string()),
    );
    if gg_core::zone::enabled() {
        // Divided by one: these are totals for a boot, not a per-frame mean, and
        // dividing a one-shot by the frames it took would report a fiction.
        println!("{}", bench::zone_table(&zones, 1));
    } else {
        println!("\n  no cpu zones — built without `cpu-timings` (§4.8)");
    }
    Ok(())
}

/// Milliseconds with enough places to see a sub-millisecond stage.
fn format_ms(d: std::time::Duration) -> String {
    format!("{:.3} ms", d.as_secs_f64() * 1e3)
}

/// `gg-golden load [pack]` — time a pack from mapped to fully resident (§4.6).
///
/// Windowless by linkage like everything else here, which is the only reason
/// this measurement is available to an automated tier at all: the shell is the
/// other thing that streams a pack, and the shell needs a window (§1.5).
///
/// It streams every asset the pack contains rather than only what a frame
/// shows, because the number the exit row asks about is a *level's* load and a
/// camera pointed at a wall would otherwise measure one mesh.
fn load(pack: Option<&str>) -> anyhow::Result<()> {
    let path = match pack {
        Some(path) => std::path::PathBuf::from(path),
        None => hall_pack()?,
    };
    let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    // What this harness times is **streaming**, which is why it renders at
    // 64x64 at all — the picture is not the subject and the frame's own cost is
    // meant to be negligible beside the upload. §6 M35's occlusion breaks that
    // assumption under one profile: it is two fullscreen passes of dependent
    // texture reads, and GPU-AV instruments every one of them, which took this
    // leg from 352 ms to 990 ms against a 500 ms budget while changing nothing
    // about how fast a pack arrives. Held off here rather than budgeted around,
    // because a load-time gate that moves when a *shading* term is added is
    // measuring the wrong thing. The instrumented coverage of those shaders is
    // not lost: the two `bench` legs above this one in `xtask gpuav` render the
    // same pass list with them on.
    gg_render::cvars::AO.set_bool(false);
    let mut renderer = gg_render::OffscreenRenderer::new((64, 64))?;
    tracing::info!(device = %renderer.device().chosen, "offscreen device");
    renderer.open_pack(&path)?;

    // Every scene in the file, named by id — the closest thing a pack has to
    // "load the level", and what a game's own `Model`s would add up to.
    let mut extracted = gg_extract::Extracted::default();
    let scenes: Vec<u64> = renderer
        .pack()
        .map(gg_render::content::Content::scene_ids)
        .unwrap_or_default();
    anyhow::ensure!(
        !scenes.is_empty(),
        "{} holds no scene — there is nothing here to time (§5.8: a check that finds nothing to \
         check passes vacuously)",
        path.display()
    );
    for (index, asset) in scenes.iter().enumerate() {
        extracted.models.push(gg_extract::Instance {
            entity: gg_ecs::Entity::from_bits(index as u64 + 1),
            offset: gg_math::render::Vec3::ZERO,
            rotation: gg_math::render::Quat::IDENTITY,
            half_extent: gg_math::render::Vec3::splat(1.0),
            color: 0x00ff_ffff,
            // Unread for an instance naming an asset: a pack's material is the
            // authored one (`Instance::surface`).
            surface: (0.0, 0.0),
            asset: *asset,
            // Unbounded: this harness exists to time a *load*, and an id culled
            // before it streams in would stop the clock by never asking.
            radius: f32::INFINITY,
            // Unread on the model path: the mesh is the shape.
            shape: gg_extract::shape::BOX,
        });
    }

    let mut frames = 0;
    let mut elapsed = None;
    while frames < LOAD_FRAMES && elapsed.is_none() {
        frames += 1;
        renderer.frame(
            &extracted,
            &gg_render::View::default(),
            [0.0, 0.0, 0.0, 1.0],
            &[],
        )?;
        elapsed = renderer
            .pack()
            .and_then(gg_render::content::Content::ready_at);
    }
    let report = renderer.shutdown();
    anyhow::ensure!(report.clean(), "unclean shutdown: {report:?} (§4.3)");

    let elapsed = elapsed.ok_or_else(|| {
        anyhow::anyhow!(
            "{} was still streaming after {frames} frames",
            path.display()
        )
    })?;
    println!(
        "gg-golden load: {} — {} KiB resident in {} ms over {frames} frame(s), budget {} ms (§6 M9)",
        path.display(),
        bytes / 1024,
        elapsed.as_millis(),
        LOAD_BUDGET.as_millis()
    );
    anyhow::ensure!(
        elapsed <= LOAD_BUDGET,
        "load to first frame is {} ms against a {} ms budget (§6 M9's exit row) — the knob is \
         `r.upload_budget`, and raising it trades a hitching frame for a shorter wait",
        elapsed.as_millis(),
        LOAD_BUDGET.as_millis()
    );
    Ok(())
}

/// Frames the field is given to gather itself before a reference is judged
/// (§6 M36).
///
/// A probe renders the scene six times, so a frame gathers a batch and the field
/// converges over several — and a reference blessed while probes are ungathered
/// is a reference of the *fallback* rather than of the feature, which is
/// `Content::pending`'s argument one milestone along. Bounded rather than a
/// spin: a field that cannot converge is a defect, and hanging is a worse way to
/// report one than failing.
const FIELD_FRAMES: usize = 24;

/// Render until the irradiance field holds every probe, and hand back the last
/// frame.
///
/// The closure takes the renderer rather than capturing it, which is what lets
/// this ask `field_pending` between calls without two borrows of the same thing.
fn gathered<F>(
    renderer: &mut gg_render::OffscreenRenderer,
    mut once: F,
) -> anyhow::Result<gg_render::OffscreenFrame>
where
    F: FnMut(&mut gg_render::OffscreenRenderer) -> anyhow::Result<gg_render::OffscreenFrame>,
{
    // As many probes a frame as one batch holds. A reference is not a session:
    // nothing here is holding a frame rate, so the rate that matters is the one
    // that converges the field in the fewest frames. `r.gi_rate`'s default is
    // the other side of that trade — frames against frame time.
    gg_render::cvars::GI_RATE.set_int(0);
    let mut frame = once(renderer)?;
    for _ in 0..FIELD_FRAMES {
        if renderer.field_pending().0 == 0 {
            return Ok(frame);
        }
        frame = once(renderer)?;
    }
    let (pending, probes) = renderer.field_pending();
    anyhow::ensure!(
        pending == 0,
        "the field was still gathering after {FIELD_FRAMES} frames ({pending} of {probes} \
         probes) — a half-built field is a reference of the warm-up, not of the feature"
    );
    Ok(frame)
}

/// How many frames the hall is streamed for before it is judged.
///
/// Streaming is frames deep by design (§4.6): frame one is what *requests* the
/// scene, so extract cannot expand it until frame two. A fixed count rather
/// than "until idle" for exactly that reason — the first idle frame is the one
/// before the meshes are known about.
const HALL_FRAMES: usize = 4;

fn hall_pack() -> anyhow::Result<std::path::PathBuf> {
    let path = std::path::PathBuf::from(HALL_PACK);
    anyhow::ensure!(
        path.is_file(),
        "{HALL_PACK} is not there — `cargo xtask assets` compiles it (§4.6). Packs are build \
         output and are never checked in, so this is a missing step and not a missing file."
    );
    Ok(path)
}

/// Render demo 04's hall out of its pack (§4.6, §4.10).
///
/// The world is built here rather than by running the demo's systems, because
/// systems run through the boundary and that is the shell's job — but every
/// number in it is the demo's own constant, so a change to where the demo puts
/// its visitor moves this reference rather than quietly diverging from it.
fn render_hall() -> Render {
    use gg_ecs::World;
    use gg_ecs::boundary::{Light, Model, Renderable};
    use gg_math::sim;

    let extent = BOXES_EXTENT;
    let pack = hall_pack()?;
    let mut world = World::new();
    world.register::<Model>()?;
    world.register::<Renderable>()?;
    world.register::<Light>()?;
    let sun = world.spawn();
    world.insert(
        sun,
        Light::sun(
            demo_04_scene::SUN_DIRECTION,
            demo_04_scene::SUN_COLOR,
            demo_04_scene::SUN_INTENSITY,
        ),
    )?;
    let hall = world.spawn();
    world.insert(
        hall,
        Model {
            tint: demo_04_scene::TINTS[0],
            ..Model::at(demo_04_scene::HALL, sim::DVec3::ZERO)
        },
    )?;
    let marker = world.spawn();
    world.insert(
        marker,
        Renderable::boxed(
            sim::DVec3::new(0.0, f64::from(demo_04_scene::MARKER_SIZE), 0.0),
            sim::Vec3::splat(demo_04_scene::MARKER_SIZE),
            0x00ff_e070,
        ),
    )?;

    let mut renderer = gg_render::OffscreenRenderer::new(extent)?;
    tracing::info!(device = %renderer.device().chosen, "offscreen device");
    renderer.open_pack(&pack)?;
    let view = gg_render::View {
        // Looking slightly down the hall from the visitor's opening pose, so
        // the floor's tiling, two pillars and the back wall are all in frame.
        pitch: -0.15,
        ..gg_render::View::default()
    };
    let mut extracted = gg_extract::Extracted::default();
    for _ in 0..HALL_FRAMES {
        extracted.clear(demo_04_scene::START_POSITION, view.frustum(extent));
        extracted.append::<Renderable>(&world)?;
        extracted.append_models::<Model>(&world, renderer.scenes())?;
        extracted.append_lights(&world)?;
        let _capture = gg_debug::capture::frame();
        renderer.frame(&extracted, &view, [0.02, 0.02, 0.03, 1.0], &[])?;
    }
    // After the stream, because a probe renders whatever is resident and a
    // field gathered over a half-loaded pack is a field of the fallback (§6 M36).
    let frame = gathered(&mut renderer, |r| {
        let _capture = gg_debug::capture::frame();
        Ok(r.frame(&extracted, &view, [0.02, 0.02, 0.03, 1.0], &[])?)
    })?;
    let pending = renderer
        .pack()
        .map_or(0, gg_render::content::Content::pending);
    anyhow::ensure!(
        pending == 0,
        "the hall was still streaming after {HALL_FRAMES} frames ({pending} asset(s) pending) — \
         judging a half-resident frame would make the reference a race"
    );
    ensure_clean(&renderer.shutdown())?;
    Ok(Capture {
        pixels: frame.pixels,
        extent,
        graph: frame.dump,
    })
}

fn atrium_pack() -> anyhow::Result<std::path::PathBuf> {
    let path = std::path::PathBuf::from(ATRIUM_PACK);
    anyhow::ensure!(
        path.is_file(),
        "{ATRIUM_PACK} is not there — `cargo xtask assets` compiles it (§4.6). Packs are build \
         output and are never checked in, so this is a missing step and not a missing file."
    );
    Ok(path)
}

/// Demo 06's atrium, lit (§6 M11).
///
/// The one scene in the roster that judges *shading*: a normal map bending the
/// light across a mortar line, an occlusion map darkening the same groove,
/// roughness spreading a highlight across ten spheres, a sun casting a shadow
/// map, four point lights falling off, and the tonemapper bringing all of it
/// into eight bits. Every constant is the demo's own, so a change to where its
/// lights are moves this reference rather than quietly diverging from it.
///
/// `phase` picks a point in the demo's sun sweep. It is the demo's own
/// `sun_direction`, not an angle restated here, which is what makes the
/// reference move when the sweep does.
fn render_atrium_at(phase: u64) -> Render {
    use gg_ecs::World;
    use gg_ecs::boundary::{Light, Model, Sky};
    use gg_math::sim;

    let extent = BOXES_EXTENT;
    let pack = atrium_pack()?;
    let mut world = World::new();
    world.register::<Model>()?;
    world.register::<Light>()?;
    world.register::<Sky>()?;
    // The compiled panorama out of the same pack (§6 M27) — what the mirror in
    // the back row reflects, and the half of this reference that would go dark
    // if the chain stopped being resident before the frame was captured.
    let sky = world.spawn();
    world.insert(
        sky,
        Sky::image(demo_06_lit::SKY, demo_06_lit::SKY_INTENSITY),
    )?;
    let atrium = world.spawn();
    world.insert(atrium, Model::at(demo_06_lit::ATRIUM, sim::DVec3::ZERO))?;
    // The sun first, which is what makes it the light the single cascade casts
    // — the same ordering the demo's `bootstrap` relies on (§6 M11).
    let sun = world.spawn();
    world.insert(
        sun,
        Light::sun(
            demo_06_lit::sun_direction(phase),
            demo_06_lit::SUN_COLOR,
            demo_06_lit::SUN_INTENSITY,
        ),
    )?;
    for (x, z) in demo_06_lit::LAMP_AT {
        let lamp = world.spawn();
        world.insert(
            lamp,
            Light::point(
                sim::DVec3::new(x, demo_06_lit::LAMP_HEIGHT, z),
                demo_06_lit::LAMP_COLOR,
                demo_06_lit::LAMP_INTENSITY,
                demo_06_lit::LAMP_RANGE,
            ),
        )?;
    }

    let mut renderer = gg_render::OffscreenRenderer::new(extent)?;
    tracing::info!(device = %renderer.device().chosen, "offscreen device");
    renderer.open_pack(&pack)?;
    let view = gg_render::View {
        // Slightly down, so the floor's shadows and the two rows of spheres are
        // both in frame.
        pitch: -0.18,
        ..gg_render::View::default()
    };
    let mut extracted = gg_extract::Extracted::default();
    for _ in 0..HALL_FRAMES {
        extracted.clear(demo_06_lit::START_POSITION, view.frustum(extent));
        extracted.append_models::<Model>(&world, renderer.scenes())?;
        extracted.append_lights(&world)?;
        let _capture = gg_debug::capture::frame();
        renderer.frame(&extracted, &view, [0.01, 0.012, 0.02, 1.0], &[])?;
    }
    // After the stream, because a probe renders whatever is resident and a
    // field gathered over a half-loaded pack is a field of the fallback (§6 M36).
    let frame = gathered(&mut renderer, |r| {
        let _capture = gg_debug::capture::frame();
        Ok(r.frame(&extracted, &view, [0.01, 0.012, 0.02, 1.0], &[])?)
    })?;
    let pending = renderer
        .pack()
        .map_or(0, gg_render::content::Content::pending);
    anyhow::ensure!(
        pending == 0,
        "the atrium was still streaming after {HALL_FRAMES} frames ({pending} asset(s) pending) — \
         judging a half-resident frame would make the reference a race"
    );
    // The claim the whole scene rests on, as a machine rather than as a look:
    // a frame with a casting sun runs the shadow pass, and one without does not
    // (§6 M11). A reference blessed from a frame that skipped it would be a
    // reference of an unlit room that happened to look plausible.
    // One pass per cascade since §6 M15.3, so the check is on the prefix — an
    // exact name would have to be kept in step with `r.shadow_cascades`, which
    // is a CVar and therefore not this test's to know.
    anyhow::ensure!(
        frame.order.iter().any(|name| name.starts_with("shadow")),
        "the atrium's sun casts, so the graph must have run a shadow pass: {:?}",
        frame.order
    );
    ensure_clean(&renderer.shutdown())?;
    Ok(Capture {
        pixels: frame.pixels,
        extent,
        graph: frame.dump,
    })
}

/// The atrium at the start of the sun's sweep.
fn render_atrium() -> Render {
    render_atrium_at(0)
}

/// The same room a quarter of the sweep on.
fn render_atrium_noon() -> Render {
    render_atrium_at(demo_06_lit::SUN_PERIOD / 4)
}

/// Demo 05's field: ten thousand parented objects over four meshes (§6 M10).
///
/// The world is built from the demo's own layout functions and composed by the
/// host's own `gg-scene`, so the reference moves when either changes rather than
/// quietly disagreeing with the game. It is the one scene that judges *batching*
/// — and it asserts the batch count as well as the pixels, because "four draws"
/// is an exit claim and a claim a harness cannot count is one nobody checks.
fn render_field() -> Render {
    use gg_ecs::World;
    use gg_ecs::boundary::{Light, Model, Node};
    use gg_math::sim;

    let extent = BOXES_EXTENT;
    let pack = field_pack()?;
    let mut world = World::new();
    world.register::<Model>()?;
    world.register::<Node>()?;
    world.register::<Light>()?;
    let sun = world.spawn();
    world.insert(
        sun,
        Light::sun(
            demo_05_many::SUN_DIRECTION,
            demo_05_many::SUN_COLOR,
            demo_05_many::SUN_INTENSITY,
        ),
    )?;
    for index in 0..demo_05_many::HUBS {
        let hub = world.spawn();
        world.insert(
            hub,
            Model::at(demo_05_many::MESHES[0], demo_05_many::hub_position(index)),
        )?;
        for slot in 0..demo_05_many::PER_HUB {
            let (offset, mesh) = demo_05_many::child_placement(slot);
            let child = world.spawn();
            world.insert(child, Node::at(hub, offset))?;
            world.insert(
                child,
                Model::at(demo_05_many::MESHES[mesh], sim::DVec3::ZERO),
            )?;
        }
    }
    // The hubs are unrotated here: this scene judges placement and batching, and
    // a pose that advanced with a tick would make the reference a clock.
    let mut hierarchy = gg_scene::Hierarchy::new();
    let composed = hierarchy.propagate(&mut world)?;
    anyhow::ensure!(
        composed.composed == demo_05_many::HUBS * demo_05_many::PER_HUB,
        "the hierarchy composed {} of {} nodes",
        composed.composed,
        demo_05_many::HUBS * demo_05_many::PER_HUB
    );

    let mut renderer = gg_render::OffscreenRenderer::new(extent)?;
    tracing::info!(device = %renderer.device().chosen, "offscreen device");
    renderer.open_pack(&pack)?;
    let view = gg_render::View {
        yaw: 0.22,
        pitch: -0.16,
        ..gg_render::View::default()
    };
    let mut extracted = gg_extract::Extracted::default();
    for _ in 0..HALL_FRAMES {
        extracted.clear(demo_05_many::START_POSITION, view.frustum(extent));
        extracted.append_models::<Model>(&world, renderer.scenes())?;
        extracted.append_lights(&world)?;
        let _capture = gg_debug::capture::frame();
        renderer.frame(&extracted, &view, [0.02, 0.02, 0.03, 1.0], &[])?;
    }
    // After the stream, because a probe renders whatever is resident and a
    // field gathered over a half-loaded pack is a field of the fallback (§6 M36).
    let frame = gathered(&mut renderer, |r| {
        let _capture = gg_debug::capture::frame();
        Ok(r.frame(&extracted, &view, [0.02, 0.02, 0.03, 1.0], &[])?)
    })?;
    let pending = renderer
        .pack()
        .map_or(0, gg_render::content::Content::pending);
    anyhow::ensure!(
        pending == 0,
        "the field was still streaming after {HALL_FRAMES} frames ({pending} asset(s) pending)"
    );
    // The exit claim, counted: every drawn object is an instance, and the draw
    // count is a property of the *content* — four meshes — not of how many
    // objects named them.
    let (instances, draws) = renderer.draw_counts();
    let total = demo_05_many::HUBS * (demo_05_many::PER_HUB + 1);
    anyhow::ensure!(
        draws == demo_05_many::MESHES.len(),
        "{total} objects batched into {draws} draws, expected {}",
        demo_05_many::MESHES.len()
    );
    anyhow::ensure!(
        instances > 0 && instances + extracted.culled == total,
        "{instances} instances + {} culled != {total} objects",
        extracted.culled
    );
    tracing::info!(
        objects = total,
        culled = extracted.culled,
        instances,
        draws,
        "field batched"
    );

    ensure_clean(&renderer.shutdown())?;
    Ok(Capture {
        pixels: frame.pixels,
        extent,
        graph: frame.dump,
    })
}

fn field_pack() -> anyhow::Result<std::path::PathBuf> {
    let path = std::path::PathBuf::from(FIELD_PACK);
    anyhow::ensure!(
        path.is_file(),
        "{FIELD_PACK} is not there — `cargo xtask assets` compiles it (§4.6)."
    );
    Ok(path)
}

fn boxes_world(frustum: gg_extract::Frustum) -> anyhow::Result<gg_extract::Extracted> {
    use gg_ecs::World;
    use gg_ecs::boundary::{Light, Renderable};
    use gg_math::sim;

    let mut world = World::new();
    world.register::<Renderable>()?;
    world.register::<Light>()?;
    // These boxes belong to no demo, so the sun is declared here — and it has to
    // be declared somewhere, because since M11 a world with no light renders by
    // the ambient term alone. An unlit reference would still catch a lost draw
    // and would stop catching a wrong normal, which is half of what this scene
    // is for.
    let sun = world.spawn();
    world.insert(
        sun,
        Light::sun(sim::Vec3::new(-0.35, -0.86, -0.37), 0x00ff_f4e0, 3.2),
    )?;
    // Depths chosen so the near box overlaps the far ones from the angled eye
    // and clears them from the straight-on one: one scene shows the colours,
    // the other shows the depth test deciding between them.
    for (position, half_extent, color) in [
        (sim::DVec3::new(-1.6, 0.0, -6.0), 1.0, 0x0030_a0ff),
        (sim::DVec3::new(1.6, 0.0, -6.0), 1.0, 0x00ff_a030),
        (sim::DVec3::new(0.0, -0.4, -3.2), 0.6, 0x0060_ff60),
    ] {
        let entity = world.spawn();
        world.insert(
            entity,
            Renderable::boxed(position, sim::Vec3::splat(half_extent), color),
        )?;
    }
    let mut extracted = gg_extract::Extracted::default();
    extracted.transforms::<Renderable>(&world, gg_math::sim::DVec3::ZERO, frustum)?;
    extracted.append_lights(&world)?;
    Ok(extracted)
}

fn render_boxes() -> Render {
    render_boxes_from(gg_render::View::default())
}

fn render_boxes_occluded() -> Render {
    render_boxes_from(gg_render::View {
        yaw: 0.14,
        pitch: -0.15,
        ..gg_render::View::default()
    })
}

/// The engine's own v1 pass list, headless (§1.5): the same `scene_graph` the
/// shell submits, with the readback pass where the present would be.
fn render_boxes_from(view: gg_render::View) -> Render {
    let extent = BOXES_EXTENT;
    // The real frustum, not `UNBOUNDED`: these references were blessed before
    // there was a culler, so a culler that rejects anything visible changes
    // them and the suite says so. Sphere bounds over-keep and never under-keep,
    // so a *correct* culler leaves every one of these images byte-identical.
    let extracted = boxes_world(view.frustum(extent))?;
    let mut renderer = gg_render::OffscreenRenderer::new(extent)?;
    tracing::info!(device = %renderer.device().chosen, "offscreen device");
    // No UI layer: these scenes gate the *renderer*, and an overlay in them
    // would put a frame counter in every reference image. The UI pass's own
    // pixels are gated offscreen in `gg-debug` instead.
    let frame = gathered(&mut renderer, |r| {
        let _capture = gg_debug::capture::frame();
        Ok(r.frame(&extracted, &view, [0.02, 0.02, 0.03, 1.0], &[])?)
    })?;
    ensure_clean(&renderer.shutdown())?;
    Ok(Capture {
        pixels: frame.pixels,
        extent,
        graph: frame.dump,
    })
}

/// Where `verify-gates` pulls the platformer's far plane: the eye stands
/// [`demo_11_platformer::CAMERA_BACK`] metres out of the z = 0 playfield, so a
/// far short of that distance culls every slab through the sixth plane alone
/// and leaves the sky.
const PLATFORMER_EATEN_FAR: f32 = 10.0;

fn render_platformer() -> Render {
    render_platformer_far(None)
}

/// The `orbit` reference's own extent, four times every other scene's.
///
/// Not a preference: demo 13's map symbols are sized in **map metres** and the
/// eye never moves, so the view is 18.6 m tall at every zoom and a `TRACE_DOT`
/// is 0.3 % of it — three pixels in the 1080p window the game is played in, and
/// *half* a pixel at `BOXES_EXTENT`. A reference where the conics are sub-pixel
/// noise gates nothing about the conics.
const ORBIT_EXTENT: (u32, u32) = (1280, 720);

/// The zoom the `orbit` reference is framed at — the demo's **own** far end,
/// not a number of this file's choosing, so the reference is the picture the
/// zoom key reaches and a re-framed map moves it.
const ORBIT_ZOOM: f64 = demo_13_orbit::ZOOM_FAR;

/// Where the mission is, at that zoom: sim ticks since the world opened. Not
/// tick 0 — the opening state has both planets near their authored anomalies
/// and the picture is a pair of rings with nothing on the move. This is the
/// epoch the flown transfer is handed to the star on (§6 M38 item 14), so what
/// the reference frames is the crossing itself.
const ORBIT_EPOCH: u64 = 4_925_342;

/// Demo 13's map, framed on the transfer (§6 M38).
///
/// The world is dealt from the demo's own constants — [`VERGE`](demo_13_orbit::VERGE),
/// [`OCHRE`](demo_13_orbit::OCHRE), the star's `mu` and radius, the dot sizes and
/// the map light — and the rings are stepped by the demo's own
/// [`sample`](demo_13_orbit::sample), rather than by `present`, which is a system
/// behind the ABI this binary cannot reach. Same tables and same stepping, so an
/// element that moves moves this reference instead of diverging from it.
///
/// What it guards that nothing else on the roster can: a picture whose subject
/// is **scale**. Every position here is an absolute `f64` metre — up to 2.3e11
/// of them — narrowed at `gg-extract`'s camera-relative seam and nowhere else
/// (§1.4), then drawn under reverse-Z where half-extents span four orders of
/// magnitude in one frame. A membrane that leaked an absolute `f32` anywhere
/// would not be subtly wrong here; the ring would be a scatter.
/// One conic's ribbon: [`TRACE`](demo_13_orbit::TRACE) segments drawn by the
/// demo's own `trace_segment`, closing on itself. Every conic this reference
/// frames is closed — the transfer and both planets — so there is no open arm
/// to leave off.
fn ring(
    world: &mut gg_ecs::World,
    points: &[gg_math::sim::DVec3],
    color: u32,
) -> anyhow::Result<()> {
    for slot in 0..points.len() {
        let segment = world.spawn();
        world.insert(
            segment,
            demo_13_orbit::trace_segment(points[slot], points[(slot + 1) % points.len()], color),
        )?;
    }
    Ok(())
}

fn render_orbit() -> Render {
    use demo_13_orbit as orbit;
    use gg_ecs::World;
    use gg_ecs::boundary::{Light, Renderable};
    use gg_math::sim;

    let extent = ORBIT_EXTENT;
    let seconds = ORBIT_EPOCH as f64 / f64::from(gg_core::DEFAULT_TICK_HZ);
    let scale = sim::powf(10.0, -ORBIT_ZOOM);

    // Where each body is at that epoch, absolutely — the same `state_at` the
    // demo's `advance` runs, which is the whole of what "on rails" means.
    let stations: Vec<(&orbit::Planet, sim::DVec3)> = [&orbit::VERGE, &orbit::OCHRE]
        .into_iter()
        .map(|planet| (planet, planet.orbit().state_at(seconds).0))
        .collect();
    // Centred on the departure planet rather than on a ship: this binary cannot
    // fly the mission (the pilot drives a systems table, and demo 10 owns the
    // `gg_game_*` names in this graph), and the map's own rule is that the
    // frame follows the thing the flight is about.
    let center = stations[0].1;
    let map = |absolute: sim::DVec3| (absolute - center) * scale;

    let mut world = World::new();
    world.register::<Renderable>()?;
    world.register::<Light>()?;

    let star = world.spawn();
    world.insert(
        star,
        Renderable::ball(
            map(sim::DVec3::ZERO),
            ((orbit::STAR_RADIUS * scale) as f32).max(orbit::MIN_DOT),
            orbit::STAR_GLOW,
        )
        .surfaced(0.0, 0.0),
    )?;
    for (planet, position) in &stations {
        let body = world.spawn();
        world.insert(
            body,
            Renderable::ball(
                map(*position),
                ((planet.radius * scale) as f32).max(orbit::MIN_DOT),
                planet.color,
            )
            .surfaced(0.0, 0.0),
        )?;
        ring(
            &mut world,
            &orbit::sample(planet.orbit(), sim::DVec3::ZERO, &map),
            orbit::dim(planet.color),
        )?;
    }
    // The ship is on the transfer the flown plan bought, stated about the star
    // at the epoch it was handed over on. Its elements are the reference's one
    // authored number and they are read off `gg-tools transfer`.
    let transfer = sim::Orbit {
        semi_major: 1.882_96e11,
        eccentricity: 0.234_62,
        inclination: 0.0,
        ascending_node: 0.0,
        argument_of_periapsis: 1.796,
        mean_anomaly: 0.0,
        mu: orbit::MU_STAR,
    };
    let ship = world.spawn();
    world.insert(
        ship,
        Renderable::ball(
            map(transfer.state_at(0.0).0),
            orbit::SHIP_DOT,
            orbit::SHIP_INK,
        )
        .surfaced(0.0, 0.0),
    )?;
    ring(
        &mut world,
        &orbit::sample(transfer, sim::DVec3::ZERO, &map),
        orbit::SHIP_TRACE_INK,
    )?;
    // The demo's own light, from the demo's own `map_light` — at the eye, so a
    // reference lit by a lamp the game does not have is not a thing this file
    // can accidentally produce (§6 M38 item 16).
    let lamp = world.spawn();
    world.insert(lamp, orbit::map_light())?;

    let view = gg_render::View {
        pitch: orbit::EYE_PITCH,
        ..gg_render::View::default()
    };
    let mut extracted = gg_extract::Extracted::default();
    extracted.clear(orbit::eye_position(), view.frustum(extent));
    extracted.append::<Renderable>(&world)?;
    extracted.append_lights(&world)?;
    let mut renderer = gg_render::OffscreenRenderer::new(extent)?;
    tracing::info!(
        device = %renderer.device().chosen,
        culled = extracted.culled,
        "offscreen device"
    );
    let frame = gathered(&mut renderer, |r| {
        let _capture = gg_debug::capture::frame();
        Ok(r.frame(&extracted, &view, [0.02, 0.02, 0.03, 1.0], &[])?)
    })?;
    ensure_clean(&renderer.shutdown())?;
    Ok(Capture {
        pixels: frame.pixels,
        extent,
        graph: frame.dump,
    })
}

/// Demo 11's level under the orthographic camera (§6 M20).
///
/// Everything is the scene's own data: the slabs, the goal, the player's
/// opening pose, the sun, and the `Eye` — half-height, position and all — are
/// read out of `scene.ggsave`, the file the shell loads and the editor
/// authors, so this reference moves with the level. `far` overrides
/// `r.ortho_far`'s default: `verify-gates` passes [`PLATFORMER_EATEN_FAR`] to
/// prove the sixth plane is load-bearing in the picture — M18 item 8's defect
/// class, expressed as pixels.
fn render_platformer_far(far: Option<f32>) -> Render {
    use gg_ecs::boundary::Renderable;

    let extent = BOXES_EXTENT;
    let (world, eye) = platformer_world()?;
    let mut view = gg_render::View {
        yaw: eye.yaw,
        pitch: eye.pitch,
        ortho: eye.ortho,
        ..gg_render::View::default()
    };
    anyhow::ensure!(
        view.ortho > 0.0,
        "the scene's eye is not orthographic — zero means perspective (§6 M20), and this scene \
         exists to gate the other projection"
    );
    if let Some(far) = far {
        view.ortho_far = far;
    }
    let mut extracted = gg_extract::Extracted::default();
    extracted.clear(eye.position, view.frustum(extent));
    extracted.append::<Renderable>(&world)?;
    extracted.append_lights(&world)?;
    let mut renderer = gg_render::OffscreenRenderer::new(extent)?;
    tracing::info!(
        device = %renderer.device().chosen,
        culled = extracted.culled,
        "offscreen device"
    );
    let frame = gathered(&mut renderer, |r| {
        let _capture = gg_debug::capture::frame();
        Ok(r.frame(&extracted, &view, [0.02, 0.02, 0.03, 1.0], &[])?)
    })?;
    ensure_clean(&renderer.shutdown())?;
    Ok(Capture {
        pixels: frame.pixels,
        extent,
        graph: frame.dump,
    })
}

/// The checked-in level as a world: every name its manifest carries —
/// the host's five protocol registrations plus demo 11's seven and `Sound` —
/// then the save loaded clean, exactly as the shell opens it (§6 M20 item 3).
fn platformer_world() -> anyhow::Result<(gg_ecs::World, gg_ecs::boundary::Eye)> {
    use gg_ecs::boundary::Eye;

    let (world, report) = platformer_loaded()?;
    anyhow::ensure!(
        report.is_clean(),
        "the scene did not load clean against this build: {report:?} — \
         `cargo run -p gg-golden -- migrate` rewrites it in this schema (§6 M26)"
    );
    let query = gg_ecs::Query::<&Eye>::new()?;
    let mut eye = None;
    world.each_ref(&query, |_, e: &Eye| eye = Some(*e));
    let eye = eye.ok_or_else(|| anyhow::anyhow!("the scene holds no eye to see it through"))?;
    Ok((world, eye))
}

/// The same file loaded, *without* judging the report — every name its manifest
/// carries: the host's six protocol registrations and demo 11's seven.
fn platformer_loaded() -> anyhow::Result<(gg_ecs::World, gg_ecs::MigrationReport)> {
    use gg_ecs::boundary::{Eye, Light, Model, Renderable, Sound, Widget};

    let path = demo_11_platformer::session::scene_path();
    let bytes = std::fs::read(&path).map_err(|e| {
        anyhow::anyhow!(
            "no level at {} ({e}) — the checked-in scene is the platformer's stage",
            path.display()
        )
    })?;
    let save = gg_ecs::Save::decode(&bytes)?;
    let mut world = gg_ecs::World::new();
    world.register::<Renderable>()?;
    world.register::<Eye>()?;
    world.register::<Model>()?;
    world.register::<Light>()?;
    world.register::<Widget>()?;
    world.register::<Sound>()?;
    world.register::<demo_11_platformer::Player>()?;
    world.register::<demo_11_platformer::Solid>()?;
    world.register::<demo_11_platformer::Goal>()?;
    world.register::<demo_11_platformer::Run>()?;
    world.register::<demo_11_platformer::Rig>()?;
    world.register::<demo_11_platformer::Cue>()?;
    world.register::<demo_11_platformer::Hud>()?;
    let report = world.load(&save)?;
    Ok((world, report))
}

/// Rewrite every checked-in `scene.ggsave` in this build's schema (§6 M26).
///
/// A **bless**, and the deliberate kind: it belongs beside `bless` rather than
/// in a tier, and it lives in this binary for the reason the scenes do — the
/// types a save has to be loaded against are the demos' own, and this is the
/// crate that already links every demo (§4.10).
///
/// Why it is a subcommand and not a hand-edit: a boundary component that gains
/// a field leaves every checked-in level one field behind, and `platformer_world`
/// refuses a migrated load on purpose — a reference image has to be rendered
/// from the level the build *declares*, not from one a migration invented on the
/// way in. That happened at §6 M24 and again at §6 M26, and the second time a
/// thing is done by hand is when it stops being done by hand.
///
/// Tick and provenance are carried rather than reset. They are the save's
/// identity (`Save::new`), and a level that changed identity because a field
/// was added would read as a different level to everything that opens one.
fn migrate() -> anyhow::Result<()> {
    let path = demo_11_platformer::session::scene_path();
    let before = std::fs::read(&path)?;
    let save = gg_ecs::Save::decode(&before)?;
    let (world, report) = platformer_loaded()?;
    if report.is_clean() {
        println!("gg-golden migrate: {} is already current", path.display());
        return Ok(());
    }
    let after = gg_ecs::Save::new(world.snapshot(), save.tick(), save.provenance()).encode();
    std::fs::write(&path, &after)?;
    // The report, not a byte count: what a reader has to check before committing
    // this is *which* component moved and how, and "4 bytes bigger" says neither.
    println!(
        "gg-golden migrate: {} rewritten, {} -> {} bytes\n{report:?}",
        path.display(),
        before.len(),
        after.len()
    );
    Ok(())
}

/// The UI scenes' extent. Small on purpose: what they gate is glyph coverage
/// and clipped edges, and a larger frame is a larger reference carrying the
/// same information (§3's image budget).
const UI_EXTENT: (u32, u32) = (480, 270);
/// Behind the UI, so a panel's alpha is visible as a blend rather than as a
/// colour. Flat: these two scenes gate the layer, not the renderer.
const UI_CLEAR: [f32; 4] = [0.10, 0.13, 0.17, 1.0];

/// The editor's extent, and it is not [`UI_EXTENT`]: `gg_ecs::boundary::CANVAS`
/// is 640×360 and the bitmap font is sampled nearest, so a non-integer fit turns
/// every stem into porridge and the reference would gate the resampler instead
/// of the panels. 1280×720 is exactly ×2.
const EDITOR_EXTENT: (u32, u32) = (1280, 720);

/// The monitor this scene is rendered on, which is none: a golden render has no
/// window, so it reports what every headless host reports and the reference is
/// therefore independent of whatever desk blessed it (§6 M15.1).
const GOLDEN_DPI: f32 = 1.0;

/// Render a UI-only frame: no world, one atlas, one draw.
fn render_ui(atlas: &gg_render::ui::Coverage<'_>, vertices: &[gg_render::ui::UiVertex]) -> Render {
    render_ui_at(UI_EXTENT, atlas, vertices)
}

/// As [`render_ui`], at an extent the caller chooses.
fn render_ui_at(
    extent: (u32, u32),
    atlas: &gg_render::ui::Coverage<'_>,
    vertices: &[gg_render::ui::UiVertex],
) -> Render {
    let mut renderer = gg_render::OffscreenRenderer::new(extent)?;
    tracing::info!(device = %renderer.device().chosen, "offscreen device");
    renderer.set_ui_atlas(atlas)?;
    let frame = {
        let _capture = gg_debug::capture::frame();
        renderer.frame(
            &gg_extract::Extracted::default(),
            &gg_render::View::default(),
            UI_CLEAR,
            vertices,
        )?
    };
    ensure_clean(&renderer.shutdown())?;
    Ok(Capture {
        pixels: frame.pixels,
        extent,
        graph: frame.dump,
    })
}

/// The editor, frozen mid-session (§6 M15's third exit row).
///
/// Not a lookalike panel authored here: this is `gg_editor::Editor` driven by
/// `gg_editor::session`'s own aiming helpers over a world shaped like a demo's,
/// so the reference moves when the editor does — the same argument
/// [`render_ui_overlay`] makes, and the reason both are worth having.
///
/// The dock is switched to the **perf** tab on purpose. The cvars tab reads a
/// process-global registry whose contents depend on which crates have
/// registered by the time this scene runs, and a reference image whose value
/// depends on scene *order* is a reference that fails for the wrong reason.
///
/// Since §6 M15.4 the selection also has a **box**, so the frame carries the
/// outline and the three gizmo arms drawn into the viewport's hole. Those are
/// the only things this editor ever draws there, and they are geometry produced
/// by a projection rather than by the layout — which is exactly the kind of
/// thing a numeric test agrees with and a picture catches.
fn render_editor() -> Render {
    use gg_editor::session::{Act, aim, frames};
    use gg_input::{ActionId, AxisId};

    let mut world = gg_ecs::World::new();
    world.register::<Spinner>()?;
    world.register::<gg_ecs::boundary::Model>()?;
    world.register::<gg_ecs::boundary::Renderable>()?;
    // Three archetypes, so the tree has variety and the mask column has
    // something to say. Which one lands on which row is archetype order and
    // therefore the ECS's business, not this scene's (`gg_editor::scan`).
    //
    // More than the tree pane holds, which is deliberate since §6 M15.1: the
    // scrollbar is part of the editor's face now, and a reference world that
    // fitted would gate every pane except the one that scrolls.
    for (spinners, models) in [(3u32, false), (40, true), (2, false)] {
        for i in 0..spinners {
            let entity = world.spawn();
            world.insert(
                entity,
                Spinner {
                    angle: f32::from(i as u16) * 0.25,
                    rate: 0.125,
                    ticks: u64::from(i) * 7,
                    awake: u32::from(i % 3 != 0),
                    _pad: 0,
                },
            )?;
            // Every one of them, and all within a few metres of the origin
            // looking down -Z: which entity lands on tree row 1 is archetype
            // order and so the ECS's business, and the selection's outline is
            // only in this reference if *whichever* one it is has a box on the
            // pane (§6 M15.4 item 2).
            world.insert(
                entity,
                gg_ecs::boundary::Renderable::boxed(
                    gg_math::sim::DVec3::new(f64::from(i % 3) - 1.0, 0.25, -6.0),
                    gg_math::sim::Vec3::splat(0.5),
                    0x00d0_9a4a,
                ),
            )?;
            if models {
                world.insert(
                    entity,
                    gg_ecs::boundary::Model::at(
                        "meshes/cube",
                        gg_math::sim::DVec3::new(f64::from(i), 1.5, -4.0),
                    ),
                )?;
            }
        }
    }

    // Select an entity, pick a field lane, bring up the perf pane, and leave
    // the pointer hovering `+` — one frame that has every pane in a live state.
    //
    // Aimed off a *placed* editor, because since §6 M15.1 the layout is the
    // operator's and a pane's rectangle is not a constant to look up.
    let mut editor = gg_editor::Editor::new(None);
    editor.place(EDITOR_EXTENT, GOLDEN_DPI);
    let at = |what: &'static str, aimed: Option<(f32, f32)>| {
        aimed.ok_or_else(|| anyhow::anyhow!("the editor's default layout has no {what}"))
    };
    let acts = [
        Act::To(at("tree", aim::tree_row(&editor, 1))?),
        Act::Settle(3),
        Act::Click,
        Act::To(at("inspector", aim::lane(&editor, 1, 0))?),
        Act::Settle(3),
        Act::Click,
        Act::To(at("perf tab", aim::tab(&editor, gg_editor::Pane::Perf))?),
        Act::Settle(3),
        Act::Click,
        Act::To(at("nudge bar", aim::plus(&editor))?),
        Act::Settle(4),
    ];
    let (click, x, y) = (ActionId::new(0), AxisId::new(0), AxisId::new(1));
    let passes: Vec<gg_rhi::PassTiming> =
        [("shadow", 0.884), ("forward-opaque", 2.113), ("ui", 0.058)]
            .iter()
            .enumerate()
            .map(|(i, (name, gpu_ms))| gg_rhi::PassTiming {
                name: (*name).to_owned(),
                gpu_ms: *gpu_ms,
                begin: i as i64,
                end: i as i64 + 1,
            })
            .collect();

    for (tick, frame) in frames(&acts, click, x, y).iter().enumerate() {
        editor.tick(
            &mut world,
            &gg_ui::router::Tick {
                motion: (frame.axes[x.index()], frame.axes[y.index()]),
                primary: frame.pressed(click),
                advance_focus: false,
                scroll: 0,
            },
            &gg_editor::Frame {
                extent: EDITOR_EXTENT,
                dpi: GOLDEN_DPI,
                tick: 41_337 + tick as u64,
                // `Stopped` since §6 M15.4, and the trade is deliberate: the
                // selection's outline and its three gizmo arms are drawn only in
                // that state, and they are a whole new drawing path — a
                // projection, a near clip and a stepped line — where the tag it
                // costs is one word in a corner that `xtask reload --editor`
                // greps for anyway.
                play: gg_editor::Play::Stopped,
                // No action map here at all: this host drives the editor from
                // authored frames, so there is nothing for the editor's own
                // camera verbs to resolve against and the reference keeps
                // showing the game's declared eye (§6 M15.2 item 2).
                input: None,
                // Nothing typed, for the same reason: the reference gates the
                // panel's empty field and its hint, which is what an operator
                // who has not clicked into it sees.
                typed: "",
                passes: &passes,
                memory: gg_rhi::MemoryUse {
                    buffers: 41,
                    buffer_bytes: 88 << 20,
                    images: 9,
                    image_bytes: 132 << 20,
                },
                save_path: "target/editor/demo-05.ggsv",
                title: "gg — demo_05_many",
                // A project, so the reference keeps gating the *game* pane and
                // not the launcher's picker (§6 M15.1 item 4) — this scene is
                // what an editor over a game looks like. A second scene for the
                // picker would be cheap and is deliberately not spent: its rows
                // are a table `session::aim::project` already aims into and
                // `xtask reload --launcher` already clicks, so a reference image
                // would gate the one part of it that is a plain list of buttons.
                project: Some("demo_05_many"),
                projects: &[],
                // No window at all here, so not maximized: the scene gates the
                // maximize glyph and the restore one is the windowed state.
                maximized: false,
                // The one host with no OS cursor to borrow, which is what
                // `draw_cursor` is for (§6 M15.1).
                reload: None,
                draw_cursor: true,
            },
        );
    }
    // The editor's own atlas and not the fallback band: its panels are set in
    // the rented face, and against `atlas::fallback()` every label here would
    // sample blank texels and the scene would gate a set of empty plates.
    render_ui_at(EDITOR_EXTENT, &editor.coverage(), editor.vertices())
}

/// A component no engine crate declares, so the inspector has to reach it
/// through the registry rather than through a type it was compiled against —
/// which is the whole of what §6 M15's inspector row claims. One field per lane
/// shape the value model knows: float, integer, and a boolean.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, gg_ecs::Component)]
#[component(id = "golden.spinner")]
#[repr(C)]
struct Spinner {
    angle: f32,
    rate: f32,
    ticks: u64,
    awake: u32,
    _pad: u32,
}

/// The real debug overlay, frozen — §6 M13's acceptance test as a picture.
///
/// Not a lookalike panel authored here: this is `gg_debug::Overlay` fed fixed
/// [`gg_debug::overlay::Stats`], so the reference moves when the overlay does
/// (§4.10). A *fresh* overlay is what makes it deterministic — with no frame on
/// record the timing rows format from a zeroed window rather than from a clock,
/// which is the only thing in the panel that would otherwise be a wall time.
fn render_ui_overlay() -> Render {
    use gg_debug::overlay::{Overlay, Stats};
    // The console's replies come out of the registry, so its knobs have to be
    // in it. Once per process: the roster may render a scene more than once.
    static REGISTERED: std::sync::Once = std::sync::Once::new();
    REGISTERED.call_once(|| {
        let _ = gg_debug::register();
    });

    let passes: Vec<gg_rhi::PassTiming> = [
        ("depth-prepass", 0.211),
        ("forward-opaque", 1.874),
        ("post", 0.402),
        ("ui", 0.061),
    ]
    .iter()
    .enumerate()
    .map(|(i, (name, gpu_ms))| gg_rhi::PassTiming {
        name: (*name).to_owned(),
        gpu_ms: *gpu_ms,
        begin: i as i64,
        end: i as i64 + 1,
    })
    .collect();
    // A distribution with a clear mode, so the chart is a shape and not a
    // rectangle: a wrong normalization is visible rather than plausible.
    let mut luminance = [0u32; gg_render::luminance::BINS];
    for (i, bin) in luminance.iter_mut().enumerate() {
        let from_mode = i as i32 - gg_render::luminance::BINS as i32 / 2;
        *bin = (900 - from_mode * from_mode * 6).max(0) as u32;
    }

    let mut overlay = Overlay::default();
    overlay.key(gg_input::Key::Backquote, true);
    for c in "d.scale".chars() {
        overlay.text(c);
    }
    overlay.key(gg_input::Key::Enter, true);
    for c in "r.expo".chars() {
        overlay.text(c);
    }
    let vertices = overlay.build(&Stats {
        extent: UI_EXTENT,
        tick: 214_748,
        passes: &passes,
        memory: gg_rhi::MemoryUse {
            buffers: 37,
            buffer_bytes: 92 << 20,
            images: 12,
            image_bytes: 148 << 20,
        },
        luminance: Some(&luminance),
    });
    render_ui(&gg_ui::atlas::fallback(), vertices)
}

/// The text-heavy scene §6 M13 asks for, on the vendored face: shaping,
/// rasterization, packing and sampling, at four sizes in one draw.
///
/// The per-backend reference sets carry something here they carry nowhere else:
/// outline rasterization is CPU float work, so this frame could differ between
/// hosts with no driver involved. Measured, it does not — the two lavapipe sets
/// are the same bytes on both operating systems — but that is an observation
/// about `zeno`, not a guarantee anything in this repo makes.
fn render_ui_text() -> Render {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .map(|root| root.join("assets/fonts/FiraMono-Regular.ttf"));
    let path = path.ok_or_else(|| anyhow::anyhow!("tools/gg-golden is two below the root"))?;
    let mut fonts = gg_ui::Fonts::default();
    let face = fonts.load(std::fs::read(&path)?, 0)?;
    let mut list = gg_ui::DrawList::default();

    let panel = gg_ui::Rect::new(
        8.0,
        8.0,
        UI_EXTENT.0 as f32 - 16.0,
        UI_EXTENT.1 as f32 - 16.0,
    );
    list.rect(panel, 0xc00c_1016);
    list.push_clip(panel);
    let mut stack = gg_ui::Stack::vertical((panel.x + 10.0, panel.y + 8.0), 4.0);
    for (text, px, color) in UI_TEXT {
        let line = fonts.metrics(face, *px).line_height;
        let cell = stack.push(fonts.layout(face, *px, text), line);
        list.glyphs(cell.x, cell.y, fonts.glyphs(), *color);
    }
    // A paragraph at a size where a lost pixel of coverage is a lost stem.
    let line = fonts.metrics(face, BODY_PX).line_height;
    stack.push(0.0, line * 0.5);
    for text in UI_BODY {
        let cell = stack.push(fonts.layout(face, BODY_PX, text), line);
        list.glyphs(cell.x, cell.y, fonts.glyphs(), 0xffb0_bcc8);
    }

    // The same face again through a clip that lands *inside* a glyph. A
    // geometric clip moves the uvs with the cut (§4.9), so these rows end in a
    // partial letter rather than a squeezed one — the failure a scissor cannot
    // have, and one no other scene in the roster would show.
    let cut = gg_ui::Rect::new(panel.x + 2.0, stack.content().bottom() + 8.0, 137.0, 44.0);
    list.push_clip(cut);
    let line = fonts.metrics(face, CUT_PX).line_height;
    let mut cutter = gg_ui::Stack::vertical((cut.x + 8.0, cut.y + 3.0), 2.0);
    for text in ["clipped mid-glyph", "and the uv with it"] {
        let cell = cutter.push(fonts.layout(face, CUT_PX, text), line);
        list.glyphs(cell.x, cell.y, fonts.glyphs(), 0xffff_c86e);
    }
    list.pop_clip();
    // And the bitmap fallback beside it, out of the same atlas in the same draw
    // — the claim §4.9 makes about one bitmap, standing next to its evidence.
    let mut fallback = gg_ui::Stack::vertical((cut.right() + 14.0, cut.y + 4.0), 3.0);
    for text in ["fallback 5x7, same atlas", "same draw, no second pass"] {
        let cell = fallback.push(gg_ui::DrawList::width(text), 8.0);
        list.text(cell.x, cell.y, text, 0xff8a_94a0);
    }
    list.pop_clip();

    render_ui(&fonts.coverage(), list.vertices())
}

/// Demo 10's board mid-game (§6 M18) — the whole game, because the whole game
/// is UI.
///
/// The state is built here rather than played to, for the reason
/// [`render_hall`] gives: a reference that was tick 137 of a session would move
/// every time the rules changed, and what this scene judges is the *picture*.
/// Everything about how that picture is composed is the demo's own — the widget
/// list comes from `declare`, the colours from `compose_well`/`cell_color`, so
/// a layout edit moves this reference instead of quietly disagreeing with it
/// (§4.10).
///
/// Rendered at exactly [`CANVAS`](gg_ecs::boundary::CANVAS): the fit is ×1, so
/// the nearest-sampled bitmap font lands on whole texels and the reference
/// gates the layout rather than a resampler.
fn render_tetris() -> Render {
    use demo_10_tetris as tetris;
    use gg_ecs::World;
    use gg_ecs::boundary::{CANVAS, Widget};

    // A board with something to read: a stack with a hole under an overhang,
    // one row nearly closed, a piece falling with its ghost below it, and a
    // held piece — every part of the layout carrying a value at once.
    let mut well = tetris::Well {
        cells: [[0; tetris::WIDTH]; tetris::HEIGHT],
    };
    for (row, filled) in [
        (tetris::HEIGHT - 1, 0b11_0111_1111u16),
        (tetris::HEIGHT - 2, 0b11_0011_0011),
        (tetris::HEIGHT - 3, 0b10_0011_0001),
        (tetris::HEIGHT - 4, 0b10_0000_0001),
    ] {
        for col in 0..tetris::WIDTH {
            if filled & (1 << col) != 0 {
                // The kind is what colours a locked cell, so vary it: a stack in
                // one colour would pass a reference that lost the mapping.
                well.cells[row][col] = (row + col) as u8 % 7 + 1;
            }
        }
    }

    let mut play = tetris::new_play(0x5445_5452_4953_0001);
    play.score = 128_400;
    play.lines = 37;
    play.level = 4;
    play.hold = 2; // T, so the hold bay is not the empty case
    // Ahead of the run in progress, so BEST and SCORE are visibly different
    // numbers rather than one repeated twice.
    let best = tetris::Best {
        score: 204_900,
        top: [0; 5],
    };
    let piece = tetris::Piece {
        col: 4,
        row: 6,
        kind: 5, // J, which is asymmetric in every rotation
        rot: 1,
        pad: [0; 2],
    };

    let grid = tetris::compose_well(&well, &play, &piece);
    let bays = tetris::compose_bays(&play);
    let mut declared = Vec::new();
    tetris::declare(|part, mut widget| {
        match part {
            // The banner belongs to a dead board and this one is alive — and
            // the menu layer to a screen this picture is not on (§6 M19); both
            // are declared at zero rects, which draw nothing.
            tetris::Part::Chrome | tetris::Part::Banner(_) | tetris::Part::Menu(_) => {}
            tetris::Part::Cell(cell) => widget.color = tetris::cell_color(&grid, &cell),
            tetris::Part::Bay(bay) => widget.color = tetris::bay_color(&bays, &bay),
            tetris::Part::Value(line) => {
                widget.set_text(&tetris::value_of(&play, &best, &line).to_string());
            }
        }
        declared.push(widget);
    });

    let mut world = World::new();
    world.register::<Widget>()?;
    for widget in declared {
        let entity = world.spawn();
        world.insert(entity, widget)?;
    }

    let mut ui = gg_ui::boundary::Ui::new()?;
    let vertices = ui
        .frame(
            &mut world,
            &gg_ui::router::Tick::default(),
            gg_ui::Fit::new(CANVAS),
        )
        .to_vec();
    render_ui_at(CANVAS, &gg_ui::atlas::fallback(), &vertices)
}

/// Heading rows: sizes that share no ppem, and characters chosen for what they
/// stress — stems and counters at small sizes, the punctuation a hinted
/// rasterizer rounds differently, and the pairs a shaper advances.
const UI_TEXT: &[(&str, u16, u32)] = &[
    ("gg-ui text", 30, 0xffff_ffff),
    ("shaped by swash, packed by our own atlas,", 15, 0xffd8_e0e8),
    ("sampled from one bitmap in one draw call.", 15, 0xffd8_e0e8),
    ("0123456789 ,.;:!? ijlI1 WMmw ()[]{} /|\\", 12, 0xff7f_d0a0),
];
/// Body copy, at the size the overlay's successor would actually use. Enough of
/// it that the packer has had to open more than one shelf by the time the frame
/// is done, which is the state the atlas spends its life in.
const UI_BODY: &[&str] = &[
    "We own the draw batching, the glyph atlas, the",
    "input routing, the layout and the styling. We",
    "rent shaping and rasterization: bidi, ligatures,",
    "hinting and fallback chains are not our fight.",
];
const BODY_PX: u16 = 11;
/// The clipped rows' size, small enough that a lost pixel of coverage shows.
const CUT_PX: u16 = 14;

/// Render demo 01's scene — the same SPIR-V and push constants the demo draws
/// with (§4.10: the golden guards the demo, not a lookalike).
fn render_triangle() -> Render {
    render_triangle_scaled(1.0)
}

/// The same frame with the transform's diagonal scaled — `1.0` is the demo's
/// own. The knob exists for `verify-gates` and nothing else: it is applied to
/// the diagonal precisely because a diagonal is symmetric under either matrix
/// convention, so the deformation is a known size without the harness having to
/// know how the shader reads the matrix.
fn render_triangle_scaled(scale: f32) -> Render {
    let extent = demo_01_triangle::GOLDEN_EXTENT;
    let mut rhi = OffscreenRhi::new(extent)?;
    tracing::info!(device = %rhi.device_report().chosen, "offscreen device");
    let pipeline = rhi.create_pipeline(&demo_01_triangle::pipeline_desc())?;
    let mut push = demo_01_triangle::push_for_extent(extent);
    push.transform[0][0] *= scale;
    push.transform[1][1] *= scale;
    let draws = [gg_rhi::DrawSpec {
        pipeline,
        push_constants: bytemuck::bytes_of(&push),
        count: demo_01_triangle::VERTEX_COUNT,
        index_buffer: None,
        indirect: None,
        depth_bias: None,
        viewport: None,
    }];

    let dest = readback_buffer(&mut rhi, extent)?;
    let mut transients = Transients::default();
    let mut frame = transients.frame(&mut rhi, extent)?;
    let backbuffer = frame.backbuffer();
    let into = frame.readback_buffer("golden.readback", dest);
    let mut declared: Vec<Declared<'_>> = demo_01_triangle::declare(backbuffer, &draws).into();
    declared.push(readback_pass(backbuffer, into));
    let compiled = frame.compile(&declared)?;
    let graph = compiled.dump();
    {
        // Scoped for the same reason as `render_mesh_of`'s: the capture must
        // close before the device it was opened against does.
        let _capture = gg_debug::capture::frame();
        rhi.execute(&compiled.passes())?;
    }
    let pixels = rhi.map_buffer(dest)?.to_vec();

    ensure_clean(&rhi.shutdown())?;
    Ok(Capture {
        pixels,
        extent,
        graph,
    })
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

/// Every gate number, printed whether it objected or not — a report that shows
/// only the failing metric teaches a reader nothing about the margin on the
/// others.
fn numbers(comparison: &Comparison, policy: Policy) -> Vec<(String, String)> {
    vec![
        (
            "differing pixels".into(),
            format!(
                "{} / {} (tolerance {})",
                comparison.diff_pixels, policy.max_diff_pixels, policy.tolerance
            ),
        ),
        (
            "max channel delta".into(),
            format!(
                "{} (benign up to {})",
                comparison.max_delta, policy.benign_delta
            ),
        ),
        (
            "worst-window DSSIM".into(),
            format!("{:.5} / {:.5}", comparison.dssim_worst, policy.max_dssim),
        ),
        ("mean DSSIM".into(), format!("{:.5}", comparison.dssim_mean)),
        (
            "mean signed error".into(),
            format!(
                "R{:+.4} G{:+.4} B{:+.4} / ±{:.4} LSB",
                comparison.channel_bias[0],
                comparison.channel_bias[1],
                comparison.channel_bias[2],
                policy.max_bias
            ),
        ),
        // Per channel above and per region here, because one number over R+G+B
        // and over the whole frame cancels two errors an eye does not: a cast
        // (+red against −blue) and a swing (one half up, the other down).
        (
            format!("worst {0}x{0} region bias", compare::REGION),
            format!(
                "{:+.4} / ±{:.4} LSB",
                comparison.region_bias,
                policy.max_region_bias()
            ),
        ),
    ]
}

/// Where failure artifacts land: beside the build products, never in the tree.
fn artifacts_root() -> PathBuf {
    PathBuf::from("target/golden")
}

/// One pixel of the 360-line frame, as a scale on a transform whose clip space
/// spans 2.0 units. Small enough that the two frames are indistinguishable side
/// by side, which is the point: the gate has to see what an eye does not.
const ONE_PIXEL: f32 = 1.0 - 2.0 / 360.0;

/// §4.10 / M7 exit: a suite that cannot fail is not a gate. This renders demo
/// 01 twice — once as its reference sees it, once with a deliberate one-pixel
/// deformation — and requires the exact gate to reject the second; then it
/// perturbs the honest frame with symmetric rounding noise and requires the
/// perceptual gate to forgive that one and *say so*. Both halves run on real
/// renders through the real graph, because a gate proven only against synthetic
/// buffers is a gate proven against the wrong thing.
fn verify_gates() -> anyhow::Result<()> {
    let policy = SCENES
        .iter()
        .find(|s| s.name == "triangle")
        .map(|s| s.policy)
        .ok_or_else(|| anyhow::anyhow!("the triangle scene left the roster"))?;

    let honest = render_triangle()?;
    let deformed = render_triangle_scaled(ONE_PIXEL)?;
    let moved = compare::compare(&deformed.pixels, &honest.pixels, honest.extent, policy)?;
    anyhow::ensure!(
        !matches!(moved.verdict(policy), Verdict::BenignDrift),
        "a one-pixel geometric change was forgiven as precision drift — the perceptual gate is \
         an escape hatch, not a second opinion"
    );
    anyhow::ensure!(
        matches!(moved.verdict(policy), Verdict::Fail),
        "a one-pixel geometric change passed both gates: {} differing pixel(s) against a budget \
         of {}, worst-window DSSIM {:.5} against {:.5}",
        moved.diff_pixels,
        policy.max_diff_pixels,
        moved.dssim_worst,
        policy.max_dssim
    );
    println!(
        "gg-golden: a one-pixel change moves {} pixel(s) (budget {}), worst-window DSSIM {:.5} \
         (budget {:.5}) — rejected",
        moved.diff_pixels, policy.max_diff_pixels, moved.dssim_worst, policy.max_dssim
    );

    let noisy = rounding_noise(&honest.pixels, 3);
    let drift = compare::compare(&noisy, &honest.pixels, honest.extent, policy)?;
    anyhow::ensure!(
        drift.diff_pixels > policy.max_diff_pixels,
        "the noise was too small for the exact gate to object to — this half proves nothing"
    );
    anyhow::ensure!(
        matches!(drift.verdict(policy), Verdict::BenignDrift),
        "symmetric ±3 LSB noise was not recognised as precision drift: {} differing pixel(s), \
         worst-window DSSIM {:.5} against {:.5}, mean bias {:+.4} and worst region {:+.4} against \
         ±{:.4}",
        drift.diff_pixels,
        drift.dssim_worst,
        policy.max_dssim,
        drift.mean_bias(),
        drift.region_bias,
        policy.max_bias
    );
    println!(
        "gg-golden: symmetric ±3 LSB noise moves {} pixel(s), worst-window DSSIM {:.5}, mean bias \
         {:+.4}, worst region {:+.4} — recorded as drift, not a regression",
        drift.diff_pixels,
        drift.dssim_worst,
        drift.mean_bias(),
        drift.region_bias
    );

    // §6 M20: the finite-far culler proven able to fail, as pixels. The sixth
    // plane exists only for the orthographic path, and M18 item 8's defect
    // class — a culler quietly wrong — has two directions: eating too much
    // shows in a picture, keeping too much never does (the frustum unit tests
    // hold that side). So the deformation eats: the far plane pulled inside
    // the playfield must take the level with it, and the exact gate must say
    // so rather than shrug. A suite whose ortho scene kept rendering with its
    // culler broken would be the M18 hole re-opened.
    let policy = SCENES
        .iter()
        .find(|s| s.name == "platformer")
        .map(|s| s.policy)
        .ok_or_else(|| anyhow::anyhow!("the platformer scene left the roster"))?;
    let level = render_platformer()?;
    let eaten = render_platformer_far(Some(PLATFORMER_EATEN_FAR))?;
    let culled = compare::compare(&eaten.pixels, &level.pixels, level.extent, policy)?;
    anyhow::ensure!(
        matches!(culled.verdict(policy), Verdict::Fail),
        "the far plane was pulled to {PLATFORMER_EATEN_FAR} m — inside the playfield — and the \
         picture forgave it: {} differing pixel(s) against {}, worst-window DSSIM {:.5}. Either \
         the sixth plane stopped culling or the level stopped being in frame; both are the \
         culler no longer load-bearing in this reference.",
        culled.diff_pixels,
        policy.max_diff_pixels,
        culled.dssim_worst,
    );
    println!(
        "gg-golden: the far plane at {PLATFORMER_EATEN_FAR} m takes the level with it — {} \
         pixel(s) moved, worst-window DSSIM {:.5} — rejected, so the ortho culler can fail (§6 \
         M20)",
        culled.diff_pixels, culled.dssim_worst
    );
    Ok(())
}

/// Symmetric per-pixel noise: `±amount` on RGB, sign alternating by pixel, so
/// it cancels in the mean the way a driver's rounding does. Deliberately not a
/// random dither — a gate whose input changes run to run reports a different
/// number every night.
fn rounding_noise(pixels: &[u8], amount: i16) -> Vec<u8> {
    pixels
        .chunks_exact(4)
        .enumerate()
        .flat_map(|(i, p)| {
            let d = if i % 2 == 0 { amount } else { -amount };
            let nudge = |v: u8, d: i16| -> u8 {
                u8::try_from((i16::from(v) + d).clamp(0, 255)).unwrap_or(v)
            };
            [nudge(p[0], d), nudge(p[1], -d), nudge(p[2], d), p[3]]
        })
        .collect()
}

fn run(filter: Option<&str>) -> anyhow::Result<()> {
    let backend = backend_id()?;
    let root = references_root().join(&backend);
    let mut failures = Vec::new();
    let mut drifted = Vec::new();
    let mut entries = Vec::new();
    let mut ran = 0usize;

    for scene in SCENES {
        if filter.is_some_and(|f| f != scene.name) {
            continue;
        }
        ran += 1;
        let Capture {
            pixels: actual,
            extent,
            ..
        } = (scene.render)()?;
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
        let comparison = compare::compare(&actual, &reference, extent, scene.policy)?;
        let heatmap = compare::heatmap(&comparison, scene.policy.tolerance);
        let panel = |status, headline: String| -> anyhow::Result<report::Entry> {
            Ok(report::Entry {
                scene: scene.name.to_string(),
                status,
                headline,
                numbers: numbers(&comparison, scene.policy),
                images: vec![
                    ("reference", png_io::encode(&reference, extent)?),
                    ("actual", png_io::encode(&actual, extent)?),
                    ("heatmap", png_io::encode(&heatmap, extent)?),
                ],
            })
        };

        match comparison.verdict(scene.policy) {
            Verdict::Pass => tracing::info!(
                scene = scene.name,
                diff_pixels = comparison.diff_pixels,
                max_delta = comparison.max_delta,
                dssim = comparison.dssim_worst,
                "golden pass"
            ),
            // Recorded, not swallowed (§4.10): the suite stays green and the
            // drift arrives in the report with its numbers attached, so a
            // reviewer decides whether to re-bless rather than never hearing.
            Verdict::BenignDrift => {
                entries.push(panel(
                    "DRIFT",
                    format!(
                        "over the exact gate's pixel budget, but no channel moved more than {} \
                         and the worst window's DSSIM is {:.5} — precision drift, not a \
                         regression",
                        comparison.max_delta, comparison.dssim_worst
                    ),
                )?);
                drifted.push(scene.name);
            }
            Verdict::Fail => {
                // On-disk artifacts as well as the report: the PNGs are what an
                // image viewer, a diff tool or the next agent reaches for.
                let out_dir = artifacts_root().join(scene.name);
                let actual_path = out_dir.join("actual.png");
                let heatmap_path = out_dir.join("diff-heatmap.png");
                png_io::write(&actual_path, &actual, extent)?;
                png_io::write(&heatmap_path, &heatmap, extent)?;
                let structural = comparison.dssim_worst > scene.policy.max_dssim;
                entries.push(panel(
                    "FAIL",
                    if structural {
                        format!(
                            "structural regression: worst-window DSSIM {:.5} exceeds {:.5} — the \
                             picture moved, not just its numbers",
                            comparison.dssim_worst, scene.policy.max_dssim
                        )
                    } else if comparison.biased(scene.policy) {
                        // A level error, which is the one thing neither the pixel
                        // budget nor DSSIM can see: say so, or the headline reads
                        // "0 differing pixels" beside a FAIL.
                        format!(
                            "a wrong level, not a moved picture: worst channel {:+.4} of ±{:.4} \
                             LSB over the frame, worst {}px region {:+.4} of ±{:.4}",
                            comparison.mean_bias(),
                            scene.policy.max_bias,
                            compare::REGION,
                            comparison.region_bias,
                            scene.policy.max_region_bias()
                        )
                    } else {
                        format!(
                            "{} differing pixel(s) against a budget of {}, worst channel delta {} \
                             — too far to call precision drift",
                            comparison.diff_pixels,
                            scene.policy.max_diff_pixels,
                            comparison.max_delta
                        )
                    },
                )?);
                failures.push(format!(
                    "{}: {} differing pixel(s), max channel delta {}, worst-window DSSIM {:.5}, \
                     mean signed error {:+.4}, worst region {:+.4} against {}/{}, {:.5} and \
                     ±{:.4} — see {} and {}",
                    scene.name,
                    comparison.diff_pixels,
                    comparison.max_delta,
                    comparison.dssim_worst,
                    comparison.mean_bias(),
                    comparison.region_bias,
                    scene.policy.tolerance,
                    scene.policy.max_diff_pixels,
                    scene.policy.max_dssim,
                    scene.policy.max_bias,
                    actual_path.display(),
                    heatmap_path.display(),
                ));
            }
        }
    }

    if !entries.is_empty() {
        let path = artifacts_root().join("report.html");
        report::write(&path, &backend, &entries)?;
        println!("gg-golden: report written to {}", path.display());
    }

    anyhow::ensure!(ran > 0, "no scene matched the filter");
    anyhow::ensure!(
        failures.is_empty(),
        "golden suite failed:\n{}",
        failures.join("\n")
    );
    if drifted.is_empty() {
        println!("gg-golden: {ran} scene(s) pass against `{backend}` references");
    } else {
        println!(
            "gg-golden: {ran} scene(s) pass against `{backend}` references \
             ({} accepted by the perceptual gate as precision drift: {})",
            drifted.len(),
            drifted.join(", ")
        );
    }
    Ok(())
}

/// `bless` writes references — and, when it overwrites one, says exactly what it
/// changed. A PNG diff is unreadable in a text review, so the reviewable artifact
/// is the same HTML report the failures use, old and new side by side with the
/// heatmap between them (§4.10: "a deliberate, reviewed act").
fn bless(filter: Option<&str>) -> anyhow::Result<()> {
    let backend = backend_id()?;
    let root = references_root().join(&backend);
    let mut blessed = 0usize;
    let mut entries = Vec::new();
    for scene in SCENES {
        if filter.is_some_and(|f| f != scene.name) {
            continue;
        }
        let Capture {
            pixels: actual,
            extent,
            ..
        } = (scene.render)()?;
        let path = root.join(format!("{}.png", scene.name));
        match path.exists().then(|| png_io::read(&path)).transpose()? {
            Some((previous, previous_extent)) if previous_extent == extent => {
                let comparison = compare::compare(&previous, &actual, extent, scene.policy)?;
                println!(
                    "gg-golden: {} changes {} pixel(s) (max channel delta {}, worst-window DSSIM \
                     {:.5})",
                    scene.name,
                    comparison.diff_pixels,
                    comparison.max_delta,
                    comparison.dssim_worst
                );
                entries.push(report::Entry {
                    scene: scene.name.to_string(),
                    status: "BLESSED",
                    headline: format!("reference rewritten at {}", path.display()),
                    numbers: numbers(&comparison, scene.policy),
                    images: vec![
                        ("previous reference", png_io::encode(&previous, extent)?),
                        ("new reference", png_io::encode(&actual, extent)?),
                        (
                            "heatmap",
                            png_io::encode(
                                &compare::heatmap(&comparison, scene.policy.tolerance),
                                extent,
                            )?,
                        ),
                    ],
                });
            }
            // A resized or first-time reference has nothing to diff against;
            // saying so is the honest report, not a silent write.
            other => println!(
                "gg-golden: {} — {} reference",
                scene.name,
                if other.is_some() { "resized" } else { "new" }
            ),
        }
        png_io::write(&path, &actual, extent)?;
        println!(
            "gg-golden: blessed {} — a deliberate, reviewed act; the image diff belongs in the PR (§4.10)",
            path.display()
        );
        blessed += 1;
    }
    if !entries.is_empty() {
        let path = artifacts_root().join("bless-report.html");
        report::write(&path, &backend, &entries)?;
        println!("gg-golden: review the change at {}", path.display());
    }
    anyhow::ensure!(blessed > 0, "no scene matched the filter");
    Ok(())
}

/// `gg-golden graph [scene]` — §4.5's render-graph dump, for every scene the
/// harness renders. Printed from the *compiled* graph the frame ran, so what a
/// reader sees is the execution order rather than a description of it.
fn graph(filter: Option<&str>) -> anyhow::Result<()> {
    let mut printed = 0usize;
    for scene in SCENES {
        if filter.is_some_and(|f| f != scene.name) {
            continue;
        }
        println!("=== {} ===", scene.name);
        print!("{}", (scene.render)()?.graph);
        printed += 1;
    }
    anyhow::ensure!(printed > 0, "no scene matched the filter");
    Ok(())
}

/// `gg-golden capture [scene]` — §4.8's "one command": render the scene exactly
/// as `run` renders it, with a RenderDoc capture bracketed around the submit,
/// and print the `.rdc`. What lands in the capture is the frame the gate judges,
/// not a re-staged lookalike of it.
///
/// Windowless, which is the whole reason the in-application API is here:
/// RenderDoc's own hotkey hangs itself on a Present, and nothing in this binary
/// presents (§1.5).
fn capture(filter: Option<&str>) -> anyhow::Result<()> {
    anyhow::ensure!(
        gg_debug::capture::available(),
        "not running under RenderDoc, so there is nothing to capture with — launch this command \
         from the RenderDoc UI (Launch Application) or under `renderdoccmd capture`. RenderDoc \
         must be in the process before the Vulkan instance is created, so it cannot be attached \
         from here (§4.8)"
    );
    let root = artifacts_root().join("captures");
    std::fs::create_dir_all(&root)?;
    let mut captured = 0usize;
    for scene in SCENES {
        if filter.is_some_and(|f| f != scene.name) {
            continue;
        }
        // A stem, not a path: RenderDoc appends the suffix itself — `_capture`
        // for these, since an offscreen frame has no present to number.
        gg_debug::capture::set_path_template(&root.join(scene.name).to_string_lossy());
        gg_debug::capture::request(1);
        let _ = (scene.render)()?;
        let path = gg_debug::capture::latest()
            .ok_or_else(|| anyhow::anyhow!("{}: RenderDoc wrote no capture", scene.name))?;
        println!("gg-golden: {} → {}", scene.name, path.display());
        captured += 1;
    }
    anyhow::ensure!(captured > 0, "no scene matched the filter");
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
    let filter = args
        .get(1)
        .map(String::as_str)
        .filter(|a| !a.starts_with('-'));
    match args.first().map(String::as_str) {
        Some("run") => run(filter),
        Some("bless") => bless(filter),
        Some("graph") => graph(filter),
        Some("verify-gates") => verify_gates(),
        Some("chaos") => chaos(filter),
        Some("capture") => capture(filter),
        // Two scenes, deliberately: `boxes` is M8's pass-list macro and the
        // baseline the archive already holds, `field` is §6 M10's ten-thousand-
        // object frame and measures the whole per-frame chain rather than only
        // the renderer's share of it.
        Some("bench") if filter == Some("field") => bench::field(
            flag(&args, "--frames").unwrap_or(BENCH_FRAMES),
            args.iter().any(|a| a == "--json"),
        ),
        Some("bench") => bench::run(
            &boxes_world(gg_render::View::default().frustum(BOXES_EXTENT))?,
            flag(&args, "--frames").unwrap_or(BENCH_FRAMES),
            args.iter().any(|a| a == "--json"),
        ),
        Some("load") => load(filter),
        Some("boot") => boot(filter),
        Some("migrate") => migrate(),
        // What `xtask gpu` refuses a software rasterizer with: the same
        // bring-up and naming the reference sets are keyed by, so the identity
        // checked is the identity the suite would run under.
        Some("backend") => {
            println!("{}", backend_id()?);
            Ok(())
        }
        _ => anyhow::bail!(
            "usage: gg-golden \
             <run|bless|graph|verify-gates|chaos|capture|bench|load|boot|migrate|backend> \
             [scene|seed|pack]"
        ),
    }
}

/// Long enough for a p99 to mean something (a 300-frame run's p99 is its third
/// worst frame), short enough that a recording is seconds.
const BENCH_FRAMES: usize = 300;

/// `--name <value>`, parsed. The only flag-shaped argument this binary takes;
/// a parser crate for one integer would be a dependency per option.
fn flag(args: &[String], name: &str) -> Option<usize> {
    let at = args.iter().position(|a| a == name)?;
    args.get(at + 1)?.parse().ok()
}
