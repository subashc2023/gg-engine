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

/// One golden scene: how to render it and how strictly to judge it.
struct Scene {
    name: &'static str,
    policy: Policy,
    render: fn() -> Render,
}

/// The roster. Seven scenes across three sources — two demos, the engine's own
/// v1 pass list, and two replay-driven captures — each with its own policy,
/// because "how strictly" is a property of what the frame contains and not of
/// the harness (§4.10 per-test config).
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
];

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
    rhi.execute(&compiled.passes())?;
    let pixels = rhi.map_buffer(dest)?.to_vec();

    let report = rhi.shutdown();
    anyhow::ensure!(
        report.clean(),
        "unclean render: {} validation message(s), {} leak(s) {:?} (§4.3, §5.4)",
        report.validation_messages,
        report.leaked_allocations.len(),
        report.leaked_allocations,
    );
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
fn boxes_world() -> anyhow::Result<gg_extract::Extracted> {
    use gg_ecs::World;
    use gg_ecs::boundary::Renderable;
    use gg_math::sim;

    let mut world = World::new();
    world.register::<Renderable>()?;
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
    extracted.transforms::<Renderable>(&world, gg_math::sim::DVec3::ZERO)?;
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
    let extracted = boxes_world()?;
    let mut renderer = gg_render::OffscreenRenderer::new(extent)?;
    tracing::info!(device = %renderer.device().chosen, "offscreen device");
    let frame = renderer.frame(&extracted, &view, [0.02, 0.02, 0.03, 1.0])?;
    let report = renderer.shutdown();
    anyhow::ensure!(
        report.clean(),
        "unclean render: {} validation message(s), {} leak(s) {:?} (§4.3, §5.4)",
        report.validation_messages,
        report.leaked_allocations.len(),
        report.leaked_allocations,
    );
    Ok(Capture {
        pixels: frame.pixels,
        extent,
        graph: frame.dump,
    })
}

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
    rhi.execute(&compiled.passes())?;
    let pixels = rhi.map_buffer(dest)?.to_vec();

    let report = rhi.shutdown();
    anyhow::ensure!(
        report.clean(),
        "unclean render: {} validation message(s), {} leak(s) {:?} (§4.3, §5.4)",
        report.validation_messages,
        report.leaked_allocations.len(),
        report.leaked_allocations,
    );
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
            format!("{:+.4} / ±{:.4} LSB", comparison.mean_bias, policy.max_bias),
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
         worst-window DSSIM {:.5} against {:.5}, mean bias {:+.4} against ±{:.4}",
        drift.diff_pixels,
        drift.dssim_worst,
        policy.max_dssim,
        drift.mean_bias,
        policy.max_bias
    );
    println!(
        "gg-golden: symmetric ±3 LSB noise moves {} pixel(s), worst-window DSSIM {:.5}, mean bias \
         {:+.4} — recorded as drift, not a regression",
        drift.diff_pixels, drift.dssim_worst, drift.mean_bias
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
                    "{}: {} differing pixel(s), max channel delta {}, worst-window DSSIM {:.5} \
                     against {}/{} and {:.5} — see {} and {}",
                    scene.name,
                    comparison.diff_pixels,
                    comparison.max_delta,
                    comparison.dssim_worst,
                    scene.policy.tolerance,
                    scene.policy.max_diff_pixels,
                    scene.policy.max_dssim,
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
        Some("graph") => graph(filter),
        Some("verify-gates") => verify_gates(),
        Some("chaos") => chaos(filter),
        _ => anyhow::bail!("usage: gg-golden <run|bless|graph|verify-gates|chaos> [scene|seed]"),
    }
}
