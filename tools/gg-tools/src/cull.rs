//! `gg-tools cull` — what giving a batch bounds bought the shadow passes, and
//! where (§6 M32).
//!
//! The subject is **pack geometry**, and that is the whole point of a separate
//! instrument. `lamps` and `lights` price a corridor of `Renderable` boxes,
//! where each box is already its own draw and has been culled per cascade since
//! §6 M15.3 and per lamp face since §6 M31 — a box scene would report that this
//! milestone changed nothing, and be right about the scene while being wrong
//! about the engine. What §6 M32 fixed is the *pack* pass, where instances are
//! sorted into batches and a batch could not be rejected because it had no
//! bounds. So this runs demo 05's ten thousand parented objects, which is that
//! case at its worst: many instances, few meshes, and therefore few enormous
//! batches.
//!
//! The A/B is `r.shadow_cull`, and it is not an approximation of the old
//! behaviour — it *is* the old behaviour, exactly. With the switch off, `fit`
//! answers `Inside` for everything, every batch is drawn whole into every view,
//! and nothing is compacted; that is the code this milestone replaced, still
//! reachable, in the same binary, one flag away. Two builds would have compared
//! two compilers as much as two algorithms.
//!
//! What is printed per row is the frame either way, the **shadow passes' own**
//! device time either way — the number the cull actually moves, as against a
//! frame time that is mostly the forward pass — and what the cull rejected, so
//! that a row where nothing improved can be told apart from a row where nothing
//! was rejected.

use gg_ecs::World;
use gg_ecs::boundary::{Light, Model, Node};
use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenRenderer, View, cvars};

/// A real frame: the shadow passes are geometry-bound, so the extent matters
/// less here than in `lamps` — but a frame nobody would render is a frame worth
/// no one's time.
const EXTENT: (u32, u32) = (1280, 720);

const WARMUP: usize = 3;
const FRAMES: usize = 11;

/// Casting lamp budgets swept. Zero is the sun alone — four cascades, which is
/// the case every pack scene has had since §6 M15.3 — and the rest add six
/// views each on top of it.
const BUDGETS: [i64; 3] = [0, 1, 4];

/// Where the eye sits over demo 05's field, and what it looks at. Chosen so the
/// frame holds a fraction of the field rather than all of it: a camera framing
/// everything has nothing off screen, and a cull's whole business is what is off
/// screen.
const EYE: [f64; 3] = [18.0, 9.0, 26.0];
const YAW: f32 = -0.5;
const PITCH: f32 = -0.32;

const PACK: &str = "target/assets/05-many.ggpack";

/// Demo 05's world, plus however many lamps this row asks to cast.
///
/// The lamps are placed over the field rather than at the eye, because a lamp
/// whose faces see nothing prices six empty passes and would make the cull look
/// free.
fn field(lamps: usize) -> anyhow::Result<World> {
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
    // Strung along the line from the eye to the middle of the field, because a
    // light is culled by the *view* frustum before it is ever considered for
    // casting: lamps placed by hub index landed off screen and silently reduced
    // the sweep to four views, which the `views` column is printed to catch.
    for index in 0..lamps {
        let t = 0.25 + 0.18 * index as f64;
        let lamp = world.spawn();
        world.insert(
            lamp,
            Light::point(
                sim::DVec3::new(EYE[0] * (1.0 - t), 4.0, EYE[2] * (1.0 - t)),
                0x00ff_e8c0,
                20.0,
                8.0,
            ),
        )?;
    }
    Ok(world)
}

/// One row's measurement: median frame ms, the shadow passes' device ms, and
/// what the cull came to.
struct Row {
    frame_ms: f64,
    shadow_ms: f64,
    draws: usize,
    cull: gg_render::ShadowCull,
}

/// Every pass this milestone can touch: the four cascades and the lamp atlas.
///
/// Matched by prefix rather than by an exact list, because the cascade passes
/// are named per cascade and a fifth would otherwise go silently unmeasured.
fn is_shadow(name: &str) -> bool {
    name.starts_with("shadow") || name.starts_with("scene.lamps") || name.contains("lamp-shadows")
}

/// Stream the pack in and resolve the hierarchy before any clock starts: the
/// first frames of any pack are a *load* (§4.6), and `Node` parenting is not
/// resolved until `Hierarchy::propagate` has run.
fn settle(renderer: &mut OffscreenRenderer, world: &World) -> anyhow::Result<()> {
    let view = View {
        yaw: YAW,
        pitch: PITCH,
        ..View::default()
    };
    let eye = sim::DVec3::new(EYE[0], EYE[1], EYE[2]);
    let mut extracted = Extracted::default();
    for _ in 0..12 {
        extracted.clear(eye, view.frustum(EXTENT));
        extracted.append_lights(world)?;
        extracted.cast_shadows(view.caster_reach(EXTENT));
        extracted.append_models::<Model>(world, renderer.scenes())?;
        renderer.frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])?;
    }
    let pending = renderer
        .pack()
        .map_or(0, gg_render::content::Content::pending);
    anyhow::ensure!(pending == 0, "{pending} asset(s) still streaming");
    Ok(())
}

fn measure(renderer: &mut OffscreenRenderer, world: &World) -> anyhow::Result<Row> {
    let view = View {
        yaw: YAW,
        pitch: PITCH,
        ..View::default()
    };
    let eye = sim::DVec3::new(EYE[0], EYE[1], EYE[2]);
    let mut extracted = Extracted::default();
    let mut times: Vec<f64> = Vec::new();
    let mut shadows: Vec<f64> = Vec::new();
    let mut row = (0, gg_render::ShadowCull::default());
    for frame in 0..WARMUP + FRAMES {
        extracted.clear(eye, view.frustum(EXTENT));
        extracted.append_lights(world)?;
        // The sweep the shadow pass needs: without it, extract culls casters to
        // the view frustum and the cascades never see what is behind the eye.
        extracted.cast_shadows(view.caster_reach(EXTENT));
        extracted.append_models::<Model>(world, renderer.scenes())?;
        let started = std::time::Instant::now();
        renderer.frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])?;
        if frame >= WARMUP {
            times.push(started.elapsed().as_secs_f64() * 1e3);
            let shadow: f64 = renderer
                .pass_timings()
                .iter()
                .filter(|t| is_shadow(&t.name))
                .map(|t| f64::from(t.gpu_ms))
                .sum();
            shadows.push(shadow);
            row = (renderer.draw_counts().1, renderer.shadow_cull());
        }
    }
    times.sort_by(f64::total_cmp);
    shadows.sort_by(f64::total_cmp);
    Ok(Row {
        frame_ms: times[times.len() / 2],
        shadow_ms: shadows[shadows.len() / 2],
        draws: row.0,
        cull: row.1,
    })
}

pub fn run(_args: &[String]) -> anyhow::Result<()> {
    let pack = std::path::PathBuf::from(PACK);
    anyhow::ensure!(
        pack.is_file(),
        "{} is not there — `cargo xtask assets` compiles it (§4.6)",
        pack.display()
    );

    println!("gg-tools cull — demo 05's field, {}x{}", EXTENT.0, EXTENT.1);
    println!(
        "  ten thousand instances of {} meshes: few batches, each enormous — the case §6 M32 fixed",
        demo_05_many::MESHES.len()
    );
    println!();
    println!(
        "{:>6}  {:>6}  {:>17}  {:>17}  {:>9}  {:>19}",
        "lamps", "views", "frame ms  (cull)", "shadow ms (cull)", "draws", "rejected/total"
    );
    println!(
        "{:>6}  {:>6}  {:>8} {:>8}  {:>8} {:>8}  {:>4} {:>4}  {:>19}",
        "", "", "off", "on", "off", "on", "off", "on", ""
    );

    for budget in BUDGETS {
        cvars::LAMPS.set_int(budget);
        cvars::LAMP_SHADOWS.set_bool(budget > 0);
        let mut world = field(budget as usize)?;
        // `Node` parenting is resolved by the scene graph, not by extract: the
        // children sit at the origin until this has run, and a field of ten
        // thousand objects stacked on one point would bound to one small sphere
        // and make the cull look extraordinary.
        gg_scene::Hierarchy::new().propagate(&mut world)?;

        let mut renderer = OffscreenRenderer::new(EXTENT)?;
        renderer.open_pack(&pack)?;
        settle(&mut renderer, &world)?;
        // Off first: it is the *old* behaviour, and reading the baseline before
        // the change is the order a measurement should be taken in.
        cvars::SHADOW_CULL.set_int(0);
        let off = measure(&mut renderer, &world)?;
        cvars::SHADOW_CULL.set_int(1);
        let on = measure(&mut renderer, &world)?;
        renderer.shutdown();

        let total = on.cull.drawn + on.cull.rejected;
        println!(
            "{:>6}  {:>6}  {:>8.2} {:>8.2}  {:>8.2} {:>8.2}  {:>4} {:>4}  {:>8}/{:<10}",
            budget,
            on.cull.views,
            off.frame_ms,
            on.frame_ms,
            off.shadow_ms,
            on.shadow_ms,
            off.draws,
            on.draws,
            on.cull.rejected,
            total
        );
        // The control that stops a green row from meaning nothing: with the
        // switch off, nothing may be rejected. A run where both columns read the
        // same is measuring one algorithm twice.
        anyhow::ensure!(
            off.cull.rejected == 0,
            "the off switch culled {} instances — the A/B is not an A/B",
            off.cull.rejected
        );
    }

    println!();
    println!(
        "  `draws` is the pack pass's batch count and is the same either way — what the cull \n  \
         removes is (batch, view) pairs in the shadow passes, which no frame-level count shows."
    );
    Ok(())
}
