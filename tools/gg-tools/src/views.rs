//! `gg-tools views` — every intermediate the frame has, as a picture (§6 M59).
//!
//! The debug view is a renderer feature and this is its harness: it drives one
//! scene once per entry in [`cvars::DEBUG_VIEWS`], writes each readback out, and
//! prints which views the frame actually had.
//!
//! **That last column is the report, not the pictures.** A view resolves to
//! `None` exactly when the pass behind it did not run, so the table is a list of
//! what this frame's graph contained — a third cascade in a two-cascade scene, a
//! lamp atlas in a scene with no casting point light, the field with `r.gi` off.
//! An operator looking at a suspect frame wants that before any image.
//!
//! It is also the only automated thing that ever *executes* the debug pass on a
//! real device. The pass writes the backbuffer, so a golden cannot hold it
//! without holding a picture of the AO buffer as a reference; the offscreen test
//! in `gg-render` proves each view declares and submits clean, and this proves
//! each one produces something a human can look at.

use std::path::PathBuf;

use anyhow::Result;
use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenRenderer, View, cvars};

use crate::field;

/// Large enough to read a crease in, small enough that thirteen of them are a
/// few megabytes.
const EXTENT: (u32, u32) = (1280, 720);

/// The field converges over `probes / r.gi_rate` frames and a view of it taken
/// before that is a picture of a transient (§6 M57) — so every view waits, not
/// just the field's, because they must all describe the same frame.
pub(crate) const WARMUP: usize = 40;

/// `frame`'s chair, so the three instruments describe the same room from the
/// same place.
const PITCH: f32 = -0.22;

/// What is being looked at. Demo 12's room needs no arguments and is the
/// default; anything else is a pack, which is the shape §6 M59 wanted for
/// Sponza and which costs nothing to leave general.
pub(crate) struct Scene {
    pub(crate) world: gg_ecs::World,
    /// The pack to open, and the label the header prints.
    pub(crate) pack: Option<(PathBuf, String)>,
    pub(crate) eye: sim::DVec3,
    pub(crate) view: View,
}

/// The value after `name` in `args`, if it is there.
pub(crate) fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// `--set r.ao_radius=4` and so on, repeatable, applied in order.
///
/// Registers first: the table is populated by the shell in a session, and
/// nothing here is a shell, so `find` sees an empty one until it runs.
pub(crate) fn apply_sets(args: &[String]) -> Result<()> {
    gg_render::cvars::register()?;
    for (i, arg) in args.iter().enumerate() {
        if arg != "--set" {
            continue;
        }
        let text = args.get(i + 1).map_or("", String::as_str);
        let (name, value) = text
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("--set wants `name=value`, got {text:?}"))?;
        let cvar =
            gg_core::cvar::find(name).ok_or_else(|| anyhow::anyhow!("no cvar named {name:?}"))?;
        cvar.set_from_str(value, gg_core::cvar::CVarSource::Cli)?;
        println!("  set {name} = {}", cvar.to_text());
    }
    Ok(())
}

/// `--pack`/`--scene`/`--eye`/`--yaw`, or demo 12's room when none is given.
pub(crate) fn scene_from(args: &[String]) -> Result<Scene> {
    match flag(args, "--pack") {
        Some(pack) => from_pack(
            PathBuf::from(pack),
            flag(args, "--scene").unwrap_or_else(|| "Sponza/scene".to_owned()),
            flag(args, "--eye").as_deref(),
            flag(args, "--yaw").as_deref(),
        ),
        None => Ok(Scene {
            world: field::world()?,
            pack: None,
            eye: sim::DVec3::new(0.0, 1.62, 8.0),
            view: View {
                pitch: PITCH,
                ..View::default()
            },
        }),
    }
}

/// The scene's name for a header, opening its pack on the way.
pub(crate) fn open(renderer: &mut OffscreenRenderer, scene: &Scene) -> Result<String> {
    Ok(match &scene.pack {
        Some((path, name)) => {
            renderer.open_pack(path)?;
            format!("{name} from {}", path.display())
        }
        None => "demo 12's room".to_owned(),
    })
}

pub fn run(args: &[String]) -> Result<()> {
    let extent = match flag(args, "--extent") {
        Some(text) => parse_extent(&text)?,
        None => EXTENT,
    };
    apply_sets(args)?;
    let scene = scene_from(args)?;
    let mut renderer = OffscreenRenderer::new(extent)?;
    let label = open(&mut renderer, &scene)?;
    let out = PathBuf::from("target/gg-tools");
    std::fs::create_dir_all(&out)?;
    println!(
        "{label} at {}x{} on {}\n",
        extent.0,
        extent.1,
        renderer.device().chosen
    );

    // Warmed with the view *off*, so what every leg below renders is one settled
    // frame rather than thirteen differently-aged ones. A pack streams, so the
    // warmup is also what makes the first view a picture of resident content
    // rather than of the fallback (§6 M36's note, one instrument along).
    cvars::DEBUG_VIEW.set_int(0);
    for _ in 0..WARMUP {
        let extracted = extract(&scene, extent, renderer.scenes())?;
        renderer.frame(&extracted, &scene.view, [0.0; 4], &[])?;
    }
    if let Some(pending) = renderer.pack().map(gg_render::content::Content::pending) {
        anyhow::ensure!(
            pending == 0,
            "{pending} asset(s) still streaming after {WARMUP} frames — every view below would \
             be a picture of the fallback rather than of the scene"
        );
    }

    println!("  view       | in this frame | file");
    for (index, name) in cvars::DEBUG_VIEWS.iter().enumerate().skip(1) {
        cvars::DEBUG_VIEW.set_int(index as i64);
        let extracted = extract(&scene, extent, renderer.scenes())?;
        let frame = renderer.frame(&extracted, &scene.view, [0.0; 4], &[])?;
        let ran = frame.order.iter().any(|pass| pass == "debug-view");
        let file = match ran {
            false => "-".to_owned(),
            true => {
                let path = out.join(format!("view-{name}.png"));
                write_png(&frame.pixels, extent, &path)?;
                path.display().to_string()
            }
        };
        println!(
            "  {name:<10} | {:<13} | {file}",
            if ran { "yes" } else { "no pass" }
        );
    }
    cvars::DEBUG_VIEW.set_int(0);

    let report = renderer.shutdown();
    anyhow::ensure!(report.clean(), "unclean render: {report:?}");
    Ok(())
}

/// One `Model` from a pack, a sun and a sky — the smallest world that shows an
/// authored scene, and deliberately not demo 14's own: this crate takes no
/// dependency on a game, and what a scene viewer needs from one is a placement.
fn from_pack(pack: PathBuf, scene: String, eye: Option<&str>, yaw: Option<&str>) -> Result<Scene> {
    use gg_ecs::boundary::{Light, Model, Sky};
    let mut world = gg_ecs::World::new();
    world.register::<Model>()?;
    world.register::<Light>()?;
    world.register::<Sky>()?;
    let model = world.spawn();
    world.insert(model, Model::at(&scene, sim::DVec3::ZERO))?;
    let sun = world.spawn();
    world.insert(
        sun,
        Light::sun(sim::Vec3::new(-0.55, -0.72, -0.42), 0x00ff_f0d8, 6.0),
    )?;
    let sky = world.spawn();
    world.insert(sky, Sky::daylight(0.6))?;
    let eye = match eye {
        Some(text) => parse_eye(text)?,
        None => sim::DVec3::new(-9.0, 1.7, 0.0),
    };
    let yaw = match yaw {
        Some(text) => text.trim().parse()?,
        None => -core::f32::consts::FRAC_PI_2,
    };
    Ok(Scene {
        world,
        pack: Some((pack, scene)),
        eye,
        view: View {
            yaw,
            ..View::default()
        },
    })
}

/// The shell's extract order (`App::extract`), as `frame` and `field` restate
/// it: a different visible set is a different frame.
pub(crate) fn extract(
    scene: &Scene,
    extent: (u32, u32),
    scenes: &dyn gg_extract::Scenes,
) -> Result<Extracted> {
    let mut extracted = Extracted::default();
    extracted.clear(scene.eye, scene.view.frustum(extent));
    extracted.append_lights(&scene.world)?;
    extracted.cast_shadows(scene.view.caster_reach(extent));
    extracted.append::<gg_ecs::boundary::Renderable>(&scene.world)?;
    // `append_models`, not `append`: a `Model` is a `SimTransform` like any
    // other, so the plain call compiles and draws the *box* the fallback
    // geometry is -- which is what a Sponza that renders as one grey cube looks
    // like, and what this comment is here to stop somebody rediscovering.
    extracted.append_models::<gg_ecs::boundary::Model>(&scene.world, scenes)?;
    Ok(extracted)
}

// Trimmed, every one of them: a shell that needs `--yaw " -1.9"` to keep a
// leading minus off the flag parser hands the space through, and `f32::parse`
// rejects it as an invalid literal — which is a run that dies after the pack has
// loaded, for a reason that reads like the number was wrong.
fn parse_eye(text: &str) -> Result<sim::DVec3> {
    let parts: Vec<&str> = text.split(',').collect();
    let [x, y, z] = parts.as_slice() else {
        anyhow::bail!("--eye wants `x,y,z`, got {text:?}");
    };
    Ok(sim::DVec3::new(
        x.trim().parse()?,
        y.trim().parse()?,
        z.trim().parse()?,
    ))
}

pub(crate) fn write_png(pixels: &[u8], extent: (u32, u32), path: &std::path::Path) -> Result<()> {
    let file = std::fs::File::create(path)?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), extent.0, extent.1);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(pixels)?;
    Ok(())
}

fn parse_extent(text: &str) -> Result<(u32, u32)> {
    let (w, h) = text
        .split_once('x')
        .ok_or_else(|| anyhow::anyhow!("--extent wants `<w>x<h>`, got {text:?}"))?;
    Ok((w.parse()?, h.parse()?))
}
