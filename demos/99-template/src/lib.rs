//! **The template** (§6 M12): a spinning lit mesh, and the shortest complete
//! game crate there is. Copy this directory, rename it, start deleting.
//!
//! Everything below the `gg_game!` block is yours. Everything in it is the
//! contract: `components` is what the host registers, `systems` is the tick
//! order (§4.1), and `actions`/`axes` are the id space this crate's
//! `input.toml` binds to (§4.7). There is no framework here — a system is a
//! plain `fn` taking `&mut GameWorld`.
//!
//! # Why this is a demo and not a document
//!
//! `cargo xtask ci` builds it and runs its tests beside every other crate's. A
//! guide written as prose rots the first time an API moves; this one stops
//! compiling instead, which is the entire reason the template *is* the guide.
//!
//! # Three things worth knowing before you delete anything
//!
//! - **`Renderable` is a box, not pack content.** It needs no `assets/` tree and
//!   no `--pack`, which is what keeps this to one file. For a real mesh, declare
//!   [`gg_ecs::boundary::Model`] and put a `.gltf` under `assets/` — demo 04 is
//!   that same step, taken.
//! - **A game that spawns no [`Light`] renders by the ambient term alone.**
//!   Lighting is not a renderer setting a game inherits; it is a component the
//!   game spawns, so forgetting it looks like a bug and is a missing line.
//! - **The spin is a function of the tick count, never of elapsed seconds.** A
//!   float time field is a grep gate (§3) and a determinism break (§1.3): two
//!   machines must agree on tick 900's exact orientation, and `dt` cannot
//!   promise that.
//!
//! Run it: `cargo xtask run 99-template`.

use gg_ecs::Component;
use gg_ecs::boundary::{Eye, GameWorld, Light, Renderable, log_level};
use gg_math::sim;

/// Ticks per full turn. A whole number, so the pose repeats exactly (§1.3).
pub const TICKS_PER_TURN: u64 = 600;
/// Where the camera sits, metres.
pub const EYE_AT: sim::DVec3 = sim::DVec3::new(0.0, 1.0, 4.0);
/// Which way the sun points. Not straight down — a vertical sun casts the one
/// shadow that proves the least.
pub const SUN: sim::Vec3 = sim::Vec3::new(-0.4, -1.0, -0.35);

/// The only sim state there is: how far round the cube has turned.
///
/// A component rather than a process-global because game state lives in the
/// world — that is what survives a reload with the world intact (§4.2.2).
/// Mutable globals and thread-locals in a game crate are a grep gate besides
/// (§3), and this comment is deliberately worded so as not to trip it.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "template.spinner")]
#[repr(C)]
pub struct Spinner {
    /// Ticks since spawn. Integer time (§2) — never a float clock.
    pub tick: u64,
}

/// Cube, sun, camera — once. Re-entrant, because a reload runs it again against
/// a world that already holds them (§6 M5).
pub fn bootstrap(world: &mut GameWorld) {
    let mut exists = false;
    // `each` refuses only an aliasing query (one component named twice), which
    // a single shared read cannot be — so there is no error here to handle.
    let _ = world.each::<&Spinner>(|_, _| exists = true);
    if exists {
        return;
    }
    let cube = world.spawn_with(Spinner { tick: 0 });
    world.put(
        cube,
        Renderable::boxed(sim::DVec3::ZERO, sim::Vec3::splat(0.6), 0x0060_a0ff),
    );
    world.spawn_with(Light::sun(SUN, 0x00ff_f4e0, 3.5));
    world.spawn_with(Eye::at(EYE_AT, 0.0, -0.2));
    world.log(log_level::INFO, "template: spinning");
}

/// Advance the spin and write the pose the renderer reads — one pass, because
/// the state and the protocol component sit on the same entity.
pub fn spin(world: &mut GameWorld) {
    // Two *distinct* components, so this cannot alias either — as above.
    let _ = world.each::<(&mut Spinner, &mut Renderable)>(|_, (spinner, shape)| {
        spinner.tick = spinner.tick.wrapping_add(1);
        let turns = (spinner.tick % TICKS_PER_TURN) as f64 / TICKS_PER_TURN as f64;
        shape.rotation = sim::DQuat::from_axis_angle(sim::DVec3::Y, turns * core::f64::consts::TAU);
    });
}

// Order in `systems` is execution order (§4.1); order in the verb lists is the
// id space a replay records (§4.7). Neither is alphabetical, neither may drift.
#[cfg(feature = "game")]
gg_ecs::gg_game! {
    components: [Spinner, Renderable, Light, Eye],
    actions: [],
    axes: [],
    systems: [bootstrap, spin],
}
