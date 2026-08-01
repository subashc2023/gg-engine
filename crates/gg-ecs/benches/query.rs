//! Query micro-benchmarks — the crate's own baseline for future internal
//! rework (§4.2, M3 exit).
//!
//! No benchmark framework: every number here is a *ratio* against a plain
//! `Vec<T>` loop measured in the same process, on the same data, in the same
//! run. That makes timer quality nearly irrelevant and removes a large
//! dependency tree from a crate with a five-dependency budget (§3). Absolute
//! nanoseconds are printed for the record; only the ratios are asserted.
//!
//! The claim under test is §4.2.2's: whole-column views let a query iterate at
//! native speed, so one FFI-shaped call per archetype match is enough and no
//! per-entity boundary crossing is needed. If that is false, M5 is building on
//! sand — which is why this runs two milestones early.
//!
//! Run: `cargo bench -p gg-ecs` (or `cargo xtask bench`, which is how the
//! nightly tier runs it).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use gg_ecs::{Component, Query, QueryAccess, World, component_id};
use gg_math::sim;

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "bench-position")]
#[repr(C)]
struct Position {
    p: sim::DVec3,
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "bench-velocity")]
#[repr(C)]
struct Velocity {
    v: sim::DVec3,
}

/// One archetype, one contiguous run of rows — the shape §4.2.2's hot path
/// claims to reach.
const ENTITIES: u32 = 100_000;
/// Repeats per measurement; the minimum is reported, since noise on a desktop
/// is one-sided.
const REPS: u32 = 25;

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
    let per_entity = best.as_secs_f64() * 1e9 / f64::from(ENTITIES);
    println!("{name:<28} {:>9.3?}  {per_entity:>7.3} ns/entity", best);
    best
}

fn main() {
    let mut positions: Vec<Position> = (0..ENTITIES)
        .map(|i| Position {
            p: sim::DVec3::new(f64::from(i), 0.0, 0.0),
        })
        .collect();
    let velocities: Vec<Velocity> = (0..ENTITIES)
        .map(|_| Velocity {
            v: sim::DVec3::new(1.0, 0.5, -0.25),
        })
        .collect();

    let mut world = World::new();
    for i in 0..ENTITIES {
        let e = world.spawn();
        world.insert(e, positions[i as usize]).unwrap();
        world.insert(e, velocities[i as usize]).unwrap();
    }

    println!("gg-ecs query baseline — {ENTITIES} entities, one archetype, best of {REPS}\n");

    // The reference: two `Vec`s and an index loop. Nothing an ECS does can be
    // faster than this, and the §4.2.2 claim is that it need not be slower.
    let native = measure("native Vec loop", || {
        for (p, v) in positions.iter_mut().zip(&velocities) {
            p.p += v.v;
        }
        black_box(&positions);
    });

    // The boundary shape: one call per archetype, then raw slices.
    let access =
        QueryAccess::new(&[component_id::<Velocity>()], &[component_id::<Position>()]).unwrap();
    let views = measure("column views", || {
        world.views(&access, |mut view| {
            let vel = view.read::<Velocity>(0);
            let pos = view.write::<Position>(0);
            for (p, v) in pos.iter_mut().zip(vel) {
                p.p += v.v;
            }
            black_box(&pos);
        });
    });

    // The ergonomic layer, to price the tuple resolution per row.
    let query = Query::<(&mut Position, &Velocity)>::new().unwrap();
    let typed = measure("typed each()", || {
        world.each(&query, |_, (pos, vel)| {
            pos.p += vel.v;
        });
    });

    // The same query from the *other side of the seam* (§4.2.2, M5's exit
    // criterion). Everything above runs host-side against `World` directly;
    // this goes out through the `repr(C)` host table — `query` fills a page of
    // archetype matches, and the per-entity loop runs on the caller's side over
    // the column pointers that came back. In a shipped build the caller is a
    // dylib; here it is this process, which changes the call's address and
    // nothing about its shape. What it prices is the claim: one call per
    // archetype is enough, so a 100k-entity pass costs one crossing.
    let boundary = {
        // SAFETY: `host_api()` is `&'static`, which is what `init` requires. It
        // is the same handshake `gg_game_init` performs, minus the dylib.
        unsafe { gg_ecs::boundary::init(gg_ecs::boundary::host_api()) };
        let ctx = gg_ecs::boundary::TickCtx {
            tick: 0,
            tick_hz: 60,
            reserved: 0,
            input: gg_ecs::boundary::InputFrame::default(),
            previous: gg_ecs::boundary::InputFrame::default(),
        };
        let handle = gg_ecs::boundary::handle(&mut world);
        // SAFETY: `handle` is this world's, `ctx` outlives the borrow, and the
        // table was installed one line above.
        let mut game = unsafe { gg_ecs::boundary::GameWorld::new(handle, &ctx) };
        measure("boundary each()", || {
            game.each::<(&mut Position, &Velocity)>(|_, (p, v)| {
                p.p += v.v;
            })
            .unwrap();
        })
    };

    // Not a query, but the other per-tick cost the dev loop pays (§4.2.1).
    let hash = measure("canonical_hash", || {
        black_box(world.canonical_hash());
    });

    let ratio = |d: Duration| d.as_secs_f64() / native.as_secs_f64();
    println!(
        "\ncolumn views {:.2}x native, typed each() {:.2}x native, boundary each() {:.2}x native \
         ({:.2}x host each()), canonical_hash {:.2}x native",
        ratio(views),
        ratio(typed),
        ratio(boundary),
        boundary.as_secs_f64() / typed.as_secs_f64(),
        ratio(hash)
    );

    // "Within noise" as a falsifiable bound rather than a feeling. The gap this
    // would catch is structural — a per-entity indirection or bounds-check
    // pattern the native loop does not pay — not a few percent of jitter.
    let mut failures = Vec::new();
    if ratio(views) > 1.25 {
        failures.push(format!(
            "column-view iteration is {:.2}x the native loop; §4.2.2's hot path assumes parity",
            ratio(views)
        ));
    }
    if ratio(typed) > 2.0 {
        failures.push(format!(
            "typed each() is {:.2}x the native loop; the tuple layer is meant to be thin",
            ratio(typed)
        ));
    }
    // The M5 criterion, and the sharper of the two comparisons: measured against
    // the *host-side* query rather than against the native loop, the difference
    // is the boundary and nothing else. A per-entity crossing would show up here
    // as a multiple; one crossing per archetype disappears into 100k rows.
    let seam = boundary.as_secs_f64() / typed.as_secs_f64();
    if seam > 1.15 {
        failures.push(format!(
            "the same query costs {seam:.2}x more through the boundary than host-side; §4.2.2 \
             budgets one FFI call per archetype, and this is what a per-entity crossing looks like"
        ));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
