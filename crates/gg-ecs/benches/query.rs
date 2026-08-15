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
//! # Two shapes, because one of them was invisible
//!
//! The 100k-entity leg is deliberately **one** archetype, which prices the
//! per-*entity* path and says so: "one crossing per archetype disappears into
//! 100k rows". The corollary went unwritten for three milestones — a
//! per-*archetype* cost divided by 100k rows is not measured, it is hidden, and
//! `view::build`'s two heap allocations per matched archetype lived there
//! unpriced until an allocation counter in another crate found them. The
//! scattered leg is the other shape: many archetypes, few rows each, where view
//! construction is most of the work rather than none of it.
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
/// The scattered leg. Rows per archetype is the number that matters: at 16 a
/// pass spends more time constructing views than iterating them, which is the
/// point — a world of small archetypes is what an editor, a UI and a game with
/// many component combinations all actually look like.
const GROUPS: u32 = 256;
const ROWS: u32 = 16;
/// Repeats per measurement; the minimum is reported, since noise on a desktop
/// is one-sided.
const REPS: u32 = 25;
/// Passes before any of them counts — caches and branch predictors.
const WARMUP: u32 = 3;

/// Markers whose *subsets* are the archetypes: entity group `i` carries the
/// marker for every set bit of `i`, so eight declarations buy 256 distinct
/// component sets that a `(Position, Velocity)` query still matches all of.
macro_rules! markers {
    ($($name:ident = $id:literal),+ $(,)?) => {
        $(
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
            #[component(id = $id)]
            #[repr(C)]
            struct $name;
        )+

        fn mark(world: &mut World, e: gg_ecs::Entity, group: u32) {
            let mut bit = 0u32;
            $(
                if group >> bit & 1 == 1 {
                    world.insert(e, $name).unwrap();
                }
                bit += 1;
            )+
            let _ = bit;
        }
    };
}

markers!(
    M0 = "bench-m0",
    M1 = "bench-m1",
    M2 = "bench-m2",
    M3 = "bench-m3",
    M4 = "bench-m4",
    M5 = "bench-m5",
    M6 = "bench-m6",
    M7 = "bench-m7",
);

fn report(name: &str, best: Duration, count: u32) -> Duration {
    let per_entity = best.as_secs_f64() * 1e9 / f64::from(count);
    println!("{name:<28} {:>9.3?}  {per_entity:>7.3} ns/entity", best);
    best
}

fn measure_n(name: &str, count: u32, mut body: impl FnMut()) -> Duration {
    for _ in 0..WARMUP {
        body(); // warm the caches and the branch predictors
    }
    let mut best = Duration::MAX;
    for _ in 0..REPS {
        let start = Instant::now();
        body();
        best = best.min(start.elapsed());
    }
    report(name, best, count)
}

fn measure(name: &str, body: impl FnMut()) -> Duration {
    measure_n(name, ENTITIES, body)
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

    // All four **interleaved**, alternating rep by rep and each keeping its own
    // best, because every claim this leg makes is a *ratio* and the gates are on
    // the ratios. Timed one after another they also measure when the machine was
    // busy: a contended stretch covering the boundary pass and not the host-side
    // one put a ~1.0 seam at 1.27 on the WSL lane against a 1.15 gate nothing
    // had regressed against, and a quiet-desk run put `typed each()` at 1.73x
    // native against a 2.0 gate for the same reason from the other direction.
    // Alternating puts every term in the same weather, which is the only way a
    // quotient prices the code rather than the scheduler. It matters more since
    // the inline columns landed: the host side got ~20% faster, so the seam rose
    // from ~0.8 toward 1.0 and the headroom that used to absorb this is spent.
    //
    // What each one is. The reference is two `Vec`s and an index loop — nothing
    // an ECS does can be faster, and the §4.2.2 claim is that it need not be
    // slower. `column views` is the boundary *shape*: one call per archetype,
    // then raw slices. `typed each()` is the ergonomic layer over it, priced per
    // row for the tuple resolution. `boundary each()` is the same query from the
    // other side of the seam (M5's exit criterion) — out through the `repr(C)`
    // host table, `query` filling a page of archetype matches and the per-entity
    // loop running on the caller's side over the column pointers that came back.
    // In a shipped build that caller is a dylib; here it is this process, which
    // changes the call's address and nothing about its shape. What it prices is
    // the claim that one call per archetype is enough, so a 100k-entity pass
    // costs one crossing.
    let access =
        QueryAccess::new(&[component_id::<Velocity>()], &[component_id::<Position>()]).unwrap();
    let query = Query::<(&mut Position, &Velocity)>::new().unwrap();
    // SAFETY: `host_api()` is `&'static`, which is what `init` requires. It is
    // the same handshake `gg_game_init` performs, minus the dylib.
    unsafe { gg_ecs::boundary::init(gg_ecs::boundary::host_api()) };
    let ctx = gg_ecs::boundary::TickCtx::detached(
        0,
        60,
        gg_ecs::boundary::InputFrame::default(),
        gg_ecs::boundary::InputFrame::default(),
    );
    let (mut native, mut views) = (Duration::MAX, Duration::MAX);
    let (mut typed, mut boundary) = (Duration::MAX, Duration::MAX);
    for rep in 0..WARMUP + REPS {
        let start = Instant::now();
        for (p, v) in positions.iter_mut().zip(&velocities) {
            p.p += v.v;
        }
        black_box(&positions);
        let t_native = start.elapsed();

        let start = Instant::now();
        world.views(&access, |mut view| {
            let vel = view.read::<Velocity>(0);
            let pos = view.write::<Position>(0);
            for (p, v) in pos.iter_mut().zip(vel) {
                p.p += v.v;
            }
            black_box(&pos);
        });
        let t_views = start.elapsed();

        let start = Instant::now();
        world.each(&query, |_, (pos, vel)| {
            pos.p += vel.v;
        });
        let t_typed = start.elapsed();

        // Rebuilt per rep so each borrow of `world` ends inside its own
        // statement, which is what lets these alternate at all. Both are a
        // pointer cast and a few stores, and neither is inside a timed region.
        let handle = gg_ecs::boundary::handle(&mut world);
        // SAFETY: `handle` is this world's, `ctx` outlives the borrow, and the
        // table was installed above.
        let mut game = unsafe { gg_ecs::boundary::GameWorld::new(handle, &ctx) };
        let start = Instant::now();
        game.each::<(&mut Position, &Velocity)>(|_, (p, v)| {
            p.p += v.v;
        })
        .unwrap();
        let t_boundary = start.elapsed();

        if rep >= WARMUP {
            native = native.min(t_native);
            views = views.min(t_views);
            typed = typed.min(t_typed);
            boundary = boundary.min(t_boundary);
        }
    }
    let native = report("native Vec loop", native, ENTITIES);
    let views = report("column views", views, ENTITIES);
    let typed = report("typed each()", typed, ENTITIES);
    let boundary = report("boundary each()", boundary, ENTITIES);

    // Not a query, but the other per-tick cost the dev loop pays (§4.2.1).
    let hash = measure("canonical_hash", || {
        black_box(world.canonical_hash());
    });

    // ---- the scattered shape: per-archetype cost, undivided by 100k rows ----

    let scattered_n = GROUPS * ROWS;
    println!("\nscattered — {GROUPS} archetypes x {ROWS} rows, best of {REPS}\n");

    // The reference keeps the *layout*, not just the arithmetic: one pair of
    // `Vec`s per group, walked group by group. Anything slower than this is the
    // ECS's per-archetype overhead, which is the only thing this leg prices.
    let mut chunks: Vec<(Vec<Position>, Vec<Velocity>)> = (0..GROUPS)
        .map(|_| {
            (
                (0..ROWS)
                    .map(|i| Position {
                        p: sim::DVec3::new(f64::from(i), 0.0, 0.0),
                    })
                    .collect(),
                (0..ROWS)
                    .map(|_| Velocity {
                        v: sim::DVec3::new(1.0, 0.5, -0.25),
                    })
                    .collect(),
            )
        })
        .collect();

    let mut scattered = World::new();
    for group in 0..GROUPS {
        for row in 0..ROWS {
            let e = scattered.spawn();
            scattered
                .insert(
                    e,
                    Position {
                        p: sim::DVec3::new(f64::from(row), 0.0, 0.0),
                    },
                )
                .unwrap();
            scattered
                .insert(
                    e,
                    Velocity {
                        v: sim::DVec3::new(1.0, 0.5, -0.25),
                    },
                )
                .unwrap();
            mark(&mut scattered, e, group);
        }
    }

    // Interleaved for the reason the leg above is, and it needs it more: this is
    // where per-archetype cost lives, so this is where a regression would be —
    // a gate that only fails when the desk happens to be quiet is one that
    // reports the desk.
    let (mut scattered_native, mut scattered_views) = (Duration::MAX, Duration::MAX);
    let mut scattered_typed = Duration::MAX;
    for rep in 0..WARMUP + REPS {
        let start = Instant::now();
        for (pos, vel) in &mut chunks {
            for (p, v) in pos.iter_mut().zip(vel.iter()) {
                p.p += v.v;
            }
        }
        black_box(&chunks);
        let t_native = start.elapsed();

        let start = Instant::now();
        scattered.views(&access, |mut view| {
            let vel = view.read::<Velocity>(0);
            let pos = view.write::<Position>(0);
            for (p, v) in pos.iter_mut().zip(vel) {
                p.p += v.v;
            }
            black_box(&pos);
        });
        let t_views = start.elapsed();

        let start = Instant::now();
        scattered.each(&query, |_, (pos, vel)| {
            pos.p += vel.v;
        });
        let t_typed = start.elapsed();

        if rep >= WARMUP {
            scattered_native = scattered_native.min(t_native);
            scattered_views = scattered_views.min(t_views);
            scattered_typed = scattered_typed.min(t_typed);
        }
    }
    let scattered_native = report("native chunk loop", scattered_native, scattered_n);
    let scattered_views = report("column views", scattered_views, scattered_n);
    let scattered_typed = report("typed each()", scattered_typed, scattered_n);

    let ratio = |d: Duration| d.as_secs_f64() / native.as_secs_f64();
    let scattered_ratio = |d: Duration| d.as_secs_f64() / scattered_native.as_secs_f64();
    println!(
        "\nscattered: column views {:.2}x native, typed each() {:.2}x native",
        scattered_ratio(scattered_views),
        scattered_ratio(scattered_typed)
    );
    println!(
        "\ncolumn views {:.2}x native, typed each() {:.2}x native, boundary each() {:.2}x native \
         ({:.2}x host each()), canonical_hash {:.2}x native",
        ratio(views),
        ratio(typed),
        ratio(boundary),
        boundary.as_secs_f64() / typed.as_secs_f64(),
        ratio(hash)
    );

    // `--json` for `xtask bench --record`, which archives this run as a baseline
    // (§4.11). Emitted before the assertions below so the numbers reach the
    // archive of a run that ends red as well — the reader wants to see how far.
    if std::env::args().any(|a| a == "--json") {
        let ns = |d: Duration| d.as_secs_f64() * 1e9 / f64::from(ENTITIES);
        let scattered_ns = |d: Duration| d.as_secs_f64() * 1e9 / f64::from(scattered_n);
        println!(
            "{{\"entities\":{ENTITIES},\"reps\":{REPS},\"ns_per_entity\":{{\"native\":{:.4},\
             \"column_views\":{:.4},\"typed_each\":{:.4},\"boundary_each\":{:.4},\
             \"canonical_hash\":{:.4}}},\"ratios\":{{\"column_views\":{:.4},\"typed_each\":{:.4},\
             \"boundary_each\":{:.4},\"seam\":{:.4},\"canonical_hash\":{:.4}}},\
             \"scattered\":{{\"groups\":{GROUPS},\"rows\":{ROWS},\"ns_per_entity\":{{\
             \"native\":{:.4},\"column_views\":{:.4},\"typed_each\":{:.4}}},\"ratios\":{{\
             \"column_views\":{:.4},\"typed_each\":{:.4}}}}}}}",
            ns(native),
            ns(views),
            ns(typed),
            ns(boundary),
            ns(hash),
            ratio(views),
            ratio(typed),
            ratio(boundary),
            boundary.as_secs_f64() / typed.as_secs_f64(),
            ratio(hash),
            scattered_ns(scattered_native),
            scattered_ns(scattered_views),
            scattered_ns(scattered_typed),
            scattered_ratio(scattered_views),
            scattered_ratio(scattered_typed)
        );
    }

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
    // The per-archetype bound. Loose against 4.2x measured, because 16 rows
    // cannot amortize much and the number moves with the machine — but tight
    // enough to name the regression it exists for: `view::build` allocating
    // again puts this at 7.6x, which is where it sat until the inline columns
    // landed.
    if scattered_ratio(scattered_views) > 6.0 {
        failures.push(format!(
            "over {GROUPS} archetypes of {ROWS} rows, column-view iteration is {:.2}x the same \
             loop over plain `Vec`s; view construction is supposed to be a few stores, and at \
             this ratio it is allocating or otherwise doing per-archetype work it need not",
            scattered_ratio(scattered_views)
        ));
    }
    // The bound the folded component id bought, and the only gate on this leg's
    // *typed* number — which was printed and ungated while it sat at 17x,
    // because term resolution hashed `DECLARED_ID` per archetype. At 4.1x it is
    // level with the untyped column-view path above it; 8.0 is loose enough for
    // a busy desk and nowhere near the 17x a `component_id` that hashed again
    // would put it back at.
    if scattered_ratio(scattered_typed) > 8.0 {
        failures.push(format!(
            "over {GROUPS} archetypes of {ROWS} rows, typed each() is {:.2}x the same loop over \
             plain `Vec`s; resolving a term per archetype is meant to be a binary search, and at \
             this ratio it is hashing something it could have folded (§4.2)",
            scattered_ratio(scattered_typed)
        ));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
