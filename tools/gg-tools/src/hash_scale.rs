//! `gg-tools hash-scale` — what the per-tick full-world passes cost at the scale
//! §6 M38 brings, and which half of the walk is responsible.
//!
//! The question M38 item 6's audit could not answer by reading. `canonical_hash`
//! is the only per-tick pass whose cost is the *whole* world rather than the
//! matched population, and `benches/baseline.txt` prices it at one point only —
//! 108 ns/entity, 100k entities, one archetype. An orbital sim is neither that
//! count nor that shape, and the walk has two costs that a single number cannot
//! separate.
//!
//! Three walks over the same state, each dropping one thing the last one did:
//!
//! - **contract** — `World::canonical_hash`: ascending entity id, with a
//!   `location()` indirection per entity into whatever archetype holds it.
//! - **rowwise** — storage order, absorbed a row at a time with the contract's
//!   own per-entity and per-column commitments, staged into the same block the
//!   contract now stages into. Same blake3 granularity, no sorted walk.
//! - **floor** — the same column bytes absorbed whole through `views_ref`, which
//!   is the shape [`gg_ecs::chunked`] would walk. It commits no entity ids: a
//!   chunk is `(archetype, start, len)` and the ids are implied, which is
//!   exactly what makes it cheaper and exactly why it is layout-locked.
//!
//! The middle leg is the one that makes the report an answer rather than a gap,
//! because the two candidate causes need different work and a contract/floor
//! ratio alone cannot tell them apart. **rowwise landing on the contract** means
//! the bill is update *granularity* — blake3 absorbing 48 bytes at a time — and
//! buffering rows into blocks recovers it while emitting the identical byte
//! stream, which is no contract change at all. **rowwise landing on the floor**
//! means the bill is the sorted walk itself, and that is the contract: §4.2.1
//! sorts by entity id precisely so the hash is layout-independent, so nothing
//! recovers it without changing what the number means.
//!
//! A fourth table prices the answer that came out of the first three: the same
//! rowwise leg staged into blocks before it reaches blake3, swept over block
//! size with **0 meaning unstaged**, so the old behaviour and the new one are in
//! one binary a parameter apart. Its two failure directions are what the shipped
//! [`gg_ecs`] block is read off — below the turn the per-`update` overhead is the
//! bill, above it the staging buffer leaves cache and the copy stops being free —
//! and every leg's digest is asserted identical, which is the whole claim.
//!
//! Both sweeps hold the payload constant so the ratio is comparable across legs:
//! the archetype spread comes from **zero-sized markers**, which add columns and
//! archetypes without adding bytes.
//!
//! Manual, windowless, no device. Nothing here gates — if a number hardens into
//! a threshold it moves to `xtask` (CLAUDE.md).

use std::hint::black_box;
use std::time::{Duration, Instant};

use gg_ecs::{Component, Query, World};
use gg_math::sim;

/// The payload, deliberately the same two components `benches/query.rs` uses so
/// its recorded 108 ns/entity is a point on this instrument's own curve rather
/// than a number from a different experiment. 48 bytes an entity — about what an
/// orbital element set costs, which is the population M38 is sizing for.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "hash-scale-position")]
#[repr(C)]
struct Position {
    p: sim::DVec3,
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "hash-scale-velocity")]
#[repr(C)]
struct Velocity {
    v: sim::DVec3,
}

/// Markers whose *subsets* are the archetypes — the bench's trick, for the same
/// reason: eight declarations buy 256 distinct component sets that one
/// `(Position, Velocity)` query still matches all of. Zero-sized, so a leg with
/// 256 archetypes hashes exactly the bytes a leg with one does and the spread
/// axis moves alone.
macro_rules! markers {
    ($($name:ident = $id:literal),+ $(,)?) => {
        $(
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
            #[component(id = $id)]
            #[repr(C)]
            struct $name;
        )+

        fn mark(world: &mut World, e: gg_ecs::Entity, group: u32) -> anyhow::Result<()> {
            let mut bit = 0u32;
            $(
                if group >> bit & 1 == 1 {
                    world.insert(e, $name)?;
                }
                bit += 1;
            )+
            let _ = bit;
            Ok(())
        }
    };
}

markers!(
    K0 = "hash-scale-k0",
    K1 = "hash-scale-k1",
    K2 = "hash-scale-k2",
    K3 = "hash-scale-k3",
    K4 = "hash-scale-k4",
    K5 = "hash-scale-k5",
    K6 = "hash-scale-k6",
    K7 = "hash-scale-k7",
);

/// The count axis. Past 100k on purpose — that is where `baseline.txt`'s reading
/// stops and where its own "well past 100k" condition for building the
/// accelerator lives.
const COUNTS: &[u32] = &[10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000];

/// The spread axis, at [`SPREAD_AT`] entities throughout. 256 is the bench's
/// scattered leg and is what an editor, a UI and a game with many component
/// combinations all look like; 1 is the contiguous ideal nothing real reaches.
const SPREADS: &[u32] = &[1, 4, 16, 64, 256];
const SPREAD_AT: u32 = 100_000;

/// A demo-sized world, for the block sweep's second population. Demos 10, 11 and
/// 12 are this order and §5.6c hashes them every tick, so a block chosen for a
/// solar system has to be free here or it is not free.
const SMALL_AT: u32 = 200;

/// The block axis, in bytes, **0 meaning unstaged** — one `blake3::update` per
/// commitment, which is what every walk in the tree did before §6 M38 item 9.
/// Up to 4 MiB because the far end has to be past this desk's L2 for the second
/// failure direction to appear at all.
const BLOCKS: &[usize] = &[
    0, 64, 256, 1_024, 4_096, 8_192, 16_384, 65_536, 262_144, 1_048_576, 4_194_304,
];

/// What `gg_ecs::hash::HASH_BLOCK` is, mirrored here because it is `pub(crate)`
/// — an instrument outside the crate cannot name it. It is read off [`BLOCKS`]
/// below, so the duplication is the report's own output and not a second
/// opinion; a change to one without the other shows up as the shipped row
/// sitting off the plateau.
const SHIPPED_BLOCK: usize = 256 * 1024;

/// Passes before any of them counts.
const WARMUP: u32 = 2;

/// A recorded session's length, for the verdict block. Sixty seconds at 60 Hz —
/// demo 10's and demo 12's recorded sessions are this order, and §5.6c's legs
/// replay them with `gg::hash=debug` in three tiers.
const SESSION_TICKS: u64 = 3_600;

pub fn run(args: &[String]) -> anyhow::Result<()> {
    anyhow::ensure!(args.is_empty(), "hash-scale takes no arguments");

    println!(
        "gg-tools hash-scale — the per-tick full-world passes (§6 M38 item 6)\n\
         payload {} bytes/entity (Position+Velocity), best of a time-bounded repeat count\n",
        core::mem::size_of::<Position>() + core::mem::size_of::<Velocity>()
    );

    println!("count sweep — one archetype, the shape `baseline.txt` priced at 100k\n");
    header();
    let mut per_entity_at = Vec::new();
    for &n in COUNTS {
        let world = build(n, 1)?;
        let row = measure(&world, n)?;
        per_entity_at.push((n, row.contract_ns_per_entity));
        print_row(&format!("{n:>9}"), &row);
    }

    println!("\nspread sweep — {SPREAD_AT} entities, split across archetypes\n");
    header();
    let mut spread_rows = Vec::new();
    for &groups in SPREADS {
        let world = build(SPREAD_AT, groups)?;
        let row = measure(&world, SPREAD_AT)?;
        let rows_each = SPREAD_AT / groups;
        print_row(&format!("{groups:>4} arch {rows_each:>6}/each"), &row);
        spread_rows.push((groups, row));
    }

    println!("\nthe other two full passes — {SPREAD_AT} entities, one archetype\n");
    let world = build(SPREAD_AT, 1)?;
    let probe = best(|| {
        black_box(world.entity_hashes());
    });
    let nan = best(|| {
        black_box(world.scan_for_nan());
    });
    println!(
        "  entity_hashes (§5.6 divergence probe)   {:>9.3?}  {:>7.1} ns/entity",
        probe,
        ns_per(probe, SPREAD_AT)
    );
    println!(
        "  scan_for_nan  (§1.13 hazard 6, tier-dev) {:>8.3?}  {:>7.1} ns/entity",
        nan,
        ns_per(nan, SPREAD_AT)
    );

    block_sweep(SPREAD_AT)?;
    // The other direction, and the one a block sized for M38's population would
    // fail in: a demo-sized world hashing every tick under §5.6c.
    block_sweep(SMALL_AT)?;
    decomposition(&spread_rows);
    verdict(&per_entity_at);
    Ok(())
}

/// The block the shipped walks stage into, read off both of its failure
/// directions. Rowwise rather than the contract because the contract's own
/// staging is not a parameter from out here — and rowwise is the leg that
/// isolates granularity anyway, which is the only term a block can move.
fn block_sweep(n: u32) -> anyhow::Result<()> {
    let world = build(n, 1)?;
    let query: Query<(&Position, &Velocity)> = Query::new()?;
    let ids = [
        gg_ecs::component_id::<Position>(),
        gg_ecs::component_id::<Velocity>(),
    ];
    println!("\nblock sweep — {n} entities, one archetype, rowwise staged into blocks\n");
    println!(
        "  {:>10} {:>10} {:>11} {:>9}",
        "block", "rowwise", "ns/entity", "vs 0"
    );
    let expected = rowwise_hash(&world, &query, &ids, 0);
    let mut unstaged = f64::NAN;
    let mut shipped = f64::NAN;
    let mut best_at = (0usize, f64::INFINITY);
    for &block in BLOCKS {
        let d = best(|| {
            black_box(rowwise_hash(&world, &query, &ids, block));
        });
        // The claim the whole item rests on, asserted rather than argued: a
        // block changes when blake3 is told, never what it is told.
        anyhow::ensure!(
            rowwise_hash(&world, &query, &ids, block) == expected,
            "block {block} changed the digest — staging is not supposed to be visible"
        );
        let secs = d.as_secs_f64();
        if block == 0 {
            unstaged = secs;
        }
        if block == SHIPPED_BLOCK {
            shipped = secs;
        }
        if secs < best_at.1 {
            best_at = (block, secs);
        }
        let label = if block == 0 {
            "unstaged".to_string()
        } else if block >= 1024 {
            format!("{} KiB", block / 1024)
        } else {
            format!("{block} B")
        };
        println!(
            "  {label:>10} {:>9.3?} {:>11.1} {:>8.2}x   {}",
            d,
            ns_per(d, n),
            unstaged / secs,
            if block == SHIPPED_BLOCK {
                "shipped"
            } else {
                ""
            }
        );
    }
    // The contract's own before/after cannot be measured from out here — its
    // block is not a parameter — so it is *derived* from same-run numbers, which
    // is sound because staging moves the granularity term and nothing else. It
    // is derived rather than compared against an earlier session on purpose: an
    // untouched pass in this report moves between sessions by more than staging
    // moves the walk.
    let contract = best(|| {
        black_box(world.canonical_hash());
    });
    if unstaged.is_finite() && shipped.is_finite() {
        let recovered = unstaged - shipped;
        println!(
            "\n  contract, derived: {:.2} ms staged against {:.2} unstaged — {:.2}x, \
             the same\n  {:.2} ms of granularity in a walk that also pays for order and bytes",
            contract.as_secs_f64() * 1e3,
            (contract.as_secs_f64() + recovered) * 1e3,
            (contract.as_secs_f64() + recovered) / contract.as_secs_f64(),
            recovered * 1e3,
        );
    }
    println!("\n  digest {expected} on every leg above, asserted");
    println!(
        "  fastest at {} against unstaged — and the far end climbing again is the\n  \
         second direction: a staging buffer past this desk's cache is read back from\n  \
         somewhere colder than it was written, so the copy stops being free.",
        if best_at.0 == 0 {
            "no block".to_string()
        } else {
            format!("{} KiB, {:.2}x", best_at.0 / 1024, unstaged / best_at.1)
        }
    );
    Ok(())
}

/// Split the contract's bill into its three parts, which is the whole reason the
/// middle leg exists. Reported at both ends of the spread axis, because one of
/// the three is the only part that moves along it.
fn decomposition(rows: &[(u32, Row)]) {
    let (Some((first_groups, first)), Some((last_groups, last))) = (rows.first(), rows.last())
    else {
        return;
    };
    println!("\nwhere the contract's time goes — ms per {SPREAD_AT} entities\n");
    println!(
        "  {:<28} {:>12} {:>12}",
        "",
        format!("{first_groups} arch"),
        format!("{last_groups} arch")
    );
    let part = |a: Duration, b: Duration| (a.as_secs_f64() - b.as_secs_f64()) * 1e3;
    println!(
        "  {:<28} {:>12.2} {:>12.2}",
        "order (sorted walk)",
        part(first.contract, first.rowwise),
        part(last.contract, last.rowwise)
    );
    println!(
        "  {:<28} {:>12.2} {:>12.2}",
        "granularity (per-row absorb)",
        part(first.rowwise, first.floor),
        part(last.rowwise, last.floor)
    );
    println!(
        "  {:<28} {:>12.2} {:>12.2}",
        "bytes (irreducible)",
        first.floor.as_secs_f64() * 1e3,
        last.floor.as_secs_f64() * 1e3
    );
}

fn header() {
    println!(
        "  {:<24} {:>10} {:>10} {:>10} {:>10} {:>9}",
        "leg", "contract", "rowwise", "floor", "ns/entity", "c/floor"
    );
}

struct Row {
    contract: Duration,
    rowwise: Duration,
    floor: Duration,
    contract_ns_per_entity: f64,
}

fn print_row(leg: &str, row: &Row) {
    // A zero floor would be a measurement failure, not a free walk; guarded so
    // the report says `inf` rather than dividing by it.
    let ratio = if row.floor.is_zero() {
        f64::INFINITY
    } else {
        row.contract.as_secs_f64() / row.floor.as_secs_f64()
    };
    println!(
        "  {leg:<24} {:>9.3?} {:>9.3?} {:>9.3?} {:>10.1} {ratio:>8.2}x",
        row.contract, row.rowwise, row.floor, row.contract_ns_per_entity
    );
}

fn measure(world: &World, n: u32) -> anyhow::Result<Row> {
    let contract = best(|| {
        black_box(world.canonical_hash());
    });
    let query: Query<(&Position, &Velocity)> = Query::new()?;
    let ids = [
        gg_ecs::component_id::<Position>(),
        gg_ecs::component_id::<Velocity>(),
    ];
    // Staged at the shipped block, so the split below is the *current* walk's
    // and the block sweep is what says what staging bought.
    let rowwise = best(|| {
        black_box(rowwise_hash(world, &query, &ids, SHIPPED_BLOCK));
    });
    let floor = best(|| {
        let mut h = gg_ecs::StateHasher::canonical();
        world.views_ref(query.access(), |view| {
            let count = view.len();
            for i in 0..view.raw_reads().len() {
                let bytes = view.read_bytes(i);
                // Same commitment the fast path makes — count and stride before
                // the bytes — so the floor is a real hash of the same state and
                // not just a memory read wearing its name.
                h.raw_column(count, bytes.len() / count.max(1), bytes);
            }
        });
        black_box(h.finish_canonical());
    });
    Ok(Row {
        contract,
        rowwise,
        floor,
        contract_ns_per_entity: ns_per(contract, n),
    })
}

/// The leg that separates the two candidate causes, and the only reason this
/// instrument could name one. Storage order like the floor, so no `location()`
/// indirection and no sorted walk — but absorbed a *row* at a time with the same
/// per-entity and per-column commitments the contract makes, so blake3 sees the
/// same update granularity it does.
///
/// `block` stages absorbs the way `gg_ecs::hash::Block` does, `0` meaning
/// straight through. The digest is a function of the byte stream and the stream
/// is a function of neither, which the sweep asserts rather than assumes.
fn rowwise_hash(
    world: &World,
    query: &Query<(&Position, &Velocity)>,
    ids: &[gg_ecs::ComponentId],
    block: usize,
) -> gg_ecs::CanonicalHash {
    fn flush(h: &mut gg_ecs::StateHasher, buf: &mut Vec<u8>) {
        if !buf.is_empty() {
            h.bytes(buf);
            buf.clear();
        }
    }
    fn stage(h: &mut gg_ecs::StateHasher, buf: &mut Vec<u8>, block: usize, bytes: &[u8]) {
        // `block == 0` falls through this arm for every slice, which is the
        // unstaged leg and costs it no branch of its own.
        if bytes.len() >= block {
            flush(h, buf);
            h.bytes(bytes);
            return;
        }
        buf.extend_from_slice(bytes);
        if buf.len() >= block {
            flush(h, buf);
        }
    }

    let mut h = gg_ecs::StateHasher::canonical();
    // Grown lazily rather than reserved, which is the shipped shape and the
    // reason a large block is safe: a world smaller than the block allocates
    // only what it uses, so a 200-entity demo hashing every tick does not pay
    // for a buffer sized for a solar system.
    let mut buf: Vec<u8> = Vec::new();
    world.views_ref(query.access(), |view| {
        let count = view.len();
        let entities = view.entities();
        let columns: Vec<&[u8]> = (0..view.raw_reads().len())
            .map(|i| view.read_bytes(i))
            .collect();
        for row in 0..count {
            if let Some(e) = entities.get(row) {
                stage(&mut h, &mut buf, block, &e.to_bits().to_le_bytes());
            }
            for (at, bytes) in columns.iter().enumerate() {
                let stride = bytes.len() / count.max(1);
                if let (Some(id), Some(slice)) =
                    (ids.get(at), bytes.get(row * stride..(row + 1) * stride))
                {
                    stage(&mut h, &mut buf, block, &id.get().to_le_bytes());
                    stage(&mut h, &mut buf, block, slice);
                }
            }
        }
    });
    flush(&mut h, &mut buf);
    h.finish_canonical()
}

/// Best of a repeat count chosen from how long one pass takes: noise on a
/// desktop is one-sided, so the minimum is the reading, and a fixed count would
/// either be noise at 10k or minutes at 1M.
fn best(mut body: impl FnMut()) -> Duration {
    for _ in 0..WARMUP {
        body();
    }
    let start = Instant::now();
    body();
    let once = start.elapsed().max(Duration::from_nanos(1));
    let reps = (Duration::from_millis(400).as_nanos() / once.as_nanos()).clamp(3, 25);
    let mut best = once;
    for _ in 0..reps {
        let start = Instant::now();
        body();
        best = best.min(start.elapsed());
    }
    best
}

fn ns_per(d: Duration, n: u32) -> f64 {
    d.as_secs_f64() * 1e9 / f64::from(n)
}

fn build(n: u32, groups: u32) -> anyhow::Result<World> {
    let mut world = World::new();
    for i in 0..n {
        let e = world.spawn();
        world.insert(
            e,
            Position {
                p: sim::DVec3::new(f64::from(i), 0.0, 0.0),
            },
        )?;
        world.insert(
            e,
            Velocity {
                v: sim::DVec3::new(1.0, 0.5, -0.25),
            },
        )?;
        if groups > 1 {
            // Contiguous blocks rather than round-robin: an archetype's rows are
            // dense either way, but spawn order deciding which archetype a
            // *neighbouring* entity id lands in is the thing the contract's
            // sorted walk pays for, and blocks are the kind case. Round-robin
            // would measure the worst one and call it the shape.
            mark(&mut world, e, i / (n / groups).max(1) % groups)?;
        }
    }
    Ok(world)
}

/// What the numbers mean for the tier that has to run them.
fn verdict(per_entity_at: &[(u32, f64)]) {
    println!("\nreadings\n");
    let first = per_entity_at.first().copied();
    let last = per_entity_at.last().copied();
    if let (Some((n0, ns0)), Some((n1, ns1))) = (first, last) {
        let drift = if ns0 > 0.0 { ns1 / ns0 } else { f64::NAN };
        println!(
            "  linearity: {ns0:.1} ns/entity at {n0} against {ns1:.1} at {n1} — {drift:.2}x.\n           \
             Flat is linear; a climb is the walk leaving cache, which is a\n           \
             property of the world's size and not of the state in it."
        );
    }
    println!(
        "\n  what a gate pays: §5.6c replays a recorded session with `gg::hash=debug`,\n  \
         and `state-hash` is in tier-dev, tier-instrumented and tier-dist-verify. At\n  \
         {SESSION_TICKS} ticks (one minute at 60 Hz) the contract column above is, per leg:"
    );
    for &(n, ns) in per_entity_at {
        let secs = ns * f64::from(n) * SESSION_TICKS as f64 / 1e9;
        println!(
            "    {n:>9} entities  {secs:>8.1} s/leg  {:>8.1} s across three tiers",
            secs * 3.0
        );
    }
    println!(
        "\n  A play session pays none of it: `tracing::debug!` evaluates its fields only\n  \
         when the callsite is live and `LOG_FILTER` is `debug,gg::hash=off`. This is a\n  \
         cost the player never sees and CI always does, which is the opposite of the\n  \
         usual shape and the reason it went unpriced for thirty-five milestones."
    );
    println!(
        "\n  Only one of the three parts above was recoverable *as the same number*, and\n  \
         it is taken: staging emits the identical byte stream, so the digest is bit-\n  \
         identical and it was not a Determinism Contract event. The figures above are\n  \
         the walk with it. The sorted order is the contract itself (§4.2.1 sorts by\n  \
         entity id so the hash is layout-independent), and skipping unchanged chunks —\n  \
         the accelerator `chunked.rs` designs — produces a layout-locked number that\n  \
         its own header says can never be the one a gate reads. So what is left of the\n  \
         bill is the contract's own definition plus the bytes, and the accelerator is\n  \
         for the bisection probe rather than for this table."
    );
}
