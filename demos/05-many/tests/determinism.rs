//! §5.6b's third subject: demo 05's canonical hash sequence, compared against
//! one checked-in baseline from Windows, from Linux and from aarch64 under qemu.
//!
//! What this covers that demo 02's replay does not is the **hierarchy**. Every
//! tick here composes five thousand transforms host-side and writes the results
//! into state the hash reads, so a `sin` that rounded differently, a quaternion
//! multiplied in another order, or a propagation order that depended on
//! something other than the tree would all land as a differing hash at a named
//! tick rather than as a picture somebody eventually notices.
//!
//! The baseline is *never* repaired to make CI green (§4.2.1). A mismatch
//! writes the fresh sequence beside it and names the first tick that differs;
//! deciding which of the two is right is a person's job.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use demo_05_many::{FREEZE, MOVE_FORWARD, MOVE_RIGHT};
use gg_ecs::World;
use gg_ecs::boundary::{
    self, AXIS_SCALE, AbiInfo, ComponentsTable, HostApiV1, InputFrame, MAX_AXES, SystemsTable,
    TickCtx, VerbsTable,
};
use gg_scene::Hierarchy;

unsafe extern "C" {
    fn gg_game_abi() -> AbiInfo;
    fn gg_game_init(api: *const HostApiV1);
    fn gg_game_components() -> ComponentsTable;
    fn gg_game_verbs() -> VerbsTable;
    fn gg_game_systems() -> SystemsTable;
}

/// Ticks in the sequence. Long enough to cover bootstrap, a spin-up, a freeze
/// and a thaw — the four regimes the propagation pass has.
const TICKS: usize = 120;

/// The script, as a pure function of the tick — so the sequence is reproducible
/// from this file alone and needs no recorded replay beside it.
///
/// A *script* rather than a `.ggreplay` because there is no player here to
/// record: demo 02 owns the recorded-input half of §5.6, and duplicating its
/// format would prove the format twice and the hierarchy once.
fn scripted(tick: usize) -> InputFrame {
    let mut axes = [0; MAX_AXES];
    // Move for a while, so the eye — and therefore nothing the hierarchy does —
    // changes underneath the same spinning field.
    if (10..40).contains(&tick) {
        axes[MOVE_FORWARD.index()] = -AXIS_SCALE;
        axes[MOVE_RIGHT.index()] = AXIS_SCALE / 2;
    }
    // Freeze at 60, thaw at 90: `just_pressed` is an edge, so each is one tick
    // down and the next up.
    let buttons = u64::from(tick == 60 || tick == 90) << FREEZE.index();
    InputFrame { buttons, axes }
}

/// Run the sequence and return one canonical hash per tick.
fn sequence() -> Vec<u128> {
    // SAFETY: the symbols are this binary's own, linked from the rlib.
    let info = unsafe { gg_game_abi() };
    assert_eq!(info, boundary::abi_info(), "one build, one description");
    // SAFETY: `host_api()` returns a `&'static` table (§4.2.2).
    unsafe { gg_game_init(core::ptr::from_ref(boundary::host_api())) };
    // SAFETY: this binary's own table, static for the process.
    let _ = unsafe { gg_game_verbs() };

    let mut world = World::new();
    // SAFETY: the tables are this binary's own and live for the process.
    let table = unsafe {
        world.adopt(&gg_game_components()).unwrap();
        gg_game_systems()
    };
    let mut hierarchy = Hierarchy::new();
    let mut previous = InputFrame::default();
    let mut hashes = Vec::with_capacity(TICKS);
    for tick in 0..TICKS {
        let input = scripted(tick);
        let ctx = TickCtx {
            tick: tick as u64,
            tick_hz: 60,
            reserved: 0,
            input,
            previous,
        };
        // SAFETY: the table is this binary's own and `ctx` outlives the call.
        unsafe { world.run_systems(&table, &ctx) }.expect("no system panicked");
        // Before the hash, exactly where the shell runs it (§4.7) — which is
        // what puts the composed transforms inside what is being compared.
        hierarchy
            .propagate(&mut world)
            .expect("the hierarchy resolves");
        hashes.push(world.canonical_hash().get());
        previous = input;
    }
    hashes
}

fn parse(text: &str) -> Vec<u128> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| u128::from_str_radix(line, 16).expect("baseline is hex, one hash per line"))
        .collect()
}

#[test]
fn the_canonical_hash_sequence_matches_every_architecture() {
    let expected = parse(include_str!("hashes.txt"));
    let actual = sequence();
    if actual == expected {
        assert_eq!(
            actual.len(),
            TICKS,
            "the sequence is not {TICKS} ticks long"
        );
        return;
    }

    // Written beside the baseline rather than over it (§4.2.1): a divergence is
    // a Determinism Contract event, and repairing the file is how one gets
    // buried instead of investigated.
    let fresh: String = actual.iter().map(|h| format!("{h:032x}\n")).collect();
    // Beside the baseline, by manifest path rather than by working directory:
    // a test runner's cwd is its own business and this file has to be findable.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/hashes.actual.txt");
    let _ = std::fs::write(&path, &fresh);
    let first = actual
        .iter()
        .zip(&expected)
        .position(|(a, b)| a != b)
        .unwrap_or(expected.len().min(actual.len()));
    panic!(
        "canonical hash diverged first at tick {first}: {:032x} vs baseline {:032x}\nfresh \
         sequence written to {} — the baseline is not repaired to make CI green (§4.2.1)",
        actual.get(first).copied().unwrap_or_default(),
        expected.get(first).copied().unwrap_or_default(),
        path.display()
    );
}

#[test]
fn the_sequence_is_the_same_however_often_it_is_run() {
    // Cheap, and it separates "this machine disagrees with the baseline" from
    // "this machine disagrees with itself" — two very different bugs.
    assert_eq!(sequence(), sequence());
}

#[test]
fn the_hierarchy_is_inside_what_the_hash_covers() {
    // The claim the whole file rests on: if propagation were downstream of the
    // hash, this sequence would be blind to every composed transform in it.
    let with = sequence();
    let mut world = World::new();
    // SAFETY: this binary's own tables, static for the process.
    let table = unsafe {
        gg_game_init(core::ptr::from_ref(boundary::host_api()));
        world.adopt(&gg_game_components()).unwrap();
        gg_game_systems()
    };
    let mut previous = InputFrame::default();
    let mut without = Vec::with_capacity(TICKS);
    for tick in 0..TICKS {
        let input = scripted(tick);
        let ctx = TickCtx {
            tick: tick as u64,
            tick_hz: 60,
            reserved: 0,
            input,
            previous,
        };
        // SAFETY: as above.
        unsafe { world.run_systems(&table, &ctx) }.expect("no system panicked");
        without.push(world.canonical_hash().get());
        previous = input;
    }
    assert_ne!(
        with, without,
        "propagating changed no hashed state — the gate would be watching nothing"
    );
}
