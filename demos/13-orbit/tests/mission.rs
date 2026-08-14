//! The mission, flown by the script that records it (§6 M38 item 14).
//!
//! `tests/game.rs` is about the two regimes; this is about the *flight* — the
//! whole of it, parking orbit to capture, through the declared systems table.
//! What it is really guarding is that `session::FLOWN` and the target's phase
//! still agree: they were solved against each other by `gg-tools transfer`, and
//! the window is one cell wide, so anything that moves the departure or the
//! target's elements breaks the mission rather than degrading it.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use demo_13_orbit::session::{self, Entry, Plan};
use demo_13_orbit::{
    CAPTURED, EVENT_CUT, EVENT_HANDOVER, EVENT_LIT, EVENT_OUTCOME, FLYING, NODE_HOLD, NODE_LEAD,
    OCHRE, TANK, VERGE, WARPS,
};
use gg_ecs::boundary::{AbiInfo, ComponentsTable, HostApiV1, SystemsTable};

// The symbols `gg_game!` exported into this crate's rlib.
unsafe extern "C" {
    fn gg_game_abi() -> AbiInfo;
    fn gg_game_init(api: *const HostApiV1);
    fn gg_game_components() -> ComponentsTable;
    fn gg_game_systems() -> SystemsTable;
}

fn entry() -> Entry {
    Entry {
        abi: gg_game_abi,
        init: gg_game_init,
        components: gg_game_components,
        systems: gg_game_systems,
    }
}

/// Where the target's pull beats the star's by the handover ratio — the width
/// of the window the crossing has to land in, restated here because a test that
/// took it from the game would agree with any value the game happened to hold.
const GRIP: f64 = 9.15e7;

#[test]
fn the_mission_flies_to_a_capture() {
    let flight = session::fly(&entry(), session::FLOWN).expect("the pilot flew");
    assert_eq!(
        flight.outcome, CAPTURED,
        "the flown plan no longer captures: closest {:.4e} m against a {GRIP:.2e} m grip",
        flight.approach
    );
    let periapsis = flight.periapsis.expect("an intercept has a periapsis");
    assert!(
        periapsis > OCHRE.radius,
        "the crossing arrived inside the planet: {periapsis:.4e} m against a radius of {:.4e}",
        OCHRE.radius
    );
    assert!(
        flight.approach < GRIP,
        "an intercept means the ship got inside the grip: {:.4e}",
        flight.approach
    );
    assert!(
        flight.spent < TANK,
        "the mission has to be affordable: {:.0} m/s of {TANK:.0}",
        flight.spent
    );
    // The transfer is a real crossing and not a fall: its apoapsis reaches the
    // target's orbit and its periapsis is still near the departure planet's.
    let transfer = flight.transfer.expect("the star took the conic");
    let orbit = transfer.orbit;
    assert!(orbit.eccentricity < 1.0);
    assert!(
        orbit.semi_major * (1.0 + orbit.eccentricity)
            > OCHRE.semi_major * (1.0 - OCHRE.eccentricity),
        "the transfer never reaches the target's orbit"
    );
    assert!((orbit.semi_major * (1.0 - orbit.eccentricity) - VERGE.semi_major).abs() < 1.0e10);
}

/// The flight log, in order — §2's event queue with a whole mission in it.
#[test]
fn the_flight_log_tells_the_mission_in_order() {
    let flight = session::fly(&entry(), session::FLOWN).expect("the pilot flew");
    let all = session::progress(&entry(), &flight.frames).expect("driven");
    // Everything but the closing restart, which despawns the ship and would
    // read as a third handover to nowhere.
    let progress = &all[..all.len() - 1];

    // The regimes, as ticks: parked about the departure planet, out to the
    // star, caught by the target, and exactly one of each crossing.
    let first = |find: &dyn Fn(&session::Progress) -> bool| progress.iter().position(find);
    let escaped = first(&|p| p.primary == 0).expect("the star took it");
    let intercept = first(&|p| p.primary == OCHRE.index).expect("the target took it");
    let captured = first(&|p| p.outcome == CAPTURED).expect("captured");
    assert!(
        escaped < intercept && intercept < captured,
        "escaped {escaped}, intercept {intercept}, captured {captured}"
    );
    assert!(
        progress[..escaped].iter().all(|p| p.primary == VERGE.index),
        "the mission opens parked about the departure planet"
    );

    // Two handovers and no more: a conic that chatters across the ratio would
    // show up here as four, and the hysteresis is what stops it.
    let handovers = (1..progress.len())
        .filter(|&i| progress[i].primary != progress[i - 1].primary)
        .count();
    assert_eq!(handovers, 2, "one crossing out, one crossing in");

    // The engine is lit exactly twice — the departure and the capture — and the
    // world is on rails everywhere else, which is what makes warp reachable for
    // all but 1578 of the mission's ticks.
    let lightings = (1..progress.len())
        .filter(|&i| progress[i].lit && !progress[i - 1].lit)
        .count();
    assert_eq!(lightings, 2, "one burn out, one burn in");
    let coasting = progress.iter().filter(|p| !p.lit).count();
    assert!(
        coasting * 2 > progress.len(),
        "most of a mission is a coast: {coasting} of {}",
        progress.len()
    );
}

/// The events the world holds, as kinds in the order they landed.
#[test]
fn the_events_are_the_mission_as_rows() {
    let flight = session::fly(&entry(), session::FLOWN).expect("the pilot flew");
    let progress = session::progress(&entry(), &flight.frames).expect("driven");
    // Lit, cut, escaped, intercept, lit, cut, outcome — seven, and then the
    // restart on the last frame takes the log with it.
    let peak = progress.iter().map(|p| p.events).max().expect("ticks");
    assert_eq!(peak, 7, "the whole mission is seven rows");
    assert_eq!(
        progress.last().expect("ticks").events,
        0,
        "the closing restart clears the log with the world"
    );
    // The kinds exist as constants and the log is what orders them; naming them
    // here is what makes the count above mean something.
    assert_ne!(EVENT_LIT, EVENT_CUT);
    assert_ne!(EVENT_HANDOVER, EVENT_OUTCOME);
}

/// The same stream twice is the same world, tick for tick — §5.6 through this
/// crate's own tables, before any shell is involved.
#[test]
fn the_stream_hashes_the_same_twice() {
    let entry = entry();
    let frames = session::frames(&entry).expect("the mission");
    let first = session::hash_sequence(&entry, &frames).expect("driven");
    let second = session::hash_sequence(&entry, &frames).expect("driven again");
    assert_eq!(first.len(), frames.len());
    assert!(
        session::divergence(
            &first,
            &second.iter().map(|h| (0, h.get())).collect::<Vec<_>>()
        )
        .is_none(),
        "two runs of one stream are one world"
    );
}

/// §5.6's material claim for this demo: the mission's per-tick canonical hashes
/// match the checked-in baseline, on every architecture the leg runs — and here
/// that is a claim about a *transcendental*, because Kepler's equation is
/// solved by `gg_math::sim`'s `sin`/`cos` on every propagation of every conic.
#[test]
fn the_recorded_mission_reproduces_its_checked_in_hash_sequence() {
    let entry = entry();
    let sequence = session::hash_sequence(&entry, &session::frames(&entry).unwrap()).unwrap();
    let path = session::baseline_path();
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "no baseline at {} ({e}) — run `cargo xtask replay --bless`",
            path.display()
        )
    });
    let baseline = session::parse_baseline(&text).unwrap();
    if let Some(found) = session::divergence(&sequence, &baseline) {
        let actual = std::env::temp_dir().join("demo13-orbit.hashes.actual");
        let _ = std::fs::write(&actual, session::encode_baseline(&sequence));
        panic!("{found} — fresh sequence at {}", actual.display());
    }
}

/// What the milestone is about, as one ratio: the mission is three thousand
/// host ticks and half a year of sim time. Warp is an `Epoch` the stream steps,
/// so this number is *in* the recording — a shell that had put it in `TickClock`
/// would replay the same frames over a flat clock and never leave the parking
/// orbit.
#[test]
fn the_sim_clock_is_not_the_host_clock() {
    let flight = session::fly(&entry(), session::FLOWN).expect("the pilot flew");
    let progress = session::progress(&entry(), &flight.frames).expect("driven");
    let spanned = progress.iter().map(|p| p.epoch).max().expect("ticks");
    assert!(
        spanned > progress.len() as u64 * 1_000,
        "{spanned} sim ticks over {} host ticks is not a warped mission",
        progress.len()
    );
    // Monotone, and the closing restart is what breaks it: an epoch that ever
    // went backwards mid-flight would be a stride computed from host state.
    let flight_ticks = &progress[..progress.len() - 1];
    assert!(
        flight_ticks.windows(2).all(|w| w[1].epoch >= w[0].epoch),
        "the epoch went backwards"
    );
    assert_eq!(
        progress.last().expect("ticks").epoch,
        0,
        "the restart takes the epoch with the world"
    );
}

/// The negative control, and the reason the constants are solved rather than
/// picked: a departure a few minutes late does not degrade the mission, it ends
/// it. The window really is one cell of the sweep wide.
#[test]
fn a_late_departure_never_arrives() {
    let late = Plan {
        wait: session::FLOWN.wait + 25,
        ..session::FLOWN
    };
    let flight = session::fly(&entry(), late).expect("the pilot flew");
    assert_eq!(flight.outcome, FLYING, "a miss is a miss, not a crash");
    assert!(
        flight.approach > GRIP * 10.0,
        "25 host ticks of parking orbit is 7 minutes and it should miss by far more than the \
         grip: {:.4e} m",
        flight.approach
    );
    assert!(flight.periapsis.is_none(), "nothing took it");
}

/// The gate's premise, held here rather than in `xtask`: `session::endless` has
/// to coast where it says it coasts and light exactly once, or `reload --burn`
/// is measuring a swap against nothing.
///
/// Guarded in the crate because a stream that stopped lighting the engine would
/// otherwise turn the reload gate green by making both runs a coast.
#[test]
fn the_coasting_stream_lights_once_and_late() {
    let frames = session::endless(8_000);
    let progress = session::progress(&entry(), &frames).expect("driven");
    let lit: Vec<usize> = progress
        .iter()
        .enumerate()
        .filter_map(|(at, p)| p.lit.then_some(at))
        .collect();
    let first = *lit.first().expect("the stream lit the engine");
    assert_eq!(
        first,
        session::LIGHT_AT,
        "the engine lit at {first}, not at the offset the stream declares"
    );
    assert!(
        lit.windows(2).all(|w| w[1] == w[0] + 1),
        "one burn, unbroken — a second light would give the gate two events to part on"
    );
    // The claim `coasting` makes to a caller choosing a swap tick, checked
    // against the world rather than against the function's own arithmetic.
    assert!(
        lit.iter().all(|&at| !session::coasting(at)),
        "`coasting` called a lit tick a coasting one"
    );
    // And the warp taps: a coast at 1x would still pass everything above, and
    // the whole of what this stream adds to its three neighbours is the 100x.
    let warped = progress[session::LIGHT_AT - 1].epoch - progress[session::LIGHT_AT - 2].epoch;
    assert_eq!(warped, 100, "the coast is supposed to run at 100x");
    assert_eq!(
        progress[session::LIGHT_AT].epoch - progress[session::LIGHT_AT - 1].epoch,
        1,
        "lighting the engine is supposed to drop warp to 1x by itself"
    );
}

/// M39's whole claim, swept: a scheduled burn starts on **its own tick**, not
/// on the first tick a stride happens to land past it.
///
/// The sweep starts at 100x because that is where the question exists. Below
/// it, [`NODE_LEAD`] divides the stride and the epoch lands on the node whether
/// anything clamps or not; the mission's own tests fly those rates. From
/// `WARPS[4]` up the lead is not a multiple of the stride, and an unclamped
/// clock steps over the node and never fires it at all — silently, since a row
/// nothing reads raises nothing.
#[test]
fn a_scheduled_burn_starts_on_its_own_tick_at_every_warp() {
    let mut clamped = 0;
    for step in 2..WARPS.len() as u32 {
        let warp = WARPS[step as usize];
        let frames = session::scheduled(step, session::scheduled_ticks(step));
        let run = session::progress(&entry(), &frames).expect("driven");

        // Computed from the stream, not read back from the world: `control`
        // schedules before `advance` steps, so the node is due `NODE_LEAD`
        // past the epoch the *previous* tick ended on. Asking the world
        // instead would grade the queue against its own bookkeeping — and at
        // 1e6 the light has already fired by the end of the plan tick, so what
        // is pending there is the cut.
        // (At 1e6 the light is already gone by the end of the plan tick, which
        // is why nothing here reads the queue for it — a stride that covers the
        // whole lead in one step never leaves the row observable.)
        let due = run[session::PLAN_AT - 1].epoch + NODE_LEAD;

        let edges = |want: bool| -> Vec<usize> {
            (1..run.len())
                .filter(|&at| run[at].lit == want && run[at - 1].lit != want)
                .collect()
        };
        let lights = edges(true);
        assert_eq!(
            lights.len(),
            1,
            "{warp}x: the node fired {} times, not once — 0 is a stride that stepped over it",
            lights.len()
        );

        let at = lights[0];
        assert_eq!(
            run[at].epoch,
            due,
            "{warp}x: the burn started at sim tick {} against a node due at {due} — {} ticks late",
            run[at].epoch,
            run[at].epoch.saturating_sub(due)
        );

        // The clamp, seen doing its work: the stride *into* the fire tick is
        // short. Without it that tick would have been `warp` long and landed
        // past the node.
        let stride = run[at].epoch - run[at - 1].epoch;
        assert!(
            stride <= warp,
            "{warp}x: a stride of {stride} exceeded the warp"
        );
        clamped += usize::from(stride < warp);

        // The burn's *end* is a queue entry too, so it lands the same way.
        let cuts = edges(false);
        assert_eq!(
            cuts.len(),
            1,
            "{warp}x: the burn ended {} times",
            cuts.len()
        );
        assert_eq!(
            run[cuts[0]].epoch,
            due + NODE_HOLD,
            "{warp}x: the cut missed its own tick"
        );
        assert_eq!(
            run[run.len() - 1].due,
            0,
            "{warp}x: a fired node stayed in the queue"
        );
    }
    assert!(
        clamped >= 3,
        "only {clamped} of the swept rates had their stride clamped — a sweep where the \
         lead divides every stride would pass with no clamp in the source at all"
    );
}
