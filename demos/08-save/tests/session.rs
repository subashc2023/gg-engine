//! Demo 08, driven the way the host drives it (§4.2.2, §6 M14).
//!
//! Nothing here calls a system directly: every tick goes through the systems
//! table against a `World` registered from the *declared* table, which is the
//! call `gg_runtime::App::sim_tick` makes. So what this exercises is the shell's
//! path, minus the shell — and the save it writes is byte-for-byte the file
//! `--save` produces.
//!
//! What it cannot cover is the tier axis: that the same file loads under dist
//! and dist-verify and reaches the same hash is `xtask reload --save`, over a
//! real shell and a real dylib.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use bytemuck::Zeroable;
use demo_08_save::{CHESTS, Chest, Progress, SESSION_LOG, session, value_of};
use gg_ecs::boundary::{
    self, AbiInfo, ComponentsTable, HostApiV1, InputFrame, Renderable, SystemsTable, TickCtx,
    VerbsTable,
};
use gg_ecs::{Component, Query, Save, SaveError, World};

// The five symbols `gg_game!` exported into this crate's rlib, declared the way
// a host reaches a dylib's tables.
unsafe extern "C" {
    fn gg_game_abi() -> AbiInfo;
    fn gg_game_init(api: *const HostApiV1);
    fn gg_game_components() -> ComponentsTable;
    fn gg_game_verbs() -> VerbsTable;
    fn gg_game_systems() -> SystemsTable;
}

/// The game's own build of `Progress`, one version on: `deaths` appended and
/// nothing removed. Same declared id, so it is the *same component* as far as a
/// save is concerned — the thing an agent's edit produces (§4.2.2).
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "demo08.progress")]
#[repr(C)]
struct ProgressV2 {
    banked: u64,
    carried: u64,
    deaths: u64,
    opened: u32,
    _pad: u32,
}

/// A loaded game and its world.
struct Game {
    world: World,
    table: SystemsTable,
    tick: u64,
    previous: InputFrame,
}

impl Game {
    fn load() -> Self {
        // SAFETY: the symbols are this binary's own, linked from the rlib.
        let info = unsafe { gg_game_abi() };
        assert_eq!(info, boundary::abi_info(), "one build, one description");
        // SAFETY: `host_api()` returns a `&'static` table (§4.2.2).
        unsafe { gg_game_init(core::ptr::from_ref(boundary::host_api())) };

        let mut world = World::new();
        // The host registers its own protocol types first, so `adopt` has
        // something to disagree with (`gg_runtime::app::adopt`).
        world.register::<Renderable>().unwrap();
        world.register::<boundary::Eye>().unwrap();
        // SAFETY: the tables are this binary's own and live for the process.
        let table = unsafe {
            world.adopt(&gg_game_components()).unwrap();
            gg_game_systems()
        };
        // Declared and unread here, but read by the shell and by `xtask` — a
        // verb list that stopped matching `input.toml` fails there, and this
        // keeps the symbol exercised in-process too.
        // SAFETY: as above.
        let verbs = unsafe { boundary::read_verbs(&gg_game_verbs()) };
        assert_eq!(verbs.axes.len(), 2, "move_x, move_z");
        Game {
            world,
            table,
            tick: 0,
            previous: InputFrame::default(),
        }
    }

    fn step(&mut self, frame: InputFrame) {
        let ctx = TickCtx {
            tick: self.tick,
            tick_hz: 60,
            reserved: 0,
            input: frame,
            previous: self.previous,
        };
        // SAFETY: the table is this binary's own, its entries live for the
        // process, and `ctx` outlives the call.
        unsafe { self.world.run_systems(&self.table, &ctx) }.expect("no system panicked");
        self.previous = frame;
        self.tick += 1;
    }

    fn play(&mut self, frames: &[InputFrame]) {
        for frame in frames {
            self.step(*frame);
        }
    }

    fn progress(&self) -> Progress {
        let query = Query::<&Progress>::new().unwrap();
        let mut found = None;
        self.world
            .each_ref(&query, |_, p: &Progress| found = Some(*p));
        found.expect("bootstrap spawns exactly one")
    }

    /// Which chests are open, ascending.
    fn opened(&self) -> Vec<u32> {
        let query = Query::<&Chest>::new().unwrap();
        let mut open = Vec::new();
        self.world.each_ref(&query, |_, c: &Chest| {
            if c.open != 0 {
                open.push(c.index);
            }
        });
        open.sort_unstable();
        open
    }

    fn save(&self) -> Vec<u8> {
        Save::new(self.world.snapshot(), self.tick, 0).encode()
    }

    fn hash(&self) -> u128 {
        self.world.canonical_hash().get()
    }
}

#[test]
fn the_first_tick_lays_out_twelve_shut_chests_and_an_avatar() {
    let mut game = Game::load();
    game.step(InputFrame::default());

    assert_eq!(game.progress(), Progress::zeroed());
    assert!(game.opened().is_empty());
    let query = Query::<&Chest>::new().unwrap();
    let mut count = 0;
    game.world.each_ref(&query, |_, c: &Chest| {
        assert_eq!(c.value, value_of(c.index as usize));
        count += 1;
    });
    assert_eq!(count, CHESTS);
}

#[test]
fn the_scripted_session_opens_three_chests_and_banks_twice() {
    let mut game = Game::load();
    game.play(&session());

    let p = game.progress();
    assert_eq!(game.opened(), vec![1, 5, 6]);
    assert_eq!(p.opened, 3);
    assert_eq!(p.carried, 0, "the session banks what it takes");
    assert_eq!(p.banked, u64::from(value_of(1) + value_of(5) + value_of(6)));
}

/// The constant `xtask reload --save` greps for, checked against the run that
/// produces it. A walk that opened a different chest would leave this table
/// describing a session nobody had.
#[test]
fn the_session_log_names_exactly_what_the_session_did() {
    let mut game = Game::load();
    game.play(&session());

    let mut opened = Vec::new();
    let mut banked = 0u64;
    for line in SESSION_LOG {
        let words: Vec<&str> = line.split_whitespace().collect();
        match words.as_slice() {
            ["chest", index, "opened", "for", value] => {
                opened.push(index.parse::<u32>().unwrap());
                assert_eq!(
                    value.parse::<u32>().unwrap(),
                    value_of(opened[opened.len() - 1] as usize)
                );
            }
            ["banked", amount] => banked += amount.parse::<u64>().unwrap(),
            _ => panic!("SESSION_LOG line `{line}` is not something the game emits"),
        }
    }
    opened.sort_unstable();
    assert_eq!(opened, game.opened());
    assert_eq!(banked, game.progress().banked);
}

/// A save written mid-session and loaded into a fresh world resumes the same
/// session — the property `--save`/`--load` sells, in one process.
#[test]
fn a_save_taken_mid_walk_resumes_into_the_same_world() {
    let frames = session();
    let (first, rest) = frames.split_at(frames.len() / 2);

    let mut straight = Game::load();
    straight.play(&frames);

    let mut interrupted = Game::load();
    interrupted.play(first);
    let file = interrupted.save();

    let mut resumed = Game::load();
    let save = Save::decode(&file).unwrap();
    assert_eq!(save.tick(), first.len() as u64);
    resumed.world.load(&save).unwrap();
    resumed.tick = save.tick();
    resumed.play(rest);

    assert_eq!(resumed.progress(), straight.progress());
    assert_eq!(resumed.opened(), straight.opened());
    assert_eq!(resumed.hash(), straight.hash());
}

/// Play mode's equality (§6 M14), in the world a player actually built: enter,
/// keep playing, stop, and the bytes are the bytes.
#[test]
fn play_mode_gives_back_the_world_that_entered_it() {
    let frames = session();
    let mut game = Game::load();
    game.play(&frames);

    let entered = game.world.snapshot().encode();
    let before = game.hash();
    // Mutate, and visibly: replaying the walk from where it stopped opens the
    // chests it passes a second time and banks again.
    game.play(&frames);
    assert_ne!(
        game.hash(),
        before,
        "play mode with nothing to undo proves nothing"
    );

    game.world
        .restore(&gg_ecs::Snapshot::decode(&entered).unwrap())
        .unwrap();
    assert_eq!(game.world.snapshot().encode(), entered, "bit for bit");
    assert_eq!(game.hash(), before);
}

/// The N → N+1 case on this demo's own component: a field appended, the
/// player's numbers where the new build keeps them.
#[test]
fn a_save_survives_a_schema_change_to_progress() {
    let mut game = Game::load();
    game.play(&session());
    let banked = game.progress().banked;
    let file = game.save();

    let mut next = World::new();
    next.register::<ProgressV2>().unwrap();
    next.register::<Chest>().unwrap();
    next.register::<Renderable>().unwrap();
    next.register::<boundary::Eye>().unwrap();
    let report = next.load(&Save::decode(&file).unwrap()).unwrap();
    assert!(!report.is_clean(), "the schema moved: {report:?}");

    let query = Query::<&ProgressV2>::new().unwrap();
    let mut found = None;
    next.each_ref(&query, |_, p: &ProgressV2| found = Some(*p));
    let p = found.unwrap();
    assert_eq!(p.banked, banked);
    assert_eq!(p.opened, 3);
    assert_eq!(p.deaths, 0, "a field the save never had");
}

/// The refusal that makes a save different from a reload: a build that stopped
/// declaring `Chest` would load this world with every chest shut again, and the
/// player would find out by walking back to one.
#[test]
fn a_build_that_forgot_the_chests_is_refused_by_their_name() {
    let mut game = Game::load();
    game.play(&session());
    let file = game.save();

    let mut forgetful = World::new();
    forgetful.register::<Progress>().unwrap();
    forgetful.register::<Renderable>().unwrap();
    forgetful.register::<boundary::Eye>().unwrap();
    let save = Save::decode(&file).unwrap();

    let refusal = forgetful.load(&save).unwrap_err();
    assert!(
        matches!(&refusal, SaveError::Dropped { declared } if declared == "demo08.chest"),
        "{refusal}"
    );
    // The reload path takes the same image happily, which is what makes the
    // refusal above a policy and not a limitation (§6 M14).
    assert!(forgetful.restore(save.snapshot()).is_ok());
}
