//! The platformer, driven the way the host drives it (§4.2.2): every tick goes
//! through the declared systems table against an adopted `World`, and buttons
//! are set on the [`InputFrame`] the same way a replay sets them — so what is
//! pinned here is what a recorded session will replay (§6 M20).
//!
//! The level is the checked-in `scene.ggsave` (§6 M20 pull 2), loaded here the
//! way the shell's probe and `--load` load it. Geometry claims *derive* from
//! what the scene holds — the pad edge, the wall face, the pit — so an authored
//! level is re-tested by these claims rather than drifting past numbers copied
//! from a table that no longer exists.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use demo_11_platformer::{
    BUFFER_TICKS, COYOTE_TICKS, CUE_DEATH, CUE_GOAL, CUE_JUMP, CUE_LAND, CUES, Cue, Goal, Hud,
    JUMP, JUMP_VELOCITY, MOVE, PLAYER_HALF_H, PLAYER_HALF_W, Player, RESTART, Rig, Run, START,
    Solid, session,
};
use gg_ecs::boundary::{
    self, AbiInfo, ActionId, ComponentsTable, Eye, HostApiV1, InputFrame, Light, Renderable, Sound,
    SystemsTable, TickCtx, Widget,
};
use gg_ecs::{Query, Save, World};

// The symbols `gg_game!` exported into this crate's rlib.
unsafe extern "C" {
    fn gg_game_abi() -> AbiInfo;
    fn gg_game_init(api: *const HostApiV1);
    fn gg_game_components() -> ComponentsTable;
    fn gg_game_systems() -> SystemsTable;
}

/// This crate's own exports as a session [`session::Entry`] — the same four
/// pointers `xtask` resolves out of the built dylib.
fn entry() -> session::Entry {
    session::Entry {
        abi: gg_game_abi,
        init: gg_game_init,
        components: gg_game_components,
        systems: gg_game_systems,
    }
}

/// Full deflection in the fixed-point axis encoding (§4.7's `AXIS_SCALE`).
const STICK: i32 = 1024;

/// `(x, y, half_w, half_h)` of one slab, metres — what the old `LEVEL` table
/// held, now read back out of the scene.
type Slab = (f64, f64, f64, f64);

struct Game {
    world: World,
    table: SystemsTable,
    tick: u64,
    /// This tick's buttons; cleared after every step, so a hold is expressed
    /// by setting it again (demo 10's harness rule).
    held: u64,
    previous: u64,
    /// This tick's `move` axis, raw fixed-point. Cleared like `held`.
    stick: i32,
}

/// Adopt the declared tables into a fresh world — the call `gg_runtime::app`
/// makes (§4.2.2).
fn assemble() -> (World, SystemsTable) {
    // SAFETY: the symbols are this binary's own, linked from the rlib.
    let info = unsafe { gg_game_abi() };
    assert_eq!(info, boundary::abi_info(), "one build, one description");
    // SAFETY: `host_api()` returns a `&'static` table (§4.2.2).
    unsafe { gg_game_init(core::ptr::from_ref(boundary::host_api())) };

    let mut world = World::new();
    // The host's protocol registrations, exactly as `gg_runtime::app::adopt`
    // makes them: the scene's manifest carries `gg.model` though this game
    // never spawns one, and a save may not lose a component by name (§6 M14).
    world.register::<Renderable>().unwrap();
    world.register::<Eye>().unwrap();
    world.register::<boundary::Model>().unwrap();
    world.register::<Light>().unwrap();
    world.register::<Widget>().unwrap();
    // SAFETY: the tables are this binary's own and live for the process.
    let (table, declared) = unsafe {
        let declared = world.adopt(&gg_game_components()).unwrap();
        (gg_game_systems(), declared)
    };
    assert_eq!(declared, 12, "seven of ours and the protocol's five");
    (world, table)
}

impl Game {
    /// The world every real session drives: the checked-in scene, loaded before
    /// any tick the way the shell loads it (§6 M20 pull 2).
    fn load() -> Self {
        let (mut world, table) = assemble();
        let bytes = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/scene.ggsave")).unwrap();
        let save = Save::decode(&bytes).unwrap();
        let report = world.load(&save).unwrap();
        assert!(report.is_clean(), "the scene is this build's own");
        Game {
            world,
            table,
            tick: save.tick(),
            held: 0,
            previous: 0,
            stick: 0,
        }
    }

    /// A world dealt by code alone — what `bootstrap` still owes a session
    /// that was handed no scene.
    fn bare() -> Self {
        let (world, table) = assemble();
        Game {
            world,
            table,
            tick: 0,
            held: 0,
            previous: 0,
            stick: 0,
        }
    }

    fn hold(&mut self, action: ActionId) -> &mut Self {
        self.held |= 1 << action.index();
        self
    }

    fn steer(&mut self, direction: i32) -> &mut Self {
        self.stick = direction * STICK;
        self
    }

    fn step(&mut self) {
        let mut axes = [0; boundary::MAX_AXES];
        axes[MOVE.index()] = self.stick;
        let ctx = TickCtx {
            tick: self.tick,
            tick_hz: 60,
            reserved: 0,
            input: InputFrame {
                buttons: self.held,
                axes,
            },
            previous: InputFrame {
                buttons: self.previous,
                axes: [0; boundary::MAX_AXES],
            },
        };
        // SAFETY: the table is this binary's own, its entries live for the
        // process, and `ctx` outlives the call.
        unsafe { self.world.run_systems(&self.table, &ctx) }.expect("no system panicked");
        self.tick += 1;
        self.previous = self.held;
        self.held = 0;
        self.stick = 0;
    }

    fn steps(&mut self, n: u64) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Hold a direction for `n` ticks.
    fn walk(&mut self, direction: i32, n: u64) {
        for _ in 0..n {
            self.steer(direction);
            self.step();
        }
    }

    fn one<T: gg_ecs::Component + Copy>(&self) -> T {
        let query = Query::<&T>::new().unwrap();
        let mut found = None;
        self.world.each_ref(&query, |_, v: &T| found = Some(*v));
        found.expect("exactly one of this component")
    }

    fn all<T: gg_ecs::Component + Copy>(&self) -> Vec<T> {
        let query = Query::<&T>::new().unwrap();
        let mut out = Vec::new();
        self.world.each_ref(&query, |_, v: &T| out.push(*v));
        out
    }

    fn player(&self) -> Player {
        self.one::<Player>()
    }

    fn run(&self) -> Run {
        self.one::<Run>()
    }

    /// Teleport the player — the only way to ask for a specific spot without
    /// playing there; every behavioural claim still goes through the table.
    fn put_player(&mut self, edit: impl Fn(&mut Player)) {
        let query = Query::<&mut Player>::new().unwrap();
        self.world.each(&query, |_, p: &mut Player| edit(p));
    }

    /// Every cue's sequence, indexed by kind, so a test says what a stretch of
    /// play sounded like by diffing (demo 10's shape).
    fn cues(&self) -> [u32; CUES] {
        let query = Query::<(&Cue, &Sound)>::new().unwrap();
        let mut out = [0; CUES];
        self.world.each_ref(&query, |_, (c, s): (&Cue, &Sound)| {
            out[c.kind as usize] = s.seq;
        });
        out
    }

    /// The HUD line's current text.
    fn hud(&self, line: u32) -> String {
        let query = Query::<(&Hud, &Widget)>::new().unwrap();
        let mut out = None;
        self.world.each_ref(&query, |_, (h, w): (&Hud, &Widget)| {
            if h.line == line {
                out = Some(w.text().to_owned());
            }
        });
        out.expect("every HUD line has a widget")
    }

    /// Step until the player is standing, capped — a test that waits forever
    /// is a hang, not a failure.
    fn settle(&mut self, cap: u64) {
        for _ in 0..cap {
            self.step();
            if self.player().grounded != 0 {
                return;
            }
        }
        panic!("never landed within {cap} ticks");
    }

    /// Every solid's box, read the way the game's own `solids` reads them —
    /// the scene is the single source the old `LEVEL` table was.
    fn slabs(&self) -> Vec<Slab> {
        let query = Query::<(&Solid, &Renderable)>::new().unwrap();
        let mut out = Vec::new();
        self.world
            .each_ref(&query, |_, (_, r): (&Solid, &Renderable)| {
                out.push((
                    r.position.x,
                    r.position.y,
                    f64::from(r.half_extent.x),
                    f64::from(r.half_extent.y),
                ));
            });
        out
    }

    /// The highest slab a fall at `x` from `below` lands on.
    fn pad_under(&self, x: f64, below: f64) -> Slab {
        self.slabs()
            .into_iter()
            .filter(|(sx, sy, shw, shh)| {
                (x - sx).abs() < shw + f64::from(PLAYER_HALF_W) && sy + shh <= below
            })
            .max_by(|a, b| (a.1 + a.3).total_cmp(&(b.1 + b.3)))
            .expect("the scene has no ground there")
    }

    /// A flush step: a slab whose left face rises out of another's top by more
    /// than the player's height — the wall the x resolve is pinned against.
    fn wall(&self) -> (Slab, Slab) {
        let slabs = self.slabs();
        for wall in &slabs {
            for ground in &slabs {
                let flush = ((wall.0 - wall.2) - (ground.0 + ground.2)).abs() < 0.02;
                if flush
                    && (wall.1 + wall.3) - (ground.1 + ground.3) > 2.0 * f64::from(PLAYER_HALF_H)
                {
                    return (*ground, *wall);
                }
            }
        }
        panic!("the scene has no wall to run into");
    }

    /// An x right of the start pad whose whole column is clear air — where a
    /// fall meets nothing before the kill plane.
    fn pit(&self) -> f64 {
        let pad = self.pad_under(START.x, START.y);
        let mut x = pad.0 + pad.2 + 0.05;
        'scan: while x < pad.0 + pad.2 + 40.0 {
            for (sx, _, shw, _) in self.slabs() {
                if (x - sx).abs() < shw + f64::from(PLAYER_HALF_W) + 0.05 {
                    x = sx + shw + f64::from(PLAYER_HALF_W) + 0.1;
                    continue 'scan;
                }
            }
            return x;
        }
        panic!("the scene has no pit to fall into");
    }

    /// The goal box's centre.
    fn goal(&self) -> (f64, f64) {
        let query = Query::<(&Goal, &Renderable)>::new().unwrap();
        let mut out = None;
        self.world
            .each_ref(&query, |_, (_, r): (&Goal, &Renderable)| {
                out = Some((r.position.x, r.position.y));
            });
        out.expect("the scene has a goal")
    }
}

/// The scene supplies the stage before any tick runs (§6 M20 pull 2), and the
/// bootstrap tick doubles none of it.
#[test]
fn the_scene_deals_the_stage() {
    let mut game = Game::load();
    let slabs = game.slabs().len();
    assert!(slabs >= 2, "a platformer level of {slabs} slab(s)");
    assert_eq!(game.all::<Player>().len(), 1);
    assert_eq!(game.all::<Goal>().len(), 1);
    assert_eq!(game.all::<Run>().len(), 1);
    assert_eq!(game.all::<Rig>().len(), 1);
    assert_eq!(game.all::<Cue>().len(), CUES);
    assert_eq!(game.all::<Hud>().len(), 3);
    assert_eq!(game.all::<Light>().len(), 1);
    // Player + slabs + goal, and nothing else draws.
    assert_eq!(game.all::<Renderable>().len(), 2 + slabs);
    game.step();
    assert_eq!(
        game.all::<Renderable>().len(),
        2 + slabs,
        "bootstrap redealt over the scene"
    );
}

/// A world handed no scene gets the player's kit and nothing to stand on: the
/// level is not the code's to deal, so falling is all there is.
#[test]
fn bootstrap_deals_the_kit_but_never_the_level() {
    let mut game = Game::bare();
    game.step();
    assert_eq!(game.all::<Player>().len(), 1);
    assert_eq!(game.all::<Cue>().len(), CUES);
    assert!(game.slabs().is_empty(), "a slab appeared out of code");
    assert!(game.all::<Goal>().is_empty(), "a goal appeared out of code");
    game.steps(240);
    assert!(game.run().deaths >= 1, "no floor, so the fall is a death");
}

/// Re-entrant bootstrap: a reload runs it again against a populated world, and
/// the world it finds must not double (§6 M5).
#[test]
fn bootstrap_run_again_spawns_nothing_new() {
    let mut game = Game::load();
    let drawn = game.all::<Renderable>().len();
    game.steps(8);
    assert_eq!(game.all::<Player>().len(), 1);
    assert_eq!(game.all::<Renderable>().len(), drawn);
    assert_eq!(game.all::<Cue>().len(), CUES);
}

/// The opening fall lands on the start pad and rests there exactly — the
/// resolve leaves contact, not an epsilon float above it — and the landing is
/// the first and only sound the opening makes.
#[test]
fn the_player_comes_to_rest_on_the_start_pad() {
    let mut game = Game::load();
    let before = game.cues();
    game.settle(60);
    game.steps(30);
    let player = game.player();
    assert_eq!(player.grounded, 1);
    let pad = game.pad_under(START.x, START.y);
    let rest = pad.1 + pad.3 + f64::from(PLAYER_HALF_H);
    assert!(
        (player.position.y - rest).abs() < 1e-9,
        "resting at {} not {rest}",
        player.position.y
    );
    let mut expected = before;
    expected[CUE_LAND as usize] += 1;
    assert_eq!(game.cues(), expected, "one landing, nothing else");
}

#[test]
fn holding_right_runs_right() {
    let mut game = Game::load();
    game.settle(60);
    let from = game.player().position.x;
    game.walk(1, 30);
    let to = game.player().position.x;
    assert!(
        to > from + 1.0,
        "thirty held ticks moved only {}",
        to - from
    );
}

/// The step block stops a run at its face, exactly — the x resolve is contact,
/// not a bounce and not a climb.
#[test]
fn the_step_block_is_a_wall() {
    let mut game = Game::load();
    game.step();
    let (ground, wall) = game.wall();
    game.put_player(|p| {
        p.position = gg_math::sim::DVec2::new(
            ground.0,
            ground.1 + ground.3 + f64::from(PLAYER_HALF_H) + 0.5,
        );
        p.velocity = gg_math::sim::DVec2::ZERO;
    });
    game.settle(30);
    // Run at the face until pinned: on flat ground x only stops at the wall.
    let mut last = f64::NAN;
    for _ in 0..600 {
        game.steer(1);
        game.step();
        let now = game.player().position.x;
        if now == last {
            break;
        }
        last = now;
    }
    let stopped = wall.0 - wall.2 - f64::from(PLAYER_HALF_W);
    let at = game.player().position.x;
    assert!(
        (at - stopped).abs() < 1e-9,
        "stopped at {at}, the wall is at {stopped}"
    );
}

/// A jump leaves the ground with exactly [`JUMP_VELOCITY`], sounds its cue,
/// and comes back down to rest.
#[test]
fn a_jump_fires_the_cue_and_returns_to_ground() {
    let mut game = Game::load();
    game.settle(60);
    let before = game.cues();
    game.hold(JUMP);
    game.step();
    let player = game.player();
    assert_eq!(player.velocity.y, JUMP_VELOCITY);
    assert_eq!(player.grounded, 0);
    assert_eq!(
        game.cues()[CUE_JUMP as usize],
        before[CUE_JUMP as usize] + 1
    );
    game.settle(120);
    assert_eq!(
        game.cues()[CUE_LAND as usize],
        before[CUE_LAND as usize] + 1,
        "the jump's own landing"
    );
}

/// Variable height: a press released at once rises well under half of a press
/// held through the rise (§6 M20 pull 4's feel counter three).
#[test]
fn a_held_jump_rises_higher_than_a_tap() {
    let rise = |held_ticks: u64| {
        let mut game = Game::load();
        game.settle(60);
        let rest = game.player().position.y;
        game.hold(JUMP);
        game.step();
        let mut peak = rest;
        for tick in 0..120 {
            if tick < held_ticks {
                game.hold(JUMP);
            }
            game.step();
            peak = peak.max(game.player().position.y);
            if game.player().grounded != 0 {
                break;
            }
        }
        peak - rest
    };
    let held = rise(60);
    let tapped = rise(0);
    assert!((1.9..2.4).contains(&held), "a held jump rose {held}");
    assert!((0.3..1.2).contains(&tapped), "a tapped jump rose {tapped}");
    assert!(held > tapped + 0.8, "held {held} vs tapped {tapped}");
}

/// Off a ledge, a jump still fires inside the coyote window…
#[test]
fn a_ledge_forgives_within_the_coyote_window() {
    let mut game = Game::load();
    game.settle(60);
    let pad = game.pad_under(START.x, START.y);
    game.put_player(|p| p.position.x = pad.0 + pad.2 - 0.5);
    let mut airborne = 0;
    for _ in 0..100 {
        game.steer(1);
        game.step();
        if game.player().grounded == 0 {
            airborne += 1;
            if airborne == 3 {
                break;
            }
        }
    }
    assert_eq!(airborne, 3, "never walked off the pad");
    assert!(game.player().coyote > 0);
    game.hold(JUMP);
    game.step();
    assert_eq!(
        game.player().velocity.y,
        JUMP_VELOCITY,
        "the ledge did not forgive"
    );
}

/// …and not one tick after it closes.
#[test]
fn the_coyote_window_closes() {
    let mut game = Game::load();
    game.settle(60);
    let before = game.cues();
    let pad = game.pad_under(START.x, START.y);
    game.put_player(|p| p.position.x = pad.0 + pad.2 - 0.5);
    for _ in 0..100 {
        game.steer(1);
        game.step();
        if game.player().grounded == 0 {
            break;
        }
    }
    game.steps(u64::from(COYOTE_TICKS) + 2);
    game.hold(JUMP);
    game.step();
    let player = game.player();
    assert!(
        player.velocity.y < 0.0,
        "jumped {} past the window",
        player.velocity.y
    );
    assert_eq!(game.cues()[CUE_JUMP as usize], before[CUE_JUMP as usize]);
}

/// A press a few ticks early is honoured on the landing tick itself.
#[test]
fn a_buffered_press_jumps_on_the_landing_tick() {
    let mut game = Game::load();
    game.settle(60);
    let rest = game.player().position.y;
    game.hold(JUMP);
    game.step();
    // Hold the rise for a full jump — a tapped one falls from its apex slower
    // than the buffer lasts — release across the apex, then press again close
    // to the ground, where the descent is a couple of ticks from landing.
    let mut pressed = false;
    let mut relaunched = false;
    for _ in 0..200 {
        let player = game.player();
        if !pressed && player.velocity.y > 0.0 {
            game.hold(JUMP);
        }
        if !pressed && player.velocity.y < 0.0 && player.position.y - rest < 0.5 {
            game.hold(JUMP);
            pressed = true;
        }
        game.step();
        if pressed && game.player().velocity.y == JUMP_VELOCITY {
            relaunched = true;
            break;
        }
    }
    assert!(pressed, "the descent never came low enough to press in");
    assert!(relaunched, "the buffered press was dropped");
    assert!(
        game.player().buffer < BUFFER_TICKS,
        "the buffer was never consumed"
    );
}

/// Falling out of the world is a death: back to the start, counted, sounded —
/// and the clock keeps running, because the time is the run's.
#[test]
fn falling_out_of_the_world_respawns_and_counts() {
    let mut game = Game::load();
    game.hold(RESTART);
    game.step();
    let started = game.run().started_at;
    let before = game.cues();
    let pit = game.pit();
    game.put_player(|p| {
        p.position = gg_math::sim::DVec2::new(pit, 3.0);
        p.velocity = gg_math::sim::DVec2::ZERO;
    });
    game.steps(120);
    let run = game.run();
    assert_eq!(run.deaths, 1);
    assert_eq!(
        run.started_at, started,
        "a death must not restart the clock"
    );
    assert_eq!(
        game.cues()[CUE_DEATH as usize],
        before[CUE_DEATH as usize] + 1
    );
    let player = game.player();
    assert_eq!(player.position.x, START.x, "respawn is the start, exactly");
}

/// Walking into the goal box stops the clock once; being there again is not a
/// second finish.
#[test]
fn reaching_the_goal_stops_the_clock_once() {
    let mut game = Game::load();
    game.hold(RESTART);
    game.step();
    let (gx, gy) = game.goal();
    let pad = game.pad_under(gx, gy);
    let before = game.cues();
    game.put_player(|p| {
        p.position =
            gg_math::sim::DVec2::new(gx - 3.0, pad.1 + pad.3 + f64::from(PLAYER_HALF_H) + 0.1);
        p.velocity = gg_math::sim::DVec2::ZERO;
    });
    game.settle(30);
    for _ in 0..300 {
        game.steer(1);
        game.step();
        if game.run().finished_at != 0 {
            break;
        }
    }
    let run = game.run();
    assert_ne!(run.finished_at, 0, "never reached the goal");
    assert_eq!(
        game.cues()[CUE_GOAL as usize],
        before[CUE_GOAL as usize] + 1
    );
    game.steps(5);
    assert_eq!(game.run().finished_at, run.finished_at);
    assert_eq!(
        game.cues()[CUE_GOAL as usize],
        before[CUE_GOAL as usize] + 1,
        "the chime repeated"
    );
}

/// `restart` resets the run in place: fresh clock, fresh deaths, the player
/// home — and the level still standing, because the level is an authored scene
/// (§6 M20 pull 2) with no table for a wipe to redeal it from.
#[test]
fn restart_resets_the_run_and_the_level_survives_it() {
    let mut game = Game::load();
    game.step();
    let slabs = game.slabs().len();
    let deaths = game.run().deaths;
    game.put_player(|p| p.position.y = -9.0);
    game.step();
    assert_eq!(game.run().deaths, deaths + 1);
    let at = game.tick;
    game.hold(RESTART);
    game.step();
    let run = game.run();
    assert_eq!(run.deaths, 0);
    assert_eq!(run.started_at, at, "the clock restarts on the restart tick");
    assert_eq!(game.player().position.x, START.x);
    assert_eq!(
        game.slabs().len(),
        slabs,
        "the ground is not the run's to take"
    );
    assert_eq!(
        game.all::<Renderable>().len(),
        2 + slabs,
        "nothing redealt, nothing doubled"
    );
}

/// The stage is seen flat (§6 M20 pull 1): the declared eye is orthographic —
/// zero would be the perspective the migration contract falls back to — and it
/// eases after the player rather than teleporting with them.
#[test]
fn the_camera_is_orthographic_and_eases_after_the_player() {
    let mut game = Game::load();
    game.settle(60);
    let eye: Eye = game.one();
    assert_eq!(eye.ortho, demo_11_platformer::CAMERA_HALF_HEIGHT);
    assert_eq!((eye.yaw, eye.pitch), (0.0, 0.0), "flat means straight-on");
    assert_eq!(eye.position.z, demo_11_platformer::CAMERA_BACK);
    let at = eye.position.x;
    game.walk(1, 30);
    let (eye, player): (Eye, Player) = (game.one(), game.player());
    assert!(eye.position.x > at, "the rig follows the run");
    assert!(
        eye.position.x < player.position.x,
        "and trails it — easing, not attachment"
    );
}

/// A world nobody touches makes no sound — the cue protocol's silence claim,
/// checked on a machine with no speakers (§1.5's audio law).
#[test]
fn an_idle_world_is_silent() {
    let mut game = Game::load();
    game.settle(60);
    game.steps(30);
    let before = game.cues();
    game.steps(90);
    assert_eq!(game.cues(), before, "an idle tick made a noise");
}

/// The recorded session is a whole run: restart, a real death in the first
/// pit, then spawn to goal — every number here re-pinned by `xtask replay
/// --bless` when the level or the feel changes (§6 M20).
#[test]
fn the_recorded_session_runs_the_level_start_to_goal() {
    let frames = session::frames(&entry()).unwrap();
    assert_eq!(frames.len(), 547, "the run changed length — re-bless");
    let progress = session::progress(&entry(), &frames).unwrap();
    let fell = progress.iter().find(|(_, p)| p.deaths == 1).unwrap().0;
    let finished = progress.iter().find(|(_, p)| p.finished).unwrap().0;
    assert_eq!(
        (fell, finished),
        (153, 517),
        "the run changed shape — re-bless"
    );
    let (_, last) = progress.last().unwrap();
    assert!(last.grounded && last.finished && last.deaths == 1);

    // The same frames through the harness, for what Progress cannot see.
    let mut game = Game::load();
    for frame in &frames {
        game.held = frame.buttons;
        game.stick = frame.axes[MOVE.index()];
        game.step();
    }
    let run = game.run();
    assert_eq!(run.started_at, 1, "the opening restart owns the clock");
    assert_eq!(run.finished_at, 517);
    assert_eq!(run.deaths, 1, "one deliberate death, and the clock kept it");
    // Five jumps: the pit, the step, three ledges. Seven landings: those
    // five, the opening fall, and the respawn. One death, one chime.
    assert_eq!(
        game.cues(),
        [5, 7, 1, 1],
        "the run changed sound — re-bless"
    );
    assert_eq!(game.hud(demo_11_platformer::HUD_GOAL), "GOAL 8.6");
}

/// §5.6's material claim for this demo: the session's per-tick canonical
/// hashes match the checked-in baseline, on every architecture the leg runs.
#[test]
fn the_recorded_session_reproduces_its_checked_in_hash_sequence() {
    let sequence = session::hash_sequence(&entry(), &session::frames(&entry()).unwrap()).unwrap();
    let path = session::baseline_path();
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "no baseline at {} ({e}) — run `cargo xtask replay --bless`",
            path.display()
        )
    });
    let baseline = session::parse_baseline(&text).unwrap();
    if let Some(found) = session::divergence(&sequence, &baseline) {
        let actual = std::env::temp_dir().join("demo11-platformer.hashes.actual");
        let _ = std::fs::write(&actual, session::encode_baseline(&sequence));
        panic!("{found} — fresh sequence at {}", actual.display());
    }
}

/// §5.6a on this runner: the same script, driven twice, agrees tick for tick —
/// and the script itself is a pure function of the scene.
#[test]
fn one_recorded_session_run_twice_on_this_runner_agrees() {
    let first = session::frames(&entry()).unwrap();
    let second = session::frames(&entry()).unwrap();
    assert_eq!(first, second, "the bot is not deterministic");
    let a = session::hash_sequence(&entry(), &first).unwrap();
    let b = session::hash_sequence(&entry(), &second).unwrap();
    assert_eq!(a, b, "one machine, two answers");
}

/// The patrol the feel gate swaps under (§6 M20 pull 4): grounded through
/// every walking stretch — where a gravity retune must be latent — airborne
/// somewhere in every cycle, and never dead or done.
#[test]
fn the_endless_patrol_keeps_a_run_in_progress() {
    let frames = session::endless(1_200);
    let progress = session::progress(&entry(), &frames).unwrap();
    let opening = progress[0].0;
    let mut airborne = 0;
    for (tick, at) in &progress {
        let offset = usize::try_from(tick - opening).unwrap();
        assert!(!at.finished, "the patrol reached the goal at tick {tick}");
        assert_eq!(at.deaths, 0, "the patrol died at tick {tick}");
        if session::walking(offset) {
            assert!(at.grounded, "airborne in a walking stretch at tick {tick}");
        }
        airborne += u32::from(!at.grounded);
    }
    assert!(airborne > 60, "the patrol never jumped");
}

/// The HUD clock counts in tenths from the run's start.
#[test]
fn the_hud_tells_the_time() {
    let mut game = Game::load();
    game.hold(RESTART);
    game.step();
    game.steps(60);
    assert_eq!(game.hud(demo_11_platformer::HUD_TIME), "TIME 1.0");
    assert_eq!(game.hud(demo_11_platformer::HUD_DEATHS), "DEATHS 0");
}
