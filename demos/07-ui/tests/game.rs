//! Demo 07, driven the way the host drives it (§4.2.2, §4.9).
//!
//! Nothing here calls a system directly: every tick goes through the systems
//! table against a `World` registered from the *declared* table, and then
//! through the real `gg_ui::Ui` — the same two calls, in the same order,
//! `gg_runtime::App::sim_tick` makes. So what this exercises is the shell's
//! path, minus the shell.
//!
//! The one thing it cannot cover is the shell's own wiring — that the four verb
//! names resolve out of a *loaded dylib* and that the geometry reaches a
//! swapchain. `xtask reload --ui` is that half, over the checked-in replay.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use demo_07_ui::{
    CLOSE, DIFFICULTY, LAYOUT, MENU, SESSION_LOG, Settings, VSYNC, centre_of, session,
};
use gg_ecs::boundary::{
    self, AbiInfo, ComponentsTable, HostApiV1, InputFrame, SystemsTable, TickCtx, VerbsTable,
    Widget, state,
};
use gg_ecs::{Query, World};
use gg_ui::Ui;
use gg_ui::router::{Binding, Tick};

// The five symbols `gg_game!` exported into this crate's rlib, declared the way
// a host reaches a dylib's tables.
unsafe extern "C" {
    fn gg_game_abi() -> AbiInfo;
    fn gg_game_init(api: *const HostApiV1);
    fn gg_game_components() -> ComponentsTable;
    fn gg_game_verbs() -> VerbsTable;
    fn gg_game_systems() -> SystemsTable;
}

/// A loaded game, its world, and the host's UI over it.
struct Game {
    world: World,
    table: SystemsTable,
    ui: Ui,
    binding: Binding,
    target: (u32, u32),
    tick: u64,
    previous: InputFrame,
}

impl Game {
    fn load(target: (u32, u32)) -> Self {
        // SAFETY: the symbols are this binary's own, linked from the rlib.
        let info = unsafe { gg_game_abi() };
        assert_eq!(info, boundary::abi_info(), "one build, one description");
        // SAFETY: `host_api()` returns a `&'static` table (§4.2.2).
        unsafe { gg_game_init(core::ptr::from_ref(boundary::host_api())) };

        let mut world = World::new();
        // The host registers its own protocol types first, so `adopt` has
        // something to disagree with (`gg_runtime::app::adopt`).
        world.register::<Widget>().unwrap();
        // SAFETY: the tables are this binary's own and live for the process.
        let (table, verbs) = unsafe {
            world.adopt(&gg_game_components()).unwrap();
            (gg_game_systems(), boundary::read_verbs(&gg_game_verbs()))
        };
        // By *name*, out of the declared list — the shell's own lookup, so a
        // renamed verb fails here rather than routing clicks nowhere.
        let binding = gg_ui::boundary::binding(&verbs).expect("demo 07 declares the four UI verbs");
        Game {
            world,
            table,
            ui: Ui::new().unwrap(),
            binding,
            target,
            tick: 0,
            previous: InputFrame::default(),
        }
    }

    /// Systems, then the UI — the shell's order, and the reason a click lands
    /// in `Widget::state` inside the tick that took it.
    fn step(&mut self, frame: InputFrame) {
        let ctx = TickCtx::detached(self.tick, 60, frame, self.previous);
        // SAFETY: the table is this binary's own, its entries live for the
        // process, and `ctx` outlives the call.
        unsafe { self.world.run_systems(&self.table, &ctx) }.expect("no system panicked");
        let held =
            |f: &InputFrame, id: gg_ecs::boundary::ActionId| f.buttons & (1 << id.index()) != 0;
        self.ui.frame(
            &mut self.world,
            &Tick {
                motion: (
                    frame.axes[self.binding.x.index()],
                    frame.axes[self.binding.y.index()],
                ),
                primary: held(&frame, self.binding.primary),
                // An edge, not a held state — `Tick::from_input` uses
                // `just_pressed` for the same reason.
                advance_focus: held(&frame, self.binding.advance_focus)
                    && !held(&self.previous, self.binding.advance_focus),
                // Demo 07 declares neither the wheel verbs nor `ui_press`, so
                // its UI never scrolls and its keyboard cannot press — the two
                // cases `gg_ui::boundary::binding` leaves optional, and the
                // reason this harness can stay this small.
                activate: false,
                scroll: 0,
            },
            gg_ui::Fit::new(self.target),
        );
        self.previous = frame;
        self.tick += 1;
    }

    fn play(&mut self, frames: &[InputFrame]) {
        for frame in frames {
            self.step(*frame);
        }
    }

    fn settings(&self) -> Settings {
        let query = Query::<&Settings>::new().unwrap();
        let mut found = None;
        self.world
            .each_ref(&query, |_, s: &Settings| found = Some(*s));
        found.expect("bootstrap spawns exactly one")
    }

    fn widget(&self, id: u64) -> Widget {
        let query = Query::<&Widget>::new().unwrap();
        let mut found = None;
        self.world.each_ref(&query, |_, w: &Widget| {
            if w.id == id {
                found = Some(*w);
            }
        });
        found.expect("every LAYOUT row is spawned")
    }

    fn hash(&self) -> u128 {
        self.world.canonical_hash().get()
    }
}

/// A pointer parked at `id`'s centre, hovering, having pressed nothing.
fn hovering(id: u64, target: (u32, u32)) -> Game {
    let mut game = Game::load(target);
    let (x, y) = centre_of(id);
    let mut frames = vec![InputFrame::default()];
    frames.push(InputFrame {
        buttons: 0,
        axes: {
            let mut axes = [0; gg_ecs::boundary::MAX_AXES];
            axes[0] = (x * gg_ecs::boundary::AXIS_SCALE as f32) as i32;
            axes[1] = (y * gg_ecs::boundary::AXIS_SCALE as f32) as i32;
            axes
        },
    });
    frames.push(InputFrame::default());
    game.play(&frames);
    game
}

#[test]
fn the_first_tick_puts_the_hud_up_with_the_menu_closed() {
    let mut game = Game::load((1280, 720));
    game.step(InputFrame::default());

    let s = game.settings();
    assert_eq!((s.open, s.vsync, s.difficulty), (0, 1, 1));
    // Every declared row exists; the menu's are hidden by a zero rect rather
    // than by being absent, so the world's shape does not change with the UI.
    for item in LAYOUT {
        let w = game.widget(item.id);
        assert_eq!(w.rect, if item.in_menu { [0.0; 4] } else { item.rect });
    }
}

/// The host writes hover into the component the game reads. One tick of lag,
/// and no more.
#[test]
fn a_pointer_over_a_button_lands_hover_in_the_world() {
    let game = hovering(MENU, (1280, 720));
    assert_eq!(game.widget(MENU).state & state::HOVERED, state::HOVERED);
    assert_eq!(game.widget(VSYNC).state, 0, "hidden, and not hoverable");
}

/// A hidden widget cannot be clicked *through* the panel that is not there:
/// the menu's rows sit at a zero rect while it is closed, so a pointer parked
/// on the difficulty button's usual place hits nothing.
#[test]
fn a_hidden_widget_takes_no_hits() {
    let game = hovering(DIFFICULTY, (1280, 720));
    assert_eq!(game.widget(DIFFICULTY).state, 0);
    assert_eq!(game.settings().difficulty, 1, "nothing cycled");
}

/// The whole scripted session, through the real router: the menu opens, one
/// switch flips, the difficulty cycles twice and the menu closes.
#[test]
fn the_scripted_session_opens_the_menu_and_changes_the_settings() {
    let mut game = Game::load((1280, 720));
    game.play(&session());

    let s = game.settings();
    assert_eq!(s.open, 0, "the session closes what it opened");
    assert_eq!(s.vsync, 0, "one click, one switch");
    assert_eq!(s.invert, 0, "the session never touches this one");
    // Two clicks from `normal`: brutal, then wrapping to calm.
    assert_eq!(s.difficulty, 0);
    assert!(s.score > 0, "the sim ran while the menu was up");
    // The log the shell-level gate reads, derived from the same clicks.
    assert_eq!(SESSION_LOG.len(), 5);
    assert_eq!(game.widget(MENU).text(), "menu");
    assert_eq!(game.widget(CLOSE).rect, [0.0; 4]);
}

/// §6 M13's criterion, end to end: the same replayed stream lands on the same
/// widgets whatever the window is — so the canonical hash, which covers
/// `Widget::state`, is identical too.
#[test]
fn a_replayed_session_lands_on_the_same_widgets_at_any_window_size() {
    let frames = session();
    let run = |target| {
        let mut game = Game::load(target);
        game.play(&frames);
        (game.settings(), game.hash())
    };
    let reference = run((1280, 720));
    assert_eq!(run((640, 360)), reference, "at canvas scale");
    assert_eq!(run((3840, 2160)), reference, "at 4K");
    assert_eq!(run((1024, 1024)), reference, "and at a square window");
}
