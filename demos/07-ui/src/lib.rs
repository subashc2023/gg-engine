//! Demo 07 — **the HUD and the settings menu** (§6 M13): a game whose UI works
//! without linking a UI framework.
//!
//! Everything on screen is declared as [`Widget`] components (§4.9). This crate
//! links `gg-ecs`, `gg-math` and `bytemuck`, exactly as demos 03 through 06 do;
//! `gg-ui` is on the other side of §3's deny pin and is the host's. The host
//! draws the widgets, routes the pointer over them, and writes what it found
//! back into [`Widget::state`] — so a click arrives here as a bit on a
//! component, one tick after it happened.
//!
//! # The whole session is a replay
//!
//! §6 M13 asks for a HUD and a settings menu driven entirely by a replay file,
//! and `tests/replays/demo07-ui.ggrp` is it: the pointer is an integral of the
//! recorded axis stream (`gg_ui::router`'s docs), so a replayed session opens
//! the menu, toggles the same switches and closes it, on every machine. `xtask
//! reload --ui` is the gate; `tests/clicks.rs` is the same claim in one process.
//!
//! # The pointer is steered onto the OS cursor
//!
//! `input.toml` binds `ui_x`/`ui_y` to `PointerX`/`PointerY` — the shell's
//! steered stream, the OS arrow differenced into canvas units (`Fit::steer`,
//! §6 M15.1) — so the software arrow gg-ui draws sits on the hardware one at
//! any window size and any monitor scale, and the OS arrow is hidden over the
//! window while it does. A human complained (§6 M19): the original binding was
//! raw `MouseX` counts treated as canvas units, which crossed a 1920-wide
//! window in a third of the hand travel and could never agree with the arrow
//! the OS was drawing. The replay property is unchanged either way — the
//! pointer is still an integral of the recorded axis stream, and the recorded
//! session predates the rebind and replays identically over it.

use gg_ecs::Component;
use gg_ecs::boundary::{
    AXIS_SCALE, BoundaryError, Eye, GameWorld, InputFrame, MAX_AXES, Renderable, Widget, log_level,
    state, widget, widget_id,
};
use gg_math::sim;

/// Panel fill, button fill, body text, and the menu's title.
const PANEL: u32 = 0xc00c_1016;
const BUTTON: u32 = 0xff1e_2c3c;
const TEXT: u32 = 0xffd8_e0e8;
const TITLE: u32 = 0xff7f_d0a0;

/// The three difficulties the menu cycles through.
pub const DIFFICULTIES: [&str; 3] = ["calm", "normal", "brutal"];

/// Open and close the settings panel.
pub const MENU: u64 = widget_id("hud.menu");
/// The three switches, and the way out.
pub const VSYNC: u64 = widget_id("settings.vsync");
/// See [`VSYNC`].
pub const INVERT: u64 = widget_id("settings.invert");
/// See [`VSYNC`].
pub const DIFFICULTY: u64 = widget_id("settings.difficulty");
/// See [`VSYNC`].
pub const CLOSE: u64 = widget_id("settings.close");

/// The parts with nothing to click: two backdrops, the two HUD readouts and the
/// menu's heading.
const HUD_PANEL: u64 = widget_id("hud.panel");
const SCORE: u64 = widget_id("hud.score");
const MODE: u64 = widget_id("hud.mode");
const MENU_PANEL: u64 = widget_id("settings.panel");
const TITLE_AT: u64 = widget_id("settings.title");

/// One declared widget, at rest.
///
/// A table rather than five spawn sites, so `bootstrap` and [`present`] cannot
/// disagree about where a widget lives — [`present`] rewrites `rect` every tick
/// to hide the menu, and "hidden" has to mean *back to this row's rectangle*.
pub struct Item {
    /// Identity, and how [`present`] finds it again.
    pub id: u64,
    /// One of [`widget`]'s kinds.
    pub kind: u32,
    /// `[x, y, w, h]` in canvas units (640×360).
    pub rect: [f32; 4],
    /// Draw order; the panels are under their contents.
    pub order: u32,
    /// Hidden — a zero rect — while the settings panel is closed. A zero rect
    /// draws nothing and cannot be hit, which is why hiding needs no second
    /// mechanism.
    pub in_menu: bool,
}

impl Item {
    /// Positional so [`LAYOUT`] reads as a table; the field docs above are the
    /// column headings.
    const fn new(id: u64, kind: u32, rect: [f32; 4], order: u32, in_menu: bool) -> Item {
        Item {
            id,
            kind,
            rect,
            order,
            in_menu,
        }
    }
}

/// The UI, in one place. Order here is spawn order and nothing else; what
/// decides the picture is [`Item::order`].
///
/// Unformatted on purpose: it is a table, and rustfmt breaks the three longest
/// rows across five lines each, which is the one thing a table must not do.
#[rustfmt::skip]
pub const LAYOUT: &[Item] = &[
    Item::new(HUD_PANEL,  widget::PANEL,  [  8.0,   8.0, 168.0,  44.0], 0, false),
    Item::new(SCORE,      widget::LABEL,  [ 16.0,  15.0, 152.0,  10.0], 1, false),
    Item::new(MODE,       widget::LABEL,  [ 16.0,  30.0, 152.0,  10.0], 1, false),
    Item::new(MENU,       widget::BUTTON, [568.0,   8.0,  64.0,  22.0], 2, false),
    Item::new(MENU_PANEL, widget::PANEL,  [200.0,  90.0, 240.0, 176.0], 3, true),
    Item::new(TITLE_AT,   widget::LABEL,  [212.0, 100.0, 216.0,  10.0], 4, true),
    Item::new(VSYNC,      widget::BUTTON, [212.0, 120.0, 216.0,  26.0], 4, true),
    Item::new(INVERT,     widget::BUTTON, [212.0, 152.0, 216.0,  26.0], 4, true),
    Item::new(DIFFICULTY, widget::BUTTON, [212.0, 184.0, 216.0,  26.0], 4, true),
    Item::new(CLOSE,      widget::BUTTON, [212.0, 224.0, 216.0,  26.0], 4, true),
];

/// How many cubes drift behind the UI, and how many ticks they take to circle.
const CUBES: u64 = 3;
const ORBIT: u64 = 900;

/// Everything the menu can change, and the counter that proves the sim is still
/// running while it is open.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "demo07.settings")]
#[repr(C)]
pub struct Settings {
    /// Ticks survived, weighted by difficulty — the HUD's only moving number.
    pub score: u64,
    /// Whether the settings panel is showing.
    pub open: u32,
    /// A switch that does nothing. The renderer's vsync is a CVar and belongs
    /// to the machine, not to the game (§4.8); what this demonstrates is a
    /// *toggle*, and pretending otherwise would put a game in charge of a
    /// setting §4.8 says it does not own.
    pub vsync: u32,
    /// Ditto, for pitch inversion — a real game setting with no camera here to
    /// apply it to.
    pub invert: u32,
    /// Index into [`DIFFICULTIES`].
    pub difficulty: u32,
}

/// Report a refused query and carry on — see demo 03 for why this is not a
/// panic: the host would report the panic and hide the load-sequence bug
/// standing behind it.
fn report(world: &GameWorld, outcome: Result<(), BoundaryError>) {
    if let Err(refused) = outcome {
        world.log(log_level::ERROR, &refused.to_string());
    }
}

/// `"on"` or `"off"`.
#[must_use]
pub fn switch(value: u32) -> &'static str {
    if value == 0 { "off" } else { "on" }
}

/// Spawn the settings, the widgets and the backdrop if there are none.
///
/// Idempotent by asking the world rather than by remembering, because
/// remembering would be state outliving a tick (§4.2.2, and a grep gate).
pub fn bootstrap(world: &mut GameWorld) {
    let mut exists = false;
    let outcome = world.each::<&Settings>(|_, _| exists = true);
    report(world, outcome);
    if exists {
        return;
    }

    world.spawn_with(Settings {
        score: 0,
        open: 0,
        vsync: 1,
        invert: 0,
        difficulty: 1,
    });
    for item in LAYOUT {
        let mut w = match item.kind {
            widget::PANEL => Widget::panel(item.rect, PANEL),
            widget::LABEL => Widget::label(item.rect, TEXT, ""),
            _ => Widget::button(item.id, item.rect, BUTTON, TEXT, ""),
        };
        // Panels and labels are not hit-tested and would not need an id; they
        // get one anyway, because [`present`] finds every widget the same way.
        w.id = item.id;
        w.order = item.order;
        world.spawn_with(w);
    }
    // Something behind the UI, so a panel's alpha has a scene to read over.
    for _ in 0..CUBES {
        world.spawn_with(Renderable::boxed(
            sim::DVec3::ZERO,
            sim::Vec3::new(0.35, 0.35, 0.35),
            0x0058_86b0,
        ));
    }
    world.spawn_with(Eye {
        position: sim::DVec3::new(0.0, 0.6, 3.2),
        yaw: 0.0,
        pitch: -0.1,
    });
    world.log(log_level::INFO, "demo 07: hud up, menu closed");
}

/// Take this tick's clicks, one system before anything reads the result.
///
/// The bit was written by the host *last* tick — that is §4.9's one frame of
/// lag, and it is uniform, which is all a replay needs.
pub fn interact(world: &mut GameWorld) {
    let mut clicked = 0u64;
    let outcome = world.each::<&Widget>(|_, w| {
        if w.state & state::CLICKED != 0 {
            clicked = w.id;
        }
    });
    report(world, outcome);
    if clicked == 0 {
        return;
    }

    let mut said = None;
    let outcome = world.each::<&mut Settings>(|_, s| {
        match clicked {
            MENU => s.open ^= 1,
            CLOSE => s.open = 0,
            VSYNC => s.vsync ^= 1,
            INVERT => s.invert ^= 1,
            DIFFICULTY => s.difficulty = (s.difficulty + 1) % DIFFICULTIES.len() as u32,
            _ => return,
        }
        // One line per settled change. `xtask reload --ui` reads exactly these:
        // a replayed pointer that landed on a different widget writes a
        // different line, which is the end-to-end half of §6 M13's criterion.
        said = Some(match clicked {
            MENU | CLOSE => format!("menu {}", if s.open == 1 { "open" } else { "closed" }),
            VSYNC => format!("vsync {}", switch(s.vsync)),
            INVERT => format!("invert-y {}", switch(s.invert)),
            _ => format!("difficulty {}", DIFFICULTIES[s.difficulty as usize]),
        });
    });
    report(world, outcome);
    if let Some(line) = said {
        world.log(log_level::INFO, &line);
    }
}

/// Score, and the cubes' drift.
pub fn simulate(world: &mut GameWorld) {
    let tick = world.tick();
    let mut step = 0;
    let outcome = world.each::<&mut Settings>(|_, s| {
        step = u64::from(s.difficulty) + 1;
        s.score += step;
    });
    report(world, outcome);

    let mut index = 0u64;
    let outcome = world.each::<&mut Renderable>(|_, r| {
        // A function of the tick alone, through `sim` trig, so the drift is
        // reproducible rather than merely smooth (§4.2.1).
        let phase = (tick + index * ORBIT / CUBES) % ORBIT;
        let turn = phase as f64 / ORBIT as f64 * core::f64::consts::TAU;
        let (sin, cos) = sim::sin_cos(turn);
        r.position = sim::DVec3::new(cos * 1.4, sin * 0.5, -1.0 + sin * 1.4);
        index += 1;
    });
    report(world, outcome);
}

/// Rewrite every widget from the settings. Last in the table, so the frame the
/// host draws is this tick's.
pub fn present(world: &mut GameWorld) {
    let mut settings = None;
    let outcome = world.each::<&Settings>(|_, s| settings = Some(*s));
    report(world, outcome);
    let Some(s) = settings else {
        return;
    };

    // Built before the walk: `each` holds the column borrow for its duration.
    let score = format!("score {:06}", s.score);
    let mode = format!("mode  {}", DIFFICULTIES[s.difficulty as usize]);
    let vsync = format!("vsync       {}", switch(s.vsync));
    let invert = format!("invert y    {}", switch(s.invert));
    let difficulty = format!("difficulty  {}", DIFFICULTIES[s.difficulty as usize]);
    let outcome = world.each::<&mut Widget>(|_, w| {
        let Some(item) = LAYOUT.iter().find(|i| i.id == w.id) else {
            return;
        };
        w.rect = if item.in_menu && s.open == 0 {
            [0.0; 4]
        } else {
            item.rect
        };
        match w.id {
            SCORE => w.set_text(&score),
            MODE => w.set_text(&mode),
            TITLE_AT => {
                w.text_color = TITLE;
                w.set_text("settings");
            }
            MENU => w.set_text(if s.open == 1 { "close" } else { "menu" }),
            VSYNC => w.set_text(&vsync),
            INVERT => w.set_text(&invert),
            DIFFICULTY => w.set_text(&difficulty),
            CLOSE => w.set_text("done"),
            _ => {}
        }
    });
    report(world, outcome);
}

/// Centre of a [`LAYOUT`] row, in canvas units — where [`session`] aims.
///
/// Panics if `id` is not in the table, which is a typo in a `const` and a test
/// that would otherwise click on empty space and pass.
#[must_use]
pub fn centre_of(id: u64) -> (f32, f32) {
    let item = LAYOUT
        .iter()
        .find(|i| i.id == id)
        .unwrap_or_else(|| panic!("no widget {id:#x} in LAYOUT"));
    (
        item.rect[0] + item.rect[2] * 0.5,
        item.rect[1] + item.rect[3] * 0.5,
    )
}

/// What [`session`] leaves in the log, in order.
///
/// The end-to-end half of §6 M13's replayed-click criterion: a pointer that
/// landed on a different widget writes a different line. `xtask reload --ui`
/// reads these out of a shell run; `tests/game.rs` reads them out of the
/// settings the same clicks produced.
pub const SESSION_LOG: &[&str] = &[
    "menu open",
    "vsync off",
    "difficulty brutal",
    "difficulty calm",
    "menu closed",
];

/// The scripted session: find the menu, open it, throw one switch, cycle the
/// difficulty twice, close it.
///
/// Authored here rather than in `xtask` so the checked-in replay and the
/// in-process test are the *same* script (demo 02's rule, §5.6). Every position
/// is [`centre_of`] a row in [`LAYOUT`], so moving a button moves the click.
#[must_use]
pub fn session() -> Vec<InputFrame> {
    let mut frames = Vec::new();
    let mut at = (0i32, 0i32);
    for (target, glide_ticks, clicks) in [
        (MENU, 40, 1),
        (VSYNC, 30, 1),
        (DIFFICULTY, 24, 2),
        (CLOSE, 24, 1),
    ] {
        glide(&mut frames, &mut at, centre_of(target), glide_ticks);
        // Settle: hover resolves against the geometry the *previous* frame
        // declared (§4.9's one frame of lag), and the press must find the
        // widget already hovered or it captures nothing.
        hold(&mut frames, 2, 0);
        for _ in 0..clicks {
            hold(&mut frames, 2, 1 << 0);
            // Released, then long enough for `interact` to see the bit and
            // `present` to redraw what it changed.
            hold(&mut frames, 6, 0);
        }
    }
    // The session must keep running after the last click, or its tail proves
    // nothing about a UI that settled.
    hold(&mut frames, 20, 0);
    frames
}

/// `ticks` frames with nothing moving and `buttons` held.
fn hold(frames: &mut Vec<InputFrame>, ticks: u32, buttons: u64) {
    for _ in 0..ticks {
        frames.push(InputFrame {
            buttons,
            axes: [0; MAX_AXES],
        });
    }
}

/// Move the pointer from `at` to `to` over `ticks`, in fixed point.
///
/// Integer division against the ticks *remaining* rather than a constant step,
/// so the last frame carries the remainder and the pointer lands exactly on the
/// target — a click one unit short of a button is a gate that fails on a
/// rounding change and names nothing.
fn glide(frames: &mut Vec<InputFrame>, at: &mut (i32, i32), to: (f32, f32), ticks: u32) {
    let target = (
        (to.0 * AXIS_SCALE as f32) as i32,
        (to.1 * AXIS_SCALE as f32) as i32,
    );
    for step in 0..ticks {
        let left = (ticks - step) as i32;
        let motion = ((target.0 - at.0) / left, (target.1 - at.1) / left);
        at.0 += motion.0;
        at.1 += motion.1;
        let mut axes = [0; MAX_AXES];
        axes[0] = motion.0;
        axes[1] = motion.1;
        frames.push(InputFrame { buttons: 0, axes });
    }
}

// Order in both verb lists is the id space (§4.7) and is what
// `gg_ui::boundary::binding` resolves against by name; order in `systems` is the
// execution order (§4.1). Neither is alphabetical and neither may drift.
#[cfg(feature = "game")]
gg_ecs::gg_game! {
    components: [Settings, Widget, Renderable, Eye],
    actions: ["ui_click", "ui_focus"],
    axes: ["ui_x", "ui_y"],
    systems: [bootstrap, interact, simulate, present],
}
