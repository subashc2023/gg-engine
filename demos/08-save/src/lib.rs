//! Demo 08 — **the session worth keeping** (§6 M14): a world someone would be
//! annoyed to lose.
//!
//! An avatar walks a field of chests. Touching a closed one opens it and adds
//! its value to what the avatar *carries*; the bank verb moves carried into
//! banked. Nothing here is clever — the point is that after a few minutes the
//! world holds numbers no rerun reproduces by accident, which is what makes a
//! save worth writing and losing one worth refusing.
//!
//! # The state is earned, so losing it is visible
//!
//! [`Progress`] and [`Chest`] are ordinary components (§4.5), so a save is the
//! same `Pod` column memcpy every reload already does — `gg_ecs::world::save`
//! adds the container and the policy, not a serializer. That policy is why this
//! demo exists in the shape it does: a build that stopped declaring [`Chest`]
//! would, under reload rules, load this world with every chest silently shut
//! again. A save refuses it by name instead.
//!
//! # The whole session is a replay
//!
//! [`session`] is the script, authored here rather than in `xtask` so the
//! checked-in replay and the in-process tests are the *same* walk (demo 02's
//! rule, §5.6). Every target is [`chest_at`] a grid position, so moving a chest
//! moves the walk — and `xtask reload --save` reads [`SESSION_LOG`] out of a
//! real shell run under three tiers.

use gg_ecs::Component;
use gg_ecs::boundary::{
    AXIS_SCALE, ActionId, AxisId, Eye, GameWorld, InputFrame, MAX_AXES, Renderable, log_level,
};
use gg_math::sim;

/// The verbs, by the index their name sits at in `gg_game!`'s lists below —
/// that order *is* the id space a replay records (§4.7).
pub const BANK: ActionId = ActionId::new(0);
pub const MOVE_X: AxisId = AxisId::new(0);
pub const MOVE_Z: AxisId = AxisId::new(1);

/// Chests across and back. Twelve is enough for a walk to miss some, which is
/// what makes "which ones did I open" state rather than a count.
pub const COLUMNS: usize = 4;
pub const ROWS: usize = 3;
pub const CHESTS: usize = COLUMNS * ROWS;

/// Metres between chests, and metres the avatar covers in one tick.
///
/// `SPACING` is a whole multiple of `MOVE_PER_TICK` and both are exact in
/// binary, so a walk of N ticks lands *on* a grid position rather than near one
/// — a script that arrived 0.03 m short would open chests on some builds and not
/// others, and the divergence would be blamed on the save.
pub const SPACING: f64 = 3.0;
pub const MOVE_PER_TICK: f64 = 0.25;

/// How close counts as touching — two boxes of half-extent 0.4 and 0.35 overlap
/// inside this, so it is "standing on it" rather than "near it".
///
/// Deliberately tight. A generous reach would let a walk that drifted half a
/// metre still open every chest, and `xtask reload --save` would stay green over
/// a session nobody could have had.
pub const REACH: f64 = 0.6;

/// Closed, open, and the avatar. `0x00RRGGBB` (§4.5).
const SHUT: u32 = 0x0075_4a2c;
const OPENED: u32 = 0x0047_b06a;
const AVATAR: u32 = 0x00e8_c86a;

/// What the player has done, on the avatar itself — the entity that earned it.
///
/// Field order is `u64`s first, which is not style: `bytemuck::Pod` refuses
/// padding, and this is the layout that has none.
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "demo08.progress")]
#[repr(C)]
pub struct Progress {
    /// Banked — safe, and what a save is really carrying.
    pub banked: u64,
    /// Carried since the last bank.
    pub carried: u64,
    pub opened: u32,
    _pad: u32,
}

/// One chest, and whether it is still worth walking to.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "demo08.chest")]
#[repr(C)]
pub struct Chest {
    /// Its position in the grid, so a log line names *which* one.
    pub index: u32,
    pub value: u32,
    pub open: u32,
    _pad: u32,
}

/// Where chest `index` stands. The grid is centred on x and walks away in −z.
#[must_use]
pub fn chest_at(index: usize) -> sim::DVec3 {
    let (col, row) = (index % COLUMNS, index / COLUMNS);
    sim::DVec3::new(
        (col as f64 - (COLUMNS as f64 - 1.0) * 0.5) * SPACING,
        0.0,
        -SPACING - row as f64 * SPACING,
    )
}

/// What chest `index` is worth. A function of the index so a test can state the
/// expected total without repeating the table.
#[must_use]
pub fn value_of(index: usize) -> u32 {
    10 * (index as u32 + 1)
}

/// Spawn the field, the avatar and the eye if there are none.
///
/// Idempotent by asking the world rather than by remembering, because
/// remembering would be state outliving a tick (§4.2.2, and a grep gate). It is
/// also what makes a loaded save survive: the first tick after a load finds the
/// avatar already there and adds nothing.
pub fn bootstrap(world: &mut GameWorld) {
    let mut exists = false;
    world.visit::<&Progress>(|_, _| exists = true);
    if exists {
        return;
    }

    let avatar = world.spawn_with(Progress {
        banked: 0,
        carried: 0,
        opened: 0,
        _pad: 0,
    });
    world.put(
        avatar,
        Renderable::boxed(sim::DVec3::ZERO, sim::Vec3::new(0.35, 0.5, 0.35), AVATAR),
    );
    for index in 0..CHESTS {
        let chest = world.spawn_with(Chest {
            index: index as u32,
            value: value_of(index),
            open: 0,
            _pad: 0,
        });
        world.put(
            chest,
            Renderable::boxed(chest_at(index), sim::Vec3::new(0.4, 0.3, 0.4), SHUT),
        );
    }
    world.spawn_with(Eye::at(sim::DVec3::new(0.0, 7.0, 4.0), 0.0, -0.8));
    world.log(log_level::INFO, "demo 08: a field of twelve chests");
}

/// Walk. Axis levels, not deltas: the bindings are key pairs, so a tick's motion
/// is one of three values and a run of N ticks covers exactly `N * MOVE_PER_TICK`.
pub fn walk(world: &mut GameWorld) {
    let step = sim::DVec3::new(
        f64::from(world.axis(MOVE_X)),
        0.0,
        f64::from(world.axis(MOVE_Z)),
    ) * MOVE_PER_TICK;
    let mut carrying = false;
    world.visit::<(&Progress, &mut Renderable)>(|_, (_, shape)| {
        shape.position += step;
        carrying = true;
    });
    debug_assert!(carrying || step == sim::DVec3::ZERO);
}

/// Open every closed chest the avatar is standing on, and bank on the verb.
///
/// One system rather than two because both need the avatar's position and the
/// progress in the same pass, and a second `each` over the same columns would be
/// a second archetype walk for one number.
pub fn collect(world: &mut GameWorld) {
    let mut at = sim::DVec3::ZERO;
    world.visit::<(&Progress, &Renderable)>(|_, (_, shape)| at = shape.position);

    // Collected before the second walk: `each` holds the column borrow for its
    // duration, and the log lines have to come out in chest order either way.
    let mut taken: Vec<(u32, u32)> = Vec::new();
    world.visit::<(&mut Chest, &mut Renderable)>(|_, (chest, shape)| {
        if chest.open != 0 || at.distance_squared(shape.position) > REACH * REACH {
            return;
        }
        chest.open = 1;
        shape.color = OPENED;
        taken.push((chest.index, chest.value));
    });

    let banking = world.just_pressed(BANK);
    let mut banked = 0;
    world.visit::<&mut Progress>(|_, progress| {
        for (_, value) in &taken {
            progress.carried += u64::from(*value);
            progress.opened += 1;
        }
        if banking {
            banked = progress.carried;
            progress.banked += progress.carried;
            progress.carried = 0;
        }
    });

    // One line per settled change, in chest order. `xtask reload --save` reads
    // exactly these: a walk that opened a different chest writes a different
    // line, which is the end-to-end half of the cross-tier criterion.
    for (index, value) in taken {
        world.log(
            log_level::INFO,
            &format!("chest {index} opened for {value}"),
        );
    }
    if banking {
        world.log(log_level::INFO, &format!("banked {banked}"));
    }
}

/// What [`session`] leaves in the log, in order.
///
/// Written out rather than derived, so a change to the walk *or* to the grid has
/// to be looked at rather than absorbed. `tests/session.rs` proves the two
/// agree.
pub const SESSION_LOG: &[&str] = &[
    "chest 1 opened for 20",
    "chest 5 opened for 60",
    "banked 80",
    "chest 6 opened for 70",
    "banked 70",
];

/// The chests [`session`] walks over, in order, and where it banks.
const WALK: &[usize] = &[1, 5, 6];

/// The scripted session: down the second lane opening two chests, bank, across
/// to the third, bank again.
///
/// Positions come from [`chest_at`], so moving a chest moves the walk — the
/// property that makes the replay a gate rather than a recording.
#[must_use]
pub fn session() -> Vec<InputFrame> {
    let mut frames = Vec::new();
    let mut at = sim::DVec3::ZERO;
    for (step, &index) in WALK.iter().enumerate() {
        walk_to(&mut frames, &mut at, chest_at(index));
        // A tick of stillness first: `collect` runs after `walk`, so the chest
        // opens on the tick the avatar arrives, and banking on that same tick
        // would depend on the two systems' order rather than on the script.
        hold(&mut frames, 1, 0);
        // Bank after every chest but the first of a pair, so the file holds both
        // a carried total and a banked one.
        if step != 0 {
            hold(&mut frames, 1, 1 << 0);
            hold(&mut frames, 2, 0);
        }
    }
    // The session must keep running after the last bank, or its tail proves
    // nothing about a world that settled.
    hold(&mut frames, 8, 0);
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

/// Walk `at` to `to`, one axis then the other, at full deflection.
///
/// Panics if the distance is not a whole number of ticks — which is a constant
/// someone changed without changing the other, and a script that stopped short
/// of every chest would otherwise pass every test that only counts ticks.
fn walk_to(frames: &mut Vec<InputFrame>, at: &mut sim::DVec3, to: sim::DVec3) {
    for (axis, from, target) in [(0, at.x, to.x), (1, at.z, to.z)] {
        let ticks = ((target - from) / MOVE_PER_TICK).round();
        assert!(
            (from + ticks * MOVE_PER_TICK - target).abs() < 1e-9,
            "{} m is not a whole number of {MOVE_PER_TICK} m ticks",
            target - from
        );
        let level = if ticks < 0.0 { -AXIS_SCALE } else { AXIS_SCALE };
        for _ in 0..ticks.abs() as u32 {
            let mut axes = [0; MAX_AXES];
            // Axis 0 is `move_x` and axis 1 is `move_z`, which is the order the
            // verb list declares — the id space a replay records (§4.7).
            axes[axis] = level;
            frames.push(InputFrame { buttons: 0, axes });
        }
    }
    *at = to;
}

// Order in both verb lists is the id space (§4.7); order in `systems` is the
// execution order (§4.1). Neither is alphabetical and neither may drift.
#[cfg(feature = "game")]
gg_ecs::gg_game! {
    components: [Progress, Chest, Renderable, Eye],
    actions: ["bank"],
    axes: ["move_x", "move_z"],
    systems: [bootstrap, walk, collect],
}
