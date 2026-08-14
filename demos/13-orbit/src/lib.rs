//! Demo 13 — **the solar system** (the ladder's pinnacle): two regimes, one
//! world, and a transfer between two planets you have to fly.
//!
//! A star, two planets on rails, and a ship parked around the first one. The
//! mission is to leave that parking orbit, cross to the second planet, and be
//! *captured* by it — a closed conic about the target, not a flyby — before the
//! tank runs out. Crashing into anything ends it the other way.
//!
//! # The two regimes, which is what this demo is for (§2)
//!
//! Everything here is in exactly one of two states, and the transition between
//! them is the milestone's subject:
//!
//! - **On rails.** [`Rails`] holds a [`sim::Orbit`] and the entity it is about.
//!   Position is `state_at` of the elapsed sim epoch — a closed form costing the
//!   same 138 ns a century out as next tick (§6 M38 item 7), which is what makes
//!   warp possible at all.
//! - **In the bubble.** [`Bubble`] holds position, velocity and the acceleration
//!   carried across steps, integrated with leapfrog at the tick rate. It exists
//!   only while the engine is lit: §6 M38 item 10 narrowed the bubble from "near
//!   a planet" to "**under non-gravitational acceleration**", because gravity
//!   alone is analytic everywhere about whichever primary dominates.
//!
//! So the engine is the regime switch. Lighting it converts elements to a state
//! vector; cutting it runs [`sim::Orbit::from_state`] on whatever the burn
//! produced. Both directions are computed from sim state, both are in the hash,
//! and neither reads anything the frame rate can move.
//!
//! # Warp is an epoch, not a clock (§6 M38 item 10)
//!
//! [`Epoch::elapsed`] is a count of *sim ticks* that advances by
//! [`Epoch::warp`] each host tick. The host's clock never changes rate:
//! `TickClock` is host state that no save and no hash can see, and
//! `MAX_TICKS_PER_FRAME` is 8, so a warp spent on the accumulator would top out
//! at the player's refresh rate. Warp is refused while a bubble exists, which
//! item 6 made an `O(1)` question — the regime is a component, so the bubble is
//! an archetype and "may I warp" is whether it has rows.
//!
//! The ship is a torch: [`THRUST`] is 250 m/s², deliberately unreasonable,
//! because a burn is the one thing that must run at 1x and nobody should sit
//! through a real one.
//!
//! # The flight log is entities, not a side table (§6 M38 item 12)
//!
//! Every regime change, handover and outcome is an [`Event`] row. §2 asked for a
//! `(tick, entity_id)` total order and that is the canonical walk's own order,
//! so the queue needs no new concept — and unlike a side table it migrates,
//! saves under M14's rules and survives a reload, which is the finding item 12
//! recorded.
//!
//! # What you are looking at
//!
//! A map view, and it is the game's decision rather than the renderer's: a
//! planet at real scale is a sub-pixel dot from anywhere useful. Positions are
//! drawn relative to the ship and multiplied by a scale the zoom keys sweep over
//! six decades, bodies are drawn at their true radius under that scale but never
//! smaller than [`MIN_DOT`], and the conics are sampled into [`Marker`] dots —
//! 64 to an orbit, each one an `Orbit` with its mean anomaly stepped, which is
//! the closed form being asked the question it is good at.
//!
//! Run it: `cargo xtask run 13-orbit`.

use gg_ecs::boundary::{ActionId, Eye, GameWorld, Light, Renderable, Widget, log_level, widget_id};
use gg_ecs::{Component, Entity};
use gg_math::sim;

pub mod session;

// ---- the system ---------------------------------------------------------

/// `GM` of the star, m³/s². The Sun's, because every number item 7 and item 8
/// measured against was at this scale and a demo that quietly used a different
/// one would not be reading the same tables.
pub const MU_STAR: f64 = 1.327_124_4e20;
/// Star radius, metres.
pub const STAR_RADIUS: f64 = 6.957e8;

/// Ticks per second the sim runs at. Read from the host every tick rather than
/// stored — a rate in a component would be float time in sim state (§2).
const fn seconds_per_tick(hz: u32) -> f64 {
    1.0 / hz as f64
}

/// The departure planet's elements, `GM` and radius — Earth's, rounded.
pub const VERGE: Planet = Planet {
    index: 1,
    mu: 3.986_004e14,
    radius: 6.371e6,
    semi_major: 1.496e11,
    eccentricity: 0.0167,
    inclination: 0.0,
    ascending_node: 0.0,
    argument_of_periapsis: 1.796,
    mean_anomaly: 0.0,
    color: 0x0038_6ea8,
};

/// The target's — Mars', rounded, and two of them are not astronomy.
///
/// [`Planet::inclination`] is **zero** where Mars' is 0.032 rad, because the
/// ship's only thrust is along its velocity: a target out of the departure
/// plane is then reachable at its nodes and nowhere else, and a mission whose
/// win condition needs a verb the game does not have is a decoration rather
/// than a mission. The tilt goes back the day there is a normal burn to spend
/// on it.
///
/// [`Planet::mean_anomaly`] is where the target has to *be* for the crossing
/// the departure buys to have something at the end of it — solved by
/// `gg-tools transfer` against the transfer the flown plan produces, and not
/// guessed. Nothing perturbs anything here, so the departure and the phase are
/// independent: the plan chooses the conic, this number chooses the meeting.
pub const OCHRE: Planet = Planet {
    index: 2,
    mu: 4.282_837e13,
    radius: 3.3895e6,
    semi_major: 2.279_39e11,
    eccentricity: 0.0934,
    inclination: 0.0,
    ascending_node: 0.865,
    argument_of_periapsis: 5.000,
    mean_anomaly: 2.982_013,
    color: 0x00b0_5a34,
};

/// A planet as it is authored — flattened into [`Body`] and [`Rails`] at
/// bootstrap. A table rather than a `scene.ggsave`: demo 11 owns the
/// editor-authored level (§6 M20), and elements are not a thing a gizmo drags.
pub struct Planet {
    /// Authored identity, and what a [`Marker`] names its conic by.
    pub index: u32,
    /// `GM`, m3/s2.
    pub mu: f64,
    /// Metres.
    pub radius: f64,
    /// The six elements, about the star.
    pub semi_major: f64,
    /// See [`Planet::semi_major`].
    pub eccentricity: f64,
    /// See [`Planet::semi_major`].
    pub inclination: f64,
    /// See [`Planet::semi_major`].
    pub ascending_node: f64,
    /// See [`Planet::semi_major`].
    pub argument_of_periapsis: f64,
    /// See [`Planet::semi_major`].
    pub mean_anomaly: f64,
    /// `0x00RRGGBB`, as the map draws it.
    pub color: u32,
}

impl Planet {
    /// The elements as the ECS holds them, about the star.
    #[must_use]
    pub const fn orbit(&self) -> sim::Orbit {
        sim::Orbit {
            semi_major: self.semi_major,
            eccentricity: self.eccentricity,
            inclination: self.inclination,
            ascending_node: self.ascending_node,
            argument_of_periapsis: self.argument_of_periapsis,
            mean_anomaly: self.mean_anomaly,
            mu: MU_STAR,
        }
    }
}

/// Radius of the ship's parking orbit about [`VERGE`], metres. ~630 km up.
pub const PARKING_RADIUS: f64 = 7.0e6;
/// Delta-v the tank is worth, m/s. The mission needs about 6 km/s of it; the
/// rest is what a bad transfer costs.
pub const TANK: f64 = 9_000.0;
/// Engine acceleration, m/s². See the module docs — a torch on purpose, because
/// the bubble is the one regime warp may not accelerate.
pub const THRUST: f64 = 250.0;

// ---- outcomes and events ------------------------------------------------

/// Still flying.
pub const FLYING: u32 = 0;
/// Closed conic about the target: the win.
pub const CAPTURED: u32 = 1;
/// Into a body.
pub const CRASHED: u32 = 2;
/// Tank empty and not captured — the trajectory cannot be changed again.
pub const STRANDED: u32 = 3;

/// Engine lit: rails → bubble.
pub const EVENT_LIT: u32 = 0;
/// Engine cut: bubble → rails, elements from the state the burn produced.
pub const EVENT_CUT: u32 = 1;
/// The conic changed primary.
pub const EVENT_HANDOVER: u32 = 2;
/// The mission ended, one way or the other. [`Event::value`] carries which.
pub const EVENT_OUTCOME: u32 = 3;

// ---- the map view -------------------------------------------------------

/// Smallest a body is ever drawn, in map metres — below this a planet at true
/// scale is invisible and the map stops being one.
pub const MIN_DOT: f32 = 0.045;
/// The ship's dot. Never scaled: it is a symbol, not a hull.
pub const SHIP_DOT: f32 = 0.055;
/// Half-width of a trajectory ribbon, in map metres.
///
/// A ribbon and not a dot since §6 M38 item 16, because 64 dots do not read as
/// a line and no count of them would: a dot's size is a constant in map metres
/// while its *spacing* is the conic's circumference over [`TRACE`], which the
/// zoom moves by five decades. Parked, the ring's samples sit 39 dot-widths
/// apart; at the far zoom, 7. What the eye was being shown was the sampling.
pub const TRACE_WIDTH: f32 = 0.022;
/// Samples per conic — 64 segments, which is a polygon no one can pick out of
/// an ellipse at any zoom this map reaches.
pub const TRACE: u32 = 64;
/// How far along a hyperbola the trace runs, in mean anomaly either side of the
/// body. An escape is a shape, and a shape needs both arms.
const HYPERBOLIC_SPAN: f64 = 2.5;

/// Zoom, as the negative base-10 log of map metres per real metre. 5.5 frames a
/// parking orbit.
pub const ZOOM_NEAR: f64 = 5.5;
/// The other end: the whole system from wherever the ship is, and the number is
/// arithmetic rather than a preference. The eye sits [`EYE_RANGE`] back at
/// `r.fov` 1.0 rad, so the map is `2 * 17 * tan(0.5)` = 18.6 map metres tall —
/// and the map is centred on the **ship**, so what has to fit is not the system
/// but a circle of twice the target's aphelion, 1e12 m across. That is
/// `18.6 / 1e12` = 1.9e-11 metres to the metre.
///
/// It was 11.5 until the `orbit` golden was framed (§6 M38 item 15), which is
/// what a picture is for: 11.5 draws the whole system inside a twelfth of the
/// height with every body one pixel, and the first correction — 10.4 — forgot
/// the centring and cropped the far ring off both sides of the frame.
pub const ZOOM_FAR: f64 = 10.75;
/// Decades per tick while a zoom key is held — a decade and a half a second.
const ZOOM_RATE: f64 = 0.025;

/// Where the eye sits in map space. The map is drawn *around the ship*, so the
/// camera never moves: zoom is a change of scale, not of distance, which is the
/// one framing that cannot lose the thing it is following.
pub const EYE_PITCH: f32 = -0.95;
/// How far back along that pitch.
pub const EYE_RANGE: f64 = 17.0;

/// The star's own dot, and the colour of the light that follows it.
pub const STAR_GLOW: u32 = 0x00ff_e9b0;
/// The map's one light, sitting where the star is drawn.
pub const STAR_INK: u32 = 0x00ff_f0d0;
/// What the map's lamp delivers at the eye, in the renderer's own units.
///
/// Read off `gg-tools map` (§6 M38 item 16), and it is a plateau: below it
/// symbols fall under the background, above it their channels clip and they
/// return the *lamp's* colour instead of their own. 8 is where both counts are
/// at their lowest together.
pub const MAP_LUX: f64 = 8.0;
/// The ship's dot.
pub const SHIP_INK: u32 = 0x00ff_ffff;
/// Its trace's.
pub const SHIP_TRACE_INK: u32 = 0x0060_ffc0;

// ---- verbs --------------------------------------------------------------

/// Burn along the velocity.
pub const BURN: ActionId = ActionId::new(0);
/// Burn against it.
pub const RETRO: ActionId = ActionId::new(1);
/// Warp up one step.
pub const WARP_UP: ActionId = ActionId::new(2);
/// Warp down one step.
pub const WARP_DOWN: ActionId = ActionId::new(3);
/// Zoom in.
pub const ZOOM_IN: ActionId = ActionId::new(4);
/// Zoom out.
pub const ZOOM_OUT: ActionId = ActionId::new(5);
/// Start the mission over.
pub const RESTART: ActionId = ActionId::new(6);

/// The warp ladder. Powers of ten, and the top of it is what makes a
/// nine-month transfer a thirty-second one — §6 M38 item 10 measured the
/// sampling hazard a stride of this size creates for the regime condition and
/// found it never binds, by an order of magnitude at its worst row.
pub const WARPS: [u64; 7] = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000];

// ---- components ---------------------------------------------------------

/// A gravitating body. `index` is authored identity — 0 is the star — and is
/// what a [`Marker`] names its subject by, since archetype order is not a name.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "orbit.body")]
#[repr(C)]
pub struct Body {
    /// `GM`, m³/s².
    pub mu: f64,
    /// Metres. Impact happens here, and so does the drawn size.
    pub radius: f64,
    /// 0 star, 1 departure, 2 target.
    pub index: u32,
    /// Padding, and it is declared rather than left: a component with a hole in
    /// it is not `Pod` and would not hash the same bytes twice (§4.2.1 hazard 4).
    pub reserved: u32,
}

/// The analytic regime: a conic about `primary`, stated at `since`.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "orbit.rails")]
#[repr(C)]
pub struct Rails {
    /// Elements, and the `mu` they are about.
    pub orbit: sim::Orbit,
    /// Whose gravity. The star's own row has no [`Rails`] at all — a root is
    /// spelled by absence, `Node`'s rule (§4.6).
    pub primary: Entity,
    /// The epoch tick [`Rails::orbit`]'s mean anomaly is stated at. Integer
    /// time, converted at the use site and never stored as seconds (§2).
    pub since: u64,
}

/// The integrated regime — only ever non-empty while the engine is lit.
///
/// Position and velocity are **relative to the primary**, which is item 8's
/// finding 6: a bubble's roundoff scales with the radius of the coordinate it
/// integrates in, and about the star that is 1.5e11 m of leading digits spent
/// on nothing.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "orbit.bubble")]
#[repr(C)]
pub struct Bubble {
    /// Metres from the primary.
    pub position: sim::DVec3,
    /// Metres per second, in the primary's frame.
    pub velocity: sim::DVec3,
    /// Last step's total acceleration, carried so kick-drift-kick costs the one
    /// force evaluation semi-implicit Euler costs (§6 M38 item 7).
    pub accel: sim::DVec3,
    /// Whose gravity is being integrated.
    pub primary: Entity,
    /// `+1` prograde, `-1` retrograde, `0` coasting inside the bubble's last
    /// tick. Kept here rather than read from input at the integration site, so
    /// what the step did is in the hash.
    pub throttle: i64,
}

/// The ship, and the mission's whole score.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "orbit.ship")]
#[repr(C)]
pub struct Ship {
    /// Delta-v left, m/s.
    pub fuel: f64,
    /// Delta-v spent, m/s — the readout that says what the flight cost.
    pub spent: f64,
    /// [`FLYING`], [`CAPTURED`], [`CRASHED`] or [`STRANDED`].
    pub outcome: u32,
    /// Ticks since the outcome landed, so the banner can say how long ago.
    pub settled: u32,
}

/// The sim epoch and the rate it advances at (§6 M38 item 10).
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "orbit.epoch")]
#[repr(C)]
pub struct Epoch {
    /// Sim ticks since the mission opened. Advances by [`Epoch::warp`] per host
    /// tick, which is the whole of what warp is.
    pub elapsed: u64,
    /// One of [`WARPS`].
    pub warp: u64,
    /// Index into [`WARPS`], so stepping needs no search.
    pub step: u32,
    /// Zoom as the map's negative log scale, in [`ZOOM_NEAR`]`..=`[`ZOOM_FAR`].
    /// On the epoch row because the view is one thing and this is the world's
    /// one singleton, not because a camera is a clock.
    pub zoom: f32,
}

/// One row of the flight log — the event queue §2 asked for, as entities.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "orbit.event")]
#[repr(C)]
pub struct Event {
    /// Sim epoch it happened at.
    pub epoch: u64,
    /// What happened: `EVENT_*`.
    pub kind: u32,
    /// Body index it concerns, or `u32::MAX`.
    pub about: u32,
    /// Kind-specific: the outcome for [`EVENT_OUTCOME`], the speed change for a
    /// cut, the new primary's index for a handover.
    pub value: f64,
}

/// A sampled point of somebody's conic. `of` is a [`Body::index`], or
/// [`Marker::SHIP`] for the ship's own trajectory.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "orbit.marker")]
#[repr(C)]
pub struct Marker {
    /// Whose conic.
    pub of: u32,
    /// Which sample, `0..`[`TRACE`].
    pub slot: u32,
}

impl Marker {
    /// [`Marker::of`] for the ship's trajectory.
    pub const SHIP: u32 = u32::MAX;
}

/// A HUD line. `slot` is the row from the top.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "orbit.readout")]
#[repr(C)]
pub struct Readout {
    /// Row index.
    pub slot: u32,
    /// Padding, declared (see [`Body::reserved`]).
    pub reserved: u32,
}

/// Rows in the HUD.
const READOUTS: u32 = 5;

// ---- bootstrap ----------------------------------------------------------

/// Star, planets, ship, traces, HUD — once. Re-entrant, because a reload runs
/// it again against a world that already holds them (§6 M5).
pub fn bootstrap(world: &mut GameWorld) {
    let mut exists = false;
    world.visit::<&Epoch>(|_, _| exists = true);
    if exists {
        return;
    }

    let star = world.spawn_with(Body {
        mu: MU_STAR,
        radius: STAR_RADIUS,
        index: 0,
        reserved: 0,
    });
    world.put(
        star,
        Renderable::ball(sim::DVec3::ZERO, MIN_DOT, STAR_GLOW).surfaced(0.0, 0.0),
    );

    for planet in [&VERGE, &OCHRE] {
        let body = world.spawn_with(Body {
            mu: planet.mu,
            radius: planet.radius,
            index: planet.index,
            reserved: 0,
        });
        world.put(
            body,
            Rails {
                orbit: planet.orbit(),
                primary: star,
                since: 0,
            },
        );
        world.put(
            body,
            Renderable::ball(sim::DVec3::ZERO, MIN_DOT, planet.color).surfaced(0.0, 0.0),
        );
        for slot in 0..TRACE {
            let dot = world.spawn_with(Marker {
                of: planet.index,
                slot,
            });
            world.put(
                dot,
                trace_segment(sim::DVec3::ZERO, sim::DVec3::ZERO, dim(planet.color)),
            );
        }
    }

    // The ship's parking orbit is authored as elements, not as a state vector:
    // it starts on rails, which is the regime a coasting body is always in.
    let verge = find_body(world, VERGE.index).unwrap_or(Entity::NONE);
    let ship = world.spawn_with(Ship {
        fuel: TANK,
        spent: 0.0,
        outcome: FLYING,
        settled: 0,
    });
    world.put(
        ship,
        Rails {
            orbit: sim::Orbit {
                semi_major: PARKING_RADIUS,
                eccentricity: 0.0,
                inclination: 0.0,
                ascending_node: 0.0,
                argument_of_periapsis: 0.0,
                mean_anomaly: 0.0,
                mu: VERGE.mu,
            },
            primary: verge,
            since: 0,
        },
    );
    world.put(
        ship,
        Renderable::ball(sim::DVec3::ZERO, SHIP_DOT, SHIP_INK).surfaced(0.0, 0.0),
    );
    for slot in 0..TRACE {
        let dot = world.spawn_with(Marker {
            of: Marker::SHIP,
            slot,
        });
        world.put(
            dot,
            trace_segment(sim::DVec3::ZERO, sim::DVec3::ZERO, SHIP_TRACE_INK),
        );
    }

    for slot in 0..READOUTS {
        let row = world.spawn_with(Readout { slot, reserved: 0 });
        world.put(row, Widget::label([0.0; 4], 0x00ff_ffff, ""));
    }

    world.spawn_with(Epoch {
        elapsed: 0,
        warp: WARPS[0],
        step: 0,
        zoom: ZOOM_NEAR as f32,
    });
    // A point light where the star is, moved with it every tick: the map is
    // recentred on the ship, so nothing in render space stays put — and its
    world.spawn_with(map_light());
    world.spawn_with(Eye::at(eye_position(), 0.0, EYE_PITCH));
    world.log(log_level::INFO, "orbit: ready");
}

/// Where the eye sits, once, for the reason [`EYE_PITCH`] gives.
#[must_use]
pub fn eye_position() -> sim::DVec3 {
    let (forward, _) = sim::fly_basis(0.0, EYE_PITCH);
    forward * -EYE_RANGE
}

/// The map's one light, at the **eye**. Public because the golden scene lights
/// its reference with this and not with a second copy of it (§4.10).
///
/// Not at the star, which is where it was until §6 M38 item 16. A map is a
/// schematic and its symbols carry colour, not shading — and a lamp anywhere
/// else in the scene leaves half of every ring unlit and prices the rest by an
/// inverse square the zoom sweeps over five decades, so the same symbol was
/// white at one end of the key and gone at the other. At the eye, every symbol
/// is [`EYE_RANGE`] away whatever the zoom and faces the light it is lit by.
#[must_use]
pub fn map_light() -> Light {
    let far = EYE_RANGE;
    Light::point(
        eye_position(),
        STAR_INK,
        (MAP_LUX * far * far) as f32,
        (far * 4.0) as f32,
    )
}

/// One segment of a trajectory, `from` to `to`: the "long thin box" the render
/// boundary names in as many words, which is the whole of what a line is here.
///
/// Matte and dielectric on purpose. Every symbol on this map is a *symbol* — it
/// carries a colour and nothing else — and a surface with any smoothness at all
/// answers a light with a white specular lobe that Fresnel drives to 1 at
/// grazing angles. On a two-pixel ribbon almost every pixel is grazing, so the
/// authored colour arrives at the screen mixed with the lamp's (§6 M38 item 16,
/// `gg-tools map`).
#[must_use]
pub fn trace_segment(from: sim::DVec3, to: sim::DVec3, color: u32) -> Renderable {
    let delta = to - from;
    let length = delta.length();
    let mut out = Renderable::boxed(
        (from + to) * 0.5,
        sim::Vec3::new(TRACE_WIDTH, TRACE_WIDTH, (length * 0.5) as f32),
        color,
    )
    .surfaced(0.0, 0.0);
    if length > 1e-12 {
        out.rotation = aim(delta / length);
    }
    out
}

/// The shortest rotation taking `+Z` onto `direction`, which must be unit.
/// Built here because `sim` carries axis-angle only, and the two degenerate
/// cases are the ones a cross product cannot normalise.
fn aim(direction: sim::DVec3) -> sim::DQuat {
    let along = sim::DVec3::Z.dot(direction).clamp(-1.0, 1.0);
    if along > 1.0 - 1e-12 {
        return sim::DQuat::IDENTITY;
    }
    if along < -1.0 + 1e-12 {
        return sim::DQuat::from_axis_angle(sim::DVec3::X, core::f64::consts::PI);
    }
    match sim::DVec3::Z.cross(direction).try_normalize() {
        Some(axis) => sim::DQuat::from_axis_angle(axis, sim::acos(along)),
        None => sim::DQuat::IDENTITY,
    }
}

/// A dimmer shade of a trace's body colour, so a ring reads as a ring and not
/// as a queue of planets.
#[must_use]
pub const fn dim(color: u32) -> u32 {
    (color >> 1) & 0x007f_7f7f
}

// ---- reading the world --------------------------------------------------

/// The entity carrying [`Body::index`] `index`, if it is spawned.
fn find_body(world: &mut GameWorld, index: u32) -> Option<Entity> {
    let mut found = None;
    world.visit::<&Body>(|entity, body| {
        if body.index == index {
            found = Some(entity);
        }
    });
    found
}

/// A body's identity and its **absolute** state at an epoch.
///
/// Velocity is carried beside position because a handover is a change of frame,
/// and a frame change that moves only the origin drops the old primary's own
/// motion — 29.8 km/s at the departure planet against a 3 km/s escape, so a
/// conic built without it is not slightly wrong but a different orbit.
#[derive(Clone, Copy)]
struct Station {
    entity: Entity,
    body: Body,
    position: sim::DVec3,
    velocity: sim::DVec3,
}

/// Every body's [`Station`] at `elapsed`.
///
/// Absolute rather than relative because the handover test compares
/// accelerations from *every* body at once, and a chain of parent lookups
/// inside that loop would read the same conic five times.
fn bodies(world: &mut GameWorld, elapsed: u64, hz: u32) -> Vec<Station> {
    let mut rows: Vec<(Entity, Body, Option<Rails>)> = Vec::new();
    world.visit::<&Body>(|entity, body| rows.push((entity, *body, None)));
    for row in &mut rows {
        row.2 = world.get::<Rails>(row.0).copied();
    }
    rows.iter()
        .map(|(entity, body, rails)| {
            let (position, velocity) = match rails {
                // One level deep by construction: a planet's primary is the
                // star, and the star has no `Rails`. A moon would make this a
                // walk, and would need the cycle refusal `Node` already has.
                Some(rails) => rails.orbit.state_at(seconds(elapsed, rails.since, hz)),
                None => (sim::DVec3::ZERO, sim::DVec3::ZERO),
            };
            Station {
                entity: *entity,
                body: *body,
                position,
                velocity,
            }
        })
        .collect()
}

/// The station of `entity`, if it is one of the bodies.
fn station(bodies: &[Station], entity: Entity) -> Option<Station> {
    bodies.iter().copied().find(|held| held.entity == entity)
}

/// Seconds between two epochs. The one place ticks become a float, and it is a
/// division at a use site rather than a field (§2).
fn seconds(elapsed: u64, since: u64, hz: u32) -> f64 {
    (elapsed.saturating_sub(since) as f64) * seconds_per_tick(hz)
}

/// The ship's absolute position, its velocity **relative to its primary**, and
/// which body that is. The two are in different frames on purpose: every reader
/// but the handover wants the position against the whole system and the
/// velocity against the conic it belongs to, and the one that wants both in one
/// frame converts explicitly.
fn ship_state(
    world: &mut GameWorld,
    ship: Entity,
    elapsed: u64,
    hz: u32,
    bodies: &[Station],
) -> Option<(sim::DVec3, sim::DVec3, Entity)> {
    if let Some(bubble) = world.get::<Bubble>(ship).copied() {
        let origin = station(bodies, bubble.primary)?;
        return Some((
            origin.position + bubble.position,
            bubble.velocity,
            bubble.primary,
        ));
    }
    let rails = world.get::<Rails>(ship).copied()?;
    let origin = station(bodies, rails.primary)?;
    let (position, velocity) = rails.orbit.state_at(seconds(elapsed, rails.since, hz));
    Some((origin.position + position, velocity, rails.primary))
}

// ---- control ------------------------------------------------------------

/// Input: throttle, warp, zoom, restart. Every one of them is a verb in the
/// action map, so a session that flies this mission records and replays like any
/// other (§4.7).
pub fn control(world: &mut GameWorld) {
    let hz = world.tick_hz();
    let Some((epoch_entity, mut epoch)) = singleton::<Epoch>(world) else {
        return;
    };
    let Some((ship_entity, ship)) = singleton::<Ship>(world) else {
        return;
    };

    if world.just_pressed(RESTART) {
        restart(world);
        return;
    }

    let zoom = f64::from(epoch.zoom)
        + f64::from(i32::from(world.pressed(ZOOM_OUT)) - i32::from(world.pressed(ZOOM_IN)))
            * ZOOM_RATE;
    epoch.zoom = zoom.clamp(ZOOM_NEAR, ZOOM_FAR) as f32;

    // Throttle first: it decides whether warp is even offered this tick.
    let flying = ship.outcome == FLYING;
    let throttle = if !flying || ship.fuel <= 0.0 {
        0
    } else {
        i64::from(world.pressed(BURN)) - i64::from(world.pressed(RETRO))
    };

    let lit = world.get::<Bubble>(ship_entity).is_some();
    match (lit, throttle) {
        (false, -1 | 1) => {
            light(world, ship_entity, epoch.elapsed, hz, throttle);
            epoch.step = 0;
            epoch.warp = WARPS[0];
        }
        (true, 0) => cut(world, ship_entity, epoch.elapsed, hz),
        (true, _) => {
            if let Some(bubble) = world.get_mut::<Bubble>(ship_entity) {
                bubble.throttle = throttle;
            }
        }
        (false, _) => {}
    }

    // Warp is refused while the bubble has rows, which is an archetype question
    // and not a search (§6 M38 item 6). The refusal is silent on purpose: the
    // HUD already says `1x`, and a log line per held key is not a report.
    if world.get::<Bubble>(ship_entity).is_none() && flying {
        let up = u32::from(world.just_pressed(WARP_UP));
        let down = u32::from(world.just_pressed(WARP_DOWN));
        epoch.step = (epoch.step + up).min(WARPS.len() as u32 - 1);
        epoch.step = epoch.step.saturating_sub(down);
        epoch.warp = WARPS[epoch.step as usize];
    } else {
        epoch.step = 0;
        epoch.warp = WARPS[0];
    }
    world.put(epoch_entity, epoch);
}

/// The one entity carrying `T`, if there is exactly one worth having.
fn singleton<T: Component + Copy>(world: &mut GameWorld) -> Option<(Entity, T)> {
    let mut found = None;
    world.visit::<&T>(|entity, value| found = Some((entity, *value)));
    found
}

/// Rails → bubble. The elements are evaluated once, at this epoch, and the
/// state vector they produce is what the integrator carries from here.
fn light(world: &mut GameWorld, ship: Entity, elapsed: u64, hz: u32, throttle: i64) {
    let Some(rails) = world.get::<Rails>(ship).copied() else {
        return;
    };
    let (position, velocity) = rails.orbit.state_at(seconds(elapsed, rails.since, hz));
    world.remove::<Rails>(ship);
    world.put(
        ship,
        Bubble {
            position,
            velocity,
            // Seeded, not assumed zero: the first step's kick reads it, and a
            // zero here would drop half a tick of gravity into the burn.
            accel: gravity(position, rails.orbit.mu) + thrust(velocity, throttle),
            primary: rails.primary,
            throttle,
        },
    );
    record(world, EVENT_LIT, elapsed, u32::MAX, throttle as f64);
    world.log(log_level::INFO, "orbit: lit");
}

/// Bubble → rails. `from_state` is the conversion §2's regime boundary is made
/// of, and it is the direction that has the degenerate cases (§6 M38 item 8).
fn cut(world: &mut GameWorld, ship: Entity, elapsed: u64, hz: u32) {
    let Some(bubble) = world.get::<Bubble>(ship).copied() else {
        return;
    };
    let Some(primary) = world.get::<Body>(bubble.primary).copied() else {
        return;
    };
    let Some(orbit) = sim::Orbit::from_state(bubble.position, bubble.velocity, primary.mu) else {
        // No conic: a radial fall or a parabola. The bubble stays lit rather
        // than inventing elements — the regime a body has no conic in is the
        // integrated one, which is what the bubble is for.
        return;
    };
    world.remove::<Bubble>(ship);
    world.put(
        ship,
        Rails {
            orbit,
            primary: bubble.primary,
            since: elapsed,
        },
    );
    let _ = hz;
    record(
        world,
        EVENT_CUT,
        elapsed,
        primary.index,
        bubble.velocity.length(),
    );
    world.log(log_level::INFO, "orbit: cut");
}

/// Despawn everything and let [`bootstrap`] run again next tick.
fn restart(world: &mut GameWorld) {
    let mut doomed: Vec<Entity> = Vec::new();
    world.visit::<&Renderable>(|entity, _| doomed.push(entity));
    world.visit::<&Widget>(|entity, _| doomed.push(entity));
    world.visit::<&Epoch>(|entity, _| doomed.push(entity));
    world.visit::<&Light>(|entity, _| doomed.push(entity));
    world.visit::<&Eye>(|entity, _| doomed.push(entity));
    world.visit::<&Event>(|entity, _| doomed.push(entity));
    doomed.sort_unstable();
    doomed.dedup();
    for entity in doomed {
        world.despawn(entity);
    }
    world.log(log_level::INFO, "orbit: restarted");
}

/// Append a row to the flight log (§6 M38 item 12).
fn record(world: &mut GameWorld, kind: u32, epoch: u64, about: u32, value: f64) {
    world.spawn_with(Event {
        epoch,
        kind,
        about,
        value,
    });
}

// ---- the tick -----------------------------------------------------------

/// Advance the epoch by the warp factor. Its own system, and it runs *after*
/// control, so the tick a burn starts on is already back at 1x.
pub fn advance(world: &mut GameWorld) {
    let Some((entity, mut epoch)) = singleton::<Epoch>(world) else {
        return;
    };
    epoch.elapsed = epoch.elapsed.saturating_add(epoch.warp);
    world.put(entity, epoch);
}

/// Gravity of one primary at a point in its frame.
fn gravity(position: sim::DVec3, mu: f64) -> sim::DVec3 {
    let radius = position.length();
    if radius <= 0.0 {
        return sim::DVec3::ZERO;
    }
    position * (-mu / (radius * radius * radius))
}

/// Engine acceleration for a throttle setting, along the velocity.
fn thrust(velocity: sim::DVec3, throttle: i64) -> sim::DVec3 {
    match velocity.try_normalize() {
        Some(direction) if throttle != 0 => direction * (THRUST * throttle as f64),
        _ => sim::DVec3::ZERO,
    }
}

/// One leapfrog step of whatever is in the bubble, then the regime, impact and
/// outcome checks that read the result.
pub fn flight(world: &mut GameWorld) {
    let hz = world.tick_hz();
    let dt = seconds_per_tick(hz);
    let Some((_, epoch)) = singleton::<Epoch>(world) else {
        return;
    };
    let Some((ship_entity, mut ship)) = singleton::<Ship>(world) else {
        return;
    };
    if ship.outcome != FLYING {
        ship.settled = ship.settled.saturating_add(1);
        world.put(ship_entity, ship);
        return;
    }

    if let Some(mut bubble) = world.get::<Bubble>(ship_entity).copied() {
        let Some(primary) = world.get::<Body>(bubble.primary).copied() else {
            return;
        };
        // Kick-drift-kick, carrying the acceleration: one force evaluation a
        // step, four orders better than the `v += a dt; p += v dt` every other
        // demo in this tree uses (§6 M38 item 7 table 4).
        let half = bubble.accel * (dt * 0.5);
        bubble.velocity += half;
        bubble.position += bubble.velocity * dt;
        let burn = thrust(bubble.velocity, bubble.throttle);
        bubble.accel = gravity(bubble.position, primary.mu) + burn;
        bubble.velocity += bubble.accel * (dt * 0.5);

        let spent = burn.length() * dt;
        ship.fuel = (ship.fuel - spent).max(0.0);
        ship.spent += spent;
        world.put(ship_entity, bubble);
        world.put(ship_entity, ship);
        return;
    }

    let bodies = bodies(world, epoch.elapsed, hz);
    handover(world, ship_entity, epoch.elapsed, hz, &bodies);
    outcome(world, ship_entity, epoch.elapsed, hz, &bodies);
}

/// The regime *condition*: which body's conic this one is, computed from state
/// and never from an element (§6 M38 item 8 reading 1).
///
/// The test is an acceleration ratio, which is the form that does not have to
/// be re-authored per planet — and it carries hysteresis, which item 8's finding
/// 7 priced at nothing: sixty-four handovers cost 7e-7 m, so the band is sized
/// by what keeps the *decision* stable rather than by what a crossing costs.
const HANDOVER_RATIO: f64 = 2.0;

fn handover(world: &mut GameWorld, ship: Entity, elapsed: u64, hz: u32, bodies: &[Station]) {
    let Some((position, velocity, current)) = ship_state(world, ship, elapsed, hz, bodies) else {
        return;
    };
    let pull = |held: &Station| {
        let radius = (position - held.position).length();
        if radius <= 0.0 {
            f64::INFINITY
        } else {
            held.body.mu / (radius * radius)
        }
    };
    let Some(held) = station(bodies, current) else {
        return;
    };
    let mut best = held;
    let mut best_pull = pull(&held);
    for candidate in bodies {
        let strength = pull(candidate);
        if strength > best_pull * HANDOVER_RATIO {
            best = *candidate;
            best_pull = strength;
        }
    }
    if best.entity == current {
        return;
    }
    // The frame change, spelled out: the ship's velocity is against the old
    // primary, and the new conic wants it against the new one. Leaving this out
    // costs the departure planet's whole orbital velocity and produces a conic
    // that is closed, plausible, and a third of the right size.
    let relative = velocity + held.velocity - best.velocity;
    let Some(orbit) = sim::Orbit::from_state(position - best.position, relative, best.body.mu)
    else {
        return;
    };
    world.put(
        ship,
        Rails {
            orbit,
            primary: best.entity,
            since: elapsed,
        },
    );
    record(
        world,
        EVENT_HANDOVER,
        elapsed,
        best.body.index,
        orbit.eccentricity,
    );
    world.log(
        log_level::INFO,
        if best.body.index == 0 {
            "orbit: escaped"
        } else {
            "orbit: intercept"
        },
    );
}

/// Impact, capture, and the empty tank — every one of them read off the state
/// the tick produced rather than off a flag some system set.
fn outcome(world: &mut GameWorld, ship_entity: Entity, elapsed: u64, hz: u32, bodies: &[Station]) {
    let Some((position, _, _)) = ship_state(world, ship_entity, elapsed, hz, bodies) else {
        return;
    };
    let Some(mut ship) = world.get::<Ship>(ship_entity).copied() else {
        return;
    };
    let Some(rails) = world.get::<Rails>(ship_entity).copied() else {
        return;
    };

    let hit = bodies
        .iter()
        .any(|held| (position - held.position).length() < held.body.radius);
    let captured = world
        .get::<Body>(rails.primary)
        .is_some_and(|body| body.index == OCHRE.index)
        && rails.orbit.eccentricity < 1.0
        && rails.orbit.semi_major * (1.0 - rails.orbit.eccentricity) > OCHRE.radius;

    let landed = if hit {
        CRASHED
    } else if captured {
        CAPTURED
    } else if ship.fuel <= 0.0 {
        // Not a soft failure: with no delta-v the conic cannot be changed
        // again, so the mission is decided whatever the trajectory does next.
        STRANDED
    } else {
        FLYING
    };
    if landed == FLYING {
        return;
    }
    ship.outcome = landed;
    ship.settled = 0;
    world.put(ship_entity, ship);
    record(world, EVENT_OUTCOME, elapsed, u32::MAX, f64::from(landed));
    world.log(
        log_level::INFO,
        match landed {
            CAPTURED => "orbit: captured",
            CRASHED => "orbit: crashed",
            _ => "orbit: stranded",
        },
    );
}

// ---- the map ------------------------------------------------------------

/// Write every drawn position: bodies, ship, traces, star light, and the HUD's
/// numbers. The map is centred on the ship and scaled by the zoom, which is a
/// game's decision about what it looks like and not a renderer setting (§2).
pub fn present(world: &mut GameWorld) {
    let hz = world.tick_hz();
    let Some((_, epoch)) = singleton::<Epoch>(world) else {
        return;
    };
    let Some((ship_entity, ship)) = singleton::<Ship>(world) else {
        return;
    };
    let bodies = bodies(world, epoch.elapsed, hz);
    let Some((ship_position, ship_velocity, primary)) =
        ship_state(world, ship_entity, epoch.elapsed, hz, &bodies)
    else {
        return;
    };
    let scale = sim::powf(10.0, -f64::from(epoch.zoom));
    let map = |absolute: sim::DVec3| (absolute - ship_position) * scale;

    for held in &bodies {
        let radius = (held.body.radius * scale) as f32;
        if let Some(shape) = world.get_mut::<Renderable>(held.entity) {
            shape.position = map(held.position);
            shape.half_extent = sim::Vec3::splat(radius.max(MIN_DOT));
        }
    }
    if let Some(shape) = world.get_mut::<Renderable>(ship_entity) {
        shape.position = sim::DVec3::ZERO;
        shape.half_extent = sim::Vec3::splat(SHIP_DOT);
    }
    // Rewritten every tick rather than only at bootstrap, so an edit to
    // `MAP_LUX` lands on a running session like every other tunable.
    let lit = map_light();
    world.visit::<&mut Light>(|_, light| *light = lit);

    // One trace table per conic, sampled before anything is written: the visit
    // below holds column borrows and cannot ask the world for an orbit.
    let mut traces: Vec<(u32, Vec<sim::DVec3>, bool)> = Vec::new();
    for held in &bodies {
        if held.body.index == 0 {
            continue;
        }
        if let Some(rails) = world.get::<Rails>(held.entity).copied() {
            let origin = station(&bodies, rails.primary).map_or(sim::DVec3::ZERO, |at| at.position);
            traces.push((
                held.body.index,
                sample(rails.orbit, origin, &map),
                rails.orbit.eccentricity < 1.0,
            ));
        }
    }
    let anchor = station(&bodies, primary);
    let ship_orbit = match world.get::<Rails>(ship_entity).copied() {
        Some(rails) => Some(rails.orbit),
        // Lit: the trace is the *osculating* conic, which is what the shape
        // would be if the engine cut this instant — the one readout a burn is
        // steered by.
        None => anchor.and_then(|at| {
            sim::Orbit::from_state(ship_position - at.position, ship_velocity, at.body.mu)
        }),
    };
    if let Some(orbit) = ship_orbit {
        let origin = anchor.map_or(sim::DVec3::ZERO, |at| at.position);
        traces.push((
            Marker::SHIP,
            sample(orbit, origin, &map),
            orbit.eccentricity < 1.0,
        ));
    }

    world.visit::<(&Marker, &mut Renderable)>(|_, (marker, shape)| {
        let Some((_, points, closed)) = traces.iter().find(|(of, _, _)| *of == marker.of) else {
            return;
        };
        let slot = marker.slot as usize;
        let Some(from) = points.get(slot) else {
            return;
        };
        // An escape trajectory has two ends and its last sample begins no
        // segment; a closed one wraps, which is what shuts the ring.
        let to = match points.get(slot + 1) {
            Some(next) => Some(next),
            None if *closed => points.first(),
            None => None,
        };
        match to {
            Some(to) => *shape = trace_segment(*from, *to, shape.color),
            None => shape.half_extent = sim::Vec3::ZERO,
        }
    });

    hud(world, epoch, ship, ship_orbit, primary, &bodies);
}

/// [`TRACE`] points of one conic, already mapped. Each sample is the same
/// elements with the mean anomaly stepped, which is the closed form asked the
/// question it is good at — and it is why a trajectory costs nothing to draw.
///
/// Public because the golden scene draws the same rings and must draw them with
/// *this* code: `present` runs behind the ABI a reference harness cannot reach,
/// and a second copy of the stepping is the second table §4.10 forbids.
pub fn sample(
    orbit: sim::Orbit,
    origin: sim::DVec3,
    map: &impl Fn(sim::DVec3) -> sim::DVec3,
) -> Vec<sim::DVec3> {
    let span = if orbit.eccentricity < 1.0 {
        core::f64::consts::TAU
    } else {
        HYPERBOLIC_SPAN * 2.0
    };
    (0..TRACE)
        .map(|slot| {
            let at = f64::from(slot) / f64::from(TRACE) * span;
            let mean_anomaly = if orbit.eccentricity < 1.0 {
                orbit.mean_anomaly + at
            } else {
                orbit.mean_anomaly + at - HYPERBOLIC_SPAN
            };
            let stepped = sim::Orbit {
                mean_anomaly,
                ..orbit
            };
            map(origin + stepped.state_at(0.0).0)
        })
        .collect()
}

// ---- the readout --------------------------------------------------------

/// Name of a body by [`Body::index`] — the one place the fiction is spelled.
fn name_of(index: u32) -> &'static str {
    match index {
        0 => "Star",
        1 => "Verge",
        _ => "Ochre",
    }
}

/// Five lines: where you are, what the conic is, what warp is doing, what the
/// tank has left, and how it ended.
fn hud(
    world: &mut GameWorld,
    epoch: Epoch,
    ship: Ship,
    orbit: Option<sim::Orbit>,
    primary: Entity,
    bodies: &[Station],
) {
    let held = station(bodies, primary);
    let index = held.map_or(0, |at| at.body.index);
    let radius = held.map_or(1.0, |at| at.body.radius);
    let conic = match orbit {
        Some(orbit) if orbit.eccentricity < 1.0 => {
            let peri = orbit.semi_major * (1.0 - orbit.eccentricity) - radius;
            let apo = orbit.semi_major * (1.0 + orbit.eccentricity) - radius;
            format!(
                "peri {:.0} km   apo {:.0} km   e {:.3}",
                peri / 1000.0,
                apo / 1000.0,
                orbit.eccentricity
            )
        }
        Some(orbit) => format!(
            "escape   peri {:.0} km   e {:.3}",
            (orbit.semi_major * (1.0 - orbit.eccentricity) - radius) / 1000.0,
            orbit.eccentricity
        ),
        None => "no conic".to_owned(),
    };
    let lines = [
        format!("about {}", name_of(index)),
        conic,
        format!("warp x{}", epoch.warp),
        format!("fuel {:.0} m/s   spent {:.0} m/s", ship.fuel, ship.spent),
        match ship.outcome {
            CAPTURED => "CAPTURED — press R".to_owned(),
            CRASHED => "CRASHED — press R".to_owned(),
            STRANDED => "STRANDED — press R".to_owned(),
            _ => String::new(),
        },
    ];
    world.visit::<(&Readout, &mut Widget)>(|_, (row, widget)| {
        let Some(text) = lines.get(row.slot as usize) else {
            return;
        };
        // A zero rect is how a widget hides (§4.9), so an empty line is not a
        // blank panel — it is nothing at all.
        widget.rect = if text.is_empty() {
            [0.0; 4]
        } else {
            [24.0, 24.0 + f32::from(row.slot as u16) * 26.0, 520.0, 24.0]
        };
        widget.id = widget_id("orbit.readout") ^ u64::from(row.slot);
        widget.set_text(text);
    });
}

// Order in `systems` is execution order (§4.1); order in the verb lists is the
// id space a replay records (§4.7). Neither is alphabetical, neither may drift.
#[cfg(feature = "game")]
gg_ecs::gg_game! {
    components: [Body, Rails, Bubble, Ship, Epoch, Event, Marker, Readout, Renderable, Light, Eye, Widget],
    actions: ["burn", "retro", "warp_up", "warp_down", "zoom_in", "zoom_out", "restart"],
    axes: [],
    systems: [bootstrap, control, advance, flight, present],
}
