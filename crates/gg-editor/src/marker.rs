//! The selection, drawn in the scene (§6 M15.4 item 2).
//!
//! **Screen-space geometry, not a render pass.** The picked box's eight corners
//! project back through the same [`Lens`] the click was cast through, and the
//! twelve edges are stepped in device pixels the way the title bar's close
//! button is ([`Editor::in_pixels`]). What that buys is a marker without an
//! editor-only pass, inside the rule the viewport already keeps: the outline is
//! UI geometry over a hole, not chrome drawn into the game's rectangle.
//!
//! What it does not buy is **occlusion** — an outline behind a wall is drawn on
//! top of the wall. Named here rather than discovered; the depth-correct version
//! reads the frame's depth buffer, which is a pass, which is a milestone.
//!
//! Why stepped rectangles and not one rotated quad per edge: `gg_ui::DrawList`
//! has no such primitive, and giving it one would mean a quad the clip stack
//! cannot cut, since clipping there is rectangle intersection. A stepped line
//! clips like everything else, so an outline of a box half off the pane stops at
//! the pane's edge instead of drawing across the tree. The cost is one quad per
//! row an edge crosses rather than one per edge, which is why [`segment`] cuts
//! to the view *before* it steps: without that, a corner near the near plane
//! projects to a coordinate whose line would be stepped for as long as it took.

use crate::pick::{EDGES, Lens, corners};
use crate::place;
use crate::{ACCENT, ANGLES, Editor, PICKED, STEPS, Tool};
use gg_ecs::boundary::Renderable;
use gg_math::sim;
use gg_ui::WidgetId;
use gg_ui::draw::{DrawList, Rect};

/// How thick an outline's edge is, in device pixels per unit of UI scale — so it
/// thickens with the panels rather than thinning into invisibility at 4K.
const STROKE: f32 = 1.0;

/// Chords in a lamp's range circle. Fixed rather than adaptive — see
/// [`Editor::circle`] — and 32 is where a circle four hundred pixels across
/// stops reading as a polygon.
const CIRCLE: usize = 32;

/// Half the arm of an unselected marker's cross, in logical units. Small: there
/// may be one of these per light in the level, and the job is "there is one
/// here", not "read this".
const MARK: f32 = 4.0;

/// Draw the markers for placements with no geometry — lights and environment
/// volumes (§6 M72).
///
/// A knob for [`crate::panels::LEGEND`]'s reason and not a second policy: these
/// are drawn *into* the picture, and a screenshot or a look at a dark corner
/// wants them gone without closing the pane. On by default, because the report
/// that built them was about not being able to find a light. Not `recorded` —
/// it moves no click and declares no widget, and the pick reads the world
/// rather than what was drawn, so a session recorded with markers off replays
/// the same selections with them on.
pub(crate) static MARKERS: gg_core::cvar::CVar = gg_core::cvar::CVar::new_bool(
    "d.editor_markers",
    true,
    "draw markers for lights and environment volumes",
);

/// Half of `ink`'s colour at the same alpha — a second line of the same thing,
/// read as further off rather than as a different subject.
///
/// Arithmetic on the packed bytes and not a blend against the background: the
/// marker is drawn over the game's picture, and what is behind it is not this
/// crate's to know.
fn dim(ink: u32) -> u32 {
    (ink & 0xff00_0000) | ((ink >> 1) & 0x007f_7f7f)
}

/// The viewport as the pixel rectangle a marker is drawn into, and the camera
/// it is drawn through.
///
/// One value rather than three arguments threaded through six signatures, and
/// it carries [`Pane::at`] so that the projection every one of them performs is
/// written once. The lens rides along because it is never separable from the
/// rectangle: a pane position is a projection, and a projection needs both.
#[derive(Clone, Copy)]
struct Pane<'a> {
    bounds: Rect,
    scale: f32,
    lens: &'a Lens,
}

impl Pane<'_> {
    /// Where `point` lands in pixels, or `None` for a point at or behind the
    /// near plane — which has no position on the pane at all, and whose
    /// projection is a division by a depth approaching zero.
    fn at(self, point: sim::DVec3) -> Option<(f32, f32)> {
        let (uv, depth) = self.lens.project(point);
        let on = (
            self.bounds.x + uv.0 as f32 * self.bounds.w,
            self.bounds.y + uv.1 as f32 * self.bounds.h,
        );
        (depth >= self.lens.near).then_some(on)
    }
}

/// How long an axis handle is, in logical units — a screen length, not a world
/// one ([`Lens::metres_per_unit`]).
const HANDLE: f64 = 34.0;
/// The grab square at a handle's tip, in logical units.
const PAD: f32 = 7.0;

/// How far out along an arm its grab band starts, as a fraction of the arm.
///
/// Not zero, and that is the whole of the arbitration (§6 M20 item 10): all
/// three arms meet at the selection's centre, so a band that reached it would
/// make every one of them cover the same pixels and the answer to "which axis"
/// would be whichever was declared last. Starting outside the meeting point
/// leaves three disjoint bands whenever the arms are not nearly parallel, and
/// declaring the nearest one last settles the case where they are ([`Editor::gizmo`]).
const BAND_FROM: f64 = 0.42;

/// How far past the tip it ends — the pad is centred *on* the tip, so the band
/// has to reach past it or the outermost half of the target would not be in it.
const BAND_TO: f64 = 1.0 + 0.5 * (PAD as f64) / HANDLE;

/// Half the grab band's width, in logical units. Wider than the drawn arm on
/// purpose: what an operator aims at is the line they can see, and a target
/// exactly as wide as a one-unit stroke is a target that is missed.
const BAND: f32 = 5.0;

/// The least a [`crate::Tool::Scale`] drag may leave on an axis, in metres. A
/// zero half-extent is a box with no thickness — invisible, unpickable, and
/// reachable by one drag too far.
const LEAST_EXTENT: f64 = 0.05;

/// What each axis is drawn in: X red, Y green, Z blue, which is the one colour
/// convention every tool an operator has used already agrees on.
pub(crate) const AXIS_INK: [u32; 3] = [0xffd0_5a4a, 0xff5a_c07a, 0xff4a_86d0];

/// The three world axes a handle translates along. **World**, not the box's own:
/// [`Renderable::position`] is a world position, so a local-axis gizmo would be
/// a second frame of reference for the operator to keep track of and a rotation
/// to invert on every write.
pub(crate) const AXES: [sim::DVec3; 3] = [sim::DVec3::X, sim::DVec3::Y, sim::DVec3::Z];

const GRIP: WidgetId = WidgetId::new("editor.gizmo");

/// One handle this tick: which of [`AXES`] it is, where its tip landed on the
/// pane, and what a metre along it covers there.
type Arm = (usize, (f64, f64), (f64, f64));

/// Logical units of drag per degree, for [`Tool::Turn`]. A screen rate and not
/// a world one: a rotation has no metres to be measured in, and one that turned
/// faster when zoomed out would be a different gesture at every framing.
const UNITS_PER_DEGREE: f64 = 1.5;

/// A gizmo drag in progress (§6 M15.4 item 3, §6 M20 item 10).
#[derive(Clone, Copy)]
pub(crate) struct Gizmo {
    entity: gg_ecs::Entity,
    /// Which of [`AXES`].
    axis: usize,
    /// What this drag writes. Latched at the press rather than read per tick,
    /// so cycling the tool mid-gesture cannot leave half a move and half a
    /// rotation applied against one origin.
    tool: Tool,
    /// The pointer where the press landed, in [`gg_input::AXIS_SCALE`]ths.
    /// Integer, for [`crate::camera`]'s reason: the differences accumulate
    /// exactly, so a replayed drag lands where the recorded one did.
    from: (i32, i32),
    /// Which component the box came out of, held for the drag rather than
    /// re-resolved per tick: a drag that changed kind mid-gesture would write
    /// the last tick's metres into a different field.
    kind: place::Kind,
    /// The whole box as it was then. Every tick writes `origin + quantized`
    /// rather than adding to the last one, so a drag out and back lands exactly
    /// where it started and a slow drag and a fast one over the same distance
    /// agree — the property that also makes the three tools one code path.
    origin: Renderable,
    /// What one metre along the axis covers on the pane, in logical units,
    /// resolved at the press and held for the drag. Recomputing it per tick
    /// would make the gizmo accelerate under a steady hand, since the thing it
    /// is attached to is moving.
    per_metre: (f64, f64),
}

impl Editor {
    /// Outline the selection inside `view`, whatever component put it there.
    ///
    /// Silent for a selection with no placement at all — a tree row may name any
    /// entity, and a `Prefs` is nowhere. Drawn in every play state, because the
    /// point of it while the game runs is watching the thing you selected move.
    ///
    /// What is drawn is [`place::Kind`]'s business and not this one's: a body
    /// and a volume are boxes, a lamp is the **sphere of its own range** (§6
    /// M72) and a sun is an arrow, because the question an operator has about
    /// each is a different question.
    pub(crate) fn mark(&mut self, world: &gg_ecs::World, view: Rect, lens: &Lens) {
        let Some(entity) = self.selected else { return };
        let Some((kind, box_)) = place::of(world, entity) else {
            return;
        };
        let Some(pane) = self.pane(view, lens) else {
            return;
        };
        // The list is already inside `Editor::tick`'s fit transform, so this
        // undoes it exactly as `in_pixels` and `text` do.
        self.list.push_transform((0.0, 0.0), 1.0 / pane.scale);
        self.shape(world, entity, kind, &box_, pane, ACCENT);
        if !matches!(kind, place::Kind::Sun) {
            // The centre, so a box too small or too far to have a readable
            // outline is still visibly the selected one. Two pixels, in the
            // tree's own selected colour.
            if let Some((x, y)) = pane.at(box_.position) {
                let dot = Rect::new(
                    x - pane.scale,
                    y - pane.scale,
                    pane.scale * 2.0,
                    pane.scale * 2.0,
                );
                self.list.rect(dot.intersect(&pane.bounds), PICKED);
            }
        }
        self.list.pop_transform();
    }

    /// Every placement in the world that has no geometry of its own, drawn
    /// where it is (§6 M72).
    ///
    /// **Always, not only when selected**, which is the whole point: a light you
    /// have to select to see is a light you have to find first. Small and in the
    /// thing's own colour — a lamp's marker is the colour it emits, so four
    /// lamps in a room are four distinguishable crosses rather than four
    /// identical ones.
    ///
    /// Bodies are skipped: they are already in the picture, and a cross on every
    /// box in the level is not a marker, it is a texture.
    pub(crate) fn markers(&mut self, world: &gg_ecs::World, view: Rect, lens: &Lens) {
        if !MARKERS.bool() {
            return;
        }
        let Some(pane) = self.pane(view, lens) else {
            return;
        };
        let selected = self.selected;
        let mut found: Vec<(gg_ecs::Entity, place::Kind, gg_ecs::boundary::Renderable)> =
            Vec::new();
        place::each(world, &mut |entity, kind, box_| {
            // The selection is drawn by `mark` at full strength, and drawing it
            // twice would leave the dim pass on top of the bright one.
            if matches!(kind, place::Kind::Body) || Some(entity) == selected {
                return;
            }
            found.push((entity, kind, box_));
        });
        if found.is_empty() {
            return;
        }
        self.list.push_transform((0.0, 0.0), 1.0 / pane.scale);
        for (entity, kind, box_) in found {
            let ink = 0xff00_0000 | box_.color;
            match kind {
                // A sun's arrow *is* its marker — it has no position to cross —
                // and a volume's box is drawn in full, because "where does the
                // light in this room stop" is the question a volume exists to
                // answer and a level holds a handful of them. A lamp gets the
                // cross alone: its shape is three circles of 32 chords, which is
                // an outline worth drawing for the one that was asked about and
                // not for every light in the level.
                place::Kind::Sun | place::Kind::Volume => {
                    self.shape(world, entity, kind, &box_, pane, ink);
                }
                _ => self.cross(box_.position, pane, ink),
            }
        }
        self.list.pop_transform();
    }

    /// The outline a placement of `kind` draws, in `ink`.
    ///
    /// Inside the caller's pixel transform, so every coordinate here is already
    /// in device pixels.
    fn shape(
        &mut self,
        world: &gg_ecs::World,
        entity: gg_ecs::Entity,
        kind: place::Kind,
        box_: &gg_ecs::boundary::Renderable,
        pane: Pane,
        ink: u32,
    ) {
        match kind {
            place::Kind::Body => {
                let corners = corners(box_);
                for (a, b) in EDGES {
                    self.edge(corners[a], corners[b], pane, ink);
                }
            }
            // Two boxes, and the second is the point of drawing a volume at all:
            // `Sky::fade` is metres *outside* the box over which the environment
            // gives way, so the inner one is where this room's light is entirely
            // its own and the outer one is where it stops mattering. An operator
            // asking "why does the light change here" is asking about the band
            // between them, and one box cannot show a band.
            place::Kind::Volume => {
                let inner = corners(box_);
                for (a, b) in EDGES {
                    self.edge(inner[a], inner[b], pane, ink);
                }
                let fade = world
                    .get::<gg_ecs::boundary::Sky>(entity)
                    .map_or(0.0, |sky| sky.fade);
                if fade > 0.0 {
                    let band = Renderable {
                        half_extent: box_.half_extent + sim::Vec3::splat(fade),
                        ..*box_
                    };
                    let outer = corners(&band);
                    for (a, b) in EDGES {
                        self.edge(outer[a], outer[b], pane, dim(ink));
                    }
                }
            }
            // Three great circles rather than a cube of the same size: a light
            // falls off to a *radius*, and a box drawn at that radius overstates
            // its corners by the root of three — which is exactly the operator's
            // question ("does this reach the wall?") answered wrongly.
            place::Kind::Lamp => {
                let reach = f64::from(box_.half_extent.x);
                for (u, v) in [
                    (sim::DVec3::X, sim::DVec3::Y),
                    (sim::DVec3::Y, sim::DVec3::Z),
                    (sim::DVec3::Z, sim::DVec3::X),
                ] {
                    self.circle(box_.position, reach, u, v, pane, ink);
                }
            }
            // The direction it travels, from the world origin, and a cross at
            // the far end so which way it points is legible when the arrow is
            // nearly end-on. A sun is not anywhere; the origin is a convention
            // and the *slope* is the information.
            place::Kind::Sun => {
                let Some(light) = world.get::<gg_ecs::boundary::Light>(entity) else {
                    return;
                };
                let travel = sim::DVec3::new(
                    f64::from(light.direction.x),
                    f64::from(light.direction.y),
                    f64::from(light.direction.z),
                );
                let Some(unit) = travel.try_normalize() else {
                    return;
                };
                let tip = unit * place::SUN_ARM;
                self.edge(-tip, tip, pane, ink);
                self.cross(tip, pane, ink);
            }
        }
    }

    /// One world-space segment, cut at the near plane and stepped onto the pane.
    fn edge(&mut self, from: sim::DVec3, to: sim::DVec3, pane: Pane, ink: u32) {
        let (mut from, mut to) = (from, to);
        let lens = pane.lens;
        let (depth_a, depth_b) = (lens.project(from).1, lens.project(to).1);
        // Both ends behind the near plane: the edge is not on the pane at all.
        // One end behind it: cut there, or the projection divides by a depth
        // approaching zero and the edge becomes a stripe.
        if depth_a < lens.near && depth_b < lens.near {
            return;
        }
        if depth_a < lens.near {
            from = lens.clip_near(from, to);
        } else if depth_b < lens.near {
            to = lens.clip_near(to, from);
        }
        let (Some(a), Some(b)) = (pane.at(from), pane.at(to)) else {
            return;
        };
        segment(&mut self.list, pane.bounds, a, b, pane.scale, ink);
    }

    /// A circle of `radius` about `centre`, in the plane the unit vectors `u`
    /// and `v` span, as [`CIRCLE`] chords.
    ///
    /// Chords and not an arc primitive for [`segment`]'s reason — `DrawList` has
    /// rectangles — and the count is fixed rather than adaptive because a
    /// circle that gains segments as the camera approaches it is a circle that
    /// shimmers while an operator walks.
    fn circle(
        &mut self,
        centre: sim::DVec3,
        radius: f64,
        u: sim::DVec3,
        v: sim::DVec3,
        pane: Pane,
        ink: u32,
    ) {
        let step = core::f64::consts::TAU / CIRCLE as f64;
        let on = |i: usize| {
            let angle = step * i as f64;
            // `gg_math::sim` rather than `std`, per §3's ban — this is drawing
            // and not sim state, and the ban is on the *call* so that no reader
            // has to work out which.
            centre + u * (radius * sim::cos(angle)) + v * (radius * sim::sin(angle))
        };
        let mut previous = on(0);
        for i in 1..=CIRCLE {
            let next = on(i);
            self.edge(previous, next, pane, ink);
            previous = next;
        }
    }

    /// A small screen-constant cross at a world point — the marker for a thing
    /// whose whole placement is one position.
    fn cross(&mut self, at: sim::DVec3, pane: Pane, ink: u32) {
        let Some((x, y)) = pane.at(at) else { return };
        let arm = MARK * pane.scale;
        let bar = STROKE * pane.scale;
        for rect in [
            Rect::new(x - arm, y - bar * 0.5, arm * 2.0, bar),
            Rect::new(x - bar * 0.5, y - arm, bar, arm * 2.0),
        ] {
            self.list.rect(rect.intersect(&pane.bounds), ink);
        }
    }

    /// The viewport as the pixel rectangle everything above draws into, or
    /// `None` when it has collapsed to nothing.
    fn pane<'a>(&self, view: Rect, lens: &'a Lens) -> Option<Pane<'a>> {
        let scale = self.fit.scale;
        let px = |v: f32| v * scale;
        let bounds = Rect::new(px(view.x), px(view.y), px(view.w), px(view.h));
        (!bounds.is_empty()).then_some(Pane {
            bounds,
            scale,
            lens,
        })
    }

    /// The three axis handles on the selection, and the drag that moves, sizes
    /// or turns it (§6 M15.4 item 3, §6 M20 item 10).
    ///
    /// **The write is what a nudge writes.** A drag resolves to a scalar along
    /// one world axis, that is quantized to the inspector's own step, and the
    /// result goes into `Renderable` through `World` like every other editor
    /// edit — hashed, replayed, and undone by the same machinery. The
    /// quantization is load-bearing rather than a nicety: the pointer is fixed
    /// point at 1/1024 (§4.7) but the camera it is projected through is *host*
    /// state, so an unquantized drag would make a hashed world value depend on a
    /// float the replay reconstructs rather than reads.
    ///
    /// **The arm is the target, not the dot at its end.** Through M20 only a
    /// pad at the tip was hit-tested, which put the whole of a handle's visible
    /// length outside the thing it looked like — the arms drew a control and
    /// answered to a fraction of it. See [`BAND_FROM`] for how three bands that
    /// meet at one point are arbitrated.
    ///
    /// Stopped only, on [`crate::camera`]'s rule: dragging a handle while the
    /// sim is writing the same field every tick is a fight, not an edit.
    pub(crate) fn gizmo(
        &mut self,
        world: &mut gg_ecs::World,
        frame: &crate::Frame,
        view: Rect,
        lens: &Lens,
    ) {
        self.arms = [None; 3];
        let stopped = matches!(frame.play, crate::Play::Stopped);
        let tool = self.tool;
        let on = self.selected.filter(|_| stopped).and_then(|entity| {
            let (kind, box_) = place::of(world, entity)?;
            // A tool the kind has no field for is refused here rather than at
            // the write, so no arm is drawn for a gesture that cannot land
            // (§6 M72) — a lamp and a volume have no rotation, and a sun has no
            // placement at all.
            if !kind.takes(tool) {
                return None;
            }
            let (centre, depth) = lens.project(box_.position);
            // Behind the camera there is no centre to hang arms off, and the
            // projection there is a division by a depth approaching zero.
            (depth >= lens.near).then_some((entity, kind, box_, centre, depth))
        });
        let Some((entity, kind, box_, centre, depth)) = on else {
            self.release(world);
            return;
        };
        let at = |uv: (f64, f64)| {
            (
                f64::from(view.x) + uv.0 * f64::from(view.w),
                f64::from(view.y) + uv.1 * f64::from(view.h),
            )
        };
        let (cx, cy) = at(centre);
        // One length for all three, from the depth of the centre alone: an
        // arm's *drawn* length still shortens as it turns towards the camera,
        // which is the foreshortening that says which way it points.
        let metres = lens.metres_per_unit(depth, f64::from(view.h)) * HANDLE;
        let mut up: Vec<Arm> = Vec::new();
        for (axis, direction) in AXES.iter().enumerate() {
            let tip = box_.position + *direction * metres;
            let (uv, tip_depth) = lens.project(tip);
            let (tx, ty) = at(uv);
            let per_metre = ((tx - cx) / metres, (ty - cy) / metres);
            // An arm pointing at or away from the camera collapses onto the
            // centre, and a band there would sit on top of the entity and
            // swallow every click meant for the pick beneath it — a handle with
            // no direction on the pane is a handle there is no way to drag, so
            // it is not offered. The tip going behind the near plane is the same
            // case seen from the other side.
            // Squared, so no root reaches this at all — §3 bans `hypot` for the
            // rounding, and the comparison never needed one.
            let span = (tx - cx) * (tx - cx) + (ty - cy) * (ty - cy);
            let least = f64::from(BAND) * 2.0;
            if span < least * least || tip_depth < lens.near {
                continue;
            }
            up.push((axis, (tx, ty), per_metre));
        }
        // Declaration order *is* the arbitration: the router takes the last
        // widget under the pointer (§4.9), so the arm the pointer is nearest
        // goes last and wins wherever two bands still overlap. Stable, because
        // it is a sort by one key and not a filter — all three are declared
        // every tick, so no id appears and vanishes as the pointer moves.
        let pointer = self.router.pointer().position();
        let aim = (f64::from(pointer.0), f64::from(pointer.1));
        let closest = up
            .iter()
            .map(|(axis, tip, _)| (*axis, to_segment(aim, (cx, cy), *tip)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(axis, _)| axis);
        up.sort_by_key(|(axis, ..)| Some(*axis) == closest);

        let mut taken = None;
        for (axis, (tx, ty), per_metre) in up {
            let response = self
                .router
                .hit(GRIP.indexed(axis as u64), band((cx, cy), (tx, ty)));
            let lit = response.held || response.hovered;
            let ink = match lit {
                true => ACCENT,
                false => AXIS_INK[axis],
            };
            self.arm((cx, cy), (tx, ty), view, ink);
            let pad = Rect::new(tx as f32 - PAD * 0.5, ty as f32 - PAD * 0.5, PAD, PAD);
            if let Some(slot) = self.arms.get_mut(axis) {
                *slot = Some((tx as f32, ty as f32));
            }
            self.list.rect(pad, ink);
            if response.held {
                taken = Some((axis, per_metre));
            }
        }
        // `held` and not a press edge: the router keeps the capture once a press
        // lands, so this is true for the whole drag and false the tick it ends.
        let Some((axis, per_metre)) = taken else {
            self.release(world);
            return;
        };
        if self.gizmo.is_none() {
            // The press edge, and the only place a drag records one: a step per
            // *tick* would make undo walk back through the gesture frame by
            // frame (§6 M15.4 item 4).
            self.history.edit(world);
        }
        let grab = *self.gizmo.get_or_insert(Gizmo {
            entity,
            kind,
            axis,
            tool: self.tool,
            from: self.router.pointer().raw(),
            origin: box_,
            per_metre,
        });
        // The pointer's travel since the press, resolved onto the arm: how far
        // along it the operator has dragged, in metres.
        let now = self.router.pointer().raw();
        let scale = f64::from(gg_input::AXIS_SCALE);
        let moved = (
            f64::from(now.0 - grab.from.0) / scale,
            f64::from(now.1 - grab.from.1) / scale,
        );
        let along = grab.per_metre.0 * grab.per_metre.0 + grab.per_metre.1 * grab.per_metre.1;
        if along < f64::EPSILON {
            // The arm points straight at the camera: there is no direction on
            // the pane to resolve a drag onto, and any answer would be noise.
            return;
        }
        let raw = (moved.0 * grab.per_metre.0 + moved.1 * grab.per_metre.1) / along;
        let want = shaped(&grab, raw, self.step);
        if place::put(world, grab.entity, grab.kind, grab.axis, &want) {
            self.edits += 1;
        }
    }

    /// End a drag, if one was running: one log line for the whole gesture rather
    /// than one per tick, on §5.6c's reasoning.
    fn release(&mut self, world: &gg_ecs::World) {
        let Some(grab) = self.gizmo.take() else {
            return;
        };
        // Each tool's own answer to "by how much", because a rotation has no
        // metres and a resize's metres are not the position's. Read back through
        // `place` rather than off `Renderable`, so a lamp's line reports the
        // range it now has rather than nothing at all.
        let by = place::of(world, grab.entity).map_or(0.0, |(_, box_)| {
            let component = |v: sim::Vec3| f64::from([v.x, v.y, v.z][grab.axis.min(2)]);
            match grab.tool {
                Tool::Move => (box_.position - grab.origin.position).length(),
                Tool::Scale => component(box_.half_extent) - component(grab.origin.half_extent),
                Tool::Turn => box_.rotation.dot(grab.origin.rotation),
            }
        });
        tracing::info!(
            entity = grab.entity.index(),
            kind = grab.kind.label(),
            tool = grab.tool.label(),
            axis = ["x", "y", "z"][grab.axis.min(2)],
            by,
            "editor: dragged"
        );
    }

    /// One axis arm, from the selection's centre to a handle's tip.
    fn arm(&mut self, from: (f64, f64), to: (f64, f64), view: Rect, color: u32) {
        let scale = self.fit.scale;
        let px = |v: f64| (v * f64::from(scale)) as f32;
        let bounds = Rect::new(
            view.x * scale,
            view.y * scale,
            view.w * scale,
            view.h * scale,
        );
        self.list.push_transform((0.0, 0.0), 1.0 / scale);
        segment(
            &mut self.list,
            bounds,
            (px(from.0), px(from.1)),
            (px(to.0), px(to.1)),
            scale,
            color,
        );
        self.list.pop_transform();
    }
}

/// The grab rectangle for an arm from `centre` to `tip`: the outer part of it,
/// inflated by [`BAND`].
///
/// An axis-aligned box around an oblique arm is a loose bound, and deliberately
/// so — the router hit-tests rectangles, and tightening this to the segment
/// would mean either a primitive `gg_ui` does not have or a hit test this crate
/// runs behind the router's back. Loose is safe because [`BAND_FROM`] keeps the
/// three apart and the nearest-arm sort settles what is left.
fn band(centre: (f64, f64), tip: (f64, f64)) -> Rect {
    let run = (tip.0 - centre.0, tip.1 - centre.1);
    let end = |t: f64| (centre.0 + run.0 * t, centre.1 + run.1 * t);
    let (from, to) = (end(BAND_FROM), end(BAND_TO));
    let (x, y) = (from.0.min(to.0) as f32, from.1.min(to.1) as f32);
    let (w, h) = ((from.0 - to.0).abs() as f32, (from.1 - to.1).abs() as f32);
    Rect::new(x - BAND, y - BAND, w + BAND * 2.0, h + BAND * 2.0)
}

/// Squared distance from `point` to the segment `from`–`to`.
///
/// Squared throughout: this only ever orders three candidates against each
/// other, and a root would add rounding to a comparison that never needed one.
fn to_segment(point: (f64, f64), from: (f64, f64), to: (f64, f64)) -> f64 {
    let run = (to.0 - from.0, to.1 - from.1);
    let len = run.0 * run.0 + run.1 * run.1;
    let reach = (point.0 - from.0, point.1 - from.1);
    // A degenerate arm is the centre itself; the caller has already dropped
    // those, and this keeps the division honest regardless.
    let t = match len < f64::EPSILON {
        true => 0.0,
        false => ((reach.0 * run.0 + reach.1 * run.1) / len).clamp(0.0, 1.0),
    };
    let off = (reach.0 - run.0 * t, reach.1 - run.1 * t);
    off.0 * off.0 + off.1 * off.1
}

/// The box a drag of `raw` metres along its arm asks for, quantized to the
/// grain at `step`.
///
/// Written from the origin every tick rather than accumulated, which is what
/// makes a drag out and back exact and a slow drag equal to a fast one — see
/// [`Gizmo::origin`]. Splitting the three tools *here* rather than at the call
/// site keeps that property one function's to hold.
fn shaped(grab: &Gizmo, raw: f64, step: usize) -> Renderable {
    let quantize = |value: f64, grain: f64| (value / grain).round() * grain;
    let mut out = grab.origin;
    match grab.tool {
        Tool::Move => {
            let grain = STEPS.get(step).copied().unwrap_or(1.0);
            out.position = grab.origin.position + AXES[grab.axis] * quantize(raw, grain);
        }
        Tool::Scale => {
            let grain = STEPS.get(step).copied().unwrap_or(1.0);
            let was = [
                grab.origin.half_extent.x,
                grab.origin.half_extent.y,
                grab.origin.half_extent.z,
            ];
            let want = (f64::from(was[grab.axis]) + quantize(raw, grain)).max(LEAST_EXTENT);
            let mut half = was;
            half[grab.axis] = want as f32;
            out.half_extent = sim::Vec3::new(half[0], half[1], half[2]);
        }
        Tool::Turn => {
            // Back out of metres into the logical units the arm is drawn in: a
            // rotation is a screen gesture, and one whose rate came from the
            // world would turn faster the further out the camera was.
            let along = grab.per_metre.0 * grab.per_metre.0 + grab.per_metre.1 * grab.per_metre.1;
            let units = raw * sim::sqrt(along);
            let grain = ANGLES.get(step).copied().unwrap_or(15.0);
            let degrees = quantize(units / UNITS_PER_DEGREE, grain);
            let turn = sim::DQuat::from_axis_angle(
                AXES[grab.axis],
                degrees * core::f64::consts::PI / 180.0,
            );
            // Pre-multiplied, because [`AXES`] are **world** axes: post-
            // multiplying would turn about the box's own, which is a second
            // frame of reference this gizmo deliberately does not have.
            out.rotation = turn
                .mul(grab.origin.rotation)
                .try_normalize()
                .unwrap_or(grab.origin.rotation);
        }
    }
    out
}

/// One edge, in device pixels, cut to `bounds` and stepped along it.
///
/// Run-length rather than per pixel: the walk visits every step of the major
/// axis but emits a rectangle only where the minor one changes, so a near-level
/// edge is two or three quads and only a 45° one costs a quad per step.
pub(crate) fn segment(
    list: &mut DrawList,
    bounds: Rect,
    from: (f32, f32),
    to: (f32, f32),
    scale: f32,
    color: u32,
) {
    let Some((from, to)) = clip(bounds, from, to) else {
        return;
    };
    let (dx, dy) = (to.0 - from.0, to.1 - from.1);
    // Clipped, so this is at most the view's diagonal; a NaN from a degenerate
    // projection saturates to zero on the cast and `max` makes it one step.
    let steps = dx.abs().max(dy.abs()).round().max(1.0);
    let flat = dx.abs() >= dy.abs();
    let stroke = (STROKE * scale).max(1.0);
    let point = |i: u32| {
        let t = i as f32 / steps;
        (from.0 + dx * t, from.1 + dy * t)
    };
    let split = |p: (f32, f32)| match flat {
        true => (p.0, p.1.floor()),
        false => (p.1, p.0.floor()),
    };
    let mut emit = |from: f32, to: f32, row: f32| {
        let (near, far) = (from.min(to), from.max(to));
        // At least a stroke long, so the run that is one step wide — and the
        // endpoint's own row, flushed below — is a dot rather than nothing.
        let along = (far - near).max(stroke);
        list.rect(
            match flat {
                true => Rect::new(near, row, along, stroke),
                false => Rect::new(row, near, stroke, along),
            },
            color,
        );
    };
    let (mut run, mut row) = split(from);
    let last = steps as u32;
    for i in 1..=last {
        let (major, minor) = split(point(i));
        if minor != row {
            emit(run, major, row);
            (run, row) = (major, minor);
        }
        // The tail: the run still open when the walk ran out, which for a line
        // ending on a fresh row is that row's single step. Without it the last
        // pixel of every sloped edge is missing and the corners do not meet.
        if i == last {
            emit(run, major, row);
        }
    }
}

/// The part of the segment inside `rect`, or `None` when it misses entirely.
///
/// Liang–Barsky: four ratios against the four edges, which needs no case
/// analysis of *which* edge was crossed and so has no corner case to get wrong.
fn clip(rect: Rect, from: (f32, f32), to: (f32, f32)) -> Option<((f32, f32), (f32, f32))> {
    let (dx, dy) = (to.0 - from.0, to.1 - from.1);
    let (mut enter, mut exit) = (0.0f32, 1.0f32);
    for (edge, distance) in [
        (-dx, from.0 - rect.x),
        (dx, rect.right() - from.0),
        (-dy, from.1 - rect.y),
        (dy, rect.bottom() - from.1),
    ] {
        if edge == 0.0 {
            // Parallel to this edge: outside it means outside, whatever the
            // other three say.
            if distance < 0.0 {
                return None;
            }
            continue;
        }
        let t = distance / edge;
        match edge < 0.0 {
            true if t > exit => return None,
            true => enter = enter.max(t),
            false if t < enter => return None,
            false => exit = exit.min(t),
        }
    }
    Some((
        (from.0 + dx * enter, from.1 + dy * enter),
        (from.0 + dx * exit, from.1 + dy * exit),
    ))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    const VIEW: Rect = Rect {
        x: 100.0,
        y: 50.0,
        w: 400.0,
        h: 300.0,
    };

    #[test]
    fn a_segment_outside_the_view_is_not_drawn_at_all() {
        assert!(clip(VIEW, (0.0, 0.0), (50.0, 40.0)).is_none());
        assert!(clip(VIEW, (600.0, 100.0), (700.0, 200.0)).is_none());
        // Parallel to an edge and beyond it — the case the ratio form divides
        // by zero on if it is not guarded.
        assert!(clip(VIEW, (0.0, 10.0), (900.0, 10.0)).is_none());
    }

    #[test]
    fn a_segment_crossing_the_view_is_cut_to_it() {
        let (a, b) = clip(VIEW, (0.0, 200.0), (900.0, 200.0)).expect("straight across");
        // To a fraction of a pixel: the cut is a ratio, so the crossing lands on
        // the edge to within the rounding of one division.
        assert!((a.0 - VIEW.x).abs() < 1e-3 && a.1 == 200.0, "{a:?}");
        assert!((b.0 - VIEW.right()).abs() < 1e-3 && b.1 == 200.0, "{b:?}");
        // Wholly inside: both ends survive untouched, so an ordinary edge is not
        // shortened by a fraction of a pixel every frame.
        let inside = ((150.0, 100.0), (300.0, 200.0));
        assert_eq!(clip(VIEW, inside.0, inside.1), Some(inside));
    }

    /// The run-length walk covers the line without gaps and without a quad per
    /// pixel: a level edge is one rectangle, a 45° one is a quad per step, and
    /// both start and end where they were told to.
    #[test]
    fn a_stepped_line_is_continuous_and_costs_what_its_slope_costs() {
        let quads = |from, to| {
            let mut list = DrawList::default();
            segment(&mut list, VIEW, from, to, 1.0, 0xffff_ffff);
            list.vertices().len() / 6
        };
        assert_eq!(quads((120.0, 100.0), (400.0, 100.0)), 1, "level");
        assert_eq!(quads((120.0, 100.0), (120.0, 300.0)), 1, "plumb");
        // 200 across and 200 down: a quad a row, plus the endpoint's own. The
        // worst case, and the only slope at which this costs what stepping
        // every pixel would.
        assert_eq!(quads((150.0, 100.0), (350.0, 300.0)), 201);
        // Shallow: 200 across, 4 down — five rows, not two hundred.
        assert_eq!(quads((150.0, 100.0), (350.0, 104.0)), 5);
    }

    /// Every quad a segment draws is inside the view, so an outline cannot leak
    /// over the pane beside it even before `DrawList`'s own clip.
    #[test]
    fn nothing_a_segment_draws_lands_outside_the_view() {
        let mut list = DrawList::default();
        // From well outside, through the view, to well outside the other side.
        segment(
            &mut list,
            VIEW,
            (-900.0, -400.0),
            (1400.0, 900.0),
            1.0,
            0xffff_ffff,
        );
        assert!(!list.vertices().is_empty(), "it does cross the view");
        for vertex in list.vertices() {
            let (x, y) = (vertex.pos[0], vertex.pos[1]);
            assert!(
                x >= VIEW.x - 1.0
                    && x <= VIEW.right() + 1.0
                    && y >= VIEW.y - 1.0
                    && y <= VIEW.bottom() + 1.0,
                "a quad at ({x}, {y}) is outside {VIEW:?}"
            );
        }
    }
}
