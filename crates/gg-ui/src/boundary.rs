//! The host side of the UI protocol (§4.9, §6 M13): `gg_ecs::boundary::Widget`
//! components in, one vertex stream out, hit state back into the world.
//!
//! A game crate may not link this crate (§3's deny pin), so it declares its UI
//! as components and this is what reads them. Nothing here decides what the UI
//! *is* — kinds, colours, text and order are all the game's, and the host's one
//! visual opinion is the hover tint, which exists because a game that had to
//! author three colours per button would author one and look dead.
//!
//! # Two passes over one query, in world order
//!
//! Widgets are copied out, sorted into draw order, drawn, then written back by
//! walking the same query a second time and indexing sequentially. That is
//! sound because nothing structural happens between the two passes and
//! `World::each`'s order is a documented guarantee — archetype creation order,
//! then dense row — rather than an accident of layout.
//!
//! # Hit-testing happens in canvas units
//!
//! Everywhere else in this crate a widget hit-tests
//! [`DrawList::place`](crate::DrawList::place)'s result, so the rectangle it
//! draws into and the rectangle it can be hit in are the same object. Here the
//! router is fed the *untransformed* canvas rect instead, and the transform is
//! left to the draw list. The reason is the replay criterion: pointer motion
//! arrives as device deltas, so a pointer integrated in target pixels lands on
//! a different widget at a different window size, and a recorded session would
//! only replay on the monitor it was recorded on. The canvas→target transform
//! is a uniform scale and an offset — invertible, no clip — so the two spaces
//! agree up to that map. **A scrollable or clipped container would break that**
//! and would have to move hit-testing back onto `place`.

use crate::draw::{DrawList, Rect};
use crate::router::{Binding, Response, Router, Tick};
use crate::{WidgetId, font};
use gg_ecs::boundary::{CANVAS, Verbs, Widget, state, widget};
use gg_ecs::{AliasError, Query, World};
use gg_input::{ActionId, AxisId, MAX_ACTIONS, MAX_AXES};
use gg_render::ui::UiVertex;

/// Accent for the focused widget's border, and the border's width in canvas
/// units. Not the game's to choose: focus is a host concept — the router owns
/// it — so the ring that shows it is host-drawn too.
const FOCUS: u32 = 0xff7f_d0a0;
const BORDER: f32 = 1.0;

/// The cursor: fill, its one-unit halo, and its height in canvas units. Host-
/// drawn for the reason [`crate::router`]'s docs give — the pointer is an
/// integral of the replayed axis stream, so it is *not* where the OS thinks the
/// mouse is and the OS cursor cannot stand in for it. The halo is what keeps it
/// visible over a light panel.
const CURSOR: u32 = 0xffff_ffff;
const CURSOR_HALO: u32 = 0xd004_0608;
const CURSOR_HEIGHT: u32 = 9;

/// The verb names a host looks for to route a game's UI (§4.7, §4.9).
///
/// Well known rather than configured. The shell has to find these in a verb
/// list it did not write, and the alternative — a config key naming a verb name
/// — is indirection with one possible answer.
pub mod verb {
    /// Pointer motion, horizontal and vertical.
    pub const X: &str = "ui_x";
    /// See [`X`].
    pub const Y: &str = "ui_y";
    /// The click/drag button.
    pub const CLICK: &str = "ui_click";
    /// Move focus to the next widget.
    pub const FOCUS: &str = "ui_focus";
    /// One wheel notch away from the operator, and one toward.
    ///
    /// **Optional, unlike the four above**, and the only pair a build may
    /// declare half of. A UI without them scrolls nothing, which is the right
    /// answer for a HUD and for every game that had a working pointer before
    /// they existed; the editor appends them like the rest (§6 M15.1).
    pub const SCROLL_UP: &str = "ui_scroll_up";
    /// See [`SCROLL_UP`].
    pub const SCROLL_DOWN: &str = "ui_scroll_down";
}

/// Resolve [`verb`]'s names against a build's declared verbs (§4.7).
///
/// `None` if any of the four is missing, which leaves a UI that draws and
/// cannot be clicked — the right answer for a HUD, and the reason this is not
/// an error. All-or-nothing because the four are one protocol: a pointer that
/// moves and cannot press is worse than one that never appeared.
#[must_use]
pub fn binding(verbs: &Verbs) -> Option<Binding> {
    // Bounded before the id is built: `ActionId::new` panics past the limit,
    // and a verb list longer than the map is the dylib's claim, not ours.
    let find = |names: &[&str], name: &str, limit: usize| {
        names.iter().position(|v| *v == name).filter(|i| *i < limit)
    };
    let action = |name| find(verbs.actions, name, MAX_ACTIONS).map(ActionId::new);
    Some(Binding {
        x: AxisId::new(find(verbs.axes, verb::X, MAX_AXES)?),
        y: AxisId::new(find(verbs.axes, verb::Y, MAX_AXES)?),
        primary: ActionId::new(find(verbs.actions, verb::CLICK, MAX_ACTIONS)?),
        advance_focus: ActionId::new(find(verbs.actions, verb::FOCUS, MAX_ACTIONS)?),
        // Absent is a UI that does not scroll, not a UI that does not route —
        // which is why these two are outside the `?`s above.
        scroll_up: action(verb::SCROLL_UP),
        scroll_down: action(verb::SCROLL_DOWN),
    })
}

/// One frame of a game's declared UI.
pub struct Ui {
    list: DrawList,
    router: Router,
    query: Query<&'static mut Widget>,
    /// This frame's widgets in world order, states filled in as they are drawn.
    widgets: Vec<Widget>,
    /// Indices into `widgets`, sorted into draw order. Separate so `widgets`
    /// keeps the world order the write-back pass walks.
    order: Vec<u32>,
}

impl Ui {
    /// Build the host's UI stage.
    ///
    /// # Errors
    ///
    /// If the query is refused, which a single mutable component cannot cause —
    /// it is propagated rather than unwrapped because engine crates do not
    /// unwrap (§3).
    pub fn new() -> Result<Ui, AliasError> {
        Ok(Ui {
            list: DrawList::default(),
            router: Router::default(),
            query: Query::new()?,
            widgets: Vec::new(),
            order: Vec::new(),
        })
    }

    /// Where the pointer is, in canvas units — what a host draws a cursor at.
    #[must_use]
    pub fn pointer(&self) -> (f32, f32) {
        self.router.pointer().position()
    }

    /// Run one tick of UI over `world` and return the geometry for it.
    ///
    /// `target` is the surface in physical pixels; the canvas is fitted into it
    /// with a uniform scale and centred, so a UI authored once is the same UI at
    /// every window size.
    pub fn frame(&mut self, world: &mut World, tick: &Tick, target: (u32, u32)) -> &[UiVertex] {
        let Ui {
            list,
            router,
            query,
            widgets,
            order,
        } = self;
        router.begin(tick, CANVAS);
        widgets.clear();
        order.clear();
        world.each(query, |_, w: &mut Widget| {
            order.push(widgets.len() as u32);
            widgets.push(*w);
        });
        // Unstable: in place, so a settled frame allocates nothing. The key is
        // total — `id` breaks a tie in `order` — so the picture never depends on
        // which archetype a widget happens to live in.
        order.sort_unstable_by_key(|i| {
            let w = &widgets[*i as usize];
            (w.order, w.id)
        });

        list.clear();
        let fit = crate::layout::Fit::new(target);
        list.push_transform(fit.offset, fit.scale);
        for index in order.iter() {
            let w = &mut widgets[*index as usize];
            let rect = Rect::new(w.rect[0], w.rect[1], w.rect[2], w.rect[3]);
            w.state = 0;
            match w.kind {
                widget::PANEL => list.rect(rect, w.color),
                widget::LABEL => {
                    // Clipped to its own rectangle: a HUD row that outgrew the
                    // space it was given is cut, never drawn over its neighbour.
                    list.push_clip(rect);
                    list.text(rect.x, rect.y, w.text(), w.text_color);
                    list.pop_clip();
                }
                widget::BUTTON => {
                    let response = router.hit(WidgetId::from_hash(w.id), rect);
                    w.state = flags(&response);
                    let fill = if response.hovered || response.held {
                        lighten(w.color)
                    } else {
                        w.color
                    };
                    match response.focused {
                        true => {
                            list.rect(rect, FOCUS);
                            list.rect(rect.inset(BORDER), fill);
                        }
                        false => list.rect(rect, fill),
                    }
                    let text = w.text();
                    // Centred on whole canvas units: the bitmap font is sampled
                    // nearest, and a half-unit offset under an integer scale
                    // drops a column out of every stem.
                    let x = (rect.x + (rect.w - DrawList::width(text)) * 0.5).floor();
                    let y = (rect.y + (rect.h - f32::from(font::CELL.1 as u16)) * 0.5).floor();
                    list.push_clip(rect);
                    list.text(x, y, text, w.text_color);
                    list.pop_clip();
                }
                // An unknown kind draws nothing. The game may have been built
                // against a newer boundary than this host (§4.2.2).
                _ => {}
            }
        }
        // Only over a UI that exists: every demo before this milestone drives a
        // camera with the same mouse, and a cursor floating over one would be
        // this crate deciding what an unrelated game looks like.
        if !widgets.is_empty() {
            cursor(list, router.pointer().position());
        }
        list.pop_transform();

        // Second pass, same query, same order — see the module docs.
        let mut at = 0;
        world.each(query, |_, w: &mut Widget| {
            if let Some(drawn) = widgets.get(at) {
                w.state = drawn.state;
            }
            at += 1;
        });
        list.vertices()
    }

    /// The geometry the last [`frame`](Self::frame) built.
    ///
    /// A host runs the UI on the *tick* — that is what puts the hit state in
    /// the world before the canonical hash reads it — and submits on the frame,
    /// which may be a different number of them.
    #[must_use]
    pub fn vertices(&self) -> &[UiVertex] {
        self.list.vertices()
    }
}

/// An arrow with its tip at `at`, in canvas units.
///
/// Rows rather than a triangle because the draw list has one primitive; the
/// first pass is the same shape dilated by a unit, which is the halo.
fn cursor(list: &mut DrawList, at: (f32, f32)) {
    for (color, grow) in [(CURSOR_HALO, 1.0), (CURSOR, 0.0)] {
        for row in 0..CURSOR_HEIGHT {
            let i = f32::from(row as u16);
            let rect = Rect::new(at.0, at.1 + i, i + 1.0, 1.0);
            list.rect(rect.inset(-grow), color);
        }
    }
}

/// A [`Response`] as the bits a game reads off [`Widget::state`].
fn flags(response: &Response) -> u32 {
    let bit = |set: bool, flag: u32| if set { flag } else { 0 };
    bit(response.hovered, state::HOVERED)
        | bit(response.held, state::HELD)
        | bit(response.clicked, state::CLICKED)
        | bit(response.focused, state::FOCUSED)
}

/// Each channel a third of the way to white, alpha untouched — the host's one
/// visual opinion (see [`Widget::color`]).
fn lighten(color: u32) -> u32 {
    let step = |shift: u32| {
        let channel = (color >> shift) & 0xff;
        (channel + (0xff - channel) / 3) << shift
    };
    (color & 0xff00_0000) | step(16) | step(8) | step(0)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::router::AXIS_SCALE;
    use gg_ecs::boundary::widget_id;

    const OK: u64 = widget_id("ok");
    const CANCEL: u64 = widget_id("cancel");
    const TARGET: (u32, u32) = (1280, 720);

    fn moved(dx: f32, dy: f32) -> Tick {
        Tick {
            motion: (
                (dx * AXIS_SCALE as f32) as i32,
                (dy * AXIS_SCALE as f32) as i32,
            ),
            ..Tick::default()
        }
    }

    /// A panel and two buttons, spawned so the button that draws *last* is
    /// spawned first — the sort has something to do, and world order alone
    /// would get the answer wrong.
    fn world() -> World {
        let mut world = World::new();
        world.register::<Widget>().expect("Widget registers");
        for (id, x, text, order) in [(CANCEL, 110.0, "cancel", 2), (OK, 10.0, "ok", 1)] {
            let entity = world.spawn();
            let mut w = Widget::button(id, [x, 10.0, 80.0, 40.0], 0xff20_3040, 0xffff_ffff, text);
            w.order = order;
            world.insert(entity, w).expect("insert");
        }
        let panel = world.spawn();
        let mut backdrop = Widget::panel([0.0, 0.0, 200.0, 60.0], 0xc00c_1016);
        backdrop.order = 0;
        world.insert(panel, backdrop).expect("insert");
        world
    }

    fn state_of(world: &World, id: u64) -> u32 {
        let query = Query::<&Widget>::new().expect("read query");
        let mut found = 0;
        world.each_ref(&query, |_, w: &Widget| {
            if w.id == id {
                found = w.state;
            }
        });
        found
    }

    /// The whole contract in one frame: geometry comes out, and the hit state
    /// goes back into the world where the game reads it next tick.
    #[test]
    fn a_declared_ui_draws_and_its_hit_state_lands_back_in_the_world() {
        let mut ui = Ui::new().expect("the query is a single mutable component");
        let mut world = world();
        // Frame one declares; the router resolves against the previous frame's
        // geometry, so nothing can be hovered yet (§4.9's one frame of lag).
        ui.frame(&mut world, &moved(50.0, 30.0), TARGET);
        assert_eq!(state_of(&world, OK), 0, "nothing was declared last frame");

        assert!(!ui.frame(&mut world, &Tick::default(), TARGET).is_empty());
        assert_eq!(state_of(&world, OK), state::HOVERED);
        assert_eq!(state_of(&world, CANCEL), 0);
    }

    /// The sort is what decides the picture, not the world. `cancel` is spawned
    /// first and drawn last, so where the two overlap it takes the hit.
    #[test]
    fn draw_order_comes_from_the_component_and_not_from_world_order() {
        let mut ui = Ui::new().expect("query");
        let mut world = World::new();
        world.register::<Widget>().expect("register");
        for (id, order) in [(CANCEL, 2), (OK, 1)] {
            let entity = world.spawn();
            let mut w = Widget::button(id, [10.0, 10.0, 80.0, 40.0], 0xff20_3040, 0, "x");
            w.order = order;
            world.insert(entity, w).expect("insert");
        }
        ui.frame(&mut world, &moved(50.0, 30.0), TARGET);
        ui.frame(&mut world, &Tick::default(), TARGET);
        assert_eq!(state_of(&world, CANCEL), state::HOVERED, "the last drawn");
        assert_eq!(state_of(&world, OK), 0);
    }

    /// §6 M13's criterion, at the level this crate can prove it: the same tick
    /// stream lands on the same widget, whatever the window is. The pointer is
    /// integrated in canvas units precisely so this holds.
    #[test]
    fn a_replayed_click_lands_on_the_same_widget_at_any_window_size() {
        let click = |target: (u32, u32)| {
            let mut ui = Ui::new().expect("query");
            let mut world = world();
            let ticks = [
                moved(140.0, 30.0),
                Tick::default(),
                Tick {
                    primary: true,
                    ..Tick::default()
                },
                Tick::default(),
            ];
            let mut clicked = Vec::new();
            for tick in &ticks {
                ui.frame(&mut world, tick, target);
                for id in [OK, CANCEL] {
                    if state_of(&world, id) & state::CLICKED != 0 {
                        clicked.push(id);
                    }
                }
            }
            clicked
        };
        assert_eq!(click(TARGET), vec![CANCEL], "pressed and released over it");
        assert_eq!(click((640, 360)), click(TARGET), "and at canvas scale");
        assert_eq!(click((3840, 2160)), click(TARGET), "and at 4K");
    }

    /// The cursor alone: two passes of `CURSOR_HEIGHT` rows, six vertices a
    /// quad. What "drew nothing" means once there is a host-drawn pointer.
    const CURSOR_ONLY: usize = 2 * CURSOR_HEIGHT as usize * 6;

    /// A widget whose kind this host does not know draws nothing rather than
    /// drawing something wrong — the game may be built against a newer
    /// boundary than the host that loaded it (§4.2.2).
    #[test]
    fn an_unknown_kind_draws_nothing() {
        let mut ui = Ui::new().expect("query");
        let mut world = World::new();
        world.register::<Widget>().expect("register");
        let entity = world.spawn();
        let mut w = Widget::panel([0.0, 0.0, 100.0, 100.0], 0xffff_ffff);
        w.kind = 9999;
        world.insert(entity, w).expect("insert");
        let drawn = ui.frame(&mut world, &Tick::default(), TARGET).len();
        assert_eq!(drawn, CURSOR_ONLY, "the cursor, and nothing of the widget");
    }

    /// A game with no UI at all gets no pointer drawn over its camera.
    #[test]
    fn an_empty_world_draws_no_cursor() {
        let mut ui = Ui::new().expect("query");
        let mut world = World::new();
        world.register::<Widget>().expect("register");
        assert!(ui.frame(&mut world, &Tick::default(), TARGET).is_empty());
    }

    /// The four names are one protocol, and a build declaring three of them
    /// gets none — see [`binding`].
    #[test]
    fn a_binding_needs_all_four_verbs() {
        let full = Verbs {
            actions: &["jump", verb::CLICK, verb::FOCUS],
            axes: &[verb::X, verb::Y],
        };
        let bound = binding(&full).expect("all four are declared");
        assert_eq!(bound.primary.index(), 1);
        assert_eq!(bound.advance_focus.index(), 2);
        assert_eq!((bound.x.index(), bound.y.index()), (0, 1));
        assert!(
            binding(&Verbs {
                axes: &[verb::X],
                ..full
            })
            .is_none()
        );
        assert!(
            binding(&Verbs {
                actions: &[verb::CLICK],
                ..full
            })
            .is_none()
        );
    }

    /// The buffers settle: a UI that reallocated per frame would fail §6 M13's
    /// allocation criterion in the one path a game actually uses.
    #[test]
    fn a_settled_frame_reuses_its_buffers() {
        let mut ui = Ui::new().expect("query");
        let mut world = world();
        for _ in 0..8 {
            ui.frame(&mut world, &moved(1.0, 1.0), TARGET);
        }
        let sizes = (ui.widgets.capacity(), ui.order.capacity());
        for _ in 0..64 {
            ui.frame(&mut world, &moved(1.0, 1.0), TARGET);
        }
        assert_eq!((ui.widgets.capacity(), ui.order.capacity()), sizes);
    }

    #[test]
    fn a_hover_lightens_toward_white_and_leaves_alpha_alone() {
        assert_eq!(lighten(0xff00_0000), 0xff55_5555);
        assert_eq!(lighten(0x80ff_ffff), 0x80ff_ffff, "white cannot lighten");
        assert_eq!(lighten(0x0020_4060) & 0xff00_0000, 0x0000_0000);
    }
}
