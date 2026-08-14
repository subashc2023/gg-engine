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
use crate::layout::Fit;
use crate::router::{Binding, Response, Router, Tick};
use crate::{WidgetId, font};
use gg_ecs::boundary::{CANVAS, Prefs, Verbs, Widget, state, widget};
use gg_ecs::{AliasError, Query, World};
use gg_input::{ActionId, AxisId, MAX_ACTIONS, MAX_AXES};
use gg_render::ui::UiVertex;

/// Accent for the focused widget's border, and the border's width in canvas
/// units. Not the game's to choose: focus is a host concept — the router owns
/// it — so the ring that shows it is host-drawn too.
const FOCUS: u32 = 0xff7f_d0a0;
const BORDER: f32 = 1.0;

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
    /// The player's preference for which arrow they see (§6 M19) — read off
    /// the world's [`Prefs`], the first walked, so the game's settings menu
    /// reaches this stage without the host relaying a flag.
    prefs: Query<&'static Prefs>,
    /// The [`Prefs`] the last frame walked — see [`Ui::prefs`].
    asked: Prefs,
    /// Whether the last frame drew the software arrow — see [`Ui::cursor_drawn`].
    arrow: bool,
    /// Whether the last frame had a hit-tested widget with area — see
    /// [`Ui::wants_pointer`]. Separate from `arrow`, which is this *and* the
    /// player's arrow preference: which arrow is drawn and whether the UI is
    /// pointed at are different questions, and only the second decides who gets
    /// the mouse.
    pointed: bool,
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
            prefs: Query::new()?,
            asked: Prefs::default(),
            arrow: false,
            pointed: false,
        })
    }

    /// Where the pointer is, in canvas units — what a host draws a cursor at.
    #[must_use]
    pub fn pointer(&self) -> (f32, f32) {
        self.router.pointer().position()
    }

    /// Whether the last [`frame`](Self::frame) drew the software arrow: the UI
    /// had something to point *at* — a hit-tested widget with area — and the
    /// world's [`Prefs`] did not ask for the hardware cursor. What a host asks
    /// to decide the OS arrow's fate: hidden while this stage draws its own,
    /// shown otherwise — over a plain HUD, and under the hardware preference,
    /// where the routing is identical and only the picture changes.
    #[must_use]
    pub fn cursor_drawn(&self) -> bool {
        self.arrow
    }

    /// Whether the last [`frame`](Self::frame) had anything to point at: a
    /// hit-tested widget with area. What a host asks to decide who holds the
    /// mouse — a game that binds the pointer verbs holds it while this is false
    /// (mouse-look, nothing to click) and gives it back while it is true (a menu
    /// is up). Unlike [`cursor_drawn`](Self::cursor_drawn) this ignores the
    /// arrow preference: a player who asked for the OS arrow still needs the
    /// pointer freed to reach the button, and only the picture differs.
    ///
    /// Derived from the world's widgets and therefore from hashed state, which
    /// is what makes it replay-safe: a recorded session opens its menu on the
    /// same tick with no window anywhere, and the pointer changes hands there
    /// too (§4.7).
    #[must_use]
    pub fn wants_pointer(&self) -> bool {
        self.pointed
    }

    /// The [`Prefs`] the last [`frame`](Self::frame) walked — the first in the
    /// world, that type's documented rule, or the defaults where there is none.
    ///
    /// Handed on rather than re-queried by every consumer: this stage reads them
    /// already, and a second walk in the shell would be a second answer to "the
    /// first `Prefs`" whenever a game declares two.
    #[must_use]
    pub fn prefs(&self) -> Prefs {
        self.asked
    }

    /// Run one tick of UI over `world` and return the geometry for it.
    ///
    /// `fit` is where the canvas sits on the surface — the whole of it in a
    /// plain run ([`Fit::new`]), the editor's game pane under one
    /// ([`Fit::inside`]). It moves the *picture* and never the hit test: the
    /// pointer is integrated in canvas units and the widgets are hit in canvas
    /// units, so a recorded session replays identically wherever the picture
    /// lands.
    pub fn frame(&mut self, world: &mut World, tick: &Tick, fit: Fit) -> &[UiVertex] {
        let Ui {
            list,
            router,
            query,
            widgets,
            order,
            prefs,
            asked,
            arrow,
            pointed,
        } = self;
        // The player's arrow preference, off the world itself (§6 M19): the
        // menu click that set it is ordinary recorded input, and no host flag
        // needs relaying.
        let mut found = None;
        world.each_ref(prefs, |_, p: &Prefs| {
            found.get_or_insert(*p);
        });
        // A world with no `Prefs` at all gets the defaults, which is the same
        // answer as a world whose `Prefs` is zeroed — the zero law, relied on
        // rather than restated.
        *asked = found.unwrap_or_default();
        let hardware = asked.hardware_cursor();
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
        list.push_transform(fit.offset, fit.scale);
        let mut hit_tested = false;
        for index in order.iter() {
            let w = &mut widgets[*index as usize];
            let rect = Rect::new(w.rect[0], w.rect[1], w.rect[2], w.rect[3]);
            w.state = 0;
            match w.kind {
                widget::PANEL => list.rect(rect, w.color),
                widget::LABEL | widget::LABEL_CENTRE | widget::LABEL_RIGHT => {
                    let text = w.text();
                    // Whole canvas units, for the reason the button arm gives.
                    // Slack goes negative when the text outgrew its rect, which
                    // moves a centred or right-aligned run left of `rect.x` and
                    // loses its head rather than its tail — the right end being
                    // the one the alignment was asked for.
                    let slack = rect.w - DrawList::width(text);
                    let x = match w.kind {
                        widget::LABEL_CENTRE => (rect.x + slack * 0.5).floor(),
                        widget::LABEL_RIGHT => (rect.x + slack).floor(),
                        _ => rect.x,
                    };
                    // Clipped to its own rectangle: a HUD row that outgrew the
                    // space it was given is cut, never drawn over its neighbour.
                    list.push_clip(rect);
                    list.text(x, rect.y, text, w.text_color);
                    list.pop_clip();
                }
                widget::BUTTON => {
                    // Area is part of the rule: a zero rect is how a game hides
                    // a widget (§4.9), and a hidden menu's buttons must not
                    // summon the arrow back over a board nothing points at.
                    hit_tested |= rect.w > 0.0 && rect.h > 0.0;
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
        // Only over a UI something can point *at*: a cursor is the pointer's
        // report of where it is, and a declaration with no hit-tested widget in
        // it has no pointer — either the build resolved no binding (§4.9's
        // all-or-nothing `binding`) or the game's UI is a HUD. Demo 10's board
        // is the case that found this: 262 widgets, none of them a button, and a
        // white arrow parked in the corner of every frame because "a UI exists"
        // was standing in for "a UI is pointed at".
        *pointed = hit_tested;
        *arrow = hit_tested && !hardware;
        if *arrow {
            crate::draw::cursor(list, router.pointer().position());
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
        ui.frame(&mut world, &moved(50.0, 30.0), Fit::new(TARGET));
        assert_eq!(state_of(&world, OK), 0, "nothing was declared last frame");

        assert!(
            !ui.frame(&mut world, &Tick::default(), Fit::new(TARGET))
                .is_empty()
        );
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
        ui.frame(&mut world, &moved(50.0, 30.0), Fit::new(TARGET));
        ui.frame(&mut world, &Tick::default(), Fit::new(TARGET));
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
                ui.frame(&mut world, tick, Fit::new(target));
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

    /// The cursor alone: two passes of the arrow's runs, six vertices a quad.
    /// What "drew nothing but the pointer" means.
    const CURSOR_ONLY: usize = 2 * crate::draw::ARROW.len() * 6;

    /// A widget whose kind this host does not know draws nothing rather than
    /// drawing something wrong — the game may be built against a newer
    /// boundary than the host that loaded it (§4.2.2). And an unknown kind is
    /// not hit-tested, so it does not bring a cursor with it either.
    #[test]
    fn an_unknown_kind_draws_nothing() {
        let mut ui = Ui::new().expect("query");
        let mut world = World::new();
        world.register::<Widget>().expect("register");
        let entity = world.spawn();
        let mut w = Widget::panel([0.0, 0.0, 100.0, 100.0], 0xffff_ffff);
        w.kind = 9999;
        world.insert(entity, w).expect("insert");
        assert!(
            ui.frame(&mut world, &Tick::default(), Fit::new(TARGET))
                .is_empty()
        );
    }

    /// The alignment the host has done for a button since M13, now askable for
    /// a label (§6 M44) — measured off the pen rather than off the kind,
    /// because the claim is that the game's own `text_width` and the host's
    /// placement are one arithmetic and not two that agree today.
    #[test]
    fn an_aligned_label_lands_where_the_games_own_measurement_says() {
        const RECT: [f32; 4] = [10.0, 20.0, 100.0, 9.0];
        const TEXT: &str = "SCORE";
        // Fit 1:1 so the pen is in canvas units and the arithmetic is visible.
        let pen = |build: fn([f32; 4], u32, &str) -> Widget| {
            let mut ui = Ui::new().expect("query");
            let mut world = World::new();
            world.register::<Widget>().expect("register");
            let label = world.spawn();
            world
                .insert(label, build(RECT, 0xffff_ffff, TEXT))
                .expect("insert");
            ui.frame(&mut world, &Tick::default(), Fit::new(CANVAS))
                .iter()
                .map(|v| v.pos[0])
                .fold(f32::INFINITY, f32::min)
        };
        let slack = RECT[2] - gg_ecs::boundary::text_width(TEXT);
        assert_eq!(pen(Widget::label), RECT[0]);
        assert_eq!(pen(Widget::label_centred), RECT[0] + slack * 0.5);
        assert_eq!(pen(Widget::label_right), RECT[0] + slack);
    }

    /// A game with no UI at all gets no pointer drawn over its camera.
    #[test]
    fn an_empty_world_draws_no_cursor() {
        let mut ui = Ui::new().expect("query");
        let mut world = World::new();
        world.register::<Widget>().expect("register");
        assert!(
            ui.frame(&mut world, &Tick::default(), Fit::new(TARGET))
                .is_empty()
        );
    }

    /// The cursor follows what can be *pointed at*, not what is declared. A HUD
    /// of panels and labels is read, never clicked — and a build that resolved
    /// no pointer binding at all would otherwise park an arrow in the corner of
    /// every frame, which is what demo 10's board found (§6 M18).
    #[test]
    fn a_cursor_is_drawn_only_over_a_ui_with_something_to_hit() {
        let mut ui = Ui::new().expect("query");
        let mut world = World::new();
        world.register::<Widget>().expect("register");
        let hud = world.spawn();
        world
            .insert(
                hud,
                Widget::label([4.0, 4.0, 60.0, 10.0], 0xffff_ffff, "hp"),
            )
            .expect("insert");
        let text = ui
            .frame(&mut world, &Tick::default(), Fit::new(TARGET))
            .len();
        assert!(text > 0, "the label drew nothing");

        let button = world.spawn();
        world
            .insert(
                button,
                Widget::button(WidgetId::new("ok").get(), [0.0; 4], 0, 0, ""),
            )
            .expect("insert");
        let hidden = ui
            .frame(&mut world, &Tick::default(), Fit::new(TARGET))
            .len();
        assert_eq!(
            hidden, text,
            "a zero rect is how a game hides a button (§4.9), and a hidden menu \
             must not summon the arrow"
        );
        assert!(!ui.cursor_drawn());

        world
            .insert(
                button,
                Widget::button(
                    WidgetId::new("ok").get(),
                    [10.0, 10.0, 40.0, 12.0],
                    0,
                    0,
                    "",
                ),
            )
            .expect("insert");
        let with_button = ui
            .frame(&mut world, &Tick::default(), Fit::new(TARGET))
            .len();
        assert_eq!(
            with_button - text,
            CURSOR_ONLY + 6, // the button's own quad, then the pointer it brought
            "one hit-tested widget is what brings the pointer"
        );
        assert!(ui.cursor_drawn());
    }

    /// The other question asked of the same frame, and the reason it is a
    /// second accessor: a player who asked for the OS arrow still has to be
    /// able to reach the button, so the hand-back cannot ride on `cursor_drawn`
    /// (§6 M21). Two answers that must part exactly here.
    #[test]
    fn the_hardware_arrow_changes_the_picture_and_not_who_holds_the_mouse() {
        let mut ui = Ui::new().expect("query");
        let mut world = World::new();
        world.register::<Widget>().expect("register");
        world.register::<Prefs>().expect("register");
        let button = world.spawn();
        world
            .insert(
                button,
                Widget::button(
                    WidgetId::new("ok").get(),
                    [10.0, 10.0, 40.0, 12.0],
                    0,
                    0,
                    "",
                ),
            )
            .expect("insert");
        ui.frame(&mut world, &Tick::default(), Fit::new(TARGET));
        assert!(ui.wants_pointer(), "a button with area is pointed at");
        assert!(ui.cursor_drawn(), "and the software arrow draws over it");

        let prefs = world.spawn();
        world
            .insert(
                prefs,
                Prefs {
                    cursor: gg_ecs::boundary::cursor::HARDWARE,
                    ..Default::default()
                },
            )
            .expect("insert");
        ui.frame(&mut world, &Tick::default(), Fit::new(TARGET));
        assert!(!ui.cursor_drawn(), "the OS arrow stands in");
        assert!(
            ui.wants_pointer(),
            "and the button is still there to be reached"
        );
        assert!(ui.prefs().hardware_cursor(), "read once, handed on");

        // And a HUD is neither: nothing to point at, so the game keeps the mouse.
        world.despawn(button);
        ui.frame(&mut world, &Tick::default(), Fit::new(TARGET));
        assert!(!ui.wants_pointer());
    }

    /// The hardware-cursor preference (`Prefs`, §6 M19): the pointer still
    /// routes — a hover still lands in the world — but no second arrow is
    /// drawn over the OS one, and `cursor_drawn` tells the host to leave the
    /// OS arrow showing.
    #[test]
    fn the_hardware_preference_keeps_the_routing_and_drops_the_arrow() {
        use gg_ecs::boundary::cursor;

        let mut ui = Ui::new().expect("query");
        let mut world = world();
        ui.frame(&mut world, &moved(50.0, 30.0), Fit::new(TARGET));
        let drawn = ui
            .frame(&mut world, &Tick::default(), Fit::new(TARGET))
            .len();
        assert!(ui.cursor_drawn());

        world.register::<Prefs>().expect("register");
        let prefs = world.spawn();
        world
            .insert(
                prefs,
                Prefs {
                    cursor: cursor::HARDWARE,
                    quiet: 0,
                    aa: 0,
                    close: 0,
                },
            )
            .expect("insert");
        let bare = ui
            .frame(&mut world, &Tick::default(), Fit::new(TARGET))
            .len();
        assert_eq!(drawn - bare, CURSOR_ONLY, "only the arrow went away");
        assert!(!ui.cursor_drawn(), "the OS arrow stands in");
        assert_eq!(
            state_of(&world, OK),
            state::HOVERED,
            "the routing is intact"
        );
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
            ui.frame(&mut world, &moved(1.0, 1.0), Fit::new(TARGET));
        }
        let sizes = (ui.widgets.capacity(), ui.order.capacity());
        for _ in 0..64 {
            ui.frame(&mut world, &moved(1.0, 1.0), Fit::new(TARGET));
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
