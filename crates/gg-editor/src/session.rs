//! The scripted editor session, as input frames (§6 M15's fourth exit row).
//!
//! Authored here rather than in `xtask` for demo 07's reason: the clicks are
//! against *this* crate's layout constants, so a panel that moves moves the
//! script with it instead of silently missing. What `xtask` supplies is the
//! verb ids, because those belong to whichever game the editor was opened over
//! (§4.7) — the same four names resolve to different indices in different verb
//! lists, and hard-coding them here would bind the session to one demo.
//!
//! Every coordinate below comes off [`crate`]'s layout rectangles by
//! arithmetic. None is a literal, which is the whole point: an editor whose
//! recorded session had to be re-authored by hand after a panel moved would
//! stop being re-recorded, and the gate would rot into a hash comparison of two
//! runs that both click on nothing.

use crate::{BAR, DOCK, EM, INSPECT, PITCH, TREE};
use gg_input::{AXIS_SCALE, MAX_AXES};
use gg_input::{ActionId, AxisId, InputFrame};

/// One step of a session.
#[derive(Clone, Copy, Debug)]
pub enum Act {
    /// Glide the pointer to a canvas position.
    To((f32, f32)),
    /// Hold still for `n` ticks — a press must find its widget already hovered,
    /// because hover resolves against the previous frame (§4.9's one-frame lag).
    Settle(u32),
    /// Press and release over whatever is under the pointer.
    Click,
}

/// Where the session aims. Every one is the centre of a rectangle `panels`
/// declares, computed the same way twice.
pub mod aim {
    use super::{BAR, DOCK, EM, INSPECT, PITCH, TREE};

    fn centre(x: f32, y: f32, w: f32, h: f32) -> (f32, f32) {
        (x + w * 0.5, y + h * 0.5)
    }

    /// The toolbar's `i`-th button, in declaration order: play, step, save.
    fn toolbar(offset: f32, width: f32) -> (f32, f32) {
        centre(BAR.x + 3.0 + offset, BAR.y + 2.0, width, 9.0)
    }

    /// Play/pause.
    #[must_use]
    pub fn play() -> (f32, f32) {
        toolbar(0.0, 34.0)
    }

    /// Advance one tick.
    #[must_use]
    pub fn step() -> (f32, f32) {
        toolbar(37.0, 28.0)
    }

    /// Write the save.
    #[must_use]
    pub fn save() -> (f32, f32) {
        toolbar(68.0, 28.0)
    }

    /// The `i`-th entity row of the current tree page.
    #[must_use]
    pub fn tree_row(i: usize) -> (f32, f32) {
        centre(
            TREE.x + 2.0,
            TREE.y + 13.0 + i as f32 * PITCH,
            TREE.w - 4.0,
            8.0,
        )
    }

    /// The `lane`-th cell of the `row`-th line of the inspector body, where row
    /// 0 is the first component's title and each field that follows takes one.
    #[must_use]
    pub fn lane(row: usize, lane: usize) -> (f32, f32) {
        centre(
            INSPECT.x + 3.0 + 8.0 * EM + lane as f32 * 36.0,
            INSPECT.y + 14.0 + row as f32 * PITCH,
            35.0,
            8.0,
        )
    }

    fn bar(offset: f32, width: f32) -> (f32, f32) {
        centre(
            INSPECT.x + 3.0 + offset,
            INSPECT.bottom() - 11.0,
            width,
            8.0,
        )
    }

    /// Cycle the nudge step.
    #[must_use]
    pub fn grain() -> (f32, f32) {
        bar(0.0, 32.0)
    }

    /// Nudge the selected lane down.
    #[must_use]
    pub fn minus() -> (f32, f32) {
        bar(35.0, 12.0)
    }

    /// Nudge it up.
    #[must_use]
    pub fn plus() -> (f32, f32) {
        bar(49.0, 12.0)
    }

    /// The dock's `i`-th tab: cvars, assets, perf.
    #[must_use]
    pub fn tab(i: usize) -> (f32, f32) {
        centre(DOCK.x + 3.0 + i as f32 * 40.0, DOCK.y + 2.0, 38.0, 9.0)
    }
}

/// The session §6 M15's gate replays: pause a running game, select an entity,
/// nudge one field of it six times across two step sizes, single-step twice,
/// play and pause again, walk the dock's tabs, and save.
///
/// It starts by *pausing*, which is not an accident of taste: the editor opens
/// on a running game because a paused one never runs its bootstrap system and
/// an editor over an empty world would inspect nothing.
#[must_use]
pub fn script() -> Vec<Act> {
    let mut acts = vec![Act::To(aim::play()), Act::Settle(3), Act::Click];
    // Tree rows are archetype order, not spawn order (see `crate::scan`), so
    // which entity row 1 holds is a property of the world and not of the
    // script. What the script needs is only that it holds *something* — the
    // caller's gate is what names the component the inspector then shows.
    acts.extend([Act::To(aim::tree_row(1)), Act::Settle(3), Act::Click]);
    // Row 0 of the inspector body is a component title; row 1 is its first
    // field, and lane 0 of that is the field's first scalar.
    acts.extend([Act::To(aim::lane(1, 0)), Act::Settle(3), Act::Click]);
    acts.push(Act::To(aim::plus()));
    acts.push(Act::Settle(3));
    acts.extend([Act::Click, Act::Click, Act::Click, Act::Click]);
    // A coarser step, one up and one down: the pair nets to zero, so what it
    // proves is that the step button moved the grain and not the value.
    acts.extend([Act::To(aim::grain()), Act::Settle(3), Act::Click]);
    acts.extend([Act::To(aim::plus()), Act::Settle(3), Act::Click]);
    acts.extend([Act::To(aim::minus()), Act::Settle(3), Act::Click]);
    // Two single ticks, then run for a while, then stop again.
    acts.extend([Act::To(aim::step()), Act::Settle(3), Act::Click, Act::Click]);
    acts.extend([Act::To(aim::play()), Act::Settle(3), Act::Click]);
    acts.push(Act::Settle(40));
    acts.push(Act::Click);
    // Every dock tab, so the golden subject and the replay cover all three.
    for i in 0..3 {
        acts.extend([Act::To(aim::tab(i)), Act::Settle(3), Act::Click]);
    }
    acts.extend([Act::To(aim::save()), Act::Settle(3), Act::Click]);
    // A tail, or the session proves nothing about an editor that settled.
    acts.push(Act::Settle(20));
    acts
}

/// Ticks a press is held, and the ones after the release.
const PRESS: u32 = 2;
const RELEASE: u32 = 6;
/// Canvas units the pointer covers per tick while gliding. Slow enough that a
/// press never lands on a widget the pointer only just entered.
const GLIDE: f32 = 8.0;

/// Turn a script into recordable frames, against the verb ids the host resolved
/// [`gg_ui::boundary::verb`]'s four names to.
#[must_use]
pub fn frames(acts: &[Act], click: ActionId, x: AxisId, y: AxisId) -> Vec<InputFrame> {
    let mut out = Vec::new();
    let mut at = (0i32, 0i32);
    let hold = |out: &mut Vec<InputFrame>, ticks: u32, down: bool| {
        for _ in 0..ticks {
            out.push(InputFrame {
                buttons: u64::from(down) << click.index(),
                axes: [0; MAX_AXES],
            });
        }
    };
    for act in acts {
        match act {
            Act::Settle(ticks) => hold(&mut out, *ticks, false),
            Act::Click => {
                hold(&mut out, PRESS, true);
                hold(&mut out, RELEASE, false);
            }
            Act::To(to) => {
                let target = (
                    (to.0 * AXIS_SCALE as f32) as i32,
                    (to.1 * AXIS_SCALE as f32) as i32,
                );
                // Chebyshev, so no square root reaches a path that authors
                // hashed input — and the pointer arrives on both axes at once.
                let reach = (target.0 - at.0).abs().max((target.1 - at.1).abs());
                let ticks = ((reach as f32 / (GLIDE * AXIS_SCALE as f32)) as u32).max(4);
                for step in 0..ticks {
                    // Divided by the ticks *remaining*, so the last frame
                    // carries the remainder and the pointer lands exactly on
                    // the target — a click one unit short names nothing.
                    let left = (ticks - step) as i32;
                    let motion = ((target.0 - at.0) / left, (target.1 - at.1) / left);
                    at = (at.0 + motion.0, at.1 + motion.1);
                    let mut axes = [0; MAX_AXES];
                    axes[x.index()] = motion.0;
                    axes[y.index()] = motion.1;
                    out.push(InputFrame { buttons: 0, axes });
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The script lands where it aims. Integer fixed point accumulates exactly,
    /// so "exactly" is the assertion and not "within a unit".
    #[test]
    fn the_pointer_arrives_on_every_target() {
        let (click, x, y) = (ActionId::new(1), AxisId::new(5), AxisId::new(6));
        let acts = script();
        let frames = frames(&acts, click, x, y);
        let mut at = (0i32, 0i32);
        let mut fed = 0;
        for act in &acts {
            match act {
                Act::Settle(n) => fed += *n as usize,
                Act::Click => fed += (PRESS + RELEASE) as usize,
                Act::To(to) => {
                    let before = fed;
                    while fed < frames.len() && frames[fed].buttons == 0 && {
                        let f = &frames[fed];
                        f.axes[x.index()] != 0 || f.axes[y.index()] != 0 || fed == before
                    } {
                        at.0 += frames[fed].axes[x.index()];
                        at.1 += frames[fed].axes[y.index()];
                        fed += 1;
                        if at
                            == (
                                (to.0 * AXIS_SCALE as f32) as i32,
                                (to.1 * AXIS_SCALE as f32) as i32,
                            )
                        {
                            break;
                        }
                    }
                    assert_eq!(
                        at,
                        (
                            (to.0 * AXIS_SCALE as f32) as i32,
                            (to.1 * AXIS_SCALE as f32) as i32
                        ),
                        "glide missed {to:?}"
                    );
                }
            }
        }
        assert_eq!(fed, frames.len(), "every frame belongs to an act");
    }

    /// Every aim is inside the panel that owns it — a target that drifted out
    /// of its rectangle would click on the panel behind it and still replay.
    #[test]
    fn every_target_is_inside_its_panel() {
        let inside = |rect: gg_ui::draw::Rect, at: (f32, f32), what: &str| {
            assert!(
                rect.contains(at.0, at.1),
                "{what} at {at:?} is outside {rect:?}"
            );
        };
        for (at, what) in [
            (aim::play(), "play"),
            (aim::step(), "step"),
            (aim::save(), "save"),
        ] {
            inside(BAR, at, what);
        }
        inside(TREE, aim::tree_row(0), "tree row 0");
        inside(TREE, aim::tree_row(crate::PAGE - 1), "tree row last");
        for row in 0..6 {
            inside(INSPECT, aim::lane(row, 0), "inspector lane");
            inside(INSPECT, aim::lane(row, 2), "inspector lane 2");
        }
        for (at, what) in [
            (aim::grain(), "grain"),
            (aim::minus(), "minus"),
            (aim::plus(), "plus"),
        ] {
            inside(INSPECT, at, what);
        }
        for i in 0..3 {
            inside(DOCK, aim::tab(i), "dock tab");
        }
    }

    /// The click count is the gate's contract: three play/pause presses, one
    /// tree row, one lane, six nudges, one grain, two single steps, three tabs
    /// and one save. Pinned because the gate counts what these produce.
    #[test]
    fn the_script_clicks_what_the_gate_expects() {
        let clicks = script().iter().filter(|a| matches!(a, Act::Click)).count();
        assert_eq!(clicks, 3 + 1 + 1 + 6 + 1 + 2 + 3 + 1);
    }
}
