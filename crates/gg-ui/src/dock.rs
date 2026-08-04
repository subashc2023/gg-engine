//! Docked layout (§4.9's owned list, §6 M15.1) — a tree of splits and tab groups
//! that tiles a rectangle, and the four edits a pointer can make to it.
//!
//! [`layout`](crate::layout) answers where cells go inside a panel. This answers
//! where the panels themselves go, which is a different question because the
//! answer is *state*: a seam the operator dragged has to still be there next
//! frame. So this is the one retained thing in an immediate-mode UI, and it is
//! retained for exactly the reason immediate mode usually is not — the layout is
//! the operator's, not the frame's.
//!
//! # Why it stays deterministic
//!
//! Every edit here is driven by the pointer, and the pointer is an integral of
//! the recorded axis stream ([`crate::router`]). A seam therefore lands where it
//! lands by construction on a replay, with one condition the caller owns: the
//! `area` handed to [`Dock::resolve`] must be the same, because a fraction of a
//! different rectangle is a different pixel. That is why a host records the
//! extent it laid the editor out in (§6 M15.1).
//!
//! # No floating panes
//!
//! Tearing a pane into its own OS window is the one docking gesture missing, and
//! deliberately: a second window is a second surface and a second swapchain, and
//! §1.5 makes every window on this project a manual act. Dropping on a *root*
//! edge is the same rearrangement without the second window.

use crate::draw::Rect;
use crate::layout::Axis;

/// Which pane a leaf holds — an index into whatever list the application keeps.
/// The dock never looks inside one; it moves them around.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PaneId(pub u16);

/// The narrowest a pane may be squeezed to, in the units the dock resolves in.
/// A seam stops here rather than at zero, because a pane dragged to nothing
/// cannot be dragged back — there is no edge left to grab.
pub const MIN: f32 = 56.0;

/// Thickness of the seam between two children. Wide enough to hit without
/// aiming, which is a real constraint when the pointer is a replayed integral
/// and not a hand.
pub const SEAM: f32 = 4.0;

/// Height of a tab strip.
pub const STRIP: f32 = 11.0;

/// Fraction of the smaller side that counts as an edge for a drop. The rest is
/// the centre, so the common gesture — drop onto a pane to tab with it — is the
/// big target and the four splits are the deliberate ones.
const EDGE: f32 = 0.28;

/// Where a node sits in the tree: one bit per level, least significant first,
/// `depth` of them. `Copy` and allocation-free, so a seam's widget id can be
/// derived from its position without a path buffer per frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Path {
    bits: u32,
    depth: u8,
}

/// Deeper than this and a path stops being addressable. Twenty-four splits is
/// more panes than a screen holds; the limit exists so the bit path cannot
/// silently alias, not because a layout is expected to approach it.
pub const MAX_DEPTH: u8 = 24;

impl Path {
    /// The path of a child: `second` picks the right or bottom one.
    ///
    /// Saturates at [`MAX_DEPTH`] rather than wrapping — a path that deep
    /// addresses the wrong node, and returning the parent's is the failure that
    /// makes a seam refuse to move instead of moving a different seam.
    #[must_use]
    pub fn child(self, second: bool) -> Path {
        match self.depth < MAX_DEPTH {
            true => Path {
                bits: self.bits | (u32::from(second) << self.depth),
                depth: self.depth + 1,
            },
            false => self,
        }
    }

    /// A stable number for this position, for a widget id.
    #[must_use]
    pub fn key(self) -> u64 {
        u64::from(self.bits) << 8 | u64::from(self.depth)
    }
}

/// A node of the layout tree.
#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    /// Two children laid along `axis` — [`Axis::Horizontal`] puts them side by
    /// side, [`Axis::Vertical`] one above the other, matching [`crate::Stack`].
    Split {
        /// Which way the children are laid.
        axis: Axis,
        /// Where the seam sits, as a fraction of the split's extent along
        /// `axis`. Clamped on resolve, never on store: a fraction that is too
        /// small for *this* window is fine again in a bigger one, and clamping
        /// on store would ratchet a layout narrower every time it was resized.
        at: f32,
        /// Left or top.
        first: Box<Node>,
        /// Right or bottom.
        second: Box<Node>,
    },
    /// A group of panes sharing one region, one visible at a time.
    Tabs {
        /// In strip order.
        panes: Vec<PaneId>,
        /// Index into `panes` of the visible one.
        active: usize,
    },
}

impl Node {
    /// A leaf holding one pane.
    #[must_use]
    pub fn pane(pane: PaneId) -> Node {
        Node::Tabs {
            panes: vec![pane],
            active: 0,
        }
    }

    /// A split of two subtrees.
    #[must_use]
    pub fn split(axis: Axis, at: f32, first: Node, second: Node) -> Node {
        Node::Split {
            axis,
            at,
            first: Box::new(first),
            second: Box::new(second),
        }
    }
}

/// One resolved tab group.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Group {
    /// Strip and body together — what a drop is tested against.
    pub region: Rect,
    /// The tab strip along the top.
    pub strip: Rect,
    /// What the active pane draws into.
    pub body: Rect,
    /// First of this group's entries in [`Dock::tabs`].
    pub first: u32,
    /// How many it has.
    pub count: u32,
    /// Which of them is visible, relative to `first`.
    pub active: u32,
    /// Where the group sits in the tree.
    pub path: Path,
}

/// One resolved tab.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tab {
    /// The pane it names.
    pub pane: PaneId,
    /// Its rectangle in the strip. Empty when the strip ran out of room, which
    /// is a narrow group and not an error.
    pub rect: Rect,
}

/// One resolved seam.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Seam {
    /// The split it belongs to.
    pub path: Path,
    /// Which way it slides — a [`Axis::Horizontal`] split has a vertical seam
    /// that moves left and right.
    pub axis: Axis,
    /// The grabbable rectangle.
    pub rect: Rect,
    /// The span the seam travels in, along `axis`: `(start, size)`. What
    /// [`Dock::steer`] converts a pointer position into a fraction with.
    pub span: (f32, f32),
}

/// Which part of a group a drop lands on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Zone {
    /// Join the group as another tab.
    Centre,
    /// Split it, taking the left half.
    Left,
    /// Split it, taking the right half.
    Right,
    /// Split it, taking the top half.
    Top,
    /// Split it, taking the bottom half.
    Bottom,
}

impl Zone {
    /// The axis and order a split for this zone is built with; `None` for
    /// [`Zone::Centre`], which is not a split.
    fn split(self) -> Option<(Axis, bool)> {
        match self {
            Zone::Centre => None,
            Zone::Left => Some((Axis::Horizontal, true)),
            Zone::Right => Some((Axis::Horizontal, false)),
            Zone::Top => Some((Axis::Vertical, true)),
            Zone::Bottom => Some((Axis::Vertical, false)),
        }
    }
}

/// Where a dragged pane would land.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Drop {
    /// The group under the pointer, named by its active pane.
    pub onto: PaneId,
    /// Which part of it.
    pub zone: Zone,
}

/// A layout tree and its last resolution.
///
/// Allocation-free once settled, like [`crate::Router`]: the three result
/// vectors are cleared and refilled per frame and the tree itself only allocates
/// when the operator rearranges it.
#[derive(Clone, Debug)]
pub struct Dock {
    root: Node,
    groups: Vec<Group>,
    tabs: Vec<Tab>,
    seams: Vec<Seam>,
}

impl Dock {
    /// A dock over `root`.
    #[must_use]
    pub fn new(root: Node) -> Dock {
        Dock {
            root,
            groups: Vec::new(),
            tabs: Vec::new(),
            seams: Vec::new(),
        }
    }

    /// The tree, for a caller persisting it.
    #[must_use]
    pub fn root(&self) -> &Node {
        &self.root
    }

    /// Replace the tree — loading a persisted layout. The next
    /// [`resolve`](Self::resolve) is what makes it visible.
    pub fn set_root(&mut self, root: Node) {
        self.root = root;
    }

    /// The groups of the last resolve, in tree order.
    #[must_use]
    pub fn groups(&self) -> &[Group] {
        &self.groups
    }

    /// Every tab of the last resolve; a [`Group`] indexes a run of these.
    #[must_use]
    pub fn tabs(&self) -> &[Tab] {
        &self.tabs
    }

    /// The seams of the last resolve.
    #[must_use]
    pub fn seams(&self) -> &[Seam] {
        &self.seams
    }

    /// Lay the tree out over `area`, `tab` wide per pane.
    ///
    /// The width closure is the caller's because a tab is as wide as its label
    /// and this module has never seen a label. Every rectangle it produces is
    /// inside `area`: the groups and the seams together cover it exactly and no
    /// two groups overlap, which is what makes "the panels are the whole window"
    /// a property rather than a hope.
    pub fn resolve<F: Fn(PaneId) -> f32>(&mut self, area: Rect, tab: F) {
        self.groups.clear();
        self.tabs.clear();
        self.seams.clear();
        // Split borrows: the walk needs `&self.root` and `&mut` the outputs.
        let root = core::mem::replace(
            &mut self.root,
            Node::Tabs {
                panes: Vec::new(),
                active: 0,
            },
        );
        let mut out = Resolved {
            groups: &mut self.groups,
            tabs: &mut self.tabs,
            seams: &mut self.seams,
        };
        walk(&root, Path::default(), area, &tab, &mut out);
        self.root = root;
    }

    /// The body rectangle a pane draws into, if it is the visible one of its
    /// group. `None` for a pane behind another tab — which is the caller's cue
    /// to draw nothing, not to draw it somewhere else.
    #[must_use]
    pub fn body_of(&self, pane: PaneId) -> Option<Rect> {
        self.groups
            .iter()
            .find(|g| self.active_of(g) == Some(pane))
            .map(|g| g.body)
    }

    /// The visible pane of a group.
    #[must_use]
    pub fn active_of(&self, group: &Group) -> Option<PaneId> {
        self.tabs
            .get((group.first + group.active) as usize)
            .map(|t| t.pane)
    }

    /// Show `pane` in its group.
    pub fn activate(&mut self, pane: PaneId) {
        activate(&mut self.root, pane);
    }

    /// Put the seam at `path` where the pointer is: `along` is the pointer's
    /// coordinate on the seam's axis, and `span` comes off the [`Seam`].
    ///
    /// Absolute rather than a delta, so a drag cannot accumulate error and the
    /// seam is under the pointer on every frame of it rather than drifting from
    /// it. Clamped to leave [`MIN`] on both sides.
    pub fn steer(&mut self, path: Path, span: (f32, f32), along: f32) {
        let (start, size) = span;
        if size <= 0.0 {
            return;
        }
        let (low, high) = range(size);
        let at = ((along - start) / size).clamp(low, high);
        if let Some(Node::Split { at: stored, .. }) = node_mut(&mut self.root, path) {
            *stored = at;
        }
    }

    /// Where a pane dropped at `(x, y)` would land, or `None` over no group.
    #[must_use]
    pub fn drop_at(&self, x: f32, y: f32) -> Option<Drop> {
        let group = self.groups.iter().find(|g| g.region.contains(x, y))?;
        let onto = self.active_of(group)?;
        Some(Drop {
            onto,
            zone: zone_of(&group.region, x, y),
        })
    }

    /// The rectangle a [`Drop`] would fill, for the caller to highlight. Purely
    /// advisory — it is what the move produces, computed the same way the move
    /// computes it.
    #[must_use]
    pub fn preview(&self, drop: Drop) -> Option<Rect> {
        let group = self
            .groups
            .iter()
            .find(|g| self.active_of(g) == Some(drop.onto))?;
        let region = group.region;
        Some(match drop.zone.split() {
            None => region,
            Some((Axis::Horizontal, first)) => {
                let w = (region.w - SEAM) * 0.5;
                let x = if first { region.x } else { region.right() - w };
                Rect::new(x, region.y, w, region.h)
            }
            Some((Axis::Vertical, first)) => {
                let h = (region.h - SEAM) * 0.5;
                let y = if first { region.y } else { region.bottom() - h };
                Rect::new(region.x, y, region.w, h)
            }
        })
    }

    /// Move `pane` onto another group.
    ///
    /// A no-op when it would not change anything — dropping a pane on its own
    /// group's centre, or moving the only pane there is. Both are the ordinary
    /// end of a drag the operator thought better of, not errors.
    pub fn move_pane(&mut self, pane: PaneId, drop: Drop) {
        if pane == drop.onto || self.panes() <= 1 {
            return;
        }
        // Detached first, so the insert sees the tree the move leaves behind:
        // a group that emptied has already collapsed and cannot be split into.
        let Some(root) = without(core::mem::replace(&mut self.root, Node::pane(pane)), pane) else {
            return;
        };
        self.root = insert(root, pane, drop);
    }

    /// How many panes the tree holds.
    #[must_use]
    pub fn panes(&self) -> usize {
        count(&self.root)
    }
}

/// The three output vectors, as one borrow the walk can carry.
struct Resolved<'a> {
    groups: &'a mut Vec<Group>,
    tabs: &'a mut Vec<Tab>,
    seams: &'a mut Vec<Seam>,
}

fn walk<F: Fn(PaneId) -> f32>(node: &Node, path: Path, area: Rect, tab: &F, out: &mut Resolved) {
    match node {
        Node::Tabs { panes, active } => {
            let strip = Rect::new(area.x, area.y, area.w, STRIP.min(area.h));
            let first = out.tabs.len() as u32;
            let mut pen = strip.x;
            for pane in panes {
                let w = tab(*pane);
                // Clipped rather than scrolled: a strip with more tabs than room
                // is a group the operator can still split, and a scrollable strip
                // is a scroll container this crate does not have yet (§4.9).
                let rect = Rect::new(pen, strip.y, w.min((strip.right() - pen).max(0.0)), strip.h);
                out.tabs.push(Tab { pane: *pane, rect });
                pen += w;
            }
            out.groups.push(Group {
                region: area,
                strip,
                body: Rect::new(area.x, strip.bottom(), area.w, area.h - strip.h),
                first,
                count: panes.len() as u32,
                active: (*active).min(panes.len().saturating_sub(1)) as u32,
                path,
            });
        }
        Node::Split {
            axis,
            at,
            first,
            second,
        } => {
            let (start, size) = match axis {
                Axis::Horizontal => (area.x, area.w),
                Axis::Vertical => (area.y, area.h),
            };
            let cut = start + size * clamp_at(*at, size);
            let (a, b, seam) = match axis {
                Axis::Horizontal => (
                    Rect::new(area.x, area.y, cut - SEAM * 0.5 - area.x, area.h),
                    Rect::new(
                        cut + SEAM * 0.5,
                        area.y,
                        area.right() - cut - SEAM * 0.5,
                        area.h,
                    ),
                    Rect::new(cut - SEAM * 0.5, area.y, SEAM, area.h),
                ),
                Axis::Vertical => (
                    Rect::new(area.x, area.y, area.w, cut - SEAM * 0.5 - area.y),
                    Rect::new(
                        area.x,
                        cut + SEAM * 0.5,
                        area.w,
                        area.bottom() - cut - SEAM * 0.5,
                    ),
                    Rect::new(area.x, cut - SEAM * 0.5, area.w, SEAM),
                ),
            };
            out.seams.push(Seam {
                path,
                axis: *axis,
                rect: seam,
                span: (start, size),
            });
            walk(first, path.child(false), a, tab, out);
            walk(second, path.child(true), b, tab, out);
        }
    }
}

/// The fractions a seam may sit at on a span `size` long, leaving [`MIN`] of
/// *pane* either side — so the half-seam each child gives up is counted, or a
/// seam driven to its limit would leave a pane [`SEAM`] narrower than the
/// minimum and one drag from unreachable.
///
/// Collapses to a centred split when `size` has no room for two minimums: a
/// window too small for the layout gets a squashed one rather than a layout
/// with a negative pane in it.
fn range(size: f32) -> (f32, f32) {
    let low = ((MIN + SEAM * 0.5) / size.max(1.0)).min(0.5);
    (low, 1.0 - low)
}

fn clamp_at(at: f32, size: f32) -> f32 {
    let (low, high) = range(size);
    at.clamp(low, high)
}

fn zone_of(region: &Rect, x: f32, y: f32) -> Zone {
    let band = region.w.min(region.h) * EDGE;
    // Nearest edge wins, so the zones meet on the diagonals of the corners
    // rather than one of them swallowing a corner by test order.
    let edges = [
        (x - region.x, Zone::Left),
        (region.right() - x, Zone::Right),
        (y - region.y, Zone::Top),
        (region.bottom() - y, Zone::Bottom),
    ];
    edges
        .iter()
        .filter(|(distance, _)| *distance < band)
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map_or(Zone::Centre, |(_, zone)| *zone)
}

fn node_mut(node: &mut Node, path: Path) -> Option<&mut Node> {
    let mut at = node;
    for level in 0..path.depth {
        let second = path.bits & (1 << level) != 0;
        match at {
            Node::Split {
                first, second: b, ..
            } => at = if second { b } else { first },
            Node::Tabs { .. } => return None,
        }
    }
    Some(at)
}

fn activate(node: &mut Node, pane: PaneId) {
    match node {
        Node::Tabs { panes, active } => {
            if let Some(i) = panes.iter().position(|p| *p == pane) {
                *active = i;
            }
        }
        Node::Split { first, second, .. } => {
            activate(first, pane);
            activate(second, pane);
        }
    }
}

fn count(node: &Node) -> usize {
    match node {
        Node::Tabs { panes, .. } => panes.len(),
        Node::Split { first, second, .. } => count(first) + count(second),
    }
}

/// The tree without `pane`, or `None` if that empties it. A split whose child
/// empties collapses into its sibling, which is what keeps a dock from
/// accumulating seams around nothing.
fn without(node: Node, pane: PaneId) -> Option<Node> {
    match node {
        Node::Tabs {
            mut panes,
            mut active,
        } => {
            if let Some(i) = panes.iter().position(|p| *p == pane) {
                panes.remove(i);
                // The tab to the left of the one that left, so closing along a
                // strip walks it instead of jumping to the end.
                active = active.min(panes.len().saturating_sub(1));
            }
            match panes.is_empty() {
                true => None,
                false => Some(Node::Tabs { panes, active }),
            }
        }
        Node::Split {
            axis,
            at,
            first,
            second,
        } => match (without(*first, pane), without(*second, pane)) {
            (Some(a), Some(b)) => Some(Node::split(axis, at, a, b)),
            (Some(only), None) | (None, Some(only)) => Some(only),
            (None, None) => None,
        },
    }
}

fn insert(node: Node, pane: PaneId, drop: Drop) -> Node {
    match node {
        Node::Tabs { panes, active } if panes.contains(&drop.onto) => match drop.zone.split() {
            None => {
                let mut panes = panes;
                panes.push(pane);
                let active = panes.len() - 1;
                Node::Tabs { panes, active }
            }
            Some((axis, first)) => {
                let group = Node::Tabs { panes, active };
                let leaf = Node::pane(pane);
                match first {
                    true => Node::split(axis, 0.5, leaf, group),
                    false => Node::split(axis, 0.5, group, leaf),
                }
            }
        },
        Node::Tabs { .. } => node,
        Node::Split {
            axis,
            at,
            first,
            second,
        } => Node::split(
            axis,
            at,
            insert(*first, pane, drop),
            insert(*second, pane, drop),
        ),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    const A: PaneId = PaneId(0);
    const B: PaneId = PaneId(1);
    const C: PaneId = PaneId(2);
    const D: PaneId = PaneId(3);

    const AREA: Rect = Rect::new(0.0, 0.0, 800.0, 600.0);

    /// Four panes in the shape the editor opens on: a left column, a right
    /// column, and the middle split above and below.
    fn tree() -> Node {
        Node::split(
            Axis::Horizontal,
            0.22,
            Node::pane(A),
            Node::split(
                Axis::Horizontal,
                0.74,
                Node::split(Axis::Vertical, 0.72, Node::pane(B), Node::pane(C)),
                Node::pane(D),
            ),
        )
    }

    fn dock() -> Dock {
        let mut dock = Dock::new(tree());
        dock.resolve(AREA, |_| 30.0);
        dock
    }

    /// The property the whole change exists for: the panels are the whole
    /// window. Every pixel of the area belongs to exactly one group or to a
    /// seam, and nothing lands outside it.
    #[test]
    fn groups_and_seams_tile_the_area_exactly() {
        let dock = dock();
        let mut covered = 0.0;
        for group in dock.groups() {
            assert_eq!(
                group.region.intersect(&AREA),
                group.region,
                "{:?} escapes the area",
                group.region
            );
            covered += group.region.w * group.region.h;
        }
        for seam in dock.seams() {
            covered += seam.rect.w * seam.rect.h;
        }
        // Seams cross one another where two splits meet, so the overlap is
        // counted twice and subtracted once — an exact figure, not a tolerance.
        let mut overlap = 0.0;
        for (i, a) in dock.seams().iter().enumerate() {
            for b in &dock.seams()[i + 1..] {
                let hit = a.rect.intersect(&b.rect);
                if !hit.is_empty() {
                    overlap += hit.w * hit.h;
                }
            }
        }
        assert!(
            (covered - overlap - AREA.w * AREA.h).abs() < 0.01,
            "covered {covered} minus {overlap} is not {}",
            AREA.w * AREA.h
        );

        for (i, a) in dock.groups().iter().enumerate() {
            for b in &dock.groups()[i + 1..] {
                assert!(
                    a.region.intersect(&b.region).is_empty(),
                    "{:?} overlaps {:?}",
                    a.region,
                    b.region
                );
            }
        }
    }

    /// A group's strip and body partition its region, and the body is what a
    /// pane is told to draw into.
    #[test]
    fn a_group_is_its_strip_and_its_body() {
        let dock = dock();
        for group in dock.groups() {
            assert_eq!(group.strip.union(&group.body), group.region);
            assert!(group.strip.intersect(&group.body).is_empty());
        }
        assert_eq!(dock.body_of(A), dock.groups().first().map(|g| g.body));
    }

    /// Dropping on a centre makes a tab; the pane leaves its old group and that
    /// group's split collapses, so the seam count falls by one.
    #[test]
    fn a_centre_drop_tabs_the_pane_and_collapses_the_seam_it_left() {
        let mut dock = dock();
        let seams = dock.seams().len();
        dock.move_pane(
            C,
            Drop {
                onto: A,
                zone: Zone::Centre,
            },
        );
        dock.resolve(AREA, |_| 30.0);
        assert_eq!(dock.panes(), 4, "moved, not lost");
        assert_eq!(dock.seams().len(), seams - 1, "B's split collapsed");
        let group = dock
            .groups()
            .iter()
            .find(|g| dock.active_of(g) == Some(C))
            .expect("C is visible");
        assert_eq!(group.count, 2, "A and C share a strip");
        // The dropped pane comes up in front — an operator who just moved it is
        // looking for it.
        assert_eq!(dock.active_of(group), Some(C));
    }

    /// An edge drop splits the group it landed on, in the right order.
    #[test]
    fn an_edge_drop_splits_the_group_it_landed_on() {
        for (zone, expect_first) in [
            (Zone::Left, true),
            (Zone::Right, false),
            (Zone::Top, true),
            (Zone::Bottom, false),
        ] {
            let mut dock = dock();
            dock.move_pane(D, Drop { onto: A, zone });
            dock.resolve(AREA, |_| 30.0);
            assert_eq!(dock.panes(), 4);
            let moved = dock
                .groups()
                .iter()
                .find(|g| dock.active_of(g) == Some(D))
                .copied()
                .expect("D is placed");
            let onto = dock
                .groups()
                .iter()
                .find(|g| dock.active_of(g) == Some(A))
                .copied()
                .expect("A is placed");
            let (moved_edge, onto_edge) = match zone {
                Zone::Left | Zone::Right => (moved.region.x, onto.region.x),
                _ => (moved.region.y, onto.region.y),
            };
            assert_eq!(
                moved_edge < onto_edge,
                expect_first,
                "{zone:?} put D on the wrong side"
            );
        }
    }

    /// Dropping a pane back on its own group changes nothing, and the last pane
    /// cannot be moved at all — there would be nothing left to drop it onto.
    #[test]
    fn a_drop_that_would_change_nothing_is_a_no_op() {
        let mut dock = dock();
        let before = dock.root().clone();
        dock.move_pane(
            A,
            Drop {
                onto: A,
                zone: Zone::Centre,
            },
        );
        assert_eq!(dock.root(), &before);

        let mut one = Dock::new(Node::pane(A));
        one.resolve(AREA, |_| 30.0);
        one.move_pane(
            A,
            Drop {
                onto: B,
                zone: Zone::Left,
            },
        );
        assert_eq!(one.panes(), 1);
    }

    /// A seam follows the pointer exactly while there is room, and stops with
    /// [`MIN`] left rather than at the edge.
    #[test]
    fn a_seam_lands_under_the_pointer_and_stops_at_the_minimum() {
        let mut dock = dock();
        let seam = dock.seams()[0];
        dock.steer(seam.path, seam.span, 500.0);
        dock.resolve(AREA, |_| 30.0);
        assert!(
            (dock.seams()[0].rect.x + SEAM * 0.5 - 500.0).abs() < 0.01,
            "seam at {:?}, wanted 500",
            dock.seams()[0].rect.x
        );

        // Dragged off the left edge: the pane keeps exactly MIN — not MIN less
        // the half-seam it gives up, which is the off-by-a-seam this pins.
        dock.steer(seam.path, seam.span, -1000.0);
        dock.resolve(AREA, |_| 30.0);
        assert!(
            (dock.groups()[0].region.w - MIN).abs() < 0.01,
            "{:?} is not MIN wide",
            dock.groups()[0].region
        );
        for group in dock.groups() {
            assert!(
                group.region.w >= MIN - 0.01,
                "{:?} is too narrow",
                group.region
            );
        }
    }

    /// A window too small for two minimums gets a squashed layout, not a
    /// negative one. The case is a mid-resize frame, not a configuration.
    #[test]
    fn an_area_with_no_room_still_produces_sane_rectangles() {
        let mut dock = dock();
        dock.resolve(Rect::new(0.0, 0.0, 40.0, 30.0), |_| 30.0);
        for group in dock.groups() {
            assert!(group.region.w >= 0.0 && group.region.h >= 0.0);
            assert!(group.body.h >= 0.0, "{:?}", group.body);
        }
    }

    /// Corners belong to the nearer edge, and the middle is centre — the map a
    /// drag's highlight is read off.
    #[test]
    fn zones_split_a_region_by_nearest_edge() {
        let r = Rect::new(0.0, 0.0, 200.0, 200.0);
        assert_eq!(zone_of(&r, 100.0, 100.0), Zone::Centre);
        assert_eq!(zone_of(&r, 5.0, 100.0), Zone::Left);
        assert_eq!(zone_of(&r, 195.0, 100.0), Zone::Right);
        assert_eq!(zone_of(&r, 100.0, 5.0), Zone::Top);
        assert_eq!(zone_of(&r, 100.0, 195.0), Zone::Bottom);
        // A corner: nearer the top than the left, so it is a top split.
        assert_eq!(zone_of(&r, 20.0, 4.0), Zone::Top);
        assert_eq!(zone_of(&r, 4.0, 20.0), Zone::Left);
    }

    /// The highlight a drag shows is the rectangle the drop actually produces.
    /// Two implementations of one rule would drift, so this pins them together.
    #[test]
    fn the_preview_is_where_the_pane_lands() {
        for zone in [Zone::Left, Zone::Right, Zone::Top, Zone::Bottom] {
            let mut dock = dock();
            let drop = Drop { onto: A, zone };
            let preview = dock.preview(drop).expect("A is placed");
            dock.move_pane(D, drop);
            dock.resolve(AREA, |_| 30.0);
            let landed = dock.body_of(D).expect("D is visible");
            let region = dock
                .groups()
                .iter()
                .find(|g| dock.active_of(g) == Some(D))
                .map(|g| g.region)
                .expect("D is placed");
            assert!(
                (region.x - preview.x).abs() < 0.01 && (region.y - preview.y).abs() < 0.01,
                "{zone:?}: landed at {region:?}, previewed {preview:?}"
            );
            assert!(!landed.is_empty());
        }
    }

    /// Activating walks the strip without touching any other group.
    #[test]
    fn activating_a_pane_shows_it_and_leaves_its_neighbours_alone() {
        let mut dock = dock();
        dock.move_pane(
            C,
            Drop {
                onto: A,
                zone: Zone::Centre,
            },
        );
        dock.resolve(AREA, |_| 30.0);
        assert_eq!(dock.body_of(A), None, "C came up in front");
        dock.activate(A);
        dock.resolve(AREA, |_| 30.0);
        assert!(dock.body_of(A).is_some() && dock.body_of(C).is_none());
        assert!(dock.body_of(B).is_some(), "another group is untouched");
    }

    /// Paths address what they were handed out for. A seam's path must still
    /// name its own split after the tree is walked again, or a drag moves a
    /// different seam.
    #[test]
    fn a_seam_path_addresses_its_own_split() {
        let mut dock = dock();
        let seams: Vec<Seam> = dock.seams().to_vec();
        assert_eq!(seams.len(), 3);
        for seam in &seams {
            assert!(
                matches!(
                    node_mut(&mut dock.root, seam.path),
                    Some(Node::Split { .. })
                ),
                "{:?} does not name a split",
                seam.path
            );
        }
        // And they are distinct, or two seams would share a widget id.
        for (i, a) in seams.iter().enumerate() {
            for b in &seams[i + 1..] {
                assert_ne!(a.path.key(), b.path.key());
            }
        }
    }

    /// Steady state allocates nothing — the same claim `Router` makes, for the
    /// same reason: this runs inside a frame.
    #[test]
    fn a_settled_resolve_reuses_its_buffers() {
        let mut dock = dock();
        for _ in 0..8 {
            dock.resolve(AREA, |_| 30.0);
        }
        let caps = (
            dock.groups.capacity(),
            dock.tabs.capacity(),
            dock.seams.capacity(),
        );
        for i in 0..64 {
            dock.resolve(AREA, |_| 30.0 + i as f32 * 0.1);
        }
        assert_eq!(
            (
                dock.groups.capacity(),
                dock.tabs.capacity(),
                dock.seams.capacity()
            ),
            caps
        );
    }
}
