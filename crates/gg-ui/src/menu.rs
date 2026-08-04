//! Menus (§4.9's owned list, §6 M15.1 item 5) — a strip of titles, one of which
//! may be dropped down.
//!
//! Geometry and one bit of state, in [`dock`](crate::dock)'s shape and for the
//! same reason: which menu is open has to survive the frame that opened it, and
//! everything else is recomputed. The caller draws — this crate hands out
//! rectangles and never a colour — and the caller measures, because a label's
//! width is a property of the font it will be set in and this module has no
//! opinion about that (`gg-editor` sets its panels in a rented face whose advance
//! is not [`crate::font`]'s cell).
//!
//! # Why the open menu is a parameter and not just a field
//!
//! [`MenuBar::resolve`] takes which menu to drop down rather than reading its own
//! state, so a caller can ask where an item *would* be without opening it. That
//! is what lets a scripted session aim at `file → save` off a closed editor
//! (`gg_editor::session`), which is the difference between a script that follows
//! the layout and a script full of literals.

use crate::draw::Rect;

/// One menu: the title in the strip, and what drops out of it.
#[derive(Clone, Copy, Debug)]
pub struct Menu<'a> {
    /// What the strip says.
    pub title: &'a str,
    /// The items, top to bottom.
    pub items: &'a [&'a str],
}

/// Height of a title cell and of an item row.
pub const ROW: f32 = 9.0;

/// Space either side of a label, inside its cell.
pub const PAD: f32 = 4.0;

/// Distance between two item rows — [`ROW`] plus the line between them.
pub const PITCH: f32 = ROW + 1.0;

/// The strip's state: which menu is down. Everything else is resolved.
#[derive(Debug, Default)]
pub struct MenuBar {
    open: Option<usize>,
    titles: Vec<Rect>,
    items: Vec<Rect>,
    panel: Option<Rect>,
}

impl MenuBar {
    /// Which menu is down.
    #[must_use]
    pub fn open(&self) -> Option<usize> {
        self.open
    }

    /// Drop `i` down, or put it away if it already is. The gesture a title
    /// click is.
    pub fn toggle(&mut self, i: usize) {
        self.open = match self.open == Some(i) {
            true => None,
            false => Some(i),
        };
    }

    /// Put whatever is down away — what a press anywhere else means.
    pub fn close(&mut self) {
        self.open = None;
    }

    /// Lay `menus` out across `bar` from `x`, dropping `open` down.
    ///
    /// `width` measures a label in the caller's own font. The panel is kept
    /// inside `bar` horizontally, so a menu near the right edge opens leftward
    /// rather than off the surface; vertically it is free, because it is drawn
    /// over whatever the bar sits above.
    pub fn resolve(
        &mut self,
        bar: Rect,
        x: f32,
        menus: &[Menu],
        open: Option<usize>,
        width: &dyn Fn(&str) -> f32,
    ) {
        self.titles.clear();
        self.items.clear();
        self.panel = None;
        let top = (bar.y + (bar.h - ROW) * 0.5).floor();
        let mut at = x;
        for menu in menus {
            let cell = Rect::new(at, top, width(menu.title) + PAD * 2.0, ROW);
            at = cell.right();
            self.titles.push(cell);
        }
        let Some((menu, title)) = open.and_then(|i| menus.get(i).zip(self.titles.get(i))) else {
            return;
        };
        if menu.items.is_empty() {
            return;
        }
        let widest = menu
            .items
            .iter()
            .map(|item| width(item) + PAD * 2.0)
            .fold(title.w, f32::max);
        let panel = Rect::new(
            title.x.min(bar.right() - widest).max(bar.x),
            bar.bottom(),
            widest,
            menu.items.len() as f32 * PITCH + 1.0,
        );
        for (k, _) in menu.items.iter().enumerate() {
            self.items.push(Rect::new(
                panel.x + 1.0,
                panel.y + 1.0 + k as f32 * PITCH,
                (panel.w - 2.0).max(0.0),
                ROW,
            ));
        }
        self.panel = Some(panel);
    }

    /// The title cells, in the order they were resolved.
    #[must_use]
    pub fn titles(&self) -> &[Rect] {
        &self.titles
    }

    /// The dropped-down panel, for the caller to plate before drawing items
    /// into it.
    #[must_use]
    pub fn panel(&self) -> Option<Rect> {
        self.panel
    }

    /// The open menu's item rows, top to bottom. Empty when nothing is down.
    #[must_use]
    pub fn items(&self) -> &[Rect] {
        &self.items
    }

    /// Where the strip ends — where a caller puts whatever follows the menus.
    #[must_use]
    pub fn right(&self) -> f32 {
        self.titles.last().map_or(0.0, Rect::right)
    }

    /// Whether `(x, y)` is over a title or over the open panel.
    ///
    /// What a press outside means is "put it away", and this is how a caller
    /// tells the two apart without asking the router about every cell.
    #[must_use]
    pub fn contains(&self, x: f32, y: f32) -> bool {
        self.titles.iter().any(|t| t.contains(x, y)) || self.panel.is_some_and(|p| p.contains(x, y))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    const BAR: Rect = Rect::new(0.0, 0.0, 200.0, 13.0);
    const MENUS: &[Menu] = &[
        Menu {
            title: "file",
            items: &["save", "quit"],
        },
        Menu {
            title: "view",
            items: &["reset layout"],
        },
    ];

    /// Six units per character — this file has no font, so a test picks one.
    fn width(text: &str) -> f32 {
        text.chars().count() as f32 * 6.0
    }

    fn resolved(open: Option<usize>) -> MenuBar {
        let mut bar = MenuBar::default();
        bar.resolve(BAR, 3.0, MENUS, open, &width);
        bar
    }

    /// Titles march along the strip, touching and never overlapping, inside the
    /// bar they were given.
    #[test]
    fn the_titles_tile_the_strip_in_order() {
        let menus = resolved(None);
        assert_eq!(menus.titles().len(), MENUS.len());
        let mut previous: Option<Rect> = None;
        for (title, menu) in menus.titles().iter().zip(MENUS) {
            assert_eq!(title.w, width(menu.title) + PAD * 2.0);
            assert!(title.y >= BAR.y && title.bottom() <= BAR.bottom());
            if let Some(before) = previous {
                assert_eq!(title.x, before.right(), "{} is not next", menu.title);
            }
            previous = Some(*title);
        }
        assert_eq!(menus.right(), previous.map_or(0.0, |r| r.right()));
        assert!(menus.panel().is_none() && menus.items().is_empty());
    }

    /// The panel hangs under its own title, one row per item, in order.
    #[test]
    fn the_open_menu_drops_under_its_title() {
        let menus = resolved(Some(0));
        let panel = menus.panel().expect("file is open");
        let title = menus.titles()[0];
        assert_eq!(panel.x, title.x);
        assert_eq!(panel.y, BAR.bottom(), "under the strip, not inside it");
        assert_eq!(menus.items().len(), 2);
        for (k, item) in menus.items().iter().enumerate() {
            assert!(panel.contains(item.x, item.y), "item {k} left the panel");
            assert!(item.bottom() <= panel.bottom() + 0.01);
            if k > 0 {
                assert_eq!(item.y - menus.items()[k - 1].y, PITCH);
            }
        }
        // Wide enough for its widest item, which is what a menu is measured by.
        assert!(panel.w >= width("quit") + PAD * 2.0);
        assert!(menus.contains(
            item_centre(menus.items()[1]).0,
            item_centre(menus.items()[1]).1
        ));
        assert!(menus.contains(title.x + 1.0, title.y + 1.0));
        assert!(!menus.contains(BAR.right() - 1.0, BAR.bottom() + 40.0));
    }

    fn item_centre(rect: Rect) -> (f32, f32) {
        (rect.x + rect.w * 0.5, rect.y + rect.h * 0.5)
    }

    /// A menu at the right edge opens leftward instead of off the surface.
    #[test]
    fn a_panel_that_would_overhang_is_pulled_back_inside() {
        let mut menus = MenuBar::default();
        menus.resolve(BAR, BAR.right() - 20.0, MENUS, Some(0), &width);
        let panel = menus.panel().expect("file is open");
        assert!(panel.right() <= BAR.right() + 0.01, "{panel:?} overhangs");
        assert!(panel.x >= BAR.x);
    }

    /// Click to open, click again to put away, and one at a time.
    #[test]
    fn one_menu_is_down_at_a_time() {
        let mut menus = resolved(None);
        assert_eq!(menus.open(), None);
        menus.toggle(0);
        assert_eq!(menus.open(), Some(0));
        menus.toggle(1);
        assert_eq!(menus.open(), Some(1));
        menus.toggle(1);
        assert_eq!(menus.open(), None);
        menus.toggle(1);
        menus.close();
        assert_eq!(menus.open(), None);
    }

    /// An index no menu has, and a menu with nothing in it: no panel, no items,
    /// and the titles are still there.
    #[test]
    fn an_absent_or_empty_menu_drops_nothing() {
        let menus = resolved(Some(9));
        assert!(menus.panel().is_none() && menus.items().is_empty());
        assert_eq!(menus.titles().len(), MENUS.len());

        let mut empty = MenuBar::default();
        empty.resolve(
            BAR,
            3.0,
            &[Menu {
                title: "edit",
                items: &[],
            }],
            Some(0),
            &width,
        );
        assert!(empty.panel().is_none() && empty.items().is_empty());
        assert_eq!(empty.titles().len(), 1);
    }
}
