//! Tetris, driven the way the host drives it (§4.2.2).
//!
//! Nothing here calls a system directly — every tick goes through the systems
//! table against a `World` registered from the *declared* table, which is the
//! path `gg-runtime` takes. That is what makes a passing test here evidence
//! about the game rather than about these functions.
//!
//! Buttons are set on the [`InputFrame`] the same way a replay sets them, so a
//! test that holds a key and a recording that holds a key are the same input to
//! the same code (§4.7).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use demo_10_tetris::{
    Banner, Bay, COLORS, Cell, HARD_DROP, HEIGHT, HIDDEN, HOLD, HOLD_BAY, HudLine, LEFT,
    MIN_GRAVITY_TICKS, NEXT_BAY, NO_PIECE, Piece, Play, RESTART, ROTATE_CW, Rules, SHAPES, WIDTH,
    Well, cells_of, clear_rows, collides, color_of, draw, gravity_for, landing_row, new_play,
    spawn_piece,
};
use gg_ecs::boundary::{
    self, AbiInfo, ActionId, ComponentsTable, HostApiV1, InputFrame, SystemsTable, TickCtx, Widget,
};
use gg_ecs::{Query, World};

// The symbols `gg_game!` exported into this crate's rlib.
unsafe extern "C" {
    fn gg_game_abi() -> AbiInfo;
    fn gg_game_init(api: *const HostApiV1);
    fn gg_game_components() -> ComponentsTable;
    fn gg_game_systems() -> SystemsTable;
}

struct Game {
    world: World,
    table: SystemsTable,
    tick: u64,
    /// This tick's buttons; cleared after every step, so `hold` is expressed by
    /// setting it again rather than by remembering to unset it.
    held: u64,
    previous: u64,
}

impl Game {
    fn load() -> Self {
        // SAFETY: the symbols are this binary's own, linked from the rlib.
        let info = unsafe { gg_game_abi() };
        assert_eq!(info, boundary::abi_info(), "one build, one description");
        // SAFETY: `host_api()` returns a `&'static` table (§4.2.2).
        unsafe { gg_game_init(core::ptr::from_ref(boundary::host_api())) };

        let mut world = World::new();
        // SAFETY: the tables are this binary's own and live for the process.
        let (table, declared) = unsafe {
            let declared = world.adopt(&gg_game_components()).unwrap();
            (gg_game_systems(), declared)
        };
        assert_eq!(declared, 9, "eight of ours and the protocol's one");
        Game {
            world,
            table,
            tick: 0,
            held: 0,
            previous: 0,
        }
    }

    fn hold(&mut self, action: ActionId) -> &mut Self {
        self.held |= 1 << action.index();
        self
    }

    fn step(&mut self) {
        let ctx = TickCtx {
            tick: self.tick,
            tick_hz: 60,
            reserved: 0,
            input: InputFrame {
                buttons: self.held,
                axes: [0; gg_ecs::boundary::MAX_AXES],
            },
            previous: InputFrame {
                buttons: self.previous,
                axes: [0; gg_ecs::boundary::MAX_AXES],
            },
        };
        // SAFETY: the table is this binary's own, its entries live for the
        // process, and `ctx` outlives the call.
        unsafe { self.world.run_systems(&self.table, &ctx) }.expect("no system panicked");
        self.tick += 1;
        self.previous = self.held;
        self.held = 0;
    }

    fn steps(&mut self, n: u64) {
        for _ in 0..n {
            self.step();
        }
    }

    fn one<T: gg_ecs::Component + Copy>(&self) -> T {
        let query = Query::<&T>::new().unwrap();
        let mut found = None;
        self.world.each_ref(&query, |_, v: &T| found = Some(*v));
        found.expect("exactly one of this component")
    }

    fn count<T: gg_ecs::Component + Copy>(&self) -> usize {
        self.all::<T>().len()
    }

    fn all<T: gg_ecs::Component + Copy>(&self) -> Vec<T> {
        let query = Query::<&T>::new().unwrap();
        let mut out = Vec::new();
        self.world.each_ref(&query, |_, v: &T| out.push(*v));
        out
    }

    /// The widget on the entity that also carries `T`, found by walking the two
    /// queries in world order — which is what the host's own UI stage does
    /// (`gg_ui::boundary`'s two passes), so a test that agrees with it is
    /// testing the same correspondence the renderer relies on.
    fn widgets_with<T: gg_ecs::Component + Copy>(&self) -> Vec<(T, Widget)> {
        let marked = Query::<(&T, &Widget)>::new().unwrap();
        let mut out = Vec::new();
        self.world
            .each_ref(&marked, |_, (t, w): (&T, &Widget)| out.push((*t, *w)));
        out
    }
}

#[test]
fn the_shape_table_is_seven_tetrominoes() {
    for (kind, rotations) in SHAPES.iter().enumerate() {
        for (rot, mask) in rotations.iter().enumerate() {
            assert_eq!(
                mask.count_ones(),
                4,
                "piece {kind} rotation {rot} has {} cells, and a tetromino has four \
                 — a mask literal is one typo from three",
                mask.count_ones()
            );
            assert_eq!(cells_of(kind as u8, rot as u8).len(), 4);
        }
        // O is the one piece rotation does not move; every other piece must
        // actually change, or a rotation key would be dead for it.
        let distinct = rotations.iter().collect::<std::collections::BTreeSet<_>>();
        if kind == 1 {
            assert_eq!(distinct.len(), 1, "O rotates onto itself");
        } else {
            assert!(distinct.len() >= 2, "piece {kind} does not rotate");
        }
    }
    assert_eq!(SHAPES.len(), COLORS.len(), "a piece without a colour");
}

/// Widgets a settled board carries: the field, both bays, and everything that
/// is not a cell. Counted from the game's own tables rather than written down,
/// so adding a legend row does not fail a number nobody can place.
fn expected_widgets() -> usize {
    let chrome = 4 + 1; // four panels and the well's surround
    let text = demo_10_tetris::KEYS.len() * 2 + 7 + 3 + 3; // legend, captions, values, banner
    (HEIGHT - HIDDEN) * WIDTH + 32 + chrome + text
}

#[test]
fn one_tick_leaves_a_board_two_bays_a_hud_and_no_scene_at_all() {
    let mut game = Game::load();
    game.step();
    assert_eq!(game.count::<Play>(), 1);
    assert_eq!(game.count::<Well>(), 1);
    assert_eq!(
        game.count::<Cell>(),
        (HEIGHT - HIDDEN) * WIDTH,
        "one widget per visible cell"
    );
    assert_eq!(game.count::<Bay>(), 32, "two four-by-four boxes");
    assert_eq!(game.count::<HudLine>(), 3);
    assert_eq!(game.count::<Banner>(), 3);
    assert_eq!(game.count::<Widget>(), expected_widgets());
}

/// Re-entrant bootstrap: a reload runs it again against a populated world, and
/// what it finds must not double (§6 M5).
#[test]
fn bootstrap_run_again_spawns_nothing_new() {
    let mut game = Game::load();
    game.steps(8);
    assert_eq!(game.count::<Play>(), 1);
    assert_eq!(game.count::<Widget>(), expected_widgets());
}

/// The board is drawn, not just held. A cell whose colour never left the empty
/// one is a `present` that ran and did nothing, which every state assertion in
/// this file would still pass.
///
/// Soft-dropped clear of the ceiling first: a piece spawns in the two *hidden*
/// rows, so a freshly bootstrapped board legitimately draws none of it.
#[test]
fn the_falling_piece_and_its_ghost_are_on_the_board() {
    let mut game = Game::load();
    for _ in 0..12 {
        game.hold(demo_10_tetris::SOFT_DROP);
        game.step();
    }
    let piece = game.one::<Piece>();
    assert!(
        piece.row >= HIDDEN as i32,
        "the piece is still above the ceiling at row {}",
        piece.row
    );

    let cells = game.widgets_with::<Cell>();
    let painted = |want: u32| cells.iter().filter(|(_, w)| w.color == want).count();
    assert_eq!(
        painted(color_of(piece.kind + 1)),
        4,
        "the falling piece is not drawn in its colour"
    );
    assert_eq!(
        painted(color_of(demo_10_tetris::GHOST_CELL)),
        4,
        "the landing preview is not drawn"
    );

    // And the ghost is at the bottom, which is the only thing that makes it a
    // landing preview rather than a second piece.
    let lowest = |want: u32| {
        cells
            .iter()
            .filter(|(_, w)| w.color == want)
            .map(|(cell, _)| i32::from(cell.row))
            .max()
            .expect("no cell in that colour")
    };
    let well = game.one::<Well>();
    let depth = cells_of(piece.kind, piece.rot)
        .iter()
        .map(|(row, _)| *row)
        .max()
        .unwrap_or(0);
    assert!(lowest(color_of(demo_10_tetris::GHOST_CELL)) > lowest(color_of(piece.kind + 1)));
    assert_eq!(
        lowest(color_of(demo_10_tetris::GHOST_CELL)) + HIDDEN as i32,
        landing_row(&well, &piece) + depth,
        "the drawn ghost is not where the piece would land"
    );
}

/// Hold is the one control with nothing on screen until it is used, so an empty
/// hold bay is indistinguishable from a broken one without pressing it.
#[test]
fn the_hold_bay_fills_only_after_a_swap() {
    let mut game = Game::load();
    game.step();
    let filled = |game: &Game, slot: u8| {
        game.widgets_with::<Bay>()
            .iter()
            .filter(|(bay, w)| bay.slot == slot && w.color != color_of(0))
            .count()
    };
    assert_eq!(
        filled(&game, NEXT_BAY),
        4,
        "the next piece is not previewed"
    );
    assert_eq!(filled(&game, HOLD_BAY), 0, "a new game holds nothing");

    game.hold(HOLD);
    game.step();
    assert_ne!(game.one::<Play>().hold, NO_PIECE);
    assert_eq!(filled(&game, HOLD_BAY), 4, "the held piece is not shown");
}

/// The HUD is the score, and a score that never reaches the label is a HUD that
/// reads zero all game.
#[test]
fn the_score_reaches_the_hud() {
    let mut game = Game::load();
    game.hold(HARD_DROP);
    game.step();
    game.step();
    let score = game.one::<Play>().score;
    assert!(score > 0, "a hard drop scored nothing");
    let shown: Vec<String> = game
        .widgets_with::<HudLine>()
        .iter()
        .map(|(_, w)| w.text().to_owned())
        .collect();
    assert!(
        shown.contains(&score.to_string()),
        "the score {score} is not on screen: {shown:?}"
    );
}

/// A zero rect draws nothing (§4.9), which is how the banner hides — so "not
/// shown" has to be asserted as a zero rect rather than assumed.
#[test]
fn the_banner_is_hidden_until_the_game_is_over() {
    let mut game = Game::load();
    game.step();
    for (_, widget) in game.widgets_with::<Banner>() {
        assert_eq!(widget.rect, [0.0; 4], "GAME OVER showed on a live board");
    }

    let mut over = false;
    for tick in 0..2000 {
        if tick % 2 == 0 {
            game.hold(HARD_DROP);
        }
        game.step();
        if game.one::<Play>().over != 0 {
            over = true;
            break;
        }
    }
    assert!(over, "hard-dropping every piece never filled the well");
    game.step();
    for (banner, widget) in game.widgets_with::<Banner>() {
        assert_eq!(widget.rect, banner.rect, "the banner stayed hidden");
        assert!(widget.rect[2] > 0.0 && widget.rect[3] > 0.0);
    }
}

#[test]
fn gravity_moves_the_piece_down_on_its_own() {
    let mut game = Game::load();
    game.step();
    let start = game.one::<Piece>().row;
    let budget = gravity_for(&Rules::DEFAULT, 1);
    game.steps(u64::from(budget) + 1);
    assert!(
        game.one::<Piece>().row > start,
        "the piece did not fall in {budget} ticks"
    );
}

/// The claim the whole engine is built on, made about this game: the same input
/// twice is the same world, bit for bit — and the bag is inside that, which is
/// the part a generator living outside the world would break (§6 M18).
#[test]
fn the_same_inputs_produce_the_same_world_hash() {
    let script = |game: &mut Game| {
        for tick in 0..400u64 {
            match tick % 7 {
                0 => {
                    game.hold(LEFT);
                }
                2 => {
                    game.hold(ROTATE_CW);
                }
                4 => {
                    game.hold(HARD_DROP);
                }
                _ => {}
            }
            game.step();
        }
    };
    let (mut a, mut b) = (Game::load(), Game::load());
    script(&mut a);
    script(&mut b);
    assert_eq!(
        a.world.canonical_hash(),
        b.world.canonical_hash(),
        "two identical sessions diverged"
    );
    // And the session did something — an equality over two empty wells would
    // pass on a game that never ran.
    assert!(a.one::<Play>().score > 0 || a.one::<Well>().cells.iter().any(|r| r.contains(&1)));
}

/// A different seed is a different game. Without this the test above would pass
/// on a generator that always returned zero.
#[test]
fn a_different_seed_deals_a_different_bag() {
    let take = |seed| {
        let mut play = new_play(seed);
        (0..14).map(|_| draw(&mut play)).collect::<Vec<_>>()
    };
    assert_eq!(take(1), take(1));
    assert_ne!(take(1), take(2));
}

/// Seven draws are the seven pieces — that is what a bag *is*, and the property
/// a plain uniform draw would fail.
#[test]
fn the_bag_deals_each_piece_once_per_seven() {
    let mut play = new_play(0xfeed_beef);
    // Align first. `new_play` takes one draw for the next-piece, so the seven
    // draws after it *straddle* two bags — and a 7-bag promises each piece once
    // per aligned group, never once per sliding window of seven. Asserting the
    // window property is how one ends up "fixing" a correct randomiser.
    for _ in 0..6 {
        draw(&mut play);
    }
    for round in 0..40 {
        let mut seen = (0..7).map(|_| draw(&mut play)).collect::<Vec<_>>();
        seen.sort_unstable();
        assert_eq!(
            seen,
            vec![0, 1, 2, 3, 4, 5, 6],
            "round {round} was not a bag"
        );
    }
}

#[test]
fn a_full_row_clears_and_the_ones_above_it_fall() {
    let mut well = Well {
        cells: [[0; WIDTH]; HEIGHT],
    };
    // A full bottom row, and a single cell resting two rows above it.
    well.cells[HEIGHT - 1] = [1; WIDTH];
    well.cells[HEIGHT - 3][4] = 5;
    assert_eq!(clear_rows(&mut well), 1);
    assert!(
        well.cells[HEIGHT - 1].iter().all(|&c| c == 0),
        "the cleared row was not emptied"
    );
    assert_eq!(
        well.cells[HEIGHT - 2][4],
        5,
        "the cell above the cleared row did not fall exactly one"
    );
}

#[test]
fn four_full_rows_go_at_once() {
    let mut well = Well {
        cells: [[0; WIDTH]; HEIGHT],
    };
    for row in HEIGHT - 4..HEIGHT {
        well.cells[row] = [2; WIDTH];
    }
    assert_eq!(clear_rows(&mut well), 4);
    assert!(well.cells.iter().all(|r| r.iter().all(|&c| c == 0)));
}

#[test]
fn a_row_with_a_hole_in_it_stays() {
    let mut well = Well {
        cells: [[0; WIDTH]; HEIGHT],
    };
    well.cells[HEIGHT - 1] = [3; WIDTH];
    well.cells[HEIGHT - 1][7] = 0;
    assert_eq!(clear_rows(&mut well), 0);
    assert_eq!(well.cells[HEIGHT - 1][0], 3, "an unfull row was moved");
}

#[test]
fn the_walls_and_the_floor_stop_a_piece() {
    let well = Well {
        cells: [[0; WIDTH]; HEIGHT],
    };
    let mut piece = spawn_piece(0);
    piece.col = -4;
    assert!(collides(&well, &piece), "a piece walked off the left");
    piece.col = WIDTH as i32;
    assert!(collides(&well, &piece), "a piece walked off the right");
    piece.col = 3;
    piece.row = HEIGHT as i32;
    assert!(collides(&well, &piece), "a piece fell through the floor");
}

#[test]
fn a_locked_cell_stops_a_piece() {
    let mut well = Well {
        cells: [[0; WIDTH]; HEIGHT],
    };
    let piece = spawn_piece(0);
    assert!(!collides(&well, &piece), "an empty well rejected a spawn");
    for (row, col) in cells_of(piece.kind, piece.rot) {
        well.cells[(piece.row + row) as usize][(piece.col + col) as usize] = 1;
    }
    assert!(collides(&well, &piece), "a piece overlapped locked cells");
}

#[test]
fn gravity_gets_faster_with_the_level_and_then_stops() {
    let rules = Rules::DEFAULT;
    assert_eq!(gravity_for(&rules, 1), rules.gravity_ticks);
    assert!(gravity_for(&rules, 5) < gravity_for(&rules, 2));
    assert!(
        gravity_for(&rules, 200) >= MIN_GRAVITY_TICKS,
        "gravity fell through its floor and the game became unplayable"
    );
}

#[test]
fn a_new_game_holds_nothing_and_has_a_next_piece() {
    let play = new_play(9);
    assert_eq!(play.hold, NO_PIECE);
    assert_eq!(play.over, 0);
    assert_eq!(play.level, 1);
    assert!((play.next as usize) < SHAPES.len());
}

/// Restart is the one action a topped-out game listens to, and it must give
/// back a playable board rather than an empty one.
#[test]
fn restart_clears_the_well_and_deals_again() {
    let mut game = Game::load();
    game.step();
    // Hard-drop every other tick until the stack reaches the ceiling. Alternate,
    // because hard drop is an *edge*: a key held down is one drop, which is what
    // makes it a hard drop rather than a repeat. Left to gravity this would take
    // ~990 ticks a piece and the test would be measuring patience.
    let mut over = false;
    for tick in 0..2000 {
        if tick % 2 == 0 {
            game.hold(HARD_DROP);
        }
        game.step();
        if game.one::<Play>().over != 0 {
            over = true;
            break;
        }
    }
    assert!(over, "hard-dropping every piece never filled the well");
    game.hold(RESTART);
    game.step();
    let play = game.one::<Play>();
    assert_eq!(play.over, 0, "restart did not resume the game");
    assert_eq!(play.score, 0);
    assert!(
        game.one::<Well>()
            .cells
            .iter()
            .all(|r| r.iter().all(|&c| c == 0)),
        "restart left the old stack on the board"
    );
}
