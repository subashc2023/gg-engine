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
    Banner, Bay, COLORS, CUE_CLEAR, CUE_HOLD, CUE_LEVEL, CUE_LOCK, CUE_MOVE, CUE_OVER, CUE_ROTATE,
    CUES, Cell, Cue, HARD_DROP, HEIGHT, HIDDEN, HOLD, HOLD_BAY, HudLine, LEFT, MIN_GRAVITY_TICKS,
    NEXT_BAY, NO_PIECE, Piece, Play, RESTART, ROTATE_CW, Rules, SHAPES, WIDTH, Well, cells_of,
    clear_rows, collides, color_of, draw, gravity_for, landing_row, new_play, spawn_piece,
    voice_of,
};
use gg_ecs::boundary::{
    self, AbiInfo, ActionId, ComponentsTable, HostApiV1, InputFrame, Sound, SystemsTable, TickCtx,
    Widget, wave,
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
        assert_eq!(declared, 11, "nine of ours and the protocol's two");
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

    /// The `Sound` on the cue entity for `kind`, as the host's observer finds
    /// it (`gg_audio::Audio::tick`'s own walk). Reading the *component* rather
    /// than a device is the point: every claim below about what this game
    /// sounds like is checked on a machine with no sound card, in the fast tier
    /// (§1.5's audio law, §6 M18 item 2).
    fn cue(&self, kind: u8) -> Sound {
        let query = Query::<(&Cue, &Sound)>::new().unwrap();
        let mut found = None;
        self.world.each_ref(&query, |_, (c, s): (&Cue, &Sound)| {
            if c.kind == kind {
                found = Some(*s);
            }
        });
        found.expect("every cue kind has an entity")
    }

    /// Overwrite the board. The only way to ask for a *four*-row clear without
    /// playing well enough to earn one; every other test drives through input.
    fn put_well(&mut self, well: Well) {
        let query = Query::<&mut Well>::new().unwrap();
        self.world.each(&query, |_, w: &mut Well| *w = well);
    }

    /// Place the falling piece. As [`put_well`](Self::put_well): asking for a
    /// specific clear means choosing the piece, and the bag will not be told.
    fn put_piece(&mut self, piece: Piece) {
        let query = Query::<&mut Piece>::new().unwrap();
        self.world.each(&query, |_, p: &mut Piece| *p = piece);
    }

    /// Every cue's sequence, so a test can say what a tick played by diffing.
    fn cues(&self) -> [u32; CUES] {
        let mut out = [0; CUES];
        for (kind, seq) in out.iter_mut().enumerate() {
            *seq = self.cue(kind as u8).seq;
        }
        out
    }
}

/// The cue kinds whose sequence moved between `before` and `after`.
fn played(before: [u32; CUES], after: [u32; CUES]) -> Vec<u8> {
    (0..CUES)
        .filter(|k| before[*k] != after[*k])
        .map(|k| k as u8)
        .collect()
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

// ------------------------------------------------------------------- the cues
//
// Every claim below is about `Sound` components in the world, which is the whole
// of what this game says about audio (§6 M18 item 2). No device is opened, no
// tier makes a noise, and the assertions still cover the thing that actually
// breaks: a cue wired to the wrong event, or to none.

/// A cue bank exists, and it is silent until something happens. The second half
/// is the one worth gating — a bank that fired on spawn would make every load
/// and every reload a chord.
#[test]
fn a_new_game_declares_a_cue_bank_that_has_not_played() {
    let game = Game::load();
    assert_eq!(
        game.count::<Cue>(),
        0,
        "the bank exists before the first tick"
    );

    let mut game = Game::load();
    game.step();
    assert_eq!(game.count::<Cue>(), CUES);
    assert_eq!(game.cues(), [0; CUES], "a fresh board played something");
    // And every one of them is a voice that would make a sound if triggered —
    // a cue wired to `wave::SILENT` is a cue that is quietly not there.
    for kind in 0..CUES as u8 {
        let voice = game.cue(kind);
        assert_ne!(voice.wave, wave::SILENT, "cue {kind} is silent");
        assert!(
            voice.gain > 0.0 && voice.ms > 0,
            "cue {kind} cannot be heard"
        );
        assert_eq!(
            voice,
            voice_of(kind),
            "cue {kind} is not the voice it declares"
        );
    }
}

/// The ordinary moves, each on the tick it happened and not before.
#[test]
fn moving_rotating_and_holding_each_play_their_own_cue() {
    let mut game = Game::load();
    game.step();

    let before = game.cues();
    game.hold(LEFT);
    game.step();
    assert_eq!(played(before, game.cues()), vec![CUE_MOVE]);

    let before = game.cues();
    game.hold(ROTATE_CW);
    game.step();
    assert_eq!(played(before, game.cues()), vec![CUE_ROTATE]);

    let before = game.cues();
    game.hold(HOLD);
    game.step();
    assert_eq!(played(before, game.cues()), vec![CUE_HOLD]);

    // A tick where nothing happened plays nothing. The trigger is the event,
    // never the passage of time — which is what makes a paused game quiet.
    let before = game.cues();
    game.step();
    assert_eq!(played(before, game.cues()), Vec::<u8>::new());
}

/// A rotation the walls refuse is silent. "Nothing happened" is the feedback,
/// and a click there would say the opposite of what the board shows.
#[test]
fn a_refused_rotation_makes_no_sound() {
    let mut game = Game::load();
    game.step();
    // Wedge the piece against the left wall, then try to turn into it. Some
    // pieces still fit, so this asserts the correspondence rather than a
    // refusal: rotated or not, the cue and the piece agree.
    for _ in 0..8 {
        game.hold(LEFT);
        game.step();
    }
    let before_piece = game.one::<Piece>();
    let before = game.cues();
    game.hold(ROTATE_CW);
    game.step();
    let turned = game.one::<Piece>().rot != before_piece.rot;
    let sounded = played(before, game.cues()).contains(&CUE_ROTATE);
    assert_eq!(turned, sounded, "the rotate cue and the piece disagree");
}

/// A lock plays the lock cue; a lock that cleared rows plays the clear cue too,
/// retuned by how many went. The retune is the claim: a tetris and a single are
/// audibly different events rather than the same blip four times.
#[test]
fn a_lock_and_a_clear_are_separate_cues_and_the_clear_is_tuned_by_its_rows() {
    let mut game = Game::load();
    game.step();

    // Hard-drop until something locks. The first drop always does.
    let before = game.cues();
    game.hold(HARD_DROP);
    game.step();
    assert!(
        played(before, game.cues()).contains(&CUE_LOCK),
        "a hard drop that locked did not play the lock cue"
    );

    // Fill the floor but for one column, then drop into it: one row, then four.
    for rows in [1u32, 4] {
        let mut game = Game::load();
        game.step();
        let one = game.cue(CUE_CLEAR);
        let four = single_clear(&mut game, rows);
        assert!(
            four.seq > one.seq,
            "clearing {rows} row(s) played no clear cue"
        );
        assert_eq!(
            four.hz_to,
            440.0 * (1.0 + rows as f32 * 0.5),
            "the clear cue was not tuned to {rows} row(s)"
        );
        assert!(four.ms >= 70 + 50 * rows);
        // Ten lines to a level, so neither of these crossed one. A level cue
        // that fired on every clear would be the same sound as the clear.
        assert_eq!(
            game.cue(CUE_LEVEL).seq,
            0,
            "{rows} row(s) announced a level"
        );
    }
}

/// Clear exactly `rows` rows at once, and return the cue that played.
///
/// The board and the piece are both written: a four-row clear is not something
/// a test can reach by holding a key, and hard-dropping whatever the bag deals
/// stacks in the middle and tops out long before it fills a column. So the well
/// gets `rows` full rows with the last column open, and the piece becomes a
/// vertical I over that column — which is the only tetromino that fits a
/// one-wide well four deep, and the reason a tetris is called one.
fn single_clear(game: &mut Game, rows: u32) -> Sound {
    let mut well = game.one::<Well>();
    for row in 0..rows as usize {
        for (col, cell) in well.cells[HEIGHT - 1 - row].iter_mut().enumerate() {
            *cell = u8::from(col != WIDTH - 1);
        }
    }
    game.put_well(well);

    // Found rather than written down: a shape table edit that renumbered the
    // pieces would silently make this test drop something else.
    let (kind, rot) = vertical_i();
    let mut piece = spawn_piece(kind);
    piece.rot = rot;
    piece.row = 0;
    piece.col = (WIDTH - 1) as i32 - cells_of(kind, rot)[0].1;
    game.put_piece(piece);

    let before = game.cue(CUE_CLEAR);
    game.hold(HARD_DROP);
    game.step();
    let now = game.cue(CUE_CLEAR);
    assert_ne!(
        now.seq, before.seq,
        "dropping the I did not clear {rows} row(s)"
    );
    now
}

/// The `(kind, rotation)` whose four cells stand in one column — the vertical I.
fn vertical_i() -> (u8, u8) {
    for kind in 0..SHAPES.len() as u8 {
        for rot in 0..4u8 {
            let cells = cells_of(kind, rot);
            let columns: std::collections::BTreeSet<_> = cells.iter().map(|c| c.1).collect();
            let rows: std::collections::BTreeSet<_> = cells.iter().map(|c| c.0).collect();
            if columns.len() == 1 && rows.len() == 4 {
                return (kind, rot);
            }
        }
    }
    panic!("no vertical I in the shape table");
}

/// Topping out plays the one long note in the game, and plays it once.
#[test]
fn a_top_out_plays_its_cue_exactly_once() {
    let mut game = Game::load();
    game.step();
    for tick in 0..2_000 {
        if tick % 2 == 0 {
            game.hold(HARD_DROP);
        }
        game.step();
        if game.one::<Play>().over != 0 {
            break;
        }
    }
    assert_ne!(game.one::<Play>().over, 0, "the well never filled");
    let at_death = game.cue(CUE_OVER);
    assert_ne!(at_death.seq, 0, "topping out was silent");

    // A finished game keeps ticking and stays quiet — `over` is a state, and a
    // cue that re-fired every tick after it would be a siren.
    let before = game.cues();
    game.steps(60);
    assert_eq!(played(before, game.cues()), Vec::<u8>::new());
}
