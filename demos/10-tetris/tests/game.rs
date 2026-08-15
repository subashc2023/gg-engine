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
    BACK, Banner, Bay, Best, COLORS, CUE_CLEAR, CUE_HOLD, CUE_LEVEL, CUE_LOCK, CUE_MOVE, CUE_MUSIC,
    CUE_OVER, CUE_ROTATE, CUES, Cell, Cue, HARD_DROP, HEIGHT, HIDDEN, HOLD, HOLD_BAY, HudLine,
    LEFT, M_CURSOR, M_DONE, M_GHOST, M_QUIT, M_RESUME, M_THEME, M_TITLE_EXIT, M_TITLE_SETTINGS,
    M_VOLUME, MENU, MIN_GRAVITY_TICKS, NEXT_BAY, NO_PIECE, Options, PAUSE, PRESETS, Piece, Play,
    RESTART, ROTATE_CW, Rules, SCREEN_PAUSED, SCREEN_PLAYING, SCREEN_SETTINGS, SCREEN_TITLE, SEED,
    SHAPES, Screen, THEMES, WIDTH, Well, cells_of, clear_rows, collides, color_of, draw,
    gravity_for, landing_row, new_play, record_top, rotate_kicked, session, spawn_piece,
    themed_color, tspin_at_lock, voice_of,
};
use gg_ecs::boundary::{
    self, AbiInfo, ActionId, ComponentsTable, HostApiV1, InputFrame, Prefs, Sound, SystemsTable,
    TickCtx, Widget, asset_id, wave,
};
use gg_ecs::{Query, World};
use gg_math::sim;

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
        assert_eq!(declared, 17, "fourteen of ours and the protocol's three");
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

    /// Bootstrap and leave the title screen (§6 M19): one tick with `RESTART`
    /// down, which is PLAY's keyboard spelling there. The board afterwards is
    /// `bootstrap`'s own — entering play deals nothing new.
    fn start(&mut self) {
        self.hold(RESTART);
        self.step();
        assert_eq!(
            self.one::<demo_10_tetris::Screen>().at,
            demo_10_tetris::SCREEN_PLAYING,
            "RESTART on the title did not start the game"
        );
    }

    fn step(&mut self) {
        let ctx = TickCtx::detached(
            self.tick,
            60,
            InputFrame {
                buttons: self.held,
                axes: [0; gg_ecs::boundary::MAX_AXES],
            },
            InputFrame {
                buttons: self.previous,
                axes: [0; gg_ecs::boundary::MAX_AXES],
            },
        );
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

    /// Overwrite the state machine. Used by one test, to reseed the bag and
    /// nothing else — see `a_reseeded_bag_diverges_by_name_and_plays_a_different_game`.
    fn put_play(&mut self, play: Play) {
        let query = Query::<&mut Play>::new().unwrap();
        self.world.each(&query, |_, p: &mut Play| *p = play);
    }

    /// Play one *recorded* frame. Every other driver here names an action; the
    /// session names a bit pattern, because that is what a replay carries.
    fn frame(&mut self, frame: InputFrame) {
        self.held = frame.buttons;
        self.step();
    }

    /// Every cue's sequence, so a test can say what a tick played by diffing.
    fn cues(&self) -> [u32; CUES] {
        let mut out = [0; CUES];
        for (kind, seq) in out.iter_mut().enumerate() {
            *seq = self.cue(kind as u8).seq;
        }
        out
    }

    /// Click a menu widget: set the bit the host would have written after last
    /// tick's frame (§4.9's lag), run the tick that reads it, then clear it —
    /// the host rewrites `state` every tick and this harness has no host, so
    /// without the clear one click would repeat forever.
    fn click(&mut self, which: u8) {
        let id = MENU[which as usize].id;
        let query = Query::<&mut Widget>::new().unwrap();
        self.world.each(&query, |_, w: &mut Widget| {
            if w.id == id {
                w.state |= gg_ecs::boundary::state::CLICKED;
            }
        });
        self.step();
        self.world.each(&query, |_, w: &mut Widget| w.state = 0);
    }

    fn screen(&self) -> u32 {
        self.one::<Screen>().at
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
    let text = demo_10_tetris::LEGEND.len() * 2 + 8 + 4 + 3; // legend, captions, values, banner
    (HEIGHT - HIDDEN) * WIDTH + 32 + chrome + text + MENU.len()
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
    assert_eq!(game.count::<HudLine>(), 4, "score, lines, level, best");
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
    game.start();
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
    game.start();
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
    game.start();
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
    game.start();
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
    game.start();
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
        game.start();
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
    game.start();
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
    //
    // Every one but the music, which is silent on purpose: the component is the
    // theme's own state (§6 M43), so a cue bank spawned already looping would
    // start it over the title screen. That is asserted below rather than
    // excused here.
    for kind in 0..CUES as u8 {
        let voice = game.cue(kind);
        assert_eq!(
            voice,
            voice_of(kind),
            "cue {kind} is not the voice it declares"
        );
        if kind == CUE_MUSIC {
            assert_eq!(voice.wave, wave::SILENT, "the theme starts on the title");
            continue;
        }
        assert_ne!(voice.wave, wave::SILENT, "cue {kind} is silent");
        // A clip carries no length of its own here: `ms` 0 *is* the clip's own
        // length (§6 M43), so the audibility test is length-or-a-clip.
        assert!(
            voice.gain > 0.0 && (voice.ms > 0 || voice.clip != 0),
            "cue {kind} cannot be heard"
        );
    }
    // Three of them are baked clips and the rest are tones, which is the shape
    // §6 M43 landed on rather than a replacement: the cue that is *retuned per
    // event* stays synthesized, because a sample cannot be pitched.
    for kind in [CUE_LOCK, CUE_LEVEL, CUE_OVER] {
        assert_eq!(game.cue(kind).wave, wave::CLIP);
        assert_ne!(game.cue(kind).clip, 0, "cue {kind} names no clip");
    }
    assert_eq!(game.cue(CUE_CLEAR).wave, wave::SQUARE);
    assert_eq!(game.cue(CUE_CLEAR).clip, 0, "a tone named a clip");
}

/// The theme, on the four edges that move it (§6 M43).
///
/// A loop is a level rather than an event, so what is under test is that the
/// component *is* the state: a `CLIP_LOOP` means playing, silence means not,
/// and nothing anywhere has a second flag that could disagree with it.
#[test]
fn the_theme_follows_the_screen_and_stops_when_the_run_does() {
    let mut game = Game::load();
    game.step();
    assert_eq!(game.cue(CUE_MUSIC).wave, wave::SILENT, "on the title");

    game.start();
    let playing = game.cue(CUE_MUSIC);
    assert_eq!(playing.wave, wave::CLIP_LOOP, "starting a game was quiet");
    assert_eq!(playing.clip, asset_id("theme"));
    assert_ne!(playing.seq, 0, "the theme was set but never triggered");

    // A tick that changes nothing leaves it alone — the whole point of the
    // component being the state is that there is no edge to re-fire.
    let steady = game.cue(CUE_MUSIC).seq;
    game.steps(30);
    assert_eq!(game.cue(CUE_MUSIC).seq, steady, "the theme retriggered");

    game.hold(PAUSE);
    game.step();
    assert_eq!(game.cue(CUE_MUSIC).wave, wave::SILENT);
    // A tick with the key up between the two presses: the transition wants a
    // fresh edge, and holding PAUSE down for two ticks is one press.
    game.step();
    game.hold(PAUSE);
    game.step();
    assert_eq!(game.screen(), SCREEN_PLAYING);
    assert_eq!(
        game.cue(CUE_MUSIC).wave,
        wave::CLIP_LOOP,
        "resume was quiet"
    );

    // And the run ending, which changes no screen at all — the game-over
    // overlay is drawn on the playing screen, so this edge is `step`'s.
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
    assert_eq!(
        game.cue(CUE_MUSIC).wave,
        wave::SILENT,
        "the theme played on over a finished game"
    );
}

/// The ordinary moves, each on the tick it happened and not before.
#[test]
fn moving_rotating_and_holding_each_play_their_own_cue() {
    let mut game = Game::load();
    game.start();

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
    game.start();
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
    game.start();

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
        game.start();
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
    game.start();
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

// ---------------------------------------------------------- the recorded game

/// §6 M18's exit row, in process: **the recorded session is a whole game.**
///
/// The numbers are pinned rather than bounded because this is a *recording*.
/// "At least one line cleared" would stay green over a session that stopped
/// being the one the shell replays, which is the failure the whole gate exists
/// to catch — and the tick count is what makes a stale `.ggrp` visible from
/// here as well as from the push tier's byte compare.
#[test]
fn the_recorded_session_plays_a_whole_game() {
    let frames = session::frames();
    assert_eq!(frames.len(), 602, "the session changed length");

    let mut game = Game::load();
    let mut over_at = None;
    let mut levels = Vec::new();
    let mut level = 1;
    for (tick, frame) in frames.iter().enumerate() {
        game.frame(*frame);
        let play = game.one::<Play>();
        if play.level != level {
            levels.push(play.level);
            level = play.level;
        }
        if play.over != 0 && over_at.is_none() {
            over_at = Some(tick);
        }
    }

    let play = game.one::<Play>();
    assert_eq!(
        over_at,
        Some(570),
        "the game did not end where it was recorded"
    );
    assert_eq!(
        levels,
        vec![2, 3],
        "the session must cross a level boundary twice"
    );
    assert_eq!(
        (play.lines, play.level, play.score),
        (24, 3, 7982),
        "the recorded game came out differently"
    );
    // Every cue the game has, sounded at least once — a session that never
    // held, never rotated and never cleared would be a drop-and-die script.
    for (kind, seq) in game.cues().iter().enumerate() {
        assert_ne!(*seq, 0, "cue {kind} never played in a whole game");
    }
    // And the tail is quiet: a dead board that kept sounding would mean `over`
    // is not a state.
    assert_ne!(play.over, 0);
    let before = game.cues();
    game.steps(60);
    assert_eq!(played(before, game.cues()), Vec::<u8>::new());
}

/// §5.6b on whatever architecture this is running on. The aarch64-under-qemu
/// leg is this same test against this same file, which is the whole of "and on
/// aarch64" in §6 M18's exit row.
#[test]
fn the_recorded_session_reproduces_its_checked_in_hash_sequence() {
    let sequence = session::hash_sequence(&session::frames()).unwrap();
    let path = session::baseline_path();
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "no baseline at {} ({e}) — `cargo xtask replay --bless` writes it, and its absence \
             means this gate is checking nothing",
            path.display()
        )
    });
    let baseline = session::parse_baseline(&text).unwrap();

    if let Some(found) = session::divergence(&sequence, &baseline) {
        // Written *beside* the baseline, never over it: a gate that repairs
        // itself on failure is a rubber stamp (§6 M0A spike 3).
        let fresh = std::env::temp_dir().join("demo10-tetris.hashes.actual");
        let _ = std::fs::write(&fresh, session::encode_baseline(&sequence));
        panic!(
            "{found}\nfresh sequence written to {}\nIf this diff is intended, \
             `cargo xtask replay --bless` — and the diff belongs in the review.",
            fresh.display()
        );
    }
}

/// §5.6a: two runs on one runner. A single run compared against a file cannot
/// see a stable-but-wrong iteration order — it would match its own baseline
/// forever.
#[test]
fn one_recorded_session_run_twice_on_this_runner_agrees() {
    let frames = session::frames();
    let first = session::hash_sequence(&frames).unwrap();
    let second = session::hash_sequence(&frames).unwrap();
    assert_eq!(first, second, "two runs of one session disagreed");
}

/// The other half of §6 M18's exit row: **the bag is world state, and a replay
/// fails by name when it diverges.**
///
/// The only edit is `Play::rng`, made on the first tick and touching nothing
/// else — same board, same script, same everything a replay carries. Two
/// separate claims come out of it, and a bag living *beside* the world instead
/// of in it would fail a different one of them:
///
/// - **the generator is hashed state.** The divergence is named at tick 0, the
///   tick the new seed lands, because `sim::Rng` reaches the canonical hash
///   through an encoding written for it (§6 M18 item 1). A generator the world
///   did not hold would leave this tick identical and the gate silent.
/// - **and the generator decides the game.** Hashing a field nothing reads
///   would satisfy the first claim on its own, so the run is played out: a
///   different bag tops out at a different tick with a different score.
#[test]
fn a_reseeded_bag_diverges_by_name_and_plays_a_different_game() {
    let frames = session::frames();
    let baseline = session::hash_sequence(&frames)
        .unwrap()
        .iter()
        .enumerate()
        .map(|(tick, hash)| (tick as u64, hash.get()))
        .collect::<Vec<_>>();

    let mut game = Game::load();
    let mut sequence = Vec::new();
    let mut over_at = None;
    for (tick, frame) in frames.iter().enumerate() {
        game.frame(*frame);
        if tick == 0 {
            let play = Play {
                rng: sim::Rng::from_seed(SEED ^ 1),
                ..game.one::<Play>()
            };
            game.put_play(play);
        }
        sequence.push(game.world.canonical_hash());
        if game.one::<Play>().over != 0 && over_at.is_none() {
            over_at = Some(tick);
        }
    }

    let found = session::divergence(&sequence, &baseline)
        .expect("a reseeded bag hashed identically — the generator is not in the world");
    assert!(
        found.starts_with("tick 0:"),
        "the reseed should be visible on the tick it lands: {found}"
    );

    // The consequence, not the write. The bot's placements were recorded against
    // the *other* bag, so this run plays badly and dies early — which is the
    // point: the pieces it was handed were different ones.
    let play = game.one::<Play>();
    assert_ne!(
        (over_at, play.lines, play.score),
        (Some(570), 24, 7982),
        "a different bag played the same game"
    );
}

/// The gate stream (§6 M18): **the same bot, never giving up.**
///
/// `xtask reload --rules` swaps a rebuilt dylib under a *running* game, so it
/// needs a session that is still being played several thousand ticks in with a
/// stack on the board — the opposite of what `frames` is for. Pinned here as
/// numbers rather than bounds for `frames`'s reason: "it cleared some lines"
/// stays green over a script that has stopped being the one the shell replays.
///
/// The bot survives all four thousand of them, which is why `endless` mirrors a
/// restart it never performs here: the length is the caller's to choose, and a
/// generator that silently stopped playing at the top-out would hand a reload
/// gate a dead board and a swap that proved nothing.
#[test]
fn the_endless_session_keeps_playing_with_a_stack_on_the_board() {
    const TICKS: usize = 4000;
    let frames = session::endless(TICKS);
    assert_eq!(frames.len(), TICKS, "trimmed to exactly what was asked for");

    let progress = session::progress(&frames).unwrap();
    let last = progress[TICKS - 1];
    assert_eq!(
        (last.lines, last.level, last.score),
        (176, 18, 216_946),
        "the endless bot played a different game than the one this gate is pinned to"
    );
    assert!(
        progress.iter().all(|p| !p.over),
        "the bot topped out — playing on is the whole requirement"
    );

    // The board a swap would land on. Empty only in the six ticks between a
    // clear that took everything and the next lock, so a gate that asserts a
    // non-empty well is asserting something that is true nearly always and
    // checked exactly.
    // Two of the empty ticks are the title screen the stream opens on (§6 M19).
    let empty = progress.iter().filter(|p| p.occupied == 0).count();
    let deepest = progress.iter().map(|p| p.occupied).max().unwrap();
    assert_eq!((empty, deepest), (8, 82));
}

/// §6 M18's save row, from the game's side: **the record outlives the game that
/// set it.**
///
/// Its own component for exactly this reason — a restart replaces `Play`, and a
/// high score kept inside it would last one game. The number is also written
/// every tick rather than at the top-out, because a session closed mid-run is
/// still a score that was reached and `--save` does not wait for a game to end.
#[test]
fn the_best_score_survives_the_game_that_set_it() {
    let mut game = Game::load();
    game.start();
    assert_eq!(game.one::<Best>().score, 0, "a new world has no record");

    // Play until the well fills, the way `restart_clears_the_well_and_deals_again`
    // does: hard drop is an edge, so every other tick.
    let mut over = false;
    for tick in 0..2000 {
        if tick % 2 == 0 {
            game.hold(HARD_DROP);
        }
        game.step();
        assert_eq!(
            game.one::<Best>().score,
            game.one::<Play>().score.max(game.one::<Best>().score),
            "the record fell behind the run that was setting it"
        );
        if game.one::<Play>().over != 0 {
            over = true;
            break;
        }
    }
    assert!(over, "hard-dropping every piece never filled the well");

    let reached = game.one::<Play>().score;
    assert!(
        reached > 0,
        "a game that scored nothing proves nothing here"
    );
    game.hold(RESTART);
    game.step();
    assert_eq!(
        game.one::<Play>().score,
        0,
        "the new game kept the old score"
    );
    assert_eq!(
        game.one::<Best>().score,
        reached,
        "the record did not survive the restart"
    );
}

// ------------------------------------------------------------ the screens

/// The game opens on the title (§6 M19), plays nothing there, and starts on
/// `RESTART` — the whole reason every other test begins with `start()`.
#[test]
fn the_title_screen_holds_the_game_until_play() {
    let mut game = Game::load();
    game.step();
    assert_eq!(game.screen(), SCREEN_TITLE);
    let before = game.one::<Piece>();
    game.steps(120);
    assert_eq!(
        game.one::<Piece>().row,
        before.row,
        "gravity ran on the title screen"
    );
    assert_eq!(game.cues(), [0; CUES], "the title screen made a sound");

    game.hold(RESTART);
    game.step();
    assert_eq!(game.screen(), SCREEN_PLAYING);
    let budget = gravity_for(&Rules::DEFAULT, 1);
    game.steps(u64::from(budget) + 1);
    assert!(game.one::<Piece>().row > before.row, "PLAY did not play");
}

/// Pause freezes exactly the sim: the piece, the clocks and the cues — and
/// resumes where it stopped. The pause key toggles both ways.
#[test]
fn pause_freezes_the_game_and_resume_continues_it() {
    let mut game = Game::load();
    game.start();
    game.steps(10);
    let frozen = game.one::<Piece>();
    let quiet = game.cues();

    game.hold(PAUSE);
    game.step();
    assert_eq!(game.screen(), SCREEN_PAUSED);
    // Held keys and gravity do nothing while paused.
    for _ in 0..200 {
        game.hold(LEFT).hold(demo_10_tetris::SOFT_DROP);
        game.step();
    }
    assert_eq!(game.one::<Piece>().col, frozen.col, "a paused piece moved");
    assert_eq!(game.one::<Piece>().row, frozen.row, "a paused piece fell");
    // The theme is the one voice a pause *does* touch, and it stops (§6 M43) —
    // which is a `seq` bump like any other, so the gameplay cues are what the
    // silence is asserted over.
    assert_eq!(
        played(quiet, game.cues())
            .into_iter()
            .filter(|kind| *kind != CUE_MUSIC)
            .collect::<Vec<_>>(),
        Vec::<u8>::new()
    );
    assert_eq!(
        game.cue(CUE_MUSIC).wave,
        wave::SILENT,
        "the theme played on over a paused game"
    );

    game.hold(PAUSE);
    game.step();
    assert_eq!(game.screen(), SCREEN_PLAYING);
    let budget = gravity_for(&Rules::DEFAULT, 1);
    game.steps(u64::from(budget) + 1);
    assert!(
        game.one::<Piece>().row > frozen.row,
        "resume did not resume"
    );

    // The RESUME button is the click spelling of the same edge.
    game.hold(PAUSE);
    game.step();
    assert_eq!(game.screen(), SCREEN_PAUSED);
    game.click(M_RESUME);
    assert_eq!(game.screen(), SCREEN_PLAYING);
}

/// QUIT files the abandoned score in the title table and deals a fresh board,
/// so PLAY from the title is always a new game.
#[test]
fn quit_records_the_run_and_returns_a_fresh_board_to_the_title() {
    let mut game = Game::load();
    game.start();
    for _ in 0..6 {
        game.hold(HARD_DROP);
        game.step();
        game.step();
    }
    let abandoned = game.one::<Play>().score;
    assert!(abandoned > 0, "six hard drops scored nothing");

    game.hold(PAUSE);
    game.step();
    game.click(M_QUIT);
    assert_eq!(game.screen(), SCREEN_TITLE);
    let best = game.one::<Best>();
    assert_eq!(best.top[0], abandoned, "walking away lost the score");
    assert_eq!(game.one::<Play>().score, 0, "QUIT kept the old run");
    assert!(
        game.one::<Well>()
            .cells
            .iter()
            .all(|r| r.iter().all(|&c| c == 0)),
        "QUIT left the old stack on the board"
    );
}

/// The settings screen: reachable from title and pause, DONE returns to where
/// it came from, and every row edits the value it names — through the real
/// systems table, with the click arriving as the host writes it (§4.9).
#[test]
fn the_settings_rows_edit_what_they_name_and_done_returns() {
    let mut game = Game::load();
    game.step();
    game.click(M_TITLE_SETTINGS);
    assert_eq!(game.screen(), SCREEN_SETTINGS);

    game.click(M_GHOST);
    assert_eq!(game.one::<Options>().ghost_off, 1, "the ghost row is dead");
    game.click(M_THEME);
    assert_eq!(game.one::<Options>().theme, 1, "the theme row is dead");
    game.click(M_VOLUME);
    let prefs = game.one::<gg_ecs::boundary::Prefs>();
    assert_eq!(
        prefs.quiet,
        gg_ecs::boundary::QUIET_MAX / 4,
        "the volume row is dead"
    );
    game.click(M_CURSOR);
    assert!(
        game.one::<gg_ecs::boundary::Prefs>().hardware_cursor(),
        "the cursor row is dead"
    );
    game.click(demo_10_tetris::M_SPEED);
    let rules = game.one::<Rules>();
    assert_eq!(
        bytemuck::bytes_of(&rules),
        bytemuck::bytes_of(&PRESETS[1].1),
        "the speed row did not write the FAST preset into Rules"
    );
    assert_eq!(game.one::<Options>().preset, 1);

    game.click(M_DONE);
    assert_eq!(
        game.screen(),
        SCREEN_TITLE,
        "DONE forgot where it came from"
    );

    // And from pause, DONE goes back to pause — not to the title.
    game.hold(RESTART);
    game.step();
    game.hold(PAUSE);
    game.step();
    game.click(demo_10_tetris::M_PAUSE_SETTINGS);
    assert_eq!(game.screen(), SCREEN_SETTINGS);
    game.click(M_DONE);
    assert_eq!(game.screen(), SCREEN_PAUSED);
}

/// A hidden ghost is a present-time choice, not a sim one: the hash-relevant
/// state is the `Options` component, and the board draws no `GHOST_CELL`.
#[test]
fn turning_the_ghost_off_takes_it_off_the_board() {
    let mut game = Game::load();
    game.start();
    game.steps(4);
    let ghosts = |game: &Game| {
        game.widgets_with::<Cell>()
            .iter()
            .filter(|(_, w)| w.color == color_of(demo_10_tetris::GHOST_CELL))
            .count()
    };
    assert_eq!(ghosts(&game), 4, "no ghost to switch off");

    game.world.each(
        &Query::<&mut Options>::new().unwrap(),
        |_, o: &mut Options| {
            o.ghost_off = 1;
        },
    );
    game.step();
    assert_eq!(ghosts(&game), 0, "the ghost survived its own setting");
}

/// A theme recolours the pieces and only the pieces — empty cells and the
/// ghost stay the well's own, so the board keeps reading as a board.
#[test]
fn a_theme_recolours_the_locked_stack() {
    let mut game = Game::load();
    game.start();
    game.hold(HARD_DROP);
    game.step();
    game.step();
    let kind = {
        let well = game.one::<Well>();
        well.cells
            .iter()
            .flatten()
            .copied()
            .find(|c| *c != 0)
            .expect("a hard drop locked nothing")
    };
    let classic = themed_color(0, kind);
    game.world.each(
        &Query::<&mut Options>::new().unwrap(),
        |_, o: &mut Options| {
            o.theme = 1;
        },
    );
    game.step();
    let painted = game
        .widgets_with::<Cell>()
        .iter()
        .filter(|(_, w)| w.color == themed_color(1, kind))
        .count();
    assert!(painted >= 1, "the neon theme never reached the board");
    assert_ne!(themed_color(1, kind), classic, "theme 1 is theme 0");
}

// ------------------------------------------------------- SRS and the T-spin

/// The kicks are SRS's published tables, checked at the wall: a vertical I
/// flush against the left wall rotates flat by kicking two columns right —
/// table entry two — rather than being refused where the old five-offset list
/// would have slid it one.
#[test]
fn srs_kicks_a_wall_bound_i_the_published_distance() {
    let well = Well {
        cells: [[0; WIDTH]; HEIGHT],
    };
    // I in rotation R occupies column 2 of its box; col -2 puts it on the wall.
    let mut piece = Piece {
        col: -2,
        row: 6,
        kind: 0,
        rot: 1,
        pad: [0; 2],
    };
    let kick = rotate_kicked(&well, &mut piece, false);
    assert_eq!(
        kick,
        Some(1),
        "the wall kick was not the table's second test"
    );
    assert_eq!(
        (piece.rot, piece.col),
        (0, 0),
        "the I did not land flat at the wall"
    );

    // Open air is always the first test, and the piece does not move.
    let mut free = spawn_piece(2);
    free.row = 8;
    assert_eq!(rotate_kicked(&well, &mut free, true), Some(0));
    assert_eq!((free.col, free.row), (3, 8));
}

/// The three-corner rule, all four verdicts: no spin without a rotation, no
/// spin under three corners, mini when the point-side corners are open, full
/// when they are filled.
#[test]
fn the_three_corner_rule_tells_full_from_mini_from_nothing() {
    let mut well = Well {
        cells: [[0; WIDTH]; HEIGHT],
    };
    // A T pointing down in a pocket at the floor: box corners at (row, 0/2)
    // and (row+2, 0/2), the bottom pair braced by the floor's neighbours.
    let piece = Piece {
        col: 0,
        row: HEIGHT as i32 - 3,
        kind: 2,
        rot: 2,
        pad: [0; 2],
    };
    well.cells[HEIGHT - 1][0] = 1;
    well.cells[HEIGHT - 1][2] = 1;
    assert_eq!(tspin_at_lock(&well, &piece, 1), 0, "two corners spun");

    well.cells[HEIGHT - 3][0] = 1;
    assert_eq!(tspin_at_lock(&well, &piece, 0), 0, "no rotation, no spin");
    assert_eq!(
        tspin_at_lock(&well, &piece, 1),
        2,
        "both point-side corners are filled: this is the full spin"
    );

    // Point up over the same pocket: the front corners are the open top pair,
    // so three corners make it a mini — unless the last kick was the table's
    // fifth, which upgrades it.
    let up = Piece { rot: 0, ..piece };
    assert_eq!(
        tspin_at_lock(&well, &up, 1),
        1,
        "the open-front spin is mini"
    );
    assert_eq!(tspin_at_lock(&well, &up, 2), 2, "the fifth kick upgrades");
}

/// A T-spin single through the real systems: the pocket is built, the spin
/// state is planted, and the lock scores `SPIN_SCORE[1] * level` — none of
/// which a plain single is worth.
#[test]
fn a_tspin_single_scores_the_spin_table() {
    let mut game = Game::load();
    game.start();
    let mut well = game.one::<Well>();
    // Bottom row full but for the T's stem column; the row above holds the
    // bracing corner.
    for (col, cell) in well.cells[HEIGHT - 1].iter_mut().enumerate() {
        *cell = u8::from(col != 1);
    }
    well.cells[HEIGHT - 3][0] = 1;
    game.put_well(well);
    game.put_piece(Piece {
        col: 0,
        row: HEIGHT as i32 - 3,
        kind: 2,
        rot: 2,
        pad: [0; 2],
    });
    let mut play = game.one::<Play>();
    play.last_rot = 1;
    play.score = 0;
    game.put_play(play);

    game.hold(HARD_DROP);
    game.step();
    let play = game.one::<Play>();
    assert_eq!(play.lines, 1, "the T-spin single cleared no row");
    assert_eq!(
        play.score,
        demo_10_tetris::SPIN_SCORE[1],
        "a full T-spin single is the spin table at level 1"
    );
    assert_eq!(play.b2b, 1, "a T-spin clear starts the back-to-back chain");
}

/// Back-to-back and combo, in one sitting: two tetrises on consecutive locks —
/// the second is worth `* 3 / 2`, plus one combo link.
#[test]
fn consecutive_tetrises_pay_back_to_back_and_the_combo_link() {
    let mut game = Game::load();
    game.start();
    let drop_tetris = |game: &mut Game| {
        // A released-keys tick first: hard drop is an edge, and the previous
        // drop's press is still the previous frame.
        game.step();
        let before = game.one::<Play>().score;
        single_clear(game, 4);
        game.one::<Play>().score - before
    };
    let first = drop_tetris(&mut game);
    assert!(
        first > demo_10_tetris::LINE_SCORE[3],
        "a tetris at level 1 is the line table plus the drop distance, got {first}"
    );
    assert_eq!(game.one::<Play>().b2b, 1, "a tetris starts the chain");
    let second = drop_tetris(&mut game);
    // The two drops fall the same distance, so the distance points cancel and
    // the difference is exactly the back-to-back half and the combo link.
    assert_eq!(
        second - first,
        demo_10_tetris::LINE_SCORE[3] * 3 / 2 - demo_10_tetris::LINE_SCORE[3]
            + demo_10_tetris::COMBO_SCORE,
        "the second consecutive tetris pays back-to-back and one combo link"
    );
}

/// The scoreboard files finished games in order and ignores the empty ones.
#[test]
fn the_title_table_keeps_the_five_best_descending() {
    let mut best = Best {
        score: 0,
        top: [0; 5],
    };
    for score in [300, 100, 500, 200, 400, 250, 600] {
        record_top(&mut best, score);
    }
    assert_eq!(best.top, [600, 500, 400, 300, 250]);
    record_top(&mut best, 0);
    assert_eq!(best.top, [600, 500, 400, 300, 250], "zero is not a game");
}

/// Theme 0 is the classic palette the golden reference pins, every palette is
/// opaque, and the non-piece cells are never themed.
#[test]
fn theme_zero_is_classic_and_every_palette_is_opaque() {
    assert_eq!(THEMES[0], COLORS);
    assert_eq!(THEMES.len(), demo_10_tetris::THEME_NAMES.len());
    for theme in THEMES {
        for color in theme {
            assert_eq!(color >> 24, 0xff, "{color:#010x} is not opaque");
        }
    }
    for theme in 0..THEMES.len() as u32 {
        assert_eq!(themed_color(theme, 0), color_of(0));
        assert_eq!(
            themed_color(theme, demo_10_tetris::GHOST_CELL),
            color_of(demo_10_tetris::GHOST_CELL)
        );
    }
}

/// The first preset is the default rules — "reset to NORMAL" must not drift.
#[test]
fn the_first_preset_is_the_default_rules() {
    assert_eq!(PRESETS[0].0, "NORMAL");
    assert_eq!(
        bytemuck::bytes_of(&PRESETS[0].1),
        bytemuck::bytes_of(&Rules::DEFAULT)
    );
}

/// The menu table is in index order with distinct widget ids — `MenuItem`
/// indexes it, and a swapped pair would wire RESUME to the QUIT handler.
#[test]
fn the_menu_table_is_in_index_order_with_distinct_ids() {
    for (index, def) in MENU.iter().enumerate() {
        assert_eq!(def.which as usize, index, "MENU[{index}] names another row");
    }
    let mut ids: Vec<u64> = MENU.iter().map(|d| d.id).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), MENU.len(), "two menu widgets share an id");
}

/// The record is on screen, and it is its own row.
#[test]
fn the_best_score_reaches_the_hud() {
    let mut game = Game::load();
    game.start();
    game.world
        .each(&Query::<&mut Best>::new().unwrap(), |_, best: &mut Best| {
            best.score = 204_900;
        });
    game.step();
    let shown: Vec<String> = game
        .widgets_with::<HudLine>()
        .into_iter()
        .map(|(_, widget)| widget.text().to_owned())
        .collect();
    assert_eq!(shown.len(), 4);
    assert_eq!(shown[3], "204900", "the fourth stats row is the record");
    assert_ne!(shown[0], shown[3], "the score row is not the record row");
}

/// Escape is the game's, not the shell's (§6 M44). Unclaimed it exits the
/// process without a tick running, which is how a shipped game loses whatever
/// the session was about to write; claimed, it is one verb that means "back" on
/// every screen and "leave" on the one screen with nowhere further back.
#[test]
fn escape_backs_out_of_every_screen_and_leaves_from_the_title() {
    let mut game = Game::load();
    game.step();
    assert_eq!(game.screen(), SCREEN_TITLE);

    // Title to settings and back out again.
    game.click(M_TITLE_SETTINGS);
    assert_eq!(game.screen(), SCREEN_SETTINGS);
    game.hold(BACK);
    game.step();
    assert_eq!(game.screen(), SCREEN_TITLE, "escape did not leave settings");
    assert_eq!(
        game.one::<Prefs>().close,
        0,
        "backing out of a screen closed the game"
    );

    // Playing to paused and back, the pause key's other spelling. A press has
    // to end before the next begins, which is what the bare step between them is.
    game.start();
    game.steps(5);
    game.hold(BACK);
    game.step();
    assert_eq!(game.screen(), SCREEN_PAUSED);
    game.step();
    game.hold(BACK);
    game.step();
    assert_eq!(game.screen(), SCREEN_PLAYING);
}

/// The two edges that end a session, and the one that does not. `Prefs::close`
/// is monotone and hashed, so the tick it is written on is the tick a replay
/// ends on and the world the shell saves on the way out (§6 M44).
#[test]
fn only_the_title_screens_exit_closes_the_game_and_it_leaves_a_clean_board() {
    let mut game = Game::load();
    game.start();
    game.steps(40);
    game.hold(PAUSE);
    game.step();
    assert_eq!(game.screen(), SCREEN_PAUSED);

    // QUIT TO TITLE files the run and deals a fresh board. It does not close.
    game.click(M_QUIT);
    assert_eq!(game.screen(), SCREEN_TITLE);
    assert_eq!(
        game.one::<Prefs>().close,
        0,
        "quitting to the title closed it"
    );
    assert_eq!(game.one::<Play>().score, 0, "the board was not dealt again");

    // EXIT does, and the board it leaves behind is the one a resumed session
    // opens on — which is the whole reason the reset above happens first.
    game.click(M_TITLE_EXIT);
    assert_ne!(game.one::<Prefs>().close, 0, "EXIT did not end the session");
    assert_eq!(game.screen(), SCREEN_TITLE);
    assert_eq!(
        game.one::<Play>().over,
        0,
        "a half-played board crossed the exit"
    );

    // And escape from the title is the same edge by another name.
    let mut second = Game::load();
    second.step();
    second.hold(BACK);
    second.step();
    assert_ne!(
        second.one::<Prefs>().close,
        0,
        "escape did not end the session"
    );
}

/// §6 M51: **a session that does not end**, weighed rather than watched.
///
/// Every other gate over this game is bounded by a script that finishes — the
/// blessed session is 602 ticks and the longest reload leg is 8,000. A player
/// plays for twenty minutes, and nothing had ever run long enough for the
/// answer to mean anything. This is eleven minutes of continuous play, and its
/// length is not a round number: the defect it was written for went off at
/// 35,624, so a shorter run would have reported a perfect result.
///
/// **Two halves that fail in opposite directions**, which is the whole reason
/// [`Footprint`](demo_10_tetris::session::Footprint) is one struct. Growth
/// catches a session that accumulates. Liveness catches the failure growth
/// cannot see and cannot even be suspicious of — a game that stopped being
/// played reports the flattest numbers a growth column can print, and that is
/// exactly what `session::endless` did before M51: the mirror positioned every
/// piece at its spawn row while the game's piece had already fallen, the two
/// boards parted once the stack rose into that distance, the game topped out on
/// a board the script did not have, and the script tapped into a dead world for
/// the remaining hundred thousand ticks.
#[test]
fn a_session_that_does_not_end_neither_grows_nor_stops() {
    // Eleven minutes at 60 Hz, past where the mirror used to part from the game.
    const TICKS: usize = 40_000;
    const STRIDE: usize = 4_000;

    let frames = session::endless(TICKS);
    assert_eq!(frames.len(), TICKS);
    let samples = session::footprint(&frames, STRIDE).unwrap();
    assert_eq!(samples.len(), TICKS / STRIDE, "one sample per window");

    // Growth. Not a budget: the numbers are *equal* to the first window's,
    // because a Tetris world is a fixed cast — a well, a play, a piece, the
    // widgets and the cues — and nothing in it is allocated by playing longer.
    // An equality is the strongest thing true here, so it is what is asserted;
    // a tolerance would forgive the first entity that started leaking.
    let first = samples[0];
    for sample in &samples {
        assert_eq!(
            (sample.entities, sample.bytes, sample.side_tables),
            (first.entities, first.bytes, first.side_tables),
            "the world grew by tick {}: {} entities and {} bytes against {} and {}",
            sample.tick,
            sample.entities,
            sample.bytes,
            first.entities,
            first.bytes
        );
    }

    // Liveness. Nearly every tick of a played game changes the world — a piece
    // falls, a clock advances — so the bar is deliberately far from the
    // threshold: what it refuses is a *frozen* world, not a slow one.
    for sample in &samples {
        assert!(
            sample.moved as usize > STRIDE * 9 / 10,
            "only {} of {STRIDE} ticks changed anything by tick {} — the game stopped being \
             played, which is the failure a growth column reports as a perfect result",
            sample.moved,
            sample.tick
        );
    }

    // And it was played *well*, which is a different claim from moving: a bot
    // wedged against a wall moves a fall clock every tick and clears nothing.
    let last = samples[samples.len() - 1];
    assert!(
        last.cleared > 1_000,
        "{} rows in {TICKS} ticks — the bot is moving without playing",
        last.cleared
    );
}

/// The regime the bot now refuses to leave, stated as a number (§6 M51).
///
/// `endless` restarts on a top-out, and before M51 that path had never once
/// executed: every gate that drives it is short enough that the bot never dies.
/// So the reseed it mirrors — `Play::rng`'s own next draw, which has to match
/// `reset_game`'s — was unexercised code, and a mismatch would have gone
/// unnoticed while producing a bot playing a game the world was not.
///
/// What grades it is *how long the game stays over*: the script taps RESTART on
/// the tick after the top-out, so a game that comes back reads two ticks and a
/// game that does not reads the rest of the session.
#[test]
fn every_game_the_endless_bot_loses_is_one_it_starts_again() {
    const TICKS: usize = 40_000;
    let progress = session::progress(&session::endless(TICKS)).unwrap();

    let (mut games, mut run, mut worst) = (0u32, 0u32, 0u32);
    for tick in &progress {
        if tick.over {
            run += 1;
            games += u32::from(run == 1);
            worst = worst.max(run);
        } else {
            run = 0;
        }
    }

    assert!(
        games > 0,
        "the bot never lost a game, so nothing restarted one"
    );
    assert_eq!(
        worst, 2,
        "a game stayed over for {worst} ticks — RESTART lands the tick after the top-out, so \
         anything longer is a restart the game refused and a mirror the script is now flying \
         blind against"
    );
}
