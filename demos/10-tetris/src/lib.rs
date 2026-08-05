//! **Demo 10 — Tetris** (§6 M18): the first artifact in this tree whose
//! requirements were not ours to choose.
//!
//! Every rule here is a *field of a component*, not a constant: [`Rules`] holds
//! gravity, DAS, ARR, lock delay and soft-drop rate, so the M15 inspector can
//! change how the game plays while it is being played, and a reload keeps the
//! stack that was on the board. That is the loop this engine was built for
//! (§1), pointed at something worth changing.
//!
//! # The whole game is UI
//!
//! Nothing here declares [`Renderable`](gg_ecs::boundary::Renderable), an eye or
//! a light. The board, both bays, the HUD and the frame are [`Widget`]
//! rectangles in [`CANVAS`] units (§4.9), which is the protocol that already has
//! the properties a 2D game needs: no projection, no camera, exact authored
//! colours — the UI pass draws to the backbuffer *after* the tonemap — and a
//! layout that is the same picture at every window size. The first draft drew
//! the well as cubes under the perspective camera and looked like a 3D game
//! about Tetris; the fix was not an orthographic projection but noticing that a
//! board of coloured squares has no third dimension to project.
//!
//! So this demo gates a different part of the engine from its neighbours: 01–06
//! and 09 exercise the scene path, and this one puts ~250 widgets through
//! `gg-ui` every tick, which is two orders of magnitude past demo 07's HUD.
//!
//! # Everything is world state, including the randomness
//!
//! The 7-bag draws from a [`sim::Rng`] living *in* [`Play`] — not beside it.
//! An RNG the sim reads but the world does not hold is one the canonical hash
//! does not cover, and a replay would diverge on the first piece without the
//! gate noticing (§4.7, §6 M18). Everything else follows the same rule: the
//! well is a byte per cell in a component, the falling piece is four numbers,
//! and there is no state in this file that a snapshot would miss.
//!
//! # Integer time, and why the tables are masks
//!
//! No elapsed seconds anywhere — gravity is a tick counter against a tick
//! budget (§1.3, and a float time field is a grep gate besides). Tetromino
//! shapes are 4x4 **bitmasks** rather than coordinate lists because the four
//! rotations are then four `u16` literals per piece that a test can check by
//! popcount; a hand-written table of 112 signed offsets is a sign error waiting
//! for a player to find it.
//!
//! Run it: `cargo xtask run 10-tetris`.

use gg_ecs::Component;
use gg_ecs::boundary::{ActionId, GameWorld, Sound, TEXT, Widget, log_level, wave};
use gg_math::sim;

/// Columns.
pub const WIDTH: usize = 10;
/// Rows *including* the two hidden ones a piece spawns into. Row 0 is the top:
/// gravity increases the index, which is the one convention that keeps the fall
/// arithmetic free of sign flips.
pub const HEIGHT: usize = 22;
/// Rows the player can see. The rest is spawn room, above the ceiling.
pub const HIDDEN: usize = 2;
/// Cells in the well, for the doc that wants a single number.
pub const CELLS: usize = WIDTH * HEIGHT;

/// The seven tetrominoes, four rotations each, as bits of a 4x4 box: bit
/// `row * 4 + col`, row 0 at the top, col 0 at the left.
///
/// Spawn orientations are SRS's. The *kicks* are not — see [`rotate`], which
/// tries a short offset list rather than SRS's per-piece tables. That is a
/// deliberate simplification and a P1: SRS kicks are what make T-spins
/// possible, and a player who knows the game will feel their absence before
/// anything else here.
pub const SHAPES: [[u16; 4]; 7] = [
    [0x00f0, 0x4444, 0x0f00, 0x2222], // I
    [0x0066, 0x0066, 0x0066, 0x0066], // O
    [0x0072, 0x0262, 0x0270, 0x0232], // T
    [0x0036, 0x0462, 0x0360, 0x0231], // S
    [0x0063, 0x0264, 0x0630, 0x0132], // Z
    [0x0071, 0x0226, 0x0470, 0x0322], // J
    [0x0074, 0x0622, 0x0170, 0x0223], // L
];

/// Move the piece one column left.
pub const LEFT: ActionId = ActionId::new(0);
/// Move the piece one column right.
pub const RIGHT: ActionId = ActionId::new(1);
/// Fall at the soft-drop rate while held.
pub const SOFT_DROP: ActionId = ActionId::new(2);
/// Drop to the floor and lock this tick.
pub const HARD_DROP: ActionId = ActionId::new(3);
/// Rotate clockwise.
pub const ROTATE_CW: ActionId = ActionId::new(4);
/// Rotate counter-clockwise.
pub const ROTATE_CCW: ActionId = ActionId::new(5);
/// Swap the falling piece with the held one; once per piece.
pub const HOLD: ActionId = ActionId::new(6);
/// Start again. The only action that does anything after a top-out.
pub const RESTART: ActionId = ActionId::new(7);

// ---------------------------------------------------------------- the layout

/// Side of a drawn cell, canvas units. The well is `WIDTH * CELL` wide, which is
/// what every horizontal number below is derived from rather than guessed.
pub const CELL: f32 = 14.0;
/// Canvas units of the cell left undrawn, so the board reads as a grid instead
/// of as one mass. A gap and not a border: a border is four more rectangles per
/// cell and two hundred cells make that eight hundred.
pub const GUTTER: f32 = 1.0;
/// Thickness of the well's surround.
pub const FRAME: f32 = 3.0;
/// Top-left of the well's first *visible* cell.
pub const WELL_AT: (f32, f32) = (250.0, 40.0);
/// Top-left of the next bay's 4x4 box, then the hold bay's.
pub const BAY_AT: [(f32, f32); 2] = [(482.0, 59.0), (102.0, 59.0)];
/// Which [`Bay`] shows the next piece, and which the held one.
pub const NEXT_BAY: u8 = 0;
/// See [`NEXT_BAY`].
pub const HOLD_BAY: u8 = 1;

/// Body text, captions, and the one accent the title is in.
const INK: u32 = 0xffc6_d2e4;
const DIM: u32 = 0xff6e_7d92;
const ACCENT: u32 = 0xff5a_d2b4;
/// Panel fill, the well's surround, and an empty cell.
const PANEL: u32 = 0xff10_1723;
const SURROUND: u32 = 0xff2c_3a52;
const EMPTY: u32 = 0xff0a_0f18;
/// Where the falling piece would land. Dim enough to read as a hint rather than
/// as a piece — the one thing a player must never confuse it with.
const GHOST: u32 = 0xff30_3c50;
/// The plate the top-out message sits on. Alpha is real here (§4.9), so the
/// dead board still reads through it — which is what a player wants to look at
/// after losing.
const SHROUD: u32 = 0xe80c_1018;

/// Piece colours, indexed by kind, `0xAARRGGBB` — the classic assignment.
/// Opaque: a [`Widget`]'s alpha is honoured, unlike a `Renderable`'s colour,
/// and a cell at zero alpha is an invisible board.
pub const COLORS: [u32; 7] = [
    0xff00_f0f0, // I
    0xfff0_f000, // O
    0xffa0_00f0, // T
    0xff00_f000, // S
    0xfff0_0000, // Z
    0xff00_00f0, // J
    0xfff0_a000, // L
];

/// The four backdrops: hold, stats, next, controls. `[x, y, w, h]`.
const PANELS: [[f32; 4]; 4] = [
    [70.0, 37.0, 120.0, 96.0],
    [70.0, 145.0, 120.0, 118.0],
    [450.0, 37.0, 120.0, 96.0],
    [450.0, 145.0, 120.0, 130.0],
];

/// The labels that never change: `(x, y, text, colour)`. A table rather than
/// seven spawn sites, so moving the stats column is one edit.
const STATIC_TEXT: [(f32, f32, &str, u32); 7] = [
    (302.0, 20.0, "TETRIS", ACCENT),
    (80.0, 45.0, "HOLD", DIM),
    (460.0, 45.0, "NEXT", DIM),
    (80.0, 153.0, "SCORE", DIM),
    (80.0, 185.0, "LINES", DIM),
    (80.0, 217.0, "LEVEL", DIM),
    (460.0, 153.0, "CONTROLS", DIM),
];

/// Where [`HudLine`] 0, 1 and 2 write their number.
const VALUE_AT: [(f32, f32); 3] = [(80.0, 165.0), (80.0, 197.0), (80.0, 229.0)];

/// The legend, which is the only documentation a player reads. Keyed by
/// *physical* position like `input.toml`, so this is honest on AZERTY too.
pub const KEYS: [(&str, &str); 6] = [
    ("A D", "MOVE"),
    ("S", "SOFT DROP"),
    ("W", "HARD DROP"),
    ("Q E", "ROTATE"),
    ("SPC", "HOLD"),
    ("R", "RESTART"),
];
/// Top-left of the legend's first row, and the two column offsets within it.
const KEYS_AT: (f32, f32) = (460.0, 169.0);
const KEYS_STEP: f32 = 12.0;
const KEYS_COLUMN: f32 = 42.0;

/// `gg-ui`'s fallback face advances six canvas units per character and inks
/// seven of eight vertically (`gg_ui::font::CELL`). Restated rather than
/// imported: §3's deny pin keeps a game crate off `gg-ui`, so the number a
/// layout measures with lives here and `text_width`'s own test is what stops it
/// from drifting.
const GLYPH_CELL: f32 = 6.0;
/// Height of a label's rectangle — the face's cell, plus one for a descender.
const LINE: f32 = 9.0;
/// What the stats column is sized for. Six digits: an ordinary game passes
/// 100000, and a rect that fitted the score at spawn would clip it later.
const WIDEST_VALUE: &str = "999999";

/// The top-out banner: a plate, then two lines centred over the well. `("", _)`
/// is the plate — a [`widget::PANEL`](gg_ecs::boundary::widget) draws no text,
/// so an empty body is how the table says which row is the backdrop.
const BANNER: [(&str, u32); 3] = [("", SHROUD), ("GAME OVER", ACCENT), ("PRESS R", INK)];
/// Height of the banner's plate, and where its top sits inside the well.
const BANNER_PLATE: (f32, f32) = (110.0, 56.0);

/// Where [`BANNER`]'s `index`th widget goes when it is shown. The plate spans
/// the well; the two lines are centred in it, which is arithmetic rather than
/// three more literals to keep in agreement.
#[must_use]
pub fn banner_rect(index: usize) -> [f32; 4] {
    let well = well_rect();
    let top = well[1] + BANNER_PLATE.0;
    if index == 0 {
        return [well[0], top, well[2], BANNER_PLATE.1];
    }
    let body = BANNER[index % BANNER.len()].0;
    let width = text_width(body);
    [
        well[0] + (well[2] - width) / 2.0,
        top + 18.0 + (index - 1) as f32 * 20.0,
        width,
        LINE,
    ]
}

/// Draw order bases (§4.9: ascending, last drawn wins). Distinct per widget
/// rather than shared, because the sort key is `(order, id)` and every
/// rectangle here has id 0 — equal keys would leave the picture's triangle
/// order up to a sort's tie-breaking rather than up to this table.
const ORDER_PANEL: u32 = 0;
const ORDER_SURROUND: u32 = 8;
const ORDER_CELL: u32 = 16;
const ORDER_BAY: u32 = 256;
const ORDER_TEXT: u32 = 512;
const ORDER_BANNER: u32 = 1024;

/// Top-left of a visible cell.
#[must_use]
pub fn cell_rect(row: usize, col: usize) -> [f32; 4] {
    [
        WELL_AT.0 + col as f32 * CELL,
        WELL_AT.1 + row as f32 * CELL,
        CELL - GUTTER,
        CELL - GUTTER,
    ]
}

/// Top-left of a bay cell. `slot` is [`NEXT_BAY`] or [`HOLD_BAY`].
#[must_use]
pub fn bay_rect(slot: u8, row: usize, col: usize) -> [f32; 4] {
    let (x, y) = BAY_AT[slot as usize % BAY_AT.len()];
    [
        x + col as f32 * CELL,
        y + row as f32 * CELL,
        CELL - GUTTER,
        CELL - GUTTER,
    ]
}

/// The well's surround, which is also the rectangle the shroud covers.
#[must_use]
pub fn well_rect() -> [f32; 4] {
    [
        WELL_AT.0 - FRAME,
        WELL_AT.1 - FRAME,
        WIDTH as f32 * CELL + 2.0 * FRAME,
        (HEIGHT - HIDDEN) as f32 * CELL + 2.0 * FRAME,
    ]
}

// ------------------------------------------------------------ the components

/// The well: one byte per cell, `0` empty and `kind + 1` otherwise.
///
/// A byte rather than a bit because the colour is the kind, and a bitset would
/// need a second array to say which piece each locked cell came from.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "tetris.well")]
#[repr(C)]
pub struct Well {
    /// `cells[row][col]`. Two dimensions rather than one flat run because
    /// `bytemuck` impls `Pod` for arrays up to 32 without `min_const_generics`
    /// and a flat 220 is past that — and because the nested form is the one
    /// where a row shift is an assignment.
    pub cells: [[u8; WIDTH]; HEIGHT],
}

/// The falling piece. Absent from the world only in the sense that a topped-out
/// game leaves [`Play::over`] set and stops moving it.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "tetris.piece")]
#[repr(C)]
pub struct Piece {
    /// Column of the 4x4 box's left edge; may go negative while a rotation is
    /// being tested against the wall.
    pub col: i32,
    /// Row of the 4x4 box's top edge.
    pub row: i32,
    /// Index into [`SHAPES`].
    pub kind: u8,
    /// Index into a kind's four rotations.
    pub rot: u8,
    /// Padding, named so `Pod` has no hidden bytes to disagree about.
    pub pad: [u8; 2],
}

/// The knobs, and the reason this demo exists in the shape it does.
///
/// Ticks, never seconds. These are what an agent or the inspector reaches for:
/// halve [`Rules::gravity_ticks`] and the game is twice as fast on the next
/// tick, with the stack intact.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "tetris.rules")]
#[repr(C)]
pub struct Rules {
    /// Ticks per cell of fall at level 1. Each level takes 8% off, floored by
    /// [`MIN_GRAVITY_TICKS`].
    pub gravity_ticks: u32,
    /// Ticks a direction must be held before it auto-repeats.
    pub das_ticks: u32,
    /// Ticks between auto-repeats once DAS has charged.
    pub arr_ticks: u32,
    /// Ticks per cell while soft drop is held.
    pub soft_drop_ticks: u32,
    /// Ticks a piece may rest on the stack before it locks. Resets on a move
    /// that succeeds, which is what makes a slide under an overhang possible.
    pub lock_delay_ticks: u32,
    /// Lines per level.
    pub lines_per_level: u32,
}

impl Rules {
    /// Where a new game starts. Roughly a guideline game at level 1.
    pub const DEFAULT: Rules = Rules {
        gravity_ticks: 48,
        das_ticks: 10,
        arr_ticks: 2,
        soft_drop_ticks: 3,
        lock_delay_ticks: 30,
        lines_per_level: 10,
    };
}

/// The floor on gravity, so a high level is fast rather than instantaneous.
pub const MIN_GRAVITY_TICKS: u32 = 2;
/// Score for clearing 1, 2, 3 and 4 rows, before the level multiplier.
pub const LINE_SCORE: [u32; 4] = [100, 300, 500, 800];
/// What [`Play::hold`] holds when it holds nothing.
pub const NO_PIECE: u8 = u8::MAX;
/// The composed-grid value meaning "the piece would land here". Outside the
/// `kind + 1` range a locked cell uses, so [`color_of`] can tell them apart
/// without a second grid.
pub const GHOST_CELL: u8 = 8;

/// Everything else. One component because it is one state machine, and
/// splitting it would only spread a single `each` across several.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "tetris.play")]
#[repr(C)]
pub struct Play {
    /// The bag's source. In the world, so the hash covers it (§6 M18).
    pub rng: sim::Rng,
    /// Ticks accumulated toward the next cell of fall.
    pub fall_accum: u32,
    /// Ticks a direction has been held.
    pub das_accum: u32,
    /// Ticks since the last auto-repeat.
    pub arr_accum: u32,
    /// Ticks the piece has been resting on the stack.
    pub lock_accum: u32,
    /// Points.
    pub score: u32,
    /// Rows cleared, all game.
    pub lines: u32,
    /// One-based, so the gravity formula reads as the rules do.
    pub level: u32,
    /// The shuffled seven, drawn front to back.
    pub bag: [u8; 7],
    /// How many of [`Play::bag`] have been drawn.
    pub bag_pos: u8,
    /// The piece after the falling one.
    pub next: u8,
    /// The held piece, or [`NO_PIECE`].
    pub hold: u8,
    /// Whether hold has been used since the last lock — one swap per piece.
    pub hold_used: u8,
    /// Set on a top-out. Only [`RESTART`] clears it.
    pub over: u8,
    /// Which direction DAS is charged for: 0 none, 1 left, 2 right.
    pub das_dir: u8,
    /// Padding to the `u64` alignment the [`Play::rng`] field imposes. Named,
    /// because `derive(Pod)` refuses a type with bytes nobody declared — which
    /// is the point: hidden padding is hashed garbage (§4.2.1 hazard 4).
    pub pad: [u8; 7],
}

/// A drawn cell of the well, addressed by where it is rather than by what is in
/// it. Two hundred of these are spawned once and then only ever recoloured — a
/// Tetris that spawned and despawned entities per lock would be measuring the
/// ECS rather than playing.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "tetris.cell")]
#[repr(C)]
pub struct Cell {
    /// Row within the *visible* field, 0 at the top.
    pub row: u8,
    /// Column.
    pub col: u8,
}

/// A drawn cell of one of the two 4x4 boxes beside the well.
///
/// One component for both because they differ only in where they are: two
/// components would be two identical queries and two identical loops.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "tetris.bay")]
#[repr(C)]
pub struct Bay {
    /// [`NEXT_BAY`] or [`HOLD_BAY`].
    pub slot: u8,
    /// Row within the 4x4 box.
    pub row: u8,
    /// Column within the 4x4 box.
    pub col: u8,
}

/// Which number a [`Widget`] shows, so `hud` can rewrite text in place.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "tetris.hudline")]
#[repr(C)]
pub struct HudLine {
    /// 0 score, 1 lines, 2 level — an index into [`VALUE_AT`].
    pub which: u8,
}

/// A widget that exists only after a top-out.
///
/// It carries its own rectangle because hiding is *a zero rect* (§6 M13's demo
/// 07 reached the same shape): a zero rect draws nothing and cannot be hit, so
/// showing something again means restoring a rectangle somebody has to have
/// kept. Keeping it on the component keeps `bootstrap` and `hud` from having
/// two opinions about where the banner is.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "tetris.banner")]
#[repr(C)]
pub struct Banner {
    /// `[x, y, w, h]`, canvas units — where this widget goes when it is shown.
    pub rect: [f32; 4],
}

// ------------------------------------------------------------------ the sound

/// Which cue an entity's [`Sound`] is. One entity per cue rather than one
/// shared: the host treats an entity as a voice, so a lock landing on the same
/// tick as a line clear needs two of them or the second cuts the first off
/// (§6 M18 item 2).
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "tetris.cue")]
#[repr(C)]
pub struct Cue {
    /// An index into [`voice_of`]'s match — `CUE_*` below.
    pub kind: u8,
    /// `Pod` refuses padding and `kind` is one byte in a four-byte struct, so
    /// the tail is explicit. Zero, always, because a component is hashed whole.
    pub pad: [u8; 3],
}

/// A piece slid sideways one cell.
pub const CUE_MOVE: u8 = 0;
/// A rotation was accepted — a refused one is silent, which is the feedback.
pub const CUE_ROTATE: u8 = 1;
/// A piece came to rest.
pub const CUE_LOCK: u8 = 2;
/// Rows went away. Pitched and lengthened by how many (§6 M18 item 2).
pub const CUE_CLEAR: u8 = 3;
/// The hold bay swapped.
pub const CUE_HOLD: u8 = 4;
/// The level went up.
pub const CUE_LEVEL: u8 = 5;
/// Top-out.
pub const CUE_OVER: u8 = 6;
/// How many there are.
pub const CUES: usize = 7;

/// The voice each cue speaks with.
///
/// Every number here is the game's, which is the protocol's whole point: a host
/// that resolved `"line_clear.wav"` would hold this taste, and changing how a
/// lock feels would not be a reloadable edit (§4.2.2). These are — the tuning
/// below is inside the dylib, so it changes while someone is playing.
#[must_use]
pub fn voice_of(kind: u8) -> Sound {
    match kind {
        // Short, quiet and low: it fires on every DAS repeat, so anything
        // longer than a tick or two at this rate is a buzz rather than a click.
        CUE_MOVE => Sound::tone(wave::SQUARE, 220.0, 16, 0.10),
        CUE_ROTATE => Sound::tone(wave::SQUARE, 330.0, 22, 0.16),
        // Downward, so a lock reads as something settling.
        CUE_LOCK => Sound::sweep(wave::TRIANGLE, 180.0, 90.0, 60, 0.30),
        // Overwritten per clear by `step` — one row and four rows are the same
        // cue at different pitches and lengths.
        CUE_CLEAR => Sound::sweep(wave::SQUARE, 440.0, 660.0, 90, 0.28),
        CUE_HOLD => Sound::sweep(wave::SINE, 520.0, 780.0, 70, 0.22),
        CUE_LEVEL => Sound::sweep(wave::SQUARE, 523.0, 1_046.0, 180, 0.26),
        // The one long note in the game, and the only one that falls a whole
        // register: the run is over and nothing else is going to play.
        _ => Sound::sweep(wave::TRIANGLE, 440.0, 55.0, 900, 0.34),
    }
}

/// The clear cue, tuned to `lines`: higher, longer and louder with each row, so
/// a tetris is audibly a different event from a single rather than the same
/// blip four times.
#[must_use]
pub fn clear_voice(lines: u32) -> Sound {
    let rows = lines.clamp(1, 4);
    Sound::sweep(
        wave::SQUARE,
        440.0,
        440.0 * (1.0 + rows as f32 * 0.5),
        70 + 50 * rows,
        0.26 + 0.05 * rows as f32,
    )
}

/// A short string built without allocating — the HUD rewrites three numbers
/// every tick, and `format!` there would be a heap allocation per tick per
/// label for text that is almost always the same (§4.9's steady-state rule,
/// applied by a game rather than to one).
struct Text {
    buf: [u8; TEXT],
    len: usize,
}

impl Text {
    /// `value` in decimal, truncated rather than panicking if it does not fit.
    fn number(value: u32) -> Text {
        let mut text = Text {
            buf: [0; TEXT],
            len: 0,
        };
        let mut digits = [0u8; 10];
        let mut count = 0;
        let mut left = value;
        loop {
            digits[count] = b'0' + u8::try_from(left % 10).unwrap_or(0);
            count += 1;
            left /= 10;
            if left == 0 || count == digits.len() {
                break;
            }
        }
        for i in (0..count).rev() {
            text.push(digits[i]);
        }
        text
    }

    fn push(&mut self, byte: u8) {
        if self.len < TEXT {
            self.buf[self.len] = byte;
            self.len += 1;
        }
    }

    /// ASCII throughout, so this cannot split a character.
    fn as_str(&self) -> &str {
        core::str::from_utf8(&self.buf[..self.len]).unwrap_or("")
    }
}

// ----------------------------------------------------------------- the rules

/// The four cells of `kind` in rotation `rot`, as `(row, col)` within the 4x4
/// box. Always four, which [`SHAPES`]' own test is what guarantees.
#[must_use]
pub fn cells_of(kind: u8, rot: u8) -> [(i32, i32); 4] {
    let mask = SHAPES[kind as usize % SHAPES.len()][rot as usize % 4];
    let mut out = [(0, 0); 4];
    let mut found = 0;
    for bit in 0..16i32 {
        if mask & (1 << bit) != 0 && found < 4 {
            out[found] = (bit / 4, bit % 4);
            found += 1;
        }
    }
    out
}

/// Would `piece` overlap the walls, the floor or a locked cell?
#[must_use]
pub fn collides(well: &Well, piece: &Piece) -> bool {
    for (row, col) in cells_of(piece.kind, piece.rot) {
        let (r, c) = (piece.row + row, piece.col + col);
        if c < 0 || c >= WIDTH as i32 || r >= HEIGHT as i32 {
            return true;
        }
        // Above the ceiling is empty by definition; a spawning piece is there.
        if r >= 0 && well.cells[r as usize][c as usize] != 0 {
            return true;
        }
    }
    false
}

/// The row `piece` would come to rest on if it fell from where it is.
///
/// The same loop hard drop runs, factored out because the ghost has to agree
/// with it exactly: a landing preview that disagrees with the drop is worse
/// than no preview.
#[must_use]
pub fn landing_row(well: &Well, piece: &Piece) -> i32 {
    let mut probe = *piece;
    loop {
        probe.row += 1;
        if collides(well, &probe) {
            return probe.row - 1;
        }
    }
}

/// Try to rotate, then try again shifted. Not SRS's kick tables — see
/// [`SHAPES`] — but enough that a rotation against a wall or a stack finds the
/// obvious place to go instead of being refused.
///
/// The offsets are tried in a fixed order, so two machines kick identically.
pub fn rotate(well: &Well, piece: &mut Piece, clockwise: bool) -> bool {
    const KICKS: [i32; 5] = [0, -1, 1, -2, 2];
    let from = piece.rot;
    piece.rot = if clockwise {
        (piece.rot + 1) % 4
    } else {
        (piece.rot + 3) % 4
    };
    for kick in KICKS {
        piece.col += kick;
        if !collides(well, piece) {
            return true;
        }
        piece.col -= kick;
    }
    piece.rot = from;
    false
}

/// Draw the next kind, refilling and reshuffling the bag when it runs out.
///
/// A 7-bag: every piece appears once per seven before any appears twice, which
/// is what stops a run of four S pieces from being the reason a player lost.
pub fn draw(play: &mut Play) -> u8 {
    if play.bag_pos as usize >= play.bag.len() {
        play.bag = [0, 1, 2, 3, 4, 5, 6];
        play.rng.shuffle(&mut play.bag);
        play.bag_pos = 0;
    }
    let kind = play.bag[play.bag_pos as usize];
    play.bag_pos += 1;
    kind
}

/// Put `kind` at the spawn position. Rows 0 and 1 are the hidden ones, so a
/// piece appears to enter from above the ceiling.
#[must_use]
pub fn spawn_piece(kind: u8) -> Piece {
    Piece {
        col: 3,
        row: 0,
        kind,
        rot: 0,
        pad: [0; 2],
    }
}

/// Write the piece into the well.
pub fn lock_piece(well: &mut Well, piece: &Piece) {
    for (row, col) in cells_of(piece.kind, piece.rot) {
        let (r, c) = (piece.row + row, piece.col + col);
        if r >= 0 && r < HEIGHT as i32 && c >= 0 && c < WIDTH as i32 {
            well.cells[r as usize][c as usize] = piece.kind + 1;
        }
    }
}

/// Remove every full row, dropping what was above it. Returns how many went.
pub fn clear_rows(well: &mut Well) -> u32 {
    let mut write = HEIGHT as i32 - 1;
    let mut cleared = 0;
    for read in (0..HEIGHT as i32).rev() {
        if well.cells[read as usize].iter().all(|&c| c != 0) {
            cleared += 1;
            continue;
        }
        if write != read {
            well.cells[write as usize] = well.cells[read as usize];
        }
        write -= 1;
    }
    // Whatever the compaction left above `write` is now empty sky.
    for row in 0..=write {
        well.cells[row as usize] = [0; WIDTH];
    }
    cleared
}

/// Ticks per cell at `level`, 8% off per level down to [`MIN_GRAVITY_TICKS`].
#[must_use]
pub fn gravity_for(rules: &Rules, level: u32) -> u32 {
    let mut ticks = rules.gravity_ticks;
    for _ in 1..level.max(1) {
        ticks = ticks * 92 / 100;
        if ticks <= MIN_GRAVITY_TICKS {
            return MIN_GRAVITY_TICKS;
        }
    }
    ticks.max(MIN_GRAVITY_TICKS)
}

/// A fresh game, seeded. The seed is an argument rather than a clock read: a
/// wall-clock seed is a game that cannot be replayed (§4.7).
#[must_use]
pub fn new_play(seed: u64) -> Play {
    let mut play = Play {
        rng: sim::Rng::from_seed(seed),
        fall_accum: 0,
        das_accum: 0,
        arr_accum: 0,
        lock_accum: 0,
        score: 0,
        lines: 0,
        level: 1,
        bag: [0; 7],
        bag_pos: 7, // empty, so the first draw shuffles
        next: 0,
        hold: NO_PIECE,
        hold_used: 0,
        over: 0,
        das_dir: 0,
        pad: [0; 7],
    };
    play.next = draw(&mut play);
    play
}

// --------------------------------------------------------------- the systems

/// The board, both bays, the HUD, the legend and the frame — once. Re-entrant,
/// because a reload runs it again over a world that already holds them (§6 M5).
pub fn bootstrap(world: &mut GameWorld) {
    let mut exists = false;
    let _ = world.each::<&Play>(|_, _| exists = true);
    if exists {
        return;
    }

    let mut play = new_play(0x5445_5452_4953_0001);
    let first = draw(&mut play);
    let piece = spawn_piece(first);
    let board = world.spawn_with(play);
    world.put(
        board,
        Well {
            cells: [[0; WIDTH]; HEIGHT],
        },
    );
    world.put(board, piece);
    world.put(board, Rules::DEFAULT);

    declare(|part, widget| {
        let entity = world.spawn_with(widget);
        match part {
            Part::Chrome => {}
            Part::Cell(cell) => world.put(entity, cell),
            Part::Bay(bay) => world.put(entity, bay),
            Part::Value(line) => world.put(entity, line),
            Part::Banner(banner) => world.put(entity, banner),
        }
    });

    // One entity per cue, so overlapping events overlap audibly (§6 M18 item
    // 2). Spawned here and never again: the host registers a `Sound` on first
    // sight without playing it, so a reload that re-ran `bootstrap` over a fresh
    // world would be silent anyway — but the early return above means it does
    // not, and the cue bank survives a reload with the stack.
    for kind in 0..CUES as u8 {
        let cue = world.spawn_with(Cue { kind, pad: [0; 3] });
        world.put(cue, voice_of(kind));
    }

    world.log(log_level::INFO, "tetris: ready");
}

/// What rewrites a declared widget once it exists. `Chrome` is the part nothing
/// ever touches again; the rest carry the marker component that finds it.
pub enum Part {
    /// Panels, the well's surround, and every label that never changes.
    Chrome,
    /// A cell of the well, recoloured by [`present`].
    Cell(Cell),
    /// A cell of a bay, recoloured by [`present`].
    Bay(Bay),
    /// A number, rewritten by [`hud`].
    Value(HudLine),
    /// Part of the top-out banner, shown or hidden by [`hud`].
    Banner(Banner),
}

/// Every widget this game declares, at rest, handed to `emit` once each in draw
/// order.
///
/// One description with two consumers: [`bootstrap`] spawns from it, and
/// `gg-golden`'s scene builds the same board *without* the boundary — §3's deny
/// pin means the harness cannot link a second `gg_game!`, so a golden scene for
/// this demo either shares this function or restates sixty rectangles and
/// drifts from them (§4.10 — the golden guards the demo, not a lookalike).
///
/// A callback rather than a returned list because the game side allocates
/// nothing: `bootstrap` runs once, but "once" is also every reload.
pub fn declare(mut emit: impl FnMut(Part, Widget)) {
    for (index, rect) in PANELS.iter().enumerate() {
        emit(
            Part::Chrome,
            ordered(Widget::panel(*rect, PANEL), ORDER_PANEL + index as u32),
        );
    }
    emit(
        Part::Chrome,
        ordered(Widget::panel(well_rect(), SURROUND), ORDER_SURROUND),
    );

    for row in 0..(HEIGHT - HIDDEN) {
        for col in 0..WIDTH {
            let cell = Cell {
                row: row as u8,
                col: col as u8,
            };
            let order = ORDER_CELL + (row * WIDTH + col) as u32;
            emit(
                Part::Cell(cell),
                ordered(Widget::panel(cell_rect(row, col), EMPTY), order),
            );
        }
    }
    for slot in [NEXT_BAY, HOLD_BAY] {
        for row in 0..4 {
            for col in 0..4 {
                let bay = Bay {
                    slot,
                    row: row as u8,
                    col: col as u8,
                };
                let order = ORDER_BAY + u32::from(slot) * 16 + (row * 4 + col) as u32;
                emit(
                    Part::Bay(bay),
                    ordered(Widget::panel(bay_rect(slot, row, col), EMPTY), order),
                );
            }
        }
    }

    let mut order = ORDER_TEXT;
    let mut label = |x: f32, y: f32, body: &str, color: u32| {
        let widget = Widget::label([x, y, text_width(body), LINE], color, body);
        order += 1;
        ordered(widget, order - 1)
    };
    for (x, y, body, color) in STATIC_TEXT {
        emit(Part::Chrome, label(x, y, body, color));
    }
    for (row, (key, action)) in KEYS.iter().enumerate() {
        let y = KEYS_AT.1 + row as f32 * KEYS_STEP;
        emit(Part::Chrome, label(KEYS_AT.0, y, key, INK));
        emit(Part::Chrome, label(KEYS_AT.0 + KEYS_COLUMN, y, action, DIM));
    }
    for (which, (x, y)) in VALUE_AT.iter().enumerate() {
        // Sized for the widest number it will ever hold, not for the "0" it
        // starts as: a label is clipped to its own rect, so a rect that fitted
        // the initial text would cut the score at one digit.
        let rect = [*x, *y, text_width(WIDEST_VALUE), LINE];
        let widget = ordered(Widget::label(rect, INK, "0"), order);
        order += 1;
        emit(Part::Value(HudLine { which: which as u8 }), widget);
    }

    // The banner is emitted hidden — a zero rect — and `hud` is the only thing
    // that ever gives it one. Emitting it visible would show GAME OVER on the
    // first frame of a new game for exactly one tick.
    for (index, (body, color)) in BANNER.iter().enumerate() {
        let widget = match body.is_empty() {
            true => Widget::panel([0.0; 4], SHROUD),
            false => Widget::label([0.0; 4], *color, body),
        };
        emit(
            Part::Banner(Banner {
                rect: banner_rect(index),
            }),
            ordered(widget, ORDER_BANNER + index as u32),
        );
    }
}

/// A widget's draw order, set after construction because the constructors do
/// not take one — order is the game's, and most games have one layer.
fn ordered(mut widget: Widget, order: u32) -> Widget {
    widget.order = order;
    widget
}

/// Canvas width of a run of text in `gg-ui`'s fallback face.
///
/// A label is clipped to its own rectangle (§4.9), so a rect narrower than its
/// text silently cuts it — a layout bug that reads as a typo. [`GLYPH_CELL`] is
/// that face's advance; a proportional face would make this a host call, which
/// §3's pin does not allow, so a game either measures like this or over-sizes
/// every rectangle it authors.
#[must_use]
pub fn text_width(text: &str) -> f32 {
    text.chars().count() as f32 * GLYPH_CELL
}

/// One tick of the game: input, gravity, lock, clear, spawn.
///
/// Input is read into locals before the query, because `each` borrows the world
/// and the two frames an edge is computed from are reached through it.
pub fn step(world: &mut GameWorld) {
    let (left, right) = (world.pressed(LEFT), world.pressed(RIGHT));
    let (tap_left, tap_right) = (world.just_pressed(LEFT), world.just_pressed(RIGHT));
    let soft = world.pressed(SOFT_DROP);
    let hard = world.just_pressed(HARD_DROP);
    let (cw, ccw) = (
        world.just_pressed(ROTATE_CW),
        world.just_pressed(ROTATE_CCW),
    );
    let hold = world.just_pressed(HOLD);
    let restart = world.just_pressed(RESTART);

    let mut topped_out = false;
    // Collected here and played after the pass, because the cue entities are
    // outside this query and a system cannot hold two aliasing views of the
    // world. `cleared` doubles as the clear cue's pitch.
    let mut fired = [false; CUES];
    let mut cleared_rows = 0;
    let _ = world.each::<(&mut Well, &mut Play, &mut Piece, &Rules)>(
        |_, (well, play, piece, rules)| {
            if play.over != 0 {
                if restart {
                    // A fresh clock-read seed would make the next game
                    // unreproducible; advancing the stream keeps it in the world.
                    let seed = play.rng.next_u64();
                    *well = Well {
                        cells: [[0; WIDTH]; HEIGHT],
                    };
                    *play = new_play(seed);
                    let first = draw(play);
                    *piece = spawn_piece(first);
                }
                return;
            }

            // Horizontal, with DAS: the tap moves once, then nothing until the
            // charge completes, then one cell per ARR.
            let dir = match (left, right) {
                (true, false) => 1u8,
                (false, true) => 2u8,
                _ => 0,
            };
            if dir != play.das_dir {
                play.das_dir = dir;
                play.das_accum = 0;
                play.arr_accum = 0;
            }
            let step_x = if tap_left {
                -1
            } else if tap_right {
                1
            } else if dir != 0 {
                play.das_accum = play.das_accum.saturating_add(1);
                if play.das_accum > rules.das_ticks {
                    play.arr_accum = play.arr_accum.saturating_add(1);
                    if play.arr_accum >= rules.arr_ticks.max(1) {
                        play.arr_accum = 0;
                        if dir == 1 { -1 } else { 1 }
                    } else {
                        0
                    }
                } else {
                    0
                }
            } else {
                0
            };
            if step_x != 0 {
                piece.col += step_x;
                if collides(well, piece) {
                    piece.col -= step_x;
                } else {
                    play.lock_accum = 0;
                    fired[CUE_MOVE as usize] = true;
                }
            }

            // The refused rotation is silent on purpose: "nothing happened" is
            // the feedback, and a click on a kick that did not fit would say the
            // opposite of what the board shows.
            if (cw || ccw) && rotate(well, piece, cw) {
                play.lock_accum = 0;
                fired[CUE_ROTATE as usize] = true;
            }

            if hold && play.hold_used == 0 {
                play.hold_used = 1;
                let swapped = if play.hold == NO_PIECE {
                    let taken = play.next;
                    play.next = draw(play);
                    taken
                } else {
                    play.hold
                };
                play.hold = piece.kind;
                *piece = spawn_piece(swapped);
                play.lock_accum = 0;
                play.fall_accum = 0;
                fired[CUE_HOLD as usize] = true;
                if collides(well, piece) {
                    play.over = 1;
                    fired[CUE_OVER as usize] = true;
                    return;
                }
            }

            // Gravity. Hard drop short-circuits it: fall until the next cell
            // would collide, then lock this tick rather than the next.
            let mut lock_now = false;
            if hard {
                let landed = landing_row(well, piece);
                play.score = play
                    .score
                    .saturating_add((landed - piece.row).max(0) as u32 * 2);
                piece.row = landed;
                lock_now = true;
            } else {
                let budget = if soft {
                    rules.soft_drop_ticks.max(1)
                } else {
                    gravity_for(rules, play.level)
                };
                play.fall_accum = play.fall_accum.saturating_add(1);
                if play.fall_accum >= budget {
                    play.fall_accum = 0;
                    piece.row += 1;
                    if collides(well, piece) {
                        piece.row -= 1;
                    } else {
                        play.lock_accum = 0;
                        if soft {
                            play.score = play.score.saturating_add(1);
                        }
                    }
                }
                // Resting on the stack: the delay is what allows a last slide.
                piece.row += 1;
                let grounded = collides(well, piece);
                piece.row -= 1;
                if grounded {
                    play.lock_accum = play.lock_accum.saturating_add(1);
                    lock_now = play.lock_accum >= rules.lock_delay_ticks;
                } else {
                    play.lock_accum = 0;
                }
            }

            if !lock_now {
                return;
            }

            lock_piece(well, piece);
            fired[CUE_LOCK as usize] = true;
            let cleared = clear_rows(well);
            if cleared > 0 {
                play.lines = play.lines.saturating_add(cleared);
                let index = (cleared.min(4) - 1) as usize;
                play.score = play
                    .score
                    .saturating_add(LINE_SCORE[index].saturating_mul(play.level));
                let was = play.level;
                play.level = play.lines / rules.lines_per_level.max(1) + 1;
                fired[CUE_CLEAR as usize] = true;
                cleared_rows = cleared;
                fired[CUE_LEVEL as usize] = play.level > was;
            }
            play.lock_accum = 0;
            play.fall_accum = 0;
            play.hold_used = 0;
            let kind = play.next;
            play.next = draw(play);
            *piece = spawn_piece(kind);
            if collides(well, piece) {
                play.over = 1;
                topped_out = true;
                fired[CUE_OVER as usize] = true;
            }
        },
    );

    // The second pass: bump `seq` on whatever happened. The host plays a `Sound`
    // whose sequence moved and writes nothing back, so this is the entire audio
    // path from the game's side (§6 M18 item 2).
    let _ = world.each::<(&Cue, &mut Sound)>(|_, (cue, sound)| {
        if !fired[cue.kind as usize % CUES] {
            return;
        }
        if cue.kind == CUE_CLEAR {
            // Retuned before the bump, not after: the host reads the whole
            // component on the tick the sequence moved.
            let seq = sound.seq;
            *sound = clear_voice(cleared_rows);
            sound.seq = seq;
        }
        sound.play();
    });

    if topped_out {
        world.log(log_level::INFO, "tetris: topped out");
    }
}

/// The well with the ghost and the falling piece composited into it — what is
/// on screen, as opposed to what is locked.
///
/// Pure, so the golden scene and the inspector-facing `present` below agree by
/// construction rather than by both being written the same day.
#[must_use]
pub fn compose_well(well: &Well, play: &Play, piece: &Piece) -> [[u8; WIDTH]; HEIGHT] {
    let mut grid = well.cells;
    if play.over != 0 {
        return grid;
    }
    // Ghost first, so the piece overwrites it where the two overlap — a piece
    // resting on the floor is its own ghost.
    let ghost = Piece {
        row: landing_row(well, piece),
        ..*piece
    };
    for (at, value) in [(&ghost, GHOST_CELL), (piece, piece.kind + 1)] {
        for (row, col) in cells_of(at.kind, at.rot) {
            let (r, c) = (at.row + row, at.col + col);
            if r >= 0 && r < HEIGHT as i32 && c >= 0 && c < WIDTH as i32 {
                grid[r as usize][c as usize] = value;
            }
        }
    }
    grid
}

/// Both 4x4 bays, indexed by slot then `row * 4 + col`. An empty hold bay is
/// all zeroes, which is what makes "nothing held" draw as background.
#[must_use]
pub fn compose_bays(play: &Play) -> [[u8; 16]; 2] {
    let mut bays = [[0u8; 16]; 2];
    for (slot, kind) in [(NEXT_BAY, play.next), (HOLD_BAY, play.hold)] {
        if kind == NO_PIECE {
            continue;
        }
        // Centred in the box, not placed at the spawn orientation's own corner:
        // the shapes sit in different quadrants of their 4x4 mask (I spans a
        // row, O a 2x2), so drawn raw the preview jumps around as pieces come.
        let cells = cells_of(kind, 0);
        let bounds = |axis: fn(&(i32, i32)) -> i32| {
            let (lo, hi) = cells
                .iter()
                .fold((3, 0), |(lo, hi), c| (lo.min(axis(c)), hi.max(axis(c))));
            (3 - (hi - lo)) / 2 - lo
        };
        let (shift_row, shift_col) = (bounds(|c| c.0), bounds(|c| c.1));
        for (row, col) in cells {
            let at = (row + shift_row) * 4 + col + shift_col;
            bays[slot as usize][at.clamp(0, 15) as usize] = kind + 1;
        }
    }
    bays
}

/// The colour a composed board puts on a [`Cell`]. The `+ HIDDEN` is the whole
/// of the mapping: a cell names a *visible* row and the grid is indexed from
/// the ceiling.
#[must_use]
pub fn cell_color(grid: &[[u8; WIDTH]; HEIGHT], cell: &Cell) -> u32 {
    color_of(grid[cell.row as usize + HIDDEN][cell.col as usize % WIDTH])
}

/// The colour composed bays put on a [`Bay`].
#[must_use]
pub fn bay_color(bays: &[[u8; 16]; 2], bay: &Bay) -> u32 {
    color_of(bays[bay.slot as usize % bays.len()][(bay.row as usize * 4 + bay.col as usize) % 16])
}

/// What a [`HudLine`] shows.
#[must_use]
pub fn value_of(play: &Play, line: &HudLine) -> u32 {
    match line.which {
        0 => play.score,
        1 => play.lines,
        _ => play.level,
    }
}

/// Colour the board and both bays from the state.
///
/// Two passes: compose into local grids first, because the board and the drawn
/// cells are different entities and one `each` cannot hold both.
pub fn present(world: &mut GameWorld) {
    let mut grid = [[0u8; WIDTH]; HEIGHT];
    let mut bays = [[0u8; 16]; 2];
    let _ = world.each::<(&Well, &Play, &Piece)>(|_, (well, play, piece)| {
        grid = compose_well(well, play, piece);
        bays = compose_bays(play);
    });

    let _ = world.each::<(&Cell, &mut Widget)>(|_, (cell, widget)| {
        widget.color = cell_color(&grid, cell);
    });
    let _ = world.each::<(&Bay, &mut Widget)>(|_, (bay, widget)| {
        widget.color = bay_color(&bays, bay);
    });
}

/// A composed-cell value's colour: `0` is the well's own dark, [`GHOST_CELL`]
/// the landing hint, anything else its piece.
#[must_use]
pub fn color_of(value: u8) -> u32 {
    match value {
        0 => EMPTY,
        GHOST_CELL => GHOST,
        _ => COLORS[(value - 1) as usize % COLORS.len()],
    }
}

/// Rewrite the three numbers, and show or hide the banner.
pub fn hud(world: &mut GameWorld) {
    let mut snapshot = None;
    let _ = world.each::<&Play>(|_, play| snapshot = Some(*play));
    let Some(play) = snapshot else {
        return;
    };
    let _ = world.each::<(&HudLine, &mut Widget)>(|_, (line, widget)| {
        widget.set_text(Text::number(value_of(&play, line)).as_str());
    });
    let _ = world.each::<(&Banner, &mut Widget)>(|_, (banner, widget)| {
        widget.rect = match play.over != 0 {
            true => banner.rect,
            false => [0.0; 4],
        };
    });
}

// Order in `systems` is execution order (§4.1); order in the verb lists is the
// id space a replay records (§4.7). Neither is alphabetical, neither may drift.
#[cfg(feature = "game")]
gg_ecs::gg_game! {
    components: [Well, Piece, Rules, Play, Cell, Bay, HudLine, Banner, Widget, Cue, Sound],
    actions: [
        "left",
        "right",
        "soft_drop",
        "hard_drop",
        "rotate_cw",
        "rotate_ccw",
        "hold",
        "restart"
    ],
    axes: [],
    systems: [bootstrap, step, present, hud],
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use gg_ecs::boundary::CANVAS;

    /// Everything the game draws, as `[x, y, w, h]` — the layout, without a
    /// world to hold it. The tests below are the only check on a canvas nobody
    /// can see from an automated tier (§1.5).
    fn every_rect() -> Vec<(&'static str, [f32; 4])> {
        let mut all = vec![("well", well_rect())];
        for rect in PANELS {
            all.push(("panel", rect));
        }
        for index in 0..BANNER.len() {
            all.push(("banner", banner_rect(index)));
        }
        for row in 0..(HEIGHT - HIDDEN) {
            for col in 0..WIDTH {
                all.push(("cell", cell_rect(row, col)));
            }
        }
        for slot in [NEXT_BAY, HOLD_BAY] {
            for row in 0..4 {
                for col in 0..4 {
                    all.push(("bay", bay_rect(slot, row, col)));
                }
            }
        }
        all
    }

    #[test]
    fn nothing_is_drawn_off_the_canvas() {
        for (what, [x, y, w, h]) in every_rect() {
            assert!(
                x >= 0.0 && y >= 0.0 && x + w <= CANVAS.0 as f32 && y + h <= CANVAS.1 as f32,
                "{what} at [{x}, {y}, {w}, {h}] leaves the {}x{} canvas",
                CANVAS.0,
                CANVAS.1
            );
        }
    }

    /// The panels sit either side of the well and must not reach it — an
    /// overlap is a HUD drawn over the board, which the draw order would then
    /// decide rather than the layout.
    #[test]
    fn the_side_panels_clear_the_board() {
        let well = well_rect();
        for panel in PANELS {
            let apart = panel[0] + panel[2] <= well[0] || panel[0] >= well[0] + well[2];
            assert!(apart, "panel {panel:?} overlaps the well {well:?}");
        }
    }

    /// Each 4x4 box has to fit inside the panel behind it, or a preview draws
    /// over bare background at one corner.
    #[test]
    fn each_bay_sits_inside_its_panel() {
        for (slot, panel) in [(HOLD_BAY, PANELS[0]), (NEXT_BAY, PANELS[2])] {
            for (row, col) in [(0, 0), (3, 3)] {
                let [x, y, w, h] = bay_rect(slot, row, col);
                assert!(
                    x >= panel[0]
                        && y >= panel[1]
                        && x + w <= panel[0] + panel[2]
                        && y + h <= panel[1] + panel[3],
                    "bay {slot} cell ({row}, {col}) escapes {panel:?}"
                );
            }
        }
    }

    /// Cells are laid out by index, so a gutter wider than the cell would make
    /// them overlap without any single rectangle looking wrong.
    #[test]
    fn adjacent_cells_do_not_touch_and_do_not_gap_more_than_the_gutter() {
        let (a, b) = (cell_rect(0, 0), cell_rect(0, 1));
        assert!(a[0] + a[2] <= b[0], "columns overlap");
        assert!(
            b[0] - (a[0] + a[2]) <= GUTTER,
            "columns gap past the gutter"
        );
        let (a, b) = (cell_rect(0, 0), cell_rect(1, 0));
        assert!(a[1] + a[3] <= b[1], "rows overlap");
    }

    /// The board is centred: a well pushed off-centre by a layout edit is the
    /// kind of thing every individual rectangle passes and the picture fails.
    #[test]
    fn the_board_is_centred_on_the_canvas() {
        let well = well_rect();
        let slack = (CANVAS.0 as f32 - (well[0] + well[2])) - well[0];
        assert!(slack.abs() < 0.5, "the well is {slack} units off centre");
    }

    /// A label is clipped to its own rectangle, so a rect narrower than its
    /// text is a silent truncation. Walked over the *declared* list rather than
    /// over the tables, because the rect a label gets is `declare`'s to compute.
    #[test]
    fn every_declared_label_is_wide_enough_for_its_text() {
        let mut checked = 0;
        declare(|_, widget| {
            let body = widget.text();
            if body.is_empty() {
                return;
            }
            // The banner is declared at a zero rect and given its real one by
            // `hud`, so it is measured against that rather than against nothing.
            let width = match widget.rect[2] {
                0.0 => banner_rect(1)[2],
                width => width,
            };
            assert!(
                width >= text_width(body),
                "{body:?} is clipped by its own {width}-unit rect"
            );
            checked += 1;
        });
        assert_eq!(
            checked,
            STATIC_TEXT.len() + KEYS.len() * 2 + VALUE_AT.len() + 2,
            "a label stopped being declared"
        );

        for (key, action) in KEYS {
            // The two columns must not collide either: the key column is
            // `KEYS_COLUMN` wide and a longer key would run into the action.
            assert!(text_width(key) <= KEYS_COLUMN, "key {key:?} is too wide");
            assert!(
                KEYS_COLUMN + text_width(action) <= PANELS[3][2] - 20.0,
                "action {action:?} runs off its panel"
            );
        }
    }

    /// Draw order is `(order, id)` and every rectangle here has id 0, so two
    /// widgets sharing an order leave the picture's triangle order to a sort's
    /// tie-breaking. Distinctness is the property, and it is easy to lose by
    /// adding a row to one of the tables.
    #[test]
    fn no_two_declared_widgets_share_a_draw_order() {
        let mut orders = Vec::new();
        declare(|_, widget| orders.push(widget.order));
        let total = orders.len();
        orders.sort_unstable();
        orders.dedup();
        assert_eq!(orders.len(), total, "two widgets share a draw order");
    }

    /// A composed cell is the well's own byte unless the piece or its ghost is
    /// over it — the mapping `present` relies on, without a world.
    #[test]
    fn the_composition_puts_the_piece_and_its_ghost_over_the_well() {
        let mut well = Well {
            cells: [[0; WIDTH]; HEIGHT],
        };
        well.cells[HEIGHT - 1][0] = 3;
        let play = new_play(4);
        let piece = spawn_piece(0);
        let grid = compose_well(&well, &play, &piece);
        assert_eq!(grid[HEIGHT - 1][0], 3, "a locked cell was overwritten");
        let ghosts = grid.iter().flatten().filter(|v| **v == GHOST_CELL).count();
        let falling = grid
            .iter()
            .flatten()
            .filter(|v| **v == piece.kind + 1)
            .count();
        assert_eq!((ghosts, falling), (4, 4));

        // A dead board shows neither: the piece stops being where the player is.
        let over = Play { over: 1, ..play };
        assert_eq!(compose_well(&well, &over, &piece), well.cells);
    }

    #[test]
    fn an_empty_hold_bay_composes_to_nothing() {
        let play = new_play(11);
        assert_eq!(play.hold, NO_PIECE);
        let bays = compose_bays(&play);
        assert_eq!(bays[HOLD_BAY as usize], [0; 16]);
        assert_eq!(
            bays[NEXT_BAY as usize].iter().filter(|v| **v != 0).count(),
            4
        );
    }

    /// The measurement the whole layout is built on. `gg_ui::font::CELL` is
    /// `(6, 8)` and a game crate may not link `gg-ui` to read it (§3), so this
    /// is the seam where the two could drift apart unnoticed — pinned here, and
    /// checked against the real face by the golden scene.
    #[test]
    fn text_is_measured_at_the_faces_cell() {
        assert_eq!(GLYPH_CELL, 6.0);
        assert_eq!(text_width(""), 0.0);
        assert_eq!(text_width("GAME OVER"), 54.0, "spaces advance the pen");
    }

    /// A number wider than four digits still has to fit the stats column, and
    /// a score reaches six digits in an ordinary game.
    #[test]
    fn a_six_digit_score_fits_the_stats_panel() {
        let text = Text::number(999_999);
        assert_eq!(text.as_str(), WIDEST_VALUE);
        let right = VALUE_AT[0].0 + text_width(text.as_str());
        assert!(
            right <= PANELS[1][0] + PANELS[1][2],
            "a six-figure score runs off the panel at {right}"
        );
        assert_eq!(Text::number(0).as_str(), "0");
    }

    /// The ghost is a landing preview, so it must be the same row hard drop
    /// puts the piece on — computed by the same function, and asserted anyway
    /// because "the same function" is a property of today's code.
    #[test]
    fn the_ghost_lands_where_the_piece_would() {
        let mut well = Well {
            cells: [[0; WIDTH]; HEIGHT],
        };
        well.cells[HEIGHT - 1] = [1; WIDTH];
        let piece = spawn_piece(2);
        let landed = landing_row(&well, &piece);
        let mut probe = Piece {
            row: landed,
            ..piece
        };
        assert!(
            !collides(&well, &probe),
            "the landing row is inside the stack"
        );
        probe.row += 1;
        assert!(
            collides(&well, &probe),
            "the piece could have fallen further"
        );
    }

    #[test]
    fn a_ghost_cell_is_neither_empty_nor_a_piece() {
        assert_eq!(color_of(0), EMPTY);
        assert_eq!(color_of(GHOST_CELL), GHOST);
        for kind in 0..SHAPES.len() as u8 {
            assert_eq!(color_of(kind + 1), COLORS[kind as usize]);
            assert_ne!(color_of(kind + 1), GHOST, "a piece is drawn as its ghost");
        }
    }

    /// Every colour on screen is opaque unless it is meant to read over
    /// something — a widget colour carries alpha and a zero one is invisible,
    /// which is the mistake a `Renderable`'s `0x00RRGGBB` habit produces.
    #[test]
    fn the_palette_is_opaque_except_where_it_is_deliberately_not() {
        for color in COLORS
            .iter()
            .chain([&EMPTY, &GHOST, &PANEL, &SURROUND, &INK, &DIM, &ACCENT])
        {
            assert_eq!(*color >> 24, 0xff, "{color:#010x} is not opaque");
        }
        const { assert!(SHROUD >> 24 < 0xff, "the shroud hides the dead board") };
    }
}
