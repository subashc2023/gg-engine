//! §6 M13's steady-state allocation criterion, as a machine.
//!
//! Every part of a UI frame is written to reuse its buffers, and every one of
//! them is one careless `format!`, `to_vec` or `collect` away from not doing so
//! — a regression no test that only checks *output* can see. So the gate is a
//! counting global allocator around the frame body itself: warm up, snapshot,
//! run more frames, and require the count not to have moved.
//!
//! The allocator is installed here rather than in the library because a
//! `#[global_allocator]` belongs to a binary, and a test binary is the only one
//! in this crate. That is also why this is the whole file: anything else run in
//! this process would be counted too.

// unwrap is permitted in tests (§2, Error handling row).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_ui::router::Tick;
use gg_ui::{DrawList, FaceId, Fonts, Router, Scratch, Stack, WidgetId};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    /// Allocation calls made **by this thread**.
    ///
    /// It was a process-global `AtomicUsize`, which counted the test harness's
    /// own threads as well: the measured window is a handful of microseconds,
    /// so one allocation from libtest's reporter landing inside it read as a
    /// regression here. That is a gate failing on what the thread next to it
    /// did rather than on its claim (§5) — `no_alloc_widgets` went red once on
    /// the WSL lane and passed thirty runs afterwards, which is the signature.
    ///
    /// Per-thread is the claim exactly: the measured code runs on the caller's
    /// thread. A path that handed work to a pool would need the global counter
    /// back, and none of the three `no_alloc` gates is on one.
    ///
    /// `const`-initialised and `Copy`: reaching it allocates nothing and
    /// registers no destructor, where a lazily-initialised TLS would recurse
    /// through the very allocator it counts.
    static CALLS: Cell<usize> = const { Cell::new(0) };
}

/// Count one call, unless TLS is already torn down — which is not a window
/// anything measures, so the miss is free.
fn charge() {
    let _ = CALLS.try_with(|calls| calls.set(calls.get() + 1));
}

fn calls() -> usize {
    CALLS.try_with(Cell::get).unwrap_or(0)
}

struct Counting;

// SAFETY: every method forwards to `System` with the same arguments and
// returns its pointer unchanged, so every guarantee `GlobalAlloc` demands is
// `System`'s and is upheld by it. The counter is the only added effect and it
// touches no memory the allocator owns.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        charge();
        // SAFETY: `layout` is the caller's, forwarded unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        charge();
        // SAFETY: as above.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // Counted: a `Vec` that grows every frame is exactly the regression
        // this file exists to catch, and it reaches the allocator here.
        charge();
        // SAFETY: `ptr` came from this allocator with `layout`, forwarded
        // unchanged; `System` is the allocator that produced it.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: as `realloc`. Frees are not counted — a frame that returns
        // memory it took before the window is not a frame that allocated.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

const EXTENT: (u32, u32) = (1280, 720);
const PX: u16 = 18;
const WHITE: u32 = 0xffff_ffff;

fn face_bytes() -> Vec<u8> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/gg-ui is two below the workspace root");
    std::fs::read(root.join("assets/fonts/FiraMono-Regular.ttf")).expect("the vendored face")
}

/// Everything a real frame does: route input, format rows, shape and rasterize
/// them, and emit geometry through both text paths and the clip stack.
fn frame(
    tick: u64,
    fonts: &mut Fonts,
    face: FaceId,
    list: &mut DrawList,
    router: &mut Router,
    scratch: &mut Scratch,
) {
    // A pointer that keeps moving, so the router re-resolves rather than
    // short-circuiting on an unchanged position.
    let motion = ((tick % 7) as i32 * 64 - 128, (tick % 5) as i32 * 64 - 128);
    router.begin(
        &Tick {
            motion,
            primary: tick.is_multiple_of(4),
            advance_focus: tick.is_multiple_of(16),
            // A notch every other second, so a scrolled pane is inside the
            // steady state this gate is measuring and not beside it.
            scroll: i32::from(tick.is_multiple_of(120)),
        },
        EXTENT,
    );
    list.clear();
    scratch.clear();

    // Rows whose text changes every frame — a cached-string implementation
    // would pass this file while allocating in the real overlay.
    let rows = [
        scratch.line(format_args!("tick    {tick:>8}")),
        scratch.line(format_args!("{:>6.2} ms", 16.0 + (tick % 32) as f32 / 32.0)),
        scratch.line(format_args!("mem  {:>7}M", tick * 3 % 4096)),
    ];
    let line = fonts.metrics(face, PX).line_height;
    // Measured, then placed — the shape a panel sized to its contents forces.
    let mut stack = Stack::vertical((16.0, 16.0), 2.0);
    for row in &rows {
        stack.push(fonts.layout(face, PX, scratch.get(*row)), line);
    }
    let panel = stack.content().inset(-8.0);
    list.rect(panel, 0xc00c_1016);
    list.push_clip(panel);
    stack.rewind();
    for (index, row) in rows.iter().enumerate() {
        let cell = stack.push(fonts.layout(face, PX, scratch.get(*row)), line);
        list.glyphs(cell.x, cell.y, fonts.glyphs(), WHITE);
        // The fallback path in the same frame: one atlas, one draw.
        list.text(cell.right() + 8.0, cell.y, "ok", WHITE);
        router.hit(WidgetId::new("row").indexed(index as u64), list.place(cell));
    }
    list.pop_clip();
}

/// The criterion: after warmup, a frame allocates nothing at all.
///
/// The digits in the rows above are chosen to cycle through every width the
/// formatter will produce, so the scratch reaches its high-water mark during
/// warmup rather than during the measured window — which is what "steady state"
/// means and what a fixed test string would quietly dodge.
#[test]
fn a_settled_ui_frame_allocates_nothing() {
    let mut fonts = Fonts::default();
    let face = fonts.load(face_bytes(), 0).unwrap();
    let mut list = DrawList::default();
    let mut router = Router::default();
    let mut scratch = Scratch::default();

    // Long enough for every glyph to be resident, every row width to have been
    // seen, and every buffer to have reached its size.
    let cold = calls();
    for tick in 0..512 {
        frame(tick, &mut fonts, face, &mut list, &mut router, &mut scratch);
    }
    let version = fonts.version();
    let vertices = list.vertices().len();

    let before = calls();
    for tick in 512..640 {
        frame(tick, &mut fonts, face, &mut list, &mut router, &mut scratch);
    }
    let allocations = calls() - before;

    // Asserted after the window closes, because the assertion machinery itself
    // formats and would be counted.
    //
    // The first assertion is the gate's own gate: a counter wired up wrongly —
    // to the wrong allocator, or optimized out — reads zero for the measured
    // window too, and this file would pass while proving nothing. Warmup runs
    // the *same* code path and must have allocated.
    assert!(
        before > cold,
        "the counter never moved during warmup, so it is not watching this path"
    );
    assert_eq!(
        allocations, 0,
        "128 settled frames asked the allocator {allocations} times"
    );
    assert_eq!(fonts.version(), version, "no glyph arrived either");
    assert!(vertices > 0, "the frame under test drew nothing");
}
