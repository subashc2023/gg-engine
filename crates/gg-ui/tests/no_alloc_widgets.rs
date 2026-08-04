//! §6 M13's allocation criterion on the path a *game* uses: `Widget`
//! components in, geometry out (§4.9).
//!
//! `no_alloc.rs` proves the draw layer and the router; `gg-debug`'s proves the
//! overlay. This proves [`Ui::frame`] — the two walks of the world, the sort
//! into draw order, and the write-back — because that is the only one of the
//! three a shipped game is on. A separate binary for the same reason the other
//! two are: a `#[global_allocator]` belongs to one process and counts
//! everything in it.
//!
//! # It asserts zero, and it used to assert a floor
//!
//! `gg_ecs::view::build` allocated two `Vec`s per matching archetype, so *any*
//! `World::each` cost one allocation a call here — nothing to do with the UI,
//! and paid by every game system on every tick. This file measured that number
//! rather than hiding it, as an exact `assert_eq!` against a two-per-frame ECS
//! floor with a message saying a *drop* below it meant the deferred fix had
//! landed and the bound was a line to delete. It had, and this is that line
//! deleted: the view
//! holds its columns inline now (`gg_ecs::view::MAX_TERMS`). Kept in the header
//! because the shape is the reusable part — a bound recorded as equality is
//! what let the fix arrive as a failing test rather than as an unnoticed
//! improvement.

// unwrap is permitted in tests (§2, Error handling row).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use gg_ecs::World;
use gg_ecs::boundary::{Widget, widget_id};
use gg_ui::Ui;
use gg_ui::router::{AXIS_SCALE, Tick};

/// Allocation calls since the process started. Relaxed: a count, not a lock.
static CALLS: AtomicUsize = AtomicUsize::new(0);

struct Counting;

// SAFETY: every method forwards to `System` with the same arguments and returns
// its pointer unchanged, so every guarantee `GlobalAlloc` demands is `System`'s
// and is upheld by it. The counter is the only added effect and it touches no
// memory the allocator owns.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        CALLS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `layout` is the caller's, forwarded unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        CALLS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: as above.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // Counted: a buffer that grows every frame reaches the allocator here.
        CALLS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `ptr` came from `System` with `layout`, forwarded unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: as `realloc`. Frees are not counted — returning memory taken
        // before the window is not allocating inside it.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

const EXTENT: (u32, u32) = (1280, 720);
/// A HUD's worth of widgets, spawned out of draw order so the sort has work.
const WIDGETS: u32 = 24;

fn world() -> World {
    let mut world = World::new();
    world.register::<Widget>().unwrap();
    for i in 0..WIDGETS {
        let entity = world.spawn();
        let x = 8.0 + f32::from(i as u16 % 6) * 100.0;
        let y = 8.0 + f32::from(i as u16 / 6) * 40.0;
        let mut w = Widget::button(
            widget_id("row").wrapping_add(u64::from(i)),
            [x, y, 90.0, 30.0],
            0xff1e_2c3c,
            0xffd8_e0e8,
            "button",
        );
        // Descending, so world order is the reverse of draw order.
        w.order = WIDGETS - i;
        world.insert(entity, w).unwrap();
    }
    world
}

/// A pointer that keeps moving, so hover re-resolves rather than
/// short-circuiting on an unchanged position, and a press every fourth tick.
fn tick_of(tick: u64) -> Tick {
    Tick {
        motion: (
            (tick % 7) as i32 * AXIS_SCALE / 16 - AXIS_SCALE / 4,
            (tick % 5) as i32 * AXIS_SCALE / 16 - AXIS_SCALE / 4,
        ),
        primary: tick.is_multiple_of(4),
        advance_focus: tick.is_multiple_of(16),
        scroll: i32::from(tick.is_multiple_of(120)),
    }
}

/// Frames in the measured window. The ECS charges nothing for them, so every
/// call the counter sees is the UI's.
const MEASURED: u64 = 128;

#[test]
fn a_settled_widget_frame_allocates_nothing_of_its_own() {
    let mut ui = Ui::new().unwrap();
    let mut world = world();

    let cold = CALLS.load(Ordering::Relaxed);
    for tick in 0..512 {
        ui.frame(&mut world, &tick_of(tick), EXTENT);
    }
    let vertices = ui.vertices().len();

    let before = CALLS.load(Ordering::Relaxed);
    for tick in 512..512 + MEASURED {
        ui.frame(&mut world, &tick_of(tick), EXTENT);
    }
    let allocations = CALLS.load(Ordering::Relaxed) - before;

    // Asserted after the window closes, because the assertion machinery itself
    // formats and would be counted. The first is the gate's own gate: a counter
    // wired to the wrong allocator reads zero for the measured window too.
    assert!(
        before > cold,
        "the counter never moved during warmup, so it is not watching this path"
    );
    assert_eq!(
        allocations, 0,
        "{MEASURED} settled widget frames asked the allocator {allocations} times — a settled \
         frame allocates nothing, so this is either the UI or something under it that stopped \
         holding its storage inline"
    );
    assert!(vertices > 0, "the frame under test drew nothing");
}
