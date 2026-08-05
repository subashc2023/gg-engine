//! §6 M13's steady-state allocation criterion on the *real* overlay.
//!
//! `gg-ui`'s own `tests/no_alloc.rs` proves the layer; this proves the M8
//! overlay reimplemented on it, which is the acceptance test the milestone
//! actually names. The two are separate binaries because a
//! `#[global_allocator]` belongs to one and counts everything in it — the same
//! reason each of these files is nothing but its allocator and one test.
//!
//! What it would catch is a regression to the shape this milestone replaced:
//! rows built as `format!`ed `String`s in a fresh `Vec`, which cost ten heap
//! allocations every frame and which no test of the *geometry* can see.

// unwrap is permitted in tests (§2, Error handling row).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_debug::overlay::{Overlay, Stats};
use gg_input::Key;
use gg_rhi::{MemoryUse, PassTiming};
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

// SAFETY: every method forwards to `System` with the same arguments and returns
// its pointer unchanged, so every guarantee `GlobalAlloc` demands is `System`'s
// and is upheld by it. The counter is the only added effect and it touches no
// memory the allocator owns.
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
        // Counted: a buffer that grows every frame reaches the allocator here,
        // and that is the regression this file exists to catch.
        charge();
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

/// The pass list a real frame carries. Owned outside the window: the names are
/// the renderer's `String`s and the overlay only reads them.
fn passes() -> Vec<PassTiming> {
    ["forward-opaque", "resolve", "tonemap", "ui"]
        .iter()
        .enumerate()
        .map(|(i, name)| PassTiming {
            name: (*name).to_owned(),
            gpu_ms: 0.0,
            begin: i as i64,
            end: i as i64 + 1,
        })
        .collect()
}

/// One frame of everything the overlay draws: stats rows whose widths change,
/// a full pass list, the histogram, and the console open over the top of it.
fn frame(tick: u64, overlay: &mut Overlay, passes: &mut [PassTiming], bins: &mut [u32; BINS]) {
    // Numbers that keep moving, so every row reformats and the scratch is
    // exercised at each width it will ever reach — a fixed string would settle
    // after one frame and prove nothing about the buffer.
    for (i, pass) in passes.iter_mut().enumerate() {
        pass.gpu_ms = (tick % 97) as f32 / 8.0 + i as f32;
    }
    bins[(tick % BINS as u64) as usize] = (tick % 1000) as u32;
    let stats = Stats {
        extent: (1920, 1080),
        tick: tick * 7919,
        passes,
        memory: MemoryUse::default(),
        luminance: Some(bins),
    };
    let vertices = overlay.build(&stats).len();
    assert!(vertices > 0, "the frame under test drew nothing");
}

const BINS: usize = gg_render::luminance::BINS;

/// The criterion: after warmup, a frame of the real overlay allocates nothing.
#[test]
fn a_settled_overlay_frame_allocates_nothing() {
    let mut overlay = Overlay::default();
    let mut passes = passes();
    let mut bins = [0u32; BINS];

    // The console is a mode and its panel is half the geometry, so it is open
    // for the whole measurement — including a typed line, which is the one
    // `String` the overlay keeps across frames and must not regrow.
    overlay.key(Key::Backquote, true);
    for c in "r.exposure 1.25".chars() {
        overlay.text(c);
    }

    let cold = calls();
    for tick in 0..512 {
        frame(tick, &mut overlay, &mut passes, &mut bins);
    }

    let before = calls();
    for tick in 512..640 {
        frame(tick, &mut overlay, &mut passes, &mut bins);
    }
    let allocations = calls() - before;

    // Asserted after the window closes, because the assertion machinery itself
    // formats and would be counted.
    //
    // The first is the gate's own gate: a counter wired to the wrong allocator,
    // or optimized away, reads zero for the measured window too and this file
    // would pass while proving nothing. Warmup runs the *same* code path.
    assert!(
        before > cold,
        "the counter never moved during warmup, so it is not watching this path"
    );
    assert_eq!(
        allocations, 0,
        "128 settled overlay frames asked the allocator {allocations} times"
    );
}
