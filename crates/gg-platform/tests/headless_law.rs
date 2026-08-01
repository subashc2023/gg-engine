//! §1.5's machine: under `GG_HEADLESS=1` a visible-window request is a panic,
//! not a window. nextest runs each test in its own process, so the env var
//! cannot leak into other tests.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_platform::{Pump, WindowDesc};

#[test]
fn visible_window_under_gg_headless_panics() {
    // SAFETY: process-per-test (nextest); no other thread reads the env yet.
    unsafe { std::env::set_var("GG_HEADLESS", "1") };
    assert!(gg_platform::headless());

    let result = std::panic::catch_unwind(|| {
        let _ = Pump::new(WindowDesc {
            title: "gg headless-law violation".into(),
            size: (320, 200),
            visible: true,
            resizable: true,
        });
    });
    let payload = result.expect_err("visible window under GG_HEADLESS must panic (§1.5)");
    let message = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
        .unwrap_or_default();
    assert!(
        message.contains("GG_HEADLESS") && message.contains("1.5"),
        "panic must cite the law: {message}"
    );
}

#[test]
fn invisible_windows_are_always_legal() {
    // SAFETY: as above.
    unsafe { std::env::set_var("GG_HEADLESS", "1") };
    let mut pump = Pump::new(WindowDesc::invisible("gg invisible ok", (320, 200)))
        .expect("invisible window under GG_HEADLESS must be legal (§1.5)");
    let _ = pump.pump();
    let (w, h) = pump.window().inner_size();
    assert!(w > 0 && h > 0);
}
