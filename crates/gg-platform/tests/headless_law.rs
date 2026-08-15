//! §1.5's machine: under `GG_HEADLESS=1` a visible-window request is a panic,
//! not a window. nextest runs each test in its own process, so the env var
//! cannot leak into other tests.
//!
//! Only the panic test is windowless — it never reaches an event loop. Every
//! test below it reaches a real window and is therefore `#[ignore]`d into
//! `cargo xtask interactive`: automated tiers are windowless *by construction*,
//! not by the window being invisible (§1.5).

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
            decorated: true,
            icon: None,
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

/// The other half of the law, and it fails in the opposite direction (§6 M47):
/// a message box is an OS window, so under `GG_HEADLESS=1` it must not be one —
/// and it must not *panic* either, because every caller is already carrying a
/// failure and a second one would bury it. Windowless: the point is that
/// nothing is constructed.
#[test]
fn an_alert_under_the_law_shows_nothing_and_says_so() {
    // SAFETY: process-per-test (nextest); no other thread reads the env yet.
    unsafe { std::env::set_var("GG_HEADLESS", "1") };
    assert!(
        !gg_platform::alert("gg", "a refusal an automated tier must never see"),
        "§1.5: an alert under the headless law is not shown"
    );
}

/// An env var is a string: `GG_HEADLESS=0` spells headless-*off*, and a parse
/// that read any set value as headless would silently skip the windowed path
/// for a user who thought they had turned it back on.
#[test]
fn zero_and_empty_read_as_not_headless() {
    // SAFETY: process-per-test (nextest); no other thread reads the env yet.
    unsafe { std::env::set_var("GG_HEADLESS", "0") };
    assert!(!gg_platform::headless(), "0 is off");
    // SAFETY: as above.
    unsafe { std::env::set_var("GG_HEADLESS", "") };
    assert!(!gg_platform::headless(), "empty is off");
    // SAFETY: as above.
    unsafe { std::env::set_var("GG_HEADLESS", "1") };
    assert!(gg_platform::headless());
    // SAFETY: as above.
    unsafe { std::env::remove_var("GG_HEADLESS") };
    assert!(!gg_platform::headless(), "unset is off");
}

/// A window is asked for at the size it wants and created at the size the
/// monitor has. Windowless — the cap is arithmetic, and the only part that
/// needs a display is the monitor it is given here as a number.
#[test]
fn a_window_never_asks_for_more_than_the_monitor_shows() {
    let desc = WindowDesc::invisible("gg cap", (1920, 1080));
    assert_eq!(desc.capped_to(None), (1920, 1080), "no monitor, no opinion");
    assert_eq!(desc.capped_to(Some((3840, 2160))), (1920, 1080), "room");
    // 1080p asked for on a 1080p screen is a window under the taskbar.
    assert_eq!(desc.capped_to(Some((1920, 1080))), (1728, 972));
    assert_eq!(desc.capped_to(Some((0, 0))), (1920, 1080), "minimized-ish");
}

/// Decoration is opt-out and nothing else moves with it: a shell that takes the
/// OS frame off is still asking for the same window at the same size, and every
/// other demo keeps the frame it always had (§6 M15.1 item 5).
#[test]
fn only_a_host_that_asks_loses_its_os_frame() {
    let plain = WindowDesc::visible_unless_headless("gg", (1280, 720));
    assert!(plain.decorated, "decorations are the default");
    let bare = plain.clone().decorations(false);
    assert!(!bare.decorated);
    assert_eq!(
        (bare.size, bare.visible, bare.resizable),
        (plain.size, plain.visible, plain.resizable)
    );
}

/// §1.5's positive half: the interactive suite's own window kind stays legal
/// under `GG_HEADLESS=1`, which is what keeps the law a rule about *visible*
/// windows rather than a ban on the suite. It creates one, so it runs where the
/// others do — on Wayland `set_visible` and `set_outer_position` are no-ops, so
/// an automated tier running this would map a surface on one platform in three.
#[test]
#[ignore = "creates a window — manual windowed suite: cargo xtask interactive (§1.5)"]
fn invisible_windows_are_always_legal() {
    // SAFETY: as above.
    unsafe { std::env::set_var("GG_HEADLESS", "1") };
    // A display-less lane (ssh, WSL without WSLg, a systemd user service, which
    // inherits neither DISPLAY nor WAYLAND_DISPLAY) cannot answer the question:
    // the legality of an invisible window is then untestable, not violated. Not
    // an environment to repair — a lane that cannot reach a display server
    // cannot put a window on the user's screen by any bug (§1.5).
    #[cfg(all(unix, not(target_os = "macos")))]
    if std::env::var_os("DISPLAY").is_none() && std::env::var_os("WAYLAND_DISPLAY").is_none() {
        eprintln!("skipped: no display server reachable (§1.5)");
        return;
    }
    let mut pump = Pump::new(WindowDesc::invisible("gg invisible ok", (320, 200)))
        .expect("invisible window under GG_HEADLESS must be legal (§1.5)");
    let _ = pump.pump();
    let (w, h) = pump.window().inner_size();
    assert!(w > 0 && h > 0);
}

/// §1.5 regression, X11 half of the same lesson: winit's un-minimize sends
/// `_NET_ACTIVE_WINDOW`, which the WM answers by *mapping* the window — under
/// WSLg that mirrors it straight onto the real Windows desktop. After the
/// storm, wherever the WM put the window, the X server must report it parked
/// off-screen. Skips when no X display is reachable (headless lanes).
#[cfg(all(unix, not(target_os = "macos")))]
#[test]
#[ignore = "storms minimize/restore, which maps a window — manual windowed suite: cargo xtask interactive (§1.5)"]
fn minimize_restore_storm_never_reaches_the_screen() {
    use raw_window_handle::HasWindowHandle;
    use x11rb::connection::Connection;
    use x11rb::protocol::xproto::ConnectionExt;

    if std::env::var_os("DISPLAY").is_none() {
        eprintln!("skipped: no X11 display (§1.5 storm needs a WM to answer)");
        return;
    }
    // Force the X11 backend: the storm is a WM code path, and it is the leg
    // the interactive suite runs (see xtask's interactive_suite for why not
    // Wayland).
    // SAFETY: process-per-test (nextest); no other thread yet.
    unsafe { std::env::remove_var("WAYLAND_DISPLAY") };

    let mut pump = Pump::new(WindowDesc::invisible("gg offscreen law", (320, 200))).unwrap();
    for _ in 0..25 {
        pump.window().set_minimized(true);
        let _ = pump.pump();
        pump.window().set_minimized(false);
        let _ = pump.pump();
    }
    // Let the WM finish answering the last activate before asking where the
    // window ended up.
    for _ in 0..10 {
        let _ = pump.pump();
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    let handle = pump.window().window_handle().unwrap();
    let xid: u32 = match handle.as_raw() {
        raw_window_handle::RawWindowHandle::Xlib(h) => h.window as u32,
        raw_window_handle::RawWindowHandle::Xcb(h) => h.window.get(),
        other => panic!("expected an X11 window, got {other:?}"),
    };

    let (conn, screen_num) = x11rb::connect(None).unwrap();
    let root = conn.setup().roots[screen_num].root;
    // translate_coordinates sees through WM reparenting: absolute root-space
    // position of the window's origin.
    let abs = conn
        .translate_coordinates(xid, root, 0, 0)
        .unwrap()
        .reply()
        .unwrap();
    assert!(
        i32::from(abs.dst_x) <= gg_platform::OFFSCREEN_POS.0 / 2,
        "after the storm the window sits at {},{} — §1.5 demands off-screen \
         (created at {:?}; the WM's map-on-activate must not move it to a monitor)",
        abs.dst_x,
        abs.dst_y,
        gg_platform::OFFSCREEN_POS,
    );
}

/// §1.5 regression, learned from the interactive suite: Win32
/// `SW_MINIMIZE`/`SW_RESTORE` *show* a window, even one created hidden. After
/// a minimize/restore storm, an invisible window must still be unreachable by
/// a human: no-activate, no taskbar/Alt-Tab presence (`WS_EX_TOOLWINDOW`),
/// and parked far off-screen where no monitor renders it.
#[cfg(windows)]
#[test]
#[ignore = "storms minimize/restore, which maps a window — manual windowed suite: cargo xtask interactive (§1.5)"]
fn minimize_restore_storm_never_reaches_the_screen() {
    use raw_window_handle::HasWindowHandle;
    use windows_sys::Win32::Foundation::RECT;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GWL_EXSTYLE, GetWindowLongPtrW, GetWindowRect, WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW,
    };

    let mut pump = Pump::new(WindowDesc::invisible("gg offscreen law", (320, 200))).unwrap();
    for _ in 0..25 {
        pump.window().set_minimized(true);
        let _ = pump.pump();
        pump.window().set_minimized(false);
        let _ = pump.pump();
    }

    let handle = pump.window().window_handle().unwrap();
    let raw_window_handle::RawWindowHandle::Win32(win32) = handle.as_raw() else {
        panic!("not a Win32 window on Windows");
    };
    let hwnd = win32.hwnd.get() as windows_sys::Win32::Foundation::HWND;
    // SAFETY: hwnd comes from the live winit window owned by this test.
    unsafe {
        let ex = GetWindowLongPtrW(hwnd, GWL_EXSTYLE);
        assert!(
            ex & WS_EX_NOACTIVATE as isize != 0,
            "invisible windows must be non-activating (§4.3)"
        );
        assert!(
            ex & WS_EX_TOOLWINDOW as isize != 0,
            "invisible windows must have no taskbar/Alt-Tab presence (§1.5)"
        );
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
        };
        assert_ne!(GetWindowRect(hwnd, &mut rect), 0);
        assert!(
            rect.left <= gg_platform::OFFSCREEN_POS.0 / 2,
            "after the storm the window sits at {},{} — §1.5 demands off-screen \
             (created at {:?}; restore must return it there, not to a monitor)",
            rect.left,
            rect.top,
            gg_platform::OFFSCREEN_POS,
        );
    }
}
