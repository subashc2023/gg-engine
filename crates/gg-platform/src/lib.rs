//! `gg-platform` — the winit wrapper (§3, §4.3). Windows are born here and
//! nowhere else, so §1.5's headless law is enforced here: under `GG_HEADLESS=1`
//! (set unconditionally by `xtask ci`, the Stop hook, and the harness) a
//! request for a *visible* window panics instead of opening one. Invisible,
//! non-activating windows — the interactive suite's kind — are always legal.
//!
//! Two drivers over one window type: [`run`] owns the OS event loop for real
//! apps (demos 00–02, later `gg-runtime`), and [`Pump`] is the polling driver
//! for tests, which need to interleave window events with their own assertions.

use raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, WindowHandle,
};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::WindowAttributes;

/// Errors from the platform layer.
#[derive(Debug, thiserror::Error)]
pub enum PlatformError {
    /// The OS event loop could not be created (e.g. no display server).
    #[error("event loop unavailable: {0}")]
    EventLoop(#[from] winit::error::EventLoopError),
    /// The OS refused the window.
    #[error("window creation failed: {0}")]
    Window(#[from] winit::error::OsError),
    /// [`Pump::new`] pumped the loop but the OS never delivered a window.
    #[error("event loop pumped {0} times without producing a window")]
    NoWindow(u32),
}

/// True when this process runs under the headless law (§1.5).
///
/// Set and not `"0"`/empty: an env var is a string, and `GG_HEADLESS=0` spells
/// headless-*off* in every shell that has tried it.
pub fn headless() -> bool {
    std::env::var_os("GG_HEADLESS").is_some_and(|v| !v.is_empty() && v != "0")
}

/// Where invisible windows live (§1.5): far beyond any plausible
/// monitor arrangement, so a window the OS "shows" (minimize/restore does)
/// still never reaches a screen. Test-visible so the regression test asserts
/// against the same number.
pub const OFFSCREEN_POS: (i32, i32) = (-30000, -30000);

/// True when a position the OS *reported* is far enough out to satisfy §1.5 —
/// the same threshold the regression tests assert against.
///
/// A half-plane rather than equality with [`OFFSCREEN_POS`], because the number
/// the platform reports back is not the number we asked for: X11 moves by the
/// parent-relative *inner* position but emits `Moved` as the *outer* one, so
/// under any decorating WM an equality guard never holds and every `Moved`
/// re-parks. The same margin covers Win32's minimized-window sentinel
/// (-32000, -32000): a minimized window is already off-screen, and moving it
/// only makes Windows re-report the sentinel — `Moved` → park → `Moved`, with
/// nothing to break the loop (winit's Win32 path forwards `WINDOWPOS` verbatim,
/// with no change-detection of its own).
fn is_parked(pos: (i32, i32)) -> bool {
    pos.0 <= OFFSCREEN_POS.0 / 2 && pos.1 <= OFFSCREEN_POS.1 / 2
}

/// What kind of window to create.
#[derive(Clone, Debug)]
pub struct WindowDesc {
    /// Title (visible windows only, but always set — debuggers read it).
    pub title: String,
    /// Inner size in physical pixels.
    pub size: (u32, u32),
    /// Visible windows are for humans (§1.5); requesting one under
    /// `GG_HEADLESS=1` is a panic by design.
    pub visible: bool,
    /// Whether the window may be resized by the OS/user.
    pub resizable: bool,
    /// Whether the OS draws a frame and a title bar. `false` hands all of it to
    /// the application — including the parts nobody thinks of as decoration:
    /// the resize border, the drag region, double-click to maximize and the
    /// system snap gestures ([`Window::begin_drag`], [`Window::begin_resize`]
    /// are what a client-side title bar rebuilds them out of).
    pub decorated: bool,
}

impl WindowDesc {
    /// The interactive suite's window (§4.3): invisible, non-activating
    /// (`WS_EX_NOACTIVATE`), absent from taskbar/Alt-Tab (`WS_EX_TOOLWINDOW`),
    /// and parked off-screen on Windows — because minimize/restore *shows* a
    /// hidden window, and §1.5 holds even then. Nightly runs steal nothing
    /// and display nothing.
    pub fn invisible(title: &str, size: (u32, u32)) -> Self {
        Self {
            title: title.to_string(),
            size,
            visible: false,
            resizable: true,
            decorated: true,
        }
    }

    /// A visible window unless the process is headless, in which case the
    /// invisible variant — the standard demo-main choice.
    pub fn visible_unless_headless(title: &str, size: (u32, u32)) -> Self {
        Self {
            visible: !headless(),
            ..Self::invisible(title, size)
        }
    }

    /// The same window with or without the OS frame on it (§6 M15.1 item 5).
    ///
    /// A caller that turns it off owes the window everything the frame was
    /// doing: a drag region, a resize border, and a way to minimize, maximize
    /// and close. A bool rather than a bare `undecorated()` because the host
    /// that wants it wants it *conditionally* — the same shell draws its own
    /// bar with the editor open and takes the OS one without.
    #[must_use]
    pub fn decorations(self, decorated: bool) -> Self {
        Self { decorated, ..self }
    }

    /// [`Self::size`] cut down to what a `monitor` that size can actually show.
    ///
    /// The requested size is a *preference*: a shell that asks for 1080p on a
    /// 1080p screen gets a window whose bottom edge is under the taskbar, and
    /// the OS then clamps it somewhere less predictable than here. Nine tenths
    /// because the frame and the taskbar are outside `inner_size` and neither is
    /// measurable portably. Invisible windows are capped too — they are parked
    /// off-screen, not off-*monitor*, and a swapchain larger than the display is
    /// its own set of driver paths.
    #[must_use]
    pub fn capped_to(&self, monitor: Option<(u32, u32)>) -> (u32, u32) {
        match monitor {
            Some((w, h)) if w > 0 && h > 0 => {
                (self.size.0.min(w * 9 / 10), self.size.1.min(h * 9 / 10))
            }
            _ => self.size,
        }
    }
}

/// What the system cursor should draw, named by meaning rather than by winit's
/// `CursorIcon` (§2, Windowing row — as for keys, the translation is this
/// crate's).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CursorShape {
    /// The ordinary arrow.
    #[default]
    Arrow,
    /// A resize border, `(x, y)` in `-1..=1` — the pair [`Window::begin_resize`]
    /// takes, so a caller's hit test names the edge once and both the shape and
    /// the gesture read it.
    Resize(i8, i8),
}

/// `(x, y)` in `-1..=1` as winit names it; `None` for `(0, 0)`, which is not an
/// edge. One mapping, because a cursor that disagreed with the drag it starts is
/// worse than no cursor at all.
fn resize_direction(x: i8, y: i8) -> Option<winit::window::ResizeDirection> {
    use winit::window::ResizeDirection as R;
    Some(match (x.signum(), y.signum()) {
        (0, -1) => R::North,
        (0, 1) => R::South,
        (-1, 0) => R::West,
        (1, 0) => R::East,
        (-1, -1) => R::NorthWest,
        (1, -1) => R::NorthEast,
        (-1, 1) => R::SouthWest,
        (1, 1) => R::SouthEast,
        _ => return None,
    })
}

/// A live OS window. Wraps winit's so nothing above `gg-platform` ever names
/// winit types (§2, Windowing row).
pub struct Window {
    inner: winit::window::Window,
    /// Whether §1.5's off-screen parking applies. Carried on the window rather
    /// than re-derived from a [`WindowDesc`] at each event: both drivers must
    /// re-park, and [`Pump`]'s handler holds no desc after creation.
    parked: bool,
}

/// The X11 half of §1.5, applied at creation because these are creation-time
/// properties (see [`Window::harden_invisible`] for the Win32 half).
///
/// `_NET_WM_WINDOW_TYPE = _NET_WM_WINDOW_TYPE_UTILITY` is the EWMH hint window
/// managers and pagers read to decide taskbar/Alt-Tab presence — the X11
/// analogue of `WS_EX_TOOLWINDOW`, and the property WSLg's RAIL bridge consults
/// when deciding whether to mirror a window onto the real Windows desktop as a
/// taskbar entry. Setting it is strictly additive to the off-screen parking.
///
/// **`with_override_redirect(true)` is deliberately NOT set here.** It would be
/// the stronger fix — an override-redirect window is entirely unmanaged, so the
/// WM never maps, moves, reparents or activates it and the `_NET_ACTIVE_WINDOW`
/// that caused the recorded WSLg incident becomes a no-op at the source. The
/// reason against it is coverage, not correctness: the only windows this code
/// creates live in the manual windowed suite, whose whole job is spamming
/// resize *and minimize*, and an unmanaged window's minimize requests are
/// ignored by the WM — so X11 would silently stop exercising the swapchain
/// suspend/resume path while `interactive_resize_minimize_spam_1000` kept
/// passing (it asserts on resize counts, not minimizes). Trading a real gate
/// for a hardening that only matters while a human is watching the suite run is
/// the wrong trade. Revisit if that suite ever gains a windowless way to drive
/// the suspend path, or if WSLg mirroring proves worse than the lost coverage.
#[allow(unused_mut, unused_variables)]
fn harden_invisible_attrs(mut attrs: WindowAttributes) -> WindowAttributes {
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        use winit::platform::x11::{WindowAttributesExtX11, WindowType};
        attrs = attrs.with_x11_window_type(vec![WindowType::Utility]);
    }
    attrs
}

/// §1.5, evaluated before anything touches the OS. Hoisted above event-loop
/// construction deliberately: `Window::create` runs *inside* the loop callback,
/// so on a display-less host the loop failed first and the law degraded from a
/// panic to an ordinary `Err`. A scheduled systemd service inherits no
/// `DISPLAY` — which is the strongest §1.5 guarantee available, and exactly why
/// the law must not be reachable only when a display server exists.
fn enforce_headless_law(desc: &WindowDesc) {
    assert!(
        !(desc.visible && headless()),
        "GG_HEADLESS=1 forbids visible windows (§1.5): `{}` requested one — \
         automated paths use invisible, non-activating windows",
        desc.title
    );
}

impl Window {
    fn create(event_loop: &ActiveEventLoop, desc: &WindowDesc) -> Result<Self, PlatformError> {
        enforce_headless_law(desc); // still checked at the birth site itself
        let size = desc.capped_to(event_loop.primary_monitor().map(|m| m.size().into()));
        let mut attrs = WindowAttributes::default()
            .with_title(&desc.title)
            .with_inner_size(PhysicalSize::new(size.0, size.1))
            .with_visible(desc.visible)
            .with_resizable(desc.resizable)
            .with_decorations(desc.decorated)
            .with_active(desc.visible);
        // §1.5, learned the hard way — twice: Win32 SW_MINIMIZE/SW_RESTORE
        // *show* a window even created hidden, and on X11 (incl. WSLg's
        // XWayland) winit's un-minimize sends _NET_ACTIVE_WINDOW, which the
        // WM answers by mapping the window. The interactive suite spams
        // exactly those, so invisible windows live far off-screen on every
        // backend: OS machinery may "show" them, but no monitor ever renders
        // a pixel.
        //
        // Wayland is the honest exception, and it is worth stating precisely
        // rather than reassuringly: winit's Wayland backend implements
        // `set_visible` and `set_outer_position` as literal no-ops ("Not
        // possible on Wayland"), and `with_active` is documented Unsupported
        // there. So on Wayland *none* of this applies — invisibility, parking
        // and non-activation are all inert. What actually keeps a Wayland
        // surface off the screen is that a surface is not mapped until a buffer
        // is committed to it, and the only thing that commits one is presenting.
        // §1.5 on Wayland therefore rests on automated tiers never presenting
        // (the window-creating tests are `#[ignore]`d into the manual suite) —
        // not on anything gg-platform does. Anything that presents on Wayland
        // maps a window, and this code cannot stop it.
        if !desc.visible {
            attrs = attrs.with_position(winit::dpi::PhysicalPosition::new(
                OFFSCREEN_POS.0,
                OFFSCREEN_POS.1,
            ));
            attrs = harden_invisible_attrs(attrs);
        }
        let window = Self {
            inner: event_loop.create_window(attrs)?,
            parked: !desc.visible,
        };
        if !desc.visible {
            window.harden_invisible();
        }
        Ok(window)
    }

    /// Re-assert the off-screen parking (§1.5) from a reported position. Both
    /// drivers call it on every `Moved`; it does nothing for a visible window or
    /// a position already [`is_parked`].
    ///
    /// Parking is not a one-time act. On Windows, `SW_RESTORE` sets
    /// `WS_VISIBLE` and winit never re-hides, so after the first un-minimize
    /// the window is genuinely visible and *only* its position keeps it off a
    /// monitor — while display-topology changes (monitor hot-plug, resolution
    /// change, RDP reconnect) can relocate a window the OS considers stray.
    ///
    /// Termination rests on [`is_parked`] and not on winit gating `Moved` behind
    /// an inner-position change: a move is issued only from a position the
    /// platform reports as on-screen, and a park that took effect reports an
    /// off-screen one under either coordinate convention. Against a WM that
    /// answers each move by dragging the window back onto a monitor it does not
    /// terminate — and must not, since conceding is a window on the user's
    /// screen; each round trip is WM-paced, not a spin.
    fn repark_after_move(&self, pos: (i32, i32)) {
        if !self.parked || is_parked(pos) {
            return;
        }
        self.inner
            .set_outer_position(winit::dpi::PhysicalPosition::new(
                OFFSCREEN_POS.0,
                OFFSCREEN_POS.1,
            ));
    }

    /// The Windows half of §1.5's "never on the user's screen" for windows OS
    /// machinery may show anyway (see [`Self::create`]): `WS_EX_NOACTIVATE`
    /// (§4.3 — cannot steal focus) plus `WS_EX_TOOLWINDOW` (no taskbar button,
    /// no Alt-Tab entry). Windows-only by nature.
    ///
    /// Note winit *does* offer `with_skip_taskbar`, and it is not a substitute:
    /// it drives the ITaskbarList COM interface, which removes the taskbar
    /// button but leaves the window in Alt-Tab. `WS_EX_TOOLWINDOW` removes
    /// both, which is why this is hand-rolled. `WS_EX_NOACTIVATE` has no winit
    /// equivalent at all (`with_active` only picks `SW_SHOWNOACTIVATE` at show
    /// time and is ignored by the `SW_RESTORE` path that matters here).
    ///
    /// Panics rather than skipping if the handle is not what it must be: a
    /// §1.5 mechanism that silently no-ops is worse than one that is absent,
    /// because the tests would still pass.
    fn harden_invisible(&self) {
        #[cfg(windows)]
        {
            let handle = match self.inner.window_handle() {
                Ok(h) => h,
                Err(e) => panic!(
                    "§1.5: cannot read the window handle to harden an invisible window ({e}) — \
                     refusing to continue with an unhardened window"
                ),
            };
            let raw_window_handle::RawWindowHandle::Win32(win32) = handle.as_raw() else {
                panic!(
                    "§1.5: expected a Win32 window handle on Windows — refusing to continue \
                     with an unhardened window"
                )
            };
            use windows_sys::Win32::UI::WindowsAndMessaging::{
                GWL_EXSTYLE, GetWindowLongPtrW, SetWindowLongPtrW, WS_EX_NOACTIVATE,
                WS_EX_TOOLWINDOW,
            };
            let hwnd = win32.hwnd.get() as windows_sys::Win32::Foundation::HWND;
            // SAFETY: hwnd comes from a live winit window on this thread.
            unsafe {
                let style = GetWindowLongPtrW(hwnd, GWL_EXSTYLE);
                SetWindowLongPtrW(
                    hwnd,
                    GWL_EXSTYLE,
                    style | WS_EX_NOACTIVATE as isize | WS_EX_TOOLWINDOW as isize,
                );
            }
        }
    }

    /// Current inner size in physical pixels.
    pub fn inner_size(&self) -> (u32, u32) {
        let s = self.inner.inner_size();
        (s.width, s.height)
    }

    /// Ask the OS to resize the window. The result arrives as
    /// [`Event::Resized`]; the OS may clamp or refuse.
    pub fn request_inner_size(&self, width: u32, height: u32) {
        let _ = self
            .inner
            .request_inner_size(PhysicalSize::new(width, height));
    }

    /// Minimize or restore — the interactive suite's second torture axis.
    pub fn set_minimized(&self, minimized: bool) {
        self.inner.set_minimized(minimized);
    }

    /// Maximize or restore.
    pub fn set_maximized(&self, maximized: bool) {
        if self.parked {
            return; // §1.5: an invisible window is never given the screen
        }
        self.inner.set_maximized(maximized);
    }

    /// Whether the OS considers the window maximized.
    pub fn is_maximized(&self) -> bool {
        self.inner.is_maximized()
    }

    /// Hand the window to the system's own move loop — what pressing a title
    /// bar does (§6 M15.1 item 5).
    ///
    /// Through the OS rather than by repositioning ourselves, because the move
    /// loop is where snapping, monitor edges and multi-DPI transitions live and
    /// none of that is worth reimplementing badly. The cost is that the OS also
    /// **eats the button release**: a caller that does not synthesize one is a
    /// caller whose UI believes the mouse is still down. `gg-runtime` does it at
    /// the call site.
    ///
    /// Best effort — Wayland refuses without a serial, and a refusal is a window
    /// that did not move rather than a broken run.
    pub fn begin_drag(&self) {
        if self.parked {
            return; // §1.5: dragging a parked window is dragging it into view
        }
        let _ = self.inner.drag_window();
    }

    /// The same for a border: `(x, y)` in `-1..=1` names which edge, so
    /// `(-1, 0)` is the left one and `(1, 1)` the bottom-right corner. `(0, 0)`
    /// is not an edge and does nothing.
    ///
    /// The direction is a pair rather than an enum because the caller's hit test
    /// already *is* one — `gg-editor` compares the pointer against the surface's
    /// four sides — and translating winit's naming is this crate's job, exactly
    /// as it is for keys.
    pub fn begin_resize(&self, x: i8, y: i8) {
        if self.parked {
            return; // §1.5, as `begin_drag`
        }
        let Some(direction) = resize_direction(x, y) else {
            return;
        };
        let _ = self.inner.drag_resize_window(direction);
    }

    /// What the system cursor draws over `shape`.
    ///
    /// The one feedback an undecorated window's resize border has: it is not
    /// drawn (§6 M15.1 item 5 — a border you can see is chrome), so without the
    /// arrows nothing tells the operator the three logical units at the edge are
    /// a grip. Cheap but not free, so a caller states it on a change.
    pub fn set_cursor_shape(&self, shape: CursorShape) {
        let icon = match shape {
            CursorShape::Arrow => winit::window::CursorIcon::Default,
            CursorShape::Resize(x, y) => match resize_direction(x, y) {
                Some(direction) => direction.into(),
                None => winit::window::CursorIcon::Default,
            },
        };
        self.inner.set_cursor(icon);
    }

    /// Schedule a redraw ([`Event::RedrawRequested`]).
    pub fn request_redraw(&self) {
        self.inner.request_redraw();
    }

    /// Put the system cursor at a surface-relative position, in physical
    /// pixels.
    ///
    /// Best effort: Wayland refuses outright and every platform refuses while
    /// the window is not focused. Used at exactly one moment — handing the
    /// pointer back after a grab (§6 M15.1) — where the failure is that the
    /// arrow reappears where the OS parked it rather than where the operator
    /// left it, which is the behaviour of every build before this one.
    pub fn set_cursor_at(&self, x: f32, y: f32) {
        let _ = self
            .inner
            .set_cursor_position(winit::dpi::PhysicalPosition::new(x, y));
    }

    /// The monitor's scale factor, `1.0` at 96 DPI.
    ///
    /// What the desktop already knows about how big a row of text should be,
    /// and the only honest input to a UI scale (§6 M15.1): a resolution says
    /// how many pixels there are and says nothing about how far away they are.
    /// Cheap — winit keeps it, so a caller may ask per frame.
    pub fn scale_factor(&self) -> f32 {
        self.inner.scale_factor() as f32
    }

    /// Hold the pointer, and say whether the arrow draws at all.
    ///
    /// `held` is what mouse-look needs to be usable: [`Event::MouseMotion`] is
    /// raw device motion and arrives with or without this; what it buys is the
    /// pointer not leaving the window, which is the difference between aiming
    /// and losing focus mid-turn. Best effort by design: `Locked` is the
    /// Wayland/macOS mode and `Confined` the Windows/X11 one, no platform has
    /// both, and a refusal is a worse pointer rather than a broken run.
    ///
    /// `hidden` is the software-cursor case — a UI drawing its own arrow at the
    /// steered pointer (§4.9) — where the OS arrow must go away *without* a
    /// grab: the position keeps flowing through [`Event::CursorMoved`], which is
    /// exactly what the steer reads. A held pointer is hidden regardless; only
    /// applied over the window, which is the OS's rule and the right one.
    pub fn set_pointer(&self, held: bool, hidden: bool) {
        use winit::window::CursorGrabMode;
        let mode = if held {
            CursorGrabMode::Locked
        } else {
            CursorGrabMode::None
        };
        if self.inner.set_cursor_grab(mode).is_err() && held {
            let _ = self.inner.set_cursor_grab(CursorGrabMode::Confined);
        }
        self.inner.set_cursor_visible(!(held || hidden));
    }
}

impl HasWindowHandle for Window {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        self.inner.window_handle()
    }
}

impl HasDisplayHandle for Window {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        self.inner.display_handle()
    }
}

/// Platform events, reduced to what the engine reacts to. Raw keyboard and
/// pointer motion land here; the action maps that turn them into game verbs
/// are `gg-input`'s at M4B (§4.7).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Event {
    /// The window exists; create GPU state now.
    WindowReady,
    /// New inner size in physical pixels. `(0, 0)` means minimized —
    /// swapchains suspend rather than recreate (§4.3).
    Resized(u32, u32),
    /// Render one frame now. [`run`] drives this continuously (poll mode) —
    /// deliberately not from OS redraw requests, which Windows never delivers
    /// to invisible windows, and every automated run uses one (§1.5). OS
    /// redraw requests also map here.
    Frame,
    /// A key changed state, identified by physical position — never by the
    /// character it produces, which depends on the user's layout (§4.7: game
    /// code never sees keycodes, but the layer that maps them has to see
    /// *something* stable).
    ///
    /// Auto-repeat is filtered out: a held key reports pressed exactly once.
    Key {
        /// Which key.
        key: Key,
        /// `true` on press, `false` on release.
        pressed: bool,
        /// The character this press produced under the user's layout, when it
        /// produced one — the debug console's input (§4.8) and, since §6 M16,
        /// the editor's prompt.
        ///
        /// Carried *on* the key event rather than sent as a second one: they
        /// arrive from the OS together, and splitting them would invent an
        /// ordering question every consumer has to answer the same way.
        ///
        /// The sim never reads it. [`feed`] routes it to
        /// [`gg_input::Input::type_char`], which is *beside* the action map and
        /// not in it: a character has no verb id and reaches no [`InputFrame`],
        /// so it can be recorded (`gg_input::replay`'s text channel) without
        /// being an action.
        ///
        /// [`InputFrame`]: gg_input::InputFrame
        text: Option<char>,
    },
    /// A mouse button changed state.
    MouseButton {
        /// Which button.
        button: MouseButton,
        /// `true` on press, `false` on release.
        pressed: bool,
    },
    /// Relative pointer motion in device units, for mouse look. Not a cursor
    /// position: the fly camera wants deltas, and deltas survive the cursor
    /// hitting the edge of the screen.
    MouseMotion {
        /// Horizontal delta, right positive.
        dx: f32,
        /// Vertical delta, down positive.
        dy: f32,
    },
    /// Where the *cursor* is, in physical pixels from the surface's top-left —
    /// the other half of the pair [`MouseMotion`](Self::MouseMotion) is one of
    /// (§6 M15.1). A UI wants this and a camera wants that, and until M15.1
    /// there was only the camera's.
    ///
    /// A position rather than a delta because the OS owns where a free cursor
    /// is: acceleration, DPI and the desktop's own clamping all happen before
    /// the number arrives, and differencing positions is the only way to agree
    /// with the arrow the operator can see. [`feed`] therefore does **not**
    /// route this — turning a physical position into a surface-space delta
    /// needs the extent and the pointer the consumer already derived, neither
    /// of which is this layer's.
    CursorMoved {
        /// Distance from the left edge of the surface.
        x: f32,
        /// Distance from the top edge of the surface.
        y: f32,
    },
    /// Wheel travel, in notches, positive away from the operator.
    ///
    /// A count of detents and not a distance: `gg_input` binds the two
    /// directions as edges rather than as an axis, for the reason
    /// [`gg_input::Wheel`] gives. A trackpad's pixels are divided into notches
    /// here, because this is the layer that knows which unit arrived.
    MouseWheel {
        /// Notches this event carried; fractional from a pixel-precise device.
        notches: f32,
    },
    /// The user asked the window to close.
    CloseRequested,
    /// The loop is over and the window is about to be destroyed — the last
    /// moment anything holding a surface derived from it may tear down.
    ///
    /// Delivered exactly once, after the event loop returns (for any reason,
    /// including an error) and before the window drops, whenever a window was
    /// created at all. Handlers return a [`Control`] verdict as usual and it is
    /// ignored; there is nothing left to control.
    ///
    /// This exists because `ash_window`'s surface creation requires the window
    /// to outlive the `VkSurfaceKHR` derived from it, and a handler that tears
    /// down only on [`Event::CloseRequested`] misses every failure path. Doing
    /// it after [`run`] returns is too late: the window is already gone.
    Exiting,
}

/// What one detent is worth to a device that reports pixels: the desktop
/// convention of three lines of text, a line being `gg_ui::font`'s cell at the
/// usual UI scale — which this crate may not name, sitting below `gg-ui`, and
/// so states as a number.
const PIXELS_PER_NOTCH: f32 = 48.0;

/// Physical key and mouse-button identity live in `gg-input` (§4.7), one layer
/// *below* this crate: the sim and replay path binds against them and must stay
/// winit-free by linkage (§1.5). Re-exported so callers still name one crate.
pub use gg_input::{Key, MouseButton};

/// Apply a raw input event to [`gg_input::Input`]. `false` for events that carry
/// no input — a window resize, a frame, a close request — which the caller was
/// going to route somewhere else anyway.
///
/// This is the translation half of the seam above, and it belongs on this side
/// of it for the same reason [`key_from_winit`] does: `gg-input` sits *below*
/// this crate and has never heard of an [`Event`]. Without it every host writes
/// the same three-arm match, which is how an app shell acquires input logic it
/// has no charter for (§3).
///
/// Keys the *app* owns — Escape, in the shell — must be handled before this is
/// called: quitting is not simulated state and must not reach the action map.
/// [`Event::CursorMoved`] is unclaimed for its own documented reason.
pub fn feed(input: &mut gg_input::Input, event: &Event) -> bool {
    match *event {
        Event::Key { key, pressed, text } => {
            // Both, not one or the other (§6 M16). The key is a verb and goes to
            // the action map; the character goes to a buffer no verb can reach,
            // for a host with somewhere to put text. A key that is both — every
            // letter, once the editor's prompt has focus — feeds both, which is
            // why `Input::type_char` and `Input::key` are separate calls rather
            // than a branch.
            if let Some(c) = text.filter(|_| pressed) {
                input.type_char(c);
            }
            input.key(key, pressed);
        }
        Event::MouseButton { button, pressed } => input.mouse_button(button, pressed),
        Event::MouseMotion { dx, dy } => input.motion(dx, dy),
        Event::MouseWheel { notches } => input.wheel(notches),
        _ => return false,
    }
    true
}

#[cfg(test)]
mod feed_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::{Event, Key, MouseButton, feed};

    /// The one thing a caller relies on: which events this claims and which it
    /// leaves alone. A shell routes the rest itself — Escape is the app's key,
    /// not the sim's — so an event silently claimed here would be an event that
    /// never reaches the code that was supposed to handle it.
    ///
    /// Windowless on purpose (§1.5): the real event stream needs a window, so
    /// without this the routing table would be covered by the manual suite alone.
    #[test]
    fn feed_claims_input_events_and_nothing_else() {
        let map = gg_input::ActionMap::parse("", &[], &[]).unwrap();
        let mut input = gg_input::Input::new(map);
        for event in [
            Event::Key {
                key: Key::W,
                pressed: true,
                text: Some('w'),
            },
            Event::MouseButton {
                button: MouseButton::Left,
                pressed: true,
            },
            Event::MouseMotion { dx: 1.0, dy: -2.0 },
            Event::MouseWheel { notches: 1.0 },
        ] {
            assert!(feed(&mut input, &event), "unclaimed: {event:?}");
        }
        for event in [
            Event::WindowReady,
            Event::Resized(1, 2),
            Event::Frame,
            Event::CloseRequested,
            Event::Exiting,
            // Deliberately unclaimed: a cursor position is not a delta, and
            // what it becomes needs an extent this layer does not have (§6
            // M15.1). A `feed` that guessed would put the arrow in the wrong
            // place at every window size but one.
            Event::CursorMoved { x: 4.0, y: 8.0 },
        ] {
            assert!(!feed(&mut input, &event), "wrongly claimed: {event:?}");
        }
    }

    /// §6 M16: the character and the key both, and the character only where
    /// something asked for it.
    ///
    /// The release half is the one worth a test: a character is produced by
    /// *pressing*, and a `text` on a release would double every keystroke in a
    /// prompt — which is exactly the kind of thing that looks like a stuck key
    /// and is actually a routing bug two crates away.
    #[test]
    fn a_press_feeds_its_character_and_its_key_and_a_release_feeds_only_the_key() {
        let map = gg_input::ActionMap::parse("", &[], &[]).unwrap();
        let mut input = gg_input::Input::new(map);
        for (pressed, text) in [(true, Some('h')), (false, Some('h')), (true, Some('i'))] {
            feed(
                &mut input,
                &Event::Key {
                    key: Key::W,
                    pressed,
                    text,
                },
            );
        }
        assert_eq!(input.take_typed(true), "hi");
        // And a drain with nothing listening keeps nothing for later.
        feed(
            &mut input,
            &Event::Key {
                key: Key::W,
                pressed: true,
                text: Some('x'),
            },
        );
        assert_eq!(input.take_typed(false), "");
        assert_eq!(input.take_typed(true), "", "a dropped tick is not deferred");
    }
}

/// Translate a winit key code. The match is the seam between winit's naming and
/// ours, and it lives here because this is the only crate that has heard of
/// winit — `gg-input` owns the names, `gg-platform` owns the translation.
fn key_from_winit(code: winit::keyboard::KeyCode) -> Option<Key> {
    use winit::keyboard::KeyCode as K;
    Some(match code {
        K::KeyA => Key::A,
        K::KeyB => Key::B,
        K::KeyC => Key::C,
        K::KeyD => Key::D,
        K::KeyE => Key::E,
        K::KeyF => Key::F,
        K::KeyG => Key::G,
        K::KeyH => Key::H,
        K::KeyI => Key::I,
        K::KeyJ => Key::J,
        K::KeyK => Key::K,
        K::KeyL => Key::L,
        K::KeyM => Key::M,
        K::KeyN => Key::N,
        K::KeyO => Key::O,
        K::KeyP => Key::P,
        K::KeyQ => Key::Q,
        K::KeyR => Key::R,
        K::KeyS => Key::S,
        K::KeyT => Key::T,
        K::KeyU => Key::U,
        K::KeyV => Key::V,
        K::KeyW => Key::W,
        K::KeyX => Key::X,
        K::KeyY => Key::Y,
        K::KeyZ => Key::Z,
        K::Digit0 => Key::Digit0,
        K::Digit1 => Key::Digit1,
        K::Digit2 => Key::Digit2,
        K::Digit3 => Key::Digit3,
        K::Digit4 => Key::Digit4,
        K::Digit5 => Key::Digit5,
        K::Digit6 => Key::Digit6,
        K::Digit7 => Key::Digit7,
        K::Digit8 => Key::Digit8,
        K::Digit9 => Key::Digit9,
        K::F1 => Key::F1,
        K::F2 => Key::F2,
        K::F3 => Key::F3,
        K::F4 => Key::F4,
        K::F5 => Key::F5,
        K::F6 => Key::F6,
        K::F7 => Key::F7,
        K::F8 => Key::F8,
        K::F9 => Key::F9,
        K::F10 => Key::F10,
        K::F11 => Key::F11,
        K::F12 => Key::F12,
        K::ArrowLeft => Key::Left,
        K::ArrowRight => Key::Right,
        K::ArrowUp => Key::Up,
        K::ArrowDown => Key::Down,
        K::Space => Key::Space,
        K::Enter => Key::Enter,
        K::Escape => Key::Escape,
        K::Tab => Key::Tab,
        K::Backspace => Key::Backspace,
        K::Delete => Key::Delete,
        K::Minus => Key::Minus,
        K::Equal => Key::Equal,
        K::BracketLeft => Key::BracketLeft,
        K::BracketRight => Key::BracketRight,
        K::Backslash => Key::Backslash,
        K::Semicolon => Key::Semicolon,
        K::Quote => Key::Quote,
        K::Comma => Key::Comma,
        K::Period => Key::Period,
        K::Slash => Key::Slash,
        K::Backquote => Key::Backquote,
        K::ShiftLeft => Key::ShiftLeft,
        K::ShiftRight => Key::ShiftRight,
        K::ControlLeft => Key::ControlLeft,
        K::ControlRight => Key::ControlRight,
        K::AltLeft => Key::AltLeft,
        K::AltRight => Key::AltRight,
        K::SuperLeft => Key::SuperLeft,
        K::SuperRight => Key::SuperRight,
        K::Home => Key::Home,
        K::End => Key::End,
        K::PageUp => Key::PageUp,
        K::PageDown => Key::PageDown,
        K::Insert => Key::Insert,
        _ => return None,
    })
}

/// Translate a winit mouse button. Buttons past the OS's numbering are dropped
/// rather than guessed at.
fn button_from_winit(button: winit::event::MouseButton) -> Option<MouseButton> {
    use winit::event::MouseButton as B;
    Some(match button {
        B::Left => MouseButton::Left,
        B::Right => MouseButton::Right,
        B::Middle => MouseButton::Middle,
        B::Back => MouseButton::Extra(0),
        B::Forward => MouseButton::Extra(1),
        B::Other(n) => MouseButton::Extra(u8::try_from(n).ok()?),
    })
}

/// Translate a winit key event, dropping auto-repeat and keys outside [`Key`].
fn key_event(event: &winit::event::KeyEvent) -> Option<Event> {
    if event.repeat {
        return None;
    }
    let winit::keyboard::PhysicalKey::Code(code) = event.physical_key else {
        return None;
    };
    key_from_winit(code).map(|key| Event::Key {
        key,
        pressed: event.state.is_pressed(),
        // One character, not a string: a dead key or an IME commit can produce
        // several, and routing those correctly is text *input* rather than a
        // debug console's keystrokes (§4.9 rents shaping at M13). Releases
        // carry none — a character is produced by pressing.
        text: event
            .state
            .is_pressed()
            .then(|| event.text.as_ref()?.chars().next())
            .flatten(),
    })
}

/// Handler verdict for [`run`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Control {
    /// Keep running.
    Continue,
    /// Leave the event loop cleanly.
    Exit,
}

struct App<'h> {
    desc: WindowDesc,
    window: Option<Window>,
    handler: &'h mut dyn FnMut(&Window, Event) -> Control,
    result: Result<(), PlatformError>,
}

impl App<'_> {
    fn dispatch(&mut self, event_loop: &ActiveEventLoop, event: Event) {
        if let Some(window) = &self.window
            && (self.handler)(window, event) == Control::Exit
        {
            event_loop.exit();
        }
    }

    /// Deliver [`Event::Exiting`] while the window is still alive. Separate
    /// from `dispatch` because there is no `ActiveEventLoop` once `run_app`
    /// has returned, and no verdict worth honoring — the loop is already over.
    fn dispatch_exiting(&mut self) {
        if let Some(window) = &self.window {
            let _ = (self.handler)(window, Event::Exiting);
        }
    }
}

impl ApplicationHandler for App<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            match Window::create(event_loop, &self.desc) {
                Ok(window) => {
                    self.window = Some(window);
                    self.dispatch(event_loop, Event::WindowReady);
                }
                Err(err) => {
                    self.result = Err(err);
                    event_loop.exit();
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let event = match event {
            WindowEvent::Resized(size) => Event::Resized(size.width, size.height),
            WindowEvent::CloseRequested => Event::CloseRequested,
            WindowEvent::RedrawRequested => Event::Frame,
            WindowEvent::KeyboardInput { event, .. } => match key_event(&event) {
                Some(event) => event,
                None => return,
            },
            WindowEvent::MouseInput { state, button, .. } => match button_from_winit(button) {
                Some(button) => Event::MouseButton {
                    button,
                    pressed: state.is_pressed(),
                },
                None => return,
            },
            WindowEvent::CursorMoved { position, .. } => Event::CursorMoved {
                x: position.x as f32,
                y: position.y as f32,
            },
            // Both units become notches here, which is the layer that knows
            // which one arrived: a wheel reports lines and a trackpad reports
            // pixels, and `gg_input::Wheel` counts detents. The divisor is a
            // line of text at the fallback cell's height — nothing above has a
            // font to ask, and the alternative is a `dy` whose meaning depends
            // on the pointing device.
            WindowEvent::MouseWheel { delta, .. } => Event::MouseWheel {
                notches: match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(at) => {
                        at.y as f32 / PIXELS_PER_NOTCH
                    }
                },
            },
            // §1.5: an invisible window that moved is a window whose parking
            // the OS just undid — re-park it. Not reported upward: nothing
            // above gg-platform has any business knowing where a window that
            // must never be seen happens to sit.
            WindowEvent::Moved(pos) => {
                if let Some(window) = &self.window {
                    window.repark_after_move((pos.x, pos.y));
                }
                return;
            }
            _ => return,
        };
        self.dispatch(event_loop, event);
    }

    /// Raw device motion, which is a different thing from the cursor position
    /// `window_event` reports and not a substitute for it: mouse look wants a
    /// delta that keeps arriving once the pointer is against the edge of the
    /// screen, and a UI wants the arrow the operator can see.
    fn device_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if let winit::event::DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            self.dispatch(
                event_loop,
                Event::MouseMotion {
                    dx: dx as f32,
                    dy: dy as f32,
                },
            );
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Continuous rendering: one frame per loop iteration in poll mode.
        // Presentation paces the loop; real frame pacing lands with the loop
        // skeleton (§4.1).
        self.dispatch(event_loop, Event::Frame);
    }
}

/// Run the OS event loop until the handler returns [`Control::Exit`] or the
/// window closes. The handler sees [`Event::WindowReady`] first.
///
/// Panics on a visible `desc` under `GG_HEADLESS=1` (§1.5), before the event
/// loop is built — so the law holds with or without a display server.
pub fn run(
    desc: WindowDesc,
    mut handler: impl FnMut(&Window, Event) -> Control,
) -> Result<(), PlatformError> {
    enforce_headless_law(&desc);
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        desc,
        window: None,
        handler: &mut handler,
        result: Ok(()),
    };
    // Dispatch Exiting before propagating any loop error: a failed run is
    // exactly the case where the handler never reached its own teardown, and
    // the window is still alive here but not one line later.
    let loop_result = event_loop.run_app(&mut app);
    app.dispatch_exiting();
    loop_result?;
    app.result
}

/// The polling driver for tests: create a window, then interleave
/// [`Pump::pump`] with assertions. Runs off the main thread (nextest runs
/// tests on worker threads), which winit permits on Windows/X11/Wayland.
pub struct Pump {
    // Field order is drop order: the window dies before its event loop.
    window: Window,
    events: Vec<Event>,
    event_loop: EventLoop<()>,
}

struct PumpApp<'a> {
    /// Present only during [`Pump::new`]'s creation pumps; later pumps never
    /// create windows.
    create: Option<PumpCreate<'a>>,
    /// The window once [`Pump`] owns it; `None` during [`Pump::new`], where it
    /// still lives inside `create`.
    window: Option<&'a Window>,
    events: &'a mut Vec<Event>,
}

impl PumpApp<'_> {
    /// The live window, from whichever of the two places currently holds it.
    fn live_window(&self) -> Option<&Window> {
        match &self.create {
            Some(create) => create.window.as_ref(),
            None => self.window,
        }
    }
}

struct PumpCreate<'a> {
    desc: &'a WindowDesc,
    window: &'a mut Option<Window>,
    error: &'a mut Option<PlatformError>,
}

impl ApplicationHandler for PumpApp<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(create) = &mut self.create
            && create.window.is_none()
        {
            match Window::create(event_loop, create.desc) {
                Ok(window) => *create.window = Some(window),
                Err(err) => {
                    *create.error = Some(err);
                    event_loop.exit();
                }
            }
        }
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::Resized(size) => self.events.push(Event::Resized(size.width, size.height)),
            WindowEvent::CloseRequested => self.events.push(Event::CloseRequested),
            WindowEvent::RedrawRequested => self.events.push(Event::Frame),
            // §1.5's re-park, and it has to exist here as well as in App:
            // every window-creating test drives Pump, so a mechanism living
            // only on the `run` path is one no test can catch no-opping — and
            // the storms that undo the parking are exactly the tests that run
            // here. Never surfaced as an Event; see App's arm.
            WindowEvent::Moved(pos) => {
                if let Some(window) = self.live_window() {
                    window.repark_after_move((pos.x, pos.y));
                }
            }
            _ => {}
        }
    }
}

impl Pump {
    /// Create the event loop and window, pumping until the OS delivers it.
    ///
    /// Panics on a visible `desc` under `GG_HEADLESS=1` (§1.5), before the
    /// event loop is built — so the law holds with or without a display server.
    pub fn new(desc: WindowDesc) -> Result<Self, PlatformError> {
        enforce_headless_law(&desc);
        let mut builder = EventLoop::builder();
        #[cfg(windows)]
        winit::platform::windows::EventLoopBuilderExtWindows::with_any_thread(&mut builder, true);
        #[cfg(all(unix, not(target_os = "macos")))]
        {
            winit::platform::x11::EventLoopBuilderExtX11::with_any_thread(&mut builder, true);
            winit::platform::wayland::EventLoopBuilderExtWayland::with_any_thread(
                &mut builder,
                true,
            );
        }
        let mut event_loop = builder.build()?;

        let mut window = None;
        let mut events = Vec::new();
        let mut error = None;
        const ATTEMPTS: u32 = 100;
        for _ in 0..ATTEMPTS {
            let mut app = PumpApp {
                create: Some(PumpCreate {
                    desc: &desc,
                    window: &mut window,
                    error: &mut error,
                }),
                window: None,
                events: &mut events,
            };
            winit::platform::pump_events::EventLoopExtPumpEvents::pump_app_events(
                &mut event_loop,
                Some(std::time::Duration::ZERO),
                &mut app,
            );
            if let Some(err) = error {
                return Err(err);
            }
            if window.is_some() {
                break;
            }
        }
        let window = window.ok_or(PlatformError::NoWindow(ATTEMPTS))?;
        Ok(Self {
            window,
            events,
            event_loop,
        })
    }

    /// The window under test.
    pub fn window(&self) -> &Window {
        &self.window
    }

    /// Deliver pending OS events; returns what arrived since the last pump.
    pub fn pump(&mut self) -> Vec<Event> {
        let mut app = PumpApp {
            create: None,
            window: Some(&self.window),
            events: &mut self.events,
        };
        winit::platform::pump_events::EventLoopExtPumpEvents::pump_app_events(
            &mut self.event_loop,
            Some(std::time::Duration::ZERO),
            &mut app,
        );
        std::mem::take(&mut self.events)
    }
}
