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
pub fn headless() -> bool {
    std::env::var_os("GG_HEADLESS").is_some()
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
}

impl WindowDesc {
    /// The interactive suite's window (§4.3): invisible and non-activating
    /// (`WS_EX_NOACTIVATE` on Windows), so even nightly runs steal nothing.
    pub fn invisible(title: &str, size: (u32, u32)) -> Self {
        Self {
            title: title.to_string(),
            size,
            visible: false,
            resizable: true,
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
}

/// A live OS window. Wraps winit's so nothing above `gg-platform` ever names
/// winit types (§2, Windowing row).
pub struct Window {
    inner: winit::window::Window,
}

impl Window {
    fn create(event_loop: &ActiveEventLoop, desc: &WindowDesc) -> Result<Self, PlatformError> {
        assert!(
            !(desc.visible && headless()),
            "GG_HEADLESS=1 forbids visible windows (§1.5): `{}` requested one — \
             automated paths use invisible, non-activating windows",
            desc.title
        );
        let attrs = WindowAttributes::default()
            .with_title(&desc.title)
            .with_inner_size(PhysicalSize::new(desc.size.0, desc.size.1))
            .with_visible(desc.visible)
            .with_resizable(desc.resizable)
            .with_active(desc.visible);
        let window = Self {
            inner: event_loop.create_window(attrs)?,
        };
        if !desc.visible {
            window.set_no_activate();
        }
        Ok(window)
    }

    /// `WS_EX_NOACTIVATE` (§4.3): should an invisible window ever be shown by
    /// OS machinery, it still cannot steal focus. Windows-only by nature.
    fn set_no_activate(&self) {
        #[cfg(windows)]
        if let Ok(handle) = self.inner.window_handle()
            && let raw_window_handle::RawWindowHandle::Win32(win32) = handle.as_raw()
        {
            use windows_sys::Win32::UI::WindowsAndMessaging::{
                GWL_EXSTYLE, GetWindowLongPtrW, SetWindowLongPtrW, WS_EX_NOACTIVATE,
            };
            let hwnd = win32.hwnd.get() as windows_sys::Win32::Foundation::HWND;
            // SAFETY: hwnd comes from a live winit window on this thread.
            unsafe {
                let style = GetWindowLongPtrW(hwnd, GWL_EXSTYLE);
                SetWindowLongPtrW(hwnd, GWL_EXSTYLE, style | WS_EX_NOACTIVATE as isize);
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

    /// Schedule a redraw ([`Event::RedrawRequested`]).
    pub fn request_redraw(&self) {
        self.inner.request_redraw();
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

/// Platform events, reduced to what the engine reacts to. Raw input joins at
/// its milestone via `gg-input`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    /// The user asked the window to close.
    CloseRequested,
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
            _ => return,
        };
        self.dispatch(event_loop, event);
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
pub fn run(
    desc: WindowDesc,
    mut handler: impl FnMut(&Window, Event) -> Control,
) -> Result<(), PlatformError> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        desc,
        window: None,
        handler: &mut handler,
        result: Ok(()),
    };
    event_loop.run_app(&mut app)?;
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
    events: &'a mut Vec<Event>,
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
            _ => {}
        }
    }
}

impl Pump {
    /// Create the event loop and window, pumping until the OS delivers it.
    pub fn new(desc: WindowDesc) -> Result<Self, PlatformError> {
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
