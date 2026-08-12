//! The windowed run: the OS event loop on the outside, `gg-core`'s frame on the
//! inside (§4.1).
//!
//! `gg-core` deliberately does not own the event loop, so this is the half that
//! decides *when* a frame happens — and the only file in the shell that has
//! heard of a window. Never reached by an automated tier (§1.5): a headless run
//! never gets here.

use std::time::Instant;

use gg_core::{Flow, FrameLoop};
use gg_platform::{Control, Event, Key, WindowDesc};

use crate::app::App;

/// Play until the window closes, the game asks to stop, or `target` frames have
/// run. Returns the frames presented.
pub fn play(app: &mut App, title: &str, target: Option<u64>) -> anyhow::Result<u64> {
    // Realtime, not locked: a windowed run is paced by the wall clock and the
    // catch-up guard is the clock's (§4.1).
    // Resuming at zero, or where a predecessor stopped: a rejuvenated session
    // resumes the sim clock rather than restarting it under a world that
    // remembers (§4.2.2).
    let mut frames = FrameLoop::default().resuming_at(app.next_tick());
    let mut last = Instant::now();
    let mut failure: Option<anyhow::Error> = None;
    // What the window was last told — held, and OS arrow hidden — so the grab
    // is a syscall on a change and not sixty a second. Starts unset: the first
    // frame always states it.
    let mut pointer_state: Option<(bool, bool)> = None;
    // The same, for the resize border's arrows (§6 M15.1 item 5).
    #[cfg(feature = "editor")]
    let mut cursor_shape: Option<gg_platform::CursorShape> = None;

    // 1080p, capped to the monitor by `WindowDesc::capped_to`. The editor is
    // why it is not 720p any more: its canvas is the window divided by a whole
    // `gg_editor::ui_scale`, so a small window is a small *working area*
    // however many pixels the screen has.
    // The editor draws its own title bar, so it takes the OS one off (§6 M15.1
    // item 5) — and only the editor: a demo has no bar to replace one with, and
    // an undecorated window with nothing drawing a close button is a window
    // nobody can close.
    let desc = WindowDesc::visible_unless_headless(title, (1920, 1080));
    gg_platform::run(desc.decorations(!app.editing()), |window, event| {
        let verdict = match event {
            // The monitor's scale factor at both edges that can change it: a
            // window that has just been made, and one that resized — which is
            // what winit sends after a drag onto a second monitor (§6 M15.1).
            Event::WindowReady => {
                app.set_dpi(window.scale_factor());
                app.attach(window).map(|()| Control::Continue)
            }
            // Painted here rather than left to the next `Event::Frame`, because
            // during a drag there is no next one: Win32 runs the resize inside
            // `DefWindowProc`'s own modal loop, so winit's loop — and with it
            // the `about_to_wait` that produces `Event::Frame` — is stopped from
            // button-down to button-up. `Rhi::resize` only raises a flag, and
            // nothing erases the client area winit just uncovered (its window
            // class carries a null background brush), so a deferred recreate
            // shows as a black band trailing the border.
            //
            // The real elapsed and not zero, which is what a tick costs and what
            // this needs: the editor's layout — the dock, and with it the game's
            // `viewport_rect` — is rebuilt by `Editor::tick`, which runs only
            // inside a sim tick. A resize paint advancing no tick presents a
            // correctly sized swapchain with the *previous* window's layout
            // drawn on it, and the strip the drag just added has nothing in it.
            // Frames painted this way still count toward `--frames`, which is
            // the honest total — they are presentations.
            Event::Resized(width, height) => {
                app.set_dpi(window.scale_factor());
                app.resize(width, height);
                let now = Instant::now();
                let elapsed = now.duration_since(last);
                last = now;
                // Minimized: the swapchain is suspended and there is nothing to
                // present into (§4.3).
                let painted = match width == 0 || height == 0 {
                    true => Ok(Flow::Continue),
                    false => frames.frame(app, elapsed),
                };
                // Off by default (§4.8's convention). What it answers: whether
                // Windows delivers a size *during* the modal drag at all, and
                // whether the swapchain followed it — the two halves of a black
                // band that only clears on button-up.
                //
                // `paint_us` is what decides the band that survives all of the
                // above. A paint costing most of a refresh is the FIFO present
                // waiting for vblank, and the fix is the present mode; a paint
                // costing a millisecond means the frame was ready and the
                // compositor showed the enlarged window before it, which no
                // amount of engine speed reaches.
                tracing::debug!(
                    target: "gg::resize",
                    width,
                    height,
                    painted = painted.is_ok() && width != 0 && height != 0,
                    swapchain = ?app.surface_extent(),
                    since_us = elapsed.as_micros(),
                    paint_us = now.elapsed().as_micros(),
                    "resized"
                );
                painted.map(|flow| match flow == Flow::Exit {
                    true => Control::Exit,
                    false => Control::Continue,
                })
            }
            // The instruments get first refusal, because an open console
            // has to be able to claim Escape from the arm below (§4.8).
            Event::Key { key, pressed, text } if app.debug_key(key, pressed, text) => {
                Ok(Control::Continue)
            }
            // Escape is the *app's* key, not the sim's: quitting is not
            // simulated state and must work identically while a replay is
            // driving, which is also why it never reaches the action map.
            //
            // It releases a held pointer *first* and quits only when there
            // was nothing to release (§6 M15.1). One key for both, in that
            // order, is what every game does — and the editor is why there
            // is now something for it to mean besides "quit".
            //
            // Unless the game's map binds it, in which case it is not the
            // app's key at all and falls through to `feed` like any other
            // (`ActionMap::claims`): a game with a pause menu owns Escape,
            // and a pause that arrived as a verb is in the replay where a
            // quit never was.
            Event::Key {
                key: Key::Escape,
                pressed: true,
                ..
            } if !app.input().claims_key(Key::Escape) => match app.release_pointer() {
                Some((x, y)) => {
                    window.set_cursor_at(x, y);
                    Ok(Control::Continue)
                }
                None => Ok(Control::Exit),
            },
            // Not `gg_platform::feed`'s: turning a physical position into a
            // canvas delta needs the surface, which is the app's.
            Event::CursorMoved { x, y } => {
                app.cursor_at(x, y);
                Ok(Control::Continue)
            }
            // Forget what the window was told, so the next frame tells it again
            // (§6 M15.1). The OS drops a grab and a cursor-hide asked for while
            // the window was unfocused, and Windows releases the clip on focus
            // *loss* — so without this the first statement is a race against a
            // focus that had not arrived, and alt-tabbing back is a game that
            // silently stopped holding the mouse. Cheap: focus changes are rare,
            // and the arm below still only calls through on a difference.
            Event::Focused(true) => {
                pointer_state = None;
                Ok(Control::Continue)
            }
            Event::Frame => {
                // What the editor's title bar asked for last tick (§6 M15.1
                // item 5). Here for `pointer_held`'s reason: the decision is
                // a tick's and the window is only reachable from this
                // closure. `Close` is the one that answers with a verdict.
                #[cfg(feature = "editor")]
                if let Some(command) = app.take_window_command()
                    && window_command(window, app.input(), command) == Control::Exit
                {
                    app.detach();
                    return Control::Exit;
                }
                // Asked every frame rather than tracked through the command
                // above: the operator can maximize by other means (a snap
                // gesture, the keyboard) and the button must still draw what
                // the window *is*.
                #[cfg(feature = "editor")]
                app.set_maximized(window.is_maximized());
                // The undecorated window's border is invisible by design, so
                // the cursor is what says it is there at all (§6 M15.1 item 5).
                #[cfg(feature = "editor")]
                {
                    let shape = match app.resize_edge() {
                        Some(edge) => gg_platform::CursorShape::Resize(edge.x, edge.y),
                        None => gg_platform::CursorShape::Arrow,
                    };
                    if cursor_shape != Some(shape) {
                        cursor_shape = Some(shape);
                        window.set_cursor_shape(shape);
                    }
                }
                // Applied here rather than at the edge that changed it: the
                // decision is a sim tick's (a press in the viewport, a UI that
                // drew its own arrow) and the window is only reachable from
                // this closure.
                if pointer_state != Some(app.pointer()) {
                    pointer_state = Some(app.pointer());
                    window.set_pointer(app.pointer().0, app.pointer().1);
                }
                let now = Instant::now();
                let elapsed = now.duration_since(last);
                last = now;
                frames.frame(app, elapsed).map(|flow| {
                    let done = target.is_some_and(|target| frames.frame_count() >= target);
                    if flow == Flow::Exit || done {
                        Control::Exit
                    } else {
                        Control::Continue
                    }
                })
            }
            Event::CloseRequested | Event::Exiting => Ok(Control::Exit),
            // Raw input, translated by the layer that owns the translation
            // (§4.7). Reached only by events the arms above did not claim,
            // which is what keeps Escape out of the action map.
            raw => {
                gg_platform::feed(app.input(), &raw);
                Ok(Control::Continue)
            }
        };
        match verdict {
            Ok(Control::Continue) => Control::Continue,
            Ok(Control::Exit) => {
                // The surface may not outlive the window, and this closure is
                // the last place where both are alive (§4.3). `Exiting` is
                // the backstop for the paths that never returned Exit.
                app.detach();
                Control::Exit
            }
            Err(refused) => {
                failure = Some(refused);
                app.detach();
                Control::Exit
            }
        }
    })?;

    match failure {
        Some(refused) => Err(refused),
        None => Ok(frames.frame_count()),
    }
}

/// Do to the window what the editor's title bar asked (§6 M15.1 item 5).
///
/// The synthesized release is not a workaround for a bug: `drag_window` and
/// `drag_resize_window` hand the pointer to the system's own modal loop, which
/// consumes the button-up that ends it. Without this the shell would return from
/// a drag believing the mouse is still down — and it belongs *here*, on the same
/// side of the seam as the syscall that ate it, so the editor's router keeps
/// deriving its edges from one ordinary input stream (§4.7).
#[cfg(feature = "editor")]
fn window_command(
    window: &gg_platform::Window,
    input: &mut gg_input::Input,
    command: gg_editor::WindowCommand,
) -> Control {
    use gg_editor::WindowCommand as Command;
    match command {
        Command::Drag => window.begin_drag(),
        Command::Resize(edge) => window.begin_resize(edge.x, edge.y),
        Command::Minimize => window.set_minimized(true),
        Command::ToggleMaximize => window.set_maximized(!window.is_maximized()),
        Command::Close => return Control::Exit,
    }
    input.mouse_button(gg_platform::MouseButton::Left, false);
    Control::Continue
}
