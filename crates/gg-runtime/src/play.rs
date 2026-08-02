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

    gg_platform::run(
        WindowDesc::visible_unless_headless(title, (1280, 720)),
        |window, event| {
            let verdict = match event {
                Event::WindowReady => {
                    // Mouse-look wants the pointer inside the window; Escape is
                    // what gives it back, because Escape is also what quits.
                    window.set_pointer_held(true);
                    app.attach(window).map(|()| Control::Continue)
                }
                Event::Resized(width, height) => {
                    app.resize(width, height);
                    Ok(Control::Continue)
                }
                // The instruments get first refusal, because an open console
                // has to be able to claim Escape from the arm below (§4.8).
                Event::Key { key, pressed, text } if app.debug_key(key, pressed, text) => {
                    Ok(Control::Continue)
                }
                // Escape is the *app's* key, not the sim's: quitting is not
                // simulated state and must work identically while a replay is
                // driving, which is also why it never reaches the action map.
                Event::Key {
                    key: Key::Escape,
                    pressed: true,
                    ..
                } => {
                    window.set_pointer_held(false);
                    Ok(Control::Exit)
                }
                Event::Frame => {
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
        },
    )?;

    match failure {
        Some(refused) => Err(refused),
        None => Ok(frames.frame_count()),
    }
}
