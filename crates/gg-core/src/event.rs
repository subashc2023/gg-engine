//! App-level events and the queue the loop drains at `pump_events` (§4.1).
//!
//! Deliberately not the platform's event enum: `gg-platform` sits *above*
//! `gg-core` in §3's dependency direction, and the sim/replay path stays
//! winit-free by linkage (§1.5). The shell translates. Input events are not here
//! either — they go straight to `gg-input` in `poll_input`, and routing them
//! through a second queue would put a copy of the input path in the loop.
//!
//! What is left is the handful the *loop itself* has to know about, which is why
//! this is a queue rather than a callback the shell can invoke whenever an OS
//! event arrives: an event handled the moment it lands is an event handled
//! mid-tick. Queued, it is drained at one defined point in the frame.

/// Something the loop, or a stage, must react to between ticks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AppEvent {
    /// New inner size in physical pixels. `(0, 0)` means minimized — the
    /// swapchain suspends rather than recreates (§4.3).
    Resized {
        /// New inner width in physical pixels.
        width: u32,
        /// New inner height in physical pixels.
        height: u32,
    },
    /// Time to shut down: the window's close button, `Escape` in a demo, or
    /// [`Events::request_quit`] from anywhere. The loop returns
    /// [`Flow::Exit`](crate::Flow) on the frame that drains it.
    CloseRequested,
}

/// Events waiting for the next `pump_events`.
///
/// Push from anywhere in the frame; the loop drains in arrival order.
#[derive(Debug, Default)]
pub struct Events {
    queued: Vec<AppEvent>,
}

impl Events {
    /// An empty queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue an event for the next `pump_events`.
    pub fn push(&mut self, event: AppEvent) {
        self.queued.push(event);
    }

    /// Ask the loop to exit at the end of this frame.
    pub fn request_quit(&mut self) {
        self.push(AppEvent::CloseRequested);
    }

    /// Whether anything is waiting.
    pub fn is_empty(&self) -> bool {
        self.queued.is_empty()
    }

    /// Take everything queued, leaving a fresh empty queue behind.
    ///
    /// Ownership moves out rather than the loop holding a `Drain`, and that is
    /// the point: a stage handler is free to `push` while its own event is being
    /// dispatched, and what it pushes lands in the *next* frame's queue instead
    /// of being invalidated or silently cleared. The empty `Vec` this leaves
    /// costs an allocation on frames that have events at all, which resize and
    /// close-requested are not.
    pub(crate) fn take(&mut self) -> Vec<AppEvent> {
        core::mem::take(&mut self.queued)
    }
}
