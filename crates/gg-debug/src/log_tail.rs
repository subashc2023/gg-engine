//! The last few hundred log lines, kept so a crash report can say what the
//! process was doing (§4.8).
//!
//! A `tracing` layer rather than a second logging call at each site: anything
//! that reaches the terminal reaches the tail, and there is no way to log
//! something the report will not see.
//!
//! It formats eagerly, on the logging thread. The alternative — keeping events
//! and rendering them at crash time — would mean allocating and walking a graph
//! of borrowed field values inside a panic hook, which is where the fewest
//! things are guaranteed to still work.

use std::collections::VecDeque;
use std::fmt::Write as _;
use std::sync::Mutex;

use tracing::field::{Field, Visit};
use tracing_subscriber::layer::{Context, Layer};

/// Lines kept. Sized to cover a few seconds of a noisy frame loop, which is the
/// window a crash report is actually read for; more is archaeology nobody does.
const CAPACITY: usize = 256;

/// A poisoned lock means a formatter panicked mid-push. The deque is still a
/// deque, so recovering beats losing the tail exactly when it is wanted.
static TAIL: Mutex<VecDeque<String>> = Mutex::new(VecDeque::new());

/// The layer that fills it. One per process; installing two would double every
/// line rather than fail.
pub(crate) struct LogTail;

impl<S: tracing::Subscriber> Layer<S> for LogTail {
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let meta = event.metadata();
        let mut line = format!("{:>5} {}:", meta.level(), meta.target());
        event.record(&mut Fields(&mut line));
        let mut tail = TAIL.lock().unwrap_or_else(|e| e.into_inner());
        if tail.len() == CAPACITY {
            tail.pop_front();
        }
        tail.push_back(line);
    }
}

/// Everything the tail holds, oldest first.
pub fn tail() -> Vec<String> {
    TAIL.lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .cloned()
        .collect()
}

/// How many lines are held right now.
pub fn len() -> usize {
    TAIL.lock().unwrap_or_else(|e| e.into_inner()).len()
}

/// The event's message and fields, appended to one line. `message` loses its
/// name because it is the sentence; everything else keeps `name=value`, which is
/// the shape the terminal layer prints and the shape a reader greps.
struct Fields<'a>(&'a mut String);

impl Visit for Fields<'_> {
    fn record_str(&mut self, field: &Field, value: &str) {
        // Writing to a `String` cannot fail, and a log tail is the last place to
        // start propagating errors from.
        let _ = match field.name() {
            "message" => write!(self.0, " {value}"),
            name => write!(self.0, " {name}={value}"),
        };
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        let _ = match field.name() {
            "message" => write!(self.0, " {value:?}"),
            name => write!(self.0, " {name}={value:?}"),
        };
    }
}
