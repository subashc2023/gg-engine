//! Where a run's input comes from (§4.7).
//!
//! Live play and a replay are two sources for one stream, and the sim must not
//! be able to tell them apart — which is why they meet here, at the [`Input`]
//! that both of them feed, rather than at a branch inside a game system.
//!
//! Recording is *inside* the live arm rather than beside it: a recorder that had
//! to be called separately is a recorder someone forgets to call on the one path
//! that mattered, and §1.2's "any bug report is a replay file" only holds if
//! every live tick is written down.

use crate::{Input, InputFrame, Recorder, Replay};

/// Live play, optionally recorded, or a recording being played back.
pub enum Drive {
    /// Hardware, optionally writing down every tick.
    Live(Option<Box<Recorder>>),
    /// A file, driving the same ticks again.
    Replay(Box<Replay>),
}

impl Drive {
    /// This tick's frame, latched from whichever source this is.
    ///
    /// Call once per *tick*, not once per frame: `just_pressed` is an edge
    /// between two ticks, and a render frame that owes three of them must not
    /// report one edge to all three.
    pub fn frame(&mut self, input: &mut Input, tick: u64) -> InputFrame {
        match self {
            Drive::Live(recorder) => {
                let frame = input.tick();
                if let Some(recorder) = recorder {
                    recorder.record(tick, frame);
                }
                frame
            }
            Drive::Replay(replay) => {
                let frame = replay.frame(tick);
                input.tick_from(frame);
                frame
            }
        }
    }

    /// This tick's typed characters, through the same seam (§6 M16): live
    /// keystrokes are written down, a replay's are read back out. Call once per
    /// tick, beside [`Drive::frame`] and at the same `tick`.
    ///
    /// Owned rather than borrowed because a caller mid-tick holds the whole
    /// host, and a `&str` out of the replay would keep this borrowed across it.
    /// The empty case allocates nothing, which is every tick nobody typed on.
    #[must_use]
    pub fn text(&mut self, tick: u64, typed: &str) -> String {
        match self {
            Drive::Live(recorder) => {
                if let Some(recorder) = recorder {
                    recorder.record_text(tick, typed);
                }
                typed.to_owned()
            }
            Drive::Replay(replay) => replay.text_at(tick).to_owned(),
        }
    }

    /// Open a replay segment at `tick`, naming the game build that produces the
    /// ticks from here on (§4.2.2). A no-op when nothing is being recorded,
    /// which is what lets the host call it at every load and every swap without
    /// asking first.
    pub fn open_segment(&mut self, tick: u64, code_hash: u128) {
        if let Drive::Live(Some(recorder)) = self {
            recorder.open_segment(tick, code_hash);
        }
    }

    /// Ticks the replay covers, if this is one — what bounds a replay run.
    #[must_use]
    pub fn ticks(&self) -> Option<u64> {
        match self {
            Drive::Replay(replay) => Some(replay.ticks()),
            Drive::Live(_) => None,
        }
    }

    /// The replay being played back, for the checks a host runs against it.
    #[must_use]
    pub fn replay(&self) -> Option<&Replay> {
        match self {
            Drive::Replay(replay) => Some(replay),
            Drive::Live(_) => None,
        }
    }

    /// Whether this run's input stream could survive the process restarting
    /// under it — which only unrecorded live play can.
    ///
    /// A recorder holds its ticks in memory and would be lost; a replay would
    /// resume at *its* tick zero while the world resumed where it left off. Both
    /// are silent corruptions of a determinism artifact, so rejuvenation
    /// (§4.2.2) asks first rather than restarting and hoping.
    #[must_use]
    pub fn survives_restart(&self) -> bool {
        matches!(self, Drive::Live(None))
    }

    /// The recording, once the run is over. `None` for a replay, which has
    /// nothing to write.
    #[must_use]
    pub fn recorder(self) -> Option<Box<Recorder>> {
        match self {
            Drive::Live(recorder) => recorder,
            Drive::Replay(_) => None,
        }
    }
}
