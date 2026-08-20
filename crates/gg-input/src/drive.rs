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

use crate::{AXIS_SCALE, Input, InputFrame, Recorder, Replay};

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
    ///
    /// How much of a frame's accumulated motion each tick spends is
    /// [`Input::frame_covered`]' business, not this one's — and live-only by
    /// construction, since a replay's frames were already divided when they were
    /// recorded and dividing them again would be a second sim.
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

    /// This tick's knob changes, through the same seam as [`Drive::text`] (§6
    /// M40): live values that moved are written down, a replay's are handed back
    /// to be applied.
    ///
    /// The two directions are not symmetric here and that is the point. A live
    /// caller has *already* got the values — it read them to diff them — so
    /// nothing is returned and nothing is applied; a replayed caller gets what
    /// the file says and applies it. Returning the live ones would restamp every
    /// knob's source as a replay's on a run that is not one.
    pub fn knobs(&mut self, tick: u64, moved: &[(&str, String)]) -> &[(u64, String, String)] {
        match self {
            Drive::Live(recorder) => {
                if let Some(recorder) = recorder {
                    recorder.record_knobs(tick, moved);
                }
                &[]
            }
            Drive::Replay(replay) => replay.knobs_at(tick),
        }
    }

    /// Record the layout a session is aimed at (§6 M40, M81). A no-op while
    /// replaying, whose surface and scale are already in the file.
    pub fn record_surface(&mut self, surface: (u32, u32), dpi: f32) {
        if let Drive::Live(Some(recorder)) = self {
            recorder.record_surface(surface, dpi);
        }
    }

    /// The surface a replay says it was recorded at, if it says (§6 M40) —
    /// what a host lays a replayed session out for when no flag overrides it.
    /// `None` for live play and for the v1/v2 files that predate the field.
    #[must_use]
    pub fn surface(&self) -> Option<(u32, u32)> {
        match self.replay()?.meta().surface {
            (0, 0) => None,
            extent => Some(extent),
        }
    }

    /// The monitor scale a replay says it was recorded at, if it says (§6 M81)
    /// — [`Drive::surface`]'s other half, and `None` on the same terms.
    ///
    /// Exact: [`AXIS_SCALE`] is a power of two.
    #[must_use]
    pub fn dpi(&self) -> Option<f32> {
        match self.replay()?.meta().dpi {
            0 => None,
            ticks => Some(ticks as f32 / AXIS_SCALE as f32),
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

    /// Whether this session is a **determinism artifact** — written down or
    /// read back — rather than a player at a keyboard.
    ///
    /// The same condition [`survives_restart`](Self::survives_restart) tests
    /// and a different claim, which is why it is its own name: that one is
    /// about a rejuvenation losing a recorder's ticks, this one is about what a
    /// *host chord* is allowed to do.
    ///
    /// What wants it: Alt+Enter writes `Prefs::display`, which is hashed world
    /// state, from a window event that never reaches [`Drive::frame`] and is
    /// therefore in no recording. A recorded session an operator happened to
    /// resize would replay to a different hash, with nothing in the file to say
    /// why — the divergence is real, and it was only ever unreachable because
    /// §1.5 keeps every gate out of the windowed loop (§6 M78).
    #[must_use]
    pub fn scripted(&self) -> bool {
        !matches!(self, Drive::Live(None))
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::ReplayMeta;

    /// A player at a keyboard is the *only* arm that is not an artifact — a
    /// recording in progress is one as surely as a replay is, and it is the arm
    /// a reader is most likely to think of as "live".
    #[test]
    fn only_an_unwritten_live_session_is_the_players_own() {
        let meta = ReplayMeta {
            engine_commit: String::new(),
            contract: 0,
            tier: String::new(),
            tick_hz: 60,
            seed: 0,
            actions: Vec::new(),
            axes: Vec::new(),
            surface: (0, 0),
            dpi: 0,
        };
        assert!(!Drive::Live(None).scripted(), "a player at a keyboard");
        assert!(
            Drive::Live(Some(Box::new(Recorder::new(meta)))).scripted(),
            "a recording in progress is a file being written"
        );
    }

    /// Both halves of the recorded layout come back as themselves, and a live
    /// session has neither to give (§6 M40, M81).
    ///
    /// The scale is asserted *exactly*: every desktop scale is a quarter step
    /// and [`AXIS_SCALE`] is a power of two, so a tolerance here would forgive
    /// the one thing that can go wrong — a host laying out at 1.49 and a file
    /// saying 1.5 pick different whole scales at the same window size.
    #[test]
    fn the_layout_a_replay_was_recorded_at_comes_back_as_itself() {
        let mut recorder = Recorder::new(ReplayMeta::new(1, "dev", 60, &[], &[]));
        recorder.record_surface((3840, 2160), 1.5);
        recorder.record(0, InputFrame::default());
        let replay = Replay::decode(&recorder.finish().encode()).unwrap();
        let drive = Drive::Replay(Box::new(replay));
        assert_eq!(drive.surface(), Some((3840, 2160)));
        assert_eq!(drive.dpi(), Some(1.5));
        // Not the file's business, and a host that read it from one would be
        // laying a live session out for a monitor it is not on.
        assert_eq!(Drive::Live(None).dpi(), None);
        assert_eq!(Drive::Live(None).surface(), None);
    }
}
