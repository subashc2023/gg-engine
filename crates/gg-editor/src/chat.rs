//! The agent panel's conversation (§6 M16).
//!
//! # Why there is text entry here at all
//!
//! §4.7 bans keystrokes from the *sim*: a character is not an action, it is not
//! in the action map, and a world edit driven by one could not be replayed. None
//! of that is an argument against typing to an agent — the panels are host state,
//! absent from the world and from the canonical hash — but a session that could
//! not be replayed would still have been a hole in §5.6c.
//!
//! So the ban's *reason* is what was removed, not the ban routed around:
//! printable input rides `gg_input::replay`'s text channel, and the two editing
//! keys are ordinary appended verbs ([`verb::TEXT_BACK`], [`verb::TEXT_SEND`])
//! in the recorded [`InputFrame`](gg_input::InputFrame). A typed prompt replays
//! character for character, and a build without the editor still refuses the
//! file by its verb list.
//!
//! Everything below is a pure function of that stream, which is the property
//! that makes a chat panel replayable rather than merely recorded.

use crate::host::verb;

/// Turns kept before the oldest is dropped. A session is an afternoon and an
/// answer is a paragraph; keeping all of it is a leak nobody would notice until
/// it mattered.
const KEPT: usize = 64;

/// Ticks the caret spends lit, and dark. Derived from the tick and not a clock
/// so a replayed frame draws the caret the recorded one did — the golden suite
/// renders this panel.
const BLINK: u64 = 20;

/// Who said a line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Who {
    /// The operator, typed.
    You,
    /// The agent, answered.
    Claude,
    /// The agent, working. A tool call, as it is made — the difference between
    /// a minute of spinner and a minute of watching something happen, which is
    /// the whole reason the stream is read line by line.
    Tool,
    /// The editor itself — what it did, and what it refused.
    Host,
}

/// One thing said.
pub(crate) struct Turn {
    pub who: Who,
    pub text: String,
}

/// The transcript and the line being typed into it.
#[derive(Default)]
pub(crate) struct Chat {
    /// What the operator is typing, before Enter. Not a `Vec<char>`: it is
    /// appended and truncated at the end only, which `String` does in place.
    pub prompt: String,
    /// Oldest first, capped at [`KEPT`].
    pub log: Vec<Turn>,
    /// An answer is outstanding. Cleared by the stream's own terminal event
    /// rather than by the panel noticing a status, so "still working" and
    /// "finished" are the same fact the transcript was built from.
    pub waiting: bool,
    /// The tick the outstanding question was asked on, for the elapsed count
    /// beside the spinner. From the tick and not a clock, for [`BLINK`]'s
    /// reason: the golden suite renders this panel.
    asked_at: Option<u64>,
    /// The conversation `claude` is holding, once its stream has named it.
    ///
    /// Held here rather than in the [`gg_agent::Ask`] because an `Ask` is one
    /// *question* and is replaced by the next one — carrying the id across is
    /// what makes the second question a follow-up instead of a cold start, which
    /// on the first real session was thirty seconds of the thirty-two it took to
    /// answer "Hello".
    pub session: Option<String>,
    /// The seam this log has already written a line about, by tick. One reload
    /// is one line, however many frames the panel draws it for.
    noted: Option<u64>,
    /// Wrapped lines the panel drew last tick. A change means something was
    /// said, which is what pins the view to the bottom — a transcript that
    /// stayed where it was would put every answer off the end of the pane.
    /// Lines and not turns, so a pane resized narrower re-pins too.
    pub drawn: usize,
}

impl Chat {
    /// Append a turn, dropping the oldest past [`KEPT`].
    pub fn say(&mut self, who: Who, text: impl Into<String>) {
        if self.log.len() == KEPT {
            self.log.remove(0);
        }
        self.log.push(Turn {
            who,
            text: text.into(),
        });
    }

    /// Apply this tick's keystrokes to the prompt.
    ///
    /// `typed` is the replay's text channel and `back` the [`verb::TEXT_BACK`]
    /// edge. Control characters are dropped rather than inserted: Backspace and
    /// Enter arrive here *as characters too* on some platforms, and they are
    /// already verbs — a field that took both would delete twice.
    pub fn edit(&mut self, typed: &str, back: bool) {
        for c in typed.chars().filter(|c| !c.is_control()) {
            self.prompt.push(c);
        }
        if back {
            // By character, not byte: `truncate` on a byte boundary inside a
            // multi-byte glyph panics, and a prompt can hold one.
            self.prompt.pop();
        }
    }

    /// Take the prompt as a question, or `None` if there is nothing to ask.
    ///
    /// Whitespace-only counts as nothing: Enter on an empty field is how a
    /// human dismisses a caret, and spending money on it would be a surprise.
    pub fn take_prompt(&mut self) -> Option<String> {
        let asked = self.prompt.trim().to_owned();
        self.prompt.clear();
        (!asked.is_empty()).then_some(asked)
    }

    /// Write one line about a reload, the first time this one is seen.
    ///
    /// The panel shows the seam in full above the transcript; this is what puts
    /// it *in* the conversation, so an answer three turns later still has the
    /// failure it was about beside it.
    pub fn note(&mut self, seam: &gg_agent::Seam) {
        if self.noted == Some(seam.tick) {
            return;
        }
        self.noted = Some(seam.tick);
        let line = match &seam.outcome {
            gg_agent::Outcome::Refused { kind, detail } => {
                format!("reload REFUSED at tick {} — {kind}: {detail}", seam.tick)
            }
            gg_agent::Outcome::Accepted => {
                format!(
                    "reloaded at tick {}, {} entities live",
                    seam.tick, seam.entities
                )
            }
        };
        self.say(Who::Host, line);
    }

    /// Note that a question has just gone out on `tick`.
    pub fn asked(&mut self, tick: u64) {
        self.waiting = true;
        self.asked_at = Some(tick);
    }

    /// Take one event off the agent's stream into the transcript.
    ///
    /// Every arm appends rather than replaces, which is what makes the panel a
    /// log of a turn instead of a view of its latest state: a tool call stays
    /// visible after the answer that followed it.
    pub fn hear(&mut self, event: &gg_agent::Event) {
        match event {
            gg_agent::Event::Session(id) => self.session = Some(id.clone()),
            gg_agent::Event::Text(text) => self.say(Who::Claude, text.clone()),
            gg_agent::Event::Tool { name, detail } => {
                let line = match detail.is_empty() {
                    true => name.clone(),
                    false => format!("{name}  {detail}"),
                };
                self.say(Who::Tool, line);
            }
            gg_agent::Event::Done => self.settled(),
            gg_agent::Event::Failed(why) => {
                self.settled();
                self.say(Who::Host, format!("failed — {why}"));
            }
        }
    }

    fn settled(&mut self) {
        self.waiting = false;
        self.asked_at = None;
    }

    /// Whole seconds the outstanding question has been outstanding, or `None`
    /// where none is. Ticks and not a clock — see [`Chat::asked_at`].
    #[must_use]
    pub fn waited(&self, tick: u64) -> Option<u64> {
        let since = self.asked_at?;
        Some(tick.saturating_sub(since) / u64::from(gg_core::DEFAULT_TICK_HZ))
    }

    /// Whether the caret is lit on `tick`. See [`BLINK`].
    #[must_use]
    pub fn caret_on(tick: u64) -> bool {
        (tick / BLINK).is_multiple_of(2)
    }
}

/// The two editing edges this tick, read by name against *this* build's map for
/// [`crate::camera`]'s reason: the ids move at every reload.
///
/// Returns `(backspace, enter)`, both `false` for a host routing no input.
pub(crate) fn edits(input: Option<&gg_input::Input>) -> (bool, bool) {
    let Some(input) = input else {
        return (false, false);
    };
    let edge = |name| crate::camera::id(input, name).is_some_and(|id| input.just_pressed(id));
    (edge(verb::TEXT_BACK), edge(verb::TEXT_SEND))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typing_accumulates_and_backspace_removes_a_character_not_a_byte() {
        let mut chat = Chat::default();
        chat.edit("why is it", false);
        chat.edit(" sé", false);
        assert_eq!(chat.prompt, "why is it sé");
        chat.edit("", true);
        // `é` is two bytes; a byte-wise truncate would have panicked or left a
        // lone continuation byte behind.
        assert_eq!(chat.prompt, "why is it s");
    }

    #[test]
    fn the_keys_that_are_verbs_are_not_also_characters() {
        let mut chat = Chat::default();
        // What a platform sends alongside the Backspace and Enter *verbs*.
        chat.edit("ab\u{8}\r\n", false);
        assert_eq!(chat.prompt, "ab", "control characters are the verbs' job");
    }

    #[test]
    fn an_empty_prompt_is_not_a_question() {
        let mut chat = Chat::default();
        assert_eq!(chat.take_prompt(), None);
        chat.edit("   ", false);
        assert_eq!(chat.take_prompt(), None, "whitespace is not a question");
        chat.edit(" why  ", false);
        assert_eq!(chat.take_prompt().as_deref(), Some("why"));
        assert!(chat.prompt.is_empty(), "asking clears the field");
    }

    #[test]
    fn one_reload_writes_one_line_however_many_frames_draw_it() {
        let mut chat = Chat::default();
        let seam = gg_agent::Seam {
            tick: 12,
            outcome: gg_agent::Outcome::Accepted,
            code_before: String::new(),
            code_after: None,
            state_before: String::new(),
            state_after: String::new(),
            entities: 3,
            changes: Vec::new(),
            load_ms: None,
            save_to_swap_ms: None,
        };
        for _ in 0..5 {
            chat.note(&seam);
        }
        assert_eq!(chat.log.len(), 1);
        let next = gg_agent::Seam { tick: 40, ..seam };
        chat.note(&next);
        assert_eq!(chat.log.len(), 2, "a second reload is a second line");
    }

    /// The failure this replaced: fourteen tool calls arrived as one blob when
    /// the process exited, so a minute of work looked like a hung panel.
    #[test]
    fn a_turn_arrives_as_it_happens_and_the_tools_stay_visible_under_the_answer() {
        let mut chat = Chat::default();
        chat.asked(600);
        assert_eq!(chat.waited(600), Some(0));
        assert_eq!(chat.waited(750), Some(2), "ticks, not a wall clock");

        for event in [
            gg_agent::Event::Session("9f-1".to_owned()),
            gg_agent::Event::Text("I'll read the seam.".to_owned()),
            gg_agent::Event::Tool {
                name: "Read".to_owned(),
                detail: "lib.rs".to_owned(),
            },
            gg_agent::Event::Tool {
                name: "gg_session".to_owned(),
                detail: String::new(),
            },
            gg_agent::Event::Text("The spin field was retyped.".to_owned()),
            gg_agent::Event::Done,
        ] {
            chat.hear(&event);
        }
        assert_eq!(chat.session.as_deref(), Some("9f-1"));
        assert!(!chat.waiting, "the stream's own end settles it");
        assert_eq!(chat.waited(900), None);
        let said: Vec<_> = chat.log.iter().map(|t| (t.who, t.text.as_str())).collect();
        assert_eq!(
            said,
            [
                (Who::Claude, "I'll read the seam."),
                (Who::Tool, "Read  lib.rs"),
                // No argument worth showing is a bare name, not a trailing gap.
                (Who::Tool, "gg_session"),
                (Who::Claude, "The spin field was retyped."),
            ]
        );
    }

    /// A failure ends the wait *and* says so. Leaving `waiting` set would spin
    /// forever; saying nothing would look like an answer of no words.
    #[test]
    fn a_failed_turn_stops_the_spinner_and_names_the_reason() {
        let mut chat = Chat::default();
        chat.asked(10);
        chat.hear(&gg_agent::Event::Failed("usage limit reached".to_owned()));
        assert!(!chat.waiting);
        assert_eq!(chat.log.len(), 1);
        assert_eq!(chat.log[0].who, Who::Host);
        assert!(chat.log[0].text.contains("usage limit reached"));
    }

    /// The session outlives the question it was learned from — that is the
    /// entire difference between a conversation and a series of cold starts.
    #[test]
    fn the_session_survives_the_turn_that_named_it() {
        let mut chat = Chat::default();
        chat.hear(&gg_agent::Event::Session("keep-me".to_owned()));
        chat.hear(&gg_agent::Event::Done);
        assert_eq!(chat.session.as_deref(), Some("keep-me"));
    }

    #[test]
    fn the_transcript_is_capped_and_keeps_the_newest() {
        let mut chat = Chat::default();
        for i in 0..KEPT + 10 {
            chat.say(Who::You, i.to_string());
        }
        assert_eq!(chat.log.len(), KEPT);
        assert_eq!(chat.log[KEPT - 1].text, (KEPT + 9).to_string());
    }
}
