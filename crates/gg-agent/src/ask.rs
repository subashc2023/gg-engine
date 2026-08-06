//! Renting Claude Code (§6 M16): a conversation with the `claude` already on
//! this machine, held off the frame thread.
//!
//! # Why the binary and not an API
//!
//! `claude -p` reads the credentials `claude auth login` wrote, so the work is
//! billed to the operator's own subscription and this process never sees a key.
//! That is a property of *not* passing `--bare`, which skips the keychain and
//! demands `ANTHROPIC_API_KEY` — so `--bare` is the one flag this must never
//! use, and the reason is worth the sentence because the flag reads like an
//! optimisation.
//!
//! Not passing it is only half of it: an inherited `ANTHROPIC_API_KEY` takes
//! precedence over the login *without* the flag, so the child's environment is
//! cleared of it in [`stream`]. Found by running the real thing and reading the
//! warning it prints, which is the only place it is said.
//!
//! # A stream, and the same session
//!
//! Two things were measured on the first real session and both were the panel's
//! fault rather than the model's. A question that ran fourteen tools showed a
//! spinner for a minute because the answer was collected with `Output::output`,
//! which returns when the process *exits*; and "Hello" cost thirty seconds
//! because every question was a cold `claude` with no memory of the last one.
//!
//! So: `--output-format stream-json`, read line by line and published as
//! [`Event`]s while the turn is still running, and `--resume` on every question
//! after the first. The session id arrives in the stream's own first line, which
//! is why capturing it costs nothing extra. Reading the stream needs a JSON
//! reader; [`crate::json`] is that, beside the writer, rather than a dependency
//! this crate's manifest says it does not have.
//!
//! # What it must not inherit
//!
//! A `claude` started in this tree picks up the project's settings, and this
//! project's `Stop` hook is `cargo xtask ci --fast`. That hook cannot run from
//! here by construction: `xtask.exe` is the *parent* of the running game, so the
//! hook's first act is to relink a binary its own process tree holds open. The
//! third measured failure was a turn that answered in eight seconds and then
//! spent seventy more watching the agent try to diagnose that lock — with every
//! process-inspection command it reached for denied, because none of them are
//! granted. `--setting-sources user` is why it no longer inherits it.
//!
//! Two more came with the operator's own config, both invisible: unrelated MCP
//! servers (an editor, a note-taker, a video suite — sixty-odd tools of context
//! on every question, and the larger half of a cold start), and a built-in set
//! that included a shell and a subagent spawner. `--strict-mcp-config` and
//! [`TOOLS`] cut both.
//!
//! # Never from an automated run
//!
//! [`Ask::spawn`] refuses under `GG_HEADLESS=1` and says so. A recorded editor
//! session replays through the same clicks as any other (§4.7), so without this
//! a gate would call out to a paid, non-deterministic service every time it ran
//! the panel — and the replay would compare two different answers. Refusing
//! makes the click land, the panel react, and the tick hash stay a function of
//! the world.

use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::mpsc::{Receiver, TryRecvError, channel};

use crate::json::Json;

/// The built-in set, and the only thing here that confines: a tool absent from
/// this list does not exist in the session, so there is nothing to deny and
/// nothing to prompt about. No network (`WebFetch`, `WebSearch`), no second
/// shell, no subagent spawner — a panel opened to ask about a reload has no
/// business with any of them, and the first real session reached for all three.
///
/// `Bash` stays: §1's loop is only closed if the agent can *edit and rebuild*,
/// and an ask that could only explain would make the panel a nicer log viewer.
/// What it may run is [`ALLOWED_TOOLS`]'s business.
const TOOLS: &str = "Read,Glob,Grep,Edit,Write,Bash";

/// The pre-approval — and, under `-p` with `--permission-mode default`, the
/// effective boundary: a print run has no prompt to fall back on, so a call
/// this list does not name is denied rather than asked. The one thing that
/// still runs past it is the CLI's own read-only `Bash` safelist (`git
/// status` and kin), which grants nothing `Read`/`Grep` had not. That pairing
/// is what "narrowed to the two build commands" rests on — `--allowedTools`
/// alone grants and does not confine (§6 M16 item 7), and `acceptEdits` was
/// measured waving `touch` and `git status` through with neither named here.
const ALLOWED_TOOLS: &str = "Read,Glob,Grep,Edit,Write,Bash(cargo build *),Bash(cargo check *),\
                             mcp__gg-engine__gg_session,mcp__gg-engine__gg_last_reload";

/// Where a request has got to. The answer is not in here — it arrived as
/// [`Event::Text`] while this was still [`Status::Running`], which is the whole
/// point of the stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Status {
    /// Nothing asked yet.
    Idle,
    /// Spawned, and events are arriving.
    Running,
    /// The turn ended cleanly.
    Done,
    /// Never started, or ended badly. The message is for the operator and names
    /// what to do about it.
    Failed(String),
}

/// Something the agent did, as it did it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    /// The id to resume next time. First line of the stream.
    Session(String),
    /// Prose. One assistant message, not one token — see [`Ask::spawn`].
    Text(String),
    /// A tool call, named for the transcript. `detail` is the argument worth
    /// showing (a path, a pattern, a command), already shortened.
    Tool { name: String, detail: String },
    /// The turn ended cleanly.
    Done,
    /// It ended badly, and this says how.
    Failed(String),
}

/// One question in flight, and the conversation it belongs to.
pub struct Ask {
    prompt: String,
    status: Status,
    session: Option<String>,
    /// `None` once the stream has ended, or when the request never started.
    rx: Option<Receiver<Event>>,
    /// The turn's child, shared with the worker so dropping the `Ask` can end
    /// it — see [`Ask::drop`].
    live: LiveRef,
}

/// The live child, and whether the panel has already ended the turn. One lock
/// holding both rather than a flag beside a slot: "park the child" and "the
/// panel is gone" must be a single observation, or a child parked in the
/// instant after the panel closed would outlive it.
#[derive(Default)]
struct Live {
    child: Option<std::process::Child>,
    ended: bool,
}

type LiveRef = std::sync::Arc<std::sync::Mutex<Live>>;

/// A poisoned lock still holds the child, and killing it is more important
/// than whatever panicked while holding the guard.
fn locked(live: &LiveRef) -> std::sync::MutexGuard<'_, Live> {
    live.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Kill, then reap. `kill` on an already-exited child is not an error, and
/// `wait` is what collects the exit status either way.
fn end(mut child: std::process::Child) {
    let _ = child.kill();
    let _ = child.wait();
}

impl Drop for Ask {
    /// Closing the panel ends the turn. Without this the child outlives the
    /// window — an agent still editing the tree with nobody watching — and an
    /// exited one is never reaped.
    fn drop(&mut self) {
        let mut live = locked(&self.live);
        live.ended = true;
        if let Some(child) = live.child.take() {
            end(child);
        }
    }
}

impl Ask {
    /// An `Ask` that has not been asked.
    #[must_use]
    pub fn idle() -> Ask {
        Ask {
            prompt: String::new(),
            status: Status::Idle,
            session: None,
            rx: None,
            live: LiveRef::default(),
        }
    }

    /// Ask `prompt`, with `claude` running in `cwd`, continuing `session` where
    /// there is one.
    ///
    /// Returns immediately: the wait is on a thread, because this is called from
    /// a frame and a `claude` turn is seconds to minutes. Refuses under
    /// `GG_HEADLESS=1` with a [`Status::Failed`] naming the reason — see the
    /// module note.
    ///
    /// Text arrives one assistant *message* at a time rather than one token:
    /// token-level deltas need `--include-partial-messages`, whose events
    /// duplicate the message that follows them, and a transcript that showed
    /// every answer twice would be a worse bug than a paragraph landing whole.
    #[must_use]
    pub fn spawn(prompt: &str, cwd: &Path, session: Option<&str>) -> Ask {
        // Set and not "0"/empty — `gg-platform`'s parse, replicated because
        // this crate deliberately has no dependencies.
        if std::env::var_os("GG_HEADLESS").is_some_and(|v| !v.is_empty() && v != "0") {
            let why = "Not asked: this is a headless run (GG_HEADLESS=1). An automated tier never \
                       calls out to an agent — the click landed and nothing was spent."
                .to_owned();
            // The refusal is an *event*, not only a status: the panel builds
            // its transcript from the stream, so a refusal with no stream would
            // never reach it and the spinner would run for the session.
            let (tx, rx) = channel();
            let _ = tx.send(Event::Failed(why.clone()));
            return Ask {
                prompt: prompt.to_owned(),
                status: Status::Failed(why),
                session: session.map(ToOwned::to_owned),
                rx: Some(rx),
                live: LiveRef::default(),
            };
        }
        let (tx, rx) = channel();
        let (prompt_owned, cwd_owned) = (prompt.to_owned(), cwd.to_owned());
        let resume = session.map(ToOwned::to_owned);
        let live = LiveRef::default();
        let shared = live.clone();
        // Detached: the panel polls. The child does *not* just die with the
        // panel — [`Ask::drop`] is what ends it.
        std::thread::spawn(move || run_claude(&prompt_owned, &cwd_owned, resume, &tx, &shared));
        Ask {
            prompt: prompt.to_owned(),
            status: Status::Running,
            session: session.map(ToOwned::to_owned),
            rx: Some(rx),
            live,
        }
    }

    /// The question, for the panel to echo back.
    #[must_use]
    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    /// The conversation to resume, once the stream has named it. Outlives the
    /// turn on purpose — the panel hands it to the *next* [`Ask::spawn`], which
    /// is what makes the second question a follow-up rather than a cold start.
    #[must_use]
    pub fn session(&self) -> Option<&str> {
        self.session.as_deref()
    }

    /// Where the request has got to.
    #[must_use]
    pub fn status(&self) -> &Status {
        &self.status
    }

    /// Everything that has happened since the last call, oldest first.
    ///
    /// Cheap enough to call every frame: non-blocking probes of a channel that
    /// is usually empty, and nothing at all once the turn has ended. Updates
    /// [`Ask::status`] and [`Ask::session`] on the way through, so a caller that
    /// only wants to draw a spinner need not inspect the events.
    pub fn drain(&mut self) -> Vec<Event> {
        let mut events = Vec::new();
        let Some(rx) = &self.rx else {
            return events;
        };
        loop {
            match rx.try_recv() {
                Ok(event) => {
                    match &event {
                        Event::Session(id) => self.session = Some(id.clone()),
                        Event::Done => self.status = Status::Done,
                        Event::Failed(why) => self.status = Status::Failed(why.clone()),
                        Event::Text(_) | Event::Tool { .. } => {}
                    }
                    let ended = matches!(event, Event::Done | Event::Failed(_));
                    events.push(event);
                    if ended {
                        self.rx = None;
                        return events;
                    }
                }
                // A closed channel with no terminal event is a panicked worker.
                // It must not read as a finished turn, which is what leaving the
                // status at `Running` forever would eventually look like.
                Err(TryRecvError::Disconnected) => {
                    self.rx = None;
                    let why = "the request thread ended without an answer".to_owned();
                    self.status = Status::Failed(why.clone());
                    events.push(Event::Failed(why));
                    return events;
                }
                Err(TryRecvError::Empty) => return events,
            }
        }
    }
}

/// The subprocess and its stream, on the worker thread.
///
/// A `--resume` naming a session this machine no longer has is a hard exit
/// before any event, so that one case retries cold rather than wedging the panel
/// on a conversation it cannot get back to. Only that case: a retry after text
/// had already arrived would say everything twice.
fn run_claude(prompt: &str, cwd: &Path, resume: Option<String>, tx: &Sender, live: &LiveRef) {
    let resumed = resume.is_some();
    let first = stream(prompt, cwd, resume, tx, live);
    let Some(why) = first.failure else { return };
    // Only the silent case, and only once. A retry after anything had been said
    // would say all of it twice, and a second failure is the real one.
    if resumed && first.published == 0 {
        let _ = tx.send(Event::Tool {
            name: "session".to_owned(),
            detail: "gone — starting a new one".to_owned(),
        });
        if let Some(why) = stream(prompt, cwd, None, tx, live).failure {
            let _ = tx.send(Event::Failed(why));
        }
        return;
    }
    let _ = tx.send(Event::Failed(why));
}

type Sender = std::sync::mpsc::Sender<Event>;

/// How a stream finished. Terminal failures are *returned* rather than sent, so
/// the retry above can swallow the one it is about to make good on.
struct Ended {
    /// Events sent to the panel — the count that decides whether a retry would
    /// be a repetition.
    published: usize,
    /// Why it did not end cleanly, where it did not. A turn that ended without
    /// a `result` line counts: the panel would otherwise wait forever.
    failure: Option<String>,
}

/// The exact `claude` argv and environment, split from [`stream`] so the
/// guarding tests pin the invocation actually spawned rather than a constant
/// they hope is used — the "property of a string" failure §6 M16 item 7 named.
fn invocation(prompt: &str, cwd: &Path, resume: Option<&str>) -> std::process::Command {
    let mut command = std::process::Command::new(claude_binary());
    command
        .current_dir(cwd)
        .args(["-p", prompt])
        // `--verbose` is not optional decoration: `stream-json` out of `-p` is
        // refused without it.
        .args(["--output-format", "stream-json", "--verbose"])
        .args(["--tools", TOOLS])
        .args(["--allowedTools", ALLOWED_TOOLS])
        // `default`, and the mode is what makes [`ALLOWED_TOOLS`] bind: `-p`
        // has no prompt to fall back on, so whatever would ask is denied.
        // `acceptEdits` was measured (this desk, 2026-08) approving `Bash`
        // commands the allowance never named — `git status`, `touch` — because
        // it waves file-touching commands through wholesale; under `default`
        // the same probe denies them while `Edit`/`Write` and the two cargo
        // forms still land promptless, since pre-approval is the list's job.
        .args(["--permission-mode", "default"])
        // Explicit rather than discovered, so the engine's own tools are present
        // without the first-use approval an auto-discovered project server asks
        // for — which is a prompt with nowhere to appear, same as above.
        .args(["--mcp-config", ".mcp.json"])
        // …and *only* those. Without this the operator's global servers come
        // too, which is sixty tools of context the panel never asked for.
        .arg("--strict-mcp-config")
        // Not the project's settings, whose `Stop` hook is the tier that would
        // relink the running game. `CLAUDE.md` is memory rather than a setting
        // source and still loads under `user` — checked, because a panel agent
        // that lost the house rules would be worse than one that runs a hook.
        // (Not `""` either: measured to drop `CLAUDE.md` entirely.)
        .args(["--setting-sources", "user"])
        // The module's billing claim, made true rather than assumed. A `claude`
        // that finds this set uses the key and says so in one warning line on
        // stderr, which is nowhere the operator is looking — and the bill leaves
        // the subscription silently. It is set in some shells on this desk, and
        // a child inherits the environment of whatever terminal launched the
        // game, so the only reliable place to be sure is here.
        .env_remove("ANTHROPIC_API_KEY");
    if let Some(id) = resume {
        command.args(["--resume", id]);
    }
    command
}

fn stream(prompt: &str, cwd: &Path, resume: Option<String>, tx: &Sender, live: &LiveRef) -> Ended {
    let mut command = invocation(prompt, cwd, resume.as_deref());
    command
        // Closed, not inherited: a `claude` that decided to read stdin would
        // otherwise be reading the terminal the *game* was launched from.
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(e) => {
            return Ended {
                published: 0,
                failure: Some(format!(
                    "could not run `{}`: {e}. Claude Code is rented, not bundled — install it \
                     and `claude auth login`, or set GG_CLAUDE to the binary.",
                    claude_binary().display()
                )),
            };
        }
    };
    // Drained on its own thread. A pipe nobody reads fills and blocks the writer,
    // and the writer here is the process whose stdout we are waiting on.
    let errors = child.stderr.take().map(|pipe| {
        std::thread::spawn(move || {
            let mut text = String::new();
            let _ = std::io::Read::read_to_string(&mut std::io::BufReader::new(pipe), &mut text);
            text
        })
    });
    let out = child.stdout.take();
    {
        let mut live = locked(live);
        if live.ended {
            // The panel closed in the instant between spawn and here.
            end(child);
            return Ended {
                published: 0,
                failure: None,
            };
        }
        live.child = Some(child);
    }
    let (mut published, mut ended) = (0usize, false);
    if let Some(out) = out {
        for line in std::io::BufReader::new(out).lines().map_while(Result::ok) {
            for event in read_line(&line) {
                ended |= matches!(event, Event::Done | Event::Failed(_));
                published += 1;
                if tx.send(event).is_err() {
                    // The panel is gone (window closed). An `acceptEdits` turn
                    // must not keep editing a tree nobody is watching, so the
                    // child ends with the panel — it never did die "with the
                    // process".
                    if let Some(child) = locked(live).child.take() {
                        end(child);
                    }
                    return Ended {
                        published,
                        failure: None,
                    };
                }
            }
        }
    }
    let Some(mut child) = locked(live).child.take() else {
        // The panel ended the turn under us; there is nobody left to report to.
        return Ended {
            published,
            failure: None,
        };
    };
    let status = child.wait();
    let stderr = errors
        .and_then(|handle| handle.join().ok())
        .unwrap_or_default();
    let ok = status.as_ref().is_ok_and(std::process::ExitStatus::success);
    if ok && ended {
        return Ended {
            published,
            failure: None,
        };
    }
    // Not `ok && published > 0`: a stream that said things and then died without
    // its `result` line has published no terminal event, and a panel waiting on
    // one would wait for the rest of the session.
    let detail = match stderr.trim() {
        "" => status.map_or_else(|e| e.to_string(), |s| s.to_string()),
        message => message.to_owned(),
    };
    Ended {
        published,
        failure: Some(format!("`claude` ended without an answer: {detail}")),
    }
}

/// One line of the stream, as zero or more events.
///
/// A line that does not parse is skipped rather than reported: the format is
/// the CLI's and gains event types on its own schedule, so an unknown line is
/// the normal case and a panel full of "could not parse" would be noise.
fn read_line(line: &str) -> Vec<Event> {
    let Some(json) = Json::parse(line.trim()) else {
        return Vec::new();
    };
    let mut events = Vec::new();
    match json.get("type").and_then(Json::str) {
        Some("system") => {
            if let Some(id) = json.get("session_id").and_then(Json::str) {
                events.push(Event::Session(id.to_owned()));
            }
        }
        Some("assistant") => {
            for block in json
                .get("message")
                .and_then(|m| m.get("content"))
                .map(Json::list)
                .unwrap_or_default()
            {
                match block.get("type").and_then(Json::str) {
                    Some("text") => {
                        if let Some(text) = block.get("text").and_then(Json::str)
                            && !text.trim().is_empty()
                        {
                            events.push(Event::Text(text.trim().to_owned()));
                        }
                    }
                    Some("tool_use") => {
                        if let Some(name) = block.get("name").and_then(Json::str) {
                            events.push(Event::Tool {
                                name: short_name(name),
                                detail: block.get("input").map(tool_detail).unwrap_or_default(),
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
        // The terminal line. `result` carries the final text, which the
        // `assistant` lines already published — republishing it would end every
        // answer with a copy of itself.
        Some("result") => events.push(match json.get("is_error").is_some_and(Json::is_true) {
            false => Event::Done,
            true => Event::Failed(
                json.get("result")
                    .and_then(Json::str)
                    .unwrap_or("the turn ended in an error")
                    .to_owned(),
            ),
        }),
        _ => {}
    }
    events
}

/// The MCP prefix is four segments of plumbing in front of the one word that
/// says what happened, in a pane about as wide as this sentence.
fn short_name(name: &str) -> String {
    name.rsplit("__").next().unwrap_or(name).to_owned()
}

/// The argument worth showing, out of a tool's input.
///
/// Named keys in priority order rather than the first string found: input maps
/// have no order worth relying on, and `Edit`'s first field being `file_path` or
/// `old_string` would otherwise be a coin toss.
fn tool_detail(input: &Json) -> String {
    const SHOWN: [&str; 7] = [
        "file_path",
        "command",
        "pattern",
        "path",
        "query",
        "url",
        "prompt",
    ];
    let Some(text) = SHOWN
        .iter()
        .find_map(|key| input.get(key).and_then(Json::str))
    else {
        return String::new();
    };
    // First line only: a `Bash` command can be a whole script, and the rest of
    // it below the fold would push the transcript off the pane.
    let first = text.lines().next().unwrap_or_default().trim();
    let short = first.rsplit(['\\', '/']).next().unwrap_or(first);
    // The tail after a separator is a file name; anything else keeps its head,
    // because a command's first words are the ones that say what it is.
    let shown = if short.len() < first.len() && !first.contains(' ') {
        short
    } else {
        first
    };
    match shown.char_indices().nth(48) {
        Some((cut, _)) => format!("{}…", &shown[..cut]),
        None => shown.to_owned(),
    }
}

/// `GG_CLAUDE`, or `claude` off `PATH`. The override exists because the binary
/// is a rental: a machine may have it somewhere this process's `PATH` does not.
fn claude_binary() -> PathBuf {
    std::env::var_os("GG_CLAUDE").map_or_else(|| PathBuf::from("claude"), PathBuf::from)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// The load-bearing one. Every other test in this tree runs headless, and
    /// without this refusal a replayed editor session would spend money and
    /// compare two different answers (§4.7, §5.6c).
    #[test]
    fn a_headless_run_refuses_to_ask_and_says_why() {
        // SAFETY: `nextest` gives each test its own process, so nothing else is
        // reading the environment while this runs, and the paired removal below
        // happens before any assertion — a failure cannot leave it set.
        unsafe { std::env::set_var("GG_HEADLESS", "1") };
        let mut ask = Ask::spawn("explain the last refusal", Path::new("."), None);
        let events = ask.drain();
        let status = ask.status().clone();
        // SAFETY: the pair of the above, and deliberately before the asserts.
        unsafe { std::env::remove_var("GG_HEADLESS") };
        let Status::Failed(message) = status else {
            panic!("a headless run asked anyway");
        };
        assert!(message.contains("GG_HEADLESS=1"), "{message}");
        assert!(message.contains("nothing was spent"), "{message}");
        // The refusal *streams*: the panel builds its transcript from events,
        // so a refusal that was only a status would never reach it and the
        // spinner would run for the session.
        assert!(
            matches!(events.as_slice(), [Event::Failed(why)] if why.contains("GG_HEADLESS=1")),
            "the refusal did not arrive as an event: {events:?}"
        );
        assert_eq!(ask.prompt(), "explain the last refusal");
    }

    /// A refusal must not carry a session forward as though it had asked, and
    /// must not lose one it was given — the next question is still a follow-up.
    #[test]
    fn a_headless_refusal_keeps_the_session_it_was_handed() {
        // SAFETY: as above — own process, paired removal before any assertion.
        unsafe { std::env::set_var("GG_HEADLESS", "1") };
        let ask = Ask::spawn("again", Path::new("."), Some("abc-123"));
        // SAFETY: the pair of the above.
        unsafe { std::env::remove_var("GG_HEADLESS") };
        assert_eq!(ask.session(), Some("abc-123"));
    }

    #[test]
    fn an_idle_ask_has_asked_nothing() {
        let mut ask = Ask::idle();
        assert_eq!(*ask.status(), Status::Idle);
        assert!(ask.prompt().is_empty());
        assert!(ask.drain().is_empty());
        assert_eq!(ask.session(), None);
    }

    /// The stream is the panel's whole view of a turn, so the shapes are pinned
    /// against a recorded sample rather than read off a `--help`.
    #[test]
    fn the_stream_reads_as_a_session_then_prose_then_tools_then_an_end() {
        let init = r#"{"type":"system","subtype":"init","session_id":"9f-1","tools":["Read"]}"#;
        assert_eq!(read_line(init), vec![Event::Session("9f-1".to_owned())]);

        let turn = r#"{"type":"assistant","message":{"content":[
            {"type":"text","text":"  I'll check the seam. "},
            {"type":"tool_use","name":"Read","input":{"file_path":"C:\\dev\\GGEngine\\a.rs"}},
            {"type":"tool_use","name":"mcp__gg-engine__gg_session","input":{}}
        ]}}"#;
        assert_eq!(
            read_line(turn),
            vec![
                Event::Text("I'll check the seam.".to_owned()),
                Event::Tool {
                    name: "Read".to_owned(),
                    detail: "a.rs".to_owned()
                },
                Event::Tool {
                    name: "gg_session".to_owned(),
                    detail: String::new()
                },
            ]
        );

        let ok = r#"{"type":"result","subtype":"success","is_error":false,"result":"done"}"#;
        assert_eq!(read_line(ok), vec![Event::Done]);
        let bad = r#"{"type":"result","is_error":true,"result":"usage limit reached"}"#;
        assert_eq!(
            read_line(bad),
            vec![Event::Failed("usage limit reached".to_owned())]
        );
    }

    /// The final `result` line repeats the whole answer. Publishing it would end
    /// every reply with a second copy of itself.
    #[test]
    fn the_terminal_line_does_not_republish_the_answer() {
        let line = r#"{"type":"result","is_error":false,"result":"the long answer again"}"#;
        assert_eq!(read_line(line), vec![Event::Done]);
    }

    /// Unknown and torn lines are the normal case: the CLI gains event types on
    /// its own schedule, and stdout is a pipe.
    #[test]
    fn a_line_this_does_not_understand_is_skipped_rather_than_reported() {
        for quiet in [
            r#"{"type":"stream_event","event":{"type":"content_block_delta"}}"#,
            r#"{"type":"user","message":{"content":[{"type":"tool_result"}]}}"#,
            r#"{"type":"assistant","message":{"content":[{"type":"text","text":"   "}]}}"#,
            r#"{"type":"assistant","message":{"conte"#,
            "",
        ] {
            assert!(read_line(quiet).is_empty(), "{quiet}");
        }
    }

    #[test]
    fn a_tool_shows_the_argument_worth_reading() {
        let detail = |input: &str| tool_detail(&Json::parse(input).expect("test input parses"));
        assert_eq!(detail(r#"{"file_path":"C:\\dev\\a.rs"}"#), "a.rs");
        assert_eq!(detail(r#"{"pattern":"fn main","path":"/x"}"#), "fn main");
        // A command keeps its head: the first words say what it is, and the tail
        // after the last slash would be an argument.
        assert_eq!(
            detail(r#"{"command":"cargo build -p demo-09-orbit"}"#),
            "cargo build -p demo-09-orbit"
        );
        // Multi-line scripts are one line here, and long ones are cut on a char
        // boundary — a byte cut inside a multi-byte glyph panics.
        assert_eq!(detail("{\"command\":\"line one\\nline two\"}"), "line one");
        assert!(detail(&format!(r#"{{"query":"é{}"}}"#, "x".repeat(80))).ends_with('…'));
        assert_eq!(detail(r#"{"unknown_key":"x"}"#), "");
    }

    /// The argv actually spawned, as strings. Every confinement test below
    /// reads this rather than a constant it hopes is used — the "property of a
    /// string" failure §6 M16 item 7 named.
    fn argv(resume: Option<&str>) -> Vec<String> {
        invocation("why", Path::new("."), resume)
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect()
    }

    fn has_pair(args: &[String], key: &str, value: &str) -> bool {
        args.windows(2).any(|w| w[0] == key && w[1] == value)
    }

    /// `--bare` is the flag that would move the bill from the operator's
    /// subscription to an API key this process does not have.
    #[test]
    fn nothing_here_reaches_for_bare_mode() {
        let args = argv(None);
        assert!(!args.iter().any(|a| a == "--bare"), "{args:?}");
        // The other half, and the one a real run caught: the key wins over the
        // login when it is merely *inherited*, no flag required.
        assert!(
            invocation("why", Path::new("."), None)
                .get_envs()
                .any(|(key, value)| key == "ANTHROPIC_API_KEY" && value.is_none()),
            "an inherited key would move the bill off the subscription"
        );
    }

    /// What each layer can actually claim, all measured on the real binary:
    /// `--tools` confines the session; the allowance pre-approves; and
    /// `--permission-mode default` is what makes the allowance *bind* — a
    /// print run has no prompt, so whatever would ask is denied. `acceptEdits`
    /// here was measured running `touch` and `git status` through `Bash` with
    /// neither in the allowance.
    #[test]
    fn bash_is_narrowed_by_default_mode_and_the_allowance_together() {
        let args = argv(None);
        assert!(has_pair(&args, "--tools", TOOLS), "{args:?}");
        assert!(has_pair(&args, "--allowedTools", ALLOWED_TOOLS), "{args:?}");
        assert!(has_pair(&args, "--permission-mode", "default"), "{args:?}");
        for absent in ["WebFetch", "WebSearch", "PowerShell", "Task"] {
            assert!(!TOOLS.contains(absent), "{TOOLS} reaches {absent}");
        }
        assert!(TOOLS.contains("Edit") && TOOLS.contains("Bash"), "{TOOLS}");
        assert!(
            ALLOWED_TOOLS.contains("mcp__gg-engine__"),
            "{ALLOWED_TOOLS}"
        );
        // The two build commands are the only `Bash` forms pre-approved, so
        // under `default` they are the only mutating commands that run.
        let bash: Vec<&str> = ALLOWED_TOOLS
            .split(',')
            .filter(|t| t.starts_with("Bash("))
            .collect();
        assert_eq!(bash, ["Bash(cargo build *)", "Bash(cargo check *)"]);
    }

    /// The `Stop` hook this tree carries rebuilds the running game from inside
    /// it. Inheriting the project's settings is how that happens.
    #[test]
    fn the_projects_own_hooks_are_not_inherited() {
        let args = argv(None);
        assert!(
            has_pair(&args, "--setting-sources", "user"),
            "the project `Stop` hook would relink the game it is running in: {args:?}"
        );
        assert!(
            args.iter().any(|a| a == "--strict-mcp-config"),
            "the operator's global MCP servers would ride along: {args:?}"
        );
    }

    /// The follow-up flag rides only when there is a session to follow.
    #[test]
    fn a_resume_is_passed_through_and_never_invented() {
        assert!(has_pair(&argv(Some("9f-1")), "--resume", "9f-1"));
        assert!(!argv(None).iter().any(|a| a == "--resume"));
    }

    #[test]
    fn the_binary_is_overridable_because_it_is_a_rental() {
        assert_eq!(claude_binary(), PathBuf::from("claude"));
    }
}
