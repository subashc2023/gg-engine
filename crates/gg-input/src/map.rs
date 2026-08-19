//! Action maps: config in, bindings out (§4.7). Game code names verbs; this is
//! the only layer that has heard of a key.
//!
//! Contexts **layer**: the active stack is consulted top down and the first
//! context that binds a source consumes it, so a UI context that binds `Escape`
//! stops the game context from also seeing it. That rule is what lets the UI
//! router (§4.9) be a consumer of action state rather than a second input path.
//!
//! # The file format, and why it is parsed here
//!
//! The map is a **subset of TOML** — table headers, and keys whose values are a
//! string or an array of strings **on one line**. Editors and formatters treat
//! it as TOML because it is TOML; the parser is ~150 lines because the subset is
//! small.
//!
//! The alternative was `toml` + `serde` in the dist graph of every shipped
//! game, for a file with three shapes in it. Should the format ever need
//! genuine TOML, swapping the crate in is mechanical and the format does not
//! move under anyone.
//!
//! **One line** is that subset's one real edge, and it is worth stating in bold
//! because the sentence above cuts both ways: a TOML formatter treats this as
//! TOML and will wrap a long array across several lines without asking, leaving
//! a file that is valid TOML and is not a map. §6 M74 walked into exactly that —
//! the wrapped array left demo 10 with no bindings at all, and what noticed was
//! a *golden image*, because the legend it draws is read off this file and the
//! key column went blank. Refused at the line the array **opens** on rather than
//! half-read, and the test below is what keeps that true.
//!
//! ```toml
//! [game.actions]
//! look = ["Tab"]                    # held while turning
//! spawn = ["F", "Mouse1"]
//!
//! [game.axes]
//! move_right = ["+D", "-A"]         # a key pair, signed
//! look_x = ["MouseX"]               # pointer motion
//! ```

use crate::key::{Key, MouseAxis, MouseButton, Wheel};
use gg_abi::InputFrame;

/// The verb identities and the ceilings on them live in `gg-abi` (§4.2.2).
///
/// Not because input belongs to the boundary — the map, the layering, the key
/// identity and the recorder are all still this crate's — but because a game
/// dylib is deny-pinned to `gg-abi`/`gg-ecs`/`gg-math` (§3) and must still be
/// able to say `input.pressed(SPAWN)`. So the *shape* of a verb crosses and the
/// judgement about what produced it does not.
///
/// An id is an index into the game's declared list, and that list's **order** is
/// the identity a replay file records: reordering it is a replay-format change,
/// appending to it is not.
pub use gg_abi::{ActionId, AxisId, MAX_ACTIONS, MAX_AXES};

/// A layer of the map, as an index into its declaration order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContextId(u8);

impl ContextId {
    /// Its index in declaration order.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Something that is either down or up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// A key by physical position.
    Key(Key),
    /// A mouse button.
    Button(MouseButton),
    /// A wheel notch. Down for exactly the tick it arrived in — see [`Wheel`].
    Wheel(Wheel),
}

/// What feeds an axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AxisSource {
    /// A digital source contributing its sign while held.
    Held(Source),
    /// Pointer motion for this tick.
    Motion(MouseAxis),
}

#[derive(Debug)]
struct ButtonBinding {
    source: Source,
    action: ActionId,
}

#[derive(Debug)]
struct AxisBinding {
    source: AxisSource,
    axis: AxisId,
    /// `+1` or `-1`, applied to every source alike — motion included, which is
    /// how a map spells an inverted look without the game knowing (§6 M81
    /// corrected this comment: it said motion's sign was ignored, and
    /// [`ActionMap::motion_axes`] has always multiplied by it, so the only
    /// written record of the convention described a behaviour the code did not
    /// have).
    sign: i32,
}

#[derive(Debug)]
struct Context {
    name: String,
    buttons: Vec<ButtonBinding>,
    axes: Vec<AxisBinding>,
    /// What survives a modal screen, as bitsets over the two id spaces.
    keep: Keep,
}

impl Context {
    /// Whether this layer binds `source` at all — the consumption test. A
    /// context that binds a key to an *axis* still eats it for the layers
    /// below, which is the point: consumption is about the key, not the verb.
    fn binds(&self, source: Source) -> bool {
        self.buttons.iter().any(|b| b.source == source)
            || self
                .axes
                .iter()
                .any(|a| a.source == AxisSource::Held(source))
    }
}

/// Which verbs reach the game while a modal screen is up (§6 M45) — the
/// `keep` list of a `[<context>.modal]` table, as two bitsets.
///
/// **A property of the action, which is why it is in the map at all.** A pause
/// screen has to keep hearing PAUSE and BACK while it refuses LEFT and
/// HARD_DROP, so suppression cannot be all-or-nothing; and whether a verb is
/// one a menu answers is a fact about the verb, not about the game's code. So
/// it is resolved by name like every other row in the file.
///
/// **An empty one withholds nothing**, so arbitration is opt-in per map: a game
/// that raises [`Prefs::modal`][m] under a map with no `[<context>.modal]` table
/// behaves exactly as it did before M45, and so does a session given no map at
/// all — which every replay-driven gate in the tree is, since a recorded stream
/// carries verb ids and needs no bindings to mean anything. Withholding
/// everything was the first spelling and `xtask reload --menu` refused it: the
/// tour reached six of ten milestones because a mapless session silenced the
/// keyboard the moment the title screen came up.
///
/// [m]: ../../gg_ecs/boundary/struct.Prefs.html#structfield.modal
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Keep {
    actions: u64,
    axes: u32,
}

impl Keep {
    /// `frame`, with everything this does not name read as idle.
    ///
    /// Applied at *delivery* rather than at recording: the stream holds what the
    /// operator did, so a replayed frame and a live one are suppressed by the
    /// same world state on the same tick, and a recording made before a game
    /// grew a menu still means what it meant.
    #[must_use]
    pub fn apply(self, frame: InputFrame) -> InputFrame {
        // Nothing declared is nothing arbitrated — see the type's docs.
        if self.actions == 0 && self.axes == 0 {
            return frame;
        }
        let mut out = InputFrame {
            buttons: frame.buttons & self.actions,
            axes: [0; MAX_AXES],
        };
        for (at, value) in frame.axes.iter().enumerate() {
            if self.axes & (1 << at) != 0 {
                out.axes[at] = *value;
            }
        }
        out
    }

    /// Whether this action survives.
    #[must_use]
    pub const fn action(self, action: ActionId) -> bool {
        self.actions & (1 << action.index()) != 0
    }

    /// Whether this axis survives.
    #[must_use]
    pub const fn axis(self, axis: AxisId) -> bool {
        self.axes & (1 << axis.index()) != 0
    }
}

/// A parsed action map: contexts in declaration order, each with its bindings.
#[derive(Debug)]
pub struct ActionMap {
    contexts: Vec<Context>,
    action_names: Vec<String>,
    axis_names: Vec<String>,
}

/// Why a map failed to load. Every variant names the line and the token, since
/// the file is hand-edited and the parser is the only thing that reads it.
#[derive(Debug, thiserror::Error)]
pub enum MapError {
    /// A line matched none of the three shapes the subset allows.
    #[error(
        "line {line}: expected a `[context.actions]` header or `name = [\"Binding\"]`, got `{text}`"
    )]
    Syntax {
        /// 1-based line number.
        line: usize,
        /// The offending line, trimmed.
        text: String,
    },
    /// A table header was not `context.actions`, `.axes` or `.modal`.
    #[error("line {line}: `[{header}]` — expected `[<context>.actions|axes|modal]`")]
    Header {
        /// 1-based line number.
        line: usize,
        /// The header as written.
        header: String,
    },
    /// A binding appeared before any table header.
    #[error("line {line}: `{name}` sits outside any `[<context>.actions]` table")]
    Homeless {
        /// 1-based line number.
        line: usize,
        /// The binding name.
        name: String,
    },
    /// The verb is not in the list the game declared.
    #[error("line {line}: `{name}` is not a declared {kind} (§4.7: the game's list is the id)")]
    UnknownVerb {
        /// 1-based line number.
        line: usize,
        /// The verb as written.
        name: String,
        /// `"action"` or `"axis"`.
        kind: &'static str,
    },
    /// The binding names no key, button or motion axis we have.
    #[error("line {line}: `{token}` is not a key, mouse button or motion axis")]
    UnknownSource {
        /// 1-based line number.
        line: usize,
        /// The binding as written.
        token: String,
    },
    /// Pointer motion bound to a button-shaped verb, or a sign on motion.
    #[error("line {line}: `{token}` is pointer motion and only an axis can take it")]
    MotionOnAction {
        /// 1-based line number.
        line: usize,
        /// The binding as written.
        token: String,
    },
    /// A `[<context>.modal]` table held a key other than `keep`.
    #[error("line {line}: `{name}` — a `.modal` table holds one key, `keep`")]
    ModalKey {
        /// 1-based line number.
        line: usize,
        /// The key as written.
        name: String,
    },
    /// A `keep` entry named neither a declared action nor a declared axis.
    #[error("line {line}: `{name}` is not a declared action or axis, so nothing can keep it")]
    UnknownKept {
        /// 1-based line number.
        line: usize,
        /// The verb as written.
        name: String,
    },
    /// The game declared more verbs than a recorded frame can carry.
    #[error("{kind} list has {count} entries, past the {max} a replay frame carries")]
    TooManyVerbs {
        /// `"action"` or `"axis"`.
        kind: &'static str,
        /// How many the game declared.
        count: usize,
        /// The ceiling.
        max: usize,
    },
}

impl ActionMap {
    /// Parse a map, resolving verb names against the game's declared lists.
    ///
    /// `actions` and `axes` are the game's verb lists; their *order* is the id
    /// space and is recorded in replay headers. A binding naming a verb outside
    /// them is an error rather than a silent drop.
    pub fn parse(text: &str, actions: &[&str], axes: &[&str]) -> Result<Self, MapError> {
        if actions.len() > MAX_ACTIONS {
            return Err(MapError::TooManyVerbs {
                kind: "action",
                count: actions.len(),
                max: MAX_ACTIONS,
            });
        }
        if axes.len() > MAX_AXES {
            return Err(MapError::TooManyVerbs {
                kind: "axis",
                count: axes.len(),
                max: MAX_AXES,
            });
        }

        let mut map = ActionMap {
            contexts: Vec::new(),
            action_names: actions.iter().map(|s| (*s).to_owned()).collect(),
            axis_names: axes.iter().map(|s| (*s).to_owned()).collect(),
        };
        let mut current: Option<(usize, Section)> = None;

        for (index, raw) in text.lines().enumerate() {
            let line = index + 1;
            let content = strip_comment(raw).trim();
            if content.is_empty() {
                continue;
            }
            if let Some(header) = content.strip_prefix('[').and_then(|h| h.strip_suffix(']')) {
                let (name, section) = header.rsplit_once('.').ok_or_else(|| MapError::Header {
                    line,
                    header: header.to_owned(),
                })?;
                let section = match section {
                    "actions" => Section::Actions,
                    "axes" => Section::Axes,
                    "modal" => Section::Modal,
                    _ => {
                        return Err(MapError::Header {
                            line,
                            header: header.to_owned(),
                        });
                    }
                };
                current = Some((map.context_for(name), section));
                continue;
            }

            let (name, value) = content.split_once('=').ok_or_else(|| MapError::Syntax {
                line,
                text: content.to_owned(),
            })?;
            let name = name.trim();
            let Some((context, section)) = current else {
                return Err(MapError::Homeless {
                    line,
                    name: name.to_owned(),
                });
            };
            let tokens = parse_strings(value.trim()).ok_or_else(|| MapError::Syntax {
                line,
                text: content.to_owned(),
            })?;

            match section {
                Section::Actions => {
                    let action = ActionId::new(find(actions, name).ok_or_else(|| {
                        MapError::UnknownVerb {
                            line,
                            name: name.to_owned(),
                            kind: "action",
                        }
                    })?);
                    for token in tokens {
                        let source = button_source(&token).ok_or_else(|| {
                            if MouseAxis::from_name(&token).is_some() {
                                MapError::MotionOnAction {
                                    line,
                                    token: token.clone(),
                                }
                            } else {
                                MapError::UnknownSource {
                                    line,
                                    token: token.clone(),
                                }
                            }
                        })?;
                        map.contexts[context]
                            .buttons
                            .push(ButtonBinding { source, action });
                    }
                }
                Section::Axes => {
                    let axis =
                        AxisId::new(find(axes, name).ok_or_else(|| MapError::UnknownVerb {
                            line,
                            name: name.to_owned(),
                            kind: "axis",
                        })?);
                    for token in tokens {
                        let (sign, bare) = match token.as_bytes().first() {
                            Some(b'+') => (1, &token[1..]),
                            Some(b'-') => (-1, &token[1..]),
                            _ => (1, token.as_str()),
                        };
                        let source = if let Some(motion) = MouseAxis::from_name(bare) {
                            AxisSource::Motion(motion)
                        } else {
                            AxisSource::Held(button_source(bare).ok_or_else(|| {
                                MapError::UnknownSource {
                                    line,
                                    token: token.clone(),
                                }
                            })?)
                        };
                        map.contexts[context]
                            .axes
                            .push(AxisBinding { source, axis, sign });
                    }
                }
                Section::Modal => {
                    if name != "keep" {
                        return Err(MapError::ModalKey {
                            line,
                            name: name.to_owned(),
                        });
                    }
                    for token in tokens {
                        // Actions first, then axes: the two id spaces are
                        // separate, and a name in neither is the error — a
                        // typo here silently withholds a verb the menu needs.
                        let keep = &mut map.contexts[context].keep;
                        match (find(actions, &token), find(axes, &token)) {
                            (Some(at), _) => keep.actions |= 1 << at,
                            (None, Some(at)) => keep.axes |= 1 << at,
                            (None, None) => {
                                return Err(MapError::UnknownKept { line, name: token });
                            }
                        }
                    }
                }
            }
        }
        Ok(map)
    }

    /// What survives a modal screen across the active layers.
    ///
    /// A union rather than the topmost layer's: a context is pushed to *add* a
    /// way to reach the game (the editor appends four verbs of its own, §6 M15),
    /// and a layer that kept nothing would otherwise silence the one below it.
    #[must_use]
    pub fn keep(&self, stack: &[ContextId]) -> Keep {
        stack.iter().fold(Keep::default(), |acc, c| {
            let keep = self.contexts[c.index()].keep;
            Keep {
                actions: acc.actions | keep.actions,
                axes: acc.axes | keep.axes,
            }
        })
    }

    /// Apply the player's own bindings over the map, reporting what it could not
    /// use (§6 M45).
    ///
    /// Same format as the map itself, because it *is* the map's format: a
    /// `[<context>.actions]` table whose rows **replace** that verb's bindings in
    /// that context wholesale rather than adding to them, so a player who
    /// rebinds `left` gets the keys they named and not those plus `A`.
    ///
    /// **Nothing here is fatal**, which is `settings.cfg`'s policy (§6 M42) and
    /// not `--load`'s: this is a file the player owns, editing it wrongly must
    /// not stop the game starting, and a line that means nothing to this build is
    /// named and dropped. A `.modal` table is refused among them — which verbs a
    /// menu answers is the game's arbitration, not a preference (see [`Keep`]).
    pub fn overlay(&mut self, text: &str) -> Vec<String> {
        let mut stale = Vec::new();
        let mut current: Option<(usize, Section)> = None;
        for raw in text.lines() {
            let content = strip_comment(raw).trim();
            if content.is_empty() {
                continue;
            }
            if let Some(header) = content.strip_prefix('[').and_then(|h| h.strip_suffix(']')) {
                current = match header.rsplit_once('.') {
                    Some((name, "actions")) => {
                        self.context(name).map(|c| (c.index(), Section::Actions))
                    }
                    Some((name, "axes")) => self.context(name).map(|c| (c.index(), Section::Axes)),
                    _ => None,
                };
                if current.is_none() {
                    stale.push(header.to_owned());
                }
                continue;
            }
            let Some((name, value)) = content.split_once('=') else {
                stale.push(content.to_owned());
                continue;
            };
            let name = name.trim();
            let Some((context, section)) = current else {
                stale.push(name.to_owned());
                continue;
            };
            let Some(tokens) = parse_strings(value.trim()) else {
                stale.push(name.to_owned());
                continue;
            };
            if !self.replace(context, section, name, &tokens) {
                stale.push(name.to_owned());
            }
        }
        stale
    }

    /// One overlay row: resolve the verb and its sources, then swap them in.
    /// All-or-nothing per row — a rebind half of whose keys are misspelled would
    /// otherwise leave a verb reachable by an arbitrary subset of them.
    fn replace(&mut self, context: usize, section: Section, name: &str, tokens: &[String]) -> bool {
        match section {
            Section::Actions => {
                let Some(at) = find_owned(&self.action_names, name) else {
                    return false;
                };
                let action = ActionId::new(at);
                let Some(sources) = tokens
                    .iter()
                    .map(|t| button_source(t))
                    .collect::<Option<Vec<_>>>()
                else {
                    return false;
                };
                let context = &mut self.contexts[context];
                context.buttons.retain(|b| b.action != action);
                context.buttons.extend(
                    sources
                        .into_iter()
                        .map(|source| ButtonBinding { source, action }),
                );
                true
            }
            Section::Axes => {
                let Some(at) = find_owned(&self.axis_names, name) else {
                    return false;
                };
                let axis = AxisId::new(at);
                let mut bound = Vec::new();
                for token in tokens {
                    let (sign, bare) = match token.as_bytes().first() {
                        Some(b'+') => (1, &token[1..]),
                        Some(b'-') => (-1, &token[1..]),
                        _ => (1, token.as_str()),
                    };
                    let source = match MouseAxis::from_name(bare) {
                        Some(motion) => AxisSource::Motion(motion),
                        None => match button_source(bare) {
                            Some(source) => AxisSource::Held(source),
                            None => return false,
                        },
                    };
                    bound.push(AxisBinding { source, axis, sign });
                }
                let context = &mut self.contexts[context];
                context.axes.retain(|a| a.axis != axis);
                context.axes.extend(bound);
                true
            }
            // Unreachable: `overlay` never enters this section. Refusing here
            // rather than there keeps the policy in one place.
            Section::Modal => false,
        }
    }

    /// Every binding this map holds, as the boundary carries them (§6 M45) —
    /// each action's spellings in the map's own order, so a legend reads left to
    /// right the way the file does.
    ///
    /// The whole map rather than the active stack: a game asks what a verb is
    /// called, and the answer must not change with which layer happens to be
    /// pushed when the menu is built.
    #[must_use]
    pub fn spellings(&self) -> Vec<gg_abi::Binding> {
        let mut out = Vec::new();
        for context in &self.contexts {
            for binding in &context.buttons {
                out.push(gg_abi::Binding::new(binding.action, &spell(binding.source)));
            }
        }
        out
    }

    /// The id of a context by name, if the map declares one.
    pub fn context(&self, name: &str) -> Option<ContextId> {
        self.contexts
            .iter()
            .position(|c| c.name == name)
            .map(|i| ContextId(i as u8))
    }

    /// The declared action names, in id order — what a replay header carries.
    pub fn action_names(&self) -> &[String] {
        &self.action_names
    }

    /// The declared axis names, in id order.
    pub fn axis_names(&self) -> &[String] {
        &self.axis_names
    }

    /// How many contexts the map declares.
    pub fn context_count(&self) -> usize {
        self.contexts.len()
    }

    /// Whether any active layer binds `source` — the consumption test asked
    /// *before* the event, which is what a host needs to know whether a key is
    /// still its own.
    ///
    /// The shell owns `Escape` (quit) only for as long as the game does not want
    /// it: a game with a pause menu binds it, and a key bound to a verb has to
    /// reach the action map rather than being eaten one layer above it. Same
    /// rule this module's header states for layering, applied to the host as the
    /// bottom layer.
    pub fn claims(&self, stack: &[ContextId], source: Source) -> bool {
        self.consumer(stack, source).is_some()
    }

    /// Whether any active layer binds `motion` to an axis — "this game does
    /// mouse-look", asked of the bindings rather than guessed from behaviour.
    ///
    /// Raw device motion, not the steered pointer: the two [`MouseAxis`] pairs
    /// are what tell a camera apart from a cursor (§4.9), which is exactly the
    /// distinction a host needs to decide whether to hold the mouse at all.
    pub fn claims_motion(&self, stack: &[ContextId], motion: MouseAxis) -> bool {
        self.motion_axes(stack, motion).next().is_some()
    }

    /// The topmost context in `stack` that binds `source`, or `None` when no
    /// active layer wants it. Layering lives here and nowhere else.
    fn consumer(&self, stack: &[ContextId], source: Source) -> Option<usize> {
        stack
            .iter()
            .rev()
            .map(|c| c.index())
            .find(|&c| self.contexts[c].binds(source))
    }

    /// Actions `source` triggers in the layer that consumes it.
    pub(crate) fn actions_for<'a>(
        &'a self,
        stack: &'a [ContextId],
        source: Source,
    ) -> impl Iterator<Item = ActionId> + 'a {
        let context = self.consumer(stack, source);
        context
            .into_iter()
            .flat_map(move |c| self.contexts[c].buttons.iter())
            .filter(move |b| b.source == source)
            .map(|b| b.action)
    }

    /// Axis contributions `source` makes while held, in the consuming layer.
    pub(crate) fn axes_for<'a>(
        &'a self,
        stack: &'a [ContextId],
        source: Source,
    ) -> impl Iterator<Item = (AxisId, i32)> + 'a {
        let context = self.consumer(stack, source);
        context
            .into_iter()
            .flat_map(move |c| self.contexts[c].axes.iter())
            .filter(move |a| a.source == AxisSource::Held(source))
            .map(|a| (a.axis, a.sign))
    }

    /// Axes fed by pointer motion in any active layer. Motion is not consumed
    /// by layering — it has no key to compete over, and a UI layer that wanted
    /// to swallow it simply does not bind it.
    pub(crate) fn motion_axes<'a>(
        &'a self,
        stack: &'a [ContextId],
        motion: MouseAxis,
    ) -> impl Iterator<Item = (AxisId, i32)> + 'a {
        stack
            .iter()
            .map(|c| c.index())
            .flat_map(move |c| self.contexts[c].axes.iter())
            .filter(move |a| a.source == AxisSource::Motion(motion))
            .map(|a| (a.axis, a.sign))
    }

    fn context_for(&mut self, name: &str) -> usize {
        if let Some(at) = self.contexts.iter().position(|c| c.name == name) {
            return at;
        }
        self.contexts.push(Context {
            name: name.to_owned(),
            buttons: Vec::new(),
            axes: Vec::new(),
            keep: Keep::default(),
        });
        self.contexts.len() - 1
    }
}

#[derive(Clone, Copy)]
enum Section {
    Actions,
    Axes,
    Modal,
}

fn find(names: &[&str], name: &str) -> Option<usize> {
    names.iter().position(|n| *n == name)
}

fn find_owned(names: &[String], name: &str) -> Option<usize> {
    names.iter().position(|n| n == name)
}

/// A source as the config spells it — the inverse of [`button_source`], and the
/// only thing that turns a parsed map back into something a player can read.
fn spell(source: Source) -> String {
    match source {
        Source::Key(key) => key.name().to_owned(),
        Source::Wheel(wheel) => wheel.name().to_owned(),
        Source::Button(button) => format!("Mouse{}", button.number()),
    }
}

fn button_source(token: &str) -> Option<Source> {
    if let Some(key) = Key::from_name(token) {
        return Some(Source::Key(key));
    }
    if let Some(wheel) = Wheel::from_name(token) {
        return Some(Source::Wheel(wheel));
    }
    MouseButton::from_name(token).map(Source::Button)
}

/// Everything before the first `#` that is not inside a string.
fn strip_comment(line: &str) -> &str {
    let mut quoted = false;
    for (at, c) in line.char_indices() {
        match c {
            '"' => quoted = !quoted,
            '#' if !quoted => return &line[..at],
            _ => {}
        }
    }
    line
}

/// `"one"` or `["one", "two"]` → the strings. No escapes: a binding name is a
/// key spelling, and permitting `\"` in one would be format surface with no use.
fn parse_strings(value: &str) -> Option<Vec<String>> {
    let inner = match value.strip_prefix('[') {
        Some(rest) => rest.strip_suffix(']')?,
        None => value,
    };
    let mut out = Vec::new();
    let mut rest = inner.trim();
    while !rest.is_empty() {
        let quoted = rest.strip_prefix('"')?;
        let (token, tail) = quoted.split_once('"')?;
        out.push(token.to_owned());
        rest = tail.trim_start().strip_prefix(',').unwrap_or(tail).trim();
    }
    (!out.is_empty()).then_some(out)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// What a shell asks before treating a key as its own (§6 M21): Escape is
    /// the app's until a game's map binds it. Asked of the *active* stack, so a
    /// context that is not pushed does not claim anything.
    #[test]
    fn a_key_is_the_apps_until_an_active_layer_binds_it() {
        const PAUSED: &str = r#"
            [game.actions]
            look = ["Escape"]
        "#;
        let map = ActionMap::parse(PAUSED, ACTIONS, AXES).unwrap();
        let game = map.context("game").unwrap();
        assert!(!map.claims(&[], Source::Key(Key::Escape)), "no layer is up");
        assert!(map.claims(&[game], Source::Key(Key::Escape)));
        assert!(!map.claims(&[game], Source::Key(Key::Tab)));
    }

    /// The subset's one real edge, refused by line rather than half-read (§6
    /// M74). A TOML formatter wraps a long array without asking, and the file
    /// that comes back is valid TOML this parser cannot read — so what a wrapped
    /// array must never be is *partly* accepted, leaving a game bound to the
    /// first few keys of a list and silently missing the rest.
    #[test]
    fn an_array_wrapped_across_lines_is_refused_and_says_which_line() {
        const WRAPPED: &str = "[game.actions]\nlook = [\n  \"Tab\",\n  \"Escape\",\n]\n";
        let err = ActionMap::parse(WRAPPED, ACTIONS, AXES).unwrap_err();
        // Line 2 — where the array *opens*, which is where a reader looks,
        // rather than line 3 where the first orphaned element is. That is the
        // value shape being checked whole rather than the tokens being swept up
        // one at a time, and it is the difference between a message that names
        // the mistake and one that names its second symptom.
        assert!(
            matches!(&err, MapError::Syntax { line: 2, .. }),
            "a wrapped array should be refused by line: {err}"
        );
        // And the one-line spelling of the same intent is fine, which is what
        // says the refusal is about the wrapping and not about the length.
        let map = ActionMap::parse(
            "[game.actions]\nlook = [\"Tab\", \"Escape\"]\n",
            ACTIONS,
            AXES,
        )
        .unwrap();
        let game = map.context("game").unwrap();
        assert!(map.claims(&[game], Source::Key(Key::Escape)));
        assert!(map.claims(&[game], Source::Key(Key::Tab)));
    }

    /// And whether the game does mouse-look at all, which is a question about
    /// *raw* motion: a menu binds the steered pointer and must not be mistaken
    /// for a camera.
    #[test]
    fn raw_motion_is_what_says_a_game_does_mouse_look() {
        const BOTH: &str = r#"
            [game.axes]
            look_x = ["MouseX"]
            move_right = ["PointerX"]
        "#;
        let map = ActionMap::parse(BOTH, ACTIONS, AXES).unwrap();
        let game = map.context("game").unwrap();
        assert!(map.claims_motion(&[game], MouseAxis::X));
        assert!(!map.claims_motion(&[game], MouseAxis::Y));
        assert!(!map.claims_motion(&[], MouseAxis::X), "no layer is up");

        const POINTER_ONLY: &str = r#"
            [game.axes]
            move_right = ["PointerX"]
            look_x = ["PointerY"]
        "#;
        let map = ActionMap::parse(POINTER_ONLY, ACTIONS, AXES).unwrap();
        let game = map.context("game").unwrap();
        assert!(
            !map.claims_motion(&[game], MouseAxis::X) && !map.claims_motion(&[game], MouseAxis::Y),
            "a pointer is not a camera"
        );
    }

    const ACTIONS: &[&str] = &["look", "spawn"];
    const AXES: &[&str] = &["move_right", "look_x"];

    const MAP: &str = r#"
        # demo bindings
        [game.actions]
        look = ["Tab"]            # held
        spawn = ["F", "Mouse1"]

        [game.axes]
        move_right = ["+D", "-A"]
        look_x = ["MouseX"]

        [ui.actions]
        spawn = ["Tab"]
    "#;

    fn map() -> ActionMap {
        ActionMap::parse(MAP, ACTIONS, AXES).unwrap()
    }

    /// A button past what a held state is kept for is refused at the **name**,
    /// where a line number exists to name it (§6 M81).
    ///
    /// It parsed and bound until then, and was dropped at the edge by a bounds
    /// check that had nothing to say — so `Mouse17` in a player's own
    /// `bindings.cfg` was a line that did nothing, in a file that was accepted.
    /// The boundary is exercised from both sides because an off-by-one here
    /// silently costs a real button instead.
    #[test]
    fn a_button_no_state_is_kept_for_is_refused_by_name() {
        assert!(MouseButton::from_name("Mouse16").is_some(), "the last one");
        assert!(MouseButton::from_name("Mouse17").is_none());
        let line = |token| format!("[game.actions]\nspawn = [\"{token}\"]\n");
        assert!(ActionMap::parse(&line("Mouse16"), ACTIONS, AXES).is_ok());
        let refused = ActionMap::parse(&line("Mouse17"), ACTIONS, AXES);
        assert!(
            matches!(&refused, Err(MapError::UnknownSource { token, .. }) if token == "Mouse17"),
            "{refused:?}"
        );
        // And every number the map hands back is one the state layer keeps, so
        // the two spellings cannot drift apart.
        for n in 1..=MouseButton::TRACKED {
            let button = MouseButton::from_number(n).unwrap();
            assert_eq!(button.number(), n);
            assert_eq!(
                MouseButton::from_name(&spell(Source::Button(button))),
                Some(button)
            );
        }
    }

    #[test]
    fn bindings_resolve_to_declared_verbs() {
        let m = map();
        let game = vec![m.context("game").unwrap()];
        let acts: Vec<_> = m.actions_for(&game, Source::Key(Key::Tab)).collect();
        assert_eq!(acts, vec![ActionId::new(0)]);
        let acts: Vec<_> = m
            .actions_for(&game, Source::Button(MouseButton::Left))
            .collect();
        assert_eq!(acts, vec![ActionId::new(1)]);
        let axes: Vec<_> = m.axes_for(&game, Source::Key(Key::A)).collect();
        assert_eq!(axes, vec![(AxisId::new(0), -1)]);
        let axes: Vec<_> = m.motion_axes(&game, MouseAxis::X).collect();
        assert_eq!(axes, vec![(AxisId::new(1), 1)]);
    }

    #[test]
    fn the_topmost_context_that_binds_a_key_consumes_it() {
        let m = map();
        let (game, ui) = (m.context("game").unwrap(), m.context("ui").unwrap());
        // ui binds Tab to `spawn`; with ui on top the game never sees `look`.
        let stack = vec![game, ui];
        let acts: Vec<_> = m.actions_for(&stack, Source::Key(Key::Tab)).collect();
        assert_eq!(acts, vec![ActionId::new(1)]);
        // ...and a key ui does not bind falls through to game.
        let axes: Vec<_> = m.axes_for(&stack, Source::Key(Key::D)).collect();
        assert_eq!(axes, vec![(AxisId::new(0), 1)]);
        // An empty stack is a map that binds nothing, not a map that binds all.
        assert_eq!(m.actions_for(&[], Source::Key(Key::Tab)).count(), 0);
    }

    #[test]
    fn every_bad_line_names_itself() {
        let cases: [(&str, &str); 5] = [
            ("[game.verbs]\nlook = [\"Tab\"]", "expected `[<context>"),
            ("[game.actions]\nlook", "expected a `[context.actions]`"),
            ("look = [\"Tab\"]", "outside any"),
            ("[game.actions]\njump = [\"Tab\"]", "not a declared action"),
            ("[game.actions]\nlook = [\"Sneeze\"]", "not a key"),
        ];
        for (text, needle) in cases {
            let err = ActionMap::parse(text, ACTIONS, AXES).unwrap_err();
            assert!(
                err.to_string().contains(needle),
                "`{text}` reported `{err}`, expected `{needle}`"
            );
        }
    }

    #[test]
    fn pointer_motion_is_refused_as_a_button() {
        let err =
            ActionMap::parse("[game.actions]\nlook = [\"MouseX\"]", ACTIONS, AXES).unwrap_err();
        assert!(err.to_string().contains("only an axis"), "{err}");
    }

    /// The arbitration M45 exists for: a menu keeps the verbs that reach it and
    /// nothing else, across both id spaces. Everything unnamed reads idle, which
    /// is the early return five demo-12 systems used to write by hand.
    #[test]
    fn a_modal_screen_keeps_what_the_map_names_and_withholds_the_rest() {
        const WITH_MENU: &str = r#"
            [game.actions]
            look = ["Tab"]
            spawn = ["F"]
            [game.axes]
            move_right = ["+D"]
            look_x = ["PointerX"]
            [game.modal]
            keep = ["look", "look_x"]
        "#;
        let menu = ActionMap::parse(WITH_MENU, ACTIONS, AXES).unwrap();
        let game = menu.context("game").unwrap();
        let keep = menu.keep(&[game]);
        assert!(keep.action(ActionId::new(0)) && !keep.action(ActionId::new(1)));
        assert!(keep.axis(AxisId::new(1)) && !keep.axis(AxisId::new(0)));

        let mut axes = [0; MAX_AXES];
        (axes[0], axes[1]) = (700, -300);
        let live = InputFrame {
            buttons: 0b11,
            axes,
        };
        let held = keep.apply(live);
        assert_eq!(held.buttons, 0b01, "only `look` crosses");
        assert_eq!(held.axes[0], 0, "a gameplay axis reads still, not stale");
        assert_eq!(held.axes[1], -300, "and the pointer keeps moving");
        // A map with no `keep` list withholds *nothing*: arbitration is opt-in
        // per map, so a game that raises the flag under one behaves as it did
        // before M45, and so does a replay-driven session given no map at all.
        assert_eq!(map().keep(&[game]).apply(live), live);
        // And nothing is withheld unless a screen is up — the flag is the game's.
        assert_eq!(keep.apply(InputFrame::default()), InputFrame::default());
    }

    #[test]
    fn a_modal_table_names_its_own_mistakes() {
        for (text, needle) in [
            ("[game.modal]\ndrop = [\"look\"]", "one key, `keep`"),
            ("[game.modal]\nkeep = [\"sneeze\"]", "nothing can keep it"),
        ] {
            let err = ActionMap::parse(text, ACTIONS, AXES).unwrap_err();
            assert!(err.to_string().contains(needle), "{err}");
        }
    }

    /// The player's own file (§6 M45): it *replaces* a verb's keys rather than
    /// adding to them, it may not touch arbitration, and nothing in it is fatal.
    #[test]
    fn a_players_bindings_replace_a_verbs_keys_and_never_stop_the_game() {
        let mut m = map();
        let game = vec![m.context("game").unwrap()];
        let stale = m.overlay("[game.actions]\nlook = [\"Escape\", \"Mouse2\"]\n");
        assert!(stale.is_empty());
        let at = |m: &ActionMap, key| m.actions_for(&game, Source::Key(key)).collect::<Vec<_>>();
        assert_eq!(at(&m, Key::Escape), vec![ActionId::new(0)]);
        assert!(at(&m, Key::Tab).is_empty(), "the old key is gone, not kept");
        assert_eq!(
            m.actions_for(&game, Source::Button(MouseButton::Right))
                .count(),
            1
        );

        // Every way a hand-edited file goes wrong, each named and each survived.
        let mut m = map();
        let stale = m.overlay(concat!(
            "[nowhere.actions]\nlook = [\"Escape\"]\n",
            "[game.actions]\njump = [\"Escape\"]\nlook = [\"Sneeze\"]\nspawn\n",
            "[game.modal]\nkeep = [\"look\"]\n",
            "[game.axes]\nlook_x = [\"MouseY\"]\n",
        ));
        // The unknown header and each row under it: every line that did
        // nothing is named, so a player fixing the file is told all of it.
        assert_eq!(
            stale,
            [
                "nowhere.actions",
                "look",
                "jump",
                "look",
                "spawn",
                "game.modal",
                "keep"
            ]
        );
        assert_eq!(
            at(&m, Key::Tab),
            vec![ActionId::new(0)],
            "and nothing moved"
        );
        // The one good row still landed, which is what makes the rest droppable.
        assert_eq!(
            m.motion_axes(&game, MouseAxis::Y).collect::<Vec<_>>(),
            vec![(AxisId::new(1), 1)]
        );
        // A row whose keys are half wrong changes nothing rather than binding
        // the half that parsed.
        let mut m = map();
        assert_eq!(
            m.overlay("[game.actions]\nspawn = [\"F\", \"Sneeze\"]\n"),
            ["spawn"]
        );
        assert_eq!(
            m.actions_for(&game, Source::Button(MouseButton::Left))
                .count(),
            1,
            "`Mouse1` is still `spawn`'s"
        );
    }

    /// What the boundary carries: every binding, by action, in the file's order.
    #[test]
    fn the_spellings_a_game_draws_its_legend_from_are_the_files_own() {
        let m = map();
        let of = |action: usize| {
            m.spellings()
                .into_iter()
                .filter(|b| b.action as usize == action)
                .map(|b| b.name().to_owned())
                .collect::<Vec<_>>()
        };
        // `look` is Tab in `game` and nothing in `ui`; `spawn` is bound in both.
        assert_eq!(of(0), ["Tab"]);
        assert_eq!(of(1), ["F", "Mouse1", "Tab"]);
        // A rebind is visible here, which is what makes the legend the player's.
        let mut m = map();
        m.overlay("[game.actions]\nlook = [\"Left\"]\n");
        assert!(
            m.spellings()
                .iter()
                .any(|b| b.action == 0 && b.name() == "Left")
        );
    }

    #[test]
    fn comments_and_quoting_survive_each_other() {
        assert_eq!(strip_comment("a = [\"#\"] # trailing"), "a = [\"#\"] ");
        assert_eq!(parse_strings("[\"A\", \"B\"]").unwrap(), ["A", "B"]);
        assert_eq!(parse_strings("\"A\"").unwrap(), ["A"]);
        assert!(parse_strings("[]").is_none());
    }
}
