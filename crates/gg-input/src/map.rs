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
//! string or an array of strings. Editors and formatters treat it as TOML
//! because it is TOML; the parser is ~150 lines because the subset is small.
//! The alternative was `toml` + `serde` in the dist graph of every shipped
//! game, for a file with three shapes in it. Should the format ever need
//! genuine TOML, swapping the crate in is mechanical and the format does not
//! move under anyone.
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

use crate::key::{Key, MouseAxis, MouseButton};

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
    /// `+1` or `-1`; ignored for [`AxisSource::Motion`], which carries its sign.
    sign: i32,
}

#[derive(Debug)]
struct Context {
    name: String,
    buttons: Vec<ButtonBinding>,
    axes: Vec<AxisBinding>,
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
    /// A table header was not `context.actions` or `context.axes`.
    #[error("line {line}: `[{header}]` — expected `[<context>.actions]` or `[<context>.axes]`")]
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
            }
        }
        Ok(map)
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
        });
        self.contexts.len() - 1
    }
}

#[derive(Clone, Copy)]
enum Section {
    Actions,
    Axes,
}

fn find(names: &[&str], name: &str) -> Option<usize> {
    names.iter().position(|n| *n == name)
}

fn button_source(token: &str) -> Option<Source> {
    if let Some(key) = Key::from_name(token) {
        return Some(Source::Key(key));
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

    #[test]
    fn comments_and_quoting_survive_each_other() {
        assert_eq!(strip_comment("a = [\"#\"] # trailing"), "a = [\"#\"] ");
        assert_eq!(parse_strings("[\"A\", \"B\"]").unwrap(), ["A", "B"]);
        assert_eq!(parse_strings("\"A\"").unwrap(), ["A"]);
        assert!(parse_strings("[]").is_none());
    }
}
