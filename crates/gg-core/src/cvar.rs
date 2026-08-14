//! The CVar registry (§4.8) — runtime-tweakable variables any crate can declare.
//!
//! The registry is homed here rather than in `gg-debug` for a dependency-direction
//! reason (§4.8): every crate that declares a CVar would otherwise point its
//! arrow at the observability crate, `gg-rhi` included. The *console* — parsing,
//! UI, and the tab-completion nobody needs at 3 a.m. — stays in `gg-debug` and
//! compiles out in dist. Config-file CVars remain in every tier, which is why
//! [`crate::config`] sits beside this and not behind a feature.
//!
//! # Declaring one
//!
//! ```
//! use gg_core::cvar::{self, CVar};
//!
//! static VSYNC: CVar = CVar::new_bool("r.vsync", true, "wait for vertical blank");
//!
//! cvar::register(&VSYNC)?;
//! assert!(VSYNC.bool());
//! # Ok::<(), gg_core::cvar::CVarError>(())
//! ```
//!
//! A `static` with interior mutability, not a handle into a table: reads are a
//! relaxed atomic load at the use site with no lookup, no lock, and no chance of
//! a stale handle. The registry exists for the *by-name* paths — config file,
//! CLI, console — and for enumeration.
//!
//! Registration is an explicit call. Link-time registration tricks (a distributed
//! slice, a `ctor` shim) would make the set of live CVars depend on link order and
//! on which objects the linker kept, which is a worse property than typing one
//! line per crate.
//!
//! # CVars are not sim state
//!
//! Nothing here is hashed or snapshotted. A CVar read inside a sim tick would
//! make that tick depend on a config file, and the replay would reproduce only on
//! machines with the same one — so sim code does not read them. The types stop at
//! `bool`/`i64`/`f64` partly for that reason: a CVar is a knob, and anything that
//! needs a string wants config or an asset, not a knob.
//!
//! **What that rule does not cover, and §6 M40 does:** the pipeline turning a
//! recorded *click* into a world write is parameterized by knobs even though the
//! sim is not. `r.fov` and `r.near` build the editor's pick ray, `d.editor_scale`
//! divides a physical click down to a logical one, `d.editor_undo` decides
//! whether the twentieth undo restores anything. Those four declare themselves
//! [`CVar::recorded`], which puts them in a replay's own channel — recorded when
//! they move, applied at the tick they moved on, and [`CVarSource::Replay`] after
//! that so the listing says who moved it.

use std::sync::RwLock;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

/// What a [`CVar`]'s bits mean.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CVarKind {
    /// `0`/`1`/`true`/`false`/`on`/`off`/`yes`/`no`.
    Bool,
    /// A signed 64-bit integer.
    Int,
    /// A finite `f64`.
    Float,
}

impl CVarKind {
    /// The word this kind goes by in an error message.
    pub fn as_str(self) -> &'static str {
        match self {
            CVarKind::Bool => "bool",
            CVarKind::Int => "int",
            CVarKind::Float => "float",
        }
    }
}

/// Who last set a CVar (§4.8). Recorded per variable rather than per pass,
/// because the question a session log has to answer is "why is this 0", and the
/// value never answers it — the source does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CVarSource {
    /// Never set: still the value its declaration gave it.
    Default = 0,
    /// A `name = value` line in the config file.
    Config = 1,
    /// A `--set name=value` on the command line.
    Cli = 2,
    /// Typed at the console. Dev tiers only — dist has no console, so a dist
    /// session cannot produce this and a report claiming it is from a dev build.
    Console = 3,
    /// A typed setter. Engine code holding a value already is a source like any
    /// other, and calling it anything else would hide the one case where a knob
    /// moved with nobody asking.
    Code = 4,
    /// Applied out of a replay's knob channel (§6 M40). The one source whose
    /// value is otherwise inexplicable — nobody on this machine set it — which
    /// is the case this enum exists for.
    Replay = 5,
    /// A `name = value` line in the player's own `gg.cfg`, beside their saves
    /// rather than beside the game (§6 M42). Distinct from [`Self::Config`] and
    /// not a spelling of it: a bug report from a shipped build has to be able to
    /// say a knob came from a file the operator never wrote and cannot see.
    Player = 6,
}

impl CVarSource {
    /// The word this source goes by in a log line.
    pub fn as_str(self) -> &'static str {
        match self {
            CVarSource::Default => "default",
            CVarSource::Config => "config",
            CVarSource::Cli => "cli",
            CVarSource::Console => "console",
            CVarSource::Code => "code",
            CVarSource::Replay => "replay",
            CVarSource::Player => "player",
        }
    }

    /// Unknown bits read back as [`CVarSource::Default`]: the field is only ever
    /// written from this enum, so the arm is unreachable rather than lossy.
    fn from_bits(bits: u8) -> Self {
        match bits {
            1 => CVarSource::Config,
            2 => CVarSource::Cli,
            3 => CVarSource::Console,
            4 => CVarSource::Code,
            5 => CVarSource::Replay,
            6 => CVarSource::Player,
            _ => CVarSource::Default,
        }
    }
}

/// A registered runtime variable.
///
/// Declare it as a `static`; the value lives in the `static`, so a read is one
/// relaxed load. Relaxed is the whole ordering story on purpose: a CVar carries
/// no happens-before relationship to anything, and a knob whose new value lands
/// one frame later is a knob working correctly.
#[derive(Debug)]
pub struct CVar {
    name: &'static str,
    help: &'static str,
    kind: CVarKind,
    default: u64,
    bits: AtomicU64,
    /// A [`CVarSource`] discriminant. Separate from `bits` on purpose: pairing
    /// them in one word would cap values at 56 bits to buy an atomicity nothing
    /// reads them as a pair anyway.
    source: AtomicU8,
    /// Set by [`CVar::recorded`]. Not atomic and not settable: whether a knob
    /// reaches a click is a property of the code reading it, so it is decided
    /// where it is declared and never at runtime.
    recorded: bool,
}

impl CVar {
    /// Declare a boolean knob. `const` so it can be a `static`, which is the
    /// only way it is meant to be declared.
    pub const fn new_bool(name: &'static str, default: bool, help: &'static str) -> Self {
        Self::new(name, CVarKind::Bool, default as u64, help)
    }

    /// Declare an integer knob.
    pub const fn new_int(name: &'static str, default: i64, help: &'static str) -> Self {
        Self::new(name, CVarKind::Int, default as u64, help)
    }

    /// Declare a float knob. Non-finite values are refused on the way in, so a
    /// non-finite default is a bug this cannot catch.
    pub const fn new_float(name: &'static str, default: f64, help: &'static str) -> Self {
        Self::new(name, CVarKind::Float, default.to_bits(), help)
    }

    const fn new(name: &'static str, kind: CVarKind, default: u64, help: &'static str) -> Self {
        Self {
            name,
            help,
            kind,
            default,
            bits: AtomicU64::new(default),
            source: AtomicU8::new(CVarSource::Default as u8),
            recorded: false,
        }
    }

    /// Declare that this knob reaches a *recorded input* — that moving it moves
    /// where a click lands or what it hits — so a replay must carry it (§6 M40).
    ///
    /// Chained onto the declaration (`CVar::new_float(…).recorded()`), which is
    /// the only place the question can be answered: the registry cannot see who
    /// reads a value or what they do with it. The set is small and reviewed —
    /// `xtask ci` compares it against `crates/gg-core/recorded-cvars.txt`, so
    /// adding one is a diff rather than a remembered sentence.
    #[must_use]
    pub const fn recorded(self) -> Self {
        // Destructured rather than `..self`: a functional update would move out
        // of a type holding atomics in const context.
        let CVar {
            name,
            help,
            kind,
            default,
            bits,
            source,
            ..
        } = self;
        CVar {
            name,
            help,
            kind,
            default,
            bits,
            source,
            recorded: true,
        }
    }

    /// Whether [`CVar::recorded`] declared this one.
    pub fn is_recorded(&self) -> bool {
        self.recorded
    }

    /// The declared default as text, in [`CVar::to_text`]'s spelling — what a
    /// [`Watch`] starts from, so the opening values fall out of the diff instead
    /// of being a second mechanism.
    fn default_text(&self) -> String {
        match self.kind {
            CVarKind::Bool => if self.default != 0 { "1" } else { "0" }.to_owned(),
            CVarKind::Int => (self.default as i64).to_string(),
            CVarKind::Float => f64::from_bits(self.default).to_string(),
        }
    }

    fn store(&self, bits: u64, source: CVarSource) {
        self.bits.store(bits, Ordering::Relaxed);
        self.source.store(source as u8, Ordering::Relaxed);
    }

    /// The name config, CLI and console reach it by.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// One line for the console listing.
    pub fn help(&self) -> &'static str {
        self.help
    }

    /// Which accessor is the right one.
    pub fn kind(&self) -> CVarKind {
        self.kind
    }

    /// Who set the current value. [`CVarSource::Default`] until something does.
    pub fn source(&self) -> CVarSource {
        CVarSource::from_bits(self.source.load(Ordering::Relaxed))
    }

    /// Reading with the wrong accessor is a `debug_assert` and, past that, a
    /// reinterpretation of the bits — the declaration and its readers are the
    /// same crate, usually the same screen, so this is a typo caught in dev
    /// rather than a `Result` every call site has to carry.
    pub fn bool(&self) -> bool {
        debug_assert_eq!(
            self.kind,
            CVarKind::Bool,
            "cvar `{}` is not a bool",
            self.name
        );
        self.bits.load(Ordering::Relaxed) != 0
    }

    /// See [`CVar::bool`] for what a wrong-kind read does.
    pub fn int(&self) -> i64 {
        debug_assert_eq!(
            self.kind,
            CVarKind::Int,
            "cvar `{}` is not an int",
            self.name
        );
        self.bits.load(Ordering::Relaxed) as i64
    }

    /// See [`CVar::bool`] for what a wrong-kind read does.
    pub fn float(&self) -> f64 {
        debug_assert_eq!(
            self.kind,
            CVarKind::Float,
            "cvar `{}` is not a float",
            self.name
        );
        f64::from_bits(self.bits.load(Ordering::Relaxed))
    }

    /// Set directly, skipping the parse. Code paths that hold a value already,
    /// and recorded as [`CVarSource::Code`] — see that variant for why.
    pub fn set_bool(&self, value: bool) {
        self.store(value as u64, CVarSource::Code);
    }

    /// Set directly, skipping the parse.
    pub fn set_int(&self, value: i64) {
        self.store(value as u64, CVarSource::Code);
    }

    /// Set directly, skipping the parse — including the finite check, which is
    /// [`CVar::set_from_str`]'s and not the type's.
    pub fn set_float(&self, value: f64) {
        self.store(value.to_bits(), CVarSource::Code);
    }

    /// Back to the declared default, source included. Not "the value at
    /// startup" — a config file applied over it is a setting like any other.
    pub fn reset(&self) {
        self.store(self.default, CVarSource::Default);
    }

    /// Whether the current value is still the declared one — what a config
    /// writer uses to write only what was changed.
    pub fn is_default(&self) -> bool {
        self.bits.load(Ordering::Relaxed) == self.default
    }

    /// The current value as the console and a written-back config file spell it.
    pub fn to_text(&self) -> String {
        match self.kind {
            CVarKind::Bool => if self.bool() { "1" } else { "0" }.to_owned(),
            CVarKind::Int => self.int().to_string(),
            CVarKind::Float => self.float().to_string(),
        }
    }

    /// Parse and store — the one entry point config, CLI and console share, so
    /// the three cannot drift on what `on` means, and so `source` is recorded on
    /// exactly the path that can set a knob without code saying so.
    ///
    /// A rejected value leaves both the value and the source untouched: a typo
    /// is not a source.
    pub fn set_from_str(&self, text: &str, source: CVarSource) -> Result<(), CVarError> {
        let text = text.trim();
        let bad = || CVarError::BadValue {
            name: self.name,
            kind: self.kind,
            value: text.to_owned(),
        };
        let bits = match self.kind {
            CVarKind::Bool => match text {
                "1" | "true" | "on" | "yes" => 1,
                "0" | "false" | "off" | "no" => 0,
                _ => return Err(bad()),
            },
            CVarKind::Int => text.parse::<i64>().map_err(|_| bad())? as u64,
            // Rejects `inf` and `NaN` by name: a knob set to either is a bug
            // report about the renderer three days later, not a setting.
            CVarKind::Float => {
                let value = text.parse::<f64>().map_err(|_| bad())?;
                if !value.is_finite() {
                    return Err(bad());
                }
                value.to_bits()
            }
        };
        self.store(bits, source);
        Ok(())
    }
}

/// Registration and by-name lookup failures. All of them are startup errors or
/// user typos — never conditions the engine recovers from silently.
#[derive(Debug, thiserror::Error)]
pub enum CVarError {
    #[error(
        "cvar `{name}` is declared twice. Two declarations of one name would leave half the \
         engine reading a value the other half never sees."
    )]
    /// Two declarations claimed one name.
    Duplicate {
        /// The name declared twice.
        name: &'static str,
    },
    #[error("no cvar named `{name}`")]
    /// A config line, flag or console command named a CVar that does not exist.
    Unknown {
        /// The name nothing answers to.
        name: String,
    },
    #[error("cvar `{name}` is a {} and cannot take `{value}`", kind.as_str())]
    /// The CVar exists and the text is not a value of its kind.
    BadValue {
        /// The cvar that refused.
        name: &'static str,
        /// What it does take.
        kind: CVarKind,
        /// What it was offered.
        value: String,
    },
}

/// Registered CVars, sorted by name.
///
/// A `static` holding a lock, not a `static mut`: the §4.2.2 ban is on state
/// smuggled across a reload inside *game* crates, and this is host-owned
/// configuration that a reload does not touch. Sorted rather than hashed keeps
/// `all()` in one order regardless of registration order (§4.2.1 hazard 3) —
/// which matters the moment anything writes a config file back out.
static REGISTRY: RwLock<Vec<&'static CVar>> = RwLock::new(Vec::new());

/// A poisoned registry lock means something panicked mid-lookup. The table is
/// still exactly as consistent as it was — every mutation under the write lock is
/// a single `insert` — so recovering beats taking the process down over a knob.
macro_rules! locked {
    (read) => {
        REGISTRY.read().unwrap_or_else(|e| e.into_inner())
    };
    (write) => {
        REGISTRY.write().unwrap_or_else(|e| e.into_inner())
    };
}

/// Make `cvar` reachable by name. Call once, at startup, from the crate that
/// declares it.
pub fn register(cvar: &'static CVar) -> Result<(), CVarError> {
    let mut registry = locked!(write);
    match registry.binary_search_by_key(&cvar.name, |c| c.name) {
        Ok(_) => Err(CVarError::Duplicate { name: cvar.name }),
        Err(at) => {
            registry.insert(at, cvar);
            Ok(())
        }
    }
}

/// Register a crate's whole set. Stops at the first duplicate — a name collision
/// is a build-time mistake and the rest of the list is not worth guessing about.
pub fn register_all(cvars: &[&'static CVar]) -> Result<(), CVarError> {
    for cvar in cvars {
        register(cvar)?;
    }
    Ok(())
}

/// The registered CVar of that name, if there is one.
pub fn find(name: &str) -> Option<&'static CVar> {
    let registry = locked!(read);
    let at = registry.binary_search_by_key(&name, |c| c.name).ok()?;
    Some(registry[at])
}

/// Set by name, as config, CLI and console all do.
pub fn set(name: &str, value: &str, source: CVarSource) -> Result<(), CVarError> {
    let cvar = find(name).ok_or_else(|| CVarError::Unknown {
        name: name.to_owned(),
    })?;
    cvar.set_from_str(value, source)
}

/// One line per registered CVar, naming the source that won (§4.8) — called
/// once, after every source has had its say.
///
/// Anything still at its declared value goes to `debug` and the rest to `info`:
/// a startup listing where every knob looks alike is a listing nobody reads,
/// and the set that was *changed* is the set a bug report needs.
pub fn log_sources() {
    for cvar in all() {
        let (value, source) = (cvar.to_text(), cvar.source());
        if source == CVarSource::Default {
            tracing::debug!(cvar = cvar.name(), value, source = source.as_str());
        } else {
            tracing::info!(
                cvar = cvar.name(),
                value,
                source = source.as_str(),
                "cvar set"
            );
        }
    }
}

/// Every registered CVar, ascending by name — the console's listing, and the
/// order a written config file would take.
pub fn all() -> Vec<&'static CVar> {
    locked!(read).clone()
}

/// The [`CVar::recorded`] ones, ascending by name (§6 M40). What the baseline
/// gate reads, and the set a [`Watch`] follows.
pub fn recorded() -> Vec<&'static CVar> {
    locked!(read)
        .iter()
        .copied()
        .filter(|c| c.recorded)
        .collect()
}

/// Follows the [`recorded`] set and reports what moved (§6 M40).
///
/// A **diff**, deliberately, rather than a hook in each of the four things that
/// can set a knob: the console bypasses the input path entirely, and a hook per
/// source would cover the ones that exist today. Polling four values a tick
/// catches every source, including the ones that do not exist yet.
///
/// It starts from the *declared defaults*, so the first [`Watch::moved`] reports
/// everything a config file or `--set` had already changed — the opening
/// snapshot is the channel's tick-0 entries and not a second mechanism.
pub struct Watch {
    last: Vec<(&'static CVar, String)>,
}

impl Watch {
    /// Start following, from the declared defaults. Call after registration:
    /// the set is fixed at construction, since a CVar registered later has no
    /// value anything recorded could have read.
    #[must_use]
    pub fn new() -> Self {
        Watch {
            last: recorded()
                .into_iter()
                .map(|c| (c, c.default_text()))
                .collect(),
        }
    }

    /// Whatever moved since the last call, ascending by name — empty on almost
    /// every tick, which is what makes polling the cheap option.
    pub fn moved(&mut self) -> Vec<(&'static str, String)> {
        let mut out = Vec::new();
        for (cvar, last) in &mut self.last {
            let now = cvar.to_text();
            if *last != now {
                out.push((cvar.name(), now.clone()));
                *last = now;
            }
        }
        out
    }
}

impl Default for Watch {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply a replay's knob records (§6 M40) — the values a recording says were in
/// force from this tick on.
///
/// A name this build does not declare is [`CVarError::Unknown`] and the run
/// stops: this is `World::load`'s policy and not `World::restore`'s (§4.5),
/// because a knob that was read while recording and is missing while replaying
/// is a run reproducing something else under the same file name.
///
/// Takes pairs rather than the replay's own record shape: a tick is the
/// caller's concept, and a registry that had learned what one is would be the
/// dependency direction §4.8 homed this module here to avoid.
pub fn apply<'a>(changes: impl IntoIterator<Item = (&'a str, &'a str)>) -> Result<(), CVarError> {
    for (name, value) in changes {
        set(name, value, CVarSource::Replay)?;
        tracing::info!(cvar = name, value, "cvar replayed");
    }
    Ok(())
}

/// How many CVars are registered.
pub fn count() -> usize {
    locked!(read).len()
}
