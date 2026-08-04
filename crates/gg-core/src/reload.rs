//! The reload host (§4.2.2) — load a game dylib, prove it belongs, hand it the
//! host table, and never unload it.
//!
//! Two halves with different lifetimes. The **loader** below is unconditional:
//! dist loads the systems table once at startup and checks it exactly as dev
//! does, because a loader that existed only in dev would make dist the untested
//! path (§1.10). The **watcher** ([`watch`], behind `hot-reload`) is the only
//! part dev has and dist does not, and the dist gate proves `notify` absent from
//! a shipping graph (§5.8).
//!
//! # The order of the checks is the contract
//!
//! [`verify`] compares the host-API version *first* and returns on the spot. The
//! version governs the layout of every other value the dylib produces — its
//! component table, its systems table, the `AbiInfo` fields after the first —
//! so reading any of them before the number agrees is reading a struct under the
//! wrong definition. The fingerprint is checked second and is defense-in-depth:
//! it catches drift the number cannot see, never the other way round.
//!
//! # Nothing is ever unloaded
//!
//! `dlclose`/`FreeLibrary` on a Rust dylib with lingering TLS destructors or a
//! stray function pointer is a classic crash source, so [`GameLib`] deliberately
//! leaks its library on drop (§4.2.2). The leak begins where the library is
//! *created*, not where it is accepted: by the time any check here can run the
//! platform has run the image's initializers and `gg_game_abi()` has executed
//! dylib code, so a **refusal** must leak exactly as an acceptance does — and
//! refusals are the common dev case, one per save after a `HOST_API_VERSION`
//! bump. [`ReloadError::leaked_bytes`] is how a refusal still reaches the
//! budget.
//!
//! That obligation does not stop at [`GameLib::load`]'s last check. Everything
//! the host does between a successful load and the swap — registering schemas,
//! restoring the snapshot through migration, rebinding the action map — can fail
//! on a library that is already mapped and can no longer be unloaded, and a
//! failure raised any other way leaks a whole dylib the budget never sees.
//! [`GameLib::refuse`] is the one route those steps take.
//!
//! That is a *budget*, not a pretense of infinity: [`LeakBudget`] counts the
//! bytes, and crossing it is what triggers rejuvenation — snapshot, restart,
//! restore — rather than an unload nobody can make safe.

use std::path::{Path, PathBuf};

use gg_abi::{
    AbiInfo, ComponentsTable, GameAbiFn, GameComponentsFn, GameInitFn, GameSystemsFn, GameVerbsFn,
    HostApiV1, SYM_GAME_ABI, SYM_GAME_COMPONENTS, SYM_GAME_INIT, SYM_GAME_SYSTEMS, SYM_GAME_VERBS,
    SystemsTable, VerbsTable,
};

pub mod rejuvenate;
#[cfg(feature = "hot-reload")]
pub mod watch;

/// The path [`GameLib::absent`] reports, and what the window is titled before a
/// project is picked. A path-shaped name for the same reason `<statically
/// linked>` is one: `GameLib::path` answers for every variant, and the angle
/// brackets are what say this one names no file.
pub const NO_PROJECT: &str = "<no project>";

/// Leaked dylib bytes a session tolerates before rejuvenation (§4.2.2).
///
/// A dev-profile game dylib is a few megabytes, so this is hundreds of reloads —
/// the ~5 s restart lands once a session rather than once an edit.
pub const DEFAULT_LEAK_BUDGET_BYTES: u64 = 1 << 30;

/// Why a dylib was refused. Every variant names the artifact, because the reader
/// is looking at two builds and needs to know which one is wrong.
#[derive(Debug, thiserror::Error)]
pub enum ReloadError {
    /// The file could not be opened as a dynamic library at all.
    #[error("cannot load `{path}`: {source}")]
    Open {
        /// The artifact.
        path: PathBuf,
        /// What the platform loader said.
        source: libloading::Error,
    },
    /// The artifact's bytes could not be read. Not a paraphrase of [`Open`]: the
    /// image *is* the size charged to the leak budget and the code hash a replay
    /// segment is named by (§4.2.2, §4.7), and neither has a safe default —
    /// zero bytes leaks a multi-megabyte dylib the budget never sees, and a
    /// constant hash makes two builds indistinguishable in a recording.
    ///
    /// [`Open`]: ReloadError::Open
    #[error("cannot read `{path}`: {source}")]
    Unreadable {
        /// The artifact.
        path: PathBuf,
        /// What the filesystem said.
        source: std::io::Error,
    },
    /// A required `extern "C"` symbol is missing — usually a game crate that
    /// never invoked the systems-table macro.
    #[error("`{path}` exports no `{symbol}` — is the systems-table macro missing?")]
    MissingSymbol {
        /// Which of the four symbols (§4.2.2).
        symbol: &'static str,
        /// The artifact.
        path: PathBuf,
        /// Bytes this refusal left resident — see [`leaked_bytes`].
        ///
        /// [`leaked_bytes`]: ReloadError::leaked_bytes
        leaked: u64,
    },
    /// The load-bearing check (§4.2.2), and the only one consulted first.
    #[error(
        "`{path}` was built against host API v{found}, this host is v{expected}. Rebuild the \
         game crate against this engine — nothing else about the dylib was read."
    )]
    HostApiMismatch {
        /// [`gg_abi::HOST_API_VERSION`].
        expected: u32,
        /// What the dylib reported.
        found: u32,
        /// The artifact.
        path: PathBuf,
        /// Bytes this refusal left resident — see [`leaked_bytes`].
        ///
        /// [`leaked_bytes`]: ReloadError::leaked_bytes
        leaked: u64,
    },
    /// Defense-in-depth: same version number, different boundary source.
    #[error(
        "`{path}` carries boundary fingerprint {found}, this host is {expected}. The host API \
         version matched, so a boundary crate changed without the version moving with it."
    )]
    Fingerprint {
        /// This host's, hex.
        expected: &'static str,
        /// The dylib's, hex.
        found: String,
        /// The artifact.
        path: PathBuf,
        /// Bytes this refusal left resident — see [`leaked_bytes`].
        ///
        /// [`leaked_bytes`]: ReloadError::leaked_bytes
        leaked: u64,
    },
    /// The dylib loaded and verified, and the host could not take it into
    /// service: a schema `World::adopt` refuses, a snapshot that will not
    /// migrate, an action map that no longer binds (§4.2.2, §4.7). Not a load
    /// failure, and not free — the image is mapped, its initializers ran, and it
    /// is never unloaded, so its bytes are this session's exactly as a refused
    /// load's are.
    #[error("`{path}` loaded but could not be adopted: {detail}")]
    Adoption {
        /// The artifact.
        path: PathBuf,
        /// Which step gave up, and why.
        detail: String,
        /// Bytes this refusal left resident — see [`leaked_bytes`].
        ///
        /// [`leaked_bytes`]: ReloadError::leaked_bytes
        leaked: u64,
    },
    /// Loading disturbed the FP environment (§4.2.1 hazard 5). Determinism-fatal
    /// and reported rather than asserted, so the host can refuse the swap and say
    /// why instead of dying inside a library initializer.
    #[error("loading `{path}` left {register} out of contract: {detail}")]
    FpEnv {
        /// The artifact.
        path: PathBuf,
        /// `MXCSR`, `FPCR`, whatever this architecture calls it.
        register: &'static str,
        /// Which bits are wrong.
        detail: String,
        /// Bytes this refusal left resident — see [`leaked_bytes`].
        ///
        /// [`leaked_bytes`]: ReloadError::leaked_bytes
        leaked: u64,
    },
    /// Rejuvenation could not hand this session to a successor (§4.2.2).
    #[error("rejuvenation failed while {detail}: {source}")]
    Rejuvenate {
        /// Which step of the handover.
        detail: &'static str,
        /// What the OS said.
        source: std::io::Error,
    },
    /// The staged handoff was not one — a stale file, or another build's.
    #[error("not a rejuvenation handoff: {detail}")]
    Handoff {
        /// What is wrong with it.
        detail: String,
    },
    /// Copying the rebuilt artifact aside failed (§4.2.2 copy-then-load).
    #[error("staging `{path}`: {source}")]
    Staging {
        /// The artifact.
        path: PathBuf,
        /// What the filesystem said.
        source: std::io::Error,
    },
}

impl ReloadError {
    /// Bytes of dylib this refusal left resident, for the caller to charge to a
    /// [`LeakBudget`].
    ///
    /// Nonzero only for the refusals that happen *after* the image was mapped
    /// and its initializers ran: those are never unloaded (§4.2.2), so a session
    /// of refused saves grows exactly as a session of accepted ones does, and a
    /// budget that could not see them would never come due.
    #[must_use]
    pub fn leaked_bytes(&self) -> u64 {
        match self {
            Self::HostApiMismatch { leaked, .. }
            | Self::Fingerprint { leaked, .. }
            | Self::MissingSymbol { leaked, .. }
            | Self::Adoption { leaked, .. }
            | Self::FpEnv { leaked, .. } => *leaked,
            _ => 0,
        }
    }

    /// Attach the leak to a refusal raised by a path that also serves the
    /// nothing-was-loaded cases ([`verify`], [`GameLib::linked`]).
    fn leaking(mut self, bytes: u64) -> Self {
        if let Self::HostApiMismatch { leaked, .. }
        | Self::Fingerprint { leaked, .. }
        | Self::MissingSymbol { leaked, .. }
        | Self::Adoption { leaked, .. }
        | Self::FpEnv { leaked, .. } = &mut self
        {
            *leaked = bytes;
        }
        self
    }
}

/// Compare a dylib's self-description against this host's.
///
/// Version first, and it returns on mismatch without touching the fingerprint —
/// the version is what makes the rest of [`AbiInfo`] interpretable.
///
/// Reports `leaked: 0`; [`GameLib::load`] is what knows whether an image was
/// mapped to reach this, and attaches the bytes.
pub fn verify(info: &AbiInfo, path: &Path) -> Result<(), ReloadError> {
    if info.host_api_version != gg_abi::HOST_API_VERSION {
        return Err(ReloadError::HostApiMismatch {
            expected: gg_abi::HOST_API_VERSION,
            found: info.host_api_version,
            path: path.to_owned(),
            leaked: 0,
        });
    }
    if info.fingerprint != gg_abi::BOUNDARY_FINGERPRINT {
        return Err(ReloadError::Fingerprint {
            expected: gg_abi::BOUNDARY_FINGERPRINT_HEX,
            found: hex(&info.fingerprint),
            path: path.to_owned(),
            leaked: 0,
        });
    }
    Ok(())
}

/// The five `extern "C"` entry points a game presents (§4.2.2), gathered.
///
/// A struct rather than five arguments because the *set* is the boundary: a host
/// that resolved four of them has not loaded a game, and the dormant static-link
/// variant ([`GameLib::linked`]) needs the same five by a route that has no
/// symbol table to resolve them from.
#[derive(Clone, Copy)]
pub struct GameEntryPoints {
    /// Self-description, read before anything else.
    pub abi: GameAbiFn,
    /// Hands the game the host table.
    pub init: GameInitFn,
    /// The component schemas this build declares.
    pub components: GameComponentsFn,
    /// The action and axis names, in id order.
    pub verbs: GameVerbsFn,
    /// The tick's systems, in execution order.
    pub systems: GameSystemsFn,
}

/// A loaded, verified game dylib and the tables it exports.
///
/// **Dropping this does not unload the library** (§4.2.2). The tables below hold
/// pointers into the dylib's static data, and every system pointer the host has
/// ever called lives there too; unloading is the crash, not the cleanup. Charge
/// the bytes to a [`LeakBudget`] instead.
pub struct GameLib {
    // ManuallyDrop *is* the "never unloaded" rule, and `load` applies it at
    // `Library::new` so a refusal cannot unload either. Without it, swapping a
    // dylib would call FreeLibrary on code the old replay segment still names.
    // `None` is the statically-linked variant (§5.9), where the game's code is
    // this binary's and there was never anything to unload.
    _lib: Option<std::mem::ManuallyDrop<libloading::Library>>,
    abi: AbiInfo,
    components: ComponentsTable,
    verbs: VerbsTable,
    systems: SystemsTable,
    path: PathBuf,
    bytes: u64,
    code_hash: u128,
}

impl GameLib {
    /// Load `path`, verify it, and hand it `host_api`.
    ///
    /// The call order is §4.2.2's: `gg_game_abi` → verify → `gg_game_init` →
    /// `gg_game_components` → `gg_game_verbs` → `gg_game_systems`. Nothing after
    /// the first is read until the version agrees.
    ///
    /// # Safety
    ///
    /// `path` must be a game dylib built against this engine — loading executes
    /// its initializers before any check here can run, so the file's provenance
    /// is the caller's to establish. `host_api` must outlive every later call
    /// into the dylib: the dylib keeps the pointer.
    pub unsafe fn load(path: &Path, host_api: &'static HostApiV1) -> Result<Self, ReloadError> {
        // Read rather than stat: the size and the content hash come from the
        // same bytes. A failure is an error and never a default — zero bytes
        // would charge the budget nothing for a dylib it is about to leak, and a
        // constant code hash would name two different builds' replay segments
        // alike (§4.2.2, §4.7).
        let image = std::fs::read(path).map_err(|source| ReloadError::Unreadable {
            path: path.to_owned(),
            source,
        })?;
        let (bytes, code) = (image.len() as u64, code_hash(&image));

        // `ManuallyDrop` here rather than on the success path: everything below
        // runs the image's initializers and then its code, so every refusal from
        // this line on would otherwise unload a library that has already touched
        // `std` and stashed the host pointer (§4.2.2, and the module header).
        //
        // SAFETY: the caller's obligation, documented above — this is the point
        // where the file's initializers run, and no check can precede it.
        let lib = std::mem::ManuallyDrop::new(unsafe { libloading::Library::new(path) }.map_err(
            |source| ReloadError::Open {
                path: path.to_owned(),
                source,
            },
        )?);

        // SAFETY: each symbol is read at the type §4.2.2 defines for it, and the
        // version check below is what makes that definition the right one. The
        // window between here and `verify` is exactly one `extern "C" fn() ->
        // AbiInfo` call, whose signature is pointer-free and layout-stable
        // across every version of the boundary by construction.
        let abi_fn: GameAbiFn = unsafe { symbol(&lib, SYM_GAME_ABI, "gg_game_abi", path, bytes)? };
        // SAFETY: `abi_fn` came from this library and takes no arguments; a
        // dylib exporting the name with a different signature is the failure the
        // fingerprint exists for, and cannot be caught before the call.
        let abi = unsafe { abi_fn() };
        verify(&abi, path).map_err(|refused| refused.leaking(bytes))?;

        // SAFETY: the version agreed, so the remaining symbols mean what this
        // build of `gg-abi` says they mean, at their declared types.
        let entries = unsafe {
            GameEntryPoints {
                abi: abi_fn,
                init: symbol(&lib, SYM_GAME_INIT, "gg_game_init", path, bytes)?,
                components: symbol(&lib, SYM_GAME_COMPONENTS, "gg_game_components", path, bytes)?,
                verbs: symbol(&lib, SYM_GAME_VERBS, "gg_game_verbs", path, bytes)?,
                systems: symbol(&lib, SYM_GAME_SYSTEMS, "gg_game_systems", path, bytes)?,
            }
        };
        // SAFETY: the entry points came from this verified library, and
        // `host_api` is `&'static`.
        unsafe {
            Self::adopt(
                Some(lib),
                &entries,
                abi,
                host_api,
                path.to_owned(),
                bytes,
                code,
            )
        }
    }

    /// The same game, linked into this binary rather than loaded (§5.9).
    ///
    /// The **dormant** half of §2's Game-code boundary row: one interface, two
    /// ways the table arrives. It exists so the fallback stays compiled — a
    /// platform that forbids dylibs is not the moment to find out this path rots
    /// — and it is deliberately no cheaper than [`load`](Self::load): the
    /// version and fingerprint are checked here too, because a statically linked
    /// game can still be a game crate built against a stale boundary in a
    /// workspace with a stale `Cargo.lock`.
    ///
    /// Nothing here leaks, so the artifact size is zero and rejuvenation
    /// (§4.2.2) never comes due — the budget it charges against is dylib bytes,
    /// and there are none.
    ///
    /// # Safety
    ///
    /// The five entry points must be one game crate's, built against this
    /// engine — the same provenance obligation [`load`](Self::load) states,
    /// discharged by the linker instead of by the operator. `host_api` must
    /// outlive every later call into them.
    pub unsafe fn linked(
        entries: &GameEntryPoints,
        host_api: &'static HostApiV1,
    ) -> Result<Self, ReloadError> {
        let path = PathBuf::from("<statically linked>");
        // SAFETY: the caller's obligation — this is `extern "C" fn() -> AbiInfo`
        // with a pointer-free, layout-stable return, exactly as in `load`.
        let abi = unsafe { (entries.abi)() };
        verify(&abi, &path)?;
        // SAFETY: verified, and the caller vouched for the entry points.
        // Code hash zero: there is no artifact to hash, and the binary's own
        // identity is the process's — the value the replay format already
        // documents for "no source for this yet" (§4.7).
        unsafe { Self::adopt(None, entries, abi, host_api, path, 0, 0) }
    }

    /// No game at all — what a host holds before a project has been picked (§6
    /// M15.1 item 4).
    ///
    /// The third way the five tables arrive, and the only one where they are
    /// empty: nothing was mapped, nothing declares a component, a verb or a
    /// system, and the host's own §4.5 protocol types are the whole registry. A
    /// variant here rather than an `Option<GameLib>` in the shell because every
    /// property a host reads off a game has a true answer with no game in it, and
    /// §3 caps the shell at wiring — a launcher's empty world is the loader's
    /// concept, not a branch in every method that touches one.
    ///
    /// The ABI it reports is this build's own. It is not a claim that some game
    /// agreed with the boundary; it is the absence of a game to disagree, and
    /// [`verify`] has nothing to do because nothing was loaded.
    #[must_use]
    pub fn absent() -> Self {
        Self {
            _lib: None,
            abi: AbiInfo {
                host_api_version: gg_abi::HOST_API_VERSION,
                fingerprint: gg_abi::BOUNDARY_FINGERPRINT,
            },
            // Null and zero-length rather than a static empty slice: `len` is what
            // every reader walks, and a null it never dereferences is the honest
            // spelling of "there are none".
            components: ComponentsTable {
                entries: std::ptr::null(),
                len: 0,
                reserved: 0,
            },
            verbs: VerbsTable {
                actions: std::ptr::null(),
                axes: std::ptr::null(),
                action_len: 0,
                axis_len: 0,
            },
            systems: SystemsTable {
                entries: std::ptr::null(),
                len: 0,
                reserved: 0,
            },
            // Named rather than empty: it is what the window is titled and what a
            // layout file is keyed by, and an empty stem would make both blank.
            path: PathBuf::from(NO_PROJECT),
            // Nothing mapped, so nothing to retire and nothing to hash — the same
            // two zeros [`linked`](Self::linked) reports, for the same reason.
            bytes: 0,
            code_hash: 0,
        }
    }

    /// Whether this is [`absent`](Self::absent) — no game, rather than a game
    /// that declares nothing. A dylib *may* declare no systems; a host asking
    /// "is there a project" must not have to guess from an empty table.
    #[must_use]
    pub fn is_absent(&self) -> bool {
        self.path == Path::new(NO_PROJECT)
    }

    /// Hand over the host table and read the three tables — the half of the load
    /// sequence that is the same whether the symbols came from a library or from
    /// the linker.
    ///
    /// # Safety
    ///
    /// `abi` must already have passed [`verify`], and `entries` must be one game
    /// build's entry points. `host_api` must outlive every later call.
    unsafe fn adopt(
        lib: Option<std::mem::ManuallyDrop<libloading::Library>>,
        entries: &GameEntryPoints,
        abi: AbiInfo,
        host_api: &'static HostApiV1,
        path: PathBuf,
        bytes: u64,
        code_hash: u128,
    ) -> Result<Self, ReloadError> {
        // SAFETY: `host_api` is `&'static`, satisfying the pointer's documented
        // lifetime requirement, and the caller has verified the game.
        unsafe { (entries.init)(host_api) };
        // SAFETY: init has run, so the tables are assembled and the pointers in
        // them are the game's statics — `&'static` by never being unloaded.
        let (components, verbs, systems) = unsafe {
            (
                (entries.components)(),
                (entries.verbs)(),
                (entries.systems)(),
            )
        };

        check_fp_env(&path, bytes)?;

        Ok(Self {
            _lib: lib,
            abi,
            components,
            verbs,
            systems,
            path,
            bytes,
            code_hash,
        })
    }

    /// What the dylib said about itself, after it was believed.
    pub fn abi(&self) -> &AbiInfo {
        &self.abi
    }

    /// The component schemas this build declares — what drives migration (§4.8).
    pub fn components(&self) -> &ComponentsTable {
        &self.components
    }

    /// The action and axis names this build declares (§4.7) — what the host
    /// parses an action map against and checks a replay's id space by name.
    pub fn verbs(&self) -> &VerbsTable {
        &self.verbs
    }

    /// The tick's systems, in execution order (§4.1).
    pub fn systems(&self) -> &SystemsTable {
        &self.systems
    }

    /// The artifact this was loaded from — the staged copy, not the source.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// What to call this game: the artifact's file stem, which is the only name
    /// a host has for it — a dylib declares components and systems and never a
    /// title. Lossy on a non-UTF-8 path rather than absent, the caller being a
    /// window caption or a file beside it (§6 M15.1).
    pub fn name(&self) -> std::borrow::Cow<'_, str> {
        self.path.file_stem().unwrap_or_default().to_string_lossy()
    }

    /// Size of the artifact, which is what retiring it costs (§4.2.2).
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// This build's identity, for the replay segment that names it (§4.7).
    ///
    /// The artifact's *content*, not its component or system tables: an edit
    /// that changes a constant inside a system body changes behaviour while
    /// leaving every declared id and schema hash exactly where it was, and a
    /// segment that could not tell those two builds apart would be answering a
    /// different question than "which code produced these ticks".
    ///
    /// Zero for a statically linked game, which has no artifact of its own.
    pub fn code_hash(&self) -> u128 {
        self.code_hash
    }

    /// Give up on this library *after* it loaded, charging its bytes.
    ///
    /// The step that fails here is the host's, not the loader's — `World::adopt`
    /// on a schema it will not take, `restore` on a snapshot it cannot migrate,
    /// the rebind against a moved verb id space — and by then the image is
    /// mapped, its initializers have run and `ManuallyDrop` has already decided
    /// it is never coming out (§4.2.2). Dropping the `GameLib` on that path
    /// unloads nothing, so a failure raised any other way is a whole dylib the
    /// [`LeakBudget`] never sees and a rejuvenation that never comes due.
    ///
    /// Zero for the statically linked game, which mapped nothing.
    #[must_use]
    pub fn refuse(&self, detail: impl std::fmt::Display) -> ReloadError {
        ReloadError::Adoption {
            path: self.path.clone(),
            detail: detail.to_string(),
            leaked: self.bytes,
        }
    }
}

impl std::fmt::Debug for GameLib {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GameLib")
            .field("path", &self.path)
            .field("bytes", &self.bytes)
            .field("systems", &self.systems.len)
            .finish()
    }
}

/// Bytes of dylib the session has leaked, against what it will tolerate.
///
/// A count would be the wrong unit (§4.2.2): what matters is resident memory,
/// and a debug dylib and a dist one differ by an order of magnitude.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeakBudget {
    spent: u64,
    budget: u64,
}

impl Default for LeakBudget {
    fn default() -> Self {
        Self::new(DEFAULT_LEAK_BUDGET_BYTES)
    }
}

impl LeakBudget {
    /// A budget of `budget` bytes, nothing spent.
    pub const fn new(budget: u64) -> Self {
        Self { spent: 0, budget }
    }

    /// Retire a dylib's bytes into the budget.
    pub fn charge(&mut self, bytes: u64) {
        self.spent = self.spent.saturating_add(bytes);
    }

    /// Bytes leaked so far.
    pub fn spent(&self) -> u64 {
        self.spent
    }

    /// Bytes the session tolerates.
    pub fn budget(&self) -> u64 {
        self.budget
    }

    /// Whether rejuvenation is due — snapshot, restart the host, restore (§4.2.2).
    ///
    /// A budget of zero is exhausted by the first reload, which is how the
    /// rejuvenation path gets exercised on demand instead of after a thousand
    /// edits.
    pub fn exhausted(&self) -> bool {
        self.spent >= self.budget
    }
}

/// Resolve one symbol, or say which one was missing.
///
/// # Safety
///
/// `T` must be the type §4.2.2 declares for `name`. Every caller is in this
/// module and passes the `gg_abi` type alias for the symbol it names.
unsafe fn symbol<T: Copy>(
    lib: &libloading::Library,
    name: &[u8],
    label: &'static str,
    path: &Path,
    leaked: u64,
) -> Result<T, ReloadError> {
    // SAFETY: the caller's obligation above. `libloading` copies the symbol out
    // by value, so nothing here outlives the borrow of `lib`.
    let found = unsafe { lib.get::<T>(name) }.map_err(|_| ReloadError::MissingSymbol {
        symbol: label,
        path: path.to_owned(),
        leaked,
    })?;
    Ok(*found)
}

/// Hazard 5 (§4.2.1) at its mandated call site: a library initializer is one of
/// the few things in a process that plausibly changes `MXCSR`. `leaked` is zero
/// for the statically linked game, which mapped nothing.
fn check_fp_env(path: &Path, leaked: u64) -> Result<(), ReloadError> {
    #[cfg(feature = "fp-assert")]
    if let Err(env) = gg_math::fpenv::check_fp_env() {
        let mut detail = Vec::new();
        if !env.round_to_nearest_even {
            detail.push("rounding is not round-to-nearest-even");
        }
        if env.flush_to_zero {
            detail.push("flush-to-zero is set");
        }
        if env.denormals_are_zero {
            detail.push("denormals-are-zero is set");
        }
        return Err(ReloadError::FpEnv {
            path: path.to_owned(),
            register: gg_math::fpenv::FP_CONTROL_REGISTER,
            detail: detail.join(", "),
            leaked,
        });
    }
    let _ = (path, leaked);
    Ok(())
}

/// A loaded artifact's identity: `blake3(domain ‖ image)`, first 16 bytes
/// little-endian. Domain-separated like every other hash in the engine (§4.2.1),
/// so a code hash and a schema hash over the same bytes can never collide.
///
/// Not a determinism input — it labels a replay segment, and two machines
/// building the same source need not produce the same artifact. What it has to
/// do is *change when the code changes*, which content hashing gives for free.
fn code_hash(image: &[u8]) -> u128 {
    let mut h = blake3::Hasher::new();
    h.update(b"gg-core/game-code/v1\0");
    h.update(image);
    let mut out = [0u8; 16];
    out.copy_from_slice(&h.finalize().as_bytes()[..16]);
    u128::from_le_bytes(out)
}

/// Lowercase hex, so a fingerprint in an error message can be pasted into a grep.
fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `GameLib` as the host holds one *after* a successful load: mapped,
    /// verified, and past the point where anything could be unloaded. Built by
    /// hand because reaching that state for real needs a game dylib, which is
    /// demo 03's and `xtask reload`'s job (`tests/reload.rs` says the same).
    fn mapped(bytes: u64) -> GameLib {
        GameLib {
            _lib: None,
            abi: AbiInfo {
                host_api_version: gg_abi::HOST_API_VERSION,
                fingerprint: gg_abi::BOUNDARY_FINGERPRINT,
            },
            components: ComponentsTable {
                entries: std::ptr::null(),
                len: 0,
                reserved: 0,
            },
            verbs: VerbsTable {
                actions: std::ptr::null(),
                axes: std::ptr::null(),
                action_len: 0,
                axis_len: 0,
            },
            systems: SystemsTable {
                entries: std::ptr::null(),
                len: 0,
                reserved: 0,
            },
            path: PathBuf::from("game.dll"),
            bytes,
            code_hash: 0,
        }
    }

    #[test]
    fn a_failure_after_the_load_still_charges_the_budget() {
        // The hole this closes: `load` returned, so every refusal it knows how to
        // charge is already behind us — and the host's own steps (adopt, restore,
        // rebind) then drop a `ManuallyDrop`ped library that unloads nothing.
        let lib = mapped(4_000);
        let refused = lib.refuse("restore: `Health` changed layout and cannot migrate");
        assert_eq!(refused.leaked_bytes(), lib.bytes());
        assert!(refused.to_string().contains("game.dll"), "{refused}");

        let mut budget = LeakBudget::new(4_000);
        budget.charge(refused.leaked_bytes());
        assert!(
            budget.exhausted(),
            "a dylib refused after loading is still a leaked dylib"
        );
    }

    #[test]
    fn no_project_declares_nothing_and_says_which_kind_of_nothing_it_is() {
        // The distinction the launcher rests on (§6 M15.1 item 4): a game that
        // declares no systems and *no game at all* have identical tables, so a
        // host reading emptiness cannot tell them apart and must be told.
        let none = GameLib::absent();
        assert!(none.is_absent());
        assert!(
            !mapped(4_000).is_absent(),
            "a real dylib is not the absence"
        );
        assert_eq!(none.components().len, 0);
        assert_eq!(none.verbs().action_len, 0);
        assert_eq!(none.verbs().axis_len, 0);
        assert_eq!(none.systems().len, 0);
        // Nothing mapped: nothing to retire, and no artifact to name a replay
        // segment by — the two zeros `linked` reports for the same reason.
        assert_eq!(none.bytes(), 0);
        assert_eq!(none.code_hash(), 0);
        assert_eq!(none.refuse("nothing to refuse").leaked_bytes(), 0);
        // The window's caption before a project is picked, so it is a name and
        // not a blank strip.
        assert_eq!(none.name(), NO_PROJECT);
        // It passes the boundary check it was never subject to: nothing loaded,
        // so there is nothing that could disagree with this build.
        assert!(verify(none.abi(), Path::new(NO_PROJECT)).is_ok());
    }

    #[test]
    fn the_statically_linked_game_leaks_nothing_to_charge() {
        // §5.9's variant mapped no image, so its adoption failures are free —
        // the budget is dylib bytes and there are none.
        assert_eq!(mapped(0).refuse("no verbs").leaked_bytes(), 0);
    }

    #[test]
    fn the_leak_survives_the_paths_that_re_raise_a_refusal() {
        // `leaking` is how a refusal raised where nothing was mapped (`verify`,
        // `linked`) picks up the bytes when a mapped caller re-raises it. A
        // variant it forgets is a variant that reports zero forever.
        let charged = mapped(0).refuse("adopt: unknown component").leaking(1_024);
        assert_eq!(charged.leaked_bytes(), 1_024);
    }
}
