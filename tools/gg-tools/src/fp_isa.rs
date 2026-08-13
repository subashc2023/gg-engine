//! `fp-isa` — which floating-point instructions the determinism path actually
//! contains, per target (§8's qemu row, §4.2.1's contract).
//!
//! # Why an instrument and not a paragraph
//!
//! §8 carries a row reading "qemu-tested ARM is not ARM silicon", rated
//! Low–Medium, whose mitigation is a cadence and whose upgrade path is a board
//! on the LAN. What the row never says is *how large* the residual is, and that
//! is answerable rather than assumable — because of one fact about qemu-user
//! that the row's wording obscures: **it executes the very bytes real silicon
//! would**. The aarch64 artifact is cross-compiled once; qemu interprets that
//! binary; a Cortex core would fetch the same instructions from the same file.
//! So everything upstream of execution — rustc's codegen, LLVM's instruction
//! selection and vectorisation, `libm`'s source, struct layout, the hash
//! protocol — is not *partially* covered by the qemu leg, it is covered
//! completely. Whatever is left is exactly one question: does an ARM CPU
//! execute *these particular instructions* the way the architecture says?
//!
//! That question has a shape, and the shape is a list. IEEE 754 requires
//! `+ - * /`, `sqrt` and `fma` to be correctly rounded — one representable
//! answer, no implementation freedom — so a divergence on those needs a
//! hardware erratum, not a design difference. The instructions with genuine
//! per-implementation freedom are the *estimate* family (`frecpe`, `frsqrte`
//! and friends), which trade exactness for a reciprocal in one cycle and which
//! a compiler emits only when told it may. Rust never tells it that
//! (RFC 3514: no contraction, no reassociation, no fast-math), so the
//! prediction is that the estimate class is empty — and a prediction that is
//! never checked is the kind of claim §5 exists to distrust.
//!
//! This prints the list. It does not gate: a threshold moves to `xtask` (see
//! the crate docs), and whether an empty estimate class is worth a gate — or
//! whether the row wants silicon regardless — is the desk's call, not this
//! file's.
//!
//! # What a green report does and does not buy
//!
//! Does: it bounds the residual to hardware erratum in a correctly-rounded
//! operation, plus a non-default `FPCR` — the latter already asserted by
//! §4.2.1 hazard 5's `fp-assert`. Does not: prove any of that on silicon. Only
//! silicon proves silicon. The honest sentence this instrument licenses is
//! "the residual is a CPU bug in `fadd`", which is a much smaller thing to
//! carry than "ARM might differ", and is what the §8 row should say if it is
//! going to stay open.
//!
//! Usage:
//!   gg-tools fp-isa [--target <triple>] [--profile dev|dist] [--dir <path>]
//!                   [--objdump <path>]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// How much freedom the architecture leaves an implementation, which is the
/// only axis that matters to a cross-architecture bit-exactness claim.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Class {
    /// No rounding happens at all — sign, move, select, compare. Bit-exact by
    /// construction on any machine that agrees on the format.
    Exact,
    /// One representable answer, mandated. IEEE 754 correctly-rounded
    /// arithmetic and the conversions/roundings defined in the same terms.
    Rounded,
    /// **The class this instrument exists to count.** Architecturally
    /// described but not correctly rounded — reciprocal and rsqrt estimates,
    /// and the Newton step instructions that pair with them. A compiler emits
    /// these only under fast-math, so one appearing here is either a
    /// dependency built with different flags or an intrinsic called by hand,
    /// and either way it is a determinism hazard wearing a performance
    /// costume.
    Estimate,
}

/// aarch64. The rule that keeps this honest is in [`classify`]: every mnemonic
/// the architecture spells with a leading `f` must appear in one of these
/// tables or be reported unclassified, so the table failing to keep up with
/// the ISA is a loud event rather than a silent pass.
const A64_EXACT: &[&str] = &[
    "fmov", "fabs", "fneg", "fcsel", "fcmp", "fcmpe", "fccmp", "fccmpe", "fmax", "fmin", "fmaxnm",
    "fminnm", "fmaxnmp", "fminnmp", "fmaxnmv", "fminnmv", "fmaxp", "fminp", "fmaxv", "fminv",
    "fcmeq", "fcmge", "fcmgt", "fcmle", "fcmlt", "facge", "facgt",
];
const A64_ROUNDED: &[&str] = &[
    "fadd", "fsub", "fmul", "fdiv", "fnmul", "fsqrt", "fmadd", "fmsub", "fnmadd", "fnmsub", "fmla",
    "fmls", "fabd", "faddp", "faddv", "fmulx", "fcvt", "fcvtl", "fcvtl2", "fcvtn", "fcvtn2",
    "fcvtxn", "fcvtxn2", "fcvtas", "fcvtau", "fcvtms", "fcvtmu", "fcvtns", "fcvtnu", "fcvtps",
    "fcvtpu", "fcvtzs", "fcvtzu", "frinta", "frinti", "frintm", "frintn", "frintp", "frintx",
    "frintz", "scvtf", "ucvtf",
];
const A64_ESTIMATE: &[&str] = &["frecpe", "frecps", "frecpx", "frsqrte", "frsqrts"];

/// x86-64, carried because the contract is a *cross-architecture* claim and
/// "both sides use only mandated arithmetic" is the half of it this instrument
/// can show. Scalar and packed SSE/AVX; the bitwise lanes (`andps`, `xorps`)
/// are how a compiler spells `fabs`/`fneg`, hence [`Class::Exact`].
const X86_EXACT: &[&str] = &[
    "movss", "movsd", "movaps", "movapd", "movups", "movupd", "andps", "andpd", "andnps", "andnpd",
    "orps", "orpd", "xorps", "xorpd", "ucomiss", "ucomisd", "comiss", "comisd", "cmpss", "cmpsd",
    "cmpps", "cmppd", "maxss", "maxsd", "maxps", "maxpd", "minss", "minsd", "minps", "minpd",
    "blendvps", "blendvpd",
];
const X86_ROUNDED: &[&str] = &[
    "addss",
    "addsd",
    "addps",
    "addpd",
    "subss",
    "subsd",
    "subps",
    "subpd",
    "mulss",
    "mulsd",
    "mulps",
    "mulpd",
    "divss",
    "divsd",
    "divps",
    "divpd",
    "sqrtss",
    "sqrtsd",
    "sqrtps",
    "sqrtpd",
    "addsubps",
    "addsubpd",
    "haddps",
    "haddpd",
    "hsubps",
    "hsubpd",
    "cvtss2sd",
    "cvtsd2ss",
    "cvtsi2ss",
    "cvtsi2sd",
    "cvtss2si",
    "cvtsd2si",
    "cvttss2si",
    "cvttsd2si",
    "cvtps2pd",
    "cvtpd2ps",
    "cvtdq2ps",
    "cvtdq2pd",
    "cvtps2dq",
    "cvtpd2dq",
    "cvttps2dq",
    "cvttpd2dq",
    "roundss",
    "roundsd",
    "roundps",
    "roundpd",
];
/// `rcp`/`rsqrt` are the x86 spelling of the same hazard, and are *worse* than
/// their ARM counterparts: the SSE forms are specified only to a relative error
/// bound, so two x86 vendors may legitimately differ. The `vfmadd*` families
/// are correctly rounded and live in [`X86_ROUNDED`] via the prefix rule.
const X86_ESTIMATE: &[&str] = &[
    "rcpss",
    "rcpps",
    "rsqrtss",
    "rsqrtps",
    "rcp14ss",
    "rcp14sd",
    "rcp14ps",
    "rcp14pd",
    "rsqrt14ss",
    "rsqrt14sd",
    "rsqrt14ps",
    "rsqrt14pd",
    "rcp28ps",
    "rsqrt28ps",
];

/// Whether a mnemonic is floating-point at all on this architecture — the
/// gate on "must be classified". On aarch64 the architecture's own naming does
/// the work (`f…`, plus the two integer→float converts); on x86 there is no
/// such rule, so the FP set is enumerated and anything outside it is genuinely
/// not FP rather than merely unrecognised.
fn is_float(arch: Arch, m: &str) -> bool {
    match arch {
        Arch::A64 => m.starts_with('f') || m == "scvtf" || m == "ucvtf",
        Arch::X86 => {
            let m = m.strip_prefix('v').unwrap_or(m); // AVX encodings of the same op
            X86_EXACT.contains(&m)
                || X86_ROUNDED.contains(&m)
                || X86_ESTIMATE.contains(&m)
                || m.starts_with("vfmadd")
                || m.starts_with("vfmsub")
                || m.starts_with("vfnmadd")
                || m.starts_with("vfnmsub")
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Arch {
    A64,
    X86,
}

/// `Some(class)` for a floating-point mnemonic, `None` for one this table does
/// not know — which the report prints rather than swallows, because an
/// unclassified FP instruction is precisely the finding this exists to make.
/// Non-FP mnemonics never reach here (see [`is_float`]).
pub fn classify(arch: Arch, mnemonic: &str) -> Option<Class> {
    // Vector forms carry a lane suffix in some disassembler dialects; the
    // operand text holds the arrangement, so the mnemonic is already bare.
    let m = mnemonic;
    let (exact, rounded, estimate) = match arch {
        Arch::A64 => (A64_EXACT, A64_ROUNDED, A64_ESTIMATE),
        Arch::X86 => (X86_EXACT, X86_ROUNDED, X86_ESTIMATE),
    };
    let bare = match arch {
        Arch::X86 => m.strip_prefix('v').unwrap_or(m),
        Arch::A64 => m,
    };
    if arch == Arch::X86
        && (bare.starts_with("fmadd")
            || bare.starts_with("fmsub")
            || bare.starts_with("fnmadd")
            || bare.starts_with("fnmsub"))
    {
        return Some(Class::Rounded); // vfmadd132ss and the other 17 spellings
    }
    if estimate.contains(&bare) {
        return Some(Class::Estimate);
    }
    if rounded.contains(&bare) {
        return Some(Class::Rounded);
    }
    if exact.contains(&bare) {
        return Some(Class::Exact);
    }
    None
}

/// One instruction's worth of finding, kept only for the classes worth naming
/// a site for — an `fadd` needs no address, an `frsqrte` needs all of them.
#[derive(Debug)]
pub struct Site {
    pub symbol: String,
    pub mnemonic: String,
}

#[derive(Debug, Default)]
pub struct Tally {
    pub counts: BTreeMap<String, (Class, u64)>,
    pub estimates: Vec<Site>,
    pub unclassified: Vec<Site>,
    /// Math routines resolved by the loader rather than compiled in — see
    /// [`IMPORTED_MATH`]. `mnemonic` carries the symbol, `symbol` the file.
    pub imported: Vec<Site>,
    pub instructions: u64,
}

/// Parse `objdump -d --no-show-raw-insn` output, in either the LLVM or GNU
/// dialect — they agree on the two lines that matter here: a symbol header
/// `<addr> <name>:` and a body line `<addr>: <mnemonic> <operands>`.
pub fn tally(arch: Arch, disassembly: &str) -> Tally {
    let mut out = Tally::default();
    let mut symbol = String::from("<no symbol>");
    for line in disassembly.lines() {
        let trimmed = line.trim();
        // Symbol header: ends in ':' and carries a <name>. Checked before the
        // body case because both contain a colon.
        if let (Some(open), true) = (trimmed.find('<'), trimmed.ends_with(">:")) {
            symbol = trimmed[open + 1..trimmed.len() - 2].to_string();
            continue;
        }
        let Some((addr, rest)) = trimmed.split_once(':') else {
            continue;
        };
        if addr.is_empty() || !addr.chars().all(|c| c.is_ascii_hexdigit()) {
            continue;
        }
        let Some(mnemonic) = rest.split_whitespace().next() else {
            continue;
        };
        let mnemonic = mnemonic.to_ascii_lowercase();
        out.instructions += 1;
        if !is_float(arch, &mnemonic) {
            continue;
        }
        let site = || Site {
            symbol: symbol.clone(),
            mnemonic: mnemonic.clone(),
        };
        match classify(arch, &mnemonic) {
            Some(Class::Estimate) => out.estimates.push(site()),
            None => out.unclassified.push(site()),
            Some(_) => {}
        }
        let class = classify(arch, &mnemonic).unwrap_or(Class::Estimate);
        let entry = out.counts.entry(mnemonic).or_insert((class, 0));
        entry.1 += 1;
    }
    out
}

/// Math routines that, if *imported* rather than compiled in, are computed by
/// code this instrument never sees — and by a copy of that code chosen at
/// runtime by the loader. That is the hole a disassembly scan has by
/// construction, and it is not hypothetical: the system libm is exactly what
/// §4.2.1 hazard 1 bans transcendentals to avoid, since glibc's `sin` is not
/// correctly rounded and differs by version. A clean instruction report over a
/// binary that calls `pow@GLIBC` is a false negative, so the two checks ship
/// together or the verdict is worth nothing.
const IMPORTED_MATH: &[&str] = &[
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",
    "cosh",
    "tanh",
    "exp",
    "exp2",
    "exp10",
    "expm1",
    "log",
    "log2",
    "log10",
    "log1p",
    "pow",
    "cbrt",
    "hypot",
    "fmod",
    "remainder",
    "lgamma",
    "tgamma",
    "erf",
    "erfc",
    "sincos",
    "ldexp",
    "frexp",
];

/// Undefined dynamic symbols that name a math routine — see [`IMPORTED_MATH`].
/// Both `f` (single) and bare (double) spellings, and the `l` long-double forms
/// which would be a different hazard again.
fn imported_math(dynamic: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in dynamic.lines() {
        // objdump -T marks undefined imports with the `*UND*` section.
        if !line.contains("*UND*") {
            continue;
        }
        let Some(name) = line.split_whitespace().last() else {
            continue;
        };
        let bare = name.split('@').next().unwrap_or(name);
        let stem = bare
            .strip_suffix('f')
            .or_else(|| bare.strip_suffix('l'))
            .unwrap_or(bare);
        if IMPORTED_MATH.contains(&bare) || IMPORTED_MATH.contains(&stem) {
            out.push(name.to_string());
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Find a disassembler that can read `target`.
///
/// The ordering is the whole content of this function. A plain `objdump` is
/// almost always present and almost always knows *one* architecture — the
/// host's — so preferring it would silently fail on the only target this
/// instrument exists for, and the cross binutils named for the triple goes
/// first when the triple is not ours. `rust-objdump` (the `llvm-tools`
/// component) is next because it knows every target the toolchain does, which
/// is how this works on a machine with no cross binutils at all.
fn disassembler(explicit: Option<&str>, target: &str) -> anyhow::Result<PathBuf> {
    let cross = match target.split('-').next() {
        // `aarch64-unknown-linux-gnu` → Debian's `aarch64-linux-gnu-objdump`.
        Some(arch) if !target.contains(std::env::consts::ARCH) => {
            Some(format!("{arch}-linux-gnu-objdump"))
        }
        _ => None,
    };
    let candidates: Vec<String> = explicit.map_or_else(
        || {
            cross
                .into_iter()
                .chain(
                    ["rust-objdump", "llvm-objdump", "objdump"]
                        .iter()
                        .map(|s| (*s).to_string()),
                )
                .collect()
        },
        |e| vec![e.to_string()],
    );
    for c in &candidates {
        if Command::new(c).arg("--version").output().is_ok() {
            return Ok(PathBuf::from(c));
        }
    }
    anyhow::bail!(
        "no disassembler found (tried {}). `rustup component add llvm-tools` provides \
         rust-objdump, which cross-disassembles every target the toolchain supports",
        candidates.join(", ")
    )
}

/// Whether `binary` was built from the sources it currently depends on, read
/// out of the `.d` cargo writes beside it.
///
/// Per-artifact and not workspace-wide, which matters: the first version of
/// this compared every binary against the newest source *anywhere*, and since
/// running an instrument normally follows editing something, that condemned the
/// whole directory. The depfile is the precise question — did any file **this
/// binary reads** change after it was linked — and it is cargo's own answer, so
/// the instrument and the build agree by construction.
///
/// `true` when it cannot tell (no depfile, unreadable mtimes): an instrument
/// that hid artifacts on a guess would be the failure it is here to prevent.
///
/// **Cargo writes those dependencies workspace-relative**, so they resolve
/// against `root` rather than against the process's directory. Found while
/// hardening the same predicate into `xtask`'s import gate, and it is the one
/// mistake this function cannot afford: an unresolvable path drops out of the
/// iterator, an empty iterator satisfies `all`, and the filter then answers
/// "current" for the whole attic — failing open, in exactly the direction the
/// six-day-stale `sincosf` report went.
fn built_from_current_sources(binary: &Path, root: &Path) -> bool {
    let Ok(built) = binary.metadata().and_then(|m| m.modified()) else {
        return true;
    };
    let Ok(dep) = std::fs::read_to_string(binary.with_extension("d")) else {
        return true;
    };
    // `<target>: <dep> <dep> …`, one rule per line. The separator is a colon
    // *followed by a space* — a bare colon would split `C:\dev\…` in half on
    // Windows, where the drive letter's colon is followed by a separator.
    dep.lines()
        .filter_map(|line| line.split_once(": "))
        .flat_map(|(_, deps)| deps.split_whitespace())
        .filter_map(|d| {
            std::fs::metadata(root.join(d))
                .and_then(|m| m.modified())
                .ok()
        })
        .all(|source| source <= built)
}

/// ELF/PE test binaries under a profile's `deps/`, minus the artefacts that are
/// not executables. Cargo leaves both the hashed binary and its `.d` beside
/// each other; only the former is worth disassembling.
///
/// **Split by age, and that is not fastidiousness.** `target/` is an attic: a
/// binary whose crate hash nothing rebuilds sits there until `cargo clean`, so
/// a census that reads the directory reads whatever anyone ever built. This
/// instrument's whole output is a claim about *the code in the tree*, and it
/// made the mistake once — the first aarch64 report (§6 M17 item 6) named a
/// `sincosf` import in a `gg-extract` artifact six days stale, and it stayed in
/// the report after the import had been designed out of the tree. Returns
/// `(current, stale)`; the caller prints the second rather than dropping it,
/// because a silent filter is the same failure one layer down.
fn artifacts(dir: &Path, root: &Path) -> anyhow::Result<(Vec<PathBuf>, Vec<PathBuf>)> {
    let (mut current, mut stale) = (Vec::new(), Vec::new());
    for entry in std::fs::read_dir(dir)
        .map_err(|e| anyhow::anyhow!("{}: {e} — build the target first", dir.display()))?
    {
        let path = entry?.path();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !path.is_file() || !matches!(ext, "" | "exe") {
            continue;
        }
        match built_from_current_sources(&path, root) {
            true => current.push(path),
            false => stale.push(path),
        }
    }
    current.sort();
    stale.sort();
    Ok((current, stale))
}

pub fn run(args: &[String]) -> anyhow::Result<()> {
    let arg = |name: &str, default: &str| -> String {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
            .unwrap_or_else(|| default.to_string())
    };
    let target = arg("--target", "aarch64-unknown-linux-gnu");
    let profile = arg("--profile", "dev");
    let arch = if target.starts_with("aarch64") {
        Arch::A64
    } else if target.starts_with("x86_64") {
        Arch::X86
    } else {
        anyhow::bail!("no instruction table for {target} — add one beside A64_/X86_ above")
    };
    // Cargo's directory name for the dev profile is `debug`; every other
    // profile names itself.
    let profile_dir = if profile == "dev" { "debug" } else { &profile };
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .ok_or_else(|| anyhow::anyhow!("no workspace root"))?
        .to_path_buf();
    let default_dir = root.join(format!("target/{target}/{profile_dir}/deps"));
    let dir = PathBuf::from(arg("--dir", &default_dir.to_string_lossy()));
    let objdump = disassembler(
        args.iter()
            .position(|a| a == "--objdump")
            .and_then(|i| args.get(i + 1).map(String::as_str)),
        &target,
    )?;

    println!(
        "gg-tools fp-isa: {target} ({profile}) via {}",
        objdump.display()
    );
    println!("gg-tools fp-isa: reading {}", dir.display());

    let (current, stale) = artifacts(&dir, &root)?;
    if !stale.is_empty() {
        println!(
            "gg-tools fp-isa: skipping {} artifact(s) whose sources changed after they were \
             linked — they are a census of a program this tree no longer describes:",
            stale.len()
        );
        for path in &stale {
            println!(
                "  {}",
                path.file_name().unwrap_or_default().to_string_lossy()
            );
        }
    }

    let mut total = Tally::default();
    let mut scanned = 0usize;
    for path in current {
        let out = Command::new(&objdump)
            .args(["-d", "--no-show-raw-insn"])
            .arg(&path)
            .output()?;
        if !out.status.success() {
            continue; // not an object file — a fingerprint, a lockfile, a script
        }
        let text = String::from_utf8_lossy(&out.stdout);
        if text.is_empty() {
            continue;
        }
        scanned += 1;
        // The other half of the answer: what this binary computes *elsewhere*.
        let dynamic = Command::new(&objdump).arg("-T").arg(&path).output()?;
        for sym in imported_math(&String::from_utf8_lossy(&dynamic.stdout)) {
            total.imported.push(Site {
                symbol: path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned(),
                mnemonic: sym,
            });
        }
        let t = tally(arch, &text);
        total.instructions += t.instructions;
        for (m, (class, n)) in t.counts {
            let entry = total.counts.entry(m).or_insert((class, 0));
            entry.1 += n;
        }
        for site in t.estimates {
            total.estimates.push(Site {
                symbol: format!(
                    "{}: {}",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    site.symbol
                ),
                mnemonic: site.mnemonic,
            });
        }
        for site in t.unclassified {
            total.unclassified.push(Site {
                symbol: format!(
                    "{}: {}",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    site.symbol
                ),
                mnemonic: site.mnemonic,
            });
        }
    }
    if scanned == 0 {
        anyhow::bail!(
            "nothing disassembled under {} — build the leg first, e.g. `cargo nextest run -p \
             gg-math --target {target} --no-run`",
            dir.display()
        );
    }

    report(&total, scanned);
    Ok(())
}

fn report(t: &Tally, binaries: usize) {
    let sum = |want: Class| -> u64 {
        t.counts
            .values()
            .filter(|(c, _)| *c == want)
            .map(|(_, n)| *n)
            .sum()
    };
    println!(
        "\n{binaries} binaries, {} instructions, {} floating-point",
        t.instructions,
        sum(Class::Exact) + sum(Class::Rounded) + sum(Class::Estimate)
    );
    println!(
        "  exact     {:>8}   sign/move/select/compare — no rounding",
        sum(Class::Exact)
    );
    println!(
        "  rounded   {:>8}   IEEE 754 correctly rounded — one mandated answer",
        sum(Class::Rounded)
    );
    println!(
        "  estimate  {:>8}   implementation freedom — the residual's whole surface",
        sum(Class::Estimate)
    );

    println!("\nby mnemonic:");
    for (m, (class, n)) in &t.counts {
        let tag = match class {
            Class::Exact => "exact",
            Class::Rounded => "rounded",
            Class::Estimate => "ESTIMATE",
        };
        println!("  {m:<12} {n:>8}  {tag}");
    }

    if !t.unclassified.is_empty() {
        println!(
            "\nUNCLASSIFIED — the table has fallen behind the ISA, and each of these is a\nfinding until someone says which class it belongs in:"
        );
        for site in t.unclassified.iter().take(40) {
            println!("  {} in {}", site.mnemonic, site.symbol);
        }
    }

    // Printed before the verdict because it *is* part of the verdict: a clean
    // instruction report over a binary that imports `pow` describes only the
    // arithmetic it happened to compile in.
    if t.imported.is_empty() {
        println!(
            "\nimported math: none — every routine is compiled in, so the scan above is the\n\
             whole arithmetic surface and not a sample of it"
        );
    } else {
        println!(
            "\nIMPORTED MATH — resolved by the loader, so this instrument never saw the code\n\
             that computes them and neither did the aarch64 leg's premise (§4.2.1 hazard 1):"
        );
        for site in t.imported.iter().take(40) {
            println!("  {} in {}", site.mnemonic, site.symbol);
        }
    }

    println!();
    if t.estimates.is_empty() && t.unclassified.is_empty() && t.imported.is_empty() {
        println!(
            "VERDICT: every floating-point instruction in this artifact is either exact or\n\
             correctly rounded, and no math routine is imported. There is no instruction\n\
             here whose result an implementation is free to choose, so a qemu-vs-silicon\n\
             divergence would require a CPU that computes a mandated answer wrongly — an\n\
             erratum, not a design difference. That is the residual §8's row carries. It is\n\
             not zero, and only silicon makes it zero; it is much smaller than \"ARM might\n\
             differ\"."
        );
    } else if t.estimates.is_empty() && t.unclassified.is_empty() {
        println!(
            "VERDICT: the compiled-in arithmetic is all exact or correctly rounded, but {} \n\
             math routine(s) are imported and this scan says nothing about them. Attribute\n\
             each before reading the clean instruction report as a clean bill.",
            t.imported.len()
        );
    } else {
        println!(
            "VERDICT: {} estimate-class and {} unclassified instruction(s). Each one is a\n\
             place two conforming implementations may legitimately disagree, which is\n\
             exactly what the cross-architecture contract forbids. Attribute each to its\n\
             crate before trusting any leg that runs this artifact.",
            t.estimates.len(),
            t.unclassified.len()
        );
        for site in t.estimates.iter().take(40) {
            println!("  {} in {}", site.mnemonic, site.symbol);
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// The three classes, on the mnemonics each exists for. `fmul` is mandated,
    /// `fmov` rounds nothing, and `frsqrte` is the one the instrument hunts.
    #[test]
    fn each_class_holds_the_instruction_it_was_written_for() {
        assert_eq!(classify(Arch::A64, "fmul"), Some(Class::Rounded));
        assert_eq!(classify(Arch::A64, "fmadd"), Some(Class::Rounded));
        assert_eq!(classify(Arch::A64, "fmov"), Some(Class::Exact));
        assert_eq!(classify(Arch::A64, "frsqrte"), Some(Class::Estimate));
        assert_eq!(classify(Arch::A64, "frecpe"), Some(Class::Estimate));
        assert_eq!(classify(Arch::X86, "mulsd"), Some(Class::Rounded));
        assert_eq!(classify(Arch::X86, "vfmadd213sd"), Some(Class::Rounded));
        assert_eq!(classify(Arch::X86, "rsqrtps"), Some(Class::Estimate));
        assert_eq!(classify(Arch::X86, "xorps"), Some(Class::Exact));
    }

    /// The property that keeps the table honest: an FP mnemonic nobody listed
    /// is *unclassified*, not silently fine. A table that quietly passed
    /// whatever it did not recognise would report a clean bill for an ISA it
    /// has never seen.
    #[test]
    fn an_unknown_float_instruction_is_reported_rather_than_assumed_safe() {
        assert_eq!(classify(Arch::A64, "fjcvtzs"), None);
        assert!(is_float(Arch::A64, "fjcvtzs"), "it is an f-instruction");
        let out = tally(
            Arch::A64,
            "0000000000000640 <sim::step>:\n 640: fjcvtzs w0, d0\n",
        );
        assert_eq!(out.unclassified.len(), 1);
        assert_eq!(out.unclassified[0].symbol, "sim::step");
        // And a non-FP mnemonic is not dragged in by the same net.
        assert!(!is_float(Arch::A64, "ldr"));
    }

    /// Both disassembler dialects, and the symbol attribution an estimate hit
    /// is only useful with.
    #[test]
    fn the_parser_reads_both_dialects_and_keeps_the_containing_symbol() {
        let llvm = "\
0000000000001000 <gg_math::sim::normalize>:
    1000: frsqrte v0.4s, v1.4s
    1004: fmul    s0, s1, s2
0000000000002000 <other>:
    2000: fadd    d0, d1, d2
";
        let out = tally(Arch::A64, llvm);
        assert_eq!(out.instructions, 3);
        assert_eq!(out.estimates.len(), 1);
        assert_eq!(out.estimates[0].symbol, "gg_math::sim::normalize");
        assert_eq!(out.counts["fmul"], (Class::Rounded, 1));
        assert_eq!(out.counts["fadd"], (Class::Rounded, 1));

        // GNU objdump indents differently and tabs its operands; same answer.
        let gnu = "0000000000001000 <sym>:\n\t1000:\tfadd\td0, d1, d2\n";
        assert_eq!(tally(Arch::A64, gnu).counts["fadd"], (Class::Rounded, 1));
    }

    /// The hole the instruction scan has by construction. A `pow@GLIBC` import
    /// is arithmetic no disassembly of *this* binary can see, so it has to be
    /// found in the dynamic table or the verdict overclaims — and a *defined*
    /// symbol of the same name must not be mistaken for one, or every binary
    /// that compiles `pow` in reports itself as importing it.
    #[test]
    fn an_imported_math_routine_is_found_and_a_compiled_in_one_is_not() {
        let table = "\
0000000000000000      DF *UND*\t0000000000000000  GLIBC_2.17  pow
0000000000000000      DF *UND*\t0000000000000000  GLIBC_2.17  sinf
0000000000000000      DF *UND*\t0000000000000000  GLIBC_2.17  memcpy
0000000000012340 g    DF .text\t0000000000000100  Base        pow
";
        let found = imported_math(table);
        assert_eq!(found, vec!["pow".to_string(), "sinf".to_string()]);
        assert!(!found.iter().any(|s| s == "memcpy"), "not a math routine");
        // The defined `pow` at .text is the same name in the same table and is
        // *not* an import; only the *UND* line counts.
        assert_eq!(found.iter().filter(|s| *s == "pow").count(), 1);
    }

    /// Lines that are not instructions — the header, blank lines, the section
    /// banner — must not become instructions, or every count is inflated by
    /// whatever the disassembler chose to print.
    #[test]
    fn only_instruction_lines_are_counted() {
        let noise = "\
demo: file format elf64-littleaarch64

Disassembly of section .text:

0000000000001000 <sym>:
    1000: fadd d0, d1, d2
";
        let out = tally(Arch::A64, noise);
        assert_eq!(out.instructions, 1, "the banner is not an instruction");
    }

    /// The whole report is a claim about the tree, and `target/` is an attic —
    /// so an artifact whose sources moved under it must drop out, and one whose
    /// age cannot be established must not. Both directions, because a filter
    /// that hid everything would read as a clean bill too.
    #[test]
    fn an_artifact_older_than_its_own_sources_is_not_current() {
        let dir = crate::output_dir().unwrap().join("fp-isa-staleness");
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("thing.rs");
        let binary = dir.join("thing");

        // Source first, then the binary: built *after* what it reads. The
        // depfile names it **relative**, the way cargo does — an absolute path
        // here is what let this test pass while the predicate resolved every
        // real dependency against the wrong directory and answered "current"
        // for all of them.
        std::fs::write(&source, "fn main() {}").unwrap();
        std::fs::write(&binary, "elf").unwrap();
        std::fs::write(
            dir.join("thing.d"),
            format!("{}: thing.rs\n", binary.display()),
        )
        .unwrap();
        assert!(built_from_current_sources(&binary, &dir));

        // Touch the source and it is a census of a program that changed. A
        // rewrite rather than a clock arithmetic, so the mtime is the real one
        // the filesystem records.
        std::thread::sleep(std::time::Duration::from_millis(50));
        std::fs::write(&source, "fn main() { /* edited */ }").unwrap();
        assert!(!built_from_current_sources(&binary, &dir));
        // The same stale binary read from the wrong root: the dependency does
        // not resolve, `all` sees nothing, and the answer flips to "current".
        assert!(built_from_current_sources(
            &binary,
            Path::new("/nonexistent")
        ));

        // No depfile is "cannot tell", which admits rather than hides.
        std::fs::remove_file(dir.join("thing.d")).unwrap();
        assert!(built_from_current_sources(&binary, &dir));
        std::fs::remove_dir_all(&dir).ok();
    }
}
