# Golden Game Engine

A Rust + Vulkan engine core built for one loop: **human plays → agent edits →
rebuild → reload with state intact**. `PLAN.md` (local, untracked) is the plan
of record and the milestone ledger. Current tag: **v0.1.0** — the core phase is
complete through M12.

## Requirements

- Rust (pinned by `rust-toolchain.toml`), `cargo-deny`, `cargo-nextest`,
  `cargo-public-api` (the §5.10 surface gate; needs the pinned nightly)
- LunarG Vulkan SDK (`VULKAN_SDK` set; supplies Slang and the loader). Slang's
  version is machine state `Cargo.lock` cannot see, so `slang-pin.toml` records
  the required version per host and `xtask shaders` asserts it first
- For the nightly aarch64 determinism leg (WSL/Linux only): the
  `aarch64-unknown-linux-gnu` rustup target, `gcc-aarch64-linux-gnu`, and
  `qemu-user` (linker and runner are wired in `.cargo/config.toml`)

## Tiers

CI is local-first: `cargo xtask ci` *is* CI, not a mirror of one. No automated
tier creates an OS window (§1.5) or binds a non-loopback socket.

| Command | What it does |
|---|---|
| `cargo xtask ci --fast` | Stop-hook tier: fmt + clippy + tests for changed crates (<30 s warm; clean tree passes instantly) |
| `cargo xtask ci --push` | Pre-push: fmt, clippy `-D warnings`, cargo-deny, grep gates, the §3 budgets, public-API baselines, all tests, FP baseline under the dist profile, `shaders --check`, `assets --check`, forced rejuvenation, the dormant static-link check, dist feature checks |
| `cargo xtask ci --nightly` | Push tier + dist gate + capability probe + aarch64-under-qemu determinism leg (WSL lane) + the `xtask reload` set + query/extract/asset benches + `dx` + headless GPU tests, GPU-assisted validation, and the golden suite on pinned lavapipe |
| `cargo xtask ci --weekly` | Nightly tier + the fresh-clone gate (`xtask ci --push` from a pristine clone) and the `cargo update` canary |
| `cargo xtask interactive` | **Manual windowed suite** — swapchain/resize/minimize torture and demo WSI runs. Creates windows, so it belongs to no automated tier; run it when touching WSI code |

## Tools

| Command | What it does |
|---|---|
| `cargo xtask run <demo> [flags]` | **Manual, creates a window.** Build the game dylib and launch `gg-runtime` over it. Flags forward to the shell (`--frames`, `--record`, `--replay`) |
| `cargo xtask new <name>` | **Manual, writes to the tree.** A new game crate from the template: copy, rename, register in `[workspace] members`, build and test it. `cargo xtask new 09-orbit` then `cargo xtask run 09-orbit` is the whole of starting a game |
| `cargo xtask probe [--system]` | Capability table against the pinned lavapipe (or the system driver); nonzero exit on any missing capability |
| `cargo xtask gpuav [--adapter <name>]` | GPU-assisted validation: instrumented shaders over the offscreen suite and the real pass list, failing on any message. Catches what the layer alone cannot see — an out-of-range bindless index, a read off the end of a device address |
| `cargo xtask shaders [--check]` | Slang → SPIR-V plus reflected push-constant layouts frozen into generated Rust. `--check` verifies the checked-in artifacts without rewriting them |
| `cargo xtask assets [--check]` | Compile every `demos/*/assets/` tree to `target/assets/<demo>.ggpack`. `--check` builds each twice cleanly and compares bytes (§4.6) |
| `cargo xtask dist` | Dist gate: build + run the exact `tier-dist` combination, prove the lab equipment unbolted and the recorder present, archive the split-debug artifact and prove a dist crash symbolizes against it |
| `cargo xtask reload [--cross-tier\|--segments\|--chaos\|--latency]` | The M5 gates over demo 03: §5.6c across tiers, replay segments across a reload, §5.11's reload cases, and the save→behaviour latency curve. Windowless |
| `cargo xtask replay [--bless]` | Replay determinism material (§5.6): the curated replay's hash sequence against its baseline. `--bless` re-authors both, deliberately |
| `cargo xtask public-api [--bless]` | The §5.10 surface gate, on the pinned nightly |
| `cargo xtask bench [--record]` | Query/extract/asset micros plus a frame macro. `--record` archives this machine's real numbers to `bench/`; never run in a tier |
| `cargo xtask dx [--record]` | §8's developer-experience benchmarks: eight tasks performed as real edits, measured in steps, lines and rebuild latency |
| `cargo xtask backlog` | The P1/P2 items, collected out of the `///`/`//!` comments that defer them. There is no backlog file |
| `cargo xtask timers [--status\|--install\|--uninstall]` | The scheduled nightly/weekly tiers. Installing changes the machine, so nothing does it implicitly |
| `cargo run -p gg-golden -- run\|bless\|graph\|chaos\|capture\|bench` | Golden image suite and its report; `bless` re-writes references, deliberately |

## Demos

Demos 00–02 predate the systems table and are their own thin mains. **Demos 03
onward are game-code crates, not binaries** (§2) — the shell loads them, so
`cargo xtask run` is the only way to play one.

| Demo | What it shows |
|---|---|
| `cargo run -p demo-00-clear` | M1: animated clear, resize-stable; `--frames N` bounds the run |
| `cargo run -p demo-01-triangle` | M2: the golden triangle, with shader hot reload in the dev tier |
| `cargo run -p demo-02-mesh` | M4A/M4B: a textured mesh under a fly camera whose state lives in the ECS. `--record`/`--replay`/`--expect-hash` |
| `cargo xtask run 03-reload` | M5: the Ugly Game — edit a system, save, keep playing |
| `cargo xtask run 04-scene` | M9: a hall loaded from a compiled `.ggpack` |
| `cargo xtask run 05-many` | M10: ten thousand parented objects in four draws |
| `cargo xtask run 06-lit` | M11: shadows, HDR and tonemapping |
| `cargo xtask run 99-template` | M12: the copy-and-delete starting point, 41 lines |

A Claude Code Stop hook runs `cargo xtask ci --fast --hook || exit 2` at the end
of every agent turn; a dirty tree with a red fast tier blocks the turn from
ending. The `|| exit 2` is the layer above the binary: exit 2 is the only code
that blocks, and `xtask` failing to *compile* exits 101 — so without it, an agent
that broke `xtask` would thereby switch off the gate watching everything else.
