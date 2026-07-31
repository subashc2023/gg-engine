# Golden Game Engine

A Rust + Vulkan engine core built for one loop: **human plays → agent edits →
rebuild → reload with state intact**. `PLAN.md` (local, untracked) is the plan
of record and the milestone ledger.

## Requirements

- Rust (pinned by `rust-toolchain.toml`), `cargo-deny`, `cargo-nextest`
- LunarG Vulkan SDK (`VULKAN_SDK` set; supplies Slang and the loader)
- For the nightly aarch64 determinism leg (WSL/Linux only): the
  `aarch64-unknown-linux-gnu` rustup target, `gcc-aarch64-linux-gnu`, and
  `qemu-user` (linker and runner are wired in `.cargo/config.toml`)

## Commands

CI is local-first: `cargo xtask ci` *is* CI, not a mirror of one.

| Command | What it does |
|---|---|
| `cargo xtask ci --fast` | Stop-hook tier: fmt + clippy + tests for changed crates (<30 s warm; clean tree passes instantly) |
| `cargo xtask ci --push` | Pre-push: fmt, clippy `-D warnings`, cargo-deny, grep gates, budgets, all tests, FP baseline under the dist profile, shader build, dist feature checks |
| `cargo xtask ci --nightly` | Push tier + dist gate + capability probe + aarch64-under-qemu determinism leg (WSL lane) |
| `cargo xtask ci --weekly` | Nightly tier + weekly gates (most land at M4B) |
| `cargo xtask probe [--system]` | Capability table against the pinned lavapipe (or the system driver); nonzero exit on any missing capability |
| `cargo xtask shaders` | Offline shader build: every `.slang` module → SPIR-V via `slangc` |
| `cargo xtask dist` | Dist gate: build + run the exact `tier-dist` combination, prove the lab equipment unbolted |
| `cargo xtask assets` / `bench` / `capture` | Stubs until their milestone (M9 / M3 / M8) |
| `cargo run -p gg-runtime` | The host shell, dev tier (Tracy + structured logs on) |

A Claude Code Stop hook runs `cargo xtask ci --fast` at the end of every agent
turn; a dirty tree with a red fast tier blocks the turn from ending.
