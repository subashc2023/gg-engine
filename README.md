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
| `cargo xtask ci --nightly` | Push tier + dist gate + capability probe + aarch64-under-qemu determinism leg (WSL lane) + headless GPU tests and the golden suite on pinned lavapipe. Windowless by construction — no automated tier creates a window (§1.5) |
| `cargo xtask ci --weekly` | Nightly tier + weekly gates (most land at M4B) |
| `cargo xtask interactive` | **Manual windowed suite** — swapchain/resize/minimize torture and demo WSI runs, dev and dist profiles. Creates windows, so it belongs to no automated tier; run it when touching WSI code |
| `cargo xtask probe [--system]` | Capability table against the pinned lavapipe (or the system driver); nonzero exit on any missing capability |
| `cargo xtask shaders [--check]` | Offline shader build: every `.slang` module compiled in-process → SPIR-V plus reflected push-constant layouts frozen into generated Rust. `--check` verifies the checked-in artifacts without rewriting them |
| `cargo xtask dist` | Dist gate: build + run the exact `tier-dist` combination, prove the lab equipment unbolted |
| `cargo xtask assets` / `bench` / `capture` | Stubs until their milestone (M9 / M3 / M8) |
| `cargo run -p gg-runtime` | The host shell, dev tier (Tracy + structured logs on) |
| `cargo run -p demo-00-clear` | Demo 00 (M1): animated clear, resize-stable; `--frames N` bounds the run; headless runs use an invisible window |
| `cargo run -p demo-01-triangle` | Demo 01 (M2): the golden triangle, with shader hot reload in the dev tier — edit `shaders/triangle.slang` and the picture changes without a restart |
| `cargo run -p gg-golden -- run` | Golden image suite: offscreen render → compare against the checked-in per-backend references (`bless` re-writes them, deliberately) |

A Claude Code Stop hook runs `cargo xtask ci --fast` at the end of every agent
turn; a dirty tree with a red fast tier blocks the turn from ending.
