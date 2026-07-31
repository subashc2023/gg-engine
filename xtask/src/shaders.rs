//! `xtask shaders` — the offline path (§4.4), M0A form: compile every `.slang`
//! module in the tree to SPIR-V via `slangc` (the CLI fallback the in-process
//! bindings are allowed to fall back to). Structured per-target from day one:
//! the one-variant enum below is the console door (§2, Console portability row).
//! Content-hashed incrementality arrives when shader counts make it matter.

use crate::util::workspace_root;
use std::path::PathBuf;

#[derive(Clone, Copy)]
pub enum Target {
    Spirv,
}

impl Target {
    fn slangc_name(self) -> &'static str {
        match self {
            Target::Spirv => "spirv",
        }
    }
    fn out_dir(self) -> &'static str {
        match self {
            Target::Spirv => "spirv",
        }
    }
}

const TARGETS: &[Target] = &[Target::Spirv];

pub fn build_all() -> anyhow::Result<()> {
    let root = workspace_root();
    let mut modules = Vec::new();
    for base in ["crates", "demos"] {
        find_slang(&root.join(base), &mut modules);
    }
    if modules.is_empty() {
        println!("xtask shaders: no .slang modules in the tree");
        return Ok(());
    }

    let slangc = slangc_path();
    for target in TARGETS {
        let out_dir = root.join("target/shaders").join(target.out_dir());
        std::fs::create_dir_all(&out_dir)?;
        for module in &modules {
            let stem = module
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let out = out_dir.join(format!("{stem}.spv"));
            crate::util::run(
                std::process::Command::new(&slangc).args([
                    &module.to_string_lossy() as &str,
                    "-target",
                    target.slangc_name(),
                    "-o",
                    &out.to_string_lossy(),
                ]),
                &format!("slangc {stem}.slang -> {}", target.slangc_name()),
            )?;
        }
    }
    println!("xtask shaders: {} module(s) built", modules.len());
    Ok(())
}

fn slangc_path() -> PathBuf {
    let exe = if cfg!(windows) {
        "slangc.exe"
    } else {
        "slangc"
    };
    if let Ok(sdk) = std::env::var("VULKAN_SDK") {
        let p = PathBuf::from(&sdk)
            .join(if cfg!(windows) { "Bin" } else { "bin" })
            .join(exe);
        if p.exists() {
            return p;
        }
    }
    if let Ok(dir) = std::env::var("SLANG_DIR") {
        let p = PathBuf::from(&dir).join("bin").join(exe);
        if p.exists() {
            return p;
        }
    }
    PathBuf::from(exe) // PATH fallback
}

fn find_slang(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|n| n == "target") {
                continue;
            }
            find_slang(&path, out);
        } else if path.extension().is_some_and(|e| e == "slang") {
            out.push(path);
        }
    }
}
