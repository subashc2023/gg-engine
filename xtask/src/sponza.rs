//! `cargo xtask sponza` (§6 M59) — fetch the one scene this repository cannot
//! hold, and compile it.
//!
//! Demo 06's module header records the refusal at §6 M11: Sponza is about fifty
//! megabytes and does not go in a git history. What it did not record is *how*
//! to get it, so "check the renderer against the scene the literature uses" was
//! a paragraph of instructions rather than a command. This is the command.
//!
//! **It reaches the network, which nothing else in `xtask` does.** That is why
//! it is manual, why no tier calls it, and why every byte it writes is checked
//! before it is kept: a captive portal answers every request with an HTML login
//! page, and 200 OK plus a `.jpg` extension is not evidence of a JPEG. The
//! magic-number check below is what turns that into a failure with a name
//! instead of a texture compiler crashing on a `<!DOCTYPE`.
//!
//! `curl` rather than an HTTP crate: it is on both hosts (Windows has shipped it
//! since 1803), it multiplexes the sixty-nine images over one connection, and it
//! keeps `xtask`'s dependency list — which §3 does not budget but this tree
//! watches anyway — free of a TLS stack for one manual command.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail, ensure};

use crate::util::workspace_root;

/// Where the glTF lives upstream. The Khronos sample repository rather than
/// Crytek's original: it is the metallic-roughness conversion every modern
/// renderer is compared on, it is CC BY 4.0, and it is the one `ggc`'s importer
/// can read without a specular-glossiness path this tree does not have.
const BASE: &str =
    "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/Sponza/glTF";

/// The one file whose name has to be known. Everything else is read out of it —
/// a checked-in list of sixty-nine hash-named JPEGs would be a list nobody can
/// review and one upstream rename away from wrong.
const ROOT: &str = "Sponza.gltf";

/// Where it lands. Under the demo, like every other demo's sources, and
/// `.gitignore`d there rather than anywhere clever.
const DEST: &str = "demos/14-sponza/assets";

/// No flags. The command does one thing, and a `--force` would be a way to
/// re-download fifty megabytes by typo — deleting `demos/14-sponza/assets` is
/// the spelling for that, and it says what it does.
pub fn run(_args: &[&str]) -> Result<()> {
    let dest = workspace_root().join(DEST);
    std::fs::create_dir_all(&dest).with_context(|| format!("creating {}", dest.display()))?;

    fetch(&dest, &[ROOT.to_owned()])?;
    let gltf = std::fs::read_to_string(dest.join(ROOT))
        .with_context(|| format!("reading the {ROOT} just fetched"))?;
    let referenced = uris(&gltf);
    ensure!(
        !referenced.is_empty(),
        "{ROOT} names no `uri` at all — what arrived is not the glTF (a captive portal, or an \
         upstream layout change), and fetching nothing would look like success"
    );
    println!("xtask sponza: {ROOT} names {} file(s)", referenced.len());
    fetch(&dest, &referenced)?;
    for name in &referenced {
        check(&dest.join(name))?;
    }

    // Built here rather than left to `xtask assets`, which skips this demo on
    // purpose (see that file's `FETCHED`): the point of the command is that one
    // invocation leaves a runnable demo.
    let out = workspace_root().join("target/assets/14-sponza.ggpack");
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let stats = ggc::build::build(&dest, &out)?;
    println!(
        "xtask sponza: {} assets in {} — {} compiled, {} reused",
        stats.assets,
        out.display(),
        stats.compiled,
        stats.reused
    );
    println!("xtask sponza: now `cargo xtask run 14-sponza --editor`");
    Ok(())
}

/// One `curl` for the whole batch: HTTP/2 over one connection, and one process
/// to check the exit status of.
///
/// `--fail` is the load-bearing flag. Without it curl writes a 404's body to the
/// file and exits zero, which is how a missing asset becomes a four-hundred-byte
/// "Not Found" that the importer reads as a corrupt JPEG.
fn fetch(dest: &Path, names: &[String]) -> Result<()> {
    let mut cmd = Command::new("curl");
    cmd.current_dir(dest).args([
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--retry",
        "3",
    ]);
    for name in names {
        // Skipped when it is already there and non-empty: this command is run
        // again after an interrupted one, and re-downloading fifty megabytes to
        // find out nothing changed is the wrong default.
        if dest.join(name).metadata().is_ok_and(|m| m.len() > 0) {
            continue;
        }
        cmd.arg("--remote-name").arg(format!("{BASE}/{name}"));
    }
    // Every name was already present — `curl` with no URL is an error and there
    // is nothing to ask for.
    if !cmd.get_args().any(|a| a == "--remote-name") {
        println!("xtask sponza: {} file(s) already present", names.len());
        return Ok(());
    }
    let status = cmd
        .status()
        .context("running `curl` — it is what this command fetches with, and it is not on PATH")?;
    ensure!(status.success(), "curl exited {status}");
    Ok(())
}

/// Every distinct `"uri"` value in a glTF document, in first-seen order.
///
/// A scan and not a JSON parse, deliberately: `xtask` holds no JSON crate and
/// this is the one place in the tree that would want one, for a field whose
/// grammar is a quoted string. What it can get wrong is a `uri` inside a string
/// *value* somewhere else in the document — which would fetch a name that 404s,
/// and `--fail` turns that into an error rather than a silent wrong file.
fn uris(gltf: &str) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for after in gltf.split("\"uri\"").skip(1) {
        let Some(rest) = after.split_once(':').map(|(_, r)| r) else {
            continue;
        };
        let Some(value) = rest.split('"').nth(1) else {
            continue;
        };
        // A data URI is content, not a file — nothing to fetch and nothing to
        // name a local path after.
        if value.starts_with("data:") || found.iter().any(|seen| seen == value) {
            continue;
        }
        found.push(value.to_owned());
    }
    found
}

/// What arrived is the kind of file its extension claims.
///
/// The failure this exists for has one shape and it is not hypothetical: an
/// intercepting proxy returns HTML with a 200, curl is satisfied, and the first
/// thing to notice is a texture compiler panicking three minutes later on a file
/// whose name gives no hint which of sixty-nine it was.
fn check(path: &PathBuf) -> Result<()> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    ensure!(!bytes.is_empty(), "{} arrived empty", path.display());
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let magic: &[u8] = match extension.as_str() {
        "jpg" | "jpeg" => &[0xff, 0xd8, 0xff],
        "png" => &[0x89, b'P', b'N', b'G'],
        // `.bin` is a raw buffer with no signature, and `.gltf` is JSON whose
        // first non-space byte is a brace. Nothing else is expected; a new
        // extension upstream should be looked at rather than waved through.
        "bin" => return Ok(()),
        "gltf" => b"{",
        other => bail!(
            "{} has an extension this command does not know how to check ({other}) — add it \
             rather than letting an unverified file through",
            path.display()
        ),
    };
    ensure!(
        bytes.starts_with(magic),
        "{} is {} bytes that do not begin like a {extension} — the usual cause is a proxy or a \
         captive portal answering with an HTML page",
        path.display(),
        bytes.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scan finds every uri once, skips a data one, and is not confused by
    /// the whitespace glTF exporters actually emit (`"uri" : "x.jpg"`).
    #[test]
    fn uris_are_read_once_each_and_data_is_skipped() {
        let doc = r#"{ "buffers":[{"uri" : "Sponza.bin"}],
                      "images":[{"uri":"a.jpg"},{"uri":"a.jpg"},
                                {"uri":"data:image/png;base64,AAAA"}] }"#;
        assert_eq!(uris(doc), ["Sponza.bin", "a.jpg"]);
    }

    /// An HTML login page saved as a `.jpg` is the failure the check is for, and
    /// it is named rather than passed on to the texture compiler.
    #[test]
    fn a_captive_portal_page_is_refused_by_name() {
        let dir = std::env::temp_dir().join("gg-sponza-check");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("portal.jpg");
        std::fs::write(&path, b"<!DOCTYPE html><html>sign in</html>").expect("write");
        let message = check(&path).expect_err("html is not a jpeg").to_string();
        assert!(message.contains("captive portal"), "{message}");
        std::fs::remove_file(&path).ok();
    }
}
