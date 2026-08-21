//! The shader's scalar constants, as Rust (§6 M86).
//!
//! HLSL has no `sizeof` and Slang exports no constant through the reflection
//! API we use, so every number both sides need — a struct's stride, a mode's
//! discriminant, a flag's bit — is typed twice. §6 M81 found that class and
//! built the answer for one file: a text scan of `pbr.slang` inside a
//! `gg-render` test, which is why the strides in that one include are the only
//! ones checked against the shader rather than against a second copy of
//! themselves. Seven modules and nine `SHADING_*` bits were still hand-carried.
//!
//! This is that scan promoted to where shaders are already read, so the numbers
//! stop being *asserted equal* and start being **the same number**: the host
//! imports `pbr::FRAME_STRIDE` instead of restating 4440 and asking a test
//! whether it guessed right.
//!
//! **Literal scalars only**, which is §6 M81's policy kept rather than widened.
//! A derived constant (`CLUSTER_BASE = FRAME_STRIDE + MAX_LIGHTS *
//! LIGHT_STRIDE`) is skipped, so the host recomputes the derivation from the
//! terms — which checks the arithmetic as well as its inputs, and is a stronger
//! claim than importing the answer. Aggregates (`float3`, `float3x3`) are
//! skipped because no host site wants one; a Slang expression grammar here would
//! be a compiler nobody asked for.

use crate::ShaderError;

/// One `static const <scalar> NAME = <literal>;` lifted out of a `.slang` file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaderConst {
    /// The shader-side name, reused verbatim as the Rust one.
    pub name: String,
    /// The Rust type this maps to: `u32`, `i32` or `f32`.
    pub rust_type: &'static str,
    /// The literal as Rust source — suffix stripped, `f32` given a decimal
    /// point so the emitted token is a float literal and not an integer.
    pub value: String,
}

/// Slang scalar types worth exporting, and what they become.
const SCALARS: &[(&str, &str)] = &[("uint", "u32"), ("int", "i32"), ("float", "f32")];

/// Scan `source` for top-level literal scalar constants, in declaration order.
///
/// Only lines that *begin* a declaration at column zero are considered: a
/// `static const` inside a function body is a local whose name may repeat, and
/// nothing outside the file can name it anyway.
///
/// # Errors
///
/// [`ShaderError::Unsupported`] if two exported constants share a name — the
/// generated module would silently keep one of them.
pub fn scan(source: &str) -> Result<Vec<ShaderConst>, ShaderError> {
    let mut found: Vec<ShaderConst> = Vec::new();
    for line in source.lines() {
        let Some(rest) = line.strip_prefix("static const ") else {
            continue;
        };
        // Comments after the semicolon are common and carry no value here.
        let Some((decl, _)) = rest.split_once(';') else {
            continue;
        };
        let Some((ty, body)) = decl.split_once(' ') else {
            continue;
        };
        let Some((_, rust_type)) = SCALARS.iter().find(|(slang, _)| *slang == ty) else {
            continue; // an aggregate, or a type no host site wants
        };
        let Some((name, literal)) = body.split_once('=') else {
            continue;
        };
        let (name, literal) = (name.trim(), literal.trim());
        // `SCREAMING_CASE` is what a shared constant looks like; a lower-case
        // one is the shader's own business. Digits are in — `OUTPUT_HDR10` and
        // `REC709` are names, and rejecting them silently dropped the HDR
        // contract from the first version of this scan. Not leading, because a
        // Rust identifier cannot start with one.
        let named = !name.is_empty()
            && name.starts_with(|c: char| c.is_ascii_uppercase())
            && name
                .chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_');
        if !named {
            continue;
        }
        let Some(value) = rust_literal(literal, rust_type) else {
            continue; // derived from other constants — the host recomputes it
        };
        if found.iter().any(|c| c.name == name) {
            return Err(ShaderError::Unsupported(format!(
                "`{name}` is declared twice; the generated module would keep one silently"
            )));
        }
        found.push(ShaderConst {
            name: name.to_owned(),
            rust_type,
            value,
        });
    }
    Ok(found)
}

/// A Slang literal as Rust source, or `None` when it is not a plain literal.
///
/// The `None` arm is the load-bearing one: it is what makes a derived constant
/// *absent* from the generated module rather than present and wrong, so a host
/// that wants one gets a compile error naming it.
fn rust_literal(literal: &str, rust_type: &str) -> Option<String> {
    let digits = literal.strip_suffix('u').unwrap_or(literal);
    let digits = digits.strip_suffix('f').unwrap_or(digits);
    if digits.is_empty() {
        return None;
    }
    match rust_type {
        "f32" => {
            // Parsed to reject an expression, then emitted as written — a
            // reformat by `f32`'s own `Display` would put a different token in a
            // byte-compared file for no reason.
            digits.parse::<f32>().ok()?;
            Some(if digits.contains(['.', 'e', 'E']) {
                digits.to_owned()
            } else {
                format!("{digits}.0")
            })
        }
        "i32" => digits.parse::<i32>().ok().map(|_| digits.to_owned()),
        _ => digits.parse::<u32>().ok().map(|_| digits.to_owned()),
    }
}

#[cfg(test)]
mod tests {
    // unwrap is permitted in tests (§2, Error handling row).
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    fn names(source: &str) -> Vec<String> {
        scan(source)
            .expect("scan")
            .into_iter()
            .map(|c| c.name)
            .collect()
    }

    #[test]
    fn literal_scalars_come_out_typed() {
        let got = scan(
            "static const uint MODE_RED = 3u;   // trailing comment\n\
             static const int  SIGNED = -2;\n\
             static const float PI = 3.5;\n\
             static const float WHOLE = 8;\n",
        )
        .expect("scan");
        let pairs: Vec<_> = got
            .iter()
            .map(|c| (c.name.as_str(), c.rust_type, c.value.as_str()))
            .collect();
        assert_eq!(
            pairs,
            [
                ("MODE_RED", "u32", "3"),
                ("SIGNED", "i32", "-2"),
                ("PI", "f32", "3.5"),
                // Emitted with a point, or the generated token is an integer.
                ("WHOLE", "f32", "8.0"),
            ]
        );
    }

    #[test]
    fn what_is_skipped_is_skipped_rather_than_guessed() {
        // Derived, aggregate, local, and lower-case — each absent from the
        // module, so a host that wants one fails to compile instead of
        // importing a number this file made up.
        let source = "static const uint BASE = 8u;\n\
                      static const uint DERIVED = BASE + 1;\n\
                      static const float3 TINT = float3(1, 0, 0);\n\
                      static const uint scratch = 1u;\n\
                      \x20   static const uint LOCAL = 2u;\n";
        assert_eq!(names(source), ["BASE"]);
    }

    #[test]
    fn a_digit_in_a_name_is_a_name() {
        // The first version of this scan accepted `[A-Z_]` only, which silently
        // dropped `OUTPUT_HDR10` — so `output_index` imported two of the post
        // pass's three contracts and failed to compile on the third. A skipped
        // constant is invisible by design, which is what makes the alphabet the
        // one part of this parser worth a test of its own.
        assert_eq!(
            names("static const uint OUTPUT_HDR10 = 1u;\nstatic const uint REC709 = 2u;\n"),
            ["OUTPUT_HDR10", "REC709"]
        );
    }

    #[test]
    fn a_repeated_name_is_an_error_rather_than_a_silent_winner() {
        let err = scan("static const uint A = 1u;\nstatic const uint A = 2u;\n")
            .expect_err("a duplicate must not be silently resolved");
        assert!(err.to_string().contains('A'), "{err}");
    }

    #[test]
    fn a_scan_that_matched_nothing_cannot_pass_unnoticed() {
        // §6 M81's parser needed `assert!(slang_const("FRAME_STRIDE") > 0)`
        // because a scan matching nothing grades nothing and every assertion
        // downstream of it goes green. Nothing here needs that guard, and the
        // reason is the point of this module: the host *imports* these, so a
        // shader reformatted past the scan takes `pbr::FRAME_STRIDE` out of the
        // generated file and `gg-render` stops compiling. This test states the
        // shape the guard rests on — an empty scan produces an empty module.
        assert!(scan("// nothing declared here\n").expect("scan").is_empty());
    }
}
