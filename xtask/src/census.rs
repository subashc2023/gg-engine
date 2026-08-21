//! What a gate graded, and the refusal that keeps *nothing* from reading as
//! *nothing wrong* (§6 M87).
//!
//! Most gates here are universal quantifiers: every `.slang` module is watched,
//! every declared dependency is reached, every clip a cue names is in the pack.
//! A universal quantifier over an empty set is **true**, so such a gate stops
//! being one the moment its population goes to zero — and it goes to zero for
//! reasons that have nothing to do with the rule: a directory renamed, a table
//! reformatted past a text scan, a filter that grew a clause. The tier prints
//! its usual green line and nobody looks at the count in it.
//!
//! Three comments in this crate already said so, in three different wordings,
//! including one asserting it of a population it did not hold for — *the
//! vacuity guard every "find and check" gate here carries*. Nine gates carried
//! one and seven did not. [`graded`] is the one spelling, and [`check`] is what
//! makes the roster a table rather than a habit: a gate on the list whose body
//! stops calling it fails the tier by name.

use crate::util::workspace_root;

/// Refuse a population of zero before grading it.
///
/// `subject` names what was counted and `remedy` says what an empty count means
/// *here* — the two halves a reader needs, since "found nothing" is a different
/// bug in a directory walk than in a text scan.
///
/// # Errors
///
/// Whenever `found` is zero.
pub fn graded(found: usize, subject: &str, remedy: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        found > 0,
        "{subject}: nothing to grade, so this gate passed by reading nothing rather than by \
         finding nothing wrong (§6 M87) — {remedy}"
    );
    Ok(())
}

/// Every gate whose subject is a population, and the file it lives in.
///
/// A table for `shell::LEGS`' reason (§6 M81): the chain of remembered
/// `is_empty()` checks it replaced was green over the seven that had none. Its
/// residual is the same one — a gate written tomorrow and not added here is not
/// checked — which is why the entry is one line beside the gate rather than a
/// registration ceremony.
const POPULATION_GATES: &[(&str, &str)] = &[
    ("assets.rs", "clips_resolve"),
    ("assets.rs", "run"),
    ("budgets.rs", "headless_parses"),
    ("budgets.rs", "imported_math_lists"),
    ("budgets.rs", "shader_block_loaders"),
    ("budgets.rs", "unused_dependencies"),
    ("budgets.rs", "watched_shader_modules"),
    ("budgets.rs", "widget_provenance"),
    ("ci.rs", "allowlist_crosscheck"),
    ("ci.rs", "greps"),
    ("ci.rs", "nightly"),
    ("ci.rs", "no_imported_math"),
    ("deps.rs", "check_folder"),
    ("dist.rs", "gate"),
    ("shaders.rs", "build_all"),
];

/// Every gate on the roster still calls [`graded`].
///
/// Textual, like the four cross-file gates in [`budgets`](crate::budgets) it
/// sits beside, and for the same reason: the alternative is a trait every gate
/// implements, which buys nothing a rename cannot already break and costs the
/// plain function signatures that make these readable.
pub fn check() -> anyhow::Result<()> {
    let root = workspace_root().join("xtask/src");
    let mut ungated = Vec::new();
    for (file, name) in POPULATION_GATES {
        let path = root.join(file);
        let text = std::fs::read_to_string(&path)
            .map_err(|e| anyhow::anyhow!("no {}: {e}", path.display()))?;
        let Some(body) = function_body(&text, name) else {
            anyhow::bail!(
                "xtask/src/{file} declares no `fn {name}` — the census roster names a gate that \
                 was renamed or deleted, so nothing is checking it (§6 M87)"
            );
        };
        if !body.contains("graded(") {
            ungated.push(format!("{file}: {name}"));
        }
    }
    anyhow::ensure!(
        ungated.is_empty(),
        "population gate(s) with no vacuity guard: {} — each grades every member of a set, so an \
         empty set passes it (§6 M87). Call `census::graded` with what was found, or drop the \
         roster line if the gate stopped being one",
        ungated.join(", ")
    );
    // The roster's own vacuity, which is the joke this module would otherwise be
    // playing on itself.
    graded(
        POPULATION_GATES.len(),
        "the census roster",
        "POPULATION_GATES is empty and this gate checked nothing",
    )?;
    println!(
        "xtask: {} population gate(s) refuse an empty population (§6 M87)",
        POPULATION_GATES.len()
    );
    Ok(())
}

/// The body of `fn <name>` in `text`, brace-matched from its signature.
///
/// Nested functions and closures come along, which is right — a guard inside one
/// is still inside the gate. String and comment contents are not skipped, so a
/// stray brace in either would end the body early; the failure that produces is
/// a *false* report of a missing guard, which is the direction to be wrong in.
fn function_body<'t>(text: &'t str, name: &str) -> Option<&'t str> {
    let at = text
        .match_indices(&format!("fn {name}("))
        .find(|(at, _)| {
            // `fn watched(` must not answer for `fn watched_names(`, and a
            // preceding identifier character means this is some other name's
            // tail.
            text[..*at]
                .chars()
                .next_back()
                .is_none_or(|c| !(c.is_ascii_alphanumeric() || c == '_'))
        })
        .map(|(at, _)| at)?;
    let open = at + text[at..].find('{')?;
    let mut depth = 0i32;
    for (offset, c) in text[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[open..=open + offset]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Every `.rs` file under `xtask/src`, so the roster can be graded for
/// *coverage* as well as for currency — see the test.
#[cfg(test)]
fn xtask_sources() -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    crate::util::walk_rs(&workspace_root().join("xtask/src"), &mut files);
    files.sort();
    files
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The guard, both ways. Trivial, and here because the whole milestone is
    /// that a check nobody exercised is a check nobody has.
    #[test]
    fn an_empty_population_is_refused_and_a_populated_one_is_not() {
        assert!(graded(0, "subject", "remedy").is_err());
        let message = graded(0, "the widget kinds", "the parse found none")
            .expect_err("zero is refused")
            .to_string();
        assert!(message.contains("the widget kinds") && message.contains("the parse found none"));
        assert!(graded(1, "subject", "remedy").is_ok());
    }

    /// The roster gate, planted red three ways — a body with no guard, a name
    /// that is not there, and the roster itself empty.
    #[test]
    fn the_roster_gate_names_a_gate_that_stopped_guarding() {
        const GATED: &str = "fn a() {\n    graded(n, \"x\", \"y\")?;\n}\n";
        const BARE: &str = "fn a() {\n    for x in xs {}\n    Ok(())\n}\n";
        assert!(
            function_body(GATED, "a")
                .expect("found")
                .contains("graded(")
        );
        assert!(!function_body(BARE, "a").expect("found").contains("graded("));
        assert!(function_body(BARE, "b").is_none());
        // A name that is another name's tail is not this name.
        assert!(function_body("fn watched_names() {\n}\n", "watched").is_none());
        // The brace matcher takes the whole body and stops at its own close.
        let nested =
            "fn a() {\n    if x {\n        graded(1, \"s\", \"r\")?;\n    }\n}\nfn b() {}\n";
        let body = function_body(nested, "a").expect("found");
        assert!(body.contains("graded(") && !body.contains("fn b"));
    }

    /// The roster against the tree: every entry resolves, and — the half a
    /// roster cannot give itself — no *other* function in `xtask/src` grades a
    /// population without being on it.
    ///
    /// Approximated by the one shape that is mechanically visible: a function
    /// that calls [`graded`] and is not on the roster. That direction catches
    /// the likelier drift (a guard added, the line forgotten) and leaves the
    /// other one — a new gate with no guard at all — where §6 M87's residual
    /// says it is.
    #[test]
    fn the_roster_and_the_tree_agree_about_who_grades_a_population() {
        check().expect("the roster gate is green on this tree");
        for file in xtask_sources() {
            let name = file
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_owned();
            if name == "census.rs" {
                continue; // this module's own calls are the machinery
            }
            let text = std::fs::read_to_string(&file).expect("a source file");
            for (at, _) in text.match_indices("graded(") {
                let line = text[..at].lines().count();
                let enclosing = text[..at]
                    .rmatch_indices("fn ")
                    .find_map(|(fat, _)| {
                        text[fat + 3..]
                            .split('(')
                            .next()
                            .map(|n| n.trim().to_owned())
                    })
                    .unwrap_or_default();
                assert!(
                    POPULATION_GATES
                        .iter()
                        .any(|(f, n)| *f == name && *n == enclosing),
                    "{name}:{line}: `{enclosing}` guards a population and is not on the roster"
                );
            }
        }
    }
}
