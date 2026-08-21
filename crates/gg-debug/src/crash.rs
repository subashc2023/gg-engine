//! Panic reports (§4.8): the file a crash leaves behind, with the log tail
//! attached.
//!
//! A panic message on a terminal nobody was watching is the whole of what a
//! stock build tells you. The report is the same message plus what the process
//! was *doing* — [`tail`](crate::tail)'s last few hundred lines, which is where
//! a device-lost report, the breadcrumbs and the frame's numbers already are —
//! and the command that produced it, so reproduction is a copy and paste.
//!
//! Composition is a pure function over data the hook has already extracted:
//! formatting a report is the one thing that must not itself panic, and a
//! function taking `&str`s is a function a test can run.

use std::io::Write as _;
use std::panic::PanicHookInfo;
use std::path::{Path, PathBuf};

/// What the report calls the process it came out of.
#[derive(Clone, Copy)]
pub struct Product {
    /// Binary name.
    pub name: &'static str,
    /// Its version.
    pub version: &'static str,
    /// Which named tier it was built as (§3).
    pub tier: &'static str,
}

/// Install the hook. Call once, after [`init`](crate::init) — the tail has to
/// exist before it can be attached, and a report from a process whose logging
/// never came up is a report of nothing.
///
/// Replaces the default hook rather than chaining it: two hooks would print the
/// panic twice, and the second line is the one that says where the report went.
pub fn install(product: Product) {
    std::panic::set_hook(Box::new(move |info| hook(product, info)));
}

fn hook(product: Product, info: &PanicHookInfo<'_>) {
    let message = message(info.payload());
    let location = info
        .location()
        .map(ToString::to_string)
        .unwrap_or_else(|| "<unknown location>".into());
    let report = compose(
        product,
        &message,
        &location,
        &std::backtrace::Backtrace::force_capture().to_string(),
        &crate::tail(),
    );
    let written = write(&std::env::temp_dir(), &report);
    // stderr directly: `tracing` may be mid-emit on this very thread, and a
    // panic hook is the last place to re-enter a subscriber.
    let mut err = std::io::stderr().lock();
    let _ = writeln!(
        err,
        "\n{} {} ({}) crashed.",
        product.name, product.version, product.tier
    );
    let _ = writeln!(err, "\n  panicked at {location}: {message}");
    let _ = match &written {
        Ok(path) => writeln!(
            err,
            "\nA report with the last {} log lines was written to:\n  {}\n\nAttach it to the bug \
             report — with the replay, if the run was recording (§1.2).",
            crate::log_tail::len(),
            path.display()
        ),
        // The report is the fallback for a crash nobody watched; if it cannot be
        // written, the terminal is all there is, so it gets the whole thing.
        Err(e) => writeln!(err, "\nNo report could be written ({e}):\n{report}"),
    };
}

/// The panic payload as a string. `&str` and `String` are what `panic!` and
/// `assert!` produce; anything else is a `panic_any` this cannot render.
fn message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        return (*s).to_owned();
    }
    if let Some(s) = payload.downcast_ref::<String>() {
        return s.clone();
    }
    "<non-string panic payload>".into()
}

/// The report's text. Pure, so the shape is a test's business and not a
/// crash's.
fn compose(
    product: Product,
    message: &str,
    location: &str,
    backtrace: &str,
    tail: &[String],
) -> String {
    let mut out = format!(
        "{} {} ({} tier) panicked\n\n\
         location: {location}\n\
         message:  {message}\n\
         platform: {} {}\n\
         command:  {}\n",
        product.name,
        product.version,
        product.tier,
        std::env::consts::OS,
        std::env::consts::ARCH,
        // The reproduction, verbatim. Not `env::args_os` prettified: a path with
        // a space in it must come back out quotable rather than plausible.
        std::env::args().collect::<Vec<_>>().join(" "),
    );
    out.push_str("\n--- log tail ---\n");
    for line in tail {
        out.push_str(line);
        out.push('\n');
    }
    out.push_str("\n--- backtrace ---\n");
    out.push_str(backtrace);
    out.push('\n');
    out
}

/// Write the report beside the process that produced it. Named by pid: two
/// crashed sessions are two files, and a session that crashes twice is not a
/// thing that happens.
fn write(dir: &Path, report: &str) -> std::io::Result<PathBuf> {
    let path = dir.join(format!("gg-crash-{}.txt", std::process::id()));
    std::fs::write(&path, report)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    const PRODUCT: Product = Product {
        name: "gg-runtime",
        version: "0.1.0",
        tier: "dev",
    };

    #[test]
    fn a_report_carries_the_panic_the_tail_and_the_command() {
        let tail = vec![
            "INFO gg::frame tick=41".to_owned(),
            "ERROR gg::crash lost".to_owned(),
        ];
        let report = compose(
            PRODUCT,
            "index out of bounds",
            "crates/gg-ecs/src/lib.rs:12:5",
            "  0: gg_ecs::query",
            &tail,
        );
        assert!(
            report.contains("gg-runtime 0.1.0 (dev tier) panicked"),
            "{report}"
        );
        assert!(report.contains("message:  index out of bounds"), "{report}");
        assert!(
            report.contains("location: crates/gg-ecs/src/lib.rs:12:5"),
            "{report}"
        );
        // The tail is the whole reason the file beats the terminal line.
        assert!(report.contains("ERROR gg::crash lost"), "{report}");
        assert!(report.contains("0: gg_ecs::query"), "{report}");
        // The command that produced it — here, the test binary's own.
        assert!(report.contains("command:  "), "{report}");
    }

    /// A tail that never filled is not an error: the report is thinner, and it
    /// still says what panicked and where.
    #[test]
    fn an_empty_tail_still_composes() {
        let report = compose(PRODUCT, "boom", "src/main.rs:1:1", "", &[]);
        assert!(report.contains("--- log tail ---"), "{report}");
        assert!(report.contains("message:  boom"), "{report}");
    }

    #[test]
    fn the_report_lands_in_a_file_named_for_the_process() {
        let dir = std::env::temp_dir().join(format!("gg-crash-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = write(&dir, "report body").unwrap();
        assert!(path.ends_with(format!("gg-crash-{}.txt", std::process::id())));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "report body");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Both payload shapes a Rust program actually produces have to render:
    /// `panic!("literal")` hands over a `&str`, `panic!("{x}")` a `String`.
    /// Anything else says so rather than reporting an empty message.
    #[test]
    fn every_payload_shape_renders() {
        assert_eq!(message(&"a literal"), "a literal");
        assert_eq!(message(&String::from("a format")), "a format");
        assert!(message(&42u32).contains("non-string"));
    }
}
