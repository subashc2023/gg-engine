//! `gg-tools mcp` — the running game's context, served to an agent (§6 M16).
//!
//! §1's loop needs the agent to know what the *editor* knows: what the last
//! reload did, whether it was refused and by what name, the state hash either
//! side of the edit, and where the recording of what the human just did is. All
//! of that is `gg-agent`'s record, published by a live shell. This is the read
//! side, spoken as MCP over stdio so `claude` running in an ordinary terminal
//! reaches it the same way the editor's panel will.
//!
//! # Why stdio and a file, and not a socket
//!
//! A running game cannot be spawned by an MCP client, so the obvious shape is an
//! HTTP server inside the shell that the client connects to. That is the wrong
//! trade here twice over: a listening socket is this tree's most expensive
//! recurring cost (see the `tracy-client` note in the workspace manifest — a
//! listener bound before `main` cost a firewall prompt per rebuilt executable),
//! and it would put an async stack in the shell to answer a question that
//! changes once per reload. So the shell *publishes* and this reads. The record
//! outlives the process that wrote it, which the socket version would not.
//!
//! Manual and windowless like every instrument, out of the shipping graph, and
//! it gates nothing.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

/// The protocol version answered when a client sends none. Clients state their
/// own in `initialize` and this server has no version-dependent behaviour — it
/// reads a file and returns text — so the client's is echoed when present.
const FALLBACK_PROTOCOL: &str = "2025-06-18";

pub fn run(args: &[String]) -> anyhow::Result<()> {
    anyhow::ensure!(
        args.is_empty(),
        "gg-tools mcp takes no arguments — the record's location comes from \
         GG_AGENT_DIR, defaulting to `target/gg-agent`"
    );
    let dir = record_dir();
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    // Line-delimited JSON-RPC. A line that is not a request at all is skipped
    // rather than fatal: killing the server on one bad frame would take the
    // human's whole session down with it.
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if let Some(response) = respond(&line, &dir) {
            writeln!(stdout, "{response}")?;
            stdout.flush()?;
        }
    }
    Ok(())
}

/// Where the shell publishes. Mirrors `gg_runtime`'s own default, and the env
/// var is how two sessions on one tree stay out of each other's record.
fn record_dir() -> PathBuf {
    std::env::var_os("GG_AGENT_DIR").map_or_else(
        || {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .ancestors()
                .nth(2)
                .unwrap_or(Path::new("."))
                .join("target/gg-agent")
        },
        PathBuf::from,
    )
}

/// One request line to one response line, or `None` for a notification.
///
/// Split from the loop so the protocol is testable without a pipe: every branch
/// below is reachable from a string, which is what the tests drive.
fn respond(line: &str, dir: &Path) -> Option<String> {
    let request: Value = serde_json::from_str(line).ok()?;
    let method = request.get("method")?.as_str()?;
    // No id is a notification, and the spec forbids answering one. `initialized`
    // arrives that way, so a server that replied would be talking out of turn on
    // its first exchange.
    let id = request.get("id")?.clone();
    let result = match method {
        "initialize" => json!({
            "protocolVersion": request
                .get("params")
                .and_then(|p| p.get("protocolVersion"))
                .and_then(Value::as_str)
                .unwrap_or(FALLBACK_PROTOCOL),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "gg-engine", "version": env!("CARGO_PKG_VERSION")},
        }),
        "tools/list" => json!({"tools": tools()}),
        "tools/call" => {
            let name = request
                .get("params")
                .and_then(|p| p.get("name"))
                .and_then(Value::as_str)
                .unwrap_or_default();
            match call(name, dir) {
                Some(text) => json!({"content": [{"type": "text", "text": text}]}),
                None => {
                    return Some(error(&id, -32602, &format!("unknown tool `{name}`")).to_string());
                }
            }
        }
        // `ping` is in the spec and costs one arm; anything else is refused by
        // code rather than by silence, so a client waiting on a reply is told.
        "ping" => json!({}),
        other => {
            return Some(error(&id, -32601, &format!("unknown method `{other}`")).to_string());
        }
    };
    Some(json!({"jsonrpc": "2.0", "id": id, "result": result}).to_string())
}

fn error(id: &Value, code: i32, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

/// Empty object schemas throughout: every question here is about *the* running
/// session, and a tool that took a path would invite an agent to guess one.
fn tools() -> Value {
    let no_arguments = json!({"type": "object", "properties": {}, "required": []});
    json!([
        {
            "name": "gg_session",
            "description": "The running GGEngine session: which game, which build tier, the \
                            current sim tick, the recording being written, and every recent \
                            reload. Returns the engine's own published record verbatim.",
            "inputSchema": no_arguments,
        },
        {
            "name": "gg_last_reload",
            "description": "What the last hot reload did: accepted or refused, the refusal's \
                            name if refused, the canonical state hash either side of it, which \
                            components migrated, and how long save-to-behaviour took against the \
                            budget `xtask reload` asserts (§9). Ask this before diagnosing a \
                            failed edit.",
            "inputSchema": no_arguments,
        },
    ])
}

fn call(name: &str, dir: &Path) -> Option<String> {
    enum Ask {
        Session,
        LastReload,
    }
    // The name is resolved before the record is read, and the order is the whole
    // point: an unknown tool is an error whether or not a session happens to be
    // running, and answering it with "no session" would send an agent chasing a
    // shell instead of its own typo.
    let ask = match name {
        "gg_session" => Ask::Session,
        "gg_last_reload" => Ask::LastReload,
        _ => return None,
    };
    let path = dir.join("session.json");
    // Absence is an answer, not an error: no record means no shell is running,
    // and an agent told *that* stops looking for a file it cannot make appear.
    let Ok(record) = std::fs::read_to_string(&path) else {
        return Some(format!(
            "No GGEngine session is publishing to `{}`. Start one with `cargo xtask run <demo>` \
             (a window) or a headless `gg-runtime` run; the record appears within a second of \
             the first tick.",
            path.display()
        ));
    };
    Some(match ask {
        Ask::Session => record,
        Ask::LastReload => last_reload(&record),
    })
}

/// The last seam, in prose, with the record's own numbers.
///
/// Prose rather than the raw object because this is the tool an agent reaches
/// for mid-diagnosis: the fields that matter are three of a dozen, and a reader
/// that had to know which three would be doing the engine's job for it.
fn last_reload(record: &str) -> String {
    let Ok(parsed) = serde_json::from_str::<Value>(record) else {
        return format!("The record is not readable as JSON:\n{record}");
    };
    let seams = parsed.get("seams").and_then(Value::as_array);
    let Some(seam) = seams.and_then(|s| s.last()) else {
        return "No reload has happened in this session yet — the game is still running the \
                build it started with. `gg_session` has the tick and the recording."
            .to_owned();
    };
    let field = |key: &str| {
        seam.get(key).map_or_else(
            || "unknown".to_owned(),
            // Strings unquoted rather than re-encoded: `Value::to_string` is the
            // JSON *representation*, so a Windows path would arrive with its
            // escapes doubled and be read out as `C:\\Users` in prose meant for
            // a reader. Everything else — numbers, `null`, `true` — has no
            // string form but its JSON one, which is what should be shown.
            |v| v.as_str().map_or_else(|| v.to_string(), str::to_owned),
        )
    };
    let refused = seam.get("outcome").and_then(Value::as_str) == Some("refused");
    let mut out = if refused {
        format!(
            "The last reload was REFUSED as `{}` at tick {}, and the game is still running the \
             last good build.\n\n{}\n\nNothing was adopted, so the state hash did not move \
             ({}).",
            field("refusal"),
            field("tick"),
            field("detail"),
            field("state_before"),
        )
    } else {
        format!(
            "The last reload was ACCEPTED at tick {}, restoring {} entities.\n\nState hash {} → \
             {}. Save to behaviour: {} ms against the budget `xtask reload` asserts (§9) \
             (within_budget {}).",
            field("tick"),
            field("entities"),
            field("state_before"),
            field("state_after"),
            field("save_to_swap_ms"),
            field("within_budget"),
        )
    };
    if let Some(changes) = seam.get("changes").and_then(Value::as_array)
        && !changes.is_empty()
    {
        out.push_str("\n\nComponents that moved (everything else was reused verbatim):");
        for change in changes {
            out.push_str(&format!("\n  {change}"));
        }
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn dir() -> PathBuf {
        std::env::temp_dir().join(format!("gg-mcp-test-{}", std::process::id()))
    }

    fn write_record(seams: &str) -> PathBuf {
        let dir = dir();
        std::fs::create_dir_all(&dir).expect("dir");
        std::fs::write(
            dir.join("session.json"),
            format!(
                r#"{{"game":"demo_03_reload","tier":"tier-dev","pid":1,"tick":9,
                     "recording":null,"seams":[{seams}]}}"#
            ),
        )
        .expect("write");
        dir
    }

    /// A notification has no `id`, and the spec forbids answering one.
    /// `notifications/initialized` is the first thing a client sends after
    /// `initialize`, so a server that replied would be out of turn immediately.
    #[test]
    fn a_notification_is_not_answered() {
        let request = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
        assert_eq!(respond(request, &dir()), None);
    }

    #[test]
    fn initialize_echoes_the_clients_protocol_version_and_declares_tools() {
        let request = r#"{"jsonrpc":"2.0","id":1,"method":"initialize",
                          "params":{"protocolVersion":"2024-11-05"}}"#;
        let response = respond(request, &dir()).expect("a response");
        let parsed: Value = serde_json::from_str(&response).expect("json");
        assert_eq!(parsed["result"]["protocolVersion"], "2024-11-05");
        assert!(parsed["result"]["capabilities"]["tools"].is_object());
        assert_eq!(parsed["id"], 1);
    }

    #[test]
    fn a_client_that_states_no_version_gets_the_fallback() {
        let request = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        let response = respond(request, &dir()).expect("a response");
        let parsed: Value = serde_json::from_str(&response).expect("json");
        assert_eq!(parsed["result"]["protocolVersion"], FALLBACK_PROTOCOL);
    }

    /// Every declared tool must be callable, and every callable tool declared —
    /// the same provenance argument §3 makes about widgets, applied to the
    /// surface an agent sees.
    #[test]
    fn every_declared_tool_answers_and_nothing_else_does() {
        let dir = write_record("");
        let request = r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#;
        let response = respond(request, &dir).expect("a response");
        let parsed: Value = serde_json::from_str(&response).expect("json");
        let listed = parsed["result"]["tools"].as_array().expect("tools");
        assert!(!listed.is_empty());
        for tool in listed {
            let name = tool["name"].as_str().expect("name");
            assert!(
                call(name, &dir).is_some(),
                "{name} is listed and not callable"
            );
        }
        assert!(call("gg_nonsense", &dir).is_none());
    }

    #[test]
    fn an_unknown_tool_is_a_json_rpc_error_rather_than_a_silence() {
        let request = r#"{"jsonrpc":"2.0","id":3,"method":"tools/call",
                          "params":{"name":"gg_nonsense","arguments":{}}}"#;
        let response = respond(request, &dir()).expect("a response");
        let parsed: Value = serde_json::from_str(&response).expect("json");
        assert_eq!(parsed["error"]["code"], -32602);
        assert_eq!(parsed["id"], 3);
    }

    #[test]
    fn an_unknown_method_is_refused_by_code() {
        let request = r#"{"jsonrpc":"2.0","id":4,"method":"resources/list"}"#;
        let response = respond(request, &dir()).expect("a response");
        let parsed: Value = serde_json::from_str(&response).expect("json");
        assert_eq!(parsed["error"]["code"], -32601);
    }

    /// Malformed input must not take the human's session down with it.
    #[test]
    fn a_line_that_is_not_json_is_skipped_rather_than_fatal() {
        assert_eq!(respond("not json at all", &dir()), None);
        assert_eq!(respond("{}", &dir()), None);
    }

    #[test]
    fn no_running_session_is_reported_as_an_answer_and_not_an_error() {
        let empty = std::env::temp_dir().join("gg-mcp-test-definitely-absent");
        let _ = std::fs::remove_dir_all(&empty);
        let text = call("gg_session", &empty).expect("an answer");
        assert!(text.contains("No GGEngine session"), "{text}");
    }

    #[test]
    fn a_refusal_is_summarised_by_name_and_says_the_hash_did_not_move() {
        let dir = write_record(
            r#"{"tick":19741,"outcome":"refused","refusal":"Open",
                "detail":"cannot load `a.dll`: LoadLibraryExW failed",
                "code_before":"aa","code_after":null,"state_before":"ed84","state_after":"ed84",
                "entities":0,"load_ms":null,"save_to_swap_ms":null,"within_budget":null,
                "changes":[]}"#,
        );
        let text = call("gg_last_reload", &dir).expect("an answer");
        assert!(text.contains("REFUSED as `Open`"), "{text}");
        assert!(text.contains("LoadLibraryExW"), "{text}");
        assert!(text.contains("did not move (ed84)"), "{text}");
    }

    /// The refusal a human sees most often names a Windows path, and the record
    /// carries it JSON-escaped. Rendering it with `Value::to_string` re-encodes
    /// that escape, so the agent is handed `C:\\dev` — a path that does not
    /// exist, in prose whose whole job is to name the file that failed.
    #[test]
    fn a_windows_path_is_rendered_once_escaped_and_not_twice() {
        let dir = write_record(
            r#"{"tick":1,"outcome":"refused","refusal":"Open",
                "detail":"cannot load `C:\\dev\\a.dll`: LoadLibraryExW failed",
                "code_before":"aa","code_after":null,"state_before":"ed84","state_after":"ed84",
                "entities":0,"load_ms":null,"save_to_swap_ms":null,"within_budget":null,
                "changes":[]}"#,
        );
        let text = call("gg_last_reload", &dir).expect("an answer");
        assert!(text.contains(r"C:\dev\a.dll"), "{text}");
        assert!(!text.contains(r"C:\\dev"), "{text}");
        // And the quotes `to_string` would have wrapped it in are gone with it.
        assert!(!text.contains("\"cannot load"), "{text}");
    }

    #[test]
    fn an_accepted_reload_reports_both_hashes_and_the_components_that_moved() {
        let dir = write_record(
            r#"{"tick":412,"outcome":"accepted","code_before":"aa","code_after":"bb",
                "state_before":"1111","state_after":"2222","entities":3,"load_ms":7,
                "save_to_swap_ms":180,"within_budget":true,
                "changes":[{"component":"demo03.cube","kind":"migrated",
                            "defaulted":["wobble"],"retyped":[]}]}"#,
        );
        let text = call("gg_last_reload", &dir).expect("an answer");
        assert!(text.contains("ACCEPTED at tick 412"), "{text}");
        assert!(text.contains("1111 → 2222"), "{text}");
        assert!(text.contains("180 ms"), "{text}");
        assert!(
            text.contains("demo03.cube") && text.contains("wobble"),
            "{text}"
        );
    }

    /// A session that has not reloaded is the common case, and "no seams" must
    /// read as a state of the world rather than as a missing field.
    #[test]
    fn a_session_with_no_reload_yet_says_so() {
        let dir = write_record("");
        let text = call("gg_last_reload", &dir).expect("an answer");
        assert!(text.contains("No reload has happened"), "{text}");
    }
}
