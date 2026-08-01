//! The `validation-suppressions.toml` mechanism (§5.4), in place from M1 while
//! the file is still empty — because the alternative is the zero-messages rule
//! getting switched off in a hurry the first time a layer is noisy, and never
//! switched back on. Every entry requires an upstream link and an expiry date;
//! an entry missing either, or past its expiry, is a CI failure of its own
//! (the tests below run in every tier).

/// One suppressed validation message, fully justified.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Suppression {
    /// The exact message id name (VUID) the layer reports.
    pub vuid: String,
    /// Why suppressing is correct.
    pub reason: String,
    /// Driver/layer version the noise was observed on.
    pub driver: String,
    /// Upstream issue link — required; a suppression nobody reported upstream
    /// is a workaround pretending to be a fact.
    pub upstream: String,
    /// Expiry date `YYYY-MM-DD` — required; suppressions are leases, not land.
    pub expires: String,
}

/// The checked-in file, compiled in so runtime filtering needs no file I/O and
/// dist builds (which compile this module out with the `validation` feature)
/// carry nothing.
pub const CHECKED_IN: &str = include_str!("../../../validation-suppressions.toml");

/// Parse the suppressions file. The format is deliberately a strict subset of
/// TOML — `[[suppression]]` blocks of `key = "value"` lines — parsed by hand
/// so the engine does not grow a TOML dependency for a file that is empty by
/// design.
pub fn parse(text: &str) -> Result<Vec<Suppression>, String> {
    let mut entries: Vec<Suppression> = Vec::new();
    let mut open = false;
    for (idx, raw) in text.lines().enumerate() {
        let line = raw.trim();
        let lineno = idx + 1;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line == "[[suppression]]" {
            entries.push(Suppression::default());
            open = true;
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(format!("line {lineno}: expected `key = \"value\"`: {line}"));
        };
        if !open {
            return Err(format!(
                "line {lineno}: key outside a [[suppression]] block"
            ));
        }
        let value = value.trim();
        let value = value
            .strip_prefix('"')
            .and_then(|v| v.strip_suffix('"'))
            .ok_or_else(|| format!("line {lineno}: value must be a quoted string"))?;
        let entry = entries
            .last_mut()
            .ok_or_else(|| format!("line {lineno}: no open entry"))?;
        match key.trim() {
            "vuid" => entry.vuid = value.to_string(),
            "reason" => entry.reason = value.to_string(),
            "driver" => entry.driver = value.to_string(),
            "upstream" => entry.upstream = value.to_string(),
            "expires" => entry.expires = value.to_string(),
            other => return Err(format!("line {lineno}: unknown key `{other}`")),
        }
    }
    Ok(entries)
}

/// Days since 1970-01-01 for a `YYYY-MM-DD` string (civil-date algorithm).
fn days_since_epoch(date: &str) -> Result<i64, String> {
    let mut parts = date.splitn(3, '-');
    let (y, m, d) = (|| {
        let y: i64 = parts.next()?.parse().ok()?;
        let m: i64 = parts.next()?.parse().ok()?;
        let d: i64 = parts.next()?.parse().ok()?;
        ((1..=12).contains(&m) && (1..=31).contains(&d)).then_some((y, m, d))
    })()
    .ok_or_else(|| format!("`{date}` is not a YYYY-MM-DD date"))?;
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Ok(era * 146097 + doe - 719468)
}

/// Validate every checked-in entry: all fields present, a real upstream link,
/// an unexpired lease. Returns the valid entries for the runtime filter.
pub fn validated(now_days: i64) -> Result<Vec<Suppression>, String> {
    let entries = parse(CHECKED_IN)?;
    for (i, e) in entries.iter().enumerate() {
        let name = if e.vuid.is_empty() {
            format!("entry {}", i + 1)
        } else {
            e.vuid.clone()
        };
        for (field, value) in [
            ("vuid", &e.vuid),
            ("reason", &e.reason),
            ("driver", &e.driver),
            ("upstream", &e.upstream),
            ("expires", &e.expires),
        ] {
            if value.is_empty() {
                return Err(format!("{name}: missing `{field}` (§5.4)"));
            }
        }
        if !e.upstream.starts_with("http") {
            return Err(format!(
                "{name}: `upstream` must be an issue link, got `{}` (§5.4)",
                e.upstream
            ));
        }
        let expiry = days_since_epoch(&e.expires)?;
        if expiry < now_days {
            return Err(format!(
                "{name}: expired {} — a suppression past its expiry is a CI failure of its own (§5.4)",
                e.expires
            ));
        }
    }
    Ok(entries)
}

/// Today in days since the epoch, from the system clock.
// Consumed by the validation-feature messenger and by tests; dist builds
// compile the whole suppressions mechanism out with the layer it serves.
#[cfg_attr(not(feature = "validation"), allow(dead_code))]
pub fn today() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| (d.as_secs() / 86_400) as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    /// The CI failure of §5.4: the checked-in file must always validate.
    #[test]
    fn checked_in_suppressions_are_valid_and_unexpired() {
        let entries = validated(today()).expect("validation-suppressions.toml is invalid");
        // Empty by design since M1; an entry appearing here is fine — it just
        // has to satisfy the lease rules the mechanism enforces above.
        for e in &entries {
            assert!(!e.vuid.is_empty());
        }
    }

    #[test]
    fn parser_accepts_a_full_entry_and_rejects_broken_ones() {
        let good = r#"
# comment
[[suppression]]
vuid = "VUID-Example-01234"
reason = "layer false positive"
driver = "lavapipe 26.1.3 / validation 1.4.x"
upstream = "https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/1"
expires = "2099-01-01"
"#;
        let entries = parse(good).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].vuid, "VUID-Example-01234");

        assert!(parse("vuid = \"x\"").is_err(), "key outside block");
        assert!(parse("[[suppression]]\nvuid = unquoted").is_err());
        assert!(parse("[[suppression]]\nnope = \"x\"").is_err());
    }

    #[test]
    fn dates_and_expiry_work() {
        assert_eq!(days_since_epoch("1970-01-01").unwrap(), 0);
        assert_eq!(days_since_epoch("2026-07-31").unwrap(), 20_665);
        assert!(days_since_epoch("not-a-date").is_err());
        assert!(days_since_epoch("2026-13-01").is_err());
        // today() sanity: after 2026-01-01, before 2100.
        assert!(today() > 20_454 && today() < 47_482);
    }
}
