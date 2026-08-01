//! §4.10's failure artifact: one self-contained HTML page with reference,
//! actual and heatmap side by side for every scene that did not pass cleanly.
//!
//! Self-contained is the requirement, not a flourish — the page embeds its PNGs
//! as data URIs so the artifact is a single file that can be attached to a
//! report or opened from anywhere, rather than a directory that only means
//! something in place. Benign drift gets a panel too: §4.10's "recorded as such"
//! is a thing a reviewer can look at, not a line that scrolled past in a log.

use std::path::Path;

/// One scene's row in the report.
pub struct Entry {
    pub scene: String,
    /// `FAIL` / `DRIFT` — also the panel's CSS class.
    pub status: &'static str,
    pub headline: String,
    /// Label/value pairs: the gate numbers, printed whether they passed or not.
    pub numbers: Vec<(String, String)>,
    /// Label → PNG bytes, in display order.
    pub images: Vec<(&'static str, Vec<u8>)>,
}

pub fn write(path: &Path, backend: &str, entries: &[Entry]) -> anyhow::Result<()> {
    let mut html = String::new();
    html.push_str(HEAD);
    html.push_str(&format!(
        "<h1>gg-golden — {} scene(s) to review</h1><p class=\"sub\">backend <code>{}</code></p>",
        entries.len(),
        escape(backend)
    ));
    for entry in entries {
        html.push_str(&format!(
            "<section class=\"{}\"><h2><span class=\"tag\">{}</span> {}</h2><p>{}</p><dl>",
            entry.status.to_lowercase(),
            entry.status,
            escape(&entry.scene),
            escape(&entry.headline)
        ));
        for (label, value) in &entry.numbers {
            html.push_str(&format!(
                "<div><dt>{}</dt><dd>{}</dd></div>",
                escape(label),
                escape(value)
            ));
        }
        html.push_str("</dl><div class=\"images\">");
        for (label, png) in &entry.images {
            html.push_str(&format!(
                "<figure><img src=\"data:image/png;base64,{}\" alt=\"{label}\"><figcaption>{label}</figcaption></figure>",
                base64(png)
            ));
        }
        html.push_str("</div></section>");
    }
    html.push_str("</body>");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, html)?;
    Ok(())
}

const HEAD: &str = "<!doctype html><meta charset=\"utf-8\"><title>gg-golden report</title>\
<style>\
body{background:#14161a;color:#d8dee9;font:14px/1.5 ui-monospace,Consolas,monospace;margin:2rem}\
h1{font-size:1.2rem;margin:0}.sub{color:#8b949e;margin:.25rem 0 2rem}\
section{border:1px solid #2b3038;border-radius:6px;padding:1rem;margin-bottom:1.5rem}\
section.fail{border-left:4px solid #e06c75}section.drift{border-left:4px solid #d19a66}\
section.blessed{border-left:4px solid #61afef}\
h2{font-size:1rem;margin:0 0 .5rem}\
.tag{font-size:.75rem;padding:.1rem .4rem;border-radius:3px;background:#2b3038;margin-right:.5rem}\
.fail .tag{background:#e06c75;color:#14161a}.drift .tag{background:#d19a66;color:#14161a}\
.blessed .tag{background:#61afef;color:#14161a}\
dl{display:flex;flex-wrap:wrap;gap:.25rem 2rem;margin:.5rem 0 1rem}\
dt{color:#8b949e;font-size:.8rem}dd{margin:0}\
.images{display:flex;flex-wrap:wrap;gap:1rem}figure{margin:0}\
img{max-width:100%;image-rendering:pixelated;border:1px solid #2b3038;background:#000}\
figcaption{color:#8b949e;font-size:.8rem;margin-top:.25rem}\
</style><body>";

fn escape(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '&' => "&amp;".to_string(),
            '<' => "&lt;".to_string(),
            '>' => "&gt;".to_string(),
            '"' => "&quot;".to_string(),
            other => other.to_string(),
        })
        .collect()
}

/// Standard base64. Written out rather than depended on: it is twelve lines,
/// and the report is the one place in the tree that needs it.
fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        for i in 0..4 {
            // 4 output chars per 3 input bytes; a short tail pads with '='.
            if i <= chunk.len() {
                let idx = ((n >> (18 - 6 * i)) & 0x3f) as usize;
                out.push(char::from(ALPHABET[idx]));
            } else {
                out.push('=');
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn base64_matches_the_rfc_test_vectors() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foob"), "Zm9vYg==");
        assert_eq!(base64(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn the_report_names_the_failing_scene_and_carries_its_images() {
        let dir = std::env::temp_dir().join("gg-golden-report-test");
        let path = dir.join("report.html");
        let png = crate::png_io::encode(&[255, 0, 0, 255], (1, 1)).unwrap();
        write(
            &path,
            "lavapipe-windows",
            &[Entry {
                scene: "mesh".into(),
                status: "FAIL",
                headline: "structural regression".into(),
                numbers: vec![("differing pixels".into(), "812 / 256".into())],
                images: vec![("reference", png.clone()), ("actual", png)],
            }],
        )
        .unwrap();
        let html = std::fs::read_to_string(&path).unwrap();
        assert!(html.contains("mesh"), "the report must name the scene");
        assert!(html.contains("812 / 256"));
        assert_eq!(html.matches("data:image/png;base64,").count(), 2);
        std::fs::remove_dir_all(&dir).ok();
    }
}
