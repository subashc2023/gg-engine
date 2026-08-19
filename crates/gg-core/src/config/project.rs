//! The project manifest (§6 M41): the same flat `name = value` file as the rest
//! of [`config`](super), answering the other question a run needs answered
//! before it starts — *what* to run, rather than how it is tuned.
//!
//! **Not a ninth responsibility.** §3's charter names config, and config is
//! already "the file beside the program that says how this run is set up"; this
//! is that file's other half, in the same format, with the same parser shape and
//! no new dependency. What it buys is the one thing a shipped folder needs and a
//! dev tree never did: a game that knows its own name. The host's only name for
//! a game today is the dylib's file stem, which is why a shipped window would
//! say `gg — demo_10_tetris`.
//!
//! ```text
//! # every path is relative to this file, so the folder can be moved
//! game    = tetris.dll
//! input   = input.toml
//! pack    = tetris.ggpack
//! title   = Falling Blocks
//! version = 1.0.0
//! window  = 1280x720
//! ```
//!
//! The one line a *demo's* manifest leaves out is `game`: a dylib is `.dll` on
//! one host and `lib….so` on another, so the folder's copy names it and the
//! checked-in one names everything else. `xtask ship` renders the difference.
//!
//! **A stale key is an error, not a warning** — the opposite policy to a CVar
//! line, and for the reason [`config`](super) already gives for leaving that
//! choice to the caller: a name nobody claims is a stale line in a *player's*
//! config and a corrupt folder in a manifest. Nobody hand-writes one of these;
//! `xtask ship` does, beside the executable it built.

use std::fmt;
use std::path::{Path, PathBuf};

/// What the manifest is called, beside the executable. One spelling: the shell
/// probes for it and `xtask ship` writes it, and a drift between the two would
/// assemble a folder that does not open.
pub const MANIFEST: &str = "game.ggproj";

/// A project, as read off disk. Paths are resolved against the manifest's own
/// directory by [`Project::open`], so a folder is movable and a run's working
/// directory decides nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Project {
    /// The game dylib. **Empty is a real value**, on `Args::game`'s convention:
    /// the file name is the host platform's (`.dll` against `lib….so`), so the
    /// manifest a *demo* checks in does not name one and the manifest `xtask
    /// ship` writes into a folder does. Sufficiency is the shell's question, not
    /// the format's — a manifest with no game and a command line with no
    /// `--game` fails exactly where it did before either existed.
    pub game: PathBuf,
    /// The action map (§4.7). Optional in the format and a mistake in practice:
    /// a game without one runs and is silently uncontrollable, which is why
    /// [`Project::open`] returns it and the shell warns rather than this module
    /// deciding.
    pub input: Option<PathBuf>,
    /// The pack a game draws out of (§4.6), for a game that has one.
    pub pack: Option<PathBuf>,
    /// What the window says, and what the player calls it.
    pub title: String,
    /// The directory name this game's own bytes live under (§6 M42), derived
    /// from the title unless the file says otherwise. Derived rather than
    /// required because two names for one game is how a rename loses a player's
    /// saves.
    pub slug: String,
    /// Which build of the game this is (§6 M73). Optional, and absent is the
    /// honest answer for a demo nobody has released — what it costs is that
    /// every artifact then looks alike to the person reporting a bug about one
    /// of them.
    pub version: Option<Version>,
    /// The window to open, when the game would rather not have the default.
    pub window: Option<(u32, u32)>,
    /// The picture in the taskbar and the title bar (§6 M46), for a game that
    /// has one. A [`icon`](super::icon) file — raw premultiplied-nothing RGBA
    /// behind an eight-byte header — because `deny.toml` bans every image
    /// decoder from the dist graph and a window icon is wanted before the pack
    /// that would otherwise hold it is open.
    pub icon: Option<PathBuf>,
    /// The folder the manifest was read from, and therefore the folder this
    /// project *is*. Empty from [`Project::parse`], which has no file to be
    /// beside.
    pub dir: PathBuf,
}

/// What build of a game this is (§6 M73) — the string a player quotes and the
/// four numbers Windows files it under, parsed once so the two cannot disagree.
///
/// **Four `u16`s because that is what the operating system's own record holds.**
/// `VS_FIXEDFILEINFO` is `major.minor.patch.build` in exactly that shape, and it
/// is the field Explorer's *Details* tab sorts on and an installer compares — so
/// a version this format accepted and that file could not hold would be a
/// version the player's machine disagrees with us about. Missing components are
/// zero: `1.2` is `1.2.0.0`, which is what everything that reads one assumes.
///
/// The text is kept whole and separately, so a **suffix survives**: `0.9.0-rc2`
/// is a real thing to ship and its numeric head is `0.9.0.0`. That is what every
/// tool in this space does, and the split is the reason it can — a suffix is for
/// the human and the quad is for the comparison.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Version {
    text: String,
    quad: [u16; 4],
}

impl Version {
    /// Parse `major[.minor[.patch[.build]]]` with an optional `-suffix`.
    ///
    /// # Errors
    /// [`ProjectError::Version`] for an empty component, a non-digit where a
    /// number belongs, a number past `u16::MAX`, or more than four of them.
    pub fn parse(text: &str) -> Result<Version, ProjectError> {
        let bad = || ProjectError::Version {
            text: text.to_owned(),
        };
        // The suffix is everything from the first `-`; the head is what has to
        // be numbers. Split before counting, or `1.0-rc2` reads as two fields.
        let head = text.split('-').next().unwrap_or("");
        let mut quad = [0u16; 4];
        let mut seen = 0;
        for part in head.split('.') {
            let slot = quad.get_mut(seen).ok_or_else(bad)?;
            // `parse` accepts a leading `+`, which is not a version component.
            if part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()) {
                return Err(bad());
            }
            *slot = part.parse().map_err(|_| bad())?;
            seen += 1;
        }
        // A lone suffix (`-rc2`) leaves `head` empty, which `split` reports as
        // one empty part and the loop above refuses. A trailing dot lands there
        // too. So `seen` is never zero here, and the check is for the reader.
        (seen > 0)
            .then(|| Version {
                text: text.to_owned(),
                quad,
            })
            .ok_or_else(bad)
    }

    /// As written, suffix and all — what a player quotes in a bug report.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// `major.minor.patch.build`, zero-filled.
    #[must_use]
    pub fn quad(&self) -> [u16; 4] {
        self.quad
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text)
    }
}

/// Why a manifest was not a project. Path-free: every caller has the path and
/// adds it as context, and a second copy inside the message reads as two files.
#[derive(Debug, thiserror::Error)]
pub enum ProjectError {
    /// A line that was not `name = value` and not a comment.
    #[error("line {line}: expected `name = value`, got `{text}`")]
    Shape {
        /// 1-based line number.
        line: usize,
        /// The line, trimmed.
        text: String,
    },
    /// A key this build does not answer to.
    #[error(
        "line {line}: `{name}` is not a project key — this folder was assembled by a different \
         build (§6 M41)"
    )]
    Unknown {
        /// 1-based line number.
        line: usize,
        /// The key.
        name: String,
    },
    /// One key, two answers.
    #[error("line {line}: `{name}` is set twice — a manifest with two answers has none")]
    Twice {
        /// 1-based line number of the second one.
        line: usize,
        /// The key.
        name: String,
    },
    /// A key with no default and no value.
    #[error("`{name}` is required: {why}")]
    Missing {
        /// The key.
        name: &'static str,
        /// What it is for.
        why: &'static str,
    },
    /// `window` that was not an extent.
    #[error("`window = {text}` wants `<w>x<h>` in positive pixels")]
    Window {
        /// The value as written.
        text: String,
    },
    /// `version` that the operating system's own record could not hold.
    #[error(
        "`version = {text}` wants up to four dot-separated numbers under 65536, optionally \
         followed by `-<suffix>` — it becomes the four the OS files this build under (§6 M73)"
    )]
    Version {
        /// The value as written.
        text: String,
    },
    /// The file itself did not read. Path-free like the rest: the caller has it.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl Project {
    /// Read and resolve a manifest. Every path in the result is absolute-ish —
    /// joined onto the manifest's directory — so nothing downstream consults the
    /// working directory.
    pub fn open(path: &Path) -> Result<Project, ProjectError> {
        let mut project = Project::parse(&std::fs::read_to_string(path)?)?;
        project.dir = path.parent().unwrap_or(Path::new(".")).to_owned();
        if !project.game.as_os_str().is_empty() {
            project.game = project.dir.join(&project.game);
        }
        project.input = project.input.map(|p| project.dir.join(p));
        project.pack = project.pack.map(|p| project.dir.join(p));
        project.icon = project.icon.map(|p| project.dir.join(p));
        Ok(project)
    }

    /// The manifest a run was pointed at, or the one beside the executable —
    /// which is the whole command line a double-click passes (§6 M41 item 2).
    ///
    /// Absent is not an error: it is every invocation in this tree. An explicit
    /// path that does not open *is*, because naming a file is asking for it.
    pub fn found(explicit: Option<&Path>) -> Result<Option<Project>, ProjectError> {
        let path = match explicit {
            Some(path) => path.to_owned(),
            None => match std::env::current_exe()?.parent() {
                Some(dir) => dir.join(MANIFEST),
                None => return Ok(None),
            },
        };
        if explicit.is_none() && !path.is_file() {
            return Ok(None);
        }
        Project::open(&path).map(Some)
    }

    /// Parse a manifest's text, leaving paths exactly as written.
    pub fn parse(text: &str) -> Result<Project, ProjectError> {
        let (mut game, mut input, mut pack) = (None, None, None);
        let (mut title, mut slug, mut window) = (None, None, None);
        let (mut icon, mut version) = (None, None);
        for (n, line) in text.lines().enumerate() {
            let line = line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let (name, value) = line.split_once('=').ok_or_else(|| ProjectError::Shape {
                line: n + 1,
                text: line.to_owned(),
            })?;
            let (name, value) = (name.trim(), value.trim().to_owned());
            let taken = match name {
                "game" => game.replace(value).is_some(),
                "input" => input.replace(value).is_some(),
                "pack" => pack.replace(value).is_some(),
                "title" => title.replace(value).is_some(),
                "slug" => slug.replace(value).is_some(),
                "window" => window.replace(value).is_some(),
                "icon" => icon.replace(value).is_some(),
                "version" => version.replace(value).is_some(),
                _ => {
                    return Err(ProjectError::Unknown {
                        line: n + 1,
                        name: name.to_owned(),
                    });
                }
            };
            if taken {
                return Err(ProjectError::Twice {
                    line: n + 1,
                    name: name.to_owned(),
                });
            }
        }
        let title = title.ok_or(ProjectError::Missing {
            name: "title",
            why: "it is what the window says and what the player calls the game",
        })?;
        let slug = match slug {
            Some(slug) => slug,
            None => slugify(&title).ok_or(ProjectError::Missing {
                name: "slug",
                why: "the title has no letters or digits to derive a directory name from",
            })?,
        };
        Ok(Project {
            game: PathBuf::from(game.unwrap_or_default()),
            input: input.map(PathBuf::from),
            pack: pack.map(PathBuf::from),
            title,
            slug,
            version: version.map(|text| Version::parse(&text)).transpose()?,
            window: window.map(|text| parse_extent(&text)).transpose()?,
            icon: icon.map(PathBuf::from),
            dir: PathBuf::new(),
        })
    }

    /// Where this game's own bytes belong: saves, its log, and the warm pipeline
    /// cache. `None` when the OS did not say — a run then keeps everything
    /// beside itself, which is wrong in a Program Files install and right in the
    /// unzipped folder itch actually hands out.
    ///
    /// This reads the environment, which [`config`](super) refuses to do, and the
    /// two are not in tension: a `GG_*` variable that silently retunes an engine
    /// is a bug report that does not reproduce, while `%LOCALAPPDATA%` is the
    /// operating system answering the only question it can answer.
    #[must_use]
    pub fn data_dir(&self) -> Option<PathBuf> {
        // Local rather than roaming on Windows: a pipeline cache is one
        // machine's, and roaming one across a domain login would be worse than
        // not having it.
        let base = if cfg!(windows) {
            std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
        } else {
            std::env::var_os("XDG_DATA_HOME")
                .map(PathBuf::from)
                .or_else(|| {
                    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share"))
                })
        }?;
        Some(base.join(&self.slug))
    }
}

/// A title as a directory name: ASCII lowercase, one `-` for every run of
/// anything else, no leading or trailing `-`. `None` when nothing survives.
fn slugify(title: &str) -> Option<String> {
    let mut out = String::with_capacity(title.len());
    for c in title.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    let slug = out.trim_matches('-');
    (!slug.is_empty()).then(|| slug.to_owned())
}

/// `<w>x<h>` in positive pixels. Here rather than in the shell because the
/// manifest and `--editor-extent` are the same syntax, and two parses of one
/// syntax drift.
pub fn parse_extent(text: &str) -> Result<(u32, u32), ProjectError> {
    let bad = || ProjectError::Window {
        text: text.to_owned(),
    };
    let (w, h) = text.split_once(['x', 'X']).ok_or_else(bad)?;
    let (w, h) = (
        w.trim().parse().map_err(|_| bad())?,
        h.trim().parse().map_err(|_| bad())?,
    );
    // Zero is refused here rather than downstream: a zero-sized window is a
    // swapchain error three layers away from the line that caused it.
    (w > 0 && h > 0).then_some((w, h)).ok_or_else(bad)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const WHOLE: &str = "\
# a folder as shipped
game   = tetris.dll
input  = input.toml
pack   = tetris.ggpack
title  = Falling Blocks
window = 1280x720
";

    #[test]
    fn a_manifest_is_the_argv_a_shipped_folder_does_not_have() {
        let project = Project::parse(WHOLE).unwrap();
        assert_eq!(project.game, PathBuf::from("tetris.dll"));
        assert_eq!(project.input, Some(PathBuf::from("input.toml")));
        assert_eq!(project.pack, Some(PathBuf::from("tetris.ggpack")));
        assert_eq!(project.title, "Falling Blocks");
        assert_eq!(project.window, Some((1280, 720)));
        // Derived, because a rename that moved a player's saves would be the
        // second name for one game.
        assert_eq!(project.slug, "falling-blocks");
    }

    #[test]
    fn only_the_name_is_required() {
        let project = Project::parse("game = g.dll\ntitle = G").unwrap();
        assert_eq!(
            (project.input, project.pack, project.window),
            (None, None, None)
        );
        assert!(matches!(
            Project::parse("game = g.dll"),
            Err(ProjectError::Missing { name: "title", .. })
        ));
        // A demo's own manifest names no dylib: `.dll` against `lib….so` is the
        // shipping host's business, and `xtask ship` writes the line.
        assert!(
            Project::parse("title = G")
                .unwrap()
                .game
                .as_os_str()
                .is_empty()
        );
    }

    #[test]
    fn a_key_this_build_does_not_answer_to_is_an_error_and_not_a_warning() {
        // The opposite of a CVar line's policy, deliberately: nobody hand-writes
        // a manifest, so an unknown key is a folder assembled by another build.
        assert!(matches!(
            Project::parse("game = g.dll\ntitle = G\nresolution = 4k"),
            Err(ProjectError::Unknown { line: 3, .. })
        ));
        assert!(matches!(
            Project::parse("game = g.dll\ntitle = G\ngame = h.dll"),
            Err(ProjectError::Twice { line: 3, .. })
        ));
        assert!(matches!(
            Project::parse("game = g.dll\ntitle = G\nnonsense"),
            Err(ProjectError::Shape { line: 3, .. })
        ));
    }

    #[test]
    fn a_window_is_two_positive_numbers_or_it_is_nothing() {
        assert_eq!(
            Project::parse("game=g\ntitle=G\nwindow = 800 X 600")
                .unwrap()
                .window,
            Some((800, 600))
        );
        for bad in ["800", "800x0", "0x600", "x", "-1x8", "800x600x1"] {
            assert!(
                matches!(
                    Project::parse(&format!("game=g\ntitle=G\nwindow={bad}")),
                    Err(ProjectError::Window { .. })
                ),
                "`{bad}` should not be an extent"
            );
        }
    }

    /// The quad is what the OS files a build under, so the parse is graded on it
    /// rather than on the string — and on the two shapes a release actually has:
    /// short (`1.0` is `1.0.0.0`) and suffixed (a candidate is still a build).
    #[test]
    fn a_version_is_four_numbers_and_whatever_a_human_wrote_around_them() {
        let of = |v: &str| Version::parse(v).unwrap();
        assert_eq!(of("1.0.0").quad(), [1, 0, 0, 0]);
        assert_eq!(of("1.0").quad(), [1, 0, 0, 0]);
        assert_eq!(of("7").quad(), [7, 0, 0, 0]);
        assert_eq!(of("2.11.3.409").quad(), [2, 11, 3, 409]);
        assert_eq!(of("65535.65535").quad(), [65535, 65535, 0, 0]);
        // The suffix rides on the text and never on the numbers, which is what
        // lets a release candidate be a version at all.
        assert_eq!(of("0.9.0-rc2").quad(), [0, 9, 0, 0]);
        assert_eq!(of("0.9.0-rc2").text(), "0.9.0-rc2");
        assert_eq!(of("1.0.0").to_string(), "1.0.0");
        for bad in [
            "",
            "1.0.0.0.0",
            "1..0",
            "1.",
            ".1",
            "v1.0",
            "1.0a",
            "65536",
            "1.0 .0",
            "+1",
            "-rc2",
            "1.-2",
        ] {
            assert!(
                matches!(Version::parse(bad), Err(ProjectError::Version { .. })),
                "`{bad}` should not be a version"
            );
        }
        assert_eq!(
            Project::parse("title=G\nversion = 3.2.1").unwrap().version,
            Some(of("3.2.1"))
        );
        assert_eq!(Project::parse("title=G").unwrap().version, None);
        assert!(matches!(
            Project::parse("title=G\nversion = latest"),
            Err(ProjectError::Version { .. })
        ));
    }

    #[test]
    fn a_slug_is_a_directory_name_and_survives_a_title_that_is_not() {
        assert_eq!(slugify("Falling Blocks").as_deref(), Some("falling-blocks"));
        assert_eq!(
            slugify("  —Gravity's  Fault!  ").as_deref(),
            Some("gravity-s-fault")
        );
        assert_eq!(slugify("GG"), Some("gg".to_owned()));
        assert_eq!(slugify("!!!"), None);
        // Spelled in the file when the derivation is not what a player's old
        // save directory is already called.
        assert_eq!(
            Project::parse("game=g\ntitle=Falling Blocks\nslug=tetris")
                .unwrap()
                .slug,
            "tetris"
        );
    }

    #[test]
    fn paths_resolve_against_the_manifest_so_a_folder_can_be_moved() {
        let dir = std::env::temp_dir().join(format!("gg-project-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(MANIFEST);
        std::fs::write(&path, WHOLE).unwrap();
        let project = Project::open(&path).unwrap();
        assert_eq!(project.game, dir.join("tetris.dll"));
        assert_eq!(project.input, Some(dir.join("input.toml")));
        assert_eq!(project.pack, Some(dir.join("tetris.ggpack")));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
