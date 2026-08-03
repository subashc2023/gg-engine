//! The reload host's checks and staging (§4.2.2).
//!
//! What is proven here is everything that does not need a real game dylib: the
//! order of the compatibility checks, the fingerprint being a live value, the
//! leak budget, and the watch → settle → stage → load path end to end (loading a
//! file that is not a dylib, which must be a named error rather than a crash).
//! Loading a *valid* dylib is demo 03's job — it is the first artifact in the
//! workspace built as one, and a fixture crate here would be that artifact with
//! a worse name.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use gg_abi::{
    AbiInfo, AbiStatus, ArchetypeMatch, BOUNDARY_FINGERPRINT, Entity, HOST_API_VERSION, HostApiV1,
    QueryDesc, WorldHandle,
};
use gg_core::reload::rejuvenate::Handoff;
use gg_core::reload::{self, GameLib, LeakBudget, ReloadError};

// A host table that answers nothing. Every test below fails before any of these
// could be called; they exist because `GameLib::load` takes a `&'static` table
// and a null one would be a different test than the one being run.
unsafe extern "C" fn spawn(_: *mut WorldHandle) -> Entity {
    Entity::new(0, 0)
}
unsafe extern "C" fn despawn(_: *mut WorldHandle, _: Entity) -> bool {
    false
}
unsafe extern "C" fn is_alive(_: *const WorldHandle, _: Entity) -> bool {
    false
}
unsafe extern "C" fn insert(
    _: *mut WorldHandle,
    _: Entity,
    _: u64,
    _: *const u8,
    _: u32,
) -> AbiStatus {
    AbiStatus::BadRequest
}
unsafe extern "C" fn remove(_: *mut WorldHandle, _: Entity, _: u64) -> bool {
    false
}
unsafe extern "C" fn get(_: *mut WorldHandle, _: Entity, _: u64) -> *mut u8 {
    core::ptr::null_mut()
}
unsafe extern "C" fn query(
    _: *mut WorldHandle,
    _: *const QueryDesc,
    _: u32,
    _: *mut ArchetypeMatch,
    _: u32,
    _: *mut u32,
) -> AbiStatus {
    AbiStatus::BadRequest
}
unsafe extern "C" fn log(_: u32, _: *const u8, _: u32) {}

static HOST_API: HostApiV1 = HostApiV1 {
    version: HOST_API_VERSION,
    reserved: 0,
    spawn,
    despawn,
    is_alive,
    insert,
    remove,
    get,
    query,
    log,
};

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("gg-core-reload").join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn this_hosts_own_description_is_the_one_that_verifies() {
    let info = AbiInfo {
        host_api_version: HOST_API_VERSION,
        fingerprint: BOUNDARY_FINGERPRINT,
    };
    assert!(reload::verify(&info, Path::new("self")).is_ok());
}

#[test]
fn a_wrong_version_is_refused_by_number_before_anything_else_is_consulted() {
    // Both checks would fail. The version must be the one reported, because it
    // is what makes the rest of the struct interpretable at all — a fingerprint
    // read under the wrong layout is a fingerprint read from the wrong bytes.
    let info = AbiInfo {
        host_api_version: HOST_API_VERSION + 1,
        fingerprint: [0; 32],
    };
    let err = reload::verify(&info, Path::new("game.dll")).unwrap_err();
    assert!(
        matches!(err, ReloadError::HostApiMismatch { found, expected, .. }
            if found == HOST_API_VERSION + 1 && expected == HOST_API_VERSION),
        "{err}"
    );
    assert!(err.to_string().contains("game.dll"), "names the artifact");
}

#[test]
fn a_modified_boundary_behind_an_unchanged_version_is_refused_by_fingerprint() {
    let mut fingerprint = BOUNDARY_FINGERPRINT;
    fingerprint[0] ^= 0xff;
    let info = AbiInfo {
        host_api_version: HOST_API_VERSION,
        fingerprint,
    };
    let err = reload::verify(&info, Path::new("game.dll")).unwrap_err();
    assert!(matches!(err, ReloadError::Fingerprint { .. }), "{err}");
    // The message carries both sides: the reader is holding two builds and needs
    // to know which one moved.
    let message = err.to_string();
    assert!(message.contains(gg_abi::BOUNDARY_FINGERPRINT_HEX));
}

#[test]
fn the_boundary_fingerprint_is_a_live_hash_and_not_a_placeholder() {
    assert_ne!(BOUNDARY_FINGERPRINT, [0; 32], "build.rs never ran");
    assert_eq!(gg_abi::BOUNDARY_FINGERPRINT_HEX.len(), 64);
    let rendered: String = BOUNDARY_FINGERPRINT
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(rendered, gg_abi::BOUNDARY_FINGERPRINT_HEX);
}

#[test]
fn a_file_that_is_not_a_dylib_is_refused_by_name() {
    // The reload path's inputs come from a compiler that may have failed halfway.
    // A truncated or empty artifact has to be an error the host can print, not a
    // crash inside the platform loader.
    let dir = scratch("not-a-dylib");
    let path = dir.join("game.dll");
    std::fs::write(&path, b"this is not a PE image").unwrap();

    // SAFETY: the path is a file this test just wrote; loading it fails in the
    // platform loader before any of its bytes are executed.
    let err = unsafe { GameLib::load(&path, &HOST_API) }.unwrap_err();
    assert!(matches!(err, ReloadError::Open { .. }), "{err}");
    assert!(err.to_string().contains("game.dll"));
}

#[test]
fn a_missing_artifact_is_an_error_rather_than_a_wait() {
    // Reported as *unreadable* rather than as a load failure: the read is what
    // produces the leak budget's charge and the replay segment's code hash
    // (§4.2.2, §4.7), so it fails on its own terms instead of defaulting to zero
    // bytes and a constant hash and letting `libloading` speak for it.
    let dir = scratch("missing");
    // SAFETY: nothing at this path to execute.
    let err = unsafe { GameLib::load(&dir.join("absent.dll"), &HOST_API) }.unwrap_err();
    assert!(matches!(err, ReloadError::Unreadable { .. }), "{err}");
    assert!(err.to_string().contains("absent.dll"), "{err}");
}

#[test]
fn a_refusal_charges_the_budget_exactly_when_it_mapped_something() {
    // Nothing was mapped, so nothing leaked: both of these fail before the
    // platform loader ran an initializer.
    let dir = scratch("leak-accounting");
    let path = dir.join("game.dll");
    std::fs::write(&path, b"not a PE image").unwrap();
    // SAFETY: the path is a file this test just wrote; the loader rejects it
    // before any byte of it is executed.
    let refused = unsafe { GameLib::load(&path, &HOST_API) }.unwrap_err();
    assert_eq!(
        refused.leaked_bytes(),
        0,
        "the image never mapped: {refused}"
    );
    // SAFETY: nothing at this path to execute.
    let absent = unsafe { GameLib::load(&dir.join("absent.dll"), &HOST_API) }.unwrap_err();
    assert_eq!(absent.leaked_bytes(), 0);

    // Past the map, every refusal leaks the whole image — `ManuallyDrop` is
    // applied at `Library::new`, not at success — and the budget has to see it
    // or a session of refused saves never rejuvenates. Hand-built because
    // reaching the post-map refusals needs a real dylib, which is demo 03's and
    // `xtask reload`'s job, not this file's.
    let after_map = ReloadError::Fingerprint {
        expected: gg_abi::BOUNDARY_FINGERPRINT_HEX,
        found: String::new(),
        path,
        leaked: 4_000,
    };
    assert_eq!(after_map.leaked_bytes(), 4_000);
    let mut budget = LeakBudget::new(4_000);
    budget.charge(after_map.leaked_bytes());
    assert!(
        budget.exhausted(),
        "a refused reload is still a leaked reload"
    );
}

#[test]
fn the_leak_budget_counts_bytes_because_a_count_would_be_the_wrong_unit() {
    // §4.2.2: a debug dylib and a dist one differ by an order of magnitude, so
    // "ten reloads" is not a memory bound and "a gigabyte" is.
    let mut budget = LeakBudget::new(10_000);
    budget.charge(4_000);
    budget.charge(4_000);
    assert!(!budget.exhausted());
    assert_eq!(budget.spent(), 8_000);
    budget.charge(4_000);
    assert!(
        budget.exhausted(),
        "crossing the budget is what triggers rejuvenation"
    );

    // A zero budget is how the forced-rejuvenation exit criterion gets exercised
    // on demand instead of after a thousand edits.
    let mut forced = LeakBudget::new(0);
    assert!(forced.exhausted());
    forced.charge(1);
    assert!(forced.exhausted());
}

#[test]
fn a_handoff_carries_the_clock_beside_the_world() {
    // The tick is the *host's* state, so it rides outside the world image: a
    // game that reads `tick` would see time run backwards if a restart resumed
    // at zero (§4.2.2).
    let staged = Handoff {
        tick: 7_777,
        world: b"opaque here: gg-core sits below the ECS".to_vec(),
    };
    let resumed = Handoff::decode(&staged.encode()).unwrap();
    assert_eq!(resumed.tick, staged.tick);
    assert_eq!(resumed.world, staged.world);
}

#[test]
fn a_handoff_that_is_not_this_builds_is_refused_rather_than_resumed() {
    // A stale file in the temp directory is a session that would resume somebody
    // else's world, so every one of these is an error and not a default.
    assert!(Handoff::decode(b"").is_err());
    assert!(Handoff::decode(b"short").is_err());
    assert!(Handoff::decode(b"NOPE____________").is_err());

    let mut wrong_format = Handoff {
        tick: 1,
        world: Vec::new(),
    }
    .encode();
    wrong_format[4] = 9;
    let message = Handoff::decode(&wrong_format).unwrap_err().to_string();
    assert!(message.contains("format 9"), "names the format: {message}");
}

#[cfg(feature = "hot-reload")]
mod watching {
    use super::*;
    use gg_core::reload::watch::Watch;

    /// Poll until something happens or the deadline passes. The watcher's
    /// settling rule needs two polls, and the OS decides when the event lands.
    fn spin(watch: &mut Watch) -> Option<Result<gg_core::reload::watch::Reloaded, ReloadError>> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while std::time::Instant::now() < deadline {
            // SAFETY: the watched artifact in these tests is a file the test
            // wrote; `HOST_API` is a `static` and outlives every call.
            if let Some(outcome) = unsafe { watch.poll(&HOST_API) } {
                return Some(outcome);
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        None
    }

    #[test]
    fn nothing_happens_until_something_is_saved() {
        let dir = scratch("quiet");
        let source = dir.join("game.dll");
        std::fs::write(&source, b"placeholder").unwrap();
        let mut watch = Watch::new(&source, &dir.join("staging")).unwrap();

        assert!(!watch.is_pending());
        for _ in 0..20 {
            // SAFETY: as in `spin` — the artifact is this test's own file.
            assert!(unsafe { watch.poll(&HOST_API) }.is_none());
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    #[test]
    fn the_first_load_waits_for_the_artifact_and_then_stops_waiting() {
        // `block_until_ready` is the startup path — there is no frame loop yet
        // to be polled from, so the wait is the watcher's own. Both ends of that
        // are the contract: a present artifact costs the two polls its settling
        // takes and not the staging patience, and an absent one ends as a named
        // error rather than as a wait with no end.
        let dir = scratch("first-load");
        let source = dir.join("game.dll");
        std::fs::write(&source, b"present, and still not a PE image").unwrap();
        let mut watch = Watch::new(&source, &dir.join("staging")).unwrap();
        let started = std::time::Instant::now();
        // SAFETY: as in `spin` — the artifact is this test's own file.
        let err = unsafe { watch.block_until_ready(&HOST_API) }.unwrap_err();
        assert!(matches!(err, ReloadError::Open { .. }), "{err}");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(1),
            "a settled artifact must not be charged the staging patience: {:?}",
            started.elapsed()
        );

        let dir = scratch("first-load-absent");
        let mut watch = Watch::new(&dir.join("never-built.dll"), &dir.join("staging")).unwrap();
        // SAFETY: there is nothing at this path to execute.
        let err = unsafe { watch.block_until_ready(&HOST_API) }.unwrap_err();
        assert!(matches!(err, ReloadError::Staging { .. }), "{err}");
        assert!(err.to_string().contains("never-built.dll"), "{err}");
    }

    #[test]
    fn a_save_is_staged_and_loaded_and_reports_by_name_when_it_is_not_a_dylib() {
        // The whole dev path in one test: watch fires, the artifact settles, it
        // is copied aside rather than loaded in place, and the load's verdict
        // comes back attached to the *staged* copy.
        let dir = scratch("save");
        let staging = dir.join("staging");
        let source = dir.join("game.dll");
        std::fs::write(&source, b"first").unwrap();
        let mut watch = Watch::new(&source, &staging).unwrap();

        std::fs::write(&source, b"rebuilt, and still not a PE image").unwrap();
        let outcome = spin(&mut watch).expect("the watcher never fired");
        let err = outcome.unwrap_err();
        assert!(matches!(err, ReloadError::Open { .. }), "{err}");

        let staged: Vec<_> = std::fs::read_dir(&staging)
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            staged,
            ["game-1.dll"],
            "loaded a numbered copy, never the artifact cargo is about to overwrite"
        );
        assert!(!watch.is_pending(), "the pending save was consumed");
    }

    #[test]
    fn a_requested_reload_needs_no_file_event() {
        // The manual reload key, and the first load at startup — both want the
        // staging path without waiting for a save that already happened.
        let dir = scratch("requested");
        let staging = dir.join("staging");
        let source = dir.join("game.dll");
        std::fs::write(&source, b"whatever").unwrap();
        let mut watch = Watch::new(&source, &staging).unwrap();

        watch.request();
        assert!(watch.is_pending());
        let outcome = spin(&mut watch).expect("a requested reload must resolve");
        assert!(outcome.is_err(), "still not a dylib");
        assert!(staging.join("game-1.dll").exists());
    }

    #[test]
    fn a_half_written_artifact_is_never_the_one_that_gets_staged() {
        // The settle rule's whole job, against a poller running as fast as the
        // machine allows — which is what a `--frames` headless run is, and how
        // this was found: a rule that asked for "the same stat twice" was
        // satisfied microseconds apart, mid-write, and staged a truncated dylib
        // that could only be reported as `file too short`.
        use std::io::Write as _;

        let dir = scratch("partial");
        let staging = dir.join("staging");
        let source = dir.join("game.dll");
        std::fs::write(&source, b"the previous build").unwrap();
        let mut watch = Watch::new(&source, &staging).unwrap();

        let whole = vec![b'x'; 512 * 1024];
        let writer = {
            let (source, whole) = (source.clone(), whole.clone());
            std::thread::spawn(move || {
                let mut file = std::fs::File::create(&source).unwrap();
                for chunk in whole.chunks(64 * 1024) {
                    file.write_all(chunk).unwrap();
                    file.flush().unwrap();
                    std::thread::sleep(std::time::Duration::from_millis(4));
                }
            })
        };

        // No sleep in this loop: a tick boundary in a bounded run is not paced.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let mut outcome = None;
        while std::time::Instant::now() < deadline && outcome.is_none() {
            // SAFETY: as in `spin` — the artifact is this test's own file.
            outcome = unsafe { watch.poll(&HOST_API) };
        }
        writer.join().unwrap();
        assert!(outcome.is_some(), "the watcher never resolved");

        let staged = std::fs::read(staging.join("game-1.dll")).expect("nothing was staged");
        assert_eq!(
            staged.len(),
            whole.len(),
            "staged {} bytes of a {}-byte artifact — a partial write was believed",
            staged.len(),
            whole.len()
        );
    }
}
