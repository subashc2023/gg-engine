//! The capture trigger (§4.8) — RenderDoc's in-application API.
//!
//! Three front doors onto one switch, the way every knob here works: the
//! `d.capture` CVar (config file, `--set`, console), F11 through the overlay,
//! and `gg-golden capture` for the frames no window ever shows. All three set a
//! frame count; the next frame rendered consumes one.
//!
//! **The module is found, never loaded.** RenderDoc hooks Vulkan by being in the
//! process before the instance is created, so a `renderdoc.dll` this crate
//! loaded itself would hook nothing and then report captures of nothing.
//! Absent means absent: every entry point below is inert, and an arming that
//! cannot fire says so and disarms rather than waiting for a frame that will
//! never be recorded.
//!
//! In-application capture rather than RenderDoc's own hotkey because the hotkey
//! needs a Present to hang itself on, and the frames worth capturing here
//! include the ones from a run that never presents (§1.5).

use std::ffi::{CStr, CString, c_char, c_void};
use std::path::PathBuf;
use std::sync::OnceLock;

use gg_core::{CVar, CVarError, cvar};

/// Frames still owed to an armed capture; `0` is idle. An `int` and not a
/// `bool` because "the next three frames" is what a stutter needs and a flag
/// cannot say.
pub static CAPTURE: CVar = CVar::new_int("d.capture", 0, "capture the next N frames to RenderDoc");

/// Register this module's knob.
///
/// # Errors
///
/// A name already taken.
pub fn register() -> Result<(), CVarError> {
    cvar::register(&CAPTURE)
}

/// `eRENDERDOC_API_Version_1_1_2` — the oldest version carrying everything
/// called here. Asking for the oldest sufficient one is the documented
/// contract: the table only ever grows, so every newer RenderDoc still answers,
/// and every field below is one a 1.1.2 host promises to have filled.
const API_VERSION: u32 = 10102;

/// `RENDERDOC_API_1_1_2`, in declaration order — **the order is the ABI**. The
/// slots we never call are `usize` rather than typed pointers: it keeps the
/// table `Sync` without an `unsafe impl` asserting something about RenderDoc's
/// memory, and it makes calling one by mistake impossible instead of merely
/// unlikely.
#[repr(C)]
struct Api {
    get_api_version: unsafe extern "C" fn(*mut i32, *mut i32, *mut i32),
    set_capture_option_u32: usize,
    set_capture_option_f32: usize,
    get_capture_option_u32: usize,
    get_capture_option_f32: usize,
    set_focus_toggle_keys: usize,
    set_capture_keys: usize,
    get_overlay_bits: usize,
    mask_overlay_bits: usize,
    remove_hooks: usize,
    unload_crash_handler: usize,
    set_capture_file_path_template: unsafe extern "C" fn(*const c_char),
    get_capture_file_path_template: usize,
    get_num_captures: unsafe extern "C" fn() -> u32,
    get_capture: unsafe extern "C" fn(u32, *mut c_char, *mut u32, *mut u64) -> u32,
    trigger_capture: usize,
    is_target_control_connected: usize,
    launch_replay_ui: usize,
    set_active_window: usize,
    start_frame_capture: unsafe extern "C" fn(*mut c_void, *mut c_void),
    is_frame_capturing: usize,
    end_frame_capture: unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32,
    trigger_multi_frame_capture: usize,
}

type GetApi = unsafe extern "C" fn(u32, *mut *mut Api) -> i32;

/// The table's address, or `0` for "not under RenderDoc". A `usize` so the cell
/// is `Sync` on its own terms.
static API: OnceLock<usize> = OnceLock::new();

fn api() -> Option<&'static Api> {
    let table = *API.get_or_init(|| resolve().unwrap_or(0));
    if table == 0 {
        return None;
    }
    // SAFETY: only `resolve` writes this cell, and only with a table RenderDoc
    // filled through its own `RENDERDOC_GetAPI`. The table lives inside the
    // injected module, which is never unloaded while this process runs.
    Some(unsafe { &*(table as *const Api) })
}

fn resolve() -> Option<usize> {
    let get_api = entry_point()?;
    let mut table: *mut Api = std::ptr::null_mut();
    // SAFETY: RenderDoc's own exported entry point, with a valid out-parameter.
    // A version it cannot serve is a non-1 return with `table` left untouched.
    let served = unsafe { get_api(API_VERSION, &mut table) };
    if served != 1 || table.is_null() {
        tracing::warn!(
            wanted = API_VERSION,
            "renderdoc is loaded but refused this API version"
        );
        return None;
    }
    let (mut major, mut minor, mut patch) = (0, 0, 0);
    // SAFETY: `table` is RenderDoc's, filled by the call above.
    unsafe { ((*table).get_api_version)(&mut major, &mut minor, &mut patch) };
    tracing::info!(
        api = format_args!("{major}.{minor}.{patch}"),
        "renderdoc attached — F11 or `d.capture 1` records a frame"
    );
    Some(table as usize)
}

#[cfg(windows)]
fn entry_point() -> Option<GetApi> {
    // Already-loaded only, never `LoadLibrary` (see the module docs).
    let module = libloading::os::windows::Library::open_already_loaded("renderdoc.dll").ok()?;
    // SAFETY: the published name and signature of RenderDoc's entry point.
    // `forget` keeps our reference from being the one that drops the module out
    // from under the pointer we just took.
    let entry = *unsafe { module.get::<GetApi>(b"RENDERDOC_GetAPI\0") }.ok()?;
    std::mem::forget(module);
    Some(entry)
}

#[cfg(not(windows))]
fn entry_point() -> Option<GetApi> {
    // The already-loaded global namespace, which is where `LD_PRELOAD` puts
    // `librenderdoc.so`. Never a `dlopen` of our own (see the module docs).
    let module = libloading::os::unix::Library::this();
    // SAFETY: as the Windows arm — the published name and signature, and a
    // module that outlives every pointer taken out of it.
    let entry = *unsafe { module.get::<GetApi>(b"RENDERDOC_GetAPI\0") }.ok()?;
    std::mem::forget(module);
    Some(entry)
}

/// Whether this process is running under RenderDoc. Everything else here is
/// inert when it is not.
pub fn available() -> bool {
    api().is_some()
}

/// Captures RenderDoc has written this session.
pub fn count() -> u32 {
    match api() {
        // SAFETY: RenderDoc's own counter; no arguments to get wrong.
        Some(api) => unsafe { (api.get_num_captures)() },
        None => 0,
    }
}

/// Arm the next `frames` rendered frames. Negative is idle, not an error: this
/// is a knob, and the console's parse is the only place a value is refused.
pub fn request(frames: i64) {
    CAPTURE.set_int(frames.max(0));
}

/// Name the next captures. RenderDoc appends `_frame<N>.rdc` to `stem`, so a
/// stem is what this takes rather than a path. Inert without RenderDoc — there
/// is nothing to name.
pub fn set_path_template(stem: &str) {
    let (Some(api), Ok(text)) = (api(), CString::new(stem)) else {
        return;
    };
    // SAFETY: `text` outlives the call, and RenderDoc copies the string.
    unsafe { (api.set_capture_file_path_template)(text.as_ptr()) };
}

/// The newest capture RenderDoc has written, if any.
pub fn latest() -> Option<PathBuf> {
    let api = api()?;
    let index = count().checked_sub(1)?;
    let mut length = 0u32;
    // SAFETY: the documented two-call form — a null buffer asks for the length,
    // terminator included. `index` is in range: it came from `count`.
    let known = unsafe { (api.get_capture)(index, std::ptr::null_mut(), &mut length, null_u64()) };
    if known != 1 || length == 0 {
        return None;
    }
    let mut buffer = vec![0u8; length as usize];
    // SAFETY: `buffer` is exactly the `length` bytes RenderDoc just asked for.
    let written =
        unsafe { (api.get_capture)(index, buffer.as_mut_ptr().cast(), &mut length, null_u64()) };
    if written != 1 {
        return None;
    }
    Some(PathBuf::from(
        CStr::from_bytes_until_nul(&buffer).ok()?.to_str().ok()?,
    ))
}

/// The timestamp out-parameter, which nothing here reads. A named function so
/// the two call sites above stay one line each.
fn null_u64() -> *mut u64 {
    std::ptr::null_mut()
}

/// Bracket one frame's GPU work: a capture opens if one is armed and closes
/// when the returned guard drops.
///
/// Drop it **before** the device — `EndFrameCapture` runs against the device
/// the capture opened on. A frame nobody armed costs one relaxed load.
#[must_use = "the capture ends when the guard drops — bind it, do not discard it"]
pub fn frame() -> Frame {
    let owed = CAPTURE.int();
    if owed <= 0 {
        return Frame { open: false };
    }
    let Some(api) = api() else {
        // Disarmed rather than left pending: an arming that quietly never fires
        // is worse than none, and no later frame would change the reason.
        CAPTURE.set_int(0);
        tracing::warn!(
            "`d.capture` is set but this process is not running under RenderDoc — launch it from \
             the RenderDoc UI or under `renderdoccmd capture`"
        );
        return Frame { open: false };
    };
    CAPTURE.set_int(owed - 1);
    // SAFETY: a null device and window is RenderDoc's documented "the one
    // device in this process", which is the only shape we have — a host holds
    // one `Gpu` (§4.3).
    unsafe { (api.start_frame_capture)(std::ptr::null_mut(), std::ptr::null_mut()) };
    Frame { open: true }
}

/// One frame's capture, open until this drops.
pub struct Frame {
    open: bool,
}

impl Drop for Frame {
    fn drop(&mut self) {
        if !self.open {
            return;
        }
        let Some(api) = api() else { return };
        // SAFETY: paired with the `start_frame_capture` that set `open`, and the
        // same null device the capture was opened against.
        let written =
            unsafe { (api.end_frame_capture)(std::ptr::null_mut(), std::ptr::null_mut()) };
        match (written, latest()) {
            (1, Some(path)) => tracing::info!(path = %path.display(), "renderdoc capture written"),
            (1, None) => tracing::info!("renderdoc capture written"),
            // RenderDoc refuses a capture that saw no work, which here means the
            // guard bracketed the wrong span rather than that the frame was empty.
            _ => {
                tracing::warn!("renderdoc discarded the capture — no work was submitted inside it")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CI never runs under RenderDoc, so what is provable here is the *absent*
    /// path — which is the path every automated tier and every shipped build
    /// takes. The live one is proven by hand: the shell under the RenderDoc UI
    /// for the windowed half, `gg-golden capture` for the windowless one.
    #[test]
    fn an_arming_that_cannot_fire_disarms_itself() {
        assert!(!available(), "CI does not run under RenderDoc");
        request(2);
        drop(frame());
        assert_eq!(
            CAPTURE.int(),
            0,
            "an armed capture must not outlive its chance"
        );
        CAPTURE.reset();
    }

    #[test]
    fn a_negative_count_arms_nothing() {
        request(-3);
        assert_eq!(CAPTURE.int(), 0);
        CAPTURE.reset();
    }

    /// Naming and reading back are inert rather than failing: `gg-golden
    /// capture` refuses up front, and every other caller names a capture it may
    /// never take.
    #[test]
    fn naming_a_capture_without_renderdoc_is_inert() {
        set_path_template("target/golden/captures/triangle");
        assert_eq!(count(), 0);
        assert!(latest().is_none());
    }
}
