//! Make the failure happen, so the code that handles one runs (§6 M85).
//!
//! Every `Err` arm in this crate's bring-up and frame paths is written against
//! a call that has never returned one. Lavapipe does not run out of memory on a
//! 64x64 target, and a device that is losing is losing for reasons no gate can
//! arrange — so nine unwind ladders are reviewed, never executed, and read
//! correct whether they are or not. Reading is what found
//! [`OffscreenRhi::execute`](crate::OffscreenRhi::execute) dropping the
//! transfer queue's acquires on three of them, and reading is also what cleared
//! two ladders that were never wrong. One in three is what a review is worth
//! here; a gate is worth more.
//!
//! There are two kinds of site and they reach different arms.
//! [`Device::allocate`](crate::device::Device::allocate) is the first, because
//! §4.3 already made it the one place every byte passes through — but a device
//! runs out of memory in only four places during an offscreen bring-up, and
//! none of them is a ladder. The nine ladders unwind a *constructor*, and most
//! of those fail for reasons that are not memory at all (a descriptor pool, a
//! query pool, a swapchain the surface will not grant), so the second kind of
//! site is [`point`], one line at the top of each fallible constructor. It
//! grades the caller's ladder; the allocator grades the constructor's own.
//!
//! Either way the failure happens **before** the real call, so the unwind that
//! follows is the real one and nothing else about the run has moved.
//!
//! Two counters grade it and they catch different leaks:
//!
//! - [`live`] is this crate's own ledger and survives the allocator being
//!   dropped, which the §4.3 leak report cannot — a refused bring-up returns
//!   `Err` and leaves no [`ShutdownReport`](crate::ShutdownReport) to ask.
//! - The validation layer is the other half. A Vulkan object outliving its
//!   device holds no memory and reaches no ledger; it is
//!   `VUID-vkDestroyDevice-device-05137` and nothing else here would say so.
//!
//! Off by default and in no tier (§2): `cargo nextest run -p gg-rhi --features
//! inject` is the only build that has it, which is also why arming is an
//! in-process call rather than an environment variable — the sweep is one
//! process taking a device per leg, not one process per allocation.

#[cfg(feature = "inject")]
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

/// Allocation attempts since [`reset`].
#[cfg(feature = "inject")]
static SEEN: AtomicU64 = AtomicU64::new(0);

/// Which attempt fails. `u64::MAX` is disarmed, which is why the index is
/// zero-based rather than a count — attempt 0 has to be reachable.
#[cfg(feature = "inject")]
static ARMED: AtomicU64 = AtomicU64::new(u64::MAX);

/// Allocations handed out and not yet returned. Signed so a double free reads
/// as its own defect instead of wrapping into a very large leak.
#[cfg(feature = "inject")]
static LIVE: AtomicI64 = AtomicI64::new(0);

/// Which [`point`] fails, for a test aimed at one arm rather than sweeping.
#[cfg(feature = "inject")]
static SITE: std::sync::Mutex<Option<&'static str>> = std::sync::Mutex::new(None);

/// Fail the `nth` site after the next [`reset`], counting from 0. A sweep's
/// spelling — [`arm_site`] is the one for a test that means a particular arm.
#[cfg(feature = "inject")]
pub fn arm(nth: u64) {
    ARMED.store(nth, Ordering::Relaxed);
}

/// Fail the [`point`] with this name, wherever in the run it falls.
///
/// An index is the wrong handle for a test about one arm: it moves whenever a
/// site is added anywhere earlier, and the failure that follows is a puzzle
/// about counting rather than about the arm.
#[cfg(feature = "inject")]
pub fn arm_site(name: &'static str) {
    if let Ok(mut site) = SITE.lock() {
        *site = Some(name);
    }
}

/// Stop failing anything. Teardown runs disarmed: a leg is about the unwind
/// after *one* failure, and a second one during cleanup would grade a path no
/// caller can reach.
#[cfg(feature = "inject")]
pub fn disarm() {
    ARMED.store(u64::MAX, Ordering::Relaxed);
    if let Ok(mut site) = SITE.lock() {
        *site = None;
    }
}

/// Zero the attempt counter. Deliberately does not touch [`live`], which is the
/// invariant *across* legs: every leg must return it to 0.
#[cfg(feature = "inject")]
pub fn reset() {
    SEEN.store(0, Ordering::Relaxed);
}

/// Allocation attempts since [`reset`] — the census a sweep's upper bound is.
#[cfg(feature = "inject")]
#[must_use]
pub fn seen() -> u64 {
    SEEN.load(Ordering::Relaxed)
}

/// Allocations outstanding. Nonzero after a leg is a leak the §4.3 report was
/// not alive to hear.
#[cfg(feature = "inject")]
#[must_use]
pub fn live() -> i64 {
    LIVE.load(Ordering::Relaxed)
}

/// Validation messages heard this process — the half of the oracle that needs
/// no live device, and the only one that sees an object leaked without memory.
#[cfg(feature = "inject")]
#[must_use]
pub fn validation_messages() -> u64 {
    crate::instance::validation_message_count()
}

/// Whether this attempt is the armed one. Always `false` without the feature,
/// and the call folds away with it.
pub(crate) fn allocating() -> bool {
    fires()
}

/// A named place a caller's unwind ladder can be made to run — one line at the
/// top of a fallible constructor, before it has built anything.
///
/// [`crate::RhiError::Loader`] rather than a variant of its own: what a ladder
/// does with an error must not depend on which error it is, and a variant only
/// this seam produces would let a `match` somewhere quietly grow a branch that
/// no shipping build can reach.
pub(crate) fn point(name: &'static str) -> Result<(), crate::RhiError> {
    // `fires` first and unconditionally: it is what advances the index, and a
    // short-circuit here would make a named arming shift every later site.
    let by_index = fires();
    if by_index || named(name) {
        return Err(crate::RhiError::Loader(format!(
            "injected failure at `{name}` (§6 M85)"
        )));
    }
    Ok(())
}

/// Whether [`arm_site`] named this one.
#[allow(unused_variables)]
fn named(name: &'static str) -> bool {
    #[cfg(feature = "inject")]
    {
        SITE.lock().is_ok_and(|site| *site == Some(name))
    }
    #[cfg(not(feature = "inject"))]
    false
}

/// One counter behind both kinds of site, so a sweep is a single index over
/// every place this run could have failed, in the order it reaches them.
fn fires() -> bool {
    #[cfg(feature = "inject")]
    {
        SEEN.fetch_add(1, Ordering::Relaxed) == ARMED.load(Ordering::Relaxed)
    }
    #[cfg(not(feature = "inject"))]
    false
}

/// An allocation the allocator granted.
pub(crate) fn allocated() {
    #[cfg(feature = "inject")]
    LIVE.fetch_add(1, Ordering::Relaxed);
}

/// An allocation returned to the allocator.
pub(crate) fn freed() {
    #[cfg(feature = "inject")]
    LIVE.fetch_sub(1, Ordering::Relaxed);
}
