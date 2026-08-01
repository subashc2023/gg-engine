//! The floating-point environment guard — determinism hazard 5 (PLAN.md
//! §4.2.1). A dependency that sets flush-to-zero or denormals-are-zero in
//! MXCSR (x86-64) or FPCR (aarch64) silently changes results process-wide;
//! audio and math libraries are known to do this. Determinism Contract v1
//! requires round-to-nearest-even with FTZ/DAZ clear, so the contract's
//! precondition gets a machine: assert at startup, re-assert once per sim tick
//! in debug builds, and re-assert after every game-dylib load (§4.2.2).
//!
//! The registers are per-thread; the guard checks the calling thread, which is
//! the sim thread at every mandated call site.

/// A decoded snapshot of the calling thread's floating-point control register.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FpEnv {
    /// Raw register value: MXCSR on x86-64, FPCR on aarch64.
    pub raw: u64,
    /// Rounding is round-to-nearest-ties-even.
    pub round_to_nearest_even: bool,
    /// Flush-to-zero is set (results are flushed; a violation).
    pub flush_to_zero: bool,
    /// Denormals-are-zero is set (inputs are flushed; a violation).
    /// AArch64's FPCR has a single FZ bit covering both; it is reported as
    /// both `flush_to_zero` and `denormals_are_zero`.
    pub denormals_are_zero: bool,
}

/// The control register's architecture-specific name, for messages.
pub const FP_CONTROL_REGISTER: &str = if cfg!(target_arch = "x86_64") {
    "MXCSR"
} else if cfg!(target_arch = "aarch64") {
    "FPCR"
} else {
    "FP control register"
};

impl FpEnv {
    /// True when the environment satisfies Determinism Contract v1:
    /// round-to-nearest-even, FTZ and DAZ clear.
    pub fn is_contract(self) -> bool {
        self.round_to_nearest_even && !self.flush_to_zero && !self.denormals_are_zero
    }
}

/// Read the calling thread's floating-point environment.
pub fn read_fp_env() -> FpEnv {
    imp::read()
}

/// Check the calling thread's FP environment against Determinism Contract v1.
///
/// Returns the offending snapshot so the caller can report it; the mandated
/// call sites use [`assert_fp_env`], which panics with the register named.
pub fn check_fp_env() -> Result<(), FpEnv> {
    let env = read_fp_env();
    if env.is_contract() { Ok(()) } else { Err(env) }
}

/// Assert Determinism Contract v1's FP environment (hazard 5), panicking with
/// a message that names the register and the expected state.
///
/// Call at startup, once per sim tick in debug builds, and after every
/// game-dylib load.
#[track_caller]
pub fn assert_fp_env() {
    if let Err(env) = check_fp_env() {
        let mut violations = Vec::new();
        if !env.round_to_nearest_even {
            violations.push("rounding is not round-to-nearest-even");
        }
        if env.flush_to_zero {
            violations.push("FTZ (flush-to-zero) is set");
        }
        if env.denormals_are_zero {
            violations.push("DAZ (denormals-are-zero) is set");
        }
        panic!(
            "FP environment violates Determinism Contract v1 (PLAN.md §4.2.1 hazard 5): \
             {FP_CONTROL_REGISTER} = {:#x} — {}; expected round-to-nearest-even with FTZ/DAZ \
             clear. Something in this process rewrote the FP control register.",
            env.raw,
            violations.join(", "),
        );
    }
}

#[cfg(target_arch = "x86_64")]
mod imp {
    use super::FpEnv;

    // MXCSR: DAZ = bit 6, RC = bits 13..14 (0b00 = round-to-nearest-even),
    // FTZ = bit 15.
    const DAZ: u32 = 1 << 6;
    const RC_MASK: u32 = 0b11 << 13;
    const FTZ: u32 = 1 << 15;

    pub fn read() -> FpEnv {
        let mut raw: u32 = 0;
        // SAFETY: STMXCSR stores MXCSR to the pointed-at u32; SSE is baseline
        // on x86_64. (The _mm_getcsr intrinsic is deprecated in favor of this.)
        unsafe {
            core::arch::asm!(
                "stmxcsr [{ptr}]",
                ptr = in(reg) &mut raw,
                options(nostack, preserves_flags)
            );
        }
        FpEnv {
            raw: u64::from(raw),
            round_to_nearest_even: raw & RC_MASK == 0,
            flush_to_zero: raw & FTZ != 0,
            denormals_are_zero: raw & DAZ != 0,
        }
    }

    /// Set or clear FTZ on the calling thread. Test-support only: the whole
    /// point of this module is that nothing else ever does this.
    #[cfg(test)]
    pub fn set_flush_to_zero(on: bool) {
        let mut raw: u32 = 0;
        // SAFETY: read-modify-write of MXCSR touching only FTZ; the test
        // restores the prior state.
        unsafe {
            core::arch::asm!(
                "stmxcsr [{ptr}]",
                ptr = in(reg) &mut raw,
                options(nostack, preserves_flags)
            );
            let new = if on { raw | FTZ } else { raw & !FTZ };
            core::arch::asm!(
                "ldmxcsr [{ptr}]",
                ptr = in(reg) &new,
                options(nostack, preserves_flags)
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod imp {
    use super::FpEnv;

    // FPCR: FZ = bit 24 (one bit flushes both inputs and outputs),
    // RMode = bits 22..23 (0b00 = round-to-nearest-even).
    const FZ: u64 = 1 << 24;
    const RMODE_MASK: u64 = 0b11 << 22;

    pub fn read() -> FpEnv {
        let raw: u64;
        // SAFETY: MRS from FPCR is always legal at EL0 and has no side effects.
        unsafe {
            core::arch::asm!("mrs {}, fpcr", out(reg) raw, options(nomem, nostack, preserves_flags));
        }
        FpEnv {
            raw,
            round_to_nearest_even: raw & RMODE_MASK == 0,
            flush_to_zero: raw & FZ != 0,
            denormals_are_zero: raw & FZ != 0,
        }
    }

    #[cfg(test)]
    pub fn set_flush_to_zero(on: bool) {
        let raw: u64;
        // SAFETY: read-modify-write of FPCR touching only FZ; the test
        // restores the prior value.
        unsafe {
            core::arch::asm!("mrs {}, fpcr", out(reg) raw, options(nomem, nostack, preserves_flags));
            let new = if on { raw | FZ } else { raw & !FZ };
            core::arch::asm!("msr fpcr, {}", in(reg) new, options(nomem, nostack, preserves_flags));
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    /// M1 exit criterion (§6): deliberately setting FTZ trips the assertion
    /// with a message naming the register and the expected state.
    #[test]
    fn ftz_trips_the_assertion_naming_the_register() {
        assert_fp_env(); // sane before we vandalize it

        imp::set_flush_to_zero(true);
        let result = std::panic::catch_unwind(assert_fp_env);
        imp::set_flush_to_zero(false);

        let payload = result.expect_err("assert_fp_env must trip while FTZ is set");
        let message = payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .unwrap();
        assert!(
            message.contains(FP_CONTROL_REGISTER),
            "panic message must name the register: {message}"
        );
        assert!(
            message.contains("flush-to-zero"),
            "panic message must name the violation: {message}"
        );
        assert!(
            message.contains("round-to-nearest-even with FTZ/DAZ clear"),
            "panic message must name the expected state: {message}"
        );

        assert_fp_env(); // restored: the test leaves no residue on this thread
    }

    #[test]
    fn default_environment_satisfies_the_contract() {
        let env = read_fp_env();
        assert!(env.round_to_nearest_even);
        assert!(!env.flush_to_zero);
        assert!(!env.denormals_are_zero);
        assert!(env.is_contract());
        assert!(check_fp_env().is_ok());
    }
}
