//! The crash path (§4.8): per-pass breadcrumbs, and the report a `DEVICE_LOST`
//! is explained by.
//!
//! A lost device reports nothing by itself — the submit returns a code and the
//! frame that caused it is already gone. Two things survive it. Markers written
//! into host-visible memory as passes open and close are readable after the loss
//! (the mapping is host memory; the device is what died), so the pass that was
//! in flight can be *named*. And `VK_EXT_device_fault`, where the driver has it,
//! says what the hardware saw — a page fault's address, a vendor fault code.
//! Neither is required: lavapipe has no device fault at all (measured, §6 M8),
//! and the breadcrumbs still name the pass.
//!
//! Unconditional, unlike the timestamps beside them. §1.6's "zero mystery
//! crashes" is a promise about *shipped* builds, where nobody attaches a
//! debugger — and the cost is two marker writes per pass, not a query pool.

use crate::RhiError;
use crate::device::Device;
use crate::resource::{BufferDesc, BufferHandle, BufferKind, Resources};
use ash::vk;

/// Passes one frame leaves crumbs for — the same reach the timestamp pool has,
/// for the same reason: the v1 graph is four passes and a report that silently
/// covered a prefix would name the wrong suspect.
const MAX_TRACKED_PASSES: usize = 32;

/// Two marks per pass: opened, closed.
const MARKS_PER_SLOT: usize = MAX_TRACKED_PASSES * 2;

const BYTES_PER_SLOT: u64 = (MARKS_PER_SLOT * 4) as u64;

/// The buffer and the range one frame-in-flight slot writes into. Copied into
/// recording like [`Stamps`](crate::timing::Stamps), so the recorder needs no
/// borrow of the whole [`Breadcrumbs`].
#[derive(Clone, Copy)]
pub(crate) struct Crumbs {
    buffer: vk::Buffer,
    base: u64,
}

impl Crumbs {
    /// Whether pass `index` is inside the buffer's reach.
    pub(crate) fn covers(&self, index: usize) -> bool {
        index < MAX_TRACKED_PASSES
    }

    /// # Safety
    /// `cmd` must be recording outside a render pass, and `index` must satisfy
    /// [`Crumbs::covers`].
    pub(crate) unsafe fn begin(&self, device: &Device, cmd: vk::CommandBuffer, index: usize) {
        // SAFETY: caller contract.
        unsafe { self.mark(device, cmd, vk::PipelineStageFlags::TOP_OF_PIPE, index * 2) };
    }

    /// # Safety
    /// As [`Crumbs::begin`].
    pub(crate) unsafe fn end(&self, device: &Device, cmd: vk::CommandBuffer, index: usize) {
        // SAFETY: caller contract.
        unsafe {
            self.mark(
                device,
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                index * 2 + 1,
            )
        };
    }

    /// # Safety
    /// `cmd` is recording; `offset` is within this slot's range.
    unsafe fn mark(
        &self,
        device: &Device,
        cmd: vk::CommandBuffer,
        stage: vk::PipelineStageFlags,
        offset: usize,
    ) {
        // SAFETY: caller contract; the offset is this slot's own.
        unsafe {
            device.write_marker(cmd, stage, self.buffer, self.base + offset as u64 * 4, MARK);
        }
    }
}

/// What a written mark holds. The value carries no information — which mark it
/// is *is* its offset — so a nonzero constant is the whole vocabulary, and
/// "zero" reliably means "the device never got here".
const MARK: u32 = 1;

/// The breadcrumb buffer, one region per frame in flight, and the pass names
/// each region's marks belong to.
pub(crate) struct Breadcrumbs {
    buffer: BufferHandle,
    raw: vk::Buffer,
    /// Per slot, the names of the passes recorded there. Cleared and refilled in
    /// place rather than rebuilt: this runs in every tier, including the one
    /// that must not allocate per frame.
    pending: Vec<Vec<String>>,
    /// Per slot, how many passes the frame declared beyond what the buffer
    /// tracks (§6 M81). Kept because the alternative is silence in exactly the
    /// tier where this report is the only diagnostic there is: the truncation
    /// is a `take` in [`Breadcrumbs::prepare`], and a graph that outgrows
    /// [`MAX_TRACKED_PASSES`] would otherwise produce a report that looks
    /// complete and stops before the suspect.
    dropped: Vec<usize>,
    /// The slot the most recently recorded frame used — the one a report reads.
    last: usize,
}

impl Breadcrumbs {
    /// Allocate the buffer. `slots` is the caller's frames in flight (1 for an
    /// offscreen context, which blocks on every execute).
    pub(crate) fn new(
        device: &mut Device,
        resources: &mut Resources,
        slots: usize,
    ) -> Result<Self, RhiError> {
        crate::inject::point("Breadcrumbs::new")?;
        // Dynamic: host-visible and persistently mapped, which is the property
        // the whole mechanism rests on — the marks must be readable *after* the
        // device that wrote them is gone.
        let buffer = resources.create_buffer(
            device,
            &BufferDesc {
                name: "gg.crash.breadcrumbs",
                size: BYTES_PER_SLOT * slots as u64,
                kind: BufferKind::Dynamic,
            },
        )?;
        Ok(Self {
            raw: resources.buffer(buffer)?.raw,
            buffer,
            pending: vec![Vec::new(); slots],
            dropped: vec![0; slots],
            last: 0,
        })
    }

    /// Clear `slot`'s marks and record which passes are about to write there.
    ///
    /// Cleared host-side, not with a recorded fill: two transfer writes to one
    /// buffer are unordered against each other without a barrier, so a recorded
    /// clear could land *after* the marks it was meant to precede. A host write
    /// before the submit is ordered by the submit itself, and the caller has
    /// already proven this slot's previous frame retired.
    pub(crate) fn prepare<'a>(
        &mut self,
        resources: &mut Resources,
        slot: usize,
        names: impl Iterator<Item = &'a str>,
    ) -> Result<Crumbs, RhiError> {
        let base = BYTES_PER_SLOT * slot as u64;
        let region = base as usize..(base + BYTES_PER_SLOT) as usize;
        resources.mapped_mut(self.buffer)?[region].fill(0);
        let pending = &mut self.pending[slot];
        pending.clear();
        let mut dropped = 0usize;
        for (at, name) in names.enumerate() {
            if at < MAX_TRACKED_PASSES {
                pending.push(name.to_owned());
            } else {
                dropped += 1;
            }
        }
        self.dropped[slot] = dropped;
        self.last = slot;
        Ok(Crumbs {
            buffer: self.raw,
            base,
        })
    }

    /// What the last recorded frame reached, as a report. Reads the mapping —
    /// legal after a device loss, which is the only time it is called.
    pub(crate) fn report(&self, resources: &Resources) -> String {
        let names = &self.pending[self.last];
        let Ok(bytes) = resources.mapped(self.buffer) else {
            return "breadcrumbs: unreadable".into();
        };
        let base = (BYTES_PER_SLOT * self.last as u64) as usize;
        let marks: Vec<u32> = bytes[base..base + BYTES_PER_SLOT as usize]
            .chunks_exact(4)
            .map(|b| u32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        format_marks(names, &marks, self.dropped[self.last])
    }
}

/// One pass's fate, as the marks record it.
fn fate(begun: bool, ended: bool) -> &'static str {
    match (begun, ended) {
        (true, true) => "completed",
        (true, false) => "IN FLIGHT",
        // Marks are written in submission order, so a pass whose predecessor is
        // still running has simply not been reached — not a second suspect.
        (false, _) => "not reached",
    }
}

/// The breadcrumb half of a device-lost report: every pass the frame declared,
/// what became of it, and the first one that never closed named as the suspect.
///
/// `dropped` is what the frame declared past [`MAX_TRACKED_PASSES`], and it is
/// said rather than assumed away: a reader who cannot tell a complete list from
/// a truncated one will read "no pass was in flight" off a frame whose suspect
/// was never in the list.
fn format_marks(names: &[String], marks: &[u32], dropped: usize) -> String {
    if names.is_empty() {
        return "breadcrumbs: no frame had been recorded".into();
    }
    let mark = |i: usize| marks.get(i).is_some_and(|m| *m != 0);
    let mut out = String::new();
    let mut suspect = None;
    for (index, name) in names.iter().enumerate() {
        let (begun, ended) = (mark(index * 2), mark(index * 2 + 1));
        if begun && !ended && suspect.is_none() {
            suspect = Some(name.as_str());
        }
        out.push_str(&format!("\n  {:<16} {}", name, fate(begun, ended)));
    }
    if dropped > 0 {
        out.push_str(&format!(
            "\n  {dropped} further pass(es) were declared and are not tracked — \
             MAX_TRACKED_PASSES is {MAX_TRACKED_PASSES}"
        ));
    }
    let headline = match suspect {
        Some(name) => format!("breadcrumbs: pass `{name}` was in flight when the device was lost"),
        // Every pass closed, or none opened: the frame is not where the fault
        // is — unless the list is short, in which case the suspect may simply
        // not be in it and this must not claim otherwise.
        None if dropped > 0 => {
            "breadcrumbs: no *tracked* pass was in flight, and the frame was truncated".into()
        }
        None => "breadcrumbs: no pass was in flight — the loss is not this frame's graph".into(),
    };
    headline + &out
}

/// One address the device faulted at.
pub(crate) struct FaultAddress {
    pub kind: &'static str,
    pub address: u64,
    /// The reported address is only accurate to a multiple of this.
    pub precision: u64,
}

/// One vendor-specific fault the driver reported.
pub(crate) struct FaultVendor {
    pub description: String,
    pub code: u64,
    pub data: u64,
}

/// What `VK_EXT_device_fault` said. Absent on a driver without the extension —
/// lavapipe, measurably (§6 M8) — which costs detail and not the report.
pub(crate) struct Fault {
    pub description: String,
    pub addresses: Vec<FaultAddress>,
    pub vendors: Vec<FaultVendor>,
}

/// How a fault address type reads in a report. Names track the spec's, because
/// the next stop after reading one is the spec or a driver bug tracker.
pub(crate) fn address_kind(kind: vk::DeviceFaultAddressTypeEXT) -> &'static str {
    match kind {
        vk::DeviceFaultAddressTypeEXT::READ_INVALID => "read of an invalid address",
        vk::DeviceFaultAddressTypeEXT::WRITE_INVALID => "write to an invalid address",
        vk::DeviceFaultAddressTypeEXT::EXECUTE_INVALID => "execute of an invalid address",
        vk::DeviceFaultAddressTypeEXT::INSTRUCTION_POINTER_UNKNOWN => {
            "instruction pointer (unknown)"
        }
        vk::DeviceFaultAddressTypeEXT::INSTRUCTION_POINTER_INVALID => {
            "instruction pointer (invalid)"
        }
        vk::DeviceFaultAddressTypeEXT::INSTRUCTION_POINTER_FAULT => {
            "instruction pointer (faulting)"
        }
        _ => "unspecified",
    }
}

fn format_fault(fault: &Fault) -> String {
    let mut out = format!("device fault: {}", fault.description);
    for a in &fault.addresses {
        out.push_str(&format!(
            "\n  {} at {:#018x} (+/- {:#x})",
            a.kind, a.address, a.precision
        ));
    }
    for v in &fault.vendors {
        out.push_str(&format!(
            "\n  vendor: {} (code {:#x}, data {:#x})",
            v.description, v.code, v.data
        ));
    }
    out
}

/// The whole report: what wrote the marks, what the frame reached, and what the
/// driver saw. One string, because it goes to `tracing::error!` — which is what
/// puts it in the log tail a panic report attaches (§4.8).
pub(crate) fn report(mechanism: &'static str, crumbs: &str, fault: Option<&Fault>) -> String {
    let fault = match fault {
        Some(fault) => format_fault(fault),
        None => "device fault: VK_EXT_device_fault absent on this driver".into(),
    };
    format!("the device was lost\n{crumbs}\nmarkers: {mechanism}\n{fault}")
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn the_pass_that_opened_and_never_closed_is_the_suspect() {
        let names = names(&["depth", "forward", "ui"]);
        // depth done, forward opened, ui never reached.
        let report = format_marks(&names, &[1, 1, 1, 0, 0, 0], 0);
        assert!(report.contains("pass `forward` was in flight"), "{report}");
        assert!(report.contains("depth            completed"), "{report}");
        assert!(report.contains("ui               not reached"), "{report}");
    }

    #[test]
    fn a_frame_that_finished_names_no_suspect() {
        let report = format_marks(&names(&["depth", "ui"]), &[1, 1, 1, 1], 0);
        assert!(report.contains("no pass was in flight"), "{report}");
        assert!(!report.contains("IN FLIGHT"), "{report}");
    }

    /// A truncated read must not silently promote a completed pass: absent marks
    /// read as zero, which is "not reached".
    #[test]
    fn missing_marks_read_as_not_reached() {
        let report = format_marks(&names(&["depth", "ui"]), &[1, 1], 0);
        assert!(report.contains("ui               not reached"), "{report}");
        assert!(report.contains("no pass was in flight"), "{report}");
    }

    #[test]
    fn a_report_before_any_frame_says_so() {
        assert!(format_marks(&[], &[], 0).contains("no frame had been recorded"));
    }

    /// A frame longer than the buffer says so rather than reading complete (§6
    /// M81). The headline is the load-bearing half: "no pass was in flight" off
    /// a list that stops before the suspect is a report that sends a reader to
    /// the wrong place, and it is *only* reachable in a tier without a
    /// debugger, since that is the whole point of §4.8.
    #[test]
    fn a_frame_longer_than_the_buffer_does_not_read_as_complete() {
        let report = format_marks(&names(&["depth", "ui"]), &[1, 1, 1, 1], 3);
        assert!(report.contains("3 further pass(es)"), "{report}");
        assert!(report.contains("truncated"), "{report}");
        assert!(
            !report.contains("the loss is not this frame's graph"),
            "{report}"
        );
        // A named suspect still leads, since the list did reach it.
        let found = format_marks(&names(&["depth", "ui"]), &[1, 1, 1, 0], 3);
        assert!(found.contains("`ui` was in flight"), "{found}");
        assert!(found.contains("3 further pass(es)"), "{found}");
    }

    #[test]
    fn a_fault_report_carries_addresses_and_vendor_codes() {
        let fault = Fault {
            description: "page fault".into(),
            addresses: vec![FaultAddress {
                kind: address_kind(vk::DeviceFaultAddressTypeEXT::READ_INVALID),
                address: 0xdead_beef,
                precision: 0x1000,
            }],
            vendors: vec![FaultVendor {
                description: "SQ hang".into(),
                code: 7,
                data: 0x42,
            }],
        };
        let report = report("VK_AMD_buffer_marker", "breadcrumbs: none", Some(&fault));
        assert!(report.contains("read of an invalid address at 0x00000000deadbeef"));
        assert!(report.contains("+/- 0x1000"));
        assert!(report.contains("vendor: SQ hang (code 0x7, data 0x42)"));
        assert!(report.contains("markers: VK_AMD_buffer_marker"));
    }

    /// The lavapipe case (§6 M8): no extension, and the report still says what
    /// the frame reached rather than nothing at all.
    #[test]
    fn a_driver_without_the_extension_still_reports_breadcrumbs() {
        let crumbs = format_marks(&names(&["ui"]), &[1, 0], 0);
        let report = report("cmd_fill_buffer", &crumbs, None);
        assert!(report.contains("pass `ui` was in flight"), "{report}");
        assert!(report.contains("VK_EXT_device_fault absent"), "{report}");
    }
}
