//! The rejection half of the derives (§4.2.1) — each case in `tests/reject/`
//! must fail to compile, with a message that names the offending field.
//!
//! `trybuild` compares against checked-in `.stderr` files. That is only stable
//! because the toolchain is pinned (§2); a toolchain bump can reword rustc's
//! rendering and require `TRYBUILD=overwrite cargo test -p gg-ecs --test
//! reject` plus a read of the diff. That is a feature, not friction: these
//! messages are the whole user interface of the ban, and a bump that degrades
//! one should be seen.

#![allow(clippy::unwrap_used, clippy::expect_used)]

#[test]
fn forbidden_shapes_do_not_compile() {
    let t = trybuild::TestCases::new();
    // §4.2.1's named list: platform-width integers, references, bare enums.
    t.compile_fail("tests/reject/usize_field.rs");
    t.compile_fail("tests/reject/reference_field.rs");
    t.compile_fail("tests/reject/bare_enum_field.rs");
    // The other two of layer 1's four hand-written diagnostics — the message
    // *is* the deliverable there, so an untested one is an unproven one.
    t.compile_fail("tests/reject/raw_pointer_field.rs");
    t.compile_fail("tests/reject/hash_map_field.rs");
    // The same list applies above the columns: side tables shed `Pod`, not the
    // hazards (§4.2.1).
    t.compile_fail("tests/reject/side_table_usize_field.rs");
    // §4.2 identity rules.
    t.compile_fail("tests/reject/missing_id.rs");
    t.compile_fail("tests/reject/tuple_struct.rs");
    t.compile_fail("tests/reject/enum_component.rs");
    t.compile_fail("tests/reject/missing_schema.rs");
    // §4.2.1 hazard 4: padding, caught by the `Pod` bound.
    t.compile_fail("tests/reject/padded_component.rs");
    // §4.2.1: the chunked accelerator is version-local by type.
    t.compile_fail("tests/reject/chunked_across_versions.rs");
}
