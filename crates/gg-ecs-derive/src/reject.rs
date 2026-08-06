//! Compile-time field-type rejection (§4.2.1).
//!
//! Two layers, because they fail differently:
//!
//! 1. **Syntactic interception** of the four mistakes that are both likely and
//!    diagnosable — `usize`/`isize`, raw pointers, references, and the
//!    unordered `std` maps. These get an error that names the field, says why,
//!    and names the replacement.
//! 2. **A trait-bound backstop** — every field must satisfy
//!    `gg_ecs::StateHash`. This is the layer that actually makes rejection
//!    *total*: a bare enum, a `gg_math::render` type, or something nobody
//!    anticipated has no impl and therefore does not compile. Layer 1 exists
//!    only to turn "trait bound not satisfied" into a sentence worth reading.
//!
//! Layer 2 alone would be correct. Layer 1 alone would be a blocklist, which is
//! the shape §1 rejects everywhere else.

use syn::spanned::Spanned;
use syn::{Ident, Type};

/// Both layers for one field. `Err` is layer 1; the returned tokens are layer 2.
pub(crate) fn field_guard(field: &Ident, ty: &Type) -> syn::Result<proc_macro2::TokenStream> {
    if let Some(reason) = diagnose(ty) {
        return Err(syn::Error::new(
            ty.span(),
            format!("field `{field}`: {reason}"),
        ));
    }
    Ok(super::encodable_guard(ty))
}

/// Layer 1. `None` means "nothing specific to say" — layer 2 still applies.
fn diagnose(ty: &Type) -> Option<String> {
    match ty {
        Type::Ptr(_) => Some(
            "raw pointers cannot be hashed state — an address is not a value, and the same \
             simulation would hash differently on every run (§4.2.1). Store an index into a \
             host-owned arena, or an `Entity`."
                .into(),
        ),
        Type::Reference(_) => Some(
            "references cannot be hashed state: components are `Pod` and cross the §4.2.2 reload \
             boundary by raw pointer, where a borrow has no meaning. Store an `Entity` or an \
             index."
                .into(),
        ),
        Type::Array(a) => diagnose(&a.elem),
        Type::Slice(s) => diagnose(&s.elem),
        Type::Paren(p) => diagnose(&p.elem),
        Type::Group(g) => diagnose(&g.elem),
        Type::Path(p) => diagnose_path(p),
        _ => None,
    }
}

fn diagnose_path(path: &syn::TypePath) -> Option<String> {
    let last = path.path.segments.last()?;
    let name = last.ident.to_string();

    // Recurse through generic arguments first, so `Vec<usize>` is caught by the
    // `usize` arm with the right message rather than passing this layer.
    if let syn::PathArguments::AngleBracketed(args) = &last.arguments {
        for arg in &args.args {
            if let syn::GenericArgument::Type(inner) = arg
                && let Some(reason) = diagnose(inner)
            {
                return Some(reason);
            }
        }
    }

    match name.as_str() {
        "usize" | "isize" => Some(format!(
            "`{name}` is platform-width, so its encoding would differ between a 32-bit and a \
             64-bit target and its meaning differs across the §4.2.2 boundary (§4.2.1). Use an \
             explicit width — `u32`/`u64` — chosen for the range the field actually needs."
        )),
        "HashMap" | "HashSet" => Some(format!(
            "`{name}` iterates in randomly-seeded order, which §4.2.1 hazard 3 names as a likelier \
             determinism break than floats. Use an `IndexMap`, a sorted `Vec`, or dense iteration."
        )),
        _ => None,
    }
}
