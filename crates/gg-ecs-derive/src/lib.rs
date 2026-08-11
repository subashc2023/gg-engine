//! `gg-ecs-derive` (§4.2, §4.2.1) — the derives `gg-ecs` cannot host itself,
//! because a `proc-macro = true` crate may export nothing else.
//!
//! Use them through `gg_ecs`, which re-exports every one; naming this crate
//! directly from a game crate would trip the §4.2.2 boundary deny pin.
//!
//! Three derives and two function-like macros. The macros exist because §4.2's
//! ids are folded here at expansion time and `ComponentId::of` is not `const`,
//! so a hand-written impl has no other way to name the constant the trait now
//! requires.

use proc_macro::TokenStream;
use quote::{quote, quote_spanned};
use syn::spanned::Spanned;
use syn::{Data, DeriveInput, Fields, Ident, LitStr, Type, parse_macro_input};

mod reject;

/// `#[derive(Component)]` with a mandatory `#[component(id = "…")]` (§4.2).
///
/// Emits the declared id, the type name for diagnostics, and a `FieldDesc` per
/// field with `offset_of!`/`size_of` filled in at compile time. Also emits the
/// `StateHash` impl: a component is hashed state by definition, so requiring a
/// second derive would only create the opportunity to forget one.
#[proc_macro_derive(Component, attributes(component))]
pub fn derive_component(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand_component(&input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// `component_id_of!("declared-id")` — the same fold the `Component` derive
/// performs, as a constant expression (§4.2).
///
/// The escape hatch that keeps [`Component`](macro@Component) hand-implementable:
/// `ComponentId::of` is a blake3 and so not `const`, which would otherwise make
/// `COMPONENT_ID` a constant no hand-written impl could name. Rare by design —
/// the derive is the path — but a trait nothing outside this crate can
/// implement is a different trait than the one §4.2 describes.
#[proc_macro]
pub fn component_id_of(input: TokenStream) -> TokenStream {
    let declared = parse_macro_input!(input as LitStr);
    if declared.value().is_empty() {
        return syn::Error::new(declared.span(), "id must not be empty")
            .into_compile_error()
            .into();
    }
    let bits = component_id_bits(&declared.value());
    quote! { ::gg_ecs::ComponentId::from_raw(#bits) }.into()
}

/// `side_table_id_of!("declared-id")` — [`component_id_of!`] under §4.2.1's
/// side-table domain, for a [`SideTable`](macro@SideTable) written by hand.
#[proc_macro]
pub fn side_table_id_of(input: TokenStream) -> TokenStream {
    let declared = parse_macro_input!(input as LitStr);
    if declared.value().is_empty() {
        return syn::Error::new(declared.span(), "id must not be empty")
            .into_compile_error()
            .into();
    }
    let bits = side_table_id_bits(&declared.value());
    quote! { ::gg_ecs::SideTableId::from_raw(#bits) }.into()
}

/// `#[derive(StateHash)]` — the protocol encoding alone, for a type hashed as
/// part of something else (§4.2.1).
///
/// Components and side tables do not need this: their derives emit it.
#[proc_macro_derive(StateHash)]
pub fn derive_state_hash(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand_state_hash(&input, "StateHash")
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// `#[derive(SideTable)]` with a mandatory `#[side_table(id = "…")]` (§4.2.1) —
/// host-owned sim state that is protocol-hashable without being `Pod`.
///
/// Same field-type rejection as `Component`, minus the `Pod` bound: that is the
/// whole point of the decoupling, so a side table may hold `Vec`s and `String`s.
#[proc_macro_derive(SideTable, attributes(side_table))]
pub fn derive_side_table(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand_side_table(&input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Named-struct fields, or an error naming what was found instead.
///
/// Tuple structs are refused deliberately, not incidentally: §4.2.2 migration
/// matches fields *by name*, so a positional component could only ever migrate
/// by position — silently retyping state on any field reorder.
fn named_fields(input: &DeriveInput, derive: &str) -> syn::Result<Vec<(Ident, Type)>> {
    let fields = match &input.data {
        Data::Struct(s) => &s.fields,
        Data::Enum(e) => {
            return Err(syn::Error::new(
                e.enum_token.span(),
                format!(
                    "{derive} cannot be derived for an enum: a bare enum's discriminant \
                     representation is not pinned by the language (§4.2.1 hazard 4). Use the \
                     newtyped-integer pattern — a `#[repr(transparent)]` struct over a `u8`/`u32` \
                     with validated accessors."
                ),
            ));
        }
        Data::Union(u) => {
            return Err(syn::Error::new(
                u.union_token.span(),
                format!("{derive} cannot be derived for a union: no field is known to be active"),
            ));
        }
    };
    match fields {
        Fields::Named(named) => named
            .named
            .iter()
            .map(|f| {
                let ident = f.ident.clone().ok_or_else(|| {
                    syn::Error::new(f.span(), "named field without an identifier")
                })?;
                Ok((ident, f.ty.clone()))
            })
            .collect(),
        // A field-less struct is legal: marker components are useful and hash
        // as their id alone.
        Fields::Unit => Ok(Vec::new()),
        Fields::Unnamed(unnamed) => Err(syn::Error::new(
            unnamed.span(),
            format!(
                "{derive} requires named fields: §4.2.2 migration matches fields by name, so a \
                 tuple struct could only migrate by position — silently retyping state whenever \
                 fields are reordered"
            ),
        )),
    }
}

/// The `id = "…"` from `#[<attr>(id = "…")]`, where `attr` is `component` or
/// `side_table`.
fn declared_id(input: &DeriveInput, attr_name: &str) -> syn::Result<LitStr> {
    let mut found: Option<LitStr> = None;
    for attr in &input.attrs {
        if !attr.path().is_ident(attr_name) {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("id") {
                let lit: LitStr = meta.value()?.parse()?;
                if lit.value().is_empty() {
                    return Err(meta.error("id must not be empty"));
                }
                found = Some(lit);
                Ok(())
            } else {
                Err(meta.error(format!(
                    "unknown `{attr_name}` option; the only one is `id = \"…\"`"
                )))
            }
        })?;
    }
    found.ok_or_else(|| {
        syn::Error::new(
            input.ident.span(),
            format!(
                "missing `#[{attr_name}(id = \"…\")]` (§4.2). The id is the type's *persisted* \
                 identity, deliberately decoupled from the Rust type path so that moving or \
                 renaming the type is free and changing the id is the visible, deliberate act a \
                 schema change ought to be. It is mandatory so the choice is made once, at \
                 birth, when it costs nothing."
            ),
        )
    })
}

/// `ComponentId::of`'s construction, performed here at expansion time (§4.2).
///
/// The duplication is the domain string and the truncation, not the hash: this
/// crate depends on the same pinned `blake3`, so there is no second
/// implementation to keep bit-identical — which is the objection
/// [`gg_ecs::ComponentId::of`]'s doc raises against a `const fn` and the reason
/// this is a proc macro instead. What *could* still drift is the domain
/// separator below, so `Registry::register` recomputes the id the ordinary way
/// and refuses a disagreement: the check is total across every component any
/// build registers, and costs the hash that used to happen there anyway.
fn component_id_bits(declared: &str) -> u64 {
    folded_id(b"gg-ecs/component-id/v1\0", declared) // hash.rs's DOMAIN_COMPONENT_ID
}

/// As [`component_id_bits`], under §4.2.1's other namespace — the domains are
/// separate so that promoting a side table to a component moves its id loudly.
fn side_table_id_bits(declared: &str) -> u64 {
    folded_id(b"gg-ecs/side-table-id/v1\0", declared) // hash.rs's DOMAIN_SIDE_TABLE_ID
}

fn folded_id(domain: &[u8], declared: &str) -> u64 {
    let mut h = blake3::Hasher::new();
    h.update(domain);
    h.update(declared.as_bytes());
    let mut out = [0u8; 8];
    out.copy_from_slice(&h.finalize().as_bytes()[..8]);
    u64::from_le_bytes(out)
}

fn expand_component(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let id = declared_id(input, "component")?;
    let fields = named_fields(input, "Component")?;
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    if !input.generics.params.is_empty() {
        return Err(syn::Error::new(
            input.generics.span(),
            "a generic component cannot have one stable id: every instantiation would claim the \
             same persisted identity with a different layout. Declare a concrete type per \
             instantiation.",
        ));
    }

    let mut checks = Vec::new();
    let mut descs = Vec::new();
    let mut hashes = Vec::new();
    for (ident, ty) in &fields {
        checks.push(reject::field_guard(ident, ty)?);
        let field_name = ident.to_string();
        // Source-token spelling, whitespace-normalized so `[u32 ; 4]` and
        // `[u32; 4]` fingerprint alike.
        let ty_token = normalize_type(ty);
        descs.push(quote! {
            ::gg_ecs::FieldDesc {
                name: #field_name,
                ty: #ty_token,
                offset: ::core::mem::offset_of!(#name, #ident),
                size: ::core::mem::size_of::<#ty>(),
            }
        });
        hashes.push(quote! { ::gg_ecs::StateHash::state_hash(&self.#ident, h); });
    }

    let type_name = name.to_string();
    let guards = guard_block(&checks);
    let id_bits = component_id_bits(&id.value());
    Ok(quote! {
        #guards

        impl #impl_generics ::gg_ecs::Component for #name #ty_generics #where_clause {
            const DECLARED_ID: &'static str = #id;
            const COMPONENT_ID: ::gg_ecs::ComponentId =
                ::gg_ecs::ComponentId::from_raw(#id_bits);
            const TYPE_NAME: &'static str = #type_name;
            const FIELDS: &'static [::gg_ecs::FieldDesc] = &[#(#descs),*];
        }

        impl #impl_generics ::gg_ecs::StateHash for #name #ty_generics #where_clause {
            fn state_hash(&self, h: &mut ::gg_ecs::StateHasher) {
                // Declaration order, field by field — never raw memory, so
                // padding cannot reach the hash (§4.2.1 hazard 4).
                #(#hashes)*
            }
        }
    })
}

fn expand_side_table(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let id = declared_id(input, "side_table")?;
    let name = &input.ident;
    if !input.generics.params.is_empty() {
        return Err(syn::Error::new(
            input.generics.span(),
            "a generic side table cannot have one stable id: every instantiation would claim the \
             same persisted identity. Declare a concrete type per instantiation.",
        ));
    }
    let type_name = name.to_string();
    // The encoding is the `StateHash` derive's, unchanged — a side table is
    // hashed state that happens to be registered by name.
    let state_hash = expand_state_hash(input, "SideTable")?;
    let id_bits = side_table_id_bits(&id.value());
    Ok(quote! {
        #state_hash

        impl ::gg_ecs::SideTable for #name {
            const DECLARED_ID: &'static str = #id;
            const SIDE_TABLE_ID: ::gg_ecs::SideTableId =
                ::gg_ecs::SideTableId::from_raw(#id_bits);
            const TYPE_NAME: &'static str = #type_name;
        }
    })
}

fn expand_state_hash(input: &DeriveInput, derive: &str) -> syn::Result<proc_macro2::TokenStream> {
    let fields = named_fields(input, derive)?;
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let mut checks = Vec::new();
    let mut hashes = Vec::new();
    for (ident, ty) in &fields {
        checks.push(reject::field_guard(ident, ty)?);
        hashes.push(quote! { ::gg_ecs::StateHash::state_hash(&self.#ident, h); });
    }

    let guards = guard_block(&checks);
    Ok(quote! {
        #guards

        impl #impl_generics ::gg_ecs::StateHash for #name #ty_generics #where_clause {
            fn state_hash(&self, h: &mut ::gg_ecs::StateHasher) {
                #(#hashes)*
            }
        }
    })
}

/// The field's type as a single-spaced token string, so incidental source
/// whitespace does not move a schema hash.
fn normalize_type(ty: &Type) -> String {
    let mut out = String::new();
    for token in quote!(#ty).to_string().split_whitespace() {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(token);
    }
    out
}

/// The trait-bound backstop, as one call per field. If `T: StateHash` does not
/// hold, this fails to compile with the error spanned at that field's type.
///
/// A *call*, not a top-level item: these are collected into a single anonymous
/// `const _` per derive by [`guard_block`], so two components with a field of
/// the same name in one module do not collide.
fn encodable_guard(ty: &Type) -> proc_macro2::TokenStream {
    quote_spanned! { ty.span() =>
        field_type_must_have_a_defined_encoding::<#ty>();
    }
}

/// Wrap per-field guards in an anonymous const. Anonymous consts never collide,
/// and nothing inside escapes into the caller's namespace.
fn guard_block(checks: &[proc_macro2::TokenStream]) -> proc_macro2::TokenStream {
    if checks.is_empty() {
        return quote! {};
    }
    quote! {
        const _: () = {
            fn field_type_must_have_a_defined_encoding<T: ::gg_ecs::StateHash + ?Sized>() {}
            #[allow(dead_code)]
            fn guards() {
                #(#checks)*
            }
        };
    }
}
