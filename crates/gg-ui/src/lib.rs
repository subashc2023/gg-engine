//! `gg-ui` (§4.9) — the UI layer we own.
//!
//! We own the draw batching, the glyph atlas, the input routing, the layout and
//! the styling, because those are what carry a framework's worldview into every
//! call site. We rent text shaping and rasterization. We refuse `egui` in an
//! engine crate.
//!
//! The renderer boundary is one slice and one bitmap: [`draw::DrawList`] hands
//! `gg_render` a slice of `UiVertex` and [`atlas::Atlas`] hands it coverage
//! texels. Neither knows what a widget is, which is what let M8's overlay be
//! built on half of this and replaced at M13 without the renderer noticing.

#![warn(missing_docs)]

pub mod atlas;
pub mod boundary;
pub mod draw;
pub mod font;
pub mod layout;
pub mod router;
pub mod scratch;
pub mod text;

pub use atlas::Atlas;
pub use boundary::Ui;
pub use draw::{DrawList, Glyph, Rect};
pub use layout::{Axis, Stack};
pub use router::{Pointer, Router, WidgetId};
pub use scratch::{Scratch, Span};
pub use text::{FaceId, Fonts};
