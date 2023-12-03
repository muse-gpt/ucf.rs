//! Canvas building blocks shared by drawing tools.
//!
//! A canvas is a size plus an ordered layer stack; a brush is a soft round stamp
//! plus a flow. Nothing here knows about documents, tools or UI — the products
//! (`OpenPaint`, `OpenPhoto`, `OpenPixel`, `OpenVoxel`) own their own models and
//! use these pieces to composite and to paint.
//!
//! ```rust
//! use geometry::Rgba;
//! use media_canvas::{Brush, BrushMode, LayerStack};
//!
//! let mut stack = LayerStack::new(64, 64);
//! let index = stack.add_layer("ink");
//! let brush = Brush::round(6.0, Rgba([255, 255, 255, 255]));
//! let frame = stack.layer_mut(index).unwrap();
//! brush.stroke(frame, &[[8.0, 32.0], [56.0, 32.0]]);
//! let composed = stack.composite(Rgba([0, 0, 0, 255]));
//! assert_eq!(composed.width, 64);
//! ```

pub mod brush;
pub mod layer;

pub use brush::{Brush, BrushMode};
pub use layer::{BlendMode, Layer, LayerStack};
