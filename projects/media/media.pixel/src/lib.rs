//! Indexed pixel art: a palette plus an index canvas.
//!
//! Pixels store palette indices, not colors. That is what makes a sprite
//! recolourable, keeps the data tiny and makes nearest-neighbour scaling
//! exact — there is no interpolation to invent shades.
//!
//! Ownership boundaries:
//! - this crate owns the index domain: the palette with its default table
//!   and hex parsing, index storage, the line/rect/flood tools, the
//!   nearest-neighbour export scale and the sprite-spec replay;
//! - editing always stays in the index domain; RGBA expansion happens only
//!   at the export boundary (`PixelCanvas::to_bitmap`), which produces
//!   `media.image` storage;
//! - file encoding lives in `media.codec` and is composed by the caller
//!   around the bitmap this crate exports; nothing here imports it.
//!
//! The semantics were absorbed from the OpenPixel core (Apache-2.0,
//! © OpenPixel Authors) with the values preserved; the one deliberate
//! correction is that a malformed hex colour is an explicit
//! `Error::invalid_input` instead of the binding layer's silent fallback
//! to transparent.
//!
//! ```rust
//! use media_pixel::{Palette, PixelCanvas};
//!
//! let mut canvas = PixelCanvas::new(8, 8);
//! canvas.rect(1, 1, 6, 6, 3, true);
//! let bitmap = canvas.to_bitmap(&Palette::default());
//! assert_eq!((bitmap.width, bitmap.height), (8, 8));
//! ```

pub mod canvas;
pub mod palette;
pub mod spec;

pub use canvas::PixelCanvas;
pub use palette::{Palette, Rgba, parse_hex};
pub use spec::{MAX_SPRITE_PIXELS, PixelOp, SpriteSpec, render_sprite};
