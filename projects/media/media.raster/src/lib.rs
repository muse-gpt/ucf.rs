//! Software rasterizer: paint primitives on the `Frame` storage from `media.image`.

pub mod paint;
pub mod png;

pub use paint::*;
pub use png::write_png;
