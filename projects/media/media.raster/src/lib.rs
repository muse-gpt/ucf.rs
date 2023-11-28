//! Software rasterizer: frame buffers and paint primitives.

pub mod paint;
pub mod png;

pub use paint::*;
pub use png::write_png;
