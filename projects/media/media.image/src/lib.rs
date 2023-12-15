//! Image storage and format conversion.
//!
//! Owns the pixel buffers the media crates share: the decoded `Bitmap`, the
//! render target `Frame`, the layout `Fit` and the linear HDR `Framebuffer`.
//! Painting lives in `media.raster`, post-processing semantics in
//! `media.post`, file encoding in `media.codec`.

mod bitmap;
mod fit;
mod frame;
mod hdr;

pub use bitmap::Bitmap;
pub use fit::Fit;
pub use frame::Frame;
pub use hdr::{Framebuffer, FramebufferEncoding};
