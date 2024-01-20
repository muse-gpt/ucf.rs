//! Frame orchestration for a media project.
//!
//! Two independent composition entries, each behind its own feature:
//!
//! - `scene` (default): [`render_project`] evaluates the scene IR, mixes the
//!   audio bed and pushes both through the encoder.
//! - `image-pipeline`: [`process_image`] decodes an image, runs one domain
//!   operation (develop recipe or effect stack) and encodes a PNG — no scene,
//!   no text, no audio in the dependency graph.

#[cfg(feature = "scene")]
pub mod render;

#[cfg(feature = "scene")]
pub use render::{RenderStats, render_project};

#[cfg(feature = "image-pipeline")]
pub mod image_pipeline;

#[cfg(feature = "image-pipeline")]
pub use image_pipeline::{ImageOperation, ProcessedImage, process_image};
