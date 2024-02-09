//! Frame orchestration for a media project.
//!
//! Independent composition entries, each behind its own feature:
//!
//! - `scene` (default): [`render_project`] evaluates the scene IR, mixes the
//!   audio bed and pushes both through the encoder.
//! - `sfx-export`: [`write_sfx`] renders one `media.synthesis` spec and
//!   encodes it through `media.codec` — no scene, no visual stack.
//! - `image-pipeline`: [`process_image`] decodes an image, runs one domain
//!   operation (develop recipe or effect stack) and encodes a PNG — no scene,
//!   no text, no audio in the dependency graph.
//! - `vector-export`: [`write_vector`] normalizes a `media.vector` document
//!   and encodes either a PNG (raster paint through `media.raster`) or the
//!   SVG artifact (through `media.codec`) — no scene, no text, no audio.

#[cfg(feature = "scene")]
pub mod render;

#[cfg(feature = "scene")]
pub use render::{RenderStats, lower_audio, render_project};

#[cfg(feature = "sfx-export")]
pub mod sfx;

#[cfg(feature = "sfx-export")]
pub use sfx::{write_sfx, write_sfx_catalogue};

#[cfg(feature = "image-pipeline")]
pub mod image_pipeline;

#[cfg(feature = "image-pipeline")]
pub use image_pipeline::{ImageOperation, ProcessedImage, process_image};

#[cfg(feature = "vector-export")]
pub mod vector_export;

#[cfg(feature = "vector-export")]
pub use vector_export::{ExportedVector, VectorFormat, render_frame, write_normalized, write_vector};
