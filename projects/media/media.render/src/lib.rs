//! Frame orchestration for a media project.
//!
//! Independent composition entries, each behind its own feature:
//!
//! - `scene` (default): [`render_project`] evaluates the scene IR, mixes the
//!   audio bed and pushes both through the encoder.
//! - `sfx-export`: [`write_sfx`] renders one `media.synthesis` spec and
//!   encodes it through `media.codec` — no scene, no visual stack.
//! - `voice-export`: [`write_voice`] renders one `media.voice` song and
//!   encodes it through `media.codec` — no scene, no visual stack.
//! - `image-pipeline`: [`process_image`] decodes an image, runs one domain
//!   operation (develop recipe or effect stack) and encodes a PNG — no scene,
//!   no text, no audio in the dependency graph.
//! - `vector-export`: [`write_vector`] normalizes a `media.vector` document
//!   and encodes either a PNG (raster paint through `media.raster`) or the
//!   SVG artifact (through `media.codec`) — no scene, no text, no audio.
//! - `voxel-export`: [`write_voxel`] resolves a `media.voxel` spec into the
//!   deterministic face list and encodes either an isometric PNG (plan paint
//!   through `media.raster`) or the OBJ artifact (through `media.codec`) —
//!   no scene, no text, no audio.
//! - `layout-export`: [`write_page`] and [`write_overview`] resolve a
//!   `media.layout` comic spec, lower its page or contact-sheet plan through
//!   `media.raster` and the `paint` layer of `media.text`, and encode a PNG
//!   through `media.codec` — fonts and images are explicit resource inputs,
//!   no scene and no audio in the dependency graph.

#[cfg(feature = "scene")]
pub mod render;

#[cfg(feature = "scene")]
pub use render::{RenderStats, lower_audio, render_project};

#[cfg(feature = "sfx-export")]
pub mod sfx;

#[cfg(feature = "sfx-export")]
pub use sfx::{write_sfx, write_sfx_catalogue};

#[cfg(feature = "sfx-export")]
pub mod music;

#[cfg(feature = "sfx-export")]
pub use music::{write_instrument, write_instrument_catalogue, write_pattern, write_pattern_catalogue};

#[cfg(feature = "voice-export")]
pub mod voice;

#[cfg(feature = "voice-export")]
pub use voice::write_voice;

#[cfg(feature = "image-pipeline")]
pub mod image_pipeline;

#[cfg(feature = "image-pipeline")]
pub use image_pipeline::{ImageOperation, ProcessedImage, process_image};

#[cfg(feature = "vector-export")]
pub mod vector_export;

#[cfg(feature = "vector-export")]
pub use vector_export::{ExportedVector, VectorFormat, render_frame, write_normalized, write_vector};

#[cfg(feature = "voxel-export")]
pub mod voxel_export;

#[cfg(feature = "voxel-export")]
pub use voxel_export::{ExportedVoxel, VoxelFormat, render_preview, write_voxel, write_voxel_shaded};

#[cfg(feature = "layout-export")]
pub mod layout_export;

#[cfg(feature = "layout-export")]
pub use layout_export::{ExportedPage, FontAssets, LayoutAssets, MediaImages, render_overview, render_page, write_overview, write_page};
