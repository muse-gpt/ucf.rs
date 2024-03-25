//! Media facade.
//!
//! Re-exports only, gated per domain: every crate stays independently usable
//! and a consumer turns off the features it does not need. `media.types` is
//! the unconditional base contract (error, hash, time base); a pure-audio
//! consumer can enable only `audio` and never pull in the visual stack.

pub use media_types as types;

#[cfg(feature = "audio")]
pub use media_audio as audio;
#[cfg(feature = "canvas")]
pub use media_canvas as canvas;
#[cfg(feature = "codec")]
pub use media_codec as codec;
#[cfg(feature = "dsp")]
pub use media_dsp as dsp;
#[cfg(feature = "image")]
pub use media_image as image;
#[cfg(feature = "layout")]
pub use media_layout as layout;
#[cfg(feature = "particles")]
pub use media_particles as particles;
#[cfg(feature = "pixel")]
pub use media_pixel as pixel;
#[cfg(feature = "post")]
pub use media_post as post;
#[cfg(feature = "voice")]
pub use media_voice as voice;
#[cfg(feature = "raster")]
pub use media_raster as raster;
#[cfg(any(
    feature = "render",
    feature = "image-pipeline",
    feature = "sfx-export",
    feature = "vector-export",
    feature = "voxel-export",
    feature = "layout-export"
))]
pub use media_render as render;
#[cfg(feature = "scene")]
pub use media_scene as scene;
#[cfg(feature = "synthesis")]
pub use media_synthesis as synthesis;
#[cfg(feature = "text")]
pub use media_text as text;
#[cfg(feature = "vector")]
pub use media_vector as vector;
#[cfg(feature = "voxel")]
pub use media_voxel as voxel;

#[cfg(feature = "vector-export")]
pub use media_render::{ExportedVector, VectorFormat, render_frame, write_vector};
#[cfg(feature = "voxel-export")]
pub use media_render::{ExportedVoxel, VoxelFormat, render_preview, write_voxel};
#[cfg(feature = "layout-export")]
pub use media_render::{ExportedPage, FontAssets, LayoutAssets, MediaImages, render_overview, render_page, write_overview, write_page};
#[cfg(feature = "image-pipeline")]
pub use media_render::{ImageOperation, ProcessedImage, process_image};
#[cfg(feature = "render")]
pub use media_render::{RenderStats, render_project};
#[cfg(feature = "sfx-export")]
pub use media_render::{write_sfx, write_sfx_catalogue};
pub use media_types::Error;
