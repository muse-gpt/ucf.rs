//! Media facade.
//!
//! Re-exports only, gated per domain: every crate stays independently usable
//! and a consumer turns off the features it does not need. `media.types` is
//! the unconditional base contract (error, hash, time base); a pure-audio
//! consumer can enable only `audio` and never pull in the visual stack.

pub use media_types as types;

#[cfg(feature = "image")]
pub use media_image as image;
#[cfg(feature = "audio")]
pub use media_audio as audio;
#[cfg(feature = "canvas")]
pub use media_canvas as canvas;
#[cfg(feature = "codec")]
pub use media_codec as codec;
#[cfg(feature = "pixel")]
pub use media_pixel as pixel;
#[cfg(feature = "particles")]
pub use media_particles as particles;
#[cfg(feature = "post")]
pub use media_post as post;
#[cfg(feature = "raster")]
pub use media_raster as raster;
#[cfg(any(feature = "render", feature = "image-pipeline"))]
pub use media_render as render;
#[cfg(feature = "scene")]
pub use media_scene as scene;
#[cfg(feature = "text")]
pub use media_text as text;

#[cfg(feature = "render")]
pub use media_render::{RenderStats, render_project};
#[cfg(feature = "image-pipeline")]
pub use media_render::{ImageOperation, ProcessedImage, process_image};
pub use media_types::Error;
