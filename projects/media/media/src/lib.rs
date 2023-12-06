//! Media facade.
//!
//! Re-exports only: the scene IR, raster, text, codec, audio, post and render
//! crates stay independently usable, and downstream products depend on this
//! facade.

pub use media_audio as audio;
pub use media_canvas as canvas;
pub use media_codec as codec;
pub use media_post as post;
pub use media_raster as raster;
pub use media_render as render;
pub use media_scene as scene;
pub use media_text as text;
pub use media_types as types;

pub use media_render::{RenderStats, render_project};
pub use media_types::Error;
