//! Codecs: host decode/encode through WIC and Media Foundation, plus the
//! PNG and WAV file writers.
//!
//! The `svg` feature adds the SVG writer. It consumes the normalized vector
//! representation from `media.vector` one-way — the vector crate never
//! imports this codec — so both export paths read the same resolved model.
//! The `obj` feature adds the OBJ writer. It consumes the exposed-face mesh
//! view from `media.voxel` one-way, for the same reason: the artifact and
//! the isometric preview are driven by the one face list.

pub mod decode;
pub mod encode;
#[cfg(feature = "obj")]
pub mod obj;
pub mod png;
#[cfg(feature = "svg")]
pub mod svg;
pub mod wav;

pub use decode::{DecodeLimits, MediaLibrary};
pub use encode::Encoder;
#[cfg(feature = "obj")]
pub use obj::{to_obj_string, write_obj};
pub use png::write_png;
#[cfg(feature = "svg")]
pub use svg::{to_svg_string, write_svg};
pub use wav::{WavPcm16, parse_wav_pcm16, write_wav};
