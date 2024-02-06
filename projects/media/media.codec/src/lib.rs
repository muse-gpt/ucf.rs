//! Codecs: host decode/encode through WIC and Media Foundation, plus the
//! PNG and WAV file writers.
//!
//! The `svg` feature adds the SVG writer. It consumes the normalized vector
//! representation from `media.vector` one-way — the vector crate never
//! imports this codec — so both export paths read the same resolved model.

pub mod decode;
pub mod encode;
pub mod png;
#[cfg(feature = "svg")]
pub mod svg;
pub mod wav;

pub use decode::{DecodeLimits, MediaLibrary};
pub use encode::Encoder;
pub use png::write_png;
#[cfg(feature = "svg")]
pub use svg::{to_svg_string, write_svg};
pub use wav::{WavPcm16, parse_wav_pcm16, write_wav};
