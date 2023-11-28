//! Host codecs: decode through WIC and Media Foundation, encode to MP4.

pub mod decode;
pub mod encode;

pub use decode::MediaLibrary;
pub use encode::Encoder;
