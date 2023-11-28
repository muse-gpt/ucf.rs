//! Shared media values.

pub mod error;
pub mod hash;

pub use error::Error;
pub use hash::hash;

/// Sample rate of the audio bed. Codecs resample imports into this rate so the
/// mixer never has to care where a clip came from.
pub const SAMPLE_RATE: u32 = 44_100;
