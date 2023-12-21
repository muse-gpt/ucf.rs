//! Shared media values.

pub mod error;
pub mod hash;
pub mod time;

pub use error::Error;
pub use hash::hash;
pub use time::TimeBase;

/// Project default audio sample rate. Track plans and codec calls carry the
/// rate explicitly; this constant is only the value the render layer picks
/// when a project does not choose one.
pub const SAMPLE_RATE: u32 = 44_100;
