//! ROCm / HIP backend (`amdhip64` + `hiprtc`).
#![warn(missing_docs)]

mod backend;
mod driver;
mod params;

pub use backend::RocmBackend;
