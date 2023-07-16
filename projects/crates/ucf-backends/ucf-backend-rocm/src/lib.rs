//! ROCm / HIP backend (`amdhip64` + emitter HSACO).
#![warn(missing_docs)]

mod backend;
mod driver;
mod params;

pub use backend::{CapturedHipGraph, RocmBackend};
