//! Host CPU reference backend (`Copy` / `Fill` / `MatMul`).
#![warn(missing_docs)]

mod backend;
mod params;
mod store;

pub use backend::CpuBackend;
pub use store::{shared_store, HostStore, SharedHostStore};
