//! Host CPU reference backend (`Copy` / `Fill` / `MatMul`).
#![warn(missing_docs)]

mod external;
mod params;
mod raster_tri;
mod store;
mod backend;

pub use backend::CpuBackend;
pub use external::HostExternalBuffer;
pub use store::{shared_store, HostStore, SharedHostStore};
