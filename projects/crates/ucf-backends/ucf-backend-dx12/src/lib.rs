//! Direct3D 12 compute backend.
#![warn(missing_docs)]

#[cfg(windows)]
mod win;

#[cfg(not(windows))]
mod stub;

#[cfg(windows)]
pub use win::Dx12Backend;

#[cfg(not(windows))]
pub use stub::Dx12Backend;
