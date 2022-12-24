//! CUDA Driver API backend (`nvcuda.dll` / `libcuda.so`).
#![warn(missing_docs)]

mod backend;
mod driver;
mod params;

pub use backend::CudaBackend;
