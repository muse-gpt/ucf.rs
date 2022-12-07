//! CUDA Driver API backend (`nvcuda.dll`).
#![warn(missing_docs)]

mod backend;
mod driver;

pub use backend::CudaBackend;
