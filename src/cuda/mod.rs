//! CUDA backend for the Spectral Sieve alignment fast path.

#[cfg(feature = "cuda")]
mod backend;
#[cfg(feature = "cuda")]
pub mod dispatcher;
#[cfg(feature = "cuda")]
mod server;

#[cfg(feature = "cuda")]
pub use backend::{CudaBackend, CudaJob, CudaResult};
#[cfg(feature = "cuda")]
pub use server::run_gpu_server;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaError {
    #[error("CUDA support is not built into this binary (rebuild with `--features cuda`)")]
    NotBuilt,
    #[cfg(feature = "cuda")]
    #[error("CUDA driver error: {0}")]
    Driver(String),
    #[cfg(feature = "cuda")]
    #[error("CUDA runtime error: {0}")]
    Runtime(String),
    #[cfg(feature = "cuda")]
    #[error("no CUDA-capable device found")]
    NoDevice,
    #[cfg(feature = "cuda")]
    #[error("read too long for the CUDA backend ({len} > {max} bytes packed)")]
    ReadTooLong { len: usize, max: usize },
}

/// Maximum packed-byte length of a read the GPU backend can handle.
#[cfg(feature = "cuda")]
pub const READ_BYTES_MAX: usize = 64;

/// `true` when this binary was compiled with `--features cuda` *and* a CUDA-capable device was.
pub fn is_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        backend::cuda_runtime_available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Stub for builds without `--features cuda`: never returns Ok.
#[cfg(not(feature = "cuda"))]
pub fn run_gpu_server(_threads: usize, _batch_bases: usize) -> Result<(), CudaError> {
    Err(CudaError::NotBuilt)
}
