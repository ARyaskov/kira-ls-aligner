#[cfg(feature = "cuda")]
use thiserror::Error;

#[cfg(feature = "cuda")]
#[derive(Debug, Error)]
pub enum CudaError {
    #[error("CUDA support is not built in this build")]
    NotBuilt,
}

#[cfg(feature = "cuda")]
pub fn available() -> bool {
    false
}

#[cfg(feature = "cuda")]
pub fn align_batch() -> Result<(), CudaError> {
    Err(CudaError::NotBuilt)
}
