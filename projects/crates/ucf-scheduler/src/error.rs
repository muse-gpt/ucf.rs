use thiserror::Error;
use ucf_ir::Domain;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Ir(#[from] ucf_ir::Error),
    #[error("no backend registered for unit `{0}`")]
    NoBackend(String),
    #[error("backend `{0}` failed: {1}")]
    Backend(String, String),
    #[error("capacity exceeded on {domain:?}: used {used} bytes, limit {limit}")]
    CapacityExceeded {
        domain: Domain,
        used: u64,
        limit: u64,
    },
}
