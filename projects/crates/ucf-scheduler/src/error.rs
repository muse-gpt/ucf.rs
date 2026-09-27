use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Ir(#[from] ucf_ir::Error),
    #[error("no backend registered for unit `{0}`")]
    NoBackend(String),
    #[error("backend `{0}` failed: {1}")]
    Backend(String, String),
}
