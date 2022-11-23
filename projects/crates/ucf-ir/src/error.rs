use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("dependency cycle detected")]
    Cycle,
    #[error("unknown resource: {0}")]
    UnknownResource(u64),
    #[error("unknown task: {0}")]
    UnknownTask(u64),
    #[error("serialization failed: {0}")]
    Serde(String),
}
