use thiserror::Error;

/// Result alias for IR operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Graph validation and serialization failures.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum Error {
    /// Task dependency cycle detected.
    #[error("dependency cycle detected")]
    Cycle,
    /// Two resources share the same id.
    #[error("duplicate resource id: {0}")]
    DuplicateResource(u64),
    /// Two tasks share the same id.
    #[error("duplicate task id: {0}")]
    DuplicateTask(u64),
    /// Edge references a resource that is not in the graph.
    #[error("unknown resource: {0}")]
    UnknownResource(u64),
    /// Edge references a task that is not in the graph.
    #[error("unknown task: {0}")]
    UnknownTask(u64),
    /// Edge has neither `from_task` nor `from_resource`.
    #[error("dependency edge to task {0} has no source")]
    EmptyEdgeSource(u64),
    /// Serialization / deserialization failed.
    #[error("serialization failed: {0}")]
    Serde(String),
    /// Binary `.ucf` magic bytes are not `UCF\0`.
    #[error("invalid .ucf magic")]
    InvalidMagic,
    /// Binary wire major is unsupported by this build.
    #[error("unsupported .ucf wire major: {found} (supported: {supported})")]
    UnsupportedWireMajor {
        /// Major found in the file.
        found: u32,
        /// Major this crate accepts.
        supported: u32,
    },
    /// Binary payload is truncated or malformed.
    #[error("truncated .ucf binary: {0}")]
    TruncatedBinary(String),
}
