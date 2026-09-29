use thiserror::Error;
use ucf_ir::Domain;

pub type Result<T> = std::result::Result<T, Error>;

/// Stable application-facing error class (downstream adapters map this, not strings).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    /// Graph failed [`ucf_ir::Graph::validate`].
    Ir,
    /// No registered backend matches placement / pin.
    NoBackend,
    /// A backend returned a driver or submit failure (including missing GPU).
    Backend,
    /// Soft capacity policy rejected the graph.
    CapacityExceeded,
}

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

impl Error {
    /// Stable class for adapters (inspect / debugger), independent of message text.
    pub fn code(&self) -> ErrorCode {
        match self {
            Error::Ir(_) => ErrorCode::Ir,
            Error::NoBackend(_) => ErrorCode::NoBackend,
            Error::Backend(_, _) => ErrorCode::Backend,
            Error::CapacityExceeded { .. } => ErrorCode::CapacityExceeded,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucf_ir::{Access, Domain, ResourceGraph, ResourceId, ResourceKind, ResourceNode, TaskGraph};

    #[test]
    fn error_code_covers_variants() {
        let ir = Error::Ir(ucf_ir::Error::UnknownResource(9));
        assert_eq!(ir.code(), ErrorCode::Ir);
        assert_eq!(
            Error::NoBackend("cpu".into()).code(),
            ErrorCode::NoBackend
        );
        assert_eq!(
            Error::Backend("cuda".into(), "missing".into()).code(),
            ErrorCode::Backend
        );
        assert_eq!(
            Error::CapacityExceeded {
                domain: Domain::Vram,
                used: 2,
                limit: 1,
            }
            .code(),
            ErrorCode::CapacityExceeded
        );
    }

    #[test]
    fn prepare_rejects_invalid_graph() {
        use crate::Scheduler;
        let graph = ucf_ir::Graph {
            resources: ResourceGraph {
                nodes: vec![ResourceNode {
                    id: ResourceId(1),
                    kind: ResourceKind::Buffer,
                    domain: Domain::Host,
                    access: Access::ReadWrite,
                    byte_size: Some(4),
                }],
            },
            tasks: TaskGraph {
                nodes: vec![],
                edges: vec![ucf_ir::DepEdge {
                    from_task: None,
                    from_resource: Some(ResourceId(1)),
                    to_task: ucf_ir::TaskId(99),
                    kind: ucf_ir::DepKind::Data,
                }],
            },
        };
        let mut sched = Scheduler::new();
        let err = sched.prepare(&graph).expect_err("unknown task");
        assert_eq!(err.code(), ErrorCode::Ir);
    }
}
