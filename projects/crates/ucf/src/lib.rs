//! Unified Compute Fabric — public entry point.
//!
//! ```rust
//! use ucf::prelude::*;
//! ```
#![warn(missing_docs)]

pub use ucf_capability as capability;
pub use ucf_ir as ir;
pub use ucf_optimize as optimize;
pub use ucf_runtime as runtime;
pub use ucf_scheduler as scheduler;
pub use ucf_types as types;

// Shared value types
pub use ucf_types::{
    Access, DepKind, Dispatch, Domain, Objective, ParamValue, Priority, ResourceId,
    ResourceKind, ResourceNode, ShaderId, TaskId, TaskKind, TaskNode,
};

// Graph IR
pub use ucf_ir::{
    chain_edges, DepEdge, Error as IrError, Graph, GraphBuilder, ResourceGraph, Result as IrResult,
    TaskGraph,
};

// Capability / optimization
pub use ucf_capability::{Feature, FeatureSet};
pub use ucf_optimize::{apply, Optimization};

// Scheduler / runtime
pub use ucf_scheduler::{
    Backend, Error as SchedulerError, NoopBackend, Result as SchedulerResult, Scheduler,
};
pub use ucf_runtime::{dry_run, CapacityPolicy, Runtime};

#[cfg(feature = "cuda")]
pub use ucf_backend_cuda as backend_cuda;

#[cfg(feature = "dx12")]
pub use ucf_backend_dx12 as backend_dx12;

/// Common imports for application code.
pub mod prelude {
    pub use crate::{
        apply, Access, Backend, CapacityPolicy, chain_edges, DepEdge, DepKind, Dispatch, Domain,
        dry_run, Feature, FeatureSet, Graph, GraphBuilder, NoopBackend, Objective, Optimization,
        Priority, ResourceGraph, ResourceId, ResourceKind, ResourceNode, Runtime, Scheduler,
        ShaderId, TaskGraph, TaskId, TaskKind, TaskNode,
    };

    #[cfg(feature = "cuda")]
    pub use crate::backend_cuda::CudaBackend;

    #[cfg(feature = "dx12")]
    pub use crate::backend_dx12::Dx12Backend;
}
