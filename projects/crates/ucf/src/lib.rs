//! Unified Compute Fabric — public entry point.
//!
//! ```rust
//! use ucf::prelude::*;
//! ```

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
pub use ucf_ir::{DepEdge, Error as IrError, Graph, ResourceGraph, Result as IrResult, TaskGraph};

// Capability / optimization
pub use ucf_capability::{Feature, FeatureSet};
pub use ucf_optimize::{apply, Optimization};

// Scheduler / runtime
pub use ucf_scheduler::{Backend, Error as SchedulerError, Result as SchedulerResult, Scheduler};
pub use ucf_runtime::{CapacityPolicy, Runtime};

#[cfg(feature = "cuda")]
pub use ucf_backend_cuda as backend_cuda;

#[cfg(feature = "dx12")]
pub use ucf_backend_dx12 as backend_dx12;

/// Common imports for application code.
pub mod prelude {
    pub use crate::{
        apply, Access, Backend, CapacityPolicy, DepEdge, DepKind, Dispatch, Domain, Feature,
        FeatureSet, Graph, Objective, Optimization, Priority, ResourceGraph, ResourceId,
        ResourceKind, ResourceNode, Runtime, Scheduler, ShaderId, TaskGraph, TaskId, TaskKind,
        TaskNode,
    };

    #[cfg(feature = "cuda")]
    pub use crate::backend_cuda::CudaBackend;

    #[cfg(feature = "dx12")]
    pub use crate::backend_dx12::Dx12Backend;
}
