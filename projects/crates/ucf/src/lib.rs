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
    chain_edges, decode, encode, DepEdge, Error as IrError, Graph, GraphBuilder, ResourceGraph,
    Result as IrResult, TaskGraph, MAGIC as UCF_MAGIC, WIRE_MAJOR as UCF_WIRE_MAJOR,
};

// Capability / optimization
pub use ucf_capability::{
    pick_descriptor_strategy, pick_pipeline_strategy, pick_sync_strategy, BackendCapabilities,
    CapabilityReport, DescriptorStrategy, Feature, FeatureSet, PipelineStrategy, SyncStrategy,
};
pub use ucf_optimize::{
    apply, attention_flops, matmul_flops, AttentionMetrics, BackendPerfRow, MatmulMetrics,
    Optimization, PerfReport, TimedSample,
};

// Scheduler / runtime
pub use ucf_scheduler::{
    fits_120hz_frame, Backend, Error as SchedulerError, NoopBackend, Result as SchedulerResult,
    Scheduler, FRAME_BUDGET_120HZ_MICROS,
};
pub use ucf_runtime::{dry_run, CapacityPolicy, Runtime};

#[cfg(feature = "cpu")]
pub use ucf_backend_cpu as backend_cpu;

#[cfg(feature = "cuda")]
pub use ucf_backend_cuda as backend_cuda;

#[cfg(feature = "dx12")]
pub use ucf_backend_dx12 as backend_dx12;

#[cfg(feature = "vulkan")]
pub use ucf_backend_vulkan as backend_vulkan;

#[cfg(feature = "rocm")]
pub use ucf_backend_rocm as backend_rocm;

/// Common imports for application code.
pub mod prelude {
    pub use crate::{
        apply, attention_flops, decode, encode, fits_120hz_frame, matmul_flops,
        pick_descriptor_strategy, pick_pipeline_strategy, pick_sync_strategy, Access,
        AttentionMetrics, Backend, BackendCapabilities, BackendPerfRow, CapacityPolicy,
        CapabilityReport, chain_edges, DepEdge, DepKind, DescriptorStrategy, Dispatch, Domain,
        dry_run, Feature, FeatureSet, Graph, GraphBuilder, MatmulMetrics, NoopBackend, Objective,
        Optimization, ParamValue, PerfReport, PipelineStrategy, Priority, ResourceGraph,
        ResourceId, ResourceKind, ResourceNode, Runtime, Scheduler, ShaderId, SyncStrategy,
        TaskGraph, TaskId, TaskKind, TaskNode, TimedSample, FRAME_BUDGET_120HZ_MICROS, UCF_MAGIC,
        UCF_WIRE_MAJOR,
    };

    #[cfg(feature = "cpu")]
    pub use crate::backend_cpu::{shared_store, CpuBackend, HostStore};

    #[cfg(feature = "cuda")]
    pub use crate::backend_cuda::CudaBackend;

    #[cfg(feature = "dx12")]
    pub use crate::backend_dx12::Dx12Backend;

    #[cfg(feature = "vulkan")]
    pub use crate::backend_vulkan::VulkanBackend;

    #[cfg(feature = "rocm")]
    pub use crate::backend_rocm::RocmBackend;
}
