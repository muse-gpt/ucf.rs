//! Unified Compute Fabric — public entry point.
//!
//! ```rust
//! use hardware::prelude::*;
//! ```
#![warn(missing_docs)]

pub use hardware_capability as capability;
pub use hardware_ir as ir;
pub use hardware_optimize as optimize;
pub use hardware_runtime as runtime;
pub use hardware_scheduler as scheduler;
pub use hardware_types as types;

// Shared value types
pub use hardware_types::{
    Access, DepKind, Dispatch, Domain, Objective, ParamValue, Priority, ResourceId,
    ResourceKind, ResourceNode, ShaderId, TaskId, TaskKind, TaskNode,
};

// Graph IR
pub use hardware_ir::{
    chain_edges, decode, encode, DepEdge, Error as IrError, Graph, GraphBuilder, ResourceGraph,
    Result as IrResult, TaskGraph, MAGIC as UCF_MAGIC, WIRE_MAJOR as UCF_WIRE_MAJOR,
};

// Capability / optimization
pub use hardware_capability::{
    pick_descriptor_strategy, pick_pipeline_strategy, pick_sync_strategy, BackendCapabilities,
    CapabilityReport, DescriptorStrategy, Feature, FeatureSet, PipelineStrategy, SyncStrategy,
};
pub use hardware_optimize::{
    apply, attention_flops, matmul_flops, AttentionMetrics, BackendPerfRow, MatmulMetrics,
    Optimization, PerfReport, TimedSample,
};

// Scheduler / runtime
pub use hardware_scheduler::{
    fits_120hz_frame, Backend, Error as SchedulerError, ErrorCode as SchedulerErrorCode,
    ExecEvent, ExecStream, ExecutionBindings, ExternalBuffer, ImmediateBridge, ImmediateEvent,
    ImmediateStream, NoopBackend, Result as SchedulerResult, Scheduler, StreamEventBridge,
    FRAME_BUDGET_120HZ_MICROS,
};
pub use hardware_runtime::{
    dry_run, event_kinds, CapacityPolicy, ExecutionDiagnostics, ExecutionEvent,
    ExecutionEventBuilder, Runtime,
};

#[cfg(feature = "cpu")]
mod session;

#[cfg(feature = "cpu")]
pub use session::{immediate_event, immediate_stream, CpuSession};

#[cfg(feature = "cpu")]
pub use hardware_backend_cpu as backend_cpu;

#[cfg(feature = "cuda")]
pub use hardware_backend_cuda as backend_cuda;

#[cfg(feature = "dx12")]
pub use hardware_backend_dx12 as backend_dx12;

#[cfg(feature = "vulkan")]
pub use hardware_backend_vulkan as backend_vulkan;

#[cfg(feature = "rocm")]
pub use hardware_backend_rocm as backend_rocm;

/// Common imports for application code.
pub mod prelude {
    pub use crate::{
        apply, attention_flops, decode, encode, fits_120hz_frame, matmul_flops,
        pick_descriptor_strategy, pick_pipeline_strategy, pick_sync_strategy, Access,
        AttentionMetrics, Backend, BackendCapabilities, BackendPerfRow, CapacityPolicy,
        CapabilityReport, chain_edges, DepEdge, DepKind, DescriptorStrategy, Dispatch, Domain,
        dry_run, event_kinds, ExecutionDiagnostics, ExecutionEvent, ExecutionEventBuilder, Feature,
        FeatureSet, Graph, GraphBuilder, MatmulMetrics, NoopBackend, Objective, Optimization,
        ParamValue, PerfReport, PipelineStrategy, Priority, ResourceGraph, ResourceId, ResourceKind,
        ResourceNode, Runtime, Scheduler, SchedulerErrorCode, ExecutionBindings, ExternalBuffer,
        ImmediateBridge, ImmediateEvent, ImmediateStream, ExecEvent, ExecStream, StreamEventBridge,
        ShaderId, SyncStrategy, TaskGraph, TaskId, TaskKind, TaskNode, TimedSample,
        FRAME_BUDGET_120HZ_MICROS, UCF_MAGIC, UCF_WIRE_MAJOR,
    };

    #[cfg(feature = "cpu")]
    pub use crate::backend_cpu::{shared_store, CpuBackend, HostExternalBuffer, HostStore};

    #[cfg(feature = "cpu")]
    pub use crate::{immediate_event, immediate_stream, CpuSession};

    #[cfg(feature = "cuda")]
    pub use crate::backend_cuda::CudaBackend;

    #[cfg(feature = "dx12")]
    pub use crate::backend_dx12::Dx12Backend;

    #[cfg(feature = "vulkan")]
    pub use crate::backend_vulkan::VulkanBackend;

    #[cfg(feature = "rocm")]
    pub use crate::backend_rocm::RocmBackend;
}
