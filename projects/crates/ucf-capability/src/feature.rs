use serde::{Deserialize, Serialize};

/// Hardware or API capability exposed by a backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Feature {
    // Compute
    CudaGraph,
    UnifiedMemory,
    TensorCore,
    DynamicParallelism,
    CooperativeGroups,
    ClusterLaunch,
    // Graphics
    Bindless,
    DescriptorHeap,
    DescriptorBuffer,
    ShaderObject,
    PipelineLibrary,
    DynamicRendering,
    RayTracing,
    WorkGraphs,
    WorkLists,
    // Sync
    EnhancedBarriers,
    Synchronization2,
}
