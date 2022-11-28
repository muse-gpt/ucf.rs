use ucf_capability::FeatureSet;
use ucf_ir::{Graph, TaskNode};

use crate::error::Result;

/// Maps UCF IR onto a concrete API (CUDA, DX12, Vulkan, …).
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;

    fn features(&self) -> FeatureSet;

    fn submit_task(&mut self, graph: &Graph, task: &TaskNode) -> Result<()>;
}
