use ucf_capability::FeatureSet;
use ucf_ir::{Graph, TaskNode};

use crate::error::Result;

/// Maps UCF IR onto a concrete API (CUDA, DX12, Vulkan, CPU, …).
pub trait Backend: Send + Sync {
    /// Stable backend id used by the scheduler for placement.
    fn name(&self) -> &str;

    /// Declared hardware / API capabilities.
    fn features(&self) -> FeatureSet;

    /// Allocate or bind graph resources before tasks run.
    fn prepare(&mut self, _graph: &Graph) -> Result<()> {
        Ok(())
    }

    /// Execute one task after dependencies have completed.
    fn submit_task(&mut self, graph: &Graph, task: &TaskNode) -> Result<()>;
}
