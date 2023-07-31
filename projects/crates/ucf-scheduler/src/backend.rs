use ucf_capability::FeatureSet;
use ucf_ir::{Graph, TaskNode};

use crate::error::Result;
use crate::exec::{ExecStream, ExecutionBindings, StreamEventBridge};

/// Maps UCF IR onto a concrete API (CUDA, DX12, Vulkan, CPU, …).
pub trait Backend: Send + Sync {
    /// Stable backend id used by the scheduler for placement.
    fn name(&self) -> &str;

    /// Declared hardware / API capabilities.
    fn features(&self) -> FeatureSet;

    /// Adopt external buffers / preferred stream from a downstream adapter.
    ///
    /// Default: ignore bindings (backend allocates privately). CPU / CUDA
    /// backends override to honor DXO / Spark device-resident objects.
    fn bind_externals(&mut self, _bindings: &ExecutionBindings) -> Result<()> {
        Ok(())
    }

    /// Allocate or bind graph resources before tasks run.
    fn prepare(&mut self, _graph: &Graph) -> Result<()> {
        Ok(())
    }

    /// Execute one task after dependencies have completed.
    fn submit_task(&mut self, graph: &Graph, task: &TaskNode) -> Result<()>;

    /// Block until prior submits on this backend are visible to other backends.
    ///
    /// Called by the scheduler when the next task is placed on a different backend.
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    /// Optional stream/event bridge for cross-stream dependencies.
    fn stream_bridge(&self) -> Option<&dyn StreamEventBridge> {
        None
    }

    /// Stream currently preferred for submits (if any).
    fn active_stream(&self) -> Option<&dyn ExecStream> {
        None
    }
}
