use ucf_capability::{Feature, FeatureSet};
use ucf_ir::Graph;
use ucf_scheduler::{Backend, Result};
use ucf_types::TaskNode;

/// Non-Windows stub: DX12 is only available on Windows.
pub struct Dx12Backend;

impl Dx12Backend {
    /// Returns a non-Windows stub backend handle.
    pub fn new() -> Result<Self> {
        Ok(Self)
    }
}

impl Default for Dx12Backend {
    fn default() -> Self {
        Self
    }
}

impl Backend for Dx12Backend {
    fn name(&self) -> &str {
        "dx12"
    }

    fn features(&self) -> FeatureSet {
        FeatureSet::new()
    }

    fn submit_task(&mut self, _graph: &Graph, _task: &TaskNode) -> Result<()> {
        Err(ucf_scheduler::Error::Backend(
            "dx12".into(),
            "DirectX 12 backend requires Windows".into(),
        ))
    }
}
