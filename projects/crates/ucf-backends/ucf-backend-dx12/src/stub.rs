use ucf_capability::{FeatureSet, WiredStrategies};
use ucf_ir::Graph;
use ucf_scheduler::{Backend, Result};
use ucf_types::TaskNode;

/// Non-Windows stub: DX12 is only available on Windows.
pub struct Dx12Backend {
    strategies: WiredStrategies,
}

impl Dx12Backend {
    /// Returns a non-Windows stub backend handle.
    pub fn new() -> Result<Self> {
        Ok(Self {
            strategies: WiredStrategies::from_features(&FeatureSet::new()),
        })
    }

    /// Degrade-chain strategies resolved when the backend opened.
    pub fn strategies(&self) -> WiredStrategies {
        self.strategies
    }

    /// Descriptor path label (always empty on the stub).
    pub fn last_descriptor_path(&self) -> &'static str {
        ""
    }

    /// Sync path label (always empty on the stub).
    pub fn last_sync_path(&self) -> &'static str {
        ""
    }

    /// Pipeline path label (always empty on the stub).
    pub fn last_pipeline_path(&self) -> &'static str {
        ""
    }
}

impl Default for Dx12Backend {
    fn default() -> Self {
        Self::new().expect("dx12 stub")
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
