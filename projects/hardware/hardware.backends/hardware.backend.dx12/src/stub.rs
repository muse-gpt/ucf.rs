use hardware_capability::{FeatureSet, WiredStrategies};
use hardware_ir::Graph;
use hardware_scheduler::{Backend, Result};
use hardware_types::{ResourceId, TaskNode};

/// Opaque id for a DX12 buffer exported via NT shared handle (CUDA import).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SharedBufferId(pub u64);

/// Non-Windows stub: DX12 is only available on Windows.
pub struct Dx12Backend {
    strategies: WiredStrategies,
}

fn unavailable<T>(op: &str) -> Result<T> {
    Err(hardware_scheduler::Error::Backend(
        "dx12".into(),
        format!("DirectX 12 backend requires Windows ({op})"),
    ))
}

impl Dx12Backend {
    /// Non-Windows stub: opening DX12 always fails so soft-skip tests can bail out.
    pub fn new() -> Result<Self> {
        unavailable("open")
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

    /// Allocate a shared buffer (stub: unavailable off Windows).
    pub fn shared_alloc(&mut self, _bytes: usize) -> Result<(SharedBufferId, isize)> {
        unavailable("shared_alloc")
    }

    /// Upload host `f32` values into a shared buffer (stub: unavailable off Windows).
    pub fn shared_write_f32(&mut self, _id: SharedBufferId, _values: &[f32]) -> Result<()> {
        unavailable("shared_write_f32")
    }

    /// NT handle for a shared buffer (stub: unavailable off Windows).
    pub fn shared_nt_handle(&self, _id: SharedBufferId) -> Result<isize> {
        unavailable("shared_nt_handle")
    }

    /// Publish a shared buffer under a graph [`ResourceId`] (stub: unavailable off Windows).
    pub fn publish_shared(&mut self, _resource: ResourceId, _shared: SharedBufferId) -> Result<()> {
        unavailable("publish_shared")
    }

    /// Upload host `f32` values into an allocated device buffer (stub: unavailable off Windows).
    pub fn write_f32(&mut self, _id: ResourceId, _values: &[f32]) -> Result<()> {
        unavailable("write_f32")
    }

    /// Download a device buffer as `f32` values (stub: unavailable off Windows).
    pub fn read_f32(&mut self, _id: ResourceId) -> Result<Vec<f32>> {
        unavailable("read_f32")
    }

    /// Download a device buffer as raw bytes (stub: unavailable off Windows).
    pub fn read_u8(&mut self, _id: ResourceId) -> Result<Vec<u8>> {
        unavailable("read_u8")
    }

    /// Run every task on this backend (stub: unavailable off Windows).
    pub fn run_prepared(&mut self, _graph: &Graph) -> Result<()> {
        unavailable("run_prepared")
    }
}

impl Default for Dx12Backend {
    fn default() -> Self {
        Self {
            strategies: WiredStrategies::from_features(&FeatureSet::new()),
        }
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
        unavailable("submit_task")
    }
}
