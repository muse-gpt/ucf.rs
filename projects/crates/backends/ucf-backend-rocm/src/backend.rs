use ucf_capability::{Feature, FeatureSet};
use ucf_ir::Graph;
use ucf_scheduler::{Backend, Result};

pub struct RocmBackend {
    device_index: u32,
}

impl RocmBackend {
    pub fn new(device_index: u32) -> Self {
        Self { device_index }
    }
}

impl Backend for RocmBackend {
    fn name(&self) -> &str {
        "rocm"
    }

    fn features(&self) -> FeatureSet {
        FeatureSet::new()
            .with(Feature::UnifiedMemory)
            .with(Feature::TensorCore)
    }

    fn submit(&mut self, graph: &Graph) -> Result<()> {
        graph.validate().map_err(ucf_scheduler::Error::from)?;
        let _ = self.device_index;
        // TODO: HIP graph / kernel launches
        Ok(())
    }
}
