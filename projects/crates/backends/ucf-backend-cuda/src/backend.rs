use ucf_capability::{Feature, FeatureSet};
use ucf_ir::Graph;
use ucf_scheduler::{Backend, Result};

/// CUDA backend skeleton. Real driver integration lands in stage 1.
pub struct CudaBackend {
    device_index: u32,
}

impl CudaBackend {
    pub fn new(device_index: u32) -> Self {
        Self { device_index }
    }
}

impl Backend for CudaBackend {
    fn name(&self) -> &str {
        "cuda"
    }

    fn features(&self) -> FeatureSet {
        FeatureSet::new()
            .with(Feature::CudaGraph)
            .with(Feature::UnifiedMemory)
            .with(Feature::TensorCore)
            .with(Feature::DynamicParallelism)
            .with(Feature::CooperativeGroups)
            .with(Feature::ClusterLaunch)
    }

    fn submit(&mut self, graph: &Graph) -> Result<()> {
        graph.validate().map_err(ucf_scheduler::Error::from)?;
        let _ = self.device_index;
        // TODO: map TaskGraph -> CUDA Graph / kernel launches
        Ok(())
    }
}
