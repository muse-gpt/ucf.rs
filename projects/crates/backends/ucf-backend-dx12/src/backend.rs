use ucf_capability::{Feature, FeatureSet};
use ucf_ir::{Graph, TaskNode};
use ucf_scheduler::{Backend, Result};

pub struct Dx12Backend;

impl Dx12Backend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Dx12Backend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for Dx12Backend {
    fn name(&self) -> &str {
        "dx12"
    }

    fn features(&self) -> FeatureSet {
        FeatureSet::new()
            .with(Feature::Bindless)
            .with(Feature::DescriptorHeap)
            .with(Feature::DynamicRendering)
            .with(Feature::RayTracing)
            .with(Feature::WorkLists)
            .with(Feature::EnhancedBarriers)
    }

    fn submit_task(&mut self, graph: &Graph, _task: &TaskNode) -> Result<()> {
        graph.validate().map_err(ucf_scheduler::Error::from)?;
        // TODO: descriptor heap, dynamic state, raster/compute queues
        Ok(())
    }
}
