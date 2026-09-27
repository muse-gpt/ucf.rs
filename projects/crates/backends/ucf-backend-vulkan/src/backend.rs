use ucf_capability::{Feature, FeatureSet};
use ucf_ir::Graph;
use ucf_scheduler::{Backend, Result};

pub struct VulkanBackend;

impl VulkanBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for VulkanBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for VulkanBackend {
    fn name(&self) -> &str {
        "vulkan"
    }

    fn features(&self) -> FeatureSet {
        FeatureSet::new()
            .with(Feature::Bindless)
            .with(Feature::DescriptorBuffer)
            .with(Feature::ShaderObject)
            .with(Feature::PipelineLibrary)
            .with(Feature::DynamicRendering)
            .with(Feature::Synchronization2)
    }

    fn submit(&mut self, graph: &Graph) -> Result<()> {
        graph.validate().map_err(ucf_scheduler::Error::from)?;
        // TODO: degradation chain selection at init time
        Ok(())
    }
}
