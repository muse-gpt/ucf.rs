//! Pick a preferred API strategy from an advertised [`FeatureSet`].
//!
//! This is a **selection table**, not a runtime rewrite of pipelines.

use crate::{Feature, FeatureSet};

/// Descriptor binding strategy along the Vulkan / DX bindless degradation chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptorStrategy {
    /// Shader Model 6.6 / bindless heap path.
    ResourceDescriptorHeap,
    /// `VK_EXT_descriptor_buffer` style.
    DescriptorBuffer,
    /// Bindless indexing without a full heap abstraction.
    BindlessIndexing,
    /// Classic descriptor sets / tables.
    TraditionalSets,
}

/// Graphics / compute pipeline object strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStrategy {
    /// Vulkan Shader Objects (or equivalent).
    ShaderObject,
    /// Pipeline libraries / PSO cache sharing.
    PipelineLibrary,
    /// Fully pre-baked PSO cache.
    PsoPrecache,
}

/// Synchronization strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncStrategy {
    /// Enhanced / split barriers (DX12 style).
    EnhancedBarriers,
    /// `VK_KHR_synchronization2`.
    Synchronization2,
    /// Legacy pipeline barriers.
    LegacyBarriers,
}

/// Choose the best descriptor strategy supported by `features`.
pub fn pick_descriptor_strategy(features: &FeatureSet) -> DescriptorStrategy {
    if features.has(Feature::Bindless) && features.has(Feature::DescriptorHeap) {
        DescriptorStrategy::ResourceDescriptorHeap
    } else if features.has(Feature::DescriptorBuffer) {
        DescriptorStrategy::DescriptorBuffer
    } else if features.has(Feature::Bindless) {
        DescriptorStrategy::BindlessIndexing
    } else {
        DescriptorStrategy::TraditionalSets
    }
}

/// Choose the best pipeline strategy supported by `features`.
pub fn pick_pipeline_strategy(features: &FeatureSet) -> PipelineStrategy {
    if features.has(Feature::ShaderObject) {
        PipelineStrategy::ShaderObject
    } else if features.has(Feature::PipelineLibrary) {
        PipelineStrategy::PipelineLibrary
    } else {
        PipelineStrategy::PsoPrecache
    }
}

/// Choose the best sync strategy supported by `features`.
pub fn pick_sync_strategy(features: &FeatureSet) -> SyncStrategy {
    if features.has(Feature::EnhancedBarriers) {
        SyncStrategy::EnhancedBarriers
    } else if features.has(Feature::Synchronization2) {
        SyncStrategy::Synchronization2
    } else {
        SyncStrategy::LegacyBarriers
    }
}
