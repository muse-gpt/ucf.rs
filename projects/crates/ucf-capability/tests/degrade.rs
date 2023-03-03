use ucf_capability::{
    pick_descriptor_strategy, pick_pipeline_strategy, pick_sync_strategy, DescriptorStrategy,
    Feature, FeatureSet, PipelineStrategy, SyncStrategy,
};

#[test]
fn descriptor_chain_prefers_heap_then_buffer_then_indexing() {
    let heap = FeatureSet::new()
        .with(Feature::Bindless)
        .with(Feature::DescriptorHeap);
    assert_eq!(
        pick_descriptor_strategy(&heap),
        DescriptorStrategy::ResourceDescriptorHeap
    );

    let buffer = FeatureSet::new().with(Feature::DescriptorBuffer);
    assert_eq!(
        pick_descriptor_strategy(&buffer),
        DescriptorStrategy::DescriptorBuffer
    );

    let indexing = FeatureSet::new().with(Feature::Bindless);
    assert_eq!(
        pick_descriptor_strategy(&indexing),
        DescriptorStrategy::BindlessIndexing
    );

    assert_eq!(
        pick_descriptor_strategy(&FeatureSet::new()),
        DescriptorStrategy::TraditionalSets
    );
}

#[test]
fn pipeline_and_sync_chains() {
    assert_eq!(
        pick_pipeline_strategy(&FeatureSet::new().with(Feature::ShaderObject)),
        PipelineStrategy::ShaderObject
    );
    assert_eq!(
        pick_pipeline_strategy(&FeatureSet::new().with(Feature::PipelineLibrary)),
        PipelineStrategy::PipelineLibrary
    );
    assert_eq!(
        pick_pipeline_strategy(&FeatureSet::new()),
        PipelineStrategy::PsoPrecache
    );

    assert_eq!(
        pick_sync_strategy(&FeatureSet::new().with(Feature::EnhancedBarriers)),
        SyncStrategy::EnhancedBarriers
    );
    assert_eq!(
        pick_sync_strategy(&FeatureSet::new().with(Feature::Synchronization2)),
        SyncStrategy::Synchronization2
    );
    assert_eq!(
        pick_sync_strategy(&FeatureSet::new()),
        SyncStrategy::LegacyBarriers
    );
}
