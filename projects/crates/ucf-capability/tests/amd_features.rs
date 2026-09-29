//! AMD-oriented `Feature` flags for ROCm capability advertising.

use ucf_capability::{Feature, FeatureSet};

/// Mirror of `RocmBackend::features` for unit assertion when HIP is absent.
fn rocm_feature_set() -> FeatureSet {
    FeatureSet::new()
        .with(Feature::UnifiedMemory)
        .with(Feature::MatrixCore)
        .with(Feature::HipGraph)
}

#[test]
fn rocm_feature_set_exposes_amd_matrix_and_hip_graph() {
    let feats = rocm_feature_set();
    assert!(feats.has(Feature::MatrixCore));
    assert!(feats.has(Feature::HipGraph));
    assert!(feats.has(Feature::UnifiedMemory));
    assert!(!feats.has(Feature::CudaGraph));
    assert!(!feats.has(Feature::TensorCore));
}
