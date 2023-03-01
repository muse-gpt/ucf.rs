use ucf_capability::{CapabilityReport, Feature, FeatureSet};

#[test]
fn capability_report_formats_sorted_features() {
    let report = CapabilityReport::from_backends([
        ("cpu", FeatureSet::new()),
        (
            "dx12",
            FeatureSet::new()
                .with(Feature::Bindless)
                .with(Feature::DynamicRendering),
        ),
    ]);
    assert!(report.contains_backend("cpu"));
    assert!(report.contains_backend("dx12"));
    assert_eq!(
        report.lines(),
        vec![
            "cpu: (none)".to_string(),
            "dx12: Bindless, DynamicRendering".to_string(),
        ]
    );
}
