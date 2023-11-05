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

#[test]
fn program_precision_is_visible_in_capability_report() {
    let report = CapabilityReport::from_backends([
        ("cpu", FeatureSet::new().with(Feature::ProgramFloat32).with(Feature::ProgramFloat64)),
        ("native", FeatureSet::new().with(Feature::ProgramFloat32)),
    ]);
    assert_eq!(report.lines(), vec![
        "cpu: ProgramFloat32, ProgramFloat64".to_string(),
        "native: ProgramFloat32".to_string(),
    ]);
}
