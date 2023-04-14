use ucf_optimize::{BackendPerfRow, PerfReport, TimedSample};

#[test]
fn perf_report_lines_and_ratio() {
    let report = PerfReport::from_rows([
        BackendPerfRow {
            backend: "dx12".into(),
            samples: vec![
                TimedSample::new("matmul", 2_000_000),
                TimedSample::new("raster_clear", 1_000_000),
            ],
        },
        BackendPerfRow {
            backend: "vulkan".into(),
            samples: vec![
                TimedSample::new("matmul", 4_000_000),
                TimedSample::new("raster_clear", 1_500_000),
            ],
        },
    ]);
    assert_eq!(report.sample_nanos("dx12", "matmul"), Some(2_000_000));
    assert_eq!(report.ratio("vulkan", "dx12", "matmul"), Some(2.0));
    let lines = report.lines();
    assert_eq!(lines.len(), 4);
    assert!(lines[0].starts_with("dx12/matmul:"));
}
