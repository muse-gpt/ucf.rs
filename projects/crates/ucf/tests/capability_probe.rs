//! Soft-skip capability probe: collect `FeatureSet` from backends that open.

use ucf_backend_cpu::CpuBackend;
use ucf_backend_cuda::CudaBackend;
use ucf_backend_dx12::Dx12Backend;
use ucf_backend_rocm::RocmBackend;
use ucf_backend_vulkan::VulkanBackend;
use ucf_capability::CapabilityReport;
use ucf_scheduler::Backend;

#[test]
fn probe_available_backends_into_capability_report() {
    let mut rows: Vec<(String, ucf_capability::FeatureSet)> = Vec::new();

    let cpu = CpuBackend::new();
    rows.push((cpu.name().into(), cpu.features()));

    match Dx12Backend::new() {
        Ok(b) => rows.push((b.name().into(), b.features())),
        Err(err) => eprintln!("skip dx12 capability probe: {err}"),
    }
    match VulkanBackend::new() {
        Ok(b) => rows.push((b.name().into(), b.features())),
        Err(err) => eprintln!("skip vulkan capability probe: {err}"),
    }
    match CudaBackend::new(0) {
        Ok(b) => rows.push((b.name().into(), b.features())),
        Err(err) => eprintln!("skip cuda capability probe: {err}"),
    }
    match RocmBackend::new(0) {
        Ok(b) => {
            let feats = b.features();
            assert!(
                feats.has(ucf_capability::Feature::MatrixCore),
                "ROCm must expose MatrixCore"
            );
            assert!(
                feats.has(ucf_capability::Feature::HipGraph),
                "ROCm must expose HipGraph"
            );
            rows.push((b.name().into(), feats));
        }
        Err(err) => eprintln!("skip rocm capability probe: {err}"),
    }

    let report = CapabilityReport::from_backends(rows);
    assert!(
        report.contains_backend("cpu"),
        "CPU backend must always contribute a row"
    );
    let lines = report.lines();
    assert!(!lines.is_empty());
    for line in &lines {
        eprintln!("{line}");
    }
}
