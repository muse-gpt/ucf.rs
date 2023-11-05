//! 程序精度能力不等同于物理设备支持的精度。

use ucf_backend_vulkan::VulkanBackend;
use ucf_capability::Feature;
use ucf_scheduler::Backend;
use std::collections::BTreeMap;
use ucf_ir::{Dispatch, Graph, ResourceGraph, TaskGraph};
use ucf_types::{Access, Domain, Objective, ParamValue, Priority, ResourceId, ResourceKind, ResourceNode, ShaderId, TaskId, TaskKind, TaskNode};

#[test]
fn native_program_advertises_only_implemented_precision() {
    let backend = VulkanBackend::new().expect("native Vulkan device required for precision verification");
    let features = backend.features();
    assert!(features.has(Feature::ProgramFloat32));
    assert!(!features.has(Feature::ProgramFloat64));
}

#[test]
fn unsupported_float64_program_fails_without_writing_output() {
    let mut builder = ucf_program::FunctionBuilder::new("precision_probe");
    let value = builder.emit_const_f64(16777217.0).unwrap();
    builder.return_value(value).unwrap();
    let program = ucf_program::ProgramBuilder::new().entry(builder).unwrap().finish().unwrap();
    let graph = Graph {
        resources: ResourceGraph { nodes: vec![ResourceNode { id: ResourceId(1), kind: ResourceKind::Buffer,
            domain: Domain::Vram, access: Access::ReadWrite, byte_size: Some(4) }] },
        tasks: TaskGraph { nodes: vec![TaskNode { id: TaskId(1), kind: TaskKind::Dispatch,
            shader: ShaderId(1), params: BTreeMap::from([
                ("program".into(), ParamValue::Program { version: program.version,
                    hash: ucf_program::program_hash(&program), body: serde_json::to_string(&program).unwrap() }),
                ("output".into(), ParamValue::I64(1)),
            ]), dispatch: Dispatch::Explicit { x: 1, y: 1, z: 1 },
            objective: Objective::MaxUtilization, priority: Priority::Batch }], edges: vec![] },
    };
    graph.validate().unwrap();
    let mut backend = VulkanBackend::new().expect("native Vulkan device required for precision verification");
    backend.prepare(&graph).unwrap();
    backend.write_f32(ResourceId(1), &[123.0]).unwrap();
    let error = backend.run_prepared(&graph).expect_err("F64 must not be silently truncated");
    assert!(error.to_string().contains("unsupported UCF GPU type"), "error={error}");
    assert_eq!(backend.read_f32(ResourceId(1)).unwrap(), [123.0]);
}
