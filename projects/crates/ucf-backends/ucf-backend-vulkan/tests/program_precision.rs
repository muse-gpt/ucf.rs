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
    eprintln!("Vulkan ProgramFloat64={}", features.has(Feature::ProgramFloat64));
}

#[test]
fn incompatible_float64_output_layout_fails_without_writing_output() {
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
    assert!(error.to_string().contains("8-byte scalar") || error.to_string().contains("does not support ProgramFloat64"), "error={error}");
    assert_eq!(backend.read_f32(ResourceId(1)).unwrap(), [123.0]);
}

#[test]
fn native_float64_program_preserves_precision_and_uniform_alignment() {
    let mut backend = VulkanBackend::new().expect("native Vulkan device required for precision verification");
    let mut builder = ucf_program::FunctionBuilder::new("precision_probe");
    let initial = builder.emit_load_uniform(0, ucf_program::Ty::f64()).unwrap();
    let increment = builder.emit_load_uniform(1, ucf_program::Ty::f64()).unwrap();
    let result = builder.emit_binary(ucf_program::BinOp::Add, initial, increment).unwrap();
    builder.return_value(result).unwrap();
    let program = ucf_program::ProgramBuilder::new().entry(builder).unwrap().finish().unwrap();
    let graph = Graph {
        resources: ResourceGraph { nodes: vec![ResourceNode { id: ResourceId(1), kind: ResourceKind::Buffer,
            domain: Domain::Vram, access: Access::ReadWrite, byte_size: Some(8) }] },
        tasks: TaskGraph { nodes: vec![TaskNode { id: TaskId(1), kind: TaskKind::Dispatch,
            shader: ShaderId(1), params: BTreeMap::from([
                ("program".into(), ParamValue::Program { version: program.version,
                    hash: ucf_program::program_hash(&program), body: serde_json::to_string(&program).unwrap() }),
                ("output".into(), ParamValue::I64(1)),
                ("uniform_0".into(), ParamValue::F64(16777216.0)),
                ("uniform_1".into(), ParamValue::F64(0.125)),
            ]), dispatch: Dispatch::Explicit { x: 1, y: 1, z: 1 },
            objective: Objective::MaxUtilization, priority: Priority::Batch }], edges: vec![] },
    };
    graph.validate().unwrap();
    backend.prepare(&graph).unwrap();
    backend.write_f64(ResourceId(1), &[123.0]).unwrap();
    if !backend.features().has(Feature::ProgramFloat64) {
        let error = backend.run_prepared(&graph).unwrap_err();
        assert!(error.to_string().contains("does not support ProgramFloat64"));
        assert_eq!(backend.read_f64(ResourceId(1)).unwrap(), [123.0]);
        return;
    }
    backend.run_prepared(&graph).unwrap();
    assert_eq!(backend.read_f64(ResourceId(1)).unwrap(), [16777216.125]);
    assert!(backend.write_f64(ResourceId(1), &[1.0, 2.0]).is_err());
}

#[test]
fn native_float64_buffers_loops_and_sqrt_match_interpreter() {
    use ucf_program::{BinOp, Expr, ProgramBuilder, StructuredFunctionBuilder, Ty, UnOp};
    let mut builder = StructuredFunctionBuilder::new("buffer_precision");
    let value = builder.let_value(Ty::f64(), builder.load_buffer(1, builder.global_id()));
    builder.bounded_for(4, |builder, _| {
        builder.store_local(value, builder.binary(BinOp::Add, Expr::LoadLocal(value), builder.const_f64(0.125)));
    });
    let result = builder.binary(BinOp::Add, Expr::LoadLocal(value), builder.unary(UnOp::Sqrt, builder.const_f64(9.0)));
    builder.store_buffer(2, builder.global_id(), result.clone());
    let program = ProgramBuilder::new().structured_entry(builder, Ty::f64(), result).unwrap().finish().unwrap();
    let graph = Graph {
        resources: ResourceGraph { nodes: [(1, Access::Write), (2, Access::Read), (3, Access::Write)].into_iter()
            .map(|(id, access)| ResourceNode { id: ResourceId(id), kind: ResourceKind::Buffer,
                domain: Domain::Vram, access, byte_size: Some(16) }).collect() },
        tasks: TaskGraph { nodes: vec![TaskNode { id: TaskId(2), kind: TaskKind::Dispatch,
            shader: ShaderId(2), params: BTreeMap::from([
                ("program".into(), ParamValue::Program { version: program.version,
                    hash: ucf_program::program_hash(&program), body: serde_json::to_string(&program).unwrap() }),
                ("output".into(), ParamValue::I64(1)), ("buffer_0".into(), ParamValue::I64(2)),
                ("buffer_1".into(), ParamValue::I64(2)), ("buffer_2".into(), ParamValue::I64(3)),
            ]), dispatch: Dispatch::Explicit { x: 2, y: 1, z: 1 },
            objective: Objective::MaxUtilization, priority: Priority::Batch }], edges: vec![] },
    };
    graph.validate().unwrap();
    let mut backend = VulkanBackend::new().expect("native Vulkan device required for precision verification");
    backend.prepare(&graph).unwrap();
    backend.write_f64(ResourceId(2), &[16777216.0, 1.0e9 + 0.25]).unwrap();
    if !backend.features().has(Feature::ProgramFloat64) {
        assert!(backend.run_prepared(&graph).unwrap_err().to_string().contains("does not support ProgramFloat64"));
        return;
    }
    backend.run_prepared(&graph).unwrap();
    assert_eq!(backend.read_f64(ResourceId(3)).unwrap(), [16777219.5, 1.0e9 + 3.75]);
    assert_eq!(backend.read_f64(ResourceId(1)).unwrap(), backend.read_f64(ResourceId(3)).unwrap());
    let mut interpreter = ucf_program::CpuInterpreter::new();
    interpreter.with_buffer(1, vec![ucf_program::Value::F64(16777216.0), ucf_program::Value::F64(1.0e9 + 0.25)]);
    interpreter.with_buffer(2, vec![ucf_program::Value::F64(0.0); 2]);
    interpreter.run_parallel(&program, 2).unwrap();
    assert_eq!(interpreter.buffer(2).unwrap(), &[ucf_program::Value::F64(16777219.5), ucf_program::Value::F64(1.0e9 + 3.75)]);
}
