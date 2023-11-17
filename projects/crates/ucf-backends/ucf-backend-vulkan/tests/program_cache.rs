use std::collections::BTreeMap;
use ucf_backend_vulkan::VulkanBackend;
use ucf_ir::{Dispatch, Graph, ResourceGraph, TaskGraph};
use ucf_program::{BinOp, FunctionBuilder, Program, ProgramBuilder, Ty};
use ucf_scheduler::Backend;
use ucf_types::{Access, Domain, Objective, ParamValue, Priority, ResourceId, ResourceKind, ResourceNode, ShaderId, TaskId, TaskKind, TaskNode};

fn program(increment: f32) -> Program {
    let mut builder = FunctionBuilder::new("generic_increment");
    let input = builder.emit_load_uniform(0, Ty::f32()).unwrap();
    let amount = builder.emit_const_f32(increment).unwrap();
    let result = builder.emit_binary(BinOp::Add, input, amount).unwrap();
    builder.return_value(result).unwrap();
    ProgramBuilder::new().entry(builder).unwrap().finish().unwrap()
}

fn graph(program: &Program) -> Graph {
    Graph {
        resources: ResourceGraph { nodes: [1, 2].into_iter().map(|id| ResourceNode {
            id: ResourceId(id), kind: ResourceKind::Buffer, domain: Domain::Vram,
            access: Access::ReadWrite, byte_size: Some(8),
        }).collect() },
        tasks: TaskGraph { nodes: vec![TaskNode {
            id: TaskId(1), kind: TaskKind::Dispatch, shader: ShaderId(1),
            params: BTreeMap::from([
                ("program".into(), ParamValue::Program { version: program.version,
                    hash: ucf_program::program_hash(program), body: serde_json::to_string(program).unwrap() }),
                ("output".into(), ParamValue::I64(1)),
                ("uniform_0".into(), ParamValue::F64(4.0)),
            ]), dispatch: Dispatch::Explicit { x: 2, y: 1, z: 1 },
            objective: Objective::MaxUtilization, priority: Priority::Batch,
        }], edges: vec![] },
    }
}

#[test]
fn cached_program_rebinds_uniforms_output_and_dispatch() {
    let mut backend = VulkanBackend::new().expect("native Vulkan device required for cache verification");
    let mut graph = graph(&program(0.5));
    backend.prepare(&graph).unwrap();
    backend.write_f32(ResourceId(2), &[123.0, 123.0]).unwrap();
    backend.run_prepared(&graph).unwrap();
    assert_eq!(backend.read_f32(ResourceId(1)).unwrap(), [4.5, 4.5]);
    let task = &mut graph.tasks.nodes[0];
    task.params.insert("uniform_0".into(), ParamValue::F64(7.0));
    task.params.insert("output".into(), ParamValue::I64(2));
    task.dispatch = Dispatch::Explicit { x: 1, y: 1, z: 1 };
    backend.run_prepared(&graph).unwrap();
    assert_eq!(backend.read_f32(ResourceId(2)).unwrap(), [7.5, 123.0]);
    assert_eq!(backend.read_f32(ResourceId(1)).unwrap(), [4.5, 4.5]);
    let stats = backend.program_cache_stats();
    assert_eq!((stats.hits, stats.misses, stats.lowerings, stats.entries), (1, 1, 1, 1));
    graph.tasks.nodes[0].params.insert("output".into(), ParamValue::I64(999));
    assert!(backend.run_prepared(&graph).is_err());
    assert_eq!(backend.read_f32(ResourceId(2)).unwrap(), [7.5, 123.0]);
}

#[test]
fn warm_cache_never_trusts_hash_without_identical_body_and_version() {
    let mut backend = VulkanBackend::new().expect("native Vulkan device required for cache verification");
    let original = program(0.5);
    let mut graph = graph(&original);
    backend.prepare(&graph).unwrap();
    backend.run_prepared(&graph).unwrap();
    for invalid in [
        ParamValue::Program { version: 1, hash: ucf_program::program_hash(&original), body: serde_json::to_string(&program(2.0)).unwrap() },
        ParamValue::Program { version: 1, hash: ucf_program::program_hash(&original) + 1, body: serde_json::to_string(&original).unwrap() },
        ParamValue::Program { version: 2, hash: ucf_program::program_hash(&original), body: serde_json::to_string(&original).unwrap() },
        ParamValue::Program { version: 1, hash: ucf_program::program_hash(&original), body: "not a program".into() },
    ] {
        graph.tasks.nodes[0].params.insert("program".into(), invalid);
        backend.write_f32(ResourceId(1), &[123.0, 123.0]).unwrap();
        assert!(backend.run_prepared(&graph).is_err());
        assert_eq!(backend.read_f32(ResourceId(1)).unwrap(), [123.0, 123.0]);
    }
    assert_eq!(backend.program_cache_stats().lowerings, 1);
    assert_eq!(backend.program_cache_stats().hits, 0);
    graph.tasks.nodes[0].params = self::graph(&program(2.0)).tasks.nodes.remove(0).params;
    backend.run_prepared(&graph).unwrap();
    assert_eq!(backend.read_f32(ResourceId(1)).unwrap(), [6.0, 6.0]);
    assert_eq!(backend.program_cache_stats().lowerings, 2);
}

#[test]
fn program_cache_is_bounded_and_evicts_least_recently_used() {
    let mut backend = VulkanBackend::new().expect("native Vulkan device required for cache verification");
    let mut graph = graph(&program(0.0));
    backend.prepare(&graph).unwrap();
    for increment in 0..16 {
        graph.tasks.nodes[0].params = self::graph(&program(increment as f32)).tasks.nodes.remove(0).params;
        backend.run_prepared(&graph).unwrap();
    }
    graph.tasks.nodes[0].params = self::graph(&program(0.0)).tasks.nodes.remove(0).params;
    backend.run_prepared(&graph).unwrap();
    graph.tasks.nodes[0].params = self::graph(&program(16.0)).tasks.nodes.remove(0).params;
    backend.run_prepared(&graph).unwrap();
    assert_eq!(backend.program_cache_stats().entries, 16);
    assert_eq!(backend.program_cache_stats().hits, 1);
    graph.tasks.nodes[0].params = self::graph(&program(1.0)).tasks.nodes.remove(0).params;
    backend.run_prepared(&graph).unwrap();
    assert_eq!(backend.program_cache_stats().lowerings, 18);
    assert_eq!(backend.read_f32(ResourceId(1)).unwrap(), [5.0, 5.0]);
}
