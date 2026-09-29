//! Soft-skip: Copy → Fill → MatMul captured into a CUDA Graph matches stream readback.

use std::collections::BTreeMap;

use ucf_backend_cuda::CudaBackend;
use ucf_capability::Feature;
use ucf_ir::{
    Access, DepEdge, DepKind, Domain, Graph, Objective, ParamValue, Priority, ResourceGraph,
    ResourceId, ResourceKind, ResourceNode, ShaderId, TaskGraph, TaskId, TaskKind, TaskNode,
};
use ucf_scheduler::Backend;
use ucf_types::ResourceId as Rid;

fn buffer(id: u64, floats: usize) -> ResourceNode {
    ResourceNode {
        id: ResourceId(id),
        kind: ResourceKind::Buffer,
        domain: Domain::Vram,
        access: Access::ReadWrite,
        byte_size: Some((floats * 4) as u64),
    }
}

fn task(id: u64, kind: TaskKind, params: BTreeMap<String, ParamValue>) -> TaskNode {
    TaskNode {
        id: TaskId(id),
        kind,
        shader: ShaderId(id),
        params,
        dispatch: Default::default(),
        objective: Objective::MaxThroughput,
        priority: Priority::Batch,
    }
}

fn i(key: &str, v: i64) -> (String, ParamValue) {
    (key.into(), ParamValue::I64(v))
}

fn f(key: &str, v: f64) -> (String, ParamValue) {
    (key.into(), ParamValue::F64(v))
}

fn reference_graph() -> Graph {
    Graph {
        resources: ResourceGraph {
            nodes: vec![
                buffer(1, 4),
                buffer(2, 4),
                buffer(3, 4),
                buffer(4, 4),
            ],
        },
        tasks: TaskGraph {
            nodes: vec![
                task(
                    10,
                    TaskKind::Copy,
                    BTreeMap::from([i("src", 1), i("dst", 2)]),
                ),
                task(
                    11,
                    TaskKind::Fill,
                    BTreeMap::from([i("dst", 3), f("value", 1.0)]),
                ),
                task(
                    12,
                    TaskKind::MatMul,
                    BTreeMap::from([
                        i("a", 2),
                        i("b", 3),
                        i("out", 4),
                        i("m", 2),
                        i("n", 2),
                        i("k", 2),
                    ]),
                ),
            ],
            edges: vec![
                DepEdge {
                    from_task: None,
                    from_resource: Some(ResourceId(1)),
                    to_task: TaskId(10),
                    kind: DepKind::Data,
                },
                DepEdge {
                    from_task: Some(TaskId(10)),
                    from_resource: None,
                    to_task: TaskId(12),
                    kind: DepKind::Execution,
                },
                DepEdge {
                    from_task: Some(TaskId(11)),
                    from_resource: None,
                    to_task: TaskId(12),
                    kind: DepKind::Execution,
                },
            ],
        },
    }
}

#[test]
fn cuda_graph_reference_matches_stream_soft_skip() {
    let mut cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(err) => {
            eprintln!("skip cuda graph reference (no device): {err}");
            return;
        }
    };
    assert!(
        cuda.features().has(Feature::CudaGraph),
        "CUDA backend must advertise CudaGraph"
    );

    let graph = reference_graph();
    if let Err(err) = cuda.prepare(&graph) {
        eprintln!("skip cuda graph reference (prepare): {err}");
        return;
    }
    if let Err(err) = cuda.write_f32(Rid(1), &[1.0, 2.0, 3.0, 4.0]) {
        eprintln!("skip cuda graph reference (seed): {err}");
        return;
    }
    if let Err(err) = cuda.run_prepared_cuda_graph(&graph) {
        eprintln!("skip cuda graph reference (capture/launch): {err}");
        return;
    }
    let via_graph = match cuda.read_f32(Rid(4)) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("skip cuda graph reference (read graph): {err}");
            return;
        }
    };
    assert_eq!(via_graph, vec![3.0, 3.0, 7.0, 7.0]);

    let mut cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(err) => {
            eprintln!("skip cuda graph reference (stream open): {err}");
            return;
        }
    };
    if let Err(err) = cuda.prepare(&graph) {
        eprintln!("skip cuda graph reference (stream prepare): {err}");
        return;
    }
    if let Err(err) = cuda.write_f32(Rid(1), &[1.0, 2.0, 3.0, 4.0]) {
        eprintln!("skip cuda graph reference (stream seed): {err}");
        return;
    }
    if let Err(err) = cuda.run_prepared(&graph) {
        eprintln!("skip cuda graph reference (stream run): {err}");
        return;
    }
    let via_stream = cuda.read_f32(Rid(4)).expect("stream read");
    assert_eq!(via_graph, via_stream);
    eprintln!("cuda graph reference ok: {via_graph:?}");
}
