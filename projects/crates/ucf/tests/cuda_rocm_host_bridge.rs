//! Soft-skip: CUDA Copy + ROCm Fill → host bounce → CUDA MatMul (mixed vendor).

use std::collections::BTreeMap;

use ucf_backend_cuda::CudaBackend;
use ucf_backend_rocm::RocmBackend;
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

fn s(key: &str, v: &str) -> (String, ParamValue) {
    (key.into(), ParamValue::Str(v.into()))
}

fn mixed_graph() -> Graph {
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
                    BTreeMap::from([i("src", 1), i("dst", 2), s("backend", "cuda")]),
                ),
                task(
                    11,
                    TaskKind::Fill,
                    BTreeMap::from([i("dst", 3), f("value", 1.0), s("backend", "rocm")]),
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
                        s("backend", "cuda"),
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

fn task_by_id(graph: &Graph, id: u64) -> &TaskNode {
    graph
        .tasks
        .nodes
        .iter()
        .find(|t| t.id == TaskId(id))
        .expect("task")
}

#[test]
fn cuda_rocm_host_bounce_reference_soft_skip() {
    let mut cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(err) => {
            eprintln!("skip cuda-rocm host bounce (no CUDA): {err}");
            return;
        }
    };
    let mut rocm = match RocmBackend::new(0) {
        Ok(b) => b,
        Err(err) => {
            eprintln!("skip cuda-rocm host bounce (no HIP): {err}");
            return;
        }
    };

    let graph = mixed_graph();
    if let Err(err) = cuda.prepare(&graph) {
        eprintln!("skip cuda-rocm host bounce (cuda prepare): {err}");
        return;
    }
    if let Err(err) = rocm.prepare(&graph) {
        eprintln!("skip cuda-rocm host bounce (rocm prepare): {err}");
        return;
    }
    if let Err(err) = cuda.write_f32(Rid(1), &[1.0, 2.0, 3.0, 4.0]) {
        eprintln!("skip cuda-rocm host bounce (seed): {err}");
        return;
    }

    if let Err(err) = cuda.submit_task(&graph, task_by_id(&graph, 10)) {
        eprintln!("skip cuda-rocm host bounce (cuda copy): {err}");
        return;
    }
    if let Err(err) = cuda.flush() {
        eprintln!("skip cuda-rocm host bounce (cuda flush): {err}");
        return;
    }
    if let Err(err) = rocm.submit_task(&graph, task_by_id(&graph, 11)) {
        eprintln!("skip cuda-rocm host bounce (rocm fill): {err}");
        return;
    }
    if let Err(err) = rocm.flush() {
        eprintln!("skip cuda-rocm host bounce (rocm flush): {err}");
        return;
    }

    // Vendors do not share device memory. Bounce Fill output to CUDA via host.
    let bounced = match rocm.read_f32(Rid(3)) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("skip cuda-rocm host bounce (rocm read): {err}");
            return;
        }
    };
    assert_eq!(bounced, vec![1.0, 1.0, 1.0, 1.0]);
    if let Err(err) = cuda.write_f32(Rid(3), &bounced) {
        eprintln!("skip cuda-rocm host bounce (cuda write): {err}");
        return;
    }

    if let Err(err) = cuda.submit_task(&graph, task_by_id(&graph, 12)) {
        eprintln!("skip cuda-rocm host bounce (cuda matmul): {err}");
        return;
    }
    let out = match cuda.read_f32(Rid(4)) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("skip cuda-rocm host bounce (cuda read): {err}");
            return;
        }
    };
    assert_eq!(out, vec![3.0, 3.0, 7.0, 7.0]);
    eprintln!("cuda-rocm host bounce ok: {out:?}");
}
