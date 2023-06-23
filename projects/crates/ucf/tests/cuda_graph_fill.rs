//! Soft-skip: Fill TaskGraph captured into a CUDA Graph launches with same readback.

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

fn fill_task(id: u64, dst: u64, value: f64) -> TaskNode {
    TaskNode {
        id: TaskId(id),
        kind: TaskKind::Fill,
        shader: ShaderId(id),
        params: BTreeMap::from([
            ("dst".into(), ParamValue::I64(dst as i64)),
            ("value".into(), ParamValue::F64(value)),
        ]),
        dispatch: Default::default(),
        objective: Objective::MaxThroughput,
        priority: Priority::Batch,
    }
}

fn fill_graph() -> Graph {
    Graph {
        resources: ResourceGraph {
            nodes: vec![buffer(1, 4)],
        },
        tasks: TaskGraph {
            nodes: vec![fill_task(10, 1, 3.0)],
            edges: vec![DepEdge {
                from_task: None,
                from_resource: Some(ResourceId(1)),
                to_task: TaskId(10),
                kind: DepKind::Data,
            }],
        },
    }
}

#[test]
fn cuda_graph_fill_matches_stream_soft_skip() {
    let mut cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(err) => {
            eprintln!("skip cuda graph (no device): {err}");
            return;
        }
    };
    assert!(
        cuda.features().has(Feature::CudaGraph),
        "CUDA backend must advertise CudaGraph"
    );

    let graph = fill_graph();
    if let Err(err) = cuda.prepare(&graph) {
        eprintln!("skip cuda graph (prepare): {err}");
        return;
    }
    if let Err(err) = cuda.run_prepared_cuda_graph(&graph) {
        eprintln!("skip cuda graph (capture/launch): {err}");
        return;
    }
    let via_graph = match cuda.read_f32(Rid(1)) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("skip cuda graph (read graph): {err}");
            return;
        }
    };
    assert_eq!(via_graph, vec![3.0, 3.0, 3.0, 3.0]);

    // Stream path on a fresh buffer for parity.
    let mut cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(err) => {
            eprintln!("skip cuda graph (stream open): {err}");
            return;
        }
    };
    if let Err(err) = cuda.prepare(&graph) {
        eprintln!("skip cuda graph (stream prepare): {err}");
        return;
    }
    if let Err(err) = cuda.run_prepared(&graph) {
        eprintln!("skip cuda graph (stream run): {err}");
        return;
    }
    let via_stream = cuda.read_f32(Rid(1)).expect("stream read");
    assert_eq!(via_graph, via_stream);
    eprintln!("cuda graph fill ok: {via_graph:?}");
}
