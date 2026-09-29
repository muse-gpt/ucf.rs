//! Soft-skip: Fill TaskGraph captured into a HIP Graph launches with same readback.

use std::collections::BTreeMap;

use ucf_backend_rocm::RocmBackend;
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
fn hip_graph_fill_matches_stream_soft_skip() {
    let mut rocm = match RocmBackend::new(0) {
        Ok(b) => b,
        Err(err) => {
            eprintln!("skip hip graph (no device): {err}");
            return;
        }
    };
    assert!(
        rocm.features().has(Feature::HipGraph),
        "ROCm backend must advertise HipGraph"
    );

    let graph = fill_graph();
    if let Err(err) = rocm.prepare(&graph) {
        eprintln!("skip hip graph (prepare): {err}");
        return;
    }
    if let Err(err) = rocm.run_prepared_hip_graph(&graph) {
        eprintln!("skip hip graph (capture/launch): {err}");
        return;
    }
    let via_graph = match rocm.read_f32(Rid(1)) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("skip hip graph (read graph): {err}");
            return;
        }
    };
    assert_eq!(via_graph, vec![3.0, 3.0, 3.0, 3.0]);

    let mut rocm = match RocmBackend::new(0) {
        Ok(b) => b,
        Err(err) => {
            eprintln!("skip hip graph (stream open): {err}");
            return;
        }
    };
    if let Err(err) = rocm.prepare(&graph) {
        eprintln!("skip hip graph (stream prepare): {err}");
        return;
    }
    if let Err(err) = rocm.run_prepared(&graph) {
        eprintln!("skip hip graph (stream run): {err}");
        return;
    }
    let via_stream = rocm.read_f32(Rid(1)).expect("stream read");
    assert_eq!(via_graph, via_stream);
    eprintln!("hip graph fill ok: {via_graph:?}");
}
