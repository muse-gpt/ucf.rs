use std::collections::BTreeMap;

use hardware_backend_cpu::CpuBackend;
use hardware_ir::{
    Access, DepEdge, DepKind, Domain, Graph, Objective, ParamValue, Priority, ResourceGraph,
    ResourceId, ResourceKind, ResourceNode, ShaderId, TaskGraph, TaskId, TaskKind, TaskNode,
};
use hardware_scheduler::Backend;
use hardware_types::ResourceId as Rid;

fn buffer(id: u64, bytes: u64) -> ResourceNode {
    ResourceNode {
        id: ResourceId(id),
        kind: ResourceKind::Buffer,
        domain: Domain::Host,
        access: Access::ReadWrite,
        byte_size: Some(bytes),
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

fn raster_graph(width: i64, height: i64) -> Graph {
    let bytes = (width * height * 4) as u64;
    Graph {
        resources: ResourceGraph {
            nodes: vec![buffer(1, bytes)],
        },
        tasks: TaskGraph {
            nodes: vec![task(
                10,
                TaskKind::Raster,
                BTreeMap::from([
                    i("dst", 1),
                    i("width", width),
                    i("height", height),
                    f("r", 1.0),
                    f("g", 0.0),
                    f("b", 0.0),
                    f("a", 1.0),
                ]),
            )],
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
fn cpu_raster_clear_readback_rgba8() {
    let width = 4i64;
    let height = 2i64;
    let graph = raster_graph(width, height);
    let mut cpu = CpuBackend::new();
    cpu.prepare(&graph).expect("prepare");
    cpu.submit_task(&graph, &graph.tasks.nodes[0])
        .expect("raster clear");

    let store = cpu.store();
    let guard = store.lock().expect("lock");
    let bytes = guard.bytes(Rid(1)).expect("readback").to_vec();
    assert_eq!(bytes.len(), (width * height * 4) as usize);
    assert!(bytes.iter().all(|&b| b == 255 || b == 0));
    assert_eq!(&bytes[0..4], &[255, 0, 0, 255]);
}
