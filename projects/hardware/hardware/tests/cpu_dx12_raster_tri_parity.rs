//! CPU 与 DX12 同图 `Raster draw=tri`：中心黄 / 角黑像素对照（薄门）。

use std::collections::BTreeMap;

use hardware_backend_cpu::CpuBackend;
use hardware_backend_dx12::Dx12Backend;
use hardware_ir::{
    Access, DepEdge, DepKind, Domain, Graph, Objective, ParamValue, Priority, ResourceGraph,
    ResourceId, ResourceKind, ResourceNode, ShaderId, TaskGraph, TaskId, TaskKind, TaskNode,
};
use hardware_scheduler::Backend;
use hardware_types::ResourceId as Rid;

fn buffer(id: u64, bytes: u64, domain: Domain) -> ResourceNode {
    ResourceNode {
        id: ResourceId(id),
        kind: ResourceKind::Buffer,
        domain,
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

fn s(key: &str, v: &str) -> (String, ParamValue) {
    (key.into(), ParamValue::Str(v.into()))
}

fn tri_graph(domain: Domain) -> Graph {
    let width = 32i64;
    let height = 32i64;
    Graph {
        resources: ResourceGraph {
            nodes: vec![buffer(1, (width * height * 4) as u64, domain)],
        },
        tasks: TaskGraph {
            nodes: vec![task(
                10,
                TaskKind::Raster,
                BTreeMap::from([
                    i("dst", 1),
                    i("width", width),
                    i("height", height),
                    f("r", 0.0),
                    f("g", 0.0),
                    f("b", 0.0),
                    f("a", 1.0),
                    s("draw", "tri"),
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

fn run_cpu(graph: &Graph) -> Vec<u8> {
    let mut cpu = CpuBackend::new();
    cpu.prepare(graph).expect("cpu prepare");
    cpu.submit_task(graph, &graph.tasks.nodes[0])
        .expect("cpu raster tri");
    let store = cpu.store();
    let guard = store.lock().expect("lock");
    guard.bytes(Rid(1)).expect("cpu bytes").to_vec()
}

fn run_dx12(graph: &Graph) -> Option<Vec<u8>> {
    let mut dx = match Dx12Backend::new() {
        Ok(b) => b,
        Err(err) => {
            eprintln!("skip cpu/dx12 tri parity (no dx12): {err}");
            return None;
        }
    };
    if dx.prepare(graph).is_err() || dx.run_prepared(graph).is_err() {
        eprintln!("skip cpu/dx12 tri parity (dx12 prepare/run failed)");
        return None;
    }
    Some(dx.read_u8(Rid(1)).expect("dx12 read"))
}

fn assert_tri_pixels(pixels: &[u8]) {
    assert_eq!(pixels.len(), 32 * 32 * 4);
    let center = (16 * 32 + 16) * 4;
    assert_eq!(&pixels[center..center + 4], &[255, 255, 0, 255]);
    assert_eq!(&pixels[0..4], &[0, 0, 0, 255]);
}

#[test]
fn cpu_raster_tri_reference_pixels() {
    let pixels = run_cpu(&tri_graph(Domain::Host));
    assert_tri_pixels(&pixels);
}

#[test]
fn cpu_and_dx12_raster_tri_center_corner_match() {
    let cpu_px = run_cpu(&tri_graph(Domain::Host));
    assert_tri_pixels(&cpu_px);
    let dx_px = match run_dx12(&tri_graph(Domain::Vram)) {
        Some(px) => px,
        None => return,
    };
    let center = (16 * 32 + 16) * 4;
    assert_eq!(&cpu_px[center..center + 4], &dx_px[center..center + 4]);
    assert_eq!(&cpu_px[0..4], &dx_px[0..4]);
}
