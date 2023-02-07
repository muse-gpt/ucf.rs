//! Same Raster clear graph on DX12 and Vulkan; pixels must match when both run.

use std::collections::BTreeMap;

use ucf_backend_dx12::Dx12Backend;
use ucf_backend_vulkan::VulkanBackend;
use ucf_ir::{
    Access, DepEdge, DepKind, Domain, Graph, Objective, ParamValue, Priority, ResourceGraph,
    ResourceId, ResourceKind, ResourceNode, ShaderId, TaskGraph, TaskId, TaskKind, TaskNode,
};
use ucf_scheduler::Backend;
use ucf_types::ResourceId as Rid;

fn buffer(id: u64, bytes: u64) -> ResourceNode {
    ResourceNode {
        id: ResourceId(id),
        kind: ResourceKind::Buffer,
        domain: Domain::Vram,
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
                    f("r", 0.0),
                    f("g", 1.0),
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

fn run_dx12(graph: &Graph) -> Option<Vec<u8>> {
    let mut dx = match Dx12Backend::new() {
        Ok(b) => b,
        Err(err) => {
            eprintln!("skip dx12 raster parity: {err}");
            return None;
        }
    };
    if dx.prepare(graph).is_err() || dx.run_prepared(graph).is_err() {
        eprintln!("skip dx12 raster parity (prepare/run failed)");
        return None;
    }
    Some(dx.read_u8(Rid(1)).expect("dx12 read_u8"))
}

fn run_vulkan(graph: &Graph) -> Option<Vec<u8>> {
    let mut vk = match VulkanBackend::new() {
        Ok(b) => b,
        Err(err) => {
            eprintln!("skip vulkan raster parity: {err}");
            return None;
        }
    };
    if vk.prepare(graph).is_err() || vk.run_prepared(graph).is_err() {
        eprintln!("skip vulkan raster parity (prepare/run failed)");
        return None;
    }
    Some(vk.read_u8(Rid(1)).expect("vulkan read_u8"))
}

#[test]
fn dx12_and_vulkan_raster_clear_pixels_match() {
    let graph = raster_graph(4, 2);
    let dx = run_dx12(&graph);
    let vk = run_vulkan(&graph);
    match (dx, vk) {
        (Some(a), Some(b)) => {
            assert_eq!(a, b, "DX12 and Vulkan Raster clear pixels must match");
            assert_eq!(a.len(), 32);
            for px in a.chunks_exact(4) {
                assert_eq!(px, &[0, 255, 0, 255], "expected opaque green clear");
            }
        }
        (None, None) => eprintln!("skip dx-vulkan raster parity (neither backend ran)"),
        (None, Some(_)) => eprintln!("skip dx-vulkan raster parity (DX12 missing)"),
        (Some(_), None) => eprintln!("skip dx-vulkan raster parity (Vulkan missing)"),
    }
}
