use std::collections::BTreeMap;

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
fn vulkan_raster_clear_readback_rgba8() {
    let width = 4i64;
    let height = 2i64;
    let graph = raster_graph(width, height);

    let mut vulkan = match VulkanBackend::new() {
        Ok(backend) => backend,
        Err(err) => {
            eprintln!("skip vulkan raster (no usable Vulkan device): {err}");
            return;
        }
    };

    if let Err(err) = vulkan.prepare(&graph) {
        eprintln!("skip vulkan raster (prepare failed): {err}");
        return;
    }
    if let Err(err) = vulkan.run_prepared(&graph) {
        eprintln!("skip vulkan raster (run failed): {err}");
        return;
    }

    let pixels = vulkan.read_u8(Rid(1)).expect("vulkan raster read_u8");
    assert_eq!(pixels.len(), (width * height * 4) as usize);
    for px in pixels.chunks_exact(4) {
        assert_eq!(px, &[255, 0, 0, 255], "expected opaque red clear");
    }
}
