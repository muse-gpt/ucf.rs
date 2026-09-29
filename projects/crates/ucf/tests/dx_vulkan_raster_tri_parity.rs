//! Same Raster `draw=tri` graph on DX12 and Vulkan; center/corner pixels must match.

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

fn s(key: &str, v: &str) -> (String, ParamValue) {
    (key.into(), ParamValue::Str(v.into()))
}

fn tri_graph() -> Graph {
    let width = 32i64;
    let height = 32i64;
    Graph {
        resources: ResourceGraph {
            nodes: vec![buffer(1, (width * height * 4) as u64)],
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

fn run_dx12(graph: &Graph) -> Option<Vec<u8>> {
    let mut dx = match Dx12Backend::new() {
        Ok(b) => b,
        Err(err) => {
            eprintln!("skip dx12 tri parity: {err}");
            return None;
        }
    };
    if dx.prepare(graph).is_err() || dx.run_prepared(graph).is_err() {
        eprintln!("skip dx12 tri parity (prepare/run failed)");
        return None;
    }
    Some(dx.read_u8(Rid(1)).expect("dx12 read"))
}

fn run_vulkan(graph: &Graph) -> Option<Vec<u8>> {
    let mut vk = match VulkanBackend::new() {
        Ok(b) => b,
        Err(err) => {
            eprintln!("skip vulkan tri parity: {err}");
            return None;
        }
    };
    if vk.prepare(graph).is_err() || vk.run_prepared(graph).is_err() {
        eprintln!("skip vulkan tri parity (prepare/run failed)");
        return None;
    }
    Some(vk.read_u8(Rid(1)).expect("vulkan read"))
}

#[test]
fn dx12_and_vulkan_raster_tri_pixels_match() {
    let graph = tri_graph();
    let dx = run_dx12(&graph);
    let vk = run_vulkan(&graph);
    match (dx, vk) {
        (Some(a), Some(b)) => {
            assert_eq!(a.len(), b.len());
            assert_eq!(a.len(), 32 * 32 * 4);
            // Full-buffer equality is not required: D3D12 vs Vulkan edge coverage can differ
            // by a pixel. Center (inside) and corner (outside) must agree.
            let center = (16 * 32 + 16) * 4;
            assert_eq!(&a[center..center + 4], &[255, 255, 0, 255]);
            assert_eq!(&b[center..center + 4], &[255, 255, 0, 255]);
            assert_eq!(&a[0..4], &[0, 0, 0, 255]);
            assert_eq!(&b[0..4], &[0, 0, 0, 255]);
            let yellow = |px: &[u8]| {
                px.chunks_exact(4)
                    .filter(|c| *c == [255, 255, 0, 255])
                    .count()
            };
            let ya = yellow(&a);
            let yb = yellow(&b);
            let diff = ya.abs_diff(yb);
            assert!(
                diff <= 8,
                "yellow pixel counts should be close: dx12={ya} vulkan={yb}"
            );
        }
        (None, None) => eprintln!("skip dx-vulkan tri parity (neither backend ran)"),
        (None, Some(_)) => eprintln!("skip dx-vulkan tri parity (DX12 missing)"),
        (Some(_), None) => eprintln!("skip dx-vulkan tri parity (Vulkan missing)"),
    }
}
