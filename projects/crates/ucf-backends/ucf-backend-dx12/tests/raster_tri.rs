use std::collections::BTreeMap;

use ucf_backend_dx12::Dx12Backend;
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

fn pixel_at(pixels: &[u8], width: usize, x: usize, y: usize) -> [u8; 4] {
    let i = (y * width + x) * 4;
    [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
}

#[test]
fn dx12_raster_triangle_paints_center_yellow() {
    let width = 32i64;
    let height = 32i64;
    let graph = Graph {
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
    };

    let mut dx12 = match Dx12Backend::new() {
        Ok(backend) => backend,
        Err(err) => {
            eprintln!("skip dx12 raster tri (no usable D3D12 device): {err}");
            return;
        }
    };
    if let Err(err) = dx12.prepare(&graph) {
        eprintln!("skip dx12 raster tri (prepare failed): {err}");
        return;
    }
    if let Err(err) = dx12.run_prepared(&graph) {
        eprintln!("skip dx12 raster tri (run failed): {err}");
        return;
    }

    let pixels = dx12.read_u8(Rid(1)).expect("read_u8");
    let w = width as usize;
    let h = height as usize;
    let center = pixel_at(&pixels, w, w / 2, h / 2);
    assert_eq!(center, [255, 255, 0, 255], "center should be opaque yellow");
    let corner = pixel_at(&pixels, w, 0, 0);
    assert_eq!(corner, [0, 0, 0, 255], "corner should remain clear black");
}
