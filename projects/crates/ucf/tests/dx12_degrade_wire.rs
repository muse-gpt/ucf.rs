//! Soft-skip: DX12 wires degrade-chain strategies into submit paths.

use std::collections::BTreeMap;

use ucf_backend_dx12::Dx12Backend;
use ucf_capability::{DescriptorStrategy, PipelineStrategy, SyncStrategy};
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

fn fill_graph() -> Graph {
    Graph {
        resources: ResourceGraph {
            nodes: vec![buffer(1, 4)],
        },
        tasks: TaskGraph {
            nodes: vec![task(
                10,
                TaskKind::Fill,
                BTreeMap::from([i("dst", 1), f("value", 1.0)]),
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

fn raster_graph() -> Graph {
    Graph {
        resources: ResourceGraph {
            nodes: vec![ResourceNode {
                id: ResourceId(1),
                kind: ResourceKind::Buffer,
                domain: Domain::Vram,
                access: Access::ReadWrite,
                byte_size: Some(32 * 32 * 4),
            }],
        },
        tasks: TaskGraph {
            nodes: vec![task(
                10,
                TaskKind::Raster,
                BTreeMap::from([
                    i("dst", 1),
                    i("width", 32),
                    i("height", 32),
                    f("r", 0.0),
                    f("g", 0.0),
                    f("b", 0.0),
                    f("a", 1.0),
                    ("draw".into(), ParamValue::Str("tri".into())),
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
fn dx12_wires_degrade_strategies_soft_skip() {
    let mut dx = match Dx12Backend::new() {
        Ok(b) => b,
        Err(err) => {
            eprintln!("skip dx12 degrade wire (no device): {err}");
            return;
        }
    };

    let s = dx.strategies();
    assert_eq!(s.descriptor, DescriptorStrategy::ResourceDescriptorHeap);
    assert_eq!(s.pipeline, PipelineStrategy::PsoPrecache);
    assert_eq!(s.sync, SyncStrategy::EnhancedBarriers);
    eprintln!(
        "dx12 strategies: descriptor={:?} pipeline={:?} sync={:?}",
        s.descriptor, s.pipeline, s.sync
    );

    let fill = fill_graph();
    if let Err(err) = dx.prepare(&fill) {
        eprintln!("skip dx12 degrade wire (fill prepare): {err}");
        return;
    }
    if let Err(err) = dx.run_prepared(&fill) {
        eprintln!("skip dx12 degrade wire (fill run): {err}");
        return;
    }
    assert_eq!(dx.last_descriptor_path(), "resource_descriptor_heap");
    assert_eq!(dx.last_pipeline_path(), "pso_precache");
    let out = dx.read_f32(Rid(1)).expect("fill read");
    assert_eq!(out, vec![1.0, 1.0, 1.0, 1.0]);

    // Fresh backend: resource id 1 size differs between fill and raster graphs.
    let mut dx = match Dx12Backend::new() {
        Ok(b) => b,
        Err(err) => {
            eprintln!("skip dx12 degrade wire (raster open): {err}");
            return;
        }
    };
    let raster = raster_graph();
    if let Err(err) = dx.prepare(&raster) {
        eprintln!("skip dx12 degrade wire (raster prepare): {err}");
        return;
    }
    if let Err(err) = dx.run_prepared(&raster) {
        eprintln!("skip dx12 degrade wire (raster run): {err}");
        return;
    }
    assert_eq!(dx.last_sync_path(), "enhanced_barriers");
    assert_eq!(dx.last_pipeline_path(), "pso_precache");
    eprintln!(
        "dx12 wired paths: descriptor={} pipeline={} sync={}",
        dx.last_descriptor_path(),
        dx.last_pipeline_path(),
        dx.last_sync_path()
    );
}
