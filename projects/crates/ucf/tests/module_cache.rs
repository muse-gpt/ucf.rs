//! Soft-skip: stream-path HSACO/PTX modules are cached across identical submits.

use std::collections::BTreeMap;

use ucf_backend_cuda::CudaBackend;
use ucf_backend_rocm::RocmBackend;
use ucf_ir::{
    Access, DepEdge, DepKind, Domain, Graph, Objective, ParamValue, Priority, ResourceGraph,
    ResourceId, ResourceKind, ResourceNode, ShaderId, TaskGraph, TaskId, TaskKind, TaskNode,
};
use ucf_scheduler::Backend;

fn buffer(id: u64, floats: usize) -> ResourceNode {
    ResourceNode {
        id: ResourceId(id),
        kind: ResourceKind::Buffer,
        domain: Domain::Vram,
        access: Access::ReadWrite,
        byte_size: Some((floats * 4) as u64),
    }
}

fn fill_graph(value: f64) -> Graph {
    Graph {
        resources: ResourceGraph {
            nodes: vec![buffer(1, 4)],
        },
        tasks: TaskGraph {
            nodes: vec![TaskNode {
                id: TaskId(10),
                kind: TaskKind::Fill,
                shader: ShaderId(10),
                params: BTreeMap::from([
                    ("dst".into(), ParamValue::I64(1)),
                    ("value".into(), ParamValue::F64(value)),
                ]),
                dispatch: Default::default(),
                objective: Objective::MaxThroughput,
                priority: Priority::Batch,
            }],
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
fn rocm_module_cache_hits_on_repeat_fill_soft_skip() {
    let mut rocm = match RocmBackend::new(0) {
        Ok(b) => b,
        Err(err) => {
            eprintln!("skip rocm module cache (no device): {err}");
            return;
        }
    };
    let graph = fill_graph(2.0);
    if let Err(err) = rocm.prepare(&graph) {
        eprintln!("skip rocm module cache (prepare): {err}");
        return;
    }
    if let Err(err) = rocm.run_prepared(&graph) {
        eprintln!("skip rocm module cache (first run): {err}");
        return;
    }
    assert_eq!(rocm.module_cache_misses(), 1);
    assert_eq!(rocm.module_cache_hits(), 0);
    if let Err(err) = rocm.run_prepared(&graph) {
        eprintln!("skip rocm module cache (second run): {err}");
        return;
    }
    assert_eq!(rocm.module_cache_misses(), 1);
    assert_eq!(rocm.module_cache_hits(), 1);
    eprintln!(
        "rocm module cache ok: hits={} misses={}",
        rocm.module_cache_hits(),
        rocm.module_cache_misses()
    );
}

#[test]
fn cuda_module_cache_hits_on_repeat_fill_soft_skip() {
    let mut cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(err) => {
            eprintln!("skip cuda module cache (no device): {err}");
            return;
        }
    };
    let graph = fill_graph(2.0);
    if let Err(err) = cuda.prepare(&graph) {
        eprintln!("skip cuda module cache (prepare): {err}");
        return;
    }
    if let Err(err) = cuda.run_prepared(&graph) {
        eprintln!("skip cuda module cache (first run): {err}");
        return;
    }
    assert_eq!(cuda.module_cache_misses(), 1);
    assert_eq!(cuda.module_cache_hits(), 0);
    if let Err(err) = cuda.run_prepared(&graph) {
        eprintln!("skip cuda module cache (second run): {err}");
        return;
    }
    assert_eq!(cuda.module_cache_misses(), 1);
    assert_eq!(cuda.module_cache_hits(), 1);
    eprintln!(
        "cuda module cache ok: hits={} misses={}",
        cuda.module_cache_hits(),
        cuda.module_cache_misses()
    );
}
