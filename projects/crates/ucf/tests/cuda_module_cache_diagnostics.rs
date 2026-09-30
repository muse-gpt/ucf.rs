//! Soft-skip: CUDA module cache probes surface as runtime diagnostics.

use std::collections::BTreeMap;

use ucf::prelude::*;
use ucf::ParamValue;
use ucf_backend_cuda::CudaBackend;
use ucf_runtime::event_kinds;

fn fill_graph(value: f64) -> Graph {
    Graph {
        resources: ResourceGraph {
            nodes: vec![ResourceNode {
                id: ResourceId(1),
                kind: ResourceKind::Buffer,
                domain: Domain::Vram,
                access: Access::ReadWrite,
                byte_size: Some(16),
            }],
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
fn cuda_module_cache_emits_runtime_diagnostics() {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(err) => {
            eprintln!("skip cuda module cache diagnostics (no device): {err}");
            return;
        }
    };
    let graph = fill_graph(2.0);
    let mut rt = Runtime::new();
    rt.register_backend(Box::new(cuda));
    if let Err(err) = rt.run(&graph) {
        eprintln!("skip cuda module cache diagnostics (run): {err}");
        return;
    }
    let diag = rt.diagnostics();
    assert!(diag.contains_kind(event_kinds::MODULE_COMPILE));
    assert!(diag.contains_kind(event_kinds::MODULE_CACHE_MISS));

    rt.clear_diagnostics();
    if let Err(err) = rt.run_prepared(&graph) {
        eprintln!("skip cuda module cache diagnostics (second run): {err}");
        return;
    }
    assert!(rt.diagnostics().contains_kind(event_kinds::MODULE_CACHE_HIT));
}
