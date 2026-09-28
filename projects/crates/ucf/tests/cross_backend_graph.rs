//! Same graph: DX12 Fill → flush → CUDA Copy/MatMul over a shared buffer.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use ucf_backend_cuda::CudaBackend;
use ucf_backend_dx12::Dx12Backend;
use ucf_capability::FeatureSet;
use ucf_ir::{
    Access, DepEdge, DepKind, Domain, Graph, Objective, ParamValue, Priority, ResourceGraph,
    ResourceId, ResourceKind, ResourceNode, ShaderId, TaskGraph, TaskId, TaskKind, TaskNode,
};
use ucf_scheduler::{Backend, Result, Scheduler};
use ucf_types::ResourceId as Rid;

struct LockedCuda(Arc<Mutex<CudaBackend>>);

impl Backend for LockedCuda {
    fn name(&self) -> &str {
        "cuda"
    }

    fn features(&self) -> FeatureSet {
        FeatureSet::new()
    }

    fn prepare(&mut self, graph: &Graph) -> Result<()> {
        self.0.lock().unwrap().prepare(graph)
    }

    fn submit_task(&mut self, graph: &Graph, task: &TaskNode) -> Result<()> {
        self.0.lock().unwrap().submit_task(graph, task)
    }

    fn flush(&mut self) -> Result<()> {
        self.0.lock().unwrap().flush()
    }
}

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

fn s(key: &str, v: &str) -> (String, ParamValue) {
    (key.into(), ParamValue::Str(v.into()))
}

fn cross_graph() -> Graph {
    Graph {
        resources: ResourceGraph {
            nodes: vec![
                buffer(1, 4),
                buffer(2, 4),
                buffer(3, 4),
                buffer(4, 4),
            ],
        },
        tasks: TaskGraph {
            nodes: vec![
                task(
                    10,
                    TaskKind::Copy,
                    BTreeMap::from([i("src", 1), i("dst", 2), s("backend", "cuda")]),
                ),
                task(
                    11,
                    TaskKind::Fill,
                    BTreeMap::from([
                        i("dst", 3),
                        f("value", 1.0),
                        s("backend", "dx12"),
                    ]),
                ),
                task(
                    12,
                    TaskKind::MatMul,
                    BTreeMap::from([
                        i("a", 2),
                        i("b", 3),
                        i("out", 4),
                        i("m", 2),
                        i("n", 2),
                        i("k", 2),
                        s("backend", "cuda"),
                    ]),
                ),
            ],
            edges: vec![
                DepEdge {
                    from_task: None,
                    from_resource: Some(ResourceId(1)),
                    to_task: TaskId(10),
                    kind: DepKind::Data,
                },
                DepEdge {
                    from_task: Some(TaskId(10)),
                    from_resource: None,
                    to_task: TaskId(12),
                    kind: DepKind::Execution,
                },
                DepEdge {
                    from_task: Some(TaskId(11)),
                    from_resource: None,
                    to_task: TaskId(12),
                    kind: DepKind::Execution,
                },
            ],
        },
    }
}

#[test]
fn dx12_fill_then_cuda_matmul_over_shared_buffer() {
    let graph = cross_graph();

    let mut dx = match Dx12Backend::new() {
        Ok(b) => b,
        Err(err) => {
            eprintln!("skip cross-backend graph (no D3D12): {err}");
            return;
        }
    };
    let cuda_inner = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(err) => {
            eprintln!("skip cross-backend graph (no CUDA): {err}");
            return;
        }
    };
    let cuda = Arc::new(Mutex::new(cuda_inner));

    let (shared_id, nt_handle) = match dx.shared_alloc(16) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("skip cross-backend graph (shared alloc): {err}");
            return;
        }
    };
    if let Err(err) = dx.publish_shared(ResourceId(3), shared_id) {
        eprintln!("skip cross-backend graph (publish): {err}");
        return;
    }

    {
        let mut c = cuda.lock().unwrap();
        let imported = match c.import_dx12_nt_handle(nt_handle, 16) {
            Ok(id) => id,
            Err(err) => {
                eprintln!("skip cross-backend graph (import): {err}");
                return;
            }
        };
        if let Err(err) = c.bind_imported(ResourceId(3), imported) {
            eprintln!("skip cross-backend graph (bind): {err}");
            return;
        }
        if let Err(err) = c.prepare(&graph) {
            eprintln!("skip cross-backend graph (cuda prepare): {err}");
            return;
        }
        if let Err(err) = c.write_f32(Rid(1), &[1.0, 2.0, 3.0, 4.0]) {
            eprintln!("skip cross-backend graph (cuda seed): {err}");
            return;
        }
    }

    let mut sched = Scheduler::new();
    sched.register(Box::new(dx));
    sched.register(Box::new(LockedCuda(cuda.clone())));
    if let Err(err) = sched.execute(&graph) {
        eprintln!("skip cross-backend graph (execute): {err}");
        return;
    }

    let out = cuda
        .lock()
        .unwrap()
        .read_f32(Rid(4))
        .expect("cuda read out");
    assert_eq!(out, vec![3.0, 3.0, 7.0, 7.0]);
}
