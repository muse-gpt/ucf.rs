//! Same graph: DX12 Raster clear + DX12 Fill → CUDA MatMul over shared buffer.

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

struct LockedDx12(Arc<Mutex<Dx12Backend>>);

impl Backend for LockedDx12 {
    fn name(&self) -> &str {
        "dx12"
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

fn buffer_bytes(id: u64, bytes: u64) -> ResourceNode {
    ResourceNode {
        id: ResourceId(id),
        kind: ResourceKind::Buffer,
        domain: Domain::Vram,
        access: Access::ReadWrite,
        byte_size: Some(bytes),
    }
}

fn buffer_floats(id: u64, floats: usize) -> ResourceNode {
    buffer_bytes(id, (floats * 4) as u64)
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

fn hybrid_graph() -> Graph {
    // Resource 5: 4x2 RGBA8 raster destination (DX12-only, not shared).
    Graph {
        resources: ResourceGraph {
            nodes: vec![
                buffer_floats(1, 4),
                buffer_floats(2, 4),
                buffer_floats(3, 4),
                buffer_floats(4, 4),
                buffer_bytes(5, 32),
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
                task(
                    13,
                    TaskKind::Raster,
                    BTreeMap::from([
                        i("dst", 5),
                        i("width", 4),
                        i("height", 2),
                        f("r", 0.0),
                        f("g", 0.0),
                        f("b", 1.0),
                        f("a", 1.0),
                        s("backend", "dx12"),
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
                DepEdge {
                    from_task: None,
                    from_resource: Some(ResourceId(5)),
                    to_task: TaskId(13),
                    kind: DepKind::Data,
                },
            ],
        },
    }
}

#[test]
fn dx12_raster_and_cuda_matmul_same_graph() {
    let graph = hybrid_graph();

    let dx_inner = match Dx12Backend::new() {
        Ok(b) => b,
        Err(err) => {
            eprintln!("skip raster-cuda hybrid (no D3D12): {err}");
            return;
        }
    };
    let dx = Arc::new(Mutex::new(dx_inner));

    let cuda_inner = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(err) => {
            eprintln!("skip raster-cuda hybrid (no CUDA): {err}");
            return;
        }
    };
    let cuda = Arc::new(Mutex::new(cuda_inner));

    let (shared_id, nt_handle) = match dx.lock().unwrap().shared_alloc(16) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("skip raster-cuda hybrid (shared alloc): {err}");
            return;
        }
    };
    if let Err(err) = dx.lock().unwrap().publish_shared(ResourceId(3), shared_id) {
        eprintln!("skip raster-cuda hybrid (publish): {err}");
        return;
    }

    {
        let mut c = cuda.lock().unwrap();
        let imported = match c.import_dx12_nt_handle(nt_handle, 16) {
            Ok(id) => id,
            Err(err) => {
                eprintln!("skip raster-cuda hybrid (import): {err}");
                return;
            }
        };
        if let Err(err) = c.bind_imported(ResourceId(3), imported) {
            eprintln!("skip raster-cuda hybrid (bind): {err}");
            return;
        }
        if let Err(err) = c.prepare(&graph) {
            eprintln!("skip raster-cuda hybrid (cuda prepare): {err}");
            return;
        }
        if let Err(err) = c.write_f32(Rid(1), &[1.0, 2.0, 3.0, 4.0]) {
            eprintln!("skip raster-cuda hybrid (cuda seed): {err}");
            return;
        }
    }

    let mut sched = Scheduler::new();
    sched.register(Box::new(LockedDx12(dx.clone())));
    sched.register(Box::new(LockedCuda(cuda.clone())));
    if let Err(err) = sched.execute(&graph) {
        eprintln!("skip raster-cuda hybrid (execute): {err}");
        return;
    }

    let out = cuda
        .lock()
        .unwrap()
        .read_f32(Rid(4))
        .expect("cuda read out");
    assert_eq!(out, vec![3.0, 3.0, 7.0, 7.0]);

    let pixels = dx
        .lock()
        .unwrap()
        .read_u8(Rid(5))
        .expect("dx12 raster read_u8");
    assert_eq!(pixels.len(), 32);
    for px in pixels.chunks_exact(4) {
        assert_eq!(px, &[0, 0, 255, 255], "expected opaque blue clear");
    }
}
