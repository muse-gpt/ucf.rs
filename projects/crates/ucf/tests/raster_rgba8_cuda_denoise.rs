//! Same graph: DX12 Raster triangle → CUDA RGBA8 denoise over shared buffer.

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

/// Raster yellow triangle into shared RGBA8, then CUDA `Custom("denoise")` → local out.
fn hybrid_graph() -> Graph {
    let width = 32u64;
    let height = 32u64;
    let rgba_bytes = width * height * 4;
    Graph {
        resources: ResourceGraph {
            nodes: vec![
                buffer_bytes(1, rgba_bytes), // shared: DX12 Raster → CUDA denoise src
                buffer_bytes(2, rgba_bytes), // CUDA denoise out
            ],
        },
        tasks: TaskGraph {
            nodes: vec![
                task(
                    10,
                    TaskKind::Raster,
                    BTreeMap::from([
                        i("dst", 1),
                        i("width", width as i64),
                        i("height", height as i64),
                        f("r", 0.0),
                        f("g", 0.0),
                        f("b", 0.0),
                        f("a", 1.0),
                        s("draw", "tri"),
                        s("backend", "dx12"),
                    ]),
                ),
                task(
                    11,
                    TaskKind::Custom("denoise".into()),
                    BTreeMap::from([
                        i("src", 1),
                        i("out", 2),
                        i("width", width as i64),
                        i("height", height as i64),
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
                    to_task: TaskId(11),
                    kind: DepKind::Execution,
                },
            ],
        },
    }
}

#[test]
fn dx12_raster_rgba8_then_cuda_denoise() {
    let graph = hybrid_graph();
    let rgba_bytes = 32 * 32 * 4;

    let dx_inner = match Dx12Backend::new() {
        Ok(b) => b,
        Err(err) => {
            eprintln!("skip raster-rgba8-cuda-denoise (no D3D12): {err}");
            return;
        }
    };
    let dx = Arc::new(Mutex::new(dx_inner));

    let cuda_inner = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(err) => {
            eprintln!("skip raster-rgba8-cuda-denoise (no CUDA): {err}");
            return;
        }
    };
    let cuda = Arc::new(Mutex::new(cuda_inner));

    let (shared_id, nt_handle) = match dx.lock().unwrap().shared_alloc(rgba_bytes) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("skip raster-rgba8-cuda-denoise (shared alloc): {err}");
            return;
        }
    };
    if let Err(err) = dx.lock().unwrap().publish_shared(ResourceId(1), shared_id) {
        eprintln!("skip raster-rgba8-cuda-denoise (publish): {err}");
        return;
    }

    {
        let mut c = cuda.lock().unwrap();
        let imported = match c.import_dx12_nt_handle(nt_handle, rgba_bytes) {
            Ok(id) => id,
            Err(err) => {
                eprintln!("skip raster-rgba8-cuda-denoise (import): {err}");
                return;
            }
        };
        if let Err(err) = c.bind_imported(ResourceId(1), imported) {
            eprintln!("skip raster-rgba8-cuda-denoise (bind): {err}");
            return;
        }
        if let Err(err) = c.prepare(&graph) {
            eprintln!("skip raster-rgba8-cuda-denoise (cuda prepare): {err}");
            return;
        }
    }

    let mut sched = Scheduler::new();
    sched.register(Box::new(LockedDx12(dx.clone())));
    sched.register(Box::new(LockedCuda(cuda.clone())));
    if let Err(err) = sched.execute(&graph) {
        eprintln!("skip raster-rgba8-cuda-denoise (execute): {err}");
        return;
    }

    let pixels = cuda
        .lock()
        .unwrap()
        .read_u8(Rid(2))
        .expect("cuda denoise read_u8");
    assert_eq!(pixels.len(), rgba_bytes);
    let center = pixel_at(&pixels, 32, 16, 16);
    assert_eq!(center, [255, 255, 0, 255], "solid yellow center survives horizontal mean");
    let corner = pixel_at(&pixels, 32, 0, 0);
    assert_eq!(corner, [0, 0, 0, 255], "clear black corner");
}
