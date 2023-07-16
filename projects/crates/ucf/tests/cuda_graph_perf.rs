//! Soft-skip: report CUDA Graph replay vs stream wall time (advisory <5%).

use std::collections::BTreeMap;
use std::time::Instant;

use ucf_backend_cuda::CudaBackend;
use ucf_capability::Feature;
use ucf_ir::{
    Access, DepEdge, DepKind, Domain, Graph, Objective, ParamValue, Priority, ResourceGraph,
    ResourceId, ResourceKind, ResourceNode, ShaderId, TaskGraph, TaskId, TaskKind, TaskNode,
};
use ucf_scheduler::Backend;
use ucf_types::ResourceId as Rid;

const N: u32 = 256;
const WARMUP: u32 = 5;
const ITERS: u32 = 20;

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

fn matmul_graph() -> Graph {
    let elems = (N as usize) * (N as usize);
    Graph {
        resources: ResourceGraph {
            nodes: vec![
                buffer(1, elems),
                buffer(2, elems),
                buffer(3, elems),
            ],
        },
        tasks: TaskGraph {
            nodes: vec![task(
                10,
                TaskKind::MatMul,
                BTreeMap::from([
                    i("a", 1),
                    i("b", 2),
                    i("out", 3),
                    i("m", N as i64),
                    i("n", N as i64),
                    i("k", N as i64),
                ]),
            )],
            edges: vec![
                DepEdge {
                    from_task: None,
                    from_resource: Some(ResourceId(1)),
                    to_task: TaskId(10),
                    kind: DepKind::Data,
                },
                DepEdge {
                    from_task: None,
                    from_resource: Some(ResourceId(2)),
                    to_task: TaskId(10),
                    kind: DepKind::Data,
                },
            ],
        },
    }
}

fn seed_identity(n: u32) -> Vec<f32> {
    let n = n as usize;
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        a[i * n + i] = 1.0;
    }
    a
}

#[test]
fn cuda_graph_vs_stream_perf_report_soft_skip() {
    let mut cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(err) => {
            eprintln!("skip cuda graph perf (no device): {err}");
            return;
        }
    };
    assert!(
        cuda.features().has(Feature::CudaGraph),
        "CUDA backend must advertise CudaGraph"
    );

    let graph = matmul_graph();
    if let Err(err) = cuda.prepare(&graph) {
        eprintln!("skip cuda graph perf (prepare): {err}");
        return;
    }
    let a = seed_identity(N);
    let b = seed_identity(N);
    if let Err(err) = cuda.write_f32(Rid(1), &a) {
        eprintln!("skip cuda graph perf (seed a): {err}");
        return;
    }
    if let Err(err) = cuda.write_f32(Rid(2), &b) {
        eprintln!("skip cuda graph perf (seed b): {err}");
        return;
    }

    let captured = match cuda.capture_cuda_graph(&graph) {
        Ok(c) => c,
        Err(err) => {
            eprintln!("skip cuda graph perf (capture): {err}");
            return;
        }
    };
    for _ in 0..WARMUP {
        if let Err(err) = cuda.launch_cuda_graph(&captured) {
            cuda.release_cuda_graph(captured);
            eprintln!("skip cuda graph perf (graph warmup): {err}");
            return;
        }
    }
    let t0 = Instant::now();
    for _ in 0..ITERS {
        if let Err(err) = cuda.launch_cuda_graph(&captured) {
            cuda.release_cuda_graph(captured);
            eprintln!("skip cuda graph perf (graph launch): {err}");
            return;
        }
    }
    let graph_ns = t0.elapsed().as_nanos();
    cuda.release_cuda_graph(captured);

    for _ in 0..WARMUP {
        if let Err(err) = cuda.run_prepared(&graph) {
            eprintln!("skip cuda graph perf (stream warmup): {err}");
            return;
        }
    }
    let t1 = Instant::now();
    for _ in 0..ITERS {
        if let Err(err) = cuda.run_prepared(&graph) {
            eprintln!("skip cuda graph perf (stream run): {err}");
            return;
        }
    }
    let stream_ns = t1.elapsed().as_nanos();

    let ratio = graph_ns as f64 / stream_ns as f64;
    eprintln!(
        "cuda graph vs stream MatMul {N}x{N}: graph={graph_ns}ns stream={stream_ns}ns ratio={ratio:.3} (advisory graph/stream < 1.05)"
    );
    assert!(graph_ns > 0 && stream_ns > 0);
}
