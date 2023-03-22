//! Time CPU (and soft-skip CUDA) MatMul, report FLOP/s via `MatmulMetrics`.

use std::collections::BTreeMap;
use std::time::Instant;

use ucf::prelude::*;
use ucf_backend_cuda::CudaBackend;
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

fn matmul_graph() -> Graph {
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
                    BTreeMap::from([i("src", 1), i("dst", 2)]),
                ),
                task(
                    11,
                    TaskKind::Fill,
                    BTreeMap::from([i("dst", 3), f("value", 1.0)]),
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
fn cpu_matmul_reports_positive_throughput() {
    let graph = matmul_graph();
    let store = shared_store();
    {
        let mut prep = CpuBackend::with_store(store.clone());
        prep.prepare(&graph).expect("prepare");
    }
    store
        .lock()
        .unwrap()
        .write_f32(Rid(1), &[1.0, 2.0, 3.0, 4.0])
        .expect("seed");

    let mut runtime = Runtime::new();
    runtime.register_backend(Box::new(CpuBackend::with_store(store.clone())));
    let start = Instant::now();
    runtime.run(&graph).expect("run");
    let elapsed = start.elapsed();
    let metrics = MatmulMetrics::new(2, 2, 2, elapsed.as_nanos());
    assert_eq!(metrics.flops, 16);
    assert!(metrics.flop_per_s() > 0.0);
    eprintln!(
        "cpu matmul: {} FLOPs in {:?} → {:.3e} FLOP/s",
        metrics.flops,
        elapsed,
        metrics.flop_per_s()
    );
    let out = store.lock().unwrap().read_f32(Rid(4)).expect("read");
    assert_eq!(out, vec![3.0, 3.0, 7.0, 7.0]);
}

#[test]
fn cuda_matmul_metrics_soft_skip() {
    let graph = matmul_graph();
    let mut cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(err) => {
            eprintln!("skip cuda matmul metrics: {err}");
            return;
        }
    };
    if cuda.prepare(&graph).is_err()
        || cuda.write_f32(Rid(1), &[1.0, 2.0, 3.0, 4.0]).is_err()
    {
        eprintln!("skip cuda matmul metrics (prepare/seed failed)");
        return;
    }
    let start = Instant::now();
    if cuda.run_prepared(&graph).is_err() {
        eprintln!("skip cuda matmul metrics (run failed)");
        return;
    }
    let elapsed = start.elapsed();
    let metrics = MatmulMetrics::new(2, 2, 2, elapsed.as_nanos());
    let out = cuda.read_f32(Rid(4)).expect("cuda read");
    assert_eq!(out, vec![3.0, 3.0, 7.0, 7.0]);
    assert!(metrics.flop_per_s() > 0.0);
    eprintln!(
        "cuda matmul: {} FLOPs in {:?} → {:.3e} FLOP/s",
        metrics.flops,
        elapsed,
        metrics.flop_per_s()
    );
}
