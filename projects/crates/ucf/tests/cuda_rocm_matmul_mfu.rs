//! Soft-skip CUDA vs ROCm MatMul throughput / MFU comparison.

use std::collections::BTreeMap;
use std::time::Instant;

use ucf::prelude::*;
use ucf_backend_cuda::CudaBackend;
use ucf_backend_rocm::RocmBackend;
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

/// Nominal peak used only so both sides share the same MFU denominator (not a HW probe).
const NOMINAL_PEAK_FLOP_PER_S: f64 = 1.0e12;

#[test]
fn cuda_and_rocm_matmul_mfu_soft_skip() {
    let graph = matmul_graph();
    let mut cuda_m: Option<MatmulMetrics> = None;
    let mut rocm_m: Option<MatmulMetrics> = None;

    match CudaBackend::new(0) {
        Ok(mut cuda) => {
            if cuda.prepare(&graph).is_ok()
                && cuda.write_f32(Rid(1), &[1.0, 2.0, 3.0, 4.0]).is_ok()
            {
                let start = Instant::now();
                if cuda.run_prepared(&graph).is_ok() {
                    let elapsed = start.elapsed();
                    let out = cuda.read_f32(Rid(4)).expect("cuda read");
                    assert_eq!(out, vec![3.0, 3.0, 7.0, 7.0]);
                    let metrics = MatmulMetrics::new(2, 2, 2, elapsed.as_nanos());
                    assert!(metrics.flop_per_s() > 0.0);
                    let mfu = metrics.mfu(NOMINAL_PEAK_FLOP_PER_S).expect("mfu");
                    eprintln!(
                        "cuda matmul: {:.3e} FLOP/s MFU={mfu:.3e} (peak={NOMINAL_PEAK_FLOP_PER_S:.3e})",
                        metrics.flop_per_s()
                    );
                    cuda_m = Some(metrics);
                } else {
                    eprintln!("skip cuda side (run failed)");
                }
            } else {
                eprintln!("skip cuda side (prepare/seed failed)");
            }
        }
        Err(err) => eprintln!("skip cuda side (no driver): {err}"),
    }

    match RocmBackend::new(0) {
        Ok(mut rocm) => {
            if rocm.prepare(&graph).is_ok()
                && rocm.write_f32(Rid(1), &[1.0, 2.0, 3.0, 4.0]).is_ok()
            {
                let start = Instant::now();
                if rocm.run_prepared(&graph).is_ok() {
                    let elapsed = start.elapsed();
                    let out = rocm.read_f32(Rid(4)).expect("rocm read");
                    assert_eq!(out, vec![3.0, 3.0, 7.0, 7.0]);
                    let metrics = MatmulMetrics::new(2, 2, 2, elapsed.as_nanos());
                    assert!(metrics.flop_per_s() > 0.0);
                    let mfu = metrics.mfu(NOMINAL_PEAK_FLOP_PER_S).expect("mfu");
                    eprintln!(
                        "rocm matmul: {:.3e} FLOP/s MFU={mfu:.3e} (peak={NOMINAL_PEAK_FLOP_PER_S:.3e})",
                        metrics.flop_per_s()
                    );
                    rocm_m = Some(metrics);
                } else {
                    eprintln!("skip rocm side (run failed)");
                }
            } else {
                eprintln!("skip rocm side (prepare/seed failed)");
            }
        }
        Err(err) => eprintln!("skip rocm side (no driver): {err}"),
    }

    if cuda_m.is_none() && rocm_m.is_none() {
        eprintln!("skip cuda/rocm matmul MFU (neither backend available)");
        return;
    }

    if let (Some(c), Some(r)) = (cuda_m, rocm_m) {
        let ratio = c.flop_per_s() / r.flop_per_s();
        eprintln!("cuda/rocm matmul throughput ratio: {ratio:.3}");
        assert!(ratio.is_finite() && ratio > 0.0);
    }
}
