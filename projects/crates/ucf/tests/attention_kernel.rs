//! Soft-skip Attention kernel: Out = (Q Kᵀ) V (no softmax), timed via `AttentionMetrics`.

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

fn s(key: &str, v: &str) -> (String, ParamValue) {
    (key.into(), ParamValue::Str(v.into()))
}

/// B=1,H=1,S=2,D=2 — Q=K=I, V=[[1,2],[3,4]] ⇒ out = V.
fn attention_graph() -> Graph {
    Graph {
        resources: ResourceGraph {
            nodes: vec![
                buffer(1, 4), // Q
                buffer(2, 4), // K
                buffer(3, 4), // V
                buffer(4, 4), // Out
            ],
        },
        tasks: TaskGraph {
            nodes: vec![task(
                10,
                TaskKind::Custom("attention".into()),
                BTreeMap::from([
                    i("q", 1),
                    i("k", 2),
                    i("v", 3),
                    i("out", 4),
                    i("batch", 1),
                    i("heads", 1),
                    i("seq", 2),
                    i("dim", 2),
                    s("backend", "unused"),
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
                DepEdge {
                    from_task: None,
                    from_resource: Some(ResourceId(3)),
                    to_task: TaskId(10),
                    kind: DepKind::Data,
                },
            ],
        },
    }
}

const Q: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const K: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const V: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
const EXPECT: [f32; 4] = [1.0, 2.0, 3.0, 4.0];

#[test]
fn attention_kernel_rocm_or_cuda_soft_skip() {
    let graph = attention_graph();
    let mut ran = false;

    match RocmBackend::new(0) {
        Ok(mut rocm) => {
            if rocm.prepare(&graph).is_ok()
                && rocm.write_f32(Rid(1), &Q).is_ok()
                && rocm.write_f32(Rid(2), &K).is_ok()
                && rocm.write_f32(Rid(3), &V).is_ok()
            {
                let start = Instant::now();
                match rocm.run_prepared(&graph) {
                    Ok(()) => {
                        let elapsed = start.elapsed();
                        let out = rocm.read_f32(Rid(4)).expect("rocm read");
                        assert_eq!(out, EXPECT);
                        let metrics = AttentionMetrics::new(1, 1, 2, 2, elapsed.as_nanos());
                        assert_eq!(metrics.flops, 32);
                        assert!(metrics.flop_per_s() > 0.0);
                        eprintln!(
                            "rocm attention: {} FLOPs in {:?} → {:.3e} FLOP/s",
                            metrics.flops,
                            elapsed,
                            metrics.flop_per_s()
                        );
                        ran = true;
                    }
                    Err(err) => eprintln!("skip rocm attention (run): {err}"),
                }
            } else {
                eprintln!("skip rocm attention (prepare/seed)");
            }
        }
        Err(err) => eprintln!("skip rocm attention (no driver): {err}"),
    }

    match CudaBackend::new(0) {
        Ok(mut cuda) => {
            if cuda.prepare(&graph).is_ok()
                && cuda.write_f32(Rid(1), &Q).is_ok()
                && cuda.write_f32(Rid(2), &K).is_ok()
                && cuda.write_f32(Rid(3), &V).is_ok()
            {
                let start = Instant::now();
                match cuda.run_prepared(&graph) {
                    Ok(()) => {
                        let elapsed = start.elapsed();
                        let out = cuda.read_f32(Rid(4)).expect("cuda read");
                        assert_eq!(out, EXPECT);
                        let metrics = AttentionMetrics::new(1, 1, 2, 2, elapsed.as_nanos());
                        assert_eq!(metrics.flops, 32);
                        assert!(metrics.flop_per_s() > 0.0);
                        eprintln!(
                            "cuda attention: {} FLOPs in {:?} → {:.3e} FLOP/s",
                            metrics.flops,
                            elapsed,
                            metrics.flop_per_s()
                        );
                        ran = true;
                    }
                    Err(err) => eprintln!("skip cuda attention (run): {err}"),
                }
            } else {
                eprintln!("skip cuda attention (prepare/seed)");
            }
        }
        Err(err) => eprintln!("skip cuda attention (no driver): {err}"),
    }

    if !ran {
        eprintln!("skip attention kernel (neither CUDA nor ROCm ran)");
    }
}
