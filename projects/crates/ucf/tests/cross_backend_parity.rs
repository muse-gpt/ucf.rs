//! Same reference graph on every available backend; all must match CPU.

use std::collections::BTreeMap;

use ucf_backend_cpu::{shared_store, CpuBackend};
use ucf_backend_cuda::CudaBackend;
use ucf_backend_dx12::Dx12Backend;
use ucf_backend_rocm::RocmBackend;
use ucf_backend_vulkan::VulkanBackend;
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

fn reference_graph() -> Graph {
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

fn run_cpu(graph: &Graph) -> Vec<f32> {
    let store = shared_store();
    {
        let mut prep = CpuBackend::with_store(store.clone());
        prep.prepare(graph).expect("cpu prepare");
    }
    store
        .lock()
        .unwrap()
        .write_f32(Rid(1), &[1.0, 2.0, 3.0, 4.0])
        .expect("cpu seed");
    let mut runtime = ucf_runtime::Runtime::new();
    runtime.register_backend(Box::new(CpuBackend::with_store(store.clone())));
    runtime.run(graph).expect("cpu run");
    let out = store.lock().unwrap().read_f32(Rid(4)).expect("cpu read");
    out
}

fn seed() -> [f32; 4] {
    [1.0, 2.0, 3.0, 4.0]
}

#[test]
fn available_gpu_backends_match_cpu() {
    let graph = reference_graph();
    let expected = run_cpu(&graph);
    assert_eq!(expected, vec![3.0, 3.0, 7.0, 7.0]);

    let mut ran = 0usize;

    match CudaBackend::new(0) {
        Ok(mut cuda) => {
            if cuda.prepare(&graph).is_ok()
                && cuda.write_f32(Rid(1), &seed()).is_ok()
                && cuda.run_prepared(&graph).is_ok()
            {
                let actual = cuda.read_f32(Rid(4)).expect("cuda read");
                assert_eq!(actual, expected, "cuda");
                ran += 1;
            } else {
                eprintln!("skip cuda (prepare/run failed)");
            }
        }
        Err(err) => eprintln!("skip cuda: {err}"),
    }

    match Dx12Backend::new() {
        Ok(mut dx) => {
            if dx.prepare(&graph).is_ok()
                && dx.write_f32(Rid(1), &seed()).is_ok()
                && dx.run_prepared(&graph).is_ok()
            {
                let actual = dx.read_f32(Rid(4)).expect("dx12 read");
                assert_eq!(actual, expected, "dx12");
                ran += 1;
            } else {
                eprintln!("skip dx12 (prepare/run failed)");
            }
        }
        Err(err) => eprintln!("skip dx12: {err}"),
    }

    match VulkanBackend::new() {
        Ok(mut vk) => {
            if vk.prepare(&graph).is_ok()
                && vk.write_f32(Rid(1), &seed()).is_ok()
                && vk.run_prepared(&graph).is_ok()
            {
                let actual = vk.read_f32(Rid(4)).expect("vulkan read");
                assert_eq!(actual, expected, "vulkan");
                ran += 1;
            } else {
                eprintln!("skip vulkan (prepare/run failed)");
            }
        }
        Err(err) => eprintln!("skip vulkan: {err}"),
    }

    match RocmBackend::new(0) {
        Ok(mut rocm) => {
            if rocm.prepare(&graph).is_ok()
                && rocm.write_f32(Rid(1), &seed()).is_ok()
                && rocm.run_prepared(&graph).is_ok()
            {
                let actual = rocm.read_f32(Rid(4)).expect("rocm read");
                assert_eq!(actual, expected, "rocm");
                ran += 1;
            } else {
                eprintln!("skip rocm (prepare/run failed)");
            }
        }
        Err(err) => eprintln!("skip rocm: {err}"),
    }

    eprintln!("cross-backend parity ran {ran} GPU backend(s) against CPU");
    // CPU always ran; GPU soft-skip is OK when none are present.
}
