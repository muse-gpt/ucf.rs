//! Soft-skip DX12 vs Vulkan wall-clock samples for MatMul and Raster clear.

use std::collections::BTreeMap;
use std::time::Instant;

use ucf::prelude::*;
use ucf_backend_dx12::Dx12Backend;
use ucf_backend_vulkan::VulkanBackend;
use ucf_types::ResourceId as Rid;

fn buffer_floats(id: u64, floats: usize) -> ResourceNode {
    ResourceNode {
        id: ResourceId(id),
        kind: ResourceKind::Buffer,
        domain: Domain::Vram,
        access: Access::ReadWrite,
        byte_size: Some((floats * 4) as u64),
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

fn matmul_graph() -> Graph {
    Graph {
        resources: ResourceGraph {
            nodes: vec![
                buffer_floats(1, 4),
                buffer_floats(2, 4),
                buffer_floats(3, 4),
                buffer_floats(4, 4),
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

fn raster_clear_graph() -> Graph {
    Graph {
        resources: ResourceGraph {
            nodes: vec![buffer_bytes(1, 32 * 32 * 4)],
        },
        tasks: TaskGraph {
            nodes: vec![task(
                10,
                TaskKind::Raster,
                BTreeMap::from([
                    i("dst", 1),
                    i("width", 32),
                    i("height", 32),
                    f("r", 1.0),
                    f("g", 0.0),
                    f("b", 0.0),
                    f("a", 1.0),
                ]),
            )],
            edges: vec![DepEdge {
                from_task: None,
                from_resource: Some(ResourceId(1)),
                to_task: TaskId(10),
                kind: DepKind::Data,
            }],
        },
    }
}

fn time_dx12() -> Option<BackendPerfRow> {
    let matmul_ns = {
        let mut dx = match Dx12Backend::new() {
            Ok(b) => b,
            Err(err) => {
                eprintln!("skip dx12 perf: {err}");
                return None;
            }
        };
        let matmul = matmul_graph();
        if dx.prepare(&matmul).is_err()
            || dx.write_f32(Rid(1), &[1.0, 2.0, 3.0, 4.0]).is_err()
        {
            eprintln!("skip dx12 perf (matmul prepare/seed)");
            return None;
        }
        let t0 = Instant::now();
        if dx.run_prepared(&matmul).is_err() {
            eprintln!("skip dx12 perf (matmul run)");
            return None;
        }
        let ns = t0.elapsed().as_nanos();
        let _ = dx.read_f32(Rid(4)).ok()?;
        ns
    };
    let raster_ns = {
        let mut dx = match Dx12Backend::new() {
            Ok(b) => b,
            Err(err) => {
                eprintln!("skip dx12 perf (raster device): {err}");
                return None;
            }
        };
        let raster = raster_clear_graph();
        if dx.prepare(&raster).is_err() {
            eprintln!("skip dx12 perf (raster prepare)");
            return None;
        }
        let t1 = Instant::now();
        if dx.run_prepared(&raster).is_err() {
            eprintln!("skip dx12 perf (raster run)");
            return None;
        }
        let ns = t1.elapsed().as_nanos();
        let _ = dx.read_u8(Rid(1)).ok()?;
        ns
    };

    Some(BackendPerfRow {
        backend: "dx12".into(),
        samples: vec![
            TimedSample::new("matmul", matmul_ns),
            TimedSample::new("raster_clear", raster_ns),
        ],
    })
}

fn time_vulkan() -> Option<BackendPerfRow> {
    let matmul_ns = {
        let mut vk = match VulkanBackend::new() {
            Ok(b) => b,
            Err(err) => {
                eprintln!("skip vulkan perf: {err}");
                return None;
            }
        };
        let matmul = matmul_graph();
        if vk.prepare(&matmul).is_err()
            || vk.write_f32(Rid(1), &[1.0, 2.0, 3.0, 4.0]).is_err()
        {
            eprintln!("skip vulkan perf (matmul prepare/seed)");
            return None;
        }
        let t0 = Instant::now();
        if vk.run_prepared(&matmul).is_err() {
            eprintln!("skip vulkan perf (matmul run)");
            return None;
        }
        let ns = t0.elapsed().as_nanos();
        let _ = vk.read_f32(Rid(4)).ok()?;
        ns
    };
    let raster_ns = {
        let mut vk = match VulkanBackend::new() {
            Ok(b) => b,
            Err(err) => {
                eprintln!("skip vulkan perf (raster device): {err}");
                return None;
            }
        };
        let raster = raster_clear_graph();
        if vk.prepare(&raster).is_err() {
            eprintln!("skip vulkan perf (raster prepare)");
            return None;
        }
        let t1 = Instant::now();
        if vk.run_prepared(&raster).is_err() {
            eprintln!("skip vulkan perf (raster run)");
            return None;
        }
        let ns = t1.elapsed().as_nanos();
        let _ = vk.read_u8(Rid(1)).ok()?;
        ns
    };

    Some(BackendPerfRow {
        backend: "vulkan".into(),
        samples: vec![
            TimedSample::new("matmul", matmul_ns),
            TimedSample::new("raster_clear", raster_ns),
        ],
    })
}

#[test]
fn dx12_vulkan_perf_report_soft_skip() {
    let mut rows = Vec::new();
    if let Some(row) = time_dx12() {
        rows.push(row);
    }
    if let Some(row) = time_vulkan() {
        rows.push(row);
    }
    if rows.is_empty() {
        eprintln!("skip dx-vulkan perf (neither backend ran)");
        return;
    }
    let report = PerfReport::from_rows(rows);
    for line in report.lines() {
        eprintln!("{line}");
    }
    for row in &report.rows {
        for sample in &row.samples {
            assert!(
                sample.elapsed_nanos > 0,
                "{} {} must be timed",
                row.backend,
                sample.label
            );
        }
    }
    if report.sample_nanos("dx12", "matmul").is_some()
        && report.sample_nanos("vulkan", "matmul").is_some()
    {
        let ratio = report
            .ratio("vulkan", "dx12", "matmul")
            .expect("matmul ratio");
        eprintln!("vulkan/dx12 matmul ratio: {ratio:.3}");
        assert!(ratio.is_finite() && ratio > 0.0);
    }
}
