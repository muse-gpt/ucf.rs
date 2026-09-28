use std::collections::BTreeMap;

use ucf_capability::{Feature, FeatureSet};
use ucf_emitter::{emit_ptx, program_from_task};
use ucf_ir::{Graph, TaskKind, TaskNode};
use ucf_scheduler::{Backend, Error as SchedulerError, Result};
use ucf_types::ResourceId;

use crate::driver::{CudaDriver, CUdeviceptr, DriverError};
use crate::params::{f32_param, resource_param, u32_param, BackendError};

struct DeviceBuffer {
    ptr: CUdeviceptr,
    bytes: usize,
}

/// NVIDIA backend via CUDA Driver API with device-resident resources.
pub struct CudaBackend {
    driver: CudaDriver,
    buffers: BTreeMap<ResourceId, DeviceBuffer>,
}

impl CudaBackend {
    /// Open device `device_index` (usually `0`).
    pub fn new(device_index: u32) -> Result<Self> {
        let driver = CudaDriver::new(device_index).map_err(map_driver_err)?;
        Ok(Self {
            driver,
            buffers: BTreeMap::new(),
        })
    }

    /// Upload host `f32` values into an allocated device buffer.
    pub fn write_f32(&mut self, id: ResourceId, values: &[f32]) -> Result<()> {
        let bytes = values.len() * 4;
        let buf = self.buffers.get(&id).ok_or_else(|| {
            map_backend_err(BackendError(format!("resource {} not allocated", id.0)))
        })?;
        if buf.bytes != bytes {
            return Err(map_backend_err(BackendError(format!(
                "resource {} expected {} bytes, got {bytes}",
                id.0, buf.bytes
            ))));
        }
        let mut host = Vec::with_capacity(bytes);
        for v in values {
            host.extend_from_slice(&v.to_le_bytes());
        }
        self.driver
            .memcpy_htod(buf.ptr, &host)
            .map_err(map_driver_err)
    }

    /// Download a device buffer as `f32` values.
    pub fn read_f32(&self, id: ResourceId) -> Result<Vec<f32>> {
        let buf = self.buffers.get(&id).ok_or_else(|| {
            map_backend_err(BackendError(format!("resource {} not allocated", id.0)))
        })?;
        let mut host = vec![0u8; buf.bytes];
        self.driver
            .memcpy_dtoh(&mut host, buf.ptr)
            .map_err(map_driver_err)?;
        Ok(host
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Run every task on this backend. Caller must [`prepare`] and seed inputs first.
    pub fn run_prepared(&mut self, graph: &Graph) -> Result<()> {
        graph.validate().map_err(SchedulerError::from)?;
        let order = graph.tasks.topological_order().map_err(SchedulerError::from)?;
        for task_id in order {
            let task = graph
                .tasks
                .nodes
                .iter()
                .find(|t| t.id == task_id)
                .expect("task in order must exist");
            self.submit_task(graph, task)?;
        }
        Ok(())
    }

    fn ptr(&self, id: ResourceId) -> Result<(CUdeviceptr, usize)> {
        self.buffers
            .get(&id)
            .map(|b| (b.ptr, b.bytes))
            .ok_or_else(|| map_backend_err(BackendError(format!("resource {} not allocated", id.0))))
    }
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        let buffers = std::mem::take(&mut self.buffers);
        for (_, buf) in buffers {
            let _ = self.driver.mem_free(buf.ptr);
        }
    }
}

impl Backend for CudaBackend {
    fn name(&self) -> &str {
        "cuda"
    }

    fn features(&self) -> FeatureSet {
        FeatureSet::new()
            .with(Feature::CudaGraph)
            .with(Feature::UnifiedMemory)
            .with(Feature::DynamicParallelism)
    }

    fn prepare(&mut self, graph: &Graph) -> Result<()> {
        for node in &graph.resources.nodes {
            let bytes = node.byte_size.ok_or_else(|| {
                map_backend_err(BackendError(format!(
                    "resource {} needs byte_size for cuda backend",
                    node.id.0
                )))
            })? as usize;
            if let Some(existing) = self.buffers.get(&node.id) {
                if existing.bytes != bytes {
                    return Err(map_backend_err(BackendError(format!(
                        "resource {} size mismatch: have {}, need {bytes}",
                        node.id.0, existing.bytes
                    ))));
                }
                continue;
            }
            let ptr = self.driver.mem_alloc(bytes).map_err(map_driver_err)?;
            let zeros = vec![0u8; bytes];
            if let Err(err) = self.driver.memcpy_htod(ptr, &zeros) {
                let _ = self.driver.mem_free(ptr);
                return Err(map_driver_err(err));
            }
            self.buffers
                .insert(node.id, DeviceBuffer { ptr, bytes });
        }
        Ok(())
    }

    fn submit_task(&mut self, _graph: &Graph, task: &TaskNode) -> Result<()> {
        match &task.kind {
            TaskKind::Copy => self.run_copy(task),
            TaskKind::Fill => self.run_fill(task),
            TaskKind::MatMul => self.run_matmul(task),
            TaskKind::Raster | TaskKind::RtTrace => Err(map_backend_err(BackendError(
                "raster and ray tracing are not supported on CUDA Driver API".into(),
            ))),
            other => Err(map_backend_err(BackendError(format!(
                "cuda backend does not implement {other:?}"
            )))),
        }
    }
}

impl CudaBackend {
    fn run_copy(&mut self, task: &TaskNode) -> Result<()> {
        let src = resource_param(task, "src").map_err(map_backend_err)?;
        let dst = resource_param(task, "dst").map_err(map_backend_err)?;
        let (src_ptr, src_bytes) = self.ptr(src)?;
        let (dst_ptr, dst_bytes) = self.ptr(dst)?;
        if src_bytes != dst_bytes {
            return Err(map_backend_err(BackendError(format!(
                "copy size mismatch: src {src_bytes} dst {dst_bytes}"
            ))));
        }
        self.driver
            .memcpy_dtod(dst_ptr, src_ptr, src_bytes)
            .map_err(map_driver_err)
    }

    fn run_fill(&mut self, task: &TaskNode) -> Result<()> {
        let dst = resource_param(task, "dst").map_err(map_backend_err)?;
        let _value = f32_param(task, "value").map_err(map_backend_err)?;
        let (ptr, bytes) = self.ptr(dst)?;
        if bytes % 4 != 0 {
            return Err(map_backend_err(BackendError(format!(
                "fill destination {} length {bytes} is not a multiple of 4",
                dst.0
            ))));
        }
        let count = (bytes / 4) as u32;
        let program = program_from_task(task);
        let ptx = emit_ptx(&program).map_err(map_emit_err)?;
        let module = self.driver.load_module(&ptx).map_err(map_driver_err)?;
        let result = (|| {
            let func = self
                .driver
                .get_function(module, &program.entry)
                .map_err(map_driver_err)?;
            self.driver
                .launch_fill(func, count, ptr)
                .map_err(map_driver_err)
        })();
        let _ = self.driver.unload_module(module);
        result
    }

    fn run_matmul(&mut self, task: &TaskNode) -> Result<()> {
        let a = resource_param(task, "a").map_err(map_backend_err)?;
        let b = resource_param(task, "b").map_err(map_backend_err)?;
        let out = resource_param(task, "out").map_err(map_backend_err)?;
        let m = u32_param(task, "m").map_err(map_backend_err)?;
        let n = u32_param(task, "n").map_err(map_backend_err)?;
        let k = u32_param(task, "k").map_err(map_backend_err)?;
        let (a_ptr, a_bytes) = self.ptr(a)?;
        let (b_ptr, b_bytes) = self.ptr(b)?;
        let (out_ptr, out_bytes) = self.ptr(out)?;
        expect_bytes(a, a_bytes, (m as usize) * (k as usize) * 4)?;
        expect_bytes(b, b_bytes, (k as usize) * (n as usize) * 4)?;
        expect_bytes(out, out_bytes, (m as usize) * (n as usize) * 4)?;

        let program = program_from_task(task);
        let ptx = emit_ptx(&program).map_err(map_emit_err)?;
        let module = self.driver.load_module(&ptx).map_err(map_driver_err)?;
        let result = (|| {
            let func = self
                .driver
                .get_function(module, &program.entry)
                .map_err(map_driver_err)?;
            self.driver
                .launch_matmul(func, a_ptr, b_ptr, out_ptr, m, n, k)
                .map_err(map_driver_err)
        })();
        let _ = self.driver.unload_module(module);
        result
    }
}

fn expect_bytes(id: ResourceId, have: usize, need: usize) -> Result<()> {
    if have != need {
        return Err(map_backend_err(BackendError(format!(
            "resource {} has {have} bytes, expected {need}",
            id.0
        ))));
    }
    Ok(())
}

fn map_driver_err(error: DriverError) -> SchedulerError {
    SchedulerError::Backend("cuda".into(), error.to_string())
}

fn map_emit_err(error: String) -> SchedulerError {
    SchedulerError::Backend("cuda".into(), error)
}

fn map_backend_err(error: BackendError) -> SchedulerError {
    SchedulerError::Backend("cuda".into(), error.0)
}
