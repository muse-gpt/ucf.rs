use std::collections::BTreeMap;

use ucf_capability::{Feature, FeatureSet};
use ucf_emitter::{emit_hsaco, program_from_task};
use ucf_ir::{Graph, TaskKind, TaskNode};
use ucf_scheduler::{Backend, Error as SchedulerError, Result};
use ucf_types::ResourceId;

use crate::driver::{
    DriverError, HipDeviceptr, HipDriver, HipFunction, HipGraph, HipGraphExec, HipModule,
};
use crate::params::{f32_param, resource_param, u32_param, BackendError};

struct DeviceBuffer {
    ptr: HipDeviceptr,
    bytes: usize,
}

/// Instantiated HIP Graph plus HSACO modules that must stay loaded until release.
pub struct CapturedHipGraph {
    graph: HipGraph,
    exec: HipGraphExec,
    modules: Vec<HipModule>,
}

/// AMD ROCm / HIP backend with device-resident resources.
pub struct RocmBackend {
    driver: HipDriver,
    buffers: BTreeMap<ResourceId, DeviceBuffer>,
}

impl RocmBackend {
    /// Open HIP device `device_index` (usually `0`).
    pub fn new(device_index: u32) -> Result<Self> {
        let driver = HipDriver::new(device_index).map_err(map_driver_err)?;
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
        let order = graph
            .tasks
            .topological_order()
            .map_err(SchedulerError::from)?;
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

    /// Capture a Copy/Fill/MatMul TaskGraph into a HIP Graph and instantiate it.
    ///
    /// Caller must [`launch_hip_graph`] then [`release_hip_graph`]. Modules stay loaded
    /// for the lifetime of the returned handle.
    pub fn capture_hip_graph(&mut self, graph: &Graph) -> Result<CapturedHipGraph> {
        graph.validate().map_err(SchedulerError::from)?;
        let order = graph.tasks.topological_order().map_err(SchedulerError::from)?;
        if order.is_empty() {
            return Err(map_backend_err(BackendError(
                "hip graph thin gate needs at least one Copy/Fill/MatMul task".into(),
            )));
        }

        enum CapturedOp {
            Copy {
                src: HipDeviceptr,
                dst: HipDeviceptr,
                bytes: usize,
            },
            Fill {
                module: HipModule,
                func: HipFunction,
                count: u32,
                ptr: HipDeviceptr,
            },
            MatMul {
                module: HipModule,
                func: HipFunction,
                a: HipDeviceptr,
                b: HipDeviceptr,
                out: HipDeviceptr,
                m: u32,
                n: u32,
                k: u32,
            },
        }

        fn unload_ops(driver: &HipDriver, ops: &[CapturedOp]) {
            for op in ops {
                match op {
                    CapturedOp::Fill { module, .. } | CapturedOp::MatMul { module, .. } => {
                        let _ = driver.unload_module(*module);
                    }
                    CapturedOp::Copy { .. } => {}
                }
            }
        }

        let mut ops: Vec<CapturedOp> = Vec::new();
        for task_id in &order {
            let task = graph
                .tasks
                .nodes
                .iter()
                .find(|t| t.id == *task_id)
                .expect("task in order must exist");
            match &task.kind {
                TaskKind::Copy => {
                    let src = match resource_param(task, "src") {
                        Ok(id) => id,
                        Err(err) => {
                            unload_ops(&self.driver, &ops);
                            return Err(map_backend_err(err));
                        }
                    };
                    let dst = match resource_param(task, "dst") {
                        Ok(id) => id,
                        Err(err) => {
                            unload_ops(&self.driver, &ops);
                            return Err(map_backend_err(err));
                        }
                    };
                    let (src_ptr, src_bytes) = match self.ptr(src) {
                        Ok(p) => p,
                        Err(err) => {
                            unload_ops(&self.driver, &ops);
                            return Err(err);
                        }
                    };
                    let (dst_ptr, dst_bytes) = match self.ptr(dst) {
                        Ok(p) => p,
                        Err(err) => {
                            unload_ops(&self.driver, &ops);
                            return Err(err);
                        }
                    };
                    if src_bytes != dst_bytes {
                        unload_ops(&self.driver, &ops);
                        return Err(map_backend_err(BackendError(format!(
                            "copy size mismatch: src {src_bytes} dst {dst_bytes}"
                        ))));
                    }
                    ops.push(CapturedOp::Copy {
                        src: src_ptr,
                        dst: dst_ptr,
                        bytes: src_bytes,
                    });
                }
                TaskKind::Fill => {
                    let dst = match resource_param(task, "dst") {
                        Ok(id) => id,
                        Err(err) => {
                            unload_ops(&self.driver, &ops);
                            return Err(map_backend_err(err));
                        }
                    };
                    if let Err(err) = f32_param(task, "value") {
                        unload_ops(&self.driver, &ops);
                        return Err(map_backend_err(err));
                    }
                    let (ptr, bytes) = match self.ptr(dst) {
                        Ok(p) => p,
                        Err(err) => {
                            unload_ops(&self.driver, &ops);
                            return Err(err);
                        }
                    };
                    if bytes % 4 != 0 {
                        unload_ops(&self.driver, &ops);
                        return Err(map_backend_err(BackendError(format!(
                            "fill destination {} length {bytes} is not a multiple of 4",
                            dst.0
                        ))));
                    }
                    let count = (bytes / 4) as u32;
                    let program = program_from_task(task);
                    let hsaco = match emit_hsaco(&program) {
                        Ok(h) => h,
                        Err(err) => {
                            unload_ops(&self.driver, &ops);
                            return Err(map_emit_err(err));
                        }
                    };
                    let module = match self.driver.load_module(&hsaco) {
                        Ok(m) => m,
                        Err(err) => {
                            unload_ops(&self.driver, &ops);
                            return Err(map_driver_err(err));
                        }
                    };
                    let func = match self.driver.get_function(module, &program.entry) {
                        Ok(f) => f,
                        Err(err) => {
                            let _ = self.driver.unload_module(module);
                            unload_ops(&self.driver, &ops);
                            return Err(map_driver_err(err));
                        }
                    };
                    ops.push(CapturedOp::Fill {
                        module,
                        func,
                        count,
                        ptr,
                    });
                }
                TaskKind::MatMul => {
                    let a = match resource_param(task, "a") {
                        Ok(id) => id,
                        Err(err) => {
                            unload_ops(&self.driver, &ops);
                            return Err(map_backend_err(err));
                        }
                    };
                    let b = match resource_param(task, "b") {
                        Ok(id) => id,
                        Err(err) => {
                            unload_ops(&self.driver, &ops);
                            return Err(map_backend_err(err));
                        }
                    };
                    let out = match resource_param(task, "out") {
                        Ok(id) => id,
                        Err(err) => {
                            unload_ops(&self.driver, &ops);
                            return Err(map_backend_err(err));
                        }
                    };
                    let m = match u32_param(task, "m") {
                        Ok(v) => v,
                        Err(err) => {
                            unload_ops(&self.driver, &ops);
                            return Err(map_backend_err(err));
                        }
                    };
                    let n = match u32_param(task, "n") {
                        Ok(v) => v,
                        Err(err) => {
                            unload_ops(&self.driver, &ops);
                            return Err(map_backend_err(err));
                        }
                    };
                    let k = match u32_param(task, "k") {
                        Ok(v) => v,
                        Err(err) => {
                            unload_ops(&self.driver, &ops);
                            return Err(map_backend_err(err));
                        }
                    };
                    let (a_ptr, a_bytes) = match self.ptr(a) {
                        Ok(p) => p,
                        Err(err) => {
                            unload_ops(&self.driver, &ops);
                            return Err(err);
                        }
                    };
                    let (b_ptr, b_bytes) = match self.ptr(b) {
                        Ok(p) => p,
                        Err(err) => {
                            unload_ops(&self.driver, &ops);
                            return Err(err);
                        }
                    };
                    let (out_ptr, out_bytes) = match self.ptr(out) {
                        Ok(p) => p,
                        Err(err) => {
                            unload_ops(&self.driver, &ops);
                            return Err(err);
                        }
                    };
                    if let Err(err) =
                        expect_bytes(a, a_bytes, (m as usize) * (k as usize) * 4)
                    {
                        unload_ops(&self.driver, &ops);
                        return Err(err);
                    }
                    if let Err(err) =
                        expect_bytes(b, b_bytes, (k as usize) * (n as usize) * 4)
                    {
                        unload_ops(&self.driver, &ops);
                        return Err(err);
                    }
                    if let Err(err) =
                        expect_bytes(out, out_bytes, (m as usize) * (n as usize) * 4)
                    {
                        unload_ops(&self.driver, &ops);
                        return Err(err);
                    }
                    let program = program_from_task(task);
                    let hsaco = match emit_hsaco(&program) {
                        Ok(h) => h,
                        Err(err) => {
                            unload_ops(&self.driver, &ops);
                            return Err(map_emit_err(err));
                        }
                    };
                    let module = match self.driver.load_module(&hsaco) {
                        Ok(m) => m,
                        Err(err) => {
                            unload_ops(&self.driver, &ops);
                            return Err(map_driver_err(err));
                        }
                    };
                    let func = match self.driver.get_function(module, &program.entry) {
                        Ok(f) => f,
                        Err(err) => {
                            let _ = self.driver.unload_module(module);
                            unload_ops(&self.driver, &ops);
                            return Err(map_driver_err(err));
                        }
                    };
                    ops.push(CapturedOp::MatMul {
                        module,
                        func,
                        a: a_ptr,
                        b: b_ptr,
                        out: out_ptr,
                        m,
                        n,
                        k,
                    });
                }
                other => {
                    unload_ops(&self.driver, &ops);
                    return Err(map_backend_err(BackendError(format!(
                        "hip graph thin gate only supports Copy/Fill/MatMul, got {other:?}"
                    ))));
                }
            }
        }

        let capture = (|| {
            self.driver
                .stream_begin_capture()
                .map_err(map_driver_err)?;
            for op in &ops {
                match op {
                    CapturedOp::Copy { src, dst, bytes } => {
                        self.driver
                            .memcpy_dtod_async(*dst, *src, *bytes)
                            .map_err(map_driver_err)?;
                    }
                    CapturedOp::Fill {
                        func, count, ptr, ..
                    } => {
                        self.driver
                            .launch_fill_async(*func, *ptr, *count)
                            .map_err(map_driver_err)?;
                    }
                    CapturedOp::MatMul {
                        func,
                        a,
                        b,
                        out,
                        m,
                        n,
                        k,
                        ..
                    } => {
                        self.driver
                            .launch_matmul_async(*func, *a, *b, *out, *m, *n, *k)
                            .map_err(map_driver_err)?;
                    }
                }
            }
            let graph = self.driver.stream_end_capture().map_err(map_driver_err)?;
            let exec = match self.driver.graph_instantiate(graph) {
                Ok(e) => e,
                Err(err) => {
                    let _ = self.driver.graph_destroy(graph);
                    return Err(map_driver_err(err));
                }
            };
            Ok((graph, exec))
        })();

        match capture {
            Ok((graph, exec)) => {
                let mut modules = Vec::new();
                for op in ops {
                    match op {
                        CapturedOp::Fill { module, .. } | CapturedOp::MatMul { module, .. } => {
                            modules.push(module);
                        }
                        CapturedOp::Copy { .. } => {}
                    }
                }
                Ok(CapturedHipGraph {
                    graph,
                    exec,
                    modules,
                })
            }
            Err(err) => {
                unload_ops(&self.driver, &ops);
                Err(err)
            }
        }
    }

    /// Launch a previously captured HIP Graph and synchronize.
    pub fn launch_hip_graph(&mut self, captured: &CapturedHipGraph) -> Result<()> {
        self.driver
            .graph_launch(captured.exec)
            .map_err(map_driver_err)
    }

    /// Destroy graph exec / graph and unload HSACO modules owned by `captured`.
    pub fn release_hip_graph(&mut self, captured: CapturedHipGraph) {
        let _ = self.driver.graph_exec_destroy(captured.exec);
        let _ = self.driver.graph_destroy(captured.graph);
        for module in captured.modules {
            let _ = self.driver.unload_module(module);
        }
    }

    /// Capture a Copy/Fill/MatMul TaskGraph into a HIP Graph, instantiate, and launch once.
    pub fn run_prepared_hip_graph(&mut self, graph: &Graph) -> Result<()> {
        let captured = self.capture_hip_graph(graph)?;
        let result = self.launch_hip_graph(&captured);
        self.release_hip_graph(captured);
        result
    }

    fn ptr(&self, id: ResourceId) -> Result<(HipDeviceptr, usize)> {
        self.buffers
            .get(&id)
            .map(|b| (b.ptr, b.bytes))
            .ok_or_else(|| {
                map_backend_err(BackendError(format!("resource {} not allocated", id.0)))
            })
    }

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
        let hsaco = emit_hsaco(&program).map_err(map_emit_err)?;
        let module = self.driver.load_module(&hsaco).map_err(map_driver_err)?;
        let result = (|| {
            let func = self
                .driver
                .get_function(module, &program.entry)
                .map_err(map_driver_err)?;
            self.driver
                .launch_fill(func, ptr, count)
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
        let hsaco = emit_hsaco(&program).map_err(map_emit_err)?;
        let module = self.driver.load_module(&hsaco).map_err(map_driver_err)?;
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

    fn run_attention(&mut self, task: &TaskNode) -> Result<()> {
        let q = resource_param(task, "q").map_err(map_backend_err)?;
        let k = resource_param(task, "k").map_err(map_backend_err)?;
        let v = resource_param(task, "v").map_err(map_backend_err)?;
        let out = resource_param(task, "out").map_err(map_backend_err)?;
        let batch = u32_param(task, "batch").map_err(map_backend_err)?;
        let heads = u32_param(task, "heads").map_err(map_backend_err)?;
        let seq = u32_param(task, "seq").map_err(map_backend_err)?;
        let dim = u32_param(task, "dim").map_err(map_backend_err)?;
        let elems = (batch as usize)
            .checked_mul(heads as usize)
            .and_then(|n| n.checked_mul(seq as usize))
            .and_then(|n| n.checked_mul(dim as usize))
            .ok_or_else(|| map_backend_err(BackendError("attention size overflow".into())))?;
        let need = elems
            .checked_mul(4)
            .ok_or_else(|| map_backend_err(BackendError("attention byte size overflow".into())))?;
        let (q_ptr, q_bytes) = self.ptr(q)?;
        let (k_ptr, k_bytes) = self.ptr(k)?;
        let (v_ptr, v_bytes) = self.ptr(v)?;
        let (out_ptr, out_bytes) = self.ptr(out)?;
        expect_bytes(q, q_bytes, need)?;
        expect_bytes(k, k_bytes, need)?;
        expect_bytes(v, v_bytes, need)?;
        expect_bytes(out, out_bytes, need)?;

        let program = program_from_task(task);
        let hsaco = emit_hsaco(&program).map_err(map_emit_err)?;
        let module = self.driver.load_module(&hsaco).map_err(map_driver_err)?;
        let result = (|| {
            let func = self
                .driver
                .get_function(module, &program.entry)
                .map_err(map_driver_err)?;
            self.driver
                .launch_attention(func, q_ptr, k_ptr, v_ptr, out_ptr, batch, heads, seq, dim)
                .map_err(map_driver_err)
        })();
        let _ = self.driver.unload_module(module);
        result
    }
}

impl Drop for RocmBackend {
    fn drop(&mut self) {
        let buffers = std::mem::take(&mut self.buffers);
        for (_, buf) in buffers {
            let _ = self.driver.mem_free(buf.ptr);
        }
    }
}

// HIP device pointers are owned here and used on the submit thread.
unsafe impl Send for RocmBackend {}
unsafe impl Sync for RocmBackend {}

impl Backend for RocmBackend {
    fn name(&self) -> &str {
        "rocm"
    }

    fn features(&self) -> FeatureSet {
        FeatureSet::new()
            .with(Feature::UnifiedMemory)
            .with(Feature::MatrixCore)
            .with(Feature::HipGraph)
    }

    fn prepare(&mut self, graph: &Graph) -> Result<()> {
        for node in &graph.resources.nodes {
            let bytes = node.byte_size.ok_or_else(|| {
                map_backend_err(BackendError(format!(
                    "resource {} needs byte_size for rocm backend",
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
            TaskKind::Custom(name) if name == "attention" => self.run_attention(task),
            TaskKind::Raster | TaskKind::RtTrace => Err(map_backend_err(BackendError(
                "raster and ray tracing are not supported on HIP".into(),
            ))),
            other => Err(map_backend_err(BackendError(format!(
                "rocm backend does not implement {other:?}"
            )))),
        }
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
    SchedulerError::Backend("rocm".into(), error.to_string())
}

fn map_emit_err(error: String) -> SchedulerError {
    SchedulerError::Backend("rocm".into(), error)
}

fn map_backend_err(error: BackendError) -> SchedulerError {
    SchedulerError::Backend("rocm".into(), error.0)
}
