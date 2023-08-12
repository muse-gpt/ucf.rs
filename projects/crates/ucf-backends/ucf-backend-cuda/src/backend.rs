use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use ucf_capability::{Feature, FeatureSet};
use ucf_emitter::{emit_ptx, program_from_task};
use ucf_ir::{Graph, TaskKind, TaskNode};
use ucf_scheduler::{
    Backend, Error as SchedulerError, ExecStream, ExecutionBindings, ExternalBuffer, Result,
    StreamEventBridge,
};
use ucf_types::ResourceId;

use crate::driver::{
    CudaDriver, CUdeviceptr, CUevent, CUexternalMemory, CUfunction, CUgraph, CUgraphExec, CUmodule,
    DriverError,
};
use crate::external::{
    CudaExecEvent, CudaExecStream, CudaExternalBuffer, CudaStreamEventBridge,
};
use crate::params::{f32_param, resource_param, u32_param, BackendError};

struct DeviceBuffer {
    ptr: CUdeviceptr,
    bytes: usize,
    /// When false, Drop must not `cuMemFree` (adapter / external-owned).
    owned: bool,
}

struct ImportedBuffer {
    ext: CUexternalMemory,
    ptr: CUdeviceptr,
    bytes: usize,
}

struct CachedCudaModule {
    module: CUmodule,
    func: CUfunction,
}

/// Opaque id for a buffer imported from a DX12 NT shared handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImportedBufferId(pub u64);

/// Instantiated CUDA Graph plus PTX modules that must stay loaded until release.
pub struct CapturedCudaGraph {
    graph: CUgraph,
    exec: CUgraphExec,
    modules: Vec<CUmodule>,
}

/// NVIDIA backend via CUDA Driver API with device-resident resources.
pub struct CudaBackend {
    driver: CudaDriver,
    buffers: BTreeMap<ResourceId, DeviceBuffer>,
    imported: BTreeMap<ImportedBufferId, ImportedBuffer>,
    resource_imports: BTreeMap<ResourceId, ImportedBufferId>,
    next_imported: u64,
    module_cache: BTreeMap<u64, CachedCudaModule>,
    module_cache_hits: u64,
    module_cache_misses: u64,
    /// Device allocations created via [`Self::allocate_external_buffer`] (freed on Drop).
    external_owned: Vec<CUdeviceptr>,
    /// Events created via [`Self::create_event`] (destroyed on Drop).
    events_owned: Vec<CUevent>,
    /// Extra streams from [`Self::create_exec_stream`] (destroyed on Drop).
    streams_owned: Vec<crate::driver::CUstream>,
    preferred_stream: Option<std::sync::Arc<dyn ExecStream>>,
}

impl CudaBackend {
    /// Open device `device_index` (usually `0`).
    pub fn new(device_index: u32) -> Result<Self> {
        let driver = CudaDriver::new(device_index).map_err(map_driver_err)?;
        Ok(Self {
            driver,
            buffers: BTreeMap::new(),
            imported: BTreeMap::new(),
            resource_imports: BTreeMap::new(),
            next_imported: 1,
            module_cache: BTreeMap::new(),
            module_cache_hits: 0,
            module_cache_misses: 0,
            external_owned: Vec::new(),
            events_owned: Vec::new(),
            streams_owned: Vec::new(),
            preferred_stream: None,
        })
    }

    /// Default CUDA stream wrapped as [`CudaExecStream`] (same stream used when no override).
    pub fn default_stream(&self) -> std::sync::Arc<CudaExecStream> {
        CudaExecStream::from_raw(self.driver.stream())
    }

    /// Create an additional CUDA stream for adapter-style bindings (destroyed with backend).
    pub fn create_exec_stream(&mut self) -> Result<std::sync::Arc<CudaExecStream>> {
        let raw = self.driver.create_stream().map_err(map_driver_err)?;
        self.streams_owned.push(raw);
        Ok(CudaExecStream::from_raw(raw))
    }

    /// Allocate a device buffer for external binding (backend frees on Drop).
    pub fn allocate_external_buffer(
        &mut self,
        byte_size: u64,
    ) -> Result<std::sync::Arc<CudaExternalBuffer>> {
        let bytes = byte_size as usize;
        let ptr = self.driver.mem_alloc(bytes).map_err(map_driver_err)?;
        let zeros = vec![0u8; bytes];
        if let Err(err) = self.driver.memcpy_htod(ptr, &zeros) {
            let _ = self.driver.mem_free(ptr);
            return Err(map_driver_err(err));
        }
        self.external_owned.push(ptr);
        Ok(CudaExternalBuffer::from_device_ptr(ptr, byte_size))
    }

    /// Create a CUDA event for [`CudaStreamEventBridge`] waits.
    pub fn create_event(&mut self) -> Result<std::sync::Arc<CudaExecEvent>> {
        let event = self.driver.event_create().map_err(map_driver_err)?;
        self.events_owned.push(event);
        Ok(CudaExecEvent::from_raw(event))
    }

    /// Stream/event bridge for this backend's driver.
    pub fn stream_event_bridge(&self) -> CudaStreamEventBridge<'_> {
        CudaStreamEventBridge::new(&self.driver)
    }

    /// Stream-path module cache hits (thin-gate counters).
    pub fn module_cache_hits(&self) -> u64 {
        self.module_cache_hits
    }

    /// Stream-path module cache misses (thin-gate counters).
    pub fn module_cache_misses(&self) -> u64 {
        self.module_cache_misses
    }

    /// Drop cached PTX modules (does not free device buffers).
    pub fn clear_module_cache(&mut self) {
        let cache = std::mem::take(&mut self.module_cache);
        for (_, entry) in cache {
            let _ = self.driver.unload_module(entry.module);
        }
        self.module_cache_hits = 0;
        self.module_cache_misses = 0;
    }

    fn cached_kernel(&mut self, code: &[u8], entry: &str) -> Result<(CUmodule, CUfunction)> {
        let key = code_key(code);
        if let Some(cached) = self.module_cache.get(&key) {
            self.module_cache_hits += 1;
            return Ok((cached.module, cached.func));
        }
        let module = self.driver.load_module(code).map_err(map_driver_err)?;
        let func = match self.driver.get_function(module, entry) {
            Ok(f) => f,
            Err(err) => {
                let _ = self.driver.unload_module(module);
                return Err(map_driver_err(err));
            }
        };
        self.module_cache
            .insert(key, CachedCudaModule { module, func });
        self.module_cache_misses += 1;
        Ok((module, func))
    }

    /// Import a D3D12 NT shared handle and map it for CUDA access.
    pub fn import_dx12_nt_handle(
        &mut self,
        nt_handle: isize,
        bytes: usize,
    ) -> Result<ImportedBufferId> {
        let (ext, ptr) = self
            .driver
            .import_d3d12_resource(nt_handle as *mut _, bytes)
            .map_err(map_driver_err)?;
        let id = ImportedBufferId(self.next_imported);
        self.next_imported += 1;
        self.imported.insert(
            id,
            ImportedBuffer {
                ext,
                ptr,
                bytes,
            },
        );
        Ok(id)
    }

    /// Bind an imported buffer to a graph [`ResourceId`] (skipped by [`prepare`]).
    pub fn bind_imported(&mut self, resource: ResourceId, imported: ImportedBufferId) -> Result<()> {
        let buf = self.imported.get(&imported).ok_or_else(|| {
            map_backend_err(BackendError(format!(
                "imported buffer {} not allocated",
                imported.0
            )))
        })?;
        if let Some(existing) = self.buffers.get(&resource) {
            if existing.bytes != buf.bytes {
                return Err(map_backend_err(BackendError(format!(
                    "resource {} size mismatch for import bind",
                    resource.0
                ))));
            }
        }
        self.resource_imports.insert(resource, imported);
        Ok(())
    }

    /// Download an imported DX12-shared buffer as `f32` values.
    pub fn read_imported_f32(&self, id: ImportedBufferId) -> Result<Vec<f32>> {
        let buf = self.imported.get(&id).ok_or_else(|| {
            map_backend_err(BackendError(format!(
                "imported buffer {} not allocated",
                id.0
            )))
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

    /// Download a device buffer as raw bytes.
    pub fn read_u8(&self, id: ResourceId) -> Result<Vec<u8>> {
        let (ptr, bytes) = self.ptr(id)?;
        let mut host = vec![0u8; bytes];
        self.driver
            .memcpy_dtoh(&mut host, ptr)
            .map_err(map_driver_err)?;
        Ok(host)
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

    /// Capture a Copy/Fill/MatMul TaskGraph into a CUDA Graph and instantiate it.
    ///
    /// Caller must [`launch_cuda_graph`] then [`release_cuda_graph`]. Modules stay loaded
    /// for the lifetime of the returned handle.
    pub fn capture_cuda_graph(&mut self, graph: &Graph) -> Result<CapturedCudaGraph> {
        graph.validate().map_err(SchedulerError::from)?;
        let order = graph.tasks.topological_order().map_err(SchedulerError::from)?;
        if order.is_empty() {
            return Err(map_backend_err(BackendError(
                "cuda graph thin gate needs at least one Copy/Fill/MatMul task".into(),
            )));
        }

        enum CapturedOp {
            Copy {
                src: CUdeviceptr,
                dst: CUdeviceptr,
                bytes: usize,
            },
            Fill {
                module: CUmodule,
                func: CUfunction,
                count: u32,
                ptr: CUdeviceptr,
            },
            MatMul {
                module: CUmodule,
                func: CUfunction,
                a: CUdeviceptr,
                b: CUdeviceptr,
                out: CUdeviceptr,
                m: u32,
                n: u32,
                k: u32,
            },
        }

        fn unload_ops(driver: &CudaDriver, ops: &[CapturedOp]) {
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
                    let ptx = match emit_ptx(&program) {
                        Ok(p) => p,
                        Err(err) => {
                            unload_ops(&self.driver, &ops);
                            return Err(map_emit_err(err));
                        }
                    };
                    let module = match self.driver.load_module(&ptx) {
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
                    let ptx = match emit_ptx(&program) {
                        Ok(p) => p,
                        Err(err) => {
                            unload_ops(&self.driver, &ops);
                            return Err(map_emit_err(err));
                        }
                    };
                    let module = match self.driver.load_module(&ptx) {
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
                        "cuda graph thin gate only supports Copy/Fill/MatMul, got {other:?}"
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
                            .launch_fill_async(*func, *count, *ptr)
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
                Ok(CapturedCudaGraph {
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

    /// Launch a previously captured CUDA Graph and synchronize.
    pub fn launch_cuda_graph(&mut self, captured: &CapturedCudaGraph) -> Result<()> {
        self.driver
            .graph_launch(captured.exec)
            .map_err(map_driver_err)
    }

    /// Destroy graph exec / graph and unload PTX modules owned by `captured`.
    pub fn release_cuda_graph(&mut self, captured: CapturedCudaGraph) {
        let _ = self.driver.graph_exec_destroy(captured.exec);
        let _ = self.driver.graph_destroy(captured.graph);
        for module in captured.modules {
            let _ = self.driver.unload_module(module);
        }
    }

    /// Capture a Copy/Fill/MatMul TaskGraph into a CUDA Graph, instantiate, and launch once.
    pub fn run_prepared_cuda_graph(&mut self, graph: &Graph) -> Result<()> {
        let captured = self.capture_cuda_graph(graph)?;
        let result = self.launch_cuda_graph(&captured);
        self.release_cuda_graph(captured);
        result
    }

    fn ptr(&self, id: ResourceId) -> Result<(CUdeviceptr, usize)> {
        if let Some(buf) = self.buffers.get(&id) {
            return Ok((buf.ptr, buf.bytes));
        }
        if let Some(imp) = self.resource_imports.get(&id) {
            let buf = self.imported.get(imp).ok_or_else(|| {
                map_backend_err(BackendError(format!(
                    "imported buffer {} missing for resource {}",
                    imp.0, id.0
                )))
            })?;
            return Ok((buf.ptr, buf.bytes));
        }
        Err(map_backend_err(BackendError(format!(
            "resource {} not allocated",
            id.0
        ))))
    }
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        self.clear_module_cache();
        let imported = std::mem::take(&mut self.imported);
        for (_, buf) in imported {
            let _ = self.driver.destroy_external_memory(buf.ext);
        }
        let buffers = std::mem::take(&mut self.buffers);
        for (_, buf) in buffers {
            if buf.owned {
                let _ = self.driver.mem_free(buf.ptr);
            }
        }
        let external_owned = std::mem::take(&mut self.external_owned);
        for ptr in external_owned {
            let _ = self.driver.mem_free(ptr);
        }
        let events_owned = std::mem::take(&mut self.events_owned);
        for event in events_owned {
            let _ = self.driver.event_destroy(event);
        }
        let streams_owned = std::mem::take(&mut self.streams_owned);
        for stream in streams_owned {
            let _ = self.driver.destroy_stream(stream);
        }
        self.driver.set_override_stream(None);
    }
}

// Imported external-memory handles are owned here and used on the submit thread.
unsafe impl Send for CudaBackend {}
unsafe impl Sync for CudaBackend {}

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

    fn bind_externals(&mut self, bindings: &ExecutionBindings) -> Result<()> {
        for (id, buf) in &bindings.buffers {
            if buf.backend_name() != "cuda" {
                return Err(SchedulerError::Backend(
                    "cuda".into(),
                    format!(
                        "external buffer for resource {} has backend `{}`, expected `cuda`",
                        id.0,
                        buf.backend_name()
                    ),
                ));
            }
            let host = buf.as_any().downcast_ref::<CudaExternalBuffer>().ok_or_else(|| {
                SchedulerError::Backend(
                    "cuda".into(),
                    format!(
                        "external buffer for resource {} is not a CudaExternalBuffer",
                        id.0
                    ),
                )
            })?;
            let bytes = host.byte_size() as usize;
            if let Some(existing) = self.buffers.get(id) {
                if existing.bytes != bytes || existing.ptr != host.device_ptr() {
                    return Err(map_backend_err(BackendError(format!(
                        "resource {} already allocated incompatibly with external bind",
                        id.0
                    ))));
                }
            } else {
                self.buffers.insert(
                    *id,
                    DeviceBuffer {
                        ptr: host.device_ptr(),
                        bytes,
                        owned: false,
                    },
                );
            }
        }
        if let Some(stream) = &bindings.stream {
            if stream.backend_name() != "cuda" {
                return Err(SchedulerError::Backend(
                    "cuda".into(),
                    format!(
                        "preferred stream backend `{}` is not `cuda`",
                        stream.backend_name()
                    ),
                ));
            }
            let cuda_stream = stream.as_any().downcast_ref::<CudaExecStream>().ok_or_else(|| {
                SchedulerError::Backend(
                    "cuda".into(),
                    "preferred stream is not a CudaExecStream".into(),
                )
            })?;
            // Foreign adapter streams are accepted; driver routes submits via override.
            self.driver
                .set_override_stream(Some(cuda_stream.raw()));
            self.preferred_stream = Some(std::sync::Arc::clone(stream));
        }
        Ok(())
    }

    fn prepare(&mut self, graph: &Graph) -> Result<()> {
        for node in &graph.resources.nodes {
            if self.resource_imports.contains_key(&node.id) {
                continue;
            }
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
            self.buffers.insert(
                node.id,
                DeviceBuffer {
                    ptr,
                    bytes,
                    owned: true,
                },
            );
        }
        Ok(())
    }

    fn submit_task(&mut self, _graph: &Graph, task: &TaskNode) -> Result<()> {
        match &task.kind {
            TaskKind::Copy => self.run_copy(task),
            TaskKind::Fill => self.run_fill(task),
            TaskKind::MatMul => self.run_matmul(task),
            TaskKind::Custom(name) if name == "denoise" => self.run_denoise(task),
            TaskKind::Custom(name) if name == "attention" => self.run_attention(task),
            TaskKind::Raster | TaskKind::RtTrace => Err(map_backend_err(BackendError(
                "raster and ray tracing are not supported on CUDA Driver API".into(),
            ))),
            other => Err(map_backend_err(BackendError(format!(
                "cuda backend does not implement {other:?}"
            )))),
        }
    }

    fn flush(&mut self) -> Result<()> {
        self.driver.synchronize().map_err(map_driver_err)
    }

    fn stream_bridge(&self) -> Option<&dyn StreamEventBridge> {
        // Lifetime: bridge borrows driver; Backend trait wants &dyn with backend lifetime.
        // We cannot return a temporary CudaStreamEventBridge. Adapters should call
        // `stream_event_bridge()` for a scoped bridge. Default None here.
        None
    }

    fn active_stream(&self) -> Option<&dyn ExecStream> {
        self.preferred_stream.as_deref()
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
        let (_module, func) = self.cached_kernel(&ptx, &program.entry)?;
        self.driver
            .launch_fill(func, count, ptr)
            .map_err(map_driver_err)
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
        let (_module, func) = self.cached_kernel(&ptx, &program.entry)?;
        self.driver
            .launch_matmul(func, a_ptr, b_ptr, out_ptr, m, n, k)
            .map_err(map_driver_err)
    }

    fn run_denoise(&mut self, task: &TaskNode) -> Result<()> {
        let src = resource_param(task, "src").map_err(map_backend_err)?;
        let out = resource_param(task, "out").map_err(map_backend_err)?;
        let width = u32_param(task, "width").map_err(map_backend_err)?;
        let height = u32_param(task, "height").map_err(map_backend_err)?;
        if width == 0 || height == 0 {
            return Err(map_backend_err(BackendError(
                "denoise width/height must be > 0".into(),
            )));
        }
        let need = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| map_backend_err(BackendError("denoise size overflow".into())))?;
        let (src_ptr, src_bytes) = self.ptr(src)?;
        let (out_ptr, out_bytes) = self.ptr(out)?;
        expect_bytes(src, src_bytes, need)?;
        expect_bytes(out, out_bytes, need)?;

        let program = program_from_task(task);
        let ptx = emit_ptx(&program).map_err(map_emit_err)?;
        let (_module, func) = self.cached_kernel(&ptx, &program.entry)?;
        self.driver
            .launch_rgba8_denoise(func, src_ptr, out_ptr, width, height)
            .map_err(map_driver_err)
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
        let need = elems.checked_mul(4).ok_or_else(|| {
            map_backend_err(BackendError("attention byte size overflow".into()))
        })?;
        let (q_ptr, q_bytes) = self.ptr(q)?;
        let (k_ptr, k_bytes) = self.ptr(k)?;
        let (v_ptr, v_bytes) = self.ptr(v)?;
        let (out_ptr, out_bytes) = self.ptr(out)?;
        expect_bytes(q, q_bytes, need)?;
        expect_bytes(k, k_bytes, need)?;
        expect_bytes(v, v_bytes, need)?;
        expect_bytes(out, out_bytes, need)?;

        let program = program_from_task(task);
        let ptx = emit_ptx(&program).map_err(map_emit_err)?;
        let (_module, func) = self.cached_kernel(&ptx, &program.entry)?;
        self.driver
            .launch_attention(func, q_ptr, k_ptr, v_ptr, out_ptr, batch, heads, seq, dim)
            .map_err(map_driver_err)
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

fn code_key(bytes: &[u8]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
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
