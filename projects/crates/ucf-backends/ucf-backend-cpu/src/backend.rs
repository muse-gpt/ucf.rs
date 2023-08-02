use std::collections::BTreeMap;
use std::sync::Arc;

use ucf_capability::FeatureSet;
use ucf_ir::{Graph, TaskKind, TaskNode};
use ucf_scheduler::{
    Backend, Error as SchedulerError, ExecStream, ExecutionBindings, ExternalBuffer,
    ImmediateBridge, Result, StreamEventBridge,
};
use ucf_types::ResourceId;

use crate::external::HostExternalBuffer;
use crate::params::{f32_param, resource_param, u32_param, BackendError};
use crate::store::{shared_store, HostStore, SharedHostStore};

/// Host CPU reference backend with readable buffer store.
pub struct CpuBackend {
    store: SharedHostStore,
    externals: BTreeMap<ResourceId, Arc<dyn ExternalBuffer>>,
    stream: Option<Arc<dyn ExecStream>>,
    bridge: ImmediateBridge,
}

impl CpuBackend {
    /// Empty backend with a private shared store.
    pub fn new() -> Self {
        Self {
            store: shared_store(),
            externals: BTreeMap::new(),
            stream: None,
            bridge: ImmediateBridge,
        }
    }

    /// Backend that shares an existing store (for seed / readback beside `Runtime`).
    pub fn with_store(store: SharedHostStore) -> Self {
        Self {
            store,
            externals: BTreeMap::new(),
            stream: None,
            bridge: ImmediateBridge,
        }
    }

    /// Clone of the shared store handle.
    pub fn store(&self) -> SharedHostStore {
        Arc::clone(&self.store)
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, HostStore>> {
        self.store
            .lock()
            .map_err(|_| Self::map_err(BackendError("cpu store lock poisoned".into())))
    }

    fn map_err(err: BackendError) -> SchedulerError {
        SchedulerError::Backend("cpu".into(), err.0)
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn features(&self) -> FeatureSet {
        FeatureSet::new()
    }

    fn bind_externals(&mut self, bindings: &ExecutionBindings) -> Result<()> {
        for (id, buf) in &bindings.buffers {
            if buf.backend_name() != "cpu" {
                return Err(SchedulerError::Backend(
                    "cpu".into(),
                    format!(
                        "external buffer for resource {} has backend `{}`, expected `cpu`",
                        id.0,
                        buf.backend_name()
                    ),
                ));
            }
            if let Some(host) = buf.as_any().downcast_ref::<HostExternalBuffer>() {
                if !Arc::ptr_eq(host.store(), &self.store) {
                    return Err(SchedulerError::Backend(
                        "cpu".into(),
                        format!(
                            "external buffer for resource {} must share the backend host store",
                            id.0
                        ),
                    ));
                }
                if host.resource_id() != *id {
                    return Err(SchedulerError::Backend(
                        "cpu".into(),
                        format!(
                            "external buffer resource id {} does not match bind key {}",
                            host.resource_id().0,
                            id.0
                        ),
                    ));
                }
            }
            self.externals.insert(*id, Arc::clone(buf));
        }
        if let Some(stream) = &bindings.stream {
            if stream.backend_name() != "cpu" {
                return Err(SchedulerError::Backend(
                    "cpu".into(),
                    format!(
                        "preferred stream backend `{}` is not `cpu`",
                        stream.backend_name()
                    ),
                ));
            }
            self.stream = Some(Arc::clone(stream));
        }
        Ok(())
    }

    fn prepare(&mut self, graph: &Graph) -> Result<()> {
        let mut store = self.lock()?;
        for node in &graph.resources.nodes {
            let bytes = node.byte_size.ok_or_else(|| {
                Self::map_err(BackendError(format!(
                    "resource {} needs byte_size for cpu backend",
                    node.id.0
                )))
            })? as usize;
            if let Some(ext) = self.externals.get(&node.id) {
                if ext.byte_size() != bytes as u64 {
                    return Err(Self::map_err(BackendError(format!(
                        "resource {} external size {} does not match graph size {bytes}",
                        node.id.0,
                        ext.byte_size()
                    ))));
                }
            }
            store.ensure(node.id, bytes).map_err(Self::map_err)?;
        }
        Ok(())
    }

    fn submit_task(&mut self, _graph: &Graph, task: &TaskNode) -> Result<()> {
        match &task.kind {
            TaskKind::Copy => self.run_copy(task).map_err(Self::map_err),
            TaskKind::Fill => self.run_fill(task).map_err(Self::map_err),
            TaskKind::MatMul => self.run_matmul(task).map_err(Self::map_err),
            other => Err(Self::map_err(BackendError(format!(
                "cpu backend does not implement {other:?}"
            )))),
        }
    }

    fn stream_bridge(&self) -> Option<&dyn StreamEventBridge> {
        Some(&self.bridge)
    }

    fn active_stream(&self) -> Option<&dyn ExecStream> {
        self.stream.as_deref()
    }
}

impl CpuBackend {
    fn run_copy(&mut self, task: &TaskNode) -> std::result::Result<(), BackendError> {
        let src = resource_param(task, "src")?;
        let dst = resource_param(task, "dst")?;
        let mut store = self
            .store
            .lock()
            .map_err(|_| BackendError("cpu store lock poisoned".into()))?;
        let src_bytes = store.bytes(src)?.to_vec();
        let dst_bytes = store.bytes_mut(dst)?;
        if src_bytes.len() != dst_bytes.len() {
            return Err(BackendError(format!(
                "copy size mismatch: src {} dst {}",
                src_bytes.len(),
                dst_bytes.len()
            )));
        }
        dst_bytes.copy_from_slice(&src_bytes);
        Ok(())
    }

    fn run_fill(&mut self, task: &TaskNode) -> std::result::Result<(), BackendError> {
        let dst = resource_param(task, "dst")?;
        let value = f32_param(task, "value")?;
        let mut store = self
            .store
            .lock()
            .map_err(|_| BackendError("cpu store lock poisoned".into()))?;
        let bytes = store.bytes_mut(dst)?;
        if bytes.len() % 4 != 0 {
            return Err(BackendError(format!(
                "fill destination {} length {} is not a multiple of 4",
                dst.0,
                bytes.len()
            )));
        }
        let enc = value.to_le_bytes();
        for slot in bytes.chunks_exact_mut(4) {
            slot.copy_from_slice(&enc);
        }
        Ok(())
    }

    fn run_matmul(&mut self, task: &TaskNode) -> std::result::Result<(), BackendError> {
        let a_id = resource_param(task, "a")?;
        let b_id = resource_param(task, "b")?;
        let out_id = resource_param(task, "out")?;
        let m = u32_param(task, "m")? as usize;
        let n = u32_param(task, "n")? as usize;
        let k = u32_param(task, "k")? as usize;

        let store = self
            .store
            .lock()
            .map_err(|_| BackendError("cpu store lock poisoned".into()))?;
        let a = store.read_f32(a_id)?;
        let b = store.read_f32(b_id)?;
        expect_len(a_id, a.len(), m * k)?;
        expect_len(b_id, b.len(), k * n)?;
        expect_len(out_id, store.len_bytes(out_id)? / 4, m * n)?;
        drop(store);

        let mut out = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0f32;
                for t in 0..k {
                    sum += a[i * k + t] * b[t * n + j];
                }
                out[i * n + j] = sum;
            }
        }

        let mut store = self
            .store
            .lock()
            .map_err(|_| BackendError("cpu store lock poisoned".into()))?;
        store.write_f32(out_id, &out)
    }
}

fn expect_len(id: ResourceId, have: usize, need: usize) -> std::result::Result<(), BackendError> {
    if have != need {
        return Err(BackendError(format!(
            "resource {} has {have} f32 values, expected {need}",
            id.0
        )));
    }
    Ok(())
}
