use std::collections::BTreeMap;
use std::sync::Arc;

use ucf_capability::FeatureSet;
use ucf_ir::{Dispatch, Graph, ParamValue, TaskKind, TaskNode};
use ucf_program::{program_hash, CpuExecResult, CpuInterpreter, Program, Value};
use ucf_scheduler::{
    Backend, Error as SchedulerError, ExecStream, ExecutionBindings, ExternalBuffer,
    ImmediateBridge, Result, StreamEventBridge,
};
use ucf_types::ResourceId;

use crate::external::HostExternalBuffer;
use crate::params::{f32_param, optional_f64_param, resource_param, u32_param, BackendError};
use crate::raster_tri;
use crate::store::{shared_store, HostStore, SharedHostStore};

/// Host CPU reference backend with readable buffer store.
pub struct CpuBackend {
    store: SharedHostStore,
    externals: BTreeMap<ResourceId, Arc<dyn ExternalBuffer>>,
    stream: Option<Arc<dyn ExecStream>>,
    bridge: ImmediateBridge,
}

/// 一次通用 UCF 程序执行的可审计结果。
#[derive(Debug, Clone, PartialEq)]
pub struct ProgramExecutionReport {
    /// 实际执行的程序哈希。
    pub program_hash: u64,
    /// 每个并行实例的返回值和循环统计。
    pub instances: Vec<CpuExecResult>,
    /// 程序执行后资源 buffer 的快照。
    pub buffers: BTreeMap<u32, Vec<Value>>,
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

    /// 通过 UCF CPU 解释器执行通用程序。
    ///
    /// 该入口不接受领域回调，也不根据任务名称选择实现。调用方必须提供
    /// 程序正文、并行实例数和 uniform 值，返回值中的哈希来自实际程序正文。
    pub fn execute_program(
        &self,
        program: &Program,
        count: u32,
        uniforms: impl IntoIterator<Item = (u32, Value)>,
    ) -> std::result::Result<ProgramExecutionReport, String> {
        self.execute_program_with_buffers(program, count, uniforms, [])
    }

    /// 执行程序并提供可读写的逻辑资源 buffer。
    pub fn execute_program_with_buffers(
        &self,
        program: &Program,
        count: u32,
        uniforms: impl IntoIterator<Item = (u32, Value)>,
        buffers: impl IntoIterator<Item = (u32, Vec<Value>)>,
    ) -> std::result::Result<ProgramExecutionReport, String> {
        let mut interpreter = CpuInterpreter::new();
        for (slot, value) in uniforms {
            interpreter.with_uniform(slot, value);
        }
        for (resource, values) in buffers {
            interpreter.with_buffer(resource, values);
        }
        let instances = interpreter
            .run_parallel(program, count)
            .map_err(|error| error.to_string())?;
        let mut output_buffers = BTreeMap::new();
        for resource in interpreter.resource_ids() {
            if let Some(values) = interpreter.buffer(resource) {
                output_buffers.insert(resource, values.to_vec());
            }
        }
        Ok(ProgramExecutionReport {
            program_hash: program_hash(program),
            instances,
            buffers: output_buffers,
        })
    }

    /// 从序列化程序执行，并校验调用方绑定的哈希。
    pub fn execute_program_bytes(
        &self,
        bytes: &[u8],
        expected_hash: u64,
        count: u32,
        uniforms: impl IntoIterator<Item = (u32, Value)>,
    ) -> std::result::Result<ProgramExecutionReport, String> {
        let program: Program = serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
        let actual_hash = program_hash(&program);
        if actual_hash != expected_hash {
            return Err(format!(
                "program hash mismatch: expected {expected_hash}, actual {actual_hash}"
            ));
        }
        self.execute_program(&program, count, uniforms)
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
            TaskKind::Dispatch => self.run_program_task(task).map_err(Self::map_err),
            TaskKind::Raster => self.run_raster(task).map_err(Self::map_err),
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
    fn run_program_task(&mut self, task: &TaskNode) -> std::result::Result<(), BackendError> {
        let (version, expected_hash, body) = match task.params.get("program") {
            Some(ParamValue::Program { version, hash, body }) => (*version, *hash, body),
            _ => {
                return Err(BackendError(
                    "dispatch task is missing a serialized UCF program".into(),
                ));
            }
        };
        if version == 0 {
            return Err(BackendError("UCF program version must be non-zero".into()));
        }
        let dispatch_count = match task.dispatch {
            Dispatch::Explicit { x, y, z } => x
                .checked_mul(y)
                .and_then(|value| value.checked_mul(z))
                .ok_or_else(|| BackendError("dispatch dimensions overflow".into()))?,
            Dispatch::Auto => {
                return Err(BackendError(
                    "generic UCF dispatch requires explicit dimensions".into(),
                ));
            }
        };
        let mut uniforms = Vec::new();
        for slot in 0..64u32 {
            let key = format!("uniform_{slot}");
            if let Some(value) = task.params.get(&key) {
                let value = match value {
                    ParamValue::F64(value) => *value,
                    ParamValue::I64(value) => *value as f64,
                    _ => {
                        return Err(BackendError(format!(
                            "{key} must be numeric for generic UCF dispatch"
                        )));
                    }
                };
                uniforms.push((slot, Value::F64(value)));
            }
        }
        let mut bound_buffers = Vec::new();
        for slot in 0..64u32 {
            let key = format!("buffer_{slot}");
            if let Some(ParamValue::I64(resource)) = task.params.get(&key) {
                let resource = ResourceId(u64::try_from(*resource).map_err(|_| {
                    BackendError(format!("{key} resource id must be non-negative"))
                })?);
                let values: Vec<Value> = self
                    .store
                    .lock()
                    .map_err(|_| BackendError("cpu store lock poisoned".into()))?
                    .read_f32(resource)?
                    .into_iter()
                    .map(|value| Value::F32(value))
                    .collect();
                bound_buffers.push((slot, resource, values));
            }
        }
        let program: Program = serde_json::from_str(body)
            .map_err(|error| BackendError(format!("invalid UCF program body: {error}")))?;
        let report = self
            .execute_program_with_buffers(
                &program,
                dispatch_count,
                uniforms,
                bound_buffers.iter().map(|(slot, _, values)| (*slot, values.clone())),
            )
            .map_err(BackendError)?;
        if report.program_hash != expected_hash {
            return Err(BackendError(format!(
                "program hash mismatch: expected {expected_hash}, actual {}",
                report.program_hash
            )));
        }
        {
            let mut store = self
                .store
                .lock()
                .map_err(|_| BackendError("cpu store lock poisoned".into()))?;
            for (slot, resource, _) in &bound_buffers {
                if let Some(values) = report.buffers.get(slot) {
                    let values: Vec<f32> = values
                        .iter()
                        .map(|value| {
                            value
                                .as_f64()
                                .map(|value| value as f32)
                                .map_err(|error| BackendError(error))
                        })
                        .collect::<std::result::Result<_, _>>()?;
                    store.write_f32(*resource, &values)?;
                }
            }
        }
        if let Some(ParamValue::I64(output)) = task.params.get("output") {
            let output = ResourceId(u64::try_from(*output).map_err(|_| {
                BackendError("generic UCF output resource id must be non-negative".into())
            })?);
            let values: Vec<f32> = report
                .instances
                .iter()
                .map(|result| {
                    result
                        .value
                        .as_f64()
                        .map(|value| value as f32)
                        .map_err(|error| BackendError(error))
                })
                .collect::<std::result::Result<_, _>>()?;
            let mut store = self
                .store
                .lock()
                .map_err(|_| BackendError("cpu store lock poisoned".into()))?;
            store.write_f32_prefix(output, &values)?;
        }
        Ok(())
    }

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

    /// Raster 清屏参考：`draw=clear` 写紧凑 RGBA8 到 `dst`（与 DX12 读回布局对齐）。
    fn run_raster(&mut self, task: &TaskNode) -> std::result::Result<(), BackendError> {
        use ucf_ir::ParamValue;

        let draw = match task.params.get("draw") {
            Some(ParamValue::Str(s)) => s.as_str(),
            None => "clear",
            Some(other) => {
                return Err(BackendError(format!(
                    "task {} param `draw` must be string, got {other:?}",
                    task.id.0
                )));
            }
        };
        if draw != "clear" && draw != "tri" {
            return Err(BackendError(format!(
                "cpu raster draw must be `clear` or `tri`, got `{draw}`"
            )));
        }

        let dst = resource_param(task, "dst")?;
        let width = u32_param(task, "width")? as usize;
        let height = u32_param(task, "height")? as usize;
        if width == 0 || height == 0 {
            return Err(BackendError("raster width/height must be > 0".into()));
        }
        let r = optional_f64_param(task, "r", 0.0);
        let g = optional_f64_param(task, "g", 0.0);
        let b = optional_f64_param(task, "b", 0.0);
        let a = optional_f64_param(task, "a", 1.0);
        let clear = [
            (r.clamp(0.0, 1.0) * 255.0).round() as u8,
            (g.clamp(0.0, 1.0) * 255.0).round() as u8,
            (b.clamp(0.0, 1.0) * 255.0).round() as u8,
            (a.clamp(0.0, 1.0) * 255.0).round() as u8,
        ];
        let pixels = if draw == "tri" {
            raster_tri::raster_clear_tri_rgba8(width, height, clear, raster_tri::TRI_RGBA)
                .map_err(|msg| BackendError(msg.into()))?
        } else {
            let need = width
                .checked_mul(height)
                .and_then(|p| p.checked_mul(4))
                .ok_or_else(|| BackendError("raster pixel count overflow".into()))?;
            let mut solid = Vec::with_capacity(need);
            for _ in 0..(width * height) {
                solid.extend_from_slice(&clear);
            }
            solid
        };
        let need = pixels.len();

        let mut store = self
            .store
            .lock()
            .map_err(|_| BackendError("cpu store lock poisoned".into()))?;
        let bytes = store.bytes_mut(dst)?;
        if bytes.len() < need {
            return Err(BackendError(format!(
                "resource {} has {} bytes, raster needs {need}",
                dst.0,
                bytes.len()
            )));
        }
        bytes[..need].copy_from_slice(&pixels);
        Ok(())
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

#[cfg(test)]
mod program_tests {
    use super::*;
    use ucf_program::{BinOp, Expr, Function, Program, ScalarTy, Stmt, Ty};
    use ucf_types::{Access, Domain, Objective, Priority, ResourceKind, ResourceNode, ShaderId, TaskId};

    #[test]
    fn execute_program_uses_interpreter_and_reports_hash() {
        let program = Program {
            version: 1,
            functions: vec![Function {
                name: "entry".into(),
                params: Vec::new(),
                locals: vec![(1, Ty::f64())],
                body: vec![
                    Stmt::Let {
                        id: 1,
                        ty: Ty::f64(),
                        init: Expr::BinOp {
                            op: BinOp::Add,
                            lhs: Box::new(Expr::LoadUniform(0)),
                            rhs: Box::new(Expr::GlobalId),
                        },
                    },
                    Stmt::Return(Expr::LoadLocal(1)),
                ],
                ret: Ty::Scalar(ScalarTy::F64),
            }],
            entry: 0,
        };
        let backend = CpuBackend::new();
        let report = backend
            .execute_program(&program, 3, [(0, Value::F64(2.0))])
            .expect("program execution");
        assert_eq!(report.program_hash, program_hash(&program));
        assert_eq!(report.instances[0].value, Value::F64(2.0));
        assert_eq!(report.instances[2].value, Value::F64(4.0));

        let bytes = serde_json::to_vec(&program).expect("encode program");
        let decoded = backend
            .execute_program_bytes(&bytes, report.program_hash, 1, [(0, Value::F64(5.0))])
            .expect("serialized program execution");
        assert_eq!(decoded.instances[0].value, Value::F64(5.0));
        assert!(backend
            .execute_program_bytes(&bytes, report.program_hash.wrapping_add(1), 1, [])
            .is_err());
    }

    #[test]
    fn dispatch_task_executes_program_body_and_writes_output_resource() {
        let program = Program {
            version: 1,
            functions: vec![Function {
                name: "global_id".into(),
                params: Vec::new(),
                locals: Vec::new(),
                body: vec![
                    Stmt::StoreBuffer {
                        resource: 1,
                        index: Expr::GlobalId,
                        value: Expr::LoadBuffer {
                            resource: 0,
                            index: Box::new(Expr::GlobalId),
                        },
                    },
                    Stmt::Return(Expr::GlobalId),
                ],
                ret: Ty::Scalar(ScalarTy::U32),
            }],
            entry: 0,
        };
        let hash = program_hash(&program);
        let body = serde_json::to_string(&program).expect("encode program");
        let mut params = BTreeMap::new();
        params.insert(
            "program".into(),
            ParamValue::Program { version: 1, hash, body },
        );
        params.insert("output".into(), ParamValue::I64(7));
        params.insert("buffer_0".into(), ParamValue::I64(8));
        params.insert("buffer_1".into(), ParamValue::I64(9));
        let task = TaskNode {
            id: TaskId(1),
            kind: TaskKind::Dispatch,
            shader: ShaderId(1),
            params,
            dispatch: Dispatch::Explicit { x: 3, y: 1, z: 1 },
            objective: Objective::MaxUtilization,
            priority: Priority::Batch,
        };
        let graph = Graph {
            resources: ucf_ir::ResourceGraph {
                nodes: [7, 8, 9]
                    .into_iter()
                    .map(|id| ResourceNode {
                        id: ResourceId(id),
                        kind: ResourceKind::Buffer,
                        domain: Domain::Host,
                        access: Access::ReadWrite,
                        byte_size: Some(12),
                    })
                    .collect(),
            },
            tasks: ucf_ir::TaskGraph { nodes: vec![task.clone()], edges: Vec::new() },
        };
        let mut backend = CpuBackend::new();
        backend.prepare(&graph).expect("prepare graph");
        backend
            .store()
            .lock()
            .unwrap()
            .write_f32(ResourceId(8), &[4.0, 5.0, 6.0])
            .unwrap();
        backend.submit_task(&graph, &task).expect("dispatch program");
        assert_eq!(backend.store().lock().unwrap().read_f32(ResourceId(7)).unwrap(), vec![0.0, 1.0, 2.0]);
        assert_eq!(backend.store().lock().unwrap().read_f32(ResourceId(9)).unwrap(), vec![4.0, 5.0, 6.0]);
    }
}
