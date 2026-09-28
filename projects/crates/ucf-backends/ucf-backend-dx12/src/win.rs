use std::collections::BTreeMap;

use ucf_capability::{Feature, FeatureSet};
use ucf_emitter::{emit_dxil, program_from_task};
use ucf_ir::Graph;
use ucf_scheduler::{Backend, Error as SchedulerError, Result};
use ucf_types::{ResourceId, TaskKind, TaskNode};
use windows::core::Interface;
use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_12_0, D3D_FEATURE_LEVEL_12_1};
use windows::Win32::Graphics::Direct3D12::{
    D3D12SerializeRootSignature, D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC,
    D3D12_COMMAND_QUEUE_FLAG_NONE, D3D12_COMPUTE_PIPELINE_STATE_DESC, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
    D3D12_FENCE_FLAG_NONE, D3D12_HEAP_FLAG_NONE, D3D12_HEAP_FLAG_SHARED, D3D12_HEAP_FLAGS,
    D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_DEFAULT,
    D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_TYPE_UPLOAD, D3D12_MEMORY_POOL_UNKNOWN,
    D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
    D3D12_RESOURCE_BARRIER_FLAG_NONE, D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
    D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATES, D3D12_RESOURCE_STATE_COMMON,
    D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE,
    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    D3D12_RESOURCE_TRANSITION_BARRIER, D3D12_ROOT_CONSTANTS, D3D12_ROOT_DESCRIPTOR,
    D3D12_ROOT_PARAMETER, D3D12_ROOT_PARAMETER_0, D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
    D3D12_ROOT_PARAMETER_TYPE_SRV, D3D12_ROOT_PARAMETER_TYPE_UAV, D3D12_ROOT_SIGNATURE_DESC,
    D3D12_ROOT_SIGNATURE_FLAG_NONE, D3D12_SHADER_BYTECODE, D3D12_SHADER_VISIBILITY_ALL,
    D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D_ROOT_SIGNATURE_VERSION_1, ID3D12CommandAllocator,
    ID3D12CommandQueue, ID3D12Device, ID3D12Fence, ID3D12GraphicsCommandList, ID3D12PipelineState,
    ID3D12Resource, ID3D12RootSignature, D3D12CreateDevice,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};
use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIAdapter1, IDXGIFactory4};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject, INFINITE};

use crate::params::{f32_param, resource_param, u32_param, BackendError};

struct DeviceBuffer {
    resource: ID3D12Resource,
    bytes: usize,
    state: D3D12_RESOURCE_STATES,
}

struct SharedBuffer {
    resource: ID3D12Resource,
    handle: HANDLE,
    bytes: usize,
    state: D3D12_RESOURCE_STATES,
}

/// Opaque id for a DX12 buffer exported via NT shared handle (CUDA import).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SharedBufferId(pub u64);

/// DirectX 12 backend with device-resident buffers and compute dispatch.
pub struct Dx12Backend {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    fence: ID3D12Fence,
    fence_value: u64,
    fence_event: HANDLE,
    buffers: BTreeMap<ResourceId, DeviceBuffer>,
    shared: BTreeMap<SharedBufferId, SharedBuffer>,
    next_shared: u64,
}

impl Dx12Backend {
    /// Opens the default DXGI adapter and creates a Direct3D 12 device and queue.
    pub fn new() -> Result<Self> {
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().map_err(dx_err)?;
            let adapter: IDXGIAdapter1 = factory
                .EnumAdapters(0)
                .map_err(dx_err)?
                .cast()
                .map_err(dx_err)?;
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_1, &mut device)
                .or_else(|_| D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device))
                .map_err(dx_err)?;
            let device = device.ok_or_else(|| dx_err_msg("D3D12CreateDevice returned null"))?;

            let queue_desc = D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                Priority: 0,
                Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
                NodeMask: 0,
            };
            let queue: ID3D12CommandQueue =
                device.CreateCommandQueue(&queue_desc).map_err(dx_err)?;
            let allocator: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .map_err(dx_err)?;
            let fence: ID3D12Fence = device
                .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                .map_err(dx_err)?;
            let fence_event = CreateEventW(None, false, false, None).map_err(dx_err)?;

            Ok(Self {
                device,
                queue,
                allocator,
                fence,
                fence_value: 0,
                fence_event,
                buffers: BTreeMap::new(),
                shared: BTreeMap::new(),
                next_shared: 1,
            })
        }
    }

    /// Allocate a `HEAP_FLAG_SHARED` buffer and return its id + Win32 NT handle for CUDA import.
    pub fn shared_alloc(&mut self, bytes: usize) -> Result<(SharedBufferId, isize)> {
        unsafe {
            let resource = self.create_buffer(
                bytes,
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_HEAP_FLAG_SHARED,
            )?;
            let handle = self
                .device
                .CreateSharedHandle(&resource, None, 0x1000_0000u32, None)
                .map_err(dx_err)?;
            let id = SharedBufferId(self.next_shared);
            self.next_shared += 1;
            self.shared.insert(
                id,
                SharedBuffer {
                    resource,
                    handle,
                    bytes,
                    state: D3D12_RESOURCE_STATE_COMMON,
                },
            );
            let zeros = vec![0u8; bytes];
            self.shared_upload(id, &zeros)?;
            Ok((id, handle.0 as isize))
        }
    }

    /// Upload host `f32` values into a shared buffer.
    pub fn shared_write_f32(&mut self, id: SharedBufferId, values: &[f32]) -> Result<()> {
        let bytes = values.len() * 4;
        let buf = self.shared.get(&id).ok_or_else(|| {
            map_backend_err(BackendError(format!("shared buffer {} not allocated", id.0)))
        })?;
        if buf.bytes != bytes {
            return Err(map_backend_err(BackendError(format!(
                "shared buffer {} expected {} bytes, got {bytes}",
                id.0, buf.bytes
            ))));
        }
        let mut host = Vec::with_capacity(bytes);
        for v in values {
            host.extend_from_slice(&v.to_le_bytes());
        }
        self.shared_upload(id, &host)
    }

    /// NT handle previously returned by [`shared_alloc`] (still owned by this backend).
    pub fn shared_nt_handle(&self, id: SharedBufferId) -> Result<isize> {
        let buf = self.shared.get(&id).ok_or_else(|| {
            map_backend_err(BackendError(format!("shared buffer {} not allocated", id.0)))
        })?;
        Ok(buf.handle.0 as isize)
    }

    fn shared_upload(&mut self, id: SharedBufferId, bytes: &[u8]) -> Result<()> {
        unsafe {
            let upload = self.create_buffer(
                bytes.len(),
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ_UPLOAD(),
                D3D12_HEAP_FLAG_NONE,
            )?;
            {
                let mut ptr = std::ptr::null_mut();
                upload.Map(0, None, Some(&mut ptr)).map_err(dx_err)?;
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast(), bytes.len());
                upload.Unmap(0, None);
            }

            self.allocator.Reset().map_err(dx_err)?;
            let list: ID3D12GraphicsCommandList = self
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &self.allocator, None)
                .map_err(dx_err)?;

            {
                let buf = self.shared.get_mut(&id).unwrap();
                transition(&list, &buf.resource, buf.state, D3D12_RESOURCE_STATE_COPY_DEST);
                buf.state = D3D12_RESOURCE_STATE_COPY_DEST;
                list.CopyBufferRegion(&buf.resource, 0, &upload, 0, bytes.len() as u64);
                transition(
                    &list,
                    &buf.resource,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_COMMON,
                );
                buf.state = D3D12_RESOURCE_STATE_COMMON;
            }

            list.Close().map_err(dx_err)?;
            let lists = [Some(list.cast().map_err(dx_err)?)];
            self.queue.ExecuteCommandLists(&lists);
            self.signal_fence()?;
            Ok(())
        }
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
        self.upload(id, &host)
    }

    /// Download a device buffer as `f32` values.
    pub fn read_f32(&mut self, id: ResourceId) -> Result<Vec<f32>> {
        let bytes = self
            .buffers
            .get(&id)
            .ok_or_else(|| {
                map_backend_err(BackendError(format!("resource {} not allocated", id.0)))
            })?
            .bytes;
        let host = self.readback(id, bytes)?;
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

    fn upload(&mut self, id: ResourceId, bytes: &[u8]) -> Result<()> {
        unsafe {
            let upload = self.create_buffer(
                bytes.len(),
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ_UPLOAD(),
                D3D12_HEAP_FLAG_NONE,
            )?;
            {
                let mut ptr = std::ptr::null_mut();
                upload
                    .Map(0, None, Some(&mut ptr))
                    .map_err(dx_err)?;
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast(), bytes.len());
                upload.Unmap(0, None);
            }

            self.allocator.Reset().map_err(dx_err)?;
            let list: ID3D12GraphicsCommandList = self
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &self.allocator, None)
                .map_err(dx_err)?;

            {
                let buf = self.buffers.get_mut(&id).unwrap();
                transition(&list, &buf.resource, buf.state, D3D12_RESOURCE_STATE_COPY_DEST);
                buf.state = D3D12_RESOURCE_STATE_COPY_DEST;
                list.CopyBufferRegion(&buf.resource, 0, &upload, 0, bytes.len() as u64);
                transition(
                    &list,
                    &buf.resource,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_COMMON,
                );
                buf.state = D3D12_RESOURCE_STATE_COMMON;
            }

            list.Close().map_err(dx_err)?;
            let lists = [Some(list.cast().map_err(dx_err)?)];
            self.queue.ExecuteCommandLists(&lists);
            self.signal_fence()?;
            Ok(())
        }
    }

    fn readback(&mut self, id: ResourceId, bytes: usize) -> Result<Vec<u8>> {
        unsafe {
            let readback = self.create_buffer(
                bytes,
                D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_HEAP_FLAG_NONE,
            )?;

            self.allocator.Reset().map_err(dx_err)?;
            let list: ID3D12GraphicsCommandList = self
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &self.allocator, None)
                .map_err(dx_err)?;

            {
                let buf = self.buffers.get_mut(&id).unwrap();
                transition(&list, &buf.resource, buf.state, D3D12_RESOURCE_STATE_COPY_SOURCE);
                buf.state = D3D12_RESOURCE_STATE_COPY_SOURCE;
                list.CopyBufferRegion(&readback, 0, &buf.resource, 0, bytes as u64);
                transition(
                    &list,
                    &buf.resource,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                );
                buf.state = D3D12_RESOURCE_STATE_COMMON;
            }

            list.Close().map_err(dx_err)?;
            let lists = [Some(list.cast().map_err(dx_err)?)];
            self.queue.ExecuteCommandLists(&lists);
            self.signal_fence()?;

            let mut ptr = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut ptr)).map_err(dx_err)?;
            let mut out = vec![0u8; bytes];
            std::ptr::copy_nonoverlapping(ptr.cast(), out.as_mut_ptr(), bytes);
            readback.Unmap(0, None);
            Ok(out)
        }
    }

    fn create_buffer(
        &self,
        bytes: usize,
        heap_type: windows::Win32::Graphics::Direct3D12::D3D12_HEAP_TYPE,
        flags: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_FLAGS,
        initial: D3D12_RESOURCE_STATES,
        heap_flags: D3D12_HEAP_FLAGS,
    ) -> Result<ID3D12Resource> {
        unsafe {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: heap_type,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
                CreationNodeMask: 0,
                VisibleNodeMask: 0,
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Alignment: 0,
                Width: bytes.max(1) as u64,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_UNKNOWN,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: flags,
            };
            let mut resource: Option<ID3D12Resource> = None;
            self.device
                .CreateCommittedResource(
                    &heap,
                    heap_flags,
                    &desc,
                    initial,
                    None,
                    &mut resource,
                )
                .map_err(dx_err)?;
            resource.ok_or_else(|| dx_err_msg("CreateCommittedResource returned null"))
        }
    }

    fn gpu_va(&self, id: ResourceId) -> Result<u64> {
        let buf = self.buffers.get(&id).ok_or_else(|| {
            map_backend_err(BackendError(format!("resource {} not allocated", id.0)))
        })?;
        Ok(unsafe { buf.resource.GetGPUVirtualAddress() })
    }

    fn run_copy(&mut self, task: &TaskNode) -> Result<()> {
        let src = resource_param(task, "src").map_err(map_backend_err)?;
        let dst = resource_param(task, "dst").map_err(map_backend_err)?;
        let src_bytes = self.buffers.get(&src).map(|b| b.bytes).ok_or_else(|| {
            map_backend_err(BackendError(format!("resource {} not allocated", src.0)))
        })?;
        let dst_bytes = self.buffers.get(&dst).map(|b| b.bytes).ok_or_else(|| {
            map_backend_err(BackendError(format!("resource {} not allocated", dst.0)))
        })?;
        if src_bytes != dst_bytes {
            return Err(map_backend_err(BackendError(format!(
                "copy size mismatch: src {src_bytes} dst {dst_bytes}"
            ))));
        }

        unsafe {
            self.allocator.Reset().map_err(dx_err)?;
            let list: ID3D12GraphicsCommandList = self
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &self.allocator, None)
                .map_err(dx_err)?;

            let src_res = self.buffers.get(&src).unwrap().resource.clone();
            let dst_res = self.buffers.get(&dst).unwrap().resource.clone();
            {
                let src_buf = self.buffers.get_mut(&src).unwrap();
                transition(&list, &src_res, src_buf.state, D3D12_RESOURCE_STATE_COPY_SOURCE);
                src_buf.state = D3D12_RESOURCE_STATE_COPY_SOURCE;
            }
            {
                let dst_buf = self.buffers.get_mut(&dst).unwrap();
                transition(&list, &dst_res, dst_buf.state, D3D12_RESOURCE_STATE_COPY_DEST);
                dst_buf.state = D3D12_RESOURCE_STATE_COPY_DEST;
            }
            list.CopyBufferRegion(&dst_res, 0, &src_res, 0, src_bytes as u64);
            {
                let src_buf = self.buffers.get_mut(&src).unwrap();
                transition(
                    &list,
                    &src_res,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                );
                src_buf.state = D3D12_RESOURCE_STATE_COMMON;
            }
            {
                let dst_buf = self.buffers.get_mut(&dst).unwrap();
                transition(
                    &list,
                    &dst_res,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_COMMON,
                );
                dst_buf.state = D3D12_RESOURCE_STATE_COMMON;
            }

            list.Close().map_err(dx_err)?;
            let lists = [Some(list.cast().map_err(dx_err)?)];
            self.queue.ExecuteCommandLists(&lists);
            self.signal_fence()
        }
    }

    fn run_fill(&mut self, task: &TaskNode) -> Result<()> {
        let dst = resource_param(task, "dst").map_err(map_backend_err)?;
        let _value = f32_param(task, "value").map_err(map_backend_err)?;
        let bytes = self.buffers.get(&dst).map(|b| b.bytes).ok_or_else(|| {
            map_backend_err(BackendError(format!("resource {} not allocated", dst.0)))
        })?;
        if bytes % 4 != 0 {
            return Err(map_backend_err(BackendError(format!(
                "fill destination {} length {bytes} is not a multiple of 4",
                dst.0
            ))));
        }
        let count = (bytes / 4) as u32;
        let program = program_from_task(task);
        let dxbc = emit_dxil(&program).map_err(map_emit_err)?;
        let (root, pso) = self.create_fill_pso(&dxbc)?;
        let dst_va = self.gpu_va(dst)?;

        unsafe {
            self.allocator.Reset().map_err(dx_err)?;
            let list: ID3D12GraphicsCommandList = self
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &self.allocator, Some(&pso))
                .map_err(dx_err)?;

            {
                let buf = self.buffers.get_mut(&dst).unwrap();
                transition(
                    &list,
                    &buf.resource,
                    buf.state,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                );
                buf.state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            }

            list.SetComputeRootSignature(&root);
            list.SetPipelineState(&pso);
            list.SetComputeRoot32BitConstant(0, count, 0);
            list.SetComputeRootUnorderedAccessView(1, dst_va);
            let groups = count.div_ceil(64).max(1);
            list.Dispatch(groups, 1, 1);

            {
                let buf = self.buffers.get_mut(&dst).unwrap();
                transition(
                    &list,
                    &buf.resource,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COMMON,
                );
                buf.state = D3D12_RESOURCE_STATE_COMMON;
            }

            list.Close().map_err(dx_err)?;
            let lists = [Some(list.cast().map_err(dx_err)?)];
            self.queue.ExecuteCommandLists(&lists);
            self.signal_fence()
        }
    }

    fn run_matmul(&mut self, task: &TaskNode) -> Result<()> {
        let a = resource_param(task, "a").map_err(map_backend_err)?;
        let b = resource_param(task, "b").map_err(map_backend_err)?;
        let out = resource_param(task, "out").map_err(map_backend_err)?;
        let m = u32_param(task, "m").map_err(map_backend_err)?;
        let n = u32_param(task, "n").map_err(map_backend_err)?;
        let k = u32_param(task, "k").map_err(map_backend_err)?;
        expect_bytes(a, self.buf_bytes(a)?, (m as usize) * (k as usize) * 4)?;
        expect_bytes(b, self.buf_bytes(b)?, (k as usize) * (n as usize) * 4)?;
        expect_bytes(out, self.buf_bytes(out)?, (m as usize) * (n as usize) * 4)?;

        let program = program_from_task(task);
        let dxbc = emit_dxil(&program).map_err(map_emit_err)?;
        let (root, pso) = self.create_matmul_pso(&dxbc)?;
        let a_va = self.gpu_va(a)?;
        let b_va = self.gpu_va(b)?;
        let out_va = self.gpu_va(out)?;

        unsafe {
            self.allocator.Reset().map_err(dx_err)?;
            let list: ID3D12GraphicsCommandList = self
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &self.allocator, Some(&pso))
                .map_err(dx_err)?;

            for id in [a, b] {
                let buf = self.buffers.get_mut(&id).unwrap();
                transition(
                    &list,
                    &buf.resource,
                    buf.state,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                );
                buf.state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            }
            {
                let buf = self.buffers.get_mut(&out).unwrap();
                transition(
                    &list,
                    &buf.resource,
                    buf.state,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                );
                buf.state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            }

            list.SetComputeRootSignature(&root);
            list.SetPipelineState(&pso);
            let constants = [m, n, k, 0u32];
            list.SetComputeRoot32BitConstants(
                0,
                constants.len() as u32,
                constants.as_ptr().cast(),
                0,
            );
            list.SetComputeRootShaderResourceView(1, a_va);
            list.SetComputeRootShaderResourceView(2, b_va);
            list.SetComputeRootUnorderedAccessView(3, out_va);
            let groups = (m * n).div_ceil(64).max(1);
            list.Dispatch(groups, 1, 1);

            for id in [a, b, out] {
                let buf = self.buffers.get_mut(&id).unwrap();
                let before = buf.state;
                transition(&list, &buf.resource, before, D3D12_RESOURCE_STATE_COMMON);
                buf.state = D3D12_RESOURCE_STATE_COMMON;
            }

            list.Close().map_err(dx_err)?;
            let lists = [Some(list.cast().map_err(dx_err)?)];
            self.queue.ExecuteCommandLists(&lists);
            self.signal_fence()
        }
    }

    fn buf_bytes(&self, id: ResourceId) -> Result<usize> {
        self.buffers
            .get(&id)
            .map(|b| b.bytes)
            .ok_or_else(|| map_backend_err(BackendError(format!("resource {} not allocated", id.0))))
    }

    fn create_fill_pso(&self, dxbc: &[u8]) -> Result<(ID3D12RootSignature, ID3D12PipelineState)> {
        let params = [
            root_constants(0, 1),
            root_uav(0),
        ];
        self.create_compute_pso(dxbc, &params)
    }

    fn create_matmul_pso(&self, dxbc: &[u8]) -> Result<(ID3D12RootSignature, ID3D12PipelineState)> {
        let params = [
            root_constants(0, 4),
            root_srv(0),
            root_srv(1),
            root_uav(0),
        ];
        self.create_compute_pso(dxbc, &params)
    }

    fn create_compute_pso(
        &self,
        dxbc: &[u8],
        params: &[D3D12_ROOT_PARAMETER],
    ) -> Result<(ID3D12RootSignature, ID3D12PipelineState)> {
        unsafe {
            let root_desc = D3D12_ROOT_SIGNATURE_DESC {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: 0,
                pStaticSamplers: std::ptr::null(),
                Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
            };
            let mut blob = None;
            let mut error_blob = None;
            D3D12SerializeRootSignature(
                &root_desc,
                D3D_ROOT_SIGNATURE_VERSION_1,
                &mut blob,
                Some(&mut error_blob),
            )
            .map_err(dx_err)?;
            let blob = blob.ok_or_else(|| dx_err_msg("D3D12SerializeRootSignature returned null"))?;
            let bytes =
                std::slice::from_raw_parts(blob.GetBufferPointer().cast(), blob.GetBufferSize());
            let root: ID3D12RootSignature =
                self.device.CreateRootSignature(0, bytes).map_err(dx_err)?;

            let cs = D3D12_SHADER_BYTECODE {
                pShaderBytecode: dxbc.as_ptr().cast(),
                BytecodeLength: dxbc.len(),
            };
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::ManuallyDrop::new(Some(root.clone())),
                CS: cs,
                NodeMask: 0,
                CachedPSO: Default::default(),
                Flags: Default::default(),
            };
            let pso = self
                .device
                .CreateComputePipelineState(&desc)
                .map_err(dx_err)?;
            Ok((root, pso))
        }
    }

    fn signal_fence(&mut self) -> Result<()> {
        unsafe {
            self.fence_value += 1;
            self.queue
                .Signal(&self.fence, self.fence_value)
                .map_err(dx_err)?;
            if self.fence.GetCompletedValue() < self.fence_value {
                self.fence
                    .SetEventOnCompletion(self.fence_value, self.fence_event)
                    .map_err(dx_err)?;
                let wait = WaitForSingleObject(self.fence_event, INFINITE);
                if wait != WAIT_OBJECT_0 {
                    return Err(dx_err_msg("fence wait failed"));
                }
            }
            Ok(())
        }
    }
}

// D3D12 COM interfaces and the fence event are owned here and used on the submit thread.
unsafe impl Send for Dx12Backend {}
unsafe impl Sync for Dx12Backend {}

impl Drop for Dx12Backend {
    fn drop(&mut self) {
        unsafe {
            let shared = std::mem::take(&mut self.shared);
            for (_, buf) in shared {
                let _ = CloseHandle(buf.handle);
            }
            let _ = CloseHandle(self.fence_event);
        }
    }
}

impl Default for Dx12Backend {
    fn default() -> Self {
        Self::new().expect("DX12 device initialization failed")
    }
}

impl Backend for Dx12Backend {
    fn name(&self) -> &str {
        "dx12"
    }

    fn features(&self) -> FeatureSet {
        FeatureSet::new()
            .with(Feature::Bindless)
            .with(Feature::DescriptorHeap)
            .with(Feature::DynamicRendering)
            .with(Feature::EnhancedBarriers)
    }

    fn prepare(&mut self, graph: &Graph) -> Result<()> {
        for node in &graph.resources.nodes {
            let bytes = node.byte_size.ok_or_else(|| {
                map_backend_err(BackendError(format!(
                    "resource {} needs byte_size for dx12 backend",
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
            let resource = self.create_buffer(
                bytes,
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_HEAP_FLAG_NONE,
            )?;
            // Zero-initialize via upload of zeros.
            self.buffers.insert(
                node.id,
                DeviceBuffer {
                    resource,
                    bytes,
                    state: D3D12_RESOURCE_STATE_COMMON,
                },
            );
            let zeros = vec![0u8; bytes];
            self.upload(node.id, &zeros)?;
        }
        Ok(())
    }

    fn submit_task(&mut self, _graph: &Graph, task: &TaskNode) -> Result<()> {
        match &task.kind {
            TaskKind::Copy => self.run_copy(task),
            TaskKind::Fill => self.run_fill(task),
            TaskKind::MatMul => self.run_matmul(task),
            TaskKind::Raster | TaskKind::RtTrace => Err(map_backend_err(BackendError(
                "raster and ray tracing are not in the dx12 thin gate".into(),
            ))),
            other => Err(map_backend_err(BackendError(format!(
                "dx12 backend does not implement {other:?}"
            )))),
        }
    }
}

fn root_constants(shader_register: u32, num: u32) -> D3D12_ROOT_PARAMETER {
    D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Constants: D3D12_ROOT_CONSTANTS {
                ShaderRegister: shader_register,
                RegisterSpace: 0,
                Num32BitValues: num,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }
}

fn root_uav(shader_register: u32) -> D3D12_ROOT_PARAMETER {
    D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR {
                ShaderRegister: shader_register,
                RegisterSpace: 0,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }
}

fn root_srv(shader_register: u32) -> D3D12_ROOT_PARAMETER {
    D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR {
                ShaderRegister: shader_register,
                RegisterSpace: 0,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }
}

fn transition(
    list: &ID3D12GraphicsCommandList,
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) {
    if before == after {
        return;
    }
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: std::mem::ManuallyDrop::new(Some(resource.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    };
    unsafe {
        list.ResourceBarrier(&[barrier]);
    }
}

/// Upload heaps start in `GENERIC_READ`.
#[allow(non_snake_case)]
fn D3D12_RESOURCE_STATE_GENERIC_READ_UPLOAD() -> D3D12_RESOURCE_STATES {
    windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_GENERIC_READ
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

fn dx_err(error: windows::core::Error) -> SchedulerError {
    SchedulerError::Backend("dx12".into(), error.to_string())
}

fn dx_err_msg(message: impl Into<String>) -> SchedulerError {
    SchedulerError::Backend("dx12".into(), message.into())
}

fn map_emit_err(error: String) -> SchedulerError {
    SchedulerError::Backend("dx12".into(), error)
}

fn map_backend_err(error: BackendError) -> SchedulerError {
    SchedulerError::Backend("dx12".into(), error.0)
}
