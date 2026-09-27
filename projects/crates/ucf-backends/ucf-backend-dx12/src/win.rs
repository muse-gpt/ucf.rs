use ucf_capability::{Feature, FeatureSet};
use ucf_emitter::{emit_dxil, needs_kernel, program_from_task};
use ucf_ir::Graph;
use ucf_scheduler::{Backend, Result};
use ucf_types::{Dispatch, TaskKind, TaskNode};
use windows::core::Interface;
use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_12_0, D3D_FEATURE_LEVEL_12_1};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_COMMAND_LIST_TYPE, D3D12_COMMAND_LIST_TYPE_COMPUTE, D3D12_COMMAND_LIST_TYPE_DIRECT,
    D3D12_COMMAND_QUEUE_DESC, D3D12_COMMAND_QUEUE_FLAG_NONE, D3D12_COMPUTE_PIPELINE_STATE_DESC,
    D3D12_ROOT_SIGNATURE_DESC, D3D12_ROOT_SIGNATURE_FLAG_NONE, D3D12_SHADER_BYTECODE,
    D3D12SerializeRootSignature, ID3D12CommandAllocator, ID3D12CommandQueue, ID3D12Device,
    ID3D12Fence, ID3D12GraphicsCommandList, ID3D12PipelineState, ID3D12RootSignature,
    D3D12CreateDevice, D3D_ROOT_SIGNATURE_VERSION_1,
};
use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIAdapter1, IDXGIFactory4};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject, INFINITE};

/// DirectX 12 backend via raw D3D12 calls.
///
/// Compute shaders are emitted by [`ucf_emitter`] as DXIL containers (not HLSL + `D3DCompile`).
pub struct Dx12Backend {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    direct_allocator: ID3D12CommandAllocator,
    compute_allocator: ID3D12CommandAllocator,
    fence: ID3D12Fence,
    fence_value: u64,
    fence_event: HANDLE,
}

impl Dx12Backend {
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
            let queue: ID3D12CommandQueue = device.CreateCommandQueue(&queue_desc).map_err(dx_err)?;

            let direct_allocator: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .map_err(dx_err)?;
            let compute_allocator: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE)
                .map_err(dx_err)?;

            let fence: ID3D12Fence = device
                .CreateFence(
                    0,
                    windows::Win32::Graphics::Direct3D12::D3D12_FENCE_FLAG_NONE,
                )
                .map_err(dx_err)?;
            let fence_event = CreateEventW(None, false, false, None).map_err(dx_err)?;

            Ok(Self {
                device,
                queue,
                direct_allocator,
                compute_allocator,
                fence,
                fence_value: 0,
                fence_event,
            })
        }
    }

    fn list_type_for(task: &TaskNode) -> D3D12_COMMAND_LIST_TYPE {
        match task.kind {
            TaskKind::Dispatch | TaskKind::MatMul | TaskKind::Custom(_) => {
                D3D12_COMMAND_LIST_TYPE_COMPUTE
            }
            _ => D3D12_COMMAND_LIST_TYPE_DIRECT,
        }
    }

    fn allocator_for(&self, list_type: D3D12_COMMAND_LIST_TYPE) -> &ID3D12CommandAllocator {
        if list_type == D3D12_COMMAND_LIST_TYPE_COMPUTE {
            &self.compute_allocator
        } else {
            &self.direct_allocator
        }
    }

    fn run_copy(&mut self, task: &TaskNode) -> Result<()> {
        let list_type = Self::list_type_for(task);
        self.execute_empty_list(list_type)
    }

    fn run_compute(&mut self, task: &TaskNode) -> Result<()> {
        let program = program_from_task(task);
        if !needs_kernel(&program) {
            return self.run_copy(task);
        }

        let dxil = emit_dxil(&program).map_err(map_emit_err)?;
        let pso = self.create_compute_pso(&dxil)?;
        self.dispatch_compute(&pso, task)
    }

    fn create_compute_pso(&self, dxil: &[u8]) -> Result<ID3D12PipelineState> {
        unsafe {
            let root_desc = D3D12_ROOT_SIGNATURE_DESC {
                NumParameters: 0,
                pParameters: std::ptr::null(),
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
            let bytes = std::slice::from_raw_parts(
                blob.GetBufferPointer().cast(),
                blob.GetBufferSize(),
            );
            let root: ID3D12RootSignature = self
                .device
                .CreateRootSignature(0, bytes)
                .map_err(dx_err)?;

            let cs = D3D12_SHADER_BYTECODE {
                pShaderBytecode: dxil.as_ptr().cast(),
                BytecodeLength: dxil.len(),
            };
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::ManuallyDrop::new(Some(root.clone())),
                CS: cs,
                NodeMask: 0,
                CachedPSO: Default::default(),
                Flags: Default::default(),
            };
            self.device
                .CreateComputePipelineState(&desc)
                .map_err(dx_err)
        }
    }

    fn dispatch_compute(&mut self, pso: &ID3D12PipelineState, task: &TaskNode) -> Result<()> {
        unsafe {
            self.compute_allocator.Reset().map_err(dx_err)?;
            let list: ID3D12GraphicsCommandList = self
                .device
                .CreateCommandList(
                    0,
                    D3D12_COMMAND_LIST_TYPE_COMPUTE,
                    &self.compute_allocator,
                    Some(pso),
                )
                .map_err(dx_err)?;

            let (x, y, z) = dispatch_xyz(task);
            list.SetPipelineState(pso);
            list.Dispatch(x, y, z);
            list.Close().map_err(dx_err)?;

            let lists = [Some(list.cast().map_err(dx_err)?)];
            self.queue.ExecuteCommandLists(&lists);
            self.signal_fence()
        }
    }

    fn execute_empty_list(&mut self, list_type: D3D12_COMMAND_LIST_TYPE) -> Result<()> {
        unsafe {
            let allocator = self.allocator_for(list_type);
            allocator.Reset().map_err(dx_err)?;
            let list: ID3D12GraphicsCommandList = self
                .device
                .CreateCommandList(0, list_type, allocator, None)
                .map_err(dx_err)?;
            list.Close().map_err(dx_err)?;

            let lists = [Some(list.cast().map_err(dx_err)?)];
            self.queue.ExecuteCommandLists(&lists);
            self.signal_fence()
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
            .with(Feature::RayTracing)
            .with(Feature::WorkLists)
            .with(Feature::EnhancedBarriers)
    }

    fn submit_task(&mut self, graph: &Graph, task: &TaskNode) -> Result<()> {
        graph.validate().map_err(ucf_scheduler::Error::from)?;
        match task.kind {
            TaskKind::Copy => self.run_copy(task),
            TaskKind::MatMul | TaskKind::Dispatch | TaskKind::Custom(_) => self.run_compute(task),
            TaskKind::Raster | TaskKind::RtTrace => {
                self.execute_empty_list(D3D12_COMMAND_LIST_TYPE_DIRECT)
            }
        }
    }
}

fn dispatch_xyz(task: &TaskNode) -> (u32, u32, u32) {
    match task.dispatch {
        Dispatch::Auto => (1, 1, 1),
        Dispatch::Explicit { x, y, z } => (x, y, z),
    }
}

fn dx_err(error: windows::core::Error) -> ucf_scheduler::Error {
    ucf_scheduler::Error::Backend("dx12".into(), error.to_string())
}

fn dx_err_msg(message: impl Into<String>) -> ucf_scheduler::Error {
    ucf_scheduler::Error::Backend("dx12".into(), message.into())
}

fn map_emit_err(error: String) -> ucf_scheduler::Error {
    ucf_scheduler::Error::Backend("dx12".into(), error)
}
