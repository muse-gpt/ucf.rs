use ucf_capability::{Feature, FeatureSet};
use ucf_emitter::{emit_ptx, needs_kernel, program_from_task};
use ucf_ir::Graph;
use ucf_scheduler::{Backend, Result};
use ucf_types::{Dispatch, TaskKind, TaskNode};

use crate::driver::{CudaDriver, DriverError};

/// NVIDIA backend via CUDA Driver API (`nvcuda.dll` / `libcuda.so`).
///
/// Shaders are emitted by [`ucf_emitter`] as PTX. SASS is produced by the driver JIT.
pub struct CudaBackend {
    driver: CudaDriver,
}

impl CudaBackend {
    pub fn new(device_index: u32) -> Result<Self> {
        let driver = CudaDriver::new(device_index).map_err(map_driver_err)?;
        Ok(Self { driver })
    }

    fn run_copy(&mut self, task: &TaskNode) -> Result<()> {
        let bytes = element_bytes(task);
        let ptr = self.driver.mem_alloc(bytes).map_err(map_driver_err)?;
        let host = vec![0u8; bytes];
        self.driver
            .memcpy_htod(ptr, &host)
            .map_err(map_driver_err)?;
        self.driver.mem_free(ptr).map_err(map_driver_err)?;
        Ok(())
    }

    fn run_kernel(&mut self, task: &TaskNode) -> Result<()> {
        let program = program_from_task(task);
        if !needs_kernel(&program) {
            return self.run_copy(task);
        }

        let ptx = emit_ptx(&program).map_err(map_emit_err)?;
        let module = self.driver.load_module(&ptx).map_err(map_driver_err)?;
        let func = self
            .driver
            .get_function(module, &program.entry)
            .map_err(map_driver_err)?;

        let count = element_count(task) as u32;
        let byte_len = count as usize * std::mem::size_of::<f32>();
        let out = self.driver.mem_alloc(byte_len).map_err(map_driver_err)?;
        self.driver
            .launch_iota_fill(func, count, out)
            .map_err(map_driver_err)?;
        self.driver.mem_free(out).map_err(map_driver_err)?;
        self.driver.unload_module(module).map_err(map_driver_err)?;
        Ok(())
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

    fn submit_task(&mut self, graph: &Graph, task: &TaskNode) -> Result<()> {
        graph.validate().map_err(ucf_scheduler::Error::from)?;
        match task.kind {
            TaskKind::Copy => self.run_copy(task),
            TaskKind::MatMul | TaskKind::Dispatch | TaskKind::Custom(_) => self.run_kernel(task),
            TaskKind::Raster | TaskKind::RtTrace => Err(backend_err(
                "raster and ray tracing are not supported on CUDA Driver API",
            )),
        }
    }
}

fn element_count(task: &TaskNode) -> usize {
    match task.dispatch {
        Dispatch::Auto => 256,
        Dispatch::Explicit { x, y, z } => {
            let count = x as u64 * y as u64 * z as u64;
            count.clamp(1, 1_048_576) as usize
        }
    }
}

fn element_bytes(task: &TaskNode) -> usize {
    element_count(task) * std::mem::size_of::<f32>()
}

fn map_driver_err(error: DriverError) -> ucf_scheduler::Error {
    ucf_scheduler::Error::Backend("cuda".into(), error.to_string())
}

fn map_emit_err(error: String) -> ucf_scheduler::Error {
    ucf_scheduler::Error::Backend("cuda".into(), error)
}

fn backend_err(message: impl Into<String>) -> ucf_scheduler::Error {
    ucf_scheduler::Error::Backend("cuda".into(), message.into())
}
