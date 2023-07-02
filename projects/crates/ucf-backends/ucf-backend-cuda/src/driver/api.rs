#![allow(non_snake_case)]

use std::ffi::{c_void, CString};
use std::path::Path;

use libloading::Library;

use super::ffi::*;

macro_rules! load_fn {
    ($lib:expr, $name:ident, $ty:ty) => {{
        let sym: libloading::Symbol<$ty> = {
            $lib.get(stringify!($name).as_bytes()).map_err(|e| {
                DriverError::new(-1, format!("symbol `{}`: {e}", stringify!($name)))
            })?
        };
        *sym
    }};
}

pub struct CudaDriver {
    _lib: Library,
    cuInit: CuInit,
    cuDeviceGet: CuDeviceGet,
    cuCtxCreate: CuCtxCreate,
    cuCtxDestroy: CuCtxDestroy,
    cuMemAlloc: CuMemAlloc,
    cuMemFree: CuMemFree,
    cuMemcpyHtoD: CuMemcpyHtoD,
    cuMemcpyDtoH: CuMemcpyDtoH,
    cuMemcpyDtoD: CuMemcpyDtoD,
    cuMemcpyDtoDAsync: CuMemcpyDtoDAsync,
    cuModuleLoadData: CuModuleLoadData,
    cuModuleUnload: CuModuleUnload,
    cuModuleGetFunction: CuModuleGetFunction,
    cuLaunchKernel: CuLaunchKernel,
    cuStreamCreate: CuStreamCreate,
    cuStreamSynchronize: CuStreamSynchronize,
    cuStreamDestroy: CuStreamDestroy,
    cuStreamBeginCapture: CuStreamBeginCapture,
    cuStreamEndCapture: CuStreamEndCapture,
    cuGraphInstantiateWithFlags: CuGraphInstantiateWithFlags,
    cuGraphLaunch: CuGraphLaunch,
    cuGraphDestroy: CuGraphDestroy,
    cuGraphExecDestroy: CuGraphExecDestroy,
    cuImportExternalMemory: CuImportExternalMemory,
    cuExternalMemoryGetMappedBuffer: CuExternalMemoryGetMappedBuffer,
    cuDestroyExternalMemory: CuDestroyExternalMemory,
    context: CUcontext,
    stream: CUstream,
}

impl CudaDriver {
    pub fn new(device_index: u32) -> Result<Self, DriverError> {
        let lib_path = driver_library_path();
        let lib = unsafe { Library::new(&lib_path) }.map_err(|e| {
            DriverError::new(
                -1,
                format!("failed to load `{}`: {e}", lib_path.display()),
            )
        })?;

        unsafe {
            let cuInit = load_fn!(lib, cuInit, CuInit);
            let cuDeviceGet = load_fn!(lib, cuDeviceGet, CuDeviceGet);
            let cuCtxCreate = load_fn!(lib, cuCtxCreate, CuCtxCreate);
            let cuCtxDestroy = load_fn!(lib, cuCtxDestroy, CuCtxDestroy);
            let cuMemAlloc = load_fn!(lib, cuMemAlloc_v2, CuMemAlloc);
            let cuMemFree = load_fn!(lib, cuMemFree_v2, CuMemFree);
            let cuMemcpyHtoD = load_fn!(lib, cuMemcpyHtoD_v2, CuMemcpyHtoD);
            let cuMemcpyDtoH = load_fn!(lib, cuMemcpyDtoH_v2, CuMemcpyDtoH);
            let cuMemcpyDtoD = load_fn!(lib, cuMemcpyDtoD_v2, CuMemcpyDtoD);
            let cuMemcpyDtoDAsync = load_fn!(lib, cuMemcpyDtoDAsync_v2, CuMemcpyDtoDAsync);
            let cuModuleLoadData = load_fn!(lib, cuModuleLoadData, CuModuleLoadData);
            let cuModuleUnload = load_fn!(lib, cuModuleUnload, CuModuleUnload);
            let cuModuleGetFunction = load_fn!(lib, cuModuleGetFunction, CuModuleGetFunction);
            let cuLaunchKernel = load_fn!(lib, cuLaunchKernel, CuLaunchKernel);
            let cuStreamCreate = load_fn!(lib, cuStreamCreate, CuStreamCreate);
            let cuStreamSynchronize = load_fn!(lib, cuStreamSynchronize, CuStreamSynchronize);
            let cuStreamDestroy = load_fn!(lib, cuStreamDestroy, CuStreamDestroy);
            let cuStreamBeginCapture = load_fn!(lib, cuStreamBeginCapture, CuStreamBeginCapture);
            let cuStreamEndCapture = load_fn!(lib, cuStreamEndCapture, CuStreamEndCapture);
            let cuGraphInstantiateWithFlags =
                load_fn!(lib, cuGraphInstantiateWithFlags, CuGraphInstantiateWithFlags);
            let cuGraphLaunch = load_fn!(lib, cuGraphLaunch, CuGraphLaunch);
            let cuGraphDestroy = load_fn!(lib, cuGraphDestroy, CuGraphDestroy);
            let cuGraphExecDestroy = load_fn!(lib, cuGraphExecDestroy, CuGraphExecDestroy);
            let cuImportExternalMemory =
                load_fn!(lib, cuImportExternalMemory, CuImportExternalMemory);
            let cuExternalMemoryGetMappedBuffer = load_fn!(
                lib,
                cuExternalMemoryGetMappedBuffer,
                CuExternalMemoryGetMappedBuffer
            );
            let cuDestroyExternalMemory =
                load_fn!(lib, cuDestroyExternalMemory, CuDestroyExternalMemory);

            check((cuInit)(0), "cuInit")?;

            let mut device: CUdevice = 0;
            check((cuDeviceGet)(&mut device, device_index as i32), "cuDeviceGet")?;

            let mut context: CUcontext = std::ptr::null_mut();
            check((cuCtxCreate)(&mut context, 0, device), "cuCtxCreate")?;

            let mut stream: CUstream = std::ptr::null_mut();
            check((cuStreamCreate)(&mut stream, 0), "cuStreamCreate")?;

            Ok(Self {
                _lib: lib,
                cuInit,
                cuDeviceGet,
                cuCtxCreate,
                cuCtxDestroy,
                cuMemAlloc,
                cuMemFree,
                cuMemcpyHtoD,
                cuMemcpyDtoH,
                cuMemcpyDtoD,
                cuMemcpyDtoDAsync,
                cuModuleLoadData,
                cuModuleUnload,
                cuModuleGetFunction,
                cuLaunchKernel,
                cuStreamCreate,
                cuStreamSynchronize,
                cuStreamDestroy,
                cuStreamBeginCapture,
                cuStreamEndCapture,
                cuGraphInstantiateWithFlags,
                cuGraphLaunch,
                cuGraphDestroy,
                cuGraphExecDestroy,
                cuImportExternalMemory,
                cuExternalMemoryGetMappedBuffer,
                cuDestroyExternalMemory,
                context,
                stream,
            })
        }
    }

    pub fn mem_alloc(&self, bytes: usize) -> Result<CUdeviceptr, DriverError> {
        let mut ptr: CUdeviceptr = 0;
        unsafe {
            check((self.cuMemAlloc)(&mut ptr, bytes), "cuMemAlloc")?;
        }
        Ok(ptr)
    }

    pub fn mem_free(&self, ptr: CUdeviceptr) -> Result<(), DriverError> {
        unsafe {
            check((self.cuMemFree)(ptr), "cuMemFree")?;
        }
        Ok(())
    }

    pub fn memcpy_htod(&self, dst: CUdeviceptr, src: &[u8]) -> Result<(), DriverError> {
        unsafe {
            check(
                (self.cuMemcpyHtoD)(dst, src.as_ptr() as *const c_void, src.len()),
                "cuMemcpyHtoD",
            )?;
        }
        Ok(())
    }

    pub fn memcpy_dtoh(&self, dst: &mut [u8], src: CUdeviceptr) -> Result<(), DriverError> {
        unsafe {
            check(
                (self.cuMemcpyDtoH)(dst.as_mut_ptr() as *mut c_void, src, dst.len()),
                "cuMemcpyDtoH",
            )?;
        }
        Ok(())
    }

    pub fn memcpy_dtod(&self, dst: CUdeviceptr, src: CUdeviceptr, bytes: usize) -> Result<(), DriverError> {
        unsafe {
            check((self.cuMemcpyDtoD)(dst, src, bytes), "cuMemcpyDtoD")?;
        }
        Ok(())
    }

    /// Device-to-device copy on the owned stream (for CUDA Graph capture).
    pub fn memcpy_dtod_async(
        &self,
        dst: CUdeviceptr,
        src: CUdeviceptr,
        bytes: usize,
    ) -> Result<(), DriverError> {
        unsafe {
            check(
                (self.cuMemcpyDtoDAsync)(dst, src, bytes, self.stream),
                "cuMemcpyDtoDAsync",
            )?;
        }
        Ok(())
    }

    /// Launch a 1D grid kernel with `(u32 count, u64 out_ptr)` parameters (fill kernels).
    pub fn launch_fill(
        &self,
        func: CUfunction,
        count: u32,
        out: CUdeviceptr,
    ) -> Result<(), DriverError> {
        self.launch_fill_on_stream(func, count, out, true)
    }

    /// Launch fill without synchronizing (for stream capture into a CUDA Graph).
    pub fn launch_fill_async(
        &self,
        func: CUfunction,
        count: u32,
        out: CUdeviceptr,
    ) -> Result<(), DriverError> {
        self.launch_fill_on_stream(func, count, out, false)
    }

    fn launch_fill_on_stream(
        &self,
        func: CUfunction,
        count: u32,
        out: CUdeviceptr,
        sync: bool,
    ) -> Result<(), DriverError> {
        let block = 256u32;
        let grid = count.saturating_add(block - 1) / block;
        let mut count_arg = count;
        let mut out_arg = out;
        let mut params: [*mut c_void; 2] = [
            (&mut count_arg as *mut u32).cast(),
            (&mut out_arg as *mut CUdeviceptr).cast(),
        ];
        self.launch(func, grid, block, &mut params, sync)
    }

    /// Launch matmul kernel with `(a, b, out, m, n, k)` device/host args.
    pub fn launch_matmul(
        &self,
        func: CUfunction,
        a: CUdeviceptr,
        b: CUdeviceptr,
        out: CUdeviceptr,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), DriverError> {
        self.launch_matmul_on_stream(func, a, b, out, m, n, k, true)
    }

    /// Launch matmul without synchronizing (for stream capture into a CUDA Graph).
    pub fn launch_matmul_async(
        &self,
        func: CUfunction,
        a: CUdeviceptr,
        b: CUdeviceptr,
        out: CUdeviceptr,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), DriverError> {
        self.launch_matmul_on_stream(func, a, b, out, m, n, k, false)
    }

    fn launch_matmul_on_stream(
        &self,
        func: CUfunction,
        a: CUdeviceptr,
        b: CUdeviceptr,
        out: CUdeviceptr,
        m: u32,
        n: u32,
        k: u32,
        sync: bool,
    ) -> Result<(), DriverError> {
        let total = m.saturating_mul(n).max(1);
        let block = 256u32;
        let grid = total.saturating_add(block - 1) / block;
        let mut a_arg = a;
        let mut b_arg = b;
        let mut out_arg = out;
        let mut m_arg = m;
        let mut n_arg = n;
        let mut k_arg = k;
        let mut params: [*mut c_void; 6] = [
            (&mut a_arg as *mut CUdeviceptr).cast(),
            (&mut b_arg as *mut CUdeviceptr).cast(),
            (&mut out_arg as *mut CUdeviceptr).cast(),
            (&mut m_arg as *mut u32).cast(),
            (&mut n_arg as *mut u32).cast(),
            (&mut k_arg as *mut u32).cast(),
        ];
        self.launch(func, grid, block, &mut params, sync)
    }

    /// Launch RGBA8 denoise: `(src, out, width, height)`.
    pub fn launch_rgba8_denoise(
        &self,
        func: CUfunction,
        src: CUdeviceptr,
        out: CUdeviceptr,
        width: u32,
        height: u32,
    ) -> Result<(), DriverError> {
        let total = width.saturating_mul(height).max(1);
        let block = 256u32;
        let grid = total.saturating_add(block - 1) / block;
        let mut src_arg = src;
        let mut out_arg = out;
        let mut width_arg = width;
        let mut height_arg = height;
        let mut params: [*mut c_void; 4] = [
            (&mut src_arg as *mut CUdeviceptr).cast(),
            (&mut out_arg as *mut CUdeviceptr).cast(),
            (&mut width_arg as *mut u32).cast(),
            (&mut height_arg as *mut u32).cast(),
        ];
        self.launch(func, grid, block, &mut params, true)
    }

    /// Launch attention: `(q, k, v, out, batch, heads, seq, dim)`.
    pub fn launch_attention(
        &self,
        func: CUfunction,
        q: CUdeviceptr,
        k: CUdeviceptr,
        v: CUdeviceptr,
        out: CUdeviceptr,
        batch: u32,
        heads: u32,
        seq: u32,
        dim: u32,
    ) -> Result<(), DriverError> {
        let total = batch
            .saturating_mul(heads)
            .saturating_mul(seq)
            .saturating_mul(dim)
            .max(1);
        let block = 256u32;
        let grid = total.saturating_add(block - 1) / block;
        let mut q_arg = q;
        let mut k_arg = k;
        let mut v_arg = v;
        let mut out_arg = out;
        let mut batch_arg = batch;
        let mut heads_arg = heads;
        let mut seq_arg = seq;
        let mut dim_arg = dim;
        let mut params: [*mut c_void; 8] = [
            (&mut q_arg as *mut CUdeviceptr).cast(),
            (&mut k_arg as *mut CUdeviceptr).cast(),
            (&mut v_arg as *mut CUdeviceptr).cast(),
            (&mut out_arg as *mut CUdeviceptr).cast(),
            (&mut batch_arg as *mut u32).cast(),
            (&mut heads_arg as *mut u32).cast(),
            (&mut seq_arg as *mut u32).cast(),
            (&mut dim_arg as *mut u32).cast(),
        ];
        self.launch(func, grid, block, &mut params, true)
    }

    fn launch(
        &self,
        func: CUfunction,
        grid: u32,
        block: u32,
        params: &mut [*mut c_void],
        sync: bool,
    ) -> Result<(), DriverError> {
        unsafe {
            check(
                (self.cuLaunchKernel)(
                    func,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    self.stream,
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "cuLaunchKernel",
            )?;
            if sync {
                check((self.cuStreamSynchronize)(self.stream), "cuStreamSynchronize")?;
            }
        }
        Ok(())
    }

    /// Begin recording work on the default stream into a CUDA Graph.
    pub fn stream_begin_capture(&self) -> Result<(), DriverError> {
        unsafe {
            check(
                (self.cuStreamBeginCapture)(self.stream, CU_STREAM_CAPTURE_MODE_GLOBAL),
                "cuStreamBeginCapture",
            )
        }
    }

    /// End stream capture and return the recorded graph.
    pub fn stream_end_capture(&self) -> Result<CUgraph, DriverError> {
        let mut graph: CUgraph = std::ptr::null_mut();
        unsafe {
            check(
                (self.cuStreamEndCapture)(self.stream, &mut graph),
                "cuStreamEndCapture",
            )?;
        }
        Ok(graph)
    }

    /// Instantiate a captured graph for launch.
    pub fn graph_instantiate(&self, graph: CUgraph) -> Result<CUgraphExec, DriverError> {
        let mut exec: CUgraphExec = std::ptr::null_mut();
        unsafe {
            check(
                (self.cuGraphInstantiateWithFlags)(&mut exec, graph, 0),
                "cuGraphInstantiateWithFlags",
            )?;
        }
        Ok(exec)
    }

    /// Launch an instantiated graph on the default stream and synchronize.
    pub fn graph_launch(&self, exec: CUgraphExec) -> Result<(), DriverError> {
        unsafe {
            check(
                (self.cuGraphLaunch)(exec, self.stream),
                "cuGraphLaunch",
            )?;
            check((self.cuStreamSynchronize)(self.stream), "cuStreamSynchronize")?;
        }
        Ok(())
    }

    pub fn graph_destroy(&self, graph: CUgraph) -> Result<(), DriverError> {
        unsafe { check((self.cuGraphDestroy)(graph), "cuGraphDestroy") }
    }

    pub fn graph_exec_destroy(&self, exec: CUgraphExec) -> Result<(), DriverError> {
        unsafe { check((self.cuGraphExecDestroy)(exec), "cuGraphExecDestroy") }
    }

    pub fn load_module(&self, ptx: &[u8]) -> Result<CUmodule, DriverError> {
        let mut module: CUmodule = std::ptr::null_mut();
        unsafe {
            check(
                (self.cuModuleLoadData)(&mut module, ptx.as_ptr() as *const c_void),
                "cuModuleLoadData",
            )?;
        }
        Ok(module)
    }

    pub fn unload_module(&self, module: CUmodule) -> Result<(), DriverError> {
        unsafe {
            check((self.cuModuleUnload)(module), "cuModuleUnload")?;
        }
        Ok(())
    }

    pub fn get_function(&self, module: CUmodule, entry: &str) -> Result<CUfunction, DriverError> {
        let name = CString::new(entry).map_err(|e| DriverError::new(-1, e.to_string()))?;
        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            check(
                (self.cuModuleGetFunction)(&mut func, module, name.as_ptr()),
                "cuModuleGetFunction",
            )?;
        }
        Ok(func)
    }

    /// Import a D3D12 shared NT handle and map it as a device buffer.
    pub fn import_d3d12_resource(
        &self,
        nt_handle: *mut c_void,
        bytes: usize,
    ) -> Result<(CUexternalMemory, CUdeviceptr), DriverError> {
        unsafe {
            let desc = CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
                type_: CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE,
                handle: CUDA_EXTERNAL_MEMORY_HANDLE {
                    win32: CUDA_EXTERNAL_MEMORY_WIN32 {
                        handle: nt_handle,
                        name: std::ptr::null(),
                    },
                },
                size: bytes as u64,
                flags: CUDA_EXTERNAL_MEMORY_DEDICATED,
            };
            let mut ext: CUexternalMemory = std::ptr::null_mut();
            check(
                (self.cuImportExternalMemory)(&mut ext, &desc),
                "cuImportExternalMemory",
            )?;
            let buf_desc = CUDA_EXTERNAL_MEMORY_BUFFER_DESC {
                offset: 0,
                size: bytes as u64,
                flags: 0,
            };
            let mut ptr: CUdeviceptr = 0;
            if let Err(err) = check(
                (self.cuExternalMemoryGetMappedBuffer)(&mut ptr, ext, &buf_desc),
                "cuExternalMemoryGetMappedBuffer",
            ) {
                let _ = (self.cuDestroyExternalMemory)(ext);
                return Err(err);
            }
            Ok((ext, ptr))
        }
    }

    pub fn destroy_external_memory(&self, ext: CUexternalMemory) -> Result<(), DriverError> {
        unsafe {
            check(
                (self.cuDestroyExternalMemory)(ext),
                "cuDestroyExternalMemory",
            )
        }
    }

    pub fn synchronize(&self) -> Result<(), DriverError> {
        unsafe {
            check((self.cuStreamSynchronize)(self.stream), "cuStreamSynchronize")?;
        }
        Ok(())
    }
}

// CUDA driver handles are owned by this struct and used from the scheduler thread.
unsafe impl Send for CudaDriver {}
unsafe impl Sync for CudaDriver {}

impl Drop for CudaDriver {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.cuStreamDestroy)(self.stream);
            let _ = (self.cuCtxDestroy)(self.context);
        }
    }
}

fn check(code: CUresult, api: &str) -> Result<(), DriverError> {
    if code == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(DriverError::new(code, api))
    }
}

fn driver_library_path() -> std::path::PathBuf {
    if cfg!(target_os = "windows") {
        Path::new("nvcuda.dll").to_path_buf()
    } else if cfg!(target_os = "macos") {
        Path::new("/usr/local/cuda/lib/libcuda.dylib").to_path_buf()
    } else {
        Path::new("libcuda.so.1").to_path_buf()
    }
}
