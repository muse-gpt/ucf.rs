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

/// HIP runtime handles owned by the ROCm backend (no `hiprtc`).
pub struct HipDriver {
    _runtime: Library,
    stream: HipStream,
    hipMalloc: HipMalloc,
    hipFree: HipFree,
    hipMemcpy: HipMemcpy,
    hipMemcpyAsync: HipMemcpyAsync,
    hipModuleLoadData: HipModuleLoadData,
    hipModuleUnload: HipModuleUnload,
    hipModuleGetFunction: HipModuleGetFunction,
    hipModuleLaunchKernel: HipModuleLaunchKernel,
    hipStreamSynchronize: HipStreamSynchronize,
    hipStreamDestroy: HipStreamDestroy,
    hipStreamBeginCapture: HipStreamBeginCapture,
    hipStreamEndCapture: HipStreamEndCapture,
    hipGraphInstantiateWithFlags: HipGraphInstantiateWithFlags,
    hipGraphLaunch: HipGraphLaunch,
    hipGraphDestroy: HipGraphDestroy,
    hipGraphExecDestroy: HipGraphExecDestroy,
}

impl HipDriver {
    /// Load HIP runtime and select `device_index`.
    pub fn new(device_index: u32) -> Result<Self, DriverError> {
        let runtime = load_first(&runtime_library_candidates())?;

        unsafe {
            let hipGetDeviceCount = load_fn!(runtime, hipGetDeviceCount, HipGetDeviceCount);
            let hipSetDevice = load_fn!(runtime, hipSetDevice, HipSetDevice);
            let hipMalloc = load_fn!(runtime, hipMalloc, HipMalloc);
            let hipFree = load_fn!(runtime, hipFree, HipFree);
            let hipMemcpy = load_fn!(runtime, hipMemcpy, HipMemcpy);
            let hipMemcpyAsync = load_fn!(runtime, hipMemcpyAsync, HipMemcpyAsync);
            let hipModuleLoadData = load_fn!(runtime, hipModuleLoadData, HipModuleLoadData);
            let hipModuleUnload = load_fn!(runtime, hipModuleUnload, HipModuleUnload);
            let hipModuleGetFunction =
                load_fn!(runtime, hipModuleGetFunction, HipModuleGetFunction);
            let hipModuleLaunchKernel =
                load_fn!(runtime, hipModuleLaunchKernel, HipModuleLaunchKernel);
            let hipStreamCreate = load_fn!(runtime, hipStreamCreate, HipStreamCreate);
            let hipStreamSynchronize =
                load_fn!(runtime, hipStreamSynchronize, HipStreamSynchronize);
            let hipStreamDestroy = load_fn!(runtime, hipStreamDestroy, HipStreamDestroy);
            let hipStreamBeginCapture =
                load_fn!(runtime, hipStreamBeginCapture, HipStreamBeginCapture);
            let hipStreamEndCapture = load_fn!(runtime, hipStreamEndCapture, HipStreamEndCapture);
            let hipGraphInstantiateWithFlags = load_fn!(
                runtime,
                hipGraphInstantiateWithFlags,
                HipGraphInstantiateWithFlags
            );
            let hipGraphLaunch = load_fn!(runtime, hipGraphLaunch, HipGraphLaunch);
            let hipGraphDestroy = load_fn!(runtime, hipGraphDestroy, HipGraphDestroy);
            let hipGraphExecDestroy = load_fn!(runtime, hipGraphExecDestroy, HipGraphExecDestroy);

            let mut count = 0i32;
            check_hip((hipGetDeviceCount)(&mut count), "hipGetDeviceCount")?;
            if count <= 0 {
                return Err(DriverError::new(-1, "no HIP devices"));
            }
            if device_index as i32 >= count {
                return Err(DriverError::new(
                    -1,
                    format!("HIP device {device_index} out of range (count={count})"),
                ));
            }
            check_hip((hipSetDevice)(device_index as i32), "hipSetDevice")?;

            let mut stream: HipStream = std::ptr::null_mut();
            check_hip((hipStreamCreate)(&mut stream), "hipStreamCreate")?;

            Ok(Self {
                _runtime: runtime,
                stream,
                hipMalloc,
                hipFree,
                hipMemcpy,
                hipMemcpyAsync,
                hipModuleLoadData,
                hipModuleUnload,
                hipModuleGetFunction,
                hipModuleLaunchKernel,
                hipStreamSynchronize,
                hipStreamDestroy,
                hipStreamBeginCapture,
                hipStreamEndCapture,
                hipGraphInstantiateWithFlags,
                hipGraphLaunch,
                hipGraphDestroy,
                hipGraphExecDestroy,
            })
        }
    }

    pub fn mem_alloc(&self, bytes: usize) -> Result<HipDeviceptr, DriverError> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        unsafe {
            check_hip((self.hipMalloc)(&mut ptr, bytes.max(1)), "hipMalloc")?;
        }
        Ok(ptr)
    }

    pub fn mem_free(&self, ptr: HipDeviceptr) -> Result<(), DriverError> {
        unsafe {
            check_hip((self.hipFree)(ptr), "hipFree")?;
        }
        Ok(())
    }

    pub fn memcpy_htod(&self, dst: HipDeviceptr, host: &[u8]) -> Result<(), DriverError> {
        unsafe {
            check_hip(
                (self.hipMemcpy)(
                    dst,
                    host.as_ptr().cast(),
                    host.len(),
                    HIP_MEMCPY_HOST_TO_DEVICE,
                ),
                "hipMemcpy HtoD",
            )?;
        }
        Ok(())
    }

    pub fn memcpy_dtoh(&self, host: &mut [u8], src: HipDeviceptr) -> Result<(), DriverError> {
        unsafe {
            check_hip(
                (self.hipMemcpy)(
                    host.as_mut_ptr().cast(),
                    src,
                    host.len(),
                    HIP_MEMCPY_DEVICE_TO_HOST,
                ),
                "hipMemcpy DtoH",
            )?;
        }
        Ok(())
    }

    pub fn memcpy_dtod(
        &self,
        dst: HipDeviceptr,
        src: HipDeviceptr,
        bytes: usize,
    ) -> Result<(), DriverError> {
        unsafe {
            check_hip(
                (self.hipMemcpy)(dst, src, bytes, HIP_MEMCPY_DEVICE_TO_DEVICE),
                "hipMemcpy DtoD",
            )?;
        }
        Ok(())
    }

    /// Device-to-device copy on the owned stream (for HIP Graph capture).
    pub fn memcpy_dtod_async(
        &self,
        dst: HipDeviceptr,
        src: HipDeviceptr,
        bytes: usize,
    ) -> Result<(), DriverError> {
        unsafe {
            check_hip(
                (self.hipMemcpyAsync)(
                    dst,
                    src,
                    bytes,
                    HIP_MEMCPY_DEVICE_TO_DEVICE,
                    self.stream,
                ),
                "hipMemcpyAsync DtoD",
            )?;
        }
        Ok(())
    }

    /// Load an emitter-produced HSACO / ELF code object.
    pub fn load_module(&self, hsaco: &[u8]) -> Result<HipModule, DriverError> {
        let mut module: HipModule = std::ptr::null_mut();
        unsafe {
            check_hip(
                (self.hipModuleLoadData)(&mut module, hsaco.as_ptr().cast()),
                "hipModuleLoadData",
            )?;
        }
        Ok(module)
    }

    pub fn unload_module(&self, module: HipModule) -> Result<(), DriverError> {
        unsafe {
            check_hip((self.hipModuleUnload)(module), "hipModuleUnload")?;
        }
        Ok(())
    }

    pub fn get_function(&self, module: HipModule, entry: &str) -> Result<HipFunction, DriverError> {
        let name = CString::new(entry).map_err(|e| DriverError::new(-1, e.to_string()))?;
        let mut func: HipFunction = std::ptr::null_mut();
        unsafe {
            check_hip(
                (self.hipModuleGetFunction)(&mut func, module, name.as_ptr()),
                "hipModuleGetFunction",
            )?;
        }
        Ok(func)
    }

    pub fn launch_fill(
        &self,
        func: HipFunction,
        out: HipDeviceptr,
        count: u32,
    ) -> Result<(), DriverError> {
        self.launch_fill_on_stream(func, out, count, true)
    }

    /// Launch fill without synchronizing (for stream capture into a HIP Graph).
    pub fn launch_fill_async(
        &self,
        func: HipFunction,
        out: HipDeviceptr,
        count: u32,
    ) -> Result<(), DriverError> {
        self.launch_fill_on_stream(func, out, count, false)
    }

    fn launch_fill_on_stream(
        &self,
        func: HipFunction,
        out: HipDeviceptr,
        count: u32,
        sync: bool,
    ) -> Result<(), DriverError> {
        let block = 256u32;
        let grid = count.div_ceil(block).max(1);
        let mut out_arg = out;
        let mut count_arg = count;
        let mut params: [*mut c_void; 2] = [
            (&mut out_arg as *mut HipDeviceptr).cast(),
            (&mut count_arg as *mut u32).cast(),
        ];
        self.launch(func, grid, block, &mut params, sync)
    }

    pub fn launch_matmul(
        &self,
        func: HipFunction,
        a: HipDeviceptr,
        b: HipDeviceptr,
        out: HipDeviceptr,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), DriverError> {
        self.launch_matmul_on_stream(func, a, b, out, m, n, k, true)
    }

    /// Launch matmul without synchronizing (for stream capture into a HIP Graph).
    pub fn launch_matmul_async(
        &self,
        func: HipFunction,
        a: HipDeviceptr,
        b: HipDeviceptr,
        out: HipDeviceptr,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), DriverError> {
        self.launch_matmul_on_stream(func, a, b, out, m, n, k, false)
    }

    fn launch_matmul_on_stream(
        &self,
        func: HipFunction,
        a: HipDeviceptr,
        b: HipDeviceptr,
        out: HipDeviceptr,
        m: u32,
        n: u32,
        k: u32,
        sync: bool,
    ) -> Result<(), DriverError> {
        let total = m.saturating_mul(n).max(1);
        let block = 256u32;
        let grid = total.div_ceil(block).max(1);
        let mut a_arg = a;
        let mut b_arg = b;
        let mut out_arg = out;
        let mut m_arg = m;
        let mut n_arg = n;
        let mut k_arg = k;
        let mut params: [*mut c_void; 6] = [
            (&mut a_arg as *mut HipDeviceptr).cast(),
            (&mut b_arg as *mut HipDeviceptr).cast(),
            (&mut out_arg as *mut HipDeviceptr).cast(),
            (&mut m_arg as *mut u32).cast(),
            (&mut n_arg as *mut u32).cast(),
            (&mut k_arg as *mut u32).cast(),
        ];
        self.launch(func, grid, block, &mut params, sync)
    }

    /// Launch attention: `(q, k, v, out, batch, heads, seq, dim)`.
    pub fn launch_attention(
        &self,
        func: HipFunction,
        q: HipDeviceptr,
        k: HipDeviceptr,
        v: HipDeviceptr,
        out: HipDeviceptr,
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
        let grid = total.div_ceil(block).max(1);
        let mut q_arg = q;
        let mut k_arg = k;
        let mut v_arg = v;
        let mut out_arg = out;
        let mut batch_arg = batch;
        let mut heads_arg = heads;
        let mut seq_arg = seq;
        let mut dim_arg = dim;
        let mut params: [*mut c_void; 8] = [
            (&mut q_arg as *mut HipDeviceptr).cast(),
            (&mut k_arg as *mut HipDeviceptr).cast(),
            (&mut v_arg as *mut HipDeviceptr).cast(),
            (&mut out_arg as *mut HipDeviceptr).cast(),
            (&mut batch_arg as *mut u32).cast(),
            (&mut heads_arg as *mut u32).cast(),
            (&mut seq_arg as *mut u32).cast(),
            (&mut dim_arg as *mut u32).cast(),
        ];
        self.launch(func, grid, block, &mut params, true)
    }

    fn launch(
        &self,
        func: HipFunction,
        grid: u32,
        block: u32,
        params: &mut [*mut c_void],
        sync: bool,
    ) -> Result<(), DriverError> {
        unsafe {
            check_hip(
                (self.hipModuleLaunchKernel)(
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
                "hipModuleLaunchKernel",
            )?;
            if sync {
                check_hip(
                    (self.hipStreamSynchronize)(self.stream),
                    "hipStreamSynchronize",
                )?;
            }
        }
        Ok(())
    }

    /// Begin recording work on the owned stream into a HIP Graph.
    pub fn stream_begin_capture(&self) -> Result<(), DriverError> {
        unsafe {
            check_hip(
                (self.hipStreamBeginCapture)(self.stream, HIP_STREAM_CAPTURE_MODE_GLOBAL),
                "hipStreamBeginCapture",
            )
        }
    }

    /// End stream capture and return the recorded graph.
    pub fn stream_end_capture(&self) -> Result<HipGraph, DriverError> {
        let mut graph: HipGraph = std::ptr::null_mut();
        unsafe {
            check_hip(
                (self.hipStreamEndCapture)(self.stream, &mut graph),
                "hipStreamEndCapture",
            )?;
        }
        Ok(graph)
    }

    /// Instantiate a captured graph for launch.
    pub fn graph_instantiate(&self, graph: HipGraph) -> Result<HipGraphExec, DriverError> {
        let mut exec: HipGraphExec = std::ptr::null_mut();
        unsafe {
            check_hip(
                (self.hipGraphInstantiateWithFlags)(&mut exec, graph, 0),
                "hipGraphInstantiateWithFlags",
            )?;
        }
        Ok(exec)
    }

    /// Launch an instantiated graph on the owned stream and synchronize.
    pub fn graph_launch(&self, exec: HipGraphExec) -> Result<(), DriverError> {
        unsafe {
            check_hip(
                (self.hipGraphLaunch)(exec, self.stream),
                "hipGraphLaunch",
            )?;
            check_hip(
                (self.hipStreamSynchronize)(self.stream),
                "hipStreamSynchronize",
            )?;
        }
        Ok(())
    }

    pub fn graph_destroy(&self, graph: HipGraph) -> Result<(), DriverError> {
        unsafe { check_hip((self.hipGraphDestroy)(graph), "hipGraphDestroy") }
    }

    pub fn graph_exec_destroy(&self, exec: HipGraphExec) -> Result<(), DriverError> {
        unsafe { check_hip((self.hipGraphExecDestroy)(exec), "hipGraphExecDestroy") }
    }
}

impl Drop for HipDriver {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.hipStreamDestroy)(self.stream);
        }
    }
}

unsafe impl Send for HipDriver {}
unsafe impl Sync for HipDriver {}

fn check_hip(code: HipError, api: &str) -> Result<(), DriverError> {
    if code == HIP_SUCCESS {
        Ok(())
    } else {
        Err(DriverError::new(code, api))
    }
}

fn load_first(candidates: &[std::path::PathBuf]) -> Result<Library, DriverError> {
    let mut last = String::new();
    for path in candidates {
        match unsafe { Library::new(path) } {
            Ok(lib) => return Ok(lib),
            Err(err) => {
                last = format!("{}: {err}", path.display());
            }
        }
    }
    Err(DriverError::new(-1, format!("failed to load HIP library ({last})")))
}

fn runtime_library_candidates() -> Vec<std::path::PathBuf> {
    if cfg!(target_os = "windows") {
        vec![
            Path::new("amdhip64.dll").to_path_buf(),
            Path::new("amdhip64_6.dll").to_path_buf(),
        ]
    } else {
        vec![
            Path::new("libamdhip64.so").to_path_buf(),
            Path::new("libamdhip64.so.6").to_path_buf(),
        ]
    }
}
