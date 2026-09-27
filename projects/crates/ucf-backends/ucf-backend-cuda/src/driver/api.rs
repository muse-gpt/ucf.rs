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
    cuModuleLoadData: CuModuleLoadData,
    cuModuleUnload: CuModuleUnload,
    cuModuleGetFunction: CuModuleGetFunction,
    cuLaunchKernel: CuLaunchKernel,
    cuStreamCreate: CuStreamCreate,
    cuStreamSynchronize: CuStreamSynchronize,
    cuStreamDestroy: CuStreamDestroy,
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
            let cuModuleLoadData = load_fn!(lib, cuModuleLoadData, CuModuleLoadData);
            let cuModuleUnload = load_fn!(lib, cuModuleUnload, CuModuleUnload);
            let cuModuleGetFunction = load_fn!(lib, cuModuleGetFunction, CuModuleGetFunction);
            let cuLaunchKernel = load_fn!(lib, cuLaunchKernel, CuLaunchKernel);
            let cuStreamCreate = load_fn!(lib, cuStreamCreate, CuStreamCreate);
            let cuStreamSynchronize = load_fn!(lib, cuStreamSynchronize, CuStreamSynchronize);
            let cuStreamDestroy = load_fn!(lib, cuStreamDestroy, CuStreamDestroy);

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
                cuModuleLoadData,
                cuModuleUnload,
                cuModuleGetFunction,
                cuLaunchKernel,
                cuStreamCreate,
                cuStreamSynchronize,
                cuStreamDestroy,
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

    pub fn launch_iota_fill(
        &self,
        func: CUfunction,
        count: u32,
        out: CUdeviceptr,
    ) -> Result<(), DriverError> {
        let block = 256u32;
        let grid = (count + block - 1) / block;
        let mut count_arg = count;
        let mut out_arg = out;
        let mut params: [*mut c_void; 2] = [
            (&mut count_arg as *mut u32).cast(),
            (&mut out_arg as *mut CUdeviceptr).cast(),
        ];
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
            check((self.cuStreamSynchronize)(self.stream), "cuStreamSynchronize")?;
        }
        Ok(())
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
