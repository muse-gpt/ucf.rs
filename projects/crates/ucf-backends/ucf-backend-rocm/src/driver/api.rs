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

/// HIP Runtime + `hiprtc` handles owned by the ROCm backend.
pub struct HipDriver {
    _runtime: Library,
    _rtc: Library,
    hipMalloc: HipMalloc,
    hipFree: HipFree,
    hipMemcpy: HipMemcpy,
    hipDeviceSynchronize: HipDeviceSynchronize,
    hipModuleLoadData: HipModuleLoadData,
    hipModuleUnload: HipModuleUnload,
    hipModuleGetFunction: HipModuleGetFunction,
    hipModuleLaunchKernel: HipModuleLaunchKernel,
    hiprtcCreateProgram: HiprtcCreateProgram,
    hiprtcCompileProgram: HiprtcCompileProgram,
    hiprtcGetCodeSize: HiprtcGetCodeSize,
    hiprtcGetCode: HiprtcGetCode,
    hiprtcDestroyProgram: HiprtcDestroyProgram,
    hiprtcGetProgramLogSize: HiprtcGetProgramLogSize,
    hiprtcGetProgramLog: HiprtcGetProgramLog,
}

impl HipDriver {
    /// Load HIP runtime / rtc libraries and select `device_index`.
    pub fn new(device_index: u32) -> Result<Self, DriverError> {
        let runtime = load_first(&runtime_library_candidates())?;
        let rtc = load_first(&rtc_library_candidates())?;

        unsafe {
            let hipGetDeviceCount = load_fn!(runtime, hipGetDeviceCount, HipGetDeviceCount);
            let hipSetDevice = load_fn!(runtime, hipSetDevice, HipSetDevice);
            let hipMalloc = load_fn!(runtime, hipMalloc, HipMalloc);
            let hipFree = load_fn!(runtime, hipFree, HipFree);
            let hipMemcpy = load_fn!(runtime, hipMemcpy, HipMemcpy);
            let hipDeviceSynchronize =
                load_fn!(runtime, hipDeviceSynchronize, HipDeviceSynchronize);
            let hipModuleLoadData = load_fn!(runtime, hipModuleLoadData, HipModuleLoadData);
            let hipModuleUnload = load_fn!(runtime, hipModuleUnload, HipModuleUnload);
            let hipModuleGetFunction =
                load_fn!(runtime, hipModuleGetFunction, HipModuleGetFunction);
            let hipModuleLaunchKernel =
                load_fn!(runtime, hipModuleLaunchKernel, HipModuleLaunchKernel);

            let hiprtcCreateProgram = load_fn!(rtc, hiprtcCreateProgram, HiprtcCreateProgram);
            let hiprtcCompileProgram = load_fn!(rtc, hiprtcCompileProgram, HiprtcCompileProgram);
            let hiprtcGetCodeSize = load_fn!(rtc, hiprtcGetCodeSize, HiprtcGetCodeSize);
            let hiprtcGetCode = load_fn!(rtc, hiprtcGetCode, HiprtcGetCode);
            let hiprtcDestroyProgram = load_fn!(rtc, hiprtcDestroyProgram, HiprtcDestroyProgram);
            let hiprtcGetProgramLogSize =
                load_fn!(rtc, hiprtcGetProgramLogSize, HiprtcGetProgramLogSize);
            let hiprtcGetProgramLog = load_fn!(rtc, hiprtcGetProgramLog, HiprtcGetProgramLog);

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

            Ok(Self {
                _runtime: runtime,
                _rtc: rtc,
                hipMalloc,
                hipFree,
                hipMemcpy,
                hipDeviceSynchronize,
                hipModuleLoadData,
                hipModuleUnload,
                hipModuleGetFunction,
                hipModuleLaunchKernel,
                hiprtcCreateProgram,
                hiprtcCompileProgram,
                hiprtcGetCodeSize,
                hiprtcGetCode,
                hiprtcDestroyProgram,
                hiprtcGetProgramLogSize,
                hiprtcGetProgramLog,
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

    pub fn memcpy_htod(&self, dst: HipDeviceptr, src: &[u8]) -> Result<(), DriverError> {
        unsafe {
            check_hip(
                (self.hipMemcpy)(
                    dst,
                    src.as_ptr().cast(),
                    src.len(),
                    HIP_MEMCPY_HOST_TO_DEVICE,
                ),
                "hipMemcpy HtoD",
            )?;
        }
        Ok(())
    }

    pub fn memcpy_dtoh(&self, dst: &mut [u8], src: HipDeviceptr) -> Result<(), DriverError> {
        unsafe {
            check_hip(
                (self.hipMemcpy)(
                    dst.as_mut_ptr().cast(),
                    src,
                    dst.len(),
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

    /// Compile HIP source with `hiprtc` and load as a module.
    pub fn compile_module(&self, hip_source: &[u8]) -> Result<HipModule, DriverError> {
        unsafe {
            let mut prog: HiprtcProgram = std::ptr::null_mut();
            let name = CString::new("ucf.hip").unwrap();
            check_rtc(
                (self.hiprtcCreateProgram)(
                    &mut prog,
                    hip_source.as_ptr().cast(),
                    name.as_ptr(),
                    0,
                    std::ptr::null(),
                    std::ptr::null(),
                ),
                "hiprtcCreateProgram",
            )?;
            let compile = (self.hiprtcCompileProgram)(prog, 0, std::ptr::null());
            if compile != HIPRTC_SUCCESS {
                let log = self.rtc_log(prog).unwrap_or_default();
                let _ = (self.hiprtcDestroyProgram)(&mut prog);
                return Err(DriverError::new(
                    compile,
                    format!("hiprtcCompileProgram failed: {log}"),
                ));
            }
            let mut size = 0usize;
            check_rtc(
                (self.hiprtcGetCodeSize)(prog, &mut size),
                "hiprtcGetCodeSize",
            )?;
            let mut code = vec![0i8; size];
            check_rtc(
                (self.hiprtcGetCode)(prog, code.as_mut_ptr()),
                "hiprtcGetCode",
            )?;
            let _ = (self.hiprtcDestroyProgram)(&mut prog);

            let mut module: HipModule = std::ptr::null_mut();
            check_hip(
                (self.hipModuleLoadData)(&mut module, code.as_ptr().cast()),
                "hipModuleLoadData",
            )?;
            Ok(module)
        }
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
        let block = 256u32;
        let grid = count.div_ceil(block).max(1);
        let mut out_arg = out;
        let mut count_arg = count;
        let mut params: [*mut c_void; 2] = [
            (&mut out_arg as *mut HipDeviceptr).cast(),
            (&mut count_arg as *mut u32).cast(),
        ];
        self.launch(func, grid, block, &mut params)
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
        self.launch(func, grid, block, &mut params)
    }

    fn launch(
        &self,
        func: HipFunction,
        grid: u32,
        block: u32,
        params: &mut [*mut c_void],
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
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "hipModuleLaunchKernel",
            )?;
            check_hip((self.hipDeviceSynchronize)(), "hipDeviceSynchronize")?;
        }
        Ok(())
    }

    unsafe fn rtc_log(&self, prog: HiprtcProgram) -> Result<String, DriverError> {
        let mut size = 0usize;
        check_rtc(
            (self.hiprtcGetProgramLogSize)(prog, &mut size),
            "hiprtcGetProgramLogSize",
        )?;
        if size == 0 {
            return Ok(String::new());
        }
        let mut buf = vec![0i8; size];
        check_rtc(
            (self.hiprtcGetProgramLog)(prog, buf.as_mut_ptr()),
            "hiprtcGetProgramLog",
        )?;
        Ok(String::from_utf8_lossy(std::slice::from_raw_parts(
            buf.as_ptr().cast(),
            size.saturating_sub(1),
        ))
        .into_owned())
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

fn check_rtc(code: HiprtcResult, api: &str) -> Result<(), DriverError> {
    if code == HIPRTC_SUCCESS {
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

fn rtc_library_candidates() -> Vec<std::path::PathBuf> {
    if cfg!(target_os = "windows") {
        vec![
            Path::new("hiprtc.dll").to_path_buf(),
            Path::new("hiprtc0605.dll").to_path_buf(),
            Path::new("hiprtc0604.dll").to_path_buf(),
            Path::new("hiprtc0600.dll").to_path_buf(),
            Path::new("amdhip64.dll").to_path_buf(),
        ]
    } else {
        vec![
            Path::new("libhiprtc.so").to_path_buf(),
            Path::new("libhiprtc.so.6").to_path_buf(),
            Path::new("libamdhip64.so").to_path_buf(),
        ]
    }
}
