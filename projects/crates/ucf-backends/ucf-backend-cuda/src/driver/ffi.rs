use std::ffi::c_void;

pub type CUresult = i32;
pub type CUdevice = i32;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUdeviceptr = u64;
pub type CUstream = *mut c_void;
pub type CUexternalMemory = *mut c_void;
pub type CUgraph = *mut c_void;
pub type CUgraphExec = *mut c_void;

pub const CUDA_SUCCESS: CUresult = 0;
pub const CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE: u32 = 5;
pub const CUDA_EXTERNAL_MEMORY_DEDICATED: u32 = 0x1;
/// `CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL`
pub const CU_STREAM_CAPTURE_MODE_GLOBAL: u32 = 0;

#[repr(C)]
pub struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
    pub type_: u32,
    pub handle: CUDA_EXTERNAL_MEMORY_HANDLE,
    pub size: u64,
    pub flags: u32,
}

#[repr(C)]
pub union CUDA_EXTERNAL_MEMORY_HANDLE {
    pub fd: i32,
    pub win32: CUDA_EXTERNAL_MEMORY_WIN32,
    pub nv_sci_buf_object: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUDA_EXTERNAL_MEMORY_WIN32 {
    pub handle: *mut c_void,
    pub name: *const c_void,
}

#[repr(C)]
pub struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC {
    pub offset: u64,
    pub size: u64,
    pub flags: u32,
}

#[repr(C)]
pub struct CUuuid {
    pub bytes: [u8; 16],
}

#[derive(Debug)]
pub struct DriverError {
    pub code: CUresult,
    pub message: String,
}

impl DriverError {
    pub fn new(code: CUresult, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for DriverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CUDA driver error {}: {}", self.code, self.message)
    }
}

impl std::error::Error for DriverError {}

pub type CuInit = unsafe extern "C" fn(u32) -> CUresult;
pub type CuDeviceGet = unsafe extern "C" fn(*mut CUdevice, i32) -> CUresult;
pub type CuCtxCreate = unsafe extern "C" fn(*mut CUcontext, u32, CUdevice) -> CUresult;
pub type CuCtxDestroy = unsafe extern "C" fn(CUcontext) -> CUresult;
pub type CuMemAlloc = unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult;
pub type CuMemFree = unsafe extern "C" fn(CUdeviceptr) -> CUresult;
pub type CuMemcpyHtoD = unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult;
pub type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult;
pub type CuMemcpyDtoD = unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize) -> CUresult;
pub type CuModuleLoadData = unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult;
pub type CuModuleUnload = unsafe extern "C" fn(CUmodule) -> CUresult;
pub type CuModuleGetFunction =
    unsafe extern "C" fn(*mut CUfunction, CUmodule, *const i8) -> CUresult;
pub type CuLaunchKernel = unsafe extern "C" fn(
    CUfunction,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    CUstream,
    *mut *mut c_void,
    *mut c_void,
) -> CUresult;
pub type CuStreamCreate = unsafe extern "C" fn(*mut CUstream, u32) -> CUresult;
pub type CuStreamSynchronize = unsafe extern "C" fn(CUstream) -> CUresult;
pub type CuStreamDestroy = unsafe extern "C" fn(CUstream) -> CUresult;
pub type CuStreamBeginCapture =
    unsafe extern "C" fn(CUstream, u32) -> CUresult;
pub type CuStreamEndCapture = unsafe extern "C" fn(CUstream, *mut CUgraph) -> CUresult;
pub type CuGraphInstantiateWithFlags =
    unsafe extern "C" fn(*mut CUgraphExec, CUgraph, u64) -> CUresult;
pub type CuGraphLaunch = unsafe extern "C" fn(CUgraphExec, CUstream) -> CUresult;
pub type CuGraphDestroy = unsafe extern "C" fn(CUgraph) -> CUresult;
pub type CuGraphExecDestroy = unsafe extern "C" fn(CUgraphExec) -> CUresult;
pub type CuImportExternalMemory = unsafe extern "C" fn(
    *mut CUexternalMemory,
    *const CUDA_EXTERNAL_MEMORY_HANDLE_DESC,
) -> CUresult;
pub type CuExternalMemoryGetMappedBuffer = unsafe extern "C" fn(
    *mut CUdeviceptr,
    CUexternalMemory,
    *const CUDA_EXTERNAL_MEMORY_BUFFER_DESC,
) -> CUresult;
pub type CuDestroyExternalMemory = unsafe extern "C" fn(CUexternalMemory) -> CUresult;
