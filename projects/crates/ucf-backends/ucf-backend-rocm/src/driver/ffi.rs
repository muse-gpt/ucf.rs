use std::ffi::c_void;

pub type HipError = i32;
pub type HipDeviceptr = *mut c_void;
pub type HipModule = *mut c_void;
pub type HipFunction = *mut c_void;
pub type HipStream = *mut c_void;

pub const HIP_SUCCESS: HipError = 0;

pub const HIP_MEMCPY_HOST_TO_DEVICE: i32 = 1;
pub const HIP_MEMCPY_DEVICE_TO_HOST: i32 = 2;
pub const HIP_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

#[derive(Debug)]
pub struct DriverError {
    pub code: i32,
    pub message: String,
}

impl DriverError {
    pub fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for DriverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HIP error {}: {}", self.code, self.message)
    }
}

impl std::error::Error for DriverError {}

pub type HipGetDeviceCount = unsafe extern "C" fn(*mut i32) -> HipError;
pub type HipSetDevice = unsafe extern "C" fn(i32) -> HipError;
pub type HipMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> HipError;
pub type HipFree = unsafe extern "C" fn(*mut c_void) -> HipError;
pub type HipMemcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32) -> HipError;
pub type HipDeviceSynchronize = unsafe extern "C" fn() -> HipError;
pub type HipModuleLoadData = unsafe extern "C" fn(*mut HipModule, *const c_void) -> HipError;
pub type HipModuleUnload = unsafe extern "C" fn(HipModule) -> HipError;
pub type HipModuleGetFunction =
    unsafe extern "C" fn(*mut HipFunction, HipModule, *const i8) -> HipError;
pub type HipModuleLaunchKernel = unsafe extern "C" fn(
    HipFunction,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    HipStream,
    *mut *mut c_void,
    *mut c_void,
) -> HipError;
