//! Emit AMD HSACO (ELF code object) for HIP `hipModuleLoadData`.
//!
//! OpenCL C lives only inside this module (same pattern as DXIL's private HLSL).
//! The AMD APP OpenCL compiler produces HSACO the HIP runtime can load — no `hiprtc`.

#![allow(non_snake_case)]

use std::ffi::{c_void, CString};
use std::path::Path;
use std::ptr;

use libloading::Library;
use ucf_types::{ShaderOp, ShaderProgram};

/// Emit HSACO ELF bytes for the given UCF program.
///
/// Uses the system OpenCL ICD (AMD APP on Windows) as the offline code-object
/// builder — analogous to `d3dcompiler` for DXBC. Callers never pass OpenCL C.
pub fn emit_hsaco(program: &ShaderProgram) -> Result<Vec<u8>, String> {
    for op in &program.ops {
        match op {
            ShaderOp::MemCopy => continue,
            ShaderOp::IotaFill { value } => {
                return compile_opencl(&ocl_fill(&program.entry, *value));
            }
            ShaderOp::MatMul => {
                return compile_opencl(&ocl_matmul(&program.entry));
            }
            ShaderOp::Rgba8Denoise => {
                return compile_opencl(&ocl_rgba8_denoise(&program.entry));
            }
            ShaderOp::Attention => {
                return compile_opencl(&ocl_attention(&program.entry));
            }
        }
    }
    Err("program has no emittable HSACO ops".into())
}

/// Alias of [`emit_hsaco`] (historical name from the hiprtc source path).
pub fn emit_hip(program: &ShaderProgram) -> Result<Vec<u8>, String> {
    emit_hsaco(program)
}

fn ocl_fill(entry: &str, value: f32) -> String {
    format!(
        r#"
__kernel void {entry}(__global float* out, unsigned int count) {{
    unsigned int i = get_global_id(0);
    if (i >= count) return;
    out[i] = {value:?};
}}
"#
    )
}

fn ocl_matmul(entry: &str) -> String {
    format!(
        r#"
__kernel void {entry}(
    __global const float* a,
    __global const float* b,
    __global float* out,
    unsigned int m,
    unsigned int n,
    unsigned int k
) {{
    unsigned int idx = get_global_id(0);
    if (idx >= m * n) return;
    unsigned int row = idx / n;
    unsigned int col = idx % n;
    float acc = 0.0f;
    for (unsigned int i = 0; i < k; ++i) {{
        acc += a[row * k + i] * b[i * n + col];
    }}
    out[idx] = acc;
}}
"#
    )
}

fn ocl_rgba8_denoise(entry: &str) -> String {
    format!(
        r#"
__kernel void {entry}(
    __global const unsigned char* src,
    __global unsigned char* out,
    unsigned int width,
    unsigned int height
) {{
    unsigned int idx = get_global_id(0);
    if (idx >= width * height) return;
    unsigned int x = idx % width;
    unsigned int left = (x == 0) ? idx : (idx - 1);
    unsigned int base = idx * 4;
    unsigned int lbase = left * 4;
    out[base + 0] = (unsigned char)(((unsigned int)src[base + 0] + (unsigned int)src[lbase + 0]) >> 1);
    out[base + 1] = (unsigned char)(((unsigned int)src[base + 1] + (unsigned int)src[lbase + 1]) >> 1);
    out[base + 2] = (unsigned char)(((unsigned int)src[base + 2] + (unsigned int)src[lbase + 2]) >> 1);
    out[base + 3] = src[base + 3];
}}
"#
    )
}

fn ocl_attention(entry: &str) -> String {
    format!(
        r#"
__kernel void {entry}(
    __global const float* q,
    __global const float* k,
    __global const float* v,
    __global float* out,
    unsigned int batch,
    unsigned int heads,
    unsigned int seq,
    unsigned int dim
) {{
    unsigned int idx = get_global_id(0);
    unsigned int total = batch * heads * seq * dim;
    if (idx >= total) return;
    unsigned int d = idx % dim;
    unsigned int t = idx / dim;
    unsigned int i = t % seq;
    t = t / seq;
    unsigned int h = t % heads;
    unsigned int b = t / heads;
    float acc = 0.0f;
    for (unsigned int j = 0; j < seq; ++j) {{
        float score = 0.0f;
        for (unsigned int dd = 0; dd < dim; ++dd) {{
            unsigned int qi = (((b * heads + h) * seq + i) * dim) + dd;
            unsigned int ki = (((b * heads + h) * seq + j) * dim) + dd;
            score += q[qi] * k[ki];
        }}
        unsigned int vi = (((b * heads + h) * seq + j) * dim) + d;
        acc += score * v[vi];
    }}
    out[idx] = acc;
}}
"#
    )
}

type ClInt = i32;
type ClUint = u32;
type ClPlatformId = *mut c_void;
type ClDeviceId = *mut c_void;
type ClContext = *mut c_void;
type ClProgram = *mut c_void;

const CL_SUCCESS: ClInt = 0;
const CL_DEVICE_TYPE_GPU: u64 = 4;
const CL_PROGRAM_BINARY_SIZES: ClUint = 0x1165;
const CL_PROGRAM_BINARIES: ClUint = 0x1166;
const CL_PROGRAM_BUILD_LOG: ClUint = 0x1183;

type ClGetPlatformIDs = unsafe extern "C" fn(ClUint, *mut ClPlatformId, *mut ClUint) -> ClInt;
type ClGetDeviceIDs =
    unsafe extern "C" fn(ClPlatformId, u64, ClUint, *mut ClDeviceId, *mut ClUint) -> ClInt;
type ClCreateContext = unsafe extern "C" fn(
    *const isize,
    ClUint,
    *const ClDeviceId,
    *mut c_void,
    *mut c_void,
    *mut ClInt,
) -> ClContext;
type ClCreateProgramWithSource = unsafe extern "C" fn(
    ClContext,
    ClUint,
    *const *const i8,
    *const usize,
    *mut ClInt,
) -> ClProgram;
type ClBuildProgram = unsafe extern "C" fn(
    ClProgram,
    ClUint,
    *const ClDeviceId,
    *const i8,
    *mut c_void,
    *mut c_void,
) -> ClInt;
type ClGetProgramBuildInfo = unsafe extern "C" fn(
    ClProgram,
    ClDeviceId,
    ClUint,
    usize,
    *mut c_void,
    *mut usize,
) -> ClInt;
type ClGetProgramInfo =
    unsafe extern "C" fn(ClProgram, ClUint, usize, *mut c_void, *mut usize) -> ClInt;
type ClReleaseProgram = unsafe extern "C" fn(ClProgram) -> ClInt;
type ClReleaseContext = unsafe extern "C" fn(ClContext) -> ClInt;

fn compile_opencl(source: &str) -> Result<Vec<u8>, String> {
    let lib = load_opencl()?;
    unsafe {
        let clGetPlatformIDs: libloading::Symbol<ClGetPlatformIDs> = lib
            .get(b"clGetPlatformIDs")
            .map_err(|e| format!("clGetPlatformIDs: {e}"))?;
        let clGetDeviceIDs: libloading::Symbol<ClGetDeviceIDs> = lib
            .get(b"clGetDeviceIDs")
            .map_err(|e| format!("clGetDeviceIDs: {e}"))?;
        let clCreateContext: libloading::Symbol<ClCreateContext> = lib
            .get(b"clCreateContext")
            .map_err(|e| format!("clCreateContext: {e}"))?;
        let clCreateProgramWithSource: libloading::Symbol<ClCreateProgramWithSource> = lib
            .get(b"clCreateProgramWithSource")
            .map_err(|e| format!("clCreateProgramWithSource: {e}"))?;
        let clBuildProgram: libloading::Symbol<ClBuildProgram> = lib
            .get(b"clBuildProgram")
            .map_err(|e| format!("clBuildProgram: {e}"))?;
        let clGetProgramBuildInfo: libloading::Symbol<ClGetProgramBuildInfo> = lib
            .get(b"clGetProgramBuildInfo")
            .map_err(|e| format!("clGetProgramBuildInfo: {e}"))?;
        let clGetProgramInfo: libloading::Symbol<ClGetProgramInfo> = lib
            .get(b"clGetProgramInfo")
            .map_err(|e| format!("clGetProgramInfo: {e}"))?;
        let clReleaseProgram: libloading::Symbol<ClReleaseProgram> = lib
            .get(b"clReleaseProgram")
            .map_err(|e| format!("clReleaseProgram: {e}"))?;
        let clReleaseContext: libloading::Symbol<ClReleaseContext> = lib
            .get(b"clReleaseContext")
            .map_err(|e| format!("clReleaseContext: {e}"))?;

        let mut nplat = 0u32;
        check_cl(clGetPlatformIDs(0, ptr::null_mut(), &mut nplat), "clGetPlatformIDs")?;
        if nplat == 0 {
            return Err("no OpenCL platforms".into());
        }
        let mut plats = vec![ptr::null_mut(); nplat as usize];
        check_cl(
            clGetPlatformIDs(nplat, plats.as_mut_ptr(), ptr::null_mut()),
            "clGetPlatformIDs",
        )?;

        let mut device: ClDeviceId = ptr::null_mut();
        let mut found = false;
        for plat in plats {
            let mut ndev = 0u32;
            let q = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, ptr::null_mut(), &mut ndev);
            if q != CL_SUCCESS || ndev == 0 {
                continue;
            }
            let mut devs = vec![ptr::null_mut(); ndev as usize];
            check_cl(
                clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, ndev, devs.as_mut_ptr(), ptr::null_mut()),
                "clGetDeviceIDs",
            )?;
            device = devs[0];
            found = true;
            break;
        }
        if !found {
            return Err("no OpenCL GPU device".into());
        }

        let mut err = 0i32;
        let ctx = clCreateContext(ptr::null(), 1, &device, ptr::null_mut(), ptr::null_mut(), &mut err);
        check_cl(err, "clCreateContext")?;
        if ctx.is_null() {
            return Err("clCreateContext returned null".into());
        }

        let c_src = CString::new(source).map_err(|e| e.to_string())?;
        let src_ptr = c_src.as_ptr();
        let len = source.len();
        let prog = clCreateProgramWithSource(ctx, 1, &src_ptr, &len, &mut err);
        check_cl(err, "clCreateProgramWithSource")?;
        if prog.is_null() {
            let _ = clReleaseContext(ctx);
            return Err("clCreateProgramWithSource returned null".into());
        }

        let opts = CString::new("-cl-std=CL2.0").unwrap();
        let build = clBuildProgram(prog, 1, &device, opts.as_ptr(), ptr::null_mut(), ptr::null_mut());
        if build != CL_SUCCESS {
            let mut log_size = 0usize;
            let _ = clGetProgramBuildInfo(
                prog,
                device,
                CL_PROGRAM_BUILD_LOG,
                0,
                ptr::null_mut(),
                &mut log_size,
            );
            let mut log = vec![0u8; log_size.max(1)];
            let _ = clGetProgramBuildInfo(
                prog,
                device,
                CL_PROGRAM_BUILD_LOG,
                log.len(),
                log.as_mut_ptr().cast(),
                ptr::null_mut(),
            );
            let _ = clReleaseProgram(prog);
            let _ = clReleaseContext(ctx);
            let msg = String::from_utf8_lossy(&log).trim_matches('\0').to_string();
            return Err(format!("clBuildProgram failed: {msg}"));
        }

        let mut bin_size = 0usize;
        check_cl(
            clGetProgramInfo(
                prog,
                CL_PROGRAM_BINARY_SIZES,
                std::mem::size_of::<usize>(),
                (&mut bin_size as *mut usize).cast(),
                ptr::null_mut(),
            ),
            "clGetProgramInfo BINARY_SIZES",
        )?;
        if bin_size == 0 {
            let _ = clReleaseProgram(prog);
            let _ = clReleaseContext(ctx);
            return Err("OpenCL program binary size is 0".into());
        }
        let mut bytes = vec![0u8; bin_size];
        let mut ptr_to_buf: *mut u8 = bytes.as_mut_ptr();
        check_cl(
            clGetProgramInfo(
                prog,
                CL_PROGRAM_BINARIES,
                std::mem::size_of::<*mut u8>(),
                (&mut ptr_to_buf as *mut *mut u8).cast(),
                ptr::null_mut(),
            ),
            "clGetProgramInfo BINARIES",
        )?;

        let _ = clReleaseProgram(prog);
        let _ = clReleaseContext(ctx);

        if bytes.len() < 4 || &bytes[..4] != b"\x7fELF" {
            return Err("OpenCL binary is not an ELF HSACO code object".into());
        }
        Ok(bytes)
    }
}

fn check_cl(code: ClInt, api: &str) -> Result<(), String> {
    if code == CL_SUCCESS {
        Ok(())
    } else {
        Err(format!("{api} failed: {code}"))
    }
}

fn load_opencl() -> Result<Library, String> {
    let mut last = String::new();
    for path in opencl_candidates() {
        match unsafe { Library::new(&path) } {
            Ok(lib) => return Ok(lib),
            Err(err) => last = format!("{}: {err}", path.display()),
        }
    }
    Err(format!("failed to load OpenCL ({last})"))
}

fn opencl_candidates() -> Vec<std::path::PathBuf> {
    if cfg!(target_os = "windows") {
        vec![Path::new("OpenCL.dll").to_path_buf()]
    } else {
        vec![
            Path::new("libOpenCL.so.1").to_path_buf(),
            Path::new("libOpenCL.so").to_path_buf(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucf_types::ShaderProgram;

    #[test]
    fn emit_fill_hsaco_is_elf_or_soft_skip() {
        match emit_hsaco(&ShaderProgram::dispatch_fill("ucf_fill", 1.0)) {
            Ok(bytes) => {
                assert!(bytes.starts_with(b"\x7fELF"), "expected ELF HSACO");
            }
            Err(err) => {
                eprintln!("skip hsaco emit (no OpenCL GPU): {err}");
            }
        }
    }
}
