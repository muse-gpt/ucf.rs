//! UCF **Wasm** 绑定（`wasm32-unknown-unknown`）。
//!
//! 导出稳定 C ABI，供 Web / Node 宿主加载。仅 CPU 后端，**不**依赖 `wgpu` crate。
//!
//! ```text
//! rustup target add wasm32-unknown-unknown
//! cargo build -p ucf-wasm --target wasm32-unknown-unknown --release
//! ```

#![warn(missing_docs)]

mod contract;
mod fill;
mod host;
mod l2;

pub use contract::{
    error_code_to_i32, reference_graph, run_diag_mask, run_matmul_contract, EXPECTED_DIAG_MASK,
    EXPECTED_MATMUL,
};
pub use fill::{run_color_fill_rgba8, write_color_fill_rgba8};
pub use host::bind_external_buffer_via_host;
pub use l2::{l2_color_fill_webgpu_status, run_l2_color_fill_webgpu};

/// Wasm 目标三元组（与 Rust target 对齐）。
pub const TARGET_TRIPLE: &str = "wasm32-unknown-unknown";

/// L0：UCF CPU Fill → RGBA8 写入线性内存（`width * height * 4` 字节）。
///
/// # Safety
///
/// `out` 必须可写且长度足够。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ucf_color_fill_rgba8(
    width: u32,
    height: u32,
    r: u8,
    g: u8,
    b: u8,
    a: u8,
    out: *mut u8,
) -> i32 {
    write_color_fill_rgba8(width, height, [r, g, b, a], out)
}

/// L2：UCF CPU `Fill` 校验后，经 `ucf_host` 提交 WebGPU 清屏。
#[unsafe(no_mangle)]
pub extern "C" fn ucf_l2_color_fill_webgpu(
    width: u32,
    height: u32,
    r: u8,
    g: u8,
    b: u8,
    a: u8,
) -> i32 {
    l2_color_fill_webgpu_status(width, height, [r, g, b, a])
}

/// 经宿主 `ucf_host` import 绑定外部 buffer 句柄（Web GPU 表项）。
#[unsafe(no_mangle)]
pub extern "C" fn ucf_bind_external_buffer(resource_id: u32, handle_id: u32) -> i32 {
    bind_external_buffer_via_host(resource_id, handle_id)
}

/// 版本编码：`major * 1_000_000 + minor * 1_000 + patch`。
#[unsafe(no_mangle)]
pub extern "C" fn ucf_version_code() -> u32 {
    parse_version_code(env!("CARGO_PKG_VERSION"))
}

/// 只读目标三元组 C 字符串。
#[unsafe(no_mangle)]
pub extern "C" fn ucf_target_triple() -> *const u8 {
    TARGET_TRIPLE.as_ptr()
}

/// 本 wasm 产物是否包含 CPU 后端（恒为 `1`）。
#[unsafe(no_mangle)]
pub extern "C" fn ucf_has_cpu_backend() -> u32 {
    1
}

/// 跑 reference Copy→Fill→MatMul 图；成功时写 4 个 `f32` 并返回 `0`。
///
/// # Safety
///
/// `out` 必须指向至少 4 个可写 `f32`。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ucf_application_contract_matmul(out: *mut f32) -> i32 {
    if out.is_null() {
        return -10;
    }
    match run_matmul_contract() {
        Ok(values) => {
            if values.len() != 4 {
                return -11;
            }
            for (i, v) in values.iter().enumerate() {
                // SAFETY: out 已校验非空，i < 4。
                unsafe {
                    out.add(i).write(*v);
                }
            }
            0
        }
        Err(code) => error_code_to_i32(code),
    }
}

/// 诊断 kind 位掩码；失败返回负 `ErrorCode`。
///
/// # Safety
///
/// `out` 非空时写入掩码。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ucf_application_contract_diag_mask(out: *mut u32) -> i32 {
    if out.is_null() {
        return -10;
    }
    match run_diag_mask() {
        Ok(mask) => {
            unsafe {
                out.write(mask);
            }
            0
        }
        Err(code) => error_code_to_i32(code),
    }
}

fn parse_version_code(v: &str) -> u32 {
    let mut parts = v.split('.');
    let major: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let minor: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let patch: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    major.saturating_mul(1_000_000) + minor.saturating_mul(1_000) + patch
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_contract_matches_application_test() {
        let out = run_matmul_contract().expect("contract");
        assert_eq!(out, EXPECTED_MATMUL.to_vec());
        let mask = run_diag_mask().expect("diag");
        assert_eq!(mask, EXPECTED_DIAG_MASK);
    }
}
