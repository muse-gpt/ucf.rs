//! L2 薄门：UCF CPU `Fill` 校验后，经 `ucf_host` 提交 WebGPU 清屏（无 `wgpu` crate）。

use hardware::SchedulerErrorCode as ErrorCode;

use crate::contract::error_code_to_i32;
use crate::fill::run_color_fill_rgba8;
use crate::host::l2_color_fill_webgpu_via_host;

/// 先跑 CPU Fill 图，再委托宿主 WebGPU 清屏。
pub fn run_l2_color_fill_webgpu(
    width: u32,
    height: u32,
    rgba: [u8; 4],
) -> Result<(), ErrorCode> {
    run_color_fill_rgba8(width, height, rgba)?;
    let code = l2_color_fill_webgpu_via_host(width, height, rgba);
    if code != 0 {
        return Err(ErrorCode::Backend);
    }
    Ok(())
}

/// C ABI：`0` 成功，负值为稳定错误码。
pub fn l2_color_fill_webgpu_status(
    width: u32,
    height: u32,
    rgba: [u8; 4],
) -> i32 {
    match run_l2_color_fill_webgpu(width, height, rgba) {
        Ok(()) => 0,
        Err(code) => error_code_to_i32(code),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_cpu_fill_path_ok_without_host() {
        // 原生测试无 wasm host import，WebGPU 步会失败。
        let err = run_l2_color_fill_webgpu(2, 2, [1, 2, 3, 255]);
        assert_eq!(err, Err(ErrorCode::Backend));
    }
}
