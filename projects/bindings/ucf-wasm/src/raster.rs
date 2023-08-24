//! UCF `TaskKind::Raster` 清屏（CPU 参考 + Web L2 宿主委托）。

use std::collections::BTreeMap;

use ucf::prelude::*;
use ucf::ParamValue;
use ucf::SchedulerErrorCode as ErrorCode;

use crate::contract::error_code_to_i32;
use crate::host::l2_raster_clear_webgpu_via_host;

/// 与 DX12 `raster_clear` 同参的 Raster 清屏图。
pub fn raster_clear_graph(width: u32, height: u32, rgba: [f32; 4]) -> Graph {
    let bytes = (width as u64) * (height as u64) * 4;
    Graph {
        resources: ResourceGraph {
            nodes: vec![ResourceNode {
                id: ResourceId(1),
                kind: ResourceKind::Buffer,
                domain: Domain::Host,
                access: Access::ReadWrite,
                byte_size: Some(bytes),
            }],
        },
        tasks: TaskGraph {
            nodes: vec![TaskNode {
                id: TaskId(10),
                kind: TaskKind::Raster,
                shader: ShaderId(10),
                params: BTreeMap::from([
                    ("dst".into(), ParamValue::I64(1)),
                    ("width".into(), ParamValue::I64(i64::from(width))),
                    ("height".into(), ParamValue::I64(i64::from(height))),
                    ("r".into(), ParamValue::F64(f64::from(rgba[0]))),
                    ("g".into(), ParamValue::F64(f64::from(rgba[1]))),
                    ("b".into(), ParamValue::F64(f64::from(rgba[2]))),
                    ("a".into(), ParamValue::F64(f64::from(rgba[3]))),
                ]),
                dispatch: Default::default(),
                objective: Objective::MaxThroughput,
                priority: Priority::Batch,
            }],
            edges: vec![DepEdge {
                from_task: None,
                from_resource: Some(ResourceId(1)),
                to_task: TaskId(10),
                kind: DepKind::Data,
            }],
        },
    }
}

fn rgba_bytes_from_linear(rgba: [f32; 4]) -> [u8; 4] {
    [
        (rgba[0].clamp(0.0, 1.0) * 255.0).round() as u8,
        (rgba[1].clamp(0.0, 1.0) * 255.0).round() as u8,
        (rgba[2].clamp(0.0, 1.0) * 255.0).round() as u8,
        (rgba[3].clamp(0.0, 1.0) * 255.0).round() as u8,
    ]
}

/// CPU 跑 Raster 清屏并读回 RGBA8。
pub fn run_raster_clear_rgba8(
    width: u32,
    height: u32,
    rgba: [f32; 4],
) -> Result<Vec<u8>, ErrorCode> {
    if width == 0 || height == 0 {
        return Ok(Vec::new());
    }
    let graph = raster_clear_graph(width, height, rgba);
    let mut session = CpuSession::open();
    session.prepare(&graph).map_err(|e| e.code())?;
    session.run_prepared(&graph).map_err(|e| e.code())?;
    session.flush().map_err(|e| e.code())?;
    let need = (width as usize)
        .checked_mul(height as usize)
        .and_then(|p| p.checked_mul(4))
        .ok_or(ErrorCode::Backend)?;
    let store = session.store();
    let guard = store.lock().map_err(|_| ErrorCode::Backend)?;
    let bytes = guard.bytes(ResourceId(1)).map_err(|_| ErrorCode::Backend)?;
    if bytes.len() < need {
        return Err(ErrorCode::Backend);
    }
    Ok(bytes[..need].to_vec())
}

/// L2：CPU Raster 校验后 WebGPU 清屏。
pub fn run_l2_raster_clear_webgpu(
    width: u32,
    height: u32,
    rgba: [f32; 4],
) -> Result<(), ErrorCode> {
    let read = run_raster_clear_rgba8(width, height, rgba)?;
    let expect = rgba_bytes_from_linear(rgba);
    if read.len() >= 4 && read[0..4] != expect {
        return Err(ErrorCode::Backend);
    }
    let px = rgba_bytes_from_linear(rgba);
    let code = l2_raster_clear_webgpu_via_host(width, height, px);
    if code != 0 {
        return Err(ErrorCode::Backend);
    }
    Ok(())
}

pub fn l2_raster_clear_webgpu_status(
    width: u32,
    height: u32,
    r: u8,
    g: u8,
    b: u8,
    a: u8,
) -> i32 {
    let rgba = [
        f32::from(r) / 255.0,
        f32::from(g) / 255.0,
        f32::from(b) / 255.0,
        f32::from(a) / 255.0,
    ];
    match run_l2_raster_clear_webgpu(width, height, rgba) {
        Ok(()) => 0,
        Err(code) => error_code_to_i32(code),
    }
}

/// Write raster readback into `out`.
///
/// # Safety
///
/// `out` must address at least `width * height * 4` bytes.
pub unsafe fn write_raster_clear_rgba8(
    width: u32,
    height: u32,
    r: u8,
    g: u8,
    b: u8,
    a: u8,
    out: *mut u8,
) -> i32 {
    if out.is_null() {
        return -10;
    }
    let rgba = [
        f32::from(r) / 255.0,
        f32::from(g) / 255.0,
        f32::from(b) / 255.0,
        f32::from(a) / 255.0,
    ];
    let need = (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(4);
    match run_raster_clear_rgba8(width, height, rgba) {
        Ok(bytes) => {
            if bytes.len() != need {
                return -11;
            }
            unsafe {
                core::ptr::copy_nonoverlapping(bytes.as_ptr(), out, bytes.len());
            }
            0
        }
        Err(code) => error_code_to_i32(code),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raster_clear_4x2_red() {
        let got = run_raster_clear_rgba8(4, 2, [1.0, 0.0, 0.0, 1.0]).expect("raster");
        assert_eq!(got.len(), 32);
        assert_eq!(&got[0..4], &[255, 0, 0, 255]);
    }
}
