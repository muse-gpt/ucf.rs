//! L0 降级：UCF CPU Fill → RGBA8 readback（Web 上屏薄门）。

use ucf::prelude::*;
use ucf::ParamValue;
use ucf::SchedulerErrorCode as ErrorCode;

use crate::contract::error_code_to_i32;

/// 纯色 Fill 图并读回 RGBA8 字节（每像素重复 `rgba`）。
pub fn run_color_fill_rgba8(width: u32, height: u32, rgba: [u8; 4]) -> Result<Vec<u8>, ErrorCode> {
    let pixels = (width as usize).saturating_mul(height as usize);
    if pixels == 0 {
        return Ok(Vec::new());
    }
    let gray = f64::from(rgba[0]) / 255.0;
    let graph = fill_graph(pixels, gray);

    let mut session = CpuSession::open();
    session.prepare(&graph).map_err(|e| e.code())?;
    session.run_prepared(&graph).map_err(|e| e.code())?;
    session.flush().map_err(|e| e.code())?;
    let floats = session.read_f32(ResourceId(1)).map_err(|e| e.code())?;
    if floats.len() != pixels {
        return Err(ErrorCode::Backend);
    }
    let expect = rgba[0] as f32 / 255.0;
    if floats.iter().any(|v| (v - expect).abs() > 1e-5) {
        return Err(ErrorCode::Backend);
    }
    let mut out = Vec::with_capacity(pixels * 4);
    for _ in 0..pixels {
        out.extend_from_slice(&rgba);
    }
    Ok(out)
}

fn fill_graph(floats: usize, value: f64) -> Graph {
    use std::collections::BTreeMap;

    Graph {
        resources: ResourceGraph {
            nodes: vec![ResourceNode {
                id: ResourceId(1),
                kind: ResourceKind::Buffer,
                domain: Domain::Host,
                access: Access::ReadWrite,
                byte_size: Some((floats * 4) as u64),
            }],
        },
        tasks: TaskGraph {
            nodes: vec![TaskNode {
                id: TaskId(10),
                kind: TaskKind::Fill,
                shader: ShaderId(10),
                params: BTreeMap::from([
                    ("dst".into(), ParamValue::I64(1)),
                    ("value".into(), ParamValue::F64(value)),
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

/// Write RGBA8 into `out` via [`run_color_fill_rgba8`]. Returns `0` or negative error code.
///
/// # Safety
///
/// `out` must address at least `width * height * 4` bytes.
pub unsafe fn write_color_fill_rgba8(
    width: u32,
    height: u32,
    rgba: [u8; 4],
    out: *mut u8,
) -> i32 {
    if out.is_null() {
        return -10;
    }
    let need = (width as usize).saturating_mul(height as usize).saturating_mul(4);
    match run_color_fill_rgba8(width, height, rgba) {
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
    fn fill_2x2_rgba8() {
        let got = run_color_fill_rgba8(2, 2, [9, 8, 7, 255]).expect("fill");
        assert_eq!(got.len(), 16);
        assert_eq!(&got[0..4], &[9, 8, 7, 255]);
        assert_eq!(&got[12..16], &[9, 8, 7, 255]);
    }
}
