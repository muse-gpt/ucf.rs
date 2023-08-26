//! CPU 软件光栅：与 `ucf_emitter` NDC 黄三角薄门对齐。

/// 与 DX12 / SPIR-V 薄门相同的 NDC 顶点（`y` 向上）。
pub const NDC_TRI_VERTS: [(f32, f32); 3] = [(-0.8, -0.8), (0.8, -0.8), (0.0, 0.8)];

/// 三角形填充色（opaque yellow）。
pub const TRI_RGBA: [u8; 4] = [255, 255, 0, 255];

/// 像素中心 → NDC（D3D 风格：`y` 向下增大）。
#[inline]
pub fn pixel_center_to_ndc(x: usize, y: usize, width: usize, height: usize) -> (f32, f32) {
    let ndc_x = (x as f32 + 0.5) / width as f32 * 2.0 - 1.0;
    let ndc_y = 1.0 - (y as f32 + 0.5) / height as f32 * 2.0;
    (ndc_x, ndc_y)
}

#[inline]
fn edge(a: (f32, f32), b: (f32, f32), p: (f32, f32)) -> f32 {
    (p.0 - a.0) * (b.1 - a.1) - (p.1 - a.1) * (b.0 - a.0)
}

/// 点 `p` 是否在 NDC 三角形内（含边）。
pub fn point_in_ndc_tri(p: (f32, f32), verts: [(f32, f32); 3]) -> bool {
    let e0 = edge(verts[0], verts[1], p);
    let e1 = edge(verts[1], verts[2], p);
    let e2 = edge(verts[2], verts[0], p);
    let has_neg = e0 < 0.0 || e1 < 0.0 || e2 < 0.0;
    let has_pos = e0 > 0.0 || e1 > 0.0 || e2 > 0.0;
    !(has_neg && has_pos)
}

/// 清屏 + NDC 黄三角 → 紧凑 RGBA8。
pub fn raster_clear_tri_rgba8(
    width: usize,
    height: usize,
    clear: [u8; 4],
    tri: [u8; 4],
) -> Result<Vec<u8>, &'static str> {
    if width == 0 || height == 0 {
        return Ok(Vec::new());
    }
    let need = width
        .checked_mul(height)
        .and_then(|p| p.checked_mul(4))
        .ok_or("raster size overflow")?;
    let mut out = vec![0u8; need];
    for y in 0..height {
        for x in 0..width {
            let ndc = pixel_center_to_ndc(x, y, width, height);
            let rgba = if point_in_ndc_tri(ndc, NDC_TRI_VERTS) {
                tri
            } else {
                clear
            };
            let i = (y * width + x) * 4;
            out[i..i + 4].copy_from_slice(&rgba);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tri_covers_center_not_corner() {
        let px = raster_clear_tri_rgba8(32, 32, [0, 0, 0, 255], TRI_RGBA).expect("raster");
        let w = 32usize;
        let center = &px[((16 * w + 16) * 4)..((16 * w + 16) * 4 + 4)];
        assert_eq!(center, &[255, 255, 0, 255]);
        assert_eq!(&px[0..4], &[0, 0, 0, 255]);
    }
}
